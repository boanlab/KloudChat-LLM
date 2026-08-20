"""Node probing over SSH.

Independent steps, each tolerant of the previous one failing:

    1. docker ps                    running containers
    2. nvidia-smi / /proc/meminfo   GPU name, count, total VRAM
    3. uname -m                     architecture
    4. vLLM /metrics                realised KV blocks, for diagnosis

A node where every step fails still yields a NodeSpec with zero capacity and
``alive=False``, so the planner excludes it deterministically.
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Mapping, Optional

from scheduler.types import GB, NodeSpec

#: OS share on a unified-memory node, excluded from GPU capacity.
_UNIFIED_RESERVE_BYTES: int = 12 * GB

#: Containers this stack owns, for telling our GPU memory from anyone else's.
#: Mirrors applier.MANAGED_SERVICE_PREFIX.
MANAGED_PREFIX: str = "vllm-"
WHISPER_CONTAINER: str = "whisper"

#: Transcription backend port. amd64 only — the probe never answers on arm64.
WHISPER_PORT: int = 9000


@dataclass(frozen=True)
class RunningWorkload:
    """One vLLM instance on a node."""

    container_name: str
    num_gpu_blocks: Optional[int] = None
    block_size: Optional[int] = None
    realized_max_len: Optional[int] = None
    realized_gpu_util: Optional[float] = None


@dataclass(frozen=True)
class NodeProbe:
    spec: NodeSpec
    alive: bool
    running_workloads: tuple[RunningWorkload, ...] = field(default_factory=tuple)
    running_containers: frozenset[str] = frozenset()
    whisper_running: bool = False
    raw_errors: tuple[str, ...] = field(default_factory=tuple)


def _ssh(host: str, cmd: str, *, timeout: int = 6) -> tuple[int, str, str]:
    """One-shot SSH: (rc, stdout, stderr). Never raises."""
    try:
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no",
             "-o", f"ConnectTimeout={timeout}",
             "-o", "LogLevel=ERROR",
             host, cmd],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    except OSError as e:
        return 1, "", str(e)


def _parse_metrics_for_kv(metrics_text: str) -> tuple[Optional[int], Optional[int]]:
    """Extract (num_gpu_blocks, block_size) from a vLLM /metrics scrape."""
    for line in metrics_text.splitlines():
        if "cache_config_info" not in line or "{" not in line:
            continue
        blocks = size = None
        inner = line[line.index("{") + 1: line.rindex("}")] if "}" in line else ""
        for pair in inner.split(","):
            k, _, v = pair.partition("=")
            v = v.strip().strip('"')
            k = k.strip()
            if k == "num_gpu_blocks" and v.isdigit():
                blocks = int(v)
            elif k == "block_size" and v.isdigit():
                size = int(v)
        if blocks is not None:
            return blocks, size
    return None, None


def _probe_vllm_config(host: str, container: str) -> tuple[Optional[int], Optional[float]]:
    """A running container's (--max-model-len, --gpu-memory-utilization).

    Creation-time arguments, readable while the model still loads. Without them,
    a service already at the target configuration would be recreated for nothing.
    """
    rc, out, _ = _ssh(host, f"docker inspect {container} --format '{{{{json .Args}}}}'")
    if rc != 0 or not out:
        return None, None
    try:
        args = json.loads(out)
    except json.JSONDecodeError:
        return None, None
    max_len: Optional[int] = None
    gpu_util: Optional[float] = None
    for i, a in enumerate(args):
        nxt = args[i + 1] if i + 1 < len(args) else ""
        if a == "--max-model-len" and str(nxt).isdigit():
            max_len = int(nxt)
        elif a == "--gpu-memory-utilization":
            try:
                gpu_util = float(nxt)
            except (TypeError, ValueError):
                pass
    return max_len, gpu_util


def _probe_vllm(host: str, container: str, port: int) -> RunningWorkload:
    """One vLLM's /metrics. The container is recorded even on failure."""
    rml, rgu = _probe_vllm_config(host, container)
    rc, out, _ = _ssh(host, f"curl -fsS http://localhost:{port}/metrics")
    if rc != 0 or not out:
        return RunningWorkload(container, realized_max_len=rml, realized_gpu_util=rgu)
    blocks, bsz = _parse_metrics_for_kv(out)
    return RunningWorkload(container, blocks, bsz, rml, rgu)


def read_env(host: str, path: str) -> dict[str, str]:
    """A node's .env as a mapping. Unreachable or missing reads empty, which
    makes the caller treat every option as changed — the safe direction."""
    code, out, _ = _ssh(host, f"cat {path} 2>/dev/null || true")
    if code != 0:
        return {}
    values: dict[str, str] = {}
    for raw in out.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def _probe_running_containers(host: str) -> set[str]:
    rc, out, _ = _ssh(host, 'docker ps --format "{{.Names}}"')
    if rc != 0:
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def _probe_whisper(host: str) -> bool:
    """Transcription backend liveness. Drives capacity reservation, not placement."""
    rc, _, _ = _ssh(host, f"curl -fsS -o /dev/null http://localhost:{WHISPER_PORT}/health")
    return rc == 0


#: What an unrecognised NVIDIA card is called. lib.sh::detect_gpu_class says
#: this, and the two have to agree: gpu_class is compared against per-class
#: tables on both sides, and a Python-side value of "nvidia a100-sxm4-80gb"
#: matches no entry that a shell-side value of "nvidia-other" would.
UNKNOWN_GPU_CLASS: str = "nvidia-other"


def _classify_gpu_name(name: str) -> str:
    """Marketing name to class token, sharing lib.sh's vocabulary.

    Returning the raw marketing name for anything unrecognised was not sharing
    it: the docstring claimed a shared vocabulary while the two sides disagreed
    on every card outside the list.
    """
    name = (name or "").lower()
    if "gb10" in name:
        return "gb10"
    if "blackwell" in name and "6000" in name:
        return "pro6000"
    if "blackwell" in name and "5000" in name:
        return "pro5000"
    if "5090" in name:
        return "rtx5090"
    if "4090" in name:
        return "rtx4090"
    # An empty probe is not the same as a card we could not name.
    return UNKNOWN_GPU_CLASS if name.strip() else "unknown"


#: Reads GPU memory per compute process and labels each with the container it
#: belongs to, or "-" for one on the host. One `docker ps` for the whole set.
_VRAM_BY_OWNER = r"""
map=$(docker ps --no-trunc --format '{{.ID}} {{.Names}}' 2>/dev/null)
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
while IFS=, read -r pid mem; do
  pid=$(echo "$pid" | tr -d ' '); mem=$(echo "$mem" | tr -d ' ')
  [ -n "$pid" ] || continue
  cid=$(sed -n 's|.*docker-\([0-9a-f]*\)\.scope.*|\1|p' "/proc/$pid/cgroup" 2>/dev/null | head -1)
  name=$(echo "$map" | awk -v c="$cid" 'c != "" && $1 == c {print $2; exit}')
  echo "${name:--} ${mem}"
done
"""


def _probe_vram_by_owner(host: str, managed: frozenset) -> tuple[int, int]:
    """(foreign bytes, our bytes) of GPU memory currently held.

    vLLM's ``--gpu-memory-utilization`` is a fraction of the card's *total* but
    the memory has to be *free*, so a card with something else already on it —
    a desktop session, somebody's notebook, another stack — has less to give than
    its size suggests. Guessing that with a fixed reserve works until it does
    not; this measures it.

    Our own containers are counted separately and deliberately not treated as
    foreign: the planner decides a target state, and what our containers hold
    today is memory the plan is free to reassign.
    """
    code, out, _ = _ssh(host, _VRAM_BY_OWNER, timeout=10)
    if code != 0 or not out.strip():
        return 0, 0
    foreign = ours = 0
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        name, mib = parts[0], int(parts[1])
        if name in managed:
            ours += mib * 1024 * 1024
        else:
            foreign += mib * 1024 * 1024
    return foreign, ours


def _probe_gpu_class(host: str) -> str:
    rc, out, _ = _ssh(host, "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1")
    return _classify_gpu_name(out or "")


def _probe_arch(host: str) -> str:
    """Node architecture. Empty on probe failure, which excludes nothing."""
    rc, out, _ = _ssh(host, "uname -m")
    m = (out or "").strip().lower()
    if m in ("x86_64", "amd64"):
        return "amd64"
    if m in ("aarch64", "arm64"):
        return "arm64"
    return ""


def _probe_gpu_count(host: str) -> int:
    """GPU count. 1 on failure — the safe default."""
    rc, out, _ = _ssh(host, "nvidia-smi -L 2>/dev/null | grep -c '^GPU'")
    if rc == 0 and out.strip().isdigit():
        return max(1, int(out.strip()))
    return 1


def _probe_card_sizes(host: str) -> tuple[tuple[int, ...], bool]:
    """(bytes per card, is_unified).

    Every card, not just the first. A node used to be sized as "the first card,
    times how many there are", which is right only where they match — and a box
    with a 4090 beside a 5090 was then handed a capacity neither card has.

    A capacity from nvidia-smi is discrete VRAM. GB10 reports [N/A] and falls
    through to /proc/meminfo, and that fall-through is what unified memory is.
    """
    rc, out, _ = _ssh(host, "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null")
    if rc == 0:
        sizes = tuple(
            int(line.strip()) * 1024 * 1024            # MiB → B
            for line in out.splitlines() if line.strip().isdigit()
        )
        if sizes:
            return sizes, False
    rc, out, _ = _ssh(host, "awk '/^MemTotal:/ {print $2}' /proc/meminfo")
    if rc == 0 and out.strip().isdigit():
        return (int(out.strip()) * 1024,), True        # kB → B
    return (), False


def probe_node(
    node_id: str, host: str, *,
    reserved_bytes: Optional[int] = None,
    services: Optional[Mapping[str, int]] = None,
    retries: int = 1,
) -> NodeProbe:
    """Probe a node, retrying ``retries`` times while it looks dead.

    One transient SSH failure would otherwise drop that node's vLLM URLs from
    LiteLLM routing, taking a running model out of service.

    Args:
        services: compose service name to port, for identifying running vLLMs.
    """
    probe = _probe_node_once(node_id, host, reserved_bytes=reserved_bytes, services=services)
    attempts = max(0, retries)
    while not probe.alive and attempts > 0:
        attempts -= 1
        probe = _probe_node_once(node_id, host, reserved_bytes=reserved_bytes, services=services)
    return probe


def _probe_node_once(
    node_id: str, host: str, *,
    reserved_bytes: Optional[int] = None,
    services: Optional[Mapping[str, int]] = None,
) -> NodeProbe:
    errors: list[str] = []
    running = _probe_running_containers(host)
    if not running:
        # Nothing placed yet is indistinguishable here — not evidence of a down node
        errors.append("docker ps returned nothing")

    card_sizes, unified = _probe_card_sizes(host)
    gpu_class = _probe_gpu_class(host)
    gpu_count = _probe_gpu_count(host)
    if card_sizes and len(card_sizes) != gpu_count and not unified:
        # nvidia-smi answered for a different set than -L listed. Trust the sizes
        # it actually reported rather than multiplying one of them out.
        gpu_count = len(card_sizes)
    # The smallest card, because a model is placed on one card and every card
    # has to be able to hold what the planner promises. A mixed box is sized by
    # its weakest device, not by its average.
    total_vram = min(card_sizes) if card_sizes else 0
    arch = _probe_arch(host)
    whisper_up = _probe_whisper(host)
    alive = bool(running) or total_vram > 0

    workloads = tuple(
        _probe_vllm(host, name, port)
        for name, port in (services or {}).items()
        if name in running
    )

    # Unified memory shares system RAM — the full capacity would claim the OS share
    usable = max(0, total_vram - _UNIFIED_RESERVE_BYTES) if unified and total_vram else None
    if card_sizes and len(set(card_sizes)) > 1:
        errors.append(
            "mixed card sizes ("
            + ", ".join(f"{s / GB:.0f}G" for s in card_sizes)
            + f") — sized by the smallest, so {sum(card_sizes) / GB:.0f}G of "
            "capacity is not all usable"
        )

    # Anything on the card that is not ours. Our own containers are excluded on
    # purpose: the plan is free to reassign what they hold.
    managed = frozenset(
        c for c in running if c.startswith(MANAGED_PREFIX) or c == WHISPER_CONTAINER
    )
    foreign, _ours = _probe_vram_by_owner(host, managed)

    spec = NodeSpec(
        node_id=node_id,
        hostname=host,
        gpu_class=gpu_class,
        total_vram_bytes=total_vram,
        reserved_bytes=reserved_bytes,
        usable_vram_bytes=usable,
        gpu_count=gpu_count,
        foreign_vram_bytes=foreign,
        arch=arch,
    )
    return NodeProbe(
        spec=spec,
        alive=alive,
        running_workloads=workloads,
        running_containers=frozenset(running),
        whisper_running=whisper_up,
        raw_errors=tuple(errors),
    )


def probe_cluster(
    nodes: Mapping[str, str],
    *,
    max_workers: int = 8,
    **kw,
) -> list[NodeProbe]:
    """Probe every (node_id, host) pair, preserving input order.

    Parallel: a dead node's connection timeout must not queue ahead of live ones.
    """
    items = list(nodes.items())
    if not items:
        return []
    workers = min(max(1, max_workers), len(items))
    if workers == 1:
        return [probe_node(nid, host, **kw) for nid, host in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda it: probe_node(it[0], it[1], **kw), items))


def node_id_from_host(host: str) -> str:
    """Short identifier from an SSH target: last IPv4 octet, else first hostname label."""
    bare = host.split("@")[-1]
    parts = bare.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        return parts[-1]
    return parts[0]
