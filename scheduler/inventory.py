"""Node probing over SSH.

Independent steps, each tolerant of the others failing:

    1. docker ps                    running containers
    2. nvidia-smi / /proc/meminfo   GPU name, count, VRAM per card
    3. uname -m                     architecture
    4. nvidia-smi compute apps      GPU memory held outside this stack
    5. VLLM_MODELS_ROOT listing     checkpoints present

A node where every step fails yields ``alive=False`` and zero capacity.
"""

from __future__ import annotations

import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Mapping, Optional

from scheduler.types import GB, NodeSpec

#: OS share on a unified-memory node, excluded from GPU capacity
#: (lib.sh has the same figure)
_UNIFIED_RESERVE_BYTES: int = 12 * GB

#: Containers this stack owns (applier.MANAGED_SERVICE_PREFIX)
MANAGED_PREFIX: str = "vllm-"


@dataclass(frozen=True)
class RunningWorkload:
    """One vLLM instance on a node."""

    container_name: str


@dataclass(frozen=True)
class NodeProbe:
    spec: NodeSpec
    alive: bool
    running_workloads: tuple[RunningWorkload, ...] = field(default_factory=tuple)
    running_containers: frozenset[str] = frozenset()
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


def read_env(host: str, path: str) -> dict[str, str]:
    """A node's .env as a mapping; unreachable or missing reads empty."""
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


def _probe_checkpoints(host: str, models_root: str) -> Optional[frozenset[str]]:
    """Checkpoint directories (those holding a config.json), or None if the root cannot be read."""
    rc, out, _ = _ssh(
        host,
        f"for d in {shlex.quote(models_root)}/*/; do "
        '[ -f "$d/config.json" ] && basename "$d"; done',
    )
    if rc != 0:
        return None
    return frozenset(line.strip() for line in out.splitlines() if line.strip())


#: Class of an unrecognised NVIDIA card; must match lib.sh::detect_gpu_class
UNKNOWN_GPU_CLASS: str = "nvidia-other"


def _classify_gpu_name(name: str) -> str:
    """Marketing name to class token, in lib.sh::detect_gpu_class's vocabulary."""
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
    return UNKNOWN_GPU_CLASS if name.strip() else "unknown"


#: GPU memory per compute process, labelled with its container or "-" for the host
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
    """(foreign bytes, our bytes) of GPU memory held; our containers are memory the plan may reassign."""
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
    """"amd64" | "arm64" | "" on probe failure."""
    rc, out, _ = _ssh(host, "uname -m")
    m = (out or "").strip().lower()
    if m in ("x86_64", "amd64"):
        return "amd64"
    if m in ("aarch64", "arm64"):
        return "arm64"
    return ""


def _probe_gpu_count(host: str) -> int:
    """GPU count; 1 on failure."""
    rc, out, _ = _ssh(host, "nvidia-smi -L 2>/dev/null | grep -c '^GPU'")
    if rc == 0 and out.strip().isdigit():
        return max(1, int(out.strip()))
    return 1


def _probe_card_sizes(host: str) -> tuple[tuple[int, ...], bool]:
    """(bytes per card, is_unified). nvidia-smi for discrete VRAM; /proc/meminfo for unified memory (GB10)."""
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
    models_root: Optional[str] = None,
    retries: int = 1,
) -> NodeProbe:
    """Probe a node, retrying ``retries`` times while it looks dead.

    Args:
        services: compose service name to port, identifying running vLLMs.
        models_root: VLLM_MODELS_ROOT on the node; omitted, checkpoints are not reported.
    """
    once = lambda: _probe_node_once(  # noqa: E731
        node_id, host, reserved_bytes=reserved_bytes, services=services,
        models_root=models_root,
    )
    probe = once()
    attempts = max(0, retries)
    while not probe.alive and attempts > 0:
        attempts -= 1
        probe = once()
    return probe


def _probe_node_once(
    node_id: str, host: str, *,
    reserved_bytes: Optional[int] = None,
    services: Optional[Mapping[str, int]] = None,
    models_root: Optional[str] = None,
) -> NodeProbe:
    errors: list[str] = []
    running = _probe_running_containers(host)
    if not running:
        errors.append("docker ps returned nothing")

    card_sizes, unified = _probe_card_sizes(host)
    gpu_class = _probe_gpu_class(host)
    gpu_count = _probe_gpu_count(host)
    if card_sizes and len(card_sizes) != gpu_count and not unified:
        gpu_count = len(card_sizes)
    # Smallest card: every card must hold what the planner promises
    total_vram = min(card_sizes) if card_sizes else 0
    arch = _probe_arch(host)
    alive = bool(running) or total_vram > 0

    workloads = tuple(
        RunningWorkload(name) for name in (services or {}) if name in running
    )

    # Unified memory shares system RAM with the OS
    usable = max(0, total_vram - _UNIFIED_RESERVE_BYTES) if unified and total_vram else None
    if card_sizes and len(set(card_sizes)) > 1:
        errors.append(
            "mixed card sizes ("
            + ", ".join(f"{s / GB:.0f}G" for s in card_sizes)
            + f") — sized by the smallest, so {sum(card_sizes) / GB:.0f}G of "
            "capacity is not all usable"
        )

    managed = frozenset(c for c in running if c.startswith(MANAGED_PREFIX))
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
        checkpoints=(
            _probe_checkpoints(host, models_root) if models_root else None
        ),
    )
    return NodeProbe(
        spec=spec,
        alive=alive,
        running_workloads=workloads,
        running_containers=frozenset(running),
        raw_errors=tuple(errors),
    )


def probe_cluster(
    nodes: Mapping[str, str],
    *,
    max_workers: int = 8,
    **kw,
) -> list[NodeProbe]:
    """Probe every (node_id, host) pair in parallel, preserving input order."""
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
