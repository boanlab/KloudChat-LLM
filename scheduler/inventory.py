"""Node probing over SSH.

Independent steps, each tolerant of the others failing:

    1. docker ps                    running containers
    2. nvidia-smi / /proc/meminfo   GPU name, count, VRAM per card
    3. uname -m                     architecture
    4. nvidia-smi compute apps      GPU memory held outside this stack
    5. VLLM_MODELS_ROOT listing     checkpoints present

A node is ``alive`` only when both ``docker ps`` and its memory were read: a
half-answered node would be planned as empty or as full. SSH is retried while
the connection itself fails.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from scheduler.types import GB, NodeSpec

#: OS share on a unified-memory node, excluded from GPU capacity; must equal
#: lib.sh::UNIFIED_RESERVE_GB
_UNIFIED_RESERVE_BYTES: int = 12 * GB

#: Containers this stack owns whatever the selection; must equal applier.MANAGED_SERVICE_PREFIX
MANAGED_PREFIX: str = "vllm-"


@dataclass(frozen=True)
class NodeProbe:
    spec: NodeSpec
    alive: bool
    #: Catalogued services running on the node
    running_services: frozenset[str] = frozenset()
    running_containers: frozenset[str] = frozenset()
    raw_errors: tuple[str, ...] = ()


#: ssh's own exit status (connection refused or reset, handshake failure),
#: never the remote command's
SSH_FAILED: int = 255

#: Attempts per command while ssh itself fails; the probes are read-only
SSH_ATTEMPTS: int = 3


def _ssh(host: str, cmd: str, *, timeout: int = 6) -> tuple[int, str, str]:
    """SSH, retried while the connection itself fails: (rc, stdout, stderr). Never raises."""
    for attempt in range(SSH_ATTEMPTS):
        if attempt:
            time.sleep(attempt)
        try:
            r = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no",
                 "-o", f"ConnectTimeout={timeout}",
                 "-o", "LogLevel=ERROR",
                 host, cmd],
                capture_output=True, text=True, timeout=timeout + 5,
            )
        except subprocess.TimeoutExpired:
            return 124, "", f"timeout after {timeout}s"
        except OSError as e:
            return 1, "", str(e)
        if r.returncode != SSH_FAILED:
            break
    return r.returncode, r.stdout, r.stderr


def read_env(host: str, path: str) -> Optional[dict[str, str]]:
    """A node's .env as a mapping; a missing file reads empty, an unreachable node None."""
    code, out, _ = _ssh(host, f"cat {path} 2>/dev/null || true")
    if code != 0:
        return None
    values: dict[str, str] = {}
    for raw in out.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def _probe_running_containers(host: str) -> Optional[set[str]]:
    """Running container names; None when ``docker ps`` could not be run."""
    rc, out, _ = _ssh(host, 'docker ps --format "{{.Names}}"')
    if rc != 0:
        return None
    return {line.strip() for line in out.splitlines() if line.strip()}


def _probe_checkpoints(host: str, models_root: str) -> Optional[frozenset[str]]:
    """Checkpoint directories (a config.json, or a diffusers model_index.json), or None if the root cannot be read."""
    rc, out, _ = _ssh(
        host,
        f"for d in {shlex.quote(models_root)}/*/; do "
        '{ [ -f "$d/config.json" ] || [ -f "$d/model_index.json" ]; } && basename "$d"; done; true',
    )
    if rc != 0:
        return None
    return frozenset(line.strip() for line in out.splitlines() if line.strip())


#: Every card outside GB10, RTX 5090 and RTX PRO 5000/6000; the planner places
#: nothing there. Must match lib.sh::detect_gpu_class.
UNSUPPORTED_GPU_CLASS: str = "unsupported"


def _classify_gpu_name(name: str) -> str:
    """Class token in lib.sh::detect_gpu_class's vocabulary; "unknown" for an empty name."""
    name = (name or "").lower()
    if "gb10" in name:
        return "gb10"
    if "blackwell" in name and "6000" in name:
        return "pro6000"
    if "blackwell" in name and "5000" in name:
        return "pro5000"
    if "5090" in name:
        return "rtx5090"
    return UNSUPPORTED_GPU_CLASS if name.strip() else "unknown"


#: GPU memory per compute process: container name or "-" for the host, MiB, card ordinal
_VRAM_BY_OWNER = r"""
map=$(docker ps --no-trunc --format '{{.ID}} {{.Names}}' 2>/dev/null)
cards=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader,nounits 2>/dev/null |
while IFS=, read -r pid mem uuid; do
  pid=$(echo "$pid" | tr -d ' '); mem=$(echo "$mem" | tr -d ' '); uuid=$(echo "$uuid" | tr -d ' ')
  [ -n "$pid" ] || continue
  cid=$(sed -n 's|.*docker-\([0-9a-f]*\)\.scope.*|\1|p' "/proc/$pid/cgroup" 2>/dev/null | head -1)
  name=$(echo "$map" | awk -v c="$cid" 'c != "" && $1 == c {print $2; exit}')
  card=$(echo "$cards" | awk -F, -v u="$uuid" '$2 == u {print $1; exit}')
  echo "${name:--} ${mem} ${card:-0}"
done
"""


def _probe_foreign_vram(host: str, managed: frozenset) -> tuple[int, dict[int, int]]:
    """GPU memory held outside ``managed``: (total bytes, bytes per card ordinal)."""
    code, out, _ = _ssh(host, _VRAM_BY_OWNER, timeout=10)
    if code != 0 or not out.strip():
        return 0, {}
    foreign = 0
    by_card: dict[int, int] = {}
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        name, mib = parts[0], int(parts[1])
        card = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        if name not in managed:
            foreign += mib * 1024 * 1024
            by_card[card] = by_card.get(card, 0) + mib * 1024 * 1024
    return foreign, by_card


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
    # Unread, not unified: a discrete node taken for unified memory would be
    # sized by its system RAM
    if rc in (SSH_FAILED, 124):
        return (), False
    rc, out, _ = _ssh(host, "awk '/^MemTotal:/ {print $2}' /proc/meminfo")
    if rc == 0 and out.strip().isdigit():
        return (int(out.strip()) * 1024,), True        # kB → B
    return (), False


def probe_node(
    node_id: str, host: str, *,
    services: Iterable[str] = (),
    models_root: Optional[str] = None,
    retries: int = 1,
) -> NodeProbe:
    """Probe a node, retrying ``retries`` times while it looks dead.

    Args:
        services: catalogued compose service names.
        models_root: VLLM_MODELS_ROOT on the node; omitted, checkpoints are not reported.
    """
    once = lambda: _probe_node_once(  # noqa: E731
        node_id, host, services=services, models_root=models_root,
    )
    probe = once()
    attempts = max(0, retries)
    while not probe.alive and attempts > 0:
        attempts -= 1
        probe = once()
    return probe


def _probe_node_once(
    node_id: str, host: str, *,
    services: Iterable[str] = (),
    models_root: Optional[str] = None,
) -> NodeProbe:
    errors: list[str] = []
    services = frozenset(services)
    listed = _probe_running_containers(host)
    if listed is None:
        errors.append("docker ps failed")
    running = listed or set()

    card_sizes, unified = _probe_card_sizes(host)
    gpu_class = _probe_gpu_class(host)
    # Only GB10 shares system memory; any other card read as unified was misread
    if unified and gpu_class not in ("gb10", "unknown"):
        errors.append(f"{gpu_class} reported no card memory")
        card_sizes, unified = (), False
    gpu_count = _probe_gpu_count(host)
    if card_sizes and len(card_sizes) != gpu_count and not unified:
        gpu_count = len(card_sizes)
    # Sized by the smallest card
    total_vram = min(card_sizes) if card_sizes else 0
    arch = _probe_arch(host)
    # Both answers or neither: a node whose containers or memory went unread
    # would be planned as empty or as full, and its services restarted or stopped
    alive = listed is not None and total_vram > 0
    if listed is not None and not total_vram:
        errors.append("GPU memory could not be read")

    # Unified memory: system RAM less the OS share
    usable = max(0, total_vram - _UNIFIED_RESERVE_BYTES) if unified and total_vram else None
    if card_sizes and len(set(card_sizes)) > 1:
        errors.append(
            "mixed card sizes ("
            + ", ".join(f"{s / GB:.0f}G" for s in card_sizes)
            + f") — sized by the smallest, so {sum(card_sizes) / GB:.0f}G of "
            "capacity is not all usable"
        )

    # vllm-* always; a catalogued service of another prefix (omni-*) only while
    # its model is selected, so an unselected hand-run container stays foreign
    managed = frozenset(c for c in running if c.startswith(MANAGED_PREFIX) or c in services)
    foreign, by_card = _probe_foreign_vram(host, managed)
    foreign_by_card = tuple(by_card.get(i, 0) for i in range(max(1, gpu_count)))

    spec = NodeSpec(
        node_id=node_id,
        hostname=host,
        gpu_class=gpu_class,
        total_vram_bytes=total_vram,
        usable_vram_bytes=usable,
        gpu_count=gpu_count,
        foreign_vram_bytes=foreign,
        foreign_vram_by_card=foreign_by_card,
        arch=arch,
        checkpoints=(
            _probe_checkpoints(host, models_root) if models_root else None
        ),
    )
    return NodeProbe(
        spec=spec,
        alive=alive,
        running_services=frozenset(s for s in services if s in running),
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
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda it: probe_node(it[0], it[1], **kw), items))


def node_id_from_host(host: str) -> str:
    """Short identifier from an SSH target: last IPv4 octet, else first hostname label."""
    bare = host.split("@")[-1]
    parts = bare.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        return parts[-1]
    return parts[0]


def node_ids_for_hosts(hosts: Sequence[str]) -> dict[str, str]:
    """Node id to SSH target, in input order.

    A short id two targets share falls back to the bare host (no ``user@``).

    Raises:
        ValueError: a target listed twice.
    """
    short = [node_id_from_host(h) for h in hosts]
    out: dict[str, str] = {}
    for host, sid in zip(hosts, short, strict=True):
        nid = sid if short.count(sid) == 1 else host.split("@")[-1]
        if nid in out:
            raise ValueError(f"VLLM_NODES lists {host} twice")
        out[nid] = host
    return out
