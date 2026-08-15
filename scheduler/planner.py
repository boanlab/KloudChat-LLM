"""Model placement across GPU nodes.

Two phases:

1. Coverage — one instance of each model at its context floor, largest first,
   onto the node with the most capacity left. Seating at target context first
   would let a large model claim a node and starve the next one.
2. Restoration — leftover capacity raises contexts toward their targets.

Replication happens only when ``replicas`` is given, after full coverage.
Unplaced models are delegated to OpenRouter with a reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from scheduler.kv_model import ADMISSION_MARGIN, kv_bytes_per_token
from scheduler.registry import ModelSpec
from scheduler.types import GB, Dtype, NodeSpec

#: --gpu-memory-utilization bounds. 1.0 swallows the driver and context share,
#: failing engine init.
MAX_GPU_UTIL = 0.95
MIN_GPU_UTIL = 0.05

#: Activation, cudagraph and non-torch buffers. Deliberately high: under-counting
#: OOMs at engine init, over-counting only costs context.
ACTIVATION_BYTES = 4 * GB


@dataclass(frozen=True)
class Placement:
    model_id: str
    node_id: str
    ctx: int
    gpu_util: float
    #: Bytes this placement occupies on the node
    charge: int


@dataclass(frozen=True)
class Delegation:
    model_id: str
    reason: str


@dataclass
class Plan:
    placements: list[Placement] = field(default_factory=list)
    delegations: list[Delegation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def for_node(self, node_id: str) -> list[Placement]:
        return [p for p in self.placements if p.node_id == node_id]


def kv_bytes(spec: ModelSpec, ctx: int) -> int:
    """KV bytes to hold context ``ctx`` for every concurrent session.

    Zero for a pooling runner: an embedding model keeps nothing between tokens,
    so the ``2 · L · H · d · ctx`` reservation is memory it never touches.
    """
    if spec.metadata is None:
        raise ValueError(f"{spec.id}: no metadata — bind() must run first")
    if spec.is_pooling:
        return 0
    per_token = kv_bytes_per_token(spec.metadata, Dtype.FP8)
    sessions = max(1, spec.concurrent_sessions)
    return int(per_token * ctx * sessions * ADMISSION_MARGIN)


def need_bytes(spec: ModelSpec, ctx: int) -> int:
    """weights + activation + KV: what vLLM actually has to claim on the node."""
    return spec.weight_bytes + ACTIVATION_BYTES + kv_bytes(spec, ctx)


def gpu_util_for(charge: int, node: NodeSpec) -> float:
    """Fraction of node capacity to request, in vLLM's --gpu-memory-utilization terms.

    Rounded up to two decimals: rounding down under-claims and fails engine init,
    and one decimal is a 10%-of-node step, too coarse to fit two models on a card.
    """
    denom = node.total_vram_bytes or node.planner_vram_bytes
    if denom <= 0:
        return MIN_GPU_UTIL
    raw = charge / denom
    stepped = math.ceil(raw * 100) / 100
    return min(MAX_GPU_UTIL, max(MIN_GPU_UTIL, stepped))


def _eligible(spec: ModelSpec, node: NodeSpec, remaining: int, ctx: int) -> bool:
    return spec.runs_on(node.arch) and remaining >= need_bytes(spec, ctx)


def plan(
    specs: Sequence[ModelSpec],
    nodes: Sequence[NodeSpec],
    *,
    reserved: Optional[dict[str, int]] = None,
    replicas: int = 1,
) -> Plan:
    """Decide the placement.

    Args:
        specs: models to deploy, with metadata bound.
        nodes: probed nodes.
        reserved: per-node bytes held by resident, unplaced workloads such as
            transcription. Subtracted before packing.
        replicas: maximum instances per model. 1 disables replication.
    """
    result = Plan()
    reserved = reserved or {}

    if not nodes:
        for spec in specs:
            result.delegations.append(Delegation(spec.id, "no GPU node available"))
        return result

    remaining = {
        n.node_id: max(0, n.planner_vram_bytes - reserved.get(n.node_id, 0))
        for n in nodes
    }
    by_id = {n.node_id: n for n in nodes}

    for node in nodes:
        if node.gpu_count > 1:
            result.notes.append(
                f"{node.node_id}: treating {node.gpu_count} GPUs as a single pool — "
                "per-card placement is not configured"
            )

    # ── 1. Coverage: largest first, one each at the context floor ─────────
    ordered = sorted(specs, key=lambda s: need_bytes(s, s.ctx_floor), reverse=True)
    for spec in ordered:
        need = need_bytes(spec, spec.ctx_floor)
        candidates = [
            n for n in nodes if _eligible(spec, n, remaining[n.node_id], spec.ctx_floor)
        ]
        if not candidates:
            result.delegations.append(Delegation(spec.id, _why_not(spec, nodes, remaining)))
            continue
        # Worst fit: spreads models across nodes, leaves room for restoration
        target = max(candidates, key=lambda n: remaining[n.node_id])
        remaining[target.node_id] -= need
        result.placements.append(
            Placement(spec.id, target.node_id, spec.ctx_floor,
                      gpu_util_for(need, target), need)
        )

    # ── 2. Restoration: grow contexts toward their targets ────────────────
    _restore_context(result, specs, by_id, remaining)

    # ── 3. Replication: only when asked for ───────────────────────────────
    if replicas > 1:
        _replicate(result, specs, nodes, remaining, replicas)
        # Replicas seat at the floor too — redistribute the remainder
        _restore_context(result, specs, by_id, remaining)

    return result


def _why_not(spec: ModelSpec, nodes: Sequence[NodeSpec], remaining: dict[str, int]) -> str:
    """Delegation reason, separating "no capacity" from "cannot serve".

    Blurring the two invites a VRAM upgrade that cannot fix an architecture.
    """
    servable = [n for n in nodes if spec.runs_on(n.arch)]
    if not servable:
        arches = ", ".join(spec.arches) or "(unrestricted)"
        return f"no architecture in this cluster can serve it (supported: {arches})"
    need = need_bytes(spec, spec.ctx_floor)
    best = max((remaining[n.node_id] for n in servable), default=0)
    return (
        f"needs {need / GB:.1f} GiB even at its {spec.ctx_floor // 1024}K context floor, "
        f"and the roomiest node has {best / GB:.1f} GiB"
    )


def _restore_context(
    plan_: Plan, specs: Sequence[ModelSpec], by_id: dict[str, NodeSpec],
    remaining: dict[str, int],
) -> None:
    """Double contexts toward their targets from each node's leftover capacity.

    Furthest-from-target first, so one model cannot take everything.
    """
    spec_by_id = {s.id: s for s in specs}
    grew = True
    while grew:
        grew = False
        for node_id, node in by_id.items():
            here = plan_.for_node(node_id)
            if not here:
                continue
            # Lowest ratio to target first
            for placement in sorted(here, key=lambda p: p.ctx / spec_by_id[p.model_id].ctx_target):
                spec = spec_by_id[placement.model_id]
                if placement.ctx >= spec.ctx_target:
                    continue
                bumped = min(spec.ctx_target, placement.ctx * 2)
                extra = need_bytes(spec, bumped) - placement.charge
                if extra <= 0 or extra > remaining[node_id]:
                    continue
                remaining[node_id] -= extra
                idx = plan_.placements.index(placement)
                plan_.placements[idx] = Placement(
                    placement.model_id, node_id, bumped,
                    gpu_util_for(placement.charge + extra, node),
                    placement.charge + extra,
                )
                grew = True
                break


def _replicate(
    plan_: Plan, specs: Sequence[ModelSpec], nodes: Sequence[NodeSpec],
    remaining: dict[str, int], replicas: int,
) -> None:
    """Extra instances, once every model has one."""
    placed = {p.model_id for p in plan_.placements}
    eligible = [s for s in specs if s.id in placed]
    counts = {s.id: 1 for s in eligible}

    grew = True
    while grew:
        grew = False
        # Fewest instances first
        for spec in sorted(eligible, key=lambda s: counts[s.id]):
            if counts[spec.id] >= replicas:
                continue
            used = {p.node_id for p in plan_.placements if p.model_id == spec.id}
            need = need_bytes(spec, spec.ctx_floor)
            candidates = [
                n for n in nodes
                if n.node_id not in used
                and _eligible(spec, n, remaining[n.node_id], spec.ctx_floor)
            ]
            if not candidates:
                continue
            target = max(candidates, key=lambda n: remaining[n.node_id])
            remaining[target.node_id] -= need
            plan_.placements.append(
                Placement(spec.id, target.node_id, spec.ctx_floor,
                          gpu_util_for(need, target), need)
            )
            counts[spec.id] += 1
            grew = True
            break
