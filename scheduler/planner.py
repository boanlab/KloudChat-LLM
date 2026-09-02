"""Model placement across GPU nodes.

1. Coverage — one instance per model at its context floor, by priority then
   size, onto the roomiest eligible node.
2. Restoration — leftover capacity raises contexts toward their targets.
3. Replication — remaining capacity takes extra instances, weighted by
   ``share``; ``replicas`` caps it, 1 turns it off.

``placement`` restricts a model to the head node or the pool before any phase.
Unplaced models are delegated to OpenRouter with a reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from scheduler.kv_model import (
    ADMISSION_MARGIN,
    kv_bytes_per_token,
    sliding_bytes_per_sequence,
)
from scheduler.registry import ModelSpec, replace
from scheduler.types import GB, Dtype, NodeSpec

#: --gpu-memory-utilization bounds; 1.0 fails engine init
MAX_GPU_UTIL = 0.95
MIN_GPU_UTIL = 0.05

#: Runtime headroom for a generate runner: activation, CUDA-graph capture, hybrid
#: conv state (GB10 measurement: budget minus weights minus reported KV cache)
ACTIVATION_BYTES = 10 * GB

#: Pooling runners capture no decode graphs and keep no per-sequence state
POOLING_ACTIVATION_BYTES = 2 * GB

#: Activation ceiling as a fraction of the card; the figure above scales with
#: concurrency, which a small card never reaches
ACTIVATION_MAX_FRACTION: float = 0.12

#: Capacity differences below this do not decide placement
CAPACITY_TIE_BYTES = 1 * GB


@dataclass(frozen=True)
class Placement:
    model_id: str
    node_id: str
    ctx: int
    #: Fraction of one card (vLLM's --gpu-memory-utilization)
    gpu_util: float
    #: Bytes occupied on the node, summed over the cards used
    charge: int
    #: --tensor-parallel-size
    tp: int = 1
    #: Concurrent sessions sized for; below the declared figure on a tight card
    sessions: int = 0
    #: CUDA device ordinals occupied on the node
    devices: tuple[int, ...] = ()


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
    """KV bytes for ``ctx`` across every concurrent session; zero for pooling runners."""
    if spec.metadata is None:
        raise ValueError(f"{spec.id}: no metadata — bind() must run first")
    if spec.is_pooling:
        return 0
    per_token = kv_bytes_per_token(spec.metadata, Dtype.FP8)
    per_seq = sliding_bytes_per_sequence(spec.metadata, Dtype.FP8)
    sessions = max(1, spec.concurrent_sessions)
    return int((per_token * ctx + per_seq) * sessions * ADMISSION_MARGIN)


def activation_bytes(spec: ModelSpec, card_bytes: Optional[int] = None) -> int:
    """Runtime headroom by runner, capped by card size; the full figure without ``card_bytes``."""
    base = POOLING_ACTIVATION_BYTES if spec.is_pooling else ACTIVATION_BYTES
    if card_bytes and card_bytes > 0:
        return min(base, max(1 * GB, int(card_bytes * ACTIVATION_MAX_FRACTION)))
    return base


def kv_shards(spec: ModelSpec) -> int:
    """Ways the KV cache divides under TP: ``min(tp, kv_heads)``; 1 for MLA (latent replicated per rank)."""
    tp = max(1, spec.tensor_parallel)
    if spec.metadata is not None and spec.metadata.kv_latent_dim:
        return 1
    heads = spec.metadata.n_kv_heads if spec.metadata else tp
    return max(1, min(tp, heads))


def per_gpu_need_bytes(spec: ModelSpec, ctx: int,
                       card_bytes: Optional[int] = None) -> int:
    """Per-card need: weight and KV slice plus the full (per-rank) activation cost."""
    tp = max(1, spec.tensor_parallel)
    return (
        spec.weight_bytes // tp
        + activation_bytes(spec, card_bytes)
        + kv_bytes(spec, ctx) // kv_shards(spec)
    )


def need_bytes(spec: ModelSpec, ctx: int, card_bytes: Optional[int] = None) -> int:
    """Node-wide need across every card the model occupies."""
    return per_gpu_need_bytes(spec, ctx, card_bytes) * max(1, spec.tensor_parallel)


def gpu_util_for(charge: int, node: NodeSpec) -> float:
    """--gpu-memory-utilization for a per-card ``charge``, rounded up to two decimals."""
    denom = node.total_vram_bytes or node.planner_vram_bytes
    if denom <= 0:
        return MIN_GPU_UTIL
    raw = charge / denom
    stepped = math.ceil(raw * 100) / 100
    return min(MAX_GPU_UTIL, max(MIN_GPU_UTIL, stepped))


def _fit_sessions(spec: ModelSpec, node: NodeSpec, free: Sequence[int],
                  card_capacity: int) -> Optional[tuple[ModelSpec, list[int]]]:
    """The spec as seatable on this node, or None.

    ``concurrent_sessions`` is halved down to fit; the context floor is never traded away.
    """
    sessions = max(1, spec.concurrent_sessions)
    while sessions >= 1:
        candidate = replace(spec, concurrent_sessions=sessions)
        cards = _assign_cards(candidate, node, free, candidate.ctx_floor, card_capacity)
        if cards is not None:
            return candidate, cards
        sessions //= 2
    return None


def _per_card(node: NodeSpec, capacity: int) -> int:
    """One card's share of ``capacity`` (node capacity after reservations)."""
    return capacity // max(1, node.gpu_count)


def _assign_cards(spec: ModelSpec, node: NodeSpec, free: Sequence[int], ctx: int,
                  card_capacity: int) -> Optional[list[int]]:
    """Cards on this node that can hold the model (emptiest first), or None."""
    if not spec.runs_on(node.arch):
        return None
    tp = max(1, spec.tensor_parallel)
    if tp > node.gpu_count:
        return None
    need = per_gpu_need_bytes(spec, ctx, card_capacity)
    if need > card_capacity:
        return None
    order = sorted(range(len(free)), key=lambda i: (-free[i], i))
    chosen = [i for i in order if free[i] >= need][:tp]
    return sorted(chosen) if len(chosen) == tp else None


def plan(
    specs: Sequence[ModelSpec],
    nodes: Sequence[NodeSpec],
    *,
    reserved: Optional[dict[str, int]] = None,
    replicas: Optional[int] = None,
    deployed: Optional[dict[str, frozenset[str]]] = None,
    head: Optional[str] = None,
) -> Plan:
    """Decide the placement.

    Args:
        specs: models to deploy, with metadata bound.
        nodes: probed nodes.
        reserved: per-node bytes held by resident workloads, subtracted first.
        replicas: cap on instances per model; None fills spare capacity, 1 disables.
        deployed: model id to node ids already running it; a near-tie keeps it there.
        head: head node id (first in NODES_VLLM). None on a single-node cluster,
            where ``placement`` constrains nothing.
    """
    result = Plan()
    reserved = reserved or {}
    deployed = deployed or {}

    if not nodes:
        for spec in specs:
            result.delegations.append(Delegation(spec.id, "no GPU node available"))
        return result

    # Per-card capacity, fixed for this plan
    card_capacity = {
        n.node_id: _per_card(
            n, max(0, n.planner_vram_bytes - reserved.get(n.node_id, 0))
        )
        for n in nodes
    }
    # Free bytes per card, by CUDA device ordinal
    free = {n.node_id: [card_capacity[n.node_id]] * max(1, n.gpu_count) for n in nodes}
    by_id = {n.node_id: n for n in nodes}

    # 1. Coverage: one each at the context floor, by priority then size
    ordered = sorted(
        specs, key=lambda s: (s.priority, need_bytes(s, s.ctx_floor)), reverse=True
    )
    for spec in ordered:
        allowed = _eligible(spec, nodes, head)
        holders = [n for n in allowed if _carries(n, spec)]
        seatable = {
            n.node_id: _fit_sessions(spec, n, free[n.node_id], card_capacity[n.node_id])
            for n in holders
        }
        candidates = [n for n in holders if seatable[n.node_id] is not None]
        if not candidates:
            result.delegations.append(
                Delegation(spec.id, _why_not(spec, allowed, free, card_capacity))
            )
            continue
        target = _worst_fit(candidates, free, incumbent=deployed.get(spec.id))
        seated, cards = seatable[target.node_id]
        if seated.concurrent_sessions < spec.concurrent_sessions:
            result.notes.append(
                f"{spec.id} on {target.node_id}: sized for "
                f"{seated.concurrent_sessions} concurrent sessions, not "
                f"{spec.concurrent_sessions} — the card has no room for more KV"
            )
        cap = card_capacity[target.node_id]
        per_card = per_gpu_need_bytes(seated, seated.ctx_floor, cap)
        for i in cards:
            free[target.node_id][i] -= per_card
        result.placements.append(
            Placement(spec.id, target.node_id, seated.ctx_floor,
                      gpu_util_for(per_card, target), per_card * len(cards),
                      max(1, spec.tensor_parallel), seated.concurrent_sessions,
                      tuple(cards))
        )

    # 2. Restoration
    _restore_context(result, specs, by_id, free, card_capacity)

    # 3. Replication, then restoration again for the replicas
    if replicas is None or replicas > 1:
        _replicate(result, specs, nodes, free, card_capacity, replicas, head)
        _restore_context(result, specs, by_id, free, card_capacity)

    return result


def _worst_fit(
    candidates: Sequence[NodeSpec],
    free: dict[str, list[int]],
    *,
    incumbent: Optional[frozenset[str]] = None,
) -> NodeSpec:
    """Roomiest node; within ``CAPACITY_TIE_BYTES`` the incumbent wins, then the lowest node id."""
    total = {n.node_id: sum(free[n.node_id]) for n in candidates}
    best = max(total.values())
    tied = [n for n in candidates if best - total[n.node_id] <= CAPACITY_TIE_BYTES]
    home = [n for n in tied if incumbent and n.node_id in incumbent]
    return sorted(home or tied, key=lambda n: n.node_id)[0]


def _eligible(spec: ModelSpec, nodes: Sequence[NodeSpec],
              head: Optional[str]) -> list[NodeSpec]:
    """Nodes allowed by ``placement``; unfiltered when ``head`` is None (single-node cluster)."""
    if head is None or spec.placement == "any":
        return list(nodes)
    if spec.placement == "head":
        return [n for n in nodes if n.node_id == head]
    return [n for n in nodes if n.node_id != head]


def _carries(node: NodeSpec, spec: ModelSpec) -> bool:
    """Node holds the checkpoint, or checkpoints were not probed."""
    return node.checkpoints is None or spec.dir in node.checkpoints


def _why_not(spec: ModelSpec, nodes: Sequence[NodeSpec], free: dict[str, list[int]],
             card_capacity: dict[str, int]) -> str:
    """Delegation reason: no eligible node, architecture, missing checkpoint, cards, or capacity.

    Measured over ``nodes`` (what ``placement`` left), not the whole cluster.
    """
    if not nodes:
        if spec.placement == "head":
            return ("the head node — the first in NODES_VLLM — is not answering, "
                    "and placement: head keeps this model off the pool")
        return ("no pool node is answering — the pool is every node in "
                "NODES_VLLM but the first")

    where = _scope(spec)
    servable = [n for n in nodes if spec.runs_on(n.arch)]
    if not servable:
        arches = ", ".join(spec.arches) or "(unrestricted)"
        return f"no architecture in {where} can serve it (supported: {arches})"

    carrying = [n for n in servable if _carries(n, spec)]
    if not carrying:
        return (
            f"no node carries the checkpoint {spec.dir!r} under VLLM_MODELS_ROOT "
            "— capacity is not the problem, the weights are not there"
        )

    tp = max(1, spec.tensor_parallel)
    wide_enough = [n for n in carrying if tp <= n.gpu_count]
    if not wide_enough:
        most = max((n.gpu_count for n in carrying), default=0)
        return (
            f"needs {tp} cards on one node for tensor parallelism, "
            f"and the widest node has {most}"
        )

    roomiest_card_capacity = max(card_capacity[n.node_id] for n in wide_enough)
    per_gpu = per_gpu_need_bytes(spec, spec.ctx_floor, roomiest_card_capacity)
    if per_gpu > roomiest_card_capacity:
        return (
            f"needs {per_gpu / GB:.1f} GiB per card at its "
            f"{spec.ctx_floor // 1024}K context floor (TP {tp}), and the largest "
            f"card holds {roomiest_card_capacity / GB:.1f} GiB"
        )

    freest = max(
        (max(free[n.node_id]) for n in wide_enough), default=0
    )
    if tp > 1:
        available = max(
            (sum(1 for f in free[n.node_id] if f >= per_gpu) for n in wide_enough),
            default=0,
        )
        return (
            f"needs {tp} cards with {per_gpu / GB:.1f} GiB each, and the best node "
            f"has {available} card(s) that free"
        )
    return (
        f"needs {per_gpu / GB:.1f} GiB on one card at its "
        f"{spec.ctx_floor // 1024}K context floor, and the emptiest card in "
        f"{where} has {freest / GB:.1f} GiB left"
    )


def _scope(spec: ModelSpec) -> str:
    """Scope a delegation reason's figures were measured over."""
    return {"head": "the head node", "pool": "the pool"}.get(
        spec.placement, "this cluster"
    )


def _restore_context(
    plan_: Plan, specs: Sequence[ModelSpec], by_id: dict[str, NodeSpec],
    free: dict[str, list[int]], card_capacity: dict[str, int],
) -> None:
    """Double contexts toward their targets from free space on their own cards, furthest-from-target first."""
    spec_by_id = {s.id: s for s in specs}
    grew = True
    while grew:
        grew = False
        for node_id, node in by_id.items():
            here = plan_.for_node(node_id)
            if not here:
                continue
            for placement in sorted(here, key=lambda p: p.ctx / spec_by_id[p.model_id].ctx_target):
                spec = spec_by_id[placement.model_id]
                # Sized as seated, not as declared
                if placement.sessions:
                    spec = replace(spec, concurrent_sessions=placement.sessions)
                if placement.ctx >= spec.ctx_target:
                    continue
                bumped = min(spec.ctx_target, placement.ctx * 2)
                cap = card_capacity[node_id]
                grown = per_gpu_need_bytes(spec, bumped, cap)
                if grown > cap:
                    continue
                extra = grown - per_gpu_need_bytes(spec, placement.ctx, cap)
                devices = placement.devices or (0,)
                if extra <= 0 or any(free[node_id][i] < extra for i in devices):
                    continue
                for i in devices:
                    free[node_id][i] -= extra
                idx = plan_.placements.index(placement)
                plan_.placements[idx] = Placement(
                    placement.model_id, node_id, bumped,
                    gpu_util_for(grown, node), grown * len(devices),
                    placement.tp, placement.sessions, placement.devices,
                )
                grew = True
                break


def _replicate(
    plan_: Plan, specs: Sequence[ModelSpec], nodes: Sequence[NodeSpec],
    free: dict[str, list[int]], card_capacity: dict[str, int],
    replicas: Optional[int], head: Optional[str] = None,
) -> None:
    """Extra instances, one per round to the model furthest below its share (``instances / share``)."""
    placed = {p.model_id for p in plan_.placements}
    eligible = [s for s in specs if s.id in placed]
    counts = {s.id: 1 for s in eligible}

    grew = True
    while grew:
        grew = False
        for spec in sorted(eligible,
                           key=lambda s: (counts[s.id] / s.share, -s.priority)):
            if replicas is not None and counts[spec.id] >= replicas:
                continue
            used = {p.node_id for p in plan_.placements if p.model_id == spec.id}
            seatable = {
                n.node_id: _fit_sessions(spec, n, free[n.node_id],
                                          card_capacity[n.node_id])
                for n in _eligible(spec, nodes, head)
                if n.node_id not in used and _carries(n, spec)
            }
            candidates = [n for n in nodes if seatable.get(n.node_id) is not None]
            if not candidates:
                continue
            target = _worst_fit(candidates, free)
            seated, cards = seatable[target.node_id]
            cap = card_capacity[target.node_id]
            per_card = per_gpu_need_bytes(seated, seated.ctx_floor, cap)
            for i in cards:
                free[target.node_id][i] -= per_card
            plan_.placements.append(
                Placement(spec.id, target.node_id, seated.ctx_floor,
                          gpu_util_for(per_card, target), per_card * len(cards),
                          max(1, spec.tensor_parallel), seated.concurrent_sessions,
                          tuple(cards))
            )
            counts[spec.id] += 1
            grew = True
            break
