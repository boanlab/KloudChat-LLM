"""Model placement across GPU nodes.

Three phases:

1. Coverage — one instance of each model at its context floor, largest first,
   onto the node with the most capacity left. Seating at target context first
   would let a large model claim a node and starve the next one.
2. Restoration — leftover capacity raises contexts toward their targets.
3. Replication — capacity coverage did not need is filled with extra instances,
   deepening models in priority order. ``replicas`` caps it; 1 turns it off.

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

#: --gpu-memory-utilization bounds. 1.0 swallows the driver and context share,
#: failing engine init.
MAX_GPU_UTIL = 0.95
MIN_GPU_UTIL = 0.05

#: Activation, cudagraph capture, the hybrid models' per-sequence conv state and
#: non-torch buffers. Measured on GB10 by subtracting weights and the KV cache
#: vLLM reports at startup from the budget it was given: 9.7 GiB for
#: qwen3.6-35b at util 0.30, 8.5 GiB for glm-4.7-flash at 0.39.
#:
#: The old 4 GiB was not conservative, it was wrong in the direction that hurts:
#: the gap comes out of the KV cache, so a placement asking for four concurrent
#: sessions delivered 2.17. It goes unnoticed while weights are small and the
#: node has slack; at 78 GiB of weights it is most of the KV pool.
ACTIVATION_BYTES = 10 * GB

#: A pooling runner captures no CUDA graphs for a decode batch and keeps no
#: per-sequence state. Charging it the generate-path figure would reserve tens of
#: gigabytes it never touches — the same mistake in the other direction.
POOLING_ACTIVATION_BYTES = 2 * GB

#: Ceiling on activation as a share of the card. The 10 GiB above was measured
#: where a card can run 128 sequences at once; a 24 GiB card runs a handful, and
#: most of that figure — CUDA-graph capture and per-sequence conv state — scales
#: with concurrency rather than with the card. Charging the large-card number to
#: a small card said a 32 GiB card could not hold a 21 GiB model it can in fact
#: hold, and the planner delegated it to OpenRouter instead.
ACTIVATION_MAX_FRACTION: float = 0.12

#: Capacity differences below this are not a packing signal. Two GB10s reported
#: usable capacity 4 KiB apart — enough, under a strict maximum, to move a 78 GiB
#: model to the other node on a re-run for no gain whatsoever.
CAPACITY_TIE_BYTES = 1 * GB


@dataclass(frozen=True)
class Placement:
    model_id: str
    node_id: str
    ctx: int
    #: Fraction of *one* card, which is what vLLM's flag means
    gpu_util: float
    #: Bytes this placement occupies on the node. For a sharded model this is
    #: whole cards, not the model's need — see `_claim_bytes`.
    charge: int
    #: --tensor-parallel-size
    tp: int = 1
    #: Concurrent sessions this placement was sized for. Below the model's
    #: declared figure when the card could not hold the declared one.
    sessions: int = 0
    #: Card indices on the node this occupies, as CUDA device ordinals. What
    #: makes two models on a multi-card node land on different cards instead of
    #: both claiming most of card 0.
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
    """KV bytes to hold context ``ctx`` for every concurrent session.

    Two terms: the full-attention layers, whose cost grows with ``ctx``, and the
    sliding-window layers, whose cost does not — they keep a bounded window per
    sequence whatever the context.

    Zero for a pooling runner: an embedding model keeps nothing between tokens,
    so the ``2 · L · H · d · ctx`` reservation is memory it never touches.
    """
    if spec.metadata is None:
        raise ValueError(f"{spec.id}: no metadata — bind() must run first")
    if spec.is_pooling:
        return 0
    per_token = kv_bytes_per_token(spec.metadata, Dtype.FP8)
    per_seq = sliding_bytes_per_sequence(spec.metadata, Dtype.FP8)
    sessions = max(1, spec.concurrent_sessions)
    return int((per_token * ctx + per_seq) * sessions * ADMISSION_MARGIN)


def activation_bytes(spec: ModelSpec, card_bytes: Optional[int] = None) -> int:
    """Runtime headroom on top of the weights, by runner and card size.

    ``card_bytes`` is what one card has to give. Without it the large-card figure
    applies unchanged, which is what every caller that does not know the hardware
    should get: it is the conservative direction.
    """
    base = POOLING_ACTIVATION_BYTES if spec.is_pooling else ACTIVATION_BYTES
    if card_bytes and card_bytes > 0:
        return min(base, max(1 * GB, int(card_bytes * ACTIVATION_MAX_FRACTION)))
    return base


def kv_shards(spec: ModelSpec) -> int:
    """How many ways the KV cache actually divides under tensor parallelism.

    Not always ``tensor_parallel``. KV is sharded by head, so a model with fewer
    KV heads than ranks has them replicated instead — TP 4 over 2 KV heads halves
    the cache, it does not quarter it. MLA is worse: one compressed latent per
    layer, replicated on every rank, so the per-card cost does not fall at all
    and the node-wide total rises with TP.
    """
    tp = max(1, spec.tensor_parallel)
    if spec.metadata is not None and spec.metadata.kv_latent_dim:
        return 1
    heads = spec.metadata.n_kv_heads if spec.metadata else tp
    return max(1, min(tp, heads))


def per_gpu_need_bytes(spec: ModelSpec, ctx: int,
                       card_bytes: Optional[int] = None) -> int:
    """What one card has to hold: its slice of the weights and KV, plus the
    activation cost, which is per-process and does not divide."""
    tp = max(1, spec.tensor_parallel)
    return (
        spec.weight_bytes // tp
        + activation_bytes(spec, card_bytes)
        + kv_bytes(spec, ctx) // kv_shards(spec)
    )


def need_bytes(spec: ModelSpec, ctx: int, card_bytes: Optional[int] = None) -> int:
    """weights + activation + KV across every card the model occupies.

    With TP 1 this is what vLLM claims on the one card it uses. Above 1 it is the
    node-wide total, which exceeds the single-card figure: activation is paid per
    rank, and an under-sharded KV cache is paid more than once.
    """
    return per_gpu_need_bytes(spec, ctx, card_bytes) * max(1, spec.tensor_parallel)


def gpu_util_for(charge: int, node: NodeSpec) -> float:
    """Fraction of one card to request, in vLLM's --gpu-memory-utilization terms.

    ``charge`` is the per-card figure: vLLM applies the fraction to each rank's
    device, so passing the node-wide total would ask every card for the whole
    model. Rounded up to two decimals: rounding down under-claims and fails engine
    init, and one decimal is a 10%-of-card step, too coarse to fit two models on
    one.
    """
    denom = node.total_vram_bytes or node.planner_vram_bytes
    if denom <= 0:
        return MIN_GPU_UTIL
    raw = charge / denom
    stepped = math.ceil(raw * 100) / 100
    return min(MAX_GPU_UTIL, max(MIN_GPU_UTIL, stepped))


def _fit_sessions(spec: ModelSpec, node: NodeSpec, free: Sequence[int],
                  card_capacity: int) -> Optional[tuple[ModelSpec, list[int]]]:
    """The spec as it can actually be seated here, or None.

    ``concurrent_sessions`` is a sizing assumption, not a capability: it says how
    much KV to reserve so several conversations can run at once, and halving it
    costs concurrency, not context or correctness. Treating it as inviolable is
    what made a model either fit at its declared width or go to OpenRouter, with
    nothing in between — so a card that could serve one conversation served none.

    The context floor is left alone. That one *is* a capability claim: deep
    research below 128K loses the context it accumulated, and a model quietly
    seated under its floor is worse than one that is honestly absent.
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
    """One card's share of what this node actually has to give.

    Derived from ``capacity`` — the pool left after resident workloads such as
    transcription are subtracted — not from the raw card size, so a node with
    something already resident on it does not promise cards it cannot give.
    """
    return capacity // max(1, node.gpu_count)


def _assign_cards(spec: ModelSpec, node: NodeSpec, free: Sequence[int], ctx: int,
                  card_capacity: int) -> Optional[list[int]]:
    """Which cards on this node can hold the model, or None.

    Placement is per card, not per node. ``gpu_util`` is a fraction of one
    device, so a node counted as a single byte pool would seat two models at 0.86
    each and hand both of them the same card — the second dies at engine init
    with the first one's memory already in it.

    Emptiest card first, which spreads models the same way worst fit spreads them
    across nodes and leaves the most room for a later context increase.
    """
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
) -> Plan:
    """Decide the placement.

    Args:
        specs: models to deploy, with metadata bound.
        nodes: probed nodes.
        reserved: per-node bytes held by resident, unplaced workloads such as
            transcription. Subtracted before packing.
        replicas: cap on instances per model. None fills whatever capacity is
            left after coverage, deepening by priority; 1 disables replication.
        deployed: model id to the node ids already running it. A model that fits
            where it is stays there; without this the plan is free to migrate it
            on a capacity difference too small to matter.
    """
    result = Plan()
    reserved = reserved or {}
    deployed = deployed or {}

    if not nodes:
        for spec in specs:
            result.delegations.append(Delegation(spec.id, "no GPU node available"))
        return result

    #: One card's share of what a node has to give, fixed for this plan.
    card_capacity = {
        n.node_id: _per_card(
            n, max(0, n.planner_vram_bytes - reserved.get(n.node_id, 0))
        )
        for n in nodes
    }
    #: Free bytes per card, indexed by CUDA device ordinal. Shrinks as models are
    #: seated. A node is a list of cards, not a byte pool: the pool version could
    #: not tell "two models, one card each" from "two models, both on card 0".
    free = {n.node_id: [card_capacity[n.node_id]] * max(1, n.gpu_count) for n in nodes}
    by_id = {n.node_id: n for n in nodes}

    # ── 1. Coverage: one each at the context floor, by priority then size ──
    #
    # Largest-first is a starvation guard: seat the small models first and a big
    # one finds every node partly used. It says nothing about which model the
    # cluster would rather keep, so a declared `priority` outranks it. Within a
    # priority level the guard still applies.
    ordered = sorted(
        specs, key=lambda s: (s.priority, need_bytes(s, s.ctx_floor)), reverse=True
    )
    for spec in ordered:
        # Each node is asked what it can seat, narrowing the session count where
        # it has to. A node that can only take the model tighter still counts.
        holders = [n for n in nodes if _carries(n, spec)]
        seatable = {
            n.node_id: _fit_sessions(spec, n, free[n.node_id], card_capacity[n.node_id])
            for n in holders
        }
        candidates = [n for n in holders if seatable[n.node_id] is not None]
        if not candidates:
            result.delegations.append(
                Delegation(spec.id, _why_not(spec, nodes, free, card_capacity))
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

    # ── 2. Restoration: grow contexts toward their targets ────────────────
    _restore_context(result, specs, by_id, free, card_capacity)

    # ── 3. Replication: fill what coverage left ───────────────────────────
    #
    # A card that coverage did not need is a card queueing requests for no
    # reason, so the default is to use it. `replicas=1` is how a caller says
    # "one of each and stop".
    if replicas is None or replicas > 1:
        _replicate(result, specs, nodes, free, card_capacity, replicas)
        # Replicas seat at the floor too — redistribute the remainder
        _restore_context(result, specs, by_id, free, card_capacity)

    return result


def _worst_fit(
    candidates: Sequence[NodeSpec],
    free: dict[str, list[int]],
    *,
    incumbent: Optional[frozenset[str]] = None,
) -> NodeSpec:
    """The roomiest node, with near-ties resolved in favour of staying put.

    Worst fit spreads models across nodes and leaves room for restoration, and it
    still decides where a model goes when the nodes genuinely differ. Within
    ``CAPACITY_TIE_BYTES`` it decides nothing, so two other things do, in order: a
    node already running this model, then the node id. The first avoids a reload
    and a window of paid OpenRouter fallback bought with noise; the second makes
    the plan reproducible, which it was not — nodes arrive in probe-completion
    order, so the same inputs did not produce the same plan twice.

    Deliberately not a preference strong enough to survive a real capacity
    difference: a model that no longer fits where it sits has to move.
    """
    total = {n.node_id: sum(free[n.node_id]) for n in candidates}
    best = max(total.values())
    tied = [n for n in candidates if best - total[n.node_id] <= CAPACITY_TIE_BYTES]
    home = [n for n in tied if incumbent and n.node_id in incumbent]
    # Within the band, order by node id alone: ranking by remaining bytes first
    # would just re-admit the noise the band exists to ignore.
    return sorted(home or tied, key=lambda n: n.node_id)[0]


def _carries(node: NodeSpec, spec: ModelSpec) -> bool:
    """Whether the node holds this model's checkpoint.

    ``checkpoints is None`` means the probe did not report them — the old
    behaviour, where placement trusts that the weights are wherever it puts the
    container. Docker does not refuse a bind mount of a missing path; it creates
    an empty directory, and vLLM then restarts forever on a missing config.json.
    """
    return node.checkpoints is None or spec.dir in node.checkpoints


def _why_not(spec: ModelSpec, nodes: Sequence[NodeSpec], free: dict[str, list[int]],
             card_capacity: dict[str, int]) -> str:
    """Delegation reason, separating "no capacity" from "cannot serve".

    Blurring the two invites a VRAM upgrade that cannot fix an architecture.
    """
    servable = [n for n in nodes if spec.runs_on(n.arch)]
    if not servable:
        arches = ", ".join(spec.arches) or "(unrestricted)"
        return f"no architecture in this cluster can serve it (supported: {arches})"

    carrying = [n for n in servable if _carries(n, spec)]
    if not carrying:
        return (
            f"no node carries the checkpoint {spec.dir!r} under VLLM_MODELS_ROOT "
            "— capacity is not the problem, the weights are not there"
        )

    # From here the comparison is against nodes that could actually run it. A
    # message quoting the free VRAM of a node without the checkpoint reads as
    # "there is room" while naming the one place the model can never go.
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
        # Splitting further is the fix here, not a bigger node
        return (
            f"needs {per_gpu / GB:.1f} GiB per card at its "
            f"{spec.ctx_floor // 1024}K context floor (TP {tp}), and the largest "
            f"card holds {roomiest_card_capacity / GB:.1f} GiB"
        )

    # It fits a card in principle, so what is missing is free cards, not size.
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
        f"{spec.ctx_floor // 1024}K context floor, and the emptiest card in the "
        f"cluster has {freest / GB:.1f} GiB left"
    )


def _restore_context(
    plan_: Plan, specs: Sequence[ModelSpec], by_id: dict[str, NodeSpec],
    free: dict[str, list[int]], card_capacity: dict[str, int],
) -> None:
    """Double contexts toward their targets from what is left on their own cards.

    Furthest-from-target first, so one model cannot take everything. The room has
    to be on the cards the placement actually holds — free space on the node's
    other card is somebody else's.
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
                # As seated, not as declared: a placement narrowed to fit must
                # not be grown back against the width it never got.
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
    replicas: Optional[int],
) -> None:
    """Extra instances, once every model has one, highest priority first.

    ``replicas`` caps the count per model; None means fill until nothing more
    seats. Either way the loop ends when no node can take another instance.
    """
    placed = {p.model_id for p in plan_.placements}
    eligible = [s for s in specs if s.id in placed]
    counts = {s.id: 1 for s in eligible}

    grew = True
    while grew:
        grew = False
        # Fewest instances first, then by declared priority. Without the
        # priority tiebreak the second copy goes to whichever model the
        # catalogue happens to list first, which is not a decision anyone made:
        # spare capacity should deepen the model the cluster least wants to
        # queue on, and that is the same ranking coverage already uses.
        for spec in sorted(eligible, key=lambda s: (counts[s.id], -s.priority)):
            if replicas is not None and counts[spec.id] >= replicas:
                continue
            used = {p.node_id for p in plan_.placements if p.model_id == spec.id}
            seatable = {
                n.node_id: _fit_sessions(spec, n, free[n.node_id],
                                          card_capacity[n.node_id])
                for n in nodes
                if n.node_id not in used and _carries(n, spec)
            }
            candidates = [n for n in nodes if seatable.get(n.node_id) is not None]
            if not candidates:
                continue
            target = _worst_fit(candidates, free)  # a new instance has no home
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
