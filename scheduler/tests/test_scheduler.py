"""Scheduler tests: the memory arithmetic and the placement policy.

    pytest scheduler/tests -q
    PYTHONPATH=. python3 scheduler/tests/test_scheduler.py   # without pytest
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

from scheduler import applier, planner, registry
from scheduler.kv_model import kv_bytes_per_token
from scheduler.types import GB, Dtype, ModelMetadata, NodeSpec


def _meta(**kw) -> ModelMetadata:
    base = dict(
        model_id="test/model", n_layers=40, n_kv_heads=8, head_dim=128,
        weight_dtype=Dtype.NVFP4, weight_bytes=20 * GB, native_ctx=131072,
    )
    base.update(kw)
    return ModelMetadata(**base)


def _spec(model_id: str, *, weight: int, ctx_floor: int = 0,
          native: int = 131072, arches=(), priority: int = 0,
          **kw) -> registry.ModelSpec:
    spec = registry.ModelSpec(
        id=model_id, hf_repo=f"org/{model_id}", dir=model_id,
        service=f"vllm-{model_id}", port=8001,
        env_prefix=f"VLLM_{model_id.upper()}", served_name=f"local/{model_id}",
        ctx_floor=ctx_floor, concurrent_sessions=1, or_twin=None, arches=arches,
        priority=priority,
    )
    return spec.bind(_meta(weight_bytes=weight, **kw), native)


def _node(node_id: str, gib: int, arch: str = "amd64",
          checkpoints=None) -> NodeSpec:
    return NodeSpec(node_id=node_id, hostname=f"user@{node_id}",
                    gpu_class="pro6000", total_vram_bytes=gib * GB, arch=arch,
                    checkpoints=checkpoints)


# ── memory arithmetic ───────────────────────────────────────────────────


def test_mla_kv_is_an_order_smaller_than_mha():
    """Sizing MLA with the MHA formula overestimates by an order of magnitude,
    which rejects placements that would have fit."""
    mha = kv_bytes_per_token(_meta(n_layers=47, n_kv_heads=20), Dtype.FP8)
    mla = kv_bytes_per_token(_meta(n_layers=47, n_kv_heads=20, kv_latent_dim=576), Dtype.FP8)
    assert mla * 5 < mha, f"MLA {mla} should be well below MHA {mha}"


def test_sliding_window_layers_cost_per_sequence_not_per_token():
    """Gemma is five sliding layers per full one. Charged as full attention they
    dwarf the model; charged as nothing they are a gigabyte nobody budgeted."""
    from scheduler.kv_model import sliding_bytes_per_sequence

    md = _meta(n_layers=5, n_kv_heads=8, head_dim=256,
               sliding_layers=25, sliding_window=1024)
    per_seq = sliding_bytes_per_sequence(md, Dtype.FP8)
    assert per_seq == 25 * 2 * 8 * 256 * 1024

    # Flat: a longer context does not move it
    assert per_seq == sliding_bytes_per_sequence(md, Dtype.FP8)
    # And a model without them pays nothing
    assert sliding_bytes_per_sequence(_meta(), Dtype.FP8) == 0


def test_sliding_layers_are_read_from_the_config():
    from scheduler import model_metadata

    assert model_metadata._count_sliding_layers({
        "layer_types": ["sliding_attention"] * 25 + ["full_attention"] * 5,
        "sliding_window": 1024,
    }) == (25, 1024)
    # No window declared is no charge, whatever the layer names say
    assert model_metadata._count_sliding_layers(
        {"layer_types": ["sliding_attention"] * 25}
    ) == (0, 0)


def test_a_hybrid_written_as_a_stride_is_still_a_hybrid():
    """Qwen3-Next states the pattern as `full_attention_interval` rather than a
    `layer_types` list. Read as pure attention it costs 4x its real KV, which is
    the difference between fitting on a card and being delegated."""
    from scheduler import model_metadata

    stride = {
        "num_hidden_layers": 48, "full_attention_interval": 4,
        "linear_key_head_dim": 128, "num_key_value_heads": 2, "head_dim": 256,
    }
    assert model_metadata._count_kv_bearing_layers(stride) == 12
    # The field alone is not enough: without linear attention every layer counts
    assert model_metadata._count_kv_bearing_layers(
        {"num_hidden_layers": 48, "full_attention_interval": 4}
    ) == 48


def test_kv_bearing_layers_drive_cost():
    """Hybrid models carry KV on only some layers; counting all of them overestimates."""
    full = kv_bytes_per_token(_meta(n_layers=40), Dtype.FP8)
    hybrid = kv_bytes_per_token(_meta(n_layers=10), Dtype.FP8)
    assert hybrid * 4 == full


def test_fp8_kv_halves_bf16():
    assert (kv_bytes_per_token(_meta(), Dtype.FP8) * 2
            == kv_bytes_per_token(_meta(), Dtype.BF16))


# ── placement policy ────────────────────────────────────────────────────


def test_every_model_placed_before_any_replica():
    """Coverage first: nothing gets a second copy while another model has none.

    Replication fills what is left, but never at the cost of a model that is not
    running anywhere — the model with no instance is the one that has to queue.
    """
    specs = [_spec("a", weight=20 * GB), _spec("b", weight=20 * GB)]
    result = planner.plan(specs, [_node("n1", 96), _node("n2", 96)])
    counts = Counter(p.model_id for p in result.placements)
    assert set(counts) == {"a", "b"}, counts
    assert max(counts.values()) - min(counts.values()) <= 1, counts
    assert not result.delegations


def test_spare_capacity_is_filled_unless_capped():
    """A card coverage did not need queues requests for nothing, so it is used."""
    specs = [_spec("a", weight=20 * GB), _spec("b", weight=20 * GB)]
    nodes = [_node("n1", 96), _node("n2", 96), _node("n3", 96)]

    once = planner.plan(specs, nodes, replicas=1)
    assert len(once.placements) == 2, "replicas=1 is how a caller turns it off"

    filled = planner.plan(specs, nodes)
    assert len(filled.placements) > 2, "the default should use what is left"

    capped = planner.plan(specs, nodes, replicas=2)
    ids = [p.model_id for p in capped.placements]
    assert ids.count("a") == 2 and ids.count("b") == 2, ids
    assert len(capped.placements) < len(filled.placements)


def test_replicas_deepen_in_priority_order():
    """Spare capacity goes to the highest-ranked model first, then down the list.

    Nodes are sized to hold exactly one of these, so the question is only which
    model the spare node gets. Without the priority tiebreak it goes to whichever
    the catalogue lists first, which is not a decision anyone made.
    """
    def specs():  # plan() binds to the specs, so build them fresh per call
        return [
            _spec("third", weight=20 * GB, priority=1),
            _spec("first", weight=20 * GB, priority=3),
            _spec("second", weight=20 * GB, priority=2),
        ]

    four = [_node(f"n{i}", 32) for i in range(1, 5)]
    ids = [p.model_id for p in planner.plan(specs(), four, replicas=2).placements]
    assert ids.count("first") == 2, ids
    assert ids.count("second") == 1 and ids.count("third") == 1, ids

    # One more node: the second copy of "second" follows; "third" still waits.
    five = four + [_node("n5", 32)]
    ids = [p.model_id for p in planner.plan(specs(), five, replicas=2).placements]
    assert ids.count("first") == 2 and ids.count("second") == 2, ids
    assert ids.count("third") == 1, ids


def test_a_model_is_not_placed_where_its_weights_are_not():
    """Docker creates the missing bind-mount path empty, so vLLM restarts forever.

    No node carries the checkpoint, so the model is delegated — and the reason
    says weights, not VRAM, because no VRAM upgrade would fix it.
    """
    spec = _spec("a", weight=20 * GB)
    nodes = [_node("n1", 96, checkpoints=frozenset({"something-else"}))]
    result = planner.plan([spec], nodes)
    assert not result.placements
    assert len(result.delegations) == 1
    reason = result.delegations[0].reason
    assert "checkpoint" in reason and "'a'" in reason, reason


def test_the_capacity_reason_only_counts_nodes_that_could_run_it():
    """A roomy node without the checkpoint is not room.

    Quoting its free VRAM reads as "there is space" while naming the one place
    the model can never go, which sends the reader looking for a packing bug.
    """
    spec = _spec("a", weight=20 * GB)
    nodes = [
        _node("roomy", 96, checkpoints=frozenset({"something-else"})),
        _node("carrier", 4, checkpoints=frozenset({"a"})),
    ]
    result = planner.plan([spec], nodes)
    assert not result.placements
    reason = result.delegations[0].reason
    assert "88" not in reason, f"quotes the roomy node it cannot use: {reason}"


def test_replicas_only_land_on_nodes_that_carry_the_checkpoint():
    """Filling spare capacity must not seat a copy onto weights that are absent.

    Two nodes with room, one of them without the checkpoint: the replica has
    nowhere to go and the model stays at one instance.
    """
    spec = _spec("a", weight=20 * GB)
    nodes = [
        _node("has", 96, checkpoints=frozenset({"a"})),
        _node("lacks", 96, checkpoints=frozenset({"something-else"})),
    ]
    result = planner.plan([spec], nodes)
    assert {p.node_id for p in result.placements} == {"has"}
    assert len(result.placements) >= 1
    assert all(p.node_id == "has" for p in result.placements), result.placements


def test_unreported_checkpoints_do_not_filter_anything():
    """`checkpoints=None` is "the probe did not say", not "the node has nothing"."""
    spec = _spec("a", weight=20 * GB)
    result = planner.plan([spec], [_node("n1", 96)])
    assert [p.node_id for p in result.placements] == ["n1"]


def test_context_restored_above_floor():
    """Capacity left after seating at the floor goes back into context."""
    spec = _spec("a", weight=20 * GB, ctx_floor=16384, native=131072)
    result = planner.plan([spec], [_node("n1", 96)])
    assert result.placements[0].ctx > spec.ctx_floor
    assert result.placements[0].ctx <= spec.ctx_target


def test_small_node_delegates_with_capacity_reason():
    spec = _spec("a", weight=40 * GB, ctx_floor=32768)
    result = planner.plan([spec], [_node("n1", 24)])
    assert not result.placements
    assert "GiB" in result.delegations[0].reason


def test_unsupported_arch_is_not_a_capacity_message():
    """Reporting an unsupported architecture as missing capacity invites someone
    to fix it by adding VRAM."""
    spec = _spec("a", weight=1 * GB, arches=("amd64",))
    result = planner.plan([spec], [_node("gb10", 128, arch="arm64")])
    assert not result.placements
    assert "architecture" in result.delegations[0].reason


def test_reservation_shrinks_capacity():
    # Sized off the activation figure rather than a literal, so a correction to
    # it moves the fixture instead of breaking the case: the model fits the bare
    # node (48 GiB less the 8 GiB default reserve) and not once 20 GiB is held.
    node = [_node("n1", 48)]
    weight = 38 * GB - planner.ACTIVATION_BYTES
    spec = _spec("a", weight=weight, ctx_floor=16384)
    assert planner.plan([spec], node).placements
    assert not planner.plan([spec], node, reserved={"n1": 20 * GB}).placements


def test_a_pooling_model_is_not_charged_decode_headroom():
    """An embedding model captures no decode CUDA graphs and holds no KV, so the
    generate-path figure would reserve tens of gigabytes it never touches."""
    weight = 4 * GB
    generate = _spec("gen", weight=weight, ctx_floor=8192)
    pooling = registry.replace(generate, runner="pooling")
    assert planner.need_bytes(pooling, 8192) < planner.need_bytes(generate, 8192)
    assert planner.need_bytes(pooling, 8192) == weight + planner.POOLING_ACTIVATION_BYTES


def test_tensor_parallel_splits_the_weights_but_not_the_activation():
    """Per card: its slice of the weights and KV, plus the full activation cost —
    that one is per process and does not divide."""
    spec = registry.replace(
        _spec("a", weight=80 * GB, ctx_floor=16384), tensor_parallel=2
    )
    per_gpu = planner.per_gpu_need_bytes(spec, 16384)
    kv = planner.kv_bytes(spec, 16384)
    assert per_gpu == 40 * GB + planner.ACTIVATION_BYTES + kv // 2
    # Node-wide it costs more than the unsharded model: activation twice over
    assert planner.need_bytes(spec, 16384) > 80 * GB + planner.ACTIVATION_BYTES + kv


def test_a_model_too_big_for_one_card_fits_across_two():
    """The case this exists for: 78 GiB of weights against a 96 GB card."""
    spec = _spec("big", weight=78 * GB, ctx_floor=16384)
    one_card = NodeSpec(node_id="n1", hostname="n1", gpu_class="pro6000",
                        total_vram_bytes=89 * GB, gpu_count=1, arch="amd64")
    two_cards = NodeSpec(node_id="n2", hostname="n2", gpu_class="pro6000",
                         total_vram_bytes=89 * GB, gpu_count=2, arch="amd64")

    assert not planner.plan([spec], [one_card]).placements
    sharded = registry.replace(spec, tensor_parallel=2)
    placed = planner.plan([sharded], [two_cards]).placements
    assert placed and placed[0].tp == 2
    # The flag is a fraction of one card, not of the node
    assert placed[0].gpu_util < 1.0


def test_a_reserved_workload_does_not_make_sharding_impossible():
    """The whole-card claim has to come out of what the node actually has. Taken
    from the raw card size it exceeded the pool on an empty node as soon as
    anything was reserved there, and no sharded model could ever be placed."""
    spec = registry.replace(_spec("a", weight=40 * GB, ctx_floor=16384),
                            tensor_parallel=2)
    node = NodeSpec(node_id="n1", hostname="n1", gpu_class="pro6000",
                    total_vram_bytes=89 * GB, gpu_count=2, arch="amd64")
    result = planner.plan([spec], [node], reserved={"n1": 6 * GB})
    assert result.placements, result.delegations[0].reason
    assert result.placements[0].tp == 2


def test_tensor_parallel_needs_the_cards_and_says_so():
    """"Not enough memory" would send someone shopping for a bigger card when
    what is missing is a second one."""
    spec = registry.replace(_spec("a", weight=10 * GB, ctx_floor=16384),
                            tensor_parallel=4)
    result = planner.plan([spec], [_node("n1", 96)])
    assert not result.placements
    assert "cards" in result.delegations[0].reason


def test_two_models_on_a_two_card_node_get_a_card_each():
    """`gpu_util` is a fraction of one device. Counted as a single byte pool the
    node seated both at 0.86 and handed them the same card, and the second died
    at engine init with the first one's memory already in it."""
    a = _spec("a", weight=30 * GB, ctx_floor=16384)
    b = _spec("b", weight=30 * GB, ctx_floor=16384)
    node = NodeSpec(node_id="n1", hostname="n1", gpu_class="pro5000",
                    total_vram_bytes=48 * GB, gpu_count=2, arch="amd64")
    result = planner.plan([a, b], [node])
    assert len(result.placements) == 2
    assert {p.devices for p in result.placements} == {(0,), (1,)}


def test_no_card_is_oversubscribed():
    """The invariant the pool could not state: whatever shares a card, the
    fractions those containers ask vLLM for have to add up to less than all of it."""
    specs = [_spec(name, weight=12 * GB, ctx_floor=16384) for name in "abcde"]
    node = NodeSpec(node_id="n1", hostname="n1", gpu_class="pro5000",
                    total_vram_bytes=48 * GB, gpu_count=2, arch="amd64")
    result = planner.plan(specs, [node])
    per_card: dict[int, float] = {}
    for p in result.placements:
        for dev in p.devices:
            per_card[dev] = per_card.get(dev, 0.0) + p.gpu_util
    assert per_card, "nothing was placed"
    for dev, used in per_card.items():
        assert used <= 1.0, f"card {dev} oversubscribed at {used:.2f}"


def test_a_sharded_model_takes_one_slice_of_each_card_it_spans():
    big = registry.replace(_spec("big", weight=20 * GB, ctx_floor=16384),
                           tensor_parallel=2)
    node = NodeSpec(node_id="n1", hostname="n1", gpu_class="pro6000",
                    total_vram_bytes=48 * GB, gpu_count=2, arch="amd64")
    result = planner.plan([big], [node])
    assert result.placements[0].devices == (0, 1)
    assert result.placements[0].gpu_util < 1.0


def test_kv_is_replicated_when_ranks_outnumber_kv_heads():
    """KV shards by head. TP 4 over 2 KV heads halves the cache, it does not
    quarter it, and MLA replicates its latent on every rank."""
    gqa = registry.replace(_spec("gqa", weight=10 * GB, ctx_floor=16384,
                                 n_kv_heads=2), tensor_parallel=4)
    assert planner.kv_shards(gqa) == 2

    mla = registry.replace(_spec("mla", weight=10 * GB, ctx_floor=16384,
                                 kv_latent_dim=576), tensor_parallel=4)
    assert planner.kv_shards(mla) == 1


# ── across card sizes ───────────────────────────────────────────────────
#
# Every case above this line runs on a 96 GiB node, which is how three sizing
# figures drifted into large-card absolutes without anyone noticing: a fixed
# 8 GiB reserve is 8% of a 96 GiB card and 33% of a 24 GiB one, and a 10 GiB
# activation charge is 42% of that same small card. The cluster this was written
# on had no small card to fail on.

#: Marketing size to what the card really reports, in GiB.
CARD_SIZES = [24, 32, 48, 80, 96]


def _narrow_kv_spec(*, sessions: int, weight: int = 21 * GB):
    """A 21 GiB model with the hybrid lineup's KV shape: 10 KiB per token."""
    spec = _spec("a", weight=weight, ctx_floor=131072,
                 n_layers=10, n_kv_heads=2, head_dim=256)
    return registry.replace(spec, concurrent_sessions=sessions)


# Looped rather than parametrized: this module also runs without pytest
# (see the docstring), and that harness calls every test_* with no arguments.
def test_headroom_never_eats_the_card():
    """Reserve plus activation has to leave room for a model on every card, or
    the planner is rejecting hardware that works."""
    spec = _spec("a", weight=1 * GB, ctx_floor=16384)
    for gib in CARD_SIZES:
        node = _node("n1", gib)
        overhead = node.effective_reserve_bytes + planner.activation_bytes(
            spec, node.per_gpu_planner_bytes
        )
        assert overhead < gib * GB * 0.45, (
            f"{gib}GiB card: {overhead / GB:.1f}GiB of headroom before any weights"
        )


def test_a_card_seats_a_model_that_fits_its_weights():
    """The floor test: a model whose weights take half the card must be
    placeable, narrowing its session count if that is what it takes."""
    for gib in CARD_SIZES:
        weight = int(gib * GB * 0.5)
        spec = _spec("a", weight=weight, ctx_floor=16384)
        result = planner.plan([spec], [_node("n1", gib)])
        assert result.placements, (
            f"{gib}GiB card rejected a {weight / GB:.0f}GiB model: "
            f"{result.delegations[0].reason}"
        )
        assert result.placements[0].gpu_util <= planner.MAX_GPU_UTIL


def test_util_stays_within_bounds_on_every_card():
    for gib in CARD_SIZES:
        spec = _spec("a", weight=int(gib * GB * 0.4), ctx_floor=16384)
        for p in planner.plan([spec], [_node("n1", gib)]).placements:
            assert planner.MIN_GPU_UTIL <= p.gpu_util <= planner.MAX_GPU_UTIL, \
                f"{gib}GiB card produced util {p.gpu_util}"


def test_a_tight_card_narrows_sessions_instead_of_delegating():
    """concurrent_sessions is a sizing assumption, not a capability. Treating it
    as inviolable made a card that could serve one conversation serve none."""
    # Shaped like the models actually deployed — 10 of 40 layers KV-bearing at
    # 2 heads — rather than the module's dense default, whose 80 KiB/token would
    # not fit a 32 GiB card at any session count and would prove nothing.
    result = planner.plan([_narrow_kv_spec(sessions=8)], [_node("n1", 32)])
    assert result.placements, result.delegations[0].reason
    assert result.placements[0].sessions < 8
    assert result.placements[0].ctx == 131072, "the context floor must not be traded away"
    assert any("concurrent sessions" in n for n in result.notes), \
        "a narrowed placement has to say so"


def test_a_roomy_card_is_not_narrowed():
    result = planner.plan([_narrow_kv_spec(sessions=8)], [_node("n1", 96)])
    assert result.placements[0].sessions == 8
    assert not result.notes


def test_gpu_class_names_agree_with_the_shell_side():
    """gpu_class is looked up in per-class tables on both sides. Returning the raw
    marketing name for an unrecognised card matched nothing the shell would have
    matched, while the docstring said the vocabulary was shared."""
    from scheduler import inventory

    assert inventory._classify_gpu_name("NVIDIA GB10") == "gb10"
    assert inventory._classify_gpu_name("NVIDIA RTX PRO 6000 Blackwell") == "pro6000"
    # lib.sh::detect_gpu_class answers "nvidia-other" for anything it cannot name
    assert inventory._classify_gpu_name("NVIDIA A100-SXM4-80GB") == "nvidia-other"
    # A failed probe is a different thing from a card we could not name
    assert inventory._classify_gpu_name("") == "unknown"


def test_a_mixed_box_is_sized_by_its_smallest_card():
    """A model runs on one card, so every card has to hold what the planner
    promised. "First card times how many" handed a 4090-plus-5090 box a capacity
    neither device has."""
    from scheduler import inventory

    sizes = (24 * GB, 32 * GB)
    assert min(sizes) * len(sizes) < sum(sizes)  # the difference the old form lost
    node = NodeSpec(node_id="n1", hostname="n1", gpu_class="mixed",
                    total_vram_bytes=min(sizes), gpu_count=len(sizes))
    # Per-card capacity is the small card's, not the average of the two
    assert node.per_gpu_planner_bytes <= 24 * GB
    assert inventory.MANAGED_PREFIX == "vllm-"


def test_memory_someone_else_holds_is_not_offered():
    """vLLM's utilisation fraction is of the card's total, but the memory has to
    be free. A card with a desktop session on it has less to give than its size."""
    clean = _node("n1", 48)
    busy = NodeSpec(node_id="n2", hostname="n2", gpu_class="pro5000",
                    total_vram_bytes=48 * GB, foreign_vram_bytes=10 * GB)
    assert busy.planner_vram_bytes == clean.planner_vram_bytes - 10 * GB

    # And it changes placement, which is the point of measuring it
    spec = _spec("a", weight=30 * GB, ctx_floor=16384)
    assert planner.plan([spec], [clean]).placements
    assert not planner.plan([spec], [busy]).placements


def test_the_reserve_scales_with_the_card():
    small, large = _node("s", 24), _node("l", 96)
    assert small.effective_reserve_bytes < large.effective_reserve_bytes
    # An explicit figure still wins — that is what the field is for
    explicit = NodeSpec(node_id="e", hostname="e", gpu_class="x",
                        total_vram_bytes=96 * GB, reserved_bytes=2 * GB)
    assert explicit.effective_reserve_bytes == 2 * GB


def test_activation_is_capped_by_the_card_not_the_constant():
    spec = _spec("a", weight=1 * GB, ctx_floor=16384)
    assert planner.activation_bytes(spec, 96 * GB) == planner.ACTIVATION_BYTES
    assert planner.activation_bytes(spec, 24 * GB) < planner.ACTIVATION_BYTES
    # Unknown hardware keeps the conservative figure
    assert planner.activation_bytes(spec, None) == planner.ACTIVATION_BYTES


def test_priority_outranks_size_when_only_one_fits():
    """Largest-first is a starvation guard, not a ranking. On a cluster that
    cannot hold both, which model keeps the card is the operator's call."""
    big = _spec("big", weight=40 * GB, ctx_floor=16384)
    small = _spec("small", weight=30 * GB, ctx_floor=16384)
    node = [_node("n1", 96)]

    by_size = planner.plan([big, small], node)
    assert [p.model_id for p in by_size.placements] == ["big"]

    preferred = registry.replace(small, priority=10)
    by_priority = planner.plan([big, preferred], node)
    assert [p.model_id for p in by_priority.placements] == ["small"]
    assert by_priority.delegations[0].model_id == "big"


def test_priority_ties_still_seat_the_largest_first():
    """Within one level the packing guard has to survive, or small models seated
    first leave a big one nowhere to go."""
    big = registry.replace(_spec("big", weight=40 * GB, ctx_floor=16384), priority=5)
    small = registry.replace(_spec("small", weight=10 * GB, ctx_floor=16384), priority=5)
    result = planner.plan([small, big], [_node("n1", 60)])
    assert [p.model_id for p in result.placements] == ["big"]


def test_a_near_tie_does_not_move_a_running_model():
    """Two identical cards reported usable capacity 4 KiB apart. Under a strict
    maximum that was enough to migrate a model — a reload, and a window of paid
    OpenRouter fallback, bought with nothing."""
    spec = _spec("a", weight=10 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96), _node("n2", 96)]
    # n2 fractionally roomier, well inside the tie band
    nodes[1] = NodeSpec(**{**nodes[1].__dict__, "total_vram_bytes": 96 * GB + 4096})
    result = planner.plan([spec], nodes, deployed={"a": frozenset({"n1"})})
    assert result.placements[0].node_id == "n1"


def test_a_real_capacity_difference_still_moves_it():
    """Staying put is a tie-break, not a pin: a model that no longer fits where
    it sits has to move, or the plan is a wish."""
    spec = _spec("a", weight=10 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96), _node("n2", 96)]
    result = planner.plan(
        [spec], nodes,
        reserved={"n1": 80 * GB},
        deployed={"a": frozenset({"n1"})},
    )
    assert result.placements[0].node_id == "n2"


def test_the_plan_does_not_depend_on_node_order():
    """Nodes arrive in probe-completion order, so an unstable tie-break made the
    same inputs produce different plans from one run to the next."""
    specs = [_spec("big", weight=40 * GB, ctx_floor=16384),
             _spec("small", weight=4 * GB, ctx_floor=16384)]
    nodes = [_node("n1", 96), _node("n2", 96)]
    forward = planner.plan(specs, nodes)
    reverse = planner.plan(specs, list(reversed(nodes)))
    assert {(p.model_id, p.node_id) for p in forward.placements} == \
           {(p.model_id, p.node_id) for p in reverse.placements}


def test_models_spread_across_nodes():
    """Models are spread out: the node with the most capacity left wins."""
    specs = [_spec(name, weight=10 * GB, ctx_floor=16384) for name in ("a", "b", "c")]
    result = planner.plan(specs, [_node("n1", 96), _node("n2", 96), _node("n3", 96)])
    assert len({p.node_id for p in result.placements}) == 3


def test_gpu_util_stays_below_one():
    """At 1.0 vLLM swallows the driver's share too and dies during engine init."""
    spec = _spec("a", weight=40 * GB, ctx_floor=16384)
    result = planner.plan([spec], [_node("n1", 96)])
    assert 0 < result.placements[0].gpu_util <= planner.MAX_GPU_UTIL


def test_node_reserve_is_respected():
    """Placing past the reservation would claim the OS's share."""
    node = _node("n1", 48)
    spec = _spec("a", weight=20 * GB, ctx_floor=16384)
    result = planner.plan([spec], [node])
    assert result.placements[0].charge <= node.planner_vram_bytes


# ── registry ────────────────────────────────────────────────────────────


def _write_yaml(text: str) -> Path:
    path = Path(tempfile.mkdtemp()) / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_derived_defaults():
    path = _write_yaml("models:\n  - id: foo\n    hf_repo: org/Foo\n")
    spec = registry.load(path)[0]
    assert spec.dir == "foo"
    assert spec.service == "vllm-foo"
    assert spec.served_name == "local/foo"
    assert spec.port == registry.BASE_PORT


def test_written_values_beat_derived():
    path = _write_yaml(
        "models:\n  - id: foo\n    hf_repo: org/Foo\n"
        "    service: custom\n    port: 9999\n    env_prefix: VLLM_FOO\n"
    )
    spec = registry.load(path)[0]
    assert (spec.service, spec.port, spec.env_prefix) == ("custom", 9999, "VLLM_FOO")


def test_ctx_floor_derived_from_native():
    path = _write_yaml("models:\n  - id: foo\n    hf_repo: org/Foo\n")
    spec = registry.load(path)[0].bind(_meta(), 262144)
    assert spec.ctx_target == 262144
    assert spec.ctx_floor == 262144 // registry.CTX_FLOOR_DIVISOR


def test_unknown_model_id_is_an_error():
    """Swallowing a typo leaves nobody able to explain the missing model."""
    path = _write_yaml("models:\n  - id: foo\n    hf_repo: org/Foo\n")
    try:
        registry.load(path, only=["typo"])
    except KeyError as exc:
        assert "typo" in str(exc)
    else:
        raise AssertionError("an undefined id must raise KeyError")


def test_size_suffixes():
    assert registry.parse_size("1GiB") == GB
    assert registry.parse_size(1024) == 1024


# ── applier ─────────────────────────────────────────────────────────────


def test_no_change_when_already_converged():
    """Applying twice must be a no-op the second time (the URLs are recorded)."""
    spec = _spec("a", weight=20 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96)]
    result = planner.plan([spec], nodes)

    change = applier.compute_diff(
        target=result, current={"n1": {spec.service}}, specs=[spec], nodes=nodes,
    )
    assert not any(a.kind == "start" for a in change.actions)


def test_stops_services_no_longer_planned():
    spec = _spec("a", weight=20 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96)]
    result = planner.plan([spec], nodes)
    change = applier.compute_diff(
        target=result, current={"n1": {spec.service, "vllm-gone"}},
        specs=[spec], nodes=nodes,
    )
    assert any(a.kind == "stop" and "vllm-gone" in a.description for a in change.actions)


def test_does_not_stop_the_nodes_transcription_backend():
    """The transcription backend is resident on the same node and compose file.

    ``current`` is the node's whole ``docker ps``, so whisper appears in it.
    Stopping it because the plan does not list it would take that node's STT down,
    and for the same reason a single-host deployment's stack containers must be
    left alone.
    """
    spec = _spec("a", weight=20 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96)]
    result = planner.plan([spec], nodes)
    change = applier.compute_diff(
        target=result,
        current={"n1": {spec.service, "whisper", "kloudchat-gateway"}},
        specs=[spec], nodes=nodes,
    )
    stopped = [a.description for a in change.actions if a.kind == "stop"]
    assert not stopped, f"nothing should be stopped, got: {stopped}"


def test_whisper_urls_hold_only_probed_backends():
    """A node with no transcription backend (arm64) must not appear in WHISPER_URLS.

    Deriving the list mechanically from the node list would include it, and a
    non-empty value stops gen-litellm-config from registering the OpenRouter STT
    fallback. That is the path by which the microphone disappears.
    """
    spec = _spec("a", weight=20 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96), _node("n2", 96, arch="arm64")]
    result = planner.plan([spec], nodes)

    urls = applier._url_csvs(result, [spec], nodes, ["user@n1"])
    assert urls["WHISPER_URLS"] == "http://n1:9000"

    # No backend answered means an empty value, which is the OpenRouter switch
    assert applier._url_csvs(result, [spec], nodes, [])["WHISPER_URLS"] == ""


def test_url_csv_written_for_placed_models():
    spec = _spec("a", weight=20 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96)]
    result = planner.plan([spec], nodes)
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
        f.write("VLLM_A_URL=\n")
        env_path = f.name
    change = applier.compute_diff(
        target=result, current={}, specs=[spec], nodes=nodes, local_env_path=env_path,
    )
    assert change.local_env["VLLM_A_URL"] == "http://n1:8001"


def test_re_applying_an_unchanged_plan_does_nothing():
    """`setup.sh all` runs apply every time. Force-recreating regardless is a
    full weight reload — twenty minutes for a 78 GiB model, for nothing."""
    spec = _spec("a", weight=20 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96)]
    result = planner.plan([spec], nodes)
    placement = result.placements[0]
    settled = {
        "n1": {
            "VLLM_A_MAX_LEN": str(placement.ctx),
            "VLLM_A_GPU_UTIL": f"{placement.gpu_util:.2f}",
        }
    }
    change = applier.compute_diff(
        target=result, current={"n1": {"vllm-a"}}, specs=[spec], nodes=nodes,
        node_env=settled,
    )
    assert change.actions == []


def test_an_unsharded_model_does_not_get_a_tp_line():
    """TP 1 is the absence of sharding and compose defaults to it. Writing it
    into a node that never had the key would force-recreate — twenty minutes of
    weight loading — to restate a default."""
    spec = _spec("a", weight=20 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96)]
    result = planner.plan([spec], nodes)
    placement = result.placements[0]
    settled = {"n1": {"VLLM_A_MAX_LEN": str(placement.ctx),
                      "VLLM_A_GPU_UTIL": f"{placement.gpu_util:.2f}"}}
    assert applier.compute_diff(
        target=result, current={"n1": {"vllm-a"}}, specs=[spec], nodes=nodes,
        node_env=settled,
    ).actions == []

    # But a node that really is sharded has to be told to stop
    was_sharded = {**settled["n1"], "VLLM_A_TP": "2"}
    change = applier.compute_diff(
        target=result, current={"n1": {"vllm-a"}}, specs=[spec], nodes=nodes,
        node_env={"n1": was_sharded},
    )
    assert any(a.description == "VLLM_A_TP=1" for a in change.actions)
    assert any(a.kind == "recreate" for a in change.actions)


def test_a_changed_option_still_recreates():
    """Idempotence must not become inertia: a new context has to reach vLLM."""
    spec = _spec("a", weight=20 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96)]
    result = planner.plan([spec], nodes)
    stale = {"n1": {"VLLM_A_MAX_LEN": "8192", "VLLM_A_GPU_UTIL": "0.99"}}
    change = applier.compute_diff(
        target=result, current={"n1": {"vllm-a"}}, specs=[spec], nodes=nodes,
        node_env=stale,
    )
    assert [a.kind for a in change.actions].count("recreate") == 1


def test_an_unreadable_node_env_recreates_rather_than_assuming():
    """A node whose .env could not be read is not a node known to be current."""
    spec = _spec("a", weight=20 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96)]
    result = planner.plan([spec], nodes)
    change = applier.compute_diff(
        target=result, current={"n1": {"vllm-a"}}, specs=[spec], nodes=nodes,
        node_env={"n1": {}},
    )
    assert any(a.kind == "recreate" for a in change.actions)


def test_dropping_a_model_clears_its_url():
    """A model removed from VLLM_MODELS is no longer in `specs`, so nothing used
    to touch its URL — and gen-litellm-config went on registering a route to the
    container the same apply had just stopped."""
    kept = _spec("a", weight=20 * GB, ctx_floor=16384)
    dropped = _spec("b", weight=20 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96)]
    result = planner.plan([kept], nodes)
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
        f.write("VLLM_A_URL=\nVLLM_B_URL=http://n1:8002\n")
        env_path = f.name
    change = applier.compute_diff(
        target=result, current={}, specs=[kept], nodes=nodes,
        local_env_path=env_path, known=[kept, dropped],
    )
    assert change.local_env["VLLM_A_URL"] == "http://n1:8001"
    assert change.local_env["VLLM_B_URL"] == ""


def test_one_node_failing_does_not_stop_the_rest():
    spec = _spec("a", weight=20 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96), _node("n2", 96)]
    result = planner.plan([spec, _spec("b", weight=20 * GB, ctx_floor=16384)], nodes)
    change = applier.compute_diff(target=result, current={}, specs=[spec], nodes=nodes)

    attempted: list[str] = []

    def runner(host: str, command: str) -> tuple[int, str]:
        attempted.append(host)
        return (1, "boom") if "n1" in host else (0, "")

    failures = applier.apply(change, runner=runner)
    assert failures and len({h for h in attempted}) == len(
        {a.host for a in change.actions})


def test_env_write_updates_in_place():
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
        f.write("KEEP=1\nVLLM_A_URL=old\n")
        path = f.name
    applier._write_local_env(path, {"VLLM_A_URL": "new", "VLLM_B_URL": "added"})
    text = Path(path).read_text()
    assert "KEEP=1" in text and "VLLM_A_URL=new" in text and "VLLM_B_URL=added" in text
    assert "old" not in text


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as exc:  # noqa: BLE001 — the harness tallies after every test runs
            failed += 1
            print(f"  FAIL {name}: {exc}")
    print(f"\n{str(failed) + ' failed' if failed else 'all passed'}")
    sys.exit(1 if failed else 0)
