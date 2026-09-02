"""Scheduler tests: memory arithmetic and placement policy. Runs with or without pytest."""

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
          placement: str = "any", share: float = 1.0,
          **kw) -> registry.ModelSpec:
    spec = registry.ModelSpec(
        id=model_id, hf_repo=f"org/{model_id}", dir=model_id,
        service=f"vllm-{model_id}", port=8001,
        env_prefix=f"VLLM_{model_id.upper()}", served_name=f"local/{model_id}",
        ctx_floor=ctx_floor, concurrent_sessions=1, arches=arches,
        priority=priority, placement=placement, share=share,
    )
    return spec.bind(_meta(weight_bytes=weight, **kw), native)


def _node(node_id: str, gib: int, arch: str = "amd64",
          checkpoints=None) -> NodeSpec:
    return NodeSpec(node_id=node_id, hostname=f"user@{node_id}",
                    gpu_class="pro6000", total_vram_bytes=gib * GB, arch=arch,
                    checkpoints=checkpoints)


# ── memory arithmetic ───────────────────────────────────────────────────


def test_mla_kv_is_an_order_smaller_than_mha():
    """MLA KV per token is far below the MHA figure for the same layer count."""
    mha = kv_bytes_per_token(_meta(n_layers=47, n_kv_heads=20), Dtype.FP8)
    mla = kv_bytes_per_token(_meta(n_layers=47, n_kv_heads=20, kv_latent_dim=576), Dtype.FP8)
    assert mla * 5 < mha, f"MLA {mla} should be well below MHA {mha}"


def test_sliding_window_layers_cost_per_sequence_not_per_token():
    """Sliding-window layers cost a flat amount per sequence, independent of context."""
    from scheduler.kv_model import sliding_bytes_per_sequence

    md = _meta(n_layers=5, n_kv_heads=8, head_dim=256,
               sliding_layers=25, sliding_window=1024)
    per_seq = sliding_bytes_per_sequence(md, Dtype.FP8)
    assert per_seq == 25 * 2 * 8 * 256 * 1024

    # Flat per sequence
    assert per_seq == sliding_bytes_per_sequence(md, Dtype.FP8)
    assert sliding_bytes_per_sequence(_meta(), Dtype.FP8) == 0


def test_sliding_layers_are_read_from_the_config():
    from scheduler import model_metadata

    assert model_metadata._count_sliding_layers({
        "layer_types": ["sliding_attention"] * 25 + ["full_attention"] * 5,
        "sliding_window": 1024,
    }) == (25, 1024)
    assert model_metadata._count_sliding_layers(
        {"layer_types": ["sliding_attention"] * 25}
    ) == (0, 0)


def test_a_hybrid_written_as_a_stride_is_still_a_hybrid():
    """A `full_attention_interval` stride counts as hybrid only alongside linear-attention keys."""
    from scheduler import model_metadata

    stride = {
        "num_hidden_layers": 48, "full_attention_interval": 4,
        "linear_key_head_dim": 128, "num_key_value_heads": 2, "head_dim": 256,
    }
    assert model_metadata._count_kv_bearing_layers(stride) == 12
    assert model_metadata._count_kv_bearing_layers(
        {"num_hidden_layers": 48, "full_attention_interval": 4}
    ) == 48


def test_an_encoder_decoder_config_is_read():
    """Encoder-decoder configs (whisper) resolve context, head width, KV heads and layers from the decoder keys."""
    from scheduler import model_metadata

    cfg = {
        "architectures": ["WhisperForConditionalGeneration"],
        "d_model": 1280, "decoder_attention_heads": 20, "decoder_layers": 32,
        "encoder_attention_heads": 20, "encoder_layers": 32,
        "max_source_positions": 1500, "max_target_positions": 448,
        "torch_dtype": "float16", "vocab_size": 51866,
    }
    assert model_metadata._resolve_native_ctx(cfg) == 448
    assert model_metadata._resolve_head_dim(cfg) == 64
    assert model_metadata._resolve_kv_heads(cfg) == 20
    assert model_metadata._count_kv_bearing_layers(cfg) == 32

    plain = {"hidden_size": 4096, "num_attention_heads": 32,
             "num_hidden_layers": 40, "max_position_embeddings": 131072}
    assert model_metadata._resolve_native_ctx(plain) == 131072
    assert model_metadata._resolve_head_dim(plain) == 128


def test_kv_bearing_layers_drive_cost():
    """KV cost scales with KV-bearing layers only."""
    full = kv_bytes_per_token(_meta(n_layers=40), Dtype.FP8)
    hybrid = kv_bytes_per_token(_meta(n_layers=10), Dtype.FP8)
    assert hybrid * 4 == full


def test_fp8_kv_halves_bf16():
    assert (kv_bytes_per_token(_meta(), Dtype.FP8) * 2
            == kv_bytes_per_token(_meta(), Dtype.BF16))


# ── placement policy ────────────────────────────────────────────────────


def test_every_model_placed_before_any_replica():
    """Every model gets one instance before any model gets a second."""
    specs = [_spec("a", weight=20 * GB), _spec("b", weight=20 * GB)]
    result = planner.plan(specs, [_node("n1", 96), _node("n2", 96)])
    counts = Counter(p.model_id for p in result.placements)
    assert set(counts) == {"a", "b"}, counts
    assert max(counts.values()) - min(counts.values()) <= 1, counts
    assert not result.delegations


def test_spare_capacity_is_filled_unless_capped():
    """Spare capacity takes replicas; `replicas` caps them and 1 disables."""
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
    """Replicas go to the highest-priority model first."""
    def specs():
        return [
            _spec("third", weight=20 * GB, priority=1),
            _spec("first", weight=20 * GB, priority=3),
            _spec("second", weight=20 * GB, priority=2),
        ]

    four = [_node(f"n{i}", 32) for i in range(1, 5)]
    ids = [p.model_id for p in planner.plan(specs(), four, replicas=2).placements]
    assert ids.count("first") == 2, ids
    assert ids.count("second") == 1 and ids.count("third") == 1, ids

    five = four + [_node("n5", 32)]
    ids = [p.model_id for p in planner.plan(specs(), five, replicas=2).placements]
    assert ids.count("first") == 2 and ids.count("second") == 2, ids
    assert ids.count("third") == 1, ids


def test_a_model_is_not_placed_where_its_weights_are_not():
    """A model whose checkpoint no node holds is delegated with a weights reason, not a capacity one."""
    spec = _spec("a", weight=20 * GB)
    nodes = [_node("n1", 96, checkpoints=frozenset({"something-else"}))]
    result = planner.plan([spec], nodes)
    assert not result.placements
    assert len(result.delegations) == 1
    reason = result.delegations[0].reason
    assert "checkpoint" in reason and "'a'" in reason, reason


def test_the_capacity_reason_only_counts_nodes_that_could_run_it():
    """A capacity reason does not quote free VRAM on a node that lacks the checkpoint."""
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
    """Replicas are seated only on nodes that hold the checkpoint."""
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
    """`checkpoints=None` filters nothing."""
    spec = _spec("a", weight=20 * GB)
    result = planner.plan([spec], [_node("n1", 96)])
    assert [p.node_id for p in result.placements] == ["n1"]


def test_context_restored_above_floor():
    """Leftover capacity raises the context above the floor, up to the target."""
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
    """An unsupported architecture is reported as such, not as missing capacity."""
    spec = _spec("a", weight=1 * GB, arches=("amd64",))
    result = planner.plan([spec], [_node("gb10", 128, arch="arm64")])
    assert not result.placements
    assert "architecture" in result.delegations[0].reason


def test_reservation_shrinks_capacity():
    # Fits the bare node (48 GiB less the 8 GiB reserve), not once 20 GiB is held
    node = [_node("n1", 48)]
    weight = 38 * GB - planner.ACTIVATION_BYTES
    spec = _spec("a", weight=weight, ctx_floor=16384)
    assert planner.plan([spec], node).placements
    assert not planner.plan([spec], node, reserved={"n1": 20 * GB}).placements


def test_a_pooling_model_is_not_charged_decode_headroom():
    """A pooling runner is charged 2 GiB of activation and no KV."""
    weight = 4 * GB
    generate = _spec("gen", weight=weight, ctx_floor=8192)
    pooling = registry.replace(generate, runner="pooling")
    assert planner.need_bytes(pooling, 8192) < planner.need_bytes(generate, 8192)
    assert planner.need_bytes(pooling, 8192) == weight + planner.POOLING_ACTIVATION_BYTES


def test_tensor_parallel_splits_the_weights_but_not_the_activation():
    """Per card: weights/N and KV/N, plus the full activation cost."""
    spec = registry.replace(
        _spec("a", weight=80 * GB, ctx_floor=16384), tensor_parallel=2
    )
    per_gpu = planner.per_gpu_need_bytes(spec, 16384)
    kv = planner.kv_bytes(spec, 16384)
    assert per_gpu == 40 * GB + planner.ACTIVATION_BYTES + kv // 2
    # Node-wide: activation paid twice
    assert planner.need_bytes(spec, 16384) > 80 * GB + planner.ACTIVATION_BYTES + kv


def test_a_model_too_big_for_one_card_fits_across_two():
    """A model too big for one card places at TP 2 across two, with gpu_util per card."""
    spec = _spec("big", weight=78 * GB, ctx_floor=16384)
    one_card = NodeSpec(node_id="n1", hostname="n1", gpu_class="pro6000",
                        total_vram_bytes=89 * GB, gpu_count=1, arch="amd64")
    two_cards = NodeSpec(node_id="n2", hostname="n2", gpu_class="pro6000",
                         total_vram_bytes=89 * GB, gpu_count=2, arch="amd64")

    assert not planner.plan([spec], [one_card]).placements
    sharded = registry.replace(spec, tensor_parallel=2)
    placed = planner.plan([sharded], [two_cards]).placements
    assert placed and placed[0].tp == 2
    assert placed[0].gpu_util < 1.0


def test_a_reserved_workload_does_not_make_sharding_impossible():
    """Per-card capacity is derived from node capacity after reservations, so sharded models still place."""
    spec = registry.replace(_spec("a", weight=40 * GB, ctx_floor=16384),
                            tensor_parallel=2)
    node = NodeSpec(node_id="n1", hostname="n1", gpu_class="pro6000",
                    total_vram_bytes=89 * GB, gpu_count=2, arch="amd64")
    result = planner.plan([spec], [node], reserved={"n1": 6 * GB})
    assert result.placements, result.delegations[0].reason
    assert result.placements[0].tp == 2


def test_tensor_parallel_needs_the_cards_and_says_so():
    """TP wider than the node reports missing cards, not missing memory."""
    spec = registry.replace(_spec("a", weight=10 * GB, ctx_floor=16384),
                            tensor_parallel=4)
    result = planner.plan([spec], [_node("n1", 96)])
    assert not result.placements
    assert "cards" in result.delegations[0].reason


def test_two_models_on_a_two_card_node_get_a_card_each():
    """Two models on a two-card node take different cards."""
    a = _spec("a", weight=30 * GB, ctx_floor=16384)
    b = _spec("b", weight=30 * GB, ctx_floor=16384)
    node = NodeSpec(node_id="n1", hostname="n1", gpu_class="pro5000",
                    total_vram_bytes=48 * GB, gpu_count=2, arch="amd64")
    result = planner.plan([a, b], [node])
    assert len(result.placements) == 2
    assert {p.devices for p in result.placements} == {(0,), (1,)}


def test_no_card_is_oversubscribed():
    """gpu_util fractions sharing a card sum to at most 1.0."""
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
    """`kv_shards` is min(TP, KV heads), and 1 for MLA."""
    gqa = registry.replace(_spec("gqa", weight=10 * GB, ctx_floor=16384,
                                 n_kv_heads=2), tensor_parallel=4)
    assert planner.kv_shards(gqa) == 2

    mla = registry.replace(_spec("mla", weight=10 * GB, ctx_floor=16384,
                                 kv_latent_dim=576), tensor_parallel=4)
    assert planner.kv_shards(mla) == 1


# ── node roles ──────────────────────────────────────────────────────────


def test_a_pool_model_leaves_the_head_node_alone():
    """A pool model is not placed on the head node even when it is roomier."""
    spec = _spec("big", weight=20 * GB, ctx_floor=16384, placement="pool")
    result = planner.plan([spec], [_node("head", 96), _node("n2", 64)], head="head")
    assert [p.node_id for p in result.placements] == ["n2"]


def test_a_head_model_does_not_follow_the_room_into_the_pool():
    spec = _spec("floor", weight=20 * GB, ctx_floor=16384, placement="head")
    result = planner.plan([spec], [_node("head", 48), _node("n2", 96)], head="head")
    assert [p.node_id for p in result.placements] == ["head"]


def test_a_full_pool_delegates_rather_than_spilling_onto_the_head():
    """A pool model with no pool seat is delegated, with a capacity reason measured over the pool."""
    resident = _spec("resident", weight=40 * GB, ctx_floor=16384, placement="pool")
    arrival = _spec("arrival", weight=40 * GB, ctx_floor=16384, placement="pool")
    result = planner.plan([resident, arrival],
                          [_node("head", 96), _node("n2", 96)], head="head")
    assert [p.node_id for p in result.placements] == ["n2"]
    assert [d.model_id for d in result.delegations] == ["arrival"]
    assert "GiB" in result.delegations[0].reason


def test_one_declared_node_has_no_pool_to_be_kept_out_of():
    """`head=None` places pool models on the only node."""
    spec = _spec("big", weight=20 * GB, ctx_floor=16384, placement="pool")
    result = planner.plan([spec], [_node("only", 96)], head=None)
    assert [p.node_id for p in result.placements] == ["only"]


def test_the_head_is_named_rather_than_taken_from_the_node_order():
    """Head placement follows the `head` argument, not node order."""
    def specs():
        return [_spec("floor", weight=20 * GB, ctx_floor=16384, placement="head"),
                _spec("top", weight=20 * GB, ctx_floor=16384, placement="pool")]

    nodes = [_node("head", 96), _node("n2", 96)]
    forward = planner.plan(specs(), nodes, head="head")
    reverse = planner.plan(specs(), list(reversed(nodes)), head="head")
    assert {(p.model_id, p.node_id) for p in forward.placements} == \
           {(p.model_id, p.node_id) for p in reverse.placements}
    assert {(p.model_id, p.node_id) for p in forward.placements} == \
           {("floor", "head"), ("top", "n2")}


def test_a_pool_model_with_no_pool_node_answering_says_so():
    """No answering pool node yields a pool reason, not a capacity one."""
    spec = _spec("top", weight=20 * GB, ctx_floor=16384, placement="pool")
    result = planner.plan([spec], [_node("head", 96)], head="head")
    assert not result.placements
    assert "pool" in result.delegations[0].reason
    assert "GiB" not in result.delegations[0].reason


def test_extra_instances_follow_the_declared_share():
    """Shares 60/40 over five pool nodes give three and two instances."""
    def specs():
        return [_spec("top", weight=20 * GB, placement="pool", share=60),
                _spec("coder", weight=20 * GB, placement="pool", share=40)]

    nodes = [_node("head", 32)] + [_node(f"n{i}", 32) for i in range(2, 7)]
    result = planner.plan(specs(), nodes, head="head")
    counts = Counter(p.model_id for p in result.placements)
    assert counts == {"top": 3, "coder": 2}, counts
    assert "head" not in {p.node_id for p in result.placements}


def test_equal_shares_still_deepen_by_fewest_instances():
    """Undeclared shares keep instance counts level."""
    specs = [_spec("a", weight=20 * GB), _spec("b", weight=20 * GB)]
    result = planner.plan(specs, [_node(f"n{i}", 32) for i in range(1, 5)])
    counts = Counter(p.model_id for p in result.placements)
    assert counts == {"a": 2, "b": 2}, counts


# ── across card sizes ───────────────────────────────────────────────────

CARD_SIZES = [24, 32, 48, 80, 96]


def _narrow_kv_spec(*, sessions: int, weight: int = 21 * GB):
    """A 21 GiB model at 10 KiB of KV per token."""
    spec = _spec("a", weight=weight, ctx_floor=131072,
                 n_layers=10, n_kv_heads=2, head_dim=256)
    return registry.replace(spec, concurrent_sessions=sessions)


# Looped, not parametrized: the module also runs without pytest
def test_headroom_never_eats_the_card():
    """Reserve plus activation stays under 45% of every card size."""
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
    """A model whose weights take half the card places on every card size."""
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
    """A tight card halves `concurrent_sessions`, keeps the context floor, and notes it."""
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
    """`_classify_gpu_name` uses lib.sh::detect_gpu_class vocabulary, with "unknown" for an empty probe."""
    from scheduler import inventory

    assert inventory._classify_gpu_name("NVIDIA GB10") == "gb10"
    assert inventory._classify_gpu_name("NVIDIA RTX PRO 6000 Blackwell") == "pro6000"
    assert inventory._classify_gpu_name("NVIDIA A100-SXM4-80GB") == "nvidia-other"
    assert inventory._classify_gpu_name("") == "unknown"


def test_a_mixed_box_is_sized_by_its_smallest_card():
    """Per-card capacity on a mixed box is the smallest card."""
    from scheduler import inventory

    sizes = (24 * GB, 32 * GB)
    assert min(sizes) * len(sizes) < sum(sizes)
    node = NodeSpec(node_id="n1", hostname="n1", gpu_class="mixed",
                    total_vram_bytes=min(sizes), gpu_count=len(sizes))
    assert node.per_gpu_planner_bytes <= 24 * GB
    assert inventory.MANAGED_PREFIX == "vllm-"


def test_memory_someone_else_holds_is_not_offered():
    """Foreign GPU memory is subtracted from planner capacity and changes placement."""
    clean = _node("n1", 48)
    busy = NodeSpec(node_id="n2", hostname="n2", gpu_class="pro5000",
                    total_vram_bytes=48 * GB, foreign_vram_bytes=10 * GB)
    assert busy.planner_vram_bytes == clean.planner_vram_bytes - 10 * GB

    spec = _spec("a", weight=30 * GB, ctx_floor=16384)
    assert planner.plan([spec], [clean]).placements
    assert not planner.plan([spec], [busy]).placements


def test_the_reserve_scales_with_the_card():
    small, large = _node("s", 24), _node("l", 96)
    assert small.effective_reserve_bytes < large.effective_reserve_bytes
    # An explicit figure wins
    explicit = NodeSpec(node_id="e", hostname="e", gpu_class="x",
                        total_vram_bytes=96 * GB, reserved_bytes=2 * GB)
    assert explicit.effective_reserve_bytes == 2 * GB


def test_activation_is_capped_by_the_card_not_the_constant():
    spec = _spec("a", weight=1 * GB, ctx_floor=16384)
    assert planner.activation_bytes(spec, 96 * GB) == planner.ACTIVATION_BYTES
    assert planner.activation_bytes(spec, 24 * GB) < planner.ACTIVATION_BYTES
    assert planner.activation_bytes(spec, None) == planner.ACTIVATION_BYTES


def test_priority_outranks_size_when_only_one_fits():
    """`priority` decides which model keeps the card; size is only the tiebreak."""
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
    """Equal priority seats the largest model first."""
    big = registry.replace(_spec("big", weight=40 * GB, ctx_floor=16384), priority=5)
    small = registry.replace(_spec("small", weight=10 * GB, ctx_floor=16384), priority=5)
    result = planner.plan([small, big], [_node("n1", 60)])
    assert [p.model_id for p in result.placements] == ["big"]


def test_a_near_tie_does_not_move_a_running_model():
    """Within the tie band the incumbent node wins."""
    spec = _spec("a", weight=10 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96), _node("n2", 96)]
    # n2 fractionally roomier, inside the tie band
    nodes[1] = NodeSpec(**{**nodes[1].__dict__, "total_vram_bytes": 96 * GB + 4096})
    result = planner.plan([spec], nodes, deployed={"a": frozenset({"n1"})})
    assert result.placements[0].node_id == "n1"


def test_a_real_capacity_difference_still_moves_it():
    """A model that no longer fits its incumbent node moves."""
    spec = _spec("a", weight=10 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96), _node("n2", 96)]
    result = planner.plan(
        [spec], nodes,
        reserved={"n1": 80 * GB},
        deployed={"a": frozenset({"n1"})},
    )
    assert result.placements[0].node_id == "n2"


def test_the_plan_does_not_depend_on_node_order():
    """The same inputs give the same plan regardless of node order."""
    specs = [_spec("big", weight=40 * GB, ctx_floor=16384),
             _spec("small", weight=4 * GB, ctx_floor=16384)]
    nodes = [_node("n1", 96), _node("n2", 96)]
    forward = planner.plan(specs, nodes)
    reverse = planner.plan(specs, list(reversed(nodes)))
    assert {(p.model_id, p.node_id) for p in forward.placements} == \
           {(p.model_id, p.node_id) for p in reverse.placements}


def test_models_spread_across_nodes():
    """Worst fit spreads models across nodes."""
    specs = [_spec(name, weight=10 * GB, ctx_floor=16384) for name in ("a", "b", "c")]
    result = planner.plan(specs, [_node("n1", 96), _node("n2", 96), _node("n3", 96)])
    assert len({p.node_id for p in result.placements}) == 3


def test_gpu_util_stays_below_one():
    """gpu_util never exceeds MAX_GPU_UTIL."""
    spec = _spec("a", weight=40 * GB, ctx_floor=16384)
    result = planner.plan([spec], [_node("n1", 96)])
    assert 0 < result.placements[0].gpu_util <= planner.MAX_GPU_UTIL


def test_node_reserve_is_respected():
    """A placement never exceeds the node reserve."""
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
    """An id in `only` that models.yaml lacks raises KeyError."""
    path = _write_yaml("models:\n  - id: foo\n    hf_repo: org/Foo\n")
    try:
        registry.load(path, only=["typo"])
    except KeyError as exc:
        assert "typo" in str(exc)
    else:
        raise AssertionError("an undefined id must raise KeyError")


def test_placement_and_share_are_declared_values():
    path = _write_yaml(
        "models:\n  - id: foo\n    hf_repo: org/Foo\n"
        "    placement: pool\n    share: 60\n"
        "  - id: bar\n    hf_repo: org/Bar\n"
    )
    pool, plain = registry.load(path)
    assert (pool.placement, pool.share) == ("pool", 60.0)
    assert (plain.placement, plain.share) == ("any", 1.0)


def test_an_unknown_placement_is_an_error():
    """An unknown `placement` raises ValueError."""
    path = _write_yaml("models:\n  - id: foo\n    hf_repo: org/Foo\n    placement: haed\n")
    try:
        registry.load(path)
    except ValueError as exc:
        assert "haed" in str(exc)
    else:
        raise AssertionError("an undefined placement must raise ValueError")


def test_an_unknown_runner_is_an_error():
    """An unknown `runner` raises ValueError."""
    path = _write_yaml("models:\n  - id: foo\n    hf_repo: org/Foo\n    runner: polling\n")
    try:
        registry.load(path)
    except ValueError as exc:
        assert "polling" in str(exc)
    else:
        raise AssertionError("an undefined runner must raise ValueError")


def test_transcription_is_sized_like_a_generate_model():
    """A transcription runner is not pooling."""
    path = _write_yaml(
        "models:\n  - id: foo\n    hf_repo: org/Foo\n    runner: transcription\n"
    )
    spec = registry.load(path)[0]
    assert spec.runner == "transcription"
    assert not spec.is_pooling


def test_a_share_of_zero_is_an_error():
    """A share of 0 or below raises ValueError."""
    for value in ("0", "-1"):
        path = _write_yaml(
            f"models:\n  - id: foo\n    hf_repo: org/Foo\n    share: {value}\n"
        )
        try:
            registry.load(path)
        except ValueError as exc:
            assert "share" in str(exc)
        else:
            raise AssertionError(f"share: {value} must raise ValueError")


def test_size_suffixes():
    assert registry.parse_size("1GiB") == GB
    assert registry.parse_size(1024) == 1024


# ── applier ─────────────────────────────────────────────────────────────


def test_no_change_when_already_converged():
    """A converged node gets no start action."""
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


def test_leaves_containers_this_stack_does_not_place_alone():
    """Containers outside the `vllm-` prefix are never stopped."""
    spec = _spec("a", weight=20 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96)]
    result = planner.plan([spec], nodes)
    change = applier.compute_diff(
        target=result,
        current={"n1": {spec.service, "kloudchat-gateway", "whisper-shim"}},
        specs=[spec], nodes=nodes,
    )
    stopped = [a.description for a in change.actions if a.kind == "stop"]
    assert not stopped, f"nothing should be stopped, got: {stopped}"


def _stt_spec(**kw):
    return _spec(applier.STT_MODEL_ID, weight=4 * GB, ctx_floor=448,
                 native=448, placement="head", **kw)


def test_whisper_urls_follow_the_transcription_placement():
    """WHISPER_URLS is the transcription model's URL, and its `{env_prefix}_URL` is not written."""
    stt = _stt_spec()
    nodes = [_node("n1", 96), _node("n2", 96)]
    result = planner.plan([stt], nodes, replicas=1, head="n1")

    urls = applier._url_csvs(result, [stt], nodes)
    assert urls["WHISPER_URLS"] == f"http://n1:{stt.port}"
    assert f"{stt.env_prefix}_URL" not in urls


def test_whisper_urls_empty_when_transcription_is_delegated():
    """An unplaced transcription model clears WHISPER_URLS."""
    stt = _stt_spec()
    other = _spec("a", weight=20 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96)]
    result = planner.plan([other], nodes)
    assert applier._url_csvs(result, [other], nodes, [stt])["WHISPER_URLS"] == ""


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
    """A node whose .env already matches the plan gets no actions."""
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
    """TP 1 is written only to undo a node that has TP set."""
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

    # A sharded node is told to stop
    was_sharded = {**settled["n1"], "VLLM_A_TP": "2"}
    change = applier.compute_diff(
        target=result, current={"n1": {"vllm-a"}}, specs=[spec], nodes=nodes,
        node_env={"n1": was_sharded},
    )
    assert any(a.description == "VLLM_A_TP=1" for a in change.actions)
    assert any(a.kind == "recreate" for a in change.actions)


def test_a_changed_option_still_recreates():
    """A changed option recreates the service exactly once."""
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
    """An empty node .env recreates the service."""
    spec = _spec("a", weight=20 * GB, ctx_floor=16384)
    nodes = [_node("n1", 96)]
    result = planner.plan([spec], nodes)
    change = applier.compute_diff(
        target=result, current={"n1": {"vllm-a"}}, specs=[spec], nodes=nodes,
        node_env={"n1": {}},
    )
    assert any(a.kind == "recreate" for a in change.actions)


def test_dropping_a_model_clears_its_url():
    """A model in `known` but not in `specs` has its URL cleared."""
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
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {exc}")
    print(f"\n{str(failed) + ' failed' if failed else 'all passed'}")
    sys.exit(1 if failed else 0)
