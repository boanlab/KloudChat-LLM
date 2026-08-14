"""Scheduler tests: the memory arithmetic and the placement policy.

    pytest scheduler/tests -q
    PYTHONPATH=. python3 scheduler/tests/test_scheduler.py   # without pytest
"""

from __future__ import annotations

import tempfile
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
          native: int = 131072, arches=(), **kw) -> registry.ModelSpec:
    spec = registry.ModelSpec(
        id=model_id, hf_repo=f"org/{model_id}", dir=model_id,
        service=f"vllm-{model_id}", port=8001,
        env_prefix=f"VLLM_{model_id.upper()}", served_name=f"local/{model_id}",
        ctx_floor=ctx_floor, concurrent_sessions=1, or_twin=None, arches=arches,
    )
    return spec.bind(_meta(weight_bytes=weight, **kw), native)


def _node(node_id: str, gib: int, arch: str = "amd64") -> NodeSpec:
    return NodeSpec(node_id=node_id, hostname=f"user@{node_id}",
                    gpu_class="pro6000", total_vram_bytes=gib * GB, arch=arch)


# ── memory arithmetic ───────────────────────────────────────────────────


def test_mla_kv_is_an_order_smaller_than_mha():
    """Sizing MLA with the MHA formula overestimates by an order of magnitude,
    which rejects placements that would have fit."""
    mha = kv_bytes_per_token(_meta(n_layers=47, n_kv_heads=20), Dtype.FP8)
    mla = kv_bytes_per_token(_meta(n_layers=47, n_kv_heads=20, kv_latent_dim=576), Dtype.FP8)
    assert mla * 5 < mha, f"MLA {mla} should be well below MHA {mha}"


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
    """A spare node does not get a second copy of the same model: coverage first."""
    specs = [_spec("a", weight=20 * GB), _spec("b", weight=20 * GB)]
    result = planner.plan(specs, [_node("n1", 96), _node("n2", 96)])
    placed = [p.model_id for p in result.placements]
    assert sorted(placed) == ["a", "b"], placed
    assert not result.delegations


def test_replicas_only_when_requested():
    specs = [_spec("a", weight=20 * GB), _spec("b", weight=20 * GB)]
    nodes = [_node("n1", 96), _node("n2", 96), _node("n3", 96)]

    once = planner.plan(specs, nodes)
    assert len(once.placements) == 2

    twice = planner.plan(specs, nodes, replicas=2)
    ids = [p.model_id for p in twice.placements]
    assert ids.count("a") == 2 and ids.count("b") == 2, ids


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
    spec = _spec("a", weight=30 * GB, ctx_floor=16384)
    node = [_node("n1", 48)]
    assert planner.plan([spec], node).placements
    assert not planner.plan([spec], node, reserved={"n1": 20 * GB}).placements


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
