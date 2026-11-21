"""Picture models (vllm-omni): catalogue, sizing, pinning, and what `apply` does for them."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from scheduler import applier, inventory, model_metadata, planner, registry
from scheduler.types import GB, NodeSpec

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = ROOT / "scheduler" / "models.yaml"
IMAGES = ("qwen-image-edit-2511", "qwen-image-2512", "z-image-turbo")


def _images(**site_fields) -> list[registry.ModelSpec]:
    specs = [s for s in registry.load(CATALOGUE) if s.is_image]
    return [
        replace(s, **site_fields.get(s.id, {})).bind(
            model_metadata.image_metadata(s.id, s.weight_bytes), 0)
        for s in specs
    ]


def _node(node_id: str, cards: int = 2, gib: int = 96, **kw) -> NodeSpec:
    return NodeSpec(node_id=node_id, hostname=f"user@10.0.0.{node_id}", gpu_class="pro6000",
                    total_vram_bytes=gib * GB, gpu_count=cards, arch="amd64", **kw)


def test_the_catalogue_lists_the_three_image_models_with_their_compose_services():
    by_id = {s.id: s for s in registry.load(CATALOGUE) if s.is_image}
    assert set(by_id) == set(IMAGES)
    expect = {
        "z-image-turbo": ("omni-z-image", 8091),
        "qwen-image-2512": ("omni-qwen-image", 8092),
        "qwen-image-edit-2511": ("omni-qwen-image-edit", 8093),
    }
    for model_id, (service, port) in expect.items():
        spec = by_id[model_id]
        assert (spec.service, spec.port) == (service, port)
        assert spec.api == "image"
        assert spec.compose_file == "docker-compose.vllm-omni.yml"
        assert spec.compose_project == "kloudchat-vllm-omni"
        assert spec.openrouter == ""
        assert spec.weight_override and spec.activation_override
        assert spec.concurrent_sessions == 1


def test_a_picture_model_is_charged_weight_plus_activation_and_no_gpu_share():
    spec = next(s for s in _images() if s.id == "z-image-turbo")
    util, per_card = planner.charge(spec, spec.ctx_floor, _node("7"), 90 * GB)
    assert util == 0.0
    assert per_card == spec.weight_override + spec.activation_override
    assert planner.kv_bytes(spec, 4096) == 0
    assert spec.ctx_floor == spec.ctx_target == 0


def test_the_site_pins_picture_models_to_nodes_and_cards():
    """Two cards of 96 GiB: the edit model alone on card 1, the other two on card 0."""
    pins = {
        "qwen-image-edit-2511": {"nodes": ("7",), "cards": (1,)},
        "qwen-image-2512": {"nodes": ("7",), "cards": (0,)},
        "z-image-turbo": {"nodes": ("7",), "cards": (0,)},
    }
    result = planner.plan(_images(**pins), [_node("2", cards=1), _node("7")], head="2")
    placed = {p.model_id: p for p in result.placements}
    assert set(placed) == set(IMAGES)
    assert {p.node_id for p in placed.values()} == {"7"}
    assert placed["qwen-image-edit-2511"].devices == (1,)
    assert placed["qwen-image-2512"].devices == (0,)
    assert placed["z-image-turbo"].devices == (0,)
    assert not result.delegations
    assert all(p.ctx == 0 and p.gpu_util == 0.0 for p in placed.values())


def test_a_node_may_be_named_by_host():
    spec = replace(_images()[0], nodes=("10.0.0.7",))
    result = planner.plan([spec], [_node("2", cards=1), _node("7")], head="2")
    assert [p.node_id for p in result.placements] == ["7"]


def test_a_picture_model_without_a_seat_is_not_placed_and_has_no_fallback():
    """A 40 GiB card holds Z-Image only; the others keep their reason and no fallback."""
    result = planner.plan(_images(), [_node("2", cards=1, gib=40)], head=None)
    assert [p.model_id for p in result.placements] == ["z-image-turbo"]
    assert {d.model_id for d in result.delegations} == {"qwen-image-2512", "qwen-image-edit-2511"}
    assert all(not d.fallback for d in result.delegations)
    assert all("weights and activation" in d.reason for d in result.delegations)


def test_a_chat_model_next_to_picture_models_still_restores_its_context():
    """Restoration skips picture models (no context to grow) and does not divide by zero."""
    chat = registry.ModelSpec(
        id="chat", hf_repo="o/chat", dir="chat", service="vllm-chat", port=8001,
        env_prefix="VLLM_CHAT", ctx_floor=32768, concurrent_sessions=1,
    ).bind(model_metadata.image_metadata("chat", 10 * GB), 131072)
    chat = replace(chat, metadata=replace(chat.metadata, n_layers=8, n_kv_heads=4, head_dim=128))
    result = planner.plan([chat, *_images()], [_node("2", cards=1), _node("7")], head="2")
    assert {p.model_id for p in result.placements} >= {"chat", *IMAGES}


def test_the_site_file_may_set_nodes_and_cards_and_nothing_else_new(tmp_path):
    site = tmp_path / "site.yaml"
    site.write_text("models:\n  - id: z-image-turbo\n    nodes: [7]\n    cards: [0]\n")
    spec = next(s for s in registry.load(CATALOGUE, site=site) if s.id == "z-image-turbo")
    assert spec.nodes == ("7",) and spec.cards == (0,)

    site.write_text("models:\n  - id: z-image-turbo\n    service: elsewhere\n")
    with pytest.raises(ValueError, match="service"):
        registry.load(CATALOGUE, site=site)

    site.write_text("models:\n  - id: z-image-turbo\n    cards: [-1]\n")
    with pytest.raises(ValueError, match="ordinals"):
        registry.load(CATALOGUE, site=site)


def _change(placements, specs, *, current=None, node_env=None, nodes=None):
    nodes = nodes or [_node("7")]
    target = planner.Plan(placements=list(placements))
    return applier.compute_diff(
        target=target, current=current or {}, specs=specs, nodes=nodes,
        layout=applier.RemoteLayout(workdir="KloudChat-LLM"),
        local_env_path=None, node_env=node_env,
    )


def _placement(spec, node_id="7", devices=(1,)):
    return planner.Placement(spec.id, node_id, 0, 0.0, 58 * GB, 1, 1, devices)


def test_apply_starts_a_picture_model_from_the_omni_compose_file_and_writes_only_its_cards():
    specs = _images()
    edit = next(s for s in specs if s.id == "qwen-image-edit-2511")
    change = _change([_placement(edit)], specs, node_env={"7": {}})
    kinds = {(a.kind, a.description) for a in change.actions}
    assert kinds == {
        ("env", "VLLM_OMNI_QWENIMAGEEDIT_DEVICES=1"),
        ("start", "start omni-qwen-image-edit"),
    }
    start = next(a for a in change.actions if a.kind == "start")
    assert "-f docker-compose.vllm-omni.yml" in start.command
    assert "kloudchat-vllm-omni" in start.command
    assert start.url == "http://10.0.0.7:8093"
    # No context, memory share or TP is written for a picture model
    assert not [a for a in change.actions if "MAX_LEN" in a.description or "GPU_UTIL" in a.description]


def test_an_unchanged_picture_model_is_left_running():
    specs = _images()
    edit = next(s for s in specs if s.id == "qwen-image-edit-2511")
    change = _change(
        [_placement(edit)], specs,
        current={"7": {"omni-qwen-image-edit"}},
        node_env={"7": {"VLLM_OMNI_QWENIMAGEEDIT_DEVICES": "1"}},
    )
    assert change.actions == []


def test_a_selected_picture_model_that_is_dropped_is_stopped_with_its_own_compose_file():
    specs = _images()
    change = _change([], specs, current={"7": {"omni-z-image"}}, node_env={"7": {}})
    (stop,) = change.actions
    assert stop.kind == "stop" and "-f docker-compose.vllm-omni.yml" in stop.command


def test_an_unselected_omni_container_is_never_stopped():
    """Only the services of selected models are managed besides vllm-*."""
    chat = registry.ModelSpec(
        id="chat", hf_repo="o/chat", dir="chat", service="vllm-chat", port=8001,
        env_prefix="VLLM_CHAT", ctx_floor=4096, concurrent_sessions=1,
    )
    change = _change([], [chat], current={"7": {"omni-z-image", "vllm-old"}}, node_env={"7": {}})
    assert [a.description for a in change.actions] == ["stop vllm-old"]


def test_picture_urls_go_to_imagegen_placed_not_vllm_placed(tmp_path):
    specs = _images()
    edit = next(s for s in specs if s.id == "qwen-image-edit-2511")
    env = tmp_path / ".env"
    env.write_text("VLLM_PLACED=\nIMAGEGEN_PLACED=\n")
    target = planner.Plan(placements=[_placement(edit)])
    change = applier.compute_diff(
        target=target, current={}, specs=specs, nodes=[_node("7")],
        local_env_path=str(env), node_env={"7": {}},
    )
    assert change.local_env == {"IMAGEGEN_PLACED": "qwen-image-edit-2511=http://10.0.0.7:8093"}


def test_a_picture_model_that_fails_to_start_is_dropped_from_imagegen_placed(tmp_path):
    env = tmp_path / ".env"
    env.write_text("VLLM_PLACED=\nIMAGEGEN_PLACED=\n")
    change = applier.ChangePlan(
        actions=[applier.NodeAction("7", "user@10.0.0.7", "start", "start omni-z-image", "false",
                                    url="http://10.0.0.7:8091")],
        local_env={"IMAGEGEN_PLACED": "z-image-turbo=http://10.0.0.7:8091"},
        local_urls={"IMAGEGEN_PLACED": "z-image-turbo=http://10.0.0.7:8091"},
    )
    failures = applier.apply(change, local_env_path=str(env), runner=lambda host, cmd: (1, "boom"))
    assert failures
    assert "IMAGEGEN_PLACED=\n" in env.read_text()


def test_a_diffusers_checkpoint_counts_as_present(monkeypatch):
    """model_index.json marks a picture model's directory, as config.json does a vLLM one."""
    seen = {}

    def fake(host, cmd, **kw):
        seen["cmd"] = cmd
        return 0, "z-image-turbo\nbge-m3\n", ""

    monkeypatch.setattr(inventory, "_ssh", fake)
    assert inventory._probe_checkpoints("h", "/models") == frozenset({"z-image-turbo", "bge-m3"})
    assert "model_index.json" in seen["cmd"] and "config.json" in seen["cmd"]


def test_selected_omni_containers_are_this_stacks_memory_not_foreign(monkeypatch):
    """A selected picture model's container is managed; an unselected one stays foreign."""
    seen = {}
    monkeypatch.setattr(inventory, "_probe_running_containers",
                        lambda host: {"vllm-bgem3", "omni-z-image", "omni-qwen-image"})
    monkeypatch.setattr(inventory, "_probe_card_sizes", lambda host: ((96 * GB,), False))
    monkeypatch.setattr(inventory, "_probe_gpu_class", lambda host: "pro6000")
    monkeypatch.setattr(inventory, "_probe_gpu_count", lambda host: 1)
    monkeypatch.setattr(inventory, "_probe_arch", lambda host: "amd64")
    monkeypatch.setattr(inventory, "_probe_checkpoints", lambda host, root: frozenset())

    def foreign(host, managed):
        seen["managed"] = managed
        return 0, {}

    monkeypatch.setattr(inventory, "_probe_foreign_vram", foreign)
    inventory._probe_node_once("7", "h", services=["omni-z-image", "vllm-bgem3"], models_root="/m")
    assert seen["managed"] == frozenset({"vllm-bgem3", "omni-z-image"})


def test_a_pinned_picture_model_is_seated_before_a_higher_priority_chat_model_takes_its_node():
    """Chat (priority 30) would pick the roomiest node, the two-card one the pictures are pinned to."""
    chat = registry.ModelSpec(
        id="chat", hf_repo="o/chat", dir="chat", service="vllm-chat", port=8001,
        env_prefix="VLLM_CHAT", ctx_floor=32768, concurrent_sessions=1, priority=30,
        gpu_util=0.92,
    ).bind(model_metadata.image_metadata("chat", 60 * GB), 131072)
    chat = replace(chat, metadata=replace(chat.metadata, n_layers=8, n_kv_heads=4, head_dim=128))
    pins = {
        "qwen-image-edit-2511": {"nodes": ("7",), "cards": (1,)},
        "qwen-image-2512": {"nodes": ("7",), "cards": (0,)},
        "z-image-turbo": {"nodes": ("7",), "cards": (0,)},
    }
    nodes = [_node("2", cards=1), _node("3", cards=1), _node("7")]
    result = planner.plan([chat, *_images(**pins)], nodes, head="2")
    placed = {p.model_id: p.node_id for p in result.placements}
    assert {m: placed[m] for m in IMAGES} == {m: "7" for m in IMAGES}
    assert placed["chat"] in {"2", "3"}
    assert not result.delegations
