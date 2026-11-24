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


# --- speech (vllm-omni /v1/audio/speech): qwen3-tts ---

def _speech() -> registry.ModelSpec:
    spec = next(s for s in registry.load(CATALOGUE) if s.id == "qwen3-tts")
    return spec.bind(model_metadata.image_metadata(spec.id, spec.weight_bytes), 0)


def _free(node_id: str, *free_gib: float) -> NodeSpec:
    """A 96 GiB-per-card node whose cards have ``free_gib`` left for this plan."""
    probe = _node(node_id, cards=len(free_gib))
    budget = probe.card_budget_bytes()
    return replace(probe, foreign_vram_by_card=tuple(max(0, budget - int(f * GB)) for f in free_gib))


def test_the_catalogue_lists_the_speech_model_on_the_omni_compose_file_without_a_fallback():
    spec = next(s for s in registry.load(CATALOGUE) if s.id == "qwen3-tts")
    assert (spec.runner, spec.api, spec.service, spec.port) == ("speech", "speech", "omni-qwen3-tts", 8094)
    assert spec.is_speech and spec.is_omni and not spec.is_image
    assert spec.compose_file == "docker-compose.vllm-omni.yml"
    assert spec.compose_project == "kloudchat-vllm-omni"
    assert spec.openrouter == "" and spec.priority == 1 and spec.replicas == 1
    assert spec.weight_override == int(4.5 * 2**30) and spec.activation_override == 5 * 2**30


def test_a_speech_model_is_charged_weight_plus_activation_with_no_kv_and_no_gpu_share():
    spec = _speech()
    util, per_card = planner.charge(spec, 0, _node("7"), 90 * GB)
    assert util == 0.0 and per_card == spec.weight_override + spec.activation_override
    assert planner.kv_bytes(spec, 4096) == 0
    assert spec.ctx_floor == spec.ctx_target == 0


def test_a_speech_model_takes_the_card_with_room_beside_the_picture_models():
    """Card 0 holds Z-Image and the 2512 (79 GiB of 96); the TTS goes to the emptier card 1."""
    pins = {
        "qwen-image-2512": {"nodes": ("7",), "cards": (0,)},
        "z-image-turbo": {"nodes": ("7",), "cards": (0,)},
    }
    images = [s for s in _images(**pins) if s.id != "qwen-image-edit-2511"]
    tts = replace(_speech(), nodes=("7",))
    result = planner.plan([*images, tts], [_node("7")], head=None)
    placed = {p.model_id: p for p in result.placements}
    assert placed["qwen3-tts"].devices == (1,)
    assert placed["qwen3-tts"].ctx == 0 and placed["qwen3-tts"].gpu_util == 0.0
    assert placed["qwen3-tts"].charge == 9.5 * GB
    assert not result.delegations


def test_a_speech_model_is_not_placed_where_no_card_has_room_and_has_no_fallback():
    """A 9 GiB card cannot take 9.5 GiB; the reason says so and nothing is delegated to OpenRouter."""
    result = planner.plan([_speech()], [_node("3", cards=1, gib=9)], head=None)
    assert not result.placements
    (d,) = result.delegations
    assert d.model_id == "qwen3-tts" and not d.fallback
    assert "weights and activation" in d.reason


def test_speech_urls_go_to_vllm_placed_so_litellm_reaches_the_node_directly(tmp_path):
    spec = _speech()
    env = tmp_path / ".env"
    env.write_text("VLLM_PLACED=\nIMAGEGEN_PLACED=\n")
    placement = planner.Placement(spec.id, "7", 0, 0.0, int(9.5 * GB), 1, 1, (1,))
    change = applier.compute_diff(
        target=planner.Plan(placements=[placement]), current={}, specs=[spec], nodes=[_node("7")],
        local_env_path=str(env), node_env={"7": {}},
    )
    assert change.local_env == {"VLLM_PLACED": "qwen3-tts=http://10.0.0.7:8094"}
    kinds = {(a.kind, a.description) for a in change.actions}
    assert kinds == {("env", "VLLM_OMNI_QWEN3TTS_DEVICES=1"), ("start", "start omni-qwen3-tts")}
    start = next(a for a in change.actions if a.kind == "start")
    assert "-f docker-compose.vllm-omni.yml" in start.command


def test_the_planner_seats_speech_only_where_free_vram_exists_never_on_a_full_flash_node():
    """Flash nodes keep 1.9-3.7 GiB free and the head 10.7: only a card with 9.5 GiB takes it."""
    nodes = [_free("3", 1.9), _free("4", 3.7), _free("5", 2.0), _free("7", 3.9, 32)]
    result = planner.plan([_speech()], nodes, head=None)
    (p,) = result.placements
    assert (p.node_id, p.devices) == ("7", (1,))

    # The head's 10.7 GiB would also seat it, but a card with more room is preferred
    result = planner.plan([_speech()], [_free("2", 10.7), *nodes], head="2")
    (p,) = result.placements
    assert (p.node_id, p.devices) == ("7", (1,))


def test_a_site_pin_to_the_small_card_leaves_speech_unplaced_with_a_reason():
    tts = replace(_speech(), nodes=("7",), cards=(0,))
    result = planner.plan([tts], [_free("7", 3.9, 32)], head=None)
    assert not result.placements
    (d,) = result.delegations
    assert not d.fallback and "weights and activation" in d.reason


def test_adding_speech_moves_nothing_that_is_already_placed():
    pins = {
        "qwen-image-edit-2511": {"nodes": ("7",), "cards": (1,)},
        "qwen-image-2512": {"nodes": ("7",), "cards": (0,)},
        "z-image-turbo": {"nodes": ("7",), "cards": (0,)},
    }
    nodes = [_node("2", cards=1), _node("7", cards=2, gib=140)]
    before = planner.plan(_images(**pins), nodes, head="2")
    after = planner.plan([*_images(**pins), replace(_speech(), nodes=("7",))], nodes, head="2")

    def keep(r):
        return sorted(
            (p.model_id, p.node_id, p.devices) for p in r.placements if p.model_id != "qwen3-tts"
        )

    assert keep(after) == keep(before)
    assert any(p.model_id == "qwen3-tts" for p in after.placements)
