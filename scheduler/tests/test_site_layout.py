"""Site overrides, declared shares, held nodes and adopting hand-started containers."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest
import yaml

from scheduler import __main__ as cli
from scheduler import applier, inventory, planner, registry
from scheduler.types import GB, Dtype, ModelMetadata, NodeSpec

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = ROOT / "scheduler" / "models.yaml"

#: A head node with 27B, retrieval and transcription; Flash on each pool node
SITE = """\
models:
  - id: qwen3.8-27b
    placement: head
    replicas: 1
    ctx_floor: 131072
    ctx_target: 131072
    gpu_util: 0.66
  - id: qwen3.8-flash
    replicas: 3
    gpu_util: 0.92
  - id: bge-m3
    replicas: 1
    gpu_util: 0.05
  - id: bge-reranker-v2-m3
    replicas: 1
    gpu_util: 0.05
  - id: whisper-large-v3
    replicas: 1
    gpu_util: 0.14
"""

#: Close to the checkpoints' config.json: (KV layers, KV heads, head dim, GiB, native)
_SHAPES = {
    "qwen3.8-27b": (16, 4, 256, 20.6, 262144),
    "qwen3.8-flash": (12, 2, 256, 78.3, 262144),
    "bge-m3": (24, 16, 64, 2.1, 8192),
    "bge-reranker-v2-m3": (24, 16, 64, 2.1, 8192),
    "whisper-large-v3": (32, 20, 64, 2.9, 448),
}

PRO6000 = int(95.59 * GB)


def _bound(site: Path) -> list[registry.ModelSpec]:
    specs = registry.load(CATALOGUE, only=list(_SHAPES), site=site)
    out = []
    for s in specs:
        layers, heads, dim, gib, native = _SHAPES[s.id]
        md = ModelMetadata(model_id=s.id, n_layers=layers, n_kv_heads=heads, head_dim=dim,
                           weight_dtype=Dtype.NVFP4, weight_bytes=int(gib * GB),
                           native_ctx=native)
        out.append(s.bind(md, s.ctx_target_override or native))
    return out


def _node(node_id: str, *, cards: int = 1, foreign=()) -> NodeSpec:
    return NodeSpec(node_id=node_id, hostname=f"solid@172.29.52.{node_id}",
                    gpu_class="pro6000", total_vram_bytes=PRO6000, gpu_count=cards,
                    arch="amd64", foreign_vram_bytes=sum(foreign),
                    foreign_vram_by_card=tuple(foreign))


@pytest.fixture
def site(tmp_path) -> Path:
    path = tmp_path / "site.yaml"
    path.write_text(SITE, encoding="utf-8")
    return path


def test_the_site_file_lays_its_fields_over_the_catalogue(site):
    by_id = {s.id: s for s in registry.load(CATALOGUE, site=site)}
    big = by_id["qwen3.8-27b"]
    assert (big.placement, big.replicas, big.gpu_util) == ("head", 1, 0.66)
    assert (big.ctx_floor, big.ctx_target_override) == (131072, 131072)
    # Untouched fields keep the catalogue's values
    assert big.port == 8001 and big.env_prefix == "VLLM_QWEN27B"
    assert by_id["qwen3.8-flash"].placement == "pool"


def test_a_missing_site_file_changes_nothing(tmp_path):
    plain = registry.load(CATALOGUE)
    assert registry.load(CATALOGUE, site=tmp_path / "absent.yaml") == plain


@pytest.mark.parametrize("body, message", [
    ("models:\n  - id: nope\n    replicas: 1\n", "not in models.yaml"),
    ("models:\n  - id: bge-m3\n    port: 9999\n", "sets port"),
    ("models:\n  - id: bge-m3\n    gpu_util: 0.99\n", "gpu_util must be within"),
])
def test_a_site_file_cannot_rename_move_or_oversize_a_model(tmp_path, body, message):
    path = tmp_path / "site.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        registry.load(CATALOGUE, site=path)


def test_a_declared_gpu_util_is_charged_as_is(site):
    specs = _bound(site)
    big = next(s for s in specs if s.id == "qwen3.8-27b")
    result = planner.plan([big], [_node("2")])
    (p,) = result.placements
    assert (p.gpu_util, p.ctx) == (0.66, 131072)
    assert p.charge == math.ceil(0.66 * PRO6000)


def test_a_declared_share_narrows_sessions_rather_than_growing():
    spec = registry.ModelSpec(
        id="m", hf_repo="o/m", dir="m", service="vllm-m", port=8001,
        env_prefix="VLLM_M", ctx_floor=131072, concurrent_sessions=8, gpu_util=0.92,
        activation_override=6 * GB,
    ).bind(ModelMetadata(model_id="m", n_layers=12, n_kv_heads=2, head_dim=256,
                         weight_dtype=Dtype.NVFP4, weight_bytes=int(78.3 * GB),
                         native_ctx=262144), 131072)
    result = planner.plan([spec], [_node("3")])
    (p,) = result.placements
    assert p.gpu_util == 0.92 and p.sessions < 8
    assert not any("below the" in n for n in result.notes)


def test_a_share_too_small_for_the_floor_is_seated_with_a_note():
    spec = registry.ModelSpec(
        id="m", hf_repo="o/m", dir="m", service="vllm-m", port=8001,
        env_prefix="VLLM_M", ctx_floor=131072, concurrent_sessions=1, gpu_util=0.30,
    ).bind(ModelMetadata(model_id="m", n_layers=12, n_kv_heads=2, head_dim=256,
                         weight_dtype=Dtype.NVFP4, weight_bytes=int(78.3 * GB),
                         native_ctx=131072), 131072)
    result = planner.plan([spec], [_node("3")])
    assert result.placements and result.placements[0].gpu_util == 0.30
    assert any("gpu_util 0.30 is below" in n for n in result.notes)


def _layout(result):
    return sorted((p.node_id, p.model_id, p.ctx, p.gpu_util) for p in result.placements)


EXPECTED = [
    ("2", "bge-m3", 8192, 0.05),
    ("2", "bge-reranker-v2-m3", 8192, 0.05),
    ("2", "qwen3.8-27b", 131072, 0.66),
    ("2", "whisper-large-v3", 448, 0.14),
    ("3", "qwen3.8-flash", 131072, 0.92),
    ("4", "qwen3.8-flash", 131072, 0.92),
    ("5", "qwen3.8-flash", 131072, 0.92),
]


def test_the_site_file_reproduces_the_running_layout(site):
    nodes = [_node(n) for n in ("2", "3", "4", "5")]
    deployed = {"qwen3.8-27b": frozenset({"2"}), "qwen3.8-flash": frozenset({"3", "4", "5"})}
    result = planner.plan(_bound(site), nodes, head="2", deployed=deployed)
    assert _layout(result) == EXPECTED
    assert not result.delegations


def test_the_layout_does_not_depend_on_what_is_running(site):
    """From empty nodes, the same layout: placement and replicas decide, not incumbency."""
    nodes = [_node(n) for n in ("2", "3", "4", "5")]
    assert _layout(planner.plan(_bound(site), nodes, head="2")) == EXPECTED


def test_without_the_site_file_the_pool_is_given_to_the_27b():
    """Why the site file is needed: the catalogue alone replicates the 27B onto a Flash node."""
    nodes = [_node(n) for n in ("2", "3", "4", "5")]
    result = planner.plan(_bound(Path("/nonexistent")), nodes, head="2")
    flash = [p.node_id for p in result.placements if p.model_id == "qwen3.8-flash"]
    assert len(flash) < 3


HOSTS = {n: f"solid@172.29.52.{n}" for n in ("2", "3", "4", "5", "7")}


@pytest.fixture
def env(tmp_path, monkeypatch):
    def write(text: str) -> None:
        (tmp_path / ".env").write_text(text, encoding="utf-8")
    monkeypatch.setattr(cli, "ENV_FILE", tmp_path / ".env")
    return write


@pytest.mark.parametrize("entry", ["solid@172.29.52.7", "172.29.52.7", "7"])
def test_a_held_node_is_named_by_target_host_or_id(env, entry):
    env(f"VLLM_NODES_HOLD={entry}\n")
    assert cli._held_nodes(HOSTS) == {"7"}


def test_a_held_node_must_be_listed_and_cannot_be_the_head(env):
    env("VLLM_NODES_HOLD=172.29.52.9\n")
    with pytest.raises(SystemExit, match="not in VLLM_NODES"):
        cli._held_nodes(HOSTS)
    env("VLLM_NODES_HOLD=172.29.52.2\n")
    with pytest.raises(SystemExit, match="head node"):
        cli._held_nodes(HOSTS)


def test_an_image_node_with_room_takes_nothing_once_held(site):
    """Without the hold, spare room on the image node's second card draws small models."""
    image = _node("7", cards=2, foreign=(int(75.3 * GB), int(56.9 * GB)))
    pool = [_node(n) for n in ("2", "3", "4", "5")]
    catalogue_only = _bound(Path("/nonexistent"))
    assert any(p.node_id == "7" for p in planner.plan(catalogue_only, pool + [image],
                                                       head="2").placements)
    # Held: the CLI leaves it out of the nodes it plans over
    assert _layout(planner.plan(_bound(site), pool, head="2")) == EXPECTED


def test_a_service_vllm_endpoints_points_at_is_never_stopped(env):
    env("VLLM_ENDPOINTS=qwen3.8-27b=http://172.29.52.2:8001,http://172.29.52.3:8004/\n")
    specs = registry.load(CATALOGUE)
    probes = [inventory.NodeProbe(spec=_node(n), alive=True) for n in ("2", "3", "4")]
    kept = cli._kept(probes, specs)
    assert kept == {"2": {"vllm-qwen27b"}, "3": {"vllm-qwenflash"}}

    change = applier.compute_diff(
        target=planner.Plan(), current={"2": {"vllm-qwen27b", "vllm-bgem3"}},
        specs=specs, nodes=[_node("2")], kept=kept,
    )
    assert [a.description for a in change.actions] == ["stop vllm-bgem3"]


def test_start_and_recreate_take_over_a_container_compose_did_not_create():
    spec = registry.load(CATALOGUE, only=["bge-m3"])[0]
    spec = spec.bind(ModelMetadata(model_id="b", n_layers=24, n_kv_heads=16, head_dim=64,
                                   weight_dtype=Dtype.FP16, weight_bytes=2 * GB,
                                   native_ctx=8192), 8192)
    nodes = [_node("2")]
    result = planner.plan([spec], nodes)
    for current, kind in (({}, "start"), ({"2": {"vllm-bgem3"}}, "recreate")):
        change = applier.compute_diff(target=result, current=current, specs=[spec],
                                      nodes=nodes, node_env={"2": {}})
        (action,) = [a for a in change.actions if a.kind == kind]
        assert "com.docker.compose.project" in action.command
        assert f"!= {applier.COMPOSE_PROJECT}" in action.command
        assert action.command.index("docker rm -f vllm-bgem3") < action.command.index("up -d")


def test_a_stop_also_stops_a_container_compose_did_not_create():
    change = applier.compute_diff(target=planner.Plan(), current={"2": {"vllm-qwenflash"}},
                                  specs=registry.load(CATALOGUE), nodes=[_node("2")])
    (action,) = change.actions
    assert "compose -f docker-compose.vllm.yml stop vllm-qwenflash" in action.command
    assert "docker stop vllm-qwenflash" in action.command


def test_the_adoption_shell_only_removes_an_unlabelled_container(tmp_path):
    """Run the generated shell against a fake docker: labelled kept, unlabelled removed."""
    log = tmp_path / "log"
    fake = tmp_path / "docker"
    fake.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> {log}\n'
        'case "$1 $2" in\n'
        '  "inspect -f") [ "$4" = vllm-composed ] && echo kloudchat-vllm || echo ""; exit 0 ;;\n'
        '  "inspect vllm-gone") exit 1 ;;\n'
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    env = {"PATH": f"{tmp_path}:/usr/bin:/bin"}
    for name in ("vllm-composed", "vllm-hand", "vllm-gone"):
        subprocess.run(["sh", "-c", applier._adopt(name)], check=True, env=env)
    removed = [line for line in log.read_text().splitlines() if line.startswith("rm")]
    assert removed == ["rm -f vllm-hand"]


class _Result:
    def __init__(self, rc, out=""):
        self.returncode, self.stdout, self.stderr = rc, out, ""


def test_ssh_is_retried_while_the_connection_fails(monkeypatch):
    calls = []

    def run(*a, **k):
        calls.append(1)
        return _Result(255) if len(calls) < 3 else _Result(0, "ok\n")

    monkeypatch.setattr(inventory.subprocess, "run", run)
    monkeypatch.setattr(inventory.time, "sleep", lambda s: None)
    assert inventory._ssh("h", "true") == (0, "ok\n", "")
    assert len(calls) == 3


def test_an_unreachable_node_env_is_none_not_empty(monkeypatch):
    monkeypatch.setattr(inventory, "_ssh", lambda h, c, **k: (255, "", "reset"))
    assert inventory.read_env("h", "KloudChat-LLM/.env") is None
    monkeypatch.setattr(inventory, "_ssh", lambda h, c, **k: (0, "", ""))
    assert inventory.read_env("h", "KloudChat-LLM/.env") == {}


@pytest.mark.parametrize("docker_ok, memory_ok", [(True, False), (False, True)])
def test_a_node_answering_in_part_is_not_alive(monkeypatch, docker_ok, memory_ok):
    """Planned on half an answer, a node would have its services stopped or doubled."""
    monkeypatch.setattr(inventory, "_probe_running_containers",
                        lambda h: {"vllm-qwen27b"} if docker_ok else None)
    monkeypatch.setattr(inventory, "_probe_card_sizes",
                        lambda h: ((PRO6000,), False) if memory_ok else ((), False))
    monkeypatch.setattr(inventory, "_probe_gpu_class", lambda h: "pro6000")
    monkeypatch.setattr(inventory, "_probe_gpu_count", lambda h: 1)
    monkeypatch.setattr(inventory, "_probe_arch", lambda h: "amd64")
    monkeypatch.setattr(inventory, "_probe_foreign_vram", lambda h, m: (0, {}))
    probe = inventory.probe_node("2", "solid@172.29.52.2")
    assert not probe.alive and probe.raw_errors


def test_flash_kv_offload_is_off_unless_set():
    services = yaml.safe_load((ROOT / "docker-compose.vllm.yml").read_text())["services"]
    command = services["vllm-qwenflash"]["command"]
    i = command.index("--kv-offloading-size")
    # Empty is vLLM's None (offloading off)
    assert command[i + 1] == "${VLLM_QWENFLASH_KV_OFFLOAD_GB:-}"


def test_a_discrete_card_is_never_sized_as_unified_memory(monkeypatch):
    """A failed nvidia-smi read must not fall through to system RAM on a discrete node."""
    calls = []

    def ssh(host, cmd, **k):
        calls.append(cmd)
        return (255, "", "reset") if "memory.total" in cmd else (0, "393216000\n", "")

    monkeypatch.setattr(inventory, "_ssh", ssh)
    assert inventory._probe_card_sizes("h") == ((), False)
    assert not any("meminfo" in c for c in calls)

    monkeypatch.setattr(inventory, "_probe_running_containers", lambda h: {"vllm-qwenflash"})
    monkeypatch.setattr(inventory, "_probe_card_sizes", lambda h: ((375 * GB,), True))
    monkeypatch.setattr(inventory, "_probe_gpu_class", lambda h: "pro6000")
    monkeypatch.setattr(inventory, "_probe_gpu_count", lambda h: 1)
    monkeypatch.setattr(inventory, "_probe_arch", lambda h: "amd64")
    monkeypatch.setattr(inventory, "_probe_foreign_vram", lambda h, m: (0, {}))
    assert not inventory.probe_node("4", "solid@172.29.52.4").alive


def test_a_declared_share_keeps_its_context_at_the_floor(site):
    """bge-m3 declares 8192 below a native 8194; a declared share does not round it up."""
    specs = [s for s in _bound(site) if s.id == "bge-m3"]
    specs = [s.bind(s.metadata, 8194) for s in specs]
    (p,) = planner.plan(specs, [_node("2")]).placements
    assert p.ctx == 8192
