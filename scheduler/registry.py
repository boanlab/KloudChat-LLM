"""models.yaml loader: declared fields merged with config.json facts. Declared values win."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from scheduler.types import ModelMetadata

#: Context floor: a quarter of native, never below CTX_FLOOR_MIN.
CTX_FLOOR_DIVISOR = 4
CTX_FLOOR_MIN = 32 * 1024

#: Host port of the first vLLM service on a node. Each further model takes +1.
BASE_PORT = 8001

#: Concurrent sessions assumed when sizing the effective KV context.
DEFAULT_CONCURRENT_SESSIONS = 4

#: "generate" | "pooling" (vLLM --runner names); "transcription" marks the STT
#: backend behind whisper-shim; "image" a vllm-omni picture model behind imagegen-shim;
#: "speech" a vllm-omni text-to-speech model, routed by LiteLLM straight to /v1/audio/speech.
RUNNERS = ("generate", "pooling", "transcription", "image", "speech")

#: Route LiteLLM registers: chat (local/ and strict-local/), embedding, rerank,
#: transcription (strict-local only, through whisper-shim), image (strict-local
#: only, through imagegen-shim), speech (strict-local only, /v1/audio/speech)
APIS = ("chat", "embedding", "rerank", "transcription", "image", "speech")
_API_BY_RUNNER = {
    "generate": "chat", "pooling": "embedding", "transcription": "transcription",
    "image": "image", "speech": "speech",
}

#: Compose file and project of each runner's services on a node
VLLM_COMPOSE_FILE = "docker-compose.vllm.yml"
VLLM_COMPOSE_PROJECT = "kloudchat-vllm"
OMNI_COMPOSE_FILE = "docker-compose.vllm-omni.yml"
OMNI_COMPOSE_PROJECT = "kloudchat-vllm-omni"

#: In-flight cap for a chat model that declares none: spill or refuse only once vLLM queues
DEFAULT_MAX_INFLIGHT = 256

#: "head": first node in VLLM_NODES; "pool": every other node; "any": unconstrained;
#: "head-first": the first instance on the head node, extra instances on any node
PLACEMENTS = ("any", "head", "head-first", "pool")

#: --gpu-memory-utilization bounds; 1.0 fails engine init
MAX_GPU_UTIL = 0.95
MIN_GPU_UTIL = 0.05

#: Fields a site file (scheduler/site.yaml) may lay over a catalogue entry:
#: where and how large, never what the model is or where it answers
SITE_FIELDS = (
    "placement", "replicas", "share", "priority", "ctx_floor", "ctx_target",
    "concurrent_sessions", "tensor_parallel", "activation", "weight", "arches",
    "gpu_util", "nodes", "cards",
)

_SIZE = re.compile(r"^\s*([\d.]+)\s*([KMGT]i?B)?\s*$", re.I)
_UNITS = {
    "b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12,
    "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40,
}


def parse_size(value: Any) -> int:
    """Bytes from an integer or a "21.4GiB" string."""
    if isinstance(value, (int, float)):
        return int(value)
    m = _SIZE.match(str(value))
    if not m:
        raise ValueError(f"unparseable size: {value!r}")
    number, unit = m.group(1), (m.group(2) or "B").lower()
    if unit not in _UNITS:
        raise ValueError(f"unknown unit: {unit}")
    return int(float(number) * _UNITS[unit])


@dataclass(frozen=True)
class ModelSpec:
    """One deployable model, with derived values filled in."""

    id: str
    hf_repo: str
    dir: str
    service: str
    port: int
    env_prefix: str
    ctx_floor: int
    concurrent_sessions: int
    #: Architectures that can serve it. Empty means all.
    arches: tuple[str, ...] = ()
    #: Placement order, highest first; ties seat the largest model first
    priority: int = 0
    #: One of PLACEMENTS. Ignored on a single-node cluster.
    placement: str = "any"
    #: Cap on this model's instances, external ones included; None fills spare capacity
    replicas: Optional[int] = None
    #: Relative weight for extra instances among models competing for the same nodes
    share: float = 1.0
    #: Cards to shard across on one node (--tensor-parallel-size). Above 1 the
    #: model takes those cards outright.
    tensor_parallel: int = 1
    #: One of RUNNERS. Pooling holds no KV cache; transcription is sized like generate.
    runner: str = "generate"
    #: One of APIS
    api: str = "chat"
    #: OpenRouter slug: the fallback twin, and the route when no vLLM serves it
    openrouter: str = ""
    #: LiteLLM request timeout in seconds; 0 keeps LiteLLM's default
    timeout: int = 0
    #: In-flight requests per node before local/ spills and strict-local/ refuses
    max_inflight: int = DEFAULT_MAX_INFLIGHT
    #: Default presence_penalty on the deployment; None sends none
    presence_penalty: Optional[float] = None
    #: Default temperature on the deployment; None sends none
    temperature: Optional[float] = None
    #: Run by hand (VLLM_ENDPOINTS), never placed; listed for its LiteLLM routes
    manual: bool = False
    #: Nodes it may run on, by node id or host; empty means any the placement allows
    nodes: tuple[str, ...] = ()
    #: CUDA device ordinals it may take on a node; empty means any card
    cards: tuple[int, ...] = ()
    #: Filled by bind()
    metadata: Optional[ModelMetadata] = None
    #: Native context; 0 before probing
    ctx_target: int = 0
    #: Declared overrides of probed values
    ctx_target_override: Optional[int] = None
    weight_override: Optional[int] = None
    #: Runtime headroom per card; None takes the planner default
    activation_override: Optional[int] = None
    #: Fixed --gpu-memory-utilization: the planner charges this share of the
    #: card instead of sizing one. None sizes it.
    gpu_util: Optional[float] = None

    @property
    def is_pooling(self) -> bool:
        return self.runner == "pooling"

    @property
    def is_image(self) -> bool:
        """A vllm-omni picture model: sized by weights and activation, no context or KV."""
        return self.runner == "image"

    @property
    def is_speech(self) -> bool:
        """A vllm-omni speech model: sized like a picture model (weights and activation, no KV)."""
        return self.runner == "speech"

    @property
    def is_omni(self) -> bool:
        """Served by vllm-omni (docker-compose.vllm-omni.yml) and sized by weights and
        activation, with no context window or KV cache: picture and speech models."""
        return self.runner in ("image", "speech")

    @property
    def compose_file(self) -> str:
        return OMNI_COMPOSE_FILE if self.is_omni else VLLM_COMPOSE_FILE

    @property
    def compose_project(self) -> str:
        return OMNI_COMPOSE_PROJECT if self.is_omni else VLLM_COMPOSE_PROJECT

    def runs_on(self, arch: str) -> bool:
        """Servable on ``arch``; an unknown arch ("") is never excluded."""
        return not self.arches or not arch or arch in self.arches

    @property
    def weight_bytes(self) -> int:
        if self.weight_override is not None:
            return self.weight_override
        return self.metadata.weight_bytes if self.metadata else 0

    def bind(self, metadata: ModelMetadata, native_ctx: int) -> "ModelSpec":
        """Copy with probe results bound."""
        if self.is_omni:
            return replace(self, metadata=metadata, ctx_target=0, ctx_floor=0)
        target = self.ctx_target_override or native_ctx
        floor = self.ctx_floor or max(CTX_FLOOR_MIN, target // CTX_FLOOR_DIVISOR)
        # Clamped for short-context models
        floor = min(floor, target) if target else floor
        return replace(self, metadata=metadata, ctx_target=target, ctx_floor=floor)


def _env_prefix_from_id(model_id: str) -> str:
    return "VLLM_" + re.sub(r"[^A-Z0-9]+", "_", model_id.upper()).strip("_")


def _spec_from_entry(entry: dict, index: int) -> ModelSpec:
    model_id = str(entry["id"]).strip()
    if not model_id:
        raise ValueError("a models.yaml entry has no id")
    manual = bool(entry.get("manual"))
    if "hf_repo" not in entry and not manual:
        raise ValueError(f"{model_id}: hf_repo is required")

    runner = str(entry.get("runner") or "generate").strip().lower()
    if runner not in RUNNERS:
        raise ValueError(
            f"{model_id}: runner {runner!r} is not one of {', '.join(RUNNERS)}"
        )
    placement = str(entry.get("placement") or "any").strip().lower()
    if placement not in PLACEMENTS:
        raise ValueError(
            f"{model_id}: placement {placement!r} is not one of "
            f"{', '.join(PLACEMENTS)}"
        )
    # Explicit 0 is an error, not the default
    raw_share = entry.get("share")
    share = 1.0 if raw_share is None else float(raw_share)
    if share <= 0:
        raise ValueError(f"{model_id}: share must be above zero, not {share}")

    api = str(entry.get("api") or _API_BY_RUNNER[runner]).strip().lower()
    if api not in APIS:
        raise ValueError(f"{model_id}: api {api!r} is not one of {', '.join(APIS)}")
    max_inflight = int(entry.get("max_inflight") or DEFAULT_MAX_INFLIGHT)
    if max_inflight <= 0:
        raise ValueError(f"{model_id}: max_inflight must be above zero")
    raw_penalty = entry.get("presence_penalty")
    raw_temperature = entry.get("temperature")

    raw_replicas = entry.get("replicas")
    replicas = None if raw_replicas is None else int(raw_replicas)
    if replicas is not None and replicas < 1:
        raise ValueError(f"{model_id}: replicas must be at least 1, not {replicas}")

    raw_util = entry.get("gpu_util")
    gpu_util = None if raw_util is None else float(raw_util)
    if gpu_util is not None and not MIN_GPU_UTIL <= gpu_util <= MAX_GPU_UTIL:
        raise ValueError(
            f"{model_id}: gpu_util must be within {MIN_GPU_UTIL}-{MAX_GPU_UTIL}, not {gpu_util}"
        )

    nodes = tuple(str(n).strip() for n in (entry.get("nodes") or ()) if str(n).strip())
    cards = tuple(int(c) for c in (entry.get("cards") or ()))
    if any(c < 0 for c in cards):
        raise ValueError(f"{model_id}: cards are CUDA ordinals, not {list(cards)}")

    env_prefix = str(entry.get("env_prefix") or _env_prefix_from_id(model_id))
    return ModelSpec(
        id=model_id,
        hf_repo=str(entry.get("hf_repo") or ""),
        manual=manual,
        nodes=nodes,
        cards=cards,
        dir=str(entry.get("dir") or model_id),
        # Derived from env_prefix so the .env key and compose service agree
        service=str(entry.get("service") or env_prefix.lower().replace("_", "-")),
        port=int(entry.get("port") or BASE_PORT + index),
        env_prefix=env_prefix,
        ctx_floor=int(entry["ctx_floor"]) if entry.get("ctx_floor") else 0,
        # An omni model has no KV to size
        concurrent_sessions=1 if runner in ("image", "speech") else int(
            entry.get("concurrent_sessions") or DEFAULT_CONCURRENT_SESSIONS
        ),
        priority=int(entry.get("priority") or 0),
        placement=placement,
        share=share,
        replicas=replicas,
        tensor_parallel=max(1, int(entry.get("tensor_parallel") or 1)),
        arches=tuple(entry.get("arches") or ()),
        runner=runner,
        api=api,
        openrouter=str(entry.get("openrouter") or "").strip(),
        timeout=int(entry.get("timeout") or 0),
        max_inflight=max_inflight,
        presence_penalty=None if raw_penalty is None else float(raw_penalty),
        temperature=None if raw_temperature is None else float(raw_temperature),
        ctx_target_override=int(entry["ctx_target"]) if entry.get("ctx_target") else None,
        weight_override=parse_size(entry["weight"]) if entry.get("weight") else None,
        activation_override=(
            parse_size(entry["activation"]) if entry.get("activation") else None
        ),
        gpu_util=gpu_util,
    )


def _overlay(entries: list[dict], site: Path) -> list[dict]:
    """Catalogue entries with the site file's fields laid over them, by id.

    Raises:
        ValueError: the site names an unknown id or a field outside SITE_FIELDS.
    """
    data = yaml.safe_load(site.read_text(encoding="utf-8")) or {}
    merged = {str(e.get("id")).strip(): dict(e) for e in entries}
    for over in data.get("models") or []:
        model_id = str(over.get("id") or "").strip()
        if model_id not in merged:
            raise ValueError(f"{site.name}: {model_id or '(no id)'} is not in models.yaml")
        fields = set(over) - {"id"}
        foreign = sorted(fields - set(SITE_FIELDS))
        if foreign:
            raise ValueError(
                f"{site.name}: {model_id} sets {', '.join(foreign)}; a site sets only "
                f"{', '.join(SITE_FIELDS)}"
            )
        merged[model_id].update({k: over[k] for k in fields})
    return [merged[str(e.get("id")).strip()] for e in entries]


def load(path: Path | str, *, only: Iterable[str] | None = None,
         site: Path | str | None = None) -> list[ModelSpec]:
    """Read models.yaml, with ``site`` laid over it when that file exists.
    With ``only``, keep those ids in the given order.

    Raises:
        KeyError: an id in ``only`` is not defined.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = data.get("models") or []
    if site is not None and Path(site).is_file():
        entries = _overlay(entries, Path(site))
    specs = [_spec_from_entry(e, i) for i, e in enumerate(entries)]

    if only is None:
        return specs

    wanted = [s.strip() for s in only if s and s.strip()]
    by_id = {s.id: s for s in specs}
    missing = [w for w in wanted if w not in by_id]
    if missing:
        raise KeyError(
            f"models not defined in models.yaml: {', '.join(missing)} "
            f"(defined: {', '.join(by_id)})"
        )
    hand_run = [w for w in wanted if by_id[w].manual]
    if hand_run:
        raise ValueError(
            f"run by hand, not deployable: {', '.join(hand_run)} "
            "(serve it and list it in VLLM_ENDPOINTS)"
        )
    return [by_id[w] for w in wanted]
