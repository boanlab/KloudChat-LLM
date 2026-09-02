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
#: backend reached through /tools/stt rather than a LiteLLM chat route.
RUNNERS = ("generate", "pooling", "transcription")

#: "head": first node in NODES_VLLM; "pool": every other node; "any": unconstrained
PLACEMENTS = ("any", "head", "pool")

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
    served_name: str
    ctx_floor: int
    concurrent_sessions: int
    #: Architectures that can serve it. Empty means all.
    arches: tuple[str, ...] = ()
    #: Placement order, highest first; ties seat the largest model first
    priority: int = 0
    #: One of PLACEMENTS. Ignored on a single-node cluster.
    placement: str = "any"
    #: Relative weight for extra instances among models competing for the same nodes
    share: float = 1.0
    #: Cards to shard across on one node (--tensor-parallel-size). Above 1 the
    #: model takes those cards outright.
    tensor_parallel: int = 1
    #: One of RUNNERS. Pooling holds no KV cache; transcription is sized like generate.
    runner: str = "generate"
    #: Filled by bind()
    metadata: Optional[ModelMetadata] = None
    #: Native context; 0 before probing
    ctx_target: int = 0
    #: Declared overrides of probed values
    ctx_target_override: Optional[int] = None
    weight_override: Optional[int] = None

    @property
    def is_pooling(self) -> bool:
        return self.runner == "pooling"

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
    if "hf_repo" not in entry:
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

    env_prefix = str(entry.get("env_prefix") or _env_prefix_from_id(model_id))
    return ModelSpec(
        id=model_id,
        hf_repo=str(entry["hf_repo"]),
        dir=str(entry.get("dir") or model_id),
        # Derived from env_prefix so the .env key and compose service agree
        service=str(entry.get("service") or env_prefix.lower().replace("_", "-")),
        port=int(entry.get("port") or BASE_PORT + index),
        env_prefix=env_prefix,
        served_name=str(entry.get("served_name") or f"local/{model_id}"),
        ctx_floor=int(entry["ctx_floor"]) if entry.get("ctx_floor") else 0,
        concurrent_sessions=int(
            entry.get("concurrent_sessions") or DEFAULT_CONCURRENT_SESSIONS
        ),
        priority=int(entry.get("priority") or 0),
        placement=placement,
        share=share,
        tensor_parallel=max(1, int(entry.get("tensor_parallel") or 1)),
        arches=tuple(entry.get("arches") or ()),
        runner=runner,
        ctx_target_override=int(entry["ctx_target"]) if entry.get("ctx_target") else None,
        weight_override=parse_size(entry["weight"]) if entry.get("weight") else None,
    )


def load(path: Path | str, *, only: Iterable[str] | None = None) -> list[ModelSpec]:
    """Read models.yaml. With ``only``, keep those ids in the given order.

    Raises:
        KeyError: an id in ``only`` is not defined.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = data.get("models") or []
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
    return [by_id[w] for w in wanted]
