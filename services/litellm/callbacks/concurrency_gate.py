"""Concurrency gate: bound in-flight load on local vLLM models.

LiteLLM's fallbacks fire on errors, not on saturation. A daemon thread polls each
gated model's vLLM /metrics (num_requests_running, num_requests_waiting) every
CONCURRENCY_GATE_TTL seconds and marks the model saturated when running >= cap
or anything is waiting. The pre-call hook then:

  local/*         rewrites the request to the OpenRouter twin (spill)
  strict-local/*  returns strict_local_unavailable; the model id never changes

Gate entries come from /app/config.yaml: hosted_vllm/* deployments give the
metrics URLs, model_info.kchat_strict_local marks reject-only aliases,
model_info.kchat_max_inflight the cap (DEFAULT_CAP for a strict alias without
one), and router_settings.fallbacks names each spill twin. CONCURRENCY_GATE_CAPS
({"local/<model>": <int>}) overrides the caps.

Unreachable metrics: a node that cannot be scraped is left out of the verdict.
With none reachable, normal aliases stay local (fail-open) and strict aliases are
rejected (fail-closed).
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Union

from litellm.integrations.custom_logger import CustomLogger

log = logging.getLogger("litellm-concurrency-gate")

POLL_TTL = float(os.environ.get("CONCURRENCY_GATE_TTL", "1.5"))
SCRAPE_TIMEOUT = float(os.environ.get("CONCURRENCY_GATE_SCRAPE_TIMEOUT", "1.0"))
# Age beyond which a strict alias's last good sample is stale
STRICT_STATE_TTL = max(POLL_TTL * 3, POLL_TTL + SCRAPE_TIMEOUT * 2)
#: Request metadata key naming the model a spilled request was rewritten from and to
SPILL_KEY = "kchat_spill"
#: model_info modes of the retrieval models
_POOLING_MODES = ("embedding", "rerank")
# In-flight cap for a strict alias whose deployments declare none (model_info
# kchat_max_inflight, from scheduler/models.yaml)
DEFAULT_CAP = 256


class StrictLocalUnavailableError(RuntimeError):
    """Strict-local request refused: capacity saturated or unknown."""

    def __init__(self) -> None:
        super().__init__("strict_local_unavailable")


def _strict_request(data: dict) -> bool:
    model = data.get("model")
    if isinstance(model, str) and model.startswith("strict-local/"):
        return True
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = (data.get("litellm_params") or {}).get("metadata")
    return isinstance(metadata, dict) and metadata.get("kchat_strict_local") is True


def _metrics_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/") + "/metrics"


def _apply_cap_overrides(caps: dict[str, int]) -> None:
    """CONCURRENCY_GATE_CAPS over the declared caps; a local/ cap also covers its strict alias."""
    env_caps = os.environ.get("CONCURRENCY_GATE_CAPS")
    if not env_caps:
        return
    try:
        requested = {k: int(v) for k, v in json.loads(env_caps).items()}
    except Exception:
        log.warning("concurrency gate: bad CONCURRENCY_GATE_CAPS, using the declared caps")
        return
    overrides: dict[str, int] = {}
    for model, cap in requested.items():
        if model.startswith("strict-local/") and cap <= 0:
            log.warning(
                "concurrency gate: ignoring nonpositive strict cap for '%s'",
                model,
            )
            continue
        overrides[model] = cap
    caps.update(overrides)
    for model, cap in overrides.items():
        if model.startswith("local/") and cap > 0:
            strict_model = "strict-local/" + model.removeprefix("local/")
            if strict_model not in requested:
                caps[strict_model] = cap


def _load_gate_map() -> dict:
    """Gate entries (metrics URLs, cap, mode, twin) from the LiteLLM config."""
    gate: dict = {}
    try:
        import yaml
        path = os.environ.get("CONFIG_FILE_PATH", "/app/config.yaml")
        cfg = yaml.safe_load(open(path)) or {}

        metrics: dict[str, list[str]] = {}
        strict_models: set = set()
        caps: dict[str, int] = {}
        for d in cfg.get("model_list", []) or []:
            lp = (d.get("litellm_params") or {})
            mi = (d.get("model_info") or {})
            name = d.get("model_name", "")
            routed = lp.get("model", "")
            # Embedding and rerank hold no chat queue: not gated on vLLM metrics
            if mi.get("mode") in _POOLING_MODES:
                continue
            if isinstance(routed, str) and routed.startswith("hosted_vllm/") and lp.get("api_base"):
                url = _metrics_url(lp["api_base"])
                if url not in metrics.setdefault(name, []):
                    metrics[name].append(url)
                if mi.get("kchat_strict_local") is True:
                    strict_models.add(name)
                declared = mi.get("kchat_max_inflight")
                if isinstance(declared, int) and declared > 0:
                    caps.setdefault(name, declared)
        # Strict aliases are always gated: unknown capacity is refused
        for name in strict_models:
            caps.setdefault(name, DEFAULT_CAP)
        _apply_cap_overrides(caps)

        twin: dict = {}
        for fb in (cfg.get("router_settings", {}) or {}).get("fallbacks", []) or []:
            for k, v in fb.items():
                if v:
                    twin[k] = v[0]

        for model, cap in caps.items():
            if model in metrics and model in strict_models and cap > 0:
                gate[model] = {
                    "metrics": metrics[model],
                    "cap": int(cap),
                    "mode": "reject",
                }
            elif model in metrics and model in twin and cap > 0:
                gate[model] = {
                    "metrics": metrics[model],
                    "cap": int(cap),
                    "mode": "spill",
                    "or": twin[model],
                }
            elif cap > 0 and model not in metrics:
                log.warning("concurrency gate: cap '%s' is inactive — no local vLLM deployment", model)
    except Exception:
        log.exception("concurrency gate: failed to load config — gate disabled")
    return gate


def _load_shim_models() -> set:
    """Strict models with no capacity gate: pictures and transcription through
    their shim, embedding and rerank straight to vLLM.

    None has vLLM chat metrics to gate on or a fallback to spill to, so each can
    only ever reach the local backend: it passes the gate.
    """
    shims = {
        "image_generation": "http://imagegen-shim:",
        "audio_transcription": "http://whisper-shim:",
    }
    try:
        import yaml
        path = os.environ.get("CONFIG_FILE_PATH", "/app/config.yaml")
        cfg = yaml.safe_load(open(path)) or {}
        deployments: dict[str, list[bool]] = {}
        for d in cfg.get("model_list", []) or []:
            lp = (d.get("litellm_params") or {})
            mi = (d.get("model_info") or {})
            name = d.get("model_name", "")
            if not isinstance(name, str) or not name.startswith("strict-local/"):
                continue
            prefix = shims.get(mi.get("mode"))
            through_shim = mi.get("kchat_strict_local") is True and (
                (prefix is not None and str(lp.get("api_base", "")).startswith(prefix))
                or (mi.get("mode") in _POOLING_MODES
                    and str(lp.get("model", "")).startswith("hosted_vllm/"))
            )
            deployments.setdefault(name, []).append(through_shim)
        twinned = {
            k
            for fb in (cfg.get("router_settings", {}) or {}).get("fallbacks", []) or []
            for k, v in fb.items()
            if v
        }
        return {
            name for name, shim in deployments.items() if all(shim) and name not in twinned
        }
    except Exception:
        log.exception("concurrency gate: failed to read shim models")
        return set()


def _scrape(url: str) -> tuple:
    """(running, waiting) from a vLLM /metrics endpoint, summed across engines."""
    running = waiting = 0.0
    found_running = found_waiting = False
    with urllib.request.urlopen(url, timeout=SCRAPE_TIMEOUT) as r:
        for line in r.read().decode("utf-8", "ignore").splitlines():
            metric = line.split("{", 1)[0].split(" ", 1)[0]
            if metric == "vllm:num_requests_running":
                found_running = True
                running += float(line.rsplit(" ", 1)[-1])
            elif metric == "vllm:num_requests_waiting":
                found_waiting = True
                waiting += float(line.rsplit(" ", 1)[-1])
    if not found_running or not found_waiting:
        raise ValueError("vllm concurrency metrics missing")
    if not all(math.isfinite(value) and value >= 0 for value in (running, waiting)):
        raise ValueError("invalid vllm concurrency metrics")
    return running, waiting


class ConcurrencyGate(CustomLogger):
    def __init__(self):
        self.gate = _load_gate_map()
        self._shim_models = _load_shim_models()
        # Strict aliases rejected until the first successful scrape
        self._saturated = {m: g["mode"] == "reject" for m, g in self.gate.items()}
        self._last_success = {m: 0.0 for m in self.gate}
        if self.gate:
            threading.Thread(target=self._poll_loop, name="concurrency-gate", daemon=True).start()
            log.warning(
                "concurrency gate ACTIVE: %s",
                {m: {"cap": g["cap"], "mode": g["mode"]} for m, g in self.gate.items()},
            )
        else:
            log.warning("concurrency gate: no gated models found — inactive")

    def _poll_loop(self):
        debug = os.environ.get("CONCURRENCY_GATE_DEBUG") == "1"
        while True:
            self._poll_once(debug=debug)
            time.sleep(POLL_TTL)

    def _poll_once(self, *, debug: bool = False) -> None:
        # Unique endpoints scraped once, in parallel, to stay within STRICT_STATE_TTL
        urls = sorted(
            {
                url
                for gate_entry in self.gate.values()
                for url in gate_entry["metrics"]
            }
        )
        samples_by_url: dict[str, tuple[float, float] | Exception] = {}
        if urls:
            try:
                with ThreadPoolExecutor(
                    max_workers=len(urls),
                    thread_name_prefix="concurrency-gate-scrape",
                ) as executor:
                    futures = {url: executor.submit(_scrape, url) for url in urls}
                    for url, future in futures.items():
                        try:
                            samples_by_url[url] = future.result()
                        except Exception as exc:
                            samples_by_url[url] = exc
            except Exception as exc:
                samples_by_url = {url: exc for url in urls}

        for model, g in self.gate.items():
            try:
                # Reachable nodes only; LiteLLM's cooldown keeps traffic off the rest
                results = [samples_by_url[url] for url in g["metrics"]]
                samples = [r for r in results if not isinstance(r, Exception)]
                if not samples:
                    raise results[0]
                running = sum(sample[0] for sample in samples)
                waiting = sum(sample[1] for sample in samples)
                # Any saturated deployment saturates the alias
                sat = any(
                    node_running >= g["cap"] or node_waiting > 0
                    for node_running, node_waiting in samples
                )
                self._last_success[model] = time.monotonic()
                prev = self._saturated.get(model, False)
                self._saturated[model] = sat
                if sat != prev:  # log transitions only
                    if sat and g["mode"] == "reject":
                        state = "REJECTING strict-local"
                    elif sat:
                        state = "SPILLING → OR"
                    else:
                        state = "back to LOCAL"
                    log.warning(
                        "concurrency gate: %s %s (running=%.0f waiting=%.0f cap=%d)",
                        model,
                        state,
                        running,
                        waiting,
                        g["cap"],
                    )
                if debug:
                    log.warning(
                        "gate poll %s: running=%.0f waiting=%.0f cap=%d sat=%s",
                        model,
                        running,
                        waiting,
                        g["cap"],
                        sat,
                    )
            except Exception as e:
                # Unobservable capacity: strict rejects, normal stays local.
                self._saturated[model] = g["mode"] == "reject"
                if debug:
                    log.warning(
                        "gate poll %s: scrape failed %s (mode=%s)",
                        model,
                        e,
                        g["mode"],
                    )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        try:
            if not isinstance(data, dict):
                return None
            model = data.get("model")
            g = self.gate.get(model)
            forced = model in os.environ.get("CONCURRENCY_GATE_FORCE", "").split(",")
            if os.environ.get("CONCURRENCY_GATE_DEBUG") == "1":
                log.warning("gate hook: model=%s gated=%s forced=%s sat=%s",
                            model, g is not None, forced, self._saturated.get(model))
            if g is None:
                # Strict alias with no gate entry: unknown capacity, rejected,
                # unless a shim alone serves it (pictures, transcription).
                if model in getattr(self, "_shim_models", set()):
                    return None
                if _strict_request(data):
                    return StrictLocalUnavailableError()
                return None
            if g["mode"] == "reject":
                metadata = data.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                    data["metadata"] = metadata
                metadata["kchat_strict_local"] = True
            last_success = getattr(self, "_last_success", {}).get(model, 0.0)
            strict_state_stale = (
                g["mode"] == "reject"
                and time.monotonic() - last_success > STRICT_STATE_TTL
            )
            if not (forced or self._saturated.get(model) or strict_state_stale):
                return None
            if g["mode"] == "reject":
                log.warning("concurrency gate: rejecting saturated strict-local model %s", model)
                # LiteLLM raises an Exception returned by this hook before routing.
                return StrictLocalUnavailableError()
            # The response headers name the spill, as LiteLLM does for a router fallback,
            # so the caller's usage record shows the request left for OpenRouter
            metadata = data.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                data["metadata"] = metadata
            metadata[SPILL_KEY] = {"from": model, "to": g["or"]}
            data["model"] = g["or"]
            if os.environ.get("CONCURRENCY_GATE_DEBUG") == "1":
                log.warning("concurrency gate: %s → spill to %s", model, g["or"])
            return data
        except Exception:
            log.exception("concurrency gate: pre_call hook failed")
            if isinstance(data, dict) and _strict_request(data):
                return StrictLocalUnavailableError()
            return None

    async def async_post_call_response_headers_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: Any,
        request_headers: Optional[dict] = None,
        litellm_call_info: Optional[dict] = None,
    ) -> Optional[dict]:
        """Headers a router fallback would carry, for a request the gate spilled."""
        del user_api_key_dict, response, request_headers, litellm_call_info
        try:
            spill = (data.get("metadata") or {}).get(SPILL_KEY) if isinstance(data, dict) else None
            if not isinstance(spill, dict) or not spill.get("to"):
                return None
            return {
                "x-litellm-attempted-fallbacks": "1",
                "x-litellm-model-group": str(spill["to"]),
            }
        except Exception:
            log.exception("concurrency gate: spill headers failed")
            return None

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: Any,
        traceback_str: Optional[str] = None,
    ) -> Any:
        """Strict-local backend failures surface as 503 strict_local_unavailable."""
        del original_exception, user_api_key_dict, traceback_str
        if not isinstance(request_data, dict) or not _strict_request(request_data):
            return None
        # An HTTPException returned here becomes the client-facing error.
        # Lazy import: the unit tests run without the proxy dependency graph.
        from fastapi import HTTPException

        return HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "strict_local_unavailable",
                    "type": "service_unavailable",
                    "code": "strict_local_unavailable",
                }
            },
        )


gate_instance = ConcurrencyGate()

# Self-registration for a config that only imports the module
try:
    import litellm as _litellm
    if gate_instance not in _litellm.callbacks:
        _litellm.callbacks.append(gate_instance)
except Exception:
    log.exception("concurrency gate: self-register failed")
