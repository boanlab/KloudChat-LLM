"""Concurrency gate — bound local-model load without crossing privacy boundaries.

Why this exists
---------------
LiteLLM's fallback chain only fires on *errors / timeout / cooldown*, not on plain
saturation: an overloaded vLLM node keeps queuing requests (no error) and `timeout`
is a blunt, time-based proxy that can't be lowered on the Deep-Research brain (its
calls legitimately run minutes). What we actually want is a *concurrency* signal:
"if this local model already has N requests in flight, send the overflow to OR."

vLLM exposes exactly that at `/metrics`:
  vllm:num_requests_running  — in-flight (decoding) sequences
  vllm:num_requests_waiting  — queued (capacity-blocked) requests

How it works
------------
A daemon thread polls each gated model's /metrics every CONCURRENCY_GATE_TTL seconds
and records a boolean "saturated" (running >= cap, or anything waiting). The async
pre-call hook reads that flag (non-blocking). A normal local alias is rewritten to
its OpenRouter fallback twin. A model whose model_info carries
`kchat_strict_local: true` is rejected with `strict_local_unavailable` instead;
the hook never changes its model id.

Config is derived from /app/config.yaml: local deployments (`hosted_vllm/local/*`)
give the metrics URL (from api_base), `model_info.kchat_strict_local` identifies
reject-only aliases, and `router_settings.fallbacks` gives normal aliases their OR
twin. Caps default to the measured PRO6000 saturation knees and can be overridden
with CONCURRENCY_GATE_CAPS (JSON: {"local/<model>": <int>}).

If /metrics is unreachable, normal local routing remains fail-open (keep local)
so a metrics blip never forces paid OR egress. Strict-local is fail-closed: an
unknown capacity state is rejected instead of entering an unbounded queue.
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

POLL_TTL = float(os.environ.get("CONCURRENCY_GATE_TTL", "1.5"))  # seconds between polls
SCRAPE_TIMEOUT = float(os.environ.get("CONCURRENCY_GATE_SCRAPE_TIMEOUT", "1.0"))
# Every unique endpoint is scraped concurrently, so a complete cycle is bounded
# by one scrape timeout rather than by ``aliases × nodes × timeout``. Keep a
# second timeout of scheduling/network margin before a known-good sample can be
# called stale.
STRICT_STATE_TTL = max(POLL_TTL * 3, POLL_TTL + SCRAPE_TIMEOUT * 2)
# In-flight caps, keyed by LiteLLM model_name. A key matching no deployment
# silently drops that model from the gate, so `_load_gate_map` warns about it.
#
# The numbers are starting points, not measurements: too high queues latency, too
# low idles the card. The ~3B-active MoEs share a cap.
#
# The 122B does not. Its cap is the one figure here derived rather than guessed:
# 78 GiB of weights leaves ~22 GiB of KV on a GB10, which is ~13 requests at the
# 128K it is deployed with. Queueing past that preempts running sequences instead
# of adding throughput, so the gate spills to OpenRouter first.
DEFAULT_CAPS = {
    "local/qwen3.6-35b": 64,
    "local/glm-4.7-flash": 64,
    "local/qwen3.5-122b-a10b": 12,
    "local/gemma-4-26b-a4b": 64,
    # 48 KiB/token, so this one's KV pool empties four times faster than the 35B's
    "local/qwen3-coder-30b": 24,
    "local/qwen3.6-27b": 32,
    "strict-local/qwen3.6-35b": 64,
    "strict-local/glm-4.7-flash": 64,
    "strict-local/qwen3.5-122b-a10b": 12,
    "strict-local/gemma-4-26b-a4b": 64,
    "strict-local/qwen3-coder-30b": 24,
    "strict-local/qwen3.6-27b": 32,
}


class StrictLocalUnavailableError(RuntimeError):
    """A strict-local request cannot queue safely at the configured capacity."""

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


def _load_gate_map() -> dict:
    """Build spill/reject gate entries from the generated LiteLLM config."""
    caps = dict(DEFAULT_CAPS)
    env_caps = os.environ.get("CONCURRENCY_GATE_CAPS")
    if env_caps:
        try:
            requested = {k: int(v) for k, v in json.loads(env_caps).items()}
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
            # Operators historically configure local/* only. Mirror that cap to
            # its strict alias unless they explicitly supplied a different one.
            for model, cap in overrides.items():
                if model.startswith("local/") and cap > 0:
                    strict_model = "strict-local/" + model.removeprefix("local/")
                    if strict_model not in requested:
                        caps[strict_model] = cap
        except Exception:
            log.warning("concurrency gate: bad CONCURRENCY_GATE_CAPS, using defaults")

    gate: dict = {}
    try:
        import yaml
        path = os.environ.get("CONFIG_FILE_PATH", "/app/config.yaml")
        cfg = yaml.safe_load(open(path)) or {}

        metrics: dict[str, list[str]] = {}
        strict_models: set = set()
        for d in cfg.get("model_list", []) or []:
            lp = (d.get("litellm_params") or {})
            mi = (d.get("model_info") or {})
            name = d.get("model_name", "")
            routed = lp.get("model", "")
            if isinstance(routed, str) and routed.startswith("hosted_vllm/local/") and lp.get("api_base"):
                url = _metrics_url(lp["api_base"])
                if url not in metrics.setdefault(name, []):
                    metrics[name].append(url)
                if mi.get("kchat_strict_local") is True:
                    strict_models.add(name)

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
            elif cap > 0:
                # A cap naming a model the config doesn't serve is a stale key,
                # not a no-op: the gate quietly stops protecting that model. Say
                # which half is missing so a rename is one log line to diagnose.
                missing = []
                if model not in metrics:
                    missing.append("no local vLLM deployment")
                if model not in twin and model not in strict_models:
                    missing.append("no OpenRouter twin in router_settings.fallbacks")
                log.warning("concurrency gate: cap '%s' is inactive — %s",
                            model, " / ".join(missing))
    except Exception:
        log.exception("concurrency gate: failed to load config — gate disabled")
    return gate


def _scrape(url: str) -> tuple:
    """Return (running, waiting) for a vLLM /metrics endpoint (summed across engines)."""
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
        # Strict capacity is unknown until the first successful scrape, which is
        # a reject state. Normal aliases keep the historic fail-open default.
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
        # A backend normally appears twice (normal and strict aliases), and a
        # multi-node deployment contributes several URLs. Scrape every unique
        # endpoint once and in parallel: sequential polling can take longer
        # than STRICT_STATE_TTL and falsely expire a healthy strict alias before
        # the next cycle reaches it.
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
                        except Exception as exc:  # one node is an unknown state
                            samples_by_url[url] = exc
            except Exception as exc:
                # Executor creation itself is rare, but strict capacity is
                # unknown in that case just as it is for a failed HTTP scrape.
                samples_by_url = {url: exc for url in urls}

        for model, g in self.gate.items():
            try:
                samples: list[tuple[float, float]] = []
                for url in g["metrics"]:
                    sample = samples_by_url[url]
                    if isinstance(sample, Exception):
                        raise sample
                    samples.append(sample)
                running = sum(sample[0] for sample in samples)
                waiting = sum(sample[1] for sample in samples)
                # LiteLLM may choose any deployment behind this alias. One
                # saturated node is therefore enough to make strict capacity
                # uncertain; normal aliases conservatively spill as well.
                sat = any(
                    node_running >= g["cap"] or node_waiting > 0
                    for node_running, node_waiting in samples
                )
                self._last_success[model] = time.monotonic()
                prev = self._saturated.get(model, False)
                self._saturated[model] = sat
                if sat != prev:  # log only on transition — operator signal, not per-request spam
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
                # Capacity is part of the strict privacy contract: if it cannot
                # be observed, reject. Normal local aliases keep their historic
                # fail-open behavior and still point only at the requested vLLM
                # until a known saturation state triggers the paid spill.
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
                # Generated privacy aliases use this namespace. A missing gate
                # entry is an unknown capacity state, never permission to call
                # the backend without the fail-closed controls.
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
                # Pinned LiteLLM 1.83.7's process_pre_call_hook_response raises
                # an Exception returned by this hook before the Router runs.
                return StrictLocalUnavailableError()
            data["model"] = g["or"]
            if os.environ.get("CONCURRENCY_GATE_DEBUG") == "1":
                log.warning("concurrency gate: %s → spill to %s", model, g["or"])
            return data
        except Exception:
            log.exception("concurrency gate: pre_call hook failed")
            if isinstance(data, dict) and _strict_request(data):
                return StrictLocalUnavailableError()
            return None

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: Any,
        traceback_str: Optional[str] = None,
    ) -> Any:
        """Normalise strict backend failures without changing normal errors."""
        del original_exception, user_api_key_dict, traceback_str
        if not isinstance(request_data, dict) or not _strict_request(request_data):
            return None
        # LiteLLM 1.83.7 treats an HTTPException returned by this hook as the
        # client-facing error. Import lazily so the callback's pure unit tests
        # do not need the full proxy dependency graph.
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

# Self-register so the dispatcher finds us even if CONFIG only triggers the import.
try:
    import litellm as _litellm
    if gate_instance not in _litellm.callbacks:
        _litellm.callbacks.append(gate_instance)
except Exception:
    log.exception("concurrency gate: self-register failed")
