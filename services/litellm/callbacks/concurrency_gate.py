"""Concurrency gate — spill local-model overload to OpenRouter to bound latency.

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
pre-call hook reads that flag (non-blocking) and, when saturated, rewrites
`data["model"]` to the model's OpenRouter fallback twin — so the overflow is served
by OR while local in-flight stays at ~cap, keeping per-stream latency bounded.

Config is derived from /app/config.yaml: local deployments (`hosted_vllm/local/*`)
give the metrics URL (from api_base), and `router_settings.fallbacks` gives the OR
twin. Caps default to the measured PRO6000 saturation knees and can be overridden
with CONCURRENCY_GATE_CAPS (JSON: {"local/<model>": <int>}).

Fail-open: if /metrics is unreachable, the model is treated as NOT saturated (keep
local) — a metrics blip must never force paid OR egress.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Any, Optional, Union

from litellm.integrations.custom_logger import CustomLogger

log = logging.getLogger("litellm-concurrency-gate")

POLL_TTL = float(os.environ.get("CONCURRENCY_GATE_TTL", "1.5"))  # seconds between polls
SCRAPE_TIMEOUT = float(os.environ.get("CONCURRENCY_GATE_SCRAPE_TIMEOUT", "1.0"))
# In-flight caps, keyed by LiteLLM model_name. A key matching no deployment
# silently drops that model from the gate, so `_load_gate_map` warns about it.
#
# The numbers are starting points, not measurements: too high queues latency, too
# low idles the card. Both models are ~3B-active MoE and share a cap.
DEFAULT_CAPS = {
    "local/qwen3.6-35b": 64,
    "local/glm-4.7-flash": 64,
}


def _metrics_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/") + "/metrics"


def _load_gate_map() -> dict:
    """Build {local_model: {"metrics", "cap", "or"}} from the litellm config."""
    caps = dict(DEFAULT_CAPS)
    env_caps = os.environ.get("CONCURRENCY_GATE_CAPS")
    if env_caps:
        try:
            caps.update({k: int(v) for k, v in json.loads(env_caps).items()})
        except Exception:
            log.warning("concurrency gate: bad CONCURRENCY_GATE_CAPS, using defaults")

    gate: dict = {}
    try:
        import yaml
        path = os.environ.get("CONFIG_FILE_PATH", "/app/config.yaml")
        cfg = yaml.safe_load(open(path)) or {}

        metrics: dict = {}
        for d in cfg.get("model_list", []) or []:
            lp = (d.get("litellm_params") or {})
            name = d.get("model_name", "")
            routed = lp.get("model", "")
            if isinstance(routed, str) and routed.startswith("hosted_vllm/local/") and lp.get("api_base"):
                metrics[name] = _metrics_url(lp["api_base"])

        twin: dict = {}
        for fb in (cfg.get("router_settings", {}) or {}).get("fallbacks", []) or []:
            for k, v in fb.items():
                if v:
                    twin[k] = v[0]

        for model, cap in caps.items():
            if model in metrics and model in twin and cap > 0:
                gate[model] = {"metrics": metrics[model], "cap": int(cap), "or": twin[model]}
            elif cap > 0:
                # A cap naming a model the config doesn't serve is a stale key,
                # not a no-op: the gate quietly stops protecting that model. Say
                # which half is missing so a rename is one log line to diagnose.
                missing = []
                if model not in metrics:
                    missing.append("no local vLLM deployment")
                if model not in twin:
                    missing.append("no OpenRouter twin in router_settings.fallbacks")
                log.warning("concurrency gate: cap '%s' is inactive — %s",
                            model, " / ".join(missing))
    except Exception:
        log.exception("concurrency gate: failed to load config — gate disabled")
    return gate


def _scrape(url: str) -> tuple:
    """Return (running, waiting) for a vLLM /metrics endpoint (summed across engines)."""
    running = waiting = 0.0
    with urllib.request.urlopen(url, timeout=SCRAPE_TIMEOUT) as r:
        for line in r.read().decode("utf-8", "ignore").splitlines():
            if line.startswith("vllm:num_requests_running{"):
                running += float(line.rsplit(" ", 1)[-1])
            elif line.startswith("vllm:num_requests_waiting{"):  # excludes _by_reason{...}
                waiting += float(line.rsplit(" ", 1)[-1])
    return running, waiting


class ConcurrencyGate(CustomLogger):
    def __init__(self):
        self.gate = _load_gate_map()
        self._saturated = {m: False for m in self.gate}
        if self.gate:
            threading.Thread(target=self._poll_loop, name="concurrency-gate", daemon=True).start()
            log.warning("concurrency gate ACTIVE: %s",
                        {m: g["cap"] for m, g in self.gate.items()})
        else:
            log.warning("concurrency gate: no gated models found — inactive")

    def _poll_loop(self):
        debug = os.environ.get("CONCURRENCY_GATE_DEBUG") == "1"
        while True:
            for model, g in self.gate.items():
                try:
                    running, waiting = _scrape(g["metrics"])
                    sat = (running >= g["cap"]) or (waiting > 0)
                    prev = self._saturated.get(model, False)
                    self._saturated[model] = sat
                    if sat != prev:  # log only on transition — operator signal, not per-request spam
                        log.warning("concurrency gate: %s %s (running=%.0f waiting=%.0f cap=%d)",
                                    model, "SPILLING → OR" if sat else "back to LOCAL",
                                    running, waiting, g["cap"])
                    if debug:
                        log.warning("gate poll %s: running=%.0f waiting=%.0f cap=%d sat=%s",
                                    model, running, waiting, g["cap"], sat)
                except Exception as e:
                    self._saturated[model] = False  # fail-open
                    if debug:
                        log.warning("gate poll %s: scrape failed %s", model, e)
            time.sleep(POLL_TTL)

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
                return None
            if not (forced or self._saturated.get(model)):
                return None
            data["model"] = g["or"]
            if os.environ.get("CONCURRENCY_GATE_DEBUG") == "1":
                log.warning("concurrency gate: %s → spill to %s", model, g["or"])
            return data
        except Exception:
            log.exception("concurrency gate: pre_call hook failed")
            return None


gate_instance = ConcurrencyGate()

# Self-register so the dispatcher finds us even if CONFIG only triggers the import.
try:
    import litellm as _litellm
    if gate_instance not in _litellm.callbacks:
        _litellm.callbacks.append(gate_instance)
except Exception:
    log.exception("concurrency gate: self-register failed")
