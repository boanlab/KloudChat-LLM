from __future__ import annotations

import asyncio
import importlib
import io
import sys
import types
from pathlib import Path

import yaml


def _load_gate_module():
    litellm = types.ModuleType("litellm")
    litellm.callbacks = []
    integrations = types.ModuleType("litellm.integrations")
    custom_logger = types.ModuleType("litellm.integrations.custom_logger")

    class CustomLogger:
        pass

    custom_logger.CustomLogger = CustomLogger
    fastapi = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail):
            self.status_code = status_code
            self.detail = detail

    fastapi.HTTPException = HTTPException
    sys.modules["litellm"] = litellm
    sys.modules["litellm.integrations"] = integrations
    sys.modules["litellm.integrations.custom_logger"] = custom_logger
    sys.modules["fastapi"] = fastapi
    sys.modules.pop("services.litellm.callbacks.concurrency_gate", None)
    return importlib.import_module("services.litellm.callbacks.concurrency_gate")


def test_gate_map_separates_normal_spill_and_strict_rejection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gate_module = _load_gate_module()
    config = {
        "model_list": [
            {
                "model_name": "local/qwen3.8-27b",
                "litellm_params": {
                    "model": "hosted_vllm/local/qwen3.8-27b",
                    "api_base": "http://qwen:8000/v1",
                },
                "model_info": {"kchat_strict_local": False, "kchat_max_inflight": 128},
            },
            {
                "model_name": "strict-local/qwen3.8-27b",
                "litellm_params": {
                    "model": "hosted_vllm/local/qwen3.8-27b",
                    "api_base": "http://qwen:8000/v1",
                },
                "model_info": {"kchat_strict_local": True, "kchat_max_inflight": 128},
            },
            {
                "model_name": "strict-local/qwen3.8-27b",
                "litellm_params": {
                    "model": "hosted_vllm/local/qwen3.8-27b",
                    "api_base": "http://qwen-2:8000/v1",
                },
                "model_info": {"kchat_strict_local": True, "kchat_max_inflight": 128},
            },
        ],
        "router_settings": {
            "fallbacks": [
                {"local/qwen3.8-27b": ["qwen/qwen3.8-27b"]},
            ]
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("CONFIG_FILE_PATH", str(config_path))

    gate = gate_module._load_gate_map()

    assert gate["local/qwen3.8-27b"]["mode"] == "spill"
    assert gate["local/qwen3.8-27b"]["or"] == "qwen/qwen3.8-27b"
    assert gate["strict-local/qwen3.8-27b"]["mode"] == "reject"
    assert gate["strict-local/qwen3.8-27b"]["metrics"] == [
        "http://qwen:8000/metrics",
        "http://qwen-2:8000/metrics",
    ]
    assert "or" not in gate["strict-local/qwen3.8-27b"]


def test_normal_cap_override_is_mirrored_to_strict_alias(tmp_path: Path, monkeypatch) -> None:
    gate_module = _load_gate_module()
    config = {
        "model_list": [
            {
                "model_name": "local/qwen3.8-27b",
                "litellm_params": {
                    "model": "hosted_vllm/local/qwen3.8-27b",
                    "api_base": "http://qwen:8000/v1",
                },
                "model_info": {"kchat_max_inflight": 128},
            },
            {
                "model_name": "strict-local/qwen3.8-27b",
                "litellm_params": {
                    "model": "hosted_vllm/local/qwen3.8-27b",
                    "api_base": "http://qwen:8000/v1",
                },
                "model_info": {"kchat_strict_local": True, "kchat_max_inflight": 128},
            },
        ],
        "router_settings": {
            "fallbacks": [{"local/qwen3.8-27b": ["qwen/qwen3.8-27b"]}]
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("CONFIG_FILE_PATH", str(config_path))
    monkeypatch.setenv("CONCURRENCY_GATE_CAPS", '{"local/qwen3.8-27b": 12}')

    gate = gate_module._load_gate_map()

    assert gate["local/qwen3.8-27b"]["cap"] == 12
    assert gate["strict-local/qwen3.8-27b"]["cap"] == 12


def test_nonpositive_overrides_cannot_disable_strict_rejection(
    tmp_path: Path, monkeypatch
) -> None:
    gate_module = _load_gate_module()
    config = {
        "model_list": [
            {
                "model_name": "local/qwen3.8-27b",
                "litellm_params": {
                    "model": "hosted_vllm/local/qwen3.8-27b",
                    "api_base": "http://qwen:8000/v1",
                },
                "model_info": {"kchat_max_inflight": 128},
            },
            {
                "model_name": "strict-local/qwen3.8-27b",
                "litellm_params": {
                    "model": "hosted_vllm/local/qwen3.8-27b",
                    "api_base": "http://qwen:8000/v1",
                },
                "model_info": {"kchat_strict_local": True, "kchat_max_inflight": 128},
            },
        ],
        "router_settings": {
            "fallbacks": [{"local/qwen3.8-27b": ["qwen/qwen3.8-27b"]}]
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("CONFIG_FILE_PATH", str(config_path))
    monkeypatch.setenv(
        "CONCURRENCY_GATE_CAPS",
        '{"local/qwen3.8-27b": 0, "strict-local/qwen3.8-27b": -1}',
    )

    gate = gate_module._load_gate_map()

    assert "local/qwen3.8-27b" not in gate
    assert gate["strict-local/qwen3.8-27b"]["mode"] == "reject"
    assert gate["strict-local/qwen3.8-27b"]["cap"] == 128


def test_caps_come_from_the_deployments_model_info(tmp_path: Path, monkeypatch) -> None:
    """kchat_max_inflight sets the cap; a strict alias without one takes DEFAULT_CAP,
    a normal alias without one is not gated."""
    gate_module = _load_gate_module()
    monkeypatch.delenv("CONCURRENCY_GATE_CAPS", raising=False)
    config = {
        "model_list": [
            {
                "model_name": "local/any-model",
                "litellm_params": {"model": "hosted_vllm/Org/Any-Model", "api_base": "http://a:8000/v1"},
                "model_info": {"kchat_max_inflight": 7},
            },
            {
                "model_name": "strict-local/any-model",
                "litellm_params": {"model": "hosted_vllm/Org/Any-Model", "api_base": "http://a:8000/v1"},
                "model_info": {"kchat_strict_local": True},
            },
            {
                "model_name": "local/uncapped",
                "litellm_params": {"model": "hosted_vllm/local/uncapped", "api_base": "http://b:8000/v1"},
            },
        ],
        "router_settings": {
            "fallbacks": [
                {"local/any-model": ["org/any-model"]},
                {"local/uncapped": ["org/uncapped"]},
            ]
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("CONFIG_FILE_PATH", str(config_path))

    gate = gate_module._load_gate_map()

    assert gate["local/any-model"] == {
        "metrics": ["http://a:8000/metrics"], "cap": 7, "mode": "spill", "or": "org/any-model",
    }
    assert gate["strict-local/any-model"]["cap"] == gate_module.DEFAULT_CAP
    assert "local/uncapped" not in gate


def test_strict_alias_starts_rejected_until_first_successful_poll(monkeypatch) -> None:
    gate_module = _load_gate_module()
    entries = {
        "local/qwen3.8-27b": {
            "metrics": ["http://qwen:8000/metrics"],
            "cap": 64,
            "mode": "spill",
            "or": "qwen/qwen3.8-27b",
        },
        "strict-local/qwen3.8-27b": {
            "metrics": ["http://qwen:8000/metrics"],
            "cap": 64,
            "mode": "reject",
        },
    }

    class ThreadWithoutStart:
        def __init__(self, **_kwargs):
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(gate_module, "_load_gate_map", lambda: entries)
    monkeypatch.setattr(gate_module.threading, "Thread", ThreadWithoutStart)

    gate = gate_module.ConcurrencyGate()

    assert gate._saturated["local/qwen3.8-27b"] is False
    assert gate._saturated["strict-local/qwen3.8-27b"] is True


def test_saturated_strict_alias_fails_without_rewriting_model() -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {
        "strict-local/qwen3.8-27b": {
            "metrics": ["http://qwen:8000/metrics"],
            "cap": 64,
            "mode": "reject",
        }
    }
    gate._saturated = {"strict-local/qwen3.8-27b": True}
    data = {"model": "strict-local/qwen3.8-27b", "messages": [{"content": "secret"}]}

    result = asyncio.run(gate.async_pre_call_hook(None, None, data, "acompletion"))

    assert isinstance(result, gate_module.StrictLocalUnavailableError)
    assert str(result) == "strict_local_unavailable"
    assert data["model"] == "strict-local/qwen3.8-27b"


def test_saturated_normal_alias_preserves_existing_openrouter_spill() -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {
        "local/qwen3.8-27b": {
            "metrics": ["http://qwen:8000/metrics"],
            "cap": 64,
            "mode": "spill",
            "or": "qwen/qwen3.8-27b",
        }
    }
    gate._saturated = {"local/qwen3.8-27b": True}
    data = {"model": "local/qwen3.8-27b"}

    result = asyncio.run(gate.async_pre_call_hook(None, None, data, "acompletion"))

    assert result is data
    assert data["model"] == "qwen/qwen3.8-27b"


def test_unsaturated_strict_alias_stays_on_vllm() -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {
        "strict-local/qwen3.8-27b": {
            "metrics": ["http://qwen:8000/metrics"],
            "cap": 64,
            "mode": "reject",
        }
    }
    gate._saturated = {"strict-local/qwen3.8-27b": False}
    gate._last_success = {"strict-local/qwen3.8-27b": gate_module.time.monotonic()}
    data = {"model": "strict-local/qwen3.8-27b"}

    result = asyncio.run(gate.async_pre_call_hook(None, None, data, "acompletion"))

    assert result is None
    assert data["model"] == "strict-local/qwen3.8-27b"
    assert data["metadata"]["kchat_strict_local"] is True


def test_unregistered_strict_alias_is_rejected() -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {}
    gate._saturated = {}
    gate._last_success = {}
    data = {"model": "strict-local/future-model"}

    result = asyncio.run(gate.async_pre_call_hook(None, None, data, "acompletion"))

    assert isinstance(result, gate_module.StrictLocalUnavailableError)


def test_strict_backend_failure_is_normalized_to_service_unavailable() -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)

    result = asyncio.run(
        gate.async_post_call_failure_hook(
            request_data={
                "model": "hosted_vllm/local/qwen3.8-27b",
                "metadata": {"kchat_strict_local": True},
            },
            original_exception=OSError("connection reset"),
            user_api_key_dict=None,
        )
    )

    assert result.status_code == 503
    assert result.detail["error"]["code"] == "strict_local_unavailable"


def test_normal_backend_failure_is_not_transformed() -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)

    result = asyncio.run(
        gate.async_post_call_failure_hook(
            request_data={"model": "local/qwen3.8-27b"},
            original_exception=OSError("connection reset"),
            user_api_key_dict=None,
        )
    )

    assert result is None


def test_stale_strict_capacity_state_is_rejected(monkeypatch) -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {
        "strict-local/qwen3.8-27b": {
            "metrics": ["http://qwen:8000/metrics"],
            "cap": 64,
            "mode": "reject",
        }
    }
    gate._saturated = {"strict-local/qwen3.8-27b": False}
    gate._last_success = {"strict-local/qwen3.8-27b": 10.0}
    monkeypatch.setattr(
        gate_module.time,
        "monotonic",
        lambda: 10.0 + gate_module.STRICT_STATE_TTL + 0.01,
    )
    data = {"model": "strict-local/qwen3.8-27b"}

    result = asyncio.run(gate.async_pre_call_hook(None, None, data, "acompletion"))

    assert isinstance(result, gate_module.StrictLocalUnavailableError)
    assert data["model"] == "strict-local/qwen3.8-27b"


def test_metrics_failure_rejects_strict_but_keeps_normal_fail_open(monkeypatch) -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {
        "local/qwen3.8-27b": {
            "metrics": ["http://qwen:8000/metrics"],
            "cap": 64,
            "mode": "spill",
            "or": "qwen/qwen3.8-27b",
        },
        "strict-local/qwen3.8-27b": {
            "metrics": ["http://qwen:8000/metrics"],
            "cap": 64,
            "mode": "reject",
        },
    }
    gate._saturated = {model: False for model in gate.gate}

    def unavailable(_url: str):
        raise OSError("metrics unavailable")

    monkeypatch.setattr(gate_module, "_scrape", unavailable)
    gate._poll_once()

    assert gate._saturated["local/qwen3.8-27b"] is False
    assert gate._saturated["strict-local/qwen3.8-27b"] is True


def test_poll_cycle_scrapes_shared_multi_node_endpoints_once_and_in_parallel(
    monkeypatch,
) -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    qwen_urls = [
        "http://qwen-1:8000/metrics",
        "http://qwen-2:8000/metrics",
    ]
    big_urls = [
        "http://big-1:8000/metrics",
        "http://big-2:8000/metrics",
    ]
    gate.gate = {
        # Normal and strict aliases share deployments.
        "local/qwen3.8-27b": {
            "metrics": qwen_urls,
            "cap": 64,
            "mode": "spill",
            "or": "qwen/qwen3.8-27b",
        },
        "strict-local/qwen3.8-27b": {
            "metrics": qwen_urls,
            "cap": 64,
            "mode": "reject",
        },
        "strict-local/qwen3-coder-next": {
            "metrics": big_urls,
            "cap": 12,
            "mode": "reject",
        },
    }
    gate._saturated = {model: True for model in gate.gate}
    gate._last_success = {model: 0.0 for model in gate.gate}

    # All four unique scrapes must be in flight together; a duplicate would
    # push a URL's call count past one.
    barrier = gate_module.threading.Barrier(4)
    lock = gate_module.threading.Lock()
    calls: dict[str, int] = {}

    def slow_scrape(url: str) -> tuple[float, float]:
        with lock:
            calls[url] = calls.get(url, 0) + 1
        barrier.wait(timeout=1.0)
        return 0.0, 0.0

    monkeypatch.setattr(gate_module, "_scrape", slow_scrape)
    gate._poll_once()

    assert calls == {url: 1 for url in sorted(qwen_urls + big_urls)}
    assert all(saturated is False for saturated in gate._saturated.values())
    assert all(last_success > 0 for last_success in gate._last_success.values())


def test_one_unknown_node_leaves_a_multi_node_strict_alias_on_the_other(monkeypatch) -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {
        "strict-local/qwen3.8-27b": {
            "metrics": [
                "http://qwen-1:8000/metrics",
                "http://qwen-2:8000/metrics",
            ],
            "cap": 64,
            "mode": "reject",
        }
    }
    gate._saturated = {"strict-local/qwen3.8-27b": True}
    gate._last_success = {"strict-local/qwen3.8-27b": 0.0}

    def one_node_unknown(url: str):
        if "qwen-2" in url:
            raise OSError("metrics unavailable")
        return 0.0, 0.0

    monkeypatch.setattr(gate_module, "_scrape", one_node_unknown)
    gate._poll_once()

    assert gate._saturated["strict-local/qwen3.8-27b"] is False


def test_saturated_reachable_node_still_spills_while_its_peer_is_down(monkeypatch) -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {
        "local/qwen3.8-27b": {
            "metrics": [
                "http://qwen-1:8000/metrics",
                "http://qwen-2:8000/metrics",
            ],
            "cap": 64,
            "mode": "spill",
            "or": "qwen/qwen3.8-27b",
        }
    }
    gate._saturated = {"local/qwen3.8-27b": False}
    gate._last_success = {"local/qwen3.8-27b": 0.0}

    def one_down_one_busy(url: str):
        if "qwen-2" in url:
            raise OSError("metrics unavailable")
        return 64.0, 3.0

    monkeypatch.setattr(gate_module, "_scrape", one_down_one_busy)
    gate._poll_once()

    assert gate._saturated["local/qwen3.8-27b"] is True


def test_http_200_without_vllm_capacity_metrics_is_a_scrape_failure(monkeypatch) -> None:
    gate_module = _load_gate_module()

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        gate_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(b"# HELP process_cpu_seconds_total\n"),
    )

    try:
        gate_module._scrape("http://qwen:8000/metrics")
    except ValueError as exc:
        assert str(exc) == "vllm concurrency metrics missing"
    else:
        raise AssertionError("missing vLLM metrics must fail closed")


def test_non_finite_capacity_metrics_are_rejected(monkeypatch) -> None:
    gate_module = _load_gate_module()

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    payload = (
        b'vllm:num_requests_running{engine="0"} NaN\n'
        b'vllm:num_requests_waiting{engine="0"} 0\n'
    )
    monkeypatch.setattr(
        gate_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(payload),
    )

    try:
        gate_module._scrape("http://qwen:8000/metrics")
    except ValueError as exc:
        assert str(exc) == "invalid vllm concurrency metrics"
    else:
        raise AssertionError("non-finite vLLM metrics must fail closed")


def test_strict_picture_model_behind_the_shim_passes_the_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gate_module = _load_gate_module()
    picture_info = {"mode": "image_generation", "kchat_strict_local": True}
    config = {
        "model_list": [
            {
                "model_name": "strict-local/z-image-turbo",
                "litellm_params": {
                    "model": "openai/z-image-turbo",
                    "api_base": "http://imagegen-shim:9100/v1",
                },
                "model_info": picture_info,
            },
            {
                "model_name": "strict-local/elsewhere-image",
                "litellm_params": {
                    "model": "openai/local/elsewhere-image",
                    "api_base": "http://somewhere:9100/v1",
                },
                "model_info": picture_info,
            },
            {
                "model_name": "strict-local/twinned-image",
                "litellm_params": {
                    "model": "openai/local/twinned-image",
                    "api_base": "http://imagegen-shim:9100/v1",
                },
                "model_info": picture_info,
            },
        ],
        "router_settings": {
            "fallbacks": [{"strict-local/twinned-image": ["vendor/image"]}]
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("CONFIG_FILE_PATH", str(config_path))

    assert gate_module._load_shim_models() == {"strict-local/z-image-turbo"}

    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {}
    gate._saturated = {}
    gate._last_success = {}
    gate._shim_models = gate_module._load_shim_models()
    data = {"model": "strict-local/z-image-turbo"}
    result = asyncio.run(gate.async_pre_call_hook(None, None, data, "aimage_generation"))
    assert result is None
    assert data["model"] == "strict-local/z-image-turbo"

    for model in ("strict-local/elsewhere-image", "strict-local/twinned-image"):
        refused = asyncio.run(
            gate.async_pre_call_hook(None, None, {"model": model}, "aimage_generation")
        )
        assert isinstance(refused, gate_module.StrictLocalUnavailableError)


def test_strict_transcription_behind_the_shim_passes_the_gate(tmp_path: Path, monkeypatch) -> None:
    """A recording has no capacity metrics and no twin: it reaches whisper-shim or nothing."""
    gate_module = _load_gate_module()
    info = {"mode": "audio_transcription", "kchat_strict_local": True}
    config = {
        "model_list": [
            {
                "model_name": "strict-local/whisper-large-v3",
                "litellm_params": {
                    "model": "openai/whisper-large-v3",
                    "api_base": "http://whisper-shim:9000/v1",
                },
                "model_info": info,
            },
            {
                "model_name": "strict-local/elsewhere-stt",
                "litellm_params": {"model": "openai/x", "api_base": "http://somewhere:9000/v1"},
                "model_info": info,
            },
        ],
        "router_settings": {"fallbacks": []},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("CONFIG_FILE_PATH", str(config_path))
    assert gate_module._load_shim_models() == {"strict-local/whisper-large-v3"}

    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {}
    gate._saturated = {}
    gate._last_success = {}
    gate._shim_models = gate_module._load_shim_models()
    data = {"model": "strict-local/whisper-large-v3"}
    assert asyncio.run(gate.async_pre_call_hook(None, None, data, "atranscription")) is None
    refused = asyncio.run(
        gate.async_pre_call_hook(None, None, {"model": "strict-local/elsewhere-stt"}, "atranscription")
    )
    assert isinstance(refused, gate_module.StrictLocalUnavailableError)


def test_a_spilled_request_carries_the_fallback_headers() -> None:
    """The caller's usage record can tell the request left: same headers as a router fallback."""
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {"local/m": {"metrics": ["http://a:8000/metrics"], "cap": 1,
                             "mode": "spill", "or": "org/m"}}
    gate._saturated = {"local/m": True}
    gate._last_success = {"local/m": 0.0}
    gate._shim_models = set()
    data = {"model": "local/m"}
    out = asyncio.run(gate.async_pre_call_hook(None, None, data, "acompletion"))
    assert out["model"] == "org/m"
    headers = asyncio.run(gate.async_post_call_response_headers_hook(out, None, None))
    assert headers == {"x-litellm-attempted-fallbacks": "1", "x-litellm-model-group": "org/m"}

    # A request that stayed local adds nothing
    gate._saturated = {"local/m": False}
    stayed = {"model": "local/m"}
    asyncio.run(gate.async_pre_call_hook(None, None, stayed, "acompletion"))
    assert asyncio.run(gate.async_post_call_response_headers_hook(stayed, None, None)) is None
