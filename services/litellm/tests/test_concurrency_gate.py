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
                "model_name": "local/qwen3.6-35b",
                "litellm_params": {
                    "model": "hosted_vllm/local/qwen3.6-35b",
                    "api_base": "http://qwen:8000/v1",
                },
                "model_info": {"kchat_strict_local": False},
            },
            {
                "model_name": "strict-local/qwen3.6-35b",
                "litellm_params": {
                    "model": "hosted_vllm/local/qwen3.6-35b",
                    "api_base": "http://qwen:8000/v1",
                },
                "model_info": {"kchat_strict_local": True},
            },
            {
                "model_name": "strict-local/qwen3.6-35b",
                "litellm_params": {
                    "model": "hosted_vllm/local/qwen3.6-35b",
                    "api_base": "http://qwen-2:8000/v1",
                },
                "model_info": {"kchat_strict_local": True},
            },
        ],
        "router_settings": {
            "fallbacks": [
                {"local/qwen3.6-35b": ["qwen/qwen3.6-35b-a3b"]},
            ]
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("CONFIG_FILE_PATH", str(config_path))

    gate = gate_module._load_gate_map()

    assert gate["local/qwen3.6-35b"]["mode"] == "spill"
    assert gate["local/qwen3.6-35b"]["or"] == "qwen/qwen3.6-35b-a3b"
    assert gate["strict-local/qwen3.6-35b"]["mode"] == "reject"
    assert gate["strict-local/qwen3.6-35b"]["metrics"] == [
        "http://qwen:8000/metrics",
        "http://qwen-2:8000/metrics",
    ]
    assert "or" not in gate["strict-local/qwen3.6-35b"]


def test_normal_cap_override_is_mirrored_to_strict_alias(tmp_path: Path, monkeypatch) -> None:
    gate_module = _load_gate_module()
    config = {
        "model_list": [
            {
                "model_name": "local/qwen3.6-35b",
                "litellm_params": {
                    "model": "hosted_vllm/local/qwen3.6-35b",
                    "api_base": "http://qwen:8000/v1",
                },
            },
            {
                "model_name": "strict-local/qwen3.6-35b",
                "litellm_params": {
                    "model": "hosted_vllm/local/qwen3.6-35b",
                    "api_base": "http://qwen:8000/v1",
                },
                "model_info": {"kchat_strict_local": True},
            },
        ],
        "router_settings": {
            "fallbacks": [{"local/qwen3.6-35b": ["qwen/qwen3.6-35b-a3b"]}]
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("CONFIG_FILE_PATH", str(config_path))
    monkeypatch.setenv("CONCURRENCY_GATE_CAPS", '{"local/qwen3.6-35b": 12}')

    gate = gate_module._load_gate_map()

    assert gate["local/qwen3.6-35b"]["cap"] == 12
    assert gate["strict-local/qwen3.6-35b"]["cap"] == 12


def test_nonpositive_overrides_cannot_disable_strict_rejection(
    tmp_path: Path, monkeypatch
) -> None:
    gate_module = _load_gate_module()
    config = {
        "model_list": [
            {
                "model_name": "local/qwen3.6-35b",
                "litellm_params": {
                    "model": "hosted_vllm/local/qwen3.6-35b",
                    "api_base": "http://qwen:8000/v1",
                },
            },
            {
                "model_name": "strict-local/qwen3.6-35b",
                "litellm_params": {
                    "model": "hosted_vllm/local/qwen3.6-35b",
                    "api_base": "http://qwen:8000/v1",
                },
                "model_info": {"kchat_strict_local": True},
            },
        ],
        "router_settings": {
            "fallbacks": [{"local/qwen3.6-35b": ["qwen/qwen3.6-35b-a3b"]}]
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("CONFIG_FILE_PATH", str(config_path))
    monkeypatch.setenv(
        "CONCURRENCY_GATE_CAPS",
        '{"local/qwen3.6-35b": 0, "strict-local/qwen3.6-35b": -1}',
    )

    gate = gate_module._load_gate_map()

    assert "local/qwen3.6-35b" not in gate
    assert gate["strict-local/qwen3.6-35b"]["mode"] == "reject"
    assert gate["strict-local/qwen3.6-35b"]["cap"] == 64


def test_strict_alias_starts_rejected_until_first_successful_poll(monkeypatch) -> None:
    gate_module = _load_gate_module()
    entries = {
        "local/qwen3.6-35b": {
            "metrics": ["http://qwen:8000/metrics"],
            "cap": 64,
            "mode": "spill",
            "or": "qwen/qwen3.6-35b-a3b",
        },
        "strict-local/qwen3.6-35b": {
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

    assert gate._saturated["local/qwen3.6-35b"] is False
    assert gate._saturated["strict-local/qwen3.6-35b"] is True


def test_saturated_strict_alias_fails_without_rewriting_model() -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {
        "strict-local/qwen3.6-35b": {
            "metrics": ["http://qwen:8000/metrics"],
            "cap": 64,
            "mode": "reject",
        }
    }
    gate._saturated = {"strict-local/qwen3.6-35b": True}
    data = {"model": "strict-local/qwen3.6-35b", "messages": [{"content": "secret"}]}

    result = asyncio.run(gate.async_pre_call_hook(None, None, data, "acompletion"))

    assert isinstance(result, gate_module.StrictLocalUnavailableError)
    assert str(result) == "strict_local_unavailable"
    assert data["model"] == "strict-local/qwen3.6-35b"


def test_saturated_normal_alias_preserves_existing_openrouter_spill() -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {
        "local/qwen3.6-35b": {
            "metrics": ["http://qwen:8000/metrics"],
            "cap": 64,
            "mode": "spill",
            "or": "qwen/qwen3.6-35b-a3b",
        }
    }
    gate._saturated = {"local/qwen3.6-35b": True}
    data = {"model": "local/qwen3.6-35b"}

    result = asyncio.run(gate.async_pre_call_hook(None, None, data, "acompletion"))

    assert result is data
    assert data["model"] == "qwen/qwen3.6-35b-a3b"


def test_unsaturated_strict_alias_stays_on_vllm() -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {
        "strict-local/qwen3.6-35b": {
            "metrics": ["http://qwen:8000/metrics"],
            "cap": 64,
            "mode": "reject",
        }
    }
    gate._saturated = {"strict-local/qwen3.6-35b": False}
    gate._last_success = {"strict-local/qwen3.6-35b": gate_module.time.monotonic()}
    data = {"model": "strict-local/qwen3.6-35b"}

    result = asyncio.run(gate.async_pre_call_hook(None, None, data, "acompletion"))

    assert result is None
    assert data["model"] == "strict-local/qwen3.6-35b"
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
                "model": "hosted_vllm/local/qwen3.6-35b",
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
            request_data={"model": "local/qwen3.6-35b"},
            original_exception=OSError("connection reset"),
            user_api_key_dict=None,
        )
    )

    assert result is None


def test_stale_strict_capacity_state_is_rejected(monkeypatch) -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {
        "strict-local/qwen3.6-35b": {
            "metrics": ["http://qwen:8000/metrics"],
            "cap": 64,
            "mode": "reject",
        }
    }
    gate._saturated = {"strict-local/qwen3.6-35b": False}
    gate._last_success = {"strict-local/qwen3.6-35b": 10.0}
    monkeypatch.setattr(
        gate_module.time,
        "monotonic",
        lambda: 10.0 + gate_module.STRICT_STATE_TTL + 0.01,
    )
    data = {"model": "strict-local/qwen3.6-35b"}

    result = asyncio.run(gate.async_pre_call_hook(None, None, data, "acompletion"))

    assert isinstance(result, gate_module.StrictLocalUnavailableError)
    assert data["model"] == "strict-local/qwen3.6-35b"


def test_metrics_failure_rejects_strict_but_keeps_normal_fail_open(monkeypatch) -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {
        "local/qwen3.6-35b": {
            "metrics": ["http://qwen:8000/metrics"],
            "cap": 64,
            "mode": "spill",
            "or": "qwen/qwen3.6-35b-a3b",
        },
        "strict-local/qwen3.6-35b": {
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

    assert gate._saturated["local/qwen3.6-35b"] is False
    assert gate._saturated["strict-local/qwen3.6-35b"] is True


def test_poll_cycle_scrapes_shared_multi_node_endpoints_once_and_in_parallel(
    monkeypatch,
) -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    qwen_urls = [
        "http://qwen-1:8000/metrics",
        "http://qwen-2:8000/metrics",
    ]
    glm_urls = [
        "http://glm-1:8000/metrics",
        "http://glm-2:8000/metrics",
    ]
    gate.gate = {
        # Normal and strict aliases intentionally share the same deployments.
        "local/qwen3.6-35b": {
            "metrics": qwen_urls,
            "cap": 64,
            "mode": "spill",
            "or": "qwen/qwen3.6-35b-a3b",
        },
        "strict-local/qwen3.6-35b": {
            "metrics": qwen_urls,
            "cap": 64,
            "mode": "reject",
        },
        "strict-local/glm-4.7-flash": {
            "metrics": glm_urls,
            "cap": 64,
            "mode": "reject",
        },
    }
    gate._saturated = {model: True for model in gate.gate}
    gate._last_success = {model: 0.0 for model in gate.gate}

    # Sequential code cannot pass this barrier: all four unique scrapes must be
    # in flight together. A duplicate normal/strict scrape would make one URL's
    # call count exceed one.
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

    assert calls == {url: 1 for url in sorted(qwen_urls + glm_urls)}
    assert all(saturated is False for saturated in gate._saturated.values())
    assert all(last_success > 0 for last_success in gate._last_success.values())


def test_one_unknown_node_rejects_a_multi_node_strict_alias(monkeypatch) -> None:
    gate_module = _load_gate_module()
    gate = gate_module.ConcurrencyGate.__new__(gate_module.ConcurrencyGate)
    gate.gate = {
        "strict-local/qwen3.6-35b": {
            "metrics": [
                "http://qwen-1:8000/metrics",
                "http://qwen-2:8000/metrics",
            ],
            "cap": 64,
            "mode": "reject",
        }
    }
    gate._saturated = {"strict-local/qwen3.6-35b": False}

    def one_node_unknown(url: str):
        if "qwen-2" in url:
            raise OSError("metrics unavailable")
        return 0.0, 0.0

    monkeypatch.setattr(gate_module, "_scrape", one_node_unknown)
    gate._poll_once()

    assert gate._saturated["strict-local/qwen3.6-35b"] is True


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
