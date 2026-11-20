"""OpenRouter run policy: open-weight strict, provider pin, ZDR-only (fixtures, no network)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from .test_config_generation import ROOT, _parse_dry_run, _run_generator

_spec = importlib.util.spec_from_file_location("or_policy", ROOT / "scripts" / "or_policy.py")
or_policy = importlib.util.module_from_spec(_spec)
sys.modules["or_policy"] = or_policy
_spec.loader.exec_module(or_policy)


def _ep(provider: str, tag: str, quant: str | None, name: str = "Model X") -> dict:
    return {"provider_name": provider, "tag": tag, "name": f"{provider} | {name}",
            "quantization": quant, "context_length": 131072, "uptime_last_30m": 99.5,
            "pricing": {"prompt": "0.0000004", "completion": "0.000002"}}


def _doc(name: str, *eps: dict) -> dict:
    return {"data": {"id": "x/y", "name": name, "endpoints": list(eps)}}


def _zdr(*rows: tuple[str, str, str]) -> dict:
    return {"data": [{"provider_name": p, "tag": t, "model_name": m, "name": f"{p} | {m}"}
                     for p, t, m in rows]}


KIMI = _doc("Kimi K3",
            _ep("Moonshot AI", "moonshotai", "unknown", "Kimi K3"),
            _ep("Wafer", "wafer/fp8", "fp8", "Kimi K3"),
            _ep("KT", "kt", None, "Kimi K3"),
            _ep("Together", "together/fp4", "fp4", "Kimi K3"))
KIMI_ZDR = _zdr(("Wafer", "wafer/fp8", "Kimi K3"))


def _eps(doc: dict, zdr: dict) -> list[dict]:
    eps = or_policy.normalize(doc, zdr)
    assert eps is not None
    return eps


def test_eligible_endpoints_disclose_quantization_and_are_zdr() -> None:
    eps = _eps(KIMI, KIMI_ZDR)
    eligible = [e["provider"] for e in eps if e["disclosed"] and e["zdr"]]
    assert eligible == ["Wafer"]
    assert {e["provider"]: e["quantization"] for e in eps}["KT"] == "unknown"


def test_open_weight_strict_filters_without_naming_a_provider() -> None:
    d = or_policy.decide("moonshotai/kimi-k3", _eps(KIMI, KIMI_ZDR), open_weight=True,
                         zdr_only=False, pin="")
    assert d["exclude"] is False and d["pinned"] is None
    assert d["provider"]["zdr"] is True and d["provider"]["allow_fallbacks"] is True
    assert "unknown" not in d["provider"]["quantizations"] and "fp8" in d["provider"]["quantizations"]
    assert "order" not in d["provider"]
    assert d["quantization"] == "fp8" and d["eligible"] == 1


def test_a_model_with_no_eligible_endpoint_is_excluded() -> None:
    doc = _doc("GLM", _ep("A", "a", None, "GLM"), _ep("B", "b", "fp8", "GLM"))
    d = or_policy.decide("z-ai/glm-5.3", _eps(doc, _zdr(("A", "a", "GLM"))), open_weight=True,
                         zdr_only=False, pin="")
    assert d["exclude"] is True and "disclosed quantization and ZDR" in d["reason"]


def test_a_pin_must_be_eligible_for_an_open_weight_model() -> None:
    eps = _eps(KIMI, KIMI_ZDR)
    ok = or_policy.decide("moonshotai/kimi-k3", eps, open_weight=True, zdr_only=False, pin="wafer")
    assert ok["provider"]["order"] == ["Wafer"] and ok["provider"]["allow_fallbacks"] is False
    assert ok["pinned"] == "Wafer" and ok["quantization"] == "fp8" and ok["zdr"] is True
    bad = or_policy.decide("moonshotai/kimi-k3", eps, open_weight=True, zdr_only=False, pin="KT")
    assert bad["pinned"] is None and "order" not in bad["provider"]
    assert any("KT" in w and "pin ignored" in w for w in bad["warnings"])
    gone = or_policy.decide("moonshotai/kimi-k3", eps, open_weight=True, zdr_only=False, pin="Nobody")
    assert gone["pinned"] is None and any("does not serve" in w for w in gone["warnings"])


def test_a_pin_on_a_closed_model_records_the_providers_facts() -> None:
    doc = _doc("Claude", _ep("Anthropic", "anthropic", "unknown", "Claude"),
               _ep("Google Vertex", "google-vertex", "unknown", "Claude"))
    eps = _eps(doc, _zdr(("Anthropic", "anthropic", "Claude")))
    d = or_policy.decide("anthropic/claude-x", eps, open_weight=False, zdr_only=False,
                         pin="Anthropic")
    assert d["provider"] == {"order": ["Anthropic"], "allow_fallbacks": False}
    assert d["quantization"] == "unknown" and d["zdr"] is True and d["policy"] == "pinned"
    other = or_policy.decide("anthropic/claude-x", eps, open_weight=False, zdr_only=False,
                             pin="Google Vertex")
    assert other["zdr"] is False


def test_zdr_only_drops_models_without_a_zdr_endpoint_and_conflicting_pins() -> None:
    doc = _doc("Claude", _ep("Anthropic", "anthropic", "unknown", "Claude"))
    none = or_policy.decide("anthropic/c", _eps(doc, _zdr()), open_weight=False, zdr_only=True, pin="")
    assert none["exclude"] is True and none["reason"] == "no ZDR endpoint"
    eps = _eps(doc, _zdr(("Anthropic", "anthropic", "Claude")))
    ok = or_policy.decide("anthropic/c", eps, open_weight=False, zdr_only=True, pin="")
    assert ok["provider"] == {"zdr": True}
    conflict = or_policy.decide("anthropic/c", _eps(KIMI, KIMI_ZDR), open_weight=False,
                                zdr_only=True, pin="KT")
    assert conflict["pinned"] is None and any("not ZDR" in w for w in conflict["warnings"])


def test_without_endpoint_data_the_filter_still_applies() -> None:
    d = or_policy.decide("moonshotai/kimi-k3", None, open_weight=True, zdr_only=False, pin="Wafer")
    assert d["exclude"] is False and d["provider"]["zdr"] is True and d["warnings"]


def test_the_endpoint_table_lists_only_eligible_ones_by_default() -> None:
    eps = _eps(KIMI, KIMI_ZDR)
    text = or_policy.table(eps, only_eligible=True)
    assert "Wafer" in text and "KT" not in text and "fp8" in text
    assert "not disclosed" in or_policy.table(eps, only_eligible=False)


def _fixtures(tmp_path: Path) -> dict[str, str]:
    d = tmp_path / "endpoints"
    d.mkdir()
    glm = _doc("GLM 5.3", _ep("A", "a", None, "GLM 5.3"))
    sonnet = _doc("Claude Sonnet 5.5", _ep("Anthropic", "anthropic", None, "Claude Sonnet 5.5"))
    fable = _doc("Claude Fable 5.2", _ep("Anthropic", "anthropic", None, "Claude Fable 5.2"))
    for slug, doc in (("moonshotai/kimi-k3", KIMI), ("z-ai/glm-5.3", glm),
                      ("anthropic/claude-sonnet-5.5", sonnet), ("anthropic/claude-fable-5.2", fable)):
        (d / f"{slug.replace('/', '__')}.json").write_text(json.dumps(doc))
    zdr = tmp_path / "zdr.json"
    zdr.write_text(json.dumps(_zdr(("Wafer", "wafer/fp8", "Kimi K3"),
                                   ("Anthropic", "anthropic", "Claude Sonnet 5.5"))))
    return {"KC_FAKE_OR_ENDPOINTS_DIR": str(d), "KC_FAKE_OR_ZDR": str(zdr)}


def _models(tmp_path: Path, env_values: dict[str, str]):
    result, _ = _run_generator(tmp_path, with_vllm=False, with_openrouter=True,
                               env_values=env_values, extra_env=_fixtures(tmp_path))
    models, _ = _parse_dry_run(result.stdout)
    return {m["model_name"]: m for m in models}, result


def test_the_default_changes_nothing(tmp_path: Path) -> None:
    by_name, _ = _models(tmp_path, {})
    kimi = by_name["moonshotai/kimi-k3"]
    assert "extra_body" not in kimi["litellm_params"]
    assert kimi["litellm_params"]["model"].endswith(":floor")
    assert not any(k in kimi["model_info"] for k in ("kchat_run_policy", "kchat_zdr"))
    assert "z-ai/glm-5.3" in by_name


def test_strict_policy_filters_open_weight_models_and_leaves_closed_ones(tmp_path: Path) -> None:
    by_name, result = _models(tmp_path, {"OR_OPEN_WEIGHT_POLICY": "strict"})
    kimi = by_name["moonshotai/kimi-k3"]
    provider = kimi["litellm_params"]["extra_body"]["provider"]
    assert provider["zdr"] is True and provider["allow_fallbacks"] is True
    assert "unknown" not in provider["quantizations"] and "order" not in provider
    # the cheapest-first sort still applies among the endpoints the filter lets through
    assert kimi["litellm_params"]["model"].endswith(":floor")
    assert kimi["model_info"]["kchat_run_policy"] == "open_weight_strict"
    assert kimi["model_info"]["kchat_quantization"] == "fp8" and kimi["model_info"]["kchat_zdr"] is True
    assert "z-ai/glm-5.3" not in by_name
    assert "z-ai/glm-5.3 (no endpoint with disclosed quantization and ZDR)" in result.stderr
    assert "extra_body" not in by_name["anthropic/claude-sonnet-5.5"]["litellm_params"]


def test_a_pin_names_the_provider_and_drops_the_price_sort(tmp_path: Path) -> None:
    by_name, _ = _models(tmp_path, {"OR_OPEN_WEIGHT_POLICY": "strict",
                                    "OR_PROVIDER_PIN": "moonshotai/kimi-k3=Wafer"})
    kimi = by_name["moonshotai/kimi-k3"]
    assert kimi["litellm_params"]["extra_body"]["provider"]["order"] == ["Wafer"]
    assert kimi["litellm_params"]["extra_body"]["provider"]["allow_fallbacks"] is False
    assert not kimi["litellm_params"]["model"].endswith(":floor")
    assert kimi["model_info"]["kchat_provider_pinned"] == "Wafer"
    assert kimi["model_info"]["kchat_quantization"] == "fp8"


def test_an_ineligible_pin_is_ignored_with_a_warning(tmp_path: Path) -> None:
    by_name, result = _models(tmp_path, {"OR_OPEN_WEIGHT_POLICY": "strict",
                                         "OR_PROVIDER_PIN": "moonshotai/kimi-k3=KT"})
    provider = by_name["moonshotai/kimi-k3"]["litellm_params"]["extra_body"]["provider"]
    assert "order" not in provider
    assert "pinned provider 'KT'" in result.stderr


def test_a_closed_model_pin_with_policy_off(tmp_path: Path) -> None:
    by_name, _ = _models(tmp_path, {"OR_PROVIDER_PIN": "anthropic/*=Anthropic"})
    sonnet = by_name["anthropic/claude-sonnet-5.5"]
    assert sonnet["litellm_params"]["extra_body"]["provider"] == {
        "order": ["Anthropic"], "allow_fallbacks": False}
    assert sonnet["model_info"]["kchat_provider_pinned"] == "Anthropic"
    assert sonnet["model_info"]["kchat_zdr"] is True
    assert "extra_body" not in by_name["moonshotai/kimi-k3"]["litellm_params"]


def test_zdr_only_asks_for_zdr_and_leaves_out_models_without_it(tmp_path: Path) -> None:
    by_name, result = _models(tmp_path, {"OR_ZDR_ONLY": "1"})
    assert by_name["anthropic/claude-sonnet-5.5"]["litellm_params"]["extra_body"]["provider"] == {"zdr": True}
    assert by_name["anthropic/claude-sonnet-5.5"]["model_info"]["kchat_zdr"] is True
    assert "anthropic/claude-fable-5.2" not in by_name
    assert "anthropic/claude-fable-5.2 (no ZDR endpoint)" in result.stderr
    assert "run policy left out" in result.stderr


@pytest.mark.parametrize("value", ["", "0"])
def test_zdr_only_is_off_unless_one(tmp_path: Path, value: str) -> None:
    by_name, _ = _models(tmp_path, {"OR_ZDR_ONLY": value})
    assert "anthropic/claude-fable-5.2" in by_name


def test_a_self_hosted_model_declares_how_it_runs(tmp_path: Path) -> None:
    """Engine and KV cache are read from the server; the weights' quantization is declared."""
    result, _ = _run_generator(
        tmp_path, with_vllm=True, with_openrouter=False,
        env_values={"KC_QWEN27B_QUANTIZATION": "NVFP4", "KC_QWEN27B_KV_CACHE": "bf16"},
    )
    models, _ = _parse_dry_run(result.stdout)
    for m in models:
        if m["model_name"] in ("local/qwen3.8-27b", "strict-local/qwen3.8-27b"):
            info = m["model_info"]
            assert info["kchat_quantization"] == "NVFP4"
            # the server's own answer beats the declaration
            assert info["kchat_kv_cache"] == "fp8"
            assert info["kchat_engine"] == "vLLM 0.33.0"
            break
    else:
        raise AssertionError("no local qwen3.8-27b entry")


def test_nothing_is_declared_when_nothing_is_known(tmp_path: Path) -> None:
    result, _ = _run_generator(tmp_path, with_vllm=False, with_openrouter=False,
                               include_all_classes=False)
    models, _ = _parse_dry_run(result.stdout)
    assert not any("kchat_quantization" in m["model_info"] for m in models)


def _script(tmp_path: Path, *args: str) -> str:
    import subprocess

    from .test_config_generation import BASH, _fake_curl

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_curl(bin_dir)
    catalogue = tmp_path / "catalogue.json"
    catalogue.write_text(json.dumps({"data": [{"id": "moonshotai/kimi-k3"}, {"id": "z-ai/glm-5.3"},
                                              {"id": "moonshotai/kimi-k3:free"}]}))
    import os
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
           "KLOUDCHAT_ENV_FILE": str(tmp_path / "none.env"),
           "KC_FAKE_OR_CATALOGUE": str(catalogue), **_fixtures(tmp_path)}
    done = subprocess.run([BASH, str(ROOT / "scripts" / "or-endpoints.sh"), *args], env=env,
                          cwd=ROOT, text=True, capture_output=True, check=False)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_the_helper_lists_eligible_endpoints_per_model(tmp_path: Path) -> None:
    out = _script(tmp_path, "moonshotai/*")
    assert "== moonshotai/kimi-k3" in out and "Wafer" in out and "fp8" in out
    assert "KT" not in out and ":free" not in out


def test_the_summary_counts_eligible_endpoints(tmp_path: Path) -> None:
    out = _script(tmp_path, "--summary")
    line = next(ln for ln in out.splitlines() if ln.startswith("moonshotai/kimi-k3 "))
    assert "1 eligible" in line
    glm = next(ln for ln in out.splitlines() if ln.startswith("z-ai/glm-5.3 "))
    assert "0 eligible" in glm and "left out" in glm
