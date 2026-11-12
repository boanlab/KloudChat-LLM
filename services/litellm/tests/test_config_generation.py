from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
GENERATOR = ROOT / "scripts" / "gen-litellm-config.sh"
FALLBACK_MARKER = "  # >>> KLOUDCHAT_FALLBACKS_START"
BASH = shutil.which("bash") or "bash"


def _require_associative_array_bash() -> None:
    version = subprocess.run(
        [BASH, "-c", 'printf "%s" "${BASH_VERSINFO[0]}"'],
        text=True,
        capture_output=True,
        check=False,
    )
    if version.returncode != 0 or int(version.stdout or "0") < 4:
        pytest.skip("gen-litellm-config.sh targets Linux and requires Bash 4+")


def _fake_curl(bin_dir: Path) -> None:
    curl = bin_dir / "curl"
    curl.write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            url=""
            for arg in "$@"; do
              case "$arg" in http://*|https://*) url="$arg" ;; esac
            done
            case "$url" in
              *qwen.test*/v1/models)
                printf '%s\n' '{"data":[{"id":"local/qwen3.8-27b","max_model_len":65536}]}' ;;
              *other.test*/v1/models)
                printf '%s\n' '{"data":[{"id":"Org/Other-7B","max_model_len":32768}]}' ;;
              *bge.test*/v1/models)
                printf '%s\n' '{"data":[{"id":"local/bge-m3","max_model_len":8192}]}' ;;
              *bge2.test*/v1/models)
                printf '%s\n' '{"data":[{"id":"BAAI/bge-m3","max_model_len":8192}]}' ;;
              *whisper.test*/v1/models)
                printf '%s\n' '{"data":[{"id":"local/whisper-large-v3","max_model_len":448}]}' ;;
              *openrouter.ai*)
                if [ -n "${KC_FAKE_OR_CATALOGUE:-}" ]; then cat "$KC_FAKE_OR_CATALOGUE"
                else printf '%s\n' '{"data":[]}'; fi ;;
              *) exit 22 ;;
            esac
            """
        )
    )
    curl.chmod(curl.stat().st_mode | stat.S_IXUSR)


def _base_config(store_prompts: str | None = "false") -> str:
    prompt_setting = (
        "" if store_prompts is None else f"  store_prompts_in_spend_logs: {store_prompts}\n"
    )
    return textwrap.dedent(
        f"""\
        model_list:
          # >>> KLOUDCHAT_AUTOGEN_START
          # <<< KLOUDCHAT_AUTOGEN_END
        router_settings:
          # >>> KLOUDCHAT_FALLBACKS_START
          fallbacks: []
          # <<< KLOUDCHAT_FALLBACKS_END
        general_settings:
          master_key: os.environ/LITELLM_MASTER_KEY
          store_model_in_db: false
        {prompt_setting.rstrip()}
        litellm_settings:
          drop_params: true
        """
    )


def _run_generator(
    tmp_path: Path,
    *,
    with_vllm: bool,
    with_openrouter: bool,
    include_all_classes: bool = False,
    dry_run: bool = True,
    store_prompts: str | None = "false",
    whisper_urls: str | None = None,
    extra_env: dict[str, str] | None = None,
    env_values: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    _require_associative_array_bash()
    env_file = tmp_path / ".env"
    config_file = tmp_path / "config.yaml"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_curl(bin_dir)

    values = {
        "OPENROUTER_API_KEY": "test-openrouter-key" if with_openrouter else "",
        "VLLM_ENDPOINTS": ",".join(
            (["qwen3.8-27b=http://qwen.test:8000"] if with_vllm else [])
            + (["bge-m3=http://bge.test:8000"] if include_all_classes else [])
        ),
        "WHISPER_URLS": (
            whisper_urls if whisper_urls is not None
            else ("" if include_all_classes else "http://whisper.test:9000")
        ),
    }
    values.update(env_values or {})
    env_file.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    config_file.write_text(_base_config(store_prompts))

    env = os.environ.copy()
    env.update(
        {
            "KLOUDCHAT_ENV_FILE": str(env_file),
            "KLOUDCHAT_LITELLM_CONFIG_FILE": str(config_file),
            "PATH": f"{bin_dir}:{env['PATH']}",
            **(extra_env or {}),
        }
    )
    args = [BASH, str(GENERATOR)]
    if dry_run:
        args.append("--dry-run")
    result = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result, config_file


def _parse_dry_run(output: str) -> tuple[list[dict], list[dict]]:
    section, fallback_section = output.split(FALLBACK_MARKER, 1)
    model_doc = yaml.safe_load("model_list:\n" + section) or {}
    router_doc = yaml.safe_load(
        "router_settings:\n" + FALLBACK_MARKER + fallback_section
    ) or {}
    models = model_doc.get("model_list") or []
    fallbacks = (router_doc.get("router_settings") or {}).get("fallbacks") or []
    return models, fallbacks


@pytest.mark.parametrize(
    ("with_vllm", "with_openrouter", "normal_boundary", "has_strict", "has_fallback"),
    [
        (False, False, None, False, False),
        # No vLLM URL: the model registers under its OpenRouter slug, not local/.
        (False, True, None, False, False),
        (True, False, "self_hosted", True, False),
        (True, True, "hybrid", True, True),
    ],
)
def test_local_and_strict_aliases_follow_deployment_topology(
    tmp_path: Path,
    with_vllm: bool,
    with_openrouter: bool,
    normal_boundary: str | None,
    has_strict: bool,
    has_fallback: bool,
) -> None:
    result, _ = _run_generator(
        tmp_path,
        with_vllm=with_vllm,
        with_openrouter=with_openrouter,
    )
    models, fallbacks = _parse_dry_run(result.stdout)
    by_name = {model["model_name"]: model for model in models}

    normal = by_name.get("local/qwen3.8-27b")
    assert (normal is not None) is (normal_boundary is not None)
    if normal is not None:
        info = normal["model_info"]
        assert info["kchat_data_boundary"] == normal_boundary
        assert info["kchat_strict_local"] is False
        assert info["kchat_privacy_only"] is False

    strict = by_name.get("strict-local/qwen3.8-27b")
    assert (strict is not None) is has_strict
    if strict is not None:
        assert strict["litellm_params"]["model"] == "hosted_vllm/local/qwen3.8-27b"
        assert strict["model_info"]["kchat_data_boundary"] == "self_hosted"
        assert strict["model_info"]["kchat_strict_local"] is True
        assert strict["model_info"]["kchat_privacy_only"] is True

    # The OpenRouter slug is the hidden failover twin behind a local deployment,
    # or the visible only route without one; never both.
    twin = by_name.get("qwen/qwen3.8-27b")
    assert (twin is not None) is with_openrouter
    if twin is not None:
        assert twin["model_info"]["kchat_data_boundary"] == "external"
        assert twin["model_info"].get("kchat_hidden", False) is with_vllm
        # A visible route declares the tool support the local alias would have.
        if not with_vllm:
            assert twin["model_info"]["supports_function_calling"] is True
            assert twin["model_info"]["supports_tool_choice"] is True

    fallback_sources = {source for mapping in fallbacks for source in mapping}
    assert ("local/qwen3.8-27b" in fallback_sources) is has_fallback
    assert not any(source.startswith("strict-local/") for source in fallback_sources)


def test_the_27b_deployment_defaults_to_a_presence_penalty(tmp_path: Path) -> None:
    """Callers that send none get 1.0 on the vLLM 27B; the OpenRouter twin is left alone."""
    result, _ = _run_generator(tmp_path, with_vllm=True, with_openrouter=True)
    models, _ = _parse_dry_run(result.stdout)
    by_name = {model["model_name"]: model for model in models}

    for alias in ("local/qwen3.8-27b", "strict-local/qwen3.8-27b"):
        assert by_name[alias]["litellm_params"]["presence_penalty"] == 1.0
    assert "presence_penalty" not in by_name["qwen/qwen3.8-27b"]["litellm_params"]


@pytest.mark.parametrize(("value", "expected"), [("", None), ("0.5", 0.5), ("high", None)])
def test_the_27b_presence_penalty_follows_the_override(
    tmp_path: Path, value: str, expected: float | None
) -> None:
    result, _ = _run_generator(
        tmp_path,
        with_vllm=True,
        with_openrouter=False,
        extra_env={"KC_QWEN27B_PRESENCE_PENALTY": value},
    )
    models, _ = _parse_dry_run(result.stdout)
    params = {m["model_name"]: m["litellm_params"] for m in models}["local/qwen3.8-27b"]
    assert params.get("presence_penalty") == expected


@pytest.mark.parametrize("with_vllm", [True, False])
def test_local_alias_never_carries_an_external_boundary(
    tmp_path: Path, with_vllm: bool
) -> None:
    """A local/ alias is never registered with an external data boundary."""
    result, _ = _run_generator(
        tmp_path, with_vllm=with_vllm, with_openrouter=True, include_all_classes=True
    )
    models, _ = _parse_dry_run(result.stdout)

    assert models
    external_locals = [
        model["model_name"]
        for model in models
        if model["model_name"].startswith("local/")
        and (model.get("model_info") or {}).get("kchat_data_boundary") == "external"
    ]
    assert external_locals == []


def test_every_generated_model_class_declares_its_boundary(tmp_path: Path) -> None:
    result, _ = _run_generator(
        tmp_path,
        with_vllm=True,
        with_openrouter=True,
        include_all_classes=True,
    )
    models, _ = _parse_dry_run(result.stdout)

    assert models
    for model in models:
        info = model.get("model_info") or {}
        assert info.get("kchat_data_boundary") in {"self_hosted", "hybrid", "external"}
        assert isinstance(info.get("kchat_strict_local"), bool)
        assert isinstance(info.get("kchat_privacy_only"), bool)

    by_name = {model["model_name"]: model for model in models}
    assert by_name["strict-local/bge-m3"]["model_info"]["kchat_data_boundary"] == "self_hosted"
    assert "text-embedding-3-small" not in by_name and "local/bge-m3" not in by_name
    assert by_name["strict-local/qwen3.8-27b"]["model_info"]["kchat_strict_local"] is True


def test_regeneration_disables_existing_prompt_storage_without_losing_config(
    tmp_path: Path,
) -> None:
    _, config_file = _run_generator(
        tmp_path,
        with_vllm=False,
        with_openrouter=False,
        dry_run=False,
        store_prompts="true",
    )
    generated = yaml.safe_load(config_file.read_text())

    assert generated["general_settings"]["store_prompts_in_spend_logs"] is False
    assert generated["litellm_settings"]["drop_params"] is True
    assert "model_list" in generated


def test_regeneration_adds_missing_prompt_storage_guard(tmp_path: Path) -> None:
    _, config_file = _run_generator(
        tmp_path,
        with_vllm=False,
        with_openrouter=False,
        dry_run=False,
        store_prompts=None,
    )

    generated = yaml.safe_load(config_file.read_text())

    assert generated["general_settings"]["store_prompts_in_spend_logs"] is False
    assert generated["litellm_settings"]["drop_params"] is True


def test_config_example_disables_prompt_and_response_storage() -> None:
    config = yaml.safe_load((ROOT / "services/litellm/config.yaml.example").read_text())

    assert config["general_settings"]["store_prompts_in_spend_logs"] is False


def test_a_transcription_backend_registers_strict_local_only(tmp_path: Path) -> None:
    """WHISPER_URLS set: strict-local/<id> behind whisper-shim, no local/ alias, no OpenRouter route."""
    result, _ = _run_generator(
        tmp_path, with_vllm=True, with_openrouter=True,
        whisper_urls="http://whisper.test:9000",
    )
    models, fallbacks = _parse_dry_run(result.stdout)
    stt = [
        m for m in models
        if (m.get("model_info") or {}).get("mode") == "audio_transcription"
        or "whisper" in m["model_name"] or "voxtral" in m["model_name"]
    ]
    assert [m["model_name"] for m in stt] == ["strict-local/whisper-large-v3"]
    entry = stt[0]
    assert entry["litellm_params"]["api_base"] == "http://whisper-shim:9000/v1"
    assert entry["litellm_params"]["model"] == "openai/whisper-large-v3"
    info = entry["model_info"]
    assert info["kchat_data_boundary"] == "self_hosted"
    assert info["kchat_strict_local"] is True and info["kchat_privacy_only"] is True
    assert not any("whisper" in k for mapping in fallbacks for k in mapping)


@pytest.mark.parametrize("whisper_urls", ["", " , "])
def test_no_transcription_backend_registers_no_audio_route_at_all(
    tmp_path: Path, whisper_urls: str
) -> None:
    """Nothing routes a recording anywhere, OpenRouter included."""
    result, _ = _run_generator(
        tmp_path, with_vllm=True, with_openrouter=True, whisper_urls=whisper_urls,
    )
    models, _ = _parse_dry_run(result.stdout)
    assert not [
        m["model_name"] for m in models
        if "voxtral" in m["model_name"] or "whisper" in m["model_name"]
        or (m.get("model_info") or {}).get("mode") == "audio_transcription"
    ]


def test_an_openrouter_model_that_takes_an_effort_says_so(tmp_path: Path) -> None:
    """Without supports_reasoning, drop_params discards a caller's reasoning_effort."""
    catalogue = tmp_path / "catalogue.json"
    catalogue.write_text(json.dumps({"data": [
        {"id": "z-ai/glm-5.3-flash", "pricing": {"prompt": "0", "completion": "0"},
         "supported_parameters": ["reasoning", "reasoning_effort", "tools"],
         "reasoning": {"mandatory": True, "supported_efforts": ["max", "high", "low"],
                       "default_effort": "max"}},
        {"id": "z-ai/glm-5.3", "pricing": {"prompt": "0", "completion": "0"},
         "supported_parameters": ["tools"]},
    ]}))
    result, _ = _run_generator(
        tmp_path, with_vllm=False, with_openrouter=True,
        extra_env={"KC_FAKE_OR_CATALOGUE": str(catalogue)},
    )
    models, _ = _parse_dry_run(result.stdout)
    by_name = {model["model_name"]: model["model_info"] for model in models}

    flash = by_name["z-ai/glm-5.3-flash"]
    assert flash["supports_reasoning"] is True
    assert flash["kchat_reasoning_efforts"] == ["max", "high", "low"]
    assert flash["kchat_reasoning_default"] == "max"
    assert flash["kchat_reasoning_mandatory"] is True
    # No effort in the catalogue: nothing claimed
    assert "supports_reasoning" not in by_name["z-ai/glm-5.3"]
    assert "kchat_reasoning_efforts" not in by_name["z-ai/glm-5.3"]


def test_each_self_hosted_image_backend_is_registered_inside_the_boundary(tmp_path: Path) -> None:
    """Every name in IMAGEGEN_BACKENDS becomes a strict-local image model behind the shim only."""
    result, _ = _run_generator(
        tmp_path, with_vllm=True, with_openrouter=True,
        env_values={"IMAGEGEN_BACKENDS": "z-image-turbo=http://gpu-2:8092, local/flux2-klein-4b=http://gpu-2:8093"},
    )
    models, _ = _parse_dry_run(result.stdout)
    shim = [m for m in models if "imagegen-shim" in str(m["litellm_params"].get("api_base"))]
    images = {m["model_name"]: m for m in shim}
    assert set(images) == {"strict-local/z-image-turbo", "strict-local/flux2-klein-4b"}
    # No local/ alias: nothing could route such a picture to OpenRouter
    assert not [m for m in models if m["model_name"] in {"local/z-image-turbo", "local/flux2-klein-4b"}]
    for name, entry in images.items():
        # The shim is addressed by the bare model name
        assert entry["litellm_params"]["model"] == f"openai/{name.split('/', 1)[1]}"
        assert entry["litellm_params"]["api_base"] == "http://imagegen-shim:9100/v1"
        info = entry["model_info"]
        assert info["mode"] == "image_generation"
        assert info["kchat_data_boundary"] == "self_hosted" and info["kchat_strict_local"] is True
        assert info["kchat_privacy_only"] is True


def test_an_image_model_on_several_nodes_is_registered_once(tmp_path: Path) -> None:
    result, _ = _run_generator(
        tmp_path, with_vllm=True, with_openrouter=True,
        env_values={"IMAGEGEN_BACKENDS": (
            "z-image-turbo=http://gpu-1:8091,local/z-image-turbo=http://gpu-2:8091,"
            "qwen-image-2512=http://gpu-1:8092,z-image-turbo=http://gpu-3:8091"
        )},
    )
    models, _ = _parse_dry_run(result.stdout)
    names = [m["model_name"] for m in models if "imagegen-shim" in str(m["litellm_params"].get("api_base"))]
    assert sorted(names) == ["strict-local/qwen-image-2512", "strict-local/z-image-turbo"]


def test_no_image_backend_registers_no_local_image_model(tmp_path: Path) -> None:
    result, _ = _run_generator(tmp_path, with_vllm=True, with_openrouter=True)
    models, _ = _parse_dry_run(result.stdout)
    assert not [m for m in models if "imagegen-shim" in str(m["litellm_params"].get("api_base"))]


def _catalogue() -> list[dict]:
    return yaml.safe_load((ROOT / "scheduler" / "models.yaml").read_text())["models"]


def _chat_entries() -> list[dict]:
    return [
        e for e in _catalogue()
        if (e.get("api") or {"generate": "chat"}.get(e.get("runner") or "generate")) == "chat"
    ]


def test_every_chat_model_in_the_catalogue_registers_from_its_endpoint(tmp_path: Path) -> None:
    """A catalogue chat model named in VLLM_ENDPOINTS gets both aliases, its cap and its twin."""
    entries = _chat_entries()
    endpoints = ",".join(f"{e['id']}=http://down-{i}.test:8000" for i, e in enumerate(entries))
    result, _ = _run_generator(
        tmp_path, with_vllm=False, with_openrouter=True,
        env_values={"VLLM_ENDPOINTS": endpoints},
    )
    models, fallbacks = _parse_dry_run(result.stdout)
    by_name = {m["model_name"]: m for m in models}
    sources = {k: v for mapping in fallbacks for k, v in mapping.items()}
    for e in entries:
        for alias in (f"local/{e['id']}", f"strict-local/{e['id']}"):
            entry = by_name[alias]
            assert entry["litellm_params"]["model"] == f"hosted_vllm/local/{e['id']}"
            assert entry["model_info"]["kchat_max_inflight"] == e.get("max_inflight", 256)
        if e.get("openrouter"):
            assert sources[f"local/{e['id']}"] == [e["openrouter"]]


def test_an_unlisted_model_found_at_a_bare_endpoint_registers_as_chat(tmp_path: Path) -> None:
    """A bare URL is asked what it serves; a name outside the catalogue keeps its served id."""
    result, _ = _run_generator(
        tmp_path, with_vllm=False, with_openrouter=True,
        env_values={"VLLM_ENDPOINTS": "http://other.test:8000, http://qwen.test:8000"},
    )
    models, fallbacks = _parse_dry_run(result.stdout)
    by_name = {m["model_name"]: m for m in models}

    other = by_name["local/Other-7B"]
    assert other["litellm_params"]["model"] == "hosted_vllm/Org/Other-7B"
    assert other["model_info"]["max_input_tokens"] == 32768 - 4096
    assert other["model_info"]["kchat_max_inflight"] == 256
    assert by_name["strict-local/Other-7B"]["model_info"]["kchat_strict_local"] is True
    assert "local/Other-7B" not in {k for mapping in fallbacks for k in mapping}
    # The catalogue model behind the other URL keeps its twin as fallback only
    assert by_name["local/qwen3.8-27b"]["litellm_params"]["presence_penalty"] == 1.0
    assert by_name["qwen/qwen3.8-27b"]["model_info"]["kchat_hidden"] is True


def test_an_unreachable_bare_endpoint_is_skipped_and_a_named_one_kept(tmp_path: Path) -> None:
    result, _ = _run_generator(
        tmp_path, with_vllm=False, with_openrouter=False,
        env_values={"VLLM_ENDPOINTS": "http://gone.test:8000,qwen3.8-flash=http://gone.test:8004"},
    )
    models, _ = _parse_dry_run(result.stdout)
    names = {m["model_name"] for m in models}
    assert "local/qwen3.8-flash" in names
    assert "gone.test:8000" in result.stderr


def test_the_schedulers_placements_register(tmp_path: Path) -> None:
    result, _ = _run_generator(
        tmp_path, with_vllm=False, with_openrouter=False,
        env_values={"VLLM_PLACED": "qwen3.8-27b=http://qwen.test:8000,bge-m3=http://bge.test:8000"},
    )
    models, _ = _parse_dry_run(result.stdout)
    by_name = {m["model_name"]: m for m in models}
    assert by_name["local/qwen3.8-27b"]["model_info"]["max_input_tokens"] == 65536 - 4096
    assert by_name["strict-local/bge-m3"]["model_info"]["mode"] == "embedding"


def test_every_env_the_generator_references_reaches_the_container() -> None:
    """`os.environ/X` in generated config must be passed to the LiteLLM service by compose."""
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    passed = set((compose["services"]["litellm"].get("environment") or {}).keys())
    referenced = set(re.findall(r"os\.environ/([A-Z0-9_]+)", GENERATOR.read_text()))
    missing = referenced - passed
    assert not missing, f"docker-compose.yml does not pass {sorted(missing)} to litellm"


def test_one_model_served_under_two_names_keeps_every_replica(tmp_path: Path) -> None:
    """Each server registers under its own served name, so no replica is dropped."""
    result, _ = _run_generator(
        tmp_path,
        with_vllm=False,
        with_openrouter=False,
        env_values={"VLLM_ENDPOINTS": "http://bge.test:8000,http://bge2.test:8000"},
    )
    models, _ = _parse_dry_run(result.stdout)
    served = sorted(
        m["litellm_params"]["model"] for m in models if m["model_name"] == "strict-local/bge-m3"
    )
    assert served == ["hosted_vllm/BAAI/bge-m3", "hosted_vllm/local/bge-m3"]
