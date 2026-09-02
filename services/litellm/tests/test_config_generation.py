from __future__ import annotations

import os
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
                printf '%s\n' '{"data":[{"id":"local/qwen3.6-35b","max_model_len":65536}]}' ;;
              *big.test*/v1/models)
                printf '%s\n' '{"data":[{"id":"local/qwen3.5-122b-a10b","max_model_len":131072}]}' ;;
              *bge.test*/v1/models)
                printf '%s\n' '{"data":[{"id":"local/bge-m3","max_model_len":8192}]}' ;;
              *whisper.test*/v1/models)
                printf '%s\n' '{"data":[{"id":"local/whisper-large-v3","max_model_len":448}]}' ;;
              *openrouter.ai*) printf '%s\n' '{"data":[]}' ;;
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
) -> tuple[subprocess.CompletedProcess[str], Path]:
    _require_associative_array_bash()
    env_file = tmp_path / ".env"
    config_file = tmp_path / "config.yaml"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_curl(bin_dir)

    values = {
        "OPENROUTER_API_KEY": "test-openrouter-key" if with_openrouter else "",
        "OPENAI_API_KEY": "test-openai-key" if include_all_classes else "",
        "VLLM_QWEN35B_URL": "http://qwen.test:8000" if with_vllm else "",
        "VLLM_QWEN122B_URL": "http://big.test:8000" if include_all_classes else "",
        "VLLM_BGEM3_URL": "http://bge.test:8000" if include_all_classes else "",
        "WHISPER_URLS": (
            whisper_urls if whisper_urls is not None
            else ("" if include_all_classes else "http://whisper.test:9000")
        ),
    }
    env_file.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    config_file.write_text(_base_config(store_prompts))

    env = os.environ.copy()
    env.update(
        {
            "KLOUDCHAT_ENV_FILE": str(env_file),
            "KLOUDCHAT_LITELLM_CONFIG_FILE": str(config_file),
            "PATH": f"{bin_dir}:{env['PATH']}",
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

    normal = by_name.get("local/qwen3.6-35b")
    assert (normal is not None) is (normal_boundary is not None)
    if normal is not None:
        info = normal["model_info"]
        assert info["kchat_data_boundary"] == normal_boundary
        assert info["kchat_strict_local"] is False
        assert info["kchat_privacy_only"] is False

    strict = by_name.get("strict-local/qwen3.6-35b")
    assert (strict is not None) is has_strict
    if strict is not None:
        assert strict["litellm_params"]["model"] == "hosted_vllm/local/qwen3.6-35b"
        assert strict["model_info"]["kchat_data_boundary"] == "self_hosted"
        assert strict["model_info"]["kchat_strict_local"] is True
        assert strict["model_info"]["kchat_privacy_only"] is True

    # The OpenRouter slug is the hidden failover twin behind a local deployment,
    # or the visible only route without one; never both.
    twin = by_name.get("qwen/qwen3.6-35b-a3b")
    assert (twin is not None) is with_openrouter
    if twin is not None:
        assert twin["model_info"]["kchat_data_boundary"] == "external"
        assert twin["model_info"].get("kchat_hidden", False) is with_vllm
        # A visible route declares the tool support the local alias would have.
        if not with_vllm:
            assert twin["model_info"]["supports_function_calling"] is True
            assert twin["model_info"]["supports_tool_choice"] is True

    fallback_sources = {source for mapping in fallbacks for source in mapping}
    assert ("local/qwen3.6-35b" in fallback_sources) is has_fallback
    assert not any(source.startswith("strict-local/") for source in fallback_sources)


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
    assert by_name["local/bge-m3"]["model_info"]["kchat_data_boundary"] == "self_hosted"
    assert by_name["text-embedding-3-small"]["model_info"]["kchat_data_boundary"] == "external"
    assert by_name["strict-local/qwen3.5-122b-a10b"]["model_info"]["kchat_strict_local"] is True


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


STT_OR_MODEL = "mistralai/voxtral-small-24b-2507"


def _stt_registered(output: str) -> bool:
    models, _ = _parse_dry_run(output)
    return any(m.get("model_name") == STT_OR_MODEL for m in models)


def test_a_placed_transcription_backend_keeps_stt_local(tmp_path: Path) -> None:
    """A live transcription backend suppresses the OpenRouter STT route."""
    result, _ = _run_generator(
        tmp_path, with_vllm=True, with_openrouter=True,
        whisper_urls="http://whisper.test:9000",
    )
    assert not _stt_registered(result.stdout)


def test_a_placed_but_dead_transcription_backend_still_delegates(tmp_path: Path) -> None:
    """A placed but unreachable transcription backend still registers OpenRouter STT."""
    result, _ = _run_generator(
        tmp_path, with_vllm=True, with_openrouter=True,
        whisper_urls="http://whisper-down.test:9000",
    )
    assert _stt_registered(result.stdout)


def test_no_transcription_backend_at_all_delegates(tmp_path: Path) -> None:
    result, _ = _run_generator(
        tmp_path, with_vllm=True, with_openrouter=True, whisper_urls="",
    )
    assert _stt_registered(result.stdout)


def test_every_chat_model_in_the_catalogue_is_registered() -> None:
    """Every generate-runner model in models.yaml has an emit_brain line and its twin named."""
    catalogue = yaml.safe_load((ROOT / "scheduler" / "models.yaml").read_text())
    generator = GENERATOR.read_text()

    for entry in catalogue["models"]:
        if (entry.get("runner") or "generate") != "generate":
            continue
        model_id = entry["id"]
        assert f'emit_brain "{model_id}"' in generator, (
            f"{model_id} is in the catalogue but gen-litellm-config.sh never "
            "registers it — it would deploy and be unreachable"
        )
        twin = entry.get("openrouter")
        if twin:
            assert twin in generator, (
                f"{model_id} declares the OpenRouter twin {twin}, which the "
                "generator does not name — its fallback would be wrong"
            )
