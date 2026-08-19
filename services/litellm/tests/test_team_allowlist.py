from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BASH = shutil.which("bash") or "bash"


def test_add_strict_preserves_restricted_team_allowlists(tmp_path: Path) -> None:
    version = subprocess.run(
        [BASH, "-c", 'printf "%s" "${BASH_VERSINFO[0]}"'],
        text=True,
        capture_output=True,
        check=False,
    )
    if version.returncode != 0 or int(version.stdout or "0") < 4:
        pytest.skip("manage.sh targets Linux and requires Bash 4+")

    env_file = tmp_path / ".env"
    env_file.write_text(
        "LITELLM_MASTER_KEY=test-master-key\n"
        "VLLM_QWEN35B_URL=http://qwen.test:8000\n"
        "VLLM_GLMFLASH_URL=\n"
        "OPENROUTER_API_KEY=\n"
    )
    capture = tmp_path / "updates.jsonl"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            url=""
            payload=""
            previous=""
            for arg in "$@"; do
              if [ "$previous" = "-d" ]; then payload="$arg"; fi
              case "$arg" in http://*|https://*) url="$arg" ;; esac
              previous="$arg"
            done
            case "$url" in
              */team/list)
                printf '%s\n' '[{"team_id":"restricted","team_alias":"restricted","models":["local/qwen3.6-35b","openai/gpt"]},{"team_id":"external","team_alias":"external","models":["openai/gpt"]},{"team_id":"unrestricted","team_alias":"unrestricted","models":[]}]' ;;
              */team/update)
                printf '%s\n' "$payload" >> "$KCHAT_TEST_CAPTURE"
                printf '%s\n' '{}' ;;
              *) printf '%s\n' '{}' ;;
            esac
            printf '%s\n' '200'
            """
        )
    )
    curl.chmod(curl.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env.update(
        {
            "ENV_FILE": str(env_file),
            "LITELLM_URL": "http://litellm.test",
            "KCHAT_TEST_CAPTURE": str(capture),
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )
    result = subprocess.run(
        [BASH, str(ROOT / "scripts" / "manage.sh"), "team", "add-strict"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    updates = [json.loads(line) for line in capture.read_text().splitlines()]
    assert updates == [
        {
            "team_id": "restricted",
            "models": [
                "local/qwen3.6-35b",
                "openai/gpt",
                "strict-local/qwen3.6-35b",
            ],
        }
    ]
    assert "1 updated, 2 unchanged, 0 failed" in result.stdout
