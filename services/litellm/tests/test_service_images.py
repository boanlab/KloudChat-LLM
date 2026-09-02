"""Build and start-up invariants for the compose stack."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
COMPOSE = ROOT / "docker-compose.yml"

#: Tags naming a stream rather than a release.
MOVING_TAGS = {"main", "master", "latest", "nightly", "edge", "dev"}


@pytest.fixture(scope="module")
def services() -> dict:
    doc = yaml.safe_load(COMPOSE.read_text()) or {}
    return doc.get("services") or {}


def _from_lines(dockerfile: Path) -> list[str]:
    return [
        line.split(maxsplit=1)[1].split(" AS ")[0].strip()
        for line in dockerfile.read_text().splitlines()
        if re.match(r"^\s*FROM\s+", line, re.IGNORECASE)
    ]


@pytest.mark.parametrize(
    "dockerfile",
    sorted((ROOT / "services").glob("*/Dockerfile")),
    ids=lambda p: p.parent.name,
)
def test_a_moving_base_tag_carries_a_digest(dockerfile: Path) -> None:
    """A base image on a moving tag must be pinned by digest."""
    for ref in _from_lines(dockerfile):
        if ref.startswith("$"):
            continue  # vLLM's ${BASE_IMAGE}; install-vllm.sh records its digest
        if "@sha256:" in ref:
            continue
        tag = ref.rsplit(":", 1)[1] if ":" in ref.rsplit("/", 1)[-1] else "latest"
        assert tag not in MOVING_TAGS, (
            f"{dockerfile.parent.name}: `{ref}` names a stream, not a release — "
            "pin it with @sha256: or name a version"
        )


def test_everything_waited_on_declares_a_start_period(services: dict) -> None:
    """Every service another one waits on (service_healthy) has a healthcheck start_period."""
    waited_on: set[str] = set()
    for svc in services.values():
        depends = svc.get("depends_on")
        if isinstance(depends, dict):
            waited_on |= {
                name
                for name, cond in depends.items()
                if (cond or {}).get("condition") == "service_healthy"
            }

    assert waited_on, "no service_healthy dependencies found — did the file move?"
    for name in sorted(waited_on):
        healthcheck = services[name].get("healthcheck") or {}
        assert healthcheck.get("start_period"), (
            f"{name}: another service waits on its health, so its first probes "
            "need a start_period — otherwise one early failure aborts the start"
        )
