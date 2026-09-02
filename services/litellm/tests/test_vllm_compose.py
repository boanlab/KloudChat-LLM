"""Invariants of the GPU-node compose file."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
COMPOSE = ROOT / "docker-compose.vllm.yml"
MODELS_YAML = ROOT / "scheduler" / "models.yaml"


@pytest.fixture(scope="module")
def services() -> dict:
    doc = yaml.safe_load(COMPOSE.read_text()) or {}
    return doc.get("services") or {}


def test_every_model_in_the_catalogue_has_a_service_here(services: dict) -> None:
    """Every models.yaml entry has the compose service the applier will start."""
    from scheduler import registry

    for spec in registry.load(MODELS_YAML):
        assert spec.service in services, (
            f"{spec.id} is in the catalogue but {spec.service} is not a service "
            "in docker-compose.vllm.yml — the scheduler would place it and the "
            "node would have nothing to start"
        )


def test_devices_are_selected_through_the_runtime_variable(services: dict) -> None:
    """NVIDIA_VISIBLE_DEVICES only: CUDA_VISIBLE_DEVICES has no "all" value and fails engine init on GB10."""
    for name, svc in services.items():
        env = svc.get("environment") or {}
        assert "CUDA_VISIBLE_DEVICES" not in env, (
            f"{name}: CUDA_VISIBLE_DEVICES fails engine init on GB10 — "
            "use NVIDIA_VISIBLE_DEVICES"
        )


def test_every_vllm_service_can_be_told_which_cards_to_use(services: dict) -> None:
    """Every vLLM service exposes NVIDIA_VISIBLE_DEVICES with an "all" default."""
    for name, svc in services.items():
        if not name.startswith("vllm-"):
            continue
        env = svc.get("environment") or {}
        value = env.get("NVIDIA_VISIBLE_DEVICES")
        assert value, f"{name}: no NVIDIA_VISIBLE_DEVICES, so the scheduler cannot pin it"
        assert value.endswith(":-all}"), (
            f"{name}: default must be 'all'. An empty or ordinal default takes cards "
            f"away from a node the scheduler has not assigned, got {value!r}"
        )


def test_every_vllm_service_declares_a_gpu_reservation(services: dict) -> None:
    for name, svc in services.items():
        if not name.startswith("vllm-"):
            continue
        devices = (((svc.get("deploy") or {}).get("resources") or {})
                   .get("reservations") or {}).get("devices") or []
        assert any(d.get("driver") == "nvidia" for d in devices), \
            f"{name}: no nvidia device reservation"
