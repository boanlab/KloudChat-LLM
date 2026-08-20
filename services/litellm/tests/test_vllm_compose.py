"""What the GPU-node compose file must not say.

Runtime facts that a unit test can still hold onto. The device-visibility one is
here because getting it wrong took a production model down: `CUDA_VISIBLE_DEVICES=0`
fails engine init on GB10 with `cudaErrorNotPermitted`, on a device that runs
fine when the variable is simply absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMPOSE = Path(__file__).resolve().parents[3] / "docker-compose.vllm.yml"


@pytest.fixture(scope="module")
def services() -> dict:
    doc = yaml.safe_load(COMPOSE.read_text()) or {}
    return doc.get("services") or {}


def test_devices_are_selected_through_the_runtime_variable(services: dict) -> None:
    """NVIDIA_VISIBLE_DEVICES, never CUDA_VISIBLE_DEVICES.

    Two reasons, and the second is the one that cost an outage. The runtime
    variable has a safe "every card" value; the CUDA one has no way to say that,
    and an empty value means *no* card — so a compose default cannot express
    "leave it alone". And on GB10, setting the CUDA one at all fails engine init.
    """
    for name, svc in services.items():
        env = svc.get("environment") or {}
        assert "CUDA_VISIBLE_DEVICES" not in env, (
            f"{name}: CUDA_VISIBLE_DEVICES fails engine init on GB10 — "
            "use NVIDIA_VISIBLE_DEVICES"
        )


def test_every_vllm_service_can_be_told_which_cards_to_use(services: dict) -> None:
    """A multi-card node needs per-service pinning, or two models land on card 0."""
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
