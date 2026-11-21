"""GPU reservations in the node compose files follow the per-service device variables."""

from __future__ import annotations

import pathlib

import yaml

from scheduler import registry

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _services(name: str) -> dict:
    return yaml.safe_load((ROOT / name).read_text())["services"]


def _gpu_request(service: dict) -> dict:
    (request,) = service["deploy"]["resources"]["reservations"]["devices"]
    return request


def _assert_pinned_by(service: dict, value: str) -> None:
    # Docker's nvidia driver sets NVIDIA_VISIBLE_DEVICES from the reservation,
    # after the environment: count: all would hand every card to the container.
    request = _gpu_request(service)
    assert "count" not in request
    assert request["device_ids"] == [value]
    assert service["environment"]["NVIDIA_VISIBLE_DEVICES"] == value


def test_each_scheduled_service_reserves_the_cards_the_scheduler_writes():
    services = _services("docker-compose.vllm.yml")
    specs = [s for s in registry.load(ROOT / "scheduler" / "models.yaml") if not s.is_image]
    assert specs
    for spec in specs:
        _assert_pinned_by(services[spec.service], f"${{{spec.env_prefix}_DEVICES:-all}}")


def test_each_image_model_reserves_its_own_cards():
    services = _services("docker-compose.vllm-omni.yml")
    for service, key in [
        ("omni-z-image", "ZIMAGE"),
        ("omni-qwen-image", "QWENIMAGE"),
        ("omni-qwen-image-edit", "QWENIMAGEEDIT"),
    ]:
        _assert_pinned_by(
            services[service],
            f"${{VLLM_OMNI_{key}_DEVICES:-${{VLLM_OMNI_DEVICES:-all}}}}",
        )


def test_each_catalogued_image_model_is_pinned_by_the_variable_the_scheduler_writes():
    """The applier writes {env_prefix}_DEVICES; the omni service must read that one."""
    services = _services("docker-compose.vllm-omni.yml")
    images = [s for s in registry.load(ROOT / "scheduler" / "models.yaml") if s.is_image]
    assert {s.service for s in images} == set(services)
    for spec in images:
        _assert_pinned_by(
            services[spec.service],
            f"${{{spec.env_prefix}_DEVICES:-${{VLLM_OMNI_DEVICES:-all}}}}",
        )
