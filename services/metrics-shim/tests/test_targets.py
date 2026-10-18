"""Targets come from the endpoint settings; image servers read as vLLM."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from targets import discover, image_text  # noqa: E402

ENV = {
    "VLLM_QWEN27B_URL": "http://172.29.52.2:8001",
    "VLLM_QWENFLASH_URL": "http://172.29.52.3:8004,http://172.29.52.5:8004/",
    "VLLM_BGEM3_URL": "http://172.29.52.2:8003",
    "WHISPER_URLS": "http://172.29.52.2:9000",
    "IMAGEGEN_BACKENDS": "local/z-image-turbo=http://172.29.52.5:8091,local/qwen-image-2512=http://172.29.52.5:8092",
    "NODES_VLLM": "boan@10.40.0.254,boan@10.40.0.253",
}


def test_every_endpoint_and_node_becomes_one_target():
    found = {t.name: t for t in discover(ENV)}
    assert {n for n, t in found.items() if t.kind == "vllm"} == {
        "vllm-172-29-52-2-8001",
        "vllm-172-29-52-3-8004",
        "vllm-172-29-52-5-8004",
        "vllm-172-29-52-2-8003",
        "vllm-172-29-52-2-9000",
    }
    assert found["image-172-29-52-5-8091"].model == "local/z-image-turbo"
    assert found["image-172-29-52-5-8091"].url == "http://172.29.52.5:8091"
    assert [t.name for t in found.values() if t.kind == "gpu"] == [
        "gpu-172-29-52-2",
        "gpu-172-29-52-3",
        "gpu-172-29-52-5",
        "gpu-10-40-0-254",
        "gpu-10-40-0-253",
    ]
    assert found["gpu-10-40-0-253"].url == "http://10.40.0.253:9835"


def test_a_node_listed_twice_is_one_target():
    env = {"VLLM_QWEN27B_URL": "http://gpu-1:8001", "VLLM_RERANK_URL": "gpu-1:8009", "NODES_VLLM": "u@gpu-1"}
    assert [t.name for t in discover(env)] == ["vllm-gpu-1-8001", "vllm-gpu-1-8009", "gpu-gpu-1"]


def test_no_settings_no_targets():
    assert discover({}) == []
    assert discover({"VLLM_QWEN27B_URL": " , ", "IMAGEGEN_BACKENDS": "broken"}) == []


def test_the_exporter_port_can_be_moved():
    env = {"NODES_VLLM": "u@gpu-9", "GPU_EXPORTER_PORT": "9900"}
    assert discover(env)[0].url == "http://gpu-9:9900"


OMNI = """\
# HELP http_request_duration_seconds Latency
http_request_duration_seconds_bucket{handler="/v1/images/generations",le="5.0",method="POST"} 1.0
http_request_duration_seconds_bucket{handler="/v1/images/generations",le="+Inf",method="POST"} 3.0
http_request_duration_seconds_count{handler="/v1/images/generations",method="POST"} 3.0
http_request_duration_seconds_sum{handler="/v1/images/generations",method="POST"} 21.5
http_request_duration_seconds_count{handler="/health",method="GET"} 99.0
process_cpu_seconds_total 12.0
"""


def test_an_image_server_reads_as_one_picture_at_a_time():
    text = image_text("local/z-image-turbo", OMNI, 3)
    m = 'model_name="local/z-image-turbo"'
    assert f"vllm:num_requests_running{{{m}}} 1" in text
    assert f"vllm:num_requests_waiting{{{m}}} 2" in text
    assert "_bucket" not in text
    assert f"vllm:e2e_request_latency_seconds_count{{{m}}} 3.0" in text
    assert f"vllm:e2e_request_latency_seconds_sum{{{m}}} 21.5" in text
    assert "99.0" not in text and "process_cpu" not in text


def test_an_image_server_without_counts_still_names_its_model():
    assert image_text("local/qwen-image-2512", None, None) == (
        'vllm:num_requests_running{model_name="local/qwen-image-2512"} 0\n'
    )
    assert "running" in image_text("m", "", 0) and "waiting{" in image_text("m", "", 0)
