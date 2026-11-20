"""Chat-style image requests become vllm-omni generations and back."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shape  # noqa: E402


def test_backends_parse_name_equals_url_pairs():
    got = shape.parse_backends(" z-image-turbo=http://gpu-2:8092/ ,bad, local/flux2-klein-4b=http://gpu-2:8093")
    assert got == {"z-image-turbo": ["http://gpu-2:8092"], "flux2-klein-4b": ["http://gpu-2:8093"]}


def test_a_name_on_several_nodes_keeps_every_url_once():
    # A leading local/ names the same model
    got = shape.parse_backends(
        "z-image-turbo=http://gpu-1:8091,local/z-image-turbo=http://gpu-2:8091/,"
        "z-image-turbo=http://gpu-2:8091,z-image-turbo=http://gpu-3:8091,empty="
    )
    assert got == {"z-image-turbo": ["http://gpu-1:8091", "http://gpu-2:8091", "http://gpu-3:8091"]}


def test_the_aspect_picks_the_familys_size():
    assert shape.size_for("local/qwen-image-2512", "16:9") == "1664x928"
    assert shape.size_for("local/z-image-turbo", "9:16") == "768x1360"
    assert shape.size_for("local/flux2-klein-4b", "") == "1024x1024"
    assert shape.size_for("local/flux2-klein-4b", "7:5") == "1024x1024"


def test_every_size_is_a_multiple_of_sixteen():
    for family in shape.FAMILIES.values():
        for size in family["sizes"].values():
            w, h = (int(v) for v in size.split("x"))
            assert w % 16 == 0 and h % 16 == 0, size


def test_the_last_user_text_is_the_prompt():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": "a red fox"}, {"type": "image_url", "image_url": {}}]},
    ]
    assert shape.prompt_of(messages) == "a red fox"


def test_a_chat_request_becomes_a_generation_with_the_familys_sampling():
    body = {
        "model": "local/z-image-turbo",
        "modalities": ["image", "text"],
        "image_config": {"aspect_ratio": "4:3"},
        "messages": [{"role": "user", "content": "a lighthouse"}],
    }
    got = shape.generation_body("local/z-image-turbo", body)
    assert got["prompt"] == "a lighthouse"
    assert got["size"] == "1152x864"
    assert got["response_format"] == "b64_json"
    assert got["num_inference_steps"] == 9 and got["guidance_scale"] == 0.0


def test_the_reply_carries_the_image_where_kloudchat_reads_it():
    reply = shape.chat_reply("local/flux2-klein-4b", "QUJD")
    message = reply["choices"][0]["message"]
    assert message["images"][0]["image_url"]["url"] == "data:image/png;base64,QUJD"
    assert reply["usage"]["completion_tokens"] == 0


def _edit_body(*urls: str) -> dict:
    parts = [{"type": "text", "text": "make the sky pink"}]
    parts += [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    return {"model": "local/qwen-image-edit-2511", "seed": 7, "messages": [{"role": "user", "content": parts}]}


def test_the_edit_model_is_its_own_family_not_qwen_images():
    assert shape.is_edit("local/qwen-image-edit-2511")
    assert not shape.is_edit("local/qwen-image-2512")
    assert not shape.is_edit("local/z-image-turbo")


def test_pictures_are_read_from_data_urls_of_the_last_user_message():
    body = _edit_body("data:image/jpeg;base64,QUJD", "data:;base64,REVG")
    assert shape.images_of(body["messages"]) == [("image/jpeg", b"ABC"), ("image/png", b"DEF")]
    assert shape.images_of([{"role": "user", "content": "text only"}]) == []


@pytest.mark.parametrize("url", ["https://example.com/a.png", "data:image/png,raw", "data:image/png;base64,@@@", "data:image/png;base64,"])
def test_a_picture_that_is_not_a_base64_data_url_is_refused(url):
    with pytest.raises(ValueError):
        shape.images_of(_edit_body(url)["messages"])


def test_an_edit_request_keeps_the_input_size_and_the_cards_sampling():
    got = shape.edit_form("local/qwen-image-edit-2511", _edit_body("data:image/png;base64,QUJD"))
    assert got["prompt"] == "make the sky pink"
    assert got["size"] == "auto" and got["seed"] == "7"
    assert got["true_cfg_scale"] == "4.0" and got["guidance_scale"] == "1.0"
    assert got["num_inference_steps"] == "40"
