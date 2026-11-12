"""Uploads are capped, concurrent transcriptions bounded, and the form forwarded with our model name."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upload import Slots, TooLarge, rebuild_form  # noqa: E402

MODEL = "local/whisper-large-v3"


class File:
    """Starlette UploadFile stand-in."""

    def __init__(self, data: bytes, filename="a.m4a", content_type="audio/mp4", size=True):
        self.data = data
        self.filename = filename
        self.content_type = content_type
        self.size = len(data) if size else None
        self.reads: list[int] = []

    async def read(self, n: int = -1) -> bytes:
        self.reads.append(n)
        return self.data if n < 0 else self.data[:n]


class Form:
    def __init__(self, items):
        self.items = items

    def multi_items(self):
        return list(self.items)


def run(coro):
    return asyncio.run(coro)


def test_model_is_replaced_and_other_fields_kept():
    form = Form([("model", "whisper-1"), ("language", "ko"),
                 ("response_format", "verbose_json"), ("file", File(b"abc"))])
    files, fields = run(rebuild_form(form, MODEL, max_bytes=10))
    assert fields == {"model": MODEL, "language": "ko", "response_format": "verbose_json"}
    assert files == {"file": ("a.m4a", b"abc", "audio/mp4")}


def test_model_is_set_when_the_client_sent_none():
    _, fields = run(rebuild_form(Form([("file", File(b"x"))]), MODEL, max_bytes=10))
    assert fields["model"] == MODEL


def test_missing_content_type_defaults_to_octet_stream():
    files, _ = run(rebuild_form(Form([("file", File(b"x", content_type=None))]),
                                MODEL, max_bytes=10))
    assert files["file"][2] == "application/octet-stream"


def test_an_upload_at_the_ceiling_passes():
    files, _ = run(rebuild_form(Form([("file", File(b"x" * 10))]), MODEL, max_bytes=10))
    assert len(files["file"][1]) == 10


def test_an_upload_with_a_declared_size_past_the_ceiling_is_refused_unread():
    big = File(b"x" * 11)
    with pytest.raises(TooLarge):
        run(rebuild_form(Form([("file", big)]), MODEL, max_bytes=10))
    assert big.reads == []


def test_an_upload_without_a_size_is_read_only_one_byte_past_the_ceiling():
    big = File(b"x" * 1000, size=False)
    with pytest.raises(TooLarge):
        run(rebuild_form(Form([("file", big)]), MODEL, max_bytes=10))
    assert big.reads == [11]


def test_the_ceiling_covers_all_files_together():
    form = Form([("file", File(b"x" * 6)), ("extra", File(b"y" * 6, size=False))])
    with pytest.raises(TooLarge):
        run(rebuild_form(form, MODEL, max_bytes=10))


def test_slots_refuse_past_the_limit_and_free_on_give():
    slots = Slots(2)
    assert slots.take() and slots.take()
    assert not slots.take()
    slots.give()
    assert slots.take()
    slots.give()
    slots.give()
    slots.give()
    assert slots.used == 0
