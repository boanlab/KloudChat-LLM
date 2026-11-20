"""Documents split into overlapping passages cut at natural breaks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chunking  # noqa: E402
from chunking import chunk_text  # noqa: E402


def test_empty_text_has_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n ") == []
    assert chunk_text(None) == []


def test_short_text_is_one_chunk():
    assert chunk_text("  one short passage.  ") == ["one short passage."]


def test_runs_of_blank_lines_collapse():
    assert chunk_text("a\n\n\n\n\nb") == ["a\n\nb"]


def test_long_text_is_cut_at_a_sentence_break_with_overlap(monkeypatch):
    monkeypatch.setattr(chunking, "CHUNK", 100)
    monkeypatch.setattr(chunking, "OVERLAP", 20)
    sentence = "This sentence is exactly forty chars. "
    text = sentence * 10
    pieces = chunk_text(text)
    assert len(pieces) > 1
    assert all(len(p) <= 100 for p in pieces)
    # Every cut but the last lands just before a sentence-ending ". ".
    assert all(p.endswith("forty chars") for p in pieces[:-1])
    # Consecutive pieces share text.
    for a, b in zip(pieces, pieces[1:], strict=False):
        assert b[:10] in a[-20:]


def test_a_break_in_the_front_half_is_ignored(monkeypatch):
    monkeypatch.setattr(chunking, "CHUNK", 100)
    monkeypatch.setattr(chunking, "OVERLAP", 10)
    text = "Title. " + "x" * 300
    pieces = chunk_text(text)
    assert len(pieces[0]) == 100


def test_text_past_the_ceiling_is_dropped(monkeypatch):
    monkeypatch.setattr(chunking, "MAX_CHARS", 50)
    pieces = chunk_text("y" * 500)
    assert pieces == ["y" * 50]


def test_chunking_always_advances(monkeypatch):
    monkeypatch.setattr(chunking, "CHUNK", 10)
    monkeypatch.setattr(chunking, "OVERLAP", 50)
    pieces = chunk_text("z" * 40)
    assert pieces and pieces[-1].endswith("z")
