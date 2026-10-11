"""Reading order and table layout of recognised lines (no models needed)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import layout  # noqa: E402


def line(text: str, x0: float, y0: float, x1: float, y1: float, conf: float = 0.9) -> dict:
    return {"text": text, "conf": conf, "box": [x0, y0, x1, y1]}


def test_kinds_are_told_by_their_first_bytes():
    assert layout.sniff(b"%PDF-1.7\n") == "pdf"
    assert layout.sniff(b"\x89PNG\r\n\x1a\n....") == "png"
    assert layout.sniff(b"\xff\xd8\xff\xe0") == "jpeg"
    assert layout.sniff(b"II*\x00....") == "tiff"
    assert layout.sniff(b"MM\x00*....") == "tiff"
    assert layout.sniff(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert layout.sniff(b"PK\x03\x04") is None
    assert layout.sniff(b"<svg") is None


def test_lines_on_one_row_read_left_to_right():
    lines = [line("둘째", 300, 100, 400, 130), line("첫째", 100, 102, 200, 128), line("다음 줄", 100, 160, 300, 190)]
    assert layout.page_text(lines, [], 1000) == "첫째 둘째\n다음 줄"


def test_two_columns_read_left_column_first():
    lines = [line("제목이 두 단에 걸친다", 100, 20, 900, 60)]
    for i in range(5):
        lines.append(line(f"왼{i}", 100, 100 + i * 40, 450, 130 + i * 40))
        lines.append(line(f"오{i}", 550, 100 + i * 40, 900, 130 + i * 40))
    text = layout.page_text(lines, [], 1000).splitlines()
    assert text[0] == "제목이 두 단에 걸친다"
    assert text[1:6] == [f"왼{i}" for i in range(5)]
    assert text[6:] == [f"오{i}" for i in range(5)]


def test_a_single_column_is_not_split():
    lines = [line(f"문장 {i} 이 페이지 폭을 거의 다 쓴다", 100, 100 + i * 40, 880, 130 + i * 40) for i in range(10)]
    assert layout.gutter(lines, 1000) is None
    assert layout.page_text(lines, [], 1000).splitlines()[3] == "문장 3 이 페이지 폭을 거의 다 쓴다"


def test_lines_in_a_ruled_table_become_rows():
    table = [100.0, 300.0, 900.0, 420.0]
    lines = [
        line("표 앞 문단", 100, 200, 600, 230),
        line("구분", 110, 310, 200, 340),
        line("금액", 500, 312, 600, 338),
        line("1학기", 110, 360, 200, 390),
        line("4,200,000원", 500, 361, 700, 389),
        line("표 뒤 문단", 100, 500, 600, 530),
    ]
    assert layout.page_text(lines, [table], 1000).splitlines() == [
        "표 앞 문단",
        "| 구분 | 금액 |",
        "| 1학기 | 4,200,000원 |",
        "표 뒤 문단",
    ]


def test_confidence_weighs_long_lines_more():
    lines = [line("가" * 30, 0, 0, 10, 10, 0.9), line("나", 0, 20, 10, 30, 0.1)]
    assert abs(layout.mean_confidence(lines) - (0.9 * 30 + 0.1) / 31) < 1e-9
    assert layout.mean_confidence([]) is None
