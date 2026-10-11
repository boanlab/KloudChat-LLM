"""Reading order and plain-text layout for recognised lines. Pure Python, no models.

A line is `{"text", "conf", "box": [x0, y0, x1, y1]}` in page pixels, optionally with
`"at": [x, y]`, its middle on the page turned level (a scan tilted a degree or two puts
the two ends of one row at different heights; reading order uses the level position).
The page's text is built from them: rows top to bottom, two columns read left then
right, and the lines inside a ruled table joined as table rows.
"""

from __future__ import annotations

from statistics import median
from typing import Iterable

#: First bytes → kind. TIFF has two byte orders; WEBP is RIFF....WEBP.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "pdf"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
)


def sniff(data: bytes) -> str | None:
    """`pdf`, `png`, `jpeg`, `tiff`, `gif`, `bmp`, `webp`, or None."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    for magic, kind in _MAGIC:
        if data.startswith(magic):
            return kind
    return None


def _height(line: dict) -> float:
    x0, y0, x1, y1 = line["box"]
    return max(1.0, y1 - y0)


def _center_y(line: dict) -> float:
    if "at" in line:
        return line["at"][1]
    return (line["box"][1] + line["box"][3]) / 2


def _left(line: dict) -> float:
    """The line's left edge, on the levelled page when the tilt is known."""
    if "at" in line:
        return line["at"][0] - (line["box"][2] - line["box"][0]) / 2
    return line["box"][0]


def level(lines: list[dict], angle: float, width: float, height: float) -> None:
    """Adds `at` to each line: its middle rotated by -`angle` (radians) about the page centre."""
    import math

    cos, sin = math.cos(-angle), math.sin(-angle)
    cx, cy = width / 2, height / 2
    for line in lines:
        x = (line["box"][0] + line["box"][2]) / 2 - cx
        y = (line["box"][1] + line["box"][3]) / 2 - cy
        line["at"] = [round(x * cos - y * sin + cx, 1), round(x * sin + y * cos + cy, 1)]


def rows(lines: list[dict]) -> list[list[dict]]:
    """Lines grouped into rows: a line joins the row whose middle is within half a
    line height of its own; each row is ordered left to right."""
    if not lines:
        return []
    tall = median(_height(line) for line in lines)
    grouped: list[list[dict]] = []
    for line in sorted(lines, key=_center_y):
        if grouped and abs(_center_y(line) - _center_y(grouped[-1][0])) <= tall * 0.5:
            grouped[-1].append(line)
        else:
            grouped.append([line])
    return [sorted(row, key=_left) for row in grouped]


def gutter(lines: list[dict], width: float) -> float | None:
    """The x of an empty vertical band between 35% and 65% of the page that splits the
    narrow lines into two columns of at least four lines each, or None."""
    if width <= 0:
        return None
    narrow = [line for line in lines if (line["box"][2] - line["box"][0]) < width * 0.48]
    if len(narrow) < 8:
        return None
    # Columns of prose fill most of their half; the cells of an undetected table do not.
    if median(line["box"][2] - line["box"][0] for line in narrow) < width * 0.25:
        return None
    step = max(1, int(width // 200))
    best: tuple[int, float] | None = None
    for x in range(int(width * 0.35), int(width * 0.65), step):
        if any(line["box"][0] < x < line["box"][2] for line in narrow):
            continue
        left = sum(1 for line in narrow if line["box"][2] <= x)
        right = sum(1 for line in narrow if line["box"][0] >= x)
        if left >= 4 and right >= 4:
            balance = min(left, right)
            if best is None or balance > best[0]:
                best = (balance, float(x))
    return best[1] if best else None


def inside(line: dict, box: Iterable[float]) -> bool:
    """Whether the line's middle lies within `box`."""
    x0, y0, x1, y1 = box
    cx = (line["box"][0] + line["box"][2]) / 2
    cy = _center_y(line)
    return x0 <= cx <= x1 and y0 <= cy <= y1


def _row_text(row: list[dict]) -> str:
    return " ".join(line["text"].strip() for line in row if line["text"].strip())


def _table_text(lines: list[dict]) -> str:
    """Table lines as `| a | b |` rows."""
    out = []
    for row in rows(lines):
        cells = [line["text"].strip() for line in row if line["text"].strip()]
        if cells:
            out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _flow(lines: list[dict], width: float) -> list[tuple[float, str]]:
    """`(top, text)` blocks for lines outside tables: full-width rows in place, and
    between them a two-column stretch read left column first."""
    split = gutter(lines, width)
    if split is None:
        return [(row[0]["box"][1], _row_text(row)) for row in rows(lines)]
    blocks: list[tuple[float, str]] = []
    stretch: list[dict] = []

    def close() -> None:
        if not stretch:
            return
        left = [line for line in stretch if line["box"][2] <= split]
        right = [line for line in stretch if line["box"][2] > split]
        text = "\n".join(_row_text(row) for row in rows(left) + rows(right))
        blocks.append((min(line["box"][1] for line in stretch), text))
        stretch.clear()

    for row in rows(lines):
        spans = any(line["box"][0] < split < line["box"][2] for line in row)
        if spans:
            close()
            blocks.append((row[0]["box"][1], _row_text(row)))
        else:
            stretch.extend(row)
    close()
    return blocks


def page_text(lines: list[dict], tables: list[list[float]], width: float) -> str:
    """The page as plain text in reading order, tables as `|` rows."""
    in_table: dict[int, list[dict]] = {i: [] for i in range(len(tables))}
    loose: list[dict] = []
    for line in lines:
        for i, box in enumerate(tables):
            if inside(line, box):
                in_table[i].append(line)
                break
        else:
            loose.append(line)
    blocks = _flow(loose, width)
    for i, box in enumerate(tables):
        if in_table[i]:
            blocks.append((box[1], _table_text(in_table[i])))
    blocks.sort(key=lambda block: block[0])
    return "\n".join(text for _, text in blocks if text).strip()


def mean_confidence(lines: list[dict]) -> float | None:
    """Recognition confidence weighted by each line's length; None with no text."""
    weight = sum(len(line["text"].strip()) for line in lines)
    if weight == 0:
        return None
    return sum(line["conf"] * len(line["text"].strip()) for line in lines) / weight


__all__ = ["gutter", "inside", "mean_confidence", "page_text", "rows", "sniff"]
