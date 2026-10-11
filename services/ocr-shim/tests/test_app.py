"""The service against real pages. Needs the image's dependencies and models; skipped
where they are absent (the lint job), run inside the image:

    docker run --rm -v $PWD/services/ocr-shim/tests:/app/tests kloudchat-ocr-shim \
        python -m pytest -q tests
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("rapidocr")
pytest.importorskip("pypdfium2")
import app as shim  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

client = TestClient(shim.app)


def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _page(text: str = "INVOICE 2025 TOTAL 4200") -> Image.Image:
    page = Image.new("RGB", (1200, 400), "white")
    ImageDraw.Draw(page).text((60, 150), text, fill="black", font=_font(56))
    return page


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _pdf(images: list[Image.Image]) -> bytes:
    buffer = io.BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:], resolution=150)
    return buffer.getvalue()


def test_a_picture_is_read_into_lines_with_boxes():
    body = client.post("/ocr", content=_png(_page())).json()
    assert "2025" in body["text"] and "4200" in body["text"]
    first = body["lines"][0]
    assert set(first) == {"text", "conf", "box"} and len(first["box"]) == 4
    assert 0 < body["confidence"] <= 1
    assert "image" not in body


def test_a_pdf_page_is_rendered_then_read():
    pdf = _pdf([_page("FIRST PAGE 111"), _page("SECOND PAGE 222")])
    body = client.post("/ocr?page=2&image=1", content=pdf).json()
    assert body["pages"] == 2 and "222" in body["text"]
    assert body["image_mime"] == "image/jpeg" and body["image"]
    assert client.post("/ocr?page=3", content=pdf).status_code == 422


def test_render_returns_the_page_picture():
    response = client.post("/render", content=_pdf([_page()]))
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(response.content)).size[0] <= shim.IMAGE_SIDE


def test_what_is_not_a_page_is_refused():
    assert client.post("/ocr", content=b"PK\x03\x04 zip").status_code == 415
    assert client.post("/ocr", content=b"").status_code == 400
    assert client.post("/ocr", content=b"%PDF-1.7 broken").status_code == 422


def test_a_picture_claiming_too_many_pixels_is_refused_before_decoding(monkeypatch):
    monkeypatch.setattr(shim, "MAX_PIXELS", 100_000)
    response = client.post("/ocr", content=_png(Image.new("L", (1000, 1000), 255)))
    assert response.status_code == 413


def test_a_large_pdf_page_is_rendered_smaller_not_refused(monkeypatch):
    monkeypatch.setattr(shim, "MAX_PIXELS", 400_000)
    body = client.post("/ocr?dpi=300", content=_pdf([_page()])).json()
    assert body["width"] * body["height"] <= 400_000 * 1.05
    assert body["dpi"] < 300


def test_a_body_past_the_cap_is_refused(monkeypatch):
    monkeypatch.setattr(shim, "MAX_BODY", 1000)
    assert client.post("/ocr", content=b"%PDF-" + b"0" * 2000).status_code == 413


def test_a_ruled_table_is_found():
    page = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(page)
    for y in (200, 300, 400, 500):
        draw.line((100, y, 1100, y), fill="black", width=3)
    for x in (100, 450, 800, 1100):
        draw.line((x, 200, x, 500), fill="black", width=3)
    for row, y in enumerate((230, 330, 430)):
        for col, x in enumerate((130, 480, 830)):
            draw.text((x, y), f"R{row}C{col} {row * 3 + col}", fill="black", font=_font(36))
    body = client.post("/ocr", content=_png(page)).json()
    assert len(body["tables"]) == 1
    assert body["text"].count("|") >= 8
