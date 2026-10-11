"""Text from page images, on CPU, inside the stack.

PaddleOCR's PP-OCRv5 detector and Korean recogniser (Apache-2.0), run through
RapidOCR on ONNX Runtime. The model files are baked into the image at build;
nothing is downloaded at run time and nothing leaves the host.

    POST /ocr     body: a PDF, TIFF, PNG, JPEG, WEBP, BMP or GIF
                  ?page=N (1-based; a PDF page or a TIFF frame), &dpi=, &image=1
                  → {width, height, lines:[{text, conf, box}], tables:[box],
                     confidence, text, pages, image?}
    POST /render  same body and page → the page as JPEG (for a vision model)
    GET  /health

A box is `[x0, y0, x1, y1]` in pixels of the rendered page; `dpi` says how the
page was rendered, so a caller can map boxes back to PDF points (×72/dpi).
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import layout
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response

LOG = logging.getLogger("ocr-shim")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

#: Request body cap. A scanned PDF of a few hundred pages fits; more is refused.
MAX_BODY = int(float(os.getenv("OCR_MAX_BODY_MB", "64")) * 1024 * 1024)
#: Pixels of one rendered or decoded page. A4 at 300 dpi is ~8.7M; an image claiming
#: more is a decompression bomb, not a page.
MAX_PIXELS = int(os.getenv("OCR_MAX_PIXELS", "40000000"))
#: Pages or frames in one document the service will open at all.
MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "2000"))
DEFAULT_DPI = int(os.getenv("OCR_DPI", "200"))
MAX_DPI = 300
#: Longest side of the picture handed back for a vision model.
IMAGE_SIDE = int(os.getenv("OCR_IMAGE_SIDE", "1600"))
#: Pages recognised at once; each uses OCR_THREADS cores.
CONCURRENCY = max(1, int(os.getenv("OCR_CONCURRENCY", "1")))
THREADS = int(os.getenv("OCR_THREADS", "4"))
TIMEOUT_SEC = float(os.getenv("OCR_TIMEOUT_SEC", "120"))
#: Requests let in at once, running or waiting; past that a request is refused with 503
#: before its body is read, so a queue of large documents never sits in memory.
ADMIT = CONCURRENCY + max(0, int(os.getenv("OCR_QUEUE", "4")))
#: Lines below this recogniser score are dropped as noise.
MIN_SCORE = float(os.getenv("OCR_MIN_SCORE", "0.5"))

ENGINE_NAME = "rapidocr/PP-OCRv5 korean"

_GATE = asyncio.Semaphore(CONCURRENCY)
_admitted = 0
_ENGINE: Any = None


def engine() -> Any:
    """The recogniser, built once. Model files come from the image, not the network."""
    global _ENGINE
    if _ENGINE is None:
        from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

        _ENGINE = RapidOCR(
            params={
                "Global.use_cls": False,
                "Global.text_score": MIN_SCORE,
                "Global.log_level": "warning",
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Det.model_type": ModelType.MOBILE,
                "Det.lang_type": LangDet.CH,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.lang_type": LangRec.KOREAN,
                "EngineConfig.onnxruntime.intra_op_num_threads": THREADS,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )
    return _ENGINE


@asynccontextmanager
async def _lifespan(_: FastAPI):
    # Load the models before the first request, so a cold page does not time out.
    await asyncio.to_thread(warm)
    LOG.info("ocr-shim ready: %s, %d page(s) at a time, %d threads", ENGINE_NAME, CONCURRENCY, THREADS)
    yield


app = FastAPI(title="KloudChat OCR Shim", lifespan=_lifespan)


class Refused(Exception):
    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


async def _body(request: Request) -> bytes:
    """The request body, refused with 413 past MAX_BODY without reading the rest."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY:
        raise HTTPException(status_code=413, detail="too_large")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY:
            raise HTTPException(status_code=413, detail="too_large")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(status_code=400, detail="empty")
    return data


def _render_pdf(data: bytes, page: int, dpi: int) -> tuple[Any, int, int]:
    """`(image, pages, dpi used)`: one PDF page as RGB pixels, scaled down if it would
    exceed MAX_PIXELS."""
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise Refused(422, "unreadable_pdf") from exc
    try:
        pages = len(pdf)
        if pages > MAX_PAGES:
            raise Refused(413, "too_many_pages")
        if not 1 <= page <= pages:
            raise Refused(422, "no_such_page")
        sheet = pdf[page - 1]
        try:
            w_pt, h_pt = sheet.get_size()
            if w_pt <= 0 or h_pt <= 0:
                raise Refused(422, "unreadable_pdf")
            scale = dpi / 72
            if w_pt * h_pt * scale * scale > MAX_PIXELS:
                scale = math.sqrt(MAX_PIXELS / (w_pt * h_pt))
            used = max(1, int(round(scale * 72)))
            bitmap = sheet.render(scale=scale, grayscale=False)
            try:
                image = bitmap.to_pil().convert("RGB")
            finally:
                bitmap.close()
        finally:
            sheet.close()
    finally:
        pdf.close()
    return image, pages, used


def count_frames(picture: Any, cap: int) -> int:
    """Frames in a picture, counted by seeking but never past `cap` + 1: Pillow's `n_frames`
    walks every frame of a GIF or TIFF, however many a crafted file claims."""
    frames = 0
    while frames <= cap:
        try:
            picture.seek(frames)
        except EOFError:
            break
        frames += 1
    return frames


def _open_image(data: bytes, page: int) -> tuple[Any, int]:
    """`(image, frames)`: one frame of a picture, size checked before any decoding."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        picture = Image.open(io.BytesIO(data))
        frames = count_frames(picture, MAX_PAGES)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise Refused(413, "too_many_pixels") from exc
    except Exception as exc:  # noqa: BLE001 — Pillow raises many types for bad input
        raise Refused(422, "unreadable_image") from exc
    if frames > MAX_PAGES:
        raise Refused(413, "too_many_pages")
    if not 1 <= page <= frames:
        raise Refused(422, "no_such_page")
    try:
        picture.seek(page - 1)
        w, h = picture.size
        if w * h > MAX_PIXELS:
            raise Refused(413, "too_many_pixels")
        image = picture.convert("RGB")
    except Refused:
        raise
    except Exception as exc:  # noqa: BLE001
        raise Refused(422, "unreadable_image") from exc
    return image, frames


def load_page(data: bytes, page: int, dpi: int) -> tuple[Any, int, int]:
    """`(image, pages, dpi)` for any accepted body."""
    kind = layout.sniff(data)
    if kind is None:
        raise Refused(415, "unsupported")
    if kind == "pdf":
        return _render_pdf(data, page, dpi)
    image, frames = _open_image(data, page)
    return image, frames, dpi


def tables(image: Any) -> list[list[float]]:
    """Ruled tables: regions where long horizontal and vertical strokes cross."""
    import cv2
    import numpy as np

    gray = np.asarray(image.convert("L"))
    h, w = gray.shape
    # Scan speckle breaks the threshold into dots that read as neither line; a median
    # pass removes it and keeps ruled strokes.
    gray = cv2.medianBlur(gray, 3)
    ink = cv2.adaptiveThreshold(
        255 - gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 25), 1))
    )
    vertical = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 60)))
    )
    grid = cv2.dilate(cv2.add(horizontal, vertical), np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[list[float]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw * bh < w * h * 0.01 or bw < w * 0.2 or bh < 20:
            continue
        crossings = cv2.bitwise_and(
            horizontal[y : y + bh, x : x + bw], vertical[y : y + bh, x : x + bw]
        )
        joints, _ = cv2.findContours(crossings, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # A box drawn around a paragraph has four corners; a table has more joints.
        if len(joints) >= 6:
            found.append([float(x), float(y), float(x + bw), float(y + bh)])
    return sorted(found, key=lambda box: box[1])


def recognise(image: Any) -> list[dict]:
    """Lines with their score and box, in no particular order; each also carries its
    position on the page turned level (`layout.level`), from the tilt of its long lines."""
    import math

    import numpy as np

    result = engine()(np.asarray(image))
    out: list[dict] = []
    if result is None or result.boxes is None:
        return out
    angles: list[float] = []
    for quad, text, score in zip(result.boxes, result.txts or (), result.scores or (), strict=False):
        xs = [float(point[0]) for point in quad]
        ys = [float(point[1]) for point in quad]
        if not str(text).strip():
            continue
        (x0, y0), (x1, y1) = quad[0], quad[1]
        run, rise = float(x1) - float(x0), float(y1) - float(y0)
        if run > 3 * (max(ys) - min(ys)) and run > 0:
            angles.append(math.atan2(rise, run))
        out.append(
            {
                "text": str(text),
                "conf": round(float(score), 4),
                "box": [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)],
            }
        )
    if angles:
        angles.sort()
        layout.level(out, angles[len(angles) // 2], image.size[0], image.size[1])
    return out


def _jpeg(image: Any, side: int) -> bytes:
    """The page for a vision model: longest side `side`, JPEG (scan noise makes PNG
    several times larger for no gain in legibility)."""
    from PIL import Image

    shrunk = image
    longest = max(image.size)
    if longest > side:
        ratio = side / longest
        shrunk = image.resize(
            (max(1, int(image.size[0] * ratio)), max(1, int(image.size[1] * ratio))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    shrunk.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


def read_page(data: bytes, page: int, dpi: int, want_image: bool) -> dict:
    started = time.monotonic()
    image, pages, used = load_page(data, page, dpi)
    width, height = image.size
    lines = recognise(image)
    boxes = tables(image)
    body: dict[str, Any] = {
        "engine": ENGINE_NAME,
        "page": page,
        "pages": pages,
        "dpi": used,
        "width": width,
        "height": height,
        "lines": lines,
        "tables": boxes,
        "confidence": layout.mean_confidence(lines),
        "text": layout.page_text(lines, boxes, width),
    }
    if want_image:
        body["image"] = base64.b64encode(_jpeg(image, IMAGE_SIDE)).decode()
        body["image_mime"] = "image/jpeg"
    body["seconds"] = round(time.monotonic() - started, 2)
    return body


async def _run(fn, *args):
    """`fn` in a worker thread, at most CONCURRENCY at a time. A page that runs past the time
    limit is answered with 504, but its slot stays taken until the thread really ends: a thread
    cannot be stopped, and releasing early would let timed-out work pile up past the limit."""
    await _GATE.acquire()
    work = asyncio.ensure_future(asyncio.to_thread(fn, *args))
    work.add_done_callback(lambda _: _GATE.release())
    try:
        return await asyncio.wait_for(asyncio.shield(work), timeout=TIMEOUT_SEC)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="timeout") from exc
    except Refused as exc:
        raise HTTPException(status_code=exc.status, detail=exc.reason) from exc


async def _admitted_body(request: Request) -> bytes:
    """The body of a request let in under ADMIT; refused with 503 before reading otherwise.
    The caller releases the admission with `_leave()`."""
    global _admitted
    if _admitted >= ADMIT:
        raise HTTPException(status_code=503, detail="busy")
    _admitted += 1
    try:
        return await _body(request)
    except BaseException:
        _leave()
        raise


def _leave() -> None:
    global _admitted
    _admitted = max(0, _admitted - 1)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "engine": ENGINE_NAME}


@app.post("/ocr")
async def ocr(
    request: Request,
    page: int = Query(1, ge=1),
    dpi: int = Query(DEFAULT_DPI, ge=72, le=MAX_DPI),
    image: int = Query(0, ge=0, le=1),
) -> dict:
    data = await _admitted_body(request)
    try:
        return await _run(read_page, data, page, dpi, bool(image))
    finally:
        _leave()


def _render(data: bytes, page: int, dpi: int) -> bytes:
    picture, _, _ = load_page(data, page, dpi)
    return _jpeg(picture, IMAGE_SIDE)


@app.post("/render")
async def render(
    request: Request,
    page: int = Query(1, ge=1),
    dpi: int = Query(DEFAULT_DPI, ge=72, le=MAX_DPI),
) -> Response:
    data = await _admitted_body(request)
    try:
        return Response(await _run(_render, data, page, dpi), media_type="image/jpeg")
    finally:
        _leave()


def warm() -> None:
    """Load every model by reading a drawn line: RapidOCR opens (and on a first run
    fetches) each model only when a page first needs it."""
    from PIL import Image, ImageDraw, ImageFont

    sample = Image.new("RGB", (480, 96), "white")
    ImageDraw.Draw(sample).text((16, 24), "OCR 2025 12", fill="black", font=ImageFont.load_default(size=40))
    recognise(sample)


