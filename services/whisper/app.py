"""OpenAI-compatible whisper server (faster-whisper backend).

POST /v1/audio/transcriptions  — multipart file + form fields. Response uses the
same schema as OpenAI: {"text": "..."}. Accepts `language` / `prompt` / `response_format`.

Model is loaded on first call and stays resident in memory (WhisperModel instance
is lazy). WHISPER_DEVICE=auto = GPU when CUDA is available, else CPU.
WHISPER_COMPUTE_TYPE = float16 (GPU default) / int8 (CPU). If the GPU/ct2 doesn't
support the configured value, falls back to int8 automatically at load time.
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper import WhisperModel

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
LOG = logging.getLogger("whisper")

MODEL_NAME    = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE        = os.getenv("WHISPER_DEVICE", "auto")
COMPUTE_TYPE  = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
DOWNLOAD_ROOT = os.getenv("HF_HOME", "/var/lib/whisper")

_model: Optional[WhisperModel] = None
_model_lock = threading.Lock()


def _get_model() -> WhisperModel:
    """Lazy-load — downloads weights + loads onto GPU on first call (cached after).
    Double-checked locking avoids duplicate init on concurrent first calls."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                LOG.info("Loading WhisperModel(%s, device=%s, compute_type=%s)",
                         MODEL_NAME, DEVICE, COMPUTE_TYPE)
                try:
                    _model = WhisperModel(
                        MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE,
                        download_root=DOWNLOAD_ROOT,
                    )
                except Exception as e:
                    # GPU/ct2 doesn't support compute_type (e.g. float16/int8_float16 on some cards) → fall back to int8.
                    if COMPUTE_TYPE == "int8":
                        raise
                    LOG.warning("compute_type=%s failed to load (%s) — falling back to int8", COMPUTE_TYPE, e)
                    _model = WhisperModel(
                        MODEL_NAME, device=DEVICE, compute_type="int8",
                        download_root=DOWNLOAD_ROOT,
                    )
    return _model


app = FastAPI(title="KloudChat Whisper", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),         # OpenAI-compat field — ignored (server is single-model)
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
) -> Response:
    # save to a tempfile so ffmpeg can read it directly — faster-whisper has no stream API.
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        m = _get_model()
        segments, info = m.transcribe(
            tmp_path,
            language=language,
            initial_prompt=prompt,
            temperature=temperature or 0.0,
            vad_filter=True,
        )
        text = "".join(seg.text for seg in segments).strip()
    except Exception as e:
        LOG.exception("transcribe failed")
        raise HTTPException(500, f"transcription failed: {e}") from e
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if response_format == "text":
        return PlainTextResponse(text)
    return JSONResponse({"text": text, "language": info.language})
