# Transcription backend for amd64 GPU nodes — the whisper service in
# docker-compose.vllm.yml. amd64 only: aarch64 ctranslate2 wheels are CPU-only,
# so arm64 nodes keep no backend and delegate STT to OpenRouter.
# GPU inference through faster-whisper (ctranslate2) needs cuBLAS and cuDNN.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
      "faster-whisper>=1.0" "fastapi>=0.110" "uvicorn[standard]>=0.27" "python-multipart>=0.0.9"

WORKDIR /app
COPY app.py /app/app.py

# Weights are cached on a host volume (/var/lib/whisper). Low-VRAM nodes should
# set WHISPER_COMPUTE_TYPE=int8.
ENV HF_HOME=/var/lib/whisper \
    WHISPER_MODEL=large-v3 \
    WHISPER_DEVICE=auto \
    WHISPER_COMPUTE_TYPE=float16 \
    WHISPER_PORT=9000
EXPOSE 9000

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${WHISPER_PORT}"]
