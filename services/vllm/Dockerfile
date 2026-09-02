# This repo's layer over the upstream vLLM image. No default base: install-vllm.sh
# passes the pin from lib.sh.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# torch._dynamo lazy-imports pytest while tracing; engine init fails without it
RUN pip install --no-cache-dir pytest

# Audio decoders for /v1/audio/transcriptions (what `vllm[audio]` adds, without
# moving the vllm pin)
RUN pip install --no-cache-dir librosa soundfile
