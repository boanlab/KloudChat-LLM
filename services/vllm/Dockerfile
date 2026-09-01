# vLLM plus pytest: torch._dynamo lazy-imports it while tracing, and engine init
# fails when it is missing
# No default: install-vllm.sh passes the pinned base from lib.sh. A default here
# would be a second place for the pin to live, and the one nobody updates.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN pip install --no-cache-dir pytest

# What `vllm[audio]` pulls in. /v1/audio/transcriptions decodes the upload before
# it reaches the model, and without these vllm-whisper starts and then fails every
# request. Named directly rather than through the extra: the base image already
# carries vllm, and installing `vllm[audio]` over it would move the pin.
RUN pip install --no-cache-dir librosa soundfile

# Avoids the GB10 (sm_121) MLA decode kernel exceeding its shared-memory limit.
# The build fails loudly if the patch anchor disappears upstream.
COPY patches/patch_triton_mla_num_stages.py /tmp/patch_triton_mla.py
RUN python3 /tmp/patch_triton_mla.py && rm -f /tmp/patch_triton_mla.py
