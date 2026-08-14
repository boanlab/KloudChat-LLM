# vLLM plus pytest: torch._dynamo lazy-imports it while tracing, and engine init
# fails when it is missing
ARG BASE_IMAGE=vllm/vllm-openai:nightly-aarch64
FROM ${BASE_IMAGE}

RUN pip install --no-cache-dir pytest

# Avoids the GB10 (sm_121) MLA decode kernel exceeding its shared-memory limit.
# The build fails loudly if the patch anchor disappears upstream.
COPY patches/patch_triton_mla_num_stages.py /tmp/patch_triton_mla.py
RUN python3 /tmp/patch_triton_mla.py && rm -f /tmp/patch_triton_mla.py
