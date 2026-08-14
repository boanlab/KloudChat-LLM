# local-deep-research MCP sidecar, on debian-slim for playwright wheel
# compatibility. mcp-proxy wraps the stdio ldr-mcp server as streamable-http.
FROM python:3.14-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "local-deep-research[mcp]==1.6.11" \
    "mcp-proxy"

# Widen iterations on quick/detailed_research to Optional[Union[int, str]]:
# a model that sends the string "2" would otherwise fail schema validation and
# degrade to plain search.
COPY patches/patch_ldr_mcp_iterations.py /tmp/patch_ldr_mcp_iterations.py
RUN python3 /tmp/patch_ldr_mcp_iterations.py

ENV PYTHONUNBUFFERED=1
EXPOSE 8081

# --pass-environment is required for the LDR_* variables to reach the server
CMD ["mcp-proxy", "--pass-environment", "--port=8081", "--host=0.0.0.0", "--transport=streamablehttp", "ldr-mcp"]
