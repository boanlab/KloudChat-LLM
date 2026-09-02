# local-deep-research MCP server, exposed as streamable-http through mcp-proxy.
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "local-deep-research[mcp]==1.6.11" \
    "mcp-proxy"

# Numeric MCP params accept strings (models send "2"); see the patch.
COPY patches/patch_ldr_mcp_iterations.py /tmp/patch_ldr_mcp_iterations.py
RUN python3 /tmp/patch_ldr_mcp_iterations.py

ENV PYTHONUNBUFFERED=1
EXPOSE 8081

# --pass-environment forwards the LDR_* variables to the server
CMD ["mcp-proxy", "--pass-environment", "--port=8081", "--host=0.0.0.0", "--transport=streamablehttp", "ldr-mcp"]
