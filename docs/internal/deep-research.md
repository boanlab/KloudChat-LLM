# Deep research: internals and deployment

Deployment and configuration of the deep-research sidecar (Local Deep
Research, LDR).

## How it works

Deep research is an MCP server that runs as a sidecar container and speaks
HTTP.

- Image built from `python:3.12-slim-bookworm` (`services/deep-research/Dockerfile`),
  with `local-deep-research[mcp]==1.6.11` and `mcp-proxy`.
- `mcp-proxy --pass-environment` wraps the stdio `ldr-mcp` server as
  streamable-http on port 8081; the gateway exposes it at `/tools/research/mcp`.
- `patches/patch_ldr_mcp_iterations.py` widens the `iterations` argument of
  the research tools to accept a string, so a model that sends `"2"` is not
  rejected by schema validation.
- The UI registers it as an HTTP MCP connector.

LDR runs iterative search over SearXNG (through `search-shim`) and calls the
model through LiteLLM.

## Environment

Set under `deep-research.environment` in `docker-compose.yml` as `LDR_*`.

| Variable | Value |
|---|---|
| `LDR_LLM_PROVIDER` | `openai_endpoint` (through LiteLLM) |
| `LDR_LLM_MODEL` | `${DEEP_RESEARCH_MODEL:-local/qwen3.8-27b}` |
| `LDR_LLM_OPENAI_ENDPOINT_URL` | `${DEEP_RESEARCH_LLM_URL:-http://litellm:8000/v1}` |
| `LDR_LLM_OPENAI_ENDPOINT_API_KEY` | `${LITELLM_MASTER_KEY}` |
| `LDR_SEARCH_TOOL` | `searxng`, with `LDR_SEARCH_ENGINE_WEB_SEARXNG_DEFAULT_PARAMS_INSTANCE_URL=http://search-shim:8080` |
| `LDR_SEARCH_ITERATIONS` | `2`, bounds run time |
| `LDR_SEARCH_QUESTIONS_PER_ITERATION` | `1` |

A run takes minutes to tens of minutes. Deep research runs on
`local/qwen3.8-27b`, whose deployment carries a 3600 s LiteLLM request
timeout for this reason.

## Deployment

- Image `${KLOUDCHAT_IMAGE_NS}/kloudchat-deep-research`, profile `tools`,
  started after `search-shim`.
- Quality follows `DEEP_RESEARCH_MODEL`. Without a local vLLM there is no
  `local/qwen3.8-27b` route; set the variable to a model the install serves
  (`qwen/qwen3.8-27b` with an OpenRouter key).

## See also

- [Tools](../tools.md) · [Model configuration](../models.md)
