# Deep research: internals and deployment

Deployment and configuration of the deep-research sidecar (Local Deep
Research, LDR).

## How it works

Deep research is the one MCP that runs as a separate sidecar container; it
speaks HTTP, while the UI's other MCPs are stdio child processes.

- Image built from `python:3.12-slim-bookworm` (`services/deep-research/Dockerfile`),
  with `local-deep-research[mcp]==1.6.11` and `mcp-proxy`.
- `mcp-proxy --pass-environment` wraps the stdio `ldr-mcp` server as
  streamable-http on port 8081; the gateway exposes it at `/tools/research/mcp`.
- `patches/patch_ldr_mcp_iterations.py` widens the `iterations` argument of
  the research tools to accept a string, so a model that sends `"2"` is not
  rejected by schema validation.
- The UI registers it as the `deep-research` HTTP connector.

Internally LDR runs iterative search over SearXNG (general and science
engines) and calls the model through LiteLLM.

## Environment

Set under `deep-research.environment` in `docker-compose.yml` as `LDR_*`.

| Variable | Value |
|---|---|
| `LDR_LLM_PROVIDER` | `openai_endpoint` (through LiteLLM) |
| `LDR_LLM_MODEL` | `${DEEP_RESEARCH_MODEL:-local/qwen3.5-122b-a10b}` |
| `LDR_LLM_OPENAI_ENDPOINT_URL` | `${DEEP_RESEARCH_LLM_URL:-http://litellm:8000/v1}` |
| `LDR_LLM_OPENAI_ENDPOINT_API_KEY` | `${LITELLM_MASTER_KEY}` |
| `LDR_SEARCH_TOOL` | `searxng` (`LDR_SEARCH_ENGINE_WEB_SEARXNG_DEFAULT_PARAMS_INSTANCE_URL=http://searxng:8080`) |
| `LDR_SEARCH_ITERATIONS` | `2`, bounds run time |
| `LDR_SEARCH_QUESTIONS_PER_ITERATION` | `1` |

A heavy query can exceed 30 minutes; the UI's connector timeout is set
accordingly.

## Deployment

- Image `${KLOUDCHAT_IMAGE_NS}/kloudchat-deep-research`, profile `tools`,
  started after `searxng`.
- Quality follows `DEEP_RESEARCH_MODEL`. Without a local vLLM, LiteLLM serves
  the same name from OpenRouter.

## See also

- [Tools](../tools.md) · [Model configuration](../models.md)
