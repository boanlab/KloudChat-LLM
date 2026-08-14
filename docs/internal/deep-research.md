# Deep research — internals and deployment

Deployment and configuration of the **deep-research sidecar (LDR)**.

## How it works

Deep research is the one MCP that runs as a **separate sidecar container**. The
other MCPs are stdio servers spawned as child processes by the UI's API; only
this one speaks HTTP.

- The playwright wheel is incompatible with the alpine-based image, so it is
  built from **debian-slim** instead.
- `mcp-proxy` wraps the internal stdio server as streamable-http at
  `http://deep-research:8081/mcp`.
- It is registered in the UI's connector catalogue as `deep-research` over HTTP,
  and users install it from the connector screen.

Internally it is Local Deep Research (LDR): ReAct with iterative reasoning, using
SearXNG's science tab (arXiv, Scholar, OpenAlex, Crossref, PubMed) to search
academic sources and the web in one pass.

## Environment

LDR settings live under `deep-research.environment` in `docker-compose.yml` as
`LDR_*`.

| Variable | Value |
|---|---|
| `LDR_LLM_PROVIDER` | `openai_endpoint` (through LiteLLM) |
| `LDR_LLM_MODEL` | `${DEEP_RESEARCH_MODEL:-local/glm-4.7-flash}` — iterative search, so the cheap floor model is the default |
| `LDR_LLM_OPENAI_ENDPOINT_URL` | `${DEEP_RESEARCH_LLM_URL:-http://litellm:8000/v1}` |
| `LDR_LLM_OPENAI_ENDPOINT_API_KEY` | `${LITELLM_MASTER_KEY}` |
| `LDR_SEARCH_TOOL` | `searxng` (instance URL in `LDR_SEARCH_ENGINE_WEB_SEARXNG_DEFAULT_PARAMS_INSTANCE_URL`) |
| `LDR_SEARCH_ITERATIONS` | `2` — bounds run time |
| `LDR_SEARCH_QUESTIONS_PER_ITERATION` | `1` |

> A heavy query (arXiv and web retries, iterations, generation) can exceed 30
> minutes, so the connector's MCP timeout is set to 60 minutes.

## Deployment

- Image: `${KLOUDCHAT_IMAGE_NS}/kloudchat-deep-research`, built from
  `services/deep-research/Dockerfile`.
- Starts with the main stack through `setup.sh all` or `docker compose up -d`,
  after `searxng`.
- Quality depends on whatever `DEEP_RESEARCH_MODEL` points at. Without a local
  vLLM, LiteLLM sends the same name to OpenRouter.

## See also

- [Tools](../tools.md) · [Model configuration](../models.md)
