# Model configuration

Which models are registered where, and how requests are routed to them.

> If you are bringing the stack up for the first time, start with the
> [README](../README.md).

## Catalogue: `lib.sh` is the source of truth

`scripts/lib.sh` holds the model setup.

| Variable | Role |
|---|---|
| `OPENAI_MODELS` / `ANTHROPIC_MODELS` / `GOOGLE_MODELS` / `XAI_MODELS` / `PERPLEXITY_MODELS` | Frontier tier, all through OpenRouter — no direct native APIs |
| `TENCENT_MODELS` / `DEEPSEEK_MODELS` / `ZAI_MODELS` / `XIAOMI_MODELS` / `MOONSHOTAI_MODELS` | Open-weight tier — 295B to 2.8T, far too large to self-host, which is exactly where renting beats owning |
| `VLLM_MODELS` | Local vLLM models (chat and floor). What each demands of a card is in `VLLM_MODEL_QUANT` and `VLLM_MODEL_WEIGHT_GB` |
| `OPENAI_EMBED_CATALOG` | OpenRouter embeddings (RAG fallback) |
| `MODEL_PRICE_IN_PM` / `MODEL_PRICE_OUT_PM` | USD per 1M tokens, for LiteLLM spend tracking |

**Pricing policy**

- **Commercial** — the OpenRouter catalogue price.
- **Local** (`local/*`) — **free (0)**. Self-hosted GPU, no per-token billing.
- **When the OpenRouter fallback fires** (node down or overloaded) — billed at
  the price of the deployment that actually served it.

  > LiteLLM computes cost against the deployment it fell back to, so local (0)
  > and OpenRouter (paid) separate themselves. The twin's price is baked into the
  > `emit_or_fallback` arguments in `gen-litellm-config.sh` — the fallback table
  > below carries the same numbers.

- **`text-embedding-3-small`** — through OpenRouter, so it is paid.

> Prices move, and the same model differs by provider. Verify against
> `curl https://openrouter.ai/api/v1/models` — the live catalogue, not a blog
> post — before trusting a billing report.

**Generated configuration**

`gen-litellm-config.sh` combines the definitions above with the environment and
regenerates the block between markers.

| Generator | Target | Marker |
|---|---|---|
| `gen-litellm-config.sh` | `services/litellm/config.yaml` | `KLOUDCHAT_AUTOGEN` |

The model picker reads LiteLLM's `/v1/models` directly, so there is nothing to
generate on the UI side.

## The model set

vLLM is the only local LLM backend, on two architectures:

- **amd64** — discrete cards (RTX 5090 / PRO 5000 / PRO 6000), standard image
  `vllm/vllm-openai:cu129-nightly`.
- **arm64** — GB10 with 128 GB of unified memory, `vllm/vllm-openai:nightly-aarch64`.

| Model (alias) | Container | Port | Quant | Role |
|---|---|---|---|---|
| `local/qwen3.6-35b` | `vllm-qwen35b` | 8001 | NVFP4 | Unified chat — conversation, artifacts, vision, deep research, coding |
| `local/glm-4.7-flash` | `vllm-glmflash` | 8002 | NVFP4 | Cheap-decode floor — titles, memory extraction, query rewriting, default chat (31.2B-A3B) |

**Why two models**

Qwen3.6-35B-A3B covers vision, a 262K context and agentic coding on its own, so
there is no reason to split by role. The one split that remains is **cost**: the
high-volume internal calls (titles, memory extraction, query rewriting) get their
own deployment so they do not eat the chat deployment's KV cache. Both are 3B
active MoE; the floor model is lighter because its context is shorter.

**Quantisation**

- Default is **NVFP4** (GB10 / RTX 5090 / PRO 5000 / PRO 6000).
- **RTX 4090** has no FP4 support and there is no int4 build of 35B-A3B to
  substitute, so `manage-vllm.sh up` refuses on that card
  ([gpu-memory.md](gpu-memory.md)).

**Parsers**

| Model | Tool parser | Reasoning parser | Notes |
|---|---|---|---|
| `qwen3.6-35b` | `qwen3_xml` | `qwen3` | Thinking is on by default upstream and turned off with `--default-chat-template-kwargs '{"enable_thinking": false}'`. Hybrid Gated-DeltaNet requires `--max-num-seqs` (cudagraph OOM without it) |
| `glm-4.7-flash` | `glm45` | `glm47` | Without the reasoning parser, chain-of-thought leaks into `content` |

## Free models

Whatever OpenRouter offers for free is queried at config-generation time and
registered. The list is not hard-coded because the free tier changes often — a
model that disappeared would still show in the picker and 404 on call. If the
query fails, nothing is added and the config is built from paid models alone.

The filter lives in `or_free_models` in `scripts/lib.sh`: zero input and output
price, text output, `:free` suffix. Guardrail models (content-safety, guard,
moderation) are excluded — they emit text but classify their input, so picking
one as a chat partner returns a verdict instead of an answer.

The UI hides models priced at 0 by default, as a guard against mistaking a
missing price for a free model. The `:free` suffix is the one exception, because
there the provider stated the price.

## Routing

### Commercial: a single OpenRouter path

| OpenRouter key | Result |
|---|---|
| Present | One route per model, named `<provider>/<id>` |
| Absent | Not registered |

`model_name` is canonical (`openai/gpt-5.6-sol`) while `litellm_params.model` is
`openrouter/<provider>/<id>:floor`.

- **`:floor` provider routing** — `gen-litellm-config.sh` appends an OpenRouter
  variant to chat and agent routes. The default `:floor` picks the cheapest
  provider for that model. `KC_OR_VARIANT` changes it: `:nitro` for throughput,
  or empty for the OpenRouter default. It does not apply to embeddings.

### vLLM (local)

- **Registration** — filling in the URL registers it under the same model name.

  | `model_name` | URL variable |
  |---|---|
  | `local/qwen3.6-35b` | `VLLM_QWEN35B_URL` |
  | `local/glm-4.7-flash` | `VLLM_GLMFLASH_URL` |

- **Discovery** — `gen-litellm-config.sh` polls `/v1/models` at each URL and
  registers only the nodes that answer.
- **Multi-node** — one deployment per node under the same model name. The
  LiteLLM router picks with `least-busy`.

**Operations**

- **Normally** — the placement step of `setup.sh all` (`scheduler apply`) decides
  what runs where, at which context, and starts it.
- **By hand** — `./scripts/manage-vllm.sh up <service>`.

What fits on which card is in the
[GPU memory guide](gpu-memory.md#per-node-class).

| Model | Node class | Notes |
|---|---|---|
| `qwen3.6-35b` | Any single NVFP4-capable GPU | Unified chat |
| `glm-4.7-flash` | PRO 5000 and up | Cheap-decode floor |

**Roles**

- **`qwen3.6-35b`** — default chat. Artifacts, deep research and coding run on
  the same deployment; `DEEP_RESEARCH_MODEL` points here, and the scheduler holds
  a 128K context floor on it for that reason.
- **`glm-4.7-flash`** — the high-volume internal calls: titles, memory
  extraction, query rewriting. Their call sites are named by the UI.
- **Artifacts** — no separate model. The client produces code and document
  artifacts on the chat deployment and the server extracts them from the
  response.
- **Media** — no local backend. Images, audio and video pass through to
  OpenRouter.

### Local to OpenRouter fallback

Local vLLM chat models fail over to the **same model on OpenRouter** through two
independent paths:

- **Node down or erroring** — `router_settings.fallbacks`, after `num_retries` is
  exhausted by errors, timeouts or cooldown.
- **Overload** — the `concurrency_gate` callback. When vLLM in-flight requests
  exceed the per-model cap, traffic spills to the same OpenRouter twin
  (`services/litellm/callbacks/concurrency_gate.py`). Plain queueing never
  triggers `fallbacks`, which is why this gate exists.

`router_settings.fallbacks`:

| Local (primary) | OpenRouter fallback (paid) |
|---|---|
| `local/qwen3.6-35b` | `qwen/qwen3.6-35b-a3b` ($0.15 / $1.00) |
| `local/glm-4.7-flash` | `z-ai/glm-4.7-flash` ($0.06 / $0.40) |

> These prices must match the `emit_brain` and `emit_or_fallback` arguments in
> `gen-litellm-config.sh`. When they change, verify against
> `https://openrouter.ai/api/v1/models` and update both.

- **Emission condition** — `emit_or_fallback` emits the twin only when the local
  primary is deployed (its URL is set), under the OpenRouter slug as model name.
- **Hidden from the picker** — the twin keeps the OpenRouter slug, so it is
  distinct from the local alias and is used only as a fallback.
- **Cost** — a fallback is **paid OpenRouter egress**. A node that dies often
  leaks money.

### Per-model `max_model_len`

- **Discovery** — `gen-litellm-config.sh` reads `max_model_len` from each
  deployment's `/v1/models` and emits it as LiteLLM's `max_input_tokens`, which
  keeps unified-memory nodes from swapping their KV cache.
- **Fallback** — if a node is unreachable, the single fallback `CTX_FALLBACK`
  (32768) is used.

The values below are what this cluster discovers today; they change with the
nodes.

| Model | Context (this cluster) | Purpose |
|---|---|---|
| `qwen3.6-35b` | 128K (262K native) | Chat, deep research, coding |
| `glm-4.7-flash` | 128K | Internal calls and default chat |

### Embeddings (RAG fallback)

There is no local embedding deployment: nothing in the app calls `/embeddings`
and there is no retrieval path yet. With an OpenAI key, only
`text-embedding-3-small` is registered (~$0.02 per 1M). OpenRouter does not serve
embedding models at all.

When a retrieval path appears, add the embedding model to
`scheduler/models.yaml`.

## Commercial defaults

```bash
OPENAI_MODELS=(gpt-5.6-sol gpt-5.6-terra gpt-5.6-luna gpt-5-nano)
ANTHROPIC_MODELS=(claude-opus-5 claude-sonnet-5 claude-haiku-4.5)
GOOGLE_MODELS=(gemini-3.1-pro-preview gemini-3.6-flash gemini-3.1-flash-lite)
XAI_MODELS=(grok-4.5)
PERPLEXITY_MODELS=(sonar)
TENCENT_MODELS=(hy3)
DEEPSEEK_MODELS=(deepseek-v4-pro deepseek-v4-flash)
ZAI_MODELS=(glm-5.2)
XIAOMI_MODELS=(mimo-v2.5)
MOONSHOTAI_MODELS=(kimi-k3)
```

> External coding clients (Claude Code, Codex) use `local/qwen3.6-35b` as well.

## Setup flow

```bash
# 1. .env — OPENROUTER_API_KEY and NODES_VLLM (URLs are recorded by the scheduler)
./scripts/gen-env.sh && $EDITOR .env

# 2. Download vLLM weights on the GPU node (skip if you have no local GPU)
./scripts/download-vllm-models.sh           # what this card can serve
./scripts/download-vllm-models.sh --help    # aliases and special targets

# 3. Generate configuration and start
./scripts/setup.sh all   # to restart the stack only: setup.sh up
```

## Media

Images, audio and video are all produced by passing through to **OpenRouter** via
LiteLLM. There is no local media GPU backend; the user picks the model in the UI.

| Kind | Path |
|---|---|
| Images and audio | `modalities` on `chat/completions` |
| Video | `/api/v1/videos` passthrough — it does not appear in `/model/info`, so the model list is declared in the UI repository |
| Transcription (STT) | `whisper-shim` to the GPU nodes' faster-whisper, or OpenRouter when no backend is deployed |

Per-tool paths are in [tools.md](tools.md).

### MCP and built-in tools

- **Built-in** — `web_search` (SearXNG), `fetch_url` (Crawl4AI),
  `execute_code` (sandbox), `create_artifact`, `create_chart`
- **MCP** — `time`, `youtube` (stdio) and `deep-research` (HTTP). The catalogue
  holds only sets that have actually been started and verified.

**There is no limit on tool count, and that is the problem.** Every active tool
ships its whole schema on every turn, and model selection accuracy degrades well
before twenty of them. Watch that number when adding connectors.

## RAG embeddings

**Not implemented.** Nothing in the app calls `/embeddings` and there is no
retrieval path, so `file_search` returning nothing is expected rather than
misconfiguration. An embedding deployment would receive no traffic.

Adding one takes three pieces: an entry in `scheduler/models.yaml`, a service in
`docker-compose.vllm.yml`, and an embedding deployment in
`gen-litellm-config.sh`.
