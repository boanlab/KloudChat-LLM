# Model configuration

Which models are registered where, and how requests are routed to them.

> First bring-up: start with the [README](../README.md).

## Where models are defined

| File | Contents |
|---|---|
| `scheduler/models.yaml` | Local models vLLM can serve, with everything placement needs: port, env prefix, context floor, priority, placement. `VLLM_MODELS` in `.env` selects which are deployed; the first target in `NODES_VLLM` is the head node |
| `scripts/lib.sh` | Commercial models routed through OpenRouter, the download table for `download-vllm-models.sh`, and declared fallback prices |

| Variable in `lib.sh` | Role |
|---|---|
| `OPENAI_MODELS` / `ANTHROPIC_MODELS` / `GOOGLE_MODELS` / `XAI_MODELS` / `PERPLEXITY_MODELS` | Frontier tier, through OpenRouter |
| `TENCENT_MODELS` / `DEEPSEEK_MODELS` / `ZAI_MODELS` / `XIAOMI_MODELS` / `MOONSHOTAI_MODELS` / `QWEN_MODELS` / `MINIMAX_MODELS` | Open-weight and hosted tier, through OpenRouter |
| `OR_IMAGE_MODELS` / `OR_AUDIO_MODELS` | Image and audio generation through OpenRouter |
| `VLLM_MODELS` (associative array) | Download alias to HF repo. Sizes in `VLLM_MODEL_WEIGHT_GB` |
| `OPENAI_EMBED_CATALOG` | OpenAI embeddings, the fallback when no local embedding model is deployed |
| `MODEL_PRICE_IN_PM` / `MODEL_PRICE_OUT_PM` / `OR_TWIN_PRICE_*` | Declared USD per 1M tokens, the fallback when the live catalogue is unreachable |

### Pricing

- Commercial: the OpenRouter catalogue price.
- Local (`local/*` and `strict-local/*`): 0.
- OpenRouter fallback for a local model: the price of the OpenRouter deployment that served it.
- `text-embedding-3-small`: paid, through OpenAI (`OPENAI_API_KEY`, passed to
  the LiteLLM container).

`gen-litellm-config.sh` downloads the OpenRouter catalogue once per run,
overlays the live prices on the declared tables and emits them for every
route; the `lib.sh` tables apply when the catalogue is unreachable. The run
prints how many prices it read and how many differed from the declared values.

`./scripts/gen-litellm-config.sh --check-prices` compares the declared values
against the catalogue and writes nothing. A declared id missing from the
catalogue is reported as `GONE`. Per-clip audio models have no `pricing` block
on OpenRouter; their figure is read from the model description.

### Generated configuration

`gen-litellm-config.sh` regenerates the block between markers in
`services/litellm/config.yaml`: `KLOUDCHAT_AUTOGEN` for `model_list`,
`KLOUDCHAT_FALLBACKS` for `router_settings.fallbacks`. The UI reads LiteLLM's
`/v1/models` directly.

Every generated deployment carries a boundary contract in `model_info`:

| Field | Meaning |
|---|---|
| `kchat_data_boundary` | `self_hosted`, `hybrid`, or `external` |
| `kchat_strict_local` | `true` only for a `strict-local/*` alias |
| `kchat_privacy_only` | Keeps the strict alias out of default selection |
| `kchat_hidden` | Fallback twins and the STT route: registered, not shown in the picker |

A `local/*` alias is `hybrid` when an OpenRouter fallback exists and
`self_hosted` without one. With no GPU URL there is no `local/*` alias; the
model registers under its OpenRouter slug as `external`.

## The model set

vLLM is the only local backend. Base images: amd64
`vllm/vllm-openai:cu129-nightly`, arm64 (GB10) `vllm/vllm-openai:nightly-aarch64`.
Compose runs `kloudchat-vllm:local`, this repo's layer over the base
(`services/vllm/Dockerfile`); `install-vllm.sh` builds it and records
`VLLM_IMAGE`, `VLLM_BASE_IMAGE` and `VLLM_BASE_DIGEST` in the node's `.env`.

The catalogue is what vLLM *can* serve. `VLLM_MODELS` selects what is
deployed; `placement` says which nodes a model may use; `priority` ranks the
models competing for the same cards. A model with no seat is delegated to
OpenRouter.

**Node** is `placement` in `models.yaml`: `head` is the first target in
`NODES_VLLM`, `pool` is every other node, `any` is wherever it fits.

| Model (alias) | Container | Port | Quant | Node | Priority | Role |
|---|---|---|---|---|---|---|
| `local/qwen3.8-27b` | `vllm-qwen27b` | 8001 | NVFP4 | any | 20 | Chat: conversation, vision, coding, deep research, titles, memory extraction. One replica per node with room |
| `local/gemma-4-26b-a4b` | `vllm-gemma26b` | 8002 | NVFP4 | pool | 1 | Fast chat: titles, memory extraction, query rewriting, quick turns. MoE 3.8B active, vision, context floor 131072 |
| `local/qwen3-coder-next` | `vllm-codernext` | 8008 | FP8 | pool | 10 | Coding. Qwen3-Coder-Next-80B-A3B, hybrid attention, 12 of 48 layers hold KV |
| `local/bge-m3` | `vllm-bgem3` | 8003 | BF16 | head | 5 | Retrieval embeddings. Pooling runner |
| `local/bge-reranker-v2-m3` | `vllm-rerank` | 8009 | BF16 | head | 2 | Retrieval reranking. Pooling runner |
| `local/whisper-large-v3` | `vllm-whisper` | 9000 | FP16 | head | 4 | Transcription, through `/tools/stt`, not a LiteLLM chat route |
| `strict-local/<model>` | same backend as its `local/` twin | | | | | Privacy alias; fails rather than leaving vLLM |

**Two chat models.** `qwen3.8-27b` is the quality tier: dense with hybrid
attention (16 of 64 layers hold KV, 32 KiB per token in fp8), native 262144
context, and MTP speculative decoding from the checkpoint's own draft head
(`VLLM_QWEN27B_SPEC_TOKENS`, default 5). Its context floor is the native
262144: a card that cannot hold that delegates rather than serving a shorter
window. It is unplaced, so every node with room gets a replica, all registered
under `local/qwen3.8-27b`; LiteLLM spreads requests across them (`least-busy`).
Each deployment carries a 3600 s request timeout and a concurrency-gate cap of
128 in-flight requests.

`gemma-4-26b-a4b` is the speed tier: a 25.2B MoE with 3.8B active, vision,
4 of 30 layers holding KV (16 KiB per token in fp8) and 1024-token sliding
windows on the other 26. Measured on GB10 it decodes at 48 tok/s
single-stream against the 27B's 15, and batches to 494 tok/s at 16 concurrent
requests. It is pool-placed (priority 1, context floor 131072) and takes what
a pool node has left after the 27B, registered under `local/gemma-4-26b-a4b`
with a 900 s timeout and a cap of 128. The UI points high-volume internal calls
(titles, memory extraction, query rewriting) and quick turns here; default
chat and deep research stay on the 27B.

**Coder.** `qwen3-coder-next` is in the catalogue but not in the default
`VLLM_MODELS`. Coverage seats every listed model once before any replica, so
listing it gives a pool card to the coder instead of a second 27B. Unlisted,
it is reachable as `qwen/qwen3-coder-next` through OpenRouter. Deployed, it
gets the same treatment as the chat tiers: a hidden OpenRouter twin
(`qwen/qwen3-coder-next`, 0.12 / 0.80 $ per 1M), a `fallbacks` line, and a
concurrency-gate cap of 32 on `local/` and `strict-local/`. `manage.sh team
sync` includes whichever of the three names applies.

**Quantisation.** Chat is NVFP4, the coder FP8, retrieval BF16, transcription
FP16. The supported cards (GB10, RTX 5090, RTX PRO 5000, RTX PRO 6000) execute
all of them. 32 GiB usable is the floor; `manage-vllm.sh up` without a service
name refuses below it.

**Parsers** (set in `docker-compose.vllm.yml`):

| Model | Tool parser | Reasoning parser | Notes |
|---|---|---|---|
| `qwen3.8-27b` | `qwen3_xml` | `qwen3` | Thinking off by default (`--default-chat-template-kwargs '{"enable_thinking": false}'`). `--max-num-seqs` bounds CUDA-graph capture for the hybrid conv-state cache |
| `gemma-4-26b-a4b` | `gemma4` | `gemma4` | Thinking off by default (`enable_thinking` in the chat template) |
| `qwen3-coder-next` | `qwen3_coder` | none | Qwen3-Coder's XML dialect |

## Free models

Whatever OpenRouter offers for free is read at config-generation time and
registered; if the query fails, nothing is added. The filter is
`or_free_models` in `scripts/lib.sh`: zero price both ways, text output,
`:free` suffix, guardrail models excluded.

## Routing

### Commercial: OpenRouter

| OpenRouter key | Result |
|---|---|
| Present | One route per model, named `<provider>/<id>` |
| Absent | Not registered |

`model_name` is canonical (`openai/gpt-6.1-sol`); `litellm_params.model` is
`openrouter/<provider>/<id>:floor`. `:floor` picks the cheapest provider.
`KC_OR_VARIANT` changes the suffix: `:nitro` for throughput, empty for the
OpenRouter default. Embeddings are unaffected.

### vLLM (local)

| `model_name` | URL variable |
|---|---|
| `local/qwen3.8-27b` | `VLLM_QWEN27B_URL` |
| `local/gemma-4-26b-a4b` | `VLLM_GEMMA26B_URL` |
| `local/qwen3-coder-next` | `VLLM_CODERNEXT_URL` |
| `local/bge-m3` | `VLLM_BGEM3_URL` |
| `local/bge-reranker-v2-m3` | `VLLM_RERANK_URL` |

- **Registration**: a non-empty URL registers the model under its `local/`
  name; each chat model also gets a `strict-local/<model>` alias over the same
  backend.
- **No URL**: no `local/*` name. With an OpenRouter key the model is reachable
  under its slug (`qwen/qwen3.8-27b`, `qwen/qwen3-coder-next`) at OpenRouter's
  price.
- **Discovery**: `gen-litellm-config.sh` polls `/v1/models` at each URL and
  registers only the nodes that answer.
- **Multi-node**: one deployment per node under the same name; the router
  picks `least-busy`.

Operations: the placement step of `setup.sh all` (`scheduler apply`) decides
what runs where and starts it. By hand: `./scripts/manage-vllm.sh up <service>`
on the node. What fits on which card: [GPU memory](gpu-memory.md#per-node-class).

**Ranking.** `placement` decides which cards a model may compete for;
`priority` decides who wins among models competing for the same ones:
`qwen3.8-27b` (20), `qwen3-coder-next` (10), `bge-m3` (5),
`whisper-large-v3` (4), `bge-reranker-v2-m3` (2), `gemma-4-26b-a4b` (1): the
speed tier takes what a pool node has left. Ties seat the largest model first.
Once every model has an instance, `share` weights extra instances among models
competing for the same nodes.

- **Artifacts**: no separate model. The UI produces artifacts on the chat
  deployment.
- **Media**: no local backend. Images, audio and video pass through to
  OpenRouter.

### Local to OpenRouter fallback

Local chat models fail over to the same model on OpenRouter through two
independent paths:

- **Node down or erroring**: `router_settings.fallbacks`, after `num_retries`
  (2) is exhausted by errors, timeouts or cooldown.
- **Overload**: the `concurrency_gate` callback
  (`services/litellm/callbacks/concurrency_gate.py`) polls each gated model's
  vLLM `/metrics` every `CONCURRENCY_GATE_TTL` seconds. When running requests
  reach the cap (128 for `local/qwen3.8-27b` and `local/gemma-4-26b-a4b`),
  traffic spills to the OpenRouter
  twin. Plain queueing never triggers `fallbacks`.

| Local (primary) | OpenRouter fallback (declared $/1M in / out) |
|---|---|
| `local/qwen3.8-27b` | `qwen/qwen3.8-27b` (0.42 / 3.00) |
| `local/gemma-4-26b-a4b` | `google/gemma-4-26b-a4b-it` (0.0675 / 0.225) |
| `local/qwen3-coder-next` | `qwen/qwen3-coder-next` (0.12 / 0.80) |

- `emit_or_fallback` emits the twin only when the local primary is deployed.
  With no local primary, `emit_brain` registers the same slug as an ordinary
  visible route. The two never both fire.
- The twin keeps the OpenRouter slug and `kchat_hidden: true`.
- A fallback is paid OpenRouter egress.

### Naming a model from outside

A caller that hard-codes `local/<m>` asserts the install has that GPU
deployment. `DEEP_RESEARCH_MODEL` is passed to the deep-research service
as-is: set it to a model the install serves. The UI's own model settings are
validated against the live catalogue on the UI side.

### Strict-local fail-closed routing

`strict-local/*` is the route for requests that must not leave vLLM. Two
independent fail-closed controls:

- The generated `router_settings.fallbacks` never contains a strict alias, so
  node errors, timeouts and cooldown never select OpenRouter.
- The concurrency gate reads `model_info.kchat_strict_local`. At the
  saturation threshold that spills a normal alias, it returns
  `strict_local_unavailable` (503) without rewriting the model id.

A `/metrics` scrape failure marks strict capacity unavailable and rejects the
request; normal aliases fail open.

`./scripts/manage.sh team add-strict` adds a strict alias to each team's
allowlist only where that team already has the matching `local/*` model.
`team sync` replaces a team's allowlist with the full generated catalogue.

### Spend-log privacy

`general_settings.store_prompts_in_spend_logs` is `false`, enforced by
`gen-litellm-config.sh` on every run. Token usage and cost are recorded;
prompt and response bodies are not.

### Per-model `max_model_len`

- **Discovery**: `gen-litellm-config.sh` reads `max_model_len` from each
  deployment's `/v1/models` and emits it, minus `KC_PRE_CALL_HEADROOM` (4096),
  as LiteLLM's `max_input_tokens`.
- **Fallback**: a node that does not answer gets `CTX_FALLBACK` (32768).

`qwen3.8-27b` serves 262144 wherever it is placed (its floor is the native
context). `gemma-4-26b-a4b` serves between its 131072 floor and 262144,
depending on the room left on its card. The scheduler writes the per-node
value as `VLLM_<PREFIX>_MAX_LEN`.

### Embeddings

`bge-m3` (`BAAI/bge-m3`, 1024 dimensions, 8K context, multilingual) serves the
retrieval index through `/tools/index`. Pooling runner: no parsers, no KV
cache. Registered with `mode: embedding`, which keeps it out of the picker.

With no local deployment and an `OPENAI_API_KEY`, `text-embedding-3-small` is
registered as the fallback. OpenRouter serves no embedding models. With
neither, the UI falls back to lexical retrieval.

## Commercial defaults

```bash
OPENAI_MODELS=(gpt-6-astra gpt-6.1-sol gpt-6-luna gpt-5.4-nano gpt-5.3-codex)
ANTHROPIC_MODELS=(claude-fable-5.1 claude-opus-5.5 claude-sonnet-5.5 claude-haiku-4.5)
GOOGLE_MODELS=(gemini-3.1-pro-preview gemini-3.8-flash gemini-3.5-flash-lite)
XAI_MODELS=(grok-4.7)
PERPLEXITY_MODELS=(sonar sonar-pro)
# Open-weight tier
TENCENT_MODELS=(hy4-preview)
DEEPSEEK_MODELS=(deepseek-v4-pro-0813 deepseek-v4.1-flash)
ZAI_MODELS=(glm-5.3 glm-5.3-flash)
XIAOMI_MODELS=(mimo-v2.6-flash)
MOONSHOTAI_MODELS=(kimi-k3)
# Qwen's hosted tier (not the local checkpoints)
QWEN_MODELS=(qwen3.8-max-0902 qwen3.8-flash qwen3-coder-plus)
MINIMAX_MODELS=(minimax-m3)
```

| Need | Model | Declared price /1M |
|---|---|---|
| Bulk work where cost dominates | `openai/gpt-6-luna` | $0.10 / $0.50 |
| Commercial coding | `openai/gpt-5.3-codex`, `qwen/qwen3-coder-plus` | $1.75 / $14, $0.65 / $3.25 |
| Search that reads more than a snippet | `perplexity/sonar-pro` | $3 / $15 |
| Speech generation | `openai/gpt-audio-mini` | $0.60 / $2.40 audio |

`sonar-pro` does not replace the stack's deep-research service, which drives
a local model over SearXNG.

## Setup flow

```bash
# 1. .env: OPENROUTER_API_KEY and NODES_VLLM (URLs are written by the scheduler)
./scripts/gen-env.sh && $EDITOR .env

# 2. Weights on the GPU node (skip without a local GPU)
./scripts/download-vllm-models.sh           # what this card can serve
./scripts/download-vllm-models.sh --help    # aliases and special targets

# 3. Generate configuration and start
./scripts/setup.sh all   # restart the stack only: setup.sh up
```

## Media

Images, audio and video pass through to OpenRouter via LiteLLM; the user picks
the model in the UI.

| Kind | Path |
|---|---|
| Images and audio | `modalities` on `chat/completions` (`OR_IMAGE_MODELS`, `OR_AUDIO_MODELS` in `lib.sh`) |
| Video | `pass_through_endpoints` in `services/litellm/config.yaml.example`, priced per request |
| Transcription (STT) | `whisper-shim` to the GPU nodes' `vllm-whisper`, or OpenRouter (`STT_OR_MODEL`) when no local backend answers |

Per-tool paths are in [tools.md](tools.md).

## Retrieval

Two stages, both local, both through the gateway.

| Stage | Model | Job |
|---|---|---|
| Recall | `local/bge-m3` (`mode: embedding`) | Nearest passages by cosine distance in pgvector |
| Precision | `local/bge-reranker-v2-m3` (`mode: rerank`) | Scores each (query, passage) pair |

`index-shim` over-fetches `limit × INDEX_RERANK_CANDIDATES` from pgvector
under a loose distance bound (`INDEX_RERANK_RECALL_DISTANCE`), reranks, and
keeps the top `limit` above `INDEX_RERANK_MIN_SCORE`.

Both stages degrade rather than fail. No reranker, or one that cannot be
reached, and search falls back to vector order; the response says which
(`"reranked": true|false`). No embedding deployment and an OpenAI key registers
`text-embedding-3-small`; with neither, the UI falls back to lexical retrieval.

Adding a stage takes three pieces: an entry in `scheduler/models.yaml`, a
service in `docker-compose.vllm.yml`, and an `emit_vllm_embed` /
`emit_vllm_rerank` call in `gen-litellm-config.sh`.
