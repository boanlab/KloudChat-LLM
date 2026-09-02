# Model configuration

Which models are registered where, and how requests are routed to them.

> Bringing the stack up for the first time: start with the
> [README](../README.md).

## Where models are defined

Two catalogues, with different jobs.

`scheduler/models.yaml`: **local models**. What vLLM can serve, and everything
placement needs (port, env prefix, context floor, priority, and which half of
the cluster the model belongs to). `VLLM_MODELS` in `.env` selects which of them
are deployed, and the order of `NODES_VLLM` names the head node.

`scripts/lib.sh`: **commercial models** routed through OpenRouter, and the
per-checkpoint download table `download-vllm-models.sh` uses.

| Variable in `lib.sh` | Role |
|---|---|
| `OPENAI_MODELS` / `ANTHROPIC_MODELS` / `GOOGLE_MODELS` / `XAI_MODELS` / `PERPLEXITY_MODELS` | Frontier tier, all through OpenRouter |
| `TENCENT_MODELS` / `DEEPSEEK_MODELS` / `ZAI_MODELS` / `XIAOMI_MODELS` / `MOONSHOTAI_MODELS` / `QWEN_MODELS` / `MINIMAX_MODELS` | Open-weight and hosted tier, too large to self-host |
| `OR_IMAGE_MODELS` / `OR_AUDIO_MODELS` | Image and audio generation through OpenRouter |
| `VLLM_MODELS` | Checkpoint alias to HF repo, for downloading. Card demands are in `VLLM_MODEL_QUANT` and `VLLM_MODEL_WEIGHT_GB` |
| `OPENAI_EMBED_CATALOG` | OpenAI embeddings, the fallback when no local one is deployed |
| `MODEL_PRICE_IN_PM` / `MODEL_PRICE_OUT_PM` / `OR_TWIN_PRICE_*` | Declared USD per 1M tokens, the fallback when the live catalogue is unreachable |

**Pricing**

- Commercial: the OpenRouter catalogue price.
- Local (`local/*` and `strict-local/*`): free (0).
- When the OpenRouter fallback fires: billed at the price of the deployment
  that served it. LiteLLM computes cost against the deployment it fell back to.
- `text-embedding-3-small`: paid, through OpenAI.

`gen-litellm-config.sh` fetches the OpenRouter catalogue once per run and emits
the live price for every route. The tables in `lib.sh` are the fallback for a
run with no key or no network. Generation prints how many prices it read and
how many differed:

```
[INFO] prices: 25 read from the catalogue, 1 differ from the declared fallback
```

`./scripts/gen-litellm-config.sh --check-prices` compares the declared
fallbacks against the catalogue and writes nothing. A declared id that has left
the catalogue is reported as `GONE`. Per-clip audio models have no `pricing`
block on OpenRouter; their figure is read from the model description and stays
a declared value.

**Generated configuration**

`gen-litellm-config.sh` combines the definitions above with `.env` and
regenerates the block between markers in `services/litellm/config.yaml`
(`KLOUDCHAT_AUTOGEN` for `model_list`, `KLOUDCHAT_FALLBACKS` for
`router_settings.fallbacks`). The model picker reads LiteLLM's `/v1/models`
directly; nothing is generated on the UI side.

Every generated deployment carries a KloudChat boundary contract in
`model_info`. Consumers treat a missing or unknown value as external.

| Field | Meaning |
|---|---|
| `kchat_data_boundary` | `self_hosted`, `hybrid`, or `external` |
| `kchat_strict_local` | `true` only for a no-egress `strict-local/*` alias |
| `kchat_privacy_only` | Keeps the strict alias out of ordinary default selection |
| `kchat_hidden` | Fallback twins and the STT route: registered, not shown in the picker |

A `local/*` alias is `hybrid` when an OpenRouter fallback exists and
`self_hosted` without one. It is never `external`: with no GPU URL there is no
local alias at all, and the model registers under its OpenRouter slug.

## The model set

vLLM is the only local LLM backend, on two architectures:

- amd64 (RTX 5090 / PRO 5000 / PRO 6000): base image `vllm/vllm-openai:cu129-nightly`
- arm64 (GB10, 128 GB unified memory): base image `vllm/vllm-openai:nightly-aarch64`

Compose runs `kloudchat-vllm:local`, this repo's layer over the base.
`install-vllm.sh` pulls the base, builds the layer, and records `VLLM_IMAGE`,
`VLLM_BASE_IMAGE` and `VLLM_BASE_DIGEST` in the node's `.env`. A rebuild pins to
the digest, so `nightly` moving upstream does not change a node.

The catalogue below is what vLLM *can* serve. `VLLM_MODELS` selects what is
deployed, `placement` says which half of the cluster a model belongs to, and
`priority` ranks the models within that half when the cards cannot hold
everything. The lowest-ranked is delegated to OpenRouter rather than squeezed
in.

**Node** is `placement` in `models.yaml`. The **head** node is the first target
in `NODES_VLLM` and carries the paths every request touches; the **pool** is
every other node and carries the card-sized models a user picks by name. A blank
means the model is placed wherever it fits.

| Model (alias) | Container | Port | Quant | Node | Priority | Role |
|---|---|---|---|---|---|---|
| `local/qwen3.6-35b` | `vllm-qwen35b` | 8001 | NVFP4 | head | 20 | Unified chat and floor: conversation, artifacts, vision, coding, titles, memory extraction |
| `local/qwen3.5-122b-a10b` | `vllm-qwen122b` | 8004 | NVFP4 | pool (share 60) | 15 | Top chat and deep research. 10B active, 128K here. Needs the card to itself |
| `local/qwen3-coder-next` | `vllm-codernext` | 8008 | FP8 | pool (share 40) | 10 | Coding (Qwen3-Coder-Next-80B-A3B). Hybrid attention, 12 of 48 layers hold KV, 12 KiB/token at 262K |
| `local/bge-m3` | `vllm-bgem3` | 8003 | BF16 | head | 5 | Retrieval embeddings. Pooling runner |
| `local/bge-reranker-v2-m3` | `vllm-rerank` | 8009 | BF16 | head | 0 | Retrieval reranking, second stage over vector search |
| `local/whisper-large-v3` | `vllm-whisper` | 9000 | FP16 | head | 4 | Transcription. Reached through `/tools/stt`, not a LiteLLM chat route |
| `local/qwen3-coder-30b` | `vllm-coder30b` | 8006 | FP8 | any | 0 | Coding, smaller. 48 KiB/token. Superseded by `qwen3-coder-next` where 75 GiB fits |
| `local/qwen3.6-27b` | `vllm-qwen27b` | 8007 | NVFP4 | any | 0 | The one dense model |
| `strict-local/<model>` | same backend as its `local/` twin | | | | | Privacy-only alias; fails rather than leaving vLLM |

**The split.** Qwen3.6-35B-A3B covers vision, a 262K context and agentic coding
on its own, and at 3B active it is cheap enough to carry the high-volume
internal calls (titles, memory extraction, query rewriting). Qwen3.5-122B-A10B
is the quality end: 10B active decodes at roughly a third the rate, and 78 GiB
of weights leaves ~22 GiB of KV on a GB10, about 13 concurrent requests at
128K. Volume goes to the 35B; the 122B is chosen when the answer is worth the
wait.

A model in the catalogue but not in `VLLM_MODELS` gets no `local/` alias; it is
reachable under its OpenRouter slug. See [Registration](#vllm-local).

**Quantisation**

- Default is NVFP4 (GB10 / RTX 5090 / PRO 5000 / PRO 6000).
- A card without FP4 cannot run the default lineup. The catalogue carries an
  AWQ int4 build of the chat model, `qwen3.6-35b-awq`, executable from compute
  capability 7.5. Point `VLLM_QWEN35B_DIR` at it; the served entry does not
  change.
- That build is for large FP4-less cards: on 48 GiB it places at 128K–256K, and
  at 26 GB of weights it does not fit a 24 GiB card. 32 GiB usable is the floor
  and `manage-vllm.sh up` refuses below it ([gpu-memory.md](gpu-memory.md)).

**Parsers**

| Model | Tool parser | Reasoning parser | Notes |
|---|---|---|---|
| `qwen3.6-35b` | `qwen3_xml` | `qwen3` | Thinking off by default via `--default-chat-template-kwargs '{"enable_thinking": false}'`. The hybrid Gated-DeltaNet needs `--max-num-seqs` for cudagraph capture |
| `qwen3.5-122b-a10b` | `qwen3_xml` | `qwen3` | Same family and chat-template controls. `--max-num-seqs` is set low for KV |
| `qwen3.6-27b` | `qwen3_xml` | `qwen3` | Same family plumbing as `qwen3.6-35b` |
| `qwen3-coder-next` | `qwen3_coder` | none | Qwen3-Coder's XML dialect (`<tool_call><function=…><parameter=…>`). `qwen3_xml` yields no tool calls |
| `qwen3-coder-30b` | `qwen3_coder` | none | Same dialect. No thinking mode |

## Free models

Whatever OpenRouter offers for free is queried at config-generation time and
registered. If the query fails, nothing is added.

The filter is `or_free_models` in `scripts/lib.sh`: zero input and output
price, text output, `:free` suffix. Guardrail models (guard, safety,
moderation) are excluded; they classify their input rather than answer.

The UI hides models priced at 0 by default. The `:free` suffix is the exception,
because there the provider stated the price.

## Routing

### Commercial: a single OpenRouter path

| OpenRouter key | Result |
|---|---|
| Present | One route per model, named `<provider>/<id>` |
| Absent | Not registered |

`model_name` is canonical (`openai/gpt-5.6-sol`) while `litellm_params.model`
is `openrouter/<provider>/<id>:floor`. `:floor` picks the cheapest provider for
that model. `KC_OR_VARIANT` changes the suffix: `:nitro` for throughput, empty
for the OpenRouter default. It does not apply to embeddings.

### vLLM (local)

- **Registration**: a non-empty URL registers the model under its `local/`
  name.

  | `model_name` | URL variable |
  |---|---|
  | `local/qwen3.6-35b` | `VLLM_QWEN35B_URL` |
  | `local/qwen3.5-122b-a10b` | `VLLM_QWEN122B_URL` |
  | `local/qwen3-coder-next` | `VLLM_CODERNEXT_URL` |
  | `local/qwen3-coder-30b` | `VLLM_CODER30B_URL` |
  | `local/qwen3.6-27b` | `VLLM_QWEN27B_URL` |
  | `local/bge-m3` | `VLLM_BGEM3_URL` |
  | `local/bge-reranker-v2-m3` | `VLLM_RERANK_URL` |

- **No URL**: no `local/*` name. With an OpenRouter key the model is reachable
  under its own slug (`qwen/qwen3.6-35b-a3b`, `qwen/qwen3.5-122b-a10b`) and
  priced as the paid route it is. A surface that names `local/<m>` stops
  resolving on a GPU-less install; see **Naming a model from outside**.
- **Discovery**: `gen-litellm-config.sh` polls `/v1/models` at each URL and
  registers only the nodes that answer.
- **Multi-node**: one deployment per node under the same model name. The
  LiteLLM router picks with `least-busy`.
- **Strict aliases**: each registered chat vLLM also gets a
  `strict-local/<model>` alias over the same backend.

**Operations**

- Normally: the placement step of `setup.sh all` (`scheduler apply`) decides
  what runs where, at which context, and starts it.
- By hand: `./scripts/manage-vllm.sh up <service>` on the node.

What fits on which card is in the
[GPU memory guide](gpu-memory.md#per-node-class).

| Model | Node class | Notes |
|---|---|---|
| `qwen3.6-35b` | Any single NVFP4-capable GPU | Unified chat and floor |
| `qwen3.5-122b-a10b` | GB10, or 2 × PRO 6000 (`tensor_parallel: 2`), alone on the cards | Top chat |
| `qwen3-coder-next` | GB10 or PRO 6000, alone on the card | Coding (FP8, 75 GiB) |
| `qwen3-coder-30b` | Any single GPU (FP8, no FP4 needed) | Coding |
| `qwen3.6-27b` | Any single NVFP4-capable GPU | Dense |

**Roles**

- `qwen3.6-35b`: default chat and the deployment volume points at. Artifacts,
  coding and the high-volume internal calls (titles, memory extraction, query
  rewriting) run here; the UI names those call sites (`KCHAT_TITLE_MODEL`). The
  scheduler holds a 128K context floor on it for coding-agent sessions. It sits
  on the head node with retrieval.
- `qwen3.5-122b-a10b`: top chat, chosen from the picker, and what the pool
  exists for (three pool nodes in five). `DEEP_RESEARCH_MODEL` points here, the
  one route that reaches it without a user choosing it. Nothing else is routed
  here by default: its KV pool admits 12 concurrent sessions.
- `qwen3-coder-next`: coding, a picker choice. 75 GiB of FP8 weights, so it
  wants a pool card to itself (two pool nodes in five). On a pool of one it
  yields to the 122B.
- `qwen3-coder-30b` and `qwen3.6-27b`: picker choices for a cluster with cards
  to spare. Check `scheduler plan` before adding them to `VLLM_MODELS`.

**Ranking.** `placement` decides which cards a model may compete for, and
`priority` decides who wins among the models competing for the same ones:
`qwen3.6-35b` (20) then `bge-m3` (5) on the head node, `qwen3.5-122b-a10b`
(15) then `qwen3-coder-next` (10) in the pool. Without a priority, coverage
seats the largest model first. The head node is ranked separately from the
pool: losing the 35B degrades every path at once, losing a pool model costs
deep research or a picker choice.

**Sharing the pool.** `share` divides the pool nodes once every model has one:
60 to `qwen3.5-122b-a10b` and 40 to `qwen3-coder-next`, so a pool of five holds
three and two. It is a weight, not a percentage, and with a pool of one it
decides nothing.

- **Artifacts**: no separate model. The client produces code and document
  artifacts on the chat deployment and the server extracts them.
- **Media**: no local backend. Images, audio and video pass through to
  OpenRouter.

### Local to OpenRouter fallback

Local vLLM chat models fail over to the same model on OpenRouter through two
independent paths:

- **Node down or erroring**: `router_settings.fallbacks`, after `num_retries`
  is exhausted by errors, timeouts or cooldown.
- **Overload**: the `concurrency_gate` callback
  (`services/litellm/callbacks/concurrency_gate.py`). When vLLM in-flight
  requests exceed the per-model cap, traffic spills to the OpenRouter twin.
  Plain queueing never triggers `fallbacks`.

`router_settings.fallbacks`, with the declared fallback prices in `lib.sh`
(live catalogue prices override them at generation):

| Local (primary) | OpenRouter fallback (paid, $/1M in / out) |
|---|---|
| `local/qwen3.6-35b` | `qwen/qwen3.6-35b-a3b` (0.14 / 1.00) |
| `local/qwen3.5-122b-a10b` | `qwen/qwen3.5-122b-a10b` (0.26 / 2.08) |
| `local/qwen3-coder-next` | `qwen/qwen3-coder-next` (live price) |
| `local/qwen3-coder-30b` | `qwen/qwen3-coder-30b-a3b-instruct` (0.07 / 0.28) |
| `local/qwen3.6-27b` | `qwen/qwen3.6-27b` (0.60 / 3.60) |

- **Emission condition**: `emit_or_fallback` emits the twin only when the local
  primary is deployed (its URL is set). With no local primary, `emit_brain`
  registers that same slug as an ordinary visible route. The two never both
  fire: two deployments under one `model_name` would split ordinary traffic
  onto the paid one.
- **Hidden from the picker**: the twin keeps the OpenRouter slug and
  `kchat_hidden: true`.
- **Cost**: a fallback is paid OpenRouter egress.

### Naming a model from outside

A caller that hard-codes `local/<m>` asserts the install has that GPU
deployment. Where it might not, read the catalogue and fall back:

| Setting | Behaviour when the name is absent |
|---|---|
| `DEEP_RESEARCH_MODEL` (this repo) | Passed through to the deep-research service as-is; set it to a model the install serves |
| `KCHAT_DEFAULT_CHAT_MODEL` (UI) | Blanked against the live catalogue; the picker falls back to its cheapest |
| `KCHAT_TITLE_MODEL` (UI) | Blanked against the live catalogue; title and memory extraction use the session's own model |

### Strict-local fail-closed routing

`strict-local/*` is the route for requests that must not leave the self-hosted
vLLM deployment. Two independent fail-closed controls:

- The generated `router_settings.fallbacks` table never contains a strict
  alias, so node errors, timeouts and cooldown never select OpenRouter.
- The concurrency gate reads `model_info.kchat_strict_local`. At the
  saturation threshold that spills a normal alias, it returns
  `strict_local_unavailable` without rewriting the model id.

A `/metrics` scrape failure marks strict capacity unavailable and rejects the
request; normal aliases keep their fail-open behaviour. If vLLM fails after a
healthy capacity check, LiteLLM retries only the strict alias's own deployments
and then returns an error.

`./scripts/manage.sh team add-strict` adds a strict alias to each team's
allowlist only where that team already has the matching `local/*` model.
`team sync` replaces a team's allowlist with the full generated catalogue.

### Spend-log privacy

`general_settings.store_prompts_in_spend_logs` is `false`, enforced by
`gen-litellm-config.sh` on every run. Token usage and cost attribution are
recorded; prompt and response bodies are not.

### Per-model `max_model_len`

- **Discovery**: `gen-litellm-config.sh` reads `max_model_len` from each
  deployment's `/v1/models` and emits it, minus `KC_PRE_CALL_HEADROOM` (4096),
  as LiteLLM's `max_input_tokens`.
- **Fallback**: if a node is unreachable, `CTX_FALLBACK` (32768) is used.

Contexts as deployed here for the placed models, and the `.env.example`
defaults for the rest. They change with the nodes.

| Model | Context | Purpose |
|---|---|---|
| `qwen3.6-35b` | 256K (262K native) | Chat, coding, internal calls |
| `qwen3.5-122b-a10b` | 128K (262K native) | Top chat and deep research. Capped by KV, not by the model |
| `qwen3-coder-next` | 256K (262K native) | Coding |
| `qwen3-coder-30b` | 128K | Coding. Capped by KV: 48 KiB/token |
| `qwen3.6-27b` | 128K | Dense |

### Embeddings

`bge-m3` (`BAAI/bge-m3`, 1024 dimensions, 8K context, multilingual) serves
KloudChat's retrieval index through `/tools/index`. It is a pooling runner: no
tool parser, no reasoning parser, no KV cache, so the planner charges it
weights and activation only. Registered with `mode: embedding`, which keeps it
out of the model picker.

With no local deployment and an `OPENAI_API_KEY`, `text-embedding-3-small` is
registered as the fallback. OpenRouter serves no embedding models. With
neither, KloudChat falls back to lexical retrieval.

## Commercial defaults

```bash
OPENAI_MODELS=(gpt-5.6-sol gpt-5.6-terra gpt-5.6-luna gpt-5-nano gpt-5.3-codex)
ANTHROPIC_MODELS=(claude-fable-5 claude-opus-5 claude-sonnet-5 claude-haiku-4.5)
GOOGLE_MODELS=(gemini-3.1-pro-preview gemini-3.7-flash gemini-3.1-flash-lite)
XAI_MODELS=(grok-4.6)
PERPLEXITY_MODELS=(sonar sonar-pro)
TENCENT_MODELS=(hy3)
DEEPSEEK_MODELS=(deepseek-v4-pro deepseek-v4-flash)
ZAI_MODELS=(glm-5.3)
XIAOMI_MODELS=(mimo-v2.5)
MOONSHOTAI_MODELS=(kimi-k3)
QWEN_MODELS=(qwen3.8-max qwen3.7-flash qwen3-coder-plus)
MINIMAX_MODELS=(minimax-m3)
```

By use case:

| Need | Model | Declared price /1M |
|---|---|---|
| Bulk work where cost dominates | `qwen/qwen3.7-flash` (1M ctx) | $0.03 / $0.13 |
| Commercial coding | `openai/gpt-5.3-codex`, `qwen/qwen3-coder-plus` | $1.75/$14, $0.65/$3.25 |
| Search that reads more than a snippet | `perplexity/sonar-pro` | $3 / $15 |
| Speech generation | `openai/gpt-audio-mini` | $0.60 / $2.40 audio |

`sonar-pro` does not replace the stack's own deep-research service, which
drives a local model over SearXNG.

## Setup flow

```bash
# 1. .env: OPENROUTER_API_KEY and NODES_VLLM (URLs are recorded by the scheduler)
./scripts/gen-env.sh && $EDITOR .env

# 2. Download vLLM weights on the GPU node (skip without a local GPU)
./scripts/download-vllm-models.sh           # what this card can serve
./scripts/download-vllm-models.sh --help    # aliases and special targets

# 3. Generate configuration and start
./scripts/setup.sh all   # to restart the stack only: setup.sh up
```

## Media

Images, audio and video pass through to OpenRouter via LiteLLM. There is no
local media backend; the user picks the model in the UI.

| Kind | Path |
|---|---|
| Images and audio | `modalities` on `chat/completions` (`OR_IMAGE_MODELS`, `OR_AUDIO_MODELS` in `lib.sh`) |
| Video | `/api/v1/videos` passthrough. Not in `/model/info`; the model list is declared in the UI repository |
| Transcription (STT) | `whisper-shim` to the GPU nodes' `vllm-whisper`, or OpenRouter (`STT_OR_MODEL`) when no local backend answers |

Per-tool paths are in [tools.md](tools.md).

### MCP and built-in tools

- Built-in: `web_search` (SearXNG), `fetch_url` (Crawl4AI), `execute_code`
  (sandbox), `create_artifact`, `create_chart`
- MCP: `deep-research` (HTTP) is the connector this repository provides; other
  connectors are configured in the UI

Every active tool ships its whole schema on every turn, and model selection
accuracy degrades well before twenty of them. Watch that number when adding
connectors.

## Retrieval

Two stages, both local, both through the gateway.

| Stage | Model | Job |
|---|---|---|
| Recall | `local/bge-m3` (`mode: embedding`) | Nearest passages by cosine distance in pgvector |
| Precision | `local/bge-reranker-v2-m3` (`mode: rerank`) | Scores each (query, passage) pair |

`index-shim` over-fetches `limit × INDEX_RERANK_CANDIDATES` from pgvector under
a loose distance bound (`INDEX_RERANK_RECALL_DISTANCE`), reranks, and keeps the
top `limit` above `INDEX_RERANK_MIN_SCORE`. The recall cut is loose on purpose:
precision is the second stage's job.

Both stages degrade rather than fail. No reranker, or one that cannot be
reached, and search falls back to vector order; the response says which
(`"reranked": true|false`). No embedding deployment and an OpenAI key registers
`text-embedding-3-small`; with neither, KloudChat falls back to lexical
retrieval.

Adding a stage takes three pieces: an entry in `scheduler/models.yaml`, a
service in `docker-compose.vllm.yml`, and an `emit_vllm_embed` /
`emit_vllm_rerank` call in `gen-litellm-config.sh`.
