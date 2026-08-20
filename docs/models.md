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
| `OPENAI_EMBED_CATALOG` | OpenAI embeddings, the fallback when no local one is deployed |
| `MODEL_PRICE_IN_PM` / `MODEL_PRICE_OUT_PM` | USD per 1M tokens, for LiteLLM spend tracking |

**Pricing policy**

- **Commercial** — the OpenRouter catalogue price.
- **Local** (`local/*` and `strict-local/*`) — **free (0)**. Self-hosted GPU,
  no per-token billing.
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

**`./scripts/gen-litellm-config.sh --check-prices`** does that comparison and
writes nothing. Worth running whenever the catalogue is touched: a wrong price
breaks no request — the model answers, the call succeeds — so it surfaces only
when someone reads a billing report and disbelieves it. An audit in this repo
found seven of eighteen declared chat prices had drifted, one of them by 14×,
and `gpt-audio` wrong on both its token rates.

It covers chat (`prompt`/`completion`), image (`image_output`) and audio
(`audio`/`audio_output`) prices, and reports a declared id that has left the
catalogue as `GONE` — that one 404s on first call. Per-clip models are the
awkward case: OpenRouter leaves their `pricing` block empty and states the figure
in the model description instead ("30 second duration clips are priced at $0.04
per clip"), so the check reads it from there rather than treating it as
unverifiable.

**Generated configuration**

`gen-litellm-config.sh` combines the definitions above with the environment and
regenerates the block between markers.

| Generator | Target | Marker |
|---|---|---|
| `gen-litellm-config.sh` | `services/litellm/config.yaml` | `KLOUDCHAT_AUTOGEN` |

The model picker reads LiteLLM's `/v1/models` directly, so there is nothing to
generate on the UI side.

Every generated deployment carries an explicit KloudChat boundary contract in
`model_info`. Consumers must treat a missing or unknown value as external rather
than deriving trust from the model name.

| Field | Meaning |
|---|---|
| `kchat_data_boundary` | `self_hosted`, `hybrid`, or `external` for the generated route |
| `kchat_strict_local` | `true` only for a no-egress `strict-local/*` alias |
| `kchat_privacy_only` | Keeps the strict alias out of ordinary default selection |

The normal `local/*` alias is `hybrid` when an OpenRouter fallback exists and
`self_hosted` without one. It is never `external`: the prefix states where a
request starts, and a deployment that spills to OpenRouter under load or on
failure still starts local. Where no GPU URL exists there is no local alias at
all — the model registers under its OpenRouter slug instead. Only
`strict-local/*` has both boolean flags set.

## The model set

vLLM is the only local LLM backend, on two architectures:

- **amd64** — discrete cards (RTX 5090 / PRO 5000 / PRO 6000), base image
  `vllm/vllm-openai:cu129-nightly`.
- **arm64** — GB10 with 128 GB of unified memory, base image
  `vllm/vllm-openai:nightly-aarch64`.

**Compose does not run the base image directly, and the pin is per node.**
`install-vllm.sh` pulls the base, builds this repo's layer over it as
`kloudchat-vllm:local`, and records `VLLM_IMAGE`, `VLLM_BASE_IMAGE` and
`VLLM_BASE_DIGEST` in that node's `.env`. A floating `nightly` fails silently
rather than loudly — the build that relocated vLLM's tool-parser registry left
every container healthy and every tool call unparsed — so `VLLM_BASE_DIGEST` is
what a rebuild should be pinned to.

It used to rebuild the pytest layer **over the tag it had pulled**, which
overwrites the tag with a local build. That is why the digest cannot simply be
read off a running node: `RepoDigests` there reads back identical to the image
`Id`, and `docker pull` cannot resolve it. Separating the two tags is what makes
the recorded digest real.

| Model (alias) | Container | Port | Quant | Role |
|---|---|---|---|---|
| `local/qwen3.5-122b-a10b` | `vllm-qwen122b` | 8004 | NVFP4 | Top chat — 10B active, 128K here. Needs the card to itself |
| `local/qwen3.6-35b` | `vllm-qwen35b` | 8001 | NVFP4 | Unified chat and floor — conversation, artifacts, vision, deep research, coding, titles, memory extraction |
| `local/gemma-4-26b-a4b` | `vllm-gemma26b` | 8005 | NVFP4 | Second family — vision, tool calling, 256K. 4B active of 26B |
| `local/qwen3-coder-30b` | `vllm-coder30b` | 8006 | FP8 | Coding. 48 KiB/token, the most KV-expensive model here |
| `local/qwen3.6-27b` | `vllm-qwen27b` | 8007 | NVFP4 | The one dense model — no routing, a different kind of answer |
| `local/glm-4.7-flash` | `vllm-glmflash` | 8002 | NVFP4 | Cheap-decode floor (31.2B-A3B). **Defined, not deployed here** — a fourth chat model costs `qwen3.6-35b` half its context on two nodes |
| `strict-local/<model>` | same backend as its `local/` twin | — | NVFP4 | Privacy-only alias; fails rather than leaving vLLM |

**Why this split**

Qwen3.6-35B-A3B covers vision, a 262K context and agentic coding on its own, and
at 3B active it is cheap enough to also carry the high-volume internal calls
(titles, memory extraction, query rewriting). It is the workhorse.

Qwen3.5-122B-A10B is the quality end, and it is not a substitute for the 35B in
any of those roles: 10B active puts its decode at roughly a third the rate, and
78 GiB of weights leaves ~22 GiB of KV on a GB10 — about 13 concurrent requests
at 128K, against the 35B's several times that. Point volume at the 35B and choose
the 122B when the answer is worth the wait.

`glm-4.7-flash` stays defined in the catalogue but is not in `VLLM_MODELS`: the
122B took its card, and a second 3B-active floor alongside the 35B earns nothing.
Its `local/` alias is therefore not registered at all — see
[Registration](#vllm-local).

**Quantisation**

- Default is **NVFP4** (GB10 / RTX 5090 / PRO 5000 / PRO 6000).
- **RTX 4090** has no FP4 support and there is no int4 build of 35B-A3B to
  substitute, so `manage-vllm.sh up` refuses on that card
  ([gpu-memory.md](gpu-memory.md)).

**Parsers**

| Model | Tool parser | Reasoning parser | Notes |
|---|---|---|---|
| `qwen3-coder-30b` | `qwen3_coder` | — | Qwen3-Coder has its own XML dialect; `qwen3_xml` is a different format and silently yields no tool calls. No thinking mode, so no reasoning parser |
| `qwen3.6-27b` | `qwen3_xml` | `qwen3` | Same family plumbing as `qwen3.6-35b` |
| `gemma-4-26b-a4b` | `gemma4` | `gemma4` | Gemma states tool calls in its own `<\|tool>` form, which no generic parser reads. Without `gemma4` the model falls back to ReAct text and the client cannot execute anything |
| `qwen3.6-35b` | `qwen3_xml` | `qwen3` | Thinking is on by default upstream and turned off with `--default-chat-template-kwargs '{"enable_thinking": false}'`. Hybrid Gated-DeltaNet requires `--max-num-seqs` (cudagraph OOM without it) |
| `qwen3.5-122b-a10b` | `qwen3_xml` | `qwen3` | Same architecture family (`Qwen3_5MoeForConditionalGeneration`) and the same chat-template controls, so the same parsers. `--max-num-seqs` is set low here for KV rather than for cudagraph capture |
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
  | `local/qwen3.5-122b-a10b` | `VLLM_QWEN122B_URL` |
  | `local/gemma-4-26b-a4b` | `VLLM_GEMMA26B_URL` |
  | `local/qwen3-coder-30b` | `VLLM_CODER30B_URL` |
  | `local/qwen3.6-27b` | `VLLM_QWEN27B_URL` |
  | `local/glm-4.7-flash` | `VLLM_GLMFLASH_URL` |

- **No URL** — no `local/*` name is created. With an OpenRouter key the model is
  still reachable, but under its own slug (`qwen/qwen3.6-35b-a3b`,
  `z-ai/glm-4.7-flash`) and priced as the paid route it is. A surface that names
  `local/<m>` therefore stops resolving on a GPU-less install and must pick from
  the catalogue — see **Naming a model from outside** below.

- **Discovery** — `gen-litellm-config.sh` polls `/v1/models` at each URL and
  registers only the nodes that answer.
- **Multi-node** — one deployment per node under the same model name. The
  LiteLLM router picks with `least-busy`.
- **Strict aliases** — each configured chat vLLM also registers a
  `strict-local/<model>` alias over the same backend. An empty URL never creates
  that alias, and never creates the plain `local/*` one either.

**Operations**

- **Normally** — the placement step of `setup.sh all` (`scheduler apply`) decides
  what runs where, at which context, and starts it.
- **By hand** — `./scripts/manage-vllm.sh up <service>`.

What fits on which card is in the
[GPU memory guide](gpu-memory.md#per-node-class).

| Model | Node class | Notes |
|---|---|---|
| `qwen3.6-35b` | Any single NVFP4-capable GPU | Unified chat and floor |
| `qwen3.5-122b-a10b` | GB10 or PRO 6000, **alone on the card** | Top chat |
| `gemma-4-26b-a4b` | Any single NVFP4-capable GPU | Second family; shares a card |
| `qwen3-coder-30b` | Any single GPU (FP8 — no FP4 needed) | Coding |
| `qwen3.6-27b` | Any single NVFP4-capable GPU | Dense |
| `glm-4.7-flash` | PRO 5000 and up | Cheap-decode floor (not deployed here) |

**Roles**

- **`qwen3.6-35b`** — default chat, and the deployment volume is pointed at.
  Artifacts, deep research and coding run here; `DEEP_RESEARCH_MODEL` points
  here, and the scheduler holds a 128K context floor on it for that reason. It
  also takes the high-volume internal calls — titles, memory extraction, query
  rewriting — which `glm-4.7-flash` used to carry. Their call sites are named by
  the UI (`KCHAT_TITLE_MODEL`).
- **`qwen3.5-122b-a10b`** — chosen from the picker when the answer is worth the
  latency. Nothing is routed to it by default, and nothing should be: at 10B
  active it decodes roughly three times slower, and its KV pool admits an order
  of magnitude fewer concurrent requests.
- **`qwen3-coder-30b`** and **`qwen3.6-27b`** — specialisations for a cluster
  with cards to spare. Neither is routed to; both are picker choices. They fit
  from three nodes up, and below that the planner delegates them rather than
  taking context away from the 35B — check `scheduler plan` before adding them
  to `VLLM_MODELS`.
- **`gemma-4-26b-a4b`** — the second opinion. Nothing routes here either; it
  exists because when a Qwen answer is wrong it tends to be wrong the same way
  twice, and a different lineage fails differently. 4B active, so it costs about
  what the 35B costs to run.

  It carries `priority: 10` in `models.yaml` against the 35B's `20`, so where a
  cluster cannot hold both, **the 35B keeps the card and this one is delegated**
  — even though it is the larger model and coverage would otherwise seat it
  first. Default chat, titles, memory extraction, deep research and the external
  coding agents all land on the 35B; losing it degrades every path at once, while
  losing the 122B removes a choice from the picker.
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
| `local/qwen3.6-35b` | `qwen/qwen3.6-35b-a3b` ($0.14 / $1.00) |
| `local/qwen3.5-122b-a10b` | `qwen/qwen3.5-122b-a10b` ($0.26 / $2.08) |
| `local/gemma-4-26b-a4b` | `google/gemma-4-26b-a4b-it` ($0.07 / $0.34) |
| `local/qwen3-coder-30b` | `qwen/qwen3-coder-30b-a3b-instruct` ($0.07 / $0.28) |
| `local/qwen3.6-27b` | `qwen/qwen3.6-27b` ($0.30 / $2.00) |
| `local/glm-4.7-flash` | `z-ai/glm-4.7-flash` ($0.06 / $0.40) |

> These prices must match the `emit_brain` and `emit_or_fallback` arguments in
> `gen-litellm-config.sh`. When they change, verify against
> `https://openrouter.ai/api/v1/models` and update both.

- **Emission condition** — `emit_or_fallback` emits the twin only when the local
  primary is deployed (its URL is set), under the OpenRouter slug as model name.
  With no local primary, `emit_brain` registers that same slug as an ordinary
  visible route. The two are mutually exclusive on purpose: both firing would put
  two deployments under one `model_name` and the router would split ordinary
  traffic onto the paid one.
- **Hidden from the picker** — the twin keeps the OpenRouter slug, so it is
  distinct from the local alias and is used only as a fallback.
- **Cost** — a fallback is **paid OpenRouter egress**. A node that dies often
  leaks money.

### Naming a model from outside

A caller that hard-codes `local/<m>` is asserting the install has that GPU
deployment. Where it might not, read the catalogue and fall back:

| Setting | Behaviour when the name is absent |
|---|---|
| `DEEP_RESEARCH_MODEL` (this repo) | Passed through to the deep-research service as-is; set it to a model the install actually serves |
| `KCHAT_DEFAULT_CHAT_MODEL` (UI) | Blanked against the live catalogue; the picker falls back to its cheapest |
| `KCHAT_TITLE_MODEL` (UI) | Blanked against the live catalogue; title and memory extraction use the session's own model |

### Strict-local fail-closed routing

`strict-local/*` is the route for requests that must not leave the self-hosted
vLLM deployment. It has two independent fail-closed controls:

- The generated `router_settings.fallbacks` table never contains a strict alias,
  so node errors, timeouts and cooldown do not select OpenRouter.
- The concurrency gate reads `model_info.kchat_strict_local`. At the same
  saturation threshold used to spill a normal local alias, it returns
  `strict_local_unavailable` without rewriting the model id.

A `/metrics` scrape failure marks strict capacity unavailable and rejects the
request; normal aliases retain their existing fail-open behavior. If vLLM fails
after a healthy capacity check, LiteLLM retries only the deployments registered
under the strict alias and then returns an error. It has no external target to
try.

After introducing strict aliases into an existing LiteLLM installation, run
`./scripts/manage.sh team add-strict`. It preserves each team's current
allowlist and adds a strict alias only where that team already has the matching
`local/*` model. Do not use `team sync` solely for this rollout: that command
intentionally replaces a team's allowlist with the full generated catalogue.

### Spend-log privacy

`general_settings.store_prompts_in_spend_logs` is `false`. LiteLLM continues to
record token usage and cost attribution, but spend rows do not retain prompt or
response bodies. Clients may additionally suppress message logging per request;
that signal is defence in depth, not a substitute for the server default.

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
| `qwen3.6-35b` | 256K (262K native) | Chat, deep research, coding, internal calls |
| `qwen3.5-122b-a10b` | **128K** (262K native) | Top chat. Capped by KV, not by the model |
| `gemma-4-26b-a4b` | 256K | Second opinion — a different family's failure modes |
| `qwen3-coder-30b` | 128K | Coding. Capped by KV: 48 KiB/token is 4.8× the 35B |
| `qwen3.6-27b` | 256K | Dense |

### Embeddings

`bge-m3` (`BAAI/bge-m3`, 1024 dimensions, 8K context, multilingual) serves
KloudChat's retrieval index through `/tools/index`. It is a **pooling** runner,
not a chat one: no tool parser, no reasoning parser, and no KV cache — so
`scheduler/planner.py` charges it weights and activation only. About 2 GiB, so
it rides along on a card already serving a chat model.

Add it to `VLLM_MODELS` like any other. Registered with `mode: embedding`,
which keeps it out of KloudChat's model picker.

With no local deployment and an OpenAI key, `text-embedding-3-small` is
registered as the fallback (~$0.02 per 1M). OpenRouter serves no embedding
models. With neither, KloudChat falls back to lexical retrieval.

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

By use case, where the catalogue would otherwise leave a gap:

| Need | Model | Price /1M |
|---|---|---|
| Bulk work where cost dominates | `qwen/qwen3.7-flash` (1M ctx) | $0.03 / $0.13 |
| Commercial coding | `openai/gpt-5.3-codex`, `qwen/qwen3-coder-plus` | $1.75/$14, $0.65/$3.25 |
| Search that reads more than a snippet | `perplexity/sonar-pro` | $3 / $15 |
| Speech at a tenth the cost | `openai/gpt-audio-mini` | $0.60 / $2.40 audio |

`sonar-pro` does not replace the stack's own deep-research service, which drives
a local model over SearXNG instead of paying per search.

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

## Retrieval

Two stages, both local, both through the gateway.

| Stage | Model | Job |
|---|---|---|
| Recall | `local/bge-m3` (`mode: embedding`) | Nearest passages by cosine distance in pgvector |
| Precision | `local/bge-reranker-v2-m3` (`mode: rerank`) | Reads each (query, passage) pair and scores it |

An embedding compares question and passage in one shared space; a reranker reads
the pair together. That is why 2.2 GiB of reranker separates a relevant passage
from an irrelevant one far more sharply than a much larger embedding model does,
and it is the cheapest quality left on the table for a retrieval layer.

`index-shim` over-fetches `limit × RERANK_CANDIDATES` from pgvector, reranks, and
keeps the top `limit`. **The cuts belong to different stages**: cosine distance
is loosened to a recall bound (`RERANK_RECALL_DISTANCE`) while reranking, because
precision is now the second stage's job. Applying the tuned cosine cut first was
the obvious arrangement and the wrong one — the reranker then only ever saw
passages that had already passed, and marginal candidates are exactly what it is
good at.

Both stages degrade rather than fail. No reranker deployed, or one that cannot be
reached, and search falls back to vector order and the cut that stage was tuned
for — the response says which happened (`"reranked": true|false`). No embedding
deployment and an OpenAI key registers `text-embedding-3-small` as the fallback;
with neither, KloudChat falls back to lexical retrieval.

Adding a stage takes three pieces: an entry in `scheduler/models.yaml`, a service
in `docker-compose.vllm.yml`, and an `emit_vllm_embed` / `emit_vllm_rerank` call
in `gen-litellm-config.sh`.
