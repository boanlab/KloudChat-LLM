# KloudChat-LLM

[![CI](https://github.com/boanlab/KloudChat-LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/boanlab/KloudChat-LLM/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

The backend plane for KloudChat: a model gateway (LiteLLM) and the tools a chat
turn calls, packaged together and exposed through **one gateway port**.

[`KloudChat`](https://github.com/boanlab/KloudChat) connects by entering that
one address in its admin screen. No backend address is compiled into the UI.

```
┌─ KloudChat (UI) ────────┐        ┌─ KloudChat-LLM ─────────────────────────┐
│  web · API · DB         │        │  gateway :8080   ← the only exposed port │
│                         │        │   /litellm/*        → litellm           │
│  admin → integrations   │──URL──▶│   /tools/search/*   → search-shim       │
│   one URL               │        │   /tools/fetch/*    → crawl4ai-shim     │
│                         │        │   /tools/exec/*     → code-interpreter  │
└─────────────────────────┘        │   /tools/research/* → deep-research     │
                                   │   /tools/stt/*      → whisper-shim      │
                                   │   /tools/index/*    → index-shim        │
                                   │                                         │
                                   │  GPU nodes: vllm-* (whisper included)   │
                                   └─────────────────────────────────────────┘
```

`/tools/*` requires no authentication. **The gateway port must only be open
inside a private network**: the code execution endpoint sits behind it. Each
backing store (the two databases, MinIO, redis, valkey) is on an internal
network shared only with the service that owns it. Internal service keys (code
execution, document fetching) are injected by the gateway; the UI never learns
them. Only `/litellm/*` and `/v1/*` pass the caller's key through.

## Quick start

```bash
./scripts/gen-env.sh          # create .env (secrets generated, external keys blank)
$EDITOR .env                  # fill in the table below
./scripts/setup.sh all
```

Images come from Docker Hub. `./scripts/setup.sh all --build` builds this
working tree's images instead.

The run ends by printing the addresses to paste into the UI admin screen.
Print them again at any time:

```bash
./scripts/setup.sh urls
```

### What to fill in

| Variable | Value | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | `sk-or-v1-...` | Commercial models and local fallback. Required without a GPU |
| `HF_TOKEN` | optional | Hugging Face gated repositories, for weight downloads |
| `NODES_VLLM` | `user@host,...` | GPU node SSH targets. The first is the head node (retrieval, transcription); the rest are the pool (the 122B, the coder) |
| `VLLM_MODELS` | model id CSV | What to deploy. Defined in `scheduler/models.yaml` |

`VLLM_*_URL` is written by the placement step inside `setup.sh all`. To manage
those by hand, run `KLOUDCHAT_SKIP_SCHEDULER=1 ./scripts/setup.sh all`.

At least one of an OpenRouter key or a vLLM node is required.

### Preparing a GPU node

```bash
./scripts/install-vllm.sh          # vLLM image and GPU runtime check
./scripts/download-vllm-models.sh  # only weights this card can serve
```

Every model a GPU node serves, transcription (`whisper-large-v3`) included, is
a vLLM container. With `NODES_VLLM` filled in, `setup.sh all` runs the
installer on each node over SSH; model downloads run on the node itself.

The downloader inspects the card first: weights that need more memory than the
card has, or a card outside the supported set, are refused with the reason.

## Layout

```
docker-compose.yml          gateway + tools + LiteLLM (composed by profiles)
docker-compose.vllm.yml     what a GPU node serves: every vLLM service
docs/                       operator documentation
scheduler/                  model placement (its own README inside)
scripts/                    setup · config generation · node install · operations
services/                   one directory per service: Dockerfile, source, config
```

### Profiles

`COMPOSE_PROFILES` in `.env` decides what comes up.

| Profile | Services |
|---|---|
| `tools` | gateway · web search · document fetch · code execution · deep research |
| `models` | LiteLLM and its database |
| `whisper` | transcription shim. `setup.sh` enables it once the model is placed |
| `index` | retrieval index (pgvector) and its shim |

The default is `tools,models`. To put tools on one machine and models on
another, enable only the profile each machine needs: the UI accepts a
different address per capability.

## Operations

```bash
./scripts/setup.sh up            # restart the stack only (no node install, no placement)
./scripts/setup.sh stop|start    # stop / resume containers, data preserved
./scripts/setup.sh urls          # integration addresses and per-capability status
./scripts/setup.sh clean         # destructive: removes containers and runtime data

./scripts/setup.sh scheduler plan     # compute placement (changes nothing)
./scripts/setup.sh scheduler apply    # apply it
./scripts/manage-vllm.sh status       # GPU node status (every vLLM service)

LITELLM_URL=http://localhost:8080/litellm ./scripts/manage.sh user usage   # LiteLLM usage and budgets
```

LiteLLM publishes no host port; `manage.sh` reaches it through `LITELLM_URL`
(shell or `.env`), which should point at the gateway. Every script prints its
usage when run without arguments.

## Supported environments

| Environment | Behaviour |
|---|---|
| Linux amd64, no GPU | OpenRouter only |
| Linux amd64 + RTX 5090 / RTX PRO 5000 Blackwell / RTX PRO 6000 Blackwell | Local GPU with OpenRouter fallback |
| Linux arm64, GB10 | Local GPU with OpenRouter fallback |
| Any other card, AMD / ROCm, Apple, macOS, Windows | Not supported. OpenRouter only |

The local lineup is `qwen3.8-27b` (chat, vision, coding, deep research),
`qwen3.5-122b` (quality and judging, a GPU node to itself), `bge-m3` and
`bge-reranker-v2-m3` (retrieval) and `whisper-large-v3` (transcription);
`qwen3-coder-next` (coding) is in the catalogue but not deployed by default.
See [docs/models.md](docs/models.md). Local serving is NVIDIA-only. Supported
cards are GB10, RTX 5090, RTX PRO 5000 and RTX PRO 6000; the download and
manage scripts refuse anything else and the scheduler places nothing there.
32 GiB usable is the floor. Where a model does not fit, the placement step
says so and delegates to OpenRouter.

## Documentation

- [`docs/`](docs/): prerequisites, environment variables, models, tools, troubleshooting
- [`scheduler/README.md`](scheduler/README.md): how placement is decided
- [`CONTRIBUTING.md`](CONTRIBUTING.md): layout, checks, conventions
- [`SECURITY.md`](SECURITY.md): threat model and how to report a vulnerability

## License

Apache-2.0, see [LICENSE](LICENSE).
