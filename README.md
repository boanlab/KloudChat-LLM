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
working tree's images instead, for a service change that is not merged yet.

The run ends by printing the addresses to paste into the UI admin screen. Print
them again at any time:

```bash
./scripts/setup.sh urls
```

### What to fill in

| Variable | Value | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | `sk-or-v1-...` | Commercial models and local fallback. Required without a GPU |
| `HF_TOKEN` | optional | Hugging Face gated repositories, for weight downloads |
| `NODES_VLLM` | `user@host,...` | GPU node SSH targets. The first is the head node (default chat, retrieval, transcription); the rest are the pool (large picker models) |
| `VLLM_MODELS` | model id CSV | What to deploy. Defined in `scheduler/models.yaml` |

`VLLM_*_URL` is written by the placement step inside `setup.sh all`. To manage
those by hand, run it as `KLOUDCHAT_SKIP_SCHEDULER=1 ./scripts/setup.sh all`.

At least one of an OpenRouter key or a vLLM node is required.

### Preparing a GPU node

```bash
./scripts/install-vllm.sh          # vLLM image and GPU runtime check
./scripts/download-vllm-models.sh  # only weights this card can serve
```

A GPU node has one role, `vllm`. Transcription (`whisper-large-v3`) is a vLLM
service like the rest, on any architecture. With `NODES_VLLM` filled in,
`setup.sh all` runs the installer on each node over SSH. Model downloads run on
the node itself.

The downloader inspects the card first: weights that need a format the card
cannot execute, or more memory than it has, are skipped with the reason.

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
another, enable only the profile each machine needs: the UI accepts a different
address per capability.

## Operations

```bash
./scripts/setup.sh up            # restart the stack only (no node install, no placement)
./scripts/setup.sh stop|start    # stop / resume containers, data preserved
./scripts/setup.sh urls          # integration addresses and per-capability status
./scripts/setup.sh clean         # destructive: removes containers and runtime data

./scripts/setup.sh scheduler plan     # compute placement (changes nothing)
./scripts/setup.sh scheduler apply    # apply it
./scripts/manage.sh user usage        # LiteLLM usage and budgets
./scripts/manage-vllm.sh status       # GPU node status (every vLLM service)
```

Every script prints its usage when run without arguments.

## Supported environments

| Environment | Behaviour |
|---|---|
| Linux amd64, no GPU | OpenRouter only |
| Linux amd64 + NVIDIA GPU (RTX 5090 / PRO 5000 / PRO 6000) | Local GPU with OpenRouter fallback |
| Linux arm64, GB10 | Local GPU with OpenRouter fallback |
| AMD / ROCm, Apple, anything else | Not supported. OpenRouter only |

Local serving is NVIDIA-only: detection, the container runtime reservation,
device pinning and the quantisation gate all go through NVIDIA interfaces, and
the default weights are NVFP4, a format with no AMD counterpart.

Two things decide whether a card can serve: size, then format. 32 GiB usable is
the floor. The default weights are NVFP4 and need compute capability 10.0 (GB10,
RTX 5090, PRO 5000/6000); an FP4-less card of 48 GiB or more runs the AWQ int4
build of the chat model instead. Where nothing fits, the placement step says so
and delegates to OpenRouter.

## Documentation

- [`docs/`](docs/): prerequisites, environment variables, models, tools, troubleshooting
- [`scheduler/README.md`](scheduler/README.md): how placement is decided
- [`CONTRIBUTING.md`](CONTRIBUTING.md): layout, checks, conventions
- [`SECURITY.md`](SECURITY.md): threat model and how to report a vulnerability

## License

Apache-2.0, see [LICENSE](LICENSE).
