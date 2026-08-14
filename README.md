# KloudChat-LLM

[![CI](https://github.com/boanlab/KloudChat-LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/boanlab/KloudChat-LLM/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

The backend plane for KloudChat: a model gateway (LiteLLM) and the tools a chat
turn calls, packaged together and exposed through **a single gateway port**.

[`KloudChat`](https://github.com/boanlab/KloudChat) connects by entering
that one address in its admin screen. No backend address is compiled into the UI.

```
┌─ KloudChat (UI) ────────┐        ┌─ KloudChat-LLM ─────────────────────────┐
│  web · API · DB         │        │  gateway :8080   ← the only exposed port │
│                         │        │   /litellm/*        → litellm           │
│  admin → integrations   │──URL──▶│   /tools/search/*   → searxng           │
│   one URL               │        │   /tools/fetch/*    → crawl4ai-shim     │
│                         │        │   /tools/exec/*     → code-interpreter  │
└─────────────────────────┘        │   /tools/research/* → deep-research     │
                                   │   /tools/stt/*      → whisper-shim      │
                                   │                                         │
                                   │  GPU nodes: vllm-* + whisper            │
                                   └─────────────────────────────────────────┘
```

`/tools/*` requires no authentication — **the gateway port must only be open
inside a private network**, because the code execution endpoint sits behind it.
Internal service keys (code execution, document fetching) are injected by the
gateway, so the UI never learns them and caller-supplied credentials are ignored.
Only `/litellm/*` passes the caller's key through, alongside `/v1/*` for
OpenAI-compatible clients that connect directly.

## Quick start

```bash
./scripts/gen-env.sh          # create .env (secrets generated, external keys blank)
$EDITOR .env                  # fill in the table below
./scripts/setup.sh all
```

The run ends by printing the addresses to paste into the UI admin screen. Print
them again at any time:

```bash
./scripts/setup.sh urls
```

### What to fill in

| Variable | Value | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | `sk-or-v1-...` | Commercial models and local fallback. Required without a GPU |
| `HF_TOKEN` | optional | Hugging Face gated repositories — weight downloads |
| `NODES_VLLM` | `user@host,...` | GPU node SSH targets. Only when serving models locally |
| `VLLM_MODELS` | model id CSV | What to deploy. Defined in `scheduler/models.yaml` |

Do not write `VLLM_*_URL` by hand — the placement step inside `setup.sh all`
decides those and records them in `.env`. To manage them yourself, set
`KLOUDCHAT_SKIP_SCHEDULER=1` and fill in the URLs.

You need **at least one** of an OpenRouter key or a vLLM node.

### Preparing a GPU node

```bash
./scripts/install-vllm.sh          # vLLM plus the transcription backend
./scripts/download-vllm-models.sh  # only weights this card can actually serve
```

A GPU node has one role, `vllm`: transcription is installed on the same node
(amd64 only — arm64/GB10 keeps no backend and delegates STT to OpenRouter).
Filling in `NODES_VLLM` lets `setup.sh all` run the installer on each node over
SSH. **Model downloads always run on the node itself.**

The downloader inspects the card first. Weights that need FP4 the card cannot
execute, or more memory than it has, are skipped with the reason — no 20 GB
download that ends in an engine that will not start.

## Layout

```
docker-compose.yml          gateway + tools + LiteLLM (composed by profiles)
docker-compose.vllm.yml     what a GPU node serves — vLLM + transcription
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
| `whisper` | transcription shim — `setup.sh` enables it once a backend answers |

The default is `tools,models`. To put tools on one machine and models on
another, enable only the profile each machine needs: the UI accepts a different
address per capability.

## Operations

```bash
./scripts/setup.sh up            # restart the stack only (no node install, no placement)
./scripts/setup.sh stop|start    # stop / resume containers, data preserved
./scripts/setup.sh urls          # integration addresses and per-capability status
./scripts/setup.sh clean         # destructive — removes containers and runtime data

./scripts/setup.sh scheduler plan     # compute placement (changes nothing)
./scripts/setup.sh scheduler apply    # apply it
./scripts/manage.sh user usage        # LiteLLM usage and budgets
./scripts/manage-vllm.sh status       # GPU node status (vLLM, transcription)
```

Every script prints its usage when run without arguments.

## Supported environments

| Environment | Behaviour |
|---|---|
| Linux amd64, no GPU | OpenRouter only |
| Linux amd64 + NVIDIA GPU (RTX 5090 / PRO 5000 / PRO 6000) | Local GPU with OpenRouter fallback |
| Linux arm64 — GB10 | Local GPU with OpenRouter fallback |

The default weights are NVFP4. A card without FP4 support (RTX 4090) cannot host
this lineup; the placement step says so explicitly and delegates to OpenRouter.

## Documentation

- [`docs/`](docs/) — prerequisites, environment variables, models, tools, troubleshooting
- [`scheduler/README.md`](scheduler/README.md) — how placement is decided
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — layout, checks, conventions
- [`SECURITY.md`](SECURITY.md) — threat model and how to report a vulnerability

## License

Apache-2.0 — see [LICENSE](LICENSE).
