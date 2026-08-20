# Operator documentation

What you need to bring the backend plane up and keep it running. This page is
the index: it points at where each thing is changed.

## Starting out

1. [Prerequisites](prerequisites.md) — hardware and software checklist
2. [Environment variables](env-reference.md) — filling in `.env`
3. Bring it up with `./scripts/setup.sh all`
4. Paste the addresses from `./scripts/setup.sh urls` into the UI admin screen
5. If something is wrong, go to [Troubleshooting](troubleshooting.md)

## Structure

One gateway port is exposed. Behind it, seven capabilities are split by path.

| Path | Service | Profile |
|---|---|---|
| `/litellm/*`, `/v1/*` | LiteLLM | `models` |
| `/tools/search/*` | SearXNG | `tools` |
| `/tools/fetch/*` | crawl4ai-shim | `tools` |
| `/tools/exec/*` | code-interpreter | `tools` |
| `/tools/research/*` | deep-research (MCP) | `tools` |
| `/tools/stt/*` | whisper-shim | `whisper` |
| `/tools/index/*` | index-shim + pgvector | `index` |

GPU nodes live outside this stack. A node runs vLLM and transcription together —
there is one node list, `NODES_VLLM` — and LiteLLM and whisper-shim call them at
the URLs recorded in `.env`. Those URLs are decided by the
[scheduler](../scheduler/README.md): vLLM from the placement result,
transcription from the backends that answered a health probe. The transcription
backend is amd64 only, so on an arm64-only cluster `WHISPER_URLS` stays empty and
STT goes to OpenRouter.

`/tools/*` is unauthenticated. The gateway port must only be open inside a
private network.

Compose puts the backing stores on their own networks, each shared with exactly
the one service that owns it: `litellm-db` with LiteLLM, `index-db` with
index-shim, MinIO and redis with code-interpreter, valkey with SearXNG. None of
them publish a port, and their networks are `internal`, so the databases are
reachable only through the service in front of them — not from the rest of the
stack, and not from the host. That matters most for code-interpreter: it runs
user-supplied code on the service plane, and cannot open a socket to a database
it does not own.

## Documents

**Setup and when something breaks**

- [Prerequisites](prerequisites.md) — hardware, multi-node SSH
- [Troubleshooting](troubleshooting.md) — first checks, restart loops, vLLM cold-start failures, where the logs are
- [Environment variables](env-reference.md) — every `.env` key and the generated secrets

**Models and nodes**

- [Model configuration](models.md) — routing, adding models, OpenRouter fallback
- [Scheduler](../scheduler/README.md) — which model lands on which node
- [GPU memory](gpu-memory.md) — what fits on each node class, and the vLLM tuning knobs

**Tools**

- [Tools](tools.md) — what the six services are and how requests reach them
- [Deep research internals](internal/deep-research.md) — the LDR sidecar

## Where to change what

| Goal | Look at |
|---|---|
| Add a commercial OpenRouter model | [models](models.md) |
| Wire an OpenRouter fallback to a local model | [models](models.md) |
| Change which models are deployed | `VLLM_MODELS` in `.env` plus [scheduler/models.yaml](../scheduler/models.yaml) |
| Add a node or rebalance | [scheduler](../scheduler/README.md) |
| Adjust vLLM memory or context | [GPU memory](gpu-memory.md#tuning-knobs) |
| Tune OpenRouter overflow under load | [env-reference](env-reference.md) |
| Choose which services run | `COMPOSE_PROFILES` in `.env` |
| Diagnose something that will not start | [troubleshooting](troubleshooting.md) |

Agent instructions, MCP connectors and user management belong to the UI — change
those in the `KloudChat` admin screen, not in this repository.
