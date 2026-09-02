# Prerequisites

What must be in place before bringing KloudChat-LLM up. The baseline is
**local GPU first, OpenRouter as fallback**.

- **Using a local GPU**: "Compose host" plus "GPU node requirements"
- **OpenRouter only**: "Compose host" is enough

## Compose host

The stack is one `docker-compose.yml`; `COMPOSE_PROFILES` in `.env` decides
what comes up. The UI (`KloudChat`) is a separate repository connected only by
URL.

| Topology | Composition |
|---|---|
| Single node | The whole backend on one machine, which doubles as the GPU node when serving locally |
| Split | Tools host (`tools`) + model host (`models`) + GPU nodes |

| Requirement | Compose host | GPU node |
|---|---|---|
| OS | Linux amd64 or arm64 | Linux amd64 or arm64 |
| Docker | Compose v2 | Compose v2 |
| Utilities | `jq curl` | `jq curl` |
| Python | 3.11+ with PyYAML (`setup.sh` installs `python3-yaml` through apt) | not needed |
| Disk | 50 GB (images and runtime data) | 100 GB+ (model weights) |
| RAM | 16 GB | 16 GB+ |
| Open ports | `GATEWAY_PORT` (8080 by default) and nothing else | vLLM 8001–8009 and transcription 9000, reachable from the compose host |

- macOS and Windows are unsupported.
- No Docker yet: `curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER`
- RAM scales with `LITELLM_NUM_WORKERS` (4 by default, ~600 MB each) and
  concurrent traffic.
- `setup.sh all` checks Docker and `.env` before anything else.

## GPU node requirements

| Requirement | Minimum |
|---|---|
| NVIDIA GPU | 32 GiB usable. RTX 5090 for the default NVFP4 lineup; an FP4-less card of 48 GB or more runs the int4 build below |
| NVIDIA Container Toolkit | every model, transcription included, is a vLLM container |
| Model disk | 100 GB |

**24 GiB cards are out of scope** (RTX 4090, RTX 3090, L4, A10). The int4 build
executes there, but at 26 GB of weights it does not fit. `manage-vllm.sh up`
refuses below 32 GiB usable.

**NVIDIA only.** The inventory reads capacity and card class from `nvidia-smi`,
compose reserves `driver: nvidia`, and the quantisation gate is written in
compute capability. The default weights are NVFP4, a Blackwell format with no
AMD counterpart.

Quantisation is gated by compute capability, so what a card can serve is a
property of the card rather than of its name:

| Weights | Needs | Cards |
|---|---|---|
| NVFP4 (default lineup) | cc ≥ 10.0 | GB10, RTX 5090, PRO 5000/6000 |
| FP8 | cc ≥ 8.9 | Ada and later, RTX 4090 included |
| AWQ int4 (`qwen3.6-35b-awq`) | cc ≥ 7.5 | Turing and later |

`download-vllm-models.sh` refuses weights the card cannot execute or hold, with
the reason.

VRAM per model:

| Model | Requirement |
|---|---|
| Chat (`qwen3.6-35b`, 21 GiB) | RTX 5090 32 GB minimum, at a reduced context. PRO 5000 48 GB or better recommended |
| Top chat (`qwen3.5-122b-a10b`, 78 GiB) | GB10, or PRO 6000 ×2 with tensor parallelism. Alone on its cards |
| Coding (`qwen3-coder-next`, 75 GiB) | GB10 or PRO 6000, alone on the card |

Deep research runs on `DEEP_RESEARCH_MODEL` (`local/qwen3.5-122b-a10b` by
default), which the scheduler holds at a 128K context floor. Occupancy figures
are in the [GPU memory guide](gpu-memory.md).

### What runs where

- Every model a node serves is a vLLM container from `docker-compose.vllm.yml`.
- Transcription is `vllm-whisper`, serving `openai/whisper-large-v3` (~3.1 GiB)
  from the same image, on any architecture.
- Backends publish their ports; LiteLLM reaches them through `VLLM_*_URL` and
  whisper-shim through `WHISPER_URLS`, both written by the scheduler.

### Transcription (STT)

- Transcribes audio uploaded through `/tools/stt`.
- Optional. With `WHISPER_URLS` empty, LiteLLM registers OpenRouter STT
  instead. Deploy it by adding `whisper-large-v3` to `VLLM_MODELS`.
- Local serving keeps audio inside the network and has no per-token billing.

### Commands to run on a GPU host

```bash
./scripts/install-vllm.sh               # vLLM image + GPU runtime check
./scripts/download-vllm-models.sh       # weights this card can serve, transcription included
```

`./scripts/install-vllm.sh --reinstall` re-pulls the base image and rebuilds
the derived one.

- vLLM base image per architecture: amd64 `vllm/vllm-openai:cu129-nightly`,
  GB10 (arm64) `vllm/vllm-openai:nightly-aarch64`. Compose runs the derived
  `kloudchat-vllm:local`.
- RTX 4090: no FP4, so the default lineup cannot run on it. `qwen3.6-35b-awq`
  can, on 48 GB cards.

## OpenRouter (no GPU required)

- `OPENROUTER_API_KEY` from https://openrouter.ai/keys. A compose host is
  enough.
- Without a local GPU this alone serves commercial models. With one it adds
  automatic fallback to the same model when a node goes down, plus the
  commercial catalogue.
- Commercial models (OpenAI, Anthropic, Google, DeepSeek and others) all go
  through OpenRouter. Direct native APIs are not supported.

## Multiple nodes

- GPU nodes: `NODES_VLLM=user@host,...` in `.env`. The scheduler places models
  and records the URLs.
- `./scripts/setup.sh vllm` rsyncs the repository to every node in
  `NODES_VLLM` and runs `install-vllm.sh` there.

**Adding a node**

1. Append the SSH target to `NODES_VLLM` in `.env`
2. `./scripts/setup.sh vllm`
3. `./scripts/setup.sh all` (or `scheduler apply` followed by
   `docker compose restart whisper-shim`) refreshes placement and URLs

**One-time setup on each remote node**

```bash
# 1) Password-less SSH from the compose host to the node
ssh-copy-id <your-user>@<gpu-node>

# 2) Non-interactive sudo for install-vllm.sh (model directory) and tune-host.sh
ssh <your-user>@<gpu-node> "echo '<your-user> ALL=(ALL) NOPASSWD:ALL' | sudo tee /etc/sudoers.d/kloudchat-<your-user>"
```

**Single node (the compose host is the GPU node)**

- The SSH step can be skipped. `install-vllm.sh`, `tune-host.sh` and
  `setup.sh clean` still call `sudo`; type the password or add the NOPASSWD
  line locally.

### vLLM routing

- The scheduler inventories GPU class and VRAM per node and places models;
  `gen-litellm-config.sh` registers one deployment per node holding each model.
- A model on one node: requests go only there.
- A model on several nodes: the router load-balances `least-busy`.
- Heterogeneous GPUs work as-is: large models on large nodes, small models
  everywhere.

### Transcription routing

- The shim keeps a 10-second `/health` cache to pick reachable nodes, then
  routes by in-flight count.
- Every backend serves the same checkpoint; the shim names it on each request.
- No stickiness. Every call is self-contained.

## DGX Spark (GB10)

- arm64, so vLLM uses the `*-aarch64` image.
- Transcription runs on the card like every other model.
- `nvidia-smi memory.total` reports `[N/A]` on unified memory, so usable VRAM
  is system RAM minus a 12 GiB OS reservation (`lib.sh::UNIFIED_RESERVE_GB`,
  `scheduler/inventory.py::_UNIFIED_RESERVE_BYTES`). The planner budgets with
  the same number.
