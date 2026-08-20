# Prerequisites

What must be in place before bringing KloudChat up. The baseline is the single
setup described in the [README](../README.md#quick-start): **local GPU first,
OpenRouter as fallback**.

- **Using a local GPU** — "Common: compose host" plus "GPU node requirements"
- **OpenRouter only** — "Common: compose host" is enough

## Common: compose host

The stack in this repository is one `docker-compose.yml`, and `COMPOSE_PROFILES`
in `.env` decides what comes up. The UI (`KloudChat`) is a separate repository
connected only by URL.

| Topology | Composition |
|---|---|
| Single node | The whole backend on one machine, which doubles as the GPU node if you serve locally |
| Split | Tools host (`tools`) + model host (`models`) + GPU nodes |

| Requirement | Compose host | GPU node |
|---|---|---|
| OS | Linux amd64 or arm64 | Linux amd64 or arm64 |
| Docker | Compose v2 | Compose v2 |
| Utilities | `jq curl wget` | `jq curl wget` |
| Disk | 50 GB (images and runtime data) | 100 GB+ (model weights) |
| RAM | 16 GB | 16 GB+ |
| Open ports | `GATEWAY_PORT` (8080 by default) and nothing else | vLLM 8001/8002 and transcription 9000, reachable from the compose host |

- **macOS and Windows are unsupported.**
- **No Docker yet** — `curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER`
- **RAM** scales with worker count (`LITELLM_NUM_WORKERS`, 4 by default at
  ~600 MB each) and concurrent traffic.
- **Automatic verification** — step 0 of `setup.sh <role>` checks the items above.

## GPU node requirements (vLLM)

| Requirement | Minimum |
|---|---|
| NVIDIA GPU | 32 GiB usable. RTX 5090 for the default NVFP4 lineup; an FP4-less card of 48 GB or more runs the int4 aliases below |
| Model disk | 100 GB |

**24 GiB cards are out of scope** — RTX 4090, RTX 3090, L4, A10. Not for want of
a format: the int4 aliases execute there. There is nowhere to put the model.
Measured across the catalogue on 24 GiB, exactly one entry places —
`gemma-4-26b-a4b-awq`, at its 32K floor and 0.92 of the card, with no room to
grow and nothing left to share the card with. The same builds on a 48 GiB
FP4-less card place at 128K–256K. `manage-vllm.sh up` refuses below the floor, by
usable VRAM rather than by card name.

**NVIDIA only.** AMD/ROCm is out of scope, and the reason is not that nobody got
to it. `nvidia-smi` is what the inventory reads capacity, card class and
per-process memory from; the compose services reserve `driver: nvidia`; devices
are pinned with `CUDA_VISIBLE_DEVICES`; the MLA attention backends are CUDA
kernels; and `gpu_supports_quant` gates on compute capability, which AMD does not
have. Each of those has an equivalent — but the default weights are **NVFP4, a
Blackwell format with no AMD counterpart**, so ROCm support means maintaining a
second model lineup with its own measured weights, KV parameters and verified
parsers. That is a project, and one that cannot be written blind: a backend
nobody has run is worse than an honest "not supported".

Quantisation is gated by compute capability, so what a card can serve is a
property of the card rather than of its name:

| Weights | Needs | Cards |
|---|---|---|
| NVFP4 (default lineup) | cc ≥ 10.0 | GB10, RTX 5090, PRO 5000/6000 |
| FP8 | cc ≥ 8.9 | Ada and later — includes RTX 4090 |
| AWQ / GPTQ int4 | cc ≥ 7.5 | Turing and later |

`download-vllm-models.sh` refuses weights the card cannot execute, with the
reason, rather than letting it fail at engine start. An RTX 4090 cannot run the
default lineup and can run the `*-awq` aliases.

VRAM per model:

| Model | Requirement |
|---|---|
| Chat (`qwen3.6-35b`) | RTX 5090 32 GB with NVFP4 minimum — placed at 128K and 2 concurrent sessions there; PRO 5000 48 GB or better recommended |
| Chat + floor | PRO 5000 48 GB (~41 GiB of weights) |

Deep research does not place a model of its own. The constraint it imposes is
`ctx_floor: 131072` on `qwen3.6-35b` in `scheduler/models.yaml` — below 128K it
loses accumulated context — so large nodes (PRO 6000, GB10) are recommended when
it is used heavily. On a unified-memory node that is tight on headroom the
planner already subtracts 12 GiB for the OS; shrink `VLLM_MODELS` if it is still
tight.

### What runs where

- **NVIDIA Container Toolkit required** — both chat and floor models are served
  by vLLM containers.
- **Transcription backend** — the last step of `install-vllm.sh`:
  - **amd64 (RTX/PRO) nodes** — a container, the `whisper` service in
    `docker-compose.vllm.yml`, pulled as `boanlab/kloudchat-whisper` (built on
    the node with `--reinstall`).
  - **GB10 (arm64) nodes** — not installed. aarch64 ctranslate2 wheels are
    CPU-only, so the card cannot be used and STT is delegated to OpenRouter
    (`voxtral-small-24b`).
- **Wiring** — backends are published on their ports; the whisper-shim container
  calls them over HTTP using `WHISPER_URLS`.

### Transcription (STT)

- **What it is for** — transcribing uploaded audio, and YouTube videos without
  subtitles.
- **It is optional.** With no backend (`WHISPER_URLS` empty) LiteLLM registers
  OpenRouter STT instead. GPU-less deployments and arm64-only clusters land here.
- **Why run it locally** — audio never leaves the network, and there is no
  per-token billing.

### Commands to run on a GPU host

```bash
./scripts/install-vllm.sh               # vLLM image + GPU runtime check + transcription
./scripts/download-vllm-models.sh       # weights this card can serve, plus transcription prewarm
```

A GPU node has one role. There is no separate transcription install: reinstall
with `./scripts/install-vllm.sh --reinstall`, or set the node up without
transcription using `--no-whisper`.

> `download-vllm-models.sh` inspects the card before downloading — compute
> capability for FP4/FP8 support, usable VRAM for capacity. Weights it cannot
> serve are skipped with the reason.

> The transcription backend **pulls** `boanlab/kloudchat-whisper` from Docker Hub
> (published by the release workflow, or `./scripts/build-push-images.sh whisper`;
> amd64 only). `--reinstall` builds it on the node instead.

- **Models served by vLLM** — chat (`qwen3.6-35b`) and floor (`glm-4.7-flash`)
- **vLLM image per architecture**
  - **amd64 (RTX 5090 / PRO 5000 / PRO 6000)** — `vllm/vllm-openai:cu129-nightly`
  - **GB10 (arm64)** — `vllm/vllm-openai:nightly-aarch64`
- **RTX 4090** — no FP4 support, so the default lineup cannot run on it. The
  `*-awq` aliases can; see the quantisation table above.

Occupancy figures are in the [GPU memory guide](gpu-memory.md).

## OpenRouter (no GPU required)

- `OPENROUTER_API_KEY` from https://openrouter.ai/keys — **no other
  prerequisites**. A compose host is enough.
- Without a local GPU this alone works, serving commercial models. With one it
  adds **automatic fallback to the same model when a node goes down**, plus the
  commercial frontier catalogue.
- Commercial models (OpenAI, Anthropic, Google, DeepSeek and others) all go
  through OpenRouter. Direct native APIs are not supported.

## Multiple nodes

- **GPU nodes** — one setting: `NODES_VLLM=user@host,...` in `.env`. The
  scheduler automates vLLM placement and records the transcription backends that
  answered in `WHISPER_URLS`, which the shim load-balances across.
- **Per node** — the same install and download steps apply, either by SSH-ing
  into each node, or in one command: `./scripts/setup.sh vllm` rsyncs the
  repository to every node in `NODES_VLLM` and runs `install-vllm.sh` there.

**Adding a node**

1. Append the SSH target to `NODES_VLLM` in `.env`
2. `./scripts/setup.sh vllm` — installs vLLM and the transcription backend there
3. `./scripts/setup.sh all` (or `scheduler apply` followed by
   `docker compose restart whisper-shim`) — refreshes placement and URLs

> To use a host outside `NODES_VLLM` for transcription only, add
> `http://<node-ip>:<port>` to `WHISPER_URLS` on the compose host by hand and
> restart the shim.

**One-time setup on each remote node**

```bash
# 1) Password-less SSH from the control node to the remote node
ssh-copy-id <your-user>@<gpu-node>

# 2) The install and manage scripts call sudo non-interactively
ssh <your-user>@<gpu-node> "echo '<your-user> ALL=(ALL) NOPASSWD:ALL' | sudo tee /etc/sudoers.d/kloudchat-<your-user>"
```

> The installers need sudo for `apt`, `systemctl` and drop-in files. Without step
> 2 they stop at a password prompt halfway through.

**Single node (the compose host is the GPU node)**

- The SSH step can be skipped, but `install-vllm.sh` (including transcription),
  `manage-vllm.sh`, `tune-host.sh` and `setup.sh clean` still call `sudo`
  directly.
- Either type the password interactively, or add the same NOPASSWD line to a
  local `/etc/sudoers.d/kloudchat-<your-user>`.

### vLLM routing

- The scheduler inventories GPU class and VRAM per node and places models;
  `gen-litellm-config.sh` then registers one deployment per node holding each
  model. What fits where is in the
  [GPU memory guide](gpu-memory.md#per-node-class).
- **A model on one node** — requests go only there.
- **A model on several nodes** — the router load-balances `least-busy`.
- **Heterogeneous GPUs** — large models on large nodes, small models everywhere,
  works as-is.

### Transcription routing

- The shim keeps a 10-second `/health` cache to pick reachable nodes, then routes
  by in-flight count.
- **Same model everywhere** — all backends are assumed to serve the same
  `WHISPER_MODEL`. Per-node models are not supported.
- **No stickiness** — every call is self-contained.

## DGX Spark (GB10)

- **Transcription is not installed.** aarch64 ctranslate2 wheels are CPU-only, so
  the card cannot be used, and CPU transcription would share unified memory with
  vLLM at roughly a third of realtime. STT on these nodes is delegated to
  OpenRouter (`voxtral-small-24b`) — an empty `WHISPER_URLS` is the switch.
- **vLLM** uses the `*-aarch64` image (GB10 is arm64).
- **VRAM detection** — `nvidia-smi memory.total` reports `[N/A]`, so
  `lib.sh::gpu_usable_vram_gb` uses system RAM minus a 12 GiB OS reservation. The
  planner budgets with the same number.
