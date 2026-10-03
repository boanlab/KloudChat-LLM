# GPU memory guide

What fits on which GPU, and how much VRAM each model occupies. Read this when
sizing a node or diagnosing an OOM.

> Figures cover every model in `scheduler/models.yaml`, deployed or not. There
> is no local media backend: images, audio and video pass through to
> OpenRouter.

## The lineup

| Model | Quant | Weights | Role |
|---|---|---:|---|
| `qwen3.8-27b` (Qwen3.8-27B) | NVFP4 | **21.3 GiB** (measured) | Chat. Dense, vision, 262144 context, MTP speculative decoding |
| `gemma-4-26b-a4b` (Gemma 4 26B-A4B) | NVFP4 | **16.4 GiB** (measured) | Fast chat. MoE 3.8B active, vision, 131072 here |
| `qwen3-coder-next` (Qwen3-Coder-Next-80B-A3B) | FP8 | **~75 GiB** (on disk) | Coding. Hybrid attention, 262144 context. A card to itself |
| `bge-m3` | BF16 | **~2 GiB** | Retrieval embeddings. Pooling, shares a card |
| `bge-reranker-v2-m3` | BF16 | **2.1 GiB** (measured) | Retrieval reranking. Pooling, shares a card |
| `whisper-large-v3` | FP16 | **3.1 GiB** (measured) | Transcription |

- NVFP4, FP8 and BF16/FP16 all execute on every supported card (GB10,
  RTX 5090, RTX PRO 5000, RTX PRO 6000).
- Weight figures are `safetensors` totals on disk. The scheduler measures them
  over SSH (`model_metadata`); `models.yaml` carries a `weight:` override only
  when that measurement is wrong.
- Overestimating a weight makes the planner delegate; underestimating produces
  an OOM at startup.

## KV cost

Only 16 of the 64 layers in `qwen3.8-27b` carry KV; the other 48 are linear
attention with fixed-size state. `gemma-4-26b-a4b` holds KV on 4 of 30
layers; the other 26 are 1024-token sliding windows. `qwen3-coder-next` holds
KV on 12 of 48 layers.

| Model | KV per token (fp8) | 128K | 256K |
|---|---:|---:|---:|
| `qwen3.8-27b` | **32 KiB** | 4.0 GiB | 8.0 GiB |
| `gemma-4-26b-a4b` | **16 KiB** | 2.0 GiB | 4.0 GiB |
| `qwen3-coder-next` | **12 KiB** | 1.5 GiB | 3.0 GiB |

Measured on GB10, from vLLM's `GPU KV cache size` line:

| Model | util | KV tokens | Concurrency |
|---|---:|---:|---|
| `qwen3.8-27b` | 0.56 | 993K | **3.8** sessions at 256K (MTP 5) |
| `gemma-4-26b-a4b` | 0.28 | 486K | **3.7** sessions at 128K |

Pooling models hold no KV: `planner.kv_bytes` returns 0 and the charge is
weights plus activation.

## Runtime overhead

KV is what remains of `gpu_util × VRAM` after weights and runtime overhead.
The planner charges 10 GiB for a generate runner (`planner.ACTIVATION_BYTES`)
and 2 GiB for a pooling one (`POOLING_ACTIVATION_BYTES`), capped at 12% of the
card (`ACTIVATION_MAX_FRACTION`). Measured on GB10 for `qwen3.8-27b` at util
0.56: budget 68.1 GiB, weights 21.3 GiB, KV 30.3 GiB (993K tokens), overhead
16.5 GiB with the MTP draft head and its KV.

If the utilisation figures of co-resident containers sum past 1.0, whichever
starts last gets only the remainder; on unified-memory nodes the OS and page
cache need their share too.

## Per node class

The planner seats a model only at its context floor or above. For
`qwen3.8-27b` the floor is the native 262144, so the need is weights +
activation + 8.8 GiB of KV per concurrent 256K session (admission margin
1.10), against the card minus its reserve (8%, clamped to 1–8 GiB; 12 GiB on
unified memory).

| Node | VRAM | `qwen3.8-27b` | `gemma-4-26b-a4b` | `qwen3-coder-next` | Notes |
|---|---:|---|---|---|---|
| RTX 5090 | 32 G | ✗ | ○ | ✗ | 29.4 GiB of capacity against the 27B's 33.9 GiB need; the 27B is delegated. The speed tier, retrieval and transcription fit |
| PRO 5000 | 48 G | ○ 1 session | ○ | ✗ | The 27B alone, or the speed tier beside retrieval |
| PRO 6000 | 96 G | ○ 4 sessions | ○ beside the 27B | ○ alone | |
| GB10 | 128 G (unified) | ○ 4 sessions | ○ beside the 27B (128K) | ○ alone | System RAM less the 12 GiB reserve |

Sessions are the planner's sizing assumption (`concurrent_sessions`, default
4), halved down to what the card holds.

- GB10 is unified memory, so `nvidia-smi` reports free VRAM as `[N/A]`. The
  planner takes the total from `/proc/meminfo` and subtracts 12 GiB for the OS
  (`scheduler/inventory.py::_UNIFIED_RESERVE_BYTES` and
  `lib.sh::UNIFIED_RESERVE_GB`; the two must agree).
- GPU memory held by processes outside this stack (anything not in a `vllm-*`
  container) is measured per card from `nvidia-smi` and charged to the card
  holding it. A card classified `unsupported` holds nothing; the plan notes
  it.
- A node with more than one card is packed per card. `gpu_util` is a fraction
  of one device; the scheduler assigns device ordinals and writes them as
  `VLLM_<PREFIX>_DEVICES`, which compose passes as `NVIDIA_VISIBLE_DEVICES`.
  Not `CUDA_VISIBLE_DEVICES`: it has no value meaning "every card", and on
  GB10 it fails engine init. An unpinned node gets `all`.
- `qwen3-coder-next` needs a card to itself and is a pool model: the head node
  holds retrieval and transcription.

## Tuning knobs

Normally nobody sets these. The placement step of `setup.sh all`
(`python3 -m scheduler apply`) measures node capacity, computes `MAX_LEN` and
`GPU_UTIL` per model, and writes them into that node's `.env` (see
[scheduler](../scheduler/README.md)). The values below are the `.env.example`
and compose defaults used when placement is skipped
(`KLOUDCHAT_SKIP_SCHEDULER=1`).

| Variable | Default | Role |
|---|---|---|
| `VLLM_QWEN27B_GPU_UTIL` | `0.55` | 21.3 GiB of weights plus KV, mm-budget and cudagraph capture |
| `VLLM_QWEN27B_MAX_LEN` | `262144` | Native context |
| `VLLM_QWEN27B_MAX_BATCHED_TOKENS` | `16384` | Lower bound for the vision mm-budget |
| `VLLM_QWEN27B_SPEC_TOKENS` | `5` | MTP draft tokens per step |
| `VLLM_QWEN27B_MAX_NUM_SEQS` | `64` | The hybrid conv-state cache bounds CUDA-graph capture; unset, capture OOMs |
| `VLLM_CODERNEXT_GPU_UTIL` / `_MAX_LEN` | `0.85` / `262144` | 75 GiB of weights; 12 KiB/token keeps the native context affordable |
| `VLLM_BGEM3_GPU_UTIL` / `_MAX_LEN` | `0.08` / `8192` | Compose defaults |
| `VLLM_RERANK_GPU_UTIL` / `_MAX_LEN` | `0.06` / `8192` | Compose defaults |
| `VLLM_WHISPER_GPU_UTIL` / `_MAX_LEN` | `0.14` / `448` | Compose defaults; 448 is the decoder window |

- `*_MAX_LEN` on the node is what vLLM serves. `gen-litellm-config.sh`
  discovers `max_model_len` from each node's `/v1/models` and emits it per
  deployment; only when discovery fails does `CTX_FALLBACK` (32768) apply.
- `manage-vllm.sh up` without a service name starts every vLLM whose weights
  are on the node. On a shared card that can sum the utilisation fractions
  past 1.0. When driving a node by hand, name the service.
- Parsers and `--kv-cache-dtype fp8` are set in `docker-compose.vllm.yml`.
  Healthcheck `start_period` is 600 s for the chat services and 120 s for
  pooling and transcription; `unhealthy` inside that window is normal.

## Transcription (STT)

`openai/whisper-large-v3` served by vLLM as `vllm-whisper`, placed and sized
like every other model.

| | |
|---|---|
| Weights | **3.1 GiB** measured (the repository ships several serialisations; vLLM reads one) |
| Context | 448, the decoder window. Longer clips are split by the server |
| Placement | `head`, priority 4, below `bge-m3`: losing embeddings turns retrieval lexical, losing this one still transcribes through OpenRouter |
| Sessions | `concurrent_sessions: 4` |

`WHISPER_URLS` is written from the placement. Empty (no room, or no GPU node)
sends STT to OpenRouter (`STT_OR_MODEL`).

## Throughput

Measured on GB10, warm, 400-token generations:

| Model | Condition | tok/s |
|---|---|---:|
| `qwen3.8-27b` | Single stream, prose, MTP 5 | 15 |
| `qwen3.8-27b` | Single stream, code, MTP 5 | 24 |
| `qwen3.8-27b` | Single stream, no speculation | 11 |
| `qwen3.8-27b` | 4 / 8 / 16 concurrent requests, aggregate | 45 / 81 / 141 |
| `gemma-4-26b-a4b` | Single stream, prose or code | 48 |
| `gemma-4-26b-a4b` | 8 / 16 concurrent requests, aggregate | 291 / 494 |

One instance batches; a second instance on the same card adds only a second
copy of the weights. Utilisation sets KV capacity, not speed.
