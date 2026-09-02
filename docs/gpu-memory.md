# GPU memory guide

What fits on which GPU, and how much VRAM each model occupies. Read this when
sizing a node or diagnosing an OOM.

> Figures cover every model in `scheduler/models.yaml`, deployed or not. There
> is no local media backend: images, audio and video pass through to
> OpenRouter.

## The lineup

| Model | Quant | Weights | Role |
|---|---|---:|---|
| `qwen3.6-35b` (Qwen3.6-35B-A3B) | NVFP4 | **21.4 GiB** (measured) | Chat and floor. Vision, 262K context, agentic coding |
| `qwen3.5-122b-a10b` (Qwen3.5-122B-A10B) | NVFP4 | **77.8 GiB** (measured) | Top chat. Vision, 128K here, 10B active |
| `qwen3-coder-next` (Qwen3-Coder-Next-80B-A3B) | FP8 | **~75 GiB** (on disk) | Coding. Hybrid attention, 262K. Wants the card to itself |
| `qwen3-coder-30b` (Qwen3-Coder-30B-A3B) | FP8 | **33.0 GiB** (measured) | Coding. Runs without FP4 support |
| `qwen3.6-27b` (Qwen3.6-27B) | NVFP4 | **20.4 GiB** (measured) | The one dense model |
| `bge-m3` | BF16 | **~2 GiB** | Retrieval embeddings. Pooling, shares a card |
| `bge-reranker-v2-m3` | BF16 | **2.1 GiB** (measured) | Retrieval reranking. Pooling, shares a card |
| `whisper-large-v3` | FP16 | **3.1 GiB** (measured) | Transcription |

- The default lineup is NVFP4 only, which needs compute capability 10.0. A card
  without FP4 runs `qwen3.6-35b-awq` instead (same served model, different
  checkpoint directory); `gpu_supports_quant` decides by capability.
- Weight figures are measured `safetensors` totals. The `-NVFP4` checkpoints
  mix 4-bit and 8-bit groups, which puts them 5–6 GiB above a pure-FP4
  calculation. Do not re-derive them from parameter counts.
- The direction of an error matters: overestimating makes the planner discard
  low-utilisation configurations and delegate; underestimating produces an OOM
  at startup.

## KV cost

Only 10 of the 40 layers in `qwen3.6-35b` carry KV (Gated-DeltaNet to
full-attention is 3:1). The other 30 are linear attention with fixed-size
state.

| Model | KV per token (fp8) | 128K | 256K |
|---|---:|---:|---:|
| `qwen3.6-35b` | **10 KiB** | 1.2 GiB | 2.5 GiB |
| `qwen3.5-122b-a10b` | **12 KiB** | 1.5 GiB | (128K here) |
| `qwen3-coder-next` (hybrid) | **12 KiB** | 1.5 GiB | 3.0 GiB |
| `qwen3-coder-30b` | **48 KiB** | 6.0 GiB | 12.0 GiB |
| `qwen3.6-27b` | 32 KiB | 4.0 GiB | 8.0 GiB |

The 122B is 3.5× the 35B in parameters but nearly the same in KV: 12 of 48
layers carry KV against 10 of 40, at the same 2 KV heads and 256 head dim.
Context is not what makes the 122B expensive; the weights are.

**Measured**, from vLLM's `GPU KV cache size` line on GB10:

| Model | util | KV tokens | Concurrency |
|---|---:|---:|---|
| `qwen3.5-122b-a10b` | 0.89 | 1.80M | **13.8×** at 128K |
| `qwen3.6-35b` | 0.35 | 1.20M | **4.6×** at 256K |

**Pooling models hold no KV.** An embedding or reranking model does one forward
pass per input, so `planner.kv_bytes` returns 0 and the charge is weights plus
activation.

## Runtime overhead

KV is whatever remains of `gpu_util × VRAM` after weights and runtime overhead.
Runtime overhead is ~10 GiB for a generate runner (`planner.ACTIVATION_BYTES`)
and 2 GiB for a pooling one (`POOLING_ACTIVATION_BYTES`), measured on GB10 by
subtracting weights and the reported KV cache from the budget:

| Model | util | budget | weights | KV reported | overhead |
|---|---:|---:|---:|---:|---:|
| `qwen3.6-35b` | 0.30 | 36.5 GiB | 21.4 GiB | 5.4 GiB (568K tokens) | **9.7 GiB** |

Activation buffers, CUDA-graph capture and the hybrid models' per-sequence conv
state. On a small card the planner scales this figure down
(`ACTIVATION_MAX_FRACTION`).

If the utilisation figures of co-resident containers sum past 1.0, whichever
starts last gets only the remainder; on unified-memory nodes the OS and page
cache need their share too.

## The 122B

**A whole-node decision.** 78 GiB of weights plus ~10 GiB of runtime leaves
~22 GiB of KV on a GB10, about 13 concurrent requests at 128K. A second model
on that card costs concurrency directly, so the planner seats it alone and
`VLLM_PREFERRED_MODELS` leaves it out of the recommended download set.

**On a 96 GB card it does not fit whole.** 77.8 GiB of weights plus ~10 GiB of
runtime is 87.8 GiB against 89.4 GiB of card, and `--gpu-memory-utilization`
tops out at 0.95. Shrinking the context does not help. Two cards and
`tensor_parallel: 2` is the answer: 39 GiB of weights per rank leaves room for
the native 256K.

| Host | TP | 128K | 256K | 256K, 24 sessions |
|---|---:|---|---|---|
| 1 × PRO 6000 | 1 | ✗ (108 GiB/card) | ✗ | ✗ |
| 2 × PRO 6000 | 2 | ✓ util 0.66 | ✓ util 0.77 | ✗ (88.5 GiB/card) |
| 4 × PRO 6000 | 4 | ✓ util 0.45 | ✓ util 0.56 | ✓ util 0.78 |

TP 4 does not quarter the KV cache: it shards by head and this model has two,
so per-card KV is the same as at TP 2. The extra cards buy weights.

## Per node class

| Node | VRAM | `qwen3.6-35b` | `qwen3.5-122b-a10b` | Notes |
|---|---:|---|---|---|
| RTX 4090 | 24 G | ✗ | ✗ | Below the 32 GiB floor. No FP4; the int4 build does not fit |
| RTX 5090 | 32 G | ○ | ✗ | 35B at a reduced context |
| PRO 5000 | 48 G | ○ | ✗ | 35B with KV headroom, plus retrieval and transcription |
| PRO 6000 | 96 G | ○ | TP 2 only | `qwen3-coder-next` fits alone |
| GB10 | 128 G (unified) | ○ | ○ | 121.6 GiB in practice; the planner reserves **12 GiB** before distributing |

- GB10 is unified memory, so `nvidia-smi` reports free VRAM as `[N/A]`. The
  planner takes the total from `/proc/meminfo` and subtracts 12 GiB for the OS
  (`scheduler/inventory.py::_UNIFIED_RESERVE_BYTES` and
  `lib.sh::UNIFIED_RESERVE_GB`; the two must agree).
- A node with more than one card is packed per card. `gpu_util` is a fraction
  of one device, so the scheduler assigns device ordinals and writes them as
  `{env_prefix}_DEVICES`, which compose passes as `NVIDIA_VISIBLE_DEVICES`.
  Not `CUDA_VISIBLE_DEVICES`: that one has no value meaning "every card", and
  on GB10 setting it fails engine init with `cudaErrorNotPermitted`. A node
  the scheduler has not pinned gets `all`.
- `gpu_util` is a fraction of the total, and vLLM needs that much to be free.
  Figures summing below 1.0 are not sufficient on their own: an earlier
  container plus page cache can still leave too little, which is what the
  12 GiB reservation protects.
- `qwen3.5-122b-a10b` and `qwen3-coder-next` each need a card to themselves.
  They are pool models for that reason: the head node holds default chat and
  retrieval, and a card-sized model cannot go where they are. See
  [models.md](models.md#where-models-are-defined).

## Tuning knobs

Normally nobody sets these. The placement step of `setup.sh all`
(`python3 -m scheduler apply`) measures node capacity, computes `MAX_LEN` and
`GPU_UTIL` per model, and writes them into that node's `.env` (see
[scheduler](../scheduler/README.md)). The values below are the `.env.example`
defaults used when placement is skipped (`KLOUDCHAT_SKIP_SCHEDULER=1`).

| Variable | Default | Rationale |
|---|---|---|
| `VLLM_QWEN35B_GPU_UTIL` | `0.55` | 21.4 GiB of weights plus KV, mm-budget and cudagraph profiling |
| `VLLM_QWEN35B_MAX_LEN` | `262144` | Native 262K |
| `VLLM_QWEN35B_MAX_BATCHED_TOKENS` | `16384` | Lower bound for the vision mm-budget |
| `VLLM_QWEN35B_MAX_NUM_SEQS` | `128` | The hybrid Gated-DeltaNet's per-sequence conv-state cache limits CUDA-graph capture; unset, cudagraph profiling OOMs |
| `VLLM_QWEN122B_GPU_UTIL` | `0.85` | 77.8 GiB of weights plus ~10 GiB of runtime; the rest is the KV pool |
| `VLLM_QWEN122B_MAX_LEN` | `131072` | 128K, not the native 262K. Depth or concurrency; see the KV table |
| `VLLM_QWEN122B_MAX_NUM_SEQS` | `32` | 36 linear-attention layers at 21.4 MiB of conv state per sequence: 2.7 GiB at 128, out of a ~22 GiB pool |
| `VLLM_QWEN122B_TP` | `1` | Cards to shard across. 2 on a PRO 6000 host |
| `VLLM_CODERNEXT_GPU_UTIL` / `_MAX_LEN` | `0.85` / `262144` | 75 GiB of weights; 12 KiB/token makes the native context affordable |
| `VLLM_CODER30B_GPU_UTIL` / `_MAX_LEN` | `0.60` / `131072` | 48 KiB/token, so the context is capped below native |
| `VLLM_QWEN27B_GPU_UTIL` / `_MAX_LEN` | `0.45` / `131072` | 32 KiB/token |

**Quantisation on cards without FP4.** `gpu_supports_quant` gates by compute
capability: NVFP4 needs 10.0, FP8 needs 8.9, AWQ/GPTQ int4 reach back to 7.5.
The catalogue carries an int4 build of the chat model (`qwen3.6-35b-awq`) for
that case. It is a download alias, not a separate deployment: point
`VLLM_QWEN35B_DIR` at it and the served entry is unchanged.

- `*_MAX_LEN` on the node is what vLLM serves. `gen-litellm-config.sh`
  discovers `max_model_len` from each node's `/v1/models` and emits it per
  deployment; only when discovery fails does `CTX_FALLBACK` (32768) apply.
- `manage-vllm.sh up` without a service name starts every vLLM whose weights
  are on the node. On a shared card that sums the utilisation fractions past
  1.0 and OOMs whichever starts last. When driving a node by hand, name the
  service.
- Parsers, `--kv-cache-dtype fp8` and the absence of `--enforce-eager` are
  set in `docker-compose.vllm.yml`. Healthcheck `start_period` is 600 s for
  the chat services, 1200 s for `vllm-qwen122b` (78 GiB to read before
  torch.compile), 120 s for pooling and transcription; `unhealthy` during
  that window is normal.

## Transcription (STT)

`openai/whisper-large-v3` served by vLLM as `vllm-whisper`, on any
architecture, placed and sized like every other model. vLLM rather than
faster-whisper because ctranslate2's aarch64 wheels are CPU-only, which would
leave a GB10 unable to use its card for it.

| | |
|---|---|
| Weights | **3.1 GiB** measured (the repository ships 24.7 GB across four serialisations; vLLM reads one) |
| Context | 448, the decoder's window. Longer clips are split by the server before decoding |
| Placement | `placement: head`, priority 4, below `bge-m3`: losing embeddings turns retrieval lexical, losing this one still transcribes through OpenRouter |
| Charged | ~13 GiB, the generate-runner headroom |

`WHISPER_URLS` is written from the placement. Empty (no room, or no GPU node)
sends STT to OpenRouter (`voxtral-small-24b`).

## Where the numbers come from

- **Weights are measured, not declared.** The scheduler reads the checkpoint
  directory size on the node over SSH (`model_metadata.measure_weight_bytes`).
  `models.yaml` carries a `weight:` override only when that measurement is
  wrong.
- **KV cache size is printed by vLLM at startup** (`GPU KV cache size: N
  tokens`, `Maximum concurrency for M tokens per request`). Where the log and
  the planner disagree, the log is right.
- **Throughput**, GB10, batch size 1, warm, median of three 400-token
  generations: `qwen3.6-35b` 47–48 tok/s. Utilisation sets KV capacity, not
  speed. Measure on an idle node; a competing request halves the figure.
