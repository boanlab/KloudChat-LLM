# GPU memory guide

What fits on which GPU, and how much VRAM each model occupies. Read this when
sizing a node or diagnosing an OOM.

> Figures cover every model in `scheduler/models.yaml`, deployed or not. There
> is no local media backend: images, audio and video pass through to
> OpenRouter.

## The lineup

| Model | Quant | Weights | Role |
|---|---|---:|---|
| `qwen3.8-27b` (Qwen3.8-27B) | NVFP4 | **21.3 GiB** (measured) | Chat and floor. Dense, vision, 262K context, agentic coding |
| `qwen3-coder-next` (Qwen3-Coder-Next-80B-A3B) | FP8 | **~75 GiB** (on disk) | Coding. Hybrid attention, 262K. Wants the card to itself |
| `bge-m3` | BF16 | **~2 GiB** | Retrieval embeddings. Pooling, shares a card |
| `bge-reranker-v2-m3` | BF16 | **2.1 GiB** (measured) | Retrieval reranking. Pooling, shares a card |
| `whisper-large-v3` | FP16 | **3.1 GiB** (measured) | Transcription |

- The default lineup is NVFP4 only, which needs compute capability 10.0. A card
  without FP4 runs `qwen3.8-27b-awq` instead (same served model, different
  checkpoint directory); `gpu_supports_quant` decides by capability.
- Weight figures are measured `safetensors` totals. The `-NVFP4` checkpoints
  mix 4-bit and 8-bit groups, which puts them 5–6 GiB above a pure-FP4
  calculation. Do not re-derive them from parameter counts.
- The direction of an error matters: overestimating makes the planner discard
  low-utilisation configurations and delegate; underestimating produces an OOM
  at startup.

## KV cost

Only 16 of the 64 layers in `qwen3.8-27b` carry KV (Gated-DeltaNet to
full-attention is 3:1). The other 48 are linear attention with fixed-size
state.

| Model | KV per token (fp8) | 128K | 256K |
|---|---:|---:|---:|
| `qwen3.8-27b` | **32 KiB** | 4.0 GiB | 8.0 GiB |
| `qwen3-coder-next` (hybrid) | **12 KiB** | 1.5 GiB | 3.0 GiB |

`qwen3-coder-next` is 3× the 27B in parameters at a third of its KV: 12 of 48
layers at 2 KV heads against 16 of 64 at 4 KV heads, both 256 head dim.
Weights are what makes the coder expensive; context is what makes the 27B
expensive.

**Measured**, from vLLM's `GPU KV cache size` line on GB10:

| Model | util | KV tokens | Concurrency |
|---|---:|---:|---|
| `qwen3.8-27b` | 0.56 | 0.99M | **3.8×** at 256K (MTP 5; 1.22M / 4.7× without) |

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
| `qwen3.8-27b` | 0.56 | 68.1 GiB | 21.3 GiB | 30.3 GiB (993K tokens) | **16.5 GiB** (MTP head and draft KV included; 9.7 GiB without) |

Activation buffers, CUDA-graph capture and the hybrid models' per-sequence conv
state. On a small card the planner scales this figure down
(`ACTIVATION_MAX_FRACTION`).

If the utilisation figures of co-resident containers sum past 1.0, whichever
starts last gets only the remainder; on unified-memory nodes the OS and page
cache need their share too.

## Per node class

| Node | VRAM | `qwen3.8-27b` | `qwen3-coder-next` | Notes |
|---|---:|---|---|---|
| RTX 4090 | 24 G | ✗ | ✗ | Below the 32 GiB floor. No FP4; the int4 build does not fit |
| RTX 5090 | 32 G | ○ | ✗ | 27B at a reduced context |
| PRO 5000 | 48 G | ○ | ✗ | 27B with KV headroom, plus retrieval and transcription |
| PRO 6000 | 96 G | ○ | ○ alone | |
| GB10 | 128 G (unified) | ○ | ○ alone | 121.6 GiB in practice; the planner reserves **12 GiB** before distributing |

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
- `qwen3-coder-next` needs a card to itself. It is a pool model for that
  reason: the head node holds retrieval and transcription, and a card-sized
  model cannot go where they are. See
  [models.md](models.md#where-models-are-defined).

## Tuning knobs

Normally nobody sets these. The placement step of `setup.sh all`
(`python3 -m scheduler apply`) measures node capacity, computes `MAX_LEN` and
`GPU_UTIL` per model, and writes them into that node's `.env` (see
[scheduler](../scheduler/README.md)). The values below are the `.env.example`
defaults used when placement is skipped (`KLOUDCHAT_SKIP_SCHEDULER=1`).

| Variable | Default | Rationale |
|---|---|---|
| `VLLM_QWEN27B_GPU_UTIL` | `0.55` | 21.3 GiB of weights plus KV, mm-budget and cudagraph profiling |
| `VLLM_QWEN27B_MAX_LEN` | `262144` | Native 262K |
| `VLLM_QWEN27B_MAX_BATCHED_TOKENS` | `16384` | Lower bound for the vision mm-budget |
| `VLLM_QWEN27B_SPEC_TOKENS` | `5` | MTP draft tokens per step. A dense 27B decodes at ~11 tok/s on GB10 without speculation; the draft head and its KV take ~19% of the KV pool |
| `VLLM_QWEN27B_MAX_NUM_SEQS` | `64` | The hybrid Gated-DeltaNet's per-sequence conv-state cache limits CUDA-graph capture; unset, cudagraph profiling OOMs |
| `VLLM_CODERNEXT_GPU_UTIL` / `_MAX_LEN` | `0.85` / `262144` | 75 GiB of weights; 12 KiB/token makes the native context affordable |

**Quantisation on cards without FP4.** `gpu_supports_quant` gates by compute
capability: NVFP4 needs 10.0, FP8 needs 8.9, AWQ/GPTQ int4 reach back to 7.5.
The catalogue carries an int4 build of the chat model (`qwen3.8-27b-awq`) for
that case. It is a download alias, not a separate deployment: point
`VLLM_QWEN27B_DIR` at it and the served entry is unchanged.

- `*_MAX_LEN` on the node is what vLLM serves. `gen-litellm-config.sh`
  discovers `max_model_len` from each node's `/v1/models` and emits it per
  deployment; only when discovery fails does `CTX_FALLBACK` (32768) apply.
- `manage-vllm.sh up` without a service name starts every vLLM whose weights
  are on the node. On a shared card that sums the utilisation fractions past
  1.0 and OOMs whichever starts last. When driving a node by hand, name the
  service.
- Parsers, `--kv-cache-dtype fp8` and the absence of `--enforce-eager` are
  set in `docker-compose.vllm.yml`. Healthcheck `start_period` is 600 s for
  the chat services, 1200 s for `vllm-codernext` (75 GiB to read before
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
  generations: `qwen3.8-27b` 15 tok/s on prose and 24 tok/s on code with MTP
  (`SPEC_TOKENS` 5), 11 tok/s without. One instance batches: 300-token
  prose at 1 / 4 / 8 / 16 concurrent requests gives 12 / 45 / 81 / 141 tok/s
  aggregate, so a second instance on the same card adds nothing but a second
  copy of the weights. Utilisation sets KV capacity, not speed. Measure on an idle node; a competing request halves the figure.
