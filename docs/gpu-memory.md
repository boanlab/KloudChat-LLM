# GPU memory guide

What fits on which GPU, and how much VRAM each model actually occupies. Read this
when sizing a node or diagnosing an OOM.

> This document describes the **current lineup** (`qwen3.5-122b-a10b`,
> `qwen3.6-35b` and, defined but not deployed here, `glm-4.7-flash`). There is no
> local media backend: images, audio and video pass through to OpenRouter.

## The lineup

| Model | Quant | Weights | Role |
|---|---|---:|---|
| `qwen3.5-122b-a10b` (Qwen3.5-122B-A10B) | NVFP4 | **77.8 GiB** (measured) | Top chat — vision, 128K here, 10B active |
| `gemma-4-26b-a4b` (Gemma-4-26B-A4B) | NVFP4 | **15.3 GiB** (measured) | Second family — vision, tools, 256K, 4B active |
| `qwen3-coder-30b` (Qwen3-Coder-30B-A3B) | FP8 | **33.0 GiB** (measured) | Coding. FP8, so it runs without FP4 support |
| `qwen3.6-27b` (Qwen3.6-27B) | NVFP4 | **20.4 GiB** (measured) | The one dense model |
| `qwen3.6-35b` (Qwen3.6-35B-A3B) | NVFP4 | **21.4 GiB** (measured) | Chat and floor — vision, 262K context, agentic coding |
| `glm-4.7-flash` (31.2B-A3B) | NVFP4 | **19.8 GiB** (measured) | Cheap-decode floor. Defined, not deployed: `qwen3.5-122b-a10b` took its card |

- **The default lineup is NVFP4 only**, which needs compute capability 10.0. A
  card without FP4 runs the AWQ int4 aliases instead — same served models,
  different checkpoint directory — and `gpu_supports_quant` decides which by
  capability rather than by card name.
- Both weight figures are **measured** `safetensors` totals. Despite the `-NVFP4`
  name, compressed-tensors `config_groups` mixes 4-bit and 8-bit, which puts them
  5–6 GiB above a pure-FP4 calculation. **Do not re-derive them from parameter
  counts.**
- The direction of an error matters: overestimating makes the planner discard
  low-utilisation configurations as `residual ≤ 0` and silently delegate, while
  underestimating produces an OOM at startup.

## KV cost — hybrid attention

Only **10 of the 40 layers** in `qwen3.6-35b` carry KV (Gated-DeltaNet to
full-attention is 3:1). The other 30 are linear attention and keep fixed-size
state regardless of sequence length.

| Model | KV per token (fp8) | 128K | 256K |
|---|---:|---:|---:|
| `qwen3.5-122b-a10b` | **12 KiB** | 1.5 GiB | — (128K here) |
| `gemma-4-26b-a4b` | 20 KiB + 100 MiB/session | 2.5 GiB | 5.0 GiB |
| `qwen3-coder-30b` | **48 KiB** | 6.0 GiB | 12.0 GiB |
| `qwen3.6-27b` | 32 KiB | 4.0 GiB | 8.0 GiB |
| `qwen3.6-35b` | **10 KiB** | 1.2 GiB | 2.5 GiB |
| `glm-4.7-flash` (MLA) | 26.4 KiB | 3.2 GiB | — (128K limit) |

- **Gemma's second term is not a typo.** Five of its six layers are
  sliding-window (1024 tokens), so they hold a bounded 100 MiB per *sequence*
  whatever the context, while only the five full-attention layers grow with it.
  Charging the sliding layers as full attention would put this model at 120 KiB
  per token and price it off every card; charging them as nothing loses half a
  gigabyte at four sessions. `kv_model.sliding_bytes_per_sequence` carries the
  flat term.
- **Scaling the model 3.5x barely moved its KV.** 48 layers against 40, of which
  12 carry KV against 10, at the same 2 KV heads and 256 head dim. Context length
  is not what makes the 122B expensive — **the weights are**.

- **The floor model's KV is the more expensive of the two.** MLA
  (`kv_lora_rank` 512 plus 64 rope) holds it to 26.4 KiB per token across 47
  full-attention layers, but that is still 2.6× the chat model. The floor model
  earns its place through **decode cost**, not KV savings.
- Applying the MHA formula to an MLA model yields 187 KiB per token, a 7×
  overestimate. That trap makes the planner treat this model as the most
  KV-expensive thing in the cluster.

**Measured on this cluster**, from vLLM's own `GPU KV cache size` line:

| Model | util | KV tokens | Concurrency |
|---|---:|---:|---|
| `qwen3.5-122b-a10b` | 0.89 | 1.80M | **13.8×** at 128K |
| `qwen3.6-35b` | 0.35 | 1.20M | **4.6×** at 256K |

The 35B held 568K tokens (2.17× at 256K) before the activation figure was
corrected below — same weights, same card, same context.

## VRAM by scenario

| Configuration | Weights | Recommended card |
|---|---:|---|
| Chat alone (35B) | ~21 GiB | RTX 5090 32 GB or better (NVFP4 required) |
| Chat + floor | ~41 GiB | PRO 5000 48 GB or better |
| Chat + floor with KV headroom | ~41 GiB + KV | PRO 6000 96 GB or GB10 |
| Add embeddings (`bge-m3`) | +~2 GiB | any of the above |
| `qwen3.5-122b-a10b` | ~78 GiB | GB10 or PRO 6000, **and nothing else on the card** |

**The 122B is a whole-node decision.** 78 GiB of weights plus ~10 GiB of runtime
leaves ~22 GiB of KV on a GB10 — about 13 concurrent requests at 128K. Adding a
second model to that card costs concurrency directly, which is why the planner
seats it alone and `VLLM_PREFERRED_MODELS` leaves it out of the recommended set.

**On a 96 GB card it does not fit at all.** 77.8 GiB of weights plus ~10 GiB of
runtime is 87.8, against 89.4 GiB of card — `--gpu-memory-utilization` tops out
at 0.95, so the KV cache would be negative. Shrinking the context does not help:
the weights are what fills the card. Two cards and `tensor_parallel: 2` is the
answer, and a good one — 39 GiB of weights per rank leaves room for the native
256K context:

| Host | TP | 128K | 256K | 256K, 24 sessions |
|---|---:|---|---|---|
| 1 × PRO 6000 | 1 | ✗ (108 GiB/card) | ✗ | ✗ |
| 2 × PRO 6000 | 2 | ✓ util 0.66 | ✓ util 0.77 | ✗ (88.5 GiB/card) |
| 4 × PRO 6000 | 4 | ✓ util 0.45 | ✓ util 0.56 | ✓ util 0.78 |

TP 4 does not quarter the KV cache — it shards by head and this model has two, so
the per-card KV is the same as at TP 2. What the extra cards buy is weights.

| `bge-reranker-v2-m3` | BF16 | **2.1 GiB** (measured) | Retrieval reranking — pooling, shares a card |

**Pooling models hold no KV.** An embedding model does one forward pass per
input and keeps nothing between tokens, so `planner.kv_bytes` returns 0 for it
and the charge is weights plus activation. Sizing it as a chat model would
reserve tens of gigabytes it never touches.

Weights are only the starting point. KV is whatever remains of
`gpu_util × VRAM` after weights and runtime overhead. If the utilisation figures
sum past 1.0, whichever container starts last gets only the remainder — and on
unified-memory nodes the OS and page cache need their share too.

**Runtime overhead is ~10 GiB, not ~4.** Measured on GB10 by subtracting weights
and the KV cache vLLM reports at startup from the budget it was given:

| Model | util | budget | weights | KV reported | ⇒ overhead |
|---|---:|---:|---:|---:|---:|
| `qwen3.6-35b` | 0.30 | 36.5 GiB | 21.4 GiB | 5.4 GiB (568K tokens) | **9.7 GiB** |
| `glm-4.7-flash` | 0.39 | 47.4 GiB | 18.6 GiB | 20.4 GiB (810K tokens) | **8.5 GiB** |

Activation buffers, CUDA-graph capture and the hybrid models' per-sequence conv
state. `planner.ACTIVATION_BYTES` carried 4 GiB, and the difference came out of
the KV cache: a placement sized for four concurrent sessions delivered 2.17.
Harmless while weights are small and the node has slack; at 78 GiB of weights it
is most of the pool. A pooling runner captures no decode graphs and keeps no
per-sequence state, so it is charged `POOLING_ACTIVATION_BYTES` (2 GiB) instead.

## Per node class

| Node | VRAM | Chat | Floor | Notes |
|---|---:|---|---|---|
| RTX 4090 | 24 G | ✗ | ✗ | Below the 32 GiB floor. The int4 aliases execute here, but one places at its 32K floor and 0.92 of the card — see prerequisites |
| RTX 5090 | 32 G | ○ | △ | One or the other; both leaves almost no KV |
| PRO 5000 | 48 G | ○ | ○ | ~41 GiB loaded, little KV headroom |
| PRO 6000 | 96 G | ○ | ○ | ~50 GiB of KV after loading both |
| GB10 | 128 G (unified) | ○ | ○ | 121.63 GiB in practice; the planner reserves **12 GiB** before distributing |

`qwen3.5-122b-a10b` fits only the last two rows, and only alone.

- **GB10 is unified memory**, so `nvidia-smi` reports free VRAM as `[N/A]`. The
  planner takes the total from `/proc/meminfo` and subtracts **12 GiB** for the
  OS (`scheduler/inventory.py::_UNIFIED_RESERVE_BYTES` and
  `lib.sh::UNIFIED_RESERVE_GB` — these two must move together).
- **A node with more than one card is packed per card.** `gpu_util` is a fraction
  of one device, so the scheduler assigns device ordinals and writes them as
  `{env_prefix}_DEVICES`, which compose passes as `CUDA_VISIBLE_DEVICES`. Driving
  such a node by hand without setting it confines every model to card 0 — safe,
  but it uses a fraction of the box.
- **`gpu_util` is a fraction of the total, and vLLM needs that much to be
  *free*.** Utilisation figures summing below 1.0 are not sufficient: an earlier
  container plus page cache can still leave too little, which is what the 12 GiB
  reservation protects.
- **`qwen3.5-122b-a10b` requires a node to itself.** The other three use a
  fraction of one card and can share.

## Tuning knobs

**Normally nobody sets these.** The placement step of `setup.sh all`
(`python3 -m scheduler apply`) measures node capacity, computes `MAX_LEN` and
`GPU_UTIL` per model, and writes them into that node's `.env`
(see [scheduler](../scheduler/README.md)). The values below are the
`.env.example` defaults used when placement is skipped — single-node manual
operation, or `KLOUDCHAT_SKIP_SCHEDULER=1`.

| Variable | Default | Rationale |
|---|---|---|
| `VLLM_QWEN35B_GPU_UTIL` | `0.55` | 21.4 GiB of NVFP4 weights, plus KV, mm-budget and cudagraph profiling |
| `VLLM_QWEN35B_MAX_LEN` | `262144` | Native 262K |
| `VLLM_QWEN35B_MAX_BATCHED_TOKENS` | `16384` | Lower bound for the vision mm-budget |
| `VLLM_QWEN35B_MAX_NUM_SEQS` | `128` | The hybrid Gated-DeltaNet's per-sequence conv-state cache limits CUDA-graph capture; unset, cudagraph profiling OOMs |
| `VLLM_GLMFLASH_GPU_UTIL` | `0.65` | 19.8 GiB of NVFP4 weights plus the MLA KV pool |
| `VLLM_GLMFLASH_MAX_LEN` | `131072` | 128K limit |
| `VLLM_GLMFLASH_ATTN_BACKEND` | per card | `install-vllm.sh` writes it from `lib.sh::mla_attention_backend`: `FLASHINFER_MLA` on GB10 (TRITON_MLA's decode kernel is one byte over that card's 101376 B shared-memory limit), `CUTLASS_MLA` on Blackwell discrete, `TRITON_MLA` on Ada. An unrecognised card gets no value, which lets vLLM choose |

**Quantisation on cards without FP4.** `gpu_supports_quant` gates by compute
capability — NVFP4 needs 10.0, FP8 needs 8.9, AWQ/GPTQ int4 reach back to 7.5.
The default lineup is NVFP4, so an Ada or Ampere card can run none of it; the
catalogue carries int4 builds of three models for exactly that case
(`gemma-4-26b-a4b-awq`, `qwen3.6-27b-awq`, `qwen3.6-35b-awq`). They are download
aliases, not separate deployments: point the model's `*_DIR` at one and the
served entry is unchanged.
| `VLLM_QWEN122B_GPU_UTIL` | `0.85` | 77.8 GiB of weights plus ~10 GiB of runtime; the rest is the KV pool, and there is not much of it |
| `VLLM_QWEN122B_MAX_LEN` | `131072` | 128K, not the native 262K. Depth or concurrency — see the KV table above |
| `VLLM_QWEN122B_MAX_NUM_SEQS` | `32` | 36 linear-attention layers at 21.4 MiB of conv state per sequence: 2.7 GiB at 128, out of a ~22 GiB pool. Queue depth the card cannot serve anyway |

- **`*_MAX_LEN` is a fallback.** Normally `gen-litellm-config.sh` discovers
  `max_model_len` from each node's `/v1/models` and emits it per deployment, so
  differing contexts across nodes are reflected in routing. Only when discovery
  fails does `CTX_FALLBACK` (32768) apply.
- **Be careful with `manage-vllm.sh up` without a service name.** It starts every
  vLLM whose weights are on the node, which on a shared card sums the utilisation
  fractions past 1.0 and OOMs whichever starts last. Leave placement to the
  scheduler; when driving a node by hand, name the service.
- The remaining command options — parsers, `--kv-cache-dtype fp8`, why
  `--enforce-eager` is forbidden — are documented inline in
  `docker-compose.vllm.yml`. The chat services use a healthcheck `start_period`
  of 600s — 1200s for `vllm-qwen122b`, which has 78 GiB to read off disk before
  torch.compile starts — and `unhealthy` during that window is normal.

## Transcription (STT)

Transcription is **amd64 only**: aarch64 ctranslate2 wheels are CPU-only, so GB10
cannot use its card for it. arm64 nodes keep no backend, which leaves
`WHISPER_URLS` empty, which is what sends STT to OpenRouter
(`voxtral-small-24b`).

vLLM can serve `whisper-large-v3` itself, on any architecture, and that was tried
here. It works — but it is a **second** transcription mechanism, and the
production target is amd64, where the resident backend already runs. Carrying two
paths to buy local STT on the arm64 test bench alone is maintenance surface for
no production gain, so the delegation stands. The cost is that dictation on an
arm64 node leaves the network; on amd64 it does not.

On amd64 nodes it is a **resident cost rather than a placement decision**.
`install-vllm.sh` brings it up alongside vLLM (the `whisper` service in
`docker-compose.vllm.yml`), and when its `/health` answers, the planner subtracts
**6 GiB** before packing vLLM onto that node
(`scheduler/__main__.WHISPER_RESERVE_BYTES`).

Weights are ~3.2 GiB for large-v3 and the 6 GiB reservation covers runtime
headroom. That figure is an estimate, not a measurement.

## Where the numbers come from

- **Weights are measured, not declared.** The scheduler reads the checkpoint
  directory size on the node over SSH (`model_metadata.measure_weight_bytes`).
  `models.yaml` carries a `weight:` override only when that measurement is wrong.
- **KV cache size is printed by vLLM at startup** (`GPU KV cache size: N tokens`,
  `Maximum concurrency for M tokens per request`). Where the log and the planner
  disagree, the log is right.
- **Throughput**, GB10, batch size 1, warm, median of three 400-token
  generations: `qwen3.6-35b` 47–48 tok/s, `glm-4.7-flash` 56 tok/s. Both are ~3B
  active MoE, so decode cost is similar and utilisation barely moves it —
  utilisation sets KV capacity, not speed. Measure on an idle node: a competing
  request halves the figure.
