# GPU memory guide

What fits on which GPU, and how much VRAM each model actually occupies. Read this
when sizing a node or diagnosing an OOM.

> This document describes the **current lineup** (`qwen3.6-35b` and
> `glm-4.7-flash`). There is no local media backend and no embedding deployment:
> images, audio and video pass through to OpenRouter, and nothing in the app
> calls `/embeddings`.

## The lineup

| Model | Quant | Weights | Role |
|---|---|---:|---|
| `qwen3.6-35b` (Qwen3.6-35B-A3B) | NVFP4 | **21.4 GiB** (measured) | Chat, vision, 262K context, agentic coding |
| `glm-4.7-flash` (31.2B-A3B) | NVFP4 | **19.8 GiB** (measured) | Cheap-decode floor |

- **NVFP4 only.** A card without FP4 support (RTX 4090) cannot host this lineup,
  and there is no int4 build of 35B-A3B to substitute.
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
| `qwen3.6-35b` | **10 KiB** | 1.2 GiB | 2.5 GiB |
| `glm-4.7-flash` (MLA) | 26.4 KiB | 3.2 GiB | — (128K limit) |

- **The floor model's KV is the more expensive of the two.** MLA
  (`kv_lora_rank` 512 plus 64 rope) holds it to 26.4 KiB per token across 47
  full-attention layers, but that is still 2.6× the chat model. The floor model
  earns its place through **decode cost**, not KV savings.
- Applying the MHA formula to an MLA model yields 187 KiB per token, a 7×
  overestimate. That trap makes the planner treat this model as the most
  KV-expensive thing in the cluster.

**Measured (GB10, utilisation 0.60)** — a 42.85 GiB KV cache holds **4.07M
tokens**, about **31× concurrency** at 128K per request.

## VRAM by scenario

| Configuration | Weights | Recommended card |
|---|---:|---|
| Chat alone | ~21 GiB | RTX 5090 32 GB or better (NVFP4 required) |
| Chat + floor | ~41 GiB | PRO 5000 48 GB or better |
| Chat + floor with KV headroom | ~41 GiB + KV | PRO 6000 96 GB or GB10 |
| Add embeddings (`bge-m3`) | +~2 GiB | any of the above |

**Pooling models hold no KV.** An embedding model does one forward pass per
input and keeps nothing between tokens, so `planner.kv_bytes` returns 0 for it
and the charge is weights plus activation. Sizing it as a chat model would
reserve tens of gigabytes it never touches.

Weights are only the starting point. KV is whatever remains of
`gpu_util × VRAM` after weights and activation (~4 GiB). If the utilisation
figures sum past 1.0, whichever container starts last gets only the remainder —
and on unified-memory nodes the OS and page cache need their share too.

## Per node class

| Node | VRAM | Chat | Floor | Notes |
|---|---:|---|---|---|
| RTX 4090 | 24 G | ✗ | ✗ | No FP4 — this lineup cannot run |
| RTX 5090 | 32 G | ○ | △ | One or the other; both leaves almost no KV |
| PRO 5000 | 48 G | ○ | ○ | ~41 GiB loaded, little KV headroom |
| PRO 6000 | 96 G | ○ | ○ | ~50 GiB of KV after loading both |
| GB10 | 128 G (unified) | ○ | ○ | 121.63 GiB in practice; the planner reserves **12 GiB** before distributing |

- **GB10 is unified memory**, so `nvidia-smi` reports free VRAM as `[N/A]`. The
  planner takes the total from `/proc/meminfo` and subtracts **12 GiB** for the
  OS (`scheduler/inventory.py::_UNIFIED_RESERVE_BYTES` and
  `lib.sh::UNIFIED_RESERVE_GB` — these two must move together).
- **`gpu_util` is a fraction of the total, and vLLM needs that much to be
  *free*.** Utilisation figures summing below 1.0 are not sufficient: an earlier
  container plus page cache can still leave too little, which is what the 12 GiB
  reservation protects.
- **No model here requires a node to itself.** Both use a fraction of one card.

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
| `VLLM_GLMFLASH_ATTN_BACKEND` | `FLASHINFER_MLA` | The default TRITON_MLA decode kernel exceeds GB10's shared-memory limit (101376 B) by 1 KB, failing engine init. Depending on the card, `CUTLASS_MLA`, `FLASHMLA` or `FLASH_ATTN_MLA` |

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
  `docker-compose.vllm.yml`. Both services use a healthcheck `start_period` of
  600s, and `unhealthy` during that window is normal.

## Transcription (STT)

Transcription is **amd64 only**: aarch64 ctranslate2 wheels are CPU-only, so GB10
cannot use its card for it. arm64 nodes keep no backend, which leaves
`WHISPER_URLS` empty, which is what sends STT to OpenRouter
(`voxtral-small-24b`).

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
