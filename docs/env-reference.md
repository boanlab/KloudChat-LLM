# Environment variables

`.env` is produced by `./scripts/gen-env.sh` from `.env.example`. Anything
written as `change-me-*` is replaced with a generated secret at that point, so
the only values a human fills in are external keys and node addresses.

```bash
./scripts/gen-env.sh          # create (skipped if .env exists)
./scripts/gen-env.sh --force  # recreate — every existing secret changes
```

## 1. Filled in by a human

| Variable | Value | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | `sk-or-v1-...` | Commercial models and local fallback. Required without a GPU |
| `HF_TOKEN` | token | Hugging Face gated repositories. Used only for weight downloads |
| `NODES_VLLM` | `user@host,...` | GPU node SSH targets. Empty means no local models. This is the only node list. **Order matters**: the first target is the head node, which holds default chat, retrieval and transcription; the rest are the pool, which holds the large picker models ([models.md](models.md#where-models-are-defined)) |
| `VLLM_MODELS` | `id,id` | Models to deploy. Defined in `scheduler/models.yaml` |

`setup.sh` refuses to continue unless at least one of `OPENROUTER_API_KEY` or a
vLLM node is present.

## 2. Exposure

| Variable | Default | Notes |
|---|---|---|
| `GATEWAY_PORT` | `8080` | The only published port |
| `COMPOSE_PROFILES` | `tools,models` | What to run. `setup.sh` appends `whisper` (the transcription shim) once the transcription model is placed. Add `index` for the retrieval index |
| `INDEX_DB_USER` | `index` | Owner role of the retrieval index's pgvector database. Its password is a generated secret — see below |
| `INDEX_EMBED_MODELS` | `local/bge-m3,text-embedding-3-small` | Embedding preference order, tried through LiteLLM by name |

## 3. Generated secrets

Filled in by `gen-env.sh`. If you create them yourself, keep the formats below.

| Variable | Format | Used by |
|---|---|---|
| `LITELLM_MASTER_KEY` | `sk-` + 64 hex | LiteLLM admin API. Also entered in the UI admin screen |
| `LITELLM_DB_PASSWORD` | 32 hex | LiteLLM's postgres |
| `LITELLM_DB_USER` | string | `kloudchat-litellm` by default |
| `SEARXNG_SECRET_KEY` | 32 hex | SearXNG session signing |
| `CODE_INTERPRETER_API_KEY` | 32 hex | Injected by the gateway into code execution requests |
| `CODE_INTERPRETER_MINIO_PASSWORD` | 32 hex | Artifact storage |
| `INDEX_DB_PASSWORD` | 32 hex | The retrieval index's pgvector database. Required whenever the `index` profile is on |
| `SCRAPER_API_KEY` | 32 hex | Injected by the gateway into document fetch requests, checked by the shim |

The two injected keys **never reach the UI**: they exist only inside the gateway.

## 4. Placement results — written by the scheduler

Do not set these by hand. The placement step of `setup.sh all` writes them.

| Variable | Contents |
|---|---|
| `VLLM_<MODEL>_URL` | CSV of node addresses serving that model |
| `VLLM_<MODEL>_MAX_LEN` | Context decided for it |
| `VLLM_<MODEL>_GPU_UTIL` | `--gpu-memory-utilization` decided for it |
| `WHISPER_URLS` | CSV of the nodes the transcription model was placed on, written by the placement step. STT goes to OpenRouter when none of them answers — placed is not the same as serving |

The prefix (`VLLM_QWEN35B` and so on) is the `env_prefix` in
`scheduler/models.yaml`.

To manage them yourself, set `KLOUDCHAT_SKIP_SCHEDULER=1` and fill in the values.

## 5. Images

| Variable | Default | Notes |
|---|---|---|
| `KLOUDCHAT_IMAGE_NS` | `boanlab` | Images are pulled and pushed as `<NS>/kloudchat-*` |
| `KLOUDCHAT_IMAGE_TAG` | `latest` | |

To publish to a different registry, change the namespace and use
`./scripts/build-push-images.sh` (or the `Publish images` workflow).

## 6. GPU node startup options

| Variable | Default | Notes |
|---|---|---|
| `VLLM_IMAGE` | (empty) | Empty uses the compose default, arm64 (GB10). amd64 nodes must name the standard image — `install-vllm.sh` records it |
| `VLLM_MODELS_ROOT` | `/var/lib/vllm/models` | Checkpoint root on the node |
| `VLLM_<MODEL>_DIR` | model id | Checkpoint directory name |
| `VLLM_<MODEL>_MAX_BATCHED_TOKENS` | `16384` | 16384 or more is required for the vision mm-budget |
| `VLLM_<MODEL>_MAX_NUM_SEQS` | `128` | CUDA-graph capture limit for hybrid models |
| `VLLM_GLMFLASH_ATTN_BACKEND` | `FLASHINFER_MLA` | Avoids the GB10 shared-memory limit |
| `VLLM_CODERNEXT_DEEP_GEMM` | `0` | DeepGEMM rejects this checkpoint's FP8 scale-factor layout on GB10. `1` where the kernel takes it |
| `VLLM_BASE_IMAGE` / `VLLM_BASE_DIGEST` | (empty) | Upstream image and the digest it resolved to, recorded by `install-vllm.sh`. Pin rebuilds to the digest |

Where the defaults come from, and how they relate to the placement step, is in
[GPU memory](gpu-memory.md#tuning-knobs).

## 7. LiteLLM behaviour

| Variable | Default | Notes |
|---|---|---|
| `LITELLM_NUM_WORKERS` | `4` | ~600 MB per worker |
| `LITELLM_LOG` | `INFO` | |
| `KC_OR_VARIANT` | `:floor` | OpenRouter provider routing variant. `:nitro` for throughput, empty for the OpenRouter default |
| `CONCURRENCY_GATE_CAPS` | built-in per-model caps | JSON map of model aliases to positive concurrency caps; nonpositive values cannot disable strict-local protection |
| `CONCURRENCY_GATE_TTL` | `1.5` | Seconds between vLLM capacity polls |
| `CONCURRENCY_GATE_SCRAPE_TIMEOUT` | `1.0` | Timeout in seconds for one vLLM metrics request |
| `CONCURRENCY_GATE_DEBUG` | (empty) | Log the overload gate's decisions |
| `CONCURRENCY_GATE_FORCE` | (empty) | Comma-separated model aliases to force through the saturated path for testing |

## 8. Deep research

| Variable | Default | Notes |
|---|---|---|
| `DEEP_RESEARCH_MODEL` | `local/qwen3.5-122b-a10b` | Model used for iterative search. Needs the 128K floor the scheduler reserves on it, and takes a pool node's KV pool for the length of a run |
| `DEEP_RESEARCH_LLM_URL` | `http://litellm:8000/v1` | LiteLLM on the same network |

## 9. Transcription

`openai/whisper-large-v3` on vLLM, placed like any other model — on any
architecture, and with `MAX_LEN` and `GPU_UTIL` written by the placement step.
Add `whisper-large-v3` to `VLLM_MODELS` to deploy it.

| Variable | Default | Notes |
|---|---|---|
| `VLLM_WHISPER_DIR` | `whisper-large-v3` | Checkpoint directory under `VLLM_MODELS_ROOT` |
| `WHISPER_MAX_UPLOAD_MB` | `100` | Upload ceiling. vLLM defaults to 25, which rejects an hour of m4a |
| `TRANSCRIBE_TIMEOUT_SEC` | `3600` | Headroom for long meeting recordings |
| `WHISPER_MODEL_NAME` | `local/whisper-large-v3` | The name the shim puts on every forwarded request. Must be one of `vllm-whisper`'s `--served-model-name` values |

## Shell-only variables

Passed on the command line rather than through `.env`.

| Variable | Effect |
|---|---|
| `KLOUDCHAT_SKIP_SCHEDULER=1` | Skip the placement step in `setup.sh all` |
| `KLOUDCHAT_REMOTE_DIR` | Repository path on remote nodes (default `KloudChat-LLM`). rsync and the placement step both use it |
| `KLOUDCHAT_VLLM_WAIT_TIMEOUT` | Deadline for waiting on vLLM readiness (default 1200s) |
| `KLOUDCHAT_VLLM_WAIT_INTERVAL` | Probe interval while waiting (default 10s) |
| `KLOUDCHAT_SERVICE_WAIT` | How long to wait for capabilities to answer after startup (default 180s) |
| `KLOUDCHAT_SCHEDULER_NO_AUTOINSTALL=1` | Do not auto-install PyYAML (when using a virtualenv) |
| `YES=1` | Skip the `clean` confirmation |
