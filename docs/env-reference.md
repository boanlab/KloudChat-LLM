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
| `NODES_VLLM` | `user@host,...` | GPU node SSH targets. Empty means no local models. Each node runs vLLM and, on amd64, transcription — this is the only node list |
| `VLLM_MODELS` | `id,id` | Models to deploy. Defined in `scheduler/models.yaml` |

`setup.sh` refuses to continue unless at least one of `OPENROUTER_API_KEY` or a
vLLM node is present.

## 2. Exposure

| Variable | Default | Notes |
|---|---|---|
| `GATEWAY_PORT` | `8080` | The only published port |
| `COMPOSE_PROFILES` | `tools,models` | What to run. `setup.sh` appends `whisper` (the transcription shim) once a backend answers |

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
| `SCRAPER_API_KEY` | string | Injected by the gateway into document fetch requests |

The two injected keys **never reach the UI**: they exist only inside the gateway.

## 4. Placement results — written by the scheduler

Do not set these by hand. The placement step of `setup.sh all` writes them.

| Variable | Contents |
|---|---|
| `VLLM_<MODEL>_URL` | CSV of node addresses serving that model |
| `VLLM_<MODEL>_MAX_LEN` | Context decided for it |
| `VLLM_<MODEL>_GPU_UTIL` | `--gpu-memory-utilization` decided for it |
| `WHISPER_URLS` | CSV of nodes whose transcription backend answered. Empty means STT goes to OpenRouter |

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

Where the defaults come from, and how they relate to the placement step, is in
[GPU memory](gpu-memory.md#tuning-knobs).

## 7. LiteLLM behaviour

| Variable | Default | Notes |
|---|---|---|
| `LITELLM_NUM_WORKERS` | `4` | ~600 MB per worker |
| `LITELLM_LOG` | `INFO` | |
| `KC_OR_VARIANT` | `:floor` | OpenRouter provider routing variant. `:nitro` for throughput, empty for the OpenRouter default |
| `CONCURRENCY_GATE_DEBUG` | (empty) | Log the overload gate's decisions |
| `CONCURRENCY_GATE_FORCE` | (empty) | Force the gate on or off |

## 8. Deep research

| Variable | Default | Notes |
|---|---|---|
| `DEEP_RESEARCH_MODEL` | `local/qwen3.6-35b` | Model used for iterative search. Needs the 128K floor the scheduler reserves on it |
| `DEEP_RESEARCH_LLM_URL` | `http://litellm:8000/v1` | LiteLLM on the same network |

## 9. Transcription

Runs on amd64 GPU nodes — `install-vllm.sh` installs it alongside vLLM, and the
planner subtracts 6 GiB per node for it. arm64 nodes do not install it and
delegate STT to OpenRouter, so these values go unused there.

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | |
| `WHISPER_DEVICE` | `auto` | GPU when CUDA is available |
| `WHISPER_COMPUTE_TYPE` | `float16` | `int8` on low-VRAM nodes |
| `WHISPER_PORT` | `9000` | Backend port on the node |
| `WHISPER_DATA_ROOT` | `/var/lib/whisper` | Weight cache, so weights are downloaded once |
| `TRANSCRIBE_TIMEOUT_SEC` | `3600` | Headroom for long meeting recordings |

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
