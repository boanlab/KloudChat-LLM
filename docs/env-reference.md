# Environment variables

`.env` is produced by `./scripts/gen-env.sh` from `.env.example`. Every
`change-me-*` value is replaced with a generated secret, so the only values a
human fills in are external keys and node addresses.

```bash
./scripts/gen-env.sh          # create (skipped if .env exists)
./scripts/gen-env.sh --force  # recreate; every existing secret changes
```

## 1. Filled in by a human

| Variable | Value | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | `sk-or-v1-...` | Commercial models and local fallback. Required without a GPU |
| `HF_TOKEN` | token | Hugging Face gated repositories. Weight downloads only |
| `NODES_VLLM` | `user@host,...` | GPU node SSH targets. Empty means no local models. **Order matters**: the first target is the head node (default chat, retrieval, transcription); the rest are the pool (large picker models). See [models.md](models.md#where-models-are-defined) |
| `VLLM_MODELS` | `id,id` | Models to deploy. Defined in `scheduler/models.yaml` |
| `OPENAI_API_KEY` | optional | Registers `text-embedding-3-small` as the embedding fallback. Not in `.env.example`; add it by hand |

`setup.sh` refuses to continue unless at least one of `OPENROUTER_API_KEY` or a
vLLM node is present.

## 2. Exposure

| Variable | Default | Notes |
|---|---|---|
| `GATEWAY_PORT` | `8080` | The only published port |
| `COMPOSE_PROFILES` | `tools,models` | What to run. `setup.sh` appends `whisper` once the transcription model is placed. Add `index` for the retrieval index |

## 3. Generated secrets

Filled in by `gen-env.sh`. If you create them yourself, keep the formats below.

| Variable | Format | Used by |
|---|---|---|
| `LITELLM_MASTER_KEY` | `sk-` + 64 hex | LiteLLM admin API. Entered in the UI admin screen. Also the password for LiteLLM's own admin UI at `http://<host>:<GATEWAY_PORT>/litellm/ui/` (user `admin`), which is reachable on purpose: the gateway is only exposed on the internal network |
| `LITELLM_DB_PASSWORD` | 32 hex | LiteLLM's postgres |
| `LITELLM_DB_USER` | string | `kloudchat-litellm` by default |
| `SEARXNG_SECRET_KEY` | 32 hex | SearXNG session signing |
| `CODE_INTERPRETER_API_KEY` | 32 hex | Injected by the gateway into code execution requests |
| `CODE_INTERPRETER_MINIO_USER` | `code-interpreter` | MinIO user for code execution artifacts |
| `CODE_INTERPRETER_MINIO_PASSWORD` | 32 hex | MinIO password |
| `CODE_INTERPRETER_MINIO_BUCKET` | `code-interpreter` | MinIO bucket |
| `SCRAPER_API_KEY` | 32 hex | Injected by the gateway into document fetch requests, checked by the shim |
| `INDEX_DB_USER` / `INDEX_DB_PASSWORD` | `index` / 32 hex | The retrieval index's pgvector database. The password is required whenever the `index` profile is on |

The two injected keys never reach the UI: they exist only inside the gateway.

## 4. Placement results, written by the scheduler

Do not set these by hand. The placement step of `setup.sh all` writes them; set
`KLOUDCHAT_SKIP_SCHEDULER=1` to manage them yourself. The prefix
(`VLLM_QWEN35B` and so on) is the `env_prefix` in `scheduler/models.yaml`.

| Variable | Where | Contents |
|---|---|---|
| `VLLM_<MODEL>_URL` | compose host | CSV of node addresses serving that model. Empty for a model that is not placed |
| `WHISPER_URLS` | compose host | CSV of the nodes the transcription model was placed on. Empty sends STT to OpenRouter |
| `VLLM_<MODEL>_MAX_LEN` | node | Context decided for it |
| `VLLM_<MODEL>_GPU_UTIL` | node | `--gpu-memory-utilization` decided for it |
| `VLLM_<MODEL>_TP` | node | Cards to shard across, from `tensor_parallel` in models.yaml. Written only when it is not 1 |
| `VLLM_<MODEL>_DEVICES` | node | `NVIDIA_VISIBLE_DEVICES` for the container. Written only on multi-card nodes |

## 5. Images

| Variable | Default | Notes |
|---|---|---|
| `KLOUDCHAT_IMAGE_NS` | `boanlab` | Images are pulled and pushed as `<NS>/kloudchat-*` |
| `KLOUDCHAT_IMAGE_TAG` | `latest` | |

`setup.sh` pulls these. The `Publish images` workflow builds and pushes an
image whenever its service directory changes on `main`. `setup.sh all --build`
runs the working tree's own images instead. To publish to another registry,
change the namespace and use `./scripts/build-push-images.sh`.

## 6. GPU node startup options

Read by `docker-compose.vllm.yml` on the node. `MAX_LEN` and `GPU_UTIL` are
overwritten by the placement step; the `.env.example` values apply when
placement is skipped.

| Variable | Default | Notes |
|---|---|---|
| `VLLM_IMAGE` | `kloudchat-vllm:local` | The image compose runs: this repo's layer over the upstream vLLM image, built and recorded by `install-vllm.sh` |
| `VLLM_BASE_IMAGE` / `VLLM_BASE_DIGEST` | (empty) | Upstream image and the digest it resolved to, recorded by `install-vllm.sh`. A rebuild pins to the digest |
| `VLLM_MODELS_ROOT` | `/var/lib/vllm/models` | Checkpoint root on the node |
| `VLLM_<MODEL>_DIR` | model `dir` in models.yaml | Checkpoint directory under the root. Point it at an `-awq` download on an FP4-less card |
| `VLLM_<MODEL>_MAX_BATCHED_TOKENS` | `16384` | Lower bound for the vision mm-budget |
| `VLLM_<MODEL>_MAX_NUM_SEQS` | `128` (35B), `32` (122B) | CUDA-graph capture limit for the hybrid models |
| `VLLM_CODERNEXT_DEEP_GEMM` | `0` | `VLLM_USE_DEEP_GEMM` for `vllm-codernext`. DeepGEMM rejects this checkpoint's FP8 scale-factor layout on GB10; `1` where the kernel takes it |
| `VLLM_WHISPER_DIR` | `whisper-large-v3` | Transcription checkpoint directory |
| `WHISPER_MAX_UPLOAD_MB` | `100` | Upload ceiling for `vllm-whisper` (`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`) |

Defaults and their rationale are in [GPU memory](gpu-memory.md#tuning-knobs).

## 7. Deep research

| Variable | Default | Notes |
|---|---|---|
| `DEEP_RESEARCH_MODEL` | `local/qwen3.5-122b-a10b` | Model for iterative search. The scheduler holds a 128K context floor on it |
| `DEEP_RESEARCH_LLM_URL` | `http://litellm:8000/v1` | LiteLLM on the same network |

## 8. Retrieval index (profile `index`)

| Variable | Default | Notes |
|---|---|---|
| `INDEX_EMBED_MODELS` | `local/bge-m3,text-embedding-3-small` | Embedding preference order, tried through LiteLLM by name |
| `INDEX_RERANK_MODEL` | `local/bge-reranker-v2-m3` | Reranker. Empty turns reranking off |
| `INDEX_RERANK_CANDIDATES` | `5` | Over-fetch factor: `limit × candidates` passages go to the reranker |
| `INDEX_RERANK_MIN_SCORE` | `0.1` | Reranker score floor |
| `INDEX_RERANK_RECALL_DISTANCE` | `0.85` | Cosine distance bound applied before reranking |
| `INDEX_LITELLM_URL` | `http://litellm:8000` | LiteLLM on the same network |

## 9. LiteLLM

| Variable | Default | Notes |
|---|---|---|
| `LITELLM_NUM_WORKERS` | `4` | ~600 MB per worker |
| `LITELLM_LOG` | `INFO` | |
| `CONCURRENCY_GATE_CAPS` | built-in per-model caps | JSON map of model aliases to positive concurrency caps |
| `CONCURRENCY_GATE_TTL` | `1.5` | Seconds between vLLM capacity polls |
| `CONCURRENCY_GATE_SCRAPE_TIMEOUT` | `1.0` | Timeout in seconds for one vLLM metrics request |
| `CONCURRENCY_GATE_DEBUG` | (empty) | Log the overload gate's decisions |
| `CONCURRENCY_GATE_FORCE` | (empty) | Comma-separated model aliases to force through the saturated path, for testing |

Read by `gen-litellm-config.sh` from the shell, not from `.env`:

| Variable | Default | Notes |
|---|---|---|
| `KC_OR_VARIANT` | `:floor` | OpenRouter provider routing suffix on chat routes. `:nitro` for throughput, empty for the OpenRouter default |
| `KC_PRE_CALL_HEADROOM` | `4096` | Tokens subtracted from each local model's context when declaring `max_input_tokens` |
| `STT_OR_MODEL` | `mistralai/voxtral-small-24b-2507` | OpenRouter STT route, registered when no local whisper answers |
| `KLOUDCHAT_ENV_FILE`, `KLOUDCHAT_LITELLM_CONFIG_FILE`, `KLOUDCHAT_LITELLM_CONFIG_EXAMPLE` | repository paths | Overrides used by the tests |

## 10. Transcription shim

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_URLS` | (written by the scheduler) | Backends the shim balances across |
| `TRANSCRIBE_TIMEOUT_SEC` | `3600` | Headroom for long recordings |

## 11. Container-level defaults

Read by the service code, with the value compose passes where it passes one.
Changing the others means editing `docker-compose.yml`.

| Container | Variable | Default | Notes |
|---|---|---|---|
| whisper-shim | `WHISPER_MODEL_NAME` | `local/whisper-large-v3` | Name set on every forwarded request. Must be one of `vllm-whisper`'s `--served-model-name` values |
| whisper-shim | `HEALTH_PROBE_TIMEOUT_SEC` | `2.0` | Per-backend health probe |
| whisper-shim | `HEALTH_CACHE_TTL_SEC` | `10` | Health cache lifetime |
| whisper-shim | `TRANSCRIBE_TIMEOUT_SEC` | `900`; compose passes `3600` | |
| index-shim | `EMBED_DIM` | `1536` | Vector column width: the widest model in `EMBED_MODELS` (`text-embedding-3-small` is 1536; `bge-m3` is 1024, zero-padded). Changing it after indexing is a migration; a mismatch makes `/health` report `degraded` |
| index-shim | `INDEX_CHUNK_CHARS` | `900` | Chunk window |
| index-shim | `INDEX_CHUNK_OVERLAP` | `150` | Chunk overlap |
| index-shim | `INDEX_MAX_DOC_CHARS` | `2000000` | Documents are truncated beyond this |
| index-shim | `EMBED_MODELS`, `RERANK_*`, `LITELLM_URL` | see section 8 | Compose maps them from the `INDEX_*` keys |
| crawl4ai-shim | `DEFAULT_TIMEOUT_MS` | `30000` | Page load timeout |
| crawl4ai-shim | `USER_AGENT` | `Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 KloudChat/1.0` | |
| code-interpreter | `MAX_EXECUTION_TIME` / `MAX_MEMORY_MB` | `30` / `512` | Sandbox limits, set in compose |
| all shims | `LOG_LEVEL` | `INFO` | |

## Shell-only variables

Passed on the command line rather than through `.env`.

| Variable | Effect |
|---|---|
| `KLOUDCHAT_SKIP_SCHEDULER=1` | Skip the placement step in `setup.sh all` |
| `KLOUDCHAT_REMOTE_DIR` | Repository path on remote nodes (default `KloudChat-LLM`). rsync and the placement step both use it |
| `KLOUDCHAT_VLLM_WAIT_TIMEOUT` | Deadline for waiting on vLLM readiness (default 1200 s) |
| `KLOUDCHAT_VLLM_WAIT_INTERVAL` | Probe interval while waiting (default 10 s) |
| `KLOUDCHAT_SERVICE_WAIT` | How long to wait for capabilities to answer after startup (default 180 s) |
| `KLOUDCHAT_SCHEDULER_NO_AUTOINSTALL=1` | Do not auto-install PyYAML through apt |
| `YES=1` | Skip the `clean` confirmation |
