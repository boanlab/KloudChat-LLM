# Environment variables

`./scripts/gen-env.sh` creates `.env` (mode 600) from `.env.example` and
replaces every `change-me-*` value with a generated secret; what is left to
fill in is external keys and node addresses. Secret values are not printed.

```bash
./scripts/gen-env.sh          # create (skipped if .env exists)
./scripts/gen-env.sh --fill   # fill only empty or change-me-* secrets; the rest stays
./scripts/gen-env.sh --force  # recreate; every secret changes, the old file kept as .env.bak.<time>
```

An existing database keeps the password it was created with: after `--force`,
or a `--fill` of `LITELLM_DB_PASSWORD` / `INDEX_DB_PASSWORD`, restore the old
value or reset that database.

`.env.example` runs in this order: keys you provide, where the models run
(`VLLM_NODES`, `VLLM_MODELS`, the servers you run by hand, then what the
scheduler writes), exposure and sizing, the optional features, and the generated
secrets last. The sections below group the same variables by who sets them.

## 1. Filled in by a human

| Variable | Value | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | `sk-or-v1-...` | Commercial models and local fallback. Required without a GPU |
| `HF_TOKEN` | token | Hugging Face gated repositories. Weight downloads only |
| `VLLM_NODES` | `user@host,...` | GPU node SSH targets. Empty means no local models. **Order matters**: the first target is the head node (27B, retrieval, transcription); the rest are the pool (Flash-Next, one per node); chat replicas land on any node with room. See [models.md](models.md#where-models-are-defined) |
| `VLLM_NODES_HOLD` | `user@host,...`, optional | Targets in `VLLM_NODES` the scheduler leaves alone: probed and listed by `scheduler inventory`, nothing placed, started or stopped there, and `setup.sh vllm` installs nothing. For a node that runs something else, such as the image models. An entry is the target, its host or its node id; the head node cannot be held. See [scheduler](../scheduler/README.md#held-nodes) |
| `VLLM_MODELS` | `id,id` | Models to deploy. Defined in `scheduler/models.yaml`. Default `qwen3.8-27b,qwen3.8-flash,bge-m3,bge-reranker-v2-m3,whisper-large-v3`; `qwen3.8-flash` needs a GB10 or RTX PRO 6000 pool node to itself (drop it on a single node) |
| `VLLM_ENDPOINTS` | `http://host:port,id=http://host:port` | vLLM servers you run yourself: a bare URL is asked what it serves, `id=URL` is registered even while down. Never written by the scheduler, which counts each entry as an instance of its model: a model served here is not placed again for coverage, and the model's `replicas` in `models.yaml` caps the total. A scheduled node's service an entry points at is never stopped. See [models.md](models.md#vllm-local) |
| `NAVER_CLIENT_ID` / `NAVER_CLIENT_SECRET` | optional | NAVER API HUB credentials; both set enables the `naver web` and `naver news` engines |

`setup.sh` refuses to continue unless at least one of `OPENROUTER_API_KEY` or a
vLLM node is present.

## 2. Exposure and sizing

| Variable | Default | Notes |
|---|---|---|
| `GATEWAY_PORT` | `8080` | The only published port |
| `COMPOSE_PROFILES` | `tools,models` | What to run. `setup.sh` appends `whisper` once the transcription model is placed and `imagegen` once `IMAGEGEN_BACKENDS` is set. Add `index` for the retrieval index, `ocr` for text from scanned pages |
| `LITELLM_NUM_WORKERS` | `4` | LiteLLM workers, ~600 MB of RAM each |
| `CODE_INTERPRETER_CPUS` / `CODE_INTERPRETER_MEM_LIMIT` | `4` / `4g` | Cores and memory (no swap) the code-execution container may use across every sandbox; one sandbox takes one core and up to 1 GiB |

## 3. Generated secrets

Filled in by `gen-env.sh`. If you create them yourself, keep the formats below.

| Variable | Format | Used by |
|---|---|---|
| `LITELLM_MASTER_KEY` | `sk-` + 64 hex | LiteLLM admin API, entered in the UI admin screen. Also the password for LiteLLM's admin UI at `http://<host>:<GATEWAY_PORT>/litellm/ui/` |
| `LITELLM_DB_PASSWORD` | 32 hex | LiteLLM's postgres |
| `LITELLM_DB_USER` | string | `kloudchat-litellm` in `.env.example`; compose falls back to `litellm` |
| `SEARXNG_SECRET_KEY` | 32 hex | SearXNG session signing |
| `TOOLS_GATEWAY_KEY` | 64 hex | Required by the gateway on every `/tools/*` request as `X-Tools-Key`; 401 otherwise. Set the same value in KloudChat's `.env` as `KCHAT_TOOLS_GATEWAY_KEY`. Compose refuses to start the gateway without it, and a value under 32 characters denies all of `/tools`. Hex only. Rotation: [tools.md](tools.md#authentication) |
| `CODE_INTERPRETER_API_KEY` | 32 hex | Injected by the gateway into code execution requests |
| `CODE_INTERPRETER_MINIO_USER` | `code-interpreter` | MinIO user for code execution artifacts |
| `CODE_INTERPRETER_MINIO_PASSWORD` | 32 hex | MinIO password |
| `CODE_INTERPRETER_MINIO_BUCKET` | `code-interpreter` | MinIO bucket |
| `SCRAPER_API_KEY` | 32 hex | Injected by the gateway into document fetch requests, checked by the shim |
| `INDEX_DB_USER` / `INDEX_DB_PASSWORD` | `index` / 32 hex | The retrieval index's pgvector database. The password is required whenever the `index` profile is on |

The two injected keys never reach the UI: they exist only inside the gateway.
`TOOLS_GATEWAY_KEY` is the one the UI does hold; the gateway strips it before
proxying.

## 4. Placement results, written by the scheduler

Do not set these by hand. The placement step of `setup.sh all` writes them; set
`KLOUDCHAT_SKIP_SCHEDULER=1` to manage them yourself. The prefix
(`VLLM_QWEN27B` and so on) is the `env_prefix` in `scheduler/models.yaml`.

| Variable | Where | Contents |
|---|---|---|
| `VLLM_PLACED` | compose host | CSV of `id=http://host:port`, one per placed instance. Empty when nothing is placed |
| `VLLM_<PREFIX>_MAX_LEN` | node (`.env` under `KLOUDCHAT_REMOTE_DIR`) | Context decided for it |
| `VLLM_<PREFIX>_GPU_UTIL` | node | `--gpu-memory-utilization` decided for it |
| `VLLM_<PREFIX>_TP` | node | Cards to shard across, from `tensor_parallel` in models.yaml. Written only when it is not 1 |
| `VLLM_<PREFIX>_DEVICES` | node | Cards for the container: the `device_ids` of its GPU reservation and its `NVIDIA_VISIBLE_DEVICES`. Empty or unset: `all`. Written only on multi-card nodes |

Set by hand on the compose host; the scheduler does not write these:

| Variable | Contents |
|---|---|
| `WHISPER_URLS` | Transcription servers (`vllm-whisper`) you run by hand, CSV of URLs. The scheduler lists its own as `whisper-large-v3` in `VLLM_PLACED`. With neither, dictation is off; audio is never sent to OpenRouter |
| `IMAGEGEN_BACKENDS` | Self-hosted image models you run by hand, as `name=URL` pairs without `/v1`, one per node serving the name. Set: `setup.sh` enables the `imagegen` profile and `gen-litellm-config.sh` registers each name as `strict-local/<name>` only. See [models.md](models.md#self-hosted-image-models) |
| `GPU_EXPORTER_PORT` | gpu-exporter port on every GPU node, default `9835`. metrics-shim finds the nodes from `VLLM_ENDPOINTS`, `VLLM_PLACED`, `WHISPER_URLS`, `IMAGEGEN_BACKENDS`, `IMAGEGEN_PLACED` and `VLLM_NODES`. See [tools.md](tools.md#metrics-vllm-and-gpu-exporter) |

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

Read by `docker-compose.vllm.yml` (and, for `VLLM_OMNI_*`, `docker-compose.vllm-omni.yml`) on the node. They are listed in
`.env.node.example`, not `.env.example`: `install-vllm.sh` creates a node's `.env` from
it when absent, and every value is also the compose default. `MAX_LEN` and
`GPU_UTIL` are overwritten by the placement step; the defaults apply when
placement is skipped.

| Variable | Default | Notes |
|---|---|---|
| `VLLM_IMAGE` | `kloudchat-vllm:local` | The image compose runs: this repo's layer over the upstream vLLM image, built and recorded by `install-vllm.sh`. Another value on a node is an override that `install-vllm.sh` keeps |
| `VLLM_BASE_IMAGE` / `VLLM_BASE_DIGEST` | (empty) | Upstream image and the digest it resolved to, recorded by `install-vllm.sh`. A rebuild pins to the digest; later runs reuse the recorded image, and `install-vllm.sh --reinstall --image <tag>` moves to another (`vllm/vllm-openai:latest` to upgrade) |
| `VLLM_QWENFLASH_ENGRAM` | (empty) | Flash-Next's `--engram-config`. Empty: `{"checkpoint_mapped": true}`, the PLE table mapped from the checkpoint (GB10). `install-vllm.sh` writes `{}` on discrete cards: the table in pinned host RAM |
| `VLLM_QWENFLASH_KV_OFFLOAD_GB` | (empty) | Flash-Next's `--kv-offloading-size`: KV blocks spilled to host RAM, in GiB. Empty is off (vLLM's `None`); worth setting on a discrete card in a host with RAM to spare (`96` on a 375 GiB RTX PRO 6000 host), never on GB10 |
| `VLLM_MODELS_ROOT` | `/var/lib/vllm/models` | Checkpoint root on the node |
| `VLLM_<PREFIX>_DIR` | model `dir` in models.yaml | Checkpoint directory under the root (`VLLM_QWEN27B_DIR`, `VLLM_QWENFLASH_DIR`, `VLLM_BGEM3_DIR`, `VLLM_RERANK_DIR`, `VLLM_WHISPER_DIR`) |
| `VLLM_OMNI_IMAGE` | `vllm/vllm-omni:v0.30.0` | Image for `docker-compose.vllm-omni.yml` |
| `VLLM_OMNI_ZIMAGE_DIR` / `VLLM_OMNI_QWENIMAGE_DIR` / `VLLM_OMNI_QWENIMAGEEDIT_DIR` | `z-image-turbo` / `qwen-image-2512` / `qwen-image-edit-2511` | Image-model checkpoint directories under `VLLM_MODELS_ROOT` |
| `VLLM_OMNI_DEVICES` | `all` | Cards for the image models (GPU reservation `device_ids` and `NVIDIA_VISIBLE_DEVICES`); the default for the three per-service variables below |
| `VLLM_OMNI_ZIMAGE_DEVICES` / `VLLM_OMNI_QWENIMAGE_DEVICES` / `VLLM_OMNI_QWENIMAGEEDIT_DEVICES` | `VLLM_OMNI_DEVICES` | Cards for `omni-z-image` / `omni-qwen-image` / `omni-qwen-image-edit` (GPU reservation `device_ids` and `NVIDIA_VISIBLE_DEVICES`), to place each image model on its own card (`0`, `1`). Takes effect on `manage-vllm-omni.sh up --recreate` |
| `VLLM_OMNI_QWEN3TTS_DIR` | `qwen3-tts` | Speech-model checkpoint directory under `VLLM_MODELS_ROOT` |
| `VLLM_OMNI_QWEN3TTS_DEVICES` | `VLLM_OMNI_DEVICES` | Cards for `omni-qwen3-tts` (GPU reservation `device_ids`) |
| `VLLM_OMNI_QWEN3TTS_STAGE0_UTIL` / `_STAGE1_UTIL` | `0.06` / `0.04` | `gpu_memory_utilization` of the talker and code2wav stages, as fractions of one card (5.7 + 3.8 GiB on 95.6 GiB). Raise on a smaller card |
| `WHISPER_MAX_UPLOAD_MB` | `100` | Upload ceiling for `vllm-whisper` (`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`) and for whisper-shim on the compose host |
| `WHISPER_MAX_CONCURRENT` | `8` | Transcriptions whisper-shim runs at once; past it, 503 |

Per-model `GPU_UTIL`, `MAX_LEN`, `MAX_BATCHED_TOKENS`, `MAX_NUM_SEQS` and `SPEC_TOKENS` defaults are in [GPU memory](gpu-memory.md#tuning-knobs).

## 7. Deep research

| Variable | Default | Notes |
|---|---|---|
| `DEEP_RESEARCH_MODEL` | `qwen3.8-27b` | Model for iterative search, served at its native 262144 context. A model id runs as `local/<id>`; a name with a slash (`qwen/qwen3.8-27b`) is used as written |
| `DEEP_RESEARCH_LLM_URL` | `http://litellm:8000/v1` | LiteLLM on the same network |

## 8. Retrieval index (profile `index`)

| Variable | Default | Notes |
|---|---|---|
| `INDEX_EMBED_MODELS` | `strict-local/bge-m3` | Embedding models, tried in order through LiteLLM by name. Local only |
| `INDEX_RERANK_MODEL` | `strict-local/bge-reranker-v2-m3` | Reranker. Empty turns reranking off |
| `INDEX_RERANK_CANDIDATES` | `5` | Over-fetch factor: `limit × candidates` passages go to the reranker |
| `INDEX_RERANK_MIN_SCORE` | `0.1` | Reranker score floor |
| `INDEX_RERANK_RECALL_DISTANCE` | `0.85` | Cosine distance bound applied before reranking |
| `INDEX_LITELLM_URL` | `http://litellm:8000` | LiteLLM on the same network |

## 9. Text from scanned pages (profile `ocr`)

| Variable | Default | Notes |
|---|---|---|
| `OCR_CPUS` / `OCR_MEM_LIMIT` | `4` / `3g` | Cores and memory the container may use. Pages are read on CPU only |
| `OCR_CONCURRENCY` / `OCR_THREADS` | `1` / `4` | Pages read at once, and ONNX Runtime threads per page. Keep `CONCURRENCY × THREADS` at or below `OCR_CPUS` |
| `OCR_MAX_BODY_MB` | `64` | Largest document accepted |
| `OCR_DPI` | `200` | Resolution a PDF page is rendered at before reading |

## 10. LiteLLM

| Variable | Default | Notes |
|---|---|---|
| `LITELLM_LOG` | `INFO` | |
| `LITELLM_URL` | `http://localhost:<GATEWAY_PORT>/litellm` | Where `manage.sh` reaches LiteLLM. Read from the shell first, then `.env` |
| `OR_PROVIDER_ALLOW` | `qwen,deepseek,anthropic,google,moonshotai,nvidia,openai,perplexity,x-ai,z-ai,bytedance-seed,meta-llama,mistralai` | OpenRouter providers `gen-litellm-config.sh` registers, for commercial, image, audio and free models. An empty value means this default, not "no limit"; `*` allows every provider. Providers outside the list (tencent, xiaomi, minimax, recraft, liquid, ...) are not emitted, and regenerating does not bring them back. `local/` and `strict-local/` routes are unaffected. Read from the shell first, then `.env` |
| `OR_MODEL_EXCLUDE` | (empty) | Comma-separated OpenRouter models to remove from what is allowed: a `provider/` prefix, a `provider/model` id, or a glob. Free models keep their `:free` suffix in the id, and `provider/model` also matches `provider/model:free`. Example: `OR_MODEL_EXCLUDE=tencent/*,liquid/*,recursive/*,thinkingmachines/*`. The generator logs one line naming everything left out, and the team allowlist (`litellm_chat_models_csv`) follows the same filters |
| `OR_OPEN_WEIGHT_POLICY` | `off` | `strict`: an open-weight model (see `OR_OPEN_WEIGHT`) runs only on OpenRouter endpoints that disclose their quantization and are ZDR; a model with none is left out. Closed models are never affected. See [Run policy](models.md#run-policy) |
| `OR_OPEN_WEIGHT` | `openai/gpt-oss-*,google/gemma-*,meta-llama/*,nvidia/nemotron-*,deepseek/*,z-ai/*,moonshotai/*,qwen/qwen3-coder*,qwen/qwen3.5-*,qwen/qwen3.6-*,qwen/qwen3-vl-*,mistralai/mistral-nemo,mistralai/ministral-*,mistralai/mistral-small-*` | Comma-separated globs of the models the strict policy treats as open-weight. Empty means this default |
| `OR_OPEN_WEIGHT_QUANTIZATIONS` | `int4,int8,fp4,fp6,fp8,fp16,bf16,fp32` | The `provider.quantizations` values sent under the strict policy: every value OpenRouter accepts except `unknown` |
| `OR_PROVIDER_PIN` | (empty) | Comma-separated `pattern=Provider[\|Provider]`, e.g. `anthropic/*=Anthropic,moonshotai/kimi-k3=Wafer`. A matching model runs on that provider only. An ineligible pin is ignored with a warning. See [Run policy](models.md#run-policy) |
| `OR_ZDR_ONLY` | `0` | `1`: every OpenRouter request carries `provider.zdr: true`; a model with no ZDR endpoint is left out |
| `CONCURRENCY_GATE_CAPS` | each model's `max_inflight` in `scheduler/models.yaml` (256 when unset), on both the `local/` and `strict-local/` alias | JSON map of model aliases to positive concurrency caps, overriding those |
| `CONCURRENCY_GATE_TTL` | `1.5` | Seconds between vLLM capacity polls |
| `CONCURRENCY_GATE_SCRAPE_TIMEOUT` | `1.0` | Timeout in seconds for one vLLM metrics request |
| `CONCURRENCY_GATE_DEBUG` | (empty) | `1` logs the overload gate's decisions |
| `CONCURRENCY_GATE_FORCE` | (empty) | Comma-separated model aliases to force through the saturated path, for testing |

Read by `gen-litellm-config.sh` from the shell, not from `.env`:

| Variable | Default | Notes |
|---|---|---|
| `KC_OR_VARIANT` | `:floor` | OpenRouter provider routing suffix on chat routes. `:nitro` for throughput, empty for the OpenRouter default |
| `KC_<PREFIX>_QUANTIZATION`, `KC_<PREFIX>_KV_CACHE`, `KC_<PREFIX>_ENGINE` | read or empty | How a self-hosted model runs, shown in the picker. `<PREFIX>` is the model's `env_prefix` without `VLLM_` (`QWEN27B`, `QWENFLASH`). The generator reads the engine version (`/version`) and KV cache dtype (`/metrics`) from the server and uses these only where the server does not say; the weights' quantization is only declared. Example for Qwen3.8-27B: `KC_QWEN27B_QUANTIZATION=NVFP4`, `KC_QWEN27B_KV_CACHE=fp8`, `KC_QWEN27B_ENGINE="vLLM 0.33.0"` |
| `KC_<PREFIX>_PRESENCE_PENALTY`, `KC_<PREFIX>_TEMPERATURE` | the model's `presence_penalty` / `temperature` in `models.yaml` | Default sampling value on a self-hosted deployment, used when a request sends none. Empty sends none |
| `KC_PRE_CALL_HEADROOM` | `4096` | Tokens subtracted from each local model's context when declaring `max_input_tokens` |
| `KLOUDCHAT_ENV_FILE`, `KLOUDCHAT_LITELLM_CONFIG_FILE`, `KLOUDCHAT_LITELLM_CONFIG_EXAMPLE` | repository paths | Overrides used by the tests |

## 11. Container-level defaults

Read by the service code, with the value compose passes where it passes one.
Changing the others means editing `docker-compose.yml`.

| Container | Variable | Default | Notes |
|---|---|---|---|
| whisper-shim | `WHISPER_MODEL_NAME` | `local/whisper-large-v3` | Name set on every forwarded request. Must be one of `vllm-whisper`'s `--served-model-name` values |
| whisper-shim | `HEALTH_PROBE_TIMEOUT_SEC` | `2.0` | Per-backend health probe |
| whisper-shim | `HEALTH_CACHE_TTL_SEC` | `10` | Health cache lifetime |
| whisper-shim | `WHISPER_MAX_UPLOAD_MB` / `WHISPER_MAX_CONCURRENT` | `100` / `8`; compose passes them from `.env` | Largest upload (413 past it, all files together), and transcriptions at once (503 with `Retry-After` past it) |
| whisper-shim | `TRANSCRIBE_TIMEOUT_SEC` | `900`; compose passes `${TRANSCRIBE_TIMEOUT_SEC:-3600}` from `.env` | Per-request timeout, sized for long recordings |
| imagegen-shim | `GENERATE_TIMEOUT_SEC` / `HEALTH_PROBE_TIMEOUT_SEC` | `600` / `2.0` | Per-picture timeout towards vllm-omni, and the per-backend health probe |
| imagegen-shim | `QWEN_IMAGE_STEPS` | `30` | Sampling steps for the Qwen-Image family |
| imagegen-shim | `QWEN_IMAGE_EDIT_STEPS` | `40` | Sampling steps for Qwen-Image-Edit |
| index-shim | `EMBED_DIM` | `1536` | Vector column width (`bge-m3` is 1024, zero-padded to it). Changing it after indexing is a migration; a mismatch makes `/health` report `degraded` |
| index-shim | `INDEX_CHUNK_CHARS` | `900` | Chunk window |
| index-shim | `INDEX_CHUNK_OVERLAP` | `150` | Chunk overlap |
| index-shim | `INDEX_MAX_DOC_CHARS` | `2000000` | Documents are truncated beyond this |
| index-shim | `INDEX_HEALTH_EMBED_TTL_SEC` / `INDEX_HEALTH_EMBED_TIMEOUT_SEC` | `300` / `3` | How often `/health` re-checks embeddings, and how long it waits for that check |
| index-shim | `EMBED_MODELS`, `RERANK_*`, `LITELLM_URL` | see section 8 | Compose maps them from the `INDEX_*` keys |
| ocr-shim | `OCR_CONCURRENCY` / `OCR_THREADS` | `1` / `4`; compose passes them from `.env` | Pages read at once, and ONNX Runtime threads per page |
| ocr-shim | `OCR_MAX_BODY_MB` / `OCR_MAX_PIXELS` / `OCR_MAX_PAGES` | `64` / `40000000` / `2000` | Largest document, largest decoded page, most pages the service opens |
| ocr-shim | `OCR_DPI` / `OCR_IMAGE_SIDE` | `200` / `1600` | Render resolution of a PDF page, and the longest side of the picture returned for a vision model |
| ocr-shim | `OCR_TIMEOUT_SEC` / `OCR_MIN_SCORE` | `120` / `0.5` | Per-page time limit (a page past it answers 504 but keeps its slot until the work ends; a request waits this long for a slot, then 503; a page running 3× this makes `/health` 503 `stuck`, 5× ends the process for a restart), and the recogniser score below which a line is dropped |
| ocr-shim | `OCR_QUEUE` | `4` | Requests that may wait for a slot; past that a request gets 503 before its body is read |
| crawl4ai-shim | `DEFAULT_TIMEOUT_MS` | `30000` | Page load timeout |
| crawl4ai-shim | `MAX_CONCURRENT_PAGES` / `QUEUE_TIMEOUT_MS` / `CACHE_TTL_S` / `CACHE_MAX_ENTRIES` | `8` / `15000` / `900` / `512` | Pages rendering at once, the wait for a slot, how long a good scrape is reused, and how many are kept |
| crawl4ai-shim | `MAX_TIMEOUT_MS` / `CACHE_MAX_ENTRY_BYTES` | `60000` / `2097152` | Ceiling on a caller's `timeout` (floor 1000 ms; `waitFor` at most the timeout), and the largest scrape the cache keeps (larger ones are served, not cached) |
| crawl4ai-shim | `ADULT_HOSTS_FILE` | `/app/adult-hosts.txt` | Hosts file of adult sites a scrape refuses; baked into the image |
| crawl4ai-shim | `USER_AGENT` | browser-like string | Sent by Chromium |
| crawl4ai-shim | `BROWSER_PROBE_INTERVAL_S` / `BROWSER_DEAD_LIMIT` / `BROWSER_EXIT_LIMIT` | `30` / `3` / `6` | How often a local page is rendered while no scrape has shown the browser working; failures in a row before `/health` answers 503, and before the process exits for Docker to restart it |
| search-shim | `SEARXNG_URL` | `http://searxng:8080` | The SearXNG behind it |
| search-shim | `MAX_CONCURRENT_SEARCHES` / `QUEUE_TIMEOUT_MS` | `12` / `10000` | Searches SearXNG runs at once, and the wait for a slot before `503 busy` |
| search-shim | `CACHE_TTL_S` / `STALE_TTL_S` / `CACHE_MAX_ENTRIES` | `900` / `21600` / `2048` | How long an answer with results is reused, how much longer it is served while SearXNG cannot answer, and how many are kept |
| search-shim | `UPSTREAM_TIMEOUT_S` / `CONNECT_TIMEOUT_S` | `20` / `3` | Whole-call and connect timeouts towards SearXNG |
| search-shim | `SAFESEARCH` | `2` | Forced on every search: 0 off, 1 moderate, 2 strict |
| code-interpreter | `MAX_EXECUTION_TIME` / `MAX_MEMORY_MB` | `30` / `1024` | Sandbox limits, set in compose; memory is a per-sandbox cgroup `memory.max` |
| code-interpreter | `MAX_SANDBOX_PIDS` / `MAX_SANDBOX_CPU_MS` | `256` / `1000` | Per-sandbox cgroup process count and CPU milliseconds per second (1000 = one core), set in compose |
| code-interpreter | `NSJAIL_BINARY` | `/usr/local/bin/nsjail-memcg` | nsjail wrapper that applies the sandbox limits |
| all shims | `LOG_LEVEL` | `INFO` | |

## Shell-only variables

Passed on the command line rather than through `.env`.

| Variable | Effect |
|---|---|
| `KLOUDCHAT_SKIP_SCHEDULER=1` | Skip the placement step in `setup.sh all` |
| `KLOUDCHAT_REMOTE_DIR` | Repository path on remote nodes, resolved from the SSH user's home (default `KloudChat-LLM`). rsync and the placement step both use it; the scheduler also reads it from `.env` |
| `KLOUDCHAT_VLLM_WAIT_TIMEOUT` | Deadline for waiting on vLLM readiness (default 1200 s) |
| `KLOUDCHAT_VLLM_WAIT_INTERVAL` | Probe interval while waiting (default 10 s) |
| `KLOUDCHAT_SERVICE_WAIT` | How long to wait for capabilities to answer after startup (default 180 s) |
| `KLOUDCHAT_SCHEDULER_NO_AUTOINSTALL=1` | Do not auto-install PyYAML through apt |
| `YES=1` | Skip the `clean` confirmation |
