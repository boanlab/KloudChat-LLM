# Troubleshooting

The entry point when something will not start or has broken. Identify the
symptom here; the neighbouring documents hold the reference detail.

## After the first run: "this means it worked"

```bash
# 1) Containers healthy
docker compose ps

# 2) Gateway and per-capability reachability
./scripts/setup.sh urls

# 3) Model catalogue (LiteLLM is not published; go through the gateway)
KEY=$(grep ^LITELLM_MASTER_KEY .env | cut -d= -f2)
curl -sf -H "Authorization: Bearer $KEY" http://localhost:8080/litellm/v1/models | jq '.data | length'
```

- Every container `running (healthy)` means normal.
- `health: starting` for more than five minutes needs diagnosis.

## Container restart loop

```bash
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.RestartCount}}"
docker inspect <name> --format '{{.RestartCount}} / {{.State.Status}} / OOMKilled: {{.State.OOMKilled}} / ExitCode: {{.State.ExitCode}}'
docker logs --tail 100 <name>
```

- `OOMKilled: true`: check with `free -h`.
- cgroup OOM: compose `mem_limit`, or the host is out of RAM.
- NVRM OOM (vLLM): see [vLLM cold-start failure](#vllm-cold-start-failure).

## vLLM cold-start failure

Symptom: a `vllm-*` container restarts forever in `starting`, with
`Engine core initialization failed` in the log.

```bash
docker logs vllm-qwen35b 2>&1 | grep -B 5 "Engine core initialization\|CUDA\|out of memory" | head -30
sudo dmesg -T | grep -iE "nvrm|oom" | tail -10
```

| Symptom | Cause | Action |
|---|---|---|
| `_initialize_kv_caches` fails | `--gpu-memory-utilization` too low: no room for weights plus KV | Raise that model's `VLLM_<MODEL>_GPU_UTIL` in the node's `.env`, or re-run [placement](../scheduler/README.md) |
| `max_num_seqs (...) exceeds available Mamba cache blocks` | The hybrid Gated-DeltaNet in qwen3.6-35b requires `max_num_seqs ≤ state blocks` during cudagraph capture | Lower `VLLM_QWEN35B_MAX_NUM_SEQS` below the cap; the log prints the block count |
| `Assertion error (layout.hpp:60): Unknown SF transformation` | DeepGEMM rejects an FP8 block-quantised scale-factor layout on this card. Fails after the weights load | Set `VLLM_CODERNEXT_DEEP_GEMM=0` in the node's `.env` |
| `ModuleNotFoundError: 'pytest'` | The derived image is missing its pytest layer | `install-vllm.sh --reinstall` |
| `NVRM: Out of memory` (dmesg) | Unified memory (GB10): page cache plus co-resident vLLM | `sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`. If it recurs, shrink that node's models and re-run [placement](../scheduler/README.md) |
| OS killer during weight load | RAM smaller than the weights | Stop other containers |

## Swap thrash (mid-stream stall)

When the vLLM KV cache is pushed into swap, per-token latency grows into
seconds and the UI sits at "thinking".

```bash
free -h                                                  # 5 GB+ of swap used is suspicious
cat /proc/sys/vm/swappiness                              # 60 by default
```

| Action | Notes |
|---|---|
| `./scripts/tune-host.sh` | Persists `vm.swappiness=10` and related sysctls |
| `sudo sysctl vm.swappiness=10` | Immediate, not persistent |
| `sudo swapoff -a && sudo swapon -a` | Clears swap; risks OOM if RAM is tight |
| Shrink `VLLM_MODELS` and re-apply placement | The real fix |

## vLLM node unreachable

The vLLM discovery in `gen-litellm-config.sh` hit a TCP failure. If no local
models appear in the UI, discovery returned nothing.

```bash
# Ports: qwen35b 8001, bge-m3 8003, qwen122b 8004, coder30b 8006, qwen27b 8007,
# codernext 8008, rerank 8009, whisper 9000
curl -sf http://<vllm-host>:8001/v1/models | jq '.data[].id'
ss -tlnp | grep -E '800[1-9]|9000'
docker ps --filter name=vllm- --format '{{.Names}}\t{{.Status}}'
```

- Container not up: `./scripts/manage-vllm.sh up vllm-qwen35b` on the GPU
  node, then try again.
- Restarting forever: see [vLLM cold-start failure](#vllm-cold-start-failure).

## LiteLLM unreachable

```bash
docker logs --tail 50 kloudchat-litellm
curl -sf http://localhost:8080/litellm/health/liveliness   # 200
curl -sf http://localhost:8080/litellm/health/readiness    # verifies backends
```

| Cause | Action |
|---|---|
| `LITELLM_MASTER_KEY` empty | Re-run `gen-env.sh`, or fill it in |
| Database migration failed | `docker logs kloudchat-litellm-db`: is postgres healthy |
| Hit `:8000` directly | Use `/litellm/*`; the gateway is the only published port |

## Models missing from the menu

```bash
KEY=$(grep ^LITELLM_MASTER_KEY .env | cut -d= -f2)
curl -sf -H "Authorization: Bearer $KEY" http://localhost:8080/litellm/v1/model/info | jq '.data | map(.model_name)'
docker exec kloudchat-litellm cat /app/config.yaml | grep -E "model_name|api_base" | head -40
```

- Missing from `model/info` too: a config generation problem. Run
  `./scripts/gen-litellm-config.sh`, then
  `docker compose up -d --force-recreate litellm`.
- Only the local models missing: `VLLM_*_URL` is empty in `.env`, or the node
  is down (see the section above).
- Present in `model/info` but not in the UI: a UI-side problem. Check the
  backend address in the admin screen and the team model allowlist
  (`./scripts/manage.sh team sync`).

## Users and keys

Accounts are created in the UI (sign-up followed by admin approval), and that
API provisions the matching LiteLLM user and per-user key. `manage.sh` is the
LiteLLM-side view of them.

```bash
./scripts/manage.sh user list                 # LiteLLM users
./scripts/manage.sh user usage --user <email> # this month's spend against budget
./scripts/manage.sh key list --user <email>
```

## Deep research failures

Symptom: a context-window error from vLLM, `This model's maximum context
length is N tokens`, surfaced as a 400. Nothing trims the request on the way
through, so an input over the serving context fails outright.

```bash
# 1) The serving context of the research model
KEY=$(grep ^LITELLM_MASTER_KEY .env | cut -d= -f2)
curl -sf -H "Authorization: Bearer $KEY" http://localhost:8080/litellm/v1/model/info \
  | jq '.data[] | select(.model_name=="local/qwen3.5-122b-a10b") | {model_name, api_base: .litellm_params.api_base, max_input: .model_info.max_input_tokens}'

# 2) If it is absent, diagnose placement
./scripts/setup.sh scheduler inventory   # per-node GPU class, VRAM, running containers
./scripts/setup.sh scheduler plan        # what context the planner chose, and why a model was delegated

# 3) The MCP itself
docker logs kloudchat-deep-research --tail 50
```

| Cause | Action |
|---|---|
| Research model not placed | `plan` prints the reason. Add a node or shrink `VLLM_MODELS`, then apply and re-run `gen-litellm-config.sh`. Until then the same name is served by OpenRouter |
| Placed but still failing | LDR's accumulated input exceeds the serving context. Reduce `LDR_SEARCH_ITERATIONS` in `docker-compose.yml`, or point `DEEP_RESEARCH_MODEL` at a model with a larger context |

## file_search returns nothing

Retrieval is opt-in. Without the `index` profile KloudChat falls back to
lexical search, so empty results are a configuration answer before they are a
fault. See [models.md](models.md#retrieval) for the two stages.

| Cause | Action |
|---|---|
| `index` not in `COMPOSE_PROFILES` | Add it and re-run `setup.sh up`; `index-db` and `index-shim` do not start otherwise |
| No embedding deployment | `INDEX_EMBED_MODELS` is tried in order: `local/bge-m3` needs a vLLM placement, `text-embedding-3-small` needs `OPENAI_API_KEY`. `GET /tools/index/health` reports whether embeddings answer |
| Collection never indexed | The index starts empty; KloudChat fills it through `PUT /tools/index/documents`. Losing the volume costs a re-index, not a document |
| Results arrive but are weak | The reranker may be missing; the search response carries `"reranked": true\|false`. Without it search falls back to vector order |

## Diagnostic helpers

| Command | Purpose |
|---|---|
| `manage.sh user list` / `team list` / `key list` | LiteLLM users, teams and virtual keys |
| `manage.sh user usage [--user <email>]` | Per-user spend against the monthly budget |
| `manage.sh user topup --user <email> --amount <N>` | Temporarily raise the monthly limit by $N. The original limit is recorded in `data/ledger/topups.json` and restored at the monthly reset |
| `manage.sh key show [--user <email>]` | Plaintext keys from the local ledger |
| `manage-vllm.sh status` | vLLM container and healthcheck status |
| `manage-vllm.sh logs <svc>` | vLLM logs |
| `setup.sh scheduler inventory` | Per-node GPU class, VRAM, running containers |
| `setup.sh scheduler plan` | Target placement (dry run) |
| `tune-host.sh --check` | Recommended sysctl values against the current ones |
| `gen-litellm-config.sh --check-prices` | Declared prices against the OpenRouter catalogue |

## Operator knobs

| Variable | Default | Purpose |
|---|---|---|
| `KLOUDCHAT_VLLM_WAIT_TIMEOUT` | 1200 s | Deadline for vLLM readiness |
| `KLOUDCHAT_VLLM_WAIT_INTERVAL` | 10 s | Probe interval |
| `KLOUDCHAT_SKIP_SCHEDULER` | (off) | Skip the placement step in `setup.sh all` |
| `KLOUDCHAT_SERVICE_WAIT` | 180 s | How long to wait for capabilities after startup |
| `KLOUDCHAT_REMOTE_DIR` | `KloudChat-LLM` | Repository path on remote nodes |

All are shell variables. The full list is in the
[environment variable reference](env-reference.md).

## Starting over (destructive)

```bash
./scripts/setup.sh clean        # remove containers and ./data
./scripts/gen-env.sh --force    # regenerate .env
./scripts/setup.sh all
```

> This cannot be undone. Copy the directories below first.

- `./data/litellm/postgres` (LiteLLM), and in the UI repository
  `./data/postgres` and `./data/minio`
- `./data/ledger`: issued virtual keys in plaintext (`keys.json`), the topup
  ledger (`topups.json`) and the team cache (`teams.json`). LiteLLM stores only
  key hashes, so losing `keys.json` loses the plaintext of every issued key.

## See also

- [Environment variables](env-reference.md): every `.env` key
- [Scheduler](../scheduler/README.md): multi-node vLLM placement
- [GPU memory](gpu-memory.md): what fits per node class, and the vLLM tuning knobs
