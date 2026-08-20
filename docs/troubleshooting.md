# Troubleshooting

The entry point when something will not start or has broken. Identify the
symptom here; the neighbouring documents hold the reference detail.

## After the first run — "this means it worked"

```bash
# 1) Containers healthy
docker compose ps

# 2) Gateway and per-capability reachability
./scripts/setup.sh urls

# 3) Model catalogue (LiteLLM is not published — go through the gateway)
KEY=$(grep ^LITELLM_MASTER_KEY .env | cut -d= -f2)
curl -sf -H "Authorization: Bearer $KEY" http://localhost:8080/litellm/v1/models | jq '.data | length'
```

- Every container `running (healthy)` means **normal**.
- `health: starting` for **more than five minutes** needs diagnosis.

## Container restart loop

```bash
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.RestartCount}}"
docker inspect <name> --format '{{.RestartCount}} / {{.State.Status}} / OOMKilled: {{.State.OOMKilled}} / ExitCode: {{.State.ExitCode}}'
docker logs --tail 100 <name>
```

- `OOMKilled: true` — check with `free -h`.
- **cgroup OOM** — compose `mem_limit`, or the host is out of RAM.
- **NVRM OOM** (vLLM) — see [vLLM cold-start failure](#vllm-cold-start-failure).

## vLLM cold-start failure

Symptom: `vllm-qwen35b` or `vllm-glmflash` restarts forever in `starting`, with
`Engine core initialization failed` in the log.

```bash
docker logs vllm-qwen35b 2>&1 | grep -B 5 "Engine core initialization\|CUDA\|out of memory" | head -30
sudo dmesg -T | grep -iE "nvrm|oom" | tail -10
```

| Symptom | Cause | Action |
|---|---|---|
| `_initialize_kv_caches` fails | `--gpu-memory-utilization` too low — no room for weights plus KV | Raise `VLLM_QWEN35B_GPU_UTIL` (or `VLLM_GLMFLASH_GPU_UTIL`) in `.env`, or re-run [placement](../scheduler/README.md) |
| `max_num_seqs (...) exceeds available Mamba cache blocks` | The hybrid Gated-DeltaNet in qwen3.6-35b requires `max_num_seqs ≤ state blocks` during cudagraph capture | Lower `VLLM_QWEN35B_MAX_NUM_SEQS` below the cap; the log prints the actual block count |
| `ModuleNotFoundError: 'pytest'` (amd64 cu129-nightly) | Upstream dynamo lazy-import regression, pytest layer missing | `install-vllm.sh --reinstall`, or `docker compose -f docker-compose.vllm.yml build` |
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
| `./scripts/tune-host.sh` | Persists `vm.swappiness=10` and friends |
| `sudo sysctl vm.swappiness=10` | Immediate, not persistent |
| `sudo swapoff -a && sudo swapon -a` | Clears swap — risks OOM if RAM is tight |
| Shrink `VLLM_MODELS` and re-apply placement | The real fix |

## vLLM node unreachable

The vLLM discovery in `gen-litellm-config.sh` hit a TCP failure. If no models
appear in the UI, discovery returned nothing.

```bash
# Model ports: chat 8001, floor 8002
curl -sf http://<vllm-host>:8001/v1/models | jq '.data[].id'
ss -tlnp | grep -E '8001|8002'
docker ps --filter name=vllm- --format '{{.Names}}\t{{.Status}}'
```

- If the container is not up, run `./scripts/manage-vllm.sh up vllm-qwen35b` on
  the GPU node and try again.
- If it restarts forever, see
  [vLLM cold-start failure](#vllm-cold-start-failure).

## LiteLLM unreachable

```bash
docker logs --tail 50 kloudchat-litellm
curl -sf http://localhost:8080/litellm/health/liveliness   # 200
curl -sf http://localhost:8080/litellm/health/readiness    # verifies backends
```

| Cause | Action |
|---|---|
| `LITELLM_MASTER_KEY` empty | Re-run `gen-env.sh`, or fill it in |
| Database migration failed | `docker logs kloudchat-litellm-db` — is postgres healthy |
| Hit `:8000` directly | Use `/litellm/*`; the gateway is the only published port |

## Models missing from the menu

```bash
KEY=$(grep ^LITELLM_MASTER_KEY .env | cut -d= -f2)
curl -sf -H "Authorization: Bearer $KEY" http://localhost:8080/litellm/v1/model/info | jq '.data | map(.model_name)'
docker exec kloudchat-litellm cat /app/config.yaml | grep -E "model_name|api_base" | head -40
```

- Missing from `model/info` too — a config generation problem: run
  `./scripts/gen-litellm-config.sh`, then
  `docker compose up -d --force-recreate litellm`.
- Only the local models missing — `VLLM_*_URL` is empty in `.env`, or the node is
  down (see the section above).
- Present in `model/info` but not in the UI — that is a UI-side problem. Check
  the backend address in the admin screen and the team model allowlist
  (`./scripts/manage.sh team sync`).

## Users and keys

Accounts themselves are created in the UI (kchat): sign-up followed by admin
approval, and that API provisions the matching LiteLLM user and per-user key.
`manage.sh` in this repository is the LiteLLM-side view of them.

```bash
./scripts/manage.sh user list                 # LiteLLM users
./scripts/manage.sh user usage --user <email> # this month's spend against budget
./scripts/manage.sh key list --user <email>
```

## Deep research / LDR failures

Symptom: a context-window error from vLLM, `This model's maximum context length
is N tokens`, surfaced as a 400. Nothing trims the request on the way through, so
an input over the serving context fails outright.

```bash
# 1) The serving context of the chat model
KEY=$(grep ^LITELLM_MASTER_KEY .env | cut -d= -f2)
curl -sf -H "Authorization: Bearer $KEY" http://localhost:8080/litellm/v1/model/info \
  | jq '.data[] | select(.model_name=="local/qwen3.6-35b") | {model_name, api_base: .litellm_params.api_base, max_input: .model_info.max_input_tokens}'

# 2) If it is absent, diagnose placement
./scripts/setup.sh scheduler inventory   # per-node vLLM and max_model_len
./scripts/setup.sh scheduler plan        # what context the planner chose

# 3) The MCP itself
docker logs kloudchat-deep-research --tail 50
```

| Cause | Action |
|---|---|
| Chat not placed | The planner could not fit it anywhere — `plan` prints the reason. Add a node or shrink `VLLM_MODELS`, then apply and re-run `gen-litellm-config.sh` |
| Placed but still failing | LDR's accumulated input exceeds the serving context. Reduce `LDR_SEARCH_ITERATIONS` or the result size, or serve chat on a node with a larger context |
| Needs a larger context | Serve chat on a node with a higher `max-model-len` (256K). No separate alias is needed — plain `local/qwen3.6-35b` routes to that node's context |

## file_search returns nothing

Retrieval is opt-in. Without the `index` profile KloudChat falls back to the
lexical search it already has, so empty results are a configuration answer before
they are a fault. See [models.md](models.md#retrieval) for the two stages.

| Cause | Action |
|---|---|
| `index` not in `COMPOSE_PROFILES` | Add it and re-run `setup.sh up` — `index-db` and `index-shim` do not start otherwise |
| No embedding deployment | `INDEX_EMBED_MODELS` is tried in order: `local/bge-m3` needs a vLLM placement, `text-embedding-3-small` needs an OpenAI key. `GET /tools/index/health` reports whether embeddings answer |
| Collection never indexed | The index is derived and starts empty — KloudChat fills it through `PUT /tools/index/documents`. Losing the volume costs a re-index, not a document |
| Results arrive but are weak | The reranker may be missing; the search response carries `"reranked": true\|false`. Without it search falls back to vector order |

## Diagnostic helpers

| Command | Purpose |
|---|---|
| `manage.sh user list` / `team list` / `key list` | LiteLLM users, teams and virtual keys |
| `manage.sh user usage [--user <email>]` | Per-user spend against the monthly budget (used %, reset date) |
| `manage.sh user topup --user <email> --amount <N>` | Temporarily raise the monthly limit by $N, preserving spend so statistics stay accurate. The original limit is recorded in `data/ledger/topups.json` and restored at the monthly reset |
| `manage-vllm.sh status` | vLLM health and models |
| `manage-vllm.sh logs <svc>` | vLLM logs |
| `setup.sh scheduler inventory` | Per-node GPU class, VRAM, running containers |
| `setup.sh scheduler plan` | Current to target placement diff (dry run) |
| `tune-host.sh --check` | Recommended sysctl values against the current ones |

## Operator knobs

| Variable | Default | Purpose |
|---|---|---|
| `KLOUDCHAT_VLLM_WAIT_TIMEOUT` | 1200s | Deadline for vLLM readiness |
| `KLOUDCHAT_VLLM_WAIT_INTERVAL` | 10s | Probe interval |
| `KLOUDCHAT_SKIP_SCHEDULER` | (off) | Skip the placement step in `setup.sh all` |
| `KLOUDCHAT_SERVICE_WAIT` | 180s | How long to wait for capabilities after startup |
| `KLOUDCHAT_REMOTE_DIR` | `KloudChat-LLM` | Repository path on remote nodes (rsync target, and where the placement step runs compose) |

The full list is in the [environment variable reference](env-reference.md).

## Starting over (destructive)

```bash
./scripts/setup.sh clean        # remove containers and ./data (LiteLLM DB, code-interpreter redis/minio)
./scripts/gen-env.sh            # regenerate .env
./scripts/setup.sh all
```

> **This cannot be undone.** Copy the directories below first.

- `./data/litellm/postgres` (LiteLLM), and in the UI repository `./data/postgres`
  and `./data/minio`
- **`./data/ledger`** — issued virtual keys in plaintext (`keys.json`), the topup
  ledger (`topups.json`) and the team cache (`teams.json`)

**Why the ledger is load-bearing**

LiteLLM stores only key hashes. Losing `keys.json` means the plaintext of every
issued key is gone for good.

## See also

- [Environment variables](env-reference.md) — every `.env` key
- [Scheduler](../scheduler/README.md) — multi-node vLLM placement
- [GPU memory](gpu-memory.md) — what fits per node class, and the vLLM tuning knobs
