# scheduler

Places the models in `VLLM_MODELS` across the GPU nodes in `NODES_VLLM`, writes
the result into each node's `.env` and the orchestrator's `VLLM_*_URL`, and
starts, stops or recreates the compose services accordingly.

```bash
python3 -m scheduler inventory     # probe results per node
python3 -m scheduler plan          # compute placement (changes nothing)
python3 -m scheduler apply         # apply it, after confirmation
python3 -m scheduler apply -y      # apply without confirmation
```

`setup.sh all` runs `apply -y` unless `KLOUDCHAT_SKIP_SCHEDULER=1`. Every
subcommand accepts `--hosts` and `--models` (CSV) to override `.env`, and
`--replicas N` to cap instances per model (`1` disables replication).

## Input

```bash
# .env
NODES_VLLM=ops@gpu-1,ops@gpu-2       # SSH targets, head node first
VLLM_MODELS=qwen3.6-35b,qwen3.5-122b-a10b,bge-m3
VLLM_MODELS_ROOT=/var/lib/vllm/models
```

Models are defined in [`models.yaml`](models.yaml). Undeclared fields come from
the checkpoint's `config.json` and its size on disk, read over SSH from the
first node that has it; `config.json` is cached under
`~/.cache/kloudchat-scheduler/models/`.

## Node roles

The first target in `NODES_VLLM` is the **head** node; every other target is the
**pool**. `placement:` in models.yaml restricts a model to one of them before any
packing:

| `placement` | Nodes | Models |
|---|---|---|
| `head` | the head node only | default chat, embeddings, reranking, transcription |
| `pool` | every node but the head | card-sized picker models |
| unset | any | — |

A pool model with no pool seat is delegated to OpenRouter rather than placed on
the head node. A single-node cluster has no pool, and `placement` is ignored.

## Placement

1. **Coverage** — one instance of each model at its context floor, ordered by
   `priority` (highest first, ties largest first), onto the eligible node with
   the most free capacity. Capacity differences under 1 GiB do not decide; within
   that band the node already running the model wins, then the lowest node id.
2. **Restoration** — remaining capacity on each card raises contexts toward
   their targets, furthest-from-target first.
3. **Replication** — remaining capacity takes extra instances, one per round to
   the model furthest below its `share:` (`instances / share`). Shares are
   weights among models competing for the same nodes.

Only nodes that hold the model's checkpoint (a directory with `config.json`
under `VLLM_MODELS_ROOT`) are candidates. A model with no seat is delegated to
OpenRouter with the reason: capacity, missing checkpoint, unsupported
architecture, or too few cards for its tensor parallelism.

## Sizing

```
need(model, context) = weights + activation + KV(context)
KV = bytes/token × context × concurrent sessions × 1.10
```

- `activation` is 10 GiB for a generate runner and 2 GiB for a pooling one,
  capped at 12% of the card. The node reserve is 8% of the card, clamped to
  1–8 GiB (12 GiB on unified-memory nodes). See
  [gpu-memory.md](../docs/gpu-memory.md).
- KV bytes per token: `2 · L_kv · H · d · β` (MHA/GQA) or `L · latent · β`
  (MLA), over the full-attention layers only. Sliding-window layers are charged a
  flat amount per sequence.
- `concurrent_sessions` is a sizing assumption: on a card that cannot hold the
  declared width it is halved down to fit, and the plan says so. The context
  floor is never traded away.
- `gpu_util` is an output: `need / card VRAM`, a fraction of **one card**.

### Tensor parallelism

`tensor_parallel: N` shards a model across N cards of one node
(`--tensor-parallel-size`); cross-node sharding is not supported.

```
per card = weights/N + activation + KV/kv_shards
node     = per card × N
```

Activation is paid per rank. KV shards by head: `kv_shards = min(N, kv_heads)`,
and 1 for MLA. A sharded model takes a slice of each card it spans; the applier
writes the device ordinals as `{env_prefix}_DEVICES`.

## Applying

Per placement the applier writes `{env_prefix}_MAX_LEN` and `_GPU_UTIL` (and
`_TP`, `_DEVICES` where they differ from the defaults) into the node's `.env`,
starts services the plan adds, stops `vllm-*` services it drops, and recreates a
running service only where an option changed. In the orchestrator's `.env` it
writes `{env_prefix}_URL` for every model in the catalogue — empty for models not
placed — and `WHISPER_URLS` for the transcription model.

## Layout

| File | Role |
|---|---|
| `models.yaml` | Model definitions |
| `registry.py` | YAML loader; declared and derived values into `ModelSpec` |
| `inventory.py` | SSH probing: GPU class, VRAM, architecture, running services, checkpoints |
| `model_metadata.py` | `config.json` to layer count, KV heads, dtype, native context |
| `kv_model.py` | KV bytes per token |
| `planner.py` | Placement decision |
| `applier.py` | Node `.env` updates, compose start/stop/recreate, orchestrator URLs |

## Tests

```bash
pytest scheduler/tests -q
PYTHONPATH=. python3 scheduler/tests/test_scheduler.py   # without pytest
```

No GPU, network or Docker needed.
