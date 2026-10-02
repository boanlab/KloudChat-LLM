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

`setup.sh all` runs `apply -y` when `NODES_VLLM` is set, unless
`KLOUDCHAT_SKIP_SCHEDULER=1`. Every subcommand accepts `--hosts` and `--models`
(CSV) to override `.env`, and `--replicas N` to cap instances per model (`1`
disables replication).

## Input

```bash
# .env
NODES_VLLM=ops@gpu-1,ops@gpu-2       # SSH targets, head node first
VLLM_MODELS=qwen3.8-27b,qwen3-coder-next,bge-m3
VLLM_MODELS_ROOT=/var/lib/vllm/models
KLOUDCHAT_REMOTE_DIR=KloudChat-LLM   # compose workdir on each node (env or .env)
```

Models are defined in [`models.yaml`](models.yaml). Undeclared fields come from
the checkpoint's `config.json` and its size on disk, read over SSH from the
first answering node that has it. `config.json` is cached under
`~/.cache/kloudchat-scheduler/models/`; the weight size (safetensors, fp32
copies excluded) is measured on every run, with an analytic estimate when the
measurement fails. A model whose `config.json` no node can read, or that declares
no context length, is delegated.

## Probing

Per node, over SSH, each step tolerant of the others failing: `docker ps`,
`nvidia-smi` (card name, count, memory per card; `/proc/meminfo` on
unified-memory nodes), `uname -m`, GPU memory held by processes outside this
stack, and the checkpoint directories (those with a `config.json`) under
`VLLM_MODELS_ROOT`. A node answering nothing is retried once, then reported as
`no answer` and left out of placement.

The card name classifies the node as `gb10`, `rtx5090`, `pro5000`, `pro6000` or
`unsupported` (the same vocabulary as `lib.sh::detect_gpu_class`). A mixed box
is sized by its smallest card and says so in `inventory`.

## Node roles

The first target in `NODES_VLLM` is the **head** node; every other target is the
**pool**. `placement:` in models.yaml restricts a model to one of them before any
packing:

| `placement` | Nodes | Models |
|---|---|---|
| `head` | the head node only | embeddings, reranking, transcription |
| `pool` | every node but the head | the coder |
| unset | any | chat |

A pool model with no pool seat is delegated to OpenRouter rather than placed on
the head node. A single-node cluster has no pool, and `placement` is ignored.

## Placement

0. **Unsupported cards** — a node classified `unsupported` holds nothing; the
   plan notes it and the node drops out of every step below.
1. **Coverage** — one instance of each model at its context floor, ordered by
   `priority` (highest first, ties largest first), onto the eligible node with
   the most free capacity. Capacity differences under 1 GiB do not decide; within
   that band the node already running the model (per `docker ps`) wins, then the
   lowest node id.
2. **Restoration** — remaining capacity on each card doubles contexts toward
   their targets, furthest-from-target first.
3. **Replication** — remaining capacity takes extra instances at the context
   floor, one per round to the model furthest below its `share:`
   (`instances / share`), then restoration runs again for them. Shares are
   weights among models competing for the same nodes.

A node is eligible for a model when `placement` allows it, its architecture is
in the model's `arches` (unrestricted by default), it holds the checkpoint, and
it has at least `tensor_parallel` cards. A model with no seat is delegated to
OpenRouter with the reason: no eligible node answering, architecture, missing
checkpoint, too few cards, or capacity.

## Sizing

```
need(model, context) = weights + activation + KV(context)
KV = bytes/token × context × concurrent sessions × 1.10
```

- `activation` is 10 GiB for a generate runner and 2 GiB for a pooling one,
  capped at 12% of the card (never below 1 GiB). See
  [gpu-memory.md](../docs/gpu-memory.md).
- Card capacity: on a discrete card, VRAM less a reserve of 8% clamped to
  1–8 GiB; on a unified-memory node (GB10), system RAM less 12 GiB. GPU memory
  held by processes outside this stack is subtracted too. A multi-card node's
  capacity is split evenly across its cards.
- KV bytes per token: `2 · L_kv · H · d · β` (MHA/GQA) or `L · latent · β`
  (MLA), over the full-attention layers only, at FP8. Sliding-window layers are
  charged a flat amount per sequence.
- `concurrent_sessions` is a sizing assumption: on a card that cannot hold the
  declared width it is halved down to fit, and the plan says so. The context
  floor is never traded away.
- `gpu_util` is an output: per-card need over the card's total VRAM, rounded up
  to two decimals and clamped to 0.05–0.95.

### Tensor parallelism

`tensor_parallel: N` shards a model across N cards of one node
(`--tensor-parallel-size`); cross-node sharding is not supported.

```
per card = weights/N + activation + KV/kv_shards
node     = per card × N
```

Activation is paid per rank. KV shards by head: `kv_shards = min(N, kv_heads)`,
and 1 for MLA. A sharded model takes a slice of each card it spans (emptiest
cards first); the applier writes the device ordinals as `{env_prefix}_DEVICES`.

## Output

`plan` prints each placement (node, model, context, `gpu_util`, GiB charged,
and the cards and TP when not the single-card default), the models delegated
to OpenRouter with their reasons, and notes.

`apply` prints the plan, then the actions it would take, and asks for
confirmation unless `-y`. Per node it:

- writes `{env_prefix}_MAX_LEN` and `{env_prefix}_GPU_UTIL` for each placed
  model into `KLOUDCHAT_REMOTE_DIR/.env`; `{env_prefix}_TP` only when above 1
  or to reset a sharded node; `{env_prefix}_DEVICES` only on
  multi-card nodes or to clear a stale value;
- starts (`compose up -d`) services the plan adds, stops `vllm-*` services it
  drops, and recreates (`--force-recreate`) a running service only where one of
  its options changed. Containers outside the `vllm-*` prefix are never touched.

In the orchestrator's `.env` it writes `{env_prefix}_URL` (a CSV of
`http://host:port`) for every model in models.yaml, empty for models not
placed, and `WHISPER_URLS` in place of the transcription model's `_URL`. Only
changed keys are written. Re-applying an unchanged plan does nothing. A failing
node does not stop the others; any failure exits 1.

## Layout

| File | Role |
|---|---|
| `models.yaml` | Model definitions |
| `registry.py` | YAML loader; declared and derived values into `ModelSpec` |
| `inventory.py` | SSH probing: GPU class, VRAM, architecture, running services, foreign memory, checkpoints |
| `model_metadata.py` | `config.json` to layer count, KV heads, dtype, native context; weight size on disk |
| `kv_model.py` | KV bytes per token and per sequence |
| `types.py` | `ModelMetadata`, `NodeSpec`, node reserve |
| `planner.py` | Placement decision |
| `applier.py` | Node `.env` updates, compose start/stop/recreate, orchestrator URLs |
| `__main__.py` | CLI; reads `.env`, binds metadata, prints and applies the plan |

## Tests

```bash
pytest scheduler/tests -q
PYTHONPATH=. python3 scheduler/tests/test_scheduler.py   # without pytest
```

No GPU, network or Docker needed.
