# scheduler

Places the models in `VLLM_MODELS` across the GPU nodes in `VLLM_NODES`, writes
the result into each node's `.env` and the orchestrator's `VLLM_PLACED`, and
starts, stops or recreates the compose services accordingly.

```bash
python3 -m scheduler inventory     # probe results per node
python3 -m scheduler plan          # compute placement (changes nothing)
python3 -m scheduler apply         # apply it, after confirmation
python3 -m scheduler apply -y      # apply without confirmation
```

`setup.sh all` runs `apply -y` when `VLLM_NODES` is set, unless
`KLOUDCHAT_SKIP_SCHEDULER=1`; `setup.sh scheduler <subcommand>` forwards to the
module. Every subcommand accepts `--hosts` and `--models` (CSV) to override
`.env`. Per model, [`models.yaml`](models.yaml) sets `replicas:` (instance cap; `1`
disables replication), `placement:` (see Node roles) and `manual: true` (hand-run,
never placed).

## Input

```bash
# .env
VLLM_NODES=ops@gpu-1,ops@gpu-2       # SSH targets, head node first
VLLM_NODES_HOLD=ops@gpu-3            # targets in VLLM_NODES left alone (see Held nodes)
VLLM_MODELS=qwen3.8-27b,qwen3.8-flash,bge-m3
VLLM_MODELS_ROOT=/var/lib/vllm/models
KLOUDCHAT_REMOTE_DIR=KloudChat-LLM   # compose workdir on each node (env or .env)
VLLM_ENDPOINTS=http://gpu-9:8001     # servers you run yourself; counted as instances
WHISPER_URLS=http://gpu-9:9000       # hand-run transcription backends; never written
VLLM_PLACED=...                      # written by `apply`; do not edit
IMAGEGEN_PLACED=...                  # picture models, written by `apply`; do not edit
```

`VLLM_ENDPOINTS` entries (`id=url`, or a bare URL asked at `/v1/models`) count as
instances of their model, transcription excepted (hand-run, it goes through
`WHISPER_URLS`). A model with one is not placed for coverage and is not
delegated, and replication and its `replicas` cap see the total. `apply` stops a
scheduler-placed instance of such a model, but never the service an entry
points at: an entry whose host is a node in `VLLM_NODES` and whose port is a
catalogued service's keeps that container running.

Models are defined in [`models.yaml`](models.yaml). Undeclared fields come from
the checkpoint's `config.json` and its size on disk, read over SSH from the
first answering node that has it. `config.json` is cached under
`~/.cache/kloudchat-scheduler/models/`; the weight size (safetensors, fp32
copies excluded) is measured on every run, with an analytic estimate when the
measurement fails. A model whose `config.json` no node can read, or that declares
no context length, is delegated.

### Site overrides

`scheduler/site.yaml`, when present (untracked, one per deployment), lays
fields over the catalogue entries by id without editing `models.yaml`. It sets
where and how large a model runs, never what it is or where it answers:
`placement`, `replicas`, `share`, `priority`, `ctx_floor`, `ctx_target`,
`concurrent_sessions`, `tensor_parallel`, `activation`, `weight`, `arches`,
`gpu_util`, `nodes` and `cards`. Any other field, or an id the catalogue lacks,
is an error. `nodes` limits a model to the nodes named by id or host and
`cards` to the CUDA ordinals it may take on one.

`gpu_util` fixes `--gpu-memory-utilization`: the planner charges that share of
the card as is instead of sizing one, narrows `concurrent_sessions` to what the
share holds, keeps the context at `ctx_floor`, and notes a share too small for
the floor. A layout that already runs is written down this way so that
`apply` finds nothing to change:

```yaml
# scheduler/site.yaml: 27B, retrieval and transcription on the head node,
# Flash on each of three pool nodes
models:
  - id: qwen3.8-27b
    placement: head
    replicas: 1
    ctx_floor: 131072
    ctx_target: 131072
    gpu_util: 0.66
  - id: qwen3.8-flash
    replicas: 3
    gpu_util: 0.92
  - id: bge-m3
    replicas: 1
    gpu_util: 0.05
  - id: bge-reranker-v2-m3
    replicas: 1
    gpu_util: 0.05
  - id: whisper-large-v3
    replicas: 1
    gpu_util: 0.14
```

The catalogue keeps the 27B, retrieval and transcription to one instance
(`replicas: 1`) and leaves Flash's `replicas` unlimited, so every pool node
takes a Flash. A site file only pins what a deployment runs differently, such
as the context or a fixed `gpu_util`.

### Node ids

Each SSH target gets a short id: the last octet of an IPv4 address, otherwise
the first hostname label (`ops@gpu-1.lan` is `gpu-1`). Two targets that would
share a short id (`10.1.0.11` and `10.2.0.11`) are identified by their bare
host instead (`10.1.0.11`, `10.2.0.11`); a target listed twice is an error and
the command exits. Ids name nodes in every printout and action.

## Probing

Per node, over SSH, each step tolerant of the others failing: `docker ps`,
`nvidia-smi` (card name, count, memory per card; `/proc/meminfo` on
unified-memory nodes), `uname -m`, GPU memory held per card by processes outside
this stack, and the checkpoint directories (those with a `config.json`) under
`VLLM_MODELS_ROOT`. Each SSH call is retried while the connection itself fails
(exit 255). A node counts as answering only when both `docker ps` and its card
memory were read: planned on half an answer it would look empty (its services
started twice) or full (its services stopped). A node that does not answer in
full is retried once, then reported as `no answer` and left out of placement and
of every action. `/proc/meminfo` stands in for card memory only when
`nvidia-smi` ran and reported none, and only for a GB10 (or a card whose name
could not be read), so a discrete node whose `nvidia-smi` call failed is never
sized by its system RAM.

The card name classifies the node as `gb10`, `rtx5090`, `pro5000`, `pro6000` or
`unsupported` (the same vocabulary as `lib.sh::detect_gpu_class`; `unknown`
when the name cannot be read). A mixed box is sized by its smallest card and
says so in `inventory`.

**Foreign memory.** Every compute process `nvidia-smi` lists is attributed to
its Docker container by cgroup. Memory held by a `vllm-*` container is this
stack's and stays available to the plan; everything else (other containers,
host processes) is foreign and charged to the card that holds it. `inventory`'s
`USABLE` column is node capacity after all foreign memory.

## Node roles

The first target in `VLLM_NODES` is the **head** node; every other target is the
**pool**. `placement:` in models.yaml restricts a model to one of them before any
packing:

| `placement` | Nodes | Models |
|---|---|---|
| `head` | the head node only | the 27B |
| `head-first` | the first instance on the head node, extra instances on any node | embeddings, reranking, transcription |
| `pool` | every node but the head | Flash-Next |
| unset | any | chat |

A pool model with no pool seat is delegated to OpenRouter rather than placed on
the head node. A single-node cluster has no pool, and `placement` is ignored.
A `head-first` model's extra instances (replication, `replicas`) go to any node, the
head included.

### Image models

`z-image-turbo`, `qwen-image-2512` and `qwen-image-edit-2511` (`runner: image`,
compose services `omni-*` in `docker-compose.vllm-omni.yml`) are placed like any
model when `VLLM_MODELS` lists them, with two differences. They are sized by the
declared `weight` plus `activation` on one card (no context, KV or
`gpu_util`), and `apply` writes only `{env_prefix}_DEVICES` to the node's `.env`.
Where they answer goes to `IMAGEGEN_PLACED` in the orchestrator's `.env`
(`IMAGEGEN_BACKENDS` stays the hand-run list). They have no OpenRouter route:
without a seat `plan` prints them under "Not placed". A site file pins them with
`nodes` (node ids or hosts) and `cards` (CUDA ordinals), for example:

```yaml
models:
  - id: qwen-image-edit-2511
    nodes: [7]
    cards: [1]
  - id: qwen-image-2512
    nodes: [7]
    cards: [0]
  - id: z-image-turbo
    nodes: [7]
    cards: [0]
```

`qwen3-tts` (`runner: speech`, `omni-qwen3-tts`, port 8094) is sized and pinned the
same way (9.5 GiB, no KV), with one difference in routing: it has no shim, so its
URL goes to `VLLM_PLACED` and LiteLLM calls `/v1/audio/speech` on the node itself.

Containers of image models not in `VLLM_MODELS` are neither stopped nor started:
their memory counts as foreign.

A node's options are read from `KLOUDCHAT_REMOTE_DIR/.env` like any other. A node
that runs its omni containers from another directory with its own `.env` has no
`VLLM_OMNI_*_DEVICES` there, so the first `apply` writes them and recreates those
containers (a picture model takes minutes to come back); check `plan` and the
action list first.

### Held nodes

A target in `VLLM_NODES_HOLD` is probed and shown by `inventory` as `held`, and
nothing else: no placement, no start, stop or recreate, no `.env` write, and
`setup.sh vllm` installs nothing there. It is for a node in `VLLM_NODES` that
runs something else, such as hand-run image models: left in the pool, its spare
room would draw replicas, and while its omni containers are down it would look
empty. An entry is the target as written in `VLLM_NODES`, its host, or its node
id; one that `VLLM_NODES` does not list, or
the head node, is an error. Listing the node in `VLLM_NODES` at all keeps it in
metrics-shim's node list and in the `inventory`.

## Placement

0. **Unsupported cards** — a node classified `unsupported` holds nothing; the
   plan notes it and the node drops out of every step below. With no node left,
   every model is delegated.
1. **Coverage** — one instance of each model without an external instance (see
   Input) at its context floor, models pinned by `nodes` or `cards` first, then
   by `priority` (highest first, ties largest first), onto the eligible node with
   the most free memory summed over its cards. Differences under 1 GiB do not
   decide; within that band the node already running the model (per
   `docker ps`) wins, then the lowest node id. On the node, the model takes the
   emptiest card(s) that hold its per-card need.
2. **Restoration** — free memory on a placement's own card(s) doubles its
   context toward the target, furthest-from-target first, until nothing grows.
3. **Replication** — remaining capacity takes extra instances at the context
   floor, one per round to the model furthest below its `share:`
   (`instances / share`, ties by priority; external instances count), on a node
   not already running it; then restoration runs again. Shares are weights among models competing for
   the same nodes.

A node is eligible for a model when `placement` allows it, its architecture is
in the model's `arches` (unrestricted by default), it holds the checkpoint, and
it has at least `tensor_parallel` cards. A model with no seat is delegated to
OpenRouter with the reason: no eligible node answering, architecture, missing
checkpoint, too few cards, or capacity.

## Sizing

```
need(model, context) = weights + activation + KV(context)
KV = (bytes/token × context + sliding bytes/sequence) × concurrent sessions × 1.10
```

- `activation` is 10 GiB for a generate runner and 2 GiB for a pooling one,
  capped at 12% of the card (never below 1 GiB). See
  [gpu-memory.md](../docs/gpu-memory.md).
- Node capacity: on discrete cards, `cards × (VRAM − reserve)`, VRAM being the
  smallest card's and the reserve 8% of it clamped to 1–8 GiB; on a unified-memory node (GB10),
  system RAM less 12 GiB. The node capacity is split evenly across its cards;
  each card's free memory is its share less the foreign memory held on that
  card.
- KV bytes per token: `2 · L_kv · H · d · β` (MHA/GQA) or `L · latent · β`
  (MLA), over the full-attention layers only, at FP8. Sliding-window layers are
  charged a flat amount per sequence.
- `concurrent_sessions` is a sizing assumption: on a card that cannot hold the
  declared width it is halved down to fit, and the plan says so. The context
  floor is never traded away.
- `gpu_util`, unless declared, is per-card need over the card's total memory,
  rounded up to two decimals and clamped to 0.05–0.95.

### Tensor parallelism

`tensor_parallel: N` shards a model across N cards of one node
(`--tensor-parallel-size`); cross-node sharding is not supported.

```
per card = weights/N + activation + KV/kv_shards
node     = per card × N
```

Activation is paid per rank. KV shards by head: `kv_shards = min(N, kv_heads)`,
and 1 for MLA. A sharded model takes a slice of each card it spans (emptiest
cards first); the applier writes the device ordinals as `{env_prefix}_DEVICES`,
which compose reserves as the container's `device_ids`.

## Output

`plan` prints each placement (node, model, context, `gpu_util`, GiB charged,
and the cards and TP when not the single-card default), the models delegated
to OpenRouter with their reasons, and notes.

`apply` prints the plan, then the actions it would take, and asks for
confirmation unless `-y`. Per node it:

- writes `{env_prefix}_MAX_LEN` and `{env_prefix}_GPU_UTIL` for each placed
  model into `KLOUDCHAT_REMOTE_DIR/.env`; `{env_prefix}_TP` only when above 1
  or to reset a sharded node; `{env_prefix}_DEVICES` only on multi-card nodes
  or to overwrite a stale value;
- starts (`compose up -d`) services the plan adds, stops `vllm-*` services it
  drops, and recreates (`--force-recreate`) a running service only where one of
  its options changed. Containers outside the `vllm-*` prefix are never touched.

The scheduler knows a service by its container name (`docker ps`), not by
compose labels, so a container started by `docker run` under the service's name
counts as that service. Acting on one takes care of the difference: before a
start or recreate, a same-named container outside the `kloudchat-vllm` compose
project is removed (it would block `compose up` on the name, and a recreate
replaces it anyway), and a stop also runs `docker stop`, since `compose stop`
skips a container it did not create. An unchanged plan leaves such a container
running untouched; the first recreate brings it under compose.

`apply` reads each node's `.env` before deciding what changed; when one cannot
be read it applies nothing rather than take every option as changed.

In the orchestrator's `.env` it writes `VLLM_PLACED` (a CSV of
`id=http://host:port`, one per instance, the transcription model included) and
`IMAGEGEN_PLACED` (the same for picture models).
`VLLM_ENDPOINTS`, `WHISPER_URLS` and `IMAGEGEN_BACKENDS`, the servers you run by
hand, are never written. Only changed keys are written. Re-applying an unchanged plan does nothing.

`python3 -m scheduler catalog` prints the catalogue as TSV (id, api, env prefix,
OpenRouter slug, timeout, max in flight, presence penalty, temperature) for the shell scripts.

**On failure.** Actions run node by node in order. The first failing action on a
node skips that node's remaining actions (an env write precedes the recreate
that reads it); the other nodes continue. The orchestrator `.env` is still
written, with the URL of every service whose start or recreate failed or was
skipped left out of the values being written. Every failed and skipped action
is printed and the command exits 1.

## Layout

| File | Role |
|---|---|
| `models.yaml` | Model definitions |
| `registry.py` | YAML loader; declared and derived values into `ModelSpec` |
| `inventory.py` | SSH probing: GPU class, VRAM, architecture, running services, foreign memory per card, checkpoints; node ids |
| `model_metadata.py` | `config.json` to layer count, KV heads, dtype, native context; weight size on disk |
| `endpoints.py` | `VLLM_ENDPOINTS` to instance counts per model |
| `kv_model.py` | KV bytes per token and per sequence |
| `types.py` | `ModelMetadata`, `NodeSpec` (reserve, per-card budget and free memory) |
| `planner.py` | Placement decision |
| `applier.py` | Node `.env` updates, compose start/stop/recreate, orchestrator URLs, failure handling |
| `__main__.py` | CLI; reads `.env`, resolves node ids, binds metadata, prints and applies the plan |

## Tests

```bash
python3 -m pytest -q scheduler
```

No GPU, network or Docker needed.
