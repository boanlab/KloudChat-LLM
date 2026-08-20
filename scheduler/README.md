# scheduler

Places the models listed in `.env` across the available GPU nodes, spreading them
as evenly as capacity allows.

```bash
python3 -m scheduler inventory     # probe results per node
python3 -m scheduler plan          # compute placement (changes nothing)
python3 -m scheduler apply         # apply it (after confirmation)
python3 -m scheduler apply -y      # apply without confirmation
```

## Input

```bash
# .env
NODES_VLLM=ops@gpu-1,ops@gpu-2       # SSH targets
VLLM_MODELS=qwen3.5-122b-a10b,qwen3.6-35b,bge-m3
VLLM_MODELS_ROOT=/var/lib/vllm/models
```

Model definitions are in [`models.yaml`](models.yaml), which carries only
identity (`id`, `hf_repo`) and the delegation path (`openrouter`). Everything
else is read from the checkpoint's `config.json` and its size on disk.

## How placement works

1. **Coverage** — one instance of each model at its context floor, onto the node
   with the most remaining capacity. Seating models at their target context first
   would let a large model claim a node and leave the next one nowhere to go.

   Order is `priority` first, then largest. Largest-first is only a starvation
   guard — seat the small models first and a big one finds every node partly
   used — so it says nothing about which model the cluster would rather keep. A
   `priority:` in models.yaml does say that, and outranks it; within one level
   the guard still applies.

   Capacity differences under `CAPACITY_TIE_BYTES` (1 GiB) do not decide
   anything: identical cards reported usable capacity kilobytes apart, and under
   a strict maximum that noise was enough to migrate a 78 GiB model on a re-run.
   Within that band a node already running the model wins, then the node id — so
   the same inputs give the same plan, which they did not before (nodes arrive in
   probe-completion order).
2. **Restoration** — the remaining capacity is divided among the models on that
   node, raising contexts toward their targets. The placement furthest from its
   target goes first, so one model cannot take everything.
3. **Replication** — only when `--replicas N` is given, and only after every
   model has one instance.

A model that finds no seat is delegated to OpenRouter with the reason attached.
Insufficient capacity and unsupported architecture are reported separately —
more VRAM does not fix the latter.

## Sizing

```
need(model, context) = weights + activation + KV(context)
KV = KV bytes/token × context × concurrent sessions × admission margin (1.10)
```

`activation` is 10 GiB for a generate runner and 2 GiB for a pooling one —
measured, not assumed: subtract weights and the KV cache vLLM reports at startup
from the budget it was given and ~9 GiB of activation buffers, CUDA-graph capture
and hybrid conv state is what remains. It carried 4 GiB, and the shortfall came
straight out of KV (see [gpu-memory.md](../docs/gpu-memory.md)).

`gpu_util` is an output, not an input: it is `need / card VRAM`, written into the
node's `.env` — a fraction of **one card**, which is what vLLM's flag means.

`activation` and the node's reserve are both scaled to the card rather than fixed:
10 GiB of runtime headroom and an 8 GiB reserve are 11% and 8% of a 96 GiB card
but 42% and 33% of a 24 GiB one, and charging the large-card figures to a small
card had the planner reject hardware that works. See `ACTIVATION_MAX_FRACTION`
and `types.default_reserve_bytes`.

**A placement can be narrower than the model asks for.** `concurrent_sessions` is
a sizing assumption — how much KV to hold so several conversations run at once —
so where a card cannot take the declared width the planner halves it down toward
one and says so in a note, rather than delegating a model the card can serve.
The context floor is not traded away the same way: that one is a capability
claim, and a model quietly seated under it is worse than one honestly absent.

### Tensor parallelism

`tensor_parallel: N` in models.yaml shards a model across N cards of one node
(`--tensor-parallel-size`). Cross-node sharding is not supported and is not
planned: it needs Ray, and the interconnect between these hosts is not the one
that makes it worth having.

```
per card = weights/N + activation + KV/kv_shards
node     = per card × N
```

Three things that are easy to get wrong, and that the planner encodes:

- **Activation does not divide.** It is paid per rank, so a sharded model costs
  the node *more* in total than the same model unsharded. What TP buys is fitting
  under a per-card ceiling, not using less memory.
- **KV shards by head, not by rank.** `kv_shards = min(N, kv_heads)`; four ranks
  over two KV heads halves the cache rather than quartering it. MLA replicates
  its single latent on every rank, so `kv_shards` is 1 and the node-wide KV cost
  rises linearly with N.
- **Placement is per card, and a sharded model takes a slice of each card it
  spans.** The node used to be packed as one byte pool, which could not tell "two
  models, one card each" from "two models, both on card 0" — and it produced the
  latter, at 0.86 of a device each. Each card now carries its own free figure,
  the plan records the device ordinals, and the applier writes them as
  `{env_prefix}_DEVICES` for the container's `CUDA_VISIBLE_DEVICES`.

The case it exists for: 78 GiB of `qwen3.5-122b-a10b` weights do not fit a 96 GB
card at all. At TP 2 each rank holds 39 GiB and the model runs at its native 256K
(~69 GiB per card at 12 sessions).

KV bytes per token is `2 · L_kv · H · d · β`; for MLA models, which cache one
compressed latent per layer, it is `L · (kv_lora_rank + rope) · β`. Confusing the
two is wrong by an order of magnitude and rejects placements that would have fit.
`L_kv` is the number of full-attention layers, not the total — hybrid models
carry KV on only some of them.

## Layout

| File | Role |
|---|---|
| `models.yaml` | Deployable model definitions |
| `registry.py` | YAML loader — merges declared and derived values into `ModelSpec` |
| `inventory.py` | SSH probing: GPU class, VRAM, architecture, running services |
| `model_metadata.py` | `config.json` to layer count, KV heads, dtype, native context |
| `kv_model.py` | KV bytes per token |
| `planner.py` | The placement decision |
| `applier.py` | Node `.env` updates, compose start/stop, and the orchestrator's URLs — including clearing the URL of a model dropped from `VLLM_MODELS` |

**Applying twice is a no-op.** The node's current `.env` is read before the diff
is built, and a service is force-recreated only where an option actually moved. A
recreate is a full weight reload — twenty minutes for `qwen3.5-122b-a10b` — and
`setup.sh all` runs `apply` every time.

## Manual operation

Setting `KLOUDCHAT_SKIP_SCHEDULER=1` in `.env` makes `setup.sh` skip this step.
You then manage `VLLM_*_URL` and `VLLM_*_{MAX_LEN,GPU_UTIL}` yourself.

## Tests

```bash
pytest scheduler/tests -q
# or, with no pytest installed:
PYTHONPATH=. python3 scheduler/tests/test_scheduler.py
```

They need no GPU, no network and no Docker: the memory arithmetic and the
placement policy run against synthetic nodes.
