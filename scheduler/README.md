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
VLLM_MODELS=qwen3.6-35b,glm-4.7-flash
VLLM_MODELS_ROOT=/var/lib/vllm/models
```

Model definitions are in [`models.yaml`](models.yaml), which carries only
identity (`id`, `hf_repo`) and the delegation path (`openrouter`). Everything
else is read from the checkpoint's `config.json` and its size on disk.

## How placement works

1. **Coverage** — one instance of each model at its context floor, largest first,
   onto the node with the most remaining capacity. Seating models at their target
   context first would let a large model claim a node and leave the next one
   nowhere to go.
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

`gpu_util` is an output, not an input: it is `need / node VRAM`, written into the
node's `.env`.

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
| `applier.py` | Node `.env` updates, compose start/stop, and the orchestrator's URLs |

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
