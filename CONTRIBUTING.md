# Contributing

Thanks for taking the time. This document covers how the repository is laid out,
how to run what you changed, and the conventions the codebase holds itself to.

## Repository layout

| Path | Contents |
|---|---|
| `docker-compose.yml` | The backend stack: gateway, tools, LiteLLM. Composed by profiles. |
| `docker-compose.vllm.yml` | What a GPU node serves: vLLM models plus the transcription backend. |
| `docs/` | Operator documentation. |
| `scheduler/` | Python package that decides which model runs on which node. |
| `scripts/` | Setup, config generation, node installation, day-2 operations. |
| `services/` | One directory per service: Dockerfile, source, config templates. |
| `.github/workflows/` | CI (shell, scheduler, compose, docs) and Docker Hub publishing. |

Nothing outside `scripts/` writes to `.env`, and nothing outside
`scripts/gen-*-config.sh` writes generated service configs. Keeping those two
rules makes a broken deployment traceable to a single writer.

## Development setup

You do not need a GPU to work on most of this repository.

```bash
./scripts/gen-env.sh          # creates .env with generated secrets
python3 -m pip install pyyaml # the scheduler's only runtime dependency
```

Without an `OPENROUTER_API_KEY` or a reachable vLLM node, `setup.sh` stops early
by design — that check lives in `step_env_validate`.

## Running the checks

CI runs exactly these. Run them before opening a pull request:

```bash
pytest scheduler/tests -q                  # or: PYTHONPATH=. python3 scheduler/tests/test_scheduler.py
ruff check scheduler services              # rules pinned in pyproject.toml
shellcheck -S warning -e SC1091 scripts/*.sh   # settings in .shellcheckrc; bash -n also works
docker compose -f docker-compose.yml config --quiet
docker compose -f docker-compose.vllm.yml config --quiet
```

No shellcheck installed? `docker run --rm -v "$PWD:/mnt" -w /mnt koalaman/shellcheck:stable
-S warning -e SC1091 scripts/*.sh` runs the same check.

The scheduler tests need no GPU, no network, and no Docker: they exercise the
memory arithmetic and the placement policy against synthetic nodes.

## Conventions

**Comments explain why, not what.** The codebase is dense with constants that
look arbitrary and are not — a 12 GiB reservation, `max_num_seqs = 128`, a
specific attention backend. Every one of those carries the failure that produced
it. When you change such a value, move its explanation with it; when you add one,
write down what happens if it is wrong in either direction.

**Measured values say so.** Weights, KV sizes and throughput numbers in the docs
are measurements from a real cluster, and they are labelled as such. If you copy a
number from a model card or a blog post, mark it as an estimate.

**Prices are load-bearing.** The per-token figures in `scripts/lib.sh` and
`scripts/gen-litellm-config.sh` drive user billing. Verify them against
`https://openrouter.ai/api/v1/models` — the live catalogue, not documentation —
and update `docs/models.md` in the same commit.

**Shell scripts are the operator interface.** They run under `set -euo pipefail`,
print progress through the `hdr`/`info`/`ok`/`warn`/`err` helpers in
`scripts/lib.sh`, and send every diagnostic to stderr so that command
substitution stays clean. A script invoked with no arguments prints its usage.

**Documentation is part of the change.** A flag, environment variable or default
that moves without its documentation moving is an incomplete change. `docs/` is
written for an operator who is looking at a broken deployment, so prefer the
concrete command over the general principle.

## Publishing images

`.github/workflows/publish-images.yml` owns the five `boanlab/kloudchat-*`
images. What it builds depends on how it was triggered:

| Trigger | Builds | Tags |
|---|---|---|
| Push to `main` | Only images whose `services/<name>/` directory changed | `latest` |
| Tag `v*` | All of them | `v1.2.3`, `1.2.3`, `1.2`, `latest` — one build, one digest |
| Manual run | All, or one chosen image | The tag you type, plus `latest` |

Selection compares the pushed range with each image's build context, so a change
under `services/whisper-shim/` rebuilds that image alone. A change outside
`services/` — docs, scheduler, scripts — publishes nothing.

Adding an image means one entry in the `catalogue` in the `select` job; the
context path is also the path watched for changes.

## Commit and pull request style

Short imperative subject, body explaining the reasoning:

```
scheduler: keep resident services out of the stop set

docker ps returns every container on a node, including ones the planner
never placed. Stopping those took the node's STT backend down.
```

Pull requests should say how the change was verified. For anything that only
runs on GPU hardware, say plainly that it was not verified there — an untested
claim is worse than an acknowledged gap.

## Reporting bugs

Open an issue with the command you ran, the output you got, and your OS,
architecture and GPU. Redact credentials: `.env` contains live LiteLLM and
OpenRouter keys, and they leak easily through pasted logs.

For security issues, follow [SECURITY.md](SECURITY.md) instead — do not open a
public issue.
