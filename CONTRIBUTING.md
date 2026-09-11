# Contributing

How the repository is laid out, how to run what you changed, and the
conventions the codebase holds itself to.

## Repository layout

| Path | Contents |
|---|---|
| `docker-compose.yml` | The backend stack: gateway, tools, LiteLLM. Composed by profiles |
| `docker-compose.vllm.yml` | What a GPU node serves: every vLLM model, transcription included |
| `docs/` | Operator documentation |
| `scheduler/` | Python package that decides which model runs on which node |
| `scripts/` | Setup, config generation, node installation, day-2 operations |
| `services/` | One directory per service: Dockerfile, source, config templates |
| `.github/workflows/` | CI and Docker Hub publishing |

Nothing outside `scripts/` and `scheduler/applier.py` writes to `.env`, and
nothing outside `scripts/gen-*-config.sh` writes generated service configs.

## Development setup

No GPU is needed for most of this repository.

```bash
./scripts/gen-env.sh                  # .env with generated secrets
python3 -m pip install pyyaml pytest ruff
```

Without an `OPENROUTER_API_KEY` or a reachable vLLM node, `setup.sh` stops
early by design (`step_env_validate`).

## Running the checks

CI runs exactly these. Run them before opening a pull request:

```bash
bash -n scripts/*.sh
shellcheck -S warning -e SC1091 scripts/*.sh          # settings in .shellcheckrc
ruff check scheduler services                         # rules in pyproject.toml
pytest scheduler/tests services/litellm/tests -q
docker compose -f docker-compose.yml config --quiet
docker compose -f docker-compose.vllm.yml config --quiet
```

No shellcheck installed?
`docker run --rm -v "$PWD:/mnt" -w /mnt koalaman/shellcheck:stable -S warning -e SC1091 scripts/*.sh`.

CI additionally verifies the pinned LiteLLM image redacts spend logs
(`services/litellm/tests/verify_spend_log_redaction.py`), checks that the
image list agrees across the compose file, `build-push-images.sh` and the
publish workflow, and checks every relative link and anchor in `*.md`.

The tests need no GPU, no network and no Docker: they exercise the memory
arithmetic, the placement policy and the LiteLLM config generation against
synthetic inputs.

## Conventions

**Comments state the current fact, briefly.** A comment says what a value is
for or what breaks without it, in a phrase rather than a paragraph. No history,
no rejected alternatives, no work log: the repository describes its present
state, and `git log` holds the rest.

**Measured values say so.** Weights, KV sizes and throughput numbers in the
docs are measurements from a real cluster and are labelled as such. A number
copied from a model card is an estimate; mark it.

**Prices are load-bearing.** The declared figures in `scripts/lib.sh` are the
fallback for `gen-litellm-config.sh`, which reads the live OpenRouter catalogue
on every run. Check drift with `./scripts/gen-litellm-config.sh --check-prices`
and update `docs/models.md` in the same commit.

**Shell scripts are the operator interface.** They run under
`set -euo pipefail`, print progress through the `hdr`/`info`/`ok`/`warn`/`err`
helpers in `scripts/lib.sh`, and send diagnostics to stderr so that command
substitution stays clean. A script invoked with no arguments prints its usage.

**Documentation is part of the change.** A flag, environment variable or
default that moves without its documentation moving is an incomplete change.
`docs/` is written for an operator looking at a broken deployment: prefer the
concrete command over the general principle.

## Publishing images

`.github/workflows/publish-images.yml` owns the six `boanlab/kloudchat-*`
images: `crawl4ai-shim`, `search-shim`, `whisper-shim`, `code-interpreter`,
`deep-research`, `index-shim`. vLLM is upstream, pulled by the GPU nodes.

| Trigger | Builds | Tags |
|---|---|---|
| Push to `main` | Only images whose `services/<name>/` directory changed | `latest` |
| Tag `v*` | All of them | `v1.2.3`, `1.2.3`, `1.2`, `latest` |
| Manual run | All, or one chosen image | The tag you type, plus `latest` |

A change outside `services/` publishes nothing. Adding an image means one entry
in the `catalogue` in the `select` job, one in the manual dropdown, one in
`BUILD_TABLE` in `scripts/build-push-images.sh`, and the compose `image:`; CI
fails when the four disagree.

## Commit and pull request style

Short imperative subject, body explaining the reasoning:

```
scheduler: keep resident services out of the stop set

docker ps returns every container on a node, including ones the planner
never placed. Stopping those took the node's STT backend down.
```

Pull requests say how the change was verified. For anything that only runs on
GPU hardware, say plainly when it was not verified there.

## Reporting bugs

Open an issue with the command you ran, the output you got, and your OS,
architecture and GPU. Redact credentials: `.env` contains live LiteLLM and
OpenRouter keys.

For security issues, follow [SECURITY.md](SECURITY.md) instead of opening a
public issue.
