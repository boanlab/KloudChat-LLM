# Qwen3-TTS placement plan

Plan for adding `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (Apache-2.0, Korean and
English, nine preset voices including `sohee`) as the self-hosted speech model
`qwen3-tts`. Nothing has been rolled out: the figures below come from driving
`scheduler.planner` offline with the layout as reported by the owner, and the
vllm-omni facts come from its docs at tag `v0.30.0`. Not run on any GPU.

## 1. Serving facts (vllm-omni v0.30.0)

| Item | Finding |
|---|---|
| Image | `vllm/vllm-omni:v0.30.0`, the pinned one, is enough: its `docs/serving/speech_api.md` and `vllm_omni/deploy/qwen3_tts.yaml` already cover Qwen3-TTS CustomVoice. No version bump |
| Command | `vllm serve /model --omni --trust-remote-code --enforce-eager --served-model-name local/qwen3-tts --stage-overrides '{...}'`; the bundled `qwen3_tts.yaml` is picked up for the model, as in the docs' stage-override example |
| Stages | Two: talker (stage 0) and code2wav (stage 1), each with its own `gpu_memory_utilization`. The bundled value is 0.3 each, 57 GiB of a 95.6 GiB card. The compose service overrides them to 0.06 and 0.04 and sets `max_num_seqs` 8 |
| `--enforce-eager` | The docs' quick start uses it; the bundled file leaves CUDA graphs on (stage 1 has an inner graph). Eager is chosen because sm_120 graph capture is untested here |
| Endpoint | `POST /v1/audio/speech`, OpenAI-shaped: `input`, `voice`, `language`, `response_format` (`wav` default, `mp3`, `flac`, `pcm`, `opus`), `instructions`, `stream`, `stream_format` (`sse` or `audio`; both need `pcm` or `wav`) |
| Voices | `vivian` (default), `ryan`, `aiden`, `dylan`, `eric`, `ono_anna`, `serena`, `sohee`, `uncle_fu`; `GET /v1/audio/voices` lists them. Languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian |
| Checkpoint | `model.safetensors` 3.57 GiB (1.92 B BF16 parameters) plus `speech_tokenizer/model.safetensors` 0.64 GiB: 4.2 GiB on disk, public, no `HF_TOKEN` |

## 2. What was added

| Piece | Choice and reason |
|---|---|
| `scheduler/models.yaml` | `qwen3-tts`, new `runner: speech` (api `speech`), service `omni-qwen3-tts`, port 8094, `priority: 1` (like the picture models, below retrieval and transcription), `replicas: 1`, no `placement` (anywhere with room) |
| Sizing | `weight: 4.5GiB` = 4.2 GiB on disk rounded up. `activation: 5GiB` = the two stage budgets less the weight: 0.06 x 95.6 + 0.04 x 95.6 = 5.7 + 3.8 = 9.5 GiB charged. No KV: the stages' KV pools live inside those budgets. Basis is the compose defaults, not a measurement; measure on the node and adjust |
| Runner kind | `ModelSpec.is_omni` (image or speech) replaces `is_image` wherever the rule is "weights plus activation, no KV, omni compose file, no OpenRouter fallback". `is_image` stays for what is picture-only: the URL goes to `IMAGEGEN_PLACED` |
| Compose | `omni-qwen3-tts` in `docker-compose.vllm-omni.yml`: `container_name`, port 8094, `VLLM_OMNI_QWEN3TTS_DEVICES` (falls back to `VLLM_OMNI_DEVICES`, then `all`) as `device_ids`, `VLLM_OMNI_QWEN3TTS_STAGE0_UTIL` / `_STAGE1_UTIL`, `VLLM_OMNI_QWEN3TTS_DIR`, `/health` check with `start_period: 600s` |
| Routing | No shim. LiteLLM's `openai/` provider already speaks `/v1/audio/speech`, and the node serves it as is, so `gen-litellm-config.sh` gets one `emit_vllm_speech`: `strict-local/qwen3-tts`, `mode: audio_speech`, `api_base: http://<node>:8094/v1`, `kchat_strict_local: true`, `self_hosted`, no `local/` alias and no OpenRouter twin. A shim (imagegen-shim style) would only add a hop; the image shim exists for the chat-completions-to-images translation, which speech does not need |
| Where the URL lives | `VLLM_PLACED` when the scheduler placed it, `VLLM_ENDPOINTS` (`qwen3-tts=http://host:8094`) when run by hand; both already feed `resolve_local_models` |
| Tooling | `manage-vllm-omni.sh download|up|status qwen3-tts`, `lib.sh` alias tables, `.env.example`, `.env.node.example`, docs |

## 3. Placement (planner output)

Layout driven into the planner (free per card, GiB): `.2` 10.7 (head), `.3` 1.9,
`.4` 3.7, `.5` 2.0 (Flash), `.7` GPU0 3.9 and GPU1 32.0. TTS charge: 9.5 GiB.

| Case | Result |
|---|---|
| All five nodes visible, no pin | `.7` GPU 1 (32.0 -> 22.5 GiB free). Worst fit: the card with most room |
| `site.yaml` `nodes: [7]`, `cards: [1]` | `.7` GPU 1, same |
| Pinned to `.7` GPU 0 | Not placed: needs 9.5 GiB, the emptiest card has 32.0 (the pin is checked against the cluster, not the card; the reason is imprecise but the outcome is right) |
| Flash nodes only (`.3` `.4` `.5`) | Not placed: best has 3.7 GiB. Never on a Flash node |
| `.7` held (`VLLM_NODES_HOLD`), no pin | Lands on **`.2`**: 9.5 of 10.7 GiB, leaving 1.2 GiB beside the 27B. Do not do this |
| `.7` held, pinned `nodes: [7]` | Not placed, no fallback, nothing else moves |

Why `.7` GPU 1: it is the only card with room to spare. GPU 0 (3.9 GiB) is too
small, the Flash nodes hold 1.9-3.7 GiB, and `.2` would drop to 1.2 GiB with the
27B, retrieval and Whisper running at 100% utilisation.

GiB accounting on `.7` (free per card, planner basis):

| Card | Before | TTS | After |
|---|---:|---:|---:|
| GPU 0 | 3.9 | 0 | 3.9 |
| GPU 1 | 32.0 | 9.5 | 22.5 |

What else moves: nothing. The test `test_adding_speech_moves_nothing_that_is_already_placed`
checks that every other placement is identical with and without `qwen3-tts`; `.2`, `.3`,
`.4`, `.5` and the three picture models are untouched.

## 4. Because `.7` is held

`VLLM_NODES_HOLD` makes `apply` place, start and stop nothing on `.7`, so the
scheduler will not start this container. Two consequences:

1. Keep `qwen3-tts` out of `VLLM_MODELS` while `.7` is held; list the hand-run
   server in `VLLM_ENDPOINTS` instead. The scheduler then ignores it and counts
   its memory as foreign on the next `plan`.
2. Put the pin in `scheduler/site.yaml` now, so an un-hold or a later move
   cannot send it to `.2`:

```yaml
models:
  - id: qwen3-tts
    nodes: [7]
    cards: [1]
```

If `.7` is ever released and `qwen3-tts` added to `VLLM_MODELS`, `apply` writes
`VLLM_OMNI_QWEN3TTS_DEVICES=1` to the node's `.env`, starts `omni-qwen3-tts`
with `docker-compose.vllm-omni.yml`, and adds `qwen3-tts=http://<.7 host>:8094` to
`VLLM_PLACED` on the orchestrator.

## 5. Env lines

Orchestrator `.env` (hand-run, `.7` held):

```
VLLM_ENDPOINTS=<existing entries>,qwen3-tts=http://<.7 host>:8094
```

`.7` node `.env` (`~/KloudChat-LLM/.env`):

```
VLLM_OMNI_QWEN3TTS_DIR=qwen3-tts
VLLM_OMNI_QWEN3TTS_DEVICES=1
# optional, defaults 0.06 / 0.04
# VLLM_OMNI_QWEN3TTS_STAGE0_UTIL=0.06
# VLLM_OMNI_QWEN3TTS_STAGE1_UTIL=0.04
```

## 6. Node operator commands (on `.7`, in the repo directory)

```bash
git pull                                        # the new compose service and script
./scripts/manage-vllm-omni.sh download qwen3-tts   # 4.2 GB, public repo, no HF_TOKEN
                                                # = hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
                                                #     --local-dir $VLLM_MODELS_ROOT/qwen3-tts
# set VLLM_OMNI_QWEN3TTS_DEVICES=1 in .env, then:
./scripts/manage-vllm-omni.sh up qwen3-tts      # = docker compose -f docker-compose.vllm-omni.yml up -d omni-qwen3-tts
./scripts/manage-vllm-omni.sh status            # "cards 1", health: healthy (allow up to 10 min)
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv   # about 9-10 GiB on GPU 1 only
curl -s http://localhost:8094/v1/audio/voices
curl -s http://localhost:8094/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"안녕하세요. 반갑습니다.","voice":"sohee","language":"Korean"}' --output /tmp/t.wav
```

Then on the orchestrator, after adding the `VLLM_ENDPOINTS` entry:

```bash
./scripts/gen-litellm-config.sh && docker compose up -d --force-recreate litellm
python3 -m scheduler plan        # confirm nothing else would move
```

Do not run `apply` for this; it is not needed while `.7` is held.

## 7. Risks

| Risk | Detail and handling |
|---|---|
| Blackwell sm_120 start-up | Not tested by anyone here; the bundled file says "verified on 1x H100". `--enforce-eager` avoids CUDA-graph capture, but the stage workers use MRV2 by default on CUDA; if the talker fails to start, set `model_runner: v1` through a custom deploy config. Check `docker logs omni-qwen3-tts` first |
| vllm-omni version | `v0.30.0` has the Qwen3-TTS docs and deploy file; I read them at that tag, not on a node. Later tags exist (v0.32, v0.34); a bump is only for a failure, and applies to all four omni services |
| Stage shares are fractions of the card | 0.06 + 0.04 is 9.5 GiB on 95.6 GiB; on a smaller card the same fractions would not fit the 3.6 GiB talker. Unmeasured: the talker needs its 3.57 GiB plus KV inside 5.7 GiB; raise `STAGE0_UTIL` if the engine reports no KV room |
| Contention with image jobs on GPU 1 | The edit model (56 + 2 GiB) shares GPU 1 with TTS: 58 + 9.5 = 67.5 of 95.6 GiB, and its activation peaks are not in the plan. Compute also contends: a long image edit slows speech while it runs. Latency, not memory, is the likely symptom |
| Free memory is a snapshot | 32 GiB free on GPU 1 is as reported today; if the edit model grows during serving, vLLM fails the stage's start check, not a running request |
| First-request latency | Model load and warm-up up to the 600 s `start_period`; with eager mode the first request after start is slower. Send one request after `up` |
| `.2` fallback | A held `.7` and an unpinned `qwen3-tts` in `VLLM_MODELS` would put it on `.2` (1.2 GiB left). Pin in `site.yaml` and keep it out of `VLLM_MODELS` |
| metrics-shim | It scrapes `VLLM_ENDPOINTS` targets as vLLM `/metrics`; the omni speech server may not expose the same series, so the node may show as down in the dashboard until checked |

## 8. Open questions for the owner

1. Is `.7` GPU 1 an acceptable seat given it shares the card with the 56 GiB edit
   model, or should TTS wait for GPU 0 to free up?
2. Hand-run via `VLLM_ENDPOINTS` (recommended while held), or release `.7` from
   the hold so `apply` manages all four omni services?
3. Should the app call it as `strict-local/qwen3-tts` through LiteLLM's
   `/v1/audio/speech`? Existing OpenRouter audio models use the same `audio_speech`
   route, so the app change should be small, but no app code was touched.
4. Default voice for Korean (`sohee`) and any request cap (`max_num_seqs` is 8)?
