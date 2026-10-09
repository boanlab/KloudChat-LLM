# Tools

The services behind the gateway, and where each request goes.

The UI stores these addresses in its admin screen and calls them from its own
tool implementations. When to use a tool, and which agent may use it, is
decided by the UI. This document covers only what the backend provides.

## Paths and the calls behind them

The gateway (`services/gateway/Caddyfile`) strips the prefix and forwards to
each service's root. The UI stores everything up to the prefix as a base URL
and appends the service-specific path.

| Capability | Base URL | What the UI calls | Service | Profile |
|---|---|---|---|---|
| Models | `/litellm` (also `/v1`) | `/litellm/v1/chat/completions`, `/litellm/key/generate` | LiteLLM | `models` |
| Web search | `/tools/search` | `/tools/search/search?q=…&format=json` | search-shim → SearXNG | `tools` |
| Document fetch | `/tools/fetch` | `/tools/fetch/v1/scrape` | crawl4ai-shim | `tools` |
| Code execution | `/tools/exec` | `/tools/exec/exec` | code-interpreter | `tools` |
| Deep research | `/tools/research` | `/tools/research/mcp` | deep-research | `tools` |
| Transcription | `/tools/stt` | `/tools/stt/v1/audio/transcriptions` | whisper-shim | `whisper` |
| Retrieval index | `/tools/index` | `/tools/index/{documents,search}` | index-shim | `index` |

The contract is identical without the gateway, so a raw service address works
too, except that the gateway also requires the tools key
([Authentication](#authentication)). `GET /health` on the gateway answers
`{"status":"ok"}` without a key.

LiteLLM is exposed at its service root because its admin API (`/key/*`,
`/user/*`, `/model/info`, `/health/*`) and admin UI (`/ui/`) do not live under
`/v1`. `/v1/*` is also open, for OpenAI-compatible clients.

## Authentication

Only `/litellm/*` and `/v1/*` pass the caller's `Authorization` header through.
The UI calls with a per-user LiteLLM virtual key, so spend is attributed to a
person.

`/tools/*` requires the shared key `TOOLS_GATEWAY_KEY` in an `X-Tools-Key`
header. A request without it, or with a different value, gets
`401 {"error":"unauthorized"}` before any upstream is contacted; the header is
removed before the request is proxied, so no service behind the gateway sees
it. `Authorization: Bearer` is not accepted in its place: every client sets
custom headers, and `/tools/fetch` already uses `Authorization` upstream.

Fail closed: compose refuses to start the gateway when `TOOLS_GATEWAY_KEY` is
unset or empty, and the Caddyfile denies every `/tools` request when the key
it was started with is shorter than 32 characters. Use hex
(`openssl rand -hex 32`, which `gen-env.sh` does): the value is substituted
into a CEL expression in the Caddyfile, and `setup.sh` rejects anything else.

The internal keys for code execution (`CODE_INTERPRETER_API_KEY`) and document
fetching (`SCRAPER_API_KEY`) are injected by the gateway; caller-supplied
credentials are ignored. **The gateway port must still only be open inside a
private network**: the key is a single shared secret, not a per-user
credential, and a sandbox that runs arbitrary code sits behind it.

Where the key lives:

| Repository | `.env` variable | Consumer |
|---|---|---|
| KloudChat-LLM | `TOOLS_GATEWAY_KEY` | the gateway (`services/gateway/Caddyfile`), `setup.sh` probes |
| KloudChat | `KCHAT_TOOLS_GATEWAY_KEY` | the API, as `TOOLS_GATEWAY_KEY`; sent only to the tool addresses |

Services inside this stack call each other by container name, not through the
gateway, so none of them needs the key.

Rotating it (tool calls fail with 401 between steps 3 and 4; pick a quiet
moment):

1. `openssl rand -hex 32`.
2. Write the value to `TOOLS_GATEWAY_KEY` here and `KCHAT_TOOLS_GATEWAY_KEY`
   in KloudChat's `.env`.
3. In KloudChat: `docker compose up -d api` (it now sends the new key).
4. Here, recreate only the gateway:
   `docker compose up -d --no-deps --force-recreate gateway`.
5. `./scripts/setup.sh urls` shows `connected` for every tool, not
   `key rejected`.

Each backing store sits on an internal network shared only with the service
that owns it (`litellm-db` with LiteLLM, `index-db` with index-shim, MinIO and
redis with code-interpreter, valkey with SearXNG); none publishes a port.

## Services

### Web search: search-shim and SearXNG

SearXNG is a metasearch front end: general search is scraped, so queries never
reach a commercial search API, but every one of them leaves this host through
a single address. Engines ban that address on burst, and one CAPTCHA suspends
the engine for every user at once. `search-shim` therefore sits in front of
SearXNG; the UI, deep research and anything else on `/tools/search` go through
it.

The shim is transparent for SearXNG's API: `/search` in JSON or HTML, and any
other path is forwarded as is, outside the cap. On `/search` it adds:

- **Strict safe search**: `SAFESEARCH` (2) replaces whatever the caller sent.
  SearXNG's own `safe_search` is 2 as well.
- **Region on the language**: a bare `language=ko` becomes `ko-KR`. SearXNG's
  `default_lang` is `ko-KR` for a request that carries no language.
- **Result cache**: a search with results is answered from memory for
  `CACHE_TTL_S` (15 min), keyed by the query (case and spacing folded) and
  every other parameter. Empty answers are not cached.
- **Coalescing**: identical searches in flight at the same moment share one
  upstream call.
- **Cap on searches in flight**: at most `MAX_CONCURRENT_SEARCHES` (12) reach
  SearXNG at once; a request past that waits up to `QUEUE_TIMEOUT_MS` (10 s).
  When the wait runs out or SearXNG fails (5xx, refused connection, timeout),
  an answer up to `STALE_TTL_S` (6 h) old is served if there is one; without
  one the caller gets `503` (`busy: too many searches in flight`) or `502`,
  both with an empty `results` list. SearXNG's own `4xx` passes through.

`/health` on the shim reports searches running, waiting and coalesced, and the
cache size. SearXNG's worker state is shared through valkey and may be empty
after a restart. Its configuration is `services/searxng/settings.yml`,
generated by `scripts/gen-searxng-config.sh` from `settings.yml.example` and
`SEARXNG_SECRET_KEY` in `.env`; `setup.sh` runs the generator and restarts
SearXNG when the file changed. Engine suspension after a block is shortened
there to minutes (`suspended_times`), so a ban heals on its own. Search terms
appear in the shim's log at `DEBUG` only.

The engine set is explicit (`use_default_settings: engines: keep_only`), aimed
at a Korean-language audience. NAVER's is the only commercial search API.

| Category | Engines | For |
|---|---|---|
| general | yandex, brave, wikipedia, openlibrary | everyday, administrative and business questions; books |
| news | duckduckgo news, bing news, reuters | current events |
| science | google scholar, openalex, semantic scholar, arxiv, pubmed, openaire | literature |
| it | github, stackoverflow, mdn, docker hub, microsoft learn, huggingface | engineering |

Not enabled: scraper engines that answer a datacenter address with a CAPTCHA
or an access denial (google, bing, duckduckgo, the naver scraper, startpage,
qwant, mojeek), and paid APIs.

NAVER's Search API is the one keyed engine, wired in as two `json_engine`
entries: `naver web` (general, 웹문서) and `naver news` (news). The API is
served by NAVER API HUB on NAVER Cloud Platform
(`naverapihub.apigw.ntruss.com`, authenticated with `X-NCP-APIGW-API-KEY-ID` /
`X-NCP-APIGW-API-KEY`). `gen-searxng-config.sh` fills the credentials from
`NAVER_CLIENT_ID` / `NAVER_CLIENT_SECRET` in `.env` and enables them only when
both are set; without a key they are written disabled and never called. The
search APIs share a quota of 775,000 calls a month per application at 50
requests a second per key, one call per engine per search. To turn them on:
create an application in NAVER API HUB, enable the 웹문서 and 뉴스 search APIs
on it (an API not enabled answers 401), put the two values in `.env`, run
`./scripts/gen-searxng-config.sh`, then `docker compose restart searxng`.

### Document fetch: crawl4ai-shim

A Firecrawl-compatible `/v1/scrape` (also `/v0/scrape`, `/v2/scrape`) in front
of Crawl4AI. The image carries Playwright and Chromium, so pages that need
rendering come back as markdown body text. Page timeout `DEFAULT_TIMEOUT_MS`
(30 s).

A URL is judged three times. Before the browser opens it, by the addresses
its host resolves to: nothing on the deployment's own network (`netguard.py`),
and no adult site (`adultlist.py`). While the page loads, every request it
makes (navigation, redirect, script, XHR) passes a route guard before it
leaves the browser; the guard follows redirects itself, judges each hop, and
aborts the request at the first refused one. After the load, the address the
browser ended up at is checked once more. The adult list is a hosts file baked
into the image at build time from a pinned commit of StevenBlack/hosts'
porn-only list (`ADULT_HOSTS_FILE`). It keeps hosts-file semantics: the listed
host is refused, `www.` folded, nothing more inferred. A refused fetch answers
`{"success": false, "error": "<reason>"}` (for example `adult sites cannot be
scraped` or `internal network addresses cannot be scraped`), which the caller
treats like an unreadable page. `/health` reports the list size. Page
addresses appear in the shim's log at `DEBUG` only.

One Chromium serves every request. At most `MAX_CONCURRENT_PAGES` (8) render
at once; a request past that waits up to `QUEUE_TIMEOUT_MS` (15 s) for a slot
and is then answered `{"success": false, "error": "busy: too many pages
rendering"}`. A successful scrape is kept in memory for `CACHE_TTL_S` (15 min)
keyed by URL, formats and the main-content flag. `/health` reports pages
rendering, waiting, and cache entries.

A scrape that finds Chromium dead starts a new one. After three such scrapes in
a row `/health` answers `503` with `"status": "browser_down"`.

### Code execution: code-interpreter

Sandboxed Python (`librecodeinterpreter`, pinned by digest in
`services/code-interpreter/Dockerfile`). The only service with `SYS_ADMIN` and
relaxed apparmor/seccomp. Execution is capped at `MAX_EXECUTION_TIME` (30 s).
Output files are stored in MinIO; the UI reads stdout and stderr.

Memory: each sandbox gets its own cgroup with `memory.max` = `MAX_MEMORY_MB`
(1024) and no swap. The upstream service only passes `--rlimit_as hard` to
nsjail, which is unlimited in a container, so the image adds two scripts:
`cgroup-init` (the compose `command`) remounts the container's cgroup v2
read-write with the `SYS_ADMIN` it already has, moves the service into a
`service` leaf and enables the memory controller; `nsjail-memcg`
(`NSJAIL_BINARY`) calls nsjail with `--use_cgroupv2 --cgroup_mem_max`. The limit
covers the whole sandbox, and a warm Python REPL is about 260 MB before user
code, so roughly 750 MB is left for data. Going over it kills the sandbox and
the call returns an error (`Invalid response from REPL`), not a `MemoryError`.
The container as a whole is capped at 4 GB (`mem_limit`, no swap).

There is no network at runtime, so no `pip install`; file skills can use only
what the image has. Besides pandas, numpy, openpyxl, xlsxwriter, python-docx,
python-pptx, reportlab and matplotlib, the Dockerfile adds pinned `pdfplumber`,
`pdfminer.six` (the base image's broken old `pdfminer` is removed), `pypdf`
and `pypdfium2`. `pdftotext` works from Python; `soffice` only from
`lang: "bash"` (the Python sandbox hides `/proc`). Tesseract has no Korean.

Output files: the working directory is `/mnt/data`, and new or changed files
under it are returned after each run, but only the first 10 by path; the rest
are dropped without an error. Files whose name starts with `code`, hidden
files, and anything under directories such as `build/`, `dist/`, `node_modules/`
or `__pycache__/` are never returned. Give outputs distinct names under a
directory such as `outputs/`.

The image bundles NanumGothic as the matplotlib default so that Korean chart
labels render.

### Deep research: deep-research

`local-deep-research` wrapped by `mcp-proxy` into an MCP streamable-http
server on port 8081. It searches through `search-shim` and calls models
through LiteLLM as `DEEP_RESEARCH_MODEL`. A call takes minutes to tens of
minutes.

The UI attaches it as an HTTP MCP connector. Internals are in
[internal/deep-research.md](internal/deep-research.md).

### Transcription: whisper-shim

A front end for the `vllm-whisper` backends (`openai/whisper-large-v3` on
vLLM, placed on the GPU nodes like any other model). `WHISPER_URLS` lists the
nodes it was placed on; with several, the shim routes by in-flight count over
a `HEALTH_CACHE_TTL_SEC` (10 s) health cache.

It sets the model name on every forwarded request to `WHISPER_MODEL_NAME`
(`local/whisper-large-v3`). vLLM answers only for a name it was started with,
while callers send whatever their client was configured with (`whisper-1`,
most often; the backend also serves that alias).

Without a backend the shim's `/health` reports `degraded`, so it is fenced
behind the `whisper` profile: `setup.sh` enables that profile once
`WHISPER_URLS` is populated.

The OpenRouter STT route is registered whenever no backend *answers*, which is
not the same as `WHISPER_URLS` being empty: that variable says where the model
was placed. `gen-litellm-config.sh` probes the URLs the same way it probes the
chat backends, so a dead one delegates rather than 5xx-ing.

### Retrieval index: index-shim

Chunks, embeds and stores documents in pgvector (`index-db`), and searches
them with a two-stage recall/rerank pipeline through LiteLLM. Endpoints:
`PUT /documents`, `POST /search`, `DELETE /documents/{id}`,
`DELETE /collections/{name}`, `GET /health` (database and embedding
availability reported separately).

- A collection holds one embedding model's vectors. It is searched and
  extended only with the model its rows hold; when that model is unavailable
  the call fails (503) rather than mixing vector spaces.
- An empty collection answers a search with no passages and no embedding
  call.
- A new collection takes the first model in `EMBED_MODELS` that answers; the
  shim remembers it for the next unpinned call. `/health` probes embeddings
  without changing that preference.

Runs under the `index` profile. Tuning is in
[env-reference.md](env-reference.md), the model side in
[models.md](models.md#retrieval).

### Models: LiteLLM

The single point for every model call. A local model whose node is down or
saturated fails over to its OpenRouter twin
([models.md](models.md#local-to-openrouter-fallback)). Configuration is
`services/litellm/config.yaml`, generated by `scripts/gen-litellm-config.sh`
from the `VLLM_*_URL` values and the OpenRouter key in `.env`. The container
receives `OPENROUTER_API_KEY` and `OPENAI_API_KEY` from `.env`. LiteLLM
publishes no host port; it is reached through the gateway at `/litellm/*`.

## Checking status

```bash
./scripts/setup.sh urls
```

It prints whether each capability is reachable through the gateway, sending
`TOOLS_GATEWAY_KEY` from `.env` on `/tools/*`. `key rejected` means the gateway
answered 401: the running gateway holds a different key than `.env`.
`not started` means that service's container is not running; check
`COMPOSE_PROFILES` first.
