---
title: Technyx Chatbot API
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Technyx Chatbot - RAG on a vector database (Supabase / pgvector)

The vector-database sibling of `../chatbot-rag`: same pipeline, guardrails
and answer quality, but the knowledge base lives in **Postgres with pgvector
(Supabase)** instead of a `.npz` file baked into the image - so updating the
site's content is a database sync, **not a rebuild and redeploy**.

A retrieval-augmented-generation (RAG) chatbot for Technyx Systems that
uses the **stock, unmodified** `google/gemma-3-1b-it` instruction-tuned
model - no LoRA, no QLoRA, no fine-tuning of any kind. All company-specific
grounding comes from retrieval: at answer time, the user's question is
embedded, the most relevant facts are pulled from a small crawled corpus by
cosine similarity, and those facts are injected into the prompt as verified
context for the model to use.

This is the "core RAG" sibling of two other projects in this workspace:

| | `../chatbot-finetune` | `../lightweight-chatbot` | `chatbot-rag` (this project) |
|---|---|---|---|
| Fine-tuning | QLoRA (GPU required) | Plain LoRA (CPU-friendly) | **none** |
| Grounding mechanism | Training data alone, later + RAG | Training data + RAG | **RAG only** |
| Base model | `google/gemma-3-1b-it` | `google/gemma-3-1b-it` | `google/gemma-3-1b-it` |
| What changes at inference | Adapter weights + retrieved context | Adapter weights + retrieved context | Prompt + retrieved context only |

The point of this project is to isolate what retrieval alone can and can't
fix, with no adapter weights changing the model's behavior at all.

## How it works

1. `scripts/crawl_technyx.py` writes the crawled facts to
   `data/knowledge/facts.jsonl`. Hand-curated "answer cards" live separately in
   `data/knowledge/answer_cards.jsonl`, so a re-crawl can never overwrite them.
2. `scripts/sync_facts.py` embeds the facts with the configured embedding model
   (`EMBEDDING_MODEL`, default `BAAI/bge-base-en-v1.5`, 768-dim) and upserts
   them into the `public.facts` table. Only facts whose text changed are
   re-embedded; facts removed from the site are deleted. The running API sees
   the change on its next query.
3. At answer time the user's question is embedded and a SQL query
   (`order by embedding <=> query`, HNSW index, cosine distance) returns the
   nearest facts plus every fact eligible for a ranking boost; the same boosts
   as before (head-office, testimonial, answer-card) are applied in
   `src/chatbot_rag/retrieval.py`, so rankings match the previous exact scan.
4. The top-k facts above the similarity threshold are rendered into a
   `Relevant information:` block, and the model answers under the same
   guardrails and fact-check as `../chatbot-rag` (`pipeline.py`).

`open_index()` refuses a database whose vectors were built with a different
embedding model than `EMBEDDING_MODEL` - re-run `scripts/sync_facts.py --full`.

## Database

Postgres with the `vector` extension; Supabase has it built in. Schema:
`supabase/migrations/*.sql` (`public.facts` with an HNSW index, `public.index_meta`;
RLS enabled with no policies, so the public Supabase Data API cannot read or
write them - the API uses a server-side connection string).

```bash
cp .env.example .env     # set DATABASE_URL (Supabase: Project Settings -> Database -> Connection string)
python scripts/migrate.py        # create the tables/indexes (idempotent; tracked in public._app_migrations)
python scripts/sync_facts.py     # load facts + answer cards (embeds on first run, ~1 min on CPU)
```

Connection strings: use the **Direct** connection or **Session pooler** (port
5432) for `migrate.py`/`sync_facts.py`; the API also works through the
**Transaction pooler** (6543). Apply schema with `migrate.py` rather than the
Supabase CLI's `db push`, so there is one record of what has been applied.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e '.[dev]'            # adds pytest, for the test suite below
```

### Hugging Face access

`google/gemma-3-1b-it` is a gated model. Accept the license at
https://huggingface.co/google/gemma-3-1b-it, then authenticate:

```bash
cp .env.example .env
# edit .env and set HF_TOKEN=<your token>
```

or run `hf auth login`. `get_hf_token()` also picks up a token saved by
that command or `notebook_login()`'s on-disk cache, not just the
environment variable.

## Running it

```bash
# Load the knowledge base into the database (see "Database" above).
python scripts/migrate.py
python scripts/sync_facts.py

# Interactive chat (RAG on by default)
python scripts/chat.py

# Same, with retrieval disabled - useful for seeing what the base model
# does on its own, for comparison
python scripts/chat.py --no-retrieval

# Qualitative WITHOUT-RAG vs WITH-RAG comparison over a fixed test set
python scripts/evaluate.py --test-cases evaluation/test_cases.jsonl

# RAG column only (faster - skips generating the no-retrieval comparison)
python scripts/evaluate.py --skip-no-rag
```

## Speed (GGUF / llama.cpp)

Plain PyTorch/transformers inference on CPU is slow: **~1.6-2.9 tokens/sec**
measured directly on `google/gemma-3-1b-it` in full precision, meaning a
full ~120-token answer can take 60-100+ seconds - not viable latency for
live chat. Exporting the same base model to a quantized GGUF file and
running it through [llama.cpp](https://github.com/ggml-org/llama.cpp)
instead measured **~12.8 tokens/sec** on the same hardware - a real
4-8x speedup, not a rounding-error improvement - while the model itself
shrinks from ~4GB (fp32) to **777MB** (Q4_K_M). `scripts/chat_gguf.py`
uses the exact same `pipeline.answer()` as `scripts/chat.py` - every
guardrail (`guardrails.py`) and the fact-check guardrail
(`fact_check.py`) apply identically; verified directly, including a
multi-turn follow-up that depends on `retrieval.build_retrieval_query`'s
history-folding. Only the token-generation backend differs
(`gguf_llm.py` instead of `llm.py`).

No LoRA adapter to merge here, unlike `../lightweight-chatbot`'s
`export_gguf.py` (this project never fine-tunes the base model at all) -
otherwise the same approach that project already measured to work.

This needs tooling deliberately kept out of `requirements.txt` (it's an
optional export/inference path, not needed for the default plain-PyTorch
path):

```bash
# 1. Into THIS project's venv (or a separate one, to avoid dependency
#    conflicts - either works for inference; step 2 needs its own venv
#    regardless):
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# 2. Clone llama.cpp itself for its HF->GGUF conversion script (not
#    pip-installable), and install its conversion requirements into a
#    SEPARATE venv (it pins transformers/protobuf versions that can
#    conflict with this project's):
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /path/to/llama.cpp
python3 -m venv /path/to/gguf-convert-venv
/path/to/gguf-convert-venv/bin/pip install -r /path/to/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# 3. Export (uses this project's venv for saving the base model +
#    quantizing, and the conversion venv's python for the HF->GGUF step):
python scripts/export_gguf.py \
    --llama-cpp-dir /path/to/llama.cpp \
    --llama-cpp-python /path/to/gguf-convert-venv/bin/python \
    --output outputs/technyx-gemma3-1b-q4_k_m.gguf

# 4. Chat (fast path, same RAG + guardrails as scripts/chat.py):
python scripts/chat_gguf.py --model outputs/technyx-gemma3-1b-q4_k_m.gguf
```

## Serving

`src/chatbot_rag/api.py` is a FastAPI wrapper around `pipeline.answer()`
for actual deployment - `scripts/chat.py`/`scripts/chat_gguf.py` are
interactive REPLs for a single trusted operator, not a service other
things can call. See `SECURITY.md` for the review this design responds
to (fail-closed auth, no stack traces in responses, the input-length
guardrail).

```bash
pip install -r requirements-serving.txt   # fastapi + uvicorn + httpx, optional

export API_KEY=<a-secret-you-choose>                              # required - fails closed without it
export GGUF_MODEL_PATH=outputs/technyx-gemma3-1b-q4_k_m.gguf       # optional - the fast backend (see "Speed")

python scripts/serve.py
# -> POST http://127.0.0.1:8000/v1/chat   {"message": "...", "history": [...]}   header: X-API-Key
# -> POST http://127.0.0.1:8000/v1/chat/stream  {"message": "..."} (public widget, SSE)
# -> GET  http://127.0.0.1:8000/health
# -> GET  http://127.0.0.1:8000/metrics    (Prometheus text format, X-Metrics-Key)
```

### Switching LLM backends

`LLM_BACKEND` picks which of three `generate_fn` implementations `api.py`
loads at startup - switching providers is a matter of changing this one env
var (plus its own config) and restarting, no code changes:

| `LLM_BACKEND` | What it runs | Needs |
| --- | --- | --- |
| `plain` (default) | Local PyTorch/transformers, any HF causal-LM checkpoint | `MODEL_NAME` (default `google/gemma-3-1b-it`), `HF_TOKEN` if gated |
| `gguf` | Local llama.cpp, ~3.6x faster on CPU (see "Speed") | `GGUF_MODEL_PATH` (from `scripts/export_gguf.py`) |
| `api` | Any OpenAI-compatible `/chat/completions` endpoint - OpenAI, DeepSeek, Groq, etc. (`api_llm.py`) | `LLM_API_BASE_URL`, `LLM_API_KEY`, `LLM_API_MODEL` |

Leaving `LLM_BACKEND` unset keeps the old behavior (`gguf` if
`GGUF_MODEL_PATH` is set, else `plain`), so existing deployments don't need
to change anything. For example, to point at DeepSeek instead of a local
model:

```bash
export LLM_BACKEND=api
export LLM_API_BASE_URL=https://api.deepseek.com
export LLM_API_KEY=<your-deepseek-key>
export LLM_API_MODEL=deepseek-chat
```

With `api`, no model weights load at all - every request is a network call
to that provider, so startup is instant but each chat request now depends
on that provider's latency/availability and costs per token. Retrieval,
guardrails, and fact-checking (`pipeline.answer()`) are identical across
all three backends either way - see `pipeline.py`'s module docstring.

### Deploying on Hugging Face Spaces (free, no card)

The repo root is a ready-to-run Docker Space (front matter at the top of this
README sets `sdk: docker`, `app_port: 7860`). Free CPU Spaces have ~16 GB RAM,
enough for the embedding model.

1. Create a **Docker** Space at huggingface.co/new-space (blank template, CPU basic).
2. Space -> Settings -> *Variables and secrets*. Secrets: `DATABASE_URL`
   (Supabase Transaction pooler, port 6543), `API_KEY`, `LLM_API_KEY`.
   Variables: `LLM_BACKEND=api`, `LLM_API_BASE_URL`, `LLM_API_MODEL`,
   `CORS_ORIGINS` (your site's origin), `EMBEDDING_MODEL=BAAI/bge-base-en-v1.5`.
3. Push this repo to the Space's git remote (`git push space main`); the
   Space builds the Dockerfile and starts automatically.
4. Check `https://<user>-<space>.hf.space/health` (reports `facts_indexed`).

The Space repo is public if the Space is public, so never commit secrets - they
live only in the Space settings. Free Spaces sleep when idle and wake on the
next request.

### Deploying the API on Render

This repository includes a Render Blueprint for the Gemini-backed
FastAPI service. In Render, connect the repository and choose Blueprint;
Render will build the root Dockerfile, use /health for readiness, and
configure the non-secret runtime values.

Set these Blueprint secrets in the Render dashboard before the first deploy:

- DATABASE_URL: Supabase Postgres connection string (Transaction or Session pooler).
- API_KEY: shared key sent by your Next.js server as X-API-Key.
- LLM_API_KEY: Gemini API key from Google AI Studio.
- CORS_ORIGINS: comma-separated exact frontend origins.

The Render image uses LLM_BACKEND=api and does not install the optional native
GGUF dependency. The local GGUF workflow remains available for development.

After deployment, verify:

    curl https://<service>.onrender.com/health
    curl -X POST https://<service>.onrender.com/v1/chat \
      -H 'Content-Type: application/json' \
      -H 'X-API-Key: <API_KEY>' \
      -d '{"message":"What services does Technyx offer?"}'

What it does and doesn't provide, honestly:

- **Auth**: one shared API key (`X-API-Key` header vs. the `API_KEY` env
  var). No per-user identity, key rotation, or scopes - adequate for a
  small deployment behind one trusted key, not multi-tenant access control.
- **Rate limiting**: in-memory, per-API-key, sliding window
  (`RATE_LIMIT_MAX_REQUESTS`/`RATE_LIMIT_WINDOW_SECONDS`, default 20/60s).
  Single-process only (see `ratelimit.py`'s docstring) - running multiple
  worker processes or replicas multiplies the effective limit rather than
  sharing it. Swap in a shared store (Redis) before scaling out.
- **Error handling**: every exception is logged server-side
  (`logger.exception`, full detail) and only ever returns a generic
  message in the response body - verified directly (a test asserts no
  repo file path or the word "Traceback" ever appears in a response).
- **Concurrency**: generation requests are serialized per process behind
  `app.state.answer_lock` - required for correctness, not just
  performance (load-tested: without it, concurrent requests corrupted the
  GGUF backend's internal state and hung the process - see `SECURITY.md`'s
  "Concurrency safety" for the full writeup). The event loop itself stays
  responsive throughout (`/health` answers in a few ms even mid-generation
  - also verified directly), so this only serializes actual model access,
  not the whole server. Scale concurrent capacity with multiple process
  replicas behind a load balancer, not by increasing in-process
  concurrency.
- **Observability**: `/metrics` (Prometheus text format) exposes request
  counts by outcome (`generated`/`guardrail_refused`/`fact_check_blocked`/
  `error`), a request-latency histogram, and auth/rate-limit rejection
  counts - point your own Prometheus/Grafana/Datadog/etc. at it (not
  behind auth by default - see `api.py`'s `/metrics` docstring for why,
  and when that assumption doesn't hold for your deployment). A separate
  `chatbot_rag.flagged` log channel records every guardrail refusal and
  fact-check-blocked response (question, response, reason) apart from
  normal request logs, for a human to actually review - the plumbing for
  review is here, the review itself is a process only you can run.
- Defaults to `127.0.0.1`, not `0.0.0.0` - set `HOST=0.0.0.0` explicitly
  to accept connections from other machines.

### Embedding the website widget

Set `CORS_ORIGINS=https://technyxsystems.com` (add a preview origin only when
needed), then include this one dependency-free script on the site:

```html
<script src="https://api.example.com/widget.js" defer></script>
```

It calls `/v1/chat/stream`, stores its opaque `conversation_id` in browser
storage, and never contains `API_KEY`. It shows retryable 401/429/503 failures
accessibly. The SSE endpoint waits for post-generation fact checking before
emitting chunks, so correctness takes precedence over first-token latency.
Open `deploy/widget-demo.html` against a configured local API to test it.

The image prefetches the configured embedding model (`EMBEDDING_MODEL`,
pinned to `BAAI/bge-base-en-v1.5` in the Dockerfile to fit small instances
- see "Deploying" below for the memory tradeoff), so retrieval works
offline at runtime after the image build. The plain Gemma backend is still
gated and must be mounted/cached with its accepted-license credentials, or
use a mounted GGUF model.

## Testing

```bash
# Fast: pure guardrail logic + real pgvector retrieval, no LLM. The tests use
# a throwaway Postgres (started automatically via the `pgserver` dev
# dependency, or set TEST_DATABASE_URL to any disposable pgvector database) -
# never DATABASE_URL, so they cannot touch your Supabase data.
pytest

# Slow: end-to-end through the real 1B model (requires HF_TOKEN/license,
# same as scripts/chat.py). Minutes, not seconds.
pytest -m slow
```

`pytest` alone runs `tests/test_guardrails.py`, `tests/test_fact_check.py`,
`tests/test_ratelimit.py` (pure decision logic - no model, no I/O),
`tests/test_retrieval.py` (real embedding model + the built index, but
never the 1B LLM), and the fast half of `tests/test_api.py` (the
auth/rate-limit/history-conversion logic as plain functions, no app boot).
`tests/test_generation.py` and the rest of `tests/test_api.py` (which
boots the real FastAPI app end-to-end) are marked `slow` and excluded by
default (see `pyproject.toml`'s `addopts`) since they load and generate
from the full model; run them explicitly before trusting a change that
touches `retrieval.py`, `guardrails.py`, `prompting.py`, `fact_check.py`,
or `api.py`. Set `GGUF_MODEL_PATH` first (see "Speed") to run the slow
suite against the fast backend instead of waiting on the plain one.

Also runs automatically in CI on every push/PR to `main` - see
`.github/workflows/tests.yml`. `.github/workflows/slow-tests.yml` runs
the real-model suite, manually triggered (needs an `HF_TOKEN` repo secret
- see that workflow's comments).

Each test encodes a specific behavior measured during this project's
development (see the docstrings/comments on each one) - they exist to
catch a regression automatically instead of requiring another manual
`scripts/evaluate.py` run and a by-eye read of the transcript, which is
how every fix in this project's history was actually verified.

## Measured results

These are real outputs from this exact pipeline (`--temperature 0`,
greedy decoding, `google/gemma-3-1b-it`, no adapter), not illustrative
examples.

**Retrieval fixes the core hallucination problem, without any training:**

| Question | Without RAG (base model alone) | With RAG |
|---|---|---|
| Where is Technyx headquartered? | "We primarily operate out of our main office in Silicon Valley." | "Our head office is in Dubai, UAE, alongside offices in McKinney Texas USA, Karachi, Pakistan, and Sydney Australia." ✅ |
| How can I contact Technyx? | "email us at support@technyx.com... phone line, which is 555-123-4567" (all fabricated) | (correct, verified contact details returned) ✅ |

**A base model with no training on refusal behavior doesn't reliably
decline on its own** - asked "What is the capital of France?" (retrieval
correctly found nothing), it answered "Our primary office is in Paris. The
capital of France is Paris." instead of declining, and fabricated things
like a fake CMS platform partnership or a fake cloud-provider partnership
when asked leading questions. Prompt instructions and per-question
guardrails each closed the exact case tested, but a *different* phrasing
of the same underlying weakness kept regressing in the next full test run
- see `src/chatbot_rag/guardrails.py` and `src/chatbot_rag/fact_check.py`
for the two different fixes this required:

- `guardrails.py` - code-level refusals for specific query *shapes*
  (nothing retrieved, a request for a concrete example with no evidence
  for one, a question about Technyx's own people that only retrieves
  client testimonials).
- `fact_check.py` - a general, question-agnostic check on the *generated
  answer itself*: any named entity (a platform, vendor, or person) the
  model mentions that isn't actually present in the retrieved context is
  replaced with a safe fallback, regardless of what question produced it.
  This is what finally closed the WordPress adversarial trap and a
  fabricated Microsoft/AWS partnership claim, both of which survived every
  earlier, more targeted fix.

**Known remaining gap**: `fact_check.py` only catches unsupported *named
entities*. A fabricated *policy* claim with no proper noun in it - asked
about a Webby Award, the model declined the award question but then added
"we don't publicly release details about specific awards we've received,"
an invented policy - isn't caught by this check. Closing that would need
a broader (and slower, since it'd likely require a second LLM call)
verification step; deliberately not built yet.

**A separate, structural reliability gap remains**: this is about wrong
*designation*, not a fabricated entity, so `fact_check.py` doesn't apply.
Asked "Is Technyx based in the US?", the model has answered both correctly
(no false claim) and incorrectly ("We have offices in McKinney, Texas,
USA. Our head office is located there." - McKinney is a real office, but
Dubai is the real head office) across different runs of this exact
question, with no code change in between - a reflection of the
non-determinism of CPU multi-threaded inference on a 1B model this close
to a decision boundary, not something a keyword guardrail can pin down.

## Project structure

- `src/chatbot_rag/retrieval.py` - embedding, index build/save/load, and
  cosine-similarity search over the facts corpus. This is the only
  "grounding" mechanism in the whole project.
- `src/chatbot_rag/guardrails.py` - code-level refusals for specific query
  shapes that don't depend on the model choosing to comply with a prompt
  instruction (see its module docstring for why prompt-only instructions
  weren't reliable enough on their own).
- `src/chatbot_rag/fact_check.py` - a general, post-generation check on
  the answer itself: any named entity not present in the retrieved
  context gets replaced with a safe fallback, regardless of which
  question produced it (see "Measured results" above for why this exists
  alongside `guardrails.py` rather than instead of it).
- `src/chatbot_rag/pipeline.py` - the shared "retrieve, check each
  guardrail, generate, fact-check the result" logic used by
  `scripts/chat.py`, `scripts/evaluate.py`, and `scripts/chat_gguf.py`
  (and exercised directly by `tests/test_generation.py`), so there's
  exactly one implementation of that decision, not several copies that
  can drift apart. Deliberately backend-agnostic (takes a `generate_fn`
  closure, not a specific model type) so both generation backends below
  get identical guardrail/fact-check guarantees.
- `src/chatbot_rag/prompting.py` - the system prompt and per-turn message
  assembly (retrieved context is prepended to the user's question, not
  folded into the system message, so it reads as per-turn and disposable).
- `src/chatbot_rag/llm.py` - loads the stock base model/tokenizer and
  generates via the tokenizer's chat template (plain PyTorch/transformers
  - the default path). No PEFT/LoRA anywhere in this project.
- `src/chatbot_rag/gguf_llm.py` - the llama.cpp-backed equivalent of
  `llm.py` (~12.8 tok/s vs. ~1.6-2.9 tok/s - see "Speed" above), for
  `scripts/chat_gguf.py`.
- `src/chatbot_rag/api.py` - the FastAPI serving layer (see "Serving"
  above) - same `pipeline.answer()`, wrapped with auth, rate limiting,
  and error handling for actual deployment.
- `src/chatbot_rag/ratelimit.py` - the small in-memory rate limiter
  `api.py` uses.
- `src/chatbot_rag/utils.py` - environment/JSONL helpers.
- `src/chatbot_rag/db.py` - Postgres connection pool (`DATABASE_URL`).
- `src/chatbot_rag/sync.py` + `scripts/sync_facts.py` - incremental facts ->
  pgvector sync (re-embeds only changed facts, deletes removed ones, refuses a
  suspicious mass delete without `--force`).
- `scripts/migrate.py` + `supabase/migrations/` - schema.
- `.github/workflows/recrawl.yml` - weekly crawl + sync with no deploy.
- `scripts/chat.py` - interactive REPL (plain PyTorch backend).
- `scripts/chat_gguf.py` - interactive REPL (llama.cpp backend - see
  "Speed" above).
- `scripts/export_gguf.py` - converts the base model to a quantized GGUF
  file for `scripts/chat_gguf.py`.
- `scripts/serve.py` - runs `api.py` via uvicorn (see "Serving" above).
- `scripts/check_data_freshness.py` - reports how long since
  `facts.jsonl`/`test_cases.jsonl` last changed (see "Updating the
  dataset" below for what this does and doesn't tell you).
- `scripts/evaluate.py` - batch WITHOUT-RAG vs WITH-RAG comparison over
  `evaluation/test_cases.jsonl`, for open-ended qualitative review (see
  "Testing" above for the automated counterpart).
- `tests/` - `test_guardrails.py`, `test_fact_check.py`,
  `test_ratelimit.py`, `test_retrieval.py`, and the fast half of
  `test_api.py` (all fast, run by default) plus `test_generation.py` and
  the slow half of `test_api.py` (real-model end-to-end, `pytest -m slow`).
- `data/knowledge/facts.jsonl` - current public Technyx facts, refreshed by
  `scripts/crawl_technyx.py`; every record retains its source URL and
  `crawled_at` timestamp.
- `SECURITY.md` - the security review this project's serving layer and
  input-length guardrail respond to.
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`,
  `deploy/Caddyfile.example` - containerized deployment (see "Deploying"
  below).

## Deploying

```bash
cp .env.example .env   # fill in DATABASE_URL and API_KEY at minimum, HF_TOKEN if not using GGUF
docker compose up --build
```

Written carefully but **not build/run-tested** in the environment this
was authored in (no Docker available there) - `docker compose up --build`
should be the first thing you actually run with this, not something to
trust blindly. See the Dockerfile's own comments for what it does and why
(no model weights baked into the image - mount a GGUF file or pass
`HF_TOKEN` at runtime instead; non-root user; healthcheck against
`/health`).

`docker-compose.yml` sets `restart: unless-stopped` (process supervision
- the container restarts automatically on crash) and Docker's own
`json-file` log driver with rotation (10MB x 5 files, ~50MB retained) -
the standard, container-native way to handle log retention, rather than
managing log files from inside the app. TLS is optional
(`docker compose --profile tls up`, via `deploy/Caddyfile.example`) - skip
it entirely if deploying to a PaaS (Render/Fly.io/Railway/etc.) that
handles TLS itself.

**Embedding model memory footprint:** the default and the Dockerfile both use
`BAAI/bge-base-en-v1.5` (~420MB, 768-dim - matches the database column), which
fits Render's `starter` plan alongside PyTorch/FastAPI (not load-tested against
that plan's RAM limit - watch memory on first deploy). `bge-large-en-v1.5`
(1024-dim) or `all-MiniLM-L6-v2` (384-dim) need a migration to change the
column dimension first, then `sync_facts.py --full`.

## Updating the dataset

When the website changes, no rebuild or deploy is needed:

```bash
python scripts/crawl_technyx.py          # refresh data/knowledge/facts.jsonl from the sitemap
git diff data/knowledge/facts.jsonl      # review what changed
python scripts/sync_facts.py --dry-run   # preview: N inserted / re-embedded / deleted
python scripts/sync_facts.py             # apply; the live API picks it up immediately
```

`.github/workflows/recrawl.yml` does the crawl + sync weekly (and on demand)
using the `DATABASE_URL` repo secret. Answer cards are edited by hand in
`data/knowledge/answer_cards.jsonl` (then `sync_facts.py`); the crawler never
touches that file. If a page an answer card summarises changes, update the
card too - cards are returned verbatim.

The crawler respects `robots.txt`, resolves sitemap indexes, skips pages that
fail instead of aborting, and splits long paragraphs into sentence-boundary
facts (see `scripts/crawl_technyx.py`).

## Limitations

- The database is a new runtime dependency: if Supabase is unreachable the
  API cannot retrieve (requests fail rather than answer ungrounded), and each
  query pays one network round trip. Keep the API and database in the same
  region.
- Changing to an embedding model with a different dimension needs a migration
  (the column is `vector(768)`).
- No fine-tuning means no *learned* refusal behavior - `guardrails.py` and
  `fact_check.py` close most of that gap in code instead (see "Measured
  results" above), but a fabricated policy claim with no named entity in
  it, and the occasional wrong-designation answer on a decision-boundary
  question, remain open. A production version of this chatbot would still
  likely benefit from combining this retrieval layer with at least light
  fine-tuning (as `../lightweight-chatbot` does) for a more fundamental
  fix to those two.
- CPU inference with `google/gemma-3-1b-it` in plain PyTorch/transformers
  is slow (~1.6-2.9 tok/s measured) - **addressed**: see "Speed" above.
  `scripts/chat_gguf.py` measured ~12.8 tok/s on the same hardware via a
  quantized GGUF export, with identical guardrail/fact-check behavior to
  the plain-PyTorch path. The plain path (`scripts/chat.py`) remains the
  default since the GGUF tooling is optional/external (see "Speed").
- No CLI/script layer, before `api.py`, could actually be called by
  anything other than a trusted local operator - **addressed**: see
  "Serving" above for what `api.py` provides (auth, rate limiting, error
  handling without leaking internals) and what it deliberately doesn't
  (multi-tenant auth, distributed rate limiting - single shared key,
  single-process state, disclosed rather than hidden).
- No automated data-freshness tracking, and no real case-study content
  exists to ground "show me examples of your work"-type questions with
  (confirmed via a corpus + live-site audit - see `guardrails.py`'s
  `NO_CONCRETE_EXAMPLES_REFUSAL`). The second is a content problem for
  Technyx to solve, not an engineering one; the first is tracked as a
  schema note - see `data/knowledge/facts.jsonl`'s missing crawl-date
  field.
