# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RAG chatbot for Technyx Systems whose knowledge base lives in Postgres/pgvector (Supabase). Sibling of `../chatbot-rag` (numpy-file index). No fine-tuning: a stock instruction-tuned model (default `google/gemma-3-1b-it`, or a remote OpenAI-compatible API) is grounded purely by retrieval over crawled website facts, plus code-level guardrails. Python >=3.11, package in `src/chatbot_rag`, entry points in `scripts/`.

## Commands

```bash
pip install -r requirements.txt && pip install -e '.[dev]'   # serving extras: requirements-serving.txt

python scripts/migrate.py         # create/upgrade the DB schema (needs DATABASE_URL)
python scripts/sync_facts.py      # facts.jsonl + answer_cards.jsonl -> pgvector (incremental; --dry-run, --full)
pytest                            # fast suite (default; addopts = -m 'not slow'); uses a throwaway pgvector DB (pgserver, or TEST_DATABASE_URL) + embedding model
pytest tests/test_retrieval.py::TestAnswerCards   # single test/class
pytest -m slow                    # real-model end-to-end (needs HF_TOKEN + accepted Gemma license, minutes)
ruff check src tests scripts && ruff format --check src tests scripts   # CI lint (line length 120)
mypy                              # configured for src/ only
python scripts/score_evaluation.py --min-retrieval-hit-rate 0.5          # CI retrieval gate

python scripts/chat.py [--no-retrieval]     # REPL, plain PyTorch backend
python scripts/chat_gguf.py --model <gguf>  # REPL, llama.cpp backend
python scripts/serve.py                     # FastAPI (needs API_KEY; fails closed without it)
python scripts/crawl_technyx.py             # refresh facts.jsonl from the sitemap, then sync_facts.py (no redeploy)
```

Config is via env vars (see `.env.example`): `LLM_BACKEND` (`plain`|`gguf`|`api`), `EMBEDDING_MODEL`, `API_KEY`, `CORS_ORIGINS`, etc.

## Architecture

Every entry point (the REPLs, `evaluate.py`, the API, slow tests) goes through `pipeline.answer()` — keep logic there rather than duplicating it. Flow:

1. `guardrails.is_input_too_long` → 2. `retrieval` (embed query folded with prior-turn history via `build_retrieval_query`, pgvector nearest-neighbour SQL query + Python-side ranking boosts) → 3. code-level refusals (`guardrails.py`: no context, concrete-example requests, employee-name requests) → 4. **answer cards**: facts with an `answer_card_keywords` field (hand-curated in `data/knowledge/answer_cards.jsonl`, separate from the crawler's `facts.jsonl`) bypass generation entirely and are returned verbatim (`_answer_card_for_query` in `pipeline.py` has subtle follow-up/specificity rules) → 5. `generate_fn(messages)` → 6. `fact_check.find_unsupported_claims` replaces the response with a fallback if it names entities absent from the retrieved context.

Key design points:
- `pipeline.py` is backend-agnostic: it takes a `generate_fn` closure. `llm.py` (transformers), `gguf_llm.py` (llama.cpp), and `api_llm.py` (OpenAI-compatible HTTP) each build one; `api.py` picks it at startup from `LLM_BACKEND`.
- Knowledge base = `public.facts` (pgvector, schema in `supabase/migrations`, applied by `scripts/migrate.py`). `index_meta` stores the embedding model; `open_index()` refuses a mismatched `EMBEDDING_MODEL`, so changing it requires `sync_facts.py --full`. Column is `vector(768)` (bge-base, the default); other dimensions need a migration. Supported models/pooling live in `_MODEL_CONFIGS` in `retrieval.py`.
- `search()` fetches nearest neighbours + all boost-eligible facts in one query, then applies the boosts in Python (results match an exact scan). `sync.py` re-embeds only facts whose `embedding_hash` changed and refuses to delete >50% of facts without `--force`.
- `api.py`: shared-key auth (`X-API-Key`), per-key and per-visitor rate limiting (`ratelimit.py`), in-memory sessions (`sessions.py`), `/v1/chat`, SSE `/v1/chat/stream` for the website widget, `/metrics`. Generation is serialized behind `app.state.answer_lock` — required for correctness (concurrent llama.cpp calls corrupt state; see `SECURITY.md`). State is single-process; scaling out needs shared stores.
- Generic error responses only; full detail goes to server logs. The `chatbot_rag.flagged` logger records refusals/blocked responses for review.

## Gotchas

- Tests must never use `DATABASE_URL` (production Supabase): `tests/conftest.py` uses `TEST_DATABASE_URL` or a private `pgserver` instance.
- `DATABASE_URL`: use the direct/Session pooler (5432) for migrate/sync; the API also works on the Transaction pooler (6543) because prepared statements are disabled in `db.py`.
- Slow tests and `tests/test_api.py`'s app-booting half are marked `slow` and excluded by default; run them when touching `retrieval.py`, `guardrails.py`, `prompting.py`, `fact_check.py`, or `api.py`.
- `tests/test_check_data_freshness.py` needs git history for the tracked data files.
- The Dockerfile has not been build-tested per its own header comment.
