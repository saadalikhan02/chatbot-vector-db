# Test suite

Run `pytest` for the fast unit and retrieval suite (needs no setup: a private pgvector Postgres is started via `pgserver`, or set `TEST_DATABASE_URL`). Run `pytest -m slow` only
with approved Gemma access; it loads the real model and generates answers.
