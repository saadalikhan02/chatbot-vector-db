"""Shared pytest fixtures.

Two tiers, matching the split in tests/README.md:

- Fast fixtures (``retrieval_index``) load only the configured embedding
  model (``EMBEDDING_MODEL``, see ``chatbot_rag.retrieval``) and a throwaway
  pgvector database - fine to use in every test.
- Slow fixtures (``model_and_tokenizer``, in test_generation.py's own
  conftest) load the full 1B-parameter Gemma model and are only pulled in
  by tests marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: loads the full 1B LLM and runs real generation (minutes, not seconds)")


@pytest.fixture(scope="session")
def test_database_url(tmp_path_factory):
    """A throwaway Postgres+pgvector database, never the real one.

    CI/dev can point ``TEST_DATABASE_URL`` at any disposable pgvector
    database (e.g. the ``pgvector/pgvector`` Docker image). Without it, the
    ``pgserver`` dev dependency starts a private local Postgres. The tests
    deliberately ignore ``DATABASE_URL`` so they can never sync into or wipe
    the production Supabase database.
    """
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        try:
            import pgserver
        except ImportError:
            pytest.skip("Set TEST_DATABASE_URL or `pip install pgserver` (dev extra) to run database tests")
        server = pgserver.get_server(tmp_path_factory.mktemp("pgdata"))
        url = server.get_uri()
    elif url == os.environ.get("DATABASE_URL"):
        pytest.fail("TEST_DATABASE_URL must not be the same as DATABASE_URL")
    return url


@pytest.fixture(scope="session")
def retrieval_index(test_database_url):
    """The real knowledge base (migrations + facts + answer cards) in the test database."""
    from chatbot_rag.retrieval import load_facts, open_index
    from chatbot_rag.sync import sync_facts

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from migrate import apply_migrations

    apply_migrations(test_database_url)
    facts = load_facts(
        REPO_ROOT / "data" / "knowledge" / "facts.jsonl", REPO_ROOT / "data" / "knowledge" / "answer_cards.jsonl"
    )
    sync_facts(facts, test_database_url)
    # api.py's lifespan reads DATABASE_URL; the slow API tests boot the real app.
    os.environ["DATABASE_URL"] = test_database_url
    index = open_index(test_database_url)
    yield index
    index.close()
