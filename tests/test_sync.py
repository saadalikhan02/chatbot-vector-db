"""Tests for the facts -> pgvector sync (src/chatbot_rag/sync.py).

Uses its own throwaway schema content in the shared test database: each test
syncs a tiny fact list, so it never depends on the full corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chatbot_rag.sync import sync_facts  # noqa: E402


def _fact(fact_id: str, text: str, **extra) -> dict:
    return {
        "fact_id": fact_id,
        "fact": text,
        "fact_type": "other",
        "source_url": "https://example.com",
        "source_page_title": "Example",
        "source_section": "Overview",
        "crawled_at": "2026-10-01T00:00:00+00:00",
        **extra,
    }


@pytest.fixture()
def db(test_database_url):
    from migrate import apply_migrations

    apply_migrations(test_database_url)
    with psycopg.connect(test_database_url, autocommit=True) as conn:
        conn.execute("truncate public.facts, public.index_meta")
    yield test_database_url
    with psycopg.connect(test_database_url, autocommit=True) as conn:
        conn.execute("truncate public.facts, public.index_meta")


def _rows(url):
    with psycopg.connect(url) as conn:
        return {
            r[0]: r
            for r in conn.execute("select fact_id, embedding_hash, updated_at, is_authoritative from public.facts")
        }


def test_first_sync_inserts_everything(db):
    result = sync_facts([_fact("a", "We build websites."), _fact("b", "Our head office is in Dubai.")], db)
    assert (result.inserted, result.re_embedded, result.metadata_only, result.deleted) == (2, 0, 0, 0)
    rows = _rows(db)
    assert set(rows) == {"a", "b"}
    assert rows["b"][3] is True and rows["a"][3] is False


def test_unchanged_facts_are_not_re_embedded(db):
    facts = [_fact("a", "We build websites."), _fact("b", "We build apps.")]
    sync_facts(facts, db)
    before = _rows(db)
    result = sync_facts(facts, db)
    assert (result.inserted, result.re_embedded, result.metadata_only) == (0, 0, 2)
    assert {k: v[1] for k, v in _rows(db).items()} == {k: v[1] for k, v in before.items()}


def test_changed_text_is_re_embedded_and_removed_fact_is_deleted(db):
    sync_facts([_fact("a", "We build websites."), _fact("b", "We build apps."), _fact("c", "We do SEO.")], db)
    before = _rows(db)
    result = sync_facts([_fact("a", "We build fast websites."), _fact("b", "We build apps.")], db, force=True)
    assert (result.inserted, result.re_embedded, result.metadata_only, result.deleted) == (0, 1, 1, 1)
    after = _rows(db)
    assert set(after) == {"a", "b"}
    assert after["a"][1] != before["a"][1]
    assert after["b"][1] == before["b"][1]


def test_dry_run_changes_nothing(db):
    sync_facts([_fact("a", "We build websites.")], db)
    result = sync_facts([_fact("a", "Changed."), _fact("b", "New.")], db, dry_run=True)
    assert result.dry_run and result.inserted == 1 and result.re_embedded == 1
    assert set(_rows(db)) == {"a"}


def test_mass_delete_is_refused_without_force(db):
    sync_facts([_fact("a", "One."), _fact("b", "Two."), _fact("c", "Three.")], db)
    with pytest.raises(RuntimeError, match="broken crawl"):
        sync_facts([_fact("a", "One.")], db)
    assert set(_rows(db)) == {"a", "b", "c"}


def test_empty_source_is_refused(db):
    with pytest.raises(ValueError):
        sync_facts([], db)


def test_unknown_fields_are_kept_in_extra(db):
    sync_facts([_fact("a", "We build websites.", custom_field="kept")], db)
    with psycopg.connect(db) as conn:
        assert conn.execute("select extra->>'custom_field' from public.facts").fetchone()[0] == "kept"
