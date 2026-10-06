"""Sync the facts JSONL files into the pgvector knowledge base.

Replaces the old "rebuild the whole .npz and redeploy" step. Only facts whose
embedded text (or embedding model) changed are re-embedded; unchanged facts
just get their metadata refreshed; facts that disappeared from the source are
deleted. All of it happens in one transaction, so a failed sync never leaves
the live knowledge base half-updated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .db import create_pool
from .retrieval import (
    EMBEDDING_MODEL_NAME,
    _embedding_text,
    _model_config,
    embed_texts,
    embedding_hash,
    is_authoritative_identity_fact,
)

# Refuse to delete more than this share of existing facts in one sync unless
# forced - protects against a broken crawl wiping the knowledge base.
MAX_DELETE_FRACTION = 0.5

_COLUMN_FIELDS = {
    "fact_id",
    "fact",
    "fact_type",
    "source_url",
    "source_page_title",
    "source_section",
    "source_excerpt",
    "confidence",
    "crawled_at",
    "retrieval_text",
    "answer_card_keywords",
}


@dataclass
class SyncResult:
    inserted: int = 0
    re_embedded: int = 0
    metadata_only: int = 0
    deleted: int = 0
    dry_run: bool = False

    def summary(self) -> str:
        prefix = "[dry run] would have: " if self.dry_run else ""
        return (
            f"{prefix}{self.inserted} inserted, {self.re_embedded} re-embedded, "
            f"{self.metadata_only} metadata-only updates, {self.deleted} deleted"
        )


def _row_params(fact: dict[str, Any], digest: str) -> dict[str, Any]:
    return {
        "fact_id": fact["fact_id"],
        "fact": fact["fact"],
        "fact_type": fact.get("fact_type") or "other",
        "source_url": fact.get("source_url"),
        "source_page_title": fact.get("source_page_title"),
        "source_section": fact.get("source_section"),
        "source_excerpt": fact.get("source_excerpt"),
        "confidence": fact.get("confidence"),
        "crawled_at": fact.get("crawled_at"),
        "retrieval_text": fact.get("retrieval_text"),
        "answer_card_keywords": fact.get("answer_card_keywords"),
        "is_authoritative": is_authoritative_identity_fact(fact),
        "extra": json.dumps({k: v for k, v in fact.items() if k not in _COLUMN_FIELDS}),
        "embedding_hash": digest,
    }


_UPSERT_SQL = """
insert into public.facts (fact_id, fact, fact_type, source_url, source_page_title, source_section,
  source_excerpt, confidence, crawled_at, retrieval_text, answer_card_keywords, is_authoritative,
  extra, embedding_hash, embedding)
values (%(fact_id)s, %(fact)s, %(fact_type)s, %(source_url)s, %(source_page_title)s, %(source_section)s,
  %(source_excerpt)s, %(confidence)s, %(crawled_at)s, %(retrieval_text)s, %(answer_card_keywords)s,
  %(is_authoritative)s, %(extra)s::jsonb, %(embedding_hash)s, %(embedding)s)
on conflict (fact_id) do update set
  fact = excluded.fact, fact_type = excluded.fact_type, source_url = excluded.source_url,
  source_page_title = excluded.source_page_title, source_section = excluded.source_section,
  source_excerpt = excluded.source_excerpt, confidence = excluded.confidence,
  crawled_at = excluded.crawled_at, retrieval_text = excluded.retrieval_text,
  answer_card_keywords = excluded.answer_card_keywords, is_authoritative = excluded.is_authoritative,
  extra = excluded.extra, embedding_hash = excluded.embedding_hash,
  embedding = excluded.embedding, updated_at = now()
"""

_METADATA_SQL = """
update public.facts set
  fact = %(fact)s, fact_type = %(fact_type)s, source_url = %(source_url)s,
  source_page_title = %(source_page_title)s, source_section = %(source_section)s,
  source_excerpt = %(source_excerpt)s, confidence = %(confidence)s, crawled_at = %(crawled_at)s,
  retrieval_text = %(retrieval_text)s, answer_card_keywords = %(answer_card_keywords)s,
  is_authoritative = %(is_authoritative)s, extra = %(extra)s::jsonb, updated_at = now()
where fact_id = %(fact_id)s
"""


def sync_facts(
    facts: list[dict[str, Any]],
    database_url: str | None = None,
    *,
    full: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> SyncResult:
    """Make the database match ``facts`` exactly. ``full`` re-embeds
    everything; ``force`` allows deleting more than MAX_DELETE_FRACTION."""
    if not facts:
        raise ValueError("Refusing to sync an empty fact list.")
    desired = {f["fact_id"]: (f, embedding_hash(f)) for f in facts}
    config = _model_config()

    pool = create_pool(database_url, max_size=2)
    try:
        with pool.connection() as conn:
            with conn.transaction():
                existing = dict(conn.execute("select fact_id, embedding_hash from public.facts").fetchall())
                col_dim = conn.execute(
                    "select atttypmod from pg_attribute "
                    "where attrelid = 'public.facts'::regclass and attname = 'embedding'"
                ).fetchone()
                if col_dim and col_dim[0] != config["dimensions"]:
                    raise RuntimeError(
                        f"{EMBEDDING_MODEL_NAME} produces {config['dimensions']}-dim vectors but "
                        f"public.facts.embedding is vector({col_dim[0]}). Add a migration that alters the "
                        f"column and recreates the HNSW index before switching models."
                    )

                to_embed = [fid for fid, (_, digest) in desired.items() if full or existing.get(fid) != digest]
                to_embed_set = set(to_embed)
                metadata_only = [fid for fid in desired if fid in existing and fid not in to_embed_set]
                to_delete = [fid for fid in existing if fid not in desired]

                result = SyncResult(
                    inserted=sum(1 for fid in to_embed if fid not in existing),
                    re_embedded=sum(1 for fid in to_embed if fid in existing),
                    metadata_only=len(metadata_only),
                    deleted=len(to_delete),
                    dry_run=dry_run,
                )
                if existing and len(to_delete) > MAX_DELETE_FRACTION * len(existing) and not force:
                    raise RuntimeError(
                        f"Sync would delete {len(to_delete)} of {len(existing)} facts - looks like a broken "
                        f"crawl. Re-run with --force if that is really intended."
                    )
                if dry_run:
                    return result

                if to_embed:
                    vectors = embed_texts([_embedding_text(desired[fid][0]) for fid in to_embed])
                    # executemany pipelines the statements: one network round trip
                    # batch instead of one per fact (matters on a remote database).
                    upserts = [
                        {**_row_params(desired[fid][0], desired[fid][1]), "embedding": np.asarray(vec)}
                        for fid, vec in zip(to_embed, vectors, strict=True)
                    ]
                    with conn.cursor() as cur:
                        cur.executemany(_UPSERT_SQL, upserts)
                if metadata_only:
                    with conn.cursor() as cur:
                        cur.executemany(
                            _METADATA_SQL, [_row_params(desired[fid][0], desired[fid][1]) for fid in metadata_only]
                        )
                if to_delete:
                    conn.execute("delete from public.facts where fact_id = any(%s)", (to_delete,))
                conn.execute(
                    "insert into public.index_meta (key, value) values ('embedding_model', %s), "
                    "('last_synced_at', now()::text) "
                    "on conflict (key) do update set value = excluded.value",
                    (EMBEDDING_MODEL_NAME,),
                )
        return result
    finally:
        pool.close()
