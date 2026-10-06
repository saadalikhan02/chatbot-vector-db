"""Postgres (Supabase) connection handling for the pgvector knowledge base.

One place owns the connection settings so the API, the sync job and the
scripts all behave the same way:

- ``DATABASE_URL`` is the only required setting. For Supabase use the
  *Session pooler* string (port 5432) or the direct connection; the
  *Transaction pooler* (port 6543) also works for the API because prepared
  statements are disabled here (``prepare_threshold=None``).
- Vectors are passed as numpy arrays via the pgvector adapter.
"""

from __future__ import annotations

import os

from psycopg_pool import ConnectionPool


def get_database_url(database_url: str | None = None) -> str:
    url = database_url or os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Point it at your Postgres/Supabase database "
            "(see .env.example and the README's 'Database' section)."
        )
    return url


def _configure(conn) -> None:
    from pgvector.psycopg import register_vector

    register_vector(conn)


def create_pool(database_url: str | None = None, *, min_size: int = 1, max_size: int = 4) -> ConnectionPool:
    """Open a small connection pool. Keep ``max_size`` low: Supabase poolers
    and small plans cap total connections, and generation is serialized per
    process anyway."""
    pool = ConnectionPool(
        get_database_url(database_url),
        min_size=min_size,
        max_size=max_size,
        configure=_configure,
        kwargs={"prepare_threshold": None},  # required for pgbouncer/Supavisor transaction mode
        open=False,
    )
    pool.open(wait=True, timeout=30)
    return pool
