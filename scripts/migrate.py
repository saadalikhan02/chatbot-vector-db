#!/usr/bin/env python3
"""Apply the SQL files in supabase/migrations to the database in DATABASE_URL.

Applied files are recorded in ``public._app_migrations`` so each runs once.
Use this instead of (not alongside) the Supabase CLI's ``db push`` so there is
a single record of what has been applied. Works on Supabase, local Postgres and
CI. Use the direct connection or the Session pooler (port 5432) for DDL.

Usage:
    python scripts/migrate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatbot_rag.db import get_database_url  # noqa: E402
from chatbot_rag.utils import load_dotenv_if_present  # noqa: E402

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "supabase" / "migrations"


def apply_migrations(database_url: str, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    applied_now: list[str] = []
    with psycopg.connect(database_url, autocommit=True, prepare_threshold=None) as conn:
        conn.execute(
            "create table if not exists public._app_migrations "
            "(name text primary key, applied_at timestamptz not null default now())"
        )
        conn.execute("alter table public._app_migrations enable row level security")
        done = {row[0] for row in conn.execute("select name from public._app_migrations")}
        for path in sorted(migrations_dir.glob("*.sql")):
            if path.name in done:
                continue
            with conn.transaction():
                conn.execute(path.read_text(encoding="utf-8"))
                conn.execute("insert into public._app_migrations (name) values (%s)", (path.name,))
            applied_now.append(path.name)
    return applied_now


def main() -> int:
    load_dotenv_if_present()
    applied = apply_migrations(get_database_url())
    print("Applied: " + ", ".join(applied) if applied else "Database is up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
