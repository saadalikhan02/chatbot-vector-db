#!/usr/bin/env python3
"""Sync data/knowledge/*.jsonl into the pgvector knowledge base (Supabase).

Run after scripts/crawl_technyx.py. Only changed facts are re-embedded, and
the running API sees the update immediately - no rebuild, no redeploy.

Usage:
    python scripts/sync_facts.py              # sync
    python scripts/sync_facts.py --dry-run    # show what would change
    python scripts/sync_facts.py --full       # re-embed everything
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatbot_rag.retrieval import load_facts  # noqa: E402
from chatbot_rag.sync import sync_facts  # noqa: E402
from chatbot_rag.utils import load_dotenv_if_present  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--facts", default="data/knowledge/facts.jsonl", type=Path, help="crawled facts")
    parser.add_argument(
        "--answer-cards",
        default="data/knowledge/answer_cards.jsonl",
        type=Path,
        help="hand-curated answer cards (never touched by the crawler)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    parser.add_argument("--full", action="store_true", help="re-embed every fact")
    parser.add_argument("--force", action="store_true", help="allow deleting more than half the existing facts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv_if_present()
    for path in (args.facts, args.answer_cards):
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 1

    facts = load_facts(args.facts, args.answer_cards)
    t0 = time.time()
    result = sync_facts(facts, full=args.full, dry_run=args.dry_run, force=args.force)
    print(f"{result.summary()} ({len(facts)} facts in source, {time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
