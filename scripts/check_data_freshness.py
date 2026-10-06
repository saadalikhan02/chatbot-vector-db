#!/usr/bin/env python3
"""Report how long it's been since data/knowledge/facts.jsonl and
evaluation/test_cases.jsonl were last updated in this repo, and warn if
that's over a configurable threshold.

Facts refreshed by scripts/crawl_technyx.py carry a true UTC ``crawled_at``
timestamp. The script also reports git history for evaluation files, which do
not have a source crawl timestamp.

Usage:
    python scripts/check_data_freshness.py
    python scripts/check_data_freshness.py --warn-after-days 30
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TRACKED_FILES = [
    "data/knowledge/facts.jsonl",
    "data/knowledge/answer_cards.jsonl",
    "evaluation/test_cases.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--warn-after-days",
        type=int,
        default=90,
        help="Print a warning (and exit 1) if the most recently updated tracked file "
        "is older than this many days (default: 90)",
    )
    return parser.parse_args()


def last_git_update(path: str) -> datetime | None:
    """Return the commit timestamp of the most recent change to ``path``
    in this repo's history, or None if the file has no git history yet
    (e.g. never committed) - callers should treat that as "unknown", not
    "just updated"."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%aI", "--", path],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout.strip()
    if not output:
        return None
    return datetime.fromisoformat(output)


def latest_crawl(path: Path) -> datetime | None:
    timestamps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and (timestamp := json.loads(line).get("crawled_at")):
            timestamps.append(datetime.fromisoformat(timestamp))
    return max(timestamps, default=None)


def main() -> int:
    args = parse_args()
    now = datetime.now(timezone.utc)

    print("## Data freshness\n")

    oldest_age_days: float | None = None
    for rel_path in TRACKED_FILES:
        path = Path(rel_path)
        if not path.exists():
            print(f"{rel_path}: NOT FOUND")
            continue

        if rel_path == "data/knowledge/facts.jsonl":
            crawled_at = latest_crawl(path)
            if crawled_at is not None:
                age_days = (now - crawled_at).total_seconds() / 86400
                print(f"{rel_path}: crawled {crawled_at.date()} ({age_days:.0f} days ago)")
                oldest_age_days = age_days if oldest_age_days is None else max(oldest_age_days, age_days)
                continue

        updated_at = last_git_update(rel_path)
        if updated_at is None:
            print(f"{rel_path}: no git history found (uncommitted?)")
            continue

        age_days = (now - updated_at).total_seconds() / 86400
        print(f"{rel_path}: last changed {updated_at.date()} ({age_days:.0f} days ago)")
        oldest_age_days = age_days if oldest_age_days is None else max(oldest_age_days, age_days)

    print()
    if oldest_age_days is None:
        print("Could not determine freshness for any tracked file.")
        return 1

    if oldest_age_days > args.warn_after_days:
        print(
            f"WARNING: oldest tracked file is {oldest_age_days:.0f} days old "
            f"(threshold: {args.warn_after_days}). Consider re-crawling in "
            "scripts/crawl_technyx.py per README's 'Updating the dataset'."
        )
        return 1

    print(f"OK: all tracked files updated within the last {args.warn_after_days} days.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
