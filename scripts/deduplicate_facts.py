#!/usr/bin/env python3
"""Deduplicate the crawled corpus by exact ``fact`` text.

For duplicate text, retain the record with the most populated source metadata;
the text is what retrieval returns, while richer provenance is more useful for
review/debugging. Run ``scripts/build_index.py`` afterwards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def metadata_score(record: dict[str, Any]) -> tuple[int, int]:
    fields = [key for key, value in record.items() if key != "fact" and value not in (None, "", [], {})]
    return len(fields), sum(len(str(record[key])) for key in fields)


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for record in records:
        fact = record["fact"]
        if fact not in chosen or metadata_score(record) > metadata_score(chosen[fact]):
            chosen[fact] = record
    return list(chosen.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/knowledge/facts.jsonl"))
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    deduplicated = deduplicate(records)
    args.input.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in deduplicated), encoding="utf-8"
    )
    print(f"Kept {len(deduplicated)} unique facts; removed {len(records) - len(deduplicated)} duplicates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
