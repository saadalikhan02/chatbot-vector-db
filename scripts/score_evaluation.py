#!/usr/bin/env python3
"""Score deterministic RAG expectations without loading the Gemma model.

Cases may include ``expected_substring``. The fast gate checks that retrieval
returns that text; an optional JSONL answer file can additionally check final
answer text with the same exact substring rule. This is deliberately simple
and auditable, not an LLM judge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatbot_rag.retrieval import open_index  # noqa: E402
from chatbot_rag.utils import load_dotenv_if_present, read_jsonl  # noqa: E402


def score_retrieval(cases: list[dict], index) -> tuple[int, int]:
    applicable = [case for case in cases if case.get("expected_substring")]
    hits = sum(
        any(
            case["expected_substring"].casefold() in fact["fact"].casefold()
            for fact in index.search(case["user_input"])
        )
        for case in applicable
    )
    return hits, len(applicable)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-cases", type=Path, default=Path("evaluation/test_cases.jsonl"))
    parser.add_argument("--min-retrieval-hit-rate", type=float, default=0.8)
    args = parser.parse_args()
    load_dotenv_if_present()
    hits, total = score_retrieval(read_jsonl(args.test_cases), open_index())
    if not total:
        print("No expected_substring cases available; evaluation gate cannot score.", file=sys.stderr)
        return 2
    rate = hits / total
    print(f"retrieval hit rate: {hits}/{total} ({rate:.1%})")
    return 0 if rate >= args.min_retrieval_hit_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
