#!/usr/bin/env python3
"""Run evaluation/test_cases.jsonl through the production API backend
(chatbot_rag.api_llm - the same generate_fn LLM_BACKEND=api uses in
production) against one configured hosted model, WITH RAG only.

This compares hosted-model answer quality on top of the same retrieval/
guardrail/fact-check pipeline every backend shares - it does not measure
retrieval's marginal value (see scripts/evaluate.py for the WITHOUT-RAG vs
WITH-RAG comparison against the local base model instead).

Qualitative by design, like scripts/evaluate.py: only 2/53 cases in the
current eval set carry an expected_substring, too few for a general
auto-grader. Read the saved transcript yourself; run this once per model
with --output pointed at different files, then diff them.

Usage:
    # Uses LLM_API_BASE_URL/LLM_API_KEY/LLM_API_MODEL from .env - the
    # currently configured production backend:
    python scripts/evaluate_api.py --output evaluation/runs/gemini-flash.txt

    # Compare a different model - flags override the .env values:
    python scripts/evaluate_api.py --base-url https://api.openai.com/v1 \
        --model gpt-4o-mini --api-key "$OPENAI_API_KEY" \
        --output evaluation/runs/gpt-4o-mini.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatbot_rag.api_llm import LLMProviderError, make_generate_fn  # noqa: E402
from chatbot_rag.pipeline import answer  # noqa: E402
from chatbot_rag.prompting import SYSTEM_PROMPT  # noqa: E402
from chatbot_rag.retrieval import open_index  # noqa: E402
from chatbot_rag.utils import load_dotenv_if_present, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=None, help="Defaults to LLM_API_BASE_URL from .env/env")
    parser.add_argument("--api-key", default=None, help="Defaults to LLM_API_KEY from .env/env")
    parser.add_argument("--model", default=None, help="Defaults to LLM_API_MODEL from .env/env")
    parser.add_argument("--test-cases", default="evaluation/test_cases.jsonl", type=Path)
    parser.add_argument("--top-k", type=int, default=8, help="Max retrieved facts per question")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=4.0,
        help="Pause between generated (non-declined) calls - a bulk eval run fires far more "
        "requests per minute than a real chat session, so provider rate limits (429) bite "
        "without this (measured: 1.0s wasn't enough against a free-tier Gemini quota - 21/53 "
        "calls got rate limited). Same politeness-delay idea as scripts/crawl_technyx.py.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Retries on a 429 (rate limit) response, with exponential backoff",
    )
    parser.add_argument("--output", type=Path, default=None, help="Also save the full transcript to this file")
    return parser.parse_args()


def _answer_with_retry(user_input, generate_fn, *, retrieval_index, system_prompt, history, top_k, max_retries):
    """Call pipeline.answer(), retrying on a 429 from the provider with
    exponential backoff (starting at max(retry-after, 15s), doubling, capped
    at 60s). Measured necessary: a free-tier RPM quota resets per-minute, so
    the provider's own retry-after hint (often ~10s) isn't long enough on
    its own - repeated 10s retries can all land inside the same exhausted
    window. Bulk sequential calls hit this in a way a single interactive
    chat turn never would - see --delay-seconds for the other half of the
    fix (spacing calls out to begin with)."""
    wait = 15.0
    for attempt in range(max_retries + 1):
        try:
            return answer(
                user_input,
                generate_fn,
                retrieval_index=retrieval_index,
                system_prompt=system_prompt,
                history=history,
                top_k=top_k,
            )
        except LLMProviderError as exc:
            if exc.http_status != 429 or attempt == max_retries:
                raise
            wait = min(max(wait, float(exc.retry_after or 0)), 60.0)
            print(f"  (rate limited, retry {attempt + 1}/{max_retries} after {wait:.0f}s)", file=sys.stderr)
            time.sleep(wait)
            wait *= 2
    raise AssertionError("unreachable")  # loop always returns or raises


def main() -> int:
    load_dotenv_if_present()
    args = parse_args()

    base_url = args.base_url or os.environ.get("LLM_API_BASE_URL")
    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    model = args.model or os.environ.get("LLM_API_MODEL")
    if not (base_url and api_key and model):
        print(
            "ERROR: need base_url/api_key/model, via --base-url/--api-key/--model or "
            "LLM_API_BASE_URL/LLM_API_KEY/LLM_API_MODEL in .env.",
            file=sys.stderr,
        )
        return 1

    if not args.test_cases.exists():
        print(f"ERROR: test cases file not found: {args.test_cases}", file=sys.stderr)
        return 1
    test_cases = read_jsonl(args.test_cases)

    retrieval_index = open_index()

    generate_fn = make_generate_fn(base_url=base_url, api_key=api_key, model=model, max_new_tokens=args.max_new_tokens)

    lines: list[str] = [f"Model: {model}", f"Base URL: {base_url}", "=" * 80]
    declined = 0
    errors = 0
    substring_checked = 0
    substring_passed = 0

    for i, case in enumerate(test_cases, start=1):
        category = case.get("category", "uncategorized")
        user_input = case["user_input"]
        history = case.get("history")

        try:
            result = _answer_with_retry(
                user_input,
                generate_fn,
                retrieval_index=retrieval_index,
                system_prompt=SYSTEM_PROMPT,
                history=history,
                top_k=args.top_k,
                max_retries=args.max_retries,
            )
        except Exception as exc:  # provider/network error - log and keep going, don't lose the rest of the run
            errors += 1
            lines.append(f"[{i}/{len(test_cases)}] category={category}\nQ: {user_input}\nERROR: {exc}")
            lines.append("-" * 80)
            continue

        if not result.generated:
            declined += 1
        elif args.delay_seconds:
            time.sleep(args.delay_seconds)
        label = "DECLINED (no model call)" if not result.generated else "GENERATED"
        entry = [
            f"[{i}/{len(test_cases)}] category={category} | {label}",
            f"Q: {user_input}",
            f"retrieved: {len(result.retrieved)} fact(s)" if result.context_block else "retrieved: none",
            f"A: {result.response}",
        ]
        expected = case.get("expected_substring")
        if expected:
            substring_checked += 1
            ok = expected.casefold() in result.response.casefold()
            substring_passed += ok
            entry.append(f"expected_substring check ({expected!r}): {'PASS' if ok else 'FAIL'}")
        lines.append("\n".join(entry))
        lines.append("-" * 80)

    summary = (
        f"\n{len(test_cases)} case(s) | {declined} declined without a model call | "
        f"{len(test_cases) - declined - errors} generated | {errors} error(s)"
    )
    if substring_checked:
        summary += f"\nexpected_substring: {substring_passed}/{substring_checked} passed"
    lines.append(summary)

    output_text = "\n".join(lines)
    print(output_text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n", encoding="utf-8")
        print(f"\nSaved transcript to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
