#!/usr/bin/env python3
"""Generate and compare WITHOUT-RAG vs WITH-RAG responses on a fixed set of
test cases.

This is a qualitative, side-by-side comparison intended to show what
retrieval actually buys you over the same base model answering from its own
(unmodified) memory - there is no fine-tuning in this project, so any
improvement here comes entirely from retrieval. There is no ground-truth
scoring; read the responses yourself. For an automated pass/fail check, see
tests/test_generation.py instead.

Usage:
    python scripts/evaluate.py --test-cases evaluation/test_cases.jsonl

    # RAG only (skip the slower no-retrieval comparison column):
    python scripts/evaluate.py --skip-no-rag
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatbot_rag.llm import load_model, load_tokenizer, make_generate_fn  # noqa: E402
from chatbot_rag.pipeline import answer  # noqa: E402
from chatbot_rag.prompting import SYSTEM_PROMPT  # noqa: E402
from chatbot_rag.retrieval import open_index  # noqa: E402
from chatbot_rag.utils import (  # noqa: E402
    get_hf_token,
    load_dotenv_if_present,
    print_environment_report,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="google/gemma-3-1b-it")
    parser.add_argument("--test-cases", default="evaluation/test_cases.jsonl", type=Path)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--skip-no-rag",
        action="store_true",
        help="Only run the RAG column (skip the no-retrieval comparison) - "
        "recommended on CPU, since generating both columns doubles the wait.",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Max retrieved facts per question")
    return parser.parse_args()


def main() -> int:
    load_dotenv_if_present()
    args = parse_args()
    print_environment_report()

    if not args.test_cases.exists():
        print(f"ERROR: test cases file not found: {args.test_cases}", file=sys.stderr)
        return 1

    test_cases = read_jsonl(args.test_cases)
    print(f"\nLoaded {len(test_cases)} test case(s) from {args.test_cases}")

    hf_token = get_hf_token()
    tokenizer = load_tokenizer(args.model_name, hf_token=hf_token)

    print(f"\nLoading base model: {args.model_name}")
    model = load_model(args.model_name, hf_token=hf_token)

    print("Connecting to the knowledge base (DATABASE_URL)")
    retrieval_index = open_index()

    generate_fn = make_generate_fn(model, tokenizer, max_new_tokens=args.max_new_tokens)

    print("\n" + "=" * 80)
    for i, case in enumerate(test_cases, start=1):
        category = case.get("category", "uncategorized")
        user_input = case["user_input"]
        # Always the RAG-aware default (args.system_prompt), never a
        # per-case override: test_cases.jsonl was copied from
        # ../lightweight-chatbot, a fine-tuned project whose system prompt
        # never mentions the "Relevant information" block this project
        # relies on for grounding. A per-case "system_prompt" field letting
        # that stale, non-RAG-aware prompt silently win was measured to
        # produce wrong answers even when retrieval found the right facts
        # (e.g. it declined to mention a real testimonial it had just
        # retrieved) - see README's "Measured results".
        system_prompt = args.system_prompt
        history = case.get("history")

        print(f"\n[{i}/{len(test_cases)}] category: {category}")
        if history:
            print("(follow-up question; prior turns omitted for brevity)")
        print(f"Q: {user_input}")

        if not args.skip_no_rag:
            # Deliberately ungated - this column is the raw, unmodified
            # baseline for comparison, so it never uses the guardrails.
            no_rag_result = answer(
                user_input,
                generate_fn,
                retrieval_index=None,
                system_prompt=system_prompt,
                history=history,
            )
            print(f"\nWITHOUT RAG:\n  {no_rag_result.response}")

        rag_result = answer(
            user_input,
            generate_fn,
            retrieval_index=retrieval_index,
            system_prompt=system_prompt,
            history=history,
            top_k=args.top_k,
        )
        print(
            f"(retrieved context: "
            f"{'yes, ' + str(len(rag_result.retrieved)) + ' fact(s)' if rag_result.context_block else 'none found'})"
        )
        label = "WITH RAG" if rag_result.generated else "WITH RAG (declined without calling the model)"
        print(f"\n{label}:\n  {rag_result.response}")
        print("\n" + "-" * 80)

    print(
        "\nReminder: this is a qualitative comparison, not an automated score. "
        "Read the responses yourself and judge whether retrieval fixed fact "
        "accuracy and reduced hallucination compared to the base model alone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
