#!/usr/bin/env python3
"""Interactively chat with the base model, grounded by retrieval (RAG).

No fine-tuning involved: this loads the stock `google/gemma-3-1b-it`
instruction-tuned model and relies entirely on retrieved facts injected
into the prompt for company-specific accuracy.

Usage:
    python scripts/chat.py

    # Compare against the same model with retrieval disabled:
    python scripts/chat.py --no-retrieval

Type 'exit' or 'quit' to end the session.
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
from chatbot_rag.utils import get_hf_token, load_dotenv_if_present, print_environment_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="google/gemma-3-1b-it", help="Base model to load")
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT, help="System prompt to prepend")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Don't keep prior turns in context (each question is independent)",
    )
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Disable RAG - answer from the base model's own memory only (for comparison)",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Max retrieved facts per question")
    return parser.parse_args()


def main() -> int:
    load_dotenv_if_present()
    args = parse_args()
    print_environment_report()

    hf_token = get_hf_token()

    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = load_tokenizer(args.model_name, hf_token=hf_token)

    print(f"Loading base model: {args.model_name} (this may take a while on CPU)")
    model = load_model(args.model_name, hf_token=hf_token)

    retrieval_index = None
    if not args.no_retrieval:
        try:
            print("Connecting to the knowledge base (DATABASE_URL)")
            retrieval_index = open_index()
        except Exception as exc:  # no DB configured/reachable: fall back to the bare model
            print(f"WARNING: knowledge base unavailable ({exc}) - answering from the base model's own memory only.")

    generate_fn = make_generate_fn(model, tokenizer, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    label = "RAG (retrieval + base model)" if retrieval_index is not None else "base model only"
    print(f"\nReady. Mode: {label}")
    print("Type 'exit' or 'quit' to end.\n")

    history: list[dict] = []
    while True:
        try:
            user_input = input("You:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        if not user_input:
            continue

        active_history = history if not args.no_history else None

        result = answer(
            user_input,
            generate_fn,
            retrieval_index,
            system_prompt=args.system_prompt,
            history=active_history,
            top_k=args.top_k,
        )

        if retrieval_index is not None:
            if result.retrieved:
                print(f"(retrieved {len(result.retrieved)} fact(s), top score {result.retrieved[0]['score']:.2f})")
            else:
                print("(no relevant facts retrieved)")
            if not result.generated:
                print("(declined without calling the model - see guardrails.py)")

        print(f"\nAssistant:\n> {result.response}\n")

        if not args.no_history:
            history.append({"role": "user", "content": [{"type": "text", "text": user_input}]})
            history.append({"role": "assistant", "content": [{"type": "text", "text": result.response}]})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
