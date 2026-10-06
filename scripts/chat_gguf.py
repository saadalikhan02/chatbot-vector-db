#!/usr/bin/env python3
"""Interactively chat with the quantized GGUF export via llama.cpp - the
fast path (measured ~10.3 tok/s vs. ~1.6-2.9 tok/s for the plain PyTorch
path in scripts/chat.py, on the same CPU, in the sibling
`../lightweight-chatbot` project - see README's "Speed").

Uses the exact same pipeline.answer() as scripts/chat.py - retrieval,
every guardrail (guardrails.py), and the fact-check guardrail
(fact_check.py) all apply identically here. Only the generation backend
differs (gguf_llm.py instead of llm.py); the reliability guarantees this
project relies on aren't backend-specific.

Requires: pip install llama-cpp-python \\
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
and a GGUF file produced by scripts/export_gguf.py.

Usage:
    python scripts/chat_gguf.py --model outputs/technyx-gemma3-1b-q4_k_m.gguf

Type 'exit' or 'quit' to end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatbot_rag.gguf_llm import load_gguf_model, make_generate_fn  # noqa: E402
from chatbot_rag.pipeline import answer  # noqa: E402
from chatbot_rag.prompting import SYSTEM_PROMPT  # noqa: E402
from chatbot_rag.retrieval import open_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a GGUF file from scripts/export_gguf.py")
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-threads", type=int, default=6, help="6 measured faster than 12 on this CPU - see README")
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Disable RAG - answer from the base model's own memory only (for comparison)",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Max retrieved facts per question")
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Don't keep prior turns in context (each question is independent)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Loading {args.model}...")
    try:
        llm = load_gguf_model(args.model, n_ctx=args.n_ctx, n_threads=args.n_threads)
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print(f"ERROR: GGUF file not found at {args.model}. Run scripts/export_gguf.py first.", file=sys.stderr)
        return 1

    generate_fn = make_generate_fn(llm, max_new_tokens=args.max_new_tokens)

    retrieval_index = None
    if not args.no_retrieval:
        try:
            print("Connecting to the knowledge base (DATABASE_URL)")
            retrieval_index = open_index()
        except Exception as exc:  # no DB configured/reachable: fall back to the bare model
            print(f"WARNING: knowledge base unavailable ({exc}) - answering from the base model's own memory only.")

    label = "RAG (retrieval + GGUF model)" if retrieval_index is not None else "GGUF model only"
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
