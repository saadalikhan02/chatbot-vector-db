"""Prompt construction for the base (not fine-tuned) instruction model.

Since this project relies entirely on RAG rather than fine-tuning, the
system prompt does more work than it would in `../lightweight-chatbot`: it
has to instruct a generic instruction-tuned model, from scratch each turn,
how to use the retrieved context and how to behave when nothing relevant
was retrieved.
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "You are the official AI assistant for Technyx Systems. Speak naturally as part of the "
    'company (use "we"/"our"), never as someone describing or citing a website. Every '
    "question below may include a 'Relevant information' block containing verified company "
    "facts retrieved for that question - if present, treat it as ground truth and answer using "
    "only those facts, in your own words. Do not invent or assume any company-specific detail "
    "(locations, dates, names, platforms, contact info) that isn't in that block, and never name "
    "a platform, tool, or vendor that doesn't appear verbatim there. A quoted testimonial's named "
    "person works for that client, not for Technyx - never call them a Technyx employee or "
    "leader. If no 'Relevant information' block is present, or it doesn't answer the question, "
    'say so directly - for example "I don\'t have verified information about that" - never '
    'guess, and never phrase it as "the website doesn\'t say" or similar.'
)


def build_messages(
    user_text: str,
    system_text: str | None = SYSTEM_PROMPT,
    history: list[dict[str, Any]] | None = None,
    context_block: str | None = None,
) -> list[dict[str, Any]]:
    """Build a chat-template messages list: optional system message, optional
    prior turns, then the new user turn.

    ``context_block`` (from retrieval.build_context_block) is retrieved,
    verified fact text - prepended to the user's question, clearly labeled,
    rather than folded into the system message, so it's obviously per-turn
    and disposable rather than a permanent instruction.
    """
    messages: list[dict[str, Any]] = []
    if system_text:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
    if history:
        # History can also come from CLI/evaluation callers, so preserve the
        # API boundary here. A client must never be able to supply a system
        # turn that outranks the verified system instructions.
        for message in history:
            if message.get("role") not in {"user", "assistant"}:
                raise ValueError("History roles must be 'user' or 'assistant'.")
            messages.append(message)

    final_user_text = f"{context_block}\n\nQuestion: {user_text}" if context_block else user_text
    messages.append({"role": "user", "content": [{"type": "text", "text": final_user_text}]})
    return messages
