"""The end-to-end answer pipeline: retrieve, apply guardrails, generate.

Factored out of scripts/chat.py and scripts/evaluate.py (which used to
each inline the same "retrieve, then check guardrails in order, generate
only if none fire" logic) so there is exactly one place that decides it -
every caller (both scripts, scripts/chat_gguf.py, and the test suite in
tests/test_generation.py) shares it, so a test failure here means a real
user-facing behavior actually changed, not that a second, possibly-
drifted copy of the logic changed.

Deliberately backend-agnostic: this module takes a ``generate_fn``
callable rather than a specific model/tokenizer pair, so the same
guardrail/fact-check logic protects every generation backend equally -
the plain-PyTorch path (llm.py) and the quantized llama.cpp path
(gguf_llm.py, ~3.6x faster - see README's "Speed") get identical
reliability guarantees. See llm.make_generate_fn / gguf_llm.make_generate_fn
for how each backend builds the closure this expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .fact_check import UNSUPPORTED_CLAIM_FALLBACK, find_unsupported_claims
from .guardrails import (
    INPUT_TOO_LONG_REFUSAL,
    NO_CONCRETE_EXAMPLES_REFUSAL,
    NO_CONTEXT_REFUSAL,
    NO_EMPLOYEE_NAMES_REFUSAL,
    is_input_too_long,
    should_refuse_employee_names_request,
    should_refuse_example_request,
    should_refuse_without_generating,
)
from .prompting import SYSTEM_PROMPT, build_messages
from .retrieval import RetrievalIndex, build_context_block, build_retrieval_query

GenerateFn = Callable[[list[dict[str, Any]]], str]


@dataclass
class AnswerResult:
    response: str
    retrieved: list[dict[str, Any]]
    context_block: str | None
    generated: bool  # False if a guardrail answered without calling the model
    unsupported_claims: list[str] = field(default_factory=list)  # non-empty if fact_check replaced the response


def _answer_card_for_query(
    retrieved: list[dict[str, Any]],
    user_input: str,
    retrieval_query: str,
) -> dict[str, Any] | None:
    """Return a complete card for the current intent, using history only for follow-ups.

    Retrieval folds the previous user turn into ``retrieval_query`` so pronoun
    follow-ups can find context. That expanded query must not decide the answer
    card by itself: otherwise a previous CMS question hijacks a later locations
    question. Specific implementation questions also need the model to combine
    platform facts instead of receiving a generic platform inventory.
    """

    def matching_card(query: str) -> dict[str, Any] | None:
        normalized = query.lower()
        for fact in retrieved:
            keywords = fact.get("answer_card_keywords")
            if not isinstance(keywords, list):
                continue
            if not any(isinstance(keyword, str) and keyword in normalized for keyword in keywords):
                continue
            fact_id = str(fact.get("fact_id", ""))
            if fact_id == "answer_card_platforms_20260922":
                specific_markers = ("mobile", "backend", "api", "integrat", "migrat", "develop", "support")
                list_markers = ("list", "which", "what cms", "any cms", "cms platforms", "platforms?")
                if any(marker in normalized for marker in specific_markers) and not any(
                    marker in normalized for marker in list_markers
                ):
                    continue
            return fact
        return None

    if card := matching_card(user_input):
        return card

    normalized_input = user_input.strip().lower()
    words = normalized_input.split()
    is_short_followup = len(words) <= 8 and any(
        marker in normalized_input for marker in ("list", "them", "those", "more", "which of", "and what")
    )
    return matching_card(retrieval_query) if is_short_followup else None


def answer(
    user_input: str,
    generate_fn: GenerateFn,
    retrieval_index: RetrievalIndex | None,
    system_prompt: str = SYSTEM_PROMPT,
    history: list[dict[str, Any]] | None = None,
    top_k: int = 8,
) -> AnswerResult:
    """Answer one question: retrieve (if a retrieval_index is given), apply
    each guardrail in turn, and only call ``generate_fn`` if none of them
    fire. Guardrail order matters no further than that all three are
    checked before generation - see guardrails.py for what each one
    catches. ``generate_fn`` takes the assembled messages list and returns
    the model's raw text response - see llm.make_generate_fn /
    gguf_llm.make_generate_fn for the two implementations."""
    retrieved: list[dict[str, Any]] = []
    context_block: str | None = None

    # Checked before anything else touches user_input (tokenization,
    # embedding, retrieval, generation) - see guardrails.py's comment on
    # MAX_USER_INPUT_CHARS for why this exists.
    if is_input_too_long(user_input):
        return AnswerResult(INPUT_TOO_LONG_REFUSAL, retrieved, context_block, generated=False)

    if retrieval_index is not None:
        query = build_retrieval_query(user_input, history)
        retrieved = retrieval_index.search(query, top_k=top_k)
        context_block = build_context_block(retrieved)

        if should_refuse_without_generating(user_input, context_block):
            return AnswerResult(NO_CONTEXT_REFUSAL, retrieved, context_block, generated=False)
        if should_refuse_example_request(user_input, context_block):
            return AnswerResult(NO_CONCRETE_EXAMPLES_REFUSAL, retrieved, context_block, generated=False)
        if should_refuse_employee_names_request(user_input, retrieved):
            return AnswerResult(NO_EMPLOYEE_NAMES_REFUSAL, retrieved, context_block, generated=False)

        if answer_card := _answer_card_for_query(retrieved, user_input, query):
            return AnswerResult(answer_card["fact"], retrieved, context_block, generated=True)

    messages = build_messages(user_input, system_text=system_prompt, history=history, context_block=context_block)
    response = generate_fn(messages)

    # Fact-check only applies when there was a retrieval_index to check
    # the response against - the "without RAG" baseline (retrieval_index
    # is None, used for comparison in scripts/evaluate.py) is deliberately
    # left as the raw, unguarded model output.
    unsupported_claims: list[str] = []
    if retrieval_index is not None:
        unsupported_claims = find_unsupported_claims(response, context_block, user_input=user_input)
        if unsupported_claims:
            response = UNSUPPORTED_CLAIM_FALLBACK

    return AnswerResult(response, retrieved, context_block, generated=True, unsupported_claims=unsupported_claims)
