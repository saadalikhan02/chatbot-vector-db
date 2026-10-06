"""Code-level guardrails - enforced in code, not just asked for in the prompt.

The system prompt (prompting.py) already asks the model to decline when no
relevant context was retrieved. Measured evaluation runs
(evaluation/test_cases.jsonl, categories "out_of_scope"/"ambiguous"/
"adversarial") showed that instruction alone is not reliable: with
retrieval finding nothing, the base model wrote an unrelated Python
program on request and fabricated live weather data ("partly cloudy,
68 degrees Fahrenheit...") instead of declining. A prompt-only guardrail
on a model that was never trained to follow it is advisory, not a
guarantee - this module makes the refusal a code-level decision instead:
when there's no retrieved context and the input isn't a plain greeting,
skip generation entirely and return a fixed refusal. This also means a
declined question costs no LLM call (faster, and zero hallucination risk
by construction, not by hoping the model complies).
"""

from __future__ import annotations

import re
from typing import Any

# Found during a security review, not a measured production incident:
# nothing in this project enforced any limit on user_input length before
# it reached tokenization/embedding/generation. A pathologically long
# input (accidental or deliberate) costs real CPU time proportional to
# its length on every layer it touches (embedding, retrieval, generation)
# with no cap - the kind of thing that's cheap to guard against and
# expensive to discover in production. 4000 characters is generous for
# any real question (every question in evaluation/test_cases.jsonl is
# under 150) while still bounding the worst case.
MAX_USER_INPUT_CHARS = 4000

INPUT_TOO_LONG_REFUSAL = (
    "That message is too long for me to process. Could you break it into a shorter, more specific question?"
)

NO_CONTEXT_REFUSAL = (
    "I don't have verified information about that. I can help with questions "
    "about Technyx Systems - our services, locations, history, and the "
    "platforms we work with."
)

# Plain small talk has no informational content for retrieval to match
# against (an empty/low-score search result is expected and correct for
# these), so it's exempted from the no-context refusal rather than being
# treated the same as an off-topic or out-of-scope question.
_GREETINGS = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
        "howdy",
        "yo",
        "greetings",
        "hiya",
    }
)


def is_greeting(user_input: str) -> bool:
    """True for plain greetings/small talk with no informational content."""
    normalized = user_input.strip().lower().rstrip("!.,?")
    return normalized in _GREETINGS


def should_refuse_without_generating(user_input: str, context_block: str | None) -> bool:
    """True when retrieval found nothing relevant and the input isn't a
    plain greeting - the exact case where an unguarded model was measured
    to answer anyway rather than declining (see module docstring)."""
    return context_block is None and not is_greeting(user_input)


# Retrieval can find *something* relevant without finding anything that
# actually answers a request for a concrete example: asked "Can you share
# examples of Technyx's past work or projects?", retrieval returned 8
# generic capability/service facts (real, but none of them a project
# write-up), and the model filled the gap itself - inventing a fake
# "regional bank" client, a fabricated "20% reduction in processing time",
# and a fictional fashion-retailer project. Confirmed by data audit
# (data/knowledge/facts.jsonl): zero facts in the whole corpus contain a
# quantified outcome ("%", "reduction", "increase", "ROI", "case study") -
# so there is nothing true this kind of request could ever be grounded in
# until real case-study content is written and crawled (see README). This
# guardrail declines instead of letting the model improvise specifics that
# don't exist in the retrieved context.
NO_CONCRETE_EXAMPLES_REFUSAL = (
    "I don't have specific case studies or project write-ups with verified "
    "details (client names, outcomes, metrics) to share right now. I can "
    "tell you about our services, the industries we work in, or share "
    "client testimonials instead."
)

_EXAMPLE_REQUEST_KEYWORDS = (
    "example",
    "case stud",
    "past project",
    "past work",
    "success stor",
    "portfolio",
    "sample of your work",
    "show me your work",
    "show me examples",
    "specific project",
)

# A retrieved fact that names an actual measurable outcome or an explicit
# case-study reference is real evidence a concrete-example request can be
# answered from; a request matching _EXAMPLE_REQUEST_KEYWORDS with none of
# this in the retrieved context has nothing solid to answer from.
_CONCRETE_EVIDENCE_PATTERN = re.compile(r"\d+\s*%|\bpercent\b|\bROI\b|\bcase stud", re.IGNORECASE)


def wants_concrete_example(user_input: str) -> bool:
    """True for requests asking for specific project examples/case studies,
    as opposed to a general capability/service question."""
    normalized = user_input.lower()
    return any(keyword in normalized for keyword in _EXAMPLE_REQUEST_KEYWORDS)


def should_refuse_example_request(user_input: str, context_block: str | None) -> bool:
    """True when the user is asking for a concrete example/case study and
    the retrieved context (which may be non-empty - see module docstring)
    contains no actual evidence of one."""
    if not wants_concrete_example(user_input):
        return False
    if not context_block:
        return True
    return not _CONCRETE_EVIDENCE_PATTERN.search(context_block)


# Client testimonials (fact_type "client") are the corpus's only facts that
# name real individuals with a role and company - and the only category of
# named person in this corpus is a client's, never anyone at Technyx itself
# (there is no "employee"/"leadership"/"team" fact_type at all). Measured
# necessary, twice: asked "Can you tell me the names of a few Technyx
# employees, other than testimonial clients?", retrieval returned mostly
# client testimonials (6 of 8 facts) and the model presented all six real
# testimonial authors - genuine people at Nestle, Majid Al Futtaim, FGS,
# Blue Barracuda, Create, and SCPS - as if they were Technyx's own
# employees. Separately, asked the shorter "Who leads Technyx?", it named
# one of those same client-side testimonial authors (SCPS's Technology
# Director) as Technyx's leader - a different phrasing of the identical
# underlying mistake (fact_type "client" retrieved, model treats the named
# person as belonging to Technyx). A prompt instruction not to do this was
# already in place and was not enough (see prompting.SYSTEM_PROMPT and
# README's "Measured results" for both measured failures) - this guardrail
# makes the refusal a code-level guarantee whenever a question about
# Technyx's own people retrieves a client testimonial.
NO_EMPLOYEE_NAMES_REFUSAL = (
    "I don't have verified information about Technyx's own employees, "
    "staff, or leadership. The named individuals in our content are client "
    "representatives who gave testimonials about working with us - not "
    "Technyx employees - so I won't present them as if they were."
)

_EMPLOYEE_QUERY_KEYWORDS = (
    "employee",
    "staff member",
    "staff name",
    "team member",
    "who works at technyx",
    "who works for technyx",
    "who leads",
    "who runs",
    "who founded",
    "who's the founder",
    "who is the founder",
    "leadership",
    "ceo",
)


def wants_employee_names(user_input: str) -> bool:
    """True for requests asking who Technyx's own people are - staff,
    employees, founders, or leadership - as opposed to a question about
    clients or testimonials."""
    normalized = user_input.lower()
    return any(keyword in normalized for keyword in _EMPLOYEE_QUERY_KEYWORDS)


def should_refuse_employee_names_request(user_input: str, retrieved: list[dict[str, Any]] | None) -> bool:
    """True when the user wants to know who Technyx's own people are and
    the retrieved facts include client testimonials - the exact mix that
    was measured to make the model present real client representatives as
    Technyx staff or leadership."""
    if not wants_employee_names(user_input):
        return False
    return any(fact.get("fact_type") == "client" for fact in (retrieved or []))


def is_input_too_long(user_input: str) -> bool:
    """True when the raw input exceeds MAX_USER_INPUT_CHARS - checked
    first, before retrieval/embedding/generation ever run, so a
    pathological input costs nothing beyond a length check."""
    return len(user_input) > MAX_USER_INPUT_CHARS
