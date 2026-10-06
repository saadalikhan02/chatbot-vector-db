"""Post-generation fact-check guardrail - the general fix for the
fabrication *class* of bug documented in README's "Limitations": WordPress,
"Google Cloud and Microsoft Azure", and "Salesforce Experience Cloud and
Adobe Experience Manager" were each a different specific claim, so a
keyword-gated guardrail (guardrails.py) or a prompt instruction
(prompting.py) that closes one phrasing doesn't reach the next one - every
fix this project made that way was verified to work for the exact case
tested and then a *different* case in the same failure family regressed in
the next full run.

This checks the *generated answer itself*, once, regardless of which
question produced it: a deliberately blunt, deterministic signal (a run of
2+ consecutive capitalized words - the same shape as a product, vendor, or
proper-noun name) that doesn't appear anywhere in the retrieved context is
treated as an unsupported claim, and the whole answer is replaced with a
safe fallback rather than risking one fabricated detail reaching the user.

Deliberately not a second LLM call (no self-critique prompt): this
project's CPU inference is already slow (see README's "Limitations"), and
a model that fabricates a fact isn't a reliable judge of its own
fabrication either. A cheap, dumb, deterministic check that can't itself
hallucinate is a better fit here than a smarter but equally fallible one.

Known limitation of this approach (not silently overlooked): it only
catches unsupported *named entities* (multi-word proper nouns). The Webby
Award-policy fabrication measured during development ("We don't publicly
release details about specific awards we've received") has no such phrase
and would not be caught by this check.
"""

from __future__ import annotations

import re

# Two-or-more consecutive capitalized words is a much stronger "this looks
# like a proper noun/product name" signal than a single capitalized word,
# which is just as often an ordinary sentence start ("We", "Our", "This") -
# requiring a run of 2+ sharply cuts false positives without needing a
# stopword list for every sentence-initial word.
_PROPER_NOUN_PHRASE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3}\b")

# A single word with an internal capital ("WordPress", "GitHub") is the
# same brand-name signal as _PROPER_NOUN_PHRASE, just without a space to
# require - measured necessary: "Does Technyx work with WordPress?" was
# answered "...has experience working with WordPress..." and
# _PROPER_NOUN_PHRASE alone never matches it (no second capitalized word
# follows "WordPress" with whitespace between them). Also matches
# legitimate compounds ("DevOps", "McKinney") - not a false positive, since
# those only get flagged when they're *not* already in the retrieved
# context for that turn, which is exactly the behavior this check wants.
_CAMEL_CASE_WORD = re.compile(r"\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b")

# The company's own name is capitalized by convention, not because it's an
# unverified specific - it doesn't need to appear in context to be "safe".
_ALWAYS_ALLOWED = frozenset({"technyx", "technyx systems"})

# A phrase from the user's own question, echoed back in a sentence that is
# itself declining to confirm it ("I don't have verified information about
# a physical office in New York"), is not a new claim - it's naming what
# was asked about while refusing to assert it. Scoped to the sentence
# level (not "anywhere in the response"), and required alongside the
# user_input check, not instead of it: measured necessary after the
# opposite bug - asked about a Microsoft/AWS partnership, the model wrote
# "I don't have verified information about that. We primarily focus on
# working with leading cloud providers like Google Cloud and Microsoft
# Azure." - a *separate*, non-refusal sentence asserting a fabricated
# partnership. Checking the whole response for a refusal phrase would have
# wrongly exempted that fabrication too; checking sentence-by-sentence
# catches it while still clearing the New York case.
_REFUSAL_PHRASES = (
    "don't have verified information",
    "don’t have verified information",
    "don't have information",
    "don’t have information",
    "not aware of",
)


def _split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text)


UNSUPPORTED_CLAIM_FALLBACK = (
    "I want to avoid stating anything I can't verify, and part of that answer "
    "isn't something I have confirmed information for. Feel free to ask about "
    "a more specific part of that question and I'll answer what's verified."
)


def find_unsupported_claims(response: str, context_block: str | None, user_input: str = "") -> list[str]:
    """Return each 2+-word capitalized phrase in ``response`` that doesn't
    appear (case-insensitively, as a substring) anywhere in
    ``context_block`` - a candidate fabricated specific (a platform,
    vendor, or named person the model added that retrieval never actually
    supplied). A phrase is exempted only if it's also in ``user_input``
    *and* appears in a sentence that's declining to confirm it (see the
    comment above ``_REFUSAL_PHRASES`` for why both conditions are
    required, not just one)."""
    context_lower = (context_block or "").lower()
    user_input_lower = user_input.lower()
    unsupported: list[str] = []
    seen: set[str] = set()
    for sentence in _split_sentences(response):
        sentence_lower = sentence.lower()
        is_refusal_sentence = any(phrase in sentence_lower for phrase in _REFUSAL_PHRASES)
        matches = list(_PROPER_NOUN_PHRASE.finditer(sentence)) + list(_CAMEL_CASE_WORD.finditer(sentence))
        for match in matches:
            phrase = match.group(0)
            key = phrase.lower()
            if key in seen or key in _ALWAYS_ALLOWED:
                continue
            seen.add(key)
            if key in context_lower:
                continue
            if is_refusal_sentence and key in user_input_lower:
                continue
            unsupported.append(phrase)
    return unsupported
