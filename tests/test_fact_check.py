"""Unit tests for src/chatbot_rag/fact_check.py - pure logic, no model.

Each case here is a real generated response captured during development
(see fact_check.py's module docstring and comments) - not synthetic
examples, so a passing suite means the actual measured failures/fixes
stay fixed.
"""

from __future__ import annotations

from chatbot_rag.fact_check import find_unsupported_claims


class TestFindUnsupportedClaims:
    def test_no_proper_nouns_returns_empty(self):
        assert find_unsupported_claims("We offer a range of services.", "Relevant information:\n- We help.") == []

    def test_phrase_present_in_context_is_not_flagged(self):
        response = "Our head office is in Dubai, UAE, alongside offices in McKinney Texas USA."
        context = "Relevant information:\n- Our head office is in Dubai, UAE, alongside offices in McKinney Texas USA."
        assert find_unsupported_claims(response, context) == []

    def test_multiword_phrase_not_in_context_is_flagged(self):
        # Measured: "We've worked across established platforms like
        # Salesforce Experience Cloud and Adobe Experience Manager" -
        # neither term exists anywhere in this project's corpus.
        response = "We've worked across platforms like Salesforce Experience Cloud and Adobe Experience Manager."
        context = "Relevant information:\n- We work with Sitecore."
        claims = find_unsupported_claims(response, context)
        assert "Salesforce Experience Cloud" in claims
        assert "Adobe Experience Manager" in claims

    def test_camel_case_single_word_brand_is_flagged(self):
        # Measured: "Does Technyx work with WordPress?" answered "...has
        # experience working with WordPress..." - a single camelCase
        # token with no second capitalized word, which the plain
        # multi-word-phrase pattern alone never matches.
        response = "Technyx has experience working with WordPress and Sitecore."
        context = "Relevant information:\n- We work with Sitecore."
        assert "WordPress" in find_unsupported_claims(response, context)

    def test_company_name_is_always_allowed(self):
        assert find_unsupported_claims("Technyx Systems offers custom software.", None) == []

    def test_refusal_echoing_the_question_is_not_flagged(self):
        # Measured false positive, now fixed: asked about a New York
        # office, the model correctly refused ("I don't have verified
        # information about a physical office in New York") and the
        # checker used to block that correct refusal because "New York"
        # isn't in the retrieved context - it's only in the question.
        response = "I don't have verified information about a physical office in New York."
        context = "Relevant information:\n- Our head office is in Dubai, UAE."
        claims = find_unsupported_claims(response, context, user_input="Does Technyx have an office in New York?")
        assert claims == []

    def test_separate_affirmative_sentence_is_still_flagged_even_if_response_also_refuses(self):
        # Measured: the opposite failure mode - a response that opens
        # with a refusal-shaped sentence but then makes an entirely
        # separate, unsupported affirmative claim in the next sentence.
        # A whole-response "contains a refusal phrase, so exempt
        # everything" rule would have missed this; per-sentence checking
        # catches it.
        response = (
            "I don't have verified information about that. We primarily focus on working with "
            "leading cloud providers like Google Cloud and Microsoft Azure."
        )
        context = "Relevant information:\n- We offer custom software and digital experience platforms."
        claims = find_unsupported_claims(
            response, context, user_input="Is Technyx partnered with Microsoft or AWS as an official partner?"
        )
        assert "Google Cloud" in claims
        assert "Microsoft Azure" in claims

    def test_echoed_question_term_without_a_refusal_sentence_is_still_flagged(self):
        # The user_input exemption requires *both* conditions - a phrase
        # merely appearing in the question is not enough on its own if
        # the sentence containing it isn't actually declining to confirm
        # it (this is the same shape as the Microsoft/AWS case, but with
        # the fabricated term literally repeated from the question).
        response = "Yes, Technyx does have an office in New York."
        context = "Relevant information:\n- Our head office is in Dubai, UAE."
        claims = find_unsupported_claims(response, context, user_input="Does Technyx have an office in New York?")
        assert "New York" in claims

    def test_no_context_at_all_still_exempts_refused_question_terms(self):
        response = "I don't have verified information about New York."
        claims = find_unsupported_claims(response, None, user_input="Does Technyx have an office in New York?")
        assert claims == []
