"""Unit tests for src/chatbot_rag/guardrails.py - pure logic, no model, no I/O.

Each test here encodes a behavior that was measured (via
scripts/evaluate.py, by eye, against the real model) to matter during this
project's development - see the docstrings/comments in guardrails.py for
the specific failure each guardrail was written to close. These tests
don't re-verify the model's behavior (that needs test_generation.py); they
verify the *decision logic* that decides whether to call the model at all.
"""

from __future__ import annotations

from chatbot_rag.guardrails import (
    INPUT_TOO_LONG_REFUSAL,
    MAX_USER_INPUT_CHARS,
    NO_CONCRETE_EXAMPLES_REFUSAL,
    NO_CONTEXT_REFUSAL,
    NO_EMPLOYEE_NAMES_REFUSAL,
    is_greeting,
    is_input_too_long,
    should_refuse_employee_names_request,
    should_refuse_example_request,
    should_refuse_without_generating,
    wants_concrete_example,
    wants_employee_names,
)


class TestIsGreeting:
    def test_plain_greetings(self):
        for text in ["Hi there!", "hello", "Hey there.", "Good morning", "howdy", "Hiya!"]:
            assert is_greeting(text), text

    def test_informational_question_is_not_a_greeting(self):
        assert not is_greeting("Hi, where is Technyx headquartered?")
        assert not is_greeting("What is the capital of France?")


class TestNoContextRefusal:
    def test_refuses_when_nothing_retrieved_and_not_a_greeting(self):
        # Measured: with no retrieved context, the base model wrote
        # unrelated Python code and invented live weather data instead of
        # declining - see guardrails.py's module docstring.
        assert should_refuse_without_generating("Write me a Python program that sorts a list.", None)
        assert should_refuse_without_generating("What's the weather like today?", None)
        assert should_refuse_without_generating("Do you support that?", None)

    def test_does_not_refuse_a_greeting_even_with_no_context(self):
        assert not should_refuse_without_generating("Hi there!", None)

    def test_does_not_refuse_when_context_was_retrieved(self):
        assert not should_refuse_without_generating(
            "Where is Technyx headquartered?", "Relevant information:\n- Our head office is in Dubai, UAE."
        )

    def test_refusal_text_is_non_empty(self):
        assert NO_CONTEXT_REFUSAL.strip()


class TestExampleRequestGuardrail:
    def test_wants_concrete_example_matches_known_phrasings(self):
        assert wants_concrete_example("Can you share examples of Technyx's past work or projects?")
        assert wants_concrete_example("Do you have any case studies?")
        assert wants_concrete_example("Show me a sample of your work")

    def test_wants_concrete_example_does_not_match_generic_questions(self):
        assert not wants_concrete_example("What services does Technyx provide?")
        assert not wants_concrete_example("Where is Technyx headquartered?")

    def test_refuses_when_no_context_at_all(self):
        assert should_refuse_example_request("Can you share examples of your past work?", None)

    def test_refuses_when_context_has_no_concrete_evidence(self):
        # Measured: retrieval returned 8 real-but-generic capability facts
        # with zero quantified outcomes, and the model invented a fake
        # "20% reduction in processing time" to fill the gap - see
        # guardrails.py's comment above NO_CONCRETE_EXAMPLES_REFUSAL, and
        # the data audit in data/knowledge/facts.jsonl this was based on.
        generic_context = "Relevant information:\n- We offer custom software and digital experience platforms."
        assert should_refuse_example_request("Can you share examples of your past work?", generic_context)

    def test_does_not_refuse_when_context_has_real_evidence(self):
        context_with_evidence = (
            "Relevant information:\n- Delivered a 20% reduction in processing time for a case study client."
        )
        assert not should_refuse_example_request("Can you share a case study?", context_with_evidence)

    def test_does_not_refuse_unrelated_questions(self):
        assert not should_refuse_example_request("What services does Technyx provide?", None)

    def test_refusal_text_is_non_empty(self):
        assert NO_CONCRETE_EXAMPLES_REFUSAL.strip()


class TestEmployeeNamesGuardrail:
    def test_wants_employee_names_matches_known_phrasings(self):
        # Both phrasings that were separately measured to trigger the same
        # misattribution bug - see guardrails.py's comment above
        # NO_EMPLOYEE_NAMES_REFUSAL.
        assert wants_employee_names("Can you tell me the names of a few Technyx employees?")
        assert wants_employee_names("Who leads Technyx?")
        assert wants_employee_names("Who founded Technyx?")

    def test_wants_employee_names_does_not_match_client_questions(self):
        assert not wants_employee_names("Which companies has Technyx worked with?")
        assert not wants_employee_names("Does Technyx have any client testimonials?")

    def test_refuses_when_retrieval_surfaces_client_testimonials(self):
        # Measured: presented real testimonial authors (Nestle, Majid Al
        # Futtaim, FGS, Blue Barracuda, Create, SCPS staff) as if they
        # were Technyx's own employees/leadership - see guardrails.py.
        retrieved = [
            {"fact_type": "other", "fact": "We're based in four locations..."},
            {"fact_type": "client", "fact": "“Technyx has been an incredible partner...” - Dina Saadeh"},
        ]
        assert should_refuse_employee_names_request("Who leads Technyx?", retrieved)

    def test_does_not_refuse_when_no_client_facts_retrieved(self):
        retrieved = [{"fact_type": "history", "fact": "Founded in 2014."}]
        assert not should_refuse_employee_names_request("Who leads Technyx?", retrieved)

    def test_does_not_refuse_unrelated_questions_even_with_client_facts(self):
        retrieved = [{"fact_type": "client", "fact": "A real testimonial."}]
        assert not should_refuse_employee_names_request("Which companies has Technyx worked with?", retrieved)

    def test_handles_empty_or_none_retrieved(self):
        assert not should_refuse_employee_names_request("Which companies has Technyx worked with?", None)
        assert not should_refuse_employee_names_request("Which companies has Technyx worked with?", [])

    def test_refusal_text_is_non_empty(self):
        assert NO_EMPLOYEE_NAMES_REFUSAL.strip()


class TestInputTooLong:
    """Added during a security review - see guardrails.py's comment above
    MAX_USER_INPUT_CHARS: nothing previously bounded user_input length
    before it reached tokenization/embedding/generation."""

    def test_normal_question_is_not_too_long(self):
        assert not is_input_too_long("Where is Technyx headquartered?")

    def test_input_at_the_limit_is_not_too_long(self):
        assert not is_input_too_long("a" * MAX_USER_INPUT_CHARS)

    def test_input_over_the_limit_is_too_long(self):
        assert is_input_too_long("a" * (MAX_USER_INPUT_CHARS + 1))

    def test_refusal_text_is_non_empty(self):
        assert INPUT_TOO_LONG_REFUSAL.strip()
