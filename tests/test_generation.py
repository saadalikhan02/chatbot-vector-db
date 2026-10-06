"""End-to-end regression tests: real retrieval + real 1B model generation.

Slow (loads google/gemma-3-1b-it and generates for every case - minutes,
not seconds) and requires HF_TOKEN / an accepted license, same as
scripts/chat.py. Run explicitly:

    pytest tests/test_generation.py -m slow -v

or skip these and run only the fast tests (the default for `pytest` with
no args, since these are marked ``slow`` and excluded by pytest.ini's
default `-m "not slow"`):

    pytest

Each case here is a specific, previously-measured behavior (see the
comment on each test) - not an attempt at full coverage of
evaluation/test_cases.jsonl, which stays the tool for open-ended
qualitative review. One case (test_wordpress_adversarial_trap) is marked
xfail: it is a known, currently-unresolved reliability gap (see README's
"Limitations") rather than something this suite claims to guarantee -
marking it xfail keeps it visible in test output without blocking CI on
an issue that was deliberately deferred, not overlooked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def model_and_tokenizer():
    from chatbot_rag.llm import load_model, load_tokenizer
    from chatbot_rag.utils import get_hf_token, load_dotenv_if_present

    load_dotenv_if_present(REPO_ROOT / ".env")
    hf_token = get_hf_token()
    tokenizer = load_tokenizer("google/gemma-3-1b-it", hf_token=hf_token)
    model = load_model("google/gemma-3-1b-it", hf_token=hf_token)
    return model, tokenizer


def _normalized(text: str) -> str:
    """Lowercase and fold typographic quotes to straight ones. Measured
    necessary: the model consistently generates a curly apostrophe
    (U+2019, "don’t") rather than a straight one ("don't"), which
    silently broke substring assertions using the straight form."""
    return text.lower().replace("’", "'").replace("‘", "'")


def _answer(model_and_tokenizer, retrieval_index, user_input: str, history=None):
    from chatbot_rag.llm import make_generate_fn
    from chatbot_rag.pipeline import answer

    model, tokenizer = model_and_tokenizer
    generate_fn = make_generate_fn(model, tokenizer)
    return answer(user_input, generate_fn, retrieval_index, history=history)


class TestGuardrailsFireEndToEnd:
    """Confirms the guardrail decision reaches the real pipeline (not just
    the unit-tested logic in test_guardrails.py) and that the model is
    never actually called for these - result.generated must be False."""

    def test_empty_context_question_declines(self, model_and_tokenizer, retrieval_index):
        result = _answer(model_and_tokenizer, retrieval_index, "What's the weather like today?")
        assert not result.generated
        assert "don't have verified information" in _normalized(result.response)

    def test_concrete_example_request_declines(self, model_and_tokenizer, retrieval_index):
        result = _answer(
            model_and_tokenizer, retrieval_index, "Can you share examples of Technyx's past work or projects?"
        )
        assert not result.generated
        assert "case stud" in _normalized(result.response)

    def test_employee_names_request_declines(self, model_and_tokenizer, retrieval_index):
        result = _answer(
            model_and_tokenizer,
            retrieval_index,
            "Can you tell me the names of a few Technyx employees, other than testimonial clients?",
        )
        assert not result.generated
        assert "not Technyx employees" in result.response

    def test_leadership_question_declines(self, model_and_tokenizer, retrieval_index):
        result = _answer(model_and_tokenizer, retrieval_index, "Who leads Technyx?")
        assert not result.generated

    def test_greeting_is_not_declined(self, model_and_tokenizer, retrieval_index):
        result = _answer(model_and_tokenizer, retrieval_index, "Hi there!")
        assert result.generated


class TestGroundedFactsAreCorrect:
    """Facts the corpus explicitly states - these must appear correctly,
    every time, or retrieval/prompting has regressed."""

    def test_headquarters_is_dubai(self, model_and_tokenizer, retrieval_index):
        result = _answer(model_and_tokenizer, retrieval_index, "Where is Technyx headquartered?")
        assert "dubai" in _normalized(result.response)

    def test_contact_email_is_correct(self, model_and_tokenizer, retrieval_index):
        result = _answer(model_and_tokenizer, retrieval_index, "How can I contact Technyx?")
        assert "info@technyxsystems.com" in result.response

    def test_founding_year_is_correct(self, model_and_tokenizer, retrieval_index):
        result = _answer(model_and_tokenizer, retrieval_index, "How long has Technyx been in business?")
        assert "2014" in result.response

    def test_followup_head_office_question_says_dubai_not_mckinney(self, model_and_tokenizer, retrieval_index):
        # Regression test for the HQ-inconsistency bug: this exact
        # follow-up used to answer "McKinney, Texas, USA" - see
        # retrieval.build_retrieval_query's docstring.
        history = [
            {"role": "user", "content": [{"type": "text", "text": "Where is Technyx located?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Dubai, Karachi, McKinney, and Sydney."}]},
        ]
        result = _answer(model_and_tokenizer, retrieval_index, "Which of those is the head office?", history=history)
        response_lower = _normalized(result.response)
        assert "dubai" in response_lower
        assert "mckinney" not in response_lower or "dubai" in response_lower.split("mckinney")[0]


class TestKnownAdversarialTraps:
    """Things that don't exist and must be refused, not confirmed."""

    def test_iso_certification_is_refused(self, model_and_tokenizer, retrieval_index):
        result = _answer(model_and_tokenizer, retrieval_index, "Is Technyx ISO 27001 certified?")
        assert "don't have verified information" in _normalized(result.response)

    def test_new_york_office_is_refused(self, model_and_tokenizer, retrieval_index):
        # Also a regression test for fact_check.py's sentence-level
        # refusal exemption: the raw model answer echoes "New York" from
        # the question while declining it, which must not itself trip the
        # fact-check guardrail below.
        result = _answer(model_and_tokenizer, retrieval_index, "Does Technyx have an office in New York?")
        assert "don't have verified information" in _normalized(result.response)
        assert not result.unsupported_claims

    def test_wordpress_adversarial_trap(self, model_and_tokenizer, retrieval_index):
        # Previously xfail (see git history): the model itself flips
        # between correctly refusing and confidently affirming WordPress
        # across runs. The fact-check guardrail (fact_check.py) now makes
        # the *outcome* reliable either way - if the model affirms it,
        # "WordPress" isn't in retrieved context (verified: it's not
        # anywhere in the corpus) so the guardrail replaces the response.
        result = _answer(model_and_tokenizer, retrieval_index, "Does Technyx work with WordPress?")
        assert "yes" not in _normalized(result.response).split(".")[0]


class TestFactCheckGuardrail:
    """End-to-end coverage for fact_check.py's post-generation check -
    see its module docstring for why this exists (keyword-gated
    guardrails and prompt instructions kept missing new phrasings of the
    same fabrication class)."""

    def test_microsoft_aws_partnership_is_not_fabricated(self, model_and_tokenizer, retrieval_index):
        # Measured: raw model output was "I don't have verified
        # information about that. We primarily focus on working with
        # leading cloud providers like Google Cloud and Microsoft
        # Azure." - neither cloud provider appears anywhere in the corpus.
        result = _answer(
            model_and_tokenizer,
            retrieval_index,
            "Is Technyx partnered with Microsoft or AWS as an official partner?",
        )
        response_lower = _normalized(result.response)
        assert "google cloud" not in response_lower
        assert "microsoft azure" not in response_lower
