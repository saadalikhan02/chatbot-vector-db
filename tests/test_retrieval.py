"""Tests for src/chatbot_rag/retrieval.py against the real built index.

Uses the configured EMBEDDING_MODEL (loaded once per session via the
retrieval_index fixture's first call - see conftest.py's embedding cache)
but never the 1B Gemma model, so this file runs in seconds, not minutes.
Each test locks in a specific retrieval fix made during development - see
the referenced comments in src/chatbot_rag/retrieval.py for the measured
failure each one closes. Exact score numbers cited in comments below were
measured against the original all-MiniLM-L6-v2 model and are illustrative
only - assertions check relative ranking/presence, not absolute scores, so
they hold regardless of which embedding model is currently configured.
"""

from __future__ import annotations

from chatbot_rag.retrieval import build_context_block, build_retrieval_query


class TestBuildRetrievalQuery:
    def test_no_history_returns_question_unchanged(self):
        assert build_retrieval_query("Where is Technyx headquartered?", None) == "Where is Technyx headquartered?"
        assert build_retrieval_query("Where is Technyx headquartered?", []) == "Where is Technyx headquartered?"

    def test_folds_the_immediately_preceding_user_turn(self):
        # Measured: "Which of those is the head office?" alone scored 0.19
        # (rank 63/883) against the fact that answers it - folding in the
        # prior turn raised that to 0.61 (top 3) - see
        # retrieval.build_retrieval_query's docstring.
        history = [
            {"role": "user", "content": [{"type": "text", "text": "Where is Technyx located?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Dubai, Karachi, McKinney, and Sydney."}]},
        ]
        query = build_retrieval_query("Which of those is the head office?", history)
        assert "Where is Technyx located?" in query
        assert "Which of those is the head office?" in query

    def test_uses_the_last_user_turn_not_the_assistant_turn(self):
        history = [
            {"role": "user", "content": [{"type": "text", "text": "What services do you offer?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Custom software and platforms."}]},
        ]
        query = build_retrieval_query("And what about pricing?", history)
        assert "What services do you offer?" in query
        assert "Custom software and platforms" not in query


class TestBuildContextBlock:
    def test_empty_results_returns_none(self):
        assert build_context_block([]) is None

    def test_renders_facts_as_a_labeled_list(self):
        block = build_context_block([{"fact": "Founded in 2014."}, {"fact": "Head office in Dubai."}])
        assert block.startswith("Relevant information:")
        assert "- Founded in 2014." in block
        assert "- Head office in Dubai." in block


class TestRetrievalIndexSearch:
    def test_off_topic_query_returns_nothing(self, retrieval_index):
        # The "found nothing" signal is what guardrails.should_refuse_
        # without_generating relies on - this must stay empty, not just
        # low-scoring, for an unrelated question.
        assert retrieval_index.search("What is the capital of France?") == []

    def test_hq_query_surfaces_the_authoritative_head_office_fact(self, retrieval_index):
        # Measured: a bare address fragment ("USA: McKinney, Texas...")
        # outranked the one fact that actually names the head office
        # (0.675 vs 0.640) - see retrieval.py's _AUTHORITATIVE_FACT_BOOST
        # comment. This must stay near the top for any HQ-style question.
        results = retrieval_index.search("Where is Technyx headquartered?", top_k=8)
        assert results, "expected at least one result for a clearly on-topic query"
        assert any("head office" in r["fact"].lower() for r in results[:3])

    def test_followup_headoffice_query_surfaces_the_right_fact_after_folding(self, retrieval_index):
        history = [
            {"role": "user", "content": [{"type": "text", "text": "Where is Technyx located?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Dubai, Karachi, McKinney, and Sydney."}]},
        ]
        query = build_retrieval_query("Which of those is the head office?", history)
        results = retrieval_index.search(query, top_k=8)
        assert any("head office" in r["fact"].lower() for r in results[:5])

    def test_testimonial_query_surfaces_client_testimonials(self, retrieval_index):
        # Measured: without a query-conditional boost, the best real
        # testimonial scored 0.527 against "any client testimonials?" -
        # just below the old top-8 cutoff, so retrieval returned zero of
        # the eleven real testimonials in the corpus - see retrieval.py's
        # _TESTIMONIAL_FACT_BOOST comment.
        results = retrieval_index.search("Does Technyx have any client testimonials?", top_k=8)
        assert any(r.get("fact_type") == "client" for r in results)

    def test_generic_capability_query_does_not_trigger_testimonial_boost(self, retrieval_index):
        # The testimonial boost is deliberately query-conditional (see
        # retrieval.py's comment on why an always-on boost was rejected) -
        # a question with no testimonial-ish keyword shouldn't force
        # client-quote facts to the top ahead of what's actually relevant.
        results = retrieval_index.search("What services does Technyx provide?", top_k=8)
        assert results
        assert results[0].get("fact_type") != "client"


class TestAnswerCards:
    def test_services_card_bypasses_generation(self, retrieval_index):
        from chatbot_rag.pipeline import answer

        def model_must_not_run(_messages):
            raise AssertionError("answer card should bypass model generation")

        result = answer("What services do you offer?", model_must_not_run, retrieval_index)
        assert result.generated
        assert "Product & Platform Engineering" in result.response
        assert "Creative & Campaign Operations" in result.response

    def test_platform_card_answers_a_follow_up_list_request(self, retrieval_index):
        from chatbot_rag.pipeline import answer

        history = [
            {"role": "user", "content": [{"type": "text", "text": "CMS platforms?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "We work across CMS platforms."}]},
        ]

        result = answer("Can you list them?", lambda _messages: "not used", retrieval_index, history=history)
        assert result.generated
        assert "Sitecore" in result.response
        assert "Payload" in result.response

    def test_frontend_card_bypasses_generation(self, retrieval_index):
        from chatbot_rag.pipeline import answer

        result = answer("Frontend technologies.", lambda _messages: "not used", retrieval_index)
        assert result.generated
        assert "React" in result.response
        assert "MUI" in result.response


class TestAnswerCardHistoryIsolation:
    def test_current_location_intent_wins_over_previous_cms_turn(self, retrieval_index):
        from chatbot_rag.pipeline import answer

        history = [
            {"role": "user", "content": [{"type": "text", "text": "Any CMS you guys have worked on?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "We work with several CMS platforms."}]},
        ]
        result = answer(
            "what locations they are based in", lambda _messages: "not used", retrieval_index, history=history
        )
        assert "Dubai" in result.response
        assert "Sitecore" not in result.response

    def test_current_contact_intent_wins_over_previous_location_turn(self, retrieval_index):
        from chatbot_rag.pipeline import answer

        history = [
            {"role": "user", "content": [{"type": "text", "text": "Technyx location?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Our head office is in Dubai."}]},
        ]
        result = answer("any contact number or email?", lambda _messages: "not used", retrieval_index, history=history)
        assert "info@technyxsystems.com" in result.response
        assert "head office" not in result.response.lower()

    def test_specific_payload_mobile_question_is_not_replaced_by_platform_inventory(self, retrieval_index):
        from chatbot_rag.pipeline import answer

        result = answer(
            "Does Technyx work in mobile apps keeping Payload CMS as backend?",
            lambda _messages: "not used",
            retrieval_index,
        )
        assert result.generated
        assert "does not explicitly confirm" in result.response
        assert "Payload" in result.response
