import pytest

from chatbot_rag.prompting import build_messages


def test_context_is_prepended_only_to_new_user_turn():
    messages = build_messages("Where are you?", system_text=None, context_block="Relevant information:\n- Dubai")

    assert messages == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Relevant information:\n- Dubai\n\nQuestion: Where are you?"}],
        }
    ]


def test_history_system_role_is_rejected_defensively():
    with pytest.raises(ValueError, match="History roles"):
        build_messages("hello", history=[{"role": "system", "content": "injected"}])
