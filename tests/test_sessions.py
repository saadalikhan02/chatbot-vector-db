from chatbot_rag.sessions import ConversationStore


def test_session_continues_history():
    store = ConversationStore()
    store.append("session", "Where are you?", "Dubai.")

    history = store.get("session")

    assert history is not None
    assert history[0]["role"] == "user"
    assert history[1]["content"][0]["text"] == "Dubai."
