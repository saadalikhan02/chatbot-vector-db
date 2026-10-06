from chatbot_rag.gguf_llm import _flatten_content


def test_flatten_content_handles_text_parts_and_unknown_shapes():
    assert (
        _flatten_content(
            [{"type": "text", "text": "one"}, {"type": "image", "url": "x"}, {"type": "text", "text": "two"}]
        )
        == "one\ntwo"
    )
    assert _flatten_content(None) == "None"
