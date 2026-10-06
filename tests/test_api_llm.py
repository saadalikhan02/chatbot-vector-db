"""Tests for the OpenAI-compatible provider adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_gemini_payload_omits_temperature(monkeypatch):
    from chatbot_rag.api_llm import generate

    captured = {}

    class FakeResponse:
        status_code = 200
        is_error = False

        def json(self):
            return {"choices": [{"message": {"content": "grounded"}}]}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, json=json)
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    assert (
        generate(
            [{"role": "user", "content": "Question"}],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key="secret",
            model="gemini-3.8-flash",
        )
        == "grounded"
    )
    assert "temperature" not in captured["json"]


@pytest.mark.parametrize(
    ("provider_status", "expected_status"),
    [(401, 502), (400, 502), (429, 429), (500, 503)],
)
def test_provider_http_errors_are_classified(monkeypatch, provider_status, expected_status):
    from chatbot_rag.api_llm import LLMProviderError, generate

    request = httpx.Request("POST", "https://provider.example/chat/completions")
    response = httpx.Response(provider_status, request=request, headers={"retry-after": "7"})

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)
    with pytest.raises(LLMProviderError) as exc_info:
        generate(
            [{"role": "user", "content": "Question"}],
            base_url="https://provider.example",
            api_key="secret",
            model="test-model",
        )
    assert exc_info.value.http_status == expected_status
    if provider_status == 429:
        assert exc_info.value.retry_after == "7"
        assert "busy" in str(exc_info.value)
    if provider_status == 500:
        assert "temporarily unable" in str(exc_info.value)


def test_provider_timeout_is_retryable(monkeypatch):
    from chatbot_rag.api_llm import LLMProviderError, generate

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ReadTimeout("timed out")),
    )
    with pytest.raises(LLMProviderError) as exc_info:
        generate(
            [{"role": "user", "content": "Question"}],
            base_url="https://provider.example",
            api_key="secret",
            model="test-model",
        )
    assert exc_info.value.http_status == 503
