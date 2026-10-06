"""Cloud, OpenAI-compatible chat-completions backend - a third
``generate_fn`` implementation alongside llm.py (plain PyTorch) and
gguf_llm.py (llama.cpp), for anyone who'd rather call a hosted model than
load weights locally.

Works with any provider exposing an OpenAI-style ``POST {base_url}/chat/
completions`` endpoint - OpenAI itself, DeepSeek, Groq, Together,
Fireworks, OpenRouter, etc. - by pointing ``base_url``/``api_key``/
``model`` at that provider's values (see api.py's lifespan() and the
LLM_BACKEND/LLM_API_* env vars in .env.example). No model weights are
loaded in this process; every call is a network request, so there's
nothing here for load_gguf_model/load_model's local-inference concerns
(dtype, device_map) to apply to.
"""

from __future__ import annotations

from typing import Any

import httpx


class LLMProviderError(RuntimeError):
    """A safe, classified failure returned by the configured LLM provider."""

    def __init__(
        self,
        *,
        provider_status: int | None,
        http_status: int,
        public_detail: str,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(public_detail)
        self.provider_status = provider_status
        self.http_status = http_status
        self.public_detail = public_detail
        self.retry_after = retry_after


def _provider_error(response: httpx.Response) -> LLMProviderError:
    status = response.status_code
    retry_after = response.headers.get("retry-after")
    if retry_after and not retry_after.isdigit():
        retry_after = None
    if status == 429:
        return LLMProviderError(
            provider_status=status,
            http_status=429,
            public_detail="The chatbot is busy right now. Please try again in a few seconds.",
            retry_after=retry_after or "10",
        )
    if status in {401, 403}:
        return LLMProviderError(
            provider_status=status,
            http_status=502,
            public_detail="The chatbot is temporarily unavailable. Please try again later.",
        )
    if 400 <= status < 500:
        return LLMProviderError(
            provider_status=status,
            http_status=502,
            public_detail="I couldn't process that request. Please try again.",
        )
    return LLMProviderError(
        provider_status=status,
        http_status=503,
        public_detail="I'm temporarily unable to generate a response. Please try again in a moment.",
        retry_after=retry_after,
    )


def _flatten_content(content: Any) -> str:
    """Same conversion as gguf_llm._flatten_content - this project's
    messages (prompting.build_messages) use the list-of-parts form HF's
    apply_chat_template expects; OpenAI-compatible APIs expect a plain
    string instead."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def generate(
    messages: list[dict[str, Any]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_new_tokens: int = 160,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint and return the
    assistant's text. Raises httpx.HTTPStatusError on a non-2xx response
    (caught by api.py's existing generic error handling - same as any
    other exception generate_fn can raise)."""
    flat_messages = [{"role": m["role"], "content": _flatten_content(m["content"])} for m in messages]
    payload: dict[str, Any] = {
        "model": model,
        "messages": flat_messages,
        "max_tokens": max_new_tokens,
    }
    # Gemini 3.8 rejects deprecated sampling parameters in its OpenAI-
    # compatible endpoint. Other providers continue to receive temperature.
    if "generativelanguage.googleapis.com" not in base_url.lower():
        payload["temperature"] = temperature
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=timeout,
        )
    except httpx.TimeoutException as exc:
        raise LLMProviderError(
            provider_status=None,
            http_status=503,
            public_detail="I'm temporarily unable to generate a response. Please try again in a moment.",
            retry_after="5",
        ) from exc
    except httpx.RequestError as exc:
        raise LLMProviderError(
            provider_status=None,
            http_status=503,
            public_detail="I'm temporarily unable to generate a response. Please try again in a moment.",
            retry_after="5",
        ) from exc

    if response.is_error:
        raise _provider_error(response)
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMProviderError(
            provider_status=response.status_code,
            http_status=502,
            public_detail="I'm temporarily unable to generate a response. Please try again in a moment.",
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise LLMProviderError(
            provider_status=response.status_code,
            http_status=502,
            public_detail="I'm temporarily unable to generate a response. Please try again in a moment.",
        )
    return content.strip()


def make_generate_fn(
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_new_tokens: int = 160,
    temperature: float = 0.0,
):
    """Build the ``generate_fn`` closure pipeline.answer() expects - same
    shape as llm.make_generate_fn / gguf_llm.make_generate_fn, backed by a
    remote API call instead of a locally loaded model."""

    def _generate(messages: list[dict[str, Any]]) -> str:
        return generate(
            messages,
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    return _generate
