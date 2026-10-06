"""llama.cpp-backed generation - the fast path.

Plain PyTorch/transformers inference on CPU (llm.py) measured ~1.6-2.9
tokens/sec in this project's own testing, meaning a full answer can take
60-100+ seconds - not viable latency for live chat. The sibling
`../lightweight-chatbot` project measured llama.cpp's quantized CPU
kernels at ~10.3 tok/s for the same base model on the same hardware - a
~3.6x speedup, plus the model shrinks from ~4GB (fp32) to well under 1GB
(Q4_K_M) - see its README's "Speed" section. scripts/export_gguf.py
produces the GGUF file this module loads; scripts/chat_gguf.py is the
interactive entry point.

Requires `llama-cpp-python`, deliberately NOT in requirements.txt - same
reasoning as the sibling project: this is an optional export/inference
path, not needed for the default (plain PyTorch) path this project uses
by default. See scripts/export_gguf.py's docstring for the install
command.

Returns the exact same ``generate_fn`` shape as llm.make_generate_fn, so
pipeline.answer() - and everything it guarantees (guardrails.py,
fact_check.py) - works identically regardless of which backend is used.
"""

from __future__ import annotations

from typing import Any


def load_gguf_model(model_path: str, n_ctx: int = 2048, n_threads: int = 6):
    """Load a GGUF file produced by scripts/export_gguf.py.

    n_threads=6 (not the CPU's full thread count) matches a measurement
    in the sibling project's README: more threads made generation
    *slower* on a 6-core/12-thread CPU (hyperthreaded "extra" threads add
    contention, not throughput, for this kind of compute) - tune this to
    your own CPU's physical core count if it differs.
    """
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(
            "llama-cpp-python is not installed. Run:\n"
            "  pip install llama-cpp-python "
            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
        ) from e

    return Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)


def _flatten_content(content: Any) -> str:
    """llama-cpp-python's create_chat_completion expects OpenAI-style
    messages (``content`` is a plain string); this project's messages
    (prompting.build_messages) use the list-of-parts form
    ``[{"type": "text", "text": "..."}]`` that HF's apply_chat_template
    expects instead. Converting here keeps that format difference a
    backend-specific detail rather than leaking into pipeline.py, which
    stays generation-backend-agnostic."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def generate(llm: Any, messages: list[dict[str, Any]], max_new_tokens: int = 160, temperature: float = 0.0) -> str:
    """Generate a response from a loaded GGUF model. temperature=0.0 is
    still effectively greedy here (llama.cpp treats near-zero temperature
    as greedy decoding), matching llm.generate's convention."""
    flat_messages = [{"role": m["role"], "content": _flatten_content(m["content"])} for m in messages]
    output = llm.create_chat_completion(
        messages=flat_messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
        repeat_penalty=1.1,
    )
    return output["choices"][0]["message"]["content"].strip()


def make_generate_fn(llm: Any, max_new_tokens: int = 160, temperature: float = 0.0):
    """Build the ``generate_fn`` closure pipeline.answer() expects - the
    llama.cpp-backed equivalent of llm.make_generate_fn, same shape."""

    def _generate(messages: list[dict[str, Any]]) -> str:
        return generate(llm, messages, max_new_tokens=max_new_tokens, temperature=temperature)

    return _generate
