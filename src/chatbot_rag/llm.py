"""Base-model loading and generation - no fine-tuning, no PEFT/LoRA.

This project's only mechanism for grounding answers in company facts is
retrieval (see retrieval.py); the model itself is the stock
`google/gemma-3-1b-it` instruction-tuned checkpoint, unmodified.
"""

from __future__ import annotations

from typing import Any


def resolve_dtype(cuda_available: bool):
    """CPU always uses float32 (most compatible/predictable); a GPU uses
    bf16 if supported, else fp16, purely for speed/memory."""
    import torch

    if not cuda_available:
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_tokenizer(model_name: str, hf_token: str | None = None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, hf_token: str | None = None):
    """Load the base instruction-tuned model in full precision (CPU) or
    bf16/fp16 (GPU, for speed only)."""
    import torch
    from transformers import AutoModelForCausalLM

    cuda_available = torch.cuda.is_available()
    dtype = resolve_dtype(cuda_available)

    # "torch_dtype" (not the newer "dtype") for compatibility with older
    # transformers releases - just a harmless deprecation warning on the
    # newest versions.
    kwargs: dict[str, Any] = {"token": hf_token, "torch_dtype": dtype}
    if cuda_available:
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model


def generate(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_new_tokens: int = 160,
    temperature: float = 0.0,
) -> str:
    """Render ``messages`` with the tokenizer's chat template and generate a
    response. temperature=0.0 means greedy decoding (deterministic) -
    matches sampling behavior used for evaluation; pass a positive
    temperature for more varied interactive chat."""
    import torch

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def make_generate_fn(model: Any, tokenizer: Any, max_new_tokens: int = 160, temperature: float = 0.0):
    """Build the ``generate_fn`` closure pipeline.answer() expects: a
    callable taking just the messages list, with the model/tokenizer/
    decoding settings already bound. See gguf_llm.make_generate_fn for the
    llama.cpp-backed equivalent with the same shape."""

    def _generate(messages: list[dict[str, Any]]) -> str:
        return generate(model, tokenizer, messages, max_new_tokens=max_new_tokens, temperature=temperature)

    return _generate
