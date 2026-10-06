#!/usr/bin/env python3
"""Convert the base google/gemma-3-1b-it model to a quantized GGUF file for
fast CPU inference via llama.cpp - see scripts/chat_gguf.py.

Why: plain PyTorch/transformers inference on CPU measured ~1.6-2.9
tokens/sec in this project's own testing (README's "Speed" section).
llama.cpp's quantized CPU kernels measured ~10.3 tokens/sec in the sibling
`../lightweight-chatbot` project for the same base model on the same
hardware - about 3.6x faster - plus the model shrinks from ~4GB (fp32) to
well under 1GB (Q4_K_M), a genuine win-win rather than a speed/quality
tradeoff.

Unlike ../lightweight-chatbot's export_gguf.py, there's no LoRA adapter to
merge here - this project never fine-tunes the base model at all, so that
project's "merge" step doesn't apply. Every other step (saving the base
model locally, Gemma's tokenizer vocab-mismatch fix, HF->GGUF conversion,
quantization) uses the same approach that project already measured to
work - see its scripts/export_gguf.py for where this design came from.

Requirements beyond requirements.txt (deliberately not in it - this is an
optional export step, not needed for the default plain-PyTorch path):
    pip install llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git <somewhere>
    pip install -r <that clone>/requirements/requirements-convert_hf_to_gguf.txt
    (in a SEPARATE venv from this project's - the conversion script pins
    transformers/protobuf versions that can conflict with this project's)

Usage:
    python scripts/export_gguf.py \\
        --llama-cpp-dir /path/to/llama.cpp/clone \\
        --llama-cpp-python /path/to/gguf-venv/bin/python \\
        --output outputs/technyx-gemma3-1b-q4_k_m.gguf
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatbot_rag.utils import get_hf_token, load_dotenv_if_present  # noqa: E402

MODEL_NAME = "google/gemma-3-1b-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--llama-cpp-dir",
        required=True,
        type=Path,
        help="Path to a local clone of https://github.com/ggml-org/llama.cpp",
    )
    parser.add_argument(
        "--llama-cpp-python",
        default=sys.executable,
        help="Python interpreter with the llama.cpp conversion requirements installed "
        "(see this script's docstring) - defaults to the current interpreter, but that "
        "usually only works if you installed those requirements into THIS project's venv, "
        "which risks dependency conflicts. A separate venv's python is recommended.",
    )
    parser.add_argument("--output", default="outputs/technyx-gemma3-1b-q4_k_m.gguf", type=Path)
    parser.add_argument(
        "--outtype",
        default="f16",
        choices=["f16", "bf16", "f32"],
        help="Intermediate precision before quantization (f16 is the usual choice)",
    )
    parser.add_argument(
        "--quant-type",
        default="Q4_K_M",
        help="llama.cpp quantization type name, e.g. Q4_K_M, Q5_K_M, Q8_0",
    )
    return parser.parse_args()


def fix_gemma_vocab_mismatch(model_dir: Path) -> None:
    """Gemma 3's shared tokenizer includes <image_soft_token> (a multimodal
    placeholder used by the larger vision-capable checkpoints), which has no
    corresponding row in this text-only model's embedding table - llama.cpp's
    converter asserts on that mismatch. Verified in ../lightweight-chatbot's
    development: this token is unused and unusable by a text-only model
    regardless, so removing it from the saved model's tokenizer files (not
    the original cached HF checkpoint) is a safe, narrow fix rather than a
    workaround for something that actually matters.
    """
    tok_path = model_dir / "tokenizer.json"
    tok = json.loads(tok_path.read_text(encoding="utf-8"))
    before = len(tok["added_tokens"])
    tok["added_tokens"] = [t for t in tok["added_tokens"] if t.get("content") != "<image_soft_token>"]
    if len(tok["added_tokens"]) != before:
        tok_path.write_text(json.dumps(tok, ensure_ascii=False), encoding="utf-8")
        print("Removed unusable <image_soft_token> entry from tokenizer.json (see docstring).")

    cfg_path = model_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    decoder = cfg.get("added_tokens_decoder", {})
    removed = [k for k, v in decoder.items() if v.get("content") == "<image_soft_token>"]
    for k in removed:
        del decoder[k]
    if removed:
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")


def save_base_model(output_dir: Path) -> None:
    """Save the stock base model + tokenizer locally - needed because
    convert_hf_to_gguf.py works off a local directory, and because
    fix_gemma_vocab_mismatch() needs to edit the tokenizer files on disk.
    No fine-tuning happens here - this is the exact same unmodified
    checkpoint scripts/chat.py loads, just persisted to disk instead of
    only living in memory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = get_hf_token()
    print(f"Loading base model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="float32", token=hf_token)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token).save_pretrained(str(output_dir))
    fix_gemma_vocab_mismatch(output_dir)


def convert_to_gguf(model_dir: Path, llama_cpp_dir: Path, llama_cpp_python: str, outtype: str, out_f16: Path) -> None:
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found under {llama_cpp_dir}")
    print(f"Converting to GGUF ({outtype})...")
    subprocess.run(
        [llama_cpp_python, str(convert_script), str(model_dir), "--outfile", str(out_f16), "--outtype", outtype],
        cwd=str(llama_cpp_dir),
        check=True,
    )


def quantize(src: Path, dst: Path, quant_type: str) -> None:
    from llama_cpp import llama_cpp as lib

    ftype_name = f"LLAMA_FTYPE_MOSTLY_{quant_type.upper()}"
    if not hasattr(lib, ftype_name):
        raise ValueError(f"Unknown quant type {quant_type!r} (looked for {ftype_name} in llama_cpp bindings)")

    print(f"Quantizing to {quant_type}...")
    params = lib.llama_model_quantize_default_params()
    params.ftype = getattr(lib, ftype_name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    rc = lib.llama_model_quantize(str(src).encode(), str(dst).encode(), ctypes.byref(params))
    if rc != 0:
        raise RuntimeError(f"llama_model_quantize failed with code {rc}")


def main() -> int:
    load_dotenv_if_present()
    args = parse_args()

    with tempfile.TemporaryDirectory(prefix="gguf_export_") as tmp:
        tmp_path = Path(tmp)
        model_dir = tmp_path / "base_model"
        f16_path = tmp_path / "model-f16.gguf"

        save_base_model(model_dir)
        convert_to_gguf(model_dir, args.llama_cpp_dir, args.llama_cpp_python, args.outtype, f16_path)
        quantize(f16_path, args.output, args.quant_type)

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"\nDone. Wrote {args.output} ({size_mb:.0f} MB).")
    print(f"Test it with: python scripts/chat_gguf.py --model {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
