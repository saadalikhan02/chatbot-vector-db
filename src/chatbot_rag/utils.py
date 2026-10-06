"""Small, dependency-light helpers: environment/hardware detection, JSONL
I/O, and Hugging Face token lookup. Kept free of any project-specific data.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_dotenv_if_present(dotenv_path: str | Path = ".env") -> None:
    """Load environment variables from a .env file if python-dotenv and the
    file are both available. Safe no-op otherwise."""
    path = Path(dotenv_path)
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=path)


def get_hf_token() -> str | None:
    """Return a Hugging Face token, checking (in order): the HF_TOKEN env
    var, the standard HUGGING_FACE_HUB_TOKEN env var, then the on-disk
    login cache written by `hf auth login` / `huggingface_hub.notebook_login()`.

    Never logs or returns the token in a printable/truncated form here -
    callers should avoid printing it entirely.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except ImportError:
        return None


@dataclass
class HardwareInfo:
    python_version: str
    torch_version: str | None
    transformers_version: str | None
    cuda_available: bool
    gpu_name: str | None


def detect_hardware() -> HardwareInfo:
    """Inspect the local environment for Python/PyTorch/Transformers/CUDA
    info. Never raises: missing packages simply show up as None."""
    torch_version: str | None = None
    cuda_available = False
    gpu_name: str | None = None

    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    transformers_version: str | None = None
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        pass

    return HardwareInfo(
        python_version=platform.python_version(),
        torch_version=torch_version,
        transformers_version=transformers_version,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
    )


def print_environment_report() -> HardwareInfo:
    hw = detect_hardware()
    print("## Environment")
    print(f"Python:       {hw.python_version}")
    print(f"PyTorch:      {hw.torch_version or 'not installed'}")
    print(f"Transformers: {hw.transformers_version or 'not installed'}")
    cuda_label = f"available ({hw.gpu_name})" if hw.cuda_available and hw.gpu_name else "not available"
    print(f"CUDA:         {cuda_label}")
    return hw


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Raises on invalid JSON so
    problems surface immediately rather than being silently skipped."""
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_num}: invalid JSON ({e.msg})") from e
    return records


def eprint(*args: Any, **kwargs: Any) -> None:
    """Print to stderr (used for warnings/errors so stdout stays clean)."""
    print(*args, file=sys.stderr, **kwargs)
