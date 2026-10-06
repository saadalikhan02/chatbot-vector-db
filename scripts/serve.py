#!/usr/bin/env python3
"""Run the FastAPI serving layer - see src/chatbot_rag/api.py and
README's "Serving" section.

Usage:
    pip install -r requirements-serving.txt
    export API_KEY=<a-secret-you-choose>       # required - see api.py
    export GGUF_MODEL_PATH=outputs/technyx-gemma3-1b-q4_k_m.gguf  # optional, faster
    python scripts/serve.py

Or directly with uvicorn (equivalent, useful for --reload during
development):
    uvicorn chatbot_rag.api:app --app-dir src --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    import uvicorn

    if not os.environ.get("API_KEY"):
        print(
            "WARNING: API_KEY is not set - every request will be rejected with 503. "
            "Set it before deploying anywhere reachable: export API_KEY=<a-secret-you-choose>",
            file=sys.stderr,
        )

    host = os.environ.get("HOST", "127.0.0.1")  # not 0.0.0.0 by default - opt in explicitly
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("chatbot_rag.api:app", host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
