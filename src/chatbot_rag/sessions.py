"""Small, in-process conversation state for the public widget.

This avoids introducing Redis for a single-process deployment. State is lost
on restart and is not shared by replicas; use a shared store before either
property becomes unacceptable. Bounds prevent a long-running public widget
from growing process memory without limit.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

_log_lock = threading.Lock()


class ConversationStore:
    def __init__(self, max_sessions: int = 1_000, ttl_seconds: float = 3_600, max_messages: int = 40):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages
        self._sessions: OrderedDict[str, tuple[float, list[dict[str, Any]]]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, conversation_id: str) -> list[dict[str, Any]] | None:
        with self._lock:
            item = self._sessions.get(conversation_id)
            if item is None or item[0] <= time.monotonic():
                self._sessions.pop(conversation_id, None)
                return None
            self._sessions.move_to_end(conversation_id)
            return [dict(message) for message in item[1]]

    def append(self, conversation_id: str, user_text: str, assistant_text: str) -> None:
        with self._lock:
            existing = self._sessions.get(conversation_id)
            history = [] if existing is None or existing[0] <= time.monotonic() else list(existing[1])
            history.extend(
                [
                    {"role": "user", "content": [{"type": "text", "text": user_text}]},
                    {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
                ]
            )
            self._sessions[conversation_id] = (time.monotonic() + self.ttl_seconds, history[-self.max_messages :])
            self._sessions.move_to_end(conversation_id)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)


def append_conversation_log(path: str | Path, record: dict[str, Any]) -> None:
    """Append JSONL review data. Messages can contain visitor PII: keep the
    path private, set a retention policy outside this process, and never add
    it to git. A lock makes each local record write coherent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock:
        with path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
