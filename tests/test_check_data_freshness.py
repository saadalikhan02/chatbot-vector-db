"""Tests for scripts/check_data_freshness.py.

Fast: only calls `git log` against files already in this repo's history
(instant) - no model, no embedding, no network.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_data_freshness import last_git_update  # noqa: E402


class TestLastGitUpdate:
    def test_returns_a_datetime_for_a_tracked_file(self):
        result = last_git_update("data/knowledge/facts.jsonl")
        assert isinstance(result, datetime)

    def test_returns_none_for_a_path_with_no_history(self):
        assert last_git_update("this/path/does/not/exist.jsonl") is None
