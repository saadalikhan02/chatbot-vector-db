import pytest

from chatbot_rag.utils import read_jsonl


def test_read_jsonl_reports_line_for_invalid_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"bad\.jsonl:2: invalid JSON"):
        read_jsonl(path)
