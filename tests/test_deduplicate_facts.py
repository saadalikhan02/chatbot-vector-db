import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "deduplicate_facts.py"
SPEC = importlib.util.spec_from_file_location("deduplicate_facts", SCRIPT)
assert SPEC and SPEC.loader
deduplicate_facts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deduplicate_facts)


def test_keeps_duplicate_record_with_richer_metadata():
    records = [
        {"fact": "React", "source_url": ""},
        {"fact": "React", "source_url": "https://example.test", "source_page_title": "Technology"},
    ]

    result = deduplicate_facts.deduplicate(records)

    assert result == [records[1]]
