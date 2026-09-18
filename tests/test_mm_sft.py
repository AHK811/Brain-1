from brain.experimental.mm_sft import MMSFTExample, write_sample_sft_jsonl, load_mm_sft_jsonl
from pathlib import Path


def test_build_prompt_text():
    ex = MMSFTExample(text="Hi", answer="Hello", domain="nlp")
    p = ex.build_prompt()
    assert "Hi" in p or "Hello" in p


def test_jsonl_roundtrip(tmp_path: Path):
    path = tmp_path / "s.jsonl"
    write_sample_sft_jsonl(path)
    rows = load_mm_sft_jsonl(path)
    assert len(rows) == 3
    assert rows[0].images
