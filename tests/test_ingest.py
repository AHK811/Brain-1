from pathlib import Path
from brain.experimental.ingest import load_any, ContentBlock, Modality
from brain.experimental.ocr import get_ocr_engine


def test_load_text(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello brain v0.4")
    r = load_any(f)
    assert not r.errors
    texts = [b.text for b in r.blocks if b.modality == Modality.TEXT]
    assert any("hello brain" in t for t in texts)


def test_load_csv(tmp_path: Path):
    f = tmp_path / "t.csv"
    f.write_text("a,b\n1,2\n3,4\n")
    r = load_any(f)
    assert any(b.modality == Modality.TABLE for b in r.blocks)


def test_load_missing():
    r = load_any("/tmp/does_not_exist_brain_v04.bin")
    assert r.errors


def test_ocr_engine_returns_something():
    eng = get_ocr_engine()
    assert eng.name in ("stub", "tesseract", "easyocr", "paddle")


def test_prompt_truncation(tmp_path: Path):
    f = tmp_path / "long.txt"
    f.write_text("x" * 100)
    r = load_any(f)
    prompt = r.to_prompt(max_chars=50)
    assert len(prompt) <= 80  # meta + truncate marker allowance
