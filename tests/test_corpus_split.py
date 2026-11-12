from __future__ import annotations

from pathlib import Path

import pytest

from brainlab.data.corpus import documents_from_text, split_files, val_stride
from brainlab.data.pack import PackConfig
from brainlab.data.prepare import byte_tokenizer, prepare

WORDS = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota", "kappa lambda mu"]


def files(count: int) -> list[tuple[str, str]]:
    return [(f"group/file-{index:04d}.txt", f"{WORDS[index % len(WORDS)]} {index}") for index in range(count)]


def test_a_small_corpus_cannot_silently_become_half_val() -> None:
    with pytest.raises(ValueError, match="give 50.0% val, wanted 0.5%"):
        split_files(files(2), 0.005)
    with pytest.raises(ValueError, match="at least 200 files"):
        split_files(files(5), 0.005)


def test_a_coarse_split_inside_tolerance_is_accepted() -> None:
    train, val = split_files(files(5), 0.25)
    assert (len(train), len(val)) == (3, 2)
    assert 2 / 5 <= 3 * 0.25


def test_val_ratio_needs_the_inverse_number_of_files() -> None:
    assert val_stride(0.005) == 200
    assert val_stride(0.25) == 4
    train, val = split_files(files(400), 0.005)
    assert len(val) == 2 and len(train) == 398


def test_document_separation_is_a_choice_not_an_assumption() -> None:
    paragraphs = "first para line one\nstill first para\n\nsecond para\n\nthird para\n"
    assert documents_from_text(paragraphs) == ["first para line one\nstill first para", "second para", "third para"]
    assert len(documents_from_text(paragraphs, doc_sep="line")) == 4
    assert documents_from_text(paragraphs, doc_sep="file") == [paragraphs.strip()]
    joined = "one two three<|endoftext|>four five<|endoftext|>"
    assert documents_from_text(joined, doc_sep="<|endoftext|>") == ["one two three", "four five"]


def test_prepare_refuses_a_tokenizer_trained_on_the_val_split(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    for index in range(4):
        path = corpus / f"file-{index}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"sentence number {index} about something else\n", encoding="utf-8")
    cfg = PackConfig(seq_len=16, shard_tokens=16 * 8, eos_id=byte_tokenizer().eos_id)
    with pytest.raises(SystemExit, match="trained on"):
        prepare(
            [(path.name, path.read_text(encoding="utf-8")) for path in sorted(corpus.glob("*.txt"))],
            byte_tokenizer(),
            tmp_path / "packed",
            cfg,
            0.5,
            engine="python",
            tokenizer_corpus=corpus,
        )


def test_prepare_accepts_a_tokenizer_trained_only_on_train_files(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    train_only = tmp_path / "train-corpus"
    for index in range(4):
        for root in (corpus, train_only):
            path = root / f"file-{index}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"sentence number {index} about something else\n", encoding="utf-8")
    for index in (0, 2):
        (train_only / f"file-{index}.txt").unlink()
    cfg = PackConfig(seq_len=16, shard_tokens=16 * 8, eos_id=byte_tokenizer().eos_id)
    report = prepare(
        [(path.name, path.read_text(encoding="utf-8")) for path in sorted(corpus.glob("*.txt"))],
        byte_tokenizer(),
        tmp_path / "packed",
        cfg,
        0.5,
        engine="python",
        doc_sep="line",
        tokenizer_corpus=train_only,
    )
    assert report["val_fraction"] == 0.5
    assert report["doc_sep"] == "line"
    assert report["val_documents"] == 2 and report["train_documents"] == 2
