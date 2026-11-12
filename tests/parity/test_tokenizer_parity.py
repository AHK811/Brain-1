from __future__ import annotations

import os
from pathlib import Path

import pytest

from brainlab.spec.config import repo_root
from brainlab.tok.parity_fixture import build_fixture, build_tokenizer, load_fixture, verify_against_file
from brainlab.tok.reference import BYTE_TOKENS, Tokenizer, pre_tokenize, split_digit_run, word_chunks, write_bin

FIXTURE = repo_root() / "tests" / "golden" / "tokenizer_parity.json"
RUST_TOKENIZER_DIR = os.environ.get("BRAINLAB_TOKENIZER_DIR", "")


@pytest.fixture(scope="module")
def fixture() -> dict:
    if not FIXTURE.is_file():
        pytest.skip(f"{FIXTURE} missing; run `python -m brainlab.tok.parity_fixture`")
    return load_fixture(FIXTURE)


def tokenizer_from_fixture(payload: dict) -> Tokenizer:
    tokens = tuple(bytes.fromhex(text) for text in payload["tokens_hex"])
    merges = tuple((int(pair[0]), int(pair[1])) for pair in payload["merges"])
    return Tokenizer(
        token_bytes=tokens,
        merges=merges,
        eos_id=int(payload["eos_id"]),
        pad_id=int(payload["pad_id"]),
    )


def test_python_reference_reproduces_the_shared_fixture(fixture: dict) -> None:
    problems = verify_against_file(build_fixture(), FIXTURE)
    assert not problems, f"the Python tokenizer drifted from the committed fixture in {problems}"


def test_fixture_vocab_is_rebuildable_and_valid(fixture: dict) -> None:
    rebuilt = tokenizer_from_fixture(fixture)
    original = build_tokenizer()
    assert rebuilt.token_bytes == original.token_bytes
    assert rebuilt.merges == original.merges
    assert rebuilt.eos_id == fixture["eos_id"] == len(fixture["tokens_hex"]) - 1
    assert rebuilt.pad_id == fixture["pad_id"] == len(fixture["tokens_hex"]) - 2
    assert rebuilt.token_bytes[:BYTE_TOKENS] == tuple(bytes([index]) for index in range(BYTE_TOKENS))


def test_pre_tokenize_and_encode_match_the_fixture(fixture: dict) -> None:
    tokenizer = tokenizer_from_fixture(fixture)
    for index, doc in enumerate(fixture["docs"]):
        assert pre_tokenize(doc) == fixture["pre_tokenize"][index], f"doc {index} pre-tokenized differently"
        assert tokenizer.encode(doc) == fixture["ids"][index], f"doc {index} encoded differently"


def test_ids_decode_back_to_the_original_text(fixture: dict) -> None:
    tokenizer = tokenizer_from_fixture(fixture)
    for index, doc in enumerate(fixture["docs"]):
        assert tokenizer.decode_ids(fixture["ids"][index]) == doc, f"doc {index} did not roundtrip"


def test_bin_format_has_no_trailing_bytes_and_roundtrips(tmp_path: Path) -> None:
    tokenizer = build_tokenizer()
    path = write_bin(tokenizer, tmp_path / "tokenizer.bin")
    raw = path.read_bytes()
    assert raw[:4] == b"BLTK"
    loaded = Tokenizer.from_bin(path)
    assert loaded.token_bytes == tokenizer.token_bytes
    assert loaded.merges == tokenizer.merges
    for doc in ("the model trains", "1234567", "日本語"):
        assert loaded.encode(doc) == tokenizer.encode(doc)


def test_word_chunking_never_splits_a_code_point() -> None:
    word = "日本語" * 12
    chunks = word_chunks(word, 32)
    assert sum(len(chunk) for chunk in chunks) == len(word.encode("utf-8"))
    for chunk in chunks:
        chunk.decode("utf-8")
    assert all(len(chunk) <= 32 for chunk in chunks)
    assert b"".join(chunks) == word.encode("utf-8")


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("1234567", ["1", "234", "567"]),
        ("1234", ["1", "234"]),
        ("123", ["123"]),
        ("42", ["42"]),
        ("12a34", ["12a34"]),
        ("٠١٢٣٤٥٦", ["٠١٢٣٤٥٦"]),
    ],
)
def test_digit_runs_split_from_the_right_in_threes(word: str, expected: list[str]) -> None:
    assert split_digit_run(word) == expected


@pytest.mark.skipif(not RUST_TOKENIZER_DIR, reason="set BRAINLAB_TOKENIZER_DIR to a rust-trained vocab to cross-check")
def test_rust_trained_vocab_is_self_consistent_across_its_two_forms() -> None:
    directory = Path(RUST_TOKENIZER_DIR)
    from_bin = Tokenizer.load(directory)
    from_files = Tokenizer.from_vocab_files(directory / "vocab.json", directory / "merges.txt")
    assert from_bin.token_bytes == from_files.token_bytes, "tokenizer.bin and vocab.json describe different vocabs"
    assert from_bin.merges == from_files.merges, "tokenizer.bin and merges.txt describe different merges"
    assert from_bin.eos_id == from_files.eos_id and from_bin.pad_id == from_files.pad_id
    documents = [doc for doc in load_fixture(FIXTURE)["docs"][:20]] if FIXTURE.is_file() else ["the model trains"]
    for doc in documents:
        assert from_bin.encode(doc) == from_files.encode(doc), f"{doc!r} encoded differently by the two forms"
        assert from_bin.decode_ids(from_bin.encode(doc)) == doc
