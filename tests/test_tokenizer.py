"""
tests/test_tokenizer.py

Automated tests for BrainTokenizer -- this is what audit Problem #4 (zero
automated tests anywhere in Brain Trail) is fixed by. Every property that
used to be "eyeballed via a printed __main__ block" now runs under pytest
and fails loudly (not silently) if broken.
"""

from pathlib import Path

import pytest

from brain.tokenizer.tokenizer import BrainTokenizer
from brain.tokenizer.special_tokens import PAD, UNK, BOS, EOS

CORPUS_PATH = str(Path(__file__).resolve().parents[1] / "data" / "sample_corpus.txt")


@pytest.fixture(scope="module")
def tokenizer():
    return BrainTokenizer.train(CORPUS_PATH, vocab_size=1024, min_frequency=1)


def test_special_tokens_present(tokenizer):
    for tok, tok_id in [(PAD, tokenizer.pad_id), (UNK, tokenizer.unk_id),
                          (BOS, tokenizer.bos_id), (EOS, tokenizer.eos_id)]:
        assert tok_id is not None
        assert tokenizer.id_to_token(tok_id) == tok


def test_vocab_size_positive(tokenizer):
    assert tokenizer.vocab_size > len([PAD, UNK, BOS, EOS])


@pytest.mark.parametrize("text", [
    "Hello Brain.",
    "The quick brown fox jumps over the lazy dog.",
    "def forward(self, x): return self.attn(x) + x",
    "SELECT * FROM users WHERE id = 1;",
    "Numbers: 100%, $50.25, user@example.com",
    "",  # edge case: empty string
    "a",  # edge case: single character
])
def test_roundtrip_lossless(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    assert decoded == text


def test_add_special_tokens_wraps_with_bos_eos(tokenizer):
    ids = tokenizer.encode("Hello Brain.", add_special_tokens=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id


def test_batch_encode_padding_shapes(tokenizer):
    batch = tokenizer.encode_batch(["short", "a much longer piece of text here"])
    lengths = {len(row) for row in batch["input_ids"]}
    assert len(lengths) == 1, "all rows in a padded batch must be equal length"
    assert len(batch["input_ids"]) == len(batch["attention_mask"]) == 2


def test_batch_encode_attention_mask_marks_padding(tokenizer):
    batch = tokenizer.encode_batch(["a", "a much longer sentence than the first one"])
    short_mask = batch["attention_mask"][0]
    assert 0 in short_mask, "shorter sequence should have padding positions masked as 0"


def test_save_and_load_roundtrip(tokenizer, tmp_path):
    save_path = tmp_path / "tok.json"
    tokenizer.save(save_path)
    reloaded = BrainTokenizer.from_file(save_path)
    assert reloaded.vocab_size == tokenizer.vocab_size
    text = "Round trip through disk."
    assert reloaded.decode(reloaded.encode(text, add_special_tokens=False)) == text


def test_unicode_handling(tokenizer):
    text = "Brægn — café — 日本語"
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.decode(ids) == text


def test_token_stats_reasonable(tokenizer):
    stats = tokenizer.token_stats("The quick brown fox jumps over the lazy dog.")
    assert stats["n_tokens"] > 0
    assert stats["fertility"] > 0
