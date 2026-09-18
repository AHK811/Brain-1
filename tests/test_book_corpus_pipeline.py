"""Regression tests for the book-corpus data pipeline built for training
data preparation: cleaning (Gutenberg boilerplate + quality filter),
deduplication (exact + MinHash near-dup), and pack/shard/read.
"""
import random

import pytest
import torch

from Datasets.Processing.cleaning import clean_text, is_low_quality, strip_gutenberg_boilerplate
from Datasets.Processing.deduplication import MinHashLSH, exact_hash
from Datasets.Processing.packing import ShardWriter, build_shards_from_texts, pack_token_stream


# ---------------------------------------------------------------------------
# cleaning.py
# ---------------------------------------------------------------------------
def test_clean_text_default_args_unchanged_from_original_stub():
    """The original clean_text(text) call site (Datasets.Pipelines.
    ingestion_pipeline) must keep getting byte-identical output."""
    raw = "Hello\x00World.   Too   many   spaces.\n\n\n\nExtra newlines.  "
    assert clean_text(raw) == "Hello World. Too many spaces.\n\nExtra newlines."


def test_strip_gutenberg_boilerplate_removes_legal_text_keeps_content():
    text = (
        "The Project Gutenberg eBook of X\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
        "Real book content here.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
        "License information..."
    )
    stripped = strip_gutenberg_boilerplate(text)
    assert "Real book content here." in stripped
    assert "PROJECT GUTENBERG" not in stripped
    assert "License information" not in stripped


def test_strip_gutenberg_boilerplate_leaves_text_untouched_if_no_markers():
    text = "Just a plain text file with no Gutenberg markers at all."
    assert strip_gutenberg_boilerplate(text) == text


def test_is_low_quality_catches_short_and_non_alpha_text():
    assert is_low_quality("12 34 ## $$ !!") is True
    assert is_low_quality("This is real prose. " * 100) is False


# ---------------------------------------------------------------------------
# deduplication.py
# ---------------------------------------------------------------------------
def test_exact_hash_matches_identical_differs_on_change():
    a = "The quick brown fox. " * 20
    b = a
    c = a.replace("quick", "slow")
    assert exact_hash(a) == exact_hash(b)
    assert exact_hash(a) != exact_hash(c)


def test_minhash_lsh_catches_near_duplicate_and_lets_different_text_through():
    random.seed(42)
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
              "story", "chapter", "once", "upon", "time", "princess", "forest"]
    book_a = " ".join(random.choices(words, k=2000))
    words_a = book_a.split()

    near_dup_words = words_a.copy()
    random.seed(1)
    for idx in random.sample(range(len(near_dup_words)), int(0.02 * len(near_dup_words))):
        near_dup_words[idx] = random.choice(words)
    near_dup = " ".join(near_dup_words)

    different_words = words_a.copy()
    random.seed(2)
    for idx in random.sample(range(len(different_words)), int(0.40 * len(different_words))):
        different_words[idx] = random.choice(words)
    different = " ".join(different_words)

    lsh = MinHashLSH(num_hashes=128, num_bands=16, jaccard_threshold=0.7)
    assert lsh.is_duplicate_and_add("a", book_a) is False
    assert lsh.is_duplicate_and_add("near_dup", near_dup) is True
    assert lsh.is_duplicate_and_add("different", different) is False


def test_minhash_lsh_rejects_bad_band_config():
    with pytest.raises(ValueError):
        MinHashLSH(num_hashes=100, num_bands=7)  # 100 not divisible by 7


# ---------------------------------------------------------------------------
# packing.py + packed_dataset.py
# ---------------------------------------------------------------------------
def test_pack_token_stream_concatenates_and_chunks_with_eos():
    docs = [[1, 2, 3], [4, 5]]
    chunks = list(pack_token_stream(docs, seq_len=4, eos_id=0))
    # stream: 1 2 3 0 4 5 0  (len 7) -> one full chunk of 4, remainder of 3 dropped by default
    assert chunks == [[1, 2, 3, 0]]


def test_pack_token_stream_keep_partial():
    docs = [[1, 2, 3]]
    chunks = list(pack_token_stream(docs, seq_len=4, eos_id=0, drop_last_partial=False))
    assert chunks == [[1, 2, 3, 0]]


def test_shard_writer_rejects_wrong_length_sequence(tmp_path):
    with ShardWriter(tmp_path, seq_len=4, vocab_size=100) as w:
        with pytest.raises(ValueError):
            w.add([1, 2, 3])  # wrong length


def test_shard_writer_dtype_selection(tmp_path):
    with ShardWriter(tmp_path / "small_vocab", seq_len=2, vocab_size=1000) as w:
        w.add([1, 2])
        meta = w.close()
    assert meta["dtype"] == "uint16"

    with ShardWriter(tmp_path / "big_vocab", seq_len=2, vocab_size=200_000) as w:
        w.add([1, 2])
        meta = w.close()
    assert meta["dtype"] == "uint32"


def test_build_shards_from_texts_end_to_end_and_read_back(tmp_path):
    from brain.tokenizer.tokenizer import BrainTokenizer
    from brain.data.packed_dataset import PackedDataset

    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("\n".join(f"Sentence number {i} for training." for i in range(30)))
    tok = BrainTokenizer.train(
        corpus_paths=[str(corpus_path)], vocab_size=300, min_frequency=1,
        save_path=str(tmp_path / "tok.json"),
    )

    texts = [f"Document {i}. " * 10 for i in range(20)]
    context_length = 16
    shard_dir = tmp_path / "shards"
    meta = build_shards_from_texts(texts, tok, shard_dir, context_length=context_length, sequences_per_shard=3)

    ds = PackedDataset(shard_dir)
    assert ds.context_length == context_length  # not context_length - 1 (the off-by-one this guards against)
    assert len(ds) == meta["total_sequences"]

    ex0 = ds[0]
    assert ex0["input_ids"].shape == (context_length,)
    assert ex0["labels"].shape == (context_length,)
    # next-token shift is internally consistent
    assert torch.equal(ex0["labels"][:-1], ex0["input_ids"][1:])

    # shard-boundary index (sequences_per_shard=3) doesn't crash or misalign
    ex_boundary = ds[3]
    assert ex_boundary["input_ids"].shape == (context_length,)

    with pytest.raises(IndexError):
        ds[len(ds)]


def test_packed_dataset_feeds_real_model_forward_backward(tmp_path):
    from torch.utils.data import DataLoader
    from brain.tokenizer.tokenizer import BrainTokenizer
    from brain.data.packed_dataset import PackedDataset
    from brain.core.config import BrainConfig
    from brain.model.brain_model import BrainForCausalLM

    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("\n".join(f"Sentence number {i} for training." for i in range(30)))
    tok = BrainTokenizer.train(
        corpus_paths=[str(corpus_path)], vocab_size=300, min_frequency=1,
        save_path=str(tmp_path / "tok.json"),
    )
    texts = [f"Document {i}. " * 8 for i in range(20)]
    context_length = 16
    build_shards_from_texts(texts, tok, tmp_path / "shards", context_length=context_length, sequences_per_shard=3)

    ds = PackedDataset(tmp_path / "shards")
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=True)))

    cfg = BrainConfig(vocab_size=tok.vocab_size, hidden_size=32, num_layers=2, num_heads=4,
                       num_kv_heads=2, intermediate_size=64, max_position_embeddings=context_length)
    model = BrainForCausalLM(cfg)
    model.train()
    out = model(batch["input_ids"], labels=batch["labels"])
    out["loss"].backward()
    assert model.token_embedding.embedding.weight.grad is not None
