"""Tokenize -> concatenate -> fixed-length pack -> binary shard, the
standard "packed sequences" strategy for LM pretraining (used by
nanoGPT/GPT-NeoX-style data prep): every document's tokens are
concatenated into one long stream (documents separated by EOS, not
padding), then sliced into fixed-length blocks. This means every training
example is fully utilized -- no padding waste the way per-document
batching with CausalCollator would have for variable-length documents.

Output format: raw token-id shards written as flat binary files (uint16
if vocab fits, else uint32), one small JSON sidecar per shard describing
how to interpret the bytes. This is intentionally NOT a database or a
fancy container format -- for pretraining-scale corpora, sequential
memory-mapped reads of flat arrays is the fastest and simplest thing that
works, and is what real large-scale pretraining pipelines actually do.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def pack_token_stream(
    documents: Iterable[list[int]],
    seq_len: int,
    eos_id: int,
    drop_last_partial: bool = True,
) -> Iterator[list[int]]:
    """Concatenate tokenized documents (inserting eos_id between them) and
    yield fixed-length seq_len chunks as they become available -- streaming,
    so the full corpus is never held in memory at once.

    drop_last_partial=True (default): the final <seq_len leftover tokens at
    the very end of the whole stream are discarded rather than padded --
    standard practice (e.g. nanoGPT's data prep) since the loss from
    dropping a partial remainder is negligible at real corpus scale, and it
    keeps every yielded sequence a uniform, pad-free length.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len ({seq_len}) must be positive")
    buffer: list[int] = []
    for doc_ids in documents:
        buffer.extend(doc_ids)
        buffer.append(eos_id)
        while len(buffer) >= seq_len:
            yield buffer[:seq_len]
            buffer = buffer[seq_len:]
    if buffer and not drop_last_partial:
        yield buffer  # caller must handle the short final sequence


@dataclass
class ShardWriter:
    """Streams packed sequences to disk as fixed-size binary shards, so a
    corpus far larger than RAM can be written without ever materializing
    the whole thing in memory. Use as a context manager.
    """
    output_dir: str | Path
    seq_len: int
    vocab_size: int
    sequences_per_shard: int = 50_000
    shard_prefix: str = "shard"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = np.uint16 if self.vocab_size <= 65_536 else np.uint32
        self._shard_idx = 0
        self._buffer: list[list[int]] = []
        self._shard_paths: list[Path] = []
        self._total_sequences = 0

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def add(self, sequence: list[int]) -> None:
        if len(sequence) != self.seq_len:
            raise ValueError(
                f"sequence length {len(sequence)} != configured seq_len {self.seq_len}"
            )
        self._buffer.append(sequence)
        self._total_sequences += 1
        if len(self._buffer) >= self.sequences_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        arr = np.array(self._buffer, dtype=self.dtype)  # (n_seqs, seq_len)
        path = self.output_dir / f"{self.shard_prefix}_{self._shard_idx:05d}.bin"
        arr.tofile(path)
        self._shard_paths.append(path)
        self._shard_idx += 1
        self._buffer = []

    def close(self) -> dict:
        self._flush()
        meta = {
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size,
            "dtype": np.dtype(self.dtype).name,
            "num_shards": len(self._shard_paths),
            "total_sequences": self._total_sequences,
            "shard_files": [p.name for p in self._shard_paths],
        }
        (self.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta


def build_shards_from_texts(
    texts: Iterable[str],
    tokenizer,
    output_dir: str | Path,
    context_length: int,
    sequences_per_shard: int = 50_000,
) -> dict:
    """End-to-end: raw cleaned document strings -> tokenized -> packed ->
    written to shards, ready for PackedDataset. `tokenizer` must implement
    `.encode(text)` returning a list of ints and have
    `.eos_id`/`.vocab_size` (matches
    brain.tokenizer.tokenizer.BrainTokenizer's interface).

    context_length is the model's actual context length. Each stored
    sequence is context_length + 1 tokens -- PackedDataset splits that into
    an input_ids/labels pair of exactly context_length each via a
    next-token shift ([:-1] / [1:]). Requesting seq_len=context_length
    directly here would silently hand the model context_length - 1 usable
    tokens per example after shifting, which is why this wrapper adds the
    1 rather than leaving it to the caller to remember.
    """
    def token_docs() -> Iterator[list[int]]:
        for text in texts:
            yield tokenizer.encode(text, add_special_tokens=False)

    stored_seq_len = context_length + 1
    with ShardWriter(output_dir, stored_seq_len, tokenizer.vocab_size, sequences_per_shard) as writer:
        for seq in pack_token_stream(token_docs(), stored_seq_len, tokenizer.eos_id):
            writer.add(seq)
        return writer.close()
