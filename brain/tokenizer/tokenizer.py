"""
brain/tokenizer/tokenizer.py

Byte-level BPE tokenizer for Brain v0.2.

This carries forward the Brain Trail prototype's tokenizer core (which was
already a genuine, correct BPE implementation via the `tokenizers` library —
the audit found no bugs here, only that it lived as a loose script rather
than a proper package module). What's new relative to the prototype:

  - vocabulary inspection + token statistics (were missing entirely)
  - a real train() convenience that mirrors CLI usage from train.py
  - explicit round-trip and special-token tests colocated in tests/

Deliberately NOT carried forward: pretrained_tokenizer.py's tiktoken wrapper
(Problem #8 in the audit) -- it wrapped a 100K-vocab encoding incompatible
with any model this tokenizer actually produces, and nothing in the real
pipeline ever imported it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer

from brain.tokenizer.special_tokens import ALL_SPECIAL_TOKENS, BOS, EOS, PAD, UNK


class BrainTokenizer:
    """Byte-level BPE tokenizer with the operational features a real
    training/inference pipeline needs: padding, batching, special-token
    handling, and introspection -- not just raw encode/decode.
    """

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer
        self.pad_id = self._tok.token_to_id(PAD)
        self.unk_id = self._tok.token_to_id(UNK)
        self.bos_id = self._tok.token_to_id(BOS)
        self.eos_id = self._tok.token_to_id(EOS)
        for name, tid in [("pad", self.pad_id), ("unk", self.unk_id),
                           ("bos", self.bos_id), ("eos", self.eos_id)]:
            if tid is None:
                raise ValueError(
                    f"Tokenizer is missing required special token '{name}' — "
                    f"was it trained with brain.tokenizer.special_tokens?"
                )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        corpus_paths: list[str] | str,
        vocab_size: int = 16384,
        min_frequency: int = 2,
        save_path: str | Path | None = None,
    ) -> "BrainTokenizer":
        """Train a byte-level BPE tokenizer from scratch on one or more
        text files and (optionally) save it to disk.

        Byte-level pre-tokenization means every raw byte, including spaces,
        becomes part of the symbol stream before merging -- decode() is
        therefore lossless by construction, not by careful bookkeeping.
        """
        if isinstance(corpus_paths, str):
            corpus_paths = [corpus_paths]

        tok = Tokenizer(BPE(unk_token=UNK))
        tok.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=ALL_SPECIAL_TOKENS,
            show_progress=False,
            # Seed the vocabulary with all 256 base byte symbols up front.
            # Without this, the Brain Trail prototype's original approach
            # (confirmed present in train_bpe_tokenizer.py too) only ever
            # learns byte symbols that happen to appear in the training
            # corpus -- any character absent from that corpus (accented
            # letters, em-dashes, CJK text, etc.) then has no token at all
            # and silently falls back to <unk>, corrupting decode. This was
            # caught by tests/test_tokenizer.py::test_unicode_handling,
            # which failed against the corpus-derived-alphabet version.
            # True byte-level BPE (as in GPT-2) must never hit <unk> for
            # any input, by construction -- this is what makes it lossless
            # for genuinely arbitrary text, not just text resembling the
            # training corpus.
            initial_alphabet=ByteLevelPreTokenizer.alphabet(),
        )
        tok.train(corpus_paths, trainer)

        wrapper = cls(tok)
        if save_path is not None:
            wrapper.save(save_path)
        return wrapper

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def from_file(cls, path: str | Path) -> "BrainTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    # ------------------------------------------------------------------
    # Core encode / decode
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = self._tok.encode(text).ids
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(
        self,
        texts: list[str],
        max_length: int | None = None,
        pad: bool = True,
        add_special_tokens: bool = True,
    ) -> dict[str, list[list[int]]]:
        """Encode a batch with padding + attention masks -- what a training
        loop's DataLoader collate function actually needs.
        """
        all_ids = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

        if max_length is None:
            max_length = max(len(ids) for ids in all_ids)

        input_ids, attention_mask = [], []
        for ids in all_ids:
            ids = ids[:max_length]
            n_pad = max_length - len(ids) if pad else 0
            input_ids.append(ids + [self.pad_id] * n_pad)
            attention_mask.append([1] * len(ids) + [0] * n_pad)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode_batch(self, batches: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batches]

    # ------------------------------------------------------------------
    # Introspection (missing entirely from the Brain Trail prototype)
    # ------------------------------------------------------------------
    def get_vocab(self) -> dict[str, int]:
        return self._tok.get_vocab()

    def token_to_id(self, token: str) -> int | None:
        return self._tok.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        return self._tok.id_to_token(token_id)

    def token_stats(self, text: str) -> dict[str, float | int]:
        """Basic statistics used to sanity-check tokenizer quality on a
        piece of text: token count, word count, and fertility (tokens per
        word -- lower is more compressive).
        """
        ids = self.encode(text, add_special_tokens=False)
        n_words = max(len(text.split()), 1)
        return {
            "n_tokens": len(ids),
            "n_words": n_words,
            "fertility": len(ids) / n_words,
            "n_chars": len(text),
            "chars_per_token": len(text) / max(len(ids), 1),
        }

    def most_common_tokens(self, text: str, top_k: int = 10) -> list[tuple[str, int]]:
        """Frequency count of tokens actually used when encoding `text` —
        useful for spotting an undertrained or miscalibrated vocabulary.
        """
        ids = self.encode(text, add_special_tokens=False)
        counts = Counter(self.decode([i], skip_special_tokens=False) for i in ids)
        return counts.most_common(top_k)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    corpus = str(Path(__file__).resolve().parents[2] / "data" / "sample_corpus.txt")
    out_path = str(Path(__file__).resolve().parents[2] / "data" / "brain_tokenizer.json")

    print(f"Training BPE tokenizer on: {corpus}")
    tok = BrainTokenizer.train(corpus, vocab_size=2048, min_frequency=2, save_path=out_path)
    print(f"Saved to: {out_path}")
    print(f"Vocab size: {tok.vocab_size}")

    reloaded = BrainTokenizer.from_file(out_path)
    print(f"Reload matches vocab size: {reloaded.vocab_size == tok.vocab_size}")

    print("\nRound-trip tests:")
    for text in [
        "The quick brown fox jumps over the lazy dog.",
        "def forward(self, x): return self.attn(x) + x",
        "supercalifragilisticexpialidocious tokenization test",
    ]:
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids)
        print(f"  {text!r} -> lossless: {decoded == text}")

    print("\nToken stats on sample corpus:")
    sample_text = Path(corpus).read_text()
    print(tok.token_stats(sample_text))
