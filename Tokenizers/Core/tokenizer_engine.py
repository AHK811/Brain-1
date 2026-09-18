from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
from Tokenizers.Normalization.normalization_pipeline import normalize
from Tokenizers.Encoding.encoder import SimpleEncoder
from Tokenizers.Vocabulary.special_tokens import ALL_SPECIAL

class TokenizerEngine:
    """Facade: prefer brain.tokenizer / HF tokenizers, else SimpleEncoder."""
    def __init__(self, backend: Any = None):
        self.backend = backend
        self._simple = SimpleEncoder()
    @classmethod
    def from_file(cls, path: str | Path) -> "TokenizerEngine":
        try:
            from brain.tokenizer.tokenizer import BrainTokenizer
            return cls(backend=BrainTokenizer.from_file(path))
        except Exception:
            return cls()
    @classmethod
    def train_bpe(cls, corpus_paths, vocab_size: int = 32768, save_path: str | None = None):
        try:
            from brain.tokenizer.tokenizer import BrainTokenizer
            tok = BrainTokenizer.train(corpus_paths, vocab_size=vocab_size, save_path=save_path)
            return cls(backend=tok)
        except Exception as e:
            eng = cls()
            eng._train_error = str(e)
            return eng
    def encode(self, text: str, add_special_tokens: bool = True, *, do_normalize: bool = True) -> list[int]:
        if do_normalize:
            text = normalize(text)
        if self.backend is not None and hasattr(self.backend, "encode"):
            return list(self.backend.encode(text, add_special_tokens=add_special_tokens))
        return self._simple.encode(text, add_special=add_special_tokens)
    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if self.backend is not None and hasattr(self.backend, "decode"):
            return self.backend.decode(list(ids), skip_special_tokens=skip_special_tokens)
        return self._simple.decode(list(ids), skip_special=skip_special_tokens)
    @property
    def vocab_size(self) -> int:
        if self.backend is not None and hasattr(self.backend, "vocab_size"):
            return int(self.backend.vocab_size)
        return len(self._simple.vocab)
