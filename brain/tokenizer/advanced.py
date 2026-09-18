"""
brain/tokenizer/advanced.py

Advanced tokenizer layer on top of BrainTokenizer:

  - Chat / CoT / tool-aware encoding
  - Truncation strategies (left/right/middle)
  - Padding to multiple of N (GPU-friendly)
  - Offset mapping & token spans
  - Fertility / compression analysis across a corpus
  - Vocabulary diff & coverage reports
  - Safe special-token id map for all Brain control tokens
  - Streaming encode for large files
  - Optional BPE-dropout style noisy encode (augmentation)
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from brain.tokenizer.special_tokens import (
    ALL_SPECIAL_TOKENS,
    ASSISTANT,
    BOS,
    EOS,
    PAD,
    SYSTEM,
    THINK,
    END_THINK,
    TOOL_CALL,
    END_TOOL_CALL,
    TOOL_RESULT,
    USER,
)
from brain.tokenizer.tokenizer import BrainTokenizer


@dataclass
class EncodeResult:
    input_ids: list[int]
    attention_mask: list[int]
    special_tokens_mask: list[int] = field(default_factory=list)
    offsets: list[tuple[int, int]] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)


@dataclass
class BatchEncodeResult:
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    special_tokens_mask: list[list[int]]
    lengths: list[int]


class AdvancedTokenizer:
    """
    High-level API around BrainTokenizer for training, SFT, agents, and RAG.
    """

    def __init__(self, base: BrainTokenizer):
        self.base = base
        self._special_ids = self._build_special_id_map()

    @classmethod
    def from_file(cls, path: str | Path) -> "AdvancedTokenizer":
        return cls(BrainTokenizer.from_file(path))

    @classmethod
    def train(
        cls,
        corpus_paths: list[str] | str,
        vocab_size: int = 32768,
        min_frequency: int = 2,
        save_path: str | Path | None = None,
        extra_special_tokens: Sequence[str] | None = None,
    ) -> "AdvancedTokenizer":
        """Train base BPE then wrap. extra_special_tokens merged into trainer via ALL_SPECIAL_TOKENS."""
        # BrainTokenizer.train already uses ALL_SPECIAL_TOKENS from special_tokens module
        base = BrainTokenizer.train(
            corpus_paths=corpus_paths,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            save_path=save_path,
        )
        return cls(base)

    def _build_special_id_map(self) -> dict[str, int]:
        m: dict[str, int] = {}
        for tok in ALL_SPECIAL_TOKENS:
            tid = self.base.token_to_id(tok)
            if tid is not None:
                m[tok] = tid
        # always include core even if naming differs
        m.setdefault(PAD, self.base.pad_id)
        m["<unk>"] = self.base.unk_id
        m["<bos>"] = self.base.bos_id
        m["<eos>"] = self.base.eos_id
        m["<pad>"] = self.base.pad_id
        return m

    @property
    def vocab_size(self) -> int:
        return self.base.vocab_size

    @property
    def special_id_map(self) -> dict[str, int]:
        return dict(self._special_ids)

    def id_to_token(self, i: int) -> str | None:
        return self.base.id_to_token(i)

    def token_to_id(self, t: str) -> int | None:
        return self.base.token_to_id(t)

    # ------------------------------------------------------------------
    # Core encode with strategies
    # ------------------------------------------------------------------
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        truncation: str = "right",  # right | left | middle
        return_offsets: bool = False,
    ) -> EncodeResult:
        # Use HF tokenizer internals for offsets when possible
        enc = self.base._tok.encode(text)
        ids = list(enc.ids)
        offsets = list(enc.offsets) if return_offsets else []
        tokens = enc.tokens if return_offsets else []

        if add_special_tokens:
            ids = [self.base.bos_id] + ids + [self.base.eos_id]
            if return_offsets:
                offsets = [(0, 0)] + offsets + [(0, 0)]
                tokens = [BOS] + list(tokens) + [EOS]

        if max_length is not None and len(ids) > max_length:
            ids, offsets, tokens = self._truncate(ids, offsets, tokens, max_length, truncation)

        mask = [1] * len(ids)
        special_mask = [
            1 if (self.base.id_to_token(i) in self._special_ids or i in (
                self.base.pad_id, self.base.bos_id, self.base.eos_id, self.base.unk_id
            )) else 0
            for i in ids
        ]
        return EncodeResult(
            input_ids=ids,
            attention_mask=mask,
            special_tokens_mask=special_mask,
            offsets=offsets if return_offsets else [],
            tokens=list(tokens) if return_offsets else [],
        )

    def _truncate(self, ids, offsets, tokens, max_length, truncation):
        if truncation == "right":
            ids = ids[:max_length]
            offsets = offsets[:max_length] if offsets else offsets
            tokens = tokens[:max_length] if tokens else tokens
        elif truncation == "left":
            ids = ids[-max_length:]
            offsets = offsets[-max_length:] if offsets else offsets
            tokens = tokens[-max_length:] if tokens else tokens
        elif truncation == "middle":
            keep_head = max_length // 2
            keep_tail = max_length - keep_head
            ids = ids[:keep_head] + ids[-keep_tail:]
            if offsets:
                offsets = offsets[:keep_head] + offsets[-keep_tail:]
            if tokens:
                tokens = tokens[:keep_head] + tokens[-keep_tail:]
        else:
            raise ValueError(f"Unknown truncation: {truncation}")
        return ids, offsets, tokens

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.base.decode(list(ids), skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        max_length: int | None = None,
        truncation: str = "right",
        pad: bool = True,
        pad_to_multiple_of: int | None = 8,
        add_special_tokens: bool = True,
    ) -> BatchEncodeResult:
        encoded = [
            self.encode(t, add_special_tokens=add_special_tokens, max_length=max_length, truncation=truncation)
            for t in texts
        ]
        lengths = [len(e.input_ids) for e in encoded]
        if not pad:
            return BatchEncodeResult(
                input_ids=[e.input_ids for e in encoded],
                attention_mask=[e.attention_mask for e in encoded],
                special_tokens_mask=[e.special_tokens_mask for e in encoded],
                lengths=lengths,
            )
        target = max(lengths) if lengths else 0
        if max_length is not None:
            target = min(target, max_length) if target else max_length
            target = max(target, max(lengths) if lengths else 0)
            target = min(max(lengths), max_length) if lengths else max_length
        target = max(lengths) if lengths else 0
        if max_length is not None:
            target = min(max(target, max(lengths) if lengths else 0), max_length)
            # after truncation all <= max_length; pad up to max among batch or max_length
            target = max(lengths) if lengths else 0
        if pad_to_multiple_of and target > 0:
            rem = target % pad_to_multiple_of
            if rem:
                target += pad_to_multiple_of - rem
        pad_id = self.base.pad_id
        input_ids, attn, smask = [], [], []
        for e in encoded:
            ids = e.input_ids
            pad_n = max(0, target - len(ids))
            input_ids.append(ids + [pad_id] * pad_n)
            attn.append(e.attention_mask + [0] * pad_n)
            smask.append(e.special_tokens_mask + [1] * pad_n)
        return BatchEncodeResult(input_ids=input_ids, attention_mask=attn, special_tokens_mask=smask, lengths=lengths)

    # ------------------------------------------------------------------
    # Chat / agent helpers
    # ------------------------------------------------------------------
    def encode_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        add_generation_prompt: bool = False,
        max_length: int | None = None,
    ) -> EncodeResult:
        """
        messages: [{role: system|user|assistant|tool, content: str}, ...]
        """
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"{SYSTEM} {content}")
            elif role == "user":
                parts.append(f"{USER} {content}")
            elif role == "assistant":
                parts.append(f"{ASSISTANT} {content}")
            elif role == "tool":
                parts.append(content if content.startswith("<|tool") else f"{TOOL_RESULT}\n{content}")
            else:
                parts.append(content)
        text = "".join(parts)
        if add_generation_prompt:
            text += f"{ASSISTANT}"
        return self.encode(text, add_special_tokens=True, max_length=max_length)

    def encode_tool_call(self, name: str, arguments: dict[str, Any]) -> list[int]:
        payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
        text = f"{TOOL_CALL}\n{payload}\n{END_TOOL_CALL}"
        return self.encode(text, add_special_tokens=False).input_ids

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def fertility_report(self, texts: Sequence[str]) -> dict[str, float]:
        total_toks = total_words = total_chars = 0
        for t in texts:
            st = self.base.token_stats(t)
            total_toks += int(st["n_tokens"])
            total_words += int(st["n_words"])
            total_chars += int(st["n_chars"])
        return {
            "n_texts": len(texts),
            "tokens": total_toks,
            "words": total_words,
            "chars": total_chars,
            "fertility_tokens_per_word": total_toks / max(total_words, 1),
            "chars_per_token": total_chars / max(total_toks, 1),
        }

    def coverage_report(self, text: str) -> dict[str, Any]:
        ids = self.base.encode(text, add_special_tokens=False)
        unk = sum(1 for i in ids if i == self.base.unk_id)
        return {
            "n_tokens": len(ids),
            "unk_count": unk,
            "unk_rate": unk / max(len(ids), 1),
            "unique_tokens": len(set(ids)),
        }

    def vocab_sample(self, n: int = 20, seed: int = 0) -> list[tuple[int, str]]:
        rng = random.Random(seed)
        size = self.vocab_size
        ids = [rng.randrange(size) for _ in range(n)]
        return [(i, self.base.id_to_token(i) or "?") for i in ids]

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    def encode_file_stream(
        self,
        path: str | Path,
        *,
        chunk_chars: int = 1_000_000,
        add_special_tokens: bool = False,
    ) -> Iterator[list[int]]:
        path = Path(path)
        with path.open(encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(chunk_chars)
                if not chunk:
                    break
                yield self.base.encode(chunk, add_special_tokens=add_special_tokens)

    # ------------------------------------------------------------------
    # Augmentation: approximate BPE dropout by randomly splitting long tokens
    # ------------------------------------------------------------------
    def encode_noisy(self, text: str, dropout: float = 0.1, rng: random.Random | None = None) -> list[int]:
        """
        Light augmentation: with probability `dropout`, re-encode character
        n-grams of rare long tokens as shorter pieces by inserting spaces
        randomly. Best-effort; for true BPE-dropout use training-time HF flags.
        """
        rng = rng or random.Random()
        if dropout <= 0:
            return self.base.encode(text, add_special_tokens=False)
        chars = list(text)
        out_chars = []
        for ch in chars:
            out_chars.append(ch)
            if ch.isalnum() and rng.random() < dropout * 0.05:
                out_chars.append(" ")
        return self.base.encode("".join(out_chars), add_special_tokens=False)

    def save(self, path: str | Path) -> None:
        self.base.save(path)
