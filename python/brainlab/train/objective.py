from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from brainlab.data.shards import TokenShards


@dataclass(frozen=True)
class ObjectiveConfig:
    thought_weight: float = 1.0
    thought_start: str = "<think>"
    thought_end: str = "</think>"

    def __post_init__(self) -> None:
        if not math.isfinite(self.thought_weight) or self.thought_weight <= 0:
            raise ValueError("thought_weight must be finite and positive")
        if not self.thought_start or not self.thought_end or self.thought_start == self.thought_end:
            raise ValueError("thought delimiters must be distinct non-empty control tokens")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ObjectiveConfig:
        unknown = set(config) - {"thought_weight", "thought_start", "thought_end"}
        if unknown:
            raise ValueError(f"unsupported objective settings: {sorted(unknown)}")
        return cls(**config)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def tokenizer_provenance(tokenizer: Any) -> dict[str, Any]:
    special_tokens = dict(getattr(tokenizer, "special_tokens", {}))
    payload = {
        "tokens": [token.hex() for token in tokenizer.token_bytes],
        "merges": list(tokenizer.merges),
        "eos_id": tokenizer.eos_id,
        "pad_id": tokenizer.pad_id,
        "max_word_bytes": tokenizer.max_word_bytes,
        "special_tokens": special_tokens,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {"sha256": digest, "vocab_size": tokenizer.vocab_size, "special_tokens": special_tokens}


def control_id(tokenizer: Any, text: str) -> int:
    if not hasattr(tokenizer, "special_id"):
        raise ValueError("this objective requires a tokenizer with explicit control tokens")
    token_id = tokenizer.special_id(text)
    if token_id is None:
        raise ValueError(f"tokenizer has no control token {text!r}")
    if list(tokenizer.encode(text, allow_special=True)) != [token_id]:
        raise ValueError(f"control token {text!r} is not atomic")
    return int(token_id)


class ThoughtSpans:
    def __init__(self, shards: TokenShards, start_id: int, end_id: int, eos_id: int) -> None:
        spans: list[tuple[int, int]] = []
        opened: int | None = None
        for base in range(0, shards.total_tokens, 1 << 20):
            tokens = shards.slice_tokens(base, min(base + (1 << 20), shards.total_tokens))
            controls = np.flatnonzero((tokens == start_id) | (tokens == end_id) | (tokens == eos_id))
            for local in controls:
                position = base + int(local)
                token = int(tokens[local])
                if token == start_id:
                    if opened is not None:
                        raise ValueError(f"nested thought control at stream token {position}")
                    opened = position + 1
                elif token == end_id:
                    if opened is None:
                        raise ValueError(f"thought end without stream context at token {position}")
                    spans.append((opened, position))
                    opened = None
                elif opened is not None:
                    raise ValueError(f"unclosed thought before document end at token {position}")
        if opened is not None:
            spans.append((opened, shards.total_tokens))
        self.starts = np.asarray([start for start, _ in spans], dtype=np.int64)
        self.ends = np.asarray([end for _, end in spans], dtype=np.int64)

    def weights(self, start: int, stop: int, thought_weight: float) -> np.ndarray:
        weights = np.ones(stop - start, dtype=np.float32)
        first = int(np.searchsorted(self.ends, start, side="right"))
        last = int(np.searchsorted(self.starts, stop, side="left"))
        for left, right in zip(self.starts[first:last], self.ends[first:last], strict=True):
            weights[max(int(left), start) - start : min(int(right), stop) - start] = thought_weight
        return weights


class ObjectiveShards:
    def __init__(self, shards: TokenShards, objective: ObjectiveConfig, tokenizer: Any = None) -> None:
        self.shards = shards
        self.seq_len = shards.seq_len
        self.objective = objective
        self.spans: ThoughtSpans | None = None
        self.tokenizer = tokenizer_provenance(tokenizer) if tokenizer is not None else None
        metadata = [shard.metadata for shard in shards.shards]
        formats = {entry.get("data_format", "pretraining") for entry in metadata}
        if len(formats) != 1:
            raise ValueError("cannot mix chat_sft and pretraining shards")
        self.data_format = formats.pop()
        if self.data_format not in {"chat_sft", "pretraining"}:
            raise ValueError(f"unsupported data format {self.data_format!r}")
        if shards.has_loss_weights:
            for entry in metadata:
                if entry.get("objective") != objective.as_dict():
                    raise ValueError("shard objective differs from requested weighting; repack explicitly")
                if self.tokenizer is not None and entry.get("tokenizer") != self.tokenizer:
                    raise ValueError("shard tokenizer differs from requested tokenizer")
            self.indices = np.asarray([
                index for index in range(len(shards)) if np.any(shards.loss_weights(index) > 0)
            ], dtype=np.int64)
        else:
            self.indices = range(len(shards))
            if tokenizer is not None and objective.thought_weight != 1.0:
                start_id = control_id(tokenizer, objective.thought_start)
                end_id = control_id(tokenizer, objective.thought_end)
                self.spans = ThoughtSpans(shards, start_id, end_id, tokenizer.eos_id)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            **self.shards.provenance,
            "data_format": self.data_format,
            "loss_weights": "sidecar" if self.shards.has_loss_weights else "control_spans" if self.spans is not None else "uniform",
            "labelled_sequences": len(self),
            "tokenizer": self.tokenizer,
            "metadata": [shard.metadata for shard in self.shards.shards],
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, ...]:
        original = int(self.indices[index])
        x, y = self.shards[original]
        weights = self.shards.loss_weights(original)
        if weights is None and self.spans is not None:
            start = (original + self.shards._offset) * self.seq_len + 1
            weights = self.spans.weights(start, start + self.seq_len, self.objective.thought_weight)
        if weights is None:
            return x, y
        y = y.copy()
        y[weights == 0] = -1
        return x, y, weights


class SequenceWindow:
    def __init__(self, shards: ObjectiveShards, offset: int) -> None:
        self.shards = shards
        self.offset = offset

    def __len__(self) -> int:
        return max(len(self.shards) - self.offset, 0)

    def __getitem__(self, index: int) -> tuple[np.ndarray, ...]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.shards[index + self.offset]


def validate_resume_config(saved: Mapping[str, Any], requested: Mapping[str, Any]) -> None:
    required = {"objective", "schedule", "peak_lrs", "data_provenance", "tokenizer"}
    for key, value in requested.items():
        if key in required and key not in saved:
            raise SystemExit(f"resume checkpoint lacks {key} provenance; use --init-from for a new run")
        if key in saved and saved[key] != value:
            raise SystemExit(f"resume {key}={saved[key]} differs from requested {value}; keep the original run settings")
