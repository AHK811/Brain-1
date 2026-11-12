from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LRSchedule:
    peak_lr: float
    warmup_steps: int
    total_steps: int
    min_lr_fraction: float = 0.1
    decay: str = "cosine"

    def __post_init__(self) -> None:
        if self.peak_lr <= 0:
            raise ValueError("peak_lr must be positive")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must be shorter than total_steps")
        if not 0 <= self.min_lr_fraction < 1:
            raise ValueError("min_lr_fraction must be in [0, 1)")
        if self.decay not in {"cosine", "linear", "constant"}:
            raise ValueError(f"unknown decay {self.decay}")

    def lr_at(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be non-negative")
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps
        remaining = self.total_steps - self.warmup_steps
        progress = min(max((step - self.warmup_steps) / remaining, 0.0), 1.0)
        floor = self.peak_lr * self.min_lr_fraction
        if self.decay == "constant":
            return self.peak_lr
        if self.decay == "linear":
            return floor + (self.peak_lr - floor) * (1.0 - progress)
        return floor + 0.5 * (self.peak_lr - floor) * (1.0 + math.cos(math.pi * progress))

    def factor_at(self, step: int) -> float:
        return self.lr_at(step) / self.peak_lr

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(cls, schedule: Mapping[str, Any], peak_lr: float) -> LRSchedule:
        return cls(
            peak_lr=peak_lr,
            warmup_steps=int(schedule["warmup_steps"]),
            total_steps=int(schedule["total_steps"]),
            min_lr_fraction=float(schedule.get("min_lr_fraction", 0.1)),
            decay=str(schedule.get("decay", "cosine")),
        )


def tokens_per_step(seq_len: int, micro_batch: int, accum_steps: int, world_size: int = 1) -> int:
    return seq_len * micro_batch * accum_steps * world_size


def steps_for_tokens(total_tokens: int, seq_len: int, micro_batch: int, accum_steps: int, world_size: int = 1) -> int:
    per_step = tokens_per_step(seq_len, micro_batch, accum_steps, world_size)
    return max(1, -(-total_tokens // per_step))
