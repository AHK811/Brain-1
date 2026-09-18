"""Lightweight eval metrics."""

from __future__ import annotations

import math
from typing import Sequence


def perplexity_from_loss(loss: float) -> float:
    try:
        return math.exp(min(max(loss, 0.0), 20.0))
    except Exception:
        return float("inf")


def preference_accuracy(chosen_rewards: Sequence[float], rejected_rewards: Sequence[float]) -> float:
    if not chosen_rewards:
        return 0.0
    wins = sum(1 for c, r in zip(chosen_rewards, rejected_rewards) if c > r)
    return wins / len(chosen_rewards)
