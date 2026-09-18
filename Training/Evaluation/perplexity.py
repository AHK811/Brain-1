from __future__ import annotations
import math

def perplexity_from_loss(loss: float) -> float:
    try:
        return math.exp(min(max(float(loss), 0.0), 20.0))
    except Exception:
        return float("inf")
