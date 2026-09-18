from __future__ import annotations
import math
def from_loss(loss: float) -> float:
    return math.exp(min(max(float(loss), 0.0), 20.0))
