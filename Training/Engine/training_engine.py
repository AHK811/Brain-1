from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import time

@dataclass
class TrainConfig:
    steps: int = 1000
    log_every: int = 50
    lr: float = 3e-4

@dataclass
class TrainingEngine:
    model: Any = None
    optimizer: Any = None
    config: TrainConfig = field(default_factory=TrainConfig)
    step_fn: Optional[Callable] = None
    history: list = field(default_factory=list)

    def train(self) -> list[dict]:
        for step in range(1, self.config.steps + 1):
            t0 = time.perf_counter()
            if self.step_fn is not None:
                loss = float(self.step_fn())
            else:
                loss = 0.0
            self.history.append({"step": step, "loss": loss, "ms": (time.perf_counter() - t0) * 1000})
            if step % self.config.log_every == 0:
                print(f"step {step} loss {loss:.4f}")
        return self.history
