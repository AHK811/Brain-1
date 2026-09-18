from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int = 32
    seq_len: int = 512
    lr: float = 3e-4
    warmup_steps: int = 500
    max_steps: int = 10000
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    bf16: bool = True
