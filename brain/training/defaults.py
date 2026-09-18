"""
brain/training/defaults.py

Recommended hyperparameters for stable pretrain + SFT on Brain v0.3 (~33M).

These are defaults, not hard constraints — override per GPU/data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PretrainDefaults:
    lr: float = 3e-4
    min_lr: float = 2e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 500
    grad_clip: float = 1.0
    dropout: float = 0.0
    # sequence
    context_len: int = 1024  # raise to 2048/4096 when VRAM allows
    # batching (effective batch ≈ batch * accum * context tokens)
    batch_size: int = 16
    grad_accum: int = 8
    # ckpt
    ckpt_every: int = 500
    gen_every: int = 1000


@dataclass
class SFTDefaults:
    lr: float = 1e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 100
    grad_clip: float = 1.0
    dropout: float = 0.0  # can set 0.05–0.1 on small repeated SFT sets
    context_len: int = 2048
    batch_size: int = 8
    grad_accum: int = 4
    # Prefer packing multiple short instructions per sequence when possible
    pack_sequences: bool = True


def apply_pretrain_defaults(cfg: Any, defaults: PretrainDefaults | None = None) -> Any:
    """Set cfg.dropout from pretrain defaults if present."""
    d = defaults or PretrainDefaults()
    if hasattr(cfg, "dropout"):
        cfg.dropout = d.dropout
    return cfg


def adamw_kwargs(defaults: PretrainDefaults | SFTDefaults) -> dict[str, Any]:
    return {
        "lr": defaults.lr,
        "weight_decay": defaults.weight_decay,
        "betas": defaults.betas,
    }
