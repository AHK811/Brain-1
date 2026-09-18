"""
FFN blocks: SwiGLU (LLaMA-style) and classic two-layer GELU MLP (GPT-2-style).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GELUMLP(nn.Module):
    """GPT-2 style FFN: Linear -> GELU -> Linear (two-layer MLP)."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = True):
        super().__init__()
        self.fc_in = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc_out = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.gelu(self.fc_in(x)))


def build_ffn(ffn_type: str, hidden_size: int, intermediate_size: int, bias: bool) -> nn.Module:
    t = (ffn_type or "swiglu").lower()
    if t in ("gelu", "gelu_mlp", "mlp", "gpt2"):
        return GELUMLP(hidden_size, intermediate_size, bias=bias)
    return SwiGLU(hidden_size, intermediate_size, bias=bias)
