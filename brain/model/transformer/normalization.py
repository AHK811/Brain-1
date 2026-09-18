"""RMSNorm (LLaMA-style) and LayerNorm (GPT-2-style)."""
from __future__ import annotations
import torch
import torch.nn as nn


class _ManualRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.eps)


def RMSNorm(hidden_size: int, eps: float = 1e-5) -> nn.Module:
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(hidden_size, eps=eps)
    return _ManualRMSNorm(hidden_size, eps=eps)


def build_norm(norm_type: str, hidden_size: int, eps: float = 1e-5) -> nn.Module:
    t = (norm_type or "rmsnorm").lower()
    if t in ("layernorm", "layer_norm", "ln", "gpt2"):
        return nn.LayerNorm(hidden_size, eps=eps)
    return RMSNorm(hidden_size, eps=eps)
