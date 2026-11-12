from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def rmsnorm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    x32 = x.float()
    inv = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 * inv * weight.float()).to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        return rmsnorm(x, self.weight, self.eps)


def rope_table(seq_len: int, head_dim: int, theta: float, device: torch.device | str, dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    emb = torch.outer(pos, inv_freq)
    emb = torch.cat((emb, emb), dim=-1)
    return emb.cos().to(dtype=dtype).view(1, 1, seq_len, head_dim), emb.sin().to(dtype=dtype).view(1, 1, seq_len, head_dim)


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, t, d = x.shape
        packed = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = packed.unbind(0)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=self.scale,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(out.transpose(1, 2).reshape(b, t, d))


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, hidden: int, dropout: float, rms_eps: float) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, rms_eps)
        self.attn = CausalSelfAttention(d_model, n_heads, head_dim, dropout)
        self.mlp_norm = RMSNorm(d_model, rms_eps)
        self.mlp = SwiGLUMLP(d_model, hidden)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))
