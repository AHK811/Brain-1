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


KVCache = tuple[Tensor, Tensor]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, dropout: float) -> None:
        super().__init__()
        if n_kv_heads < 1 or n_heads % n_kv_heads or n_heads * head_dim != d_model:
            raise ValueError("invalid grouped-query attention dimensions")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.kv_dim = n_kv_heads * head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, d_model + 2 * self.kv_dim, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: KVCache | None = None) -> tuple[Tensor, KVCache]:
        b, t, d = x.shape
        q, k, v = self.qkv(x).split((d, self.kv_dim, self.kv_dim), dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        offset = 0
        if cache is not None:
            offset = cache[0].shape[2]
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        mask = None
        if offset and t > 1:
            query_positions = offset + torch.arange(t, device=x.device)
            mask = torch.arange(k.shape[2], device=x.device)[None, :] <= query_positions[:, None]
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=offset == 0,
            scale=self.scale,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.n_heads != self.n_kv_heads,
        )
        return self.proj(out.transpose(1, 2).reshape(b, t, d)), (k, v)


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SparseMoE(nn.Module):
    def __init__(self, d_model: int, hidden: int, n_experts: int, experts_per_token: int) -> None:
        super().__init__()
        self.experts_per_token = experts_per_token
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLUMLP(d_model, hidden) for _ in range(n_experts))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        flat = x.reshape(-1, x.shape[-1])
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = F.linear(flat.float(), self.router.weight.float())
            selected = scores.argsort(dim=-1, descending=True, stable=True)[:, :self.experts_per_token]
            weights = scores.gather(1, selected).softmax(dim=-1)
            probabilities = scores.softmax(dim=-1)
        out = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            tokens, slots = torch.where(selected == index)
            if tokens.numel():
                values = expert(flat[tokens])
                contribution = values * weights[tokens, slots, None].to(values.dtype)
                out.index_add_(0, tokens, contribution.to(out.dtype))
        if self.training:
            frequency = F.one_hot(selected, len(self.experts)).float().mean(dim=(0, 1))
            balance = len(self.experts) * (probabilities.mean(dim=0) * frequency).sum()
        else:
            balance = x.new_zeros((), dtype=torch.float32)
        return out.view_as(x), balance


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 hidden: int, dropout: float, rms_eps: float, n_experts: int, experts_per_token: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, rms_eps)
        self.attn = CausalSelfAttention(d_model, n_heads, n_kv_heads, head_dim, dropout)
        self.mlp_norm = RMSNorm(d_model, rms_eps)
        self.mlp = (SwiGLUMLP(d_model, hidden) if n_experts == 1 else
                    SparseMoE(d_model, hidden, n_experts, experts_per_token))

    def _forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: KVCache | None) -> tuple[Tensor, Tensor, KVCache]:
        attention, updated = self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + attention
        normalized = self.mlp_norm(x)
        if isinstance(self.mlp, SparseMoE):
            residual, balance = self.mlp(normalized)
        else:
            residual, balance = self.mlp(normalized), x.new_zeros((), dtype=torch.float32)
        return x + residual, balance, updated

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        x, balance, _ = self._forward(x, cos, sin, None)
        return x, balance

    def forward_cached(self, x: Tensor, cos: Tensor, sin: Tensor, cache: KVCache | None) -> tuple[Tensor, KVCache]:
        x, _, updated = self._forward(x, cos, sin, cache)
        return x, updated
