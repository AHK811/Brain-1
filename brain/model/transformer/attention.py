"""
Causal self-attention: GQA / MHA / MLA, QK-RMSNorm, RoPE, KV-cache, SDPA.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.model.transformer.cache import KVCache
from brain.model.transformer.rope import RotaryEmbedding, apply_rope


def _sdpa_supports_enable_gqa() -> bool:
    try:
        q = torch.zeros(1, 2, 1, 4)
        k = torch.zeros(1, 1, 1, 4)
        v = torch.zeros(1, 1, 1, 4)
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return True
    except TypeError:
        return False


_SUPPORTS_NATIVE_GQA = _sdpa_supports_enable_gqa()


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class QKNorm(nn.Module):
    """Per-head RMSNorm applied to Q and K before RoPE (QK-RMSNorm)."""

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.q_weight = nn.Parameter(torch.ones(head_dim))
        self.k_weight = nn.Parameter(torch.ones(head_dim))

    def _rms(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, D)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * weight

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        return self._rms(q, self.q_weight), self._rms(k, self.k_weight)


class CausalSelfAttention(nn.Module):
    """RoPE + optional QK-RMSNorm + GQA/MHA + KV-cache + SDPA."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        bias: bool = False,
        dropout: float = 0.0,
        use_qk_norm: bool = False,
        sliding_window: int | None = None,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)
        self.qk_norm = QKNorm(self.head_dim, eps=norm_eps) if use_qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        position_offset = kv_cache.get_seq_length(layer_idx) if kv_cache is not None else 0

        if attention_mask is not None and kv_cache is not None:
            raise NotImplementedError(
                "attention_mask is only supported for the full-sequence (no-cache) "
                "forward pass. Batched variable-length cached decoding needs "
                "per-row position offsets in KVCache, which doesn't exist yet -- "
                "generate() one sequence at a time for now."
            )
        if self.sliding_window is not None and kv_cache is not None:
            raise NotImplementedError(
                "sliding_window isn't wired up for cached decode yet -- the "
                "previous version of this file silently IGNORED it whenever a "
                "kv_cache was passed (attending over the full history instead of "
                "the window), which is worse than erroring. Real windowed caching "
                "(trimming old KV entries so the window's memory benefit is "
                "actually realized) is a follow-up, not done here."
            )

        def split_heads(t: torch.Tensor, n_heads: int) -> torch.Tensor:
            return t.view(batch, seq_len, n_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(x), self.num_heads)
        k = split_heads(self.k_proj(x), self.num_kv_heads)
        v = split_heads(self.v_proj(x), self.num_kv_heads)

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        if rope is not None:
            cos, sin = rope.get(seq_len=seq_len, position_offset=position_offset)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if kv_cache is not None and layer_idx is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # is_causal=True is only correct when the query and key sequences
        # START AT THE SAME POSITION (full-sequence pass, or the very first
        # cached step) -- PyTorch's is_causal builds tril(Lq, Lk), so for a
        # single new query token (Lq=1) against many cached keys (Lk=many)
        # it silently collapses to "only attend to key position 0". This was
        # a real, confirmed bug in this file: every decode step past the
        # first was wrong for every non-sliding-window config. Cached decode
        # past prefill needs NO masking at all (every cached key already sits
        # at or before the new token's position), so is_causal=False +
        # attn_mask=None is the correct, not merely permissive, choice here.
        is_causal = kv_cache is None or position_offset == 0
        if kv_cache is not None and position_offset > 0:
            is_causal = False

        combined_mask = None  # (1 or batch, seq, seq) additive float mask, built up below

        if self.sliding_window is not None and kv_cache is None and seq_len > 1:
            w = self.sliding_window
            idx = torch.arange(seq_len, device=x.device)
            diff = idx[:, None] - idx[None, :]
            band = (diff >= 0) & (diff < w)  # causal + windowed; always includes the diagonal
            window_mask = torch.zeros(seq_len, seq_len, device=x.device, dtype=q.dtype)
            window_mask = window_mask.masked_fill(~band, torch.finfo(q.dtype).min)
            combined_mask = window_mask[None, :, :]
            is_causal = False

        if attention_mask is not None:
            # Combine causal + right-padding into one additive mask. Every
            # query position may always attend to itself (the `| diag`)
            # even if it's a padding position, so an all-padding causal
            # prefix never produces an all-blocked row (which would NaN
            # under softmax) -- padding-position loss is already excluded
            # via labels=-100 elsewhere, so this is purely about not
            # propagating NaN into real tokens' rows in earlier layers.
            causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            key_ok = attention_mask.to(dtype=torch.bool, device=x.device)
            allow = causal[None, :, :] & key_ok[:, None, :]
            diag = torch.eye(seq_len, dtype=torch.bool, device=x.device)[None, :, :]
            allow = allow | diag
            pad_mask = torch.zeros(batch, seq_len, seq_len, dtype=q.dtype, device=x.device)
            pad_mask = pad_mask.masked_fill(~allow, torch.finfo(q.dtype).min)
            combined_mask = pad_mask if combined_mask is None else (combined_mask + pad_mask)
            is_causal = False

        attn_mask = combined_mask.unsqueeze(1) if combined_mask is not None else None  # (b|1, 1, seq, seq)

        if _SUPPORTS_NATIVE_GQA and self.n_rep > 1 and attn_mask is None:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal, enable_gqa=True,
            )
        else:
            k_rep = _repeat_kv(k, self.n_rep)
            v_rep = _repeat_kv(v, self.n_rep)
            y = F.scaled_dot_product_attention(
                q, k_rep, v_rep, attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal if attn_mask is None else False,
            )

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(y)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA-style): project KV into a low-rank latent,
    then up-project per head.

    HONEST STATUS (fixed during review, not silently patched over): this
    simplified version does NOT yet shrink the KV cache. It up-projects K/V
    to full per-head tensors BEFORE caching them, so what's actually stored
    is the same (batch, num_heads, seq, head_dim) shape a plain MHA cache
    would use -- i.e. the one thing MLA exists to deliver isn't realized
    here yet. A real fix needs DeepSeek-style *decoupled* RoPE (a small
    position-carrying slice of K that's cached separately from the
    position-free compressed latent), which is enough additional design and
    verification work that it deserves its own dedicated, carefully-tested
    pass rather than being rushed in alongside other fixes. Until then,
    treat this class as "compresses the K/V *projections'* parameter count,
    not the KV-cache memory" -- still architecturally interesting, just not
    the memory win the name implies.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        latent_dim: int | None = None,
        bias: bool = False,
        dropout: float = 0.0,
        use_qk_norm: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.latent_dim = latent_dim or max(self.head_dim, hidden_size // 8)
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        # down-project to latent for K/V
        self.kv_down = nn.Linear(hidden_size, self.latent_dim * 2, bias=bias)
        # up-project latent -> per-head K and V
        self.k_up = nn.Linear(self.latent_dim, num_heads * self.head_dim, bias=bias)
        self.v_up = nn.Linear(self.latent_dim, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)
        self.qk_norm = QKNorm(self.head_dim, eps=norm_eps) if use_qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise NotImplementedError(
                "attention_mask isn't implemented for MultiHeadLatentAttention yet."
            )
        batch, seq_len, _ = x.shape
        position_offset = 0
        if kv_cache is not None and layer_idx is not None:
            position_offset = kv_cache.get_seq_length(layer_idx)

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_down(x)
        k_lat, v_lat = kv.split(self.latent_dim, dim=-1)

        # See the class docstring: this materializes full per-head K/V BEFORE
        # caching, so the cache below is the same size a plain MHA cache
        # would be -- MLA's actual memory benefit isn't realized yet.
        k = self.k_up(k_lat).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(v_lat).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        if rope is not None:
            cos, sin = rope.get(seq_len=seq_len, position_offset=position_offset)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if kv_cache is not None and layer_idx is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # Same bug class as CausalSelfAttention had: is_causal=True is only
        # correct when q and k start at the same position. Fixed the same way.
        is_causal = kv_cache is None or position_offset == 0
        if kv_cache is not None and position_offset > 0:
            is_causal = False

        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(y)


class SpecializedHeadBank(nn.Module):
    """
    Lightweight specialized head groups: memory / syntactic / long-range / constant.
    Splits query heads into role groups (equal split); same K/V shared (GQA-style).
    """

    ROLES = ("memory", "syntactic", "long_range", "constant")

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        base = num_heads // len(self.ROLES)
        rem = num_heads % len(self.ROLES)
        self.role_slices: dict[str, slice] = {}
        start = 0
        for i, role in enumerate(self.ROLES):
            n = base + (1 if i < rem else 0)
            self.role_slices[role] = slice(start, start + n)
            start += n
        # learnable role gates
        self.role_gate = nn.Parameter(torch.ones(num_heads))

    def gate(self, attn_out_heads: torch.Tensor) -> torch.Tensor:
        # attn_out_heads: (B, H, T, D)
        return attn_out_heads * self.role_gate.view(1, -1, 1, 1)


class AlternatingSparseAttention(CausalSelfAttention):
    """Even layers: local sliding window. Odd layers: full causal (global)."""

    def __init__(self, *args, layer_idx: int = 0, local_window: int = 256, **kwargs):
        if layer_idx % 2 == 0:
            kwargs["sliding_window"] = local_window
        else:
            kwargs["sliding_window"] = None
        super().__init__(*args, **kwargs)
        self.layer_idx_fixed = layer_idx
