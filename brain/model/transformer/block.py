"""Pre-norm transformer block: Attention + FFN with residual stream."""
from __future__ import annotations

import torch
import torch.nn as nn

from brain.core.config import BrainConfig
from brain.model.transformer.attention import (
    AlternatingSparseAttention,
    CausalSelfAttention,
    MultiHeadLatentAttention,
)
from brain.model.transformer.cache import KVCache
from brain.model.transformer.mlp import build_ffn
from brain.model.transformer.normalization import build_norm
from brain.model.transformer.rope import RotaryEmbedding


class TransformerBlock(nn.Module):
    """
    Residual stream:
      x = x + Attn(Norm(x))
      x = x + FFN(Norm(x))
    Pre-norm (GPT-2 later / LLaMA style): norm before sublayers.
    """

    def __init__(self, config: BrainConfig, layer_idx: int = 0):
        super().__init__()
        self.input_norm = build_norm(
            getattr(config, "norm_type", "rmsnorm"), config.hidden_size, config.norm_eps
        )
        attn_type = (getattr(config, "attention_type", None) or "gqa").lower()
        use_qk = bool(getattr(config, "use_qk_norm", False))
        window = getattr(config, "sliding_window", None)

        if attn_type == "mla":
            self.attention = MultiHeadLatentAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                latent_dim=getattr(config, "mla_latent_dim", None),
                bias=config.attention_bias,
                dropout=config.dropout,
                use_qk_norm=use_qk,
                norm_eps=config.norm_eps,
            )
        elif getattr(config, "sparse_alternating", False):
            self.attention = AlternatingSparseAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                bias=config.attention_bias,
                dropout=config.dropout,
                use_qk_norm=use_qk,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
                local_window=int(getattr(config, "sparse_local_window", 256) or 256),
            )
        else:
            self.attention = CausalSelfAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                bias=config.attention_bias,
                dropout=config.dropout,
                use_qk_norm=use_qk,
                sliding_window=window,
                norm_eps=config.norm_eps,
            )

        self.post_attention_norm = build_norm(
            getattr(config, "norm_type", "rmsnorm"), config.hidden_size, config.norm_eps
        )
        self.mlp = build_ffn(
            getattr(config, "ffn_type", "swiglu"),
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding | None = None,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.input_norm(x),
            rope,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            attention_mask=attention_mask,
        )
        x = x + self.mlp(self.post_attention_norm(x))
        return x
