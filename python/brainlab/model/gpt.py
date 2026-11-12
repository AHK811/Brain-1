from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from brainlab.model.layers import RMSNorm, TransformerBlock, rope_table
from brainlab.spec.config import ModelSpec


class TransformerLM(nn.Module):
    def __init__(self, spec: ModelSpec, gradient_checkpoint: bool = False) -> None:
        super().__init__()
        spec.validate()
        self.spec = spec
        self.gradient_checkpoint = gradient_checkpoint
        self.token_emb = nn.Embedding(spec.vocab_size, spec.d_model)
        self.drop = nn.Dropout(spec.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                spec.d_model,
                spec.n_heads,
                spec.head_dim,
                spec.mlp_hidden,
                spec.dropout,
                spec.rms_norm_eps,
            )
            for _ in range(spec.n_layers)
        )
        self.final_norm = RMSNorm(spec.d_model, spec.rms_norm_eps)
        self._rope: dict[tuple[int, str, torch.dtype], tuple[Tensor, Tensor]] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.spec.init_std
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
        residual_std = std / math.sqrt(2 * self.residual_scale_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=residual_std)

    @property
    def residual_scale_layers(self) -> int:
        return self.spec.residual_scale_layers

    def rope(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        key = (int(seq_len), str(device), dtype)
        cached = self._rope.get(key)
        if cached is None:
            cached = rope_table(seq_len, self.spec.head_dim, self.spec.rope_theta, device, dtype)
            self._rope[key] = cached
        return cached

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def lm_head(self, x: Tensor) -> Tensor:
        return F.linear(x, self.token_emb.weight)

    def hidden(self, ids: Tensor) -> Tensor:
        if ids.shape[-1] > self.spec.context_len:
            raise ValueError(f"sequence {ids.shape[-1]} exceeds context_len {self.spec.context_len}")
        x = self.drop(self.token_emb(ids))
        cos, sin = self.rope(ids.shape[-1], x.device, x.dtype)
        for block in self.blocks:
            if self.gradient_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        return self.final_norm(x)

    def forward(self, ids: Tensor, targets: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        h = self.hidden(ids)
        logits = self.lm_head(h.float())
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, self.spec.vocab_size), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def last_token_hidden(self, ids: Tensor) -> Tensor:
        return self.hidden(ids)[:, -1, :]
