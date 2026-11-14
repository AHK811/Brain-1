from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from brainlab.model.layers import KVCache, RMSNorm, SparseMoE, TransformerBlock, rope_table
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
                spec.n_kv_heads,
                spec.head_dim,
                spec.mlp_hidden,
                spec.dropout,
                spec.rms_norm_eps,
                spec.n_experts,
                spec.experts_per_token,
            )
            for _ in range(spec.n_layers)
        )
        self.final_norm = RMSNorm(spec.d_model, spec.rms_norm_eps)
        self.lm_head = nn.Linear(spec.d_model, spec.vocab_size, bias=False)
        if spec.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight
        self._rope: dict[tuple[str, torch.dtype], tuple[Tensor, Tensor]] = {}
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
            experts = block.mlp.experts if isinstance(block.mlp, SparseMoE) else (block.mlp,)
            for expert in experts:
                nn.init.normal_(expert.down.weight, mean=0.0, std=residual_std)

    @property
    def residual_scale_layers(self) -> int:
        return self.spec.residual_scale_layers

    def rope(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        key = (str(device), dtype)
        cached = self._rope.get(key)
        if cached is None:
            cached = rope_table(self.spec.context_len, self.spec.head_dim, self.spec.rope_theta, device, dtype)
            self._rope[key] = cached
        return cached[0][:, :, :seq_len], cached[1][:, :, :seq_len]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _hidden(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        if ids.ndim != 2 or not 0 < ids.shape[-1] <= self.spec.context_len:
            raise ValueError(f"expected nonempty B,T token ids with T <= context_len {self.spec.context_len}")
        x = self.drop(self.token_emb(ids))
        cos, sin = self.rope(ids.shape[-1], x.device, x.dtype)
        balance = x.new_zeros((), dtype=torch.float32)
        for block in self.blocks:
            if self.gradient_checkpoint and self.training:
                x, auxiliary = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x, auxiliary = block(x, cos, sin)
            balance = balance + auxiliary
        return self.final_norm(x), balance / self.spec.n_layers

    def hidden(self, ids: Tensor) -> Tensor:
        return self._hidden(ids)[0]

    def forward(self, ids: Tensor, targets: Tensor | None = None,
                loss_weights: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        h, balance = self._hidden(ids)
        logits = self.lm_head(h)
        if targets is None:
            return logits
        if targets.shape != ids.shape:
            raise ValueError("targets must have the same B,T shape as ids")
        losses = F.cross_entropy(logits.float().reshape(-1, self.spec.vocab_size),
                                 targets.reshape(-1), ignore_index=-1, reduction="none").view_as(targets)
        weights = targets.ne(-1).to(losses.dtype)
        if loss_weights is not None:
            if loss_weights.shape != targets.shape:
                raise ValueError("loss_weights must have the same B,T shape as targets")
            if not torch.isfinite(loss_weights).all() or (loss_weights < 0).any():
                raise ValueError("loss_weights must be finite and non-negative")
            weights = weights * loss_weights
        loss = (losses * weights).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
        if self.training:
            loss = loss + self.spec.router_aux_loss_coef * balance
        return logits, loss

    @torch.no_grad()
    def forward_cached(self, ids: Tensor, cache: list[KVCache] | None = None) -> tuple[Tensor, list[KVCache]]:
        if self.training:
            raise ValueError("KV caching requires model.eval()")
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("expected nonempty B,T token ids")
        offset = 0
        if cache is not None:
            if len(cache) != len(self.blocks):
                raise ValueError("cache must contain one K/V pair per layer")
            offset = cache[0][0].shape[2]
            expected = (ids.shape[0], self.spec.n_kv_heads, offset, self.spec.head_dim)
            for k, v in cache:
                if tuple(k.shape) != expected or tuple(v.shape) != expected:
                    raise ValueError("cache K/V shapes do not match model, batch or sequence length")
                if k.device != ids.device or v.device != ids.device or k.dtype != v.dtype:
                    raise ValueError("cache device or dtype mismatch")
        length = offset + ids.shape[1]
        if length > self.spec.context_len:
            raise ValueError(f"cached sequence exceeds context_len {self.spec.context_len}")
        x = self.token_emb(ids)
        cos, sin = self.rope(length, x.device, x.dtype)
        cos, sin = cos[:, :, offset:], sin[:, :, offset:]
        updated = []
        for index, block in enumerate(self.blocks):
            x, kv = block.forward_cached(x, cos, sin, None if cache is None else cache[index])
            updated.append(kv)
        return self.lm_head(self.final_norm(x)), updated

    @torch.no_grad()
    def last_token_hidden(self, ids: Tensor) -> Tensor:
        return self.hidden(ids)[:, -1, :]
