"""
brain/model/initialization.py

Weight initialization, applied once after the full model is constructed.

Uses the GPT-2 / nanoGPT scheme: all Linear/Embedding weights ~ N(0, 0.02),
EXCEPT the two projections that write directly back into the residual
stream (attention's o_proj, SwiGLU's down_proj), which get an extra
1/sqrt(2 * num_layers) scale-down. Without that scale-down, variance
in the residual stream compounds additively over depth (each of the
2*num_layers sublayers adds roughly full-scale noise into the same
stream), and a 12-layer model is deep enough for this to visibly
destabilize early training. This is a well-established fix (GPT-2,
nanoGPT), not a Brain-specific invention.

The Brain Trail prototype used plain default PyTorch initialization
(Linear's kaiming-uniform default) with no depth-aware scaling at all --
harmless at 2 layers, exactly the kind of thing that stops being harmless
at 12.
"""

from __future__ import annotations

import math

import torch.nn as nn

from brain.model.transformer.attention import CausalSelfAttention
from brain.model.transformer.mlp import SwiGLU


def init_weights(model: nn.Module, num_layers: int, base_std: float = 0.02) -> None:
    residual_std = base_std / math.sqrt(2 * num_layers)

    # Tag the two "writes back into the residual stream" projections so the
    # generic pass below can single them out, without hardcoding parameter
    # NAMES (which would silently break if a module gets renamed).
    residual_write_modules: set[int] = set()
    for module in model.modules():
        if isinstance(module, CausalSelfAttention):
            residual_write_modules.add(id(module.o_proj))
        elif isinstance(module, SwiGLU):
            residual_write_modules.add(id(module.down_proj))

    for module in model.modules():
        if isinstance(module, nn.Linear):
            std = residual_std if id(module) in residual_write_modules else base_std
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=base_std)
            if module.padding_idx is not None:
                with __import__("torch").no_grad():
                    module.weight[module.padding_idx].fill_(0.0)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    from brain.core.config import BrainConfig
    from brain.model.transformer.block import TransformerBlock

    cfg = BrainConfig(hidden_size=32, num_layers=4, num_heads=4, num_kv_heads=2, intermediate_size=64)
    blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
    init_weights(blocks, num_layers=cfg.num_layers)

    o_proj_std = blocks[0].attention.o_proj.weight.std().item()
    q_proj_std = blocks[0].attention.q_proj.weight.std().item()
    expected_residual_std = 0.02 / math.sqrt(2 * cfg.num_layers)

    print(f"q_proj (non-residual-write) weight std: {q_proj_std:.4f} (expected ~0.02)")
    print(f"o_proj (residual-write) weight std: {o_proj_std:.4f} (expected ~{expected_residual_std:.4f})")
    print(f"o_proj std correctly scaled down: {o_proj_std < q_proj_std}")
