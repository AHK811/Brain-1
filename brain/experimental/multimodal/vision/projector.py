"""Vision → Brain hidden projector (Phase B — trainable)."""

from __future__ import annotations

import torch
import torch.nn as nn


class VisionProjector(nn.Module):
    """
    Maps encoder tokens (B, N, vision_dim) → (B, N, hidden_size).

    depth=2 MLP is the LLaVA-style default; depth=1 is a single linear.
    """

    def __init__(
        self,
        vision_dim: int = 768,
        hidden_size: int = 384,
        depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.hidden_size = hidden_size
        layers: list[nn.Module] = []
        if depth <= 1:
            layers.append(nn.Linear(vision_dim, hidden_size))
        else:
            layers.extend([
                nn.Linear(vision_dim, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            ])
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, vision_feats: torch.Tensor) -> torch.Tensor:
        return self.net(vision_feats)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
