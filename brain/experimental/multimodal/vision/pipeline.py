"""
End-to-end: image path(s) → encoder → projector → LM-space tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from brain.experimental.multimodal.vision.encoder import VisionEncoder, get_vision_encoder
from brain.experimental.multimodal.vision.preprocess import load_image_batch
from brain.experimental.multimodal.vision.projector import VisionProjector


@dataclass
class VisionTower(nn.Module):
    """Frozen encoder + trainable projector."""

    def __init__(
        self,
        hidden_size: int = 384,
        vision_dim: int = 768,
        n_tokens: int = 64,
        projector_depth: int = 2,
        encoder: Optional[VisionEncoder] = None,
        prefer_encoder: Optional[str] = None,
    ):
        super().__init__()
        self.encoder = encoder or get_vision_encoder(
            prefer_encoder, dim=vision_dim, n_tokens=n_tokens
        )
        # align projector input dim to encoder
        dim = getattr(self.encoder, "info", None)
        vdim = dim.dim if dim is not None else vision_dim
        self.projector = VisionProjector(vision_dim=vdim, hidden_size=hidden_size, depth=projector_depth)
        self.n_tokens = n_tokens
        self.hidden_size = hidden_size

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_images(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) → (B, N, hidden_size) in LM space."""
        with torch.no_grad():
            feats = self.encoder.encode_images(images)
        # if encoder returns (B, D), expand tokens
        if feats.dim() == 2:
            feats = feats.unsqueeze(1).expand(-1, self.n_tokens, -1).contiguous()
        return self.projector(feats)

    def encode_paths(self, paths: list[str | Path], device: str | torch.device = "cpu") -> torch.Tensor:
        size = 224
        info = getattr(self.encoder, "info", None)
        if info is not None:
            size = info.input_size
        batch = load_image_batch(paths, size=size).to(device)
        self.to(device)
        return self.forward(batch)
