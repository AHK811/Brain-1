"""
Open-source vision encoders for Brain v0.4 Phase B.

Backends (optional installs):
  - transformers CLIP / SigLIP  (pip install transformers)
  - open_clip                   (pip install open_clip_torch)
  - torchvision ResNet stub     (always if torchvision present)
  - random stub                 (always — for wiring tests)

Encoders are frozen by default; only the projector is trained at first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Union

import torch
import torch.nn as nn


@dataclass
class VisionEncoderInfo:
    name: str
    dim: int
    grid: int  # e.g. 16 → 16x16 patches if applicable
    input_size: int = 224


class VisionEncoder(Protocol):
    info: VisionEncoderInfo
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) float 0..1 or normalized → (B, N, D) or (B, D)."""
        ...


class StubVisionEncoder(nn.Module):
    """Deterministic-ish stub for pipeline tests without heavy deps."""
    def __init__(self, dim: int = 768, n_tokens: int = 64):
        super().__init__()
        self.info = VisionEncoderInfo(name="stub", dim=dim, grid=int(n_tokens ** 0.5) or 8)
        self.n_tokens = n_tokens
        self.dim = dim
        self.proj = nn.Conv2d(3, dim, kernel_size=16, stride=16)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) → (B, N, D)
        if images.dim() != 4:
            raise ValueError("expected (B,3,H,W)")
        x = self.proj(images)  # (B, D, h, w)
        b, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, h*w, D)
        if x.size(1) > self.n_tokens:
            x = x[:, : self.n_tokens]
        elif x.size(1) < self.n_tokens:
            pad = self.n_tokens - x.size(1)
            x = torch.cat([x, torch.zeros(b, pad, d, device=x.device, dtype=x.dtype)], dim=1)
        return x


class CLIPVisionEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        from transformers import CLIPVisionModel, CLIPImageProcessor
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        hidden = self.model.config.hidden_size
        self.info = VisionEncoderInfo(name=f"clip:{model_name}", dim=hidden, grid=7, input_size=224)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        # expects already preprocessed (B,3,H,W) in CLIP norm space ideally
        out = self.model(pixel_values=images)
        # last_hidden_state: (B, 1+N, D) — drop CLS for patch tokens optional
        return out.last_hidden_state[:, 1:, :]  # (B, N, D)


class OpenCLIPVisionEncoder(nn.Module):
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        super().__init__()
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.visual
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.preprocess = preprocess
        # ViT-B-32 → 512 or 768 depending on variant
        dim = getattr(self.model, "output_dim", None) or 512
        self.info = VisionEncoderInfo(name=f"open_clip:{model_name}", dim=int(dim), grid=7, input_size=224)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model(images)
        if feats.dim() == 2:
            feats = feats.unsqueeze(1)  # (B, 1, D)
        return feats


def get_vision_encoder(
    prefer: Optional[str] = None,
    *,
    dim: int = 768,
    n_tokens: int = 64,
) -> VisionEncoder:
    order = []
    if prefer:
        order.append(prefer)
    order.extend(["clip", "open_clip", "stub"])
    for name in order:
        try:
            if name in ("clip", "transformers"):
                return CLIPVisionEncoder()
            if name == "open_clip":
                return OpenCLIPVisionEncoder()
            if name == "stub":
                return StubVisionEncoder(dim=dim, n_tokens=n_tokens)
        except Exception:
            continue
    return StubVisionEncoder(dim=dim, n_tokens=n_tokens)
