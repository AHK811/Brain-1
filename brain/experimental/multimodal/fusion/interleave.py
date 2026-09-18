"""
Fuse vision tokens with text embeddings for a multimodal forward.

Two modes:
  1) String template (SFT data): <|image_start|> <|image_pad|>*N <|image_end|>
  2) Tensor fusion: replace pad positions with projected vision features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

IMAGE_START = "<|image_start|>"
IMAGE_END = "<|image_end|>"
IMAGE_PAD = "<|image_pad|>"


def build_interleaved_prompt(text: str, n_image_tokens: int = 64) -> str:
    pads = " ".join([IMAGE_PAD] * n_image_tokens)
    return f"{IMAGE_START} {pads} {IMAGE_END}\n{text}"


@dataclass
class FusionResult:
    inputs_embeds: torch.Tensor      # (B, T, H)
    attention_mask: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None


def inject_vision_into_embeds(
    text_embeds: torch.Tensor,
    vision_tokens: torch.Tensor,
    *,
    image_pad_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    text_embeds: (B, T, H)
    vision_tokens: (B, N, H)  — already projected
    image_pad_mask: (B, T) bool — True where IMAGE_PAD tokens sit

    If mask is None, prepend vision tokens to the sequence.
    """
    if image_pad_mask is None:
        return torch.cat([vision_tokens, text_embeds], dim=1)

    out = text_embeds.clone()
    b, t, h = out.shape
    n = vision_tokens.size(1)
    for i in range(b):
        positions = image_pad_mask[i].nonzero(as_tuple=False).flatten()
        if len(positions) == 0:
            continue
        use = min(len(positions), n)
        out[i, positions[:use]] = vision_tokens[i, :use]
    return out


def build_pad_mask_from_ids(
    input_ids: torch.Tensor,
    image_pad_id: int,
) -> torch.Tensor:
    return input_ids == image_pad_id
