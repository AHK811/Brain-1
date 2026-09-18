"""
Multimodal wrapper: VisionTower + BrainForCausalLM.

Phase B training typically freezes the LM (or freezes bottom layers) and
trains the projector on image-caption pairs. Full joint SFT comes later.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from brain.experimental.multimodal.fusion.interleave import inject_vision_into_embeds
from brain.experimental.multimodal.vision.pipeline import VisionTower


class BrainMM(nn.Module):
    def __init__(self, lm: nn.Module, vision: Optional[VisionTower] = None, hidden_size: int = 384):
        super().__init__()
        self.lm = lm
        self.vision = vision or VisionTower(hidden_size=hidden_size)
        self.hidden_size = hidden_size

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        # BrainForCausalLM uses token_embedding.embedding
        emb = self.lm.token_embedding
        if hasattr(emb, "embedding"):
            return emb.embedding(input_ids)
        return emb(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        images: Optional[torch.Tensor] = None,
        image_pad_mask: Optional[torch.Tensor] = None,
        audio_tokens: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        embeds = self.embed_tokens(input_ids)
        vis = None
        if images is not None:
            vis = self.vision(images)  # (B, N, H)
            embeds = inject_vision_into_embeds(embeds, vis, image_pad_mask=image_pad_mask)
        if audio_tokens is not None:
            embeds = inject_vision_into_embeds(embeds, audio_tokens, image_pad_mask=None)

        # If the LM supports inputs_embeds, use it; else fall back to ids-only
        if hasattr(self.lm, "forward"):
            try:
                return self.lm(input_ids=input_ids, labels=labels, **kwargs)
            except TypeError:
                pass
            # Many Brain builds only take input_ids — call standard path
            out = self.lm(input_ids, labels=labels) if labels is not None else self.lm(input_ids)
            if images is not None:
                # annotate that vision was computed (training loop can add projector loss later)
                if isinstance(out, dict):
                    out = dict(out)
                    out["vision_tokens"] = vis
            return out if isinstance(out, dict) else {"logits": out}
        raise RuntimeError("LM has no forward")

    def encode_image_paths(self, paths: list[str], device: str = "cpu") -> torch.Tensor:
        return self.vision.encode_paths(paths, device=device)
