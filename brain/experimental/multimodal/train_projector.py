"""
Minimal projector training step (image–caption contrastive or caption LM loss).

This is a sketch for Colab/notebooks — not a full trainer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def projector_caption_step(
    mm_model,
    *,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    One step: freeze encoder (already no-grad in VisionTower.encode),
    train projector + optional LM via causal LM loss on captions.
    """
    mm_model.train()
    # freeze LM optionally
    for p in mm_model.lm.parameters():
        p.requires_grad = False
    for p in mm_model.vision.projector.parameters():
        p.requires_grad = True

    optimizer.zero_grad(set_to_none=True)
    out = mm_model(input_ids, images=images, labels=labels)
    loss = out["loss"] if isinstance(out, dict) else out.loss
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def contrastive_step(
    image_feats: torch.Tensor,
    text_feats: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    image_feats: (B, D), text_feats: (B, D) — L2-normalized preferred.
    """
    image_feats = F.normalize(image_feats, dim=-1)
    text_feats = F.normalize(text_feats, dim=-1)
    logits = image_feats @ text_feats.T / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.T, targets)
    return (loss_i + loss_t) / 2
