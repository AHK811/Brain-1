"""Image load + resize + normalize for vision encoders."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import torch


def load_image_tensor(
    path: str | Path,
    size: int = 224,
    mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073),
    std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711),
) -> torch.Tensor:
    """Return (1, 3, size, size) float tensor, CLIP-style norm when possible."""
    path = Path(path)
    try:
        from PIL import Image
        import torchvision.transforms as T
        img = Image.open(path).convert("RGB")
        tfm = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return tfm(img).unsqueeze(0)
    except Exception:
        # fallback: zeros
        return torch.zeros(1, 3, size, size)


def load_image_batch(paths: list[str | Path], size: int = 224) -> torch.Tensor:
    tensors = [load_image_tensor(p, size=size) for p in paths]
    return torch.cat(tensors, dim=0)
