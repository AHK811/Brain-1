"""
brain/inference/speed.py

Speed-related helpers for Brain v0.3.

PyTorch SDPA already dispatches to FlashAttention / memory-efficient
kernels when the GPU and dtype support them. This module documents
that path and provides small utilities to force or inspect backends.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def sdpa_backends_available() -> dict[str, bool]:
    """Report which SDPA backends the current torch build can use."""
    info = {
        "math": True,
        "flash": False,
        "mem_efficient": False,
    }
    if not torch.cuda.is_available():
        return info
    try:
        if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
            info["flash"] = bool(torch.backends.cuda.flash_sdp_enabled())
        if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled"):
            info["mem_efficient"] = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
    except Exception:
        pass
    return info


@contextmanager
def prefer_flash_attention(enabled: bool = True) -> Iterator[None]:
    """
    Prefer FlashAttention / mem-efficient SDPA when available.

    Usage:
        with prefer_flash_attention():
            out = model(...)
    """
    if not torch.cuda.is_available() or not enabled:
        yield
        return
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True,
        ):
            yield
    except Exception:
        yield


def recommend_amp_dtype() -> torch.dtype:
    """bf16 on Ampere+, else fp16."""
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16
