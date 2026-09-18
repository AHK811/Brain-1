"""
Audio encoder scaffolds for native audio tokens (optional path).

Default practical path remains ASR → text. This module supports future
audio-token training (mel → conv encoder → projector).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class AudioEncoderInfo:
    name: str
    dim: int
    n_tokens: int


class MelStubEncoder(nn.Module):
    """
    Lightweight learnable frontend: treats waveform/mel as (B, 1, T) or (B, n_mels, T)
    and pools to fixed token count. Useful for wiring tests without torchaudio.
    """

    def __init__(self, dim: int = 512, n_tokens: int = 32, in_ch: int = 1):
        super().__init__()
        self.info = AudioEncoderInfo(name="mel_stub", dim=dim, n_tokens=n_tokens)
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(64, dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.n_tokens = n_tokens
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, N, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv(x)  # (B, D, T')
        b, d, t = h.shape
        h = h.transpose(1, 2)  # (B, T', D)
        if t >= self.n_tokens:
            # adaptive pool via index
            idx = torch.linspace(0, t - 1, self.n_tokens, device=h.device).long()
            h = h[:, idx, :]
        else:
            pad = self.n_tokens - t
            h = torch.cat([h, torch.zeros(b, pad, d, device=h.device, dtype=h.dtype)], dim=1)
        return h


class AudioProjector(nn.Module):
    def __init__(self, audio_dim: int = 512, hidden_size: int = 384, depth: int = 2):
        super().__init__()
        if depth <= 1:
            self.net = nn.Linear(audio_dim, hidden_size)
        else:
            self.net = nn.Sequential(
                nn.Linear(audio_dim, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


class AudioTower(nn.Module):
    def __init__(self, hidden_size: int = 384, audio_dim: int = 512, n_tokens: int = 32):
        super().__init__()
        self.encoder = MelStubEncoder(dim=audio_dim, n_tokens=n_tokens)
        self.projector = AudioProjector(audio_dim=audio_dim, hidden_size=hidden_size)
        self.n_tokens = n_tokens

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # allow training encoder later by removing no_grad
            feats = self.encoder(audio)
        return self.projector(feats)
