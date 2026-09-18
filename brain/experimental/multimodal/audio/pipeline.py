"""Audio pipeline: file → ASR text and/or audio tokens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from brain.experimental.multimodal.audio.asr import ASRResult, transcribe
from brain.experimental.multimodal.audio.encoder import AudioTower


@dataclass
class AudioBundle:
    path: str
    asr: ASRResult
    tokens: Optional[torch.Tensor] = None  # (1, N, H) if tower used


def process_audio(
    path: str | Path,
    *,
    tower: Optional[AudioTower] = None,
    asr_backend: str = "auto",
    model_size: str = "base",
    waveform: Optional[torch.Tensor] = None,
) -> AudioBundle:
    path = Path(path)
    asr = transcribe(path, backend=asr_backend, model_size=model_size)
    tokens = None
    if tower is not None and waveform is not None:
        tokens = tower(waveform)
    return AudioBundle(path=str(path), asr=asr, tokens=tokens)
