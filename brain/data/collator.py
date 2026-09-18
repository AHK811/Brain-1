"""Batch collators for causal LM and preference pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class CausalCollator:
    pad_id: int = 0

    def __call__(self, sequences: Sequence[list[int]]) -> dict[str, torch.Tensor]:
        max_len = max(len(s) for s in sequences)
        b = len(sequences)
        input_ids = torch.full((b, max_len), self.pad_id, dtype=torch.long)
        attn = torch.zeros(b, max_len, dtype=torch.long)
        for i, s in enumerate(sequences):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1
        labels = input_ids.clone()
        labels[attn == 0] = -100  # common ignore index; model may use pad instead
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
