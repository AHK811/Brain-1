"""Token embeddings and optional learned absolute positional embeddings (GPT-2)."""
from __future__ import annotations
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, pad_id: int | None = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class LearnedPositionalEmbedding(nn.Module):
    """GPT-2 style absolute learned position embeddings."""

    def __init__(self, max_position_embeddings: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, seq_len: int, offset: int = 0, device=None) -> torch.Tensor:
        positions = torch.arange(offset, offset + seq_len, device=device)
        positions = positions.clamp(max=self.max_position_embeddings - 1)
        return self.embedding(positions)  # (T, H)
