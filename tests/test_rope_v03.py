"""v0.3 RoPE behavior."""
import torch
from brain.model.transformer.rope import RotaryEmbedding, apply_rope


def test_rope_auto_extends():
    r = RotaryEmbedding(head_dim=8, max_position_embeddings=10, theta=10000.0)
    cos, sin = r.get(seq_len=5, position_offset=8)
    assert cos.shape == (5, 4)
    assert r.max_position_embeddings >= 13


def test_rope_ntk_constructs():
    r = RotaryEmbedding(
        head_dim=16, max_position_embeddings=32, theta=500000.0,
        scaling_type="ntk", scaling_factor=2.0,
    )
    q = torch.randn(1, 2, 8, 16)
    cos, sin = r.get(8)
    out = apply_rope(q, cos, sin)
    assert out.shape == q.shape
