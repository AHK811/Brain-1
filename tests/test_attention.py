"""
tests/test_attention.py

Covers causal behavior, RoPE, SDPA, and GQA. Also contains the MANDATORY
test from the implementation prompt's Step 10: cached vs. uncached
generation must match within numerical tolerance, or implementation must
stop. This is what audit Problem #1 (KV-cache TypeError in the original
prototype) is verified fixed against, at both the attention-module and
block level.
"""

import pytest
import torch

from brain.model.transformer.attention import CausalSelfAttention, _repeat_kv
from brain.model.transformer.block import TransformerBlock
from brain.model.transformer.cache import KVCache
from brain.model.transformer.rope import RotaryEmbedding, apply_rope
from brain.core.config import BrainConfig


@pytest.fixture
def rope():
    return RotaryEmbedding(head_dim=8, max_position_embeddings=64, theta=10000.0)


# --------------------------------------------------------------------------
# RoPE
# --------------------------------------------------------------------------
def test_rope_preserves_vector_norm(rope):
    torch.manual_seed(0)
    q = torch.randn(1, 2, 8, 8)
    cos, sin = rope.get(seq_len=8)
    q_rot = apply_rope(q, cos, sin)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4)


def test_rope_stepped_matches_full_sequence(rope):
    """The property cached decoding depends on: rotating token i alone at
    position_offset=i must equal rotating the whole sequence at once."""
    torch.manual_seed(0)
    q = torch.randn(1, 2, 6, 8)
    full_cos, full_sin = rope.get(seq_len=6, position_offset=0)
    full_rot = apply_rope(q, full_cos, full_sin)

    stepped = torch.empty_like(q)
    for i in range(6):
        c, s = rope.get(seq_len=1, position_offset=i)
        stepped[:, :, i:i+1, :] = apply_rope(q[:, :, i:i+1, :], c, s)

    assert torch.allclose(full_rot, stepped, atol=1e-6)


def test_rope_auto_extends_beyond_initial_max():
    """v0.3: RoPE tables grow on demand instead of raising (graceful long context)."""
    r = RotaryEmbedding(head_dim=8, max_position_embeddings=10)
    cos, sin = r.get(seq_len=5, position_offset=8)  # 8+5=13 > 10
    assert cos.shape[0] == 5 and sin.shape[0] == 5
    assert r.max_position_embeddings >= 13


# --------------------------------------------------------------------------
# GQA repeat_kv
# --------------------------------------------------------------------------
def test_repeat_kv_shape_and_values():
    x = torch.arange(2 * 1 * 3 * 4).reshape(2, 1, 3, 4).float()
    out = _repeat_kv(x, n_rep=3)
    assert out.shape == (2, 3, 3, 4)
    assert torch.equal(out[:, 0], out[:, 1])  # repeated heads are identical copies


def test_repeat_kv_noop_when_n_rep_1():
    x = torch.randn(2, 4, 3, 8)
    assert torch.equal(_repeat_kv(x, n_rep=1), x)


# --------------------------------------------------------------------------
# THE MANDATORY TEST: cached vs uncached must match, at both levels
# --------------------------------------------------------------------------
@pytest.mark.parametrize("num_heads,num_kv_heads,hidden", [(4, 4, 32), (4, 2, 32), (6, 2, 36)])
def test_attention_cached_matches_uncached(num_heads, num_kv_heads, hidden):
    torch.manual_seed(0)
    attn = CausalSelfAttention(hidden, num_heads, num_kv_heads)
    attn.eval()
    rope = RotaryEmbedding(hidden // num_heads, max_position_embeddings=32)

    x = torch.randn(1, 7, hidden)
    with torch.no_grad():
        out_full = attn(x, rope)

        cache = KVCache(num_layers=1)
        outs = [attn(x[:, i:i+1, :], rope, kv_cache=cache, layer_idx=0) for i in range(7)]
        out_cached = torch.cat(outs, dim=1)

    assert torch.allclose(out_full, out_cached, atol=1e-5), (
        f"KV cache mismatch for num_heads={num_heads}, num_kv_heads={num_kv_heads}: "
        f"max diff {(out_full - out_cached).abs().max().item():.2e}"
    )


def test_attention_prefill_then_decode_matches_full_sequence():
    """Realistic usage pattern: prefill a multi-token prompt, then decode
    one token at a time -- not just single-token steps from the start."""
    torch.manual_seed(0)
    hidden, heads, kv_heads = 32, 4, 2
    attn = CausalSelfAttention(hidden, heads, kv_heads)
    attn.eval()
    rope = RotaryEmbedding(hidden // heads, max_position_embeddings=32)

    x = torch.randn(1, 8, hidden)
    with torch.no_grad():
        out_full = attn(x, rope)

        cache = KVCache(num_layers=1)
        prefill_out = attn(x[:, :5, :], rope, kv_cache=cache, layer_idx=0)
        step_outs = [attn(x[:, i:i+1, :], rope, kv_cache=cache, layer_idx=0) for i in range(5, 8)]
        out_mixed = torch.cat([prefill_out] + step_outs, dim=1)

    assert torch.allclose(out_full, out_mixed, atol=1e-5)


def test_block_level_cached_matches_uncached():
    """Same equivalence test one level up, through a full pre-norm block
    (attention + SwiGLU + both residuals), not attention in isolation."""
    torch.manual_seed(0)
    cfg = BrainConfig(hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=2,
                       intermediate_size=64, max_position_embeddings=32)
    block = TransformerBlock(cfg)
    block.eval()
    rope = RotaryEmbedding(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)

    x = torch.randn(1, 6, cfg.hidden_size)
    with torch.no_grad():
        out_full = block(x, rope)
        cache = KVCache(num_layers=1)
        outs = [block(x[:, i:i+1, :], rope, kv_cache=cache, layer_idx=0) for i in range(6)]
        out_cached = torch.cat(outs, dim=1)

    assert torch.allclose(out_full, out_cached, atol=1e-5)


def test_causal_masking_blocks_future_tokens():
    """A token's output must not change if we alter tokens AFTER it in the
    sequence -- the defining property of causal attention."""
    torch.manual_seed(0)
    hidden, heads, kv_heads = 32, 4, 2
    attn = CausalSelfAttention(hidden, heads, kv_heads)
    attn.eval()
    rope = RotaryEmbedding(hidden // heads, max_position_embeddings=32)

    x = torch.randn(1, 6, hidden)
    x_altered = x.clone()
    x_altered[:, 4:, :] = torch.randn(1, 2, hidden)  # change only the LAST two tokens

    with torch.no_grad():
        out_original = attn(x, rope)
        out_altered = attn(x_altered, rope)

    # Positions 0-3 must be untouched by changes to positions 4-5
    assert torch.allclose(out_original[:, :4, :], out_altered[:, :4, :], atol=1e-6)
    # Position 4-5 SHOULD differ (they were the ones changed, or attend to a changed neighbor)
    assert not torch.allclose(out_original[:, 4:, :], out_altered[:, 4:, :], atol=1e-6)
