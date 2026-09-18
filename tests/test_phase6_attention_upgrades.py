"""Regression tests for the Phase 6 attention-upgrade fixes.

Context for future readers: the uploaded Phase 6 work (QK-Norm, MLA, sliding
window, GPT-2-style toggles) initially shipped with `is_causal=True` passed
unconditionally in both CausalSelfAttention and MultiHeadLatentAttention.
PyTorch's SDPA interprets is_causal=True with mismatched query/key lengths
(the exact shape you get during cached decode: 1 new token vs. many cached
keys) as "only attend to key position 0" -- so every decode step past the
first was silently wrong for every non-sliding-window config. This file
locks in the fix so it can't regress unnoticed the way it did the first
time (the existing parametrized test in test_attention.py would already
have caught it had it been run before shipping).
"""
import pytest
import torch

from brain.core.config import BrainConfig
from brain.model.brain_model import BrainForCausalLM
from brain.model.transformer.attention import CausalSelfAttention, MultiHeadLatentAttention
from brain.model.transformer.cache import KVCache
from brain.model.transformer.rope import RotaryEmbedding
from Models import create_model


# ---------------------------------------------------------------------------
# The critical is_causal fix, with the NEW features active (qk_norm)
# ---------------------------------------------------------------------------
def test_qk_norm_attention_cache_matches_uncached():
    torch.manual_seed(0)
    hidden, heads, kv_heads = 32, 4, 2
    attn = CausalSelfAttention(hidden, heads, kv_heads, use_qk_norm=True)
    attn.eval()
    rope = RotaryEmbedding(hidden // heads, max_position_embeddings=32)

    x = torch.randn(1, 7, hidden)
    with torch.no_grad():
        out_full = attn(x, rope)
        cache = KVCache(num_layers=1)
        outs = [attn(x[:, i:i+1, :], rope, kv_cache=cache, layer_idx=0) for i in range(7)]
        out_cached = torch.cat(outs, dim=1)

    assert torch.allclose(out_full, out_cached, atol=1e-5)


def test_mla_cache_matches_uncached():
    """MultiHeadLatentAttention had the identical is_causal bug -- confirm
    the fix independently, since it's a separate forward() implementation."""
    torch.manual_seed(0)
    hidden, heads = 32, 4
    mla = MultiHeadLatentAttention(hidden, heads, latent_dim=8)
    mla.eval()
    rope = RotaryEmbedding(hidden // heads, max_position_embeddings=32)

    x = torch.randn(1, 7, hidden)
    with torch.no_grad():
        out_full = mla(x, rope)
        cache = KVCache(num_layers=1)
        outs = [mla(x[:, i:i+1, :], rope, kv_cache=cache, layer_idx=0) for i in range(7)]
        out_cached = torch.cat(outs, dim=1)

    assert torch.allclose(out_full, out_cached, atol=1e-5)


def test_deep64_full_model_cache_matches_uncached_at_scale():
    """The end-to-end version of the above, on the actual real-scale
    brain-deep64 config (839M params, 64 layers, GQA 2:1, QK-norm, NTK
    RoPE) -- not just a tiny synthetic config."""
    torch.manual_seed(0)
    model = create_model("deep64")
    model.eval()
    prompt = torch.randint(0, model.config.vocab_size, (1, 5))

    out_nocache = model.generate(prompt, max_new_tokens=6, temperature=0.0, use_cache=False)
    torch.manual_seed(0)
    out_cache = model.generate(prompt, max_new_tokens=6, temperature=0.0, use_cache=True)
    assert torch.equal(out_nocache, out_cache)


# ---------------------------------------------------------------------------
# attention_mask: restored after being dropped in the Phase 6 rewrite
# ---------------------------------------------------------------------------
def test_attention_mask_matches_unpadded_run():
    torch.manual_seed(0)
    cfg = BrainConfig(vocab_size=256, hidden_size=32, num_layers=2, num_heads=4,
                       num_kv_heads=2, intermediate_size=64, max_position_embeddings=64)
    model = BrainForCausalLM(cfg)
    model.eval()

    seq = torch.randint(0, 256, (1, 5))
    mask = torch.cat([torch.ones(1, 5, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long)], dim=1)
    padded = torch.cat([seq, torch.randint(0, 256, (1, 3))], dim=1)
    with torch.no_grad():
        alone = model(seq)["logits"]
        masked = model(padded, attention_mask=mask)["logits"][:, :5]
    assert torch.allclose(alone, masked, atol=1e-5)


def test_attention_mask_with_kv_cache_raises_clear_error():
    cfg = BrainConfig(vocab_size=256, hidden_size=32, num_layers=1, num_heads=4,
                       num_kv_heads=2, intermediate_size=64, max_position_embeddings=64)
    model = BrainForCausalLM(cfg)
    cache = KVCache(num_layers=1)
    seq = torch.randint(0, 256, (1, 5))
    with pytest.raises(NotImplementedError):
        model(seq, kv_cache=cache, attention_mask=torch.ones_like(seq))


def test_sliding_window_with_kv_cache_raises_instead_of_silently_ignoring():
    """Previously: sliding_window was silently IGNORED whenever a kv_cache
    was passed (fell through to full-history attention). An explicit error
    is safer than silently attending outside the declared window."""
    hidden, heads, kv_heads = 32, 4, 2
    attn = CausalSelfAttention(hidden, heads, kv_heads, sliding_window=4)
    rope = RotaryEmbedding(hidden // heads, max_position_embeddings=32)
    cache = KVCache(num_layers=1)
    x = torch.randn(1, 1, hidden)
    with pytest.raises(NotImplementedError):
        attn(x, rope, kv_cache=cache, layer_idx=0)


# ---------------------------------------------------------------------------
# attention_type="mha" validation (previously a silent no-op)
# ---------------------------------------------------------------------------
def test_mha_with_mismatched_kv_heads_raises():
    with pytest.raises(ValueError):
        BrainConfig(hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=2,
                    intermediate_size=64, max_position_embeddings=32, attention_type="mha")


def test_mha_with_matching_kv_heads_is_accepted():
    cfg = BrainConfig(hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=4,
                       intermediate_size=64, max_position_embeddings=32, attention_type="mha")
    assert cfg.attention_type == "mha"


def test_invalid_attention_type_raises():
    with pytest.raises(ValueError):
        BrainConfig(hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=4,
                    intermediate_size=64, max_position_embeddings=32, attention_type="bogus")


# ---------------------------------------------------------------------------
# RoPE dynamic extension preserves device/dtype
# ---------------------------------------------------------------------------
def test_rope_extension_preserves_dtype():
    r = RotaryEmbedding(head_dim=8, max_position_embeddings=10)
    r.cos = r.cos.to(torch.float64)
    r.sin = r.sin.to(torch.float64)
    cos, sin = r.get(seq_len=5, position_offset=8)  # forces extension past 10
    assert cos.dtype == torch.float64 and sin.dtype == torch.float64


# ---------------------------------------------------------------------------
# brain-deep64 parameter count, locked
# ---------------------------------------------------------------------------
def test_deep64_param_count_locked():
    assert create_model("deep64").count_parameters()["total"] == 839_001_088


def test_250m_500m_unaffected_by_qk_norm_default():
    """Guard against the exact regression that happened once already:
    use_qk_norm=True was silently turned on for these two previously-locked
    configs, changing their verified parameter counts by a few thousand."""
    assert create_model("250m").config.use_qk_norm is False
    assert create_model("500m").config.use_qk_norm is False
