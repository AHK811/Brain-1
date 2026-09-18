"""
tests/test_model.py

Automated tests for BrainConfig, RMSNorm, embeddings, and the assembled
BrainForCausalLM: shape, forward, backward, parameter count, weight tying,
and checkpoint save/load. Fixes audit Problem #4 (zero automated tests) at
the model level.
"""

import math

import pytest
import torch
import torch.nn as nn

from brain.core.config import BrainConfig
from brain.model.brain_model import BrainForCausalLM
from brain.model.embeddings import TokenEmbedding
from brain.model.transformer.normalization import RMSNorm, _ManualRMSNorm


@pytest.fixture
def tiny_config():
    return BrainConfig(
        vocab_size=256, hidden_size=32, num_layers=2, num_heads=4,
        num_kv_heads=2, intermediate_size=64, max_position_embeddings=64,
    )


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def test_config_rejects_indivisible_hidden_size():
    with pytest.raises(ValueError):
        BrainConfig(hidden_size=385, num_heads=6)


def test_config_rejects_odd_head_dim():
    # hidden_size=48, num_heads=16 -> head_dim=3, odd, invalid for RoPE pairing
    with pytest.raises(ValueError):
        BrainConfig(hidden_size=48, num_heads=16, num_kv_heads=16)


def test_config_rejects_kv_heads_not_dividing_heads():
    with pytest.raises(ValueError):
        BrainConfig(hidden_size=384, num_heads=6, num_kv_heads=4)


def test_config_yaml_roundtrip(tmp_path):
    cfg = BrainConfig(hidden_size=64, num_heads=4, num_kv_heads=2)
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    reloaded = BrainConfig.from_yaml(path)
    assert reloaded == cfg


def test_real_brain_0_2_config_param_count():
    cfg = BrainConfig.from_yaml("configs/model/brain_0_2_30m.yaml")
    model = BrainForCausalLM(cfg)
    total = model.count_parameters()["total"]
    # Independently computed in the audit report; regression-test it here
    # so a future accidental config edit can't silently drift the model size.
    assert total == 26_944_896


# --------------------------------------------------------------------------
# RMSNorm
# --------------------------------------------------------------------------
def test_rmsnorm_native_matches_manual_fallback():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    native = nn.RMSNorm(16, eps=1e-5)
    manual = _ManualRMSNorm(16, eps=1e-5)
    manual.weight.data.copy_(native.weight.data)
    assert torch.allclose(native(x), manual(x), atol=1e-6)


def test_rmsnorm_output_rms_is_one_with_unit_weight():
    norm = RMSNorm(16)
    with torch.no_grad():
        norm.weight.fill_(1.0)
    x = torch.randn(4, 16) * 5  # arbitrary scale
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
def test_embedding_padding_idx_stays_zero():
    emb = TokenEmbedding(vocab_size=50, hidden_size=8, pad_id=0)
    assert torch.all(emb.embedding.weight[0] == 0)


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------
def test_model_forward_shape(tiny_config):
    model = BrainForCausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 10))
    out = model(input_ids)
    assert out["logits"].shape == (2, 10, tiny_config.vocab_size)


def test_model_backward_produces_gradients(tiny_config):
    model = BrainForCausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 10))
    out = model(input_ids, labels=input_ids.clone())
    out["loss"].backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_weight_tying_shares_storage(tiny_config):
    model = BrainForCausalLM(tiny_config)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.embedding.weight.data_ptr()


def test_untied_embeddings_do_not_share_storage():
    cfg = BrainConfig(vocab_size=256, hidden_size=32, num_layers=1, num_heads=4,
                       num_kv_heads=2, intermediate_size=64, tie_word_embeddings=False)
    model = BrainForCausalLM(cfg)
    assert model.lm_head.weight.data_ptr() != model.token_embedding.embedding.weight.data_ptr()


def test_checkpoint_save_load_identical_output(tiny_config, tmp_path):
    torch.manual_seed(0)
    model = BrainForCausalLM(tiny_config)
    model.eval()
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 6))

    ckpt_path = tmp_path / "model.pt"
    model.save_checkpoint(ckpt_path)
    reloaded = BrainForCausalLM.load_checkpoint(ckpt_path)
    reloaded.eval()

    with torch.no_grad():
        out1 = model(input_ids)["logits"]
        out2 = reloaded(input_ids)["logits"]
    assert torch.allclose(out1, out2, atol=1e-6)
    assert reloaded.config == tiny_config


def test_residual_write_projections_have_scaled_down_init(tiny_config):
    model = BrainForCausalLM(tiny_config)
    o_proj_std = model.blocks[0].attention.o_proj.weight.std().item()
    q_proj_std = model.blocks[0].attention.q_proj.weight.std().item()
    assert o_proj_std < q_proj_std


def test_real_brain_0_3_config_param_count():
    cfg = BrainConfig.from_yaml("configs/model/brain_0_3_35m.yaml")
    model = BrainForCausalLM(cfg)
    total = model.count_parameters()["total"]
    assert total == 33_236_352, f"unexpected v0.3 param count: {total}"
