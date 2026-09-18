from __future__ import annotations
import pytest
from brain.core.config import BrainConfig
from Models.Foundation.config import ModelSizeConfig
from Models.Foundation.model_factory import create_model, build_brain_config
from Models.Foundation.model_registry import default_registry

def test_unknown_size_raises():
    with pytest.raises(ValueError, match="Unknown model size"):
        create_model("does-not-exist")

def test_config_rejects_bad_vocab():
    with pytest.raises(ValueError, match="vocab_size"):
        BrainConfig(vocab_size=0)

def test_config_rejects_bad_ctx():
    with pytest.raises(ValueError, match="max_position_embeddings"):
        BrainConfig(max_position_embeddings=0)

def test_config_rejects_small_intermediate():
    with pytest.raises(ValueError, match="intermediate_size"):
        BrainConfig(hidden_size=384, num_heads=6, num_kv_heads=2, intermediate_size=10)

def test_o_mini_params_locked():
    m = create_model("o-mini")
    assert m.count_parameters()["total"] == 33_236_352

def test_gpt2_small_params_locked():
    m = create_model("gpt2-small")
    total = m.count_parameters()["total"]
    # GPT-2 Small class with vocab 32k + SwiGLU — expect ~120–150M
    assert 100_000_000 < total < 160_000_000
    cfg = m.config
    assert cfg.hidden_size == 768
    assert cfg.num_layers == 12
    assert cfg.num_heads == 12
    assert cfg.num_kv_heads == 12
    assert cfg.max_position_embeddings == 1024

def test_250m_params_locked():
    sc = ModelSizeConfig.brain1_250m()
    # Build config only is enough if memory tight — construct model
    m = create_model(sc)
    assert m.count_parameters()["total"] == 304_137_216

def test_registry_lifecycle():
    reg = default_registry()
    assert "brain-o-mini" in reg.list()
    assert "brain-gpt2-small" in reg.list()
    lc = reg.lifecycle("brain-o-mini")
    assert lc is not None and lc.weights == "trained"
    assert reg.lifecycle("brain-gpt2-small").weights == "architecture_only"
