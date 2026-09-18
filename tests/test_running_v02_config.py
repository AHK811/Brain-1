"""Lock the restored v0.2 baseline so it cannot silently drift again."""
from brain.core.config import BrainConfig
from brain.model.brain_model import BrainForCausalLM


def test_v02_yaml_exact_fields():
    cfg = BrainConfig.from_yaml("configs/model/brain_0_2_30m.yaml")
    assert cfg.name == "brain-0.2"
    assert cfg.vocab_size == 16384
    assert cfg.max_position_embeddings == 512
    assert cfg.hidden_size == 384
    assert cfg.num_layers == 12
    assert cfg.num_heads == 6
    assert cfg.num_kv_heads == 2
    assert cfg.intermediate_size == 1152
    assert cfg.tie_word_embeddings is True


def test_v02_param_count_locked():
    cfg = BrainConfig.from_yaml("configs/model/brain_0_2_30m.yaml")
    total = BrainForCausalLM(cfg).count_parameters()["total"]
    assert total == 26_944_896


def test_v03_param_count_locked():
    cfg = BrainConfig.from_yaml("configs/model/brain_0_3_35m.yaml")
    total = BrainForCausalLM(cfg).count_parameters()["total"]
    assert total == 33_236_352
