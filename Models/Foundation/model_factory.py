from __future__ import annotations
from typing import Any
from Models.Foundation.config import ModelSizeConfig

_ALIASES = {
    "o-mini": "brain_o_mini", "brain-o-mini": "brain_o_mini", "33m": "brain_o_mini",
    "v04": "brain_o_mini", "v0.3": "brain_o_mini", "v0.4": "brain_o_mini",
    "gpt2": "gpt2_small", "gpt2-small": "gpt2_small", "117m": "gpt2_small",
    "gpt2-medium": "gpt2_medium", "345m": "gpt2_medium",
    "gpt2-large": "gpt2_large", "774m": "gpt2_large",
    "gpt2-xl": "gpt2_xl", "1.5b": "gpt2_xl", "1558m": "gpt2_xl",
    "brain-gpt2-small": "brain_gpt2_small", "124m": "brain_gpt2_small",
    "gpt2-plus": "brain_gpt2_plus", "brain-gpt2-plus": "brain_gpt2_plus",
    "deep64": "brain_deep64", "brain-deep64": "brain_deep64",
    "mla": "brain_mla", "brain-mla": "brain_mla",
    "brain1": "brain1_250m", "250m": "brain1_250m", "brain1-250m": "brain1_250m",
    "brain2": "brain2_500m", "500m": "brain2_500m", "brain2-500m": "brain2_500m",
    "v5": "brain_v5_emergent", "brain-v5": "brain_v5_emergent",
    "brain-v5-emergent": "brain_v5_emergent",
    "v5-96": "brain_v5_96", "brain-v5-96": "brain_v5_96",
}

def build_brain_config(size: ModelSizeConfig):
    from brain.core.config import BrainConfig
    return BrainConfig(
        name=size.name,
        vocab_size=size.vocab_size,
        max_position_embeddings=size.max_position_embeddings,
        hidden_size=size.hidden_size,
        num_layers=size.num_layers,
        num_heads=size.num_heads,
        num_kv_heads=size.num_kv_heads,
        intermediate_size=size.intermediate_size,
        rope_theta=size.rope_theta,
        rope_scaling_type=size.rope_scaling_type,
        rope_scaling_factor=size.rope_scaling_factor,
        tie_word_embeddings=True,
        dropout=0.0,
        use_qk_norm=size.use_qk_norm,
        attention_type=size.attention_type,
        mla_latent_dim=size.mla_latent_dim,
        sliding_window=size.sliding_window,
        ffn_type=size.ffn_type,
        norm_type=size.norm_type,
        position_embedding_type=size.position_embedding_type,
        attention_bias=size.attention_bias,
        mlp_bias=size.mlp_bias,
        sparse_alternating=getattr(size, "sparse_alternating", False),
        sparse_local_window=getattr(size, "sparse_local_window", 256),
    )

def create_model(size: str | ModelSizeConfig = "o-mini") -> Any:
    if isinstance(size, str):
        key = size.lower().replace("_", "-")
        alias = _ALIASES.get(key)
        if alias is None:
            raise ValueError(
                f"Unknown model size: {size!r}. Expected one of: {', '.join(sorted(set(_ALIASES)))}"
            )
        size = getattr(ModelSizeConfig, alias)()
    from brain.model.brain_model import BrainForCausalLM
    return BrainForCausalLM(build_brain_config(size))
