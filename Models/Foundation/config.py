from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# Official GPT-2 BPE vocab size (not 1M — OpenAI GPT-2 uses 50257)
GPT2_VOCAB = 50257

@dataclass
class ModelSizeConfig:
    name: str
    vocab_size: int = 32768
    max_position_embeddings: int = 4096
    hidden_size: int = 384
    num_layers: int = 12
    num_heads: int = 6
    num_kv_heads: int = 2
    intermediate_size: int = 1152
    rope_theta: float = 500000.0
    rope_scaling_type: Optional[str] = "ntk"
    rope_scaling_factor: float = 2.0
    use_qk_norm: bool = False
    attention_type: str = "gqa"
    mla_latent_dim: Optional[int] = None
    sliding_window: Optional[int] = None
    ffn_type: str = "swiglu"
    norm_type: str = "rmsnorm"
    position_embedding_type: str = "rope"
    attention_bias: bool = False
    mlp_bias: bool = False
    sparse_alternating: bool = False
    sparse_local_window: int = 256

    @staticmethod
    def brain_o_mini():
        return ModelSizeConfig(
            name="brain-o-mini", vocab_size=32768, max_position_embeddings=4096,
            hidden_size=384, num_layers=12, num_heads=6, num_kv_heads=2,
            intermediate_size=1152, use_qk_norm=False, attention_type="gqa",
        )

    # ---- Classic GPT-2 family (Radford et al.) ----
    @staticmethod
    def gpt2_small():
        """GPT-2 Small ~117M: 12L, d=768, 12 heads, FFN 3072, ctx 1024, vocab 50257."""
        return ModelSizeConfig(
            name="gpt2-small", vocab_size=GPT2_VOCAB, max_position_embeddings=1024,
            hidden_size=768, num_layers=12, num_heads=12, num_kv_heads=12,
            intermediate_size=3072, rope_theta=10000.0, rope_scaling_type=None,
            rope_scaling_factor=1.0, attention_type="mha", ffn_type="gelu_mlp",
            norm_type="layernorm", position_embedding_type="learned",
            attention_bias=True, mlp_bias=True,
        )

    @staticmethod
    def gpt2_medium():
        """GPT-2 Medium ~345M: 24L, d=1024, 16 heads, FFN 4096."""
        return ModelSizeConfig(
            name="gpt2-medium", vocab_size=GPT2_VOCAB, max_position_embeddings=1024,
            hidden_size=1024, num_layers=24, num_heads=16, num_kv_heads=16,
            intermediate_size=4096, rope_theta=10000.0, rope_scaling_type=None,
            rope_scaling_factor=1.0, attention_type="mha", ffn_type="gelu_mlp",
            norm_type="layernorm", position_embedding_type="learned",
            attention_bias=True, mlp_bias=True,
        )

    @staticmethod
    def gpt2_large():
        """GPT-2 Large ~774M: 36L, d=1280, 20 heads, FFN 5120."""
        return ModelSizeConfig(
            name="gpt2-large", vocab_size=GPT2_VOCAB, max_position_embeddings=1024,
            hidden_size=1280, num_layers=36, num_heads=20, num_kv_heads=20,
            intermediate_size=5120, rope_theta=10000.0, rope_scaling_type=None,
            rope_scaling_factor=1.0, attention_type="mha", ffn_type="gelu_mlp",
            norm_type="layernorm", position_embedding_type="learned",
            attention_bias=True, mlp_bias=True,
        )

    @staticmethod
    def gpt2_xl():
        """GPT-2 XL ~1.5B: 48L, d=1600, 25 heads, FFN 6400."""
        return ModelSizeConfig(
            name="gpt2-xl", vocab_size=GPT2_VOCAB, max_position_embeddings=1024,
            hidden_size=1600, num_layers=48, num_heads=25, num_kv_heads=25,
            intermediate_size=6400, rope_theta=10000.0, rope_scaling_type=None,
            rope_scaling_factor=1.0, attention_type="mha", ffn_type="gelu_mlp",
            norm_type="layernorm", position_embedding_type="learned",
            attention_bias=True, mlp_bias=True,
        )

    @staticmethod
    def brain_gpt2_small():
        """Brain hybrid: GPT-2 Small width + RoPE/SwiGLU option (32k vocab)."""
        return ModelSizeConfig(
            name="brain-gpt2-small", vocab_size=32768, max_position_embeddings=1024,
            hidden_size=768, num_layers=12, num_heads=12, num_kv_heads=12,
            intermediate_size=3072, rope_theta=10000.0, rope_scaling_type=None,
            rope_scaling_factor=1.0, attention_type="mha", ffn_type="gelu_mlp",
            norm_type="layernorm", position_embedding_type="learned",
            attention_bias=True, mlp_bias=True,
        )

    @staticmethod
    def brain_gpt2_plus():
        return ModelSizeConfig(
            name="brain-gpt2-plus", vocab_size=32768, max_position_embeddings=2048,
            hidden_size=768, num_layers=12, num_heads=16, num_kv_heads=8,
            intermediate_size=3072, rope_theta=10000.0, rope_scaling_type="ntk",
            rope_scaling_factor=2.0, use_qk_norm=True, attention_type="gqa",
            ffn_type="swiglu", norm_type="rmsnorm", position_embedding_type="rope",
        )

    @staticmethod
    def brain_deep64():
        # The "add more attention / upgrade the model" tier: deeper (64L vs
        # brain2-500m's 32L), richer GQA ratio (2:1 vs 500m's 5:1 -- a real
        # "more KV heads" upgrade, not just bigger), QK-Norm (the actual
        # attention-mechanism upgrade -- see attention.py), and wider overall
        # (~839M vs 500m's ~592.5M). rope_scaling_type is "ntk" (verified,
        # not the experimental "yarn" approximation -- see rope.py's
        # docstring) so this config only uses fully-validated components.
        return ModelSizeConfig(
            name="brain-deep64", vocab_size=32768, max_position_embeddings=8192,
            hidden_size=1024, num_layers=64, num_heads=16, num_kv_heads=8,
            intermediate_size=3072, rope_theta=500000.0, rope_scaling_type="ntk",
            rope_scaling_factor=2.0, use_qk_norm=True, attention_type="gqa",
        )

    @staticmethod
    def brain_mla():
        return ModelSizeConfig(
            name="brain-mla", vocab_size=32768, max_position_embeddings=2048,
            hidden_size=768, num_layers=12, num_heads=16, num_kv_heads=16,
            intermediate_size=3072, rope_theta=10000.0, rope_scaling_type="ntk",
            rope_scaling_factor=2.0, use_qk_norm=True, attention_type="mla",
            mla_latent_dim=128,
        )

    @staticmethod
    def brain1_250m():
        # use_qk_norm intentionally False: this exact architecture was
        # verified against the spec brief and locked by a regression test
        # (304,137,216 params) in the prior review round. Turning on
        # use_qk_norm here would silently change that already-verified
        # number by a few thousand params -- if you want QK-norm applied
        # to a 250M-class model, use brain_gpt2_plus/brain_deep64-style
        # configs (or bump this to a new named variant) rather than
        # mutating the one that's already been verified and locked.
        return ModelSizeConfig(
            name="brain1-250m", vocab_size=32768, max_position_embeddings=8192,
            hidden_size=1024, num_layers=24, num_heads=16, num_kv_heads=4,
            intermediate_size=2816, use_qk_norm=False, attention_type="gqa",
        )

    @staticmethod
    def brain2_500m():
        # See brain1_250m's note: use_qk_norm intentionally False to
        # preserve the already-verified, locked 592,528,640 count.
        return ModelSizeConfig(
            name="brain2-500m", vocab_size=32768, max_position_embeddings=8192,
            hidden_size=1280, num_layers=32, num_heads=20, num_kv_heads=4,
            intermediate_size=3456, use_qk_norm=False, attention_type="gqa",
        )

    @staticmethod
    def brain_v5_emergent():
        """v0.5 flagship: wider, deeper, sparse alternating, QK-norm, NTK."""
        return ModelSizeConfig(
            name="brain-v5-emergent",
            vocab_size=32768,
            max_position_embeddings=8192,
            hidden_size=1536,
            num_layers=48,
            num_heads=24,
            num_kv_heads=8,
            intermediate_size=6144,
            rope_theta=500000.0,
            rope_scaling_type="ntk",
            rope_scaling_factor=2.0,
            use_qk_norm=True,
            attention_type="gqa",
            ffn_type="swiglu",
            norm_type="rmsnorm",
            position_embedding_type="rope",
            sparse_alternating=True,
            sparse_local_window=256,
        )

    @staticmethod
    def brain_v5_96():
        """96 layers x 96 heads research scaffold."""
        return ModelSizeConfig(
            name="brain-v5-96",
            vocab_size=32768,
            max_position_embeddings=4096,
            hidden_size=1536,
            num_layers=96,
            num_heads=96,
            num_kv_heads=16,
            intermediate_size=6144,
            rope_theta=500000.0,
            rope_scaling_type="ntk",
            rope_scaling_factor=2.0,
            use_qk_norm=True,
            attention_type="gqa",
            sparse_alternating=True,
            sparse_local_window=128,
        )
