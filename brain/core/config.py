"""
brain/core/config.py

Single source of truth for model architecture hyperparameters.

Fixes CRITICAL problem #2 from the audit: the original Brain Trail prototype
hardcoded `D_MODEL, N_HEADS, N_LAYERS = 64, 4, 2` as bare literals copy-pasted
across six separate files. Every model, script, and checkpoint in Brain v0.2
instead goes through this one dataclass, loaded from a YAML file on disk.

This module has zero dependency on torch, FastAPI, databases, or anything
else — it is pure configuration and knows nothing about how it's used.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass
class BrainConfig:
    """Architecture configuration for a Brain causal language model.

    Every field has an explicit default so `BrainConfig()` alone constructs
    a valid (if minimal) model — but real configs should always come from a
    YAML file under configs/model/, not from code, per the "don't hardcode
    model configuration" engineering rule.
    """

    name: str = "brain-0.3"

    # --- vocabulary / sequence -------------------------------------------------
    vocab_size: int = 32768
    max_position_embeddings: int = 4096

    # --- transformer body --------------------------------------------------
    hidden_size: int = 384
    num_layers: int = 12
    num_heads: int = 6
    num_kv_heads: int = 2          # < num_heads enables GQA; == num_heads is plain MHA
    intermediate_size: int = 1152  # SwiGLU inner dimension

    # --- positional encoding ------------------------------------------------
    rope_theta: float = 500000.0   # higher base for long-context stability (v0.3)
    # RoPE scaling (None / "linear" / "ntk"). Used when extending beyond original training length.
    rope_scaling_type: str | None = "ntk"
    rope_scaling_factor: float = 2.0   # effective context ≈ original * factor when using scaling

    # --- normalization -------------------------------------------------------
    norm_eps: float = 1e-5

    # --- parameter-efficiency toggles ----------------------------------------
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = True

    # --- regularization --------------------------------------------------------
    dropout: float = 0.0  # pretraining default; SFT configs should override this

    # --- attention upgrades (Phase 6) ----------------------------------------
    use_qk_norm: bool = False          # RMSNorm Q and K per-head before RoPE
    attention_type: str = "gqa"        # "mha" | "gqa" | "mla"
    mla_latent_dim: int | None = None  # for MLA: latent rank (defaults to head_dim)
    sliding_window: int | None = None  # local attention window; None = full causal

    # --- GPT-2 style options -------------------------------------------------
    ffn_type: str = "swiglu"              # "swiglu" | "gelu_mlp"
    norm_type: str = "rmsnorm"            # "rmsnorm" | "layernorm"
    position_embedding_type: str = "rope" # "rope" | "learned" | "rope_and_learned"

    sparse_alternating: bool = False
    sparse_local_window: int = 256

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        head_dim = self.hidden_size // self.num_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be even for RoPE (rotates "
                f"dimensions in pairs)"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads}) for grouped-query attention"
            )
        if self.num_kv_heads > self.num_heads:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) cannot exceed "
                f"num_heads ({self.num_heads})"
            )
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size ({self.vocab_size}) must be positive")
        if self.max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings ({self.max_position_embeddings}) must be positive"
            )
        if self.intermediate_size <= self.hidden_size:
            raise ValueError(
                f"intermediate_size ({self.intermediate_size}) should exceed "
                f"hidden_size ({self.hidden_size})"
            )
        if self.attention_type not in ("mha", "gqa", "mla"):
            raise ValueError(
                f"attention_type ({self.attention_type!r}) must be 'mha', 'gqa', or 'mla'"
            )
        if self.attention_type == "mha" and self.num_kv_heads != self.num_heads:
            # Previously a SILENT no-op: attention_type="mha" was accepted with
            # num_kv_heads < num_heads and quietly built GQA anyway (block.py's
            # dispatch only branches on "mla" vs "not mla" -- "mha" and "gqa"
            # took the identical code path). Declaring "mha" is now a real
            # constraint, not a label that could be wrong without complaint.
            raise ValueError(
                f"attention_type='mha' requires num_kv_heads == num_heads "
                f"(got num_kv_heads={self.num_kv_heads}, num_heads={self.num_heads}); "
                f"use attention_type='gqa' for num_kv_heads < num_heads"
            )
        if self.ffn_type not in ("swiglu", "gelu_mlp"):
            raise ValueError(f"ffn_type ({self.ffn_type!r}) must be 'swiglu' or 'gelu_mlp'")
        if self.norm_type not in ("rmsnorm", "layernorm"):
            raise ValueError(f"norm_type ({self.norm_type!r}) must be 'rmsnorm' or 'layernorm'")
        if self.position_embedding_type not in ("rope", "learned", "rope_and_learned"):
            raise ValueError(
                f"position_embedding_type ({self.position_embedding_type!r}) must be "
                f"'rope', 'learned', or 'rope_and_learned'"
            )

    # ------------------------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def uses_gqa(self) -> bool:
        return self.num_kv_heads < self.num_heads

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "BrainConfig":
        path = Path(path)
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        # Allow either a flat file or one nested under a top-level "model:" key
        data: dict[str, Any] = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        known_fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known_fields
        if unknown:
            raise ValueError(f"Unknown config field(s) in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump({"model": dataclasses.asdict(self)}, f, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrainConfig":
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = BrainConfig()
    print("Default BrainConfig:")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    print(f"\nhead_dim = {cfg.head_dim}")
    print(f"uses_gqa = {cfg.uses_gqa}")

    # Round-trip test
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = Path(tmp) / "test_config.yaml"
        cfg.to_yaml(yaml_path)
        reloaded = BrainConfig.from_yaml(yaml_path)
        print(f"\nRound-trip (save -> load) matches: {cfg == reloaded}")

    # Validation test: bad config should raise
    try:
        BrainConfig(hidden_size=385, num_heads=6)
        print("FAIL: should have raised on non-divisible hidden_size/num_heads")
    except ValueError as e:
        print(f"\nValidation correctly rejected bad config: {e}")
