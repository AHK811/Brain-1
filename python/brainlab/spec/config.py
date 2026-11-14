from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - python 3.10 only
    import tomli as tomllib

SPEC_VERSION = "2"
_MODEL_KEYS = (
    "name",
    "d_model",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "head_dim",
    "vocab_size",
    "context_len",
    "mlp_hidden",
    "n_experts",
    "experts_per_token",
    "router_aux_loss_coef",
    "rms_norm_eps",
    "rope_theta",
    "dropout",
    "tie_embeddings",
    "qkv_fused",
    "init_std",
    "residual_scale_layers",
    "param_budget",
    "param_tolerance",
)


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    context_len: int
    mlp_hidden: int
    n_experts: int
    experts_per_token: int
    router_aux_loss_coef: float
    rms_norm_eps: float
    rope_theta: float
    dropout: float
    tie_embeddings: bool
    qkv_fused: bool
    init_std: float
    residual_scale_layers: int
    param_budget: int
    param_tolerance: float

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def qkv_out(self) -> int:
        return self.d_model + 2 * self.kv_dim

    @property
    def attn_scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def param_count(self) -> int:
        return sum(math.prod(shape) for shape in self.tensor_shapes().values())

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        d, h, v = self.d_model, self.mlp_hidden, self.vocab_size
        shapes = {"token_emb.weight": (v, d), "final_norm.weight": (d,)}
        if not self.tie_embeddings:
            shapes["lm_head.weight"] = (v, d)
        for i in range(self.n_layers):
            p = f"layers.{i}."
            shapes[p + "attn_norm.weight"] = (d,)
            shapes[p + "attn_qkv.weight"] = (self.qkv_out, d)
            shapes[p + "attn_proj.weight"] = (d, d)
            shapes[p + "mlp_norm.weight"] = (d,)
            if self.n_experts > 1:
                shapes[p + "mlp_router.weight"] = (self.n_experts, d)
            for expert in range(self.n_experts):
                prefix = p if self.n_experts == 1 else f"{p}experts.{expert}."
                shapes[prefix + "mlp_gate.weight"] = (h, d)
                shapes[prefix + "mlp_up.weight"] = (h, d)
                shapes[prefix + "mlp_down.weight"] = (d, h)
        return shapes

    def validate(self) -> None:
        dimensions = (self.d_model, self.n_layers, self.n_heads, self.n_kv_heads,
                      self.head_dim, self.mlp_hidden, self.residual_scale_layers, self.param_budget)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("model dimensions and parameter budget must be positive integers")
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError(f"n_heads*head_dim={self.n_heads * self.head_dim} != d_model={self.d_model}")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if type(self.n_experts) is not int or self.n_experts not in {1, 4, 8}:
            raise ValueError("n_experts must be 1, 4 or 8")
        if self.experts_per_token not in {1, 2} or self.experts_per_token > self.n_experts:
            raise ValueError("experts_per_token must be 1 or 2 and no greater than n_experts")
        if not math.isfinite(self.router_aux_loss_coef) or self.router_aux_loss_coef < 0:
            raise ValueError("router_aux_loss_coef must be finite and non-negative")
        if not 0 <= self.dropout < 1 or not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("invalid dropout or RoPE theta")
        if not self.qkv_fused:
            raise ValueError("weights_format.md only defines fused qkv")
        if self.vocab_size > 65535:
            raise ValueError("vocab must fit uint16 packed shards")
        if self.vocab_size < 259:
            raise ValueError("vocab must hold 256 byte tokens plus merges and specials")
        if self.mlp_hidden % 64:
            raise ValueError("mlp_hidden should be a multiple of 64 for tiled kernels")
        if self.context_len < 8:
            raise ValueError("context_len too small to pack sequences")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.head_dim % 2:
            raise ValueError("rope split-half needs an even head_dim")
        got, want = self.param_count, self.param_budget
        drift = abs(got - want) / want
        if drift > self.param_tolerance:
            raise ValueError(f"param_count {got} is {drift:.3%} from budget {want}, tolerance {self.param_tolerance:.3%}")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelSpec:
        return cls(**{k: raw[k] for k in _MODEL_KEYS})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def metadata(self) -> dict[str, str]:
        return {
            "brainlab.spec_version": SPEC_VERSION,
            "brainlab.name": self.name,
            "brainlab.config_json": json.dumps(self.to_dict(), separators=(",", ":")),
            "brainlab.param_count": str(self.param_count),
        }


def repo_root() -> Path:
    env = os.environ.get("BRAINLAB_ROOT")
    if env:
        return Path(env).resolve()
    seeds = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    for start in seeds:
        for parent in [start, *start.parents]:
            if (parent / "spec" / "model_config.toml").is_file():
                return parent
    raise FileNotFoundError("spec/model_config.toml not found; set BRAINLAB_ROOT")


def load_raw_config(root: Path | None = None) -> dict[str, Any]:
    path = (root or repo_root()) / "spec" / "model_config.toml"
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_model_spec(root: Path | None = None) -> ModelSpec:
    spec = ModelSpec.from_dict(load_raw_config(root)["model"])
    spec.validate()
    return spec


def spec_from_metadata(meta: dict[str, str]) -> ModelSpec:
    if meta.get("brainlab.spec_version") != SPEC_VERSION:
        raise ValueError(f"unsupported spec_version {meta.get('brainlab.spec_version')!r}")
    spec = ModelSpec.from_dict(json.loads(meta["brainlab.config_json"]))
    declared = int(meta["brainlab.param_count"])
    if declared != spec.param_count:
        raise ValueError(f"metadata param_count {declared} != computed {spec.param_count}")
    return spec
