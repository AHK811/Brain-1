from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class ModelLifecycle:
    name: str
    params: int
    weights: str  # "trained" | "architecture_only"
    notes: str = ""

class ModelRegistry:
    def __init__(self):
        self._builders: dict[str, Callable[[], Any]] = {}
        self._lifecycle: dict[str, ModelLifecycle] = {}

    def register(self, name: str, builder: Callable[[], Any], lifecycle: ModelLifecycle | None = None) -> None:
        self._builders[name] = builder
        if lifecycle:
            self._lifecycle[name] = lifecycle

    def build(self, name: str) -> Any:
        if name not in self._builders:
            raise KeyError(name)
        return self._builders[name]()

    def list(self) -> list[str]:
        return sorted(self._builders)

    def lifecycle(self, name: str) -> ModelLifecycle | None:
        return self._lifecycle.get(name)

    def status_table(self) -> list[dict]:
        rows = []
        for name in self.list():
            lc = self._lifecycle.get(name)
            rows.append({
                "name": name,
                "params": lc.params if lc else None,
                "weights": lc.weights if lc else "unknown",
                "notes": lc.notes if lc else "",
            })
        return rows


def default_registry() -> ModelRegistry:
    """Lifecycle status for the Foundation model family.

    Honesty check performed during review: no .pt/.ckpt/.safetensors file
    ships anywhere in this repo, so nothing here can accurately claim
    "trained" weights, including brain-o-mini -- doing so would repeat the
    exact overclaiming problem this registry exists to prevent. If a real
    trained checkpoint exists outside this repo and gets wired up, flip
    only that entry's `weights` field once that's actually true.

    Parameter counts below are cross-checked against live construction in
    tests/test_registry_lifecycle.py (`count_parameters()["total"]`) rather
    than trusted as hardcoded literals -- gpt2-small's count here was
    previously wrong (138,431,232 claimed vs. 124,439,040 actual) precisely
    because nothing verified it against the real model.
    """
    from Models.Foundation.model_factory import create_model
    reg = ModelRegistry()
    reg.register("brain-o-mini", lambda: create_model("o-mini"), ModelLifecycle(
        "brain-o-mini", 33_236_352, "architecture_only",
        "v0.3/v0.4 lineage; architecture + numerics verified, no shipped checkpoint"
    ))
    reg.register("brain-gpt2-small", lambda: create_model("gpt2-small"), ModelLifecycle(
        "brain-gpt2-small", 124_439_040, "architecture_only", "Phase 5 GPT-2 Small class; train next"
    ))
    reg.register("brain-gpt2-plus", lambda: create_model("gpt2-plus"), ModelLifecycle(
        "brain-gpt2-plus", 131_354_496, "architecture_only", "GQA 2:1 KV ratio + QK-norm over GPT-2-small width"
    ))
    reg.register("brain-deep64", lambda: create_model("deep64"), ModelLifecycle(
        "brain-deep64", 839_001_088, "architecture_only",
        "deep (64L) + rich GQA (2:1) + QK-norm upgrade tier; pretraining-ready, not trained"
    ))
    reg.register("brain-mla", lambda: create_model("mla"), ModelLifecycle(
        "brain-mla", 128_995_200, "architecture_only",
        "EXPERIMENTAL: causal-masking bug fixed during review, but KV cache is "
        "NOT yet compressed (materializes full per-head K/V before caching) -- "
        "see MultiHeadLatentAttention's docstring. Don't treat this as a "
        "memory-efficient config yet."
    ))
    reg.register("brain1-250m", lambda: create_model("250m"), ModelLifecycle(
        "brain1-250m", 304_137_216, "architecture_only", "beyond GPT-2 Small"
    ))
    reg.register("brain2-500m", lambda: create_model("500m"), ModelLifecycle(
        "brain2-500m", 592_528_640, "architecture_only", "beyond GPT-2 Medium"
    ))
    return reg
