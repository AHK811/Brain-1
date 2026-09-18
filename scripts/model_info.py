"""
scripts/model_info.py

Step 13 requirement: report the parameter count via an explicit command,
never just claim a number. Run with a config path OR a Models.create_model()
key (o-mini, 250m, 500m, gpt2-small, gpt2-plus, deep64, mla, ...) to see the
real breakdown for that architecture.

Usage:
    PYTHONPATH=. python3 scripts/model_info.py configs/model/brain_0_2_30m.yaml
    PYTHONPATH=. python3 scripts/model_info.py deep64
"""

import sys
from pathlib import Path

from brain.core.config import BrainConfig
from brain.model.brain_model import BrainForCausalLM


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "configs/model/brain_0_2_30m.yaml"
    if Path(arg).exists():
        config = BrainConfig.from_yaml(arg)
        model = BrainForCausalLM(config)
        label = arg
    else:
        from Models.Foundation.model_factory import create_model
        model = create_model(arg)
        config = model.config
        label = f"create_model({arg!r})"
    counts = model.count_parameters()

    print(f"Brain config: {config.name}  ({label})")
    print("-" * 50)
    for k, v in config.to_dict().items():
        print(f"  {k}: {v}")
    print("-" * 50)
    print("Parameter breakdown:")
    for k, v in counts.items():
        pct = f" ({100 * v / counts['total']:.1f}%)" if k != "total" and k != "trainable" else ""
        print(f"  {k:14s}: {v:>12,}{pct}")
    print("-" * 50)
    print(f"Total: {counts['total']:,} ({counts['total']/1e6:.2f}M)")

    if model.rope.effective_theta != config.rope_theta:
        print("-" * 50)
        print(
            f"NOTE: RoPE scaling ({config.rope_scaling_type} x{config.rope_scaling_factor}) "
            f"is applied unconditionally from construction, not only once generation exceeds "
            f"max_position_embeddings -- the theta actually running is "
            f"{model.rope.effective_theta:,.0f}, not the configured {config.rope_theta:,.0f}."
        )


if __name__ == "__main__":
    main()
