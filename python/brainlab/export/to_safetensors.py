from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import safetensors
import torch
from safetensors.torch import load_file, save_file

from brainlab.model.gpt import TransformerLM
from brainlab.spec.config import ModelSpec, load_model_spec, spec_from_metadata

TORCH_DTYPE = {"f32": torch.float32, "f16": torch.float16}
MODULE_TO_FORMAT = {
    "token_emb.weight": "token_emb.weight",
    "final_norm.weight": "final_norm.weight",
}
BLOCK_TO_FORMAT = {
    "attn_norm.weight": "attn_norm.weight",
    "attn.qkv.weight": "attn_qkv.weight",
    "attn.proj.weight": "attn_proj.weight",
    "mlp_norm.weight": "mlp_norm.weight",
    "mlp.gate.weight": "mlp_gate.weight",
    "mlp.up.weight": "mlp_up.weight",
    "mlp.down.weight": "mlp_down.weight",
}
FORMAT_TO_BLOCK = {v: k for k, v in BLOCK_TO_FORMAT.items()}


def to_weight_names(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if name.startswith("blocks."):
            _, index, rest = name.split(".", 2)
            formatted = BLOCK_TO_FORMAT.get(rest)
            if formatted is None:
                raise KeyError(f"no weights_format name for module parameter {name!r}")
            out[f"layers.{index}.{formatted}"] = tensor
        elif name in MODULE_TO_FORMAT:
            out[MODULE_TO_FORMAT[name]] = tensor
        else:
            raise KeyError(f"no weights_format name for module parameter {name!r}")
    return out


def to_module_names(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if name.startswith("layers."):
            _, index, rest = name.split(".", 2)
            module = FORMAT_TO_BLOCK.get(rest)
            if module is None:
                raise KeyError(f"weights {name!r} not in spec/weights_format.md")
            out[f"blocks.{index}.{module}"] = tensor
        elif name in MODULE_TO_FORMAT:
            out[MODULE_TO_FORMAT[name]] = tensor
        else:
            raise KeyError(f"weights {name!r} not in spec/weights_format.md")
    return out


def validate_names(tensors: dict[str, torch.Tensor], spec: ModelSpec) -> None:
    expected = spec.tensor_shapes()
    missing = sorted(set(expected) - set(tensors))
    extra = sorted(set(tensors) - set(expected))
    if missing or extra:
        raise ValueError(f"checkpoint mismatch: missing={missing[:6]} extra={extra[:6]}")
    bad = {n: (tuple(tensors[n].shape), expected[n]) for n in expected if tuple(tensors[n].shape) != expected[n]}
    if bad:
        raise ValueError(f"shape mismatch: {list(bad.items())[:4]}")
    total = sum(t.numel() for t in tensors.values())
    if total != spec.param_count:
        raise ValueError(f"tensor element total {total} != spec param_count {spec.param_count}")


def save_model(
    model: TransformerLM,
    path: Path,
    dtype: torch.dtype = torch.float32,
    extra_meta: dict[str, str] | None = None,
) -> Path:
    spec = model.spec
    spec.validate()
    state = {k: v.detach().to(dtype=dtype).contiguous() for k, v in model.state_dict().items()}
    formatted = to_weight_names(state)
    validate_names(formatted, spec)
    meta = spec.metadata()
    meta["brainlab.source"] = "python/brainlab/export/to_safetensors.py"
    meta["brainlab.producer_versions"] = json.dumps(
        {"torch": torch.__version__, "safetensors": safetensors.__version__},
        separators=(",", ":"),
    )
    if extra_meta:
        meta.update(extra_meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(formatted, str(path), metadata=meta)
    return path


def load_metadata(path: Path) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        return dict(handle.metadata() or {})


def load_tensors(path: Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    return {k: v.to(device).contiguous() for k, v in load_file(str(path), device=device).items()}


def spec_from_checkpoint(path: Path) -> ModelSpec:
    spec = spec_from_metadata(load_metadata(path))
    spec.validate()
    return spec


def load_model(
    path: Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    spec: ModelSpec | None = None,
) -> TransformerLM:
    spec = spec or spec_from_checkpoint(path)
    tensors = load_tensors(path, device)
    validate_names(tensors, spec)
    model = TransformerLM(spec)
    model.to(device=device, dtype=dtype)
    model.load_state_dict({k: v for k, v in to_module_names(tensors).items()})
    return model.eval()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brainlab-export")
    parser.add_argument("--checkpoint", type=Path, required=True, help="torch .pt with a model key")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=tuple(TORCH_DTYPE), default="f32")
    parser.add_argument("--spec", type=Path, default=None, help="override spec/model_config.toml")
    args = parser.parse_args(argv)

    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.spec:
        spec = load_model_spec_from(args.spec)
    else:
        spec = bundle["spec"]
        if isinstance(spec, dict):
            spec = ModelSpec.from_dict(spec)
    model = TransformerLM(spec)
    model.load_state_dict(bundle["model"])
    path = save_model(model, args.out, dtype=TORCH_DTYPE[args.dtype], extra_meta={"brainlab.checkpoint_step": str(bundle.get("step", -1))})
    print(f"wrote {path} params={model.num_params()} dtype={args.dtype}")
    return 0


def load_model_spec_from(path: Path) -> ModelSpec:
    import tomllib

    with path.open("rb") as fh:
        return ModelSpec.from_dict(tomllib.load(fh)["model"])


if __name__ == "__main__":
    raise SystemExit(main())
