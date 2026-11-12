from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.optim.optimizer import Optimizer

from brainlab.model.gpt import TransformerLM
from brainlab.spec.config import ModelSpec

CHECKPOINT_FORMAT = 1


def save_checkpoint(
    path: Path,
    model: TransformerLM,
    step: int,
    optims: Sequence[Optimizer] = (),
    scaler: Any | None = None,
    cfg: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    include_optimizer: bool = True,
) -> Path:
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "spec": model.spec.to_dict(),
        "cfg": dict(cfg or {}),
        "metrics": dict(metrics or {}),
    }
    if include_optimizer:
        payload["optimizers"] = [optim.state_dict() for optim in optims]
        payload["scaler"] = scaler.state_dict() if scaler is not None else None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_payload(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    format_version = int(payload.get("format", 0))
    if format_version != CHECKPOINT_FORMAT:
        raise ValueError(f"checkpoint format {format_version} != {CHECKPOINT_FORMAT}")
    if "model" not in payload or "spec" not in payload:
        raise ValueError("checkpoint is missing model or spec")
    return payload


def load_checkpoint(
    path: Path,
    model: TransformerLM | None = None,
    optims: Sequence[Optimizer] = (),
    scaler: Any | None = None,
    map_location: str = "cpu",
    strict_spec: bool = True,
) -> tuple[TransformerLM, dict[str, Any]]:
    payload = load_payload(path, map_location)
    spec = ModelSpec.from_dict(payload["spec"])
    spec.validate()
    if model is None:
        model = TransformerLM(spec)
    elif strict_spec and model.spec.to_dict() != spec.to_dict():
        raise ValueError("checkpoint spec differs from the model it is being loaded into")
    model.load_state_dict(payload["model"], strict=True)
    if optims and "optimizers" in payload and payload["optimizers"]:
        saved = payload["optimizers"]
        if len(saved) != len(optims):
            raise ValueError(f"checkpoint has {len(saved)} optimizer states, {len(optims)} were requested")
        for optim, state_dict in zip(optims, saved, strict=True):
            optim.load_state_dict(state_dict)
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    return model, {"step": int(payload["step"]), "cfg": payload.get("cfg", {}), "metrics": payload.get("metrics", {}), "spec": spec}


def rotate(directory: Path, keep: int) -> list[Path]:
    if keep < 1:
        raise ValueError("keep must be at least 1")
    checkpoints = sorted(directory.glob("step-*.pt"), key=lambda p: _step_of(p))
    removed: list[Path] = []
    while len(checkpoints) - len(removed) > keep:
        oldest = checkpoints[0]
        oldest.unlink()
        removed.append(oldest)
        checkpoints = checkpoints[1:]
    return removed


def _step_of(path: Path) -> int:
    stem = path.stem
    return int(stem.split("-")[1]) if "-" in stem else -1


def latest_checkpoint(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    candidates = [p for p in directory.glob("step-*.pt") if _step_of(p) >= 0]
    return max(candidates, key=_step_of) if candidates else None
