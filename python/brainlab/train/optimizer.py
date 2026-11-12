from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from brainlab.model.gpt import TransformerLM

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def zeropower_newton_schulz(matrix: Tensor, steps: int = 5) -> Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"newton-schulz needs a matrix, got ndim {matrix.ndim}")
    a, b, c = NS_COEFFICIENTS
    x = matrix.float()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.mT
    return x


class Muon(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.1,
        nesterov: bool = True,
    ) -> None:
        if lr <= 0 or ns_steps < 1:
            raise ValueError("lr must be positive and ns_steps at least 1")
        super().__init__(
            params,
            defaults={"lr": lr, "momentum": momentum, "ns_steps": ns_steps, "weight_decay": weight_decay, "nesterov": nesterov},
        )
        self.nonfinite_updates = torch.zeros((), dtype=torch.int32)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.ndim != 2:
                    raise ValueError(f"Muon only handles matrices, got ndim {grad.ndim}")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buffer = state["momentum_buffer"]
                buffer.mul_(group["momentum"]).add_(grad)
                direction = torch.add(grad, buffer, alpha=group["momentum"]) if group["nesterov"] else buffer
                update = zeropower_newton_schulz(direction, group["ns_steps"])
                finite = torch.isfinite(update).all()
                if self.nonfinite_updates.device != p.device:
                    self.nonfinite_updates = self.nonfinite_updates.to(p.device)
                self.nonfinite_updates += (~finite).to(torch.int32)
                buffer.copy_(torch.where(finite, buffer, torch.zeros_like(buffer)))
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                lr = group["lr"]
                one = torch.ones((), device=p.device, dtype=p.dtype)
                p.mul_(torch.where(finite, one - lr * group["weight_decay"], one))
                p.add_(torch.where(finite, update, torch.zeros_like(update)).reshape(p.shape), alpha=-lr * scale)
        return loss

    def skipped_updates(self) -> int:
        return int(self.nonfinite_updates)


def split_params(model: TransformerLM) -> dict[str, list[tuple[str, Tensor]]]:
    groups: dict[str, list[tuple[str, Tensor]]] = {"matrix": [], "embedding": [], "norm": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and name.startswith("blocks."):
            groups["matrix"].append((name, param))
        elif param.ndim >= 2:
            groups["embedding"].append((name, param))
        elif param.ndim == 1:
            groups["norm"].append((name, param))
        else:
            raise ValueError(f"{name}: scalar parameters are not supported by this spec")
    assigned = [id(param) for bucket in groups.values() for _, param in bucket]
    trainable = sum(1 for param in model.parameters() if param.requires_grad)
    if len(assigned) != trainable or len(set(assigned)) != trainable:
        raise ValueError(
            f"the parameter split covers {len(set(assigned))} distinct of {trainable} trainable parameters"
        )
    return groups


def build_optimizers(
    model: TransformerLM,
    mode: str = "muon",
    muon: dict[str, Any] | None = None,
    adamw: dict[str, Any] | None = None,
) -> list[Optimizer]:
    muon_cfg: dict[str, Any] = {"lr": 0.02, "momentum": 0.95, "ns_steps": 5, "weight_decay": 0.1}
    adam_cfg: dict[str, Any] = {"lr": 3e-4, "betas": (0.9, 0.95), "weight_decay": 0.1, "eps": 1e-8}
    muon_cfg.update(muon or {})
    adam_cfg.update(adamw or {})
    groups = split_params(model)
    decayed = [p for _, p in groups["matrix"] + groups["embedding"]]
    nodecay = [p for _, p in groups["norm"]]

    if mode == "muon":
        return [
            Muon(
                [p for _, p in groups["matrix"]],
                lr=muon_cfg["lr"],
                momentum=muon_cfg["momentum"],
                ns_steps=muon_cfg["ns_steps"],
                weight_decay=muon_cfg["weight_decay"],
            ),
            AdamW(
                [
                    {"params": [p for _, p in groups["embedding"]], "weight_decay": adam_cfg["weight_decay"]},
                    {"params": nodecay, "weight_decay": 0.0},
                ],
                lr=adam_cfg["lr"],
                betas=tuple(adam_cfg["betas"]),
                eps=adam_cfg["eps"],
            ),
        ]
    if mode == "adamw":
        return [
            AdamW(
                [
                    {"params": decayed, "weight_decay": adam_cfg["weight_decay"]},
                    {"params": nodecay, "weight_decay": 0.0},
                ],
                lr=adam_cfg["lr"],
                betas=tuple(adam_cfg["betas"]),
                eps=adam_cfg["eps"],
            )
        ]
    raise ValueError(f"unknown optimizer mode {mode!r}")


def optimizers_state(optims: Sequence[Optimizer]) -> list[dict[str, Any]]:
    return [optim.state_dict() for optim in optims]


def load_optimizers_state(optims: Sequence[Optimizer], states: Sequence[dict[str, Any]]) -> None:
    if len(optims) != len(states):
        raise ValueError(f"{len(optims)} optimizers but {len(states)} saved states")
    for optim, state in zip(optims, states, strict=True):
        optim.load_state_dict(state)


def peak_lrs(optims: Sequence[Optimizer]) -> list[list[float]]:
    return [[float(group["lr"]) for group in optim.param_groups] for optim in optims]


def scale_lr(optims: Sequence[Optimizer], factor: float, peaks: Sequence[Sequence[float]]) -> dict[str, float]:
    if factor < 0:
        raise ValueError("lr factor must be non-negative")
    if len(peaks) != len(optims):
        raise ValueError(f"{len(peaks)} peak schedules for {len(optims)} optimizers")
    applied: dict[str, float] = {}
    for index, (optim, group_peaks) in enumerate(zip(optims, peaks, strict=True)):
        if len(group_peaks) != len(optim.param_groups):
            raise ValueError(f"optimizer {index} gained or lost a parameter group")
        for group, peak in zip(optim.param_groups, group_peaks, strict=True):
            group["lr"] = peak * factor
            applied[f"optim{index}"] = group["lr"]
    return applied
