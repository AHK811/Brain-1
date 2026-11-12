from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from brainlab.model.layers import rmsnorm


def rmsnorm_ref(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    return rmsnorm(x, weight, eps)


def softmax_rows(x: Tensor) -> Tensor:
    x32 = x.float()
    shifted = x32 - x32.max(dim=-1, keepdim=True).values
    exp = shifted.exp()
    return (exp / exp.sum(dim=-1, keepdim=True)).to(dtype=x.dtype)


def attention_ref(q: Tensor, k: Tensor, v: Tensor, causal: bool = True) -> Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        mask = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = softmax_rows_f32(scores)
    return torch.matmul(probs, v.float()).to(dtype=q.dtype)


def softmax_rows_f32(x: Tensor) -> Tensor:
    shifted = x - x.max(dim=-1, keepdim=True).values
    exp = shifted.exp()
    return exp / exp.sum(dim=-1, keepdim=True)


def matmul_ref(a: Tensor, b: Tensor) -> Tensor:
    return torch.matmul(a.float(), b.float()).to(dtype=a.dtype)


def swiglu_ref(x: Tensor, gate: Tensor, up: Tensor, down: Tensor) -> Tensor:
    x32 = x.float()
    hidden = F.silu(torch.matmul(x32, gate.float().t())) * torch.matmul(x32, up.float().t())
    return torch.matmul(hidden, down.float().t()).to(dtype=x.dtype)
