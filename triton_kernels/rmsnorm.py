from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    width,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    x = tl.load(x_ptr + row * width + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(x * x, axis=0) / width + eps)
    y = (x * inverse * w).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row * width + cols, y, mask=mask)


@triton.jit
def rmsnorm_kernel_looped(
    x_ptr,
    w_ptr,
    y_ptr,
    width,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * width
    sum_sq = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, width, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        chunk = tl.load(x_ptr + base + cols, mask=cols < width, other=0.0).to(tl.float32)
        sum_sq += chunk * chunk
    inverse = tl.rsqrt(tl.sum(sum_sq, axis=0) / width + eps)
    for start in range(0, width, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < width
        chunk = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + base + cols, (chunk * inverse * w).to(y_ptr.dtype.element_ty), mask=mask)


def run(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rows, width = x.shape
    output = torch.empty_like(x)
    block = triton.next_power_of_2(width)
    if block <= 4096:
        warps = min(16, max(1, block // 128))
        rmsnorm_kernel[(rows,)](x, weight, output, width, eps, BLOCK=block, num_warps=warps)
    else:
        rmsnorm_kernel_looped[(rows,)](x, weight, output, width, eps, BLOCK=2048, num_warps=8)
    return output


def supports(width: int) -> bool:
    return width <= (1 << 20)
