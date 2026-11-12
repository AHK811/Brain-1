from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, width, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    x = tl.load(x_ptr + row * width + cols, mask=mask, other=float("-inf")).to(tl.float32)
    maximum = tl.max(x, axis=0)
    exponent = tl.exp(x - maximum)
    exponent = tl.where(mask, exponent, 0.0)
    total = tl.sum(exponent, axis=0)
    tl.store(y_ptr + row * width + cols, (exponent / total).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def softmax_kernel_looped(x_ptr, y_ptr, width, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * width
    maximum = float("-inf")
    for start in range(0, width, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        chunk = tl.load(x_ptr + base + cols, mask=cols < width, other=float("-inf")).to(tl.float32)
        maximum = tl.maximum(maximum, tl.max(chunk, axis=0))
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, width, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        chunk = tl.load(x_ptr + base + cols, mask=cols < width, other=float("-inf")).to(tl.float32)
        total += tl.where(cols < width, tl.exp(chunk - maximum), 0.0)
    inverse = 1.0 / tl.sum(total, axis=0)
    for start in range(0, width, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < width
        chunk = tl.load(x_ptr + base + cols, mask=mask, other=float("-inf")).to(tl.float32)
        tl.store(y_ptr + base + cols, (tl.exp(chunk - maximum) * inverse).to(y_ptr.dtype.element_ty), mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    rows, width = x.shape
    output = torch.empty_like(x)
    block = triton.next_power_of_2(width)
    if block <= 4096:
        softmax_kernel[(rows,)](x, output, width, BLOCK=block, num_warps=min(16, max(1, block // 128)))
    else:
        softmax_kernel_looped[(rows,)](x, output, width, BLOCK=1024, num_warps=8)
    return output
