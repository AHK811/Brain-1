from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n
    a_cursor = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_cursor = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, k, BLOCK_K):
        remaining = k - start_k
        a = tl.load(a_cursor, mask=mask_m[:, None] & (offs_k[None, :] < remaining), other=0.0)
        b = tl.load(b_cursor, mask=(offs_k[:, None] < remaining) & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_cursor += BLOCK_K * stride_ak
        b_cursor += BLOCK_K * stride_bk
    c = acc.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c, mask=mask_m[:, None] & mask_n[None, :])


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"inner dimensions differ {k} vs {k2}")
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)
    block_m = 64 if m >= 64 else triton.next_power_of_2(max(m, 16))
    block_n = 64 if n >= 64 else triton.next_power_of_2(max(n, 16))
    block_k = 32
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    matmul_kernel[grid](
        a,
        b,
        output,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=4,
        num_stages=3,
    )
    return output
