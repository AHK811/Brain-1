from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    seq_len,
    head_dim: tl.constexpr,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    plane = tl.program_id(1)
    base = plane * seq_len * head_dim
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    dmask = offs_d < head_dim
    row_mask = offs_m < seq_len
    q = tl.load(q_ptr + base + offs_m[:, None] * head_dim + offs_d[None, :], mask=row_mask[:, None] & dmask[None, :], other=0.0)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    bound = tl.minimum((pid_m + 1) * BLOCK_M, seq_len)
    for start_n in range(0, bound, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        k = tl.load(k_ptr + base + offs_n[:, None] * head_dim + offs_d[None, :], mask=n_mask[:, None] & dmask[None, :], other=0.0)
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        causal = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal & n_mask[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        probability = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(probability, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptr + base + offs_n[:, None] * head_dim + offs_d[None, :], mask=n_mask[:, None] & dmask[None, :], other=0.0)
        acc += tl.dot(probability.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(o_ptr + base + offs_m[:, None] * head_dim + offs_d[None, :], acc.to(o_ptr.dtype.element_ty), mask=row_mask[:, None] & dmask[None, :])


def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch, heads, seq, head_dim = q.shape
    planes = batch * heads
    reshape = lambda tensor: tensor.reshape(planes, seq, head_dim).contiguous()
    query, key, value = reshape(q), reshape(k), reshape(v)
    output = torch.empty_like(query)
    scale = 1.0 / math.sqrt(head_dim)
    block_d = triton.next_power_of_2(head_dim)
    block_m = 64 if seq >= 64 else triton.next_power_of_2(seq)
    block_n = 64
    grid = (triton.cdiv(seq, block_m), planes)
    attention_kernel[grid](
        query,
        key,
        value,
        output,
        seq,
        head_dim=head_dim,
        sm_scale=scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return output.reshape(batch, heads, seq, head_dim)
