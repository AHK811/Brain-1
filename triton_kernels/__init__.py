from __future__ import annotations

from typing import Callable

import torch

from triton_kernels import attention, matmul, rmsnorm, softmax

KERNELS: dict[str, Callable[..., torch.Tensor]] = {
    "rmsnorm": rmsnorm.run,
    "softmax": softmax.run,
    "attention": attention.run,
    "matmul": matmul.run,
}


def available() -> tuple[bool, str]:
    try:
        import triton
    except Exception as error:
        return False, f"triton import failed: {error!r}"
    if not torch.cuda.is_available():
        return False, "no CUDA device visible to torch"
    try:
        compute = torch.cuda.get_device_capability()
    except Exception as error:
        return False, f"device capability query failed: {error!r}"
    if compute[0] < 7:
        return False, f"sm_{compute[0]}{compute[1]} is below the supported minimum sm_70"
    return True, f"triton {triton.__version__} on sm_{compute[0]}{compute[1]}"


def names() -> tuple[str, ...]:
    return tuple(KERNELS)
