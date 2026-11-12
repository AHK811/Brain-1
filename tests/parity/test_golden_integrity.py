from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from brainlab.export.make_golden import build, entropy, operand_arrays, output_name
from brainlab.kernels.torch_ops import attention_ref, matmul_ref, rmsnorm_ref, softmax_rows
from brainlab.spec.workloads import Workload, select

from tests.conftest import workload_keys


def test_golden_index_lists_every_selected_workload(golden_index: dict, golden_dir: Path) -> None:
    listed = {record["file"] for record in golden_index["workloads"]}
    for workload in select():
        if workload.op == "forward":
            continue
        assert workload.golden_file in listed, f"{workload.key} missing from index.json"
        assert (golden_dir / workload.golden_file).is_file()


def test_reference_rmsnorm_matches_float64_math(golden_dir: Path) -> None:
    for key in workload_keys("rmsnorm", "fp32"):
        from tests.conftest import get_workload

        workload = get_workload(key)
        tensors = load_file(str(golden_dir / workload.golden_file))
        x = tensors["x"].numpy().astype(np.float64)
        w = tensors["w"].numpy().astype(np.float64)
        y = tensors["y"].numpy().astype(np.float64)
        expected = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + workload.eps) * w
        max_abs = float(np.abs(expected - y).max())
        assert max_abs <= 1e-5, f"{key}: the PyTorch reference disagrees with float64 by {max_abs:.2e}"
        variance = float((y * y).mean())
        assert 0.4 < variance < 2.5, f"{key}: normalised output variance {variance} is not near 1"


def test_golden_generation_is_deterministic(tmp_path: Path, spec) -> None:
    from tests.conftest import get_workload

    workload = get_workload("rmsnorm_fp32_1x512")
    first = build(workload, 0, tmp_path / "a", spec, None, False)
    second = build(workload, 0, tmp_path / "b", spec, None, False)
    a = load_file(str(tmp_path / "a" / workload.golden_file))
    b = load_file(str(tmp_path / "b" / workload.golden_file))
    assert a.keys() == b.keys()
    for name in a:
        assert torch.equal(a[name], b[name]), f"{name} changed between runs; seeds are not stable"
    assert first["file"] == second["file"]


def test_softmax_golden_rows_sum_to_one(golden_dir: Path) -> None:
    for key in workload_keys("softmax", "fp32"):
        from tests.conftest import get_workload

        workload = get_workload(key)
        tensors = load_file(str(golden_dir / workload.golden_file))
        totals = tensors["y"].sum(dim=-1)
        assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5), f"{key}: rows do not sum to 1"


def test_attention_golden_is_causal(golden_dir: Path) -> None:
    from tests.conftest import get_workload

    workload = get_workload("attention_fp32_2x8x512x64")
    tensors = load_file(str(golden_dir / workload.golden_file))
    q, k, v = tensors["q"], tensors["k"], tensors["v"]
    reference = attention_ref(q, k, v)
    assert torch.allclose(reference, tensors["y"], atol=1e-4, rtol=1e-4)
    perturbed_k = k.clone()
    perturbed_k[:, :, -1, :] += 100.0
    later = attention_ref(q, perturbed_k, v)
    assert torch.allclose(later[:, :, 0, :], tensors["y"][:, :, 0, :], atol=1e-6), "position 0 saw a future key"


def test_matmul_golden_matches_reference(golden_dir: Path) -> None:
    for key in workload_keys("matmul", "fp32"):
        from tests.conftest import get_workload

        workload = get_workload(key)
        tensors = load_file(str(golden_dir / workload.golden_file))
        got = matmul_ref(tensors["a"], tensors["b"])
        assert torch.allclose(got, tensors["y"], atol=1e-5, rtol=1e-5)


def test_fp16_goldens_are_exactly_representable(golden_dir: Path) -> None:
    for key in workload_keys("rmsnorm", "fp16"):
        from tests.conftest import get_workload

        workload = get_workload(key)
        tensors = load_file(str(golden_dir / workload.golden_file))
        for name in ("x", "w"):
            half = tensors[name]
            assert half.dtype == torch.float16
            assert torch.equal(half.float().half(), half)


EXPECTED_DTYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}


def test_flop_formulas_count_useful_work() -> None:
    from tests.conftest import get_workload

    matmul = get_workload("matmul_fp32_1024x512x1536")
    assert matmul.flops == 2.0 * 1024 * 512 * 1536

    attention = get_workload("attention_fp32_2x8x512x64")
    dense = 4.0 * 2 * 8 * 512 * 512 * 64
    assert attention.flops == 2.0 * 2 * 8 * 512 * 513 * 64, "causal attention must count the triangle"
    half = dense / 2.0
    assert abs(attention.flops - half) / half < 2e-3, "counting masked pairs doubles attention GFLOP/s"

    logits = 2.0 * 1024 * 512 * 16384
    assert logits == 17179869184.0, "the output projection is 17.2 GFLOP at 1x1024, not 34.4"


def test_every_golden_tensor_uses_its_declared_dtype(golden_index: dict, golden_dir: Path) -> None:
    for record in golden_index["workloads"]:
        declared = record["brainlab.dtype"]
        tensors = load_file(str(golden_dir / record["file"]))
        for name, tensor in tensors.items():
            wanted = torch.uint16 if name == "ids" else EXPECTED_DTYPE[declared]
            assert tensor.dtype == wanted, (
                f"{record['brainlab.key']}: tensor {name} is {tensor.dtype} but the index declares "
                f"{declared}; a file named fp32 that stores float64 makes every downstream comparison wrong"
            )
