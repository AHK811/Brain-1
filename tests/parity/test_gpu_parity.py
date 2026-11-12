from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bench.harness import peak
from bench.harness.backends import NativeBackend, native_backends, python_backends
from bench.harness.run_all import context_for, load_inputs, run_one
from brainlab.spec.config import load_model_spec, repo_root
from brainlab.spec.workloads import select

SMALLEST = ("rmsnorm_fp32_1024x512", "softmax_fp32_2000x384", "attention_fp32_4x8x256x64", "matmul_fp32_512x1344x512")

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("no CUDA device visible; these rows can only be measured on a GPU host", allow_module_level=True)


def workload_for(key: str):
    spec = load_model_spec()
    matches = [item for item in select(dtypes=["fp32"]) if item.key == key]
    assert matches, f"no fp32 workload named {key}"
    return matches[0].with_model(spec.param_count, spec.vocab_size)


@pytest.fixture(scope="module")
def golden_root() -> Path:
    directory = repo_root() / "tests" / "golden"
    if not (directory / "index.json").is_file():
        pytest.skip(f"{directory / 'index.json'} missing; run `just golden -- --model --raw`")
    return directory


def test_gpu_backends_are_registered() -> None:
    names = {backend.name for backend in [*python_backends(), *native_backends()]}
    assert "triton_rmsnorm" in names and "cuda_ceiling" in names


@pytest.mark.parametrize("key", SMALLEST)
def test_gpu_backends_match_the_reference(key: str, golden_root: Path, tmp_path: Path) -> None:
    workload = workload_for(key)
    arrays = load_inputs(golden_root, workload)
    context = context_for(workload, repo_root(), golden_root, golden_root / "raw", tmp_path, 30, 6, 1, 900, "")
    context["run_device"] = "cuda"
    name = torch.cuda.get_device_name(0)
    ceiling = peak.peak_for(name, None)
    rows = []
    for backend in [*python_backends(), *native_backends()]:
        if workload.op not in backend.ops:
            continue
        if isinstance(backend, NativeBackend):
            if backend.device != "cuda":
                continue
            ok, note = backend.available(context)
            if not ok:
                rows.append((backend.name, "unavailable", note))
                continue
            rows.append(run_one(workload, backend, arrays, context, None, 30, 6, name, ceiling))
        elif backend.device == "cuda":
            ok, note = backend.available()
            if not ok:
                rows.append((backend.name, "unavailable", note))
                continue
            rows.append(run_one(workload, backend, arrays, context, None, 30, 6, name, ceiling))
    executed = [row for row in rows if isinstance(row, dict) and row.get("status") == "ok"]
    if not executed:
        pytest.skip(f"no gpu backend available for {key}: {rows}")
    for row in executed:
        assert row["parity_pass"], f"{key} on {row['backend']}: {row}"
    if ceiling.value is None:
        pytest.skip(f"{name} has no published peak bandwidth; pass --peak-gbs")
    for row in executed:
        assert 0.0 < row["pct_of_peak"] <= 130.0, f"{row['backend']} claims {row['pct_of_peak']}% of {ceiling.value} GB/s"
