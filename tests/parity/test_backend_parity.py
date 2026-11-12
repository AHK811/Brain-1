from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bench.harness import peak
from bench.harness.backends import NativeBackend, default_binary_paths, native_backends, python_backends
from bench.harness.run_all import context_for, load_inputs, model_for, parity_metrics, run_one
from brainlab.spec.config import load_model_spec, repo_root
from brainlab.spec.workloads import Workload, select

SMALLEST = (
    "rmsnorm_fp32_1x512",
    "softmax_fp32_2000x384",
    "matmul_fp32_512x1344x512",
    "attention_fp32_4x8x256x64",
)
CEILING = peak.Peak(None, "cpu rows are not compared against a GPU peak", "cpu")


def workload_for(key: str) -> Workload:
    spec = load_model_spec()
    matches = [item for item in select(dtypes=["fp32"]) if item.key == key]
    if not matches:
        pytest.fail(f"no fp32 workload named {key}; the workload table and this test have drifted")
    return matches[0].with_model(spec.param_count, spec.vocab_size)


@pytest.fixture(scope="module")
def golden_root(golden_dir: Path) -> Path:
    from tests.conftest import missing_goldens

    missing = missing_goldens(golden_dir, include_raw=True)
    if missing:
        pytest.skip(
            f"{len(missing)} golden files are missing, for example {missing[0]}; "
            "run `just golden -- --model --raw`"
        )
    return golden_dir


def harness_row(backend_name: str, workload: Workload, golden_root: Path, tmp_path: Path) -> dict[str, Any]:
    arrays = load_inputs(golden_root, workload)
    context = context_for(workload, repo_root(), golden_root, golden_root / "raw", tmp_path, 5, 2, 1, 600, "")
    context["run_device"] = "cpu"
    model, _ = model_for("cpu", repo_root(), golden_root)
    for backend in [*python_backends(), *native_backends()]:
        if backend.name != backend_name or workload.op not in backend.ops:
            continue
        ok, note = backend.available(context) if isinstance(backend, NativeBackend) else backend.available()
        if not ok:
            return {"status": "unavailable", "note": note}
        return run_one(workload, backend, arrays, context, model, 5, 2, "cpu", CEILING)
    return {"status": "not_registered", "note": f"no backend named {backend_name} claims op {workload.op}"}


@pytest.mark.parametrize("key", SMALLEST)
def test_pytorch_eager_matches_the_golden_reference(key: str, golden_root: Path, tmp_path: Path) -> None:
    row = harness_row("pytorch_eager", workload_for(key), golden_root, tmp_path)
    assert row.get("parity_pass"), f"{key}: {row}"
    assert row["ms_median"] > 0.0, f"{key}: a zero or negative time is not a measurement"


@pytest.mark.parametrize("key", SMALLEST)
def test_rust_cpu_matches_the_golden_reference(key: str, golden_root: Path, tmp_path: Path) -> None:
    binary = default_binary_paths(repo_root())["rust"]
    row = harness_row("rust_cpu", workload_for(key), golden_root, tmp_path)
    if row.get("status") == "unavailable":
        pytest.skip(f"rust_cpu not runnable from {binary or 'rust/target/release/brainlab'}: {row['note']}")
    assert row.get("parity_pass"), f"{key}: {row}"


def test_rust_rows_carry_their_hardware_label(golden_root: Path, tmp_path: Path) -> None:
    workload = workload_for("rmsnorm_fp32_1024x512")
    context = context_for(workload, repo_root(), golden_root, golden_root / "raw", tmp_path, 5, 2, 1, 600, "")
    context["run_device"] = "cpu"
    backend = next(item for item in native_backends() if item.name == "rust_cpu" and workload.op in item.ops)
    if not backend.available(context)[0]:
        pytest.skip("rust_cpu binary unavailable")
    row = run_one(workload, backend, load_inputs(golden_root, workload), context, None, 5, 2, "cpu", CEILING)
    assert row["status"] == "ok"
    assert "cpu" in row["hardware"].lower() or "core" in row["hardware"].lower(), row["hardware"]
    assert row["traffic_bytes"] > 0
    assert row["gb_s"] is not None and row["gb_s"] > 0
    assert row["ms_min"] <= row["ms_median"] <= row["ms_p90"] or row["ms_min"] <= row["ms_median"]


def test_a_missing_binary_is_reported_as_unavailable_not_as_zero_time(tmp_path: Path) -> None:
    workload = workload_for("rmsnorm_fp32_1x512")
    root = repo_root()
    context = context_for(workload, root, root / "tests" / "golden", root / "tests" / "golden" / "raw", tmp_path, 3, 1, 1, 60, "")
    context["rust"] = str(tmp_path / "brainlab-that-does-not-exist")
    backend = next(item for item in native_backends() if item.name == "rust_cpu" and workload.op in item.ops)
    ok, note = backend.available(context)
    assert not ok
    assert "does not exist" in note
    outcome = backend.execute(workload, context)
    assert outcome.error and outcome.produced is None
    assert outcome.milliseconds() is None


def test_parity_verdict_belongs_to_the_harness_not_the_backend(golden_root: Path) -> None:
    workload = workload_for("rmsnorm_fp32_1x512")
    reference = np.asarray(load_inputs(golden_root, workload)["y"], dtype=np.float32)
    exact = parity_metrics(workload, reference.copy(), reference)
    assert exact["parity_pass"] and exact["worst_element_over_budget"] <= 0.0
    blown = parity_metrics(workload, np.full_like(reference, 10.0), reference)
    assert not blown["parity_pass"] and blown["worst_element_over_budget"] > 0.0
    wrong_shape = parity_metrics(workload, reference.reshape(-1, 1), reference)
    assert wrong_shape["status"] == "shape_mismatch" and not wrong_shape["parity_pass"]
    nan = parity_metrics(workload, np.full_like(reference, np.nan), reference)
    assert not nan["parity_pass"] and not nan["finite"]
    tolerance = workload.tol
    just_under = reference + np.full_like(reference, tolerance.atol * 0.5)
    assert parity_metrics(workload, just_under, reference)["parity_pass"]
    just_over = reference + np.full_like(reference, tolerance.atol * 2.0 + 1.0)
    assert not parity_metrics(workload, just_over, reference)["parity_pass"]
