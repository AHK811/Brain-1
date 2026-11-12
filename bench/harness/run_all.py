from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from bench.harness import backends as backend_module
from bench.harness import hostinfo, peak, timing
from bench.harness.backends import NativeBackend, Outcome, PythonBackend, detach_numpy
from brainlab.export.raw_io import NP_DTYPE
from brainlab.model.gpt import TransformerLM
from brainlab.spec.config import load_model_spec, repo_root
from brainlab.spec.workloads import Workload, select
from brainlab.util import configure_stdout, write_json

SCHEMA = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="brainlab-bench", description="One harness, every backend, identical inputs.")
    parser.add_argument("--ops", nargs="+", default=["rmsnorm"])
    parser.add_argument("--dtypes", nargs="+", default=["fp32", "fp16"], choices=["fp32", "fp16"])
    parser.add_argument("--keys", nargs="*", default=None, help="restrict to exact workload keys")
    parser.add_argument("--backends", nargs="*", default=None, help="filter by backend name")
    parser.add_argument("--exclude", nargs="*", default=[], help="drop backend names")
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="interleave backends round-robin for N rounds and pool their medians (median/p90/MAD across rounds)",
    )
    parser.add_argument("--threads", type=int, default=0, help="backend worker threads, 0 means all cores")
    parser.add_argument("--golden-dir", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--peak-gbs", type=float, default=None, help="override device peak bandwidth")
    parser.add_argument("--tag", default="")
    parser.add_argument("--list-backends", action="store_true")
    parser.add_argument("--fail-on-parity", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args(argv)


def load_inputs(golden_dir: Path, workload: Workload) -> dict[str, np.ndarray]:
    path = golden_dir / workload.golden_file
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run `just golden`")
    tensors = load_file(str(path), device="cpu")
    return {name: tensor.numpy() for name, tensor in tensors.items()}


def to_torch(arrays: dict[str, np.ndarray], device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for name, array in arrays.items():
        tensor = torch.from_numpy(np.array(array, copy=True))
        if tensor.dtype in (torch.float32, torch.float16) and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        converted[name] = tensor.to(device)
    return converted


def parity_metrics(workload: Workload, produced: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if produced.shape != expected.shape:
        return {
            "status": "shape_mismatch",
            "produced_shape": list(produced.shape),
            "expected_shape": list(expected.shape),
            "parity_pass": False,
        }
    actual = np.asarray(produced, dtype=np.float64)
    reference = np.asarray(expected, dtype=np.float64)
    delta = np.abs(actual - reference)
    tol = workload.tol
    denominator = float(np.sqrt((reference * reference).sum()))
    numerator = float(np.sqrt((delta * delta).sum()))
    rel_l2 = numerator / denominator if denominator > 0 else numerator
    budget = tol.atol + tol.rtol * np.abs(reference)
    worst_over = float((delta - budget).max())
    finite = bool(np.isfinite(actual).all())
    return {
        "status": "checked",
        "max_abs_err": float(delta.max()),
        "mean_abs_err": float(delta.mean()),
        "rel_l2_err": rel_l2,
        "worst_element_over_budget": worst_over,
        "atol": tol.atol,
        "rtol": tol.rtol,
        "rel_l2_limit": tol.rel_l2,
        "finite": finite,
        "parity_pass": bool(worst_over <= 0.0 and rel_l2 <= tol.rel_l2 and finite),
    }


NP_DTYPE_SUFFIX = {"fp32": "f32", "fp16": "f16"}


def expected_name(workload: Workload) -> str:
    return "logits" if workload.op == "forward" else "y"


def raw_paths(raw_dir: Path, workload: Workload) -> dict[str, str]:
    directory = raw_dir / workload.key
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} missing; run `just golden -- --raw`")
    suffix = NP_DTYPE_SUFFIX[workload.dtype]
    return {
        "raw_dir": str(directory),
        "x": str(directory / f"x.{suffix}.bin"),
        "w": str(directory / f"w.{suffix}.bin"),
        "y": str(directory / f"y.{suffix}.bin"),
    }


def context_for(
    workload: Workload,
    root: Path,
    golden_dir: Path,
    raw_dir: Path,
    out_dir: Path,
    reps: int,
    warmup: int,
    threads: int,
    timeout: int,
    model_path: str,
) -> dict[str, str]:
    dims = workload.dims
    paths = raw_paths(raw_dir, workload) if workload.op != "forward" else {"raw_dir": str(raw_dir)}
    shapes = workload.output_shapes(workload.vocab or None)
    output_shape = "x".join(str(part) for part in shapes[expected_name(workload)])
    context = {
        "key": workload.key,
        "golden_dir": str(golden_dir),
        "raw_dir": str(raw_dir),
        "reps": str(reps),
        "warmup": str(warmup),
        "threads": str(threads),
        "dtype": workload.dtype,
        "eps": repr(workload.eps),
        "rows": str(dims[0]),
        "cols": str(dims[1]) if len(dims) > 1 else "0",
        "out": str(out_dir / f"{workload.key}.out.f32.bin"),
        "output_shape": output_shape,
        "timeout": str(timeout),
        "model": model_path,
        **paths,
    }
    context.update(backend_module.default_binary_paths(root))
    return context


def model_for(device: str, root: Path, golden_dir: Path) -> tuple[TransformerLM | None, str]:
    path = golden_dir / "model.safetensors"
    if not path.is_file():
        return None, ""
    from brainlab.export.to_safetensors import load_model

    model = load_model(path, device=device, dtype=torch.float32)
    return model, str(path)


def run_one(
    workload: Workload,
    backend: PythonBackend | NativeBackend,
    arrays: dict[str, np.ndarray],
    context: dict[str, str],
    model: TransformerLM | None,
    reps: int,
    warmup: int,
    device_name: str,
    peak_value: peak.Peak,
) -> dict[str, Any]:
    name = backend.name
    row: dict[str, Any] = {
        "key": workload.key,
        "op": workload.op,
        "dtype": workload.dtype,
        "dims": list(workload.dims),
        "backend": name,
        "hardware": backend.hardware,
        "device": getattr(backend, "device", "cpu"),
        "traffic_bytes": workload.traffic_bytes,
        "flops": workload.flops,
    }
    if isinstance(backend, NativeBackend):
        outcome = backend.execute(workload, context)
    else:
        tensors = to_torch({k: v for k, v in arrays.items() if k != expected_name(workload)}, context["run_device"], TORCH[workload.dtype])
        outcome = backend.execute(workload, tensors, model, reps, warmup)

    if outcome.error:
        row.update({"status": "error", "error": outcome.error, "parity_pass": False})
        if outcome.extra.get("command"):
            row["command"] = outcome.extra["command"]
        return row

    expected = arrays.get(expected_name(workload))
    if expected is None:
        row.update({"status": "no_expected_output", "parity_pass": False})
        return row
    metrics = parity_metrics(workload, outcome.produced, expected)
    ms = outcome.milliseconds()
    row.update(metrics)
    row["status"] = "ok"
    row["native"] = outcome.native
    row["process_ms"] = outcome.process_ms
    if ms is not None:
        row["ms_median"] = ms
        row["gb_s"] = workload.traffic_bytes / (ms * 1e-3) / 1e9 if workload.traffic_bytes else None
        row["gflops"] = workload.flops / (ms * 1e-3) / 1e9 if workload.flops else None
        if peak_value.value and context["run_device"] == "cuda":
            row["pct_of_peak"] = 100.0 * row["gb_s"] / peak_value.value
    if outcome.timing is not None:
        row["timing"] = outcome.timing.as_dict()
        row["ms_min"] = outcome.timing.min_ms
        row["ms_p90"] = outcome.timing.p90_ms
        row["ms_mad_pct"] = outcome.timing.mad_pct
    else:
        for candidate in ("ms_min", "ms_p90"):
            value = backend_module.dig(outcome.native, (candidate,), ("result", candidate), ("result", "summary", candidate))
            if value is not None:
                row[candidate] = value
    return row


TORCH = {"fp32": torch.float32, "fp16": torch.float16}


def one_round(
    backend: PythonBackend | NativeBackend,
    workload: Workload,
    prepared: dict[str, object],
    arrays: dict[str, np.ndarray],
    context: dict[str, str],
    model: TransformerLM | None,
    reps: int,
    warmup: int,
) -> Outcome:
    if isinstance(backend, NativeBackend):
        return backend.execute(workload, context)
    if backend.name not in prepared:
        tensors = to_torch(
            {key: value for key, value in arrays.items() if key != expected_name(workload)},
            context["run_device"],
            TORCH[workload.dtype],
        )
        prepared[backend.name] = (
            backend.build(workload, model) if backend.needs_model else backend.build(workload)
        )
        prepared[backend.name + ":tensors"] = tensors
    call = prepared[backend.name]
    tensors = prepared[backend.name + ":tensors"]
    device = context["run_device"]
    try:
        estimate = timing.probe_cost(lambda: call(tensors), device)
        used_reps = timing.choose_reps(estimate, max(3, min(reps, 5)), reps, 400.0)
        used_warmup = timing.warmup_for(estimate, max(3, min(warmup, 5)), 120.0)
        measured = timing.time_calls(lambda: call(tensors), used_reps, used_warmup, device)
        result = call(tensors)
    except Exception as error:
        return Outcome(error=f"{type(error).__name__}: {error}")
    return Outcome(produced=detach_numpy(result), timing=measured, native={"reps_used": used_reps})


def pooled_row(
    workload: Workload,
    backend: PythonBackend | NativeBackend,
    samples: list[tuple[float, np.ndarray | None]],
    arrays: dict[str, np.ndarray],
    peak_value: peak.Peak,
    device: str,
) -> dict[str, Any]:
    produced = next((value for _, value in reversed(samples) if value is not None), None)
    expected = arrays.get(expected_name(workload))
    row: dict[str, Any] = {
        "key": workload.key,
        "op": workload.op,
        "dtype": workload.dtype,
        "dims": list(workload.dims),
        "backend": backend.name,
        "hardware": backend.hardware,
        "device": getattr(backend, "device", "cpu"),
        "traffic_bytes": workload.traffic_bytes,
        "flops": workload.flops,
        "rounds": len(samples),
    }
    if produced is None or expected is None:
        row.update({"status": "no_output", "parity_pass": False})
        return row
    row.update(parity_metrics(workload, produced, expected))
    medians = [median for median, _ in samples]
    summary = timing.summarize(medians)
    ms = summary.median_ms
    row["status"] = "ok"
    row["ms_median"] = ms
    row["ms_min"] = summary.min_ms
    row["ms_p90"] = summary.p90_ms
    row["ms_mad_pct"] = summary.mad_pct
    row["round_medians_ms"] = list(summary.samples_ms)
    row["gb_s"] = workload.traffic_bytes / (ms * 1e-3) / 1e9 if workload.traffic_bytes else None
    row["gflops"] = workload.flops / (ms * 1e-3) / 1e9 if workload.flops else None
    if peak_value.value and device == "cuda":
        row["pct_of_peak"] = 100.0 * row["gb_s"] / peak_value.value
    return row


def run_interleaved(
    workload: Workload,
    usable: list[PythonBackend | NativeBackend],
    arrays: dict[str, np.ndarray],
    context: dict[str, str],
    model: TransformerLM | None,
    reps: int,
    warmup: int,
    rounds: int,
    peak_value: peak.Peak,
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[float, np.ndarray | None]]]]:
    samples: dict[str, list[tuple[float, np.ndarray | None]]] = {backend.name: [] for backend in usable}
    prepared: dict[str, object] = {}
    failures: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for _ in range(rounds):
        for backend in usable:
            outcome = one_round(backend, workload, prepared, arrays, context, model, reps, warmup)
            if outcome.error or outcome.timing is None:
                failures[backend.name] = outcome.error or "no timing"
                continue
            samples[backend.name].append((outcome.timing.median_ms, outcome.produced))
    for backend in usable:
        if not samples[backend.name]:
            rows.append(
                {
                    "key": workload.key,
                    "op": workload.op,
                    "dtype": workload.dtype,
                    "dims": list(workload.dims),
                    "backend": backend.name,
                    "hardware": backend.hardware,
                    "device": getattr(backend, "device", "cpu"),
                    "status": "error",
                    "error": failures.get(backend.name, "no successful round"),
                    "parity_pass": False,
                }
            )
            continue
        rows.append(pooled_row(workload, backend, samples[backend.name], arrays, peak_value, context["run_device"]))
    return rows, samples


def selected_workloads(args: argparse.Namespace) -> tuple[Workload, ...]:
    if args.keys:
        from brainlab.spec.workloads import get

        return tuple(get(key) for key in args.keys)
    return select(args.ops, args.dtypes)


def shown(value: float | None, default: float) -> float:
    return default if value is None else float(value)


def report_row(row: dict[str, Any]) -> None:
    marker = "PASS" if row.get("parity_pass") else (row.get("status") or "FAIL")
    spread = row.get("ms_mad_pct")
    print(
        f"    {marker:>12s}  ms={shown(row.get('ms_median'), float('nan')):.4f}  "
        f"GB/s={shown(row.get('gb_s'), 0.0):.1f}  rel_l2={shown(row.get('rel_l2_err'), float('nan')):.3e}"
        + (f"  rounds={row['rounds']} MAD={spread:.1f}%" if spread is not None else "")
        + (f"  {row.get('error') or row.get('note') or ''}" if marker != "PASS" else "")
    )




def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else None)
    configure_stdout()
    root = repo_root()
    golden_dir = args.golden_dir or (root / "tests" / "golden")
    raw_dir = args.raw_dir or (golden_dir / "raw")
    out_dir = args.out_dir or (root / "bench" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = [*backend_module.python_backends(), *backend_module.native_backends()]
    if args.backends:
        registry = [item for item in registry if item.name in set(args.backends)]
    if args.exclude:
        registry = [item for item in registry if item.name not in set(args.exclude)]

    if args.list_backends:
        context = dict(backend_module.default_binary_paths(root))
        for backend in registry:
            ok, note = backend.available(context) if isinstance(backend, NativeBackend) else backend.available()
            print(
                f"{backend.name:22s} ops={sorted(backend.ops)!s:60s} dtypes={sorted(backend.dtypes)!s:20s} "
                f"{'available' if ok else 'unavailable'}: {note}"
            )
        return 0

    threads = args.threads or (os.cpu_count() or 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(threads)
    torch_threads = torch.get_num_threads()
    if torch_threads != threads:
        print(f"warning: torch uses {torch_threads} threads but the harness asked for {threads}; comparison is not thread-matched")
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else f"cpu:{os.uname().machine if hasattr(os, 'uname') else 'unknown'}"
    peak_value = peak.peak_for(device_name, args.peak_gbs)
    spec = load_model_spec()
    model, model_path = model_for(device, root, golden_dir)

    workloads = tuple(workload.with_model(spec.param_count, spec.vocab_size) for workload in selected_workloads(args))
    print(f"device={device} {device_name} peak={peak_value.value} GB/s ({peak_value.source}) threads={threads}")
    print(f"workloads={len(workloads)} backends={len(registry)} golden={golden_dir}")

    rows: list[dict[str, Any]] = []
    for workload in workloads:
        try:
            arrays = load_inputs(golden_dir, workload)
        except FileNotFoundError as error:
            print(f"skip {workload.key}: {error}")
            continue
        context = context_for(
            workload,
            root,
            golden_dir,
            raw_dir,
            out_dir,
            args.reps,
            args.warmup,
            threads,
            args.timeout,
            model_path,
        )
        context["run_device"] = device
        usable: list[PythonBackend | NativeBackend] = []
        for backend in registry:
            if workload.op not in backend.ops:
                continue
            if workload.dtype not in backend.dtypes:
                continue
            ok, note = backend.available(context) if isinstance(backend, NativeBackend) else backend.available()
            if not ok:
                rows.append(
                    {
                        "key": workload.key,
                        "op": workload.op,
                        "dtype": workload.dtype,
                        "dims": list(workload.dims),
                        "backend": backend.name,
                        "hardware": backend.hardware,
                        "device": getattr(backend, "device", "cpu"),
                        "status": "unavailable",
                        "note": note,
                        "parity_pass": False,
                    }
                )
                continue
            if workload.op == "forward" and isinstance(backend, PythonBackend) and backend.needs_model and model is None:
                rows.append(
                    {
                        "key": workload.key,
                        "op": workload.op,
                        "backend": backend.name,
                        "status": "unavailable",
                        "note": "tests/golden/model.safetensors missing; run `just golden -- --model`",
                        "parity_pass": False,
                    }
                )
                continue
            usable.append(backend)
        if args.rounds > 0 and usable:
            print(f"  {workload.key} :: {len(usable)} backends x {args.rounds} interleaved rounds", flush=True)
            interleaved, _ = run_interleaved(
                workload, usable, arrays, context, model, args.reps, args.warmup, args.rounds, peak_value
            )
            for row in interleaved:
                report_row(row)
            rows.extend(interleaved)
            continue
        for backend in usable:
            print(f"  {workload.key} :: {backend.name} ...", flush=True)
            row = run_one(workload, backend, arrays, context, model, args.reps, args.warmup, device_name, peak_value)
            report_row(row)
            rows.append(row)

    document = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tag": args.tag,
        "spec": spec.to_dict(),
        "peak": {"value_gbytes_per_s": peak_value.value, "source": peak_value.source, "device": device_name},
        "host": hostinfo.collect(root, device, device == "cuda"),
        "settings": {
            "reps": args.reps,
            "warmup": args.warmup,
            "rounds": args.rounds,
            "threads": threads,
            "torch_threads": torch_threads,
            "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "ops": list(args.ops),
            "dtypes": args.dtypes,
            "golden_dir": str(golden_dir),
        },
        "rows": rows,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "-".join(args.ops)
    path = out_dir / f"{stamp}_{slug}.json"
    write_json(path, document)
    failures = [row for row in rows if row.get("status") == "ok" and not row.get("parity_pass")]
    print(f"\nwrote {path}  rows={len(rows)}  parity_failures={len(failures)}")
    if failures and args.fail_on_parity:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
