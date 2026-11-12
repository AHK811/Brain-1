from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bench.harness import timing
from brainlab.export.raw_io import read_raw
from brainlab.kernels.torch_ops import attention_ref, matmul_ref, rmsnorm_ref, softmax_rows
from brainlab.spec.workloads import Workload

TORCH_DTYPE = {"fp32": torch.float32, "fp16": torch.float16}
CPU_LABEL = f"cpu x {os.cpu_count() or 1} threads"


def dig(blob: Any, *paths: Sequence[str]) -> float | None:
    for path in paths:
        cursor = blob
        for part in path:
            if not isinstance(cursor, dict) or part not in cursor:
                cursor = None
                break
            cursor = cursor[part]
        if isinstance(cursor, (int, float)) and not isinstance(cursor, bool):
            return float(cursor)
    return None


@dataclass
class Outcome:
    produced: np.ndarray | None = None
    native: dict[str, Any] = field(default_factory=dict)
    timing: timing.Timing | None = None
    process_ms: float | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.produced is not None

    def milliseconds(self) -> float | None:
        if self.timing is not None:
            return self.timing.median_ms
        return dig(
            self.native,
            ("ms_median",),
            ("result", "ms_median"),
            ("result", "summary", "ms_median"),
            ("timing", "ms_median"),
        )


def detach_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", dtype=torch.float32).numpy()
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    raise TypeError(f"backend returned {type(value).__name__}, expected a tensor")


@dataclass(frozen=True)
class PythonBackend:
    name: str
    ops: frozenset[str]
    hardware: str
    device: str
    build: Callable[..., Callable[[dict[str, torch.Tensor]], Any]]
    availability: tuple[bool, str] = (True, "importable")
    needs_model: bool = False
    probe: Callable[[], tuple[bool, str]] | None = None
    dtypes: frozenset[str] = frozenset({"fp32", "fp16"})

    def available(self) -> tuple[bool, str]:
        if self.device == "cuda" and not torch.cuda.is_available():
            return False, "no CUDA device visible"
        if self.probe is not None:
            return self.probe()
        return self.availability

    def execute(
        self,
        workload: Workload,
        tensors: dict[str, torch.Tensor],
        model: torch.nn.Module | None,
        reps: int,
        warmup: int,
    ) -> Outcome:
        call = self.build(workload, model) if self.needs_model else self.build(workload)
        device = "cuda" if next(iter(tensors.values())).is_cuda else "cpu"
        try:
            probe = call(tensors)
            del probe
            estimate = timing.probe_cost(lambda: call(tensors), device)
            used_reps = timing.choose_reps(estimate, max(3, min(reps, 5)), reps, 400.0)
            used_warmup = timing.warmup_for(estimate, max(3, min(warmup, 5)), 120.0)
            measured = timing.time_calls(lambda: call(tensors), used_reps, used_warmup, device)
            result = call(tensors)
        except Exception as error:
            return Outcome(error=f"{type(error).__name__}: {error}")
        return Outcome(
            produced=detach_numpy(result),
            timing=measured,
            native={"reps_used": measured.reps, "warmup_used": used_warmup},
            extra={"probe_ms": estimate},
        )


@dataclass(frozen=True)
class NativeBackend:
    name: str
    ops: frozenset[str]
    hardware: str
    device: str
    template: tuple[str, ...]
    requires: tuple[str, ...]
    dtypes: frozenset[str] = frozenset({"fp32", "fp16"})

    def available(self, context: dict[str, str]) -> tuple[bool, str]:
        for key in self.requires:
            path = context.get(key, "")
            if not path:
                return False, f"{key} is not configured"
            if not Path(path).is_file():
                return False, f"{key} path {path} does not exist"
        return True, " ".join(context.get(key, "") for key in self.requires)

    def command(self, context: dict[str, str]) -> list[str]:
        return [part.format(**context) for part in self.template]

    def execute(self, workload: Workload, context: dict[str, str]) -> Outcome:
        command = self.command(context)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=float(context.get("timeout", "900")),
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return Outcome(error=f"launch failed: {error!r}", extra={"command": command})
        process_ms = (time.perf_counter() - started) * 1000.0
        if completed.returncode != 0:
            return Outcome(
                error=f"exit {completed.returncode}: {completed.stderr.strip()[:400]}",
                process_ms=process_ms,
                extra={"command": command, "stdout": completed.stdout[:400]},
            )
        payload = parse_json_blob(completed.stdout)
        if payload is None:
            return Outcome(error=f"no JSON on stdout: {completed.stdout[:200]!r}", process_ms=process_ms)
        produced: np.ndarray | None = None
        out_path = context.get("out", "")
        if out_path and Path(out_path).is_file():
            dims = tuple(int(part) for part in context.get("output_shape", "").split("x") if part)
            produced = (
                read_raw(Path(out_path), "fp32", dims)
                if dims
                else np.fromfile(out_path, dtype=np.float32)
            )
        return Outcome(produced=produced, native=payload, process_ms=process_ms, extra={"command": command})


def parse_json_blob(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        try:
            parsed = json.loads(stripped[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def eager_call(workload: Workload) -> Callable[[dict[str, torch.Tensor]], Any]:
    eps = workload.eps
    if workload.op == "rmsnorm":
        return lambda tensors: rmsnorm_ref(tensors["x"], tensors["w"], eps)
    if workload.op == "softmax":
        return lambda tensors: softmax_rows(tensors["x"])
    if workload.op == "attention":
        return lambda tensors: attention_ref(tensors["q"], tensors["k"], tensors["v"])
    if workload.op == "matmul":
        return lambda tensors: matmul_ref(tensors["a"], tensors["b"])
    raise ValueError(f"no eager implementation for {workload.op}")


def compiled_call(workload: Workload) -> Callable[[dict[str, torch.Tensor]], Any]:
    import brainlab.kernels.torch_ops as reference

    compiled = {
        "rmsnorm": torch.compile(reference.rmsnorm_ref),
        "softmax": torch.compile(reference.softmax_rows),
        "matmul": torch.compile(reference.matmul_ref),
        "attention": torch.compile(reference.attention_ref),
    }
    function = compiled[workload.op]
    eps = workload.eps
    if workload.op == "rmsnorm":
        return lambda tensors: function(tensors["x"], tensors["w"], eps)
    if workload.op == "attention":
        return lambda tensors: function(tensors["q"], tensors["k"], tensors["v"])
    if workload.op == "matmul":
        return lambda tensors: function(tensors["a"], tensors["b"])
    return lambda tensors: function(tensors["x"])


def forward_call(workload: Workload, model: torch.nn.Module | None) -> Callable[[dict[str, torch.Tensor]], Any]:
    if model is None:
        raise RuntimeError("the forward backend needs a loaded model")

    def call(tensors: dict[str, torch.Tensor]) -> Any:
        with torch.no_grad():
            return model(tensors["ids"].long())

    return call


def triton_call(workload: Workload) -> Callable[[dict[str, torch.Tensor]], Any]:
    import triton_kernels

    kernel = triton_kernels.KERNELS[workload.op]
    if workload.op == "rmsnorm":
        return lambda tensors: kernel(tensors["x"], tensors["w"], workload.eps)
    if workload.op == "matmul":
        return lambda tensors: kernel(tensors["a"], tensors["b"])
    if workload.op == "attention":
        return lambda tensors: kernel(tensors["q"], tensors["k"], tensors["v"])
    return lambda tensors: kernel(tensors["x"])


def triton_probe() -> tuple[bool, str]:
    try:
        import triton_kernels
    except Exception as error:
        return False, f"import failed: {type(error).__name__}: {error}"
    return triton_kernels.available()


_COMPILED: tuple[bool, str] | None = None


def compile_probe() -> tuple[bool, str]:
    global _COMPILED
    if _COMPILED is None:
        if not hasattr(torch, "compile"):
            _COMPILED = (False, f"torch {torch.__version__} has no compile()")
        else:
            try:
                import time

                started = time.perf_counter()
                doubled = torch.compile(lambda value: value.relu())
                probe = torch.randn(16, 16)
                doubled(probe)
                _COMPILED = (True, f"torch {torch.__version__} inductor ready in {time.perf_counter() - started:.1f}s")
            except Exception as error:
                _COMPILED = (False, f"{type(error).__name__}: {str(error)[:160]}")
    return _COMPILED


def python_backends() -> list[PythonBackend]:
    kernel_ops = frozenset({"rmsnorm", "softmax", "attention", "matmul"})
    backends = [
        PythonBackend("pytorch_eager", kernel_ops, CPU_LABEL, "cpu", eager_call),
        PythonBackend("pytorch_compile", kernel_ops, CPU_LABEL, "cpu", compiled_call, probe=compile_probe),
        PythonBackend("pytorch_eager_forward", frozenset({"forward"}), CPU_LABEL, "cpu", forward_call, needs_model=True),
    ]
    ok, note = triton_probe()
    cuda_label = "cuda via triton"
    for op in sorted(kernel_ops):
        backends.append(
            PythonBackend(
                name=f"triton_{op}",
                ops=frozenset({op}),
                hardware=cuda_label,
                device="cuda",
                build=triton_call,
                availability=(ok, note),
            )
        )
    return backends


def native_backends() -> list[NativeBackend]:
    return [
        NativeBackend(
            name="rust_cpu",
            ops=frozenset({"rmsnorm", "softmax", "attention", "matmul"}),
            hardware=CPU_LABEL,
            device="cpu",
            dtypes=frozenset({"fp32"}),
            template=(
                "{rust}",
                "bench",
                "--golden-dir",
                "{golden_dir}",
                "--key",
                "{key}",
                "--reps",
                "{reps}",
                "--warmup",
                "{warmup}",
                "--threads",
                "{threads}",
                "--out",
                "{out}",
            ),
            requires=("rust",),
        ),
        NativeBackend(
            name="rust_cpu",
            ops=frozenset({"forward"}),
            hardware=CPU_LABEL,
            device="cpu",
            template=(
                "{rust}",
                "forward",
                "--model",
                "{model}",
                "--golden-dir",
                "{golden_dir}",
                "--key",
                "{key}",
                "--reps",
                "{reps}",
                "--warmup",
                "{warmup}",
                "--out",
                "{out}",
            ),
            requires=("rust", "model"),
        ),
        NativeBackend(
            name="cuda_ceiling",
            ops=frozenset({"rmsnorm"}),
            hardware="cuda nvcc hand-written",
            device="cuda",
            template=(
                "{cuda}",
                "--key",
                "{key}",
                "--x",
                "{x}",
                "--w",
                "{w}",
                "--y",
                "{y}",
                "--rows",
                "{rows}",
                "--cols",
                "{cols}",
                "--eps",
                "{eps}",
                "--dtype",
                "{dtype}",
                "--reps",
                "{reps}",
                "--warmup",
                "{warmup}",
                "--out",
                "{out}",
            ),
            requires=("cuda",),
        ),
        NativeBackend(
            name="mojo_cpu",
            ops=frozenset({"rmsnorm"}),
            hardware=CPU_LABEL,
            device="cpu",
            dtypes=frozenset({"fp32"}),
            template=(
                "{mojo}",
                "--key",
                "{key}",
                "--x",
                "{x}",
                "--w",
                "{w}",
                "--y",
                "{y}",
                "--rows",
                "{rows}",
                "--cols",
                "{cols}",
                "--eps",
                "{eps}",
                "--reps",
                "{reps}",
                "--warmup",
                "{warmup}",
                "--out",
                "{out}",
            ),
            requires=("mojo",),
        ),
    ]


def default_binary_paths(root: Path) -> dict[str, str]:
    exe = ".exe" if os.name == "nt" else ""
    candidates = {
        "rust": [
            root / "rust" / "target" / "release" / f"brainlab{exe}",
            root / "target" / "release" / f"brainlab{exe}",
        ],
        "cuda": [root / "cuda" / "build" / f"rmsnorm{exe}"],
        "mojo": [root / "build" / f"mojo-driver{exe}"],
    }
    resolved: dict[str, str] = {}
    for key, paths in candidates.items():
        hit = next((path for path in paths if path.is_file()), None)
        override = os.environ.get(f"BRAINLAB_{key.upper()}_CMD")
        resolved[key] = override if override else (str(hit) if hit else "")
    return resolved
