#!/usr/bin/env bash
set -euo pipefail

ROOT="${BRAINLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ -n "${BRAINLAB_PYTHON:-}" ]; then
  PYTHON="$BRAINLAB_PYTHON"
else
  case "${OSTYPE:-}" in
    msys*|cygwin*|win32*) PYTHON=python; VENV_PYTHON="$ROOT/.venv/Scripts/python.exe" ;;
    *) PYTHON=python3; VENV_PYTHON="$ROOT/.venv/bin/python" ;;
  esac
  if [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
  fi
fi
OUT="$ROOT/bench/results/gpu_check.json"
mkdir -p "$ROOT/bench/results"
rm -f "$OUT"

"$PYTHON" - "$OUT" <<'PY'
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def command_text(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return ""


def first_line(text, default="none"):
    return text.splitlines()[0].strip() if text else default


report = {
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    "device_count": 0,
    "name": "none",
    "compute_capability": "0.0",
    "memory_mib": 0,
    "multiprocessors": 0,
    "driver": first_line(command_text([
        "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
    ])),
    "power_limit_w": "n/a",
    "fp16": False,
    "bf16": False,
    "bf16_supported": False,
    "peak_gbytes_per_s": 0,
    "torch": "none",
    "torch_cuda": "none",
    "cuda_available": False,
    "cuda_functional": False,
    "device": "cpu",
    "python": platform.python_version(),
    "interpreter": sys.executable,
    "nvcc": "none",
    "triton": "none",
    "mojo": first_line(command_text(["mojo", "--version"])),
    "errors": [],
}
nvcc = re.search(r"release\s+([0-9.]+)", command_text(["nvcc", "--version"]))
if nvcc:
    report["nvcc"] = nvcc.group(1)
try:
    report["triton"] = importlib.metadata.version("triton")
except importlib.metadata.PackageNotFoundError:
    pass

try:
    import torch

    report["torch"] = str(torch.__version__)
    report["torch_cuda"] = torch.version.cuda or "none"
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["device_count"] = torch.cuda.device_count()
    if not report["cuda_available"] or not report["device_count"]:
        raise RuntimeError("PyTorch has no available CUDA device")

    props = torch.cuda.get_device_properties(0)
    report.update({
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "memory_mib": props.total_memory // 2**20,
        "multiprocessors": props.multi_processor_count,
    })
    uuid = getattr(props, "uuid", None)
    if uuid:
        report["power_limit_w"] = first_line(command_text([
            "nvidia-smi", "-i", str(uuid), "--query-gpu=power.limit",
            "--format=csv,noheader,nounits",
        ]), "n/a")

    def check_matmul(dtype):
        x = torch.full((32, 32), 0.125, device="cuda:0", dtype=dtype)
        y = x @ x
        torch.cuda.synchronize(0)
        if not torch.isfinite(y).all().item():
            raise RuntimeError(f"{dtype} CUDA matmul produced non-finite outputs")

    check_matmul(torch.float16)
    report["fp16"] = True
    report["cuda_functional"] = True
    report["device"] = "cuda"
    try:
        with torch.cuda.device(0):
            report["bf16_supported"] = bool(props.major >= 8 and torch.cuda.is_bf16_supported())
        if report["bf16_supported"]:
            check_matmul(torch.bfloat16)
            report["bf16"] = True
    except Exception as error:
        report["errors"].append(f"bf16 check failed; use fp16: {type(error).__name__}: {error}")
except Exception as error:
    report["errors"].append(f"CUDA check failed: {type(error).__name__}: {error}")

out = Path(sys.argv[1])
out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(f"python={report['interpreter']} torch={report['torch']} built_cuda={report['torch_cuda']}")
print(f"device={report['name']} count={report['device_count']} cc={report['compute_capability']} "
      f"mem={report['memory_mib']}MiB fp16={report['fp16']} bf16={report['bf16']}")
for error in report["errors"]:
    print(error)
if not report["cuda_functional"]:
    print("No functional CUDA in this interpreter. Enable a GPU runtime on Colab/Kaggle, check the "
          "NVIDIA driver and CUDA_VISIBLE_DEVICES, and install a compatible CUDA-enabled PyTorch "
          "in the interpreter shown above. For CPU training use BRAINLAB_DEVICE=cpu.")
print(f"wrote {out}")
raise SystemExit(0 if report["cuda_functional"] else 1)
PY
