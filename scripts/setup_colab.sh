#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BRAINLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEVICE="${BRAINLAB_DEVICE:-auto}"
TORCH_PIN="${BRAINLAB_TORCH_PIN:-2.4.1}"
SAFETENSORS_PIN="${BRAINLAB_SAFETENSORS_PIN:-0.4.4}"
MOJO_INSTALL="${BRAINLAB_MOJO_INSTALL:-1}"
FORCE_TORCH="${BRAINLAB_FORCE_TORCH:-0}"
SKIP_RUST="${BRAINLAB_SKIP_RUST:-0}"
OPS="${BRAINLAB_GOLDEN_OPS:-rmsnorm softmax attention matmul}"
WITH_MODEL="${BRAINLAB_GOLDEN_MODEL:-1}"
LOG="$ROOT/bench/results/setup_colab.log"

note() { printf '[setup] %s\n' "$*"; }
fail() { note "ERROR: $*" >&2; exit 1; }
run_required() {
  note "$1"
  shift
  "$@" 2>&1 | tee -a "$LOG"
}
trap 'fail "Required step failed at line $LINENO. See $LOG."' ERR

case "$DEVICE" in
  auto|cpu|cuda) ;;
  *) fail "BRAINLAB_DEVICE must be auto, cpu, or cuda (got $DEVICE)" ;;
esac
cd "$ROOT"
ROOT="$PWD"
LOG="$ROOT/bench/results/setup_colab.log"
mkdir -p "$ROOT/bench/results" "$ROOT/data"
rm -f "$ROOT/bench/results/env.json"

if [ -n "${BRAINLAB_PYTHON:-}" ]; then
  PYTHON="$BRAINLAB_PYTHON"
else
  case "${OSTYPE:-}" in
    msys*|cygwin*|win32*) PYTHON=python; VENV_PYTHON="$ROOT/.venv/Scripts/python.exe" ;;
    *) PYTHON=python3; VENV_PYTHON="$ROOT/.venv/bin/python" ;;
  esac
  if [ ! -x "$VENV_PYTHON" ]; then
    run_required "creating $ROOT/.venv with access to preinstalled packages" \
      "$PYTHON" -m venv --system-site-packages "$ROOT/.venv"
  fi
  [ -x "$VENV_PYTHON" ] || fail "No venv interpreter at $VENV_PYTHON; set BRAINLAB_PYTHON explicitly."
  PYTHON="$VENV_PYTHON"
fi
note "root=$ROOT python=$PYTHON device=$DEVICE"
run_required "checking Python" "$PYTHON" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11 or newer is required; set BRAINLAB_PYTHON to a compatible interpreter.")
print(f"Python {sys.version.split()[0]} at {sys.executable}")
PY

PROBE="$("$PYTHON" - "$DEVICE" "$FORCE_TORCH" <<'PY'
import importlib.util
import subprocess
import sys

device, force = sys.argv[1:]
version = "none"
visible = False
if importlib.util.find_spec("torch") is not None:
    try:
        import torch
    except Exception:
        if force != "1":
            raise
        print("Existing torch import failed; attempting explicitly forced reinstall.", file=sys.stderr)
    else:
        version = str(torch.__version__)
        if device == "auto":
            try:
                visible = torch.cuda.is_available()
            except Exception as error:
                print(f"CUDA discovery failed: {error}", file=sys.stderr)
if device == "auto":
    if not visible:
        try:
            smi = subprocess.run([
                "nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
            ], capture_output=True, text=True, timeout=15, check=True)
            visible = bool(smi.stdout.strip())
        except (OSError, subprocess.SubprocessError, UnicodeError):
            pass
    device = "cuda" if visible else "cpu"
sys.stdout.write(f"{version}|{device}")
PY
)"
IFS='|' read -r TORCH_NOW INSTALL_DEVICE <<< "$PROBE"
run_required "upgrading pip" "$PYTHON" -m pip install --quiet --upgrade pip
if [ "$TORCH_NOW" = "none" ] || [ "$FORCE_TORCH" = "1" ]; then
  TORCH_SPEC="torch>=2.4"
  TORCH_FLAGS=()
  if [ "$FORCE_TORCH" = "1" ]; then
    TORCH_SPEC="torch==$TORCH_PIN"
    TORCH_FLAGS=(--force-reinstall)
  fi
  if [ "$INSTALL_DEVICE" = "cpu" ]; then
    DEFAULT_INDEX="https://download.pytorch.org/whl/cpu"
  else
    DEFAULT_INDEX="https://download.pytorch.org/whl/cu128"
    if [ "$FORCE_TORCH" = "1" ]; then
      case "$TORCH_PIN" in
        2.4.*|2.5.*) DEFAULT_INDEX="https://download.pytorch.org/whl/cu124" ;;
        2.6.*) DEFAULT_INDEX="https://download.pytorch.org/whl/cu126" ;;
      esac
    fi
  fi
  TORCH_INDEX="${BRAINLAB_TORCH_INDEX_URL:-$DEFAULT_INDEX}"
  note "installing $TORCH_SPEC for $INSTALL_DEVICE; override the wheel channel with BRAINLAB_TORCH_INDEX_URL"
  run_required "installing PyTorch (pin and wheel channel must be compatible)" \
    "$PYTHON" -m pip install --quiet "${TORCH_FLAGS[@]}" "$TORCH_SPEC" --index-url "$TORCH_INDEX"
else
  note "keeping preinstalled torch=$TORCH_NOW; only BRAINLAB_FORCE_TORCH=1 permits replacement"
fi
TORCH_FINAL="$("$PYTHON" -c 'import sys, torch; sys.stdout.write(str(torch.__version__))')"
run_required "installing project dependencies without replacing torch=$TORCH_FINAL" \
  "$PYTHON" -m pip install --quiet -e ".[dev,data]" "torch==$TORCH_FINAL" "safetensors>=$SAFETENSORS_PIN"
run_required "checking required imports and CPU tensor execution" "$PYTHON" - <<'PY'
import brainlab
import huggingface_hub
import numpy as np
import safetensors
import torch

x = torch.from_numpy(np.ones((8, 8), dtype=np.float32))
y = (x @ x).numpy()
if not np.isfinite(y).all():
    raise SystemExit("CPU PyTorch/NumPy execution produced non-finite outputs")
print(f"torch={torch.__version__} numpy={np.__version__} safetensors={safetensors.__version__}")
PY

if BRAINLAB_ROOT="$ROOT" BRAINLAB_PYTHON="$PYTHON" bash "$ROOT/scripts/check_gpu.sh" 2>&1 | tee -a "$LOG"; then
  CUDA_OK=1
else
  CUDA_OK=0
fi
case "$DEVICE" in
  cuda)
    [ "$CUDA_OK" = "1" ] || fail "BRAINLAB_DEVICE=cuda requires a working PyTorch CUDA matmul. Check the GPU runtime, NVIDIA driver and CUDA_VISIBLE_DEVICES. If torch is CPU-only or incompatible, use BRAINLAB_FORCE_TORCH=1 with a compatible BRAINLAB_TORCH_PIN and BRAINLAB_TORCH_INDEX_URL, or select BRAINLAB_DEVICE=cpu."
    SELECTED_DEVICE=cuda
    ;;
  auto)
    if [ "$CUDA_OK" = "1" ]; then SELECTED_DEVICE=cuda; else SELECTED_DEVICE=cpu; fi
    ;;
  cpu) SELECTED_DEVICE=cpu ;;
esac
note "selected device=$SELECTED_DEVICE"

PINNED_RUST="$("$PYTHON" - "$ROOT/rust-toolchain.toml" <<'PY'
import sys
import tomllib
from pathlib import Path

path = Path(sys.argv[1])
config = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
sys.stdout.write(config.get("toolchain", {}).get("channel", "stable"))
PY
)"
RUST_TOOLCHAIN="${BRAINLAB_RUST_TOOLCHAIN:-$PINNED_RUST}"
CARGO_VERSION=skipped
if [ "$SKIP_RUST" != "1" ]; then
  if ! command -v cargo >/dev/null 2>&1; then
    note "installing rust $RUST_TOOLCHAIN through rustup"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain "$RUST_TOOLCHAIN" --profile minimal >> "$LOG" 2>&1 \
      || note "optional rustup installation failed"
  fi
  if [ -f "$HOME/.cargo/env" ]; then . "$HOME/.cargo/env"; fi
  CARGO_VERSION="$(cargo --version 2>/dev/null)" || CARGO_VERSION=none
fi
note "cargo=$CARGO_VERSION"

if [ "$MOJO_INSTALL" = "1" ] && ! command -v mojo >/dev/null 2>&1; then
  note "installing the modular toolkit"
  curl -fsSL https://modular.com/install.sh | sh -s -- --yes >> "$LOG" 2>&1 \
    || note "optional modular installation failed"
  if [ -f "$HOME/.modular/env.sh" ]; then . "$HOME/.modular/env.sh"; fi
fi
MOJO_VERSION="$(mojo --version 2>/dev/null)" || MOJO_VERSION=none
note "mojo=$MOJO_VERSION"

EXTRA=(--raw)
if [ "$WITH_MODEL" = "1" ]; then
  EXTRA+=(--model)
fi
read -r -a GOLDEN_OPS <<< "$OPS"
run_required "generating golden tensors for: $OPS" \
  "$PYTHON" -m brainlab.export.make_golden --ops "${GOLDEN_OPS[@]}" "${EXTRA[@]}"

run_required "writing final environment report" "$PYTHON" - \
  "$ROOT/bench/results/env.json" "$ROOT/bench/results/gpu_check.json" \
  "$DEVICE" "$SELECTED_DEVICE" "$FORCE_TORCH" "$TORCH_PIN" "$RUST_TOOLCHAIN" \
  "$CARGO_VERSION" "$MOJO_VERSION" <<'PY'
import json
import platform
import sys
from pathlib import Path

import huggingface_hub
import numpy
import safetensors
import torch

out, gpu_path, requested, selected, force, pin, rust, cargo, mojo = sys.argv[1:]
gpu = json.loads(Path(gpu_path).read_text(encoding="utf-8"))
report = {
    "gpu": gpu["name"],
    "gpu_memory_mib": gpu["memory_mib"],
    "driver": gpu["driver"],
    "python": f"Python {platform.python_version()}",
    "interpreter": sys.executable,
    "torch": str(torch.__version__),
    "torch_cuda": torch.version.cuda or "none",
    "numpy": numpy.__version__,
    "safetensors": safetensors.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_functional": gpu["cuda_functional"],
    "device_requested": requested,
    "device": selected,
    "pinned": {"torch": pin if force == "1" else None, "rust": rust},
    "cargo": cargo,
    "mojo": mojo,
    "log": "bench/results/setup_colab.log",
}
Path(out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(f"wrote {out}")
PY
note "done"
