#!/usr/bin/env bash
set -uo pipefail

ROOT="${BRAINLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${BRAINLAB_PYTHON:-python3}"
TORCH_PIN="${BRAINLAB_TORCH_PIN:-2.4.1}"
TRITON_PIN="${BRAINLAB_TRITON_PIN:-3.0.0}"
NUMPY_PIN="${BRAINLAB_NUMPY_PIN:-1.26.4}"
SAFETENSORS_PIN="${BRAINLAB_SAFETENSORS_PIN:-0.4.4}"
PINNED_RUST="$(sed -n 's/^channel = "\([^"]*\)".*/\1/p' "$ROOT/rust-toolchain.toml" 2>/dev/null | head -1)"
RUST_TOOLCHAIN="${BRAINLAB_RUST_TOOLCHAIN:-${PINNED_RUST:-stable}}"
MOJO_INSTALL="${BRAINLAB_MOJO_INSTALL:-1}"
FORCE_TORCH="${BRAINLAB_FORCE_TORCH:-0}"
SKIP_RUST="${BRAINLAB_SKIP_RUST:-0}"
OPS="${BRAINLAB_GOLDEN_OPS:-rmsnorm softmax attention matmul}"
WITH_MODEL="${BRAINLAB_GOLDEN_MODEL:-1}"
LOG="$ROOT/bench/results/setup_colab.log"
mkdir -p "$ROOT/bench/results" "$ROOT/data"

note() { printf '[setup] %s\n' "$*"; }
record() { printf '%s\n' "$*" >> "$LOG"; }

note "root=$ROOT python=$PYTHON"
cd "$ROOT" || exit 1

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  GPU_MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
  DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
else
  GPU_NAME="none"; GPU_MEM="0"; DRIVER="none"
fi
note "gpu=$GPU_NAME mem=${GPU_MEM}MiB driver=$DRIVER"

if [ ! -x "$ROOT/.venv/bin/python" ] && [ "$PYTHON" = "python3" ]; then
  note "no venv; creating $ROOT/.venv"
  "$PYTHON" -m venv "$ROOT/.venv" >> "$LOG" 2>&1 || note "venv creation failed, using system python"
  [ -x "$ROOT/.venv/bin/python" ] && PYTHON="$ROOT/.venv/bin/python"
fi
[ -x "$ROOT/.venv/Scripts/python.exe" ] && PYTHON="$ROOT/.venv/Scripts/python.exe"

note "python is $("$PYTHON" --version 2>&1)"
"$PYTHON" -m pip install --quiet --upgrade pip >> "$LOG" 2>&1
"$PYTHON" -m pip install --quiet -e ".[dev]" >> "$LOG" 2>&1 || note "editable install failed"

TORCH_NOW="$("$PYTHON" -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null || echo none)"
CUDA_BUILT="$("$PYTHON" -c 'import torch; print(torch.version.cuda or "none")' 2>/dev/null || echo none)"
note "torch=$TORCH_NOW cuda=$CUDA_BUILT pinned=$TORCH_PIN"
if [ "$TORCH_NOW" != "$TORCH_PIN" ] && [ "$FORCE_TORCH" = "1" ]; then
  note "installing pinned torch $TORCH_PIN for cuda $CUDA_BUILT"
  "$PYTHON" -m pip install --quiet "torch==$TORCH_PIN+$CUDA_BUILT" \
    --index-url "https://download.pytorch.org/whl/${CUDA_BUILT/./}" >> "$LOG" 2>&1 \
    || note "pinned torch install failed; keeping $TORCH_NOW"
elif [ "$TORCH_NOW" != "$TORCH_PIN" ]; then
  note "keeping the preinstalled torch; set BRAINLAB_FORCE_TORCH=1 to pin it"
fi

if [ "$GPU_NAME" != "none" ] && [ "$GPU_NAME" != "" ]; then
  TRITON_NOW="$("$PYTHON" -c 'import triton; print(triton.__version__)' 2>/dev/null || echo none)"
  note "triton=$TRITON_NOW pinned=$TRITON_PIN"
  if [ "$TRITON_NOW" = "none" ]; then
    "$PYTHON" -m pip install --quiet "triton==$TRITON_PIN" >> "$LOG" 2>&1 || "$PYTHON" -m pip install --quiet "triton>=$TRITON_PIN" >> "$LOG" 2>&1 || note "triton install failed"
  fi
  "$PYTHON" -m pip install --quiet huggingface_hub >> "$LOG" 2>&1 || true
fi
"$PYTHON" -m pip install --quiet "safetensors>=$SAFETENSORS_PIN" >> "$LOG" 2>&1 || true
"$PYTHON" -c "import numpy; print('numpy', numpy.__version__)" >> "$LOG" 2>&1 || true

if [ "$SKIP_RUST" != "1" ] && ! command -v cargo >/dev/null 2>&1; then
  note "installing rust $RUST_TOOLCHAIN through rustup"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain "$RUST_TOOLCHAIN" --profile minimal >> "$LOG" 2>&1 \
    || note "rustup failed"
fi
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
if command -v cargo >/dev/null 2>&1; then
  note "cargo=$(cargo --version)"
else
  note "cargo unavailable; rust rows will be skipped"
fi

if [ "$MOJO_INSTALL" = "1" ] && ! command -v mojo >/dev/null 2>&1; then
  note "installing the modular toolkit"
  curl -fsSL https://modular.com/install.sh | sh -s -- --yes >> "$LOG" 2>&1 || note "modular install failed"
  [ -f "$HOME/.modular/env.sh" ] && . "$HOME/.modular/env.sh"
fi
command -v mojo >/dev/null 2>&1 && note "mojo=$(mojo --version 2>&1 | head -1)" || note "mojo unavailable"

"$PYTHON" -c "import torch; print('cuda_available', torch.cuda.is_available())" 2>&1 | tail -1

if [ "$WITH_MODEL" = "1" ]; then
  EXTRA=(--model --raw)
else
  EXTRA=(--raw)
fi
note "generating golden tensors for: $OPS"
PYTHONPATH="$ROOT/python" "$PYTHON" -m brainlab.export.make_golden --ops $OPS "${EXTRA[@]}" >> "$LOG" 2>&1 \
  || note "golden generation failed, see $LOG"

cat > "$ROOT/bench/results/env.json" <<EOF
{
  "gpu": "$GPU_NAME",
  "gpu_memory_mib": ${GPU_MEM:-0},
  "driver": "$DRIVER",
  "python": "$("$PYTHON" --version 2>&1)",
  "torch": "$TORCH_NOW",
  "torch_cuda": "$CUDA_BUILT",
  "pinned": {"torch": "$TORCH_PIN", "triton": "$TRITON_PIN", "numpy": "$NUMPY_PIN", "safetensors": "$SAFETENSORS_PIN", "rust": "$RUST_TOOLCHAIN"},
  "cargo": "$(command -v cargo >/dev/null 2>&1 && cargo --version || echo none)",
  "mojo": "$(command -v mojo >/dev/null 2>&1 && mojo --version 2>&1 | head -1 || echo none)",
  "log": "bench/results/setup_colab.log"
}
EOF
note "wrote bench/results/env.json"
note "done"
exit 0
