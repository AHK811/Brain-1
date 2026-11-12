#!/usr/bin/env bash
set -uo pipefail

ROOT="${BRAINLAB_ROOT:-/mnt/d/BRAIN-LAB}"
VENV="${BRAINLAB_VENV:-$HOME/brainlab-venv}"
BIN="${BRAINLAB_RUST_BIN:-$HOME/brainlab-target/release/brainlab}"
PYTHON="$VENV/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "no venv at $VENV; run: WITH_PYTHON=1 bash scripts/wsl_setup.sh" >&2
  exit 1
fi
if [ ! -x "$BIN" ]; then
  echo "no rust binary at $BIN; run: bash scripts/wsl_build.sh build" >&2
  exit 1
fi

cd "$ROOT" || exit 1
export BRAINLAB_ROOT="$ROOT"
export BRAINLAB_RUST_CMD="$BIN"
export PYTHONPATH="$ROOT:$ROOT/python"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"

exec "$PYTHON" -m bench.harness.run_all \
  --ops "${OPS:-rmsnorm}" \
  --dtypes "${DTYPES:-fp32}" \
  --reps "${REPS:-30}" \
  --warmup "${WARMUP:-6}" \
  --threads "${THREADS:-$(nproc)}" \
  --tag "${TAG:-wsl-single-host}" \
  "$@"
