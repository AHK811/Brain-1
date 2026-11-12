#!/usr/bin/env bash
set -uo pipefail

ROOT="${BRAINLAB_ROOT:-/mnt/d/BRAIN-LAB}"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/brainlab-target}"
export PATH="$CARGO_HOME/bin:$PATH"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR" "$ROOT/bench/results"
BIN="${FLOPS_BIN:-$CARGO_TARGET_DIR/release/flops}"

cd "$ROOT/rust" || exit 2
if [ ! -x "$BIN" ]; then
  cargo build --release -q -p brainlab-infer-cpu --bin flops || exit 1
fi

"$BIN" --sweep "$@" | tee "$LOG_DIR/flops.json" || {
  echo "flops probe failed" >&2
  exit 1
}
echo "wrote $LOG_DIR/flops.json"
