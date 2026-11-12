#!/usr/bin/env bash
set -uo pipefail

ROOT="${BRAINLAB_ROOT:-/mnt/d/BRAIN-LAB}"
MODEL="${MODEL:-$ROOT/tests/golden/model.safetensors}"
GOLDEN="${GOLDEN:-$ROOT/tests/golden}"
KEYS="${KEYS:-forward_fp32_1x1024 forward_fp32_4x1024}"
REPS="${REPS:-3}"
DECODE="${DECODE:-24}"
THREADS="${THREADS:-$(nproc)}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/brainlab-target}"
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
BIN="$CARGO_TARGET_DIR/release/brainlab"
mkdir -p "$ROOT/logs"

cd "$ROOT/rust" || exit 2
cargo build --release -q -p brainlab-infer-cpu --bin brainlab --features profile || exit 1

for key in $KEYS; do
  out="$ROOT/logs/profile_${key}.json"
  "$BIN" --threads "$THREADS" profile --model "$MODEL" --golden-dir "$GOLDEN" --key "$key" \
    --reps "$REPS" --decode "$DECODE" > "$out" || { echo "profile failed for $key" >&2; exit 1; }
  echo "wrote $out"
done
