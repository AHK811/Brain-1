#!/usr/bin/env bash
set -uo pipefail

ROOT="${BRAINLAB_ROOT:-/mnt/d/BRAIN-LAB}"
FIXTURE="$ROOT/tests/golden/datapack_fixture"
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/brainlab-target}"

if [ ! -d "$FIXTURE/corpus" ] || [ ! -f "$FIXTURE/tokenizer/tokenizer.bin" ]; then
  echo "inputs missing; run: python -m brainlab.tok.parity_fixture --datapack tests/golden/datapack_fixture" >&2
  exit 1
fi

rm -rf "$FIXTURE/rust"
cd "$ROOT/rust" || exit 1
cargo run --release -q -p brainlab-datapack --bin pack -- \
  --corpus "$FIXTURE/corpus" \
  --tokenizer "$FIXTURE/tokenizer" \
  --out "$FIXTURE/rust" \
  --seq-len 16 \
  --shard-tokens 48 \
  --val-ratio 0.25 \
  --report "$FIXTURE/rust/report.json"

echo "rust shards:"
find "$FIXTURE/rust" -type f | sed "s|$ROOT/||" | sort
