#!/usr/bin/env bash
set -euo pipefail

ROOT="${BRAINLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${1:-build}"

SETUP_TMP="/tmp/brainlab-wsl-setup.$$.sh"
tr -d '\r' < "$ROOT/scripts/wsl_setup.sh" > "$SETUP_TMP"
source "$SETUP_TMP" >/dev/null
rm -f "$SETUP_TMP"

export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/brainlab-target}"
export BRAINLAB_GOLDEN="${BRAINLAB_GOLDEN:-$ROOT/tests/golden}"

cd "$ROOT/rust"
case "$MODE" in
  build)
    cargo build --workspace --release
    ;;
  test)
    cargo test --workspace --release
    ;;
  check)
    cargo clippy --workspace --all-targets --release -- -D warnings
    ;;
  list)
    cargo run --release -p brainlab-infer-cpu --bin brainlab -- list --golden-dir "$BRAINLAB_GOLDEN"
    exit 0
    ;;
  *)
    echo "unknown mode: $MODE (use build, test, check, list)" >&2
    exit 2
    ;;
esac

BIN="$CARGO_TARGET_DIR/release/brainlab"
if [ -x "$BIN" ]; then
  echo "binary=$BIN"
  "$BIN" list --golden-dir "$BRAINLAB_GOLDEN" | head -12
fi
