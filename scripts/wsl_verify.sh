#!/usr/bin/env bash
set -uo pipefail

ROOT="${BRAINLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STEPS="${STEPS:-fmt,clippy,test}"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/brainlab-target}"
export PATH="$CARGO_HOME/bin:$PATH"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"
cd "$ROOT/rust" || exit 2

{
  if command -v git >/dev/null 2>&1 && git -C "$ROOT" rev-parse HEAD >/dev/null 2>&1; then
    git -C "$ROOT" rev-parse HEAD
  else
    echo "no-git"
  fi
  rustc -V
  cargo -V
  printf "code_fingerprint="
  { find "$ROOT/rust/crates" -name '*.rs' -print0 | sort -z | xargs -0 cat; cat "$ROOT/rust/Cargo.lock"; } | sha256sum | cut -c1-64
} > "$LOG_DIR/commit.txt"
cat "$LOG_DIR/commit.txt"

FAILED=0
for step in ${STEPS//,/ }; do
  case "$step" in
    fmt)
      echo "== cargo fmt --check"
      if cargo fmt --all -- --check > "$LOG_DIR/fmt.log" 2>&1; then
        echo "   clean"
      else
        echo "   UNFORMATTED (see logs/fmt.log)"
        head -20 "$LOG_DIR/fmt.log"
        FAILED=$((FAILED + 1))
      fi
      ;;
    clippy)
      echo "== cargo clippy -D warnings"
      if cargo clippy --workspace --all-targets --release -- -D warnings > "$LOG_DIR/clippy.log" 2>&1; then
        echo "   clean"
      else
        grep -E "^error" "$LOG_DIR/clippy.log" | sort | uniq -c | sort -rn | head -15
        FAILED=$((FAILED + 1))
      fi
      ;;
    test)
      echo "== cargo test --workspace --release"
      if cargo test --workspace --release > "$LOG_DIR/test.log" 2>&1; then
        grep -E "^test result" "$LOG_DIR/test.log" | sed 's/^/   /'
      else
        grep -E "^test .* FAILED|^error|panicked" "$LOG_DIR/test.log" | head -20
        FAILED=$((FAILED + 1))
      fi
      ;;
    *)
      echo "unknown step: $step (use fmt,clippy,test)" >&2
      exit 2
      ;;
  esac
done

echo "rust verify finished with $FAILED failing step(s)"
exit "$FAILED"
