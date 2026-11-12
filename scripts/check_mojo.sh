#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PINNED="${MOJO_PINNED_VERSION:-}"
RESULTS="$ROOT/bench/results"
STATUS="$RESULTS/mojo_status.json"
LOG="$RESULTS/mojo_build.log"
mkdir -p "$RESULTS"

if command -v modular >/dev/null 2>&1 && [ -z "${MOJO_BIN:-}" ]; then
  modular --version > "$RESULTS/modular_version.txt" 2>&1 || true
fi
MOJO="${MOJO_BIN:-mojo}"

if ! command -v "$MOJO" >/dev/null 2>&1; then
  printf '{"installed": false, "reason": "%s not on PATH", "pinned": "%s"}\n' "$MOJO" "$PINNED" > "$STATUS"
  echo "mojo is not installed; wrote $STATUS"
  exit 0
fi

VERSION="$("$MOJO" --version 2>&1 | head -1)"
printf 'mojo version: %s\n' "$VERSION"
if [ -n "$PINNED" ] && [[ "$VERSION" != *"$PINNED"* ]]; then
  printf 'warning: installed %s does not match the pinned %s\n' "$VERSION" "$PINNED"
fi

cd "$ROOT"
if "$MOJO" build mojo/driver.mojo -o "$ROOT/build/mojo-driver" > "$LOG" 2>&1; then
  BUILD_OK=true
else
  BUILD_OK=false
fi

if [ -x "$ROOT/build/mojo-driver" ] && [ -f tests/golden/raw/rmsnorm_fp32_4096x512/x.f32.bin ]; then
  "$ROOT/build/mojo-driver" \
    --key rmsnorm_fp32_4096x512 \
    --x tests/golden/raw/rmsnorm_fp32_4096x512/x.f32.bin \
    --w tests/golden/raw/rmsnorm_fp32_4096x512/w.f32.bin \
    --y tests/golden/raw/rmsnorm_fp32_4096x512/y.f32.bin \
    --rows 4096 --cols 512 --reps 50 --warmup 10 2>&1 | tail -20
  RUN_OK=true
else
  RUN_OK=false
fi

EXCERPT="$(tail -c 400 "$LOG" 2>/dev/null | tr -d '\n"' | head -c 300)"
printf '{"installed": true, "version": "%s", "pinned": "%s", "build_ok": %s, "run_ok": %s, "log": "%s", "error_excerpt": "%s"}\n' \
  "$(printf '%s' "$VERSION" | tr -d '"')" "$PINNED" "$BUILD_OK" "$RUN_OK" "bench/results/mojo_build.log" "$EXCERPT" > "$STATUS"
cat "$STATUS"
