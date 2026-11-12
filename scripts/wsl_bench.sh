#!/usr/bin/env bash
set -uo pipefail

ROOT="${BRAINLAB_ROOT:-/mnt/d/BRAIN-LAB}"
BIN="${BRAINLAB_RUST_BIN:-$HOME/brainlab-target/release/brainlab}"
GOLDEN="${BRAINLAB_GOLDEN:-$ROOT/tests/golden}"
OUT="$ROOT/bench/results/rust"
REPS="${REPS:-60}"
WARMUP="${WARMUP:-12}"
THREADS="${THREADS:-$(nproc)}"
mkdir -p "$OUT"

if [ ! -x "$BIN" ]; then
  echo "rust binary not found at $BIN; run scripts/wsl_build.sh build first" >&2
  exit 1
fi

KEYS=(
  rmsnorm_fp32_4096x512
  rmsnorm_fp32_1024x512
  rmsnorm_fp32_2000x384
  rmsnorm_fp32_1x512
  softmax_fp32_4096x512
  softmax_fp32_1024x1024
  softmax_fp32_2000x384
  matmul_fp32_1024x512x1536
  matmul_fp32_2048x512x512
  matmul_fp32_512x1344x512
  attention_fp32_4x8x256x64
  attention_fp32_2x8x512x64
)
if [ "${ONLY:-}" = "rmsnorm" ]; then
  KEYS=(rmsnorm_fp32_4096x512 rmsnorm_fp32_1024x512 rmsnorm_fp32_2000x384 rmsnorm_fp32_1x512)
fi
if [ -f "${KEYFILE:-}" ]; then
  mapfile -t KEYS < "$KEYFILE"
fi

FAILED=0
for key in "${KEYS[@]}"; do
  if "$BIN" bench --golden-dir "$GOLDEN" --key "$key" --reps "$REPS" --warmup "$WARMUP" --threads "$THREADS" \
    > "$OUT/$key.json" 2> "$OUT/$key.err"; then
    printf '%-32s %s\n' "$key" ok
  else
    printf '%-32s FAILED %s\n' "$key" "$(head -c 200 "$OUT/$key.err")"
    FAILED=$((FAILED + 1))
  fi
done

if [ "${FORWARD:-0}" = "1" ] && [ -f "$GOLDEN/model.safetensors" ]; then
  for key in forward_fp32_1x512 forward_fp32_1x1024 forward_fp32_4x1024; do
    if "$BIN" forward --model "$GOLDEN/model.safetensors" --golden-dir "$GOLDEN" --key "$key" \
      --reps "${FWD_REPS:-3}" --warmup "${FWD_WARMUP:-1}" --threads "$THREADS" > "$OUT/$key.json" 2> "$OUT/$key.err"; then
      printf '%-32s %s\n' "$key" ok
    else
      printf '%-32s FAILED %s\n' "$key" "$(head -c 200 "$OUT/$key.err")"
      FAILED=$((FAILED + 1))
    fi
  done
fi

if [ -f "$GOLDEN/model.safetensors" ]; then
  "$BIN" self-test --model "$GOLDEN/model.safetensors" --prefill 64 --steps 8 > "$OUT/selftest.json" 2> "$OUT/selftest.err" \
    || printf 'self-test FAILED %s\n' "$(head -c 200 "$OUT/selftest.err")"
fi

if [ -f "$GOLDEN/model.safetensors" ]; then
  "$BIN" info --model "$GOLDEN/model.safetensors" > "$OUT/model_info.json" 2> "$OUT/model_info.err" \
    || printf 'info FAILED %s\n' "$(head -c 200 "$OUT/model_info.err")"
fi
echo "wrote $OUT for ${#KEYS[@]} keys, failures=$FAILED"
exit "$FAILED"
