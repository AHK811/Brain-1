#!/usr/bin/env bash
set -uo pipefail

ROOT="${BRAINLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${BRAINLAB_PYTHON:-python3}"
[ -x "$ROOT/.venv/bin/python" ] && PYTHON="$ROOT/.venv/bin/python"
[ -x "$ROOT/.venv/Scripts/python.exe" ] && PYTHON="$ROOT/.venv/Scripts/python.exe"
OUT="$ROOT/bench/results/gpu_check.json"
mkdir -p "$ROOT/bench/results"

CUDA_VISIBLE="${CUDA_VISIBLE_DEVICES:-all}"
DEVICE_COUNT=0
NAME="none"
CC="0.0"
MEM_MIB=0
POWER="n/a"
BF16=false
FP16=false
TORCH_VER="none"

if command -v nvidia-smi >/dev/null 2>&1; then
  LINE="$(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,power.limit --format=csv,noheader,nounits 2>/dev/null | head -1)"
  if [ -n "$LINE" ]; then
    NAME="$(printf '%s' "$LINE" | cut -d, -f1 | sed 's/^ *//')"
    CC="$(printf '%s' "$LINE" | cut -d, -f2 | tr -d ' ')"
    MEM_MIB="$(printf '%s' "$LINE" | cut -d, -f3 | tr -d ' ')"
    DRIVER="$(printf '%s' "$LINE" | cut -d, -f4 | tr -d ' ')"
    POWER="$(printf '%s' "$LINE" | cut -d, -f5 | tr -d ' ')"
    DEVICE_COUNT=1
  fi
else
  DRIVER="none"
fi

case "$NAME" in
  *"Tesla T4"*|*" T4 "*) PEAK=320 ;;
  *"A100-SXM4-40GB"*) PEAK=1555 ;;
  *"A100"*"80GB"*) PEAK=2039 ;;
  *"A10"*) PEAK=600 ;;
  *"L4"*) PEAK=300 ;;
  *"RTX 4000 Ada"*) PEAK=180 ;;
  *) PEAK=0 ;;
esac
if [ "$PEAK" -eq 0 ] && [ -n "$CC" ]; then
  echo "no published peak bandwidth on file for \"$NAME\"; pass --peak-gbs to the harness"
fi

TORCH_LINE="$("$PYTHON" - "$CC" <<'PY' 2>&1
import sys
try:
    import torch
except Exception as error:
    print(f"torch import failed: {error}")
    raise SystemExit(1)
print(f"torch={torch.__version__} built_cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"device={props.name} capability={props.major}.{props.minor} memory_mib={props.total_memory // 2**20} sms={props.multi_processor_count}")
    print(f"bf16={torch.cuda.is_bf16_supported()}")
    x = torch.randn(1024, 512, device="cuda")
    y = (x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()).half()
    torch.cuda.synchronize()
    z = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    for _ in range(20):
        z @ z
    torch.cuda.synchronize()
    print("fp16_matmul=ok")
else:
    print("bf16=False")
PY
)"
printf '%s\n' "$TORCH_LINE"
echo "$TORCH_LINE" | grep -q "available=True" && DEVICE_COUNT=1
FP16=$(echo "$TORCH_LINE" | grep -q "fp16_matmul=ok" && echo true || echo false)
BF16=$(echo "$TORCH_LINE" | sed -n 's/.*bf16=\(True\|False\).*/\1/p' | tr 'A-Z' 'a-z')
CAP=$(echo "$TORCH_LINE" | sed -n 's/.*capability=\([0-9.]*\).*/\1/p' | head -1)
[ -n "$CAP" ] && CC="$CAP"
SMS=$(echo "$TORCH_LINE" | sed -n 's/.*sms=\([0-9]*\).*/\1/p')
NVCC="$(command -v nvcc >/dev/null 2>&1 && nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' || echo none)"
MOJO="$(command -v mojo >/dev/null 2>&1 && mojo --version 2>&1 | head -1 || echo none)"
TRITON="$("$PYTHON" -c 'import triton; print(triton.__version__)' 2>/dev/null || echo none)"

cat > "$OUT" <<EOF
{
  "cuda_visible_devices": "$CUDA_VISIBLE",
  "device_count": $DEVICE_COUNT,
  "name": "$NAME",
  "compute_capability": "$CC",
  "memory_mib": ${MEM_MIB:-0},
  "multiprocessors": "${SMS:-0}",
  "driver": "${DRIVER:-none}",
  "power_limit_w": "$POWER",
  "fp16": $FP16,
  "bf16": $(echo "$BF16" | grep -q true && echo true || echo false),
  "peak_gbytes_per_s": $PEAK,
  "torch": "$TORCH_LINE",
  "nvcc": "$NVCC",
  "triton": "$TRITON",
  "mojo": "$MOJO"
}
EOF

echo
echo "device=$NAME cc=$CC mem=${MEM_MIB}MiB driver=${DRIVER:-none} peak=${PEAK}GB/s"
echo "nvcc=$NVCC triton=$TRITON mojo=$MOJO"
if [ "$DEVICE_COUNT" -eq 0 ]; then
  echo "no CUDA device visible; the harness will record CPU rows only"
  echo "wrote $OUT"
  exit 1
fi
case "$CC" in
  7.5) echo "T4 class: fp16 + GradScaler, no bf16, expect ~320 GB/s ceiling on memory-bound ops" ;;
  8.0|8.6|8.9|9.0) echo "ampere or newer: bf16 available" ;;
  *) echo "capability $CC outside the tuned list; pass --peak-gbs explicitly" ;;
esac
if [ "$PEAK" -gt 0 ]; then
  echo "suggest: just bench-rmsnorm  (harness reads $OUT)  or  --peak-gbs $PEAK"
fi
echo "wrote $OUT"
exit 0
