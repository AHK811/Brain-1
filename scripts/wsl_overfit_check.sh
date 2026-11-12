#!/usr/bin/env bash
set -uo pipefail

ROOT="${BRAINLAB_ROOT:-/mnt/d/BRAIN-LAB}"
VENV="${BRAINLAB_VENV:-$HOME/brainlab-venv}"
PY="$VENV/bin/python"
STEPS="${STEPS:-150}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$ROOT" || exit 2

echo "== one batch of data: 2 sequences, repeated"
"$PY" - "$WORK" <<'PYEOF' || exit 1
import sys
from pathlib import Path

import numpy as np

from brainlab.data.pack import PackConfig, pack_and_write

work = Path(sys.argv[1])
rng = np.random.default_rng(0)
train_docs = [[int(v) for v in rng.integers(2, 250, size=32)] for _ in range(2)]
val_docs = [[int(v) for v in rng.integers(2, 250, size=32)] for _ in range(8)]
pack_and_write(train_docs, work / "train", "train", PackConfig(seq_len=32, shard_tokens=64, eos_id=1))
pack_and_write(val_docs, work / "val", "val", PackConfig(seq_len=32, shard_tokens=64, eos_id=1))
PYEOF

"$PY" -m brainlab.train.train \
  --train-dir "$WORK/train" --val-dir "$WORK/val" --seq-len 32 \
  --micro-batch 1 --accum 1 --steps "$STEPS" --eval-every "$STEPS" --eval-batches 1 \
  --save-every "$STEPS" --log-every 10 --device cpu --dtype fp32 --seed 1 \
  --out "$WORK/ckpt" >"$WORK/run.out" 2>&1
status=$?
grep '"event": "train"' "$WORK/run.out" | tail -3
grep -E '^diverged|^resumed' "$WORK/run.out" | head -2
[ "$status" = 0 ] || { echo "training exited $status"; tail -12 "$WORK/run.out"; exit 1; }

"$PY" - "$WORK/run.out" "$STEPS" <<'PYEOF' || exit 1
import json
import math
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
        if line.startswith('{"')]
train = [row for row in rows if row.get("event") == "train"]
steps, final = int(sys.argv[2]), train[-1]
floor = math.log(16384)
print(f"loss at step {train[0]['step']} {train[0]['loss']:.4f}  final loss {final['loss']:.4f}  "
      f"uniform floor {floor:.4f}  final/floor {final['loss'] / floor:.3f}")
if final["loss"] > 1.0:
    print("FAIL: 33M parameters could not memorize 64 tokens; the loss, optimizer or "
          "data path is wrong and no parity test would have caught it")
    sys.exit(1)
print("ok: the model memorised a single batch, so forward, loss and optimiser move together")
PYEOF
