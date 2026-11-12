#!/usr/bin/env bash
set -uo pipefail

ROOT="${BRAINLAB_ROOT:-/mnt/d/BRAIN-LAB}"
VENV="${BRAINLAB_VENV:-$HOME/brainlab-venv}"
PY="$VENV/bin/python"
FIXTURE="$ROOT/tests/golden/datapack_fixture/rust"
STEPS="${STEPS:-8}"
HALF=$((STEPS / 2))
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$ROOT" || exit 2

if [ ! -d "$FIXTURE/train" ]; then
  echo "the datapack fixture is missing; run scripts/wsl_datapack_fixture.sh" >&2
  exit 2
fi

COMMON=(--train-dir "$FIXTURE/train" --val-dir "$FIXTURE/val" --seq-len 16
        --micro-batch 2 --accum 2 --eval-every "$HALF" --eval-batches 1
        --save-every "$HALF" --log-every 1 --device cpu --dtype fp32 --seed 7)

echo "== uninterrupted: $STEPS steps"
"$PY" -m brainlab.train.train "${COMMON[@]}" --steps "$STEPS" --out "$WORK/straight" >"$WORK/straight.out" 2>&1 || {
  echo "straight run failed"; tail -20 "$WORK/straight.out"; exit 1
}

echo "== split: run to $STEPS, kill it at step $HALF, resume to $STEPS"
"$PY" -m brainlab.train.train "${COMMON[@]}" --steps "$STEPS" --out "$WORK/split" >"$WORK/split1.out" 2>&1 &
pid=$!
MARK="$WORK/split/step-$(printf '%06d' "$HALF").pt"
for _ in $(seq 1 900); do
  if [ -f "$MARK" ]; then break; fi
  if ! kill -0 "$pid" 2>/dev/null; then break; fi
  sleep 2
done
if [ ! -f "$MARK" ]; then
  echo "the checkpoint at step $HALF never appeared"; tail -20 "$WORK/split1.out"; kill "$pid" 2>/dev/null; exit 1
fi
kill -TERM "$pid" 2>/dev/null
wait "$pid" 2>/dev/null
echo "   killed the run after writing $MARK"

"$PY" -m brainlab.train.train "${COMMON[@]}" --steps "$STEPS" --out "$WORK/split" >"$WORK/split2.out" 2>&1 || {
  echo "resume failed"; tail -20 "$WORK/split2.out"; exit 1
}
grep -m1 "^resumed" "$WORK/split2.out" || echo "   no resume line found"

"$PY" - "$WORK" "$STEPS" <<'PYEOF' || exit 1
import json
import sys
from pathlib import Path

import torch

work, steps = Path(sys.argv[1]), int(sys.argv[2])


def best(directory):
    payload = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    return int(payload["step"]), round(float(payload["metrics"]["val_loss"]), 6)


def curve(directory):
    rows = {}
    for line in (directory / "train_log.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("event") == "train":
            rows[record["step"]] = record
        if record.get("event") == "config":
            rows["config"] = record
    return rows


straight = curve(work / "straight")
split = curve(work / "split")
start_batch = split["config"]["start_batch"]
expected = (steps // 2) * split["config"]["accum_steps"]
worst = max(abs(straight[s]["loss"] - split[s]["loss"]) for s in range(1, steps + 1))
print(f"resumed start_batch={start_batch} expected={expected}")
print(f"worst |loss difference| over {steps} steps = {worst:.3e}")
for s in range(1, steps + 1):
    print(f"  step {s}: straight {straight[s]['loss']:.6f} split {split[s]['loss']:.6f}")
if worst > 1e-6:
    print("FAIL: a resumed run followed different data or a different schedule")
    sys.exit(1)
if start_batch != expected:
    print("FAIL: the checkpoint did not record the data cursor")
    sys.exit(1)
straight_best = best(work / "straight")
split_best = best(work / "split")
print(f"best.pt straight={straight_best} split={split_best}")
if straight_best != split_best:
    print("FAIL: the resumed run overwrote best.pt, so best_val did not survive the checkpoint")
    sys.exit(1)
print("ok: resume reproduced the uninterrupted loss curve and kept the same best.pt")
PYEOF

