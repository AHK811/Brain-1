#!/usr/bin/env bash
set -uo pipefail

ROOT="${BRAINLAB_ROOT:-/mnt/d/BRAIN-LAB}"
VENV="${BRAINLAB_VENV:-$HOME/brainlab-venv}"
PY="$VENV/bin/python"
STEPS="${STEPS:-500}"
MASKED_STEPS="${MASKED_STEPS:-100}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$ROOT" || exit 2

echo "== overfit baseline: one 10-line math/code snippet, single micro-batch, $STEPS steps"
"$PY" - "$WORK" <<'PYEOF' || exit 1
import sys
from pathlib import Path

from brainlab.data.pack import PackConfig, pack_and_write
from brainlab.data.sft import prepare_sft
from brainlab.tok.reference import SPECIAL_TOKENS, Tokenizer
from brainlab.train.objective import ObjectiveConfig

work = Path(sys.argv[1])
tokenizer = Tokenizer(
    token_bytes=tuple(bytes([b]) for b in range(256)),
    merges=(),
    eos_id=256,
    pad_id=257,
)
snippet = """\
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def fib(n):
    x, y = 0, 1
    for _ in range(n):
        x, y = y, x + y
    return x

# euler: sum of 1/k^2 -> pi^2/6 for large k
""".splitlines()
ids = [tokenizer.encode(line, allow_special=False) for line in snippet]
cfg = PackConfig(seq_len=64, shard_tokens=512, eos_id=tokenizer.eos_id)
pack_and_write(ids, work / "train", "train", cfg)
pack_and_write([*ids, *ids], work / "val", "val", cfg)

chat_tok = Tokenizer(
    token_bytes=tuple(bytes([b]) for b in range(256)) + tuple(label.encode("ascii") for label in SPECIAL_TOKENS),
    merges=(),
    eos_id=256 + len(SPECIAL_TOKENS) - 1,
    pad_id=256 + len(SPECIAL_TOKENS) - 2,
)
objective = ObjectiveConfig(thought_weight=2.0)
records = [
    {"messages": [
        {"role": "user", "content": "explain euclid's algorithm"},
        {"role": "assistant", "content": [
            {"type": "thought", "text": "a mod b shrinks b each step so it terminates"},
            {"type": "text", "text": "gcd repeatedly replaces (a, b) with (b, a mod b)."},
        ]},
    ]}
    for _ in range(4)
]
chat_cfg = PackConfig(seq_len=64, shard_tokens=512, eos_id=chat_tok.eos_id)
prepare_sft(records, chat_tok, work / "chat_train", chat_cfg, objective, "train")
prepare_sft(records[:2], chat_tok, work / "chat_val", chat_cfg, objective, "val")
PYEOF

"$PY" -m brainlab.train.train \
  --train-dir "$WORK/train" --val-dir "$WORK/val" --seq-len 64 \
  --micro-batch 1 --accum 1 --steps "$STEPS" --eval-every "$STEPS" --eval-batches 1 \
  --save-every "$STEPS" --log-every 25 --device cpu --dtype fp32 --seed 1 \
  --out "$WORK/ckpt" >"$WORK/run.out" 2>&1
status=$?
grep '"event": "train"' "$WORK/run.out" | tail -2
grep -E '^diverged|^resumed' "$WORK/run.out" | head -2
[ "$status" = 0 ] || { echo "training exited $status"; tail -12 "$WORK/run.out"; exit 1; }

"$PY" - "$WORK/run.out" "$STEPS" <<'PYEOF' || exit 1
import json
import math
import sys
from pathlib import Path

from brainlab.spec.config import load_model_spec

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
        if line.startswith('{"')]
train = [row for row in rows if row.get("event") == "train"]
steps, final = int(sys.argv[2]), train[-1]
floor = math.log(load_model_spec().vocab_size)
print(f"loss at step {train[0]['step']} {train[0]['loss']:.4f}  final loss {final['loss']:.4f}  "
      f"uniform floor {floor:.4f}  final/floor {final['loss'] / floor:.3f}")
if final["loss"] > 0.1 * floor:
    print("FAIL: the model could not memorise the single snippet; the loss, objective or "
          "data path is wrong and no parity test would have caught it")
    sys.exit(1)
print("ok: the model memorised one micro-batch, so forward, loss and optimiser move together")
PYEOF

echo "== masked chat case: assistant-only labels with 2.0x thought spans, $MASKED_STEPS steps"
"$PY" -m brainlab.train.train \
  --train-dir "$WORK/chat_train" --val-dir "$WORK/chat_val" --seq-len 64 \
  --micro-batch 1 --accum 1 --steps "$MASKED_STEPS" --eval-every "$MASKED_STEPS" --eval-batches 1 \
  --save-every "$MASKED_STEPS" --log-every 10 --device cpu --dtype fp32 --seed 1 \
  --out "$WORK/chat_ckpt" >"$WORK/chat.out" 2>&1
status=$?
grep '"event": "train"' "$WORK/chat.out" | tail -2
[ "$status" = 0 ] || { echo "chat training exited $status"; tail -12 "$WORK/chat.out"; exit 1; }

"$PY" - "$WORK/chat.out" <<'PYEOF' || exit 1
import json
import math
import sys
from pathlib import Path

from brainlab.spec.config import load_model_spec

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
        if line.startswith('{"')]
train = [row for row in rows if row.get("event") == "train"]
final = train[-1]
floor = math.log(load_model_spec().vocab_size)
print(f"masked final loss {final['loss']:.4f}  uniform floor {floor:.4f}")
if final["loss"] > 0.25 * floor:
    print("FAIL: the weighted loss ignored the sidecar mask and behaved like uniform labels")
    sys.exit(1)
print("ok: the weighted objective trained on the assistant labels")
PYEOF

echo "overfit baseline passed"
