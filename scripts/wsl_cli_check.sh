#!/usr/bin/env bash
set -uo pipefail

ROOT="${BRAINLAB_ROOT:-/mnt/d/BRAIN-LAB}"
VENV="${BRAINLAB_VENV:-$HOME/brainlab-venv}"
PY="$VENV/bin/python"
FIXTURE="$ROOT/tests/golden/datapack_fixture/rust"
cd "$ROOT" || exit 2

FAILED=0

run() {
  local label="$1"
  local want="$2"
  shift 2
  "$@" >"$LOG" 2>&1
  local got=$?
  if [ "$got" = "$want" ]; then
    echo "   ok   $label exit=$got"
  else
    echo "   FAIL $label exit=$got want=$want"
    sed 's/^/        /' "$LOG" | tail -3
    FAILED=$((FAILED + 1))
  fi
}

LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

echo "== exit codes the documentation promises"
run "leak finds shared spans" 1 "$PY" -m brainlab.data.leakage \
  --train-dir "$FIXTURE/train" --val-dir "$FIXTURE/val" --seq-len 16 --span 8 --sequences 20
run "leak cannot compare a span wider than a sequence" 1 "$PY" -m brainlab.data.leakage \
  --train-dir "$FIXTURE/train" --val-dir "$FIXTURE/val" --seq-len 16 --span 40 --sequences 20
run "ppl --help" 0 "$PY" -m brainlab.eval.perplexity --help
run "interpreter exit status propagates" 3 "$PY" -c "import sys; sys.exit(3)"

echo "== brainlab leak is reachable through the console entry point"
if "$PY" -c "from brainlab.cli import COMMANDS; assert COMMANDS['leak'] == ('brainlab.data.leakage', 'main')"; then
  echo "   ok   cli COMMANDS wires leak"
else
  echo "   FAIL cli does not wire leak"
  FAILED=$((FAILED + 1))
fi

echo "cli check finished with $FAILED failing step(s)"
exit $((FAILED > 0))
