#!/usr/bin/env bash
set -uo pipefail

ROOT="${BRAINLAB_ROOT:-/mnt/d/BRAIN-LAB}"
VENV="${BRAINLAB_VENV:-$HOME/brainlab-venv}"
PY="$VENV/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

FAILED=0

echo "== ignored paths that hold source"
SOURCE_IGNORED="$(git -C "$ROOT" ls-files --others --ignored --exclude-standard -- \
    python rust/crates scripts docs spec tests bench/harness mojo triton_kernels cuda \
  | grep -E '\.(py|rs|sh|md|toml|mojo|cu|cuh|yaml|yml)$' | grep -vE '__pycache__|/target/|\.venv/' || true)"
if [ -n "$SOURCE_IGNORED" ]; then
  echo "$SOURCE_IGNORED" | sed 's/^/   HIDDEN /'
  FAILED=$((FAILED + 1))
else
  echo "   clean"
fi

echo "== clean clone of committed HEAD"
git clone --quiet "$ROOT" "$WORK/clone" || exit 2
CLONE="$(cd "$WORK/clone" && pwd)"
echo "   files=$(git -C "$CLONE" ls-files | wc -l) head=$(git -C "$CLONE" rev-parse --short HEAD)"

echo "== imports from the clone"
PYTHONPATH="$CLONE/python:$CLONE" "$PY" - <<'EOF' || FAILED=$((FAILED + 1))
import importlib
import pkgutil
import sys

import brainlab

failures = []
names = ["brainlab"]
for module in pkgutil.walk_packages(brainlab.__path__, "brainlab."):
    names.append(module.name)
for name in sorted(names):
    if ".cuda" in name or "triton" in name:
        continue
    try:
        importlib.import_module(name)
    except Exception as error:
        failures.append((name, repr(error)))
for name, error in failures:
    print(f"   FAIL {name}: {error}")
print(f"   imported {len(names) - len(failures)}/{len(names)} modules")
sys.exit(1 if failures else 0)
EOF

echo "== every python package the tree has is in the clone"
MISSING="$(comm -13 \
  <(cd "$ROOT" && git ls-files 'python/**/*.py' 'python/*.py' | sort) \
  <(cd "$ROOT" && find python -name '*.py' -not -path '*/__pycache__/*' | sort))"
if [ -n "$MISSING" ]; then
  echo "$MISSING" | sed 's/^/   UNTRACKED /'
  FAILED=$((FAILED + 1))
else
  echo "   clean"
fi

echo "== wheel contents"
"$PY" -m pip wheel --quiet --no-deps --wheel-dir "$WORK/wheel" "$CLONE" 2>"$WORK/wheel.err" || {
  echo "   build failed: $(tail -3 "$WORK/wheel.err")"
  FAILED=$((FAILED + 1))
}
WHEEL="$(ls "$WORK"/wheel/*.whl 2>/dev/null | head -1)"
if [ -n "$WHEEL" ]; then
  "$PY" - "$WHEEL" <<'EOF'
import sys
import zipfile

names = zipfile.ZipFile(sys.argv[1]).namelist()
data = sorted(n for n in names if "/data/" in n)
print(f"   {len(names)} entries, data package files={len(data)}")
for entry in data:
    print(f"   {entry}")
sys.exit(0 if data else 1)
EOF
else
  FAILED=$((FAILED + 1))
fi

echo "clone check finished with $FAILED failing step(s)"
exit $((FAILED > 0))
