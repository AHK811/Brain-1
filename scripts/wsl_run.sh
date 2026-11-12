#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: wsl_run.sh script.sh [args...]" >&2
  exit 2
fi

SCRIPT="$1"
shift
DIRECTORY="$(cd "$(dirname "$SCRIPT")" && pwd)"
export BRAINLAB_ROOT="$(cd "$DIRECTORY/.." && pwd)"
COPY="/tmp/brainlab-wsl-run-$(basename "$SCRIPT")"
tr -d '\r' < "$SCRIPT" > "$COPY"
bash "$COPY" "$@"
