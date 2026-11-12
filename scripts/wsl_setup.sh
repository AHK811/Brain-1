#!/usr/bin/env bash
set -euo pipefail

export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export PATH="$CARGO_HOME/bin:$PATH"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/brainlab-target}"

if ! command -v cc >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq || sudo apt-get update -qq
    apt-get install -y -qq build-essential curl ca-certificates >/dev/null || \
      sudo apt-get install -y -qq build-essential curl ca-certificates
  else
    echo "no cc and no apt-get: install a C compiler first" >&2
    exit 1
  fi
fi

if ! command -v cargo >/dev/null 2>&1; then
  command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 1; }
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
fi

source "$CARGO_HOME/env"
rustup show active-toolchain || rustup toolchain install stable
cargo --version
rustc --version
nproc
free -g | awk 'NR==2 {print "mem_total_gb=" $2}'
mkdir -p "$CARGO_TARGET_DIR"
echo "CARGO_TARGET_DIR=$CARGO_TARGET_DIR"

if [ "${WITH_PYTHON:-0}" = "1" ]; then
  VENV="${BRAINLAB_VENV:-$HOME/brainlab-venv}"
  ROOT="${BRAINLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  needs_apt=0
  python3 -m venv --help >/dev/null 2>&1 || needs_apt=1
  python3 -c "import ensurepip" >/dev/null 2>&1 || needs_apt=1
  python3 -m pip -V >/dev/null 2>&1 || needs_apt=1
  ls /usr/include/python3.*/Python.h >/dev/null 2>&1 || needs_apt=1
  if [ "$needs_apt" = "1" ]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get install -y -qq python3-venv python3-pip python3-dev >/dev/null 2>&1 || \
      sudo -n apt-get install -y -qq python3-venv python3-pip python3-dev >/dev/null 2>&1 || \
      echo "apt could not install the python packages; continuing with what is present" >&2
  fi
  if ! python3 -m venv --help >/dev/null 2>&1 || ! python3 -c "import ensurepip" >/dev/null 2>&1; then
    echo "python3 -m venv is unavailable and apt could not install python3-venv; re-run with sudo" >&2
    exit 1
  fi
  if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import pip" >/dev/null 2>&1; then
    rm -rf "$VENV"
    python3 -m venv "$VENV"
  fi
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch
  "$VENV/bin/python" -m pip install --quiet -e "$ROOT[dev]"
  "$VENV/bin/python" -c "import numpy, torch, safetensors; print('venv', torch.__version__, numpy.__version__)"
  echo "BRAINLAB_VENV=$VENV"
fi
