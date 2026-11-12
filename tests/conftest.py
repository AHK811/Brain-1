from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from brainlab.spec.config import load_model_spec, repo_root
from brainlab.spec.workloads import WORKLOADS, Workload, select

GOLDEN_DIR = ROOT / "tests" / "golden"


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    if not (GOLDEN_DIR / "index.json").is_file():
        pytest.skip(f"{GOLDEN_DIR / 'index.json'} missing; run `just golden`")
    return GOLDEN_DIR


@pytest.fixture(scope="session")
def golden_index(golden_dir: Path) -> dict:
    return json.loads((golden_dir / "index.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def spec():
    return load_model_spec()


@pytest.fixture(scope="session")
def root() -> Path:
    return repo_root()


def workload_keys(op: str, dtype: str | None = None) -> list[str]:
    return [item.key for item in select([op], [dtype] if dtype else None)]


def get_workload(key: str) -> Workload:
    return next(item for item in WORKLOADS if item.key == key)
