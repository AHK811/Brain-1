from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from brainlab.spec.config import load_model_spec, repo_root
from brainlab.spec.workloads import WORKLOADS, Workload, select

GOLDEN_DIR = ROOT / "tests" / "golden"


def missing_goldens(directory: Path, include_raw: bool = False) -> list[str]:
    records = json.loads((directory / "index.json").read_text(encoding="utf-8"))["workloads"]
    missing = [record["file"] for record in records if not (directory / record["file"]).is_file()]
    if not include_raw:
        return missing
    for record in records:
        if record["brainlab.op"] == "forward":
            continue
        bundle = directory / "raw" / record["brainlab.key"]
        manifest = bundle / "manifest.json"
        if not manifest.is_file():
            missing.append(str(manifest.relative_to(directory)))
            continue
        tensors = json.loads(manifest.read_text(encoding="utf-8"))
        for name, entry in tensors.items():
            if name == "__meta__":
                continue
            blob = bundle / entry["file"]
            if not blob.is_file():
                missing.append(str(blob.relative_to(directory)))
    return missing


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    index = GOLDEN_DIR / "index.json"
    if not index.is_file():
        pytest.skip(f"{index} missing; run `just golden`")
    missing = missing_goldens(GOLDEN_DIR)
    if missing:
        pytest.skip(
            f"{len(missing)} golden tensors are missing, for example {missing[0]}; "
            "the index is committed but the tensors are not - run `just golden`"
        )
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
