from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

NP_DTYPE = {"fp32": np.float32, "fp16": np.float16}


def write_raw_bundle(out_dir: Path, arrays: dict[str, np.ndarray], dtype: str, meta: dict[str, Any]) -> Path:
    if dtype not in NP_DTYPE:
        raise ValueError(f"dtype {dtype} has no raw bridge; only {sorted(NP_DTYPE)} are portable")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, Any]] = {"__meta__": meta}
    for name, arr in arrays.items():
        target = out_dir / f"{name}.{dtype}.bin"
        contiguous = np.ascontiguousarray(arr.astype(NP_DTYPE[dtype], copy=False))
        contiguous.tofile(target)
        manifest[name] = {
            "dtype": dtype,
            "shape": list(contiguous.shape),
            "nbytes": contiguous.nbytes,
            "file": target.name,
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_dir


def read_raw(path: Path, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.fromfile(path, dtype=NP_DTYPE[dtype])
    count = int(np.prod(shape)) if shape else 1
    if arr.size != count:
        raise ValueError(f"{path}: expected {count} elements for shape {shape}, found {arr.size}")
    return arr.reshape(shape)


def load_manifest(out_dir: Path) -> dict[str, Any]:
    return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))


def manifest_entries(manifest: dict[str, Any]) -> dict[str, tuple[str, tuple[int, ...]]]:
    return {
        name: (entry["dtype"], tuple(entry["shape"]))
        for name, entry in manifest.items()
        if name != "__meta__"
    }


def load_golden(golden_path: Path) -> dict[str, np.ndarray]:
    return load_file(str(golden_path))
