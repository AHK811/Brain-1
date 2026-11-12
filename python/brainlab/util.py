from __future__ import annotations

import json
import os
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def human_bytes(count: int | float) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:,.1f} TB"


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median of empty sequence")
    return float(statistics.median(values))


def robust_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("stats of empty sequence")
    ordered = sorted(float(v) for v in values)
    return {
        "min": ordered[0],
        "median": float(statistics.median(ordered)),
        "mean": float(statistics.fmean(ordered)),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1) + 0.5))],
        "max": ordered[-1],
        "mad": float(statistics.median([abs(v - statistics.median(ordered)) for v in ordered])),
    }


def rel_l2(actual: Any, expected: Any) -> float:
    import numpy as np

    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    denom = float(np.sqrt((b * b).sum()))
    num = float(np.sqrt(((a - b) ** 2).sum()))
    return num / denom if denom > 0 else num


def write_json(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
