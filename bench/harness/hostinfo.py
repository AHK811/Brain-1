from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from bench.harness.peak import nvidia_smi_fields


def module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def command_text(argv: list[str], timeout: int = 15) -> str:
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error!r}"
    if completed.returncode != 0:
        return f"exit {completed.returncode}: {completed.stderr.strip()[:200]}"
    return completed.stdout.strip()[:400]


def read_status(root: Path, name: str) -> dict[str, Any]:
    path = root / "bench" / "results" / name
    if not path.is_file():
        return {"missing": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return {"error": str(error), "path": str(path)}


def collect(root: Path, device: str, with_cuda: bool) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "torch": module_version("torch"),
        "numpy": module_version("numpy"),
        "triton": module_version("triton"),
        "safetensors": module_version("safetensors"),
        "device_requested": device,
    }
    smi = nvidia_smi_fields()
    if smi:
        info["nvidia_smi"] = smi
    if with_cuda:
        import torch

        properties = torch.cuda.get_device_properties(0)
        info["cuda"] = {
            "name": properties.name,
            "capability": f"sm_{properties.major}{properties.minor}",
            "multi_processor_count": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
            "torch_cuda_version": torch.version.cuda or "unknown",
        }
    info["rust"] = command_text(["cargo", "--version"])
    info["mojo"] = read_status(root, "mojo_status.json")
    return info
