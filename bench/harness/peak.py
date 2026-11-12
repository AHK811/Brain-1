from __future__ import annotations

import subprocess
from dataclasses import dataclass

KNOWN_PEAK_GBPS = {
    "NVIDIA Tesla T4": 320.0,
    "Tesla T4": 320.0,
    "T4": 320.0,
    "A100-SXM4-40GB": 1555.0,
    "A100-SXM4-80GB": 2039.0,
    "A100-PCIE-80GB": 1935.0,
    "A100": 1555.0,
    "A10G": 600.0,
    "A10 ": 600.0,
    "L4": 300.0,
    "L40S": 864.0,
    "RTX 4000 Ada": 180.0,
    "RTX 3090": 936.0,
    "RTX 4090": 1008.0,
}


@dataclass(frozen=True)
class Peak:
    value: float | None
    source: str
    device: str


def nvidia_smi_fields() -> dict[str, str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if completed.returncode != 0:
        return {}
    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [value.strip() for value in line.split(",")]
    keys = ("name", "driver_version", "memory_total_mib", "max_memory_clock_mhz")
    return dict(zip(keys, parts))


def peak_for(device_name: str, override: float | None = None) -> Peak:
    if override is not None and override > 0:
        return Peak(value=override, source="cli-override", device=device_name)
    for key, value in KNOWN_PEAK_GBPS.items():
        if key.lower() in device_name.lower():
            return Peak(value=value, source="known-spec", device=device_name)
    return Peak(value=None, source="unknown", device=device_name)


def derive_from_clocks(max_memory_clock_mhz: str | None, bus_bits: int | None, multiplier: float = 2.0) -> float | None:
    if not max_memory_clock_mhz or not bus_bits:
        return None
    try:
        clock = float(max_memory_clock_mhz)
    except ValueError:
        return None
    return clock * 1e6 * multiplier * bus_bits / 8.0 / 1e9
