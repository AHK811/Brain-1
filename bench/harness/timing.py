from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Timing:
    samples_ms: tuple[float, ...]
    median_ms: float
    min_ms: float
    p90_ms: float
    mad_pct: float
    reps: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "ms_median": self.median_ms,
            "ms_min": self.min_ms,
            "ms_p90": self.p90_ms,
            "ms_mad_pct": self.mad_pct,
            "reps": self.reps,
            "samples_ms": list(self.samples_ms),
        }


def summarize(samples_ms: Sequence[float]) -> Timing:
    if not samples_ms:
        raise ValueError("no timing samples")
    ordered = sorted(float(sample) for sample in samples_ms)
    middle = statistics.median(ordered)
    deviations = statistics.median([abs(sample - middle) for sample in ordered]) if len(ordered) > 2 else 0.0
    index = max(0, min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1)))))
    return Timing(
        samples_ms=tuple(ordered),
        median_ms=middle,
        min_ms=ordered[0],
        p90_ms=ordered[index],
        mad_pct=(deviations / middle * 100.0) if middle > 0 else 0.0,
        reps=len(ordered),
    )


def synchronize(device: str) -> None:
    if device != "cuda":
        return
    import torch

    torch.cuda.synchronize()


def time_calls(
    call: Callable[[], object],
    reps: int,
    warmup: int,
    device: str = "cpu",
) -> Timing:
    if reps < 1:
        raise ValueError("reps must be at least 1")
    import torch

    for _ in range(warmup):
        call()
    synchronize(device)
    if device == "cuda":
        samples: list[float] = []
        for _ in range(reps):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            stop.record()
            stop.synchronize()
            samples.append(float(start.elapsed_time(stop)))
        return summarize(samples)
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000.0)
    return summarize(samples)


def probe_cost(call: Callable[[], object], device: str) -> float:
    import torch

    synchronize(device)
    started = time.perf_counter()
    call()
    synchronize(device)
    return (time.perf_counter() - started) * 1000.0


def choose_reps(estimate_ms: float, min_reps: int, max_reps: int, budget_ms: float) -> int:
    if estimate_ms <= 0:
        return max(min_reps, min(max_reps, 50))
    fitted = int(budget_ms / estimate_ms)
    return max(min_reps, min(max_reps, fitted))


def warmup_for(estimate_ms: float, minimum: int, budget_ms: float) -> int:
    if estimate_ms <= 0:
        return minimum
    return max(minimum, int(min(50.0, budget_ms / estimate_ms)))
