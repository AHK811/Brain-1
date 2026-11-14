from __future__ import annotations

import math

import pytest

from brainlab.spec.config import load_raw_config
from brainlab.train.schedule import LRSchedule, steps_for_tokens, tokens_per_step


def wsd(min_lr_fraction: float = 0.0, **kwargs) -> LRSchedule:
    return LRSchedule(peak_lr=0.02, warmup_steps=10, total_steps=100, min_lr_fraction=min_lr_fraction, decay="wsd", **kwargs)


def test_wsd_holds_the_peak_then_decays_linearly_to_zero() -> None:
    schedule = wsd()
    assert schedule.lr_at(0) == pytest.approx(0.002), "warmup starts at one warmup step of the peak"
    assert schedule.lr_at(9) == pytest.approx(0.02)
    assert all(schedule.lr_at(step) == pytest.approx(0.02) for step in range(10, 71)), (
        "the stable phase runs to stable_fraction of the run"
    )
    assert schedule.lr_at(85) == pytest.approx(0.02 * (1.0 - 15.0 / 30.0))
    assert schedule.lr_at(99) == pytest.approx(0.02 / 30.0)
    assert schedule.lr_at(100) == pytest.approx(0.0)
    assert schedule.factor_at(35) == 1.0 and schedule.factor_at(100) == pytest.approx(0.0)


def test_wsd_is_monotonic_after_warmup_and_never_exceeds_the_peak() -> None:
    schedule = wsd()
    values = [schedule.lr_at(step) for step in range(101)]
    assert values[:10] == sorted(values[:10]), "warmup rises"
    assert all(later <= earlier + 1e-12 for earlier, later in zip(values[10:-1], values[11:], strict=True))
    assert max(values) == pytest.approx(0.02)
    assert min(values) == pytest.approx(0.0)


def test_wsd_floor_is_the_peak_times_the_min_fraction() -> None:
    schedule = wsd(min_lr_fraction=0.1)
    assert schedule.lr_at(100) == pytest.approx(0.002)
    assert schedule.lr_at(70) == pytest.approx(0.02)


def test_a_longer_stable_fraction_defers_the_decay() -> None:
    early, late = wsd(stable_fraction=0.5), wsd(stable_fraction=0.9)
    assert early.lr_at(75) < late.lr_at(75), "the shorter stable phase has already decayed"
    assert late.lr_at(75) == pytest.approx(0.02)
    assert early.lr_at(50) == pytest.approx(0.02) == late.lr_at(50)


def test_the_other_decays_keep_their_contracts() -> None:
    steps = 100
    common = {"peak_lr": 1.0, "warmup_steps": 0, "total_steps": steps}
    cosine = LRSchedule(decay="cosine", min_lr_fraction=0.0, **common)
    assert cosine.lr_at(0) == pytest.approx(1.0)
    assert cosine.lr_at(50) == pytest.approx(0.5)
    assert cosine.lr_at(steps) == pytest.approx(0.0)
    linear = LRSchedule(decay="linear", min_lr_fraction=0.0, **common)
    assert linear.lr_at(25) == pytest.approx(0.75)
    assert linear.lr_at(steps) == pytest.approx(0.0)
    constant = LRSchedule(decay="constant", **common)
    assert all(constant.lr_at(step) == 1.0 for step in range(steps + 1))
    assert constant.factor_at(7) == 1.0


def test_warmup_shorter_than_total_and_a_positive_peak_are_enforced() -> None:
    for kwargs, message in (
        ({"peak_lr": 0.0}, "peak_lr must be finite and positive"),
        ({"total_steps": 0}, "total_steps must be positive"),
        ({"warmup_steps": -1}, "warmup_steps must be non-negative"),
        ({"warmup_steps": 100}, "shorter than total_steps"),
        ({"min_lr_fraction": 1.0}, "min_lr_fraction"),
        ({"min_lr_fraction": -0.1}, "min_lr_fraction"),
        ({"stable_fraction": 0.0}, "stable_fraction"),
        ({"stable_fraction": 1.0}, "stable_fraction"),
        ({"decay": "exponential"}, "unknown decay"),
        ({"decay": "wsd", "warmup_steps": 80}, "before the WSD decay phase"),
        ({"decay": "cosine", "stable_fraction": float("nan")}, "stable_fraction"),
        ({"decay": "wsd", "peak_lr": float("inf")}, "peak_lr"),
    ):
        base = {"peak_lr": 0.02, "warmup_steps": 10, "total_steps": 100, "min_lr_fraction": 0.0, "decay": "cosine"}
        with pytest.raises(ValueError, match=message):
            LRSchedule(**{**base, **kwargs})
    schedule = LRSchedule(peak_lr=0.02, warmup_steps=10, total_steps=100)
    with pytest.raises(ValueError, match="non-negative"):
        schedule.lr_at(-1)


def test_schedule_comes_from_the_configured_decay_not_a_default() -> None:
    raw = load_raw_config()
    schedule = LRSchedule.from_config(raw["schedule"], float(raw["optimizer"]["muon"]["lr"]))
    assert schedule.decay == "wsd"
    assert schedule.stable_fraction == pytest.approx(0.7)
    assert schedule.min_lr_fraction == pytest.approx(0.0)
    assert schedule.warmup_steps < schedule.total_steps * schedule.stable_fraction
    assert schedule.lr_at(schedule.total_steps) == pytest.approx(0.0)
    assert schedule.as_dict()["decay"] == "wsd"
    floored = LRSchedule.from_config({**raw["schedule"], "min_lr_fraction": 0.5}, 0.02)
    assert floored.lr_at(floored.total_steps) == pytest.approx(0.01), "the floor is a fraction of the peak"
    assert floored.decay == "wsd"


def test_step_and_token_arithmetic_rounds_up_a_partial_epoch() -> None:
    assert tokens_per_step(1024, 8, 4) == 1024 * 8 * 4
    assert tokens_per_step(1024, 8, 4, world_size=2) == 1024 * 8 * 4 * 2
    assert steps_for_tokens(1000, 8, 4, 3) == math.ceil(1000 / (8 * 4 * 3))
    assert steps_for_tokens(0, 8, 4, 3) == 1, "an empty corpus still yields one step rather than a divide-by-zero"
    assert steps_for_tokens(768, 8, 4, 3) == 8 == steps_for_tokens(769, 8, 4, 3) - 1
