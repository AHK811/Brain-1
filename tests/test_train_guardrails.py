from __future__ import annotations

import pytest

from brainlab.train.train import divergence_reason

CALM = dict(
    mean_loss=4.2,
    grad_norm=12.5,
    skipped=False,
    consecutive_skips=0,
    zero_grad_streak=0,
    scale=65536.0,
    amp_enabled=True,
    muon_skips=0,
    skip_limit=20,
    zero_grad_limit=3,
    scale_floor=1.0,
)


def held_out(**overrides) -> str:
    return divergence_reason(**{**CALM, **overrides})


def test_a_healthy_step_reports_nothing() -> None:
    assert held_out() == ""
    assert held_out(amp_enabled=False, scale=1.0) == ""


def test_transient_inf_grads_during_scaling_are_not_divergence() -> None:
    assert held_out(skipped=True, grad_norm=float("inf"), mean_loss=float("inf")) == ""
    assert held_out(consecutive_skips=5) == ""


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"skipped": False, "mean_loss": float("nan")}, "non-finite loss"),
        ({"skipped": False, "grad_norm": float("inf")}, "non-finite gradient norm"),
        ({"consecutive_skips": 20}, "consecutive steps"),
        ({"scale": 0.5}, "loss scale collapsed"),
        ({"amp_enabled": False, "muon_skips": 1}, "non-finite updates"),
        ({"zero_grad_streak": 3, "grad_norm": 0.0}, "exactly zero"),
    ],
)
def test_each_failure_mode_has_a_name(overrides: dict, fragment: str) -> None:
    reason = held_out(**overrides)
    assert fragment in reason, f"{overrides} produced {reason!r}"


def test_the_zero_gradient_collapse_that_looks_healthy_is_caught() -> None:
    assert held_out(amp_enabled=False, scale=1.0, grad_norm=0.0, mean_loss=9.704061, zero_grad_streak=2) == ""
    assert "exactly zero" in held_out(
        amp_enabled=False, scale=1.0, grad_norm=0.0, mean_loss=9.704061, zero_grad_streak=3
    )


def test_amp_is_required_for_the_scale_floor_but_not_for_the_rest() -> None:
    assert held_out(amp_enabled=False, scale=0.001) == ""
    assert "collapsed" in held_out(amp_enabled=True, scale=0.001)
