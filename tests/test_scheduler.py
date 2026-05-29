"""Unit tests for the DPM-Solver++(2M) / Adams-Bashforth 2 scheduler.

Covers:
 - Backward-compat: first step is byte-identical to Euler (warm-up path)
 - Multi-step smoke: solver runs 12 steps without error, shape preserved
 - Invalid scheduler: generate() raises ValueError before touching any model
 - Second-order accuracy: solver error is O(dt²) on dx/dt = -kx
"""
from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from lance_mlx.scheduler.solvers import DPMSolverPlusPlus2M


# ── solver unit tests ──────────────────────────────────────────────────────────

def test_first_step_is_euler():
    """First step must be byte-identical to latents - velocity * dt (Euler)."""
    solver = DPMSolverPlusPlus2M()
    velocity = mx.array([1.0, 2.0, 3.0])
    latents = mx.array([10.0, 20.0, 30.0])
    dt = 0.1

    result = solver.step(velocity, latents, dt)
    expected = latents - velocity * dt
    mx.eval(result, expected)

    assert mx.allclose(result, expected).item(), (
        "First step must be Euler warm-up — no previous velocity available"
    )


def test_solver_runs_twelve_steps_shape_preserved():
    """Smoke: solver completes 12 steps without error; output shape matches input."""
    solver = DPMSolverPlusPlus2M()
    latents = mx.zeros((1, 2304, 48))
    dt = 1.0 / 12

    for _ in range(12):
        velocity = mx.ones_like(latents) * 0.5
        latents = solver.step(velocity, latents, dt)

    mx.eval(latents)
    assert latents.shape == (1, 2304, 48)


def test_solver_reset_clears_state():
    """reset() must restore first-step Euler behaviour."""
    solver = DPMSolverPlusPlus2M()
    v = mx.array([1.0])
    x = mx.array([5.0])

    solver.step(v, x, 0.1)   # primes _v_prev
    solver.reset()

    result = solver.step(v, x, 0.1)
    expected = x - v * 0.1
    mx.eval(result, expected)
    assert mx.allclose(result, expected).item()


@pytest.mark.parametrize("n_steps", [8, 16, 32])
def test_second_order_accuracy_linear_ode(n_steps: int):
    """Solver achieves second-order (O(dt²)) accuracy on dx/dt = -kx.

    Analytical solution: x(t) = x0 * exp(-k*t).
    We integrate from t=0 to t=T with n_steps uniform steps and compare
    the final value against the exact solution.  Doubling n_steps should
    reduce the error by ~4× (second-order), not ~2× (first-order Euler).
    """
    import math

    k = 2.0
    x0 = 1.0
    T = 1.0

    def _run(n: int) -> float:
        solver = DPMSolverPlusPlus2M()
        x = mx.array([x0])
        dt = T / n
        t = 0.0
        for _ in range(n):
            # velocity for dx/dt = -kx  →  v = k*x  (so x_next = x - v*dt)
            v = mx.array([k * float(x[0])])
            x = solver.step(v, x, dt)
            t += dt
        mx.eval(x)
        return float(x[0])

    exact = math.exp(-k * T)
    err_coarse = abs(_run(n_steps) - exact)
    err_fine   = abs(_run(n_steps * 2) - exact)

    # Adams-Bashforth 2 is second-order: halving dt should cut error by ~4×.
    # Allow a generous 2.5× threshold to account for the Euler warm-up step.
    assert err_coarse > 0, "coarse run should not hit machine precision"
    assert err_fine < err_coarse, "finer grid must reduce error"
    ratio = err_coarse / err_fine
    assert ratio > 2.5, (
        f"Expected >2.5× error reduction when doubling steps "
        f"(second-order), got {ratio:.2f}× at n_steps={n_steps}"
    )


# ── pipeline-level validation ──────────────────────────────────────────────────

def _make_pipeline():
    """Minimal TextToImagePipeline with mocked sub-components."""
    from lance_mlx.pipeline.t2i import TextToImagePipeline
    return TextToImagePipeline(
        lance_model=MagicMock(),
        vae_decoder=MagicMock(),
        processor=MagicMock(),
        text_config=MagicMock(),
        image_pad_token_id=0,
        video_pad_token_id=1,
        vision_start_token_id=2,
        vision_end_token_id=3,
    )


def test_invalid_scheduler_raises():
    """scheduler='foo' must raise ValueError with 'Unknown scheduler' in the message."""
    pipe = _make_pipeline()
    with pytest.raises(ValueError, match="Unknown scheduler"):
        pipe.generate("a red apple", scheduler="foo")
