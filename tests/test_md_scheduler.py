"""Zero-memory validation of the opt-in `scheduler=` knob on the MultiDiffusion
drivers (t2v_multidiff / t2i_multidiff), which reuses the base
pipeline's DPMSolverPlusPlus2M (PR #4) at the global blended-velocity step.

These tests load NO model weights — they validate the *integration mechanics*
on a closed-form linear ODE, which is the part PR #4's "validate at your scale"
caveat does not cover: (1) DPM is a correct higher-order integrator in the
flow-matching velocity form over the real shifted schedule; (2) feeding the
solver a TAPER-BLENDED velocity (the MD step) is well-defined and reproduces a
single-field integration when the blend reconstructs the same field; (3) the
dispatch is byte-identical to legacy Euler at the default.

The perceptual model A/B (does DPM hurt MD output given cfg_renorm channel +
overlap-blend) is intentionally NOT here — it needs the real towers and a
monitored memory window.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from lance_mlx.model.flow_head import timestep_schedule
from lance_mlx.scheduler.solvers import DPMSolverPlusPlus2M


# --------------------------------------------------------------------------- #
# A closed-form linear test ODE matching the loop's update convention.
#
# The MD loop integrates  x_{i+1} = x_i - v_i * dt_i  with t descending 1 -> 0
# and dt_i = sched[i] - sched[i+1] > 0, sum(dt_i) = 1. With v(x) = a*x this is
# forward integration of dx/dtau = -a*x over tau in [0, 1], whose exact solution
# is x(1) = x0 * exp(-a). We use that as the oracle.
# --------------------------------------------------------------------------- #

A_RATE = 1.5  # ODE stiffness; smooth enough that AB2 should beat Euler.


def _euler_final(x0, sched, num_steps):
    """Plain Euler loop (the legacy MD step), v(x) = A_RATE * x."""
    x = x0
    for i in range(num_steps):
        dt = sched[i] - sched[i + 1]
        x = x - (A_RATE * x) * dt
    return x


def _dpm_final(x0, sched, num_steps):
    """DPM loop via solver.step — the exact dispatch the drivers now use."""
    solver = DPMSolverPlusPlus2M()
    x = x0
    for i in range(num_steps):
        dt = sched[i] - sched[i + 1]
        v = A_RATE * x
        x = solver.step(v, x, dt)
    return x


def _abs_err(x_final):
    oracle = math.exp(-A_RATE)  # x0 = 1.0
    return abs(float(x_final) - oracle)


# --------------------------------------------------------------------------- #
# 1. DPM is a CORRECT, higher-order integrator in the flow-matching form.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("num_steps", [8, 10, 12, 16])
def test_dpm_beats_euler_on_linear_ode(num_steps):
    """At a fixed (low) step count DPM's error is no worse than Euler, and
    strictly better in the regime PR #4 targets (~12 steps)."""
    sched = timestep_schedule(num_steps=num_steps, shift=3.5)
    x0 = mx.ones((1, 4, 4, 8), dtype=mx.float32)
    e_euler = _abs_err(mx.mean(_euler_final(x0, sched, num_steps)))
    e_dpm = _abs_err(mx.mean(_dpm_final(x0, sched, num_steps)))
    # DPM must not be worse; allow a hair of fp slack.
    assert e_dpm <= e_euler + 1e-6, f"steps={num_steps}: dpm {e_dpm} > euler {e_euler}"


def test_dpm_strictly_better_at_twelve_steps():
    """The PR #4 operating point: 12 steps. DPM should be clearly more accurate."""
    sched = timestep_schedule(num_steps=12, shift=3.5)
    x0 = mx.ones((1, 8), dtype=mx.float32)
    e_euler = _abs_err(mx.mean(_euler_final(x0, sched, 12)))
    e_dpm = _abs_err(mx.mean(_dpm_final(x0, sched, 12)))
    assert e_dpm < e_euler, f"dpm {e_dpm} !< euler {e_euler}"


def test_both_converge_as_steps_increase():
    """Sanity: both integrators converge to the oracle; neither diverges on the
    non-uniform shifted schedule (variable-dt AB2 ratio r = dt_prev/dt stable)."""
    x0 = mx.ones((1, 8), dtype=mx.float32)
    prev_e_euler = prev_e_dpm = None
    for num_steps in (8, 16, 32, 64):
        sched = timestep_schedule(num_steps=num_steps, shift=3.5)
        e_euler = _abs_err(mx.mean(_euler_final(x0, sched, num_steps)))
        e_dpm = _abs_err(mx.mean(_dpm_final(x0, sched, num_steps)))
        assert math.isfinite(e_euler) and math.isfinite(e_dpm)
        if prev_e_euler is not None:
            assert e_euler <= prev_e_euler + 1e-7
            assert e_dpm <= prev_e_dpm + 1e-7
        prev_e_euler, prev_e_dpm = e_euler, e_dpm
    # At 64 steps both are essentially exact.
    assert e_dpm < 1e-3 and e_euler < 1e-2


# --------------------------------------------------------------------------- #
# 2. First step is an Euler warm-up — so DPM == Euler when only one step runs
#    (e.g. an anchor-seeded MD tail that starts near the end of the schedule).
# --------------------------------------------------------------------------- #

def test_dpm_first_step_is_euler():
    solver = DPMSolverPlusPlus2M()
    x = mx.array([1.0, 2.0, -3.0], dtype=mx.float32)
    v = mx.array([0.5, -1.0, 0.25], dtype=mx.float32)
    dt = 0.137
    got = solver.step(v, x, dt)
    want = x - v * dt
    assert float(mx.max(mx.abs(got - want))) == 0.0


# --------------------------------------------------------------------------- #
# 3. The MD blend point: feeding the solver a TAPER-BLENDED velocity that
#    reconstructs the true field yields an identical trajectory to feeding the
#    true field directly. This is what makes threading the solver at MD's
#    `velocity = v_accum * inv_counts` step well-defined.
# --------------------------------------------------------------------------- #

def _blended_velocity(x):
    """Mimic the MD global step: K overlapping windows each contribute a
    taper-weighted slice of v(x)=A*x; summed and normalised by inv_counts they
    reconstruct the full field exactly (this is the MD partition-of-unity)."""
    v_true = A_RATE * x
    n = x.shape[-1]
    # two overlapping windows over the last axis with a linear taper in overlap.
    w = np.zeros((2, n), dtype=np.float32)
    w[0, : n // 2 + 1] = 1.0
    w[1, n // 2 - 1:] = 1.0
    counts = w.sum(0)
    inv = mx.array((1.0 / counts), dtype=mx.float32)
    accum = mx.zeros_like(x)
    for k in range(2):
        accum = accum + v_true * mx.array(w[k])
    return accum * inv  # == v_true (partition of unity)


def test_blended_velocity_matches_direct_under_dpm():
    sched = timestep_schedule(num_steps=12, shift=3.5)
    x0 = mx.ones((1, 16), dtype=mx.float32)

    # direct-field DPM loop
    s_direct = DPMSolverPlusPlus2M()
    xd = x0
    for i in range(12):
        dt = sched[i] - sched[i + 1]
        xd = s_direct.step(A_RATE * xd, xd, dt)

    # blended-field DPM loop (the MD mechanism)
    s_blend = DPMSolverPlusPlus2M()
    xb = x0
    for i in range(12):
        dt = sched[i] - sched[i + 1]
        xb = s_blend.step(_blended_velocity(xb), xb, dt)

    assert float(mx.max(mx.abs(xd - xb))) < 1e-5


# --------------------------------------------------------------------------- #
# 4. Default (euler) dispatch is byte-identical to the legacy MD step, and the
#    driver branch (`solver.step` vs `- v*dt`) reproduces the right path.
# --------------------------------------------------------------------------- #

def _md_dispatch(solver, velocity, latents, dt):
    """Exact copy of the branch threaded into the MD drivers."""
    if solver is not None:
        return solver.step(velocity, latents, dt)
    return latents - velocity * dt


def test_euler_default_is_byte_identical():
    x = mx.random.normal((1, 3, 8, 8, 4))
    v = mx.random.normal((1, 3, 8, 8, 4))
    dt = 0.091
    legacy = x - v * dt
    dispatched = _md_dispatch(None, v, x, dt)  # scheduler="euler" -> solver is None
    assert float(mx.max(mx.abs(legacy - dispatched))) == 0.0


def test_dpm_dispatch_diverges_after_warmup():
    """With a solver, step 1 == euler but step 2+ differs (AB2 extrapolation),
    confirming the dpm branch is actually engaged."""
    solver = DPMSolverPlusPlus2M()
    x = mx.ones((1, 8), dtype=mx.float32)
    sched = timestep_schedule(num_steps=12, shift=3.5)
    # step 1
    v1 = A_RATE * x
    x1_dpm = _md_dispatch(solver, v1, x, float(sched[0] - sched[1]))
    x1_eul = x - v1 * float(sched[0] - sched[1])
    assert float(mx.max(mx.abs(x1_dpm - x1_eul))) == 0.0
    # step 2 should differ
    v2 = A_RATE * x1_dpm
    x2_dpm = _md_dispatch(solver, v2, x1_dpm, float(sched[1] - sched[2]))
    x2_eul = x1_dpm - v2 * float(sched[1] - sched[2])
    assert float(mx.max(mx.abs(x2_dpm - x2_eul))) > 1e-6


# --------------------------------------------------------------------------- #
# 5. Solver reset behavior: a fresh solver starts with an Euler warm-up.
# --------------------------------------------------------------------------- #

def test_fresh_solver_per_window_resets_history():
    # Run one "window" to load history.
    s = DPMSolverPlusPlus2M()
    x = mx.ones((1, 8), dtype=mx.float32)
    s.step(A_RATE * x, x, 0.1)
    s.step(A_RATE * x, x, 0.1)
    assert s._v_prev is not None  # history present
    # A NEW window uses a fresh solver -> first step is euler again.
    s2 = DPMSolverPlusPlus2M()
    assert s2._v_prev is None
    v = mx.array([1.0, -2.0], dtype=mx.float32)
    xw = mx.array([0.3, 0.7], dtype=mx.float32)
    got = s2.step(v, xw, 0.2)
    assert float(mx.max(mx.abs(got - (xw - v * 0.2)))) == 0.0


def test_explicit_reset_equals_fresh():
    s = DPMSolverPlusPlus2M()
    x = mx.ones((1, 4), dtype=mx.float32)
    s.step(A_RATE * x, x, 0.1)
    s.reset()
    assert s._v_prev is None and s._dt_prev is None


# --------------------------------------------------------------------------- #
# 6. The drivers validate the scheduler kwarg (reusing PR #4's contract).
# --------------------------------------------------------------------------- #

def test_drivers_reject_unknown_scheduler():
    # Pure-Python guard fires before any model work — safe to call with a dummy.
    from lance_mlx.pipeline import t2v_multidiff, t2i_multidiff
    with pytest.raises(ValueError, match="scheduler"):
        t2v_multidiff.generate_multidiff(None, "x", total_frames=5, scheduler="rk4")
    with pytest.raises(ValueError, match="scheduler"):
        t2i_multidiff.generate_multidiff_spatial(
            None, "x", height=1024, width=1024, scheduler="rk4")


def test_window_prompts_must_match_window_count():
    """Per-phase prompts are one prompt per MD window. Mismatched schedules
    should fail before any model work so long-run drivers do not waste time."""
    from lance_mlx.pipeline import t2v_multidiff
    with pytest.raises(ValueError, match="window_prompts length"):
        t2v_multidiff.generate_multidiff(
            None, "x", total_frames=17, window_frames=17,
            overlap_lat=2, window_prompts=["early", "late"])
