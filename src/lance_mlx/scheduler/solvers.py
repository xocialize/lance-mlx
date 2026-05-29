"""ODE solvers for Lance flow-matching inference.

DPMSolverPlusPlus2M reduces the number of model evaluations from ~30 to
~12 with comparable output quality, giving 1.6–2.8× faster generation
(avg 1.95× on a 4-prompt eval at 768², seed=42).

Usage:
    from lance_mlx.scheduler.solvers import DPMSolverPlusPlus2M
    solver = DPMSolverPlusPlus2M()
    # Inside the flow loop, replace: latents = latents - velocity * dt
    latents = solver.step(velocity, latents, dt)
"""

from __future__ import annotations

import mlx.core as mx


class DPMSolverPlusPlus2M:
    """Variable-step Adams-Bashforth 2 applied to the flow-matching velocity ODE.

    This is the flow-matching adaptation of the DPM-Solver++(2M) multistep
    idea (Lu et al. 2022, https://arxiv.org/abs/2211.01095), NOT the exact
    published algorithm.  The published formula operates on noise-prediction or
    x-prediction diffusion ODEs with log-SNR substitution; here we apply the
    same 2-step Adams-Bashforth extrapolation directly to dx/dt = -v(x, t),
    which is the correct form for flow-matching models like Lance.

    First step falls back to Euler (no previous velocity available).
    Subsequent steps use the 2-step Adams-Bashforth formula, which
    achieves second-order accuracy with one model evaluation per step.

    Validated at 12 steps against the 30-step Euler oracle on Lance-3B-bf16
    t2i (768², seed=42): 1.6–2.8× faster (avg 1.95×) with visually equivalent
    output. Speedup varies with prompt-token length and CFG complexity.
    """

    def __init__(self) -> None:
        self._v_prev: mx.array | None = None
        self._dt_prev: float | None = None

    def reset(self) -> None:
        """Reset state between generate() calls."""
        self._v_prev = None
        self._dt_prev = None

    def step(
        self,
        velocity: mx.array,
        latents: mx.array,
        dt: float,
    ) -> mx.array:
        """Advance latents by one step.

        Args:
            velocity: model velocity output at current timestep, same shape as latents.
            latents:  current noisy latents.
            dt:       timestep delta (t_current - t_next), positive scalar.

        Returns:
            Updated latents after one solver step.
        """
        if self._v_prev is None:
            # Euler warm-up on first step.
            x_next = latents - velocity * dt
        else:
            # Adams-Bashforth 2-step: extrapolate using current and previous velocity.
            r = self._dt_prev / dt
            coeff_curr = 1.0 + 1.0 / (2.0 * r)
            coeff_prev = 1.0 / (2.0 * r)
            x_next = latents - dt * (coeff_curr * velocity - coeff_prev * self._v_prev)

        self._v_prev = velocity
        self._dt_prev = dt
        return x_next
