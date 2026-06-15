"""Shared primitives for the MultiDiffusion drivers (t2v_multidiff, t2i_multidiff).

Each helper here used to live as a verbatim copy in two or more driver modules
(the t2i spatial driver was built by inverting the t2v temporal one, and both
copied the seeding/CFG blocks). This module is a pure extraction — every function
is byte-for-byte the operation sequence of the copies it replaces, so the drivers'
outputs are unchanged.

Everything is model-free (geometry, schedules, noise indexing, velocity algebra)
and unit-tested from tests/test_multidiff_windows.py + tests/test_t2i_multidiff.py.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


# ------------------------------------------------------------------ windowing

def window_starts(total: int, win: int, overlap: int) -> list[int]:
    """Uniform-size window starts covering [0, total) along ONE axis (latent
    frames for t2v, latent rows/cols for t2i).

    Every window is exactly `win` cells so all windows share ONE prompt template
    and ONE prefix cache. The last window is pulled back to end exactly at
    `total` (it may overlap its predecessor by more than `overlap` — the blend
    handles uneven overlap fine). Requires total >= win.
    """
    if win > total:
        raise ValueError(f"win={win} > total={total}; lower window size or raise extent")
    stride = win - overlap
    if stride < 1:
        raise ValueError(f"overlap={overlap} >= win={win} (stride<1)")
    starts: list[int] = []
    s = 0
    while True:
        s_clamped = min(s, total - win)
        if not starts or s_clamped != starts[-1]:
            starts.append(s_clamped)
        if s_clamped >= total - win:
            break
        s += stride
    return starts


def block_shuffle_idx(total: int, win: int, rng: np.random.Generator) -> np.ndarray:
    """FreeNoise tiling index for one axis: [0..win) verbatim for the first
    block, then an independent permutation of the base block per subsequent
    block (truncated at the end). Gathering a base noise block with this index
    gives every position past the first window a value FROM the base vocabulary
    (shared across the whole extent) while the per-block shuffle keeps blocks
    non-identical (no static/looping repeat). Consumes `rng` once per extra
    block — callers that index multiple axes share one rng in axis order."""
    idx = np.empty(total, dtype=np.int32)
    idx[:min(win, total)] = np.arange(min(win, total))
    for start in range(win, total, win):
        end = min(start + win, total)
        idx[start:end] = rng.permutation(win)[: end - start]
    return idx


def half_pixel_coords(out: int, src: int) -> np.ndarray:
    """align_corners=False source coordinates for resampling `src` samples to
    `out`, clipped so the first/last outputs land exactly on the first/last
    sources. The shared formula behind both anchor upsamplers (spatial in
    t2i_multidiff, temporal in t2v_multidiff)."""
    return np.clip((np.arange(out) + 0.5) * src / out - 0.5, 0, src - 1)


# ----------------------------------------------------------- anchor (B7) seed

def img2img_init(anchor: mx.array, noise: mx.array, sched, num_steps: int,
                 t_start: float) -> tuple[mx.array, int]:
    """SDEdit seed: re-noise a clean anchor z0 to a schedule time and pick step0.

    step0 = first schedule step whose t falls at/below `t_start` (so the loop runs
    range(step0, num_steps)). We re-noise the anchor to that step's EXACT time
    t0=sched[step0] using the given noise: rectified-flow's forward path
    z_t=(1-t)*z0 + t*eps is the exact inverse of the loop's z=z-v*dt update, so
    re-noising to a real schedule value keeps the truncated loop self-consistent
    (no schedule mismatch). Lower t_start -> later step0 -> fewer refine steps ->
    anchor dominates. Returns (seeded_latents, step0).
    """
    step0 = next((i for i in range(num_steps) if float(sched[i]) <= t_start),
                 num_steps - 1)
    t0 = float(sched[step0])
    seeded = (1.0 - t0) * anchor.astype(noise.dtype) + t0 * noise
    return seeded, step0


def img2img_dense(anchor: mx.array, noise: mx.array, sched, t_start: float):
    """Dense-tail img2img seed: re-noise the anchor to EXACTLY t_start and return a
    FRESH full-resolution schedule over [t_start, 0].

    `img2img_init` (tail mode) runs only the few original-schedule steps that fall
    below t_start — under shift=3.5 step-clustering near t=1 that is ~7-13 of 30
    steps. Here we instead scale the standard [1,0] schedule by t_start so all
    num_steps model evals land inside [0, t_start] (clustering shape preserved),
    integrating the SAME ODE from t_start to 0 with full resolution. The B7b A/B
    showed dense ≈ tail on every metric (the residual softness is the MD
    overlap-blend floor, not a step-count artifact), so "tail" stays the default
    and this is the documented knob. Returns (seeded, loop_sched); the caller
    runs the full range(num_steps) over loop_sched.
    """
    loop_sched = t_start * sched                       # [1,0] -> [t_start,0]
    seeded = (1.0 - t_start) * anchor.astype(noise.dtype) + t_start * noise
    return seeded, loop_sched


# ------------------------------------------------------------------------ CFG

def cfg_window(cfg_interval: tuple[float, float] | None) -> tuple[float, float]:
    """(lo, hi) gate for per-step CFG activation; None -> always active."""
    if cfg_interval is None:
        return float("-inf"), float("inf")
    return float(cfg_interval[0]), float(cfg_interval[1])


def cfg_combine(v_cond: mx.array, v_uncond: mx.array, cfg_scale: float,
                renorm_type: str, renorm_min: float) -> mx.array:
    """Classifier-free-guidance combine + renorm — the exact block from
    `TextToVideoPipeline.generate` (and t2i's), shared by every per-window
    velocity path. "global" renorms by the scalar L2 ratio, "channel" per-channel
    (the production default; see t2v.generate's cfg_renorm_type note), anything
    else returns the raw CFG combination."""
    v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
    if renorm_type == "global":
        ratio = mx.sqrt(mx.sum(v_cond * v_cond)) / (mx.sqrt(mx.sum(v_cfg * v_cfg)) + 1e-8)
        return v_cfg * mx.clip(ratio, renorm_min, 1.0)
    if renorm_type == "channel":
        nc = mx.sqrt(mx.sum(v_cond * v_cond, axis=-1, keepdims=True))
        nf = mx.sqrt(mx.sum(v_cfg * v_cfg, axis=-1, keepdims=True))
        return v_cfg * mx.clip(nc / (nf + 1e-8), renorm_min, 1.0)
    return v_cfg
