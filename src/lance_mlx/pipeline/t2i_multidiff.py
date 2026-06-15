"""t2i_multidiff — high-resolution images on Lance via SPATIAL MultiDiffusion.

The single-shot t2i ceiling is 1024² (64×64 latent) because the learned positional
embedding table `latent_pos_embed.pos_embed` is physically [4096,2048] = 64×64. Any
larger image needs MultiDiffusion (Bar-Tal et al.) — its ORIGINAL domain: overlapping
SPATIAL windows, each ≤64×64 latent, denoised together at every step with their
velocities averaged in the overlapping latent cells. The windows co-evolve over the
30 steps into one globally consistent image; nothing is ever asked to outpaint.

This is the spatial inversion of `t2v_multidiff.py` (which windows the FRAME axis).
We keep the 5D tensor shape (1, 1, h_lat, w_lat, C) with the frame axis frozen to 1
and window over axes 2 (h) and 3 (w). See SPATIAL_MD_PLAN.md for the full design.

Why it fits 16 GB: the denoise loop is bounded by ONE ≤64×64 window's forward (4096
tokens, under the ~9,216-token wall), regardless of final image size. We denoise
windows sequentially within each step and accumulate into a small global velocity
buffer. The VAE decode is already spatially tiled and size-generic (per-tile-bounded
peak, invariant to image size). So memory is ~constant in resolution; only wall-clock
grows (window count is O(area)).

Coordinate regime — RELATIVE, and it is FORCED (not a tuning choice). The lpe table
caps at 64×64, so for any grid >64×64 an ABSOLUTE per-window index (r0+r)*64+(c0+c)
with r0+r≥64 or c0+c≥64 is an out-of-bounds gather, AND mRoPE h/w integers >63 are
rotation angles never trained. So every window is anchored at latent origin: lpe index
r*64+c for window-local r,c, and the SAME window-local position-ids — identical for
every window (exactly the t2v `relative` branch, which also returns the template
unchanged for every window). Windows differ ONLY by which global latent slice they read
and where their velocity pads back. Consequence: neighbours share no absolute coordinate,
so global layout emerges purely from overlap blending — the relative-MD condition that
B6 FreeNoise + B7 anchor-seed were built to stabilise; expect the same mild overlap-blend
softness floor (B7b). The win is RESOLUTION; mild softness is the documented cost.

Each window state is built via t2i's own `_prepare_state` / `_step_velocity` with
`latent_pos_base=None` — IDENTICAL to production single-shot t2i — so every window is a
maximally in-distribution native ≤64×64 image. (t2v MD used base=0 because t2v's
single-shot convention is base=0; t2i's is text-anchor None.)
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from .t2i import (
    T2I_INSTRUCTION,
    VAE_LATENT_CHANNELS,
    VAE_SPATIAL_DOWNSAMPLE,
    TextToImagePipeline,
)
from ._md_common import (
    block_shuffle_idx,
    cfg_combine,
    cfg_window,
    half_pixel_coords,
    img2img_dense,
    img2img_init,
    window_starts,
)
from lance_mlx.model.flow_head import timestep_schedule
from lance_mlx.scheduler.solvers import DPMSolverPlusPlus2M

MAX_LATENT_SIDE = 64  # latent_pos_embed table is 64×64 = 4096 → the single-shot ceiling


# --------------------------------------------------------------------- windowing

def _grid_window_starts(
    h_lat: int, w_lat: int, h_win: int, w_win: int, overlap: int
) -> list[tuple[int, int]]:
    """Cartesian product of per-axis starts → list of (r0, c0) window origins.

    Each window must stay within the 64×64 lpe table (h_win<=64, w_win<=64); the
    WHOLE grid h_lat/w_lat may exceed 64 (that is the whole point).
    """
    if h_win > MAX_LATENT_SIDE or w_win > MAX_LATENT_SIDE:
        raise ValueError(
            f"window {h_win}×{w_win} exceeds the {MAX_LATENT_SIDE}×{MAX_LATENT_SIDE} "
            f"lpe table cap; each window must be ≤{MAX_LATENT_SIDE} per axis")
    rs = window_starts(h_lat, h_win, overlap)
    cs = window_starts(w_lat, w_win, overlap)
    return [(r, c) for r in rs for c in cs]


def _window_lpe(h_win: int, w_win: int) -> mx.array:
    """Window-local latent_pos_embed gather indices, RELATIVE (anchored at origin):
    index(r, c) = r*64 + c for window-local r∈[0,h_win), c∈[0,w_win). Shared by every
    window (relative regime). All indices are < 4096 by construction (h_win,w_win≤64)."""
    idx = [r * MAX_LATENT_SIDE + c for r in range(h_win) for c in range(w_win)]
    return mx.array(idx, dtype=mx.int32)


def _taper2d(h_win: int, w_win: int, kind: str) -> mx.array:
    """Per-window spatial blend weights, shape (1,1,h_win,w_win,1).

    uniform -> all ones (recovers the plain MultiDiffusion average; ×1.0 is an IEEE
               no-op so the baseline numerics are byte-identical).
    cosine  -> separable outer product of two 1-D Hann windows (peaks in the centre,
               tapers toward the window edges so poorly-conditioned border cells
               contribute less). Floored away from 0 via (n+1)/(W+1) so no covered
               cell can end up with zero total weight.
    """
    if kind == "uniform":
        w = np.ones((h_win, w_win), dtype=np.float32)
    elif kind == "cosine":
        hh = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(h_win) + 1) / (h_win + 1))
        ww = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(w_win) + 1) / (w_win + 1))
        w = np.outer(hh, ww).astype(np.float32)
    else:
        raise ValueError(f"blend_taper must be 'uniform' or 'cosine', got {kind!r}")
    return mx.array(w.reshape(1, 1, h_win, w_win, 1))


def _inv_counts(
    starts: list[tuple[int, int]], h_lat: int, w_lat: int,
    h_win: int, w_win: int, taper2d: mx.array,
) -> mx.array:
    """Reciprocal of the overlap-add of the per-window taper across the full grid,
    shape (1,1,h_lat,w_lat,1). velocity = sum_w(taper·v_w) * inv_counts is the
    (taper-weighted) MultiDiffusion average. Asserts full coverage (no 1/0)."""
    taper_np = np.array(taper2d).reshape(h_win, w_win)
    wsum = np.zeros((h_lat, w_lat), dtype=np.float32)
    for (r0, c0) in starts:
        wsum[r0:r0 + h_win, c0:c0 + w_win] += taper_np
    if not np.all(wsum > 0):
        raise ValueError(
            "window union does not cover the full grid (some cell has weight 0); "
            "check window/overlap vs grid size")
    return mx.array((1.0 / wsum).reshape(1, 1, h_lat, w_lat, 1))


def _pad_window(v_win: mx.array, r0: int, c0: int, h_lat: int, w_lat: int) -> list:
    """The 2-D pad spec that places a (1,1,h_win,w_win,C) window velocity into the
    global (1,1,h_lat,w_lat,C) buffer at origin (r0,c0). Factored out so the exact
    extents are unit-testable (the spatial inversion of t2v's 1-D frame pad)."""
    h_win = v_win.shape[2]
    w_win = v_win.shape[3]
    return [(0, 0), (0, 0), (r0, h_lat - r0 - h_win), (c0, w_lat - c0 - w_win), (0, 0)]

# ------------------------------------------------------------------ noise init

def _freenoise_spatial(h_win, w_win, h_lat, w_lat, C, dtype, seed) -> mx.array:
    """Spatial FreeNoise (the (h,w) inversion of t2v's frame-axis FreeNoise):
    every latent cell's noise is drawn from ONE h_win×w_win base block, tiled across
    the grid with an independent row/col permutation per non-origin tile. Windows
    therefore share one noise 'vocabulary' (keeps independently-denoised windows from
    drifting onto different global content) while the per-tile shuffle avoids a
    visibly periodic tile pattern. Built with two mx.take (axis=2 rows, axis=3 cols),
    mirroring t2v's single axis-1 take. GATED — default remains 'independent' until
    the @1536² A/B decides (the temporal-drift rationale is weaker spatially)."""
    mx.random.seed(seed)
    base = mx.random.normal((1, 1, h_win, w_win, C))   # one shared noise block
    rng = np.random.default_rng(seed)                  # consumed rows first, then cols
    lat = mx.take(base, mx.array(block_shuffle_idx(h_lat, h_win, rng)), axis=2)
    lat = mx.take(lat, mx.array(block_shuffle_idx(w_lat, w_win, rng)), axis=3)
    return lat.astype(dtype)


def _upsample_latent(anchor: mx.array, h_lat: int, w_lat: int) -> mx.array:
    """Bilinearly upsample a clean anchor latent (1,1,sh,sw,C) → (1,1,h_lat,w_lat,C)
    in LATENT space (align_corners=False, separable). The B7 spatial anchor: a
    coherent 1024² (64×64) single-shot z0 is upsampled to the larger target grid,
    then img2img-seeded so every window inherits one globally-attended composition.
    Numpy (the anchor is tiny, 64×64×48 ≈ 0.8 MB). UNVALIDATED vs latent artifacts —
    gated knob, A/B'd against unseeded MD (SPATIAL_MD_PLAN open question 3)."""
    a = np.array(anchor.astype(mx.float32))[0, 0]            # (sh, sw, C)
    sh, sw, _ = a.shape
    ys = half_pixel_coords(h_lat, sh)
    xs = half_pixel_coords(w_lat, sw)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, sh - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, sw - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = a[y0] * (1 - wy) + a[y1] * wy                      # (h_lat, sw, C)
    out = top[:, x0, :] * (1 - wx) + top[:, x1, :] * wx      # (h_lat, w_lat, C)
    return mx.array(out[None, None]).astype(anchor.dtype)


# ---------------------------------------------------------- per-window velocity

def _window_cfg_velocity(
    pipe: TextToImagePipeline, *,
    cond_state: dict, uncond_state: dict | None,
    z_win: mx.array, t: mx.array, lpe_indices: mx.array,
    h_win: int, w_win: int,
    cfg_scale_step: float, cfg_renorm_type: str, cfg_renorm_min: float,
) -> mx.array:
    """One window's CFG velocity, shape (1,1,h_win,w_win,C).

    Reuses t2i's PRODUCTION-validated `_step_velocity` per arm (the GEN-only cached
    path, identical numerics to single-shot t2i) and replicates `generate`'s CFG
    renorm block (t2i.py:309-331). Every window is the same size + relative-anchored,
    so it shares cond/uncond state + lpe; only z_win differs.
    """
    n_lat_win = h_win * w_win
    v_cond = pipe._step_velocity(
        state=cond_state, latents=z_win, t=t,
        lpe_indices=lpe_indices, n_lat=n_lat_win, h_lat=h_win, w_lat=w_win,
    )
    if uncond_state is not None and cfg_scale_step > 1.0:
        v_uncond = pipe._step_velocity(
            state=uncond_state, latents=z_win, t=t,
            lpe_indices=lpe_indices, n_lat=n_lat_win, h_lat=h_win, w_lat=w_win,
        )
        return cfg_combine(v_cond, v_uncond, cfg_scale_step,
                           cfg_renorm_type, cfg_renorm_min)
    return v_cond


# --------------------------------------------------------------------- decode

def _decode_image(pipe: TextToImagePipeline, latents: mx.array, *,
                  verbose: bool):
    """LOSSLESS memory-bounded VAE decode — mirrors TextToImagePipeline.generate's
    decode block. decode_streaming equals dec(z) BIT-IDENTICALLY (vs the old lossy
    decode_tiled blend) with a per-tile-bounded peak; n scales with the latent's own
    h_lat/w_lat. Returns a PIL.Image."""
    from mlx_video.models.wan_2.vae22 import denormalize_latents
    from lance_mlx.model.vae_stream import decode_streaming, suggest_spatial_tiles
    from PIL import Image

    if verbose:
        mx.reset_peak_memory()
    pipe._materialize_vae()   # relay: the decoder was left lazy until its phase
    z = denormalize_latents(latents).astype(pipe.vae_decoder.conv2.weight.dtype)
    n_tiles = suggest_spatial_tiles(z.shape[2], z.shape[3])
    decoded = decode_streaming(pipe.vae_decoder, z, chunk_lat=1, spatial_tiles=n_tiles)
    mx.eval(decoded)
    if verbose:
        print(f"  [decode] peak={mx.get_peak_memory()/1e9:.2f}GB "
              f"active={mx.get_active_memory()/1e9:.2f}GB n_tiles={n_tiles}")
    img_t = decoded[0, 0].astype(mx.float32)        # (H', W', 3)
    img_np = np.array(img_t)
    img_u8 = ((img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_u8)


# ----------------------------------------------------------------- the driver

def generate_multidiff_spatial(
    pipe: TextToImagePipeline,
    prompt: str,
    *,
    height: int,
    width: int,
    window_lat: int = 64,            # latent cells per window axis (<=64, the lpe cap)
    overlap_lat: int = 16,           # latent-cell overlap between adjacent windows
    noise_init: str = "independent", # "independent" (default until A/B) | "freenoise"
    blend_taper: str = "uniform",    # "uniform" (plain MD average) | "cosine" (edge-tapered)
    anchor_latents: mx.array | None = None,   # B7: clean z0 (1,1,h_lat,w_lat,C), upsampled from a 1024² anchor
    anchor_t_start: float = 0.55,
    anchor_schedule: str = "tail",   # "tail" (DEFAULT) | "dense"
    num_steps: int = 30,
    timestep_shift: float = 3.5,
    scheduler: str = "euler",
    cfg_scale: float = 4.0,
    cfg_renorm_type: str = "channel",
    cfg_renorm_min: float = 0.0,
    cfg_interval: tuple[float, float] | None = None,
    seed: int = 42,
    instruction: str = T2I_INSTRUCTION,
    verbose: bool = False,
    return_latents: bool = False,
):
    """Spatial MultiDiffusion t2i for images larger than 1024². See module docstring.

    Memory is bounded by ONE ≤64×64 window's forward regardless of `height`/`width`;
    only window count (and thus wall-clock) grows with area. Returns a PIL.Image
    (or the final latents if return_latents=True).

    `scheduler`: integration scheme for the global (taper-blended) MD step,
    reusing the base pipeline's solver (PR #4). "euler" (default) or "dpm"
    (DPM-Solver++(2M)). Default is "euler": DPM is OPT-IN AND EXPERIMENTAL for
    MultiDiffusion — the multistep extrapolation runs on the blended velocity
    field, untested against the MD overlap-blend + `cfg_renorm_type="channel"`
    default. The spatial-MD step is ONE global integration over the schedule, so
    a single solver advances the whole latent buffer. Validate before relying on
    `scheduler="dpm"`.
    """
    if scheduler not in ("euler", "dpm"):
        raise ValueError(f"Unknown scheduler {scheduler!r}. Use 'euler' or 'dpm'.")
    assert height % VAE_SPATIAL_DOWNSAMPLE == 0 and width % VAE_SPATIAL_DOWNSAMPLE == 0
    h_lat = height // VAE_SPATIAL_DOWNSAMPLE
    w_lat = width // VAE_SPATIAL_DOWNSAMPLE
    h_win = min(window_lat, h_lat)
    w_win = min(window_lat, w_lat)
    starts = _grid_window_starts(h_lat, w_lat, h_win, w_win, overlap_lat)

    cfg_lo, cfg_hi = cfg_window(cfg_interval)

    if verbose:
        print(f"[spatial-md] {height}×{width} -> {h_lat}×{w_lat} latent")
        print(f"  window {h_win}×{w_win} (overlap {overlap_lat}); "
              f"{len(starts)} windows @ {starts}")
        print(f"  per-window tokens={h_win*w_win} (wall≈9216)  "
              f"noise_init={noise_init} taper={blend_taper}")

    # --- one window-sized prompt state per arm, shared across all windows --------
    # latent_pos_base=None == production single-shot t2i (text-anchor), so every
    # window is a maximally in-distribution native ≤64×64 image.
    cond_state = pipe._prepare_state(
        prompt=prompt, instruction=instruction,
        n_lat=h_win * w_win, h_lat=h_win, w_lat=w_win, verbose=verbose,
        latent_pos_base=None,
    )
    uncond_state = None
    if cfg_scale > 1.0:
        uncond_state = pipe._prepare_state(
            prompt="", instruction=instruction,
            n_lat=h_win * w_win, h_lat=h_win, w_lat=w_win, verbose=False,
            latent_pos_base=None,
        )

    # --- prefill text prefix ONCE per arm, then free the UND tower ----------------
    # Under deferred loading (memory_mode="relay") the GEN tower is lazy. A
    # full-routed prefill's mx.where would evaluate the *_moe_gen branch and
    # materialize it, defeating the deferral — so prefill UND-only (bit-identical
    # on an all-TEXT prefix), then free UND and materialize GEN for the loop. The
    # two towers are never co-resident. Mirrors t2v_multidiff / t2i.generate.
    gen_deferred = bool(getattr(pipe.lance_model, "_gen_deferred", False))
    cond_state["caches"] = pipe.lance_model.prefill_prefix(
        cond_state["prefix_embeds"],
        position_ids=cond_state["position_ids_prefix"],
        position_group=cond_state["position_group_prefix"],
        mask=cond_state["mask_prefix"],
        und_only=gen_deferred,
    )
    if uncond_state is not None:
        uncond_state["caches"] = pipe.lance_model.prefill_prefix(
            uncond_state["prefix_embeds"],
            position_ids=uncond_state["position_ids_prefix"],
            position_group=uncond_state["position_group_prefix"],
            mask=uncond_state["mask_prefix"],
            und_only=gen_deferred,
        )
    setup_peak = mx.get_peak_memory() / 1e9
    if gen_deferred:
        stats = pipe.lance_model.free_und_tower(deferred=True)
        pipe.lance_model.materialize_gen_tower()
    else:
        stats = pipe.lance_model.free_und_tower()
    if verbose:
        print(f"  [opt] prefilled shared prefix; freed UND "
              f"{stats['freed_bytes']/1e9:.2f}GB, active {stats['active_after']/1e9:.2f}GB")
        print(f"  [mem] setup peak (incl prefill) = {setup_peak:.2f}GB")
    mx.reset_peak_memory()   # so the loop peak below is the per-window forward alone

    # --- shared window-local lpe + blend weights ---------------------------------
    lpe_indices = _window_lpe(h_win, w_win)
    _dtype = pipe.lance_model.embed_tokens.weight.dtype
    if noise_init == "independent":
        mx.random.seed(seed)
        latents = mx.random.normal((1, 1, h_lat, w_lat, VAE_LATENT_CHANNELS)).astype(_dtype)
    elif noise_init == "freenoise":
        latents = _freenoise_spatial(h_win, w_win, h_lat, w_lat, VAE_LATENT_CHANNELS, _dtype, seed)
    else:
        raise ValueError(f"noise_init must be 'independent' or 'freenoise', got {noise_init!r}")

    taper_f32 = _taper2d(h_win, w_win, blend_taper)                          # (1,1,h_win,w_win,1) f32
    inv_counts = _inv_counts(starts, h_lat, w_lat, h_win, w_win, taper_f32).astype(latents.dtype)
    taper = taper_f32.astype(latents.dtype)   # bf16 for the velocity multiply (numpy needs the f32 one)

    sched = timestep_schedule(num_steps=num_steps, shift=timestep_shift)
    loop_sched = sched

    # --- B7: img2img-seed from an (upsampled) coherent 1024² anchor --------------
    step0 = 0
    if anchor_latents is not None:
        want = (1, 1, h_lat, w_lat, VAE_LATENT_CHANNELS)
        if tuple(anchor_latents.shape) != want:
            raise ValueError(f"anchor_latents shape {tuple(anchor_latents.shape)} != {want}")
        if not (0.0 < anchor_t_start <= 1.0):
            raise ValueError(f"anchor_t_start must be in (0,1], got {anchor_t_start}")
        if anchor_schedule == "tail":
            latents, step0 = img2img_init(anchor_latents, latents, sched, num_steps, anchor_t_start)
        elif anchor_schedule == "dense":
            latents, loop_sched = img2img_dense(anchor_latents, latents, sched, anchor_t_start)
        else:
            raise ValueError(f"anchor_schedule must be 'tail' or 'dense', got {anchor_schedule!r}")
        mx.eval(latents)
        if verbose:
            print(f"  [anchor] seed t_start={anchor_t_start} mode={anchor_schedule} "
                  f"-> step0={step0}, refining {num_steps-step0}/{num_steps} steps")

    # One solver per generate(): the spatial-MD step is a single global
    # integration over the schedule (per-window velocities blended, then ONE
    # update advances the whole buffer), so AB2 history is the global trajectory.
    solver = DPMSolverPlusPlus2M() if scheduler == "dpm" else None

    # --- denoise loop: windows sequential within each step (memory-bounded) ------
    tg = time.perf_counter()
    for step in range(step0, num_steps):
        t = loop_sched[step]
        dt = loop_sched[step] - loop_sched[step + 1]
        t_scalar = float(t.item()) if hasattr(t, "item") else float(t)
        cfg_active = (t_scalar > cfg_lo) and (t_scalar <= cfg_hi)
        cfg_scale_step = cfg_scale if cfg_active else 1.0

        v_accum = mx.zeros_like(latents)
        for (r0, c0) in starts:
            z_win = latents[:, :, r0:r0 + h_win, c0:c0 + w_win, :]
            v_win = _window_cfg_velocity(
                pipe, cond_state=cond_state, uncond_state=uncond_state,
                z_win=z_win, t=t, lpe_indices=lpe_indices, h_win=h_win, w_win=w_win,
                cfg_scale_step=cfg_scale_step,
                cfg_renorm_type=cfg_renorm_type, cfg_renorm_min=cfg_renorm_min,
            )
            v_accum = v_accum + mx.pad(v_win * taper, _pad_window(v_win, r0, c0, h_lat, w_lat))
            mx.eval(v_accum)
        velocity = v_accum * inv_counts      # (taper-weighted) MultiDiffusion blend

        if solver is not None:
            latents = solver.step(velocity, latents, dt)
        else:
            latents = latents - velocity * dt    # Euler (matches single-shot)
        mx.eval(latents)

        if verbose:
            lat_np = latents.astype(mx.float32)
            print(f"  step {step+1}/{num_steps} t={t_scalar:.4f} dt={float(dt):.4f}  "
                  f"mean={float(mx.mean(lat_np)):.3f} std={float(mx.std(lat_np)):.3f}  "
                  f"peak={mx.get_peak_memory()/1e9:.2f}GB")

    gen_s = time.perf_counter() - tg
    if verbose:
        print(f"  [spatial-md] gen {gen_s:.1f}s peak={mx.get_peak_memory()/1e9:.2f}GB "
              f"active={mx.get_active_memory()/1e9:.2f}GB")

    if return_latents:
        mx.eval(latents)
        return latents

    # Relay shed cascade (mirrors t2i.generate): under relay the pipe is
    # single-shot anyway (UND already gone), so drop the dead prefix caches +
    # GEN tower before decoding; the decode peak is then VAE-only + transient.
    if pipe.memory_mode == "relay":
        cond_state["caches"] = None
        if uncond_state is not None:
            uncond_state["caches"] = None
        shed = pipe.lance_model.free_gen_tower()
        if verbose:
            print(f"  [opt] shed GEN tower before decode: freed "
                  f"{shed['freed_bytes']/1e9:.2f}GB, active "
                  f"{shed['active_after']/1e9:.2f}GB")

    return _decode_image(pipe, latents, verbose=verbose)
