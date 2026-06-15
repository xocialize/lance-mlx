"""t2v_multidiff — long video on Lance via MultiDiffusion fused parallel windows.

Phase D proved that *autoregressive* temporal chunking is a dead end: Lance has
no temporal-extension prior (trained on whole-clip t2v + co-temporal video_edit
reconstruction, never on "given a prefix, generate the future"), so any
conditioning trick that asks window i>0 to continue window i-1 produces a hard
appearance seam (~1.2 latent jump, invariant to every knob). See
PHASE_D_FINDINGS.md.

This module takes the opposite approach — **MultiDiffusion** (Bar-Tal et al.;
the video analogues are FreeNoise / Gen-L-Video). Instead of chaining windows
one-after-another, ALL overlapping temporal windows are denoised *together*, at
every step, and their predicted velocities are **averaged in the overlapping
latent frames**. The windows co-evolve over the 30 steps into one globally
consistent sample. Nothing is ever asked to "extend": each window is a vanilla
t2v forward — exactly what the model can do — and the per-step overlap-coupling
is what stitches them into a longer clip.

Why this fits a 16 GB node:
  * The denoise loop is the memory wall (≈9,216 latent tokens → 15.22 GB),
    LINEAR in the window's token count, independent of total length. We denoise
    windows **sequentially within each step** and accumulate into a small global
    velocity buffer, so peak memory is bounded by ONE window's forward — never
    the whole clip. Total length is then limited only by the latent_pos_embed
    table (31 latent frames = 121 output frames), not by memory.
  * The text prefix is identical for every window (same prompt), so we
    `prefill_prefix` ONCE per CFG arm, `free_und_tower` (the 6.17 GB win) ONCE,
    and every window runs the GEN-only `gen_loop_forward` against that shared
    cache.

Two coordinate regimes (the `window_coords` flag, mirroring Phase D's discipline
of testing the conditioning regime explicitly):
  * "relative" (default, lowest-risk): every window is anchored at temporal
    origin (lpe frame 0..W-1, position-id t-axis 0..W-1). This is *exactly* the
    production-validated single-shot config (`latent_pos_base=0`,
    `mape_anchor=None`) with fewer frames — maximally in-distribution. Global
    motion emerges purely from the overlap coupling. Isolates the question
    "does blending stitch windows seamlessly?" with no OOD confound.
  * "absolute": window starting at global latent-frame s gets lpe frame s+f and
    position-id t-axis s+f, so the model is told where in the clip each window
    sits. More temporal direction, but the per-window forward sits at an
    untested temporal offset (the model only ever saw offset-0 clips that also
    contained frames 0..s). Use after "relative" to add motion direction if the
    relative regime sloshes.

Noise init (`noise_init`, B6 result): the plain MD baseline samples independent
noise for every latent frame, which lets the independently-denoised windows
drift onto different global content -> mid-clip blur + framing drift. FreeNoise
(Qiu et al.) instead draws every frame's noise from a single W-frame base block,
block-shuffled beyond the first window, so all windows share one noise
"vocabulary". The 256²×49f A/B (seed 42, relative) was decisive and FREE (same
gen time + memory): mid/ends sharpness 0.56 -> 1.29 (blur gone), drift
0.179 -> 0.165, motion preserved. `noise_init="freenoise"` is therefore the
default; `"independent"` reproduces the old baseline. The `blend_taper="cosine"`
edge-taper adds marginal sharpness but neutral/worse drift; overlap=4 was worse
on both AND 2.4× slower -> both rejected. See PHASE_D_FINDINGS / memory.

Anchor seeding (`anchor_latents` + `anchor_t_start`, B7): FreeNoise fixes *blur*
but each window is still an independent sample, so composition can still drift.
B7 pins composition by img2img-seeding every window from ONE coherent single-shot
clip. A real `generate(return_latents=True)` at a length single-shot reaches
coherently gives a clean z0 spanning the WHOLE clip (full global attention = one
agreed-upon framing); we re-noise it to a schedule time t0=sched[step0] (nearest
step at/below `anchor_t_start`) and run MD over only the schedule tail. Because
the rectified-flow forward z_t=(1-t)z0+t·eps is the exact inverse of the loop's
z=z-v·dt, the truncated loop is self-consistent — windows inherit the anchor's
global framing and only repaint local detail. Lower t_start = trust the anchor
more (less drift, less window freedom); higher = more MD freedom. Anchor-gen and
MD must be SEPARATE processes (each calls free_und_tower once), so the anchor is
saved/loaded as a tiny latent .npy. With anchor_latents=None this is the plain
MD path (step0=0, full schedule) — byte-identical to B6.

`anchor_schedule` picks how many refine steps run below t_start. "tail" (DEFAULT)
runs only the original-schedule steps that fall under t_start (~7-13 of 30 with
shift=3.5). "dense" instead scales the standard [1,0] schedule by t_start so ALL
num_steps evals land inside [0, t_start] (same ODE, more steps). The B7b A/B
(256²×49f) showed dense ≈ tail on EVERY metric (sharp 112-127 vs 116-131, SSIM,
drift, motion all within noise) and visually identical — the residual softness vs
the single-shot oracle is the MD overlap-blend FLOOR, not a step-count artifact, so
more steps don't recover it. tail is therefore the default (2-4× faster, 138-277s
vs 586-648s, for the same output). "dense" kept as a documented knob.

The CFG path (channel renorm), schedule, and per-window forward mirror
`TextToVideoPipeline.generate` exactly so each window reproduces the validated
single-shot behaviour. We reuse the pipeline's `_prepare_state` for prompt
assembly and drive the model heads directly here.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from .t2v import (
    MAX_LATENT_FRAMES,
    MAX_LATENT_SIDE,
    T2V_INSTRUCTION,
    VAE_LATENT_CHANNELS,
    VAE_SPATIAL_DOWNSAMPLE,
    VAE_TEMPORAL_DOWNSAMPLE,
    TextToVideoPipeline,
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


def _window_lpe_and_pos(
    template_lpe: mx.array,          # (n_lat_win,) frame f -> f*64^2 + r*64 + c
    template_pos_lat: mx.array,      # (3, 1, n_lat_win) — t-axis = f (base=0)
    start: int,
    window_coords: str,
) -> tuple[mx.array, mx.array]:
    """Per-window latent_pos_embed indices and position-ids for window @ `start`.

    relative -> unchanged template (every window anchored at temporal origin).
    absolute -> add `start` to the temporal frame term of the lpe index
                (start * 64^2) and to the t-axis (axis 0) of the latent
                position-ids; h/w axes untouched.
    """
    if window_coords == "relative" or start == 0:
        return template_lpe, template_pos_lat
    if window_coords != "absolute":
        raise ValueError(f"window_coords must be 'relative' or 'absolute', got {window_coords!r}")
    lpe = template_lpe + start * (MAX_LATENT_SIDE ** 2)
    pos = np.array(template_pos_lat)
    pos[0, 0, :] = pos[0, 0, :] + start          # shift temporal axis only
    return lpe, mx.array(pos)


def _window_velocity(
    pipe: TextToVideoPipeline,
    *,
    z_win: mx.array,                 # (1, W, h_lat, w_lat, C)
    t: mx.array,
    lpe_indices: mx.array,           # (n_lat_win,)
    cond_state: dict,
    uncond_state: dict | None,
    cfg_scale_step: float,
    cfg_renorm_type: str,
    cfg_renorm_min: float,
    W: int,
    h_lat: int,
    w_lat: int,
) -> mx.array:
    """One window's CFG velocity, shaped (1, W, h_lat, w_lat, C).

    Reuses t2v's PRODUCTION-validated `_step_velocity` per arm (the GEN-only
    cached path — identical numerics to single-shot t2v) and the shared
    `cfg_combine` renorm, exactly as t2i_multidiff does with t2i's. The states
    are per-window shallow copies whose "position_ids_lat" carries the window's
    own (relative or absolute) coords against the shared per-arm prefix cache.
    """
    n_lat_win = W * h_lat * w_lat
    v_cond = pipe._step_velocity(
        state=cond_state, latents=z_win, t=t,
        lpe_indices=lpe_indices, n_lat=n_lat_win, t_lat=W, h_lat=h_lat, w_lat=w_lat,
    )
    if uncond_state is not None and cfg_scale_step > 1.0:
        v_uncond = pipe._step_velocity(
            state=uncond_state, latents=z_win, t=t,
            lpe_indices=lpe_indices, n_lat=n_lat_win, t_lat=W, h_lat=h_lat, w_lat=w_lat,
        )
        return cfg_combine(v_cond, v_uncond, cfg_scale_step,
                           cfg_renorm_type, cfg_renorm_min)
    return v_cond


def _taper_weights(W: int, kind: str) -> mx.array:
    """Per-window frame weights for the MultiDiffusion blend, shape (1,W,1,1,1).

    uniform -> all ones (recovers the plain MultiDiffusion average; multiplying a
               velocity by exactly 1.0 is an IEEE no-op so the baseline numerics
               are byte-identical).
    cosine  -> a raised cosine (Hann) that peaks mid-window and tapers toward the
               edges, so a window's poorly-conditioned boundary frames contribute
               less to the blend. Floored away from 0 (uses (n+1)/(W+1)) so no
               covered frame can end up with zero total weight.
    """
    if kind == "uniform" or W == 1:
        w = np.ones(W, dtype=np.float32)
    elif kind == "cosine":
        n = np.arange(W)
        w = (0.5 - 0.5 * np.cos(2 * np.pi * (n + 1) / (W + 1))).astype(np.float32)
    else:
        raise ValueError(f"blend_taper must be 'uniform' or 'cosine', got {kind!r}")
    return mx.array(w.reshape(1, W, 1, 1, 1))


def _freenoise_latents(W, total_t, h_lat, w_lat, C, dtype, seed) -> mx.array:
    """FreeNoise (Qiu et al.) correlated init: every latent frame's noise is drawn
    from a single W-frame base block, reshuffled in blocks of W beyond the first
    window. Long-range frames therefore share the same noise 'vocabulary', which
    is what keeps independently-denoised windows from drifting onto different
    global content — while the per-block shuffle keeps frames non-identical so the
    clip doesn't go static/loop (the failure mode of naive noise repetition).
    """
    mx.random.seed(seed)
    base = mx.random.normal((1, W, h_lat, w_lat, C))      # W distinct noise frames
    rng = np.random.default_rng(seed)
    idx = block_shuffle_idx(total_t, W, rng)
    latents = mx.take(base, mx.array(idx), axis=1)        # (1, total_t, h, w, C)
    return latents.astype(dtype)


# ------------------------------------------------- temporal latent upsample (D)

def _upsample_latent_temporal(anchor: mx.array, t_lat: int,
                              interp: str = "bilinear") -> mx.array:
    """Upsample a clean anchor latent (1,st,h,w,C) -> (1,t_lat,h,w,C) along the
    TEMPORAL axis (align_corners=False), the time-axis mirror of
    t2i_multidiff._upsample_latent. The B7 spatial anchor pins composition but is
    fixed-length: `generate_multidiff(anchor_latents=...)` needs an anchor of the
    FULL target length, so today the only anchor available is a single-shot at the
    same length -> B7 can pin but never EXTEND past single-shot reach. This builds
    a full-length anchor by stretching a SHORTER coherent clip's latent trajectory
    in time, so every MD window inherits a global *motion skeleton* (not just a
    frozen frame). Decoded alone it is a blend (ghosting) clip; the value is
    whether the img2img refine repaints it into real motion (the held-out A/B).

    `interp`:
      "bilinear" (DEFAULT, byte-stable) -- straight-line (linear-in-time) blend of
        the two neighbour frames. Cheap, but at a midpoint between frames pointing
        in different directions the per-frame NORM sags below both neighbours
        (vector cancellation) -> the decoded midpoint washes out, and the blurry
        overlap reads as mesh/fringe texture after refine (the reach-A/B blocker).
      "slerp" -- spherical-arc blend: slerp the per-frame unit DIRECTIONS on the
        hypersphere and lerp the per-frame MAGNITUDES separately, so the
        interpolated frame's norm is the lerp of the two neighbour norms (no sag)
        while the direction rides the great-circle arc. Fixes the washout; it
        still blends POSITIONS, so it cannot remove ghosting from large motion
        (that needs flow-warp, Phase 2). Endpoints + identity + constant-field are
        exact, same as bilinear.

    Endpoints are preserved (out frame 0 -> src 0, out frame t_lat-1 -> src st-1
    under the clip), and t_lat == st is the identity. Numpy: the anchor is tiny
    (e.g. 13x16x16x48 ~ 0.6 MB), so the interp cost is negligible vs the refine."""
    a = np.array(anchor.astype(mx.float32))[0]               # (st, h, w, C)
    st = a.shape[0]
    if t_lat == st:
        return anchor
    # align_corners=False source coordinates, clipped to the valid range so the
    # first/last output frames land exactly on the first/last source frames.
    ts = half_pixel_coords(t_lat, st)
    f0 = np.floor(ts).astype(int)
    f1 = np.minimum(f0 + 1, st - 1)
    wt = ts - f0                                             # (t_lat,)
    v0 = a[f0]                                               # (t_lat, h, w, C)
    v1 = a[f1]
    if interp == "bilinear":
        w = wt[:, None, None, None]
        out = v0 * (1 - w) + v1 * w
    elif interp == "slerp":
        # flatten each frame to a vector; arc the direction, lerp the magnitude.
        sh = v0.shape
        p0 = v0.reshape(t_lat, -1).astype(np.float64)
        p1 = v1.reshape(t_lat, -1).astype(np.float64)
        n0 = np.linalg.norm(p0, axis=1, keepdims=True)      # (t_lat,1)
        n1 = np.linalg.norm(p1, axis=1, keepdims=True)
        eps = 1e-12
        u0 = p0 / (n0 + eps)
        u1 = p1 / (n1 + eps)
        dot = np.clip((u0 * u1).sum(1, keepdims=True), -1.0, 1.0)
        omega = np.arccos(dot)                              # angle between frames
        so = np.sin(omega)
        w = wt[:, None]
        small = so < 1e-7                                   # parallel -> lerp the dir too
        c0 = np.where(small, 1.0 - w, np.sin((1.0 - w) * omega) / (so + eps))
        c1 = np.where(small, w,       np.sin(w * omega)        / (so + eps))
        direction = c0 * u0 + c1 * u1                       # ~unit arc
        magnitude = (1.0 - w) * n0 + w * n1                 # norm = lerp(n0,n1): no sag
        out = (direction * magnitude).reshape(sh).astype(np.float32)
    else:
        raise ValueError(f"interp must be 'bilinear' or 'slerp', got {interp!r}")
    return mx.array(out[None]).astype(anchor.dtype)


def generate_multidiff(
    pipe: TextToVideoPipeline,
    prompt: str,
    *,
    total_frames: int,
    height: int = 512,
    width: int = 512,
    window_frames: int = 17,         # OUTPUT frames per window -> W latent = (wf-1)//4+1
    overlap_lat: int = 2,            # latent-frame overlap between adjacent windows
    window_coords: str = "relative",
    window_prompts: list[str] | None = None,  # optional per-window prompt schedule; len must match windows
    noise_init: str = "freenoise",   # B6 WINNER: correlated init. "independent" = old baseline.
    blend_taper: str = "uniform",    # "uniform" (B6 winner) | "cosine" (edge-tapered blend)
    anchor_latents: mx.array | None = None,   # B7: clean z0 (1,total_t,h_lat,w_lat,C) from a coherent single-shot
    anchor_t_start: float = 0.55,    # B7: img2img (SDEdit) start time; only used when anchor_latents is set
    anchor_schedule: str = "tail",   # B7: "tail" (orig schedule tail; DEFAULT — 2-4× faster, same quality as dense per B7b) | "dense" (fresh num_steps over [t_start,0])
    num_steps: int = 30,
    timestep_shift: float = 3.5,
    scheduler: str = "euler",
    cfg_scale: float = 4.0,
    cfg_renorm_type: str = "channel",
    cfg_renorm_min: float = 0.0,
    cfg_interval: tuple[float, float] | None = None,
    cfg_uncond_mode: str = "empty_prompt",
    seed: int = 42,
    instruction: str = T2V_INSTRUCTION,
    verbose: bool = False,
    return_latents: bool = False,
) -> mx.array:
    """MultiDiffusion long-video generation. See module docstring.

    `window_frames`/`overlap_lat` set the window in OUTPUT frames and the
    overlap in LATENT frames; W_latent = (window_frames-1)//4 + 1 must keep
    W_latent * h_lat * w_lat under the ~9,216-token loop-memory wall.

    `scheduler`: integration scheme for the global (taper-blended) MD step,
    reusing the base pipeline's solver (PR #4). "euler" (default) or "dpm"
    (DPM-Solver++(2M)). Default is "euler": DPM is OPT-IN AND EXPERIMENTAL for
    MultiDiffusion. The base t2v already flags DPM-at-fewer-steps as untested
    against `cfg_renorm_type="channel"` (the MD default), and MD adds its own
    lossy overlap-blend on top, so the multistep extrapolation runs on a blended
    velocity field rather than a single window's. The MD path is ONE global
    integration over the shared schedule, so a single solver advances the full
    buffer (one warm-up Euler step, then AB2). Validate output at your target
    length before relying on `scheduler="dpm"`.

    """
    if scheduler not in ("euler", "dpm"):
        raise ValueError(f"Unknown scheduler {scheduler!r}. Use 'euler' or 'dpm'.")
    assert height % VAE_SPATIAL_DOWNSAMPLE == 0 and width % VAE_SPATIAL_DOWNSAMPLE == 0
    h_lat = height // VAE_SPATIAL_DOWNSAMPLE
    w_lat = width // VAE_SPATIAL_DOWNSAMPLE
    total_t = (total_frames - 1) // VAE_TEMPORAL_DOWNSAMPLE + 1
    W = (window_frames - 1) // VAE_TEMPORAL_DOWNSAMPLE + 1
    n_lat_win = W * h_lat * w_lat

    if window_coords == "absolute":
        assert total_t <= MAX_LATENT_FRAMES, (
            f"absolute coords need total_t={total_t} <= {MAX_LATENT_FRAMES} "
            f"(latent_pos_embed table); use 'relative' or shorter video")
    starts = window_starts(total_t, W, overlap_lat)
    if window_prompts is not None and len(window_prompts) != len(starts):
        raise ValueError(
            f"window_prompts length {len(window_prompts)} must match "
            f"{len(starts)} windows for starts={starts}"
        )

    cfg_lo, cfg_hi = cfg_window(cfg_interval)

    pipe.lance_model.set_rope_fp32(False)
    pipe.lance_model.set_attention_fp32(False)

    if verbose:
        print(f"[multidiff] {total_frames}f×{height}×{width} -> total_t={total_t} latent")
        print(f"  window W={W} latent ({window_frames}f), overlap={overlap_lat}, "
              f"coords={window_coords}")
        print(f"  windows @ {starts}  ({len(starts)} forwards/arm/step)")
        print(f"  per-window tokens={n_lat_win}  (wall≈9216)")
        print(f"  noise_init={noise_init}  blend_taper={blend_taper}")

    # --- prompt template(s): shared prompt by default, optional per-window text -
    cond_states = []
    prompt_seq = [prompt] if window_prompts is None else list(window_prompts)
    for pi, prompt_i in enumerate(prompt_seq):
        cond_states.append(pipe._prepare_state(
            prompt=prompt_i, instruction=instruction,
            n_lat=n_lat_win, t_lat=W, h_lat=h_lat, w_lat=w_lat,
            verbose=(verbose and pi == 0),
            mape_anchor=None, uncond_no_text=False,
            spatial_merge_size=1, prompt_format="ours", latent_pos_base=0,
        ))
    cond_state = cond_states[0]
    uncond_state = None
    if cfg_scale > 1.0:
        uncond_state = pipe._prepare_state(
            prompt="", instruction=instruction,
            n_lat=n_lat_win, t_lat=W, h_lat=h_lat, w_lat=w_lat, verbose=False,
            mape_anchor=None, uncond_no_text=(cfg_uncond_mode == "no_text"),
            spatial_merge_size=1, prompt_format="ours", latent_pos_base=0,
        )

    # --- prefill text prefix ONCE per arm, then free UND tower -----------------
    # Under deferred loading (memory_mode != parallel) the GEN tower is lazy. A
    # full-routed prefill's mx.where would evaluate the *_moe_gen branch and
    # materialize it, defeating the deferral — so prefill UND-only (bit-identical
    # on an all-TEXT prefix), then free UND and materialize GEN for the loop. The
    # two towers are never co-resident, so MD holds ~one tower (~5.5 GB) and fits
    # 16 GB. Mirrors t2v.generate's relay path (task #34).
    gen_deferred = bool(getattr(pipe.lance_model, "_gen_deferred", False))
    for cond_i in cond_states:
        cond_i["caches"] = pipe.lance_model.prefill_prefix(
            cond_i["prefix_embeds"],
            position_ids=cond_i["position_ids_prefix"],
            position_group=cond_i["position_group_prefix"],
            mask=cond_i["mask_prefix"],
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
    setup_peak = mx.get_peak_memory() / 1e9      # single-tower prefill transient (fixed cost)
    if gen_deferred:
        stats = pipe.lance_model.free_und_tower(deferred=True)
        pipe.lance_model.materialize_gen_tower()
    else:
        stats = pipe.lance_model.free_und_tower()
    if verbose:
        prefix_mode = "shared prefix" if window_prompts is None else "window prompts"
        print(f"  [opt] prefilled {len(starts)}×{prefix_mode}; freed UND "
              f"{stats['freed_bytes']/1e9:.2f} GB, active {stats['active_after']/1e9:.2f} GB")
        _tower = "single-tower (UND only)" if gen_deferred else "dual-tower"
        print(f"  [mem] setup peak ({_tower} prefill) = {setup_peak:.2f} GB")
    # Reset the high-water so the loop peak below is the per-window forward cost
    # ALONE — the headline number, which scales with window size, NOT total
    # length. Total memory ceiling = max(setup_peak, loop_peak); for windows
    # under the ~9,216-token wall the loop stays below the fixed setup peak, so
    # MD holds ~constant memory for arbitrary length.
    mx.reset_peak_memory()

    # --- per-window lpe + position-ids (precomputed; cheap) --------------------
    # Each window gets a SHALLOW state copy whose "position_ids_lat" carries the
    # window's own coords (relative: the shared template, unchanged; absolute:
    # temporally shifted). Caches/embeds are shared by reference, so
    # `_step_velocity` runs per window against the one per-arm prefix cache.
    template_lpe = mx.array(
        [f * (MAX_LATENT_SIDE ** 2) + r * MAX_LATENT_SIDE + c
         for f in range(W) for r in range(h_lat) for c in range(w_lat)],
        dtype=mx.int32,
    )
    win_lpe, win_cond, win_uncond = [], [], []
    for wi, s in enumerate(starts):
        cond_base = cond_state if window_prompts is None else cond_states[wi]
        lpe_s, pos_c = _window_lpe_and_pos(
            template_lpe, cond_base["position_ids_lat"], s, window_coords)
        win_lpe.append(lpe_s)
        win_cond.append({**cond_base, "position_ids_lat": pos_c})
        if uncond_state is not None:
            _, pos_u = _window_lpe_and_pos(
                template_lpe, uncond_state["position_ids_lat"], s, window_coords)
            win_uncond.append({**uncond_state, "position_ids_lat": pos_u})
        else:
            win_uncond.append(None)

    # --- init ONE global noise tensor; windows slice & blend into it -----------
    _dtype = pipe.lance_model.embed_tokens.weight.dtype
    if noise_init == "independent":
        mx.random.seed(seed)
        latents = mx.random.normal((1, total_t, h_lat, w_lat, VAE_LATENT_CHANNELS)).astype(_dtype)
    elif noise_init == "freenoise":
        latents = _freenoise_latents(
            W, total_t, h_lat, w_lat, VAE_LATENT_CHANNELS, _dtype, seed)
    else:
        raise ValueError(f"noise_init must be 'independent' or 'freenoise', got {noise_init!r}")

    # per-window frame taper (uniform=ones recovers the plain average); the global
    # per-frame normalizer is the overlap-add of that taper across all windows.
    taper = _taper_weights(W, blend_taper)                    # (1,W,1,1,1)
    taper_np = np.array(taper).reshape(W)
    wsum = np.zeros(total_t, dtype=np.float32)
    for s in starts:
        wsum[s:s + W] += taper_np
    inv_counts = mx.array((1.0 / wsum).reshape(1, total_t, 1, 1, 1)).astype(latents.dtype)

    sched = timestep_schedule(num_steps=num_steps, shift=timestep_shift)
    loop_sched = sched               # anchor "dense" mode swaps in a [t_start,0] schedule
    if verbose:
        print(f"  schedule: {[round(float(sched[i]),4) for i in range(min(6,num_steps+1))]} ...")

    # --- B7: img2img-seed all windows from a coherent single-shot anchor -------
    # The anchor is a clean z0 (1,total_t,...) from a real single-shot generate
    # (full global attention = coherent composition). We re-noise it to a schedule
    # time t0 and let MD refine only the tail of the schedule. The rectified-flow
    # forward path z_t=(1-t)*z0 + t*eps is EXACTLY the inverse of the loop's
    # z=z-v*dt update, so re-noising to a real schedule value t0=sched[step0] keeps
    # the truncated loop self-consistent (no schedule mismatch). step0 = first step
    # whose t falls at/below anchor_t_start; the windows then inherit global framing
    # from the anchor and only repaint local detail -> attacks MD drift at the root.
    step0 = 0
    if anchor_latents is not None:
        want = (1, total_t, h_lat, w_lat, VAE_LATENT_CHANNELS)
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
            print(f"  [anchor] img2img seed t_start={anchor_t_start} mode={anchor_schedule} "
                  f"-> step0={step0}, top t={float(loop_sched[step0]):.4f}; "
                  f"refining {num_steps-step0}/{num_steps} steps from anchor")

    # One solver per generate(): the MD step below is a single global integration
    # over loop_sched (the per-window velocities are blended, then ONE update
    # advances the whole buffer), so the AB2 history is the global trajectory's.
    solver = DPMSolverPlusPlus2M() if scheduler == "dpm" else None

    tg = time.perf_counter()
    for step in range(step0, num_steps):
        t = loop_sched[step]
        dt = loop_sched[step] - loop_sched[step + 1]
        t_scalar = float(t.item()) if hasattr(t, "item") else float(t)
        cfg_active = (t_scalar > cfg_lo) and (t_scalar <= cfg_hi)
        cfg_scale_step = cfg_scale if cfg_active else 1.0

        # accumulate window velocities into a global buffer (overlaps summed)
        v_accum = mx.zeros_like(latents)
        for wi, s in enumerate(starts):
            z_win = latents[:, s:s + W, :, :, :]
            v_win = _window_velocity(
                pipe, z_win=z_win, t=t, lpe_indices=win_lpe[wi],
                cond_state=win_cond[wi], uncond_state=win_uncond[wi],
                cfg_scale_step=cfg_scale_step,
                cfg_renorm_type=cfg_renorm_type, cfg_renorm_min=cfg_renorm_min,
                W=W, h_lat=h_lat, w_lat=w_lat,
            )
            # taper-weight then pad to full length and add (bounded mem: eval now)
            pad = [(0, 0), (s, total_t - s - W), (0, 0), (0, 0), (0, 0)]
            v_accum = v_accum + mx.pad(v_win * taper, pad)
            mx.eval(v_accum)

        velocity = v_accum * inv_counts          # MultiDiffusion (taper-weighted) blend
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
        print(f"  [multidiff] gen {gen_s:.1f}s  peak={mx.get_peak_memory()/1e9:.2f}GB "
              f"active={mx.get_active_memory()/1e9:.2f}GB")

    if return_latents:
        mx.eval(latents)
        return latents

    # Relay shed cascade (mirrors t2v.generate): the VAE decode never touches the
    # MoT backbone, and under relay the pipe is single-shot anyway (UND is already
    # gone) — so drop the dead prefix caches + GEN tower before decoding. Decode
    # peak then ≈ VAE-only + transient instead of GEN (~5.5 GB) + transient.
    # parallel keeps GEN resident (headroom machines; matches single-shot).
    if pipe.memory_mode == "relay":
        for cond_i in cond_states:
            cond_i["caches"] = None
        if uncond_state is not None:
            uncond_state["caches"] = None
        win_cond = win_uncond = None     # per-window copies also hold cache refs
        shed = pipe.lance_model.free_gen_tower()
        if verbose:
            print(f"  [opt] shed GEN tower before decode: freed "
                  f"{shed['freed_bytes']/1e9:.2f} GB, active "
                  f"{shed['active_after']/1e9:.2f} GB")

    return _decode(pipe, latents, verbose=verbose)


def _decode(
    pipe: TextToVideoPipeline,
    latents: mx.array,
    *,
    verbose: bool,
) -> mx.array:
    """LOSSLESS streaming VAE decode (bit-identical to whole-seq) — mirrors
    generate()'s decode block, bounding the decode transient for long clips.
    chunk_lat=1 = flat-in-length temporal stream; max_n=4 caps the suffix tiling
    (video = multi-chunk → the per-tile-cache U-curve is live)."""
    from mlx_video.models.wan_2.vae22 import denormalize_latents
    from lance_mlx.model.vae_stream import decode_streaming, suggest_spatial_tiles

    if verbose:
        mx.reset_peak_memory()
    pipe._materialize_vae()   # relay: the decoder was left lazy until its phase
    z = denormalize_latents(latents).astype(pipe.vae_decoder.conv2.weight.dtype)
    n_tiles = suggest_spatial_tiles(z.shape[2], z.shape[3], max_n=4)
    decoded = decode_streaming(pipe.vae_decoder, z, chunk_lat=1, spatial_tiles=n_tiles)
    mx.eval(decoded)
    if verbose:
        print(f"  [decode] peak={mx.get_peak_memory()/1e9:.2f}GB "
              f"active={mx.get_active_memory()/1e9:.2f}GB")

    frames_np = np.array(decoded[0].astype(mx.float32))
    return ((frames_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
