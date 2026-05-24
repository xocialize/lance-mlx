#!/usr/bin/env python3
"""Phase 5n / D6 — position-IDs at t>0 vs LPE at f>0 disambiguation.

D5 showed a CLIFF at t_lat=2 (not a gradient): introducing any second
latent frame drops detail by ~28%. The cause must be one of:
  - position-IDs at t > 0 (mrope sees varying t-axis values)
  - LPE indexing into f > 0 entries (additive embedding from higher
    table slots)
  - both contribute additively

Run four variants at t_lat=2, num_frames=5, on the D4 prompt:

  A. baseline                 — current production code (degraded text)
  B. flatten t-axis to 0      — all latent tokens get t-axis=0 (no mrope variation)
  C. flatten LPE indices to f=0 — all latent tokens get the same f=0 LPE entries
  D. flatten both             — combined

Decision tree:
  - B recovers ≈ t_lat=1 quality:  position-IDs at t>0 are the bug.
  - C recovers ≈ t_lat=1 quality:  LPE at f>0 is the bug.
  - D recovers but B and C don't:  both contribute; both flatten needed.
  - None recovers:                 the bug is elsewhere (mask shape or
                                   attention behavior with larger latent block).

Cost: ~3 min on M5 Max (load + 4 generations + grid).

Implementation: use the public TextToVideoPipeline._prepare_state and
_step_velocity, but surgically rewrite state["position_ids"] and
lpe_indices between state-build and Euler-loop. Doesn't touch
production code.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LANCE_VIDEO_WEIGHTS = REPO_ROOT.parent / "lance-mlx-models" / "Lance-3B-Video-bf16"
VAE_SAFETENSORS     = LANCE_VIDEO_WEIGHTS / "vae.safetensors"
OUT_DIR             = REPO_ROOT / "notes" / "phase5n_diagnostics" / "d6_position_ids_vs_lpe"

PROMPT = ("A medium-close photographic portrait shows a tabby cat seated "
          "in a sunlit garden holding a vivid red OCTAGONAL STOP poster "
          "with bold white letters. The cat has bright green eyes and "
          "expressive whiskers; the background has soft greenery.")
HEIGHT = WIDTH = 384
NUM_FRAMES = 5            # → t_lat=2
SEED = 42
NUM_STEPS = 30


def fft_hf(img_hwc):
    gray = img_hwc.mean(axis=-1).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    mask = np.ones_like(mag)
    mask[cy - r:cy + r, cx - r:cx + r] = 0
    return float((mag * mask).sum())


def flatten_t_axis(state, flatten: bool):
    """Mutate state['position_ids'] so latent positions get t-axis = 0."""
    if not flatten:
        return state
    pos_ids = np.array(state["position_ids"])             # (3, 1, T)
    latent_positions = np.array(state["latent_positions_arr"])
    pos_ids[0, 0, latent_positions] = 0                   # t-axis = 0
    state["position_ids"] = mx.array(pos_ids)
    return state


def build_lpe_indices(t_lat, h_lat, w_lat, flatten_f: bool):
    """Return mx.array of LPE indices. flatten_f=True forces f=0 for all."""
    from lance_mlx.pipeline.t2v import MAX_LATENT_SIDE
    idxs = []
    for f in range(t_lat):
        eff_f = 0 if flatten_f else f
        for r in range(h_lat):
            for c in range(w_lat):
                idxs.append(eff_f * (MAX_LATENT_SIDE ** 2) + r * MAX_LATENT_SIDE + c)
    return mx.array(idxs, dtype=mx.int32)


def run_variant(pipe, *, label, flatten_t, flatten_f):
    """Custom Euler loop at t_lat=2 with surgical overrides."""
    from lance_mlx.model.flow_head import timestep_schedule
    from lance_mlx.pipeline.t2v import (
        T2V_INSTRUCTION, VAE_SPATIAL_DOWNSAMPLE,
        VAE_TEMPORAL_DOWNSAMPLE, VAE_LATENT_CHANNELS,
    )

    h_lat = HEIGHT // VAE_SPATIAL_DOWNSAMPLE
    w_lat = WIDTH  // VAE_SPATIAL_DOWNSAMPLE
    t_lat = (NUM_FRAMES - 1) // VAE_TEMPORAL_DOWNSAMPLE + 1
    n_lat = t_lat * h_lat * w_lat
    lpe_indices = build_lpe_indices(t_lat, h_lat, w_lat, flatten_f=flatten_f)

    print(f"  [{label}] t_lat={t_lat}, n_lat={n_lat}, "
          f"flatten_t={flatten_t}, flatten_f={flatten_f}")

    cond_state = pipe._prepare_state(
        prompt=PROMPT, instruction=T2V_INSTRUCTION,
        n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat,
        verbose=False,
        mape_anchor=None, uncond_no_text=False,
        spatial_merge_size=1, prompt_format="ours",
        latent_pos_base=0,
    )
    uncond_state = pipe._prepare_state(
        prompt="", instruction=T2V_INSTRUCTION,
        n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat,
        verbose=False,
        mape_anchor=None, uncond_no_text=False,
        spatial_merge_size=1, prompt_format="ours",
        latent_pos_base=0,
    )
    cond_state   = flatten_t_axis(cond_state,   flatten=flatten_t)
    uncond_state = flatten_t_axis(uncond_state, flatten=flatten_t)

    mx.random.seed(SEED)
    latents = mx.random.normal((1, t_lat, h_lat, w_lat, VAE_LATENT_CHANNELS))
    latents = latents.astype(pipe.lance_model.embed_tokens.weight.dtype)
    sched = timestep_schedule(num_steps=NUM_STEPS, shift=3.5)

    t0 = time.perf_counter()
    for step in range(NUM_STEPS):
        t = sched[step]
        dt = sched[step] - sched[step + 1]

        v_cond = pipe._step_velocity(
            state=cond_state, latents=latents, t=t,
            lpe_indices=lpe_indices, n_lat=n_lat,
            t_lat=t_lat, h_lat=h_lat, w_lat=w_lat,
        )
        v_uncond = pipe._step_velocity(
            state=uncond_state, latents=latents, t=t,
            lpe_indices=lpe_indices, n_lat=n_lat,
            t_lat=t_lat, h_lat=h_lat, w_lat=w_lat,
        )
        v_cfg = v_uncond + 4.0 * (v_cond - v_uncond)

        # channel renorm (production default)
        norm_cond = mx.sqrt(mx.sum(v_cond * v_cond, axis=-1, keepdims=True))
        norm_cfg  = mx.sqrt(mx.sum(v_cfg  * v_cfg,  axis=-1, keepdims=True))
        scale = mx.clip(norm_cond / (norm_cfg + 1e-8), 0.0, 1.0)
        velocity = v_cfg * scale

        latents = latents - velocity * dt
        mx.eval(latents)
    dt_total = time.perf_counter() - t0

    # VAE decode
    from mlx_video.models.wan_2.vae22 import denormalize_latents
    z = denormalize_latents(latents).astype(pipe.vae_decoder.conv2.weight.dtype)
    decoded = pipe.vae_decoder(z)
    mx.eval(decoded)
    frames_t = decoded[0]
    frames_np = np.array(frames_t.astype(mx.float32))
    frames_u8 = ((frames_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    print(f"  [{label}] Euler+decode: {dt_total + 1:.1f}s, T_decoded={frames_u8.shape[0]}")
    return frames_u8


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw, ImageFont

    print(f"=== Phase 5n / D6 — position-IDs vs LPE at t_lat=2 ===")
    print(f"  prompt: {len(PROMPT.split())}-word D4 cat-STOP-sign")
    print(f"  scale:  {HEIGHT}×{WIDTH}, num_frames={NUM_FRAMES} (t_lat=2)")
    print(f"  seed:   {SEED}\n")

    print(f"Loading TextToVideoPipeline ...")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=LANCE_VIDEO_WEIGHTS,
        vae_safetensors=VAE_SAFETENSORS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s\n")

    variants = [
        ("A_baseline",      False, False),
        ("B_tflat",         True,  False),
        ("C_lpe_fflat",     False, True),
        ("D_both_flat",     True,  True),
    ]

    runs = []
    for label, ft, ff in variants:
        print(f"──── {label} ────")
        frames = run_variant(pipe, label=label, flatten_t=ft, flatten_f=ff)
        # representative frame
        T = frames.shape[0]
        rep_idx = T // 2
        rep = frames[rep_idx]
        hf = fft_hf(rep)
        # save all frames
        sub = OUT_DIR / label
        sub.mkdir(exist_ok=True)
        for i in range(T):
            Image.fromarray(frames[i]).save(sub / f"f{i:02d}.png")
        runs.append((label, ft, ff, frames, rep_idx, rep, hf))
        print(f"  rep frame idx={rep_idx}: mean={rep.mean():.2f}  "
              f"std={rep.std():.2f}  FFT_HF={hf:.2e}\n")

    # ──── grid: 4 panels (A baseline, B t-flat, C lpe-flat, D both) ────
    print(f"──── Building 4-panel comparison grid ────")
    cell = HEIGHT
    cols = len(runs)
    margin = 12
    pad = 50
    grid_w = cols * cell + (cols + 1) * margin
    grid_h = cell + 2 * margin + pad
    grid = Image.new('RGB', (grid_w, grid_h), 'black')
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 13)
    except Exception:
        font = ImageFont.load_default()
    for i, (label, ft, ff, frames, rep_idx, rep, hf) in enumerate(runs):
        x = margin + i * (cell + margin)
        y = margin + pad
        grid.paste(Image.fromarray(rep), (x, y))
        tag = []
        if ft: tag.append("t=0")
        if ff: tag.append("f=0")
        tag_s = " ".join(tag) if tag else "baseline"
        annot = f"{label}\n[{tag_s}]\nHF={hf:.2e}"
        draw.text((x + 4, y - pad + 4), annot, fill='yellow', font=font)
    grid_path = OUT_DIR / "_compare_grid.png"
    grid.save(grid_path)
    print(f"  saved: {grid_path}")

    # ──── summary table ────────────────────────────────────────────────
    print(f"\n──── Summary ────────────────────────────────────────────────")
    a_hf = runs[0][6]
    print(f"  {'variant':>14s} {'flatten_t':>10s} {'flatten_f':>10s} "
          f"{'FFT_HF':>12s} {'ΔvsA':>8s}")
    for label, ft, ff, frames, rep_idx, rep, hf in runs:
        delta = (hf - a_hf) / a_hf * 100
        print(f"  {label:>14s} {str(ft):>10s} {str(ff):>10s} "
              f"{hf:>12.3e} {delta:>+7.1f}%")

    # D5 baseline: at t_lat=1, HF=3.92e+08. We want to know if any variant
    # recovers to that level. If A is ~2.8e+08, we need to compare each
    # variant's HF to 3.92e+08 (the t_lat=1 target).
    t1_target = 3.92e+08
    print(f"\n  Reference: t_lat=1 baseline (D5)  HF=3.92e+08")
    for label, ft, ff, frames, rep_idx, rep, hf in runs:
        recovery_pct = (hf - a_hf) / (t1_target - a_hf) * 100 if (t1_target - a_hf) > 0 else 0
        print(f"  {label:>14s}  HF={hf:.2e}  "
              f"recovery to t_lat=1: {recovery_pct:+.1f}%")

    print(f"\n=== Decision ===")
    print(f"  - If B (t-axis flat) recovers most → position-IDs at t>0 is the bug.")
    print(f"    Fix: pin latent t-axis = 0 in t2v._build_position_ids.")
    print(f"  - If C (LPE f-flat) recovers most → LPE at f>0 is the bug.")
    print(f"    Fix: investigate LPE table semantics for f>0 entries.")
    print(f"  - If D recovers but B,C partial → both contribute; need to fix both.")
    print(f"  - If none recovers → cause is elsewhere (mask, attention).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
