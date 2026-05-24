#!/usr/bin/env python3
"""Phase 5n / D2 — CFG renorm scale logger.

Directly tests open hypothesis H3 from L2 audit:
"CFG renorm at higher token counts — not yet tested (~16 min t2v rerun)".

Reimplements the Euler loop outside the pipeline with per-step logging
of CFG renormalization statistics:
  - norm_cond:  L2 norm of v_cond per spatial cell
  - norm_cfg:   L2 norm of v_cfg  per spatial cell (= v_uncond + scale*(v_cond - v_uncond))
  - ratio:      norm_cond / norm_cfg   (the "would-be" scale factor)
  - scale:      clip(ratio, cfg_renorm_min, 1.0)  (the applied scale factor)

Under the channel renorm in production: when ratio << 1, the renorm clamps
v_cfg back to |v_cond|, silently SUPPRESSING the high-cfg-scale velocity.

Hypothesis: at higher n_lat (more spatial cells), some cells produce
disproportionate norm_cfg values, pulling the effective scale down.
This is plausibly the cause of prompt-adherence degradation at scale.

Compares:
  - t2i at 256², 384², 512²     (n_lat = 256, 576, 1024)
  - t2v at 256²×9f, 256²×17f    (n_lat = 768, 1280)
                                  (and 384²×9f if time permits → 1728)

For each run, prints per-step:
  step  t       cond_mean   cfg_mean   ratio_mean   ratio_min   scale_mean   v_reduce_%

If t2v's `scale_mean` consistently sits well below 1.0 while t2i's stays
near 1.0, CFG renorm is silently suppressing prompt-following.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LANCE_IMAGE_WEIGHTS = REPO_ROOT.parent / "lance-mlx-models" / "Lance-3B-bf16"
LANCE_VIDEO_WEIGHTS = REPO_ROOT.parent / "lance-mlx-models" / "Lance-3B-Video-bf16"
VAE_SAFETENSORS     = LANCE_IMAGE_WEIGHTS / "vae.safetensors"
OUT_DIR             = REPO_ROOT / "notes" / "phase5n_diagnostics" / "d2_cfg_renorm"

# Shared diagnostic prompts — concrete subject + setting so the LLM produces
# meaningful conditioning gradient. Use the same prompt across pipelines so
# the only variable is n_lat scaling.
T2I_PROMPT = ("A photorealistic tabby cat sitting in a sunlit garden, "
              "holding a colorful red STOP sign.")
T2V_PROMPT = ("A red panda surfing on a turquoise wave at sunset, "
              "spray of water visible.")


def renorm_stats(v_cond_mx, v_cfg_mx, cfg_renorm_min=0.0):
    """Reproduce t2i/t2v CFG renorm computation; return per-step stats."""
    # Channel renorm: per-cell L2 along channel axis
    norm_cond = mx.sqrt(mx.sum(v_cond_mx * v_cond_mx, axis=-1, keepdims=True))
    norm_cfg  = mx.sqrt(mx.sum(v_cfg_mx  * v_cfg_mx,  axis=-1, keepdims=True))
    ratio = norm_cond / (norm_cfg + 1e-8)
    scale = mx.clip(ratio, cfg_renorm_min, 1.0)

    nc_np = np.array(norm_cond.astype(mx.float32)).ravel()
    nf_np = np.array(norm_cfg.astype(mx.float32)).ravel()
    r_np  = np.array(ratio.astype(mx.float32)).ravel()
    s_np  = np.array(scale.astype(mx.float32)).ravel()

    return {
        "cond_mean":  float(nc_np.mean()),
        "cond_max":   float(nc_np.max()),
        "cfg_mean":   float(nf_np.mean()),
        "cfg_max":    float(nf_np.max()),
        "ratio_mean": float(r_np.mean()),
        "ratio_min":  float(r_np.min()),
        "ratio_med":  float(np.median(r_np)),
        "scale_mean": float(s_np.mean()),
        "scale_min":  float(s_np.min()),
        "n_cells":    int(r_np.size),
        # What fraction of cells have ratio < 1 (i.e., would be clipped)?
        "frac_clipped":  float((r_np < 1.0).mean()),
        # Effective velocity-magnitude reduction (1 = no reduction)
        "v_mag_after_renorm_pct": float(s_np.mean() * 100),
    }


def run_t2i(pipe, prompt, *, height, width, seed=42, num_steps=30,
            cfg_scale=4.0, cfg_renorm_min=0.0):
    """Custom Euler loop for t2i with CFG renorm stats logging."""
    from lance_mlx.model.flow_head import timestep_schedule

    VAE_SPATIAL_DOWNSAMPLE = 16
    VAE_LATENT_CHANNELS = 48
    h_lat = height // VAE_SPATIAL_DOWNSAMPLE
    w_lat = width  // VAE_SPATIAL_DOWNSAMPLE
    n_lat = h_lat * w_lat
    max_side = 64
    lpe_indices = mx.array(
        [r * max_side + c for r in range(h_lat) for c in range(w_lat)],
        dtype=mx.int32,
    )

    from lance_mlx.pipeline.t2i import T2I_INSTRUCTION
    cond_state = pipe._prepare_state(
        prompt=prompt, instruction=T2I_INSTRUCTION,
        n_lat=n_lat, h_lat=h_lat, w_lat=w_lat, verbose=False,
        latent_pos_base=None,
    )
    uncond_state = pipe._prepare_state(
        prompt="", instruction=T2I_INSTRUCTION,
        n_lat=n_lat, h_lat=h_lat, w_lat=w_lat, verbose=False,
        latent_pos_base=None,
    )

    mx.random.seed(seed)
    latents = mx.random.normal((1, 1, h_lat, w_lat, VAE_LATENT_CHANNELS))
    latents = latents.astype(pipe.lance_model.embed_tokens.weight.dtype)
    sched = timestep_schedule(num_steps=num_steps, shift=3.5)

    per_step = []
    t0 = time.perf_counter()
    for step in range(num_steps):
        t = sched[step]
        dt = sched[step] - sched[step + 1]

        v_cond = pipe._step_velocity(
            state=cond_state, latents=latents, t=t,
            lpe_indices=lpe_indices, n_lat=n_lat,
            h_lat=h_lat, w_lat=w_lat,
        )
        v_uncond = pipe._step_velocity(
            state=uncond_state, latents=latents, t=t,
            lpe_indices=lpe_indices, n_lat=n_lat,
            h_lat=h_lat, w_lat=w_lat,
        )
        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)

        stats = renorm_stats(v_cond, v_cfg, cfg_renorm_min)
        stats["step"] = step
        stats["t"] = float(t)
        per_step.append(stats)

        # apply channel renorm (matches production path)
        norm_cond = mx.sqrt(mx.sum(v_cond * v_cond, axis=-1, keepdims=True))
        norm_cfg  = mx.sqrt(mx.sum(v_cfg  * v_cfg,  axis=-1, keepdims=True))
        scale = mx.clip(norm_cond / (norm_cfg + 1e-8), cfg_renorm_min, 1.0)
        velocity = v_cfg * scale

        latents = latents - velocity * dt
        mx.eval(latents)

    return per_step, time.perf_counter() - t0, n_lat


def run_t2v(pipe, prompt, *, height, width, num_frames, seed=42,
            num_steps=30, cfg_scale=4.0, cfg_renorm_min=0.0):
    """Custom Euler loop for t2v with CFG renorm stats logging."""
    from lance_mlx.model.flow_head import timestep_schedule
    from lance_mlx.pipeline.t2v import (
        T2V_INSTRUCTION,
        VAE_SPATIAL_DOWNSAMPLE,
        VAE_TEMPORAL_DOWNSAMPLE,
        VAE_LATENT_CHANNELS,
        MAX_LATENT_SIDE,
    )

    h_lat = height // VAE_SPATIAL_DOWNSAMPLE
    w_lat = width  // VAE_SPATIAL_DOWNSAMPLE
    t_lat = (num_frames - 1) // VAE_TEMPORAL_DOWNSAMPLE + 1
    n_lat = t_lat * h_lat * w_lat
    lpe_indices = mx.array(
        [
            f * (MAX_LATENT_SIDE ** 2) + r * MAX_LATENT_SIDE + c
            for f in range(t_lat) for r in range(h_lat) for c in range(w_lat)
        ],
        dtype=mx.int32,
    )

    cond_state = pipe._prepare_state(
        prompt=prompt, instruction=T2V_INSTRUCTION,
        n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat, verbose=False,
        mape_anchor=None, uncond_no_text=False,
        spatial_merge_size=1, prompt_format="ours",
        latent_pos_base=0,
    )
    uncond_state = pipe._prepare_state(
        prompt="", instruction=T2V_INSTRUCTION,
        n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat, verbose=False,
        mape_anchor=None, uncond_no_text=False,
        spatial_merge_size=1, prompt_format="ours",
        latent_pos_base=0,
    )

    mx.random.seed(seed)
    latents = mx.random.normal((1, t_lat, h_lat, w_lat, VAE_LATENT_CHANNELS))
    latents = latents.astype(pipe.lance_model.embed_tokens.weight.dtype)
    sched = timestep_schedule(num_steps=num_steps, shift=3.5)

    per_step = []
    t0 = time.perf_counter()
    for step in range(num_steps):
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
        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)

        stats = renorm_stats(v_cond, v_cfg, cfg_renorm_min)
        stats["step"] = step
        stats["t"] = float(t)
        per_step.append(stats)

        norm_cond = mx.sqrt(mx.sum(v_cond * v_cond, axis=-1, keepdims=True))
        norm_cfg  = mx.sqrt(mx.sum(v_cfg  * v_cfg,  axis=-1, keepdims=True))
        scale = mx.clip(norm_cond / (norm_cfg + 1e-8), cfg_renorm_min, 1.0)
        velocity = v_cfg * scale

        latents = latents - velocity * dt
        mx.eval(latents)

    return per_step, time.perf_counter() - t0, n_lat


def print_per_step(per_step, header):
    print(f"\n  {header}")
    print(f"  {'step':>4s} {'t':>6s} {'cond_mean':>10s} {'cfg_mean':>10s} "
          f"{'ratio_min':>10s} {'ratio_med':>10s} {'scale_mean':>11s} "
          f"{'frac_clip':>10s}")
    for s in per_step:
        print(f"  {s['step']:>4d} {s['t']:>6.3f} "
              f"{s['cond_mean']:>10.4f} {s['cfg_mean']:>10.4f} "
              f"{s['ratio_min']:>10.4f} {s['ratio_med']:>10.4f} "
              f"{s['scale_mean']:>11.4f} {s['frac_clipped']:>10.3f}")


def summarize(per_step, label, n_lat):
    sm = np.array([s["scale_mean"] for s in per_step])
    fc = np.array([s["frac_clipped"] for s in per_step])
    rm = np.array([s["ratio_med"] for s in per_step])
    return {
        "label": label,
        "n_lat": n_lat,
        "num_steps": len(per_step),
        "scale_mean_overall": float(sm.mean()),
        "scale_mean_first_half": float(sm[:len(sm)//2].mean()),
        "scale_mean_last_half":  float(sm[len(sm)//2:].mean()),
        "frac_clipped_overall":  float(fc.mean()),
        "ratio_med_overall":     float(rm.mean()),
    }


def main(argv) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    do_t2i = "t2i" in argv or "all" in argv or len(argv) <= 1
    do_t2v = "t2v" in argv or "all" in argv or len(argv) <= 1

    summaries = []

    # --- t2i runs ---
    if do_t2i:
        print(f"\n╔══════ T2I runs ══════════════════════════════════════════════════╗")
        print(f"  loading TextToImagePipeline ...")
        t0 = time.perf_counter()
        from lance_mlx.pipeline.t2i import TextToImagePipeline
        pipe_t2i = TextToImagePipeline.from_pretrained(
            lance_weights_dir=LANCE_IMAGE_WEIGHTS,
            vae_safetensors=VAE_SAFETENSORS,
        )
        print(f"  loaded in {time.perf_counter()-t0:.1f}s")

        for (h, w) in [(256, 256), (384, 384), (512, 512)]:
            print(f"\n──── t2i {h}×{w}  (n_lat={h*w//256}) ────────────────────────────")
            ps, dt, n_lat = run_t2i(pipe_t2i, T2I_PROMPT, height=h, width=w)
            print(f"  Euler loop: {dt:.1f}s for {len(ps)} steps")
            print_per_step(ps, f"t2i {h}×{w}")
            s = summarize(ps, f"t2i_{h}x{w}", n_lat)
            summaries.append(s)
            print(f"  → overall scale_mean={s['scale_mean_overall']:.4f}  "
                  f"frac_clipped={s['frac_clipped_overall']:.3f}")

        del pipe_t2i
        import gc; gc.collect()
        mx.metal.clear_cache()

    # --- t2v runs ---
    if do_t2v:
        print(f"\n╔══════ T2V runs ══════════════════════════════════════════════════╗")
        print(f"  loading TextToVideoPipeline ...")
        t0 = time.perf_counter()
        from lance_mlx.pipeline.t2v import TextToVideoPipeline
        pipe_t2v = TextToVideoPipeline.from_pretrained(
            lance_weights_dir=LANCE_VIDEO_WEIGHTS,
            vae_safetensors=VAE_SAFETENSORS,
        )
        print(f"  loaded in {time.perf_counter()-t0:.1f}s")

        cases = [
            (256, 256, 9),    # n_lat = 3*16*16 = 768
            (256, 256, 17),   # n_lat = 5*16*16 = 1280
            (384, 384, 9),    # n_lat = 3*24*24 = 1728
        ]
        for (h, w, nf) in cases:
            t_lat = (nf - 1) // 4 + 1
            print(f"\n──── t2v {h}×{w}×{nf}f  (t_lat={t_lat}, n_lat={t_lat*h*w//256}) ────")
            ps, dt, n_lat = run_t2v(pipe_t2v, T2V_PROMPT,
                                     height=h, width=w, num_frames=nf)
            print(f"  Euler loop: {dt:.1f}s for {len(ps)} steps")
            print_per_step(ps, f"t2v {h}×{w}×{nf}f")
            s = summarize(ps, f"t2v_{h}x{w}x{nf}f", n_lat)
            summaries.append(s)
            print(f"  → overall scale_mean={s['scale_mean_overall']:.4f}  "
                  f"frac_clipped={s['frac_clipped_overall']:.3f}")

    # --- final summary ---
    print(f"\n╔══════ Summary: CFG renorm scale by config ════════════════════════╗")
    print(f"  {'label':<24s} {'n_lat':>6s} {'scale_avg':>10s} {'first_half':>11s} "
          f"{'last_half':>10s} {'frac_clip':>10s} {'ratio_med':>10s}")
    for s in summaries:
        print(f"  {s['label']:<24s} {s['n_lat']:>6d} "
              f"{s['scale_mean_overall']:>10.4f} {s['scale_mean_first_half']:>11.4f} "
              f"{s['scale_mean_last_half']:>10.4f} {s['frac_clipped_overall']:>10.3f} "
              f"{s['ratio_med_overall']:>10.4f}")

    # save JSON
    out_json = OUT_DIR / "summary.json"
    out_json.write_text(json.dumps(summaries, indent=2))
    print(f"\n  saved: {out_json}")

    print(f"\n=== Interpretation ===")
    print(f"  scale_mean ≈ 1.0  → CFG signal preserved, prompt adherence undamped")
    print(f"  scale_mean < 0.7  → CFG signal heavily clamped — prompt adherence suppressed")
    print(f"  frac_clip ≈ 1.0   → essentially every spatial cell is being clamped")
    print(f"  ratio_med (true ratio): if << 1, even median cell would be suppressed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
