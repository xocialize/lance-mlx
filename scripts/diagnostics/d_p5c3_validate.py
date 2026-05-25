#!/usr/bin/env python3
"""Phase 5c-3e — validate Lance-3B-AWQ-INT4 against bf16 (+ optionally 8bit-und).

Same 4-prompt sweep as Phase 5c-2 for direct comparability. Reads:
  bf16        : Lance-3B-bf16        — reference (everything else compared to it)
  8bit-und    : Lance-3B-8bit-und    — Phase 5c-2 naive baseline (~80% HF loss)
  AWQ-INT4    : Lance-3B-AWQ-INT4    — Phase 5c-3 calibrated quant (this session)

Builds a 3×4 comparison grid + reports per-prompt pixel + HF deltas.

Decision rule:
  - AWQ-INT4 within 10% HF of bf16 → ship as production 4-bit variant
  - AWQ-INT4 between -20% and -10% → publish as "preview", flag known limitations
  - AWQ-INT4 below -20% → debugging needed
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT.parent / "lance-mlx-models"
BF16_WEIGHTS    = MODELS_DIR / "Lance-3B-bf16"
Q8_UND_WEIGHTS  = MODELS_DIR / "Lance-3B-8bit-und"
AWQ_WEIGHTS     = MODELS_DIR / "Lance-3B-AWQ-INT4"
AWQ_INT8_WEIGHTS = MODELS_DIR / "Lance-3B-AWQ-INT8"
VAE_SAFETENSORS = BF16_WEIGHTS / "vae.safetensors"
OUT_DIR = REPO_ROOT / "notes" / "phase5n_diagnostics" / "phase5c3_awq_port" / "validation"

PROMPTS = [
    ("P1_cat_stop",   "A cat holds a poster with rainbow text \"STOP\""),
    ("P2_dragon",     "A fantasy dragon, its body is dark purple gradient, "
                      "its scales shine with dark gold light, its wings are "
                      "covered with dark patterns, spitting dark purple flames "
                      "from its mouth, surrounded by ink-colored clouds and "
                      "glowing stars, with a mysterious starry sky in the background."),
    ("P3_cat_skate",  "This photorealistic, Fish-eye lens, low-angle shot "
                      "captures a ginger tabby cat confidently balancing on a "
                      "skateboard in a sun-dappled park. The cat, with bright "
                      "orange fur, large round amber eyes, and a raised tail, "
                      "gazes directly at the viewer."),
    ("P4_cat_dog",    "A cat and a dog taking a selfie in a snow-covered "
                      "cabin mirror, with scarves and winter hats on. Frost "
                      "on the window and warm indoor lighting add seasonal "
                      "atmosphere."),
]

HEIGHT = WIDTH = 384
SEED = 42
NUM_STEPS = 30


def fft_hf(arr):
    gray = arr.mean(axis=-1).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    mask = np.ones_like(mag); mask[cy - r:cy + r, cx - r:cx + r] = 0
    return float((mag * mask).sum())


def run_variant(label, weights_dir):
    from lance_mlx.pipeline.t2i import TextToImagePipeline
    print(f"\n╔══ {label} ({weights_dir.name}) ══════════════════════════")
    t0 = time.perf_counter()
    pipe = TextToImagePipeline.from_pretrained(
        lance_weights_dir=weights_dir, vae_safetensors=VAE_SAFETENSORS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    results = {}
    for plabel, prompt in PROMPTS:
        t0 = time.perf_counter()
        img = pipe.generate(
            prompt=prompt, height=HEIGHT, width=WIDTH,
            num_steps=NUM_STEPS, cfg_scale=4.0, seed=SEED, verbose=False,
        )
        dt = time.perf_counter() - t0
        arr = np.array(img)
        hf = fft_hf(arr)
        print(f"  {plabel:>14s}:  {dt:>5.1f}s  mean={arr.mean():.1f}  "
              f"std={arr.std():.1f}  HF={hf:.2e}")
        img.save(OUT_DIR / f"{label}_{plabel}.png")
        results[plabel] = (arr, dt, hf)
    del pipe
    import gc; gc.collect(); mx.clear_cache()
    return results


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw, ImageFont

    print(f"=== Phase 5c-3e validation: bf16 vs 8bit-und vs AWQ-INT4 ===")
    print(f"  prompts: {len(PROMPTS)}  scale: {HEIGHT}×{WIDTH}  seed={SEED}\n")

    results = {}
    for label, dir_ in [
        ("bf16",     BF16_WEIGHTS),
        ("8bit-und", Q8_UND_WEIGHTS),
        ("AWQ-INT4", AWQ_WEIGHTS),
        ("AWQ-INT8", AWQ_INT8_WEIGHTS),
    ]:
        if not dir_.exists():
            print(f"SKIP {label}: {dir_} not found")
            continue
        results[label] = run_variant(label, dir_)

    # ─── grid ──────────────────────────────────────────────────────────────
    print(f"\n──── Building 3×4 comparison grid ────")
    cell = HEIGHT
    rows = list(results.keys())
    cols = [plabel for plabel, _ in PROMPTS]
    margin = 12
    pad = 30
    grid_w = len(cols) * cell + (len(cols) + 1) * margin
    grid_h = len(rows) * (cell + pad + margin) + margin
    grid = Image.new('RGB', (grid_w, grid_h), 'black')
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 13)
    except Exception:
        font = ImageFont.load_default()
    for r, label in enumerate(rows):
        for c, plabel in enumerate(cols):
            x = margin + c * (cell + margin)
            y = margin + r * (cell + pad + margin) + pad
            arr, _, hf = results[label][plabel]
            grid.paste(Image.fromarray(arr), (x, y))
            color = {'bf16': 'lightgreen', '8bit-und': 'orange',
                     'AWQ-INT4': 'cyan', 'AWQ-INT8': 'yellow'}.get(label, 'white')
            draw.text((x + 4, y - pad + 4), f"{label}  {plabel}  HF={hf:.1e}",
                      fill=color, font=font)
    grid_path = OUT_DIR / "_compare_grid.png"
    grid.save(grid_path)
    print(f"  saved: {grid_path}")

    # ─── numeric comparison ────────────────────────────────────────────────
    print(f"\n──── HF detail & pixel diff vs bf16 ────")
    print(f"  {'prompt':>14s}  {'variant':>10s}  {'HF':>10s}  {'HF Δ%':>7s}  "
          f"{'pix_mean_diff':>14s}  {'pix_95p':>8s}")
    if "bf16" not in results:
        print("  no bf16 reference; skipping deltas")
        return 0
    for plabel, _ in PROMPTS:
        a = results["bf16"][plabel][0]
        hf_a = results["bf16"][plabel][2]
        for label in rows:
            if label == "bf16":
                print(f"  {plabel:>14s}  {label:>10s}  {hf_a:>10.2e}  "
                      f"{'baseline':>7s}  {'-':>14s}  {'-':>8s}")
            else:
                b = results[label][plabel][0]
                hf_b = results[label][plabel][2]
                d_hf = (hf_b - hf_a) / hf_a * 100
                d_pix = np.abs(a.astype(np.int32) - b.astype(np.int32))
                print(f"  {plabel:>14s}  {label:>10s}  {hf_b:>10.2e}  "
                      f"{d_hf:>+6.1f}%  {d_pix.mean():>14.2f}  "
                      f"{np.percentile(d_pix, 95):>8.1f}")

    print(f"\n=== Verdict heuristic ===")
    if "AWQ-INT4" in results:
        avg_hf_delta = np.mean([
            (results["AWQ-INT4"][p][2] - results["bf16"][p][2]) / results["bf16"][p][2] * 100
            for p, _ in PROMPTS
        ])
        print(f"  AWQ-INT4 average HF Δ vs bf16: {avg_hf_delta:+.1f}%")
        if avg_hf_delta > -10:
            print(f"  → SHIP: HF detail preserved within 10% of bf16")
        elif avg_hf_delta > -20:
            print(f"  → PREVIEW: moderate detail loss; publish with caveats")
        else:
            print(f"  → INVESTIGATE: substantial detail loss; debug before shipping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
