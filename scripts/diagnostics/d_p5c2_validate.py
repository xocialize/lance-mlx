#!/usr/bin/env python3
"""Phase 5c-2 validation — bf16 vs Lance-3B-8bit-und.

Naive 8-bit on UND tower only (GEN tower kept bf16). mlx-lm's DWQ
harness has a `bits < 8` gate, so DWQ isn't applicable at 8-bit; this
script just tests whether 8-bit's precision (256 levels vs 16 for 4-bit)
is enough to preserve t2i quality without calibration.

Runs 4 diverse oracle prompts through bf16 baseline and 8bit-und, same
seed, builds a 2×4 grid. Verdict: ship if 4/4 visually equivalent.

Prompts span quality axes:
  P1: cat + STOP poster      — text rendering (5c-1's hardest)
  P2: fantasy dragon          — saturated colors + complex creature
  P3: cat on skateboard       — photorealism + complex scene
  P4: cat + dog selfie        — multi-subject composition
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BF16_WEIGHTS = REPO_ROOT.parent / "lance-mlx-models" / "Lance-3B-bf16"
Q8_UND_WEIGHTS = REPO_ROOT.parent / "lance-mlx-models" / "Lance-3B-8bit-und"
VAE_SAFETENSORS = BF16_WEIGHTS / "vae.safetensors"
OUT_DIR = REPO_ROOT / "notes" / "phase5n_diagnostics" / "phase5c2_validation"

# Subset of t2i oracle (tests/fixtures/results/t2i_sample_*/prompt.json)
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


def load_pipe(weights_dir: Path):
    from lance_mlx.pipeline.t2i import TextToImagePipeline
    return TextToImagePipeline.from_pretrained(
        lance_weights_dir=weights_dir,
        vae_safetensors=VAE_SAFETENSORS,
    )


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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw, ImageFont

    print(f"=== Phase 5c-2 validation: bf16 vs Lance-3B-8bit-und ===")
    print(f"  prompts:  {len(PROMPTS)}")
    print(f"  scale:    {HEIGHT}×{WIDTH}  seed={SEED}  steps={NUM_STEPS}\n")

    results = {}   # {variant: {prompt_label: (img_np, dt, hf)}}

    for variant_label, weights_dir in [("bf16", BF16_WEIGHTS), ("8bit-und", Q8_UND_WEIGHTS)]:
        print(f"\n╔══ {variant_label} ({weights_dir.name}) ══════════════════════════")
        print(f"  loading ...")
        t0 = time.perf_counter()
        pipe = load_pipe(weights_dir)
        print(f"  loaded in {time.perf_counter()-t0:.1f}s")

        results[variant_label] = {}
        for label, prompt in PROMPTS:
            t0 = time.perf_counter()
            img = pipe.generate(
                prompt=prompt, height=HEIGHT, width=WIDTH,
                num_steps=NUM_STEPS, cfg_scale=4.0, seed=SEED, verbose=False,
            )
            dt = time.perf_counter() - t0
            img_np = np.array(img)
            hf = fft_hf(img_np)
            print(f"  {label:>14s}:  {dt:>5.1f}s  mean={img_np.mean():.1f}  "
                  f"std={img_np.std():.1f}  HF={hf:.2e}")
            img.save(OUT_DIR / f"{variant_label}_{label}.png")
            results[variant_label][label] = (img_np, dt, hf)

        del pipe
        import gc; gc.collect(); mx.clear_cache()

    # ──── Side-by-side per-pair comparison + composite grid ────────────────
    print(f"\n──── Building 2×4 comparison grid ────")
    cell = HEIGHT
    rows = 2
    cols = len(PROMPTS)
    margin = 12
    pad = 32
    grid_w = cols * cell + (cols + 1) * margin
    grid_h = rows * (cell + pad + margin) + margin
    grid = Image.new('RGB', (grid_w, grid_h), 'black')
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 14)
        font_b = ImageFont.truetype('/System/Library/Fonts/Helvetica-Bold.ttc', 16)
    except Exception:
        font = ImageFont.load_default()
        font_b = font

    for r, variant in enumerate(["bf16", "8bit-und"]):
        for c, (label, _) in enumerate(PROMPTS):
            x = margin + c * (cell + margin)
            y = margin + r * (cell + pad + margin) + pad
            img_np, dt, hf = results[variant][label]
            grid.paste(Image.fromarray(img_np), (x, y))
            tag = f"{variant}  {label}\nHF={hf:.1e}  {dt:.1f}s"
            color = 'lightgreen' if variant == 'bf16' else 'lightblue'
            draw.text((x + 4, y - pad + 4), tag, fill=color, font=font)

    grid.save(OUT_DIR / "_compare_grid.png")
    print(f"  saved: {OUT_DIR / '_compare_grid.png'}")

    # ──── numeric comparison ─────────────────────────────────────────────
    print(f"\n──── Pixel + HF delta between bf16 and 8bit-und ────")
    print(f"  {'prompt':>14s}  {'pix_mean_diff':>14s}  {'pix_95p':>8s}  "
          f"{'HF bf16':>10s}  {'HF q8':>10s}  {'HF Δ%':>8s}")
    for label, _ in PROMPTS:
        a = results["bf16"][label][0].astype(np.int32)
        b = results["8bit-und"][label][0].astype(np.int32)
        d = np.abs(a - b)
        hf_a = results["bf16"][label][2]
        hf_b = results["8bit-und"][label][2]
        hf_d = (hf_b - hf_a) / hf_a * 100
        print(f"  {label:>14s}  {d.mean():>14.2f}  {np.percentile(d,95):>8.1f}  "
              f"{hf_a:>10.2e}  {hf_b:>10.2e}  {hf_d:>+7.1f}%")

    print(f"\n=== Verdict heuristic ===")
    print(f"  - pix_mean_diff < 20:  likely visually equivalent")
    print(f"  - 20 ≤ pix_mean_diff < 40:  minor degradation, inspect")
    print(f"  - pix_mean_diff ≥ 40:  significant divergence")
    print(f"  - HF Δ% within ±10%:  detail preserved")
    print(f"  - HF Δ% < -20%:  substantial blurring")
    print(f"\n  Open _compare_grid.png for the visual judgment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
