#!/usr/bin/env python3
"""Phase 5c-2 sanity check: full-8-bit vs UND-only-8-bit on 1 prompt.

5c-2 UND-only-8-bit failed catastrophically on 4/4 prompts. The 5b
note claimed full-8-bit was 'production quality for Lance_3B image' but
the HF tag says broken. Resolve the contradiction with 1 prompt that
made it through 5c-2 best (P3 cat skateboard) on both variants.

If full-8-bit ALSO fails: Lance needs calibration for any quant period.
If full-8-bit succeeds (or is much better): UND-only is a special bad case;
investigate why our skip-list interacts badly with this Lance_3B.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT.parent / "lance-mlx-models"
VAE_SAFETENSORS = MODELS_DIR / "Lance-3B-bf16" / "vae.safetensors"
OUT_DIR = REPO_ROOT / "notes" / "phase5n_diagnostics" / "phase5c2_validation"

PROMPT = ("This photorealistic, Fish-eye lens, low-angle shot captures a "
          "ginger tabby cat confidently balancing on a skateboard in a "
          "sun-dappled park. The cat, with bright orange fur, large round "
          "amber eyes, and a raised tail, gazes directly at the viewer.")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw, ImageFont
    from lance_mlx.pipeline.t2i import TextToImagePipeline

    print(f"=== Phase 5c-2 sanity: full-8-bit vs UND-only-8-bit ===\n")
    results = {}
    for label, sub in [
        ("bf16",       "Lance-3B-bf16"),
        ("8bit-und",   "Lance-3B-8bit-und"),
        ("8bit-full",  "Lance-3B-8bit"),
        ("4bit-und",   "Lance-3B-4bit-und"),
    ]:
        weights = MODELS_DIR / sub
        if not weights.exists():
            print(f"  SKIP {label}: {weights} not found")
            continue
        print(f"╔══ {label} ({sub}) ═════════════════════════════")
        t0 = time.perf_counter()
        try:
            pipe = TextToImagePipeline.from_pretrained(
                lance_weights_dir=weights, vae_safetensors=VAE_SAFETENSORS,
            )
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            continue
        print(f"  loaded in {time.perf_counter()-t0:.1f}s")
        t0 = time.perf_counter()
        img = pipe.generate(
            prompt=PROMPT, height=384, width=384,
            num_steps=30, cfg_scale=4.0, seed=42, verbose=False,
        )
        dt = time.perf_counter() - t0
        arr = np.array(img)
        # FFT high freq
        gray = arr.mean(axis=-1).astype(np.float32)
        f = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.abs(f)
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 8
        mask = np.ones_like(mag); mask[cy-r:cy+r, cx-r:cx+r] = 0
        hf = float((mag * mask).sum())
        print(f"  {dt:.1f}s  mean={arr.mean():.1f}  std={arr.std():.1f}  HF={hf:.2e}")
        img.save(OUT_DIR / f"sanity_{label}.png")
        results[label] = (arr, dt, hf)
        del pipe
        import gc; gc.collect(); mx.clear_cache()

    # Build single-row grid
    print(f"\n──── Building grid ────")
    panels = list(results.items())
    cell = 384; margin = 12; pad = 30
    cols = len(panels)
    grid = Image.new('RGB', (cols * cell + (cols + 1) * margin, cell + 2 * margin + pad), 'black')
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 14)
    except Exception:
        font = ImageFont.load_default()
    for i, (label, (arr, dt, hf)) in enumerate(panels):
        x = margin + i * (cell + margin)
        y = margin + pad
        grid.paste(Image.fromarray(arr), (x, y))
        draw.text((x + 4, y - pad + 4), f"{label}\nHF={hf:.1e}",
                  fill='lightgreen' if label == 'bf16' else 'orange', font=font)
    out = OUT_DIR / "_sanity_all_variants.png"
    grid.save(out)
    print(f"  saved: {out}")
    print(f"\n──── HF detail relative to bf16 ────")
    if "bf16" in results:
        hf0 = results["bf16"][2]
        for label, (arr, dt, hf) in panels:
            d = (hf - hf0) / hf0 * 100
            print(f"  {label:<14s}  HF={hf:.2e}  Δ={d:+7.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
