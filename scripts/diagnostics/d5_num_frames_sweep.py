#!/usr/bin/env python3
"""Phase 5n / D5 — num_frames cliff vs gradient localization.

D4 proved t2v at t_lat=1 ≈ t2i (Lance_3B_Video weights are fine);
degradation appears at t_lat>1. D5 narrows which class of multi-frame
code is the cause.

Sweep num_frames ∈ {1, 5, 9, 13} → t_lat ∈ {1, 2, 3, 4} on the SAME
prompt + seed used in D4 (cat-STOP-sign at 384²).

Decision tree:
  - CLIFF at t_lat=2 (num_frames=5 already shows D4-style damage):
    → "anything triggered by t-axis variation" is the cause.
    → Next bisect: force t-axis = 0 for all latent tokens at t_lat=2;
      if that fixes it, position-IDs at t>0 are the bug;
      if not, LPE indexing into f>0 entries is the bug.
  - GRADIENT (gradual quality drop as t_lat grows):
    → sequence-length effect: mask size, attention dilution, or
      softmax flattening at longer keys.
    → Next bisect: harder, may need attention/mask instrumentation.
  - REVERSE (multi-frame somehow recovers at higher t_lat): unexpected;
    would suggest a t_lat-edge-case bug at small multi-frame.

Cost: ~3-4 min on M5 Max (load + 4 generations + grid).
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
OUT_DIR             = REPO_ROOT / "notes" / "phase5n_diagnostics" / "d5_num_frames_sweep"

# Same prompt + seed as D4 — direct comparison
PROMPT = ("A medium-close photographic portrait shows a tabby cat seated "
          "in a sunlit garden holding a vivid red OCTAGONAL STOP poster "
          "with bold white letters. The cat has bright green eyes and "
          "expressive whiskers; the background has soft greenery.")
HEIGHT = WIDTH = 384
SEED = 42
NUM_STEPS = 30

# num_frames=1,5,9,13 -> t_lat=1,2,3,4
NUM_FRAMES_VARIANTS = [1, 5, 9, 13]


def fft_hf(img_hwc):
    """High-freq FFT energy — proxy for sharpness/detail."""
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

    print(f"=== Phase 5n / D5 — num_frames cliff/gradient localization ===")
    print(f"  prompt:        <{len(PROMPT.split())}-word D4 prompt>")
    print(f"  scale:         {HEIGHT}×{WIDTH}, seed={SEED}, steps={NUM_STEPS}")
    print(f"  num_frames:    {NUM_FRAMES_VARIANTS}")
    print(f"  t_lat (= ⌈n/4⌉): {[(n-1)//4+1 for n in NUM_FRAMES_VARIANTS]}\n")

    print(f"Loading TextToVideoPipeline ...")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=LANCE_VIDEO_WEIGHTS,
        vae_safetensors=VAE_SAFETENSORS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s\n")

    runs = []
    for nf in NUM_FRAMES_VARIANTS:
        t_lat = (nf - 1) // 4 + 1
        h_lat = HEIGHT // 16
        n_lat = t_lat * h_lat * h_lat
        label = f"nf={nf}_tlat={t_lat}"
        print(f"──── {label}  (n_lat={n_lat}) ────")
        t0 = time.perf_counter()
        frames = pipe.generate(
            prompt=PROMPT,
            height=HEIGHT, width=WIDTH, num_frames=nf,
            num_steps=NUM_STEPS, cfg_scale=4.0, seed=SEED,
            verbose=False,
        )
        dt = time.perf_counter() - t0
        T = frames.shape[0]
        print(f"  generated in {dt:.1f}s; T_decoded={T}")
        # Save each frame.
        sub = OUT_DIR / label
        sub.mkdir(exist_ok=True)
        for i in range(T):
            Image.fromarray(frames[i]).save(sub / f"f{i:02d}.png")
        # The "representative" frame for comparison:
        # At t_lat=1, T_decoded=3 (per D1 finding) — take last (sharpest per D1b).
        # At t_lat>=2, T_decoded=(t_lat-1)*4+1+... — take the middle frame.
        if t_lat == 1:
            rep_idx = T - 1
        else:
            rep_idx = T // 2
        rep = frames[rep_idx]
        hf = fft_hf(rep)
        print(f"  rep frame idx={rep_idx}: mean={rep.mean():.2f}  "
              f"std={rep.std():.2f}  FFT_HF={hf:.2e}")
        runs.append((label, t_lat, n_lat, frames, rep_idx, hf))

    # ──── Comparison grid ────────────────────────────────────────────────
    print(f"\n──── Building comparison grid ────")
    cell = HEIGHT
    cols = len(runs)
    margin = 12
    pad = 36
    grid_w = cols * cell + (cols + 1) * margin
    grid_h = cell + 2 * margin + pad
    grid = Image.new('RGB', (grid_w, grid_h), 'black')
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 14)
    except Exception:
        font = ImageFont.load_default()
    for i, (label, t_lat, n_lat, frames, rep_idx, hf) in enumerate(runs):
        x = margin + i * (cell + margin)
        y = margin + pad
        grid.paste(Image.fromarray(frames[rep_idx]), (x, y))
        annot = f"{label}  n_lat={n_lat}\nHF={hf:.1e}"
        draw.text((x + 4, y - pad + 4), annot, fill='yellow', font=font)
    grid_path = OUT_DIR / "_compare_grid.png"
    grid.save(grid_path)
    print(f"  saved: {grid_path}")

    # ──── Summary table ─────────────────────────────────────────────────
    print(f"\n──── Summary: detail by t_lat ─────────────────────────────")
    print(f"  {'t_lat':>5s} {'num_frames':>10s} {'n_lat':>6s} {'FFT_HF':>12s} "
          f"{'Δ_HF_vs_t1':>11s}")
    hf0 = runs[0][5]
    for label, t_lat, n_lat, frames, rep_idx, hf in runs:
        nf = int(label.split('_')[0].split('=')[1])
        delta = (hf - hf0) / hf0 * 100
        print(f"  {t_lat:>5d} {nf:>10d} {n_lat:>6d} {hf:>12.3e} "
              f"{delta:>10.1f}%")

    print(f"\n=== Cliff/gradient interpretation ===")
    print(f"  - CLIFF at t_lat=2: large drop nf=1→nf=5, smaller after")
    print(f"    → position-IDs at t>0 or LPE indexing")
    print(f"  - GRADIENT: monotonic drop across all t_lat values")
    print(f"    → sequence-length effect (mask growth / attention dilution)")
    print(f"  - Look at the grid PNG for visual judgment of text rendering")
    print(f"    quality — the 'cat with STOP sign' subject's text legibility")
    print(f"    is the most discriminative signal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
