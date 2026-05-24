#!/usr/bin/env python3
"""Phase 5n / D4 — pipeline-code vs weights isolation.

D1/D2/D3 all refuted. By elimination, the remaining live hypotheses are
training-data imbalance and possibly something in multi-frame code we
haven't fingered yet. D4 directly tests "is the multi-frame code path
the cause?" by collapsing the variable count.

Three runs, identical prompt+seed:

  A.  t2i with Lance_3B at 384²              (baseline; production image)
  B.  t2v with Lance_3B_Video at 384²×1f     (single-frame video; t_lat=1,
                                              equivalent compute to t2i)
  C.  t2v with Lance_3B_Video at 384²×9f     (multi-frame baseline)

Decision tree:
  - A ≈ B (same quality): the multi-frame code path is the gap.
    Follow-up: bisect what's different in t2v's t_lat>1 forward pass
    (mask shape, position-IDs at t>0, attention dilution, etc.).
  - A ≠ B (B is also degraded): the bug is in Lance_3B_Video's weights
    (semantic divergence from Lance_3B) — strongly supports
    training-data-imbalance answer. Conclude.
  - B much worse than A AND C: t2v code at t_lat=1 hits a bad edge
    case unique to the single-frame path. Less likely but worth
    flagging.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LANCE_IMAGE_WEIGHTS = REPO_ROOT.parent / "lance-mlx-models" / "Lance-3B-bf16"
LANCE_VIDEO_WEIGHTS = REPO_ROOT.parent / "lance-mlx-models" / "Lance-3B-Video-bf16"
VAE_SAFETENSORS     = LANCE_IMAGE_WEIGHTS / "vae.safetensors"
OUT_DIR             = REPO_ROOT / "notes" / "phase5n_diagnostics" / "d4_pipeline_isolation"

# Use a discriminative prompt that t2i is known to produce well — the
# Phase 0 t2i oracle "cat-STOP-poster" prompt (prompt id 000001).
PROMPT = ("A medium-close photographic portrait shows a tabby cat seated "
          "in a sunlit garden holding a vivid red OCTAGONAL STOP poster "
          "with bold white letters. The cat has bright green eyes and "
          "expressive whiskers; the background has soft greenery.")

HEIGHT = WIDTH = 384
SEED = 42
NUM_STEPS = 30


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Phase 5n / D4 — pipeline-code vs weights isolation ===")
    print(f"  prompt:   <{len(PROMPT.split())}-word t2i oracle-style prompt>")
    print(f"  scale:    {HEIGHT}×{WIDTH}  (n_lat = {HEIGHT*WIDTH//256} for single frame)")
    print(f"  seed:     {SEED}, steps: {NUM_STEPS}\n")

    from PIL import Image

    # ============= A. t2i baseline =================================
    print(f"──── A. t2i baseline (Lance_3B + t2i pipeline) ────")
    print(f"  loading TextToImagePipeline ...")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2i import TextToImagePipeline
    pipe_t2i = TextToImagePipeline.from_pretrained(
        lance_weights_dir=LANCE_IMAGE_WEIGHTS,
        vae_safetensors=VAE_SAFETENSORS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    t0 = time.perf_counter()
    img_a = pipe_t2i.generate(
        prompt=PROMPT, height=HEIGHT, width=WIDTH,
        num_steps=NUM_STEPS, cfg_scale=4.0, seed=SEED, verbose=False,
    )
    dt = time.perf_counter() - t0
    img_a_np = np.array(img_a)   # (H, W, 3)
    print(f"  generated in {dt:.1f}s; shape={img_a_np.shape}")
    print(f"  stats: mean={img_a_np.mean():.2f} std={img_a_np.std():.2f}")
    img_a.save(OUT_DIR / "A_t2i_baseline.png")

    # Free t2i pipeline before loading t2v (28GB+15GB simultaneously is
    # avoidable; sequential keeps memory pressure modest).
    del pipe_t2i
    import gc; gc.collect(); mx.clear_cache()

    # ============= B. t2v at num_frames=1 (single-frame video) ======
    print(f"\n──── B. t2v with num_frames=1 (Lance_3B_Video + t2v pipeline) ────")
    print(f"  loading TextToVideoPipeline ...")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    pipe_t2v = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=LANCE_VIDEO_WEIGHTS,
        vae_safetensors=VAE_SAFETENSORS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    t0 = time.perf_counter()
    frames_b = pipe_t2v.generate(
        prompt=PROMPT, height=HEIGHT, width=WIDTH, num_frames=1,
        num_steps=NUM_STEPS, cfg_scale=4.0, seed=SEED, verbose=False,
    )
    dt = time.perf_counter() - t0
    print(f"  generated in {dt:.1f}s; T_decoded={frames_b.shape[0]}")
    # Save all decoded frames (will be 3 for t_lat=1 per D1 finding).
    for i in range(frames_b.shape[0]):
        Image.fromarray(frames_b[i]).save(OUT_DIR / f"B_t2v_1f_decoded{i}.png")
    # "Comparable" frame for visual A/B: take the last decoded frame
    # (highest detail per D1b). Also save frame 0 for completeness.
    img_b_np = frames_b[-1]
    img_b0_np = frames_b[0]
    print(f"  frame 0 stats: mean={img_b0_np.mean():.2f} std={img_b0_np.std():.2f}")
    print(f"  last frame stats: mean={img_b_np.mean():.2f} std={img_b_np.std():.2f}")

    # ============= C. t2v at num_frames=9 (production video baseline) =====
    print(f"\n──── C. t2v with num_frames=9 (production multi-frame) ────")
    t0 = time.perf_counter()
    frames_c = pipe_t2v.generate(
        prompt=PROMPT, height=HEIGHT, width=WIDTH, num_frames=9,
        num_steps=NUM_STEPS, cfg_scale=4.0, seed=SEED, verbose=False,
    )
    dt = time.perf_counter() - t0
    print(f"  generated in {dt:.1f}s; T_decoded={frames_c.shape[0]}")
    # Save key frames.
    T = frames_c.shape[0]
    mid = T // 2
    for i in [0, mid, T - 1]:
        Image.fromarray(frames_c[i]).save(OUT_DIR / f"C_t2v_9f_decoded{i:02d}.png")
    img_c_np = frames_c[mid]
    print(f"  mid frame stats: mean={img_c_np.mean():.2f} std={img_c_np.std():.2f}")

    # ============= Comparison grid ====================================
    print(f"\n──── Building comparison grid ────")
    from PIL import ImageDraw, ImageFont
    cell = HEIGHT
    panels = [
        ("A. t2i baseline", img_a_np),
        ("B. t2v 1f, decoded frame 0", img_b0_np),
        ("B. t2v 1f, decoded last", img_b_np),
        ("C. t2v 9f, mid frame", img_c_np),
    ]
    cols = 4
    margin = 12
    pad = 30
    grid_w = cols * cell + (cols + 1) * margin
    grid_h = cell + 2 * margin + pad
    grid = Image.new('RGB', (grid_w, grid_h), 'black')
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 14)
    except Exception:
        font = ImageFont.load_default()
    for i, (label, arr) in enumerate(panels):
        x = margin + i * (cell + margin)
        y = margin + pad
        grid.paste(Image.fromarray(arr), (x, y))
        draw.text((x + 4, y - pad + 5), label, fill='yellow', font=font)
    grid_path = OUT_DIR / "_compare_grid.png"
    grid.save(grid_path)
    print(f"  saved: {grid_path}")

    # ============= Per-pixel diff stats ===============================
    print(f"\n──── Pixel-level deltas between t2i and t2v variants ────")
    def stats(name, a, b):
        d = np.abs(a.astype(np.int32) - b.astype(np.int32))
        print(f"  {name:>40s}:  mean diff={d.mean():.2f}  max={d.max()}  "
              f"95p={np.percentile(d, 95):.1f}")
    stats("A (t2i) vs B (t2v 1f, frame 0)",     img_a_np, img_b0_np)
    stats("A (t2i) vs B (t2v 1f, last frame)",  img_a_np, img_b_np)
    stats("A (t2i) vs C (t2v 9f, mid frame)",   img_a_np, img_c_np)
    stats("B (last) vs C (mid)",                img_b_np, img_c_np)

    print(f"\n=== Interpretation guide ===")
    print(f"  - Visual: open _compare_grid.png. Is B visibly similar to A?")
    print(f"  - If A ≈ B (same subject, similar detail/composition):")
    print(f"    → t2v at t_lat=1 produces image-quality output.")
    print(f"    → multi-frame code is the gap; bisect mask/positions/attention.")
    print(f"  - If B is washed/blurry/different even at t_lat=1:")
    print(f"    → video weights diverged from image weights OR t2v code")
    print(f"      at t_lat=1 already degrades (possible edge case).")
    print(f"    → strongly supports training-data answer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
