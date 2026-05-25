#!/usr/bin/env python3
"""Phase 5c-3c — collect AWQ calibration activation stats.

Loads Lance bf16, installs ActStats hooks on all 504 quant-target
Linears, runs the same 4-prompt diagnostic sweep used in Phase 5c-2
validation, persists per-Linear sum_abs + token counts to disk.

The output drives the AWQ alpha-search in Phase 5c-3d/e — for each
fusion group, act_mean = sum_abs / n_tokens is fed into
`awq_search_scale` to determine the per-channel scale that minimizes
INT4 quantization error.

Why these 4 prompts: same set used in Phase 5c-2 validation, gives
direct comparability with the naive-8bit baseline numbers and exercises:
  P1: text rendering (cat + STOP poster — Lance's hardest)
  P2: saturated color + complex creature (fantasy dragon)
  P3: photorealism + complex scene (cat on skateboard)
  P4: multi-subject composition (cat + dog selfie)

Both UND and GEN tower consumers receive identical stats per call,
since Lance's MoE routing computes both expert paths on the full
sequence then `mx.where`-selects per token. One t2i sweep covers both.

Cost: ~90s (model load + 4× 30 Euler steps × 2 CFG arms = 240 forwards).
Disk: ~2 MB for 504 (in_features,) fp32 arrays.

Usage:
    .venv/bin/python scripts/quant/calibrate_awq.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parents[2]
LANCE_WEIGHTS = REPO_ROOT.parent / "lance-mlx-models" / "Lance-3B-bf16"
VAE_SAFETENSORS = LANCE_WEIGHTS / "vae.safetensors"
OUT_DIR = REPO_ROOT / "notes" / "phase5n_diagnostics" / "phase5c3_awq_port" / "act_stats"

# Same 4 prompts as Phase 5c-2 — direct comparability + good coverage
CALIBRATION_PROMPTS = [
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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Phase 5c-3c — AWQ calibration ===")
    print(f"  scale:    {HEIGHT}×{WIDTH}  seed={SEED}  steps={NUM_STEPS}")
    print(f"  prompts:  {len(CALIBRATION_PROMPTS)} (UND + GEN consumers exercised on every forward)")
    print(f"  out:      {OUT_DIR}\n")

    print(f"Loading Lance bf16 ...")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2i import TextToImagePipeline
    pipe = TextToImagePipeline.from_pretrained(
        lance_weights_dir=LANCE_WEIGHTS, vae_safetensors=VAE_SAFETENSORS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s\n")

    from lance_mlx.quant.calibrate import install_act_stats, save_act_stats

    print(f"Installing ActStats hooks ...")
    t0 = time.perf_counter()
    stats = install_act_stats(pipe.lance_model, verbose=True)
    print(f"  installed in {time.perf_counter()-t0:.1f}s\n")

    print(f"Running calibration sweep ...")
    for label, prompt in CALIBRATION_PROMPTS:
        t0 = time.perf_counter()
        _ = pipe.generate(
            prompt=prompt, height=HEIGHT, width=WIDTH,
            num_steps=NUM_STEPS, cfg_scale=4.0, seed=SEED, verbose=False,
        )
        dt = time.perf_counter() - t0
        # Live progress: pick one representative consumer to show stats are
        # accumulating.
        probe = stats.get("layers.0.self_attn.q_proj")
        if probe is not None and probe.sum_abs is not None:
            n_tok = probe.n_tokens
            mean_chan = float(probe.act_mean.mean())
            print(f"  {label:>14s}:  {dt:>5.1f}s  "
                  f"probe layers.0.q_proj: n_tokens={n_tok:>7d}  mean(|x|)={mean_chan:.4f}")
        else:
            print(f"  {label:>14s}:  {dt:>5.1f}s  (probe empty?)")

    print(f"\nCoverage summary (after all 4 prompts):")
    by_token_count = {}
    no_data = []
    for path, s in stats.items():
        if s.sum_abs is None or s.n_tokens == 0:
            no_data.append(path)
        else:
            # Group by token count bucket for sanity check
            bucket = s.n_tokens // 10000 * 10000
            by_token_count[bucket] = by_token_count.get(bucket, 0) + 1
    print(f"  modules with non-zero stats: {len(stats) - len(no_data)} / {len(stats)}")
    if no_data:
        print(f"  ⚠️  {len(no_data)} modules NEVER invoked:")
        for p in no_data[:6]:
            print(f"     {p}")
        if len(no_data) > 6:
            print(f"     ... and {len(no_data) - 6} more")
    print(f"  token-count distribution (bucketed by 10k):")
    for bucket, count in sorted(by_token_count.items()):
        print(f"    {bucket:>7d}+ tokens: {count:>4d} modules")

    # Spot-check a few specific paths
    print(f"\nSpot-check act_mean stats for representative paths:")
    for sample in [
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.q_proj_moe_gen",
        "layers.18.mlp.up_proj",
        "layers.35.mlp_moe_gen.down_proj",
    ]:
        s = stats.get(sample)
        if s and s.act_mean is not None:
            m = s.act_mean
            print(f"  {sample:<40s}  n_tok={s.n_tokens:>7d}  "
                  f"mean={float(m.mean()):.4f}  "
                  f"min={float(m.min()):.4f}  max={float(m.max()):.4f}")

    # Save.
    print(f"\nSaving ...")
    save_act_stats(stats, OUT_DIR, metadata={
        "calibration_prompts": [label for label, _ in CALIBRATION_PROMPTS],
        "height": HEIGHT, "width": WIDTH,
        "num_steps": NUM_STEPS, "seed": SEED, "cfg_scale": 4.0,
        "source_weights": str(LANCE_WEIGHTS.name),
        "n_target_linears": len(stats),
        "n_with_data": len(stats) - len(no_data),
    })

    print(f"\n=== Done ===")
    print(f"  Output: {OUT_DIR}")
    print(f"    act_stats.safetensors  ({(OUT_DIR / 'act_stats.safetensors').stat().st_size / 1e6:.1f} MB)")
    print(f"    act_stats_meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
