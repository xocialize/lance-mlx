#!/usr/bin/env python3
"""x2t_image oracle gate with UPSTREAM-EXACT settings.

Replicates the Phase-0 PyTorch capture's exact configuration, derived from
bytedance/Lance's inference_lance.sh + validation_dataset.py:
  - ViT preprocessing: BucketResize(616², bucket AR center-crop, bicubic)
    + DivisibleCrop(28) + CLIP normalize  (RESOLUTION=video_480p)
  - Vision span [<|vision_start|>..<|vision_end|>]: full (bidirectional)
    attention in prefill (dataset attn_mode "full")
  - mRoPE via get_rope_index (apply_qwen_2_5_vl_pos_emb=true in the shell)
  - Lance prompt template, video_pad placeholder, greedy, KV cache

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/46_upstream_exact_gate.py \
        [--cases 01,02,03,04,05,06] [--ablate]

--ablate runs each case 3 ways: hf-causal (old), upstream-causal,
upstream-full-attn — isolating each factor's contribution.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="01,02,03,04,05,06")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--resolution", default="video_480p")
    ap.add_argument("--fp32", action="store_true",
                    help="cast LLM + ViT to fp32 (A100 autocast keeps norms "
                         "fp32; uniform bf16 has a different noise structure)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--lance-weights", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-bf16"))
    ap.add_argument("--vit-weights", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16/vit.safetensors"))
    args = ap.parse_args()

    result_dirs = sorted(Path(".").glob("tests/fixtures/results/x2t_image_sample_*"))
    oracle = json.loads((result_dirs[0] / "result.json").read_text())

    print("=== Loading pipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.understanding import UnderstandingPipeline
    from lance_mlx.pipeline.upstream_und_preprocess import load_image_upstream
    pipe = UnderstandingPipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vit_safetensors=args.vit_weights,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    if args.fp32:
        import mlx.core as mx
        from mlx.utils import tree_map
        pipe.lance_model.update(
            tree_map(lambda p: p.astype(mx.float32),
                     pipe.lance_model.parameters()))
        pipe.vision_model.update(
            tree_map(lambda p: p.astype(mx.float32),
                     pipe.vision_model.parameters()))
        mx.eval(pipe.lance_model.parameters(), pipe.vision_model.parameters())
        print("  cast LLM + ViT to fp32")

    if args.ablate:
        variants = [
            ("hf-causal", dict(preprocess="hf", vision_full_attn=False)),
            ("up-causal", dict(preprocess="upstream", vision_full_attn=False, resolution=args.resolution)),
            ("up-full  ", dict(preprocess="upstream", vision_full_attn=True, resolution=args.resolution)),
        ]
    else:
        variants = [
            ("up-full  ", dict(preprocess="upstream", vision_full_attn=True, resolution=args.resolution)),
        ]

    n_match = 0
    for case in args.cases.split(","):
        fname = f"image-understanding-case-{case}.png"
        entry = next(e for e in oracle if e["image"].endswith(fname))
        image = load_image_upstream(Path("tests/fixtures/images") / fname)

        print(f"\n=== case {case} ===")
        print(f"  Q:        {entry['question']!r}")
        print(f"  PyTorch:  {entry['answer']!r}")
        for name, kw in variants:
            t0 = time.perf_counter()
            answer = pipe.generate(
                image, entry["question"],
                max_new_tokens=args.max_new_tokens, **kw,
            )
            dt = time.perf_counter() - t0
            exact = answer.strip() == entry["answer"].strip()
            if name.startswith("up-full") and exact:
                n_match += 1
            tag = "✓ EXACT" if exact else "≠"
            print(f"  [{name}] {tag} ({dt:.1f}s): {answer!r}")

    print(f"\n=== upstream-exact EXACT matches: {n_match}/{len(args.cases.split(','))} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
