#!/usr/bin/env python3
"""Cross-feed: PyTorch-fp32 ViT features → MLX Lance decoder.

If the decoder produces the PyTorch capture's answers when given PT ViT
features, the remaining oracle mismatches are carried by ViT backend drift
(PT-vs-MLX fp32 already shows worst-token cosine ~0.886 after the merger);
if not, the decoder still owes an explanation.

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/49_crossfeed_pt_vit.py \
        --cases 01,02,03,04,06 --resolution image_768res
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="01,02,03,04,05,06")
    ap.add_argument("--resolution", default="image_768res")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--lance-weights", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-bf16"))
    ap.add_argument("--vit-weights", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16/vit.safetensors"))
    args = ap.parse_args()

    result_dirs = sorted(Path(".").glob("tests/fixtures/results/x2t_image_sample_*"))
    oracle = json.loads((result_dirs[0] / "result.json").read_text())

    # ---- PT ViT (fp32) ------------------------------------------------------
    import torch
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLVisionConfig,
    )
    from safetensors.torch import load_file
    cfg = json.loads((args.lance_weights / "config.json").read_text())
    vc = dict(cfg["vision_config"])
    vc.setdefault("in_chans", vc.pop("in_channels", 3))
    pt_vit = Qwen2_5_VisionTransformerPretrainedModel(
        Qwen2_5_VLVisionConfig(**{k: v for k, v in vc.items() if k != "model_type"}))
    sd = load_file(str(args.vit_weights))
    sd = {k.removeprefix("vision_tower."): v for k, v in sd.items()}
    pt_vit.load_state_dict(sd, strict=False)
    pt_vit = pt_vit.float().eval()

    # ---- MLX pipeline -------------------------------------------------------
    import mlx.core as mx
    from lance_mlx.pipeline.understanding import UnderstandingPipeline
    from lance_mlx.pipeline.upstream_und_preprocess import (
        load_image_upstream, preprocess_und_image,
    )
    pipe = UnderstandingPipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vit_safetensors=args.vit_weights,
    )

    class PTVisionShim:
        """Stands in for pipe.vision_model: returns precomputed PT features."""
        def __init__(self, original):
            self.patch_embed = original.patch_embed  # dtype probe
            self.features = None

        def __call__(self, pixel_values, grid_thw):
            return self.features

    shim = PTVisionShim(pipe.vision_model)
    pipe.vision_model = shim

    for case in args.cases.split(","):
        fname = f"image-understanding-case-{case}.png"
        entry = next(e for e in oracle if e["image"].endswith(fname))
        img = load_image_upstream(Path("tests/fixtures/images") / fname)

        pixels, grid = preprocess_und_image(img, resolution=args.resolution)
        with torch.no_grad():
            out = pt_vit(hidden_states=torch.from_numpy(pixels).float(),
                         grid_thw=torch.from_numpy(grid.astype(np.int64)))
            feats = (out if isinstance(out, torch.Tensor)
                     else out.pooler_output).numpy()
        shim.features = mx.array(feats).astype(mx.bfloat16)

        answer = pipe.generate(
            img, entry["question"],
            max_new_tokens=args.max_new_tokens,
            preprocess="upstream", resolution=args.resolution,
            vision_full_attn=True,
        )
        exact = answer.strip() == entry["answer"].strip()
        print(f"\n=== case {case} {'✓ EXACT' if exact else '≠'} ===")
        print(f"  PT-ViT→MLX: {answer!r}")
        print(f"  PyTorch:    {entry['answer']!r}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
