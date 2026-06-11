#!/usr/bin/env python3
"""Run RockTalk/Lance-3B-MLX's X→T (image understanding) on the lance-mlx
Phase-0 oracle cases, using THEIR bundled lance_mlx code + THEIR weights,
byte-as-published.

Purpose: cross-check whether the 2/6 chart-value misses (case-02 "29%",
case-04 "1.3 billion") reproduce in the only other MLX port of Lance. If
RockTalk also misses them → shared MLX-ecosystem ceiling (weights/precision).
If RockTalk answers correctly → the gap is pipeline code, not a ceiling.

Run:
    cd /Volumes/DEV_ARCHIVE/lance-mlx && HF_HUB_DISABLE_XET=1 \
    uv run python /tmp/rocktalk-code/run_x2t_oracle.py --cases 02,04
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROCKTALK = Path("/Volumes/DEV_VOL1/VideoResearch/rocktalk-image-weights")
LANCE = Path("/Volumes/DEV_ARCHIVE/lance-mlx")

# Their bundled lance_mlx + inference.py must shadow our installed lance_mlx.
sys.path.insert(0, str(ROCKTALK))

import mlx.core as mx
import numpy as np
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="02,04")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    import lance_mlx
    assert str(ROCKTALK) in lance_mlx.__file__, \
        f"wrong lance_mlx resolved: {lance_mlx.__file__}"
    from lance_mlx.lance import Lance
    import inference as rt_inference
    assert str(ROCKTALK) in rt_inference.__file__

    print(f"lance_mlx: {lance_mlx.__file__}")

    # --- LLM ---------------------------------------------------------------
    t0 = time.perf_counter()
    cfg_json = json.loads((ROCKTALK / "config.json").read_text())
    lance_cfg = rt_inference.build_lance_config(cfg_json)
    model = Lance(lance_cfg)
    model.load_weights(
        list(mx.load(str(ROCKTALK / "model.safetensors")).items()), strict=True,
    )
    mx.eval(model.parameters())
    print(f"LLM loaded in {time.perf_counter()-t0:.1f}s")

    # --- ViT (mlx_vlm VisionModel, their vit.safetensors) -------------------
    t0 = time.perf_counter()
    import inspect
    from mlx_vlm.models.qwen2_5_vl.config import VisionConfig
    from mlx_vlm.models.qwen2_5_vl.vision import VisionModel
    vit_cfg_json = json.loads((ROCKTALK / "vit_config.json").read_text())
    if "in_chans" in vit_cfg_json and "in_channels" not in vit_cfg_json:
        vit_cfg_json["in_channels"] = vit_cfg_json.pop("in_chans")
    fields = set(inspect.signature(VisionConfig).parameters)
    kwargs = {k: v for k, v in vit_cfg_json.items() if k in fields}
    kwargs.setdefault("model_type", "qwen2_5_vl")
    vision_model = VisionModel(VisionConfig(**kwargs))
    raw = mx.load(str(ROCKTALK / "vit.safetensors"))
    # RockTalk keeps the upstream "vision_tower." key prefix; mlx_vlm's
    # VisionModel expects bare keys.
    raw = {k.removeprefix("vision_tower."): v for k, v in raw.items()}
    saved = vision_model.sanitize(raw)
    vision_model.load_weights(list(saved.items()))
    mx.eval(vision_model.parameters())
    model.attach_vit(vision_model)
    print(f"ViT loaded in {time.perf_counter()-t0:.1f}s")

    # --- Processor + oracle --------------------------------------------------
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    tok = processor.tokenizer
    image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")          # 151645
    endoftext_id = tok.convert_tokens_to_ids("<|endoftext|>")    # 151643

    oracle_dir = sorted(LANCE.glob("tests/fixtures/results/x2t_image_sample_*"))[0]
    oracle = json.loads((oracle_dir / "result.json").read_text())

    embed_dtype = model.language_model.model.embed_tokens.weight.dtype
    instruction = "Look at the image carefully and answer the question."

    for case in args.cases.split(","):
        fname = f"image-understanding-case-{case}.png"
        entry = next(e for e in oracle if e["image"].endswith(fname))
        image = Image.open(LANCE / "tests/fixtures/images" / fname).convert("RGB")

        # Lance prompt template (same as our port's "lance" style). Their
        # x2t splices ViT features by explicit positions, so the pad-token
        # identity (image vs video pad) is inert in their pipeline.
        text = (
            f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"<|vision_start|><|image_pad|><|vision_end|>{entry['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = processor(images=image, text=text, return_tensors="np")
        input_ids = np.asarray(inputs["input_ids"][0])
        pixel_values = mx.array(np.asarray(inputs["pixel_values"]))
        grid_thw = np.asarray(inputs["image_grid_thw"])           # (1, 3)
        _, gh, gw = (int(x) for x in grid_thw[0])
        merge = vision_model.config.spatial_merge_size if hasattr(vision_model, "config") else 2
        h_m, w_m = gh // merge, gw // merge

        vit_dtype = vision_model.patch_embed.proj.weight.dtype
        feats = vision_model(pixel_values.astype(vit_dtype), mx.array(grid_thw))
        if feats.ndim == 2:
            feats = feats[None]
        feats = feats.astype(embed_dtype)

        positions = np.where(input_ids == image_pad_id)[0]
        assert positions.size == h_m * w_m, \
            f"pad count {positions.size} != grid {h_m}x{w_m}"

        t0 = time.perf_counter()
        out_ids = model.x2t_generate(
            prompt_text_ids=mx.array(input_ids.astype(np.int32)),
            visual_embeds=feats,
            image_token_positions=mx.array(positions.astype(np.int32)),
            image_grid_hw=(h_m, w_m),
            max_new_tokens=args.max_new_tokens,
            eos_token_id=im_end_id,
            temperature=0.0,
        )
        dt = time.perf_counter() - t0

        clean = [i for i in out_ids if i not in (im_end_id, endoftext_id)]
        answer = tok.decode(clean, skip_special_tokens=True).strip()
        print(f"\n=== case {case} ({dt:.1f}s, {len(out_ids)} tok, grid {h_m}x{w_m}) ===")
        print(f"  Q:        {entry['question']!r}")
        print(f"  RockTalk: {answer!r}")
        print(f"  PyTorch:  {entry['answer']!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
