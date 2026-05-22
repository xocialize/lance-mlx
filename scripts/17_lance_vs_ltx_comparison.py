#!/usr/bin/env python3
"""Run the LTX eval-prompt set through Lance t2v for side-by-side comparison.

Reads /Volumes/DEV_VOL1/VideoResearch/ltx-mlx-eval/prompts/eval_prompts.json
(8 prompts at seed 1234) and runs each through Lance_3B_Video t2v at
spatial 480×704 (matches LTX) × 17 frames (Lance-tractable scale).

Outputs land at /tmp/lance_vs_ltx/<prompt_id>/{video.mp4,mid.png,meta.json}
plus a top-level summary.json indexing all results.

LTX comparable outputs already exist at:
  /Volumes/DEV_VOL1/VideoResearch/ltx-mlx-eval/outputs/phase4/phase4_<id>_ltx23/video.mp4
  /Volumes/DEV_VOL1/VideoResearch/ltx-mlx-eval/outputs/phase4/phase4_<id>_sulphur/video.mp4

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/17_lance_vs_ltx_comparison.py \\
        --lance-weights /Volumes/.../Lance-3B-Video-bf16 \\
        --vae-weights /Volumes/.../Wan22-VAE-bf16/vae.safetensors \\
        --out-dir /tmp/lance_vs_ltx
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


LTX_PROMPTS = Path("/Volumes/DEV_VOL1/VideoResearch/ltx-mlx-eval/prompts/eval_prompts.json")
LTX_OUTPUTS = Path("/Volumes/DEV_VOL1/VideoResearch/ltx-mlx-eval/outputs/phase4")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance-weights", type=Path, required=True,
                    help="Lance_3B_Video weights directory")
    ap.add_argument("--vae-weights", type=Path, required=True,
                    help="Wan2.2 VAE safetensors")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/lance_vs_ltx"))
    ap.add_argument("--height", type=int, default=480,
                    help="Spatial height; LTX used 480")
    ap.add_argument("--width", type=int, default=704,
                    help="Spatial width; LTX used 704")
    ap.add_argument("--num-frames", type=int, default=17,
                    help="Lance frames; LTX used 97 but takes 2-3h at Lance scale. "
                         "17 keeps generation tractable.")
    ap.add_argument("--steps", type=int, default=30,
                    help="Lance default 30 (non-distilled); LTX uses 8 (distilled).")
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--prompts-subset", type=str, default="",
                    help="Comma-separated prompt ids to run (e.g. p01_fox_grass,p05_pixar_sloth). "
                         "Empty = all.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load LTX prompt set.
    eval_data = json.loads(LTX_PROMPTS.read_text())
    prompts = eval_data["prompts"]
    if args.prompts_subset:
        wanted = set(args.prompts_subset.split(","))
        prompts = [p for p in prompts if p["id"] in wanted]

    print(f"┏━━ Lance vs LTX t2v comparison ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ prompts     : {len(prompts)}")
    print(f"┃ scale       : {args.num_frames}f × {args.width}×{args.height}")
    print(f"┃ Lance config: {args.steps} steps, CFG={args.cfg_scale}")
    print(f"┃ out         : {args.out_dir}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Load pipeline ONCE.
    print("\n=== Loading TextToVideoPipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vae_safetensors=args.vae_weights,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    summary = {
        "scale": [args.height, args.width],
        "num_frames": args.num_frames,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "lance_weights": str(args.lance_weights),
        "ltx_outputs_root": str(LTX_OUTPUTS),
        "results": [],
    }

    import imageio
    import numpy as np
    from PIL import Image

    for i, p in enumerate(prompts):
        pid = p["id"]
        prompt = p["text"]
        seed = p["seed"]
        out_sub = args.out_dir / pid
        out_sub.mkdir(parents=True, exist_ok=True)
        mp4_path = out_sub / "video.mp4"
        mid_path = out_sub / "mid.png"
        meta_path = out_sub / "meta.json"

        ltx23_mp4 = LTX_OUTPUTS / f"phase4_{pid}_ltx23" / "video.mp4"
        sulphur_mp4 = LTX_OUTPUTS / f"phase4_{pid}_sulphur" / "video.mp4"

        print(f"\n=== [{i+1}/{len(prompts)}] {pid}  seed={seed} ===")
        print(f"  prompt: {prompt!r}")
        print(f"  LTX23  : {'✓' if ltx23_mp4.exists() else '✗ missing'} {ltx23_mp4}")
        print(f"  Sulphur: {'✓' if sulphur_mp4.exists() else '✗ missing'} {sulphur_mp4}")

        if mp4_path.exists():
            print(f"  SKIP — output already exists: {mp4_path}")
            continue

        t0 = time.perf_counter()
        try:
            frames = pipe.generate(
                prompt,
                num_frames=args.num_frames,
                height=args.height, width=args.width,
                num_steps=args.steps, cfg_scale=args.cfg_scale,
                seed=seed,
            )
        except Exception as e:
            print(f"  ERROR generating {pid}: {e}")
            summary["results"].append({
                "id": pid, "prompt": prompt, "seed": seed,
                "error": str(e),
            })
            continue
        elapsed = time.perf_counter() - t0

        # Save MP4.
        with imageio.get_writer(mp4_path, fps=12, codec="libx264") as writer:
            for f in frames:
                writer.append_data(f)
        # Save middle frame.
        mid = frames[frames.shape[0] // 2]
        Image.fromarray(mid).save(mid_path)
        # Compute inter-frame MAD.
        diffs = [
            float(np.abs(frames[j].astype(np.float32) - frames[j-1].astype(np.float32)).mean())
            for j in range(1, frames.shape[0])
        ]
        inter_mad = float(np.mean(diffs)) if diffs else 0.0

        meta = {
            "id": pid, "prompt": prompt, "seed": seed,
            "category": p.get("category", ""),
            "height": args.height, "width": args.width,
            "num_frames": args.num_frames,
            "decoded_frames": int(frames.shape[0]),
            "steps": args.steps, "cfg_scale": args.cfg_scale,
            "wall_clock_s": round(elapsed, 1),
            "inter_frame_mad": round(inter_mad, 2),
            "ltx23_reference": str(ltx23_mp4) if ltx23_mp4.exists() else None,
            "sulphur_reference": str(sulphur_mp4) if sulphur_mp4.exists() else None,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        summary["results"].append(meta)
        print(f"  generated {frames.shape[0]} frames in {elapsed:.1f}s "
              f"(inter-MAD {inter_mad:.2f}/255)")
        print(f"  saved → {mp4_path}, {mid_path}")

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n✓ Done. Summary at {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
