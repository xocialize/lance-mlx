"""CLI entry point for lance-mlx.

Usage:
    lance-mlx generate --task {t2i,t2v,image_edit,video_edit,x2t_image,x2t_video} [...]

Weights can be a local directory OR a HuggingFace repo ID (e.g.
``mlx-community/Lance-3B-bf16``).  The VAE (``vae.safetensors``) and ViT
(``vit.safetensors``) are expected alongside ``model.safetensors`` inside the
same weights directory.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Weight-path resolution
# ---------------------------------------------------------------------------

def _resolve_weights(weights: str) -> Path:
    """Return a local Path for *weights*, downloading from HF if needed."""
    p = Path(weights)
    if p.exists():
        return p
    # Treat as HF repo ID (contains '/' but is not an absolute or relative path).
    if "/" in weights and not weights.startswith("."):
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=weights)
        return Path(local)
    raise FileNotFoundError(f"Weights not found: {weights!r}")


# ---------------------------------------------------------------------------
# Task handlers
# ---------------------------------------------------------------------------

def _run_t2i(args, weights_dir: Path) -> int:
    from lance_mlx.pipeline.t2i import TextToImagePipeline
    vae = weights_dir / "vae.safetensors"
    if not vae.exists():
        print(f"ERROR: VAE not found at {vae}", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    pipe = TextToImagePipeline.from_pretrained(weights_dir, vae)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    img = pipe.generate(
        args.prompt or "",
        height=args.resolution, width=args.resolution,
        num_steps=args.steps, cfg_scale=args.cfg,
        timestep_shift=args.timestep_shift, seed=args.seed,
    )
    out = Path(args.output) / f"t2i_{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"  saved → {out}")
    return 0


def _run_t2v(args, weights_dir: Path) -> int:
    import numpy as np
    import imageio.v3 as iio
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    vae = weights_dir / "vae.safetensors"
    if not vae.exists():
        print(f"ERROR: VAE not found at {vae}", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    pipe = TextToVideoPipeline.from_pretrained(weights_dir, vae)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    frames = pipe.generate(
        args.prompt or "",
        num_frames=args.frames, height=args.resolution, width=args.resolution,
        num_steps=args.steps, cfg_scale=args.cfg,
        timestep_shift=args.timestep_shift, seed=args.seed,
    )
    if not isinstance(frames, np.ndarray):
        frames = np.array(frames)
    out = Path(args.output) / f"t2v_{args.seed}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(out), frames, fps=args.fps, codec="libx264", quality=8, macro_block_size=1)
    print(f"  saved → {out}  ({len(frames)} frames @ {args.fps} fps)")
    return 0


def _run_image_edit(args, weights_dir: Path) -> int:
    from lance_mlx.pipeline.image_edit import ImageEditPipeline
    if not args.image:
        print("ERROR: --image required for image_edit", file=sys.stderr)
        return 1
    vae = weights_dir / "vae.safetensors"
    if not vae.exists():
        print(f"ERROR: VAE not found at {vae}", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    pipe = ImageEditPipeline.from_pretrained(weights_dir, vae)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    img = pipe.generate(
        args.image, args.prompt or "",
        height=args.resolution, width=args.resolution,
        num_steps=args.steps, cfg_scale=args.cfg,
        timestep_shift=args.timestep_shift, seed=args.seed,
    )
    out = Path(args.output) / f"image_edit_{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"  saved → {out}")
    return 0


def _run_video_edit(args, weights_dir: Path) -> int:
    import numpy as np
    import imageio.v3 as iio
    from lance_mlx.pipeline.video_edit import VideoEditPipeline
    if not args.video:
        print("ERROR: --video required for video_edit", file=sys.stderr)
        return 1
    vae = weights_dir / "vae.safetensors"
    if not vae.exists():
        print(f"ERROR: VAE not found at {vae}", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    pipe = VideoEditPipeline.from_pretrained(weights_dir, vae)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    frames = pipe.generate(
        args.video, args.prompt or "",
        height=args.resolution, width=args.resolution,
        num_frames=args.frames, num_steps=args.steps,
        cfg_scale=args.cfg, seed=args.seed,
    )
    if not isinstance(frames, np.ndarray):
        frames = np.array(frames)
    out = Path(args.output) / f"video_edit_{args.seed}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(out), frames, fps=args.fps, codec="libx264", quality=8, macro_block_size=1)
    print(f"  saved → {out}  ({len(frames)} frames @ {args.fps} fps)")
    return 0


def _run_understanding(args, weights_dir: Path, *, video: bool) -> int:
    from lance_mlx.pipeline.understanding import UnderstandingPipeline
    vit = weights_dir / "vit.safetensors"
    if not vit.exists():
        print(f"ERROR: vit.safetensors not found at {vit}", file=sys.stderr)
        return 1
    media = args.video if video else args.image
    if not media:
        flag = "--video" if video else "--image"
        print(f"ERROR: {flag} required for {'x2t_video' if video else 'x2t_image'}", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    pipe = UnderstandingPipeline.from_pretrained(weights_dir, vit)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    if video:
        answer = pipe.generate_video(media, args.prompt or "")
    else:
        answer = pipe.generate(media, args.prompt or "")
    print(f"\n{answer}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(prog="lance-mlx", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Run a generation or understanding task")
    g.add_argument("--task", required=True,
                   choices=["t2i", "t2v", "image_edit", "video_edit", "x2t_image", "x2t_video"])
    g.add_argument("--prompt", help="Text prompt (or question for x2t_*)")
    g.add_argument("--image", type=Path, help="Input image (i2v, image_edit, x2t_image)")
    g.add_argument("--video", type=Path, help="Input video (video_edit, x2t_video)")
    g.add_argument("--weights", required=True, help="MLX weights repo or local path")
    g.add_argument("--output", type=Path, default=Path("outputs"))
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--steps", type=int, default=30)
    g.add_argument("--cfg", type=float, default=4.0)
    g.add_argument("--timestep-shift", type=float, default=3.5)
    g.add_argument("--resolution", type=int, default=768)
    g.add_argument("--frames", type=int, default=50)
    g.add_argument("--fps", type=int, default=12)

    args = parser.parse_args()

    weights_dir = _resolve_weights(args.weights)
    print(f"weights: {weights_dir}")

    dispatch = {
        "t2i":        lambda: _run_t2i(args, weights_dir),
        "t2v":        lambda: _run_t2v(args, weights_dir),
        "image_edit": lambda: _run_image_edit(args, weights_dir),
        "video_edit": lambda: _run_video_edit(args, weights_dir),
        "x2t_image":  lambda: _run_understanding(args, weights_dir, video=False),
        "x2t_video":  lambda: _run_understanding(args, weights_dir, video=True),
    }
    return dispatch[args.task]()


if __name__ == "__main__":
    sys.exit(main())
