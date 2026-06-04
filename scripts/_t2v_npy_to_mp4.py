#!/usr/bin/env python3
"""Convert t2v frame-stack .npy files to .mp4 for visual quality review.

Frames are (T, H, W, 3) uint8 in [0,255] (as returned by generate()). CPU-only,
no model load. Writes <name>.mp4 next to each <name>.npy.

Optional B8 frame interpolation (INTERP>1): inserts INTERP-1 motion-compensated
frames between every pair via DIS optical-flow bidirectional warp+blend, and
bumps fps by the same factor so the clip's wall-clock duration is unchanged —
i.e. SMOOTHER playback at the same length, zero GPU. DIS won a ground-truth
held-out-reconstruction A/B (beats linear blend and ffmpeg-aobmc on every test
clip; see scripts/_t2v_interp_eval.py). 256²×61f -> 181f at INTERP=3.

Usage:
  .venv/bin/python scripts/_t2v_npy_to_mp4.py outputs/speed_sweep [FPS] [INTERP]
  .venv/bin/python scripts/_t2v_npy_to_mp4.py outputs/x/clip.npy 8 3
FPS default 8 (a 13-frame clip → ~1.6 s); playback choice only, not intrinsic.
INTERP default 1 (off). DIS preset via env INTERP_PRESET (ultrafast|fast|medium).
"""
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def dis_interpolate(frames: np.ndarray, factor: int, preset: str = "medium") -> np.ndarray:
    """Insert factor-1 optical-flow frames between each pair: T -> (T-1)*factor+1.

    Bidirectional DIS flow + backward warp + linear blend. CPU-only (cv2).
    """
    import cv2  # lazy: only needed when interpolating
    pmap = {
        "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }
    dis = cv2.DISOpticalFlow_create(pmap[preset])
    H, W = frames[0].shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    out = []
    for i in range(len(frames) - 1):
        f0, f1 = frames[i], frames[i + 1]
        g0 = cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        F01 = dis.calc(g0, g1, None)   # f0 -> f1
        F10 = dis.calc(g1, g0, None)   # f1 -> f0
        out.append(f0)
        for k in range(1, factor):
            t = k / factor
            w0 = cv2.remap(f0, gx - t * F01[..., 0], gy - t * F01[..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            w1 = cv2.remap(f1, gx - (1 - t) * F10[..., 0], gy - (1 - t) * F10[..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            blend = (1 - t) * w0.astype(np.float32) + t * w1.astype(np.float32)
            out.append(blend.clip(0, 255).astype(np.uint8))
    out.append(frames[-1])
    return np.stack(out)


def convert(npy: Path, fps: int, interp: int) -> str:
    arr = np.load(npy)
    if arr.dtype != np.uint8:
        arr = arr.clip(0, 255).astype(np.uint8)
    src_t = arr.shape[0]
    suffix = ""
    if interp > 1:
        preset = os.environ.get("INTERP_PRESET", "medium")
        arr = dis_interpolate(arr, interp, preset)
        fps = fps * interp           # keep wall-clock duration constant
        suffix = f", interp x{interp}/{preset}: {src_t}->{arr.shape[0]}f"
    mp4 = npy.with_name(npy.stem + (f"_x{interp}" if interp > 1 else "")).with_suffix(".mp4")
    imageio.mimwrite(mp4, list(arr), fps=fps, codec="libx264",
                     quality=8, macro_block_size=None)
    return f"{mp4.name}  ({arr.shape[0]} frames @ {fps}fps, {arr.shape[1]}×{arr.shape[2]}{suffix})"


def main():
    if len(sys.argv) < 2:
        print(__doc__); raise SystemExit(1)
    target = Path(sys.argv[1])
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    interp = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    npys = sorted(target.glob("*.npy")) if target.is_dir() else [target]
    if not npys:
        print(f"no .npy under {target}"); raise SystemExit(1)
    for npy in npys:
        print("wrote", convert(npy, fps, interp), flush=True)


if __name__ == "__main__":
    main()
