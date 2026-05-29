#!/usr/bin/env python3
"""DPM-Solver++(2M) 12-step vs Euler 30-step — 4-prompt visual A/B diagnostic grid.

Loads the first 4 prompts from prompts/t2i_eval.json. Generates an 8-image
grid (4 rows × 2 columns) plus a Markdown timing report.

  col 0: DPM-Solver++(2M)  12 steps
  col 1: Euler             30 steps
Same seed across all 8 images.

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/43_dpm_euler_ab_grid.py \\
        [--lance-weights PATH] [--vae-weights PATH] \\
        [--seed 42] [--output-dir outputs/ab_grid]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

STEPS_COL  = [12, 30]
SCHED_COL  = ["dpm", "euler"]
LABELS_COL = ["DPM++ 12-step", "Euler 30-step"]

CELL_SIZE = 512
LABEL_H   = 22
HEADER_H  = 38


# ── helpers ──────────────────────────────────────────────────────────────────

def _short(text: str, n: int = 52) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def _burn_label(img, text: str):
    from PIL import Image, ImageDraw
    bar = Image.new("RGB", (img.width, LABEL_H), (25, 25, 25))
    ImageDraw.Draw(bar).text((5, 4), text, fill=(210, 210, 210))
    out = Image.new("RGB", (img.width, img.height + LABEL_H))
    out.paste(bar, (0, 0))
    out.paste(img, (0, LABEL_H))
    return out


def _col_header(text: str, w: int, h: int):
    from PIL import Image, ImageDraw
    cell = Image.new("RGB", (w, h), (18, 18, 55))
    ImageDraw.Draw(cell).text((8, 9), text, fill=(190, 190, 255))
    return cell


def build_grid(rows: list[list]) -> "Image.Image":
    from PIL import Image
    n_rows, n_cols = len(rows), len(rows[0])
    cell_h = rows[0][0].height   # includes LABEL_H already
    w = n_cols * CELL_SIZE
    h = HEADER_H + n_rows * cell_h
    canvas = Image.new("RGB", (w, h), (8, 8, 8))
    for col, lbl in enumerate(LABELS_COL):
        canvas.paste(_col_header(lbl, CELL_SIZE, HEADER_H), (col * CELL_SIZE, 0))
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            if img.size[0] != CELL_SIZE:
                img = img.resize((CELL_SIZE, img.size[1] * CELL_SIZE // img.size[0]))
            canvas.paste(img, (c * CELL_SIZE, HEADER_H + r * cell_h))
    return canvas


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    _snap = ("/Volumes/Crucial500Gb/HUGGINGFACE_HUB_ACTIVE"
             "/models--mlx-community--Lance-3B-bf16/snapshots"
             "/00792588cb6c45cada48cca5c4f77075d82ad34d")
    ap.add_argument("--lance-weights", type=Path, default=Path(_snap))
    ap.add_argument("--vae-weights",   type=Path, default=Path(_snap) / "vae.safetensors")
    ap.add_argument("--prompts-json",  type=Path, default=Path("prompts/t2i_eval.json"))
    ap.add_argument("--seed",          type=int,  default=42)
    ap.add_argument("--cfg-scale",     type=float, default=4.0)
    ap.add_argument("--height",        type=int,  default=768)
    ap.add_argument("--width",         type=int,  default=768)
    ap.add_argument("--n-prompts",     type=int,  default=4)
    ap.add_argument("--output-dir",    type=Path, default=Path("outputs/ab_grid"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    eval_data = json.loads(args.prompts_json.read_text())
    all_prompts = eval_data["prompts"][:args.n_prompts]
    print(f"Prompts loaded: {len(all_prompts)} from {args.prompts_json}\n")

    # Load pipeline
    print("=== Loading pipeline ===")
    t_load = time.perf_counter()
    from lance_mlx.pipeline.t2i import TextToImagePipeline
    pipe = TextToImagePipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vae_safetensors=args.vae_weights,
    )
    load_sec = time.perf_counter() - t_load
    print(f"  loaded in {load_sec:.1f}s\n")

    # Timing log: list of dicts
    timing_rows: list[dict] = []
    grid_rows: list[list] = []
    total = len(all_prompts) * len(SCHED_COL)
    done  = 0

    for row_idx, entry in enumerate(all_prompts):
        pid    = entry["id"]
        prompt = entry["prompt"]
        row_imgs = []

        for col_idx, (sched, steps, label) in enumerate(
                zip(SCHED_COL, STEPS_COL, LABELS_COL)):
            done += 1
            print(f"[{done}/{total}] {pid}  {label}")
            print(f"  {_short(prompt)!r}")

            t0 = time.perf_counter()
            img = pipe.generate(
                prompt,
                height=args.height, width=args.width,
                num_steps=steps, cfg_scale=args.cfg_scale,
                seed=args.seed,
                scheduler=sched,
            )
            elapsed = time.perf_counter() - t0
            print(f"  {elapsed:.1f}s\n")

            # Save individual image
            img_path = args.output_dir / f"{pid}_{sched}{steps}.png"
            img.save(img_path)

            timing_rows.append({
                "id": pid, "scheduler": label, "steps": steps,
                "seed": args.seed, "elapsed_s": round(elapsed, 1),
                "img": img_path.name,
            })

            cell_label = f"{label} | {pid} | {elapsed:.0f}s | seed={args.seed}"
            row_imgs.append(_burn_label(img, cell_label))

        grid_rows.append(row_imgs)

    # Save grid
    print("=== Assembling grid ===")
    grid = build_grid(grid_rows)
    grid_path = args.output_dir / "ab_grid.png"
    grid.save(grid_path)
    print(f"✓ Grid → {grid_path}  ({grid_path.stat().st_size // 1024} KB)")

    # Write Markdown report
    md_path = args.output_dir / "ab_report.md"
    _write_md(md_path, timing_rows, grid_path, load_sec, args)
    print(f"✓ Report → {md_path}")
    return 0


def _write_md(path: Path, rows: list[dict], grid_path: Path,
              load_sec: float, args) -> None:
    lines = [
        "# DPM-Solver++(2M) vs Euler — A/B Timing Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}  ",
        f"**Model:** Lance-3B-bf16  ",
        f"**Seed:** {args.seed}  ",
        f"**Resolution:** {args.width}×{args.height}  ",
        f"**CFG scale:** {args.cfg_scale}  ",
        f"**Pipeline load time:** {load_sec:.1f}s  ",
        "",
        "## Grid",
        "",
        f"![A/B grid]({grid_path.name})",
        "",
        "## Per-image timings",
        "",
        "| Prompt ID | Scheduler | Steps | Time (s) | vs Euler |",
        "|-----------|-----------|-------|----------|----------|",
    ]

    # Pair DPM and Euler rows for speedup calculation
    by_id: dict[str, dict] = {}
    for r in rows:
        by_id.setdefault(r["id"], {})[r["scheduler"]] = r

    for pid, pair in by_id.items():
        dpm   = pair.get("DPM++ 12-step")
        euler = pair.get("Euler 30-step")
        if dpm and euler:
            speedup = euler["elapsed_s"] / dpm["elapsed_s"]
            lines.append(
                f"| {pid} | DPM++ 12-step | 12 | {dpm['elapsed_s']} | **{speedup:.2f}×** faster |"
            )
            lines.append(
                f"| {pid} | Euler 30-step | 30 | {euler['elapsed_s']} | baseline |"
            )
        elif dpm:
            lines.append(f"| {pid} | DPM++ 12-step | 12 | {dpm['elapsed_s']} | — |")
        elif euler:
            lines.append(f"| {pid} | Euler 30-step | 30 | {euler['elapsed_s']} | baseline |")

    # Summary averages
    dpm_times   = [r["elapsed_s"] for r in rows if r["scheduler"] == "DPM++ 12-step"]
    euler_times = [r["elapsed_s"] for r in rows if r["scheduler"] == "Euler 30-step"]
    if dpm_times and euler_times:
        avg_dpm   = sum(dpm_times)   / len(dpm_times)
        avg_euler = sum(euler_times) / len(euler_times)
        lines += [
            "",
            "## Summary",
            "",
            f"| Metric | DPM++ 12-step | Euler 30-step |",
            f"|--------|--------------|---------------|",
            f"| Avg time/image | {avg_dpm:.1f}s | {avg_euler:.1f}s |",
            f"| Speedup | **{avg_euler/avg_dpm:.2f}×** | 1.00× |",
            f"| Total (4 images) | {sum(dpm_times):.0f}s | {sum(euler_times):.0f}s |",
            "",
            "## Individual images",
            "",
        ]
        for r in rows:
            lines.append(f"- **{r['id']} / {r['scheduler']}** — `{r['img']}` ({r['elapsed_s']}s)")

    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
