#!/usr/bin/env python3
"""Build a side-by-side HTML comparison page from Lance + LTX outputs.

Reads:
  /tmp/lance_vs_ltx/<id>/{video.mp4,mid.png,meta.json}                  Lance outputs
  /Volumes/.../ltx-mlx-eval/outputs/phase4/phase4_<id>_ltx23/video.mp4   LTX-2.3 outputs
  /Volumes/.../ltx-mlx-eval/outputs/phase4/phase4_<id>_sulphur/video.mp4 Sulphur outputs (optional)

Writes:
  /tmp/lance_vs_ltx/comparison.html       — browser-viewable side-by-side
  /tmp/lance_vs_ltx/comparison.md          — markdown table for repo notes
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path


LTX_ROOT = Path("/Volumes/DEV_VOL1/VideoResearch/ltx-mlx-eval/outputs/phase4")
LANCE_MIDS_DIR = Path("/tmp/lance_vs_ltx_ltx_mids")  # precomputed LTX mid frames


def encode_image(path: Path) -> str:
    """Base64-encode an image for inline HTML."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance-dir", type=Path, default=Path("/tmp/lance_vs_ltx"))
    args = ap.parse_args()

    summary_path = args.lance_dir / "summary.json"
    if not summary_path.exists():
        print(f"No summary.json at {summary_path}. Run scripts/17_lance_vs_ltx_comparison.py first.")
        return 1
    summary = json.loads(summary_path.read_text())

    # HTML report
    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Lance MLX vs LTX-2.3 — Apple Silicon t2v comparison</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,sans-serif;max-width:1400px;margin:2em auto;padding:0 1em;color:#222}",
        "h1{font-size:1.6em} h2{font-size:1.1em;margin-top:2em}",
        "table{border-collapse:collapse;margin:1em 0;width:100%}",
        "td,th{border:1px solid #ccc;padding:8px;vertical-align:top}",
        "td.frame{padding:0;text-align:center}",
        "td.frame img{max-width:100%;height:auto;display:block}",
        ".meta{font-size:.85em;color:#666;padding:6px 8px}",
        ".prompt{font-style:italic;color:#444}",
        ".cfg{background:#f6f6f6;padding:1em;border-radius:6px;margin:1em 0;font-size:.9em;font-family:monospace}",
        "</style></head><body>",
        "<h1>Lance MLX vs LTX-2.3 — Apple Silicon t2v comparison</h1>",
        "<p>Side-by-side middle-frame comparison on the LTX eval-prompt set, all running on M5 Max via MLX.</p>",
        "<div class='cfg'>",
        f"Lance config: <b>{summary['num_frames']}f × {summary['scale'][1]}×{summary['scale'][0]}</b>, "
        f"{summary['steps']} steps, CFG={summary['cfg_scale']}<br>",
        "LTX-2.3 config: 97f × 704×448, 8 steps (distilled). Source: <code>github.com/xocialize-code/ltx-mlx-eval</code><br>",
        "Both at the same seed per prompt.",
        "</div>",
    ]

    md_parts = [
        "# Lance MLX vs LTX-2.3 — Apple Silicon t2v comparison",
        "",
        f"All 14 prompts from the LTX eval set, both models on M5 Max via MLX.",
        "",
        f"- **Lance config:** {summary['num_frames']}f × {summary['scale'][1]}×{summary['scale'][0]}, "
        f"{summary['steps']} steps, CFG={summary['cfg_scale']}",
        f"- **LTX-2.3 config:** 97f × 704×448, 8 steps (distilled pipeline)",
        f"- **Same seed per prompt** ({summary['results'][0].get('seed', 1234) if summary['results'] else 1234})",
        "",
        "| Prompt | Lance MLX | LTX-2.3 | Lance time | LTX inter-MAD vs Lance inter-MAD |",
        "|---|---|---|---|---|",
    ]

    html_parts.append("<table><tr><th>Prompt</th><th>Lance MLX (bf16)</th><th>LTX-2.3 (distilled)</th></tr>")
    for r in summary["results"]:
        if "error" in r:
            html_parts.append(f"<tr><td>{r['id']}<br><span class='prompt'>{r['prompt']}</span></td>"
                              f"<td colspan='2'>ERROR: {r['error']}</td></tr>")
            continue
        lance_mid = args.lance_dir / r['id'] / "mid.png"
        ltx_mid = LANCE_MIDS_DIR / f"{r['id']}_ltx23.png"

        if lance_mid.exists():
            lance_img = f"data:image/png;base64,{encode_image(lance_mid)}"
        else:
            lance_img = ""
        if ltx_mid.exists():
            ltx_img = f"data:image/png;base64,{encode_image(ltx_mid)}"
        else:
            ltx_img = ""

        html_parts.append(
            f"<tr><td><b>{r['id']}</b><br><span class='prompt'>{r['prompt']}</span>"
            f"<br><span class='meta'>category: {r.get('category','')} · "
            f"Lance {r['wall_clock_s']}s · inter-MAD {r['inter_frame_mad']}</span></td>"
            f"<td class='frame'><img src='{lance_img}' alt='Lance'/></td>"
            f"<td class='frame'><img src='{ltx_img}' alt='LTX-2.3'/></td></tr>"
        )

        # Markdown: relative links so the .md file works in github
        lance_md = f"![lance]({r['id']}/mid.png)" if lance_mid.exists() else "—"
        ltx_md = (f"![ltx]({LANCE_MIDS_DIR.relative_to(args.lance_dir.parent) if LANCE_MIDS_DIR.is_relative_to(args.lance_dir.parent) else LANCE_MIDS_DIR}/{r['id']}_ltx23.png)"
                  if ltx_mid.exists() else "—")
        md_parts.append(
            f"| **{r['id']}** ({r.get('category','')})<br>_{r['prompt'][:80]}..._ | "
            f"{lance_md} | {ltx_md} | "
            f"{r['wall_clock_s']}s | "
            f"Lance MAD {r['inter_frame_mad']} |"
        )

    html_parts.append("</table>")
    # Stats summary
    valid = [r for r in summary["results"] if "error" not in r]
    if valid:
        total_lance_s = sum(r["wall_clock_s"] for r in valid)
        avg_lance_s = total_lance_s / len(valid)
        avg_mad = sum(r["inter_frame_mad"] for r in valid) / len(valid)
        html_parts.append(
            f"<h2>Summary stats</h2>"
            f"<ul>"
            f"<li>Lance: {len(valid)} prompts, {total_lance_s:.0f}s total ({avg_lance_s:.0f}s/prompt avg)</li>"
            f"<li>Avg inter-frame MAD (Lance): {avg_mad:.2f}/255</li>"
            f"</ul>"
        )

    html_parts.append("</body></html>")

    html_path = args.lance_dir / "comparison.html"
    html_path.write_text("\n".join(html_parts))
    print(f"✓ wrote {html_path}")
    print(f"  open with: open {html_path}")

    md_path = args.lance_dir / "comparison.md"
    md_path.write_text("\n".join(md_parts))
    print(f"✓ wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
