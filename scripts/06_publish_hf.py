#!/usr/bin/env python3
"""Phase 5c — publish converted Lance weights to mlx-community on HuggingFace.

DRY-RUN BY DEFAULT. Requires --commit AND interactive coordination checklist
confirmation (or --yes-i-coordinated for non-interactive use).

Publish targets (in order):
    1. mlx-community/Lance-3B-bf16
    2. mlx-community/Lance-3B-8bit
    3. mlx-community/Lance-3B-4bit
    4. mlx-community/Lance-3B-Video-bf16
    5. mlx-community/Lance-3B-Video-8bit
    6. mlx-community/Lance-3B-Video-4bit

Each repo gets:
    - Converted MLX safetensors
    - Generated README with architecture + benchmarks + parity results
    - Verbatim LICENSE (Apache 2.0) and NOTICE (attribution to ByteDance / Alibaba)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


@dataclass
class PublishTarget:
    repo_id: str
    src_dir: Path
    variant: str  # "3B" or "3B-Video"
    dtype_label: str  # "bf16" / "8bit" / "4bit" / "mixed"
    upstream: str = "bytedance-research/Lance"


TARGETS_TEMPLATE = [
    # (suffix, variant, dtype_label, src_subdir)
    ("Lance-3B-bf16",        "3B",       "bf16",  "Lance-3B-bf16"),
    ("Lance-3B-8bit",        "3B",       "8bit",  "Lance-3B-8bit"),
    ("Lance-3B-4bit",        "3B",       "4bit",  "Lance-3B-4bit"),
    ("Lance-3B-Video-bf16",  "3B-Video", "bf16",  "Lance-3B-Video-bf16"),
    ("Lance-3B-Video-8bit",  "3B-Video", "8bit",  "Lance-3B-Video-8bit"),
    ("Lance-3B-Video-4bit",  "3B-Video", "4bit",  "Lance-3B-Video-4bit"),
]


README_TEMPLATE = dedent("""\
    ---
    license: apache-2.0
    base_model: bytedance-research/Lance
    library_name: lance-mlx
    tags:
    - mlx
    - lance
    - unified-multimodal
    - text-to-image
    - text-to-video
    - image-editing
    - video-editing
    - apple-silicon
    pipeline_tag: any-to-any
    ---

    # {repo_name}

    > **Note:** "Lance" here refers to **ByteDance Intelligent Creation Lab's unified multimodal model** ([arXiv:2605.18678](https://arxiv.org/abs/2605.18678)), **not** [Lance/LanceDB](https://github.com/lancedb/lance) (the columnar data format).

    MLX-converted weights of [{upstream}](https://huggingface.co/{upstream}) for use with [`lance-mlx`](https://github.com/xocialize/lance-mlx) on Apple Silicon.

    - **Architecture:** Dual-stream Mixture-of-Transformer-Experts. 3B active / ~12B total LLM params.
    - **Variant:** {variant} ({tasks})
    - **Quantization:** {dtype_label}
    - **Source:** [bytedance-research/Lance]({upstream_url})
    - **License:** Apache 2.0

    ## Usage

    ```bash
    pip install lance-mlx
    lance-mlx generate --task t2i --prompt "A red fox in tall grass at golden hour" \\
        --weights {repo_id} --seed 42 --cfg 4.0 --steps 30
    ```

    ## Benchmarks (M5 Max 128 GB, macOS 26.2)

    | Task | Resolution / Frames | Wall-clock | Peak RSS |
    |---|---|---|---|
    | t2i  | 768²                | TODO       | TODO     |
    | t2v  | 480p × 50           | TODO       | TODO     |
    | x2t_image | n/a            | TODO       | TODO     |

    ## Parity vs PyTorch reference

    TODO: FID / CLIPScore / LPIPS results.

    ## Attribution

    - Upstream weights: [bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance) (Apache 2.0)
    - Wan2.2 VAE: Alibaba Wan-AI team (Apache 2.0)
    - Qwen2.5-VL ViT (vision encoder init): Alibaba Qwen team (Apache 2.0)
    - MLX conversion: [xocialize/lance-mlx](https://github.com/xocialize/lance-mlx)
    - Substrate packages: [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm), [Blaizzy/mlx-video](https://github.com/Blaizzy/mlx-video)

    ## Citation

    ```bibtex
    @article{{fu2026lance,
      title={{Lance: Unified Multimodal Modeling by Multi-Task Synergy}},
      author={{Fu, Fengyi and Huang, Mengqi and Wu, Shaojin and others}},
      journal={{arXiv preprint arXiv:2605.18678}},
      year={{2026}}
    }}
    ```
    """)


NOTICE_TEMPLATE = dedent("""\
    {repo_name}

    This product is a derivative of Lance (bytedance-research/Lance), originally
    created by ByteDance Intelligent Creation Lab and released under the Apache
    License 2.0.

    Components:

      Lance LLM weights
          Copyright (c) ByteDance Intelligent Creation Lab
          Licensed under the Apache License, Version 2.0

      Wan2.2 3D causal VAE
          Copyright (c) Alibaba Group / Wan-AI team
          Licensed under the Apache License, Version 2.0

      Qwen2.5-VL ViT (vision encoder, used as init)
          Copyright (c) Alibaba Group / Qwen team
          Licensed under the Apache License, Version 2.0

    MLX conversion and packaging by MVS Collective (https://github.com/xocialize/lance-mlx)
    Licensed under the Apache License, Version 2.0
""")


VARIANT_TASKS = {
    "3B": "t2i, image_edit, x2t_image",
    "3B-Video": "t2v, video_edit, x2t_video (plus all 3B tasks)",
}


def confirm_coordination() -> bool:
    print(dedent("""
        Phase 5b coordination checklist — confirm each:

        [ ] Issue opened on https://github.com/Blaizzy/mlx-vlm and/or
            https://github.com/Blaizzy/mlx-video about the Lance MLX port
        [ ] PR(s) opened for any code changes needed in those packages
        [ ] Email sent to Lance paper corresponding authors (Mengqi Huang,
            Jianzhu Guo) with heads-up about the port
        [ ] Apache 2.0 attribution requirements verified (LICENSE + NOTICE
            staged in every repo)
        [ ] Wan2.2_VAE.pth has been converted to safetensors (no pickle uploads)
        [ ] Write access to mlx-community org confirmed (huggingface-cli whoami)
        [ ] Cross-link plan with bytedance-research/Lance HF page noted in README
    """))
    return input("All confirmed? Type 'yes' to proceed: ").strip().lower() == "yes"


def stage_target(target: PublishTarget, stage_root: Path) -> Path:
    suffix = target.repo_id.split("/")[-1]
    staging = stage_root / suffix
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    if not target.src_dir.exists():
        raise FileNotFoundError(f"Source weights missing: {target.src_dir}")

    for entry in target.src_dir.iterdir():
        if entry.is_file():
            shutil.copy2(entry, staging / entry.name)

    readme = README_TEMPLATE.format(
        repo_id=target.repo_id,
        repo_name=suffix,
        upstream=target.upstream,
        upstream_url=f"https://huggingface.co/{target.upstream}",
        variant=target.variant,
        dtype_label=target.dtype_label,
        tasks=VARIANT_TASKS[target.variant],
    )
    (staging / "README.md").write_text(readme)
    (staging / "NOTICE").write_text(NOTICE_TEMPLATE.format(repo_name=suffix))

    # Bundle the Apache LICENSE (use a local copy or download once)
    license_path = Path("LICENSE")
    if license_path.exists():
        shutil.copy2(license_path, staging / "LICENSE")
    else:
        print(f"  ⚠ LICENSE not at repo root — fetch Apache 2.0 verbatim into {staging / 'LICENSE'}",
              file=sys.stderr)

    return staging


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-root", type=Path, default=Path.home() / "models" / "mlx")
    parser.add_argument("--stage-root", type=Path, default=Path("outputs/phase5_stage"))
    parser.add_argument("--commit", action="store_true",
                        help="Actually push to HuggingFace. Default is dry-run.")
    parser.add_argument("--yes-i-coordinated", action="store_true")
    parser.add_argument("--only", default="",
                        help="Comma-separated suffixes to publish (default: all)")
    args = parser.parse_args()

    if args.commit and not args.yes_i_coordinated:
        if not confirm_coordination():
            print("Aborted by user.", file=sys.stderr)
            return 1

    only = set(args.only.split(",")) if args.only else None
    targets = []
    for suffix, variant, dtype_label, src_subdir in TARGETS_TEMPLATE:
        if only and suffix not in only:
            continue
        targets.append(PublishTarget(
            repo_id=f"mlx-community/{suffix}",
            src_dir=args.mlx_root / src_subdir,
            variant=variant,
            dtype_label=dtype_label,
        ))

    args.stage_root.mkdir(parents=True, exist_ok=True)

    for target in targets:
        print(f"\n=== {target.repo_id} ===")
        try:
            staging = stage_target(target, args.stage_root)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}", file=sys.stderr)
            continue
        print(f"  staged at: {staging}")
        for f in sorted(staging.iterdir()):
            size_mb = f.stat().st_size / 1024**2
            print(f"    {f.name}  ({size_mb:.1f} MB)")

        if args.commit:
            # TODO(claude-code): uncomment after first successful dry-run inspection
            # from huggingface_hub import create_repo, upload_folder
            # create_repo(target.repo_id, repo_type="model", exist_ok=True)
            # upload_folder(
            #     folder_path=str(staging),
            #     repo_id=target.repo_id,
            #     repo_type="model",
            #     commit_message=f"Initial MLX conversion from {target.upstream}",
            # )
            # print(f"  ✓ pushed to https://huggingface.co/{target.repo_id}")
            print(f"  TODO: uncomment upload code in this script after dry-run validates contents")
        else:
            print(f"  DRY-RUN — not pushing. Add --commit to push.")

    if not args.commit:
        print(f"\nDry-run complete. Review {args.stage_root}/ before re-running with --commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
