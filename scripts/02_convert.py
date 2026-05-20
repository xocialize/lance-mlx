#!/usr/bin/env python3
"""Phase 1b — convert Lance HF safetensors → MLX format.

Reads bytedance-research/Lance's HF checkpoint (Lance_3B or Lance_3B_Video),
applies the verified key-map (see `notes/phase1b_converter_design.md` for the
full spec), optionally casts dtype to bf16/fp16/fp32, and writes a
single-file MLX safetensors plus provenance metadata.

Two modes:

  --dry-run         Header-only validation via huggingface_hub. No 24 GB
                    download, no file writes. Reports key-map coverage,
                    unmapped keys, expected output counts. Use this to
                    validate the converter logic any time the RULES list
                    changes.

  (default)         Full conversion. Downloads (or uses --local-dir) the
                    HF safetensors, applies the key-map, casts dtype,
                    writes `<output>/model.safetensors`, `<output>/vit.safetensors`
                    (if the variant bundles a ViT), `<output>/config.json`,
                    and `<output>/conversion_report.json`.

Usage:
    # Validate the rule list against both checkpoints without downloading:
    uv run python scripts/02_convert.py \\
        --hf-repo bytedance-research/Lance \\
        --variant lance_3b \\
        --dry-run

    # Real conversion (downloads ~24-28 GB):
    uv run python scripts/02_convert.py \\
        --hf-repo bytedance-research/Lance \\
        --variant lance_3b \\
        --output ~/models/mlx/Lance-3B-bf16 \\
        --dtype bf16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Key-map rules — see notes/phase1b_converter_design.md for full spec.
# Order matters (first match wins). Each rule produces (mlx_key, output_target)
# where output_target is "llm" (goes to model.safetensors) or "vit" (goes to
# vit.safetensors).
# ---------------------------------------------------------------------------

# (pattern, replacement, target)
RULE_TUPLES = [
    # 1. Embeddings + LM head — strip `language_model.{model.}?` prefix
    (r"^language_model\.model\.embed_tokens\.weight$",       r"embed_tokens.weight",            "llm"),
    (r"^language_model\.lm_head\.weight$",                   r"lm_head.weight",                  "llm"),

    # 2. Final norms — strip prefix; preserve `_moe_gen` suffix
    (r"^language_model\.model\.norm\.weight$",               r"norm.weight",                     "llm"),
    (r"^language_model\.model\.norm_moe_gen\.weight$",       r"norm_moe_gen.weight",             "llm"),

    # 3. Layer body — strip prefix; preserve everything else including
    #    layer index and _moe_gen suffixes per piece.
    (r"^language_model\.model\.layers\.(\d+)\.(.+)$",        r"layers.\1.\2",                    "llm"),

    # 4. Lance-specific root-level tensors — no prefix to strip.
    (r"^llm2vae\.(weight|bias)$",                            r"llm2vae.\1",                      "llm"),
    (r"^vae2llm\.(weight|bias)$",                            r"vae_in_proj.vae2llm.\1",          "llm"),
    (r"^latent_pos_embed\.pos_embed$",                       r"latent_pos_embed.pos_embed",      "llm"),

    # 5. TimestepEmbedder — Sequential mlp.0/.2 → scaffold proj_in/proj_out
    (r"^time_embedder\.mlp\.0\.(weight|bias)$",              r"time_embedder.proj_in.\1",        "llm"),
    (r"^time_embedder\.mlp\.2\.(weight|bias)$",              r"time_embedder.proj_out.\1",       "llm"),

    # 6. Bundled ViT (Lance_3B_Video only) — strip `vit_model.` prefix and
    #    route to the separate vit.safetensors output so mlx-vlm's
    #    Qwen2_5_VLVisionModel can load it natively.
    (r"^vit_model\.(.+)$",                                   r"\1",                              "vit"),
]

RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(p), r, t) for p, r, t in RULE_TUPLES
]

# Keys whose values stay F32 even under --dtype bf16/fp16 (numerical stability
# for normalization scales).
KEEP_F32_PATTERNS = [
    re.compile(r".*_layernorm.*\.weight$"),
    re.compile(r"^(.*\.)?norm(_moe_gen)?\.weight$"),
    re.compile(r".*\.q_norm.*\.weight$"),
    re.compile(r".*\.k_norm.*\.weight$"),
]

VARIANT_DIRS = {
    "lance_3b":       "Lance_3B",
    "lance_3b_video": "Lance_3B_Video",
}

# Expected total numels (in B params) — empirically measured from
# huggingface_hub header inspection 2026-05-20. If the input safetensors
# disagrees with these by more than rounding error, something has changed
# upstream and the converter should fail loudly.
#
# Note: Lance_3B and Lance_3B_Video share all 1021 LLM key NAMES, but one
# tensor differs in SHAPE: latent_pos_embed.pos_embed is [4096, 2048] in
# Lance_3B (image-only: 64x64 spatial grid) vs [126976, 2048] in
# Lance_3B_Video (image + video: 64x64 spatial × ~31 temporal slots). The
# extra ~250M params land entirely in this one tensor.
EXPECTED_LLM_PARAMS = {
    "lance_3b":       6.185e9,
    "lance_3b_video": 6.437e9,   # LLM portion only; ViT separates out to vit.safetensors
}
EXPECTED_VIT_PARAMS = {
    "lance_3b":       0.0,        # ships ViT separately as Qwen2.5-VL-ViT/vit.safetensors
    "lance_3b_video": 0.669e9,    # 390 tensors, ~669M params
}


def remap(hf_key: str) -> tuple[str | None, str | None]:
    """Apply the first matching rule. Returns (mlx_key, target) or (None, None)."""
    for pat, repl, target in RULES:
        if pat.match(hf_key):
            return pat.sub(repl, hf_key), target
    return None, None


def should_keep_f32(mlx_key: str) -> bool:
    return any(p.match(mlx_key) for p in KEEP_F32_PATTERNS)


# ---------------------------------------------------------------------------
# Dry-run path: use huggingface_hub.parse_safetensors_file_metadata
# ---------------------------------------------------------------------------

def dry_run(repo_id: str, filename: str, variant: str) -> int:
    from huggingface_hub import parse_safetensors_file_metadata

    print(f"=== DRY RUN: hf://{repo_id}/{filename} (variant={variant}) ===")
    md = parse_safetensors_file_metadata(repo_id=repo_id, filename=filename)

    n_total = len(md.tensors)
    mapped_to: dict[str, list[tuple[str, str, tuple[int, ...]]]] = defaultdict(list)
    unmapped: list[str] = []
    target_numel: Counter[str] = Counter()
    f32_keys: list[str] = []

    for hf_key, info in md.tensors.items():
        mlx_key, target = remap(hf_key)
        if mlx_key is None:
            unmapped.append(hf_key)
            continue
        shape = tuple(info.shape)
        numel = 1
        for d in shape:
            numel *= d
        mapped_to[target].append((hf_key, mlx_key, shape))
        target_numel[target] += numel
        if should_keep_f32(mlx_key):
            f32_keys.append(mlx_key)

    print(f"\n  total tensors:      {n_total}")
    print(f"  mapped:             {n_total - len(unmapped)}")
    print(f"  unmapped:           {len(unmapped)}")
    for target, entries in mapped_to.items():
        print(f"  → {target}.safetensors: {len(entries)} tensors, "
              f"{target_numel[target]/1e9:.3f} B params")
    print(f"  KEEP_F32 keys (under --dtype bf16/fp16): {len(f32_keys)}")

    # Validation gates
    issues: list[str] = []
    if unmapped:
        issues.append(f"{len(unmapped)} unmapped keys (first 5):")
        for k in unmapped[:5]:
            issues.append(f"  - {k}")
    llm_pb = target_numel.get("llm", 0) / 1e9
    if abs(llm_pb - EXPECTED_LLM_PARAMS[variant] / 1e9) > 0.01:
        issues.append(f"LLM param count drift: {llm_pb:.3f} B vs expected "
                      f"{EXPECTED_LLM_PARAMS[variant]/1e9:.3f} B")
    vit_pb = target_numel.get("vit", 0) / 1e9
    if abs(vit_pb - EXPECTED_VIT_PARAMS[variant] / 1e9) > 0.05:
        issues.append(f"ViT param count drift: {vit_pb:.3f} B vs expected "
                      f"{EXPECTED_VIT_PARAMS[variant]/1e9:.3f} B")

    if issues:
        print("\nFAIL:")
        for line in issues:
            print(f"  {line}")
        return 1
    print("\n✓ All keys mapped, param counts match Phase 1a expectations.")
    return 0


# ---------------------------------------------------------------------------
# Real conversion path
# ---------------------------------------------------------------------------

def resolve_source(repo_id: str | None, local_dir: Path | None, variant: str) -> Path:
    if local_dir:
        if not (local_dir / "model.safetensors").exists():
            print(f"ERROR: {local_dir/'model.safetensors'} not found", file=sys.stderr)
            raise SystemExit(1)
        return local_dir
    # HF-fetch path: download just the variant subdir.
    from huggingface_hub import snapshot_download
    subdir = VARIANT_DIRS[variant]
    print(f"Downloading hf://{repo_id} subdir={subdir}/ via snapshot_download ...")
    repo_root = Path(snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{subdir}/*"],
    ))
    return repo_root / subdir


def convert(src_dir: Path, out_dir: Path, dtype: str, variant: str) -> int:
    """Real conversion. Requires mlx and the actual safetensors weight bytes."""
    import mlx.core as mx

    dtype_map = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}
    target_dtype = dtype_map[dtype]

    in_file = src_dir / "model.safetensors"
    print(f"Loading {in_file} ...")
    state = mx.load(str(in_file))   # dict[str, mx.array]
    print(f"  loaded {len(state)} tensors")

    llm_state: dict[str, "mx.array"] = {}
    vit_state: dict[str, "mx.array"] = {}
    unmapped: list[str] = []

    for hf_key, tensor in state.items():
        mlx_key, target = remap(hf_key)
        if mlx_key is None:
            unmapped.append(hf_key)
            continue
        # Cast dtype (keep F32 for normalization scales).
        if should_keep_f32(mlx_key):
            tensor = tensor.astype(mx.float32)
        else:
            tensor = tensor.astype(target_dtype)
        if target == "llm":
            llm_state[mlx_key] = tensor
        elif target == "vit":
            vit_state[mlx_key] = tensor

    if unmapped:
        print(f"FAIL: {len(unmapped)} unmapped keys (first 5):", file=sys.stderr)
        for k in unmapped[:5]:
            print(f"  {k}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    llm_out = out_dir / "model.safetensors"
    print(f"Writing {llm_out} ({len(llm_state)} tensors) ...")
    mx.save_safetensors(str(llm_out), llm_state)

    if vit_state:
        vit_out = out_dir / "vit.safetensors"
        print(f"Writing {vit_out} ({len(vit_state)} tensors) ...")
        mx.save_safetensors(str(vit_out), vit_state)
    else:
        print(f"  no bundled ViT in {variant}; load Qwen2.5-VL-ViT/vit.safetensors separately")

    # Copy llm_config.json from source (if present), patching runtime overrides.
    src_cfg = src_dir / "llm_config.json"
    if src_cfg.exists():
        cfg = json.loads(src_cfg.read_text())
        # Per inference_lance.sh — runtime override that the safetensors actually reflects.
        cfg["tie_word_embeddings"] = False
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        print(f"  wrote config.json (tie_word_embeddings: false applied)")

    # Provenance report.
    report = {
        "source_dir": str(src_dir),
        "variant": variant,
        "dtype": dtype,
        "n_llm_tensors": len(llm_state),
        "n_vit_tensors": len(vit_state),
        "unmapped_keys": unmapped,
    }
    (out_dir / "conversion_report.json").write_text(json.dumps(report, indent=2))
    total = sum(t.size for t in llm_state.values()) + sum(t.size for t in vit_state.values())
    print(f"\n✓ Converted {total/1e9:.2f} B params to {dtype} at {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf-repo", help="HuggingFace repo_id (e.g. bytedance-research/Lance)")
    src.add_argument("--local-dir", type=Path,
                     help="Local dir containing model.safetensors + llm_config.json")
    ap.add_argument("--variant", choices=list(VARIANT_DIRS), default="lance_3b")
    ap.add_argument("--output", type=Path,
                    help="Destination dir (required unless --dry-run)")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--dry-run", action="store_true",
                    help="Header-only validation, no download, no file writes")
    args = ap.parse_args()

    if args.dry_run:
        if not args.hf_repo:
            print("ERROR: --dry-run requires --hf-repo (header fetch is HF-only)", file=sys.stderr)
            return 1
        filename = f"{VARIANT_DIRS[args.variant]}/model.safetensors"
        return dry_run(args.hf_repo, filename, args.variant)

    if args.output is None:
        print("ERROR: --output is required unless --dry-run", file=sys.stderr)
        return 1

    src_dir = resolve_source(args.hf_repo, args.local_dir, args.variant)
    return convert(src_dir, args.output, args.dtype, args.variant)


if __name__ == "__main__":
    sys.exit(main())
