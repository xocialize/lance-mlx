#!/usr/bin/env python3
"""Phase 5c-3d — apply AWQ scales + INT4 quantize Lance.

Pipeline:
  1. Load Lance bf16
  2. Load activation stats (from phase5c-3c calibrate_awq.py)
  3. For each layer × fusion group:
       - run awq_search_scale to find per-input-channel s
       - apply: norm.weight /= s, consumer.weight *= s.reshape(1, -1)
       (all still bf16; no quantization yet)
  4. Run nn.quantize on the modified model (skipping only the always-bf16
     small modules: time_embedder, llm2vae). Both UND and GEN towers
     get quantized — that's the whole point of AWQ.
  5. Save Lance-3B-AWQ-INT4/

Validation in 5c-3e using the same 4-prompt sweep as 5c-2.

Usage:
    .venv/bin/python scripts/quant/apply_awq_quantize.py [--bits 4] [--group-size 128]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import quantize_model
from mlx_vlm.models.qwen2_5_vl.config import TextConfig

from lance_mlx.model import LanceModel
from lance_mlx.quant import (
    awq_search_scale, apply_scale_to_norm_and_consumers,
    FUSION_GROUPS, QUANT_SUFFIXES, load_act_stats,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT.parent / "lance-mlx-models" / "Lance-3B-bf16"
DEFAULT_STATS = REPO_ROOT / "notes" / "phase5n_diagnostics" / "phase5c3_awq_port" / "act_stats"
DEFAULT_OUT = REPO_ROOT.parent / "lance-mlx-models" / "Lance-3B-AWQ-INT4"


# Always-skip small / numerics-sensitive modules (from scripts/16_quantize.py)
SKIP_PATTERNS_ALWAYS = (
    "time_embedder.proj_in",
    "time_embedder.proj_out",
    "llm2vae",
)


def make_quant_predicate():
    """Skip only the always-bf16 small modules. Quantize EVERYTHING else
    including both UND and GEN towers — AWQ scale fusion should have made
    GEN-tower quantization viable."""
    def pred(path: str, module: nn.Module) -> bool:
        return not any(p in path for p in SKIP_PATTERNS_ALWAYS)
    return pred


def navigate(root, dotted: str):
    """Walk model.<...> via attr / list-index."""
    o = root
    for p in dotted.split("."):
        o = o[int(p)] if p.isdigit() else getattr(o, p)
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-grid", type=int, default=20,
                    help="Alpha grid resolution (default 20 = 21 alphas tested)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"┏━━ Phase 5c-3d — Lance AWQ + INT4 quantization ━━━━━━━━━━━━━━━")
    print(f"┃ src        : {args.src}")
    print(f"┃ stats      : {args.stats}")
    print(f"┃ out        : {args.out}")
    print(f"┃ bits       : {args.bits}")
    print(f"┃ group_size : {args.group_size}")
    print(f"┃ n_grid     : {args.n_grid}  ({args.n_grid + 1} alphas tested per group)")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ─── 1. Load Lance bf16 ─────────────────────────────────────────────────
    print(f"\n=== Loading Lance bf16 ===")
    t0 = time.perf_counter()
    cfg = json.loads((args.src / "config.json").read_text())
    text_cfg = TextConfig(
        model_type=cfg["model_type"], hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"], intermediate_size=cfg["intermediate_size"],
        num_attention_heads=cfg["num_attention_heads"], rms_norm_eps=cfg["rms_norm_eps"],
        vocab_size=cfg["vocab_size"], num_key_value_heads=cfg.get("num_key_value_heads"),
        max_position_embeddings=cfg.get("max_position_embeddings", 128000),
        rope_theta=cfg.get("rope_theta", 1e6),
        rope_scaling=cfg.get("rope_scaling"),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
    )
    saved = mx.load(str(args.src / "model.safetensors"))
    n_lat = saved["latent_pos_embed.pos_embed"].shape[0]
    model = LanceModel(text_cfg, num_latent_positions=n_lat)
    model.load_weights(list(saved.items()))
    mx.eval(model.parameters())
    bf16_bytes = sum(int(v.nbytes) for v in saved.values())
    n_layers = cfg["num_hidden_layers"]
    del saved
    print(f"  loaded {n_layers} layers in {time.perf_counter()-t0:.1f}s; "
          f"bf16 footprint {bf16_bytes / 1e9:.2f} GB")

    # ─── 2. Load activation stats ──────────────────────────────────────────
    print(f"\n=== Loading activation stats ===")
    t0 = time.perf_counter()
    stats = load_act_stats(args.stats)
    print(f"  loaded {len(stats)} stat entries in {time.perf_counter()-t0:.1f}s")

    # ─── 3. AWQ search + scale fusion per layer × fusion-group ─────────────
    print(f"\n=== AWQ scale-fusion ({n_layers} layers × {len(FUSION_GROUPS)} groups) ===")
    t0 = time.perf_counter()
    awq_applied = 0
    awq_skipped = 0
    awq_meta: dict[str, dict] = {}

    for li in range(n_layers):
        layer_prefix = f"layers.{li}"
        for norm_sub, consumer_subs in FUSION_GROUPS.items():
            norm_path = f"{layer_prefix}.{norm_sub}"
            try:
                norm_mod = navigate(model, norm_path)
            except AttributeError:
                # e.g. layer N might not have q_norm_moe_gen if structure varies
                continue

            consumer_mods = []
            consumer_paths = []
            for sub in consumer_subs:
                cpath = f"{layer_prefix}.{sub}"
                try:
                    cmod = navigate(model, cpath)
                    consumer_mods.append(cmod)
                    consumer_paths.append(cpath)
                except AttributeError:
                    pass
            if not consumer_mods:
                continue

            # Pull act_means (in calibration namespace, prefix is the same as
            # navigate path since LanceModel's named_modules() returns
            # 'layers.N.self_attn.q_proj' for the swapped Linears).
            act_means = [
                stats[cp].act_mean if cp in stats else None
                for cp in consumer_paths
            ]

            consumer_weights = [m.weight for m in consumer_mods]
            result = awq_search_scale(
                consumer_weights, act_means,
                n_bit=args.bits, group_size=args.group_size, n_grid=args.n_grid,
            )
            if result is None:
                awq_skipped += 1
                continue

            new_norm, new_consumers = apply_scale_to_norm_and_consumers(
                norm_mod.weight, consumer_weights, result.s,
            )
            norm_mod.weight = new_norm
            for cmod, nw in zip(consumer_mods, new_consumers):
                cmod.weight = nw
            awq_applied += 1
            awq_meta[norm_path] = {
                "best_alpha": result.best_alpha,
                "best_err": result.best_err,
                "consumers": consumer_paths,
            }

        if (li + 1) % 6 == 0:
            print(f"  layer {li+1:>2d}/{n_layers}:  applied={awq_applied}  skipped={awq_skipped}")

    print(f"  AWQ scale-fusion done in {time.perf_counter()-t0:.1f}s "
          f"(applied={awq_applied}, skipped={awq_skipped})")
    mx.eval(model.parameters())

    # ─── 4. Quantize the modified model ────────────────────────────────────
    print(f"\n=== Quantizing modified model (nn.quantize, both towers) ===")
    t0 = time.perf_counter()
    quant_config = dict(cfg)
    quantized_model, quantized_config = quantize_model(
        model=model,
        config=quant_config,
        group_size=args.group_size,
        bits=args.bits,
        mode="affine",
        quant_predicate=make_quant_predicate(),
    )
    # Annotate the config so loader code knows this is AWQ-calibrated
    quantized_config["quantization"]["awq"] = True
    quantized_config["quantization"]["awq_calibration"] = {
        "n_grid": args.n_grid,
        "calibration_stats_dir": str(args.stats.name),
        "fusion_groups_applied": awq_applied,
        "fusion_groups_skipped": awq_skipped,
    }
    mx.eval(quantized_model.parameters())
    print(f"  quantize done in {time.perf_counter()-t0:.1f}s")

    # ─── 5. Save ───────────────────────────────────────────────────────────
    print(f"\n=== Writing quantized model ===")
    t0 = time.perf_counter()

    def flatten(prefix, tree, out):
        if isinstance(tree, mx.array):
            out[prefix] = tree
        elif isinstance(tree, dict):
            for k, v in tree.items():
                flatten(f"{prefix}.{k}" if prefix else k, v, out)
        elif isinstance(tree, list):
            for i, v in enumerate(tree):
                flatten(f"{prefix}.{i}" if prefix else str(i), v, out)

    flat: dict[str, mx.array] = {}
    flatten("", dict(quantized_model.parameters()), flat)
    mx.save_safetensors(str(args.out / "model.safetensors"), flat)
    quant_bytes = sum(int(v.nbytes) for v in flat.values())
    print(f"  wrote {len(flat)} tensors in {time.perf_counter()-t0:.1f}s; "
          f"footprint {quant_bytes / 1e9:.2f} GB  ({quant_bytes/bf16_bytes:.1%} of bf16)")

    (args.out / "config.json").write_text(json.dumps(quantized_config, indent=2))
    report = {
        "source_dir": str(args.src),
        "stats_dir": str(args.stats),
        "bits": args.bits,
        "group_size": args.group_size,
        "mode": "affine+awq",
        "n_grid": args.n_grid,
        "bf16_bytes": bf16_bytes,
        "quantized_bytes": quant_bytes,
        "compression_ratio": quant_bytes / bf16_bytes,
        "fusion_groups_applied": awq_applied,
        "fusion_groups_skipped": awq_skipped,
        "skip_patterns_always": list(SKIP_PATTERNS_ALWAYS),
        "awq_per_group": awq_meta,
    }
    (args.out / "quantization_report.json").write_text(json.dumps(report, indent=2))
    print(f"  wrote config.json + quantization_report.json")

    print(f"\n=== Copying auxiliary files ===")
    for fname in ["tokenizer.json", "vocab.json", "tokenizer_config.json",
                  "generation_config.json", "llm_config.json",
                  "vit.safetensors", "vae.safetensors"]:
        src_path = args.src / fname
        if src_path.exists():
            shutil.copy(src_path, args.out / fname)
            print(f"  copied {fname}")

    print(f"\n✓ AWQ-INT4 quantization complete. Output: {args.out}")

    # Alpha distribution summary
    alphas = [m["best_alpha"] for m in awq_meta.values()]
    if alphas:
        import statistics
        print(f"\nBest-alpha distribution across {len(alphas)} fusion groups:")
        print(f"  min={min(alphas):.2f}  max={max(alphas):.2f}  "
              f"mean={statistics.mean(alphas):.3f}  median={statistics.median(alphas):.3f}")
        # Histogram bucket
        buckets = {0.0: 0, 0.2: 0, 0.4: 0, 0.6: 0, 0.8: 0, 1.0: 0}
        for a in alphas:
            for b in sorted(buckets.keys()):
                if a <= b:
                    buckets[b] += 1
                    break
        print(f"  histogram:")
        for b, c in sorted(buckets.items()):
            print(f"    α ≤ {b:.1f}:  {c:>3d}  {'█' * c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
