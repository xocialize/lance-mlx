# Phase 1b — `scripts/02_convert.py` design

**Status:** Design only. Implementation deferred to next session per the Option-A scope. No 24 GB download required for *writing* this doc — the design is fully derivable from `notes/phase1a_keys.md` and the scaffolds.

**Source of truth for key naming:**
- HF side: `notes/phase1a_lance_3b_keys_full.txt` (gitignored; regenerate with `scripts/01_inspect_keys.py --remote-repo bytedance-research/Lance --remote-file Lance_3B/model.safetensors`).
- MLX side: `src/lance_mlx/model/*.py` attribute names.

---

## What the converter does

Read an HF safetensors checkpoint, translate every tensor key from the HF naming convention to the MLX-side naming convention (which mostly mirrors HF but with a few intentional simplifications and the unavoidable `_moe_gen` split), optionally cast dtype, write out a single-file MLX-format safetensors.

**Inputs:**
- `--hf-repo bytedance-research/Lance` + `--hf-subdir Lance_3B` (or `Lance_3B_Video`), OR
- `--local-dir ~/models/Lance/Lance_3B` (pre-downloaded weights — skip HF fetch).
- `--output ~/models/mlx/Lance-3B-bf16` (target directory).
- `--dtype bf16|fp16|fp32` (default: bf16).
- `--variant lance_3b|lance_3b_video` (auto-detected from subdir; can override).

**Outputs:**
- `<output>/model.safetensors` — single-file MLX-format with translated keys.
- `<output>/config.json` — copy of `llm_config.json` from HF with `tie_word_embeddings: false` (the runtime override) and any other `inference_lance.sh` flags relevant at inference baked in.
- `<output>/conversion_report.json` — provenance: source repo + commit hash, key-map applied, numel sums (input/output), dtype histogram, list of any unmapped keys for follow-up.

---

## Key-map specification

### Naming convention on the MLX side

Choices I'm making deliberately to keep the converter mostly 1:1:

| Aspect | HF side | MLX side | Rationale |
|---|---|---|---|
| Top-level LLM wrap | `language_model.model.*` | `*` (strip both) | Our `LanceModel` *is* the LLM root; no nested `LanguageModel`. |
| Lm head | `language_model.lm_head.weight` | `lm_head.weight` | Same. |
| Per-expert flag | bare = UND, `_moe_gen` = GEN | suffix preserved at layer level | Simpler converter; small ergonomic cost when reading MLX state. |
| Attention biases on Q/K/V | present, distinct tensors | present, distinct tensors | 1:1. |
| Flow head | `llm2vae.{w,b}` (root) | `llm2vae.{w,b}` (root attribute on LanceModel) | 1:1 — see D5 docstring. |
| VAE input | `vae2llm.{w,b}` (root) | `vae_in_proj.vae2llm.{w,b}` (nested in LanceModel) | Module-level grouping; converter adds the `vae_in_proj.` prefix. |
| Latent pos embed | `latent_pos_embed.pos_embed` (root) | `latent_pos_embed.pos_embed` (root attribute) | 1:1. |
| Timestep embedder | `time_embedder.mlp.{0,2}.{w,b}` | `time_embedder.proj_{in,out}.{w,b}` | Mapped per D2 docstring. |
| Bundled ViT (Video) | `vit_model.*` | `vit_model.*` (pass through) | Loaded by mlx-vlm's Qwen2.5-VL ViT path; preserve the prefix. |

### Rules in order (apply first matching rule)

```python
RULES: list[tuple[Pattern, str]] = [
    # 1. Embeddings + LM head — strip `language_model.{model.}?` prefix
    (r"^language_model\.model\.embed_tokens\.weight$",       r"embed_tokens.weight"),
    (r"^language_model\.lm_head\.weight$",                   r"lm_head.weight"),

    # 2. Final norms — strip prefix; preserve `_moe_gen` suffix
    (r"^language_model\.model\.norm\.weight$",               r"norm.weight"),
    (r"^language_model\.model\.norm_moe_gen\.weight$",       r"norm_moe_gen.weight"),

    # 3. Layer body — strip prefix; keep layer index + everything after self_attn/mlp/layernorm
    #    Examples:
    #      language_model.model.layers.5.self_attn.q_proj.weight   → layers.5.self_attn.q_proj.weight
    #      language_model.model.layers.5.self_attn.q_proj_moe_gen.weight
    #                                                                → layers.5.self_attn.q_proj_moe_gen.weight
    #      language_model.model.layers.5.mlp.gate_proj.weight      → layers.5.mlp.gate_proj.weight
    #      language_model.model.layers.5.mlp_moe_gen.gate_proj.weight
    #                                                                → layers.5.mlp_moe_gen.gate_proj.weight
    #      language_model.model.layers.5.input_layernorm.weight    → layers.5.input_layernorm.weight
    #      language_model.model.layers.5.input_layernorm_moe_gen.weight
    #                                                                → layers.5.input_layernorm_moe_gen.weight
    (r"^language_model\.model\.layers\.(\d+)\.(.+)$",        r"layers.\1.\2"),

    # 4. Lance-specific root-level tensors
    (r"^llm2vae\.(weight|bias)$",                            r"llm2vae.\1"),
    (r"^vae2llm\.(weight|bias)$",                            r"vae_in_proj.vae2llm.\1"),
    (r"^latent_pos_embed\.pos_embed$",                       r"latent_pos_embed.pos_embed"),

    # 5. TimestepEmbedder: mlp.0/.2 → proj_in/proj_out
    (r"^time_embedder\.mlp\.0\.(weight|bias)$",              r"time_embedder.proj_in.\1"),
    (r"^time_embedder\.mlp\.2\.(weight|bias)$",              r"time_embedder.proj_out.\1"),

    # 6. Bundled ViT (Lance_3B_Video only) — pass through unchanged.
    (r"^vit_model\..+$",                                     r"\g<0>"),
]
```

Anything that doesn't match an explicit rule is an UNMAPPED warning → log + skip (do NOT silently include — that's how converter bugs hide).

### Expected key counts after mapping

| Component | Lance_3B count | Lance_3B_Video count | Where it lives in MLX |
|---|---:|---:|---|
| `embed_tokens.weight` | 1 | 1 | `LanceModel.embed_tokens` |
| `lm_head.weight` | 1 | 1 | `LanceModel.lm_head` |
| `layers.<N>.self_attn.{q,k,v,o}_proj.{weight,bias}` | 252 | 252 | `LanceModel.layers[N].self_attn` |
| `layers.<N>.self_attn.{q,k,v,o}_proj_moe_gen.*` | 252 | 252 | same (LanceMoTAttention sibling) |
| `layers.<N>.self_attn.{q,k}_norm.weight` | 72 | 72 | same |
| `layers.<N>.self_attn.{q,k}_norm_moe_gen.weight` | 72 | 72 | same |
| `layers.<N>.mlp.{gate,up,down}_proj.weight` | 108 | 108 | `LanceModel.layers[N].mlp` |
| `layers.<N>.mlp_moe_gen.{gate,up,down}_proj.weight` | 108 | 108 | sibling `mlp_moe_gen` |
| `layers.<N>.{input,post_attention}_layernorm{,_moe_gen}.weight` | 144 | 144 | inside `LanceMoTDecoderLayer` |
| `norm.weight` + `norm_moe_gen.weight` | 2 | 2 | `LanceModel.norm` + `.norm_moe_gen` |
| `llm2vae.{weight,bias}` | 2 | 2 | `LanceModel.llm2vae` (FlowHead) |
| `vae_in_proj.vae2llm.{weight,bias}` | 2 | 2 | `LanceModel.vae_in_proj` |
| `latent_pos_embed.pos_embed` | 1 | 1 | `LanceModel.latent_pos_embed` |
| `time_embedder.{proj_in,proj_out}.{weight,bias}` | 4 | 4 | `LanceModel.time_embedder` |
| `vit_model.*` | 0 | 390 | `LanceModel.vit_model` (or sibling — see open question below) |
| **Total** | **1021** | **1411** | |

Sums match Phase 1a inspection. Numel validation gate: `sum(numel for k,v in mlx_state.items()) == 6.19 B` for Lance_3B; `7.11 B` for Lance_3B_Video.

---

## Dtype cast strategy

- **Default:** cast everything to bf16. M5 Max has hardware bf16 matmul; saves ~50% storage vs F32; <0.01% quality cost on any model that was trained in bf16 mixed-precision (which Lance was — `inference_lance.sh` uses `--mixed_precision bf16`).
- **Override:** `--dtype fp16` (downgrades to half-precision; bigger numerical drift, sometimes faster on older Apple Silicon — not relevant on M5), `--dtype fp32` (lossless but 2x disk; useful for parity-debugging).
- **Exception — keep F32 for `*_layernorm*.weight`:** RMSNorm scales are scalar-per-channel and benefit from F32 numerical stability. Convention from FLUX/SDXL ports. Adds <1 MB to the file. Implement as a per-key whitelist.
- **Special — `latent_pos_embed.pos_embed`** is currently F32 in upstream (8.4 M × 4 B = 33 MB). bf16 (17 MB) is fine here; positional embeddings are inherently fuzzy.

---

## Implementation skeleton

```python
#!/usr/bin/env python3
"""Phase 1b — convert HF Lance safetensors → MLX format."""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
from safetensors import safe_open
from safetensors.numpy import save_file as numpy_save_file

from huggingface_hub import snapshot_download

# Rule list verbatim from this design doc (RULES table above).
RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(p), r) for p, r in [
        (r"^language_model\.model\.embed_tokens\.weight$", r"embed_tokens.weight"),
        (r"^language_model\.lm_head\.weight$",             r"lm_head.weight"),
        (r"^language_model\.model\.norm\.weight$",         r"norm.weight"),
        (r"^language_model\.model\.norm_moe_gen\.weight$", r"norm_moe_gen.weight"),
        (r"^language_model\.model\.layers\.(\d+)\.(.+)$",  r"layers.\1.\2"),
        (r"^llm2vae\.(weight|bias)$",                       r"llm2vae.\1"),
        (r"^vae2llm\.(weight|bias)$",                       r"vae_in_proj.vae2llm.\1"),
        (r"^latent_pos_embed\.pos_embed$",                  r"latent_pos_embed.pos_embed"),
        (r"^time_embedder\.mlp\.0\.(weight|bias)$",         r"time_embedder.proj_in.\1"),
        (r"^time_embedder\.mlp\.2\.(weight|bias)$",         r"time_embedder.proj_out.\1"),
        (r"^vit_model\..+$",                                r"\g<0>"),
    ]
]

# Keys that stay F32 even when --dtype bf16
KEEP_F32_PATTERNS = [
    re.compile(r".*_layernorm.*\.weight$"),
    re.compile(r"^norm.*\.weight$"),
    re.compile(r".*\.q_norm.*\.weight$"),
    re.compile(r".*\.k_norm.*\.weight$"),
]


def remap(hf_key: str) -> str | None:
    """Apply the first matching rule. Returns None if unmapped (caller logs)."""
    for pat, repl in RULES:
        m = pat.match(hf_key)
        if m:
            return pat.sub(repl, hf_key)
    return None


def cast_dtype(value: Any, key: str, target: str) -> Any:
    if any(p.match(key) for p in KEEP_F32_PATTERNS):
        return value.astype(np.float32)
    return value.astype({"bf16": ml_dtypes.bfloat16, "fp16": np.float16, "fp32": np.float32}[target])


def main() -> int:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--hf-repo")
    grp.add_argument("--local-dir", type=Path)
    ap.add_argument("--hf-subdir", default="Lance_3B",
                    choices=["Lance_3B", "Lance_3B_Video"])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    src_dir = args.local_dir or Path(snapshot_download(
        args.hf_repo, allow_patterns=[f"{args.hf_subdir}/*"]
    )) / args.hf_subdir

    in_st = src_dir / "model.safetensors"
    out_st = args.output / "model.safetensors"
    args.output.mkdir(parents=True, exist_ok=True)

    mapped: dict[str, Any] = {}
    unmapped: list[str] = []
    total_in_numel = 0
    total_out_numel = 0

    with safe_open(str(in_st), framework="numpy") as f:
        for hf_key in f.keys():
            arr = f.get_tensor(hf_key)
            total_in_numel += arr.size
            mlx_key = remap(hf_key)
            if mlx_key is None:
                unmapped.append(hf_key)
                continue
            mapped[mlx_key] = cast_dtype(arr, mlx_key, args.dtype)
            total_out_numel += mapped[mlx_key].size

    if unmapped:
        print(f"WARN: {len(unmapped)} unmapped keys (first 5):", file=sys.stderr)
        for k in unmapped[:5]:
            print(f"  {k}", file=sys.stderr)
        # Hard-fail unless --allow-unmapped
        return 2

    assert total_in_numel == total_out_numel, (
        f"numel mismatch: {total_in_numel:,} → {total_out_numel:,}"
    )

    numpy_save_file(mapped, str(out_st))
    # Also copy + munge config.json
    src_cfg = src_dir / "llm_config.json"
    if src_cfg.exists():
        cfg = json.loads(src_cfg.read_text())
        cfg["tie_word_embeddings"] = False  # runtime override from inference_lance.sh
        (args.output / "config.json").write_text(json.dumps(cfg, indent=2))

    report = {
        "source": str(in_st),
        "variant": args.hf_subdir,
        "dtype": args.dtype,
        "tensors_in": total_in_numel,
        "tensors_out": total_out_numel,
        "n_keys": len(mapped),
        "unmapped_keys": unmapped,
    }
    (args.output / "conversion_report.json").write_text(json.dumps(report, indent=2))
    print(f"OK: wrote {out_st} ({total_out_numel/1e9:.2f} B params, {len(mapped)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

(The existing `scripts/02_convert.py` is a 4.4 KB stub; replace wholesale with this skeleton in Phase 1b.)

---

## Open questions for Phase 1b

1. **Where does the bundled ViT go in the MLX tree?** `LanceModel.vit_model = Qwen2_5_VLVisionTower(...)` from mlx-vlm? Or a sibling `vit_model` not on the LanceModel? Default proposal: sibling attribute on the top-level package's `Pipeline` class (the orchestrator) — `LanceModel` is the LLM-only backbone. Confirm by reading mlx-vlm's Qwen2.5-VL composition pattern.

2. **Is the bundled `vit_model.*` in Lance_3B_Video byte-identical to the separate `Qwen2.5-VL-ViT/vit.safetensors`?** Both ship in the same HF repo. Quick test: header-only inspect both and shape-diff. If identical, prefer loading from the bundle (one less file to fetch). If different, the bundle is fine-tuned for video and should override.

3. **Should the converter validate against the scaffold's expected attribute names?** Could parse `src/lance_mlx/model/*.py` AST for `self.<name> = ...` patterns and warn if the converter would produce a key that doesn't have a destination. Belt-and-suspenders; adds complexity. Default: defer to a separate `scripts/03_validate_conversion.py` later.

4. **Lance_3B vs Lance_3B_Video — both, or one at a time?** Phase 1b ships the converter; the user can run it twice with `--hf-subdir`. No need to handle both in one invocation.

5. **`llm_config.json` munging — what other runtime overrides from `inference_lance.sh` need to be baked in?** Per `notes/phase0_config_factory.md`: `tie_word_embeddings: false`, `vae_model_type: wan`, `latent_patch_size: [1,1,1]`, `vit_type: qwen_2_5_vl_original`, possibly more. Compile the full list from the shell script + decide which apply at inference vs training only.

---

## Validation gate for Phase 1b

The converter is "done" when:

1. ✅ `uv run python scripts/02_convert.py --hf-repo bytedance-research/Lance --hf-subdir Lance_3B --output ~/models/mlx/Lance-3B-bf16 --dtype bf16` succeeds with 0 unmapped keys.
2. ✅ `conversion_report.json` shows `tensors_in == tensors_out` and `n_keys == 1021` for Lance_3B.
3. ✅ Same for `--hf-subdir Lance_3B_Video` → `n_keys == 1411`.
4. ✅ `mx.load("~/models/mlx/Lance-3B-bf16/model.safetensors")` loads without errors.
5. ⏸ DEFERRED to Phase 1d: instantiate `LanceModel(config)`, load the converted state, run a random-init forward pass on dummy input, confirm output shapes. Requires LanceMoTLayer to be implemented (currently raises NotImplementedError).

Once 1–4 pass, Phase 1b is complete and we move to Phase 2 (understanding pipeline using x2t_image fixtures).
