"""Shared LanceModel construction + weight-load helper, quantization-aware.

This is the one place where pipelines learn how to load Lance weights from a
directory — whether bf16 (vanilla) or 8/4-bit (quantized via `scripts/16_quantize.py`).

Quantization detection: if `config.json` contains a `"quantization"` block
(written by mlx-lm's `quantize_model`), we apply `nn.quantize` to the model
with matching params BEFORE `load_weights` — so the quantized module shapes
match the safetensors layout.

Skip list mirrors `scripts/16_quantize.py::SKIP_PATTERNS` so that the same
modules are skipped at quantize-time and load-time.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen2_5_vl.config import TextConfig

from lance_mlx.model import LanceModel


# Must match scripts/16_quantize.py::SKIP_PATTERNS_ALWAYS exactly.
SKIP_PATTERNS_ALWAYS = (
    "time_embedder.proj_in",
    "time_embedder.proj_out",
    "llm2vae",
)

# Mirror of scripts/16_quantize.py::SKIP_PATTERNS_GEN_TOWER.
SKIP_PATTERNS_GEN_TOWER = (
    "_moe_gen",
)


def _build_skip_predicate(skip_gen_tower: bool):
    skip = list(SKIP_PATTERNS_ALWAYS)
    if skip_gen_tower:
        skip += list(SKIP_PATTERNS_GEN_TOWER)
    skip_tuple = tuple(skip)
    def pred(path: str, module: nn.Module) -> bool:
        return not any(p in path for p in skip_tuple)
    return pred


def build_text_config(cfg: dict) -> TextConfig:
    """Build a TextConfig from a Lance config.json dict."""
    return TextConfig(
        model_type=cfg["model_type"],
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        intermediate_size=cfg["intermediate_size"],
        num_attention_heads=cfg["num_attention_heads"],
        rms_norm_eps=cfg["rms_norm_eps"],
        vocab_size=cfg["vocab_size"],
        num_key_value_heads=cfg.get("num_key_value_heads"),
        max_position_embeddings=cfg.get("max_position_embeddings", 128000),
        rope_theta=cfg.get("rope_theta", 1e6),
        rope_scaling=cfg.get("rope_scaling"),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
    )


def load_lance_model(
    lance_weights_dir: Path | str,
    *,
    defer_gen_tower: bool = False,
) -> LanceModel:
    """Load LanceModel from a directory, applying quantization if config says so.

    Handles three layouts:
      - bf16: config.json has no 'quantization' block; load weights as-is.
      - 8-bit / 4-bit (mlx-lm quantize_model output): config.json has
        'quantization' = {bits, group_size, mode}; apply nn.quantize first.

    Returns an eval'd LanceModel ready to run.

    `defer_gen_tower` (deferred-load mode): if True, eval ONLY the non-`_moe_gen`
    parameters (UND + shared + small GEN heads) and leave every `_moe_gen` array
    lazy/mmap-backed (~5.5 GB un-materialized). `mx.load` is lazy and
    `load_weights` only assigns references, so the GEN tower is not materialized
    until something evaluates it. This drops the load footprint from ~12 GB to
    ~6.8 GB. The caller must run a **UND-only prefill** (the full routed forward's
    `mx.where(gen_mask, *_moe_gen, *und)` would otherwise touch — and materialize —
    the deferred GEN weights), then free the UND tower, then `materialize_gen_tower()`
    before the GEN-only denoise loop. The model retains a reference to the mmap'd
    `saved` weights (`model._saved_ref`) until GEN is materialized; `model._gen_deferred`
    flags the state. Probe-verified on real Lance-3B bf16: load active 6.82 GB.
    """
    lance_weights_dir = Path(lance_weights_dir)
    cfg = json.loads((lance_weights_dir / "config.json").read_text())
    text_cfg = build_text_config(cfg)

    saved = mx.load(str(lance_weights_dir / "model.safetensors"))
    num_latent_positions = saved["latent_pos_embed.pos_embed"].shape[0]
    model = LanceModel(text_cfg, num_latent_positions=num_latent_positions)

    quant = cfg.get("quantization")
    if quant is not None and "bits" in quant and "group_size" in quant:
        # Apply same quantization the saved weights were created with.
        # **Weight-file-aware predicate** (mirrors mlx_vlm/utils.py:349):
        # only quantize modules whose `.scales` key actually exists in the
        # saved weights. This avoids the static-skip-pattern bug where the
        # load-time predicate could disagree with the save-time predicate
        # (e.g. when a module's dimensions are/aren't divisible by group_size,
        # or when the LanceMoTAttention's GEN-side projections are/aren't
        # included). Disagreement silently corrupts modules — visible as
        # vertical-stripe artifacts in t2i.
        def class_predicate(path: str, m: nn.Module) -> bool:
            if not hasattr(m, "to_quantized"):
                return False
            if hasattr(m, "weight") and m.weight.size % quant["group_size"] != 0:
                return False
            return f"{path}.scales" in saved
        nn.quantize(
            model,
            group_size=quant["group_size"],
            bits=quant["bits"],
            mode=quant.get("mode", "affine"),
            class_predicate=class_predicate,
        )

    model.load_weights(list(saved.items()))

    if defer_gen_tower:
        # Eval only the non-`_moe_gen` params; leave the GEN tower lazy/mmap-backed.
        from mlx.utils import tree_flatten
        keep = [v for k, v in tree_flatten(model.parameters()) if "_moe_gen" not in k]
        mx.eval(keep)
        # Retain ONLY the deferred GEN entries so the mmap stays alive until they
        # are materialized. Pinning the whole `saved` dict would ALSO hold the
        # already-materialized UND weights, blocking free_und_tower (it could not
        # release UND before materialize_gen_tower → a transient both-towers blip).
        model._saved_ref = {k: v for k, v in saved.items() if "_moe_gen" in k}
        model._gen_deferred = True
    else:
        mx.eval(model.parameters())
        model._gen_deferred = False
    return model
