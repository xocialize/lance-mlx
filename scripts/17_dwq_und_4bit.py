#!/usr/bin/env python3
"""Phase 5c — DWQ-calibrate a pre-quantized Lance LLM (UND tower).

Takes a Lance model that's already been pre-quantized via scripts/16_quantize.py
(typically `--bits 4 --skip-gen-tower`), then runs mlx-lm's DWQ harness to
distill the UND-tower QuantizedLinear scales+biases against the bf16 teacher.

The GEN tower stays at bf16, which means t2i / image_edit / t2v image quality
is unaffected. DWQ targets the UND tower's text-generation logits — recovers
x2t_image quality and (since the lm_head pathway uses the calibrated tower)
the text-side instruction following.

Why this is the path of least resistance:
  - mlx-lm's `quant.dwq.dwq_quantize` works out of the box on any model that
    is `model(tokens) → logits` callable
  - Our LanceModel returns hidden states (not logits) and needs position_ids
    + position_group; we wrap it in `LanceTextLogitsWrapper` to expose a
    standard LLM API for text-only forward
  - DWQ only unfreezes `scales` + `biases` of QuantizedLinear modules with
    `bits < 8`, so it leaves the bf16 GEN tower entirely alone

Usage:
    # Step 1: pre-quantize (separately, faster — skip if already done)
    HF_HUB_DISABLE_XET=1 uv run python scripts/16_quantize.py \\
        --lance-weights /Volumes/.../Lance-3B-bf16 \\
        --out-dir       /tmp/Lance-3B-4bit-und-prequant \\
        --bits 4 --group-size 64 --skip-gen-tower

    # Step 2: DWQ calibrate
    HF_HUB_DISABLE_XET=1 uv run python scripts/17_dwq_und_4bit.py \\
        --teacher-weights /Volumes/.../Lance-3B-bf16 \\
        --student-weights /tmp/Lance-3B-4bit-und-prequant \\
        --out-dir         /tmp/Lance-3B-4bit-und-DWQ \\
        --num-samples 256 --max-seq-length 512 --batch-size 2 \\
        --learning-rate 2e-5
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
import mlx.optimizers as optimizers
import numpy as np
from mlx_lm.quant.dwq import dwq_quantize
from mlx_vlm.models.qwen2_5_vl.config import TextConfig
from transformers import AutoProcessor

from lance_mlx.model import LanceModel


# A small mixed-domain text calibration corpus. Includes:
#   - Generic descriptive prose (probes broad language coverage)
#   - Lance-style chat-template prompts (matches deployment distribution)
#   - Image-description prose (probes the visual-grounding text used in x2t)
# Per Reza2kn's findings, ~17-256 well-chosen samples are sufficient for DWQ
# to recover most of the quality gap. We use 64 base lines, repeated/sliced
# to reach `--num-samples`.
CALIBRATION_CORPUS = [
    # Generic prose
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump.",
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
    "All happy families are alike; each unhappy family is unhappy in its own way. Everything was in confusion in the Oblonskys' house.",
    "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, I thought I would sail.",
    # Technical / reference
    "MLX is an array framework for machine learning research on Apple silicon, brought to you by Apple machine learning research.",
    "Apple Silicon unified memory means CPU and GPU share the same memory pool, so there's no need to copy data between them.",
    "Quantization reduces the precision of model weights, trading some accuracy for substantial reductions in memory and compute.",
    "A diffusion model learns to invert a gradual noising process by predicting the noise added at each timestep of a Markov chain.",
    "Transformers attend to all positions in a sequence using a learned similarity score between query and key projections.",
    # Lance-style instruction prompts (text portion of chat templates)
    "<|im_start|>user\nDescribe the image in detail.<|im_end|>\n<|im_start|>assistant\nThe image shows a domestic cat with orange fur",
    "<|im_start|>user\nWhat objects are visible in this scene?<|im_end|>\n<|im_start|>assistant\nI can see a wooden table with three apples",
    "<|im_start|>user\nGenerate an image of a sunset over mountains.<|im_end|>\n<|im_start|>assistant\nI'll create an image showing the warm",
    "<|im_start|>user\nEdit the image to add a red hat to the person.<|im_end|>\n<|im_start|>assistant\nI've added a vibrant red baseball cap",
    "<|im_start|>user\nWhat is happening in this video?<|im_end|>\n<|im_start|>assistant\nThe video depicts a person walking through a garden",
    # Image-description prose (probes visual-grounding text)
    "A photograph of a red double-decker bus driving past Big Ben on a sunny London morning, with tourists visible on the sidewalk.",
    "A close-up portrait of a young woman with auburn hair, smiling warmly while standing in front of a wooden bookshelf.",
    "An aerial view of the Grand Canyon at sunrise, with golden light illuminating the layered rock formations.",
    "A bowl of fresh strawberries and blueberries on a marble countertop, with a sprig of mint leaves as garnish.",
    "A vintage typewriter with a half-finished page, sitting on an oak desk beside a steaming cup of coffee.",
    # Conversational
    "The recipe calls for two cups of flour, one cup of sugar, three eggs, half a cup of butter, and a teaspoon of vanilla extract.",
    "The treaty was signed in 1648 by representatives of the Holy Roman Empire, the Spanish Empire, and various other European powers.",
    "When you press the button, the elevator will smoothly descend to the ground floor in approximately twelve seconds.",
    "He carefully unfolded the ancient map, its edges worn smooth by countless hands across the centuries since its making.",
    "She typed quickly, fingers dancing across the keyboard as she composed a thoughtful response to the difficult question.",
    # More technical
    "Attention scores are computed as the scaled dot product of queries and keys, then normalized via softmax to form weights.",
    "Flow matching is a generative modeling technique where the model learns a velocity field that transports noise to data.",
    "Mixture of experts (MoE) routes each token through a sparse selection of expert MLPs based on a learned gating function.",
    "Rotary position embeddings encode absolute position by rotating the query and key vectors in pairs of dimensions before attention.",
    "Group query attention shares key and value projections across multiple query heads, reducing KV cache memory by the share factor.",
    # Code-adjacent
    "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)  # classic recursive definition of the factorial function.",
    "Initialize a list comprehension to filter even numbers from a range: evens = [x for x in range(100) if x % 2 == 0]",
    "Class definitions in Python use the class keyword followed by the name and optional parentheses listing base classes.",
    "Async functions return coroutines that must be awaited or scheduled on an event loop to actually execute their body.",
    "TypeScript adds static type annotations to JavaScript, catching many errors at compile time rather than runtime.",
    # Lance-domain prompts continued
    "<|im_start|>user\nA red panda surfing on a sunny wave with a painted surfboard.<|im_end|>\n<|im_start|>assistant\nGenerating the scene",
    "<|im_start|>user\nA cat holding a colorful STOP sign in front of a brick wall.<|im_end|>\n<|im_start|>assistant\nCreating that image",
    "<|im_start|>user\nRemove the hat from the person in the image.<|im_end|>\n<|im_start|>assistant\nI'll edit out the hat carefully",
    "<|im_start|>user\nWhat color is the car in the image?<|im_end|>\n<|im_start|>assistant\nThe car in the image is bright red",
    "<|im_start|>user\nDescribe the architectural style of the building.<|im_end|>\n<|im_start|>assistant\nThe building exhibits Gothic Revival",
    # Long-form descriptive
    "The sun rose over the eastern mountains, casting long shadows across the valley where the small village was just beginning to wake.",
    "In the laboratory, rows of beakers and test tubes lined the polished countertops, each containing precisely measured samples for the experiment.",
    "The orchestra tuned their instruments as the conductor walked onto the stage, the audience hushed in anticipation of the symphony's opening notes.",
    "Through the dense fog, the lighthouse beam swept across the rocky shore, warning ships of the treacherous rocks that lurked just below the surface.",
    "She climbed the spiral staircase carefully, one hand on the worn wooden banister, the other holding a flickering candle to light her way.",
    # Mixed
    "Quantum computing leverages superposition and entanglement to perform certain calculations exponentially faster than classical computers.",
    "The chef arranged the ingredients in a precise mise en place, ensuring everything was within arm's reach before the dinner rush began.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy captured by chlorophyll molecules.",
    "The novel's protagonist embarks on a journey of self-discovery that takes him across three continents and spans nearly two decades.",
    "Crystal structures form when atoms arrange themselves into highly ordered repeating patterns that extend in three-dimensional space.",
    "The market analyst predicted a downturn based on declining consumer confidence indicators and tightening monetary policy from the central bank.",
    "Marine biologists study the complex ecosystems of coral reefs, which support a quarter of all ocean species despite covering less than one percent of the seafloor.",
    "The medieval cathedral's flying buttresses redirect the outward thrust of the high vaulted ceiling, allowing for taller walls and larger windows.",
    "Modern cryptography relies on mathematical problems that are easy to verify but computationally infeasible to solve without the secret key.",
    "The biotech startup raised forty million dollars in Series B funding to advance its CRISPR-based gene therapy through clinical trials.",
    # Tail-end Lance-style
    "<|im_start|>user\nGenerate a video of a butterfly landing on a flower.<|im_end|>\n<|im_start|>assistant\nI'll create a video showing the delicate moment",
    "<|im_start|>user\nIs there a person in the picture?<|im_end|>\n<|im_start|>assistant\nNo, the image contains no people — only landscape",
    "<|im_start|>user\nWhat time of day does this look like?<|im_end|>\n<|im_start|>assistant\nBased on the warm orange lighting and long shadows",
    "<|im_start|>user\nGenerate a portrait of a wise old wizard.<|im_end|>\n<|im_start|>assistant\nCreating a portrait of an elderly wizard with a long",
    "<|im_start|>user\nChange the background to a sunset beach.<|im_end|>\n<|im_start|>assistant\nI'll edit the background to show a warm tropical beach",
]


class LanceTextLogitsWrapper(nn.Module):
    """Adapts LanceModel to expose a standard `model(tokens) → logits` API
    that mlx-lm's DWQ harness can drive. Text-only forward path:
    `position_group = 0` for all tokens (UND tower exclusively)."""

    def __init__(self, lance_model: LanceModel):
        super().__init__()
        self.lance = lance_model
        # mlx-lm.quant.dwq uses `model.layers[0]` for optional grad checkpointing
        self.layers = lance_model.layers

    def __call__(self, x: mx.array) -> mx.array:
        B, T = x.shape
        # Text-only position_ids: same value across all 3 mRoPE channels
        pos = mx.arange(T, dtype=mx.int32)
        position_ids = mx.broadcast_to(pos[None, None, :], (3, B, T))
        # All TEXT — exercises UND tower only
        position_group = mx.zeros((T,), dtype=mx.int32)
        # Forward through Lance
        h = self.lance(
            input_ids=x,
            position_ids=position_ids,
            position_group=position_group,
        )
        # Apply lm_head to get logits over vocab
        return self.lance.lm_head(h)


def load_lance(weights_dir: Path) -> tuple[LanceModel, dict]:
    """Load a Lance model from a (possibly quantized) directory."""
    cfg = json.loads((weights_dir / "config.json").read_text())
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
    saved = mx.load(str(weights_dir / "model.safetensors"))
    n_lat_positions = saved["latent_pos_embed.pos_embed"].shape[0]
    model = LanceModel(text_cfg, num_latent_positions=n_lat_positions)

    # If this checkpoint has a `quantization` block, apply nn.quantize
    # before load_weights (mirrors src/lance_mlx/model/_loader.py)
    if "quantization" in cfg:
        q = cfg["quantization"]
        skip_gen = q.get("skip_gen_tower", False)
        skip_always = ("time_embedder.proj_in", "time_embedder.proj_out", "llm2vae")
        skip_gen_pat = ("_moe_gen",) if skip_gen else ()
        all_skip = skip_always + skip_gen_pat

        def class_predicate(path: str, module: nn.Module) -> bool:
            # Quantize only if a corresponding `.scales` tensor was saved
            # (the most robust check — matches mlx-vlm's approach)
            return f"{path}.scales" in saved

        nn.quantize(
            model, group_size=q["group_size"], bits=q["bits"],
            class_predicate=class_predicate,
        )

    model.load_weights(list(saved.items()))
    mx.eval(model.parameters())
    return model, cfg


def build_calibration(tokenizer, num_samples: int, max_seq_length: int) -> list[tuple[np.ndarray, int]]:
    """Tokenize calibration corpus, yielding (tokens, offset) tuples ready
    for mlx-lm's iterate_batches. offset=0 means all tokens contribute to loss.

    Notes:
      - mlx-lm's iterate_batches pads each batch to nearest multiple of 32, so
        we accept any non-trivial length (>=8 tokens) rather than filtering
        for >=32. Qwen's BPE is efficient: median corpus line ≈ 25 tokens,
        max ≈ 35, so a >=32 filter rejects ~95% of lines and infinite-loops.
      - Augmentation: once we exhaust the base corpus, we concatenate random
        pairs of lines to reach the target sample count without pure repeats.
      - Hard cap on iterations to prevent any future infinite-loop bug.
    """
    corpus = list(CALIBRATION_CORPUS)
    out = []
    rng = np.random.default_rng(0)
    max_iter = num_samples * 20  # safety cap
    iters = 0
    while len(out) < num_samples and iters < max_iter:
        iters += 1
        line = corpus[len(out) % len(corpus)]
        # Concatenate a second random line once we've used each base line once
        if len(out) >= len(corpus):
            line = line + " " + corpus[int(rng.integers(len(corpus)))]
        toks = tokenizer.encode(line, add_special_tokens=False)[:max_seq_length]
        if len(toks) < 8:
            continue
        out.append((np.array(toks, dtype=np.int32), 0))
    if len(out) < num_samples:
        raise RuntimeError(
            f"build_calibration: only got {len(out)}/{num_samples} samples "
            f"after {iters} iterations — corpus produces too-short tokenizations"
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-weights", type=Path, required=True,
                    help="bf16 Lance directory (used as DWQ teacher).")
    ap.add_argument("--student-weights", type=Path, required=True,
                    help="Pre-quantized Lance directory (from scripts/16_quantize.py).")
    ap.add_argument("--out-dir", type=Path, required=True)
    # Defaults aligned to mlx-lm's quant.dwq defaults (which are tuned for
    # standard LLM DWQ workloads). Our earlier 2e-5 + AdamW + 64 samples
    # overshot dramatically and made loss WORSE (0.174 → 0.953). Now using
    # 1e-6 + Adam(bias_correction=True) + 256 samples (a middle ground —
    # mlx-lm default 2048 would be ~30min wall-clock; 256 is ~5min).
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--num-valid", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"┏━━ Phase 5c — Lance UND-tower DWQ ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ teacher     : {args.teacher_weights}")
    print(f"┃ student     : {args.student_weights}")
    print(f"┃ output      : {args.out_dir}")
    print(f"┃ samples     : {args.num_samples} train + {args.num_valid} valid")
    print(f"┃ max_seq_len : {args.max_seq_length}")
    print(f"┃ batch_size  : {args.batch_size}")
    print(f"┃ learning_rate: {args.learning_rate}")
    print(f"┃ temperature : {args.temperature}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # --- 1. Load teacher + student ----------------------------------------
    print(f"\n=== Loading teacher (bf16) ===")
    t0 = time.perf_counter()
    teacher_lance, _ = load_lance(args.teacher_weights)
    teacher = LanceTextLogitsWrapper(teacher_lance)
    teacher.eval()
    teacher.freeze()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    print(f"\n=== Loading student (pre-quantized) ===")
    t0 = time.perf_counter()
    student_lance, student_cfg = load_lance(args.student_weights)
    if "quantization" not in student_cfg:
        print("✗ ERROR: student checkpoint has no 'quantization' block in config.json")
        print("  Run scripts/16_quantize.py first to produce a pre-quantized model.")
        return 1
    student = LanceTextLogitsWrapper(student_lance)
    # CRITICAL: freeze everything first. mlx-lm's DWQ assumes a frozen model
    # then selectively unfreezes QuantizedLinear scales+biases. Without this
    # freeze, the bf16 GEN tower (and embeddings, lm_head) are trainable by
    # default → DWQ trains ALL of them → corrupts the model. The previous
    # run had trainable_parameters=49% which should have been ~0.4%.
    student.freeze()
    print(f"  loaded + frozen in {time.perf_counter()-t0:.1f}s")
    q = student_cfg["quantization"]
    print(f"  student quant: bits={q['bits']}, group_size={q['group_size']}, "
          f"skip_gen_tower={q.get('skip_gen_tower', False)}")

    # --- 2. Tokenizer + calibration data ----------------------------------
    # Load via AutoProcessor (same path the t2i/t2v/etc pipelines use).
    # AutoTokenizer on the local Lance dir hangs because Lance's config.json
    # advertises model_type=qwen2_5_vl which AutoTokenizer tries to resolve
    # via the multimodal processor pipeline.
    print(f"\n=== Building calibration set ===")
    t0 = time.perf_counter()
    print(f"  loading processor from Qwen/Qwen2.5-VL-3B-Instruct (cached after first download)...", flush=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    tokenizer = processor.tokenizer
    print(f"  tokenizer loaded ({time.perf_counter()-t0:.1f}s)", flush=True)
    t1 = time.perf_counter()
    train_data = build_calibration(tokenizer, args.num_samples, args.max_seq_length)
    valid_data = build_calibration(tokenizer, args.num_valid, args.max_seq_length)
    print(f"  built {len(train_data)} train + {len(valid_data)} valid samples "
          f"in {time.perf_counter()-t1:.1f}s")
    print(f"  sample lengths: min={min(len(t) for t, _ in train_data)}, "
          f"max={max(len(t) for t, _ in train_data)}, "
          f"avg={int(np.mean([len(t) for t, _ in train_data]))}")

    # --- 3. Build target_fn (closure over teacher) ------------------------
    # mlx_lm.dwq's dwq_quantize calls `target_fn(batch, it, split)` and expects
    # either logits OR (logits, indices) where indices are top-K vocab IDs.
    # Computing full logits is fine for our scale.
    def target_fn(batch: mx.array, it: int, split: str) -> mx.array:
        # batch: (B, T) int tokens
        logits = teacher(batch)
        return mx.stop_gradient(logits)

    # --- 4. Optimizer (Adam over scales+biases only) ----------------------
    # mlx-lm's quant.dwq uses Adam with bias_correction=True (NOT AdamW).
    # Switching this was material: AdamW @ 2e-5 → catastrophic divergence;
    # Adam(bias_correction=True) @ 1e-6 is the published baseline.
    opt = optimizers.Adam(learning_rate=args.learning_rate, bias_correction=True)

    # --- 5. Run DWQ -------------------------------------------------------
    print(f"\n=== Running DWQ ({args.num_samples // args.batch_size} steps) ===")
    t0 = time.perf_counter()
    dwq_quantize(
        model=student,
        target_fn=target_fn,
        opt=opt,
        train_data=train_data,
        valid_data=valid_data,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        dtype=mx.bfloat16,
        temperature=args.temperature,
    )
    print(f"\n  DWQ completed in {time.perf_counter()-t0:.1f}s")
    print(f"  peak memory: {mx.get_peak_memory() / 1e9:.2f} GB")

    # --- 6. Save the DWQ'd student weights --------------------------------
    print(f"\n=== Writing DWQ'd weights ===")
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
    flatten("", dict(student_lance.parameters()), flat)
    mx.save_safetensors(str(args.out_dir / "model.safetensors"), flat)
    print(f"  wrote {len(flat)} tensors in {time.perf_counter()-t0:.1f}s")

    # --- 7. Copy + update config -----------------------------------------
    out_cfg = dict(student_cfg)
    out_cfg["quantization"]["dwq"] = True
    out_cfg["quantization"]["dwq_num_samples"] = args.num_samples
    out_cfg["quantization"]["dwq_learning_rate"] = args.learning_rate
    out_cfg["quantization"]["dwq_temperature"] = args.temperature
    (args.out_dir / "config.json").write_text(json.dumps(out_cfg, indent=2))

    report = {
        "teacher_dir": str(args.teacher_weights),
        "student_dir": str(args.student_weights),
        "bits": q["bits"],
        "group_size": q["group_size"],
        "skip_gen_tower": q.get("skip_gen_tower", False),
        "dwq_num_samples": args.num_samples,
        "dwq_num_valid": args.num_valid,
        "dwq_max_seq_length": args.max_seq_length,
        "dwq_batch_size": args.batch_size,
        "dwq_learning_rate": args.learning_rate,
        "dwq_temperature": args.temperature,
    }
    (args.out_dir / "dwq_report.json").write_text(json.dumps(report, indent=2))

    # Copy auxiliary files
    for fname in ["tokenizer.json", "vocab.json", "tokenizer_config.json",
                  "generation_config.json", "llm_config.json",
                  "vit.safetensors", "vae.safetensors"]:
        src_path = args.student_weights / fname
        if src_path.exists():
            shutil.copy(src_path, args.out_dir / fname)
            print(f"  copied {fname}")

    print(f"\n✓ DWQ complete. Output: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
