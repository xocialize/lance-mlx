"""Phase 5c-3c — calibration-stats collection for AWQ.

Drop-in replacement for nn.Linear that accumulates per-input-channel
sum(|x|) across forward calls. Used to feed `act_mean = sum_abs /
n_tokens` into `awq_search_scale`.

Design choice — subclass + swap, not monkey-patch:

  MLX has no built-in forward-hook mechanism (unlike PyTorch's
  register_forward_hook). The cleanest workaround is to subclass
  nn.Linear, override __call__ to record stats then delegate to
  super(), and walk the loaded model swapping every target Linear
  with the subclass. Weights are copied by reference (not duplicated),
  so memory cost is just one ActStats object per target ≈ 4 KB each,
  totaling ~2 MB for Lance's 504 quant-target Linears.

Usage:

    from lance_mlx.pipeline.t2i import TextToImagePipeline
    from lance_mlx.quant.calibrate import (
        install_act_stats, save_act_stats,
    )

    pipe = TextToImagePipeline.from_pretrained(...)
    stats = install_act_stats(pipe.lance_model)
    for prompt in CALIBRATION_PROMPTS:
        pipe.generate(prompt, ...)
    save_act_stats(stats, out_dir)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from lance_mlx.quant.awq import QUANT_SUFFIXES


# ============================================================================
# Stats accumulator
# ============================================================================


@dataclass
class ActStats:
    """Per-input-channel running stats. Matches Reza2kn/lance-quant's
    save format: {sum_abs: array, n_tokens: int, n_calls: int}."""
    sum_abs: mx.array | None = None
    n_tokens: int = 0
    n_calls: int = 0

    def update(self, x: mx.array) -> None:
        """Accumulate per-channel sum of |x|. x is (..., in_features)."""
        # Flatten all leading dims → (N, in_features).
        flat = x.reshape(-1, x.shape[-1])
        n = flat.shape[0]
        if n == 0:
            return
        # fp32 accumulator — sum_abs grows linearly with n_tokens; bf16 would
        # saturate around n_tokens ~ 2^16 for typical activation magnitudes.
        abs_sum = mx.sum(mx.abs(flat.astype(mx.float32)), axis=0)
        if self.sum_abs is None:
            self.sum_abs = abs_sum
        else:
            self.sum_abs = self.sum_abs + abs_sum
        self.n_tokens += int(n)
        self.n_calls += 1
        # Force materialization — otherwise the lazy graph grows unbounded
        # across the calibration run.
        mx.eval(self.sum_abs)

    def reset(self) -> None:
        self.sum_abs = None
        self.n_tokens = 0
        self.n_calls = 0

    @property
    def act_mean(self) -> mx.array | None:
        """mean(|activations|) per channel — what awq_search_scale wants."""
        if self.sum_abs is None or self.n_tokens == 0:
            return None
        return self.sum_abs / self.n_tokens


# ============================================================================
# Linear subclass
# ============================================================================


class ActStatsLinear(nn.Linear):
    """nn.Linear that records per-channel input activation stats.

    Drop-in replacement: same parameters (weight, optional bias),
    identical forward output. Adds .stats attribute for accumulation.
    """
    @classmethod
    def from_linear(cls, src: nn.Linear) -> "ActStatsLinear":
        """Build from an existing nn.Linear, sharing weights by reference."""
        out_features, in_features = src.weight.shape
        has_bias = "bias" in src
        inst = cls(in_features, out_features, bias=has_bias)
        inst.weight = src.weight
        if has_bias:
            inst.bias = src.bias
        inst.stats = ActStats()
        return inst

    def __call__(self, x: mx.array) -> mx.array:
        self.stats.update(x)
        return super().__call__(x)


# ============================================================================
# Model walker — swap target Linears with ActStatsLinear
# ============================================================================


def _set_at_path(root, dotted_path: str, value) -> None:
    """Navigate model.<...>.<child_name> and replace child with value.

    Handles numeric indices for list-like sub-modules (e.g. layers.0)."""
    parts = dotted_path.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = value
    else:
        setattr(parent, last, value)


def install_act_stats(
    model: nn.Module,
    *,
    quant_suffixes: tuple[str, ...] = QUANT_SUFFIXES,
    verbose: bool = True,
) -> dict[str, ActStats]:
    """Walk model, replace every Linear whose path matches one of the given
    suffixes with an ActStatsLinear. Returns a dict {path: ActStats}.

    The dict shares references with the swapped modules — accumulating
    stats during forward passes shows up in this dict automatically.
    """
    # First pass: collect targets (don't mutate during iteration). MLX's
    # named_modules() recurses through nested sub-modules — does the right
    # thing for Lance's LanceModel → layers[i] → self_attn → q_proj depth.
    targets: list[tuple[str, nn.Linear]] = []
    for path, child in model.named_modules():
        if isinstance(child, nn.Linear) and any(path.endswith(s) for s in quant_suffixes):
            targets.append((path, child))

    if verbose:
        print(f"[act_stats] found {len(targets)} target Linears to instrument")

    stats_by_path: dict[str, ActStats] = {}
    for path, src in targets:
        wrapped = ActStatsLinear.from_linear(src)
        _set_at_path(model, path, wrapped)
        stats_by_path[path] = wrapped.stats

    if verbose:
        # Sanity-check coverage by suffix
        by_suffix: dict[str, int] = {s: 0 for s in quant_suffixes}
        for path in stats_by_path:
            for s in quant_suffixes:
                if path.endswith(s):
                    by_suffix[s] += 1
                    break
        print(f"[act_stats] coverage by suffix:")
        for s, count in sorted(by_suffix.items()):
            print(f"  {s:>30s}  {count:>4d}")

    return stats_by_path


# ============================================================================
# Save / load
# ============================================================================


def save_act_stats(stats: dict[str, ActStats], out_dir: Path | str,
                   metadata: dict | None = None) -> None:
    """Persist accumulated stats to an output directory:
       <out>/act_stats.safetensors    — {path + '.sum_abs': float32 array}
       <out>/act_stats_meta.json      — {path: {n_tokens, n_calls}}, plus
                                         optional calibration metadata.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pack sum_abs tensors into safetensors.
    arrays = {}
    meta = {"per_path": {}, "calibration": metadata or {}}
    skipped = []
    for path, s in stats.items():
        if s.sum_abs is None or s.n_tokens == 0:
            skipped.append(path)
            continue
        arrays[f"{path}.sum_abs"] = s.sum_abs.astype(mx.float32)
        meta["per_path"][path] = {
            "n_tokens": int(s.n_tokens),
            "n_calls": int(s.n_calls),
        }
    if not arrays:
        raise RuntimeError("save_act_stats: no non-empty stats to save")

    mx.save_safetensors(str(out_dir / "act_stats.safetensors"), arrays)
    (out_dir / "act_stats_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[act_stats] saved {len(arrays)} stats to {out_dir}")
    if skipped:
        print(f"[act_stats] WARNING: {len(skipped)} paths had no data (modules never invoked):")
        for p in skipped[:8]:
            print(f"  {p}")
        if len(skipped) > 8:
            print(f"  ... and {len(skipped) - 8} more")


def load_act_stats(in_dir: Path | str) -> dict[str, ActStats]:
    """Inverse of save_act_stats. Returns the {path: ActStats} dict
    ready for the AWQ apply pipeline."""
    in_dir = Path(in_dir)
    arrays = mx.load(str(in_dir / "act_stats.safetensors"))
    meta = json.loads((in_dir / "act_stats_meta.json").read_text())
    per_path = meta["per_path"]
    stats: dict[str, ActStats] = {}
    for full_key, arr in arrays.items():
        assert full_key.endswith(".sum_abs")
        path = full_key[: -len(".sum_abs")]
        m = per_path.get(path, {})
        stats[path] = ActStats(
            sum_abs=arr, n_tokens=m.get("n_tokens", 0), n_calls=m.get("n_calls", 0),
        )
    return stats
