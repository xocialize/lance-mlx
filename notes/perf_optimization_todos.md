# Performance optimization TODOs (post PR #4 review)

**Date:** 2026-05-29
**Context:** [PR #4](https://github.com/xocialize/lance-mlx/pull/4) added
the DPM-Solver++(2M) scheduler to `t2i.py`, giving ~2× wall-clock speedup
on image generation (4-prompt A/B: 1.6–2.8× depending on prompt complexity,
average 1.95× per the contributor's `Docs/DPM-vs-Euler/` measurements).

During PR review the contributor proposed adding a "Perf-1: mx.compile()
on Euler loop" item to BACKLOG.md. We asked them to revert that BACKLOG
addition (out of scope for a feature PR) but the underlying optimization
is genuinely valuable. **We may pick these up ourselves before the
contributor's follow-up lands.** This file is our internal TODO record
so the work isn't lost.

## TODO-1 — Extend DPM scheduler to remaining generation pipelines

**Status:** ✅ **Landed 2026-06-05.** `scheduler="euler" | "dpm"` kwarg
is now exposed on `t2v.py`, `image_edit.py`, and `video_edit.py` in
addition to PR #4's `t2i.py`. Default remains `"euler"` on all four
(opt-in for DPM). Validation regression tests cover all four pipelines
(ValueError on unknown scheduler before any model state is touched).
Per-pipeline docstring caveats added: t2v notes the Phase 5m
cfg_renorm interaction at fewer steps is untested at production video
scales; image_edit / video_edit note the richer per-step velocity
field (clean-ref + noisy-target cross-attention) adds error budget
to multistep extrapolation. Validate on your target use case before
relying on DPM for edit / video tasks.

**Goal:** Apply the same `DPMSolverPlusPlus2M` solver landed in PR #4
to the other three Euler-based generation pipelines:

| File | Pipeline | Current step pattern |
|------|----------|---------------------|
| `src/lance_mlx/pipeline/t2v.py` | TextToVideoPipeline | `latents = latents - velocity * dt` |
| `src/lance_mlx/pipeline/image_edit.py` | ImageEditPipeline | Same |
| `src/lance_mlx/pipeline/video_edit.py` | VideoEditPipeline | Same |

**Expected gain:** Same 1.6–2.8× wall-clock per pipeline. Largest user
impact for `t2v.py` — video generations are much longer than image
generations, so the absolute time saved per call is biggest there.

**Effort:** Low. Mechanical change per file:
1. Import `DPMSolverPlusPlus2M` from `lance_mlx.scheduler.solvers`
2. Add `scheduler: str = "euler"` kwarg to `generate()` with the same
   `ValueError` validation pattern PR #4 uses
3. Instantiate `solver = DPMSolverPlusPlus2M() if scheduler == "dpm"
   else None` before the flow loop
4. Replace `latents = latents - velocity * dt` with `solver.step(...)`
   if solver is enabled
5. Add backward-compat unit test pinning Euler output before/after

**Validation:** Each pipeline should be tested at small scale on a
representative prompt:
- t2v: 256²×9f, panda surf prompt (Phase 5j oracle), seed=42, both
  schedulers at 30/12 steps; compare PSNR on mid-frame
- image_edit: a "remove hat" prompt at 384²; visual A/B
- video_edit: 256²×9f mirror of image_edit prompt; visual A/B

**Dependency:** PR #4 must land first (provides `solvers.py`).

**Caveat — t2v specifically:** Video runs 30 Euler steps × 2 CFG arms
× variable t_lat. DPM at fewer steps may interact with our `cfg_renorm`
scaling (Phase 5m) in ways we haven't characterized. Recommend keeping
Euler-30 as default for t2v even after landing DPM, given video quality
is more sensitive to renorm interactions than image. The 5c-3h
"compounding through long forward passes" finding applies here too —
fewer steps means less compounding but also fewer corrections, and the
trade-off may not be as favorable as for t2i.

## TODO-2 — `mx.compile()` on flow-loop bodies (Perf-1)

**Status:** ✅ **Empirically investigated and closed 2026-06-05. No
practical win for Lance; not shipping.**

**Empirical result (`scripts/diagnostics/d_todo2_compile_ab_v2.py`,
isolated A/B on lance_model forward at 256², Mac17,7 / M5 Max / 128 GB):**

- Output equivalence: **bit-identical** (max|Δ| = 0.00e+00 across 6
  forwards) — `mx.compile` preserves correctness.
- Baseline forward mean: **808 ms**
- Compiled steady-state (forwards 2-6, excludes JIT trace): **831 ms**
- Steady-state delta: **-2.8% (regression)**
- One-time JIT trace overhead: **+853 ms** on first call

**Why no win:** Lance-3B is ~6.2B params; per-step compute is dominated
by `mx.fast.scaled_dot_product_attention` + matmul kernels that are
already heavily fused at the kernel level. `mx.compile` graph fusion
can't reduce dispatch overhead meaningfully when the existing path is
already mostly kernel-bound, but it does pay tracing overhead.

**Note on the original contributor's ~5% claim:** Their external repo
returned 404 before we could re-verify their measurement methodology.
Possibilities: they were testing against a smaller model where dispatch
overhead is proportionally larger, or measuring Metal cache warm-up
as compile effect (the first benchmark we ran here showed the same
~5% artifact when not properly isolating compile from warm-up).

**Why not ship as opt-in anyway:**

1. Negative steady-state speedup (regression, not just neutral)
2. JIT trace overhead penalizes single-shot users specifically
3. Adds kwarg surface area to all 4 pipelines for zero benefit
4. Easy for future-us to re-run the benchmark if MLX gains better JIT
   optimization at this scale; the v2 script is the regression guard.

**Scripts kept for future reference:**
- `scripts/diagnostics/d_todo2_compile_ab.py` — v1 (inconclusive;
  monkey-patch didn't intercept `self.lance_model(...)` syntax)
- `scripts/diagnostics/d_todo2_compile_ab_v2.py` — v2 (proper
  isolated forward A/B; the one to re-run if MLX changes)

If MLX upstream lands compile optimizations that close this gap (e.g.
better graph fusion across nn.Module dispatch boundaries, or fused
multi-layer kernels), re-run the v2 benchmark — if steady-state delta
flips positive we'd revisit then.

---

### Original framing (preserved for context)

(Original goal description follows; left in place for the record now
that the empirical investigation has closed the question.)

**Goal:** Wrap the per-step compute body in each generation pipeline
with `mx.compile()`. MLX JIT fuses the compute graph across the step,
eliminates Python-dispatch overhead, and collapses intermediate
allocations. No numerical change.

**Expected gain — empirically unknown.** Originally the PR #4
contributor claimed 2-4× wall-clock; their own external repo's README
showed ~5%. The latter is more believable for a compute-bound 13.6B
forward (Apple Silicon Linear ops are mostly memory-bandwidth-bound,
which `mx.compile` helps less than dispatch-bound workloads). With the
external repo no longer accessible we cannot reference their
measurement; we should treat the gain as **unknown until measured**.
A reasonable expectation: 5-15% wall-clock on top of whatever
scheduler (Euler or DPM) is selected, much larger on the warm-up case
if we ship batch-style usage patterns.

**Effort:** Low *per pipeline*. The per-step compute does cond + uncond
forwards plus the CFG renorm + Euler/DPM step; that fits a single
`mx.compile()`-able callable. Affects:
- `t2i.py`, `t2v.py`, `image_edit.py`, `video_edit.py`

**Validation:**
1. Pre/post wall-clock measurement on the same prompt+seed; should be
   non-decreasing (compile shouldn't slow anything down)
2. Output byte-identity check (`mx.compile` is graph optimization, not
   precision change — outputs must match exactly modulo nondeterminism
   in mx.fast.scaled_dot_product_attention)
3. Compile latency cost: `mx.compile` has a first-call warm-up. For
   single-image use cases the warm-up can dominate; for batch / server
   use it amortizes. Measure both.

**Caveats / risks:**
- `mx.compile` requires the inner function to have stable input shapes.
  The flow loop's per-step inputs ARE stable shapes (latents + velocity
  + dt), so should compile cleanly. Verify with a small smoke test
  before applying broadly.
- The CFG path has a conditional (`if cfg_scale_step > 1.0`) that
  branches on a runtime value — may need to factor into two compiled
  functions or use `mx.compile` with shape-static branches.
- Interaction with our existing `mx.eval(latents)` inside the loop
  (placed for Metal command-buffer hygiene): `mx.compile` may move the
  eval boundary. Verify command buffers don't accidentally grow past
  the 10s timeout at production scales (768²×17f t2v is the largest
  shipped envelope).
- Conflict potential with PR #4's `solver.step()` — the solver is
  stateful (stores `_v_prev`, `_dt_prev`). Stateful operations don't
  compose cleanly with `mx.compile` (state changes break referential
  transparency). Likely need to keep the solver `.step()` call OUTSIDE
  the compiled body and compile only the model-forward + CFG-renorm
  portion. This separation should still capture most of the gain since
  the model forward dominates the per-step cost.

**Dependency:** Independent of TODO-1. Can be done in any order. If
both land, the combined speedup may not be multiplicative (some compile
gains overlap with multistep gains since both reduce the dispatch
overhead per "useful" computation).

## Suggested ordering

1. **TODO-1 first** (t2v DPM port — highest absolute user impact)
2. **TODO-2 second** (mx.compile across pipelines — applies to whatever
   scheduler we ship by then)

Both could fit in one Phase 6a session: ~2-3 hours including
validation.

## Out of scope (for the record)

The PR #4 contributor also proposed two BACKLOG items unrelated to
performance:
- **Feature-1:** Multi-turn understanding FastAPI server
- **Feature-2:** Mode 3 Text → Text pipeline

These were asked to be surfaced as separate issues rather than
BACKLOG edits during PR review. If/when they land we can evaluate
independently. Neither is in our immediate critical path.
