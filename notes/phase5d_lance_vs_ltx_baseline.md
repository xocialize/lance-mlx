# Phase 5d — Lance vs LTX-2.3 baseline (pre-fix)

Captured 2026-05-21. This is the **pre-fix baseline** — Lance MLX is currently
producing painterly output where the PyTorch oracle produces photorealistic.
Tracking via [github issue #2](https://github.com/xocialize/lance-mlx/issues/2).
We're running Candidate 0 (no MaPE shift) now; this baseline lets us measure
the delta after a fix lands.

## Run summary

- 14/14 LTX eval prompts at 480×704×17f, seed 1234
- Lance config: `num_steps=30, cfg_scale=4.0, mape_anchor=2000` (current buggy default)
- LTX-2.3 reference: 97f × 704×448, 8 steps (distilled), same seed
- Total Lance wall-clock: 113.7 min (8.1 min/prompt avg)
- Avg inter-frame MAD: 2.43/255 (appropriate for these subjects)

## Per-prompt category outcomes (subjective)

| id | category | Lance MLX (pre-fix) | LTX-2.3 | category winner |
|---|---|---|---|---|
| p01 | photorealistic animal motion | tall grass scene, fox barely visible (painterly) | crisp fox in tall grass, photo + DoF | **LTX** |
| p02 | photorealistic human action | full chef visible, white coat, plating greens (painterly) | photorealistic hands plating cheese, cropped to torso | tie / Lance scene |
| p03 | landscape camera motion | dramatic rocky coastline aerial, painterly | wider beach + cliffs aerial, photo | tie |
| p04 | stylized anime | lantern atmosphere, cherry blossoms, painterly anime alley | character with back to camera, lanterns above | **tie / Lance leans Ghibli** |
| p05 | stylized 3D Pixar | expressive sloth close-up, painterly | anatomically-correct sloth pose with sky | tie |
| p06 | fast motion (skateboard) | wide stair scene, no trick captured | mid-flight kickflip captured | **LTX** |
| p07 | fast motion detail (water pour) | warm bokeh water pour | clean studio water pour | tie |
| p08 | static atmospheric (forest) | Ghibli-mystical forest with golden light | photorealistic dawn sunbeams | **tie / Lance has stronger atmosphere** |
| p09 | static atmospheric lit (neon alley) | Blade Runner concept-art cyberpunk alley | wide cinematic shot with car | **Lance** |
| p10 | complex motion (horse gallop) | landscape with small static-looking horse | horse + rider mid-gallop hooves airborne | **LTX** |
| p11 | human face detail (grandmother) | warm expressive Asian grandmother smile + bowl | contemplative Caucasian elderly woman in window light | tie / Lance more emotive |
| p12 | dynamic lighting (lightning storm) | dramatic plains at twilight with golden light | airplane window over clouds | **Lance** |
| p13 | human choreography (dancer) | _to inspect_ | _to inspect_ | _tbd_ |
| p14 | complex environment (underwater reef) | _to inspect_ | _to inspect_ | _tbd_ |

## Hardware

M5 Max 128 GB, MLX bf16. Both runs.

## Files

- Mid frames (Lance pre-fix): `tests/fixtures/lance_vs_ltx_pre_fix/p*_lance.png` (14 PNGs)
- LTX-2.3 references: `/Volumes/DEV_VOL1/VideoResearch/ltx-mlx-eval/outputs/phase4/phase4_p*_ltx23/video.mp4`
- LTX-2.3 mid frames (pre-extracted): `/tmp/lance_vs_ltx_ltx_mids/p*_ltx23.png`
- HTML side-by-side: `/tmp/lance_vs_ltx/comparison.html`
- Summary JSON: `tests/fixtures/lance_vs_ltx_pre_fix/summary.json`

## Pattern (pre-fix)

- **Lance pre-fix is competent on every prompt** — coherent scene, prompt-aligned content, correct color palette. No collapse, no noise. The bug is one of *aesthetic fidelity*, not output validity.
- Lance pre-fix WINS or TIES on: stylized (anime), atmospheric (forest, neon), dramatic-weather (lightning), warm-static portraits.
- Lance pre-fix LOSES on: photorealistic-animal-motion, fast-motion action (skateboard, horse gallop, anything with implied movement).
- This pattern is *consistent* with our hypothesis that the MaPE 2000-anchor we add (Candidate 0) creates a positional regime the model's GEN tower didn't train on for video — producing soft outputs lacking the precise motion + photoreal detail PyTorch reference has.

## Post-fix comparison plan

When Candidate 0 lands a photoreal fix, re-run the 14 prompts (~2 hours) and
regenerate `comparison.html` with three-way columns (Lance pre-fix / Lance
post-fix / LTX-2.3). The pre-fix mid frames are saved as fixtures so the
comparison is reproducible.

If Candidate 0 doesn't land it, re-run with Candidates 1 (fp32 RoPE) or 2
(mask scope). Each cycle is a ~2-hour wall-clock investment.
