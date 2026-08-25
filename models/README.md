# models/ — gatekeeper model inventory

Weight files (`*.keras`) are not committed, per `.gitignore`, and neither is `archive/`. The
reproducibility contract is training script + `data/processed/manifest_dedup.csv` and
`dedup_*.csv` + a fixed seed.

This file records which model is current and what has been archived, so the lineage stays
traceable.

## Top-level models

These were measured under different label definitions and on different test distributions, so
their F1 values are not directly comparable with each other.

**`gatekeeper_task2_mvp.keras`** — current recommendation, promoted 2026-06-25. Task 2 expanded
the negatives to fix person-driven false triggers (covariate shift). Against v4_mvp on the same
held-out probe:

- No-screen probe FP more than halved: at the same deployment threshold of 0.55, 51.1% down to
  21.7% (int8, 235 images). At matched recall of about 62%, v4 sits near 45% FP while task2 is
  at 30%, so task2 Pareto-dominates along the whole FP versus person-plus-screen recall curve.
- The honest cost: 5-seed test F1 0.757 to 0.734 (−0.023) and FN 0.230 to 0.290. The test set
  changed composition when 558 negatives were added, so it is not strictly the same measurement
  as v4's. This is a rebalancing that trades FP for FN, and it is not being dressed up as
  anything else.
- int8: all 11 operators on the TFLite Micro whitelist, fully int8, 32.4 KB. Same ESP32 budget
  as v4.
- The deployment threshold is not frozen. See the three-threshold table in
  `docs/threshold-tradeoff.md` for the FN, FP and probe trade-off; the choice is still open.
  Person-plus-screen recall was measured on the expanded probe (181 images, CI ±7pp) and its
  absolute value is lower than the earlier estimate from the 51-image probe. See the note in
  section 1 and the Pareto comparison in section 3 of that document.

**`gatekeeper_v4_mvp.keras`** — the best model before task2. Phase 3.4-B `v4_narrow`, narrowed
back to the MVP boundary at 1752 images. 5-seed F1 0.757 ± 0.012, FN 0.230; single-seed test F1
0.783 at threshold 0.55. Ambiguous hard negatives such as phone apps, TV menus and product
packaging were removed (see `docs/label-scope.md`). Its main weakness, the one task2 fixed, is
high person-driven false triggering: no-screen probe FP of 51.1% at threshold 0.55.

**`gatekeeper_v2_best.keras`** — phase 3.3 `p33_screen`, on the original clean distribution of
1518 images. Test F1 0.760 at threshold 0.45.

**`gatekeeper_v3_robust.keras`** — phase 3.4 `v3_screen`, wide boundary at 1950 images including
the ambiguous hard negatives. Test F1 0.736 at threshold 0.4. Its aggregate F1 is dragged down
by the ambiguous negatives; kept as a reference point for wide-boundary robustness.

All of these meet the ESP32 budget: 72 KB activations, 24.3 KB int8 weights, all operators
whitelisted. Metrics and verdicts are in `docs/gatekeeper-training-log.md`. Whether to proceed
to task2 hardware measurement is still to be decided.

**Correction, 2026-06-18, after a real export.** The "all operators whitelisted" claim above
originally came from a static operator count in `model.py`; it was not verified by an actual
`.tflite` export until 2026-06-18. The claim needs a qualifier: the whitelist holds only when
the model is exported with a fixed batch of 1. The default dynamic batch (-1) introduces three
non-whitelisted operators — SHAPE, STRIDED_SLICE and PACK — through flatten's dynamic reshape.
The export script `scripts/export_tflite.py` now pins `batch_shape=(1,96,96,1)`. Measured on
`gatekeeper_v4_mvp_int8.tflite`: 9 operators, all whitelisted, fully int8, weight data buffer
24.4 KB (agreeing with the 24.3 KB estimate), quantisation near-lossless at ΔF1 within 0.007.

Like `.keras` files, `.tflite` files are not committed and have to be copied to the device
manually. Still outstanding: ESP32 on-device verification — actual tensor arena occupancy
against on-chip SRAM, TFLM kernel numerical agreement, and measured latency and power.

## Archive

`models/archive/` holds one-off experiments. Nothing is deleted, only moved.

| File | Origin |
|---|---|
| gatekeeper_v1 / r1 / r2 | First version, trained on leaked data. Withdrawn |
| gatekeeper_dedup_v1 | Phase 2 dedup baseline (test F1 0.659 at 0.5) |
| gatekeeper_p32_posmult15 / posmult20 / focal | Phase 3.2 class weight and focal loss. Did not beat 3.1 |
| gatekeeper_p33_screen / screen_pm15 / screen_focal | Phase 3.3 augmentation experiments. `screen` was best and was promoted to v2_best |
| gatekeeper_smoke / timing_probe | Pipeline smoke test and timing probe |

When the phase 3.4 data expansion trains v3, if it is better it becomes the new top-level model,
this table is updated, and v2 is archived.
