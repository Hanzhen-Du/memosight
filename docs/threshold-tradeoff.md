# Deployment threshold ledger

2026-06-25. Model: `gatekeeper_task2_mvp`, the single-seed deployment model.

This is the input to decision point 1, fixing the deployment threshold. It gives measured
FN, FP and probe trade-offs at each candidate threshold, plus a fair comparison against v4 on
the expanded 181-image probe, where the confidence interval is narrower.

## 0. What each column was measured with

Read this before comparing anything across columns.

The test columns (FN, FP, recall, F1) are keras float, from `evaluate.py --pr-sweep` run on
`dedup_test.csv` (n=369: 159 positive, 210 negative). This is the project's established test
measurement, the same source as the 5-seed report, on the single-seed deployment model.

The two probe columns are int8 tflite under deployment conditions, from `probe_fp_test.py` run
on the held-out probes. `probe_person_noscreen` has n=235 with ground truth "do not record", so
its FP rate is the fraction judged as record. `probe_person_screen` has n=181 after expansion,
with ground truth "record", so its recall is the fraction judged as record.

Caveat on mixing the two: test is keras and the probes are int8. Section 6 of the task2 report
shows int8 quantisation costs only 0.006 F1 on test at threshold 0.5, so the operating point
barely moves, but strictly speaking the test column and the probe columns do not come from the
same engine. The trade-off that actually drives the decision, probe FP against recall, is
already entirely in int8 deployment terms, so this does not affect which threshold to pick.
There is no existing labelled int8 test evaluation script, so the established keras test
measurement is kept.

## 1. Three candidate thresholds

task2 deployment model.

| Threshold | test FN | test FP | noscreen probe FP | person+screen probe recall |
|---|---|---|---|---|
| 0.40 | 42/159 = 26.4% | 55/210 = 26.2% | 71/235 = 30.2% | 112/181 = 61.9% (95% CI ±7.1pp) |
| 0.45 | 46/159 = 28.9% | 46/210 = 21.9% | 64/235 = 27.2% | 100/181 = 55.2% (95% CI ±7.2pp) |
| 0.50 | 50/159 = 31.5% | 37/210 = 17.6% | 56/235 = 23.8% | 92/181 = 50.8% (95% CI ±7.3pp) |

How to read it, given that for this product FN matters more than FP — a missed screen is a
memory lost permanently:

Going from 0.50 to 0.40 buys test FN 31.5% to 26.4%, which is 8 fewer missed positives, and
person-plus-screen recall 50.8% to 61.9%, up 11 points. It costs test FP 17.6% to 26.2% and
no-screen probe FP 23.8% to 30.2%, up 6.4 points.

So having already halved probe FP relative to the old v4's 51%, moving further down in
threshold within task2 trades roughly 5–6 points of recall and FN improvement for roughly 3–6
points of extra FP per 0.05 step. All three settings keep no-screen FP between 24% and 30%,
still far below v4's 51%.

If FN is the priority, 0.40 gives the lowest FN at 26.4% and the highest person-plus-screen
recall at 61.9%, while no-screen FP is still only 30%.

One note on the recall figures. The absolute person-plus-screen recall is about 10 points lower
than the old 51-image probe suggested, which read 60.8% at threshold 0.50. The small probe was
optimistic and noisy at ±14pp. Expanding to 181 images moved the point estimate down and
narrowed the interval to ±7pp, and that is the more trustworthy value. Section 3 shows task2
still Pareto-dominates v4 even so.

## 2. Full test threshold sweep

keras, `evaluate.py --pr-sweep`, excerpt.

| Threshold | Precision | Recall | F1 | FN rate | FP rate |
|---|---|---|---|---|---|
| 0.35 | 0.660 | 0.780 | 0.715 | 0.220 | 0.305 |
| 0.40 | 0.680 | 0.736 | 0.707 | 0.264 | 0.262 |
| 0.45 | 0.711 | 0.711 | 0.711 | 0.289 | 0.219 |
| 0.50 | 0.747 | 0.686 | 0.715 | 0.315 | 0.176 |
| 0.55 | 0.805 | 0.648 | 0.718 | 0.352 | 0.119 |

F1 sits on a plateau of 0.707 to 0.718 across 0.40 to 0.55, so F1 does not distinguish these
settings. What distinguishes them is the FN/FP trade.

## 3. Against v4 on the expanded probes

int8, same 181 and 235 images for both models.

Person-plus-screen recall (n=181):

| Threshold | v4 recall | task2 recall |
|---|---|---|
| 0.40 | 82.9% | 61.9% |
| 0.45 | 77.9% | 55.2% |
| 0.50 | 73.5% | 50.8% |
| 0.55 | 69.1% | 42.0% |

No-screen probe FP (n=235):

| Threshold | v4 FP | task2 FP |
|---|---|---|
| 0.40 | 60.9% | 30.2% |
| 0.50 | 53.6% | 23.8% |
| 0.55 | 51.1% | 21.7% |
| 0.60 | 47.7% | — |
| 0.65 | 43.4% | — |
| 0.70 | 37.0% | — |

Pareto verdict, same measurement and same probes: v4's no-screen FP is at least 37% at any
threshold, its best being 0.70, while task2's worst is 30.2% at 0.40. The two FP ranges do not
overlap. At matched recall of about 62%, task2 sits at 30% FP while v4 needs a threshold near
0.62 and gives about 45%.

task2 therefore still Pareto-dominates v4 along the whole no-screen-FP versus
person-plus-screen-recall curve, and expanding the probe did not change that conclusion — if
anything it made it more conservative.

Note that at the *same* threshold task2's recall is lower than v4's. That is because task2's
score distribution shifted down as a whole. A fair comparison has to be made at matched
operating points, not at matched thresholds.

## 4. Artifacts

- task2: `data/processed/probe_personscreen_audit_task2/`, `probe_noscreen_audit_task2/`
- v4 comparison: `data/processed/probe_personscreen_audit_v4/`, `probe_noscreen_audit_v4/`
- Reports: `docs/probes/probe_personscreen_after_expanded.md`,
  `probe_noscreen_after_task2_thresholds.md`, `probe_personscreen_v4_on_expanded.md`,
  `probe_noscreen_v4_grid.md`
- Probe expansion: 51 grew to 182 downloaded, then 1 was removed for being a near-duplicate of
  a training positive (correlation 0.99, moved to `data/_quarantine_task2_probe/`), leaving 181
  clean held-out images.

## 5. Zero-leakage verification

All pass after the expansion.

- `guard_probe_overlap.py` over data/raw (2749) against both probes (416): overlap_hits = 0
- probe against probe, person_screen 181 against person_noscreen 235: overlap = 0
- `check_leakage.py` over manifest_dedup (2440): 0 cross-split pairs, 0 within-split pairs
- The leakage check built into `probe_fp_test` on the expanded probe: n_leaked = 0
