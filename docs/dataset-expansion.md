# Task 2: expanding the training set to fix person-driven false triggers

Covariate shift. 2026-06-24. Model: `gatekeeper_task2_mvp` (5 seeds plus a single-seed
deployment model).

## 0. Conclusion

The fix worked, and it worked in the strong sense of Pareto dominance, but it needs two things
alongside it: the deployment threshold has to be re-calibrated, and a small regression in the
test-F1 measurement has to be accepted.

On the held-out probe, the new model cuts person-driven false triggers by more than half at
matched recall (51% to about 24%), or equivalently gains 31 percentage points of recall at
matched false-positive rate. It sits below the old model at every point on the "probe FP versus
people-plus-screen recall" curve.

The apparent over-correction seen earlier at the old fixed threshold of 0.55, where recall
seemed to drop from 61% to 47%, was an artefact of not re-calibrating: the new model's score
distribution shifted downward as a whole. Re-calibrating to roughly 0.45–0.50 makes it
disappear.

The one honest reservation: 5-seed test F1 goes from 0.757 to 0.734 (−0.023) and FN rises from
0.230 to 0.290. The test set itself changed when the negatives were added, so this is not a
strictly like-for-like comparison.

## 1. What was done to the data

Three download buckets — the comparison the advisor specified, plus a probe to catch
over-correction.

- Bucket A, people without screens, negative. 16 keywords x 35, net +560, into
  `data/raw/negative_clean/` (label 0). Deliberately non-office everyday scenes: street,
  market, commute, home, sport, service, outdoors, stands, train carriages.
- Bucket B, screens without people, positive. 9 keywords x 28, net +131. 120 were skipped by
  global ID dedup because they were the same image as an existing positive, which is correct.
  Into `data/raw/positive/` (label 1).
- Bucket C, people plus screen, held-out probe. 6 keywords x 18, net +51, into
  `data/probe_person_screen/`. Never enters training.

Leakage control, treated as a hard gate and fully passed:

- New script `scripts/guard_probe_overlap.py` cross-checks training images against both probe
  directories by ID and by perceptual hash. It found 2 new negatives colliding with the
  held-out `probe_person_noscreen` by Pexels ID (13200581 and 36299324). Those were
  quarantined by moving them to `data/_quarantine_task2/` rather than deleted, and the
  re-check reports 0 overlap.
- `check_leakage.py` over the 2440-image deduplicated pool reports 0 cross-split near-duplicate
  pairs and 0 within-split pairs.
- At probe time, a further ID and perceptual check confirms 0 leakage from either probe into
  the training pool.

Dataset size after dedup, under the same narrowed label boundary:

| Pool | Baseline v4 | task2 | Δ |
|---|---|---|---|
| Deduplicated training pool | 1752 | 2440 | +688 |
| Negatives (neg_clean + neg_noise) | 829 | 1387 | +558, the core fix |
| Positives | 923 | 1053 | +130 |

Excluded subclasses carry over from the baseline: `cosmetic_packaging_closeup,
grocery_product_label, product_packaging_text, smartphone_apps_home_screen,
tv_streaming_menu_screen`.

## 2. Headline: person-driven false triggers

Probe `probe_person_noscreen`, 235 images, ground truth "do not record", int8 deployment
measurement.

| Threshold | FP old (v4) | FP new (task2) | Δ |
|---|---|---|---|
| 0.50 | 53.6% | 23.8% | −29.8pp |
| 0.55 (old deployment point) | 51.1% | 21.7% | −29.4pp |
| 0.70 | 37.0% | 11.1% | −25.9pp |

The old baseline of 51.1% matches the roughly 51% the advisor described. It was locked in with
the same script before any data was touched.

## 3. Over-correction check

Probe `probe_person_screen`, 51 images, ground truth "record", int8. Recall here means the
fraction judged as record.

| Threshold | Recall old (v4) | Recall new (task2) |
|---|---|---|
| 0.50 | 66.7% | 60.8% |
| 0.55 | 60.8% | 47.1% |
| 0.45 | 74.5% | 64.7% |

Read at a fixed 0.55, the new model's recall dropping from 61% to 47% looks like it has learned
to ignore anything with a person in it. That reading is a threshold artefact. See section 4.

## 4. The trade-off curve, compared fairly

The new model's score distribution shifted down as a whole — the val-F1-optimal threshold moved
from 0.55 to roughly 0.35–0.50 — so the models must be compared at re-calibrated thresholds.

Equivalent operating points, int8:

- At matched recall of 60.8%: the old model needs threshold 0.55 and gives probe FP 51.1%; the
  new model needs only 0.49 and gives probe FP 24.7%. Same recall, less than half the false
  positives.
- At matched FP of 51%: the old model at 0.55 gives recall 60.8%; the new model at 0.22 gives
  recall 92.2%. Same FP, 31 percentage points more recall.
- Across the whole sweep, the new model's FP is below the old model's at every threshold.

The new model therefore dominates the old one along the entire probe-FP versus
people-plus-screen-recall curve. The recall regression in section 3 exists only because the old
threshold of 0.55 was carried over unchanged.

Recommended deployment threshold, re-calibrated on val rather than fitted on the probe: val F1
peaks on a plateau between 0.35 and 0.50 (0.733 at 0.50, 0.705 at 0.55). Taking roughly
0.45–0.50 gives probe FP around 24–27%, down from 51%, with people-plus-screen recall around
61–65%, at or above the old 60.8%. All three success criteria hold simultaneously at that
point.

## 5. Test set, stated honestly

5-seed deduplicated re-split, threshold 0.5.

| Metric | Baseline v4 | task2 | Δ |
|---|---|---|---|
| F1 | 0.7568 ± 0.0123 | 0.7343 ± 0.0248 | −0.0225 |
| recall | 0.7698 | 0.7095 | −0.060 |
| FN rate | 0.2302 | 0.2905 | +0.060 |
| FP rate | 0.2905 | 0.1667 | −0.124 |
| precision | 0.7527 | 0.7652 | +0.013 |
| accuracy | 0.7411 | 0.7800 | +0.039 |

Test F1 slips by 0.023 and FN rises slightly. This is a rebalancing that trades FP for FN.
Three qualifications apply. The test set itself changed composition when 558 negatives were
added, so the negative share rose and it is not strictly the same measurement as the baseline's
test set. The fair comparison, on the fixed held-out probes in sections 2 and 4, shows the new
model ahead. And the regression is about one to two baseline standard deviations while test FP
improves substantially.

Taken together this is not grounds for rejecting the change, but the regression has to be
labelled accurately rather than glossed over.

## 6. int8 export verification

`gatekeeper_task2_mvp_int8.tflite`:

- Operators: all 11 are on the TFLite Micro whitelist.
- Dtypes: 17 int8 and 5 int32 tensors, with zero float32 internal tensors. Fully int8.
- Size: 32.4 KB file, 24.3 KB of weights, matching 24,874 parameters at 1 byte each and
  agreeing with the earlier estimate.
- Quantisation loss: test ΔF1 of −0.006 at threshold 0.5 and −0.014 at 0.55. Acceptable.

## 7. Success criteria, decided one by one

| Criterion | Verdict |
|---|---|
| Probe FP drops substantially from about 51% | Met. 51% to about 24% at matched recall (−27pp); at the old threshold, 51.1% to 21.7% |
| Original test F1/FN/FP do not regress | Partly. Test FP improves a lot, but F1 is −0.023 and FN +0.06. The test set has changed, so this is a measurement regression that must be labelled |
| People-plus-screen recall holds up | Met, after re-calibration. Recall is 61–65% at 0.45–0.50, at or above the old 60.8%. At the old 0.55 it regresses, which is the threshold artefact |

## 8. Remaining risks and limitations

1. No manual QC. The 560 new negatives have not been checked one by one for readable screens
   creeping in, which cannot be judged reliably by an automatic check.
   `data/processed/task2_qc_montages/*.png` holds a 9-image montage per subclass for spot
   checking. The keywords deliberately avoid office, meeting and classroom scenes to reduce the
   risk.
2. The person-plus-screen probe is only 51 images, because global dedup skipped a lot of what
   some keywords returned. Recall has a 95% confidence interval of roughly ±14pp, so the
   direction of the conclusion is sound but individual point values are noisy.
3. The deployment threshold has not been committed. This report gives a recommended range only;
   no deployment configuration, README or model selection was changed. Still to be decided.
4. Not done: promoting task2_mvp to top-level best, ESP32 on-device verification, and
   re-measuring the test set under an aligned protocol.

## 9. Artifacts

- Models: `models/gatekeeper_task2_mvp.keras`, `_float32.tflite`, `_int8.tflite` (gitignored).
- Data manifests: `data/processed/manifest.csv`, `manifest_dedup.csv`,
  `dedup_{train,val,test}.csv`.
- Metrics: `docs/results/variance_results_task2.json`,
  `docs/probes/probe_fp_{before,after}.md`, `probe_personscreen_{before,after}.md`,
  `leakage_task2_dedup.csv`.
- New scripts: `scripts/guard_probe_overlap.py`, `scripts/probe_fp_test.py`, and three
  `keywords_task2_*.json` files.
- QC montages: `data/processed/task2_qc_montages/`.
- Quarantine: `data/_quarantine_task2/`, holding the 2 negatives that collided with the probe.
  Moved, not deleted.
