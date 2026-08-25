# Gatekeeper training log, phases 2 and 3

2026-06-17.

All metrics are based on the deduplicated manifest (`dedup_train/val/test.csv`, seed 42). Test
is the deciding split; val is used for tuning and for choosing thresholds.

The honest starting point after deduplication is test F1 0.659 and FN 0.408 at threshold 0.5.
The acceptance target is test F1 at or above 0.85, with FN clearly below that starting point.

**Correction, 2026-06-18, after a real export.** Where this file says the operators are all on
the TFLite Micro whitelist, that originally came from a static operator count in `model.py` and
had not been verified by exporting. The first real `.tflite` export showed the whitelist
guarantee holds *only* when the model is exported with a fixed batch of 1. The default dynamic
batch (-1) pulls in three non-whitelisted operators — SHAPE, STRIDED_SLICE and PACK — via
flatten's dynamic reshape. Exported with `batch_shape=(1,96,96,1)` pinned, the measured result
is 9 operators all whitelisted, fully int8, 24.4 KB of weights, and near-lossless quantisation.
Read the whitelist claims below as "verified by export at batch 1", not as a modelling-time
estimate.

## Phase 2: full metrics

Baseline model `gatekeeper_dedup_v1.keras`, threshold 0.5.

Configuration: `bn_momentum=0.9, patience=15, start_from_epoch=20, epochs=80` (early stop at
65, best at 50), `augment=base, class_weight=balanced, lr=1e-3`. 24,874 parameters.

| split | accuracy | precision | recall | F1 | FN rate | FP rate | confusion (tn,fp,fn,tp) |
|---|---|---|---|---|---|---|---|
| val (tuning) | 0.8097 | 0.8046 | 0.7292 | 0.7650 | 0.2708 | 0.1308 | 113,17,26,70 |
| test (deciding) | 0.7403 | 0.7436 | 0.5918 | 0.6591 | 0.4082 | 0.1504 | 113,20,40,58 |

Machine-readable: `docs/results/phase2_metrics.json`.

The val-to-test gap is large: val F1 0.765 against test 0.659. The weights selected by
early stopping on val_loss give low recall on test positives (0.592) and FN as high as 0.408.
That is the honest starting point.

## Phase 3: raising the score

Ordered lightest intervention first, checking at each step whether FN actually came down.

Rule: pick the threshold on val by maximising val F1, breaking ties toward lower FN, then apply
that threshold to test for the verdict. Each step is compared against the previous one on test
FN.

### 3.1 Threshold sweep

Free, so it goes first.

The val threshold sweep peaks at 0.25 (val F1 0.8325, recall 0.906, FN 0.094, FP 0.200).
Applying threshold 0.25 to test:

| Stage | Threshold | test F1 | test recall | test FN rate | test FP rate | test acc |
|---|---|---|---|---|---|---|
| Baseline (3.0) | 0.50 | 0.6591 | 0.5918 | 0.4082 | 0.1504 | 0.7403 |
| 3.1 threshold | 0.25 | 0.7196 | 0.7857 | 0.2143 | 0.2932 | 0.7359 |

FN goes from 0.408 to 0.214, a real drop of 0.194. F1 goes 0.659 to 0.720 and recall 0.592 to
0.786. The cost is FP rising from 0.150 to 0.293, which is acceptable — for a gatekeeper, a
miss is worse than a false trigger.

One free step nearly halves FN, but F1 is still well below 0.85, so on to 3.2.

For reference, 3.1 lifts the baseline model's val F1 from 0.765 at threshold 0.5 to a peak of
0.8325 at 0.25, with test F1 0.720. Later steps are compared on "test F1 and FN after picking
the threshold on val".

### 3.2 Class weighting and focal loss

Three configurations retrained on dedup seed 42 (`scripts/run_phase3.py`), with the threshold
still chosen on val by maximum F1.

| Configuration | val threshold | val F1 | val FN | test F1 | test FN | test FP |
|---|---|---|---|---|---|---|
| 3.1 baseline + threshold (control) | 0.25 | 0.8325 | 0.094 | 0.7196 | 0.2143 | 0.2932 |
| pos_weight x1.5 | 0.40 | 0.7917 | 0.208 | 0.7090 | 0.3163 | 0.1805 |
| pos_weight x2.0 | 0.45 | 0.7895 | 0.219 | 0.7526 | 0.2551 | 0.1729 |
| focal (γ=2, α=0.75) | 0.55 | 0.8041 | 0.188 | 0.7385 | 0.2653 | 0.1880 |

3.2 does not beat 3.1. All three configurations have val F1 between 0.79 and 0.80, below 3.1's
0.8325, and test FN between 0.255 and 0.316, all above 3.1's 0.214. FN did not come down. The
val and test rankings also contradict each other — focal is best on val while pos_weight x2.0
is best on test — which is what a 226-sample val set with high variance looks like.

Diagnosis: `class_weight=balanced` was already active in the baseline, so adding more weighting
or switching to focal only translates the decision boundary, which is equivalent to adjusting
the threshold. It does not improve the model's PR curve.

The real bottleneck is the val-to-test generalisation gap (val F1 0.83 against test 0.72), which
data augmentation should address directly. Following the rule of not stacking on top of a
change that did not help, 3.3 does not build on 3.2. It returns to the base configuration and
tests the screen augmentation on its own.

### 3.3 Augmentation tuned for screen scenes

Small rotation up to ±7 degrees, scaling, and JPEG compression noise at quality 45–95, layered
on top of the base flip/brightness/contrast augmentation. Magnitudes are kept small so text
stays readable. Three configurations retrained:

| Configuration | val threshold | val F1 | test F1 | test FN | test FP |
|---|---|---|---|---|---|
| screen (base configuration) | 0.45 | 0.8125 | 0.7600 | 0.2245 | 0.1955 |
| screen + pos x1.5 | 0.45 | 0.7638 | 0.6939 | 0.3061 | 0.2256 |
| screen + focal | 0.65 | 0.8000 | 0.7282 | 0.2755 | 0.1955 |

`p33_screen`, the pure screen augmentation, reaches test F1 0.760, the highest so far — above
3.2's pos_weight x2.0 at 0.753 and 3.1's 0.720.

A warning that belongs with that number: on val, `p33_screen` peaks at 0.8125, slightly *below*
the baseline's 0.8325, and its 0.04 advantage on test sits inside the ±0.024 noise band measured
by the variance check. It cannot be claimed that screen augmentation reliably beats baseline.
Its real value is robustness for wearable deployment, since rotation and compression noise are
closer to how frames will actually be captured. That is the reason to keep it as a deployment
candidate, not the metric.

## Phase 3 summary and operating points

### Threshold curve for the best model

`gatekeeper_p33_screen.keras`, excerpt.

| Selection basis | Threshold | test F1 | test recall | test FN | test FP |
|---|---|---|---|---|---|
| Max val F1 | 0.45 | 0.7600 | 0.7755 | 0.2245 | 0.1955 |
| FN priority on val (val F1 still 0.81) | 0.35 | 0.7306 | 0.8163 | 0.1837 | 0.3083 |
| Balanced (gatekeeper power trade-off) | 0.40 | 0.7393 | 0.7959 | 0.2041 | 0.2632 |

For a gatekeeper, a miss should be suppressed harder than a false trigger, but every false
trigger is directly the power cost of waking the expensive downstream stage. The final
threshold should be chosen on the task2 power-versus-missed-capture Pareto curve; these are the
options.

### Progress so far

Test split, positive class = 1.

| Milestone | test F1 | test FN | Note |
|---|---|---|---|
| First honest baseline after dedup, at 0.5 | 0.659 | 0.408 | Starting point after the leakage fix |
| 3.1 threshold at 0.25 | 0.720 | 0.214 | Free, halves FN, the most reliable single step |
| 3.2 class weight / focal | — | — | Inside the noise band, did not beat 3.1 |
| 3.3 screen augmentation at 0.45 | 0.760 | 0.224 | Highest test F1; at 0.35 FN drops to 0.184 |

### ESP32 portability budget

The architecture (`build_model`) did not change at any point: 24,874 parameters. Re-checked with
`scripts/model.py`:

- Peak single-layer activation 72 KB, concurrent input plus output within a layer 144 KB, both
  under 256 KB.
- int8 weights about 24.3 KB, under 100 KB.
- Operators: Conv2D, ReLU, MaxPool2D, AveragePooling2D, Reshape, Dense, Softmax, with batch
  norm folded into Conv at export. All on the TFLite Micro whitelist.

### Acceptance criteria

| Criterion | Status |
|---|---|
| Credibility report complete, verdict on whether 0.80 is real | Pass. Accuracy is real and stable; FN 0.277 was an artefact |
| Full metrics including F1, reproducible from one script | Pass (`evaluate.py`) |
| test F1 at or above 0.85 | Not met. Best is 0.760 |
| FN clearly below the first honest values (0.408 / 0.337) | Pass. Down to 0.18–0.22 |
| Still inside the ESP32 budget and the TFLM whitelist | Pass |
| Key decisions and failures recorded | Pass |

Phases 3.1 to 3.3, all lightweight, took test F1 from 0.659 to 0.760 and FN from 0.408 to
0.18–0.22. The F1 target of 0.85 was not reached with the current data and current backbone,
and progress has plateaued — the val-to-test gap plus the high variance of a 226-sample val set.

Going further means entering higher-risk territory:

- 3.4, more data. Requires fixing `download_images.py` to deduplicate by Pexels ID first, to
  avoid re-introducing leakage. Stop before starting this and get a decision.
- 3.5, a stronger backbone. Must still fit the ESP32 budget and the whitelist. Last resort. Stop
  and get a decision.

This round stops at the 3.4 boundary after completing 3.3.

## Phase 3.4: targeted data expansion

Authorised and executed. Contains a negative result.

### What was done

1. Fixed `download_images.py`. The old logic deduplicated by Pexels ID only within a single
   folder, so the same image was downloaded under multiple keywords and leaked across splits.
   It now deduplicates by ID globally across keywords. A self-check found 85 cross-folder
   duplicate IDs in the existing `data/raw`.
2. Targeted the expansion using the FN/FP analysis. High-FN positives were
   laptop_screen_code 0.333, powerpoint 0.300, and classroom/projector 0.24. High-FP hard
   negatives were office_interior 0.471, product_packaging 0.385, phone_lock 0.348 and
   video_playback 0.280.
3. Found that Pexels returns the same set of images for closely related keywords — paging deeper
   on the same concept adds essentially nothing. Expanding the dataset requires keywords for
   genuinely different concepts. 14 new keywords were added.
4. Result: raw grew 1628 to 2060 (+432 unique images) and positives 751 to 1030. After dedup,
   1518 grew to 1950, with the same 110 removed as before, meaning the new images introduced no
   new duplicates. The re-split moved val from 226 to 291 and test from 231 to 295, with the
   positive-to-negative ratio going from 0.74 to 0.90. `check_leakage` reports 0 cross-split
   leakage.

### 5-seed variance, before and after the expansion

Same protocol, base augmentation, test as the deciding split.

| Metric | Old (1518) | New (1950) | Change |
|---|---|---|---|
| accuracy | 0.790 ± 0.014 | 0.721 ± 0.014 | −0.069 |
| F1 | 0.756 ± 0.024 | 0.699 ± 0.024 | −0.057 |
| recall | 0.771 ± 0.068 | 0.694 ± 0.069 | −0.077 |
| FN rate | 0.229 ± 0.068 | 0.306 ± 0.069 | +0.077 |
| FP rate | 0.197 ± 0.045 | 0.255 ± 0.067 | +0.058 |

Aggregate F1 fell rather than rose after adding data, and the standard deviation barely moved
(F1 still ±0.024, recall still ±0.069).

That falsifies the hypothesis that the 226-sample val set's high variance was the main cause of
the plateau. Adding data did not make the metrics steadier, so the variance is intrinsic rather
than small-sample sampling noise.

### Diagnosis: why did it get worse?

v3 model, old subclasses against new ones.

| Group | Positive FN | Negative FP |
|---|---|---|
| Old subclasses | 0.135 (better than v2's 0.206) | 0.288 |
| New subclasses | 0.214 | 0.500 |

Adding data made the model stronger on the original distribution — FN on old positives went
from 0.206 to 0.135 — but the new concepts widened the problem distribution and made it harder.
The new negatives (smartphone apps, TV menus, product labels — the classic "has text but should
not be recorded" cases) have FP as high as 0.50, and are the new bottleneck. The drop in
aggregate F1 reflects a harder and more realistic distribution rather than a loss of capability.

Some of the new positives are also hard: conference keynote, webinar and data dashboard scenes
have FN of 0.3 to 0.5 with low mean positive probability, and in several of those images the
text is not prominent, putting them close to the boundary.

### v3_robust full metrics

Deployment candidate, threshold 0.4 chosen on val.

| split | acc | precision | recall | F1 | FN rate | FP rate |
|---|---|---|---|---|---|---|
| val (tuning) | 0.7766 | 0.7110 | 0.8913 | 0.7910 | 0.1087 | 0.3268 |
| test (deciding) | 0.7322 | 0.6875 | 0.7914 | 0.7358 | 0.2086 | 0.3205 |

ESP32: architecture unchanged, peak activation 72 KB, int8 weights 24.3 KB, all operators
whitelisted. Within budget.

v3's test F1 of 0.736 is not directly comparable with v2_best's 0.760, because the test sets
differ — 295 images on a harder distribution against 231. The same-protocol variance comparison
in the table above is the one that decides.

### Acceptance criteria at the end of 3.4

| Criterion | Status |
|---|---|
| test F1 at or above 0.85 | Not met. 0.70 under the same protocol, lower than before the expansion |
| FN below the first honest value (0.408) | Pass. 0.21 at the deployment point |
| ESP32 budget and TFLM whitelist | Pass |
| Process recorded | Pass |

### Verdict: would more data reach 85%, and should the current model go to hardware?

Verdict: simply adding more generic Pexels data is unlikely to reach 0.85 and may lower
aggregate F1 further. The recommendation is to accept the current model for task2 hardware
measurement, and to treat "reach 85%" as a separate scope question requiring a decision.

The reasoning:

1. The variance is intrinsic, not driven by sample size. Adding 432 images and expanding val by
   29% left the standard deviation unchanged. The bottleneck is not that there is too little
   data, so more of the same will not converge it to 0.85.
2. The new bottleneck is ambiguity in the boundary itself. The new hard negatives have FP 0.50.
   Whether phone app text, a TV menu or a product label counts as "useful text" is semantically
   unclear. That is a label-definition problem, not a data-volume problem, and another thousand
   phone screenshots will not rescue an ill-defined boundary.
3. Available data is close to exhausted. Paging deeper on the same concept in Pexels adds
   nothing, so expanding further means introducing messier new concepts, which is exactly what
   lowered the aggregate metrics.
4. On the clean original distribution the model is trending well — FN on old positives is
   already 0.135. Narrowing the scope back to the launch boundary of whiteboards, documents,
   slides and code screens (`docs/label-scope.md`) would leave the existing model close to
   usable. It was phase 3.4 that widened the negative boundary out to phones, televisions and
   packaging, which are hard ambiguous classes the launch scope should not have included.

Three options, all beyond what this round decides on its own:

- A. Accept the current model and go to task2 hardware measurement, using v2_best (F1 0.760 on
  the original clean distribution) or v3_robust (more robust but on a harder distribution), and
  produce the power-versus-missed-capture Pareto curve. Recommended: the MVP deliverable is the
  Pareto curve, not a single F1 number.
- B. Narrow the label boundary. Define phone app, TV menu and product label as outside the
  positive trigger scope and remove the ambiguous items from the hard negatives, then retrain.
  Aggregate F1 should recover as the boundary gets cleaner.
- C. Unblock 3.5 and change the backbone. If 0.85 is required while keeping the wide boundary,
  a higher-capacity model is needed — but that conflicts with the "very small specialised
  gatekeeper" premise, which has already been ruled out.

All three are direction decisions, so this round stops here.

## Phase 3.4-B: narrowing the label boundary back to the MVP definition

Option B was chosen and executed.

### The boundary change

Formally recorded in `docs/label-scope.md`.

Positive class is the launch definition of a useful text screen: projector, computer screen,
slides, whiteboard, document page, code screen.

Ambiguous hard negatives removed — cases with text that are not launch scenes: product
packaging and labels (`product_packaging_text` plus 3.4's `grocery_product_label` and
`cosmetic_packaging_closeup`), phone apps (`smartphone_apps_home_screen`) and TV menus
(`tv_streaming_menu_screen`). Five subclasses, 198 images.

Clear negatives kept: phone lock screens, screens playing video, signage and street signs, book
spines, and textless landscapes, portraits, interiors and food.

Removed rows are archived in `data/processed/manifest_out_of_scope.csv`. The images are not
deleted, only excluded from training and evaluation, so the change stays traceable.

Data: 2060 dropped to 1862 after removal, and 1752 after dedup (923 positive, 829 negative,
with the positive-to-negative ratio flipping to 1.11). Re-split gives val 261 and test 265.
`check_leakage` reports 0 cross-split leakage.

### 5-seed variance across the three label definitions

Same protocol, base augmentation, test as the deciding split, threshold 0.5.

| Metric | Baseline 1518 (clean, old) | Wide boundary 1950 (3.4) | Narrowed 1752 (B) |
|---|---|---|---|
| accuracy | 0.790 ± 0.014 | 0.721 ± 0.014 | 0.741 ± 0.018 |
| F1 | 0.756 ± 0.024 | 0.699 ± 0.024 | 0.757 ± 0.012 |
| recall | 0.771 ± 0.068 | 0.694 ± 0.069 | 0.770 ± 0.066 |
| FN rate | 0.229 ± 0.068 | 0.306 ± 0.069 | 0.230 ± 0.066 |
| FP rate | 0.197 ± 0.045 | 0.255 ± 0.067 | 0.290 ± 0.100 |

Narrowing lifts F1 from the wide boundary's 0.699 back to 0.757, and halves the standard
deviation from 0.024 to 0.012. That confirms the 3.4 diagnosis from the opposite direction: the
drop under the wide boundary was caused by boundary ambiguity, and removing the ambiguous
negatives both restores F1 to the clean baseline level and makes it steadier. FN and recall
recover in step, at 0.230 and 0.770.

The cost is FP rising to 0.290 ± 0.100. The reason is clear. Removing the "has text but do not
record" negatives — packaging, apps, TV menus — took away the examples that taught the model
that text alone is not a reason to record, and the dataset now leans positive. The model
therefore tilts toward predicting positive, so false triggers on the remaining negatives (video
screens, signage) rise, and FP is unstable across seeds at std 0.10.

Measurement note: this table is under the MVP label definition and is not directly comparable
with the wide-boundary numbers. Even the comparison against baseline 1518 is only approximate,
since 1752 includes the harder positives added in 3.4.

### v4_mvp full metrics

Deployment candidate, threshold 0.55 chosen on val, single seed 42.

| split | acc | precision | recall | F1 | FN rate | FP rate |
|---|---|---|---|---|---|---|
| val (tuning) | 0.7893 | 0.7943 | 0.8116 | 0.8029 | 0.1884 | 0.2358 |
| test (deciding) | 0.7736 | 0.7883 | 0.7770 | 0.7826 | 0.2230 | 0.2302 |

ESP32: architecture unchanged, peak activation 72 KB, int8 weights 24.3 KB, all operators
whitelisted. Within budget.

### Acceptance criteria at the end of 3.4-B

| Criterion | Status |
|---|---|
| test F1 at or above 0.85 | Not met. 5-seed 0.757 ± 0.012; best single seed 0.783 |
| FN below the first honest value (0.408) | Pass. 0.22, a clear drop |
| Did the standard deviation narrow | Pass. F1 std 0.024 to 0.012, halved |
| ESP32 budget and TFLM whitelist | Pass |
| Boundary change recorded in `docs/label-scope.md` | Pass |

### Verdict: is the model reliable enough for task2 hardware measurement?

Technical judgement: under the narrowed MVP boundary the model is a stable, portable, honest
artifact with FN under control, and it is technically ready to begin hardware Pareto
characterisation. It is not a finished 0.85-F1 classifier, and FP is currently its weaker
dimension.

In favour: F1 0.757 ± 0.012, with variance halved, which suggests the problem is more learnable
and more stable under a clean boundary. FN 0.22, far below the first honest 0.408. The
threshold is an existing knob for trading FN, FP and power against each other — moving the test
threshold from 0.45 to 0.55 moves FP from 0.29 to 0.23 and FN from 0.17 to 0.22. And the ESP32
budget is met. The task2 MVP deliverable is a power-versus-missed-capture Pareto curve, which
needs exactly this: a stable, portable gatekeeper with an adjustable FN, rather than a single
high F1.

Against: F1 has not reached 0.85. FP sits at roughly 0.23 to 0.29 and is unstable across seeds
at std 0.10, meaning about a quarter of non-launch frames would falsely wake the downstream
stage. That is a power cost, and quantifying it is precisely what task2 is for, but it also
means the false-positive dimension is not yet mature.

Whether to move to hardware is a decision to be taken, not one to be assumed. If the goal is a
Pareto curve validating the cascade concept, the current model is good enough and honest about
its limits. If the expectation is a clean 0.85 classifier before touching hardware, it is not
there.

If the goal is to keep pushing F1 instead of going to hardware, the best value next is not more
data — 3.4 already showed that does not work. It would be, in order: hard-negative augmentation
targeting the retained negatives with high FP (video screens, signage); shifting the threshold
and class weights toward lower FP and accepting slightly lower recall; and only if neither is
enough, revisiting capacity (3.5, already ruled out).
