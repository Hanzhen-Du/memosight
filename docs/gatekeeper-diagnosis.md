# Why the gatekeeper's accuracy stopped improving

Consolidated report, 2026-07-28.

Two intuitive explanations for the F1 plateau were tested and both turned out to be wrong.
Ruling them out is what located the actual bottleneck, so this report is organised around the
elimination rather than around the final number.

Every figure here was measured on this machine. Sources: `docs/model-capacity-sweep.md`,
`docs/targeted-negatives-retrain.md`, `docs/false-positive-diagnosis.md`,
`docs/threshold-tradeoff.md`, `docs/data-leakage-audit.md`, `docs/dataset-expansion.md`, and
the experiment log. Raw JSON is under `docs/results/task1_results/` and
`docs/results/task2b_results/`.

## 0. Summary

The change in "training method" was not a change to the model. It was a change to the
experimental protocol. The old way was one training run, one seed, un-deduplicated data, and
metrics picked after the fact. The new way is five seeds, deduplication by Pexels ID with a
zero-leakage check, fixed held-out probes, a threshold fixed at 0.40, and int8 deployment
measurement. The immediate consequence: the FN of 0.277 reported under the old protocol does
not survive, and the true value is 0.337. Part of the old "good number" was data leakage and
part was seed luck.

Accuracy did not stop improving for lack of effort. Two hypotheses were eliminated one at a
time.

- Capacity. Parameters swept 3.8x, from 24.9k to 94.5k. Test F1 sat on a 0.736–0.769 plateau
  and the three largest models were all at or below baseline. Falsified.
- Training-set size. 249 targeted indoor negatives added and retrained. The primary metric,
  no-screen probe FP, moved 0.331 to 0.314, inside noise, and recall regressed with it
  (FN 0.187 to 0.262). Falsified, and net negative.

The bottleneck is in the data distribution and the input representation, not in capacity and
not in sample count. Two mechanisms, both with measurements behind them:

- Covariate shift. False triggers concentrate in built indoor environments: office 0.559,
  living room 0.52, meeting room 0.324, with outdoor and restaurant scenes near zero. Face
  count is negatively correlated with false triggers at −0.093, so the model is not firing at
  people, it is being fooled by rooms.
- Representation limit. At 96x96 greyscale, "bright indoor scene with a rectangular
  screen-like region" is a feature shared almost equally by the positive class (a screen with
  text) and the negative class (the switched-off screen, white wall or window in the same
  office). There is not enough resolution to separate a blank screen from a text-bearing one.

In one line: F1 around 0.77 is not a tuning failure, it is the ceiling at 96x96 greyscale on
the current data distribution. Moving past it means changing the input representation or
decomposing the task (section 6), not adding parameters or images.

## 1. What "old" and "new" refer to

The words carry two separate meanings in this project and they are easy to conflate.

| | Meaning A: methodology | Meaning B: the baseline in this comparison |
|---|---|---|
| Old | Before phase 1: single run, single seed, original un-deduplicated splits, argmax threshold at 0.5, no fixed probe | `baseline` = the old small architecture (8,16,32,64; 24.9k parameters) |
| New | After phase 1: 5 seeds, dedup with zero-leakage verification, fixed held-out probes, threshold fixed at 0.40, int8 deployment measurement | Five larger candidates A/B/C/D/E (74k–94.5k parameters) |

One clarification matters for reading section 3: `baseline` in the task1 report is not the old
model's old numbers. It is the old architecture re-run under the new protocol. The comparison
table therefore compares architectures under one strict protocol with no methodology
difference mixed in, which is what makes a conclusion possible at all.

### 1.1 Three concrete problems with the old method

These were confirmed by measurement, not assigned in hindsight.

| Problem | Evidence | Consequence |
|---|---|---|
| Data leakage | Perceptual hashing plus pixel confirmation found 132 near-duplicate pairs across 88 groups. 49 groups spanned splits and 48 of those were purely positive-class. Root cause: `download_images.py` fetched images under multiple keywords, so the same Pexels photo was downloaded repeatedly into different subclass folders (83 of 88 groups share a Pexels ID) | Inflated positive-class recall, suppressed apparent FN |
| Single-seed luck | Re-run across 5 seeds, recall drifts between 0.66 and 0.85 (std 0.068) | A single run is not reproducible |
| No held-out real-photo probe | Only a test split, no independent "people without screen" or "people plus text screen" photo set | Covariate-shift errors are invisible |

The cost, tallied: after dedup, FN at seed 42 goes from 0.277 to 0.337 and FP rate from 0.130
to roughly 0.197. True 5-seed F1 is 0.756 ± 0.024 at threshold 0.5 under keras measurement.
The old conclusion that FN and FP were both below 0.30 was an artefact of leakage and luck
together, and has been withdrawn.

### 1.2 The fixed protocol every experiment since has followed

- Five seeds, `[42, 1, 7, 123, 2024]`, reported as mean ± std, with no seed selection.
- Threshold fixed at 0.40, never adjusted to fit a result. The threshold trade-off itself is
  a separate question, in `docs/threshold-tradeoff.md`.
- Leakage control: `dedup_resplit.py` deduplicates by connected component, then
  `check_leakage.py` must report 0 cross-split and 0 within-split duplicates, and
  `guard_probe_overlap.py` must report 0 overlap between the training pool and the probes.
- Fixed held-out probes of real photographs, never used in training, so before/after
  comparisons stay valid: `noscreen` 235 images (ground truth: do not record),
  `person_screen` 181 images (ground truth: record), `indoor_env_v2` 64 images (added in
  task2b).
- int8 deployment measurement: every probe metric is taken on the exported int8 tflite with
  deployment preprocessing (cv2 greyscale, resize to 96 with INTER_AREA, quantise), not on
  training-time float numbers. Quantisation cost across all candidates is within 0.005 F1.
- Budget gate: before a candidate is trained, its ESP32 budget is checked statically —
  concurrent activations under 256 KB, int8 weights under 100 KB, operators restricted to the
  TFLite Micro whitelist.

## 2. How the three experiments were run

| | Experiment 1: capacity sweep (task1) | Experiment 2: add people negatives (task2) | Experiment 3: targeted indoor negatives (task2b) |
|---|---|---|---|
| Hypothesis | Not enough capacity | Not enough people-negative coverage (covariate shift) | Not enough indoor *environment* negative coverage |
| Action | Fix the data, sweep 6 architectures (baseline + A/B/C/D/E), 24.9k to 94.5k parameters | Add 558 negatives (street, market, commute, home — deliberately non-office people) and 130 positives | On top of task2, add 249 more "empty indoor environment / blank screen" negatives |
| Variable | Architecture only | Data only | Data only |
| Scale | 6 candidates x 5 seeds = 30 training runs | 5 seeds plus a single-seed deployment model | 5 seeds, architecture fixed at C_wide_uniform |
| Outcome | Plateau; only C is marginally better (+0.021, about 1σ) | Worked. People-driven FP 51% to 24%, Pareto-dominant | Primary objective missed, and recall regressed |

Experiment 2 is the only one that worked, and it does show that the data direction is
effective on *some* distributions: from the old v4 to task2, people-driven false triggers were
more than halved at matched recall. Experiment 3 took one more step in that same direction and
did not get there. Section 3.2 explains why.

## 3. The elimination chain

### 3.1 Capacity is not the bottleneck

Experiment 1, 3.8x parameter sweep.

| Candidate | Parameters | test F1 (5-seed) | ΔF1 vs baseline | noscreen FP (lower is better) | screen recall (higher is better) |
|---|---:|---|---:|---|---|
| baseline | 24.9k | 0.7487 ± 0.0210 | — | 0.361 ± 0.053 | 0.625 ± 0.058 |
| A_wide_late | 85.3k | 0.7421 ± 0.0232 | −0.0066 | 0.391 ± 0.097 | 0.645 ± 0.095 |
| B_deep_stack | 74.3k | 0.7392 ± 0.0358 | −0.0095 | 0.346 ± 0.081 | 0.600 ± 0.083 |
| C_wide_uniform | 60.9k | 0.7694 ± 0.0148 | +0.0207 | 0.331 ± 0.091 | 0.582 ± 0.095 |
| D_five_stage | 80.6k | 0.7359 ± 0.0199 | −0.0128 | 0.375 ± 0.120 | 0.600 ± 0.124 |
| E_combo | 94.5k | 0.7357 ± 0.0208 | −0.0130 | 0.392 ± 0.108 | 0.635 ± 0.099 |

Three things to read out of that table.

Test F1 lands between 0.736 and 0.769 for every candidate. If capacity were the constraint,
scaling monotonically should pay off monotonically; instead there is a plateau plus noise, and
the three largest models (A at 85k, D at 81k, E at 95k) are all at or below baseline.

Most of the differences are the size of seed variance. ΔF1 ranges from −0.013 to +0.021 while
single-seed std runs 0.015 to 0.036, so the gaps are hard to separate from noise statistically.
The one positive signal, C, is about 1σ. We do not treat that as evidence that complexity
solved anything, only as the best configuration available inside the budget.

Capacity does not fix false triggering. Probe FP is stuck at 0.33 to 0.39 at every capacity
with no downward trend as parameters grow.

There is a useful side result for the hardware track: all six candidates cleared the budget.
Fully int8 with zero float32 tensors, operators entirely within the TFLite Micro whitelist
(only 5 distinct operators), int8 weights at most 92.3 KB against a 100 KB limit, concurrent
activations at most 180 KB against 256 KB. The accuracy ceiling is not being imposed by the
hardware budget. There is headroom, and no way to spend it usefully.

### 3.2 More data did not lift it either

Experiment 3, 249 added negatives.

| Metric | task1 (before) | task2b (after) | Δ | Comparable? | Reading |
|---|---|---|---:|---|---|
| noscreen FP, lower is better (primary) | 0.331 ± 0.091 | 0.314 ± 0.093 | −0.017 | Fixed probe, comparable | Well inside ±0.09 noise. Objective missed |
| screen recall, higher is better | 0.582 ± 0.095 | 0.521 ± 0.118 | −0.061 | Fixed probe, comparable | Recall regressed |
| indoor_env_v2 FP, lower is better | 0.328 [1] | 0.250 [1] | −0.078 | Fixed probe, comparable | The one improvement |
| test F1 | 0.769 ± 0.015 | 0.704 ± 0.034 | −0.066 | Re-split, not strictly comparable | Worse |
| test FN, lower is better | 0.187 ± 0.060 | 0.262 ± 0.056 | +0.074 | As above | Worse: more missed captures |
| test recall, higher is better | 0.813 ± 0.060 | 0.738 ± 0.056 | −0.074 | As above | Worse |

[1] A fair single-model comparison, seed 42 against seed 42. The indoor_env_v2 probe was only
built during task2b, so task1 has no 5-seed baseline on it.

Why one probe improved and the other did not move:

- `indoor_env_v2` is *empty* indoor environments — lobbies, libraries, reception desks, no
  people. That is the same distribution as the negatives we added (empty rooms, blank
  screens), so the model learned it: FP 0.328 to 0.250.
- `noscreen` is *people* in indoor scenes — colleagues talking in an office, meeting rooms,
  family living rooms. Those frames contain people and are a different distribution from
  "empty room" negatives, so FP did not move.

We fixed half the distribution, and not the half that was failing. On top of that, the 249
extra negatives skewed the class balance further negative (1387 to 1636), pushing the decision
boundary toward "do not record", which raised FN and lowered recall without buying any
reduction in noscreen FP. Net result: not worth it. The conclusion was to keep task1's
C_wide_uniform as the current gatekeeper.

### 3.3 The real bottleneck is distribution plus representation

All 59 false positives on the `noscreen` probe were examined individually
(C_wide_uniform int8 at threshold 0.40, FP = 59/235 = 0.251).

**False triggers split cleanly by scene. The model is fooled by rooms, not by people.**

| Scene | n | FP rate | Mean face count |
|---|---:|---:|---:|
| office_colleagues_conversation | 34 | 0.559 | 0.41 |
| family_home_living_room | 25 | 0.52 | 0.08 |
| people_meeting_room_talking | 34 | 0.324 | 0.29 |
| coworkers_standing_meeting | 29 | 0.241 | 0.59 |
| group_friends_indoor_candid | 29 | 0.207 | 0.90 |
| people_street_candid | 25 | 0.08 | 0.20 |
| friends_cafe_group | 29 | 0.034 | 0.52 |
| people_restaurant_dining | 30 | 0.0 | 0.23 |

The high-FP cluster is offices, meeting rooms and living rooms, which are exactly the rooms
that contain monitors, whiteboards, projectors and documents. Outdoor street scenes and dining
are close to zero.

**How false positives differ from correct rejections.**

| Dimension | FP (n=59) | Correct rejection (n=176) | Difference |
|---|---:|---:|---:|
| Brightness | 0.601 | 0.415 | +0.186 (largest single-dimension gap) |
| Screen-like rectangle hit rate | 0.237 | 0.119 | +0.118 (roughly 2x) |
| Contrast | 0.253 | 0.231 | +0.022 |
| Face-count proxy | 0.339 | 0.432 | −0.093 (negatively correlated) |

The negative correlation on face count directly falsifies the intuitive "more people means
more false triggers" assumption. Several of the highest-scoring false positives contain zero
detected frontal faces. Scattering another batch of people photographs at the problem will not
help much; what needs covering is that class of indoor environment itself, plus screen-like
surfaces with no text on them.

**Where this lands.** At 96x96 greyscale the trigger signal the gatekeeper has learned is
"bright indoor scene with a rectangular screen-like region", and that signal is shared almost
equally between the positive class (a screen with text in an office) and the negative class
(the switched-off screen, window, picture frame or white wall in the same office). A blank
screen and a text-bearing screen are close to inseparable at that resolution, because the
letterforms have already been destroyed by the downscale. Adding more negatives of the same
kind therefore either does nothing (noscreen did not move) or costs recall (FN rose) — which
is exactly what experiment 3 produced.

## 4. Conclusion

Two of the most intuitive hypotheses were eliminated by experiment, which located the
bottleneck in a third place.

| Hypothesis | How it was tested | Result | Cost |
|---|---|---|---|
| H1: not enough capacity | 6 architectures x 5 seeds, 3.8x parameter sweep, data held fixed | Falsified. F1 plateau 0.736–0.769, the three largest models at or below baseline | 30 training runs |
| H2: not enough data | 249 targeted negatives, retrained, 5 seeds, before/after on fixed probes | Falsified, and net negative. Primary metric −0.017 inside noise, FN +0.074 | One data collection round plus 5 training runs |
| H3: distribution and representation | All 59 false positives examined by scene, brightness, screen-like geometry and face count | Best current explanation. False triggers driven by built indoor environments; 96x96 greyscale cannot separate a blank screen from a text-bearing one | — |

The elimination is itself the result. Without it the default behaviour would have been to tune
parameters and download more images indefinitely. What can now be stated is a negative
conclusion that is actionable: the next unit of effort should not go into parameter count or
image count.

One more number worth stating plainly. With H1 and H2 both eliminated, the gatekeeper's actual
working point on the fixed probes is noscreen FP around 0.33 and person-plus-screen recall
around 0.58, at threshold 0.40, int8, across 5 seeds. That is not a good pair of numbers, but
it is reproducible, leakage-free, and not seed-picked.

## 5. What has not been ruled out

The scope of the elimination should not be overstated. None of the following has been
falsified, and each remains open.

1. Higher input resolution (128 or 160) has not been tried. It is the most direct test of the
   H3 representation hypothesis, but it breaks the ESP32 budget because concurrent activations
   would exceed 256 KB. It was skipped for budget reasons, not because it looks unpromising.
   If H3 is to be tested, this is the first experiment to run.
2. Distribution-matched negatives have not been tried — people, plus office/meeting/living
   room, plus a screen with no text. task2b added empty rooms, which is not this. The risk is
   known: such images very easily contain a readable screen, which would make them positives,
   so QC cost is high. And once they enter training, the `noscreen` probe is demoted from a
   held-out generalisation test to an in-distribution one, meaning part of any FP reduction
   would be "trained on this kind of image" rather than real generalisation. That is a
   methodological trade-off, not a free win.
3. A two-stage cascade has not been tried — a coarse "is there a screen-like region" filter
   followed by a very cheap "is there text present" discriminator.
4. No architecture-family exploration. Only width, depth and stage count within one CNN family
   were swept; depthwise-separable convolutions, attention and similar were not tried.
5. The dataset is still small (2440–2689 images in the deduplicated training pool). The
   accurate statement of what H2 falsified is "adding 249 more images in that direction did
   not help", not "the dataset is large enough".
6. The probes are posed Pexels photographs, not real frames grabbed from a head-mounted
   camera. Numbers on real frames need to be re-measured with a Pi and a camera, which is on
   the hardware to-do list.

## 6. Options for what to do next

Not pre-selected; this is the decision to be made.

| Option | What it directly tests | Cost | Risk |
|---|---|---|---|
| (a) Raise resolution to 128/160 | H3, the representation hypothesis: can a blank screen be told from a text-bearing one | Medium (retrain plus re-estimate budget) | Breaks the ESP32 budget, so the hardware target would need redefining. No pressure on a Pi |
| (b) Distribution-matched negatives (people, indoor, no text screen) | Whether H2's problem was the distribution rather than the amount | High (manual QC) | Demotes the probe to an in-distribution test; risk of contaminating the negative class |
| (c) Two-stage cascade (screen-like region, then text presence) | Splitting one hard binary decision into two easy ones | Medium-high (new model plus new labelling) | More on-device compute and complexity |
| (d) Accept the current FP floor and work the trade-off curve instead | Not chasing F1; deliver the best operating point on power versus missed captures | Low (threshold sweep data already exists) | Does not fix accuracy, but it is exactly the Pareto curve advisor question 3 asks for |

Recommendation: run (a) and (d) in parallel. (a) because it is the only direct test of H3, and
because the deployment target has already relaxed from ESP32 to Raspberry Pi 5 in the
`hardware/` line, which makes the resolution budget much less binding. (d) because it delivers
value without depending on an accuracy improvement, and it is the answer to advisor question 3
(see `docs/pareto-method.md`). (b) has the worst cost-to-benefit ratio — experiment 3 already
paid that tuition once.

## Appendix: commands to reproduce

```bash
# Capacity sweep: full 5-seed pipeline for one candidate
# (train, test at 0.40, int8 export with whitelist and ΔF1 checks, both probes)
PYTHONPATH=scripts .venv/bin/python scripts/run_candidate.py \
    --tag C_wide_uniform --channels 16,32,64,64 --convs 1

# Per-image false-trigger diagnosis (read-only, no retraining)
PYTHONPATH=scripts .venv/bin/python scripts/task2b_fp_diagnosis.py

# Leakage verification
.venv/bin/python scripts/check_leakage.py --manifest data/processed/manifest_dedup.csv
.venv/bin/python scripts/guard_probe_overlap.py
```

Provenance: every metric in this report was measured on this machine, from
`docs/results/task1_results/*.json`, `docs/results/task2b_results/*` and the per-probe
`probe_fp_summary.json`. None of it is a design-time estimate or a figure quoted from
literature. No 5-seed number was seed-picked, and every comparison that is not strictly
like-for-like is marked as such in its table.
