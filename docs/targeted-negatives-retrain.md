# Task 2b retrain: targeted indoor negatives, before and after

Completed 2026-06-26. Architecture C_wide_uniform (16,32,64,64), 5 seeds
`[42,1,7,123,2024]`, threshold 0.40, int8 deployment measurement.

## 0. Conclusion

The primary objective was missed, and it cost recall and FN. This model should not be
promoted.

No-screen FP, the original problem of false triggering on people in indoor scenes, did not
come down: 0.331 to 0.314, a change of −0.017 that sits well inside ±0.09 noise.

The cost is unambiguous. Test FN went from 0.187 to 0.262, so more missed captures. Test recall
fell from 0.813 to 0.738, person-plus-screen recall from 0.582 to 0.521, and test F1 from 0.769
to 0.704.

There is one improvement. On a newly held-out generalisation probe of empty indoor
environments, `indoor_env_v2`, FP went from 0.328 to 0.250 (seed 42 against seed 42, −0.078).
The new negatives were learned on their own distribution — empty rooms and blank screens — but
that did not transfer to the people-plus-indoor distribution that `noscreen` measures.

The diagnosis: what was added were empty-room and blank screen-like-surface negatives, while
the residual FP on `noscreen` comes from frames containing people in indoor environments.
Empty-room negatives improved empty-room recognition and did nothing for "still fooled by the
room when a person is present". On top of that, the 249 extra negatives skewed the classes
further negative (1387 to 1636) and pushed the decision boundary toward "do not record", so FN
rose and recall fell without any reduction in no-screen FP. Net result: not worth it.

## 1. Data and process, verified

Merge: `prepare_dataset` rebuilt the manifest at 2998 rows, `dedup_resplit` reproduced the
phase 3.4-B boundary narrowing by removing 198 rows across 5 ambiguous subclasses, and full
perceptual dedup gave 2689 images (1636 negative, 1053 positive). Net +249 negatives, positives
unchanged, and the new negatives introduced no new near-duplicates.

Leakage: `check_leakage` on `manifest_dedup` reports 0 cross-split and 0 within-split
duplicates; the guard reports zero overlap between training and all three probes.

All three probes are fixed and held out (noscreen 235, person_screen 181, indoor_env_v2 64), so
before and after are comparable on them. Test and val changed distribution because of the
re-split and the new negatives entering, so those are not strictly comparable and are treated
as reference only. The probes carry the argument.

## 2. Before and after

5-seed mean ± std.

| Metric | task1 (before) | task2b (after) | Δ | Comparable? | Reading |
|---|---|---|---:|---|---|
| noscreen_fp, lower better | 0.331 ± 0.091 | 0.314 ± 0.093 | −0.017 | Fixed probe, comparable | Inside noise. Primary objective missed |
| screen_recall, higher better | 0.582 ± 0.095 | 0.521 ± 0.118 | −0.061 | Fixed probe, comparable | Recall regressed |
| indoor_env_fp, lower better | 0.328 [1] | 0.244 ± 0.066 | −0.08 | Fixed probe, comparable | The one improvement, on the generalisation probe |
| test_f1, higher better | 0.769 ± 0.015 | 0.704 ± 0.034 | −0.066 | Re-split, not strictly comparable | Worse, partly because the test set itself got harder |
| test_fn, lower better | 0.187 ± 0.060 | 0.262 ± 0.056 | +0.074 | As above | Worse: more missed captures |
| test_recall, higher better | 0.813 ± 0.060 | 0.738 ± 0.056 | −0.074 | As above | Worse |
| test_fp, lower better | 0.226 ± 0.054 | 0.234 ± 0.074 | +0.008 | As above | Flat |
| val_f1 | 0.770 ± 0.020 | 0.719 ± 0.018 | −0.051 | New pool, not comparable | Also down on the new distribution |

[1] The `indoor_env_v2` probe was only built during task2b, so task1 has no 5-seed baseline on
it. The figure quoted is a fresh measurement of the two seed-42 deployment int8 models on the
same 64 probe images (task1 0.3281 against task2b 0.2500), which is a fair seed-42-to-seed-42
comparison. The 0.244 in the task2b column is the 5-seed mean, given for reference.

## 3. Why noscreen did not move while indoor_env_v2 did

The two probes are different distributions, and the new negatives only covered one of them.

- `indoor_env_v2` is *empty* indoor environments — lobbies, libraries, lounges, reception desks,
  no people. That is the same distribution as the added negatives, so the model learned it and
  FP fell from 0.328 to 0.250.
- `noscreen` is *people* in indoor scenes — colleagues talking in an office, people speaking in
  a meeting room, family living rooms. Those frames contain people, which is a different
  distribution from empty-room negatives, so FP did not move.

The phase-one diagnosis said false triggering was driven by the environment and inversely
correlated with faces, so pure environment negatives (empty rooms) were added. That turned out
to work on empty environments while the residual FP on `noscreen` occurs in indoor frames that
contain people. The model is still fooled by indoor and screen-like geometry in the background
when a person is present, and empty-room negatives do not teach it otherwise. Half the
distribution was fixed, and not the half that was failing.

This is consistent with, and sharpens, the task1 conclusion. At 96x96 greyscale the gatekeeper
treats "bright indoor scene plus rectangular screen-like region" as its trigger signal, and
that signal is shared almost equally between the positive class (an office with a text-bearing
screen) and the negative class (the same office with people and no text). Adding more negatives
of the same kind is either insufficient (noscreen did not move) or costs recall (FN rose). This
may be a limit of the representation and resolution rather than only of dataset size.

## 4. Recommendation

1. Do not promote the task2b model. It did not improve the primary objective, no-screen FP, on
   the fixed probe, and recall and FN both regressed clearly. Keep task1's
   `gatekeeper_task1_C_wide_uniform` as the current gatekeeper. The task1 model files are still
   in place and were not overwritten.
2. The training pool can be rolled back. The merge replaced `data/processed/manifest_dedup.csv`
   with the task2b pool of 2689. The old pool is backed up at
   `data/processed/_pretask2b_backup/` and can be copied back to reproduce the task1 protocol.
   Whether to roll back is still to be decided; nothing was reverted automatically.
3. If work on no-screen FP continues, the options are below. None of these were started.
   - (a) Match the distribution. Add negatives that are people, plus an indoor office, meeting
     room or living room, plus a screen with no text — rather than empty rooms — so they cover
     the case that actually fails. High risk: people plus an office very easily includes a
     readable screen, which would make the image a positive, and that contamination is harder
     to QC.
   - (b) Raise resolution to 128 or 160 so the model has a chance to distinguish a blank screen
     from one with text. This breaks the ESP32 budget and would need the constraints
     re-estimated.
   - (c) Two-stage. Have the gatekeeper coarsely detect a screen-like region, then run a very
     cheap text-presence discriminator, attacking the blank-screen versus text-screen boundary
     directly.
   - (d) Accept the FP floor and shift to tuning the threshold along the no-screen and
     power-versus-missed-capture curves instead.

## 5. Notes on honesty

The test and val regressions are partly caused by the re-split and by the new negatives making
the test set harder, not entirely by the model getting worse. That is why the argument above
rests on the fixed probes. Even on the fixed probes, though, the result is still negative:
noscreen flat, screen recall down, only indoor_env_v2 improved.

Single-seed variance is large — no-screen FP ranges from 0.179 to 0.430 across seeds — so every
headline conclusion here is based on the 5-seed mean and no seed was picked.

This round did not achieve its objective of reducing no-screen FP. Recording that plainly is
the point: an honest "it did not go down" is worth more than a flattering number obtained by
selecting a good seed.
