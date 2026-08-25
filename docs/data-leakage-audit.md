# Credibility audit of the gatekeeper's first numbers

Task 1, phase 1. 2026-06-17.

Method follows the pattern used from the start: record the failure honestly, diagnose it, then
fix it. Every metric below states which split it came from. Test is the deciding split; val is
for tuning.

From this phase onward all metrics are based on the deduplicated manifest
(`data/processed/manifest_dedup.csv` and `dedup_*.csv`). The original 1628-image splits are
withdrawn and are not used to decide anything.

## 0. Summary

| Question | Answer |
|---|---|
| Is there leakage between train, val and test? | Yes, and not a small amount. 49 duplicate groups span splits, and 48 of them are purely positive-class. |
| Is the 0.80 accuracy trustworthy? | Yes, and it is stable. After dedup and re-split, 5 seeds give 0.790 ± 0.014, essentially the original 0.81. |
| Are the first-version FN 0.277 and FP 0.130 trustworthy? | No. They are optimistic. Leakage sat almost entirely in the positive class, inflating recall and suppressing apparent FN. After dedup, the same seed 42 gives FN 0.337. |
| What is the real F1? | The first version never reported one. After dedup it is 0.756 ± 0.024, a genuine gap from the 0.85 target. |

The short version: accuracy of 0.80 is real and stable, but the claim that FN and FP were both
below 0.30 was produced by leakage plus one lucky seed. The real problem to solve is F1 around
0.76 with recall that moves between 0.66 and 0.85 depending on the split.

## 1. Leakage check

### 1.1 Method

`scripts/check_leakage.py` runs two passes, cheap first then precise.

1. Perceptual-hash coarse pass. Compute pHash (DCT) and dHash over all 1628 processed 96x96
   greyscale images, compare every pair by Hamming distance, and keep pairs with pHash
   distance at most 6 as candidates.
2. Pixel-level confirmation. For each candidate pair compute the Pearson correlation and
   normalised MSE over the 96x96 pixels. Only pairs correlating at 0.90 or above count as real
   near-duplicates, which filters out hash collisions.

Deduplication is `scripts/dedup_resplit.py`: build connected components (union-find) over the
confirmed near-duplicate pairs, keep one representative per group (lexicographically smallest
path, so it is deterministic and reproducible), then re-split 70/15/15 stratified by top-level
class, matching what `prepare_dataset.py` does.

pHash and dHash are implemented directly with numpy and `cv2.dct` rather than pulling in
`imagehash` — the library was not installed and the algorithms are standard.

### 1.2 Findings

- 134 pHash candidate pairs reduced to 132 confirmed near-duplicate pairs, of which 125 are
  near-identical at correlation 0.999 or above.
- By connected component: 88 duplicate groups covering 198 images. Largest group is 3 images
  (66 pairs, 22 triples).
- 49 groups span splits, which is the leakage. 48 of those are purely positive-class; one
  contains a negative.
- Cross-split duplicate pairs by split: test/train 33, train/val 27, test/val 4.

### 1.3 Root cause, confirmed

83 of the 88 duplicate groups share a Pexels image ID (the numeric segment in the filename).
`download_images.py` fetches images by keyword, so the same stock photo gets downloaded
repeatedly under different keywords and lands in different positive subclass folders — one
image ending up in `powerpoint_slide`, `classroom_projector_slides` and
`projector_screen_presentation` at once. The stratified split then splits by top-level class,
which scatters copies of one image across train, val and test, so the training set has seen
test images. Because the leakage sits in the positive class, it inflates positive-class
performance specifically.

### 1.4 Effect of dedup and re-split

| | Full | After dedup |
|---|---|---|
| Total | 1628 | 1518 (110 removed) |
| Positive | 751 | 644 (107 removed) |
| Negative | 877 | 874 (3 removed) |

The seed 42 re-split after dedup gives train 1061 / val 226 / test 231, with a positive-to-
negative ratio around 0.737 against the original 0.856 — the drop is because so many positive
duplicates were removed.

Re-checked: running the leakage check again over the three deduplicated splits reports 0
cross-split duplicate pairs and 0 within-split duplicates. The leakage is gone.

Deduplicated manifest: `data/processed/manifest_dedup.csv`. Candidate pair detail:
`docs/results/leakage_candidates.csv`.

## 2. Variance check: is 0.80 stable or was it one lucky run?

### 2.1 Method

`scripts/run_variance.py` takes the deduplicated manifest and re-splits and retrains under
each of 5 seeds `{42,1,7,123,2024}`. Leakage is eliminated first, because otherwise the
variance measurement is itself contaminated.

Configuration frozen at the best combination from the first-version fixes: `bn_momentum=0.9,
patience=15, start_from_epoch=20, epochs=80, augment=True, class_weight=balanced, lr=1e-3,
monitor=val_loss (restore best)`. Threshold 0.5 (argmax).

### 2.2 Results

Test split, positive class = 1 = record.

| seed | epochs | accuracy | F1 | recall | precision | FN rate | FP rate |
|---|---|---|---|---|---|---|---|
| 42 | 50 | 0.7922 | 0.7303 | 0.6633 | 0.8125 | 0.3367 | 0.1128 |
| 1 | 54 | 0.8009 | 0.7830 | 0.8469 | 0.7281 | 0.1531 | 0.2331 |
| 7 | 45 | 0.8052 | 0.7783 | 0.8061 | 0.7524 | 0.1939 | 0.1955 |
| 123 | 60 | 0.7835 | 0.7619 | 0.8163 | 0.7143 | 0.1837 | 0.2406 |
| 2024 | 59 | 0.7662 | 0.7245 | 0.7245 | 0.7245 | 0.2755 | 0.2030 |
| mean ± std | | 0.7896 ± 0.0139 | 0.7556 ± 0.0241 | 0.7714 ± 0.0675 | 0.7464 ± 0.0354 | 0.2286 ± 0.0675 | 0.1970 ± 0.0455 |

Machine-readable: `docs/results/variance_results.json`.

### 2.3 Reading

Accuracy at 0.79 ± 0.014 is stable. It was not one lucky run, and it agrees with the original
0.81 within noise. That number can be trusted.

Recall and FN vary a lot: recall spans 0.66 to 0.85 with std 0.068. The decision boundary
moves noticeably between splits, which is precisely why the phase 3 plan to sweep the
threshold is the right treatment — accuracy is stable while the precision/recall mix is not.

Leakage really was hiding the FN problem. After dedup, seed 42 gives FN 0.337, worse than the
0.277 reported under the same conditions with leakage, and FP rate rises from 0.130 to about
0.197. The first version's "FN and FP both below 0.30" came from contaminated data and that
seed's luck together.

Real F1 is about 0.756, roughly 0.09 short of the 0.85 target. That is a genuine gap to close,
not measurement error.

## 3. Consequences for phases 2 and 3

1. Switch the baseline. All later training and evaluation uses the deduplicated manifest and
   splits (`data/processed/manifest_dedup.csv` plus `dedup_*.csv`). The contaminated original
   `train/val/test.csv` is not to be used again.
2. Correct the target baseline. The real starting point is F1 0.756 and mean FN 0.229, not the
   first version's 0.277. Any claimed FN reduction in phase 3 must be measured against this.
3. Priority evidence. High recall variance with stable accuracy means the first phase 3 step,
   sweeping the decision threshold, is both free and well targeted, so it should come first.
   `scripts/evaluate.py` already supports `--pr-sweep` and re-scoring at an arbitrary
   threshold.
4. Fix the root cause in the data layer. To prevent this recurring, `download_images.py` should
   deduplicate by Pexels image ID at download time. Whether to change the download script is
   still to be decided.

## 4. Failure and diagnosis record

Failure: the first version reported FN 0.277 and treated it as passing. It was an optimistic
estimate produced by leakage plus single-seed luck.

Diagnosis: perceptual hashing plus pixel confirmation located the cross-split duplicate pairs;
connected components plus the 83-of-88 Pexels-ID sharing rate pinned the root cause to the same
image being downloaded under multiple keywords.

Fix: connected-component dedup keeping one image per group, plus a stratified re-split, took
leakage to zero. A 5-seed variance check then established a trustworthy baseline: accuracy
stable, F1 0.756, FN and recall unstable.

## Appendix: scripts added in this phase

None of these add a dependency; they reuse numpy, pandas, cv2 and tf.

- `scripts/check_leakage.py` — cross-split duplicate and leakage check (pHash + dHash coarse
  pass, pixel confirmation).
- `scripts/dedup_resplit.py` — connected-component dedup plus stratified re-split, with `build`
  and `split` subcommands.
- `scripts/evaluate.py` — full metrics in one command (accuracy, precision, recall, F1,
  confusion matrix, FN, FP), at any threshold, with `--pr-sweep`.
- `scripts/run_variance.py` — the 5-seed variance harness.
