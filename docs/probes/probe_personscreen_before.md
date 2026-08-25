# Probe false-trigger audit
> Measures the gatekeeper's false-trigger rate directly on real "people present, no screen text" scenes. By definition every probe image should be judged do-not-record, so any record is a false positive. Diagnostic only: nothing was retrained and no data was changed.

## Method
- Probe directory: `data/probe_person_screen/` (gitignored). Preprocessing matches deployment: cv2 greyscale, resize to 96 with INTER_AREA, divide by 255, then quantise for int8.
- Leakage control: a double check by Pexels ID from the filename and by perceptual hash (pHash within threshold and pixel correlation above threshold, reusing the same criteria as `check_leakage`), removing any probe image that collides with or near-duplicates train, val or test.
- Two gatekeepers are scored: `keras (float)` and `int8 (.tflite)`, the latter with a self-contained int8 runtime reproducing the deployment preprocessing in `hardware/infer.py`. Deployment threshold 0.55; 0.5, 0.7 and 0.9 are also listed.

## 1. Leakage check
- Probe images: 51. Removed as leaked: 0 (Pexels ID collisions 0, perceptual near-duplicates 0). Clean probe set: 51 images, used for the FP measurement.

## 2. False-trigger rate on the clean probe set (ground truth is do-not-record throughout)

**keras(float)**

| Threshold | Judged record / total | FP rate |
|---|---|---|
| 0.5 | 32/51 | **62.7%** |
| 0.55 | 30/51 | **58.8%**  (deployment) |
| 0.7 | 23/51 | **45.1%** |
| 0.9 | 9/51 | **17.6%** |

**int8(.tflite)**

| Threshold | Judged record / total | FP rate |
|---|---|---|
| 0.5 | 34/51 | **66.7%** |
| 0.55 | 31/51 | **60.8%**  (deployment) |
| 0.7 | 23/51 | **45.1%** |
| 0.9 | 10/51 | **19.6%** |

## 3. False-positive cases for manual review
- Montage: `data/processed/probe_personscreen_before/montage_fp_cases.png` (red text is the probability of record)
- Grad-CAM: not produced (--no-gradcam).

## 4. Reading
- Caveat: only 51 clean probe images, so the FP rate is statistically noisy. Expand to around 200 before drawing conclusions.
- At the int8 deployment threshold of 0.55, FP is **60.8%**, which is high. Combined with the audit gap of about 0, this looks like covariate or context shift: the studio-portrait negatives do not cover people in cluttered real scenes. The fix is more scene diversity among the people negatives rather than simply more of them. Check the Grad-CAM to see whether attention locks onto people.
- If keras and int8 FP differ noticeably, quantisation has moved the operating point and the deployment threshold must be re-calibrated against int8.
