# Probe false-trigger audit
> Measures the gatekeeper's false-trigger rate directly on real "people present, no screen text" scenes. By definition every probe image should be judged do-not-record, so any record is a false positive. Diagnostic only: nothing was retrained and no data was changed.

## Method
- Probe directory: `data/probe_person_noscreen/` (gitignored). Preprocessing matches deployment: cv2 greyscale, resize to 96 with INTER_AREA, divide by 255, then quantise for int8.
- Leakage control: a double check by Pexels ID from the filename and by perceptual hash (pHash within threshold and pixel correlation above threshold, reusing the same criteria as `check_leakage`), removing any probe image that collides with or near-duplicates train, val or test.
- Two gatekeepers are scored: `keras (float)` and `int8 (.tflite)`, the latter with a self-contained int8 runtime reproducing the deployment preprocessing in `hardware/infer.py`. Deployment threshold 0.55; 0.5, 0.7 and 0.9 are also listed.

## 1. Leakage check
- Probe images: 235. Removed as leaked: 0 (Pexels ID collisions 0, perceptual near-duplicates 0). Clean probe set: 235 images, used for the FP measurement.

## 2. False-trigger rate on the clean probe set (ground truth is do-not-record throughout)

**keras(float)**

| Threshold | Judged record / total | FP rate |
|---|---|---|
| 0.4 | 139/235 | **59.2%** |
| 0.45 | 129/235 | **54.9%** |
| 0.5 | 120/235 | **51.1%** |
| 0.55 | 117/235 | **49.8%**  (deployment) |
| 0.6 | 105/235 | **44.7%** |
| 0.65 | 93/235 | **39.6%** |
| 0.7 | 80/235 | **34.0%** |

**int8(.tflite)**

| Threshold | Judged record / total | FP rate |
|---|---|---|
| 0.4 | 143/235 | **60.9%** |
| 0.45 | 135/235 | **57.5%** |
| 0.5 | 126/235 | **53.6%** |
| 0.55 | 120/235 | **51.1%**  (deployment) |
| 0.6 | 112/235 | **47.7%** |
| 0.65 | 102/235 | **43.4%** |
| 0.7 | 87/235 | **37.0%** |

## 3. False-positive cases for manual review
- Montage: `data/processed/probe_noscreen_audit_v4/montage_fp_cases.png` (red text is the probability of record)
- Grad-CAM: not produced (--no-gradcam).

## 4. Reading
- At the int8 deployment threshold of 0.55, FP is **51.1%**, which is high. Combined with the audit gap of about 0, this looks like covariate or context shift: the studio-portrait negatives do not cover people in cluttered real scenes. The fix is more scene diversity among the people negatives rather than simply more of them. Check the Grad-CAM to see whether attention locks onto people.
- If keras and int8 FP differ noticeably, quantisation has moved the operating point and the deployment threshold must be re-calibrated against int8.
