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
| 0.5 | 59/235 | **25.1%** |
| 0.55 | 53/235 | **22.6%**  (deployment) |
| 0.7 | 26/235 | **11.1%** |
| 0.9 | 2/235 | **0.9%** |

**int8(.tflite)**

| Threshold | Judged record / total | FP rate |
|---|---|---|
| 0.5 | 56/235 | **23.8%** |
| 0.55 | 51/235 | **21.7%**  (deployment) |
| 0.7 | 26/235 | **11.1%** |
| 0.9 | 2/235 | **0.9%** |

## 3. False-positive cases for manual review
- Montage: `data/processed/probe_fp_after/montage_fp_cases.png` (red text is the probability of record)
- Grad-CAM: not produced (--no-gradcam).

## 4. Reading
- At the int8 deployment threshold of 0.55, FP is **21.7%**, which is moderate. There is some context shift, so more diverse real-scene people negatives are worth adding. Also check whether the false-positive cases are mostly borderline images such as partially visible screens or reflections.
- If keras and int8 FP differ noticeably, quantisation has moved the operating point and the deployment threshold must be re-calibrated against int8.
