# Probe false-trigger audit
> Measures the gatekeeper's false-trigger rate directly on real "people present, no screen text" scenes. By definition every probe image should be judged do-not-record, so any record is a false positive. Diagnostic only: nothing was retrained and no data was changed.

## Method
- Probe directory: `data/probe_person_screen/` (gitignored). Preprocessing matches deployment: cv2 greyscale, resize to 96 with INTER_AREA, divide by 255, then quantise for int8.
- Leakage control: a double check by Pexels ID from the filename and by perceptual hash (pHash within threshold and pixel correlation above threshold, reusing the same criteria as `check_leakage`), removing any probe image that collides with or near-duplicates train, val or test.
- Two gatekeepers are scored: `keras (float)` and `int8 (.tflite)`, the latter with a self-contained int8 runtime reproducing the deployment preprocessing in `hardware/infer.py`. Deployment threshold 0.55; 0.5, 0.7 and 0.9 are also listed.

## 1. Leakage check
- Probe images: 181. Removed as leaked: 0 (Pexels ID collisions 0, perceptual near-duplicates 0). Clean probe set: 181 images, used for the FP measurement.

## 2. False-trigger rate on the clean probe set (ground truth is do-not-record throughout)

**keras(float)**

| Threshold | Judged record / total | FP rate |
|---|---|---|
| 0.4 | 144/181 | **79.6%** |
| 0.45 | 135/181 | **74.6%** |
| 0.5 | 129/181 | **71.3%** |
| 0.55 | 119/181 | **65.8%**  (deployment) |

**int8(.tflite)**

| Threshold | Judged record / total | FP rate |
|---|---|---|
| 0.4 | 150/181 | **82.9%** |
| 0.45 | 141/181 | **77.9%** |
| 0.5 | 133/181 | **73.5%** |
| 0.55 | 125/181 | **69.1%**  (deployment) |

## 3. False-positive cases for manual review
- Montage: `data/processed/probe_personscreen_audit_v4/montage_fp_cases.png` (red text is the probability of record)
- Grad-CAM: not produced (--no-gradcam).

## 4. Reading
- At the int8 deployment threshold of 0.55, FP is **69.1%**, which is high. Combined with the audit gap of about 0, this looks like covariate or context shift: the studio-portrait negatives do not cover people in cluttered real scenes. The fix is more scene diversity among the people negatives rather than simply more of them. Check the Grad-CAM to see whether attention locks onto people.
- If keras and int8 FP differ noticeably, quantisation has moved the operating point and the deployment threshold must be re-calibrated against int8.
