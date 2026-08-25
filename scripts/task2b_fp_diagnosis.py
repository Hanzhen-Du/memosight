#!/usr/bin/env python3
"""Task 2b, phase one. Diagnose what the false positives have in common: C_wide_uniform int8
on the noscreen probe at threshold 0.40.

Read-only, no retraining, and low memory. Images are processed one at a time; each
full-resolution image is downsampled and released immediately, and the full-resolution batch is
never held in memory.

For every image on the noscreen probe it extracts:
  - the prediction score, under int8 deployment conditions (cv2 greyscale, resize to 96 with
    INTER_AREA, quantise), matching probe_fp_test;
  - the scene label, taken from the subdirectory name;
  - brightness as the mean and contrast as the standard deviation, over the 96 greyscale in
    [0,1];
  - a face-count proxy (Haar frontal face, run on greyscale at 320px or less). This is only a
    rough signal for how many people or frontal faces are present;
  - a screen-like rectangle signal, found by looking for large bright quadrilateral contours in
    greyscale at 256px or less. There is no screen anywhere on the noscreen probe, so a hit is
    a misleading geometric cue.

False positives are cut at 0.40 (score >= 0.40). It exports the FP list with scores, an
aggregation by scene, and a dimension-by-dimension comparison of false positives against
correct rejections.

Writes docs/false-positive-diagnosis.md and
docs/results/task2b_results/noscreen_fp_per_image.csv.

Leakage control: the probe is used for evaluation only and never enters training; this script
only reads. It still checks by Pexels ID that the probe does not collide with the training
pool, which should be zero.

"""
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

from probe_fp_test import (collect_images, manifest_id_set, photo_ids,
                           load_int8, int8_predict_one, INPUT_SIZE)

MODEL = Path("models/task1_candidates/gatekeeper_task1_C_wide_uniform_int8.tflite")
PROBE = Path("data/probe_person_noscreen")
LEAK_MANIFEST = Path("data/processed/manifest.csv")
THR = 0.40
OUT_MD = Path("docs/false-positive-diagnosis.md")
OUT_CSV = Path("docs/results/task2b_results/noscreen_fp_per_image.csv")

_FACE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def scene_of(path: Path) -> str:
    """The scene label is the probe subdirectory name; each subdirectory under noscreen is one
    kind of scene."""
    rel = path.relative_to(PROBE)
    return rel.parts[0] if len(rel.parts) > 1 else "(root)"


def _resize_max(gray: np.ndarray, max_side: int) -> np.ndarray:
    h, w = gray.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return gray
    return cv2.resize(gray, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def face_count(gray_small: np.ndarray) -> int:
    faces = _FACE.detectMultiScale(gray_small, scaleFactor=1.1,
                                   minNeighbors=5, minSize=(24, 24))
    return len(faces)


def screen_like_rect(gray_small: np.ndarray) -> int:
    """Roughly detect large bright quadrilateral regions: windows, frames, closed laptops,
    doorways, all misleading geometric cues. Returns 1 on a hit.

    Heuristic and noisy: threshold, find contours, keep convex 4-corner polygons from
    approxPolyDP with an area of at least 8% of the frame and brightness above the global
    median. It is a group-level signal only and is not evidence about any single image.
    """
    h, w = gray_small.shape[:2]
    area = h * w
    med = float(np.median(gray_small))
    _, th = cv2.threshold(gray_small, max(med, 110), 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.08 * area:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return 1
    return 0


def main() -> int:
    assert MODEL.exists(), f"model does not exist: {MODEL}"
    interp = load_int8(MODEL)
    mids = manifest_id_set(LEAK_MANIFEST)

    imgs = collect_images(PROBE)
    rows, leaked = [], 0
    for p in imgs:
        if photo_ids(p.stem) & mids:
            leaked += 1
            continue
        full = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if full is None:
            continue
        g96 = cv2.resize(full, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        g_face = _resize_max(full, 320)
        g_rect = _resize_max(full, 256)
        del full  # release the full-resolution image at once; memory holds one image briefly
        score = float(int8_predict_one(interp, g96))
        rows.append({
            "file": str(p.relative_to(PROBE)),
            "scene": scene_of(p),
            "score": round(score, 4),
            "is_fp": int(score >= THR),
            "brightness": round(float(g96.mean()) / 255.0, 4),
            "contrast": round(float(g96.std()) / 255.0, 4),
            "faces": face_count(g_face),
            "screen_rect": screen_like_rect(g_rect),
        })

    n = len(rows)
    fps = [r for r in rows if r["is_fp"]]
    crs = [r for r in rows if not r["is_fp"]]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["score"]))

    # Aggregate by scene
    scenes = sorted({r["scene"] for r in rows})
    per_scene = []
    for s in scenes:
        sr = [r for r in rows if r["scene"] == s]
        sf = [r for r in sr if r["is_fp"]]
        per_scene.append({
            "scene": s, "n": len(sr), "fp": len(sf),
            "fp_rate": round(len(sf) / len(sr), 3) if sr else 0.0,
            "mean_score": round(float(np.mean([r["score"] for r in sr])), 3),
            "faces_mean": round(float(np.mean([r["faces"] for r in sr])), 2),
        })
    per_scene.sort(key=lambda d: (-d["fp_rate"], -d["n"]))

    def mean(key, sub):
        return round(float(np.mean([r[key] for r in sub])), 3) if sub else 0.0

    # -- write the markdown report --
    L = []
    L.append("# Per-image diagnosis of no-screen probe false positives\n")
    L.append(f"Model: `{MODEL}`, the task1 winner C_wide_uniform in int8, at threshold {THR}. "
             f"Measured under int8 deployment preprocessing: cv2 greyscale, resize to 96 with "
             f"INTER_AREA, quantise.\n")
    L.append(f"Probe: noscreen, {n} images, with {leaked} removed by the Pexels-ID leakage "
             f"check. The probe is used for evaluation only and never enters any training "
             f"set.\n")
    L.append("\n## Overview\n")
    L.append(f"- False positives, judged as record at score >= {THR}: "
             f"{len(fps)}/{n} = {len(fps)/n:.3f}\n")
    L.append(f"- Correct rejections: {len(crs)}/{n}\n")
    L.append(f"- Score distribution: min {min(r['score'] for r in rows):.3f} / "
             f"median {np.median([r['score'] for r in rows]):.3f} / "
             f"max {max(r['score'] for r in rows):.3f}\n")

    L.append("\n## By scene, sorted by false-positive rate\n")
    L.append("| Scene (subdirectory) | n | FP | FP rate | Mean score | Mean face count |\n"
             "|---|---:|---:|---:|---:|---:|\n")
    for d in per_scene:
        L.append(f"| {d['scene']} | {d['n']} | {d['fp']} | {d['fp_rate']} | "
                 f"{d['mean_score']} | {d['faces_mean']} |\n")

    L.append("\n## False positives against correct rejections, means\n")
    L.append("| Dimension | FP (n={}) | Correct rejection (n={}) | Difference |\n|---|---:|---:|---:|\n"
             .format(len(fps), len(crs)))
    for key, label in [("brightness", "Brightness"), ("contrast", "Contrast"),
                       ("faces", "Face-count proxy"),
                       ("screen_rect", "Screen-like rectangle hit rate")]:
        a, b = mean(key, fps), mean(key, crs)
        L.append(f"| {label} | {a} | {b} | {round(a-b,3):+} |\n")

    L.append(f"\n## Full false-positive list, all {len(fps)}, by descending score\n")
    L.append("| # | score | Scene | File | Faces | Brightness | Screen-like |\n"
             "|---:|---:|---|---|---:|---:|---:|\n")
    for i, r in enumerate(sorted(fps, key=lambda r: -r["score"]), 1):
        L.append(f"| {i} | {r['score']} | {r['scene']} | {r['file']} | "
                 f"{r['faces']} | {r['brightness']} | {r['screen_rect']} |\n")

    L.append(f"\n## Near-threshold correct rejections "
             f"({THR} > score >= {THR-0.08:.2f})\n")
    near = sorted([r for r in crs if r["score"] >= THR - 0.08], key=lambda r: -r["score"])
    L.append(f"{len(near)} images. These are the borderline cases a little more of the same kind "
             "of negative would most plausibly push down.\n\n")
    L.append("| score | Scene | File |\n|---:|---|---|\n")
    for r in near:
        L.append(f"| {r['score']} | {r['scene']} | {r['file']} |\n")

    OUT_MD.write_text("".join(L), encoding="utf-8")

    print(f"[diag] n={n} leak={leaked} FP={len(fps)} ({len(fps)/n:.3f}) → {OUT_MD}")
    print("[diag] per-scene FP rate:", {d["scene"]: d["fp_rate"] for d in per_scene})
    print(f"[diag] CSV → {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
