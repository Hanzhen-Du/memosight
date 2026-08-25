#!/usr/bin/env python3
"""Probe false-trigger test: measure the gatekeeper's false-positive rate directly on real
"people present, no screen text" scenes.

Background from the person-bias audit: at the dataset level, positives and negatives contain
people at almost the same rate (gap about 0), so the "positives more often contain people"
count imbalance does not hold. That does not rule out firing at real people, though. The
likelier explanation is covariate or context shift: the people in the training negatives are
mostly clean studio portraits (`people_portrait`), which do not cover people in cluttered real
scenes. This script measures the symptom directly. It runs the gatekeeper over a probe set of
"people present, no screen text" images, all of which are do-not-record (0) by definition, so
any record (1) is a false positive, and reports the FP rate at the deployment threshold and
above.

Why this distinguishes the two causes:
- If probe FP is high while the person-count gap is about 0, it is covariate or context shift
  (studio portraits not covering real-scene people). The fix is more scene diversity in the
  people negatives, not simply more of them.
- If probe FP is low, then the impression that it fires whenever a person is in frame is
  confounded by something else (threshold, lighting, camera pipeline). Look there instead.

It does four things:
  1. Read the probe set (real photographs at any resolution). If the directory is empty, print
     the collection spec and exit.
  2. Leakage control: cross-check by Pexels ID from the filename and by perceptual hash (pHash
     plus pixel correlation, same criteria as check_leakage), removing any probe image that
     collides with or near-duplicates train, val or test, so the FP rate is measured on a clean
     probe set.
  3. Score the clean probe set with both gatekeepers, keras (float) and int8 (.tflite), and
     report the FP rate at each threshold (deployment 0.55, argmax 0.5, and two tighter ones at
     0.7 and 0.9). Ground truth is all 0, so FP rate is simply the fraction judged as record.
  4. Save a montage of false-positive cases plus Grad-CAM for a few of them, to see whether the
     heat actually falls on people. Heat on FP samples is the hard evidence of a shortcut, more
     so than heat on positives. Writes docs/probes/probe_fp_audit.md.

Measurement choices, written down so the run can be audited:
- Probe images are at arbitrary resolution, so both evaluations use deployment preprocessing:
  cv2 greyscale, resize to 96x96 with INTER_AREA, divide by 255, and for int8 quantise using
  the model's own parameters. That is exactly the deployment-time behaviour we want to measure.
  The int8 path reuses load_model/predict from `hardware/infer.py` so it matches the Pi
  exactly; keras uses the same greyscale and resize.
- Nothing is retrained and no data is changed. The probe set and all outputs stay under data/
  and are excluded by .gitignore.

Dependencies: opencv, numpy, pandas, tensorflow (keras plus a tf.lite fallback). Reuses
phash64/popcount64 from scripts/check_leakage.py and the int8 runtime from hardware/infer.py.
Adds no new dependency.

Examples:
  .venv/bin/python scripts/probe_fp_test.py                 # empty directory: print the collection spec
  .venv/bin/python scripts/probe_fp_test.py --probe-dir data/probe_person_noscreen
  .venv/bin/python scripts/probe_fp_test.py --no-gradcam --thresholds 0.5,0.55,0.7,0.9
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Reuse the leakage check's hash implementation, as dedup_resplit does, so the dedup and
# leakage criteria stay identical across the whole repository.
from check_leakage import phash64, popcount64  # noqa: E402

INPUT_SIZE = 96
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Deployment operating point: the threshold tuned for v4_mvp, about 0.55 (see models/README).
# The threshold is applied to p(record).
DEPLOY_THRESHOLD = 0.55


# ----------------------------- collection spec -----------------------------
PROBE_SPEC = f"""\
======================== Probe set collection spec ========================
Goal: measure the gatekeeper's false-trigger rate on real "people present, no screen text"
scenes. Every image is do-not-record by definition.

Directory layout (under data/, already gitignored):
  data/probe_person_noscreen/
      *.jpg / *.png ...            # flat is fine; subdirectories per scene also work, the scan recurses

Count: aim for about 200 images. Below roughly 80 the FP rate is not stable; at 200 the 95%
confidence interval is around +/- 3-4 percentage points.

Content. The point is to differ in character from the clean studio portraits in the
people_portrait training negatives:
  Include  people in a meeting room, with no readable projector or screen text
  Include  people working in an office, with the screen unreadable or out of frame
  Include  people in cluttered real scenes: cafes, homes, streets, group photos
  Include  multiple people, half-body or seated, busy backgrounds, natural light, unposed
  Exclude  clean studio portraits, which are the people_portrait negatives and already covered
  Exclude  anything with readable screen, whiteboard, document or slide text, which would make
           the image a positive and contaminate the probe

Fetching from Pexels (optional):
  1. Use the existing scripts/download_images.py with a probe config. Example queries:
     "people meeting room", "office team working", "friends cafe group",
     "people working laptop office", "group people indoor candid". Output to
     data/probe_person_noscreen/. download_images.py already deduplicates globally by Pexels ID.
  2. This script then applies a second leakage check, cross-referencing Pexels ID from the
     filename and perceptual hash, and automatically removes any probe image that collides with
     or near-duplicates train, val or test. So even if a query returns an image that appears in
     training, it will not contaminate the FP rate; it is excluded and reported.

Once the images are in place, re-run:
  .venv/bin/python scripts/probe_fp_test.py --probe-dir data/probe_person_noscreen
==========================================================================="""

# ----------------------------- Pexels ID parsing -----------------------------
def photo_ids(stem: str) -> set[str]:
    """Extract candidate Pexels image ids from a filename stem: numeric tokens of length 6+.

    Covers both naming schemes: raw `<slug>_<seq4>_<id>` with the id last, and processed
    `<slug>_<seq4>_<id>_<hash8>` with the id second from last. seq is 4 digits, so the length
    threshold of 6 excludes it.
    """
    return {t for t in re.split(r"[_\-.]", stem) if t.isdigit() and len(t) >= 6}


def manifest_id_set(manifest: Path) -> set[str]:
    df = pd.read_csv(manifest)
    ids: set[str] = set()
    for p in df["path"]:
        ids |= photo_ids(Path(p).stem)
    return ids


# ----------------------------- perceptual-hash leakage check -----------------------------
def load_gray96(path: Path) -> np.ndarray | None:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    if g.shape != (INPUT_SIZE, INPUT_SIZE):
        g = cv2.resize(g, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return g


def build_manifest_index(manifest: Path, data_root: Path) -> dict:
    """Load the whole manifest as 96x96 greyscale and precompute pHash, centred vectors and
    norms, so pixel correlation can be vectorised."""
    df = pd.read_csv(manifest)
    n = len(df)
    flats = np.zeros((n, INPUT_SIZE * INPUT_SIZE), np.float32)
    phashes = np.zeros(n, np.uint64)
    bad = 0
    for i, rel in enumerate(df["path"]):
        g = load_gray96(data_root / rel)
        if g is None:
            bad += 1
            continue
        flats[i] = g.astype(np.float32).flatten()
        phashes[i] = phash64(g)
    centered = flats - flats.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1)
    if bad:
        print(f"  [warning] {bad} manifest images failed to load and were ignored", file=sys.stderr)
    return {
        "phashes": phashes,
        "centered": centered,
        "norm": np.where(norm == 0, 1.0, norm),
        "paths": df["path"].to_numpy(),
        "splits": df["split"].to_numpy() if "split" in df else np.array(["?"] * n),
    }


def perceptual_match(gray96: np.ndarray, idx: dict, phash_th: int, pixel_corr: float) -> dict | None:
    """Return the strongest near-duplicate match between this probe image and the manifest, if
    one reaches the threshold, otherwise None."""
    ph = phash64(gray96)
    ham = popcount64(np.uint64(ph) ^ idx["phashes"])
    cand = np.nonzero(ham <= phash_th)[0]
    if cand.size == 0:
        return None
    v = gray96.astype(np.float32).flatten()
    v = v - v.mean()
    vn = np.linalg.norm(v) or 1.0
    best = None
    for j in cand:
        corr = float(np.dot(v, idx["centered"][j]) / (vn * idx["norm"][j]))
        if corr >= pixel_corr and (best is None or corr > best["pixel_corr"]):
            best = {"manifest_path": str(idx["paths"][j]), "split": str(idx["splits"][j]),
                    "phash_hamming": int(ham[j]), "pixel_corr": round(corr, 4)}
    return best


# ----------------------------- inference -----------------------------
def keras_scores(model, grays: list[np.ndarray]) -> np.ndarray:
    """Preprocess a batch of greyscale images of any resolution under deployment rules, run
    inference, and return the array of p(record)."""
    import tensorflow as tf  # local import
    batch = np.zeros((len(grays), INPUT_SIZE, INPUT_SIZE, 1), np.float32)
    for i, g in enumerate(grays):
        r = cv2.resize(g, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        batch[i, ..., 0] = r.astype(np.float32) / 255.0
    probs = model.predict(tf.convert_to_tensor(batch), verbose=0)
    return probs[:, 1]


# -- Self-contained int8 runtime. Reproduces the deployment preprocessing in hardware/infer.py
#    exactly: greyscale, resize to 96 with INTER_AREA, divide by 255, then quantise to int8
#    using the model's own scale and zero_point. The output is dequantised back to
#    probabilities and p(record) is taken as softmax[1].
#    infer.py only existed on the hardware branch, so the logic is inlined here to keep this
#    audit runnable standalone. It is line-for-line equivalent and can be swapped for
#    `import infer` wherever that module is available.
def _make_interpreter(model_path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter  # preferred runtime on the Pi
        return Interpreter(model_path=str(model_path))
    except ImportError:
        import tensorflow as tf  # laptop fallback, functionally equivalent
        return tf.lite.Interpreter(model_path=str(model_path))


def load_int8(model_path: Path):
    it = _make_interpreter(model_path)
    it.allocate_tensors()
    return it


def int8_predict_one(interp, gray: np.ndarray) -> float:
    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    ind = interp.get_input_details()[0]
    outd = interp.get_output_details()[0]
    scale, zp = ind["quantization"]
    if scale == 0:  # float-input model: feed float directly
        q = x.reshape(1, INPUT_SIZE, INPUT_SIZE, 1).astype(np.float32)
    else:
        q = np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)
        q = q.reshape(1, INPUT_SIZE, INPUT_SIZE, 1)
    interp.set_tensor(ind["index"], q)
    interp.invoke()
    y = interp.get_tensor(outd["index"])[0]
    o_scale, o_zp = outd["quantization"]
    probs = (y.astype(np.float32) - o_zp) * o_scale if o_scale else y.astype(np.float32)
    return float(probs[1])  # p(record)


def int8_scores(interp, grays: list[np.ndarray]) -> np.ndarray:
    """Take 2D greyscale (skipping colour conversion; resize and quantisation happen inside)
    and return the array of p(record)."""
    out = np.zeros(len(grays), np.float32)
    for i, g in enumerate(grays):
        out[i] = int8_predict_one(interp, g)
    return out


def fp_table(scores: np.ndarray, thresholds: list[float]) -> list[dict]:
    """Ground truth is all 0, so the FP rate is the fraction judged as record (score >= th)."""
    n = len(scores)
    rows = []
    for t in thresholds:
        fp = int((scores >= t).sum())
        rows.append({"threshold": round(t, 3), "n": n, "fp": fp,
                     "fp_rate": round(fp / n, 4) if n else 0.0})
    return rows


# ----------------------------- Grad-CAM on false-positive cases -----------------------------
def gradcam_on(model_path: Path, items: list[tuple[str, np.ndarray, float]],
               out_dir: Path) -> dict:
    """Grad-CAM over the false-positive cases: block4_relu with respect to p(record).
    items is a list of (name, gray, score)."""
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, compile=False)
        gm = tf.keras.Model(model.input,
                            [model.get_layer("block4_relu").output,
                             model.get_layer("logits").output])

        def cam(gray96):
            x = tf.convert_to_tensor((gray96.astype(np.float32) / 255.0)[None, ..., None])
            with tf.GradientTape() as tape:
                conv, logits = gm(x, training=False)
                loss = logits[:, 1]
            grads = tape.gradient(loss, conv)
            w = tf.reduce_mean(grads, axis=(1, 2))
            c = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], axis=-1)[0])
            return (c / (tf.reduce_max(c) + 1e-8)).numpy()

        saved = []
        for name, gray, score in items:
            g96 = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
            m = cam(g96)
            disp = cv2.cvtColor(cv2.resize(gray, (288, 288), interpolation=cv2.INTER_AREA),
                                cv2.COLOR_GRAY2BGR)
            heat = cv2.applyColorMap(cv2.resize((m * 255).astype(np.uint8), (288, 288)),
                                     cv2.COLORMAP_JET)
            over = cv2.addWeighted(disp, 0.55, heat, 0.45, 0)
            cv2.putText(over, f"p(rec)={score:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            path = out_dir / f"gradcam_fp_{name}.png"
            cv2.imwrite(str(path), np.hstack([disp, over]))
            saved.append(path.name)
        print(f"  Grad-CAM (FP): {len(saved)} saved to {out_dir} "
              "(original left, heatmap right; heat on people is hard evidence of a shortcut)")
        return {"ok": True, "saved": saved}
    except Exception as e:  # noqa: BLE001
        print(f"  Grad-CAM skipped: {type(e).__name__}: {e}")
        return {"ok": False, "saved": [], "note": f"{type(e).__name__}: {e}"}


def fp_montage(items: list[tuple[str, np.ndarray, float]], out_dir: Path, n: int) -> str | None:
    """Montage of false-positive cases, meaning probe images judged as record at the deployment
    threshold. Each cell is labelled with p(record)."""
    items = items[:n]
    if not items:
        return None
    thumbs = []
    for _, gray, score in items:
        t = cv2.cvtColor(cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_GRAY2BGR)
        cv2.putText(t, f"{score:.2f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)
        thumbs.append(t)
    cols = min(6, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    grid = np.full((rows * 160, cols * 160, 3), 30, np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * 160:(r + 1) * 160, c * 160:(c + 1) * 160] = t
    path = out_dir / "montage_fp_cases.png"
    cv2.imwrite(str(path), grid)
    return path.name


# ----------------------------- report document -----------------------------
def write_markdown(md_path: Path, results: dict | None, probe_dir: Path, out_dir: Path) -> None:
    L = ["# Probe false-trigger audit\n"]
    L.append("> Measures the gatekeeper's false-trigger rate directly on real "
             "\"people present, no screen text\" scenes. By definition every probe image should be "
             "judged do-not-record, so any record is a false positive. Diagnostic only: nothing "
             "was retrained and no data was changed.\n")
    L.append("\n## Method\n")
    L.append(f"- Probe directory: `{probe_dir}/` (gitignored). Preprocessing matches "
             "deployment: cv2 greyscale, resize to 96 with INTER_AREA, divide by 255, then "
             "quantise for int8.\n")
    L.append("- Leakage control: a double check by Pexels ID from the filename and by "
             "perceptual hash (pHash within threshold and pixel correlation above threshold, "
             "reusing the same criteria as `check_leakage`), removing any probe image that "
             "collides with or near-duplicates train, val or test.\n")
    L.append("- Two gatekeepers are scored: `keras (float)` and `int8 (.tflite)`, the latter "
             "with a self-contained int8 runtime reproducing the deployment preprocessing in "
             "`hardware/infer.py`. Deployment threshold 0.55; 0.5, 0.7 and 0.9 are also "
             "listed.\n")

    if results is None:
        L.append("\n## Collection spec (the probe directory is currently empty)\n")
        L.append("```\n" + PROBE_SPEC + "\n```\n")
        L.append("\n## Results\n_Generated once probe images are in place and this script is re-run._\n")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("".join(L), encoding="utf-8")
        return

    lk = results["leakage"]
    L.append("\n## 1. Leakage check\n")
    L.append(f"- Probe images: {results['n_total']}. Removed as leaked: {lk['n_leaked']} "
             f"(Pexels ID collisions {lk['n_id']}, perceptual near-duplicates "
             f"{lk['n_perceptual']}). Clean probe set: {results['n_clean']} images, used for "
             "the FP measurement.\n")
    if lk["examples"]:
        L.append("- Examples of leakage (probe image and the split/image it matched):\n")
        for e in lk["examples"][:8]:
            tag = e.get("manifest_path", f"ID={e.get('id')}")
            L.append(f"  - `{e['probe']}` → {e.get('split','?')} `{tag}`"
                     f"(corr={e.get('pixel_corr','-')})\n")

    L.append("\n## 2. False-trigger rate on the clean probe set "
             "(ground truth is do-not-record throughout)\n")
    for tag, rows in (("keras(float)", results["fp_keras"]), ("int8(.tflite)", results["fp_int8"])):
        L.append(f"\n**{tag}**\n\n| Threshold | Judged record / total | FP rate |\n|---|---|---|\n")
        for r in rows:
            mark = "  (deployment)" if abs(r["threshold"] - DEPLOY_THRESHOLD) < 1e-6 else ""
            L.append(f"| {r['threshold']} | {r['fp']}/{r['n']} | **{r['fp_rate']*100:.1f}%**{mark} |\n")

    L.append("\n## 3. False-positive cases for manual review\n")
    if results.get("montage"):
        L.append(f"- Montage: `{out_dir}/{results['montage']}` "
                 "(red text is the probability of record)\n")
    gc = results.get("gradcam", {})
    if gc.get("ok"):
        L.append(f"- Grad-CAM: `{out_dir}/gradcam_fp_*.png` ({len(gc['saved'])} images, "
                 "original left, heatmap right). If the heat on a false-positive sample locks "
                 "onto a face or a body, that is hard evidence of a shortcut.\n")
    else:
        L.append(f"- Grad-CAM: not produced ({gc.get('note','-')}).\n")

    dep = next(r for r in results["fp_int8"] if abs(r["threshold"] - DEPLOY_THRESHOLD) < 1e-6)
    rate = dep["fp_rate"] * 100
    L.append("\n## 4. Reading\n")
    if results["n_clean"] < 60:
        L.append(f"- Caveat: only {results['n_clean']} clean probe images, so the FP rate is "
                 "statistically noisy. Expand to around 200 before drawing conclusions.\n")
    if rate >= 25:
        L.append(f"- At the int8 deployment threshold of 0.55, FP is **{rate:.1f}%**, which is "
                 "high. Combined with the audit gap of about 0, this looks like covariate or "
                 "context shift: the studio-portrait negatives do not cover people in cluttered "
                 "real scenes. The fix is more scene diversity among the people negatives rather "
                 "than simply more of them. Check the Grad-CAM to see whether attention locks "
                 "onto people.\n")
    elif rate >= 10:
        L.append(f"- At the int8 deployment threshold of 0.55, FP is **{rate:.1f}%**, which is "
                 "moderate. There is some context shift, so more diverse real-scene people "
                 "negatives are worth adding. Also check whether the false-positive cases are "
                 "mostly borderline images such as partially visible screens or reflections.\n")
    else:
        L.append(f"- At the int8 deployment threshold of 0.55, FP is **{rate:.1f}%**, which is "
                 "low. The gatekeeper does not broadly false-trigger on real people, so the "
                 "impression that it fires whenever a person is in frame is more likely "
                 "confounded by the threshold, the lighting or the camera pipeline. Investigate "
                 "there.\n")
    L.append("- If keras and int8 FP differ noticeably, quantisation has moved the operating "
             "point and the deployment threshold must be re-calibrated against int8.\n")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("".join(L), encoding="utf-8")


# ----------------------------- main -----------------------------
def collect_images(probe_dir: Path) -> list[Path]:
    if not probe_dir.exists():
        return []
    return sorted(p for p in probe_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Probe false-trigger test: measure the gatekeeper's false-positive rate on "
                    "real people-present, no-screen-text scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--probe-dir", type=Path, default=Path("data/probe_person_noscreen"),
                   help="probe image directory (real photographs at any resolution; scanned recursively)")
    p.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"),
                   help="the full train/val/test manifest, used for the leakage check")
    p.add_argument("--data-root", type=Path, default=Path("data/processed"))
    p.add_argument("--keras-model", type=Path, default=Path("models/gatekeeper_v4_mvp.keras"))
    p.add_argument("--int8-model", type=Path, default=Path("models/gatekeeper_v4_mvp_int8.tflite"))
    p.add_argument("--out", type=Path, default=Path("data/processed/probe_fp_audit"),
                   help="output directory for montage, Grad-CAM, CSV and JSON (gitignored)")
    p.add_argument("--md-out", type=Path, default=Path("docs/probes/probe_fp_audit.md"))
    p.add_argument("--thresholds", type=str, default="0.5,0.55,0.7,0.9")
    p.add_argument("--phash-th", type=int, default=6, help="perceptual hash Hamming threshold (same as check_leakage)")
    p.add_argument("--pixel-corr", type=float, default=0.90, help="pixel correlation threshold (same as check_leakage)")
    p.add_argument("--keep-leaked", action="store_true", help="do not remove leaked probe images; report only. For debugging")
    p.add_argument("--no-leakage-check", action="store_true", help="skip the leakage check (not recommended)")
    p.add_argument("--no-gradcam", action="store_true")
    p.add_argument("--gradcam-n", type=int, default=10)
    p.add_argument("--montage-n", type=int, default=18)
    p.add_argument("--limit", type=int, default=None, help="use only the first N probe images (smoke test)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    if DEPLOY_THRESHOLD not in thresholds:
        thresholds = sorted(set(thresholds + [DEPLOY_THRESHOLD]))

    args.out.mkdir(parents=True, exist_ok=True)
    images = collect_images(args.probe_dir)
    if args.limit:
        images = images[:args.limit]

    if not images:
        print(PROBE_SPEC)
        print(f"\nProbe directory `{args.probe_dir}` is empty or missing. "
              f"The collection spec has been written to {args.md_out}.")
        write_markdown(args.md_out, None, args.probe_dir, args.out)
        return 0

    print(f"{len(images)} probe images at {args.probe_dir}")

    # -- leakage check --
    leaked: dict[str, dict] = {}
    n_id = n_perc = 0
    if not args.no_leakage_check:
        print("Leakage check (Pexels ID plus perceptual hash)...")
        mids = manifest_id_set(args.manifest)
        idx = build_manifest_index(args.manifest, args.data_root)
        for img in images:
            ids = photo_ids(img.stem)
            hit_id = ids & mids
            if hit_id:
                leaked[str(img)] = {"probe": img.name, "id": sorted(hit_id)[0],
                                    "split": "?", "reason": "pexels_id"}
                n_id += 1
                continue
            g = load_gray96(img)
            if g is None:
                continue
            m = perceptual_match(g, idx, args.phash_th, args.pixel_corr)
            if m:
                leaked[str(img)] = {"probe": img.name, "reason": "perceptual", **m}
                n_perc += 1
        print(f"  leaked: {n_id} by Pexels ID, {n_perc} by perceptual near-duplicate "
              f"({len(leaked)} total)")

    clean = images if args.keep_leaked else [i for i in images if str(i) not in leaked]
    print(f"  clean probe images: {len(clean)}"
          + (" (--keep-leaked: nothing removed)" if args.keep_leaked else ""))
    if not clean:
        print("No clean probe images left; everything was judged leaked. "
              "Check whether the probe source overlaps the training set.")
        return 1

    # -- load clean probes as greyscale, keeping the original resolution for visualisation
    #    and deployment preprocessing --
    names, grays = [], []
    for img in clean:
        g = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        names.append(re.sub(r"[^A-Za-z0-9]+", "_", img.stem)[:60])
        grays.append(g)

    # -- run both gatekeepers --
    import tensorflow as tf  # noqa: E402
    print(f"keras inference: {args.keras_model}")
    kmodel = tf.keras.models.load_model(args.keras_model, compile=False)
    s_keras = keras_scores(kmodel, grays)
    print(f"int8 inference: {args.int8_model}")
    interp = load_int8(args.int8_model)
    s_int8 = int8_scores(interp, grays)

    fp_keras = fp_table(s_keras, thresholds)
    fp_int8 = fp_table(s_int8, thresholds)

    # -- print --
    print("\n" + "=" * 60)
    print("Probe FP (ground truth is do-not-record throughout; "
          "FP rate is the fraction judged as record)")
    print("=" * 60)
    for tag, rows in (("keras(float)", fp_keras), ("int8 (.tflite)", fp_int8)):
        print(f"\n[{tag}]  threshold   record/total   FP rate")
        for r in rows:
            mark = "  (deployment)" if abs(r["threshold"] - DEPLOY_THRESHOLD) < 1e-6 else ""
            print(f"          {r['threshold']:<5}  {r['fp']:>3}/{r['n']:<3}  "
                  f"{r['fp_rate']*100:5.1f}%{mark}")

    # -- false-positive cases at the int8 deployment threshold: montage plus Grad-CAM --
    fp_items = sorted(
        [(names[i], grays[i], float(s_int8[i])) for i in range(len(grays))
         if s_int8[i] >= DEPLOY_THRESHOLD],
        key=lambda t: -t[2])
    # Keep descending by score so the most confident false positives come first; those are the
    # ones worth looking at in the montage and Grad-CAM
    montage = fp_montage(fp_items, args.out, args.montage_n)
    if montage:
        print(f"  FP montage -> {args.out/montage}")
    gradcam = {"ok": False, "saved": [], "note": "--no-gradcam"}
    if not args.no_gradcam and fp_items:
        gradcam = gradcam_on(args.keras_model, fp_items[:args.gradcam_n], args.out)

    # -- write the per-image CSV --
    det = pd.DataFrame({
        "probe": [c.name for c in clean[:len(grays)]],
        "p_record_keras": np.round(s_keras, 4),
        "p_record_int8": np.round(s_int8, 4),
        "fp_at_deploy_int8": (s_int8 >= DEPLOY_THRESHOLD).astype(int),
    })
    det.to_csv(args.out / "probe_scores.csv", index=False)

    results = {
        "n_total": len(images), "n_clean": len(grays),
        "leakage": {"n_leaked": len(leaked), "n_id": n_id, "n_perceptual": n_perc,
                    "examples": list(leaked.values())},
        "fp_keras": fp_keras, "fp_int8": fp_int8,
        "deploy_threshold": DEPLOY_THRESHOLD,
        "montage": montage, "gradcam": gradcam,
    }
    (args.out / "probe_fp_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(args.md_out, results, args.probe_dir, args.out)
    print(f"\nAudit document -> {args.md_out}; "
          f"summary JSON -> {args.out/'probe_fp_summary.json'}")

    dep_k = next(r for r in fp_keras if abs(r["threshold"] - DEPLOY_THRESHOLD) < 1e-6)
    dep_i = next(r for r in fp_int8 if abs(r["threshold"] - DEPLOY_THRESHOLD) < 1e-6)
    print("\nRESULT " + json.dumps({
        "n_clean": len(grays), "n_leaked": len(leaked),
        "fp_rate_keras@0.55": dep_k["fp_rate"], "fp_rate_int8@0.55": dep_i["fp_rate"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
