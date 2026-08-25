#!/usr/bin/env python3
"""Task 2b, phase two. QC pre-filter: surface the new negatives that may contain a readable
text screen, for manual review. It deletes nothing.

Motivation: the new task 2b negatives (indoor offices, meeting rooms and homes with screen-like
surfaces) very easily include images with readable screen, whiteboard, slide or code-screen
text, which would make them positives. Mislabelling those as negatives would contaminate the
negative class. This script scores each image heuristically for "looks like a text screen" and
sorts them, producing a shortlist as a CSV plus a montage for focused manual review. It only
labels and ranks; it never deletes anything. Deletion is a human decision.

Heuristics, all in cv2, resizing each image to at most 256 first to respect the memory limits:
  - rect_area: the largest bright quadrilateral as a fraction of the frame (screen, whiteboard,
    window, picture frame and similar screen-like regions). Larger is more suspicious.
  - text_regions: the number of glyph-like connected components found by MSER, filtered to
    small size and a reasonable aspect ratio. Dense text scores high. This is classic text
    region detection and it false-fires on texture and patterns, so it is a ranking signal
    only, never evidence.
  - suspicion = normalised text_regions x (0.5 + 0.5 x has a bright rectangle). Plenty of text
    AND a screen-like region is the most suspicious combination.

Sorted descending, with flagged = suspicion at or above a percentile threshold, defaulting to
the 70th percentile, which flags roughly the most suspicious 30%.

Output: docs/results/task2b_results/qc_text_screen_suspects.csv plus one or more montages, each
cell labelled with the filename and its text and rect scores.

This is a deliberately over-inclusive filter. Flagged does not mean the image contains a text
screen, and not being flagged does not guarantee it is clean. The final judgement is entirely
manual.

"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
DET = 256   # maximum detection resolution
_MSER = cv2.MSER_create()
_MSER.setMinArea(10)
_MSER.setMaxArea(4000)


def collect(dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        if d.exists():
            out += sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)
    return out


def resize_max(gray: np.ndarray, m: int) -> np.ndarray:
    h, w = gray.shape[:2]
    s = m / max(h, w)
    return gray if s >= 1.0 else cv2.resize(gray, (int(round(w*s)), int(round(h*s))),
                                            interpolation=cv2.INTER_AREA)


def bright_rect_area(g: np.ndarray) -> float:
    """Largest bright quadrilateral as a fraction of the frame: screens, whiteboards, windows,
    frames and similar screen-like regions. 0 if there is none."""
    h, w = g.shape[:2]
    area = h * w
    med = float(np.median(g))
    _, th = cv2.threshold(g, max(med, 110), 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.05 * area:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best = max(best, a / area)
    return round(best, 4)


def text_region_count(g: np.ndarray) -> int:
    """Count of glyph-like MSER connected components: small, with a reasonable aspect ratio and
    not extremely elongated. Dense text scores high."""
    regions, _ = _MSER.detectRegions(g)
    n = 0
    h, w = g.shape[:2]
    for r in regions:
        x, y, bw, bh = cv2.boundingRect(r.reshape(-1, 1, 2))
        if bh == 0 or bw == 0:
            continue
        ar = bw / bh
        a = bw * bh
        if 0.05 * h <= bh <= 0.30 * h and 0.15 <= ar <= 8.0 and a <= 0.08 * h * w:
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Task 2b QC pre-filter scoring how likely an image contains "
                                                "a text screen. Deletes nothing.")
    ap.add_argument("--dir", type=Path, action="append", required=True,
                    help="directory to scan; may be repeated")
    ap.add_argument("--out-csv", type=Path,
                    default=Path("docs/results/task2b_results/qc_text_screen_suspects.csv"))
    ap.add_argument("--montage-dir", type=Path,
                    default=Path("docs/results/task2b_results"))
    ap.add_argument("--montage-top", type=int, default=48)
    ap.add_argument("--pct", type=float, default=70.0, help="percentile threshold above which an image is flagged")
    args = ap.parse_args()

    imgs = collect(args.dir)
    rows = []
    for p in imgs:
        full = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if full is None:
            continue
        g = resize_max(full, DET)
        del full
        tr = text_region_count(g)
        ra = bright_rect_area(g)
        rows.append({"path": str(p), "text_regions": tr, "rect_area": ra,
                     "brightness": round(float(g.mean())/255.0, 3)})

    if not rows:
        print("[qc] no images to scan")
        return 0
    tmax = max(r["text_regions"] for r in rows) or 1
    for r in rows:
        r["suspicion"] = round((r["text_regions"]/tmax) * (0.5 + 0.5*(r["rect_area"] > 0)), 4)
    rows.sort(key=lambda r: -r["suspicion"])
    thr = float(np.percentile([r["suspicion"] for r in rows], args.pct))
    for r in rows:
        r["flagged"] = int(r["suspicion"] >= thr and r["text_regions"] > 0)
    n_flag = sum(r["flagged"] for r in rows)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "suspicion", "text_regions",
                                          "rect_area", "brightness", "flagged"])
        w.writeheader()
        w.writerows(rows)

    # Montage of the top-N most suspicious, each cell labelled with its text and rect scores
    top = [r for r in rows if r["flagged"]][:args.montage_top]
    if top:
        thumbs = []
        for r in top:
            g = cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            t = cv2.cvtColor(cv2.resize(g, (160, 160), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_GRAY2BGR)
            cv2.putText(t, f"t{r['text_regions']} r{r['rect_area']:.2f}", (3, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            thumbs.append(t)
        cols = min(6, len(thumbs))
        rows_n = (len(thumbs) + cols - 1) // cols
        grid = np.full((rows_n*160, cols*160, 3), 30, np.uint8)
        for i, t in enumerate(thumbs):
            rr, cc = divmod(i, cols)
            grid[rr*160:(rr+1)*160, cc*160:(cc+1)*160] = t
        mp = args.montage_dir / "qc_text_screen_suspects_montage.png"
        cv2.imwrite(str(mp), grid)
        print(f"[qc] montage -> {mp} (top {len(thumbs)} most suspicious)")

    print(f"[qc] scanned {len(rows)} images, flagged {n_flag} as possibly containing a text screen "
          f"(percentile threshold {thr:.3f}) -> {args.out_csv}")
    print("[qc] For manual review only. No images were deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
