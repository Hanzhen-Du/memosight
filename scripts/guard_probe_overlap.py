#!/usr/bin/env python3
"""Zero-overlap guard between the probes and the training set. A hard leakage gate for task 2.

Why it is needed: the global dedup in `download_images.py` only applies by Pexels ID within a
single `--output-root` tree. Task 2's training negatives and positives are downloaded into
`data/raw`, so the dedup baseline covers only `data/raw` and cannot see the two probe
directories (`data/probe_person_noscreen/` and `data/probe_person_screen/` live outside
`data/raw`). A newly downloaded training image could therefore collide with the same Pexels
image, or a near-duplicate, in a probe. The probes are a held-out evaluation set, and any
overlap contaminates the FP and recall figures.

What this script does:
  1. Scan every training image under `data/raw/**` and every probe image in both probe
     directories.
  2. Find overlaps by two criteria, matching check_leakage:
     - Pexels ID: an identical numeric token of 6 or more digits in the filename means the same
       image.
     - Perceptual near-duplicate: 96x96 greyscale pHash Hamming distance within --phash-th AND
       pixel Pearson correlation at or above --pixel-corr.
  3. Any matching TRAINING image is quarantined by moving it to `data/_quarantine_task2/`,
     preserving its relative path, so the probes stay a complete held-out set. Nothing on the
     probe side is touched. Per-item details plus a machine-readable conclusion line are
     printed.

Only the training side is modified, because the probes are the gold holdout. Images are moved
rather than deleted, which keeps the change traceable and reversible and honours the rule
against deleting anything under data/.

--dry-run is off by default, so the script acts. Running with --dry-run first to review the
list is the safer sequence.

Dependencies: opencv and numpy, both already in requirements.txt. Reuses phash64 and popcount64
from scripts/check_leakage.py.

Examples:
  .venv/bin/python scripts/guard_probe_overlap.py --dry-run
  .venv/bin/python scripts/guard_probe_overlap.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from check_leakage import phash64, popcount64  # one hash definition for the whole repository

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
INPUT_SIZE = 96


def photo_ids(stem: str) -> set[str]:
    """Extract candidate Pexels image ids from a filename stem: numeric tokens of 6 or more
    digits, matching probe_fp_test."""
    return {t for t in re.split(r"[_\-.]", stem) if t.isdigit() and len(t) >= 6}


def collect(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def load_gray96(path: Path) -> np.ndarray | None:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    if g.shape != (INPUT_SIZE, INPUT_SIZE):
        g = cv2.resize(g, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return g


def build_index(paths: list[Path]) -> dict:
    """For a batch of images, compute pHash, centred vectors and norms for the vectorised
    pixel correlation, and record each image's Pexels id."""
    n = len(paths)
    flats = np.zeros((n, INPUT_SIZE * INPUT_SIZE), np.float32)
    phashes = np.zeros(n, np.uint64)
    ids: list[set[str]] = []
    ok = np.ones(n, bool)
    for i, p in enumerate(paths):
        ids.append(photo_ids(p.stem))
        g = load_gray96(p)
        if g is None:
            ok[i] = False
            continue
        flats[i] = g.astype(np.float32).flatten()
        phashes[i] = phash64(g)
    centered = flats - flats.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1)
    return {"paths": paths, "phashes": phashes, "centered": centered,
            "norm": np.where(norm == 0, 1.0, norm), "ids": ids, "ok": ok}


def main() -> int:
    ap = argparse.ArgumentParser(description="Zero-overlap guard between probes and training. Matching "
                                                "training images are quarantined.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--train-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--probe-dir", type=Path, action="append",
                    default=None, help="probe directory; may be repeated. Defaults to both probe directories.")
    ap.add_argument("--quarantine", type=Path, default=Path("data/_quarantine_task2"))
    ap.add_argument("--phash-th", type=int, default=6)
    ap.add_argument("--pixel-corr", type=float, default=0.90)
    ap.add_argument("--dry-run", action="store_true", help="report only; move nothing")
    args = ap.parse_args()

    probe_dirs = args.probe_dir or [Path("data/probe_person_noscreen"),
                                    Path("data/probe_person_screen")]
    train = collect(args.train_root)
    probes: list[Path] = []
    for d in probe_dirs:
        probes += collect(d)
    print(f"{len(train)} training images at {args.train_root}")
    print(f"{len(probes)} probe images at {[str(d) for d in probe_dirs]}")
    if not train or not probes:
        print("training set or probes are empty; nothing to check.")
        return 0

    pidx = build_index(probes)
    probe_id_union: set[str] = set().union(*pidx["ids"]) if pidx["ids"] else set()

    hits = []  # (train_path, reason, probe_path, corr)
    for tp in train:
        tids = photo_ids(tp.stem)
        id_hit = tids & probe_id_union
        if id_hit:
            # Find the probe that collides on this id, taking the first
            j = next((k for k, s in enumerate(pidx["ids"]) if tids & s), None)
            pp = pidx["paths"][j].name if j is not None else f"id={sorted(id_hit)[0]}"
            hits.append((tp, "pexels_id", pp, 1.0))
            continue
        g = load_gray96(tp)
        if g is None:
            continue
        ph = phash64(g)
        ham = popcount64(np.uint64(ph) ^ pidx["phashes"])
        cand = np.nonzero((ham <= args.phash_th) & pidx["ok"])[0]
        if cand.size == 0:
            continue
        v = g.astype(np.float32).flatten()
        v = v - v.mean()
        vn = np.linalg.norm(v) or 1.0
        best = None
        for j in cand:
            corr = float(np.dot(v, pidx["centered"][j]) / (vn * pidx["norm"][j]))
            if corr >= args.pixel_corr and (best is None or corr > best[1]):
                best = (pidx["paths"][j], corr)
        if best is not None:
            hits.append((tp, "perceptual", best[0].name, round(best[1], 4)))

    n_id = sum(1 for h in hits if h[1] == "pexels_id")
    n_perc = sum(1 for h in hits if h[1] == "perceptual")
    print(f"\noverlaps found: {len(hits)} training images "
          f"({n_id} by Pexels ID, {n_perc} by perceptual near-duplicate)")
    for tp, reason, pp, corr in hits[:40]:
        rel = tp.relative_to(args.train_root)
        print(f"  [{reason}] data/raw/{rel}  ↔  {pp}  (corr={corr})")
    if len(hits) > 40:
        print(f"  ...({len(hits) - 40} more omitted)")

    moved = 0
    if hits and not args.dry_run:
        for tp, *_ in hits:
            rel = tp.relative_to(args.train_root)
            dst = args.quarantine / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tp), str(dst))
            moved += 1
        print(f"\nquarantined {moved} images by moving, not deleting, into {args.quarantine}/")
    elif hits:
        print(f"\n(dry-run) would quarantine {len(hits)} images into {args.quarantine}/; nothing moved")
    else:
        print("\nZero overlap: no collisions or near-duplicates between the training set and either probe.")

    print(f"\nGUARD overlap_hits={len(hits)} id={n_id} perceptual={n_perc} "
          f"moved={moved} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
