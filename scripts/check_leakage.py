#!/usr/bin/env python3
"""Data leakage check: find duplicate and near-duplicate images across train, val and test.

Two passes, cheap first then precise:
  1. Perceptual-hash coarse pass. Compute pHash (DCT) and dHash over each processed 96x96
     greyscale image, compare every pair by Hamming distance, and keep pairs within
     --phash-th as candidates. Perceptual hashes are robust to brightness and slight
     rescaling, which is exactly what "looks the same" needs.
  2. Pixel-level confirmation. For each candidate pair compute the Pearson correlation and
     normalised MSE over the 96x96 pixels. Only pairs correlating at or above --pixel-corr
     count as real near-duplicates, which filters out hash collisions.

Confirmed pairs are then classified by whether they span splits, and the cross-split count is
the headline number, because that is the leakage.

Dependencies: numpy, pandas, opencv-python, all already in requirements.txt.
This modifies no data. It reads images and writes one CSV report listing the candidate pairs.

Examples:
  .venv/bin/python scripts/check_leakage.py
  .venv/bin/python scripts/check_leakage.py --phash-th 6 --pixel-corr 0.90
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def phash64(gray: np.ndarray) -> np.uint64:
    """DCT perceptual hash: 32x32, DCT, take the top-left 8x8 excluding DC, compare against
    the median, giving 64 bits."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    block = dct[:8, :8].flatten()
    med = np.median(block[1:])  # take the median after dropping the [0,0] DC component
    bits = block > med
    bits[0] = False  # the DC bit is fixed and does not participate
    out = np.uint64(0)
    for b in bits:
        out = (out << np.uint64(1)) | np.uint64(1 if b else 0)
    return out


def dhash64(gray: np.ndarray) -> np.uint64:
    """Difference hash: 9x8, compare adjacent columns, giving 8x8 = 64 bits."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int32)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    out = np.uint64(0)
    for b in bits:
        out = (out << np.uint64(1)) | np.uint64(1 if b else 0)
    return out


def popcount64(x: np.ndarray) -> np.ndarray:
    """Population count over a uint64 array, vectorised with the SWAR algorithm."""
    x = x.astype(np.uint64)
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    h01 = np.uint64(0x0101010101010101)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return (x * h01) >> np.uint64(56)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Check for duplicate and near-duplicate images across splits, which is data leakage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=Path("data/processed"))
    p.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    p.add_argument("--phash-th", type=int, default=6,
                   help="pHash Hamming distance threshold; pairs at or below it become candidates")
    p.add_argument("--pixel-corr", type=float, default=0.90,
                   help="pixel Pearson correlation threshold; at or above it, a pair is a real near-duplicate")
    p.add_argument("--out", type=Path,
                   default=Path("docs/results/leakage_candidates.csv"))
    args = p.parse_args()

    df = pd.read_csv(args.manifest)
    n = len(df)
    print(f"read manifest: {n} images, split distribution:")
    print(df["split"].value_counts().to_string())

    # Load every image as 96x96 greyscale and compute pHash and dHash at the same time.
    imgs = np.zeros((n, 96 * 96), dtype=np.float32)
    phashes = np.zeros(n, dtype=np.uint64)
    dhashes = np.zeros(n, dtype=np.uint64)
    bad = 0
    for i, rel in enumerate(df["path"]):
        g = cv2.imread(str(args.data_root / rel), cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"[warning] cannot read: {rel}", file=sys.stderr)
            bad += 1
            continue
        if g.shape != (96, 96):
            g = cv2.resize(g, (96, 96), interpolation=cv2.INTER_AREA)
        imgs[i] = g.astype(np.float32).flatten()
        phashes[i] = phash64(g)
        dhashes[i] = dhash64(g)
    if bad:
        print(f"[warning] {bad} images failed to load.", file=sys.stderr)

    # Precompute each image's mean and standard deviation for the vectorised Pearson correlation.
    mean = imgs.mean(axis=1, keepdims=True)
    centered = imgs - mean
    norm = np.linalg.norm(centered, axis=1)  # L2 norm of each row
    norm_safe = np.where(norm == 0, 1.0, norm)

    splits = df["split"].to_numpy()
    paths = df["path"].to_numpy()
    subclasses = df["subclass"].to_numpy() if "subclass" in df else np.array([""] * n)
    labels = df["label"].to_numpy()

    # All-pairs coarse pass: for each i, compute the pHash Hamming distance to every j > i and
    # keep the candidates.
    records = []
    for i in range(n):
        ph = popcount64(phashes[i] ^ phashes[i + 1:])
        cand = np.nonzero(ph <= args.phash_th)[0]
        for off in cand:
            j = i + 1 + int(off)
            # Pixel Pearson correlation, as a vectorised dot product of two rows over the
            # product of their norms
            corr = float(np.dot(centered[i], centered[j]) / (norm_safe[i] * norm_safe[j]))
            dh = int(popcount64(np.array([dhashes[i] ^ dhashes[j]], dtype=np.uint64))[0])
            mse = float(np.mean((imgs[i] - imgs[j]) ** 2)) / (255.0 ** 2)
            records.append({
                "path_a": paths[i], "path_b": paths[j],
                "split_a": splits[i], "split_b": splits[j],
                "label_a": int(labels[i]), "label_b": int(labels[j]),
                "subclass_a": subclasses[i], "subclass_b": subclasses[j],
                "phash_hamming": int(ph[off]), "dhash_hamming": dh,
                "pixel_corr": round(corr, 4), "norm_mse": round(mse, 5),
                "cross_split": splits[i] != splits[j],
            })

    cand_df = pd.DataFrame(records)
    print(f"\npHash candidate pairs (Hamming <= {args.phash_th}): {len(cand_df)}")
    if cand_df.empty:
        print("no near-duplicate candidates at all, so there is no leakage.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cand_df.to_csv(args.out, index=False)
        return 0

    # Precise pass: only pixel correlation at or above the threshold counts as a real
    # near-duplicate.
    confirmed = cand_df[cand_df["pixel_corr"] >= args.pixel_corr].copy()
    exact = confirmed[confirmed["pixel_corr"] >= 0.999]
    near = confirmed[confirmed["pixel_corr"] < 0.999]

    print(f"pixel-confirmed (corr >= {args.pixel_corr}): {len(confirmed)} real near-duplicate "
          f"pairs, of which {len(exact)} are near-identical at corr >= 0.999 and {len(near)} "
          f"are highly similar")

    cross = confirmed[confirmed["cross_split"]]
    within = confirmed[~confirmed["cross_split"]]
    print(f"\n>>> cross-split duplicate pairs, which is the leakage: {len(cross)}")
    print(f"    within-split duplicate pairs, not leakage but still a data-quality issue: {len(within)}")

    if len(cross):
        print("\ncross-split duplicate pairs, first 30:")
        cols = ["split_a", "split_b", "label_a", "label_b",
                "phash_hamming", "pixel_corr", "path_a", "path_b"]
        print(cross[cols].head(30).to_string(index=False))
        print("\ncross-split duplicates counted by split pair:")
        pair_key = cross.apply(
            lambda r: " ↔ ".join(sorted([r["split_a"], r["split_b"]])), axis=1)
        print(pair_key.value_counts().to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cand_df.to_csv(args.out, index=False)
    print(f"\ncandidate pair details written to {args.out} ({len(cand_df)} rows)")

    # Emit a machine-readable conclusion line.
    print(f"\nLEAKAGE cross_split_pairs={len(cross)} within_split_pairs={len(within)} "
          f"exact={len(exact)} near={len(near)} candidates={len(cand_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
