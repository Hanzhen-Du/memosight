#!/usr/bin/env python3
"""Deduplicate and re-split, fixing the cross-split leakage found by check_leakage.py.

Root cause of the leakage: the same Pexels original was downloaded under several keywords and
landed in different positive subclass folders. The stratified split, which splits the positive
class as a whole, then scattered copies of one image across train, val and test, inflating the
evaluation.

Two steps:
  1. build: over the full manifest, compute pHash and pixel correlation on the processed
     images, build connected components from pairs where the pHash Hamming distance is within
     the threshold AND pixel correlation is at or above corr, keep one representative per group
     (the lexicographically smallest path, so it is deterministic), and write the deduplicated
     manifest to manifest_dedup.csv.
  2. split: read the deduplicated manifest and produce a 70/15/15 split stratified by source
     class with --seed, matching prepare_dataset.py, writing
     <prefix>train.csv, val.csv and test.csv.

After dedup each image appears exactly once, which eliminates cross-split leakage at the root.
The split logic is unchanged; only the sample set is deduplicated.

Dependencies: numpy, pandas, opencv-python. Nothing new. It reads images and writes CSVs; the
original images are untouched.

Examples:
  .venv/bin/python scripts/dedup_resplit.py build
  .venv/bin/python scripts/dedup_resplit.py split --seed 42 --prefix dedup_
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Reuse the hash implementation from the leakage check, so the dedup criteria and the leakage
# report agree exactly.
from check_leakage import phash64, popcount64  # noqa: E402

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
CSV_FIELDS = ["path", "label", "source", "subclass", "split"]


def find(parent: dict, x: str) -> str:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(parent: dict, a: str, b: str) -> None:
    ra, rb = find(parent, a), find(parent, b)
    if ra != rb:
        parent[ra] = rb


def build_dedup_manifest(data_root: Path, manifest: Path, out: Path,
                         phash_th: int, pixel_corr: float,
                         exclude_subclasses: list[str] | None = None,
                         out_excluded: Path | None = None) -> None:
    df = pd.read_csv(manifest)

    # Boundary narrowing: drop the ambiguous subclasses that have text but are not MVP launch
    # trigger scenes (phone apps, TV menus, product packaging).
    # The images are not deleted, only removed from the manifest; the removed rows are archived
    # to a separate manifest so the change stays traceable.
    if exclude_subclasses:
        mask = df["subclass"].isin(exclude_subclasses)
        excluded = df[mask].copy()
        df = df[~mask].copy()
        print(f"boundary narrowing: excluding subclasses {exclude_subclasses}")
        print(f"  removed {len(excluded)} images, by subclass:")
        print(excluded["subclass"].value_counts().to_string())
        if out_excluded is not None:
            out_excluded.parent.mkdir(parents=True, exist_ok=True)
            excluded.to_csv(out_excluded, index=False)
            print(f"  archive manifest (images kept, only excluded from training and evaluation): {out_excluded}")

    n = len(df)
    paths = df["path"].to_numpy()
    imgs = np.zeros((n, 96 * 96), dtype=np.float32)
    phashes = np.zeros(n, dtype=np.uint64)
    for i, rel in enumerate(paths):
        g = cv2.imread(str(data_root / rel), cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"[warning] cannot read: {rel}", file=sys.stderr)
            continue
        if g.shape != (96, 96):
            g = cv2.resize(g, (96, 96), interpolation=cv2.INTER_AREA)
        imgs[i] = g.astype(np.float32).flatten()
        phashes[i] = phash64(g)

    centered = imgs - imgs.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1)
    norm_safe = np.where(norm == 0, 1.0, norm)

    parent = {p: p for p in paths}
    n_pairs = 0
    for i in range(n):
        ph = popcount64(phashes[i] ^ phashes[i + 1:])
        for off in np.nonzero(ph <= phash_th)[0]:
            j = i + 1 + int(off)
            corr = float(np.dot(centered[i], centered[j]) / (norm_safe[i] * norm_safe[j]))
            if corr >= pixel_corr:
                union(parent, paths[i], paths[j])
                n_pairs += 1

    groups: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        groups[find(parent, p)].append(p)
    # The representative of each group is the lexicographically smallest path, which is
    # deterministic and reproducible.
    keep = {min(members) for members in groups.values()}
    dedup_df = df[df["path"].isin(keep)].copy()

    multi = [m for m in groups.values() if len(m) > 1]
    removed = n - len(keep)
    print(f"{n} images total: {n_pairs} confirmed near-duplicate pairs across {len(multi)} "
          f"groups, {removed} removed, {len(keep)} remaining after dedup")
    print("label distribution after dedup:")
    print(dedup_df["label"].value_counts().to_string())

    out.parent.mkdir(parents=True, exist_ok=True)
    dedup_df.to_csv(out, index=False)
    print(f"deduplicated manifest written to {out}")


def stratified_split(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """70/15/15 stratified by source class, the same logic as prepare_dataset.py, with test
    taking the remainder."""
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    for source in sorted(df["source"].unique()):
        group = df[df["source"] == source].to_dict("records")
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])
        out["train"].extend(group[:n_train])
        out["val"].extend(group[n_train:n_train + n_val])
        out["test"].extend(group[n_train + n_val:])
    return {k: pd.DataFrame(v) for k, v in out.items()}


def do_split(manifest: Path, out_dir: Path, prefix: str, seed: int,
             quiet: bool = False) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(manifest)
    splits = stratified_split(df, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sdf in splits.items():
        sdf = sdf.copy()
        sdf["split"] = name
        sdf[CSV_FIELDS].to_csv(out_dir / f"{prefix}{name}.csv", index=False)
    if not quiet:
        print(f"[seed={seed}] split of the deduplicated manifest ({len(df)} images):")
        for name in ("train", "val", "test"):
            s = splits[name]
            pos = int((s["label"] == 1).sum())
            print(f"  {name:<5} {len(s):>4} | pos {pos:>3} | neg {len(s) - pos:>3} | "
                  f"pos/neg={pos / max(1, len(s) - pos):.3f}")
        print(f"  → {out_dir}/{prefix}{{train,val,test}}.csv")
    return splits


def main() -> int:
    p = argparse.ArgumentParser(description="Deduplicate and re-split, stratified.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="build the deduplicated manifest")
    pb.add_argument("--data-root", type=Path, default=Path("data/processed"))
    pb.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    pb.add_argument("--out", type=Path, default=Path("data/processed/manifest_dedup.csv"))
    pb.add_argument("--phash-th", type=int, default=6)
    pb.add_argument("--pixel-corr", type=float, default=0.90)
    pb.add_argument("--exclude-subclass", action="append", default=None,
                    help="exclude these subclasses; may be repeated. Used for boundary narrowing, with the "
                         "removed rows archived to a separate manifest.")
    pb.add_argument("--excluded-out", type=Path,
                    default=Path("data/processed/manifest_out_of_scope.csv"),
                    help="output path for the archive manifest of removed rows.")

    ps = sub.add_parser("split", help="stratified split of the deduplicated manifest")
    ps.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_dedup.csv"))
    ps.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ps.add_argument("--prefix", type=str, default="dedup_")
    ps.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.cmd == "build":
        build_dedup_manifest(args.data_root, args.manifest, args.out,
                             args.phash_th, args.pixel_corr,
                             exclude_subclasses=args.exclude_subclass,
                             out_excluded=args.excluded_out)
    elif args.cmd == "split":
        do_split(args.manifest, args.out_dir, args.prefix, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
