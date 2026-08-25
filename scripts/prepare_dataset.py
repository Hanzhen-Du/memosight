#!/usr/bin/env python3
"""Data preparation pipeline: downscale, label, split. Packages raw images into gatekeeper
training data.

Takes the raw images filed by directory under data/raw/ and turns them into a form the
gatekeeper can train on directly:

  1. Downscale. Walk every image under the three top-level class directories, including
     subdirectories, convert to size x size single-channel greyscale, and downsample with area
     interpolation (INTER_AREA) to avoid moire. Save into data/processed/, preserving the
     original relative directory structure and appending a hash of the original path to the
     filename to avoid collisions.
  2. Label from the directory structure rather than per image:
        positive/        label 1 (record)
        negative_noise/  label 0 (do not record, text that is not a launch scene)
        negative_clean/  label 0 (do not record, no text)
     Produces manifest.csv, one row per processed image: relative path, label, top-level source
     class, and subclass.
  3. Split 70/15/15 into train, val and test, stratified by top-level source class so the
     positive/negative ratio is consistent across splits. The random seed is fixed, so the
     split is reproducible. Writes train.csv, val.csv and test.csv.

This prepares data only. It does not train and does not build a model.

Dependencies: opencv-python and numpy (see requirements.txt). CSVs are written with the
standard library, so pandas is not required.

Examples:
  python3 scripts/prepare_dataset.py                  # run with defaults
  python3 scripts/prepare_dataset.py --dry-run        # scan and print statistics only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2

# Top-level directory to label. The gatekeeper is strictly binary: positive means record (1),
# everything else means do not record (0).
CLASS_LABELS: dict[str, int] = {
    "positive": 1,
    "negative_noise": 0,
    "negative_clean": 0,
}

# Accepted image extensions, compared lowercase.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Split ratios, which must sum to 1. test takes the remainder so floating-point error cannot
# drop samples.
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Columns of the manifest and of each split CSV.
CSV_FIELDS = ["path", "label", "source", "subclass", "split"]


def find_images(input_root: Path) -> list[dict]:
    """Walk every image under the three class directories, including subdirectories, and
    return a list of sample records.

    Each record is {src: absolute path to the original, source: top-level class,
    subclass: subdirectory name, label}. Class directories not present in CLASS_LABELS are
    skipped with a warning.
    """
    samples: list[dict] = []
    for source, label in CLASS_LABELS.items():
        class_dir = input_root / source
        if not class_dir.is_dir():
            print(f"[warning] class directory not found, skipping: {class_dir}", file=sys.stderr)
            continue
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            # The subclass is the first-level subdirectory under the class directory. Images
            # sitting directly in the class root are recorded as "_root".
            rel_parts = path.relative_to(class_dir).parts
            subclass = rel_parts[0] if len(rel_parts) > 1 else "_root"
            samples.append(
                {"src": path, "source": source, "subclass": subclass, "label": label}
            )
    return samples


def processed_relpath(sample: dict, input_root: Path) -> Path:
    """Build the processed image's path relative to output_root: mirror the original relative
    structure, append an 8-character path hash to avoid collisions, and always use .png."""
    rel = sample["src"].relative_to(input_root)
    digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:8]
    return rel.with_name(f"{rel.stem}_{digest}.png")


def stratified_split(
    samples: list[dict], seed: int
) -> dict[str, list[dict]]:
    """Split 70/15/15, stratified by top-level source class.

    Each class is split independently by ratio and the parts are then merged, which keeps the
    proportion of the three classes, and therefore of positives and negatives, the same in each
    split as in the whole set. test takes the remainder so no sample is dropped.
    """
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        buckets[s["source"]].append(s)

    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for source in sorted(buckets):
        group = buckets[source][:]
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])
        out["train"].extend(group[:n_train])
        out["val"].extend(group[n_train : n_train + n_val])
        out["test"].extend(group[n_train + n_val :])  # everything left goes to test
    return out


def print_stats(title: str, rows: list[dict]) -> None:
    """Print a set's total, positives, negatives and positive-to-negative ratio."""
    total = len(rows)
    pos = sum(1 for r in rows if int(r["label"]) == 1)
    neg = total - pos
    ratio = f"{pos / neg:.3f}" if neg else "∞"
    print(f"  {title:<8} total {total:>4} | pos {pos:>4} | neg {neg:>4} | pos/neg = {ratio}")


def write_image(sample: dict, dst: Path, size: int) -> bool:
    """Read, convert to greyscale, downscale to size x size with INTER_AREA, write PNG.
    Returns False if the image cannot be read."""
    img = cv2.imread(str(sample["src"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[warning] unreadable, skipping: {sample['src']}", file=sys.stderr)
        return False
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst), resized))


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write one CSV, with the columns in CSV_FIELDS."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in CSV_FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gatekeeper data preparation: downscale, label, stratified split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--size", type=int, default=96, help="output side length in pixels")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the split")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scan and print statistics only; write nothing",
    )
    args = parser.parse_args()

    input_root: Path = args.input_root
    output_root: Path = args.output_root

    if not input_root.is_dir():
        print(f"[error] input root does not exist: {input_root}", file=sys.stderr)
        return 1

    # 1) scan and label
    samples = find_images(input_root)
    if not samples:
        print(f"[error] no images found under {input_root}.", file=sys.stderr)
        return 1

    # Precompute each record's processed relative path. A dry run needs it too, for the
    # manifest statistics.
    for s in samples:
        s["path"] = str(processed_relpath(s, input_root)).replace("\\", "/")

    print(f"found {len(samples)} images under {input_root}/")
    by_source = defaultdict(int)
    for s in samples:
        by_source[s["source"]] += 1
    for source in CLASS_LABELS:
        print(f"  {source:<16} {by_source.get(source, 0):>4} images (label={CLASS_LABELS[source]})")

    # 3) stratified split
    splits = stratified_split(samples, args.seed)
    for split_name, rows in splits.items():
        for r in rows:
            r["split"] = split_name

    mode = "DRY RUN, nothing written" if args.dry_run else "live"
    print(f"\n=== split statistics [{mode}] | seed={args.seed} | 70/15/15 stratified by source class ===")
    all_rows = splits["train"] + splits["val"] + splits["test"]
    print_stats("all", all_rows)
    for split_name in ("train", "val", "test"):
        print_stats(split_name, splits[split_name])

    # 2) and write to disk
    if not args.dry_run:
        print(f"\nprocessing images into {output_root}/ ({args.size}x{args.size} greyscale PNG)...")
        written = 0
        failed = 0
        for s in all_rows:
            dst = output_root / s["path"]
            if write_image(s, dst, args.size):
                written += 1
            else:
                failed += 1
        # the full manifest plus the three split CSVs
        output_root.mkdir(parents=True, exist_ok=True)
        write_csv(output_root / "manifest.csv", all_rows)
        for split_name in ("train", "val", "test"):
            write_csv(output_root / f"{split_name}.csv", splits[split_name])
        print(f"images written: {written} succeeded, {failed} failed.")
        print(
            f"wrote {output_root}/manifest.csv, train.csv, val.csv, test.csv"
        )
        if failed:
            print(
                f"[note] {failed} images failed to read and were skipped, but they are still "
                f"listed in the CSVs. Check the warnings above or re-run before training.",
                file=sys.stderr,
            )

    # 4) class imbalance note
    total_pos = sum(1 for s in samples if s["label"] == 1)
    total_neg = len(samples) - total_pos
    print(
        f"\n[note] the classes are imbalanced: {total_pos} positive against {total_neg} "
        f"negative (ratio about {total_pos / total_neg:.2f}). Handle this with class weights "
        f"during training rather than resampling here, which would change the distribution."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
