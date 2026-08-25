#!/usr/bin/env python3
"""Extract video frames and batch-downscale to greyscale. A data preparation tool.

Turns raw material (lecture videos, slide and whiteboard screenshots) into the low-resolution
greyscale images the gatekeeper trains on, while keeping an untouched full-resolution master
copy.

Input can be either:
  - a video file, sampled at a fixed time interval
  - a folder of images, walked for the common image formats

Output, under <output_root>/<run timestamp>/:
  - raw/   full-resolution masters (video frames at original resolution in colour, images
           copied verbatim)
  - gray/  low-resolution greyscale (converted to greyscale first, then resized)

Filenames carry the run timestamp and an incrementing sequence number, which prevents
overwrites and keeps raw/ and gray/ in one-to-one correspondence.

Dependencies: opencv-python (see requirements.txt).

Examples:
  python3 scripts/extract_frames.py ./ppt_screenshots/ --size 128 128
  python3 scripts/extract_frames.py lecture.mp4 --interval 5 --output-root data/processed
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit(
        "Missing dependency: opencv-python. Install it with: pip install opencv-python\n"
        "(It is listed in requirements.txt; this script will not install it for you.)"
    )

# Default sampling interval for video, in seconds. Lecture, whiteboard and slide content
# changes slowly, so 2 seconds balances catching page turns against collecting duplicates.
DEFAULT_INTERVAL = 2.0
# Default target size (width, height) for the greyscale output. 96x96 is the standard
# greyscale input for TinyML Visual Wake Words, which suits the portability, int8 quantisation
# and Raspberry Pi deployment route.
DEFAULT_SIZE = (96, 96)
# Default output root.
DEFAULT_OUTPUT_ROOT = "data/processed"

# Extensions treated as images when the input is a folder.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# Extensions treated as video when the input is a file.
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video frames and batch-downscale to greyscale, producing gatekeeper "
                    "training images while keeping full-resolution masters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="input: a video file, or a folder of images.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help="sampling interval in seconds. Applies to video input only.",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=list(DEFAULT_SIZE),
        help="target size of the greyscale output: width height, in pixels.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="output root; a subdirectory named after the run timestamp is created under it.",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="do not keep full-resolution masters. Only meaningful for video; image input never "
             "modifies the originals, it just skips copying them.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="reduce progress output.",
    )
    return parser.parse_args(argv)


def classify_input(input_path: Path) -> str:
    """Decide whether the input is 'video' or 'imagedir', exiting with an error otherwise."""
    if input_path.is_dir():
        return "imagedir"
    if input_path.is_file():
        if input_path.suffix.lower() in VIDEO_EXTS:
            return "video"
        if input_path.suffix.lower() in IMAGE_EXTS:
            # A single image is handled as image input, as a folder containing one file.
            return "imagedir"
        sys.exit(
            f"unrecognised file type: {input_path.suffix}\n"
            f"supported video extensions: {sorted(VIDEO_EXTS)}\n"
            f"supported image extensions: {sorted(IMAGE_EXTS)}"
        )
    sys.exit(f"input does not exist: {input_path}")


def make_output_dirs(output_root: Path, run_stamp: str, keep_raw: bool) -> tuple[Path, Path | None]:
    """Create <output_root>/<run_stamp>/{raw,gray} and return (gray_dir, raw_dir)."""
    run_dir = output_root / run_stamp
    gray_dir = run_dir / "gray"
    gray_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = None
    if keep_raw:
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
    return gray_dir, raw_dir


def to_low_gray(image, size: tuple[int, int]):
    """Convert to greyscale then resize to the target size, given as (width, height)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.resize takes dsize as (width, height). INTER_AREA gives better quality when shrinking.
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def process_video(
    input_path: Path,
    gray_dir: Path,
    raw_dir: Path | None,
    interval: float,
    size: tuple[int, int],
    run_stamp: str,
    quiet: bool,
) -> int:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        sys.exit(f"cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        sys.exit(f"cannot read the frame rate, so time-based sampling is impossible: {input_path}")

    step = max(1, round(fps * interval))  # sample one frame every `step` frames
    if not quiet:
        print(f"video fps={fps:.2f}, interval {interval}s, so one frame every {step}")

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            name = f"{run_stamp}_{saved:06d}.png"
            if raw_dir is not None:
                cv2.imwrite(str(raw_dir / name), frame)
            cv2.imwrite(str(gray_dir / name), to_low_gray(frame, size))
            saved += 1
            if not quiet and saved % 50 == 0:
                print(f"  extracted {saved} frames...")
        frame_idx += 1

    cap.release()
    return saved


def process_imagedir(
    input_path: Path,
    gray_dir: Path,
    raw_dir: Path | None,
    size: tuple[int, int],
    run_stamp: str,
    quiet: bool,
) -> int:
    if input_path.is_file():
        images = [input_path]
    else:
        images = sorted(
            p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
    if not images:
        sys.exit(f"no images found in the folder: {input_path}")

    saved = 0
    skipped = 0
    for src in images:
        image = cv2.imread(str(src))
        if image is None:
            skipped += 1
            if not quiet:
                print(f"  skipping unreadable file: {src}")
            continue
        # The master keeps its original extension and is copied verbatim, with no re-encoding
        # and no loss; the greyscale output is always png.
        name_stem = f"{run_stamp}_{saved:06d}"
        if raw_dir is not None:
            shutil.copy2(src, raw_dir / f"{name_stem}{src.suffix.lower()}")
        cv2.imwrite(str(gray_dir / f"{name_stem}.png"), to_low_gray(image, size))
        saved += 1
        if not quiet and saved % 50 == 0:
            print(f"  processed {saved} images...")

    if skipped and not quiet:
        print(f"  skipped {skipped} unreadable files in total.")
    return saved


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser()
    size = (int(args.size[0]), int(args.size[1]))
    keep_raw = not args.no_raw

    kind = classify_input(input_path)

    # Run timestamp to the second, used as this run's output subdirectory name and filename
    # prefix, which prevents overwrites.
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).expanduser()
    gray_dir, raw_dir = make_output_dirs(output_root, run_stamp, keep_raw)

    if not args.quiet:
        print(f"input: {input_path} ({'video' if kind == 'video' else 'images'})")
        print(f"target greyscale size: {size[0]}x{size[1]}")
        print(f"output directory: {(output_root / run_stamp).resolve()}")
        print(f"keep full-resolution masters: {'yes' if keep_raw else 'no'}")
        print("-" * 40)

    if kind == "video":
        saved = process_video(
            input_path, gray_dir, raw_dir, args.interval, size, run_stamp, args.quiet
        )
    else:
        saved = process_imagedir(input_path, gray_dir, raw_dir, size, run_stamp, args.quiet)

    print("-" * 40)
    print(f"done: {saved} greyscale images written to {gray_dir.resolve()}")
    if raw_dir is not None:
        print(f"      masters written to {raw_dir.resolve()}")


if __name__ == "__main__":
    main()
