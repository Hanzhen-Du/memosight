#!/usr/bin/env python3
"""Minimal Camera Module 3 self-test using Picamera2. The first script to run once the camera
is physically connected.

It does three things:
  1. Grabs a few frames to confirm the camera produces images with sensible shapes.
  2. Saves one full-resolution JPEG (full_res.jpg).
  3. Saves one 96x96 greyscale PNG (gray_96.png), which is exactly what the gatekeeper sees,
     so you can check by eye whether text or a screen is still recognisable at that resolution.

No inference and no loop. This only verifies that the camera path works.

Dependencies: picamera2, opencv-headless (cv2), numpy.
  picamera2 ships with Raspberry Pi OS but is not always preinstalled. If it is missing:
      sudo apt install -y python3-picamera2
  (Picamera2 depends on libcamera; the apt package pulls in the system dependencies, so pip is
  not recommended.)

Example, on the Pi:
  python3 hardware/camera_test.py --outdir hardware/captures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:  # give an actionable fix rather than a bare traceback
    raise SystemExit(
        "picamera2 not found. On Raspberry Pi OS, install it with:\n"
        "    sudo apt install -y python3-picamera2\n"
        "(This also installs libcamera and the other system dependencies. Do not use pip.)"
    )

INPUT_SIZE = 96


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Camera Module 3 self-test using Picamera2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--outdir", type=Path, default=Path("hardware/captures"))
    ap.add_argument("--frames", type=int, default=5, help="how many frames to grab before saving")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    picam = Picamera2()
    # Still-capture configuration; the main stream runs at full resolution (None means the
    # sensor's default full frame). RGB888 makes saving a JPEG straightforward.
    config = picam.create_still_configuration(main={"format": "RGB888"})
    picam.configure(config)
    picam.start()
    try:
        frame = None
        for i in range(args.frames):  # discard the first frames so AE and AWB converge
            frame = picam.capture_array("main")
            print(f"frame {i}: shape={frame.shape} dtype={frame.dtype}")
        assert frame is not None

        # 1) Full-resolution JPEG. Picamera2 gives RGB and cv2 writes BGR, so convert.
        full_path = args.outdir / "full_res.jpg"
        cv2.imwrite(str(full_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # 2) What the gatekeeper actually sees: greyscale, resized to 96x96 with INTER_AREA
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray96 = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        gray_path = args.outdir / "gray_96.png"
        cv2.imwrite(str(gray_path), gray96)

        print(f"\nfull-resolution frame: {frame.shape} -> {full_path}")
        print(f"gatekeeper input: {gray96.shape} (96x96 greyscale) -> {gray_path}")
        print("Camera path OK. Next step: cascade.py.")
    finally:
        picam.stop()  # make sure the camera is released
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
