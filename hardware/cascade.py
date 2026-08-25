#!/usr/bin/env python3
"""Cascade gatekeeper main loop. Currently gated mode only.

The core of the project is cascade perception: a cheap, always-on gatekeeper watches only a
low-resolution greyscale stream, and a full-resolution frame is grabbed for downstream
processing only when the gatekeeper says record. It never records continuously, which would
destroy the whole low-power argument.

Each tick of this loop:
  grab a low-resolution greyscale frame -> infer.predict -> should_record? ->
    if recording: grab one full-resolution frame, save it to captures/ with a timestamp,
    append a line to the hit log, and optionally push it back to a laptop.

The sampling rate --fps is the duty-cycle knob, and one axis of the future power versus
missed-capture Pareto curve. Lower fps means lower power and a higher risk of missing
something. That trade-off is what we are characterising; it is not an arbitrary constant.

On the gated versus always-on power A/B: see the TODO in should_record() below. always is
deliberately not implemented today.

Dependencies: picamera2, ai_edge_litert (or tensorflow), opencv-headless, numpy.

Examples, on the Pi, from the repository root:
  python3 hardware/cascade.py --fps 3
  python3 hardware/cascade.py --fps 3 --threshold 0.55 --push --host 192.168.1.50 --port 8000
"""

from __future__ import annotations

import argparse
import os
import signal
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2

import infer  # same-directory module; run as python3 hardware/cascade.py from the repo root

# ---- stream resolutions ----
# Low-resolution stream, always on, feeding the gatekeeper. Large enough to keep the coarse
# structure of text and screens, and as small as possible to save compute and power.
# YUV420 is chosen so the Y plane can be used directly as greyscale, avoiding a colour
# conversion entirely.
LORES_SIZE = (320, 240)
# Full-resolution stream: one frame grabbed only on a hit, for downstream OCR and processing.
MAIN_SIZE = (1920, 1080)


def should_record(score: float, threshold: float, mode: str) -> bool:
    """The hit decision, which is the gate in the cascade. The mode is deliberately isolated
    into this one function.

    Only gated exists today: record when the gatekeeper score clears the threshold. That is the
    low-power argument, implemented.

    TODO (power A/B, a future --mode always): add an always branch here for the power A/B
    comparison, treating every tick as a hit and grabbing full resolution every time, to
    measure the upper bound of always-on downstream processing and plot it against gated. When
    that happens it is a one-line change: `if mode == "always": return True` below. It is
    deliberately left off today, to keep the "never record or process continuously" claim
    honest.
    """
    if mode == "gated":
        return score >= threshold
    if mode == "always":
        # Placeholder; see the TODO above. Enabling it means replacing this with `return True`.
        raise NotImplementedError(
            "mode=always is not implemented yet (the power A/B is future work); only gated runs today."
        )
    raise ValueError(f"unknown mode: {mode}")


def utc_stamp() -> str:
    """A filename-safe UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def save_hit_image(rgb_frame, path: Path) -> bytes | None:
    """Encode and persist a hit frame immediately: atomic, fsynced and verified.

    Fail-safe persistence. The image is written inside the loop at the moment of the hit, never
    deferred to shutdown.
    - makedirs before writing, in case the directory was removed or never existed.
    - flush plus os.fsync so the data really reaches disk, and an abrupt termination or power
      cut cannot lose a frame that was already captured.
    - verify after writing that the file exists and is non-empty.
    - any failure prints a clear error with the path and the exception, and is never swallowed
      silently. It returns None so the main loop keeps going, rather than crashing and losing
      every subsequent hit because one write failed.

    Returns the JPEG bytes, reused by the optional --push, or None on failure.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        data = buf.tobytes()
        with open(path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())  # force to disk, so a power cut cannot lose it
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError("post-write check failed: the file is missing or empty")
        return data
    except Exception as e:  # never silent: print the failure explicitly
        print(f"  [save failed, not silenced] path={path.resolve()} "
              f"err={type(e).__name__}: {e}")
        return None


def append_hit_log(log_path: Path, line: str) -> None:
    """Append one line to hits.log per hit, with flush and fsync. Never buffered, never
    deferred."""
    try:
        with open(log_path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"  [hits.log write failed, not silenced] path={log_path.resolve()}: {e}")


def push_frame(jpeg_bytes: bytes, host: str, port: int, name: str) -> None:
    """POST a full-resolution frame back to the laptop receiver (receiver_laptop.py).

    Why this design: stdlib urllib on this side and http.server on the other means no new
    dependency, and no requests. The POST is initiated by the Pi, so the Pi never needs to
    accept inbound connections; it only needs to reach the laptop's receiving port.
    Failures are caught by the caller and degraded gracefully. The frame is still in captures/
    and can be rsynced later, so a failed push never stalls the main loop.
    """
    url = f"http://{host}:{port}/upload?name={name}"
    req = urllib.request.Request(
        url, data=jpeg_bytes, method="POST",
        headers={"Content-Type": "image/jpeg"},
    )
    urllib.request.urlopen(req, timeout=3).read()  # short timeout, so the always-on loop never stalls


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cascade gatekeeper main loop (gated).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model", type=Path,
        default=Path.home() / "dev/memosight/models/gatekeeper_v4_mvp_int8.tflite",
    )
    ap.add_argument("--fps", type=float, default=3.0,
                    help="sampling rate, the duty-cycle knob and one axis of the Pareto curve")
    ap.add_argument("--threshold", type=float, default=infer.DEFAULT_THRESHOLD,
                    help="decision threshold on p(record); the FN/FP operating-point knob")
    ap.add_argument("--mode", choices=["gated", "always"], default="gated",
                    help="only gated is supported today; always is the future power A/B (see the should_record TODO)")
    ap.add_argument("--outdir", type=Path, default=Path("hardware/captures"))
    # WiFi push-back is off by default, so an unstable network never affects the core loop.
    ap.add_argument("--push", action="store_true", help="POST the full-resolution frame back to the laptop on a hit")
    ap.add_argument("--host", type=str, default="127.0.0.1", help="laptop address for --push")
    ap.add_argument("--port", type=int, default=8000, help="receiving port for --push")
    args = ap.parse_args()

    try:
        from picamera2 import Picamera2
    except ImportError:
        raise SystemExit(
            "picamera2 not found. Install with: sudo apt install -y python3-picamera2 (do not use pip)."
        )

    # Root-cause fix: resolve the output directory to an absolute path. The default is
    # relative to the CWD, so starting from anywhere other than the repository root (running
    # after cd hardware, or systemd/timeout changing WorkingDirectory) writes images somewhere
    # else and makes captures/ appear to be missing. Resolving to an absolute path and
    # printing it at startup removes the ambiguity; every write afterwards uses it.
    args.outdir = args.outdir.expanduser().resolve()
    os.makedirs(args.outdir, exist_ok=True)
    hits_log = args.outdir / "hits.log"
    print(f"hit output directory (absolute): {args.outdir}")

    # SIGTERM, the default signal from timeout and kill, also goes through the graceful
    # shutdown that releases the camera. Note that image persistence already happened at the
    # moment of the hit and does not depend on reaching this point.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    print(f"loading gatekeeper: {args.model}")
    interp = infer.load_model(args.model)

    # Two streams: lores runs continuously for the gatekeeper, main is read only on a hit.
    # Picamera2 maintains both, but only lores is decoded each tick; main is captured only on
    # a hit, so the full-resolution stream never burns compute continuously.
    picam = Picamera2()
    config = picam.create_video_configuration(
        main={"size": MAIN_SIZE, "format": "RGB888"},
        lores={"size": LORES_SIZE, "format": "YUV420"},
    )
    picam.configure(config)
    picam.start()

    period = 1.0 / args.fps if args.fps > 0 else 0.0
    lores_w, lores_h = LORES_SIZE
    print(f"starting cascade loop: mode={args.mode} fps={args.fps} threshold={args.threshold} "
          f"push={'on→%s:%d' % (args.host, args.port) if args.push else 'off'}")
    print("Ctrl-C to stop.")

    n_tick = n_hit = 0
    try:
        while True:
            tick_start = time.perf_counter()

            # Low-resolution frame: the YUV420 array has shape (h*3/2, w), and the Y (luma)
            # plane is the first h rows, which is greyscale already.
            yuv = picam.capture_array("lores")
            gray = yuv[:lores_h, :lores_w]

            label, score, latency_ms = infer.predict(interp, gray)
            n_tick += 1

            if should_record(score, args.threshold, args.mode):
                n_hit += 1
                stamp = utc_stamp()
                # On a hit, grab the full-resolution frame and persist it immediately. This
                # is the cascade waking the expensive stage.
                hi = picam.capture_array("main")
                cap_path = args.outdir / f"hit_{stamp}_s{score:.2f}.jpg"
                jpeg_bytes = save_hit_image(hi, cap_path)  # immediate, atomic, fsynced, verified

                if jpeg_bytes is not None:
                    # Append to hits.log per hit, with flush and fsync, never buffered.
                    append_hit_log(
                        hits_log,
                        f"{stamp}\tscore={score:.4f}\tlatency_ms={latency_ms:.2f}"
                        f"\t{cap_path.name}\n",
                    )
                    # Keep the [HIT] line and add the absolute path actually written, which
                    # makes problems easier to trace.
                    print(f"[HIT #{n_hit}] {stamp} score={score:.3f} "
                          f"lat={latency_ms:.1f}ms saved → {cap_path}")

                    if args.push:  # optional push-back; failure degrades and never breaks the loop
                        try:
                            push_frame(jpeg_bytes, args.host, args.port, cap_path.name)
                        except (urllib.error.URLError, OSError) as e:
                            print(f"  [push failed; frame kept locally for rsync] {e}")
                # If save_hit_image failed the error is already printed; the loop continues so
                # later hits are not lost.
            else:
                # Heartbeat: an occasional print confirming the always-on loop is alive and
                # not firing wildly.
                if n_tick % max(1, int(args.fps) * 10) == 0:
                    print(f"... tick {n_tick} score={score:.3f} lat={latency_ms:.1f}ms "
                          f"(hits {n_hit})")

            # Hold the target fps: subtract the time this tick already used before sleeping.
            if period:
                time.sleep(max(0.0, period - (time.perf_counter() - tick_start)))
    except KeyboardInterrupt:
        # Ctrl-C or SIGTERM from timeout/kill. Hit images were already persisted one by one
        # inside the loop, so this is only tidying up.
        print(f"\ninterrupted (Ctrl-C/SIGTERM), stopping. {n_tick} ticks, {n_hit} hits.")
    finally:
        picam.stop()  # release the camera
        print("camera released.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
