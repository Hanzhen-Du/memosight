#!/usr/bin/env python3
"""Pure inference latency benchmark for the gatekeeper, to be run on the Raspberry Pi.

Measures inference alone: a random int8 input is fed straight to invoke, with no camera capture
and no preprocessing, so the number reflects only the model's compute cost on that hardware and
is comparable across devices and models. (There was no standalone benchmark script before this;
timing had only ever been an ad-hoc probe.)

Method: 20 warm-up iterations to let caches, threads and clocks settle, then 200 timed
iterations, reporting mean, p50, p95, p99 and throughput in inferences per second. The random
input is generated from the interpreter's own input dtype and shape.

Dependencies: ai_edge_litert (or tensorflow), numpy. Reuses infer.load_model so the runtime
matches.

Example, on the Pi:
  python3 hardware/benchmark_latency.py \
      --model ~/dev/memosight/models/gatekeeper_v4_mvp_int8.tflite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import infer  # same-directory module; run from hardware/

WARMUP = 20
TIMED = 200


def random_input(interpreter, seed: int = 0) -> np.ndarray:
    """Build a random input matching the model input tensor's dtype and shape."""
    d = interpreter.get_input_details()[0]
    shape = tuple(int(s) for s in d["shape"])
    dtype = np.dtype(d["dtype"])
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(info.min, info.max + 1, size=shape, dtype=dtype)
    return rng.random(size=shape).astype(dtype)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pure inference latency benchmark for the gatekeeper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model", type=Path,
        default=Path.home() / "dev/memosight/models/gatekeeper_v4_mvp_int8.tflite",
    )
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--runs", type=int, default=TIMED)
    args = ap.parse_args()

    print(f"loading model: {args.model}")
    interp = infer.load_model(args.model)
    in_detail = interp.get_input_details()[0]
    print(f"input: shape={list(in_detail['shape'])} dtype={np.dtype(in_detail['dtype']).name}")

    x = random_input(interp)
    idx = in_detail["index"]

    # time.perf_counter is a high-resolution monotonic clock. Time each iteration separately
    # to avoid amortisation error.
    import time
    print(f"warm-up, {args.warmup} iterations ...")
    for _ in range(args.warmup):
        interp.set_tensor(idx, x)
        interp.invoke()

    print(f"timing, {args.runs} iterations ...")
    lat = np.empty(args.runs, dtype=np.float64)
    for i in range(args.runs):
        t0 = time.perf_counter()
        interp.set_tensor(idx, x)
        interp.invoke()
        lat[i] = (time.perf_counter() - t0) * 1000.0  # ms

    mean = lat.mean()
    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
    thr = 1000.0 / mean if mean > 0 else float("inf")

    print("\n===== pure inference latency (ms) =====")
    print(f"  samples    : {args.runs}")
    print(f"  mean   : {mean:.3f}")
    print(f"  p50    : {p50:.3f}")
    print(f"  p95    : {p95:.3f}")
    print(f"  p99    : {p99:.3f}")
    print(f"  min/max: {lat.min():.3f} / {lat.max():.3f}")
    print(f"  throughput : {thr:.1f} inferences/s")
    print("\nNote: this is inference only. A real cascade tick also pays for camera capture and\n      preprocessing on top of this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
