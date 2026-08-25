"""Gatekeeper edge inference. A shared, importable module rather than a script.

Wraps the int8 .tflite gatekeeper: load, preprocess a camera frame, predict (label, score,
latency). Both benchmark_latency.py and cascade.py use this module, so preprocessing and
inference have exactly one definition.

Design choices, written down so the setup can be audited:
- The runtime is LiteRT (`ai_edge_litert.Interpreter`) when available, which is the recommended
  runtime on Raspberry Pi (`tf.lite.Interpreter` is slated for deprecation from TF 2.20). If
  ai_edge_litert is not installed on the Pi, it falls back to `tensorflow.lite.Interpreter`,
  which also lets this module be tested on a laptop.
- Preprocessing matches training and export exactly: single-channel greyscale, resize to 96x96
  with INTER_AREA (built for downscaling, best anti-aliasing), normalise pixels to [0,1] float
  (matching train.py's convert_image_dtype), then quantise to int8 using the model's own input
  quantisation parameters (scale, zero_point). Those parameters are read from the interpreter
  and never hardcoded, so the values fed to the int8 graph match what the converter calibrated
  against.
- The threshold is not baked into predict. predict returns the argmax label plus the p(record)
  score, leaving the caller (cascade.py) to apply a configurable threshold, which keeps the
  FN/FP operating point adjustable.

Dependencies on the Pi: ai_edge_litert (or tensorflow), opencv-headless (cv2), numpy.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

# Gatekeeper input side length, matching training and export: 96x96x1 greyscale.
INPUT_SIZE = 96

# Default decision threshold, applied to p(record) = softmax[1]. 0.5 is equivalent to argmax.
# The deployment operating point tuned for v4_mvp is about 0.55 (see models/README). The
# threshold is the FN/FP knob and is meant to be overridden by the caller; this is only a safe
# default.
DEFAULT_THRESHOLD = 0.5


def _make_interpreter(model_path: str):
    """Prefer LiteRT, fall back to tf.lite. Returns an interpreter that has not been
    allocated yet."""
    try:
        from ai_edge_litert.interpreter import Interpreter  # recommended runtime on the Pi
        return Interpreter(model_path=model_path)
    except ImportError:
        # Fallback for laptop testing, or when ai_edge_litert is not installed. Functionally
        # equivalent.
        import tensorflow as tf  # noqa: local import, so the Pi does not hard-depend on all of TF
        return tf.lite.Interpreter(model_path=model_path)


def load_model(path: str | Path):
    """Load an int8 .tflite, allocate tensors, and return an interpreter ready for
    inference."""
    interp = _make_interpreter(str(path))
    interp.allocate_tensors()
    return interp


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Normalise a camera frame to 2D greyscale uint8.

    Accepts 2D greyscale, such as the Y plane of Picamera2's YUV420 lores stream, which is the
    cheapest option, or 3D colour.

    Note that Picamera2 gives RGB by default, so COLOR_RGB2GRAY applies the standard luma
    (0.299R + 0.587G + 0.114B), which closely matches the luma from decoding a PNG to a single
    channel. If the source were actually BGR, only the R and B weights swap, which is
    negligible for a low-resolution greyscale gatekeeper. The best option is still for the
    caller to supply greyscale directly (see cascade.py).
    """
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    raise ValueError(f"unrecognised frame shape: {frame.shape}")


def preprocess(frame: np.ndarray, interpreter) -> np.ndarray:
    """Camera frame to (1,96,96,1) int8, matching training and export.

    The interpreter is needed to read the input tensor's quantisation parameters (scale,
    zero_point) rather than hardcoding them.
    """
    gray = _to_gray(frame)
    # INTER_AREA is the recommended interpolation for shrinking; it is equivalent to area
    # averaging and anti-aliases well, which suits a low-resolution greyscale input.
    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0  # to [0,1], matching convert_image_dtype in training

    in_detail = interpreter.get_input_details()[0]
    scale, zero_point = in_detail["quantization"]  # the model's own input quantisation parameters
    if scale == 0:  # an int8 model should never be 0; as a safeguard, feed float models float
        q = x
    else:
        # q = round(real / scale) + zero_point, clamped to the int8 range
        q = np.round(x / scale + zero_point)
        q = np.clip(q, -128, 127).astype(np.int8)
    return q.reshape(1, INPUT_SIZE, INPUT_SIZE, 1)


def predict(interpreter, frame: np.ndarray) -> tuple[int, float, float]:
    """Run the gatekeeper on one frame.

    Returns (label, score, latency_ms):
      label      argmax class, 0 for do-not-record and 1 for record
      score      p(record) = softmax[1], dequantised, for the caller to threshold
      latency_ms pure inference time (set_tensor, invoke, get_tensor), excluding
                 preprocessing, so cascade's hit log and the benchmark are directly
                 comparable.
    """
    x = preprocess(frame, interpreter)
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    t0 = time.perf_counter()
    interpreter.set_tensor(in_detail["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(out_detail["index"])[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    o_scale, o_zp = out_detail["quantization"]
    probs = (y.astype(np.float32) - o_zp) * o_scale if o_scale else y.astype(np.float32)
    label = int(np.argmax(probs))
    score = float(probs[1])  # p(record)
    return label, score, latency_ms
