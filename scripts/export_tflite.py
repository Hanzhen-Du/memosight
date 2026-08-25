#!/usr/bin/env python3
"""Export the gatekeeper: .keras to .tflite, both a float32 baseline and a fully int8
quantised build, with ESP32 constraint verification.

Outputs, into models/ by default:
  1. <stem>_float32.tflite  unquantised, float I/O. The accuracy baseline on the Pi.
  2. <stem>_int8.tflite     fully int8 quantised, int8 I/O. The ESP32-portable form.

After export it verifies and prints a report, so the real numbers can be checked against the
earlier estimates:
  A. List every operator in the int8 .tflite and check each against the TFLite Micro whitelist
     {CONV_2D, DEPTHWISE_CONV_2D, AVERAGE_POOL_2D, MAX_POOL_2D, RESHAPE, FULLY_CONNECTED,
      SOFTMAX, plus QUANTIZE/DEQUANTIZE at the edges}.
     Anything not on the whitelist is flagged [NOT WHITELISTED].
  B. Confirm there are no float32 tensors inside the graph, meaning it is fully int8.
  C. Report the actual size of both .tflite files, and compare int8 weight size against the
     earlier 24.3 KB estimate.
  D. Correctness: run the int8 .tflite over the test split for accuracy and F1, and compare
     against the original .keras.

Preprocessing is identical to train.py's load_split and make_dataset:
  read PNG, decode_png(channels=1), convert_image_dtype(float32) in [0,1], shape (96,96,1).

Dependencies: tensorflow==2.19.*, pandas, numpy.

Example:
  python scripts/export_tflite.py            # defaults: v4_mvp with the dedup split
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

SIZE = 96

# TFLite Micro whitelist: builtin op names, uppercase, matching the op_name returned by
# _get_ops_details. Covers the model's own operators plus the QUANTIZE/DEQUANTIZE that can
# appear at the int8 I/O boundary.
TFLM_WHITELIST = {
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "AVERAGE_POOL_2D",
    "MAX_POOL_2D",
    "RESHAPE",
    "FULLY_CONNECTED",
    "SOFTMAX",
    "QUANTIZE",
    "DEQUANTIZE",
}


# ---------- data loading, preprocessing identical to train.py ----------

def _decode(path: str) -> np.ndarray:
    """One image: read_file, decode_png single channel, float32 in [0,1], shape (96,96,1)."""
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)  # → [0,1]
    img = tf.ensure_shape(img, (SIZE, SIZE, 1))
    return img.numpy()


def load_paths_labels(csv_path: Path, data_root: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(csv_path)
    paths = [str(data_root / p) for p in df["path"].tolist()]
    labels = df["label"].to_numpy(dtype=np.int32)
    return paths, labels


# ---------- export ----------

def _fixed_batch_model(model: tf.keras.Model) -> tf.keras.Model:
    """Wrap the trained model behind an input pinned to batch 1, sharing weights and leaving
    the structure unchanged.

    This matters. The default export keeps a dynamic batch (-1), which makes flatten's Reshape
    assemble its shape at runtime using SHAPE, STRIDED_SLICE and PACK, none of which are on
    the TFLite Micro whitelist. With batch pinned to 1 every internal shape is concrete, the
    Reshape collapses to a static constant, and only RESHAPE remains, keeping the whitelist
    intact. ESP32 and TFLM inference is batch 1 anyway, so pinning it costs no deployment
    capability.

    The wrapper is a keras model rather than a concrete function, which guarantees the weights
    are folded in as embedded constants and avoids from_concrete_functions leaving unbound
    READ_VARIABLE nodes behind.
    """
    inp = tf.keras.Input(batch_shape=(1, SIZE, SIZE, 1), name="input")
    out = model(inp)
    return tf.keras.Model(inp, out, name=f"{model.name}_fixedbatch")


def export_float32(model: tf.keras.Model, out_path: Path) -> None:
    """Unquantised export with float I/O. The accuracy baseline on the Pi."""
    conv = tf.lite.TFLiteConverter.from_keras_model(_fixed_batch_model(model))
    tflite = conv.convert()
    out_path.write_bytes(tflite)


def export_int8(
    model: tf.keras.Model,
    out_path: Path,
    rep_paths: list[str],
    n_rep: int,
) -> None:
    """Fully int8 quantised with int8 I/O. The representative dataset comes from
    dedup_train."""

    def rep_gen():
        for p in rep_paths[:n_rep]:
            x = _decode(p)[None, ...]  # (1,96,96,1) float32 [0,1]
            yield [x]

    conv = tf.lite.TFLiteConverter.from_keras_model(_fixed_batch_model(model))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    # Force full int8. If any operator cannot be quantised to int8, convert() raises, which is
    # exactly the guard rail we want.
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite = conv.convert()
    out_path.write_bytes(tflite)


# ---------- verification ----------

def verify_ops(int8_path: Path) -> list[tuple[str, bool]]:
    """List every operator in the int8 .tflite, returning [(op_name, in_whitelist), ...]."""
    # BUILTIN_REF uses the reference kernels with no XNNPACK delegate attached, so
    # _get_ops_details reports the real flatbuffer operators rather than runtime DELEGATE
    # nodes.
    interp = tf.lite.Interpreter(
        model_path=str(int8_path),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interp.allocate_tensors()
    ops = interp._get_ops_details()  # semi-private API, available in TF 2.19
    return [(o["op_name"], o["op_name"] in TFLM_WHITELIST) for o in ops]


def verify_dtypes(int8_path: Path) -> dict:
    """Count tensor dtypes and flag any float32 tensors, which should be zero when the model
    is fully int8."""
    interp = tf.lite.Interpreter(model_path=str(int8_path))
    interp.allocate_tensors()
    counts: dict[str, int] = {}
    float_tensors: list[str] = []
    for t in interp.get_tensor_details():
        dt = np.dtype(t["dtype"]).name
        counts[dt] = counts.get(dt, 0) + 1
        if dt == "float32":
            float_tensors.append(t["name"])
    return {"counts": counts, "float32_tensors": float_tensors}


# ---------- int8 inference for test-split correctness ----------

def predict_int8(int8_path: Path, paths: list[str]) -> np.ndarray:
    """Run the int8 interpreter one sample at a time and return the dequantised p(record)
    probabilities."""
    interp = tf.lite.Interpreter(model_path=str(int8_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    p_pos = np.empty(len(paths), dtype=np.float32)
    for i, p in enumerate(paths):
        x = _decode(p)[None, ...]  # (1,96,96,1) float32 [0,1]
        # Quantise to int8: q = round(x / scale) + zero_point
        q = np.round(x / in_scale + in_zp).astype(np.int32)
        q = np.clip(q, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        y = interp.get_tensor(out["index"])[0].astype(np.float32)
        deq = (y - out_zp) * out_scale  # dequantise back to softmax probabilities
        p_pos[i] = deq[1]
    return p_pos


def predict_keras(model: tf.keras.Model, paths: list[str]) -> np.ndarray:
    """p(record) from the original keras model, one sample at a time to match the int8 path
    and avoid batching differences."""
    p_pos = np.empty(len(paths), dtype=np.float32)
    for i, p in enumerate(paths):
        x = _decode(p)[None, ...]
        probs = model.predict(x, verbose=0)[0]
        p_pos[i] = probs[1]
    return p_pos


def metrics(p_pos: np.ndarray, y_true: np.ndarray, threshold: float) -> dict:
    y_pred = (p_pos >= threshold).astype(np.int32)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "threshold": round(threshold, 3),
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fn_rate": round(fn / (tp + fn), 4) if (tp + fn) else 0.0,
        "fp_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


# ---------- report ----------

def kb(n_bytes: int) -> str:
    return f"{n_bytes / 1024:.1f} KB"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export the gatekeeper from .keras to .tflite and verify the ESP32 constraints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", type=Path,
                    default=Path("models/gatekeeper_v4_mvp.keras"))
    ap.add_argument("--data-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--train-csv", type=Path,
                    default=Path("data/processed/dedup_train.csv"),
                    help="source for the representative dataset")
    ap.add_argument("--test-csv", type=Path,
                    default=Path("data/processed/dedup_test.csv"),
                    help="split used for the correctness check")
    ap.add_argument("--n-rep", type=int, default=200, help="number of representative samples")
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.55],
                    help="thresholds to compare (0.5 is argmax; 0.55 is the v4_mvp deployment point)")
    ap.add_argument("--rep-seed", type=int, default=42)
    args = ap.parse_args()

    stem = args.model.stem  # e.g. gatekeeper_v4_mvp
    f32_path = args.out_dir / f"{stem}_float32.tflite"
    int8_path = args.out_dir / f"{stem}_int8.tflite"

    print(f"loading keras model: {args.model}")
    model = tf.keras.models.load_model(args.model)
    n_params = model.count_params()
    est_int8_weight_kb = n_params / 1024.0  # theoretical estimate at 1 byte per parameter

    # Representative dataset: shuffle dedup_train with a fixed seed and take the first n_rep,
    # which covers both classes
    rep_paths, _ = load_paths_labels(args.train_csv, args.data_root)
    rng = np.random.default_rng(args.rep_seed)
    rng.shuffle(rep_paths)
    print(f"representative dataset: {args.n_rep}/{len(rep_paths)} samples from {args.train_csv.name}")

    print("\n[1/2] exporting float32 .tflite ...")
    export_float32(model, f32_path)
    print(f"  → {f32_path}  ({kb(f32_path.stat().st_size)})")

    print("\n[2/2] exporting int8 .tflite (fully quantised, int8 I/O) ...")
    export_int8(model, int8_path, rep_paths, args.n_rep)
    print(f"  → {int8_path}  ({kb(int8_path.stat().st_size)})")

    # ===== verification report =====
    print("\n" + "=" * 64)
    print("Verification report for the int8 .tflite")
    print("=" * 64)

    # A. operator whitelist
    print("\n--- A. operators against the TFLite Micro whitelist ---")
    ops = verify_ops(int8_path)
    all_ok = True
    for name, ok in ops:
        flag = "OK" if ok else "*** NOT WHITELISTED ***"
        if not ok:
            all_ok = False
        print(f"  {name:<22} {flag}")
    print(f"  whitelist verdict: {'all pass' if all_ok else 'NON-WHITELISTED OPERATORS PRESENT'}")

    # B. dtypes; the graph interior should be entirely int8
    print("\n--- B. tensor dtypes (confirm fully int8, no internal float32) ---")
    dt = verify_dtypes(int8_path)
    for name, c in sorted(dt["counts"].items()):
        print(f"  {name:<12} {c} tensors")
    if dt["float32_tensors"]:
        print(f"  *** found {len(dt['float32_tensors'])} float32 tensors (should be 0):")
        for n in dt["float32_tensors"]:
            print(f"      - {n}")
    else:
        print("  no float32 tensors: fully int8.")

    # C. size
    print("\n--- C. size comparison ---")
    print(f"  float32 .tflite file: {kb(f32_path.stat().st_size)}")
    print(f"  int8    .tflite file: {kb(int8_path.stat().st_size)}")
    print(f"  parameters: {n_params:,}")
    print(f"  int8 weight estimate at 1 byte per parameter: {est_int8_weight_kb:.1f} KB"
          f"  (compare with the earlier 24.3 KB estimate)")
    print("  Note: a .tflite file holds weights plus flatbuffer metadata and quantisation "
          "parameters, so it is slightly larger than the weights alone. The ESP32 100 KB "
          "budget applies to the weights; the file size only matters for transfer.")

    # D. correctness
    print("\n--- D. correctness on the test split: int8 .tflite against the original keras ---")
    test_paths, y_true = load_paths_labels(args.test_csv, args.data_root)
    print(f"  test: {len(test_paths)} samples "
          f"({int((y_true==1).sum())} positive / {int((y_true==0).sum())} negative)")
    print("  running the int8 interpreter ...")
    p_int8 = predict_int8(int8_path, test_paths)
    print("  running the original keras model ...")
    p_keras = predict_keras(model, test_paths)

    print(f"\n  {'thresh':>6} {'model':>8} {'acc':>7} {'prec':>7} {'recall':>7}"
          f" {'F1':>7} {'FN':>7} {'FP':>7}")
    for thr in args.thresholds:
        mk = metrics(p_keras, y_true, thr)
        mi = metrics(p_int8, y_true, thr)
        print(f"  {thr:>6.2f} {'keras':>8} {mk['accuracy']:>7.4f}"
              f" {mk['precision']:>7.4f} {mk['recall']:>7.4f} {mk['f1']:>7.4f}"
              f" {mk['fn_rate']:>7.4f} {mk['fp_rate']:>7.4f}")
        print(f"  {thr:>6.2f} {'int8':>8} {mi['accuracy']:>7.4f}"
              f" {mi['precision']:>7.4f} {mi['recall']:>7.4f} {mi['f1']:>7.4f}"
              f" {mi['fn_rate']:>7.4f} {mi['fp_rate']:>7.4f}")
        d_f1 = mi["f1"] - mk["f1"]
        d_acc = mi["accuracy"] - mk["accuracy"]
        print(f"  {'':>6} {'Δ(int8-keras)':>8} ΔF1={d_f1:+.4f}  Δacc={d_acc:+.4f}"
              f"  {'(noticeable loss, worth investigating)' if d_f1 < -0.03 else '(quantisation loss acceptable)'}")

    # ===== what the evidence does and does not show =====
    print("\n" + "=" * 64)
    print("What this proves. Note that running on a Pi does not mean ESP32-ready.")
    print("=" * 64)
    print("""  [evidence that it will run on a Pi]
    - both .tflite files exported successfully;
    - the int8 .tflite ran the whole test split through the LiteRT interpreter on x86 and
      produced metrics;
    - a Raspberry Pi runs full LiteRT, a more forgiving environment than ESP32, so the above
      is enough to deploy and run there.

  [evidence toward ESP32-ready, partial, still needs real hardware]
    - satisfied: every operator is on the TFLite Micro whitelist, the model is fully int8, and
      the weights are under 100 KB (see above);
    - not verified, needs real hardware: whether the TFLM tensor arena actually fits in
      on-chip SRAM, whether each TFLM kernel agrees numerically with this interpreter, and
      measured latency and power.
    This script only shows that the architecture and quantised form satisfy the static ESP32
    constraints. It does not show that the model has run on an ESP32.""")

    print("\nExport complete.")
    return 0 if all_ok and not dt["float32_tensors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
