#!/usr/bin/env python3
"""Gatekeeper evaluation: recompute the full metric set in one command (accuracy, precision,
recall, F1, confusion matrix, FN, FP).

Loads a saved .keras model and a split CSV and computes the full metrics. The positive class is
1, meaning record.
The decision threshold is configurable, applied to the softmax p(record), defaulting to 0.5
which is equivalent to argmax. Phase 3 uses it to push FN down.
--pr-sweep sweeps a range of thresholds and prints a precision/recall/F1/FN/FP table.

Running this after any model change re-scores everything with a fixed, comparable measurement.

Dependencies: tensorflow, numpy, pandas. Nothing new. Reuses train.py's data loading.

Examples:
  .venv/bin/python scripts/evaluate.py --model models/gatekeeper_v1.keras \
      --val-csv data/processed/dedup_val.csv --test-csv data/processed/dedup_test.csv
  .venv/bin/python scripts/evaluate.py --model models/xx.keras \
      --test-csv data/processed/dedup_test.csv --pr-sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from train import load_split, make_dataset


def metrics_from_probs(probs: np.ndarray, y_true: np.ndarray, threshold: float) -> dict:
    """Threshold p(record) = probs[:,1] and compute the full metrics. The positive class is 1."""
    p_pos = probs[:, 1]
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
        "n": total, "n_pos": int((y_true == 1).sum()), "n_neg": int((y_true == 0).sum()),
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fn_rate": round(fn / (tp + fn), 4) if (tp + fn) else 0.0,
        "fp_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def predict_split(model: tf.keras.Model, csv: Path, data_root: Path,
                  size: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    paths, labels = load_split(csv, data_root)
    ds = make_dataset(paths, labels, size, batch_size, shuffle=False, seed=0)
    probs = model.predict(ds, verbose=0)
    return probs, labels


def print_metrics(name: str, m: dict) -> None:
    print(f"\n===== {name} (threshold={m['threshold']}, positive class = 1 = record) =====")
    print(f"samples: {m['n']} ({m['n_pos']} positive / {m['n_neg']} negative)")
    print(f"Accuracy {m['accuracy']:.4f} | Precision {m['precision']:.4f} | "
          f"Recall {m['recall']:.4f} | F1 {m['f1']:.4f}")
    print(f"FN rate {m['fn_rate']:.4f} | FP rate {m['fp_rate']:.4f}")
    c = m["confusion"]
    print("confusion matrix:      predicted 0   predicted 1")
    print(f"  actual 0 (skip)      {c['tn']:>6}        {c['fp']:>6}")
    print(f"  actual 1 (record)    {c['fn']:>6}        {c['tp']:>6}")


def main() -> int:
    p = argparse.ArgumentParser(description="Recompute the full metric set for a gatekeeper model.")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/processed"))
    p.add_argument("--val-csv", type=Path, default=None)
    p.add_argument("--test-csv", type=Path, default=Path("data/processed/dedup_test.csv"))
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--pr-sweep", action="store_true", help="sweep thresholds and print a precision/recall/F1 table")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    model = tf.keras.models.load_model(args.model)
    print(f"model: {args.model} ({model.count_params():,} parameters)")

    out = {"model": str(args.model), "threshold": args.threshold}
    for split_name, csv in (("val", args.val_csv), ("test", args.test_csv)):
        if csv is None:
            continue
        probs, labels = predict_split(model, csv, args.data_root, args.size, args.batch_size)
        m = metrics_from_probs(probs, labels, args.threshold)
        tag = f"{split_name} ({'deciding' if split_name == 'test' else 'tuning'})"
        print_metrics(tag, m)
        out[split_name] = m

        if args.pr_sweep:
            print(f"\n----- {split_name} threshold sweep, looking for the point that minimises FN -----")
            print(f"{'thr':>5} {'prec':>7} {'recall':>7} {'F1':>7} {'FN':>7} {'FP':>7}")
            for t in np.arange(0.05, 1.0, 0.05):
                mm = metrics_from_probs(probs, labels, float(t))
                print(f"{t:>5.2f} {mm['precision']:>7.4f} {mm['recall']:>7.4f} "
                      f"{mm['f1']:>7.4f} {mm['fn_rate']:>7.4f} {mm['fp_rate']:>7.4f}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\nmetrics JSON written to {args.json_out}")
    print("\nRESULT " + json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
