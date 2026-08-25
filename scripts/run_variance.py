#!/usr/bin/env python3
"""Variance check: re-split and retrain across 5 seeds, reporting mean and standard deviation
of accuracy and F1.

This answers whether 0.80 is a stable value or one lucky run. It works on the deduplicated
manifest, because leakage has to be eliminated first or the variance measurement is itself
contaminated. For each seed: stratified re-split of the deduplicated manifest, retrain with the
frozen best configuration, evaluate on test.

The frozen best configuration, from the first round of fixes:
  bn_momentum=0.9, patience=15, start_from_epoch=20, epochs=80,
  augment=True, class_weight=balanced, lr=1e-3, monitor=val_loss (restore best).

Dependencies: tensorflow, numpy, pandas. Nothing new. Reuses the model, train and
dedup_resplit modules.

Examples:
  .venv/bin/python scripts/run_variance.py --seeds 42 1 7 123 2024
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from model import build_model
from train import make_dataset, compute_class_weight
from dedup_resplit import do_split
from evaluate import metrics_from_probs

BEST = dict(bn_momentum=0.9, patience=15, start_from_epoch=20, epochs=80,
            lr=1e-3, batch_size=32, size=96)


def run_one(seed: int, manifest: Path, data_root: Path, out_dir: Path) -> dict:
    splits = do_split(manifest, out_dir, prefix=f"var_s{seed}_", seed=seed, quiet=True)
    tf.keras.utils.set_random_seed(seed)

    def to_ds(df, shuffle, augment):
        paths = [str(data_root / p) for p in df["path"]]
        labels = df["label"].to_numpy(dtype=np.int32)
        return make_dataset(paths, labels, BEST["size"], BEST["batch_size"],
                            shuffle=shuffle, seed=seed, augment=augment), labels

    train_ds, _ = to_ds(splits["train"], True, True)
    val_ds, _ = to_ds(splits["val"], False, False)
    test_ds, test_labels = to_ds(splits["test"], False, False)
    cw = compute_class_weight(splits["train"]["label"].to_numpy())

    model = build_model(BEST["size"], bn_momentum=BEST["bn_momentum"])
    model.compile(optimizer=tf.keras.optimizers.Adam(BEST["lr"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cb = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=BEST["patience"],
        start_from_epoch=BEST["start_from_epoch"],
        restore_best_weights=True, verbose=0)]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=BEST["epochs"],
                     class_weight=cw, callbacks=cb, verbose=0)

    probs = model.predict(test_ds, verbose=0)
    m = metrics_from_probs(probs, test_labels, 0.5)
    m["seed"] = seed
    m["stopped_epoch"] = len(hist.history["loss"])
    print(f"[seed={seed}] epochs={m['stopped_epoch']:>2} | "
          f"acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
          f"recall={m['recall']:.4f} prec={m['precision']:.4f} "
          f"FN={m['fn_rate']:.4f} FP={m['fp_rate']:.4f}")
    return m


def main() -> int:
    p = argparse.ArgumentParser(description="5-seed variance check on the deduplicated manifest.")
    p.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_dedup.csv"))
    p.add_argument("--data-root", type=Path, default=Path("data/processed"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7, 123, 2024])
    p.add_argument("--json-out", type=Path, default=Path("docs/results/variance_results.json"))
    args = p.parse_args()

    print(f"variance check: {len(args.seeds)} seeds over {args.manifest}; test is the deciding split")
    results = [run_one(s, args.manifest, args.data_root, args.out_dir) for s in args.seeds]

    def stat(key):
        vals = np.array([r[key] for r in results], dtype=float)
        return float(vals.mean()), float(vals.std())

    summary = {}
    print("\n===== summary (test split, positive class = 1) =====")
    for key in ("accuracy", "f1", "recall", "precision", "fn_rate", "fp_rate"):
        mu, sd = stat(key)
        summary[key] = {"mean": round(mu, 4), "std": round(sd, 4)}
        print(f"  {key:<10} {mu:.4f} ± {sd:.4f}")

    out = {"seeds": args.seeds, "config": BEST, "per_seed": results, "summary": summary}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nresults written to {args.json_out}")
    print("RESULT " + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
