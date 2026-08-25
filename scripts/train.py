#!/usr/bin/env python3
"""Gatekeeper CNN training. Parameterised, so iterating never requires editing the code.

Steps:
  1. Load 96x96 greyscale PNGs from train/val(/test).csv with tf.data, normalising pixels to
     [0,1].
  2. Optional light augmentation: horizontal flip plus small brightness and contrast jitter,
     applied to train only.
  3. class_weight="balanced" handles the imbalance; nothing is downsampled.
  4. sparse_categorical_crossentropy loss with the Adam optimiser.
  5. EarlyStopping with restore_best_weights, plus ModelCheckpoint saving the best model.
  6. After training, compute accuracy, FN rate, FP rate, the confusion matrix and the
     prediction distribution on val and optionally test. The prediction distribution confirms
     both classes are predicted and the model has not collapsed. Everything is printed, and a
     single RESULT <json> line is emitted for easy parsing.

Every tunable knob is a command-line argument, so iterating changes parameters rather than
code structure.

Dependencies: tensorflow==2.19.*, pandas, numpy.

Example, the first round of fixes combined:
  python scripts/train.py --bn-momentum 0.9 --patience 15 \
      --start-from-epoch 20 --epochs 80 --tag r1 --eval-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from model import build_model  # scripts/model.py, in the same directory

AUTOTUNE = tf.data.AUTOTUNE


def load_split(csv_path: Path, data_root: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(csv_path)
    paths = [str(data_root / p) for p in df["path"].tolist()]
    labels = df["label"].to_numpy(dtype=np.int32)
    return paths, labels


def make_dataset(
    paths: list[str],
    labels: np.ndarray,
    size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    augment: bool = False,
) -> tf.data.Dataset:
    """Read PNG, greyscale, normalise to [0,1], optionally cache, shuffle and augment, then
    batch."""

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)  # → [0,1]
        img = tf.ensure_shape(img, (size, size, 1))
        return img, label

    def _augment(img, label):
        # Light augmentation: horizontal flip plus small brightness and contrast jitter,
        # clipped back to [0,1] afterwards.
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.10)
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()  # decode once and cache; augmentation happens after the cache so each
                     # epoch differs
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def compute_class_weight(labels: np.ndarray) -> dict[int, float]:
    n_total = len(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    return {0: n_total / (2.0 * n_neg), 1: n_total / (2.0 * n_pos)}


def evaluate(model: tf.keras.Model, ds: tf.data.Dataset, y_true: np.ndarray) -> dict:
    """Compute accuracy, FN rate, FP rate, the confusion matrix and the prediction
    distribution. The positive class is 1, meaning record. Returns a dict."""
    probs = model.predict(ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    total = tn + fp + fn + tp
    pred_dist = [int((y_pred == 0).sum()), int((y_pred == 1).sum())]
    return {
        "n": total,
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "fn_rate": round(fn / (tp + fn), 4) if (tp + fn) else 0.0,
        "fp_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "pred_dist": pred_dist,  # [count predicted 0, count predicted 1]
        "collapsed": (pred_dist[0] == 0 or pred_dist[1] == 0),
    }


def print_eval(name: str, m: dict) -> None:
    print(f"\n===== {name} evaluation (positive class = 1 = record) =====")
    print(f"samples: {m['n']} ({m['n_pos']} positive / {m['n_neg']} negative)")
    print(f"prediction distribution [predicted 0, predicted 1]: {m['pred_dist']}  "
          f"{'*** COLLAPSED ***' if m['collapsed'] else '(both classes present, no collapse)'}")
    print(f"accuracy : {m['accuracy']:.4f}")
    print(f"FN rate  : {m['fn_rate']:.4f}")
    print(f"FP rate  : {m['fp_rate']:.4f}")
    c = m["confusion"]
    print("confusion matrix:      predicted 0   predicted 1")
    print(f"  actual 0 (skip)      {c['tn']:>6}        {c['fp']:>6}")
    print(f"  actual 1 (record)    {c['fn']:>6}        {c['tp']:>6}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Train the gatekeeper CNN, fully parameterised.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=Path("data/processed"))
    p.add_argument("--train-csv", type=Path, default=Path("data/processed/train.csv"))
    p.add_argument("--val-csv", type=Path, default=Path("data/processed/val.csv"))
    p.add_argument("--test-csv", type=Path, default=Path("data/processed/test.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("models"))
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--start-from-epoch", type=int, default=20)
    p.add_argument("--monitor", type=str, default="val_loss")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bn-momentum", type=float, default=0.9)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--eval-test", action="store_true", help="also evaluate on test after training")
    args = p.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    train_paths, train_labels = load_split(args.train_csv, args.data_root)
    val_paths, val_labels = load_split(args.val_csv, args.data_root)
    print(f"[{args.tag}] train {len(train_paths)} | val {len(val_paths)}")

    train_ds = make_dataset(train_paths, train_labels, args.size, args.batch_size,
                            shuffle=True, seed=args.seed, augment=args.augment)
    val_ds = make_dataset(val_paths, val_labels, args.size, args.batch_size,
                          shuffle=False, seed=args.seed)

    class_weight = (compute_class_weight(train_labels)
                    if args.class_weight == "balanced" else None)
    print(f"class_weight={class_weight}")

    model = build_model(args.size, bn_momentum=args.bn_momentum)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / f"gatekeeper_{args.tag}.keras"
    mode = "max" if "accuracy" in args.monitor else "min"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor=args.monitor,
                                           mode=mode, save_best_only=True, verbose=0),
        tf.keras.callbacks.EarlyStopping(monitor=args.monitor, mode=mode,
                                         patience=args.patience,
                                         start_from_epoch=args.start_from_epoch,
                                         restore_best_weights=True, verbose=1),
    ]

    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                     class_weight=class_weight, callbacks=callbacks, verbose=2)

    # best epoch, according to the monitored metric
    mon = hist.history.get(args.monitor, [])
    best_epoch = (int(np.argmax(mon)) if mode == "max" else int(np.argmin(mon))) + 1 if mon else None
    stopped_epoch = len(mon)

    val_m = evaluate(model, val_ds, val_labels)
    print_eval("val", val_m)

    test_m = None
    if args.eval_test:
        test_paths, test_labels = load_split(args.test_csv, args.data_root)
        test_ds = make_dataset(test_paths, test_labels, args.size, args.batch_size,
                               shuffle=False, seed=args.seed)
        test_m = evaluate(model, test_ds, test_labels)
        print_eval("test", test_m)

    print(f"\nbest model saved to {ckpt_path} (best epoch={best_epoch}, {stopped_epoch} epochs run)")

    result = {
        "tag": args.tag,
        "config": {
            "bn_momentum": args.bn_momentum, "lr": args.lr,
            "batch_size": args.batch_size, "epochs": args.epochs,
            "patience": args.patience, "start_from_epoch": args.start_from_epoch,
            "monitor": args.monitor, "augment": args.augment,
            "class_weight": args.class_weight, "seed": args.seed,
        },
        "best_epoch": best_epoch, "stopped_epoch": stopped_epoch,
        "ckpt": str(ckpt_path), "val": val_m, "test": test_m,
    }
    print("RESULT " + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
