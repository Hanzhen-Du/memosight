#!/usr/bin/env python3
"""守门员小 CNN —— 训练（参数化，供自主迭代修复塌缩）。

流程：
  1. 从 train/val(/test).csv 用 tf.data 加载 96×96 灰度 PNG，像素归一化到 [0,1]。
  2. (可选) 轻量数据增强：水平翻转 + 小幅亮度/对比度抖动（仅 train）。
  3. class_weight(balanced) 处理不平衡，不下采样。
  4. 损失 sparse_categorical_crossentropy，优化器 Adam。
  5. EarlyStopping(restore_best_weights) + ModelCheckpoint(存最优)。
  6. 训完在 val(及可选 test) 上算：准确率、FN rate、FP rate、混淆矩阵、
     **预测分布(确认两类都有、没塌缩)**。打印 + 输出一行 RESULT <json> 便于解析。

所有授权可调旋钮都走命令行参数，迭代时只改参数、不改代码结构。

依赖：tensorflow==2.19.*、pandas、numpy。

示例（第一轮修复组合）：
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

from model import build_model  # 同目录 scripts/model.py

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
    """读 PNG → 灰度 → [0,1] →(可选 cache/打乱/增强)→ batch。"""

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)  # → [0,1]
        img = tf.ensure_shape(img, (size, size, 1))
        return img, label

    def _augment(img, label):
        # 轻量增强：水平翻转 + 小幅亮度/对比度抖动；抖动后 clip 回 [0,1]。
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.10)
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()  # 解码一次缓存（增强在 cache 之后，保证每 epoch 不同）
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
    """算 acc/FN rate/FP rate/混淆矩阵/预测分布（正类=1=记）。返回 dict。"""
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
        "pred_dist": pred_dist,  # [预测为0的数, 预测为1的数]
        "collapsed": (pred_dist[0] == 0 or pred_dist[1] == 0),
    }


def print_eval(name: str, m: dict) -> None:
    print(f"\n===== {name} 评估（正类=1=记）=====")
    print(f"样本数：{m['n']}（正 {m['n_pos']} / 负 {m['n_neg']}）")
    print(f"预测分布 [判不记0, 判记1]：{m['pred_dist']}  "
          f"{'*** 塌缩! ***' if m['collapsed'] else '(两类都有，未塌缩)'}")
    print(f"准确率 Accuracy : {m['accuracy']:.4f}")
    print(f"漏报率 FN rate  : {m['fn_rate']:.4f}")
    print(f"误报率 FP rate  : {m['fp_rate']:.4f}")
    c = m["confusion"]
    print("混淆矩阵：           预测=不记(0)  预测=记(1)")
    print(f"  真实=不记(0)         {c['tn']:>6}     {c['fp']:>6}")
    print(f"  真实=记  (1)         {c['fn']:>6}     {c['tp']:>6}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="守门员小 CNN 训练（参数化）。",
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
    p.add_argument("--eval-test", action="store_true", help="训完也在 test 上复核")
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

    # best epoch（按 monitor）
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

    print(f"\n最优模型已存：{ckpt_path}（best epoch={best_epoch}, 共训 {stopped_epoch} epoch）")

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
