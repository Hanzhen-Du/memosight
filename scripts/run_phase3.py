#!/usr/bin/env python3
"""阶段3 提分实验 harness —— class weight / focal loss / 屏幕增强，自动在 val 选阈值。

为不污染已冻结的 train.py（baseline 复现契约），阶段3 所有新旋钮集中在此：
  --loss {ce,focal}        交叉熵 / focal loss
  --focal-gamma            focal 聚焦参数（默认 2.0）
  --focal-alpha            focal 正类权重 alpha（默认 0.75，偏向召回）
  --pos-weight-mult        在 balanced class_weight 基础上对正类再乘的倍率（压 FN）
  --aug-level {base,screen} base=翻转+亮度对比度；screen 额外加 小幅旋转/缩放 + JPEG 压缩噪声
  --class-weight {balanced,none}

固定：dedup seed=42 切分、bn_momentum=0.9、patience=15、start_from_epoch=20、
epochs=80、lr=1e-3、val_loss restore best。**阈值在 val 上选（最大化 val F1），再用到 test 裁定。**

依赖：tensorflow、numpy、pandas（无新增依赖）。复用 model/train/evaluate 模块。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from model import build_model
from train import load_split, compute_class_weight
from evaluate import metrics_from_probs

AUTOTUNE = tf.data.AUTOTUNE


def make_ds(paths, labels, size, batch, shuffle, seed, aug_level):
    rot = tf.keras.layers.RandomRotation(0.02, fill_mode="nearest")  # ±~7°

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.ensure_shape(img, (size, size, 1))
        return img, label

    def _aug_base(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.10)
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
        return tf.clip_by_value(img, 0.0, 1.0), label

    def _aug_screen(img, label):
        # base + 小幅旋转/缩放 + JPEG 压缩噪声（贴合屏幕拍摄，幅度小以保文字可读）
        img, label = _aug_base(img, label)
        img = rot(img, training=True)
        u8 = tf.image.convert_image_dtype(tf.clip_by_value(img, 0.0, 1.0), tf.uint8)
        u8 = tf.image.random_jpeg_quality(u8, 45, 95)
        img = tf.image.convert_image_dtype(u8, tf.float32)
        return tf.clip_by_value(img, 0.0, 1.0), label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE).cache()
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    if aug_level == "base":
        ds = ds.map(_aug_base, num_parallel_calls=AUTOTUNE)
    elif aug_level == "screen":
        ds = ds.map(_aug_screen, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


def make_focal(gamma: float, alpha: float):
    """2 类 softmax 概率输出的 focal loss（返回逐样本损失，便于 Keras 归约/加权）。"""
    def focal(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        p_t = tf.gather(y_pred, y_true, batch_dims=1)
        p_t = tf.clip_by_value(p_t, 1e-7, 1.0)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1.0 - alpha)
        return -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
    return focal


def select_threshold(probs, labels):
    """在给定（val）概率上扫阈值，返回最大化 F1 的阈值（平手取更低 FN）。"""
    best = (0.5, -1.0, 1.0)  # (thr, f1, fn)
    for t in np.arange(0.05, 1.0, 0.05):
        m = metrics_from_probs(probs, labels, float(t))
        if (m["f1"], -m["fn_rate"]) > (best[1], -best[2]):
            best = (round(float(t), 2), m["f1"], m["fn_rate"])
    return best[0]


def main() -> int:
    p = argparse.ArgumentParser(description="阶段3 提分实验。")
    p.add_argument("--name", required=True, help="实验名（用于模型/结果标识）")
    p.add_argument("--data-root", type=Path, default=Path("data/processed"))
    p.add_argument("--train-csv", type=Path, default=Path("data/processed/dedup_train.csv"))
    p.add_argument("--val-csv", type=Path, default=Path("data/processed/dedup_val.csv"))
    p.add_argument("--test-csv", type=Path, default=Path("data/processed/dedup_test.csv"))
    p.add_argument("--loss", choices=["ce", "focal"], default="ce")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=0.75)
    p.add_argument("--pos-weight-mult", type=float, default=1.0)
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    p.add_argument("--aug-level", choices=["base", "screen"], default="base")
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--start-from-epoch", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bn-momentum", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("models"))
    p.add_argument("--results", type=Path, default=Path("docs/results/phase3_results.json"))
    args = p.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    tr_paths, tr_labels = load_split(args.train_csv, args.data_root)
    va_paths, va_labels = load_split(args.val_csv, args.data_root)
    te_paths, te_labels = load_split(args.test_csv, args.data_root)

    train_ds = make_ds(tr_paths, tr_labels, args.size, args.batch_size, True, args.seed, args.aug_level)
    val_ds = make_ds(va_paths, va_labels, args.size, args.batch_size, False, args.seed, "none")
    test_ds = make_ds(te_paths, te_labels, args.size, args.batch_size, False, args.seed, "none")

    # class weight（balanced，可对正类再乘倍率）
    cw = None
    if args.class_weight == "balanced":
        cw = compute_class_weight(tr_labels)
        cw[1] *= args.pos_weight_mult
    print(f"[{args.name}] loss={args.loss} pos_mult={args.pos_weight_mult} "
          f"aug={args.aug_level} class_weight={cw}")

    loss = (make_focal(args.focal_gamma, args.focal_alpha) if args.loss == "focal"
            else "sparse_categorical_crossentropy")
    model = build_model(args.size, bn_momentum=args.bn_momentum)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=loss, metrics=["accuracy"])
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                           patience=args.patience,
                                           start_from_epoch=args.start_from_epoch,
                                           restore_best_weights=True, verbose=0)]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                     class_weight=cw, callbacks=cb, verbose=0)

    ckpt = args.out_dir / f"gatekeeper_{args.name}.keras"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save(ckpt)

    val_probs = model.predict(val_ds, verbose=0)
    test_probs = model.predict(test_ds, verbose=0)
    thr = select_threshold(val_probs, va_labels)

    out = {"name": args.name, "stopped_epoch": len(hist.history["loss"]),
           "selected_threshold": thr, "ckpt": str(ckpt),
           "config": {"loss": args.loss, "focal_gamma": args.focal_gamma,
                      "focal_alpha": args.focal_alpha, "pos_weight_mult": args.pos_weight_mult,
                      "class_weight": args.class_weight, "aug_level": args.aug_level}}
    for split, probs, labels in (("val", val_probs, va_labels), ("test", test_probs, te_labels)):
        out[f"{split}@0.5"] = metrics_from_probs(probs, labels, 0.5)
        out[f"{split}@thr"] = metrics_from_probs(probs, labels, thr)

    tt = out["test@thr"]
    print(f"[{args.name}] best epoch≈{out['stopped_epoch']} | val选阈值={thr} | "
          f"TEST@thr: F1={tt['f1']:.4f} recall={tt['recall']:.4f} "
          f"FN={tt['fn_rate']:.4f} FP={tt['fp_rate']:.4f} acc={tt['accuracy']:.4f}")

    # 追加到结果 JSON（累积所有实验）
    allres = []
    if args.results.exists():
        allres = json.loads(args.results.read_text())
    allres = [r for r in allres if r["name"] != args.name] + [out]
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(allres, ensure_ascii=False, indent=2))
    print("RESULT " + json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
