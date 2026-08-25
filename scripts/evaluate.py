#!/usr/bin/env python3
"""守门员模型评估 —— 一键复评完整指标（accuracy/precision/recall/F1/混淆矩阵/FN/FP）。

加载一个已存的 .keras 模型 + 一个 split CSV，算出完整指标。正类=1=记。
支持自定义决策阈值（对 softmax 的 p(记) 卡阈值，默认 0.5=argmax），供 Phase 3 压 FN。
支持 --pr-sweep：扫一串阈值打印 precision/recall/F1/FN/FP 曲线表。

每次改动模型后跑这个脚本即可一键复评，指标口径固定、可对比。

依赖：tensorflow、numpy、pandas（无新增依赖）。复用 train.py 的数据加载。

示例：
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
    """对 p(记)=probs[:,1] 卡阈值，算完整指标。正类=1。"""
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
    print(f"\n===== {name}（阈值={m['threshold']}，正类=1=记）=====")
    print(f"样本 {m['n']}（正 {m['n_pos']} / 负 {m['n_neg']}）")
    print(f"Accuracy {m['accuracy']:.4f} | Precision {m['precision']:.4f} | "
          f"Recall {m['recall']:.4f} | F1 {m['f1']:.4f}")
    print(f"FN rate {m['fn_rate']:.4f} | FP rate {m['fp_rate']:.4f}")
    c = m["confusion"]
    print("混淆矩阵：           预测=不记(0)  预测=记(1)")
    print(f"  真实=不记(0)         {c['tn']:>6}     {c['fp']:>6}")
    print(f"  真实=记  (1)         {c['fn']:>6}     {c['tp']:>6}")


def main() -> int:
    p = argparse.ArgumentParser(description="守门员模型完整指标复评。")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/processed"))
    p.add_argument("--val-csv", type=Path, default=None)
    p.add_argument("--test-csv", type=Path, default=Path("data/processed/dedup_test.csv"))
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--pr-sweep", action="store_true", help="扫阈值打印 PR/F1 曲线表")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    model = tf.keras.models.load_model(args.model)
    print(f"模型：{args.model}（参数 {model.count_params():,}）")

    out = {"model": str(args.model), "threshold": args.threshold}
    for split_name, csv in (("val", args.val_csv), ("test", args.test_csv)):
        if csv is None:
            continue
        probs, labels = predict_split(model, csv, args.data_root, args.size, args.batch_size)
        m = metrics_from_probs(probs, labels, args.threshold)
        tag = f"{split_name}（{'最终裁定' if split_name == 'test' else '调参用'}）"
        print_metrics(tag, m)
        out[split_name] = m

        if args.pr_sweep:
            print(f"\n----- {split_name} 阈值扫描（找压 FN 的甜点）-----")
            print(f"{'thr':>5} {'prec':>7} {'recall':>7} {'F1':>7} {'FN率':>7} {'FP率':>7}")
            for t in np.arange(0.05, 1.0, 0.05):
                mm = metrics_from_probs(probs, labels, float(t))
                print(f"{t:>5.2f} {mm['precision']:>7.4f} {mm['recall']:>7.4f} "
                      f"{mm['f1']:>7.4f} {mm['fn_rate']:>7.4f} {mm['fp_rate']:>7.4f}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\n指标 JSON 已写出：{args.json_out}")
    print("\nRESULT " + json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
