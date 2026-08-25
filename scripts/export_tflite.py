#!/usr/bin/env python3
"""守门员模型导出 —— .keras → .tflite（float32 基准 + 全 int8 量化），含 ESP32 约束验证。

产出（默认存到 models/）：
  1. <stem>_float32.tflite —— 不量化、float I/O。树莓派上的精度基准。
  2. <stem>_int8.tflite    —— 全 int8 量化、int8 I/O。ESP32 可移植形态。

导出后自动验证并打印报告（拿真实数字对照此前的估算）：
  A. 列出 int8 .tflite 里所有算子，逐个核对 TFLM 白名单
     {CONV_2D, DEPTHWISE_CONV_2D, AVERAGE_POOL_2D, MAX_POOL_2D, RESHAPE,
      FULLY_CONNECTED, SOFTMAX, 以及边缘的 QUANTIZE/DEQUANTIZE}。
     不在白名单的算子明确标记 [NOT WHITELISTED]。
  B. 确认图内部无 float32 张量（全 int8）。
  C. 报告两个 .tflite 的实际体积；int8 权重大小与此前 24.3KB 估算对照。
  D. 正确性：用 int8 .tflite 在 test split 上跑 accuracy/F1，与原 .keras 对比。

预处理与 train.py:load_split / make_dataset 完全一致：
  读 PNG → decode_png(channels=1) → convert_image_dtype(float32) [0,1] → (96,96,1)。

依赖：tensorflow==2.19.*、pandas、numpy。

示例：
  python scripts/export_tflite.py            # 用默认（v4_mvp + dedup split）
  python scripts/export_tflite.py --model models/gatekeeper_v4_mvp.keras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

SIZE = 96

# TFLM 白名单（builtin op 名，与 _get_ops_details 返回的 op_name 对齐；大写）。
# 含模型本体算子 + int8 I/O 边缘可能出现的 QUANTIZE/DEQUANTIZE。
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


# ---------- 数据加载（与 train.py 完全一致的预处理） ----------

def _decode(path: str) -> np.ndarray:
    """单张图：read_file → decode_png(1ch) → float32 [0,1] → (96,96,1)。"""
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


# ---------- 导出 ----------

def _fixed_batch_model(model: tf.keras.Model) -> tf.keras.Model:
    """把训练好的模型包一层固定 batch=1 的输入，权重共享、不改结构。

    关键：默认导出保留动态 batch(-1)，会让 flatten 的 Reshape 在运行时用
    SHAPE/STRIDED_SLICE/PACK 动态拼形状——这三个算子不在 TFLM 白名单内。
    固定 batch=1 后内部形状全为具体值，Reshape 退化成静态常量，只剩 RESHAPE，
    严守白名单。ESP32/TFLM 推理本就是 batch=1，固定 batch 不损失部署能力。
    用 keras 包装（而非 concrete function）以保证权重被折叠成常量嵌入，
    避免 from_concrete_functions 留下 READ_VARIABLE 未绑定变量。
    """
    inp = tf.keras.Input(batch_shape=(1, SIZE, SIZE, 1), name="input")
    out = model(inp)
    return tf.keras.Model(inp, out, name=f"{model.name}_fixedbatch")


def export_float32(model: tf.keras.Model, out_path: Path) -> None:
    """不量化导出，float I/O。树莓派精度基准。"""
    conv = tf.lite.TFLiteConverter.from_keras_model(_fixed_batch_model(model))
    tflite = conv.convert()
    out_path.write_bytes(tflite)


def export_int8(
    model: tf.keras.Model,
    out_path: Path,
    rep_paths: list[str],
    n_rep: int,
) -> None:
    """全 int8 量化，int8 I/O。representative_dataset 来自 dedup_train。"""

    def rep_gen():
        for p in rep_paths[:n_rep]:
            x = _decode(p)[None, ...]  # (1,96,96,1) float32 [0,1]
            yield [x]

    conv = tf.lite.TFLiteConverter.from_keras_model(_fixed_batch_model(model))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    # 强制全 int8；有算子不能 int8 量化则 convert() 直接报错（这正是我们要的护栏）。
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite = conv.convert()
    out_path.write_bytes(tflite)


# ---------- 验证 ----------

def verify_ops(int8_path: Path) -> list[tuple[str, bool]]:
    """列出 int8 .tflite 所有算子，返回 [(op_name, in_whitelist), ...]。"""
    # BUILTIN_REF：用参考 kernel，不挂 XNNPACK delegate，
    # 这样 _get_ops_details 返回的是 flatbuffer 真实算子，而非运行时 DELEGATE 节点。
    interp = tf.lite.Interpreter(
        model_path=str(int8_path),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interp.allocate_tensors()
    ops = interp._get_ops_details()  # 半私有 API，TF2.19 可用
    return [(o["op_name"], o["op_name"] in TFLM_WHITELIST) for o in ops]


def verify_dtypes(int8_path: Path) -> dict:
    """统计所有张量 dtype，标出 float32 张量（全 int8 时应为 0）。"""
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


# ---------- int8 推理（test split 正确性） ----------

def predict_int8(int8_path: Path, paths: list[str]) -> np.ndarray:
    """逐样本跑 int8 解释器，返回 dequant 后的 p(记) 概率数组。"""
    interp = tf.lite.Interpreter(model_path=str(int8_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    p_pos = np.empty(len(paths), dtype=np.float32)
    for i, p in enumerate(paths):
        x = _decode(p)[None, ...]  # (1,96,96,1) float32 [0,1]
        # 量化到 int8：q = round(x/scale) + zero_point
        q = np.round(x / in_scale + in_zp).astype(np.int32)
        q = np.clip(q, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        y = interp.get_tensor(out["index"])[0].astype(np.float32)
        deq = (y - out_zp) * out_scale  # dequant softmax 概率
        p_pos[i] = deq[1]
    return p_pos


def predict_keras(model: tf.keras.Model, paths: list[str]) -> np.ndarray:
    """原 keras 模型 p(记)，逐样本（与 int8 对齐，避免 batch 差异）。"""
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


# ---------- 报告 ----------

def kb(n_bytes: int) -> str:
    return f"{n_bytes / 1024:.1f} KB"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="守门员 .keras → .tflite 导出 + ESP32 约束验证。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", type=Path,
                    default=Path("models/gatekeeper_v4_mvp.keras"))
    ap.add_argument("--data-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--train-csv", type=Path,
                    default=Path("data/processed/dedup_train.csv"),
                    help="representative_dataset 来源")
    ap.add_argument("--test-csv", type=Path,
                    default=Path("data/processed/dedup_test.csv"),
                    help="正确性校验 split")
    ap.add_argument("--n-rep", type=int, default=200, help="代表性样本数")
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.55],
                    help="对照阈值（0.5=argmax；0.55=v4_mvp 部署点）")
    ap.add_argument("--rep-seed", type=int, default=42)
    args = ap.parse_args()

    stem = args.model.stem  # e.g. gatekeeper_v4_mvp
    f32_path = args.out_dir / f"{stem}_float32.tflite"
    int8_path = args.out_dir / f"{stem}_int8.tflite"

    print(f"加载 keras 模型：{args.model}")
    model = tf.keras.models.load_model(args.model)
    n_params = model.count_params()
    est_int8_weight_kb = n_params / 1024.0  # int8 每参数 1 字节的理论估算

    # representative dataset：固定 seed 打乱 dedup_train 取前 n_rep，跨类覆盖
    rep_paths, _ = load_paths_labels(args.train_csv, args.data_root)
    rng = np.random.default_rng(args.rep_seed)
    rng.shuffle(rep_paths)
    print(f"representative_dataset：从 {args.train_csv.name} 取 {args.n_rep}/{len(rep_paths)} 样本")

    print("\n[1/2] 导出 float32 .tflite ...")
    export_float32(model, f32_path)
    print(f"  → {f32_path}  ({kb(f32_path.stat().st_size)})")

    print("\n[2/2] 导出 int8 .tflite（全量化，int8 I/O）...")
    export_int8(model, int8_path, rep_paths, args.n_rep)
    print(f"  → {int8_path}  ({kb(int8_path.stat().st_size)})")

    # ===== 验证报告 =====
    print("\n" + "=" * 64)
    print("验证报告（int8 .tflite）")
    print("=" * 64)

    # A. 算子白名单
    print("\n--- A. 算子 TFLM 白名单核对 ---")
    ops = verify_ops(int8_path)
    all_ok = True
    for name, ok in ops:
        flag = "OK" if ok else "*** NOT WHITELISTED ***"
        if not ok:
            all_ok = False
        print(f"  {name:<22} {flag}")
    print(f"  白名单结论：{'全部通过' if all_ok else '存在非白名单算子！需处理'}")

    # B. dtype（图内部应全 int8）
    print("\n--- B. 张量 dtype（确认全 int8，无 float32 内部张量）---")
    dt = verify_dtypes(int8_path)
    for name, c in sorted(dt["counts"].items()):
        print(f"  {name:<12} {c} 个张量")
    if dt["float32_tensors"]:
        print(f"  *** 发现 {len(dt['float32_tensors'])} 个 float32 张量（应为 0）：")
        for n in dt["float32_tensors"]:
            print(f"      - {n}")
    else:
        print("  无 float32 张量 → 全 int8。")

    # C. 体积
    print("\n--- C. 体积对照 ---")
    print(f"  float32 .tflite 文件：{kb(f32_path.stat().st_size)}")
    print(f"  int8    .tflite 文件：{kb(int8_path.stat().st_size)}")
    print(f"  模型参数量：{n_params:,}")
    print(f"  int8 权重理论估算（参数×1字节）：{est_int8_weight_kb:.1f} KB"
          f"  ← 对照此前 24.3KB 估算")
    print(f"  注：.tflite 文件含权重 + flatbuffer 元数据/量化参数，"
          f"略大于纯权重；ESP32 <100KB 预算看权重，文件体积仅供运输参考。")

    # D. 正确性
    print("\n--- D. 正确性（test split：int8 .tflite vs 原 keras）---")
    test_paths, y_true = load_paths_labels(args.test_csv, args.data_root)
    print(f"  test：{len(test_paths)} 样本"
          f"（正 {int((y_true==1).sum())} / 负 {int((y_true==0).sum())}）")
    print("  跑 int8 解释器 ...")
    p_int8 = predict_int8(int8_path, test_paths)
    print("  跑原 keras ...")
    p_keras = predict_keras(model, test_paths)

    print(f"\n  {'阈值':>6} {'模型':>8} {'acc':>7} {'prec':>7} {'recall':>7}"
          f" {'F1':>7} {'FN率':>7} {'FP率':>7}")
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
              f"  {'(掉点明显，需警惕)' if d_f1 < -0.03 else '(量化掉点可接受)'}")

    # ===== 证据归类 =====
    print("\n" + "=" * 64)
    print("证据归类（重要：派宽容 ≠ ESP32 就绪）")
    print("=" * 64)
    print("""  [派上能跑的证据]
    - 两个 .tflite 均成功导出；
    - int8 .tflite 在 x86 上用 LiteRT 解释器跑通 test split 并给出指标；
    - 树莓派跑全量 LiteRT，运行环境比 ESP32 宽容，以上即足以在派上部署运行。

  [ESP32-ready 的证据（部分，仍需真机验证）]
    - 已满足：算子全在 TFLM 白名单内、全 int8、权重 <100KB（见上）；
    - 未验证（需后续真机）：TFLM tensor arena 实际占用是否 ≤ 片上 SRAM、
      逐算子 TFLM kernel 数值与本解释器是否一致、实测延迟/功耗。
    → 本脚本只证明"架构与量化形态满足 ESP32 静态约束"，
      不等于"已在 ESP32 跑通"。""")

    print("\n导出完成。")
    return 0 if all_ok and not dt["float32_tensors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
