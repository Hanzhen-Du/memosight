#!/usr/bin/env python3
"""Task1 候选预算估算 —— 训练前先算 参数量 / 峰值激活 / int8权重 / 算子，超标直接跳过不训。

ESP32 硬约束（淘汰线）：峰值激活 < 256KB · int8 权重 < 100KB · 仅 TFLM 白名单算子。
本脚本对一组候选 build_model 配置：
  - 逐层算 int8 激活（H×W×C 字节）：报告 单层峰值 + 并发峰值(同层输入+输出，更接近 TFLM arena)。
  - int8 权重估算 = count_params×1B（与 export_tflite.py 同口径；基线 24874→24.3KB 实测吻合）。
  - 静态算子核对：build_model 只用 Conv2D/BN(折叠)/ReLU/MaxPool/AveragePool/Reshape/Dense/Softmax，
    全在 TFLM 白名单内；真实白名单仍以 export_tflite.py 实测 .tflite 为准（task2 踩过动态算子坑）。
  - 给出 PASS / SKIP（任一预算超标即 SKIP，不进入训练）。

不训练、不导出，纯静态核算。依赖：tensorflow。

示例：.venv/bin/python scripts/estimate_budget.py
"""

from __future__ import annotations

import json

import tensorflow as tf

from model import build_model

ACT_BUDGET_KB = 256
W_BUDGET_KB = 100
BYTES_INT8 = 1

# 候选清单：(标签, 描述, channels, convs_per_stage)
CANDIDATES = [
    ("baseline", "基线 task2_mvp 架构", (8, 16, 32, 64), 1),
    ("A_wide_late", "晚期加宽通道(早期不动→激活廉价)", (8, 16, 64, 128), 1),
    ("B_deep_stack", "每 stage 堆 2 个 Conv(全程加深)", (8, 16, 32, 64), 2),
    ("C_wide_uniform", "整体加宽(含早期 block1→16,吃激活)", (16, 32, 64, 64), 1),
    ("D_five_stage", "加一个 stage(更深下采样+抽象)", (8, 16, 32, 64, 96), 1),
    ("E_combo", "晚期加宽+晚期加深(综合,逼近权重上限)", (8, 16, 48, 64), (1, 1, 2, 2)),
]

# int8 .tflite 折叠后仍占独立 arena 张量的层类型：BatchNorm 折进 Conv、ReLU 融合为 Conv 的
# fused activation，均不留独立张量。故激活核算只数下列"产张量"层 + 输入。
_TENSOR_LAYER_TYPES = (
    tf.keras.layers.InputLayer,
    tf.keras.layers.Conv2D,
    tf.keras.layers.MaxPooling2D,
    tf.keras.layers.AveragePooling2D,
    tf.keras.layers.Reshape,
    tf.keras.layers.Dense,
    tf.keras.layers.Softmax,
)


def _elems(shape) -> int:
    n = 1
    for d in shape[1:]:
        if d is not None:
            n *= int(d)
    return n


def analyze(model: tf.keras.Model) -> dict:
    # 折叠后图（BN/ReLU 不留独立张量）的"产张量"层序列——这才是 ESP32 TFLM arena 实际持有的张量。
    seq = [(l.name, tuple(l.output.shape), _elems(l.output.shape))
           for l in model.layers if isinstance(l, _TENSOR_LAYER_TYPES)]
    single_peak_kb, single_at = 0.0, ""
    for name, _, e in seq:
        kb = e * BYTES_INT8 / 1024
        if kb > single_peak_kb:
            single_peak_kb, single_at = kb, name
    concur_peak_kb, concur_at = 0.0, ""
    for (n_in, _, e_in), (n_out, _, e_out) in zip(seq, seq[1:]):
        kb = (e_in + e_out) * BYTES_INT8 / 1024
        if kb > concur_peak_kb:
            concur_peak_kb, concur_at = kb, f"{n_in}->{n_out}"
    params = model.count_params()
    final_spatial = None
    for name, shp, _ in seq:
        if name == "gap_avgpool":
            # gap 输入是前一层；取 flatten 前最后 conv/pool 的空间
            pass
    # 末端进入 gap 前的空间维：找 gap_avgpool 的输入形状
    gap = next(l for l in model.layers if l.name == "gap_avgpool")
    in_hw = int(gap.input.shape[1])
    return {
        "params": params,
        "int8_weight_kb": round(params / 1024, 1),
        "single_peak_kb": round(single_peak_kb, 1),
        "single_at": single_at,
        "concur_peak_kb": round(concur_peak_kb, 1),
        "concur_at": concur_at,
        "pre_gap_spatial": in_hw,
        "n_layers": len(seq),
    }


def main() -> int:
    rows = []
    for tag, desc, channels, convs in CANDIDATES:
        m = build_model(96, bn_momentum=0.9, channels=channels, convs_per_stage=convs,
                        name=f"gk_{tag}")
        a = analyze(m)
        act_peak = max(a["single_peak_kb"], a["concur_peak_kb"])
        act_ok = a["concur_peak_kb"] < ACT_BUDGET_KB
        w_ok = a["int8_weight_kb"] < W_BUDGET_KB
        verdict = "PASS" if (act_ok and w_ok) else "SKIP(超预算)"
        rows.append({"tag": tag, "desc": desc, "channels": list(channels),
                     "convs_per_stage": convs if isinstance(convs, int) else list(convs),
                     **a, "act_peak_kb": act_peak, "verdict": verdict})

    print(f"{'候选':<16}{'channels':<22}{'conv/stg':<10}{'参数量':>9}"
          f"{'int8权重KB':>11}{'单层峰KB':>10}{'并发峰KB':>10}{'结论':>14}")
    print("-" * 104)
    for r in rows:
        cps = r["convs_per_stage"]
        print(f"{r['tag']:<16}{str(r['channels']):<22}{str(cps):<10}{r['params']:>9,}"
              f"{r['int8_weight_kb']:>11}{r['single_peak_kb']:>10}{r['concur_peak_kb']:>10}"
              f"{r['verdict']:>14}")
    print("-" * 104)
    print(f"预算：并发激活 < {ACT_BUDGET_KB}KB · int8权重 < {W_BUDGET_KB}KB · 仅TFLM白名单")
    print("（并发峰=同层输入+输出，更接近 ESP32 TFLM tensor arena 实占；单层峰为参考）")
    base = rows[0]
    print(f"\n基线参照：参数 {base['params']:,} / int8权重 {base['int8_weight_kb']}KB / "
          f"并发激活 {base['concur_peak_kb']}KB @ {base['concur_at']}")
    print("\nBUDGET_JSON " + json.dumps(rows, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
