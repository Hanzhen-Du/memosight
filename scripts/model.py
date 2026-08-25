#!/usr/bin/env python3
"""守门员第一版小 CNN —— 建模 + 逐层激活内存核算（ESP32 移植硬关卡）。

模型（守门员二分类：记=1 / 不记=0）：
  输入 96×96×1 灰度
  4 个 block：Conv(3×3,'same') + BatchNorm + ReLU + MaxPool(2×2)
              通道 8 → 16 → 32 → 64，空间 96→48→24→12→6
  收尾：AveragePooling2D(6×6 全图) → Reshape(64) → Dense(2) → Softmax

> 关于"全局平均池化"的实现选择（重要，影响 TFLM 可移植性）：
>   任务原文写 GlobalAveragePooling2D，但 Keras 的 GAP 导出 TFLite 时
>   通常变成 MEAN(reduce_mean) 算子，而 MEAN 不在 TFLM 白名单里
>   (白名单：Conv2D/DepthwiseConv2D/AveragePool2D/MaxPool2D/Reshape/
>    FullyConnected/Softmax)。为严守白名单，这里用 AveragePooling2D 对
>   整张 6×6 feature map 做平均（等价于全局平均池化）+ Reshape 拉平，
>   三个算子全在白名单内。语义等价，导出可控。
>
> ⚠️ 导出验证修正（2026-06-18）：上面"全在白名单内"原是建模期的**静态算子核算**，
>   从未真正导出验证过。2026-06-18 首次实际导出 .tflite 才发现：若以默认**动态
>   batch(-1)** 导出，flatten 的 Reshape 会在运行时用 SHAPE/STRIDED_SLICE/PACK
>   动态拼形状——这三个算子**不在 TFLM 白名单内**。白名单保证**仅当以固定
>   batch_shape=(1,96,96,1) 导出**时成立（形状全静态、Reshape 退化为常量）。
>   导出脚本 `scripts/export_tflite.py` 已固定 batch=1 并自动核对算子表。

> 关于 BatchNorm：仅训练用。导出 int8 TFLite 时，Conv(use_bias=False)+BN
>   会被 TFLite 转换器自动折叠进前面的 Conv，不会留独立 BN 算子。

激活内存核算（int8 下每个元素 1 字节）：
  对每层输出 feature map 计算 H×W×C 字节，打印表格并标出峰值层。
  **峰值 > 256KB 立即判定为超预算（脚本返回非零、打印 STOP），不许估算了事。**

依赖：tensorflow==2.19.*（见 requirements.txt）。

示例：
  python scripts/model.py            # 建模 + 打印内存核算表
  python scripts/model.py --size 96
"""

from __future__ import annotations

import argparse
import sys

import tensorflow as tf

# ESP32 峰值激活内存硬预算（KB）。超过即停。
ACT_MEM_BUDGET_KB = 256

# int8 量化后每个激活元素占用字节数。
BYTES_PER_ELEM_INT8 = 1


def build_model(
    size: int = 96,
    bn_momentum: float = 0.99,
    channels: tuple[int, ...] = (8, 16, 32, 64),
    convs_per_stage: int | tuple[int, ...] = 1,
    name: str = "gatekeeper_v1",
) -> tf.keras.Model:
    """构建守门员小 CNN（Functional API，保证各层输出形状是具体值，便于内存核算）。

    bn_momentum：BatchNorm 滑动平均的 momentum。短训练时默认 0.99 收敛太慢、
    会导致推理统计量不准（训练/推理塌缩）；可调小（0.9/0.8）让滑动统计量更快跟上。
    注意：这是训练超参，不改网络结构；导出时 BN 仍会折叠进 Conv。

    复杂度旋钮（Task1：在 ESP32 预算内推高准确率）：
      - channels：每个 stage 的通道数，长度=stage 数。默认 (8,16,32,64) 复现基线。
        加宽早期 stage（96×96 空间）极吃激活内存，加宽晚期 stage 极廉价——见 §激活核算。
      - convs_per_stage：每个 stage 在 pool 前堆几个 Conv+BN+ReLU（int 或与 channels 等长的元组）。
        默认 1 复现基线，逐 stage 各层命名与基线**完全一致**（block{i}_conv/_bn/_relu/_pool），
        保证 baseline .keras 可无缝复训/加载。
    结构铁律（保 TFLM 白名单）：tail 一律 AveragePooling2D(整张 HxW) → Reshape → Dense(2) → Softmax，
    不引入 GAP/MEAN 等非白名单算子；不改 Conv/Pool/BN 的算子种类，只改通道/深度/stage 数。
    """
    n_stages = len(channels)
    if isinstance(convs_per_stage, int):
        convs_per_stage = (convs_per_stage,) * n_stages
    if len(convs_per_stage) != n_stages:
        raise ValueError(
            f"convs_per_stage 长度 {len(convs_per_stage)} 必须等于 channels 长度 {n_stages}"
        )

    inputs = tf.keras.Input(shape=(size, size, 1), name="input")
    x = inputs
    for i, (ch, nconv) in enumerate(zip(channels, convs_per_stage), start=1):
        for j in range(nconv):
            # nconv==1 时层名与基线完全一致（无数字后缀），保证可复现/可加载旧权重。
            suffix = "" if nconv == 1 else str(j + 1)
            # Conv 不带 bias：后面接 BN，BN 的 beta 充当偏置；这样 Conv+BN 折叠最干净。
            x = tf.keras.layers.Conv2D(
                ch, 3, padding="same", use_bias=False, name=f"block{i}_conv{suffix}"
            )(x)
            x = tf.keras.layers.BatchNormalization(
                momentum=bn_momentum, name=f"block{i}_bn{suffix}"
            )(x)
            x = tf.keras.layers.ReLU(name=f"block{i}_relu{suffix}")(x)
        x = tf.keras.layers.MaxPooling2D(2, name=f"block{i}_pool")(x)

    # 全局平均池化的 TFLM 安全写法：对剩余 HxW 做 AveragePool2D，再 Reshape 拉平。
    pooled_hw = x.shape[1]  # 基线为 6；更多 stage 时更小
    x = tf.keras.layers.AveragePooling2D(pool_size=pooled_hw, name="gap_avgpool")(x)
    x = tf.keras.layers.Reshape((x.shape[-1],), name="flatten")(x)
    x = tf.keras.layers.Dense(2, name="logits")(x)
    outputs = tf.keras.layers.Softmax(name="softmax")(x)
    return tf.keras.Model(inputs, outputs, name=name)


def _elems(shape) -> int:
    """非 batch 维度元素数之积（shape[0] 是 batch=None，跳过）。"""
    n = 1
    for d in shape[1:]:
        if d is not None:
            n *= int(d)
    return n


def activation_memory_report(model: tf.keras.Model) -> tuple[float, str]:
    """逐层打印输出 feature map 的 int8 激活内存，返回 (峰值KB, 峰值层名)。"""
    print(f"\n{'层名':<16}{'输出形状':<22}{'激活内存(KB)':>14}")
    print("-" * 52)
    peak_kb = 0.0
    peak_layer = ""
    # 收集 (层名, 输出shape, 元素数) 供后面算并发占用。
    seq: list[tuple[str, tuple, int]] = []
    for layer in model.layers:
        out_shape = tuple(layer.output.shape)
        elems = _elems(out_shape)
        kb = elems * BYTES_PER_ELEM_INT8 / 1024
        seq.append((layer.name, out_shape, elems))
        marker = ""
        if kb > peak_kb:
            peak_kb = kb
            peak_layer = layer.name
        print(f"{layer.name:<16}{str(out_shape):<22}{kb:>14.2f}")
    # 标出峰值
    print("-" * 52)
    print(f"单层输出峰值：{peak_kb:.2f} KB  @ {peak_layer}")

    # 参考：更接近 ESP32 TFLM tensor arena 的并发占用估计
    # （顺序网络执行某层时，需同时持有该层输入+输出张量）。
    concur_peak = 0.0
    concur_at = ""
    for (n_in, _, e_in), (n_out, _, e_out) in zip(seq, seq[1:]):
        kb = (e_in + e_out) * BYTES_PER_ELEM_INT8 / 1024
        if kb > concur_peak:
            concur_peak = kb
            concur_at = f"{n_in}→{n_out}"
    print(
        f"参考(更接近 ESP32 arena，同层输入+输出并发)："
        f"{concur_peak:.2f} KB  @ {concur_at}"
    )
    return peak_kb, peak_layer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="守门员小 CNN 建模 + 激活内存核算。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size", type=int, default=96, help="输入方形边长（像素）")
    args = parser.parse_args()

    model = build_model(args.size)
    model.summary()

    print(f"\n权重参数量：{model.count_params():,}")

    peak_kb, peak_layer = activation_memory_report(model)

    # 硬关卡判定（任务规定的指标：单层输出 feature map 峰值）
    print(f"\n激活内存预算：{ACT_MEM_BUDGET_KB} KB")
    if peak_kb > ACT_MEM_BUDGET_KB:
        print(
            f"\n*** STOP：单层激活峰值 {peak_kb:.2f} KB @ {peak_layer} "
            f"超过预算 {ACT_MEM_BUDGET_KB} KB。停止，不进入训练。***",
            file=sys.stderr,
        )
        return 1
    print(
        f"通过：单层激活峰值 {peak_kb:.2f} KB @ {peak_layer} "
        f"≤ {ACT_MEM_BUDGET_KB} KB，预算内。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
