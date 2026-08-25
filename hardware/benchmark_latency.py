#!/usr/bin/env python3
"""守门员纯推理延迟基准（树莓派端跑）。

测的是**纯推理**延迟：随机 int8 输入直接喂 invoke，不含相机采集/预处理，
这样数字只反映模型本身在该硬件上的算力开销，便于跨设备/跨模型对比。
（项目此前没有独立的 benchmark 脚本——计时只是临时探针；本文件把它固化下来。）

方法：20 次 warm-up（让缓存/线程/时钟稳定）+ 200 次计时，报告
mean / p50 / p95 / p99 / 吞吐(推理/秒)。随机输入按 interpreter 的输入 dtype/shape 生成。

依赖：ai_edge_litert（或 tensorflow）、numpy。复用 infer.load_model 保证运行时一致。

示例（Pi 上）：
  python3 hardware/benchmark_latency.py \
      --model ~/dev/memosight/models/gatekeeper_v4_mvp_int8.tflite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import infer  # 同目录模块；从 hardware/ 目录运行

WARMUP = 20
TIMED = 200


def random_input(interpreter, seed: int = 0) -> np.ndarray:
    """按模型输入张量的 dtype/shape 造一份随机输入。"""
    d = interpreter.get_input_details()[0]
    shape = tuple(int(s) for s in d["shape"])
    dtype = np.dtype(d["dtype"])
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(info.min, info.max + 1, size=shape, dtype=dtype)
    return rng.random(size=shape).astype(dtype)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="守门员纯推理延迟基准。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model", type=Path,
        default=Path.home() / "dev/memosight/models/gatekeeper_v4_mvp_int8.tflite",
    )
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--runs", type=int, default=TIMED)
    args = ap.parse_args()

    print(f"加载模型：{args.model}")
    interp = infer.load_model(args.model)
    in_detail = interp.get_input_details()[0]
    print(f"输入：shape={list(in_detail['shape'])} dtype={np.dtype(in_detail['dtype']).name}")

    x = random_input(interp)
    idx = in_detail["index"]

    # 用 time.perf_counter（高精度单调时钟）。逐次计时，避免分摊误差。
    import time
    print(f"warm-up {args.warmup} 次 ...")
    for _ in range(args.warmup):
        interp.set_tensor(idx, x)
        interp.invoke()

    print(f"计时 {args.runs} 次 ...")
    lat = np.empty(args.runs, dtype=np.float64)
    for i in range(args.runs):
        t0 = time.perf_counter()
        interp.set_tensor(idx, x)
        interp.invoke()
        lat[i] = (time.perf_counter() - t0) * 1000.0  # ms

    mean = lat.mean()
    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
    thr = 1000.0 / mean if mean > 0 else float("inf")

    print("\n===== 纯推理延迟（ms）=====")
    print(f"  样本数 : {args.runs}")
    print(f"  mean   : {mean:.3f}")
    print(f"  p50    : {p50:.3f}")
    print(f"  p95    : {p95:.3f}")
    print(f"  p99    : {p99:.3f}")
    print(f"  min/max: {lat.min():.3f} / {lat.max():.3f}")
    print(f"  吞吐   : {thr:.1f} 推理/秒")
    print("\n注：纯推理基准，不含相机采集/预处理；实际 cascade 单 tick 还要加这两块。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
