"""守门员边缘推理 —— 共享模块（importable，不是脚本）。

封装 int8 .tflite 守门员：load → preprocess(相机帧) → predict(label, score, latency)。
benchmark_latency.py / cascade.py 都复用本模块，保证"预处理 + 推理"口径唯一。

设计选择（写进注释，便于审计）：
- 运行时优先用 LiteRT(`ai_edge_litert.Interpreter`)——这是树莓派上的官方推荐运行时
  （TF 2.20 起 `tf.lite.Interpreter` 计划弃用）。若 Pi 上没装 ai_edge_litert，
  回退到 `tensorflow.lite.Interpreter`，这样本模块在笔记本上也能跑通自测。
- 预处理与训练/导出**口径一致**：单通道灰度 → resize 96×96 用 INTER_AREA（缩小专用，
  抗锯齿最好）→ 像素归一化到 [0,1] float（对应 train.py 的 convert_image_dtype）→
  再按模型**自身**的输入量化参数(scale, zero_point)量化成 int8。量化参数从 interpreter
  读，**绝不硬编码**——这样喂进 int8 图的数值与转换器标定时完全一致。
- 阈值不写死在 predict 里：predict 返回 argmax label + p(记) 概率分数，
  由调用方(cascade.py)用可配置阈值裁定"记/不记"，便于将来调 FN/FP 工作点。

依赖（Pi 端）：ai_edge_litert（或 tensorflow）、opencv-headless(cv2)、numpy。
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

# 守门员输入边长（与训练/导出一致：96×96×1 灰度）。
INPUT_SIZE = 96

# 默认决策阈值：对 p(记)=softmax[1] 卡阈值。0.5 等价 argmax。
# v4_mvp 调过的部署工作点约 0.55（见 models/README）；阈值是 FN/FP 工作点旋钮，
# 留给调用方覆盖，这里只给一个安全默认。
DEFAULT_THRESHOLD = 0.5


def _make_interpreter(model_path: str):
    """优先 LiteRT，回退 tf.lite。返回未 allocate 的 interpreter。"""
    try:
        from ai_edge_litert.interpreter import Interpreter  # Pi 上的推荐运行时
        return Interpreter(model_path=model_path)
    except ImportError:
        # 笔记本自测 / 尚未装 ai_edge_litert 时回退。功能等价。
        import tensorflow as tf  # noqa: 局部导入，避免 Pi 上强依赖整个 TF
        return tf.lite.Interpreter(model_path=model_path)


def load_model(path: str | Path):
    """加载 int8 .tflite，allocate_tensors，返回可直接推理的 interpreter。"""
    interp = _make_interpreter(str(path))
    interp.allocate_tensors()
    return interp


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """把相机帧统一成 2D 灰度 uint8。

    接受：2D 灰度（如 Picamera2 lores 的 YUV420 Y 平面，最省）、或 3D 彩色。
    注：Picamera2 默认给 RGB；用 COLOR_RGB2GRAY 走标准 luma(0.299R+0.587G+0.114B)，
    与 PNG 解码到单通道的 luma 近似一致。若上游其实是 BGR，仅 R/B 权重对调，
    对低分辨率灰度守门员影响可忽略——但最优做法是上游直接给灰度（见 cascade.py）。
    """
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    raise ValueError(f"无法识别的帧形状：{frame.shape}")


def preprocess(frame: np.ndarray, interpreter) -> np.ndarray:
    """相机帧 → (1,96,96,1) int8，口径同训练/导出。

    需要 interpreter 以读取输入张量的量化参数（scale, zero_point），不硬编码。
    """
    gray = _to_gray(frame)
    # INTER_AREA：缩小图像的推荐插值，等价区域平均，抗锯齿；与"低分辨率灰度输入"目标一致。
    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0  # → [0,1]，对应训练 convert_image_dtype

    in_detail = interpreter.get_input_details()[0]
    scale, zero_point = in_detail["quantization"]  # 模型自带的输入量化参数
    if scale == 0:  # 理论上 int8 模型不会是 0；保险：float 输入模型直接喂 float
        q = x
    else:
        # q = round(real/scale) + zero_point，钳到 int8 范围
        q = np.round(x / scale + zero_point)
        q = np.clip(q, -128, 127).astype(np.int8)
    return q.reshape(1, INPUT_SIZE, INPUT_SIZE, 1)


def predict(interpreter, frame: np.ndarray) -> tuple[int, float, float]:
    """对一帧做守门员推理。

    返回 (label, score, latency_ms)：
      label      —— argmax 类别（0=不记 / 1=记）
      score      —— p(记)=softmax[1]，dequant 后的概率，供调用方卡阈值
      latency_ms —— **纯推理**耗时（set_tensor+invoke+get_tensor），不含预处理；
                    这样 cascade 的 hits log 与 benchmark 口径一致、可比。
    """
    x = preprocess(frame, interpreter)
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    t0 = time.perf_counter()
    interpreter.set_tensor(in_detail["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(out_detail["index"])[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    o_scale, o_zp = out_detail["quantization"]
    probs = (y.astype(np.float32) - o_zp) * o_scale if o_scale else y.astype(np.float32)
    label = int(np.argmax(probs))
    score = float(probs[1])  # p(记)
    return label, score, latency_ms
