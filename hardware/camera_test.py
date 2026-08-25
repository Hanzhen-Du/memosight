#!/usr/bin/env python3
"""Camera Module 3 最小自检（Picamera2）—— 相机物理接上后**第一个**要跑的脚本。

干三件事：
  1. 抓几帧，确认相机能出图、形状正常；
  2. 存一张全分辨率 JPEG（full_res.jpg）；
  3. 存一张缩放到 96×96 的灰度 PNG（gray_96.png）——即守门员真正会看到的输入，
     肉眼检查"低分辨率灰度还能不能看出是不是文字/屏幕"。

不做推理、不做循环——纯粹验证相机链路通不通。

依赖：picamera2、opencv-headless(cv2)、numpy。
  picamera2 随 Raspberry Pi OS 发行，但**不一定预装**；缺了用：
      sudo apt install -y python3-picamera2
  （Picamera2 依赖 libcamera，apt 包能把系统依赖一起装好；不建议 pip 装。）

示例（Pi 上）：
  python3 hardware/camera_test.py --outdir hardware/captures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:  # 给出可执行的修复指引，而不是裸 traceback
    raise SystemExit(
        "未找到 picamera2。Raspberry Pi OS 上安装：\n"
        "    sudo apt install -y python3-picamera2\n"
        "（它会一并装好 libcamera 等系统依赖；勿用 pip。）"
    )

INPUT_SIZE = 96


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Camera Module 3 自检（Picamera2）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--outdir", type=Path, default=Path("hardware/captures"))
    ap.add_argument("--frames", type=int, default=5, help="抓几帧后再存图")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    picam = Picamera2()
    # 静态拍照配置，main 流走全分辨率（None=用传感器默认全幅）。RGB888 便于存 JPEG。
    config = picam.create_still_configuration(main={"format": "RGB888"})
    picam.configure(config)
    picam.start()
    try:
        frame = None
        for i in range(args.frames):  # 丢掉前几帧，让自动曝光/白平衡收敛
            frame = picam.capture_array("main")
            print(f"frame {i}: shape={frame.shape} dtype={frame.dtype}")
        assert frame is not None

        # 1) 全分辨率 JPEG（注意 Picamera2 给 RGB，cv2 存图按 BGR，转一下）
        full_path = args.outdir / "full_res.jpg"
        cv2.imwrite(str(full_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # 2) 守门员实际输入：灰度 + INTER_AREA 缩到 96×96
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray96 = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        gray_path = args.outdir / "gray_96.png"
        cv2.imwrite(str(gray_path), gray96)

        print(f"\n全分辨率帧: {frame.shape} → {full_path}")
        print(f"守门员输入: {gray96.shape}（灰度 96×96）→ {gray_path}")
        print("相机链路 OK。下一步可跑 cascade.py。")
    finally:
        picam.stop()  # 确保释放相机
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
