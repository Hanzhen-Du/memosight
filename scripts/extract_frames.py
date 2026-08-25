#!/usr/bin/env python3
"""视频抽帧 + 批量降清灰度 —— MemoSight 数据准备工具。

把原始素材（讲课视频、PPT/白板截图等）处理成守门员训练用的低分辨率灰度图，
同时保留一份高清原图母本，互不破坏。

输入可以是：
  - 一个视频文件（按固定时间间隔抽帧）
  - 一个装着图片的文件夹（遍历常见图片格式）

输出（在 <output_root>/<运行时间戳>/ 下）：
  - raw/   高清原图母本（视频帧存原始分辨率彩色 / 图片原样拷贝）
  - gray/  低清灰度图（先转灰度、再 resize 到目标尺寸）

文件名带运行时间戳 + 递增序号，天然防覆盖，且 raw/ 与 gray/ 一一对应。

依赖：opencv-python（见 requirements.txt）。

示例：
  python3 scripts/extract_frames.py lecture.mp4
  python3 scripts/extract_frames.py ./ppt_screenshots/ --size 128 128
  python3 scripts/extract_frames.py lecture.mp4 --interval 5 --output-root data/processed
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit(
        "缺少依赖 opencv-python。请先安装：pip install opencv-python\n"
        "（依赖已列在 requirements.txt；本脚本不会替你自动安装。）"
    )

# 视频抽帧默认间隔（秒）。讲课/白板/PPT 内容变化慢，2 秒兼顾覆盖翻页与去重。
DEFAULT_INTERVAL = 2.0
# 低清灰度图默认目标尺寸（宽, 高）。96x96 是 TinyML Visual Wake Words 的标准灰度输入，
# 贴合可移植 / int8 量化 / 树莓派部署路线。
DEFAULT_SIZE = (96, 96)
# 默认输出根目录。
DEFAULT_OUTPUT_ROOT = "data/processed"

# 当输入是文件夹时，视为图片的扩展名。
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# 当输入是文件时，视为视频的扩展名。
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="视频抽帧 + 批量降清灰度，生成守门员训练用的低清灰度图（保留高清母本）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="输入：一个视频文件，或一个装着图片的文件夹。",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help="视频抽帧间隔（秒）。仅对视频输入生效。",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=list(DEFAULT_SIZE),
        help="低清灰度图目标尺寸：宽 高（像素）。",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="输出根目录；脚本会在其下新建一个以运行时间戳命名的子目录。",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="不保留高清原图母本（仅对视频抽帧有意义；图片输入永远不改母本，只是不拷贝）。",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少进度输出。",
    )
    return parser.parse_args(argv)


def classify_input(input_path: Path) -> str:
    """判断输入是 'video' 还是 'imagedir'，否则报错退出。"""
    if input_path.is_dir():
        return "imagedir"
    if input_path.is_file():
        if input_path.suffix.lower() in VIDEO_EXTS:
            return "video"
        if input_path.suffix.lower() in IMAGE_EXTS:
            # 单张图片也按图片处理（当作只含一张图的"文件夹"）。
            return "imagedir"
        sys.exit(
            f"无法识别的文件类型：{input_path.suffix}\n"
            f"支持的视频后缀：{sorted(VIDEO_EXTS)}\n"
            f"支持的图片后缀：{sorted(IMAGE_EXTS)}"
        )
    sys.exit(f"输入不存在：{input_path}")


def make_output_dirs(output_root: Path, run_stamp: str, keep_raw: bool) -> tuple[Path, Path | None]:
    """创建 <output_root>/<run_stamp>/{raw,gray}，返回 (gray_dir, raw_dir)。"""
    run_dir = output_root / run_stamp
    gray_dir = run_dir / "gray"
    gray_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = None
    if keep_raw:
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
    return gray_dir, raw_dir


def to_low_gray(image, size: tuple[int, int]):
    """转灰度后 resize 到目标尺寸。size 为 (宽, 高)。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.resize 的 dsize 参数是 (宽, 高)。缩小用 INTER_AREA 质量更好。
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def process_video(
    input_path: Path,
    gray_dir: Path,
    raw_dir: Path | None,
    interval: float,
    size: tuple[int, int],
    run_stamp: str,
    quiet: bool,
) -> int:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        sys.exit(f"无法打开视频：{input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        sys.exit(f"读不到视频帧率（FPS），无法按时间间隔抽帧：{input_path}")

    step = max(1, round(fps * interval))  # 每隔多少帧抽一张
    if not quiet:
        print(f"视频 FPS={fps:.2f}，间隔 {interval}s → 每 {step} 帧抽 1 张")

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            name = f"{run_stamp}_{saved:06d}.png"
            if raw_dir is not None:
                cv2.imwrite(str(raw_dir / name), frame)
            cv2.imwrite(str(gray_dir / name), to_low_gray(frame, size))
            saved += 1
            if not quiet and saved % 50 == 0:
                print(f"  已抽 {saved} 帧...")
        frame_idx += 1

    cap.release()
    return saved


def process_imagedir(
    input_path: Path,
    gray_dir: Path,
    raw_dir: Path | None,
    size: tuple[int, int],
    run_stamp: str,
    quiet: bool,
) -> int:
    if input_path.is_file():
        images = [input_path]
    else:
        images = sorted(
            p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
    if not images:
        sys.exit(f"文件夹里没找到图片：{input_path}")

    saved = 0
    skipped = 0
    for src in images:
        image = cv2.imread(str(src))
        if image is None:
            skipped += 1
            if not quiet:
                print(f"  跳过读不出的文件：{src}")
            continue
        # 母本备份保留原扩展名（原样拷贝，不重新编码，零损失）；灰度图统一存 png。
        name_stem = f"{run_stamp}_{saved:06d}"
        if raw_dir is not None:
            shutil.copy2(src, raw_dir / f"{name_stem}{src.suffix.lower()}")
        cv2.imwrite(str(gray_dir / f"{name_stem}.png"), to_low_gray(image, size))
        saved += 1
        if not quiet and saved % 50 == 0:
            print(f"  已处理 {saved} 张...")

    if skipped and not quiet:
        print(f"  共跳过 {skipped} 个无法读取的文件。")
    return saved


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser()
    size = (int(args.size[0]), int(args.size[1]))
    keep_raw = not args.no_raw

    kind = classify_input(input_path)

    # 运行时间戳：精确到秒，作为本次运行的输出子目录名 + 文件名前缀，防覆盖。
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).expanduser()
    gray_dir, raw_dir = make_output_dirs(output_root, run_stamp, keep_raw)

    if not args.quiet:
        print(f"输入：{input_path}（{'视频' if kind == 'video' else '图片'}）")
        print(f"目标灰度尺寸：{size[0]}x{size[1]}")
        print(f"输出目录：{(output_root / run_stamp).resolve()}")
        print(f"保留高清母本：{'是' if keep_raw else '否'}")
        print("-" * 40)

    if kind == "video":
        saved = process_video(
            input_path, gray_dir, raw_dir, args.interval, size, run_stamp, args.quiet
        )
    else:
        saved = process_imagedir(input_path, gray_dir, raw_dir, size, run_stamp, args.quiet)

    print("-" * 40)
    print(f"完成：共输出 {saved} 张低清灰度图 → {gray_dir.resolve()}")
    if raw_dir is not None:
        print(f"      高清母本 → {raw_dir.resolve()}")


if __name__ == "__main__":
    main()
