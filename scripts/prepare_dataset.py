#!/usr/bin/env python3
"""数据准备流水线（降清 → 标签 → 切分）—— MemoSight 守门员训练数据打包工具。

把 data/raw/ 下按目录分好类的原始图片，处理成守门员训练直接可用的形态：

  1. 降清：遍历三大类目录下所有图片（含子目录），转成 size×size 单通道灰度，
     用区域插值（INTER_AREA）下采样避免摩尔纹，保存到 data/processed/。
     保留原始相对目录结构，文件名追加原路径 hash 防重名。
  2. 标签：直接用目录结构映射，不逐张标注。
        positive/        → 标签 1（记）
        negative_noise/  → 标签 0（不记，噪声文字）
        negative_clean/  → 标签 0（不记，无文字）
     产出 manifest.csv：每行 = 处理后图片相对路径 + 标签 + 原始大类来源 + 细分子类。
  3. 切分：按 70/15/15 切 train/val/test，按「来源大类」分层抽样（stratified），
     保证每个 split 里正负比例一致；固定随机种子可复现。
     输出 train.csv / val.csv / test.csv。

只准备数据，不训练、不建模。

依赖：opencv-python、numpy（见 requirements.txt）。仅用标准库写 CSV，不依赖 pandas。

示例：
  python3 scripts/prepare_dataset.py                  # 用默认参数实跑
  python3 scripts/prepare_dataset.py --dry-run        # 只扫描打印统计，不写文件
  python3 scripts/prepare_dataset.py --size 96 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2

# 目录大类 → (标签, 是否正例)。守门员只做二分类：positive=记=1，其余=不记=0。
CLASS_LABELS: dict[str, int] = {
    "positive": 1,
    "negative_noise": 0,
    "negative_clean": 0,
}

# 接受的图片后缀（小写比较）。
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 切分比例，必须和为 1。test 用剩余量兜底，避免浮点误差丢样本。
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# manifest / 各 split CSV 的列。
CSV_FIELDS = ["path", "label", "source", "subclass", "split"]


def find_images(input_root: Path) -> list[dict]:
    """遍历三大类目录下所有图片（含子目录），返回样本记录列表。

    每条记录：{src(原图绝对路径), source(大类), subclass(子目录名), label}。
    跳过不在 CLASS_LABELS 里的大类目录，并提醒。
    """
    samples: list[dict] = []
    for source, label in CLASS_LABELS.items():
        class_dir = input_root / source
        if not class_dir.is_dir():
            print(f"[警告] 找不到大类目录，跳过：{class_dir}", file=sys.stderr)
            continue
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            # 子类 = 大类目录下的第一级子目录名；若图片直接放在大类根下，记为 "_root"。
            rel_parts = path.relative_to(class_dir).parts
            subclass = rel_parts[0] if len(rel_parts) > 1 else "_root"
            samples.append(
                {"src": path, "source": source, "subclass": subclass, "label": label}
            )
    return samples


def processed_relpath(sample: dict, input_root: Path) -> Path:
    """计算处理后图片相对 output_root 的路径：镜像原始相对结构 + 8位路径hash 防重名，统一存为 .png。"""
    rel = sample["src"].relative_to(input_root)
    digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:8]
    return rel.with_name(f"{rel.stem}_{digest}.png")


def stratified_split(
    samples: list[dict], seed: int
) -> dict[str, list[dict]]:
    """按「来源大类」分层抽样切 70/15/15。

    在每个大类内部独立按比例切分，再合并 —— 这样保证每个 split 里
    三大类（进而正/负）的比例都与全集一致。test 取剩余，不丢样本。
    """
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        buckets[s["source"]].append(s)

    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for source in sorted(buckets):
        group = buckets[source][:]
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])
        out["train"].extend(group[:n_train])
        out["val"].extend(group[n_train : n_train + n_val])
        out["test"].extend(group[n_train + n_val :])  # 剩余全给 test
    return out


def print_stats(title: str, rows: list[dict]) -> None:
    """打印一个集合的总数 / 正例 / 反例 / 正负比例。"""
    total = len(rows)
    pos = sum(1 for r in rows if int(r["label"]) == 1)
    neg = total - pos
    ratio = f"{pos / neg:.3f}" if neg else "∞"
    print(f"  {title:<6} 共 {total:>4} | 正 {pos:>4} | 负 {neg:>4} | 正/负 = {ratio}")


def write_image(sample: dict, dst: Path, size: int) -> bool:
    """读图→灰度→INTER_AREA 降清到 size×size→写 PNG。读失败返回 False。"""
    img = cv2.imread(str(sample["src"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[警告] 无法读取，跳过：{sample['src']}", file=sys.stderr)
        return False
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst), resized))


def write_csv(path: Path, rows: list[dict]) -> None:
    """写一个 CSV（列见 CSV_FIELDS）。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in CSV_FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MemoSight 守门员数据准备：降清→标签→分层切分。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--size", type=int, default=96, help="输出方形边长（像素）")
    parser.add_argument("--seed", type=int, default=42, help="切分随机种子")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只扫描和打印统计，不写任何文件",
    )
    args = parser.parse_args()

    input_root: Path = args.input_root
    output_root: Path = args.output_root

    if not input_root.is_dir():
        print(f"[错误] 输入根目录不存在：{input_root}", file=sys.stderr)
        return 1

    # 1) 扫描 + 标签
    samples = find_images(input_root)
    if not samples:
        print(f"[错误] 在 {input_root} 下没扫到任何图片。", file=sys.stderr)
        return 1

    # 预填每条记录的处理后相对路径（dry-run 也要，统计 manifest 用）。
    for s in samples:
        s["path"] = str(processed_relpath(s, input_root)).replace("\\", "/")

    print(f"扫描到 {len(samples)} 张图片，来自 {input_root}/")
    by_source = defaultdict(int)
    for s in samples:
        by_source[s["source"]] += 1
    for source in CLASS_LABELS:
        print(f"  {source:<16} {by_source.get(source, 0):>4} 张 (label={CLASS_LABELS[source]})")

    # 3) 分层切分
    splits = stratified_split(samples, args.seed)
    for split_name, rows in splits.items():
        for r in rows:
            r["split"] = split_name

    mode = "DRY-RUN（不写文件）" if args.dry_run else "实跑"
    print(f"\n=== 切分统计 [{mode}] | seed={args.seed} | 70/15/15 分层（按来源大类）===")
    all_rows = splits["train"] + splits["val"] + splits["test"]
    print_stats("全集", all_rows)
    for split_name in ("train", "val", "test"):
        print_stats(split_name, splits[split_name])

    # 2) + 写盘
    if not args.dry_run:
        print(f"\n开始处理图片 → {output_root}/（{args.size}×{args.size} 灰度 PNG）…")
        written = 0
        failed = 0
        for s in all_rows:
            dst = output_root / s["path"]
            if write_image(s, dst, args.size):
                written += 1
            else:
                failed += 1
        # manifest（全集）+ 三个 split CSV
        output_root.mkdir(parents=True, exist_ok=True)
        write_csv(output_root / "manifest.csv", all_rows)
        for split_name in ("train", "val", "test"):
            write_csv(output_root / f"{split_name}.csv", splits[split_name])
        print(f"图片写盘完成：成功 {written}，失败 {failed}。")
        print(
            f"已写出：{output_root}/manifest.csv、train.csv、val.csv、test.csv"
        )
        if failed:
            print(
                f"[注意] 有 {failed} 张图读取失败已跳过，但仍在 CSV 里 —— "
                f"训练前请按上面警告核查或重跑。",
                file=sys.stderr,
            )

    # 4) 不平衡提醒
    total_pos = sum(1 for s in samples if s["label"] == 1)
    total_neg = len(samples) - total_pos
    print(
        f"\n[提醒] 当前两类不平衡：正 {total_pos} vs 负 {total_neg} "
        f"（正/负 ≈ {total_pos / total_neg:.2f}）。"
        f"训练阶段请用 class weight 处理，不要在这里重采样改变数据分布。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
