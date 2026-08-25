#!/usr/bin/env python3
"""数据泄漏检查 —— 检测 train/val/test 间的重复 / 近重复图。

方法（两道关卡，先廉价后精确）：
  1. 感知哈希粗筛：对每张处理后 96×96 灰度图算 pHash(DCT) + dHash，
     全对比较 Hamming 距离，挑出 pHash 距离 ≤ --phash-th 的候选对。
     （感知哈希对亮度/轻微缩放鲁棒，专为"看起来一样"设计。）
  2. 像素级二次确认：对候选对计算 96×96 像素的 Pearson 相关系数 + 归一化 MSE，
     只有相关性 ≥ --pixel-corr 的才判定为真·近重复，滤掉哈希碰撞误报。

判定后按是否跨 split 分类，**重点报告跨 split 的重复对数量**（这才是泄漏）。

依赖：numpy、pandas、opencv-python（均在 requirements.txt 内，无新增依赖）。
不修改任何数据，只读图 + 写一份 CSV 报告（候选对明细）。

示例：
  .venv/bin/python scripts/check_leakage.py
  .venv/bin/python scripts/check_leakage.py --phash-th 6 --pixel-corr 0.90
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def phash64(gray: np.ndarray) -> np.uint64:
    """DCT 感知哈希：32×32 → DCT → 取左上 8×8（去掉 DC）→ 与中位数比较 → 64 位。"""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    block = dct[:8, :8].flatten()
    med = np.median(block[1:])  # 去掉 [0,0] DC 分量再取中位数
    bits = block > med
    bits[0] = False  # DC 位固定，不参与
    out = np.uint64(0)
    for b in bits:
        out = (out << np.uint64(1)) | np.uint64(1 if b else 0)
    return out


def dhash64(gray: np.ndarray) -> np.uint64:
    """差值哈希：9×8 → 相邻列比较 → 8×8=64 位。"""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int32)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    out = np.uint64(0)
    for b in bits:
        out = (out << np.uint64(1)) | np.uint64(1 if b else 0)
    return out


def popcount64(x: np.ndarray) -> np.ndarray:
    """对 uint64 数组做按位 1 计数（SWAR 算法，向量化）。"""
    x = x.astype(np.uint64)
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    h01 = np.uint64(0x0101010101010101)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return (x * h01) >> np.uint64(56)


def main() -> int:
    p = argparse.ArgumentParser(
        description="跨 split 重复/近重复（数据泄漏）检查。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=Path("data/processed"))
    p.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    p.add_argument("--phash-th", type=int, default=6,
                   help="pHash 汉明距离阈值，≤ 此值进入候选（粗筛）")
    p.add_argument("--pixel-corr", type=float, default=0.90,
                   help="像素 Pearson 相关阈值，≥ 此值判定真近重复（精筛）")
    p.add_argument("--out", type=Path,
                   default=Path("docs/results/leakage_candidates.csv"))
    args = p.parse_args()

    df = pd.read_csv(args.manifest)
    n = len(df)
    print(f"读取 manifest：{n} 张图，split 分布：")
    print(df["split"].value_counts().to_string())

    # 载入全部图（96×96 灰度），同时算 pHash / dHash。
    imgs = np.zeros((n, 96 * 96), dtype=np.float32)
    phashes = np.zeros(n, dtype=np.uint64)
    dhashes = np.zeros(n, dtype=np.uint64)
    bad = 0
    for i, rel in enumerate(df["path"]):
        g = cv2.imread(str(args.data_root / rel), cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"[警告] 读不到：{rel}", file=sys.stderr)
            bad += 1
            continue
        if g.shape != (96, 96):
            g = cv2.resize(g, (96, 96), interpolation=cv2.INTER_AREA)
        imgs[i] = g.astype(np.float32).flatten()
        phashes[i] = phash64(g)
        dhashes[i] = dhash64(g)
    if bad:
        print(f"[警告] {bad} 张读取失败。", file=sys.stderr)

    # 预算每张图的均值/标准差，供向量化 Pearson 相关。
    mean = imgs.mean(axis=1, keepdims=True)
    centered = imgs - mean
    norm = np.linalg.norm(centered, axis=1)  # 每行 L2 范数
    norm_safe = np.where(norm == 0, 1.0, norm)

    splits = df["split"].to_numpy()
    paths = df["path"].to_numpy()
    subclasses = df["subclass"].to_numpy() if "subclass" in df else np.array([""] * n)
    labels = df["label"].to_numpy()

    # 全对粗筛：对每个 i，算它与 j>i 的 pHash 汉明距离，挑候选。
    records = []
    for i in range(n):
        ph = popcount64(phashes[i] ^ phashes[i + 1:])
        cand = np.nonzero(ph <= args.phash_th)[0]
        for off in cand:
            j = i + 1 + int(off)
            # 像素 Pearson 相关（向量化的两行点积 / 范数积）
            corr = float(np.dot(centered[i], centered[j]) / (norm_safe[i] * norm_safe[j]))
            dh = int(popcount64(np.array([dhashes[i] ^ dhashes[j]], dtype=np.uint64))[0])
            mse = float(np.mean((imgs[i] - imgs[j]) ** 2)) / (255.0 ** 2)
            records.append({
                "path_a": paths[i], "path_b": paths[j],
                "split_a": splits[i], "split_b": splits[j],
                "label_a": int(labels[i]), "label_b": int(labels[j]),
                "subclass_a": subclasses[i], "subclass_b": subclasses[j],
                "phash_hamming": int(ph[off]), "dhash_hamming": dh,
                "pixel_corr": round(corr, 4), "norm_mse": round(mse, 5),
                "cross_split": splits[i] != splits[j],
            })

    cand_df = pd.DataFrame(records)
    print(f"\npHash 候选对（汉明 ≤ {args.phash_th}）：{len(cand_df)} 对")
    if cand_df.empty:
        print("没有任何近重复候选 —— 不存在泄漏。")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cand_df.to_csv(args.out, index=False)
        return 0

    # 精筛：像素相关 ≥ 阈值 才算真·近重复。
    confirmed = cand_df[cand_df["pixel_corr"] >= args.pixel_corr].copy()
    exact = confirmed[confirmed["pixel_corr"] >= 0.999]
    near = confirmed[confirmed["pixel_corr"] < 0.999]

    print(f"像素确认（corr ≥ {args.pixel_corr}）：{len(confirmed)} 对真近重复"
          f"（其中近乎一致 corr≥0.999：{len(exact)} 对；高相似：{len(near)} 对）")

    cross = confirmed[confirmed["cross_split"]]
    within = confirmed[~confirmed["cross_split"]]
    print(f"\n>>> 跨 split 重复对（= 泄漏）：{len(cross)} 对")
    print(f"    split 内部重复对（非泄漏，但影响数据质量）：{len(within)} 对")

    if len(cross):
        print("\n跨 split 重复对明细（前 30）：")
        cols = ["split_a", "split_b", "label_a", "label_b",
                "phash_hamming", "pixel_corr", "path_a", "path_b"]
        print(cross[cols].head(30).to_string(index=False))
        print("\n跨 split 重复按 (split 对) 计数：")
        pair_key = cross.apply(
            lambda r: " ↔ ".join(sorted([r["split_a"], r["split_b"]])), axis=1)
        print(pair_key.value_counts().to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cand_df.to_csv(args.out, index=False)
    print(f"\n候选对明细已写出：{args.out}（{len(cand_df)} 行）")

    # 给出机读结论行。
    print(f"\nLEAKAGE cross_split_pairs={len(cross)} within_split_pairs={len(within)} "
          f"exact={len(exact)} near={len(near)} candidates={len(cand_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
