#!/usr/bin/env python3
"""去重 + 分层重切分 —— 修复 check_leakage.py 查出的跨 split 泄漏。

泄漏根因：同一张 Pexels 原图被多个关键词重复下载，落入不同 positive 子类目录，
分层切分（按大类 positive 整体切）把同图副本散到 train/val/test → 评估虚高。

本脚本两步：
  1. build：对全量 manifest 的处理后图算 pHash + 像素相关，按"pHash 汉明 ≤ th
     且像素相关 ≥ corr"构连通分量，每组留一张代表（字典序最小路径，确定性），
     写出去重后清单 manifest_dedup.csv。
  2. split：读去重清单，按 --seed 做与 prepare_dataset.py 一致的"按来源大类分层"
     70/15/15 切分，写 <prefix>train.csv / val.csv / test.csv。

去重后单图只出现一次，从根上消除跨 split 泄漏；切分逻辑与原版一致，仅样本去重。

依赖：numpy、pandas、opencv-python（无新增依赖）。只读图 + 写 CSV，不动原图。

示例：
  .venv/bin/python scripts/dedup_resplit.py build
  .venv/bin/python scripts/dedup_resplit.py split --seed 42 --prefix dedup_
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# 复用泄漏检查里的哈希实现，保证去重判据与泄漏报告完全一致。
from check_leakage import phash64, popcount64  # noqa: E402

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
CSV_FIELDS = ["path", "label", "source", "subclass", "split"]


def find(parent: dict, x: str) -> str:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(parent: dict, a: str, b: str) -> None:
    ra, rb = find(parent, a), find(parent, b)
    if ra != rb:
        parent[ra] = rb


def build_dedup_manifest(data_root: Path, manifest: Path, out: Path,
                         phash_th: int, pixel_corr: float,
                         exclude_subclasses: list[str] | None = None,
                         out_excluded: Path | None = None) -> None:
    df = pd.read_csv(manifest)

    # 边界收窄：剔除"有文字但非 MVP 首发触发场景"的歧义子类（手机app/TV菜单/商品包装）。
    # 不删图，只从清单剔除；被剔除行另存归档清单，保留可追溯。
    if exclude_subclasses:
        mask = df["subclass"].isin(exclude_subclasses)
        excluded = df[mask].copy()
        df = df[~mask].copy()
        print(f"边界收窄：剔除子类 {exclude_subclasses}")
        print(f"  剔除 {len(excluded)} 张（按子类）：")
        print(excluded["subclass"].value_counts().to_string())
        if out_excluded is not None:
            out_excluded.parent.mkdir(parents=True, exist_ok=True)
            excluded.to_csv(out_excluded, index=False)
            print(f"  归档清单（不删图，仅出训练/评估）：{out_excluded}")

    n = len(df)
    paths = df["path"].to_numpy()
    imgs = np.zeros((n, 96 * 96), dtype=np.float32)
    phashes = np.zeros(n, dtype=np.uint64)
    for i, rel in enumerate(paths):
        g = cv2.imread(str(data_root / rel), cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"[警告] 读不到：{rel}", file=sys.stderr)
            continue
        if g.shape != (96, 96):
            g = cv2.resize(g, (96, 96), interpolation=cv2.INTER_AREA)
        imgs[i] = g.astype(np.float32).flatten()
        phashes[i] = phash64(g)

    centered = imgs - imgs.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1)
    norm_safe = np.where(norm == 0, 1.0, norm)

    parent = {p: p for p in paths}
    n_pairs = 0
    for i in range(n):
        ph = popcount64(phashes[i] ^ phashes[i + 1:])
        for off in np.nonzero(ph <= phash_th)[0]:
            j = i + 1 + int(off)
            corr = float(np.dot(centered[i], centered[j]) / (norm_safe[i] * norm_safe[j]))
            if corr >= pixel_corr:
                union(parent, paths[i], paths[j])
                n_pairs += 1

    groups: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        groups[find(parent, p)].append(p)
    # 每组代表 = 字典序最小路径（确定性，可复现）。
    keep = {min(members) for members in groups.values()}
    dedup_df = df[df["path"].isin(keep)].copy()

    multi = [m for m in groups.values() if len(m) > 1]
    removed = n - len(keep)
    print(f"全量 {n} 张 → 确认近重复对 {n_pairs}，重复组 {len(multi)}，"
          f"移除 {removed} 张 → 去重后 {len(keep)} 张")
    print("去重后标签分布：")
    print(dedup_df["label"].value_counts().to_string())

    out.parent.mkdir(parents=True, exist_ok=True)
    dedup_df.to_csv(out, index=False)
    print(f"去重清单已写出：{out}")


def stratified_split(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """按 source 大类分层 70/15/15，与 prepare_dataset.py 同逻辑（test 取剩余）。"""
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    for source in sorted(df["source"].unique()):
        group = df[df["source"] == source].to_dict("records")
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])
        out["train"].extend(group[:n_train])
        out["val"].extend(group[n_train:n_train + n_val])
        out["test"].extend(group[n_train + n_val:])
    return {k: pd.DataFrame(v) for k, v in out.items()}


def do_split(manifest: Path, out_dir: Path, prefix: str, seed: int,
             quiet: bool = False) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(manifest)
    splits = stratified_split(df, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sdf in splits.items():
        sdf = sdf.copy()
        sdf["split"] = name
        sdf[CSV_FIELDS].to_csv(out_dir / f"{prefix}{name}.csv", index=False)
    if not quiet:
        print(f"[seed={seed}] 切分（去重清单 {len(df)} 张）：")
        for name in ("train", "val", "test"):
            s = splits[name]
            pos = int((s["label"] == 1).sum())
            print(f"  {name:<5} {len(s):>4} | 正 {pos:>3} | 负 {len(s) - pos:>3} | "
                  f"正/负={pos / max(1, len(s) - pos):.3f}")
        print(f"  → {out_dir}/{prefix}{{train,val,test}}.csv")
    return splits


def main() -> int:
    p = argparse.ArgumentParser(description="去重 + 分层重切分。")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="构建去重清单")
    pb.add_argument("--data-root", type=Path, default=Path("data/processed"))
    pb.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    pb.add_argument("--out", type=Path, default=Path("data/processed/manifest_dedup.csv"))
    pb.add_argument("--phash-th", type=int, default=6)
    pb.add_argument("--pixel-corr", type=float, default=0.90)
    pb.add_argument("--exclude-subclass", action="append", default=None,
                    help="剔除指定子类（可重复）；边界收窄用，被剔除行另存归档清单。")
    pb.add_argument("--excluded-out", type=Path,
                    default=Path("data/processed/manifest_out_of_scope.csv"),
                    help="被剔除行的归档清单输出路径。")

    ps = sub.add_parser("split", help="对去重清单分层切分")
    ps.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_dedup.csv"))
    ps.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ps.add_argument("--prefix", type=str, default="dedup_")
    ps.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.cmd == "build":
        build_dedup_manifest(args.data_root, args.manifest, args.out,
                             args.phash_th, args.pixel_corr,
                             exclude_subclasses=args.exclude_subclass,
                             out_excluded=args.excluded_out)
    elif args.cmd == "split":
        do_split(args.manifest, args.out_dir, args.prefix, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
