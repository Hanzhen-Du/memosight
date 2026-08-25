#!/usr/bin/env python3
"""探针—训练 双向零重叠守门 —— Task2 防泄漏硬门槛。

为什么需要它：`download_images.py` 的全局去重只在单个 `--output-root` 树内按 Pexels-ID 生效。
Task2 的训练负/正例下载到 `data/raw`（去重基线只覆盖 data/raw），**看不到**两个探针目录
（`data/probe_person_noscreen/`、`data/probe_person_screen/` 在 data/raw 之外），
故新下载的训练图有可能与探针撞同一张 Pexels 图（或近重复）。探针是 held-out 评估集，
任何训练图与之重叠都会污染 FP / 召回读数。

本脚本做的事：
  1. 扫所有训练图（`data/raw/**`）与所有探针图（两个探针目录）。
  2. 双判据找重叠（与 check_leakage 同口径）：
     - Pexels-ID：文件名里 ≥6 位纯数字 token 相同 ⇒ 同一张图。
     - 感知近重复：96×96 灰度 pHash 汉明 ≤ --phash-th 且 像素 Pearson 相关 ≥ --pixel-corr。
  3. 命中的**训练**图 → **quarantine（移动到 `data/_quarantine_task2/`，不删除，保留原相对路径）**，
     使探针保持完整 held-out；探针侧一张不动。打印逐条明细 + 机读结论行。

口径：只动训练侧（探针是 gold holdout）。移动而非删除（可追溯、可恢复，遵守"不删 data/"红线）。
默认 --dry-run=False 实跑；先 --dry-run 看清单再实跑更稳。

依赖：opencv、numpy（已在 requirements.txt）。复用 scripts/check_leakage.py 的 phash64/popcount64。

示例：
  .venv/bin/python scripts/guard_probe_overlap.py --dry-run
  .venv/bin/python scripts/guard_probe_overlap.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from check_leakage import phash64, popcount64  # 全仓一致的哈希口径

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
INPUT_SIZE = 96


def photo_ids(stem: str) -> set[str]:
    """从文件名 stem 提取候选 Pexels 图片 id：≥6 位纯数字 token（与 probe_fp_test 同口径）。"""
    return {t for t in re.split(r"[_\-.]", stem) if t.isdigit() and len(t) >= 6}


def collect(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def load_gray96(path: Path) -> np.ndarray | None:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    if g.shape != (INPUT_SIZE, INPUT_SIZE):
        g = cv2.resize(g, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return g


def build_index(paths: list[Path]) -> dict:
    """对一批图算 pHash + 中心化向量 + 范数（供向量化像素相关），并记录每图的 Pexels-id。"""
    n = len(paths)
    flats = np.zeros((n, INPUT_SIZE * INPUT_SIZE), np.float32)
    phashes = np.zeros(n, np.uint64)
    ids: list[set[str]] = []
    ok = np.ones(n, bool)
    for i, p in enumerate(paths):
        ids.append(photo_ids(p.stem))
        g = load_gray96(p)
        if g is None:
            ok[i] = False
            continue
        flats[i] = g.astype(np.float32).flatten()
        phashes[i] = phash64(g)
    centered = flats - flats.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1)
    return {"paths": paths, "phashes": phashes, "centered": centered,
            "norm": np.where(norm == 0, 1.0, norm), "ids": ids, "ok": ok}


def main() -> int:
    ap = argparse.ArgumentParser(description="探针—训练 双向零重叠守门（命中训练图 → quarantine）。",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--train-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--probe-dir", type=Path, action="append",
                    default=None, help="探针目录（可重复）；默认两个探针目录。")
    ap.add_argument("--quarantine", type=Path, default=Path("data/_quarantine_task2"))
    ap.add_argument("--phash-th", type=int, default=6)
    ap.add_argument("--pixel-corr", type=float, default=0.90)
    ap.add_argument("--dry-run", action="store_true", help="只报告，不移动")
    args = ap.parse_args()

    probe_dirs = args.probe_dir or [Path("data/probe_person_noscreen"),
                                    Path("data/probe_person_screen")]
    train = collect(args.train_root)
    probes: list[Path] = []
    for d in probe_dirs:
        probes += collect(d)
    print(f"训练图 {len(train)} 张 @ {args.train_root}")
    print(f"探针图 {len(probes)} 张 @ {[str(d) for d in probe_dirs]}")
    if not train or not probes:
        print("训练或探针为空，无需核对。")
        return 0

    pidx = build_index(probes)
    probe_id_union: set[str] = set().union(*pidx["ids"]) if pidx["ids"] else set()

    hits = []  # (train_path, reason, probe_path, corr)
    for tp in train:
        tids = photo_ids(tp.stem)
        id_hit = tids & probe_id_union
        if id_hit:
            # 找到具体撞 id 的探针（取第一张）
            j = next((k for k, s in enumerate(pidx["ids"]) if tids & s), None)
            pp = pidx["paths"][j].name if j is not None else f"id={sorted(id_hit)[0]}"
            hits.append((tp, "pexels_id", pp, 1.0))
            continue
        g = load_gray96(tp)
        if g is None:
            continue
        ph = phash64(g)
        ham = popcount64(np.uint64(ph) ^ pidx["phashes"])
        cand = np.nonzero((ham <= args.phash_th) & pidx["ok"])[0]
        if cand.size == 0:
            continue
        v = g.astype(np.float32).flatten()
        v = v - v.mean()
        vn = np.linalg.norm(v) or 1.0
        best = None
        for j in cand:
            corr = float(np.dot(v, pidx["centered"][j]) / (vn * pidx["norm"][j]))
            if corr >= args.pixel_corr and (best is None or corr > best[1]):
                best = (pidx["paths"][j], corr)
        if best is not None:
            hits.append((tp, "perceptual", best[0].name, round(best[1], 4)))

    n_id = sum(1 for h in hits if h[1] == "pexels_id")
    n_perc = sum(1 for h in hits if h[1] == "perceptual")
    print(f"\n重叠命中：{len(hits)} 张训练图（Pexels-ID {n_id} / 感知近重复 {n_perc}）")
    for tp, reason, pp, corr in hits[:40]:
        rel = tp.relative_to(args.train_root)
        print(f"  [{reason}] data/raw/{rel}  ↔  {pp}  (corr={corr})")
    if len(hits) > 40:
        print(f"  …(余 {len(hits) - 40} 条省略)")

    moved = 0
    if hits and not args.dry_run:
        for tp, *_ in hits:
            rel = tp.relative_to(args.train_root)
            dst = args.quarantine / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tp), str(dst))
            moved += 1
        print(f"\n已 quarantine（移动，不删除）{moved} 张 → {args.quarantine}/")
    elif hits:
        print(f"\n(dry-run) 将 quarantine {len(hits)} 张到 {args.quarantine}/（未移动）")
    else:
        print("\n零重叠：训练集与两个探针集无任何撞图/近重复。✓")

    print(f"\nGUARD overlap_hits={len(hits)} id={n_id} perceptual={n_perc} "
          f"moved={moved} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
