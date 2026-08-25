#!/usr/bin/env python3
"""Task2b 阶段二 · QC 预筛：把"疑似含可读文字屏幕"的新负例挑出来供人眼复核（不删任何图）。

动机：task2b 新负例（室内办公/会议/居家 + 屏状表面）极易混入含可读屏幕/白板/PPT/代码屏文字的图
（=正类），若误标为负例会污染负类。本脚本用启发式给每张图打"文字屏疑似度"并排序，产出 shortlist
（CSV + 拼图）供用户白天重点人眼复核。**只标注、只排序、不自动删除任何图**（删除决定权在用户）。

启发式（cv2，单图先 resize≤256 再处理，守内存护栏）：
  - rect_area：最大「亮四边形」面积占比（屏/白板/窗/画框等屏状区域，越大越可疑）。
  - text_regions：MSER 检到的「类字形」连通域数（小尺寸+合理长宽比），文字密集→数值高。经典文本区域检测，
    对纹理/图案会误报，故仅作排序信号、不作判据。
  - suspicion = text_regions 归一化 × (0.5 + 0.5·有亮矩形)：文字多 *且* 有屏状区域 → 最可疑。
排序降序，flagged = suspicion ≥ 分位阈（默认 70 分位，约标出最可疑 30%）。
产出：docs/results/task2b_results/qc_text_screen_suspects.csv + 一张/多张拼图（每格标 文件名+text/rect）。

⚠️ 这是"宁可多标、不可漏标"的粗筛：flagged 不等于"含文字屏"，未 flagged 也不保证干净。
最终判定 100% 由用户人眼完成。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
DET = 256   # 检测分辨率上限
_MSER = cv2.MSER_create()
_MSER.setMinArea(10)
_MSER.setMaxArea(4000)


def collect(dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        if d.exists():
            out += sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)
    return out


def resize_max(gray: np.ndarray, m: int) -> np.ndarray:
    h, w = gray.shape[:2]
    s = m / max(h, w)
    return gray if s >= 1.0 else cv2.resize(gray, (int(round(w*s)), int(round(h*s))),
                                            interpolation=cv2.INTER_AREA)


def bright_rect_area(g: np.ndarray) -> float:
    """最大亮四边形面积占比（屏/白板/窗/框等屏状区域），无则 0。"""
    h, w = g.shape[:2]
    area = h * w
    med = float(np.median(g))
    _, th = cv2.threshold(g, max(med, 110), 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.05 * area:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best = max(best, a / area)
    return round(best, 4)


def text_region_count(g: np.ndarray) -> int:
    """MSER 类字形连通域数：小尺寸 + 合理长宽比 + 非极端细长。文字密集→高。"""
    regions, _ = _MSER.detectRegions(g)
    n = 0
    h, w = g.shape[:2]
    for r in regions:
        x, y, bw, bh = cv2.boundingRect(r.reshape(-1, 1, 2))
        if bh == 0 or bw == 0:
            continue
        ar = bw / bh
        a = bw * bh
        if 0.05 * h <= bh <= 0.30 * h and 0.15 <= ar <= 8.0 and a <= 0.08 * h * w:
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Task2b 文字屏疑似度 QC 预筛（不删图）。")
    ap.add_argument("--dir", type=Path, action="append", required=True,
                    help="要扫描的目录（可多次）")
    ap.add_argument("--out-csv", type=Path,
                    default=Path("docs/results/task2b_results/qc_text_screen_suspects.csv"))
    ap.add_argument("--montage-dir", type=Path,
                    default=Path("docs/results/task2b_results"))
    ap.add_argument("--montage-top", type=int, default=48)
    ap.add_argument("--pct", type=float, default=70.0, help="flagged 分位阈")
    args = ap.parse_args()

    imgs = collect(args.dir)
    rows = []
    for p in imgs:
        full = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if full is None:
            continue
        g = resize_max(full, DET)
        del full
        tr = text_region_count(g)
        ra = bright_rect_area(g)
        rows.append({"path": str(p), "text_regions": tr, "rect_area": ra,
                     "brightness": round(float(g.mean())/255.0, 3)})

    if not rows:
        print("[qc] 无图可扫")
        return 0
    tmax = max(r["text_regions"] for r in rows) or 1
    for r in rows:
        r["suspicion"] = round((r["text_regions"]/tmax) * (0.5 + 0.5*(r["rect_area"] > 0)), 4)
    rows.sort(key=lambda r: -r["suspicion"])
    thr = float(np.percentile([r["suspicion"] for r in rows], args.pct))
    for r in rows:
        r["flagged"] = int(r["suspicion"] >= thr and r["text_regions"] > 0)
    n_flag = sum(r["flagged"] for r in rows)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "suspicion", "text_regions",
                                          "rect_area", "brightness", "flagged"])
        w.writeheader()
        w.writerows(rows)

    # 拼图：疑似度最高的 top-N（每格标 text/rect）
    top = [r for r in rows if r["flagged"]][:args.montage_top]
    if top:
        thumbs = []
        for r in top:
            g = cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            t = cv2.cvtColor(cv2.resize(g, (160, 160), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_GRAY2BGR)
            cv2.putText(t, f"t{r['text_regions']} r{r['rect_area']:.2f}", (3, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            thumbs.append(t)
        cols = min(6, len(thumbs))
        rows_n = (len(thumbs) + cols - 1) // cols
        grid = np.full((rows_n*160, cols*160, 3), 30, np.uint8)
        for i, t in enumerate(thumbs):
            rr, cc = divmod(i, cols)
            grid[rr*160:(rr+1)*160, cc*160:(cc+1)*160] = t
        mp = args.montage_dir / "qc_text_screen_suspects_montage.png"
        cv2.imwrite(str(mp), grid)
        print(f"[qc] 拼图 → {mp}（top {len(thumbs)} 疑似）")

    print(f"[qc] 扫描 {len(rows)} 张，flagged 疑似含文字屏 {n_flag} 张（分位阈 {thr:.3f}）→ {args.out_csv}")
    print(f"[qc] ⚠️ 仅供人眼复核，未删除任何图。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
