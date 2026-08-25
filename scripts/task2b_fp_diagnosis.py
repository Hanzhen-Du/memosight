#!/usr/bin/env python3
"""Task2b 阶段一 · 误判诊断：C_wide_uniform(int8) 在 noscreen 探针 @0.40 的 FP 共性分析。

只读、不重训、低内存（逐图处理，全分辨率单图读入后立即降采样并释放，绝不整批驻留全分辨率）。
对 noscreen 探针每张图提取可分析维度：
  - 预测分数（int8 部署口径，cv2 灰度→resize96 INTER_AREA→量化，与 probe_fp_test 一致）
  - 场景标签（来自子目录名）
  - 亮度 mean / 对比度 std（96 灰度，[0,1]）
  - 人脸数代理（Haar frontalface，在 ≤320px 灰度上跑；仅作粗略「人数/正脸」信号）
  - 类屏矩形信号（在 ≤256px 灰度上找大面积亮四边形轮廓；noscreen 本无屏，命中即"几何误导线索"）
按 @0.40 切 FP（score≥0.40）。导出 FP 清单+分数、按场景聚合、FP vs 正确拒识的维度对比。
写 docs/false-positive-diagnosis.md + docs/results/task2b_results/noscreen_fp_per_image.csv。

防泄漏：探针仅用于评估，绝不进训练；这里只读不写训练集。仍按 Pexels-ID 核对探针与训练池无撞图（应=0）。
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

from probe_fp_test import (collect_images, manifest_id_set, photo_ids,
                           load_int8, int8_predict_one, INPUT_SIZE)

MODEL = Path("models/task1_candidates/gatekeeper_task1_C_wide_uniform_int8.tflite")
PROBE = Path("data/probe_person_noscreen")
LEAK_MANIFEST = Path("data/processed/manifest.csv")
THR = 0.40
OUT_MD = Path("docs/false-positive-diagnosis.md")
OUT_CSV = Path("docs/results/task2b_results/noscreen_fp_per_image.csv")

_FACE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def scene_of(path: Path) -> str:
    """场景标签 = 探针子目录名（noscreen 下每个子目录是一类人像场景）。"""
    rel = path.relative_to(PROBE)
    return rel.parts[0] if len(rel.parts) > 1 else "(root)"


def _resize_max(gray: np.ndarray, max_side: int) -> np.ndarray:
    h, w = gray.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return gray
    return cv2.resize(gray, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def face_count(gray_small: np.ndarray) -> int:
    faces = _FACE.detectMultiScale(gray_small, scaleFactor=1.1,
                                   minNeighbors=5, minSize=(24, 24))
    return len(faces)


def screen_like_rect(gray_small: np.ndarray) -> int:
    """粗检大面积亮四边形区域（窗/框/合盖笔电/门洞等几何误导线索）。命中返回 1。

    启发式、有噪声：阈值二值化→找轮廓→approxPolyDP 取 4 角凸多边形，
    面积 ≥ 全图 8% 且亮度高于全局中位数。仅作群体层面信号，不作单图判据。
    """
    h, w = gray_small.shape[:2]
    area = h * w
    med = float(np.median(gray_small))
    _, th = cv2.threshold(gray_small, max(med, 110), 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.08 * area:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return 1
    return 0


def main() -> int:
    assert MODEL.exists(), f"模型不存在: {MODEL}"
    interp = load_int8(MODEL)
    mids = manifest_id_set(LEAK_MANIFEST)

    imgs = collect_images(PROBE)
    rows, leaked = [], 0
    for p in imgs:
        if photo_ids(p.stem) & mids:
            leaked += 1
            continue
        full = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if full is None:
            continue
        g96 = cv2.resize(full, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        g_face = _resize_max(full, 320)
        g_rect = _resize_max(full, 256)
        del full  # 立即释放全分辨率，内存只持单图瞬时
        score = float(int8_predict_one(interp, g96))
        rows.append({
            "file": str(p.relative_to(PROBE)),
            "scene": scene_of(p),
            "score": round(score, 4),
            "is_fp": int(score >= THR),
            "brightness": round(float(g96.mean()) / 255.0, 4),
            "contrast": round(float(g96.std()) / 255.0, 4),
            "faces": face_count(g_face),
            "screen_rect": screen_like_rect(g_rect),
        })

    n = len(rows)
    fps = [r for r in rows if r["is_fp"]]
    crs = [r for r in rows if not r["is_fp"]]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["score"]))

    # 按场景聚合
    scenes = sorted({r["scene"] for r in rows})
    per_scene = []
    for s in scenes:
        sr = [r for r in rows if r["scene"] == s]
        sf = [r for r in sr if r["is_fp"]]
        per_scene.append({
            "scene": s, "n": len(sr), "fp": len(sf),
            "fp_rate": round(len(sf) / len(sr), 3) if sr else 0.0,
            "mean_score": round(float(np.mean([r["score"] for r in sr])), 3),
            "faces_mean": round(float(np.mean([r["faces"] for r in sr])), 2),
        })
    per_scene.sort(key=lambda d: (-d["fp_rate"], -d["n"]))

    def mean(key, sub):
        return round(float(np.mean([r[key] for r in sub])), 3) if sub else 0.0

    # ── 写 markdown ──
    L = []
    L.append("# Task2b 阶段一 · noscreen 探针 FP 误判诊断\n")
    L.append(f"模型：`{MODEL}`（C_wide_uniform int8，task1 胜出）　阈值 **@{THR}**　"
             f"口径：int8 部署预处理（cv2 灰度→resize96 INTER_AREA→量化）。\n")
    L.append(f"探针：noscreen **{n}** 张（leak 核对剔除 {leaked} 张，按 Pexels-ID）。"
             f"⚠️ 探针仅评估，**不入任何训练集**。\n")
    L.append(f"\n## 总览\n")
    L.append(f"- FP（被误判为「记」，score≥{THR}）：**{len(fps)}/{n} = {len(fps)/n:.3f}**\n")
    L.append(f"- 正确拒识：{len(crs)}/{n}\n")
    L.append(f"- 全体分数：min {min(r['score'] for r in rows):.3f} / "
             f"median {np.median([r['score'] for r in rows]):.3f} / "
             f"max {max(r['score'] for r in rows):.3f}\n")

    L.append(f"\n## 按场景聚合（FP 率降序）\n")
    L.append("| 场景（子目录） | n | FP | FP率 | 平均分 | 平均人脸数 |\n|---|---:|---:|---:|---:|---:|\n")
    for d in per_scene:
        L.append(f"| {d['scene']} | {d['n']} | {d['fp']} | {d['fp_rate']} | "
                 f"{d['mean_score']} | {d['faces_mean']} |\n")

    L.append(f"\n## FP vs 正确拒识 · 维度对比（均值）\n")
    L.append("| 维度 | FP（n={}） | 正确拒识（n={}） | 差异 |\n|---|---:|---:|---:|\n"
             .format(len(fps), len(crs)))
    for key, label in [("brightness", "亮度"), ("contrast", "对比度"),
                       ("faces", "人脸数代理"), ("screen_rect", "类屏矩形命中率")]:
        a, b = mean(key, fps), mean(key, crs)
        L.append(f"| {label} | {a} | {b} | {round(a-b,3):+} |\n")

    L.append(f"\n## FP 清单（score 降序，全部 {len(fps)} 张）\n")
    L.append("| # | score | 场景 | 文件 | 人脸 | 亮度 | 类屏 |\n|---:|---:|---|---|---:|---:|---:|\n")
    for i, r in enumerate(sorted(fps, key=lambda r: -r["score"]), 1):
        L.append(f"| {i} | {r['score']} | {r['scene']} | {r['file']} | "
                 f"{r['faces']} | {r['brightness']} | {r['screen_rect']} |\n")

    L.append(f"\n## 借近阈值的「擦边正确拒识」（{THR}>score≥{THR-0.08:.2f}，最易翻车）\n")
    near = sorted([r for r in crs if r["score"] >= THR - 0.08], key=lambda r: -r["score"])
    L.append(f"共 {len(near)} 张（这些是再补一点同类负例最可能压下去的边缘案例）：\n\n")
    L.append("| score | 场景 | 文件 |\n|---:|---|---|\n")
    for r in near:
        L.append(f"| {r['score']} | {r['scene']} | {r['file']} |\n")

    OUT_MD.write_text("".join(L), encoding="utf-8")

    print(f"[diag] n={n} leak={leaked} FP={len(fps)} ({len(fps)/n:.3f}) → {OUT_MD}")
    print("[diag] per-scene FP率:", {d["scene"]: d["fp_rate"] for d in per_scene})
    print(f"[diag] CSV → {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
