#!/usr/bin/env python3
"""OCR 预处理前 vs 后对比：同一批正例图，Tesseract 基础 vs 增强(deskew+自适应二值化+小字放大)。

- 直接对原图 OCR 两次（enhance=False / True），对比字符数、有效性、文本片段。
- 仅当【增强后】OCR 看起来有效时才调 DeepSeek（省 token），展示被"救回"的图的 tags。
- 用与 e2e 相同的 seed 抽样，保证是同一批图。

用法：.venv/bin/python scripts/compare_ocr_preprocess.py [--n 10] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import test_e2e_images as e2e            # noqa: E402  复用抽样/工具
from pipeline.enrich import DeepSeekEnricher  # noqa: E402
from pipeline.ocr import TesseractOCR         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import random
    images = e2e.stratified_sample(args.n, random.Random(args.seed))

    ocr_basic = TesseractOCR(lang="chi_sim+eng", enhance=False)
    ocr_enh = TesseractOCR(lang="chi_sim+eng", enhance=True)
    enricher = DeepSeekEnricher()

    print(f"同一批 {len(images)} 张（seed={args.seed}）。OCR 基础 vs 增强(deskew+二值化+放大)\n")
    rows = []
    for i, img in enumerate(images, 1):
        short = f"{img.parent.name}/{img.name}"
        b = ocr_basic.ocr(img).replace("\n", " ").strip()
        e = ocr_enh.ocr(img).replace("\n", " ").strip()
        b_valid = e2e.ocr_looks_valid(b)
        e_valid = e2e.ocr_looks_valid(e)
        # 只在增强后有效时调 DeepSeek（省 token）
        tags = None
        if e_valid:
            tags = enricher.enrich(e, {"timestamp": "2026-07-06T10:00:00+00:00",
                                       "trigger_confidence": 0.9})
        rows.append({"n": i, "img": short, "b": b, "e": e,
                     "b_valid": b_valid, "e_valid": e_valid, "tags": tags})
        print(f"[{i}/{len(images)}] {short}\n"
              f"    基础  chars={len(b):5d} valid={b_valid}  | {e2e.first_words(b, 12)[:70]}\n"
              f"    增强  chars={len(e):5d} valid={e_valid}  | {e2e.first_words(e, 12)[:70]}\n"
              f"    增强后 tags={tags}")

    # ---- 汇总 ----
    b_ok = sum(1 for r in rows if r["b_valid"])
    e_ok = sum(1 for r in rows if r["e_valid"])
    rescued = [r for r in rows if r["e_valid"] and not r["b_valid"]]
    lost = [r for r in rows if r["b_valid"] and not r["e_valid"]]

    print("\n" + "=" * 90)
    print("汇总（OCR 有效文本 启发式≥5词）")
    print("=" * 90)
    print(f"基础预处理有效: {b_ok}/{len(rows)}")
    print(f"增强预处理有效: {e_ok}/{len(rows)}")
    print(f"被救回（基础无效→增强有效）: {len(rescued)}  {[r['n'] for r in rescued]}")
    print(f"变差（基础有效→增强无效）: {len(lost)}  {[r['n'] for r in lost]}")
    if rescued:
        print("\n被救回的图 + 增强后 DeepSeek tags：")
        for r in rescued:
            print(f"  #{r['n']} {r['img']}\n     tags={r['tags']}")


if __name__ == "__main__":
    main()
