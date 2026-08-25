#!/usr/bin/env python3
"""OCR preprocessing, before against after: the same set of positive images, Tesseract base
against enhanced (deskew, adaptive threshold, upscaling small text).

- OCR each original twice, with enhance=False and enhance=True, and compare character count,
  validity and a text excerpt.
- Only call DeepSeek when the enhanced OCR looks valid, which saves tokens, to show the tags
  for images the enhancement rescued.
- Sample with the same seed as the end-to-end test, so it is the same set of images.

Usage: .venv/bin/python scripts/compare_ocr_preprocess.py [--n 10] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import test_e2e_images as e2e            # noqa: E402  reuse its sampling and helpers
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

    print(f"the same {len(images)} images (seed={args.seed}). OCR base against enhanced "
          "(deskew, threshold, upscale)\n")
    rows = []
    for i, img in enumerate(images, 1):
        short = f"{img.parent.name}/{img.name}"
        b = ocr_basic.ocr(img).replace("\n", " ").strip()
        e = ocr_enh.ocr(img).replace("\n", " ").strip()
        b_valid = e2e.ocr_looks_valid(b)
        e_valid = e2e.ocr_looks_valid(e)
        # Only call DeepSeek when the enhanced result is valid, which saves tokens
        tags = None
        if e_valid:
            tags = enricher.enrich(e, {"timestamp": "2026-07-06T10:00:00+00:00",
                                       "trigger_confidence": 0.9})
        rows.append({"n": i, "img": short, "b": b, "e": e,
                     "b_valid": b_valid, "e_valid": e_valid, "tags": tags})
        print(f"[{i}/{len(images)}] {short}\n"
              f"    base      chars={len(b):5d} valid={b_valid}  | {e2e.first_words(b, 12)[:70]}\n"
              f"    enhanced  chars={len(e):5d} valid={e_valid}  | {e2e.first_words(e, 12)[:70]}\n"
              f"    enhanced tags={tags}")

    # ---- summary ----
    b_ok = sum(1 for r in rows if r["b_valid"])
    e_ok = sum(1 for r in rows if r["e_valid"])
    rescued = [r for r in rows if r["e_valid"] and not r["b_valid"]]
    lost = [r for r in rows if r["b_valid"] and not r["e_valid"]]

    print("\n" + "=" * 90)
    print("Summary. OCR text counts as valid by the heuristic of at least 5 words.")
    print("=" * 90)
    print(f"base preprocessing valid:     {b_ok}/{len(rows)}")
    print(f"enhanced preprocessing valid: {e_ok}/{len(rows)}")
    print(f"rescued (base invalid, enhanced valid): {len(rescued)}  {[r['n'] for r in rescued]}")
    print(f"broken (base valid, enhanced invalid):  {len(lost)}  {[r['n'] for r in lost]}")
    if rescued:
        print("\nrescued images and their DeepSeek tags after enhancement:")
        for r in rescued:
            print(f"  #{r['n']} {r['img']}\n     tags={r['tags']}")


if __name__ == "__main__":
    main()
