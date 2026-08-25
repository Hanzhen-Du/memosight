#!/usr/bin/env python3
"""End-to-end verification: sample N positive images from data/raw/positive/, run each through
the whole loop, and summarise the results.

The chain is entirely real except the gatekeeper trigger, which is mocked: real image,
grab_frame, real Tesseract OCR, real DeepSeek enrichment, stored to SQLite in a separate test
database so the production memory database is untouched, then a summary table and statistics.

Usage:
  .venv/bin/python scripts/test_e2e_images.py [--n 10] [--seed 42] [--enricher deepseek]

Safety:
- Uses a separate test database, data/mvp_demo/test_e2e.db by default, and never writes to the
  production one.
- grab_frame copies the test image into frames_dir before processing, so the originals in
  data/raw/positive/ are never touched.
- Images are resized by max_side inside the OCR preprocessing before anything else, which
  avoids memory problems.
- Only N images are sampled, which bounds DeepSeek token spend.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import config as config_mod            # noqa: E402
from pipeline.connectivity import ConnectivityMock   # noqa: E402
from pipeline.enrich import (                         # noqa: E402
    CloudEnricher, DeepSeekEnricher, EnricherConfigError, MockCloudEnricher,
)
from pipeline.pipeline import build_pipeline          # noqa: E402

POSITIVE_ROOT = REPO_ROOT / "data" / "raw" / "positive"
IMG_EXTS = (".jpg", ".jpeg", ".png")


def list_all_images(folder: Path) -> list[Path]:
    """List every image in a folder recursively, sorted by filename. Used to run a curated
    demo set rather than a sample."""
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]
    return sorted(imgs, key=lambda p: p.name)


def stratified_sample(n: int, rng: random.Random) -> list[Path]:
    """Stratified sampling across categories: take one image from as many different scene
    subdirectories as possible, for broad coverage."""
    cats = [d for d in sorted(POSITIVE_ROOT.iterdir()) if d.is_dir()]
    cats = [d for d in cats if any(f.suffix.lower() in IMG_EXTS for f in d.iterdir())]
    rng.shuffle(cats)
    picked: list[Path] = []
    # Take one per category until n is reached; if there are not enough categories, go round again
    round_no = 0
    while len(picked) < n and round_no < 10:
        for d in cats:
            if len(picked) >= n:
                break
            imgs = [f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS]
            imgs = [f for f in imgs if f not in picked]
            if imgs:
                picked.append(rng.choice(imgs))
        round_no += 1
    return picked[:n]


def make_enricher(choice: str):
    if choice == "mock":
        return MockCloudEnricher()
    if choice == "claude":
        return CloudEnricher()
    return DeepSeekEnricher()


def first_words(text: str, n: int = 15) -> str:
    toks = text.split()
    s = " ".join(toks[:n])
    return (s + " …") if len(toks) > n else s


def ocr_looks_valid(text: str) -> bool:
    """Heuristic for whether OCR produced usable text: at least 5 words of 3 or more
    characters. Deliberately loose, and only used for coarse statistics."""
    # The escaped range is the CJK block; behaviour is identical to writing the characters out
    words = re.findall(r"[A-Za-z\u4e00-\u9fff]{3,}", text)
    return len(words) >= 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dir", default=None,
                    help="run every image in this folder, without sampling. For a curated demo set")
    ap.add_argument("--enricher", choices=("deepseek", "claude", "mock"), default="deepseek")
    ap.add_argument("--db", default=str(REPO_ROOT / "data" / "mvp_demo" / "test_e2e.db"))
    args = ap.parse_args()

    if args.dir:
        images = list_all_images(Path(args.dir))
        print(f"running all {len(images)} images in {args.dir}, no sampling\n")
    else:
        images = stratified_sample(args.n, random.Random(args.seed))
        print(f"sampled with seed={args.seed}: {len(images)} images across "
              f"{len({p.parent.name for p in images})} categories\n")

    cfg = config_mod.Config(
        db_path=Path(args.db),
        frames_dir=REPO_ROOT / "data" / "mvp_demo" / "test_frames",
        cache_dir=REPO_ROOT / "data" / "mvp_demo" / "test_cache",
    )
    pipe = build_pipeline(cfg=cfg, connectivity=ConnectivityMock(online=True),
                          enricher=make_enricher(args.enricher))
    print(f"OCR engine={pipe.ocr.name} | enricher={pipe.ingest.transport.enricher.name} | "
          f"db={args.db}\n")

    rows = []
    for i, img in enumerate(images, 1):
        short = f"{img.parent.name}/{img.name}"
        rec = {"n": i, "img": short, "ocr": "", "tags": None, "status": "?", "err": ""}
        try:
            card = pipe.capture(img, trigger_confidence=0.9)  # mocked trigger, frame, OCR, enrich, store
            rec["ocr"] = (card.ocr_text or "").replace("\n", " ").strip()
            rec["tags"] = card.tags
            rec["status"] = card.status  # done means enrichment succeeded; pending means a transient
                                         # DeepSeek failure put it in the queue
        except EnricherConfigError as e:
            rec["status"] = "config-error"
            rec["err"] = str(e)[:80]
        except Exception as e:  # frame grab, OCR and similar failures, recorded as they are
            rec["status"] = "error"
            rec["err"] = f"{type(e).__name__}: {e}"[:80]
        rows.append(rec)
        print(f"[{i}/{len(images)}] {short} -> status={rec['status']}, "
              f"ocr_chars={len(rec['ocr'])}, tags={rec['tags']}")

    # ---- summary table ----
    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)
    hdr = f'{"#":<3}{"image (category/name)":<48}{"first ~15 OCR words":<44}{"tags":<44}{"ok?"}'
    print(hdr)
    print("-" * 100)
    for r in rows:
        ok = "ok" if r["status"] == "done" else ("pending" if r["status"] == "pending" else "fail")
        tags = ",".join(r["tags"]) if r["tags"] else ("[]" if r["tags"] == [] else "-")
        print(f'{r["n"]:<3}{r["img"][:46]:<48}{first_words(r["ocr"])[:42]:<44}{tags[:42]:<44}{ok}')
        if r["err"]:
            print(f'     ! {r["err"]}')

    # ---- statistics ----
    total = len(rows)
    done = sum(1 for r in rows if r["status"] == "done")
    pending = sum(1 for r in rows if r["status"] == "pending")
    errored = sum(1 for r in rows if r["status"] in ("error", "config-error"))
    ocr_valid = sum(1 for r in rows if ocr_looks_valid(r["ocr"]))
    ocr_thin = total - ocr_valid
    tags_nonempty = sum(1 for r in rows if r["tags"])
    tags_empty = sum(1 for r in rows if r["tags"] == [])

    print("\n" + "=" * 100)
    print("Statistics")
    print("=" * 100)
    print(f"total: {total}")
    print(f"end-to-end cards stored (status=done): {done}/{total}")
    print(f"  of which DeepSeek returned non-empty tags: {tags_nonempty}; "
          f"returned [] because the text was noise: {tags_empty}")
    print(f"queued as pending after a transient DeepSeek failure: {pending}")
    print(f"errors in OCR, frame grab and similar: {errored}")
    print(f"OCR produced usable text by the 5-word heuristic: {ocr_valid}/{total}; "
          f"essentially empty or noise: {ocr_thin}/{total}")
    ds_calls = done + pending  # how many DeepSeek calls were actually attempted
    if ds_calls:
        print(f"DeepSeek call success rate: {done}/{ds_calls} = {done/ds_calls*100:.0f}%")

    pipe.close()


if __name__ == "__main__":
    main()
