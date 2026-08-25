#!/usr/bin/env python3
"""端到端验证：从 data/raw/positive/ 随机抽 N 张正例图，逐张走完整条闭环，汇总结果。

链路（全真，除守门员触发是 mock）：真图 → grab_frame → 真 Tesseract OCR → 真 DeepSeek enrich
→ 存 SQLite（**独立测试库**，不污染正式 memory 库）→ 汇总表 + 统计。

用法：
  .venv/bin/python scripts/test_e2e_images.py [--n 10] [--seed 42] [--enricher deepseek]

安全：
- 用独立测试库（默认 data/mvp_demo/test_e2e.db），不写正式库。
- grab_frame 只**拷贝**测试图到 frames_dir 再处理，原图 data/raw/positive/ 绝不动。
- 加载图片在 OCR preprocess 里先 resize（max_side）再处理，避免内存问题。
- 只抽 N 张，控制 DeepSeek token 消耗。
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
    """按文件名排序列出文件夹里所有图（递归）。用于跑精选演示图，不抽样。"""
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]
    return sorted(imgs, key=lambda p: p.name)


def stratified_sample(n: int, rng: random.Random) -> list[Path]:
    """跨类别分层抽样：尽量从不同场景子目录各抽一张，覆盖多类别。"""
    cats = [d for d in sorted(POSITIVE_ROOT.iterdir()) if d.is_dir()]
    cats = [d for d in cats if any(f.suffix.lower() in IMG_EXTS for f in d.iterdir())]
    rng.shuffle(cats)
    picked: list[Path] = []
    # 先每类抽 1 张，直到凑够 n；类别不够再回头多抽
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
    """启发式：认为 OCR 出了有效文本 = 至少 5 个 >=3 字母的词（宽松，仅用于统计粗分类）。"""
    words = re.findall(r"[A-Za-z一-鿿]{3,}", text)
    return len(words) >= 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dir", default=None,
                    help="跑指定文件夹里所有图（不抽样，用于精选演示图）")
    ap.add_argument("--enricher", choices=("deepseek", "claude", "mock"), default="deepseek")
    ap.add_argument("--db", default=str(REPO_ROOT / "data" / "mvp_demo" / "test_e2e.db"))
    args = ap.parse_args()

    if args.dir:
        images = list_all_images(Path(args.dir))
        print(f"跑文件夹 {args.dir} 全部 {len(images)} 张（不抽样）\n")
    else:
        images = stratified_sample(args.n, random.Random(args.seed))
        print(f"抽样 seed={args.seed}，抽到 {len(images)} 张（跨 {len({p.parent.name for p in images})} 个类别）\n")

    cfg = config_mod.Config(
        db_path=Path(args.db),
        frames_dir=REPO_ROOT / "data" / "mvp_demo" / "test_frames",
        cache_dir=REPO_ROOT / "data" / "mvp_demo" / "test_cache",
    )
    pipe = build_pipeline(cfg=cfg, connectivity=ConnectivityMock(online=True),
                          enricher=make_enricher(args.enricher))
    print(f"OCR 引擎={pipe.ocr.name} | enricher={pipe.ingest.transport.enricher.name} | db={args.db}\n")

    rows = []
    for i, img in enumerate(images, 1):
        short = f"{img.parent.name}/{img.name}"
        rec = {"n": i, "img": short, "ocr": "", "tags": None, "status": "?", "err": ""}
        try:
            card = pipe.capture(img, trigger_confidence=0.9)  # mock 触发→帧→OCR→enrich→存
            rec["ocr"] = (card.ocr_text or "").replace("\n", " ").strip()
            rec["tags"] = card.tags
            rec["status"] = card.status  # done=enrich成功 / pending=DeepSeek瞬时失败入队
        except EnricherConfigError as e:
            rec["status"] = "config-error"
            rec["err"] = str(e)[:80]
        except Exception as e:  # 抓帧/OCR 等异常，如实记录
            rec["status"] = "error"
            rec["err"] = f"{type(e).__name__}: {e}"[:80]
        rows.append(rec)
        print(f"[{i}/{len(images)}] {short} -> status={rec['status']}, "
              f"ocr_chars={len(rec['ocr'])}, tags={rec['tags']}")

    # ---- 汇总表 ----
    print("\n" + "=" * 100)
    print("汇总表")
    print("=" * 100)
    hdr = f'{"#":<3}{"图片(类别/名)":<48}{"OCR前~15词":<44}{"tags":<44}{"成功?"}'
    print(hdr)
    print("-" * 100)
    for r in rows:
        ok = "✓" if r["status"] == "done" else ("~pending" if r["status"] == "pending" else "✗")
        tags = ",".join(r["tags"]) if r["tags"] else ("[]" if r["tags"] == [] else "-")
        print(f'{r["n"]:<3}{r["img"][:46]:<48}{first_words(r["ocr"])[:42]:<44}{tags[:42]:<44}{ok}')
        if r["err"]:
            print(f'     ! {r["err"]}')

    # ---- 统计 ----
    total = len(rows)
    done = sum(1 for r in rows if r["status"] == "done")
    pending = sum(1 for r in rows if r["status"] == "pending")
    errored = sum(1 for r in rows if r["status"] in ("error", "config-error"))
    ocr_valid = sum(1 for r in rows if ocr_looks_valid(r["ocr"]))
    ocr_thin = total - ocr_valid
    tags_nonempty = sum(1 for r in rows if r["tags"])
    tags_empty = sum(1 for r in rows if r["tags"] == [])

    print("\n" + "=" * 100)
    print("统计")
    print("=" * 100)
    print(f"总数: {total}")
    print(f"端到端存卡成功(status=done): {done}/{total}")
    print(f"  其中 DeepSeek 返回非空 tags: {tags_nonempty}；返回 []（乱码/无意义）: {tags_empty}")
    print(f"DeepSeek 瞬时失败入 pending: {pending}")
    print(f"OCR/抓帧等报错: {errored}")
    print(f"OCR 出了有效文本(启发式≥5词): {ocr_valid}/{total}；基本空/乱码: {ocr_thin}/{total}")
    ds_calls = done + pending  # 真正尝试调用 DeepSeek 的次数
    if ds_calls:
        print(f"DeepSeek 调用成功率: {done}/{ds_calls} = {done/ds_calls*100:.0f}%")

    pipe.close()


if __name__ == "__main__":
    main()
