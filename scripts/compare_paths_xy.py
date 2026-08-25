#!/usr/bin/env python3
"""路径X vs 路径Y 真实对比：同一批图，两条管线各跑一遍，结果存独立测试库。

    路径X（现有管线）：图 → 真 Tesseract OCR（enhance=False）→ 真 DeepSeek 打标 → memory card
    路径Y（新方案）  ：图 → resize≤1024 → base64 → 真 Claude 多模态 → 完整 card
                       {description, tags, extracted_text}

用途：给导师的架构决策材料（"未来要记的不只文字，是否该改成直接传图？"）。
**本脚本不改管线默认行为**，只是把两条路径并排跑一遍、如实记录。

用法：
  .venv/bin/python scripts/compare_paths_xy.py                  # 跑全部 10 张
  .venv/bin/python scripts/compare_paths_xy.py --limit 2        # 先小规模试跑（省钱）
  .venv/bin/python scripts/compare_paths_xy.py --only-y         # 只补跑路径Y

安全 / 成本：
- 独立测试库 data/mvp_demo/comparison_xy.db，**不写正式 memory 库**。
- 调用量硬上限 = 图数 × 2（路径X 一次 DeepSeek + 路径Y 一次 Claude），跑前打印、跑后核对。
- 源图只读（grab_frame 拷贝副本；VisionEnricher 只 imread），demo/images/ 绝不改动。
- 每张图两路都有 try/except，单图失败不中断整批，错误如实记进库。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pipeline import config as config_mod                      # noqa: E402
from pipeline.connectivity import ConnectivityMock             # noqa: E402
from pipeline.db import CardStore                              # noqa: E402
from pipeline.enrich import (                                   # noqa: E402
    DeepSeekEnricher, EnricherConfigError, EnricherError, VisionEnricher,
)
from pipeline.enrich.vision_enricher import (                   # noqa: E402
    PRICE_IN_PER_MTOK, PRICE_OUT_PER_MTOK,
)
from pipeline.models import utc_now_iso                        # noqa: E402
from pipeline.pipeline import build_pipeline                   # noqa: E402

IMG_DIR = REPO_ROOT / "demo" / "images"
DB_PATH = REPO_ROOT / "data" / "mvp_demo" / "comparison_xy.db"
IMG_EXTS = (".jpg", ".jpeg", ".png")

# 对比结果表：一行 = 一张图 × 两条路径。与 CardStore 的 memory_cards 共存于同一 db 文件
# （路径X 仍然真的走完整管线写 memory_cards；本表是给 HTML 用的、按图名对齐的汇总视图）。
COMPARISON_SCHEMA = """
CREATE TABLE IF NOT EXISTS comparison (
    image_name       TEXT PRIMARY KEY,
    -- 路径X（Tesseract OCR + DeepSeek）
    x_card_id        INTEGER,          -- memory_cards.id（路径X 真管线产出的卡）
    x_ocr_text       TEXT,
    x_tags           TEXT,             -- JSON 数组；NULL = 未生成
    x_status         TEXT,             -- done / pending / error / config-error
    x_error          TEXT,
    x_seconds        REAL,             -- 端到端墙钟耗时（本机 + 当前网络，非 Pi 实测）
    -- 路径Y（Claude 多模态）
    y_description    TEXT,
    y_tags           TEXT,             -- JSON 数组
    y_extracted_text TEXT,
    y_parse_ok       INTEGER,
    y_refusal        INTEGER,
    y_status         TEXT,             -- done / error / config-error
    y_error          TEXT,
    y_raw_output     TEXT,             -- 解析失败时留档诊断
    y_in_tokens      INTEGER,
    y_out_tokens     INTEGER,
    y_cost_usd       REAL,
    y_image_bytes    INTEGER,          -- resize+JPEG 后真正发出去的字节数
    y_seconds        REAL,
    created_at       TEXT
);
"""


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(COMPARISON_SCHEMA)
    conn.commit()
    return conn


def upsert(conn: sqlite3.Connection, image_name: str, **fields) -> None:
    """按 image_name upsert 指定列（只更新传入的列，另一条路径的列不动）。"""
    conn.execute(
        "INSERT OR IGNORE INTO comparison (image_name, created_at) VALUES (?, ?)",
        (image_name, utc_now_iso()),
    )
    if fields:
        sets = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(
            f"UPDATE comparison SET {sets} WHERE image_name = ?",
            (*fields.values(), image_name),
        )
    conn.commit()


def list_images(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS),
        key=lambda p: p.name,
    )


# ---------- 路径X ----------

def run_path_x(images: list[Path], conn: sqlite3.Connection) -> int:
    """现有管线：Tesseract OCR(enhance=False) → DeepSeek 打标 → 存 memory_cards。返回调用次数。"""
    cfg = config_mod.Config(
        db_path=DB_PATH,
        frames_dir=REPO_ROOT / "data" / "mvp_demo" / "cmp_frames",
        cache_dir=REPO_ROOT / "data" / "mvp_demo" / "cmp_cache",
    )
    pipe = build_pipeline(
        cfg=cfg,
        connectivity=ConnectivityMock(online=True),
        enricher=DeepSeekEnricher(),
    )
    print(f"路径X: OCR={pipe.ocr.name}(enhance={getattr(pipe.ocr, 'enhance', 'n/a')}) "
          f"| enricher={pipe.ingest.transport.enricher.name}")
    calls = 0
    for i, img in enumerate(images, 1):
        t0 = time.monotonic()
        rec: dict = {"x_status": "?", "x_error": None}
        try:
            card = pipe.capture(img, trigger_confidence=0.9)
            calls += 1  # 真正尝试了一次 DeepSeek 调用
            rec.update(
                x_card_id=card.id,
                x_ocr_text=card.ocr_text or "",
                x_tags=json.dumps(card.tags, ensure_ascii=False) if card.tags is not None else None,
                x_status=card.status,
            )
        except EnricherConfigError as e:
            calls += 1
            rec.update(x_status="config-error", x_error=str(e)[:300])
        except Exception as e:  # 抓帧 / OCR 等异常，如实记录
            rec.update(x_status="error", x_error=f"{type(e).__name__}: {e}"[:300])
        rec["x_seconds"] = round(time.monotonic() - t0, 3)
        upsert(conn, img.name, **rec)
        ntags = len(json.loads(rec["x_tags"])) if rec.get("x_tags") else 0
        print(f"  [X {i}/{len(images)}] {img.name[:44]:<46} status={rec['x_status']:<8} "
              f"ocr={len(rec.get('x_ocr_text') or '')}字符 tags={ntags} ({rec['x_seconds']}s)")
        if rec["x_error"]:
            print(f"      ! {rec['x_error'][:120]}")
    pipe.close()
    return calls


# ---------- 路径Y ----------

def run_path_y(images: list[Path], conn: sqlite3.Connection) -> int:
    """新方案：图 → VisionEnricher（Claude 多模态）→ 完整 card。返回调用次数。"""
    enr = VisionEnricher()
    print(f"路径Y: enricher={enr.name} | model={enr.model} | max_side={enr.max_side} "
          f"| max_tokens={enr.max_tokens}")
    calls = 0
    for i, img in enumerate(images, 1):
        t0 = time.monotonic()
        rec: dict = {"y_status": "?", "y_error": None}
        try:
            card = enr.enrich_image(img, {"timestamp": utc_now_iso(), "trigger_confidence": 0.9})
            calls += 1
            rec.update(
                y_description=card.description,
                y_tags=json.dumps(card.tags, ensure_ascii=False),
                y_extracted_text=card.extracted_text,
                y_parse_ok=int(card.parse_ok),
                y_refusal=int(card.refusal),
                y_status="done",
                y_raw_output=(card.raw_output if not card.parse_ok else None),
                y_in_tokens=card.usage.get("input_tokens", 0),
                y_out_tokens=card.usage.get("output_tokens", 0),
                y_cost_usd=card.cost_usd(),
                y_image_bytes=card.image_bytes_sent,
            )
        except EnricherConfigError as e:
            calls += 1
            rec.update(y_status="config-error", y_error=str(e)[:300])
        except EnricherError as e:
            calls += 1
            rec.update(y_status="error", y_error=str(e)[:300])
        except Exception as e:
            rec.update(y_status="error", y_error=f"{type(e).__name__}: {e}"[:300])
        rec["y_seconds"] = round(time.monotonic() - t0, 3)
        upsert(conn, img.name, **rec)
        ntags = len(json.loads(rec["y_tags"])) if rec.get("y_tags") else 0
        desc = (rec.get("y_description") or "")[:52]
        print(f"  [Y {i}/{len(images)}] {img.name[:44]:<46} status={rec['y_status']:<8} "
              f"tags={ntags} ({rec['y_seconds']}s)")
        if desc:
            print(f"      → {desc}…")
        if rec["y_error"]:
            print(f"      ! {rec['y_error'][:120]}")
    return calls


# ---------- 汇总 ----------

def summarize(conn: sqlite3.Connection) -> None:
    rows = list(conn.execute("SELECT * FROM comparison ORDER BY image_name"))
    total = len(rows)

    def x_tags(r):
        return json.loads(r["x_tags"]) if r["x_tags"] else []

    def y_tags(r):
        return json.loads(r["y_tags"]) if r["y_tags"] else []

    x_done = sum(1 for r in rows if r["x_status"] == "done")
    x_tagged = sum(1 for r in rows if x_tags(r))
    y_done = sum(1 for r in rows if r["y_status"] == "done")
    y_content = sum(1 for r in rows if (r["y_description"] or "").strip() or y_tags(r))
    y_tagged = sum(1 for r in rows if y_tags(r))
    rescued = sum(1 for r in rows if not x_tags(r) and y_tags(r))
    both_fail = sum(1 for r in rows if not x_tags(r) and not y_tags(r))

    in_tok = sum(r["y_in_tokens"] or 0 for r in rows)
    out_tok = sum(r["y_out_tokens"] or 0 for r in rows)
    cost = sum(r["y_cost_usd"] or 0.0 for r in rows)
    img_bytes = sum(r["y_image_bytes"] or 0 for r in rows)
    x_text_bytes = sum(len((r["x_ocr_text"] or "").encode("utf-8")) for r in rows)

    print("\n" + "=" * 96)
    print("对比统计")
    print("=" * 96)
    print(f"图片总数: {total}")
    print(f"路径X  端到端 done: {x_done}/{total} | 产出非空标签: {x_tagged}/{total}")
    print(f"路径Y  调用成功  : {y_done}/{total} | 产出有效内容: {y_content}/{total} "
          f"| 非空标签: {y_tagged}/{total}")
    print(f"Y 在「X 无标签」的 {total - x_tagged} 张图上救回: {rescued}")
    print(f"两路都没产出: {both_fail}")
    print("-" * 96)
    print(f"路径Y 成本（实测 token）: 输入 {in_tok:,} tok + 输出 {out_tok:,} tok "
          f"= ${cost:.4f}（单价 ${PRICE_IN_PER_MTOK}/${PRICE_OUT_PER_MTOK} 每百万）")
    if total:
        print(f"  单张均摊: ${cost/total:.5f}  |  推算 1000 张: ${cost/total*1000:.2f}")
    print(f"传输字节: 路径Y 图 {img_bytes/1024:.0f} KB  vs  路径X 文本 {x_text_bytes/1024:.1f} KB "
          f"（{img_bytes/max(x_text_bytes,1):.0f}×）")
    print("=" * 96)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(IMG_DIR))
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 张（小规模试跑省钱）")
    ap.add_argument("--start", type=int, default=0,
                    help="从第 N 张开始（配合 --limit 分批跑，避免重跑已完成的图重复花钱）")
    ap.add_argument("--only-x", action="store_true", help="只跑路径X")
    ap.add_argument("--only-y", action="store_true", help="只跑路径Y")
    args = ap.parse_args()

    all_images = list_images(Path(args.dir))
    images = all_images[args.start:]
    if args.limit:
        images = images[: args.limit]
    if not images:
        sys.exit(f"没有找到图片: {args.dir}")

    do_x = not args.only_y
    do_y = not args.only_x
    budget = len(images) * (int(do_x) + int(do_y))
    print(f"图片 {len(images)} 张（{args.dir}，共 {len(all_images)} 张，从第 {args.start} 张起）")
    print(f"调用预算上限: {len(images)} × {int(do_x) + int(do_y)} 路 = {budget} 次\n")

    conn = open_db(Path(args.db))
    calls = 0
    if do_x:
        calls += run_path_x(images, conn)
        print()
    if do_y:
        calls += run_path_y(images, conn)

    summarize(conn)
    print(f"\n实际调用次数: {calls}（预算 {budget}）"
          f"{' ✓ 未超' if calls <= budget else ' ⚠ 超预算！'}")
    print(f"结果库: {args.db}")
    conn.close()


if __name__ == "__main__":
    main()
