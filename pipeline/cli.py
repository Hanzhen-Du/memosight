"""命令行查询/演示 CLI —— 验证"可回忆"闭环。

子命令：
  list                       列出全部 memory card
  show <id>                  看单条详情
  search <keyword>           按关键词搜 ocr_text / tags
  pending                    看 pending 待处理队列
  ingest <image> [...]       跑一次捕捉（mock 触发→帧→OCR→enrich/入队→存）
  process-pending            恢复联网后批量补传 pending
  demo                       内置端到端演示（合成一张文字图跑完整闭环）

DB 路径：默认 data/mvp_demo/memosight.db，可用环境变量 MEMOSIGHT_DB 覆盖。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config as config_mod
from .env import load_env
from .connectivity import ConnectivityMock
from .db import CardStore
from .enrich import (
    CloudEnricher,
    DeepSeekEnricher,
    EnricherConfigError,
    MockCloudEnricher,
)
from .models import MemoryCard
from .pipeline import build_pipeline

ENRICHER_CHOICES = ("deepseek", "claude", "mock")


def _make_enricher(args):
    """默认 DeepSeekEnricher；--enricher 选择 deepseek/claude/mock；--mock-enrich 是 mock 的简写。"""
    if getattr(args, "mock_enrich", False):
        return MockCloudEnricher()
    choice = getattr(args, "enricher", "deepseek")
    if choice == "mock":
        return MockCloudEnricher()
    if choice == "claude":
        return CloudEnricher()
    return DeepSeekEnricher()


def _fmt_card_line(c: MemoryCard) -> str:
    tags = ",".join(c.tags) if c.tags else "-"
    text = (c.ocr_text or "").replace("\n", " ")
    if len(text) > 48:
        text = text[:47] + "…"
    return (
        f"#{c.id:<4} [{c.status:<7}] conf={c.trigger_confidence:.2f} "
        f"{c.timestamp}  ocr='{text}'  tags=[{tags}]"
    )


def _print_card_detail(c: MemoryCard) -> None:
    print(f"── memory card #{c.id} " + "─" * 30)
    print(f"  status            : {c.status}")
    print(f"  timestamp         : {c.timestamp}")
    print(f"  trigger_confidence: {c.trigger_confidence}")
    print(f"  raw_image_policy  : {c.raw_image_policy}")
    print(f"  created_at        : {c.created_at}")
    print(f"  enriched_at       : {c.enriched_at or '-'}")
    print(f"  tags              : {c.tags if c.tags else '- (pending)'}")
    print(f"  ocr_text          :\n{c.ocr_text}")


def _store(cfg: config_mod.Config) -> CardStore:
    cfg.ensure_dirs()
    return CardStore(cfg.db_path)


def cmd_list(args, cfg):
    with _store(cfg) as store:
        cards = store.list_all(limit=args.limit)
        if not cards:
            print("(空) 还没有 memory card。用 `ingest` 或 `demo` 造一张。")
            return
        for c in cards:
            print(_fmt_card_line(c))
        print(f"\n共 {len(cards)} 张（done={store.count('done')} pending={store.count('pending')}）")


def cmd_show(args, cfg):
    with _store(cfg) as store:
        c = store.get(args.id)
        if not c:
            print(f"未找到 #{args.id}")
            sys.exit(1)
        _print_card_detail(c)


def cmd_search(args, cfg):
    with _store(cfg) as store:
        cards = store.search(args.keyword, limit=args.limit)
        if not cards:
            print(f"没有匹配 '{args.keyword}' 的卡片。")
            return
        for c in cards:
            print(_fmt_card_line(c))
        print(f"\n匹配 {len(cards)} 张。")


def cmd_pending(args, cfg):
    with _store(cfg) as store:
        cards = store.list_pending()
        if not cards:
            print("pending 队列为空。")
            return
        for c in cards:
            print(_fmt_card_line(c))
        print(f"\npending 共 {len(cards)} 张。")


def cmd_ingest(args, cfg):
    conn = ConnectivityMock(online=not args.offline)
    pipe = build_pipeline(cfg=cfg, connectivity=conn, enricher=_make_enricher(args))
    try:
        card = pipe.capture(
            source_image=Path(args.image),
            trigger_confidence=args.confidence,
            raw_image_policy=args.policy,
        )
        if card is None:
            print("守门员未触发（confidence 低于阈值），未记录。")
            return
        state = "断网入队(pending)" if args.offline else "联网直存(done)"
        print(f"已捕捉 → {state}  (OCR={pipe.ocr.name}, enricher={pipe.ingest.transport.enricher.name})")
        _print_card_detail(card)
    except EnricherConfigError as e:
        print(f"[enricher 配置错误] {e}\n提示：用 --mock-enrich 可离线跑（返回 mock 假标签）。")
        sys.exit(2)
    finally:
        pipe.close()


def cmd_process_pending(args, cfg):
    pipe = build_pipeline(
        cfg=cfg, connectivity=ConnectivityMock(online=True), enricher=_make_enricher(args)
    )
    try:
        done = pipe.process_pending()
        print(f"补传完成 {len(done)} 张。")
        for c in done:
            print(_fmt_card_line(c))
    except EnricherConfigError as e:
        print(f"[enricher 配置错误] {e}\n提示：用 --mock-enrich 可离线跑。")
        sys.exit(2)
    finally:
        pipe.close()


def cmd_demo(args, cfg):
    """合成一张文字图，跑完整闭环，再搜出来，证明可回忆。"""
    import cv2
    import numpy as np

    cfg.ensure_dirs()
    demo_img = Path(cfg.frames_dir).parent / "demo_source.png"
    img = np.full((240, 1000, 3), 255, dtype=np.uint8)
    cv2.putText(img, "MEMOSIGHT DEMO ROADMAP", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.imwrite(str(demo_img), img)

    pipe = build_pipeline(cfg=cfg, connectivity=ConnectivityMock(online=True),
                          enricher=_make_enricher(args))
    try:
        enr = pipe.ingest.transport.enricher.name
        print(f"[1] mock 触发 → 抓帧 {demo_img.name} → OCR({pipe.ocr.name}) → enrich({enr}) → 存库")
        card = pipe.capture(demo_img, trigger_confidence=0.93)
        _print_card_detail(card)
        kw = "MEMOSIGHT" if pipe.ocr.name == "tesseract" else "stub"
        print(f"\n[2] 按关键词 '{kw}' 搜索：")
        for c in pipe.store.search(kw):
            print("   " + _fmt_card_line(c))
    except EnricherConfigError as e:
        print(f"[enricher 配置错误] {e}\n提示：用 `demo --mock-enrich` 可离线跑。")
        sys.exit(2)
    finally:
        pipe.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="memosight", description="MemoSight MVP 查询/演示 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="列出全部卡片")
    pl.add_argument("--limit", type=int, default=None)
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("show", help="看单条详情")
    ps.add_argument("id", type=int)
    ps.set_defaults(func=cmd_show)

    pf = sub.add_parser("search", help="按关键词搜 ocr_text/tags")
    pf.add_argument("keyword")
    pf.add_argument("--limit", type=int, default=None)
    pf.set_defaults(func=cmd_search)

    pp = sub.add_parser("pending", help="看 pending 队列")
    pp.set_defaults(func=cmd_pending)

    pi = sub.add_parser("ingest", help="跑一次捕捉（mock 触发→帧→OCR→enrich→存）")
    pi.add_argument("image", help="测试图片路径（替代高清抓帧）")
    pi.add_argument("--confidence", type=float, default=0.9)
    pi.add_argument("--offline", action="store_true", help="模拟断网 → 入 pending 队列")
    pi.add_argument("--policy", default=None,
                    choices=list(config_mod.VALID_RAW_IMAGE_POLICIES),
                    help="raw_image_policy（默认取配置 delete）")
    pi.add_argument("--enricher", choices=ENRICHER_CHOICES, default="deepseek",
                    help="选 enricher（默认 deepseek）")
    pi.add_argument("--mock-enrich", action="store_true",
                    help="= --enricher mock（离线/省 token，返回 mock 假标签）")
    pi.set_defaults(func=cmd_ingest)

    pr = sub.add_parser("process-pending", help="恢复联网后批量补传 pending")
    pr.add_argument("--enricher", choices=ENRICHER_CHOICES, default="deepseek",
                    help="选 enricher（默认 deepseek）")
    pr.add_argument("--mock-enrich", action="store_true", help="= --enricher mock")
    pr.set_defaults(func=cmd_process_pending)

    pd = sub.add_parser("demo", help="内置端到端演示")
    pd.add_argument("--enricher", choices=ENRICHER_CHOICES, default="deepseek",
                    help="选 enricher（默认 deepseek）")
    pd.add_argument("--mock-enrich", action="store_true", help="= --enricher mock（离线）")
    pd.set_defaults(func=cmd_demo)
    return p


def main(argv=None) -> None:
    load_env()  # 从项目根 .env 加载密钥（阶段二真 Claude API 用；本阶段 mock 无害）
    args = build_parser().parse_args(argv)
    cfg = config_mod.default_config()
    args.func(args, cfg)


if __name__ == "__main__":
    main()
