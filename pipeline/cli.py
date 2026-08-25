"""Command-line recall and demo CLI, which is how the recall loop is verified.

Subcommands:
  list                       list every memory card
  show <id>                  show one card in detail
  search <keyword>           search ocr_text and tags by keyword
  pending                    show the pending queue
  ingest <image> [...]       run one capture (mocked trigger, frame, OCR, enrich or queue, store)
  process-pending            backfill the pending queue once connectivity returns
  demo                       built-in end-to-end demo, synthesising a text image and running the
                             whole loop

Database path: data/mvp_demo/memosight.db by default, overridable with the MEMOSIGHT_DB
environment variable.
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
    """Defaults to DeepSeekEnricher. --enricher selects deepseek, claude or mock, and
    --mock-enrich is shorthand for mock."""
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
            print("(empty) no memory cards yet. Create one with `ingest` or `demo`.")
            return
        for c in cards:
            print(_fmt_card_line(c))
        print(f"\n{len(cards)} cards (done={store.count('done')} pending={store.count('pending')})")


def cmd_show(args, cfg):
    with _store(cfg) as store:
        c = store.get(args.id)
        if not c:
            print(f"#{args.id} not found")
            sys.exit(1)
        _print_card_detail(c)


def cmd_search(args, cfg):
    with _store(cfg) as store:
        cards = store.search(args.keyword, limit=args.limit)
        if not cards:
            print(f"no cards match '{args.keyword}'.")
            return
        for c in cards:
            print(_fmt_card_line(c))
        print(f"\n{len(cards)} matches.")


def cmd_pending(args, cfg):
    with _store(cfg) as store:
        cards = store.list_pending()
        if not cards:
            print("the pending queue is empty.")
            return
        for c in cards:
            print(_fmt_card_line(c))
        print(f"\n{len(cards)} cards pending.")


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
            print("the gatekeeper did not fire (confidence below the threshold); nothing recorded.")
            return
        state = "offline, queued as pending" if args.offline else "online, stored as done"
        print(f"captured: {state}  (OCR={pipe.ocr.name}, enricher={pipe.ingest.transport.enricher.name})")
        _print_card_detail(card)
    except EnricherConfigError as e:
        print(f"[enricher configuration error] {e}\nHint: --mock-enrich runs offline and returns mock tags.")
        sys.exit(2)
    finally:
        pipe.close()


def cmd_process_pending(args, cfg):
    pipe = build_pipeline(
        cfg=cfg, connectivity=ConnectivityMock(online=True), enricher=_make_enricher(args)
    )
    try:
        done = pipe.process_pending()
        print(f"backfilled {len(done)} cards.")
        for c in done:
            print(_fmt_card_line(c))
    except EnricherConfigError as e:
        print(f"[enricher configuration error] {e}\nHint: --mock-enrich runs offline.")
        sys.exit(2)
    finally:
        pipe.close()


def cmd_demo(args, cfg):
    """Synthesise a text image, run the whole loop, then search the result back out to show
    recall works."""
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
        print(f"[1] mocked trigger -> grab {demo_img.name} -> OCR({pipe.ocr.name}) -> enrich({enr}) -> store")
        card = pipe.capture(demo_img, trigger_confidence=0.93)
        _print_card_detail(card)
        kw = "MEMOSIGHT" if pipe.ocr.name == "tesseract" else "stub"
        print(f"\n[2] searching for '{kw}':")
        for c in pipe.store.search(kw):
            print("   " + _fmt_card_line(c))
    except EnricherConfigError as e:
        print(f"[enricher configuration error] {e}\nHint: `demo --mock-enrich` runs offline.")
        sys.exit(2)
    finally:
        pipe.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="memosight", description="MemoSight MVP recall and demo CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list every card")
    pl.add_argument("--limit", type=int, default=None)
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("show", help="show one card in detail")
    ps.add_argument("id", type=int)
    ps.set_defaults(func=cmd_show)

    pf = sub.add_parser("search", help="search ocr_text and tags by keyword")
    pf.add_argument("keyword")
    pf.add_argument("--limit", type=int, default=None)
    pf.set_defaults(func=cmd_search)

    pp = sub.add_parser("pending", help="show the pending queue")
    pp.set_defaults(func=cmd_pending)

    pi = sub.add_parser("ingest", help="run one capture: mocked trigger, frame, OCR, enrich, store")
    pi.add_argument("image", help="path to a test image, standing in for a full-resolution grab")
    pi.add_argument("--confidence", type=float, default=0.9)
    pi.add_argument("--offline", action="store_true", help="simulate being offline, so the card is queued as pending")
    pi.add_argument("--policy", default=None,
                    choices=list(config_mod.VALID_RAW_IMAGE_POLICIES),
                    help="raw_image_policy (defaults to the configured value, delete)")
    pi.add_argument("--enricher", choices=ENRICHER_CHOICES, default="deepseek",
                    help="which enricher to use (default deepseek)")
    pi.add_argument("--mock-enrich", action="store_true",
                    help="same as --enricher mock: offline, no tokens spent, returns mock tags")
    pi.set_defaults(func=cmd_ingest)

    pr = sub.add_parser("process-pending", help="backfill the pending queue once connectivity returns")
    pr.add_argument("--enricher", choices=ENRICHER_CHOICES, default="deepseek",
                    help="which enricher to use (default deepseek)")
    pr.add_argument("--mock-enrich", action="store_true", help="= --enricher mock")
    pr.set_defaults(func=cmd_process_pending)

    pd = sub.add_parser("demo", help="built-in end-to-end demo")
    pd.add_argument("--enricher", choices=ENRICHER_CHOICES, default="deepseek",
                    help="which enricher to use (default deepseek)")
    pd.add_argument("--mock-enrich", action="store_true", help="same as --enricher mock: offline")
    pd.set_defaults(func=cmd_demo)
    return p


def main(argv=None) -> None:
    load_env()  # load keys from the project-root .env; harmless when the enricher is mocked
    args = build_parser().parse_args(argv)
    cfg = config_mod.default_config()
    args.func(args, cfg)


if __name__ == "__main__":
    main()
