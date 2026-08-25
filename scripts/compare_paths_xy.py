#!/usr/bin/env python3
"""Path X against path Y, measured for real: run the same images through both pipelines and
store the results in a separate test database.

    Path X (the existing pipeline): image -> real Tesseract OCR (enhance=False) -> real DeepSeek
                                    tagging -> memory card
    Path Y (the proposed change):   image -> resize to 1024 or less -> base64 -> real Claude
                                    multimodal -> complete card

Purpose: material for the architecture decision, given that what needs recording in future is
not only text, so the question is whether to send the image directly.
This script does not change the pipeline's default behaviour. It runs both paths side by side
and records what happened.

Usage:
  .venv/bin/python scripts/compare_paths_xy.py                  # run all 10 images
  .venv/bin/python scripts/compare_paths_xy.py --limit 2        # small trial run first, to limit cost
  .venv/bin/python scripts/compare_paths_xy.py --only-y         # re-run path Y only

Safety and cost:
- A separate test database, data/mvp_demo/comparison_xy.db. The production memory database is
  never written to.
- Hard call ceiling of image count x 2 (one DeepSeek call for path X, one Claude call for path
  Y), printed before the run and checked afterwards.
- Source images are read-only: grab_frame copies them and VisionEnricher only imreads, so
  demo/images/ is never modified.
- Both paths are wrapped in try/except per image, so one failure does not stop the batch and
  the error is recorded as it happened.
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

# Comparison table: one row per image, covering both paths. It lives in the same database file
# as CardStore's memory_cards. Path X still runs the full pipeline and writes memory_cards; this
# table is a summary view aligned by image name, for the HTML report.
COMPARISON_SCHEMA = """
CREATE TABLE IF NOT EXISTS comparison (
    image_name       TEXT PRIMARY KEY,
    -- path X (Tesseract OCR plus DeepSeek)
    x_card_id        INTEGER,          -- memory_cards.id, the card the real path-X pipeline produced
    x_ocr_text       TEXT,
    x_tags           TEXT,             -- JSON array; NULL means not generated
    x_status         TEXT,             -- done / pending / error / config-error
    x_error          TEXT,
    x_seconds        REAL,             -- end-to-end wall clock on this machine and network, not the Pi
    -- path Y (Claude multimodal)
    y_description    TEXT,
    y_tags           TEXT,             -- JSON array
    y_extracted_text TEXT,
    y_parse_ok       INTEGER,
    y_refusal        INTEGER,
    y_status         TEXT,             -- done / error / config-error
    y_error          TEXT,
    y_raw_output     TEXT,             -- kept for diagnosis when parsing fails
    y_in_tokens      INTEGER,
    y_out_tokens     INTEGER,
    y_cost_usd       REAL,
    y_image_bytes    INTEGER,          -- bytes actually sent, after resize and JPEG encoding
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
    """Upsert the given columns by image_name, touching only the columns passed in and leaving
    the other path's columns alone."""
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


# ---------- path X ----------

def run_path_x(images: list[Path], conn: sqlite3.Connection) -> int:
    """The existing pipeline: Tesseract OCR with enhance=False, DeepSeek tagging, stored to
    memory_cards. Returns the number of calls made."""
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
    print(f"path X: OCR={pipe.ocr.name}(enhance={getattr(pipe.ocr, 'enhance', 'n/a')}) "
          f"| enricher={pipe.ingest.transport.enricher.name}")
    calls = 0
    for i, img in enumerate(images, 1):
        t0 = time.monotonic()
        rec: dict = {"x_status": "?", "x_error": None}
        try:
            card = pipe.capture(img, trigger_confidence=0.9)
            calls += 1  # one DeepSeek call was actually attempted
            rec.update(
                x_card_id=card.id,
                x_ocr_text=card.ocr_text or "",
                x_tags=json.dumps(card.tags, ensure_ascii=False) if card.tags is not None else None,
                x_status=card.status,
            )
        except EnricherConfigError as e:
            calls += 1
            rec.update(x_status="config-error", x_error=str(e)[:300])
        except Exception as e:  # frame grab, OCR and similar failures, recorded as they are
            rec.update(x_status="error", x_error=f"{type(e).__name__}: {e}"[:300])
        rec["x_seconds"] = round(time.monotonic() - t0, 3)
        upsert(conn, img.name, **rec)
        ntags = len(json.loads(rec["x_tags"])) if rec.get("x_tags") else 0
        print(f"  [X {i}/{len(images)}] {img.name[:44]:<46} status={rec['x_status']:<8} "
              f"ocr={len(rec.get('x_ocr_text') or '')} chars tags={ntags} ({rec['x_seconds']}s)")
        if rec["x_error"]:
            print(f"      ! {rec['x_error'][:120]}")
    pipe.close()
    return calls


# ---------- path Y ----------

def run_path_y(images: list[Path], conn: sqlite3.Connection) -> int:
    """The proposed change: image to VisionEnricher (Claude multimodal) to a complete card.
    Returns the number of calls made."""
    enr = VisionEnricher()
    print(f"path Y: enricher={enr.name} | model={enr.model} | max_side={enr.max_side} "
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


# ---------- summary ----------

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
    print("Comparison statistics")
    print("=" * 96)
    print(f"images: {total}")
    print(f"path X  end-to-end done: {x_done}/{total} | non-empty tags: {x_tagged}/{total}")
    print(f"path Y  calls succeeded: {y_done}/{total} | usable content: {y_content}/{total} "
          f"| non-empty tags: {y_tagged}/{total}")
    print(f"images where X produced no tags: {total - x_tagged}; Y rescued {rescued} of them")
    print(f"neither path produced anything: {both_fail}")
    print("-" * 96)
    print(f"path Y cost from measured tokens: {in_tok:,} in + {out_tok:,} out "
          f"= ${cost:.4f} (at ${PRICE_IN_PER_MTOK}/${PRICE_OUT_PER_MTOK} per million)")
    if total:
        print(f"  per image: ${cost/total:.5f}  |  extrapolated to 1000: ${cost/total*1000:.2f}")
    print(f"bytes uploaded: path Y images {img_bytes/1024:.0f} KB against path X text "
          f"{x_text_bytes/1024:.1f} KB "
          f"({img_bytes/max(x_text_bytes,1):.0f}x)")
    print("=" * 96)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(IMG_DIR))
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--limit", type=int, default=None, help="run only the first N images, as a cheap trial")
    ap.add_argument("--start", type=int, default=0,
                    help="start from image N. With --limit this runs in batches, avoiding paying twice for "
                         "images already done")
    ap.add_argument("--only-x", action="store_true", help="run path X only")
    ap.add_argument("--only-y", action="store_true", help="run path Y only")
    args = ap.parse_args()

    all_images = list_images(Path(args.dir))
    images = all_images[args.start:]
    if args.limit:
        images = images[: args.limit]
    if not images:
        sys.exit(f"no images found: {args.dir}")

    do_x = not args.only_y
    do_y = not args.only_x
    budget = len(images) * (int(do_x) + int(do_y))
    print(f"{len(images)} images from {args.dir} ({len(all_images)} total, starting at {args.start})")
    print(f"call budget ceiling: {len(images)} x {int(do_x) + int(do_y)} paths = {budget}\n")

    conn = open_db(Path(args.db))
    calls = 0
    if do_x:
        calls += run_path_x(images, conn)
        print()
    if do_y:
        calls += run_path_y(images, conn)

    summarize(conn)
    print(f"\ncalls actually made: {calls} (budget {budget})"
          f"{' - within budget' if calls <= budget else ' - OVER BUDGET'}")
    print(f"results database: {args.db}")
    conn.close()


if __name__ == "__main__":
    main()
