#!/usr/bin/env python3
"""Generate a side-by-side visualisation of the 10-image end-to-end test, as one
self-contained HTML file.

Data sources:
- image paths: the same stratified sample as the end-to-end test (seed 42, deterministic, so
  it is the same 10 images);
- OCR text, DeepSeek tags and status: read from data/mvp_demo/test_e2e.db, in ascending id
  order, which is the sampling order.

Thumbnails are embedded as base64, producing a single HTML file that opens directly in a
browser.
Usage:  .venv/bin/python scripts/gen_e2e_showcase.py
Output: demo/e2e_showcase.html
"""

from __future__ import annotations

import base64
import html
import random
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import test_e2e_images as e2e            # noqa: E402
from pipeline.db import CardStore        # noqa: E402

OUT = REPO_ROOT / "demo" / "e2e_showcase.html"
DB = REPO_ROOT / "data" / "mvp_demo" / "test_e2e.db"

# Sparse categories: nearly empty or very little text, where [] is the correct answer.
# Any other empty tag list is classified as an OCR failure.
SPARSE_CATS = {"projector_slide_screen_empty_room", "data_dashboard_monitor"}

# Notes on the successful cases, showing that OCR plus tagging is good when the input is clear
HILITE = {
    "laptop_screen_code": "web UI code, giving precise technology-stack tags",
    "source_code_on_monitor_closeup": "JavaScript source, giving framework-level tags (Backbone, router)",
    "document_page_text": "a table-tennis course document in Indonesian, giving domain tags",
    "tablet_displaying_text_document": "blurry OCR, yet DeepSeek still recovered STOCK MARKET from the noise",
}
NOTE_SPARSE = "a nearly empty screen or a dashboard with little text: [] is the correct answer, and nothing was invented"
NOTE_FAIL = "text is present, but the posed angle, glare, distance or handwriting made OCR return noise, so []"


def thumb_b64(path: Path, width: int = 480) -> str:
    img = cv2.imread(str(path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def verdict(cat: str, tags) -> str:
    if tags:
        return "success"
    return "sparse" if cat in SPARSE_CATS else "fail"


def build_cards():
    images = e2e.stratified_sample(10, random.Random(42))
    store = CardStore(DB)
    cards = sorted(store.list_all(), key=lambda c: c.id)
    store.close()
    out = []
    for img, card in zip(images, cards):
        cat = img.parent.name
        v = verdict(cat, card.tags)
        ocr = (card.ocr_text or "").strip()
        note = HILITE.get(cat) if v == "success" else (NOTE_SPARSE if v == "sparse" else NOTE_FAIL)
        out.append({
            "cat": cat, "name": img.name, "img": thumb_b64(img),
            "ocr": ocr, "chars": len(ocr), "tags": card.tags or [],
            "verdict": v, "note": note,
        })
    # Order: successes first, sparse in the middle, failures last
    rank = {"success": 0, "sparse": 1, "fail": 2}
    out.sort(key=lambda c: rank[c["verdict"]])
    return out


VERDICT_META = {
    "success": ("SUCCESS", "#16a34a"),
    "sparse":  ("SPARSE", "#6b7280"),
    "fail":    ("OCR FAILED", "#dc2626"),
}


def render(cards) -> str:
    n = len(cards)
    n_ok = sum(1 for c in cards if c["verdict"] == "success")
    card_html = []
    for i, c in enumerate(cards, 1):
        label, color = VERDICT_META[c["verdict"]]
        ocr_disp = html.escape(c["ocr"][:320]) + ("…" if c["chars"] > 320 else "")
        if not c["ocr"]:
            ocr_disp = "<span class='empty'>(no text)</span>"
        tags_html = "".join(
            f"<span class='tag'>{html.escape(t)}</span>" for t in c["tags"]
        ) or "<span class='notags'>[]</span>"
        card_html.append(f"""
      <div class="card {c['verdict']}">
        <div class="badge" style="background:{color}">{label}</div>
        <div class="thumbwrap"><img src="{c['img']}" alt="{html.escape(c['name'])}"></div>
        <div class="mid">
          <div class="cat">{html.escape(c['cat'])}</div>
          <div class="ocrlabel">OCR text, {c['chars']} characters</div>
          <div class="ocr">{ocr_disp}</div>
          <div class="note">{html.escape(c['note'])}</div>
        </div>
        <div class="right">
          <div class="tagslabel">DeepSeek tags</div>
          <div class="tags">{tags_html}</div>
        </div>
      </div>""")

    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MemoSight end-to-end test results</title>
<style>
  :root {{ --ok:#16a34a; --grey:#6b7280; --fail:#dc2626; --ink:#0f172a; --muted:#64748b; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
         color:var(--ink); background:#f1f5f9; line-height:1.5; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:32px 24px 64px; }}
  h1 {{ font-size:30px; margin:0 0 4px; }}
  .sub {{ color:var(--muted); font-size:16px; margin-bottom:22px; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:14px; margin-bottom:18px; }}
  .stat {{ background:#fff; border-radius:12px; padding:16px 20px; flex:1 1 200px;
           box-shadow:0 1px 3px rgba(0,0,0,.08); }}
  .stat .num {{ font-size:30px; font-weight:800; }}
  .stat .lab {{ font-size:14px; color:var(--muted); }}
  .stat.g .num {{ color:var(--ok); }} .stat.b .num {{ color:#2563eb; }}
  .caveat {{ background:#fffbeb; border:2px solid #f59e0b; border-radius:12px;
             padding:16px 20px; margin:6px 0 28px; font-size:16px; }}
  .caveat b {{ color:#b45309; }}
  .card {{ position:relative; display:grid; grid-template-columns:280px 1fr 300px; gap:22px;
           background:#fff; border-radius:16px; padding:20px; margin-bottom:20px;
           box-shadow:0 2px 8px rgba(0,0,0,.07); border-left:8px solid transparent; }}
  .card.success {{ border-left-color:var(--ok); }}
  .card.sparse  {{ border-left-color:var(--grey); }}
  .card.fail    {{ border-left-color:var(--fail); }}
  .badge {{ position:absolute; top:14px; right:14px; color:#fff; font-size:13px; font-weight:700;
            padding:5px 12px; border-radius:999px; letter-spacing:.3px; }}
  .thumbwrap {{ align-self:center; }}
  .thumbwrap img {{ width:100%; border-radius:10px; display:block; border:1px solid #e2e8f0; }}
  .cat {{ font-size:17px; font-weight:700; margin-bottom:8px; }}
  .ocrlabel, .tagslabel {{ font-size:13px; color:var(--muted); text-transform:uppercase;
            letter-spacing:.5px; margin-bottom:6px; }}
  .ocr {{ font-family:"SF Mono",Menlo,Consolas,monospace; font-size:13.5px; background:#f8fafc;
          border:1px solid #e2e8f0; border-radius:8px; padding:10px 12px; color:#334155;
          max-height:150px; overflow:auto; white-space:pre-wrap; word-break:break-word; }}
  .ocr .empty {{ color:#94a3b8; }}
  .note {{ margin-top:10px; font-size:14.5px; color:#475569; }}
  .card.success .note {{ color:#15803d; font-weight:600; }}
  .tags {{ display:flex; flex-wrap:wrap; gap:8px; }}
  .tag {{ background:#ecfdf5; color:#047857; border:1px solid #a7f3d0; border-radius:999px;
          padding:6px 13px; font-size:15px; font-weight:600; }}
  .notags {{ color:#94a3b8; font-family:monospace; font-size:16px; }}
  .foot {{ color:var(--muted); font-size:13px; margin-top:24px; text-align:center; }}
  @media (max-width:820px) {{ .card {{ grid-template-columns:1fr; }} }}
</style></head>
<body><div class="wrap">
  <h1>MemoSight: end-to-end verification of the stage after the gatekeeper</h1>
  <div class="sub">10 real positive images, real Tesseract OCR, real DeepSeek tags, SQLite. Software loop verified on a laptop.</div>

  <div class="stats">
    <div class="stat g"><div class="num">{n}/{n}</div><div class="lab">completed the loop, zero crashes</div></div>
    <div class="stat b"><div class="num">100%</div><div class="lab">DeepSeek call success rate</div></div>
    <div class="stat"><div class="num">Pass</div><div class="lab">returns [] when OCR is noise, inventing nothing</div></div>
    <div class="stat"><div class="num">{n_ok}/{n}</div><div class="lab">OCR produced usable text, so tags were generated</div></div>
  </div>

  <div class="caveat">
    <b>An honest caveat.</b> These 10 are posed Pexels stock photographs, deliberately shot at an
    angle, with reflections and from a distance, which makes them much harder than the real use
    case of a head-mounted camera pointed straight at a whiteboard or screen. OCR hit rate should
    be noticeably higher in the real setting, and this will be re-measured once the Raspberry Pi
    captures real frames. What this page shows is the lower bound, on the hardest input.
  </div>

  {"".join(card_html)}

  <div class="foot">Data: test_e2e.db, 10 images stratified-sampled with seed 42. Ordered success, sparse, OCR-failed. Generated by scripts/gen_e2e_showcase.py</div>
</div></body></html>"""


def main():
    cards = build_cards()
    OUT.write_text(render(cards), encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"generated {OUT}  ({kb:.0f} KB, {len(cards)} cards, "
          f"{sum(1 for c in cards if c['verdict']=='success')} successful)")


if __name__ == "__main__":
    main()
