#!/usr/bin/env python3
"""Laptop-side receiver for the full-resolution frames pushed back by cascade.py --push.

Pairs with cascade.push_frame: listens over HTTP and writes the JPEG body of
POST /upload?name=<file> into --outdir. Pure stdlib (http.server), so no new dependency.

Run this on the laptop, not the Pi:
  python3 hardware/receiver_laptop.py --outdir ~/memosight_received --port 8000
Then on the Pi:
  python3 hardware/cascade.py --push --host <laptop ip> --port 8000

Security note: this is a minimal plaintext receiver. Use it on a trusted local network only
and never expose it to the internet.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def make_handler(outdir: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 (http.server requires this capitalisation)
            parsed = urlparse(self.path)
            if parsed.path != "/upload":
                self.send_error(404, "only POST /upload")
                return
            # Sanitise the filename: take the basename only, to block path traversal.
            qs = parse_qs(parsed.query)
            raw = qs.get("name", ["frame.jpg"])[0]
            name = Path(raw).name or "frame.jpg"

            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            (outdir / name).write_bytes(data)

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            print(f"received {name} ({len(data)} bytes) → {outdir / name}")

        def log_message(self, *a):  # silence the default per-request log; ours is cleaner
            pass

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Laptop-side receiver for full-resolution frames, paired with cascade.py --push.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--outdir", type=Path, default=Path.home() / "memosight_received")
    ap.add_argument("--host", type=str, default="0.0.0.0", help="address to listen on")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.outdir))
    print(f"receiver listening on http://{args.host}:{args.port}/upload, saving to {args.outdir}")
    print("Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
