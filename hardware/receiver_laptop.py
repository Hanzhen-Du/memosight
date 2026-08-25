#!/usr/bin/env python3
"""笔记本端接收器 —— 收 cascade.py --push 推回的高清帧。

与 cascade.push_frame 配对：监听 HTTP，把 POST /upload?name=<file> 的 body(JPEG)
落盘到 --outdir。纯 stdlib(http.server)，**零新依赖**。

在**笔记本**上跑（不是 Pi）：
  python3 hardware/receiver_laptop.py --outdir ~/memosight_received --port 8000
然后 Pi 上：
  python3 hardware/cascade.py --push --host <笔记本IP> --port 8000

安全提示：这是个极简明文接收器，**只在可信局域网用**，别暴露到公网。
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def make_handler(outdir: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 (http.server 约定大写)
            parsed = urlparse(self.path)
            if parsed.path != "/upload":
                self.send_error(404, "only POST /upload")
                return
            # 文件名净化：只取 basename，挡路径穿越。
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

        def log_message(self, *a):  # 静音默认每请求日志，自己打更干净的
            pass

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(
        description="笔记本端高清帧接收器（配 cascade.py --push）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--outdir", type=Path, default=Path.home() / "memosight_received")
    ap.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.outdir))
    print(f"接收器监听 http://{args.host}:{args.port}/upload → 存到 {args.outdir}")
    print("Ctrl-C 退出。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
