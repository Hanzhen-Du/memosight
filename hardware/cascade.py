#!/usr/bin/env python3
"""级联守门员主循环（今天：仅 gated 模式）。

项目命门 = 级联感知：廉价、常开的守门员只看**低分辨率灰度**流；
只有守门员判"记"时，才去抓一帧**高分辨率**图做下游重处理。
**绝不连续录像**——那会毁掉整个低功耗论点。

本循环每个 tick：
  抓低分辨率灰度帧 → infer.predict → should_record? →
    若记：抓 1 帧高清 → 存到 captures/（带时间戳）→ 写一行 hits log →（可选）推回笔记本。

采样率 --fps 是**占空比旋钮**（duty cycle），也是将来"功耗 vs 漏报"Pareto 曲线的一个轴。
fps 越低越省电、漏报风险越高；这是要表征的权衡，不是随便定的常数。

—— 关于 gated vs always 的功耗 A/B：见下方 should_record() 的 TODO。今天**不实现** always。

依赖：picamera2、ai_edge_litert（或 tensorflow）、opencv-headless、numpy。

示例（Pi 上，从仓库根目录）：
  python3 hardware/cascade.py --fps 3
  python3 hardware/cascade.py --fps 3 --threshold 0.55 --push --host 192.168.1.50 --port 8000
"""

from __future__ import annotations

import argparse
import os
import signal
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2

import infer  # 同目录模块（从仓库根目录运行：python3 hardware/cascade.py）

# ---- 流分辨率 ----
# 低分辨率流：常开喂守门员。够大以保留文字/屏幕的粗结构，又尽量小以省算力/功耗。
# 用 YUV420 是为了直接拿 Y 平面当灰度——免一次彩色→灰度转换，最省。
LORES_SIZE = (320, 240)
# 高分辨率流：仅命中时取一帧，供下游 OCR/重处理。
MAIN_SIZE = (1920, 1080)


def should_record(score: float, threshold: float, mode: str) -> bool:
    """命中决策——级联的"门"。结构上把模式隔离在这一个函数里。

    今天只有 gated：守门员分数过阈值才记（这就是低功耗论点的实现）。

    TODO(power-AB / 未来 --mode always)：在此加 always 分支做"功耗 A/B 对照"——
    强制每个 tick 都当作命中、都抓高清，用来量出 always-on 全程重处理的功耗上界，
    与 gated 对比画 Pareto。届时**只需一行**：在下面 `if mode == "always": return True`。
    今天**故意不开**，以恪守"绝不连续录像/连续重处理"的论点。
    """
    if mode == "gated":
        return score >= threshold
    if mode == "always":
        # 占位：见上 TODO。开启即把本行替换为 `return True`。
        raise NotImplementedError(
            "mode=always 尚未实现（功耗 A/B 留作将来）；今天只跑 gated。"
        )
    raise ValueError(f"未知 mode：{mode}")


def utc_stamp() -> str:
    """文件名安全的 UTC 时间戳。"""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def save_hit_image(rgb_frame, path: Path) -> bytes | None:
    """命中即把高清帧编码落盘——立即、原子、fsync、并校验。

    fail-safe 持久化：图像写在循环内、命中当下，**不**推迟到退出/关机。
    - 写前 makedirs（防目录中途被删/不存在）；
    - flush + os.fsync 把数据真正落到磁盘，abrupt termination/断电也不丢已命中的图；
    - 写后校验文件存在且非空；
    - 任何失败都**打印清晰错误（path + 异常），绝不静默吞掉**，并返回 None
      让主循环继续（不因一次写失败而崩掉、丢掉后续命中）。
    返回 JPEG 字节（供可选 --push 复用），失败返回 None。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("cv2.imencode 返回失败")
        data = buf.tobytes()
        with open(path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())  # 落盘，抗断电
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError("写后校验失败：文件不存在或为空")
        return data
    except Exception as e:  # 不静默：把失败显式打出来
        print(f"  [保存失败，未静默] path={path.resolve()} "
              f"err={type(e).__name__}: {e}")
        return None


def append_hit_log(log_path: Path, line: str) -> None:
    """逐命中追加 hits.log，append + flush + fsync，不缓冲、不推迟。"""
    try:
        with open(log_path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"  [hits.log 写入失败，未静默] path={log_path.resolve()}: {e}")


def push_frame(jpeg_bytes: bytes, host: str, port: int, name: str) -> None:
    """把高清帧 POST 回笔记本接收器（receiver_laptop.py）。

    选型理由：用 stdlib urllib + 对端 http.server，**零新依赖**（不引 requests）。
    POST 是 Pi 主动发起——Pi 不需被入站访问，只要能连到笔记本的接收端口即可。
    失败由调用方捕获并降级（帧仍在 captures/，可事后 rsync），不拖垮主循环。
    """
    url = f"http://{host}:{port}/upload?name={name}"
    req = urllib.request.Request(
        url, data=jpeg_bytes, method="POST",
        headers={"Content-Type": "image/jpeg"},
    )
    urllib.request.urlopen(req, timeout=3).read()  # 短超时，别卡住常开循环


def main() -> int:
    ap = argparse.ArgumentParser(
        description="级联守门员主循环（gated）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model", type=Path,
        default=Path.home() / "dev/memosight/models/gatekeeper_v4_mvp_int8.tflite",
    )
    ap.add_argument("--fps", type=float, default=3.0,
                    help="采样率/占空比旋钮，也是 Pareto 的一个轴")
    ap.add_argument("--threshold", type=float, default=infer.DEFAULT_THRESHOLD,
                    help="p(记) 决策阈值；FN/FP 工作点旋钮")
    ap.add_argument("--mode", choices=["gated", "always"], default="gated",
                    help="今天只支持 gated；always 是将来功耗 A/B（见 should_record TODO）")
    ap.add_argument("--outdir", type=Path, default=Path("hardware/captures"))
    # WiFi 推回：默认关，确保网络不稳时核心循环照常跑。
    ap.add_argument("--push", action="store_true", help="命中时把高清帧 POST 回笔记本")
    ap.add_argument("--host", type=str, default="127.0.0.1", help="--push 的笔记本 IP")
    ap.add_argument("--port", type=int, default=8000, help="--push 的接收端口")
    args = ap.parse_args()

    try:
        from picamera2 import Picamera2
    except ImportError:
        raise SystemExit(
            "未找到 picamera2。安装：sudo apt install -y python3-picamera2（勿用 pip）。"
        )

    # 根因修复：把输出目录解析成**绝对路径**。默认值是 CWD 相对路径，
    # 若从非仓库根目录启动（如 cd hardware 后运行、或 systemd/timeout 改了
    # WorkingDirectory），相对路径会把图写到别处，于是预期位置"captures/ 不存在"。
    # 解析为绝对路径 + 启动即打印，消除歧义；之后所有写入都用这个绝对路径。
    args.outdir = args.outdir.expanduser().resolve()
    os.makedirs(args.outdir, exist_ok=True)
    hits_log = args.outdir / "hits.log"
    print(f"命中输出目录（绝对路径）：{args.outdir}")

    # SIGTERM（timeout/kill 默认信号）也走优雅关闭释放相机。
    # 注意：图像持久化在命中当下即完成，**不依赖**能否走到这里。
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    print(f"加载守门员：{args.model}")
    interp = infer.load_model(args.model)

    # 多流：lores 常开喂守门员、main 仅命中时取。两流由 Picamera2 同时维护，
    # 但我们**每 tick 只解码 lores**；main 只在命中时 capture，避免高清流持续耗算力。
    picam = Picamera2()
    config = picam.create_video_configuration(
        main={"size": MAIN_SIZE, "format": "RGB888"},
        lores={"size": LORES_SIZE, "format": "YUV420"},
    )
    picam.configure(config)
    picam.start()

    period = 1.0 / args.fps if args.fps > 0 else 0.0
    lores_w, lores_h = LORES_SIZE
    print(f"开始级联循环：mode={args.mode} fps={args.fps} threshold={args.threshold} "
          f"push={'on→%s:%d' % (args.host, args.port) if args.push else 'off'}")
    print("Ctrl-C 退出。")

    n_tick = n_hit = 0
    try:
        while True:
            tick_start = time.perf_counter()

            # 低分辨率帧：YUV420 数组形状 (h*3/2, w)，Y(亮度)平面即前 h 行 → 天然灰度。
            yuv = picam.capture_array("lores")
            gray = yuv[:lores_h, :lores_w]

            label, score, latency_ms = infer.predict(interp, gray)
            n_tick += 1

            if should_record(score, args.threshold, args.mode):
                n_hit += 1
                stamp = utc_stamp()
                # 命中即取高清帧并**立即落盘**（级联的"重处理唤醒"）。
                hi = picam.capture_array("main")
                cap_path = args.outdir / f"hit_{stamp}_s{score:.2f}.jpg"
                jpeg_bytes = save_hit_image(hi, cap_path)  # 立即/原子/fsync/校验

                if jpeg_bytes is not None:
                    # 逐命中追加 hits.log（flush+fsync，不缓冲）。
                    append_hit_log(
                        hits_log,
                        f"{stamp}\tscore={score:.4f}\tlatency_ms={latency_ms:.2f}"
                        f"\t{cap_path.name}\n",
                    )
                    # 保留 [HIT] 行，并补上真正写入的**绝对路径**便于排查。
                    print(f"[HIT #{n_hit}] {stamp} score={score:.3f} "
                          f"lat={latency_ms:.1f}ms saved → {cap_path}")

                    if args.push:  # 可选推回；失败降级，不中断循环
                        try:
                            push_frame(jpeg_bytes, args.host, args.port, cap_path.name)
                        except (urllib.error.URLError, OSError) as e:
                            print(f"  [push 失败，已留本地待 rsync] {e}")
                # 若 save_hit_image 失败：错误已打印，主循环继续，不丢后续命中。
            else:
                # 心跳：低频打印，确认常开循环活着、没在乱触发。
                if n_tick % max(1, int(args.fps) * 10) == 0:
                    print(f"... tick {n_tick} score={score:.3f} lat={latency_ms:.1f}ms "
                          f"(hits {n_hit})")

            # 维持目标 fps：减去本 tick 已耗时间再睡。
            if period:
                time.sleep(max(0.0, period - (time.perf_counter() - tick_start)))
    except KeyboardInterrupt:
        # Ctrl-C 或 SIGTERM(timeout/kill)。命中图已在循环内逐张落盘，此处只收尾。
        print(f"\n收到中断(Ctrl-C/SIGTERM)，停止。共 {n_tick} ticks / {n_hit} hits。")
    finally:
        picam.stop()  # 释放相机
        print("相机已释放。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
