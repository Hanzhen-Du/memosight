#!/usr/bin/env python3
"""路径X（本地 OCR）vs 路径Y（直接传图）—— 端侧每次触发能耗的对比测量框架。

回答导师质疑：「本地 OCR 未必比直接把图传到网络更省电，而且传输更快」。
这个质疑是**合理的**，框架的立场是**不预设结论**——两条路径的能耗构成不同，谁赢取决于实测参数。

    路径X（本地 OCR）：守门员触发 → 抓高清帧 → **本地 OCR** → 只传文本（~KB）
    路径Y（直接传图）：守门员触发 → 抓高清帧 → JPEG 编码 → **传整张图**（~MB）

    E_X = E_capture + E_ocr        + E_tx(text_bytes)
    E_Y = E_capture + E_jpeg_encode + E_tx(image_bytes)
                      └ 计算侧 X 贵 ┘   └ 传输侧 Y 贵 ┘

`E_capture` 两条路径都有、且相同，比较时会抵消；仍然照测照记（它决定"每次触发"的绝对成本，
是任务C Pareto 曲线功耗轴的输入）。

═══════════════════════════════════════════════════════════════════════════
⚠️  本轮（2026-07-28）**没有真实功耗数据**。树莓派连不上，功耗计未接。
    默认的 `MockPowerMeter` 产生的是**按文档化参数表合成的占位数字**，
    **不是实测值，不得作为任何结论引用**。它的唯一作用是把整条流程跑通，
    证明"功耗计一接上，换掉 PowerMeter 一个类即可出真数据"。
    真实测量待办项见 `docs/power-measurement-method.md` 的 H1–H5。
═══════════════════════════════════════════════════════════════════════════

测量方法：**差值法**（differential method）
    先测系统空载功率 P_idle（守门员常开、无触发），再测执行某段任务时的功率 P_active，
    该段任务的**增量能耗** = (P_active − P_idle) × t。
    这样测到的是"这件事本身多花了多少电"，把板子本底功耗剥离掉——
    因为路径X/Y共享同一块板子的本底，只有增量部分才是二者的真实差异。

用法：
    # 模拟跑通（当前唯一可跑的模式）
    .venv/bin/python scripts/power_compare_framework.py --trials 30 --meter mock

    # Pi + 功耗计就绪后
    python3 scripts/power_compare_framework.py --trials 30 --meter ina219

    # 只看交叉点分析（在什么带宽/图大小下，路径X 才真的更省）
    .venv/bin/python scripts/power_compare_framework.py --crossover-only
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

# ═════════════════════════════ 1. 功耗计接口 ═════════════════════════════


class PowerMeter:
    """功耗计抽象接口。真机实现只需覆盖 `read_power_mw()`。

    约定：`read_power_mw()` 返回**当前瞬时功率（毫瓦）**，调用应尽量廉价（<1ms），
    因为 `measure()` 会在任务执行期间高频轮询它做积分。
    """

    name = "abstract"
    is_real = False  # 真机实现必须置 True —— CSV / 报告靠这个字段区分真假数据

    def read_power_mw(self) -> float:
        raise NotImplementedError

    def measure(self, fn, poll_hz: float = 200.0) -> tuple[float, float, float]:
        """执行 fn()，期间轮询功率并积分。

        返回 (平均功率 mW, 耗时 s, 能耗 mJ)。能耗 = 平均功率 × 耗时（mW × s = mJ）。
        注意：这是**绝对**能耗，尚未减去 idle 本底；差值在 `Trial` 里算。
        """
        samples: list[float] = []
        t0 = time.perf_counter()
        # 采样与任务并发：真机上建议用独立采样线程或功耗计自带的高速缓冲。
        # 这里用"任务前后 + 任务中回调"的简化积分；mock 模式下 fn 本身会驱动采样。
        result = fn(lambda: samples.append(self.read_power_mw()))
        dt = time.perf_counter() - t0
        if not samples:  # fn 没回调采样 → 至少取首尾两点
            samples = [self.read_power_mw()]
        p_mean = sum(samples) / len(samples)
        return p_mean, dt, p_mean * dt * 1000.0  # mW × s → mJ (×1000 因 dt 单位是 s)

    def idle_baseline_mw(self, seconds: float = 2.0) -> float:
        """测空载本底功率（守门员常开、无触发时）。差值法的减数。"""
        raise NotImplementedError


@dataclass
class MockParams:
    """⚠️ 占位参数表 —— 全部是**量级合理的假设值，不是实测**。

    真机就绪后这张表整个作废，由 `INA219PowerMeter` 的实测读数取代。
    这里写成显式数据类而不是散落的魔数，是为了让"哪些数字是假的"一目了然。
    量级依据（仅用于让模拟不至于荒唐，**非实测、非引用**）：Pi 5 空载数瓦级、
    WiFi 发射期功率高于空载、Tesseract 单帧 OCR 在 Pi 上是百毫秒~秒级。
    """
    p_idle_mw: float = 2800.0          # 板子 + 守门员常开的本底功率
    p_ocr_mw: float = 4200.0           # OCR 计算期间的总功率（CPU 满载）
    p_encode_mw: float = 3300.0        # JPEG 编码期间总功率
    p_capture_mw: float = 3600.0       # 相机抓高清帧期间总功率
    p_radio_tx_mw: float = 3900.0      # WiFi 发射期间总功率
    ocr_ms_mean: float = 850.0         # 单帧 OCR 耗时
    ocr_ms_jitter: float = 220.0
    encode_ms_mean: float = 45.0       # JPEG 编码耗时
    encode_ms_jitter: float = 10.0
    capture_ms_mean: float = 120.0     # 抓高清帧耗时
    capture_ms_jitter: float = 25.0
    radio_wake_ms: float = 180.0       # 每次传输的射频唤醒/关联固定开销
    throughput_kbps: float = 6000.0    # 有效吞吐（含协议开销），WiFi 实测常远低于标称
    throughput_jitter: float = 1800.0  # 网络状况波动 —— 这正是"答案取决于实测"的原因之一
    image_bytes_mean: float = 900_000  # 1920×1080 JPEG q85 量级
    image_bytes_jitter: float = 250_000
    text_bytes_mean: float = 1_400     # 一屏文字的 OCR 输出
    text_bytes_jitter: float = 700


class MockPowerMeter(PowerMeter):
    """⚠️ 模拟功耗计 —— 产生合成读数，**不是实测**。仅用于跑通流程。"""

    name = "mock"
    is_real = False

    def __init__(self, params: MockParams, seed: int = 42):
        self.p = params
        self.rng = random.Random(seed)
        self._level_mw = params.p_idle_mw  # 当前"正在做什么"决定的功率档位

    def set_level(self, mw: float) -> None:
        self._level_mw = mw

    def read_power_mw(self) -> float:
        # 加一点读数噪声，让积分不是常数（真功耗计也有噪声）
        return self._level_mw + self.rng.gauss(0, self._level_mw * 0.01)

    def idle_baseline_mw(self, seconds: float = 2.0) -> float:
        self.set_level(self.p.p_idle_mw)
        return sum(self.read_power_mw() for _ in range(64)) / 64


class INA219PowerMeter(PowerMeter):
    """真机功耗计占位实现 —— **待硬件就绪**（见 `docs/power-measurement-method.md` H1–H5）。

    接线（供将来实现参考）：INA219 高侧串在 Pi 5 的 5V 供电回路上，I²C 接 Pi 或第二块板。
    第二块板读数更干净（不让被测板承担采样开销），但需要时间同步。

    真机实现清单：
      1. 打开 I²C（smbus2 / adafruit-circuitpython-ina219），设分流电阻与量程；
      2. `read_power_mw()` 返回 bus_voltage × current（或直接读 power 寄存器）；
      3. `idle_baseline_mw()` 在守门员常开、**确保无触发**的窗口里采 ≥2s；
      4. 校准：先用已知阻性负载核对读数，再上被测系统；
      5. 采样率 ≥200Hz，否则积不准短任务（JPEG 编码只有几十毫秒）。
    """

    name = "ina219"
    is_real = True

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "INA219PowerMeter 尚未实现 —— 需要 Pi + 功耗计接上后才能写和验证。\n"
            "本轮（Pi 连不上）只提供框架；请用 --meter mock 跑模拟验证。\n"
            "待办见 docs/power-measurement-method.md 的 H1–H5。")


METERS = {"mock": MockPowerMeter, "ina219": INA219PowerMeter}


# ═════════════════════════════ 2. 两条路径 ═════════════════════════════


@dataclass
class Trial:
    """一次触发的完整记录。CSV 的一行。"""
    trial: int
    path: str                  # "X_local_ocr" | "Y_send_image"
    meter: str                 # 功耗计类型
    is_real: bool              # ⚠️ False = 模拟数据，不得当结论
    p_idle_mw: float           # 差值法的减数
    capture_ms: float
    compute_ms: float          # X=OCR 耗时；Y=JPEG 编码耗时
    tx_bytes: int
    tx_ms: float               # 含射频唤醒开销
    e_capture_mj: float        # 以下均为**扣除 idle 后的增量能耗**
    e_compute_mj: float
    e_tx_mj: float
    e_total_mj: float
    latency_ms: float          # 端到端时延（导师说"传图更快"，这一列就是用来验证的）
    notes: str = ""


def _jitter(rng: random.Random, mean: float, spread: float, lo: float = 0.0) -> float:
    return max(lo, rng.gauss(mean, spread / 2.0))


def _run_segment(meter: PowerMeter, level_mw: float, duration_ms: float,
                 p_idle_mw: float) -> tuple[float, float]:
    """执行一个耗电段，返回 (实际耗时 ms, 扣除 idle 后的增量能耗 mJ)。

    真机上 `level_mw` 参数会被忽略 —— 真实功率由功耗计读出，而不是我们设定。
    模拟模式下用它来驱动 MockPowerMeter 的档位。
    """
    if isinstance(meter, MockPowerMeter):
        meter.set_level(level_mw)

    def work(sample):
        # 模拟模式：不真的 sleep 整段时间（跑 30 次触发要几十秒），
        # 而是按段长比例采样若干点做积分。真机模式下这里换成真实的任务调用。
        n = max(4, int(duration_ms / 5))
        for _ in range(n):
            sample()
        return None

    p_mean, _, _ = meter.measure(work)
    dt_s = duration_ms / 1000.0
    e_incremental_mj = (p_mean - p_idle_mw) * dt_s   # ← 差值法核心
    return duration_ms, e_incremental_mj


def run_path_x(meter: PowerMeter, p: MockParams, rng: random.Random,
               p_idle_mw: float, trial: int) -> Trial:
    """路径X：本地 OCR，只传文本。计算贵、传输便宜。"""
    cap_ms, e_cap = _run_segment(meter, p.p_capture_mw,
                                 _jitter(rng, p.capture_ms_mean, p.capture_ms_jitter, 1),
                                 p_idle_mw)
    ocr_ms, e_ocr = _run_segment(meter, p.p_ocr_mw,
                                 _jitter(rng, p.ocr_ms_mean, p.ocr_ms_jitter, 1),
                                 p_idle_mw)
    text_bytes = int(_jitter(rng, p.text_bytes_mean, p.text_bytes_jitter, 64))
    tput = _jitter(rng, p.throughput_kbps, p.throughput_jitter, 200)
    tx_ms = p.radio_wake_ms + (text_bytes * 8 / 1000.0) / tput * 1000.0
    tx_ms, e_tx = _run_segment(meter, p.p_radio_tx_mw, tx_ms, p_idle_mw)
    return Trial(
        trial=trial, path="X_local_ocr", meter=meter.name, is_real=meter.is_real,
        p_idle_mw=round(p_idle_mw, 1), capture_ms=round(cap_ms, 1),
        compute_ms=round(ocr_ms, 1), tx_bytes=text_bytes, tx_ms=round(tx_ms, 1),
        e_capture_mj=round(e_cap, 2), e_compute_mj=round(e_ocr, 2),
        e_tx_mj=round(e_tx, 2), e_total_mj=round(e_cap + e_ocr + e_tx, 2),
        latency_ms=round(cap_ms + ocr_ms + tx_ms, 1),
        notes="compute=Tesseract OCR; tx=text only")


def run_path_y(meter: PowerMeter, p: MockParams, rng: random.Random,
               p_idle_mw: float, trial: int) -> Trial:
    """路径Y：直接传高清图。计算几乎为零、传输贵。"""
    cap_ms, e_cap = _run_segment(meter, p.p_capture_mw,
                                 _jitter(rng, p.capture_ms_mean, p.capture_ms_jitter, 1),
                                 p_idle_mw)
    enc_ms, e_enc = _run_segment(meter, p.p_encode_mw,
                                 _jitter(rng, p.encode_ms_mean, p.encode_ms_jitter, 1),
                                 p_idle_mw)
    img_bytes = int(_jitter(rng, p.image_bytes_mean, p.image_bytes_jitter, 10_000))
    tput = _jitter(rng, p.throughput_kbps, p.throughput_jitter, 200)
    tx_ms = p.radio_wake_ms + (img_bytes * 8 / 1000.0) / tput * 1000.0
    tx_ms, e_tx = _run_segment(meter, p.p_radio_tx_mw, tx_ms, p_idle_mw)
    return Trial(
        trial=trial, path="Y_send_image", meter=meter.name, is_real=meter.is_real,
        p_idle_mw=round(p_idle_mw, 1), capture_ms=round(cap_ms, 1),
        compute_ms=round(enc_ms, 1), tx_bytes=img_bytes, tx_ms=round(tx_ms, 1),
        e_capture_mj=round(e_cap, 2), e_compute_mj=round(e_enc, 2),
        e_tx_mj=round(e_tx, 2), e_total_mj=round(e_cap + e_enc + e_tx, 2),
        latency_ms=round(cap_ms + enc_ms + tx_ms, 1),
        notes="compute=JPEG encode only; tx=full image")


# ═════════════════════════════ 3. 交叉点分析 ═════════════════════════════


def crossover_analysis(p: MockParams) -> list[dict]:
    """在什么条件下路径X（本地 OCR）才真的更省？—— 解析解，不依赖功耗计。

    忽略两路共有的 capture，令二者传输功率相同：
        E_X − E_Y = (P_ocr−P_idle)·t_ocr − (P_enc−P_idle)·t_enc
                    + (P_tx−P_idle)·(t_txX − t_txY)
    其中 t_txY − t_txX ≈ (image_bytes − text_bytes)·8 / throughput。
    令差为 0，解出**盈亏平衡吞吐**：低于它传图太慢太贵→X 赢；高于它网络便宜→Y 赢。

    这个函数的**结构**是真的（能量守恒 + 线性传输模型），代入的**参数**目前是占位值。
    真机测出 P_ocr / t_ocr / P_tx 后，同一函数直接给出真实交叉点。
    """
    dp_ocr = p.p_ocr_mw - p.p_idle_mw
    dp_enc = p.p_encode_mw - p.p_idle_mw
    dp_tx = p.p_radio_tx_mw - p.p_idle_mw
    d_bytes = p.image_bytes_mean - p.text_bytes_mean
    # 计算侧 X 比 Y 多花的能耗 (mJ) 与时间 (ms)
    e_compute_penalty = (dp_ocr * p.ocr_ms_mean - dp_enc * p.encode_ms_mean) / 1000.0
    t_compute_penalty = p.ocr_ms_mean - p.encode_ms_mean
    rows = []
    for tput_kbps in (500, 1000, 2000, 4000, 6000, 8000, 12000, 25000, 50000):
        d_tx_ms = (d_bytes * 8 / 1000.0) / tput_kbps * 1000.0
        e_tx_saving = dp_tx * d_tx_ms / 1000.0     # X 在传输侧省下的能耗 (mJ)
        net_e = e_tx_saving - e_compute_penalty    # >0 → 路径X 更省电
        net_t = d_tx_ms - t_compute_penalty        # >0 → 路径X 更快
        rows.append({
            "throughput_kbps": tput_kbps,
            "tx_time_saved_ms": round(d_tx_ms, 1),
            "e_tx_saved_mj": round(e_tx_saving, 1),
            "e_compute_penalty_mj": round(e_compute_penalty, 1),
            "net_mj_x_minus_y": round(-net_e, 1),   # 负 = X 更省电
            "winner_energy": "X_local_ocr" if net_e > 0 else "Y_send_image",
            "winner_latency": "X_local_ocr" if net_t > 0 else "Y_send_image",
        })
    # 盈亏平衡吞吐：省电与省时是**两个不同的方程**，交叉点一般不重合。
    #   能耗： dp_tx · Δt_tx == e_compute_penalty
    #   时延： Δt_tx        == t_compute_penalty
    be_energy = ((dp_tx * (d_bytes * 8 / 1000.0)) / e_compute_penalty
                 if e_compute_penalty > 0 else float("inf"))
    be_latency = ((d_bytes * 8 / 1000.0) / (t_compute_penalty / 1000.0)
                  if t_compute_penalty > 0 else float("inf"))
    return rows, be_energy, be_latency


# ═════════════════════════════ 4. 主流程 ═════════════════════════════


BANNER_MOCK = """
╔══════════════════════════════════════════════════════════════════════╗
║  ⚠️  模拟模式（--meter mock）—— 以下所有能耗数字都是**合成占位值**    ║
║     不是实测、不得引用为结论。真数字需 Pi + 功耗计（见 docs）。      ║
║     本次运行的目的只有一个：证明框架能跑通、CSV 结构可用。           ║
╚══════════════════════════════════════════════════════════════════════╝"""


def summarize(trials: list[Trial]) -> dict:
    out = {}
    for path in ("X_local_ocr", "Y_send_image"):
        rows = [t for t in trials if t.path == path]
        if not rows:
            continue
        n = len(rows)
        out[path] = {
            "n": n,
            "e_total_mj": sum(t.e_total_mj for t in rows) / n,
            "e_compute_mj": sum(t.e_compute_mj for t in rows) / n,
            "e_tx_mj": sum(t.e_tx_mj for t in rows) / n,
            "tx_bytes": sum(t.tx_bytes for t in rows) / n,
            "latency_ms": sum(t.latency_ms for t in rows) / n,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=30, help="每条路径跑多少次触发")
    ap.add_argument("--meter", choices=list(METERS), default="mock")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/power_compare_trials.csv"))
    ap.add_argument("--crossover-only", action="store_true",
                    help="只做交叉点分析（不需要功耗计，纯解析）")
    args = ap.parse_args()

    params = MockParams()
    rng = random.Random(args.seed)

    if args.crossover_only:
        rows, be_e, be_t = crossover_analysis(params)
        print("\n交叉点分析（公式结构真实，代入参数为占位值 —— 待实测替换）")
        print(f"{'吞吐 kbps':>10} {'省下传输 ms':>12} {'省下能耗 mJ':>12} "
              f"{'OCR多花 mJ':>11} {'净 X−Y mJ':>11}  {'省电赢家':<13} {'省时赢家':<13}")
        for r in rows:
            print(f"{r['throughput_kbps']:>10} {r['tx_time_saved_ms']:>12.1f} "
                  f"{r['e_tx_saved_mj']:>12.1f} {r['e_compute_penalty_mj']:>11.1f} "
                  f"{r['net_mj_x_minus_y']:>11.1f}  {r['winner_energy']:<13} "
                  f"{r['winner_latency']:<13}")
        print(f"\n盈亏平衡吞吐（能耗）≈ {be_e:,.0f} kbps —— 网络比这快 → 传图更省电。")
        print(f"盈亏平衡吞吐（时延）≈ {be_t:,.0f} kbps —— 网络比这快 → 传图更快。")
        if be_t > be_e:
            print(f"→ 注意 {be_e:,.0f}–{be_t:,.0f} kbps 之间存在**矛盾区**："
                  f"传图更省电，但本地 OCR 更快。**省电和省时不是同一个问题。**")
        print("⚠️ 上述数字由占位参数算出，**不是结论**。真机测出 P_ocr/t_ocr/P_tx 后重跑本函数。")
        return 0

    try:
        meter = METERS[args.meter](params, seed=args.seed)
    except NotImplementedError as e:
        print(f"\n❌ {e}", file=sys.stderr)
        return 2

    if not meter.is_real:
        print(BANNER_MOCK)

    p_idle = meter.idle_baseline_mw()
    print(f"\n[1/3] idle 本底功率（差值法减数）：{p_idle:.1f} mW  "
          f"[{'实测' if meter.is_real else '模拟'}]")

    print(f"[2/3] 跑 {args.trials} 次触发 × 2 条路径 …")
    trials: list[Trial] = []
    for i in range(args.trials):
        trials.append(run_path_x(meter, params, rng, p_idle, i))
        trials.append(run_path_y(meter, params, rng, p_idle, i))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(trials[0]).keys()))
        w.writeheader()
        for t in trials:
            w.writerow(asdict(t))
    print(f"[3/3] 写出 {len(trials)} 行 → {args.out}")

    s = summarize(trials)
    print(f"\n{'路径':<14} {'总能耗 mJ':>11} {'计算 mJ':>10} {'传输 mJ':>10} "
          f"{'传输字节':>11} {'时延 ms':>10}")
    for path, v in s.items():
        print(f"{path:<14} {v['e_total_mj']:>11.1f} {v['e_compute_mj']:>10.1f} "
              f"{v['e_tx_mj']:>10.1f} {v['tx_bytes']:>11,.0f} {v['latency_ms']:>10.1f}")

    if len(s) == 2:
        dx = s["X_local_ocr"]["e_total_mj"] - s["Y_send_image"]["e_total_mj"]
        dl = s["X_local_ocr"]["latency_ms"] - s["Y_send_image"]["latency_ms"]
        verdict = "路径X（本地OCR）更省" if dx < 0 else "路径Y（传图）更省"
        lat = "路径X更快" if dl < 0 else "路径Y更快（导师的直觉在此参数下成立）"
        print(f"\n能耗差 X−Y = {dx:+.1f} mJ/次 → {verdict}")
        print(f"时延差 X−Y = {dl:+.1f} ms/次 → {lat}")

    if not meter.is_real:
        print("\n⚠️ 重申：以上全部是模拟数字，**不是实测结论**。"
              "框架已跑通；真数据待 Pi + 功耗计（见 docs/power-measurement-method.md H1–H5）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
