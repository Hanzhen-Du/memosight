#!/usr/bin/env python3
"""记录频率 → 功耗 / 漏报 / 误唤醒 的 Pareto 曲线框架。

回答导师问题 3：「长时间看屏幕但只想记关键信息，怎么选记录频率？」
——把"记录频率"这个模糊的产品旋钮，翻译成三个可测量的量，扫出权衡曲线，让工作点可选而不是拍脑袋。

    记录频率由三个旋钮控制：
      ① 守门员触发阈值 threshold —— 越低越爱记（漏报↓、误唤醒↑）
      ② 守门员采样率 fps         —— 越高越不容易错过（漏报↓、常开功耗↑）
      ③ 去抖间隔 debounce        —— 两次重处理的最小间隔（功耗↓、可能漏掉紧接着的新屏）

    三个指标：
      ① miss rate      漏报率：该记的屏幕没记下来
      ② power          平均功耗：常开守门员 + 触发×重处理
      ③ false-wake     误唤醒率：不该记的画面触发了昂贵的重处理

═══════════════════════════════════════════════════════════════════════════════
数据真假分层（**这是本框架最重要的一件事，读结果前必须先看**）：

  ✅ **真实软件评估**（不需要任何硬件，本轮已算出真数字）：
     - 每帧漏报率 / 误唤醒率随阈值的变化 —— 来自当前守门员在**固定 held-out 探针**上的真实分数
       (person_screen 181 张 GT=记 / person_noscreen 235 张 GT=不记，int8 部署口径，零泄漏核验过)

  🔶 **模型推算**（结构可辩护，但含显式假设，需真实录像验证）：
     - 单帧漏报 → 整段漏报（多次观看、帧间高度相关）
     - 去抖导致的额外漏报
     - 场景先验（多大比例的时间里真的有该记的屏）

  ⚠️ **占位数字**（**不是实测，禁止引用**，待 Pi + 功耗计）：
     - 功耗轴的绝对值：每 tick 守门员能耗、每次触发重处理能耗、板子本底功率
       → 复用 `power_compare_framework.MockParams`，真机替换后本框架自动出真值
═══════════════════════════════════════════════════════════════════════════════

用法：
    .venv/bin/python scripts/pareto_framework.py                 # 扫描 + 出图
    .venv/bin/python scripts/pareto_framework.py --trigger-path Y  # 重处理走"传图"路径
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from power_compare_framework import MockParams  # noqa: E402  复用同一套能耗占位参数

# ───────────────────────── 探针（真实软件评估的数据源）─────────────────────────
PROBE_SCREEN = Path("data/processed/probe_person_screen_audit_cwu/probe_scores.csv")
PROBE_NOSCREEN = Path("data/processed/probe_person_noscreen_audit_cwu/probe_scores.csv")
SCORE_COL = "p_record_int8"   # int8 部署口径（与真机跑的是同一个数值管线）


# ═══════════════════════ 1. 能耗模型（⚠️ 占位参数）═══════════════════════


@dataclass
class EnergyModel:
    """⚠️ 全部为占位值 —— 真机测出后替换（见 `docs/pareto-method.md` H1/H6）。

    结构是真的（能量 = 常开本底 + 采样×每tick + 触发率×每次重处理），
    只有代入的**数值**是假的。
    """
    p_base_mw: float = 2600.0       # 板子本底（不含守门员）
    e_tick_mj: float = 12.0         # 每 tick：低分辨率抓帧 + int8 守门员推理
    e_trigger_mj: float = 0.0       # 每次触发的重处理能耗（由 --trigger-path 决定，见下）

    def avg_power_mw(self, fps: float, triggers_per_s: float) -> float:
        """平均功耗 = 本底 + 采样功耗 + 触发功耗。

        单位：mJ/s == mW，所以 (次/秒 × mJ/次) 直接就是 mW，可以相加。
        """
        return (self.p_base_mw
                + fps * self.e_tick_mj
                + triggers_per_s * self.e_trigger_mj)


def trigger_energy_mj(params: MockParams, path: str) -> float:
    """每次触发的重处理能耗（mJ）—— 与任务B 的两条路径口径一致。

    ⚠️ 占位值。这里直接引用 `power_compare_framework` 的同一套参数，
    保证两个框架的能耗口径不打架：任务B 测出真值后，本框架自动跟着变。
    """
    dp_cap = params.p_capture_mw - params.p_idle_mw
    dp_ocr = params.p_ocr_mw - params.p_idle_mw
    dp_enc = params.p_encode_mw - params.p_idle_mw
    dp_tx = params.p_radio_tx_mw - params.p_idle_mw
    e_cap = dp_cap * params.capture_ms_mean / 1000.0

    def e_tx(nbytes: float) -> float:
        t_ms = params.radio_wake_ms + (nbytes * 8 / 1000.0) / params.throughput_kbps * 1000.0
        return dp_tx * t_ms / 1000.0

    if path == "X":   # 本地 OCR，只传文本
        return e_cap + dp_ocr * params.ocr_ms_mean / 1000.0 + e_tx(params.text_bytes_mean)
    if path == "Y":   # 直接传图
        return e_cap + dp_enc * params.encode_ms_mean / 1000.0 + e_tx(params.image_bytes_mean)
    raise ValueError(f"未知重处理路径：{path}（应为 X 或 Y）")


# ═══════════════════ 2. 探针 → 每帧漏报 / 误唤醒（✅ 真实）═══════════════════


def load_probe_scores() -> tuple[pd.Series, pd.Series]:
    """载入两个固定 held-out 探针的真实分数（当前守门员 int8 部署产物）。"""
    missing = [p for p in (PROBE_SCREEN, PROBE_NOSCREEN) if not p.exists()]
    if missing:
        raise SystemExit(
            "❌ 缺少探针分数文件：\n  " + "\n  ".join(str(m) for m in missing) +
            "\n请先用当前守门员重打分：\n"
            "  PYTHONPATH=scripts .venv/bin/python scripts/probe_fp_test.py \\\n"
            "    --probe-dir data/probe_person_screen \\\n"
            "    --keras-model models/task1_candidates/gatekeeper_task1_C_wide_uniform.keras \\\n"
            "    --int8-model models/task1_candidates/gatekeeper_task1_C_wide_uniform_int8.tflite \\\n"
            "    --out data/processed/probe_person_screen_audit_cwu --no-gradcam")
    return (pd.read_csv(PROBE_SCREEN)[SCORE_COL],
            pd.read_csv(PROBE_NOSCREEN)[SCORE_COL])


def frame_metrics(screen: pd.Series, noscreen: pd.Series, thr: float) -> tuple[float, float]:
    """✅ 真实：给定阈值下的**每帧**漏报率与误唤醒率。

    miss_frame = 该记的屏幕中，单帧被判"不记"的比例  = 1 − recall
    false_wake = 不该记的画面中，被判"记"的比例      = FP 率
    """
    recall = float((screen >= thr).mean())
    false_wake = float((noscreen >= thr).mean())
    return 1.0 - recall, false_wake


# ═══════════════════ 3. 每帧 → 整段（🔶 模型，含显式假设）═══════════════════


def _phi(z: float) -> float:
    """标准正态 CDF（stdlib 实现，避免引入 scipy）。"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class SessionModel:
    """🔶 把"单帧漏报"换算成"整段漏报"的模型。**假设需用真实录像验证（见 `docs/pareto-method.md` H7）。**

    ⚠️ 这里刻意**不用**「每帧独立试验」那种模型（session_miss = miss_frame^k）。
    那个模型会给出荒谬的结论：多看几次漏报就趋近 0。现实不是这样——
    **有些屏幕是系统性漏的**：模型对它的打分本来就远低于阈值，看一百次还是一百次判错。

    本模型用**逐屏分数**而不是聚合漏报率：
        一块屏 i 的探针分数 s_i 是它的"固有难度"；不同视角/抖动带来分数扰动 σ。
        单次观看漏掉它的概率 = Φ((thr − s_i)/σ)
        看 k_eff 次全漏的概率  = Φ((thr − s_i)/σ)^k_eff
        整段漏报率 = 对所有屏取平均
    σ→0 时自动退化为"每帧漏报率"（分数远低于阈值的屏永远漏 = 漏报地板），这是正确行为。
    """

    t_visible_s: float = 20.0   # 一块该记的屏幕在视野里停留多久（假设）
    rho: float = 0.9            # 帧间相关系数：连续帧看同一块屏，判错高度相关（假设）
    p_screen: float = 0.15      # 场景先验：多大比例的时间里视野内真有该记的屏（假设）
    score_jitter: float = 0.10  # σ：同一块屏在不同视角/抖动下的分数扰动（假设，待录像标定）

    def effective_looks(self, fps: float) -> float:
        """有效独立观看次数。

        看 k = fps × t_visible 次，但连续帧几乎是同一张图 —— 判错高度相关，
        不能当成 k 次独立试验（那会严重低估漏报）。用相关系数折算：
            k_eff = 1 + (k − 1)(1 − ρ)
        ρ=1 → k_eff=1（完全相关，多看无用）；ρ=0 → k_eff=k（完全独立，教科书情形）。
        默认 ρ=0.9 是**保守假设**，宁可高估漏报。
        """
        k = max(1.0, fps * self.t_visible_s)
        return 1.0 + (k - 1.0) * (1.0 - self.rho)

    def session_miss(self, screen_scores, thr: float, fps: float,
                     debounce_s: float, triggers_per_s: float) -> float:
        """整段漏报率 = 多次观看仍全漏 + 被去抖挡掉。"""
        k_eff = self.effective_looks(fps)
        sigma = max(1e-6, self.score_jitter)
        miss_multi = float(
            sum(_phi((thr - s) / sigma) ** k_eff for s in screen_scores)
            / len(screen_scores))
        # 去抖：只有当去抖窗口比屏幕停留时间还长时，才可能整块屏都被挡在窗口里。
        # D ≤ t_visible 时，窗口过期后仍有机会看到 → 不额外漏。
        if debounce_s > self.t_visible_s:
            shadow = min(1.0, triggers_per_s * debounce_s)          # 处于去抖阴影的时间占比
            p_debounce_miss = shadow * (1.0 - self.t_visible_s / debounce_s)
        else:
            p_debounce_miss = 0.0
        return miss_multi + (1.0 - miss_multi) * p_debounce_miss


# ═══════════════════════════ 4. 扫描 ═══════════════════════════


@dataclass
class ConfigResult:
    threshold: float
    fps: float
    debounce_s: float
    miss_rate_frame: float       # ✅ 真实（探针实测）
    false_wake_rate: float       # ✅ 真实（探针实测）
    miss_rate_session: float     # 🔶 模型
    triggers_per_min: float      # 🔶 模型（含场景先验）
    power_mw: float              # ⚠️ 占位（能耗模型 + mock 参数）
    power_controllable_mw: float # ⚠️ 占位；扣掉板子本底后**这个旋钮真正能控制的**那部分功耗
    battery_hours_1000mah: float # ⚠️ 占位（同上，仅供直觉，别当规格）
    is_pareto: bool = False
    data_status: str = "miss/false_wake=REAL_probe; power=MOCK_placeholder"


def sweep(screen, noscreen, energy: EnergyModel, session: SessionModel,
          thresholds, fps_list, debounce_list) -> list[ConfigResult]:
    rows: list[ConfigResult] = []
    for thr in thresholds:
        miss_f, fwake = frame_metrics(screen, noscreen, thr)
        recall = 1.0 - miss_f
        for fps in fps_list:
            # 每 tick 触发概率 = 有屏时正确触发 + 无屏时误触发（场景先验是假设）
            p_trigger = session.p_screen * recall + (1 - session.p_screen) * fwake
            raw_tps = fps * p_trigger
            for deb in debounce_list:
                tps = min(raw_tps, 1.0 / deb) if deb > 0 else raw_tps  # 去抖封顶
                miss_s = session.session_miss(screen, thr, fps, deb, tps)
                power = energy.avg_power_mw(fps, tps)
                rows.append(ConfigResult(
                    threshold=round(thr, 3), fps=fps, debounce_s=deb,
                    miss_rate_frame=round(miss_f, 4),
                    false_wake_rate=round(fwake, 4),
                    miss_rate_session=round(miss_s, 4),
                    triggers_per_min=round(tps * 60, 2),
                    power_mw=round(power, 1),
                    power_controllable_mw=round(power - energy.p_base_mw, 2),
                    battery_hours_1000mah=round(1000 * 3.7 / power, 2) if power > 0 else 0.0,
                ))
    return rows


def mark_pareto(rows: list[ConfigResult]) -> list[ConfigResult]:
    """标记 Pareto 前沿：**同时**最小化 (功耗, 整段漏报)。

    一个配置在前沿上 = 不存在另一个配置在两个指标上都不差、且至少一个更好。
    """
    for a in rows:
        a.is_pareto = not any(
            (b.power_mw <= a.power_mw and b.miss_rate_session <= a.miss_rate_session)
            and (b.power_mw < a.power_mw or b.miss_rate_session < a.miss_rate_session)
            for b in rows)
    # 并列去重：很多配置指标完全相同（例如去抖间隔小于触发间隔时，去抖根本没生效）。
    # 同一 (功耗, 漏报) 上只保留一个代表，取**最大去抖间隔**——指标一样时，
    # 去抖越大越省事（更少唤醒尖峰、对突发场景更稳），是免费的选择。
    best: dict[tuple[float, float], ConfigResult] = {}
    for r in rows:
        if not r.is_pareto:
            continue
        key = (r.power_mw, r.miss_rate_session)
        cur = best.get(key)
        if cur is None or r.debounce_s > cur.debounce_s:
            best[key] = r
    keep = {id(r) for r in best.values()}
    for r in rows:
        if r.is_pareto and id(r) not in keep:
            r.is_pareto = False
    return rows


# ═══════════════════════════ 5. 出图 ═══════════════════════════
# 说明：本机 venv **没有 matplotlib**，且项目规则禁止未经用户同意装包。
# 所以这里实现双后端：有 matplotlib 就用它，没有就用 PIL 手绘（零新依赖）。
# 用户批准装 matplotlib 后，本函数自动走 matplotlib 分支，无需改代码。

PLOT_W, PLOT_H = 1100, 720
MARGIN = dict(l=95, r=250, t=80, b=80)


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    path = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _thr_color(t: float, tmin: float, tmax: float) -> tuple[int, int, int]:
    """阈值 → 颜色（低阈值=爱记=红，高阈值=保守=蓝）。"""
    f = (t - tmin) / (tmax - tmin) if tmax > tmin else 0.5
    return (int(220 - 170 * f), int(70 + 60 * f), int(60 + 170 * f))


def plot_pareto_pil(rows: list[ConfigResult], out_png: Path) -> None:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (PLOT_W, PLOT_H), "white")
    d = ImageDraw.Draw(img)
    x0, y0 = MARGIN["l"], PLOT_H - MARGIN["b"]
    x1, y1 = PLOT_W - MARGIN["r"], MARGIN["t"]

    # 功耗跨两个数量级（十几 mW ~ 几千 mW）→ **对数横轴**，否则前沿全挤在左边一条线上。
    xs = [max(1e-3, r.power_controllable_mw) for r in rows]
    ys = [r.miss_rate_session for r in rows]
    lxlo, lxhi = math.log10(min(xs)), math.log10(max(xs))
    lxlo, lxhi = lxlo - (lxhi - lxlo) * 0.06, lxhi + (lxhi - lxlo) * 0.06
    ylo, yhi = min(ys), max(ys)
    pad_y = (yhi - ylo) * 0.08 or 0.01
    ylo, yhi = max(0.0, ylo - pad_y), min(1.0, yhi + pad_y)

    def px(v): return x0 + (math.log10(max(1e-3, v)) - lxlo) / (lxhi - lxlo) * (x1 - x0)
    def py(v): return y0 - (v - ylo) / (yhi - ylo) * (y0 - y1)

    f_tick, f_lab, f_title, f_small = _font(13), _font(15, True), _font(19, True), _font(11)

    # 网格 + 刻度（横轴按 10 的整数次幂 + 2/5 中间刻度）
    decade_ticks = []
    dec = math.floor(lxlo)
    while dec <= math.ceil(lxhi):
        for m in (1, 2, 5):
            v = m * 10 ** dec
            if lxlo <= math.log10(v) <= lxhi:
                decade_ticks.append(v)
        dec += 1
    for v in decade_ticks:
        gx = px(v)
        d.line([(gx, y0), (gx, y1)], fill="#eeeeee")
        lab = f"{v:.0f}" if v >= 1 else f"{v:g}"
        d.text((gx - 4 * len(lab), y0 + 8), lab, font=f_tick, fill="#444")
    for i in range(6):
        gy = y0 - (y0 - y1) * i / 5
        d.line([(x0, gy), (x1, gy)], fill="#eeeeee")
        d.text((x0 - 52, gy - 7), f"{(ylo + (yhi - ylo) * i / 5) * 100:.0f}%",
               font=f_tick, fill="#444")
    d.line([(x0, y0), (x1, y0)], fill="#333", width=2)
    d.line([(x0, y0), (x0, y1)], fill="#333", width=2)

    tmin = min(r.threshold for r in rows)
    tmax = max(r.threshold for r in rows)

    # 非前沿点（浅灰）
    for r in rows:
        if not r.is_pareto:
            x, y = px(r.power_controllable_mw), py(r.miss_rate_session)
            d.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill="#cccccc")

    # Pareto 前沿：连线 + 彩色点
    front = sorted([r for r in rows if r.is_pareto], key=lambda r: r.power_controllable_mw)
    if len(front) > 1:
        d.line([(px(r.power_controllable_mw), py(r.miss_rate_session)) for r in front],
               fill="#222222", width=2)
    for r in front:
        x, y = px(r.power_controllable_mw), py(r.miss_rate_session)
        c = _thr_color(r.threshold, tmin, tmax)
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=c, outline="white", width=2)

    # 标题 / 轴标签
    d.text((MARGIN["l"], 24), "MemoSight Gatekeeper — Power vs Miss-rate Pareto Front",
           font=f_title, fill="#111")
    d.text((MARGIN["l"], 50),
           "Recording frequency swept via threshold / sampling fps / debounce interval",
           font=f_tick, fill="#666")
    d.text((x0 + (x1 - x0) / 2 - 200, y0 + 38),
           "Controllable power (mW, log scale)  =  total \u2212 board idle baseline",
           font=f_lab, fill="#111")
    # \u7eb5\u8f74\u6807\u7b7e\uff1a\u5355\u72ec\u6e32\u67d3\u518d\u65cb\u8f6c 90\u00b0\uff08\u9010\u5b57\u6bcd\u7ad6\u6392\u592a\u96be\u770b\uff09
    ylab = Image.new("RGB", (240, 22), "white")
    ImageDraw.Draw(ylab).text((0, 0), "Miss rate  (session, modeled)", font=f_lab, fill="#111")
    img.paste(ylab.rotate(90, expand=True), (18, (y1 + y0) // 2 - 120))

    # 图例
    lx = x1 + 25
    d.text((lx, y1), "Pareto front", font=f_lab, fill="#111")
    d.line([(lx, y1 + 24), (lx + 30, y1 + 24)], fill="#222", width=2)
    d.ellipse([lx + 10, y1 + 18, lx + 22, y1 + 30], fill=_thr_color(tmin, tmin, tmax),
              outline="white", width=2)
    d.text((lx + 40, y1 + 17), "optimal configs", font=f_tick, fill="#444")
    d.ellipse([lx + 12, y1 + 46, lx + 20, y1 + 54], fill="#cccccc")
    d.text((lx + 40, y1 + 44), "dominated", font=f_tick, fill="#444")

    d.text((lx, y1 + 80), "Threshold", font=f_lab, fill="#111")
    for i in range(6):
        t = tmin + (tmax - tmin) * i / 5
        yy = y1 + 106 + i * 20
        d.ellipse([lx + 10, yy, lx + 22, yy + 12], fill=_thr_color(t, tmin, tmax),
                  outline="white", width=1)
        d.text((lx + 40, yy), f"{t:.2f}", font=f_tick, fill="#444")

    # 前沿工作点标注（只标几个，避免糊成一片）
    if front:
        for r in (front[0], front[len(front) // 2], front[-1]):
            x, y = px(r.power_controllable_mw), py(r.miss_rate_session)
            d.text((x + 10, y - 14),
                   f"thr{r.threshold:.2f}/{r.fps:g}fps/{r.debounce_s:g}s",
                   font=f_small, fill="#333")

    # ⚠️ 数据真假声明（图上必须带，防止被单独截图引用）
    d.rectangle([MARGIN["l"], PLOT_H - 46, PLOT_W - 25, PLOT_H - 6], fill="#fff4f4",
                outline="#e0b4b4")
    d.text((MARGIN["l"] + 10, PLOT_H - 40),
           "X-axis (power) = MODEL with PLACEHOLDER energy params — NOT measured. "
           "Needs Raspberry Pi + power meter.", font=f_small, fill="#a33")
    d.text((MARGIN["l"] + 10, PLOT_H - 25),
           "Y-axis (miss) = REAL probe scores (n=181 held-out) + session model "
           "(rho/t_visible assumptions). False-wake = REAL (n=235).",
           font=f_small, fill="#a33")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)


def plot_pareto(rows: list[ConfigResult], out_png: Path) -> str:
    """有 matplotlib 用 matplotlib，没有则 PIL 手绘。返回所用后端名。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plot_pareto_pil(rows, out_png)
        return "PIL (matplotlib 未安装；装包需用户批准)"

    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    # 配色：阈值是**连续量级**（不是有中性基准点的极性量）→ 用**单色顺序色阶**（蓝，浅→深）。
    # 不用 coolwarm 这类发散色阶：发散色阶暗示"中间有个中性零点"，阈值没有这种语义。
    SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
    # 蓝色阶 250→700（离散点画在浅底上，最浅一档不低于 250 档以保证可见）
    SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("gk_blue", SEQ)

    fig, ax = plt.subplots(figsize=(11, 6.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    dom = [r for r in rows if not r.is_pareto]
    front = sorted([r for r in rows if r.is_pareto], key=lambda r: r.power_controllable_mw)

    # 被支配的配置：退到背景，只交代"搜索空间长什么样"
    ax.scatter([r.power_controllable_mw for r in dom], [r.miss_rate_session for r in dom],
               s=13, c="#d8d8d4", linewidths=0, zorder=1)
    # 前沿连线压在点之下；点带 2px 表面色描边，重叠时仍能分辨
    ax.plot([r.power_controllable_mw for r in front], [r.miss_rate_session for r in front],
            "-", c=INK_2, lw=2, zorder=2, solid_capstyle="round")
    sc = ax.scatter([r.power_controllable_mw for r in front],
                    [r.miss_rate_session for r in front],
                    s=110, c=[r.threshold for r in front], cmap=cmap,
                    edgecolors=SURFACE, linewidths=2, zorder=3)

    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("Gatekeeper threshold", color=INK_2, fontsize=10)
    cbar.ax.tick_params(labelsize=9, colors=INK_2)
    cbar.outline.set_visible(False)

    ax.set_xscale("log")
    ax.set_xlabel("Controllable power (mW, log)  =  total \u2212 board idle baseline",
                  fontsize=11, color=INK)
    ax.set_ylabel("Miss rate  (session, modeled)", fontsize=11, color=INK)
    ax.set_title("MemoSight Gatekeeper \u2014 Power vs Miss-rate Pareto Front",
                 fontsize=14, color=INK, fontweight="bold", loc="left", pad=18)
    ax.text(0, 1.025, "Recording frequency swept via threshold / sampling fps / debounce "
            f"interval  \u00b7  {len(rows)} configs, {len(front)} on the front",
            transform=ax.transAxes, fontsize=9.5, color=INK_2)

    ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    ax.grid(alpha=0.35, color="#e6e6e2", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d0d0cc")
    ax.tick_params(colors=INK_2, labelsize=9.5)

    # 选择性直标：只标前沿两端 + 中间一个，不给每个点都挂数字
    for r, above in ((front[0], True), (front[len(front) // 2], False), (front[-1], False)):
        ax.annotate(f"thr {r.threshold:.2f} \u00b7 {r.fps:g}fps \u00b7 {r.debounce_s:g}s",
                    (r.power_controllable_mw, r.miss_rate_session),
                    textcoords="offset points", xytext=(10, 7 if above else -15),
                    fontsize=8.5, color=INK_2)

    # 图例：身份不靠颜色单独承载
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="-", c=INK_2, mfc=SEQ[2], mec=SURFACE, mew=1.5,
               ms=9, label="Pareto front (optimal)"),
        Line2D([], [], marker="o", ls="", mfc="#d8d8d4", mec="#d8d8d4", ms=6,
               label="dominated"),
    ], loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK_2)

    # \u26a0\ufe0f 数据真假声明：图被单独截图时也必须跟着走
    fig.subplots_adjust(bottom=0.18, top=0.86)
    fig.text(0.008, 0.055,
             "X-axis (power) = MODEL with PLACEHOLDER energy params \u2014 NOT measured. "
             "Needs Raspberry Pi + power meter.", fontsize=8.5, color="#a33")
    fig.text(0.008, 0.02,
             "Y-axis (miss) = REAL probe scores (n=181 held-out) + session model "
             "(sigma / rho / t_visible assumptions).  False-wake = REAL (n=235).",
             fontsize=8.5, color="#a33")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return "matplotlib"


# ═══════════════════════════ 6. 主流程 ═══════════════════════════

BANNER = """
╔════════════════════════════════════════════════════════════════════════════╗
║  数据真假分层：                                                            ║
║   ✅ 漏报/误唤醒随阈值的变化 = 真实（固定 held-out 探针 181/235，零泄漏）  ║
║   🔶 单帧→整段、去抖影响、场景先验 = 模型（显式假设，待真实录像验证）      ║
║   ⚠️ 功耗绝对值 = 占位参数，**不是实测**，待 Pi + 功耗计（见 docs H1/H6）  ║
╚════════════════════════════════════════════════════════════════════════════╝"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresholds", default="0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    ap.add_argument("--fps", default="0.2,0.5,1,2,3,5")
    ap.add_argument("--debounce", default="0.5,2,5,10,30,60",
                    help="去抖间隔（秒）：两次重处理之间的最小间隔")
    ap.add_argument("--trigger-path", choices=["X", "Y"], default="X",
                    help="重处理走哪条路径（X=本地OCR只传文本 / Y=直接传图）—— 见任务B")
    ap.add_argument("--p-screen", type=float, default=0.15, help="🔶 场景先验（假设）")
    ap.add_argument("--t-visible", type=float, default=20.0, help="🔶 屏幕停留秒数（假设）")
    ap.add_argument("--rho", type=float, default=0.9, help="🔶 帧间相关系数（假设）")
    ap.add_argument("--score-jitter", type=float, default=0.10,
                    help="🔶 σ：同一块屏不同视角下的分数扰动（假设，待录像标定）")
    ap.add_argument("--out-csv", type=Path, default=Path("outputs/pareto_sweep.csv"))
    ap.add_argument("--out-png", type=Path, default=Path("outputs/pareto_power_vs_miss.png"))
    args = ap.parse_args()

    print(BANNER)
    screen, noscreen = load_probe_scores()
    print(f"\n[1/4] 探针（✅ 真实）：person_screen n={len(screen)}（GT=记）、"
          f"person_noscreen n={len(noscreen)}（GT=不记）")
    print("      模型：task1 C_wide_uniform int8（seed42 部署产物，= 会烧进 Pi 的那个文件）")

    params = MockParams()
    e_trig = trigger_energy_mj(params, args.trigger_path)
    energy = EnergyModel(e_trigger_mj=e_trig)
    session = SessionModel(t_visible_s=args.t_visible, rho=args.rho,
                           p_screen=args.p_screen, score_jitter=args.score_jitter)
    print(f"[2/4] 能耗模型（⚠️ 占位）：每次触发 {e_trig:.0f} mJ（路径{args.trigger_path}）、"
          f"每 tick {energy.e_tick_mj:.0f} mJ、本底 {energy.p_base_mw:.0f} mW")

    thresholds = [float(x) for x in args.thresholds.split(",")]
    fps_list = [float(x) for x in args.fps.split(",")]
    deb_list = [float(x) for x in args.debounce.split(",")]
    rows = mark_pareto(sweep(screen, noscreen, energy, session,
                             thresholds, fps_list, deb_list))
    n_front = sum(r.is_pareto for r in rows)
    print(f"[3/4] 扫描 {len(thresholds)}×{len(fps_list)}×{len(deb_list)} = {len(rows)} 个配置，"
          f"其中 {n_front} 个在 Pareto 前沿")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    backend = plot_pareto(rows, args.out_png)
    print(f"[4/4] CSV → {args.out_csv}　PNG → {args.out_png}（后端：{backend}）")

    front = sorted([r for r in rows if r.is_pareto], key=lambda r: r.power_controllable_mw)
    print(f"\nPareto 前沿工作点（按功耗升序，共 {len(front)} 个）")
    print(f"{'阈值':>6} {'fps':>5} {'去抖s':>6} {'可控mW':>8} {'总mW':>8} {'整段漏报':>9} "
          f"{'每帧漏报':>9} {'误唤醒':>8} {'触发/分':>8}")
    for r in front:
        print(f"{r.threshold:>6.2f} {r.fps:>5g} {r.debounce_s:>6g} "
              f"{r.power_controllable_mw:>8.2f} {r.power_mw:>8.1f} "
              f"{r.miss_rate_session*100:>8.1f}% {r.miss_rate_frame*100:>8.1f}% "
              f"{r.false_wake_rate*100:>7.1f}% {r.triggers_per_min:>8.2f}")

    print("\n⚠️ 功耗列是占位模型输出，**不是实测**。曲线的**形状/取舍结构**可信，"
          "**绝对值不可信**。真数字待 Pi + 功耗计。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
