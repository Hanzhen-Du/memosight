#!/usr/bin/env python3
"""Pareto curve framework: recording frequency against power, missed captures and false wakes.

Answers advisor question 3: someone spends a long time looking at screens but only wants the
important things recorded, so how should the recording frequency be chosen? The point is to
turn "recording frequency", a vague product knob, into three measurable quantities, sweep the
trade-off, and make the operating point a choice rather than a guess.

    Recording frequency is controlled by three knobs:
      1. Gatekeeper trigger threshold. Lower means more eager to record: fewer misses, more
         false wakes.
      2. Gatekeeper sampling rate, fps. Higher means less likely to miss something: fewer
         misses, higher always-on power.
      3. Debounce interval. The minimum gap between two downstream runs: lower power, but a
         new screen appearing immediately afterwards may be missed.

    Three metrics:
      1. miss rate      a screen that should have been recorded was not
      2. power          average power: always-on gatekeeper plus triggers times downstream cost
      3. false-wake     a frame that should not be recorded woke the expensive downstream stage

Provenance of the numbers. This is the most important thing about this framework and must be
read before any result:

  Measured, needing no hardware, computed this round:
     - How per-frame miss rate and false-wake rate vary with threshold, taken from the current
       gatekeeper's real scores on fixed held-out probes (person_screen 181 images with ground
       truth record, person_noscreen 235 with ground truth do-not-record, int8 deployment
       measurement, leakage verified as zero).

  Modelled, defensible in structure but carrying explicit assumptions that need validating
  against real recorded video:
     - per-frame miss converted to whole-segment miss (repeated viewing, highly correlated
       frames)
     - the extra misses caused by debounce
     - the scene prior, meaning what fraction of the time a recordable screen is actually
       present

  Placeholder, not measured and not to be cited, pending a Pi and a power meter:
     - the absolute values on the power axis: gatekeeper energy per tick, downstream energy per
       trigger, and the board's baseline draw.
       These reuse `power_compare_framework.MockParams`, so once the real measurement lands this
       framework produces real values automatically.

Usage:
    .venv/bin/python scripts/pareto_framework.py                   # sweep and plot
    .venv/bin/python scripts/pareto_framework.py --trigger-path Y  # downstream via the upload-image path
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
from power_compare_framework import MockParams  # noqa: E402  same placeholder energy parameters

# ----------------------- probes: the source of the measured software evaluation -----------------------
PROBE_SCREEN = Path("data/processed/probe_person_screen_audit_cwu/probe_scores.csv")
PROBE_NOSCREEN = Path("data/processed/probe_person_noscreen_audit_cwu/probe_scores.csv")
SCORE_COL = "p_record_int8"   # int8 deployment measurement, the same numeric pipeline the device runs


# ======================= 1. energy model (placeholder parameters) =======================


@dataclass
class EnergyModel:
    """Every value here is a placeholder, to be replaced once measured on hardware
    (see `docs/pareto-method.md`, H1 and H6).

    The structure is real: energy equals the always-on baseline, plus sampling rate times
    per-tick cost, plus trigger rate times per-downstream-run cost. Only the numbers
    substituted into it are fabricated.
    """
    p_base_mw: float = 2600.0       # board baseline, excluding the gatekeeper
    e_tick_mj: float = 12.0         # per tick: low-resolution grab plus int8 gatekeeper inference
    e_trigger_mj: float = 0.0       # downstream energy per trigger, set by --trigger-path below

    def avg_power_mw(self, fps: float, triggers_per_s: float) -> float:
        """Average power = baseline + sampling power + trigger power.

        Units: mJ/s == mW, so (events per second) x (mJ per event) is already mW and the terms
        add directly.
        """
        return (self.p_base_mw
                + fps * self.e_tick_mj
                + triggers_per_s * self.e_trigger_mj)


def trigger_energy_mj(params: MockParams, path: str) -> float:
    """Downstream energy per trigger, in mJ, consistent with the two paths in the power
    comparison.

    Placeholder values. These reference the same parameters as `power_compare_framework` so the
    two frameworks cannot disagree: once the power comparison is measured for real, this
    framework follows automatically.
    """
    dp_cap = params.p_capture_mw - params.p_idle_mw
    dp_ocr = params.p_ocr_mw - params.p_idle_mw
    dp_enc = params.p_encode_mw - params.p_idle_mw
    dp_tx = params.p_radio_tx_mw - params.p_idle_mw
    e_cap = dp_cap * params.capture_ms_mean / 1000.0

    def e_tx(nbytes: float) -> float:
        t_ms = params.radio_wake_ms + (nbytes * 8 / 1000.0) / params.throughput_kbps * 1000.0
        return dp_tx * t_ms / 1000.0

    if path == "X":   # local OCR, text upload only
        return e_cap + dp_ocr * params.ocr_ms_mean / 1000.0 + e_tx(params.text_bytes_mean)
    if path == "Y":   # upload the whole image
        return e_cap + dp_enc * params.encode_ms_mean / 1000.0 + e_tx(params.image_bytes_mean)
    raise ValueError(f"unknown downstream path: {path} (expected X or Y)")


# =================== 2. probes to per-frame miss and false wake (measured) ===================


def load_probe_scores() -> tuple[pd.Series, pd.Series]:
    """Load the real scores from both fixed held-out probes, produced by the current
    gatekeeper's int8 deployment artifact."""
    missing = [p for p in (PROBE_SCREEN, PROBE_NOSCREEN) if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing probe score files:\n  " + "\n  ".join(str(m) for m in missing) +
            "\nRe-score them with the current gatekeeper first:\n"
            "  PYTHONPATH=scripts .venv/bin/python scripts/probe_fp_test.py \\\n"
            "    --probe-dir data/probe_person_screen \\\n"
            "    --keras-model models/task1_candidates/gatekeeper_task1_C_wide_uniform.keras \\\n"
            "    --int8-model models/task1_candidates/gatekeeper_task1_C_wide_uniform_int8.tflite \\\n"
            "    --out data/processed/probe_person_screen_audit_cwu --no-gradcam")
    return (pd.read_csv(PROBE_SCREEN)[SCORE_COL],
            pd.read_csv(PROBE_NOSCREEN)[SCORE_COL])


def frame_metrics(screen: pd.Series, noscreen: pd.Series, thr: float) -> tuple[float, float]:
    """Measured: per-frame miss rate and false-wake rate at a given threshold.

    miss_frame = fraction of should-record screens judged do-not-record in a single frame,
                 which is 1 - recall
    false_wake = fraction of should-not-record frames judged record, which is the FP rate
    """
    recall = float((screen >= thr).mean())
    false_wake = float((noscreen >= thr).mean())
    return 1.0 - recall, false_wake


# =================== 3. per-frame to whole-segment (modelled, explicit assumptions) ===================


def _phi(z: float) -> float:
    """Standard normal CDF, implemented with the stdlib to avoid pulling in scipy."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class SessionModel:
    """Modelled: converts per-frame miss rate into whole-segment miss rate. The assumptions
    need validating against real recorded video (see `docs/pareto-method.md`, H7).

    The independent-trials model (session_miss = miss_frame^k) is deliberately not used. It
    produces an absurd conclusion: look a few more times and the miss rate approaches zero.
    Reality does not work that way, because some screens are missed systematically. The model
    scores them far below the threshold, so a hundred looks produce a hundred wrong answers.

    This model works from per-screen scores rather than an aggregate miss rate:
        probe score s_i for screen i represents its intrinsic difficulty; viewing angle and
        shake perturb the score by sigma.
        probability of missing it in one look   = Phi((thr - s_i) / sigma)
        probability of missing it k_eff times   = Phi((thr - s_i) / sigma) ^ k_eff
        whole-segment miss rate                 = the mean over all screens
    As sigma approaches 0 this degenerates to the per-frame miss rate, meaning screens scoring
    far below the threshold are always missed. That floor is correct behaviour.
    """

    t_visible_s: float = 20.0   # how long a recordable screen stays in view (assumption)
    rho: float = 0.9            # inter-frame correlation: consecutive frames see the same
                                # screen, so errors are highly correlated (assumption)
    p_screen: float = 0.15      # scene prior: what fraction of the time a recordable screen is
                                # actually in view (assumption)
    score_jitter: float = 0.10  # sigma: score perturbation for one screen across viewing
                                # angles and shake (assumption, to be calibrated from video)

    def effective_looks(self, fps: float) -> float:
        """Effective number of independent looks.

        There are k = fps * t_visible looks, but consecutive frames are nearly the same image,
        so errors are highly correlated and they cannot be treated as k independent trials —
        that would badly underestimate the miss rate. Discount by the correlation coefficient:
            k_eff = 1 + (k − 1)(1 − ρ)
        rho=1 gives k_eff=1 (perfectly correlated, extra looks are useless); rho=0 gives
        k_eff=k (fully independent, the textbook case). The default rho=0.9 is the conservative
        assumption, preferring to overestimate misses.
        """
        k = max(1.0, fps * self.t_visible_s)
        return 1.0 + (k - 1.0) * (1.0 - self.rho)

    def session_miss(self, screen_scores, thr: float, fps: float,
                     debounce_s: float, triggers_per_s: float) -> float:
        """Whole-segment miss rate: missed on every look, plus blocked by debounce."""
        k_eff = self.effective_looks(fps)
        sigma = max(1e-6, self.score_jitter)
        miss_multi = float(
            sum(_phi((thr - s) / sigma) ** k_eff for s in screen_scores)
            / len(screen_scores))
        # Debounce only loses a whole screen when the debounce window outlasts the screen's
        # dwell time. When D <= t_visible the window expires while the screen is still there,
        # so there is no extra miss.
        if debounce_s > self.t_visible_s:
            shadow = min(1.0, triggers_per_s * debounce_s)          # fraction of time inside the debounce shadow
            p_debounce_miss = shadow * (1.0 - self.t_visible_s / debounce_s)
        else:
            p_debounce_miss = 0.0
        return miss_multi + (1.0 - miss_multi) * p_debounce_miss


# =========================== 4. sweep ===========================


@dataclass
class ConfigResult:
    threshold: float
    fps: float
    debounce_s: float
    miss_rate_frame: float       # measured on the probes
    false_wake_rate: float       # measured on the probes
    miss_rate_session: float     # modelled
    triggers_per_min: float      # modelled, includes the scene prior
    power_mw: float              # placeholder: energy model with mock parameters
    power_controllable_mw: float # placeholder; power minus the board baseline, meaning the
                                 # part these knobs actually control
    battery_hours_1000mah: float # placeholder; for intuition only, not a specification
    is_pareto: bool = False
    data_status: str = "miss/false_wake=REAL_probe; power=MOCK_placeholder"


def sweep(screen, noscreen, energy: EnergyModel, session: SessionModel,
          thresholds, fps_list, debounce_list) -> list[ConfigResult]:
    rows: list[ConfigResult] = []
    for thr in thresholds:
        miss_f, fwake = frame_metrics(screen, noscreen, thr)
        recall = 1.0 - miss_f
        for fps in fps_list:
            # Per-tick trigger probability = correct triggers when a screen is present plus
            # false triggers when it is not. The scene prior is an assumption
            p_trigger = session.p_screen * recall + (1 - session.p_screen) * fwake
            raw_tps = fps * p_trigger
            for deb in debounce_list:
                tps = min(raw_tps, 1.0 / deb) if deb > 0 else raw_tps  # capped by debounce
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
    """Mark the Pareto frontier, minimising power and whole-segment miss rate together.

    A configuration is on the frontier when no other configuration is at least as good on both
    metrics and better on at least one.
    """
    for a in rows:
        a.is_pareto = not any(
            (b.power_mw <= a.power_mw and b.miss_rate_session <= a.miss_rate_session)
            and (b.power_mw < a.power_mw or b.miss_rate_session < a.miss_rate_session)
            for b in rows)
    # Deduplicate ties: many configurations give identical metrics, for instance whenever the
    # debounce interval is shorter than the trigger interval and debounce never engages.
    # Keep one representative per (power, miss) pair, choosing the largest debounce interval —
    # when the metrics are identical, a longer debounce is strictly easier on the system (fewer
    # wake spikes, steadier under bursts), so it is a free choice.
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


# =========================== 5. plotting ===========================
# matplotlib is not installed in this venv, and adding a dependency needs approval, so there
# are two backends: use matplotlib when it is available, otherwise draw with PIL, which adds
# nothing. Installing matplotlib makes this function take the matplotlib branch automatically,
# with no code change.

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
    """Threshold to colour: low threshold means eager to record (red), high means conservative
    (blue)."""
    f = (t - tmin) / (tmax - tmin) if tmax > tmin else 0.5
    return (int(220 - 170 * f), int(70 + 60 * f), int(60 + 170 * f))


def plot_pareto_pil(rows: list[ConfigResult], out_png: Path) -> None:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (PLOT_W, PLOT_H), "white")
    d = ImageDraw.Draw(img)
    x0, y0 = MARGIN["l"], PLOT_H - MARGIN["b"]
    x1, y1 = PLOT_W - MARGIN["r"], MARGIN["t"]

    # Power spans two orders of magnitude, tens of mW to thousands, so the x axis is
    # logarithmic. Otherwise the whole frontier collapses into a line on the left.
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

    # Grid and ticks: powers of ten on the x axis, with 2 and 5 as intermediate ticks
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

    # Non-frontier points, in light grey
    for r in rows:
        if not r.is_pareto:
            x, y = px(r.power_controllable_mw), py(r.miss_rate_session)
            d.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill="#cccccc")

    # Pareto frontier: connecting line plus coloured points
    front = sorted([r for r in rows if r.is_pareto], key=lambda r: r.power_controllable_mw)
    if len(front) > 1:
        d.line([(px(r.power_controllable_mw), py(r.miss_rate_session)) for r in front],
               fill="#222222", width=2)
    for r in front:
        x, y = px(r.power_controllable_mw), py(r.miss_rate_session)
        c = _thr_color(r.threshold, tmin, tmax)
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=c, outline="white", width=2)

    # Title and axis labels
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

    # Legend
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

    # Label a few frontier operating points; labelling all of them turns into mush
    if front:
        for r in (front[0], front[len(front) // 2], front[-1]):
            x, y = px(r.power_controllable_mw), py(r.miss_rate_session)
            d.text((x + 10, y - 14),
                   f"thr{r.threshold:.2f}/{r.fps:g}fps/{r.debounce_s:g}s",
                   font=f_small, fill="#333")

    # Provenance notice. This has to stay on the plot, so a screenshot of it alone cannot be
    # cited as measured data
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
    """Use matplotlib when available, otherwise draw with PIL. Returns the backend name."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plot_pareto_pil(rows, out_png)
        return "PIL (matplotlib not installed; installing it needs approval)"

    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    # Colour: threshold is a continuous magnitude, not a polar quantity with a neutral
    # midpoint, so use a single-hue sequential scale (blue, light to dark). A diverging scale
    # like coolwarm would imply a neutral zero in the middle, which threshold does not have.
    SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
    # Blue scale 250 to 700. Discrete points sit on a light background, so the lightest step
    # stays at 250 or above to remain visible
    SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("gk_blue", SEQ)

    fig, ax = plt.subplots(figsize=(11, 6.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    dom = [r for r in rows if not r.is_pareto]
    front = sorted([r for r in rows if r.is_pareto], key=lambda r: r.power_controllable_mw)

    # Dominated configurations recede into the background; they only convey the shape of the
    # search space
    ax.scatter([r.power_controllable_mw for r in dom], [r.miss_rate_session for r in dom],
               s=13, c="#d8d8d4", linewidths=0, zorder=1)
    # The frontier line sits under the points; each point gets a 2px surface-coloured outline
    # so overlapping points stay distinguishable
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

    # Selective direct labelling: both ends of the frontier plus one in the middle, rather
    # than a number on every point
    for r, above in ((front[0], True), (front[len(front) // 2], False), (front[-1], False)):
        ax.annotate(f"thr {r.threshold:.2f} \u00b7 {r.fps:g}fps \u00b7 {r.debounce_s:g}s",
                    (r.power_controllable_mw, r.miss_rate_session),
                    textcoords="offset points", xytext=(10, 7 if above else -15),
                    fontsize=8.5, color=INK_2)

    # Legend: identity is not carried by colour alone
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="-", c=INK_2, mfc=SEQ[2], mec=SURFACE, mew=1.5,
               ms=9, label="Pareto front (optimal)"),
        Line2D([], [], marker="o", ls="", mfc="#d8d8d4", mec="#d8d8d4", ms=6,
               label="dominated"),
    ], loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK_2)

    # Provenance notice, which must travel with the plot if it is screenshotted alone
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


# =========================== 6. main ===========================

BANNER = """
+----------------------------------------------------------------------------+
| Provenance of the numbers below:                                           |
|   MEASURED    miss and false-wake against threshold, from fixed held-out   |
|               probes (181/235 images, zero leakage)                        |
|   MODELLED    per-frame to whole-segment, debounce effect, scene prior     |
|               (explicit assumptions, pending validation on real video)     |
|   PLACEHOLDER absolute power values, NOT measured, pending a Pi and a      |
|               power meter (see docs, H1/H6)                                |
+----------------------------------------------------------------------------+"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresholds", default="0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    ap.add_argument("--fps", default="0.2,0.5,1,2,3,5")
    ap.add_argument("--debounce", default="0.5,2,5,10,30,60",
                    help="debounce interval in seconds: the minimum gap between two downstream runs")
    ap.add_argument("--trigger-path", choices=["X", "Y"], default="X",
                    help="which downstream path to cost (X = local OCR, text upload only; Y = upload the whole image)")
    ap.add_argument("--p-screen", type=float, default=0.15, help="modelled: scene prior (assumption)")
    ap.add_argument("--t-visible", type=float, default=20.0, help="modelled: screen dwell time in seconds (assumption)")
    ap.add_argument("--rho", type=float, default=0.9, help="modelled: inter-frame correlation coefficient (assumption)")
    ap.add_argument("--score-jitter", type=float, default=0.10,
                    help="modelled: sigma, score perturbation for one screen across viewing angles "
                         "(assumption, to be calibrated from video)")
    ap.add_argument("--out-csv", type=Path, default=Path("outputs/pareto_sweep.csv"))
    ap.add_argument("--out-png", type=Path, default=Path("outputs/pareto_power_vs_miss.png"))
    args = ap.parse_args()

    print(BANNER)
    screen, noscreen = load_probe_scores()
    print(f"\n[1/4] Probes (measured): person_screen n={len(screen)} (ground truth record), "
          f"person_noscreen n={len(noscreen)} (ground truth do-not-record)")
    print("      Model: task1 C_wide_uniform int8 (the seed 42 deployment artifact, "
          "the same file that would be flashed to the Pi)")

    params = MockParams()
    e_trig = trigger_energy_mj(params, args.trigger_path)
    energy = EnergyModel(e_trigger_mj=e_trig)
    session = SessionModel(t_visible_s=args.t_visible, rho=args.rho,
                           p_screen=args.p_screen, score_jitter=args.score_jitter)
    print(f"[2/4] Energy model (placeholder): {e_trig:.0f} mJ per trigger "
          f"(path {args.trigger_path}), {energy.e_tick_mj:.0f} mJ per tick, "
          f"{energy.p_base_mw:.0f} mW baseline")

    thresholds = [float(x) for x in args.thresholds.split(",")]
    fps_list = [float(x) for x in args.fps.split(",")]
    deb_list = [float(x) for x in args.debounce.split(",")]
    rows = mark_pareto(sweep(screen, noscreen, energy, session,
                             thresholds, fps_list, deb_list))
    n_front = sum(r.is_pareto for r in rows)
    print(f"[3/4] Swept {len(thresholds)}x{len(fps_list)}x{len(deb_list)} = {len(rows)} "
          f"configurations, {n_front} of them on the Pareto frontier")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    backend = plot_pareto(rows, args.out_png)
    print(f"[4/4] CSV -> {args.out_csv}   PNG -> {args.out_png} (backend: {backend})")

    front = sorted([r for r in rows if r.is_pareto], key=lambda r: r.power_controllable_mw)
    print(f"\nPareto frontier operating points, by ascending power ({len(front)} total)")
    print(f"{'thresh':>6} {'fps':>5} {'deb s':>6} {'ctrl mW':>8} {'tot mW':>8} {'seg miss':>9} "
          f"{'frm miss':>9} {'falsewk':>8} {'trig/min':>8}")
    for r in front:
        print(f"{r.threshold:>6.2f} {r.fps:>5g} {r.debounce_s:>6g} "
              f"{r.power_controllable_mw:>8.2f} {r.power_mw:>8.1f} "
              f"{r.miss_rate_session*100:>8.1f}% {r.miss_rate_frame*100:>8.1f}% "
              f"{r.false_wake_rate*100:>7.1f}% {r.triggers_per_min:>8.2f}")

    print("\nThe power columns are placeholder model output, NOT measurements. The shape of "
          "the curve and the structure of the trade-off are trustworthy; the absolute values "
          "are not. Real numbers need a Pi and a power meter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
