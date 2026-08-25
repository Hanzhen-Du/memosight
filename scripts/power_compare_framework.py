#!/usr/bin/env python3
"""Path X (local OCR) against path Y (upload the image): a framework for measuring on-device
energy per trigger.

Answers the advisor's objection that local OCR is not necessarily cheaper than sending the
image over the network, and that sending it may be faster. The objection is reasonable, and
this framework deliberately presupposes no answer: the two paths spend energy in different
places, and which one wins depends on parameters that have to be measured.

    Path X (local OCR):    gatekeeper fires -> grab full-res frame -> local OCR -> upload text only (~KB)
    Path Y (upload image): gatekeeper fires -> grab full-res frame -> JPEG encode -> upload the whole image (~MB)

    E_X = E_capture + E_ocr          + E_tx(text_bytes)
    E_Y = E_capture + E_jpeg_encode  + E_tx(image_bytes)
                      |- X pays here -|  |- Y pays here -|

`E_capture` occurs on both paths and is identical, so it cancels in the comparison. It is still
measured and recorded, because it determines the absolute cost of one trigger and feeds the
power axis of the Pareto curve.

No real power data exists as of 2026-07-28. The Raspberry Pi is unreachable and the power meter
is not wired up. The default `MockPowerMeter` produces synthetic numbers from a documented
parameter table. Those are not measurements and must not be cited as any kind of conclusion.
Their only purpose is to exercise the whole pipeline and demonstrate that once a meter is
attached, replacing one PowerMeter class is enough to produce real data. The outstanding
measurements are H1 to H5 in `docs/power-measurement-method.md`.

Measurement method: the differential method.
    First measure the system's idle power P_idle, with the gatekeeper running and no trigger.
    Then measure power P_active while a segment of work executes. The incremental energy of
    that segment is (P_active - P_idle) * t.
    This measures how much extra energy the work itself cost, with the board's baseline draw
    stripped out. Paths X and Y share the same board baseline, so only the increment is the
    real difference between them.

Usage:
    # Simulated run, currently the only runnable mode
    .venv/bin/python scripts/power_compare_framework.py --trials 30 --meter mock

    # Once the Pi and the meter are ready
    python3 scripts/power_compare_framework.py --trials 30 --meter ina219

    # Crossover analysis only: at what bandwidth and image size is path X genuinely cheaper
    .venv/bin/python scripts/power_compare_framework.py --crossover-only
"""

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

# ============================= 1. power meter interface =============================


class PowerMeter:
    """Abstract power meter. A real implementation only needs to override `read_power_mw()`.

    Contract: `read_power_mw()` returns instantaneous power in milliwatts, and the call should
    be cheap (under 1 ms), because `measure()` polls it at high rate during a task to
    integrate.
    """

    name = "abstract"
    is_real = False  # real implementations must set this True. The CSV and the reports rely on
                     # this field to tell measured data from synthetic

    def read_power_mw(self) -> float:
        raise NotImplementedError

    def measure(self, fn, poll_hz: float = 200.0) -> tuple[float, float, float]:
        """Run fn(), polling power throughout and integrating.

        Returns (mean power mW, elapsed s, energy mJ), where energy is mean power times
        elapsed time (mW * s = mJ).
        Note this is absolute energy; the idle baseline has not been subtracted yet. The
        subtraction happens in `Trial`.
        """
        samples: list[float] = []
        t0 = time.perf_counter()
        # Sampling concurrently with the task: on real hardware, use a separate sampling
        # thread or the meter's own high-speed buffer. Here the integration is simplified to
        # before, after, and a callback during the task; in mock mode fn itself drives the
        # sampling.
        result = fn(lambda: samples.append(self.read_power_mw()))
        dt = time.perf_counter() - t0
        if not samples:  # fn did not call back; fall back to the endpoints
            samples = [self.read_power_mw()]
        p_mean = sum(samples) / len(samples)
        return p_mean, dt, p_mean * dt * 1000.0  # mW * s -> mJ (x1000 because dt is in seconds)

    def idle_baseline_mw(self, seconds: float = 2.0) -> float:
        """Measure the idle baseline: gatekeeper running, no trigger. The subtrahend in the
        differential method."""
        raise NotImplementedError


@dataclass
class MockParams:
    """Placeholder parameter table. Every value is a plausible order-of-magnitude assumption,
    not a measurement.

    Once real hardware is available this whole table is void and is replaced by readings from
    `INA219PowerMeter`. It is written as an explicit dataclass rather than magic numbers
    scattered through the code so that it is obvious at a glance which numbers are fabricated.

    Basis for the magnitudes, used only to keep the simulation from being absurd and not to be
    cited: an idle Pi 5 draws several watts, WiFi transmission draws more than idle, and
    single-frame Tesseract OCR on a Pi takes hundreds of milliseconds to about a second.
    """
    p_idle_mw: float = 2800.0          # baseline: board plus the always-on gatekeeper
    p_ocr_mw: float = 4200.0           # total power during OCR (CPU saturated)
    p_encode_mw: float = 3300.0        # total power during JPEG encoding
    p_capture_mw: float = 3600.0       # total power while the camera grabs a full-res frame
    p_radio_tx_mw: float = 3900.0      # total power during WiFi transmission
    ocr_ms_mean: float = 850.0         # single-frame OCR duration
    ocr_ms_jitter: float = 220.0
    encode_ms_mean: float = 45.0       # JPEG encoding duration
    encode_ms_jitter: float = 10.0
    capture_ms_mean: float = 120.0     # full-res frame grab duration
    capture_ms_jitter: float = 25.0
    radio_wake_ms: float = 180.0       # fixed radio wake-up and association cost per transmission
    throughput_kbps: float = 6000.0    # effective throughput including protocol overhead;
                                       # measured WiFi is usually far below the nominal rate
    throughput_jitter: float = 1800.0  # network variability, one of the reasons the answer
                                       # depends on measurement
    image_bytes_mean: float = 900_000  # order of magnitude for a 1920x1080 JPEG at q85
    image_bytes_jitter: float = 250_000
    text_bytes_mean: float = 1_400     # OCR output for one screenful of text
    text_bytes_jitter: float = 700


class MockPowerMeter(PowerMeter):
    """Simulated power meter. Produces synthetic readings, not measurements. For exercising
    the pipeline only."""

    name = "mock"
    is_real = False

    def __init__(self, params: MockParams, seed: int = 42):
        self.p = params
        self.rng = random.Random(seed)
        self._level_mw = params.p_idle_mw  # power level implied by whatever is running now

    def set_level(self, mw: float) -> None:
        self._level_mw = mw

    def read_power_mw(self) -> float:
        # Add reading noise so the integral is not constant; a real meter is noisy too
        return self._level_mw + self.rng.gauss(0, self._level_mw * 0.01)

    def idle_baseline_mw(self, seconds: float = 2.0) -> float:
        self.set_level(self.p.p_idle_mw)
        return sum(self.read_power_mw() for _ in range(64)) / 64


class INA219PowerMeter(PowerMeter):
    """Placeholder for the real power meter, pending hardware (see
    `docs/power-measurement-method.md`, H1 to H5).

    Wiring, for whoever implements this: an INA219 high-side in the Pi 5's 5 V supply loop,
    with I2C going to the Pi or to a second board. Reading from a second board is cleaner,
    since it keeps the sampling overhead off the board under test, but it needs time
    synchronisation.

    Implementation checklist:
      1. Open I2C (smbus2 or adafruit-circuitpython-ina219) and set the shunt resistor and
         range.
      2. `read_power_mw()` returns bus_voltage * current, or reads the power register directly.
      3. `idle_baseline_mw()` samples for at least 2 s in a window where the gatekeeper is
         running and no trigger can occur.
      4. Calibrate: verify the readings against a known resistive load before attaching the
         system under test.
      5. Sample at 200 Hz or above, otherwise short tasks integrate inaccurately — JPEG
         encoding takes only tens of milliseconds.
    """

    name = "ina219"
    is_real = True

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "INA219PowerMeter is not implemented yet. It can only be written and verified "
            "once a Pi and a power meter are connected.\n"
            "This build provides the framework only; use --meter mock for a simulated run.\n"
            "The outstanding work is H1 to H5 in docs/power-measurement-method.md.")


METERS = {"mock": MockPowerMeter, "ina219": INA219PowerMeter}


# ============================= 2. the two paths =============================


@dataclass
class Trial:
    """The complete record of one trigger. One row of the CSV."""
    trial: int
    path: str                  # "X_local_ocr" | "Y_send_image"
    meter: str                 # meter type
    is_real: bool              # False means simulated data and must not be cited
    p_idle_mw: float           # the subtrahend in the differential method
    capture_ms: float
    compute_ms: float          # OCR duration for X, JPEG encode duration for Y
    tx_bytes: int
    tx_ms: float               # includes the radio wake-up cost
    e_capture_mj: float        # from here on, all energies are incremental over idle
    e_compute_mj: float
    e_tx_mj: float
    e_total_mj: float
    latency_ms: float          # end-to-end latency. This is the column that tests the
                               # claim that uploading the image is faster
    notes: str = ""


def _jitter(rng: random.Random, mean: float, spread: float, lo: float = 0.0) -> float:
    return max(lo, rng.gauss(mean, spread / 2.0))


def _run_segment(meter: PowerMeter, level_mw: float, duration_ms: float,
                 p_idle_mw: float) -> tuple[float, float]:
    """Run one energy-consuming segment and return (actual duration ms, incremental energy
    over idle in mJ).

    On real hardware the `level_mw` argument is ignored, because actual power comes from the
    meter rather than from us. In simulated mode it drives MockPowerMeter's level.
    """
    if isinstance(meter, MockPowerMeter):
        meter.set_level(level_mw)

    def work(sample):
        # Simulated mode does not actually sleep for the whole segment, since 30 triggers
        # would take tens of seconds. Instead it samples a number of points proportional to
        # the segment length and integrates. On real hardware this becomes the actual task call.
        n = max(4, int(duration_ms / 5))
        for _ in range(n):
            sample()
        return None

    p_mean, _, _ = meter.measure(work)
    dt_s = duration_ms / 1000.0
    e_incremental_mj = (p_mean - p_idle_mw) * dt_s   # the differential method, in one line
    return duration_ms, e_incremental_mj


def run_path_x(meter: PowerMeter, p: MockParams, rng: random.Random,
               p_idle_mw: float, trial: int) -> Trial:
    """Path X: local OCR, text upload only. Expensive to compute, cheap to transmit."""
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
    """Path Y: upload the full-resolution image. Almost no computation, expensive to
    transmit."""
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


# ============================= 3. crossover analysis =============================


def crossover_analysis(p: MockParams) -> list[dict]:
    """Under what conditions is path X (local OCR) genuinely cheaper? An analytical answer,
    with no power meter needed.

    Ignoring capture, which both paths share, and assuming the same transmission power for
    both:
        E_X − E_Y = (P_ocr−P_idle)·t_ocr − (P_enc−P_idle)·t_enc
                    + (P_tx−P_idle)·(t_txX − t_txY)
    where t_txY - t_txX is about (image_bytes - text_bytes) * 8 / throughput.
    Setting the difference to zero gives the break-even throughput: below it, transmission is
    too slow and too expensive and X wins; above it, the network is cheap and Y wins.

    The structure of this function is real (energy conservation plus a linear transmission
    model); only the parameters substituted into it are placeholders. Once P_ocr, t_ocr and
    P_tx are measured on hardware, the same function gives the real crossover.
    """
    dp_ocr = p.p_ocr_mw - p.p_idle_mw
    dp_enc = p.p_encode_mw - p.p_idle_mw
    dp_tx = p.p_radio_tx_mw - p.p_idle_mw
    d_bytes = p.image_bytes_mean - p.text_bytes_mean
    # Extra energy (mJ) and time (ms) that X spends on computation relative to Y
    e_compute_penalty = (dp_ocr * p.ocr_ms_mean - dp_enc * p.encode_ms_mean) / 1000.0
    t_compute_penalty = p.ocr_ms_mean - p.encode_ms_mean
    rows = []
    for tput_kbps in (500, 1000, 2000, 4000, 6000, 8000, 12000, 25000, 50000):
        d_tx_ms = (d_bytes * 8 / 1000.0) / tput_kbps * 1000.0
        e_tx_saving = dp_tx * d_tx_ms / 1000.0     # energy X saves on transmission (mJ)
        net_e = e_tx_saving - e_compute_penalty    # > 0 means path X uses less energy
        net_t = d_tx_ms - t_compute_penalty        # > 0 means path X is faster
        rows.append({
            "throughput_kbps": tput_kbps,
            "tx_time_saved_ms": round(d_tx_ms, 1),
            "e_tx_saved_mj": round(e_tx_saving, 1),
            "e_compute_penalty_mj": round(e_compute_penalty, 1),
            "net_mj_x_minus_y": round(-net_e, 1),   # negative means X uses less energy
            "winner_energy": "X_local_ocr" if net_e > 0 else "Y_send_image",
            "winner_latency": "X_local_ocr" if net_t > 0 else "Y_send_image",
        })
    # Break-even throughput. Saving energy and saving time are two different equations, and
    # their crossovers generally do not coincide.
    #   energy:  dp_tx * delta_t_tx == e_compute_penalty
    #   latency: delta_t_tx         == t_compute_penalty
    be_energy = ((dp_tx * (d_bytes * 8 / 1000.0)) / e_compute_penalty
                 if e_compute_penalty > 0 else float("inf"))
    be_latency = ((d_bytes * 8 / 1000.0) / (t_compute_penalty / 1000.0)
                  if t_compute_penalty > 0 else float("inf"))
    return rows, be_energy, be_latency


# ============================= 4. main =============================


BANNER_MOCK = """
+---------------------------------------------------------------------------+
| SIMULATED MODE (--meter mock). Every energy number below is synthetic.    |
| Not measured, and not to be cited as a conclusion. Real figures need a Pi |
| and a power meter (see docs). The only purpose of this run is to show     |
| that the framework executes and the CSV structure is usable.              |
+---------------------------------------------------------------------------+"""


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
    ap.add_argument("--trials", type=int, default=30, help="how many triggers to run per path")
    ap.add_argument("--meter", choices=list(METERS), default="mock")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/power_compare_trials.csv"))
    ap.add_argument("--crossover-only", action="store_true",
                    help="crossover analysis only; purely analytical, no power meter needed")
    args = ap.parse_args()

    params = MockParams()
    rng = random.Random(args.seed)

    if args.crossover_only:
        rows, be_e, be_t = crossover_analysis(params)
        print("\nCrossover analysis. The formula structure is real; the substituted parameters "
              "are placeholders pending measurement.")
        print(f"{'kbps':>10} {'tx saved ms':>12} {'e saved mJ':>12} "
              f"{'ocr cost mJ':>11} {'net X-Y mJ':>11}  {'lower energy':<13} {'lower latency':<13}")
        for r in rows:
            print(f"{r['throughput_kbps']:>10} {r['tx_time_saved_ms']:>12.1f} "
                  f"{r['e_tx_saved_mj']:>12.1f} {r['e_compute_penalty_mj']:>11.1f} "
                  f"{r['net_mj_x_minus_y']:>11.1f}  {r['winner_energy']:<13} "
                  f"{r['winner_latency']:<13}")
        print(f"\nBreak-even throughput (energy) is about {be_e:,.0f} kbps. "
              "Above this, uploading the image uses less energy.")
        print(f"Break-even throughput (latency) is about {be_t:,.0f} kbps. "
              "Above this, uploading the image is faster.")
        if be_t > be_e:
            print(f"Note the contradictory band between {be_e:,.0f} and {be_t:,.0f} kbps: "
                  "uploading the image uses less energy while local OCR is still faster. "
                  "Saving energy and saving time are not the same question.")
        print("These figures come from placeholder parameters and are not conclusions. "
              "Re-run this once P_ocr, t_ocr and P_tx have been measured on hardware.")
        return 0

    try:
        meter = METERS[args.meter](params, seed=args.seed)
    except NotImplementedError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 2

    if not meter.is_real:
        print(BANNER_MOCK)

    p_idle = meter.idle_baseline_mw()
    print(f"\n[1/3] Idle baseline power (the differential subtrahend): {p_idle:.1f} mW  "
          f"[{'measured' if meter.is_real else 'simulated'}]")

    print(f"[2/3] Running {args.trials} triggers across 2 paths...")
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
    print(f"[3/3] Wrote {len(trials)} rows to {args.out}")

    s = summarize(trials)
    print(f"\n{'path':<14} {'total mJ':>11} {'compute mJ':>10} {'tx mJ':>10} "
          f"{'tx bytes':>11} {'latency ms':>10}")
    for path, v in s.items():
        print(f"{path:<14} {v['e_total_mj']:>11.1f} {v['e_compute_mj']:>10.1f} "
              f"{v['e_tx_mj']:>10.1f} {v['tx_bytes']:>11,.0f} {v['latency_ms']:>10.1f}")

    if len(s) == 2:
        dx = s["X_local_ocr"]["e_total_mj"] - s["Y_send_image"]["e_total_mj"]
        dl = s["X_local_ocr"]["latency_ms"] - s["Y_send_image"]["latency_ms"]
        verdict = ("path X, local OCR, uses less energy" if dx < 0
                   else "path Y, upload image, uses less energy")
        lat = ("path X is faster" if dl < 0
               else "path Y is faster, so the advisor's intuition holds under these parameters")
        print(f"\nEnergy difference X-Y = {dx:+.1f} mJ per trigger -> {verdict}")
        print(f"Latency difference X-Y = {dl:+.1f} ms per trigger -> {lat}")

    if not meter.is_real:
        print("\nTo repeat: everything above is simulated, not a measured conclusion. The "
              "framework runs; real data needs a Pi and a power meter (see "
              "docs/power-measurement-method.md, H1 to H5).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
