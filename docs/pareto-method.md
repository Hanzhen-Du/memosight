# Method for the power versus missed-capture curve

2026-07-28. Framework: `scripts/pareto_framework.py`. Outputs: `outputs/pareto_sweep.csv`
(396 configurations) and `outputs/pareto_power_vs_miss.png` (an example plot).

**Read this table before reading any number below.**

| | Content | Status |
|---|---|---|
| Measured | How per-frame miss rate and false-wake rate vary with threshold | Real scores from the current gatekeeper on fixed held-out probes: 181 + 235 images, int8 deployment measurement, leakage verified as zero |
| Modelled | Per-frame miss to whole-segment miss, the effect of debounce, the scene prior | The structure is defensible but carries explicit assumptions that need calibrating against real recorded video (H7) |
| Placeholder | The absolute values on the power axis | Not measured. Waiting on a Pi and a power meter (H1/H6). The shape of the curve is trustworthy; the absolute values are not |

## 0. Translating the question

The advisor's question was: someone spends a long time looking at screens but only wants the
important things recorded, so how should the recording frequency be chosen?

"Recording frequency" is product language. Answering it with data means turning it into knobs
that can be adjusted and metrics that can be measured.

Three knobs, all software parameters that need no change to the model:

| Knob | Physical meaning | Effect of increasing it |
|---|---|---|
| `threshold` | How confident the gatekeeper must be that something is worth recording | More conservative: fewer false wakes, more misses |
| `fps` | How many times per second the gatekeeper looks (its duty cycle) | More diligent: fewer misses, higher always-on power |
| `debounce` | Minimum interval between two downstream runs | Cheaper: lower power, but a new screen appearing immediately after one may be missed |

Three metrics:

| Metric | Definition | Why it matters |
|---|---|---|
| miss | Fraction of screens that should have been recorded and were not | The critical one for the product. A miss means that memory is gone for good |
| power | Average power: always-on gatekeeper plus triggers times downstream cost | Determines battery life, which is the premise of an always-on wearable |
| false-wake | Fraction of frames that should not be recorded but woke the expensive downstream stage | Wasted energy, plus junk memory cards |

There is no single optimum, which is exactly why a Pareto curve is the right answer: it gives
the set of configurations that are not dominated, so the operating point becomes an explainable
choice rather than an arbitrary constant.

## 1. How each metric is computed

### Miss rate: two layers, the first of which is real data

Layer one, measured: per-frame miss rate.

Taken directly from the current gatekeeper's real scores on the `person_screen` probe (181
images, ground truth "record"):

```
miss_frame(thr) = fraction of scores below thr = 1 - recall
```

Layer two, modelled: whole-segment miss rate. A screen stays in view for a while, so the
gatekeeper looks at it many times.

The textbook independent-trials model (`miss^k`) is deliberately not used here. It produces an
absurd conclusion: look a few more times and the miss rate approaches zero. Reality does not
work that way, because some screens are missed systematically — the gatekeeper scores them far
below the threshold, and looking a hundred times gets it wrong a hundred times.

Instead the model works from per-screen scores, which is why the probe's per-image scores are
needed rather than an aggregate rate:

```
For screen i, probe score s_i is its intrinsic difficulty.
Viewing angle and shake perturb the score by sigma.
  probability of missing it once   = Phi((thr - s_i) / sigma)
  probability of missing it k_eff times = Phi((thr - s_i) / sigma) ^ k_eff
  whole-segment miss rate          = the mean over all screens
```

`k_eff = 1 + (k - 1)(1 - rho)`, where `k = fps * dwell time`. `rho` is the inter-frame
correlation: consecutive frames are nearly the same image, so errors are highly correlated.
`rho = 0.9` is the conservative assumption, in the sense of preferring to overestimate misses.

As sigma approaches 0 the model degenerates to the per-frame miss rate, which means the miss
rate has a floor — screens scoring far below the threshold are always missed. That is correct
behaviour, and it is where this model beats `miss^k`.

Assumptions, all of which need calibrating against real recorded video (H7): `sigma = 0.10`,
`rho = 0.9`, screen dwell time 20 s, scene prior `p_screen = 0.15`.

### False-wake rate: measured

From real scores on the `person_noscreen` probe (235 images, ground truth "do not record"):
`false_wake(thr) = fraction of scores at or above thr`. This is entirely a software evaluation
and needs no hardware.

### Power: real structure, placeholder parameters

```
average power = board baseline + fps * energy per tick + trigger rate * energy per downstream run
(units: mJ/s == mW, so "events per second * mJ per event" adds directly as mW)

trigger rate = fps * [ p_screen * recall + (1 - p_screen) * false_wake ], then capped by debounce at 1/debounce
```

The trigger-rate formula is what stitches the three metrics together. Lowering the threshold
raises recall (fewer misses) but also raises false_wake, which raises the trigger rate, which
raises power. That is where the trade-off comes from mathematically. It was not designed in by
hand.

The energy of one downstream run is taken directly from the two paths in the power comparison
(`--trigger-path X|Y`), so the two frameworks stay consistent. Once that measurement is real,
this framework's power axis becomes real with it.

## 2. Reading the plot

![Pareto](../outputs/pareto_power_vs_miss.png)

- Each grey point is one (threshold, fps, debounce) configuration. The sweep covers
  11 x 6 x 6 = 396 of them.
- The coloured points joined by the black line are the Pareto frontier: no other configuration
  is at least as good on both power and miss rate while being better on one. Every grey point
  is dominated, so choosing one is simply waste.
- Colour encodes threshold, red at 0.25 (eager to record) through blue at 0.75 (conservative).
- The x axis is logarithmic because controllable power spans two orders of magnitude, from tens
  of mW to thousands.
- The x axis is *controllable* power, meaning total power minus the board baseline — the part
  these three knobs actually govern. Section 4 explains why.

To use it for a decision: fix an acceptable miss rate first, which is a product judgement (say
"no worse than 25%"), draw a horizontal line, and take where it crosses the frontier. That is
the lowest-power configuration at that miss rate. It works the other way too: fix a power
budget from the battery-life requirement, draw a vertical line, and the crossing is the
configuration with the lowest miss rate inside that budget.

Example frontier. The power column is placeholder; the shape is trustworthy, the absolute
values are not.

| Threshold | fps | Debounce | Controllable power mW | Segment miss | Frame miss | False wake | Triggers/min |
|---|---|---|---|---|---|---|---|
| 0.75 | 0.2 | 10s | 23.4 | 80.4% | 81.8% | 5.1% | 0.85 |
| 0.60 | 0.2 | 10s | 46.4 | 68.3% | 71.3% | 12.3% | 1.78 |
| 0.25 | 1 | 30s | 61.5 | 48.0% | 33.1% | 39.6% | 2.00 |
| 0.25 | 0.2 | 10s | 132.2 | 30.1% | 33.1% | 39.6% | 5.24 |
| 0.25 | 2 | 10s | 172.6 | 17.4% | 33.1% | 39.6% | 6.00 |
| 0.25 | 5 | 10s | 208.6 | 11.4% | 33.1% | 39.6% | 6.00 |

Three structural facts fall out of that:

1. Pushing the miss rate from 80% down to 11% costs about 9x the controllable power, 23 to
   209 mW. That is the magnitude of the trade-off.
2. The low-miss end of the frontier is entirely low threshold (0.25) plus high fps. Looking more
   often is better value than lowering the decision standard, because raising fps only costs
   always-on power while lowering the threshold also raises false wakes (to 39.6%) and therefore
   trigger power.
3. False-wake rate climbs from 5% to 40% along the frontier. That is a third dimension and it is
   not on the plot. If the product has no tolerance for junk cards, the right-hand end of the
   frontier is unavailable and the high-threshold end is where to look.

## 3. What is real and what waits for hardware

| Part | Status | Note |
|---|---|---|
| How miss and false-wake vary with threshold | Measured | Real probe scores, computed this round, no hardware needed |
| The direction and magnitude of the threshold trade-off | Measured | For example 39.6% false wake at 0.25, 81.8% miss at 0.75 |
| Per-frame to whole-segment conversion | Modelled | Three assumptions (sigma, rho, dwell time) needing calibration against real video (H7) |
| Scene prior `p_screen` | Assumption | Needs statistics from real worn-camera recordings |
| Absolute power values | Placeholder | Waiting on a Pi and a power meter (H1/H6) |
| Power *structure* (baseline + sampling + triggers) | Real | Energy conservation, independent of the parameters |

In short: the shape of the curve and the structure of the trade-off can be presented now; the
numbers on the x axis have to wait for hardware.

## 4. One finding worth calling out: the baseline swallows the knobs

With the placeholder parameters, the board's baseline draw, on the order of 2600 mW, is an
order of magnitude larger than anything these three knobs control (23 to 209 mW). Total power
moves from 2623 mW to 2809 mW, a change of only 7%. Adjusting the recording frequency has
almost no effect on a Pi 5's overall battery life.

The specific number is a placeholder, but the conclusion is insensitive to it: an idle Pi 5
draws watts while gatekeeper inference draws milliwatts.

This is not a flaw in the framework. It is a real signal about the platform.

- On a Pi 5, the argument that an always-on gatekeeper saves power is essentially void, because
  the baseline consumes everything. The Pi is a prototyping platform, not the product form
  factor.
- The cascade's power argument only holds on a platform whose baseline draw is the same order
  of magnitude as the gatekeeper, meaning something in the ESP32 class, at milliwatts. That is
  the reason the gatekeeper has been kept portable, small and int8-capable from the start.
- This should be stated up front when presenting the result: a Pareto curve measured on a Pi
  will come out flat, and a convincing power curve has to be produced on ESP32-class hardware.
  The value of measuring on the Pi is calibrating the energy coefficient of each action — mJ
  per tick, mJ per OCR run — and those coefficients can be extrapolated to the target platform.

## 5. Honest boundaries

1. This uses the single seed 42 model, not the 5-seed mean. That is deliberate: the seed 42 int8
   file is the deployment artifact that would be flashed onto the Pi, so the Pareto curve should
   describe that one. But it is worth knowing how it compares to the 5-seed mean:

   | | seed 42 (this curve) | 5-seed mean (task1 report) |
   |---|---|---|
   | noscreen FP at 0.40 | 25.1% | 33.1% ± 9.1% |
   | screen recall at 0.40 | 47.5% | 58.2% ± 9.5% |

   Seed 42's score distribution sits low overall, so it is optimistic on false wakes and
   pessimistic on recall relative to the mean. A different seed shifts the curve without
   changing its shape. Do not describe this curve as the model's average behaviour.

2. The probes are posed Pexels photographs, not real frames from a head-mounted camera. Real
   frames are blurrier, more angled and darker, so the numbers will probably be worse.
   Re-measuring on real frames is H7.

3. The whole-segment miss model has not been validated at all. Sigma, rho and dwell time are
   plausible values that were chosen, not measured. A single stretch of real worn-camera video
   with manual "which screens should be recorded" labels would calibrate all three.

4. The debounce model is crude. It only accounts for misses when the debounce window is longer
   than a screen's dwell time, and does not model the case where the content of one screen
   changes (page turns, scrolling) and should be recorded again. Advancing slides is normal in
   practice, so this underestimates the cost of debounce.

5. The 396 configurations are a grid sweep, not continuous optimisation. The true optimum may
   lie between grid points. What this framework gives is a set of candidate operating points,
   not a mathematical optimum.

## 6. What to do once the Pi is available (H1, H6, H7)

1. Calibrate the energy coefficients (H1/H6). Use the differential method from the power
   comparison to measure gatekeeper energy per tick, energy per downstream run, and the board
   baseline. Replace the three placeholder fields in `EnergyModel`; nothing else in this
   framework changes, and the power axis becomes real.
2. Calibrate the session model (H7). Record real worn-camera video, label it, and measure
   average screen dwell time, score perturbation sigma, inter-frame correlation rho and the
   scene prior `p_screen`. Replace the `SessionModel` defaults.
3. Validate the curve. Pick three operating points on the frontier and actually run them on the
   Pi (`hardware/cascade.py --fps N --threshold T`), measure power and miss rate, and check
   whether they land on the predicted curve. This is the real test of whether the curve can be
   trusted.
4. Extrapolate to an ESP32-class platform. Recompute the curve using the calibrated energy
   coefficients and the ESP32's baseline draw (section 4).

## Appendix: commands

```bash
# Sweep and plot. Runnable now; power values are placeholders
.venv/bin/python scripts/pareto_framework.py

# Downstream cost taken from the upload-image path (path Y in the power comparison)
.venv/bin/python scripts/pareto_framework.py --trigger-path Y

# Sensitivity to the session-model assumptions
.venv/bin/python scripts/pareto_framework.py --rho 0.7 --t-visible 10 --score-jitter 0.15

# Re-score the probes, needed when the model changes
PYTHONPATH=scripts .venv/bin/python scripts/probe_fp_test.py \
  --probe-dir data/probe_person_screen \
  --keras-model models/task1_candidates/gatekeeper_task1_C_wide_uniform.keras \
  --int8-model models/task1_candidates/gatekeeper_task1_C_wide_uniform_int8.tflite \
  --out data/processed/probe_person_screen_audit_cwu --no-gradcam
```

On plotting: matplotlib is not installed in this environment, so `pareto_framework.py`
implements two backends — it uses matplotlib when present and falls back to drawing with PIL,
which adds no dependency. The plot above came from the PIL backend. Installing matplotlib
switches backends automatically with no code change.
