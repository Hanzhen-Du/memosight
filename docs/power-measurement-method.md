# Measuring whether local OCR actually saves power

2026-07-28. Framework code: `scripts/power_compare_framework.py`.

The advisor's objection was that local OCR is not necessarily cheaper than uploading the image,
and that uploading may well be faster. This document sets out how to decide that with
measurement rather than assertion.

**This file contains no measured power figures.** The Raspberry Pi is currently unreachable and
the power meter is not wired up. What this round delivers is the measurement framework, the
method, and a simulated run proving the framework works. Every energy value that appears here
was synthesised from a placeholder parameter table (`MockParams`), for one purpose only: to
show the pipeline runs end to end, that the CSV structure is usable, and that swapping in a
real meter means replacing a single class. These numbers must not go into a report and must not
be treated as conclusions. The real measurements still to be taken are H1 to H5 in section 6.

## 0. The objection is correct, and more so than it first appears

"Local OCR saves power" is an assumption this project has relied on from the start and never
tested. It may be wrong, for concrete reasons:

- Local OCR is several hundred milliseconds of saturated CPU, which is expensive computation on
  an edge device.
- Uploading an image moves many more bytes, but WiFi transmission is high power for a short
  time. On a fast network that window may be shorter than the OCR.
- The upload path barely computes at all: one JPEG encode, a few tens of milliseconds.

So this framework does not presuppose an answer. It splits both paths into separately
measurable components and lets the measurements decide, and it works out explicitly under what
conditions each side wins rather than giving a single blanket answer.

There is also a distinction worth drawing out: saving power and saving time are two different
questions, and they can have different answers (section 4). The advisor's remark actually
contains two independent claims, and they need separate answers.

## 1. What each path spends energy on

```
Path X (local OCR):    gatekeeper fires -> grab full-res frame -> local OCR -> upload text only (~1.4 KB)
Path Y (upload image): gatekeeper fires -> grab full-res frame -> JPEG encode -> upload whole image (~900 KB)

E_X = E_capture + E_ocr          + E_tx(text_bytes)
E_Y = E_capture + E_jpeg_encode  + E_tx(image_bytes)
      |-- same --|  |- X pays here -|  |------ Y pays here ------|
```

`E_capture` is identical on both paths and cancels in the comparison. Only two terms actually
differ:

| | Path X | Path Y | Character |
|---|---|---|---|
| Compute energy | OCR, several hundred ms of saturated CPU | JPEG encode, tens of ms | Roughly fixed, does not vary with the network |
| Transmission energy | ~1.4 KB | ~900 KB | Varies sharply with image size and network conditions |

That is what makes this a genuine trade-off rather than a foregone conclusion: one term is
fixed and the other floats, and their relative size depends on network throughput, an external
variable we do not control. On a train and on office WiFi the answer may be opposite.

The transmission energy model implemented in the framework:

```
t_tx = t_radio_wake + (bytes * 8) / throughput
E_tx = (P_radio_tx - P_idle) * t_tx
```

`t_radio_wake` is a fixed radio wake-up and association cost paid on every transmission,
independent of byte count. That term is particularly unkind to path X, which still pays a full
wake-up to send 1.4 KB and has nothing to amortise it over. Comparing on byte count alone gets
this wrong.

## 2. Method: differential measurement

You cannot read "how much energy the OCR used" directly, because the meter reads the power of
the whole board, which includes the system's idle draw.

```
1) Measure the idle baseline: gatekeeper running, in a window guaranteed to have no trigger,
   sampled for at least 2 s  ->  P_idle
2) Measure during the task: run that segment, poll power at high rate and integrate
   ->  P_active, t
3) Incremental energy for the segment = (P_active - P_idle) * t
```

The subtraction is not optional. Path X and path Y run on the same board and share the same
idle draw, so only the increment above the baseline is the real difference between them.
Comparing total power lets the baseline dilute the difference and makes the two paths look
similar, which is a measurement artefact rather than a finding.

Sampling rate must be at least 200 Hz. JPEG encoding takes only tens of milliseconds, and
sampling too slowly integrates short tasks inaccurately.

Wiring plan, not yet implemented: an INA219 high-side on the Pi 5's 5 V supply, read over I2C.
Reading from a second board is cleaner, since it keeps the sampling overhead off the board
under test, at the cost of needing time synchronisation.

## 3. What gets recorded

One row per trigger, written to `outputs/power_compare_trials.csv`.

| Field | Meaning |
|---|---|
| `trial` / `path` | Trigger number / `X_local_ocr` or `Y_send_image` |
| `meter` / `is_real` | Meter type / whether this is a real measurement. `is_real=False` means simulated data and must not be cited |
| `p_idle_mw` | The idle baseline for this trial, the subtrahend in the differential method |
| `capture_ms` | Time to grab the full-resolution frame, common to both paths |
| `compute_ms` | OCR time for X, JPEG encode time for Y |
| `tx_bytes` / `tx_ms` | Bytes transmitted / transmission time, including radio wake-up |
| `e_capture_mj` / `e_compute_mj` / `e_tx_mj` | Incremental energy per component, in mJ |
| `e_total_mj` | Total incremental on-device energy per trigger, in mJ. The primary metric |
| `latency_ms` | End-to-end latency. This is the column that tests the "uploading is faster" claim |

Components are recorded separately rather than just a total so the question "which segment is
expensive" can be answered later, not only "which path is expensive".

## 4. Why this is a real trade-off: crossover analysis

The framework includes an analytical mode that needs no power meter (`--crossover-only`).
Setting `E_X - E_Y = 0` and solving gives the break-even throughput.

Saving power and saving time are different equations, and their crossovers do not coincide:

```
Energy break-even:  (P_tx - P_idle) * dt_tx  ==  the extra energy OCR costs over encoding
Latency break-even:              dt_tx       ==  the extra time OCR costs over encoding
```

The table below is computed from placeholder parameters. The values mean nothing. What means
something is the structure it shows.

| Throughput | Lower energy | Lower latency |
|---|---|---|
| 500 – 6,000 kbps | X, local OCR | X, local OCR |
| 6,773 – 8,930 kbps | Y, upload image | X, local OCR (the contradictory band) |
| Above 12,000 kbps | Y, upload image | Y, upload image |

Three structural conclusions that do not depend on the specific parameter values:

1. A break-even throughput necessarily exists. The faster the network, the better uploading
   looks; the slower the network, the better local OCR looks. "Local OCR saves power" therefore
   cannot be an unconditional claim. It is conditional, and the condition is the network.
2. There is a band where uploading uses less energy while local OCR is still faster. The two
   claims in the advisor's remark, that it may not save power and that uploading is faster, can
   both be true, or one can be true and the other false. They have to be measured and answered
   separately.
3. Larger images, slower networks and faster OCR all push toward local OCR. That is an
   actionable lever: if measurement favours uploading, the conclusion can still be changed by
   lowering upload resolution or quality — at the cost of downstream OCR quality, which is a
   different trade-off curve.

## 5. Simulated run: the framework works

```bash
.venv/bin/python scripts/power_compare_framework.py --trials 30 --meter mock
.venv/bin/python scripts/power_compare_framework.py --crossover-only
```

Evidence it runs: 30 triggers across 2 paths produced 60 rows of structured CSV with per
component energy, byte counts and latency all present; the differential logic works end to end
(idle sampling, per-segment integration, baseline subtraction); and the crossover analysis
returns both break-even points.

The simulated output saying something like "path X saves 203.7 mJ per trigger" is a product of
the placeholder parameters, not a finding. A different but equally plausible set of placeholders
flips it — which is exactly why this has to be measured for real.

## 6. Swapping in real hardware

Everything synthetic in the framework is confined to `MockPowerMeter` and `MockParams`. The real
version is:

```python
class INA219PowerMeter(PowerMeter):
    name, is_real = "ina219", True
    def read_power_mw(self): ...       # read the INA219 power register
    def idle_baseline_mw(self, s=2.0): ...
```

Nothing else changes — the differential method, the segmentation, the CSV and the crossover
analysis all stay as they are. Once `is_real=True`, the CSV's `is_real` column becomes true and
the numbers may be cited.

Measurements to take once that is in place, H1 to H5:

| # | What | How |
|---|---|---|
| H1 | Gatekeeper always-on baseline, P_idle | Sample for at least 2 s in a no-trigger window, repeat 5 times, take the stable value |
| H2 | Path X OCR compute energy | Run Tesseract on a real single frame, differential method |
| H3 | Path X text transmission energy | Really transmit about 1.4 KB, including radio wake-up |
| H4 | Path Y image transmission energy | Really transmit about 900 KB, including radio wake-up |
| H5 | End-to-end latency for both paths | The `latency_ms` column of the same CSV |

Protocol requirements, so the numbers mean something:

- Alternate the paths (X, Y, X, Y, ...). Do not run 30 X trials and then 30 Y trials, or board
  temperature and network drift get folded into the path difference.
- At least 30 triggers per path, reported as median and interquartile range. Network latency is
  long-tailed and the mean is dragged by a few extreme values.
- Record the network conditions, measuring throughput at the same time. The conclusion depends
  on them, and without that record the result cannot be interpreted.
- Run a round under two network conditions, good WiFi and weak signal, to test directly whether
  the crossover in section 4 really exists.

## 7. What this framework does not answer

The answer depends on measurements not yet taken. The value of the framework is not that it
gives an answer, but that it turns a vague argument — "surely local processing saves power?" —
into a question with a definite crossover that data can settle.

Factors explicitly outside its scope, which should be stated when discussing the result so the
power conclusion is not over-extended:

1. Privacy. Uploading the image sends the raw frame off the device; uploading text sends only
   the OCR result. Local-first is the founding reason for this project, so even if uploading
   wins on power that does not automatically mean the architecture should change. Power is one
   dimension, not the only one.
2. Offline availability. Path Y is completely unusable without a network. Path X can still
   produce a memory card and queue it for later (`pipeline/` already implements the pending
   queue).
3. Cloud OCR may be better. Sending the image allows OCR far stronger than Tesseract. That is a
   real advantage for path Y, unrelated to power, and it will affect the final choice.
4. This framework measures only per-trigger energy. It does not include the continuous draw of
   the always-on gatekeeper, which is what the task C Pareto curve computes
   (`docs/pareto-method.md`). Both share the same energy model.

## Appendix: commands

```bash
# Simulated run, currently the only runnable mode. The numbers are not real
.venv/bin/python scripts/power_compare_framework.py --trials 30 --meter mock

# Crossover analysis, purely analytical, no meter required
.venv/bin/python scripts/power_compare_framework.py --crossover-only

# Once the Pi and meter are ready. Today this errors out and points at the to-do list
python3 scripts/power_compare_framework.py --trials 30 --meter ina219
```
