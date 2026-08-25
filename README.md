# MemoSight

**A wearable visual memory assistant that turns what you look at into searchable memory cards —
without recording continuously.**

The hard part is not OCR. It is that always-on perception, on-device inference, and all-day
battery life pull against each other. MemoSight's bet is a **cascade**: a tiny always-on
classifier ("the gatekeeper") watches a low-resolution greyscale stream and decides whether a
frame is worth capturing at all. Only when it fires does anything expensive happen.

```
 camera stream (low-res greyscale, ~3 fps)
        │
        ▼
 ┌──────────────────┐   not worth it
 │    GATEKEEPER    │ ─────────────────▶  discard, stay in low-power loop
 │  96x96 int8 CNN  │                     (the common case)
 │     32.4 KB      │
 └──────────────────┘
        │ worth it
        ▼
 full-resolution grab ──▶ OCR ──▶ enrichment ──▶ ┌────────────────┐
                                                 │  memory card   │
        raw frame deleted by default             │  timestamp     │
                                                 │  text          │
                                    offline?     │  tags          │
                                    queue and    └────────────────┘
                                    backfill later      │
                                                        ▼
                                              SQLite · list / search / show
```

The gatekeeper is where the research effort goes. Everything downstream is deliberately
built from mature off-the-shelf parts.

---

## Where it stands

| | |
|---|---|
| **Gatekeeper model** | 96×96 greyscale CNN, 60,882 parameters |
| **Deployed artifact** | 32.4 KB int8 `.tflite` (24.4 KB weight buffer), fully int8, 9 operators — all on the TFLite Micro whitelist |
| **Quantisation cost** | ≤ 0.007 F1 |
| **Memory budget** | 59.5 KB int8 weights, 180 KB peak concurrent activations (targets: 100 KB / 256 KB) |
| **Accuracy** | test F1 **0.769 ± 0.015**, 5 seeds, fixed threshold, zero-leakage splits |
| **Operating point** | 0.33 false-trigger rate on a 235-image held-out probe, 0.58 recall on a 181-image one |
| **Downstream pipeline** | capture → OCR → enrich → SQLite → CLI recall, with an offline queue. 114 tests |
| **Not yet measured** | on-device latency and power. See [honest boundaries](#what-is-measured-and-what-is-not) |

The accuracy number is not good, and the interesting part of this project is *why we can say
that precisely*.

---

## The part worth reading: ruling things out

F1 sat around 0.77 and would not move. Two explanations were obvious. Both were tested, and
both turned out to be wrong.

**H1 — "the model is too small."**
Six architectures × 5 seeds, parameters swept 3.8× from 24.9k to 94.5k.
Test F1 stayed on a **0.736–0.769 plateau**, and the three *largest* models were no better
than the smallest. False-trigger rate stayed pinned at 0.33–0.39 at every capacity.
**Falsified** — 30 training runs.

**H2 — "there isn't enough data."**
249 targeted negatives added, retrained, scored on the same fixed probe.
Primary metric moved **−0.017, well inside ±0.09 noise**, and bought a real regression:
FN 0.187 → 0.262. The new data helped only the distribution it matched — a separate
empty-room probe improved 0.328 → 0.250 — and did not transfer.
**Falsified, and net negative.** The model was not promoted.

**H3 — distribution and representation.**
All 59 false positives on the probe were analysed individually. They cluster by *room*, not
by subject: office conversation 0.559, living room 0.52, meeting room 0.324 — and 0.08
outdoors, 0.0 in restaurants. False positives are brighter (+0.186) and hit a
screen-like-rectangle detector twice as often (+0.118).

And the finding that redirected the project: **face count is negatively correlated with false
triggers (−0.093).** The model is not being fooled by people. It is being fooled by the
*rooms* people are in — which happen to be exactly the rooms that contain monitors,
whiteboards and projectors. At 96×96 greyscale, a blank screen and a text-bearing screen
share almost every feature the network can see; letterforms are already gone at that
resolution.

**Why this matters more than the F1 number.** The conclusion is actionable and negative:
*the next unit of effort should not go into parameter count or image count.* Without the
elimination chain, the default would have been to keep tuning and keep downloading forever.
What has **not** been ruled out is written down too — higher input resolution (untested
because it breaks the microcontroller budget, not because it is unpromising), distribution-matched
negatives, and a two-stage cascade — in
[`docs/gatekeeper-diagnosis.md`](docs/gatekeeper-diagnosis.md) §5.

### A number that had to be retracted

An earlier version of this project reported FN 0.277 and called it a pass. It was not.
Perceptual hashing plus pixel confirmation found **132 near-duplicate pairs across 88 groups,
49 of them spanning train/test/val, 48 of those purely positive-class** — the downloader had
fetched the same stock photo under several keywords, so copies scattered across splits and
inflated recall. Deduplicated and re-split, the same configuration scores **FN 0.337**.

Everything since runs under a fixed protocol: 5 seeds reported as mean ± std with no seed
selection, dedup verified to zero cross-split overlap, thresholds fixed before the run, and
probe sets that never enter training. [`docs/data-leakage-audit.md`](docs/data-leakage-audit.md)

---

## What is measured, and what is not

Portfolio repositories tend to blur this line. This one does not.

**Measured on this project's own data:** every accuracy, F1, FN/FP and probe number; model
size, operator list and quantisation loss (from the actual exported `.tflite`, not a static
estimate); the 114-test suite; the 10-image end-to-end run; the OCR-versus-multimodal
comparison and its dollar cost.

**Modelled, with the assumptions stated inline:** the frame-to-segment miss-rate conversion in
the Pareto sweep.

**Not measured at all:** on-device inference latency and power draw. `hardware/` contains the
Pi cascade loop and a benchmark harness, but **they have not been run on hardware yet**, so no
latency or power figure is quoted anywhere in this repository. The Pareto curve's *shape* is
derived from measured probe scores; its absolute power axis is a placeholder and is labelled
as one. [`docs/pareto-method.md`](docs/pareto-method.md)

One correction is also on the record: the "all operators whitelisted" claim was originally a
static estimate. Real export showed it holds *only* when the batch dimension is pinned to 1 —
dynamic-batch export silently introduces three non-whitelisted operators via `flatten`'s
reshape. The exporter now pins it.

---

## Quick start

```bash
git clone https://github.com/hd2592-sketch/memosight.git && cd memosight
python3 -m venv .venv && . .venv/bin/activate
```

**Try the pipeline** — no camera, no trained model, no API key. The demo synthesises a frame,
runs it through the whole chain and searches the result back out:

```bash
pip install opencv-python numpy python-dotenv pytesseract pillow
python -m pipeline.cli demo --mock-enrich
python -m pipeline.cli list
python -m pipeline.cli search roadmap
```

`--mock-enrich` keeps it fully offline. Without the `tesseract-ocr` system binary installed
the pipeline falls back to a stub OCR and the loop still runs end to end. To ingest your own
image, and to watch the offline queue backfill:

```bash
python -m pipeline.cli ingest path/to/image.png --confidence 0.9
python -m pipeline.cli ingest path/to/image.png --offline
python -m pipeline.cli process-pending
```

```bash
python -m unittest discover -s tests      # 114 tests, offline, no API calls
```

For the real enrichment backends instead of the mock, add `pip install openai anthropic` and
put `DEEPSEEK_API_KEY` / `ANTHROPIC_API_KEY` in a `.env` file. Keys are read from there and
never hardcoded.

**Train and export a gatekeeper.** This is the heavier path — it pulls in TensorFlow, and it
needs the image set, which `scripts/download_images.py` rebuilds from Pexels given a
`PEXELS_API_KEY`:

```bash
pip install -r requirements.txt
python scripts/prepare_dataset.py                       # build manifest
python scripts/dedup_resplit.py build && python scripts/dedup_resplit.py split
python scripts/check_leakage.py --manifest data/processed/manifest_dedup.csv   # must report 0/0
PYTHONPATH=scripts python scripts/run_candidate.py \
    --tag C_wide_uniform --channels 16,32,64,64 --convs 1   # train, eval, export int8, probe
```

`run_candidate.py` is the whole judgement in one command: train across seeds, evaluate at a
fixed threshold, export int8, verify the operator list against the TFLite Micro whitelist,
then score both held-out probes.

## Layout

```
gatekeeper/   what the gatekeeper is and the constraints it is designed against
scripts/      data acquisition, dedup, training, export, evaluation, sweeps
models/       model inventory and provenance (weights are not committed)
pipeline/     capture -> OCR -> enrich -> SQLite -> CLI recall
hardware/     Raspberry Pi cascade loop and inference benchmark
tests/        114 tests, stdlib unittest
docs/         engineering reports — see docs/README.md for the reading order
demo/         self-contained end-to-end run report
data/         not committed; split manifests under data/processed/ are, for reproducibility
```

Model weights and image data stay out of git. The reproducibility contract is
*training script + committed split manifest + fixed seed*.

## Reports

Start with [`docs/gatekeeper-diagnosis.md`](docs/gatekeeper-diagnosis.md) — the elimination
chain above, in full. [`docs/README.md`](docs/README.md) indexes the rest: the leakage audit,
the capacity sweep, both dataset interventions (one that worked, one that did not), the
per-image false-positive analysis, the threshold ledger, and the two measurement-method notes
for power and Pareto.

Each report opens with an English summary; the bodies are in Chinese, as originally written.
