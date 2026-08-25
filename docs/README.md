# docs/ — engineering reports

Working reports from the gatekeeper research line and the downstream pipeline. Each report
opens with an English summary; the body is in Chinese, as originally written.

Every number in these documents is measured on this project's own data. Estimates, models
and placeholders are labelled as such inline — most importantly the power axis in
`pareto-method.md`, which is **not** yet instrumented.

## Read in this order

| # | Document | What it establishes |
|---|---|---|
| 1 | [`gatekeeper-diagnosis.md`](gatekeeper-diagnosis.md) | **Start here.** The elimination chain: capacity ruled out, data volume ruled out, bottleneck localised to representation + distribution. |
| 2 | [`data-leakage-audit.md`](data-leakage-audit.md) | Why the first set of numbers was retracted; how leakage was found and removed. |
| 3 | [`label-scope.md`](label-scope.md) | The positive/negative label definition, and the one time it was deliberately narrowed. |
| 4 | [`model-capacity-sweep.md`](model-capacity-sweep.md) | 6 architectures × 5 seeds, 24.9k → 94.5k parameters. Capacity hypothesis, falsified. |
| 5 | [`dataset-expansion.md`](dataset-expansion.md) | The data intervention that worked: person-driven false triggers halved, Pareto-dominant. |
| 6 | [`targeted-negatives-retrain.md`](targeted-negatives-retrain.md) | The data intervention that did not. Negative result, reported in full. |
| 7 | [`false-positive-diagnosis.md`](false-positive-diagnosis.md) | Per-image analysis of 59 false positives — the measurement that redirected the project. |
| 8 | [`threshold-tradeoff.md`](threshold-tradeoff.md) | Per-threshold FN/FP ledger for choosing the deployment operating point. |
| 9 | [`gatekeeper-training-log.md`](gatekeeper-training-log.md) | Chronological training log, including a correction notice about operator-whitelist claims. |
| 10 | [`pipeline-architecture.md`](pipeline-architecture.md) | Downstream capture → OCR → enrich → SQLite → recall, and its three swappable interfaces. |
| 11 | [`power-measurement-method.md`](power-measurement-method.md) | How to compare local-OCR vs upload-image on energy rather than byte count. |
| 12 | [`pareto-method.md`](pareto-method.md) | Method for the power-versus-missed-capture curve. Read its provenance table first. |

## Supporting data

- `results/` — machine-readable metrics behind the reports (per-seed JSON, sweep CSVs, leakage candidate lists).
- `probes/` — raw per-probe evaluation dumps, before/after each intervention.
