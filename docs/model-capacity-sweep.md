# Capacity sweep: baseline plus five candidate architectures

Completed 2026-06-25. Protocol: 5 seeds `[42,1,7,123,2024]`, threshold 0.40, both probes
scored across 5 seeds on int8, no depthwise-separable variants. Every metric is taken under
ESP32 deployment conditions, meaning the exported int8 model with deployment preprocessing.
Training ran on local CPU; the GPU libraries were not loaded, which does not affect the
numbers.

All metrics here are under the narrowed MVP label definition (`docs/label-scope.md`) and are
not directly comparable with numbers taken under the wider boundary that included ambiguous
negatives.

## 0. Conclusions

Capacity is not the current bottleneck. Sweeping parameters from 24.9k to 94.5k, a factor of
3.8, leaves test F1 hovering on a 0.736–0.769 plateau. Most of the larger candidates (A, B, D,
E) match or fall below baseline, and the differences are generally no larger than one seed's
standard deviation.

The only net gain is `C_wide_uniform` (16,32,64,64): test F1 0.7694 ± 0.0148, which is +0.0207
over baseline or about 1σ. It also has the lowest FN (0.187), the highest val F1 (0.770), the
smallest 5-seed variance (0.015, so the most stable), and the lowest no-screen false-trigger
rate (0.331).

The real bottleneck is the data and the labels rather than model capacity. The evidence is
that test F1 plateaus under a 3.8x parameter sweep, that between-candidate differences are the
same size as seed variance, and that person-driven false triggering on the probe (no-screen FP
0.33–0.39) is not pushed down at any capacity. Adding capacity does not fix covariate-shift
errors.

Budget and portability are clear across the board. All six candidates are fully int8 with zero
float32 tensors, use only TFLite Micro whitelisted operators (5 distinct ones), keep int8
weights at or below 92.3 KB against a 100 KB limit and concurrent activations at or below
180 KB against 256 KB, and lose at most 0.005 F1 to quantisation.

The recommendation is to adopt `C_wide_uniform` as the new gatekeeper architecture (section 4),
while directing the next unit of effort at the data rather than at a larger model.

## 1. Architecture, budget and portability

No seed variance in this table; these are static properties.

| Candidate | channels | convs/stage | Parameters | int8 weights KB | Concurrent activations KB [1] | float32 tensors | Operator whitelist |
|---|---|---|---:|---:|---:|---:|:--:|
| baseline | 8,16,32,64 | 1 | 24,874 | 24.3 | 90 | 0 | Pass |
| A_wide_late | 8,16,64,128 | 1 | 85,290 | 83.3 | 90 | 0 | Pass |
| B_deep_stack | 8,16,32,64 | 2 | 74,314 | 72.6 | 144 | 0 | Pass |
| C_wide_uniform | 16,32,64,64 | 1 | 60,882 | 59.5 | 180 | 0 | Pass |
| D_five_stage | 8,16,32,64,96 | 1 | 80,618 | 78.7 | 90 | 0 | Pass |
| E_combo | 8,16,48,64 | 1,1,2,2 | 94,506 | 92.3 | 90 | 0 | Pass |

[1] Peak concurrent activations come from `estimate_budget.py`, computed statically before
training. Budget ceilings are 256 KB for concurrent activations and 100 KB for int8 weights.

The operator set is identical across candidates and entirely within the TFLite Micro
whitelist: `AVERAGE_POOL_2D, CONV_2D, FULLY_CONNECTED, MAX_POOL_2D, SOFTMAX`. Every candidate
clears its budget with no non-whitelisted operators and no residual float32, so adding
complexity did not cost portability.

## 2. Accuracy

5-seed mean ± std, test at threshold 0.40, int8 deployment measurement.

| Candidate | Parameters | val F1 | test F1 | ΔF1 vs base [2] | test FN | test FP | test recall | test acc |
|---|---:|---|---|---:|---|---|---|---|
| baseline | 24.9k | 0.7611 ± 0.0302 | 0.7487 ± 0.0210 | — | 0.211 ± 0.049 | 0.240 ± 0.041 | 0.789 | 0.772 |
| A_wide_late | 85.3k | 0.7598 ± 0.0050 | 0.7421 ± 0.0232 | −0.0066 | 0.209 ± 0.071 | 0.256 ± 0.060 | 0.791 | 0.764 |
| B_deep_stack | 74.3k | 0.7659 ± 0.0125 | 0.7392 ± 0.0358 | −0.0095 | 0.238 ± 0.086 | 0.223 ± 0.058 | 0.762 | 0.771 |
| C_wide_uniform | 60.9k | 0.7703 ± 0.0197 | 0.7694 ± 0.0148 | +0.0207 | 0.187 ± 0.060 | 0.226 ± 0.054 | 0.813 | 0.791 |
| D_five_stage | 80.6k | 0.7577 ± 0.0300 | 0.7359 ± 0.0199 | −0.0128 | 0.221 ± 0.068 | 0.256 ± 0.093 | 0.779 | 0.759 |
| E_combo | 94.5k | 0.7537 ± 0.0169 | 0.7357 ± 0.0208 | −0.0130 | 0.214 ± 0.077 | 0.263 ± 0.061 | 0.786 | 0.758 |

[2] ΔF1 is the candidate's mean test F1 minus baseline's (0.7487). Only C is positive; the
rest are negative and within noise.

## 3. Both probes, quantisation loss, and value for money

The probes are held-out real photographs, checked for leakage by Pexels ID: no-screen 235
images with 0 leaked, screen 181 images with 0 leaked.

| Candidate | noscreen FP [3], lower better | screen recall [4], higher better | Quantisation ΔF1 (int8 − keras) | Parameter multiple vs base | Verdict |
|---|---|---|---:|---:|---|
| baseline | 0.361 ± 0.053 | 0.625 ± 0.058 | −0.004 | 1.0x | Reference |
| A_wide_late | 0.391 ± 0.097 | 0.645 ± 0.095 | +0.005 | 3.4x | Net negative, no activation increase, 3.4x the weights for nothing |
| B_deep_stack | 0.346 ± 0.081 | 0.600 ± 0.083 | −0.001 | 3.0x | test F1 down, activations double to 144 KB |
| C_wide_uniform | 0.331 ± 0.091 | 0.582 ± 0.095 | −0.001 | 2.4x | The only net gain: +0.021 F1 for 2.4x parameters |
| D_five_stage | 0.375 ± 0.120 | 0.600 ± 0.124 | +0.001 | 3.2x | Among the lowest test F1, high variance |
| E_combo | 0.392 ± 0.108 | 0.635 ± 0.099 | +0.001 | 3.8x | Largest model, worst test F1, close to the weight ceiling |

[3] noscreen FP is the fraction of real "people, no screen" photographs judged as record.
Lower is better; it measures person-driven false triggering.
[4] screen recall is the fraction of real "people plus a useful text screen" photographs judged
as record. Higher is better.

Quantisation loss is within 0.005 F1 for every candidate, so int8 export costs no meaningful
accuracy.

## 4. Recommendation

Adopt `C_wide_uniform` (channels 16,32,64,64, 1 conv per stage, 60.9k parameters) as the new
gatekeeper architecture.

Several metrics agree, which makes this less likely to be a single lucky seed:

- The only candidate that beats baseline on test F1 (0.7694 against 0.7487, +0.0207).
- The smallest 5-seed variance at ±0.0148, so it is not only more accurate but the most stable
  (baseline is ±0.021 and the rest are wider).
- Lowest FN (0.187), highest recall (0.813), highest val F1 (0.770) and lowest no-screen false
  triggering (0.331) — four key metrics leading at once.
- Budget is clear: int8 weights 59.5 KB against 100, concurrent activations 180 KB against 256,
  all operators whitelisted, quantisation loss −0.001 and negligible.
- The seed 42 deployment artifacts are saved at
  `models/task1_candidates/gatekeeper_task1_C_wide_uniform{.keras,_int8.tflite}`.

The caveat, stated plainly: +0.0207 is only about 1σ, not a decisive lead. C also costs
activation memory, rising to 180 KB, the highest of the candidates, because widening the early
layers is expensive at high spatial resolution. That is still inside the 256 KB budget but the
headroom is narrower. Treat C as the best configuration available inside the budget, not as
evidence that complexity solved the accuracy problem.

## 5. Why the bottleneck is data and labels rather than capacity

Sweeping parameters 3.8x, from 24.9k to 94.5k, produced these facts.

1. Test F1 plateaus at 0.736–0.769, and the three largest models (A at 85k, D at 81k, E at 95k)
   are all at or below baseline. If capacity were the constraint, scaling monotonically should
   pay off monotonically. Instead there is a plateau plus noise.
2. Between-candidate differences are about the size of seed variance. Most ΔF1 values, from
   −0.013 to +0.021, fall inside single-seed std of 0.015 to 0.036, which is statistically hard
   to separate from noise. Even C, the only credible positive signal, is about 1σ.
3. Capacity does not fix person-driven false triggering. No-screen FP is stuck at 0.33–0.39 at
   every capacity. That is the covariate-shift error diagnosed during task2 — the training
   distribution under-covers "people, no screen" — and a bigger model cannot repair it.
4. The pattern that widening early layers (C) helps slightly while widening late layers,
   deepening, or adding stages (A, B, D, E) does not, suggests the model is *mildly*
   under-resourced for low-level texture and edge features. That marginal gain is quickly
   absorbed by the data ceiling.

The gatekeeper's accuracy ceiling is set by data and label quality, not by parameter count.
Continuing to stack capacity is poor value. The next investment should go to the data:

- Targeted expansion of "people, no screen" negatives to push down person-driven false
  triggering, since no-screen FP at 0.33–0.39 is the single largest weakness.
- Review coverage of "people plus a useful text screen" positives and screen recall, which is
  low at 0.58–0.65.
- Run subsequent data experiments on `C_wide_uniform`, the most accurate and most stable
  architecture inside the budget, as the new control baseline.

## Appendix: reproducing this

```bash
# Full 5-seed pipeline for one candidate
# (train, test at 0.40, int8 export with whitelist and ΔF1 checks, both probes)
PYTHONPATH=scripts .venv/bin/python scripts/run_candidate.py \
    --tag C_wide_uniform --channels 16,32,64,64 --convs 1
# Result JSON: docs/results/task1_results/<tag>.json
# Models: models/task1_candidates/ (gitignored)
```

Provenance: every number here was measured on this machine across 5 seeds
(`docs/results/task1_results/*.json`), not estimated at design time. The budget columns come
from the static analysis in `scripts/estimate_budget.py`.
