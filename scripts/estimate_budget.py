#!/usr/bin/env python3
"""Budget estimation for task 1 candidates: compute parameter count, peak activation, int8
weight size and operator set before training, and skip anything over budget without training it.

Hard ESP32 constraints, which are the elimination line: peak activation under 256 KB, int8
weights under 100 KB, and only TFLite Micro whitelisted operators.

For each candidate build_model configuration this script:
  - computes int8 activation per layer (H x W x C bytes) and reports both the single-layer peak
    and the concurrent peak (one layer's input plus output, which is closer to the TFLM arena);
  - estimates int8 weights as count_params x 1 byte, the same measure export_tflite.py uses. The
    baseline's 24,874 parameters give 24.3 KB, which matched the real export;
  - checks the operator set statically: build_model uses only Conv2D, BatchNorm (folded), ReLU,
    MaxPool, AveragePool, Reshape, Dense and Softmax, all whitelisted. The authoritative check
    is still the real .tflite measured by export_tflite.py, since task 2 was caught out by
    dynamic operators;
  - prints PASS or SKIP. Any budget exceeded means SKIP and the candidate is not trained.

It neither trains nor exports; this is purely static accounting. Depends on tensorflow.

Example: .venv/bin/python scripts/estimate_budget.py

"""

from __future__ import annotations

import json

import tensorflow as tf

from model import build_model

ACT_BUDGET_KB = 256
W_BUDGET_KB = 100
BYTES_INT8 = 1

# Candidates: (tag, description, channels, convs_per_stage)
CANDIDATES = [
    ("baseline", "baseline task2_mvp architecture", (8, 16, 32, 64), 1),
    ("A_wide_late", "widen late channels; early stages untouched, so activations stay cheap", (8, 16, 64, 128), 1),
    ("B_deep_stack", "two Convs per stage, deeper throughout", (8, 16, 32, 64), 2),
    ("C_wide_uniform", "widen uniformly, including block1 to 16, which costs activation memory", (16, 32, 64, 64), 1),
    ("D_five_stage", "one more stage, deeper downsampling and abstraction", (8, 16, 32, 64, 96), 1),
    ("E_combo", "widen and deepen late, combined; approaches the weight ceiling", (8, 16, 48, 64), (1, 1, 2, 2)),
]

# Layer types that still occupy their own arena tensor after int8 .tflite folding. BatchNorm
# folds into Conv and ReLU becomes Conv's fused activation, so neither leaves a separate
# tensor. Activation accounting therefore counts only the tensor-producing layers below, plus
# the input.
_TENSOR_LAYER_TYPES = (
    tf.keras.layers.InputLayer,
    tf.keras.layers.Conv2D,
    tf.keras.layers.MaxPooling2D,
    tf.keras.layers.AveragePooling2D,
    tf.keras.layers.Reshape,
    tf.keras.layers.Dense,
    tf.keras.layers.Softmax,
)


def _elems(shape) -> int:
    n = 1
    for d in shape[1:]:
        if d is not None:
            n *= int(d)
    return n


def analyze(model: tf.keras.Model) -> dict:
    # The sequence of tensor-producing layers in the folded graph, where BN and ReLU leave no
    # separate tensor. These are the tensors the ESP32 TFLM arena actually holds.
    seq = [(l.name, tuple(l.output.shape), _elems(l.output.shape))
           for l in model.layers if isinstance(l, _TENSOR_LAYER_TYPES)]
    single_peak_kb, single_at = 0.0, ""
    for name, _, e in seq:
        kb = e * BYTES_INT8 / 1024
        if kb > single_peak_kb:
            single_peak_kb, single_at = kb, name
    concur_peak_kb, concur_at = 0.0, ""
    for (n_in, _, e_in), (n_out, _, e_out) in zip(seq, seq[1:]):
        kb = (e_in + e_out) * BYTES_INT8 / 1024
        if kb > concur_peak_kb:
            concur_peak_kb, concur_at = kb, f"{n_in}->{n_out}"
    params = model.count_params()
    final_spatial = None
    for name, shp, _ in seq:
        if name == "gap_avgpool":
            # gap's input is the previous layer, so take the spatial size of the last conv or
            # pool before flatten
            pass
    # Spatial dimension entering gap at the tail: find gap_avgpool's input shape
    gap = next(l for l in model.layers if l.name == "gap_avgpool")
    in_hw = int(gap.input.shape[1])
    return {
        "params": params,
        "int8_weight_kb": round(params / 1024, 1),
        "single_peak_kb": round(single_peak_kb, 1),
        "single_at": single_at,
        "concur_peak_kb": round(concur_peak_kb, 1),
        "concur_at": concur_at,
        "pre_gap_spatial": in_hw,
        "n_layers": len(seq),
    }


def main() -> int:
    rows = []
    for tag, desc, channels, convs in CANDIDATES:
        m = build_model(96, bn_momentum=0.9, channels=channels, convs_per_stage=convs,
                        name=f"gk_{tag}")
        a = analyze(m)
        act_peak = max(a["single_peak_kb"], a["concur_peak_kb"])
        act_ok = a["concur_peak_kb"] < ACT_BUDGET_KB
        w_ok = a["int8_weight_kb"] < W_BUDGET_KB
        verdict = "PASS" if (act_ok and w_ok) else "SKIP (over budget)"
        rows.append({"tag": tag, "desc": desc, "channels": list(channels),
                     "convs_per_stage": convs if isinstance(convs, int) else list(convs),
                     **a, "act_peak_kb": act_peak, "verdict": verdict})

    print(f"{'candidate':<16}{'channels':<22}{'conv/stg':<10}{'params':>9}"
          f"{'int8 KB':>11}{'peak KB':>10}{'concur KB':>10}{'verdict':>18}")
    print("-" * 104)
    for r in rows:
        cps = r["convs_per_stage"]
        print(f"{r['tag']:<16}{str(r['channels']):<22}{str(cps):<10}{r['params']:>9,}"
              f"{r['int8_weight_kb']:>11}{r['single_peak_kb']:>10}{r['concur_peak_kb']:>10}"
              f"{r['verdict']:>14}")
    print("-" * 104)
    print(f"budget: concurrent activation < {ACT_BUDGET_KB} KB, int8 weights < {W_BUDGET_KB} KB, "
          "TFLM whitelist only")
    print("concurrent peak is one layer's input plus output, which is closer to what the ESP32 "
          "TFLM tensor arena actually holds; the single-layer peak is for reference")
    base = rows[0]
    print(f"\nbaseline reference: {base['params']:,} params, {base['int8_weight_kb']} KB int8 "
          f"weights, {base['concur_peak_kb']} KB concurrent activation at {base['concur_at']}")
    print("\nBUDGET_JSON " + json.dumps(rows, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
