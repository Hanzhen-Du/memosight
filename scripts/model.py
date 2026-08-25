#!/usr/bin/env python3
"""The gatekeeper CNN: model definition plus per-layer activation memory accounting, which is
the hard gate for ESP32 portability.

Model (binary gatekeeper: record = 1, do not record = 0):
  input 96x96x1 greyscale
  4 blocks: Conv(3x3, same) + BatchNorm + ReLU + MaxPool(2x2)
            channels 8, 16, 32, 64; spatial 96 -> 48 -> 24 -> 12 -> 6
  tail: AveragePooling2D over the whole 6x6 -> Reshape(64) -> Dense(2) -> Softmax

On the choice of global average pooling, which affects TFLM portability. The obvious
implementation is GlobalAveragePooling2D, but Keras GAP usually exports to TFLite as a MEAN
(reduce_mean) operator, and MEAN is not on the TFLite Micro whitelist (which covers Conv2D,
DepthwiseConv2D, AveragePool2D, MaxPool2D, Reshape, FullyConnected and Softmax). To stay inside
the whitelist, this averages the whole 6x6 feature map with AveragePooling2D, which is
equivalent to global average pooling, then flattens with Reshape. All three operators are
whitelisted, the semantics are the same, and the export is predictable.

Correction after export verification, 2026-06-18. The claim above that everything is
whitelisted was originally a static operator count made while modelling, never verified by an
actual export. The first real .tflite export showed that with the default dynamic batch (-1),
flatten's Reshape assembles its shape at runtime via SHAPE, STRIDED_SLICE and PACK, none of
which are whitelisted. The whitelist guarantee holds only when the model is exported with
batch_shape=(1,96,96,1) pinned, so every shape is static and the Reshape collapses to a
constant. The export script `scripts/export_tflite.py` now pins batch 1 and checks the operator
table automatically.

On BatchNorm: it is used during training only. When exporting to int8 TFLite, Conv with
use_bias=False followed by BN is folded into the preceding Conv by the converter, so no
separate BN operator survives.

Activation memory accounting, at 1 byte per element under int8: for each layer's output feature
map, compute H*W*C bytes, print a table and mark the peak layer. A peak above 256 KB is
immediately over budget — the script returns non-zero and prints STOP rather than settling for
an estimate.

Dependencies: tensorflow==2.19.* (see requirements.txt).

Example:
  python scripts/model.py            # build the model and print the memory table
"""

from __future__ import annotations

import argparse
import sys

import tensorflow as tf

# Hard ESP32 peak activation memory budget, in KB. Exceeding it stops the run.
ACT_MEM_BUDGET_KB = 256

# Bytes per activation element after int8 quantisation.
BYTES_PER_ELEM_INT8 = 1


def build_model(
    size: int = 96,
    bn_momentum: float = 0.99,
    channels: tuple[int, ...] = (8, 16, 32, 64),
    convs_per_stage: int | tuple[int, ...] = 1,
    name: str = "gatekeeper_v1",
) -> tf.keras.Model:
    """Build the gatekeeper CNN with the functional API, which keeps every layer's output
    shape concrete and therefore accountable for memory.

    bn_momentum is the momentum of BatchNorm's moving averages. The default of 0.99 converges
    too slowly for short training runs, leaving inference statistics inaccurate and causing the
    train/inference mismatch. Lowering it (0.9 or 0.8) lets the moving statistics keep up. This
    is a training hyperparameter and does not change the network structure; BN is still folded
    into Conv at export.

    Complexity knobs, used in task1 to push accuracy inside the ESP32 budget:
      - channels: channel count per stage; its length is the number of stages. The default
        (8,16,32,64) reproduces the baseline. Widening an early stage, at 96x96 spatial
        resolution, is very expensive in activation memory; widening a late stage is very
        cheap. See the accounting section.
      - convs_per_stage: how many Conv+BN+ReLU to stack before the pool in each stage, as an
        int or a tuple the same length as channels. The default of 1 reproduces the baseline,
        and the per-stage layer names match the baseline exactly
        (block{i}_conv, _bn, _relu, _pool), so a baseline .keras can be retrained or loaded
        without changes.

    Structural rule, to keep the TFLM whitelist intact: the tail is always
    AveragePooling2D over the full HxW, then Reshape, Dense(2) and Softmax. No GAP, MEAN or
    other non-whitelisted operator is introduced, and the operator types of Conv, Pool and BN
    never change. Only channel counts, depth and stage count vary.
    """
    n_stages = len(channels)
    if isinstance(convs_per_stage, int):
        convs_per_stage = (convs_per_stage,) * n_stages
    if len(convs_per_stage) != n_stages:
        raise ValueError(
            f"convs_per_stage has length {len(convs_per_stage)}, which must equal the "
            f"channels length {n_stages}"
        )

    inputs = tf.keras.Input(shape=(size, size, 1), name="input")
    x = inputs
    for i, (ch, nconv) in enumerate(zip(channels, convs_per_stage), start=1):
        for j in range(nconv):
            # With nconv == 1 the layer names match the baseline exactly, with no numeric
            # suffix, so old weights stay loadable and results reproducible.
            suffix = "" if nconv == 1 else str(j + 1)
            # Conv without bias: BN follows and its beta acts as the bias, which makes the
            # Conv+BN fold cleanest.
            x = tf.keras.layers.Conv2D(
                ch, 3, padding="same", use_bias=False, name=f"block{i}_conv{suffix}"
            )(x)
            x = tf.keras.layers.BatchNormalization(
                momentum=bn_momentum, name=f"block{i}_bn{suffix}"
            )(x)
            x = tf.keras.layers.ReLU(name=f"block{i}_relu{suffix}")(x)
        x = tf.keras.layers.MaxPooling2D(2, name=f"block{i}_pool")(x)

    # TFLM-safe global average pooling: AveragePool2D over the remaining HxW, then Reshape to
    # flatten.
    pooled_hw = x.shape[1]  # 6 in the baseline, smaller with more stages
    x = tf.keras.layers.AveragePooling2D(pool_size=pooled_hw, name="gap_avgpool")(x)
    x = tf.keras.layers.Reshape((x.shape[-1],), name="flatten")(x)
    x = tf.keras.layers.Dense(2, name="logits")(x)
    outputs = tf.keras.layers.Softmax(name="softmax")(x)
    return tf.keras.Model(inputs, outputs, name=name)


def _elems(shape) -> int:
    """Product of the non-batch dimensions; shape[0] is the batch of None and is skipped."""
    n = 1
    for d in shape[1:]:
        if d is not None:
            n *= int(d)
    return n


def activation_memory_report(model: tf.keras.Model) -> tuple[float, str]:
    """Print the int8 activation memory of each layer's output feature map, and return
    (peak KB, peak layer name)."""
    print(f"\n{'layer':<16}{'output shape':<22}{'activation KB':>14}")
    print("-" * 52)
    peak_kb = 0.0
    peak_layer = ""
    # Collect (layer name, output shape, element count) for the concurrent-usage estimate below.
    seq: list[tuple[str, tuple, int]] = []
    for layer in model.layers:
        out_shape = tuple(layer.output.shape)
        elems = _elems(out_shape)
        kb = elems * BYTES_PER_ELEM_INT8 / 1024
        seq.append((layer.name, out_shape, elems))
        marker = ""
        if kb > peak_kb:
            peak_kb = kb
            peak_layer = layer.name
        print(f"{layer.name:<16}{str(out_shape):<22}{kb:>14.2f}")
    # Mark the peak
    print("-" * 52)
    print(f"peak single-layer output: {peak_kb:.2f} KB at {peak_layer}")

    # A closer proxy for the TFLM tensor arena on ESP32: while a sequential network executes
    # one layer, it holds that layer's input and output tensors at the same time.
    concur_peak = 0.0
    concur_at = ""
    for (n_in, _, e_in), (n_out, _, e_out) in zip(seq, seq[1:]):
        kb = (e_in + e_out) * BYTES_PER_ELEM_INT8 / 1024
        if kb > concur_peak:
            concur_peak = kb
            concur_at = f"{n_in}→{n_out}"
    print(
        f"reference (closer to the ESP32 arena; input and output of one layer held together): "
        f"{concur_peak:.2f} KB  @ {concur_at}"
    )
    return peak_kb, peak_layer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the gatekeeper CNN and account for activation memory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size", type=int, default=96, help="input side length in pixels")
    args = parser.parse_args()

    model = build_model(args.size)
    model.summary()

    print(f"\nparameters: {model.count_params():,}")

    peak_kb, peak_layer = activation_memory_report(model)

    # Hard gate, on the specified metric: the peak single-layer output feature map
    print(f"\nactivation memory budget: {ACT_MEM_BUDGET_KB} KB")
    if peak_kb > ACT_MEM_BUDGET_KB:
        print(
            f"\n*** STOP: peak single-layer activation {peak_kb:.2f} KB at {peak_layer} "
            f"exceeds the {ACT_MEM_BUDGET_KB} KB budget. Halting before training. ***",
            file=sys.stderr,
        )
        return 1
    print(
        f"Pass: peak single-layer activation {peak_kb:.2f} KB at {peak_layer} "
        f"is within the {ACT_MEM_BUDGET_KB} KB budget."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
