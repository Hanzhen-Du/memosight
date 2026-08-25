#!/usr/bin/env python3
"""Full 5-seed pipeline for one task 1 candidate: train, evaluate at a threshold, export int8
with whitelist and dF1 checks, then score both probes for FP and recall.

Runs a complete 5-seed evaluation for one complexity candidate (channels plus
convs_per_stage), entirely under ESP32 deployment conditions.

  For each seed:
    1. do_split re-splits the deduplicated pool, the same way run_variance does, then trains
       with the frozen best hyperparameters.
    2. Compute F1, FN and FP on val and test at threshold thr, in keras.
    3. Export int8 in memory with batch pinned to 1, then check every operator against the
       TFLite Micro whitelist and confirm all dtypes are int8.
    4. Compute F1 on test with the int8 model, giving the quantisation loss
       dF1 = F1(int8) - F1(keras) at the same threshold.
    5. Score both held-out probes with the int8 model: noscreen FP at thr, and
       person-plus-screen recall at thr.
       Probe leakage control is by Pexels ID. Perceptual near-duplicates were verified as zero
       across the whole set at the end of task 2 and the data has not changed since.

  Aggregate mean and standard deviation across seeds and write
  docs/results/task1_results/<tag>.json. The seed 42 model is kept as this candidate's
  deployment artifact.

Architecture constants (parameters, activations, weights, operator set) have no seed variance
and are recorded separately.

Dependencies: reuses model, train, dedup_resplit, evaluate, export_tflite and probe_fp_test.
Nothing new.

Example:
  .venv/bin/python scripts/run_candidate.py --tag A_wide_late --channels 8,16,64,128 --convs 1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import tensorflow as tf

from model import build_model
from train import make_dataset, compute_class_weight
from dedup_resplit import do_split
from evaluate import metrics_from_probs
from export_tflite import (export_int8, verify_ops, verify_dtypes,
                           predict_int8, metrics as int8_metrics)
from probe_fp_test import (collect_images, manifest_id_set, photo_ids, fp_table,
                           INPUT_SIZE)

BEST = dict(bn_momentum=0.9, patience=15, start_from_epoch=20, epochs=80,
            lr=1e-3, batch_size=32, size=96)
NOSCREEN = Path("data/probe_person_noscreen")
SCREEN = Path("data/probe_person_screen")
INDOOR = Path("data/probe_indoor_env_v2")  # task2b held-out indoor-environment generalisation
                                           # probe; negatives, so it measures FP


def load_clean_probe_grays(probe_dir: Path, manifest_ids: set[str]) -> tuple[list[np.ndarray], int, int]:
    """Load the probes as greyscale, removing any image colliding with the training pool by
    Pexels ID. Perceptual near-duplicates were verified as zero across the whole set in task 2.

    Probe images are originals at arbitrary resolution (noscreen 235, screen 181, and a single
    image can be several MB). They are resized to INPUT_SIZE x INPUT_SIZE with INTER_AREA at
    load time. That is identical to the deployment preprocessing in int8_predict_one below, a
    second 96-to-96 resize being the identity, so the numbers are exactly equivalent. The point
    is to avoid holding hundreds of full-resolution greyscale images in memory at once, which
    once reached about 9.5 GB and was killed by the OOM killer. After the resize the whole
    batch is only a few MB."""
    imgs = collect_images(probe_dir)
    grays, leaked = [], 0
    for p in imgs:
        if photo_ids(p.stem) & manifest_ids:
            leaked += 1
            continue
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            grays.append(cv2.resize(g, (INPUT_SIZE, INPUT_SIZE),
                                    interpolation=cv2.INTER_AREA))
    return grays, len(imgs), leaked


def int8_probe_rate(int8_path: Path, grays: list[np.ndarray], thr: float) -> float:
    """Fraction of probe images the int8 model judges as record (score >= thr). On noscreen
    that is the FP rate; on screen it is recall."""
    from probe_fp_test import load_int8, int8_scores
    interp = load_int8(int8_path)
    scores = int8_scores(interp, grays)
    return fp_table(scores, [thr])[0]["fp_rate"]


def run_seed(seed, tag, channels, convs, thr, manifest, data_root, out_dir,
             tmp_dir, n_rep, probe_cache) -> dict:
    tf.keras.backend.clear_session()
    splits = do_split(manifest, out_dir, prefix=f"t1_{tag}_s{seed}_", seed=seed, quiet=True)
    tf.keras.utils.set_random_seed(seed)

    def to_ds(df, shuffle, augment):
        paths = [str(data_root / p) for p in df["path"]]
        labels = df["label"].to_numpy(dtype=np.int32)
        return make_dataset(paths, labels, BEST["size"], BEST["batch_size"],
                            shuffle=shuffle, seed=seed, augment=augment), paths, labels

    train_ds, train_paths, _ = to_ds(splits["train"], True, True)
    val_ds, _, val_labels = to_ds(splits["val"], False, False)
    test_ds, test_paths, test_labels = to_ds(splits["test"], False, False)
    cw = compute_class_weight(splits["train"]["label"].to_numpy())

    model = build_model(BEST["size"], bn_momentum=BEST["bn_momentum"],
                        channels=channels, convs_per_stage=convs, name=f"gk_{tag}")
    model.compile(optimizer=tf.keras.optimizers.Adam(BEST["lr"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cb = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=BEST["patience"],
        start_from_epoch=BEST["start_from_epoch"], restore_best_weights=True, verbose=0)]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=BEST["epochs"],
                     class_weight=cw, callbacks=cb, verbose=0)

    val_probs = model.predict(val_ds, verbose=0)
    test_probs = model.predict(test_ds, verbose=0)
    val_m = metrics_from_probs(val_probs, val_labels, thr)
    test_m = metrics_from_probs(test_probs, test_labels, thr)

    # int8 export in memory, plus verification
    int8_path = tmp_dir / f"{tag}_s{seed}_int8.tflite"
    rng = np.random.default_rng(42)
    rep = list(train_paths)
    rng.shuffle(rep)
    export_int8(model, int8_path, rep, n_rep)
    ops = verify_ops(int8_path)
    all_wl = all(ok for _, ok in ops)
    dt = verify_dtypes(int8_path)
    n_float = len(dt["float32_tensors"])

    # Quantisation loss dF1, same threshold, on test
    p_int8 = predict_int8(int8_path, test_paths)
    int8_f1 = int8_metrics(p_int8, test_labels, thr)["f1"]
    quant_df1 = round(int8_f1 - test_m["f1"], 4)

    # Probes, int8
    noscreen_fp = int8_probe_rate(int8_path, probe_cache["noscreen"], thr)
    screen_recall = int8_probe_rate(int8_path, probe_cache["screen"], thr)
    # task2b held-out indoor-environment probe (negatives, so FP). If the directory is missing
    # or empty, record None, so reusing this for task1 does not break
    indoor_env_fp = (int8_probe_rate(int8_path, probe_cache["indoor"], thr)
                     if probe_cache.get("indoor") else None)

    # seed 42 is kept as the deployment artifact; the other int8 temporaries stay in tmp,
    # which is gitignored
    if seed == 42:
        keras_path = Path("models/task1_candidates") / f"gatekeeper_task1_{tag}.keras"
        keras_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(keras_path)
        (Path("models/task1_candidates") / f"gatekeeper_task1_{tag}_int8.tflite").write_bytes(
            int8_path.read_bytes())

    r = {"seed": seed, "stopped_epoch": len(hist.history["loss"]),
         "val_f1": val_m["f1"], "test_f1": test_m["f1"],
         "test_fn": test_m["fn_rate"], "test_fp": test_m["fp_rate"],
         "test_recall": test_m["recall"], "test_acc": test_m["accuracy"],
         "quant_df1": quant_df1, "all_whitelisted": all_wl, "n_float32": n_float,
         "noscreen_fp": noscreen_fp, "screen_recall": screen_recall,
         "indoor_env_fp": indoor_env_fp,
         "ops": sorted({name for name, _ in ops})}
    ind_s = f"{indoor_env_fp:.3f}" if indoor_env_fp is not None else "n/a"
    print(f"[{tag} seed={seed}] ep={r['stopped_epoch']:>2} valF1={val_m['f1']:.4f} "
          f"testF1={test_m['f1']:.4f} FN={test_m['fn_rate']:.4f} FP={test_m['fp_rate']:.4f} "
          f"ΔF1q={quant_df1:+.4f} WL={'Y' if all_wl else 'N'} "
          f"noscrFP={noscreen_fp:.3f} scrRec={screen_recall:.3f} indoorFP={ind_s}", flush=True)
    return r


def agg(rows, key):
    v = np.array([r[key] for r in rows], float)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Full 5-seed pipeline for one task 1 candidate.")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--channels", required=True, help="comma separated, for example 8,16,64,128")
    ap.add_argument("--convs", default="1", help="an int, or a comma-separated list the same length as channels, for example 1,1,2,2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7, 123, 2024])
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_dedup.csv"))
    ap.add_argument("--leak-manifest", type=Path, default=Path("data/processed/manifest.csv"),
                    help="source of the full id set used for the probe leakage check; a superset is more conservative")
    ap.add_argument("--data-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--tmp-dir", type=Path,
                    default=Path("data/processed/task1_int8_tmp"))
    ap.add_argument("--n-rep", type=int, default=200)
    ap.add_argument("--results-dir", type=Path, default=Path("docs/results/task1_results"))
    args = ap.parse_args()

    channels = tuple(int(c) for c in args.channels.split(","))
    convs_list = [int(c) for c in args.convs.split(",")]
    convs = convs_list[0] if len(convs_list) == 1 else tuple(convs_list)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    # Architecture constants, which have no seed variance
    cm = build_model(BEST["size"], bn_momentum=BEST["bn_momentum"],
                     channels=channels, convs_per_stage=convs)
    params = cm.count_params()

    # Clean the probes once and reuse across seeds. Leakage control by id; perceptual
    # near-duplicates were verified as zero across the whole set at the end of task 2
    mids = manifest_id_set(args.leak_manifest)
    ns_grays, ns_tot, ns_leak = load_clean_probe_grays(NOSCREEN, mids)
    sc_grays, sc_tot, sc_leak = load_clean_probe_grays(SCREEN, mids)
    in_grays, in_tot, in_leak = load_clean_probe_grays(INDOOR, mids)
    print(f"[{args.tag}] probes: noscreen {len(ns_grays)}/{ns_tot} (leaked {ns_leak}) | "
          f"screen {len(sc_grays)}/{sc_tot}(leak {sc_leak}) | "
          f"indoor_env_v2 {len(in_grays)}/{in_tot}(leak {in_leak})", flush=True)
    probe_cache = {"noscreen": ns_grays, "screen": sc_grays, "indoor": in_grays}

    rows = [run_seed(s, args.tag, channels, convs, args.threshold, args.manifest,
                     args.data_root, args.out_dir, args.tmp_dir, args.n_rep, probe_cache)
            for s in args.seeds]

    summary_keys = ["val_f1", "test_f1", "test_fn", "test_fp", "test_recall",
                    "test_acc", "quant_df1", "noscreen_fp", "screen_recall"]
    if in_grays:  # only aggregate when the indoor_env_v2 probe exists; otherwise each seed
                  # records None and there is nothing to average
        summary_keys.append("indoor_env_fp")
    summary = {k: agg(rows, k) for k in summary_keys}
    out = {
        "tag": args.tag, "channels": list(channels),
        "convs_per_stage": convs if isinstance(convs, int) else list(convs),
        "params": params, "int8_weight_kb": round(params / 1024, 1),
        "threshold": args.threshold, "seeds": args.seeds,
        "all_whitelisted": all(r["all_whitelisted"] for r in rows),
        "max_float32_tensors": max(r["n_float32"] for r in rows),
        "ops_union": sorted(set().union(*[set(r["ops"]) for r in rows])),
        "probe_n": {"noscreen": len(ns_grays), "screen": len(sc_grays),
                    "indoor_env_v2": len(in_grays)},
        "per_seed": rows, "summary": summary,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / f"{args.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    ind_done = f"  indoorFP={summary['indoor_env_fp']['mean']:.3f}" if "indoor_env_fp" in summary else ""
    print(f"\n[{args.tag}] DONE  testF1={summary['test_f1']['mean']:.4f}±{summary['test_f1']['std']:.4f}"
          f"  noscrFP={summary['noscreen_fp']['mean']:.3f}  scrRec={summary['screen_recall']['mean']:.3f}"
          f"{ind_done}"
          f"  WL={'all' if out['all_whitelisted'] else 'FAIL'}  → {args.results_dir/(args.tag+'.json')}",
          flush=True)
    print("RESULT " + json.dumps(out["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
