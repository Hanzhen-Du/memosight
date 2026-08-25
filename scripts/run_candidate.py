#!/usr/bin/env python3
"""Task1 单候选全程流水线（5-seed）—— 训练→评估@thr→int8导出+白名单+ΔF1→双探针FP/召回。

对一个复杂度候选（channels + convs_per_stage）跑完整 5-seed 评估，全部在 ESP32 部署口径下：
  每个 seed：
    1. do_split 重切分去重池（与 run_variance 同口径）→ 训练（冻结 BEST 超参）。
    2. val/test 在阈值 thr 下算 F1/FN/FP（keras）。
    3. int8 导出（in-memory，固定 batch=1）→ 逐算子查 TFLM 白名单 + 全 int8 dtype 核对。
    4. int8 在 test 上算 F1 → 量化掉点 ΔF1 = F1(int8) − F1(keras)，同阈值。
    5. 两个 held-out 探针（int8）：noscreen FP@thr、person+screen 召回@thr。
       探针防泄漏：按 Pexels-ID 核对（感知近重复已在 task2 收尾时全量核验=0，数据未变）。
  跨 seed 聚合 mean±std，写 docs/results/task1_results/<tag>.json。seed42 模型存为该候选 deploy 产物。

架构常量（参数/激活/权重/算子集）无 seed 方差，单独记录。

依赖：复用 model/train/dedup_resplit/evaluate/export_tflite/probe_fp_test，无新增。
示例：
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
INDOOR = Path("data/probe_indoor_env_v2")  # task2b held-out 室内环境泛化探针（负例→测 FP）


def load_clean_probe_grays(probe_dir: Path, manifest_ids: set[str]) -> tuple[list[np.ndarray], int, int]:
    """读探针为灰度，按 Pexels-ID 剔除与训练池撞图者（感知近重复已在 task2 全量核验=0）。

    探针为任意分辨率原图（noscreen 235 / screen 181，单张可达数 MB）。**在加载时即
    resize 到 INPUT_SIZE×INPUT_SIZE（INTER_AREA）**——这与下游 int8_predict_one 的部署预处理
    一字不差（96→96 二次 resize 为恒等），数值完全等价；目的是避免把数百张全分辨率灰度图同时
    驻留内存（曾导致 ~9.5GB 占用 → OOM 杀进程）。resize 后整批仅约数 MB。"""
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
    """探针在 int8 模型下判「记」(score>=thr) 的比例：noscreen→FP率, screen→召回。"""
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

    # int8 导出（in-memory）+ 验证
    int8_path = tmp_dir / f"{tag}_s{seed}_int8.tflite"
    rng = np.random.default_rng(42)
    rep = list(train_paths)
    rng.shuffle(rep)
    export_int8(model, int8_path, rep, n_rep)
    ops = verify_ops(int8_path)
    all_wl = all(ok for _, ok in ops)
    dt = verify_dtypes(int8_path)
    n_float = len(dt["float32_tensors"])

    # 量化掉点 ΔF1（同阈值，test）
    p_int8 = predict_int8(int8_path, test_paths)
    int8_f1 = int8_metrics(p_int8, test_labels, thr)["f1"]
    quant_df1 = round(int8_f1 - test_m["f1"], 4)

    # 探针（int8）
    noscreen_fp = int8_probe_rate(int8_path, probe_cache["noscreen"], thr)
    screen_recall = int8_probe_rate(int8_path, probe_cache["screen"], thr)
    # task2b held-out 室内环境探针（负例→FP）；目录缺/空则记 None（task1 复用时不破坏）
    indoor_env_fp = (int8_probe_rate(int8_path, probe_cache["indoor"], thr)
                     if probe_cache.get("indoor") else None)

    # seed42 留作 deploy 产物；其余 int8 临时文件保留在 tmp（gitignored）
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
    ap = argparse.ArgumentParser(description="Task1 单候选 5-seed 全程流水线。")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--channels", required=True, help="逗号分隔，如 8,16,64,128")
    ap.add_argument("--convs", default="1", help="int 或逗号分隔与 channels 等长，如 1,1,2,2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7, 123, 2024])
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_dedup.csv"))
    ap.add_argument("--leak-manifest", type=Path, default=Path("data/processed/manifest.csv"),
                    help="探针防泄漏核对用的全集 ID 来源（superset，更保守）")
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

    # 架构常量（无 seed 方差）
    cm = build_model(BEST["size"], bn_momentum=BEST["bn_momentum"],
                     channels=channels, convs_per_stage=convs)
    params = cm.count_params()

    # 探针清洗（一次，跨 seed 复用：ID 防泄漏；感知近重复已在 task2 收尾全量核验=0）
    mids = manifest_id_set(args.leak_manifest)
    ns_grays, ns_tot, ns_leak = load_clean_probe_grays(NOSCREEN, mids)
    sc_grays, sc_tot, sc_leak = load_clean_probe_grays(SCREEN, mids)
    in_grays, in_tot, in_leak = load_clean_probe_grays(INDOOR, mids)
    print(f"[{args.tag}] 探针：noscreen {len(ns_grays)}/{ns_tot}(leak {ns_leak}) | "
          f"screen {len(sc_grays)}/{sc_tot}(leak {sc_leak}) | "
          f"indoor_env_v2 {len(in_grays)}/{in_tot}(leak {in_leak})", flush=True)
    probe_cache = {"noscreen": ns_grays, "screen": sc_grays, "indoor": in_grays}

    rows = [run_seed(s, args.tag, channels, convs, args.threshold, args.manifest,
                     args.data_root, args.out_dir, args.tmp_dir, args.n_rep, probe_cache)
            for s in args.seeds]

    summary_keys = ["val_f1", "test_f1", "test_fn", "test_fp", "test_recall",
                    "test_acc", "quant_df1", "noscreen_fp", "screen_recall"]
    if in_grays:  # indoor_env_v2 探针存在时才聚合（否则各 seed 记 None，不可平均）
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
