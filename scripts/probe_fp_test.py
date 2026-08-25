#!/usr/bin/env python3
"""探针集误触发测试 —— 直接量「真实有人、无屏幕文字」场景下守门员的误触发率(FP)。

承接 person-bias 审计的结论：数据集层面正/负含人率几乎相等（gap≈0），「正例更常带人」
的数量失衡**不成立**。但这不排除"对着真人就触发"——可能是**协变量/语境偏移**：训练负类
里的人主要是干净影棚人像(`people_portrait`)，覆盖不到杂乱实景里的人。本脚本**直接量症状**：
把守门员跑在一个「有人、无屏幕文字」的探针集上，所有图按定义都该「不记(=0)」，
任何「记(=1)」都是 FP。报告部署阈值(及更高阈值)下的 FP 率。

为什么这样能区分两种病因（见审计 notebook 第 5 节）：
- 若探针 FP **高** 而 person-count gap≈0 ⇒ **协变量/语境偏移**（影棚人像不覆盖实景人），
  解法是**增负类人像的场景多样性**，不是单纯增量。
- 若探针 FP **低** ⇒ 线上"对着人就触发"的印象被混淆了（阈值/光照/相机管线），去那边查。

做四件事：
  1. 读探针集（任意分辨率的真实照片；若目录空则打印投喂规格并退出）。
  2. 防泄漏：用 Pexels-ID(文件名) + 感知哈希(pHash+像素相关，复用 check_leakage 同口径)
     双重核对，剔除任何与 train/val/test 撞图/近重复的探针，保证 FP 测在"干净探针"上。
  3. 对干净探针跑 **keras(float)** 与 **int8(.tflite)** 两个守门员，报告各阈值的 FP 率
     （部署阈值 0.55 + argmax 0.5 + 两个更紧的 0.7/0.9）。GT 全 0，FP 率 = 判「记」比例。
  4. 存 FP 案例拼图 + 对若干 FP 图做 Grad-CAM（看热力是否真落在人身上 —— FP 样本上的
     热力才是捷径的硬证据，比正例上的更说明问题）。写 docs/probes/probe_fp_audit.md。

口径选择（写进注释，便于审计）：
- 探针图是任意分辨率，故两条评估都用**部署预处理**：cv2 灰度 → resize 96×96 INTER_AREA
  → /255（int8 再按模型量化参数量化）。这正是我们想量的"部署时行为"。int8 直接复用
  `hardware/infer.py` 的 load_model/predict（与树莓派端口径唯一）；keras 用同样的灰度+缩放。
- **不重训、不改数据**。探针集与所有产物按 .gitignore 留在 data/ 下，不入库。

依赖：opencv、numpy、pandas、tensorflow（keras + tf.lite 回退）。复用 scripts/check_leakage.py
的 phash64/popcount64 与 hardware/infer.py 的 int8 运行时，无新增依赖。

示例：
  .venv/bin/python scripts/probe_fp_test.py                 # 目录空→打印投喂规格
  .venv/bin/python scripts/probe_fp_test.py --probe-dir data/probe_person_noscreen
  .venv/bin/python scripts/probe_fp_test.py --no-gradcam --thresholds 0.5,0.55,0.7,0.9
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# 复用泄漏检查的哈希实现（同 dedup_resplit 的做法），保证去重/泄漏判据全仓一致。
from check_leakage import phash64, popcount64  # noqa: E402

INPUT_SIZE = 96
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# 部署工作点：v4_mvp 调过的阈值 ~0.55（见 models/README）。对 p(记) 卡阈值。
DEPLOY_THRESHOLD = 0.55


# ───────────────────────────── 投喂规格 ─────────────────────────────
PROBE_SPEC = f"""\
━━━━━━━━━━━━━━━━━━━━━━━━ 探针集投喂规格 ━━━━━━━━━━━━━━━━━━━━━━━━
目标：测「真实有人、无屏幕文字」时守门员的误触发率。每张图按定义都该「不记」。

目录结构（在 data/ 下，已 gitignored）：
  data/probe_person_noscreen/
      *.jpg / *.png ...            # 平铺即可；也可按场景建子目录，脚本递归扫描

数量：建议 **~200 张**（最少 ~80 张才有稳定的 FP 率；200 张时 FP 的 95%CI 约 ±3–4pp）。

内容（关键：要与训练负类 people_portrait 的"干净影棚人像"在性质上拉开）：
  ✓ 会议室里有人（但**没有**可读的投影/屏幕文字）
  ✓ 办公室里的人在工作（屏幕**不可读**或不入镜）
  ✓ 咖啡馆 / 居家 / 街头 / 合影等杂乱实景里的人
  ✓ 多人、半身/坐姿、背景杂乱、自然光、非摆拍
  ✗ 不要影棚干净人像（那是 people_portrait 负类，已覆盖）
  ✗ 不要任何带可读屏幕/白板/文档/PPT 文字的图（那会变成"该记"，污染探针）

从 Pexels 取（可选）：
  1. 用现成 scripts/download_images.py + 一个探针配置（query 例：
     "people meeting room","office team working","friends cafe group",
     "people working laptop office","group people indoor candid"），输出到
     data/probe_person_noscreen/。download_images.py 自带按 Pexels-ID 全局去重。
  2. **本脚本会再做一道防泄漏**：按 Pexels-ID(文件名) + 感知哈希双重核对，
     自动剔除任何与 train/val/test 撞图/近重复的探针——所以即便 query 命中了
     训练里出现过的图也不会污染 FP（会被排除并报告）。

放好图后重跑：
  .venv/bin/python scripts/probe_fp_test.py --probe-dir data/probe_person_noscreen
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


# ───────────────────────────── Pexels-ID 解析 ─────────────────────────────
def photo_ids(stem: str) -> set[str]:
    """从文件名 stem 提取候选 Pexels 图片 id：长度≥6 的纯数字 token。

    覆盖两种命名：raw `<slug>_<seq4>_<id>`（id 在末尾）与 processed
    `<slug>_<seq4>_<id>_<hash8>`（id 在倒数第二段）。seq 为 4 位，<6 被排除。
    """
    return {t for t in re.split(r"[_\-.]", stem) if t.isdigit() and len(t) >= 6}


def manifest_id_set(manifest: Path) -> set[str]:
    df = pd.read_csv(manifest)
    ids: set[str] = set()
    for p in df["path"]:
        ids |= photo_ids(Path(p).stem)
    return ids


# ───────────────────────────── 感知哈希泄漏核对 ─────────────────────────────
def load_gray96(path: Path) -> np.ndarray | None:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    if g.shape != (INPUT_SIZE, INPUT_SIZE):
        g = cv2.resize(g, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return g


def build_manifest_index(manifest: Path, data_root: Path) -> dict:
    """载入 manifest 全量 96×96 灰度，预算 pHash + 中心化向量 + 范数（供向量化像素相关）。"""
    df = pd.read_csv(manifest)
    n = len(df)
    flats = np.zeros((n, INPUT_SIZE * INPUT_SIZE), np.float32)
    phashes = np.zeros(n, np.uint64)
    bad = 0
    for i, rel in enumerate(df["path"]):
        g = load_gray96(data_root / rel)
        if g is None:
            bad += 1
            continue
        flats[i] = g.astype(np.float32).flatten()
        phashes[i] = phash64(g)
    centered = flats - flats.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1)
    if bad:
        print(f"  [警告] manifest 有 {bad} 张读取失败（忽略）", file=sys.stderr)
    return {
        "phashes": phashes,
        "centered": centered,
        "norm": np.where(norm == 0, 1.0, norm),
        "paths": df["path"].to_numpy(),
        "splits": df["split"].to_numpy() if "split" in df else np.array(["?"] * n),
    }


def perceptual_match(gray96: np.ndarray, idx: dict, phash_th: int, pixel_corr: float) -> dict | None:
    """返回该探针图与 manifest 的最强近重复匹配（若达阈值），否则 None。"""
    ph = phash64(gray96)
    ham = popcount64(np.uint64(ph) ^ idx["phashes"])
    cand = np.nonzero(ham <= phash_th)[0]
    if cand.size == 0:
        return None
    v = gray96.astype(np.float32).flatten()
    v = v - v.mean()
    vn = np.linalg.norm(v) or 1.0
    best = None
    for j in cand:
        corr = float(np.dot(v, idx["centered"][j]) / (vn * idx["norm"][j]))
        if corr >= pixel_corr and (best is None or corr > best["pixel_corr"]):
            best = {"manifest_path": str(idx["paths"][j]), "split": str(idx["splits"][j]),
                    "phash_hamming": int(ham[j]), "pixel_corr": round(corr, 4)}
    return best


# ───────────────────────────── 推理 ─────────────────────────────
def keras_scores(model, grays: list[np.ndarray]) -> np.ndarray:
    """对一批灰度图(任意分辨率)按部署口径预处理后批量推理，返回 p(记) 数组。"""
    import tensorflow as tf  # 局部导入
    batch = np.zeros((len(grays), INPUT_SIZE, INPUT_SIZE, 1), np.float32)
    for i, g in enumerate(grays):
        r = cv2.resize(g, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        batch[i, ..., 0] = r.astype(np.float32) / 255.0
    probs = model.predict(tf.convert_to_tensor(batch), verbose=0)
    return probs[:, 1]


# ── int8 运行时（自含）。口径严格复刻 hardware/infer.py 的部署预处理：
#    灰度 → resize96 INTER_AREA → /255 → 按模型自带量化参数(scale,zero_point)量化成 int8；
#    输出按输出量化参数 dequant 回概率，取 p(记)=softmax[1]。
#    说明：infer.py 仅存在于 hardware/* 分支，故此处内联，保证本审计分支可独立运行；
#    逻辑与之逐行等价，将来在含 infer.py 的分支可直接换成 `import infer`。
def _make_interpreter(model_path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter  # 树莓派推荐运行时
        return Interpreter(model_path=str(model_path))
    except ImportError:
        import tensorflow as tf  # 笔记本回退，功能等价
        return tf.lite.Interpreter(model_path=str(model_path))


def load_int8(model_path: Path):
    it = _make_interpreter(model_path)
    it.allocate_tensors()
    return it


def int8_predict_one(interp, gray: np.ndarray) -> float:
    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    ind = interp.get_input_details()[0]
    outd = interp.get_output_details()[0]
    scale, zp = ind["quantization"]
    if scale == 0:  # float 输入模型：直接喂 float
        q = x.reshape(1, INPUT_SIZE, INPUT_SIZE, 1).astype(np.float32)
    else:
        q = np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)
        q = q.reshape(1, INPUT_SIZE, INPUT_SIZE, 1)
    interp.set_tensor(ind["index"], q)
    interp.invoke()
    y = interp.get_tensor(outd["index"])[0]
    o_scale, o_zp = outd["quantization"]
    probs = (y.astype(np.float32) - o_zp) * o_scale if o_scale else y.astype(np.float32)
    return float(probs[1])  # p(记)


def int8_scores(interp, grays: list[np.ndarray]) -> np.ndarray:
    """传 2D 灰度（跳过色彩转换，内部自做 resize+量化），返回 p(记) 数组。"""
    out = np.zeros(len(grays), np.float32)
    for i, g in enumerate(grays):
        out[i] = int8_predict_one(interp, g)
    return out


def fp_table(scores: np.ndarray, thresholds: list[float]) -> list[dict]:
    """GT 全 0：FP 率 = 判「记」(score≥th) 的比例。"""
    n = len(scores)
    rows = []
    for t in thresholds:
        fp = int((scores >= t).sum())
        rows.append({"threshold": round(t, 3), "n": n, "fp": fp,
                     "fp_rate": round(fp / n, 4) if n else 0.0})
    return rows


# ───────────────────────────── Grad-CAM（FP 案例） ─────────────────────────────
def gradcam_on(model_path: Path, items: list[tuple[str, np.ndarray, float]],
               out_dir: Path) -> dict:
    """对 FP 案例做 Grad-CAM（block4_relu 对 p(记)）。items=[(name, gray, score)]。"""
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, compile=False)
        gm = tf.keras.Model(model.input,
                            [model.get_layer("block4_relu").output,
                             model.get_layer("logits").output])

        def cam(gray96):
            x = tf.convert_to_tensor((gray96.astype(np.float32) / 255.0)[None, ..., None])
            with tf.GradientTape() as tape:
                conv, logits = gm(x, training=False)
                loss = logits[:, 1]
            grads = tape.gradient(loss, conv)
            w = tf.reduce_mean(grads, axis=(1, 2))
            c = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], axis=-1)[0])
            return (c / (tf.reduce_max(c) + 1e-8)).numpy()

        saved = []
        for name, gray, score in items:
            g96 = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
            m = cam(g96)
            disp = cv2.cvtColor(cv2.resize(gray, (288, 288), interpolation=cv2.INTER_AREA),
                                cv2.COLOR_GRAY2BGR)
            heat = cv2.applyColorMap(cv2.resize((m * 255).astype(np.uint8), (288, 288)),
                                     cv2.COLORMAP_JET)
            over = cv2.addWeighted(disp, 0.55, heat, 0.45, 0)
            cv2.putText(over, f"p(rec)={score:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            path = out_dir / f"gradcam_fp_{name}.png"
            cv2.imwrite(str(path), np.hstack([disp, over]))
            saved.append(path.name)
        print(f"  Grad-CAM(FP): {len(saved)} 张 → {out_dir}（左原图/右热力；热力落在人身=捷径硬证据）")
        return {"ok": True, "saved": saved}
    except Exception as e:  # noqa: BLE001
        print(f"  Grad-CAM 跳过：{type(e).__name__}: {e}")
        return {"ok": False, "saved": [], "note": f"{type(e).__name__}: {e}"}


def fp_montage(items: list[tuple[str, np.ndarray, float]], out_dir: Path, n: int) -> str | None:
    """FP 案例拼图（部署阈值下判「记」的探针），每格标 p(记)。"""
    items = items[:n]
    if not items:
        return None
    thumbs = []
    for _, gray, score in items:
        t = cv2.cvtColor(cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_GRAY2BGR)
        cv2.putText(t, f"{score:.2f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)
        thumbs.append(t)
    cols = min(6, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    grid = np.full((rows * 160, cols * 160, 3), 30, np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * 160:(r + 1) * 160, c * 160:(c + 1) * 160] = t
    path = out_dir / "montage_fp_cases.png"
    cv2.imwrite(str(path), grid)
    return path.name


# ───────────────────────────── 报告文档 ─────────────────────────────
def write_markdown(md_path: Path, results: dict | None, probe_dir: Path, out_dir: Path) -> None:
    L = ["# 探针集误触发测试（probe FP audit）\n"]
    L.append("> 直接量「真实有人、无屏幕文字」场景下守门员的误触发率(FP)。"
             "每张探针图按定义都该「不记」，任何「记」都是 FP。**仅诊断，未重训、未改数据。**\n")
    L.append("\n## 方法\n")
    L.append(f"- 探针目录：`{probe_dir}/`（gitignored）。预处理=部署口径"
             "（cv2 灰度→resize96 INTER_AREA→/255；int8 再量化）。\n")
    L.append("- 防泄漏：Pexels-ID(文件名) + 感知哈希(pHash≤阈 且 像素相关≥阈，复用 "
             "`check_leakage` 同口径) 双重核对，剔除与 train/val/test 撞图/近重复的探针。\n")
    L.append("- 两个守门员：`keras(float)` 与 `int8(.tflite)`（自含 int8 运行时，口径复刻 "
             "`hardware/infer.py` 部署预处理）。部署阈值 0.55，另列 0.5/0.7/0.9。\n")

    if results is None:
        L.append("\n## 投喂规格（探针目录当前为空）\n")
        L.append("```\n" + PROBE_SPEC + "\n```\n")
        L.append("\n## 结果\n_待投喂探针图后重跑本脚本生成。_\n")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("".join(L), encoding="utf-8")
        return

    lk = results["leakage"]
    L.append("\n## 1. 防泄漏核对\n")
    L.append(f"- 探针总数 {results['n_total']}；剔除泄漏 {lk['n_leaked']} 张"
             f"（Pexels-ID 撞 {lk['n_id']} / 感知近重复 {lk['n_perceptual']}）"
             f"；**干净探针 {results['n_clean']} 张**用于 FP 测。\n")
    if lk["examples"]:
        L.append("- 泄漏样例（探针 → 命中的 split/图）：\n")
        for e in lk["examples"][:8]:
            tag = e.get("manifest_path", f"ID={e.get('id')}")
            L.append(f"  - `{e['probe']}` → {e.get('split','?')} `{tag}`"
                     f"（corr={e.get('pixel_corr','-')}）\n")

    L.append("\n## 2. 误触发率（FP）—— 干净探针，GT 全=不记\n")
    for tag, rows in (("keras(float)", results["fp_keras"]), ("int8(.tflite)", results["fp_int8"])):
        L.append(f"\n**{tag}**\n\n| 阈值 | 判「记」/总数 | FP 率 |\n|---|---|---|\n")
        for r in rows:
            mark = "  ← 部署" if abs(r["threshold"] - DEPLOY_THRESHOLD) < 1e-6 else ""
            L.append(f"| {r['threshold']} | {r['fp']}/{r['n']} | **{r['fp_rate']*100:.1f}%**{mark} |\n")

    L.append("\n## 3. FP 案例（人眼复核）\n")
    if results.get("montage"):
        L.append(f"- 拼图：`{out_dir}/{results['montage']}`（红字=p(记)）\n")
    gc = results.get("gradcam", {})
    if gc.get("ok"):
        L.append(f"- Grad-CAM：`{out_dir}/gradcam_fp_*.png`（{len(gc['saved'])} 张，"
                 "左原图/右热力）。**FP 样本上热力若锁定人脸/人身，即捷径硬证据。**\n")
    else:
        L.append(f"- Grad-CAM：未产出（{gc.get('note','-')}）。\n")

    dep = next(r for r in results["fp_int8"] if abs(r["threshold"] - DEPLOY_THRESHOLD) < 1e-6)
    rate = dep["fp_rate"] * 100
    L.append("\n## 4. 读数\n")
    if results["n_clean"] < 60:
        L.append(f"- ⚠ 干净探针仅 {results['n_clean']} 张，FP 率统计噪声大，先补到 ~200 再下结论。\n")
    if rate >= 25:
        L.append(f"- int8 部署阈值(0.55)下 FP **{rate:.1f}%**：**偏高**。结合审计 gap≈0 ⇒ "
                 "更像**协变量/语境偏移**（影棚人像负类不覆盖杂乱实景人），解法是"
                 "**增负类人像的场景多样性**而非单纯增量。看 Grad-CAM 是否锁人确认。\n")
    elif rate >= 10:
        L.append(f"- int8 部署阈值(0.55)下 FP **{rate:.1f}%**：**中等**。有一定语境偏移，"
                 "值得补多样化的实景人像负例；同时核对 FP 案例是否多为「半屏/反光」等边界图。\n")
    else:
        L.append(f"- int8 部署阈值(0.55)下 FP **{rate:.1f}%**：**低**。守门员在真实人像上并不普遍"
                 "误触发——线上「对着人就触发」的印象更可能被阈值/光照/相机管线混淆，去那边查。\n")
    L.append("- keras 与 int8 的 FP 若明显不同，说明量化改变了工作点，部署阈值需按 int8 重标。\n")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("".join(L), encoding="utf-8")


# ───────────────────────────── 主流程 ─────────────────────────────
def collect_images(probe_dir: Path) -> list[Path]:
    if not probe_dir.exists():
        return []
    return sorted(p for p in probe_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)


def main() -> int:
    p = argparse.ArgumentParser(
        description="探针集误触发(FP)测试：量真实有人、无屏幕文字时守门员的误触发率。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--probe-dir", type=Path, default=Path("data/probe_person_noscreen"),
                   help="探针图目录（任意分辨率真实照片；递归扫描）")
    p.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"),
                   help="train/val/test 全集，用于防泄漏核对")
    p.add_argument("--data-root", type=Path, default=Path("data/processed"))
    p.add_argument("--keras-model", type=Path, default=Path("models/gatekeeper_v4_mvp.keras"))
    p.add_argument("--int8-model", type=Path, default=Path("models/gatekeeper_v4_mvp_int8.tflite"))
    p.add_argument("--out", type=Path, default=Path("data/processed/probe_fp_audit"),
                   help="产物目录（拼图/Grad-CAM/CSV/JSON，gitignored）")
    p.add_argument("--md-out", type=Path, default=Path("docs/probes/probe_fp_audit.md"))
    p.add_argument("--thresholds", type=str, default="0.5,0.55,0.7,0.9")
    p.add_argument("--phash-th", type=int, default=6, help="感知哈希汉明阈（同 check_leakage）")
    p.add_argument("--pixel-corr", type=float, default=0.90, help="像素相关阈（同 check_leakage）")
    p.add_argument("--keep-leaked", action="store_true", help="不剔除泄漏探针（仅报告，调试用）")
    p.add_argument("--no-leakage-check", action="store_true", help="跳过防泄漏核对（不建议）")
    p.add_argument("--no-gradcam", action="store_true")
    p.add_argument("--gradcam-n", type=int, default=10)
    p.add_argument("--montage-n", type=int, default=18)
    p.add_argument("--limit", type=int, default=None, help="只取前 N 张探针（冒烟用）")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    if DEPLOY_THRESHOLD not in thresholds:
        thresholds = sorted(set(thresholds + [DEPLOY_THRESHOLD]))

    args.out.mkdir(parents=True, exist_ok=True)
    images = collect_images(args.probe_dir)
    if args.limit:
        images = images[:args.limit]

    if not images:
        print(PROBE_SPEC)
        print(f"\n探针目录 `{args.probe_dir}` 为空/不存在 —— 已把投喂规格写入 {args.md_out}。")
        write_markdown(args.md_out, None, args.probe_dir, args.out)
        return 0

    print(f"探针图 {len(images)} 张 @ {args.probe_dir}")

    # ── 防泄漏核对 ──
    leaked: dict[str, dict] = {}
    n_id = n_perc = 0
    if not args.no_leakage_check:
        print("防泄漏核对（Pexels-ID + 感知哈希）…")
        mids = manifest_id_set(args.manifest)
        idx = build_manifest_index(args.manifest, args.data_root)
        for img in images:
            ids = photo_ids(img.stem)
            hit_id = ids & mids
            if hit_id:
                leaked[str(img)] = {"probe": img.name, "id": sorted(hit_id)[0],
                                    "split": "?", "reason": "pexels_id"}
                n_id += 1
                continue
            g = load_gray96(img)
            if g is None:
                continue
            m = perceptual_match(g, idx, args.phash_th, args.pixel_corr)
            if m:
                leaked[str(img)] = {"probe": img.name, "reason": "perceptual", **m}
                n_perc += 1
        print(f"  泄漏：Pexels-ID {n_id} 张 / 感知近重复 {n_perc} 张（共 {len(leaked)}）")

    clean = images if args.keep_leaked else [i for i in images if str(i) not in leaked]
    print(f"  干净探针：{len(clean)} 张" + ("（--keep-leaked：未剔除）" if args.keep_leaked else ""))
    if not clean:
        print("没有干净探针可测（全部判为泄漏）。检查探针来源是否与训练集重叠。")
        return 1

    # ── 载入干净探针灰度（保留原分辨率给可视化/部署预处理）──
    names, grays = [], []
    for img in clean:
        g = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        names.append(re.sub(r"[^A-Za-z0-9]+", "_", img.stem)[:60])
        grays.append(g)

    # ── 两个守门员推理 ──
    import tensorflow as tf  # noqa: E402
    print(f"keras 推理：{args.keras_model}")
    kmodel = tf.keras.models.load_model(args.keras_model, compile=False)
    s_keras = keras_scores(kmodel, grays)
    print(f"int8 推理：{args.int8_model}")
    interp = load_int8(args.int8_model)
    s_int8 = int8_scores(interp, grays)

    fp_keras = fp_table(s_keras, thresholds)
    fp_int8 = fp_table(s_int8, thresholds)

    # ── 打印 ──
    print("\n" + "=" * 60)
    print("探针 FP（GT 全=不记；FP 率=判「记」比例）")
    print("=" * 60)
    for tag, rows in (("keras(float)", fp_keras), ("int8 (.tflite)", fp_int8)):
        print(f"\n[{tag}]  阈值   判记/总   FP率")
        for r in rows:
            mark = "  ← 部署" if abs(r["threshold"] - DEPLOY_THRESHOLD) < 1e-6 else ""
            print(f"          {r['threshold']:<5}  {r['fp']:>3}/{r['n']:<3}  "
                  f"{r['fp_rate']*100:5.1f}%{mark}")

    # ── FP 案例（按 int8 部署阈值）拼图 + Grad-CAM ──
    fp_items = sorted(
        [(names[i], grays[i], float(s_int8[i])) for i in range(len(grays))
         if s_int8[i] >= DEPLOY_THRESHOLD],
        key=lambda t: -t[2])
    # 保持按分数降序：最「自信」的 FP 排在前，拼图/Grad-CAM 先看这些
    montage = fp_montage(fp_items, args.out, args.montage_n)
    if montage:
        print(f"  FP 拼图 → {args.out/montage}")
    gradcam = {"ok": False, "saved": [], "note": "--no-gradcam"}
    if not args.no_gradcam and fp_items:
        gradcam = gradcam_on(args.keras_model, fp_items[:args.gradcam_n], args.out)

    # ── 落盘逐图 CSV ──
    det = pd.DataFrame({
        "probe": [c.name for c in clean[:len(grays)]],
        "p_record_keras": np.round(s_keras, 4),
        "p_record_int8": np.round(s_int8, 4),
        "fp_at_deploy_int8": (s_int8 >= DEPLOY_THRESHOLD).astype(int),
    })
    det.to_csv(args.out / "probe_scores.csv", index=False)

    results = {
        "n_total": len(images), "n_clean": len(grays),
        "leakage": {"n_leaked": len(leaked), "n_id": n_id, "n_perceptual": n_perc,
                    "examples": list(leaked.values())},
        "fp_keras": fp_keras, "fp_int8": fp_int8,
        "deploy_threshold": DEPLOY_THRESHOLD,
        "montage": montage, "gradcam": gradcam,
    }
    (args.out / "probe_fp_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(args.md_out, results, args.probe_dir, args.out)
    print(f"\n审计文档 → {args.md_out}；汇总 JSON → {args.out/'probe_fp_summary.json'}")

    dep_k = next(r for r in fp_keras if abs(r["threshold"] - DEPLOY_THRESHOLD) < 1e-6)
    dep_i = next(r for r in fp_int8 if abs(r["threshold"] - DEPLOY_THRESHOLD) < 1e-6)
    print("\nRESULT " + json.dumps({
        "n_clean": len(grays), "n_leaked": len(leaked),
        "fp_rate_keras@0.55": dep_k["fp_rate"], "fp_rate_int8@0.55": dep_i["fp_rate"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
