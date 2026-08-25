# Task2 阈值对照表 —— 决策点 1 的"账"（2026-06-25）

> **Summary (EN).** The deployment-threshold ledger. Per-threshold FN / FP / probe numbers
> so the operating point can be chosen explicitly rather than defaulted to argmax. Moving
> 0.50 → 0.40 buys test FN 31.5% → 26.4% and person-plus-screen recall 50.8% → 61.9%, paid
> for with test FP 17.6% → 26.2%. Measurement-basis caveats (keras float for test, int8 for
> probes) are stated up front.

模型：`gatekeeper_task2_mvp`（部署单 seed）。本文件为**决策点 1**（点死部署阈值）提供逐阈值的
FN / FP / 探针 trade-off 实测数，并附扩充后探针（181 张，CI 收窄）的 vs‑v4 公平对比。

## 0. 口径声明（务必先读，避免误比）
- **test 列（FN/FP/recall/F1）= keras float**，来自 `evaluate.py --pr-sweep`，跑在 `dedup_test.csv`
  （n=369：正 159 / 负 210）。这是项目既定的 test 口径（与 5-seed 报告同源，单 seed 部署模型）。
- **两个探针列 = int8(.tflite) 部署口径**，来自 `probe_fp_test.py`，跑在 held-out 探针上。
  `probe_person_noscreen`（n=235，GT=不记，FP 率=判「记」比例）；
  `probe_person_screen`（**扩充后 n=181**，GT=记，"召回"=判「记」比例）。
- ⚠ **混口径限制**：test=keras / 探针=int8。§6 报告显示 int8 量化在 test 上 ΔF1 仅 −0.006(@0.5)，
  工作点漂移很小；但严格说 test 列与探针列不是同一引擎。决策的核心 trade-off（探针 FP vs 召回）
  本身已是 int8 部署口径，故此限制不影响阈值取舍的方向。无现成 int8-test 带标评估脚本，故沿用既定 keras-test。

## 1. 三阈值对照表（task2 部署模型）

| 阈值 | test FN | test FP | noscreen 探针 FP | person+screen 探针召回 |
|---|---|---|---|---|
| **0.40** | **42/159 = 26.4%** | 55/210 = 26.2% | 71/235 = **30.2%** | 112/181 = **61.9%** (95%CI ±7.1pp) |
| **0.45** | **46/159 = 28.9%** | 46/210 = 21.9% | 64/235 = **27.2%** | 100/181 = **55.2%** (95%CI ±7.2pp) |
| **0.50** | **50/159 = 31.5%** | 37/210 = 17.6% | 56/235 = **23.8%** | 92/181 = **50.8%** (95%CI ±7.3pp) |

读法（你的产品 FN > FP，漏屏=永久丢记忆）：
- 从 0.50 → 0.40，**test FN 31.5%→26.4%（少漏 8 张正例）**、**person+screen 召回 50.8%→61.9%（+11pp）**；
  代价是 **test FP 17.6%→26.2%、noscreen 探针 FP 23.8%→30.2%（+6.4pp）**。
- 即"换探针 FP 砍半（vs 旧 v4 的 51%）"后，**在 task2 内部再往低阈值走**，每降 0.05 大致是
  "召回/FN 改善 ~5–6pp ↔ 两类 FP 抬升 ~3–6pp"。三档的 noscreen FP（24–30%）都仍远低于旧 v4 的 51%。
- **若按 FN 优先**：0.40 给最低 FN（26.4%）、最高 person+screen 召回（61.9%），noscreen FP 仍仅 30%。

> 注：person+screen 召回的绝对值比旧 51 张小探针（@0.50 曾读到 60.8%）**低约 10pp**——
> 小探针偏乐观且噪声大（±14pp）。扩充到 181 张后点值下移、CI 收窄到 ±7pp，**这是更可信的真值**。
> 即便如此，下文 §3 显示 task2 仍在"FP vs 召回"曲线上帕累托压住 v4。

## 2. 完整 test 阈值扫描（keras，evaluate.py --pr-sweep，节选）

| thr | prec | recall | F1 | FN率 | FP率 |
|---|---|---|---|---|---|
| 0.35 | 0.660 | 0.780 | 0.715 | 0.220 | 0.305 |
| 0.40 | 0.680 | 0.736 | 0.707 | 0.264 | 0.262 |
| 0.45 | 0.711 | 0.711 | 0.711 | 0.289 | 0.219 |
| 0.50 | 0.747 | 0.686 | 0.715 | 0.315 | 0.176 |
| 0.55 | 0.805 | 0.648 | 0.718 | 0.352 | 0.119 |

（F1 在 0.40–0.55 平台基本持平 0.707–0.718，故 F1 不区分这几档；区分的是 FN/FP 的取舍。）

## 3. 扩充探针上的 vs‑v4 公平对比（int8，同 181/235 张探针）

person+screen 召回（n=181）：

| 阈值 | v4 召回 | task2 召回 |
|---|---|---|
| 0.40 | 82.9% | 61.9% |
| 0.45 | 77.9% | 55.2% |
| 0.50 | 73.5% | 50.8% |
| 0.55 | 69.1% | 42.0% |

noscreen 探针 FP（n=235）：

| 阈值 | v4 FP | task2 FP |
|---|---|---|
| 0.40 | 60.9% | 30.2% |
| 0.50 | 53.6% | 23.8% |
| 0.55 | 51.1% | 21.7% |
| 0.60 | 47.7% | — |
| 0.65 | 43.4% | — |
| 0.70 | 37.0% | — |

**帕累托判定（同口径同探针）**：v4 的 noscreen FP **任何阈值都 ≥37%**（@0.70），而 task2 **最差也只 30.2%**（@0.40）——
两者 FP 区间**不重叠**。在匹配召回 ~62% 处：task2 FP 30%，v4 需 ~0.62 阈值、FP ~45%。
→ **task2 在"noscreen FP vs person+screen 召回"曲线上仍全程帕累托占优**，扩充探针后结论不变（实为更保守）。
注意：同**阈值**下 task2 召回低于 v4，是因 task2 整体分数分布下移；公平比较须看匹配工作点，不能看同阈值。

## 4. 数据产物
- task2: `data/processed/probe_personscreen_audit_task2/`、`probe_noscreen_audit_task2/`
- v4 对比: `data/processed/probe_personscreen_audit_v4/`、`probe_noscreen_audit_v4/`
- md: `docs/probes/probe_personscreen_after_expanded.md`、`probe_noscreen_after_task2_thresholds.md`、
  `probe_personscreen_v4_on_expanded.md`、`probe_noscreen_v4_grid.md`
- 探针扩充：51→**182 下载**，剔 1 张与训练正例近重复（corr 0.99，移入 `data/_quarantine_task2_probe/`）→ **181 张干净 held-out**

## 5. 零泄漏核验（扩充后全过）
- `guard_probe_overlap.py`（data/raw 2749 ↔ 两探针 416）：overlap_hits=0 ✓
- probe↔probe（person_screen 181 ↔ person_noscreen 235）：overlap=0 ✓
- `check_leakage.py`（manifest_dedup 2440）：跨 split 0 对、split 内 0 对 ✓
- `probe_fp_test` 内置防泄漏：扩充探针 n_leaked=0 ✓
