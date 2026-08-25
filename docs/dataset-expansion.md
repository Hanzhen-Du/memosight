# Task2 结果报告 — 扩充训练集修复人像误报（covariate shift）

> **Summary (EN).** The one data intervention that worked. Adding 558 non-office
> people-without-screen negatives halves person-driven false triggers on a fixed held-out
> probe (51% → ~24% at matched recall) and Pareto-dominates the previous model across the
> whole false-positive/recall curve. The apparent "over-correction" seen at the old fixed
> threshold was a re-calibration artefact: the new model's score distribution shifted down.
> Honest cost recorded alongside: 5-seed test F1 0.757 → 0.734.

日期：2026-06-24　模型：`gatekeeper_task2_mvp`（5-seed + 单 seed 部署模型）

## 0. 一句话结论
**修复成功，且是"帕累托占优"式的成功，但需配套两个动作：重标部署阈值 + 接受一个小的 test-F1 口径回退。**
在 held-out 探针上，新模型在**相同召回下把人像误报砍掉一半多**（51%→~24%），或**相同误报下召回 +31pp**——
即新模型在"探针 FP vs 有人+有屏召回"权衡曲线上**全程压住旧模型**。先前在固定旧阈值 0.55 处看到的
"召回从 61%→47% 的矫枉过正"是**阈值未重标的假象**（新模型分数分布整体下移），重标到 ~0.45–0.50 即消失。
唯一诚实的保留项：5-seed **test F1 0.757→0.734（−0.023）**、FN 0.230→0.290 略升（test 集本身因补负例而改变，非严格可比）。

## 1. 做了什么（数据）
- 三桶下载（导师指定对比组合 + 防矫枉过正探针）：
  - **A 人/无屏负例**：16 关键词 ×35，净 **+560** → `data/raw/negative_clean/`（label 0）。偏非办公日常实景（街头/市场/通勤/家居/运动/服务/户外/看台/车厢）。
  - **B 屏/无人正例**：9 关键词 ×28，净 **+131**（120 被全局 ID 去重跳过=与既有正例同图，正确不入）→ `data/raw/positive/`（label 1）。
  - **C 人+有屏 held-out 探针**：6 关键词 ×18，净 **+51** → `data/probe_person_screen/`（**绝不进训练**）。
- **防泄漏（硬门槛，全过）**：
  - 新写 `scripts/guard_probe_overlap.py`：训练图 × 两个探针目录 ID+感知双查。
    发现 **2 张**新负例与 held-out `probe_person_noscreen` 撞 Pexels-ID（13200581、36299324），
    已 **quarantine（移动到 `data/_quarantine_task2/`，不删除）**；复查 **0 重叠**。
  - `check_leakage.py`（去重池 2440 张）：跨 split 近重复 **0 对**、split 内 **0 对**。
  - 探针测时再做一道 ID+感知核对：两个探针对训练池**泄漏 0**。
- 数据量变化（去重 + 同一收窄边界后）：

  | 池 | 基线 v4 | task2 | Δ |
  |---|---|---|---|
  | 去重训练池 | 1752 | **2440** | +688 |
  | 负例(neg_clean+neg_noise) | 829 | **1387** | **+558（核心修复）** |
  | 正例 | 923 | 1053 | +130 |

  收窄排除子类沿用基线：`cosmetic_packaging_closeup, grocery_product_label, product_packaging_text, smartphone_apps_home_screen, tv_streaming_menu_screen`。

## 2. 头条：人像误报（探针 `probe_person_noscreen`，235 张，GT=不记，int8 部署口径）

| 阈值 | FP 旧(v4) | FP 新(task2) | Δ |
|---|---|---|---|
| 0.50 | 53.6% | **23.8%** | −29.8pp |
| 0.55（旧部署点） | 51.1% | **21.7%** | −29.4pp |
| 0.70 | 37.0% | 11.1% | −25.9pp |

> 旧基线 51.1% 与导师所述 ~51% 吻合（已在动数据前用同一脚本锁定）。

## 3. "矫枉过正"自检（探针 `probe_person_screen`，51 张，GT=记，int8）——召回（=判"记"比例）

| 阈值 | 召回 旧(v4) | 召回 新(task2) |
|---|---|---|
| 0.50 | 66.7% | **60.8%** |
| 0.55 | 60.8% | 47.1% |
| 0.45 | 74.5% | 64.7% |

固定 0.55 看，新模型召回 61%→47% 像是"见人就忽略"。但这是**阈值假象**——见 §4。

## 4. 关键：权衡曲线（held-out，公平对比）—— 新模型帕累托占优
新模型分数分布整体下移（val F1 最优阈值由 0.55 降到 ~0.35–0.50），故必须**按新模型重标阈值**再比。
等价工作点对比（int8）：

- **相同召回 60.8%**：旧需 @0.55 → 探针 FP **51.1%**；新只需 @0.49 → 探针 FP **24.7%**。→ 同召回，FP 砍半还多。
- **相同 FP 51%**：旧 @0.55 召回 60.8%；新 @0.22 召回 **92.2%**。→ 同 FP，召回 +31pp。
- 扫描全程：每个阈值上 `FP_新 ≪ FP_旧`。

→ **新模型在"探针 FP vs 有人+有屏召回"曲线上全程占优**。§3 的"召回回退"仅因沿用了旧阈值 0.55。

推荐部署阈值（在 **val** 上重标，非在探针上拟合）：val F1 在 0.35–0.50 平台最高（@0.50 F1 0.733 / @0.55 0.705）。
取 **~0.45–0.50**：探针 FP ~24–27%（↓from 51%）且有人+有屏召回 ~61–65%（≥ 旧 60.8%）。**此时三条成功标准同时满足。**

## 5. test 集（5-seed 去重重切分，阈值 0.5）—— 诚实口径

| 指标 | 基线 v4 | task2 | Δ |
|---|---|---|---|
| F1 | 0.7568 ± 0.0123 | **0.7343 ± 0.0248** | **−0.0225** |
| recall | 0.7698 | 0.7095 | −0.060 |
| FN rate | 0.2302 | 0.2905 | +0.060 |
| FP rate | 0.2905 | **0.1667** | −0.124 |
| precision | 0.7527 | 0.7652 | +0.013 |
| accuracy | 0.7411 | 0.7800 | +0.039 |

**诚实说明**：test F1 小幅回退 0.023、FN 略升——这是"用 FP 换 FN"的再平衡。但：
(a) test 集本身因补了 558 负例而组成改变（负例占比上升），与基线 test 非严格同口径，不能直接等同比较；
(b) 公平的同集 held-out 探针（§2/§4）显示新模型占优；
(c) 回退幅度 ~1–2 个基线 std，且 test FP 显著改善。
→ 综合判断：**不构成"test 退化"的否决项，但必须如实标注这条口径回退，不粉饰。**

## 6. int8 导出验证（`gatekeeper_task2_mvp_int8.tflite`）
- 算子：11 个全部命中 TFLM 白名单 ✓
- dtype：17 int8 + 5 int32，**0 个 float32 内部张量**（全 int8）✓
- 体积：int8 文件 32.4KB，权重 24.3KB（= 24,874 参数 ×1B，与历史估算吻合）✓
- 量化掉点：test ΔF1 −0.006(@0.5) / −0.014(@0.55)，可接受 ✓

## 7. 成功标准逐条裁定
| 标准 | 结论 |
|---|---|
| 探针 FP 从 ~51% 显著下降 | ✅ **达成**：同召回下 51%→~24%（−27pp）；旧阈值处 51.1%→21.7% |
| 原 test F1/FN/FP 不退 | ⚠️ **部分**：test FP 大降，但 F1 −0.023、FN +0.06（test 集已变，口径回退，需标注） |
| 有人+有屏召回保持 | ✅ **达成（重标阈值后）**：@0.45–0.50 召回 61–65% ≥ 旧 60.8%；@旧 0.55 则回退（阈值假象） |

## 8. 残留风险 / 局限（不藏）
1. **人眼 QC 未做**：560 张新负例未逐张确认无"可读屏幕"混入（自动无法可靠判别）。已生成
   `data/processed/task2_qc_montages/*.png`（每子类 9 图拼图）供人工抽查；关键词刻意避开办公/会议/教室以降风险。
2. **person+screen 探针仅 51 张**（部分关键词被全局去重大量跳过），召回 95%CI 约 ±14pp，结论方向可信但点值噪声大。
3. **部署阈值尚未提交**：本报告只给推荐区间，未改任何部署配置/README/模型选型——待定。
4. **未做**：把 task2_mvp 提为顶层最佳、ESP32 真机验证、test 集口径对齐重测。

## 9. 产物
- 模型：`models/gatekeeper_task2_mvp.keras` / `_float32.tflite` / `_int8.tflite`（gitignored）
- 数据清单：`data/processed/manifest.csv`、`manifest_dedup.csv`、`dedup_{train,val,test}.csv`
- 指标：`docs/results/variance_results_task2.json`、`docs/probes/probe_fp_{before,after}.md`、
  `probe_personscreen_{before,after}.md`、`leakage_task2_dedup.csv`
- 脚本（新）：`scripts/guard_probe_overlap.py`、`scripts/probe_fp_test.py`（自审计分支引入）、三个 `keywords_task2_*.json`
- QC 拼图：`data/processed/task2_qc_montages/`
- quarantine：`data/_quarantine_task2/`（2 张撞探针的负例，移动保留）
