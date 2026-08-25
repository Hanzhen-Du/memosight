# 守门员模型可信度校核报告（Task1 · 阶段 1）

> **Summary (EN).** Credibility audit of the first reported numbers. Perceptual hashing plus
> pixel-level confirmation found 132 near-duplicate pairs / 88 duplicate groups, 49 of them
> spanning train/val/test splits and 48 of those purely positive-class. Root cause: the
> Pexels downloader fetched the same stock photo under several keywords, so copies landed in
> different class folders and then in different splits. After connected-component dedup and
> a stratified re-split, cross-split leakage is 0. The corrected baseline is F1 0.756 ± 0.024
> over 5 seeds — the earlier "FN and FP both below 0.30" claim was retracted as an artefact
> of leakage plus single-seed luck.

日期：2026-06-17 
方法学：沿用第一版"如实记录失败 → 诊断 → 下药"。所有指标标明 split；**test 为最终裁定，val 用于调参**。

> **基线声明（重要）**：自阶段 1 起，所有指标均基于 **dedup 去重清单**（`data/processed/manifest_dedup.csv` 及 `dedup_*.csv`）。**原始 1628 张 split 已作废，不再用于任何裁定。**

---

## 0. 结论速览（TL;DR）

| 问题 | 结论 |
|---|---|
| train/val/test 之间有数据泄漏吗？ | **有，且不轻**。49 个重复组跨 split，其中 48 个是纯正例组。 |
| 0.80 的 **accuracy** 可信吗？ | **可信、且稳定**。去重重切分后 5 seed = **0.790 ± 0.014**，与原 0.81 基本一致。 |
| 第一版报的 **FN 0.277 / FP 0.130** 可信吗？ | **不可信，偏乐观**。泄漏（几乎全在正类）虚高了 recall、压低了表观 FN。去重后同 seed=42 的 FN 实为 **0.337**。 |
| 真实 **F1** 是多少？ | 第一版从未报过 F1。去重后真实 F1 = **0.756 ± 0.024**，距 0.85 目标有实打实的差距。 |

**一句话**：准确率 0.80 是真的、稳的；但"FN/FP 都 < 0.30"的乐观结论是泄漏（+ 单次 seed 运气）造出来的假象。真正要解决的是 F1≈0.76、且 recall 在不同切分下不稳（0.66–0.85）。

---

## 1. 数据泄漏检查

### 1.1 方法
脚本 `scripts/check_leakage.py`，两道关卡（先廉价后精确）：
1. **感知哈希粗筛**：对全量 1628 张处理后 96×96 灰度图算 pHash(DCT) + dHash，全对比较 Hamming 距离，挑出 pHash 汉明 ≤ 6 的候选对。
2. **像素级二次确认**：对候选对算 96×96 像素的 Pearson 相关系数 + 归一化 MSE，仅相关性 ≥ 0.90 才判真·近重复，滤掉哈希碰撞误报。

去重用 `scripts/dedup_resplit.py`：对确认的近重复对建**连通分量**（并查集），每组留一张代表（字典序最小路径，确定性可复现），再按与 `prepare_dataset.py` 一致的"按大类分层 70/15/15"重切分。

> 注：未用 `imagehash` 等第三方库（环境未安装该库），pHash/dHash 用 `numpy + cv2.dct` 自实现，标准做法。

### 1.2 发现
- pHash 候选对 134 → 像素确认真近重复 **132 对**（近乎一致 corr≥0.999：125 对）。
- 按连通分量：**88 个重复组**，涉及 198 张图，最大组 3 张（66 组成对、22 组三连）。
- **跨 split 的重复组（= 泄漏）：49 个**，其中 **48 个纯正例组**、1 个含负例。
- 跨 split 重复对按 split 计数：`test↔train 33`、`train↔val 27`、`test↔val 4`。

### 1.3 根因（已确证）
**83/88 个重复组的成员共享同一 Pexels 图片 ID**（文件名中段数字）。即 `download_images.py` 用多个关键词抓图时，**同一张库存图被不同关键词重复下载**，落入不同的 positive 子类目录（如同一图同时进 `powerpoint_slide` / `classroom_projector_slides` / `projector_screen_presentation`）。而分层切分按"来源大类（positive）"整体切，于是同图副本被分散到 train/val/test，造成训练集"见过"测试集的图。泄漏集中在正类，故**虚高正类表现**。

### 1.4 去重 + 重切分的影响
| | 全量 | 去重后 |
|---|---|---|
| 总数 | 1628 | **1518**（移除 110） |
| 正例 | 751 | 644（移除 **107**） |
| 负例 | 877 | 874（移除 3） |

去重后 seed=42 重切分：train 1061 / val 226 / test 231（正负比 ~0.737，原 0.856——因移除了大量正例重复）。
**复查确认**：对去重后的三 split 重跑泄漏检查 → 跨 split 重复 **0 对**、内部重复 0 对，泄漏根除。

去重后清单：`data/processed/manifest_dedup.csv`；候选对明细：`docs/results/leakage_candidates.csv`。

---

## 2. 方差校核（0.80 是稳定值还是单次幸运？）

### 2.1 方法
脚本 `scripts/run_variance.py`：在**去重后**清单上，用 5 个 seed `{42,1,7,123,2024}` 各自重切分 + 重训（先消泄漏再测方差，否则方差本身被污染）。
**冻结最佳配置**（第一版修复后的组合）：`bn_momentum=0.9, patience=15, start_from_epoch=20, epochs=80, augment=True, class_weight=balanced, lr=1e-3, monitor=val_loss(restore best)`。阈值 0.5（argmax）。

### 2.2 结果（test split，正类=1=记）

| seed | epochs | accuracy | F1 | recall | precision | FN rate | FP rate |
|---|---|---|---|---|---|---|---|
| 42 | 50 | 0.7922 | 0.7303 | 0.6633 | 0.8125 | **0.3367** | 0.1128 |
| 1 | 54 | 0.8009 | 0.7830 | 0.8469 | 0.7281 | 0.1531 | 0.2331 |
| 7 | 45 | 0.8052 | 0.7783 | 0.8061 | 0.7524 | 0.1939 | 0.1955 |
| 123 | 60 | 0.7835 | 0.7619 | 0.8163 | 0.7143 | 0.1837 | 0.2406 |
| 2024 | 59 | 0.7662 | 0.7245 | 0.7245 | 0.7245 | 0.2755 | 0.2030 |
| **均值±std** | | **0.7896 ± 0.0139** | **0.7556 ± 0.0241** | **0.7714 ± 0.0675** | **0.7464 ± 0.0354** | **0.2286 ± 0.0675** | **0.1970 ± 0.0455** |

机读结果：`docs/results/variance_results.json`。

### 2.3 解读
- **accuracy 0.79 ± 0.014 → 稳定**。不是单次幸运；与第一版 0.81 在噪声范围内一致。准确率这个数可信。
- **recall / FN 方差大**（recall 0.66–0.85，std 0.068）。决策边界位置在不同切分下漂移明显——这恰好说明 **Phase 3 的"阈值调整"是对症的**（accuracy 稳但 precision/recall 配比不稳）。
- **泄漏确实掩盖了 FN 问题**：去重后 seed=42 的 FN = 0.337，比泄漏版同条件报的 0.277 更差；FP 率整体也从 0.130 升到 ~0.197。即第一版"FN/FP 都 < 0.30"是被污染数据 + 该 seed 运气共同造出的假象。
- **真实 F1 ≈ 0.756**，离 0.85 目标差约 0.09，是实打实要补的差距，不是测量误差。

---

## 3. 对后续阶段的影响（交接给 Phase 2/3）

1. **基线口径切换**：后续所有训练/评估改用去重清单与去重切分（`data/processed/manifest_dedup.csv` + `dedup_*.csv`），不要再用被污染的原始 `train/val/test.csv`。
2. **校正目标基线**：真实起点是 **F1 0.756 / FN 0.229（均值）**，不是第一版的 0.277。Phase 3 的"FN 下降"应以此为基准衡量。
3. **优先级佐证**：recall 方差大 + accuracy 稳 → Phase 3 第 1 步"决策阈值扫描"零成本且最对症，应先做；评估工具 `scripts/evaluate.py` 已支持 `--pr-sweep` 阈值扫描与任意阈值复评。
4. **数据侧根因**：若要从源头防止再次泄漏，应在 `download_images.py` 下载阶段按 Pexels 图片 ID 去重（待与已确认是否动下载脚本）。

---

## 4. 失败/诊断记录（方法学留痕）

- **失败**：第一版报的 FN 0.277 被当作"达标"，实为泄漏 + 单 seed 运气的乐观估计。
- **诊断**：感知哈希 + 像素确认定位 64 跨 split 重复对，连通分量 + Pexels-ID 共享率 83/88 锁定"同图多关键词下载"根因。
- **下药**：连通分量去重（留 1 张）+ 分层重切分，泄漏归零；5-seed 方差校核给出可信基线（acc 稳、F1 0.756、FN/recall 不稳）。

## 附：本阶段新增脚本（均无新增依赖，复用 numpy/pandas/cv2/tf）
- `scripts/check_leakage.py` — 跨 split 重复/泄漏检查（pHash+dHash 粗筛 + 像素确认）。
- `scripts/dedup_resplit.py` — 连通分量去重 + 分层重切分（`build` / `split` 两子命令）。
- `scripts/evaluate.py` — 完整指标一键复评（acc/precision/recall/F1/混淆/FN/FP，支持任意阈值与 `--pr-sweep`）。
- `scripts/run_variance.py` — 5-seed 方差校核 harness。
