# Task2b 重训报告 — 针对性室内负例扩充：前后对比（诚实版）

> **Summary (EN).** A negative result, reported in full. Adding 249 further "empty indoor
> room / blank screen" negatives failed its primary objective: probe false-positive rate
> moved 0.331 → 0.314, well inside ±0.09 noise, while FN rose 0.187 → 0.262 and recall fell.
> The added data only helped the distribution it matched (a separate empty-room probe
> improved 0.328 → 0.250) and did not transfer to the people-plus-indoor distribution that
> actually fails. Conclusion: the model was not promoted.

完成 2026-06-26　结构 C_wide_uniform(16,32,64,64)　5-seed `[42,1,7,123,2024]` @0.40　int8 部署口径。

## 0. 结论（TL;DR）：**主目标未达成，且换来召回/FN 退化——不建议提升该模型。**

- **noscreen FP（原始问题：人+室内场景误触发）没降**：0.331 → 0.314（Δ−0.017，远在 ±0.09 噪声内）。
- **代价明确**：test FN 0.187→0.262（+0.074，更多漏报）、test recall 0.813→0.738、person+screen 召回 0.582→0.521、test F1 0.769→0.704。
- **唯一正向**：在**全新留出的"空室内环境"泛化探针 indoor_env_v2** 上，FP 0.328 → 0.250（seed42 对 seed42，−0.078）——
  即新负例**在自己的分布（空房间/空白屏）上学到了**，但**没迁移到原始 noscreen 的"人+室内"分布**。
- 诊断：我补的是**空房间/空白屏状表面**负例；而 noscreen 的残余 FP 是**有人 + 室内环境**。空房间负例改善了空房间识别，
  却没解决"有人时仍被室内环境骗"。叠加 +249 负例使类别更偏负（1387→1636），把决策边界推向"不记"→ FN↑、召回↓，
  却没换来 noscreen FP 下降。**净结果：不划算。**

## 1. 数据与流程（已核验）
- 合并：prepare_dataset 重建 manifest（2998），dedup_resplit **复刻 phase 3.4→B 边界收窄**（剔 5 个歧义子类 198 行），
  全量感知去重 → **dedup 2689（1636 neg / 1053 pos）**，净 **+249 负例**，正例不变，新负例零新增近重复。
- 泄漏：check_leakage(manifest_dedup) 跨/内 split **0/0**；guard 训练 vs 三探针**零重叠**。
- 三探针均为固定 held-out（noscreen 235 / person_screen 181 / indoor_env_v2 64），**前后可比**；
  test/val 因重切分+新负例进入而**分布改变，非严格可比**（仅作参考，故重点看探针）。

## 2. 前后对比（5-seed mean ± std）

| 指标 | task1（前） | task2b（后） | Δ | 可比性 | 判读 |
|---|---|---|---:|---|---|
| **noscreen_fp** ↓ | 0.331 ± 0.091 | 0.314 ± 0.093 | **−0.017** | 固定探针·可比 | ❌ 噪声内，**主目标未达成** |
| **screen_recall** ↑ | 0.582 ± 0.095 | 0.521 ± 0.118 | **−0.061** | 固定探针·可比 | ❌ 召回退化 |
| indoor_env_fp ↓ | 0.328¹ | 0.244 ± 0.066 | **−0.08** | 固定探针·可比 | ✅ 唯一正向（泛化探针） |
| test_f1 ↑ | 0.769 ± 0.015 | 0.704 ± 0.034 | −0.066 | ⚠️ 重切分·不严格可比 | 退（部分受测试集变难影响） |
| test_fn ↓ | 0.187 ± 0.060 | 0.262 ± 0.056 | +0.074 | ⚠️ 同上 | 退·更多漏报 |
| test_recall ↑ | 0.813 ± 0.060 | 0.738 ± 0.056 | −0.074 | ⚠️ 同上 | 退 |
| test_fp ↓ | 0.226 ± 0.054 | 0.234 ± 0.074 | +0.008 | ⚠️ 同上 | 平 |
| val_f1 | 0.770 ± 0.020 | 0.719 ± 0.018 | −0.051 | 新池 val·不可比 | （新分布上也降） |

¹ indoor_env_v2 探针是 task2b 才建的，task1 无 5-seed 基线；此处用**两个 seed42 部署 int8 模型**在同 64 张探针上现测
（task1 0.3281 vs task2b 0.2500），seed42 对 seed42 公平。task2b 列 0.244 为 5-seed 均值（供参考）。

## 3. 为什么 noscreen 没降、indoor_env_v2 却降了？（核心洞察）
**两个探针的分布不同，新负例只覆盖了其中一类：**
- `indoor_env_v2`（**空**室内环境：大堂/图书馆/休息区/前台，无人）≈ 我补的负例（空房间/空白屏）→ **同分布，学到了**，FP 0.328→0.250。
- `noscreen`（**有人** + 室内：办公室同事交谈/会议室说话/家庭客厅）→ 画面含**人**，与"空房间"负例**不同分布**，FP 没动。

> 阶段一诊断说"误触发由环境驱动、人脸反相关"，于是我补了**纯环境（空房间）**负例。结果证明：
> 这对**空环境**有效，但 noscreen 的残余 FP 出现在**有人**的室内画面里——模型在"有人"时仍被背景的
> 室内/屏状几何骗，而纯空房间负例没教它这一点。**修对了一半的分布，没对准真正出错的那一半。**

这也与 task1 结论一致并加深：守门员在 96×96 灰度下，把"亮室内+矩形屏状区域"当触发信号，
而该信号在正类（有文字屏的办公室）与负类（有人但无文字的同款办公室）之间**高度共享**；
仅靠加同类负例，要么不够（noscreen 没动），要么以牺牲召回为代价（FN↑）。**可能是表征/分辨率层面的界，不只是数据量。**

## 4. 推荐
1. **不要提升 task2b 模型**：它在固定探针上没改善主目标 noscreen FP，且召回/FN 明显退化。
   **保留 task1 `gatekeeper_task1_C_wide_uniform` 为当前守门员。**（task1 模型文件仍在，未被覆盖。）
2. **训练池可回退**：合并把 `data/processed/manifest_dedup.csv` 换成了 task2b 池（2689）；
   旧池备份在 `data/processed/_pretask2b_backup/`，若要复现 task1 口径可 cp 回。**是否回退由你定**（我未自动改回）。
3. **若继续攻 noscreen FP**，方向（需你拍板，均非本轮自动做）：
   - (a) **对准分布**：补"**有人** + 室内办公/会议/客厅 + 无文字屏"负例（而非空房间），直接覆盖 noscreen 出错的那类；
     但风险高（有人+办公室极易混入可读屏=正类，污染更难 QC）。
   - (b) **提分辨率**（128/160）让模型有机会区分"空白 vs 含文字"的屏——但**破 ESP32 预算**，需重估边界。
   - (c) **两段式**：守门员先粗筛"有屏状区域"，再接一个极廉价"文字存在性"判别（攻"空白屏 vs 文字屏"这条正负边界）。
   - (d) 接受 FP 地板，转而在 noscreen/功耗-漏报曲线上调阈值权衡。

## 5. 诚实备注
- test/val 退化部分由**重切分 + 新负例使测试集变难**造成，不全是模型变差；故正文以**固定探针**为准绳。
  即便如此，固定探针口径下结论仍为负（noscreen 平、screen 召回退、仅 indoor_env_v2 进）。
- 单 seed 方差大（noscreen 各 seed 0.179–0.430）：本报告所有主结论基于 5-seed 均值，未挑 seed。
- 本轮**未达成"降 noscreen FP"的目标**——按项目方法论如实记录："诚实的没降，胜过挑一个好 seed 的假降"。
