# models/ — 守门员模型清单

模型权重文件（`*.keras`）按 `.gitignore` **不入库**（含 `archive/`）。
复现契约 = 训练脚本 + `data/processed/manifest_dedup.csv` / `dedup_*.csv` + 固定 seed。
本 README 记录"哪个是当前最佳、归档了什么"，是可追溯的清单。

## 顶层模型（不同口径，不可直接比 F1——test 分布/边界不同）

- **`gatekeeper_task2_mvp.keras`** —（**当前推荐**，2026-06-25 提为顶层最佳）Task2 扩充负例修复
  人像误报（covariate shift）。在**同一 held-out 探针**上 vs v4_mvp 的关键差异：
  - **noscreen 探针 FP 砍半**：同部署阈值 0.55，51.1%→**21.7%**（int8，235 张）；
    匹配召回 ~62% 处 v4 FP ~45% 而 task2 仅 30%——**"FP vs person+screen 召回"曲线上帕累托全程占优**。
  - **诚实代价：5-seed test F1 0.757→0.734（−0.023）**、FN 0.230→0.290（test 集因补 558 负例而组成改变，
    与 v4 非严格同口径）。这是"用 FP 换 FN"的再平衡，不粉饰。
  - int8 全 TFLM 白名单(11 算子)、全 int8、32.4KB；ESP32 预算同 v4。
  - **部署阈值未冻结**：见 `docs/threshold-tradeoff.md` 三阈值对照表（FN/FP/探针 trade-off），
    待定。person+screen 召回经扩充探针（181 张，CI ±7pp）实测，绝对值低于早期 51 张小探针估计——
    详见该表 §1 注与 §3 帕累托对比。
- `gatekeeper_v4_mvp.keras` —（Task2 前的最佳）阶段 3.4-B `v4_narrow`，收窄回 MVP 边界 1752。
  5-seed F1 **0.757±0.012**、FN 0.230；单 seed test@0.55 F1 0.783。
  剔除了手机app/TV菜单/商品包装等歧义硬负例（见 `docs/label-scope.md` 的边界定义）。**人像误报偏高**
  （noscreen 探针 FP @0.55 = 51.1%）是被 task2 修复的主要缺陷。
- `gatekeeper_v2_best.keras` — 阶段 3.3 `p33_screen`，原 1518 清晰分布。test F1 0.760（@0.45）。
- `gatekeeper_v3_robust.keras` — 阶段 3.4 `v3_screen`，宽边界 1950（含歧义硬负例）。
  test F1 0.736（@0.4）；聚合 F1 被歧义负例拖低，留作"宽边界鲁棒性"参考。

三者 ESP32 预算均达标（激活 72KB / int8 24.3KB / 全 TFLM 白名单）。
指标与判定见 `docs/gatekeeper-training-log.md`。**是否进 Task2 硬件实测待定。**

> ⚠️ **修正（2026-06-18，导出实测）**：上行"全 TFLM 白名单"原为 `model.py` 的**静态算子核算**，
> 直到 2026-06-18 才首次真实导出 .tflite 验证。结论需加限定：**仅当以固定 batch=1 导出时白名单成立**；
> 默认动态 batch(-1) 导出会引入 SHAPE/STRIDED_SLICE/PACK 三个非白名单算子（flatten 的动态 Reshape 所致）。
> 导出脚本 `scripts/export_tflite.py` 已固定 `batch_shape=(1,96,96,1)`。`gatekeeper_v4_mvp_int8.tflite` 实测：
> 9 个算子全部白名单、全 int8、权重数据缓冲 24.4KB（与 24.3KB 估算吻合）、量化近无损（ΔF1 ±0.007 内）。
> .tflite 同 .keras 一样按 `.gitignore` 不入库，部署需手动传到派/板。
> **仍未做**：ESP32 真机验证（tensor arena 实占 ≤ 片上 SRAM、TFLM kernel 数值一致性、实测延迟/功耗）——见 TODO。

## 归档（`models/archive/`，一次性实验，不删只移）

| 文件 | 来源 |
|---|---|
| gatekeeper_v1 / r1 / r2 | 第一版（泄漏数据上训练，已作废） |
| gatekeeper_dedup_v1 | 阶段 2 dedup 基线（test F1 0.659@0.5） |
| gatekeeper_p32_posmult15 / posmult20 / focal | 阶段 3.2 class weight/focal（未超 3.1） |
| gatekeeper_p33_screen / screen_pm15 / screen_focal | 阶段 3.3 增强实验（screen 为最佳，已提为 v2_best） |
| gatekeeper_smoke / timing_probe | 流水线烟测/计时探针 |

> 注：阶段 3.4 补数据后将训练 v3，若更优则提为新顶层最佳、本表更新、v2 归档。
