# 标签边界定义（守门员二分类的正/负例口径）

> **Summary (EN).** The binary label definition the gatekeeper is trained and judged against,
> and the record of one deliberate narrowing of it. Ambiguous "text but not our scene"
> subclasses (phone app UI, TV menus, product packaging) were pulled out of both the positive
> class and the hard-negative pool after an experiment showed they cost aggregate F1
> (0.756 → 0.70) without narrowing seed variance, while the model improved on the clean
> distribution. Diagnosed as label ambiguity rather than insufficient data. Numbers measured
> under the narrowed definition are **not** directly comparable with numbers measured under
> the wider one.

本文件是守门员标签口径的**唯一权威定义**。所有训练 / 评估报告在引用「MVP 原定义口径」时都指本文件。

---

## 1. 正类（记 = 1）

首发触发场景的**有用文字屏幕**：

- 投影幕
- 电脑屏文字
- 课件 / PPT
- 白板
- 文档页
- 代码屏

## 2. 明确排除（不算正类触发，也**不**作为训练负例里的歧义硬负例）

- 手机 App 界面文字
- 电视 / 流媒体菜单文字
- 商品包装 / 标签文字

这三类「有文字但非首发场景」在语义上边界模糊，纳入会让二分类的决策边界变糊。
被剔除的子类归档在 `data/processed/manifest_out_of_scope.csv`——**图不删，只是不进训练/评估**。

## 3. 有效负类（清晰的「不该记」）

招牌 / 路牌、书脊、手机锁屏、正在放视频的屏幕，以及无文字的风景 / 人像 / 室内 / 食物等。

---

## 4. 这次收窄的依据（2026-06-17，阶段 3.4 → B）

补数据实验引入上述歧义硬负例后：

| 指标 | 宽边界（含歧义负例） | 收窄后 |
|---|---|---|
| 聚合 test F1 | 0.70 | **0.756** |
| 5-seed 方差 | 未收窄 | 未收窄 |
| 正例 FN（清晰旧分布） | 0.206 | **0.135** |

即：加入歧义负例**同时**拉低了聚合 F1 且没有换来方差收窄，而模型在清晰分布上反而变好。
诊断结论是**边界歧义**，不是样本量不足，故收窄回 §1 的首发定义。

完整过程见 `docs/gatekeeper-training-log.md`。

> ⚠️ **口径提示**：收窄后的指标为「MVP 原定义口径」，与含歧义负例的「宽边界口径」**不可直接比较**。
> 各报告中凡涉及跨口径对比处，均已在表内单独标注。
