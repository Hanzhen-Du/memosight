# Task2b 阶段一 · noscreen 探针 FP 误判诊断

> **Summary (EN).** Per-image analysis of all 235 held-out probe images (59 false positives).
> Errors cluster by room type — office conversation 0.559, living room 0.52, meeting room
> 0.324 — and fall to ~0 outdoors and in restaurants. False positives are brighter (+0.186)
> and hit the screen-like-rectangle detector twice as often (+0.118), but contain *fewer*
> faces (−0.093). This is the measurement that redirected the project away from collecting
> more people photographs.

模型：`models/task1_candidates/gatekeeper_task1_C_wide_uniform_int8.tflite`（C_wide_uniform int8，task1 胜出）　阈值 **@0.4**　口径：int8 部署预处理（cv2 灰度→resize96 INTER_AREA→量化）。
探针：noscreen **235** 张（leak 核对剔除 0 张，按 Pexels-ID）。⚠️ 探针仅评估，**不入任何训练集**。

## 总览
- FP（被误判为「记」，score≥0.4）：**59/235 = 0.251**
- 正确拒识：176/235
- 全体分数：min 0.000 / median 0.160 / max 0.922

## 按场景聚合（FP 率降序）
| 场景（子目录） | n | FP | FP率 | 平均分 | 平均人脸数 |
|---|---:|---:|---:|---:|---:|
| office_colleagues_conversation | 34 | 19 | 0.559 | 0.457 | 0.41 |
| family_home_living_room | 25 | 13 | 0.52 | 0.383 | 0.08 |
| people_meeting_room_talking | 34 | 11 | 0.324 | 0.333 | 0.29 |
| coworkers_standing_meeting | 29 | 7 | 0.241 | 0.258 | 0.59 |
| group_friends_indoor_candid | 29 | 6 | 0.207 | 0.197 | 0.9 |
| people_street_candid | 25 | 2 | 0.08 | 0.15 | 0.2 |
| friends_cafe_group | 29 | 1 | 0.034 | 0.076 | 0.52 |
| people_restaurant_dining | 30 | 0 | 0.0 | 0.098 | 0.23 |

## FP vs 正确拒识 · 维度对比（均值）
| 维度 | FP（n=59） | 正确拒识（n=176） | 差异 |
|---|---:|---:|---:|
| 亮度 | 0.601 | 0.415 | +0.186 |
| 对比度 | 0.253 | 0.231 | +0.022 |
| 人脸数代理 | 0.339 | 0.432 | -0.093 |
| 类屏矩形命中率 | 0.237 | 0.119 | +0.118 |

## FP 清单（score 降序，全部 59 张）
| # | score | 场景 | 文件 | 人脸 | 亮度 | 类屏 |
|---:|---:|---|---|---:|---:|---:|
| 1 | 0.9219 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0024_4343205.jpeg | 0 | 0.6117 | 0 |
| 2 | 0.9102 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0010_4343207.jpeg | 0 | 0.6157 | 0 |
| 3 | 0.8984 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0030_7432116.jpeg | 0 | 0.7474 | 0 |
| 4 | 0.8828 | family_home_living_room | family_home_living_room/family_home_living_room_0022_8120951.jpeg | 0 | 0.7223 | 1 |
| 5 | 0.8789 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0027_7845080.jpeg | 1 | 0.5424 | 0 |
| 6 | 0.8594 | coworkers_standing_meeting | coworkers_standing_meeting/coworkers_standing_meeting_0006_7653572.jpeg | 0 | 0.7476 | 1 |
| 7 | 0.8477 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0017_6950047.jpeg | 0 | 0.5993 | 0 |
| 8 | 0.8125 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0028_7432114.jpeg | 0 | 0.6712 | 0 |
| 9 | 0.7969 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0012_8204404.jpeg | 0 | 0.7328 | 1 |
| 10 | 0.7891 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0023_7964185.jpeg | 1 | 0.57 | 0 |
| 11 | 0.7812 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0019_6950093.jpeg | 0 | 0.5016 | 1 |
| 12 | 0.7773 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0014_7495168.jpeg | 1 | 0.6464 | 1 |
| 13 | 0.7461 | family_home_living_room | family_home_living_room/family_home_living_room_0006_8120953.jpeg | 0 | 0.6703 | 0 |
| 14 | 0.7266 | coworkers_standing_meeting | coworkers_standing_meeting/coworkers_standing_meeting_0024_7652461.jpeg | 0 | 0.5583 | 1 |
| 15 | 0.7266 | family_home_living_room | family_home_living_room/family_home_living_room_0025_8120623.jpeg | 0 | 0.6591 | 0 |
| 16 | 0.707 | coworkers_standing_meeting | coworkers_standing_meeting/coworkers_standing_meeting_0021_7993566.jpeg | 0 | 0.649 | 0 |
| 17 | 0.6992 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0009_7644014.jpeg | 0 | 0.6335 | 0 |
| 18 | 0.6914 | family_home_living_room | family_home_living_room/family_home_living_room_0002_3875141.jpeg | 0 | 0.6869 | 1 |
| 19 | 0.6836 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0025_4347461.jpeg | 0 | 0.5232 | 1 |
| 20 | 0.6758 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0020_165907.jpeg | 0 | 0.5245 | 1 |
| 21 | 0.668 | coworkers_standing_meeting | coworkers_standing_meeting/coworkers_standing_meeting_0017_8068143.jpeg | 2 | 0.6775 | 0 |
| 22 | 0.668 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0022_7433844.jpeg | 0 | 0.659 | 1 |
| 23 | 0.6602 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0003_8204399.jpeg | 1 | 0.6939 | 0 |
| 24 | 0.6602 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0024_8133989.jpeg | 0 | 0.4259 | 0 |
| 25 | 0.6406 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0015_8068146.jpeg | 1 | 0.6784 | 0 |
| 26 | 0.6406 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0027_6950159.jpeg | 1 | 0.6551 | 0 |
| 27 | 0.6328 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0026_7964354.jpeg | 1 | 0.5067 | 0 |
| 28 | 0.6172 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0002_7964210.jpeg | 0 | 0.4083 | 0 |
| 29 | 0.6094 | coworkers_standing_meeting | coworkers_standing_meeting/coworkers_standing_meeting_0018_8068161.jpeg | 0 | 0.6255 | 0 |
| 30 | 0.5977 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0007_7433850.jpeg | 0 | 0.6488 | 0 |
| 31 | 0.5898 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0001_8602417.png | 0 | 0.7359 | 1 |
| 32 | 0.5586 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0005_4345107.jpeg | 0 | 0.6223 | 0 |
| 33 | 0.5586 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0029_7433859.jpeg | 0 | 0.5546 | 0 |
| 34 | 0.5508 | family_home_living_room | family_home_living_room/family_home_living_room_0014_36777501.jpeg | 0 | 0.5926 | 0 |
| 35 | 0.543 | family_home_living_room | family_home_living_room/family_home_living_room_0019_17158663.jpeg | 0 | 0.6577 | 0 |
| 36 | 0.5352 | coworkers_standing_meeting | coworkers_standing_meeting/coworkers_standing_meeting_0010_7869114.jpeg | 0 | 0.4971 | 0 |
| 37 | 0.5352 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0009_36393928.jpeg | 2 | 0.6402 | 1 |
| 38 | 0.5352 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0030_8847199.jpeg | 1 | 0.3025 | 0 |
| 39 | 0.5234 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0016_14340485.jpeg | 1 | 0.6145 | 0 |
| 40 | 0.5234 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0017_7793921.jpeg | 0 | 0.713 | 0 |
| 41 | 0.5156 | family_home_living_room | family_home_living_room/family_home_living_room_0003_280239.jpeg | 0 | 0.7144 | 0 |
| 42 | 0.5078 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0031_32441078.jpeg | 1 | 0.6973 | 0 |
| 43 | 0.5 | family_home_living_room | family_home_living_room/family_home_living_room_0023_28272350.jpeg | 0 | 0.7674 | 0 |
| 44 | 0.4766 | family_home_living_room | family_home_living_room/family_home_living_room_0018_35430055.jpeg | 0 | 0.5608 | 0 |
| 45 | 0.4648 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0004_7433930.jpeg | 1 | 0.5215 | 0 |
| 46 | 0.457 | family_home_living_room | family_home_living_room/family_home_living_room_0005_34541788.jpeg | 0 | 0.4529 | 0 |
| 47 | 0.457 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0019_7964507.jpeg | 0 | 0.5119 | 1 |
| 48 | 0.4492 | friends_cafe_group | friends_cafe_group/friends_cafe_group_0021_20140970.jpeg | 0 | 0.4479 | 0 |
| 49 | 0.4492 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0003_8367619.jpeg | 0 | 0.6223 | 0 |
| 50 | 0.4414 | family_home_living_room | family_home_living_room/family_home_living_room_0008_7114188.jpeg | 0 | 0.6906 | 0 |
| 51 | 0.4414 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0008_23496874.jpeg | 0 | 0.6013 | 0 |
| 52 | 0.4336 | family_home_living_room | family_home_living_room/family_home_living_room_0007_8763082.jpeg | 1 | 0.5163 | 1 |
| 53 | 0.418 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0005_7876794.jpeg | 0 | 0.6934 | 0 |
| 54 | 0.418 | people_street_candid | people_street_candid/people_street_candid_0025_33259432.jpeg | 0 | 0.5726 | 0 |
| 55 | 0.4102 | coworkers_standing_meeting | coworkers_standing_meeting/coworkers_standing_meeting_0015_6930265.jpeg | 1 | 0.6252 | 0 |
| 56 | 0.4102 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0021_8826744.jpeg | 3 | 0.5651 | 0 |
| 57 | 0.4102 | people_street_candid | people_street_candid/people_street_candid_0016_32242667.jpeg | 0 | 0.288 | 0 |
| 58 | 0.4023 | family_home_living_room | family_home_living_room/family_home_living_room_0013_6957830.jpeg | 0 | 0.6551 | 0 |
| 59 | 0.4023 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0015_10782244.jpeg | 0 | 0.409 | 0 |

## 借近阈值的「擦边正确拒识」（0.4>score≥0.32，最易翻车）
共 9 张（这些是再补一点同类负例最可能压下去的边缘案例）：

| score | 场景 | 文件 |
|---:|---|---|
| 0.3906 | family_home_living_room | family_home_living_room/family_home_living_room_0017_8583811.jpeg |
| 0.375 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0027_10423482.jpeg |
| 0.3594 | coworkers_standing_meeting | coworkers_standing_meeting/coworkers_standing_meeting_0012_7964369.jpeg |
| 0.3438 | people_street_candid | people_street_candid/people_street_candid_0012_13200581.jpeg |
| 0.3398 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0002_27869785.jpeg |
| 0.3398 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0017_27868438.jpeg |
| 0.3398 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0009_3867842.jpeg |
| 0.3242 | family_home_living_room | family_home_living_room/family_home_living_room_0010_7114420.jpeg |
| 0.3242 | people_restaurant_dining | people_restaurant_dining/people_restaurant_dining_0007_12181619.jpeg |

---

## 共性结论（诚实版）

**口径说明**：本诊断用 seed42 部署产物（单模型），FP=0.251；task1 报告的 noscreen_fp=0.331±0.091 是 5-seed 均值。
0.251 落在该区间内，单模型偏乐观一点，但 FP 的**结构/共性**与模型无关，下述结论稳健。

### 共性 1（最强、可直接落地）：FP 高度集中在「室内办公/会议/居家」场景，户外/餐饮近乎零
按场景 FP 率清晰分两簇：
- **高 FP 簇（0.32–0.56）**：`office_colleagues_conversation` 0.559、`family_home_living_room` 0.52、`people_meeting_room_talking` 0.324
- **中 FP 簇（0.21–0.24）**：`coworkers_standing_meeting` 0.241、`group_friends_indoor_candid` 0.207
- **零/低 FP 簇（0–0.08）**：`people_street_candid` 0.08、`friends_cafe_group` 0.034、`people_restaurant_dining` 0.0

> 模型不是被「人」骗，而是被**正类触发场景所在的室内建成环境**骗——办公室/会议室/客厅正是显示器、白板、
> 投影、电视、文档所在地。这些房间的**亮墙、窗、画框、关闭的屏幕/白板、书架**带有「屏幕/文档相邻」的
> 几何与亮度线索，即便画面里没有任何可读文字、且有人在场，守门员仍误触发。户外街景与餐饮（食物、暗光、
> 无矩形屏状结构）则几乎不触发。

### 共性 2：FP 图显著更亮（+0.186），且约 2× 更可能含「类屏矩形」（+0.118）
- 亮度：FP 0.601 vs 正确拒识 0.415（**最大单维差异**）。
- 类屏矩形命中率：FP 0.237 vs 0.119（FP 约 2 倍）——窗/画框/关屏/白板等大块亮四边形是几何误导线索
  （启发式、有噪声，仅群体层面成立；约 1/4 的 FP 含此线索，是**助攻**而非唯一主因）。

### 共性 3（反直觉、与 task1 诊断一致）：人脸数**反相关**（−0.093），不是「人多→误触发」
FP 图平均人脸数（0.34）反而**低于**正确拒识（0.43），多张高分 FP 检到 0 张正脸。
→ **再排证「count-imbalance」假说**：误触发由**环境/背景**驱动，不是人数。这与 task1「真瓶颈是协变量偏移
而非数量失衡」的结论一致，也意味着**再撒一批人像收效有限**——要补的是**那类室内环境本身 + 无文字的屏状表面**。

### 与 task2 设计的张力（关键，需项目决定）
task2 的人像负例**刻意排除**了办公/会议/教室场景，理由之一是「探针偏办公/会议，训练负例若镜像探针场景，
探针就不再是公平的 held-out 泛化测试」（见 `keywords_task2_neg_people.json` 的 `_design`）。
本诊断恰恰证明：**那个「人≠记 会从街头/市场泛化到办公/客厅」的赌注，在室内建成环境上失败了。**
- 含义：守门员对这些**高混淆室内场景**无法靠泛化解决，需要**同分布**覆盖。
- 代价：一旦把办公/会议/客厅负例纳入训练，探针在这些场景上就从「held-out 泛化测试」变成「同分布测试」，
  FP 下降里有一部分是「训过同类」而非「真泛化」。**这是方法论取舍，留给用户在阶段二定**（见 expansion_plan 的停点）。
- 另一风险：办公/会议/客厅图**极易混入可读屏幕/白板文字**（=正类），若误标为负例会污染负类。故新关键词
  **刻意偏向 empty/blank/off/textless**（空房间、空白白板、关闭的显示器/电视、空投影幕）以**压低污染**，
  且**纳入训练前必须人眼 QC 剔除任何含可读文字屏的图**（task2 既定协议，无法自主完成→列为停点交接项）。
