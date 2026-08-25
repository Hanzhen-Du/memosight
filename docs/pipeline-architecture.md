# MemoSight MVP 闭环 · 阶段一 — 端到端演示说明 & 模块结构

> **Summary (EN).** Architecture of the downstream pipeline: gatekeeper trigger → full-res
> frame → OCR → payload → enrichment → SQLite memory card → CLI recall, with an offline
> queue that backfills when connectivity returns. Documents the three swappable interfaces
> (OCR / enricher / transport) that keep the research layer independent of vendor choice,
> and the raw-image retention policy (delete by default).

**分支：** `ai/mvp-pipeline`　**日期：** 2026-07-06　**状态：** M1–M5 全部跑通（44 tests，2 skip）

本阶段把守门员之后的"感知→记忆→可回忆"后半段做成了**能演示所有功能的半成品**：
纯软件 + mock，不碰真硬件、不接真 API。

---

## 1. 数据流（全部已跑通）

```
守门员触发(mock 信号+置信度)          MockGatekeeper.trigger()
        │
   高清抓帧(测试图替代)               capture.grab_frame()  → data/mvp_demo/frames/
        │
   本地 OCR                          OCRInterface → TesseractOCR(真) / StubOCR(测试)
        │  ocr_text
   打包 {ocr_text + 元数据}          packaging.build_payload()
        │  (timestamp/trigger_confidence/raw_image_policy)
   ┌────┴─────────── is_online() mock ───────────┐
   │联网                                          │断网
   transport.upload → enrich(mock tags)          存 pending(tags 空)
   → status=done 存库                            IngestService._queue()
        │                                             │ 恢复联网
        │                                        process_pending() 批量补 tags → done
   隐私：raw_image_policy 处理原始帧             privacy.apply_raw_image_policy()
   (默认 delete；可 cache)
        │
   SQLite 存储                       db.CardStore  → data/mvp_demo/memosight.db
        │
   命令行查询                        cli.py: list / show / search / pending
```

## 2. 三个可替换接口（可扩展性命脉）

| 接口 | 抽象基类 | 本阶段实现 | 未来替换 |
|------|----------|-----------|----------|
| **OCR** | `pipeline/ocr/base.py::OCRInterface` | `TesseractOCR`（真引擎）/ `StubOCR`（测试） | 手机端 / 云端 OCR |
| **Enricher** | `pipeline/enrich/base.py::EnricherInterface` | `CloudEnricher`（**mock**，返回 `mock:` 假标签） | 真 Anthropic Claude API |
| **传输/上传** | `pipeline/transport/base.py::UploadInterface` | `DirectUploadMock`（Pi 直连 mock） | 经手机中转 |

> **tags 是唯一由"云端大模型"生成的字段** —— 本阶段唯一来源是 CloudEnricher(mock)，
> 不写规则实现；断网时也绝不用规则伪造 tags，只入 pending 队列等真云端。

## 3. 模块清单

```
pipeline/
  models.py        MemoryCard 数据模型（对应固定 SQLite 表）
  db.py            CardStore：schema + CRUD + pending 队列 + search
  config.py        Config：raw_image_policy 默认 delete、目录、OCR 语言
  packaging.py     Payload：ocr_text + 元数据 打包
  connectivity.py  Connectivity ABC + ConnectivityMock（可切换 is_online）
  capture.py       MockGatekeeper（mock 触发）+ grab_frame（抓帧）
  privacy.py       apply_raw_image_policy（delete / cache）
  ingest.py        IngestService：联网直存 / 断网入队 / 恢复批量补传
  pipeline.py      MemoSightPipeline 编排 + build_pipeline() 工厂
  cli.py           命令行查询/演示
  ocr/             OCR 接口 + Tesseract + Stub
  enrich/          Enricher 接口 + CloudEnricher(mock)
  transport/       传输接口 + DirectUploadMock
tests/             44 个 unittest（stdlib，无需 pytest）
```

## 4. 怎么跑

```bash
# 全部测试（tesseract 未装时 2 个真引擎测试 skip）
.venv/bin/python -m unittest discover -s tests -v

# 内置端到端演示（合成文字图跑完整闭环 + 搜出来）
.venv/bin/python -m pipeline.cli demo

# 手动捕捉一张图（真实/合成图片皆可）
.venv/bin/python -m pipeline.cli ingest <image.png> --confidence 0.9
.venv/bin/python -m pipeline.cli ingest <image.png> --offline    # 模拟断网入队
.venv/bin/python -m pipeline.cli process-pending                 # 恢复联网补传

# 查询
.venv/bin/python -m pipeline.cli list
.venv/bin/python -m pipeline.cli search <keyword>
.venv/bin/python -m pipeline.cli show <id>
.venv/bin/python -m pipeline.cli pending

# DB 路径可用环境变量覆盖：MEMOSIGHT_DB=/path/to.db
```

## 5. 本阶段是 mock / stub 的部分（阶段二再换真）

- **OCR 引擎选定 = Tesseract**（用户 2026-07-06 拍板）。pytesseract+pillow 已装进 venv；
  **系统二进制需另行安装：
  ```
  sudo apt install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng
  ```
  装好后 `test_ocr.py` 里 2 个 skip 会转为真跑；`cli demo` 的 OCR 会从占位文本变成真识别
  （"MEMOSIGHT DEMO ROADMAP"）。二进制缺失时管线自动回退 StubOCR，闭环照常演示。
- **Enricher = CloudEnricher(mock)**：返回 `mock:` 前缀的假标签（诚实标注）。阶段二换真 Claude API + prompt。
  - 密钥基础设施已铺好：`pipeline/env.py::load_env()` 用 python-dotenv 从项目根 `.env` 自动加载
    `ANTHROPIC_API_KEY`（不依赖终端 export，任何进程可读），`get/require_anthropic_api_key()` 从
    `os.environ` 读取。**绝不硬编码密钥**，`.env` 已 gitignore。已在 CLI 入口和 `build_pipeline()` 里调用。
- **传输 = DirectUploadMock**：Pi 直连的假实现。阶段二可加手机中转实现。
- **守门员/摄像头 = MockGatekeeper + 测试图**：阶段二接真守门员(task1 C_wide_uniform int8)+真摄像头。

## 6. 测试覆盖（44 tests, 2 skip）

- `test_db.py`(14) 模型校验 / CRUD / search / enrich 状态迁移 / FIFO pending / 重开持久化
- `test_ocr.py`(10, 2 skip) preprocess resize+灰度 / StubOCR / Tesseract 真引擎(待二进制)
- `test_enrich.py`(9) 接口 / mock 标签形状 / 可复现 / 置信度标签 / 模拟失败 / 打包
- `test_queue.py`(6) 联网直存 / 断网入队 / 恢复补传 / 云端失败回退 / 失败留 pending
- `test_e2e.py`(5) 在线捕捉+回忆 / 未触发不记 / 断网→恢复 / 默认删帧 / cache 保留帧

## 7. 待办 / 阶段二入口

- [ ] 跑 apt 命令装 tesseract 二进制（见 §5），我再复核真引擎中文/英文效果。
- [ ] 审查本阶段，决定进阶段二：接真 Claude API（enricher）+ 真守门员/摄像头联调 + 手机中转。
