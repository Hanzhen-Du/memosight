# hardware/ — 树莓派 5 边缘部署

守门员在 **Raspberry Pi 5 + Camera Module 3** 上的端侧部署代码。
落地项目命门——**级联感知**：常开守门员只看低分辨率灰度流，只有判"记"时才抓高清做下游处理。**不连续录像**。

## 目标环境
- Raspberry Pi 5，Camera Module 3，Raspberry Pi OS。
- 运行时：**LiteRT**（`ai_edge_litert`）。代码会在缺它时回退到 `tensorflow.lite`，方便笔记本自测。
- 模型已在 Pi 上：`~/dev/memosight/models/gatekeeper_v4_mvp_int8.tflite`
  （int8，输入 `(1,96,96,1)` int8 灰度，输出 `(1,2)` int8）。

## 文件
| 文件 | 作用 |
|---|---|
| `infer.py` | 共享推理模块（importable）：`load_model` / `preprocess` / `predict`。预处理口径与训练一致（灰度、INTER_AREA、按模型量化参数转 int8）。 |
| `benchmark_latency.py` | 纯推理延迟基准：随机 int8 输入，20 warm-up + 200 计时，报 mean/p50/p95/p99/吞吐。 |
| `camera_test.py` | 相机自检（Picamera2）：抓几帧，存全分辨率 JPEG + 96×96 灰度 PNG。**相机接上后第一个跑。** |
| `cascade.py` | 级联主循环（今天仅 `gated`）：低分辨率流→守门员→命中才抓高清存盘+记日志+(可选)推回。 |
| `receiver_laptop.py` | **笔记本端**接收器，配 `cascade.py --push`。 |

## Pi 端依赖
优先用已装的：`ai_edge_litert`、`opencv-headless`(cv2)、`numpy`。

**可能需要另装**：
```bash
sudo apt install -y python3-picamera2   # Camera Module 3 驱动，随 Pi OS 发行但不一定预装；含 libcamera 系统依赖，勿用 pip
```
`camera_test.py` / `cascade.py` 需要 `picamera2`；`infer.py` / `benchmark_latency.py` 不需要。

## 从笔记本同步代码 + 模型到 Pi（手动）
代码进 git，但**模型/数据按 `.gitignore` 不入库**，需手动传：
```bash
# 笔记本上执行。<pi> = pi@<树莓派IP>
# 1) 代码：在 Pi 上 git pull 本分支，或直接 scp hardware/
rsync -av hardware/ <pi>:~/dev/memosight/hardware/
# 2) 模型（不在 git 里，必须单独传）
scp models/gatekeeper_v4_mvp_int8.tflite <pi>:~/dev/memosight/models/
```

## 在 Pi 上运行（仓库根目录 `~/dev/memosight`）
```bash
# 0) 相机自检（接上相机后第一步）
python3 hardware/camera_test.py --outdir hardware/captures

# 1) 纯推理延迟基准
python3 hardware/benchmark_latency.py \
    --model ~/dev/memosight/models/gatekeeper_v4_mvp_int8.tflite

# 2) 级联主循环（gated）。--fps 是占空比/Pareto 旋钮。
python3 hardware/cascade.py --fps 3
python3 hardware/cascade.py --fps 3 --threshold 0.55   # 调 FN/FP 工作点
```

## 可选：命中帧 WiFi 推回笔记本
默认**关**（`--push` 才开），网络不稳也不影响核心循环；命中帧始终先存本地 `captures/`。
```bash
# 笔记本上先起接收器
python3 hardware/receiver_laptop.py --outdir ~/memosight_received --port 8000
# Pi 上开 push（<laptop> = 笔记本局域网 IP）
python3 hardware/cascade.py --fps 3 --push --host <laptop> --port 8000
```
推回用 stdlib HTTP POST（零新依赖，Pi 主动发起）。推失败会降级——帧留在 `captures/`，可事后 `rsync` 拉回。**仅限可信局域网。**

## 运行产物（不入库）
`hardware/captures/`（命中高清帧 + `hits.log`）是运行时产物，由 `hardware/.gitignore` 排除。

## 路线图（今天**未**做，明确留 TODO）
- **gated vs always 功耗 A/B**：`cascade.should_record()` 已把决策隔离成一个函数，
  将来开 `--mode always` 只需一行（见该函数 TODO）。今天不实现，恪守低功耗论点。
- **ESP32 真机验证**：本目录是 Pi（宽容环境）；ESP32-ready 仍需另测 arena 占用 / kernel 数值一致性 / 延迟功耗。
- **功耗测量 + Pareto 曲线**：扫 `--fps`（和将来的分辨率/阈值）画"功耗 vs 漏报"。
