# hardware/ — Raspberry Pi 5 edge deployment

On-device deployment code for the gatekeeper, targeting a Raspberry Pi 5 with Camera Module 3.

This is where the project's central idea becomes real. The always-on gatekeeper watches only a
low-resolution greyscale stream, and a full-resolution grab plus downstream processing happens
only when it decides the frame is worth recording. Nothing is recorded continuously.

## Target environment

- Raspberry Pi 5, Camera Module 3, Raspberry Pi OS.
- Runtime: LiteRT (`ai_edge_litert`). The code falls back to `tensorflow.lite` when it is
  missing, which makes the same code testable on a laptop.
- The model is already on the Pi at
  `~/dev/memosight/models/gatekeeper_v4_mvp_int8.tflite`: int8, input `(1,96,96,1)` int8
  greyscale, output `(1,2)` int8.

## Files

| File | Purpose |
|---|---|
| `infer.py` | Shared, importable inference module: `load_model`, `preprocess`, `predict`. Preprocessing matches training exactly (greyscale, INTER_AREA, quantised to int8 using the model's own parameters). |
| `benchmark_latency.py` | Pure inference latency benchmark: random int8 input, 20 warm-up iterations then 200 timed, reporting mean, p50, p95, p99 and throughput. |
| `camera_test.py` | Camera self-test via Picamera2: grab a few frames and save a full-resolution JPEG plus a 96x96 greyscale PNG. Run this first once the camera is connected. |
| `cascade.py` | The cascade main loop, currently `gated` only: low-resolution stream, gatekeeper, and on a hit grab full resolution, save it, log it, and optionally push it back. |
| `receiver_laptop.py` | Laptop-side receiver, paired with `cascade.py --push`. |

## Dependencies on the Pi

Prefer what is already installed: `ai_edge_litert`, `opencv-headless` (cv2), `numpy`.

You may need to install:

```bash
sudo apt install -y python3-picamera2   # Camera Module 3 driver. Ships with Pi OS but is not always preinstalled.
                                        # It pulls in libcamera system dependencies, so do not install it with pip.
```

`camera_test.py` and `cascade.py` need `picamera2`; `infer.py` and `benchmark_latency.py` do
not.

## Syncing code and the model to the Pi

Code is in git, but models and data are excluded by `.gitignore`, so the model has to be copied
manually.

```bash
# Run on the laptop. <pi> = pi@<raspberry pi ip>
# 1) Code: either git pull on the Pi, or copy hardware/ across
rsync -av hardware/ <pi>:~/dev/memosight/hardware/
# 2) Model: not in git, must be transferred separately
scp models/gatekeeper_v4_mvp_int8.tflite <pi>:~/dev/memosight/models/
```

## Running on the Pi

From the repository root, `~/dev/memosight`.

```bash
# 0) Camera self-test. First thing to run once the camera is connected
python3 hardware/camera_test.py --outdir hardware/captures

# 1) Pure inference latency benchmark
python3 hardware/benchmark_latency.py \
    --model ~/dev/memosight/models/gatekeeper_v4_mvp_int8.tflite

# 2) Cascade main loop, gated mode. --fps is the duty-cycle knob from the Pareto curve
python3 hardware/cascade.py --fps 3
python3 hardware/cascade.py --fps 3 --threshold 0.55   # move the FN/FP operating point
```

## Optional: push hit frames to a laptop over WiFi

Off by default; `--push` enables it. An unstable network does not affect the core loop, because
hit frames are always written to local `captures/` first.

```bash
# Start the receiver on the laptop
python3 hardware/receiver_laptop.py --outdir ~/memosight_received --port 8000
# Enable push on the Pi. <laptop> = the laptop's LAN address
python3 hardware/cascade.py --fps 3 --push --host <laptop> --port 8000
```

The push is a stdlib HTTP POST initiated by the Pi, so it adds no dependency. A failed push
degrades gracefully: the frame stays in `captures/` and can be pulled with `rsync` later. Use
this on a trusted local network only.

## Runtime output

`hardware/captures/`, holding hit frames at full resolution plus `hits.log`, is runtime output
and is excluded by `hardware/.gitignore`.

## Roadmap

Not done yet, recorded here as explicit TODOs.

- Gated versus always-on power A/B. `cascade.should_record()` already isolates the decision into
  a single function, so adding `--mode always` later is a one-line change (see the TODO in that
  function). It is deliberately not implemented yet, to keep the low-power claim honest.
- ESP32 on-device verification. This directory targets the Pi, which is a forgiving environment.
  Being ESP32-ready still requires separately measuring arena occupancy, kernel numerical
  agreement, and latency and power.
- Power measurement and the Pareto curve. Sweep `--fps`, and later resolution and threshold, to
  plot power against missed captures.
