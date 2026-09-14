# WildGuard — Wildfire Detection on a Jetson Orin Nano

WildGuard detects wildfire smoke on a single NVIDIA Jetson Orin Nano, within the board's
thermal and memory limits, by splitting the work across two tiers:

- an **always-on tracking tier** that watches the camera cheaply and continuously, and
- an **on-demand verification tier** — a vision-language model and text classifier — that
  is stored on a microSD Express card and started only when the tracker's confidence
  crosses a threshold.

The tracking tier separates smoke from cloud by behaviour over time rather than appearance
in one frame: a plume stays anchored to its source, rises and grows, while a cloud drifts
rigidly. Fragmented detections are grouped and classified by a kinematic and texture
classifier over Multiple Hypothesis Tracking.

Keeping the heavy tier asleep instead of resident cuts system energy by 29%, and holding
it as a resident daemon rather than starting a container on demand returns a verdict about
ten times faster for about nine times less energy per event.

This repository was prepared for the **2026 SD Association Student Competition**. It
contains the system, the measurement harnesses, and the results behind every figure and
table in the submitted report.

**Authors:** Kfir Shauly, Ori Cohen — Technion, Israel Institute of Technology.
Supervised by Harel Yadid, VISL.

---

## Relationship to EmberEye

WildGuard builds on **EmberEye**, an earlier Technion project by Roy Cohen and Itay Hovav,
whose documentation is preserved at the end of this file.

EmberEye ran a vision-language model continuously on the camera feed. That works, but a
permanently loaded model leaves nothing for the rest of the system on an 8 GB board and
holds the device at its thermal ceiling, which rules out unattended off-grid deployment.

WildGuard keeps EmberEye's vision-language model and classifier **unmodified, as a fixed
black-box workload**, and changes when and how they run: a cheap tracker decides when the
expensive tier is needed, and that tier's weights live on the SD Express card rather than
in memory. Everything measured in the report follows from that change.

---

## Repository layout

- `nano_llm/` and `classifier/` — the EmberEye vision-language model and text classifier,
  used unmodified.
- `fire_watcher/` — the wake service. Tracker confidence drives a state machine that
  pre-warms, wakes and sleeps the verification tier. Contains the daemon, the loaders and
  an MHT simulator for testing without a fire.
- `resident_vlm/` — the resident VLM daemon and the full-system benchmark
  (`bench.py`, `sampler.py`, `analyze.py`). See `resident_vlm/RUNBOOK.txt` for how to run
  it and the conditions that must hold.
- `mmap_sandbox/` — storage experiments.
- `mmap_sandbox/card_eval/` — drive evaluation with standard tools (fio, iostat, gdsio)
  and the stall probe. SD Express versus SSD testing closed on 2026-09-13.
- `important commands` — operational runbook: starting, monitoring, tearing down and
  recovering the system, plus the platform gotchas worth knowing before you hit them.

---

## Where the report's numbers come from

`REPORT_SOURCES.md` maps each claim to its data. For detail on the drive comparison, read
`mmap_sandbox/card_eval/HANDOFF.md`, which also records the caveats and the runs that must
not be quoted.

| Result | Data |
|---|---|
| Drive characterisation (fio / gdsio) | `mmap_sandbox/results/card_eval/sd_express_20260911_211324`, `ssd_20260912_004218` |
| Platform stall, both drives | `mmap_sandbox/results/card_eval/stall_probe_*` |
| Drive identity, link width, power states, thermals | `mmap_sandbox/results/card_eval/nvme_diag_*` |
| Resident wake vs cold start | `resident_vlm/results/bench_20260911_220613` (card), `bench_20260912_013746` (SSD) |
| Burst vs streaming reads | `resident_vlm/results/waketest_20260913_150739` (card), `waketest_20260913_154137` (SSD) |
| Gated vs always-on power | `mmap_sandbox/results/perf_burst_vs_continuous` |

---

## Hardware

- NVIDIA Jetson Orin Nano Super Developer Kit, 25 W power mode
- SanDisk microSD Express 512 GB (`SDSQXFN-512G-GN4NN`) via the supplied M.2 adapter,
  carrying the operating system, container images, model weights and results
- USB or CSI camera at `/dev/video0`
- Display connected for `display://0`

The comparison drive used in the report is a Crucial P5 Plus 1 TB NVMe SSD
(`CT1000P5PSSD8`).

---

## Running the system

Prerequisites, installation and troubleshooting are covered in the EmberEye documentation
below and still apply. Once the environment is set up, the fused system managed by the
wake service is started with:

```bash
cd /path/to/wildfire_detection
xhost +local:root
docker compose -f docker-compose_fused_system.yaml --profile watcher up -d fire_watcher
```

To stop it:

```bash
docker compose -f docker-compose_fused_system.yaml --profile watcher stop fire_watcher
```

To watch what it is doing:

```bash
docker logs -f fire_watcher_fused
tail -f fire_watcher/watch.jsonl
```

`important commands` has the rest, including how to force a wake by hand, how to run the
tracker alone against the camera or a recorded clip, and how to recover if a run is
interrupted.

---
---

# EmberEye — Real-Time Wildfire Detection System

*The documentation below is from the EmberEye project by Roy Cohen and Itay Hovav, on
which WildGuard is built. Its setup and troubleshooting sections still apply.*

EmberEye is a real-time wildfire detection system designed to run on NVIDIA Jetson edge devices.  
The system uses a camera feed, a Vision-Language Model (VLM), and a semantic classifier to detect possible wildfire or smoke events in real time.

The system is containerized with Docker Compose and is intended to run directly on a Jetson device with a connected camera.

---

## System Overview

The system contains two main Docker services:

1. **wildfire_detection**  
   Reads the camera stream, runs the VLM, generates scene descriptions, displays the video output, and sends descriptions to the classifier.

2. **wildfire_classifier**  
   Receives the text descriptions, classifies them as wildfire / no-wildfire using a text classifier, and writes the results to an output file.

The general flow is:

```text
Camera
  ↓
wildfire_detection container
  ↓
VLM scene description
  ↓
UDP communication
  ↓
wildfire_classifier container
  ↓
FIRE / NO_FIRE classification
```

---

## Tested Hardware

Recommended hardware:

- NVIDIA Jetson Orin Nano
- USB camera or CSI camera exposed as `/dev/video0`
- SSD / microSD Express / fast storage
- Monitor connected to the Jetson for `display://0`

---

## Required Dependencies

Before running the project, install the following on the Jetson host machine.

### 1. Docker

```bash
sudo apt update
sudo apt install -y docker.io
```

Enable Docker without `sudo`:

```bash
sudo usermod -aG docker $USER
```

After this command, log out and log back in, or reboot.

Check Docker:

```bash
docker --version
```

---

### 2. Docker Compose

Depending on your Jetson installation, one of these should work:

```bash
sudo apt install -y docker-compose
```

or:

```bash
sudo apt install -y docker-compose-plugin
```

Check Docker Compose:

```bash
docker compose version
```

---

### 3. NVIDIA Container Runtime

The project uses NVIDIA GPU acceleration inside Docker containers.

Check that NVIDIA Docker runtime is available:

```bash
docker info | grep -i nvidia
```

If it is missing, install NVIDIA Container Runtime according to the Jetson / JetPack version you are using.

---

### 4. v4l-utils

This is required for controlling the camera frame rate with `v4l2-ctl`.

Install it with:

```bash
sudo apt install -y v4l-utils
```

Check that the camera exists:

```bash
ls /dev/video*
```

Check camera capabilities:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

---

### 5. X11 Display Access

The system displays the video output using the Jetson display.  
Before running Docker, allow the container to access the host display.

Run:

```bash
xhost +local:root
```

This must be done once per login/session before starting the container.

---

## Before Running

Go into the project directory:

```bash
cd /path/to/wildfire_detection
```

---

## Step 1 — Allow Docker to Use the Display

```bash
xhost +local:root
```

---

## Step 2 — Set Camera FPS

```bash
v4l2-ctl -d /dev/video0 --set-parm=15
```

This reduces processing load and makes the real-time pipeline more stable.

---

## Step 3 — Run the System with the Camera

```bash
docker compose -f docker-compose_complete_system.yaml run --rm -e VIDEO_INPUT="/dev/video0" -e VIDEO_OUTPUT="display://0" wildfire_detection
```

---

## Output Files

The classifier writes results to:

```text
data/classifier_out/classifier_results.jsonl
```

The VLM text output is written to:

```text
data/out_txt
```

If clip saving is enabled, clips are saved under:

```text
data/classifier_out/clips/
```

---

## Camera Troubleshooting

### Check if the camera exists

```bash
ls /dev/video*
```

If `/dev/video0` does not exist, try another camera index, for example `/dev/video1`.

### Check camera formats

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

---

## Display Troubleshooting

If you get display errors, run:

```bash
xhost +local:root
```

Also check:

```bash
echo $DISPLAY
```

The compose file usually expects `DISPLAY=:1`. If your Jetson uses another display value,
update the compose file or override the variable.

---

## Docker Troubleshooting

```bash
docker pull dustynv/nano_llm:r36.4.0    # pull image manually
docker ps                                # running containers
docker ps -a                             # all containers
docker compose -f docker-compose_complete_system.yaml down    # stop all
docker compose -f docker-compose_complete_system.yaml logs -f # logs
```

---

## Notes

- The first run can take time because Python dependencies and models may be downloaded.
- A stable internet connection is recommended for the first run.
- The Jetson may need swap memory because VLM models consume significant RAM.
- Use fast storage if possible.
- For best results, use a clear camera feed and stable lighting.

---

## EmberEye Authors

- Roy Cohen
- Itay Hovav

Technion — Israel Institute of Technology
