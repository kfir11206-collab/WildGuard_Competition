# EmberEye — Real-Time Wildfire Detection System

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

## Project Structure

Expected project structure:

```text
wildfire_detection/
│
├── docker-compose_complete_system.yaml
│
├── classifier/
│   └── classifier.py
│
├── nano_llm/
│   └── video.py
│
├── data/
│   ├── out_txt
│   └── classifier_out/
│
├── hf-cache/
│
└── README.md
```

Make sure you run all commands from inside the project directory.

---

## Before Running

Go into the project directory:

```bash
cd /path/to/wildfire_detection
```

Example:

```bash
cd ~/projects/wildfire_detection
```

---

## Step 1 — Allow Docker to Use the Display

Run:

```bash
xhost +local:root
```

This allows the Docker container to open the display window.

---

## Step 2 — Set Camera FPS

Set the camera frame rate to 15 FPS:

```bash
v4l2-ctl -d /dev/video0 --set-parm=15
```

This reduces processing load and makes the real-time pipeline more stable.

---

## Step 3 — Run the System with the Camera

From inside the project directory, run:

```bash
docker compose -f docker-compose_complete_system.yaml run --rm -e VIDEO_INPUT="/dev/video0" -e VIDEO_OUTPUT="display://0" wildfire_detection
```

This command runs the `wildfire_detection` service using the live camera at `/dev/video0`.

---

## Full Run Sequence

Use this exact sequence:

```bash
xhost +local:root
v4l2-ctl -d /dev/video0 --set-parm=15
docker compose -f docker-compose_complete_system.yaml run --rm -e VIDEO_INPUT="/dev/video0" -e VIDEO_OUTPUT="display://0" wildfire_detection
```

---

## Running the Full Compose System

If you want to start all services defined in the compose file, use:

```bash
docker compose -f docker-compose_complete_system.yaml up
```

To run in the background:

```bash
docker compose -f docker-compose_complete_system.yaml up -d
```

To stop:

```bash
docker compose -f docker-compose_complete_system.yaml down
```

To see logs:

```bash
docker compose -f docker-compose_complete_system.yaml logs -f
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

If `/dev/video0` does not exist, try another camera index, for example:

```bash
/dev/video1
```

Then run with:

```bash
docker compose -f docker-compose_complete_system.yaml run --rm -e VIDEO_INPUT="/dev/video1" -e VIDEO_OUTPUT="display://0" wildfire_detection
```

---

### Check camera formats

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

---

### Set FPS again

```bash
v4l2-ctl -d /dev/video0 --set-parm=15
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

The compose file usually expects:

```text
DISPLAY=:1
```

If your Jetson uses another display value, update the compose file or override the variable.

---

## Docker Troubleshooting

### Pull image manually

```bash
docker pull dustynv/nano_llm:r36.4.0
```

### Check running containers

```bash
docker ps
```

### Check all containers

```bash
docker ps -a
```

### Stop all compose services

```bash
docker compose -f docker-compose_complete_system.yaml down
```

### View logs

```bash
docker compose -f docker-compose_complete_system.yaml logs -f
```

---

## Notes

- The first run can take time because Python dependencies and models may be downloaded.
- A stable internet connection is recommended for the first run.
- The Jetson may need swap memory because VLM models consume significant RAM.
- Use fast storage if possible.
- For best results, use a clear camera feed and stable lighting.

---

## Main Command

The most important command is:

```bash
docker compose -f docker-compose_complete_system.yaml run --rm -e VIDEO_INPUT="/dev/video0" -e VIDEO_OUTPUT="display://0" wildfire_detection
```

Run it only after:

```bash
xhost +local:root
v4l2-ctl -d /dev/video0 --set-parm=15
```

---

## Authors

- Roy Cohen
- Itay Hovav

Technion — Israel Institute of Technology
