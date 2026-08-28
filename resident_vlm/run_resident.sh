#!/bin/bash
set -e
ROOT=/home/student/Documents/projects/wildfire_detection
docker run --rm --name resident_vlm --runtime nvidia --network host \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -e HF_HOME=/hf-cache -e HF_HUB_CACHE=/hf-cache -e TRANSFORMERS_CACHE=/hf-cache -e HF_HUB_OFFLINE=1 \
  -e CUDA_CACHE_PATH=/hf-cache/.nv-compute-cache \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libcudnn.so.9 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PYTHONUNBUFFERED=1 \
  -v $ROOT/hf-cache:/hf-cache \
  -v $ROOT/nano_llm/vision/video.py:/opt/NanoLLM/nano_llm/vision/video.py \
  -v $ROOT/nano_llm/models/__init__.py:/opt/NanoLLM/nano_llm/models/__init__.py \
  -v $ROOT/nano_llm/models/awq.py:/opt/NanoLLM/nano_llm/models/awq.py \
  -v $ROOT/resident_vlm:/resident_vlm \
  -v $ROOT/data:/data \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v /etc/nv_tegra_release:/etc/nv_tegra_release \
  -v /tmp/nv_jetson_model:/tmp/nv_jetson_model \
  -v /etc/localtime:/etc/localtime:ro \
  --device /dev/video0 --device /dev/video1 --device /dev/bus/usb \
  -w /resident_vlm \
  harel314/orin-r39.2:cu132-nano_llm \
  bash -lc '
    QPATH=$(python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(\"Efficient-Large-Model/VILA1.5-3b-AWQ\",\"llm/vila-1.5-3b-w4-g128-awq-v2.pt\"))")
    QPATH="$QPATH" python3 -u /resident_vlm/resident_daemon.py \
      --sock /resident_vlm/resident.sock \
      --fused /resident_vlm/results/fused_state.pt \
      --log /resident_vlm/results/daemon.jsonl \
      --text-out /data/out_txt \
      '"$*"'
  '
