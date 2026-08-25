#!/bin/bash
set -e
ROOT=/home/student/Documents/projects/wildfire_detection
docker run --rm --runtime nvidia --network host \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e HF_HOME=/hf-cache -e HF_HUB_CACHE=/hf-cache -e TRANSFORMERS_CACHE=/hf-cache -e HF_HUB_OFFLINE=1 \
  -e CUDA_CACHE_PATH=/hf-cache/.nv-compute-cache \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libcudnn.so.9 \
  -e PYTHONUNBUFFERED=1 \
  -v $ROOT/hf-cache:/hf-cache \
  -v $ROOT/nano_llm/vision/video.py:/opt/NanoLLM/nano_llm/vision/video.py \
  -v $ROOT/nano_llm/models/__init__.py:/opt/NanoLLM/nano_llm/models/__init__.py \
  -v $ROOT/nano_llm/models/awq.py:/opt/NanoLLM/nano_llm/models/awq.py \
  -v $ROOT/mmap_sandbox:/work/mmap_sandbox \
  -w /work \
  harel314/orin-r39.2:cu132-nano_llm \
  bash -lc '
    QPATH=$(python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(\"Efficient-Large-Model/VILA1.5-3b-AWQ\",\"llm/vila-1.5-3b-w4-g128-awq-v2.pt\"))")
    echo "QPATH=$QPATH"
    QPATH="$QPATH" python3 -u mmap_sandbox/resident_spike/hybrid_evict_reload.py
  '
