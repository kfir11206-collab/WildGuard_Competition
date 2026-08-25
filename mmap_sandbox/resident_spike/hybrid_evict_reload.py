#!/usr/bin/env python3
"""
Hybrid resident-VLM spike.

The plan we are testing: keep the expensive-to-COMPUTE part of the VLM resident
(the ~47s make_quant_attn step) but push the ~1.7GB of WEIGHTS back to the SD
card while asleep, and stream them in from the card on each wake.

make_quant_attn fuses q/k/v into a single qkv_proj, so the fused model's weight
names no longer match the original checkpoint. Stage 1 therefore writes the
FUSED state_dict to the card once; stages 2 and 3 reload from that file.

  STAGE 1  baseline : load the model, run one inference, save fused weights.
  STAGE 2  reload   : re-read the fused weights from the SD into the SAME model
                      and check we did NOT re-run make_quant_attn.
  STAGE 3  evict    : drop the weights to give memory back, then reload from
                      the SD and confirm inference still works.

Runs inside the wildfire docker image. Edits no nano_llm/ file.
"""
import os, gc, json, time
import torch
from nano_llm import NanoLLM, ChatHistory

QUANT = os.environ["QPATH"]
MODEL = os.environ.get("MODEL", "Efficient-Large-Model/VILA1.5-3b-AWQ")
RESULTS = "mmap_sandbox/resident_spike/results"
FUSED = os.path.join(RESULTS, "fused_state.pt")
PROMPT = "What is 2+2?"

import nano_llm.models.awq as awq
fuse_calls = {"n": 0, "seconds": 0.0}
_orig_fuse = awq.make_quant_attn
def counting_fuse(*a, **k):
    t0 = time.monotonic()
    r = _orig_fuse(*a, **k)
    torch.cuda.synchronize()
    fuse_calls["n"] += 1
    fuse_calls["seconds"] += time.monotonic() - t0
    return r
awq.make_quant_attn = counting_fuse


def gpu_mb():
    return round(torch.cuda.memory_allocated() / 1e6)


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024)
    return None


def cached_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("Cached:"):
                return round(int(line.split()[1]) / 1024)
    return None


def drop_page_cache(path):
    os.sync()
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def meta_params(module):
    return sum(1 for p in module.parameters() if p.device.type == "meta")


def infer(model, text):
    chat = ChatHistory(model)
    chat.append(role="user", text=text)
    out = model.generate(chat.embed_chat()[0], streaming=False)
    return str(out).strip()


def stream_from_sd(model):
    sd = torch.load(FUSED, map_location="cpu", mmap=True)
    report = model.model.load_state_dict(sd, assign=True, strict=False)
    model.model.to("cuda")
    torch.cuda.synchronize()
    del sd
    gc.collect()
    return len(report.missing_keys), len(report.unexpected_keys)


def stage(name):
    print(f"\n===== {name} =====", flush=True)


results = {}

stage("STAGE 1  baseline load")
t0 = time.monotonic()
model = NanoLLM.from_pretrained(MODEL, api="awq", quantization=QUANT, vision_api="hf")
load_s = round(time.monotonic() - t0, 2)
base_answer = infer(model, PROMPT)

os.makedirs(RESULTS, exist_ok=True)
if os.path.exists(FUSED):
    save_s = None
else:
    t0 = time.monotonic()
    torch.save({k: v.cpu() for k, v in model.model.state_dict().items()}, FUSED)
    save_s = round(time.monotonic() - t0, 2)

results["baseline"] = {
    "load_seconds": load_s, "fuse_calls": fuse_calls["n"],
    "fuse_seconds": round(fuse_calls["seconds"], 2),
    "gpu_mb": gpu_mb(), "rss_mb": rss_mb(),
    "fused_save_seconds": save_s,
    "fused_file_gb": round(os.path.getsize(FUSED) / 1e9, 2),
    "answer": base_answer,
}
print(json.dumps(results["baseline"], indent=2), flush=True)

stage("STAGE 2  reload fused weights from SD (structure kept)")
fuse_before = fuse_calls["n"]
t0 = time.monotonic()
missing, unexpected = stream_from_sd(model)
reload_s = round(time.monotonic() - t0, 2)
answer2 = infer(model, PROMPT)
results["reload"] = {
    "reload_seconds": reload_s,
    "fuse_calls_added": fuse_calls["n"] - fuse_before,
    "missing_keys": missing, "unexpected_keys": unexpected,
    "gpu_mb": gpu_mb(), "rss_mb": rss_mb(),
    "answer_matches_baseline": answer2 == base_answer,
}
print(json.dumps(results["reload"], indent=2), flush=True)

stage("STAGE 3  evict to free memory, then wake from SD")
mb_awake, rss_awake = gpu_mb(), rss_mb()
model.model.to("meta")
gc.collect(); torch.cuda.empty_cache()
mb_asleep, rss_asleep = gpu_mb(), rss_mb()
evicted = meta_params(model.model)
fuse_before = fuse_calls["n"]
t0 = time.monotonic()
missing, unexpected = stream_from_sd(model)
wake_s = round(time.monotonic() - t0, 2)
still_meta = meta_params(model.model)
answer3 = infer(model, PROMPT)
results["evict_wake"] = {
    "gpu_mb_awake": mb_awake, "gpu_mb_asleep": mb_asleep,
    "gpu_mb_freed": mb_awake - mb_asleep,
    "rss_mb_awake": rss_awake, "rss_mb_asleep": rss_asleep,
    "rss_mb_freed": rss_awake - rss_asleep,
    "params_evicted": evicted, "params_still_meta_after_wake": still_meta,
    "wake_seconds": wake_s, "fuse_calls_added": fuse_calls["n"] - fuse_before,
    "missing_keys": missing, "unexpected_keys": unexpected,
    "gpu_mb_after_wake": gpu_mb(), "rss_mb_after_wake": rss_mb(),
    "answer_matches_baseline": answer3 == base_answer,
}
print(json.dumps(results["evict_wake"], indent=2), flush=True)

stage("STAGE 4  cold-cache wake (page cache dropped for the fused file)")
mb_awake, rss_awake = gpu_mb(), rss_mb()
model.model.to("meta")
gc.collect(); torch.cuda.empty_cache()
mb_asleep4 = gpu_mb()
cached_before = cached_mb()
drop_page_cache(FUSED)
cached_after = cached_mb()
fuse_before = fuse_calls["n"]
t0 = time.monotonic()
missing, unexpected = stream_from_sd(model)
cold_wake_s = round(time.monotonic() - t0, 2)
answer4 = infer(model, PROMPT)
results["cold_wake"] = {
    "page_cache_mb_before_drop": cached_before,
    "page_cache_mb_after_drop": cached_after,
    "page_cache_mb_evicted": cached_before - cached_after,
    "gpu_mb_freed": mb_awake - mb_asleep4,
    "cold_wake_seconds": cold_wake_s,
    "warm_wake_seconds": results["evict_wake"]["wake_seconds"],
    "fuse_calls_added": fuse_calls["n"] - fuse_before,
    "missing_keys": missing, "unexpected_keys": unexpected,
    "params_still_meta_after_wake": meta_params(model.model),
    "answer_matches_baseline": answer4 == base_answer,
}
print(json.dumps(results["cold_wake"], indent=2), flush=True)

with open(os.path.join(RESULTS, "hybrid.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\n[spike] wrote results/hybrid.json", flush=True)
