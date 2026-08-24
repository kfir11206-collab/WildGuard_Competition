#!/usr/bin/env python3
# Non-invasive probe: breaks open the ~66s "AWQ LLM load + GPU init" black box
# by monkeypatching the load functions with timers at runtime. Touches no
# nano_llm/ file. Run inside the wildfire image with the model files mounted.
import os, time, json, sys
import torch

QUANT = os.environ["QPATH"]
MODEL = os.environ.get("MODEL", "Efficient-Large-Model/VILA1.5-3b-AWQ")
FORCE_NO_MMAP = os.environ.get("NO_MMAP") == "1"

timings = []

def stamp(name, dt):
    timings.append((name, round(dt, 2)))
    print(f"[probe] {name:38s} {dt:7.2f}s", flush=True)

def wrap(mod, name, sync=True):
    orig = getattr(mod, name)
    def timed(*a, **k):
        t0 = time.monotonic()
        r = orig(*a, **k)
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        stamp(name, time.monotonic() - t0)
        return r
    setattr(mod, name, timed)

import nano_llm.models.awq as awq
for fn in ("make_quant_linear", "make_quant_attn", "make_quant_norm"):
    if hasattr(awq, fn):
        wrap(awq, fn)

# torch.load: force mmap off if asked, to push the SD read into THIS call
_orig_load = torch.load
def load_timed(*a, **k):
    if FORCE_NO_MMAP:
        k["mmap"] = False
    t0 = time.monotonic()
    r = _orig_load(*a, **k)
    stamp(f"torch.load(mmap={k.get('mmap')})", time.monotonic() - t0)
    return r
torch.load = load_timed
awq.torch.load = load_timed

# time the fp16 skeleton construction (LlamaForCausalLM(config).half())
if hasattr(awq, "LlamaForCausalLM"):
    _Llama = awq.LlamaForCausalLM
    class TimedLlama(_Llama):
        def __init__(self, *a, **k):
            t0 = time.monotonic()
            super().__init__(*a, **k)
            stamp("LlamaForCausalLM(config)", time.monotonic() - t0)
    awq.LlamaForCausalLM = TimedLlama

def gpu_mb():
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.memory_allocated() / 1e6)

from nano_llm import NanoLLM
print(f"[probe] loading (NO_MMAP={FORCE_NO_MMAP}) ...", flush=True)
t0 = time.monotonic()
model = NanoLLM.from_pretrained(MODEL, api="awq", quantization=QUANT, vision_api="hf")
total = time.monotonic() - t0
torch.cuda.synchronize() if torch.cuda.is_available() else None

accounted = sum(dt for _, dt in timings)
remainder = round(total - accounted, 2)
out = {
    "total_from_pretrained_s": round(total, 2),
    "phases": timings,
    "remainder_(skeleton_move_misc)_s": remainder,
    "gpu_mem_allocated_mb": gpu_mb(),
    "no_mmap": FORCE_NO_MMAP,
}
print("[probe] RESULT:", json.dumps(out, indent=2), flush=True)
os.makedirs("mmap_sandbox/resident_spike/results", exist_ok=True)
tag = "nommap" if FORCE_NO_MMAP else "mmap"
with open(f"mmap_sandbox/resident_spike/results/phases_{tag}.json", "w") as f:
    json.dump(out, f, indent=2)
