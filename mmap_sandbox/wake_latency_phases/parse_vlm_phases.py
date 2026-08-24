#!/usr/bin/env python3
import json
import re
import sys
from datetime import datetime

MARKERS = [
    ("with AWQ", "python + imports"),
    ("patching model config", "torch/CUDA/framework init"),
    ("Benchmarking AWQ model", "AWQ LLM load + GPU init"),
    ("tokens/second", "startup self-benchmark"),
    ("loaded siglip vision model", "vision tower load"),
]

RE_TS = r"(\d{4}-\d{2}-\d{2}T[\d:.]+Z)"


def to_epoch(stamp):
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


def all_ts(text, marker):
    pattern = RE_TS + r"(?:(?!\d{4}-\d{2}-\d{2}T\d)[^\n])*?" + re.escape(marker)
    return [to_epoch(m) for m in re.findall(pattern, text)]


def main():
    vlm_log, watcher_log = sys.argv[1], sys.argv[2]
    text = open(vlm_log).read()
    stamps = {marker: all_ts(text, marker)[-1] for marker, _ in MARKERS}
    warnings = all_ts(text, "TRANSFORMERS_CACHE` is deprecated")
    first = max(w for w in warnings if w < stamps["with AWQ"])

    events = [json.loads(l) for l in open(watcher_log)]
    awake = next(e for e in events if e["event"] == "awake")
    total = awake["loader"]["first_output_seconds"]
    compose_up = awake["loader"]["compose_up_seconds"]

    phases = [("docker compose up", compose_up)]
    prev = first
    for marker, label in MARKERS:
        phases.append((label, round(stamps[marker] - prev, 2)))
        prev = stamps[marker]
    accounted = compose_up + (prev - first)
    phases.append(("container boot + camera + first inference", round(total - accounted, 2)))

    out = {"total_wake_seconds": total, "phases": phases}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
