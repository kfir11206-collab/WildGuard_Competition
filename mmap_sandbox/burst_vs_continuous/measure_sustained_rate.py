#!/usr/bin/env python3
"""Measure sustained cold sequential-read throughput of the real VLM+Classifier
files on the SD card. Drops page cache first, walks each file in the same
16 MB chunks fire_watcher.py uses, and reports per-chunk + overall MB/s.

Used to pick the pacing rate for the 'continuous' mode in burst_vs_continuous.py.
"""
import json
import mmap
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "fire_watcher"))
from model_files import default_model_files

CHUNK = 16 * 1024 * 1024


def drop_cache(path):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def main():
    files = default_model_files()
    for f in files:
        drop_cache(f)

    chunks = []
    total = 0
    t_start = time.perf_counter()
    for path in files:
        fd = os.open(path, os.O_RDONLY)
        mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        size = len(mm)
        for off in range(0, size, CHUNK):
            c0 = time.perf_counter()
            mm[off:min(off + CHUNK, size)]
            c1 = time.perf_counter()
            n = min(off + CHUNK, size) - off
            chunks.append({"t": c1 - t_start, "mb": n / 1e6, "mbps": n / 1e6 / max(c1 - c0, 1e-9)})
            total += n
        mm.close()
        os.close(fd)
    t_total = time.perf_counter() - t_start

    overall_mbps = total / 1e6 / t_total
    rates = [c["mbps"] for c in chunks]
    n = len(rates)
    first_quarter = rates[: max(1, n // 4)]
    last_quarter = rates[-max(1, n // 4):]

    print(f"{len(files)} files, {total/1e6:.1f} MB total, {len(chunks)} chunks")
    print(f"overall (sustained) throughput: {overall_mbps:.1f} MB/s over {t_total:.2f}s")
    print(f"first-quarter chunk avg: {sum(first_quarter)/len(first_quarter):.1f} MB/s")
    print(f"last-quarter chunk avg:  {sum(last_quarter)/len(last_quarter):.1f} MB/s")
    print(f"min/max chunk: {min(rates):.1f} / {max(rates):.1f} MB/s")

    out = {
        "files": files,
        "total_mb": round(total / 1e6, 1),
        "total_s": round(t_total, 3),
        "overall_mbps": round(overall_mbps, 1),
        "chunks": chunks,
    }
    outpath = os.path.join(os.path.dirname(__file__), "..", "results", "sustained_rate", "sustained_rate.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"saved -> {outpath}")


if __name__ == "__main__":
    main()
