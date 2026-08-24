#!/usr/bin/env python3
import json
import mmap
import os
import sys
import time

CHUNK = 16 * 1024 * 1024

total = 0
t0 = time.perf_counter()
for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    mm.madvise(mmap.MADV_SEQUENTIAL)
    size = len(mm)
    for off in range(0, size, CHUNK):
        mm[off : min(off + CHUNK, size)]
    total += size
    mm.close()
    os.close(fd)

print(json.dumps({"mb": round(total / 1e6), "load_seconds": round(time.perf_counter() - t0, 2)}))
