#!/usr/bin/env python3
import json
import mmap
import os
import socket
import sys
import time

from model_files import default_model_files

CHUNK = 16 * 1024 * 1024

sock_path = sys.argv[1]
files = sys.argv[2:] or default_model_files()

maps = []
for path in files:
    fd = os.open(path, os.O_RDONLY)
    maps.append(mmap.mmap(fd, 0, prot=mmap.PROT_READ))

if os.path.exists(sock_path):
    os.unlink(sock_path)
srv = socket.socket(socket.AF_UNIX)
srv.bind(sock_path)
srv.listen(1)
print("stub ready", flush=True)

while True:
    conn, _ = srv.accept()
    line = conn.makefile().readline().strip()
    if line == "quit":
        break
    t0 = time.perf_counter()
    total = 0
    for mm in maps:
        size = len(mm)
        for off in range(0, size, CHUNK):
            mm[off : min(off + CHUNK, size)]
        total += size
    reply = json.dumps({"mb": round(total / 1e6), "load_seconds": round(time.perf_counter() - t0, 2)})
    conn.sendall(reply.encode() + b"\n")
    conn.close()
