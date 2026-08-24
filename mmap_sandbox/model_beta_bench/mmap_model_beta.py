#!/usr/bin/env python3
import argparse
import json
import mmap
import os
import resource
import struct
import time

ADVICE = {
    "normal": mmap.MADV_NORMAL,
    "sequential": mmap.MADV_SEQUENTIAL,
    "random": mmap.MADV_RANDOM,
    "willneed": mmap.MADV_WILLNEED,
}


def drop_file_cache(path):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def read_bytes_from_disk():
    with open("/proc/self/io") as f:
        for line in f:
            key, value = line.split(":")
            if key == "read_bytes":
                return int(value)


def parse_safetensors_header(mm):
    header_len = struct.unpack("<Q", mm[:8])[0]
    header = json.loads(mm[8 : 8 + header_len].decode())
    header.pop("__metadata__", None)
    return header, 8 + header_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--advice", choices=ADVICE, default="normal")
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--tensor")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    if not args.warm:
        drop_file_cache(args.file)

    fd = os.open(args.file, os.O_RDONLY)
    size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    mm.madvise(ADVICE[args.advice])

    io_start = read_bytes_from_disk()
    t_start = time.perf_counter()
    bursts = []

    if args.tensor:
        header, data_start = parse_safetensors_header(mm)
        info = header[args.tensor]
        begin, end = info["data_offsets"]
        t0 = time.perf_counter()
        blob = mm[data_start + begin : data_start + end]
        dt = time.perf_counter() - t0
        processed = len(blob)
        print(
            f"tensor={args.tensor} shape={info['shape']} dtype={info['dtype']} "
            f"bytes={len(blob)} read_ms={dt * 1000:.1f}"
        )
    else:
        processed = size
        chunk = args.chunk_mb * 1024 * 1024
        for offset in range(0, size, chunk):
            t0 = time.perf_counter()
            blob = mm[offset : min(offset + chunk, size)]
            dt = time.perf_counter() - t0
            mbps = len(blob) / 1e6 / dt if dt > 0 else float("inf")
            bursts.append({"offset_mb": offset / 1e6, "size_mb": len(blob) / 1e6, "ms": dt * 1000, "mbps": mbps})
            print(f"  burst @ {offset / 1e6:7.0f} MB  {len(blob) / 1e6:5.0f} MB in {dt * 1000:7.1f} ms  ({mbps:6.0f} MB/s)")

    total_s = time.perf_counter() - t_start
    io_read = read_bytes_from_disk() - io_start
    majflt = resource.getrusage(resource.RUSAGE_SELF).ru_majflt

    print(
        f"advice={args.advice} warm={args.warm} chunk_mb={args.chunk_mb} "
        f"total_s={total_s:.2f} avg_mbps={processed / 1e6 / total_s:.0f} "
        f"disk_read_mb={io_read / 1e6:.0f} major_faults={majflt}"
    )

    if args.json_out:
        record = {
            "file": args.file,
            "file_size_mb": size / 1e6,
            "advice": args.advice,
            "warm": args.warm,
            "chunk_mb": args.chunk_mb,
            "total_s": total_s,
            "avg_mbps": processed / 1e6 / total_s,
            "disk_read_mb": io_read / 1e6,
            "major_faults": majflt,
            "load_average": os.getloadavg()[0],
            "bursts": bursts,
        }
        with open(args.json_out, "a") as f:
            f.write(json.dumps(record) + "\n")

    mm.close()
    os.close(fd)


if __name__ == "__main__":
    main()
