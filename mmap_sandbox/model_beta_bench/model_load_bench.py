#!/usr/bin/env python3
import argparse
import os
import resource
import time

import torch


def drop_file_cache(path):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def read_io_counters():
    counters = {}
    with open("/proc/self/io") as f:
        for line in f:
            key, value = line.split(":")
            counters[key] = int(value)
    return counters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mode", choices=["plain", "mmap"], required=True)
    parser.add_argument("--warm", action="store_true")
    args = parser.parse_args()

    if not args.warm:
        drop_file_cache(args.ckpt)

    io_start = read_io_counters()

    t0 = time.perf_counter()
    state_dict = torch.load(args.ckpt, map_location="cpu", mmap=(args.mode == "mmap"))
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    total_bytes = 0
    for name, tensor in state_dict.items():
        dest = torch.empty_like(tensor)
        dest.copy_(tensor)
        total_bytes += tensor.numel() * tensor.element_size()
        del dest
    t_consume = time.perf_counter() - t0

    usage = resource.getrusage(resource.RUSAGE_SELF)
    io_end = read_io_counters()

    print(
        f"mode={args.mode} warm={args.warm} "
        f"load_s={t_load:.2f} consume_s={t_consume:.2f} total_s={t_load + t_consume:.2f} "
        f"peak_rss_mb={usage.ru_maxrss / 1024:.0f} majflt={usage.ru_majflt} "
        f"read_mb={(io_end['read_bytes'] - io_start['read_bytes']) / 1e6:.0f} "
        f"tensor_mb={total_bytes / 1e6:.0f}"
    )


if __name__ == "__main__":
    main()
