import json, os, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SIZES = [4096, 16384, 65536, 262144, 1 << 20]
VOLUME = 256 << 20


def big_file():
    sr = json.load(open(ROOT / "results/sustained_rate/sustained_rate.json"))
    files = [(os.path.getsize(f), f) for f in sr["files"] if os.path.exists(f)]
    if not files:
        raise SystemExit("no model files exist")
    return max(files)[1]


def drop(f):
    fd = os.open(f, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def sweep(f, pattern):
    size = os.path.getsize(f)
    rng = random.Random(1234)
    results = []
    for bs in SIZES:
        drop(f)
        n = VOLUME // bs
        if pattern == "random":
            offs = [rng.randrange(0, max(1, size - bs)) // bs * bs for _ in range(n)]
        else:
            offs = [(i * bs) % (size - bs) for i in range(n)]
        fd = os.open(f, os.O_RDONLY)
        t0 = time.perf_counter()
        got = 0
        for off in offs:
            got += len(os.pread(fd, bs, off))
        dt = time.perf_counter() - t0
        os.close(fd)
        results.append({"pattern": pattern, "block_bytes": bs, "n": n,
                        "mb": got / 1e6, "s": dt, "mbps": (got / 1e6) / dt,
                        "iops": n / dt})
        print(f"{pattern:9s} bs={bs:>8d}  {(got/1e6)/dt:7.1f} MB/s  {n/dt:9.0f} IOPS")
    return results


def main():
    f = big_file()
    print(f"file: {f}  ({os.path.getsize(f)/1e6:.0f} MB)")
    rows = sweep(f, "random") + sweep(f, "sequential")
    out = ROOT / "results/rand_read_sweep"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"file": f, "volume_mb": VOLUME / 1e6, "rows": rows},
              open(out / "rand_read_sweep.json", "w"), indent=2)


if __name__ == "__main__":
    main()
