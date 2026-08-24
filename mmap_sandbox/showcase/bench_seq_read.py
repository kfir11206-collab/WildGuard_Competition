import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHUNK = 16 << 20


def model_files():
    sr = json.load(open(ROOT / "results/sustained_rate/sustained_rate.json"))
    files = [f for f in sr["files"] if os.path.exists(f)]
    if not files:
        raise SystemExit("no model files from sustained_rate.json exist")
    return files


def drop(files):
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def read_once(files):
    drop(files)
    chunks = []
    total = 0
    t0 = time.perf_counter()
    prev = t0
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        while True:
            b = os.read(fd, CHUNK)
            if not b:
                break
            now = time.perf_counter()
            total += len(b)
            chunks.append({"t": now - t0, "mb": len(b) / 1e6,
                           "mbps": (len(b) / 1e6) / max(now - prev, 1e-9)})
            prev = now
        os.close(fd)
    total_s = time.perf_counter() - t0
    return {"total_mb": total / 1e6, "total_s": total_s,
            "overall_mbps": (total / 1e6) / total_s, "chunks": chunks}


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    files = model_files()
    runs = []
    for i in range(reps):
        r = read_once(files)
        r["rep"] = i
        r["load1"] = os.getloadavg()[0]
        runs.append(r)
        print(f"rep {i}: {r['overall_mbps']:.1f} MB/s over {r['total_mb']:.0f} MB in {r['total_s']:.2f}s")
        time.sleep(1)
    out = ROOT / "results/seq_read_rerun"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"files": files, "chunk_mb": CHUNK / 1e6, "runs": runs},
              open(out / "seq_read_rerun.json", "w"), indent=2)
    vals = [r["overall_mbps"] for r in runs]
    print(f"\nmean {sum(vals)/len(vals):.1f} MB/s  min {min(vals):.1f}  max {max(vals):.1f}  n={len(vals)}")


if __name__ == "__main__":
    main()
