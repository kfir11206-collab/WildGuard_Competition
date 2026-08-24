import json, os, shutil, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHUNK = 8 << 20
MARK = 256 << 20


def main():
    target_gb = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    target = int(target_gb * (1 << 30))
    path = HERE / "_writetmp.bin"

    free = shutil.disk_usage(HERE).free
    if free < target + (2 << 30):
        raise SystemExit(f"not enough free space: {free/1e9:.1f} GB free, need {target_gb+2:.0f}")

    buf = os.urandom(CHUNK)
    marks = []
    written = 0
    since = 0
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    t0 = time.perf_counter()
    prev = t0
    try:
        while written < target:
            os.write(fd, buf)
            written += CHUNK
            since += CHUNK
            if since >= MARK:
                os.fsync(fd)
                now = time.perf_counter()
                marks.append({"cum_gb": written / (1 << 30),
                              "t": now - t0,
                              "mbps": (since / 1e6) / (now - prev)})
                prev = now
                since = 0
        os.fsync(fd)
    finally:
        os.close(fd)
        os.unlink(path)

    total_s = time.perf_counter() - t0
    out = ROOT / "results/write_curve"
    out.mkdir(parents=True, exist_ok=True)
    overall = (written / 1e6) / total_s
    json.dump({"target_gb": target_gb, "chunk_mb": CHUNK / 1e6,
               "mark_mb": MARK / 1e6, "total_s": total_s,
               "overall_mbps": overall, "marks": marks},
              open(out / "write_curve.json", "w"), indent=2)
    vals = [m["mbps"] for m in marks]
    print(f"wrote {written/(1<<30):.1f} GiB in {total_s:.1f}s  overall {overall:.1f} MB/s")
    print(f"per-mark: first {vals[0]:.0f}  peak {max(vals):.0f}  last {vals[-1]:.0f}  min {min(vals):.0f} MB/s")


if __name__ == "__main__":
    main()
