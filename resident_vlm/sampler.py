#!/usr/bin/env python3
"""Board-level power / thermal / SD sampler.

Reads the INA3221 rails straight from sysfs rather than parsing tegrastats,
because tegrastats only timestamps to the second and a wake is a few seconds long.
"""
import json
import threading
import time
from pathlib import Path

HWMON = next(Path("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon").glob("hwmon*"))
THERMAL = Path("/sys/class/thermal")


def _rails():
    out = {}
    for label in HWMON.glob("in*_label"):
        idx = label.name[2:-6]
        curr = HWMON / f"curr{idx}_input"
        volt = HWMON / f"in{idx}_input"
        if curr.exists() and volt.exists():
            out[label.read_text().strip()] = (volt, curr)
    return out


def _zones():
    out = {}
    for z in sorted(THERMAL.glob("thermal_zone*")):
        try:
            out[(z / "type").read_text().strip()] = z / "temp"
        except OSError:
            continue
    return out


RAILS = _rails()
ZONES = _zones()


def read_tj():
    try:
        return int(ZONES["tj-thermal"].read_text()) / 1000.0
    except (KeyError, OSError, ValueError):
        return None


DRIVE_LABELS = {"SDSQXFN": "sd_express", "CT1000P5PSSD8": "ssd"}


def device_stamp(dev="nvme0n1"):
    base = Path("/sys/block") / dev
    out = {"dev": dev}
    for field in ("model", "serial", "firmware_rev"):
        try:
            out[field] = (base / "device" / field).read_text().strip()
        except OSError:
            out[field] = None
    try:
        out["size_bytes"] = int((base / "size").read_text()) * 512
    except OSError:
        out["size_bytes"] = None
    model = out["model"] or ""
    out["label"] = next(
        (label for key, label in DRIVE_LABELS.items() if key in model), "unknown"
    )
    return out


def _diskstats(dev):
    with open("/proc/diskstats") as f:
        for line in f:
            p = line.split()
            if p[2] == dev:
                return {
                    "disk_reads_completed": int(p[3]),
                    "disk_read_bytes": int(p[5]) * 512,
                    "disk_write_bytes": int(p[9]) * 512,
                    "disk_read_ms": int(p[6]),
                    "disk_io_in_flight": int(p[11]),
                    "disk_io_ms": int(p[12]),
                }
    return {}


def _vmstat():
    """pswpin/pswpout count swap pages moved - lets the analysis separate SD
    reads that are model weights from SD reads that are only swap traffic."""
    out = {}
    try:
        with open("/proc/vmstat") as f:
            for line in f:
                k, _, v = line.partition(" ")
                if k in ("pswpin", "pswpout"):
                    out[f"{k}_bytes"] = int(v) * 4096
    except OSError:
        pass
    return out


def _meminfo():
    want = {"MemAvailable:": "mem_available_mb", "Cached:": "cached_mb",
            "SwapFree:": "swap_free_mb"}
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k = line.split()[0]
            if k in want:
                out[want[k]] = int(line.split()[1]) // 1024
    return out


def sample(dev="nvme0n1"):
    rec = {"t": time.monotonic(), "wall": time.time()}
    for name, (volt, curr) in RAILS.items():
        try:
            mw = int(volt.read_text()) * int(curr.read_text()) / 1000.0
        except (OSError, ValueError):
            continue
        rec[f"power_{name.lower()}_mw"] = round(mw, 1)
    for name, path in ZONES.items():
        try:
            rec[f"temp_{name.split('-')[0]}_c"] = int(path.read_text()) / 1000.0
        except (OSError, ValueError):
            continue
    rec.update(_diskstats(dev))
    rec.update(_meminfo())
    rec.update(_vmstat())
    return rec


class Sampler(threading.Thread):
    def __init__(self, path, hz=10.0, dev="nvme0n1"):
        super().__init__(daemon=True)
        self.path = Path(path)
        self.period = 1.0 / hz
        self.dev = dev
        self.stopping = threading.Event()
        self.marks = []

    def mark(self, name):
        self.marks.append({"name": name, "t": time.monotonic(), "wall": time.time()})

    def run(self):
        with open(self.path, "w") as f:
            while not self.stopping.is_set():
                f.write(json.dumps(sample(self.dev)) + "\n")
                f.flush()
                self.stopping.wait(self.period)

    def stop(self):
        self.stopping.set()
        self.join(timeout=5)
        self.path.with_suffix(".marks.json").write_text(json.dumps(self.marks, indent=2))
