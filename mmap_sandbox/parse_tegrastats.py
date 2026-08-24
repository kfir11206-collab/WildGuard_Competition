#!/usr/bin/env python3
import json
import re
import sys
from datetime import datetime

RE_TS = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")
RE_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
RE_SWAP = re.compile(r"SWAP (\d+)/(\d+)MB")
RE_CPU = re.compile(r"CPU \[([^\]]+)\]")
RE_GR3D = re.compile(r"GR3D_FREQ (\d+)%")
RE_TEMP = re.compile(r"(\w+)@([\d.]+)C")
RE_POWER = re.compile(r"(VDD_\w+) (\d+)mW")


def parse_line(line):
    m = RE_TS.match(line)
    if not m:
        return None
    rec = {"epoch": datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S").timestamp()}
    m = RE_RAM.search(line)
    rec["ram_mb"], rec["ram_total_mb"] = int(m.group(1)), int(m.group(2))
    m = RE_SWAP.search(line)
    rec["swap_mb"] = int(m.group(1))
    cores = []
    for c in RE_CPU.search(line).group(1).split(","):
        if c.startswith("off"):
            cores.append(0.0)
        else:
            cores.append(float(c.split("%")[0]))
    rec["cpu_pct"] = round(sum(cores) / len(cores), 1)
    rec["gpu_pct"] = int(RE_GR3D.search(line).group(1))
    for name, val in RE_TEMP.findall(line):
        rec[f"temp_{name}"] = float(val)
    for name, mw in RE_POWER.findall(line):
        rec.setdefault(f"power_{name.lower()}_mw", int(mw))
    return rec


def main():
    records = []
    for line in open(sys.argv[1]):
        rec = parse_line(line)
        if rec:
            records.append(rec)
    with open(sys.argv[2], "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    dur = records[-1]["epoch"] - records[0]["epoch"] if records else 0
    print(f"{len(records)} samples over {dur:.0f}s -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
