#!/usr/bin/env python3
"""
stress_test_mht.py
===============================================================================
Empirical scaling ceiling for the MHT tracker.

Bypasses the sensor entirely: mock detections are injected straight into
MHTTracker.update() on a wide virtual grid, so the population is set by this
script rather than by how many blobs fit in a frame. Caps are raised in step
with the target so the tracker never clamps before the hardware does.

Per scale step it reports:
  * compute      -- mean / p95 ms inside update(), and the FPS that implies
  * RAM          -- process RSS, and MemAvailable left on the box
  * data rate    -- bytes actually handed to the store, as MB/s

Two data-rate columns matter and they say different things:
  MB/s @30fps    what the tier would emit IF the tracker kept real time
  MB/s @real     what it actually emits at the FPS this scale can sustain
The second is the honest one, and it is the number that peaks and then falls.

SAFETY: an unguarded run OOMs the board and the kernel picks the victim, which
on a desktop Jetson is often the session rather than this process. The default
guard aborts while MemAvailable is still healthy. To see a true OOM ceiling,
run it in a container with a memory limit so the cgroup kills only that:

    docker compose -f docker-compose_fused_system.yaml --profile test \\
      run --rm --entrypoint python3 mht_test -u stress_test_mht.py --to-oom

Run:
    python3 stress_test_mht.py
    python3 stress_test_mht.py --scales 1000,10000,100000 --frames 40
===============================================================================
"""

import argparse
import os
import pickle
import resource
import statistics
import sys
import time

MHT1 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MHT1)

from mht_tracker import Detection, HypothesisStore, MHTTracker
from mht_gpu import BACKENDS, CUPY_AVAILABLE


class CountingStore(HypothesisStore):
    """Real in-RAM store that also tallies what the cold tier writes."""

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.bytes = 0

    def offload(self, track):
        blob = pickle.dumps(track, protocol=pickle.HIGHEST_PROTOCOL)
        self._mem[track.id] = blob
        self.calls += 1
        self.bytes += len(blob)

    def take(self):
        c, b = self.calls, self.bytes
        self.calls = self.bytes = 0
        return c, b


class DiscardStore(CountingStore):
    """Counts the write but keeps nothing, isolating tracker RAM from store RAM."""

    def offload(self, track):
        blob = pickle.dumps(track, protocol=pickle.HIGHEST_PROTOCOL)
        self.calls += 1
        self.bytes += len(blob)

    def restore(self, track_id):
        return None

    def drop(self, track_id):
        pass

    def keys(self):
        return []

    def nbytes(self):
        return 0

    def __len__(self):
        return 0


def mem_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return float("inf")


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Injector:
    """Keeps `pop` live slots on a wide grid and churns a fraction each frame.

    Retired slots simply stop being fed, so the tracker ages them through
    misses -> warm -> cold, which is what drives store traffic. Spacing is far
    wider than the gate so every slot owns its own track family.
    """

    def __init__(self, pop, churn, spacing=64):
        self.pop = pop
        self.k = max(1, int(pop * churn))
        self.spacing = spacing
        self.next_slot = 0
        self.slots = [self._new() for _ in range(pop)]

    def _new(self):
        i = self.next_slot
        self.next_slot += 1
        cols = 4096
        return (i, (i % cols) * self.spacing, (i // cols) * self.spacing)

    def frame(self, t):
        self.slots = self.slots[self.k:] + [self._new() for _ in range(self.k)]
        jitter = 0.5 if t % 2 else -0.5
        out = []
        for _, gx, gy in self.slots:
            cx = gx + jitter
            cy = gy + jitter
            out.append(Detection(cx=cx, cy=cy, x=int(cx) - 10, y=int(cy) - 10,
                                 w=20, h=20, iso=0.1))
        return out


def run_scale(pop, frames, churn, keep_store, guard_mb, verbose, backend='loop'):
    store = CountingStore() if keep_store else DiscardStore()
    trk = BACKENDS[backend](
        store=store,
        max_active_tracks=pop,
        max_tentative_tracks=pop,
        max_warm_tracks=max(1, pop // 2),
        max_dormant_records=pop * 2,
    )
    inj = Injector(pop, churn)

    rss0 = rss_mb()
    times, rate_bytes = [], []
    aborted = None

    for t in range(frames):
        dets = inj.frame(t)
        avail = mem_available_mb()
        if avail < guard_mb:
            aborted = f"MemAvailable {avail:.0f} MB < guard {guard_mb:.0f} MB"
            break
        try:
            t0 = time.perf_counter()
            trk.update(dets)
            times.append((time.perf_counter() - t0) * 1000)
        except MemoryError:
            aborted = "MemoryError in update()"
            break
        _, b = store.take()
        rate_bytes.append(b)
        if verbose and t % 10 == 0:
            print(f"      frame {t:3d}  {times[-1]:8.1f} ms  "
                  f"tracks {len(trk.tracks):6d}  rss {rss_mb():7.0f} MB",
                  flush=True)

    if not times:
        return {"pop": pop, "aborted": aborted or "no frames", "mean": None}

    warm = max(1, len(times) // 4)
    body = times[warm:] or times
    mean = statistics.mean(body)
    p95 = sorted(body)[int(len(body) * 0.95) - 1] if len(body) > 1 else body[0]
    fps = 1000.0 / mean
    bpf = statistics.mean(rate_bytes[warm:] or rate_bytes)

    return {
        "pop": pop,
        "dets": len(inj.slots),
        "mean": mean,
        "p95": p95,
        "fps": fps,
        "tracks": len(trk.tracks),
        "dormant": len(trk._dormant),
        "store_mb": store.nbytes() / 1e6,
        "bpf": bpf,
        "mbs30": bpf * 30 / 1e6,
        "mbs_real": bpf * fps / 1e6,
        "rss": rss_mb(),
        "rss_delta": rss_mb() - rss0,
        "avail": mem_available_mb(),
        "aborted": aborted,
    }


def main():
    ap = argparse.ArgumentParser(description="MHT scaling stress test")
    ap.add_argument("--scales", default="100,300,1000,3000,10000,30000,100000,300000",
                    help="comma-separated track populations")
    ap.add_argument("--frames", type=int, default=30, help="frames per scale step")
    ap.add_argument("--churn", type=float, default=0.10,
                    help="fraction of slots retired per frame (drives evictions)")
    ap.add_argument("--guard-mb", type=float, default=700.0,
                    help="abort while MemAvailable is still above this")
    ap.add_argument("--to-oom", action="store_true",
                    help="disable the guard; only safe inside a memory-limited container")
    ap.add_argument("--keep-store", action="store_true",
                    help="retain offloaded blobs in RAM (default discards, to isolate tracker RAM)")
    ap.add_argument("--backend", default="loop", choices=list(BACKENDS),
                    help="cost-matrix backend: loop (stock), numpy (vectorised CPU), cupy (GPU)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    guard = 0.0 if args.to_oom else args.guard_mb
    scales = [int(s) for s in args.scales.split(",") if s.strip()]

    print(f"MemAvailable at start: {mem_available_mb():.0f} MB   "
          f"guard: {'DISABLED (--to-oom)' if args.to_oom else f'{guard:.0f} MB'}")
    print(f"frames/step {args.frames}   churn {args.churn:.0%}   "
          f"store {'kept' if args.keep_store else 'discarded'}   "
          f"backend {args.backend}"
          + ("" if args.backend != "cupy" else f"   (cupy available: {CUPY_AVAILABLE})"))
    print()
    hdr = (f"{'pop':>8} {'dets/f':>8} | {'mean ms':>9} {'p95 ms':>9} {'FPS':>8} "
           f"| {'tracks':>8} {'cold':>7} | {'B/frame':>10} {'MB/s@30':>9} {'MB/s@real':>10} "
           f"| {'RSS MB':>8} {'avail':>8}")
    print(hdr)
    print("-" * len(hdr))

    for pop in scales:
        r = run_scale(pop, args.frames, args.churn, args.keep_store, guard,
                      args.verbose, args.backend)
        if r["mean"] is None:
            print(f"{pop:>8} {'-':>8} | ABORTED: {r['aborted']}")
            break
        print(f"{r['pop']:>8} {r['dets']:>8} | {r['mean']:9.1f} {r['p95']:9.1f} {r['fps']:8.2f} "
              f"| {r['tracks']:>8} {r['dormant']:>7} | {r['bpf']:>10,.0f} {r['mbs30']:9.2f} "
              f"{r['mbs_real']:10.3f} | {r['rss']:8.0f} {r['avail']:8.0f}")
        if r["aborted"]:
            print(f"{'':>8} {'':>8} | stopped mid-step: {r['aborted']}")
            break
        if r["fps"] < 0.5:
            print(f"{'':>8} {'':>8} | FPS below 0.5 -- past any usable regime, stopping")
            break

    print()
    print("MB/s@real is the sustainable rate: bytes/frame x the FPS that scale can hold.")
    print("Card reference: ~600-700 MB/s burst, ~220-250 MB/s sustained after throttle.")


if __name__ == "__main__":
    main()
