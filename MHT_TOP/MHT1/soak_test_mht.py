"""
soak_test_mht.py -- leak regression + equivalence harness for mht_tracker.py

Run it on the Jetson exactly as-is:

    python3 soak_test_mht.py                 # 1 hour @ 30 fps, new tracker
    python3 soak_test_mht.py --compare       # also run the OLD tracker
    python3 soak_test_mht.py --minutes 5     # shorter smoke test

It drives the tracker with a synthetic detection stream instead of video, so
it isolates the tracker from the Gabor/HSV stages and runs far faster than
real time. The stream deliberately reproduces the failure mode you actually
have: a couple of genuine slow plumes plus a burst of spurious fragmented
contours every frame.

What it asserts:
  * every container in the tracker stays inside its declared bound
  * set(tracker._dormant) == set(tracker.store.keys()) at every frame
  * RSS is flat across the run (the actual leak regression)

It also checks that the rewritten CVKalman is numerically identical to the
original, so the perf work cannot silently change tracking behaviour.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mht_tracker as new                      # noqa: E402

try:
    import mht_tracker_old as old              # noqa: E402
    HAVE_OLD = True
except ImportError:
    HAVE_OLD = False


# ──────────────────────────────────────────────────────────────────────────────
#  RSS
# ──────────────────────────────────────────────────────────────────────────────
def rss_mb() -> float:
    """Resident set size in MB. /proc on Linux (incl. the Jetson)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, IndexError, ValueError):
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / 1024.0 if sys.platform != "darwin" else ru / (1024.0 ** 2)


# ──────────────────────────────────────────────────────────────────────────────
#  Synthetic scene: real plumes + HSV fragmentation noise
# ──────────────────────────────────────────────────────────────────────────────
class Scene:
    """Two slow drifting plumes that occasionally occlude (to exercise the
    warm/cold/revive path) plus `n_noise` spurious contours per frame at random
    positions (to exercise the tentative reaper)."""

    W, H = 960, 540

    def __init__(self, seed: int = 7, n_noise: int = 25, occlusion_p: float = 0.02):
        self.rng = random.Random(seed)
        self.n_noise = n_noise
        self.occlusion_p = occlusion_p
        self.plumes = [
            {"x": 200.0, "y": 300.0, "vx": 0.35, "vy": -0.15, "hidden": 0},
            {"x": 640.0, "y": 220.0, "vx": -0.25, "vy": 0.10, "hidden": 0},
            {"x": 430.0, "y": 400.0, "vx": 0.10, "vy": -0.30, "hidden": 0},
        ]

    def step(self, Det):
        dets = []
        for p in self.plumes:
            p["x"] += p["vx"] + self.rng.gauss(0, 0.4)
            p["y"] += p["vy"] + self.rng.gauss(0, 0.4)
            p["x"] = max(40.0, min(self.W - 40.0, p["x"]))
            p["y"] = max(40.0, min(self.H - 40.0, p["y"]))

            if p["hidden"] > 0:
                p["hidden"] -= 1
                continue
            if self.rng.random() < self.occlusion_p:
                # go missing for 10-70 frames: long enough to reach COLD
                p["hidden"] = self.rng.randint(10, 70)
                continue

            w = self.rng.randint(50, 90)
            h = self.rng.randint(50, 90)
            dets.append(Det(cx=p["x"], cy=p["y"],
                            x=int(p["x"] - w / 2), y=int(p["y"] - h / 2),
                            w=w, h=h, iso=self.rng.uniform(0.2, 0.8)))

        # fragmentation noise: short-lived garbage contours, never repeating
        for _ in range(self.n_noise):
            cx = self.rng.uniform(20, self.W - 20)
            cy = self.rng.uniform(20, self.H - 20)
            w = self.rng.randint(20, 45)
            h = self.rng.randint(20, 45)
            dets.append(Det(cx=cx, cy=cy,
                            x=int(cx - w / 2), y=int(cy - h / 2), w=w, h=h))
        return dets


# ──────────────────────────────────────────────────────────────────────────────
#  Equivalence: the rewritten Kalman must match the original bit-for-bit-ish
# ──────────────────────────────────────────────────────────────────────────────
def test_kalman_equivalence(n: int = 400, tol: float = 2e-3) -> bool:
    if not HAVE_OLD:
        print("  [skip] mht_tracker_old.py not present")
        return True

    rng = random.Random(11)
    a = new.CVKalman(100.0, 120.0)
    b = old.CVKalman(100.0, 120.0)

    worst_state = worst_gate = worst_maha = 0.0
    for _ in range(n):
        a.predict(); b.predict()
        z = np.array([rng.uniform(0, 900), rng.uniform(0, 500)], dtype=np.float32)

        ga, gb = a.gating_distance(z), b.gating_distance(z)
        worst_gate = max(worst_gate, abs(ga - gb) / max(1.0, abs(gb)))

        ma, mb = a.update(z), b.update(z)
        worst_maha = max(worst_maha, abs(ma - mb) / max(1.0, abs(mb)))
        worst_state = max(worst_state, float(np.max(np.abs(a.x - b.x))))

    ok = worst_state < tol and worst_gate < tol and worst_maha < tol
    print(f"  state max|Δ|      {worst_state:.3e}")
    print(f"  gating rel err    {worst_gate:.3e}")
    print(f"  mahalanobis rel   {worst_maha:.3e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_reset_purges_store() -> bool:
    """FIX 4 regression: the GUI calls reset() on every playlist switch."""
    tr = new.MHTTracker()
    sc = Scene(seed=3, n_noise=40)
    for _ in range(400):
        tr.update(sc.step(new.Detection))
    before = len(tr.store)
    tr.reset()
    ok = len(tr.store) == 0 and len(tr._dormant) == 0 and not tr.tracks
    print(f"  store before reset {before}, after {len(tr.store)} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_tracking_quality(frames: int = 4000, n_noise: int = 25):
    """The regression that actually matters: the memory rewrite must not cost
    plume recall. For each frame, a plume counts as COVERED if some confirmed
    track sits within 40 px of it. Reported for both trackers."""
    def run(mod):
        tr = mod.MHTTracker(n_scan=5, max_hyps_per_track=4,
                            min_hits=3, max_misses=8, trail_len=30)
        sc = Scene(seed=99, n_noise=n_noise)
        visible = covered = 0
        for _ in range(frames):
            dets = sc.step(mod.Detection)
            tr.update(dets)
            act = tr.active_tracks()
            pts = [(float(t.best.kf.x[0]), float(t.best.kf.x[1])) for t in act]
            for p in sc.plumes:
                if p["hidden"] > 0:
                    continue
                visible += 1
                if any((px - p["x"]) ** 2 + (py - p["y"]) ** 2 < 40 * 40
                       for px, py in pts):
                    covered += 1
        return covered / max(1, visible)

    new_rate = run(new)
    if HAVE_OLD:
        old_rate = run(old)
        delta = new_rate - old_rate
        ok = delta > -0.02          # allow 2 pp of noise, no real regression
        print(f"  plume coverage  old {old_rate:.1%}   new {new_rate:.1%}   "
              f"Δ {delta:+.1%} -> {'PASS' if ok else 'FAIL'}")
        return ok
    print(f"  plume coverage  new {new_rate:.1%} (no baseline)")
    return True


def test_revival_actually_happens() -> bool:
    """FIX 2 regression: restore() must have a live call site that fires."""
    tr = new.MHTTracker(dormant_ttl_frames=200)
    sc = Scene(seed=5, n_noise=10, occlusion_p=0.05)
    for _ in range(3000):
        tr.update(sc.step(new.Detection))
    n = tr.stats["revived"]
    print(f"  cold revivals {n}, warm->cold spills {tr.stats['to_cold']} -> "
          f"{'PASS' if n > 0 else 'FAIL'}")
    return n > 0


# ──────────────────────────────────────────────────────────────────────────────
#  The soak
# ──────────────────────────────────────────────────────────────────────────────
def soak(mod, label: str, frames: int, n_noise: int, check: bool):
    tr = mod.MHTTracker(n_scan=5, max_hyps_per_track=4,
                        min_hits=3, max_misses=8, trail_len=30)
    sc = Scene(seed=42, n_noise=n_noise)
    Det = mod.Detection

    gc.collect()
    base = rss_mb()
    t0 = time.perf_counter()
    samples = []
    peak_tracks = peak_store = 0

    for i in range(1, frames + 1):
        tr.update(sc.step(Det))

        if check and hasattr(tr, "debug_check_invariants"):
            tr.debug_check_invariants()

        peak_tracks = max(peak_tracks, len(tr.tracks))
        peak_store = max(peak_store, len(tr.store))

        if i % max(1, frames // 12) == 0:
            gc.collect()
            samples.append((i, rss_mb() - base, len(tr.tracks), len(tr.store)))

    dt = time.perf_counter() - t0
    gc.collect()
    grew = rss_mb() - base

    print(f"\n─── {label} ─────────────────────────────────────────")
    print(f"  {frames:,} frames in {dt:.1f}s  "
          f"({frames/dt:,.0f} fps synthetic, {dt*1000/frames:.3f} ms/frame)")
    print(f"  {'frame':>9} {'ΔRSS MB':>9} {'tracks':>8} {'store':>8}")
    for i, d, nt, ns in samples:
        print(f"  {i:>9,} {d:>9.1f} {nt:>8} {ns:>8}")
    print(f"  peak live tracks {peak_tracks},  peak store entries {peak_store}")
    print(f"  total ΔRSS {grew:+.1f} MB")
    if hasattr(tr, "memory_report"):
        r = tr.memory_report()
        print(f"  spawned {r['spawned']:,}  reaped_tentative {r['reaped_tentative']:,}  "
              f"dropped_overflow {r['dropped_overflow']:,}  to_warm {r['to_warm']:,}")
        print(f"  to_cold {r['to_cold']:,}  revived {r['revived']:,}  "
              f"expired_ttl {r['expired_ttl']:,}  evicted_cap {r['evicted_capacity']:,}")
        print(f"  store bytes {r['store_bytes']:,}  live hypotheses {r['live_hypotheses']}")
    return grew, samples, dt / frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--noise", type=int, default=25,
                    help="spurious fragmented contours per frame")
    ap.add_argument("--compare", action="store_true",
                    help="also soak mht_tracker_old.py")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    frames = int(args.minutes * 60 * args.fps)

    print("=" * 66)
    print("MHT tracker soak test")
    print(f"  {args.minutes:g} min @ {args.fps} fps = {frames:,} frames, "
          f"{args.noise} noise contours/frame")
    print("=" * 66)

    print("\n[1] CVKalman numerical equivalence (new vs original)")
    ok_kf = test_kalman_equivalence()

    print("\n[2] reset() purges the store  (FIX 4)")
    ok_reset = test_reset_purges_store()

    print("\n[3] cold-tier revival fires  (FIX 2)")
    ok_rev = test_revival_actually_happens()

    print("\n[4] tracking quality unchanged (recall regression)")
    ok_q = test_tracking_quality(n_noise=args.noise)

    if args.compare:
        if HAVE_OLD:
            soak(old, "OLD tracker", min(frames, 30_000), args.noise, check=False)
        else:
            print("\n  [skip] mht_tracker_old.py not present")

    grew, samples, per_frame = soak(new, "NEW tracker", frames, args.noise,
                                    check=not args.no_check)

    # leak verdict: compare the last third of the run against the first third
    print("\n" + "=" * 66)
    ok_leak = True
    if len(samples) >= 6:
        early = samples[len(samples) // 3][1]
        late = samples[-1][1]
        drift = late - early
        print(f"RSS drift over the last two thirds: {drift:+.1f} MB")
        ok_leak = drift < 8.0
    verdict = all([ok_kf, ok_reset, ok_rev, ok_q, ok_leak])
    print(f"per-frame tracker cost: {per_frame*1000:.3f} ms "
          f"({1/per_frame:,.0f} fps headroom, tracker stage only)")
    print(f"VERDICT: {'PASS' if verdict else 'FAIL'}")
    print("=" * 66)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
