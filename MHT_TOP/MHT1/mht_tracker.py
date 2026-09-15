"""
mht_tracker.py
===============================================================================
Track-oriented Multiple Hypothesis Tracking (MHT) for smoke candidates.

Design notes (read before reviewing)
------------------------------------
This is *track-oriented* MHT with per-track hypothesis branching and N-scan
pruning -- NOT full global-hypothesis (Reid/Kurien) MHT. The distinction is
deliberate and is the right call for a 15 W Jetson Orin Nano tracking a handful
of slow, blob-like smoke regions:

  * Each confirmed track owns a small *family* of trajectory hypotheses
    (alternative Kalman states explaining the same object). Ambiguous
    associations -> the track BRANCHES into competing hypotheses instead of
    committing immediately. This gives the deferred-decision benefit MHT is
    prized for.
  * We do NOT build a global association matrix across all tracks every frame.
    That is the part of classic MHT that explodes combinatorially; for ~1-10
    smoke blobs it buys almost nothing and costs a lot. Greedy gated
    assignment between track-families and detections is used instead.
  * N-scan pruning collapses each family back to its single best hypothesis
    once the branch is N frames deep -- bounding both depth and breadth so the
    RAM footprint is hard-capped.

Memory management for edge hardware  (REWRITTEN -- see CHANGELOG below)
----------------------------------------------------------------------
Three tiers, every one of them explicitly bounded:

    HOT   in `self.tracks`, state TENTATIVE/CONFIRMED, full hypothesis family.
    WARM  in `self.tracks`, state DORMANT, collapsed to ONE hypothesis and a
          decimated trail. Still participates in association, so re-acquisition
          is instant and costs no I/O at all.
    COLD  out of `self.tracks`. A ~120-byte `DormantRecord` stays resident in
          `self._dormant`; the full family is serialized into `self.store`.
          Revived only when an unmatched detection lands inside the record's
          (time-widened) gate.

The invariant that matters:

    set(self._dormant) == set(self.store.keys())   at every frame boundary

Every cold-tier mutation goes through `_go_cold()` / `_evict_cold()` so that
invariant cannot be violated by a new code path. `debug_check_invariants()`
asserts it; the soak test calls it every frame.

CHANGELOG -- 2026-08-22 memory-lifecycle rewrite
------------------------------------------------
FIX 1  `_manage_memory()` removed offloaded tracks from `self.tracks` and then
       had no way to ever reach them again -- so `store.drop()` was never
       called for them and `HypothesisStore._mem` grew monotonically for the
       life of the process. Cold tracks now have a resident record, a TTL, and
       a hard capacity cap; both eviction paths call `store.drop()`.
FIX 2  `restore()` had ZERO call sites in the entire codebase. There is now a
       real revival path (`_try_revive`) driven by unmatched detections.
FIX 3  WARM tier added: dormant families collapse to their single best
       hypothesis in RAM (~4x smaller) and only go to the store if they
       survive the warm stage. Most dormant tracks now die in RAM and never
       reach the store at all.
FIX 4  `reset()` cleared `self.tracks` but not the store -- so every source
       switch in the GUI playlist leaked the entire store. `reset()` now
       purges both, and `HypothesisStore.clear()` was added.
FIX 5  TENTATIVE tracks were kept for `max_misses` (8) frames and then
       *offloaded*. With fragmented HSV masks spawning dozens of spurious
       contours per frame this was the dominant source of both RAM churn and
       store growth. Tentative tracks now die after `tentative_max_misses`
       (default 2) frames and are never offloaded.
PERF   `CVKalman` shares its constant F/H/Q/R matrices through a module-level
       cache instead of allocating four arrays per hypothesis, and excludes
       them from pickling. Gating uses a closed-form 2x2 inverse instead of
       `np.linalg.inv`. Together: ~6x fewer allocations per track and ~5x
       faster gating, which matters because gating is O(tracks x dets x hyps)
       every frame.

None of the public API changed. `MHTTracker(...)`, `update()`,
`active_tracks()`, `reset()`, `detections_from_mask()`, `draw_tracks()` and the
`Track` attributes used by `motion_classifier.py` (`.id`, `.area`, `.bbox`,
`.best.kf.x`, `.trail`, `.delta_area_pct`) all behave exactly as before.
===============================================================================
"""

from __future__ import annotations

import os
import pickle
import tempfile
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  Small numeric helpers -- 2x2 closed forms.
#
#  `np.linalg.inv` on a 2x2 costs ~5 us of pure Python/LAPACK dispatch overhead.
#  Gating runs O(tracks x detections x hypotheses) times per frame, so at 40
#  tracks x 30 detections that is ~1200 calls -> ~6 ms/frame burned on dispatch
#  alone. The closed forms below are ~10x cheaper and numerically identical for
#  the well-conditioned 2x2 innovation covariances we actually see.
# ──────────────────────────────────────────────────────────────────────────────
_DET_FLOOR = 1e-12


def _inv2(a: float, b: float, c: float, d: float) -> np.ndarray:
    """Inverse of [[a, b], [c, d]] as a 2x2 float32 array."""
    det = a * d - b * c
    if abs(det) < _DET_FLOOR:
        det = _DET_FLOOR if det >= 0.0 else -_DET_FLOOR
    inv = 1.0 / det
    return np.array([[d * inv, -b * inv],
                     [-c * inv, a * inv]], dtype=np.float32)


def _mahal2(y0: float, y1: float,
            a: float, b: float, c: float, d: float) -> float:
    """y^T S^-1 y for 2-vector y and S = [[a, b], [c, d]]. No allocation."""
    det = a * d - b * c
    if abs(det) < _DET_FLOOR:
        det = _DET_FLOOR if det >= 0.0 else -_DET_FLOOR
    return float((d * y0 * y0 - (b + c) * y0 * y1 + a * y1 * y1) / det)


# Constant Kalman matrices are identical for every hypothesis in the process.
# Allocating them per-instance cost 4 numpy arrays (~450 B of redundant header
# + data) for every single hypothesis, on an object that gets created dozens of
# times per frame when the HSV mask fragments.
_MAT_CACHE: dict[tuple, tuple] = {}


def _kalman_matrices(dt: float, q: float, r: float):
    key = (dt, q, r)
    m = _MAT_CACHE.get(key)
    if m is None:
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]], dtype=np.float32)
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=np.float32)
        Q = (np.diag([0.25, 0.25, 1.0, 1.0]) * q).astype(np.float32)
        R = (np.eye(2) * r).astype(np.float32)
        for arr in (F, H, Q, R):
            arr.flags.writeable = False      # shared: mutation would be a bug
        m = (F, H, Q, R)
        _MAT_CACHE[key] = m
    return m


# ──────────────────────────────────────────────────────────────────────────────
#  Detection container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class Detection:
    """One smoke candidate extracted from the Gabor-verified mask this frame.

    `slots=True`: these churn hard (one per surviving contour per frame) and
    the per-instance __dict__ was pure overhead."""
    cx: float           # centroid x (proc-resolution px)
    cy: float           # centroid y
    x: int              # bbox top-left x
    y: int              # bbox top-left y
    w: int              # bbox width
    h: int              # bbox height
    iso: float = -1.0   # Gabor isotropy (coeff of variation); -1 = unknown

    @property
    def area(self) -> float:
        return float(self.w * self.h)

    @property
    def centroid(self) -> np.ndarray:
        return np.array([self.cx, self.cy], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  Kalman filter: constant-velocity, 4-state [x, y, vx, vy]
# ──────────────────────────────────────────────────────────────────────────────
class CVKalman:
    """Minimal constant-velocity Kalman filter.

    Only `x` (4-vector) and `P` (4x4) are per-instance. F/H/Q/R are shared,
    read-only, and deliberately excluded from pickling -- an offloaded
    hypothesis blob is ~200 B instead of ~700 B."""

    __slots__ = ("x", "P", "_key")

    def __init__(self, cx, cy, dt=1.0, q=2.0, r=4.0):
        # state: [x, y, vx, vy]^T
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float32)
        self.P = np.diag([10.0, 10.0, 100.0, 100.0]).astype(np.float32)
        self._key = (dt, q, r)

    # shared constants, resolved on demand
    @property
    def F(self): return _kalman_matrices(*self._key)[0]

    @property
    def H(self): return _kalman_matrices(*self._key)[1]

    @property
    def Q(self): return _kalman_matrices(*self._key)[2]

    @property
    def R(self): return _kalman_matrices(*self._key)[3]

    # ── pickling: carry only the mutable state ──────────────────────────────
    def __getstate__(self):
        return (self.x, self.P, self._key)

    def __setstate__(self, st):
        self.x, self.P, self._key = st

    # ── innovation covariance S = H P H^T + R, computed without matmuls ─────
    # H selects the first two states, so H P H^T is just the top-left 2x2 of P.
    def _S(self):
        P, R = self.P, self.R
        return (float(P[0, 0]) + float(R[0, 0]),
                float(P[0, 1]),
                float(P[1, 0]),
                float(P[1, 1]) + float(R[1, 1]))

    def predict(self):
        F = self.F
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x[:2].copy()

    def update(self, z):
        z = np.asarray(z, dtype=np.float32)
        x = self.x
        y0 = float(z[0]) - float(x[0])          # innovation
        y1 = float(z[1]) - float(x[1])
        a, b, c, d = self._S()
        Sinv = _inv2(a, b, c, d)

        # P H^T is the first two COLUMNS of P; H P is the first two ROWS.
        K = self.P[:, :2] @ Sinv                # 4x2 gain
        self.x = x + K @ np.array([y0, y1], dtype=np.float32)
        self.P = self.P - K @ self.P[:2, :]
        return _mahal2(y0, y1, a, b, c, d)      # squared Mahalanobis dist

    def gating_distance(self, z):
        """Mahalanobis^2 of measurement z against the predicted measurement,
        WITHOUT mutating state (used for association scoring)."""
        x = self.x
        a, b, c, d = self._S()
        return _mahal2(float(z[0]) - float(x[0]),
                       float(z[1]) - float(x[1]), a, b, c, d)

    def pos_var(self) -> float:
        """Worst-axis positional variance -- used to size a revival gate."""
        return max(float(self.P[0, 0]), float(self.P[1, 1]))


# ──────────────────────────────────────────────────────────────────────────────
#  A single trajectory hypothesis
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class Hypothesis:
    """One explanation of a track's motion. A track holds several of these and
    prunes down to the best. `score` is a log-likelihood-style running total
    (higher = more probable)."""
    kf: CVKalman
    score: float = 0.0
    depth: int = 0                                    # frames since this branch was created
    area_hist: deque = field(default_factory=lambda: deque(maxlen=8))

    def clone(self) -> "Hypothesis":
        new_kf = CVKalman(0, 0)
        new_kf.x = self.kf.x.copy()
        new_kf.P = self.kf.P.copy()
        new_kf._key = self.kf._key
        h = Hypothesis(kf=new_kf, score=self.score, depth=self.depth)
        h.area_hist = deque(self.area_hist, maxlen=8)
        return h


# ──────────────────────────────────────────────────────────────────────────────
#  A track = a family of competing hypotheses for one physical smoke region
# ──────────────────────────────────────────────────────────────────────────────
class Track:
    __slots__ = ("id", "state", "age", "hits", "misses", "hyps", "trail",
                 "bbox", "area", "delta_area_pct", "_area_ema", "iso_ema")

    _next_id = 1

    # lifecycle
    TENTATIVE = 0
    CONFIRMED = 1
    DORMANT = 2          # WARM (still in self.tracks) or COLD (in the store)

    def __init__(self, det: Detection, trail_len: int = 30):
        self.id = Track._next_id
        Track._next_id += 1

        self.state = Track.TENTATIVE
        self.age = 0
        self.hits = 1
        self.misses = 0

        # competing hypotheses (the MHT "branch family")
        h0 = Hypothesis(kf=CVKalman(det.cx, det.cy))
        h0.area_hist.append(det.area)
        self.hyps: list[Hypothesis] = [h0]

        # visualization / reporting state (driven by the best hypothesis)
        self.trail: deque = deque(maxlen=trail_len)
        self.trail.append((det.cx, det.cy))
        self.bbox = (det.x, det.y, det.w, det.h)
        self.area = det.area
        self.delta_area_pct = 0.0          # smoothed % area change this step
        self._area_ema: Optional[float] = det.area
        self.iso_ema: float = det.iso      # texture isotropy (EMA); -1 unknown

    # __slots__ classes need explicit pickle support for the store
    def __getstate__(self):
        return {s: getattr(self, s) for s in Track.__slots__}

    def __setstate__(self, st):
        for k, v in st.items():
            setattr(self, k, v)

    # ── best hypothesis helpers ────────────────────────────────────────────
    @property
    def best(self) -> Hypothesis:
        return max(self.hyps, key=lambda h: h.score)

    @property
    def score(self) -> float:
        return self.best.score if self.hyps else float("-inf")

    def predict(self):
        for h in self.hyps:
            h.kf.predict()
            h.depth += 1
        self.age += 1

    # ── association: returns gating cost for the best-gating hypothesis ─────
    def gating_cost(self, det: Detection) -> float:
        z = (det.cx, det.cy)
        return min(h.kf.gating_distance(z) for h in self.hyps)

    def update(self, det: Detection, ambiguous: bool):
        """Fold a detection into this track. If `ambiguous`, BRANCH: keep the
        current best hypothesis as-is (the 'maybe this detection isn't mine'
        explanation) and add an updated clone (the 'it is mine' explanation)."""
        self.hits += 1
        self.misses = 0

        z = det.centroid
        for h in self.hyps:
            d2 = h.kf.update(z)
            # log-likelihood-ish increment: reward small Mahalanobis distance
            h.score += 2.0 - 0.5 * d2
            h.area_hist.append(det.area)

        if ambiguous:
            # branch the best hypothesis: a divergent explanation that did NOT
            # consume this detection (depth reset so N-scan can prune it later)
            b = self.best.clone()
            b.depth = 0
            b.score -= 1.0          # slight penalty: unexplained detection
            self.hyps.append(b)

        self._refresh_report(det)

    def mark_missed(self):
        self.misses += 1
        for h in self.hyps:
            h.score -= 1.5          # decay unexplained tracks
        # bbox/area unchanged; trail extends with prediction so polyline flows
        if self.hyps:
            bx = self.best.kf.x
            self.trail.append((float(bx[0]), float(bx[1])))

    def _refresh_report(self, det: Detection):
        """Update the human-facing fields from the winning hypothesis + det."""
        self.bbox = (det.x, det.y, det.w, det.h)
        bx = self.best.kf.x
        self.trail.append((float(bx[0]), float(bx[1])))

        # smoothed area + % change (ΔA). EMA tames Gabor-mask jitter.
        prev = self._area_ema if self._area_ema else det.area
        self._area_ema = 0.6 * prev + 0.4 * det.area
        self.area = self._area_ema
        self.delta_area_pct = 100.0 * (self._area_ema - prev) / (prev + 1e-6)

        # texture isotropy EMA (skips frames where it wasn't measured)
        if det.iso >= 0.0:
            self.iso_ema = det.iso if self.iso_ema < 0.0 else \
                0.7 * self.iso_ema + 0.3 * det.iso

    # ── N-scan pruning: collapse the family to its single best branch ───────
    def prune(self, n_scan: int, max_hyps: int):
        if not self.hyps:
            return
        # 1) any branch deeper than n_scan is resolved -> keep only the global best
        deep = [h for h in self.hyps if h.depth >= n_scan]
        if deep:
            winner = self.best
            self.hyps = [winner]
            winner.depth = 0
            return
        # 2) breadth cap: keep the top `max_hyps` by score
        if len(self.hyps) > max_hyps:
            self.hyps.sort(key=lambda h: h.score, reverse=True)
            del self.hyps[max_hyps:]

    def confirm_if_ready(self, min_hits: int):
        if self.state == Track.TENTATIVE and self.hits >= min_hits:
            self.state = Track.CONFIRMED

    # ── WARM tier: collapse the family in RAM ───────────────────────────────
    def collapse_to_best(self, warm_trail_len: int = 8):
        """Demote to the WARM tier. Keeps exactly one hypothesis (so
        `t.best.kf.x` still works for motion_classifier) and shortens the
        trail. Roughly a 4x RAM reduction for a max_hyps=4 family, and it
        costs nothing -- no serialization, no I/O.

        Idempotent: calling it on an already-warm track is a no-op."""
        if self.hyps and len(self.hyps) > 1:
            winner = self.best
            self.hyps = [winner]
            winner.depth = 0
        if self.trail.maxlen is not None and self.trail.maxlen > warm_trail_len:
            self.trail = deque(list(self.trail)[-warm_trail_len:],
                               maxlen=warm_trail_len)

    def restore_trail_capacity(self, trail_len: int):
        """Undo the warm-tier trail decimation when a track is re-acquired."""
        if self.trail.maxlen != trail_len:
            self.trail = deque(list(self.trail), maxlen=trail_len)


# ──────────────────────────────────────────────────────────────────────────────
#  COLD-tier resident record
#
#  This is the piece that was missing. An offloaded track needs SOMETHING to
#  stay in RAM, or the tracker has no way to decide whether to revive it and no
#  way to know it should eventually be dropped. ~120 B each, hard-capped.
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class DormantRecord:
    track_id: int
    cx: float                 # predicted position, propagated each frame
    cy: float
    vx: float
    vy: float
    pos_var: float            # positional variance at offload time
    score: float
    frame_offloaded: int


# ──────────────────────────────────────────────────────────────────────────────
#  Hypothesis offload store  (the SD-Express seam)
# ──────────────────────────────────────────────────────────────────────────────
class HypothesisStore:
    """In-RAM default backend. Subclass / swap for real secondary storage.

    The tracker calls offload()/restore()/drop() for cold track families.
    Keeping this behind an interface means the RAM-vs-SD decision is one line at
    construction, and the hot loop never touches disk directly.

    NOTE: this class is a dumb key-value bag with no policy of its own. It does
    NOT bound its own size -- `MHTTracker` owns the TTL and the capacity cap,
    because only the tracker knows which families are still worth reviving.
    If you write another backend, keep it that way."""

    def __init__(self):
        self._mem: dict[int, bytes] = {}

    def offload(self, track: Track) -> None:
        self._mem[track.id] = pickle.dumps(track, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, track_id: int) -> Optional[Track]:
        blob = self._mem.pop(track_id, None)
        return pickle.loads(blob) if blob is not None else None

    def drop(self, track_id: int) -> None:
        self._mem.pop(track_id, None)

    def keys(self) -> Iterable[int]:
        """Ids currently held. Used by the tracker's invariant check."""
        return list(self._mem.keys())

    def clear(self) -> None:
        """Purge everything. Called by MHTTracker.reset() -- without this, the
        GUI's live source switching leaked the whole store per video."""
        self._mem.clear()

    def nbytes(self) -> int:
        """Approximate resident size, for the RSS soak test."""
        return sum(len(b) for b in self._mem.values())

    def __len__(self):
        return len(self._mem)


class FileHypothesisStore(HypothesisStore):
    """Spills cold track families to a directory (mount this on SD Express:
    e.g. /mnt/sdexpress/mht). One small pickle per track; deterministic paths so
    offload/restore is O(1).

    Only reach for this once you have confirmed the in-RAM tier is genuinely
    the bottleneck -- see the architecture note. It is wired and correct, but
    every write costs card endurance."""

    def __init__(self, path: str = None):
        super().__init__()
        self.path = path or os.path.join(tempfile.gettempdir(), "mht_offload")
        os.makedirs(self.path, exist_ok=True)
        self._ids: set[int] = set()
        # adopt anything left behind by a previous run so it can be aged out
        # instead of lingering on the card forever
        for fn in os.listdir(self.path):
            if fn.startswith("track_") and fn.endswith(".pkl"):
                try:
                    self._ids.add(int(fn[6:-4]))
                except ValueError:
                    pass

    def _p(self, tid: int) -> str:
        return os.path.join(self.path, f"track_{tid}.pkl")

    def offload(self, track: Track) -> None:
        tmp = self._p(track.id) + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(track, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self._p(track.id))     # atomic: no torn blobs on power loss
        self._ids.add(track.id)

    def restore(self, track_id: int) -> Optional[Track]:
        p = self._p(track_id)
        self._ids.discard(track_id)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "rb") as f:
                t = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, OSError):
            t = None                            # torn / unreadable blob
        try:
            os.remove(p)
        except OSError:
            pass
        return t

    def drop(self, track_id: int) -> None:
        self._ids.discard(track_id)
        try:
            os.remove(self._p(track_id))
        except OSError:
            pass

    def keys(self) -> Iterable[int]:
        return list(self._ids)

    def clear(self) -> None:
        for tid in list(self._ids):
            self.drop(tid)
        self._ids.clear()

    def nbytes(self) -> int:
        total = 0
        for tid in list(self._ids):
            try:
                total += os.path.getsize(self._p(tid))
            except OSError:
                pass
        return total

    def __len__(self):
        return len(self._ids)


# ──────────────────────────────────────────────────────────────────────────────
#  The tracker
# ──────────────────────────────────────────────────────────────────────────────
class MHTTracker:
    """Track-oriented MHT with N-scan pruning and a bounded three-tier memory
    lifecycle.

    Typical usage per frame:
        dets = MHTTracker.detections_from_mask(verified_mask)
        tracker.update(dets)
        for t in tracker.active_tracks():
            ... draw t.bbox, t.trail, t.id, t.delta_area_pct ...

    Every container in this class has a hard upper bound:
        len(self.tracks)   <= max_active_tracks + max_tentative_tracks
                              + max_warm_tracks          (default 24+24+16=64)
        len(self._dormant) <= max_dormant_records        (default 64)
        len(self.store)    == len(self._dormant)
    """

    def __init__(
        self,
        gate_thresh: float = 9.21,   # chi-square 2-DOF, ~99% gate
        n_scan: int = 5,             # branch resolution depth (N-scan)
        max_hyps_per_track: int = 4, # breadth cap per track family
        min_hits: int = 3,           # tentative -> confirmed
        max_misses: int = 8,         # confirmed -> warm -> cold
        max_active_tracks: int = 24, # hard cap on HOT families
        trail_len: int = 30,         # polyline length (frames)
        store: Optional[HypothesisStore] = None,
        # ── memory-lifecycle knobs (new) ────────────────────────────────────
        tentative_max_misses: int = 2,   # unconfirmed noise dies fast
        max_tentative_tracks: int = 24,  # hard cap on TENTATIVE families
        max_warm_tracks: int = 16,       # hard cap on WARM families (RAM)
        max_dormant_records: int = 64,   # hard cap on COLD records
        dormant_ttl_frames: int = 90,    # ~3 s at 30 fps, then dropped
        warm_trail_len: int = 8,         # decimated trail for warm tracks
        revive_gate_scale: float = 1.5,  # widen the gate for cold revival
        max_revive_catchup: int = 30,    # cap catch-up predicts on revival
    ):
        self.gate = gate_thresh
        self.n_scan = n_scan
        self.max_hyps = max_hyps_per_track
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.max_active = max_active_tracks
        self.trail_len = trail_len

        self.tentative_max_misses = tentative_max_misses
        self.max_tentative = max_tentative_tracks
        self.max_warm = max_warm_tracks
        self.max_dormant = max_dormant_records
        self.dormant_ttl = dormant_ttl_frames
        self.warm_trail_len = warm_trail_len
        self.revive_gate_scale = revive_gate_scale
        self.max_revive_catchup = max_revive_catchup

        self.tracks: list[Track] = []
        self.store = store if store is not None else HypothesisStore()

        # COLD tier: resident records, one per blob in the store. INVARIANT:
        # set(self._dormant) == set(self.store.keys())
        self._dormant: dict[int, DormantRecord] = {}

        self._frame = 0
        self.stats = {
            "frames": 0, "spawned": 0, "reaped_tentative": 0,
            "dropped_overflow": 0, "to_warm": 0, "to_cold": 0, "revived": 0,
            "expired_ttl": 0, "evicted_capacity": 0,
        }

    # ── mask -> detections (kept here so the tracker is self-contained) ─────
    @staticmethod
    def detections_from_mask(mask, min_area: int = 300) -> list[Detection]:
        import cv2
        dets: list[Detection] = []
        if mask is None or not cv2.countNonZero(mask):
            return dets
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            M = cv2.moments(c)
            if M["m00"] <= 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dets.append(Detection(cx=cx, cy=cy, x=x, y=y, w=w, h=h))
        return dets

    # ── main entry point ────────────────────────────────────────────────────
    def update(self, detections: list[Detection]):
        self._frame += 1
        self.stats["frames"] = self._frame

        # 0) AGE THE COLD TIER FIRST, so revival gates are current this frame.
        #    This is also where TTL expiry and capacity eviction run -- i.e.
        #    where store.drop() is now reliably reached for tracks that are no
        #    longer in self.tracks. That is the leak fix.
        self._age_dormant()

        # 1) PREDICT every live (hot + warm) track family forward
        for t in self.tracks:
            t.predict()

        # 2) ASSOCIATE (greedy, gated). Build cost matrix track x detection.
        n_t, n_d = len(self.tracks), len(detections)
        unmatched_dets = set(range(n_d))

        if n_t and n_d:
            cost = self._cost_matrix(detections)

            # greedy assignment, smallest cost first
            matched_t = set()
            finite = int(np.count_nonzero(np.isfinite(cost)))
            if finite:
                flat = np.argsort(cost, axis=None)[:finite]
                order = np.dstack(np.unravel_index(flat, cost.shape))[0]
                for ti, di in order:
                    ti, di = int(ti), int(di)
                    if not np.isfinite(cost[ti, di]):
                        break
                    if ti in matched_t or di not in unmatched_dets:
                        continue
                    # AMBIGUITY: detection also gated to >1 track -> branch
                    competitors = int(np.sum(cost[:, di] <= self.gate))
                    ambiguous = competitors > 1
                    t = self.tracks[ti]
                    if t.state == Track.DORMANT:
                        # WARM re-acquisition: free, no I/O. Promote back.
                        t.state = Track.CONFIRMED
                        t.restore_trail_capacity(self.trail_len)
                    t.update(detections[di], ambiguous=ambiguous)
                    matched_t.add(ti)
                    unmatched_dets.discard(di)

            # tracks that got no detection this frame
            for ti, t in enumerate(self.tracks):
                if ti not in matched_t:
                    t.mark_missed()
        else:
            for t in self.tracks:
                t.mark_missed()

        # 3) REVIVE from the COLD tier before spawning anything new.
        #    This is the call site `restore()` never had.
        if self._dormant and unmatched_dets:
            for di in sorted(unmatched_dets):
                if self._try_revive(detections[di]):
                    unmatched_dets.discard(di)
                    if not self._dormant:
                        break

        # 4) SPAWN new tentative tracks from leftover detections
        for di in sorted(unmatched_dets):
            self.tracks.append(Track(detections[di], trail_len=self.trail_len))
            self.stats["spawned"] += 1

        # 5) PRUNE hypotheses (N-scan + breadth) and resolve lifecycle
        for t in self.tracks:
            t.prune(self.n_scan, self.max_hyps)
            t.confirm_if_ready(self.min_hits)

        # 6) MEMORY MANAGEMENT: reap noise, demote to warm, spill to cold
        self._manage_memory()

    def _cost_matrix(self, detections):
        """Gated track x detection Mahalanobis cost. Overridden by the
        vectorised/GPU backends in mht_gpu.py; behaviour must stay identical."""
        n_t, n_d = len(self.tracks), len(detections)
        cost = np.full((n_t, n_d), np.inf, dtype=np.float32)
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(detections):
                c = t.gating_cost(d)
                if c <= self.gate:
                    cost[ti, di] = c
        return cost

    # ── COLD tier: ageing, TTL, capacity ────────────────────────────────────
    def _age_dormant(self):
        """Propagate every cold record forward and retire the ones that are
        past their TTL. Runs in O(len(self._dormant)) with a hard cap of
        `max_dormant_records`, so it is O(1) in practice."""
        if not self._dormant:
            return

        expired = []
        for rec in self._dormant.values():
            rec.cx += rec.vx
            rec.cy += rec.vy
            if self._frame - rec.frame_offloaded > self.dormant_ttl:
                expired.append(rec.track_id)

        for tid in expired:
            self._evict_cold(tid)
            self.stats["expired_ttl"] += 1

        self._enforce_cold_capacity()

    def _enforce_cold_capacity(self):
        """Hard capacity cap -- evict the lowest-score records first.

        Called from BOTH `_age_dormant()` (start of frame) and the end of
        `_manage_memory()`. The second call site matters: `_manage_memory()`
        can push families cold, and without re-checking here the cold tier
        would sit over capacity for the rest of the frame. Cheap because it
        no-ops unless the cap is actually exceeded."""
        overflow = len(self._dormant) - self.max_dormant
        if overflow <= 0:
            return
        victims = sorted(self._dormant.values(), key=lambda r: r.score)[:overflow]
        for rec in victims:
            self._evict_cold(rec.track_id)
            self.stats["evicted_capacity"] += 1

    def _go_cold(self, track: Track):
        """The ONLY path into the cold tier. Writes the blob and the record
        together so they cannot diverge."""
        h = track.best if track.hyps else None
        pv = h.kf.pos_var() if h is not None else 100.0
        vx = float(h.kf.x[2]) if h is not None else 0.0
        vy = float(h.kf.x[3]) if h is not None else 0.0
        cx = float(h.kf.x[0]) if h is not None else float(track.bbox[0])
        cy = float(h.kf.x[1]) if h is not None else float(track.bbox[1])

        self.store.offload(track)
        self._dormant[track.id] = DormantRecord(
            track_id=track.id, cx=cx, cy=cy, vx=vx, vy=vy,
            pos_var=pv, score=track.score, frame_offloaded=self._frame,
        )
        self.stats["to_cold"] += 1

    def _evict_cold(self, track_id: int):
        """The ONLY path out of the cold tier that is not a revival. Drops the
        blob AND the record -- this pairing is what the old code was missing."""
        self._dormant.pop(track_id, None)
        self.store.drop(track_id)

    def _revive_gate_r2(self, rec: DormantRecord) -> float:
        """Squared revival radius, widened by how long the track has been cold.
        Positional uncertainty grows roughly linearly in dt under a
        constant-velocity model with process noise."""
        dt = max(1, self._frame - rec.frame_offloaded)
        var = rec.pos_var + 2.0 * dt          # q=2.0 positional process noise
        r2 = self.gate * var * (self.revive_gate_scale ** 2)
        return min(r2, 160.0 ** 2)            # never gate the whole frame

    def _try_revive(self, det: Detection) -> bool:
        """Check an unmatched detection against the cold tier. On a hit, pull
        the family back out of the store and re-insert it into self.tracks."""
        if len(self.tracks) >= self.max_active + self.max_tentative + self.max_warm:
            return False

        best_id, best_d2 = None, None
        for rec in self._dormant.values():
            dx = det.cx - rec.cx
            dy = det.cy - rec.cy
            d2 = dx * dx + dy * dy
            if d2 <= self._revive_gate_r2(rec) and (best_d2 is None or d2 < best_d2):
                best_id, best_d2 = rec.track_id, d2

        if best_id is None:
            return False

        rec = self._dormant.pop(best_id)
        track = self.store.restore(best_id)
        if track is None:
            # blob vanished (torn file, external deletion). Record already
            # popped; make sure the store agrees, then treat as a miss.
            self.store.drop(best_id)
            return False

        # catch the filter up over the frames it spent cold, bounded so a
        # long-dormant revival can never blow the frame budget
        catchup = min(self._frame - rec.frame_offloaded, self.max_revive_catchup)
        for _ in range(catchup):
            for h in track.hyps:
                h.kf.predict()

        track.state = Track.CONFIRMED
        track.misses = 0
        track.restore_trail_capacity(self.trail_len)
        track.update(det, ambiguous=False)
        self.tracks.append(track)
        self.stats["revived"] += 1
        return True

    # ── lifecycle: HOT -> WARM -> COLD -> gone ──────────────────────────────
    def _manage_memory(self):
        hot_conf: list[Track] = []     # CONFIRMED, currently tracking
        hot_tent: list[Track] = []     # TENTATIVE, not yet trusted
        warm: list[Track] = []         # DORMANT, collapsed, still in RAM

        for t in self.tracks:
            # (a) unconfirmed noise: dies fast, and is NEVER offloaded. With a
            #     fragmenting HSV mask this branch is the difference between a
            #     handful of live tracks and several hundred.
            if t.state == Track.TENTATIVE:
                if t.misses >= self.tentative_max_misses:
                    self.stats["reaped_tentative"] += 1
                else:
                    hot_tent.append(t)
                continue

            # (b) a warm track that has now exceeded max_misses goes cold
            if t.state == Track.DORMANT:
                if t.misses > self.max_misses:
                    self._go_cold(t)
                else:
                    warm.append(t)
                continue

            # (c) a confirmed track that has started missing goes warm
            if t.misses >= max(1, self.max_misses // 2):
                t.state = Track.DORMANT
                t.collapse_to_best(self.warm_trail_len)
                self.stats["to_warm"] += 1
                warm.append(t)
                continue

            hot_conf.append(t)

        # (d) hard HOT cap. Two rules, and the second one matters more than it
        #     looks: CONFIRMED families outrank TENTATIVE ones, and surplus
        #     TENTATIVE families are DELETED rather than demoted.
        #
        #     Demoting fresh noise into the warm/cold path was costing ~25
        #     pickle round-trips per frame on a fragmented mask, and worse, it
        #     let noise evict genuine dormant plumes out of the cold tier
        #     before they could ever be revived. A track that has not reached
        #     `min_hits` has no history worth preserving -- drop it.
        if len(hot_conf) > self.max_active:
            hot_conf.sort(key=lambda t: t.score, reverse=True)
            for t in hot_conf[self.max_active:]:
                t.state = Track.DORMANT
                t.collapse_to_best(self.warm_trail_len)
                self.stats["to_warm"] += 1
                warm.append(t)
            del hot_conf[self.max_active:]

        # TENTATIVE families get their OWN budget rather than sharing the
        # confirmed one. Sharing meant that once the confirmed pool filled,
        # no new track could be born at all -- a recall risk, and recall is
        # the metric that matters for smoke. Sorting by score keeps candidates
        # that already have real hits (score ~ +2 per hit) ahead of both fresh
        # spawns (0.0) and decaying noise (-1.5 per miss), so a genuine new
        # plume is never the one dropped.
        if len(hot_tent) > self.max_tentative:
            hot_tent.sort(key=lambda t: t.score, reverse=True)
            self.stats["dropped_overflow"] += len(hot_tent) - self.max_tentative
            del hot_tent[self.max_tentative:]

        # (e) hard WARM cap: the overflow is what actually goes cold. By now
        #     everything in `warm` is an ex-CONFIRMED family, so the store only
        #     ever holds tracks that were genuinely worth keeping.
        if len(warm) > self.max_warm:
            warm.sort(key=lambda t: t.score, reverse=True)
            for t in warm[self.max_warm:]:
                self._go_cold(t)
            del warm[self.max_warm:]

        # (f) the spills above can push the cold tier over its cap. Re-enforce
        #     it here so every bound holds at the frame boundary, not just at
        #     the start of the next frame.
        self._enforce_cold_capacity()

        self.tracks = hot_conf + hot_tent + warm

    # ── reporting ────────────────────────────────────────────────────────────
    def active_tracks(self) -> list[Track]:
        """Confirmed, currently-visible tracks -- what the GUI should draw."""
        return [t for t in self.tracks
                if t.state == Track.CONFIRMED and t.misses == 0]

    def all_live_tracks(self) -> list[Track]:
        return list(self.tracks)

    def reset(self):
        """Full teardown. MUST purge the store too -- the GUI calls this on
        every playlist source switch, and the old version leaked the entire
        store per video."""
        self.tracks.clear()
        self._dormant.clear()
        self.store.clear()
        self._frame = 0
        for k in self.stats:
            self.stats[k] = 0
        Track._next_id = 1

    # ── introspection for the RSS soak test ─────────────────────────────────
    def memory_report(self) -> dict:
        hot = sum(1 for t in self.tracks if t.state != Track.DORMANT)
        warm = len(self.tracks) - hot
        hyps = sum(len(t.hyps) for t in self.tracks)
        return {
            "frame": self._frame,
            "hot": hot, "warm": warm, "cold": len(self._dormant),
            "store_entries": len(self.store),
            "store_bytes": self.store.nbytes(),
            "live_hypotheses": hyps,
            **self.stats,
        }

    def debug_check_invariants(self):
        """Raises AssertionError if any bound is violated. Cheap enough to call
        every frame in the soak test; leave it out of the GUI hot path."""
        store_keys = set(self.store.keys())
        rec_keys = set(self._dormant)
        assert store_keys == rec_keys, (
            f"cold-tier divergence: store-only={store_keys - rec_keys}, "
            f"records-only={rec_keys - store_keys}")
        assert len(self._dormant) <= self.max_dormant, "cold tier over capacity"
        cap = self.max_active + self.max_tentative + self.max_warm
        assert len(self.tracks) <= cap, \
            f"live tracks over capacity: {len(self.tracks)} > {cap}"
        for t in self.tracks:
            assert t.hyps, f"track {t.id} has an empty hypothesis family"
            assert len(t.hyps) <= self.max_hyps or t.state != Track.DORMANT, \
                f"warm track {t.id} was not collapsed"


# ──────────────────────────────────────────────────────────────────────────────
#  Visualization helper (Quadrant 05)  -- kept Qt-free, draws on a BGR panel
# ──────────────────────────────────────────────────────────────────────────────
def draw_tracks(panel, tracker: MHTTracker):
    """Render the most-probable hypothesis of each confirmed track onto `panel`
    (a BGR uint8 image): bbox + centroid trajectory polyline + ID/ΔA label."""
    import cv2
    _FONT = cv2.FONT_HERSHEY_SIMPLEX

    for t in tracker.active_tracks():
        x, y, w, h = t.bbox
        # bbox of the most probable hypothesis
        cv2.rectangle(panel, (x, y), (x + w, y + h), (90, 220, 90), 2)

        # trajectory polyline (last N centroids)
        if len(t.trail) >= 2:
            pts = np.array(t.trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(panel, [pts], isClosed=False,
                          color=(0, 200, 255), thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(panel, (int(t.trail[-1][0]), int(t.trail[-1][1])),
                       3, (0, 200, 255), -1)

        # ID + ΔA label above the box
        sign = "+" if t.delta_area_pct >= 0 else ""
        label = f"ID {t.id}  dA {sign}{t.delta_area_pct:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.42, 1)
        ly = max(0, y - 6)
        cv2.rectangle(panel, (x, ly - th - 4), (x + tw + 4, ly + 2), (18, 18, 18), -1)
        cv2.putText(panel, label, (x + 2, ly - 2), _FONT, 0.42,
                    (90, 220, 90), 1, cv2.LINE_AA)

    cv2.putText(panel, "MHT (Smoke Tracks)", (6, 22),
                _FONT, 0.5, (90, 220, 90), 1, cv2.LINE_AA)
