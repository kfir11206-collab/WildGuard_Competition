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

Memory management for edge hardware
-----------------------------------
`HypothesisStore` is the offload seam. Dormant / low-score track families are
serialized out of RAM to fast secondary storage (SD Express on the Orin Nano)
and rehydrated only if a nearby detection reappears. The default backend keeps
everything in RAM (a dict) so the module runs anywhere; swap in
`FileHypothesisStore(path)` to actually spill to disk. Nothing else in the
tracker changes -- the store is the single seam.

No OpenCV / NumPy-heavy ops in the hot loop beyond small fixed-size matmuls,
so this stays cheap relative to the Gabor stage upstream.
===============================================================================
"""

from __future__ import annotations

import math
import os
import pickle
import tempfile
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  Detection container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Detection:
    """One smoke candidate extracted from the Gabor-verified mask this frame."""
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
    """Minimal constant-velocity Kalman filter. Pure NumPy, fixed 4x4 / 2x4
    matrices -- a few microseconds per predict/update on the Orin Nano CPU."""

    __slots__ = ("x", "P", "F", "H", "Q", "R")

    def __init__(self, cx, cy, dt=1.0, q=2.0, r=4.0):
        # state: [x, y, vx, vy]^T
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float32)
        self.P = np.diag([10.0, 10.0, 100.0, 100.0]).astype(np.float32)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        # process noise (random-accel model, scaled by q)
        self.Q = (np.diag([0.25, 0.25, 1.0, 1.0]) * q).astype(np.float32)
        # measurement noise
        self.R = (np.eye(2) * r).astype(np.float32)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, z):
        z = np.asarray(z, dtype=np.float32)
        y = z - self.H @ self.x                       # innovation
        S = self.H @ self.P @ self.H.T + self.R       # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)      # 4x2 gain
        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P
        return float(y @ np.linalg.inv(S) @ y)        # squared Mahalanobis dist

    def gating_distance(self, z):
        """Mahalanobis^2 of measurement z against the predicted measurement,
        WITHOUT mutating state (used for association scoring)."""
        z = np.asarray(z, dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(y @ np.linalg.inv(S) @ y)


# ──────────────────────────────────────────────────────────────────────────────
#  A single trajectory hypothesis
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
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
        h = Hypothesis(kf=new_kf, score=self.score, depth=self.depth)
        h.area_hist = deque(self.area_hist, maxlen=8)
        return h


# ──────────────────────────────────────────────────────────────────────────────
#  A track = a family of competing hypotheses for one physical smoke region
# ──────────────────────────────────────────────────────────────────────────────
class Track:
    _next_id = 1

    # lifecycle
    TENTATIVE = 0
    CONFIRMED = 1
    DORMANT = 2          # offloaded / waiting for re-acquisition

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

    # ── best hypothesis helpers ────────────────────────────────────────────
    @property
    def best(self) -> Hypothesis:
        return max(self.hyps, key=lambda h: h.score)

    @property
    def score(self) -> float:
        return self.best.score

    def predict(self):
        for h in self.hyps:
            h.kf.predict()
            h.depth += 1
        self.age += 1

    # ── association: returns gating cost for the best-gating hypothesis ─────
    def gating_cost(self, det: Detection) -> float:
        return min(h.kf.gating_distance(det.centroid) for h in self.hyps)

    def update(self, det: Detection, ambiguous: bool):
        """Fold a detection into this track. If `ambiguous`, BRANCH: keep the
        current best hypothesis as-is (the 'maybe this detection isn't mine'
        explanation) and add an updated clone (the 'it is mine' explanation)."""
        self.hits += 1
        self.misses = 0

        gain_per_hyp = []
        for h in self.hyps:
            d2 = h.kf.update(det.centroid)
            # log-likelihood-ish increment: reward small Mahalanobis distance
            h.score += 2.0 - 0.5 * d2
            h.area_hist.append(det.area)
            gain_per_hyp.append(h)

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
            self.hyps = self.hyps[:max_hyps]

    def confirm_if_ready(self, min_hits: int):
        if self.state == Track.TENTATIVE and self.hits >= min_hits:
            self.state = Track.CONFIRMED


# ──────────────────────────────────────────────────────────────────────────────
#  Hypothesis offload store  (the SD-Express seam)
# ──────────────────────────────────────────────────────────────────────────────
class HypothesisStore:
    """In-RAM default backend. Subclass / swap for real secondary storage.

    The tracker calls offload()/restore() for dormant low-score track families.
    Keeping this behind an interface means the RAM-vs-SD decision is one line at
    construction, and the hot loop never touches disk directly."""

    def __init__(self):
        self._mem: dict[int, bytes] = {}

    def offload(self, track: Track) -> None:
        self._mem[track.id] = pickle.dumps(track, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, track_id: int) -> Optional[Track]:
        blob = self._mem.pop(track_id, None)
        return pickle.loads(blob) if blob is not None else None

    def drop(self, track_id: int) -> None:
        self._mem.pop(track_id, None)

    def __len__(self):
        return len(self._mem)


class FileHypothesisStore(HypothesisStore):
    """Spills dormant track families to a directory (mount this on SD Express:
    e.g. /mnt/sdexpress/mht). One small pickle per track; deterministic paths so
    offload/restore is O(1) and the filesystem does the LRU for us."""

    def __init__(self, path: str = None):
        super().__init__()
        self.path = path or os.path.join(tempfile.gettempdir(), "mht_offload")
        os.makedirs(self.path, exist_ok=True)

    def _p(self, tid: int) -> str:
        return os.path.join(self.path, f"track_{tid}.pkl")

    def offload(self, track: Track) -> None:
        with open(self._p(track.id), "wb") as f:
            pickle.dump(track, f, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, track_id: int) -> Optional[Track]:
        p = self._p(track_id)
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            t = pickle.load(f)
        os.remove(p)
        return t

    def drop(self, track_id: int) -> None:
        p = self._p(track_id)
        if os.path.exists(p):
            os.remove(p)

    def __len__(self):
        return len(os.listdir(self.path))


# ──────────────────────────────────────────────────────────────────────────────
#  The tracker
# ──────────────────────────────────────────────────────────────────────────────
class MHTTracker:
    """Track-oriented MHT with N-scan pruning and edge-friendly offloading.

    Typical usage per frame:
        dets = MHTTracker.detections_from_mask(verified_mask)
        tracker.update(dets)
        for t in tracker.active_tracks():
            ... draw t.bbox, t.trail, t.id, t.delta_area_pct ...
    """

    def __init__(
        self,
        gate_thresh: float = 9.21,   # chi-square 2-DOF, ~99% gate
        n_scan: int = 5,             # branch resolution depth (N-scan)
        max_hyps_per_track: int = 4, # breadth cap per track family
        min_hits: int = 3,           # tentative -> confirmed
        max_misses: int = 8,         # confirmed -> dormant -> deleted
        max_active_tracks: int = 24, # hard RAM cap; overflow gets offloaded
        trail_len: int = 30,         # polyline length (frames)
        store: Optional[HypothesisStore] = None,
    ):
        self.gate = gate_thresh
        self.n_scan = n_scan
        self.max_hyps = max_hyps_per_track
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.max_active = max_active_tracks
        self.trail_len = trail_len

        self.tracks: list[Track] = []
        self.store = store if store is not None else HypothesisStore()

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
        # 1) PREDICT every track family forward
        for t in self.tracks:
            t.predict()

        # 2) ASSOCIATE (greedy, gated). Build cost matrix track x detection.
        n_t, n_d = len(self.tracks), len(detections)
        unmatched_dets = set(range(n_d))

        if n_t and n_d:
            cost = np.full((n_t, n_d), np.inf, dtype=np.float32)
            for ti, t in enumerate(self.tracks):
                for di, d in enumerate(detections):
                    c = t.gating_cost(d)
                    if c <= self.gate:
                        cost[ti, di] = c

            # greedy assignment, smallest cost first
            matched_t = set()
            order = np.dstack(np.unravel_index(np.argsort(cost, axis=None),
                                               cost.shape))[0]
            track_match: dict[int, int] = {}
            for ti, di in order:
                ti, di = int(ti), int(di)
                if cost[ti, di] == np.inf:
                    break
                if ti in matched_t or di not in unmatched_dets:
                    continue
                # AMBIGUITY: this detection was also gated to >1 track -> branch
                competitors = int(np.sum(cost[:, di] <= self.gate))
                ambiguous = competitors > 1
                self.tracks[ti].update(detections[di], ambiguous=ambiguous)
                matched_t.add(ti)
                unmatched_dets.discard(di)
                track_match[ti] = di

            # tracks that got no detection this frame
            for ti, t in enumerate(self.tracks):
                if ti not in matched_t:
                    t.mark_missed()
        else:
            for t in self.tracks:
                t.mark_missed()

        # 3) SPAWN new tentative tracks from leftover detections
        for di in unmatched_dets:
            self.tracks.append(Track(detections[di], trail_len=self.trail_len))

        # 4) PRUNE hypotheses (N-scan + breadth) and resolve lifecycle
        for t in self.tracks:
            t.prune(self.n_scan, self.max_hyps)
            t.confirm_if_ready(self.min_hits)

        # 5) MEMORY MANAGEMENT: delete dead, offload dormant, cap active RAM
        self._manage_memory()

    # ── lifecycle + offload ──────────────────────────────────────────────────
    def _manage_memory(self):
        survivors: list[Track] = []
        for t in self.tracks:
            if t.misses > self.max_misses:
                # fully dead -> ensure no stale offload copy lingers
                self.store.drop(t.id)
                continue
            if t.misses > 0 and t.state == Track.CONFIRMED \
                    and t.misses >= self.max_misses // 2:
                # dormant: stop carrying it in RAM, spill to secondary storage.
                # It will be restored if a detection re-enters its gate region.
                t.state = Track.DORMANT
                self.store.offload(t)
                continue
            survivors.append(t)

        # hard RAM cap: if still over budget, offload the lowest-score families
        if len(survivors) > self.max_active:
            survivors.sort(key=lambda t: t.score, reverse=True)
            keep, spill = survivors[:self.max_active], survivors[self.max_active:]
            for t in spill:
                t.state = Track.DORMANT
                self.store.offload(t)
            survivors = keep

        self.tracks = survivors

    # ── reporting ────────────────────────────────────────────────────────────
    def active_tracks(self) -> list[Track]:
        """Confirmed, currently-visible tracks -- what the GUI should draw."""
        return [t for t in self.tracks
                if t.state == Track.CONFIRMED and t.misses == 0]

    def all_live_tracks(self) -> list[Track]:
        return list(self.tracks)

    def reset(self):
        self.tracks.clear()
        Track._next_id = 1


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