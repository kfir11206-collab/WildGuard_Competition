"""
motion_classifier.py
===============================================================================
Group-level SMOKE vs CLOUD discrimination from motion (Stage 6).

Why groups?
-----------
HSV-mask fragmentation means ONE physical smoke plume often becomes SEVERAL
MHT tracks, while a cumulus cloud is usually ONE track. Classifying single
tracks therefore punishes smoke twice: each fragment carries only part of the
plume's kinematic signature. This module first CLUSTERS active tracks into
physical-object groups (union-find over dilated bounding boxes), then
classifies each GROUP from the aggregate motion of all its members.

Signals (each mapped to a smoke-likeness in [0, 1], then weighted):

  upward      net mean velocity of all members points up (smoke rises,
              clouds translate laterally with the wind).
  anchoring   the group's BOTTOM edge stays put while the TOP rises — a
              plume is tied to its source; a cloud's whole box translates.
  wander      1 - straightness, where straightness = |net displacement| /
              path length of the group centroid. Clouds drift in a straight
              line; smoke meanders / billows.
  spread      circular spread of member velocity DIRECTIONS. Cloud fragments
              share one wind vector (coherent); smoke fragments diverge from
              the source. Neutral (0.5) when the group has a single member.
  growth      relative growth of the group's total area over the window.
              Smoke grows; a cloud's projected area is ~constant.

Output per group: smoke_score in [0,1] (EMA-smoothed) and a verdict with
hysteresis: SMOKE (>0.60), CLOUD (<0.40), "?" in between. Thresholds, window
and weights are constructor arguments so calibrate_kinematic.py can sweep
them.

Duck-typing: a "track" only needs .bbox = (x, y, w, h), .area, and a
velocity readable from .best.kf.x[2:4] (falls back to (0,0)). No Qt, no
OpenCV in the update path; draw_groups() imports cv2 lazily like
mht_tracker.draw_tracks.
===============================================================================
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  Per-frame snapshot of one group
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class _GroupObs:
    cx: float; cy: float          # area-weighted centroid
    x0: float; y0: float          # union bbox
    x1: float; y1: float
    area: float                   # total member area
    v: tuple                      # mean member velocity (vx, vy)
    dirs: list                    # member velocity directions [rad] (|v| gated)
    iso: float = -1.0             # area-weighted member texture isotropy


# ──────────────────────────────────────────────────────────────────────────────
#  A group = one physical object (possibly several fragment tracks)
# ──────────────────────────────────────────────────────────────────────────────
class Group:
    _next_id = 1

    def __init__(self, window: int):
        self.id = Group._next_id
        Group._next_id += 1
        self.hist: deque[_GroupObs] = deque(maxlen=window)
        self.track_ids: set[int] = set()
        self.score_ema: float = 0.5       # start undecided
        self.verdict: str = "?"
        self.signals: dict = {}           # last raw signals, for calibration
        self.misses = 0

    @property
    def bbox(self):
        o = self.hist[-1]
        return int(o.x0), int(o.y0), int(o.x1 - o.x0), int(o.y1 - o.y0)

    @property
    def n_members(self) -> int:
        return len(self.track_ids)


# ──────────────────────────────────────────────────────────────────────────────
#  The classifier
# ──────────────────────────────────────────────────────────────────────────────
class GroupMotionClassifier:

    def __init__(
        self,
        window: int = 30,          # frames of history per group
        min_frames: int = 12,      # don't judge younger groups
        merge_margin: float = 0.35,# bbox dilation (fraction of size) for grouping
        max_misses: int = 10,      # group GC
        ema: float = 0.15,         # score smoothing
        smoke_thr: float = 0.60,   # score above -> SMOKE
        cloud_thr: float = 0.40,   # score below -> CLOUD
        weights: dict | None = None,
        v_gate: float = 0.15,      # px/frame; below = "not really moving"
    ):
        self.window = window
        self.min_frames = min_frames
        self.margin = merge_margin
        self.max_misses = max_misses
        self.ema = ema
        self.smoke_thr = smoke_thr
        self.cloud_thr = cloud_thr
        self.v_gate = v_gate
        self.w = weights or {
            "upward":    0.28,
            "anchoring": 0.26,
            "wander":    0.16,
            "spread":    0.14,
            "growth":    0.16,
        }
        # Optional TRAINED model (see train_motion_classifier.py). When set,
        # it replaces the hand-weighted average with a calibrated logistic
        # regression over the signal vector. Load with load_model(path).
        self._model: dict | None = None
        self.major_life = 25       # frames a group must live to raise an alarm
        self.model_info = ""       # short description for the GUI
        self.groups: list[Group] = []

    def alarm(self) -> tuple[float, "Group | None"]:
        """The LIVE version of the trainer's clip decision: the best score
        among MAJOR groups (alive >= major_life frames). Returns
        (best_score, group). Alarm when best_score >= self.smoke_thr."""
        cap = min(self.major_life, self.window)
        best, bg = 0.0, None
        for g in self.groups:
            if g.misses == 0 and len(g.hist) >= cap and g.score_ema > best:
                best, bg = g.score_ema, g
        return best, bg

    def load_model(self, path: str) -> bool:
        """Load a motion_weights.json produced by train_motion_classifier.py.
        Returns True on success; silently keeps hand weights on failure."""
        import json, os
        try:
            if not os.path.isfile(path):
                return False
            with open(path, "r") as f:
                m = json.load(f)
            assert {"features", "mean", "std", "coef", "bias"} <= set(m)
            self._model = m
            if "smoke_thr" in m:
                self.smoke_thr = float(m["smoke_thr"])
            if "cloud_thr" in m:
                self.cloud_thr = float(m["cloud_thr"])
            self.major_life = int(m.get("major_life", 25))
            acc = m.get("loco_balanced_acc")
            self.model_info = (f"trained on {m.get('n_clips','?')} clips"
                               + (f", LOCO {acc*100:.0f}%" if acc else "")
                               + f", thr {self.smoke_thr:.2f}")
            return True
        except Exception:
            self._model = None
            return False

    def _model_score(self, s: dict) -> float:
        m = self._model
        x = np.array([float(s.get(k, 0.5)) for k in m["features"]], np.float64)
        x = (x - np.asarray(m["mean"])) / (np.asarray(m["std"]) + 1e-9)
        z = float(np.clip(x @ np.asarray(m["coef"]) + m["bias"], -30.0, 30.0))
        return float(1.0 / (1.0 + math.exp(-z)))

    def reset(self):
        self.groups.clear()
        Group._next_id = 1

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _track_velocity(t) -> tuple:
        try:
            vx, vy = float(t.best.kf.x[2]), float(t.best.kf.x[3])
        except Exception:
            vx, vy = 0.0, 0.0
        return vx, vy

    def _cluster(self, tracks) -> list[list]:
        """Union-find over dilated-bbox overlap -> lists of tracks."""
        n = len(tracks)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            parent[find(i)] = find(j)

        boxes = []
        for t in tracks:
            x, y, w, h = t.bbox
            m = self.margin * max(w, h)
            boxes.append((x - m, y - m, x + w + m, y + h + m))

        for i in range(n):
            for j in range(i + 1, n):
                a, b = boxes[i], boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    union(i, j)

        buckets: dict[int, list] = {}
        for i in range(n):
            buckets.setdefault(find(i), []).append(tracks[i])
        return list(buckets.values())

    @staticmethod
    def _observe(members) -> _GroupObs:
        """Aggregate one frame's member tracks into a group observation."""
        areas = np.array([max(t.area, 1.0) for t in members], np.float32)
        cxs = np.array([t.bbox[0] + t.bbox[2] / 2 for t in members], np.float32)
        cys = np.array([t.bbox[1] + t.bbox[3] / 2 for t in members], np.float32)
        wsum = float(areas.sum())
        cx = float((cxs * areas).sum() / wsum)
        cy = float((cys * areas).sum() / wsum)
        x0 = min(t.bbox[0] for t in members)
        y0 = min(t.bbox[1] for t in members)
        x1 = max(t.bbox[0] + t.bbox[2] for t in members)
        y1 = max(t.bbox[1] + t.bbox[3] for t in members)

        vs = [GroupMotionClassifier._track_velocity(t) for t in members]
        vx = float((np.array([v[0] for v in vs]) * areas).sum() / wsum)
        vy = float((np.array([v[1] for v in vs]) * areas).sum() / wsum)
        dirs = [math.atan2(v[1], v[0]) for v in vs
                if math.hypot(v[0], v[1]) > 0.15]
        isos = np.array([getattr(t, "iso_ema", -1.0) for t in members], np.float32)
        ok = isos >= 0.0
        iso = float((isos[ok] * areas[ok]).sum() / areas[ok].sum()) if ok.any() else -1.0
        return _GroupObs(cx, cy, x0, y0, x1, y1, wsum, (vx, vy), dirs, iso)

    # ── group identity across frames ─────────────────────────────────────────
    def _match_groups(self, clusters):
        """Greedy centroid matching of this frame's clusters to live groups.
        Group identity is intentionally sticky: fragments merging/splitting
        between frames keep feeding the same history."""
        obs = [self._observe(c) for c in clusters]
        used = set()
        assignments: list[tuple] = []           # (group | None, cluster_i)

        for gi, g in enumerate(self.groups):
            if not g.hist:
                continue
            last = g.hist[-1]
            best_ci, best_d = -1, 1e18
            for ci, o in enumerate(obs):
                if ci in used:
                    continue
                d = (o.cx - last.cx) ** 2 + (o.cy - last.cy) ** 2
                # gate: within half the group diagonal (+ slack for young ones)
                diag = max((last.x1 - last.x0), (last.y1 - last.y0)) + 40.0
                if d < best_d and d <= diag * diag:
                    best_ci, best_d = ci, d
            if best_ci >= 0:
                used.add(best_ci)
                assignments.append((g, best_ci))
            else:
                g.misses += 1

        for ci in range(len(obs)):
            if ci not in used:
                g = Group(self.window)
                self.groups.append(g)
                assignments.append((g, ci))

        for g, ci in assignments:
            g.misses = 0
            g.hist.append(obs[ci])
            g.track_ids = {t.id for t in clusters[ci]}

        self.groups = [g for g in self.groups if g.misses <= self.max_misses]

    # ── signals -> smoke score ───────────────────────────────────────────────
    def _score(self, g: Group) -> dict:
        H = list(g.hist)
        v = np.array([o.v for o in H], np.float32)            # (T,2)
        vmean = v.mean(axis=0)
        speed = float(np.hypot(*vmean))

        # 1) upward: direction of the net group velocity. Rising -> smoke-ish,
        #    lateral wind-drift -> actively cloud-ish, sinking -> cloud-ish.
        if speed < self.v_gate:
            upward = 0.5
        else:
            upness = -vmean[1] / speed          # +1 straight up, -1 down
            lateral = abs(vmean[0]) / speed     # 1 = pure horizontal drift
            upward = float(np.clip(0.5 + 0.5 * upness - 0.25 * lateral, 0.0, 1.0))

        # 2) anchoring: 2D movement of the bottom-center point vs top-center.
        #    Plume: source pinned, top rises -> ~1. Rigid translation (cloud):
        #    both move equally -> 0.5.
        xc = np.array([(o.x0 + o.x1) / 2 for o in H])
        top = np.array([o.y0 for o in H]); bot = np.array([o.y1 for o in H])
        top_mv = float(np.hypot(np.diff(xc), np.diff(top)).sum())
        bot_mv = float(np.hypot(np.diff(xc), np.diff(bot)).sum())
        tot = top_mv + bot_mv
        anchoring = 0.5 if tot < 2.0 else float(np.clip(top_mv / tot, 0.0, 1.0))

        # 3) wander: 1 - straightness of the centroid path
        cx = np.array([o.cx for o in H]); cy = np.array([o.cy for o in H])
        steps = np.hypot(np.diff(cx), np.diff(cy))
        path = float(steps.sum())
        net = float(math.hypot(cx[-1] - cx[0], cy[-1] - cy[0]))
        wander = 0.5 if path < 2.0 else float(np.clip(1.0 - net / path, 0.0, 1.0))

        # 4) spread: circular spread of member directions (multi-member only)
        dirs = [d for o in H for d in o.dirs]
        if g.n_members < 2 or len(dirs) < 6:
            spread = 0.5
        else:
            R = math.hypot(np.mean(np.cos(dirs)), np.mean(np.sin(dirs)))
            spread = float(np.clip(1.0 - R, 0.0, 1.0))    # R=1 coherent -> 0

        # 5) growth: relative area change per window (robust ends)
        a = np.array([o.area for o in H], np.float32)
        a0 = float(np.median(a[: max(3, len(a) // 4)]))
        a1 = float(np.median(a[-max(3, len(a) // 4):]))
        rel = (a1 - a0) / (a0 + 1e-3)
        growth = float(np.clip(0.5 + rel, 0.0, 1.0))      # +50% -> 1.0

        # 6) elong: vertical elongation of the group bbox (median h/w).
        #    Smoke columns are tall; wind-sheared clouds are wide.
        bw = np.array([o.x1 - o.x0 for o in H]); bh = np.array([o.y1 - o.y0 for o in H])
        elong = float(np.clip(np.median(bh / (bw + 1e-3)) / 2.0, 0.0, 1.0))

        # 7) vy: SIGNED vertical velocity in px/frame (rising 2 px/f -> 1.0).
        #    Unlike 'upward' this is not normalized by speed, so slow lateral
        #    drift can't hide behind the ratio.
        vy_sig = float(np.clip(0.5 - vmean[1] / 4.0, 0.0, 1.0))

        # 8) vx_abs: horizontal speed magnitude (wind drift indicator)
        vx_abs = float(np.clip(abs(vmean[0]) / 3.0, 0.0, 1.0))

        # 9) bot_still: absolute 2D stillness of the bottom edge per frame.
        #    A plume's source is pinned regardless of what the top does.
        bot_rate = bot_mv / max(len(H) - 1, 1)
        bot_still = float(np.clip(1.0 - bot_rate / 2.0, 0.0, 1.0))

        # 10) gabor_iso: texture isotropy of the members (0 = perfectly
        #     isotropic smoke-like texture). Neutral 0.5 when unmeasured.
        isos = [o.iso for o in H if o.iso >= 0.0]
        gabor_iso = 0.5 if not isos else float(np.clip(np.mean(isos) / 1.5, 0.0, 1.0))

        # 11) life: how much of the window this group has survived
        life = float(min(len(H) / max(self.window, 1), 1.0))

        return {"upward": upward, "anchoring": anchoring, "wander": wander,
                "spread": spread, "growth": growth, "elong": elong,
                "vy": vy_sig, "vx_abs": vx_abs, "bot_still": bot_still,
                "gabor_iso": gabor_iso, "life": life,
                "speed": speed, "n_members": g.n_members}

    # ── main entry point ─────────────────────────────────────────────────────
    def update(self, tracks) -> list[Group]:
        """Feed this frame's ACTIVE tracks. Returns the live groups."""
        clusters = self._cluster(list(tracks))
        self._match_groups(clusters)

        for g in self.groups:
            if g.misses > 0 or len(g.hist) < self.min_frames:
                continue
            s = self._score(g)
            g.signals = s
            if self._model is not None:
                raw = self._model_score(s)
            else:
                raw = sum(self.w[k] * s[k] for k in self.w) / sum(self.w.values())
            g.score_ema = (1 - self.ema) * g.score_ema + self.ema * raw
            # hysteresis: only flip when clearly past a threshold
            if g.score_ema >= self.smoke_thr:
                g.verdict = "SMOKE"
            elif g.score_ema <= self.cloud_thr:
                g.verdict = "CLOUD"
            elif g.verdict == "?":
                g.verdict = "?"        # keep previous verdict inside the band
        return [g for g in self.groups if g.misses == 0]


# ──────────────────────────────────────────────────────────────────────────────
#  Visualization (Quadrant 05 overlay) — Qt-free, lazy cv2 import
# ──────────────────────────────────────────────────────────────────────────────
def draw_groups(panel, clf: GroupMotionClassifier):
    """Draw group boxes + SMOKE/CLOUD verdicts on a BGR panel. Call AFTER
    draw_tracks so the group layer sits on top."""
    import cv2
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    COLORS = {"SMOKE": (60, 120, 245), "CLOUD": (245, 200, 90), "?": (160, 160, 160)}

    for g in clf.groups:
        if g.misses > 0 or not g.hist:
            continue
        x, y, w, h = g.bbox
        col = COLORS[g.verdict]
        cv2.rectangle(panel, (x, y), (x + w, y + h), col, 2)
        label = f"G{g.id} {g.verdict}  {g.score_ema:.2f}  ({g.n_members})"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.46, 1)
        ly = min(panel.shape[0] - 4, y + h + th + 6)
        cv2.rectangle(panel, (x, ly - th - 4), (x + tw + 4, ly + 2), (18, 18, 18), -1)
        cv2.putText(panel, label, (x + 2, ly - 2), _FONT, 0.46, col, 1, cv2.LINE_AA)