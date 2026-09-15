"""
wildfire_gui.py
===============================================================================
Wildfire Detection Pipeline — Interactive Desktop GUI
Built with PyQt5 + OpenCV + PyWavelets

Pipeline stages:
  1. IIR Background Subtraction  — motion mask
  2. HSV Color Analysis          — smoke candidates
  3. Gabor Filter                — smoke texture verification

Panels:
  01 Original  |  02 IIR Motion  |  03 HSV Colour  |  04 Gabor·Smoke

Run:
    pip install PyQt5 opencv-python numpy PyWavelets
    python wildfire_gui.py                      # uses webcam (device 0)
    python wildfire_gui.py --video fire.mp4     # uses a file
===============================================================================
"""

import sys
import os
import argparse
import time

import cv2
import numpy as np

# All image-processing logic lives in the shared, Qt-free core so the GUI and
# the Optuna optimizer (optimize_pipeline.py) run identical detection code.
from wildfire_core import WildfirePipeline, MAX_PROC_W
from mht_tracker import MHTTracker, draw_tracks
from motion_classifier import GroupMotionClassifier, draw_groups

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider, QVBoxLayout,
    QHBoxLayout, QGroupBox, QScrollArea,
    QSizePolicy, QFrame,
    QListWidget, QListWidgetItem, QPushButton, QFileDialog, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon, QColor

# ==============================================================================
#  COLOUR PALETTE
# ==============================================================================
P = {
    "bg":           "#0d1117",
    "panel_bg":     "#161b22",
    "border":       "#30363d",
    "accent_fire":  "#f0883e",
    "accent_smoke": "#8b949e",
    "accent_go":    "#3fb950",
    "text_primary": "#e6edf3",
    "text_muted":   "#8b949e",
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {P['bg']};
    color: {P['text_primary']};
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    font-size: 12px;
}}
QGroupBox {{
    border: 1px solid {P['border']};
    border-radius: 6px;
    margin-top: 10px;
    padding: 8px 6px 6px 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: {P['accent_fire']};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {P['border']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {P['accent_fire']};
    border: 2px solid {P['bg']};
    width: 14px; height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {P['accent_fire']};
    border-radius: 2px;
}}
QScrollBar:vertical {{
    background: {P['bg']};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {P['border']};
    border-radius: 4px;
}}
"""

# ==============================================================================
#  PROCESSING THREAD
# ==============================================================================

class ProcessingThread(QThread):
    frames_ready = pyqtSignal(QPixmap, QPixmap, QPixmap, QPixmap, QPixmap)
    error_signal = pyqtSignal(str)

    def __init__(self, source, params: dict, ground_truth: str = ""):
        super().__init__()
        self.source = source
        self.params = params
        self._mutex = QMutex()
        self._running = True
        self._pending_source = None   # set by request_source() for live swap
        self._ground_truth = ground_truth   # "CLEAN" / "SMOKE" / "" — drawn as a tag
        self._rec_dir = None

        # All detection state (background model, Gabor bank)
        # lives inside the shared pipeline — single source of truth with the
        # optimizer. The GUI only adds the visualization panels on top.
        self.pipe = WildfirePipeline(params, max_proc_w=MAX_PROC_W)

        # Stage 5: temporal tracking of smoke candidates (MHT).
        # To spill dormant tracks to SD Express on the Orin Nano instead of RAM,
        # pass store=FileHypothesisStore("/mnt/sdexpress/mht").
        self.tracker = MHTTracker(
            n_scan=5, max_hyps_per_track=4,
            min_hits=3, max_misses=8, trail_len=30,
        )

        # Stage 6: SMOKE vs CLOUD from group motion. Fragment tracks of one
        # physical plume are clustered into a single group and judged from
        # their aggregate movement (see motion_classifier.py).
        self.group_clf = GroupMotionClassifier()
        # If a trained model exists (train_motion_classifier.py output),
        # use it instead of the hand-tuned weights.
        _mw = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "motion_weights.json")
        if self.group_clf.load_model(_mw):
            print(f"[motion] loaded trained model: {_mw}")

    def stop(self):
        with QMutexLocker(self._mutex):
            self._running = False

    def request_source(self, source, ground_truth: str = ""):
        """Thread-safe: queue a new video/camera source to switch to live."""
        with QMutexLocker(self._mutex):
            self._pending_source = source
            self._pending_gt = ground_truth

    def set_recording(self, out_dir):
        with QMutexLocker(self._mutex):
            self._rec_dir = out_dir

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error_signal.emit(f"Cannot open: {self.source}")
            return
        target_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
        _last_t     = time.perf_counter()
        _measured_fps = 0.0
        self._pending_gt = ""
        writers, writers_dir = None, None

        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break
                pending = self._pending_source
                self._pending_source = None
                pending_gt = getattr(self, "_pending_gt", "")
                self._pending_gt = ""
                rec_dir = self._rec_dir

            # Live source switch requested via keypress
            if pending is not None and pending != self.source:
                cap.release()
                cap = cv2.VideoCapture(pending)
                if not cap.isOpened():
                    self.error_signal.emit(f"Cannot open: {pending}")
                    # fall back to the old source so the app keeps running
                    cap = cv2.VideoCapture(self.source)
                else:
                    self.source = pending
                    self._ground_truth = pending_gt
                    target_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    self.pipe.reset()   # clear bg model
                    self.tracker.reset()
                    self.group_clf.reset()

            ret, frame = cap.read()
            if not ret:
                if isinstance(self.source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # Measure actual pipeline FPS
            now = time.perf_counter()
            dt  = now - _last_t
            _last_t = now
            if dt > 0:
                _measured_fps = 0.9 * _measured_fps + 0.1 * (1.0 / dt)

            _frame_start = time.perf_counter()

            # Downscale large frames — all stages run on this smaller resolution
            h0, w0 = frame.shape[:2]
            if w0 > MAX_PROC_W:
                frame = cv2.resize(frame, (MAX_PROC_W, int(h0 * MAX_PROC_W / w0)),
                                   interpolation=cv2.INTER_AREA)

            p        = dict(self.params)   # atomic snapshot of slider values
            gray_f32 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_u8  = cv2.convertScaleAbs(gray_f32)

            # Pre-build the dimmed base panels the core will draw its overlays on.
            # Passing a `panel` makes each stage method draw exactly the same
            # visualization the old inline code did — while the detection logic
            # itself lives only in wildfire_core.
            hsv_panel      = (frame * 0.35).astype(np.uint8)
            gabor_panel    = (frame * 0.35).astype(np.uint8)
            mht_panel      = (frame * 0.35).astype(np.uint8)

            motion = self.pipe.iir(gray_f32, p)
            smoke_m = self.pipe.hsv(frame, motion, p, panel=hsv_panel)
            smoke_v = self.pipe.gabor(gray_u8, smoke_m, p, panel=gabor_panel)

            # Stage 5 — MHT: extract smoke candidates from the verified mask,
            # step the tracker, and draw the most probable hypotheses.
            dets = MHTTracker.detections_from_mask(smoke_v, min_area=300)
            for d in dets:   # texture isotropy feeds the group classifier
                _, d.iso = self.pipe.texture_stats(
                    gray_u8, (d.x, d.y, d.w, d.h), p)
            self.tracker.update(dets)
            draw_tracks(mht_panel, self.tracker)

            # Stage 6 — group motion verdicts drawn on top of the tracks
            self.group_clf.update(self.tracker.active_tracks())
            draw_groups(mht_panel, self.group_clf)

            # Live ALARM banner — same decision rule as training: best score
            # among MAJOR (long-lived) groups vs the trained threshold.
            best, bg = self.group_clf.alarm()
            ph, pw = mht_panel.shape[:2]
            if bg is not None and best >= self.group_clf.smoke_thr:
                btxt = f"SMOKE ALERT  G{bg.id}  {best:.2f}"
                bcol = (60, 60, 235)
            else:
                btxt = f"monitoring   best {best:.2f}" if bg is not None \
                    else "monitoring"
                bcol = (120, 190, 120)
            (bw_, bh_), _ = cv2.getTextSize(btxt, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            bx = (pw - bw_) // 2
            cv2.rectangle(mht_panel, (bx - 8, 6), (bx + bw_ + 8, 18 + bh_),
                          (18, 18, 18), -1)
            cv2.putText(mht_panel, btxt, (bx, 12 + bh_),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, bcol, 2, cv2.LINE_AA)
            if self.group_clf.model_info:
                cv2.putText(mht_panel, self.group_clf.model_info,
                            (6, ph - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (150, 150, 150), 1, cv2.LINE_AA)

            iir_panel = cv2.cvtColor(motion, cv2.COLOR_GRAY2BGR)

            # In auto mode, show the parameters the pipeline actually chose
            # this frame so tuning behaviour is visible live.
            if p.get("iir_auto", 0):
                a_eff, t_eff = self.pipe.iir_effective
                txt = f"AUTO  a={a_eff:.3f}  thr={t_eff}  c={p.get('iir_auto_gain',1.0):.2f}"
                (tw2, th2), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(iir_panel, (6, 6), (14 + tw2, 18 + th2), (0, 0, 0), -1)
                cv2.putText(iir_panel, txt, (10, 10 + th2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 166, 88), 1,
                            cv2.LINE_AA)

            # Draw FPS overlay on original frame copy
            display_frame = frame.copy()
            fps_text = f"FPS: {_measured_fps:.1f}"
            (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(display_frame, (6, 6), (14 + tw, 18 + th), (0, 0, 0), -1)
            cv2.putText(display_frame, fps_text, (10, 10 + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 80), 1, cv2.LINE_AA)

            # Ground-truth tag (top-right) from the folder name: CLEAN / SMOKE
            gt = self._ground_truth
            if gt:
                fw = display_frame.shape[1]
                gt_text = f"GT: {gt}"
                col = (90, 220, 90) if gt == "CLEAN" else (60, 120, 245)  # green / orange-red
                (gtw, gth), _ = cv2.getTextSize(gt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                gx = fw - gtw - 14
                cv2.rectangle(display_frame, (gx - 6, 6), (fw - 4, 18 + gth), (0, 0, 0), -1)
                cv2.putText(display_frame, gt_text, (gx, 12 + gth),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

            if writers is not None and rec_dir != writers_dir:
                for wr in writers:
                    wr.release()
                writers = None
            if rec_dir and writers is None:
                rec_size = display_frame.shape[1::-1]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writers = [cv2.VideoWriter(os.path.join(rec_dir, n + ".mp4"),
                                           fourcc, target_fps, rec_size)
                           for n in REC_NAMES]
                writers.append(cv2.VideoWriter(
                    os.path.join(rec_dir, "all_panels.mp4"), fourcc, target_fps,
                    (3 * rec_size[0], 2 * rec_size[1])))
                writers_dir = rec_dir
            if writers is not None:
                panels = [f if f.shape[1::-1] == rec_size else cv2.resize(f, rec_size)
                          for f in (display_frame, iir_panel, hsv_panel,
                                    gabor_panel, mht_panel)]
                for wr, f in zip(writers, panels + [_grid(panels)]):
                    wr.write(f)

            self.frames_ready.emit(
                _to_pixmap(display_frame),
                _to_pixmap(iir_panel),
                _to_pixmap(hsv_panel),
                _to_pixmap(gabor_panel),
                _to_pixmap(mht_panel),
            )
            elapsed_ms = (time.perf_counter() - _frame_start) * 1000
            self.msleep(max(1, int(1000 / target_fps) - int(elapsed_ms)))

        cap.release()
        for wr in writers or []:
            wr.release()


# ==============================================================================
#  UTILITIES
# ==============================================================================

def _to_pixmap(bgr: np.ndarray) -> QPixmap:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qi  = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qi)


REC_NAMES = ["01_original", "02_iir_motion", "03_hsv_colour",
             "04_gabor_smoke", "05_mht_tracks"]


def _grid(panels):
    h, w = panels[0].shape[:2]
    out = np.zeros((2 * h, 3 * w, 3), np.uint8)
    for i, f in enumerate(panels[:3]):
        out[:h, i * w:(i + 1) * w] = f
    for i, f in enumerate(panels[3:]):
        x = w // 2 + i * w
        out[h:, x:x + w] = f
    return out

# ==============================================================================
#  SLIDER ROW WIDGET
# ==============================================================================

class SliderRow(QWidget):
    value_changed = pyqtSignal(float)

    def __init__(self, label, lo, hi, default, decimals=2, parent=None):
        super().__init__(parent)
        self._decimals = decimals
        self._scale    = 10 ** decimals
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(6)

        lbl = QLabel(label)
        lbl.setFixedWidth(126)
        lbl.setStyleSheet(f"color: {P['text_muted']}; font-size: 11px;")
        lay.addWidget(lbl)

        self._s = QSlider(Qt.Horizontal)
        self._s.setMinimum(int(lo * self._scale))
        self._s.setMaximum(int(hi * self._scale))
        self._s.setValue(int(default * self._scale))
        lay.addWidget(self._s, 1)

        self._vl = QLabel(f"{default:.{decimals}f}")
        self._vl.setFixedWidth(46)
        self._vl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._vl.setStyleSheet(
            f"color: {P['accent_fire']}; font-weight: 600; font-size: 11px;")
        lay.addWidget(self._vl)

        self._s.valueChanged.connect(self._changed)

    def _changed(self, raw):
        v = raw / self._scale
        self._vl.setText(f"{v:.{self._decimals}f}")
        self.value_changed.emit(v)

    def value(self):
        return self._s.value() / self._scale

# ==============================================================================
#  VIDEO PANEL WIDGET
# ==============================================================================

class VideoPanel(QWidget):
    def __init__(self, number: str, title: str, accent: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        title_bar = QWidget()
        title_bar.setFixedHeight(26)
        title_bar.setStyleSheet(
            f"background: {P['panel_bg']}; border-radius: 4px 4px 0 0;")
        tb_lay = QHBoxLayout(title_bar)
        tb_lay.setContentsMargins(8, 0, 8, 0)

        num_lbl = QLabel(number)
        num_lbl.setStyleSheet(
            f"color: {accent}; font-size: 10px; font-weight: 700; letter-spacing: 1px;")
        tb_lay.addWidget(num_lbl)

        t_lbl = QLabel(title.upper())
        t_lbl.setStyleSheet(
            f"color: {P['text_muted']}; font-size: 9px; letter-spacing: 0.8px;")
        tb_lay.addWidget(t_lbl, 1)
        lay.addWidget(title_bar)

        strip = QFrame()
        strip.setFixedHeight(2)
        strip.setStyleSheet(f"background: {accent};")
        lay.addWidget(strip)

        self._img = QLabel()
        self._img.setAlignment(Qt.AlignCenter)
        self._img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img.setStyleSheet("background: #000;")
        lay.addWidget(self._img, 1)

    def update_frame(self, pix: QPixmap):
        self._img.setPixmap(pix.scaled(
            self._img.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

# ==============================================================================
#  MAIN WINDOW
# ==============================================================================

class WildfireApp(QMainWindow):

    PANEL_DEFS = [
        ("01", "Original",          P["text_muted"]),
        ("02", "IIR Motion",        "#58a6ff"),
        ("03", "HSV Colour",        "#f0883e"),
        ("04", "Gabor · Smoke",     "#8b949e"),
        ("05", "MHT · Tracks",      "#3fb950"),
    ]

    def __init__(self, source, folder=None):
        super().__init__()
        self.setWindowTitle("🔥  Wildfire Detection Pipeline")
        self.setMinimumSize(1500, 860)

        # Playlist state. You point at a PARENT folder that contains "smoke" and
        # "clean" subfolders; videos from BOTH are listed, and each video carries
        # the ground-truth label of the subfolder it lives in. _playlist holds
        # (path, label) tuples; _play_idx is the one currently playing.
        self.folder = None
        self.ground_truth = ""                 # label of the current video
        self._playlist: list[tuple[str, str]] = []
        self._play_idx = -1
        self.setFocusPolicy(Qt.StrongFocus)

        # Resolve the initial source: explicit folder > single video > webcam.
        if folder:
            self._load_folder(folder, autoplay=False)
            if self._playlist:
                self._play_idx = 0
                source = self._playlist[0][0]
                self.ground_truth = self._playlist[0][1]   # tag the first clip
            else:
                self._play_idx = -1
        self.source = source

        self.params: dict = {
            # Stage 1 — IIR
            "iir_auto":              0,      # 1 = adaptive alpha/threshold
            "iir_auto_gain":         1.0,    # c: scales the auto threshold
            "iir_alpha":             0.05,
            "iir_threshold":         12,
            # Stage 2 — HSV Smoke
            "smoke_h_min":           0,
            "smoke_h_max":           179,
            "smoke_s_min":           0,
            "smoke_s_max":           153,
            "smoke_v_min":           30,
            "smoke_v_max":           255,
            # Stage 3 — Gabor
            "gabor_ksize":           15,
            "gabor_sigma":           3.0,
            "gabor_lambda":          8.0,
            "gabor_gamma":           0.5,
            "gabor_std_thresh":      0.4,
        }

        self._build_ui()
        self._start_thread()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root_widget = QWidget()
        self.setCentralWidget(root_widget)
        root = QVBoxLayout(root_widget)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        root.addWidget(self._make_header())

        # Main horizontal layout: controls (left) | video grid (right)
        main_row = QWidget()
        main_lay = QHBoxLayout(main_row)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(8)

        # ── Left: controls, full height ───────────────────────────────────────
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setFixedWidth(300)
        ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ctrl_inner = QWidget()
        cl = QVBoxLayout(ctrl_inner)
        cl.setSpacing(7)
        cl.setContentsMargins(4, 4, 4, 4)
        self._build_playlist(cl)
        self._build_capture(cl)
        self._build_controls(cl)
        cl.addStretch()
        ctrl_scroll.setWidget(ctrl_inner)
        main_lay.addWidget(ctrl_scroll)

        # ── Right: 5 video panels — 3 top + 2 bottom ─────────────────────────
        vid_grid = QWidget()
        vg = QVBoxLayout(vid_grid)
        vg.setContentsMargins(0, 0, 0, 0)
        vg.setSpacing(5)
        self._panels: list[VideoPanel] = []

        top_row = QWidget()
        top_lay = QHBoxLayout(top_row)
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setSpacing(5)

        bot_row = QWidget()
        bot_lay = QHBoxLayout(bot_row)
        bot_lay.setContentsMargins(0, 0, 0, 0)
        bot_lay.setSpacing(5)

        for i, (num, title, accent) in enumerate(self.PANEL_DEFS):
            vp = VideoPanel(num, title, accent)
            self._panels.append(vp)
            if i < 3:
                top_lay.addWidget(vp, 1)
            else:
                bot_lay.addWidget(vp, 1)

        vg.addWidget(top_row, 1)
        vg.addWidget(bot_row, 1)

        main_lay.addWidget(vid_grid, 1)

        root.addWidget(main_row, 1)

        # Status bar
        sb = self.statusBar()
        sb.setStyleSheet(
            f"background:{P['panel_bg']}; color:{P['text_muted']}; font-size:10px;")
        self._status_lbl = QLabel("● RUNNING")
        self._status_lbl.setStyleSheet(
            f"color:{P['accent_go']}; font-weight:600; font-size:11px;")
        sb.addPermanentWidget(self._status_lbl)
        src_lbl = QLabel(f"Source: {self.source}")
        src_lbl.setStyleSheet(
            f"color:{P['text_muted']}; font-size:10px; padding:0 10px;")
        sb.addWidget(src_lbl)

    def _make_header(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(46)
        w.setStyleSheet(f"background:{P['panel_bg']}; border-radius:6px;")
        hl = QHBoxLayout(w)
        hl.setContentsMargins(16, 0, 16, 0)
        title = QLabel("🔥  WILDFIRE DETECTION PIPELINE")
        title.setStyleSheet(
            f"color:{P['accent_fire']}; font-size:15px; font-weight:700;"
            " letter-spacing:1.5px;")
        hl.addWidget(title)
        hl.addStretch()
        sub = QLabel("Open a parent folder (smoke/ + clean/) · click a thumbnail or ◀ ▶ to switch")
        sub.setStyleSheet(
            f"color:{P['text_muted']}; font-size:11px; letter-spacing:0.5px;")
        hl.addWidget(sub)
        return w

    def _build_controls(self, layout: QVBoxLayout):
        def grp(title, sliders):
            g  = QGroupBox(title)
            gl = QVBoxLayout(g)
            gl.setSpacing(3)
            for key, lbl, lo, hi, dflt, dec in sliders:
                row = SliderRow(lbl, lo, hi, dflt, dec)
                row.value_changed.connect(lambda v, k=key: self.params.update({k: v}))
                gl.addWidget(row)
            layout.addWidget(g)

        # Stage 1 is built manually so the Auto checkbox can grey out the
        # manual sliders. Auto mode picks alpha from measured object speed
        # and threshold from the frame's noise floor (see wildfire_core.iir).
        g1 = QGroupBox("Stage 1 — IIR Background")
        g1l = QVBoxLayout(g1)
        g1l.setSpacing(3)

        auto_cb = QCheckBox("Auto  (adapt to object speed)")
        auto_cb.setCursor(Qt.PointingHandCursor)
        auto_cb.setStyleSheet(
            f"QCheckBox{{color:{P['text_primary']}; font-size:11px;}}"
            f"QCheckBox::indicator{{width:13px; height:13px;}}")
        g1l.addWidget(auto_cb)

        iir_rows = []
        for key, lbl, lo, hi, dflt, dec in [
                ("iir_alpha",     "Alpha",     0.01, 0.3, 0.05, 2),
                ("iir_threshold", "Threshold", 1,    63, 10,   0)]:
            row = SliderRow(lbl, lo, hi, dflt, dec)
            row.value_changed.connect(lambda v, k=key: self.params.update({k: v}))
            g1l.addWidget(row)
            iir_rows.append(row)

        # c-knob for auto mode: effective threshold = c * (auto threshold).
        # c > 1 -> stricter mask (collects less), c < 1 -> looser.
        gain_row = SliderRow("Auto Gain (c)", 0.5, 3.0, 1.0, 2)
        gain_row.value_changed.connect(
            lambda v: self.params.update({"iir_auto_gain": v}))
        gain_row.setEnabled(False)          # only meaningful in auto mode
        g1l.addWidget(gain_row)

        def _auto_toggled(state):
            on = bool(state)
            self.params.update({"iir_auto": 1 if on else 0})
            for r in iir_rows:
                r.setEnabled(not on)   # sliders inactive while auto drives
            gain_row.setEnabled(on)
        auto_cb.stateChanged.connect(_auto_toggled)

        # Reproduce the TRAINING conditions on startup: if a trained model
        # exists, apply the Stage-1 settings it was trained with, so live
        # masks match what the model saw. (Other sliders already share the
        # trainer's defaults; override manually if you trained with --params.)
        try:
            import json as _json
            _mwp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "motion_weights.json")
            with open(_mwp, "r") as _f:
                _tp = _json.load(_f).get("trained_with_params", {})
            if _tp.get("iir_auto"):
                auto_cb.setChecked(True)     # triggers _auto_toggled
                _g = float(_tp.get("iir_auto_gain", 1.0))
                gain_row._s.setValue(int(_g * gain_row._scale))
                print(f"[motion] Stage-1 synced to training: auto=1, c={_g}")
        except Exception:
            pass
        layout.addWidget(g1)
        grp("Stage 2 — HSV  Smoke", [
            ("smoke_h_min", "Hue Min", 0,   179, 0,   0),
            ("smoke_h_max", "Hue Max", 0,   179, 179, 0),
            ("smoke_s_min", "Sat Min", 0,   255, 0,   0),
            ("smoke_s_max", "Sat Max", 0,   255, 153, 0),
            ("smoke_v_min", "Val Min", 0,   255, 30,  0),
            ("smoke_v_max", "Val Max", 0,   255, 255, 0),
        ])
        grp("Stage 3 — Gabor  (Smoke Texture)", [
            ("gabor_ksize",      "Kernel Size",   5,   31,  21,   0),
            ("gabor_sigma",      "Sigma",         0.5, 10,  2.12,  1),
            ("gabor_lambda",     "Lambda",        2,   30,  3.82,  1),
            ("gabor_gamma",      "Gamma",         0.1, 2.0, 1.9,  1),
            ("gabor_std_thresh", "Std Threshold", 1,   80,  77.28, 1),
        ])

    # ── Thread ────────────────────────────────────────────────────────────────
    def _start_thread(self):
        self._thread = ProcessingThread(self.source, self.params,
                                        ground_truth=self.ground_truth)
        self._thread.frames_ready.connect(self._update_panels)
        self._thread.error_signal.connect(self._on_error)
        self._thread.start()

    def _update_panels(self, p1, p2, p3, p4, p5):
        for panel, pix in zip(self._panels, [p1, p2, p3, p4, p5]):
            panel.update_frame(pix)

    def _on_error(self, msg):
        self._status_lbl.setText(f"⚠  {msg}")
        self._status_lbl.setStyleSheet(
            "color:#f85149; font-weight:600; font-size:11px;")

    # ── Folder playlist + live source switching ───────────────────────────────
    VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
    THUMB_W, THUMB_H = 150, 90          # playlist thumbnail size (px)

    def _build_playlist(self, layout: QVBoxLayout):
        """Folder picker + the clickable video THUMBNAIL grid (top of controls)."""
        box = QGroupBox("VIDEO SOURCE")
        bl = QVBoxLayout(box)
        bl.setSpacing(5)

        btn = QPushButton("📂  Open Folder…")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton{{background:{P['border']}; color:{P['text_primary']};"
            f" border:none; border-radius:4px; padding:6px; font-weight:600;}}"
            f"QPushButton:hover{{background:{P['accent_fire']}; color:#000;}}")
        btn.clicked.connect(self._open_folder)
        bl.addWidget(btn)

        self._folder_lbl = QLabel("Pick a parent folder with smoke/ and clean/")
        self._folder_lbl.setWordWrap(True)
        self._folder_lbl.setStyleSheet(
            f"color:{P['text_muted']}; font-size:10px;")
        bl.addWidget(self._folder_lbl)

        # Icon-mode list: one thumbnail per video, label tag shown under it.
        self._playlist_widget = QListWidget()
        self._playlist_widget.setViewMode(QListWidget.IconMode)
        self._playlist_widget.setIconSize(QSize(self.THUMB_W, self.THUMB_H))
        self._playlist_widget.setGridSize(QSize(self.THUMB_W + 14, self.THUMB_H + 30))
        self._playlist_widget.setResizeMode(QListWidget.Adjust)
        self._playlist_widget.setMovement(QListWidget.Static)
        self._playlist_widget.setWordWrap(True)
        self._playlist_widget.setSpacing(4)
        self._playlist_widget.setStyleSheet(
            f"QListWidget{{background:{P['bg']}; color:{P['text_primary']};"
            f" border:1px solid {P['border']}; border-radius:4px; font-size:9px;}}"
            f"QListWidget::item{{padding:2px; border-radius:3px;}}"
            f"QListWidget::item:selected{{background:{P['accent_fire']}; color:#000;}}")
        self._playlist_widget.setMinimumHeight(260)
        self._playlist_widget.itemClicked.connect(
            lambda it: self._play_index(self._playlist_widget.row(it)))
        bl.addWidget(self._playlist_widget)

        hint = QLabel("◀ / ▶  prev / next video")
        hint.setStyleSheet(f"color:{P['text_muted']}; font-size:10px;")
        bl.addWidget(hint)

        layout.addWidget(box)
        if self._playlist:
            n_smoke = sum(1 for _, l in self._playlist if l == "SMOKE")
            n_clean = sum(1 for _, l in self._playlist if l == "CLEAN")
            self._folder_lbl.setText(
                f"{os.path.basename(os.path.normpath(self.folder))}  ·  "
                f"{n_smoke} smoke / {n_clean} clean")
            self._refresh_playlist_widget()

    CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")

    def _build_capture(self, layout: QVBoxLayout):
        box = QGroupBox("CAPTURE")
        bl = QHBoxLayout(box)
        bl.setSpacing(5)
        style = (
            f"QPushButton{{background:{P['border']}; color:{P['text_primary']};"
            f" border:none; border-radius:4px; padding:6px; font-weight:600;}}"
            f"QPushButton:hover{{background:{P['accent_fire']}; color:#000;}}"
            f"QPushButton:checked{{background:#f85149; color:#000;}}")
        shot = QPushButton("📷  Screenshot")
        shot.clicked.connect(self._take_screenshot)
        self._rec_btn = QPushButton("⏺  Record")
        self._rec_btn.setCheckable(True)
        self._rec_btn.toggled.connect(self._toggle_record)
        for b in (shot, self._rec_btn):
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(style)
            bl.addWidget(b)
        layout.addWidget(box)

    def _take_screenshot(self):
        os.makedirs(self.CAPTURE_DIR, exist_ok=True)
        path = os.path.join(self.CAPTURE_DIR,
                            time.strftime("screenshot_%Y%m%d_%H%M%S.png"))
        self.grab().save(path)
        self.statusBar().showMessage(f"Screenshot saved: {path}", 8000)

    def _toggle_record(self, on):
        if on:
            self._rec_dir = os.path.join(self.CAPTURE_DIR,
                                         time.strftime("rec_%Y%m%d_%H%M%S"))
            os.makedirs(self._rec_dir, exist_ok=True)
            self._thread.set_recording(self._rec_dir)
            self._rec_btn.setText("⏹  Stop")
            self.statusBar().showMessage(f"Recording to {self._rec_dir}")
        else:
            self._thread.set_recording(None)
            self._rec_btn.setText("⏺  Record")
            self.statusBar().showMessage(f"Recording saved: {self._rec_dir}", 8000)

    def _open_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select the parent folder (containing smoke/ and clean/)", "")
        if path:
            self._load_folder(path, autoplay=True)

    def _scan_videos(self, path: str) -> list[tuple[str, str]]:
        """Return [(video_path, LABEL), ...]. Looks for 'smoke'/'clean'
        subfolders of `path` (case-insensitive) and labels each video by the
        subfolder it lives in. If neither subfolder exists, falls back to
        scanning `path` directly and labels by the chosen folder's own name."""
        def vids_in(d):
            return sorted(
                os.path.join(d, f) for f in os.listdir(d)
                if f.lower().endswith(self.VIDEO_EXTS))

        out: list[tuple[str, str]] = []
        subs = {e.lower(): os.path.join(path, e)
                for e in os.listdir(path)
                if os.path.isdir(os.path.join(path, e))}
        found_sub = False
        for name, label in (("smoke", "SMOKE"), ("clean", "CLEAN")):
            if name in subs:
                found_sub = True
                for v in vids_in(subs[name]):
                    out.append((v, label))

        if not found_sub:
            # leaf folder: label by its own name if it is smoke/clean
            base = os.path.basename(os.path.normpath(path)).lower()
            label = "SMOKE" if base == "smoke" else "CLEAN" if base == "clean" else ""
            out = [(v, label) for v in vids_in(path)]
        return out

    def _load_folder(self, path: str, autoplay: bool):
        """Scan `path` (parent of smoke/ & clean/) and (optionally) start the
        first clip."""
        if not os.path.isdir(path):
            if hasattr(self, "_status_lbl"):
                self._on_error(f"Not a folder: {path}")
            return
        playlist = self._scan_videos(path)
        if not playlist:
            if hasattr(self, "_status_lbl"):
                self._on_error(f"No videos under {os.path.basename(path)}")
            return

        self.folder = path
        self._playlist = playlist
        self._play_idx = -1

        # UI may not exist yet when called from __init__ (autoplay=False).
        if hasattr(self, "_folder_lbl"):
            n_smoke = sum(1 for _, l in playlist if l == "SMOKE")
            n_clean = sum(1 for _, l in playlist if l == "CLEAN")
            self._folder_lbl.setText(
                f"{os.path.basename(os.path.normpath(path))}  ·  "
                f"{n_smoke} smoke / {n_clean} clean")
            self._refresh_playlist_widget()
        if autoplay:
            self._play_index(0)

    def _make_thumb(self, video_path: str) -> QIcon:
        """Grab a representative frame from the video and build a QIcon. Cheap:
        one seek + one decode per video, done once at load time."""
        cap = cv2.VideoCapture(video_path)
        frame = None
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 2:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 3)   # skip black intros
            ok, frame = cap.read()
        cap.release()

        if frame is None:
            ph = np.full((self.THUMB_H, self.THUMB_W, 3), 30, np.uint8)
            cv2.putText(ph, "no preview", (8, self.THUMB_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
            frame = ph
        thumb = cv2.resize(frame, (self.THUMB_W, self.THUMB_H),
                           interpolation=cv2.INTER_AREA)
        return QIcon(_to_pixmap(thumb))

    def _refresh_playlist_widget(self):
        self._playlist_widget.clear()
        for path, label in self._playlist:
            name = os.path.basename(path)
            short = name if len(name) <= 18 else name[:15] + "…"
            item = QListWidgetItem(self._make_thumb(path), f"[{label}] {short}")
            item.setToolTip(f"{label} — {name}")
            # tint the caption by class for at-a-glance scanning
            item.setForeground(QColor("#7ee787") if label == "CLEAN"
                               else QColor("#ff7b72"))
            self._playlist_widget.addItem(item)
        if 0 <= self._play_idx < len(self._playlist):
            self._playlist_widget.setCurrentRow(self._play_idx)

    def _play_index(self, idx: int):
        """Switch live playback to playlist entry `idx`."""
        if not self._playlist:
            return
        idx = max(0, min(idx, len(self._playlist) - 1))
        self._play_idx = idx
        src, label = self._playlist[idx]
        self.source = src
        self.ground_truth = label
        self._thread.request_source(src, ground_truth=label)

        if hasattr(self, "_playlist_widget"):
            self._playlist_widget.setCurrentRow(idx)
        name = os.path.basename(src)
        tag = label or "—"
        self._status_lbl.setText(f"● [{tag}]  {idx+1}/{len(self._playlist)}  {name}")
        self._status_lbl.setStyleSheet(
            f"color:{P['accent_go']}; font-weight:600; font-size:11px;")

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Right:          # next video
            if self._playlist:
                self._play_index(self._play_idx + 1)
            return
        if ev.key() == Qt.Key_Left:           # previous video
            if self._playlist:
                self._play_index(self._play_idx - 1)
            return
        if ev.text().lower() == "o":          # quick "open folder"
            self._open_folder()
            return
        super().keyPressEvent(ev)

    def closeEvent(self, ev):
        self._thread.stop()
        self._thread.wait(2000)
        super().closeEvent(ev)

# ==============================================================================
#  ENTRY POINT
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="Wildfire Detection GUI")
    ap.add_argument("--video", "-v", default=None,
                    help="Path to a single video file. Omit to use webcam (device 0).")
    ap.add_argument("--folder", "-f", default=None,
                    help="Path to a PARENT folder containing 'smoke/' and 'clean/' "
                         "subfolders. Videos from both are listed and tagged by "
                         "their subfolder. Overrides --video.")
    args = ap.parse_args()

    source = args.video if args.video else 0
    if isinstance(source, str) and not os.path.isfile(source):
        print(f"[WARN] File not found: {source!r} — falling back to webcam.")
        source = 0

    folder = args.folder
    if folder and not os.path.isdir(folder):
        print(f"[WARN] Folder not found: {folder!r} — ignoring.")
        folder = None

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    win = WildfireApp(source, folder=folder)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()