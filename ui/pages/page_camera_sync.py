"""
Cameras Synchronisation page (Image Processing section).

GUI around core/camera_sync.py -- the N-camera (1 to 8) pipeline that used to be the stand-alone 3-camera script
``images_processing2.py``:

    1. Prepare      master dark / flat (automatic rejection of bad frames), gain maps, calibrated previews
    2. Alignment    rotation (a line drawn on each camera) then translation (a point), composed in ONE warp
    3. Crop         one rectangle per camera, the intersection is applied to all of them
    4. Process      calibration + alignment + crop (streamed), intensity normalisation, gradient
                    homogenisation, optional denoising, 16-bit exports, timestamps.csv, FFV1 video

The OpenCV windows of the script (cv2.imshow / selectROI) are replaced by modal matplotlib dialogs
(``ImagePickDialog``) because the app ships opencv-python-headless.
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np
from PySide6.QtCore import Qt, QThread, QObject, Signal, QUrl, QTimer
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog, QDialog, QProgressBar, QPlainTextEdit,
    QGroupBox, QMessageBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from core import camera_sync as cs
from core.parallel import OperationCancelled
from ui.widgets import Card, make_scroll_page
from ui.plot_canvas import PlotCanvas

CAM_LABELS = [f"Camera {c + 1}" for c in range(cs.MAX_CAMERAS)]


# ----------------------------------------------------------------------------------------------
# interactive picking dialog (replaces cv2.imshow + mouse callbacks / cv2.selectROI)
# ----------------------------------------------------------------------------------------------
class ImagePickDialog(QDialog):
    """Shows an image and collects clicks. mode: "line" (2 clicks), "point" (1 click) or "rect" (drag).

    Zoom/pan with the toolbar (deselect the tool before clicking) or the mouse wheel. `points` holds the
    picked image coordinates (sub-pixel), `rect` the (x, y, w, h) rectangle."""

    def __init__(self, title: str, image: np.ndarray, mode: str, instruction: str, initial_rect=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 760)
        self.mode = mode
        self.points: list = []
        self.rect = None
        self._artists: list = []
        layout = QVBoxLayout(self)
        info = QLabel(instruction)
        info.setWordWrap(True)
        layout.addWidget(info)
        self.fig = Figure(figsize=(9, 6.5))
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(cs.display_uint8(image), cmap="gray", interpolation="nearest", vmin=0, vmax=255)
        self.ax.set_xlabel("x (px)")
        self.ax.set_ylabel("y (px)")
        self.fig.tight_layout()
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.status = QLabel("")
        layout.addWidget(self.status)
        row = QHBoxLayout()
        self.reset_btn = QPushButton("Reset")
        self.ok_btn = QPushButton("Accept")
        self.ok_btn.setEnabled(False)
        cancel = QPushButton("Cancel")
        self.reset_btn.clicked.connect(self._reset)
        self.ok_btn.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        row.addStretch(1)
        for b in (self.reset_btn, cancel, self.ok_btn):
            row.addWidget(b)
        layout.addLayout(row)

        self._selector = None
        if mode == "rect":
            self._selector = RectangleSelector(self.ax, self._on_rect, useblit=True, button=[1], minspanx=3, minspany=3,
                                               spancoords="pixels", interactive=True)
            if initial_rect is not None:
                x, y, w, h = initial_rect
                self._selector.extents = (x - 0.5, x + w - 0.5, y - 0.5, y + h - 0.5)
                self.rect = tuple(int(v) for v in initial_rect)
                self.ok_btn.setEnabled(True)
                self._update_status()
        else:
            self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._update_status()

    def _toolbar_busy(self) -> bool:
        return bool(getattr(self.toolbar, "mode", ""))

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.button != 1 or self._toolbar_busy():
            return
        p = (float(event.xdata), float(event.ydata))
        if self.mode == "point":
            self.points = [p]
        elif len(self.points) < 2:
            self.points.append(p)
        else:
            return
        self._redraw()

    def _on_rect(self, eclick, erelease):
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        x0, y0 = int(round(x0 + 0.5)), int(round(y0 + 0.5))
        x1, y1 = int(round(x1 + 0.5)), int(round(y1 + 0.5))
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            self.rect = (x0, y0, x1 - x0, y1 - y0)
        self.ok_btn.setEnabled(self.rect is not None)
        self._update_status()

    def _on_scroll(self, event):
        if event.inaxes is not self.ax:
            return
        f = 0.8 if event.button == "up" else 1.25
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.set_xlim(event.xdata - (event.xdata - x0) * f, event.xdata + (x1 - event.xdata) * f)
        self.ax.set_ylim(event.ydata - (event.ydata - y0) * f, event.ydata + (y1 - event.ydata) * f)
        self.canvas.draw_idle()

    def _redraw(self):
        for a in self._artists:
            try:
                a.remove()
            except Exception:
                pass
        self._artists = []
        for p in self.points:
            self._artists += self.ax.plot([p[0]], [p[1]], "o", color="#00ff66", ms=6)
        if len(self.points) == 2:
            self._artists += self.ax.plot([self.points[0][0], self.points[1][0]], [self.points[0][1], self.points[1][1]],
                                          "-", color="#00ff66", lw=1.5)
        need = 2 if self.mode == "line" else 1
        self.ok_btn.setEnabled(len(self.points) == need)
        self.canvas.draw_idle()
        self._update_status()

    def _reset(self):
        self.points = []
        self.rect = None
        if self._selector is not None:
            self._selector.set_visible(False)
            self._selector.update()
        self.ok_btn.setEnabled(False)
        self._redraw()

    def _update_status(self):
        if self.mode == "rect":
            self.status.setText("Rectangle: x=%d y=%d w=%d h=%d" % self.rect if self.rect else "Drag a rectangle.")
        elif self.points:
            self.status.setText("  ".join("(%.1f, %.1f)" % p for p in self.points))
        else:
            self.status.setText("No point yet.")


# ----------------------------------------------------------------------------------------------
# workers
# ----------------------------------------------------------------------------------------------
class _PrepareWorker(QObject):
    progress = Signal(str, float)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, cfg, cancel_event):
        super().__init__()
        self.cfg, self.cancel_event = cfg, cancel_event

    def run(self):
        try:
            prep = cs.prepare_calibration(self.cfg, progress=lambda m, f: self.progress.emit(m, -1.0 if f is None else float(f)),
                                          cancel_check=self.cancel_event.is_set)
            self.finished.emit(prep)
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - reported in the log, never a modal popup from a worker
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _ProcessWorker(QObject):
    progress = Signal(str, float)
    finished = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, prep, alignment, crop_bbox, rois, crop_reason, cancel_event):
        super().__init__()
        self.args = (prep, alignment, crop_bbox, rois)
        self.crop_reason, self.cancel_event = crop_reason, cancel_event

    def run(self):
        try:
            rep = cs.process_sequences(*self.args, progress=lambda m, f: self.progress.emit(m, -1.0 if f is None else float(f)),
                                       cancel_check=self.cancel_event.is_set, crop_reason=self.crop_reason)
            self.finished.emit(rep)
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _StretchWorker(QObject):
    progress = Signal(str, float)
    finished = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, cfg, cancel_event):
        super().__init__()
        self.cfg, self.cancel_event = cfg, cancel_event

    def run(self):
        try:
            rep = cs.restretch_existing(self.cfg, progress=lambda m, f: self.progress.emit(m, -1.0 if f is None else float(f)),
                                        cancel_check=self.cancel_event.is_set)
            self.finished.emit(rep)
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------------------------------
# small widgets
# ----------------------------------------------------------------------------------------------
class _FolderEdit(QWidget):
    def __init__(self, placeholder: str = "", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        btn = QPushButton("…")
        btn.setFixedWidth(30)
        btn.clicked.connect(self._browse)
        lay.addWidget(self.edit, 1)
        lay.addWidget(btn)

    def _browse(self):
        p = QFileDialog.getExistingDirectory(self, "Select folder", self.edit.text() or "")
        if p:
            self.edit.setText(p)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, t: str):
        self.edit.setText(t or "")


def _dspin(lo, hi, val, step=0.1, dec=2, suffix="", special: str | None = None) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    if special:
        w.setSpecialValueText(special)
    w.setKeyboardTracking(False)
    return w


def _ispin(lo, hi, val, suffix="", special: str | None = None) -> QSpinBox:
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    if special:
        w.setSpecialValueText(special)
    return w


def _combo(items) -> QComboBox:
    c = QComboBox()
    for key, label in items:
        c.addItem(label, key)
    return c


def _row(*items) -> QHBoxLayout:
    lay = QHBoxLayout()
    for it in items:
        if isinstance(it, str):
            lay.addWidget(QLabel(it))
        elif it is None:
            lay.addStretch(1)
        else:
            lay.addWidget(it)
    lay.addStretch(1)
    return lay


# ----------------------------------------------------------------------------------------------
# page
# ----------------------------------------------------------------------------------------------
class CameraSyncPage(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._prep = None
        self._alignment = None
        self._aligned_previews = None
        self._crop_bbox = None
        self._crop_rois = None
        self._crop_reason = None
        self._thread = None
        self._worker = None
        self._cancel = threading.Event()
        self._chain = False
        self._bindings: dict = {}      # config field -> (widget, kind)
        self._cam_rows: list = []      # per camera: the widgets that are hidden when the camera is not used

        scroll, layout = make_scroll_page(
            "Cameras Synchronisation",
            "Calibrates (dark / flat), aligns (rotation + translation), crops and time-stamps the images of N cameras "
            f"(1 to {cs.MAX_CAMERAS}; with a single camera there is nothing to align), homogenises the illumination "
            "gradients and exports a time-ordered 16-bit sequence and video.",
        )
        self._build_folders_card(layout)
        self._build_timing_card(layout)
        self._build_calibration_card(layout)
        self._build_alignment_card(layout)
        self._build_intensity_card(layout)
        self._build_homogenisation_card(layout)
        self._build_denoise_card(layout)
        self._build_stretch_card(layout)
        self._build_export_card(layout)
        self._build_run_card(layout)
        layout.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self._on_n_cameras_changed()
        self._update_enabled_states()

    # ------------------------------------------------------------------ binding helpers
    def _bind(self, name: str, widget, kind: str = "auto"):
        self._bindings[name] = (widget, kind)
        return widget

    @staticmethod
    def _get(widget, kind):
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            return widget.currentData()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            v = widget.value()
            return (None if v == widget.minimum() and widget.specialValueText() else v) if kind == "optional" else v
        if isinstance(widget, (QLineEdit, _FolderEdit)):
            return widget.text().strip()
        raise TypeError(type(widget))

    @staticmethod
    def _set(widget, value, kind):
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QComboBox):
            i = widget.findData(value)
            if i >= 0:
                widget.setCurrentIndex(i)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.setValue(widget.minimum() if value is None else value)
        elif isinstance(widget, (QLineEdit, _FolderEdit)):
            widget.setText("" if value is None else str(value))

    def collect_config(self) -> cs.CameraSyncConfig:
        cfg = cs.CameraSyncConfig()
        for name, (widget, kind) in self._bindings.items():
            setattr(cfg, name, self._get(widget, kind))
        n = int(cfg.n_cameras)
        cfg.dark_folders = [w.text() for w in self._dark_edits[:n]]
        cfg.flat_folders = [w.text() for w in self._flat_edits[:n]]
        cfg.light_folders = [w.text() or None for w in self._light_edits[:n]]
        cfg.input_folders = [w.text() for w in self._input_edits[:n]]
        cfg.camera_offsets_ns = [s.value() for s in self._offset_spins[:n]]
        cfg.flat_pixel_gain_limits = (self._pix_lo.value(), self._pix_hi.value())
        cfg.flat_total_gain_limits = (self._tot_lo.value(), self._tot_hi.value())
        cfg.timestamp_ns = int(cfg.timestamp_ns)
        cfg.background_kernel = int(cfg.background_kernel) if cfg.background_kernel else None
        return cfg

    def apply_config(self, cfg: cs.CameraSyncConfig):
        for name, (widget, kind) in self._bindings.items():
            self._set(widget, getattr(cfg, name), kind)
        self._on_n_cameras_changed()
        for edits, vals in ((self._dark_edits, cfg.dark_folders), (self._flat_edits, cfg.flat_folders),
                            (self._light_edits, cfg.light_folders), (self._input_edits, cfg.input_folders)):
            for e, v in zip(edits, vals):
                e.setText(v or "")
        for s, v in zip(self._offset_spins, cfg.camera_offsets_ns):
            s.setValue(v)
        self._pix_lo.setValue(cfg.flat_pixel_gain_limits[0])
        self._pix_hi.setValue(cfg.flat_pixel_gain_limits[1])
        self._tot_lo.setValue(cfg.flat_total_gain_limits[0])
        self._tot_hi.setValue(cfg.flat_total_gain_limits[1])
        self._update_enabled_states()

    # ------------------------------------------------------------------ cards
    def _build_folders_card(self, layout):
        card = Card("1. Cameras and folders",
                    "One row per camera: its dark frames, flat-field frames and image sequence (the dark and flat folders must hold "
                    "the SAME camera as the sequence on the same row). Images: BMP / TIFF / PNG, sorted naturally (img2 before img10).")
        self.n_cameras_spin = self._bind("n_cameras", _ispin(1, cs.MAX_CAMERAS, 3))
        self.n_cameras_spin.setToolTip("1 = a single camera: calibration, homogenisation, denoising and exports only (no alignment).")
        self.n_cameras_spin.valueChanged.connect(self._on_n_cameras_changed)
        card.add_layout(_row("Number of cameras", self.n_cameras_spin))
        n_max = cs.MAX_CAMERAS
        self._dark_edits = [_FolderEdit("dark frames") for _ in range(n_max)]
        self._flat_edits = [_FolderEdit("flat-field frames") for _ in range(n_max)]
        self._light_edits = [_FolderEdit("static light (optional)") for _ in range(n_max)]
        self._input_edits = [_FolderEdit("image sequence") for _ in range(n_max)]
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        for col, label in enumerate(("Dark frames", "Flat-field frames", "Static light (optional)", "Image sequence"), start=1):
            grid.addWidget(QLabel(f"<b>{label}</b>"), 0, col)
            grid.setColumnStretch(col, 1)
        for c in range(n_max):
            lab = QLabel(f"<b>{CAM_LABELS[c]}</b>")
            grid.addWidget(lab, c + 1, 0)
            widgets = [lab]
            for col, edits in enumerate((self._dark_edits, self._flat_edits, self._light_edits, self._input_edits), start=1):
                grid.addWidget(edits[c], c + 1, col)
                widgets.append(edits[c])
            self._cam_rows.append(widgets)
        card.add_layout(grid)
        self.master_output_edit = self._bind("master_output", _FolderEdit("output folder"))
        card.add_layout(_row("Output folder", self.master_output_edit))
        self.bit_depth_combo = self._bind("input_bit_depth", _combo([(16, "16-bit"), (8, "8-bit")]))
        self.skip_start_spin = self._bind("skip_start", _ispin(0, 100000, 10))
        self.skip_end_spin = self._bind("skip_end", _ispin(0, 100000, 80))
        self.max_calib_spin = self._bind("max_calibration_frames", _ispin(1, 10000, 100))
        card.add_layout(_row("Bit depth of the raw images", self.bit_depth_combo, "Skip first frames", self.skip_start_spin,
                             "Skip last frames", self.skip_end_spin, "Max dark/flat frames used", self.max_calib_spin))
        layout.addWidget(card)

    def _build_timing_card(self, layout):
        card = Card("2. Time stamps",
                    "Frame time = t0 + original frame index x step + camera offset. The output sequence is ordered by this time "
                    "(camera 1, 2, 3, ..., 1, 2, 3, ... when the offsets are smaller than the step). A timestamps.csv is always written.")
        self.use_ts_check = self._bind("use_timestamps", QCheckBox("Use time stamps (order frames by time)"))
        self.use_ts_check.setChecked(True)
        self.ts0_spin = self._bind("timestamp_ns", _dspin(-1e15, 1e15, 0, 1, 0, " ns"))
        self.ts_step_spin = self._bind("timestamp_step", _dspin(0, 1e12, 530, 1, 3, " ns"))
        default_offsets = (0, 150, 330, 400, 480, 560, 640, 720)
        self._offset_spins = [_dspin(-1e12, 1e12, default_offsets[c], 1, 3, " ns") for c in range(cs.MAX_CAMERAS)]
        self.burn_check = self._bind("burn_timestamp", QCheckBox("Burn the time into the exported pixels (bottom right, white on black)"))
        self.burn_check.setToolTip("Off by default: text drawn in the image modifies what DIC later correlates. "
                                   "The time of every frame is in timestamps.csv.")
        card.add_layout(_row(self.use_ts_check, self.burn_check))
        card.add_layout(_row("t0", self.ts0_spin, "Step", self.ts_step_spin))
        self._offset_labels = [QLabel(f"cam {c + 1}") for c in range(cs.MAX_CAMERAS)]
        self._offset_row = _row("Offsets:", *[w for c in range(cs.MAX_CAMERAS) for w in (self._offset_labels[c], self._offset_spins[c])])
        card.add_layout(self._offset_row)
        for c in range(cs.MAX_CAMERAS):
            self._cam_rows[c].extend([self._offset_labels[c], self._offset_spins[c]])
        layout.addWidget(card)

    def _build_calibration_card(self, layout):
        card = Card("3. Calibration (dark / flat-field)",
                    "Bad dark and flat frames are rejected automatically (saturated, under-exposed, outlier mean). "
                    "Master = median of the accepted frames. The gain map is computed once per camera.")
        self.flat_method_combo = self._bind("flat_method", _combo([
            ("robust", "Robust (repairs hot/dead pixels, keeps the vignetting, soft limits)"),
            ("legacy", "Legacy (small gaussian + hard clamp, as in the script)")]))
        self.static_light_check = self._bind("use_static_light", QCheckBox("Subtract the static light pattern (needs the 'static light' folders)"))
        card.add_layout(_row("Flat-field method", self.flat_method_combo))
        card.add_layout(_row(self.static_light_check))
        self.dark_scaling_combo = self._bind("dark_scaling", _combo([
            ("per_frame", "Fit the dark-current amplitude of every frame (recommended)"), ("none", "None: subtract the master dark as it is")]))
        self.dark_scaling_combo.setToolTip("Dark current grows with exposure time and sensor temperature. If the darks were recorded in another state "
                                           "than the sequence (e.g. later, when the sensor is hotter) the hot pixels are over-subtracted and come out "
                                           "black. The factor is the least-squares slope of the high-pass of the frame on that of the master dark "
                                           "(the scene is uncorrelated with the dark pattern). It is reported per camera, with a warning when it is "
                                           "not close to 1.")
        card.add_layout(_row("Dark scaling", self.dark_scaling_combo))
        self.pedestal_spin = self._bind("calibration_pedestal", _dspin(0, 65535, 0, 25, 1, " ADU", special="off"))
        self.pedestal_spin.setToolTip("Black pixels = pixels darker than the master dark (dark specimen, dark frames taken in other conditions, "
                                      "noise amplified at the edges). Without a pedestal their value is replaced by the noise floor. "
                                      "A pedestal of about 3-5 x the noise (in ADU after calibration) keeps the noise symmetric. "
                                      "After 'Prepare/Process' see alignment_check/negative_pixels_camN.png and the warnings of the report.")
        card.add_layout(_row("Calibration pedestal (keeps the noise of dark pixels instead of clipping it at 0)", self.pedestal_spin))
        self.lowfreq_spin = self._bind("flat_lowfreq_sigma_frac", _dspin(0.005, 0.5, 0.05, 0.005, 3))
        self._pix_lo, self._pix_hi = _dspin(0.1, 1.0, 0.7, 0.05, 2), _dspin(1.0, 5.0, 1.4, 0.05, 2)
        self.pix_smooth_spin = self._bind("flat_pixel_smooth_sigma_px", _dspin(0, 20, 1.0, 0.25, 2, " px"))
        self._tot_lo, self._tot_hi = _dspin(0.01, 1.0, 0.6, 0.05, 2), _dspin(1.0, 50.0, 1.6, 0.1, 2)
        self.robust_box = QWidget()
        rl = QVBoxLayout(self.robust_box)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addLayout(_row("Vignetting scale (sigma / min image side)", self.lowfreq_spin, "Dust/PRNU limits", self._pix_lo, self._pix_hi,
                          "Flat noise smoothing", self.pix_smooth_spin))
        rl.addLayout(_row("Total gain limits", self._tot_lo, self._tot_hi))
        self._tot_lo.setToolTip("The flat-field response is limited to this range: dividing by a response of 0.3 amplifies the noise "
                                "(and the noise of dark pixels) by 3.3 at that place. Widen only if the vignetting is real.")
        card.add(self.robust_box)
        self.gain_kernel_spin = self._bind("gain_smooth_kernel_frac", _dspin(0.001, 0.5, 0.02, 0.005, 3))
        self.min_gain_spin = self._bind("min_gain", _dspin(0.01, 1.0, 0.6, 0.05, 2))
        self.max_gain_spin = self._bind("max_gain", _dspin(1.0, 50.0, 1.6, 0.05, 2))
        self.legacy_box = QWidget()
        ll = QVBoxLayout(self.legacy_box)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addLayout(_row("Gain smoothing kernel (fraction of min side)", self.gain_kernel_spin, "Min gain", self.min_gain_spin,
                          "Max gain", self.max_gain_spin))
        card.add(self.legacy_box)
        self.flat_method_combo.currentIndexChanged.connect(self._update_enabled_states)

        adv = QGroupBox("Rejection thresholds (advanced)")
        adv.setCheckable(True)
        adv.setChecked(False)
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setContentsMargins(0, 0, 0, 0)
        il.addLayout(_row("Dark: max saturated fraction", self._bind("dark_saturation_max_frac", _dspin(0, 1, 0.01, 0.005, 4)),
                          "outlier = median + k x std, k", self._bind("dark_outlier_std_factor", _dspin(0.5, 20, 3.0, 0.5, 2)),
                          "min mean fraction", self._bind("dark_min_mean_frac", _dspin(0, 1, 0.0, 0.01, 3))))
        il.addLayout(_row("Flat: max saturated fraction", self._bind("flat_saturation_max_frac", _dspin(0, 1, 0.01, 0.005, 4)),
                          "max under-exposed fraction", self._bind("flat_underexposed_max_frac", _dspin(0, 1, 0.5, 0.05, 3)),
                          "min mean fraction", self._bind("flat_min_mean_frac", _dspin(0, 1, 0.02, 0.005, 4))))
        v = QVBoxLayout(adv)
        v.addWidget(inner)
        inner.setVisible(False)
        adv.toggled.connect(inner.setVisible)
        card.add(adv)
        layout.addWidget(card)

    def _build_alignment_card(self, layout):
        card = Card("4. Alignment and crop",
                    "Levelling (optional): a baseline drawn on the reference camera is made horizontal for all cameras. "
                    "Rotation: you draw a line along the same feature on each camera (click the two ends in the same order). "
                    "Translation: you click the same point on each camera. Both are composed into one resampling.")
        self.ref_cam_combo = self._bind("reference_camera_index", _combo([(c, CAM_LABELS[c]) for c in range(3)]))
        self.level_check = self._bind("align_level_reference", QCheckBox("Level the reference camera first (draw a baseline that is rotated to horizontal)"))
        self.level_check.setChecked(True)
        self.align_first_check = self._bind("align_use_first_image", QCheckBox("Use the first image (otherwise the median of calibrated frames)"))
        self.align_first_check.setChecked(True)
        self.interp_combo = self._bind("align_interpolation", _combo([("cubic", "Cubic (recommended)"), ("lanczos", "Lanczos"), ("linear", "Linear")]))
        self.autocrop_check = self._bind("autocrop", QCheckBox("Interactive crop (one rectangle per camera, intersection applied to all)"))
        self.autocrop_check.setChecked(True)
        self.margin_spin = self._bind("autocrop_margin_px", _ispin(0, 10000, 0, " px"))
        card.add_layout(_row("Reference camera", self.ref_cam_combo, "Interpolation", self.interp_combo))
        card.add_layout(_row(self.level_check))
        card.add_layout(_row(self.align_first_check))
        card.add_layout(_row(self.autocrop_check, "Margin", self.margin_spin))
        self.align_canvas = PlotCanvas(figsize=(9, 3.2), show_toolbar=False)
        card.add(self.align_canvas)
        layout.addWidget(card)

    def _build_intensity_card(self, layout):
        card = Card("5. Intensity equalisation (inter + intra camera)",
                    "Pairwise log-gain solver (as in N_cameras_images_processing.py): for every pair of frames the median log "
                    "ratio of their common pixels is measured, and one multiplicative gain per frame is solved for the whole "
                    "sequence, so that all frames of all cameras get the same brightness. Only a scaling: the speckle is not "
                    "modified. The older method (smoothed median scales) is kept as an option.")
        self.int_method_combo = self._bind("intensity_method", _combo([
            ("pairwise", "Pairwise log-gain solver (recommended)"), ("median_scale", "Median scales, smoothed (older)"),
            ("none", "None")]))
        self.eq_gmin_spin = self._bind("equalize_gain_min", _dspin(0.01, 1, 0.6, 0.05, 2))
        self.eq_gmax_spin = self._bind("equalize_gain_max", _dspin(1, 100, 1.6, 0.1, 2))
        self.eq_thumb_spin = self._bind("equalize_thumb_side", _ispin(16, 4096, 160, " px"))
        self.eq_pairs_spin = self._bind("equalize_max_pairs", _ispin(1, 100000000, 40000))
        self.eq_pairs_spin.setToolTip("All pairs of frames are used up to this number; above it a sparse graph (time neighbours + "
                                      "a few anchors) is used. 3 cameras x 300 frames = 405 000 pairs, so the sparse graph applies.")
        self.eq_thumb_spin.setToolTip("The pairs are compared on thumbnails of this size (longest side): fast and less noisy.")
        card.add_layout(_row("Method", self.int_method_combo))
        card.add_layout(_row("Gain limits", self.eq_gmin_spin, self.eq_gmax_spin, "Thumbnail", self.eq_thumb_spin,
                             "Max. pairs (else sparse graph)", self.eq_pairs_spin))
        self.gnorm_check = self._bind("use_global_intensity_norm", QCheckBox("Use a global reference level"))
        self.gnorm_check.setChecked(True)
        self.gmode_combo = self._bind("global_intensity_mode", _combo([("first_frame", "First frame"), ("median", "Median"), ("mean", "Mean")]))
        self.smooth_combo = self._bind("smoothing_method", _combo([("gaussian", "Gaussian"), ("ema", "Exponential (EMA)"), ("moving_average", "Moving average")]))
        self.gsigma_spin = self._bind("gaussian_sigma", _dspin(0, 1e6, 0, 1, 2, special="auto"), "optional")
        self.gfactor_spin = self._bind("gaussian_sigma_auto_factor", _dspin(0.01, 100, 1.0, 0.1, 2))
        self.gradpct_spin = self._bind("gradient_threshold_percentile", _dspin(0, 100, 85, 1, 1))
        self.blend_spin = self._bind("blend_strength", _dspin(0, 1, 0.9, 0.05, 2))
        self.minscale_spin = self._bind("min_scale", _dspin(0.01, 1, 0.5, 0.05, 2))
        self.maxscale_spin = self._bind("max_scale", _dspin(1, 100, 2.0, 0.1, 2))
        card.add_layout(_row("Older method:", self.gnorm_check, "Reference", self.gmode_combo, "Smoothing", self.smooth_combo))
        card.add_layout(_row("Gaussian sigma (frames)", self.gsigma_spin, "auto factor x N", self.gfactor_spin,
                             "Gradient percentile", self.gradpct_spin, "Blend", self.blend_spin))
        card.add_layout(_row("Scale clamp", self.minscale_spin, self.maxscale_spin))
        self.tone_combo = self._bind("tone_matching", _combo([("none", "None (recommended)"),
                                                              ("lut", "Match every frame to the first frame of the reference camera (LUT)")]))
        self.tone_combo.setToolTip("Overlap-masked, slope-limited look-up table of the N-camera script. In that script it was computed "
                                   "but never applied to the output. It forces the grey-level histogram of every frame onto that of one "
                                   "frame, so it also removes the REAL brightness changes of the specimen.")
        card.add_layout(_row("Tone matching", self.tone_combo))
        self.int_method_combo.currentIndexChanged.connect(self._update_enabled_states)
        layout.addWidget(card)

    def _build_homogenisation_card(self, layout):
        card = Card("6. Gradient homogenisation (background) - optional",
                    "Off by default (the N-camera pipeline does none). Divide mode: the image is divided by its estimated illumination shape (unit mean, so the frame brightness is "
                    "preserved), then ONE global gain maps the 99.9th percentile of the sequence to the chosen fraction of 65535. "
                    "The legacy mode subtracts a gaussian blur and stretches every frame to 0..65535 (it destroys the brightness "
                    "relation between frames).")
        self.bg_mode_combo = self._bind("background_mode", _combo([
            ("none", "None (default: the speckle is left untouched)"), ("divide", "Divide by illumination shape"),
            ("legacy", "Legacy: subtract + per-frame stretch")]))
        self.bg_model_combo = self._bind("background_model", _combo([("gaussian", "Gaussian, masked local-linear (handles borders)"),
                                                                       ("polynomial", "Robust polynomial surface")]))
        self.bg_kernel_spin = self._bind("background_kernel", _ispin(0, 100000, 0, " px", "auto"), "optional")
        self.bg_scale_spin = self._bind("background_kernel_scale", _dspin(0.01, 10, 1.0, 0.1, 2))
        self.poly_spin = self._bind("background_poly_order", _ispin(1, 8, 3))
        self.out_scaling_combo = self._bind("output_scaling", _combo([("none", "No rescaling"), ("global", "One global gain (percentile)")]))
        self.out_target_spin = self._bind("output_target_fraction", _dspin(0.1, 1.0, 0.92, 0.02, 2))
        card.add_layout(_row("Mode", self.bg_mode_combo, "Model", self.bg_model_combo))
        card.add_layout(_row("Kernel (sigma = 0.15 x kernel)", self.bg_kernel_spin, "scale of min side", self.bg_scale_spin,
                             "Polynomial order", self.poly_spin))
        card.add_layout(_row("Output scaling", self.out_scaling_combo, "target fraction of 65535", self.out_target_spin))
        self.bg_mode_combo.currentIndexChanged.connect(self._update_enabled_states)
        self.bg_model_combo.currentIndexChanged.connect(self._update_enabled_states)
        prev_row = QHBoxLayout()
        self.hom_cam_combo = _combo([(c, CAM_LABELS[c]) for c in range(3)])
        self.hom_preview_btn = QPushButton("Preview on the calibrated frame")
        self.hom_preview_btn.setToolTip("Needs step 1 (Prepare). Shows the frame, the illumination estimate and the result, "
                                        "with the mean intensity profile along x and y before/after.")
        self.hom_preview_btn.clicked.connect(self._on_hom_preview)
        prev_row.addWidget(QLabel("Preview camera"))
        prev_row.addWidget(self.hom_cam_combo)
        prev_row.addWidget(self.hom_preview_btn)
        prev_row.addStretch(1)
        card.add_layout(prev_row)
        self.hom_canvas = PlotCanvas(figsize=(10, 5.2))
        card.add(self.hom_canvas)
        layout.addWidget(card)

    def _build_denoise_card(self, layout):
        card = Card("7. Denoising and despeckle",
                    "Off by default: denoising smooths the speckle that DIC correlates. Spatial denoisers work on float data "
                    "(no 8-bit round trip); the temporal median is taken per camera.")
        self.denoiser_combo = self._bind("denoiser", _combo([("none", "None"), ("bilateral", "Bilateral"), ("nlmeans", "Non-local means"),
                                                              ("bm3d", "BM3D (falls back to non-local means)"), ("tv", "Total variation"),
                                                              ("temporal_median", "Temporal median (per camera)")]))
        card.add_layout(_row("Denoiser", self.denoiser_combo))
        card.add_layout(_row("Bilateral: diameter", self._bind("bilateral_diameter", _ispin(1, 50, 2)),
                             "sigma colour (8-bit units)", self._bind("bilateral_sigma_color", _dspin(1, 500, 75, 5, 1)),
                             "sigma space", self._bind("bilateral_sigma_space", _dspin(1, 500, 75, 5, 1))))
        card.add_layout(_row("NL-means: patch", self._bind("nlmeans_patch_size", _ispin(1, 31, 7)),
                             "distance", self._bind("nlmeans_patch_distance", _ispin(1, 61, 11)),
                             "h factor", self._bind("nlmeans_h_factor", _dspin(0.01, 10, 0.8, 0.1, 2))))
        card.add_layout(_row("BM3D sigma", self._bind("bm3d_sigma", _dspin(0, 1, 0, 0.005, 4, special="auto"), "optional"),
                             "TV weight", self._bind("tv_weight", _dspin(0.001, 5, 0.02, 0.005, 3)),
                             "Temporal median window", self._bind("temporal_median_window", _ispin(3, 21, 3))))
        self.despeckle_check = self._bind("despeckle", QCheckBox("Despeckle"))
        card.add_layout(_row(self.despeckle_check, self._bind("despeckle_mode", _combo([("median", "Median"), ("morphology", "Morphology"), ("hybrid", "Hybrid")])),
                             "size (odd)", self._bind("despeckle_size", _ispin(1, 31, 1))))
        layout.addWidget(card)

    def _build_stretch_card(self, layout):
        card = Card("8. Contrast stretch (bring out the details)",
                    "Like Brightness & Contrast > Auto in ImageJ: a linear window [min, max] is mapped to the full 16-bit range. "
                    "It is written to all_stretched/ (same file names as all_processed) and to a second video; all_processed is "
                    "NOT modified. The pixels outside the window are clipped: use this copy to look at the data, not to correlate. "
                    "One common window for the whole sequence keeps the brightness consistent between frames and cameras.")
        self.stretch_mode_combo = self._bind("stretch_mode", _combo([
            ("imagej_auto", "ImageJ Auto (histogram based)"), ("percentile", "Saturated pixels (%)"),
            ("manual", "Manual min / max"), ("none", "None (no stretched copy)")]))
        self.stretch_scope_combo = self._bind("stretch_scope", _combo([
            ("global", "One window for the whole sequence (recommended)"), ("per_camera", "One window per camera"),
            ("per_frame", "One window per frame (like ImageJ on each image; brightness may flicker)")]))
        self.stretch_sat_spin = self._bind("stretch_saturated_pct", _dspin(0, 49.9, 0.35, 0.05, 2, " %"))
        self.stretch_min_spin = self._bind("stretch_min", _dspin(0, 65535, 0, 100, 0))
        self.stretch_max_spin = self._bind("stretch_max", _dspin(1, 65535, 65535, 100, 0))
        self.stretch_gamma_spin = self._bind("stretch_gamma", _dspin(0.1, 5, 1.0, 0.1, 2))
        self.stretch_clahe_check = self._bind("stretch_clahe", QCheckBox("Local contrast (CLAHE) after the stretch"))
        self.stretch_clip_spin = self._bind("stretch_clahe_clip", _dspin(0.1, 40, 2.0, 0.5, 1))
        self.stretch_tiles_spin = self._bind("stretch_clahe_tiles", _ispin(1, 64, 8))
        self.stretch_sample_spin = self._bind("stretch_sample_frames", _ispin(1, 100000, 200))
        for w in (self.stretch_mode_combo, self.stretch_scope_combo, self.stretch_clahe_check):
            (w.currentIndexChanged if isinstance(w, QComboBox) else w.toggled).connect(self._update_enabled_states)
        card.add_layout(_row("Mode", self.stretch_mode_combo, "Window", self.stretch_scope_combo))
        card.add_layout(_row("Saturated pixels", self.stretch_sat_spin, "Manual min", self.stretch_min_spin, "max", self.stretch_max_spin,
                             "Gamma", self.stretch_gamma_spin))
        card.add_layout(_row(self.stretch_clahe_check, "clip limit", self.stretch_clip_spin, "tiles", self.stretch_tiles_spin,
                             "Frames used for the histograms", self.stretch_sample_spin))
        prow = QHBoxLayout()
        self.stretch_preview_btn = QPushButton("Preview")
        self.stretch_preview_btn.setToolTip("Uses a frame of all_processed when the sequence has been processed, otherwise the "
                                            "homogenised preview of the camera chosen in step 6.")
        self.stretch_apply_btn = QPushButton("Re-stretch the processed sequence")
        self.stretch_apply_btn.setToolTip("Rebuilds only all_stretched/ (and its video) from the existing all_processed/ - no reprocessing.")
        self.stretch_preview_btn.clicked.connect(self._on_stretch_preview)
        self.stretch_apply_btn.clicked.connect(self._on_stretch_apply)
        prow.addWidget(self.stretch_preview_btn)
        prow.addWidget(self.stretch_apply_btn)
        prow.addStretch(1)
        card.add_layout(prow)
        self.stretch_canvas = PlotCanvas(figsize=(10, 3.6))
        card.add(self.stretch_canvas)
        layout.addWidget(card)

    def _build_export_card(self, layout):
        card = Card("9. Exports", "Everything is 16-bit TIFF. The processed sequence (all_processed/proc_XXXXXX.tif) is time-ordered; "
                                  "the video is lossless FFV1 (needs ffmpeg).")
        self.exp_calib = self._bind("export_calibrated", QCheckBox("calibrated_camN (before alignment)"))
        self.exp_crop = self._bind("export_cropped", QCheckBox("cropped_camN (aligned + cropped)"))
        self.exp_comb = self._bind("export_combined_cropped", QCheckBox("all_croppedANDaligned (time-ordered)"))
        self.exp_proc = self._bind("export_processed", QCheckBox("all_processed (after homogenisation / denoising; needed to re-stretch later)"))
        for c in (self.exp_calib, self.exp_crop, self.exp_comb, self.exp_proc):
            c.setChecked(True)
        card.add_layout(_row(self.exp_calib, self.exp_crop))
        card.add_layout(_row(self.exp_comb, self.exp_proc))
        self.cmp_check = self._bind("save_comparisons", QCheckBox("Comparison images"))
        self.cmp_check.setChecked(True)
        card.add_layout(_row(self.cmp_check, "every", self._bind("comparison_every", _ispin(1, 100000, 10)), "frames, tile height",
                             self._bind("comparison_tile_height", _ispin(50, 4000, 400, " px"))))
        self.video_check = self._bind("export_video", QCheckBox("Videos (lossless; the stretched copy gets <name>_stretched)"))
        self.video_check.setChecked(True)
        self.video_name = self._bind("video_filename", QLineEdit("processed.mkv"))
        card.add_layout(_row(self.video_check, "fps", self._bind("video_fps", _ispin(1, 1000, 24)), "base name (extension set by the formats)", self.video_name))
        self.vid_mkv = self._bind("export_video_mkv", QCheckBox("MKV (FFV1, 16-bit)"))
        self.vid_avi = self._bind("export_video_avi", QCheckBox("AVI (FFV1, 16-bit)"))
        self.vid_mp4 = self._bind("export_video_mp4", QCheckBox("MP4 (H.264 lossless)"))
        for c in (self.vid_mkv, self.vid_avi, self.vid_mp4):
            c.setChecked(True)
        self.vid_mp4.toggled.connect(self._update_enabled_states)
        self.vid_mp4.setToolTip("MP4 cannot carry FFV1 or 16-bit video: the frames are rescaled to 8 or 10 bit (v / 65535 x (2^bits - 1)) "
                                "and then encoded exactly (H.264 qp 0, full range, High 4:4:4 Predictive). Use MKV / AVI for the 16-bit data. "
                                "MP4 plays in VLC / mpv / ffplay; many other players do not support lossless H.264.")
        self.mp4_depth_combo = self._bind("mp4_bit_depth", _combo([(8, "8-bit"), (10, "10-bit")]))
        card.add_layout(_row("Formats", self.vid_mkv, self.vid_avi, self.vid_mp4, "MP4 depth", self.mp4_depth_combo))
        self.video_cam_check = self._bind("export_video_per_camera", QCheckBox("one video per camera (cam<N>_<file name>) in addition to the time-ordered one"))
        self.video_cam_check.setChecked(True)
        card.add_layout(_row(self.video_cam_check))
        self.noburn_check = self._bind("export_no_timestamp_copy", QCheckBox("also write all_processed_no_timestamp/ when the time is burned in"))
        self.noburn_check.setChecked(True)
        card.add_layout(_row(self.noburn_check))
        layout.addWidget(card)

    def _build_run_card(self, layout):
        card = Card("10. Run", "Step by step, or all steps in a row. The alignment and crop are saved in "
                              "rotation_translation_report.json and can be reloaded.")
        row = QHBoxLayout()
        self.prepare_btn = QPushButton("1  Prepare calibration")
        self.align_btn = QPushButton("2  Align (rotation + translation)")
        self.crop_btn = QPushButton("3  Crop")
        self.run_btn = QPushButton("4  Process and export")
        self.all_btn = QPushButton("Run all steps")
        self.stop_btn = QPushButton("Stop")
        self.prepare_btn.clicked.connect(self._on_prepare_clicked)
        self.align_btn.clicked.connect(lambda: self._do_alignment())
        self.crop_btn.clicked.connect(lambda: self._do_crop())
        self.run_btn.clicked.connect(self._on_process_clicked)
        self.all_btn.clicked.connect(self._on_run_all)
        self.stop_btn.clicked.connect(self._on_stop)
        for b in (self.prepare_btn, self.align_btn, self.crop_btn, self.run_btn, self.all_btn, self.stop_btn):
            row.addWidget(b)
        card.add_layout(row)
        row2 = QHBoxLayout()
        self.load_align_btn = QPushButton("Load alignment + crop from a report…")
        self.save_cfg_btn = QPushButton("Save settings…")
        self.load_cfg_btn = QPushButton("Load settings…")
        self.open_btn = QPushButton("Open output folder")
        self.load_align_btn.clicked.connect(self._on_load_alignment)
        self.save_cfg_btn.clicked.connect(self._on_save_settings)
        self.load_cfg_btn.clicked.connect(self._on_load_settings)
        self.open_btn.clicked.connect(self._on_open_output)
        for b in (self.load_align_btn, self.save_cfg_btn, self.load_cfg_btn, self.open_btn):
            row2.addWidget(b)
        row2.addStretch(1)
        card.add_layout(row2)
        self.state_label = QLabel("")
        card.add(self.state_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        card.add(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMinimumHeight(150)
        card.add(self.log)
        layout.addWidget(card)

    # ------------------------------------------------------------------ number of cameras
    def _n_cameras(self) -> int:
        return int(self.n_cameras_spin.value())

    @staticmethod
    def _refill_combo(combo, n: int):
        keep = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for c in range(n):
            combo.addItem(CAM_LABELS[c], c)
        combo.blockSignals(False)
        i = combo.findData(keep)
        combo.setCurrentIndex(i if i >= 0 else 0)

    def _on_n_cameras_changed(self, *_):
        n = self._n_cameras()
        for c, widgets in enumerate(self._cam_rows):
            for w in widgets:
                w.setVisible(c < n)
        self._refill_combo(self.ref_cam_combo, n)
        self._refill_combo(self.hom_cam_combo, n)
        self.ref_cam_combo.setEnabled(n > 1)
        # a different number of cameras invalidates everything computed so far
        if self._prep is not None and self._prep.cfg.n_cameras != n:
            self._prep = self._alignment = self._aligned_previews = None
            self._crop_bbox = self._crop_rois = self._crop_reason = None
            self._update_enabled_states()

    # ------------------------------------------------------------------ state
    def _update_enabled_states(self, *_):
        robust = self.flat_method_combo.currentData() == "robust"
        self.robust_box.setVisible(robust)
        self.legacy_box.setVisible(not robust)
        mode = self.bg_mode_combo.currentData()
        model = self.bg_model_combo.currentData()
        self.bg_model_combo.setEnabled(mode == "divide")
        self.bg_kernel_spin.setEnabled(mode != "none" and not (mode == "divide" and model == "polynomial"))
        self.bg_scale_spin.setEnabled(self.bg_kernel_spin.isEnabled())
        self.poly_spin.setEnabled(mode == "divide" and model == "polynomial")
        self.out_scaling_combo.setEnabled(mode != "legacy")
        self.out_target_spin.setEnabled(mode != "legacy")
        imeth = self.int_method_combo.currentData()
        for w in (self.eq_gmin_spin, self.eq_gmax_spin, self.eq_thumb_spin, self.eq_pairs_spin):
            w.setEnabled(imeth == "pairwise")
        for w in (self.gnorm_check, self.gmode_combo, self.smooth_combo, self.gsigma_spin, self.gfactor_spin, self.gradpct_spin,
                  self.blend_spin, self.minscale_spin, self.maxscale_spin):
            w.setEnabled(imeth == "median_scale")
        self.mp4_depth_combo.setEnabled(self.vid_mp4.isChecked())
        smode, sclahe = self.stretch_mode_combo.currentData(), self.stretch_clahe_check.isChecked()
        self.stretch_scope_combo.setEnabled(smode != "none")
        self.stretch_sat_spin.setEnabled(smode == "percentile")
        self.stretch_min_spin.setEnabled(smode == "manual")
        self.stretch_max_spin.setEnabled(smode == "manual")
        self.stretch_gamma_spin.setEnabled(smode != "none")
        self.stretch_clahe_check.setEnabled(smode != "none")
        self.stretch_clip_spin.setEnabled(smode != "none" and sclahe)
        self.stretch_tiles_spin.setEnabled(smode != "none" and sclahe)
        self.stretch_sample_spin.setEnabled(smode in ("imagej_auto", "percentile") and self.stretch_scope_combo.currentData() != "per_frame")
        busy = self._thread is not None
        self.stretch_preview_btn.setEnabled(not busy and smode != "none")
        self.stretch_apply_btn.setEnabled(not busy and smode != "none")
        self.prepare_btn.setEnabled(not busy)
        self.align_btn.setEnabled(not busy and self._prep is not None)
        self.crop_btn.setEnabled(not busy and self._alignment is not None)
        self.run_btn.setEnabled(not busy and self._alignment is not None)
        self.all_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.hom_preview_btn.setEnabled(not busy and self._prep is not None)
        mark = lambda ok: "done" if ok else "to do"
        self.state_label.setText(
            f"Prepare: {mark(self._prep is not None)}   |   Alignment: {mark(self._alignment is not None)}   |   "
            f"Crop: {'done' if self._crop_bbox is not None else ('skipped' if self._crop_reason else 'to do')}")

    def _log(self, msg: str):
        self.log.appendPlainText(msg)

    def _on_progress(self, msg: str, frac: float):
        if frac >= 0:
            self.progress.setValue(int(frac * 1000))
        self.state_label.setText(msg)

    def _start_thread(self, worker, on_finished):
        self._cancel.clear()
        self._thread = QThread()
        self._worker = worker
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        for sig in (worker.finished, worker.failed, worker.cancelled):
            sig.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_done)
        self._thread.start()
        self._update_enabled_states()

    def _on_thread_done(self):
        self._thread = None
        self._worker = None
        self._update_enabled_states()

    def _on_failed(self, msg: str):
        self._chain = False
        self._log("ERROR: " + msg)
        QMessageBox.warning(self, "Cameras Synchronisation", msg)

    def _on_cancelled(self):
        self._chain = False
        self._log("Stopped by user.")

    def _on_stop(self):
        self._cancel.set()
        self._chain = False

    # ------------------------------------------------------------------ steps
    def _validated_config(self):
        cfg = self.collect_config()
        try:
            cfg.validate()
        except cs.CameraSyncError as exc:
            self._log("Settings: " + str(exc))
            QMessageBox.warning(self, "Cameras Synchronisation", str(exc))
            return None
        return cfg

    def _on_prepare_clicked(self):
        cfg = self._validated_config()
        if cfg is None:
            return
        self._prep = self._alignment = self._aligned_previews = None
        self._crop_bbox = self._crop_rois = self._crop_reason = None
        self._log("Preparing calibration (master dark/flat, gain maps) ...")
        self.progress.setValue(0)
        self._start_thread(_PrepareWorker(cfg, self._cancel), self._on_prepared)

    def _on_prepared(self, prep):
        self._prep = prep
        for cam in range(prep.cfg.n_cameras):
            d = prep.diagnostics[f"cam{cam + 1}"]
            self._log(f"Camera {cam + 1}: darks accepted {d['dark']['accepted_dark_frames']}/{d['dark']['total_dark_frames']}, "
                      f"flats accepted {d['flat']['accepted_flat_frames']}/{d['flat']['total_flat_frames']}"
                      + (" (WARNING: " + d["flat"]["warning"] + ")" if "warning" in d["flat"] else ""))
        self._log(f"{len(prep.files[0])} frames per camera after trimming; image {prep.image_shape[1]}x{prep.image_shape[0]} px.")
        self._update_enabled_states()
        if self._chain:
            self._thread_then(self._chain_after_prepare)

    def _thread_then(self, fn):
        """Run `fn` once the worker thread is fully finished (dialogs must not open while it is quitting)."""
        QTimer.singleShot(0, lambda: fn() if self._thread is None else self._thread_then(fn))

    def _chain_after_prepare(self):
        if not self._chain:
            return
        if not self._do_alignment() or not self._chain:
            self._chain = False
            return
        if not self._do_crop():
            self._chain = False
            return
        self._on_process_clicked()

    def _on_run_all(self):
        self._chain = True
        self._on_prepare_clicked()

    # ---- alignment
    def _do_alignment(self) -> bool:
        prep = self._prep
        if prep is None:
            return False
        cfg = prep.cfg
        n = cfg.n_cameras
        ref = int(self.ref_cam_combo.currentData())
        interp = self.interp_combo.currentData()
        h, w = prep.image_shape
        size = (w, h)
        level = 0.0
        if self.level_check.isChecked():
            dlg = ImagePickDialog(
                f"Levelling - draw the baseline on camera {ref + 1}" + ("  (reference)" if n > 1 else ""), prep.previews[ref], "line",
                "Click the two ends of a line that must become HORIZONTAL (e.g. the bar or specimen edge), left end first. "
                "The image is rotated about its centre; with several cameras all of them follow. Zoom with the wheel; Accept when done.",
                parent=self)
            if not dlg.exec():
                self._log("Alignment cancelled.")
                return False
            level = cs.leveling_angle(tuple(dlg.points))
        if n == 1:
            # a single camera has nothing to align: identity, or the levelling rotation alone; the crop step is still available
            self._alignment = (cs.make_alignment((h, w), 0, [level], [(0.0, 0.0)], "levelling_only") if level
                               else cs.identity_alignment((h, w), 0, 1))
            self._after_alignment()
            self._log("Single camera: no alignment needed" + (f"; levelling rotation {level:.3f} deg." if level else "."))
            return True
        order = [ref] + [c for c in range(n) if c != ref]
        lines = {}
        for cam in order:
            dlg = ImagePickDialog(
                f"Rotation - draw a line on camera {cam + 1}" + ("  (reference)" if cam == ref else ""), prep.previews[cam], "line",
                "Click the two ends of a line along a feature visible on all cameras (e.g. a straight edge). Click the ends in the "
                "SAME order on every camera (e.g. left to right). Zoom with the wheel; Accept when done.", parent=self)
            if not dlg.exec():
                self._log("Alignment cancelled.")
                return False
            lines[cam] = tuple(dlg.points)
        angles = [level] * n                      # rotations about the same centre add up: levelling + relative rotation
        for cam in order[1:]:
            angles[cam] = level + cs.rotation_to_apply(lines[ref], lines[cam])
        rot_prev = [cs.warp_image(prep.previews[c], cs.build_affine((h, w), angles[c]), size, interp) if angles[c] else prep.previews[c]
                    for c in range(n)]
        pts = {}
        for cam in order:
            dlg = ImagePickDialog(
                f"Translation - click the same point on camera {cam + 1}" + ("  (reference)" if cam == ref else ""), rot_prev[cam], "point",
                "Images are already rotated. Click ONE point that is clearly the same physical point on every camera "
                "(a corner, a mark ...). Accept when done.", parent=self)
            if not dlg.exec():
                self._log("Alignment cancelled.")
                return False
            pts[cam] = dlg.points[0]
        trans = [(0.0, 0.0)] * n
        for cam in order[1:]:
            trans[cam] = (pts[ref][0] - pts[cam][0], pts[ref][1] - pts[cam][1])
        self._alignment = cs.make_alignment((h, w), ref, angles, trans)
        self._after_alignment()
        self._log("Alignment: rotations (deg) " + ", ".join(f"cam{c + 1}={a:.3f}" for c, a in enumerate(angles))
                  + "; translations (px) " + ", ".join(f"cam{c + 1}=({t[0]:.1f},{t[1]:.1f})" for c, t in enumerate(trans)))
        return True

    def _after_alignment(self):
        prep = self._prep
        al = self._alignment
        h, w = prep.image_shape
        interp = self.interp_combo.currentData()
        self._aligned_previews = [cs.warp_image(prep.previews[c], al.matrix(c), (w, h), interp) if not cs.is_identity(al.matrix(c))
                                  else prep.previews[c] for c in range(prep.cfg.n_cameras)]
        self._crop_bbox = self._crop_rois = None
        self._crop_reason = None
        # red/green composite: reference (red) vs camera (green) -> yellow where they coincide
        fig = self.align_canvas.figure
        fig.clear()
        ref = al.reference_camera
        others = [c for c in range(prep.cfg.n_cameras) if c != ref]
        if not others:
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(cs.display_uint8(self._aligned_previews[ref]), cmap="gray")
            ax.set_title(f"cam {ref + 1} (single camera, no alignment)", fontsize=9)
            ax.axis("off")
        ncol = min(len(others), 4)
        nrow = int(np.ceil(len(others) / ncol)) if others else 0
        for k, c in enumerate(others):
            ax = fig.add_subplot(nrow, ncol, k + 1)
            rgb = np.dstack([cs.display_uint8(self._aligned_previews[ref]), cs.display_uint8(self._aligned_previews[c]),
                             np.zeros((h, w), np.uint8)])
            ax.imshow(rgb)
            ax.set_title(f"cam {ref + 1} (red) / cam {c + 1} (green)", fontsize=9)
            ax.axis("off")
        self.align_canvas.refresh()
        try:
            self._write_alignment_report()
        except OSError as exc:
            self._log(f"Could not write the alignment report: {exc}")
        self._update_enabled_states()

    def _write_alignment_report(self):
        cfg = self.collect_config()
        os.makedirs(cfg.master_output, exist_ok=True)
        al = self._alignment
        data = {"reference_camera": al.reference_camera, "rotation": {f"cam{c + 1}": a for c, a in enumerate(al.angles_deg)},
                "translation": {f"cam{c + 1}": {"dx": t[0], "dy": t[1]} for c, t in enumerate(al.translations_px)},
                "image_size": list(al.image_size), "method": al.method, "n_cameras": len(al.angles_deg),
                "pre_rotation_deg_reference_camera": float(al.angles_deg[al.reference_camera])}
        if self._crop_bbox is not None:
            data["autocrop"] = {"enabled": True, "intersection_bbox": dict(zip("xywh", map(int, self._crop_bbox)))}
        with open(cfg.report_path, "w") as fh:
            json.dump(data, fh, indent=2)

    # ---- crop
    def _do_crop(self) -> bool:
        prep = self._prep
        if prep is None or self._alignment is None:
            return False
        if not self.autocrop_check.isChecked():
            self._crop_bbox, self._crop_rois, self._crop_reason = None, None, "disabled"
            self._update_enabled_states()
            return True
        h, w = prep.image_shape
        interp = self.interp_combo.currentData()
        n = prep.cfg.n_cameras
        masks = [cs.warp_valid_mask((h, w), self._alignment.matrix(c), (w, h), interp) if not cs.is_identity(self._alignment.matrix(c))
                 else np.ones((h, w), bool) for c in range(n)]
        common = masks[0]
        for m in masks[1:]:
            common = common & m
        proposal = cs.largest_true_rectangle(common)
        rois = []
        for cam in range(n):
            dlg = ImagePickDialog(
                f"Crop - select the region on camera {cam + 1}" if n > 1 else "Crop - select the region to keep",
                self._aligned_previews[cam], "rect",
                ("Drag the rectangle to keep (aligned image). The proposal is the largest area valid on all %d cameras. "
                 "The intersection of the %d rectangles is applied to every camera." % (n, n)) if n > 1 else
                "Drag the rectangle to keep. The proposal is the largest valid area.", initial_rect=proposal, parent=self)
            if not dlg.exec() or dlg.rect is None:
                self._log("Crop cancelled.")
                return False
            rois.append(dlg.rect)
        bbox, reason = cs.intersect_rois(rois, (h, w), int(self.margin_spin.value()))
        if bbox is None:
            self._log(f"The intersection of the rectangles is unusable ({reason}); crop disabled.")
            self._crop_bbox, self._crop_rois, self._crop_reason = None, rois, reason
        else:
            self._crop_bbox, self._crop_rois, self._crop_reason = bbox, rois, None
            self._log("Crop: x=%d y=%d w=%d h=%d" % bbox)
        try:
            self._write_alignment_report()
        except OSError:
            pass
        self._update_enabled_states()
        return True

    # ---- processing
    def _on_process_clicked(self):
        if self._prep is None or self._alignment is None:
            return
        cfg = self._validated_config()
        if cfg is None:
            self._chain = False
            return
        # the settings may have changed since 'Prepare': calibration-related ones need a new Prepare
        old = self._prep.cfg
        stale = [n for n in ("dark_folders", "flat_folders", "light_folders", "input_folders", "input_bit_depth", "skip_start",
                             "skip_end", "n_cameras", "flat_method", "use_static_light", "flat_lowfreq_sigma_frac", "flat_pixel_gain_limits",
                             "flat_pixel_smooth_sigma_px", "flat_total_gain_limits", "gain_smooth_kernel_frac", "min_gain", "max_gain", "calibration_pedestal", "dark_scaling")
                 if getattr(cfg, n) != getattr(old, n)]
        if stale:
            msg = "Calibration settings changed since step 1 (" + ", ".join(stale) + "): run 'Prepare calibration' again."
            self._log(msg)
            QMessageBox.information(self, "Cameras Synchronisation", msg)
            self._chain = False
            return
        self._prep.cfg = cfg           # everything else (exports, homogenisation, denoising ...) is read at processing time
        if cfg.autocrop and self._crop_bbox is None and self._crop_reason in (None, "disabled"):
            self._log("Crop not done yet: do step 3 first (or untick the interactive crop).")
            self._chain = False
            return
        self._log("Processing ...")
        self.progress.setValue(0)
        bbox = self._crop_bbox if cfg.autocrop else None
        worker = _ProcessWorker(self._prep, self._alignment, bbox, self._crop_rois if bbox else None,
                                self._crop_reason if bbox is None else None, self._cancel)
        self._start_thread(worker, self._on_processed)

    def _on_processed(self, report: dict):
        self._chain = False
        self.progress.setValue(1000)
        self._log(f"Done: {report.get('n_output_frames')} frames written to {self._prep.cfg.master_output}")
        for wmsg in report.get("warnings", []):
            self._log("Warning: " + wmsg)
        v = report.get("video_export")
        if v is not None:
            self._log("Video: " + (", ".join(v["files"].values()) if v["enabled"] else "not created (" + str(v["reason"]) + ")"))
            if v["enabled"] and v["failed"]:
                self._log("Video formats that failed: " + "; ".join(f"{k}: {m}" for k, m in v["failed"].items()))
        eq = report.get("intensity_equalisation")
        if eq and eq.get("method") == "pairwise_log_gain":
            self._log(f"Intensity equalisation: {eq['n_frames']} frames, {eq['n_equations']} pairs ({eq['pair_mode']}); gains "
                      f"{eq['gain_min']:.3f}-{eq['gain_max']:.3f}; log mismatch {eq['rms_log_mismatch_before']:.4f} -> "
                      f"{eq['rms_log_mismatch_after']:.4f} (gains.json written)")
        tm = report.get("tone_matching")
        if tm:
            self._log(f"Tone matching to {tm['reference']}: {tm['paths_used']}")
        st = report.get("contrast_stretch")
        if st:
            w = st.get("window") or st.get("window_per_camera") or st.get("window_range")
            self._log(f"Contrast stretch ({st['mode']}, {st['scope']}): window {w}, written to {st['folder']}")
        g = report.get("output_gain")
        if g:
            self._log(f"Output gain: {g['gain']:.4g} ({g['mode']})")

    # ---- preview of the homogenisation
    def _on_hom_preview(self):
        if self._prep is None:
            return
        cfg = self.collect_config()
        cam = int(self.hom_cam_combo.currentData())
        img = self._prep.previews[cam]
        valid = np.ones(img.shape, bool)
        mode = cfg.background_mode
        if mode == "divide":
            out, bg = cs.homogenise_divide(img, valid, cfg)
        elif mode == "legacy":
            out, bg = cs.uniformise_background_legacy(img, cfg), None
        else:
            out, bg = img, None
        fig = self.hom_canvas.figure
        fig.clear()
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 2])
        for k, (arr, title) in enumerate(((img, "calibrated"), (bg if bg is not None else np.zeros_like(img), "illumination estimate"),
                                          (out, "result"))):
            ax = fig.add_subplot(gs[0, k])
            ax.imshow(cs.display_uint8(arr), cmap="gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        h, w = img.shape
        band_y, band_x = slice(h // 2 - h // 10, h // 2 + h // 10), slice(w // 2 - w // 10, w // 2 + w // 10)
        for k, (label, a_in, a_out, x) in enumerate((("mean along y (central band)", img[band_y, :].mean(axis=0), out[band_y, :].mean(axis=0), np.arange(w)),
                                                      ("mean along x (central band)", img[:, band_x].mean(axis=1), out[:, band_x].mean(axis=1), np.arange(h)))):
            ax = fig.add_subplot(gs[1, k])
            ax.plot(x, a_in, label="calibrated", lw=1)
            ax.plot(x, a_out, label="result", lw=1)
            if k == 0 and bg is not None:
                ax.plot(x, bg[band_y, :].mean(axis=0), label="estimate", lw=1, ls="--")
            ax.set_title(label, fontsize=9)
            ax.legend(fontsize=7)
        ax = fig.add_subplot(gs[1, 2])
        edge = lambda a: float(np.mean([a[:h // 20].mean(), a[-h // 20:].mean(), a[:, :w // 20].mean(), a[:, -w // 20:].mean()]) /
                               a[h // 3: 2 * h // 3, w // 3: 2 * w // 3].mean())
        ax.axis("off")
        ax.text(0.0, 0.8, f"edge / centre intensity\ncalibrated: {edge(img):.3f}\nresult: {edge(out):.3f}\n(1.000 = flat)",
                fontsize=10, va="top")
        self.hom_canvas.refresh()

    # ---- contrast stretch
    def _stretch_source_image(self):
        """(image, description): a processed frame when available, otherwise the homogenised preview."""
        cfg = self.collect_config()
        folder = cfg.folder("processed")
        if os.path.isdir(folder):
            names = sorted(f for f in os.listdir(folder) if f.lower().endswith(".tif"))
            if names:
                img = cs.read_image(os.path.join(folder, names[len(names) // 2]), 16)
                return img, f"all_processed/{names[len(names) // 2]}"
        if self._prep is None:
            return None, ""
        cam = int(self.hom_cam_combo.currentData())
        img = self._prep.previews[cam]
        valid = np.ones(img.shape, bool)
        if cfg.background_mode == "divide":
            img = cs.homogenise_divide(img, valid, cfg)[0]
        elif cfg.background_mode == "legacy":
            img = cs.uniformise_background_legacy(img, cfg)
        g = float(np.percentile(img, 99.9))
        return np.clip(img * (cfg.output_target_fraction * cs.MAX16 / g if g > 1e-6 else 1.0), 0, cs.MAX16), f"homogenised preview, camera {cam + 1}"

    def _on_stretch_preview(self):
        img, what = self._stretch_source_image()
        if img is None:
            QMessageBox.information(self, "Cameras Synchronisation", "Run step 1 (Prepare calibration) or process the sequence first.")
            return
        cfg = self.collect_config()
        if cfg.stretch_mode == "none":
            return
        try:
            cfg.validate(check_folders=False)
        except cs.CameraSyncError as exc:
            QMessageBox.warning(self, "Cameras Synchronisation", str(exc))
            return
        img16 = np.clip(np.rint(img), 0, cs.MAX16).astype(np.uint16)
        hist = cs.image_histogram16(img16)
        lo, hi = cs.stretch_limits(hist, cfg)
        out = cs.apply_stretch(img16, lo, hi, cfg.stretch_gamma, (cfg.stretch_clahe_clip, cfg.stretch_clahe_tiles) if cfg.stretch_clahe else None)
        fig = self.stretch_canvas.figure
        fig.clear()
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
        for k, (arr, title) in enumerate(((img16, "before"), (out, "after"))):
            ax = fig.add_subplot(gs[0, k])
            ax.imshow(arr, cmap="gray", vmin=0, vmax=65535)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        ax = fig.add_subplot(gs[0, 2])
        centers = np.arange(0, 65536, 256) + 128
        ax.semilogy(centers, np.add.reduceat(hist, np.arange(0, 65536, 256)) + 1, lw=1)
        ax.axvline(lo, color="tab:red", lw=1, label=f"min {lo:.0f}")
        ax.axvline(hi, color="tab:green", lw=1, label=f"max {hi:.0f}")
        ax.set_title("histogram and window (one frame)", fontsize=9)
        ax.legend(fontsize=7)
        fig.suptitle(what + ("   [the window of a run is computed on the whole sequence]" if cfg.stretch_scope != "per_frame" else ""), fontsize=8)
        self.stretch_canvas.refresh()

    def _on_stretch_apply(self):
        cfg = self._validated_config()
        if cfg is None:
            return
        if cfg.stretch_mode == "none":
            return
        self._log("Contrast stretch of the existing all_processed sequence ...")
        self.progress.setValue(0)
        self._start_thread(_StretchWorker(cfg, self._cancel), self._on_stretched)

    def _on_stretched(self, rep: dict):
        self.progress.setValue(1000)
        w = rep.get("window") or rep.get("window_per_camera") or rep.get("window_range")
        self._log(f"Stretch done ({rep['mode']}, {rep['scope']}): {rep['n_frames']} frames -> {rep['folder']}; window {w}; "
                  f"clipped low {100 * rep['mean_fraction_clipped_low']:.2f} %, high {100 * rep['mean_fraction_clipped_high']:.2f} %")
        v = rep.get("video")
        if v is not None:
            self._log("Stretched video: " + (", ".join(v["files"].values()) if v["enabled"] else "not created (" + str(v["reason"]) + ")"))
        self._update_enabled_states()

    # ---- load / save
    def _on_load_alignment(self):
        if self._prep is None:
            QMessageBox.information(self, "Cameras Synchronisation", "Run step 1 (Prepare calibration) first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Alignment report", self.collect_config().master_output, "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
            n = self._prep.cfg.n_cameras
            n_rep = len(data["rotation"])
            if n_rep != n:
                raise ValueError(f"the report holds {n_rep} camera(s) but the current setting is {n}")
            ref = int(data["reference_camera"])
            angles = [float(data["rotation"][f"cam{c + 1}"]) for c in range(n)]
            trans = [(float(data["translation"][f"cam{c + 1}"]["dx"]), float(data["translation"][f"cam{c + 1}"]["dy"])) for c in range(n)]
            self._alignment = cs.make_alignment(self._prep.image_shape, ref, angles, trans, "loaded_from_report")
            self._after_alignment()
            crop = data.get("autocrop", {})
            if crop.get("enabled") and "intersection_bbox" in crop:
                b = crop["intersection_bbox"]
                self._crop_bbox = (int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]))
                self._crop_rois = None
                self._crop_reason = None
            self.ref_cam_combo.setCurrentIndex(self.ref_cam_combo.findData(ref))
            self._log(f"Alignment loaded from {path}" + (" (with crop)" if self._crop_bbox else " (no crop in the report)"))
        except (OSError, KeyError, ValueError) as exc:
            QMessageBox.warning(self, "Cameras Synchronisation", f"Could not read the alignment: {exc}")
        self._update_enabled_states()

    def _on_save_settings(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", "camera_sync_settings.json", "JSON (*.json)")
        if path:
            with open(path, "w") as fh:
                json.dump(self.collect_config().to_dict(), fh, indent=2)
            self._log("Settings saved: " + path)

    def _on_load_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as fh:
                self.apply_config(cs.CameraSyncConfig.from_dict(json.load(fh)))
            self._update_enabled_states()
            self._log("Settings loaded: " + path)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Cameras Synchronisation", f"Could not read the settings: {exc}")

    def _on_open_output(self):
        out = self.collect_config().master_output
        if out and os.path.isdir(out):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(out)))
