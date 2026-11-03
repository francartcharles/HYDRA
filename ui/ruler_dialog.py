"""
A small modal dialog for calibrating pixel size directly from the
reference frame: click two points a known real-world distance apart on
the image, type that distance in millimetres, and read back the
resulting scale (mm per pixel of whatever frame it was given -- the
Image Correlation page always passes it the RAW reference frame, to
match core.project.DICSettings.pixel_size_mm's own "mm per raw-frame
pixel" convention).
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QDialogButtonBox,
    QInputDialog,
)

from ui.plot_canvas import PlotCanvas
from ui import theme


class RulerDialog(QDialog):
    """Shows `frame_uint8` (already contrast-stretched, 2D grayscale or
    3D RGB) and lets the user click two points on it, then enter the
    real-world distance between them. `result_pixel_size_mm` holds the
    computed mm-per-pixel scale once the user confirms both the points
    and the distance; it stays None if the dialog is cancelled."""

    def __init__(self, frame_uint8: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Measure scale on reference frame")
        self.resize(760, 620)
        self.result_pixel_size_mm: float | None = None
        self._points = []  # up to 2 (x, y) pixel points
        self._point_artists = []
        self._line_artist = None

        layout = QVBoxLayout(self)
        self.hint_label = QLabel(
            "Click two points a known real-world distance apart -- e.g. the two ends of a "
            "ruler visible in the shot, or a specimen dimension you already know -- then confirm "
            "to type that distance in millimetres."
        )
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

        self.canvas = PlotCanvas(figsize=(8, 6))
        layout.addWidget(self.canvas)

        controls = QHBoxLayout()
        self.clear_button = QPushButton("Clear points")
        self.clear_button.clicked.connect(self._clear_points)
        controls.addWidget(self.clear_button)
        controls.addStretch(1)
        self.distance_label = QLabel("Click two points on the image.")
        self.distance_label.setObjectName("cardSubtitle")
        controls.addWidget(self.distance_label)
        layout.addLayout(controls)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        self._ok_button = buttons.button(QDialogButtonBox.Ok)
        self._ok_button.setEnabled(False)
        layout.addWidget(buttons)

        self._frame_shape = frame_uint8.shape[:2]
        self.canvas.clear()
        self._ax = self.canvas.figure.add_subplot(111)
        cmap = "gray" if frame_uint8.ndim == 2 else None
        self._ax.imshow(frame_uint8, cmap=cmap, origin="upper")
        height, width = self._frame_shape
        self._ax.set_xlim(0, width)
        self._ax.set_ylim(height, 0)
        self._ax.set_xlabel("x (px)")
        self._ax.set_ylabel("y (px)")
        self.canvas.refresh()

        self._cid = self.canvas.canvas.mpl_connect("button_press_event", self._on_click)
        # The dialog's PlotCanvas creates its own pyplot-registered Figure
        # (see PlotCanvas.close_figure's docstring) -- this dialog is
        # created and destroyed per measurement rather than kept for the
        # page's lifetime, so it must release it explicitly, however the
        # dialog closes (OK, Cancel, or the window's own close button).
        self.finished.connect(lambda _result: self.canvas.close_figure())

    def _on_click(self, event):
        if event.inaxes != self._ax or event.xdata is None or event.ydata is None:
            return
        if len(self._points) >= 2:
            self._clear_points()
        self._points.append((event.xdata, event.ydata))
        marker, = self._ax.plot(event.xdata, event.ydata, "o", color=theme.ACCENT, markersize=7)
        self._point_artists.append(marker)
        if len(self._points) == 2:
            (x0, y0), (x1, y1) = self._points
            self._line_artist, = self._ax.plot([x0, x1], [y0, y1], "-", color=theme.ACCENT, linewidth=1.5)
            pixel_distance = math.hypot(x1 - x0, y1 - y0)
            self.distance_label.setText(
                f"{pixel_distance:.2f} px selected -- click OK to enter the real distance."
            )
            self._ok_button.setEnabled(True)
        else:
            self.distance_label.setText("Click a second point.")
            self._ok_button.setEnabled(False)
        self.canvas.refresh()

    def _clear_points(self):
        for artist in self._point_artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._point_artists = []
        if self._line_artist is not None:
            try:
                self._line_artist.remove()
            except (ValueError, NotImplementedError):
                pass
            self._line_artist = None
        self._points = []
        self._ok_button.setEnabled(False)
        self.distance_label.setText("Click two points on the image.")
        self.canvas.refresh()

    def _on_accept(self):
        if len(self._points) != 2:
            return
        (x0, y0), (x1, y1) = self._points
        pixel_distance = math.hypot(x1 - x0, y1 - y0)
        if pixel_distance <= 0:
            self.distance_label.setText("The two points must be different -- click again.")
            self._clear_points()
            return
        distance_mm, ok = QInputDialog.getDouble(
            self, "Real-world distance", "Distance between the two points (mm):",
            1.0, 1e-6, 1e9, 4,
        )
        if not ok:
            return
        self.result_pixel_size_mm = distance_mm / pixel_distance
        self.accept()
