"""
Point-probe picker, for the Image Correlation page's "Probes" card.

Unlike `RectPicker`/`RoiMeshPicker` (ui/roi_picker.py), which each draw a
single REGION, `ProbePicker` places any number of individual POINTS on
the displayed (preprocessed) frame -- left-click adds one at the clicked
position, right-click removes whichever existing one is closest to the
click. Each placed point is later matched (see the Image Correlation
page's `_nearest_mesh_index`) to its nearest correlation mesh point, so
a probe reads whatever computed field is currently selected AT that
specific location, across every processed frame -- letting a user watch
a field's time evolution at a handful of chosen spots (e.g. the neck vs.
a point far from it) instead of only the ROI-wide mean.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton

from ui.plot_canvas import PlotCanvas
from ui import theme


class ProbePicker(QWidget):
    """
    probesChanged() fires (no arguments -- call `get_probes()` to read
    the current list) whenever a probe is added or removed by the user,
    directly on the canvas or via the buttons below it.
    """

    probesChanged = Signal()

    def __init__(self, figsize=(8, 5.5), parent=None):
        super().__init__(parent)
        self._frame_shape = None  # (height, width) of the currently shown frame
        self._probes: list[tuple[float, float]] = []  # (x, y), preprocessed-frame pixel coords
        self._ax = None
        self._click_cid = None
        self._marker_artist = None
        self._label_artists: list = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        controls = QHBoxLayout()
        self.remove_last_button = QPushButton("Remove last probe")
        self.remove_last_button.clicked.connect(self.remove_last_probe)
        controls.addWidget(self.remove_last_button)
        self.clear_button = QPushButton("Clear probes")
        self.clear_button.clicked.connect(self.clear_probes)
        controls.addWidget(self.clear_button)
        controls.addStretch(1)
        self.status_label = QLabel("Click on the image to place a probe.")
        self.status_label.setObjectName("cardSubtitle")
        controls.addWidget(self.status_label)
        layout.addLayout(controls)

        self.canvas = PlotCanvas(figsize=figsize)
        layout.addWidget(self.canvas)

        self.show_placeholder("Load a video or image sequence to see frames here.")

    def show_placeholder(self, message: str):
        self._ax = None
        self._click_cid = None
        self._marker_artist = None
        self._label_artists = []
        self.canvas.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True,
                color=theme.TEXT_MUTED, fontsize=10, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.refresh()

    def set_frame(self, frame_uint8: np.ndarray, keep_probes: bool = True):
        """Displays a new frame (2D grayscale or 3D RGB uint8 array,
        already contrast-stretched -- see frame_source.to_display_uint8).

        Rebuilds the axes from scratch every call and reconnects the
        click handler, then redraws the existing probes on top if
        `keep_probes` -- letting the user scrub through frames without
        losing the probes they already placed. `keep_probes=False`
        clears them instead (the caller uses this whenever the
        preprocessed frame's own pixel coordinate system just changed --
        a new source, or a crop/downscale change -- so old positions
        would no longer point at the same physical spot)."""
        height, width = frame_uint8.shape[:2]
        self._frame_shape = (height, width)

        self.canvas.clear()
        self._ax = self.canvas.figure.add_subplot(111)
        cmap = "gray" if frame_uint8.ndim == 2 else None
        self._ax.imshow(frame_uint8, cmap=cmap, origin="upper")
        self._ax.set_xlim(0, width)
        self._ax.set_ylim(height, 0)
        self._ax.set_xlabel("x (px)")
        self._ax.set_ylabel("y (px)")

        self._click_cid = self.canvas.figure.canvas.mpl_connect("button_press_event", self._on_click)

        if not keep_probes:
            self._probes = []

        self._draw_markers()
        self._update_status()

        if self.canvas.toolbar is not None:
            self.canvas.toolbar.update()
        self.canvas.refresh()

    def get_probes(self) -> list[tuple[float, float]]:
        return list(self._probes)

    def set_probes(self, probes):
        """Programmatically sets the probe list (e.g. restoring a saved
        session), without emitting `probesChanged` (the caller already
        knows the value it's setting)."""
        self._probes = [(float(x), float(y)) for x, y in probes] if probes else []
        self._draw_markers()
        self._update_status()
        self.canvas.refresh()

    def remove_last_probe(self):
        if not self._probes:
            return
        self._probes.pop()
        self._draw_markers()
        self._update_status()
        self.canvas.refresh()
        self.probesChanged.emit()

    def clear_probes(self):
        if not self._probes:
            return
        self._probes = []
        self._draw_markers()
        self._update_status()
        self.canvas.refresh()
        self.probesChanged.emit()

    # -- click handling -----------------------------------------------------

    def _on_click(self, event):
        if self._ax is None or event.inaxes != self._ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:  # left click: add
            self._probes.append((float(event.xdata), float(event.ydata)))
        elif event.button == 3:  # right click: remove the closest existing probe
            self._remove_nearest(event.xdata, event.ydata)
        else:
            return
        self._draw_markers()
        self._update_status()
        self.canvas.refresh()
        self.probesChanged.emit()

    def _remove_nearest(self, x: float, y: float):
        if not self._probes:
            return
        dist2 = [((px - x) ** 2 + (py - y) ** 2) for px, py in self._probes]
        self._probes.pop(int(np.argmin(dist2)))

    # -- drawing --------------------------------------------------------------

    def _draw_markers(self):
        for artist in self._label_artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._label_artists = []
        if self._marker_artist is not None:
            try:
                self._marker_artist.remove()
            except (ValueError, NotImplementedError):
                pass
            self._marker_artist = None
        if self._ax is None or not self._probes:
            return
        xs = [p[0] for p in self._probes]
        ys = [p[1] for p in self._probes]
        self._marker_artist = self._ax.scatter(
            xs, ys, s=70, marker="x", color=theme.DANGER, linewidths=2.2, zorder=6,
        )
        for i, (x, y) in enumerate(self._probes):
            self._label_artists.append(self._ax.annotate(
                f"P{i + 1}", (x, y), textcoords="offset points", xytext=(6, 6),
                color=theme.DANGER, fontsize=9, fontweight="bold",
            ))

    def _update_status(self):
        n = len(self._probes)
        if n == 0:
            self.status_label.setText("Click on the image to place a probe.")
        else:
            self.status_label.setText(
                f"{n} probe(s) placed -- left-click to add another, right-click to remove the closest one."
            )
