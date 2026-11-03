"""
Interactive rectangle/polygon picker(s), for the Image Correlation page.

`RectPicker` is the simple, rectangle-only base: click-and-drag a rectangle
directly on the displayed frame (mirrors span_picker.TimeWindowPicker's
click-and-drag approach for 1D time windows, here applied to a 2D image).
Used as-is for the preprocessing crop region, which only ever needs a
rectangle.

`RoiMeshPicker` is a separate, standalone widget (not a subclass of
`RectPicker` -- its selector can be a rectangle, a polygon, or an oval,
which doesn't fit RectPicker's rectangle-only rectChanged/set_rect/
get_rect interface) for the DIC ROI itself: the same click-and-drag
rectangle, a click-to-place polygon (matplotlib's PolygonSelector --
click each vertex, close by clicking the first vertex again or pressing
Enter), or a click-and-drag oval (matplotlib's EllipseSelector -- same
click-and-drag gesture as the rectangle, but the ROI is the ellipse
inscribed in the dragged bounding box, a better fit for a cylindrical
specimen viewed from the side), plus a preview of the correlation mesh
-- a marker at every point the current subset size / step would produce
-- inside the ROI.

`RoiMeshPicker` also places point PROBES on the same frame: since every
one of its three selectors already consumes plain left-click (dragging a
rectangle/oval, or placing a polygon vertex), a probe is placed with a
RIGHT-click instead -- free across all three shape modes (none of
RectangleSelector/PolygonSelector/EllipseSelector bind right-click to
anything by default) -- and removed with the "Remove last probe" /
"Clear probes" buttons rather than a second click gesture, so there's no
ambiguity between "right-click to add" and "right-click near an existing
one to remove it". Each placed point is later matched (see the Image
Correlation page's `_nearest_mesh_index`) to its nearest correlation mesh
point, so a probe reads whatever computed field is currently selected AT
that specific location, across every processed frame -- letting a user
watch a field's time evolution at a handful of chosen spots (e.g. the
neck vs. a point far from it) instead of only the ROI-wide mean.
Probes are independent of the ROI itself (surviving a shape-mode switch
or a "Clear ROI"), but -- like the ROI -- are reset whenever the
underlying frame's pixel coordinate system changes (`set_frame(...,
keep_region=False)`), since old probe positions would no longer point at
the same physical spot.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from matplotlib.widgets import RectangleSelector, PolygonSelector, EllipseSelector

from core.dic_correlation import (
    ROI, Ellipse, MeshGrid, build_mesh, build_mesh_from_polygon, build_mesh_from_ellipse,
)
from ui.plot_canvas import PlotCanvas
from ui.widgets import _make_help_button
from ui import theme


class RectPicker(QWidget):
    """
    rectChanged(x, y, width, height) fires whenever the rectangle is
    drawn, resized or moved by the user -- pixel coordinates, ints,
    already clamped to the currently-shown frame.
    """

    rectChanged = Signal(int, int, int, int)

    def __init__(self, figsize=(8, 5.5), region_label: str = "region", parent=None,
                 help_text: str = "", help_title: str = ""):
        super().__init__(parent)
        self._region_label = region_label
        self._frame_shape = None  # (height, width) of the currently shown frame
        self._rect: ROI | None = None
        self._ax = None
        self._selector = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        controls = QHBoxLayout()
        self.reset_view_button = QPushButton("Reset view")
        self.reset_view_button.clicked.connect(self.reset_view)
        controls.addWidget(self.reset_view_button)
        self.clear_button = QPushButton(f"Clear {region_label}")
        self.clear_button.clicked.connect(self.clear_rect)
        controls.addWidget(self.clear_button)
        controls.addStretch(1)
        self.status_label = QLabel(f"Click and drag on the image to set the {region_label}.")
        self.status_label.setObjectName("cardSubtitle")
        controls.addWidget(self.status_label)
        if help_text:
            controls.addWidget(_make_help_button(self, help_title, help_text))
        layout.addLayout(controls)

        self.canvas = PlotCanvas(figsize=figsize)
        layout.addWidget(self.canvas)

        self.show_placeholder("Load a video or image sequence to see frames here.")

    def show_placeholder(self, message: str):
        self._ax = None
        self._selector = None
        self.canvas.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True,
                color=theme.TEXT_MUTED, fontsize=10, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.refresh()

    def set_frame(self, frame_uint8: np.ndarray, keep_rect: bool = True):
        """Displays a new frame (2D grayscale or 3D RGB uint8 array,
        already contrast-stretched -- see frame_source.to_display_uint8).

        Rebuilds the axes and its RectangleSelector from scratch every
        call (a matplotlib RectangleSelector isn't meant to survive
        `Axes.clear()`), then redraws the existing rectangle on top if
        `keep_rect` -- letting the user scrub through frames without
        losing the rectangle they already drew."""
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

        self._selector = RectangleSelector(
            self._ax, self._on_select, useblit=True,
            button=[1], minspanx=2, minspany=2,
            interactive=True,
            props=dict(facecolor=theme.ACCENT, edgecolor=theme.ACCENT, alpha=0.2, fill=True),
        )

        if not keep_rect:
            self._rect = None

        if self._rect is not None:
            self._rect = self._rect.clamped(width, height)
            self._apply_selector_extents()
        self._draw_extra()
        self._update_status()

        if self.canvas.toolbar is not None:
            self.canvas.toolbar.update()
        self.canvas.refresh()

    def reset_view(self):
        if self._ax is None or self._frame_shape is None:
            return
        height, width = self._frame_shape
        self._ax.set_xlim(0, width)
        self._ax.set_ylim(height, 0)
        self.canvas.refresh()

    def clear_rect(self):
        self._rect = None
        if self._selector is not None:
            self._selector.extents = (0, 0, 0, 0)
        self._draw_extra()
        self._update_status()
        self.canvas.refresh()
        self.rectChanged.emit(0, 0, 0, 0)

    def set_rect(self, x: int, y: int, width: int, height: int):
        """Programmatically sets the rectangle (e.g. restoring a saved
        session), without emitting `rectChanged` (the caller already
        knows the value it's setting)."""
        if width <= 0 or height <= 0:
            return
        self._rect = ROI(int(x), int(y), int(width), int(height))
        if self._frame_shape is not None:
            frame_height, frame_width = self._frame_shape
            self._rect = self._rect.clamped(frame_width, frame_height)
        self._apply_selector_extents()
        self._draw_extra()
        self._update_status()
        self.canvas.refresh()

    def get_rect(self) -> ROI | None:
        return self._rect

    def _apply_selector_extents(self):
        if self._selector is None or self._rect is None:
            return
        self._selector.extents = (
            self._rect.x, self._rect.x + self._rect.width,
            self._rect.y, self._rect.y + self._rect.height,
        )

    def _on_select(self, eclick, erelease):
        if self._frame_shape is None or eclick.xdata is None or erelease.xdata is None:
            return
        frame_height, frame_width = self._frame_shape
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        rect = ROI(int(round(x0)), int(round(y0)),
                   max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0))))
        self._rect = rect.clamped(frame_width, frame_height)
        self._draw_extra()
        self._update_status()
        self.canvas.refresh()
        self.rectChanged.emit(self._rect.x, self._rect.y, self._rect.width, self._rect.height)

    # -- hooks for subclasses ---------------------------------------------

    def _draw_extra(self):
        """Called whenever the rectangle changes, after it's applied to
        the selector -- overridden by RoiMeshPicker to draw the mesh."""

    def _update_status(self):
        if self._rect is None:
            self.status_label.setText(f"Click and drag on the image to set the {self._region_label}.")
            return
        self.status_label.setText(
            f"{self._region_label.capitalize()}: {self._rect.width} x {self._rect.height} px "
            f"at ({self._rect.x}, {self._rect.y})"
        )


class RoiMeshPicker(QWidget):
    """
    The DIC ROI picker: a rectangle (click-and-drag) or a polygon
    (click each vertex, close by clicking the first vertex again or
    pressing Enter) drawn directly on the displayed frame, plus a preview
    of the correlation mesh -- a marker at every point the current subset
    size / step would produce -- inside it.

    `regionChanged()` fires (no arguments -- call `get_rect()` or
    `get_polygon()`, whichever matches `get_shape_mode()`, to read the
    current value) whenever the ROI is drawn, resized, moved or completed
    by the user, or cleared.

    `probesChanged()` fires (no arguments -- call `get_probes()` to read
    the current list) whenever a probe is added or removed, directly on
    the canvas (right-click to add) or via the buttons below it.

    Standalone rather than a `RectPicker` subclass: a polygon ROI doesn't
    fit RectPicker's rectangle-only rectChanged/set_rect/get_rect
    interface, and swapping between a RectangleSelector and a
    PolygonSelector on the same axes needs its own bookkeeping (`_activate_
    selector`) that a shared base class would only complicate.
    """

    regionChanged = Signal()
    probesChanged = Signal()

    def __init__(self, figsize=(8, 5.5), parent=None, help_text: str = "", help_title: str = ""):
        super().__init__(parent)
        self._shape_mode = "rectangle"  # "rectangle" | "polygon" | "oval"
        self._frame_shape = None  # (height, width) of the currently shown frame
        self._rect: ROI | None = None
        self._polygon: list[tuple[float, float]] | None = None
        self._ellipse: Ellipse | None = None
        self._ax = None
        self._selector = None
        self._mesh_artist = None
        self._subset_size = 21
        self._step = 7
        self._probes: list[tuple[float, float]] = []  # (x, y), preprocessed-frame pixel coords
        self._probe_marker_artist = None
        self._probe_label_artists: list = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        controls = QHBoxLayout()
        self.reset_view_button = QPushButton("Reset view")
        self.reset_view_button.clicked.connect(self.reset_view)
        controls.addWidget(self.reset_view_button)
        self.clear_button = QPushButton("Clear ROI")
        self.clear_button.clicked.connect(self.clear_region)
        controls.addWidget(self.clear_button)
        controls.addStretch(1)
        self.status_label = QLabel("Click and drag on the image to set the ROI.")
        self.status_label.setObjectName("cardSubtitle")
        controls.addWidget(self.status_label)
        if help_text:
            controls.addWidget(_make_help_button(self, help_title, help_text))
        layout.addLayout(controls)

        probe_controls = QHBoxLayout()
        self.remove_last_probe_button = QPushButton("Remove last probe")
        self.remove_last_probe_button.clicked.connect(self.remove_last_probe)
        probe_controls.addWidget(self.remove_last_probe_button)
        self.clear_probes_button = QPushButton("Clear probes")
        self.clear_probes_button.clicked.connect(self.clear_probes)
        probe_controls.addWidget(self.clear_probes_button)
        probe_controls.addStretch(1)
        self.probe_status_label = QLabel("Right-click on the image to place a probe.")
        self.probe_status_label.setObjectName("cardSubtitle")
        probe_controls.addWidget(self.probe_status_label)
        layout.addLayout(probe_controls)

        self.canvas = PlotCanvas(figsize=figsize)
        layout.addWidget(self.canvas)
        # Connected once -- self.canvas.canvas (the FigureCanvasQTAgg) is
        # created once in PlotCanvas.__init__ and never replaced; only its
        # Figure's contents are cleared/rebuilt on every set_frame, so a
        # single persistent connection here (unlike ProbePicker's old
        # per-set_frame mpl_connect, which would have stacked one
        # connection per frame change) correctly survives frame changes.
        # Filtered to button 3 (right-click) so it never interferes with
        # any of the three ROI selectors above, which all bind left-click.
        self.canvas.canvas.mpl_connect("button_press_event", self._on_probe_click)

        self.show_placeholder("Load a video or image sequence to see frames here.")

    # -- shape mode ---------------------------------------------------------

    def set_shape_mode(self, mode: str):
        """Switches the ROI drawing tool between "rectangle", "polygon"
        and "oval". Clears whichever region was set under the previous
        mode (the three aren't interchangeable values) and re-activates
        the matching selector on the current frame, if any. Emits
        `regionChanged` (the ROI value effectively became empty), unless
        there was nothing to clear and the mode already matched."""
        mode = mode if mode in ("polygon", "oval") else "rectangle"
        if mode == self._shape_mode:
            return
        self._shape_mode = mode
        self._rect = None
        self._polygon = None
        self._ellipse = None
        self._activate_selector()
        self._draw_extra()
        self._update_status()
        self.canvas.refresh()
        self.regionChanged.emit()

    def get_shape_mode(self) -> str:
        return self._shape_mode

    # -- placeholder / frame --------------------------------------------

    def show_placeholder(self, message: str):
        self._ax = None
        self._selector = None
        self._mesh_artist = None
        self._probe_marker_artist = None
        self._probe_label_artists = []
        self.canvas.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True,
                color=theme.TEXT_MUTED, fontsize=10, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.refresh()

    def set_frame(self, frame_uint8: np.ndarray, keep_region: bool = True):
        """Displays a new frame (2D grayscale or 3D RGB uint8 array,
        already contrast-stretched -- see frame_source.to_display_uint8).

        Rebuilds the axes and its selector from scratch every call
        (neither RectangleSelector nor PolygonSelector are meant to
        survive `Axes.clear()`), then redraws the existing ROI -- and the
        existing probes -- on top if `keep_region` -- letting the user
        scrub through frames without losing the ROI/probes they already
        placed. `keep_region=False` also clears the probes, not just the
        ROI: both are positions in the preprocessed frame's pixel
        coordinate system, so a caller only ever sets it False when that
        system just changed (a new source, or a crop/downscale change),
        at which point old probe positions would no longer point at the
        same physical spot either."""
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

        if not keep_region:
            self._rect = None
            self._polygon = None
            self._ellipse = None
            self._probes = []
        elif self._rect is not None:
            self._rect = self._rect.clamped(width, height)

        self._activate_selector()
        self._draw_extra()
        self._update_status()
        self._update_probe_status()

        if self.canvas.toolbar is not None:
            self.canvas.toolbar.update()
        self.canvas.refresh()

    def reset_view(self):
        if self._ax is None or self._frame_shape is None:
            return
        height, width = self._frame_shape
        self._ax.set_xlim(0, width)
        self._ax.set_ylim(height, 0)
        self.canvas.refresh()

    def clear_region(self):
        self._rect = None
        self._polygon = None
        self._ellipse = None
        if self._selector is not None:
            if self._shape_mode == "polygon":
                self._selector.clear()
            else:
                self._selector.extents = (0, 0, 0, 0)
        self._draw_extra()
        self._update_status()
        self.canvas.refresh()
        self.regionChanged.emit()

    # -- programmatic get/set --------------------------------------------

    def set_rect(self, x: int, y: int, width: int, height: int):
        """Programmatically sets a rectangular ROI (e.g. restoring a
        saved session), without emitting `regionChanged` (the caller
        already knows the value it's setting). Only takes visual effect
        while in "rectangle" shape mode -- call `set_shape_mode
        ("rectangle")` first if needed."""
        if width <= 0 or height <= 0:
            return
        self._rect = ROI(int(x), int(y), int(width), int(height))
        if self._frame_shape is not None:
            frame_height, frame_width = self._frame_shape
            self._rect = self._rect.clamped(frame_width, frame_height)
        if self._shape_mode == "rectangle":
            self._apply_selector_extents()
            self._draw_extra()
            self._update_status()
            self.canvas.refresh()

    def set_polygon(self, vertices: list):
        """Programmatically sets a polygon ROI (e.g. restoring a saved
        session), without emitting `regionChanged`. Only takes visual
        effect while in "polygon" shape mode -- call `set_shape_mode
        ("polygon")` first if needed."""
        vertices = list(vertices)
        if len(vertices) < 3:
            return
        self._polygon = [(float(v[0]), float(v[1])) for v in vertices]
        if self._shape_mode == "polygon":
            if self._selector is not None:
                self._selector.verts = self._polygon
            self._draw_extra()
            self._update_status()
            self.canvas.refresh()

    def set_ellipse(self, cx: float, cy: float, rx: float, ry: float):
        """Programmatically sets an oval ROI (e.g. restoring a saved
        session), without emitting `regionChanged`. Only takes visual
        effect while in "oval" shape mode -- call `set_shape_mode
        ("oval")` first if needed."""
        if rx <= 0 or ry <= 0:
            return
        self._ellipse = Ellipse(float(cx), float(cy), float(rx), float(ry))
        if self._shape_mode == "oval":
            self._apply_ellipse_selector_extents()
            self._draw_extra()
            self._update_status()
            self.canvas.refresh()

    def get_rect(self) -> ROI | None:
        return self._rect if self._shape_mode == "rectangle" else None

    def get_polygon(self) -> list | None:
        return self._polygon if self._shape_mode == "polygon" else None

    def get_ellipse(self) -> Ellipse | None:
        return self._ellipse if self._shape_mode == "oval" else None

    def set_mesh_params(self, subset_size: int, step: int):
        self._subset_size = subset_size
        self._step = step
        self._draw_extra()
        self._update_status()
        self.canvas.refresh()

    def get_mesh(self) -> MeshGrid | None:
        if self._shape_mode == "rectangle":
            if self._rect is None:
                return None
            return build_mesh(self._rect, self._subset_size, self._step)
        if self._shape_mode == "oval":
            if self._ellipse is None:
                return None
            return build_mesh_from_ellipse(self._ellipse, self._subset_size, self._step)
        if self._polygon is None or len(self._polygon) < 3:
            return None
        return build_mesh_from_polygon(self._polygon, self._subset_size, self._step)

    # -- probes ---------------------------------------------------------

    def get_probes(self) -> list[tuple[float, float]]:
        return list(self._probes)

    def set_probes(self, probes):
        """Programmatically sets the probe list (e.g. restoring a saved
        session), without emitting `probesChanged` (the caller already
        knows the value it's setting)."""
        self._probes = [(float(x), float(y)) for x, y in probes] if probes else []
        self._draw_extra()
        self._update_probe_status()
        self.canvas.refresh()

    def remove_last_probe(self):
        if not self._probes:
            return
        self._probes.pop()
        self._draw_extra()
        self._update_probe_status()
        self.canvas.refresh()
        self.probesChanged.emit()

    def clear_probes(self):
        if not self._probes:
            return
        self._probes = []
        self._draw_extra()
        self._update_probe_status()
        self.canvas.refresh()
        self.probesChanged.emit()

    def _on_probe_click(self, event):
        if self._ax is None or event.inaxes != self._ax or event.button != 3:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._probes.append((float(event.xdata), float(event.ydata)))
        self._draw_extra()
        self._update_probe_status()
        self.canvas.refresh()
        self.probesChanged.emit()

    def _draw_probes(self):
        for artist in self._probe_label_artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._probe_label_artists = []
        if self._probe_marker_artist is not None:
            try:
                self._probe_marker_artist.remove()
            except (ValueError, NotImplementedError):
                pass
            self._probe_marker_artist = None
        if self._ax is None or not self._probes:
            return
        xs = [p[0] for p in self._probes]
        ys = [p[1] for p in self._probes]
        self._probe_marker_artist = self._ax.scatter(
            xs, ys, s=70, marker="x", color=theme.DANGER, linewidths=2.2, zorder=6,
        )
        for i, (x, y) in enumerate(self._probes):
            self._probe_label_artists.append(self._ax.annotate(
                f"P{i + 1}", (x, y), textcoords="offset points", xytext=(6, 6),
                color=theme.DANGER, fontsize=9, fontweight="bold",
            ))

    def _update_probe_status(self):
        n = len(self._probes)
        if n == 0:
            self.probe_status_label.setText("Right-click on the image to place a probe.")
        else:
            self.probe_status_label.setText(
                f"{n} probe(s) placed -- right-click to add another, "
                f"or use the buttons to remove one."
            )

    # -- selector plumbing -------------------------------------------------

    def _activate_selector(self):
        """(Re)creates the matplotlib selector matching `_shape_mode` on
        the current axes -- called whenever the axes is rebuilt (a new
        frame) or the shape mode changes, since neither selector type
        survives either of those on its own."""
        self._selector = None
        if self._ax is None:
            return
        if self._shape_mode == "rectangle":
            self._selector = RectangleSelector(
                self._ax, self._on_rect_select, useblit=True,
                button=[1], minspanx=2, minspany=2,
                interactive=True,
                props=dict(facecolor=theme.ACCENT, edgecolor=theme.ACCENT, alpha=0.2, fill=True),
            )
            if self._rect is not None:
                self._apply_selector_extents()
        elif self._shape_mode == "oval":
            self._selector = EllipseSelector(
                self._ax, self._on_ellipse_select, useblit=True,
                button=[1], minspanx=2, minspany=2,
                interactive=True,
                props=dict(facecolor=theme.ACCENT, edgecolor=theme.ACCENT, alpha=0.2, fill=True),
            )
            if self._ellipse is not None:
                self._apply_ellipse_selector_extents()
        else:
            self._selector = PolygonSelector(
                self._ax, self._on_polygon_select, useblit=True,
                props=dict(color=theme.ACCENT, linewidth=1.5, alpha=0.6),
                handle_props=dict(color=theme.ACCENT),
            )
            if self._polygon is not None:
                # Programmatic vertex restore -- doesn't fire onselect,
                # same convention as RectPicker.set_rect's extents restore.
                self._selector.verts = self._polygon

    def _apply_selector_extents(self):
        if self._selector is None or self._rect is None or self._shape_mode != "rectangle":
            return
        self._selector.extents = (
            self._rect.x, self._rect.x + self._rect.width,
            self._rect.y, self._rect.y + self._rect.height,
        )

    def _apply_ellipse_selector_extents(self):
        if self._selector is None or self._ellipse is None or self._shape_mode != "oval":
            return
        e = self._ellipse
        self._selector.extents = (e.cx - e.rx, e.cx + e.rx, e.cy - e.ry, e.cy + e.ry)

    def _on_rect_select(self, eclick, erelease):
        if self._frame_shape is None or eclick.xdata is None or erelease.xdata is None:
            return
        frame_height, frame_width = self._frame_shape
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        rect = ROI(int(round(x0)), int(round(y0)),
                   max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0))))
        self._rect = rect.clamped(frame_width, frame_height)
        self._draw_extra()
        self._update_status()
        self.canvas.refresh()
        self.regionChanged.emit()

    def _on_ellipse_select(self, eclick, erelease):
        # EllipseSelector's onselect callback matches RectangleSelector's
        # own (eclick/erelease each carry the DRAGGED BOUNDING BOX corner
        # -- the ellipse itself is the one inscribed in that box, center
        # at its middle, semi-axes half its width/height).
        if self._frame_shape is None or eclick.xdata is None or erelease.xdata is None:
            return
        frame_height, frame_width = self._frame_shape
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        bbox = ROI(int(round(x0)), int(round(y0)),
                   max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0)))).clamped(frame_width, frame_height)
        self._ellipse = Ellipse(bbox.x + bbox.width / 2.0, bbox.y + bbox.height / 2.0,
                                 bbox.width / 2.0, bbox.height / 2.0)
        self._draw_extra()
        self._update_status()
        self.canvas.refresh()
        self.regionChanged.emit()

    def _on_polygon_select(self, verts):
        self._polygon = [(float(x), float(y)) for x, y in verts] if len(verts) >= 3 else None
        self._draw_extra()
        self._update_status()
        self.canvas.refresh()
        self.regionChanged.emit()

    # -- mesh preview / status ----------------------------------------------

    def _draw_extra(self):
        if self._mesh_artist is not None:
            try:
                self._mesh_artist.remove()
            except (ValueError, NotImplementedError):
                pass
            self._mesh_artist = None
        if self._ax is None:
            return
        mesh = self.get_mesh()
        if mesh is not None and mesh.n_points:
            xx, yy = mesh.grid_xy()
            if mesh.mask is not None:
                xx = xx[mesh.mask]
                yy = yy[mesh.mask]
            self._mesh_artist = self._ax.scatter(
                np.ravel(xx), np.ravel(yy), s=6, color=theme.SUCCESS, marker="+", linewidths=0.8,
            )
        self._draw_probes()

    def _update_status(self):
        if self._shape_mode == "rectangle":
            if self._rect is None:
                self.status_label.setText("Click and drag on the image to set the ROI.")
                return
            base = f"ROI: {self._rect.width} x {self._rect.height} px at ({self._rect.x}, {self._rect.y})"
        elif self._shape_mode == "oval":
            if self._ellipse is None:
                self.status_label.setText("Click and drag on the image to set the oval ROI.")
                return
            e = self._ellipse
            base = (f"ROI: oval, {e.rx * 2:.0f} x {e.ry * 2:.0f} px, "
                    f"centered at ({e.cx:.0f}, {e.cy:.0f})")
        else:
            if self._polygon is None or len(self._polygon) < 3:
                self.status_label.setText(
                    "Click to place polygon vertices, then click the first vertex again "
                    "(or press Enter) to close the ROI."
                )
                return
            base = f"ROI: polygon, {len(self._polygon)} vertices"

        mesh = self.get_mesh()
        if mesh is None or mesh.n_points == 0:
            self.status_label.setText(f"{base} -- too small for a {self._subset_size}px subset.")
        else:
            self.status_label.setText(
                f"{base}  |  Mesh: {mesh.n_points_x} x {mesh.n_points_y} grid, "
                f"{mesh.n_valid_points} point(s) in the ROI"
            )
