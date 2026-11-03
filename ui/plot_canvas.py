"""
Embedded matplotlib canvas.

Every processing/correction routine in the original app ended with a
blocking `plt.show()` in its own OS window. Here, the same matplotlib
figures are drawn into a canvas that lives directly inside the page, so
results appear inline and the app is never waiting on a separate window
the user has to close. Includes a compact zoom/pan toolbar and a
live cursor-position readout, since inspecting a signal by eye is most of
the point of plotting it.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QLabel

from ui import theme
from ui.widgets import _make_help_button

plt.rcParams.update({
    "figure.facecolor": theme.BG_CARD,
    "axes.facecolor": theme.BG_CARD,
    "axes.edgecolor": theme.BORDER,
    "axes.labelcolor": theme.TEXT,
    "xtick.color": theme.TEXT_MUTED,
    "ytick.color": theme.TEXT_MUTED,
    "text.color": theme.TEXT,
    "legend.facecolor": theme.BG_CARD,
    "legend.edgecolor": theme.BORDER,
    "grid.color": theme.BORDER,
    "figure.autolayout": True,
})


class PlotCanvas(QWidget):
    """A resizable inline plot area with zoom/pan and a cursor-position
    readout. Call `figure` to get a matplotlib Figure to draw on, then
    `refresh()`."""

    def __init__(self, figsize=(8, 4), parent=None, show_toolbar: bool = True,
                 help_text: str = "", help_title: str = ""):
        # help_text: HTML (<p>/<ul>/<li>...), shown in a scrollable dialog
        # (see ui/widgets.py's HelpDialog/show_help) by a "?" button added
        # at the top-right of this canvas's control row when non-empty.
        super().__init__(parent)
        self.figure = plt.figure(figsize=figsize)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self._value_lookup = None  # optional callable(xdata, ydata) -> str | None, see set_value_lookup
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        if show_toolbar:
            bar_row = QHBoxLayout()
            bar_row.setContentsMargins(0, 0, 0, 0)
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.toolbar.setIconSize(self.toolbar.iconSize() * 0.7)
            bar_row.addWidget(self.toolbar)
            bar_row.addStretch(1)
            self.cursor_label = QLabel("")
            self.cursor_label.setObjectName("cardSubtitle")
            bar_row.addWidget(self.cursor_label)
            if help_text:
                bar_row.addWidget(_make_help_button(self, help_title, help_text))
            layout.addLayout(bar_row, 0)
        else:
            self.toolbar = None
            self.cursor_label = None
            if help_text:
                help_row = QHBoxLayout()
                help_row.setContentsMargins(0, 0, 0, 0)
                help_row.addStretch(1)
                help_row.addWidget(_make_help_button(self, help_title, help_text))
                layout.addLayout(help_row, 0)

        layout.addWidget(self.canvas, 1)     # the plot takes ALL spare height; the toolbar row keeps its own
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def set_value_lookup(self, fn):
        """Sets (or, with None, clears) a callback invoked on every mouse
        move over the canvas with the cursor's (xdata, ydata) in DATA
        coordinates; its return value -- a short string, or None/"" for
        nothing extra -- is appended to the existing "x = ... y = ..."
        cursor readout. For a page whose plot is a field over space (an
        isocontour, a scatter of tracked points, ...), this is how a
        viewer reads off the actual value under the cursor rather than
        just its position -- e.g. the Image Correlation page's Results
        canvas uses it to show the field value at the nearest tracked
        point. Called on every mouse-move, so the callback should be
        cheap (look up/interpolate against data already computed for the
        current view, not redo a whole computation from scratch)."""
        self._value_lookup = fn

    def _on_mouse_move(self, event):
        if self.cursor_label is None:
            return
        if event.inaxes is not None and event.xdata is not None and event.ydata is not None:
            text = f"x = {event.xdata:.6g}   y = {event.ydata:.6g}"
            if self._value_lookup is not None:
                extra = self._value_lookup(event.xdata, event.ydata)
                if extra:
                    text += f"   {extra}"
            self.cursor_label.setText(text)
        else:
            self.cursor_label.setText("")

    def clear(self):
        self.figure.clear()

    def refresh(self):
        self.canvas.draw_idle()

    def close_figure(self):
        """Releases this canvas's Figure from pyplot's global figure
        registry. `__init__` creates the figure through the pyplot
        interface (`plt.figure(...)`) so the toolbar/rcParams machinery
        above stays in one place -- but pyplot then keeps every such
        figure alive until it's explicitly closed, even after the widget
        holding it is deleted. Every page in ui/pages/*.py creates its
        PlotCanvas instances once and keeps them for the page's lifetime,
        so this never needs calling there; only code that creates and
        discards PlotCanvas instances *during* a page's lifetime (e.g. a
        dynamic per-channel plot list) must call this before dropping its
        reference, or pyplot accumulates one leaked figure per discard
        (matplotlib's own "more than 20 figures have been opened" warning)."""
        plt.close(self.figure)
