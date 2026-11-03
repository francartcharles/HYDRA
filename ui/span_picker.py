"""
Interactive time-window picker.

Replaces manually typing pulse-window start/end times (xd1/xd2, or
xd1..xd6 for the viscoelastic two-bar dispersion correction) with
click-and-drag selection directly on the raw signal plot -- matching how
someone would naturally do this by eye.

One or more named "windows" (e.g. Incident / Reflected / Transmitted) can
be defined; a combo box selects which window the next drag sets, and each
window is drawn as a distinct shaded, labeled span so all of them stay
visible at once for comparison.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton,
)
from matplotlib.widgets import SpanSelector

from ui.plot_canvas import PlotCanvas
from ui.widgets import _make_help_button
from ui import theme


class TimeWindowPicker(QWidget):
    """
    windows: list of (key, label, color) tuples, e.g.
        [("xd", "Pulse window", "#5b8cff")]
    or, for the viscoelastic 2-bar case:
        [("incident", "Incident", "#5b8cff"),
         ("reflected", "Reflected", "#ff9f43"),
         ("transmitted", "Transmitted", "#4cd97b")]

    windowChanged(key, start, end) fires whenever a window is set/moved.
    """

    windowChanged = Signal(str, float, float)

    def __init__(self, windows: list[tuple[str, str, str]], figsize=(9, 3.2), parent=None,
                 help_text: str = "", help_title: str = ""):
        super().__init__(parent)
        self._window_defs = {key: (label, color) for key, label, color in windows}
        self._values: dict[str, tuple[float, float] | None] = {key: None for key, _, _ in windows}
        self._spans = {}  # key -> matplotlib axvspan artist
        self._time = None
        self._series: list[tuple[str, "np.ndarray", str]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Setting window:"))
        self.window_combo = QComboBox()
        for key, label, color in windows:
            self.window_combo.addItem(label, userData=key)
        controls.addWidget(self.window_combo)
        self.reset_view_button = QPushButton("Reset view")
        self.reset_view_button.clicked.connect(self.reset_view)
        controls.addWidget(self.reset_view_button)
        controls.addStretch(1)
        self.status_label = QLabel("Click and drag on the plot to set the selected window.")
        self.status_label.setObjectName("cardSubtitle")
        controls.addWidget(self.status_label)
        if help_text:
            controls.addWidget(_make_help_button(self, help_title, help_text))
        layout.addLayout(controls)

        self.canvas = PlotCanvas(figsize=figsize)
        layout.addWidget(self.canvas)

        self._ax = None
        self._ax2 = None
        self._selector = None
        self.show_placeholder("Load a signal to see it here.")

    def set_signal(self, time, series: list[tuple[str, "np.ndarray", str]], secondary_series=None,
                   primary_ylabel: str = "amplitude", secondary_ylabel: str = "amplitude", xlabel: str = "time (s)"):
        """time: 1D array shared by all series.
        series: list of (label, values, color), plotted on the primary (left) Y axis.
        secondary_series: optional list of (label, values, color), plotted on a
        second, independent (right) Y axis via `twinx()` -- for overlaying two
        signals with very different scales (e.g. a bar gage signal and a
        velocity profile) so both can actually be seen at once."""
        import numpy as np  # local import: only needed when actually plotting

        self._time = np.asarray(time)
        self._series = series
        self.canvas.clear()
        self._ax = self.canvas.figure.add_subplot(111)
        for label, values, color in series:
            n = min(len(self._time), len(values))
            self._ax.plot(self._time[:n], values[:n], label=label, color=color, linewidth=1)
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(primary_ylabel)

        self._ax2 = None
        if secondary_series:
            self._ax2 = self._ax.twinx()
            for label, values, color in secondary_series:
                n = min(len(self._time), len(values))
                self._ax2.plot(self._time[:n], values[:n], label=label, color=color, linewidth=1, linestyle="--")
            self._ax2.set_ylabel(secondary_ylabel)

        handles1, labels1 = self._ax.get_legend_handles_labels()
        if self._ax2 is not None:
            handles2, labels2 = self._ax2.get_legend_handles_labels()
            handles1, labels1 = handles1 + handles2, labels1 + labels2
        if len(handles1) > 1:
            self._ax.legend(handles1, labels1, fontsize=8)

        self._spans = {}
        for key, value in self._values.items():
            if value is not None:
                self._draw_span(key, *value)

        self._selector = SpanSelector(
            self._ax, self._on_select, "horizontal", useblit=True,
            props=dict(alpha=0.25, facecolor=theme.ACCENT),
            interactive=True, drag_from_anywhere=True,
        )
        # Rebuilding the axes from scratch (clear() + add_subplot()) leaves
        # the toolbar's own "Home" button pointing at the *previous*
        # (possibly empty/placeholder) view, since matplotlib's navigation
        # history isn't told about the new axes automatically -- reset it
        # here so "Home" also correctly returns to the full new signal.
        if self.canvas.toolbar is not None:
            self.canvas.toolbar.update()
        self.canvas.refresh()

    def reset_view(self):
        """Rescale the axes to show the full currently-plotted signal(s) --
        a guaranteed one-click way back to "see everything", independent
        of the toolbar's own zoom/pan history."""
        if self._ax is None:
            return
        self._ax.relim()
        self._ax.autoscale()
        if self._ax2 is not None:
            self._ax2.relim()
            self._ax2.autoscale()
        self.canvas.refresh()

    def show_placeholder(self, message: str):
        """Draw a clearly-visible 'nothing to show yet, here's why' message
        instead of leaving a blank canvas that's the same color as its card
        (which looks broken, not just empty)."""
        self._ax = None
        self._ax2 = None
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

    def _on_select(self, start: float, end: float):
        if start == end:
            return
        key = self.window_combo.currentData()
        lo, hi = (start, end) if start <= end else (end, start)
        self._values[key] = (lo, hi)
        self._draw_span(key, lo, hi)
        label, _ = self._window_defs[key]
        self.status_label.setText(f"{label}: {lo:.6g} s -> {hi:.6g} s")
        self.canvas.refresh()
        self.windowChanged.emit(key, lo, hi)

    def _draw_span(self, key: str, start: float, end: float):
        if self._ax is None:
            return
        if key in self._spans and self._spans[key] is not None:
            try:
                self._spans[key].remove()
            except (ValueError, NotImplementedError):
                pass
        label, color = self._window_defs[key]
        span = self._ax.axvspan(start, end, color=color, alpha=0.18, label=f"{label} window")
        self._spans[key] = span

    def get_window(self, key: str) -> tuple[float, float] | None:
        return self._values.get(key)

    def window_keys(self) -> set[str]:
        """The set of window keys this picker was actually built with (its
        `windows` constructor argument) -- lets a caller restoring saved
        window values (e.g. from `project.corrections_windows`, which is a
        single flat dict shared across every Correction type/setup
        combination) tell which of those saved keys are even meaningful
        for the CURRENT picker, instead of blindly trying to restore all
        of them (see `_draw_span`, which looks a key up in
        `self._window_defs` and would raise KeyError for one this picker
        doesn't define)."""
        return set(self._window_defs.keys())

    def set_window(self, key: str, start: float, end: float):
        """Programmatically set a window (e.g. restoring a saved value)."""
        self._values[key] = (start, end)
        if self._ax is not None:
            self._draw_span(key, start, end)
            self.canvas.refresh()
