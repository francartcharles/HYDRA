"""
Shared widget toolkit.

Provides one reusable "Label + mean Entry + dispersion Entry + Normal/
Uniform toggle" group, and one reusable "Label + file-picker button" group,
so every page that needs a Monte-Carlo parameter field or a file-select
row (once per physical parameter, once per bar, once per test type...)
uses the same two widgets below instead of building its own each time.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout,
    QFrame, QRadioButton, QButtonGroup, QGridLayout, QFileDialog, QComboBox,
    QScrollArea, QSizePolicy, QCheckBox, QToolButton, QDialog, QTextBrowser,
)
import numpy as np

from core.mc import MCValue, Distribution


class HelpDialog(QDialog):
    """Non-modal-looking but modal (exec()'d) popup for a "?" help button's
    content -- a scrollable rich-text panel rather than a plain QMessageBox,
    since help content here is meant to be broken into a short lead
    paragraph plus bullet points (see `show_help`'s docstring), not one
    run-on block of text, and can include a "Technical details" section
    that's fairly long for some pages."""

    def __init__(self, title: str, html: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title or "Help")
        self.resize(560, 440)
        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setObjectName("helpBrowser")
        browser.setOpenExternalLinks(False)
        browser.setHtml(html)
        layout.addWidget(browser)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        close_button = QPushButton("Close")
        close_button.setObjectName("primary")
        close_button.clicked.connect(self.accept)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)


def show_help(parent, title: str, html: str) -> None:
    """Opens a `HelpDialog` with the given (already-HTML) content. `html`
    is expected to use <p> for short paragraphs and <ul>/<li> for bullet
    points -- e.g. "<p>What this does, one or two sentences.</p><ul><li>How
    to use it, point one.</li><li>Point two.</li></ul><p><b>Technical
    details:</b> ...</p>" -- so the popup reads as distinct points rather
    than one dense block, per every "?" button in the app."""
    HelpDialog(title, html, parent).exec()


def _make_help_button(parent: QWidget, title: str, html: str) -> QToolButton:
    """Builds the small "?" icon button used throughout the app (card
    headers, plot canvases, time-window pickers) to open a `HelpDialog`
    with that widget's usage/technical notes."""
    button = QToolButton(parent)
    button.setObjectName("helpButton")
    button.setText("?")
    button.setToolTip("Help")
    button.setCursor(Qt.PointingHandCursor)
    button.clicked.connect(lambda: show_help(parent, title, html))
    return button


class Card(QFrame):
    """A titled panel. Replaces the ad-hoc Frame+Label combos used for
    every section of every original popup window."""

    def __init__(self, title: str, subtitle: str = "", parent=None, help_html: str = ""):
        super().__init__(parent)
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 16)
        outer.setSpacing(10)

        if title:
            title_row = QHBoxLayout()
            title_row.setContentsMargins(0, 0, 0, 0)
            t = QLabel(title)
            t.setObjectName("cardTitle")
            title_row.addWidget(t)
            title_row.addStretch(1)
            if help_html:
                title_row.addWidget(_make_help_button(self, title, help_html))
            outer.addLayout(title_row)
        if subtitle:
            s = QLabel(subtitle)
            s.setObjectName("cardSubtitle")
            s.setWordWrap(True)
            outer.addWidget(s)

        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        outer.addLayout(self.body)

    def add(self, widget: QWidget):
        self.body.addWidget(widget)
        return widget

    def add_layout(self, layout):
        self.body.addLayout(layout)
        return layout

    def clear(self):
        """Removes every widget/layout added via add()/add_layout(), for a
        page that rebuilds a card's contents when the thing it configures
        changes shape (e.g. the Viscoelastic Model Calibration page
        swapping its parameter rows for whichever rheological model is
        selected). Nested layouts are cleared recursively first, since Qt
        does not delete a layout's child widgets just because the layout
        itself is discarded."""
        _clear_layout(self.body)


def _clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
            continue
        child_layout = item.layout()
        if child_layout is not None:
            _clear_layout(child_layout)


_NAME_COL_WIDTH = 260
_VALUE_COL_WIDTH = 110


class MCParamTable(QWidget):
    """A column-aligned table of Monte-Carlo parameters: name, mean value,
    dispersion value, Normal/Uniform distribution choice -- with a proper
    header row.

    The header and every data row share one QGridLayout, so a cell in a
    given column is guaranteed to line up with every other cell in that
    column -- separate grids per row would size their columns independently
    and drift out of alignment.

    This single class covers what would otherwise be a large block of
    hand-built widgets duplicated per parameter.
    """

    changed = Signal()

    def __init__(self, mc_values: list[MCValue], parent=None):
        super().__init__(parent)
        self.mc_values = mc_values
        self._mean_edits: dict[str, QLineEdit] = {}
        self._dispersion_edits: dict[str, QLineEdit] = {}

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(16)
        grid.setColumnMinimumWidth(0, _NAME_COL_WIDTH)
        grid.setColumnMinimumWidth(1, _VALUE_COL_WIDTH)
        grid.setColumnMinimumWidth(2, _VALUE_COL_WIDTH)
        grid.setColumnStretch(3, 1)

        for col, text in enumerate(["Parameter", "Mean", "Dispersion", "Distribution"]):
            lbl = QLabel(text)
            lbl.setObjectName("cardSubtitle")
            grid.addWidget(lbl, 0, col)

        for row, mc_value in enumerate(mc_values, start=1):
            label_text = mc_value.name + (f" ({mc_value.unit})" if mc_value.unit else "")
            name_label = QLabel(label_text)
            grid.addWidget(name_label, row, 0)

            mean_edit = QLineEdit(_fmt(mc_value.mean))
            grid.addWidget(mean_edit, row, 1)
            self._mean_edits[id(mc_value)] = mean_edit

            dispersion_edit = QLineEdit(_fmt(mc_value.dispersion))
            grid.addWidget(dispersion_edit, row, 2)
            self._dispersion_edits[id(mc_value)] = dispersion_edit

            dist_box = QHBoxLayout()
            dist_box.setContentsMargins(0, 0, 0, 0)
            dist_box.setSpacing(24)  # the two radio buttons read as touching at the default ~6px spacing
            normal_radio = QRadioButton("Normal")
            uniform_radio = QRadioButton("Uniform")
            group = QButtonGroup(self)
            group.addButton(normal_radio)
            group.addButton(uniform_radio)
            (normal_radio if mc_value.distribution == Distribution.NORMAL else uniform_radio).setChecked(True)
            dist_box.addWidget(normal_radio)
            dist_box.addWidget(uniform_radio)
            dist_box.addStretch(1)
            grid.addLayout(dist_box, row, 3)

            mean_edit.editingFinished.connect(self._make_mean_handler(mc_value, mean_edit))
            dispersion_edit.editingFinished.connect(self._make_dispersion_handler(mc_value, dispersion_edit))
            normal_radio.toggled.connect(self._make_distribution_handler(mc_value, normal_radio))

    def refresh(self):
        """Re-read every mc_value into its widgets (e.g. after loading a
        parameter file that changed values from underneath the widgets)."""
        for mc_value in self.mc_values:
            self._mean_edits[id(mc_value)].setText(_fmt(mc_value.mean))
            self._dispersion_edits[id(mc_value)].setText(_fmt(mc_value.dispersion))

    def _make_mean_handler(self, mc_value: MCValue, edit: QLineEdit):
        def handler():
            try:
                mc_value.mean = float(edit.text())
                self.changed.emit()
            except ValueError:
                edit.setText(_fmt(mc_value.mean))
        return handler

    def _make_dispersion_handler(self, mc_value: MCValue, edit: QLineEdit):
        def handler():
            try:
                mc_value.dispersion = float(edit.text())
                self.changed.emit()
            except ValueError:
                edit.setText(_fmt(mc_value.dispersion))
        return handler

    def _make_distribution_handler(self, mc_value: MCValue, normal_radio: QRadioButton):
        def handler(checked):
            mc_value.distribution = Distribution.NORMAL if checked else Distribution.UNIFORM
            self.changed.emit()
        return handler


class FileSelectRow(QWidget):
    """Inline 'label + path + Browse' row.

    File pickers are inherently OS dialogs (there is no reasonable
    in-window replacement for "choose a file from disk"), but unlike the
    original app, choosing a file never itself opens a separate feature
    window -- it's one row embedded directly in the page you're already on.
    """

    file_selected = Signal(str)

    def __init__(self, label: str, filter_str: str = "Text files (*.txt);;All files (*)", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setMinimumWidth(180)
        layout.addWidget(lbl)
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("No file selected")
        layout.addWidget(self.path_edit, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse(filter_str))
        layout.addWidget(browse)

    def _browse(self, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", filter_str)
        if path:
            self.path_edit.setText(path)
            self.file_selected.emit(path)

    def path(self) -> str:
        return self.path_edit.text()


class TimeSourceRow(QWidget):
    """"Time file (.txt)" row, plus an alternative: check a box to generate
    a uniform time vector from a time step instead of loading one from a
    file. Many raw exports only have a signal/load column with no separate
    time column of their own, or the time column is redundant with a
    documented, constant sample rate -- this avoids having to hand-build
    and save a throwaway time file on disk just to satisfy a "Time file"
    uploader.

    Drop-in alternative to `FileSelectRow` for every "Time file" row in the
    app: connect `file_selected(str)` exactly as before (a `FileSelectRow`
    is embedded and used whenever "Generate..." is unchecked). Connect
    `dtChanged(float)` to react when a positive time step is entered/edited
    while "Generate..." is checked -- the page must supply how many samples
    the vector should have on its own (typically: the length of whatever
    companion signal/load file was already loaded on that page) and build
    it with `build(n)`, which returns `dt * np.arange(n)` from the last
    valid time step (or `None` if none has been entered yet)."""

    file_selected = Signal(str)
    dtChanged = Signal(float)
    modeChanged = Signal(bool)  # True == generate-from-time-step mode

    def __init__(self, label: str = "Time file (.txt)", filter_str: str = "Text files (*.txt);;All files (*)", parent=None):
        super().__init__(parent)
        self._dt = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.file_row = FileSelectRow(label, filter_str)
        self.file_row.file_selected.connect(self.file_selected)
        layout.addWidget(self.file_row)

        self.generate_check = QCheckBox("Generate from a time step instead of loading a file")
        self.generate_check.toggled.connect(self._on_toggle)
        layout.addWidget(self.generate_check)

        self.dt_edit = QLineEdit()
        self.dt_edit.setPlaceholderText("e.g. 2e-7")
        self.dt_edit.editingFinished.connect(self._on_dt_changed)
        self.dt_row = LabeledField("Time step (s)", self.dt_edit)
        self.dt_row.setVisible(False)
        layout.addWidget(self.dt_row)

    def _on_toggle(self, checked):
        self.file_row.setVisible(not checked)
        self.dt_row.setVisible(checked)
        self.modeChanged.emit(checked)
        if checked and self._dt is not None:
            self.dtChanged.emit(self._dt)

    def _on_dt_changed(self):
        text = self.dt_edit.text().strip()
        if not text:
            return
        try:
            dt = float(text)
        except ValueError:
            self.dt_edit.setText("" if self._dt is None else _fmt(self._dt))
            return
        if dt <= 0:
            self.dt_edit.setText("" if self._dt is None else _fmt(self._dt))
            return
        self._dt = dt
        self.dtChanged.emit(dt)

    def is_generate_mode(self) -> bool:
        return self.generate_check.isChecked()

    def dt(self):
        return self._dt

    def set_dt(self, dt) -> None:
        """Restores a previously-entered time step (e.g. from a loaded Setup
        Profile/session) into the field and internal state, without emitting
        `dtChanged` -- the caller decides separately whether/when to
        (re)build the time vector from it."""
        self._dt = dt
        self.dt_edit.setText("" if dt is None else _fmt(dt))

    def set_generate_mode(self, generate: bool) -> None:
        self.generate_check.setChecked(generate)

    def build(self, n: int):
        """Returns `dt * np.arange(n)` from the last valid time step, or
        `None` if no valid time step has been entered yet or `n <= 0`."""
        if self._dt is None or n <= 0:
            return None
        return self._dt * np.arange(n)


class LabeledField(QWidget):
    """Simple single-value label + input row for non-Monte-Carlo settings
    (names, checkboxes-as-combo, counts...)."""

    def __init__(self, label: str, widget: QWidget, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setMinimumWidth(230)
        layout.addWidget(lbl)
        layout.addWidget(widget, 1)
        self.widget = widget


class PageHeader(QWidget):
    def __init__(self, title: str, description: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("pageTitle")
        layout.addWidget(t)
        if description:
            d = QLabel(description)
            d.setObjectName("pageDescription")
            d.setWordWrap(True)
            layout.addWidget(d)


def make_scroll_page(title: str, description: str = "") -> tuple[QScrollArea, QVBoxLayout]:
    """Standard page scaffold: scrollable column of Cards under a title.
    Every page in ui/pages/*.py starts from this instead of re-declaring
    a Toplevel + Canvas + Scrollbar trio (as AutoScrollbar/MaFrame did)."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    inner = QWidget()
    inner.setObjectName("page")
    layout = QVBoxLayout(inner)
    layout.setContentsMargins(24, 20, 24, 24)
    layout.setSpacing(14)
    layout.addWidget(PageHeader(title, description))
    scroll.setWidget(inner)
    return scroll, layout


def _fmt(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"
