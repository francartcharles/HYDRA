"""Large pop-up window for a frame-by-frame field view (image-correlation
results, crack-tip "Verify visually", X-FEM / DIC isocontours, crack tracing).

The page's own canvas is MOVED into the pop-up (and put back, in the same place,
when the pop-up is closed) instead of being redrawn elsewhere: everything the page
draws on it -- and every click handler it has -- keeps working, just in a big
window. The pop-up adds a transport bar that drives the page's own frame slider:

    Play / Pause   Reset (back to the first frame)   Go to frame [ n ] [Go]
    speed (frames per second)   Loop    ---- slider ----   "Frame i / N"

Frame numbers shown here are 1-based ("Frame 40/101", like the plot titles).
Space = Play/Pause, Left/Right arrows = previous/next frame.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox, QVBoxLayout,
)


def _find_owner_layout(widget):
    """(layout, index) of the QBoxLayout that directly holds `widget`, searching nested layouts."""
    parent = widget.parentWidget()
    root = parent.layout() if parent is not None else None
    if root is None:
        return None

    def search(layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() is widget:
                return layout, i
            sub = item.layout()
            if sub is not None:
                found = search(sub)
                if found:
                    return found
        return None

    return search(root)


class FramePopup(QDialog):
    def __init__(self, parent, title: str, canvas, slider, info_labels=(), fps: float = 8.0):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        self.setModal(False)
        self._canvas = canvas
        self._slider = slider
        self._info_labels = list(info_labels)
        self._owner = None            # (layout, index) the canvas came from
        self._placeholder = None
        self._docked_out = False

        root = QVBoxLayout(self)
        self._info = QLabel("")
        self._info.setObjectName("cardSubtitle")
        self._info.setWordWrap(True)
        self._info.setVisible(False)
        root.addWidget(self._info)

        self._canvas_holder = QVBoxLayout()
        root.addLayout(self._canvas_holder, 1)

        bar = QHBoxLayout()
        self.play_button = QPushButton("▶  Play")
        self.play_button.setCheckable(True)
        self.play_button.setMinimumWidth(96)
        self.play_button.toggled.connect(self._on_play_toggled)
        bar.addWidget(self.play_button)
        self.reset_button = QPushButton("⏮  Reset")
        self.reset_button.setToolTip("Stop and go back to the first frame.")
        self.reset_button.clicked.connect(self._on_reset)
        bar.addWidget(self.reset_button)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Go to frame"))
        self.goto_spin = QSpinBox()
        self.goto_spin.setRange(1, 1)
        self.goto_spin.setMinimumWidth(80)
        self.goto_spin.setToolTip("Frame number as in the plot title (1 = first frame).")
        self.goto_spin.editingFinished.connect(self._on_goto)
        bar.addWidget(self.goto_spin)
        self.goto_button = QPushButton("Go")
        self.goto_button.clicked.connect(self._on_goto)
        bar.addWidget(self.goto_button)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Speed"))
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.2, 60.0)
        self.fps_spin.setDecimals(1)
        self.fps_spin.setValue(float(fps))
        self.fps_spin.setSuffix(" frames/s")
        self.fps_spin.setToolTip("Playback speed (a heavy frame may play slower than asked).")
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        bar.addWidget(self.fps_spin)
        self.loop_check = QCheckBox("Loop")
        bar.addWidget(self.loop_check)
        bar.addStretch(1)
        self.frame_label = QLabel("")
        bar.addWidget(self.frame_label)
        root.addLayout(bar)

        self.popup_slider = QSlider(Qt.Orientation.Horizontal)
        self.popup_slider.valueChanged.connect(self._on_popup_slider)
        root.addWidget(self.popup_slider)

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._tick)
        self._sync_timer = QTimer(self)            # follows changes made on the page itself
        self._sync_timer.setInterval(250)
        self._sync_timer.timeout.connect(self._sync_from_page)

        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._key_toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self._key_step(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self._key_step(+1))

        self._dock_out()
        self._sync_from_page()
        self._sync_timer.start()
        screen = self.screen().availableGeometry() if self.screen() is not None else None
        if screen is not None:
            self.resize(int(screen.width() * 0.85), int(screen.height() * 0.88))
        else:
            self.resize(1400, 900)

    # -- moving the page's canvas in and out ------------------------------------

    def _dock_out(self):
        from PySide6.QtWidgets import QBoxLayout
        found = _find_owner_layout(self._canvas)
        if found is not None and isinstance(found[0], QBoxLayout):
            layout, index = found
            self._owner = (layout, index, layout.stretch(index))
            self._placeholder = QLabel("Shown in the pop-up window -- close it to bring the view back here.")
            self._placeholder.setObjectName("cardSubtitle")
            self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._placeholder.setMinimumHeight(60)
            layout.removeWidget(self._canvas)
            layout.insertWidget(index, self._placeholder, self._owner[2])
        self._canvas_holder.addWidget(self._canvas, 1)
        self._canvas.show()
        self._docked_out = True

    def _dock_back(self):
        if not self._docked_out:
            return
        self._docked_out = False
        self._canvas_holder.removeWidget(self._canvas)
        if self._owner is not None:
            layout, index, stretch = self._owner
            if self._placeholder is not None:
                layout.removeWidget(self._placeholder)
                self._placeholder.deleteLater()
                self._placeholder = None
            layout.insertWidget(index, self._canvas, stretch)
            self._canvas.show()
        self._canvas.refresh() if hasattr(self._canvas, "refresh") else None

    def closeEvent(self, event):  # noqa: N802 (Qt API)
        self._play_timer.stop()
        self._sync_timer.stop()
        self._dock_back()
        super().closeEvent(event)

    def done(self, result):  # Esc / close
        self._play_timer.stop()
        self._sync_timer.stop()
        self._dock_back()
        super().done(result)

    # -- transport --------------------------------------------------------------

    def _range(self):
        return int(self._slider.minimum()), int(self._slider.maximum())

    def _set_page_frame(self, value: int):
        lo, hi = self._range()
        value = max(lo, min(hi, int(value)))
        if value != self._slider.value():
            self._slider.setValue(value)          # the page's own handler redraws
        self._sync_from_page()

    def _typing(self) -> bool:
        return self.goto_spin.hasFocus() or self.fps_spin.hasFocus()

    def _key_toggle_play(self):
        if not self._typing() and self.play_button.isEnabled():
            self.play_button.toggle()

    def _key_step(self, delta: int):
        if not self._typing():
            self._step(delta)

    def _step(self, delta: int):
        self._set_page_frame(self._slider.value() + delta)

    def _on_play_toggled(self, on: bool):
        self.play_button.setText("⏸  Pause" if on else "▶  Play")
        if on:
            lo, hi = self._range()
            if self._slider.value() >= hi and hi > lo:
                self._set_page_frame(lo)           # restart from the beginning when at the end
            self._play_timer.start(max(15, int(1000.0 / self.fps_spin.value())))
        else:
            self._play_timer.stop()

    def _on_fps_changed(self, _value):
        if self._play_timer.isActive():
            self._play_timer.start(max(15, int(1000.0 / self.fps_spin.value())))

    def _tick(self):
        lo, hi = self._range()
        v = self._slider.value()
        if v >= hi:
            if self.loop_check.isChecked() and hi > lo:
                self._set_page_frame(lo)
            else:
                self.play_button.setChecked(False)
            return
        self._set_page_frame(v + 1)

    def _on_reset(self):
        self.play_button.setChecked(False)
        self._set_page_frame(self._range()[0])

    def _on_goto(self):
        lo, _hi = self._range()
        self._set_page_frame(lo + self.goto_spin.value() - 1)

    def _on_popup_slider(self, value: int):
        self._set_page_frame(value)

    # -- follow the page --------------------------------------------------------

    def _sync_from_page(self):
        lo, hi = self._range()
        n = hi - lo + 1
        enabled = bool(self._slider.isEnabled()) and hi > lo
        for w in (self.play_button, self.reset_button, self.goto_spin, self.goto_button, self.popup_slider):
            w.setEnabled(enabled)
        v = int(self._slider.value())
        self.popup_slider.blockSignals(True)
        self.popup_slider.setRange(lo, hi)
        self.popup_slider.setValue(v)
        self.popup_slider.blockSignals(False)
        self.goto_spin.blockSignals(True)
        self.goto_spin.setRange(1, max(1, n))
        if not self.goto_spin.hasFocus():
            self.goto_spin.setValue(v - lo + 1)
        self.goto_spin.blockSignals(False)
        self.frame_label.setText(f"Frame {v - lo + 1} / {max(1, n)}")
        texts = [lb.text() for lb in self._info_labels if lb is not None and lb.text()]
        self._info.setText("   |   ".join(texts))
        self._info.setVisible(bool(texts))
        if not enabled and self.play_button.isChecked():
            self.play_button.setChecked(False)


def make_popout_button(owner, title: str, canvas, slider, info_labels=(), text: str = "\u2922 Pop out"):
    """A small button that opens the large pop-up for `canvas` driven by `slider`."""
    button = QPushButton(text)
    button.setToolTip(
        "Open this view in a large window with Play / Pause / Reset / Go to frame and a slider "
        "(Space = play/pause, Left/Right = previous/next frame). Close the window to bring it back here."
    )
    button.clicked.connect(lambda: open_frame_popup(owner, title, title, canvas, slider, info_labels))
    return button


def open_frame_popup(owner, key: str, title: str, canvas, slider, info_labels=()):
    """Opens (or raises, if already open) the pop-up for `canvas`; `owner` is the page, which keeps
    the open pop-ups in `owner._frame_popups` so a view cannot be popped out twice."""
    from PySide6.QtWidgets import QBoxLayout
    found = _find_owner_layout(canvas)
    if found is None or not isinstance(found[0], QBoxLayout):
        return None                                   # the view is already in a pop-up / cannot be moved
    popups = owner.__dict__.setdefault("_frame_popups", {})
    existing = popups.get(key)
    if existing is not None and existing.isVisible():
        existing.raise_()
        existing.activateWindow()
        return existing
    dlg = FramePopup(owner, title, canvas, slider, info_labels)
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    dlg.destroyed.connect(lambda *_: popups.pop(key, None))
    popups[key] = dlg
    dlg.show()
    return dlg
