"""A combo box whose items carry a check box -- several can be ticked at once.

Used by the "Crack Tip Detection" card: the drop-down lists every traced
crack, all ticked by default, and the detection runs on every ticked one.

Behaviour
---------
* clicking an item in the drop-down toggles its check box and keeps the list
  open (so several cracks can be ticked in one go); the clicked item also
  becomes the combo's *current* item (the one the rest of the card previews);
* the closed combo shows a short summary ("All cracks (4)", "Crack 1, Crack 3",
  "None ticked");
* `checkedChanged` is emitted whenever a tick changes.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QComboBox, QStyle, QStyledItemDelegate, QStyleOptionComboBox, QStylePainter


class CheckableComboBox(QComboBox):
    checkedChanged = Signal()

    def __init__(self, parent=None, noun: str = "item", all_text: str | None = None):
        super().__init__(parent)
        self._noun = noun
        self._all_text = all_text
        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        # a plain styled delegate draws a real check box in the popup on every
        # platform style (the native combo-popup delegate may not)
        self.view().setItemDelegate(QStyledItemDelegate(self.view()))
        # the popup view's viewport receives the clicks: intercept them so a
        # click toggles the tick instead of closing the popup
        self.view().viewport().installEventFilter(self)
        self.view().installEventFilter(self)

    # -- items --------------------------------------------------------------

    def add_check_item(self, text: str, data=None, checked: bool = True, color: str | None = None) -> None:
        item = QStandardItem(text)
        if color:
            pix = QPixmap(12, 12)
            pix.fill(QColor(color))
            item.setIcon(QIcon(pix))
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
        item.setData(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        item.setData(data, Qt.ItemDataRole.UserRole)
        self._model.appendRow(item)

    def clear_items(self) -> None:
        self._model.clear()

    def checked_states(self) -> list:
        m = self._model
        return [m.item(i).checkState() == Qt.CheckState.Checked for i in range(m.rowCount())]

    def checked_data(self) -> list:
        m = self._model
        return [m.item(i).data(Qt.ItemDataRole.UserRole) for i in range(m.rowCount()) if m.item(i).checkState() == Qt.CheckState.Checked]

    def items_info(self) -> list:
        """[(text, data, checked)] for every item."""
        m = self._model
        return [(m.item(i).text(), m.item(i).data(Qt.ItemDataRole.UserRole),
                 m.item(i).checkState() == Qt.CheckState.Checked) for i in range(m.rowCount())]

    def count_items(self) -> int:
        return self._model.rowCount()

    def set_all_checked(self, checked: bool) -> None:
        m = self._model
        for i in range(m.rowCount()):
            m.item(i).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.update()
        self.checkedChanged.emit()

    def set_checked_at(self, index: int, checked: bool) -> None:
        item = self._model.item(index)
        if item is not None:
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self.update()
            self.checkedChanged.emit()

    # -- summary text on the closed combo -----------------------------------

    def summary_text(self) -> str:
        m = self._model
        n = m.rowCount()
        names = [m.item(i).text() for i in range(n) if m.item(i).checkState() == Qt.CheckState.Checked]
        if n == 0:
            return f"(no {self._noun})"
        if not names:
            return "None ticked"
        if len(names) == n:
            return self._all_text.format(n=n) if self._all_text else f"All {self._noun}s ({n})"
        # item texts look like "Crack 2 (auto) (58 pts)": keep the part before " ("
        short = [t.split(" (")[0] for t in names]
        text = ", ".join(short)
        return text if len(text) <= 40 else f"{len(names)} of {n} {self._noun}s ticked"

    def paintEvent(self, _event):  # noqa: N802 (Qt API)
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = self.summary_text()
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)

    # -- clicks inside the popup ---------------------------------------------

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        if obj is self.view().viewport() and event.type() == QEvent.Type.MouseButtonRelease:
            index = self.view().indexAt(event.position().toPoint() if hasattr(event, "position") else event.pos())
            if index.isValid():
                item = self._model.itemFromIndex(index)
                if item is not None and item.isCheckable():
                    item.setCheckState(Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked)
                    self.setCurrentIndex(index.row())
                    self.update()
                    self.checkedChanged.emit()
                return True                      # swallow: keep the popup open
        if obj is self.view().viewport() and event.type() == QEvent.Type.MouseButtonPress:
            return False
        return super().eventFilter(obj, event)
