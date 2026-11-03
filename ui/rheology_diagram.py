"""
A row of big, clickable rheological-model buttons for the Viscoelastic
Model Calibration page: each draws the spring/dashpot schematic of one
model in core/rheology_models.py, with its name underneath. Exactly one is
checked at a time (they share a QButtonGroup set up by the page), and the
checked one is highlighted -- it is the model actually used by the
Bayesian calibration and "Test parameters" modes.

Drawing lives here (not in core/rheology_models.py) to keep that module
free of any Qt dependency; `MODEL_DRAW_FUNCS` below is the only place that
maps a model key to how it's drawn.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QAbstractButton, QSizePolicy

from core.rheology_models import (
    MODEL_BURGERS, MODEL_GENERALIZED_MAXWELL_2, MODEL_KELVIN_VOIGT, MODEL_SLS,
)
from ui import theme

_ORANGE = "#ff9f43"
_PURPLE = "#c17aff"


def _spring_points(x0: float, x1: float, y: float, n_coils: int = 6, amplitude: float = 9.0) -> list[QPointF]:
    """Zig-zag "spring" path from (x0, y) to (x1, y), with straight lead-in/
    lead-out stubs so it reads clearly as connecting to whatever it's wired
    to on either side."""
    stub = (x1 - x0) * 0.14
    body0, body1 = x0 + stub, x1 - stub
    n_segments = n_coils * 2
    points = [QPointF(x0, y), QPointF(body0, y)]
    for i in range(1, n_segments):
        frac = i / n_segments
        px = body0 + frac * (body1 - body0)
        py = y + (amplitude if i % 2 else -amplitude)
        points.append(QPointF(px, py))
    points.append(QPointF(body1, y))
    points.append(QPointF(x1, y))
    return points


def _draw_spring(painter: QPainter, x0: float, x1: float, y: float, label: str, color: QColor, label_above: bool = True):
    pts = _spring_points(x0, x1, y)
    path = QPainterPath(pts[0])
    for p in pts[1:]:
        path.lineTo(p)
    painter.setPen(QPen(color, 2.0))
    painter.drawPath(path)
    _draw_element_label(painter, label, (x0 + x1) / 2, y, above=label_above)


def _draw_dashpot(painter: QPainter, x0: float, x1: float, y: float, label: str, color: QColor, label_above: bool = False):
    """Piston-in-cylinder dashpot symbol: a rod from x0 to the piston head,
    the piston head, then an open-ended cylinder (top/bottom/right walls
    only) from just past the piston to x1, plus the closing stub to x1."""
    height = 11.0
    rod_end = x0 + (x1 - x0) * 0.38
    cyl_start = x0 + (x1 - x0) * 0.52
    stub = (x1 - x0) * 0.06

    pen = QPen(color, 2.0)
    painter.setPen(pen)
    painter.drawLine(QPointF(x0, y), QPointF(rod_end, y))
    painter.drawLine(QPointF(rod_end, y - height * 0.9), QPointF(rod_end, y + height * 0.9))
    painter.drawLine(QPointF(cyl_start, y - height), QPointF(x1 - stub, y - height))
    painter.drawLine(QPointF(cyl_start, y + height), QPointF(x1 - stub, y + height))
    painter.drawLine(QPointF(x1 - stub, y - height), QPointF(x1 - stub, y + height))
    painter.drawLine(QPointF(x1 - stub, y), QPointF(x1, y))
    _draw_element_label(painter, label, (x0 + x1) / 2, y, above=label_above)


def _draw_element_label(painter: QPainter, text: str, cx: float, y: float, above: bool):
    font = QFont(painter.font())
    font.setPointSize(8)
    font.setItalic(True)
    painter.setFont(font)
    painter.setPen(QColor(theme.TEXT_MUTED))
    rect = QRectF(cx - 34, y - 24 if above else y + 8, 68, 15)
    painter.drawText(rect, Qt.AlignHCenter | Qt.AlignVCenter, text)


def _draw_end_plates(painter: QPainter, x_left: float, x_right: float, y_top: float, y_bot: float):
    pen = QPen(QColor(theme.TEXT), 2.2)
    painter.setPen(pen)
    painter.drawLine(QPointF(x_left, y_top - 18), QPointF(x_left, y_bot + 18))
    painter.drawLine(QPointF(x_right, y_top - 18), QPointF(x_right, y_bot + 18))


def _draw_parallel_branches(painter: QPainter, rect: QRectF, branches: list):
    """Draws `branches` (each a callable(painter, x0, x1, y)) as evenly
    spaced horizontal lanes between two rigid end plates -- the shared
    layout behind every "X in parallel with Y [in parallel with Z]" model
    here."""
    plate_w = 8.0
    x_left = rect.left() + plate_w
    x_right = rect.right() - plate_w
    n = len(branches)
    margin = rect.height() * (0.30 if n <= 2 else 0.16)
    y_top = rect.top() + margin
    y_bot = rect.bottom() - margin
    ys = [y_top + i * (y_bot - y_top) / (n - 1) for i in range(n)] if n > 1 else [(y_top + y_bot) / 2]

    _draw_end_plates(painter, x_left, x_right, ys[0], ys[-1])

    branch_pen = QPen(QColor(theme.TEXT), 1.5)
    for y in ys:
        painter.setPen(branch_pen)
        painter.drawLine(QPointF(rect.left(), y), QPointF(x_left, y))
        painter.drawLine(QPointF(x_right, y), QPointF(rect.right(), y))
    for branch, y in zip(branches, ys):
        branch(painter, x_left, x_right, y)


def _maxwell_arm(label_e: str, label_n: str, color: QColor):
    """A spring-then-dashpot branch (a Maxwell arm), for use as one lane of
    `_draw_parallel_branches`."""
    def branch(painter, x0, x1, y):
        x_mid = x0 + (x1 - x0) * 0.42
        _draw_spring(painter, x0, x_mid, y, label_e, color, label_above=True)
        _draw_dashpot(painter, x_mid, x1, y, label_n, color, label_above=False)
    return branch


def draw_kelvin_voigt(painter: QPainter, rect: QRectF):
    accent = QColor(theme.ACCENT)
    orange = QColor(_ORANGE)
    _draw_parallel_branches(painter, rect, [
        lambda p, x0, x1, y: _draw_spring(p, x0, x1, y, "E", accent),
        lambda p, x0, x1, y: _draw_dashpot(p, x0, x1, y, "η", orange),
    ])


def draw_sls(painter: QPainter, rect: QRectF):
    """Spring E1 in series with a Kelvin-Voigt unit (E2 parallel n2) --
    the "3-parameter Kelvin" topology matching the formula computed by
    core/rheology_models.py's `_sls_modulus`."""
    accent = QColor(theme.ACCENT)
    orange = QColor(_ORANGE)
    plate_w = 8.0
    x_left = rect.left() + plate_w
    x_right = rect.right() - plate_w
    y = rect.top() + rect.height() * 0.5
    x_mid = x_left + (x_right - x_left) * 0.38

    _draw_end_plates(painter, x_left, x_right, y, y)
    painter.setPen(QPen(QColor(theme.TEXT), 1.5))
    _draw_spring(painter, x_left, x_mid, y, "E1", accent, label_above=True)

    kv_rect = QRectF(x_mid, rect.top(), rect.right() - x_mid, rect.height())
    _draw_parallel_branches(painter, kv_rect, [
        lambda p, x0, x1, yy: _draw_spring(p, x0, x1, yy, "E2", orange),
        lambda p, x0, x1, yy: _draw_dashpot(p, x0, x1, yy, "n2", orange),
    ])


def draw_generalized_maxwell_2(painter: QPainter, rect: QRectF):
    accent = QColor(theme.ACCENT)
    orange = QColor(_ORANGE)
    purple = QColor(_PURPLE)
    _draw_parallel_branches(painter, rect, [
        lambda p, x0, x1, y: _draw_spring(p, x0, x1, y, "E∞", accent),
        _maxwell_arm("E1", "n1", orange),
        _maxwell_arm("E2", "n2", purple),
    ])


def draw_burgers(painter: QPainter, rect: QRectF):
    accent = QColor(theme.ACCENT)
    orange = QColor(_ORANGE)
    plate_w = 8.0
    x_left = rect.left() + plate_w
    x_right = rect.right() - plate_w
    y = rect.top() + rect.height() * 0.5
    x_mid = x_left + (x_right - x_left) * 0.46

    _draw_end_plates(painter, x_left, x_right, y, y)

    branch_pen = QPen(QColor(theme.TEXT), 1.5)
    painter.setPen(branch_pen)
    x_e1_end = x_left + (x_mid - x_left) * 0.5
    _draw_spring(painter, x_left, x_e1_end, y, "E1", accent, label_above=True)
    _draw_dashpot(painter, x_e1_end, x_mid, y, "n1", accent, label_above=False)

    kv_rect = QRectF(x_mid, rect.top(), rect.right() - x_mid, rect.height())
    _draw_parallel_branches(painter, kv_rect, [
        lambda p, x0, x1, yy: _draw_spring(p, x0, x1, yy, "E2", orange),
        lambda p, x0, x1, yy: _draw_dashpot(p, x0, x1, yy, "n2", orange),
    ])


MODEL_DRAW_FUNCS = {
    MODEL_KELVIN_VOIGT: draw_kelvin_voigt,
    MODEL_SLS: draw_sls,
    MODEL_GENERALIZED_MAXWELL_2: draw_generalized_maxwell_2,
    MODEL_BURGERS: draw_burgers,
}


class RheologyModelButton(QAbstractButton):
    """A big, checkable button: the model's schematic, its name underneath.
    Highlighted (accent border/fill) while checked. Intended to be put in a
    QButtonGroup with `setExclusive(True)` alongside the app's other
    rheological-model buttons so exactly one is ever active."""

    def __init__(self, model_key: str, model_name: str, short_formula: str, parent=None):
        super().__init__(parent)
        self.model_key = model_key
        self.model_name = model_name
        self.short_formula = short_formula
        self._draw_fn = MODEL_DRAW_FUNCS[model_key]
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(210, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def sizeHint(self):
        return QSize(230, 210)

    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)

        selected = self.isChecked()
        hovered = self.underMouse()
        border_color = theme.ACCENT if (selected or hovered) else theme.BORDER
        border_width = 2.4 if selected else 1.4
        painter.setPen(QPen(QColor(border_color), border_width))
        painter.setBrush(QColor(theme.BG_CARD if selected else theme.BG))
        painter.drawRoundedRect(rect, 10, 10)

        diagram_rect = QRectF(rect.left() + 18, rect.top() + 12, rect.width() - 36, rect.height() - 58)
        self._draw_fn(painter, diagram_rect)

        name_font = QFont(painter.font())
        name_font.setPointSize(10)
        name_font.setBold(True)
        painter.setFont(name_font)
        painter.setPen(QColor(theme.ACCENT if selected else theme.TEXT))
        name_rect = QRectF(rect.left() + 4, rect.bottom() - 40, rect.width() - 8, 28)
        painter.drawText(name_rect, Qt.AlignHCenter | Qt.AlignVCenter | Qt.TextWordWrap, self.model_name)

        formula_font = QFont(painter.font())
        formula_font.setPointSize(8)
        formula_font.setBold(False)
        painter.setFont(formula_font)
        painter.setPen(QColor(theme.TEXT_MUTED))
        formula_rect = QRectF(rect.left(), rect.bottom() - 16, rect.width(), 14)
        painter.drawText(formula_rect, Qt.AlignHCenter | Qt.AlignVCenter, self.short_formula)
