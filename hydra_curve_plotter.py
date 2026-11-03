#!/usr/bin/env python3
"""
HYDRA Curve Plotter (PySide6 edition)
======================================

Standalone tool to overlay curves from data files on a single matplotlib
figure. Two kinds of source files are understood:

  - shpb_app's own "Results & Export" exports (.csv or .txt -- "Export as
    .csv...", "Export as .txt...", "Export full test").

  - Any other table you already have: an Excel workbook (.xlsx/.xlsm/
    .xls, with a header row and numeric columns on its first sheet), or a
    plain text table (.txt) with a header line and tab- or
    multi-space-separated numeric rows -- the shape you get by copying a
    range straight out of Excel and pasting it as text. Either kind may
    use '.' OR ',' as the decimal separator (a French-locale spreadsheet
    normally displays/saves numbers as "16,0517"); this is handled
    automatically and needs no setting.

Covers both ways of comparing results:

  1. The SAME quantity across SEVERAL TESTS (e.g. "Pressure vs Volumic
     strain" for a dozen different specimens) -- load every test's CSV,
     then "Quick add: same quantity across selected tests...".

  2. SEVERAL QUANTITIES from the SAME TEST (e.g. axial stress, stress
     deviator and pressure, from gages and from laser, all against axial
     strain for one specimen) -- load that one CSV, then "Quick add:
     multiple quantities from one test...".

Either quick-add just populates the "Curves" list below the usual way;
you can freely add/edit/remove/reorder/recolor individual curves (or mix
both styles -- several quantities from several tests at once) from there.
Each curve also has its own, independent line style (solid/dashed/
dotted/dash-dot) and line width, and its own uncertainty-band checkbox --
all set from the "Add curve.../Edit..." dialog and shown at a glance in
the Curves table's Style/Width/Band/Color columns. Each curve can also be
individually CROPPED: select it in the list and use the "3. Crop selected
curve" panel's double-handle slider to drop points from its start and/or
end, so only the zone you actually want (e.g. the loaded portion of a
test, excluding an elastic unload tail or leading noise) ends up on the
plot.

FORMAT TEMPLATES: "Save template..." (Export / session card, on both tabs)
stores every formatting parameter -- legend/grid/band/font settings, data
transform, axis limits, texts, and each curve's colour, line style, width,
band, shown/hidden flag and legend label -- in a small .json file. "Load template..."
applies it to the curves ALREADY loaded: you choose which parts to apply
(texts, axis limits and legend labels are off by default, since they
describe one dataset); curve styles and labels are matched by list order.

The window opens maximized (filling the screen) by default and has two
tabs:

  - "Plot": everything described above, plus per-curve Show/hide
    checkboxes and an optional display-only DATA TRANSFORM per axis:
    invert the sign, start every curve at 0 (subtract its first value),
    multiply by a factor, and/or NORMALISE (divide by a reference value you
    type for each axis, e.g. strain / yield strain) -- the source data are
    never modified.

  - "Analyse": simple model calibrations on the curves of the Plot tab
    (same crop, colours, styles). Per curve, independently: linear
    (a*x+b), power (a*x^b), polynomial (degree 1-8), logarithmic
    (a*ln(x)+b) or exponential (a*exp(b*x)), optionally weighted by the
    curve's own measurement uncertainty (its "_std" column, 1/sigma
    weights), on a point window chosen with a double-handle slider --
    two small squares of the curve's colour mark the window's ends on the
    plot. FIT GROUPS let several curves share ONE regression: create a
    group ("+ New group" / "+ Group with all curves"), then pick it in
    each curve's "Fit with" box. The members' fit windows are pooled and
    the curves' own scatter around the common trend is estimated and
    added to the measurement uncertainty (two-pass variance pooling), so
    the result reflects curve-to-curve variability; per-curve RMS
    residuals are listed. Every fit (individual or grouped) is drawn as a
    thick line in the curve's / group's colour, optionally extrapolated
    over the whole axis, with a shaded band (fit uncertainty of the line,
    or curve-to-curve spread, ± k σ) -- see "Fit appearance". The Analyse
    plot also uses the Plot tab's title and axis limits. Power/exponential
    are fitted through a log
    transform (weights propagated accordingly); R² is always reported in
    the original y units.

This is a PySide6 port of the original Tkinter tool -- same feature set,
same file formats, plus curve cropping and in-place legend-label editing
(double-click a curve's "Label" cell in the table). PySide6 (rather than
PyQt6) was chosen deliberately: it's the exact same Qt binding HYDRA's
own shpb_app/main.py already uses, so nothing new needs installing if
HYDRA already runs on this machine. Requirements: Python 3.9+, numpy,
matplotlib, and PySide6 (``pip install PySide6`` if it isn't already
present). Reading .xlsx/.xlsm/.xls files additionally needs openpyxl
(``pip install openpyxl``) -- only imported if you actually load one, so
it's fine not to have it if you only ever use .csv/.txt.

Run with:   python hydra_curve_plotter.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np

# Pin the Qt binding explicitly *before* matplotlib picks one, so this
# always renders through PySide6 even if some other Qt binding (PyQt6,
# PyQt5, ...) also happens to be installed in the same environment --
# matplotlib's "QtAgg" backend otherwise just uses whichever one it finds
# first, which would be a silent, confusing thing to depend on. PySide6 is
# the binding HYDRA's own UI (shpb_app) is built on.
os.environ.setdefault("QT_API", "pyside6")

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT


# ---------------------------------------------------------------------------
# Column -> friendly label mapping (mirrors shpb_app's own QUANTITY_LABELS,
# ui/pages/page_results_export.py, plus the "confinement_*"-prefixed keys
# that same page's exporter adds when a confinement/triaxial analysis was
# computed -- core/confinement.py's per_draw_confinement_quantities and its
# "_laser" variants). Exported CSV/.txt column names carry no units of
# their own, so this is what turns e.g. "confinement_pressure_laser" into
# "Pressure (laser) [MPa]" in the dropdowns below. Unknown columns (a
# future app version, or a hand-edited file) still work via the generic
# fallback in pretty_label() -- this dict is a convenience, not a
# requirement.
# ---------------------------------------------------------------------------
QUANTITY_LABELS = {
    "strain": "Strain (true) [-]",
    "stress": "Stress (true) [MPa]",
    "nominal_strain": "Nominal (engineering) strain [-]",
    "nominal_stress": "Nominal (engineering) stress [MPa]",
    "strain_rate": "Strain rate [1/s]",
    "nominal_strain_rate": "Nominal strain rate [1/s]",
    "dic_strain": "DIC Strain [-]",
    "dic_strain_rate": "DIC Strain Rate [1/s]",
    "strain_gage_strain": "Strain Gage Strain [-]",
    "temperature": "Temperature [K]",
    "displacement": "Displacement [mm]",
    "load": "Load [N]",
    "force_input": "Input-bar force [N]",
    "force_output": "Output-bar force [N]",
    "force_average": "Average force [N]",
    "inertia_stress": "Inertia stress correction [MPa]",
    "friction_stress": "Friction stress correction [MPa]",
    "punch_strain": "Punch strain correction [-]",
    "balance_quality_force": "Dynamic equilibrium: work-based [-]",
    "balance_quality_waves": "Dynamic equilibrium: wave energy [-]",
    "balance_quality_force_instant": "Dynamic equilibrium: instantaneous [-]",
    "signal": "Custom sensor signal (calibrated)",
    "incident": "Incident wave (raw, cut)",
    "incident_corrected": "Incident wave (dispersion-corrected)",
    "reflected": "Reflected wave (raw, cut)",
    "reflected_corrected": "Reflected wave (dispersion-corrected)",
    "transmitted": "Transmitted wave (raw, cut)",
    "transmitted_corrected": "Transmitted wave (dispersion-corrected)",
    "time": "Time [s]",
    # Confinement / triaxial analysis (core/confinement.py), as exported
    # with the "confinement_" prefix:
    "confinement_strain": "Axial strain (confinement) [-]",
    "confinement_axial_stress": "Axial stress [MPa]",
    "confinement_radial_stress": "Radial stress [MPa]",
    "confinement_pressure": "Pressure [MPa]",
    "confinement_deviator": "Stress deviator [MPa]",
    "confinement_radial_strain": "Radial strain [-]",
    "confinement_volumetric_strain": "Volumic strain [-]",
    "confinement_equivalent_strain": "Equivalent strain [-]",
    "confinement_axial_stress_laser": "Axial stress (laser) [MPa]",
    "confinement_pressure_laser": "Pressure (laser) [MPa]",
    "confinement_deviator_laser": "Stress deviator (laser) [MPa]",
}

_SENSOR_SIGNAL_RE = re.compile(r"^sensor_(\d+)_signal$")


def pretty_label(key: str) -> str:
    """Friendly display name for a raw CSV column name, falling back to a
    generically-prettified version of the key itself for anything not in
    QUANTITY_LABELS (a future quantity, a hand-renamed column, ...) rather
    than refusing to show it."""
    if key in QUANTITY_LABELS:
        return QUANTITY_LABELS[key]
    if key.endswith("_std"):
        base = key[: -len("_std")]
        return f"{pretty_label(base)} (std-dev)"
    m = _SENSOR_SIGNAL_RE.match(key)
    if m:
        return f"Custom sensor {m.group(1)} signal (Monte-Carlo cut)"
    text = key.replace("confinement_", "").replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else key


# ---------------------------------------------------------------------------
# Loading exported files
# ---------------------------------------------------------------------------

def _locale_float(token: str) -> float:
    """Parses a number that may use either '.' or ',' as the decimal
    separator -- a table pasted straight out of a French-locale Excel (or
    typed by hand there) commonly uses ',' (e.g. "16,0517"), while
    shpb_app's own exports -- and English-locale files -- use '.'. A
    token with BOTH separators (a thousands-grouped "1.234,56" or
    "1,234.56") is handled by treating whichever one comes LAST as the
    actual decimal point and discarding the earlier one, so either
    grouping convention works; a token with only ',' has it swapped for
    '.' outright. Scientific notation ("-4,13984E-13") survives this
    unchanged, since only the comma/dot characters are touched."""
    token = token.strip()
    if not token:
        raise ValueError("empty value")
    if "," in token and "." in token:
        token = token.replace(".", "").replace(",", ".") if token.rfind(",") > token.rfind(".") \
            else token.replace(",", "")
    elif "," in token:
        token = token.replace(",", ".")
    return float(token)


def _split_text_row(line: str) -> list[str]:
    """Splits one line of a plain text table into cells. Tries a literal
    TAB character first -- how Excel's own copy/paste-as-text produces a
    multi-word column header like "Stress Deviator (MPa)" without
    breaking it apart at its internal spaces; if the line has no tabs at
    all, falls back to runs of 2-or-more spaces, which still keeps a
    single space inside a header label (e.g. "Pressure (MPa)") intact
    while treating a wider gap as the real column boundary."""
    if "\t" in line:
        return [c.strip() for c in line.split("\t")]
    return [c.strip() for c in re.split(r"\s{2,}", line.strip())]


def _load_plain_text_table(path: str) -> dict[str, np.ndarray]:
    """Reads a plain text table that is neither one of shpb_app's own
    "#"-header exports nor a comma-separated .csv: one header line of
    column names, then one numeric row per line, columns separated by a
    TAB or by 2+ spaces (see _split_text_row), each number allowed to use
    either '.' or ',' as its own decimal separator (see _locale_float).
    This is the shape a table copy-pasted straight out of Excel -- or
    typed by hand in a French-locale spreadsheet and saved as .txt --
    naturally comes out as."""
    base = os.path.basename(path)
    with open(path, "r", newline="") as f:
        lines = [ln.rstrip("\n").rstrip("\r") for ln in f]
    lines = [ln for ln in lines if ln.strip() != ""]
    if not lines:
        raise ValueError(f"'{base}' is empty.")
    header = [h for h in _split_text_row(lines[0]) if h != ""]
    if not header:
        raise ValueError(f"'{base}': could not read a header row.")
    ncols = len(header)
    data_lines = lines[1:]
    if not data_lines:
        raise ValueError(f"'{base}': header found but no data rows.")
    values: list[list[float]] = [[] for _ in range(ncols)]
    for r_idx, line in enumerate(data_lines, start=2):
        cells = _split_text_row(line)
        if len(cells) != ncols:
            raise ValueError(
                f"'{base}': row {r_idx} has {len(cells)} value(s), expected {ncols} (to match the header)."
            )
        for col_idx, cell in enumerate(cells):
            try:
                values[col_idx].append(_locale_float(cell))
            except ValueError as exc:
                raise ValueError(
                    f"'{base}': row {r_idx}, column '{header[col_idx]}' ('{cell}') is not numeric."
                ) from exc
    return {name: np.array(vals, dtype=float) for name, vals in zip(header, values)}


def _load_xlsx(path: str) -> dict[str, np.ndarray]:
    """Reads the first worksheet of an .xlsx/.xlsm/.xls file into
    {column_name: array}, exactly like load_table() does for .csv/.txt --
    a header row of plain column names, followed by numeric data rows.
    Decimal-separator locale (comma vs dot) is a non-issue here: Excel
    always stores numbers as actual binary floats regardless of how
    they're DISPLAYED (a cell showing "16,0517" under a French locale is
    still just the float 16.0517 underneath), so openpyxl hands back real
    Python numbers with no text parsing needed -- unlike the plain-text
    path above, which has to actually interpret the character. The rare
    case of a numeric-looking value stored as literal TEXT in the sheet
    is still handled, via the same comma/dot-flexible _locale_float()."""
    try:
        import openpyxl
    except ImportError as exc:
        raise ValueError(
            "Reading .xlsx files needs the 'openpyxl' package (pip install openpyxl)."
        ) from exc
    base = os.path.basename(path)
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001 - surface any backend error plainly
        raise ValueError(f"'{base}': could not open as an Excel file ({exc}).") from exc
    try:
        ws = wb.worksheets[0]
        rows = [row for row in ws.iter_rows(values_only=True)
                if any(v is not None and str(v).strip() != "" for v in row)]
    finally:
        wb.close()
    if not rows:
        raise ValueError(f"'{base}' is empty.")
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    header = [h for h in header if h != ""]
    if not header:
        raise ValueError(f"'{base}': could not read a header row.")
    ncols = len(header)
    data_rows = rows[1:]
    if not data_rows:
        raise ValueError(f"'{base}': header found but no data rows.")
    values: list[list[float]] = [[] for _ in range(ncols)]
    for r_idx, row in enumerate(data_rows, start=2):
        if len(row) < ncols:
            raise ValueError(f"'{base}': row {r_idx} has fewer cells than the header ({len(row)} vs {ncols}).")
        for col_idx in range(ncols):
            cell = row[col_idx]
            if cell is None:
                raise ValueError(f"'{base}': row {r_idx}, column '{header[col_idx]}' is empty.")
            if isinstance(cell, (int, float)):
                values[col_idx].append(float(cell))
            else:
                try:
                    values[col_idx].append(_locale_float(str(cell)))
                except ValueError as exc:
                    raise ValueError(
                        f"'{base}': row {r_idx}, column '{header[col_idx]}' ('{cell}') is not numeric."
                    ) from exc
    return {name: np.array(vals, dtype=float) for name, vals in zip(header, values)}


def load_table(path: str) -> dict[str, np.ndarray]:
    """Reads a data table into {column_name: array}. Three source shapes
    are recognized, dispatched mostly by extension:

      - .xlsx / .xlsm / .xls -- any Excel workbook with a header row and
        numeric columns on its first sheet (see _load_xlsx). This covers
        importing a file like the one in the user's own request: a
        spreadsheet with "Pressure (MPa)" / "Stress Deviator (MPa)"
        columns, independent of whatever decimal-separator locale Excel
        happens to be displaying it in.

      - shpb_app's own "Results & Export" exports -- sniffed from the
        first line rather than the extension (so a renamed file still
        works): ".csv" (comma-separated, plain header row) or ".txt"
        (tab-separated, header on a "# ..." first line, see
        page_results_export.py's _write_columns). Every column in one of
        these is guaranteed the same length by the exporter itself (it
        truncates every quantity to the shortest one before writing), so
        this never has to reconcile differing lengths within one file.

      - any other plain text table (typically ".txt") -- a header line of
        column names, TAB- or multi-space-separated data rows, where each
        number may use '.' OR ',' as its decimal separator (see
        _load_plain_text_table / _locale_float). This is what a table
        copy-pasted straight out of Excel, or saved as .txt from a
        French-locale spreadsheet, naturally looks like."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        return _load_xlsx(path)

    base = os.path.basename(path)
    with open(path, "r", newline="") as f:
        first_line = f.readline()
    if first_line.lstrip().startswith("#"):
        header = [h.strip() for h in first_line.lstrip("#").strip().split("\t") if h.strip()]
        if not header:
            raise ValueError(f"'{base}': could not read a header from the first line.")
        try:
            data = np.genfromtxt(path, delimiter="\t", skip_header=1, dtype=float)
        except Exception as exc:  # noqa: BLE001 - surface as a plain, readable message
            raise ValueError(f"'{base}': could not parse as a tab-delimited export ({exc}).") from exc
        if data.size == 0:
            raise ValueError(f"'{base}': header found but no data rows.")
        if data.ndim == 1:
            data = data.reshape(-1, 1) if len(header) == 1 else data.reshape(1, -1)
        if data.shape[1] != len(header):
            raise ValueError(
                f"'{base}': header has {len(header)} column(s) but the data has {data.shape[1]}."
            )
        return {name: data[:, i] for i, name in enumerate(header)}

    if ext != ".csv":
        # Not a shpb_app "#"-header export and not a (comma-separated)
        # .csv -- assume a plain, locale-flexible text table instead of
        # forcing it through the comma-delimited CSV reader below, which
        # would badly mis-split a comma-decimal number like "16,0517"
        # into two separate "columns".
        return _load_plain_text_table(path)

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"'{base}' is empty.")
        header = [h.strip() for h in header if h.strip() != ""]
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not header:
        raise ValueError(f"'{base}': could not read a header row.")
    if not rows:
        raise ValueError(f"'{base}': header found but no data rows.")
    try:
        arr = np.array(rows, dtype=float)
    except ValueError as exc:
        raise ValueError(f"'{base}' contains a non-numeric value ({exc}).") from exc
    if arr.shape[1] != len(header):
        raise ValueError(f"'{base}': header has {len(header)} column(s) but the data has {arr.shape[1]}.")
    return {name: arr[:, i] for i, name in enumerate(header)}


def default_label(path: str) -> str:
    """A sensible default curve/file label from a file path -- the file's
    own name, minus extension (matches how the app names exports after
    the test's own label, e.g. "HEX8-2.csv" -> "HEX8-2")."""
    return os.path.splitext(os.path.basename(path))[0]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

PALETTE = [
    # The first six are the user's preferred colours, in this order.
    "#00dada", "#ff7f0e", "#000000", "#d62728", "#00ea00", "#0055ff",
    "#1f77b4", "#2ca02c", "#9467bd", "#17becf",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#aec7e8", "#ffbb78",
    "#98df8a", "#c5b0d5", "#c49c94", "#f7b6d2",
]


def next_color(used: list[str]) -> str:
    for c in PALETTE:
        if c not in used:
            return c
    return PALETTE[len(used) % len(PALETTE)]


# Matplotlib linestyle codes, paired with a friendly display name -- shown
# in the curve dialog's "Line style" combo and in the Curves table's
# "Style" column. A plain list of (label, code) pairs rather than a dict,
# so the dialog's combo keeps this exact display order (dict insertion
# order would too, in modern Python, but a list makes that guarantee
# obvious rather than incidental).
LINESTYLE_CHOICES: list[tuple[str, str]] = [
    ("Solid", "-"),
    ("Dashed", "--"),
    ("Dotted", ":"),
    ("Dash-dot", "-."),
]
_LINESTYLE_CODES = {code: label for label, code in LINESTYLE_CHOICES}


def pretty_linestyle(code: str) -> str:
    """Friendly name for a stored matplotlib linestyle code, e.g. for the
    Curves table's "Style" column -- falls back to the raw code itself for
    anything outside LINESTYLE_CHOICES (a hand-edited session file, or a
    matplotlib code this tool doesn't offer in its own combo, such as a
    custom dash tuple) rather than refusing to show it."""
    return _LINESTYLE_CODES.get(code, code)


def _parse_float(text: str) -> float | None:
    """Parses an axis-limit field: blank (or unparseable) means "leave this
    bound on auto", not an error -- matplotlib's set_xlim/set_ylim already
    treat a None bound that way, so this just turns "" into that sentinel."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_factor(text: str) -> float:
    """A multiplier field: accepts '.' or ',' decimals and exponent notation
    ("1e-3"); blank or unparseable means 1 (no scaling), never an error."""
    text = (text or "").strip().replace(",", ".")
    if not text:
        return 1.0
    try:
        value = float(text)
    except ValueError:
        return 1.0
    return value if np.isfinite(value) else 1.0


def _parse_norm(text: str) -> float:
    """A "normalise by" field: the reference value the axis is DIVIDED by.
    Same number syntax as _parse_factor; blank, unparseable, zero or
    non-finite means 1 (no normalisation) -- a typo must never produce
    inf/NaN curves."""
    value = _parse_factor(text)
    return value if value != 0.0 else 1.0


def _first_finite(a) -> float:
    a = np.asarray(a, dtype=float)
    idx = np.flatnonzero(np.isfinite(a))
    return float(a[idx[0]]) if len(idx) else 0.0


def transform_curve(x, y, std=None, *, flip_x=False, flip_y=False, zero_x=False, zero_y=False,
                    factor_x=1.0, factor_y=1.0, norm_x=1.0, norm_y=1.0):
    """Display-time transformation of one (already cropped) curve:
         v' = sign * factor * (v - v_first) / norm   (v_first only if "zero")
    where v_first is the curve's first finite value, i.e. the curve is
    shifted to start at 0 BEFORE it is scaled/inverted, and `norm` is the
    user's reference value the axis is normalised by (1 = none). The
    measurement uncertainty `std` (a spread of y, not a position) is only
    scaled by |factor_y / norm_y|: an offset or a sign flip doesn't change it. Returns
    (x', y', std'), std' being None if std was None. The source data are
    never modified."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if zero_x:
        x = x - _first_finite(x)
    if zero_y:
        y = y - _first_finite(y)
    norm_x = norm_x if norm_x else 1.0
    norm_y = norm_y if norm_y else 1.0
    fx, fy = factor_x / norm_x, factor_y / norm_y
    sx = -fx if flip_x else fx
    sy = -fy if flip_y else fy
    std_t = None if std is None else np.asarray(std, dtype=float) * abs(fy)
    return x * sx, y * sy, std_t


def resolve_crop(n: int, crop_lo: int, crop_hi: int) -> tuple[int, int]:
    """Clamps a saved (crop_lo, crop_hi) pair to a curve's actual point
    count `n`, returning the (lo, hi) *inclusive* point indices to keep.

    crop_hi < 0 is the sentinel CurveSpec.crop_hi defaults to, meaning
    "keep to the last point" -- this is what lets a crop set against one
    version of a file keep showing the full tail, un-chopped, if the
    source file is later reloaded with a slightly different point count,
    instead of silently truncating real data that was never meant to be
    cropped. Both bounds are independently clamped into [0, n-1] and
    swapped if they'd otherwise cross, so a slider drag (or a hand-edited
    session file) can never produce an out-of-range or empty-looking crop
    by accident. For an empty curve (n == 0) returns (0, -1), which the
    caller's own x[lo:hi+1] slicing turns into an empty slice -- there is
    nothing meaningful to clamp into."""
    if n <= 0:
        return 0, -1
    lo = max(0, min(int(crop_lo), n - 1))
    hi = n - 1 if crop_hi is None or crop_hi < 0 else max(0, min(int(crop_hi), n - 1))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


@dataclass
class LoadedFile:
    path: str
    label: str
    columns: dict  # name -> np.ndarray (not included in asdict() calls below -- handled separately)

    @property
    def keys(self) -> list[str]:
        return list(self.columns.keys())

    @property
    def plottable_keys(self) -> list[str]:
        """Columns worth offering as an axis choice -- everything except
        the "_std" companions, which are uncertainty bands for another
        column, not something you'd plot as its own curve."""
        return [k for k in self.columns if not k.endswith("_std")]


@dataclass
class CurveSpec:
    file_label: str
    x_key: str
    y_key: str
    label: str
    color: str
    show_band: bool = False
    linestyle: str = "-"
    linewidth: float = 1.8
    visible: bool = True  # hidden curves are neither plotted nor analysed
    # Which of this curve's own points to keep, by point INDEX (not by X
    # value -- indices are what "drop N points from the start/end" means,
    # and stay meaningful even if two curves share the same X range but
    # different sampling). crop_hi == -1 is a sentinel for "to the last
    # point" -- see resolve_crop()'s docstring for why that -- not simply
    # recording the point count at crop time -- is the robust choice.
    crop_lo: int = 0
    crop_hi: int = -1
    # Analyse-tab (curve fitting) settings, all per-curve. fit_lo/fit_hi
    # index into the curve's points AFTER cropping (i.e. the zone actually
    # shown on the plot), same -1-means-"to the end" sentinel convention as
    # crop_hi. fit_kind is one of the codes in FIT_KINDS ("none" = no fit).
    fit_kind: str = "none"
    fit_degree: int = 1
    fit_weighted: bool = True
    fit_lo: int = 0
    fit_hi: int = -1
    # Name of the FitGroup this curve's fit window is pooled into ("" = the
    # curve is fitted on its own, with fit_kind/fit_degree above).
    fit_group: str = ""


@dataclass
class FitGroup:
    """One regression shared by several curves (Analyse tab): the fit
    windows of every curve whose `fit_group` equals `name` are pooled
    (fit_combined) and ONE model is fitted to them. `color` "" = use the
    colour of the first member curve."""
    name: str
    kind: str = "linear"
    degree: int = 2
    color: str = ""


# ---------------------------------------------------------------------------
# Curve-fitting / calibration engine (Analyse tab). Pure numpy -- no Qt, so
# it can be unit-tested without a display.
# ---------------------------------------------------------------------------

FIT_KINDS: list[tuple[str, str]] = [
    ("None", "none"),
    ("Linear  y = a·x + b", "linear"),
    ("Power  y = a·x^b", "power"),
    ("Polynomial (degree 1–8)", "polynomial"),
    ("Logarithmic  y = a·ln(x) + b", "log"),
    ("Exponential  y = a·exp(b·x)", "exp"),
]


@dataclass
class FitResult:
    """Outcome of fitting one model to one (optionally weighted) sample.

    `params` / `param_stderr` are in the same order (for linear: [a, b];
    power/log/exp: [a, b]; polynomial: highest degree first, numpy's own
    order). `r_squared` is always computed in the ORIGINAL x/y units, even
    for models fitted through a log transform, since that's the number a
    reader can actually interpret. `predict` evaluates the fitted model at
    arbitrary x (used to draw the fit line)."""
    kind: str
    params: np.ndarray
    param_stderr: np.ndarray
    r_squared: float
    equation: str
    n_points: int
    predict: Callable[[np.ndarray], np.ndarray]
    # band(x, k) -> (lower, upper): +/- k standard errors of the FITTED CURVE
    # (confidence band of the mean trend, from the parameter covariance --
    # NOT of individual points). None when the covariance is unavailable
    # (as many points as parameters). Linear-in-parameters models give an
    # additive band; power/exp are linear in log space, so their band is
    # multiplicative (stays positive).
    band: Callable | None = None


def _format_g(value: float, sig: int = 4) -> str:
    return f"{value:.{sig}g}"


def _signed_term(value: float, suffix: str) -> str:
    """One equation term, e.g. (2.5, "x") -> "2.5·x", (-0.3, "") ->
    "−0.3". The sign is carried as a leading minus so _join_terms()
    can render " − " between terms without ever producing "+ -"."""
    text = _format_g(abs(value))
    body = f"{text}·{suffix}" if suffix else text
    return ("−" if value < 0 else "") + body


def _join_terms(terms: list[str]) -> str:
    out = terms[0]
    for t in terms[1:]:
        out += f" − {t[1:]}" if t.startswith("−") else f" + {t}"
    return out


def _polyfit_weighted(x, y, degree: int, w=None):
    """numpy.polyfit(..., cov=True) with graceful degradation: numpy's own
    convention is that `w` is 1/sigma (NOT 1/sigma**2) for Gaussian
    uncertainties, which is the convention used throughout this module.
    If there aren't more points than parameters the covariance can't be
    estimated -- returns NaN standard errors instead of raising, so a very
    narrow fit window degrades to "no error bars" rather than crashing."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_params = degree + 1
    if len(x) < n_params:
        raise ValueError(f"Need at least {n_params} points for this model (got {len(x)}).")
    if len(x) == n_params:
        coeffs = np.polyfit(x, y, degree, w=w)
        return coeffs, np.full((n_params, n_params), np.nan)
    coeffs, cov = np.polyfit(x, y, degree, w=w, cov=True)
    return coeffs, cov


def _r_squared(y, y_pred, w=None) -> float:
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if w is None:
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    else:
        w = np.asarray(w, dtype=float)
        if np.sum(w ** 2) <= 0:
            return float("nan")
        ybar = np.average(y, weights=w ** 2)
        ss_res = float(np.sum((w * (y - y_pred)) ** 2))
        ss_tot = float(np.sum((w * (y - ybar)) ** 2))
    if ss_tot <= 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _make_band(basis, cov, predict, multiplicative: bool = False):
    """Builds FitResult.band from a design-row function `basis(x) -> (n, p)`
    and the parameter covariance `cov` (p x p, same parameter order as the
    basis columns): var(x) = phi(x)^T C phi(x). Returns None if `cov` has
    non-finite entries."""
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or not np.all(np.isfinite(cov)):
        return None

    def band(xx, k: float = 2.0):
        xx = np.asarray(xx, dtype=float)
        phi = np.asarray(basis(xx), dtype=float)
        var = np.einsum("ij,jk,ik->i", phi, cov, phi)
        sd = np.sqrt(np.clip(var, 0.0, None))
        mid = np.asarray(predict(xx), dtype=float)
        if multiplicative:
            return mid * np.exp(-k * sd), mid * np.exp(k * sd)
        return mid - k * sd, mid + k * sd

    return band


def _safe_log(xx):
    xx = np.asarray(xx, dtype=float)
    out = np.full(xx.shape, np.nan)
    pos = xx > 0
    out[pos] = np.log(xx[pos])
    return out


def fit_linear(x, y, w=None) -> FitResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    coeffs, cov = _polyfit_weighted(x, y, 1, w=w)
    a, b = float(coeffs[0]), float(coeffs[1])
    stderr = np.sqrt(np.diag(cov))

    def predict(xx):
        return a * np.asarray(xx, dtype=float) + b

    r2 = _r_squared(y, predict(x), w=w)
    eq = "y = " + _join_terms([_signed_term(a, "x"), _signed_term(b, "")])
    band = _make_band(lambda xx: np.column_stack([xx, np.ones_like(xx)]), cov, predict)
    return FitResult("linear", np.array([a, b]), stderr, r2, eq, len(x), predict, band)


def fit_polynomial(x, y, degree: int, w=None) -> FitResult:
    degree = max(1, min(int(degree), 8))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    coeffs, cov = _polyfit_weighted(x, y, degree, w=w)
    stderr = np.sqrt(np.diag(cov))

    def predict(xx):
        return np.polyval(coeffs, np.asarray(xx, dtype=float))

    r2 = _r_squared(y, predict(x), w=w)
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        suffix = "" if power == 0 else ("x" if power == 1 else f"x^{power}")
        terms.append(_signed_term(float(c), suffix))
    return FitResult("polynomial", np.asarray(coeffs, dtype=float), stderr, r2,
                     "y = " + _join_terms(terms), len(x), predict,
                     _make_band(lambda xx: np.vander(xx, degree + 1), cov, predict))


def fit_power(x, y, w=None) -> FitResult:
    """y = a·x^b via the log-log linearization ln y = ln a + b·ln x
    (the same thing a spreadsheet "power trendline" does). Points with
    x <= 0 or y <= 0 can't be log-transformed and are left out of the FIT
    only. The linear-space weight is propagated into log space: since
    d(ln y)/dy = 1/y, sigma(ln y) ~ sigma_y / y, so w_log = y·w."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    wv = None if w is None else np.asarray(w, dtype=float)
    if mask.sum() < 2:
        raise ValueError("Power fit needs at least 2 points with x > 0 and y > 0.")
    xf, yf = x[mask], y[mask]
    w_log = None if wv is None else wv[mask] * yf
    coeffs, cov = _polyfit_weighted(np.log(xf), np.log(yf), 1, w=w_log)
    b, ln_a = float(coeffs[0]), float(coeffs[1])
    a = float(np.exp(ln_a))
    # params = [a, b]; polyfit's cov is ordered [slope(b), intercept(ln a)].
    stderr = np.array([a * np.sqrt(cov[1, 1]), np.sqrt(cov[0, 0])])

    def predict(xx):
        xx = np.asarray(xx, dtype=float)
        out = np.full(xx.shape, np.nan)
        pos = xx > 0
        out[pos] = a * xx[pos] ** b
        return out

    r2 = _r_squared(yf, predict(xf), w=None if wv is None else wv[mask])
    return FitResult("power", np.array([a, b]), stderr, r2,
                     f"y = {_format_g(a)}·x^{_format_g(b)}", int(mask.sum()), predict,
                     _make_band(lambda xx: np.column_stack([_safe_log(xx), np.ones_like(xx)]), cov, predict,
                                multiplicative=True))


def fit_log(x, y, w=None) -> FitResult:
    """y = a·ln(x) + b -- linear in ln(x), so y (and its weight) are used
    as-is. Points with x <= 0 are left out of the fit only."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = x > 0
    wv = None if w is None else np.asarray(w, dtype=float)
    if mask.sum() < 2:
        raise ValueError("Logarithmic fit needs at least 2 points with x > 0.")
    xf, yf = x[mask], y[mask]
    wf = None if wv is None else wv[mask]
    coeffs, cov = _polyfit_weighted(np.log(xf), yf, 1, w=wf)
    a, b = float(coeffs[0]), float(coeffs[1])
    stderr = np.sqrt(np.diag(cov))

    def predict(xx):
        xx = np.asarray(xx, dtype=float)
        out = np.full(xx.shape, np.nan)
        pos = xx > 0
        out[pos] = a * np.log(xx[pos]) + b
        return out

    r2 = _r_squared(yf, predict(xf), w=wf)
    eq = "y = " + _join_terms([_signed_term(a, "ln(x)"), _signed_term(b, "")])
    band = _make_band(lambda xx: np.column_stack([_safe_log(xx), np.ones_like(xx)]), cov, predict)
    return FitResult("log", np.array([a, b]), stderr, r2, eq, int(mask.sum()), predict, band)


def fit_exp(x, y, w=None) -> FitResult:
    """y = a·exp(b·x) via ln y = ln a + b·x. Points with y <= 0 are
    left out of the fit only; the weight is propagated into log space the
    same way as fit_power (w_log = y·w)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = y > 0
    wv = None if w is None else np.asarray(w, dtype=float)
    if mask.sum() < 2:
        raise ValueError("Exponential fit needs at least 2 points with y > 0.")
    xf, yf = x[mask], y[mask]
    w_log = None if wv is None else wv[mask] * yf
    coeffs, cov = _polyfit_weighted(xf, np.log(yf), 1, w=w_log)
    b, ln_a = float(coeffs[0]), float(coeffs[1])
    a = float(np.exp(ln_a))
    stderr = np.array([a * np.sqrt(cov[1, 1]), np.sqrt(cov[0, 0])])

    def predict(xx):
        return a * np.exp(b * np.asarray(xx, dtype=float))

    r2 = _r_squared(yf, predict(xf), w=None if wv is None else wv[mask])
    return FitResult("exp", np.array([a, b]), stderr, r2,
                     f"y = {_format_g(a)}·exp({_format_g(b)}·x)", int(mask.sum()), predict,
                     _make_band(lambda xx: np.column_stack([xx, np.ones_like(xx)]), cov, predict,
                                multiplicative=True))


_FIT_FUNCS = {"linear": fit_linear, "power": fit_power, "polynomial": fit_polynomial,
              "log": fit_log, "exp": fit_exp}


def _run_fit(kind: str, x, y, w=None, degree: int = 1) -> FitResult:
    """Dispatches a stored FIT_KINDS code to the function that fits it --
    the single place that mapping lives (used by the per-curve path and,
    via fit_combined, the pooled path)."""
    if kind not in _FIT_FUNCS:
        raise ValueError(f"Unknown fit model '{kind}'.")
    if kind == "polynomial":
        return fit_polynomial(x, y, degree, w=w)
    return _FIT_FUNCS[kind](x, y, w=w)


def fit_combined(samples: list, kind: str, degree: int = 1):
    """Fits ONE model to several curves pooled together, accounting for how
    much the curves disagree with each other -- not just each point's own
    measurement uncertainty. `samples` is one (x, y, w_or_None) tuple per
    curve (w in numpy's 1/sigma convention). Returns (FitResult,
    per_curve_rms), where per_curve_rms[k] is curve k's RMS residual around
    the final combined fit, in y units -- which makes the between-curve
    variability visible instead of silently folded into the parameters.

    Method (a two-pass, method-of-moments-style variance pooling -- the
    simple cousin of a random-effects model, deliberately kept to what
    numpy alone can compute):
      1. Fit all pooled points once. Measurement weights are used in this
         pass only if EVERY curve has them (mixing curves with and without
         a measurement sigma would mean inventing one for the others).
      2. Measure each curve's own residual variance around that first fit
         -- how much that curve disagrees with the shared trend.
      3. Re-fit, weighting each point by 1/sqrt(total variance), where
         total variance = its measurement variance (if available) + its
         curve's between-curve variance, added in quadrature. A curve that
         disagrees a lot with the others is therefore down-weighted
         automatically."""
    if not samples:
        raise ValueError("No curve selected for the combined fit.")
    all_x = np.concatenate([np.asarray(s[0], dtype=float) for s in samples])
    all_y = np.concatenate([np.asarray(s[1], dtype=float) for s in samples])
    all_have_w = all(s[2] is not None for s in samples)

    w_pass1 = np.concatenate([np.asarray(s[2], dtype=float) for s in samples]) if all_have_w else None
    pass1 = _run_fit(kind, all_x, all_y, w=w_pass1, degree=degree)

    var_between_parts = []
    for xk, yk, _wk in samples:
        resid = np.asarray(yk, dtype=float) - pass1.predict(xk)
        resid = resid[np.isfinite(resid)]
        var_k = float(np.mean(resid ** 2)) if len(resid) else 0.0
        var_between_parts.append(np.full(len(xk), max(var_k, 1e-30)))
    var_between = np.concatenate(var_between_parts)

    if all_have_w:
        meas_var = np.concatenate([1.0 / np.clip(np.asarray(s[2], dtype=float), 1e-300, None) ** 2 for s in samples])
    else:
        meas_var = np.zeros_like(var_between)
    w_final = 1.0 / np.sqrt(np.clip(meas_var + var_between, 1e-300, None))

    final = _run_fit(kind, all_x, all_y, w=w_final, degree=degree)

    per_curve_rms = []
    for xk, yk, _wk in samples:
        resid = np.asarray(yk, dtype=float) - final.predict(xk)
        resid = resid[np.isfinite(resid)]
        per_curve_rms.append(float(np.sqrt(np.mean(resid ** 2))) if len(resid) else float("nan"))
    return final, np.array(per_curve_rms)


def group_fit(samples: list, kind: str, degree: int = 1):
    """Fit for one fit unit: a single curve (one sample) or several pooled
    curves (fit_combined). Returns (FitResult, per_curve_rms, pooled_rms),
    rms in y units around the final fit (pooled_rms over all points)."""
    if not samples:
        raise ValueError("No curve in this fit.")
    if len(samples) == 1:
        xk, yk, wk = samples[0]
        res = _run_fit(kind, xk, yk, w=wk, degree=degree)
        rms_each = np.array([_rms_resid(res, xk, yk)])
    else:
        res, rms_each = fit_combined(samples, kind, degree)
    all_x = np.concatenate([np.asarray(s[0], dtype=float) for s in samples])
    all_y = np.concatenate([np.asarray(s[1], dtype=float) for s in samples])
    return res, rms_each, _rms_resid(res, all_x, all_y)


def _rms_resid(res: "FitResult", x, y) -> float:
    r = np.asarray(y, dtype=float) - res.predict(np.asarray(x, dtype=float))
    r = r[np.isfinite(r)]
    return float(np.sqrt(np.mean(r ** 2))) if len(r) else float("nan")


BAND_MODES: list[tuple[str, str]] = [
    ("None", "none"),
    ("Fit uncertainty (confidence band of the line)", "param"),
    ("Curve-to-curve spread (± k · RMS residual)", "spread"),
]
EXTEND_MODES: list[tuple[str, str]] = [
    ("Fit window only", "window"),
    ("Range of the group's curves", "data"),
    ("Full axis (extrapolate)", "axes"),
]


def fit_band_arrays(res: "FitResult", xs, mode: str, k: float, rms: float | None = None):
    """(lower, upper) arrays of the shaded band around the fitted line at
    `xs`, or None. mode "param": +/- k standard errors of the fitted curve
    (res.band); mode "spread": +/- k times the RMS residual of the pooled
    points (constant width -- includes the scatter between curves)."""
    xs = np.asarray(xs, dtype=float)
    if mode == "param" and res.band is not None:
        return res.band(xs, k)
    if mode == "spread" and rms is not None and np.isfinite(rms):
        mid = res.predict(xs)
        return mid - k * rms, mid + k * rms
    return None


# ---------------------------------------------------------------------------
# Custom double-handle range slider (PySide6 has no built-in one). Used to
# crop a curve down to the point-index window you want to keep.
# ---------------------------------------------------------------------------

def _slider_value_from_pos(pos_x: float, width: int, minimum: int, maximum: int, margin: int) -> int:
    """Maps a pixel X position inside a RangeSlider's track (track pixel
    width `width`, with `margin` px reserved on each side for the handle's
    own radius, so a handle's visual center never goes outside the
    widget) to the nearest integer value in [minimum, maximum]. A plain,
    Qt-free function -- deliberately factored out of RangeSlider itself so
    the pixel<->value math can be unit-tested without a real widget/event
    loop/display (this sandbox has no PySide6 installed at all)."""
    usable = max(1, width - 2 * margin)
    frac = (pos_x - margin) / usable
    frac = min(1.0, max(0.0, frac))
    if maximum <= minimum:
        return minimum
    return minimum + round(frac * (maximum - minimum))


def _slider_pos_from_value(value: int, width: int, minimum: int, maximum: int, margin: int) -> float:
    """Inverse of _slider_value_from_pos: the pixel X position of a given
    value's handle center."""
    span = maximum - minimum
    frac = 0.0 if span <= 0 else (value - minimum) / span
    frac = min(1.0, max(0.0, frac))
    usable = max(1, width - 2 * margin)
    return margin + frac * usable


class RangeSlider(QtWidgets.QWidget):
    """A horizontal double-handle range slider: drag either handle, or
    click anywhere on the track to jump the nearest handle there, to set
    an integer [low, high] range. Not a built-in Qt widget, so implemented
    directly here with QPainter -- deliberately simple (a track line, a
    highlighted sub-range, two round handles) to keep the one part of this
    tool that can't be exercised in a headless/no-PySide6 environment as
    small and low-risk as possible; all of its actual pixel<->value math
    lives in the two plain functions above, which *are* tested that way.

    Emits rangeChanged(low, high) continuously while dragging (for live
    preview) and rangeCommitted(low, high) once on release."""

    rangeChanged = Signal(int, int)
    rangeCommitted = Signal(int, int)

    HANDLE_RADIUS = 7
    MARGIN = HANDLE_RADIUS + 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self._minimum = 0
        self._maximum = 100
        self._low = 0
        self._high = 100
        self._dragging: str | None = None  # "low" | "high" | None
        self.setMinimumHeight(30)
        self.setMinimumWidth(140)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # -- value access ---------------------------------------------------
    def setRange(self, minimum: int, maximum: int):
        self._minimum = int(minimum)
        self._maximum = max(int(maximum), self._minimum)
        self._low = max(self._minimum, min(self._low, self._maximum))
        self._high = max(self._minimum, min(self._high, self._maximum))
        if self._high < self._low:
            self._low, self._high = self._high, self._low
        self.update()

    def setValues(self, low: int, high: int):
        low = max(self._minimum, min(int(low), self._maximum))
        high = max(self._minimum, min(int(high), self._maximum))
        if high < low:
            low, high = high, low
        self._low, self._high = low, high
        self.update()

    def values(self) -> tuple[int, int]:
        return self._low, self._high

    def minimum(self) -> int:
        return self._minimum

    def maximum(self) -> int:
        return self._maximum

    # -- pixel <-> value, via the plain module-level helpers -------------
    def _pos(self, value: int) -> float:
        return _slider_pos_from_value(value, self.width(), self._minimum, self._maximum, self.MARGIN)

    def _value(self, pos_x: float) -> int:
        return _slider_value_from_pos(pos_x, self.width(), self._minimum, self._maximum, self.MARGIN)

    # -- painting ---------------------------------------------------------
    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        mid_y = self.height() // 2
        track_color = self.palette().color(QtGui.QPalette.ColorRole.Mid)
        accent = QtGui.QColor("#3a7bd5")

        track_pen = QtGui.QPen(track_color, 3)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.drawLine(self.MARGIN, mid_y, self.width() - self.MARGIN, mid_y)

        x_lo, x_hi = self._pos(self._low), self._pos(self._high)
        range_pen = QtGui.QPen(accent, 4)
        range_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(range_pen)
        painter.drawLine(int(round(x_lo)), mid_y, int(round(x_hi)), mid_y)

        for value, name in ((self._low, "low"), (self._high, "high")):
            x = self._pos(value)
            fill = accent if self._dragging == name else QtGui.QColor("#ffffff")
            painter.setBrush(QtGui.QBrush(fill))
            painter.setPen(QtGui.QPen(accent, 2))
            painter.drawEllipse(QtCore.QPointF(x, mid_y), float(self.HANDLE_RADIUS), float(self.HANDLE_RADIUS))
        painter.end()

    # -- interaction --------------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        x_lo, x_hi = self._pos(self._low), self._pos(self._high)
        # Whichever handle is pixel-closer to the click wins -- including a
        # click on the bare track between the two handles, which then just
        # jumps that nearer handle straight to the click point. A common,
        # low-surprise range-slider convention that needs no extra "what if
        # you click between the handles" special-casing.
        self._dragging = "low" if abs(x - x_lo) <= abs(x - x_hi) else "high"
        self._drag_to(x)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._dragging is None:
            return
        self._drag_to(event.position().x())

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if self._dragging is not None:
            self._dragging = None
            self.update()
            self.rangeCommitted.emit(self._low, self._high)

    def _drag_to(self, x: float):
        value = self._value(x)
        if self._dragging == "low":
            self._low = min(value, self._high)
        else:
            self._high = max(value, self._low)
        self.update()
        self.rangeChanged.emit(self._low, self._high)


# ---------------------------------------------------------------------------
# Analyse-tab building blocks
# ---------------------------------------------------------------------------

class _IndexRangeSelector(QtWidgets.QWidget):
    """A RangeSlider plus two synced spin boxes, a Reset button and a status
    label: pick a [from, to] window of point indices. The same composite as
    the Plot tab's crop panel, packaged as a widget because the Analyse tab
    needs one per curve. (The Plot tab's crop panel itself keeps its own
    original wiring -- deliberately not refactored onto this widget, to
    avoid touching already-working code.)

    rangeChanged(lo, hi) fires after every change, whether it came from
    dragging the slider, typing in a spin box, or Reset."""

    rangeChanged = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.slider = RangeSlider()
        layout.addWidget(self.slider)

        spin_row = QtWidgets.QHBoxLayout()
        spin_row.addWidget(QtWidgets.QLabel("From point:"))
        self.lo_spin = QtWidgets.QSpinBox()
        spin_row.addWidget(self.lo_spin)
        spin_row.addWidget(QtWidgets.QLabel("to:"))
        self.hi_spin = QtWidgets.QSpinBox()
        spin_row.addWidget(self.hi_spin)
        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.setToolTip("Use the whole (cropped) curve for the fit again.")
        reset_btn.clicked.connect(self._on_reset)
        spin_row.addWidget(reset_btn)
        spin_row.addStretch(1)
        layout.addLayout(spin_row)

        self.slider.rangeChanged.connect(self._on_slider_changed)
        self.lo_spin.valueChanged.connect(self._on_spin_changed)
        self.hi_spin.valueChanged.connect(self._on_spin_changed)

    def _block(self, blocked: bool):
        for w in (self.slider, self.lo_spin, self.hi_spin):
            w.blockSignals(blocked)

    def setPointCount(self, n: int):
        top = max(0, n - 1)
        self._block(True)
        self.slider.setRange(0, top)
        self.lo_spin.setRange(0, top)
        self.hi_spin.setRange(0, top)
        self._block(False)

    def setValues(self, lo: int, hi: int):
        self._block(True)
        self.slider.setValues(lo, hi)
        lo, hi = self.slider.values()
        self.lo_spin.setValue(lo)
        self.hi_spin.setValue(hi)
        self._block(False)

    def values(self) -> tuple[int, int]:
        return self.slider.values()

    def setStatusText(self, text: str):
        self.status_label.setText(text)

    def _on_slider_changed(self, lo: int, hi: int):
        for spin, value in ((self.lo_spin, lo), (self.hi_spin, hi)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.rangeChanged.emit(lo, hi)

    def _on_spin_changed(self, _value=None):
        lo, hi = self.lo_spin.value(), self.hi_spin.value()
        if hi < lo:
            lo, hi = hi, lo
        self.slider.blockSignals(True)
        self.slider.setValues(lo, hi)
        self.slider.blockSignals(False)
        self.rangeChanged.emit(lo, hi)

    def _on_reset(self):
        self.setValues(self.slider.minimum(), self.slider.maximum())
        self.rangeChanged.emit(*self.slider.values())


class _CurveFitCard(QtWidgets.QGroupBox):
    """One curve's independent fit settings in the Analyse tab: model type,
    polynomial degree (only shown for Polynomial), whether to weight by the
    curve's own measurement-uncertainty ("_std") column, and the window of
    its (already cropped) points the fit is computed on -- chosen with a
    double-handle slider. The card itself is just a settings container;
    MainWindow._replot_analyse() computes the fit and fills in the status/
    result texts."""

    changed = Signal()

    def __init__(self, curve_index: int, curve: CurveSpec, n_points: int, has_std: bool,
                 group_names: list[str] | None = None, parent=None):
        super().__init__(curve.label, parent)
        self.curve_index = curve_index

        swatch = f"QGroupBox {{ border: 2px solid {curve.color}; border-radius: 6px; margin-top: 10px; padding-top: 8px; }}" \
                 f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}"
        self.setStyleSheet(swatch)

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        # Which regression this curve belongs to: its own, or a shared fit group.
        self.group_combo = QtWidgets.QComboBox()
        self.group_combo.addItem("Individual (own model below)", "")
        for name in (group_names or []):
            self.group_combo.addItem(f"Group: {name}", name)
        gidx = self.group_combo.findData(curve.fit_group)
        self.group_combo.setCurrentIndex(gidx if gidx >= 0 else 0)
        self.group_combo.setToolTip("Pool this curve's fit window with the other curves of a fit group "
                                    "(one shared regression, one line + band). Create groups in the "
                                    "\"Fit groups\" box above.")
        self.group_combo.currentIndexChanged.connect(self._on_group_changed)
        form.addRow("Fit with:", self.group_combo)

        self.model_combo = QtWidgets.QComboBox()
        for display_label, code in FIT_KINDS:
            self.model_combo.addItem(display_label, code)
        idx = self.model_combo.findData(curve.fit_kind)
        self.model_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        form.addRow("Model:", self.model_combo)

        self.degree_label = QtWidgets.QLabel("Degree:")
        self.degree_spin = QtWidgets.QSpinBox()
        self.degree_spin.setRange(1, 8)
        self.degree_spin.setValue(max(1, min(int(curve.fit_degree), 8)))
        self.degree_spin.valueChanged.connect(lambda _v: self.changed.emit())
        form.addRow(self.degree_label, self.degree_spin)

        self.weight_check = QtWidgets.QCheckBox("Weight by measurement uncertainty (±σ / \"_std\" column)")
        self.weight_check.setEnabled(has_std)
        self.weight_check.setChecked(bool(curve.fit_weighted) and has_std)
        if not has_std:
            self.weight_check.setToolTip("This curve's file has no \"_std\" column for its Y quantity.")
        self.weight_check.toggled.connect(lambda _c: self.changed.emit())
        form.addRow("", self.weight_check)
        layout.addLayout(form)

        layout.addWidget(QtWidgets.QLabel("Fit region (points of the cropped curve):"))
        self.region = _IndexRangeSelector()
        self.region.setPointCount(n_points)
        lo, hi = resolve_crop(n_points, curve.fit_lo, curve.fit_hi)
        self.region.setValues(lo, hi)
        self.region.rangeChanged.connect(lambda _lo, _hi: self.changed.emit())
        layout.addWidget(self.region)

        self.result_label = QtWidgets.QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.result_label)

        self._update_degree_visibility()

    def _on_model_changed(self, _index=None):
        self._update_degree_visibility()
        self.changed.emit()

    def _on_group_changed(self, _index=None):
        self._update_degree_visibility()
        self.changed.emit()

    def in_group(self) -> bool:
        return bool(self.group_combo.currentData())

    def _update_degree_visibility(self):
        grouped = self.in_group()
        # In a group the model/degree belong to the GROUP; the individual
        # model is inert (windows and weighting stay per-curve).
        self.model_combo.setEnabled(not grouped)
        is_poly = self.model_combo.currentData() == "polynomial"
        self.degree_label.setVisible(is_poly)
        self.degree_spin.setVisible(is_poly)
        self.degree_spin.setEnabled(not grouped)

    def apply_to(self, curve: CurveSpec):
        """Writes this card's widget state back into the CurveSpec it
        represents, keeping CurveSpec the single source of truth (and what
        a saved session stores). A window reaching the last point is stored
        as the -1 "to the end" sentinel, like crop_hi."""
        curve.fit_group = self.group_combo.currentData() or ""
        curve.fit_kind = self.model_combo.currentData()
        curve.fit_degree = self.degree_spin.value()
        curve.fit_weighted = self.weight_check.isChecked()
        lo, hi = self.region.values()
        curve.fit_lo = lo
        curve.fit_hi = -1 if hi >= self.region.slider.maximum() else hi

    def setResultText(self, text: str):
        self.result_label.setText(text)


class _FitGroupCard(QtWidgets.QGroupBox):
    """Settings of one FitGroup: name, model (+ degree), line/band colour
    (auto = colour of the first member curve) and a delete button. The
    result text (equation, R², per-curve spread) is filled in by
    MainWindow._replot_analyse()."""

    changed = Signal()
    renamed = Signal(str, str)        # old name, new name (already made unique by the caller if needed)
    delete_requested = Signal(str)    # group name
    colorChanged = Signal()

    def __init__(self, group: FitGroup, parent=None):
        super().__init__("Fit group", parent)
        self.group_name = group.name
        self._color = group.color
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        name_row = QtWidgets.QHBoxLayout()
        self.name_edit = QtWidgets.QLineEdit(group.name)
        self.name_edit.editingFinished.connect(self._on_name_edited)
        name_row.addWidget(self.name_edit, 1)
        self.delete_btn = QtWidgets.QPushButton("Delete")
        self.delete_btn.clicked.connect(lambda _c=False: self.delete_requested.emit(self.group_name))
        name_row.addWidget(self.delete_btn)
        form.addRow("Name:", name_row)

        self.model_combo = QtWidgets.QComboBox()
        for display_label, code in FIT_KINDS:
            if code != "none":
                self.model_combo.addItem(display_label, code)
        idx = self.model_combo.findData(group.kind)
        self.model_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        form.addRow("Model:", self.model_combo)

        self.degree_label = QtWidgets.QLabel("Degree:")
        self.degree_spin = QtWidgets.QSpinBox()
        self.degree_spin.setRange(1, 8)
        self.degree_spin.setValue(max(1, min(int(group.degree), 8)))
        self.degree_spin.valueChanged.connect(lambda _v: self.changed.emit())
        form.addRow(self.degree_label, self.degree_spin)

        color_row = QtWidgets.QHBoxLayout()
        self.color_btn = QtWidgets.QPushButton()
        self.color_btn.setToolTip("Colour of this group's line and band. Auto = colour of the first curve of the group.")
        self.color_btn.clicked.connect(self._pick_color)
        color_row.addWidget(self.color_btn)
        auto_btn = QtWidgets.QPushButton("Auto")
        auto_btn.clicked.connect(self._reset_color)
        color_row.addWidget(auto_btn)
        color_row.addStretch(1)
        form.addRow("Colour:", color_row)
        layout.addLayout(form)

        self.result_label = QtWidgets.QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.result_label)

        self._update_color_button()
        self._update_degree_visibility()

    def _update_degree_visibility(self):
        is_poly = self.model_combo.currentData() == "polynomial"
        self.degree_label.setVisible(is_poly)
        self.degree_spin.setVisible(is_poly)

    def _update_color_button(self):
        if self._color:
            self.color_btn.setText(self._color)
            self.color_btn.setStyleSheet(f"QPushButton {{ border-left: 14px solid {self._color}; padding-left: 6px; }}")
        else:
            self.color_btn.setText("auto (first curve)")
            self.color_btn.setStyleSheet("")

    def _on_model_changed(self, _index=None):
        self._update_degree_visibility()
        self.changed.emit()

    def _on_name_edited(self):
        new = self.name_edit.text().strip()
        if not new or new == self.group_name:
            self.name_edit.setText(self.group_name)
            return
        old, self.group_name = self.group_name, new
        self.renamed.emit(old, new)

    def _pick_color(self):
        initial = QtGui.QColor(self._color) if self._color else QtGui.QColor("#1f77b4")
        chosen = QtWidgets.QColorDialog.getColor(initial, self, "Group colour")
        if chosen.isValid():
            self._color = chosen.name()
            self._update_color_button()
            self.changed.emit()

    def _reset_color(self, _checked=False):
        self._color = ""
        self._update_color_button()
        self.changed.emit()

    def apply_to(self, group: FitGroup):
        group.kind = self.model_combo.currentData()
        group.degree = self.degree_spin.value()
        group.color = self._color

    def setResultText(self, text: str):
        self.result_label.setText(text)


# ---------------------------------------------------------------------------
# Small reusable dialog: pick an (x, y) pair from one file's columns
# ---------------------------------------------------------------------------

def _fill_key_combo(combo: QtWidgets.QComboBox, keys: list[str], keep_if_possible: str | None = None):
    """Repopulates a QComboBox with one item per column key, showing the
    pretty label as the text and the RAW key as that item's userData (a
    native QComboBox feature) -- so callers read back the raw key with
    combo.currentData() and never have to parse the displayed text, with
    no risk of ambiguity even if two different keys happened to prettify
    to the same string."""
    combo.blockSignals(True)
    combo.clear()
    for k in keys:
        combo.addItem(pretty_label(k), k)
    if keep_if_possible in keys:
        combo.setCurrentIndex(keys.index(keep_if_possible))
    elif keys:
        combo.setCurrentIndex(0)
    combo.blockSignals(False)


class CurveDialog(QtWidgets.QDialog):
    """Add or edit a single curve. `app` is the main window (used to read
    the currently-loaded files and to suggest an unused color)."""

    def __init__(self, app: "MainWindow", existing: CurveSpec | None = None):
        super().__init__(app)
        self.app = app
        self.result: CurveSpec | None = None
        self.setWindowTitle("Edit curve" if existing else "Add curve")

        form = QtWidgets.QFormLayout()

        self.file_combo = QtWidgets.QComboBox()
        self.file_combo.addItems([f.label for f in app.files])
        self.file_combo.currentIndexChanged.connect(self._on_file_changed)
        form.addRow("File / test:", self.file_combo)

        self.x_combo = QtWidgets.QComboBox()
        self.x_combo.currentIndexChanged.connect(self._on_xy_changed)
        form.addRow("X axis:", self.x_combo)

        self.y_combo = QtWidgets.QComboBox()
        self.y_combo.currentIndexChanged.connect(self._on_xy_changed)
        form.addRow("Y axis:", self.y_combo)

        self.label_edit = QtWidgets.QLineEdit()
        form.addRow("Label:", self.label_edit)

        color_row = QtWidgets.QHBoxLayout()
        self._color = existing.color if existing else next_color(app.used_colors())
        self.swatch = QtWidgets.QLabel()
        self.swatch.setFixedSize(28, 20)
        self._update_swatch()
        pick_btn = QtWidgets.QPushButton("Choose...")
        pick_btn.clicked.connect(self._pick_color)
        color_row.addWidget(self.swatch)
        color_row.addWidget(pick_btn)
        color_row.addStretch(1)
        form.addRow("Color:", color_row)

        self.band_check = QtWidgets.QCheckBox("Show uncertainty band (needs a matching \"_std\" column)")
        form.addRow("", self.band_check)

        # Line style and width are per-curve (each curve keeps its own,
        # independent of every other curve's) -- unlike the band's sigma
        # multiplier/opacity, which stay plot-wide settings since they're
        # about how uncertainty is visualized in general, not a per-curve
        # identity the way style/width/color are.
        self.linestyle_combo = QtWidgets.QComboBox()
        for display_label, _code in LINESTYLE_CHOICES:
            self.linestyle_combo.addItem(display_label, _code)
        form.addRow("Line style:", self.linestyle_combo)

        self.linewidth_spin = QtWidgets.QDoubleSpinBox()
        self.linewidth_spin.setRange(0.5, 6.0)
        self.linewidth_spin.setSingleStep(0.25)
        self.linewidth_spin.setValue(existing.linewidth if existing else 1.8)
        form.addRow("Line width:", self.linewidth_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        # Pre-fill from `existing`, or sensible defaults for a brand-new curve.
        if existing is not None:
            idx = self.file_combo.findText(existing.file_label)
            if idx >= 0:
                self.file_combo.setCurrentIndex(idx)
            self._refresh_keys(keep_x=existing.x_key, keep_y=existing.y_key)
            self.label_edit.setText(existing.label)
            self.band_check.setChecked(existing.show_band)
            self._update_band_availability()
            style_idx = self.linestyle_combo.findData(existing.linestyle)
            self.linestyle_combo.setCurrentIndex(style_idx if style_idx >= 0 else 0)
        elif app.files:
            self.file_combo.setCurrentIndex(len(app.files) - 1)  # most-recently-loaded file by default
            self._refresh_keys()
            self._suggest_label()

    def _current_file(self) -> LoadedFile | None:
        return self.app.files_by_label.get(self.file_combo.currentText())

    def _refresh_keys(self, keep_x: str | None = None, keep_y: str | None = None):
        f = self._current_file()
        keys = f.plottable_keys if f else []
        _fill_key_combo(self.x_combo, keys, keep_if_possible=keep_x)
        _fill_key_combo(self.y_combo, keys, keep_if_possible=keep_y)
        self._update_band_availability()

    def _on_file_changed(self, _index=None):
        self._refresh_keys()
        self._suggest_label()

    def _on_xy_changed(self, _index=None):
        self._update_band_availability()
        self._suggest_label()

    def _update_band_availability(self):
        f = self._current_file()
        y_key = self.y_combo.currentData()
        available = bool(f and y_key and f"{y_key}_std" in f.columns)
        self.band_check.setEnabled(available)
        if not available:
            self.band_check.setChecked(False)

    def _suggest_label(self):
        f = self._current_file()
        y_key = self.y_combo.currentData()
        if f and y_key:
            self.label_edit.setText(f"{f.label} — {pretty_label(y_key)}")

    def _update_swatch(self):
        self.swatch.setStyleSheet(f"background-color: {self._color}; border: 1px solid #888;")

    def _pick_color(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._color), self, "Curve color")
        if color.isValid():
            self._color = color.name()
            self._update_swatch()

    def _on_ok(self):
        f = self._current_file()
        x_key, y_key = self.x_combo.currentData(), self.y_combo.currentData()
        if f is None or x_key is None or y_key is None:
            QtWidgets.QMessageBox.warning(self, "Incomplete", "Pick a file, an X axis and a Y axis first.")
            return
        label = self.label_edit.text().strip() or f"{f.label} — {pretty_label(y_key)}"
        self.result = CurveSpec(
            file_label=f.label, x_key=x_key, y_key=y_key, label=label,
            color=self._color, show_band=self.band_check.isChecked(),
            linestyle=self.linestyle_combo.currentData(), linewidth=self.linewidth_spin.value(),
        )
        self.accept()


# ---------------------------------------------------------------------------
# Quick-add dialogs matching the two comparison styles shown in the request
# ---------------------------------------------------------------------------

class QuickAddAcrossFilesDialog(QtWidgets.QDialog):
    """"Same quantity across several tests": pick X/Y once, add one curve
    per selected file, each labeled with that file's own name."""

    def __init__(self, app: "MainWindow", files: list[LoadedFile]):
        super().__init__(app)
        self.app = app
        self.files = files
        self.result: list[CurveSpec] | None = None
        self.setWindowTitle("Quick add: same quantity across tests")

        layout = QtWidgets.QVBoxLayout(self)
        names = ", ".join(f.label for f in files) or "(none selected)"
        names_label = QtWidgets.QLabel(f"Tests ({len(files)}): {names}")
        names_label.setWordWrap(True)
        layout.addWidget(names_label)

        form = QtWidgets.QFormLayout()
        common = self._common_keys()
        self.x_combo = QtWidgets.QComboBox()
        _fill_key_combo(self.x_combo, common)
        self.x_combo.currentIndexChanged.connect(self._update_band_availability)
        form.addRow("X axis:", self.x_combo)

        self.y_combo = QtWidgets.QComboBox()
        _fill_key_combo(self.y_combo, common)
        self.y_combo.currentIndexChanged.connect(self._update_band_availability)
        form.addRow("Y axis:", self.y_combo)
        layout.addLayout(form)

        self.band_check = QtWidgets.QCheckBox("Show uncertainty band on every curve (where available)")
        layout.addWidget(self.band_check)

        if not common:
            warn = QtWidgets.QLabel("These files have no column in common -- nothing to quick-add.")
            warn.setStyleSheet("color: #b00020;")
            warn.setWordWrap(True)
            layout.addWidget(warn)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Add curves")
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(bool(common))
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_band_availability()

    def _common_keys(self) -> list[str]:
        if not self.files:
            return []
        common = set(self.files[0].plottable_keys)
        for f in self.files[1:]:
            common &= set(f.plottable_keys)
        # Keep a stable, sensible order (first file's own column order).
        return [k for k in self.files[0].plottable_keys if k in common]

    def _update_band_availability(self, *_args):
        y_key = self.y_combo.currentData()
        available = bool(y_key) and any(f"{y_key}_std" in f.columns for f in self.files)
        self.band_check.setEnabled(available)
        if not available:
            self.band_check.setChecked(False)

    def _on_ok(self):
        x_key, y_key = self.x_combo.currentData(), self.y_combo.currentData()
        if not x_key or not y_key:
            QtWidgets.QMessageBox.warning(self, "Incomplete", "Pick an X and a Y axis first.")
            return
        used = self.app.used_colors()
        curves = []
        skipped = []
        for f in self.files:
            if x_key not in f.columns or y_key not in f.columns:
                skipped.append(f.label)
                continue
            color = next_color(used)
            used.append(color)
            show_band = self.band_check.isChecked() and f"{y_key}_std" in f.columns
            curves.append(CurveSpec(file_label=f.label, x_key=x_key, y_key=y_key,
                                     label=f.label, color=color, show_band=show_band))
        if skipped:
            QtWidgets.QMessageBox.information(
                self, "Some tests skipped",
                "These tests don't have both chosen columns and were skipped:\n" + "\n".join(skipped),
            )
        self.result = curves
        self.accept()


class QuickAddWithinFileDialog(QtWidgets.QDialog):
    """"Several quantities from the same test": pick one file and an X
    axis once, then multi-select every Y quantity to overlay."""

    def __init__(self, app: "MainWindow", preselect: LoadedFile | None = None):
        super().__init__(app)
        self.app = app
        self.result: list[CurveSpec] | None = None
        self.setWindowTitle("Quick add: multiple quantities from one test")

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.file_combo = QtWidgets.QComboBox()
        self.file_combo.addItems([f.label for f in app.files])
        self.file_combo.currentIndexChanged.connect(self._refresh)
        form.addRow("File / test:", self.file_combo)

        self.x_combo = QtWidgets.QComboBox()
        self.x_combo.currentIndexChanged.connect(self._refresh_y_list)
        form.addRow("X axis:", self.x_combo)
        layout.addLayout(form)

        layout.addWidget(QtWidgets.QLabel("Y quantities (ctrl/shift-click, or drag, to pick several):"))
        self.y_list = QtWidgets.QListWidget()
        self.y_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.y_list.setMinimumHeight(220)
        layout.addWidget(self.y_list)

        self.band_check = QtWidgets.QCheckBox("Show uncertainty band on every curve (where available)")
        layout.addWidget(self.band_check)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Add curves")
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if preselect is not None:
            idx = self.file_combo.findText(preselect.label)
            if idx >= 0:
                self.file_combo.setCurrentIndex(idx)
        elif app.files:
            self.file_combo.setCurrentIndex(len(app.files) - 1)
        self._refresh()

    def _current_file(self) -> LoadedFile | None:
        return self.app.files_by_label.get(self.file_combo.currentText())

    def _refresh(self, *_args):
        f = self._current_file()
        keys = f.plottable_keys if f else []
        default_x = next((k for k in ("strain", "nominal_strain", "confinement_strain", "time") if k in keys), None)
        _fill_key_combo(self.x_combo, keys, keep_if_possible=default_x)
        self._refresh_y_list()

    def _refresh_y_list(self, *_args):
        f = self._current_file()
        x_key = self.x_combo.currentData()
        self.y_list.clear()
        if f is None:
            return
        for k in f.plottable_keys:
            if k == x_key:
                continue
            item = QtWidgets.QListWidgetItem(pretty_label(k))
            item.setData(Qt.ItemDataRole.UserRole, k)
            self.y_list.addItem(item)

    def _on_ok(self):
        f = self._current_file()
        x_key = self.x_combo.currentData()
        selected = self.y_list.selectedItems()
        if f is None or x_key is None or not selected:
            QtWidgets.QMessageBox.warning(self, "Incomplete", "Pick a file, an X axis, and at least one Y quantity.")
            return
        used = self.app.used_colors()
        curves = []
        for item in selected:
            y_key = item.data(Qt.ItemDataRole.UserRole)
            color = next_color(used)
            used.append(color)
            show_band = self.band_check.isChecked() and f"{y_key}_std" in f.columns
            curves.append(CurveSpec(file_label=f.label, x_key=x_key, y_key=y_key,
                                     label=pretty_label(y_key), color=color, show_band=show_band))
        self.result = curves
        self.accept()


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Format templates: every formatting parameter of the Plot tab (and each
# curve's look), saved to / loaded from a small .json file, so a figure style
# can be re-applied to other curves. Pure data helpers first (no Qt).
# ---------------------------------------------------------------------------

TEMPLATE_KIND = "hydra_curve_plotter_template"

# Defaults of every formatting setting -- also what an old session file
# without some key falls back to.
FORMAT_DEFAULTS: dict = {
    "title": "", "xlabel": "", "ylabel": "",
    "legend_loc": "best", "grid": True, "sigma": 2.0, "alpha": 0.22,
    "axislabel_fontsize": 20, "legend_fontsize": 14, "tick_fontsize": 14, "title_fontsize": 14,
    "xmin": "", "xmax": "", "ymin": "", "ymax": "",
    "flip_x": False, "flip_y": False, "zero_x": False, "zero_y": False, "factor_x": "1", "factor_y": "1",
    "norm_x": "", "norm_y": "",
}

# What a template can apply, in groups the user ticks when loading one:
# (group id, label shown in the dialog, the FORMAT_DEFAULTS keys it covers).
# The "curves" group has no keys: it is the per-curve style list.
TEMPLATE_GROUPS: list[tuple[str, str, list[str]]] = [
    ("plot", "Legend, grid, band σ/opacity and font sizes",
     ["legend_loc", "grid", "sigma", "alpha", "axislabel_fontsize", "legend_fontsize",
      "tick_fontsize", "title_fontsize"]),
    ("transform", "Data transform (invert sign / start at 0 / factor / normalise)",
     ["flip_x", "flip_y", "zero_x", "zero_y", "factor_x", "factor_y", "norm_x", "norm_y"]),
    ("limits", "Axis limits", ["xmin", "xmax", "ymin", "ymax"]),
    ("text", "Title and axis label texts", ["title", "xlabel", "ylabel"]),
    ("curves", "Curve styles (colour, line style/width, band, shown) - by list order", []),
    ("labels", "Legend labels (curve names) - by list order", []),
]

# CurveSpec fields that are pure appearance (not data identity, crop or fit).
CURVE_STYLE_FIELDS = ("color", "linestyle", "linewidth", "show_band", "visible")


def curve_style_dict(curve: "CurveSpec") -> dict:
    """One curve's template entry: its appearance fields plus its legend
    label (the label is stored always, but only applied if the user ticks
    "Legend labels" when loading)."""
    d = {k: getattr(curve, k) for k in CURVE_STYLE_FIELDS}
    d["label"] = curve.label
    return d


def make_template(settings: dict, curves: list) -> dict:
    """The template file's content: the formatting settings plus one style
    entry per curve, in list order."""
    return {
        "kind": TEMPLATE_KIND, "format_version": 1,
        "settings": {k: settings[k] for k in FORMAT_DEFAULTS if k in settings},
        "curve_styles": [curve_style_dict(c) for c in curves],
    }


def template_available_groups(template: dict) -> list[str]:
    """Group ids this template actually contains something for."""
    settings = template.get("settings", {}) if isinstance(template, dict) else {}
    out = []
    for gid, _label, keys in TEMPLATE_GROUPS:
        if gid == "curves":
            if template.get("curve_styles"):
                out.append(gid)
        elif gid == "labels":
            # Templates saved before labels were stored simply don't offer it.
            if any(isinstance(st, dict) and st.get("label") for st in template.get("curve_styles", [])):
                out.append(gid)
        elif any(k in settings for k in keys):
            out.append(gid)
    return out


def template_settings_for_groups(template: dict, groups) -> dict:
    """The settings sub-dict covered by the chosen groups (only keys the
    template really has)."""
    settings = template.get("settings", {})
    keys = {k for gid, _l, ks in TEMPLATE_GROUPS if gid in groups for k in ks}
    return {k: settings[k] for k in keys if k in settings}


def restyle_curves(curves: list, styles: list, apply_style: bool = True, apply_labels: bool = False) -> int:
    """Applies the template's entries to the loaded curves BY LIST ORDER
    (curve i gets entry i); curves beyond the template's length are left
    untouched. `apply_style` covers colour/line style/width/band/shown,
    `apply_labels` the legend label (an empty/missing label never overwrites
    one). A malformed entry/field is skipped rather than crashing. Returns
    the number of curves modified."""
    n = 0
    for curve, st in zip(curves, styles):
        if not isinstance(st, dict):
            continue
        changed = False
        if apply_labels and isinstance(st.get("label"), str) and st["label"].strip():
            curve.label = st["label"].strip()
            changed = True
        if not apply_style:
            n += changed
            continue
        try:
            if "color" in st and isinstance(st["color"], str):
                curve.color = st["color"]
            if "linestyle" in st and isinstance(st["linestyle"], str):
                curve.linestyle = st["linestyle"]
            if "linewidth" in st:
                curve.linewidth = max(0.1, float(st["linewidth"]))
            if "show_band" in st:
                curve.show_band = bool(st["show_band"])
            if "visible" in st:
                curve.visible = bool(st["visible"])
        except (TypeError, ValueError):
            n += changed
            continue
        n += 1
    return n


class TemplateApplyDialog(QtWidgets.QDialog):
    """Asks which parts of a loaded template to apply (a style may be wanted
    without its title, or without its axis limits...). `selected` holds the
    ticked group ids after exec() returns accepted."""

    def __init__(self, parent, available: list[str], n_template_curves: int, n_loaded_curves: int):
        super().__init__(parent)
        self.setWindowTitle("Apply template")
        self.selected: set[str] = set()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Apply which parts of the template to the current figure?"))
        self._checks: dict[str, QtWidgets.QCheckBox] = {}
        for gid, label, _keys in TEMPLATE_GROUPS:
            if gid not in available:
                continue
            chk = QtWidgets.QCheckBox(label)
            # Texts, limits and legend labels describe one specific dataset: offered, but off by default.
            chk.setChecked(gid not in ("text", "limits", "labels"))
            self._checks[gid] = chk
            layout.addWidget(chk)
        if "curves" in available or "labels" in available:
            note = QtWidgets.QLabel(
                f"The template holds {n_template_curves} curve entr{'y' if n_template_curves == 1 else 'ies'}; "
                f"{n_loaded_curves} curve(s) are loaded. Curve 1 gets entry 1, and so on; extra curves are left unchanged."
            )
            note.setWordWrap(True)
            note.setStyleSheet("color: #666; font-style: italic;")
            layout.addWidget(note)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_ok(self):
        self.selected = {gid for gid, chk in self._checks.items() if chk.isChecked()}
        self.accept()


# Curve table column indices, named so the rest of the class never has
# magic numbers scattered through it.
COL_SHOW, COL_FILE, COL_X, COL_Y, COL_LABEL, COL_BAND, COL_STYLE, COL_WIDTH, COL_COLOR = range(9)


def export_size_inches(width, height, unit, dpi, current_size):
    """Exported figure size in inches from the width/height boxes.
    width/height <= 0 mean "auto": None if both are auto (save as shown);
    if only one is given the other follows the current on-screen aspect
    ratio. `unit` is "cm", "in" or "px" (px converted with `dpi`)."""
    if width <= 0 and height <= 0:
        return None
    factor = {"cm": 1.0 / 2.54, "in": 1.0, "px": 1.0 / max(float(dpi), 1.0)}[unit]
    cw, ch = float(current_size[0]), float(current_size[1])
    aspect = ch / cw if cw > 0 else 1.0
    w_in = width * factor if width > 0 else height * factor / aspect
    h_in = height * factor if height > 0 else width * factor * aspect
    if not (0.1 <= w_in <= 200 and 0.1 <= h_in <= 200):
        raise ValueError(f"Requested size {w_in:.2f} x {h_in:.2f} in is out of range (0.1 - 200 in per side).")
    return (w_in, h_in)


def save_figure_sized(fig, path, dpi, size_in):
    """savefig at `size_in` (inches) -- the layout is recomputed for that
    size and the on-screen figure is restored afterwards. With
    size_in=None, behaves as before (as displayed, bbox tight). The exact
    size is kept (no tight cropping) so the file has the requested
    dimensions: w_in*dpi x h_in*dpi pixels."""
    if size_in is None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return
    old = tuple(fig.get_size_inches())
    try:
        fig.set_size_inches(size_in[0], size_in[1], forward=False)
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 - a too-small figure can't be tightened; save anyway
            pass
        fig.savefig(path, dpi=dpi)
    finally:
        fig.set_size_inches(old[0], old[1], forward=False)
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001
            pass



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HYDRA Curve Plotter")
        self.resize(1320, 840)
        self.setMinimumSize(1000, 640)

        self.files: list[LoadedFile] = []
        self.files_by_label: dict[str, LoadedFile] = {}
        self.curves: list[CurveSpec] = []
        self._xlabel_auto = True
        self._ylabel_auto = True
        self._title_auto = True
        self._updating_table = False  # guards against itemChanged firing while we rebuild the table ourselves
        self._crop_curve_index: int | None = None

        self._replot_timer = QtCore.QTimer(self)
        self._replot_timer.setSingleShot(True)
        self._replot_timer.setInterval(40)  # ~25 fps -- live preview while dragging the crop slider, without redrawing on every single pixel of motion
        self._replot_timer.timeout.connect(self.replot)

        # Analyse tab: same debounce idea, for its own (separate) canvas.
        self._fit_cards: list[_CurveFitCard] = []
        self._group_cards: list[_FitGroupCard] = []
        self.fit_groups: list[FitGroup] = []
        self._analyse_timer = QtCore.QTimer(self)
        self._analyse_timer.setSingleShot(True)
        self._analyse_timer.setInterval(60)
        self._analyse_timer.timeout.connect(self._replot_analyse)

        self._build_ui()
        self._refresh_file_list()
        self._refresh_curve_table()
        self._refresh_crop_panel()
        self._refresh_analyse_cards()
        self.replot()

    # -- color bookkeeping -------------------------------------------------
    def used_colors(self) -> list[str]:
        return [c.color for c in self.curves]

    # -- a single place both replot() and the crop panel read a curve's data from, so they can never disagree -------
    def _curve_source(self, curve: CurveSpec):
        """Returns (file, x_full, y_full) for a curve's own file/columns
        (already truncated to their common length), or None if the file
        isn't loaded or doesn't have both chosen columns."""
        f = self.files_by_label.get(curve.file_label)
        if f is None or curve.x_key not in f.columns or curve.y_key not in f.columns:
            return None
        x = np.asarray(f.columns[curve.x_key])
        y = np.asarray(f.columns[curve.y_key])
        n = min(len(x), len(y))
        return f, x[:n], y[:n]

    def _sync_after_curves_changed(self):
        """Every place that adds/edits/removes/reorders curves, or
        swaps out which files back them, needs to refresh all of these
        together -- bundled here once so a future change can't forget one
        of them (as a scattered call-everywhere pattern risks)."""
        self._refresh_curve_table()
        self._refresh_crop_panel()
        self._refresh_analyse_cards()
        self.replot()

    # -- UI construction -----------------------------------------------------
    def _build_ui(self):
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(False)
        self.tabs.setStyleSheet(self._TAB_STYLE)
        self.tabs.addTab(self._build_plot_tab(), "Plot")
        self._analyse_tab = self._build_analyse_tab()
        self.tabs.addTab(self._analyse_tab, "Analyse")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        self.statusBar().showMessage("Load one or more exported .csv/.txt/.xlsx files to begin.")

    # Rounded, dark-friendly tab bar in the spirit of HYDRA's own: built only
    # from palette roles (so it follows the light/dark theme instead of
    # hard-coding colours); the selected tab takes the highlight colour and
    # the others sit slightly lower, which gives the "raised tab" look.
    _TAB_STYLE = """
        QTabWidget::pane { border: 1px solid palette(mid); border-radius: 6px; top: -1px; }
        QTabBar::tab {
            padding: 7px 26px; margin-right: 3px;
            border: 1px solid palette(mid);
            border-top-left-radius: 9px; border-top-right-radius: 9px;
            background: palette(button); color: palette(buttontext);
            font-weight: 600;
        }
        QTabBar::tab:selected { background: palette(highlight); color: palette(highlightedtext); }
        QTabBar::tab:!selected { margin-top: 3px; }
        QTabBar::tab:!selected:hover { background: palette(midlight); }
    """

    def _build_plot_tab(self) -> QtWidgets.QWidget:
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)

        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(4, 4, 4, 4)
        self._build_files_panel(left_layout)
        self._build_curves_panel(left_layout)
        self._build_crop_panel(left_layout)
        self._build_transform_panel(left_layout)
        self._build_settings_panel(left_layout)
        self._build_export_panel(left_layout)
        left_layout.addStretch(1)

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidget(left_container)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(460)
        left_scroll.setMaximumWidth(620)

        right = QtWidgets.QWidget()
        self._build_plot_panel(right)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        return splitter

    def _build_analyse_tab(self) -> QtWidgets.QWidget:
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)

        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(4, 4, 4, 4)

        help_label = QtWidgets.QLabel(
            "Calibrate a simple model on the curves of the Plot tab (same crop, colours, styles, title and axis limits). "
            "Drag the double slider of a curve to choose the points the regression uses; the two small "
            "squares on the plot show where the window starts and ends. Each curve is fitted on its own, or "
            "several curves can share one fit (fit groups). "
            "With weighting on, each point counts as 1/σ (its \"_std\" column)."
        )
        help_label.setWordWrap(True)
        left_layout.addWidget(help_label)

        # --- fit groups: several curves -> ONE regression -------------------------
        gbox = QtWidgets.QGroupBox("Fit groups (several curves, one fit)")
        gl = QtWidgets.QVBoxLayout(gbox)
        gl.addWidget(QtWidgets.QLabel(
            "A group pools the fit windows of the curves assigned to it (choose \"Fit with\" on each curve "
            "card) into ONE regression, drawn as a single line + band in the group's colour. The "
            "scatter between curves is added to the measurement σ, so curves that disagree weigh less."))
        gl.itemAt(0).widget().setWordWrap(True)
        btn_row = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("+ New group")
        add_btn.clicked.connect(lambda _c=False: self._on_add_group(False))
        all_btn = QtWidgets.QPushButton("+ Group with all curves")
        all_btn.setToolTip("Creates a group and assigns every visible curve to it.")
        all_btn.clicked.connect(lambda _c=False: self._on_add_group(True))
        btn_row.addWidget(add_btn)
        btn_row.addWidget(all_btn)
        btn_row.addStretch(1)
        gl.addLayout(btn_row)
        left_layout.addWidget(gbox)

        self.group_cards_container = QtWidgets.QWidget()
        self.group_cards_layout = QtWidgets.QVBoxLayout(self.group_cards_container)
        self.group_cards_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.group_cards_container)

        # --- fit appearance (applies to every fit, individual or grouped) ----------
        abox = QtWidgets.QGroupBox("Fit appearance")
        aform = QtWidgets.QFormLayout(abox)
        self.fit_lw_spin = QtWidgets.QDoubleSpinBox()
        self.fit_lw_spin.setRange(0.5, 12.0)
        self.fit_lw_spin.setSingleStep(0.5)
        self.fit_lw_spin.setValue(2.6)
        self.fit_lw_spin.valueChanged.connect(self._on_analyse_changed)
        aform.addRow("Line width:", self.fit_lw_spin)
        self.fit_extend_combo = QtWidgets.QComboBox()
        for label, code in EXTEND_MODES:
            self.fit_extend_combo.addItem(label, code)
        self.fit_extend_combo.setCurrentIndex(self.fit_extend_combo.findData("axes"))
        self.fit_extend_combo.setToolTip("How far the fitted line (and its band) is drawn. \"Full axis\" extrapolates "
                                         "across the whole plot (use the Plot tab's axis limits to frame it).")
        self.fit_extend_combo.currentIndexChanged.connect(self._on_analyse_changed)
        aform.addRow("Draw line over:", self.fit_extend_combo)
        self.fit_band_combo = QtWidgets.QComboBox()
        for label, code in BAND_MODES:
            self.fit_band_combo.addItem(label, code)
        self.fit_band_combo.setCurrentIndex(self.fit_band_combo.findData("param"))
        self.fit_band_combo.currentIndexChanged.connect(self._on_analyse_changed)
        aform.addRow("Band:", self.fit_band_combo)
        self.fit_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.fit_sigma_spin.setRange(0.1, 10.0)
        self.fit_sigma_spin.setSingleStep(0.5)
        self.fit_sigma_spin.setValue(2.0)
        self.fit_sigma_spin.setPrefix("± ")
        self.fit_sigma_spin.setSuffix(" σ")
        self.fit_sigma_spin.valueChanged.connect(self._on_analyse_changed)
        aform.addRow("Band width:", self.fit_sigma_spin)
        self.fit_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.fit_alpha_spin.setRange(0.0, 1.0)
        self.fit_alpha_spin.setSingleStep(0.05)
        self.fit_alpha_spin.setValue(0.25)
        self.fit_alpha_spin.valueChanged.connect(self._on_analyse_changed)
        aform.addRow("Band opacity:", self.fit_alpha_spin)
        self.fit_legend_check = QtWidgets.QCheckBox("Add each fit to the legend")
        self.fit_legend_check.toggled.connect(self._on_analyse_changed)
        aform.addRow(self.fit_legend_check)
        self.fit_markers_check = QtWidgets.QCheckBox("Show the fit-window highlight and square markers")
        self.fit_markers_check.setChecked(True)
        self.fit_markers_check.toggled.connect(self._on_analyse_changed)
        aform.addRow(self.fit_markers_check)
        left_layout.addWidget(abox)

        # --- per-curve cards (rebuilt by _refresh_analyse_cards) --------------------
        self.fit_cards_container = QtWidgets.QWidget()
        self.fit_cards_layout = QtWidgets.QVBoxLayout(self.fit_cards_container)
        self.fit_cards_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.fit_cards_container)
        self._build_export_panel(left_layout, analyse=True)
        left_layout.addStretch(1)

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidget(left_container)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(460)
        left_scroll.setMaximumWidth(620)

        right = QtWidgets.QWidget()
        rlayout = QtWidgets.QVBoxLayout(right)
        rlayout.setContentsMargins(0, 0, 0, 0)
        self.analyse_fig = Figure(figsize=(7.5, 6.0), dpi=100)
        self.analyse_ax = self.analyse_fig.add_subplot(111)
        self.analyse_canvas = FigureCanvasQTAgg(self.analyse_fig)
        rlayout.addWidget(self.analyse_canvas)
        self.analyse_toolbar = NavigationToolbar2QT(self.analyse_canvas, right)
        rlayout.addWidget(self.analyse_toolbar)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        return splitter

    def _build_files_panel(self, parent_layout):
        box = QtWidgets.QGroupBox("1. Files")
        layout = QtWidgets.QVBoxLayout(box)

        self.file_list = QtWidgets.QListWidget()
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_list.setFixedHeight(120)
        layout.addWidget(self.file_list)

        btn_row = QtWidgets.QHBoxLayout()
        load_btn = QtWidgets.QPushButton("Load CSV/TXT...")
        load_btn.clicked.connect(self._on_load_files)
        rename_btn = QtWidgets.QPushButton("Rename...")
        rename_btn.clicked.connect(self._on_rename_file)
        remove_btn = QtWidgets.QPushButton("Remove")
        remove_btn.clicked.connect(self._on_remove_files)
        for b in (load_btn, rename_btn, remove_btn):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)
        parent_layout.addWidget(box)

    def _build_curves_panel(self, parent_layout):
        box = QtWidgets.QGroupBox("2. Curves")
        layout = QtWidgets.QVBoxLayout(box)

        self.curve_table = QtWidgets.QTableWidget(0, 9)
        self.curve_table.setHorizontalHeaderLabels(["Show", "File", "X", "Y", "Label", "Band", "Style", "Width", "Color"])
        self.curve_table.verticalHeader().setVisible(False)
        # ExtendedSelection + SelectRows: native click-drag across rows,
        # and ctrl/shift-click, to select several curves at once -- so
        # "Remove selected" below can delete a whole batch in one go.
        self.curve_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.curve_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        # Only the Label cells actually carry the Qt.ItemFlag.ItemIsEditable
        # flag (set per-item in _refresh_curve_table), so this only ever
        # opens inline text editing on that column -- editing a legend
        # label is then just "double-click it and type", no dialog needed.
        self.curve_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.curve_table.setColumnWidth(COL_SHOW, 44)
        self.curve_table.setColumnWidth(COL_FILE, 80)
        self.curve_table.setColumnWidth(COL_X, 80)
        self.curve_table.setColumnWidth(COL_Y, 80)
        self.curve_table.setColumnWidth(COL_LABEL, 130)
        self.curve_table.setColumnWidth(COL_BAND, 40)
        self.curve_table.setColumnWidth(COL_STYLE, 70)
        self.curve_table.setColumnWidth(COL_WIDTH, 50)
        self.curve_table.setColumnWidth(COL_COLOR, 60)
        self.curve_table.setMinimumHeight(220)
        self.curve_table.itemChanged.connect(self._on_curve_item_changed)
        self.curve_table.itemSelectionChanged.connect(self._refresh_crop_panel)
        self.curve_table.cellDoubleClicked.connect(self._on_curve_cell_double_clicked)
        layout.addWidget(self.curve_table)

        btn_row1 = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add curve...")
        add_btn.clicked.connect(self._on_add_curve)
        edit_btn = QtWidgets.QPushButton("Edit...")
        edit_btn.clicked.connect(self._on_edit_curve)
        remove_btn = QtWidgets.QPushButton("Remove selected")
        remove_btn.clicked.connect(self._on_remove_curve)
        up_btn = QtWidgets.QPushButton("Up")
        up_btn.clicked.connect(lambda: self._on_move_curve(-1))
        down_btn = QtWidgets.QPushButton("Down")
        down_btn.clicked.connect(lambda: self._on_move_curve(1))
        for b in (add_btn, edit_btn, remove_btn, up_btn, down_btn):
            btn_row1.addWidget(b)
        layout.addLayout(btn_row1)

        hint = QtWidgets.QLabel("Tip: double-click a curve's Label cell to rename it directly.")
        hint.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(hint)

        btn_row2 = QtWidgets.QHBoxLayout()
        across_btn = QtWidgets.QPushButton("Quick add: same quantity\nacross selected tests...")
        across_btn.clicked.connect(self._on_quick_add_across)
        within_btn = QtWidgets.QPushButton("Quick add: several quantities\nfrom one test...")
        within_btn.clicked.connect(self._on_quick_add_within)
        btn_row2.addWidget(across_btn)
        btn_row2.addWidget(within_btn)
        layout.addLayout(btn_row2)

        parent_layout.addWidget(box)

    def _build_crop_panel(self, parent_layout):
        box = QtWidgets.QGroupBox("3. Crop selected curve (drop points from the start/end)")
        layout = QtWidgets.QVBoxLayout(box)
        self.crop_group = box

        self.crop_status_label = QtWidgets.QLabel("Select exactly one curve above to crop its start/end.")
        self.crop_status_label.setWordWrap(True)
        layout.addWidget(self.crop_status_label)

        self.crop_slider = RangeSlider()
        self.crop_slider.rangeChanged.connect(self._on_crop_slider_changed)
        self.crop_slider.rangeCommitted.connect(self._on_crop_committed)
        layout.addWidget(self.crop_slider)

        spin_row = QtWidgets.QHBoxLayout()
        spin_row.addWidget(QtWidgets.QLabel("Keep from point:"))
        self.crop_lo_spin = QtWidgets.QSpinBox()
        self.crop_lo_spin.setRange(0, 0)
        self.crop_lo_spin.valueChanged.connect(self._on_crop_spin_changed)
        spin_row.addWidget(self.crop_lo_spin)
        spin_row.addWidget(QtWidgets.QLabel("to:"))
        self.crop_hi_spin = QtWidgets.QSpinBox()
        self.crop_hi_spin.setRange(0, 0)
        self.crop_hi_spin.valueChanged.connect(self._on_crop_spin_changed)
        spin_row.addWidget(self.crop_hi_spin)
        reset_btn = QtWidgets.QPushButton("Reset (keep all points)")
        reset_btn.clicked.connect(self._on_crop_reset)
        spin_row.addWidget(reset_btn)
        spin_row.addStretch(1)
        layout.addLayout(spin_row)

        self.crop_group.setEnabled(False)
        parent_layout.addWidget(box)

    def _build_transform_panel(self, parent_layout):
        box = QtWidgets.QGroupBox("4. Data transform (optional, display only)")
        grid = QtWidgets.QGridLayout(box)
        grid.addWidget(QtWidgets.QLabel("X axis"), 0, 1, Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(QtWidgets.QLabel("Y axis"), 0, 2, Qt.AlignmentFlag.AlignCenter)

        self.flip_x_check, self.flip_y_check = QtWidgets.QCheckBox(), QtWidgets.QCheckBox()
        self.zero_x_check, self.zero_y_check = QtWidgets.QCheckBox(), QtWidgets.QCheckBox()
        self.factor_x_edit, self.factor_y_edit = QtWidgets.QLineEdit("1"), QtWidgets.QLineEdit("1")
        for e in (self.factor_x_edit, self.factor_y_edit):
            e.setPlaceholderText("1")
            e.setMaximumWidth(90)
            e.setToolTip("Multiplier applied to the data of this axis ('.' or ',' decimals, or e.g. 1e-3).")
            e.textChanged.connect(self._on_transform_changed)

        self.norm_x_edit, self.norm_y_edit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for e in (self.norm_x_edit, self.norm_y_edit):
            e.setPlaceholderText("none")
            e.setMaximumWidth(90)
            e.setToolTip("Normalisation: the data of this axis are DIVIDED by this reference value "
                         "(e.g. a yield stress or strain). Blank, 0 or invalid = no normalisation. "
                         "'.' or ',' decimals, or e.g. 1e-3.")
            e.textChanged.connect(self._on_transform_changed)

        rows = [
            ("Invert sign", self.flip_x_check, self.flip_y_check),
            ("Start at 0 (subtract first value)", self.zero_x_check, self.zero_y_check),
            ("Multiply by", self.factor_x_edit, self.factor_y_edit),
            ("Normalise: divide by", self.norm_x_edit, self.norm_y_edit),
        ]
        for r, (text, wx, wy) in enumerate(rows, start=1):
            grid.addWidget(QtWidgets.QLabel(text), r, 0)
            grid.addWidget(wx, r, 1, Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(wy, r, 2, Qt.AlignmentFlag.AlignCenter)
        for c in (self.flip_x_check, self.flip_y_check, self.zero_x_check, self.zero_y_check):
            c.toggled.connect(self._on_transform_changed)
        # Axis limits are given in DISPLAYED units, so inverting an axis
        # mirrors its limit fields too (see _mirror_axis_limits).
        self.flip_x_check.toggled.connect(lambda _c: self._mirror_axis_limits(self.xmin_edit, self.xmax_edit))
        self.flip_y_check.toggled.connect(lambda _c: self._mirror_axis_limits(self.ymin_edit, self.ymax_edit))

        note = QtWidgets.QLabel(
            "Applied to every shown curve after cropping: value → sign × factor × (value − first value) ÷ "
            "normalisation value. Bands (±σ) are scaled by |Y factor ÷ Y normalisation|. Axis limits and "
            "labels stay as typed (displayed units). The Analyse tab fits the transformed data."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #666; font-style: italic;")
        grid.addWidget(note, len(rows) + 1, 0, 1, 3)
        parent_layout.addWidget(box)

    @staticmethod
    def _mirror_axis_limits(min_edit: QtWidgets.QLineEdit, max_edit: QtWidgets.QLineEdit):
        """Inverting an axis' sign maps the visible window [lo, hi] to
        [-hi, -lo]. Without this, limits typed before the inversion (say
        X: 0 to 0.1) would suddenly sit entirely outside the flipped data
        and the plot would come out empty. Blank/unparseable bounds stay
        blank (= auto). Toggling the box again restores the original text."""
        lo, hi = _parse_float(min_edit.text()), _parse_float(max_edit.text())

        def fmt(v):
            return "" if v is None else ("0" if v == 0 else format(-v, ".10g"))

        new_lo, new_hi = fmt(hi), fmt(lo)
        min_edit.setText(new_lo)
        max_edit.setText(new_hi)

    def _transform_options(self) -> dict:
        return dict(
            flip_x=self.flip_x_check.isChecked(), flip_y=self.flip_y_check.isChecked(),
            zero_x=self.zero_x_check.isChecked(), zero_y=self.zero_y_check.isChecked(),
            factor_x=_parse_factor(self.factor_x_edit.text()), factor_y=_parse_factor(self.factor_y_edit.text()),
            norm_x=_parse_norm(self.norm_x_edit.text()), norm_y=_parse_norm(self.norm_y_edit.text()),
        )

    def _on_transform_changed(self, *_args):
        self._replot_timer.start()
        self._analyse_timer.start()

    def _build_settings_panel(self, parent_layout):
        box = QtWidgets.QGroupBox("5. Plot settings")
        form = QtWidgets.QFormLayout(box)

        self.title_edit = QtWidgets.QLineEdit()
        self.title_edit.textEdited.connect(lambda _t: setattr(self, "_title_auto", False))
        form.addRow("Title:", self.title_edit)

        self.xlabel_edit = QtWidgets.QLineEdit()
        self.xlabel_edit.textEdited.connect(lambda _t: setattr(self, "_xlabel_auto", False))
        form.addRow("X label:", self.xlabel_edit)

        self.ylabel_edit = QtWidgets.QLineEdit()
        self.ylabel_edit.textEdited.connect(lambda _t: setattr(self, "_ylabel_auto", False))
        form.addRow("Y label:", self.ylabel_edit)

        self.legend_loc_combo = QtWidgets.QComboBox()
        self.legend_loc_combo.addItems([
            "best", "upper right", "upper left", "lower right", "lower left",
            "center right", "center left", "upper center", "lower center", "none",
        ])
        form.addRow("Legend:", self.legend_loc_combo)

        self.grid_check = QtWidgets.QCheckBox("Show grid")
        self.grid_check.setChecked(True)
        form.addRow("", self.grid_check)

        # Defaults below match the user's own preferred plot style.
        self.sigma_spin = QtWidgets.QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 5.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(2.0)
        form.addRow("Band σ multiplier:", self.sigma_spin)

        self.alpha_spin = QtWidgets.QDoubleSpinBox()
        self.alpha_spin.setRange(0.05, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(0.22)
        form.addRow("Band opacity:", self.alpha_spin)
        # Line style and width used to be one shared, plot-wide setting
        # here -- they're now per-curve (see CurveDialog's own "Line
        # style"/"Line width" fields, and the Curves table's Style/Width
        # columns), so there is no longer a single global value to show.

        form.addRow(self._separator())
        form.addRow(QtWidgets.QLabel("Axis limits (blank = auto):"))

        self.xmin_edit, self.xmax_edit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        xlim_row = QtWidgets.QHBoxLayout()
        xlim_row.addWidget(self.xmin_edit)
        xlim_row.addWidget(QtWidgets.QLabel("to"))
        xlim_row.addWidget(self.xmax_edit)
        form.addRow("X:", xlim_row)

        self.ymin_edit, self.ymax_edit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        ylim_row = QtWidgets.QHBoxLayout()
        ylim_row.addWidget(self.ymin_edit)
        ylim_row.addWidget(QtWidgets.QLabel("to"))
        ylim_row.addWidget(self.ymax_edit)
        form.addRow("Y:", ylim_row)

        form.addRow(self._separator())
        form.addRow(QtWidgets.QLabel("Font sizes:"))

        # Defaults below match the user's own preferred plot style.
        self.axislabel_fontsize_spin = self._fontsize_spin(20)
        form.addRow("Axis titles:", self.axislabel_fontsize_spin)
        self.legend_fontsize_spin = self._fontsize_spin(14)
        form.addRow("Legend:", self.legend_fontsize_spin)
        self.tick_fontsize_spin = self._fontsize_spin(14)
        form.addRow("Tick labels:", self.tick_fontsize_spin)
        self.title_fontsize_spin = self._fontsize_spin(14)
        form.addRow("Plot title:", self.title_fontsize_spin)

        replot_btn = QtWidgets.QPushButton("Replot")
        replot_btn.clicked.connect(self.replot)
        form.addRow(replot_btn)

        parent_layout.addWidget(box)

    @staticmethod
    def _fontsize_spin(default: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(4, 48)
        spin.setValue(default)
        return spin

    @staticmethod
    def _separator() -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        return line

    def _build_export_panel(self, parent_layout, analyse: bool = False):
        """The same card on both tabs. "Save figure" exports the figure of
        the tab it sits on (each tab keeps its own DPI box); the session
        buttons are global (one session file covers both tabs)."""
        box = QtWidgets.QGroupBox("6. Export / session")
        layout = QtWidgets.QVBoxLayout(box)

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("DPI:"))
        dpi_spin = QtWidgets.QSpinBox()
        dpi_spin.setRange(72, 1200)
        dpi_spin.setValue(300)
        if analyse:
            self.analyse_dpi_spin = dpi_spin
        else:
            self.dpi_spin = dpi_spin
        row1.addWidget(dpi_spin)
        export_btn = QtWidgets.QPushButton("Save figure as...")
        export_btn.clicked.connect(lambda _checked=False, a=analyse: self._on_export_figure(a))
        row1.addWidget(export_btn)
        row1.addStretch(1)
        layout.addLayout(row1)

        # Exported image size. 0 = "auto": that dimension follows the figure
        # as currently shown (if only one is set, the other keeps the
        # on-screen aspect ratio; if both are auto, the figure is saved as
        # displayed, cropped tight like before).
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(QtWidgets.QLabel("Size:"))
        w_spin, h_spin = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for sp_, tip in ((w_spin, "Exported image width (0 = auto)"), (h_spin, "Exported image height (0 = auto)")):
            sp_.setRange(0.0, 100000.0)
            sp_.setDecimals(2)
            sp_.setSingleStep(1.0)
            sp_.setSpecialValueText("auto")
            sp_.setToolTip(tip)
            sp_.setValue(0.0)
        unit_combo = QtWidgets.QComboBox()
        for label, key in (("cm", "cm"), ("inch", "in"), ("px", "px")):
            unit_combo.addItem(label, key)
        unit_combo.setToolTip("Unit of the width/height boxes. px uses the DPI above to convert to inches.")
        size_row.addWidget(w_spin)
        size_row.addWidget(QtWidgets.QLabel("x"))
        size_row.addWidget(h_spin)
        size_row.addWidget(unit_combo)
        size_row.addStretch(1)
        layout.addLayout(size_row)
        if analyse:
            self.analyse_size_widgets = (w_spin, h_spin, unit_combo)
        else:
            self.size_widgets = (w_spin, h_spin, unit_combo)

        row2 = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Save session...")
        save_btn.clicked.connect(self._on_save_session)
        load_btn = QtWidgets.QPushButton("Load session...")
        load_btn.clicked.connect(self._on_load_session)
        row2.addWidget(save_btn)
        row2.addWidget(load_btn)
        row2.addStretch(1)
        layout.addLayout(row2)

        row3 = QtWidgets.QHBoxLayout()
        save_tpl_btn = QtWidgets.QPushButton("Save template...")
        save_tpl_btn.setToolTip("Save all formatting (plot settings, data transform, each curve's colour/line style/width/band) to a reusable file.")
        save_tpl_btn.clicked.connect(self._on_save_template)
        load_tpl_btn = QtWidgets.QPushButton("Load template...")
        load_tpl_btn.setToolTip("Apply a saved format template to the curves already loaded.")
        load_tpl_btn.clicked.connect(self._on_load_template)
        row3.addWidget(save_tpl_btn)
        row3.addWidget(load_tpl_btn)
        row3.addStretch(1)
        layout.addLayout(row3)

        parent_layout.addWidget(box)

    def _build_plot_panel(self, parent: QtWidgets.QWidget):
        layout = QtWidgets.QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(7.5, 6.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar2QT(self.canvas, parent)
        layout.addWidget(self.toolbar)

    # -- files ---------------------------------------------------------------
    def _refresh_file_list(self):
        self.file_list.clear()
        for f in self.files:
            self.file_list.addItem(f.label)

    def _unique_label(self, base: str) -> str:
        if base not in self.files_by_label:
            return base
        i = 2
        while f"{base}_{i}" in self.files_by_label:
            i += 1
        return f"{base}_{i}"

    def _on_load_files(self):
        paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Load data files", "",
            "All supported files (*.csv *.txt *.xlsx *.xlsm *.xls);;"
            "CSV/Text exports (*.csv *.txt);;Excel workbooks (*.xlsx *.xlsm *.xls);;All files (*)",
        )
        if not paths:
            return
        loaded, failed = [], []
        for path in paths:
            try:
                columns = load_table(path)
            except (OSError, ValueError) as exc:
                failed.append(f"{os.path.basename(path)}: {exc}")
                continue
            label = self._unique_label(default_label(path))
            lf = LoadedFile(path=path, label=label, columns=columns)
            self.files.append(lf)
            self.files_by_label[label] = lf
            loaded.append(label)
        self._refresh_file_list()
        if failed:
            QtWidgets.QMessageBox.warning(self, "Some files could not be loaded", "\n".join(failed))
        if loaded:
            self.statusBar().showMessage(f"Loaded {len(loaded)} file(s): {', '.join(loaded)}")

    def _selected_files(self) -> list[LoadedFile]:
        return [self.files[self.file_list.row(item)] for item in self.file_list.selectedItems()]

    def _on_rename_file(self):
        sel = self._selected_files()
        if len(sel) != 1:
            QtWidgets.QMessageBox.information(self, "Rename", "Select exactly one file to rename.")
            return
        f = sel[0]
        new_label, ok = QtWidgets.QInputDialog.getText(self, "Rename file", "New label:", text=f.label)
        new_label = new_label.strip()
        if not ok or not new_label or new_label == f.label:
            return
        if new_label in self.files_by_label:
            QtWidgets.QMessageBox.critical(self, "Rename", f"'{new_label}' is already used by another loaded file.")
            return
        del self.files_by_label[f.label]
        old_label = f.label
        f.label = new_label
        self.files_by_label[new_label] = f
        # Curves remember their source by label -- keep them pointing at
        # the renamed file instead of silently losing their data source.
        for c in self.curves:
            if c.file_label == old_label:
                c.file_label = new_label
        self._refresh_file_list()
        self._sync_after_curves_changed()

    def _on_remove_files(self):
        sel = self._selected_files()
        if not sel:
            return
        labels = {f.label for f in sel}
        used_by_curves = sorted({c.label for c in self.curves if c.file_label in labels})
        if used_by_curves:
            answer = QtWidgets.QMessageBox.question(
                self, "Remove files",
                "These files are used by existing curves, which will also be removed:\n"
                + "\n".join(used_by_curves) + "\n\nContinue?",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self.files = [f for f in self.files if f.label not in labels]
        for lbl in labels:
            self.files_by_label.pop(lbl, None)
        self.curves = [c for c in self.curves if c.file_label not in labels]
        self._refresh_file_list()
        self._sync_after_curves_changed()

    # -- curves ---------------------------------------------------------------
    def _refresh_curve_table(self):
        self._updating_table = True
        self.curve_table.setRowCount(len(self.curves))
        for row, c in enumerate(self.curves):
            color = QtGui.QColor(c.color)
            values = {
                COL_FILE: c.file_label, COL_X: pretty_label(c.x_key), COL_Y: pretty_label(c.y_key),
                COL_LABEL: c.label, COL_BAND: "✓" if c.show_band else "",
                COL_STYLE: pretty_linestyle(c.linestyle), COL_WIDTH: f"{c.linewidth:g}", COL_COLOR: c.color,
            }
            show_item = QtWidgets.QTableWidgetItem("")
            show_item.setFlags(
                Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
            )
            show_item.setCheckState(Qt.CheckState.Checked if c.visible else Qt.CheckState.Unchecked)
            show_item.setToolTip("Show / hide this curve (hidden curves are also left out of the Analyse tab)")
            self.curve_table.setItem(row, COL_SHOW, show_item)
            for col, text in values.items():
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(
                    Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                    | (Qt.ItemFlag.ItemIsEditable if col == COL_LABEL else Qt.ItemFlag.NoItemFlags)
                )
                if col == COL_COLOR:
                    item.setBackground(color)
                    item.setForeground(QtGui.QColor("#000000") if color.lightnessF() > 0.5 else QtGui.QColor("#ffffff"))
                else:
                    item.setForeground(color)
                self.curve_table.setItem(row, col, item)
        self._updating_table = False

    def _selected_curve_indices(self) -> list[int]:
        """Every selected row's index into self.curves, in order -- the
        table is in "extended" select mode, so this can be several rows
        at once (click-drag, or ctrl/shift-click)."""
        rows = {idx.row() for idx in self.curve_table.selectionModel().selectedRows()}
        return sorted(rows)

    def _selected_curve_index(self) -> int | None:
        """Like _selected_curve_indices(), but for actions that only make
        sense on exactly one curve (Edit, Up, Down) -- returns None, with
        an explanatory message if several rows are selected, rather than
        silently acting on just one of them."""
        sel = self._selected_curve_indices()
        if len(sel) == 1:
            return sel[0]
        if len(sel) > 1:
            QtWidgets.QMessageBox.information(
                self, "Select one curve", "This action works on a single curve at a time -- select just one row."
            )
        return None

    def _on_curve_item_changed(self, item: QtWidgets.QTableWidgetItem):
        """Fires on ANY cell edit -- including the ones _refresh_curve_table()
        itself makes while rebuilding the table, which is exactly what
        self._updating_table guards against. Only the Label column is ever
        actually editable (see its ItemFlag setup above), so this is
        effectively "rename a curve's legend label in place"."""
        if self._updating_table or item.column() not in (COL_LABEL, COL_SHOW):
            return
        row = item.row()
        if not (0 <= row < len(self.curves)):
            return
        if item.column() == COL_SHOW:
            self.curves[row].visible = item.checkState() == Qt.CheckState.Checked
            self._refresh_analyse_cards()
            self.replot()
            return
        new_label = item.text().strip() or self.curves[row].label
        self.curves[row].label = new_label
        if item.text() != new_label:
            self._updating_table = True
            item.setText(new_label)
            self._updating_table = False
        self._refresh_analyse_cards()
        self.replot()

    def _on_curve_cell_double_clicked(self, row: int, column: int):
        # The Label column already enters Qt's own inline text editing on
        # double-click (it's the only editable column); every other
        # column instead opens the full Edit dialog, matching the
        # original one-dialog-does-everything workflow for file/X/Y/
        # color/band.
        if column in (COL_LABEL, COL_SHOW):
            return
        self.curve_table.selectRow(row)
        self._on_edit_curve()

    def _on_add_curve(self):
        if not self.files:
            QtWidgets.QMessageBox.information(self, "Add curve", "Load at least one file first.")
            return
        dlg = CurveDialog(self)
        if dlg.exec() and dlg.result is not None:
            if self._xlabel_auto:
                self.xlabel_edit.setText(pretty_label(dlg.result.x_key))
            if self._ylabel_auto:
                self.ylabel_edit.setText(pretty_label(dlg.result.y_key))
            self.curves.append(dlg.result)
            self._sync_after_curves_changed()

    def _on_edit_curve(self):
        idx = self._selected_curve_index()
        if idx is None:
            return
        dlg = CurveDialog(self, existing=self.curves[idx])
        if dlg.exec() and dlg.result is not None:
            # Preserve the crop that was set on the curve being edited --
            # the dialog only ever touches file/axes/label/color/band, so
            # losing a carefully-set crop just because you renamed a
            # curve would be a nasty surprise.
            old = self.curves[idx]
            for name in ("crop_lo", "crop_hi", "visible", "fit_kind", "fit_degree",
                         "fit_weighted", "fit_lo", "fit_hi"):
                setattr(dlg.result, name, getattr(old, name))
            self.curves[idx] = dlg.result
            self._sync_after_curves_changed()

    def _on_remove_curve(self):
        indices = self._selected_curve_indices()
        if not indices:
            return
        for idx in sorted(indices, reverse=True):  # delete from the end so earlier indices stay valid
            del self.curves[idx]
        self._sync_after_curves_changed()

    def _on_move_curve(self, delta: int):
        idx = self._selected_curve_index()
        if idx is None:
            return
        new_idx = idx + delta
        if not (0 <= new_idx < len(self.curves)):
            return
        self.curves[idx], self.curves[new_idx] = self.curves[new_idx], self.curves[idx]
        self._refresh_curve_table()
        self.curve_table.selectRow(new_idx)
        self._refresh_crop_panel()
        self._refresh_analyse_cards()
        self.replot()

    def _on_quick_add_across(self):
        files = self._selected_files()
        if len(files) < 2:
            QtWidgets.QMessageBox.information(
                self, "Quick add across tests",
                "Select two or more files in the Files list above first (ctrl/shift-click).",
            )
            return
        dlg = QuickAddAcrossFilesDialog(self, files)
        if dlg.exec() and dlg.result:
            if self._xlabel_auto:
                self.xlabel_edit.setText(pretty_label(dlg.result[0].x_key))
            if self._ylabel_auto:
                self.ylabel_edit.setText(pretty_label(dlg.result[0].y_key))
            self.curves.extend(dlg.result)
            self._sync_after_curves_changed()

    def _on_quick_add_within(self):
        if not self.files:
            QtWidgets.QMessageBox.information(self, "Quick add within a test", "Load at least one file first.")
            return
        sel = self._selected_files()
        dlg = QuickAddWithinFileDialog(self, preselect=sel[0] if sel else None)
        if dlg.exec() and dlg.result:
            if self._xlabel_auto:
                self.xlabel_edit.setText(pretty_label(dlg.result[0].x_key))
            self.curves.extend(dlg.result)
            self._sync_after_curves_changed()

    # -- crop panel ---------------------------------------------------------------
    def _refresh_crop_panel(self):
        indices = self._selected_curve_indices()
        if len(indices) != 1:
            self.crop_group.setEnabled(False)
            self._crop_curve_index = None
            self.crop_status_label.setText(
                "Select exactly one curve above to crop its start/end."
                if not indices else
                "Select a single curve to crop (several are currently selected)."
            )
            return
        idx = indices[0]
        curve = self.curves[idx]
        source = self._curve_source(curve)
        if source is None:
            self.crop_group.setEnabled(False)
            self._crop_curve_index = None
            self.crop_status_label.setText(f"'{curve.label}': source data not available (file or column missing).")
            return

        _f, x, _y = source
        n = len(x)
        self._crop_curve_index = idx
        self.crop_group.setEnabled(n > 1)
        lo, hi = resolve_crop(n, curve.crop_lo, curve.crop_hi)

        for w in (self.crop_slider, self.crop_lo_spin, self.crop_hi_spin):
            w.blockSignals(True)
        self.crop_slider.setRange(0, max(0, n - 1))
        self.crop_slider.setValues(lo, hi)
        self.crop_lo_spin.setRange(0, max(0, n - 1))
        self.crop_hi_spin.setRange(0, max(0, n - 1))
        self.crop_lo_spin.setValue(lo)
        self.crop_hi_spin.setValue(hi)
        for w in (self.crop_slider, self.crop_lo_spin, self.crop_hi_spin):
            w.blockSignals(False)

        self._update_crop_status(curve, x, lo, hi, n)

    def _update_crop_status(self, curve: CurveSpec, x_full: np.ndarray, lo: int, hi: int, n: int):
        kept = hi - lo + 1
        self.crop_status_label.setText(
            f"'{curve.label}': keeping points {lo}–{hi} of 0–{n - 1} ({kept} pt(s)) "
            f"— x: {x_full[lo]:.6g} → {x_full[hi]:.6g}"
        )

    def _on_crop_slider_changed(self, lo: int, hi: int):
        if self._crop_curve_index is None:
            return
        curve = self.curves[self._crop_curve_index]
        curve.crop_lo, curve.crop_hi = lo, hi
        for spin, value in ((self.crop_lo_spin, lo), (self.crop_hi_spin, hi)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        source = self._curve_source(curve)
        if source is not None:
            _f, x, _y = source
            self._update_crop_status(curve, x, lo, hi, len(x))
        self._replot_timer.start()  # debounced -- see __init__

    def _on_crop_spin_changed(self, _value=None):
        if self._crop_curve_index is None:
            return
        lo, hi = self.crop_lo_spin.value(), self.crop_hi_spin.value()
        if hi < lo:
            lo, hi = hi, lo
        self.crop_slider.blockSignals(True)
        self.crop_slider.setValues(lo, hi)
        self.crop_slider.blockSignals(False)
        self._on_crop_slider_changed(*self.crop_slider.values())
        self._refresh_analyse_cards()

    def _on_crop_committed(self, *_args):
        """End of a crop-slider drag: the fit cards' point counts depend on the
        crop, so rebuild them once now (not on every pixel of the drag)."""
        self._refresh_analyse_cards()

    def _on_crop_reset(self):
        if self._crop_curve_index is None:
            return
        self.curves[self._crop_curve_index].crop_lo = 0
        self.curves[self._crop_curve_index].crop_hi = -1
        self._refresh_crop_panel()
        self._refresh_analyse_cards()
        self.replot()

    # -- analyse tab: cards, fit groups, fits ------------------------------------
    # -- fit groups ---------------------------------------------------------------
    def _group_names(self) -> list[str]:
        return [g.name for g in self.fit_groups]

    def _unique_group_name(self, base: str, ignore: str | None = None) -> str:
        names = {g.name for g in self.fit_groups if g.name != ignore}
        if base not in names:
            return base
        k = 2
        while f"{base} {k}" in names:
            k += 1
        return f"{base} {k}"

    def _on_add_group(self, all_curves: bool = False):
        name = self._unique_group_name(f"Fit {len(self.fit_groups) + 1}")
        self.fit_groups.append(FitGroup(name=name))
        if all_curves:
            for c in self.curves:
                if c.visible:
                    c.fit_group = name
        self._refresh_analyse_cards()

    def _on_delete_group(self, name: str):
        self.fit_groups = [g for g in self.fit_groups if g.name != name]
        for c in self.curves:
            if c.fit_group == name:
                c.fit_group = ""
        self._refresh_analyse_cards()

    def _on_rename_group(self, old: str, new: str):
        unique = self._unique_group_name(new, ignore=old)
        for g in self.fit_groups:
            if g.name == old:
                g.name = unique
        for c in self.curves:
            if c.fit_group == old:
                c.fit_group = unique
        self._refresh_analyse_cards()

    def _refresh_analyse_cards(self):
        """Rebuilds the fit-group cards and one fit card per drawable curve
        (a card's slider range depends on the curve's cropped length, so any
        crop/curve/group change simply rebuilds them; the settings
        themselves live in CurveSpec / FitGroup and are re-read by each new
        card)."""
        for layout in (self.fit_cards_layout, self.group_cards_layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
        self._fit_cards = []
        self._group_cards = []

        # A curve pointing at a group that no longer exists is fitted individually.
        known = set(self._group_names())
        for c in self.curves:
            if c.fit_group and c.fit_group not in known:
                c.fit_group = ""

        for group in self.fit_groups:
            gcard = _FitGroupCard(group)
            gcard.changed.connect(self._on_analyse_changed)
            gcard.renamed.connect(self._on_rename_group)
            gcard.delete_requested.connect(self._on_delete_group)
            self.group_cards_layout.addWidget(gcard)
            self._group_cards.append(gcard)

        names = self._group_names()
        for i, curve in enumerate(self.curves):
            source = self._curve_source(curve)
            if source is None or not curve.visible:
                continue
            f, x, _y = source
            lo, hi = resolve_crop(len(x), curve.crop_lo, curve.crop_hi)
            has_std = f"{curve.y_key}_std" in f.columns
            card = _CurveFitCard(i, curve, hi - lo + 1, has_std, group_names=names)
            card.changed.connect(self._on_analyse_changed)
            self.fit_cards_layout.addWidget(card)
            self._fit_cards.append(card)

        if not self._fit_cards:
            hint = QtWidgets.QLabel("No curve yet: load files and add curves in the Plot tab.")
            hint.setWordWrap(True)
            self.fit_cards_layout.addWidget(hint)

        self._analyse_timer.start()

    def _on_analyse_changed(self, *_args):
        for card in self._fit_cards:
            if 0 <= card.curve_index < len(self.curves):
                card.apply_to(self.curves[card.curve_index])
        for gcard, group in zip(self._group_cards, self.fit_groups):
            gcard.apply_to(group)
        self._analyse_timer.start()

    def _on_tab_changed(self, index: int):
        if self.tabs.widget(index) is self._analyse_tab:
            self._analyse_timer.start()  # picks up font/label changes made in the Plot tab

    def _fit_style_dict(self) -> dict:
        return {
            "linewidth": self.fit_lw_spin.value(), "extend": self.fit_extend_combo.currentData(),
            "band": self.fit_band_combo.currentData(), "sigma": self.fit_sigma_spin.value(),
            "alpha": self.fit_alpha_spin.value(), "legend": self.fit_legend_check.isChecked(),
            "markers": self.fit_markers_check.isChecked(),
        }

    def _apply_fit_style(self, st: dict):
        self.fit_lw_spin.setValue(float(st.get("linewidth", 2.6)))
        idx = self.fit_extend_combo.findData(st.get("extend", "axes"))
        self.fit_extend_combo.setCurrentIndex(idx if idx >= 0 else 0)
        idx = self.fit_band_combo.findData(st.get("band", "param"))
        self.fit_band_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.fit_sigma_spin.setValue(float(st.get("sigma", 2.0)))
        self.fit_alpha_spin.setValue(float(st.get("alpha", 0.25)))
        self.fit_legend_check.setChecked(bool(st.get("legend", False)))
        self.fit_markers_check.setChecked(bool(st.get("markers", True)))

    @staticmethod
    def _darker(color: str, factor: float = 0.55) -> tuple:
        from matplotlib.colors import to_rgb
        r, g, b = to_rgb(color)
        return (r * factor, g * factor, b * factor)

    @staticmethod
    def _fit_weights(std, lo: int, hi: int):
        """1/sigma weights for the points lo..hi of a curve's std array, or
        None if there are none to use. A point whose sigma is missing/zero/
        negative gets the median of the valid sigmas (neither dropped nor
        given infinite weight); if no sigma is valid at all -> no weights."""
        if std is None:
            return None
        sw = np.asarray(std, dtype=float)[lo:hi + 1].copy()
        good = np.isfinite(sw) & (sw > 0)
        if not good.any():
            return None
        sw[~good] = float(np.median(sw[good]))
        return 1.0 / sw

    @staticmethod
    def _param_names(kind: str, n_params: int) -> list[str]:
        if kind == "polynomial":
            return [f"c{n_params - 1 - k}" for k in range(n_params)]
        return ["a", "b"][:n_params] if n_params <= 2 else [f"p{k}" for k in range(n_params)]

    def _result_text(self, res: "FitResult", weighted: bool) -> str:
        names = self._param_names(res.kind, len(res.params))
        parts = []
        for name, val, err in zip(names, res.params, res.param_stderr):
            parts.append(f"{name} = {val:.5g}" + (f" ± {err:.2g}" if np.isfinite(err) else ""))
        return (f"{res.equation}\nR² = {res.r_squared:.5f}   (n = {res.n_points}"
                f"{', weighted' if weighted else ''})\n" + ",  ".join(parts))

    def _replot_analyse(self):
        ax = self.analyse_ax
        ax.clear()
        st = self._fit_style_dict()

        # ---- 1. data curves + the per-curve fit windows ---------------------------
        entries = []
        drawn = 0
        for card in self._fit_cards:
            curve = self.curves[card.curve_index]
            source = self._curve_source(curve)
            if source is None:
                card.setResultText("Source data not available.")
                continue
            f, x, y = source
            n = len(x)
            lo, hi = resolve_crop(n, curve.crop_lo, curve.crop_hi)
            xc, yc = x[lo:hi + 1], y[lo:hi + 1]
            nc = len(xc)
            std = None
            std_key = f"{curve.y_key}_std"
            if std_key in f.columns:
                std_full = np.asarray(f.columns[std_key])[:n][lo:hi + 1]
                std = std_full if len(std_full) == nc else None
            # Same display transform as the Plot tab: fits are computed on what is shown.
            xc, yc, std = transform_curve(xc, yc, std, **self._transform_options())
            lw = max(0.1, curve.linewidth)
            ax.plot(xc, yc, color=curve.color, linewidth=lw, linestyle=curve.linestyle, label=curve.label, zorder=4)
            drawn += 1

            flo, fhi = resolve_crop(nc, curve.fit_lo, curve.fit_hi)
            xf, yf = xc[flo:fhi + 1], yc[flo:fhi + 1]
            w = self._fit_weights(std, flo, fhi) if (curve.fit_weighted and std is not None) else None

            if st["markers"]:
                # Highlight the fitted stretch, then the two square markers that
                # follow the slider handles (same colour as the curve).
                ax.plot(xf, yf, color=curve.color, linewidth=lw * 3.0, alpha=0.25, solid_capstyle="round",
                        label="_nolegend_", zorder=2)
                ax.plot([xc[flo], xc[fhi]], [yc[flo], yc[fhi]], linestyle="none", marker="s", markersize=9,
                        markerfacecolor=curve.color, markeredgecolor="black", markeredgewidth=1.2,
                        label="_nolegend_", zorder=6)
            card.region.setStatusText(
                f"Fit window: points {flo}–{fhi} of 0–{nc - 1} ({fhi - flo + 1} pt(s)) "
                f"— x: {xc[flo]:.6g} → {xc[fhi]:.6g}"
            )

            finite = np.isfinite(xf) & np.isfinite(yf)
            if w is not None:
                finite &= np.isfinite(w)
            entries.append({
                "card": card, "curve": curve, "xc": xc[np.isfinite(xc)],
                "xs": xf[finite], "ys": yf[finite], "w": w[finite] if w is not None else None,
            })

        # ---- 2. title, labels, axis limits (shared with the Plot tab), then freeze the
        #         limits so extrapolated fit lines/bands never rescale the view ------------
        title = self.title_edit.text().strip()
        if title:
            ax.set_title(title, fontsize=self.title_fontsize_spin.value())
        if not drawn:
            ax.text(0.5, 0.5, "No curve to analyse", transform=ax.transAxes, ha="center", va="center", alpha=0.5)
        xlabel, ylabel = self.xlabel_edit.text().strip(), self.ylabel_edit.text().strip()
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.axislabel_fontsize_spin.value())
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.axislabel_fontsize_spin.value())
        xmin, xmax = _parse_float(self.xmin_edit.text()), _parse_float(self.xmax_edit.text())
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)
        ymin, ymax = _parse_float(self.ymin_edit.text()), _parse_float(self.ymax_edit.text())
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # ---- 3. fits: one unit per fit group, plus one per ungrouped curve with a model ----
        units = []
        gcards = {g.name: c for g, c in zip(self.fit_groups, self._group_cards)}
        for g in self.fit_groups:
            members = [e for e in entries if e["curve"].fit_group == g.name]
            color = g.color or (members[0]["curve"].color if members else "#555555")
            units.append({"name": g.name, "kind": g.kind, "degree": g.degree, "color": color,
                          "members": members, "card": gcards.get(g.name), "group": True})
        for e in entries:
            c = e["curve"]
            if not c.fit_group:
                if c.fit_kind == "none":
                    e["card"].setResultText("")
                    continue
                units.append({"name": c.label, "kind": c.fit_kind, "degree": c.fit_degree, "color": c.color,
                              "members": [e], "card": e["card"], "group": False})

        for u in units:
            card, members = u["card"], u["members"]
            samples, labels = [], []
            for e in members:
                if len(e["xs"]) >= 2:
                    samples.append((e["xs"], e["ys"], e["w"]))
                    labels.append(e["curve"].label)
                    if u["group"]:
                        e["card"].setResultText(f"Pooled in fit group \"{u['name']}\".")
                elif u["group"]:
                    e["card"].setResultText("Too few points in the window: excluded from the group fit.")
            if not samples:
                if card is not None:
                    card.setResultText("No curve assigned to this group." if not members
                                       else "Not enough points in the fit windows.")
                else:
                    members[0]["card"].setResultText("Not enough points in the fit window.")
                continue
            try:
                res, rms_each, rms_all = group_fit(samples, u["kind"], u["degree"])
            except (ValueError, np.linalg.LinAlgError) as exc:
                if card is not None:
                    card.setResultText(f"Fit impossible: {exc}")
                continue

            if st["extend"] == "axes":
                x0, x1 = min(xlim), max(xlim)
            elif st["extend"] == "data":
                allx = np.concatenate([m["xc"] for m in members if len(m["xc"])])
                x0, x1 = float(allx.min()), float(allx.max())
            else:
                allx = np.concatenate([sm[0] for sm in samples])
                x0, x1 = float(allx.min()), float(allx.max())
            xs = np.linspace(x0, x1, 400)

            band_note = ""
            if st["band"] != "none":
                band = fit_band_arrays(res, xs, st["band"], st["sigma"], rms_all)
                if band is None:
                    band_note = "\n(Band unavailable: not enough points to estimate the fit uncertainty.)"
                else:
                    ax.fill_between(xs, band[0], band[1], color=u["color"], alpha=st["alpha"],
                                    linewidth=0, label="_nolegend_", zorder=1)
            fit_label = f"{u['name']} fit" if st["legend"] else "_nolegend_"
            ax.plot(xs, res.predict(xs), color=u["color"], linestyle="-", linewidth=st["linewidth"],
                    label=fit_label, zorder=3, solid_capstyle="butt")

            meas = all(sm[2] is not None for sm in samples)
            text = self._result_text(res, meas)
            if len(samples) > 1:
                spread = "\n".join(f"  {lab}: RMS = {r:.4g}" for lab, r in zip(labels, rms_each))
                text += (f"\nCurves pooled: {len(samples)}; measurement σ "
                         + ("included" if meas else "not used (not every curve has one)")
                         + " + between-curve variance.\nResidual spread per curve:\n" + spread)
            if card is not None:
                card.setResultText(text + band_note)

        if self.grid_check.isChecked():
            ax.grid(True, alpha=0.3)
        loc = self.legend_loc_combo.currentText()
        if drawn and loc != "none":
            ax.legend(loc=loc, fontsize=self.legend_fontsize_spin.value())
        ax.tick_params(axis="both", labelsize=self.tick_fontsize_spin.value())
        self.analyse_fig.tight_layout()
        self.analyse_canvas.draw_idle()

    # -- plotting ---------------------------------------------------------------
    def replot(self):
        self.ax.clear()
        sigma = max(0.0, self.sigma_spin.value())
        alpha = min(max(self.alpha_spin.value(), 0.0), 1.0)

        drawn, skipped, hidden = 0, [], 0
        topts = self._transform_options()
        for c in self.curves:
            if not c.visible:
                hidden += 1
                continue
            source = self._curve_source(c)
            if source is None:
                skipped.append(c.label)
                continue
            f, x, y = source
            n = len(x)
            lo, hi = resolve_crop(n, c.crop_lo, c.crop_hi)
            xc, yc = x[lo:hi + 1], y[lo:hi + 1]
            std = None
            std_key = f"{c.y_key}_std"
            if c.show_band and std_key in f.columns:
                std = np.asarray(f.columns[std_key])[:n][lo:hi + 1]
            xc, yc, std = transform_curve(xc, yc, std, **topts)
            lw = max(0.1, c.linewidth)
            self.ax.plot(xc, yc, color=c.color, linewidth=lw, linestyle=c.linestyle, label=c.label)
            if std is not None:
                m = min(len(xc), len(std))
                self.ax.fill_between(
                    xc[:m], yc[:m] - sigma * std[:m], yc[:m] + sigma * std[:m],
                    color=c.color, alpha=alpha, linewidth=0,
                )
            drawn += 1

        title = self.title_edit.text().strip()
        if title:
            self.ax.set_title(title, fontsize=self.title_fontsize_spin.value())
        xlabel, ylabel = self.xlabel_edit.text().strip(), self.ylabel_edit.text().strip()
        if xlabel:
            self.ax.set_xlabel(xlabel, fontsize=self.axislabel_fontsize_spin.value())
        if ylabel:
            self.ax.set_ylabel(ylabel, fontsize=self.axislabel_fontsize_spin.value())
        if self.grid_check.isChecked():
            self.ax.grid(True, alpha=0.3)
        loc = self.legend_loc_combo.currentText()
        if drawn and loc != "none":
            self.ax.legend(loc=loc, fontsize=self.legend_fontsize_spin.value())
        self.ax.tick_params(axis="both", labelsize=self.tick_fontsize_spin.value())

        # Axis limits: each of the four fields is independent and defaults
        # to "auto" when blank (or not a valid number) -- e.g. you can pin
        # just the X origin to 0 and leave every other bound alone.
        xmin, xmax = _parse_float(self.xmin_edit.text()), _parse_float(self.xmax_edit.text())
        if xmin is not None or xmax is not None:
            self.ax.set_xlim(left=xmin, right=xmax)
        ymin, ymax = _parse_float(self.ymin_edit.text()), _parse_float(self.ymax_edit.text())
        if ymin is not None or ymax is not None:
            self.ax.set_ylim(bottom=ymin, top=ymax)

        self.fig.tight_layout()
        self.canvas.draw_idle()

        msg = f"{drawn} curve(s) plotted from {len(self.files)} file(s)."
        if hidden:
            msg += f" {hidden} hidden."
        if skipped:
            msg += f" {len(skipped)} curve(s) skipped (source column missing): {', '.join(skipped)}."
        self.statusBar().showMessage(msg)

    # -- export / session ---------------------------------------------------------------
    @staticmethod
    def _ensure_extension(path: str, default_ext: str) -> str:
        """Qt's save dialog doesn't auto-append an extension the way
        Tkinter's `defaultextension=` did -- add one back on if the user
        typed a bare filename."""
        return path if os.path.splitext(path)[1] else path + default_ext

    def _on_export_figure(self, analyse: bool = False):
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save figure", "",
            "PNG image (*.png);;PDF document (*.pdf);;SVG image (*.svg);;All files (*)",
        )
        if not path:
            return
        ext = ".png"
        if "pdf" in selected_filter.lower():
            ext = ".pdf"
        elif "svg" in selected_filter.lower():
            ext = ".svg"
        path = self._ensure_extension(path, ext)
        try:
            fig, dpi = ((self.analyse_fig, self.analyse_dpi_spin.value()) if analyse
                        else (self.fig, self.dpi_spin.value()))
            w_spin, h_spin, unit_combo = self.analyse_size_widgets if analyse else self.size_widgets
            size_in = export_size_inches(w_spin.value(), h_spin.value(), unit_combo.currentData(),
                                         dpi, tuple(fig.get_size_inches()))
            save_figure_sized(fig, path, dpi, size_in)
            if analyse:
                self.analyse_canvas.draw_idle()
            else:
                self.canvas.draw_idle()
        except Exception as exc:  # noqa: BLE001 - surface any backend error plainly
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        extra = f" ({size_in[0] * 2.54:.1f} x {size_in[1] * 2.54:.1f} cm at {dpi} DPI)" if size_in else ""
        self.statusBar().showMessage(f"Figure saved to {os.path.basename(path)}{extra}")

    def _format_settings_dict(self) -> dict:
        """Every formatting setting of the Plot tab, keyed like
        FORMAT_DEFAULTS -- shared by the session file and the templates."""
        return {
            "title": self.title_edit.text(), "xlabel": self.xlabel_edit.text(), "ylabel": self.ylabel_edit.text(),
            "legend_loc": self.legend_loc_combo.currentText(), "grid": self.grid_check.isChecked(),
            "sigma": self.sigma_spin.value(), "alpha": self.alpha_spin.value(),
            "xmin": self.xmin_edit.text(), "xmax": self.xmax_edit.text(),
            "ymin": self.ymin_edit.text(), "ymax": self.ymax_edit.text(),
            "axislabel_fontsize": self.axislabel_fontsize_spin.value(),
            "legend_fontsize": self.legend_fontsize_spin.value(),
            "tick_fontsize": self.tick_fontsize_spin.value(),
            "title_fontsize": self.title_fontsize_spin.value(),
            "flip_x": self.flip_x_check.isChecked(), "flip_y": self.flip_y_check.isChecked(),
            "zero_x": self.zero_x_check.isChecked(), "zero_y": self.zero_y_check.isChecked(),
            "factor_x": self.factor_x_edit.text(), "factor_y": self.factor_y_edit.text(),
            "norm_x": self.norm_x_edit.text(), "norm_y": self.norm_y_edit.text(),
        }

    def _apply_format_settings(self, s: dict):
        """Writes the settings present in `s` (keys of FORMAT_DEFAULTS; any
        subset) into the widgets. Order matters: the data transform goes
        BEFORE the axis limits, because toggling "Invert sign" mirrors the
        limit fields (see _mirror_axis_limits) and the limits being applied
        are already expressed in the displayed units."""
        if "title" in s:
            self.title_edit.setText(str(s["title"])); self._title_auto = False
        if "xlabel" in s:
            self.xlabel_edit.setText(str(s["xlabel"])); self._xlabel_auto = False
        if "ylabel" in s:
            self.ylabel_edit.setText(str(s["ylabel"])); self._ylabel_auto = False
        if "legend_loc" in s:
            idx = self.legend_loc_combo.findText(str(s["legend_loc"]))
            self.legend_loc_combo.setCurrentIndex(idx if idx >= 0 else 0)
        if "grid" in s:
            self.grid_check.setChecked(bool(s["grid"]))
        if "sigma" in s:
            self.sigma_spin.setValue(float(s["sigma"]))
        if "alpha" in s:
            self.alpha_spin.setValue(float(s["alpha"]))
        for spin, key in ((self.axislabel_fontsize_spin, "axislabel_fontsize"), (self.legend_fontsize_spin, "legend_fontsize"),
                          (self.tick_fontsize_spin, "tick_fontsize"), (self.title_fontsize_spin, "title_fontsize")):
            if key in s:
                spin.setValue(int(s[key]))
        for chk, key in ((self.flip_x_check, "flip_x"), (self.flip_y_check, "flip_y"),
                         (self.zero_x_check, "zero_x"), (self.zero_y_check, "zero_y")):
            if key in s:
                chk.setChecked(bool(s[key]))
        if "factor_x" in s:
            self.factor_x_edit.setText(str(s["factor_x"]))
        if "factor_y" in s:
            self.factor_y_edit.setText(str(s["factor_y"]))
        if "norm_x" in s:
            self.norm_x_edit.setText(str(s["norm_x"]))
        if "norm_y" in s:
            self.norm_y_edit.setText(str(s["norm_y"]))
        for edit, key in ((self.xmin_edit, "xmin"), (self.xmax_edit, "xmax"),
                          (self.ymin_edit, "ymin"), (self.ymax_edit, "ymax")):
            if key in s:
                edit.setText(str(s[key]))

    def _on_save_template(self):
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save format template", "", "Format templates (*.json)")
        if not path:
            return
        path = self._ensure_extension(path, ".json")
        try:
            with open(path, "w") as fh:
                json.dump(make_template(self._format_settings_dict(), self.curves), fh, indent=2)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.statusBar().showMessage(
            f"Template saved to {os.path.basename(path)} ({len(self.curves)} curve style(s) + all plot settings).")

    def _on_load_template(self):
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load format template", "", "Format templates (*.json)")
        if not path:
            return
        try:
            with open(path, "r") as fh:
                template = json.load(fh)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return
        if not isinstance(template, dict) or template.get("kind") != TEMPLATE_KIND:
            QtWidgets.QMessageBox.critical(
                self, "Not a template",
                "This file is not a format template (a session file? use \"Load session...\" for those).")
            return
        available = template_available_groups(template)
        if not available:
            QtWidgets.QMessageBox.information(self, "Empty template", "This template contains nothing to apply.")
            return
        styles = template.get("curve_styles", [])
        dlg = TemplateApplyDialog(self, available, len(styles), len(self.curves))
        if not (dlg.exec() and dlg.selected):
            return
        try:
            self._apply_format_settings(template_settings_for_groups(template, dlg.selected))
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "Template error", f"A template value is invalid: {exc}")
            return
        use_style, use_labels = "curves" in dlg.selected, "labels" in dlg.selected
        n_styled = restyle_curves(self.curves, styles, apply_style=use_style, apply_labels=use_labels) \
            if (use_style or use_labels) else 0
        self._sync_after_curves_changed()
        self._on_transform_changed()
        msg = f"Template '{os.path.basename(path)}' applied."
        if use_style or use_labels:
            msg += f" {n_styled} curve(s) updated"
            if len(self.curves) > len(styles):
                msg += f" ({len(self.curves) - len(styles)} extra curve(s) left unchanged)"
            msg += "."
        self.statusBar().showMessage(msg)

    def _session_dict(self) -> dict:
        return {
            "format_version": 6,  # v6: fit groups (replace the old "combine all curves" option) + fit appearance; v5: curves carry `visible`, data-transform options
            "files": [{"path": f.path, "label": f.label} for f in self.files],
            "curves": [asdict(c) for c in self.curves],
            "fit_groups": [asdict(g) for g in self.fit_groups],
            "fit_style": self._fit_style_dict(),
            "settings": self._format_settings_dict(),
        }

    def _on_save_session(self):
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(self, "Save session", "", "Session files (*.json)")
        if not path:
            return
        path = self._ensure_extension(path, ".json")
        try:
            with open(path, "w") as fh:
                json.dump(self._session_dict(), fh, indent=2)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.statusBar().showMessage(f"Session saved to {os.path.basename(path)}")

    def _on_load_session(self):
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Load session", "", "Session files (*.json)")
        if not path:
            return
        try:
            with open(path, "r") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.files, self.files_by_label = [], {}
        missing = []
        for finfo in data.get("files", []):
            try:
                columns = load_table(finfo["path"])
            except (OSError, ValueError, KeyError) as exc:
                missing.append(f"{finfo.get('label', finfo.get('path', '?'))}: {exc}")
                continue
            lf = LoadedFile(path=finfo["path"], label=finfo["label"], columns=columns)
            self.files.append(lf)
            self.files_by_label[lf.label] = lf

        self.curves = []
        for cinfo in data.get("curves", []):
            try:
                self.curves.append(CurveSpec(**cinfo))
            except TypeError:
                continue  # a curve referencing fields from a future/foreign session format -- skip, don't crash

        s = data.get("settings", {})
        # Missing keys (an older session file) fall back to the defaults.
        # (Line width is per-curve since v3 and travels with each curve's own
        # CurveSpec dict above.)
        self._apply_format_settings({**FORMAT_DEFAULTS, **s})

        # Fit groups (v6). An older session with "combine all curves" (v4/v5)
        # becomes one group holding every curve.
        self.fit_groups = []
        for ginfo in data.get("fit_groups", []):
            try:
                self.fit_groups.append(FitGroup(**ginfo))
            except TypeError:
                continue
        if not self.fit_groups and s.get("combine_fits"):
            self.fit_groups.append(FitGroup(name="Combined", kind=s.get("combined_model", "linear"),
                                            degree=max(1, min(int(s.get("combined_degree", 2)), 8))))
            for c in self.curves:
                c.fit_group = "Combined"

        self._refresh_file_list()
        self._sync_after_curves_changed()
        # Style widgets AFTER the cards were rebuilt for the loaded curves: their
        # change handlers read the cards back into the curves.
        self._apply_fit_style(data.get("fit_style", {}))
        self._on_analyse_changed()
        msg = f"Session loaded: {len(self.files)} file(s), {len(self.curves)} curve(s)."
        if missing:
            msg += " Could not reload: " + "; ".join(missing)
            QtWidgets.QMessageBox.warning(self, "Some files could not be reloaded", "\n".join(missing))
        self.statusBar().showMessage(msg)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    # Opens filling the screen by default, as requested -- showMaximized()
    # rather than showFullScreen(): it still keeps the window's title bar
    # and OS window controls (so it can be moved/restored/un-maximized
    # normally, and dialogs stay clearly separate windows), whereas true
    # fullscreen hides all of that. Easy to swap for window.showFullScreen()
    # here if borderless fullscreen is what's actually wanted instead.
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
