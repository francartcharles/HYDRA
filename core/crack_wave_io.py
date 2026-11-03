"""
Self-sufficient data loading for the Crack & Wave Analysis standalone tool.

Like every other page under "Standalone Tools" (Free-Decay Damping,
Viscoelastic Model Calibration, ...), this tool loads its own input files
rather than depending on another page's live, in-memory state:
core/project.py's `DICSettings` docstring is explicit that a DIC run's
computed fields (`DICResult`) are intentionally NOT persisted anywhere,
kept only on `ImageCorrelationPage` itself, "re-run on demand". The one
thing the Image Correlation page already produces that IS meant to leave
the app is its "Export DIC results" CSV
(`page_image_correlation.py:_on_export_results`) -- a tidy, one-row-per-
point-per-frame dump of the full displacement/strain field time series.
This module reads that CSV back into a small, page-local data model
(`CrackWaveDataset`), so a user who pushed the correlation mesh very fine
to resolve elastic-wave fronts and crack-tip strain localization can bring
that run's fields into this tool without re-running DIC.

The companion background image sequence (for displaying a chosen frame
behind the strain-field overlay while tracing a crack skeleton) is loaded
with the existing `core.frame_source.open_frame_source` -- no new loader
needed, video or image-folder input both already work there.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from core.dic_kinematics import von_mises_strain
from core.dic_fields import principal_strains, max_shear_strain

# (key, label) for the fields this tool lets the user threshold/trace on --
# a subset of page_image_correlation.py's own FIELD_CHOICES (same key
# names and label wording), limited to the strain fields relevant to wave
# fronts and crack-tip localization (this tool has no use for the
# displacement/velocity/acceleration fields also offered there).
FIELD_CHOICES = [
    ("eps_eq", "Equivalent strain (von Mises)"),
    ("exx", "exx -- horizontal strain"),
    ("eyy", "eyy -- vertical strain"),
    ("exy", "exy -- shear strain"),
    # Orientation-independent measures (invariants of the in-plane strain
    # tensor -- see core/dic_fields.py): they follow a crack/wave front
    # whatever its direction relative to the image axes.
    ("e1", "e1 -- max principal strain"),
    ("e2", "e2 -- min principal strain"),
    ("gmax", "gmax -- max shear strain (e1 - e2)/2"),
]

_REQUIRED_COLUMNS = ["frame_index", "point_row", "point_col", "x0", "y0", "u", "v", "exx", "eyy", "exy"]


@dataclass
class CrackWaveFrame:
    """One frame's fields, each a 2D array shaped (n_points_y, n_points_x)
    -- same grid layout as core/dic_processing.py's `DICFrameResult`, and
    the same NaN-for-excluded-point convention."""
    frame_index: int
    time_s: float | None
    u: np.ndarray
    v: np.ndarray
    exx: np.ndarray
    eyy: np.ndarray
    exy: np.ndarray
    correlation: np.ndarray
    eps_eq: np.ndarray  # von Mises equivalent strain, derived (plane-strain assumption -- see von_mises_strain)
    # Derived from exx/eyy/exy in __post_init__ when not given (see
    # core.dic_fields): max / min in-plane principal strain and the max
    # in-plane shear strain (e1 - e2)/2 (tensor convention, same as exy).
    e1: np.ndarray | None = None
    e2: np.ndarray | None = None
    gmax: np.ndarray | None = None
    # Displacement magnitude sqrt(u^2 + v^2) (px): the norm of the displacement
    # VECTOR, independent of the image axes. Derived in __post_init__.
    umag: np.ndarray | None = None

    def __post_init__(self):
        if self.e1 is None or self.e2 is None:
            self.e1, self.e2 = principal_strains(self.exx, self.eyy, self.exy)
        if self.gmax is None:
            self.gmax = max_shear_strain(self.exx, self.eyy, self.exy)
        if self.umag is None:
            self.umag = np.hypot(np.asarray(self.u, dtype=float), np.asarray(self.v, dtype=float))

    def field(self, key: str) -> np.ndarray | None:
        return getattr(self, key, None)


@dataclass
class CrackWaveDataset:
    """A full loaded DIC-export CSV: the (regular) point grid it was
    measured on, plus every frame's fields, ready for the crack-skeleton
    tracing tool and the later wave-speed / crack-velocity / K1c stages to
    read from directly -- no dependency on the Image Correlation page or
    a live `DICResult`."""
    xs: np.ndarray            # 1D, x pixel coords of the grid columns (reference frame), length n_points_x
    ys: np.ndarray            # 1D, y pixel coords of the grid rows (reference frame), length n_points_y
    frames: list              # list[CrackWaveFrame], sorted by frame_index
    source_path: str = ""
    # Recovered from the export's own u_mm/u and v_mm/v ratios (see
    # `_recover_pixel_size_mm`) -- None if the export was written with no
    # pixel size set on the Image Correlation page (blank u_mm/v_mm
    # columns), same "None means not recoverable" convention used
    # throughout this app rather than a fabricated/guessed value. Read by
    # the page to auto-fill its own "Pixel size" field (Material &
    # Calibration card) when it's still unset.
    pixel_size_mm: float | None = None

    @property
    def n_points_x(self) -> int:
        return int(self.xs.size)

    @property
    def n_points_y(self) -> int:
        return int(self.ys.size)

    @property
    def frame_indices(self) -> list:
        return [f.frame_index for f in self.frames]

    def frame_result(self, frame_index: int) -> CrackWaveFrame | None:
        for f in self.frames:
            if f.frame_index == frame_index:
                return f
        return None

    def field_range(self, key: str) -> tuple:
        """(min, max) of `key` across every loaded frame, ignoring NaN --
        used to seed a sensible default auto-detection threshold. Returns
        (0.0, 0.0) if every value is NaN (nothing measured for that
        field)."""
        values = []
        for f in self.frames:
            arr = f.field(key)
            if arr is not None:
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    values.append(finite)
        if not values:
            return 0.0, 0.0
        allv = np.concatenate(values)
        return float(allv.min()), float(allv.max())


class CrackWaveLoadError(ValueError):
    """Raised for a CSV that isn't a (recognizable) DIC results export --
    message is shown to the user as-is, so it should stay short and
    specific."""


def load_dic_export_csv(path: str) -> CrackWaveDataset:
    """Reads a CSV produced by the Image Correlation page's "Export DIC
    results" button (`_on_export_results`) back into a `CrackWaveDataset`.

    The exported file is one row per (frame, grid point) -- this
    reconstructs the regular (point_row, point_col) grid from the rows
    themselves (rather than requiring the grid shape as a separate input),
    derives each point's fixed reference position (x0, y0) from its first
    occurrence, and computes the von Mises equivalent strain field per
    frame the same way the rest of the app does
    (`core.dic_kinematics.von_mises_strain`, plane-strain/ezz=0
    assumption) since the export CSV itself only carries exx/eyy/exy, not
    the derived eps_eq (and the principal / max-shear strains e1, e2, gmax, which
    CrackWaveFrame derives itself). Also recovers a pixel size (mm/px), when the
    export's own u_mm/v_mm columns allow it -- see
    `_recover_pixel_size_mm` -- so this tool's own "Pixel size" field can
    auto-fill itself from the loaded CSV instead of always requiring a
    switch to the Digital Image Correlation tab.

    Raises `CrackWaveLoadError` for a file missing the columns this relies
    on, or with no data rows at all."""
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            missing = [c for c in _REQUIRED_COLUMNS if c not in fieldnames]
            if missing:
                raise CrackWaveLoadError(
                    f"Not a recognized DIC results export -- missing column(s): {', '.join(missing)}."
                )
            has_mm = "u_mm" in fieldnames and "v_mm" in fieldnames
            rows = list(reader)
    except OSError as exc:
        raise CrackWaveLoadError(f"Could not read file: {exc}") from exc

    if not rows:
        raise CrackWaveLoadError("The file has no data rows.")

    point_rows = sorted({int(r["point_row"]) for r in rows})
    point_cols = sorted({int(r["point_col"]) for r in rows})
    row_index = {r: i for i, r in enumerate(point_rows)}
    col_index = {c: i for i, c in enumerate(point_cols)}
    n_points_y, n_points_x = len(point_rows), len(point_cols)

    xs = np.full(n_points_x, np.nan)
    ys = np.full(n_points_y, np.nan)

    by_frame: dict = {}
    for r in rows:
        fi = int(r["frame_index"])
        by_frame.setdefault(fi, []).append(r)

    def _float_or_nan(value: str) -> float:
        if value is None or value == "":
            return float("nan")
        try:
            return float(value)
        except ValueError:
            return float("nan")

    frames = []
    pixel_size_ratios = []  # per-row u_mm/u, v_mm/v -- see _recover_pixel_size_mm
    for fi in sorted(by_frame.keys()):
        u = np.full((n_points_y, n_points_x), np.nan)
        v = np.full((n_points_y, n_points_x), np.nan)
        exx = np.full((n_points_y, n_points_x), np.nan)
        eyy = np.full((n_points_y, n_points_x), np.nan)
        exy = np.full((n_points_y, n_points_x), np.nan)
        corr = np.full((n_points_y, n_points_x), np.nan)
        time_s = None
        for r in by_frame[fi]:
            iy = row_index[int(r["point_row"])]
            ix = col_index[int(r["point_col"])]
            if np.isnan(xs[ix]):
                xs[ix] = _float_or_nan(r.get("x0"))
            if np.isnan(ys[iy]):
                ys[iy] = _float_or_nan(r.get("y0"))
            u_val = _float_or_nan(r.get("u"))
            v_val = _float_or_nan(r.get("v"))
            u[iy, ix] = u_val
            v[iy, ix] = v_val
            exx[iy, ix] = _float_or_nan(r.get("exx"))
            eyy[iy, ix] = _float_or_nan(r.get("eyy"))
            exy[iy, ix] = _float_or_nan(r.get("exy"))
            corr[iy, ix] = _float_or_nan(r.get("correlation"))
            t_raw = r.get("time_s")
            if t_raw not in (None, ""):
                try:
                    time_s = float(t_raw)
                except ValueError:
                    pass
            if has_mm:
                u_mm = _float_or_nan(r.get("u_mm"))
                v_mm = _float_or_nan(r.get("v_mm"))
                if np.isfinite(u_mm) and np.isfinite(u_val) and abs(u_val) > 1e-9:
                    pixel_size_ratios.append(u_mm / u_val)
                if np.isfinite(v_mm) and np.isfinite(v_val) and abs(v_val) > 1e-9:
                    pixel_size_ratios.append(v_mm / v_val)
        eps_eq = von_mises_strain(exx, eyy, exy)
        frames.append(CrackWaveFrame(
            frame_index=fi, time_s=time_s, u=u, v=v, exx=exx, eyy=eyy, exy=exy,
            correlation=corr, eps_eq=eps_eq,
        ))

    # A reference-position column that never got a valid (non-NaN) value
    # from any frame (e.g. a hand-edited/truncated export) would otherwise
    # silently corrupt every pixel-coordinate lookup downstream (skeleton
    # tracing, threshold detection) -- fail loudly here instead.
    if np.any(np.isnan(xs)) or np.any(np.isnan(ys)):
        raise CrackWaveLoadError(
            "Some grid points never got a reference position (x0/y0) -- the file may be truncated or corrupted."
        )

    return CrackWaveDataset(xs=xs, ys=ys, frames=frames, source_path=path,
                             pixel_size_mm=_recover_pixel_size_mm(pixel_size_ratios))


def _recover_pixel_size_mm(ratios: list) -> float | None:
    """Median of every pooled, finite, strictly-positive u_mm/u (and v_mm/
    v) ratio -- the same pixel-size-recovery approach `core.dic_result_io`
    uses for the Image Correlation page's own CSV reimport, applied here
    to this tool's CSV loader too: a row's u_mm/v_mm columns are only
    populated by the export (`page_image_correlation.py::_on_export_
    results`) when a pixel size was set at export time, and dividing them
    back by the matching u/v recovers that same mm-per-(working)-pixel
    factor. Pooling the whole file (rather than trusting a single row)
    keeps this steady against the export's own %.6g rounding. None if the
    file carried no u_mm/v_mm data to recover from."""
    finite = [r for r in ratios if np.isfinite(r) and r > 0]
    if not finite:
        return None
    return float(np.median(finite))
