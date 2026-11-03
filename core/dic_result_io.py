"""
Reopening a previously-exported DIC results CSV on the Digital Image
Correlation page itself.

`DICResult` (core/dic_processing.py) is intentionally never persisted --
core/project.py's `DICSettings` docstring is explicit that a correlation
run's tracked fields are kept in memory only, "re-run on demand", to keep
the session file small. The one thing that *does* leave the app is the
page's own "Export results (.csv)..." button (`_on_export_results`), a
tidy one-row-per-point-per-frame dump of the full displacement/strain
field time series. This module reads that CSV back into a `DICResult`
(plus the two calibration numbers it lets a caller recover), so a past
run's fields can be brought back onto the page without re-running the
correlation.

What the export CSV carries is exactly the *reconstructible* half of a
`DICResult`:
  - the correlation mesh's point grid (from x0/y0, i.e. `mesh.grid_xy()`
    of the ORIGINAL run) and each frame's tracked position/displacement/
    strain/correlation fields -- a lossless round-trip for all of this,
    same units, same NaN-for-excluded-point convention.
  - the run's SETTINGS (subset size, ROI shape, search radius, tracking
    mode, thresholds, downscale/crop/frame-step preprocessing) and the
    raw-image-derived uncertainty ingredients (`noise_sigma`, `sssig_x`,
    `sssig_y`) are simply not in the file, so an imported `DICResult`
    always has `noise_sigma=sssig_x=sssig_y=None` -- exactly the same
    "None means not estimated" state `compute_uncertainty()` already
    treats as "no uncertainty estimate available" for any other reason,
    so nothing downstream needs to special-case an imported result.
  - `mesh.subset_size` is set to `mesh.step` (recovered exactly from the
    grid spacing) as an inert placeholder: `core/dic_uncertainty.py` only
    ever reads `mesh.subset_size` after already checking `noise_sigma is
    not None`, and an imported result's `noise_sigma` is always None, so
    this placeholder is genuinely never read for any real computation --
    it only needs *some* positive integer to satisfy the dataclass.
  - `mesh.mask` is left None: a masked-out (outside a non-rectangular ROI)
    point and a point that simply never tracked are indistinguishable in
    the export (both are NaN in every frame) and behave identically
    downstream (excluded from every mean/plot/uncertainty calculation),
    so collapsing them back into "just NaN, no mask" loses nothing.

Two calibration numbers a caller may find useful to re-apply to the page
are recovered from the data itself, when the export included them:
  - `mm_per_working_px`: the exported `u_mm`/`v_mm` columns are only
    written when a pixel size was set at export time (see
    `_on_export_results`'s `scale = self._mm_per_working_px()`) -- when
    present, dividing them back by `u`/`v` recovers that same scale
    factor (mm per PREPROCESSED-frame pixel, i.e. already including
    whatever downscale factor was in effect for that run).
  - `fps`: the exported `time_s` column is only written when a camera
    frame rate was set (`fps = self._effective_fps()`); `frame_index` is
    always written and exact, so `(frame_index - ref_index) / time_s`
    recovers that same effective frame rate (frame rate x time_scale).

Both are recovered as the median ratio pooled over every (frame, point)
row where the needed values are present and non-degenerate (a whole-file
average rather than trusting a single row, which would be thrown off by
roundoff in the export's own %.6g/%.4g formatting) -- see
`_median_ratio`. Neither recovery can separate the two factors that
originally multiplied together to make it (pixel_size_mm x downscale_
factor; frame_rate_hz x time_scale): the caller applies the recovered
product as one factor (pixel size / frame rate) and neutralizes the
other (downscale=1 / time_scale=1.0), which reproduces every downstream
calculation identically regardless of the original individual split --
see `page_image_correlation.py`'s `_mm_per_working_px`/`_effective_fps`.
A nonzero `time_offset_s` at export time is not recoverable at all (an
additive constant, invisible in a ratio) and is assumed to have been 0.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from core.dic_correlation import MeshGrid
from core.dic_processing import DICFrameResult, DICResult

_REQUIRED_COLUMNS = ["frame_index", "point_row", "point_col", "x0", "y0",
                     "x", "y", "u", "v", "exx", "eyy", "exy", "correlation"]


class DicResultLoadError(ValueError):
    """Raised for a CSV that isn't a (recognizable) DIC results export --
    message is shown to the user as-is, so it should stay short and
    specific."""


@dataclass
class DicResultImport:
    """What `load_dic_result_csv` hands back: the reconstructed result,
    plus whatever calibration could be recovered from the file (each
    None if the export didn't carry what that recovery needs)."""
    result: DICResult
    mm_per_working_px: float | None
    fps: float | None
    source_path: str = ""


def _float_or_nan(value) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _median_positive(values: list) -> float | None:
    """Median of every finite, strictly-positive value in `values`, or
    None if there are none. Used to pool a whole file's worth of per-row
    scale/rate estimates into one steadier number, rather than trusting
    any single row (which formatting roundoff in the export's own
    %.4g/%.6g could throw off, especially near a tiny denominator)."""
    finite = [v for v in values if np.isfinite(v) and v > 0]
    if not finite:
        return None
    return float(np.median(finite))


def load_dic_result_csv(path: str) -> DicResultImport:
    """Reads a CSV produced by the Image Correlation page's "Export
    results (.csv)..." button (`_on_export_results`) back into a
    `DicResultImport`.

    The exported file is one row per (frame, grid point) -- this
    reconstructs the regular (point_row, point_col) grid and the
    correlation mesh's spacing from the rows themselves (rather than
    requiring the mesh shape as a separate input), the same way
    `core.crack_wave_io.load_dic_export_csv` reads the same file format
    for the Crack & Wave Analysis tool.

    Raises `DicResultLoadError` for a file missing the columns this
    relies on, with no data rows at all, or where the point grid can't
    be recovered (e.g. fewer than 2 distinct columns/rows, so a step
    spacing can't be measured)."""
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            missing = [c for c in _REQUIRED_COLUMNS if c not in fieldnames]
            if missing:
                raise DicResultLoadError(
                    f"Not a recognized DIC results export -- missing column(s): {', '.join(missing)}."
                )
            has_time = "time_s" in fieldnames
            has_mm = "u_mm" in fieldnames and "v_mm" in fieldnames
            rows = list(reader)
    except OSError as exc:
        raise DicResultLoadError(f"Could not read file: {exc}") from exc

    if not rows:
        raise DicResultLoadError("The file has no data rows.")

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

    frame_indices = sorted(by_frame.keys())
    ref_index = frame_indices[0]

    u_over_mm, v_over_mm = [], []
    dframe_over_time = []

    frames = []
    for fi in frame_indices:
        x = np.full((n_points_y, n_points_x), np.nan)
        y = np.full((n_points_y, n_points_x), np.nan)
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
            x[iy, ix] = _float_or_nan(r.get("x"))
            y[iy, ix] = _float_or_nan(r.get("y"))
            u_val = _float_or_nan(r.get("u"))
            v_val = _float_or_nan(r.get("v"))
            u[iy, ix] = u_val
            v[iy, ix] = v_val
            exx[iy, ix] = _float_or_nan(r.get("exx"))
            eyy[iy, ix] = _float_or_nan(r.get("eyy"))
            exy[iy, ix] = _float_or_nan(r.get("exy"))
            corr[iy, ix] = _float_or_nan(r.get("correlation"))

            if has_mm:
                u_mm = _float_or_nan(r.get("u_mm"))
                v_mm = _float_or_nan(r.get("v_mm"))
                u_over_mm.append(u_mm / u_val if np.isfinite(u_mm) and np.isfinite(u_val) and abs(u_val) > 1e-9 else float("nan"))
                v_over_mm.append(v_mm / v_val if np.isfinite(v_mm) and np.isfinite(v_val) and abs(v_val) > 1e-9 else float("nan"))

            t_raw = r.get("time_s") if has_time else None
            if t_raw not in (None, ""):
                try:
                    time_s = float(t_raw)
                except ValueError:
                    pass

        if fi != ref_index and time_s is not None:
            dframe_over_time.append((fi - ref_index) / time_s if time_s != 0 else float("nan"))

        # An entire frame's strain array being all-NaN (never populated at
        # export time, e.g. a run where strain estimation failed for that
        # frame) is collapsed back to None -- same "None means the whole
        # array is unavailable" convention `DICFrameResult` already uses,
        # vs. per-point NaN for an individual excluded point.
        exx_out = None if np.all(np.isnan(exx)) else exx
        eyy_out = None if np.all(np.isnan(eyy)) else eyy
        exy_out = None if np.all(np.isnan(exy)) else exy

        frames.append(DICFrameResult(
            frame_index=fi, x=x, y=y, u=u, v=v, correlation=corr,
            exx=exx_out, eyy=eyy_out, exy=exy_out,
        ))

    if np.any(np.isnan(xs)) or np.any(np.isnan(ys)):
        raise DicResultLoadError(
            "Some grid points never got a reference position (x0/y0) -- the file may be truncated or corrupted."
        )

    # Step spacing recovered from the grid itself -- median rather than a
    # single difference, so one irregular gap (e.g. a ROI-shape cutout
    # trimming the bounding-box grid unevenly) doesn't throw it off.
    if n_points_x >= 2:
        step_x = float(np.median(np.diff(np.sort(xs))))
    elif n_points_y >= 2:
        step_x = float(np.median(np.diff(np.sort(ys))))
    else:
        step_x = 1.0
    step = max(1, int(round(step_x))) if np.isfinite(step_x) and step_x > 0 else 1

    mesh = MeshGrid(xs=xs, ys=ys, subset_size=step, step=step, mask=None)
    result = DICResult(mesh=mesh, frame_indices=frame_indices, frames=frames)

    # Pool both axes' per-row scale estimates together (u_mm/u and v_mm/v
    # should agree, both being the same mm-per-working-pixel factor) for a
    # steadier estimate than either alone.
    mm_per_working_px = _median_positive(u_over_mm + v_over_mm) if has_mm else None
    fps = _median_positive(dframe_over_time)

    return DicResultImport(result=result, mm_per_working_px=mm_per_working_px, fps=fps, source_path=path)
