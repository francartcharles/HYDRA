"""
Toe (foot) compensation for a stress-strain curve.

A "toe" -- a slow, low-stiffness initial rise before the curve reaches
its real (steeper) elastic slope -- is a common, genuine artifact of
specimen/fixture seating (faces not perfectly flat/parallel, a residual
gap closing), independent of any wave-timing misalignment. The standard
correction: fit a tangent line to the steepest, most stable (most linear)
part of the curve, extrapolate it back to zero stress, and use that
strain value as the new "true" zero -- shifting (and trimming) the curve
accordingly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LinearRegionFit:
    start_index: int
    end_index: int
    slope: float
    intercept: float
    r_squared: float
    strain_offset: float  # strain value where the tangent line crosses zero stress


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r_squared


def fit_linear_region(strain: np.ndarray, stress: np.ndarray, start_index: int, end_index: int) -> LinearRegionFit:
    """Fits a straight line to `strain[start_index:end_index]` /
    `stress[start_index:end_index]` (e.g. a region picked manually on the
    plot) and returns it, including where it crosses zero stress."""
    x, y = np.asarray(strain[start_index:end_index]), np.asarray(stress[start_index:end_index])
    slope, intercept, r2 = _fit_line(x, y)
    strain_offset = -intercept / slope if slope != 0 else float(x[0])
    return LinearRegionFit(start_index=start_index, end_index=end_index, slope=slope, intercept=intercept,
                            r_squared=r2, strain_offset=strain_offset)


def detect_linear_region(
    strain: np.ndarray, stress: np.ndarray, window_fraction: float = 0.12, r_squared_threshold: float = 0.995,
    search_fraction: float = 0.6,
) -> LinearRegionFit:
    """Automatically finds the longest, most stable (highest R^2) linear
    segment of the curve -- searched only within the first
    `search_fraction` of the data (the elastic region is expected early,
    before yielding/nonlinearity), using a sliding window of
    `window_fraction` of the total length (deliberately fairly wide, so a
    short, spurious near-linear blip elsewhere on the curve -- e.g. right
    at a kink -- can't win just for having a locally steep slope over a
    handful of points). Picks the window with the HIGHEST R^2 among those
    reaching `r_squared_threshold`; if none do, falls back to the window
    with the overall highest R^2 (not the steepest one -- steepness alone
    is a poor, easily-fooled criterion here)."""
    strain = np.asarray(strain)
    stress = np.asarray(stress)
    n = len(strain)
    window = max(5, int(n * window_fraction))
    search_end = max(window + 1, int(n * search_fraction))

    candidates = []
    for start in range(0, search_end - window):
        end = start + window
        slope, intercept, r2 = _fit_line(strain[start:end], stress[start:end])
        candidates.append((start, end, slope, intercept, r2))

    if not candidates:
        # Degenerate (very short curve) -- just use everything.
        slope, intercept, r2 = _fit_line(strain, stress)
        strain_offset = -intercept / slope if slope != 0 else float(strain[0])
        return LinearRegionFit(0, n, slope, intercept, r2, strain_offset)

    good = [c for c in candidates if c[4] >= r_squared_threshold]
    pool = good if good else candidates
    start, end, slope, intercept, r2 = max(pool, key=lambda c: c[4])
    strain_offset = -intercept / slope if slope != 0 else float(strain[start])
    return LinearRegionFit(start_index=start, end_index=end, slope=slope, intercept=intercept,
                            r_squared=r2, strain_offset=strain_offset)


def apply_toe_correction(strain: np.ndarray, stress: np.ndarray, strain_offset: float, trim: bool = True):
    """Shifts `strain` so the tangent line's zero-stress crossing becomes
    the new strain=0, and (if `trim`) discards any points that would end
    up at negative strain (the removed "toe")."""
    strain = np.asarray(strain)
    stress = np.asarray(stress)
    corrected = strain - strain_offset
    if trim:
        keep = corrected >= 0
        return corrected[keep], stress[keep]
    return corrected, stress


def infer_true_strain_relation(nominal_strain, true_strain) -> str:
    """Figures out how `true_strain` relates to `nominal_strain`, by
    checking which known relation the data actually matches at a
    representative point:
    - compression: true = -log(1 - nominal)
    - tension:     true = log(1 + nominal)
    - shear:       true = nominal (SHPB shear processing uses no log
      transform at all -- see process_shpb.py, `t_s = list(n_s)`)
    Needed because a toe correction only directly shifts NOMINAL strain
    (a simple, linear re-zeroing); true strain must be *recomputed* from
    the corrected nominal strain via this same relation in the
    compression/tension case, since their relationship isn't linear."""
    nominal_strain = np.asarray(nominal_strain)
    true_strain = np.asarray(true_strain)
    nonzero = np.where(np.abs(nominal_strain) > 1e-6)[0]
    if len(nonzero) == 0:
        return "shear"
    idx = nonzero[len(nonzero) // 2]
    n_val, t_val = float(nominal_strain[idx]), float(true_strain[idx])

    if abs(t_val - n_val) < 1e-9 * max(1.0, abs(n_val)):
        return "shear"
    try:
        pred_compression = -math.log(1 - n_val) if n_val < 1 else math.inf
    except ValueError:
        pred_compression = math.inf
    try:
        pred_tension = math.log(1 + n_val) if n_val > -1 else math.inf
    except ValueError:
        pred_tension = math.inf
    errors = {"compression": abs(t_val - pred_compression), "tension": abs(t_val - pred_tension),
              "shear": abs(t_val - n_val)}
    return min(errors, key=errors.get)


def true_strain_from_nominal(nominal_strain, relation: str) -> np.ndarray:
    """Recomputes true strain from (corrected) nominal strain, using
    whichever relation `infer_true_strain_relation` found."""
    nominal_strain = np.asarray(nominal_strain, dtype=float)
    if relation == "tension":
        return np.log(1 + nominal_strain)
    if relation == "shear":
        return nominal_strain.copy()
    return -np.log(1 - np.clip(nominal_strain, None, 0.999999))
