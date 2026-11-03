"""
Full Monte-Carlo covariance matrices, ported from export_res.py.

The app already reports the per-point standard deviation of each result
quantity (the diagonal of this matrix, in effect) as the uncertainty
envelope shown everywhere. The original additionally computes the FULL
covariance matrix across points -- e.g. `np.cov(stress)` on a
(n_points, n_draws) array of every Monte-Carlo draw's stress curve --
which also captures how uncertainty at one strain/time level correlates
with uncertainty at another (a shared random draw of, say, bar diameter,
shifts the whole curve together rather than each point independently).

Matches the original's own preparation step exactly: each draw has its own
strain axis (since strain is itself Monte-Carlo-dependent), so before
stacking draws into one matrix, every draw's quantity is *linearly
interpolated onto one common strain grid* -- `N` points from a reference
("cut") strain value onward (`corr_file.bay_cut_value` in the original) --
so covariance is computed between values that actually correspond to the
same strain level across draws, not just the same array index.
"""
from __future__ import annotations

import numpy as np
import scipy.interpolate as sci

# Human-readable labels for every quantity a covariance matrix can be
# computed for (ui/pages/page_covariance.py's own Quantity combo) --
# lives here rather than in that page so it's a neutral, dependency-free
# single source of truth both that page AND ui/pages/page_bayes_calib.py
# (the Bayesian Calibration page's per-role "which physical quantity does
# this equation symbol represent" picker, Calibration tests card) can
# import without one UI page importing another.
QUANTITY_LABELS = {
    "strain": "Strain (true)", "nominal_strain": "Strain (nominal/engineering)",
    "stress": "Stress (true)", "nominal_stress": "Stress (nominal/engineering)",
    "strain_rate": "Strain rate", "nominal_strain_rate": "Nominal strain rate",
    "temperature": "Temperature", "displacement": "Displacement",
    "load": "Load", "force_input": "Input-bar force", "force_output": "Output-bar force",
    "axial_stress": "Triaxial: axial stress (gage)", "radial_stress": "Triaxial: radial stress",
    "pressure": "Triaxial: hydrostatic pressure (gage)", "deviator": "Triaxial: stress deviator (gage)",
    "radial_strain": "Triaxial: radial strain", "volumetric_strain": "Triaxial: volumetric strain",
    "equivalent_strain": "Triaxial: equivalent strain",
    "radial_strain_rate": "Triaxial: radial strain rate", "volumetric_strain_rate": "Triaxial: volumetric strain rate",
    "equivalent_strain_rate": "Triaxial: equivalent strain rate",
    "axial_stress_laser": "Triaxial: axial stress (laser)", "pressure_laser": "Triaxial: hydrostatic pressure (laser)",
    "deviator_laser": "Triaxial: stress deviator (laser)",
    # Real per-Monte-Carlo-draw confinement quantities (computed inside the
    # Processing tab's own loop, see core.confinement.
    # per_draw_confinement_quantities) -- these show up in a plain (non-
    # "Confinement / Triaxial") result's own quantity list once computed,
    # since they're just more per-draw keys like any other.
    "confinement_radial_stress": "Triaxial (in-loop): radial stress",
    "confinement_pressure": "Triaxial (in-loop): hydrostatic pressure",
    "confinement_deviator": "Triaxial (in-loop): stress deviator",
    "confinement_radial_strain": "Triaxial (in-loop): radial strain",
    "confinement_volumetric_strain": "Triaxial (in-loop): volumetric strain",
    "confinement_equivalent_strain": "Triaxial (in-loop): equivalent strain",
    "confinement_radial_strain_rate": "Triaxial (in-loop): radial strain rate",
    "confinement_volumetric_strain_rate": "Triaxial (in-loop): volumetric strain rate",
    "confinement_equivalent_strain_rate": "Triaxial (in-loop): equivalent strain rate",
    "signal": "Signal (calibrated)",  # a plain Custom Sensor's own quantity (page_covariance.py's _on_result_changed)
}


def interpolate_draws_to_common_grid(
    per_draw_strain: list, per_draw_quantity: list,
    strain_start: float, strain_end: float, n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolates each draw's own (strain, quantity) curve onto
    one shared strain grid of `n_points` points from `strain_start` to
    `strain_end` (values before `strain_start` are ignored, matching the
    original's `bay_cut_value` reference-strain cutoff). Draws whose own
    strain range doesn't fully cover [strain_start, strain_end] are
    skipped (a linear interpolator can't be evaluated outside its range).

    Returns (strain_grid, stacked) where `stacked` has shape
    (n_valid_draws, n_points).
    """
    strain_grid = np.linspace(strain_start, strain_end, num=n_points)
    rows = []
    for strain, quantity in zip(per_draw_strain, per_draw_quantity):
        strain = np.asarray(strain)
        quantity = np.asarray(quantity)
        n = min(len(strain), len(quantity))
        strain, quantity = strain[:n], quantity[:n]
        if strain_grid.min() < strain.min() or strain_grid.max() > strain.max():
            continue  # this draw doesn't cover the requested range
        interp = sci.interp1d(strain, quantity, kind="linear")
        rows.append(interp(strain_grid))
    if not rows:
        return strain_grid, np.empty((0, n_points))
    return strain_grid, np.array(rows)


def compute_covariance_matrix(per_draw_arrays: list) -> np.ndarray | None:
    """`per_draw_arrays`: one 1D array per Monte-Carlo draw, ALREADY on a
    common grid (e.g. from `interpolate_draws_to_common_grid` above).
    Stacks them into (n_draws, n_points), transposes to
    (n_points, n_draws) -- matching the original's
    `np.transpose(...)` before `np.cov(...)` -- and returns the
    (n_points, n_points) covariance matrix. Returns None if there's fewer
    than 2 draws (a covariance needs at least 2 observations)."""
    if per_draw_arrays is None or len(per_draw_arrays) < 2:
        return None
    stacked = np.asarray(per_draw_arrays)  # (n_draws, n_points)
    if stacked.ndim == 1:
        return None
    return np.cov(stacked.T)  # (n_points, n_points)

