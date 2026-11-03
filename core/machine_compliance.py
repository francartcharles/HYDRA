"""
Quasi-static machine (load-frame) compliance correction, ported from
process_t.py's `if corr_file.disp_comp_check == 1 and corr_file.load_comp_check == 1:`
block.

A rigid calibration specimen (or the frame alone) is loaded through the
same load range as the real test, recording the load-frame's OWN
displacement at each load (its "compliance"). This is used to build a
load -> machine-displacement curve (via a PCHIP monotonic interpolant,
matching the original), which is then evaluated at the real test's
measured load signal and subtracted from the measured displacement --
leaving (approximately) just the specimen's own deformation.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def apply_compliance_correction(
    load: np.ndarray, displacement: np.ndarray,
    calib_load: np.ndarray, calib_displacement: np.ndarray,
) -> np.ndarray:
    """Returns a corrected displacement array (same length as `displacement`),
    with the machine's own load-dependent deflection removed and the result
    re-zeroed to its own minimum -- matches the original exactly."""
    calib_load = np.asarray(calib_load, dtype=float)
    calib_displacement = np.asarray(calib_displacement, dtype=float)
    order = np.argsort(calib_load)
    calib_load = calib_load[order]
    calib_displacement = calib_displacement[order]

    # Manual fix (preserved across future deliveries): PchipInterpolator
    # requires a strictly increasing x-array -- drop any calibration point
    # that doesn't strictly exceed the previously-kept one (duplicate or
    # non-increasing load values after sorting), keeping both arrays in
    # sync, before interpolating.
    keep_indices = [0]
    for i in range(1, len(calib_load)):
        if calib_load[i] > calib_load[keep_indices[-1]]:
            keep_indices.append(i)
    calib_load = calib_load[keep_indices]
    calib_displacement = calib_displacement[keep_indices]

    interp = PchipInterpolator(calib_load, calib_displacement, extrapolate=True)
    corr_disp = interp(load)
    corr_disp0 = interp(calib_load)
    baseline = corr_disp0[0]
    corr_disp = corr_disp - baseline

    corrected = np.asarray(displacement, dtype=float) - corr_disp
    corrected = corrected - corrected.min()
    return corrected
