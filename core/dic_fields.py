"""
Orientation-independent strain measures and spatial smoothing for DIC
fields (pure numpy, no Qt, no SciPy).

Why this exists: exx / eyy / exy depend on the camera axes, so a crack that
runs at, say, 35 degrees to the horizontal shows up split between the three
components. The in-plane principal strains and the maximum shear strain are
invariants of the 2D strain tensor -- they highlight a localization band
whatever its direction.

Conventions (identical to core/dic_processing._grid_strain and
core/dic_kinematics.von_mises_strain): `exy` is the TENSOR shear strain
(half the engineering shear gamma_xy). With

    c = (exx + eyy) / 2        r = sqrt( ((exx - eyy) / 2)^2 + exy^2 )

the in-plane principal strains are  e1 = c + r  (maximum) and
e2 = c - r  (minimum), and the maximum in-plane shear strain reported here
is  gamma_max = (e1 - e2) / 2 = r  (radius of Mohr's circle, tensor
convention, consistent with exy). The engineering maximum shear is 2 * r.

Spatial smoothing (`smooth_nan_gaussian`) is a NaN-aware Gaussian filter
("normalized convolution"): lost-track / excluded points neither contribute
to their neighbours nor get filled in. Smooth the strain COMPONENTS first and
derive the invariants afterwards (see `smoothed_strain_fields`): the
invariants are non-linear (square roots), so smoothing them directly would
keep the positive bias that measurement noise puts on them.
"""
from __future__ import annotations

import numpy as np


def principal_strains(exx, eyy, exy):
    """(e1, e2): maximum and minimum in-plane principal strains, e1 >= e2."""
    exx = np.asarray(exx, dtype=float)
    eyy = np.asarray(eyy, dtype=float)
    exy = np.asarray(exy, dtype=float)
    center = 0.5 * (exx + eyy)
    radius = np.hypot(0.5 * (exx - eyy), exy)
    return center + radius, center - radius


def max_shear_strain(exx, eyy, exy):
    """Maximum in-plane shear strain (e1 - e2) / 2 (tensor convention, same
    as exy; multiply by 2 for the engineering definition e1 - e2)."""
    exx = np.asarray(exx, dtype=float)
    eyy = np.asarray(eyy, dtype=float)
    exy = np.asarray(exy, dtype=float)
    return np.hypot(0.5 * (exx - eyy), exy)


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Normalized 1D Gaussian kernel truncated at 3 sigma (radius >= 1)."""
    radius = max(1, int(np.ceil(3.0 * float(sigma))))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    return k / k.sum()


def _convolve_axis_zero_pad(a: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = (kernel.size - 1) // 2
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(a, pad, mode="constant", constant_values=0.0)
    out = np.zeros_like(a, dtype=float)
    for j, kj in enumerate(kernel):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(j, j + n)
        out += kj * padded[tuple(sl)]
    return out


def smooth_nan_gaussian(field, sigma: float) -> np.ndarray:
    """Gaussian smoothing of a 2D field on the mesh grid, `sigma` in mesh
    points (0 or negative = unchanged copy). NaN points are ignored by their
    neighbours' average (weights are renormalized, which also handles the
    grid edges) and stay NaN in the output -- nothing is invented inside a
    hole."""
    a = np.asarray(field, dtype=float)
    if a.ndim != 2 or not sigma or sigma <= 0:
        return a.copy()
    valid = np.isfinite(a)
    if not valid.any():
        return a.copy()
    kernel = gaussian_kernel_1d(sigma)
    num = np.where(valid, a, 0.0)
    wgt = valid.astype(float)
    for axis in (0, 1):
        num = _convolve_axis_zero_pad(num, kernel, axis)
        wgt = _convolve_axis_zero_pad(wgt, kernel, axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / wgt
    out[~valid | (wgt < 1e-12)] = np.nan
    return out


def smoothed_strain_fields(exx, eyy, exy, sigma: float = 0.0) -> dict:
    """Strain components (optionally smoothed with `smooth_nan_gaussian`)
    and the invariants derived FROM THE SMOOTHED components. Keys: exx, eyy,
    exy, e1, e2, gmax."""
    if sigma and sigma > 0:
        exx, eyy, exy = (smooth_nan_gaussian(c, sigma) for c in (exx, eyy, exy))
    else:
        exx, eyy, exy = (np.asarray(c, dtype=float) for c in (exx, eyy, exy))
    e1, e2 = principal_strains(exx, eyy, exy)
    return {"exx": exx, "eyy": eyy, "exy": exy, "e1": e1, "e2": e2,
            "gmax": max_shear_strain(exx, eyy, exy)}
