"""
Kinematics derived from an already-computed DICResult: velocity,
acceleration, equivalent (von Mises) strain, strain rate, and an
optional "Dynamic Force" field built from a caller-supplied material
density -- see `compute_kinematics`.

Everything here is computed from a DICResult's own tracked positions
(`.x`/`.y`) and strains (`.exx`/`.eyy`/`.exy`, see
core/dic_processing._grid_strain), never re-derives them, and never
touches the DIC tracking itself -- this module only differentiates and
combines what `run_dic` already produced. A point that is NaN in a given
frame (lost track, excluded by the correlation-quality or neighbor-
consistency filter -- see core/dic_processing.py) stays NaN here too,
and `np.gradient`'s central-difference scheme means that only the
frame(s) immediately adjacent to the loss are affected -- not the whole
sequence.

Von Mises equivalent strain/strain-rate both use the same plane-strain
approximation: the out-of-plane strain ezz is taken as 0, since a single
2D camera view never measures it (a true 3D-DIC or stereo setup would
give ezz directly). This is the standard, simplest assumption for 2D
DIC and is clearly documented at the point of use (`von_mises_strain`)
rather than silently baked in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.dic_processing import DICResult


def von_mises_strain(exx, eyy, exy):
    """Equivalent (von Mises) strain from the in-plane strain tensor
    (`exx`, `eyy`, `exy` -- `exy` the TENSOR shear strain, i.e. half the
    engineering shear gamma_xy, matching core.dic_processing._grid_strain's
    own convention), under the plane-strain assumption ezz = 0 (see the
    module docstring):

        eps_eq = sqrt( (2/3)*(exx^2 + eyy^2) + (4/3)*exy^2 )

    the standard eps_eq = sqrt(2/3 * eij_dev : eij_dev) reduced to plane
    strain with engineering shear gamma_xy = 2*exy. The exact same
    formula, applied to the strain-RATE tensor instead of the strain
    tensor, gives the equivalent von Mises strain RATE -- see
    `compute_kinematics`, which calls this function both ways."""
    exx = np.asarray(exx, dtype=float)
    eyy = np.asarray(eyy, dtype=float)
    exy = np.asarray(exy, dtype=float)
    return np.sqrt((2.0 / 3.0) * (exx ** 2 + eyy ** 2) + (4.0 / 3.0) * exy ** 2)


@dataclass
class Kinematics:
    """Per-frame derived fields computed from a DICResult -- see
    `compute_kinematics`. Every array is shaped like a DICFrameResult's
    own fields (mesh.n_points_y, mesh.n_points_x), one array per
    processed frame, in the same order as the source DICResult's
    `.frames` -- so e.g. `kinematics.vx[i]` lines up with
    `dic_result.frames[i]`.

    Velocity/acceleration are in `length_unit`/s and `length_unit`/s^2,
    where `length_unit` is "mm" if `compute_kinematics` was given a
    pixel size, otherwise "px". Strain and strain rate are always
    dimensionless / per-second (a strain doesn't depend on the pixel-to-
    mm scale). `dynamic_force` (in Newtons, F = rho * area * a) is None
    unless a pixel size, a material density AND a cross-sectional area
    were all available."""
    frame_indices: list[int]
    t_s: np.ndarray  # seconds since the reference frame, one entry per processed frame
    length_unit: str  # "mm" or "px" -- whatever vx/ax/... are actually expressed in

    vx: list  # list[np.ndarray]
    vy: list
    v_mag: list
    ax: list
    ay: list
    a_mag: list

    eps_eq: list  # list[np.ndarray] -- von Mises equivalent strain

    exx_rate: list
    eyy_rate: list
    exy_rate: list
    eps_eq_rate: list  # von Mises equivalent strain RATE

    dynamic_force: list | None = None  # None unless rho_kg_m3, area_m2 AND a pixel size were given


def _gradient_over_time(stack: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """`np.gradient` of `stack` (shape `(n_frames, ...)`) along its first
    axis with respect to `t_s` (shape `(n_frames,)`, not required to be
    uniformly spaced -- `np.gradient` handles that directly when given
    the coordinate array). A NaN at one frame only ever reaches that same
    point's OWN rate at the immediately adjacent frame(s) (its central-
    difference neighbors), never propagates across the whole sequence or
    to other points."""
    if stack.shape[0] < 2:
        return np.full_like(stack, np.nan)
    return np.gradient(stack, t_s, axis=0)


def compute_kinematics(result: DICResult, fps: float, mm_per_px: float | None = None,
                        rho_kg_m3: float | None = None, area_m2: float | None = None) -> Kinematics:
    """Derives velocity, acceleration, equivalent (von Mises) strain and
    strain rate -- and, if `rho_kg_m3`, `area_m2` and `mm_per_px` are all
    given, a "Dynamic Force" field -- from `result`'s tracked positions/
    strains.

    `fps` (frames actually PROCESSED per second -- the camera's own
    frame rate already divided by any `frame_step` decimation, see
    core/dic_preprocessing) is required: without a time axis, "per
    second" has no meaning. Raises ValueError if `fps` is not positive.

    `mm_per_px`, if given, converts displacement (and so velocity/
    acceleration) from pixels to millimeters before differentiating;
    left as None, velocity/acceleration stay in px/s, px/s^2 (see
    `Kinematics.length_unit`).

    `rho_kg_m3` and `area_m2`, if both given (and `mm_per_px` is also
    available, since a physical density/area needs a physical
    acceleration to combine with), fill `dynamic_force` = rho * area *
    a_mag, with a_mag first converted to SI (m/s^2) -- Newton's second
    law, F = m * a, applied with the mass of the tracked material
    expressed as a density times a cross-sectional area (rho * area
    standing in for "mass per unit length" of a prismatic specimen of
    that constant cross-section, e.g. area = width * thickness for a
    flat specimen -- see the Image Correlation page's calibration
    fields). This IS dimensionally a real force: kg/m^3 * m^2 * m/s^2 =
    kg*m/s^2 = N. It's still a simplification -- it assumes the
    specimen's cross-section stays at its nominal `area_m2` everywhere
    along its length and doesn't shrink locally (e.g. necking), and it
    evaluates `a_mag` POINTWISE (so the field varies over the ROI, most
    meaningfully read as an ROI mean, i.e. the inertial force of a
    uniform-cross-section segment accelerating like the ROI's own
    average) -- but it is no longer a made-up combination of units."""
    if not fps or fps <= 0:
        raise ValueError("A positive frame rate (fps) is required to compute kinematics.")

    n_frames = len(result.frames)
    if n_frames == 0:
        raise ValueError("The DIC result has no processed frames.")
    frame_indices = [f.frame_index for f in result.frames]
    ref_index = frame_indices[0]
    t_s = np.array([(fi - ref_index) / fps for fi in frame_indices], dtype=float)

    scale = mm_per_px if mm_per_px else 1.0
    length_unit = "mm" if mm_per_px else "px"

    x_stack = np.stack([f.x for f in result.frames]) * scale
    y_stack = np.stack([f.y for f in result.frames]) * scale

    v_x_stack = _gradient_over_time(x_stack, t_s)
    v_y_stack = _gradient_over_time(y_stack, t_s)
    v_mag_stack = np.hypot(v_x_stack, v_y_stack)

    a_x_stack = _gradient_over_time(v_x_stack, t_s)
    a_y_stack = _gradient_over_time(v_y_stack, t_s)
    a_mag_stack = np.hypot(a_x_stack, a_y_stack)

    has_strain = all(f.exx is not None for f in result.frames)
    if has_strain:
        exx_stack = np.stack([f.exx for f in result.frames])
        eyy_stack = np.stack([f.eyy for f in result.frames])
        exy_stack = np.stack([f.exy for f in result.frames])
        eps_eq_stack = von_mises_strain(exx_stack, eyy_stack, exy_stack)
        exx_rate_stack = _gradient_over_time(exx_stack, t_s)
        eyy_rate_stack = _gradient_over_time(eyy_stack, t_s)
        exy_rate_stack = _gradient_over_time(exy_stack, t_s)
        eps_eq_rate_stack = von_mises_strain(exx_rate_stack, eyy_rate_stack, exy_rate_stack)
        eps_eq_list = list(eps_eq_stack)
        exx_rate_list = list(exx_rate_stack)
        eyy_rate_list = list(eyy_rate_stack)
        exy_rate_list = list(exy_rate_stack)
        eps_eq_rate_list = list(eps_eq_rate_stack)
    else:
        nan_grid = [np.full(a_mag_stack.shape[1:], np.nan) for _ in range(n_frames)]
        eps_eq_list = list(nan_grid)
        exx_rate_list = list(nan_grid)
        eyy_rate_list = list(nan_grid)
        exy_rate_list = list(nan_grid)
        eps_eq_rate_list = list(nan_grid)

    dynamic_force_list = None
    if rho_kg_m3 and area_m2 and mm_per_px:
        a_mag_m_s2 = a_mag_stack * 0.001  # mm/s^2 -> m/s^2 (length_unit is "mm" here, since mm_per_px was given)
        dynamic_force_stack = rho_kg_m3 * area_m2 * a_mag_m_s2  # F = rho * A * a, in Newtons
        dynamic_force_list = list(dynamic_force_stack)

    return Kinematics(
        frame_indices=frame_indices,
        t_s=t_s,
        length_unit=length_unit,
        vx=list(v_x_stack), vy=list(v_y_stack), v_mag=list(v_mag_stack),
        ax=list(a_x_stack), ay=list(a_y_stack), a_mag=list(a_mag_stack),
        eps_eq=eps_eq_list,
        exx_rate=exx_rate_list, eyy_rate=eyy_rate_list, exy_rate=exy_rate_list,
        eps_eq_rate=eps_eq_rate_list,
        dynamic_force=dynamic_force_list,
    )


def nominal_to_true_strain(e):
    """Converts a small/nominal (linearized, engineering-type) strain --
    e.g. DIC's own exx/eyy/eps_eq, see core/dic_processing._grid_strain's
    own docstring -- to logarithmic (true, Hencky) strain:

        eps_true = ln(1 + e)

    the standard 1D conversion (stretch ratio lambda = 1 + e, eps_true =
    ln(lambda)) -- exact for a genuinely uniaxial 1D elongation/
    shortening, an approximation otherwise (a 2D/3D strain STATE has no
    single well-defined "true strain" scalar the way a 1D length change
    does; applying it to a normal strain component, or to the always-
    non-negative equivalent/effective strain eps_eq, is standard
    practice and reduces to the exact 1D case there). NOT meaningful for
    a shear strain component (exy) -- shear has no stretch-ratio
    interpretation this formula relies on -- so callers never convert
    exy (see the Image Correlation page's `_publish_dic_strain`).

    Undefined (NaN) wherever `1 + e <= 0` (more than 100% nominal
    shortening) rather than raising -- physically that point has been
    crushed to zero or negative length, which no strain measure can
    meaningfully describe."""
    e = np.asarray(e, dtype=float)
    with np.errstate(invalid="ignore"):
        out = np.log1p(e)
    return np.where(1.0 + e > 0, out, np.nan)


def nominal_to_true_strain_rate(rate, nominal_strain):
    """The matching true-strain-RATE conversion for
    `nominal_to_true_strain`: d/dt[ln(1 + e)] = (de/dt) / (1 + e), so
    this needs both the nominal strain RATE and the nominal strain
    itself at the same instant (same subject to the same 1D-stretch
    approximation, and the same NaN-outside-domain behavior, as
    `nominal_to_true_strain`)."""
    rate = np.asarray(rate, dtype=float)
    e = np.asarray(nominal_strain, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = rate / (1.0 + e)
    return np.where(1.0 + e > 0, out, np.nan)


def smooth_series(series, window: int):
    """Centered moving average of a 1D series (typically a per-frame
    mean over the ROI -- e.g. for the acceleration/dynamic-force-vs-time
    plot's smoothing slider), NaN-aware: each output point averages
    whichever of its `window` neighbors (centered; `window` is rounded up
    to the next odd number if even) are finite, so a run of NaN (frames
    where every tracked point was lost) doesn't spread any further than
    it already covers, and a single noisy spike is pulled toward its
    surroundings rather than propagated. `window <= 1` returns an
    unchanged copy."""
    series = np.asarray(series, dtype=float)
    if window <= 1 or series.size == 0:
        return series.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    n = series.size
    out = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        chunk = series[lo:hi]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            out[i] = float(np.mean(finite))
    return out
