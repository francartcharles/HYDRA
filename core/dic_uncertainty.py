"""
Measurement-uncertainty model for Digital Image Correlation (DIC)
results: displacement, strain, strain rate, velocity, acceleration and
dynamic force.

WHY THIS EXISTS -- and what it is NOT
--------------------------------------
There is no single, universally "correct" way to state a DIC
measurement's uncertainty: unlike a strain gage (a calibrated
transducer with a datasheet accuracy), DIC's uncertainty depends on the
speckle pattern's own texture, the camera's noise, the correlation
mesh's own settings (subset size, step), and how much the material
actually deforms during the test. The metrology literature on this
(Sutton et al.'s DIC textbook; Bornert et al., "Assessment of Digital
Image Correlation Measurement Errors: Methodology and Results", Exp
Mech 2009; Wang, Sutton et al., "Quantitative Error Assessment in
Pattern Matching", Strain 2009; Lecompte et al. 2006 on speckle
pattern/subset-size effects) converges on combining THREE distinct
error mechanisms, which is what this module implements:

1. A photometric, sub-pixel REGISTRATION floor -- how precisely the
   correlation peak itself can be located given the reference subset's
   own texture and the camera's pixel noise. This has a closed form,
   the same one industrial DIC packages (e.g. Correlated Solutions'
   VIC-2D) use to report a per-point "u/v error" map: the correlation
   peak's curvature grows with the subset's own intensity gradient
   content (a well-textured speckle subset gives a sharp, precisely
   located peak; a flat, low-contrast one gives a shallow, poorly
   located peak), so

       sigma_u_floor = noise_sigma / sqrt(SSSIG_x)
       sigma_v_floor = noise_sigma / sqrt(SSSIG_y)

   where SSSIG ("Sum of Squared Subset Intensity Gradients") is summed
   over the REFERENCE subset only (see `subset_sssig_xy`) and
   `noise_sigma` is the camera/compression pixel-intensity noise level
   (see `estimate_noise_sigma`). This is the Cramér-Rao-type lower
   bound on registration precision -- a FLOOR, not the whole story.

2. A CORRELATION-QUALITY inflation -- real matches degrade for reasons
   the floor above doesn't capture on its own (out-of-plane motion,
   lighting change, partial decorrelation, motion blur): the further
   the actual per-frame peak NCC value falls from a perfect match, the
   less that frame's registration should be trusted, even at a point
   whose reference texture looked great (see `correlation_inflation`).

3. A SUBSET-AVERAGING (distortion) term -- this directly answers "what
   about the level of distortion of the elements": each subset is
   tracked as one rigid, translating template (see
   core/dic_processing.py's own module docstring), so if the true
   displacement field varies noticeably *within* one subset -- i.e. the
   local strain is large relative to the subset size -- the single
   position the correlation reports is some blend across that
   variation, not any single material point's exact position. The
   scale of that blend is approximately (local strain magnitude x
   subset size), the classic "keep the subset small relative to the
   deformation gradient" rule of thumb from the speckle-pattern
   literature (see `distortion_term`).

All three are combined in quadrature (assumed uncorrelated -- a
simplification; #1 and #2 in particular are not fully independent in
reality, but treating them as such gives a workable, and if anything
conservative-leaning (the terms don't partially cancel), envelope).

THE RIGOROUS ALTERNATIVE THIS DOES NOT REPLACE
------------------------------------------------
If a validated, instrument-grade uncertainty budget is needed (e.g. for
a report that must state a certified measurement uncertainty), the
standard practice is an EMPIRICAL calibration this module cannot
substitute for: run the exact same DIC pipeline (same subset size,
step, speckle pattern) on (a) two images of the SAME undeformed
specimen (a "zero/rigid-body" test) to directly measure the actual
noise floor including every real-world effect at once, and/or (b) a
synthetically-warped version of a real reference image with a KNOWN
applied displacement/strain field, to measure the actual bias and
scatter against ground truth. That gives one empirical number
(replacing terms 1-2 above) validated for THIS specific setup, whereas
everything below is a predictive model with no ground truth to check
itself against.

WHAT THIS MODULE DOES NOT MODEL
---------------------------------
Lens distortion/calibration error, out-of-plane (3D) motion in a 2D-DIC
setup, vibration/motion blur beyond what's already reflected in a lower
correlation coefficient, and the fixed sub-pixel INTERPOLATION BIAS
some correlation schemes have near integer-pixel displacements (this
app's 3-point parabolic peak fit has a small such bias, on the order of
a few hundredths of a pixel in the worst case -- see
core/dic_processing.py's `_subpixel_peak`) are all outside this model.
`_MIN_SIGMA_PX` below is a floor precisely to acknowledge that no real
system beats a small fraction of a pixel regardless of how good the
speckle pattern is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import convolve2d

# No real DIC system -- however good the speckle pattern -- registers
# sub-pixel displacement to better than this many pixels (interpolation
# bias, camera quantization, etc., none of which the pure photometric
# floor below captures). Also keeps sigma_u_floor from blowing up to an
# unphysical near-zero value wherever SSSIG happens to be very large.
_MIN_SIGMA_PX = 0.005


def _to_grayscale_f64(array: np.ndarray) -> np.ndarray:
    """Collapses a possibly multi-channel frame/patch (h, w) or (h, w,
    n_channels) to a single-channel float64 array by averaging the
    channel axis. `estimate_noise_sigma`/`subset_sssig_xy` below are
    written for a 2-D image (`scipy.signal.convolve2d` and
    `np.gradient`'s 2-output unpacking both require exactly 2
    dimensions) -- but the DIC correlation engine itself (`cv2.
    matchTemplate`) happily runs on color (h, w, 3) frames too, since it
    is never told to convert to grayscale. Without this conversion, a
    color source (a common case for consumer/industrial cameras) would
    make every call here raise, an error `run_dic`'s best-effort `except
    Exception` would then swallow silently."""
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 3:
        return array.mean(axis=2)
    return array


def estimate_noise_sigma(image: np.ndarray) -> float:
    """Blind estimate of the reference frame's own pixel-intensity noise
    standard deviation, via Immerkaer's fast noise estimator (J.
    Immerkaer, "Fast Noise Variance Estimation", Computer Vision and
    Image Understanding, 1996): convolve with a discrete Laplacian-type
    kernel chosen so smooth image content cancels out, leaving (mostly)
    just noise, then take a scale estimate of the result. Cheap (one
    small convolution) and needs no dark-frame/flat-field calibration
    shot, at the cost of being a rough estimate -- a genuinely textured
    speckle pattern has real high-frequency content of its own that
    this can't perfectly separate from noise, so this tends to be a
    (mildly) pessimistic (too-high) estimate rather than an
    optimistic one, which is the safer direction for an uncertainty
    floor."""
    img = _to_grayscale_f64(image)
    h, w = img.shape[:2]
    if h < 3 or w < 3:
        return 0.0
    # Immerkaer's kernel: zero mean, unit sum of squares scaling folded
    # into the normalization below.
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = convolve2d(img, kernel, mode="valid")
    sigma = np.sqrt(np.pi / 2.0) * np.mean(np.abs(conv)) / 6.0
    return float(sigma) if np.isfinite(sigma) else 0.0


def subset_sssig_xy(patch: np.ndarray) -> tuple[float, float]:
    """Sum of Squared Subset Intensity Gradients (SSSIG) of a reference
    subset patch, in x and y separately -- the two denominators in the
    Cramér-Rao-type registration-precision floor (see the module
    docstring, term 1). A high-contrast, richly-textured speckle subset
    gives a large SSSIG (sharp, well-localized correlation peak, so a
    small `sigma_u_floor`); a flat or low-contrast subset gives a small
    one (shallow peak, large floor)."""
    patch = _to_grayscale_f64(patch)
    if patch.shape[0] < 2 or patch.shape[1] < 2:
        return 0.0, 0.0
    gy, gx = np.gradient(patch)
    return float(np.sum(gx ** 2)), float(np.sum(gy ** 2))


def crlb_sigma_px(noise_sigma: float, sssig: np.ndarray) -> np.ndarray:
    """Cramér-Rao-type sub-pixel registration floor (term 1): `noise_
    sigma / sqrt(sssig)`, floored at `_MIN_SIGMA_PX` and NaN wherever
    `sssig` itself is NaN (an excluded/untracked point)."""
    sssig = np.asarray(sssig, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = noise_sigma / np.sqrt(sssig)
    return np.where(np.isfinite(sigma), np.maximum(sigma, _MIN_SIGMA_PX), np.nan)


def correlation_inflation(correlation: np.ndarray, corr_floor: float = 0.5, exponent: float = 2.0) -> np.ndarray:
    """Term 2 (see module docstring): a multiplicative inflation factor
    on the registration floor, growing as this frame's own peak NCC
    value falls away from a perfect match (1.0). `correlation` below
    `corr_floor` is clamped to it -- this is a smooth degradation model
    for an otherwise-still-tracked point, not a substitute for
    `run_dic`'s own `correlation_threshold`, which drops a point
    outright once its match becomes unreliable. `exponent` = 2.0 is a
    modeling choice (not derived from first principles): it keeps the
    inflation mild for a good-but-imperfect match (NCC = 0.98 ->
    inflation ~1.04) while still visibly penalizing a poor one (NCC =
    0.7 -> inflation ~2)."""
    c = np.asarray(correlation, dtype=float)
    c = np.clip(c, corr_floor, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(c), (1.0 / c) ** exponent, np.nan)


def distortion_term_px(local_strain_mag: np.ndarray, subset_size_px: float, k: float = 0.5) -> np.ndarray:
    """Term 3 (see module docstring): the subset-averaging/distortion
    contribution to displacement uncertainty, `k * |local strain| *
    subset_size_px` -- the scale of the within-subset displacement
    variation a single rigid-template match cannot resolve. `k` = 0.5
    approximates a linear displacement gradient across the subset (the
    match settles near the subset's own center, so the worst-case
    mismatch at the subset edge, `|strain| * subset_size/2`, is a
    reasonable order-of-magnitude estimate of the resulting position
    error) -- a modeling choice, not a rigorous derivation."""
    strain = np.asarray(local_strain_mag, dtype=float)
    return k * np.abs(strain) * subset_size_px


def combine_displacement_uncertainty(sigma_floor_px: np.ndarray, inflation: np.ndarray,
                                      distortion_px: np.ndarray) -> np.ndarray:
    """Combines the three terms in quadrature (assumed independent --
    see module docstring): `sqrt((sigma_floor*inflation)^2 +
    distortion^2)`."""
    registration = np.asarray(sigma_floor_px, dtype=float) * np.asarray(inflation, dtype=float)
    distortion = np.asarray(distortion_px, dtype=float)
    return np.sqrt(registration ** 2 + distortion ** 2)


def strain_uncertainty_from_displacement(sigma_u_px: np.ndarray, sigma_v_px: np.ndarray,
                                          step_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagates per-point displacement uncertainty (`sigma_u_px`,
    `sigma_v_px`, both in the same pixel units as the mesh's own `step`)
    to the strain fields' own uncertainty, via the same central-
    difference formula `core/dic_processing.py`'s `_grid_strain` uses
    (`np.gradient` on a regular grid of spacing `step_px`).

    For an interior point, exx = (u[i,j+1] - u[i,j-1]) / (2*step); if
    the two u samples have independent noise of variance sigma_u^2
    each, Var(exx) = 2*sigma_u^2 / (2*step)^2 = sigma_u^2 / (2*step^2),
    so sigma_exx = sigma_u / (step * sqrt(2)) -- and likewise for eyy
    from sigma_v. exy = 0.5*(du/dy + dv/dx) combines a y-derivative of u
    with an x-derivative of v, each with the SAME per-point sigma_u/
    sigma_v (a point's own registration uncertainty doesn't depend on
    which direction it's later differentiated in), added in quadrature
    and halved.

    Uses the nominal grid spacing `step_px` everywhere, including at
    the mesh's edges (where `np.gradient` actually uses a one-sided,
    single-step difference, not the two-sided one this formula
    assumes) -- a documented approximation, slightly UNDER-stating the
    true edge-point uncertainty (a one-sided difference of the same
    step size is noisier than the two-sided one, not less)."""
    su = np.asarray(sigma_u_px, dtype=float)
    sv = np.asarray(sigma_v_px, dtype=float)
    denom = step_px * np.sqrt(2.0)
    sigma_exx = su / denom
    sigma_eyy = sv / denom
    sigma_exy = 0.5 * np.sqrt(su ** 2 + sv ** 2) / denom
    return sigma_exx, sigma_eyy, sigma_exy


def von_mises_strain_uncertainty(exx: np.ndarray, eyy: np.ndarray, exy: np.ndarray,
                                  sigma_exx: np.ndarray, sigma_eyy: np.ndarray,
                                  sigma_exy: np.ndarray) -> np.ndarray:
    """First-order (linearized) error propagation of `sigma_exx`/
    `sigma_eyy`/`sigma_exy` through `core.dic_kinematics.von_mises_
    strain`'s own formula (eps_eq = sqrt((2/3)(exx^2+eyy^2) +
    (4/3)*exy^2)), i.e. sigma_eps_eq^2 = (d eps_eq/d exx)^2*sigma_exx^2
    + (d eps_eq/d eyy)^2*sigma_eyy^2 + (d eps_eq/d exy)^2*sigma_exy^2,
    assuming exx/eyy/exy's own errors are independent (a simplification
    -- in reality they share the same underlying u/v noise, see
    `strain_uncertainty_from_displacement`'s own docstring -- but
    treating them as independent here is the same kind of workable
    approximation used throughout this module). NaN (rather than a
    divide-by-zero) wherever eps_eq itself is 0."""
    exx = np.asarray(exx, dtype=float)
    eyy = np.asarray(eyy, dtype=float)
    exy = np.asarray(exy, dtype=float)
    eps_eq = np.sqrt((2.0 / 3.0) * (exx ** 2 + eyy ** 2) + (4.0 / 3.0) * exy ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_exx = (2.0 / 3.0) * exx / eps_eq
        d_eyy = (2.0 / 3.0) * eyy / eps_eq
        d_exy = (4.0 / 3.0) * exy / eps_eq
        out = np.sqrt((d_exx * sigma_exx) ** 2 + (d_eyy * sigma_eyy) ** 2 + (d_exy * sigma_exy) ** 2)
    return np.where(eps_eq > 0, out, np.nan)


def time_derivative_uncertainty(sigma_value, dt_s: float):
    """Propagates a quantity's own uncertainty through ONE central-
    difference-in-time step (the same formula `core.dic_kinematics`'s
    `_gradient_over_time` uses, assuming a roughly uniform `dt_s`, see
    that function's own docstring on non-uniform spacing): d/dt at an
    interior frame is (x[i+1]-x[i-1])/(2*dt), so sigma_rate = sigma_
    value / (dt_s * sqrt(2)) -- structurally the same formula as
    `strain_uncertainty_from_displacement`'s spatial version, just
    differentiating in time instead of space. Chaining this call twice
    (uncertainty in position -> velocity -> acceleration) treats each
    stage's noise as independent of the last, which isn't exactly true
    (consecutive stages reuse overlapping position samples) -- a
    simplification flagged here rather than hidden, and one that this
    module does not attempt to correct for."""
    return np.asarray(sigma_value, dtype=float) / (dt_s * np.sqrt(2.0))


def effective_independent_count(n_active_points: int, roi_area_px2: float, subset_size_px: float) -> float:
    """The "effective" number of independent samples behind an ROI-mean
    -- always <= the raw number of tracked points `n_active_points`,
    since neighboring DIC points are NOT independent measurements: their
    subsets overlap when `step < subset_size` (a common setting), and
    even where they don't, neighboring points on a continuous material
    are physically linked, so their tracking errors tend to move
    together rather than average away like independent noise would.
    Approximates the correlation length as one subset width, i.e. treats
    the ROI as covered by `roi_area_px2 / subset_size_px^2` independent
    "patches" -- a standard, if rough, correction for spatially-
    correlated noise when averaging over an area (see e.g. the
    "effective sample size" concept in spatial statistics). Clamped to
    [1, n_active_points] so this never claims BETTER-than-raw-count
    precision, nor a non-positive/zero effective count."""
    if n_active_points <= 0 or subset_size_px <= 0:
        return 1.0
    n_eff = roi_area_px2 / (subset_size_px ** 2)
    return float(np.clip(n_eff, 1.0, n_active_points))


def roi_mean_uncertainty(sigma_field: np.ndarray, n_eff: float) -> float:
    """Standard error of an ROI mean given the per-point uncertainty
    field `sigma_field` (only its finite/active entries are used) and
    the effective independent sample count `n_eff` (see
    `effective_independent_count`): the per-point sigmas' own RMS
    (root-mean-square, the right way to combine several DIFFERENT
    sigma values into one representative one for points of varying
    quality) divided by `sqrt(n_eff)` (the usual standard-error-of-the-
    mean reduction, but over the effective rather than raw point
    count). NaN if no point has a finite sigma."""
    finite = np.asarray(sigma_field, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    rms = float(np.sqrt(np.mean(finite ** 2)))
    return rms / np.sqrt(max(n_eff, 1.0))


def _hypot_uncertainty(x, y, sigma_x, sigma_y) -> np.ndarray:
    """Error propagation through `mag = hypot(x, y)`: d(mag)/dx = x/mag,
    d(mag)/dy = y/mag, combined in quadrature with sigma_x/sigma_y (used
    for displacement magnitude, velocity magnitude, acceleration
    magnitude -- every "_mag" field `core.dic_kinematics` derives this
    way). Falls back to 0 at the degenerate `mag == 0` point (rather
    than a divide-by-zero) -- an edge case (exactly zero displacement/
    velocity), not a meaningfully wrong answer either way at that single
    point."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mag = np.hypot(x, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = np.where(mag > 0, x / mag, 0.0)
        dy = np.where(mag > 0, y / mag, 0.0)
    return np.sqrt((dx * np.asarray(sigma_x, dtype=float)) ** 2 + (dy * np.asarray(sigma_y, dtype=float)) ** 2)


@dataclass
class DicUncertainty:
    """Per-frame uncertainty (1-sigma) arrays mirroring
    `core.dic_kinematics.Kinematics`'s own layout: one array per
    processed frame, each shaped like a `DICFrameResult`'s own fields,
    in the same order as the source `DICResult.frames` -- so
    `uncertainty.sigma_u[i]` lines up with `dic_result.frames[i]` (and
    with `kinematics.vx[i]` etc. where those exist).

    `sigma_u`/`sigma_v`/`sigma_disp_mag` are in `length_unit` (mm if a
    pixel size was given to `compute_uncertainty`, else px, matching
    `Kinematics.length_unit`'s own convention); the strain sigmas are
    always dimensionless. The rate/velocity/acceleration/force sigmas
    are None unless a computed `Kinematics` was given to
    `compute_uncertainty` (no time axis, no rate uncertainty -- same
    condition as `Kinematics` itself needing a frame rate)."""
    length_unit: str
    sigma_u: list
    sigma_v: list
    sigma_disp_mag: list
    sigma_exx: list
    sigma_eyy: list
    sigma_exy: list
    sigma_eps_eq: list

    sigma_vx: list | None = None
    sigma_vy: list | None = None
    sigma_v_mag: list | None = None
    sigma_ax: list | None = None
    sigma_ay: list | None = None
    sigma_a_mag: list | None = None

    sigma_exx_rate: list | None = None
    sigma_eyy_rate: list | None = None
    sigma_exy_rate: list | None = None
    sigma_eps_eq_rate: list | None = None

    sigma_dynamic_force: list | None = None  # None unless rho_kg_m3, area_m2 AND a pixel size were given, same condition as Kinematics.dynamic_force


def compute_uncertainty(result, kin=None, mm_per_px: float | None = None,
                         rho_kg_m3: float | None = None, area_m2: float | None = None,
                         corr_floor: float = 0.5, corr_exponent: float = 2.0,
                         k_distortion: float = 0.5) -> DicUncertainty | None:
    """Builds a full `DicUncertainty` from a `core.dic_processing.
    DICResult` (`result` -- needs its `noise_sigma`/`sssig_x`/`sssig_y`,
    see that module) and, optionally, an already-computed
    `core.dic_kinematics.Kinematics` (`kin`, duck-typed here rather than
    imported, to avoid a core/-internal import cycle -- needs `.t_s`,
    `.vx`/`.vy`, `.ax`/`.ay`, `.exx_rate`/`.eyy_rate`/`.exy_rate`) for
    the rate/velocity/acceleration/force fields. `mm_per_px`/
    `rho_kg_m3`/`area_m2` mirror `compute_kinematics`'s own parameters
    and should be the SAME values used to compute `kin`, so the
    uncertainty bands line up with the values they bound.

    Returns None if `result` has no noise/SSSIG estimate at all (see
    `DICResult`'s own docstring on when that can happen).

    `corr_floor`/`corr_exponent` -> `correlation_inflation`;
    `k_distortion` -> `distortion_term_px`; see those functions'
    docstrings for what each controls."""
    if result.noise_sigma is None or result.sssig_x is None or result.sssig_y is None:
        return None

    scale = mm_per_px if mm_per_px else 1.0
    length_unit = "mm" if mm_per_px else "px"
    subset_size = result.mesh.subset_size
    step = result.mesh.step

    sigma_floor_x = crlb_sigma_px(result.noise_sigma, result.sssig_x)
    sigma_floor_y = crlb_sigma_px(result.noise_sigma, result.sssig_y)

    sigma_u_list, sigma_v_list, sigma_disp_mag_list = [], [], []
    sigma_exx_list, sigma_eyy_list, sigma_exy_list, sigma_eps_eq_list = [], [], [], []
    sigma_u_px_list, sigma_v_px_list = [], []  # kept in PIXELS too, for the strain propagation below
    for fr in result.frames:
        inflation = correlation_inflation(fr.correlation, corr_floor, corr_exponent)
        if fr.exx is not None:
            local_strain_mag = np.sqrt(fr.exx ** 2 + fr.eyy ** 2 + 2.0 * fr.exy ** 2)
        else:
            local_strain_mag = np.zeros_like(sigma_floor_x)
        distortion = distortion_term_px(local_strain_mag, subset_size, k_distortion)

        sigma_u_px = combine_displacement_uncertainty(sigma_floor_x, inflation, distortion)
        sigma_v_px = combine_displacement_uncertainty(sigma_floor_y, inflation, distortion)
        sigma_u_px_list.append(sigma_u_px)
        sigma_v_px_list.append(sigma_v_px)

        sigma_u = sigma_u_px * scale
        sigma_v = sigma_v_px * scale
        sigma_u_list.append(sigma_u)
        sigma_v_list.append(sigma_v)
        sigma_disp_mag_list.append(_hypot_uncertainty(fr.u * scale, fr.v * scale, sigma_u, sigma_v))

        sexx, seyy, sexy = strain_uncertainty_from_displacement(sigma_u_px, sigma_v_px, step)
        sigma_exx_list.append(sexx)
        sigma_eyy_list.append(seyy)
        sigma_exy_list.append(sexy)
        if fr.exx is not None:
            sigma_eps_eq_list.append(von_mises_strain_uncertainty(fr.exx, fr.eyy, fr.exy, sexx, seyy, sexy))
        else:
            sigma_eps_eq_list.append(np.full(sexx.shape, np.nan))

    out = DicUncertainty(length_unit, sigma_u_list, sigma_v_list, sigma_disp_mag_list,
                          sigma_exx_list, sigma_eyy_list, sigma_exy_list, sigma_eps_eq_list)

    if kin is not None and len(kin.t_s) >= 2:
        dt = float(np.median(np.diff(kin.t_s)))
        if dt > 0:
            sigma_vx_list, sigma_vy_list, sigma_v_mag_list = [], [], []
            sigma_ax_list, sigma_ay_list, sigma_a_mag_list = [], [], []
            sigma_exxr_list, sigma_eyyr_list, sigma_exyr_list, sigma_eqr_list = [], [], [], []
            want_force = bool(rho_kg_m3 and area_m2 and mm_per_px)
            sigma_force_list = [] if want_force else None

            for i in range(len(result.frames)):
                svx = time_derivative_uncertainty(sigma_u_list[i], dt)
                svy = time_derivative_uncertainty(sigma_v_list[i], dt)
                sigma_vx_list.append(svx)
                sigma_vy_list.append(svy)
                sigma_v_mag_list.append(_hypot_uncertainty(kin.vx[i], kin.vy[i], svx, svy))

                sax = time_derivative_uncertainty(svx, dt)
                say = time_derivative_uncertainty(svy, dt)
                sigma_ax_list.append(sax)
                sigma_ay_list.append(say)
                a_mag_sigma = _hypot_uncertainty(kin.ax[i], kin.ay[i], sax, say)
                sigma_a_mag_list.append(a_mag_sigma)

                sexxr = time_derivative_uncertainty(sigma_exx_list[i], dt)
                seyyr = time_derivative_uncertainty(sigma_eyy_list[i], dt)
                sexyr = time_derivative_uncertainty(sigma_exy_list[i], dt)
                sigma_exxr_list.append(sexxr)
                sigma_eyyr_list.append(seyyr)
                sigma_exyr_list.append(sexyr)
                sigma_eqr_list.append(
                    von_mises_strain_uncertainty(kin.exx_rate[i], kin.eyy_rate[i], kin.exy_rate[i],
                                                  sexxr, seyyr, sexyr)
                )

                if want_force:
                    # Same mm/s^2 -> m/s^2 conversion (x0.001) compute_kinematics
                    # applies to a_mag itself before F = rho*A*a.
                    sigma_force_list.append(rho_kg_m3 * area_m2 * a_mag_sigma * 0.001)

            out.sigma_vx, out.sigma_vy, out.sigma_v_mag = sigma_vx_list, sigma_vy_list, sigma_v_mag_list
            out.sigma_ax, out.sigma_ay, out.sigma_a_mag = sigma_ax_list, sigma_ay_list, sigma_a_mag_list
            out.sigma_exx_rate, out.sigma_eyy_rate = sigma_exxr_list, sigma_eyyr_list
            out.sigma_exy_rate, out.sigma_eps_eq_rate = sigma_exyr_list, sigma_eqr_list
            out.sigma_dynamic_force = sigma_force_list

    return out
