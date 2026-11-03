"""
K1c (critical stress-intensity factor) from the near-tip DIC displacement
field -- Stage 4, the last of the Crack & Wave Analysis standalone tool's
stages (see core/crack_tip_tracking.py for Stage 3's tip position / arc-
length history, which this stage reuses rather than re-detecting the tip
a second time).

Method: static (quasi-static) linear elastic fracture mechanics (LEFM).
At every frame where Stage 3 found a tip position, this module samples the
DIC's in-plane DISPLACEMENT field (u, v -- already loaded per
core/crack_wave_io.py, distinct from the strain fields used for tracing/
wave-speed/tip-detection) in an annulus around that tip, rotates it into
the crack's local (tangent, normal) frame, and fits it against the closed-
form Williams/Irwin leading-order mode I + II near-tip displacement field
(r, theta polar coordinates centered on the tip, theta measured from the
local crack direction -- i.e. theta = 0 is straight ahead of the tip,
theta = +/-pi is along the crack's two open faces):

    u' =  (K_I / 2*mu) * sqrt(r/2pi) * cos(t/2) * (kappa - 1 + 2*sin^2(t/2))
        + (K_II/ 2*mu) * sqrt(r/2pi) * sin(t/2) * (kappa + 1 + 2*cos^2(t/2))
        + Tx
    v' =  (K_I / 2*mu) * sqrt(r/2pi) * sin(t/2) * (kappa + 1 - 2*cos^2(t/2))
        - (K_II/ 2*mu) * sqrt(r/2pi) * cos(t/2) * (kappa - 1 - 2*sin^2(t/2))
        + Ty

kappa is the Kolosov constant (3 - 4*nu in plane strain, (3-nu)/(1+nu) in
plane stress) and mu = E / (2*(1+nu)) the shear modulus, both from the
page's Material & Calibration card. Tx, Ty are an unknown local rigid-body
offset, included as two more free fit parameters rather than assumed
zero -- a DIC displacement field is measured relative to the (arbitrary)
reference frame, not relative to the crack tip, so without them the fit
would silently absorb that offset into a biased K_I/K_II. This makes the
model LINEAR in its 4 unknowns (K_I, K_II, Tx, Ty), so a single ordinary
least-squares solve handles every frame -- no non-linear/iterative fit
needed -- and comes with a directly checkable R^2 per frame, the same
"trust the fit only as far as its R^2" convention used throughout this
tool (see core/elastic_wave_speed.py, core/crack_tip_tracking.py).

Sampling deliberately covers a full annulus around the tip (not just a
forward-facing cone, unlike core.crack_tip_tracking's "singular_fit" tip-
DETECTION method) because the crack-opening displacement across the two
crack faces (theta near +/-pi) is exactly where the fit is most sensitive
to K_I -- it is excluded only right at the faces themselves
(`flank_exclusion_deg`), where DIC correlation commonly fails on the real
discontinuity in the image.

This is "static first" per the user's explicit choice for this tool -- a
dynamic (Freund-type, crack-velocity-dependent kernel) fit is a possible
later addition, not built here.

K1c itself is not "whichever frame has the highest K_I" automatically --
the physically meaningful instant (onset of unstable growth) is a
judgement call this tool cannot make reliably on its own. So this module
computes the whole K_I(t) / K_II(t) history (`compute_k_time_series`) for
the user to inspect, and picks a default "critical frame" (the first frame
with a valid tip detection and a successful fit -- i.e. at crack
initiation) that the user can override in the UI.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from core.crack_tip_tracking import path_tangents, smoothed_tip_arc_length_and_positions


class K1cError(ValueError):
    """Raised when a K1c fit can't be computed as asked -- message is
    shown to the user as-is, so kept short and specific."""


# ---------------------------------------------------------------------------
# Material constants
# ---------------------------------------------------------------------------

def kolosov_kappa(poisson_ratio: float, plane_stress: bool) -> float:
    """The Kolosov constant kappa used in the Williams/Irwin near-tip
    displacement field: (3 - nu) / (1 + nu) in plane stress, 3 - 4*nu in
    plane strain."""
    if plane_stress:
        return (3.0 - poisson_ratio) / (1.0 + poisson_ratio)
    return 3.0 - 4.0 * poisson_ratio


def shear_modulus(young_modulus_pa: float, poisson_ratio: float) -> float:
    """mu = E / (2*(1+nu)), in the same units as `young_modulus_pa`."""
    return young_modulus_pa / (2.0 * (1.0 + poisson_ratio))


# ---------------------------------------------------------------------------
# flank_exclusion_deg pre-fill
# ---------------------------------------------------------------------------

def recommend_flank_exclusion_deg(path_positions: np.ndarray, path_arc_length: np.ndarray,
                                   tip_arc_length_px: float, r_max_px: float,
                                   base_deg: float = 15.0, min_deg: float = 8.0,
                                   max_deg: float = 40.0) -> dict:
    """A starting-point `flank_exclusion_deg` pre-fill, from how much the
    traced path's own RAW tangent disagrees with its `path_tangents`-
    smoothed tangent right where the fit's own annulus sits (arc length
    `tip_arc_length_px`, window `2*r_max_px` -- the SAME smoothing window
    `compute_k_time_series`'s own `tangent_window_px` defaults to for
    exactly this reason, see its docstring).

    Why this, specifically: `flank_exclusion_deg` exists because the
    Williams/Irwin fit's `theta=0` ("ahead of the tip") is defined
    relative to the traced path's own local tangent -- so the wider that
    tangent's own uncertainty, the wider the safety margin needed to keep
    the excluded wedge from actually eating into the "ahead" region (too
    narrow) or, in the opposite direction, from needlessly discarding
    good near-face samples the fit could have used (too wide, once the
    tangent is trustworthy). `path_tangents`'s own docstring already
    quantifies this for a real auto-traced skeleton: pixel-grid
    skeletonization zigzag alone -- even along an overall dead-straight
    crack -- routinely produces a RAW tangent error of "~16 deg median (up
    to 45 deg)" relative to a properly smoothed one, exactly the
    difference this function measures directly for the trace at hand
    rather than assuming that generic figure applies unchanged to every
    dataset.

    `base_deg` (15 degrees, matching this app's own prior, unexposed
    hardcoded default) is the floor -- a perfectly straight, zero-
    disagreement trace still gets that much exclusion, since some margin
    is always warranted for ordinary DIC decorrelation right at the crack
    faces (the flank exclusion's OTHER job, unrelated to tangent
    uncertainty at all -- see `fit_k_single_frame`'s module docstring).
    The raw-vs-smoothed disagreement is added on top of that floor and
    the whole thing is clamped to `[min_deg, max_deg]`, a generous but
    finite range (an exclusion approaching 90 degrees would leave almost
    no annulus to fit against at all).

    This is only ever a pre-fill of the existing, still-editable spin
    box -- never applied silently -- so, exactly like
    `core.xfem_dic.recommend_mesh_parameters`, a rough heuristic is an
    acceptable trade-off for saving the user a first guess. Returns
    `{"flank_exclusion_deg", "tangent_disagreement_deg"}` -- the second
    entry only for the status message to explain WHY a given value was
    suggested."""
    from core.crack_tip_tracking import path_tangents

    path_positions = np.asarray(path_positions, dtype=float)
    path_arc_length = np.asarray(path_arc_length, dtype=float)
    if len(path_positions) < 2 or not np.isfinite(tip_arc_length_px):
        return {"flank_exclusion_deg": base_deg, "tangent_disagreement_deg": 0.0}

    window_px = 2.0 * r_max_px if r_max_px and np.isfinite(r_max_px) and r_max_px > 0 else 0.0
    raw_tangents = path_tangents(path_positions, path_arc_length, window_px=0.0)
    smoothed_tangents = path_tangents(path_positions, path_arc_length, window_px=window_px)
    idx = int(np.argmin(np.abs(path_arc_length - tip_arc_length_px)))
    raw_t = raw_tangents[idx]
    smooth_t = smoothed_tangents[idx]

    dot = float(np.clip(np.dot(raw_t, smooth_t), -1.0, 1.0))
    disagreement_deg = float(np.degrees(np.arccos(dot)))
    suggested = float(np.clip(base_deg + disagreement_deg, min_deg, max_deg))
    return {"flank_exclusion_deg": round(suggested, 1),
            "tangent_disagreement_deg": round(disagreement_deg, 1)}


# ---------------------------------------------------------------------------
# Field sampling (mirrors core/crack_tip_tracking.py's `_sample_field` --
# kept as its own small copy rather than importing a private helper across
# modules, matching this tool's existing per-stage self-sufficiency, e.g.
# core/elastic_wave_speed.py's own `_resolve_times`)
# ---------------------------------------------------------------------------

def _sample_field(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    interpolator = RegularGridInterpolator((ys, xs), field, bounds_error=False, fill_value=np.nan)
    query = np.column_stack([points_xy[:, 1], points_xy[:, 0]])  # (y, x) order
    return interpolator(query)


def _sample_annulus(tip_xy, tangent_xy, r_min_px, r_max_px, flank_exclusion_deg, n_r, n_theta):
    """Points on a regular (r, theta) grid around `tip_xy`, in the local
    frame defined by `tangent_xy` (theta = 0, "ahead" of the tip) and its
    90-degree normal (theta = +pi/2) -- same forward-facing convention as
    core.crack_tip_tracking's singular-fit tip detector, but swept across
    the FULL angular range except a small window excluded right at the
    crack's two faces (theta near +/-pi). Returns (points_xy (N,2),
    r_px (N,), theta (N,))."""
    tx, ty = tangent_xy
    nx, ny = -ty, tx  # unit normal, 90 deg from the tangent
    radii = np.linspace(r_min_px, r_max_px, n_r)
    excl = np.radians(flank_exclusion_deg)
    thetas = np.linspace(-np.pi + excl, np.pi - excl, n_theta)

    pts, r_list, theta_list = [], [], []
    for r in radii:
        for theta in thetas:
            c, s = np.cos(theta), np.sin(theta)
            dx = r * (c * tx + s * nx)
            dy = r * (c * ty + s * ny)
            pts.append((tip_xy[0] + dx, tip_xy[1] + dy))
            r_list.append(r)
            theta_list.append(theta)
    return np.asarray(pts), np.asarray(r_list), np.asarray(theta_list)


# ---------------------------------------------------------------------------
# Single-frame fit
# ---------------------------------------------------------------------------

@dataclass
class KFitResult:
    K_I_pa_sqrt_m: float | None
    K_II_pa_sqrt_m: float | None
    r_squared: float | None
    n_points_used: int
    tx_m: float | None  # fitted local rigid-body offset (diagnostic only)
    ty_m: float | None
    K_I_stderr_pa_sqrt_m: float | None = None   # 1-sigma formal fit uncertainty -- see
    K_II_stderr_pa_sqrt_m: float | None = None  # `fit_k_single_frame`'s own docstring
    # T-stress (opt-in "extended Williams series" term, see `fit_k_single_frame`'s
    # `include_t_stress` argument) -- `None`/`None` whenever that option is off,
    # exactly like every other field above when the fit itself couldn't be run.
    t_stress_pa: float | None = None
    t_stress_stderr_pa: float | None = None


def fit_k_single_frame(u_field: np.ndarray, v_field: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                        tip_xy, tangent_xy, kappa: float, mu_pa: float, pixel_size_mm: float,
                        r_min_px: float, r_max_px: float, flank_exclusion_deg: float = 15.0,
                        n_radial_samples: int = 8, n_angular_samples: int = 16,
                        include_t_stress: bool = False, include_affine: bool = False) -> KFitResult:
    """Fits K_I, K_II (plus a local rigid-body offset Tx, Ty) to the DIC
    displacement field `u_field`/`v_field` around `tip_xy`, by ordinary
    least squares against the linear-in-(K_I, K_II, Tx, Ty) Williams/Irwin
    model (see module docstring). Returns a `KFitResult` with every K_*/
    r_squared/tx/ty field `None` (n_points_used still reported) if fewer
    than 6 valid (non-NaN in both u and v) samples were found -- not
    enough to safely resolve 4 unknowns (5 once `include_t_stress` adds a
    5th, see below).

    `K_I_stderr_pa_sqrt_m`/`K_II_stderr_pa_sqrt_m` are the formal 1-sigma
    parameter uncertainties from this same ordinary-least-squares fit --
    since the model is LINEAR in (K_I, K_II, Tx, Ty), the classical OLS
    result applies directly and exactly (no linearization/approximation
    needed, unlike a nonlinear fit): with residual variance
    `sigma^2 = ss_res / (n_equations - n_params)` and design matrix `X`,
    the parameter covariance is `sigma^2 * inv(X^T X)`, and each
    parameter's own stderr is the square root of its diagonal entry. This
    is ALWAYS computed (no opt-in flag, unlike X-FEM-DIC's own
    `estimate_uncertainty` -- see core/xfem_dic.py's `fit_xfem_dic_frame`)
    because it costs almost nothing on top of the `lstsq` solve already
    being done: inverting a 4x4 (or 5x5) matrix, not re-fitting anything.
    Both are `None` whenever there isn't at least one more equation than
    parameter (`2*n_valid <= n_params`, below the existing `n_valid < 6`
    floor already returning early above, so this only matters if that
    floor is ever lowered) or the normal matrix `X^T X` is singular (a
    degenerate sample geometry -- caught via `np.linalg.pinv` returning a
    very large or non-finite value, in which case `None` is reported
    rather than a misleadingly huge number).

    `include_t_stress` (opt-in, default off -- an ALTERNATIVE/ADDITIONAL
    method, not a replacement for the 4-parameter fit above, since the
    user may want to cross-check one against the other) extends the
    model with a 5th free parameter, the T-stress (the leading NON-
    singular, constant-stress term of the full Williams eigenfunction
    series -- see Williams 1957): a uniform stress T parallel to the
    crack faces, superposed on the K_I/K_II singular field. Its own
    displacement contribution, derived by hand from the general Williams
    series (n=2 term of the series both K_I and K_II are themselves the
    n=1 term of) and cross-checked against the classical far-field result
    for a plate under uniform uniaxial stress T (u = (T/E)*x,
    v = -(nu*T/E)*y in plane stress -- the T-stress term is, physically,
    exactly this SAME uniform-stress field, just centered on the crack tip
    instead of applied at infinity):

        u_T = T * (kappa + 1) / (8*mu) * x'
        v_T = T * (kappa - 3) / (8*mu) * y'

    where x' = r*cos(theta), y' = r*sin(theta) are the local (tangent,
    normal) offsets from the tip -- i.e. this fit actually solves for the
    coefficient `A2 = T/4` (so the design matrix stays perfectly linear
    and dimensionally uniform with the K_I/K_II columns, all three
    representing an amplitude in front of an `r^(n/2)` Williams term with
    n=1 for K_I/K_II and n=2 for T-stress) and converts back to physical
    T-stress (`T = 4*A2`) only in the returned `KFitResult.t_stress_pa`.
    (No extra minus sign is needed in `v_T` above -- `kappa - 3` is
    already negative for any physical Poisson's ratio, e.g. in plane
    stress `(kappa-3)/(2*mu) = -4*nu/E`, which combined with `T = 4*A2`
    reproduces the classical `v = -(nu*T/E)*y` exactly, sign included.)
    Unlike K_I/K_II, T-stress has NO sign convention to reconcile against
    an outside source: it was derived directly from THIS module's own
    K_I/K_II displacement formulas (same Williams series, same theta
    convention), so it is automatically self-consistent with them.

    Why bother: T-stress is the standard "beyond K_I/K_II" fracture
    parameter reported alongside them in the mixed-mode DIC/FEM literature
    (it does not itself drive crack growth in the singular sense K_I/K_II
    do, but it is known to affect the SIZE of the near-tip plastic/process
    zone and the crack path's directional stability -- see e.g. Williams
    1957 and the T-stress literature it started). Including it here also
    absorbs into an explicit, reported parameter whatever small non-
    singular constant-stress component genuinely exists in the sampled
    field, rather than silently letting the 4-parameter fit's K_I/K_II
    absorb a biased sliver of it (a T-stress term and a K_I term are NOT
    orthogonal on a finite, one-sided annular sample the way they are in
    the infinite-domain limit the eigenfunction expansion is built for)."""
    if r_max_px <= r_min_px:
        raise K1cError("The K1c search radius's max must be larger than its min.")
    if not pixel_size_mm or pixel_size_mm <= 0:
        raise K1cError("A pixel size (mm/px) is required to compute K1c in physical units.")
    if include_affine and include_t_stress:
        raise K1cError("include_affine already contains the T-stress term: use one or the other.")

    pts, r_px, theta = _sample_annulus(
        tip_xy, tangent_xy, r_min_px, r_max_px, flank_exclusion_deg, n_radial_samples, n_angular_samples,
    )
    u_vals = _sample_field(u_field, xs, ys, pts)
    v_vals = _sample_field(v_field, xs, ys, pts)
    valid = np.isfinite(u_vals) & np.isfinite(v_vals)
    n_valid = int(np.count_nonzero(valid))
    n_params = 8 if include_affine else (5 if include_t_stress else 4)
    if 2 * n_valid < n_params + 2:  # keep the existing safety margin (>= 6 pts for 4 params)
        return KFitResult(None, None, None, n_valid, None, None)

    m_per_px = pixel_size_mm * 1e-3
    r_m = r_px[valid] * m_per_px
    theta_v = theta[valid]
    tx, ty = tangent_xy
    # Rotate the globally-measured (image x, y) displacement components into
    # the crack-local (tangent, normal) frame the Williams/Irwin model below
    # (u', v') is expressed in -- see the module docstring, which already
    # claims this rotation happens. u' = u.t, v' = u.n where t=(tx,ty) is the
    # tangent and n=(-ty,tx) its normal, i.e. u' = u*tx + v*ty,
    # v' = -u*ty + v*tx. Skipping this rotation would only match the model
    # when tangent_xy is exactly (1, 0) -- a crack path running exactly
    # horizontal in the image. Any other tangent orientation mixes true
    # K_I and K_II together in the fit result, up to and including a
    # spuriously negative K_I on a crack that is genuinely, physically
    # opening.
    u_m = (u_vals[valid] * tx + v_vals[valid] * ty) * m_per_px
    v_m = (-u_vals[valid] * ty + v_vals[valid] * tx) * m_per_px

    sqrt_term = np.sqrt(r_m / (2.0 * np.pi))
    c2 = np.cos(theta_v / 2.0)
    s2 = np.sin(theta_v / 2.0)
    fI_u = sqrt_term * c2 * (kappa - 1.0 + 2.0 * s2 ** 2) / (2.0 * mu_pa)
    fII_u = sqrt_term * s2 * (kappa + 1.0 + 2.0 * c2 ** 2) / (2.0 * mu_pa)
    fI_v = sqrt_term * s2 * (kappa + 1.0 - 2.0 * c2 ** 2) / (2.0 * mu_pa)
    fII_v = -sqrt_term * c2 * (kappa - 1.0 - 2.0 * s2 ** 2) / (2.0 * mu_pa)

    n = n_valid
    design = np.zeros((2 * n, n_params))
    target = np.zeros(2 * n)
    design[:n, 0] = fI_u
    design[:n, 1] = fII_u
    target[:n] = u_m
    design[n:, 0] = fI_v
    design[n:, 1] = fII_v
    target[n:] = v_m
    if include_t_stress:
        x_local_m = r_m * np.cos(theta_v)
        y_local_m = r_m * np.sin(theta_v)
        fT_u = (kappa + 1.0) / (2.0 * mu_pa) * x_local_m
        fT_v = (kappa - 3.0) / (2.0 * mu_pa) * y_local_m
        design[:n, 2] = fT_u
        design[n:, 2] = fT_v
        design[:n, 3] = 1.0
        design[n:, 4] = 1.0
        tx_col, ty_col = 3, 4
    else:
        design[:n, 2] = 1.0
        design[n:, 3] = 1.0
        tx_col, ty_col = 2, 3
        if include_affine:
            # Full uniform displacement GRADIENT around the tip (the r^1 terms of the
            # Williams series generalised to any smooth background): u' = a_xx x' + a_xy y',
            # v' = a_yx x' + a_yy y' -- uniform strains AND rigid rotation. At the radii of a
            # DIC fit the K field of a modest K is only ~0.01-0.05 px, the same order as a
            # background strain of 1e-3 acting over 1 mm, so leaving these out lets them leak
            # into K_I/K_II (a uniform opening strain has exactly the symmetry of K_I).
            x_local_m = r_m * np.cos(theta_v)
            y_local_m = r_m * np.sin(theta_v)
            design[:n, 4] = x_local_m
            design[:n, 5] = y_local_m
            design[n:, 6] = x_local_m
            design[n:, 7] = y_local_m

    # Column (Jacobi) scaling before the MAIN solve too -- not just the
    # covariance below. `np.linalg.lstsq`'s own SVD-based solver applies
    # an internal `rcond` cutoff exactly like `pinv` does (see the long
    # comment below): with `include_t_stress`, the T-stress column
    # (`fT_u`/`fT_v`, an `r^1` near-tip basis) is roughly one to two
    # orders of magnitude smaller than the K_I/K_II columns (an `r^(1/2)`
    # basis) over the SAME annulus, on top of the ~13-order-of-magnitude
    # gap already existing against the Tx/Ty ("1.0") columns -- which
    # would otherwise silently truncate the T-stress DIRECTION to
    # (numerically) zero, well before that direction's own genuine
    # residual/noise floor would justify it, rather than raising an error
    # -- a subtly wrong T-stress with no warning sign. Scaling first (then
    # un-scaling the solution) removes
    # the disparity `rcond` would otherwise misjudge, exactly as it does
    # for the covariance's own `pinv` call.
    col_scale_main = np.sqrt(np.sum(design ** 2, axis=0))
    col_scale_main[col_scale_main < 1e-300] = 1.0
    solution_scaled, _residuals, _rank, _sv = np.linalg.lstsq(
        design / col_scale_main, target, rcond=None,
    )
    solution = solution_scaled / col_scale_main
    K_I, K_II = float(solution[0]), float(solution[1])
    A2 = float(solution[2]) if include_t_stress else None
    Tx, Ty = float(solution[tx_col]), float(solution[ty_col])
    predicted = design @ solution
    ss_res = float(np.sum((target - predicted) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    dof = 2 * n - n_params
    K_I_stderr = K_II_stderr = T_stderr = None
    if dof > 0:
        sigma2 = ss_res / dof
        # Column (Jacobi) scaling before inverting X^T X -- WITHOUT it,
        # `pinv` gets this design matrix badly wrong: K_I/K_II's own
        # columns (`fI_u`/`fI_v`/`fII_u`/`fII_v`) are ~1e-13 in magnitude
        # (near-tip Williams basis values, in meters-per-(Pa*sqrt(m))),
        # while Tx/Ty's columns are exactly 1.0 -- a ~1e13 disparity that
        # makes X^T X's own singular values span ~26 orders of magnitude.
        # `pinv`'s default `rcond` cutoff is relative to the LARGEST
        # singular value, so it silently treats the tiny-but-real K_I/K_II
        # directions as numerically zero and drops them, which doesn't
        # blow their variance up (as an ill-conditioned/near-singular fit
        # normally would) but instead truncates it to near-nothing -- a
        # silently WRONG (falsely tiny) stderr if this scaling is skipped.
        # Scaling every column to unit L2 norm first (same Jacobi
        # preconditioning trick core/xfem_dic.py's own Gauss-Newton solve
        # already uses, see `fit_xfem_dic_frame`'s `col_scale`) equalizes
        # every parameter's column magnitude before `pinv` ever sees the
        # matrix, so its rcond cutoff no longer discriminates between
        # them; the true covariance is then recovered by un-scaling
        # (dividing by each column's own scale squared). The T-stress
        # column (when present) is on yet another magnitude again -- same
        # treatment, no special-casing needed. Reuses `col_scale_main`
        # (the exact same per-column scale already applied to the MAIN
        # solve above) rather than recomputing it.
        design_scaled = design / col_scale_main
        cov_scaled = sigma2 * np.linalg.pinv(design_scaled.T @ design_scaled)
        cov = cov_scaled / np.outer(col_scale_main, col_scale_main)
        var_K_I, var_K_II = cov[0, 0], cov[1, 1]
        if np.isfinite(var_K_I) and var_K_I >= 0:
            K_I_stderr = float(np.sqrt(var_K_I))
        if np.isfinite(var_K_II) and var_K_II >= 0:
            K_II_stderr = float(np.sqrt(var_K_II))
        if include_t_stress:
            var_A2 = cov[2, 2]
            if np.isfinite(var_A2) and var_A2 >= 0:
                T_stderr = float(4.0 * np.sqrt(var_A2))  # T = 4*A2, stderr scales the same way

    t_stress_pa = 4.0 * A2 if A2 is not None else None
    return KFitResult(K_I_pa_sqrt_m=K_I, K_II_pa_sqrt_m=K_II, r_squared=float(r_squared),
                       n_points_used=n_valid, tx_m=Tx, ty_m=Ty,
                       K_I_stderr_pa_sqrt_m=K_I_stderr, K_II_stderr_pa_sqrt_m=K_II_stderr,
                       t_stress_pa=t_stress_pa, t_stress_stderr_pa=T_stderr)


# ---------------------------------------------------------------------------
# Time series over a Stage 3 tip track
# ---------------------------------------------------------------------------

@dataclass
class KFrameSample:
    frame_order_index: int   # index into tip_result.times / arc_length_s / positions_xy
    source_frame_index: int  # index into dataset.frames -- the exact frame this sample's u/v came from
    time_s: float
    tip_xy_px: np.ndarray
    fit: KFitResult


@dataclass
class K1cTimeSeries:
    leaf_index: int
    tip_method: str  # the Stage 3 method ("threshold" | "singular_fit") that produced the tip track
    young_modulus_pa: float
    poisson_ratio: float
    plane_stress: bool
    pixel_size_mm: float
    r_min_px: float
    r_max_px: float
    samples: list  # list[KFrameSample], one per Stage 3 frame with a detected tip
    default_critical_index: int | None  # index into `samples`, or None if nothing fit successfully

    def sample_at(self, index: int) -> KFrameSample | None:
        if 0 <= index < len(self.samples):
            return self.samples[index]
        return None


def compute_k_time_series(dataset, tip_result, young_modulus_pa: float, poisson_ratio: float,
                           plane_stress: bool, pixel_size_mm: float, r_min_px: float, r_max_px: float,
                           flank_exclusion_deg: float = 15.0, n_radial_samples: int = 8,
                           n_angular_samples: int = 16, min_r_squared_for_default: float = 0.5,
                           tangent_window_px: float | None = None,
                           position_window_px: float | None = None) -> K1cTimeSeries:
    """Fits K_I(t)/K_II(t) at every frame of `tip_result` (a
    core.crack_tip_tracking.TipTrackResult -- Stage 3's output, so "Track
    tip" must be run first) that has a detected tip position, reusing that
    frame's exact displacement field (via `tip_result.
    source_frame_positions`) and the traced path's local tangent at that
    tip position (via `core.crack_tip_tracking.path_tangents` on
    `tip_result.path_positions`).

    `min_r_squared_for_default` only affects which frame is offered as the
    DEFAULT critical frame (the first one with a fit quality at or above
    it) -- every frame's fit is still computed and returned regardless, so
    the user can inspect/override in the UI.

    `tangent_window_px` controls how the local crack-direction (tangent)
    used to orient each frame's near-tip fit is derived from the traced
    path (see `path_tangents`): a REAL auto-traced skeleton is only
    straight at a scale of many pixels, and a per-point tangent taken
    literally between two path samples `resample_step_px` apart mistakes
    pixel-grid skeletonization zigzag for real crack curvature -- even a
    handful of degrees of this is enough to visibly corrupt the K_I/K_II
    split, well before it shows up as "obviously wrong" tip positions.
    Defaulting to `None` uses `2.0 * r_max_px` -- the tangent only needs
    to be accurate over the same near-tip region the fit itself samples,
    so tying it to the fit's own outer radius is the natural scale; pass
    `0.0` to fall back to the raw per-point tangent instead.

    `position_window_px` controls whether each frame's near-tip fit is
    centered on the RAW per-frame tracked tip position (`None`/`0`, the
    default) or on a smoothed position (a positive window, in the same
    arc-length-px units as `tangent_window_px`) via
    `core.crack_tip_tracking.smoothed_tip_arc_length_and_positions`. This
    is the origin-side counterpart to `tangent_window_px`: the tangent
    direction is smoothed independently of the raw track (see above), and
    this option lets the fit's polar ORIGIN be smoothed the same way --
    because the near-tip basis blows up as 1/sqrt(r) close to the origin,
    position noise dominates K_I/K_II fluctuation far more than it
    dominates the tip track itself (this matters most for a dynamic,
    freely propagating crack tip, where frame-to-frame detection noise on
    a fast-moving tip is unavoidable and expected). When set, the SAME
    smoothed arc length also selects which point of `tip_result.
    path_arc_length`/`tangents` orients that frame's fit, so the origin
    and orientation stay mutually consistent; every OTHER reported/
    plotted quantity (`tip_result.arc_length_s`, distances, CSV columns)
    is left exactly as Stage 3 tracked it -- only this fit's internal
    (r, theta) origin changes."""
    if not np.isfinite(tip_result.arc_length_s).any():
        raise K1cError("This tip track has no detected tip positions -- re-run \"Track tip\" first.")

    kappa = kolosov_kappa(poisson_ratio, plane_stress)
    mu = shear_modulus(young_modulus_pa, poisson_ratio)

    # `path_tangents` points toward increasing arc length by construction
    # -- but which end of the traced path arc length is measured FROM
    # (node 0, see core/crack_tip_tracking.py's `resample_path`) can be
    # either the crack's true initiation point or its current tip,
    # depending on which end happened to be traced/placed first: a crack
    # traced from its tip back toward its origin has arc length running
    # backward relative to its actual growth. The Williams/Irwin model
    # this fit uses is NOT invariant to a 180-degree tangent flip (theta
    # is measured from "ahead of the tip", and the crack's two open faces
    # sit at theta = +/-pi -- swap the tangent's sign and the fit reads
    # the crack faces as "ahead" and the actual growth direction as a
    # "face", corrupting the K_I/K_II split, not merely flipping a sign).
    # So the tangent direction is oriented from the SAME data this tip
    # track's own arc-length-vs-time trend already carries, exactly the
    # way `extract_tip_ridge_in_polygon` decides its own monotonic-advance
    # direction: a decreasing trend (arc length going down as the tip
    # physically advances, i.e. a reverse-traced path) flips every
    # tangent, so "ahead of the tip" always matches physical growth
    # regardless of which end was traced first.
    tangent_sign = 1.0
    valid_arc = np.isfinite(tip_result.arc_length_s)
    if np.count_nonzero(valid_arc) >= 2:
        t_valid = tip_result.times[valid_arc]
        s_valid = tip_result.arc_length_s[valid_arc]
        if not np.allclose(t_valid, t_valid[0]):
            slope = float(np.polyfit(t_valid, s_valid, 1)[0])
            if slope < 0:
                tangent_sign = -1.0
    window_px = tangent_window_px if tangent_window_px is not None else 2.0 * r_max_px
    tangents = path_tangents(tip_result.path_positions, tip_result.path_arc_length, window_px=window_px) * tangent_sign

    # Position smoothing (opt-in, off by default -- see `position_window_px`
    # in the docstring above): a smoothed arc-length/position pair, used as
    # BOTH the fit's origin and the tangent lookup below, so the two stay
    # mutually consistent. `arc_length_for_fit`/`positions_for_fit` default
    # right back to the raw Stage-3 track when smoothing is off, so the
    # rest of this loop doesn't need an `if` per frame.
    if position_window_px:
        arc_length_for_fit, positions_for_fit = smoothed_tip_arc_length_and_positions(
            tip_result.times, tip_result.arc_length_s, tip_result.path_positions,
            tip_result.path_arc_length, position_window_px,
        )
    else:
        arc_length_for_fit = tip_result.arc_length_s
        positions_for_fit = tip_result.positions_xy

    samples = []
    for i in range(len(tip_result.times)):
        if not np.isfinite(tip_result.arc_length_s[i]):
            continue
        tip_xy = positions_for_fit[i]
        path_idx = int(np.argmin(np.abs(tip_result.path_arc_length - arc_length_for_fit[i])))
        tangent = tangents[path_idx]
        source_idx = int(tip_result.source_frame_positions[i])
        frame = dataset.frames[source_idx]

        fit = fit_k_single_frame(
            frame.u, frame.v, dataset.xs, dataset.ys, tip_xy, tangent, kappa, mu, pixel_size_mm,
            r_min_px, r_max_px, flank_exclusion_deg, n_radial_samples, n_angular_samples,
        )
        samples.append(KFrameSample(
            frame_order_index=i, source_frame_index=source_idx, time_s=float(tip_result.times[i]),
            tip_xy_px=tip_xy, fit=fit,
        ))

    default_index = None
    for i, sample in enumerate(samples):
        if sample.fit.K_I_pa_sqrt_m is not None and (
            sample.fit.r_squared is None or sample.fit.r_squared >= min_r_squared_for_default
        ):
            default_index = i
            break
    if default_index is None:
        for i, sample in enumerate(samples):
            if sample.fit.K_I_pa_sqrt_m is not None:
                default_index = i
                break

    return K1cTimeSeries(
        leaf_index=tip_result.leaf_index, tip_method=tip_result.method,
        young_modulus_pa=young_modulus_pa, poisson_ratio=poisson_ratio, plane_stress=plane_stress,
        pixel_size_mm=pixel_size_mm, r_min_px=r_min_px, r_max_px=r_max_px,
        samples=samples, default_critical_index=default_index,
    )


def smooth_k_series(values: np.ndarray, window: int) -> np.ndarray:
    """Robust moving-MEDIAN smoothing of an already-computed K_I(t) or
    K_II(t) sample sequence (or any other per-frame series the same
    shape), for DISPLAY/EXPORT only -- it never feeds back into the per-
    frame fit itself. For actually reducing K_I/K_II fluctuation at its
    source (the near-tip fit's polar origin), see `compute_k_time_series`'s
    `position_window_px` instead; this function is a purely cosmetic,
    additive overlay on top of whatever that produced -- the raw values
    this smooths are never modified or discarded by it.

    A median (rather than a mean) is used specifically because it rejects
    the occasional single-frame outlier fit (a bad DIC correlation, a
    spurious detection) without that one bad frame dragging its whole
    window's average down with it -- exactly the failure mode a mean-
    based moving average would keep.

    `window` is the total window width in SAMPLES (i.e. entries of
    `K1cTimeSeries.samples`, in that list's own order -- irregular or
    skipped frames are fine, this only cares about sample order, not real
    time spacing), clamped to an odd width >= 1 so every window is
    centered on the frame it's smoothing. NaN entries (a fit that
    returned `None` for `K_I_pa_sqrt_m`/`K_II_pa_sqrt_m`, e.g. too few
    valid displacement samples that frame -- see the caller's own NaN
    conversion) are skipped when building each window rather than
    propagating, so one missing frame doesn't blank out its neighbors'
    smoothed values; a window with zero finite samples in range leaves
    that output entry NaN too.

    Returns an array the same length as `values`."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    half = w // 2

    out = np.full(n, np.nan)
    for i in range(n):
        window_vals = values[max(0, i - half):min(n, i + half + 1)]
        finite = window_vals[np.isfinite(window_vals)]
        if len(finite) > 0:
            out[i] = float(np.median(finite))
    return out


def positive_value_mean_and_domain_bounds(x: np.ndarray, y: np.ndarray):
    """For a per-frame series `y(x)` (e.g. K_I(distance) -- but generic,
    also reused as-is for core/xfem_dic.py's own K_I(distance)), returns
    `(mean_of_positive_y | None, domain_bounds)` where `domain_bounds` is
    a list of `(x_start, x_end)` pairs, one per maximal contiguous run (in
    `x` order) of samples with `y > 0`.

    This is the basis of K1c's (and X-FEM-DIC's) "critical K_I" value:
    K_I <= 0 has no physical meaning (a closed/compressed crack tip has no
    mode-I stress intensity -- unlike K_II, whose SIGN is physically
    meaningful, the shear sense, so K_II is never filtered this way
    anywhere in this app), so the reported "critical" K_I -- and the
    plot's domain-boundary markers -- reflect only the positive-K_I
    portion of a trace, run by run, since a real (jagged, noisy) trace
    commonly dips below zero more than once, not just at its very start or
    end.

    NaN/non-finite entries in either `x` or `y` are dropped first (never
    treated as a domain boundary in their own right). Samples are sorted
    by `x` before scanning for runs, so "contiguous" means contiguous IN
    DISTANCE -- every caller in this app already hands these in that
    order, but sorting here is cheap insurance against one that someday
    isn't. Returns `(None, [])` if no sample has y > 0."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return None, []
    order = np.argsort(x[valid])
    x_sorted = x[valid][order]
    y_sorted = y[valid][order]
    positive = y_sorted > 0.0
    if not np.any(positive):
        return None, []
    mean_val = float(np.mean(y_sorted[positive]))

    bounds = []
    n_pts = len(x_sorted)
    i = 0
    while i < n_pts:
        if not positive[i]:
            i += 1
            continue
        j = i
        while j < n_pts and positive[j]:
            j += 1
        bounds.append((float(x_sorted[i]), float(x_sorted[j - 1])))
        i = j
    return mean_val, bounds


def export_k1c_csv(entries: list, path: str, pixel_size_mm: float | None = None,
                    smoothing_window: int | None = None) -> None:
    """Writes every given K1c result's full K_I(t)/K_II(t) history to one
    CSV, long format (one row per frame sample, a `trace_label` column
    tells traces apart) -- mirrors core/crack_tip_tracking.py's
    `export_tip_roi_csv`/`export_roi_waves_csv`. `entries` is the page's
    own `{"label", "tip_result", "series"}` dict list, exactly as built by
    the K1c card's "Compute K1c" (see `page_crack_wave_analysis.py`'s
    `_on_compute_k1c`) and then filtered down to whichever the user leaves
    checked in the results list -- passed straight through rather than
    re-fetched, so this never needs `dataset` again. Raises `OSError` on a
    write failure, left for the caller to report as-is.

    `smoothing_window`, if given (in samples -- see `smooth_k_series`),
    appends two more columns, `K_I_Pa_sqrt_m_smoothed`/
    `K_II_Pa_sqrt_m_smoothed`: a robust (median) `smooth_k_series` of each
    trace's own already-computed K_I(t)/K_II(t) columns above. This is
    purely ADDITIVE -- the raw per-frame fit columns before it are always
    written unchanged -- matching the plot's optional dotted smoothed
    overlay, which never replaces its raw markers either.

    The final column, `K_I_critical_mean_positive_Pa_sqrt_m`, is this
    trace's own "critical" K_I -- the mean of every POSITIVE K_I sample in
    the trace (see `positive_value_mean_and_domain_bounds`; negative K_I
    has no physical meaning, unlike K_II, whose sign is meaningful and so
    is never filtered this way) -- repeated identically on every row of
    that trace, for easy per-trace filtering/pivoting in a spreadsheet.
    Blank when the trace has no positive K_I sample at all. K_II's own
    "critical" value is still `is_default_critical`'s single flagged row,
    unaffected by this column."""
    import csv

    header = ["trace_label", "leaf_index", "tip_method", "frame_order_index", "source_frame_index",
              "time_s", "arc_length_px"]
    if pixel_size_mm:
        header.append("arc_length_mm")
    header += ["K_I_Pa_sqrt_m", "K_II_Pa_sqrt_m", "r_squared", "n_points_used", "is_default_critical"]
    if smoothing_window:
        header += ["K_I_Pa_sqrt_m_smoothed", "K_II_Pa_sqrt_m_smoothed"]
    header += ["K_I_critical_mean_positive_Pa_sqrt_m"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for entry in entries:
            label = entry["label"]
            tip_result = entry["tip_result"]
            series = entry["series"]
            default_index = series.default_critical_index if series.default_critical_index is not None else -1
            k1_smoothed = k2_smoothed = None
            if smoothing_window:
                k1_raw = np.array([
                    s.fit.K_I_pa_sqrt_m if s.fit.K_I_pa_sqrt_m is not None else np.nan
                    for s in series.samples
                ])
                k2_raw = np.array([
                    s.fit.K_II_pa_sqrt_m if s.fit.K_II_pa_sqrt_m is not None else np.nan
                    for s in series.samples
                ])
                k1_smoothed = smooth_k_series(k1_raw, smoothing_window)
                k2_smoothed = smooth_k_series(k2_raw, smoothing_window)
            distances_all = np.array([
                tip_result.arc_length_s[s.frame_order_index] for s in series.samples
            ])
            k1_all = np.array([
                s.fit.K_I_pa_sqrt_m if s.fit.K_I_pa_sqrt_m is not None else np.nan
                for s in series.samples
            ])
            mean_positive_ki, _bounds = positive_value_mean_and_domain_bounds(distances_all, k1_all)
            mean_positive_ki_str = f"{mean_positive_ki:.6g}" if mean_positive_ki is not None else ""
            for i, sample in enumerate(series.samples):
                arc = tip_result.arc_length_s[sample.frame_order_index]
                arc_finite = np.isfinite(arc)
                row = [label, series.leaf_index, series.tip_method, sample.frame_order_index,
                       sample.source_frame_index, f"{sample.time_s:.9g}",
                       f"{arc:.6f}" if arc_finite else ""]
                if pixel_size_mm:
                    row.append(f"{arc * pixel_size_mm:.6f}" if arc_finite else "")
                fit = sample.fit
                row += [
                    f"{fit.K_I_pa_sqrt_m:.6g}" if fit.K_I_pa_sqrt_m is not None else "",
                    f"{fit.K_II_pa_sqrt_m:.6g}" if fit.K_II_pa_sqrt_m is not None else "",
                    f"{fit.r_squared:.6f}" if fit.r_squared is not None else "",
                    fit.n_points_used,
                    "1" if i == default_index else "0",
                ]
                if smoothing_window:
                    row += [
                        f"{k1_smoothed[i]:.6g}" if np.isfinite(k1_smoothed[i]) else "",
                        f"{k2_smoothed[i]:.6g}" if np.isfinite(k2_smoothed[i]) else "",
                    ]
                row.append(mean_positive_ki_str)
                writer.writerow(row)
