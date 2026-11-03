"""
Interaction-integral (domain form, a.k.a. "M-integral") K_I/K_II extraction
-- a SECOND, INDEPENDENT alternative to core/crack_k1c.py's near-tip
Williams/Irwin ring fit, added specifically so the two can cross-check each
other (the user's own explicit request: "il est important pour moi
d'obtenir des resultats robustes verifiables par plusieurs methodes").
Like core/crack_k1c.py, this operates on the already-exported standard
subset-DIC displacement field (u, v) -- not X-FEM-DIC's own mesh -- so it
is directly comparable to that ring fit, sample point for sample point,
the same tip position/tangent/radial window.

Method, in one paragraph: the ring fit above treats K_I/K_II as amplitudes
in a POINTWISE least-squares fit of the raw near-tip DISPLACEMENT field.
This module instead uses an ENERGY-based (domain-integral) definition:
the classical J-integral decomposes, for a superposition of the ACTUAL
field (1) and a chosen AUXILIARY field (2) of known K_I/K_II, into
J(1) + J(2) + M(1,2), where M(1,2) -- the "interaction integral" -- is
LINEAR in each field's own K_I/K_II (unlike J itself, which is quadratic).
Choosing the auxiliary field to be pure mode I (K_I=1, K_II=0) isolates
K_I^(1) from a single evaluation of M; pure mode II isolates K_II^(1). This
is the standard method introduced by Yau, Wang & Corten (1980) and given
its now-standard domain (area-integral) form by Moran & Shih (1987) --
used throughout the FEM fracture-mechanics literature (ABAQUS's own
"contour integral" K-extraction, MOOSE, FRANC3D, etc.) and adapted to DIC
displacement fields by Rethore, Gravouil, Morestin & Combescure (2005,
Int. J. Fracture) for exactly this kind of measured, noisy, gridded field.

Why implement a second, structurally DIFFERENT method rather than a second
POINTWISE fit: the two methods' error sources barely overlap. The ring fit
is a pointwise comparison against the raw (possibly noisy/decorrelated)
displacement AT a handful of sample points; a wrong ring fit and a wrong
interaction integral computed from the SAME noisy field are not
independent measurements of "the same mistake" in the sense of repeating
an identical calculation, but they ARE sensitive to different aspects of
that field (pointwise value vs. its area-integrated spatial derivative) --
so mutual agreement between the two is genuine, if partial, corroboration,
and a persistent large disagreement is a meaningful warning sign, exactly
the kind of cross-check the user asked for.

Practical adaptations made for a REGULAR-GRID DIC FIELD (flagged here
explicitly, as this tool's own documentation convention requires, rather
than silently departing from the textbook FEM recipe):

  - The textbook method integrates over a smooth domain around the tip
    with a smooth "plateau" weight function q (q=1 on an inner contour
    close to the tip, ramping down to q=0 on an outer contour). Here, for
    architectural consistency with core/crack_k1c.py's own r_min_px/
    r_max_px annulus (the SAME two numbers, so the two methods are
    directly comparable, not fit over unrelated domains), q is instead a
    single CONICAL ramp across the WHOLE annulus: q=1 at r=r_min_px,
    q=0 at r=r_max_px, linear in between. This is simpler than a plateau
    (no separate "how wide is the plateau" choice to make) and still
    satisfies the one property the derivation actually needs (q=1 at the
    inner edge, q=0 at the outer edge, smooth in between) -- but it does
    mean the WHOLE domain contributes to dq/dr, not just an outer shell,
    which is a real (if standard-conforming) simplification worth naming.
  - The actual field's spatial derivatives (needed for its strain/stress,
    not just its raw displacement) are estimated by finite difference
    (`np.gradient`) on the DIC's own regular (xs, ys) pixel grid, then
    interpolated onto the (r, theta) sample grid -- inherently noisier
    than a FEM solution's own shape-function-exact strain field, since
    differentiating already-noisy DIC data amplifies its noise. This is
    the same fundamental trade-off `core/crack_k1c.py`'s own module
    docstring already names for the ring fit ("inheriting whatever noise/
    decorrelation the standard DIC's own correlation already has"), just
    showing up here as derivative noise instead of value noise.
  - The measured field components (u, v) are first rotated into the
    crack's local (tangent, normal) frame -- `u1 = u*tx + v*ty`,
    `u2 = -u*ty + v*tx` -- and it is THESE rotated components that get
    differentiated (via `_local_gradient`) to build the local
    displacement-gradient tensor `du1/dx1, du1/dx2, du2/dx1, du2/dx2` the
    interaction integral needs. Both the SAMPLE geometry (already built in
    the local frame, via the identical `_sample_annulus`-style grid the
    ring fit itself uses) and the field components themselves must rotate
    together -- differentiation direction alone is not enough, since u/v
    are vector components that transform under rotation just like
    position does. `core.crack_k1c.fit_k_single_frame` and
    `core.xfem_dic.fit_xfem_dic_frame` apply the same component rotation
    for the same reason: skipping it would silently mix true K_I/K_II
    together for any tangent not exactly horizontal in the image (see
    each module's own comment at its rotation site).

References: J.R. Rice (1968) for J itself; J.F. Yau, S.S. Wang & H.T.
Corten, "A mixed-mode crack analysis of isotropic solids using conservation
laws of elasticity", J. Appl. Mech. 47 (1980), for the interaction-
integral decomposition; B. Moran & C.F. Shih, "Crack tip and associated
domain integrals from momentum and energy balance", Eng. Fract. Mech. 27
(1987), for its now-standard domain (area) form; J. Rethore, A. Gravouil,
F. Morestin & A. Combescure, "Estimation of mixed-mode stress intensity
factors using digital image correlation and an interaction integral",
Int. J. Fracture 132 (2005), for its adaptation to a DIC displacement
field specifically (the direct precedent for this module).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class InteractionIntegralError(ValueError):
    """Raised when an interaction-integral fit can't be computed as asked
    -- message shown to the user as-is, so kept short and specific."""


# ---------------------------------------------------------------------------
# Field sampling (own small copy, same reasoning as core/crack_k1c.py's own
# `_sample_field` -- see that module's comment on this convention)
# ---------------------------------------------------------------------------

def _sample_field(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    interpolator = RegularGridInterpolator((ys, xs), field, bounds_error=False, fill_value=np.nan)
    query = np.column_stack([points_xy[:, 1], points_xy[:, 0]])  # (y, x) order
    return interpolator(query)


def _domain_grid(tip_xy, tangent_xy, r_min_px, r_max_px, flank_exclusion_deg, n_r, n_theta):
    """Cell-CENTER (r, theta, points_xy) samples for a midpoint-rule 2D
    integration over the annulus, plus each cell's own (dr, dtheta) --
    constant across the grid since both r and theta are evenly spaced --
    so the caller's area element is simply `r_center * dr * dtheta` per
    sample, no edge/double-counting correction needed (unlike sampling
    directly on a `linspace` of INCLUSIVE endpoints, as core/crack_k1c.py's
    ring fit does for its own, non-integral, pointwise purpose)."""
    tx, ty = tangent_xy
    nx, ny = -ty, tx
    excl = np.radians(flank_exclusion_deg)
    r_edges = np.linspace(r_min_px, r_max_px, n_r + 1)
    theta_edges = np.linspace(-np.pi + excl, np.pi - excl, n_theta + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    dr = (r_max_px - r_min_px) / n_r
    dtheta = (2.0 * np.pi - 2.0 * excl) / n_theta

    r_grid, theta_grid = np.meshgrid(r_centers, theta_centers, indexing="ij")
    r_flat = r_grid.ravel()
    theta_flat = theta_grid.ravel()
    c, s = np.cos(theta_flat), np.sin(theta_flat)
    dx = r_flat * (c * tx + s * nx)
    dy = r_flat * (c * ty + s * ny)
    points_xy = np.column_stack([tip_xy[0] + dx, tip_xy[1] + dy])
    return points_xy, r_flat, theta_flat, dr, dtheta


def _local_gradient(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, points_xy: np.ndarray,
                     tangent_xy, normal_xy):
    """The ACTUAL field's own spatial derivative along the local tangent
    (`d/dx1`) and normal (`d/dx2`) directions, at `points_xy` -- by
    central finite difference on `field`'s own regular (xs, ys) pixel
    grid (`np.gradient`), then bilinearly interpolated onto `points_xy`
    exactly like `_sample_field` does for a raw value. Units: same as
    `field` per PIXEL (not yet converted to meters/converted to a strain
    -- the caller does that once, after combining with `pixel_size_mm`)."""
    d_dy, d_dx = np.gradient(field, ys, xs)
    dx_interp = _sample_field(d_dx, xs, ys, points_xy)
    dy_interp = _sample_field(d_dy, xs, ys, points_xy)
    tx, ty = tangent_xy
    nx, ny = normal_xy
    d_dx1 = dx_interp * tx + dy_interp * ty
    d_dx2 = dx_interp * nx + dy_interp * ny
    return d_dx1, d_dx2


def _stress_from_strain(eps11, eps22, eps12, young_modulus_pa: float, poisson_ratio: float,
                         plane_stress: bool):
    """Standard isotropic linear-elastic constitutive law, plane stress or
    plane strain -- the one place this module needs `young_modulus_pa`/
    `poisson_ratio` directly rather than only through `kappa`/`mu_pa` (the
    interaction integral needs the actual STRESS tensor, not just the
    Williams displacement amplitude `core.crack_k1c.fit_k_single_frame`
    stops at)."""
    mu_pa = young_modulus_pa / (2.0 * (1.0 + poisson_ratio))
    if plane_stress:
        c = young_modulus_pa / (1.0 - poisson_ratio ** 2)
        sigma11 = c * (eps11 + poisson_ratio * eps22)
        sigma22 = c * (eps22 + poisson_ratio * eps11)
    else:
        c = young_modulus_pa / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        sigma11 = c * ((1.0 - poisson_ratio) * eps11 + poisson_ratio * eps22)
        sigma22 = c * ((1.0 - poisson_ratio) * eps22 + poisson_ratio * eps11)
    sigma12 = 2.0 * mu_pa * eps12
    return sigma11, sigma22, sigma12


# ---------------------------------------------------------------------------
# Auxiliary (Williams, unit-K) fields -- SAME displacement formulas (same
# theta convention, same sign for K_II) as core/crack_k1c.py's own
# `fit_k_single_frame`, so the two methods share one Williams-series sign
# convention throughout this app, not two independently-sourced ones.
# ---------------------------------------------------------------------------

def _angular_functions(theta: np.ndarray, kappa: float, mode: str):
    """`(g_u, g_v)` -- the angular part of the Williams near-tip
    displacement field for a unit-K auxiliary field, `mode` "I" or "II",
    simplified via the half-angle identities `2*sin^2(t/2) = 1 - cos(t)`
    and `2*cos^2(t/2) = 1 + cos(t)` applied directly to
    `fit_k_single_frame`'s own basis functions (`fI_u`, `fII_u`, etc.) --
    e.g. `cos(t/2)*(kappa-1+2*sin^2(t/2))` there becomes
    `cos(t/2)*(kappa-cos(t))` here, an algebraically exact rewrite (not an
    approximation), just easier to differentiate by hand/finite-difference
    correctly. Returns plain g(theta), i.e. WITHOUT the `1/sqrt(2*pi)`
    normalization or the `1/(2*mu)` prefactor -- see `_auxiliary_gradient`
    for where those are applied."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    cos_t = np.cos(theta)
    if mode == "I":
        g_u = c * (kappa - cos_t)
        g_v = s * (kappa - cos_t)
    elif mode == "II":
        g_u = s * (kappa + 2.0 + cos_t)
        g_v = -c * (kappa - 2.0 + cos_t)
    else:
        raise ValueError(f"Unknown auxiliary mode {mode!r} (expected 'I' or 'II').")
    return g_u, g_v


def _angular_derivative(theta: np.ndarray, kappa: float, mode: str, h: float = 1e-6):
    """`g'(theta)` for both `_angular_functions` components, by central
    finite difference -- deliberately NOT a hand-differentiated closed
    form: `_angular_functions`' own trig expressions are intricate enough
    (products of half-angle and full-angle terms) that a transcription
    error in a by-hand derivative is a real risk, whereas a numerical
    derivative of an already-verified (see module tests) analytic
    function is essentially exact at this `h` (central-difference error
    ~h^2 ~1e-12, utterly negligible next to every other approximation
    already in this pipeline)."""
    g_u_plus, g_v_plus = _angular_functions(theta + h, kappa, mode)
    g_u_minus, g_v_minus = _angular_functions(theta - h, kappa, mode)
    return (g_u_plus - g_u_minus) / (2.0 * h), (g_v_plus - g_v_minus) / (2.0 * h)


def _auxiliary_gradient(r_px: np.ndarray, theta: np.ndarray, kappa: float, mu_pa: float,
                         pixel_size_mm: float, mode: str):
    """Full displacement-gradient tensor (`du1/dx1, du1/dx2, du2/dx1,
    du2/dx2`, local tangent/normal frame, in 1/meter) of the unit-K
    (`K=1 Pa*sqrt(m)`) auxiliary Williams field, `mode` "I" or "II", at
    each `(r_px, theta)` sample -- by the standard identity for any
    `F(r, theta) = sqrt(r) * h(theta)`:

        dF/dx1 = r^(-1/2) * [ h(theta)/2 * cos(theta) - h'(theta) * sin(theta) ]
        dF/dx2 = r^(-1/2) * [ h(theta)/2 * sin(theta) + h'(theta) * cos(theta) ]

    (`x1` = local tangent direction, `x2` = local normal direction, i.e.
    the SAME chain rule any crack-tip stress-field derivation from a
    displacement potential uses -- `r`, `theta` here are already the
    LOCAL polar coordinates `core.crack_k1c._sample_annulus` builds), with
    `h(theta) = g(theta) / sqrt(2*pi)` from `_angular_functions`, and the
    `1/(2*mu)` Williams prefactor applied once for both components."""
    m_per_px = pixel_size_mm * 1e-3
    r_m = r_px * m_per_px
    r_m_safe = np.where(r_m > 0, r_m, np.nan)  # r=0 is the true singularity -- see caller's own r_min>0 guard

    g_u, g_v = _angular_functions(theta, kappa, mode)
    gp_u, gp_v = _angular_derivative(theta, kappa, mode)
    inv_sqrt_r = 1.0 / np.sqrt(r_m_safe)
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    prefactor = 1.0 / (2.0 * mu_pa)

    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def _cartesian_grad(g, gp):
        dF_dx1 = inv_sqrt_r * (0.5 * g * cos_t - gp * sin_t) * inv_sqrt_2pi
        dF_dx2 = inv_sqrt_r * (0.5 * g * sin_t + gp * cos_t) * inv_sqrt_2pi
        return prefactor * dF_dx1, prefactor * dF_dx2

    du1_dx1, du1_dx2 = _cartesian_grad(g_u, gp_u)
    du2_dx1, du2_dx2 = _cartesian_grad(g_v, gp_v)
    return du1_dx1, du1_dx2, du2_dx1, du2_dx2


# ---------------------------------------------------------------------------
# The interaction integral itself
# ---------------------------------------------------------------------------

@dataclass
class InteractionIntegralResult:
    K_I_pa_sqrt_m: float | None
    K_II_pa_sqrt_m: float | None
    n_points_used: int
    # Domain-independence diagnostic (see module docstring): the SAME
    # integral re-evaluated over the inner half and outer half of the
    # annulus separately (each with its own conical q, 1 at its own inner
    # edge, 0 at its own outer edge) -- large disagreement between
    # `K_I_pa_sqrt_m` and either sub-domain value is a genuine warning
    # sign (the interaction integral's whole justification is that it
    # should NOT depend on the exact domain used), not a formal
    # uncertainty in the OLS-stderr sense `core.crack_k1c.KFitResult` has.
    K_I_inner_half_pa_sqrt_m: float | None = None
    K_I_outer_half_pa_sqrt_m: float | None = None
    K_II_inner_half_pa_sqrt_m: float | None = None
    K_II_outer_half_pa_sqrt_m: float | None = None


def _interaction_integral(a_grad, a_stress, mode, r_px, theta, kappa, mu_pa, pixel_size_mm,
                           young_modulus_pa, poisson_ratio, plane_stress, dq_dx1, dq_dx2, area_weight):
    """`M^(1,2)` itself (see module docstring) for a given ACTUAL-field
    gradient/stress (`a_grad`, `a_stress` -- already restricted to the
    finite/valid samples, same shape as `r_px`/`theta`) against a unit-K
    auxiliary field of the given `mode` ("I" or "II"), summed over the
    already-built quadrature grid (`r_px`, `theta`, `dq_dx1`, `dq_dx2`,
    `area_weight` -- see `_domain_grid`/`_integrate_domain`)."""
    a_du1_dx1, a_du1_dx2, a_du2_dx1, a_du2_dx2 = a_grad
    a_s11, a_s22, a_s12 = a_stress
    b_du1_dx1, b_du1_dx2, b_du2_dx1, b_du2_dx2 = _auxiliary_gradient(
        r_px, theta, kappa, mu_pa, pixel_size_mm, mode,
    )
    b_eps11, b_eps22, b_eps12 = b_du1_dx1, b_du2_dx2, 0.5 * (b_du1_dx2 + b_du2_dx1)
    b_s11, b_s22, b_s12 = _stress_from_strain(b_eps11, b_eps22, b_eps12, young_modulus_pa,
                                               poisson_ratio, plane_stress)
    W12 = a_s11 * b_eps11 + a_s22 * b_eps22 + 2.0 * a_s12 * b_eps12
    F1 = (a_s11 * b_du1_dx1 + a_s12 * b_du2_dx1) + (b_s11 * a_du1_dx1 + b_s12 * a_du2_dx1) - W12
    F2 = (a_s12 * b_du1_dx1 + a_s22 * b_du2_dx1) + (b_s12 * a_du1_dx1 + b_s22 * a_du2_dx1)
    integrand = F1 * dq_dx1 + F2 * dq_dx2
    return float(np.sum(integrand * area_weight))


def _integrate_domain(u_field, v_field, xs, ys, tip_xy, tangent_xy, kappa, mu_pa,
                       young_modulus_pa, poisson_ratio, plane_stress, pixel_size_mm,
                       r_min_px, r_max_px, n_radial_samples, n_angular_samples,
                       flank_exclusion_deg):
    """`(K_I, K_II, n_finite)` -- K_I/K_II extracted from the interaction
    integral evaluated against a pure-mode-I and pure-mode-II unit-K
    auxiliary field, over the annulus `[r_min_px, r_max_px]` with a single
    conical `q` spanning that whole annulus (see module docstring).

    K_I/K_II are recovered as `M(actual, aux=mode) / M(aux=mode, aux=mode)`
    -- i.e. NORMALIZED by the SAME unit-K auxiliary field's own self-
    interaction integral over the exact SAME (possibly flank-truncated)
    domain, rather than by the textbook closed-form constant `E*/2` a
    FULL (untruncated) domain would need. This is more than a
    simplification: using the textbook constant here instead would give a
    silently WRONG K_II specifically. The textbook derivation implicitly
    assumes the domain integral sweeps the FULL angular range around the
    tip (crack faces included, where their traction-free boundary
    condition is exactly what makes the standard derivation's reduction
    to K_I/K_II valid in the first place) -- `flank_exclusion_deg`
    deliberately breaks that assumption (see module docstring: it is
    reused from `core.crack_k1c.fit_k_single_frame` for architectural/
    comparability reasons, not because the interaction integral itself
    needs it). For an exact synthetic mode-II field, the naive
    `K_II = (E*/2)*M` recovers only ~84% of the true K_II at this
    module's default 15-degree flank exclusion (mode I's own shortfall is
    much smaller, ~0.4%, because a symmetric mode-I field has much less
    of its interaction-integral "signal" concentrated near the excluded
    crack-face wedge than an antisymmetric mode-II field does), an error
    that shrinks toward zero as `flank_exclusion_deg` shrinks toward 0,
    i.e. toward the textbook domain. Dividing by the SAME unit-K field's
    own truncated self-interaction corrects for this EXACTLY (not
    approximately): both the numerator and denominator lose the same
    (mode- and domain-shape-dependent) fraction of their idealized value
    to the same truncation, so the ratio is truncation-independent again
    by construction -- this recovers the true K_I/K_II to within
    quadrature error (<1%) at the default 15-degree exclusion, not just
    in the untruncated limit (see this module's unit tests).

    Returns `(None, None, n_finite)` if fewer than half the sampled grid
    cells have a finite actual-field gradient (too many NaN/decorrelated
    samples to trust an area integral, which -- unlike the ring fit's
    per-point OLS -- has no natural way to just drop a bad point without
    biasing the domain average)."""
    tx, ty = tangent_xy
    nx, ny = -ty, tx
    points_xy, r_px, theta, dr_px, dtheta = _domain_grid(
        tip_xy, tangent_xy, r_min_px, r_max_px, flank_exclusion_deg,
        n_radial_samples, n_angular_samples,
    )
    # Rotate the measured GLOBAL (image x, y) displacement field into the
    # crack-local (tangent, normal) frame BEFORE differentiating -- see the
    # module docstring's "practical adaptations" note. u1 (tangential) and
    # u2 (normal) are full arrays here (same grid as u_field/v_field), not
    # single sample values, so `_local_gradient` below still differentiates
    # w.r.t. the true pixel grid; only the field's own COMPONENTS are
    # rotated, not its sample positions (`_domain_grid` already built those
    # in the local frame on its own).
    u1_field = u_field * tx + v_field * ty
    u2_field = -u_field * ty + v_field * tx
    a_du1_dx1, a_du1_dx2 = _local_gradient(u1_field, xs, ys, points_xy, (tx, ty), (nx, ny))
    a_du2_dx1, a_du2_dx2 = _local_gradient(u2_field, xs, ys, points_xy, (tx, ty), (nx, ny))
    m_per_px = pixel_size_mm * 1e-3
    # NO unit conversion needed here, deliberately: `u_field`/`v_field`
    # AND `xs`/`ys` are both in the SAME pixel units (this app's whole
    # convention, e.g. `core.crack_k1c.fit_k_single_frame`'s own
    # `u_vals`/`v_vals`), so `d(u_field)/d(x_pixel)` -- a px-per-px ratio
    # -- is ALREADY dimensionless and numerically IDENTICAL to the true
    # physical strain `d(u_meters)/d(x_meters)`: the `m_per_px` factor
    # that would convert each of numerator and denominator separately
    # cancels exactly between them. Do NOT re-apply `m_per_px` here:
    # `_local_gradient`'s output is already the dimensionless strain, and
    # dividing by `m_per_px` again would inflate K_I/K_II by roughly
    # `1/m_per_px`.

    finite = np.isfinite(a_du1_dx1) & np.isfinite(a_du1_dx2) & np.isfinite(a_du2_dx1) & np.isfinite(a_du2_dx2)
    n_finite = int(np.count_nonzero(finite))
    if n_finite < r_px.size // 2 or n_finite < 8:
        return None, None, n_finite

    r_px_f = r_px[finite]
    theta_f = theta[finite]
    a_eps11 = a_du1_dx1[finite]
    a_eps22 = a_du2_dx2[finite]
    a_eps12 = 0.5 * (a_du1_dx2[finite] + a_du2_dx1[finite])
    a_stress = _stress_from_strain(a_eps11, a_eps22, a_eps12, young_modulus_pa, poisson_ratio, plane_stress)
    actual_grad = (a_du1_dx1[finite], a_du1_dx2[finite], a_du2_dx1[finite], a_du2_dx2[finite])

    q_prime_r = -1.0 / ((r_max_px - r_min_px) * m_per_px)  # conical q, constant slope, in 1/meter
    dq_dx1 = q_prime_r * np.cos(theta_f)
    dq_dx2 = q_prime_r * np.sin(theta_f)
    area_weight = r_px_f * dr_px * dtheta * (m_per_px ** 2)  # dA in m^2 (r*dr*dtheta is in px^2)

    K = {}
    for mode in ("I", "II"):
        M_actual = _interaction_integral(actual_grad, a_stress, mode, r_px_f, theta_f, kappa, mu_pa,
                                          pixel_size_mm, young_modulus_pa, poisson_ratio, plane_stress,
                                          dq_dx1, dq_dx2, area_weight)
        # Self-normalization (see this function's own docstring): the SAME
        # unit-K auxiliary field used as `M_actual`'s aux argument, now
        # ALSO used as the "actual" field, against itself -- i.e. its own
        # truncated self-interaction, whose theoretical (untruncated)
        # value is the textbook `2/E*` (self-check: setting `mode`'s own
        # `_auxiliary_gradient`/`_stress_from_strain` output as BOTH
        # arguments below reduces `_interaction_integral` to exactly
        # M(unit_mode, unit_mode)).
        b_du1_dx1, b_du1_dx2, b_du2_dx1, b_du2_dx2 = _auxiliary_gradient(
            r_px_f, theta_f, kappa, mu_pa, pixel_size_mm, mode,
        )
        b_eps11, b_eps22, b_eps12 = b_du1_dx1, b_du2_dx2, 0.5 * (b_du1_dx2 + b_du2_dx1)
        b_stress = _stress_from_strain(b_eps11, b_eps22, b_eps12, young_modulus_pa, poisson_ratio, plane_stress)
        b_grad = (b_du1_dx1, b_du1_dx2, b_du2_dx1, b_du2_dx2)
        M_self = _interaction_integral(b_grad, b_stress, mode, r_px_f, theta_f, kappa, mu_pa,
                                        pixel_size_mm, young_modulus_pa, poisson_ratio, plane_stress,
                                        dq_dx1, dq_dx2, area_weight)
        K[mode] = M_actual / M_self if M_self else None

    return K["I"], K["II"], n_finite


def fit_k_interaction_integral_single_frame(
    u_field: np.ndarray, v_field: np.ndarray, xs: np.ndarray, ys: np.ndarray,
    tip_xy, tangent_xy, kappa: float, mu_pa: float, young_modulus_pa: float, poisson_ratio: float,
    plane_stress: bool, pixel_size_mm: float, r_min_px: float, r_max_px: float,
    flank_exclusion_deg: float = 15.0, n_radial_samples: int = 12, n_angular_samples: int = 24,
) -> InteractionIntegralResult:
    """Fits K_I, K_II to the DIC displacement field `u_field`/`v_field`
    around `tip_xy` via the domain-form interaction integral -- see the
    module docstring for the method, its references, and the practical
    adaptations made for a regular-grid DIC field. An ALTERNATIVE to
    `core.crack_k1c.fit_k_single_frame` (same inputs, directly comparable
    -- pass the exact same `tip_xy`/`tangent_xy`/`r_min_px`/`r_max_px`/
    `flank_exclusion_deg` to compare the two on identical terms), not a
    replacement for it.

    `r_min_px` MUST be strictly positive here (unlike the ring fit, which
    tolerates `r_min_px=0`): the auxiliary field's STRESS -- not just its
    displacement -- is singular at r=0 (a genuine `1/sqrt(r)` blow-up, the
    whole physical point of K_I/K_II), so a domain that reached r=0 would
    integrate through that singularity rather than merely evaluating a
    finite (if large) basis function there.

    `n_radial_samples`/`n_angular_samples` default higher than the ring
    fit's own (12x24 = 288 cells vs. the ring fit's default 8x16 = 128
    points) -- an area integral's accuracy depends on the density of the
    quadrature grid approximating it, whereas the ring fit is an exact
    OLS regardless of how many points happen to be sampled (more points
    there only improve its STATISTICAL precision, not remove a
    discretization error this integral doesn't have an equivalent
    exemption from).

    Returns `InteractionIntegralResult` with every field `None` (besides
    `n_points_used`) if too few of the sampled domain's cells landed on a
    finite (non-NaN) actual-field gradient -- see `_integrate_domain`."""
    if r_max_px <= r_min_px:
        raise InteractionIntegralError(
            "The interaction integral's search radius's max must be larger than its min."
        )
    if not r_min_px or r_min_px <= 0:
        raise InteractionIntegralError(
            "The interaction integral needs a strictly positive r_min (unlike the ring fit) -- "
            "its auxiliary field's stress is singular at r=0."
        )
    if not pixel_size_mm or pixel_size_mm <= 0:
        raise InteractionIntegralError(
            "A pixel size (mm/px) is required to compute K_I/K_II in physical units."
        )

    K_I, K_II, n_finite = _integrate_domain(
        u_field, v_field, xs, ys, tip_xy, tangent_xy, kappa, mu_pa, young_modulus_pa, poisson_ratio,
        plane_stress, pixel_size_mm, r_min_px, r_max_px, n_radial_samples, n_angular_samples,
        flank_exclusion_deg,
    )
    if K_I is None:
        return InteractionIntegralResult(None, None, n_finite)

    # Domain-independence diagnostic: split the SAME annulus at its
    # geometric midpoint radius into an inner and outer half, each with
    # its OWN conical q (1 at ITS inner edge, 0 at ITS outer edge) --
    # re-using `_integrate_domain` unchanged, just with a narrower
    # `r_min_px`/`r_max_px` window each time. See `InteractionIntegralResult`'s
    # own docstring for why this is a meaningful check here specifically.
    r_mid_px = 0.5 * (r_min_px + r_max_px)
    n_r_half = max(4, n_radial_samples // 2)
    K_I_inner, K_II_inner, _n_in = _integrate_domain(
        u_field, v_field, xs, ys, tip_xy, tangent_xy, kappa, mu_pa, young_modulus_pa, poisson_ratio,
        plane_stress, pixel_size_mm, r_min_px, r_mid_px, n_r_half, n_angular_samples, flank_exclusion_deg,
    )
    K_I_outer, K_II_outer, _n_out = _integrate_domain(
        u_field, v_field, xs, ys, tip_xy, tangent_xy, kappa, mu_pa, young_modulus_pa, poisson_ratio,
        plane_stress, pixel_size_mm, r_mid_px, r_max_px, n_r_half, n_angular_samples, flank_exclusion_deg,
    )

    return InteractionIntegralResult(
        K_I_pa_sqrt_m=K_I, K_II_pa_sqrt_m=K_II, n_points_used=n_finite,
        K_I_inner_half_pa_sqrt_m=K_I_inner, K_I_outer_half_pa_sqrt_m=K_I_outer,
        K_II_inner_half_pa_sqrt_m=K_II_inner, K_II_outer_half_pa_sqrt_m=K_II_outer,
    )
