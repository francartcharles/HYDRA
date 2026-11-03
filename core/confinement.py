"""
Confining-cell (quasi-oedometric) and laser-velocimetry stress
computation, ported from display_results.py's `confine_show()` /
laser-stress block.

This is a post-processing step: it takes the already-computed axial
stress/strain from a Processing tab and combines it with (a) an optional
confinement-cell strain-gage signal to derive the radial stress, and/or
(b) an optional laser-velocimetry signal to derive an independent measure
of the axial (output) stress -- exactly the "confine_check" / "laser_check"
options in the original.

Two confinement-cell models, matching `corr_file.nonlinear_check`:
- LINEAR (0): radial stress is a linear blend of two calibration
  constants C1/C2 between zero strain and a reference strain.
- NONLINEAR (1): two 6th-order polynomial fits (orthoradial cell strain
  -> radial stress), one calibrated at the specimen's initial height and
  one at its height at the reference strain (typically from a numerical
  simulation of the cell), blended by how far the specimen has currently
  compressed between those two states.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sanitize_std_array(std, n: int) -> np.ndarray:
    """A per-index uncertainty array (e.g. axial_stress_std, laser_stress_
    std) used directly as an rng.normal() `scale` array: NaN entries (a
    custom sensor's Monte-Carlo cut is padded with NaN outside that
    sensor's own recorded time span, see core.custom_sensor.
    cut_and_resample, and that NaN can flow all the way here through
    ResultsExportPage._sensor_signal_for_result) are replaced with 0 --
    "no known perturbation at that point" -- rather than left to produce a
    NaN draw at that one index."""
    if std is None or not len(std):
        return np.zeros(n)
    return np.nan_to_num(np.asarray(std[:n], dtype=float), nan=0.0)


def _sanitize_scalar_std(std) -> float:
    """A per-curve uncertainty array collapsed to a SINGLE scalar (e.g.
    `defl_std`, perturbing the whole cell-deflection curve by one draw-
    wide offset). A plain np.mean() here means even one NaN anywhere in
    the input -- routine now that it can carry a Monte-Carlo-cut sensor's
    cut_and_resample NaN padding -- poisons the mean into NaN, which
    rng.normal(0, nan) then turns into a NaN SCALAR added to every sample
    of the curve it perturbs, silently NaN-ing out an entire downstream
    envelope with no exception raised to catch it (rng.normal happily
    returns nan rather than erroring on a NaN scale). np.nanmean ignores
    the NaN entries instead, using whatever finite dispersion the covered
    portion of the curve does have; only an ALL-NaN (or empty) input falls
    back to 0.0 (no perturbation)."""
    if std is None or not len(std):
        return 0.0
    arr = np.asarray(std, dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else 0.0


def laser_stress(
    ob_diameter: float, ob_rho: float, ob_c0: float, specimen_diameter: float,
    laser_velocity: np.ndarray,
) -> np.ndarray:
    """Axial (output) stress derived from a laser-measured output-bar
    surface velocity, instead of the output-bar strain gage. Matches:
    `(D0**2/Dspe**2) * 0.5 * rho0 * C0 * laser_velocity / 1e6` (MPa).
    """
    return (ob_diameter ** 2 / specimen_diameter ** 2) * 0.5 * ob_rho * ob_c0 * laser_velocity / 1e6


def linear_cell_radial_stress(
    axial_strain: np.ndarray, cell_deflection: np.ndarray,
    cell_modulus: float, C1: float, C2: float, ref_strain: float,
) -> np.ndarray:
    """Radial stress (MPa) from the confinement cell's own strain-gage
    deflection, using the linear cell-calibration model. Matches:
    `E_cell * ((axial_strain/ref_strain)*C2 + (1-axial_strain/ref_strain)*C1) / 1e6`
    where `E_cell = cell_modulus * cell_deflection` (the "effective cell
    modulus x deflection" term in the original, `Ecell_deff`).
    """
    n = min(len(axial_strain), len(cell_deflection))
    eps, defl = np.asarray(axial_strain[:n]), np.asarray(cell_deflection[:n])
    e_cell_deff = cell_modulus * defl
    blend = (eps / ref_strain) * C2 + (1 - eps / ref_strain) * C1
    return e_cell_deff * blend / 1e6


def _eval_poly_no_constant(coeffs: np.ndarray, x) -> np.ndarray:
    """Evaluates a degree-6 `np.polyfit` polynomial WITHOUT its constant
    term (coeffs[6]) -- matches the original exactly:
    `c0*x**6 + c1*x**5 + c2*x**4 + c3*x**3 + c4*x**2 + c5*x`, forcing the
    curve through the origin (zero orthoradial strain -> zero radial
    stress), which is physically sensible for this calibration."""
    x = np.asarray(x)
    return sum(coeffs[k] * x ** (6 - k) for k in range(6))


def fit_nonlinear_calibration(
    cell_diameter: float, ortho_disp_ini, radial_ini, ortho_disp_esp, radial_esp,
) -> tuple[np.ndarray, np.ndarray]:
    """Fits the two degree-6 calibration polynomials the nonlinear cell
    model needs (orthoradial cell strain -> radial stress, at the
    specimen's initial height and at its height at `ref_strain`) ONCE --
    factored out of `nonlinear_cell_radial_stress` so a caller that
    evaluates that model many times against the SAME calibration curves
    (e.g. once per Monte-Carlo draw in a Processing tab's own loop, see
    `nonlinear_cell_radial_stress_precomputed`/`per_draw_confinement_
    quantities`) doesn't re-fit them on every single call -- `np.polyfit`
    is by far the most expensive part of this model."""
    r_cell = cell_diameter / 2
    ortho_def_ini = np.asarray(ortho_disp_ini) / r_cell
    ortho_def_esp = np.asarray(ortho_disp_esp) / r_cell
    radial_ini_mpa = np.abs(np.asarray(radial_ini)) / 1e6
    radial_esp_mpa = np.abs(np.asarray(radial_esp)) / 1e6
    ff1 = np.polyfit(ortho_def_ini, radial_ini_mpa, 6)
    ff2 = np.polyfit(ortho_def_esp, radial_esp_mpa, 6)
    return ff1, ff2


def nonlinear_cell_radial_stress_precomputed(
    axial_nominal_strain: np.ndarray, cell_deflection: np.ndarray,
    specimen_initial_length: float, ref_strain: float,
    poly_ini: np.ndarray, poly_esp: np.ndarray,
) -> np.ndarray:
    """Same blend as `nonlinear_cell_radial_stress`, but taking the two
    already-fitted calibration polynomials (`fit_nonlinear_calibration`)
    instead of re-fitting them from the raw calibration curves every
    call."""
    l_ini = specimen_initial_length
    l_esp = (1 - ref_strain) * l_ini

    n = min(len(axial_nominal_strain), len(cell_deflection))
    strain, defl = np.asarray(axial_nominal_strain[:n]), np.asarray(cell_deflection[:n])
    dhv = (1 - strain) * l_ini  # current specimen height, from the nominal (engineering) strain

    drad1 = _eval_poly_no_constant(poly_ini, defl) * (dhv - l_esp) / (l_ini - l_esp)
    drad2 = _eval_poly_no_constant(poly_esp, defl) * (l_ini - dhv) / (l_ini - l_esp)
    return drad1 + drad2


def nonlinear_cell_radial_stress(
    axial_nominal_strain: np.ndarray, cell_deflection: np.ndarray,
    specimen_initial_length: float, ref_strain: float, cell_diameter: float,
    ortho_disp_ini, radial_ini, ortho_disp_esp, radial_esp,
) -> np.ndarray:
    """Radial stress (MPa) from the confinement cell's own strain-gage
    deflection, using the nonlinear cell-calibration model: two 6th-order
    polynomial fits of orthoradial cell strain -> radial stress -- one
    calibrated at the specimen's initial height (`_ini`), one at its
    height at `ref_strain` (`_esp`, typically from separate numerical
    simulations of the confinement cell at those two states) -- blended by
    how far the specimen has currently compressed between those two
    reference heights.

    Fits the two calibration polynomials fresh every call -- fine for a
    one-off (mean-curve) evaluation, but see `fit_nonlinear_calibration` +
    `nonlinear_cell_radial_stress_precomputed` for a caller that needs to
    evaluate this model many times (e.g. per Monte-Carlo draw).
    """
    poly_ini, poly_esp = fit_nonlinear_calibration(
        cell_diameter, ortho_disp_ini, radial_ini, ortho_disp_esp, radial_esp,
    )
    return nonlinear_cell_radial_stress_precomputed(
        axial_nominal_strain, cell_deflection, specimen_initial_length, ref_strain, poly_ini, poly_esp,
    )


@dataclass
class StressInvariants:
    pressure: np.ndarray   # hydrostatic pressure p = (axial + 2*radial) / 3
    deviator: np.ndarray   # stress deviator q = |axial - radial|


def stress_invariants(axial_stress: np.ndarray, radial_stress: np.ndarray) -> StressInvariants:
    """Matches: `pressure = (axial + 2*radial)/3`, `deviator = abs(axial - radial)`."""
    n = min(len(axial_stress), len(radial_stress))
    axial, radial = np.asarray(axial_stress[:n]), np.asarray(radial_stress[:n])
    pressure = (axial + 2 * radial) / 3
    deviator = np.abs(axial - radial)
    return StressInvariants(pressure=pressure, deviator=deviator)


@dataclass
class ConfinementEnvelope:
    """Mean +/- 1-sigma across a Monte-Carlo repetition of the confinement
    calculation, for axial/radial/pressure/deviator -- matches the
    `m_axial/d_axial`, `m_radial/d_radial`, ... pairs computed in
    `display_results.py`'s `confine_show()`."""
    axial_mean: np.ndarray
    axial_std: np.ndarray
    radial_mean: np.ndarray
    radial_std: np.ndarray
    pressure_mean: np.ndarray
    pressure_std: np.ndarray
    deviator_mean: np.ndarray
    deviator_std: np.ndarray


def confinement_envelope(
    axial_strain: np.ndarray, axial_stress: np.ndarray, axial_stress_std: np.ndarray | None,
    cell_deflection: np.ndarray, cell_deflection_std: np.ndarray | None,
    cell_modulus: float, C1: float, C2: float, ref_strain: float,
    n_draws: int = 200, cell_modulus_std_fraction: float = 0.02,
    rng: np.random.Generator | None = None,
) -> ConfinementEnvelope:
    """Repeats the linear-cell radial-stress + invariants calculation
    `n_draws` times, perturbing the axial stress and cell deflection by
    their own uncertainty (and the cell modulus by a fixed 2% std, matching
    the original's `E_cd = np.random.normal(E_c, 0.02*E_c, N)`), and
    returns the mean +/- 1-sigma envelope for each quantity."""
    rng = rng or np.random.default_rng()
    n = min(len(axial_strain), len(axial_stress), len(cell_deflection))
    strain = np.asarray(axial_strain[:n])
    axial = np.asarray(axial_stress[:n])
    defl = np.asarray(cell_deflection[:n])
    axial_std = _sanitize_std_array(axial_stress_std, n)
    defl_std = _sanitize_scalar_std(cell_deflection_std)

    axial_draws, radial_draws, pressure_draws, deviator_draws = [], [], [], []
    for _ in range(n_draws):
        axial_d = axial + rng.normal(0, axial_std) if np.any(axial_std) else axial
        defl_d = defl + rng.normal(0, defl_std) if defl_std else defl
        modulus_d = rng.normal(cell_modulus, cell_modulus_std_fraction * cell_modulus)
        radial_d = linear_cell_radial_stress(strain, defl_d, modulus_d, C1, C2, ref_strain)
        inv = stress_invariants(axial_d, radial_d)
        axial_draws.append(axial_d[: len(radial_d)])
        radial_draws.append(radial_d)
        pressure_draws.append(inv.pressure)
        deviator_draws.append(inv.deviator)

    axial_arr, radial_arr = np.array(axial_draws), np.array(radial_draws)
    pressure_arr, deviator_arr = np.array(pressure_draws), np.array(deviator_draws)
    return ConfinementEnvelope(
        axial_mean=axial_arr.mean(axis=0), axial_std=axial_arr.std(axis=0),
        radial_mean=radial_arr.mean(axis=0), radial_std=radial_arr.std(axis=0),
        pressure_mean=pressure_arr.mean(axis=0), pressure_std=pressure_arr.std(axis=0),
        deviator_mean=deviator_arr.mean(axis=0), deviator_std=deviator_arr.std(axis=0),
    )


def nonlinear_confinement_envelope(
    axial_nominal_strain: np.ndarray, axial_stress: np.ndarray, axial_stress_std: np.ndarray | None,
    cell_deflection: np.ndarray, cell_deflection_std: np.ndarray | None,
    specimen_initial_length: float, ref_strain: float, cell_diameter: float,
    ortho_disp_ini, radial_ini, ortho_disp_esp, radial_esp,
    n_draws: int = 200,
    rng: np.random.Generator | None = None,
) -> ConfinementEnvelope:
    """Same idea as `confinement_envelope` above, but for the nonlinear
    cell model -- matches the original's own per-draw loop (`rand_val[i]`
    perturbing the cell deflection, `rand_val_2[i]` perturbing the axial
    stress) around the two polynomial calibration curves."""
    rng = rng or np.random.default_rng()
    n = min(len(axial_nominal_strain), len(axial_stress), len(cell_deflection))
    strain = np.asarray(axial_nominal_strain[:n])
    axial = np.asarray(axial_stress[:n])
    defl = np.asarray(cell_deflection[:n])
    axial_std = _sanitize_std_array(axial_stress_std, n)
    defl_std = _sanitize_scalar_std(cell_deflection_std)

    axial_draws, radial_draws, pressure_draws, deviator_draws = [], [], [], []
    for _ in range(n_draws):
        axial_d = axial + rng.normal(0, axial_std) if np.any(axial_std) else axial
        defl_d = defl + rng.normal(0, defl_std) if defl_std else defl
        radial_d = nonlinear_cell_radial_stress(
            strain, defl_d, specimen_initial_length, ref_strain, cell_diameter,
            ortho_disp_ini, radial_ini, ortho_disp_esp, radial_esp,
        )
        inv = stress_invariants(axial_d, radial_d)
        axial_draws.append(axial_d[: len(radial_d)])
        radial_draws.append(radial_d)
        pressure_draws.append(inv.pressure)
        deviator_draws.append(inv.deviator)

    axial_arr, radial_arr = np.array(axial_draws), np.array(radial_draws)
    pressure_arr, deviator_arr = np.array(pressure_draws), np.array(deviator_draws)
    return ConfinementEnvelope(
        axial_mean=axial_arr.mean(axis=0), axial_std=axial_arr.std(axis=0),
        radial_mean=radial_arr.mean(axis=0), radial_std=radial_arr.std(axis=0),
        pressure_mean=pressure_arr.mean(axis=0), pressure_std=pressure_arr.std(axis=0),
        deviator_mean=deviator_arr.mean(axis=0), deviator_std=deviator_arr.std(axis=0),
    )


def radial_strain(axial_strain, cell_strain, alpha00: float, alpha0_l2: float, n_iter: int = 50) -> np.ndarray:
    """Radial (orthoradial) strain from the confinement cell's own strain
    signal, via an implicit relation (solved here by fixed-point
    iteration, since radial_strain appears on both sides):

        eps_rad = (2/3) * (1 - eps_axial*(1 + eps_axial/2)) * alpha00 * eps_cell
                  + ((1 + eps_rad)**2 / 3) * alpha0_l2 * eps_cell

    `alpha00`/`alpha0_l2` are calibration constants from a numerical
    simulation of the confinement cell (entered on the Special Tests page).
    """
    axial_strain = np.asarray(axial_strain)
    cell_strain = np.asarray(cell_strain)
    n = min(len(axial_strain), len(cell_strain))
    eps_axial = axial_strain[:n]
    eps_cell = cell_strain[:n]
    a_term = (2 / 3) * (1 - eps_axial * (1 + eps_axial / 2)) * alpha00 * eps_cell
    b_term = alpha0_l2 * eps_cell / 3
    eps_rad = np.zeros(n)
    for _ in range(n_iter):
        eps_rad = a_term + b_term * (1 + eps_rad) ** 2
    return eps_rad


def volumetric_strain(axial_strain, radial_strain_value) -> np.ndarray:
    """eps_v = (1 + eps_axial)(1 + eps_rad)^2 - 1"""
    n = min(len(axial_strain), len(radial_strain_value))
    eps_axial = np.asarray(axial_strain[:n])
    eps_rad = np.asarray(radial_strain_value[:n])
    return (1 + eps_axial) * (1 + eps_rad) ** 2 - 1


def equivalent_strain(axial_strain, radial_strain_value) -> np.ndarray:
    """eps_eq = (2/3) |ln(1 + eps_axial) - eps_rad|"""
    n = min(len(axial_strain), len(radial_strain_value))
    eps_axial = np.asarray(axial_strain[:n])
    eps_rad = np.asarray(radial_strain_value[:n])
    return (2 / 3) * np.abs(np.log(1 + eps_axial) - eps_rad)


def strain_rate_from_strain(strain, time) -> np.ndarray:
    """Time derivative d(strain)/dt of a strain curve sampled at `time`,
    same length as the (common-length-truncated) inputs -- central
    differences inside, one-sided at the two ends (numpy.gradient). Used for
    the radial / volumetric / equivalent strain RATES of the confinement
    analysis; it differentiates exactly the strain definitions returned by
    radial_strain / volumetric_strain / equivalent_strain (so the volumetric
    rate is negative while the specimen compacts, and the equivalent rate
    can change sign on unloading).

    A time vector with duplicated/non-increasing stamps (e.g. a repeated row
    in the raw time file) would make a plain gradient divide by zero: in
    that case the median of the strictly positive steps is used as a uniform
    step instead, the same idea as process_shpb._robust_dt. Raises
    ValueError if no usable step exists."""
    e = np.asarray(strain, dtype=float)
    t = np.asarray(time, dtype=float)
    n = min(len(e), len(t))
    e, t = e[:n], t[:n]
    if n < 2:
        return np.zeros(n)
    dt = np.diff(t)
    if np.all(dt > 0):
        return np.gradient(e, t)
    good = dt[np.isfinite(dt) & (dt > 0)]
    if not len(good):
        raise ValueError("Cannot compute strain rates: the time vector has no increasing samples.")
    return np.gradient(e, float(np.median(good)))


def per_draw_confinement_quantities(
    axial_strain: np.ndarray, axial_stress: np.ndarray, cell_deflection: np.ndarray,
    conf,  # a ConfinementSettings instance
    specimen_initial_length: float,
    poly_ini: np.ndarray | None = None, poly_esp: np.ndarray | None = None,
    time: np.ndarray | None = None,
) -> dict:
    """Confinement-derived quantities (radial stress, hydrostatic
    pressure, stress deviator, radial/volumetric/equivalent strain and --
    when `time` is given -- their strain RATES) for a
    SINGLE Monte-Carlo draw's own axial strain/stress and confinement-cell
    signal -- the true-propagation counterpart of `triaxial_synthetic_
    draws` above. Meant to be called once per draw INSIDE a Processing
    tab's own Monte-Carlo loop (see page_process.py's `_SHPBTab`/`_DITab`/
    `_QSTab._on_run`), so these quantities inherit that draw's own real
    specimen/bar/gage-cut realization instead of being resampled from
    already-aggregated mean curves afterward in Results & Export.

    `axial_strain`/`axial_stress` should be whichever this app's own
    Results & Export page prefers for a given result (nominal/engineering
    if available, else true) -- matching `_refresh_confinement`'s own
    choice, so the same quantities feed both the plotted mean curve and
    this per-draw uncertainty.

    For the nonlinear cell model, pass `poly_ini`/`poly_esp` (from
    `fit_nonlinear_calibration`, computed ONCE outside the Monte-Carlo
    loop) to avoid re-fitting the same two calibration polynomials on
    every single draw; they're fitted here (once, this call only) if
    omitted.

    `time` is this draw's own time vector (same sampling as `axial_strain`);
    if given, the radial/volumetric/equivalent strain rates are returned as
    well, differentiated per draw so they inherit that draw's own
    realization like every other quantity here.

    Returns a dict with `confinement_`-prefixed keys, ready to be merged
    straight into a Monte-Carlo loop's per-draw dict -- `_aggregate_
    quantities`/`_store_raw_draws` then pick them up automatically, with
    zero extra plumbing, exactly like every other per-draw quantity."""
    if conf.nonlinear_enabled:
        if poly_ini is None or poly_esp is None:
            poly_ini, poly_esp = fit_nonlinear_calibration(
                conf.cell_diameter, conf.calib_ortho_disp_ini, conf.calib_radial_ini,
                conf.calib_ortho_disp_esp, conf.calib_radial_esp,
            )
        radial = nonlinear_cell_radial_stress_precomputed(
            axial_strain, cell_deflection, specimen_initial_length, conf.ref_strain, poly_ini, poly_esp,
        )
    else:
        radial = linear_cell_radial_stress(
            axial_strain, cell_deflection, conf.cell_modulus, conf.C1, conf.C2, conf.ref_strain,
        )

    n = min(len(axial_strain), len(axial_stress), len(radial))
    strain_n = np.asarray(axial_strain[:n])
    axial_n = np.asarray(axial_stress[:n])
    radial_n = np.asarray(radial[:n])
    inv = stress_invariants(axial_n, radial_n)

    # Same sign-convention flip as _refresh_confinement: this app keeps
    # compression positive everywhere, but the strain formulas below
    # expect axial strain negative in compression.
    axial_strain_signed = -strain_n
    cell_n = np.asarray(cell_deflection[:n])
    eps_rad = radial_strain(axial_strain_signed, cell_n, conf.alpha00, conf.alpha0_l2)
    eps_v = volumetric_strain(axial_strain_signed, eps_rad)
    eps_eq = equivalent_strain(axial_strain_signed, eps_rad)

    out = {
        "confinement_radial_stress": radial_n,
        "confinement_pressure": inv.pressure,
        "confinement_deviator": inv.deviator,
        "confinement_radial_strain": eps_rad,
        "confinement_volumetric_strain": eps_v,
        "confinement_equivalent_strain": eps_eq,
    }
    if time is not None and len(time) >= 2:
        out["confinement_radial_strain_rate"] = strain_rate_from_strain(eps_rad, time)
        out["confinement_volumetric_strain_rate"] = strain_rate_from_strain(eps_v, time)
        out["confinement_equivalent_strain_rate"] = strain_rate_from_strain(eps_eq, time)
    return out


def triaxial_synthetic_draws(
    axial_nominal_strain: np.ndarray, axial_stress: np.ndarray, axial_stress_std: np.ndarray | None,
    cell_deflection: np.ndarray, cell_deflection_std: np.ndarray | None,
    conf,  # a ConfinementSettings instance
    specimen_initial_length: float,
    laser_stress_value: np.ndarray | None = None, laser_stress_std: np.ndarray | None = None,
    n_draws: int = 200,
    rng: np.random.Generator | None = None,
    time: np.ndarray | None = None,
) -> dict:
    """Generates `n_draws` synthetic Monte-Carlo-like draws (by perturbing
    the axial stress and cell deflection by their own reported
    uncertainty, the same way `confinement_envelope` does) of every
    triaxial quantity at once -- axial/radial/pressure/deviator stress and
    radial/volumetric/equivalent strain, plus their laser-based stress
    variants if a laser axial stress is supplied -- for feeding into the
    Covariance Analysis page (which needs individual draws, not just a
    mean +/- std envelope).

    If `time` (the result's time vector) is given, the radial/volumetric/
    equivalent strain rates ("radial_strain_rate", ...) are added too.

    Returns a dict of {quantity_name: list_of_per_draw_arrays}, i.e. the
    same shape `project.last_raw_draws[...]` already uses elsewhere.
    """
    rng = rng or np.random.default_rng()
    n = min(len(axial_nominal_strain), len(axial_stress), len(cell_deflection))
    strain = np.array(axial_nominal_strain[:n], dtype=float)
    if len(strain) and abs(strain[0]) < 1e-6:
        # Snap a near-zero first point to exactly 0.0 -- a tiny residual
        # (floating-point noise from upstream processing) here otherwise
        # means "Reference (start) strain = 0.0" on the Covariance
        # Analysis page finds no draw actually starting AT 0, requiring a
        # manual nudge of the reference value every time. Physically, the
        # first point IS the start of the test, so it should be exactly 0.
        strain[0] = 0.0
    axial = np.asarray(axial_stress[:n])
    defl = np.asarray(cell_deflection[:n])
    axial_std = _sanitize_std_array(axial_stress_std, n)
    defl_std = _sanitize_scalar_std(cell_deflection_std)

    axial_strain_signed = -strain  # this app keeps compression positive; the strain formulas expect it negative

    out = {
        "axial_stress": [], "radial_stress": [], "pressure": [], "deviator": [],
        "radial_strain": [], "volumetric_strain": [], "equivalent_strain": [],
    }
    with_rates = time is not None and len(time) >= 2
    if with_rates:
        out.update({"radial_strain_rate": [], "volumetric_strain_rate": [], "equivalent_strain_rate": []})
        time_arr = np.asarray(time, dtype=float)
    has_laser = laser_stress_value is not None and len(laser_stress_value) > 0
    if has_laser:
        out.update({"axial_stress_laser": [], "pressure_laser": [], "deviator_laser": []})
        laser_axial = np.asarray(laser_stress_value[:n])
        laser_std = _sanitize_std_array(laser_stress_std, len(laser_axial))

    for _ in range(n_draws):
        axial_d = axial + rng.normal(0, axial_std) if np.any(axial_std) else axial
        defl_d = defl + rng.normal(0, defl_std) if defl_std else defl

        if conf.nonlinear_enabled:
            radial_d = nonlinear_cell_radial_stress(
                strain, defl_d, specimen_initial_length, conf.ref_strain, conf.cell_diameter,
                conf.calib_ortho_disp_ini, conf.calib_radial_ini, conf.calib_ortho_disp_esp, conf.calib_radial_esp,
            )
        else:
            modulus_d = rng.normal(conf.cell_modulus, conf.cell_modulus_std_fraction * conf.cell_modulus)
            radial_d = linear_cell_radial_stress(strain, defl_d, modulus_d, conf.C1, conf.C2, conf.ref_strain)

        m = min(n, len(radial_d))
        inv = stress_invariants(axial_d[:m], radial_d[:m])
        eps_rad = radial_strain(axial_strain_signed[:m], defl_d[:m], conf.alpha00, conf.alpha0_l2)
        eps_v = volumetric_strain(axial_strain_signed[:m], eps_rad)
        eps_eq = equivalent_strain(axial_strain_signed[:m], eps_rad)

        out["axial_stress"].append(axial_d[:m])
        out["radial_stress"].append(radial_d[:m])
        out["pressure"].append(inv.pressure)
        out["deviator"].append(inv.deviator)
        out["radial_strain"].append(eps_rad)
        out["volumetric_strain"].append(eps_v)
        out["equivalent_strain"].append(eps_eq)
        if with_rates:
            tm = time_arr[:m]
            out["radial_strain_rate"].append(strain_rate_from_strain(eps_rad, tm))
            out["volumetric_strain_rate"].append(strain_rate_from_strain(eps_v, tm))
            out["equivalent_strain_rate"].append(strain_rate_from_strain(eps_eq, tm))

        if has_laser:
            laser_d = laser_axial[:m] + rng.normal(0, laser_std[:m]) if np.any(laser_std) else laser_axial[:m]
            inv_las = stress_invariants(laser_d, radial_d[:m])
            out["axial_stress_laser"].append(laser_d)
            out["pressure_laser"].append(inv_las.pressure)
            out["deviator_laser"].append(inv_las.deviator)

    out["strain"] = [strain[: len(v)] for v in out["axial_stress"]]  # common x-axis, one copy per draw
    return out
