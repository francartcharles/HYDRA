"""
Pochhammer-Chree dispersion correction.

Ported from FFT_dispersion.py / FFT_dispersion_ID.py / FFT_dispersion_visco.py
/ FFT_dispersion_visco_ID.py. Those four files were ~95% identical
copy-pastes of each other along two independent axes:

    elastic bars          <-> viscoelastic (standard linear solid) bars
    two-bar setup (SHPB)   <-> single output-bar setup (Direct Impact, "_ID")

That is preserved here as one function per axis-combination (the actual
per-frequency Bessel root-solving physics is untouched from the original),
but all module-level globals (`corr_file.heat_check`, `test_type.sp_Tempa_value`,
...) are now explicit parameters, and nothing calls `plt.show()` or a Tkinter
messagebox -- callers get back plain arrays/dataclasses to plot or display
however they like, plus an optional `progress_cb(i, total)` callback instead
of a printed progress bar.

NOTE: this module is numerically faithful to the original but has not been
re-validated against real experimental data (no test data / no way to run
sympy's `nsolve` root search in this environment) -- treat the physics as
"ported, not re-verified" and sanity-check against a known-good original run
before relying on it for a real analysis.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sympy as sym
from sympy import besselj
from sympy.functions import Abs

from core.models import Bar, Specimen
from core.rheology_models import RheologyModel
from core.parallel import run_map

ProgressCB = Optional[Callable[[int, int], None]]


def _require_window(i1: int, i2: int, label: str) -> None:
    """Raises a clear, actionable error instead of letting a cryptic
    IndexError ('index 1 is out of bounds for axis 0 with size 0') surface
    from deep inside the FFT/root-search code below when a pulse window's
    start/end don't actually bracket at least a couple of samples.

    A window can end up empty when a value left over from a different
    Correction type/setup is reused for a differently-scoped window --
    `project.corrections_windows` keys are shared across setups (e.g. the
    viscoelastic two-bar case's "Incident" window and the Direct-Impact
    case's single "Transmitted" window are BOTH stored under key "xd1"),
    so a stale value from one setup may fall outside the other's
    (possibly much shorter/differently-timed) loaded signal.
    `ui/pages/page_corrections.py`'s `_on_run` snaps such an out-of-range
    value to the *nearest* available sample for both the start and the
    end, which can collapse both onto the same edge sample when the stale
    value is far enough outside the signal's time range, producing a
    zero-length window. `ui/pages/page_corrections.py`'s
    `_refresh_picker_signal` also skips restoring an out-of-range window
    in the first place; this check is the second, independent line of
    defense in the actual computation, for any other way a window could
    end up empty."""
    if i2 - i1 < 2:
        raise ValueError(
            f"The {label} window is empty or too narrow (its start and end resolved to the "
            "same or adjacent sample). Re-drag it on the \"Pulse window(s)\" plot -- this can "
            "happen when a window value is left over from a different Correction type/setup "
            "or a previously-loaded dataset."
        )


def _index_of_nearest(time_pulse: list[float], target: float) -> int:
    """Index of the `time_pulse` sample closest to `target`.

    A nearest-sample search rather than an exact-match lookup, since
    `time_pulse` is not rounded to a fixed decimal precision before being
    stored: rounding to microsecond precision would silently collapse
    distinct samples together for any time step under ~1 microsecond --
    exactly what a "generate the time vector from a fixed time step" pulse
    window uses when the step is a fraction of a microsecond (a common
    2-5 MHz bar-gage sample rate). The caller already snaps xd1/xd2 to the
    nearest ACTUAL `time_pulse` sample before calling in here, so this is
    normally an exact (zero-distance) match anyway -- searching for the
    nearest one instead of demanding an exact hit is simply more robust,
    at negligible extra cost, and needs no rounding at all."""
    return min(range(len(time_pulse)), key=lambda i: abs(time_pulse[i] - target))


def _bandwidth_rolloff_weight(
    freq: np.ndarray, radius: float, c0_ref: float,
    max_normalized_freq: float, rolloff_fraction: float = 0.25,
) -> np.ndarray:
    """Smooth (raised-cosine) 0..1 weight applied to the CORRECTED spectrum
    itself -- not a filter on the raw signal -- to cap how far the
    Pochhammer-Chree amplitude/attenuation correction's own gain is allowed
    to run with frequency.

    Why this exists: `elastic_correction`'s `_corrected`'s `m2[i]/m1[i]`
    factor and `viscoelastic_correction`'s `exp(-(damping/cp)*z)`
    attenuation term are not bounded above by 1 -- at high frequency they
    routinely exceed 1, since they exist to UNDO the bar's own geometric
    dispersion, not to suppress anything. A real strain-gage signal always
    carries some broadband electronic/quantization noise floor -- negligible
    at the pulse's genuine (low) frequencies, but not literally zero at high
    ones. A correction gain above 1 up there amplifies exactly that noise
    floor rather than any genuine bar-transmitted signal, which is the
    well-known mechanism behind the high-frequency ringing bursts a
    Pochhammer-Chree correction can produce on real data -- this module's
    own `phase_velocity_ratio`/`freq_over_c` diagnostic (plotted as "Cp/C0"
    vs "f.a/C0" on the Corrections page) stays smooth and well-behaved
    across the same range, so it is not the root search itself failing;
    amplifying an ordinary sensor noise floor is inherent to this kind of
    correction, on any implementation, not a bug in the root solve.

    Returns an array the same length as `freq`: 1.0 below
    `(1 - rolloff_fraction) * max_normalized_freq * c0_ref / radius`, a
    raised-cosine taper down to 0.0 by `max_normalized_freq * c0_ref /
    radius`, and 0.0 beyond -- so the correction's own gain is smoothly
    capped rather than left unbounded, without introducing a NEW sharp
    (Gibbs-ringing-prone) discontinuity of its own to replace the old one.
    `max_normalized_freq <= 0` (or `None`) disables this entirely (returns
    an all-ones array, i.e. every frequency corrected exactly as before) --
    this is what the Corrections page's "Limit correction bandwidth"
    checkbox being unticked maps to.

    The default cutoff (0.6, i.e. `f*radius/c0_ref <= 0.6`) is a practical,
    round-number heuristic -- not derived from a first-principles bound for
    every possible bar material/geometry -- picked to sit comfortably
    beyond where a typical SHPB pulse's own genuine energy has already
    decayed to near nothing, while still being well inside the range this
    module's own Cp/C0 curve is normally inspected over. Like every other
    "recommended default" in this app, it is meant to be adjusted (via the
    Corrections page's own field) if a specific bar/pulse combination needs
    a tighter or looser bound; treat it as a starting point, not a proven
    optimum."""
    freq = np.asarray(freq, dtype=float)
    if max_normalized_freq is None or max_normalized_freq <= 0 or radius <= 0 or c0_ref <= 0:
        return np.ones_like(freq)
    f_cut = max_normalized_freq * c0_ref / radius
    f_start = max(0.0, 1.0 - rolloff_fraction) * f_cut
    w = np.ones_like(freq)
    if f_cut <= f_start:
        # Degenerate (rolloff_fraction ~ 0, or f_cut itself ~ 0): a hard
        # cutoff instead of a taper, rather than dividing by ~zero below.
        w[freq >= f_cut] = 0.0
        return w
    in_taper = (freq > f_start) & (freq < f_cut)
    w[in_taper] = 0.5 * (1.0 + np.cos(np.pi * (freq[in_taper] - f_start) / (f_cut - f_start)))
    w[freq >= f_cut] = 0.0
    return w


@dataclass
class DispersionResult:
    time: np.ndarray
    incident: Optional[np.ndarray] = None
    reflected: Optional[np.ndarray] = None
    transmitted: np.ndarray = None
    incident_corrected: Optional[np.ndarray] = None
    reflected_corrected: Optional[np.ndarray] = None
    transmitted_corrected: np.ndarray = None
    freq_over_c: dict = None       # {'incident': RRi, 'reflected': RRr, 'transmitted': RRt}
    phase_velocity_ratio: dict = None  # {'incident': CCi, ...}
    temperature_profile: Optional[tuple] = None  # (distance, temperature) if heat_check


def _bar_elastic_wave_speed(
    bar: Bar,
    heat_check: bool,
    type_bars_check: int,
    ambient_temp: float,
    t0: float,
    tm_value: tuple[float, float] | None,
    table,
    gage_temp: float,
) -> tuple[float, float, float, Optional[tuple]]:
    """Elastic wave speed / Poisson ratio for one bar, optionally corrected
    for a temperature gradient along the bar (heat_check).

    Returns (C0, nu, E, temperature_profile_or_None).
    """
    z = float(np.mean(bar.PG_pa))
    if not heat_check:
        return float(np.mean(bar.C0_pa)), float(np.mean(bar.nu_pa)), float(np.mean(bar.e_pa)), None

    h = 20
    zew = np.linspace(0, z, num=150)
    if type_bars_check in (1, 2):
        k_therm = 40 if type_bars_check == 1 else 220
        temp_profile = ambient_temp + (t0 - ambient_temp) * np.exp(
            -2 * np.sqrt(h / (k_therm * np.mean(bar.D_pa))) * zew
        )
        t_mean = float(np.mean(temp_profile))
        tm0, tm1 = tm_value
        rte = (1 - (t_mean / tm0) * math.exp(tm1 * (1 - (tm0 / t_mean)))) / (
            1 - (293 / tm0) * math.exp(tm1 * (1 - (tm0 / 293)))
        )
        e_mat = rte * float(np.mean(bar.e_pa))
        c0 = math.sqrt(e_mat / float(np.mean(bar.rho_pa)))
        nu = float(np.mean(bar.nu_pa))
        return c0, nu, e_mat, (zew, temp_profile)

    # type_bars_check == 3: use imported temperature-dependent property tables
    k_therm = table.value("lambda", gage_temp, default=130) if table else 130
    temp_profile = ambient_temp + (t0 - ambient_temp) * np.exp(
        -2 * np.sqrt(h / (k_therm * np.mean(bar.D_pa))) * zew
    )
    t_mean = float(np.mean(temp_profile))
    e_mat = table.value("e", t_mean, default=float(np.mean(bar.e_pa))) if table else float(np.mean(bar.e_pa))
    rho = table.value("rho", t_mean, default=float(np.mean(bar.rho_pa))) if table else float(np.mean(bar.rho_pa))
    nu = table.value("nu", t_mean, default=float(np.mean(bar.nu_pa))) if table else float(np.mean(bar.nu_pa))
    c0 = math.sqrt(e_mat / rho)
    return c0, nu, e_mat, (zew, temp_profile)


def _solve_pochhammer_branch(freq, radius, c0, nu, progress_cb: ProgressCB, label: str):
    """Solve the elastic Pochhammer-Chree dispersion relation frequency by
    frequency for one wave (incident/reflected/transmitted), returning the
    phase-velocity solutions and the M1/M2 correction factors."""
    beta = (1 - 2 * nu) / (1 - nu)
    cp_sol = [c0]
    m1 = [1.0]
    m2 = [1.0]

    cp = sym.symbols("Cp")
    x = ((cp / c0) ** 2) * (1 + nu)
    nl = 2 * math.pi * freq / cp
    h_expr = nl * sym.sqrt(beta * x - 1)
    k_expr = nl * sym.sqrt(2 * x - 1)

    total = len(freq) - 1
    for i in range(total):
        dsol = sym.nsolve(
            ((x - 1) ** 2) * radius * h_expr[i + 1] * (besselj(0, radius * h_expr[i + 1]) / besselj(1, radius * h_expr[i + 1]))
            - (beta * x - 1) * (x - radius * k_expr[i + 1] * (besselj(0, radius * k_expr[i + 1]) / besselj(1, radius * k_expr[i + 1]))),
            cp_sol[-1],
        )
        x_a = ((dsol / c0) ** 2) * (1 + nu)
        nl_a = 2 * math.pi * freq[i + 1] / dsol
        h_a = nl_a * sym.sqrt(beta * x_a - 1)
        k_a = nl_a * sym.sqrt(2 * x_a - 1)
        dm1 = (2 * (1 + (1 - beta * x_a) / (x_a - 1))) / (
            radius * h_a * (besselj(0, radius * h_a) / besselj(1, radius * h_a))
            + radius * k_a * (besselj(0, radius * k_a) / besselj(1, radius * k_a)) * (1 - beta * x_a) / (x_a - 1)
        )
        cp_sol.append(float(Abs(dsol)))
        m1.append(float(complex(dm1).real))
        m2.append(float((dsol / c0) ** 2))
        if progress_cb:
            progress_cb(i + 1, total)
    return cp_sol, m1, m2


def _pochhammer_branch_task(context, item):
    """core.parallel.run_map worker for `_solve_pochhammer_branches` below
    -- runs in a separate process, so `progress_cb` is always None here
    (a Qt-bound callback can't cross a process boundary); the caller
    reports branch-level progress instead, from the calling thread."""
    radius, c0, nu, label = item
    return _solve_pochhammer_branch(context["freq"], radius, c0, nu, None, label)


def _solve_pochhammer_branches(freq, branches: list[tuple], progress_cb: ProgressCB, n_workers: int = 1):
    """Solves several independent elastic Pochhammer-Chree branches (e.g.
    incident/reflected/transmitted), each via `_solve_pochhammer_branch`,
    returning their (cp_sol, m1, m2) results in the same order as
    `branches` (each a `(radius, c0, nu, label)` tuple).

    Each branch's own frequency-by-frequency solve is sequential
    internally -- every step seeds from the previous one (a continuation
    method) -- so this parallelizes only ACROSS the (up to 3) branches,
    never within one: genuine but bounded speedup (up to 3x on 3+ cores),
    not the N x of an embarrassingly-parallel loop.

    `n_workers <= 1` (the default) or a single branch: identical to the
    original sequential calls, with full per-frequency progress via
    `progress_cb`. `n_workers > 1`: runs the branches concurrently and
    reports coarser, whole-branch progress instead (per-frequency progress
    isn't available across a process boundary)."""
    if n_workers <= 1 or len(branches) <= 1:
        return [_solve_pochhammer_branch(freq, radius, c0, nu, progress_cb, label)
                for radius, c0, nu, label in branches]

    raw = run_map(
        _pochhammer_branch_task, branches, n_workers=n_workers,
        context={"freq": freq}, progress_callback=progress_cb,
    )
    out = []
    for _, value, err in raw:
        if err is not None:
            raise err
        out.append(value)
    return out


def _cut_two_bar_windows(ib: Bar, ob: Bar, specimen: Specimen, timet: np.ndarray, xd1: float, xd2: float, sg_unc_d: np.ndarray):
    """Shared window-cutting logic (incident/reflected on IB, transmitted on
    OB) used by both the elastic correction and the no-correction pass-through."""
    specimen_c0 = math.sqrt(np.mean(specimen.e_pa) / np.mean(specimen.rho_pa))
    idx1 = int(np.searchsorted(timet, xd1, side="right") - 1)
    idx2 = int(np.searchsorted(timet, xd2, side="right") - 1)
    _require_window(idx1, idx2, "incident")
    ti_pulse = idx2 - idx1

    sxd1 = xd1 + 2 * np.mean(ib.PG_pa) / np.mean(ib.C0_pa)
    idx_r1 = int(np.searchsorted(timet, sxd1, side="right") - 1)
    idx_r2 = idx_r1 + ti_pulse

    ssxd1 = (
        xd1 + np.mean(ob.PG_pa) / np.mean(ob.C0_pa) + np.mean(specimen.L_pa) / specimen_c0
        + np.mean(ib.PG_pa) / np.mean(ib.C0_pa)
    )
    idx_t1 = int(np.searchsorted(timet, ssxd1, side="right") - 1)
    idx_t2 = idx_t1 + ti_pulse

    time1 = timet[idx1:idx2] - xd1
    incident = ib.signal[idx1:idx2] * np.mean(sg_unc_d)
    reflected = ib.signal[idx_r1:idx_r2] * np.mean(sg_unc_d)
    transmitted = ob.signal[idx_t1:idx_t2] * np.mean(sg_unc_d)
    return time1, incident, reflected, transmitted


def no_correction(
    ib: Bar, ob: Bar, specimen: Specimen, timet: np.ndarray,
    xd1: float, xd2: float, sg_unc_d: np.ndarray,
    progress_cb: ProgressCB = None,
) -> DispersionResult:
    """Cut the incident/reflected/transmitted windows without applying any
    Pochhammer-Chree dispersion correction -- for setups (short bars, low
    frequency content...) where the correction isn't wanted. "Corrected"
    signals are identical to the raw cut ones; phase-velocity curves are
    omitted since there's nothing to plot there."""
    time1, incident, reflected, transmitted = _cut_two_bar_windows(ib, ob, specimen, timet, xd1, xd2, sg_unc_d)
    if progress_cb:
        progress_cb(1, 1)
    return DispersionResult(
        time=time1, incident=incident, reflected=reflected, transmitted=transmitted,
        incident_corrected=incident, reflected_corrected=reflected, transmitted_corrected=transmitted,
        freq_over_c={}, phase_velocity_ratio={},
    )


def no_correction_direct_impact(
    ob: Bar, specimen: Specimen, xd1: float, xd2: float,
    sg_unc_d: np.ndarray, time_pulse: list[float],
    progress_cb: ProgressCB = None,
) -> DispersionResult:
    """Direct-Impact equivalent of `no_correction`: cut the transmitted
    window without applying any dispersion correction."""
    i1 = _index_of_nearest(time_pulse, xd1)
    i2 = _index_of_nearest(time_pulse, xd2)
    _require_window(i1, i2, "transmitted")
    time1 = np.asarray(time_pulse[i1:i2]) - xd1
    transmitted = ob.signal[i1:i2] * np.mean(sg_unc_d)
    if progress_cb:
        progress_cb(1, 1)
    return DispersionResult(
        time=time1, transmitted=transmitted, transmitted_corrected=transmitted,
        freq_over_c={}, phase_velocity_ratio={},
    )


def elastic_correction(
    ib: Bar, ob: Bar, specimen: Specimen, timet: np.ndarray,
    xd1: float, xd2: float, sg_unc_d: np.ndarray,
    plat_check: bool = False, plat_emod: np.ndarray | None = None,
    plat_rho: np.ndarray | None = None, plat_len: np.ndarray | None = None,
    heat_check: bool = False, type_bars_check: int = 1,
    ambient_temp: float = 293.0, t0: float = 293.0,
    tm_value: tuple[float, float] | None = None,
    gage_temp_ib: float = 293.0, gage_temp_ob: float = 293.0,
    table_ib=None, table_ob=None,
    progress_cb: ProgressCB = None,
    n_workers: int = 1,
    max_normalized_freq: float = 0.6,
) -> DispersionResult:
    """Standard two-bar (SHPB) elastic dispersion correction.
    Ported from FFT_dispersion.py: corr_disp(). `n_workers > 1` solves the
    incident/reflected/transmitted branches concurrently (see
    `_solve_pochhammer_branches`). `max_normalized_freq` caps the
    correction's own frequency-dependent gain above `f*radius/c0 >
    max_normalized_freq` (per branch, its own radius/c0) -- see
    `_bandwidth_rolloff_weight`'s docstring for why this is needed to avoid
    amplifying a real signal's high-frequency noise floor into visible
    ringing; `<= 0` disables it (unbounded gain, the original ported
    behavior)."""
    specimen_c0 = math.sqrt(np.mean(specimen.e_pa) / np.mean(specimen.rho_pa))
    plat_c0 = math.sqrt(np.mean(plat_emod) / np.mean(plat_rho)) if plat_check else 0.0

    idx1 = int(np.searchsorted(timet, xd1, side="right") - 1)
    idx2 = int(np.searchsorted(timet, xd2, side="right") - 1)
    _require_window(idx1, idx2, "incident")
    ti_pulse = idx2 - idx1

    sxd1 = xd1 + 2 * np.mean(ib.PG_pa) / np.mean(ib.C0_pa)
    if plat_check:
        sxd1 += 2 * np.mean(plat_len) / plat_c0
    idx_r1 = int(np.searchsorted(timet, sxd1, side="right") - 1)
    idx_r2 = idx_r1 + ti_pulse

    ssxd1 = (
        xd1 + np.mean(ob.PG_pa) / np.mean(ob.C0_pa) + np.mean(specimen.L_pa) / specimen_c0
        + np.mean(ib.PG_pa) / np.mean(ib.C0_pa)
    )
    if plat_check:
        ssxd1 += 2 * np.mean(plat_len) / plat_c0
    idx_t1 = int(np.searchsorted(timet, ssxd1, side="right") - 1)
    idx_t2 = idx_t1 + ti_pulse

    time1 = timet[idx1:idx2] - xd1
    incident = ib.signal[idx1:idx2] * np.mean(sg_unc_d)
    reflected = ib.signal[idx_r1:idx_r2] * np.mean(sg_unc_d)
    transmitted = ob.signal[idx_t1:idx_t2] * np.mean(sg_unc_d)

    dt = time1[1] - time1[0]
    n = len(incident)
    yi, yr, yt = np.fft.rfft(incident), np.fft.rfft(reflected), np.fft.rfft(transmitted)
    freq = np.fft.rfftfreq(n, d=dt)
    ai, ar, at = np.abs(yi), np.abs(yr), np.abs(yt)
    pi, pr, pt = np.angle(yi), np.angle(yr), np.angle(yt)

    z1, z2 = float(np.mean(ib.PG_pa)), float(np.mean(ob.PG_pa))
    c01, nu1, _, temp1 = _bar_elastic_wave_speed(ib, heat_check, type_bars_check, ambient_temp, t0, tm_value, table_ib, gage_temp_ib)
    c02, nu2, _, temp2 = _bar_elastic_wave_speed(ob, heat_check, type_bars_check, ambient_temp, t0, tm_value, table_ob, gage_temp_ob)

    a1, a2 = float(np.mean(ib.D_pa)) / 2, float(np.mean(ob.D_pa)) / 2

    (cp_i, m1i, m2i), (cp_r, m1r, m2r), (cp_t, m1t, m2t) = _solve_pochhammer_branches(
        freq,
        [(a1, c01, nu1, "incident"), (a1, c01, nu1, "reflected"), (a2, c02, nu2, "transmitted")],
        progress_cb, n_workers,
    )

    def _corrected(amp, phase, freq, z, c0, cp_sol, m1, m2, radius):
        n_phase = [phase[i] - (2 * cmath.pi * freq[i] * z / c0) * (c0 / cp_sol[i] - 1) for i in range(len(phase))]
        fft_new = np.array([(m2[i] / m1[i]) * amp[i] * np.exp(1.0j * n_phase[i]) for i in range(len(n_phase))])
        # Cap the correction's own gain above max_normalized_freq -- see
        # _bandwidth_rolloff_weight's docstring. All-ones (no-op) when
        # max_normalized_freq <= 0.
        fft_new = fft_new * _bandwidth_rolloff_weight(np.asarray(freq), radius, c0, max_normalized_freq)
        # `n=n` (the window's own length, from the enclosing scope): numpy's
        # irfft cannot infer an odd original length from the spectrum alone
        # and silently defaults to assuming even length, which for an
        # odd-length window (routine here -- windows are user-dragged, not
        # padded to a fixed size) reconstructs a signal that is not merely
        # one sample short but numerically wrong throughout. See
        # core/visco_calibration.py's `_predict_pulse` for the same fix
        # with a worked numerical example in the comment there.
        sig = np.fft.irfft(fft_new, n=n)
        return sig

    new_incident = -np.absolute(_corrected(ai, pi, freq, z1, c01, cp_i, m1i, m2i, a1))
    new_reflected = np.absolute(_corrected(ar, pr, freq, -z1, c01, cp_r, m1r, m2r, a1))
    new_transmitted = -np.absolute(_corrected(at, pt, freq, -z2, c02, cp_t, m1t, m2t, a2))

    if len(new_incident) < len(incident):
        new_incident = np.concatenate((new_incident, [new_incident[-1]]))
    if len(new_reflected) < len(reflected):
        new_reflected = np.concatenate((new_reflected, [new_reflected[-1]]))
    if len(new_transmitted) < len(transmitted):
        new_transmitted = np.concatenate((new_transmitted, [new_transmitted[-1]]))

    return DispersionResult(
        time=time1, incident=incident, reflected=reflected, transmitted=transmitted,
        incident_corrected=new_incident, reflected_corrected=new_reflected, transmitted_corrected=new_transmitted,
        freq_over_c={"incident": [f * a1 / c01 for f in freq], "reflected": [f * a1 / c01 for f in freq], "transmitted": [f * a2 / c02 for f in freq]},
        phase_velocity_ratio={"incident": [c / c01 for c in cp_i], "reflected": [c / c01 for c in cp_r], "transmitted": [c / c02 for c in cp_t]},
        temperature_profile=(temp1, temp2) if heat_check else None,
    )


def elastic_correction_direct_impact(
    ob: Bar, specimen: Specimen, xd1: float, xd2: float,
    sg_unc_d: np.ndarray, time_pulse: list[float],
    heat_check: bool = False, type_bars_check: int = 1,
    ambient_temp: float = 293.0, t0: float = 293.0,
    tm_value: tuple[float, float] | None = None,
    gage_temp_ob: float = 293.0, table_ob=None,
    progress_cb: ProgressCB = None,
    n_workers: int = 1,
    max_normalized_freq: float = 0.6,
) -> DispersionResult:
    """Single output-bar (Direct Impact) elastic dispersion correction.
    Ported from FFT_dispersion_ID.py: corr_disp(). Only one wave branch
    (transmitted) exists here -- unlike the two-bar `elastic_correction`,
    there's nothing to run concurrently, so `n_workers` is accepted (for a
    uniform call signature across all four correction functions) but has
    no effect. `max_normalized_freq` -- see `elastic_correction`'s
    docstring and `_bandwidth_rolloff_weight`."""
    i1 = _index_of_nearest(time_pulse, xd1)
    i2 = _index_of_nearest(time_pulse, xd2)
    _require_window(i1, i2, "transmitted")
    time1 = np.asarray(time_pulse[i1:i2]) - xd1
    transmitted = ob.signal[i1:i2] * np.mean(sg_unc_d)

    dt = time1[1] - time1[0]
    n = len(transmitted)
    yt = np.fft.rfft(transmitted)
    freq = np.fft.rfftfreq(n, d=dt)
    at, pt = np.abs(yt), np.angle(yt)

    z2 = float(np.mean(ob.PG_pa))
    c02, nu2, _, temp2 = _bar_elastic_wave_speed(ob, heat_check, type_bars_check, ambient_temp, t0, tm_value, table_ob, gage_temp_ob)
    a2 = float(np.mean(ob.D_pa)) / 2

    cp_t, m1t, m2t = _solve_pochhammer_branch(freq, a2, c02, nu2, progress_cb, "transmitted")
    n_phase = [pt[i] - (2 * cmath.pi * freq[i] * (-z2) / c02) * (c02 / cp_t[i] - 1) for i in range(len(pt))]
    fft_new = np.array([(m2t[i] / m1t[i]) * at[i] * np.exp(1.0j * n_phase[i]) for i in range(len(n_phase))])
    fft_new = fft_new * _bandwidth_rolloff_weight(freq, a2, c02, max_normalized_freq)  # see elastic_correction
    new_transmitted = -np.absolute(np.fft.irfft(fft_new, n=n))  # see elastic_correction's `_corrected` for why n=n

    return DispersionResult(
        time=time1, transmitted=transmitted, transmitted_corrected=new_transmitted,
        freq_over_c={"transmitted": [f * a2 / c02 for f in freq]},
        phase_velocity_ratio={"transmitted": [c / c02 for c in cp_t]},
        temperature_profile=(temp2,) if heat_check else None,
    )


def _solve_visco_branch(freq, e_branch, c_branch, radius: float, rho: float, nu: float, beta: float,
                         progress_cb: ProgressCB = None):
    """Per-frequency Pochhammer-Chree root search for one wave branch
    (incident/reflected/transmitted) of the viscoelastic correction --
    shared by the two-bar and direct-impact variants below.

    `e_branch` is the model's COMPLEX modulus per frequency bin, used for
    mu/lam (the actual physics being solved, same role as
    core/visco_calibration.py's `_complex_modulus_and_speed`). `c_branch`
    is the wave speed built from the modulus's MAGNITUDE only
    (`sqrt(abs(E)/rho)`), used solely to seed `nsolve`'s initial guess -- a
    numerical root solver converges more reliably from a real-valued
    starting point than a complex one, and the seed's exact value does not
    bias which root is found, only how quickly.

    NOTE: mu/lam must be derived from `e_branch` directly, not from
    `c_branch**2 * rho`. Since c_branch is built from abs(E), that would
    substitute the real-only magnitude of the modulus for its actual
    complex value everywhere in the root search except the initial guess --
    discarding the loss modulus (the material's damping/attenuation
    behavior) from the very physics this function exists to solve.
    `e_branch` is passed in separately from `c_branch` specifically so the
    two roles stay distinct."""
    b = sym.symbols("B")
    b_sol = [0]
    k_sol = [0, (2 * cmath.pi * freq[1]) / c_branch[1]]
    atn = [0]
    cp_sol = [c_branch[0]]
    cp = sym.symbols("Cp")
    total = len(freq) - 1
    for i in range(total):
        mu = e_branch[i + 1] / (2 * (1 + nu))
        lam = nu * e_branch[i + 1] / ((1 + nu) * (1 - 2 * nu))
        p = sym.sqrt((rho * (2 * cmath.pi * freq[i + 1]) ** 2) / (lam + 2 * mu) - b**2)
        q = sym.sqrt((rho * (2 * cmath.pi * freq[i + 1]) ** 2) / mu - b**2)
        try:
            dsol = sym.nsolve(
                (2 * p / radius) * (q**2 + b**2) * besselj(1, p * radius) * besselj(1, q * radius)
                - (q**2 - b**2) ** 2 * besselj(0, p * radius) * besselj(1, q * radius)
                - 4 * (b**2) * p * q * besselj(0, q * radius) * besselj(1, p * radius),
                k_sol[-1],
            )
            b_sol.append(dsol)
            k_sol.append(float(complex(dsol).real))
            cp_sol.append(2 * math.pi * freq[i + 1] / k_sol[-1])
            atn.append(float(abs(complex(dsol).imag)))
        except ValueError:
            x = ((cp / c_branch[i + 1]) ** 2) * (1 + nu)
            nl = 2 * cmath.pi * freq[i + 1] / cp
            h_e = nl * sym.sqrt(beta * x - 1)
            k_e = nl * sym.sqrt(2 * x - 1)
            dsol = sym.nsolve(
                ((x - 1) ** 2) * radius * h_e * (besselj(0, radius * h_e) / besselj(1, radius * h_e))
                - (beta * x - 1) * (x - radius * k_e * (besselj(0, radius * k_e) / besselj(1, radius * k_e))),
                cp_sol[-1],
            )
            cp_sol.append(float(complex(dsol).real))
            k_sol.append(2 * math.pi * freq[i + 1] / cp_sol[-1])
            atn.append(atn[-1])
            b_sol.append(k_sol[-1] - atn[-1] * 1j)
        if progress_cb:
            progress_cb(i + 1, total)
    return cp_sol, atn


def _visco_branch_task(context, item):
    """core.parallel.run_map worker for `_solve_visco_branches` below --
    runs in a separate process, so `progress_cb` is always None here (a
    Qt-bound callback can't cross a process boundary); the caller reports
    branch-level progress instead, from the calling thread."""
    e_branch, c_branch, radius, rho, nu, beta = item
    return _solve_visco_branch(context["freq"], e_branch, c_branch, radius, rho, nu, beta, None)


def _solve_visco_branches(freq, branches: list[tuple], progress_cb: ProgressCB, n_workers: int = 1):
    """Viscoelastic counterpart of `_solve_pochhammer_branches`: solves
    several independent branches (e.g. incident/reflected/transmitted),
    each via `_solve_visco_branch`, returning their (cp_sol, atn) results
    in the same order as `branches` (each a `(e_branch, c_branch, radius,
    rho, nu, beta)` tuple). See `_solve_pochhammer_branches`'s docstring
    for why this is bounded (up to 3x) parallelism across branches, never
    within one, and why parallel progress is coarser (whole branches, not
    per-frequency)."""
    if n_workers <= 1 or len(branches) <= 1:
        return [_solve_visco_branch(freq, *branch, progress_cb) for branch in branches]

    raw = run_map(
        _visco_branch_task, branches, n_workers=n_workers,
        context={"freq": freq}, progress_callback=progress_cb,
    )
    out = []
    for _, value, err in raw:
        if err is not None:
            raise err
        out.append(value)
    return out


def _visco_phase_correction(phase, freq, z, cp_sol):
    """The dispersion phase-correction term: reference every frequency's
    solved phase velocity `cp_sol[i]` back to the branch's own static/DC
    value `cp_sol[0]` -- the same structure as `elastic_correction`'s
    `_corrected` (`c0/cp_sol[i] - 1`, with `c0` fixed) and
    core/visco_calibration.py's `_predict_pulse`
    (`c[0]/Abs(cp_sol[a]) - 1`, with `c[0]` fixed).

    `c_ref` must be the branch's fixed DC value (`cp_sol[0]`), not the
    per-frequency `cp_sol[i]` itself: using `cp_sol[i]` in both places
    would make the correction identically `1 - 1 = 0` for every frequency
    bin regardless of material or model, silently turning the phase
    correction into a no-op. Referencing every frequency back to the same
    fixed speed is what makes this a genuine dispersion correction,
    matching the structurally identical formulas in `elastic_correction`'s
    `_corrected` and core/visco_calibration.py's `_predict_pulse`."""
    c_ref = cp_sol[0]
    return [phase[i] - (2 * cmath.pi * freq[i] * z / c_ref) * (c_ref / cp_sol[i] - 1) for i in range(len(phase))]


def _representative_frequency(window_signal: np.ndarray, dt: float) -> float:
    """Energy-weighted centroid frequency (Hz) of a windowed pulse's
    amplitude spectrum, excluding the DC bin -- used by
    `estimate_viscoelastic_windows` to pick a single frequency at which to
    evaluate a viscoelastic bar's phase velocity for auto-placing the
    reflected/transmitted windows. A centroid (rather than e.g. just the
    single loudest bin) is used because a real SHPB pulse's energy is
    spread over a band, not concentrated at one frequency, and the
    centroid is the standard "typical propagation frequency" estimate for
    a band-limited pulse."""
    n = len(window_signal)
    if n < 4 or dt <= 0:
        return 0.0
    spectrum = np.fft.rfft(window_signal)
    freq = np.fft.rfftfreq(n, d=dt)
    power = np.abs(spectrum) ** 2
    if len(power) > 1:
        power = power.copy()
        power[0] = 0.0  # DC carries no propagation-timing information
    total = float(power.sum())
    if total <= 0:
        return 0.0
    return float(np.sum(freq * power) / total)


def _visco_phase_velocity_at(
    freq_hz: float, model: RheologyModel, params: dict, radius: float, rho: float, nu: float,
    progress_cb: ProgressCB = None,
) -> float:
    """Solves the viscoelastic Pochhammer-Chree relation at a single
    representative frequency (`_solve_visco_branch` always also needs a
    leading DC bin to seed from) -- far cheaper than the full per-bin
    correction, since placing a window only needs one ballpark phase
    velocity, not the exact per-bin correction the run itself computes."""
    beta = (1 - 2 * nu) / (1 - nu)
    freq = np.array([0.0, max(float(freq_hz), 1.0)])
    e_branch = model.complex_modulus(freq, params)
    c_branch = [cmath.sqrt(abs(e) / rho) for e in e_branch]
    cp_sol, _ = _solve_visco_branch(freq, e_branch, c_branch, radius, rho, nu, beta, progress_cb)
    return float(cp_sol[1])


def estimate_viscoelastic_windows(
    ib: Bar, ob: Bar, specimen: Specimen, timet: np.ndarray,
    xd1: float, xd2: float, nu: float,
    model: RheologyModel, params: dict,
    progress_cb: ProgressCB = None,
) -> tuple[float, float, float, float]:
    """Auto-places the reflected/transmitted pulse windows for the
    viscoelastic two-bar correction FROM the (user-drawn) incident window
    alone, the same way `_cut_two_bar_windows`/`elastic_correction`
    already derive their `sxd1`/`ssxd1` from `xd1` for the elastic case --
    by adding the wave's travel time over each leg of its path (gauge ->
    interface, through the specimen, interface -> gauge) to the incident
    window's own start.

    The elastic version builds that travel time from a single, constant
    wave speed C0 = sqrt(E/rho). That is a poor approximation for a
    viscoelastic bar, whose wave speed can vary strongly with excitation
    frequency (see this module's docstring and `_solve_visco_branch`) --
    using it would place the reflected/transmitted windows in roughly the
    wrong spot, which is exactly the "arbitrary" manual cut this function
    is meant to replace. Instead of C0, this evaluates the *model's*
    actual phase velocity (a real per-frequency Pochhammer-Chree solve,
    not a non-dispersive guess) at the incident pulse's own energy-
    weighted dominant frequency -- see `_representative_frequency` --
    which is enough to place a window well without paying for a full
    per-bin solve just to do so.

    Returns (xd3, xd4, xd5, xd6): a drop-in replacement for the reflected/
    transmitted windows a user would otherwise drag by hand. The caller
    should still let the user drag any of them afterward to override this
    estimate -- it is a (much better informed) starting point, not a
    guaranteed-correct cut, since a single representative frequency still
    cannot capture a genuinely broadband pulse's full dispersion."""
    idx1 = int(np.searchsorted(timet, xd1, side="right") - 1)
    idx2 = int(np.searchsorted(timet, xd2, side="right") - 1)
    incident = np.asarray(ib.signal[idx1:idx2], dtype=float)
    if len(incident) < 4:
        raise ValueError("Incident window is too short to estimate a representative frequency from -- widen it first.")
    dt = float(np.median(np.diff(timet)))
    f_rep = _representative_frequency(incident, dt)

    a1, a2 = float(np.mean(ib.D_pa)) / 2, float(np.mean(ob.D_pa)) / 2
    rho1, rho2 = float(np.mean(ib.rho_pa)), float(np.mean(ob.rho_pa))
    z1, z2 = float(np.mean(ib.PG_pa)), float(np.mean(ob.PG_pa))
    specimen_c0 = math.sqrt(np.mean(specimen.e_pa) / np.mean(specimen.rho_pa))

    c1 = _visco_phase_velocity_at(f_rep, model, params, a1, rho1, nu, progress_cb)
    c2 = _visco_phase_velocity_at(f_rep, model, params, a2, rho2, nu, progress_cb)

    width = xd2 - xd1
    xd3 = xd1 + 2 * z1 / c1
    xd4 = xd3 + width
    xd5 = xd1 + z2 / c2 + np.mean(specimen.L_pa) / specimen_c0 + z1 / c1
    xd6 = xd5 + width

    return xd3, xd4, xd5, xd6


def viscoelastic_correction(
    ib: Bar, ob: Bar, specimen: Specimen, timet: np.ndarray,
    xd1: float, xd2: float, xd3: float, xd4: float, xd5: float, xd6: float,
    sg_unc_d: np.ndarray, nu: float, damping: float,
    model: RheologyModel, params: dict,
    progress_cb: ProgressCB = None,
    n_workers: int = 1,
    max_normalized_freq: float = 0.6,
) -> DispersionResult:
    """Viscoelastic two-bar dispersion correction, generic over which
    core.rheology_models.RheologyModel is selected (originally hard-coded
    to the Standard Linear Solid -- see that module's docstring for what
    else is available and why). Ported from FFT_dispersion_visco.py:
    corr_disp() -- see `_solve_visco_branch` and `_visco_phase_correction`'s
    docstrings for the physics each one is responsible for getting right.
    `max_normalized_freq` -- see `elastic_correction`'s docstring and
    `_bandwidth_rolloff_weight`; applied here too (the user explicitly
    asked for the same high-frequency-ringing safeguard in the
    viscoelastic case, not just the elastic one), using each branch's own
    DC (zero-frequency) modulus-derived speed `abs(c_*[0])` as the
    reference speed, since a viscoelastic bar's wave speed is itself
    frequency-dependent -- there is no single fixed `c0` here the way the
    elastic case has one."""
    idx = {}
    for label, t in zip(("i1", "i2", "i3", "i4", "i5", "i6"), (xd1, xd2, xd3, xd4, xd5, xd6)):
        idx[label] = int(np.searchsorted(timet, t, side="right") - 1)
    _require_window(idx["i1"], idx["i2"], "incident")
    nref = idx["i2"] - idx["i1"]
    idx["i4"] = idx["i3"] + nref
    idx["i6"] = idx["i5"] + nref

    time1 = timet[idx["i1"]:idx["i2"]] - xd1
    incident = ib.signal[idx["i1"]:idx["i2"]] * np.mean(sg_unc_d)
    reflected = ib.signal[idx["i3"]:idx["i4"]] * np.mean(sg_unc_d)
    transmitted = ob.signal[idx["i5"]:idx["i6"]] * np.mean(sg_unc_d)

    dt = time1[1] - time1[0]
    n = len(incident)
    yi, yr, yt = np.fft.rfft(incident), np.fft.rfft(reflected), np.fft.rfft(transmitted)
    freq = np.fft.rfftfreq(n, d=dt)
    ai, ar, at = np.abs(yi), np.abs(yr), np.abs(yt)
    pi, pr, pt = np.angle(yi), np.angle(yr), np.angle(yt)

    z1, z2 = float(np.mean(ib.PG_pa)), float(np.mean(ob.PG_pa))

    # The *wave speed* c_i/c_r/c_t is built from the *magnitude* of the
    # complex modulus (abs(E)), used only to seed nsolve's initial guess
    # inside `_solve_visco_branch`; mu/lam (the actual physics being
    # solved) use the complex modulus directly -- see that function's
    # docstring for why these are deliberately different quantities.
    e_i = model.complex_modulus(freq, params)
    e_r = model.complex_modulus(freq, params)
    e_t = model.complex_modulus(freq, params)
    c_i = [cmath.sqrt(abs(e) / np.mean(ib.rho_pa)) for e in e_i]
    c_r = [cmath.sqrt(abs(e) / np.mean(ib.rho_pa)) for e in e_r]
    c_t = [cmath.sqrt(abs(e) / np.mean(ob.rho_pa)) for e in e_t]

    a1, a2 = float(np.mean(ib.D_pa)) / 2, float(np.mean(ob.D_pa)) / 2
    beta = (1 - 2 * nu) / (1 - nu)

    (cp_i, atn_i), (cp_r, atn_r), (cp_t, atn_t) = _solve_visco_branches(
        freq,
        [(e_i, c_i, a1, np.mean(ib.rho_pa), nu, beta),
         (e_r, c_r, a1, np.mean(ib.rho_pa), nu, beta),
         (e_t, c_t, a2, np.mean(ob.rho_pa), nu, beta)],
        progress_cb, n_workers,
    )

    n_pi = _visco_phase_correction(pi, freq, z1, cp_i)
    n_pr = _visco_phase_correction(pr, freq, -z1, cp_r)
    n_pt = _visco_phase_correction(pt, freq, -z2, cp_t)

    # `n=n`: see core/visco_calibration.py's `_predict_pulse` for why this
    # is required (not optional) for a user-dragged, possibly odd-length
    # window -- numpy's irfft cannot infer an odd original length from the
    # spectrum alone and silently reconstructs a numerically wrong signal
    # without it, not merely a one-sample-short one.
    w_i = _bandwidth_rolloff_weight(freq, a1, abs(c_i[0]), max_normalized_freq)
    w_r = _bandwidth_rolloff_weight(freq, a1, abs(c_r[0]), max_normalized_freq)
    w_t = _bandwidth_rolloff_weight(freq, a2, abs(c_t[0]), max_normalized_freq)
    new_i = np.fft.irfft(np.array([np.exp(-(damping / cp_i[i]) * z1) * cmath.exp(-atn_i[i] * z1) * ai[i] * cmath.rect(1, n_pi[i]) for i in range(len(n_pi))]) * w_i, n=n).real
    new_r = np.fft.irfft(np.array([np.exp(-(damping / cp_r[i]) * (-z1)) * cmath.exp(-atn_r[i] * (-z1)) * ar[i] * cmath.rect(1, n_pr[i]) for i in range(len(n_pr))]) * w_r, n=n).real
    new_t = np.fft.irfft(np.array([np.exp(-(damping / cp_t[i]) * (-z2)) * cmath.exp(-atn_t[i] * (-z2)) * at[i] * cmath.rect(1, n_pt[i]) for i in range(len(n_pt))]) * w_t, n=n).real

    return DispersionResult(
        time=time1, incident=incident, reflected=reflected, transmitted=transmitted,
        incident_corrected=new_i, reflected_corrected=new_r, transmitted_corrected=new_t,
        freq_over_c={"incident": list(freq), "reflected": list(freq), "transmitted": list(freq)},
        phase_velocity_ratio={"incident": cp_i, "reflected": cp_r, "transmitted": cp_t},
    )


def viscoelastic_correction_direct_impact(
    ob: Bar, specimen: Specimen, xd1: float, xd2: float,
    sg_unc_d: np.ndarray, time_pulse: list[float],
    nu: float, damping: float,
    model: RheologyModel, params: dict,
    progress_cb: ProgressCB = None,
    n_workers: int = 1,
    max_normalized_freq: float = 0.6,
) -> DispersionResult:
    """Viscoelastic single output-bar (Direct Impact) dispersion
    correction -- Direct-Impact counterpart of `viscoelastic_correction`,
    sharing its `_solve_visco_branch`/`_visco_phase_correction`. Ported
    from FFT_dispersion_visco_ID.py: corr_disp(). Only one wave branch
    (transmitted) exists here -- unlike `viscoelastic_correction`,
    there's nothing to run concurrently, so (as with
    `elastic_correction_direct_impact`) `n_workers` is accepted for a
    uniform call signature but has no effect. `max_normalized_freq` -- see
    `viscoelastic_correction`'s docstring and `_bandwidth_rolloff_weight`."""
    i1 = _index_of_nearest(time_pulse, xd1)
    i2 = _index_of_nearest(time_pulse, xd2)
    _require_window(i1, i2, "transmitted")
    time1 = np.asarray(time_pulse[i1:i2]) - xd1
    transmitted = ob.signal[i1:i2] * np.mean(sg_unc_d)

    dt = time1[1] - time1[0]
    n = len(transmitted)
    yt = np.fft.rfft(transmitted)
    freq = np.fft.rfftfreq(n, d=dt)
    at, pt = np.abs(yt), np.angle(yt)

    z2 = float(np.mean(ob.PG_pa))
    e_t = model.complex_modulus(freq, params)
    c_t = [cmath.sqrt(abs(e) / np.mean(ob.rho_pa)) for e in e_t]
    a2 = float(np.mean(ob.D_pa)) / 2
    beta = (1 - 2 * nu) / (1 - nu)

    cp_t, atn_t = _solve_visco_branch(freq, e_t, c_t, a2, np.mean(ob.rho_pa), nu, beta, progress_cb)
    n_pt = _visco_phase_correction(pt, freq, -z2, cp_t)
    # NOTE: the damping term below deliberately uses the *seed* wave speed
    # c_t[i] (abs(E)-based), not the solved cp_t[i] -- matching the
    # original FFT_dispersion_visco_ID.py exactly. This differs from the
    # two-bar `viscoelastic_correction` above, which uses cp_t[i] there;
    # there isn't clear evidence which convention is the intended one, so
    # it is left exactly as ported rather than guessed at.
    w_t = _bandwidth_rolloff_weight(freq, a2, abs(c_t[0]), max_normalized_freq)
    new_t = -np.absolute(
        np.fft.irfft(np.array([
            np.exp(-(damping / c_t[i]) * (-z2)) * at[i] * cmath.exp(-atn_t[i] * (-z2)) * cmath.rect(1, n_pt[i])
            for i in range(len(n_pt))
        ]) * w_t, n=n)  # see viscoelastic_correction for why n=n
    )

    return DispersionResult(
        time=time1, transmitted=transmitted, transmitted_corrected=new_t,
        freq_over_c={"transmitted": [f * a2 / np.mean([abs(c) for c in c_t]) for f in freq]},
        phase_velocity_ratio={"transmitted": [cp_t[i] / abs(c_t[i]) for i in range(len(cp_t))]},
    )


def dispersion_result_to_dict(result: "DispersionResult") -> dict:
    """Converts a DispersionResult's array fields to plain lists, for JSON
    export (session save, Setup Profile, "Export full test")."""
    def _list(x):
        return list(x) if x is not None else None

    return {
        "time": _list(result.time),
        "incident": _list(result.incident),
        "reflected": _list(result.reflected),
        "transmitted": _list(result.transmitted),
        "incident_corrected": _list(result.incident_corrected),
        "reflected_corrected": _list(result.reflected_corrected),
        "transmitted_corrected": _list(result.transmitted_corrected),
        # freq_over_c/phase_velocity_ratio/temperature_profile are
        # intermediate diagnostics, not needed to USE the corrected
        # signal downstream (Processing tab only reads the *_corrected
        # arrays and .time) -- skipped to keep the exported JSON smaller.
    }


def dispersion_result_from_dict(data: dict) -> "DispersionResult":
    """Reconstructs a DispersionResult from `dispersion_result_to_dict`'s
    output, so it behaves exactly like a freshly-computed one wherever
    it's read (e.g. Processing tab's "Use dispersion-corrected signal")."""
    def _arr(x):
        return np.asarray(x) if x is not None else None

    return DispersionResult(
        time=_arr(data.get("time")),
        incident=_arr(data.get("incident")),
        reflected=_arr(data.get("reflected")),
        transmitted=_arr(data.get("transmitted")),
        incident_corrected=_arr(data.get("incident_corrected")),
        reflected_corrected=_arr(data.get("reflected_corrected")),
        transmitted_corrected=_arr(data.get("transmitted_corrected")),
    )
