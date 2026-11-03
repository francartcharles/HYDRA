"""
SHPB (Split Hopkinson Pressure Bar) processing engine.

Ported from process_SHPB.py: process_SHPB(). This is the single most
complex module in the original application: per Monte-Carlo draw, it
applies (in order) an optional bar-temperature-gradient correction, an
optional platen (WC platen) transmission correction, converts the cut
incident/reflected/transmitted waves into force/velocity/displacement at
each bar-specimen interface, then applies a punch (indentation), friction
and inertia correction to the resulting stress-strain curve, for each of
the four test kinematics (compression, tension, notched
Shear-Compression/Tension, and shear).

Ported as literally as possible (same variable names, same branch order,
same formulas) specifically to minimize transcription risk in code this
dense -- readability was sacrificed for traceability against the original
on purpose. Removed: the `psutil` memory guard tied to a Windows
messagebox `quit()`, the Tkinter progress-bar update, and the dependence
on `corr_file`/`import_bar_tab` module globals (now explicit parameters).

NOT YET independently re-validated against real experimental data or the
original GUI's output for the full matrix of options (heat x platen x
punch x friction x inertia x viscoelastic bars x 4 kinematics) -- only
exercised with synthetic signals on the simplest option combination. Cross
-check important results against the original application before relying
on this for a real analysis, especially with several corrections enabled
at once.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.integrate as integrate

from core.models import Bar, Specimen


@dataclass
class SHPBOptions:
    type_check: int                 # 1 compression, 2 tension, 3 notched (Shear-Compression/Tension), 4 shear
    comp_check: int = 0              # 1 cylindrical, 2 square (type_check in (1,3))
    tens_check: int = 0              # 1 axisymmetric, 2 flat (type_check == 2)
    shear_check: int = 0             # 1 hat, 2 double, 3 simple (type_check == 4)
    tenscomp_check: int = 1          # sign convention for type_check in (3, 4): 1 compression-like, 2 tension-like
    force_check: int = 1             # 1: use F_input, 2: use F_output, 3: use F_average = (F_input+F_output)/2
    heat_check: bool = False
    type_bars_check: int = 1         # 1/2: analytic RTE model, 3: imported material tables
    plat_check: bool = False
    punch_check: bool = False
    inertia_check: bool = False
    disp_visco: bool = False         # bars are viscoelastic; use IB/OB.e11/e22/n22
    disp_ela: bool = False           # use pre-dispersion-corrected waves supplied in `AA`
    sk_check: bool = False


@dataclass
class SHPBResult:
    force_input: list
    force_output: list
    force_average: list
    displacement: list
    strain: list
    nominal_strain: list
    nominal_stress: list
    stress: list
    strain_rate: list
    nominal_strain_rate: list
    time: np.ndarray
    temperature: list
    inertia_stress: list
    punch_strain: list
    friction_stress: list
    balance_quality_force: list
    balance_quality_waves: list
    balance_quality_force_instant: list
    incident: np.ndarray
    reflected: np.ndarray
    transmitted: np.ndarray


def _select_stress_force(F_input, F_output, F_average, force_check: int):
    """Picks which of the three already-computed force signals
    (`F_input`/`F_output`/`F_average`, `F_average = (F_input+F_output)/2`)
    feeds the stress calculation, per `SHPBOptions.force_check`
    (1/2/3) -- factored out since the three `type_check` branches below
    (compression/tension, notched, shear) each compute their own
    F_input/F_output/F_average locally but must all honor the same
    user choice identically."""
    if force_check == 1:
        return F_input
    if force_check == 3:
        return F_average
    return F_output


def _bar_gradient_wave_speed_modulus(
    bar: Bar, d: int, gage_temp: float, t0: float,
    type_bars_check: int, tm_value: tuple[float, float], table,
):
    """The `E_mat_*` / `C0_mat_*` "material at the interface" values and the
    `E*B` "bar surface" values under a bar-temperature-gradient correction,
    for one bar. Mirrors the heat_check==1 block of the original almost
    line-for-line (both the RTE-model branch and the imported-table branch)."""
    tm0, tm1 = tm_value
    if type_bars_check in (1, 2):
        rte_bar = (1 - (gage_temp / tm0) * math.exp(tm1 * (1 - (tm0 / gage_temp)))) / (
            1 - (293 / tm0) * math.exp(tm1 * (1 - (tm0 / 293)))
        )
        rte_mat = (1 - (t0 / tm0) * math.exp(tm1 * (1 - (tm0 / t0)))) / (
            1 - (293 / tm0) * math.exp(tm1 * (1 - (tm0 / 293)))
        )
        e_mat = rte_mat * bar.e_pa[d]
        c0_mat = math.sqrt(e_mat / bar.rho_pa[d])
        e_bar_surface = rte_bar * bar.e_pa[d]
        return e_mat, c0_mat, e_bar_surface

    # type_bars_check == 3: imported temperature-dependent tables
    e_at_gage = table.value("e", gage_temp, default=bar.e_pa[d]) if table else bar.e_pa[d]
    e_at_t0 = table.value("e", t0, default=bar.e_pa[d]) if table else bar.e_pa[d]
    rho_at_t0 = table.value("rho", t0, default=bar.rho_pa[d]) if table else bar.rho_pa[d]
    e_mat = e_at_t0
    c0_mat = math.sqrt(e_mat / rho_at_t0)
    e_bar_surface = e_at_gage
    return e_mat, c0_mat, e_bar_surface


def _clamp_nominal_strain(values):
    """Keeps nominal (engineering) strain strictly within (-1, 1) before it
    feeds a logarithmic (true) strain conversion (`-math.log(1 - v)` for
    compression, `math.log(1 + v)` for tension) -- without this, a
    genuinely large measured/corrected pulse (or a punch correction that
    pushes an already-large value further) crashes EVERY Monte-Carlo draw
    at once with a bare, undiagnosable "math domain error" instead of
    producing a finite (if extreme) result the user can actually look at
    and judge for themselves.

    Only clamps the specific side each formula is undefined past exactly
    +/-1 (compression's `-log(1-v)` only misbehaves as v -> 1; tension's
    `log(1+v)` only misbehaves as v -> -1) -- clamping both sides
    unconditionally is a no-op for any value already inside the domain, so
    this is a pure safety net, not a change to correct-domain behavior.
    Shared by core/process_shpb.py and core/process_di.py, which are both
    exposed to this same class of domain error."""
    return [min(max(v, -0.99999), 0.99999) for v in values]


def _require_notched_geometry(specimen: Specimen, d: int) -> None:
    """Raises a clear, actionable error instead of letting the notched
    (`type_check == 3`) branch below silently divide by zero (a cascade of
    "invalid value"/"divide by zero" RuntimeWarnings, then either garbage
    output or a downstream crash) when the specimen's notch radius hasn't
    actually been configured.

    `type_check` (1 compression / 2 tension / 3 notched / 4 shear) is not
    chosen per Processing tab -- both the SHPB and Direct Impact tabs derive
    it from the single, project-wide "Test mode" set on the Test Setup page
    (see `_test_mode_options()` in ui/pages/page_process.py). If a shear-
    compression/tension (notched) test was processed earlier in the same
    session and Test Setup was never switched back before processing an
    unrelated compression/tension/shear test on another tab, this branch
    still runs with `specimen.re_pa` (notch radius) left at its unconfigured
    default of 0, and `denom = 2000 * specimen.re_pa[d] / cos(...)` is then
    zero -- the actual root cause behind a "divide by zero encountered"/
    "invalid value encountered" RuntimeWarning storm with no fatal
    exception, or a much later, harder-to-diagnose crash/garbage result.
    """
    if not (float(specimen.re_pa[d]) > 0):
        raise ValueError(
            "This is being processed as a notched specimen (Shear-Compression/Tension), but the "
            "notch radius is 0/unconfigured -- check the \"Test mode\" on the Test Setup page (it "
            "applies to every Processing tab, so switching it for one test also affects any other "
            "test processed afterward) and the notch radius / displacement-at-yield fields on the "
            "Specimen page."
        )


def _index_of_nearest(time_pulse: list, target: float) -> int:
    """Index of the `time_pulse` sample closest to `target`.

    A nearest-sample search rather than an exact-match lookup, since
    `time_pulse` is not rounded to a fixed decimal precision before being
    stored (ui/pages/page_process.py's Direct Impact tab): rounding to
    microsecond precision would silently collapse distinct samples
    together for any time step under ~1 microsecond -- exactly what a
    "generate the time vector from a fixed time step" pulse window uses
    when the step is a fraction of a microsecond (a common 2-5 MHz
    bar-gage sample rate). The caller already snaps xd1/xd2 to the nearest
    ACTUAL `time_pulse` sample before calling in here, so this is normally
    an exact (zero-distance) match anyway -- searching for the nearest one
    instead of demanding an exact hit is simply more robust, at negligible
    extra cost, and needs no rounding at all. Identical in spirit to
    core/dispersion.py's own `_index_of_nearest` (kept separate rather than
    imported, so this module doesn't pick up dispersion.py's sympy
    dependency just for this one small helper)."""
    return min(range(len(time_pulse)), key=lambda i: abs(time_pulse[i] - target))


def _robust_dt(time1) -> float:
    """Per-sample time step for the strain-rate/strain-rate-rate finite
    differences below, in place of the original script's `time1[2] - time1[1]`
    (a single hardcoded pair of indices, standing in for "the" sample
    spacing under the assumption of uniform sampling).

    That assumption breaks the moment those TWO SPECIFIC samples happen to
    share a timestamp -- e.g. one duplicated row from the raw time file, a
    downsampling artifact, or a saturated-then-repeated DAQ sample right at
    that position -- even though the signal is fine everywhere else. `dt`
    would then be exactly 0, and every `t_sr`/`n_srsr`/`inertia_stress`
    computation using it divides by zero for the WHOLE draw.

    The median of every consecutive difference is robust to a handful of
    bad samples anywhere in the array, and for a genuinely uniformly-sampled
    signal it gives the same answer `time1[2] - time1[1]` was trying to
    approximate. Shared by core/process_shpb.py and core/process_di.py."""
    diffs = np.diff(np.asarray(time1, dtype=float))
    dt = float(np.median(diffs)) if len(diffs) else 0.0
    if dt == 0:
        raise ValueError(
            "Could not determine a time step for the strain-rate calculation -- too many samples "
            "in this pulse window share the same timestamp as their neighbor. Check the loaded time "
            "file and any downsampling settings."
        )
    return dt


def process_shpb(
    options: SHPBOptions,
    IB: Bar, OB: Bar, specimen: Specimen,
    timet: np.ndarray, xd1: float, xd2: float,
    d: int,
    SG_unc_d: np.ndarray, T0_pa: np.ndarray, friction: np.ndarray,
    k_value: Optional[list] = None, sk_value: Optional[list] = None,
    plat_nu: Optional[np.ndarray] = None, plat_rho: Optional[np.ndarray] = None,
    plat_emod: Optional[np.ndarray] = None, plat_Dia: Optional[np.ndarray] = None,
    plat_Len: Optional[np.ndarray] = None,
    Tm_value: Optional[tuple] = None, T_gages_value: Optional[tuple] = None,
    material_table_ib=None, material_table_ob=None,
    AA: Optional[list] = None,
    output_window_offset: float = 0.0,
) -> SHPBResult:
    """Process one Monte-Carlo draw `d` of an SHPB test. `IB`/`OB`/`specimen`
    must already be Monte-Carlo-expanded (`.expand(N)` called beforehand,
    and `.expand_viscoelastic(...)` too if `options.disp_visco`)."""

    # ---- Cutting waves --------------------------------------------------
    specimen_C0_pa = math.sqrt(specimen.e_pa[d] / specimen.rho_pa[d])
    plat_C0 = math.sqrt(plat_emod[d] / plat_rho[d]) if options.plat_check else 0.0

    idx_i1 = int(np.searchsorted(timet, xd1, side="right") - 1)
    idx_i2 = int(np.searchsorted(timet, xd2, side="right") - 1)
    ti_pulse = idx_i2 - idx_i1

    sxd1 = xd1 + 2 * (IB.PG_pa[d]) / IB.C0_pa[d]
    if options.plat_check:
        sxd1 += 2 * plat_Len[d] / plat_C0
    idx_r1 = int(np.searchsorted(timet, sxd1, side="right") - 1)
    idx_r2 = idx_r1 + ti_pulse

    ssxd1 = xd1 + OB.PG_pa[d] / OB.C0_pa[d] + specimen.L_pa[d] / specimen_C0_pa + IB.PG_pa[d] / IB.C0_pa[d]
    if options.plat_check:
        ssxd1 += 2 * plat_Len[d] / plat_C0
    ssxd1 += output_window_offset  # manual correction for residual misalignment (e.g. an initial-slip artifact)
    idx_t1 = int(np.searchsorted(timet, ssxd1, side="right") - 1)
    idx_t2 = idx_t1 + ti_pulse

    time1 = timet[idx_i1:idx_i2] - xd1
    incident = IB.signal[idx_i1:idx_i2] * SG_unc_d[d]
    reflected = IB.signal[idx_r1:idx_r2] * SG_unc_d[d]
    transmitted = OB.signal[idx_t1:idx_t2] * SG_unc_d[d]

    n = len(incident)
    if len(reflected) < n:
        reflected = np.concatenate([reflected, np.zeros(n - len(reflected))])
    if len(transmitted) < n:
        transmitted = np.concatenate([transmitted, np.zeros(n - len(transmitted))])

    if options.disp_ela or options.disp_visco:
        # Use waves already corrected upstream (e.g. by core/dispersion.py)
        # instead of the raw cut waves.
        incident, reflected, transmitted = AA[0][:], AA[1][:], AA[2][:]
        if output_window_offset:
            # The AA arrays are a fixed, already-cut signal that knows
            # nothing about `output_window_offset` (unlike the raw-cut
            # path above, which already bakes it into ssxd1/idx_t1) --
            # apply it here too, as an equivalent sample shift, so the
            # manual realignment correction actually has an effect when
            # using the dispersion-corrected signal.
            dt_step = timet[1] - timet[0] if len(timet) > 1 else 1.0
            shift = int(round(output_window_offset / dt_step))
            if shift > 0 and len(transmitted) > shift:
                transmitted = np.concatenate([transmitted[shift:], np.full(shift, transmitted[-1])])
            elif shift < 0 and len(transmitted) > -shift:
                transmitted = np.concatenate([np.full(-shift, transmitted[0]), transmitted[:shift]])

    # ---- Bar <-> punch/platen geometry -----------------------------------
    # F_input/F_output are computed from strain measured by gages mounted
    # ON THE BARS, using 1D wave theory (F = E_bar * A_bar * strain) -- that
    # relationship holds at the bar's own cross-section regardless of
    # what's beyond the bar's end, so the area term (DB_punchI/O) must
    # always be the BAR's own diameter, whether or not a platen is used.
    #
    # nuB_punchI/O / EB_punchI/O are used ONLY inside the punch (indentation)
    # correction, and are always the bar's OWN properties too: indentation
    # happens at the bar surface, against the specimen directly when
    # there's no platen, or against the platen (rigid relative to the bar)
    # when platens are used. `punch_diameter_I/O` is the diameter of
    # whichever disc is doing the indenting (specimen normally, platen when
    # plat_check) and is resolved at each punch-correction call site below.
    #
    # NOTE: this deliberately does NOT use the platen's own diameter/
    # modulus for the F_input/output area, even with plat_check on. Using
    # the platen's (typically smaller) diameter there would model force
    # propagation through the platen's own cross-section instead of the
    # bar's, under-estimating F_input/F_output relative to the true
    # bar-gage reading whenever platens are enabled -- the bar's own
    # diameter is what the strain gages actually measure against,
    # regardless of what's beyond the bar's end.
    DB_punchI, DB_punchO = IB.D_pa[d], OB.D_pa[d]
    nuB_punchI, EB_punchI = IB.nu_pa[d], IB.e_pa[d]
    nuB_punchO, EB_punchO = OB.nu_pa[d], OB.e_pa[d]
    punch_diameter_I = plat_Dia[d] if options.plat_check else None  # None -> caller uses specimen/D_eq
    punch_diameter_O = plat_Dia[d] if options.plat_check else None

    if options.plat_check:
        E_mat_IB, E_mat_OB = IB.e_pa[d], OB.e_pa[d]
        C0_mat_IB, C0_mat_OB = IB.C0_pa[d], OB.C0_pa[d]
    else:
        E_mat_IB = E_mat_OB = C0_mat_IB = C0_mat_OB = None  # set below

    # ---- Bar-temperature-gradient (heat) correction ----------------------
    if options.heat_check:
        gage_t_ib, gage_t_ob = T_gages_value
        e_mat_ib, c0_mat_ib, e_surface_ib = _bar_gradient_wave_speed_modulus(
            IB, d, gage_t_ib, T0_pa[d], options.type_bars_check, Tm_value, material_table_ib)
        e_mat_ob, c0_mat_ob, e_surface_ob = _bar_gradient_wave_speed_modulus(
            OB, d, gage_t_ob, T0_pa[d], options.type_bars_check, Tm_value, material_table_ob)
        if options.plat_check:
            # Matches the original: with heat_check AND plat_check both on,
            # E_mat/C0_mat become the platen's properties (the "material on
            # the other side of the bar interface" is the platen here).
            E_mat_IB = E_mat_OB = plat_emod[d]
            C0_mat_IB = C0_mat_OB = plat_C0
        else:
            E_mat_IB, C0_mat_IB = e_mat_ib, c0_mat_ib
            E_mat_OB, C0_mat_OB = e_mat_ob, c0_mat_ob

        C0IB = math.sqrt(e_surface_ib / IB.rho_pa[d])
        C0OB = math.sqrt(e_surface_ob / OB.rho_pa[d])
        SBI = math.pi * (IB.D_pa[d] / 2) ** 2
        SBO = math.pi * (OB.D_pa[d] / 2) ** 2
        TBP_input = 2 * C0IB * e_surface_ib * SBI / (C0IB * E_mat_IB * SBI + C0_mat_IB * e_surface_ib * SBI)
        TBP_output = 2 * C0OB * e_surface_ob * SBO / (C0OB * E_mat_OB * SBO + C0_mat_OB * e_surface_ob * SBO)
        TPB_input = 2 * C0_mat_IB * E_mat_IB * SBI / (C0IB * E_mat_IB * SBI + C0_mat_IB * e_surface_ib * SBI)
        TPB_output = 2 * C0_mat_OB * E_mat_OB * SBO / (C0OB * E_mat_OB * SBO + C0_mat_OB * e_surface_ob * SBO)
        incident = incident * TBP_input
        reflected = reflected / TPB_input
        transmitted = transmitted / TPB_output
        # EB_punch is always the bar's own (heat-adjusted) modulus now,
        # regardless of plat_check -- e_mat_ib/e_mat_ob are the bar's own
        # values computed above, before any platen substitution into
        # E_mat_IB/E_mat_OB.
        EB_punchI, EB_punchO = e_mat_ib, e_mat_ob

    if not options.heat_check and not options.plat_check:
        E_mat_IB, E_mat_OB = IB.e_pa[d], OB.e_pa[d]
        C0_mat_IB, C0_mat_OB = IB.C0_pa[d], OB.C0_pa[d]

    def _viscoelastic_modulus_and_speed():
        """Shared `disp_visco` branch: complex-modulus-real-part correction
        applied to the bar/wave conversion, identical across all three
        kinematics blocks in the original."""
        ft = time1[1] - time1[0]
        NN = 2 * len(incident) - 1
        freqi = np.fft.rfftfreq(NN, d=ft)
        E1, E2, n2 = IB.e11[d], IB.e22[d], IB.n22[d]
        EE11 = []
        for f in freqi:
            E = np.reciprocal((1 / E1) + (E2 / (E2 ** 2 + (2 * math.pi * f * n2) ** 2))
                               - 1j * ((2 * math.pi * f * n2) / (E2 ** 2 + (2 * math.pi * f * n2) ** 2)))
            EE11.append(E.real)
        CC11 = [math.sqrt(e / IB.rho_pa[d]) for e in EE11]
        CC22 = [math.sqrt(e / OB.rho_pa[d]) for e in EE11]
        return EE11, CC11, CC22

    # ------------------------------------------------------------------
    # type_check == 3: notched specimen (Shear-Compression/Tension)
    # ------------------------------------------------------------------
    if options.type_check == 3:
        _require_notched_geometry(specimen, d)
        sign = -1 if options.tenscomp_check == 1 else 1
        F_input = sign * (E_mat_IB * math.pi * (DB_punchI / 2) ** 2) * (incident + reflected) * IB.CPG_pa[d]
        F_output = sign * (E_mat_OB * math.pi * (DB_punchO / 2) ** 2) * transmitted * OB.CPG_pa[d]
        F_average = (F_input + F_output) / 2
        V_input = sign * C0_mat_IB * (incident - reflected) * IB.CPG_pa[d]
        V_output = sign * C0_mat_OB * transmitted * OB.CPG_pa[d]

        if options.disp_visco:
            EE11, CC11, CC22 = _viscoelastic_modulus_and_speed()
            F_input = [sign * (EE11[i] * math.pi * (DB_punchI / 2) ** 2) * (incident[i] + reflected[i]) * IB.CPG_pa[d] for i in range(len(incident))]
            F_output = [sign * (EE11[i] * math.pi * (DB_punchO / 2) ** 2) * transmitted[i] * OB.CPG_pa[d] for i in range(len(transmitted))]
            F_average = [(F_input[i] + F_output[i]) / 2 for i in range(len(F_input))]
            V_input = [sign * CC11[i] * (incident[i] - reflected[i]) * IB.CPG_pa[d] for i in range(len(incident))]
            V_output = [sign * CC22[i] * transmitted[i] * OB.CPG_pa[d] for i in range(len(transmitted))]

        dt = time1[1] - time1[0]
        displacement = [0.0]
        for i in range(len(V_input)):
            displacement.append(displacement[-1] + (V_input[i] - V_output[i]) * dt * 1000)
        displacement = displacement[:-1]

        incident = incident * IB.CPG_pa[d]
        reflected = reflected * IB.CPG_pa[d]
        transmitted = transmitted * OB.CPG_pa[d]

        bq1, bq2, bq3 = _balance_quality(F_input, F_output, displacement, incident, reflected, transmitted, time1)

        FF = _select_stress_force(F_input, F_output, F_average, options.force_check)
        denom = 2000 * specimen.re_pa[d] / math.cos(specimen.NoA_pa[d])
        if not options.sk_check:
            t_s = [
                float(k_value[0]) * (displacement[i] - 1000 * specimen.dy_pa[d]) / denom
                + float(k_value[1]) * ((displacement[i] - 1000 * specimen.dy_pa[d]) / denom) ** 2
                + float(k_value[2]) * ((displacement[i] - 1000 * specimen.dy_pa[d]) / denom) ** 3
                for i in range(len(displacement))
            ]
            t_stress = [
                float(k_value[3]) * (1 - float(k_value[4]) * t_s[i]) * (abs(FF[i]) ** float(k_value[5]))
                / (1000 * specimen.th_pa[d] * 1000 * specimen.D_pa[d])
                for i in range(len(FF))
            ]
        else:
            t_s, t_stress = [], []
            for i in range(len(displacement)):
                kv = k_value if displacement[i] <= float(sk_value[6]) else sk_value
                t_s.append(
                    float(kv[0]) * (displacement[i] - 1000 * specimen.dy_pa[d]) / denom
                    + float(kv[1]) * ((displacement[i] - 1000 * specimen.dy_pa[d]) / denom) ** 2
                    + float(kv[2]) * ((displacement[i] - 1000 * specimen.dy_pa[d]) / denom) ** 3
                )
            for i in range(len(FF)):
                kv = k_value if displacement[i] <= float(sk_value[6]) else sk_value
                t_stress.append(
                    float(kv[3]) * (1 - float(kv[4]) * t_s[i]) * (abs(FF[i]) ** float(kv[5]))
                    / (1000 * specimen.th_pa[d] * 1000 * specimen.D_pa[d])
                )

        t_sr = [(t_s[i + 1] - t_s[i]) / (time1[i + 1] - time1[i]) for i in range(len(t_s) - 1)]
        t_srsr = [(t_sr[i + 1] - t_sr[i]) / (time1[i + 1] - time1[i]) for i in range(len(t_sr) - 1)]

        p_s = 0
        if options.punch_check:
            punch_d_I = punch_diameter_I if punch_diameter_I is not None else specimen.D_pa[d]
            punch_d_O = punch_diameter_O if punch_diameter_O is not None else specimen.D_pa[d]
            # `_punch_correction_notched` returns one correction value per
            # element of whatever array it's given, so the array passed in
            # here must be this branch's own actual, full-length stress
            # array (`t_stress`) -- not a placeholder -- so that `p_s`
            # (and then `t_s`, sliced to `range(len(p_s))`) comes out at
            # the same length as `t_stress` itself. This is exactly
            # analogous to how the compression/tension and shear branches
            # both pass their own real `n_stress` array here.
            p_s = _punch_correction_notched(specimen, d, punch_d_I, punch_d_O, IB.D_pa[d], OB.D_pa[d], nuB_punchI, nuB_punchO,
                                             EB_punchI, EB_punchO, t_stress, options.plat_check, plat_emod, plat_Len)
            t_s = [t_s[i] - p_s[i] for i in range(len(p_s))]

        s0 = specimen.L_pa[d] / specimen.D_pa[d]
        friction_stress = 0
        if friction[d] != 0:
            m = 1 - friction[d] / (3 * s0)
            t_stress = [m * v for v in t_stress]
            friction_stress = [friction[d] / (3 * s0) * v for v in t_stress]
            t_stress = [t_stress[i] - friction_stress[i] for i in range(len(friction_stress))]

        inertia_stress = 0
        if options.inertia_check:
            cos_a = math.cos(math.pi / 4)
            SS = (specimen.D_pa[d] / cos_a) * specimen.th_pa[d]
            D = 2 * math.sqrt(SS / math.pi)
            s0i = 2 * (specimen.re_pa[d] / cos_a) / D
            c1 = specimen.rho_pa[d] * (D ** 2) * (s0i ** 2 - 3 / 16) / 12
            c2 = 3 * specimen.rho_pa[d] * (D ** 2) / 64
            inertia_stress = [c1 * (t_sr[i] ** 2 + t_srsr[i]) / 1e6 + c2 * t_srsr[i] / 1e6 for i in range(len(t_srsr))]
            inertia_stress.append(inertia_stress[-1])
            t_stress = [t_stress[i] - inertia_stress[i] for i in range(len(inertia_stress))]
            t_stress.append(t_stress[-1])

        n_s, n_stress = t_s, t_stress
        # This kinematics has no separate nominal/true-strain distinction
        # (`n_s`/`n_stress` are just aliased to the k-coefficient-fitted
        # `t_s`/`t_stress` above) -- `t_sr` (already computed for the
        # inertia correction) is its own strain-rate equivalent. `n_sr`
        # must still be set here: `SHPBResult` further down unconditionally
        # reads it for `nominal_strain_rate`, same as every other branch.
        n_sr = t_sr

    # ------------------------------------------------------------------
    # type_check in (1, 2): compression / tension
    # ------------------------------------------------------------------
    elif options.type_check in (1, 2):
        sign = -1 if options.type_check == 1 else 1
        F_input = sign * (E_mat_IB * math.pi * (DB_punchI / 2) ** 2) * (incident + reflected) * IB.CPG_pa[d]
        F_output = sign * (E_mat_OB * math.pi * (DB_punchO / 2) ** 2) * transmitted * OB.CPG_pa[d]
        F_average = (F_input + F_output) / 2
        V_input = sign * C0_mat_IB * (incident - reflected) * IB.CPG_pa[d]
        V_output = sign * C0_mat_OB * transmitted * OB.CPG_pa[d]
        n_sr = (V_input - V_output) / specimen.L_pa[d]

        if options.disp_visco:
            EE11, CC11, CC22 = _viscoelastic_modulus_and_speed()
            F_input = [sign * (EE11[i] * math.pi * (DB_punchI / 2) ** 2) * (incident[i] + reflected[i]) * IB.CPG_pa[d] for i in range(len(incident))]
            F_output = [sign * (EE11[i] * math.pi * (DB_punchO / 2) ** 2) * transmitted[i] * OB.CPG_pa[d] for i in range(len(transmitted))]
            F_average = [(F_input[i] + F_output[i]) / 2 for i in range(len(F_input))]
            V_input = [sign * CC11[i] * (incident[i] - reflected[i]) * IB.CPG_pa[d] for i in range(len(incident))]
            V_output = [sign * CC22[i] * transmitted[i] * OB.CPG_pa[d] for i in range(len(transmitted))]
            n_sr = np.array([(V_input[i] - V_output[i]) / specimen.L_pa[d] for i in range(len(V_input))])

        incident = incident * IB.CPG_pa[d]
        reflected = reflected * IB.CPG_pa[d]
        transmitted = transmitted * OB.CPG_pa[d]

        int_sr, n_s, displacement = 0.0, [], []
        for i in range(len(time1) - 1):
            int_sr += n_sr[i] * (time1[i + 1] - time1[i])
            int_sr = min(max(int_sr, -0.99999), 0.99999)  # see _clamp_nominal_strain's docstring
            n_s.append(int_sr)
            # Expressed in mm: the original computes this branch's
            # displacement in meters (specimen.L_pa[d] is in meters) while
            # its own type_check==3 (notched) branch expresses displacement
            # in mm (an inherent unit inconsistency between branches in the
            # original script itself) -- standardized to mm here to match
            # common usage and the rest of this app.
            displacement.append(1000 * specimen.L_pa[d] * int_sr)

        F_input, F_output, F_average = F_input[:-1], F_output[:-1], F_average[:-1]
        bq1, bq2, bq3 = _balance_quality(F_input, F_output, displacement, incident, reflected, transmitted, time1)

        D_eq_sp = None
        if options.tens_check == 1 or options.comp_check == 1:
            section = math.pi * (specimen.D_pa[d] / 2) ** 2
        else:
            section = specimen.wi_pa[d] * specimen.thi_pa[d]
            D_eq_sp = 2 * math.sqrt(section / math.pi)

        FF = _select_stress_force(F_input, F_output, F_average, options.force_check)
        n_stress = np.asarray(FF) / section / 1e6
        t_stress = [n_stress[i] * (1 - 2 * specimen.nu_pa[d] * n_s[i]) for i in range(len(n_stress))] if options.type_check == 1 \
            else [n_stress[i] * (1 + 2 * specimen.nu_pa[d] * n_s[i]) for i in range(len(n_stress))]

        # True (logarithmic) strain, derived from the nominal (engineering)
        # strain `n_s` built above: `t_s` applies the log transform
        # (`-log(1-v)` for compression, `log(1+v)` for tension) on top of
        # `n_s`, rather than aliasing `t_s = n_s` directly -- `displacement`
        # (built from `n_s` alone) is the nominal/engineering quantity,
        # while `t_s`/`t_stress` (and everything downstream of them) are
        # true-strain quantities, so the two must not be conflated.
        if options.type_check == 1:
            t_s = [-math.log(1 - v) for v in n_s]
        else:
            t_s = [math.log(1 + v) for v in n_s]
        p_s = 0
        if options.punch_check:
            t_stress, t_s = [], []
            d_eq = specimen.D_pa[d] if D_eq_sp is None else D_eq_sp
            punch_d = punch_diameter_I if punch_diameter_I is not None else d_eq
            p_s = _punch_correction_generic(specimen, d, d_eq, punch_d, punch_d, IB.D_pa[d], OB.D_pa[d], nuB_punchI, nuB_punchO,
                                             EB_punchI, EB_punchO, n_stress, options.plat_check, plat_emod, plat_Len)
            n_s = _clamp_nominal_strain([n_s[i] - p_s[i] for i in range(len(p_s))])
            for i in range(len(time1) - 1):
                t_s.append(-math.log(1 - n_s[i]))
                t_stress.append(n_stress[i] * (1 - 2 * specimen.nu_pa[d] * n_s[i]))

        s0 = specimen.L_pa[d] / (specimen.D_pa[d] if D_eq_sp is None else D_eq_sp)
        friction_stress = 0
        if friction[d] != 0:
            m = 1 - friction[d] / (3 * s0)
            n_stress = [m * v for v in n_stress]
            t_stress = [m * v for v in t_stress]
            friction_stress = [friction[d] / (3 * s0) * v for v in t_stress]
            t_stress = [t_stress[i] - friction_stress[i] for i in range(len(friction_stress))]

        t_sr, n_srsr = [], []
        n_sr = n_sr[:-1]
        dt = _robust_dt(time1)
        for i in range(len(t_s) - 1):
            t_sr_i = (t_s[i + 1] - t_s[i]) / dt
            n_srsr_i = (n_sr[i + 1] - n_sr[i]) / dt if i + 1 < len(n_sr) else (t_sr[-1] if t_sr else 0.0)
            if t_sr_i == 0 and t_sr:
                t_sr_i = t_sr[-1]
            t_sr.append(t_sr_i)
            n_srsr.append(n_srsr_i)

        n_sr = n_sr[:-1]
        inertia_stress = 0
        if options.inertia_check:
            d_eq = specimen.D_pa[d] if D_eq_sp is None else D_eq_sp
            c1 = specimen.rho_pa[d] * (d_eq ** 2) * (s0 ** 2 - 3 / 16) / 12
            c2 = 3 * specimen.rho_pa[d] * (d_eq ** 2) / 64
            inertia_stress = [c1 * (n_sr[i] ** 2 + n_srsr[i]) / 1e6 + c2 * n_srsr[i] / 1e6 for i in range(len(n_srsr))]
            inertia_stress.append(inertia_stress[-1])
            t_stress = [t_stress[i] - inertia_stress[i] for i in range(len(t_stress))]

    # ------------------------------------------------------------------
    # type_check == 4: shear (hat-shaped / double / simple)
    # ------------------------------------------------------------------
    else:
        l_u = (specimen.H2_pa[d] - specimen.H1_pa[d]) if options.shear_check == 1 else specimen.wi_pa[d]

        sign = -1 if options.tenscomp_check == 1 else 1
        F_input = sign * (E_mat_IB * math.pi * (IB.D_pa[d] / 2) ** 2) * (incident + reflected) * IB.CPG_pa[d]
        F_output = sign * (E_mat_OB * math.pi * (OB.D_pa[d] / 2) ** 2) * transmitted * OB.CPG_pa[d]
        F_average = (F_input + F_output) / 2
        V_input = sign * C0_mat_IB * (incident - reflected) * IB.CPG_pa[d]
        V_output = sign * C0_mat_OB * transmitted * OB.CPG_pa[d]
        n_sr = (V_input - V_output) / l_u

        if options.disp_visco:
            EE11, CC11, CC22 = _viscoelastic_modulus_and_speed()
            F_input = [sign * (EE11[i] * math.pi * (IB.D_pa[d] / 2) ** 2) * (incident[i] + reflected[i]) * IB.CPG_pa[d] for i in range(len(incident))]
            F_output = [sign * (EE11[i] * math.pi * (OB.D_pa[d] / 2) ** 2) * transmitted[i] * OB.CPG_pa[d] for i in range(len(transmitted))]
            F_average = [(F_input[i] + F_output[i]) / 2 for i in range(len(F_input))]
            V_input = [sign * CC11[i] * (incident[i] - reflected[i]) * IB.CPG_pa[d] for i in range(len(incident))]
            V_output = [sign * CC22[i] * transmitted[i] * OB.CPG_pa[d] for i in range(len(transmitted))]
            n_sr = np.array([(V_input[i] - V_output[i]) / l_u for i in range(len(V_input))])

        incident = incident * IB.CPG_pa[d]
        reflected = reflected * IB.CPG_pa[d]
        transmitted = transmitted * OB.CPG_pa[d]

        int_sr, n_s, displacement = 0.0, [], []
        for i in range(len(time1) - 1):
            int_sr += n_sr[i] * (time1[i + 1] - time1[i])
            int_sr = min(max(int_sr, -0.99999), 0.99999)  # see _clamp_nominal_strain's docstring
            n_s.append(int_sr)
            # Same mm standardization as the type_check in (1,2) branch above.
            displacement.append(1000 * l_u * int_sr)

        F_input, F_output, F_average = F_input[:-1], F_output[:-1], F_average[:-1]
        bq1, bq2, bq3 = _balance_quality(F_input, F_output, displacement, incident, reflected, transmitted, time1)

        if options.shear_check == 1:
            SAS = math.pi * (specimen.R1_pa[d] + specimen.R2_pa[d]) * (specimen.H2_pa[d] - specimen.H1_pa[d])
        elif options.shear_check == 2:
            SAS = 2 * specimen.thi_pa[d] * specimen.L_pa[d]
        else:
            SAS = specimen.thi_pa[d] * specimen.L_pa[d]

        FF = _select_stress_force(F_input, F_output, F_average, options.force_check)
        n_stress = np.asarray(FF) / SAS / 1e6
        t_stress = list(n_stress)
        t_s = list(n_s)

        if options.shear_check == 1:
            D_eq1, D_eq2 = 2 * specimen.R3_pa[d], 2 * specimen.R1_pa[d]
        elif options.shear_check == 2:
            SS1, SS2 = specimen.thi_pa[d] * specimen.wiL_pa[d], 2 * specimen.thi_pa[d] * specimen.wiL_pa[d]
            D_eq1, D_eq2 = math.sqrt(4 * SS1 / math.pi), math.sqrt(4 * SS2 / math.pi)
        else:
            SS = specimen.thi_pa[d] * specimen.wiL_pa[d]
            D_eq1 = D_eq2 = math.sqrt(4 * SS / math.pi)
        D_eq = (D_eq1 + D_eq2) / 2

        if options.punch_check:
            t_stress, t_s = [], []
            punch_d_I = punch_diameter_I if punch_diameter_I is not None else D_eq1
            punch_d_O = punch_diameter_O if punch_diameter_O is not None else D_eq
            p_s = _punch_correction_shear(specimen, d, D_eq, D_eq1, punch_d_I, punch_d_O, IB.D_pa[d], OB.D_pa[d], nuB_punchI, nuB_punchO,
                                           EB_punchI, EB_punchO, n_stress, options.plat_check, plat_emod, plat_Len)
            n_s = [n_s[i] - p_s[i] for i in range(len(p_s))]
            for i in range(len(time1) - 1):
                t_s.append(n_s[i] / 2)
                t_stress.append(n_stress[i])

        if options.shear_check == 1:
            s0 = specimen.H3_pa[d] / D_eq
        else:
            s0 = specimen.leL_pa[d] / D_eq

        friction_stress = 0
        if friction[d] != 0:
            m = 1 - friction[d] / (3 * s0)
            n_stress = [m * v for v in n_stress]
            t_stress = [m * v for v in t_stress]
            friction_stress = [friction[d] / (3 * s0) * v for v in t_stress]
            t_stress = [t_stress[i] - friction_stress[i] for i in range(len(friction_stress))]

        n_sr = n_sr[:-1]
        t_sr, n_srsr = [], []
        dt = _robust_dt(time1)
        for i in range(len(t_s) - 1):
            t_sr.append((t_s[i + 1] - t_s[i]) / dt)
            # `n_srsr` (strain-RATE-rate, for the inertia correction below)
            # differentiates `n_sr` (strain rate) once, by the standard
            # forward-difference `(n_sr[i+1]-n_sr[i])/dt` -- the true
            # second derivative of strain, needed by the inertia
            # correction's `t_sr**2 + t_srsr` term. Matches the analogous
            # `n_srsr_i = (n_sr[i+1]-n_sr[i])/dt` already used in the
            # compression/tension branch above.
            if i + 1 < len(n_sr):
                n_srsr.append((n_sr[i + 1] - n_sr[i]) / dt)
            else:
                n_srsr.append(n_srsr[-1] if n_srsr else 0.0)

        n_sr = n_sr[:-1]
        inertia_stress = 0
        if options.inertia_check:
            if options.shear_check == 1:
                D_eq2 = 2 * math.sqrt((specimen.R2_pa[d] + specimen.R1_pa[d]) * (specimen.H2_pa[d] - specimen.H1_pa[d]))
            elif options.shear_check == 2:
                D_eq2 = math.sqrt(8 * SAS / math.pi)
            else:
                D_eq2 = math.sqrt(4 * SAS / math.pi)
            c1 = specimen.rho_pa[d] * (D_eq2 ** 2) * (s0 ** 2 - 3 / 16) / 12
            c2 = 3 * specimen.rho_pa[d] * (D_eq2 ** 2) / 64
            inertia_stress = [c1 * (n_sr[i] ** 2 + n_srsr[i]) / 1e6 + c2 * n_srsr[i] / 1e6 for i in range(len(n_srsr))]
            inertia_stress.append(inertia_stress[-1])
            t_stress = [t_stress[i] - inertia_stress[i] for i in range(len(t_stress))]

    # ---- Adiabatic temperature rise (shared by all kinematics) ----------
    coeff_temp = specimen.xi_pa[d] / (specimen.cp_pa[d] * specimen.rho_pa[d])
    temperature = [T0_pa[d]]
    for i in range(len(t_s) - 1):
        temperature.append(temperature[-1] + coeff_temp * t_stress[i] * 1e6 * (t_s[i + 1] - t_s[i]))

    return SHPBResult(
        force_input=list(F_input), force_output=list(F_output), force_average=list(F_average),
        displacement=list(displacement), strain=list(t_s), nominal_strain=list(n_s),
        nominal_stress=list(n_stress) if not isinstance(n_stress, list) else n_stress,
        stress=list(t_stress), strain_rate=list(t_sr),
        nominal_strain_rate=list(n_sr) if not isinstance(n_sr, list) else n_sr,
        time=time1, temperature=temperature,
        inertia_stress=inertia_stress if isinstance(inertia_stress, list) else [],
        punch_strain=p_s if isinstance(p_s, list) else [],
        friction_stress=friction_stress if isinstance(friction_stress, list) else [],
        balance_quality_force=bq1, balance_quality_waves=bq2, balance_quality_force_instant=bq3,
        incident=incident, reflected=reflected, transmitted=transmitted,
    )


def _balance_quality(F_input, F_output, displacement, incident, reflected, transmitted, time1):
    """'Balance quality' diagnostics (from force, and from wave energy),
    shared identically across all three kinematics blocks in the original."""
    dint_fi = dint_fo = 0.0
    int_fi, int_fo = [], []
    for i in range(len(F_input) - 1):
        dint_fi += F_input[i] * (displacement[i + 1] - displacement[i])
        dint_fo += F_output[i] * (displacement[i + 1] - displacement[i])
        int_fi.append(dint_fi)
        int_fo.append(dint_fo)
    bq1 = [int_fo[r] / int_fi[r] for r in range(len(int_fo))]
    bq3 = [F_output[r] / F_input[r] for r in range(len(int_fo)) if r < len(F_output) and r < len(F_input)]

    dint_i = dint_r = dint_t = 0.0
    int_i, int_r, int_t = [], [], []
    for i in range(len(time1) - 1):
        dint_i += incident[i] * (time1[i + 1] - time1[i])
        dint_r += reflected[i] * (time1[i + 1] - time1[i])
        dint_t += transmitted[i] * (time1[i + 1] - time1[i])
        int_i.append(dint_i)
        int_r.append(dint_r)
        int_t.append(dint_t)
    bq2 = [int_t[r] / (int_i[r] + int_r[r]) for r in range(len(int_i))]
    return bq1, bq2, bq3


def _elliptic_punch_factor(d_ratio: float) -> float:
    """Shared elliptic-integral geometry factor `Hx` used by every punch
    (indentation) correction branch below. This formula assumes the punch/
    platen is at least as wide as the specimen (d_ratio = specimen_diameter
    / punch_diameter < 1); outside that range the integrand goes negative
    under the square root, which is what "math domain error" means here."""
    if not (0 < d_ratio < 1):
        raise ValueError(
            f"Punch/platen correction requires the punch or platen diameter to be larger than the "
            f"specimen diameter (got specimen/punch ratio = {d_ratio:.4g}, must be < 1). "
            f"Check the specimen diameter and the bar/platen diameter."
        )
    Ex = integrate.quad(lambda x: math.sqrt(1 - (d_ratio ** 2) * math.sin(x) ** 2), 0, math.pi / 2)[0]
    Kx = integrate.quad(lambda x: 1 / math.sqrt(1 - (d_ratio ** 2) * math.sin(x) ** 2), 0, math.pi / 2)[0]
    return 2 - (d_ratio + 1 / d_ratio) * Ex - (d_ratio - 1 / d_ratio) * Kx


def _punch_correction_notched(specimen, d, punch_diameter_I, punch_diameter_O, bar_diameter_I, bar_diameter_O, nuB_punchI, nuB_punchO,
                               EB_punchI, EB_punchO, n_stress, plat_check, plat_emod, plat_Len):
    d_ratioI = punch_diameter_I / bar_diameter_I
    KpI = _elliptic_punch_factor(d_ratioI) * (16 / (3 * math.pi ** 2)) * (1 - nuB_punchI ** 2) / (punch_diameter_I * EB_punchI)
    d_ratioO = punch_diameter_O / bar_diameter_O
    KpO = _elliptic_punch_factor(d_ratioO) * (16 / (3 * math.pi ** 2)) * (1 - nuB_punchO ** 2) / (punch_diameter_O * EB_punchO)
    area = math.pi * specimen.D_pa[d] ** 2 / 4
    if plat_check:
        return [1e6 * (KpI + KpO) * n_stress[q] * area / specimen.L_pa[d]
                - 2 * (n_stress[q] / plat_emod[d]) * (plat_Len[d] / specimen.L_pa[d]) for q in range(len(n_stress))]
    return [1e6 * (KpI + KpO) * n_stress[q] * area / specimen.L_pa[d] for q in range(len(n_stress))]


def _punch_correction_generic(specimen, d, d_eq_sp, punch_diameter_I, punch_diameter_O, bar_diameter_I, bar_diameter_O, nuB_punchI, nuB_punchO,
                               EB_punchI, EB_punchO, n_stress, plat_check, plat_emod, plat_Len):
    d_ratioI = punch_diameter_I / bar_diameter_I
    KpI = _elliptic_punch_factor(d_ratioI) * (16 / (3 * math.pi ** 2)) * (1 - nuB_punchI ** 2) / (punch_diameter_I * EB_punchI)
    d_ratioO = punch_diameter_O / bar_diameter_O
    KpO = _elliptic_punch_factor(d_ratioO) * (16 / (3 * math.pi ** 2)) * (1 - nuB_punchO ** 2) / (punch_diameter_O * EB_punchO)
    area = math.pi * specimen.D_pa[d] ** 2 / 4
    if plat_check:
        return [1e6 * (KpI + KpO) * n_stress[q] * area / specimen.L_pa[d]
                - 2 * (n_stress[q] / plat_emod[d]) * (plat_Len[d] / specimen.L_pa[d]) for q in range(len(n_stress))]
    return [1e6 * (KpI + KpO) * n_stress[q] * area / specimen.L_pa[d] for q in range(len(n_stress))]


def _punch_correction_shear(specimen, d, D_eq, D_eq1, punch_diameter_I, punch_diameter_O, bar_diameter_I, bar_diameter_O, nuB_punchI, nuB_punchO,
                             EB_punchI, EB_punchO, n_stress, plat_check, plat_emod, plat_Len):
    d_ratioI = punch_diameter_I / bar_diameter_I
    KpI = _elliptic_punch_factor(d_ratioI) * (16 / (3 * math.pi ** 2)) * (1 - nuB_punchI ** 2) / (D_eq * EB_punchI)
    d_ratioO = punch_diameter_O / bar_diameter_O
    KpO = _elliptic_punch_factor(d_ratioO) * (16 / (3 * math.pi ** 2)) * (1 - nuB_punchO ** 2) / (D_eq * EB_punchO)
    area = math.pi * specimen.D_pa[d] ** 2 / 4
    if plat_check:
        return [1e6 * (KpI + KpO) * n_stress[q] * area / specimen.L_pa[d]
                - 2 * (n_stress[q] / plat_emod[d]) * (plat_Len[d] / specimen.L_pa[d]) for q in range(len(n_stress))]
    return [1e6 * (KpI + KpO) * n_stress[q] * area / specimen.L_pa[d] for q in range(len(n_stress))]


def _process_shpb_task(context: dict, d: int) -> SHPBResult:
    """Worker entry point for core.parallel.run_map (see ui/pages/
    page_process.py's SHPB run handler): `context` holds every
    `process_shpb()` argument except the draw index `d` itself, so it can
    be pickled once per worker process (see `core/parallel.py`) instead of
    once per draw."""
    return process_shpb(d=d, **context)
