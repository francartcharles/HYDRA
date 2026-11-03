"""
Direct Impact processing engine.

Ported from process_DI.py: process_DI(). This is the single-output-bar
counterpart of process_shpb.py: instead of an incident+reflected wave
pair on an input bar, the specimen's incident velocity comes from a
separate measurement (`velocity_c`, e.g. a laser gate signal), and only
the transmitted wave on the output bar is used.

Ported as literally as possible (same variable names, branch order and
formulas) to keep it traceable against the original line-for-line. The
bar-temperature-gradient helper is shared with core/process_shpb.py since
it's already per-bar.

Removed: the `psutil` memory guard, the Tkinter progress-bar update, and
the dependence on `corr_file`/`import_bar_tab` module globals (now
explicit parameters).

NOT YET independently re-validated against real experimental data across
the full option matrix (heat x platen x punch x friction x inertia x
viscoelastic bar x 4 kinematics) -- exercised with synthetic and real
signals only on the simplest option combination so far. Cross-check
important results against the original application before relying on
this for a real analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.models import Bar, Specimen
from core.process_shpb import (
    _bar_gradient_wave_speed_modulus, _elliptic_punch_factor, _clamp_nominal_strain,
    _require_notched_geometry, _robust_dt, _index_of_nearest,
)


@dataclass
class DIOptions:
    type_check: int                 # 1 compression, 2 tension, 3 notched (Shear-Compression/Tension), 4 shear
    comp_check: int = 0
    tens_check: int = 0
    shear_check: int = 0             # 1 hat, 2 double, 3 simple
    tenscomp_check: int = 1
    heat_check: bool = False
    type_bars_check: int = 1
    plat_check: bool = False
    punch_check: bool = False
    inertia_check: bool = False
    disp_visco: bool = False
    disp_ela: bool = False
    sk_check: bool = False


@dataclass
class DIResult:
    force_output: list
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
    transmitted: np.ndarray


def process_di(
    options: DIOptions,
    OB: Bar, specimen: Specimen,
    velocity_c: np.ndarray, time_pulse: list, xd1: float, xd2: float,
    d: int,
    SG_unc_d: np.ndarray, f_inc_velocity: np.ndarray, T0_pa: np.ndarray, friction: np.ndarray,
    k_value: Optional[list] = None, sk_value: Optional[list] = None,
    plat_nu=None, plat_rho=None, plat_emod=None, plat_Dia=None, plat_Len=None,
    Tm_value: Optional[tuple] = None, T_gages_value: Optional[tuple] = None,
    material_table_ob=None,
    AA: Optional[list] = None,
) -> DIResult:
    """Process one Monte-Carlo draw `d` of a Direct Impact test. `OB`/
    `specimen` must already be Monte-Carlo-expanded, and `.expand_viscoelastic`
    too if `options.disp_visco`. `time_pulse` is a plain Python list (the
    original indexes it with `.index(round(xd1, 6))`; see `_index_of_nearest`
    for why this now uses a nearest-match lookup instead)."""

    # ---- Cutting waves --------------------------------------------------
    xdt = xd2 - xd1
    i_i1 = _index_of_nearest(time_pulse, xd1)
    i_i2 = _index_of_nearest(time_pulse, xd2)
    i_it = _index_of_nearest(time_pulse, xdt)

    time1 = np.asarray(time_pulse[i_i1:i_i2]) - xd1
    transmitted = OB.signal[i_i1:i_i2] * SG_unc_d[d]
    velocity = velocity_c[0:i_it]

    if options.disp_ela or options.disp_visco:
        transmitted = AA[0][:]

    # ---- Platen correction ----------------------------------------------
    if options.plat_check:
        plat_C0 = math.sqrt(plat_emod[d] / plat_rho[d])
        SBO = math.pi * (OB.D_pa[d] / 2) ** 2
        S_pla = math.pi * (plat_Dia[d] / 2) ** 2
        TPB_output = 2 * plat_C0 * plat_emod[d] * S_pla / (OB.C0_pa[d] * plat_emod[d] * S_pla + plat_C0 * OB.e_pa[d] * SBO)
        transmitted = transmitted / TPB_output
        E_mat_OB, C0_mat_OB, rho_mat_OB = plat_emod[d], plat_C0, plat_rho[d]
    else:
        rho_mat_OB = OB.rho_pa[d]
        E_mat_OB = C0_mat_OB = None  # set below

    # F_output is computed from strain measured by a gage mounted ON THE
    # BAR, using 1D wave theory (F = E_bar * A_bar * strain) -- that
    # relationship holds at the bar's own cross-section regardless of
    # what's beyond the bar's end, so DB_punch (the area term) must always
    # be the bar's own diameter, whether or not a platen is used. (Differs
    # from the original, which used the platen's diameter here when
    # plat_check -- that under-estimated F_output whenever platens were
    # enabled, since the platen is smaller than the bar.)
    DB_punch = OB.D_pa[d]

    # Punch (indentation) correction always happens at the BAR surface --
    # against the specimen directly when there's no platen, or against the
    # platen when platens are used -- so nuB_punch/EB_punch (used ONLY in
    # the punch correction, never in F_output above) are always the bar's
    # OWN properties; only the "punch diameter" (specimen vs platen)
    # changes, resolved at each punch-correction call site via
    # `punch_diameter` below, against `bar_diameter` (always OB.D_pa[d]).
    # (Differs from the original, which used the platen's own properties
    # for nuB_punch/EB_punch too -- changed on the user's explicit
    # direction after finding the original's assumption physically wrong.)
    nuB_punch, EB_punch = OB.nu_pa[d], OB.e_pa[d]
    bar_diameter = OB.D_pa[d]
    punch_diameter = plat_Dia[d] if options.plat_check else None  # None -> caller uses specimen/D_eq

    # ---- Bar-temperature-gradient (heat) correction ----------------------
    if options.heat_check:
        gage_t_ob = T_gages_value[1] if T_gages_value else 293.0
        e_mat_ob, c0_mat_ob, e_surface_ob = _bar_gradient_wave_speed_modulus(
            OB, d, gage_t_ob, T0_pa[d], options.type_bars_check, Tm_value, material_table_ob)
        if not options.plat_check:
            E_mat_OB, C0_mat_OB = e_mat_ob, c0_mat_ob
        C0OB = math.sqrt(e_surface_ob / OB.rho_pa[d])
        SBO = math.pi * (OB.D_pa[d] / 2) ** 2
        TPB_output = 2 * C0_mat_OB * E_mat_OB * SBO / (C0OB * E_mat_OB * SBO + C0_mat_OB * e_surface_ob * SBO)
        transmitted = transmitted / TPB_output
        # EB_punch is always the bar's own (heat-adjusted) modulus, regardless of plat_check.
        EB_punch = e_mat_ob

    if not options.heat_check:
        E_mat_OB, C0_mat_OB = OB.e_pa[d], OB.C0_pa[d]

    def _viscoelastic_modulus_and_speed():
        ft = time1[1] - time1[0]
        NN = 2 * len(transmitted) - 1
        freqi = np.fft.rfftfreq(NN, d=ft)
        E1, E2, n2 = OB.e11[d], OB.e22[d], OB.n22[d]
        EE11 = []
        for f in freqi:
            E = np.reciprocal((1 / E1) + (E2 / (E2 ** 2 + (2 * math.pi * f * n2) ** 2))
                               - 1j * ((2 * math.pi * f * n2) / (E2 ** 2 + (2 * math.pi * f * n2) ** 2)))
            EE11.append(E.real)
        CC22 = [math.sqrt(e / OB.rho_pa[d]) for e in EE11]
        return EE11, CC22

    # ------------------------------------------------------------------
    # type_check == 3: notched specimen (Shear-Compression/Tension)
    # ------------------------------------------------------------------
    if options.type_check == 3:
        _require_notched_geometry(specimen, d)
        sign = -1 if options.tenscomp_check == 1 else 1
        F_output = sign * (E_mat_OB * math.pi * (DB_punch / 2) ** 2) * transmitted * OB.CPG_pa[d]
        output_m = sign * transmitted * OB.CPG_pa[d]
        V_input = velocity * f_inc_velocity[d]

        dtime = time1[1] - time1[0]
        int_d = dint_eps = 0.0
        displacement1, int_eps = [], []
        for v, om in zip(V_input, output_m):
            int_d += v * dtime * 1000
            dint_eps += om * dtime
            int_eps.append(dint_eps)
            displacement1.append(int_d)
        displacement = [displacement1[i] - 1000 * C0_mat_OB * int_eps[i] for i in range(len(displacement1))]

        if options.disp_visco:
            EE11, CC22 = _viscoelastic_modulus_and_speed()
            F_output = [sign * (EE11[i] * math.pi * (DB_punch / 2) ** 2) * transmitted[i] * OB.CPG_pa[d] for i in range(len(transmitted))]
            int_d = dint_eps = 0.0
            displacement1, int_eps = [], []
            for v, om in zip(V_input, output_m):
                int_d += v * dtime * 1000
                dint_eps += om * dtime
                int_eps.append(dint_eps)
                displacement1.append(int_d)
            displacement = [displacement1[i] - 1000 * CC22[i] * int_eps[i] for i in range(len(displacement1))]

        denom = 2000 * specimen.re_pa[d] / math.cos(specimen.NoA_pa[d])
        if not options.sk_check:
            t_s = [
                float(k_value[0]) * (displacement[i] - 1000 * specimen.dy_pa[d]) / denom
                + float(k_value[1]) * ((displacement[i] - 1000 * specimen.dy_pa[d]) / denom) ** 2
                + float(k_value[2]) * ((displacement[i] - 1000 * specimen.dy_pa[d]) / denom) ** 3
                for i in range(len(displacement))
            ]
            t_s = [v - t_s[0] for v in t_s]
            t_stress = [
                float(k_value[3]) * (1 - float(k_value[4]) * t_s[i]) * (abs(F_output[i]) ** float(k_value[5]))
                / (1000 * specimen.th_pa[d] * 1000 * specimen.D_pa[d])
                for i in range(len(F_output))
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
            t_s = [v - t_s[0] for v in t_s]
            for i in range(len(F_output)):
                kv = k_value if displacement[i] <= float(sk_value[6]) else sk_value
                t_stress.append(
                    float(kv[3]) * (1 - float(kv[4]) * t_s[i]) * (abs(F_output[i]) ** float(kv[5]))
                    / (1000 * specimen.th_pa[d] * 1000 * specimen.D_pa[d])
                )

        displacement = [v / 1000 for v in displacement]
        t_sr = [(t_s[i + 1] - t_s[i]) / dtime for i in range(len(t_s) - 1)]
        t_srsr = [(t_sr[i + 1] - t_sr[i]) / dtime for i in range(len(t_sr) - 1)]
        n_sr = list(np.zeros(len(t_sr)))

        p_s = 0
        if options.punch_check:
            punch_d = punch_diameter if punch_diameter is not None else specimen.D_pa[d]
            d_ratio = punch_d / bar_diameter
            Kp = _elliptic_punch_factor(d_ratio) * (16 / (3 * math.pi ** 2)) * (1 - nuB_punch ** 2) / (punch_d * EB_punch)
            p_s = [2e6 * Kp * t_stress[i] * (math.pi * specimen.D_pa[d] ** 2 / 4) / specimen.D_pa[d] for i in range(len(t_stress))]
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

        transmitted = transmitted * OB.CPG_pa[d]
        n_s, n_stress = t_s, t_stress

    # ------------------------------------------------------------------
    # type_check in (1, 2): compression / tension
    # ------------------------------------------------------------------
    elif options.type_check in (1, 2):
        sign = -1 if options.type_check == 1 else 1
        output_m = sign * transmitted * OB.CPG_pa[d]
        V_input = velocity * f_inc_velocity[d]

        int_d = dint_eps = 0.0
        displacement, int_eps = [], []
        for i in range(len(time1) - 1):
            int_d += V_input[i] * (time1[i + 1] - time1[i])
            dint_eps += output_m[i] * (time1[i + 1] - time1[i])
            displacement.append(int_d)
            int_eps.append(dint_eps)
        n_s = _clamp_nominal_strain(
            [(displacement[i] - C0_mat_OB * int_eps[i]) / specimen.L_pa[d] for i in range(len(displacement))]
        )

        n_sr, t_s = [], []
        for h in range(len(n_s) - 1):
            n_sr.append((n_s[h + 1] - n_s[h]) / (time1[h + 1] - time1[h]))
            t_s.append(-math.log(1 - n_s[h]))

        F_output = [math.pi * (DB_punch / 2) ** 2 * rho_mat_OB * (C0_mat_OB ** 2) * output_m[j] for j in range(len(displacement))]

        if options.disp_visco:
            EE11, CC22 = _viscoelastic_modulus_and_speed()
            int_d = dint_eps = 0.0
            displacement, int_eps = [], []
            for i in range(len(time1) - 1):
                int_d += V_input[i] * (time1[i + 1] - time1[i])
                dint_eps += output_m[i] * (time1[i + 1] - time1[i])
                displacement.append(int_d)
                int_eps.append(dint_eps)
            n_s = _clamp_nominal_strain(
                [(displacement[i] - CC22[i] * int_eps[i]) / specimen.L_pa[d] for i in range(len(displacement))]
            )
            n_sr, t_s = [], []
            for h in range(len(n_s) - 1):
                n_sr.append((n_s[h + 1] - n_s[h]) / (time1[h + 1] - time1[h]))
                t_s.append(-math.log(1 - n_s[h]))
            F_output = [math.pi * (DB_punch / 2) ** 2 * rho_mat_OB * (CC22[j] ** 2) * output_m[j] for j in range(len(displacement))]

        D_eq_sp = None
        if options.tens_check == 1 or options.comp_check == 1:
            section = math.pi * (specimen.D_pa[d] / 2) ** 2
        else:
            section = specimen.wi_pa[d] * specimen.thi_pa[d]
            D_eq_sp = 2 * math.sqrt(section / math.pi)

        n_stress = np.asarray(F_output) / section / 1e6
        t_stress = [n_stress[i] * (1 - 2 * specimen.nu_pa[d] * n_s[i]) for i in range(len(n_stress))] if options.type_check == 1 \
            else [n_stress[i] * (1 + 2 * specimen.nu_pa[d] * n_s[i]) for i in range(len(n_stress))]

        p_s = 0
        if options.punch_check:
            t_stress, t_s2 = [], []
            d_eq = specimen.D_pa[d] if D_eq_sp is None else D_eq_sp
            punch_d = punch_diameter if punch_diameter is not None else d_eq
            d_ratio = punch_d / bar_diameter
            Kp = _elliptic_punch_factor(d_ratio) * (16 / (3 * math.pi ** 2)) * (1 - nuB_punch ** 2) / (punch_d * EB_punch)
            p_s = [2e6 * Kp * n_stress[q] * (math.pi * d_eq ** 2 / 4) / specimen.L_pa[d] for q in range(len(n_stress))]
            n_s = _clamp_nominal_strain([n_s[i] - p_s[i] for i in range(len(p_s))])
            for i in range(len(time1) - 1):
                t_s2.append(-math.log(1 - n_s[i]))
                t_stress.append(n_stress[i] * (1 - 2 * specimen.nu_pa[d] * n_s[i]))
            t_s = t_s2[:-1]

        s0 = specimen.L_pa[d] / (specimen.D_pa[d] if D_eq_sp is None else D_eq_sp)
        friction_stress = 0
        if friction[d] != 0:
            m = 1 - friction[d] / (3 * s0)
            n_stress = [m * v for v in n_stress]
            t_stress = [m * v for v in t_stress]
            friction_stress = [friction[d] / (3 * s0) * v for v in t_stress]

        n_sr = n_sr + [n_sr[-1]]
        t_s = t_s + [t_s[-1]]
        t_sr, n_srsr = [], []
        dt = _robust_dt(time1)
        for i in range(len(n_sr) - 1):
            t_sr_i = (t_s[i + 1] - t_s[i]) / dt
            n_srsr_i = (n_sr[i + 1] - n_sr[i]) / dt
            if t_sr_i == 0 and t_sr:
                t_sr_i = t_sr[-1]
            if n_srsr_i == 0 and n_srsr:
                n_srsr_i = n_srsr[-1]
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
            t_stress = [t_stress[i] - inertia_stress[i] for i in range(len(n_stress))]

        transmitted = transmitted * OB.CPG_pa[d]
        transmitted = transmitted[:-1]

    # ------------------------------------------------------------------
    # type_check == 4: shear (hat-shaped / double / simple)
    # ------------------------------------------------------------------
    else:
        sign = -1 if options.tenscomp_check == 1 else 1
        output_m = sign * transmitted * OB.CPG_pa[d]
        V_input = velocity * f_inc_velocity[d]

        int_d = dint_eps = 0.0
        displacement, int_eps = [], []
        for i in range(len(time1) - 1):
            int_d += V_input[i] * (time1[i + 1] - time1[i])
            dint_eps += output_m[i] * (time1[i + 1] - time1[i])
            displacement.append(int_d)
            int_eps.append(dint_eps)

        l_u = (specimen.R2_pa[d] - specimen.R1_pa[d]) if options.shear_check == 1 else specimen.wi_pa[d]
        n_s = [(displacement[i] - C0_mat_OB * int_eps[i]) / l_u for i in range(len(displacement))]

        if options.shear_check == 1:
            SAS = math.pi * (specimen.R1_pa[d] + specimen.R2_pa[d]) * (specimen.H2_pa[d] - specimen.H1_pa[d])
        elif options.shear_check == 2:
            SAS = 2 * specimen.thi_pa[d] * specimen.L_pa[d]
        else:
            SAS = specimen.thi_pa[d] * specimen.L_pa[d]

        n_sr, t_s = [], []
        for h in range(len(n_s) - 1):
            n_sr.append((n_s[h + 1] - n_s[h]) / (time1[h + 1] - time1[h]))
            t_s.append(n_s[h])

        F_output = [math.pi * (DB_punch / 2) ** 2 * rho_mat_OB * (C0_mat_OB ** 2) * output_m[j] for j in range(len(displacement))]

        if options.disp_visco:
            EE11, CC22 = _viscoelastic_modulus_and_speed()
            int_d = dint_eps = 0.0
            displacement, int_eps = [], []
            for i in range(len(time1) - 1):
                int_d += V_input[i] * (time1[i + 1] - time1[i])
                dint_eps += output_m[i] * (time1[i + 1] - time1[i])
                displacement.append(int_d)
                int_eps.append(dint_eps)
            n_s = [(displacement[i] - CC22[i] * int_eps[i]) / (specimen.R2_pa[d] - specimen.R1_pa[d]) for i in range(len(displacement))]
            SAS = math.pi * (specimen.R1_pa[d] + specimen.R2_pa[d]) * (specimen.H2_pa[d] - specimen.H1_pa[d])
            n_sr, t_s = [], []
            for h in range(len(n_s) - 1):
                n_sr.append((n_s[h + 1] - n_s[h]) / (time1[h + 1] - time1[h]))
                t_s.append(n_s[h])
            F_output = [math.pi * (DB_punch / 2) ** 2 * rho_mat_OB * (CC22[j] ** 2) * output_m[j] for j in range(len(displacement))]

        n_stress = np.asarray(F_output) / SAS / 1e6
        t_stress = list(n_stress)

        if options.shear_check == 1:
            D_eq1, D_eq2 = 2 * specimen.R3_pa[d], 2 * specimen.R1_pa[d]
        elif options.shear_check == 2:
            SS1, SS2 = specimen.thi_pa[d] * specimen.wiL_pa[d], 2 * specimen.thi_pa[d] * specimen.wiL_pa[d]
            D_eq1, D_eq2 = math.sqrt(4 * SS1 / math.pi), math.sqrt(4 * SS2 / math.pi)
        else:
            SS = specimen.thi_pa[d] * specimen.wiL_pa[d]
            D_eq1 = D_eq2 = math.sqrt(4 * SS / math.pi)
        D_eq = (D_eq1 + D_eq2) / 2

        p_s = 0
        if options.punch_check:
            t_stress, t_s2 = [], []
            punch_d_shear = punch_diameter if punch_diameter is not None else D_eq
            d_ratio = punch_d_shear / bar_diameter
            Kp = _elliptic_punch_factor(d_ratio) * (16 / (3 * math.pi ** 2)) * (1 - nuB_punch ** 2) / (D_eq * EB_punch)
            p_s = [2e6 * Kp * n_stress[q] * (math.pi * D_eq ** 2 / 4) / specimen.L_pa[d] for q in range(len(n_stress))]
            n_s = [n_s[i] - p_s[i] for i in range(len(p_s))]
            for i in range(len(time1) - 1):
                t_s2.append(n_s[i])
                t_stress.append(n_stress[i])
            t_s = t_s2[:-1]
            p_s = p_s[:-1]

        s0 = (specimen.H3_pa[d] if options.shear_check == 1 else specimen.leL_pa[d]) / D_eq
        friction_stress = 0
        if friction[d] != 0:
            m = 1 - friction[d] / (3 * s0)
            n_stress = [m * v for v in n_stress]
            t_stress = [m * v for v in t_stress]
            friction_stress = [friction[d] / (3 * s0) * v for v in t_stress]
            friction_stress = friction_stress[:-1]

        t_sr, n_srsr = [], []
        dt = _robust_dt(time1)
        for i in range(len(n_sr) - 1):
            t_sr.append((t_s[i + 1] - t_s[i]) / dt)
            n_srsr.append((n_sr[i + 1] - n_sr[i]) / dt)

        n_sr, t_sr, t_stress = n_sr[:-1], t_sr[:-1], t_stress[:-1]

        if options.shear_check == 1:
            D_eq2 = 2 * math.sqrt((specimen.R2_pa[d] + specimen.R1_pa[d]) * (specimen.H2_pa[d] - specimen.H1_pa[d]))
        elif options.shear_check == 2:
            D_eq2 = math.sqrt(8 * SAS / math.pi)
        else:
            D_eq2 = math.sqrt(4 * SAS / math.pi)

        inertia_stress = 0
        if options.inertia_check:
            c1 = specimen.rho_pa[d] * (D_eq2 ** 2) * (s0 ** 2 - 3 / 16) / 12
            c2 = 3 * specimen.rho_pa[d] * (D_eq2 ** 2) / 64
            inertia_stress = [c1 * (n_sr[i] ** 2 + n_srsr[i]) / 1e6 + c2 * n_srsr[i] / 1e6 for i in range(len(n_srsr))]
            inertia_stress.append(inertia_stress[-1])
            inertia_stress.append(inertia_stress[-1])
            t_stress = [t_stress[i] - inertia_stress[i] for i in range(len(t_stress))]
            inertia_stress = inertia_stress[:-1]

        transmitted = transmitted * OB.CPG_pa[d]

    n_sr = n_sr + [n_sr[-1]]
    t_sr = t_sr + [t_sr[-1]]
    time1 = time1[:-1]

    coeff_temp = specimen.xi_pa[d] / (specimen.cp_pa[d] * specimen.rho_pa[d])
    temperature = [T0_pa[d]]
    for i in range(len(t_s) - 1):
        temperature.append(temperature[-1] + coeff_temp * t_stress[i] * 1e6 * (t_s[i + 1] - t_s[i]))

    # `displacement` is in mm already for type_check == 3 (matches the
    # original, which multiplies by 1000 there). For type_check in (1, 2, 4)
    # the original leaves it in meters (it's used mid-calculation to derive
    # strain, so it can't just be rescaled inline without breaking that) --
    # converted to mm only here, at the very end, for consistency with the
    # notched branch and with common usage.
    displacement_mm = displacement if options.type_check == 3 else [1000 * v for v in displacement]

    return DIResult(
        force_output=list(F_output), displacement=displacement_mm, strain=list(t_s),
        nominal_strain=list(n_s), nominal_stress=list(n_stress), stress=list(t_stress),
        strain_rate=list(t_sr), nominal_strain_rate=list(n_sr), time=time1, temperature=temperature,
        inertia_stress=inertia_stress if isinstance(inertia_stress, list) else [],
        punch_strain=p_s if isinstance(p_s, list) else [],
        friction_stress=friction_stress if isinstance(friction_stress, list) else [],
        transmitted=transmitted,
    )


def _process_di_task(context: dict, d: int) -> DIResult:
    """Worker entry point for core.parallel.run_map (see ui/pages/
    page_process.py's Direct Impact run handler): `context` holds every
    `process_di()` argument except the draw index `d` itself, so it can be
    pickled once per worker process (see `core/parallel.py`) instead of
    once per draw."""
    return process_di(d=d, **context)
