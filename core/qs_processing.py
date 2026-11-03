"""
Quasi-static (Load-Displacement) stress/strain processing.

Ported from process_QS.py: process_QS(). The physics/formulas for each of
the four test kinematics are unchanged from the original:
    type_check == 1 -> compression (true strain)
    type_check == 2 -> tension (true strain)
    type_check == 3 -> notched specimen, Shear-Compression/Tension, fitted
                       k-coefficient polynomial (optionally split into two
                       regimes by displacement via sk_check/sk_value)
    type_check == 4 -> shear (hat-shaped, or double/simple)
Removed are the Tkinter progress-bar widget updates, the `psutil` memory
guard tied to a Windows messagebox `quit()`, and the implicit dependence
on a shared mutable `d` loop index into module-level specimen arrays.
Call this once per Monte-Carlo draw `d`; the caller loops and aggregates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.process_shpb import _clamp_nominal_strain, _require_notched_geometry


@dataclass
class QSResult:
    load: list
    displacement: list
    stress: list
    strain: list
    strain_rate: list
    temperature: list
    nominal_stress: list | None = None
    nominal_strain: list | None = None
    nominal_strain_rate: list | None = None


def process_qs(
    *, type_check: int, shear_check: int = 0,
    tens_check: int = 0, comp_check: int = 0,
    displacement: np.ndarray, load: np.ndarray, timet: np.ndarray,
    specimen, d: int,
    coeff_load: float, coeff_dd: float,
    t0: float,
    k_value: list[float] | None = None, sk_value: list[float] | None = None, sk_check: bool = False,
) -> QSResult:
    """Process one Monte-Carlo draw `d` of a quasi-static test.

    `specimen` exposes the Monte-Carlo-expanded arrays used by the
    original code directly (`specimen.D_pa[d]`, `specimen.re_pa[d]`, ...).
    """
    nominal_stress = nominal_strain = nominal_strain_rate = None

    if type_check in (1, 2):
        section = (
            math.pi * (specimen.D_pa[d] / 2) ** 2
            if tens_check == 1 or comp_check == 1
            else specimen.wi_pa[d] * specimen.thi_pa[d]
        )
        n_s = _clamp_nominal_strain(
            [coeff_dd * displacement[g] / (1000 * specimen.L_pa[d]) for g in range(len(displacement))]
        )
        n_stress = [coeff_load * load[j] / (section * 1e6) for j in range(len(load))]

        if type_check == 1:  # compression: logarithmic (true) strain convention
            t_s = [-math.log(1 - n_s[i]) for i in range(len(n_s))]
            t_stress = [n_stress[k] * (1 - 2 * specimen.nu_pa[d] * n_s[k]) for k in range(len(n_stress))]
        else:  # tension
            t_s = [math.log(1 + n_s[i]) for i in range(len(n_s))]
            t_stress = [n_stress[k] * (1 + 2 * specimen.nu_pa[d] * n_s[k]) for k in range(len(n_stress))]

        t_sr = [(t_s[i + 1] - t_s[i]) / (timet[i + 1] - timet[i]) for i in range(len(t_s) - 1)]
        n_sr = [(n_s[i + 1] - n_s[i]) / (timet[i + 1] - timet[i]) for i in range(len(n_s) - 1)]
        full_disp = [coeff_dd * v for v in displacement]
        full_load = [coeff_load * v for v in load]
        nominal_stress, nominal_strain, nominal_strain_rate = n_stress, n_s, n_sr

    elif type_check == 3:
        # Notched specimen (Shear-Compression/Tension), fitted k-coefficient
        # polynomial. `dy_pa` (displacement at yield) offsets the strain
        # proxy only in the un-split (sk_check == 0) branch, matching
        # process_QS.py.
        _require_notched_geometry(specimen, d)
        dd = list(displacement)
        denom = 2000 * specimen.re_pa[d] / math.cos(specimen.NoA_pa[d])

        if not sk_check:
            t_s = [
                float(k_value[0]) * (coeff_dd * dd[i] - specimen.dy_pa[d]) / denom
                + float(k_value[1]) * ((coeff_dd * dd[i] - specimen.dy_pa[d]) / denom) ** 2
                + float(k_value[2]) * ((coeff_dd * dd[i] - specimen.dy_pa[d]) / denom) ** 3
                for i in range(len(dd))
            ]
            t_stress = [
                float(k_value[3]) * (1 - float(k_value[4]) * t_s[k])
                * (coeff_load * abs(load[k]) ** float(k_value[5]))
                / (1000 * specimen.th_pa[d] * 1000 * specimen.D_pa[d])
                for k in range(len(load))
            ]
        else:
            t_s, t_stress = [], []
            for i in range(len(dd)):
                kv = k_value if dd[i] <= float(sk_value[6]) else sk_value
                t_s.append(
                    float(kv[0]) * (coeff_dd * dd[i]) / denom
                    + float(kv[1]) * ((coeff_dd * dd[i]) / denom) ** 2
                    + float(kv[2]) * ((coeff_dd * dd[i]) / denom) ** 3
                )
            for i in range(len(load)):
                kv = k_value if dd[i] <= float(sk_value[6]) else sk_value
                t_stress.append(
                    float(kv[3]) * (1 - float(kv[4]) * t_s[i])
                    * (coeff_load * abs(load[i]) ** float(kv[5]))
                    / (1000 * specimen.th_pa[d] * 1000 * specimen.D_pa[d])
                )

        t_s = [v - t_s[0] for v in t_s]
        t_sr = [(t_s[i + 1] - t_s[i]) / (timet[i + 1] - timet[i]) for i in range(len(t_s) - 1)]
        full_disp = [coeff_dd * v for v in displacement]
        full_load = [coeff_load * v for v in load]

    elif type_check == 4:
        if shear_check == 1:  # hat-shaped
            sas = math.pi * (specimen.R1_pa[d] + specimen.R2_pa[d]) * (specimen.H2_pa[d] - specimen.H1_pa[d])
            l0 = specimen.R2_pa[d] - specimen.R1_pa[d]
        else:  # double (2) or simple (3) shear
            sas = (2 if shear_check == 2 else 1) * specimen.thi_pa[d] * specimen.L_pa[d]
            l0 = specimen.wi_pa[d]
        t_s = [coeff_dd * v / (1000 * l0) for v in displacement]
        t_stress = [(coeff_load * load[k] / sas) / 1e6 for k in range(len(load))]
        t_sr = [(t_s[i + 1] - t_s[i]) / (timet[i + 1] - timet[i]) for i in range(len(t_s) - 1)]
        full_disp = [coeff_dd * v for v in displacement]
        full_load = [coeff_load * v for v in load]
    else:
        raise ValueError(f"Unknown type_check={type_check}")

    # Adiabatic temperature-rise model (Taylor-Quinney), same for every kinematics.
    coeff_temp = specimen.xi_pa[d] / (specimen.cp_pa[d] * specimen.rho_pa[d])
    temperature = [t0]
    for i in range(len(t_s) - 1):
        temperature.append(temperature[-1] + coeff_temp * t_stress[i] * 1e6 * (t_s[i + 1] - t_s[i]))

    return QSResult(
        load=full_load, displacement=full_disp, stress=t_stress, strain=t_s,
        strain_rate=t_sr, temperature=temperature,
        nominal_stress=nominal_stress, nominal_strain=nominal_strain, nominal_strain_rate=nominal_strain_rate,
    )


def _process_qs_task(context: dict, item: tuple) -> "QSResult":
    """Worker entry point for core.parallel.run_map (see ui/pages/
    page_process.py's Load/Displacement Data run handler). `item` is
    `(d, coeff_load, coeff_dd)` -- the three per-draw values -- and
    `context` holds every other `process_qs()` argument, pickled once per
    worker process (see `core/parallel.py`) instead of once per draw."""
    d, coeff_load, coeff_dd = item
    return process_qs(d=d, coeff_load=coeff_load, coeff_dd=coeff_dd, **context)
