"""
Strain Gage Bridge Coefficient ("Coefficient Pont de Gauge") calculator,
ported from test_type.py's `CPG_comp`/`CPG_comp_value`.

Converts the raw strain-gage bridge signal (voltage, after amplifier gain)
into strain via `strain = CPG * V_bridge`, so `CPG` is what a `Bar`'s
"Strain Gage Bridge Coefficient" (`SG_coeff`, aliased `CPG_pa` in the
Monte-Carlo expansion -- see core/models.py) actually holds. This module
computes that coefficient (and its uncertainty) from bridge-level physical
inputs, for whichever of the three standard Wheatstone bridge wiring
options was used:

- Quarter-bridge: 1 active gage, 3 fixed completion resistors.
- Half-bridge: 2 active gages (axial + lateral, exploiting the bar's own
  Poisson ratio for a lateral-strain compensation term).
- Full-bridge: 4 active gages (2 axial + 2 lateral).

Formulas are ported as literally as possible from the original for
traceability; NOT independently re-validated against the original
application's own output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class BridgeType(Enum):
    QUARTER = "quarter"
    HALF = "half"
    FULL = "full"


BRIDGE_LABELS = {
    BridgeType.QUARTER: "Quarter-Bridge",
    BridgeType.HALF: "Half-Bridge",
    BridgeType.FULL: "Full-Bridge",
}

# Whether the Poisson-ratio compensation term (and its uncertainty) applies
# -- only quarter-bridge has no lateral gage to compensate for.
USES_POISSON_RATIO = {
    BridgeType.QUARTER: False,
    BridgeType.HALF: True,
    BridgeType.FULL: True,
}

_REFERENCE_TEMPERATURE = 293.0  # K -- gage factor is specified at this temperature

# Poisson ratio Micro-Measurements/Vishay uses to calibrate a gage's stated
# gage factor (Tech Note TN-509, "Errors Due to Transverse Sensitivity in
# Strain Gages"): F = Fa * (1 - nu0*Kt), where F is the manufacturer's
# stated gage factor, Fa the gage's true axial-only gage factor, and Kt its
# transverse sensitivity coefficient.
TRANSVERSE_NU0_DEFAULT = 0.285

# Manufacturer thermal-output polynomial degree range (package inserts
# typically print one coefficient per order, 3rd through 7th) -- shared by
# every "Thermal Output Correction" UI in the app: this page's own static,
# single-temperature correction below, and Custom Sensors' time-varying
# one (core.custom_sensor.CustomSensor.thermal_offset_strain), which reuses
# thermal_output_strain() itself pointwise over a temperature profile.
THERMAL_MIN_DEGREE = 3
THERMAL_MAX_DEGREE = 7
THERMAL_DEFAULT_DEGREE = 4


@dataclass
class CPGResult:
    value: float  # coefficient, 1/V
    std: float


def compute_cpg(
    bridge_type: BridgeType,
    gf: float, gf_std: float,
    vex: float, vex_std: float,
    gain: float, gain_std: float,
    temperature: float, temperature_std: float,
    sg_t_pct: float,
    nub: float = 0.0, nub_std: float = 0.0,
) -> CPGResult:
    """Computes the strain-gage bridge coefficient (1/V) and its standard
    deviation.

    `sg_t_pct` is the gage factor's temperature coefficient, in %/K (as
    entered by the user -- divided by 100 internally, matching the
    original's `SG_T = value/100`). `nub`/`nub_std` (bar Poisson ratio) are
    only used -- and only required to be nonzero -- for half-/full-bridge.
    """
    if gf == 0 or vex == 0 or gain == 0 or temperature == 0:
        raise ValueError("Gage Factor, Input Tension, Gain and Temperature must all be nonzero.")

    sg_t = sg_t_pct / 100.0
    thermal = gf * (1 + sg_t * (temperature - _REFERENCE_TEMPERATURE)) * vex

    if bridge_type == BridgeType.QUARTER:
        value = (4 / thermal) / gain
        rel_sq = (
            (gf_std / gf) ** 2 + (vex_std / vex) ** 2
            + (gain_std / gain) ** 2 + (temperature_std / temperature) ** 2
        )
    else:
        if nub == 0:
            raise ValueError("Bar Poisson Ratio must be nonzero for half-/full-bridge.")
        factor = 4 if bridge_type == BridgeType.HALF else 2
        value = (factor / (thermal * (1 + nub))) / gain
        rel_sq = (
            (gf_std / gf) ** 2 + (vex_std / vex) ** 2
            + (gain_std / gain) ** 2 + (temperature_std / temperature) ** 2
            + (nub_std / nub) ** 2
        )

    std = value * math.sqrt(rel_sq)
    return CPGResult(value=value, std=std)


def transverse_sensitivity_correction(
    result: CPGResult,
    kt_pct: float, kt_pct_std: float,
    nu: float, nu_std: float,
    nu0: float = TRANSVERSE_NU0_DEFAULT,
) -> tuple[CPGResult, float]:
    """Scales a bridge coefficient to correct for the strain gage's own
    transverse sensitivity Kt (%, read off its datasheet), for a single
    axial gage under a uniaxial stress field -- this app's usual case, one
    active gage sensing strain along a bar's axis. Applied uniformly here
    regardless of bridge type, as a first-order approximation (this
    module's formulas are ported/documented, not independently validated
    against a reference implementation -- see the module docstring).

    Because `strain = CPG * V_bridge`, scaling the coefficient by the same
    factor that would correct the strain scales every strain derived from
    it identically, so the correction is folded directly into the
    coefficient rather than needing to be applied downstream, per sample.

    Derived from two relations in Vishay Tech Note TN-509 ("Errors Due to
    Transverse Sensitivity in Strain Gages"): the manufacturer's stated
    gage factor F relates to the gage's true axial-only gage factor Fa by
    F = Fa*(1 - nu0*Kt) (the gage is itself calibrated on a uniaxial
    specimen, where transverse strain = -nu0 * axial strain); and, for
    this app's own uniaxial stress field on the bar being measured
    (transverse strain = -nu * axial strain, with nu the bar's actual
    Poisson ratio), the true axial strain relates to the indicated one by
    strain_true = strain_indicated * (1 - nu0*Kt) / (1 - nu*Kt).

    `nu`/`nu_std` is this app's existing "Bar Poisson Ratio" input (the
    actual Poisson ratio of the bar under test); `nu0` is the Poisson
    ratio Vishay/Micro-Measurements calibrates gage factors against
    (0.285, TN-509) -- a fixed reference constant, not a measured
    quantity, so it carries no uncertainty of its own here.

    Returns the corrected coefficient/std alongside the plain correction
    factor (for display).
    """
    kt = kt_pct / 100.0
    kt_std = kt_pct_std / 100.0
    denom = 1 - nu * kt
    if denom == 0:
        raise ValueError(
            "Transverse sensitivity correction is singular for this "
            "Transverse Sensitivity / Bar Poisson Ratio combination."
        )
    factor = (1 - nu0 * kt) / denom

    # d(factor)/d(kt) and d(factor)/d(nu), both in closed form -- the
    # nu0*nu*kt cross-terms cancel in the first derivative.
    dfactor_dkt = (nu - nu0) / denom ** 2
    dfactor_dnu = (1 - nu0 * kt) * kt / denom ** 2
    factor_var = (dfactor_dkt * kt_std) ** 2 + (dfactor_dnu * nu_std) ** 2

    value = result.value * factor
    rel_sq = 0.0
    if result.value != 0:
        rel_sq += (result.std / result.value) ** 2
    if factor != 0:
        rel_sq += factor_var / factor ** 2
    std = abs(value) * math.sqrt(rel_sq)
    return CPGResult(value=value, std=std), factor


def thermal_output_strain(coeffs: list[float], t_ref: float, t_op: float) -> float:
    """Apparent strain (microstrain) produced by the strain gage's own
    thermal output alone (no mechanical load), from the manufacturer's
    thermal-output polynomial for that gage/specimen-material pairing
    (`coeffs[i]` is the order-`i` coefficient, printed on the gage
    package, in microstrain per unit^i, `unit` being whichever of
    Fahrenheit/Celsius the coefficients were given in).

    Per Vishay Tech Note TN-504-1 ("Strain Gage Thermal Output and Gage
    Factor Variation with Temperature"), eq. 3: the strain to subtract
    from an indicated reading is the polynomial's value at the current
    operating temperature MINUS its value at the temperature the bridge
    was balanced/zeroed at (`t_ref`) -- not just the polynomial evaluated
    at the operating temperature alone, since any thermal output already
    present at the balance temperature is already zeroed out by the
    balancing itself.
    """
    def _poly(t: float) -> float:
        return sum(c * t ** i for i, c in enumerate(coeffs))

    return _poly(t_op) - _poly(t_ref)
