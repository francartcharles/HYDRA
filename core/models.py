"""
Data model for bars and specimens.

The legacy code kept every property as a loose module-level global
(`test_type.sp_Tempa_value`, `corr_file.e1`, ...) and passed signals around
as bare numpy arrays on ad-hoc objects. Here everything a bar/specimen needs
is grouped into one real object so pages/core functions can be tested and
reused without importing half a dozen GUI modules for their side effects.

Field names ending in `_pa` are the Monte-Carlo-expanded arrays (N draws),
matching the convention already used in the original numerical code
(`OB.D_pa`, `IB.rho_pa`, ...) so the ported FFT-dispersion / processing
functions did not need to be renamed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from core.mc import MCParameterSet, Distribution
from core.lowpass import apply_lowpass_filter, DEFAULT_LOWPASS_CUTOFF_HZ


def _mc_array(pset: MCParameterSet, key: str, n: int, rng=None) -> np.ndarray:
    return pset[key].sample(n, rng)


@dataclass
class Bar:
    """Input bar (IB) or output bar (OB)."""
    name: str = ""
    params: MCParameterSet = field(default_factory=lambda: MCParameterSet("Bar"))
    signal: np.ndarray = field(default_factory=lambda: np.array([]))
    # Unmodified loaded gage signal (after optional downsampling, BEFORE
    # sign/offset): `signal = signal_raw * sign - offset`. Set by BarPage
    # on file load, or by core.setup_profile when a full-test JSON with
    # embedded bar signals is loaded; empty if only `signal` is known.
    signal_raw: np.ndarray = field(default_factory=lambda: np.array([]))
    sign: float = 1.0     # +1 or -1 -- current sign multiplier applied to the raw gage signal
    offset: float = 0.0   # current additive zero-offset subtracted from the (sign-corrected) raw signal
    # Optional low-pass filter (see core.lowpass.apply_lowpass_filter),
    # applied on top of `signal` (already sign/offset corrected -- see
    # ui/pages/page_bar.py's `_apply_sign_offset`) by `filtered_signal()`
    # below. Set from the Dispersion Corrections page ("Signal filtering"
    # card), independently for the input and output bar.
    lowpass_enabled: bool = False
    lowpass_cutoff_hz: float = DEFAULT_LOWPASS_CUTOFF_HZ

    # Monte-Carlo-expanded arrays, populated by `expand()`
    rho_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    C0_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    D_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    e_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    L_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    PG_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    nu_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    CPG_pa: np.ndarray = field(default_factory=lambda: np.array([]))   # alias of SG_coeff_pa (process_SHPB/DI's name for it)
    # Per-draw viscoelastic (Standard Linear Solid) bar constants, only
    # populated when a viscoelastic dispersion correction is used
    # upstream (see Bar.expand_viscoelastic below).
    e11: np.ndarray = field(default_factory=lambda: np.array([]))
    e22: np.ndarray = field(default_factory=lambda: np.array([]))
    n22: np.ndarray = field(default_factory=lambda: np.array([]))

    # Canonical parameter order, confirmed against real saved bar config
    # files (each line = one parameter, in this order): density, wave
    # velocity, elastic modulus, diameter, length, gage position, gage
    # coefficient, Poisson ratio. Used by core/param_io.py to load/save
    # these files -- keep in sync if this order ever changes.
    PARAM_ORDER = ["rho", "C0", "e", "D", "L", "PG", "SG_coeff", "nu"]

    @classmethod
    def new(cls, name: str) -> "Bar":
        pset = MCParameterSet(name)
        pset.add("rho", "Density", "kg/m^3")
        pset.add("C0", "Wave Velocity", "m/s")
        pset.add("e", "Elastic Modulus", "Pa")
        pset.add("D", "Diameter", "m")
        pset.add("L", "Length", "m")
        pset.add("PG", "Strain Gage Bridge Position", "m")
        pset.add("SG_coeff", "Strain Gage Bridge Coefficient", "1/V")
        pset.add("nu", "Poisson Ratio", "-")
        return cls(name=name, params=pset)

    def filtered_signal(self, dt: float | None) -> np.ndarray:
        """`self.signal` (already sign/offset corrected), additionally
        low-pass filtered if `lowpass_enabled` is set -- see
        core.lowpass.apply_lowpass_filter for the filter itself and its
        fail-soft behavior (invalid cutoff/dt/too-short signal -> returned
        unfiltered). Pure/non-mutating: always derived fresh from
        `self.signal`, never written back to it, so toggling the filter
        off then on again can't accumulate repeated filtering. `dt` is
        not stored on `Bar` (the shared acquisition time vector lives on
        whichever page/run is using this bar, e.g. `page_corrections.py`'s
        own `self._timet`) -- pass `None` to skip filtering when it isn't
        known."""
        if not self.lowpass_enabled or len(self.signal) == 0:
            return self.signal
        return apply_lowpass_filter(self.signal, dt, self.lowpass_cutoff_hz)

    def expand(self, n: int, rng=None) -> None:
        """Populate the *_pa Monte-Carlo arrays from the current parameters."""
        self.rho_pa = _mc_array(self.params, "rho", n, rng)
        self.C0_pa = _mc_array(self.params, "C0", n, rng)
        self.D_pa = _mc_array(self.params, "D", n, rng)
        self.e_pa = _mc_array(self.params, "e", n, rng)
        self.L_pa = _mc_array(self.params, "L", n, rng)
        self.PG_pa = _mc_array(self.params, "PG", n, rng)
        self.nu_pa = _mc_array(self.params, "nu", n, rng)
        # CPG_pa is process_SHPB.py/process_DI.py's name for the strain-gage
        # bridge coefficient's Monte-Carlo array (confirmed in process_t.py:
        # `IB.CPG_pa = np.random.normal(float(IB_props[6][0]), ...)`, and
        # IB_props[6] is exactly the SG_coeff slot in Bar.PARAM_ORDER).
        self.CPG_pa = _mc_array(self.params, "SG_coeff", n, rng)

    def expand_viscoelastic(self, n: int, e1: float, e1_std: float, e2: float, e2_std: float,
                             n2: float, n2_std: float, rng=None) -> None:
        """Populate e11/e22/n22 -- the per-draw Standard-Linear-Solid bar
        constants used by process_SHPB.py/process_DI.py's `disp_visco == 1`
        branch. Only needed when the bars are viscoelastic (see
        core.project.ViscoelasticSettings)."""
        rng = rng or np.random.default_rng()
        self.e11 = rng.normal(e1, e1_std, size=n) if e1_std else np.full(n, e1)
        self.e22 = rng.normal(e2, e2_std, size=n) if e2_std else np.full(n, e2)
        self.n22 = rng.normal(n2, n2_std, size=n) if n2_std else np.full(n, n2)


# Test type taxonomy, verified line-for-line against test_type.py's spec_frame()
# and import_spec_file() (which reads/validates a specimen file by its
# expected line count per mode). There are actually FOUR top-level test
# modes, not three -- "Shear-Compression/Tension" (a notched specimen with
# fitted k-coefficients, process_t's `type_check == 3`) is a distinct mode
# from plain "Shear" (type_check == 4, hat/double/simple).
TEST_MODES = ["Compression", "Tension", "Shear-Compression/Tension (Notched)", "Shear"]
COMPRESSION_GEOMETRIES = ["Cylindrical", "Square/Rectangular"]
TENSION_GEOMETRIES = ["Axisymmetric", "Flat"]
SHEAR_GEOMETRIES = ["Hat-shaped", "Double shear", "Simple shear"]
LOADING_REGIMES = ["Load/Displacement Data", "Hopkinson Bars (SHPB)", "Direct Impact"]

# Exact parameter order for each (mode, geometry) combination, verified
# against test_type.py: import_spec_file() field-by-field (including its
# own line-count validation, e.g. "if l_check != 7: showerror(...)").
# `None` as the geometry key means the mode has no sub-geometry choice.
SPECIMEN_FIELD_ORDER: dict[tuple[str, str | None], list[str]] = {
    ("Compression", "Cylindrical"): ["rho", "D", "L", "e", "nu", "xi", "cp"],
    ("Compression", "Square/Rectangular"): ["rho", "thickness", "L", "width", "e", "nu", "xi", "cp"],
    ("Tension", "Axisymmetric"): ["rho", "D", "L", "e", "nu", "xi", "cp"],
    ("Tension", "Flat"): ["rho", "thickness", "L", "width", "e", "nu", "xi", "cp"],
    ("Shear-Compression/Tension (Notched)", None): [
        "rho", "D", "e", "nu", "xi", "cp",
        "effective_zone", "disp_yield", "notch_radius", "total_length", "notch_angle",
    ],
    ("Shear", "Hat-shaped"): ["rho", "e", "nu", "xi", "cp", "H1", "H2", "H3", "R1", "R2", "R3", "total_length"],
    ("Shear", "Double shear"): ["rho", "thickness", "L", "width", "e", "nu", "xi", "cp", "head_width", "total_length"],
    ("Shear", "Simple shear"): ["rho", "thickness", "L", "width", "e", "nu", "xi", "cp", "head_width", "total_length"],
}


@dataclass
class Specimen:
    """Specimen geometry + thermal properties.

    Which of the geometric fields are meaningful depends on `test_mode` /
    `geometry` (see SPECIMEN_FIELD_ORDER above), mirroring the original
    spec_frame() layout which showed a different subset of entries per
    selected radio button.

    The *_pa array attribute names below are deliberately identical to
    the ones process_QS.py / process_SHPB.py / process_DI.py read
    directly (confirmed by grepping all three for `specimen.*_pa`) so
    those engines can be pointed at this object with zero renaming.
    """
    name: str = "specimen"
    test_mode: str = "Compression"
    geometry: str | None = "Cylindrical"
    loading_regime: str = "Hopkinson Bars (SHPB)"

    params: MCParameterSet = field(default_factory=lambda: MCParameterSet("Specimen"))

    rho_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    e_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    D_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    L_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    nu_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    thi_pa: np.ndarray = field(default_factory=lambda: np.array([]))    # general "thickness"
    wi_pa: np.ndarray = field(default_factory=lambda: np.array([]))     # general "width"
    xi_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    cp_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    th_pa: np.ndarray = field(default_factory=lambda: np.array([]))    # notched: "effective zone thickness"
    re_pa: np.ndarray = field(default_factory=lambda: np.array([]))    # notched: notch radius
    NoA_pa: np.ndarray = field(default_factory=lambda: np.array([]))   # notched: notch angle
    dy_pa: np.ndarray = field(default_factory=lambda: np.array([]))    # notched: displacement at yield
    H1_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    H2_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    H3_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    R1_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    R2_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    R3_pa: np.ndarray = field(default_factory=lambda: np.array([]))
    wiL_pa: np.ndarray = field(default_factory=lambda: np.array([]))   # shear double/simple: "specimen head width"
    leL_pa: np.ndarray = field(default_factory=lambda: np.array([]))   # shear double/simple: "specimen total length"

    @classmethod
    def new(cls, name: str = "specimen") -> "Specimen":
        pset = MCParameterSet(name)
        pset.add("rho", "Density", "kg/m^3")
        pset.add("D", "Diameter", "m")
        pset.add("L", "Length", "m")
        pset.add("thickness", "Thickness", "m")
        pset.add("width", "Width", "m")
        pset.add("e", "Elastic Modulus", "Pa")
        pset.add("nu", "Poisson Ratio", "-")
        pset.add("xi", "Taylor-Quinney Coefficient", "-")
        pset.add("cp", "Specific Heat", "J/(kg*K)")
        # Notched specimen (Shear-Compression/Tension, type_check == 3)
        pset.add("effective_zone", "Effective Zone Thickness", "m")
        pset.add("disp_yield", "Displacement at Yielding", "m")
        pset.add("notch_radius", "Notch Radius", "m")
        pset.add("notch_angle", "Notch Angle", "deg")
        pset.add("total_length", "Total Length", "m")
        # Shear: hat-shaped
        pset.add("H1", "H1", "m")
        pset.add("H2", "H2", "m")
        pset.add("H3", "H3", "m")
        pset.add("R1", "R1", "m")
        pset.add("R2", "R2", "m")
        pset.add("R3", "R3", "m")
        # Shear: double/simple
        pset.add("head_width", "Specimen Head Width", "m")
        return cls(name=name, params=pset)

    def expand(self, n: int, rng=None) -> None:
        self.rho_pa = _mc_array(self.params, "rho", n, rng)
        self.e_pa = _mc_array(self.params, "e", n, rng)
        self.D_pa = _mc_array(self.params, "D", n, rng)
        self.L_pa = _mc_array(self.params, "L", n, rng)
        self.nu_pa = _mc_array(self.params, "nu", n, rng)
        self.thi_pa = _mc_array(self.params, "thickness", n, rng)
        self.wi_pa = _mc_array(self.params, "width", n, rng)
        self.xi_pa = _mc_array(self.params, "xi", n, rng)
        self.cp_pa = _mc_array(self.params, "cp", n, rng)
        self.th_pa = _mc_array(self.params, "effective_zone", n, rng)
        self.re_pa = _mc_array(self.params, "notch_radius", n, rng)
        self.NoA_pa = _mc_array(self.params, "notch_angle", n, rng)
        self.dy_pa = _mc_array(self.params, "disp_yield", n, rng)
        self.H1_pa = _mc_array(self.params, "H1", n, rng)
        self.H2_pa = _mc_array(self.params, "H2", n, rng)
        self.H3_pa = _mc_array(self.params, "H3", n, rng)
        self.R1_pa = _mc_array(self.params, "R1", n, rng)
        self.R2_pa = _mc_array(self.params, "R2", n, rng)
        self.R3_pa = _mc_array(self.params, "R3", n, rng)
        self.wiL_pa = _mc_array(self.params, "head_width", n, rng)
        self.leL_pa = _mc_array(self.params, "total_length", n, rng)
