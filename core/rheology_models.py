"""
Rheological (constitutive) models available for the viscoelastic bar
calibration (core/visco_calibration.py, core/bayes_visco_calibration.py).

Each model supplies only its own frequency-domain complex modulus E*(f);
everything downstream of that -- the Lame-like mu/lambda, the per-frequency
Pochhammer-Chree root search, the FFT-domain pulse propagation in
core/visco_calibration.py's `_predict_pulse` -- is generic in E*(f) and
does not otherwise depend on which rheological model produced it. That
seam is what makes it possible to offer several models here without
duplicating the (expensive, delicate) dispersion physics per model.

The bar's Poisson ratio `nu` is deliberately NOT one of a model's
parameters: it is a separate, always-present property of the bar material,
independent of which relaxation model is used for its modulus.

Four models are offered, from simplest to most representative of real
(especially amorphous/semi-crystalline) polymers such as PMMA, polycarbonate
or nylon -- materials a single Debye relaxation (the Standard Linear Solid
below) often fits only loosely over the strain-rate range of a typical SHPB
test, because their actual loss behaviour spans a broader, less symmetric
spectrum:

- Kelvin-Voigt (2 params: E, eta) -- simplest damped-elastic baseline.
- Standard Linear Solid / Zener (3 params: E1, E2, n2) -- single relaxation
  time (a spring E1 in series with a Kelvin-Voigt unit E2||n2 -- the
  "3-parameter Kelvin" representation; see the note on `_sls_modulus`
  below, its formula was re-derived and checked numerically against both
  classical SLS topologies to pin down which one it actually is, since the
  app's own existing code comments describe the other one); the model this
  app originally shipped with, and the only one `core/dispersion.py`'s
  "Viscoelastic bars" correction currently implements (see the note on
  `MODEL_SLS` below).
- Generalized Maxwell / Wiechert, 2 branches (5 params: E_inf, E1, n1, E2,
  n2) -- two relaxation times ("order 2"): a broader, asymmetric relaxation
  spectrum, closer to what PMMA/PC/nylon typically show than a single
  Debye term.
- Burgers (4 params: E1, n1, E2, n2) -- a Maxwell element in series with a
  Kelvin-Voigt element (a literal combination of the two simplest models):
  captures instantaneous elasticity + retarded (creep) elasticity + genuine
  long-term viscous flow, which matters for thermoplastics like nylon under
  sustained or repeated loading and which none of the three "solid" models
  above (all of which have a finite long-term modulus) can represent.

`core/dispersion.py`'s `viscoelastic_correction`/`viscoelastic_correction_direct_impact`
(the "Viscoelastic bars" Dispersion Corrections option) are generic over
all four models here, via `model.complex_modulus(freq, params)` -- the
Dispersion Corrections page offers the same model picker as the
calibration page below, and `core/project.py`'s `ViscoelasticSettings`
stores a generic `{model_key, params, params_std}` rather than fixed
`e1/e2/n2` fields (a `migrate_viscoelastic_dict` upgrades any older
session/profile file saved before this). The calibration page's "Apply to
Dispersion Corrections" button works for whichever of the four models
produced the calibration result, not just Standard Linear Solid.

NOTE: the Monte-Carlo bar expansion in `core/models.py`
(`Bar.expand_viscoelastic`) is a SEPARATE, unrelated code path (feeding
Monte-Carlo uncertainty propagation elsewhere in the app, not the
Dispersion Corrections page) and is still hard-coded to the Standard
Linear Solid modulus -- extending it to the other models would be its own,
separate change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

_EPS_FREQ = 1e-6  # Hz -- avoids a literal division by zero at the DC bin
# for a model (Burgers) whose long-term modulus is genuinely zero.
#
# IMPORTANT: this must NOT be applied to freq=0 for a model that already
# has a perfectly well-defined (finite, exactly real) modulus there
# (Kelvin-Voigt, SLS, Generalized Maxwell all do -- see each function's
# math). core/visco_calibration.py's `_predict_pulse` uses this bin's
# modulus to build a reference wave speed `c[0]` that it then reuses,
# UNPERTURBED, throughout its phase-correction formula for every frequency
# bin, and that formula's final step (`cmath.rect`) requires the result to
# come out exactly real -- Python refuses to silently drop even a
# vanishingly small nonzero imaginary part when converting to float. Nudging
# freq=0 to `_EPS_FREQ` for a model whose true DC value is already exactly
# real (as the three models above are) turns that exact zero imaginary part
# into a tiny-but-nonzero one and crashes that conversion ("can't convert
# complex to float") -- this was tried once and caused exactly that crash.
# So `_safe_freq` is used only inside `_burgers_modulus`, the one model
# that actually has a singularity at w=0, and even there the DC bin's
# result is forced back to exactly real afterward (see its comment).


@dataclass(frozen=True)
class ParamSpec:
    key: str
    short: str      # compact label, e.g. "E1" -- used in results/diagrams
    label: str      # full label, e.g. "E1 (Pa)" -- used in the input rows
    unit: str
    bayes_bounds: tuple[float, float]  # sensible default (min, max) for the Bayesian search
    fixed_default: float               # sensible default value for "Test parameters"


@dataclass(frozen=True)
class RheologyModel:
    key: str
    display_name: str
    short_formula: str  # e.g. "E1 ∥ (E2 — n2)" -- shown under the diagram
    params: tuple[ParamSpec, ...]
    complex_modulus: Callable[[np.ndarray, dict], np.ndarray]
    formula_html: str
    polymer_note: str

    def param_keys(self) -> tuple[str, ...]:
        return tuple(p.key for p in self.params)


def _safe_freq(freq: np.ndarray) -> np.ndarray:
    freq = np.asarray(freq, dtype=float)
    return np.where(freq == 0.0, _EPS_FREQ, freq)


def _kelvin_voigt_modulus(freq, params: dict) -> np.ndarray:
    e, eta = params["e"], params["eta"]
    w = 2 * np.pi * np.asarray(freq, dtype=float)
    return e + 1j * w * eta


def _sls_modulus(freq, params: dict) -> np.ndarray:
    """Standard Linear Solid (SLS), "3-parameter Kelvin" representation:
    the compliance of a spring E1 in SERIES with a Kelvin-Voigt unit
    (spring E2 in parallel with dashpot n2). `draw_sls` in
    ui/rheology_diagram.py draws this series-Kelvin topology to match
    what this formula computes."""
    e1, e2, n2 = params["e1"], params["e2"], params["n2"]
    w = 2 * np.pi * np.asarray(freq, dtype=float)
    j1 = (1.0 / e1) + (e2 / (e2 ** 2 + (w * n2) ** 2))
    j2 = (w * n2) / (e2 ** 2 + (w * n2) ** 2)
    return 1.0 / (j1 - 1j * j2)


def _generalized_maxwell_2_modulus(freq, params: dict) -> np.ndarray:
    e_inf, e1, n1, e2, n2 = params["e_inf"], params["e1"], params["n1"], params["e2"], params["n2"]
    w = 2 * np.pi * np.asarray(freq, dtype=float)
    tau1, tau2 = n1 / e1, n2 / e2
    return (
        e_inf
        + e1 * (1j * w * tau1) / (1 + 1j * w * tau1)
        + e2 * (1j * w * tau2) / (1 + 1j * w * tau2)
    )


def _burgers_modulus(freq, params: dict) -> np.ndarray:
    """This model's true DC (zero-frequency) modulus is exactly zero -- the
    bare Maxwell arm flows freely under any sustained stress -- which is a
    problem two levels deep for `core.visco_calibration._predict_pulse`:

    1. It would make that function's reference wave speed
       `c[0] = sqrt(E(0)/density)` zero or (evaluated at a tiny nudge away
       from 0, as `_EPS_FREQ` does for the singularity below) a vanishingly
       small *complex* number, and it divides by `c[0]` in its phase-
       correction formula for *every* frequency bin, not just this one --
       either blows that formula up or leaves a residual imaginary part
       that crashes its final `cmath.rect()` call (a python complex-to-
       float conversion fails on any nonzero imaginary part, however
       tiny -- this was tried and did crash).
    2. A true zero-frequency reference is not the physically relevant one
       for a millisecond-scale SHPB pulse anyway -- the material never
       actually holds a sustained (DC) load during the test.

    So the DC bin is evaluated at the pulse's own lowest nonzero frequency
    (`freq[1]`, the fundamental set by the windowed pulse's duration --
    already used elsewhere in `_predict_pulse` as its root-search seed)
    instead of at literal zero, keeping only that evaluation's real
    (storage) part as the reference stiffness and dropping its imaginary
    (loss) part -- the same simplification "a reference speed" always is,
    just made explicit here since this model has no better DC value to
    fall back on. Numerically this lands within a fraction of a percent of
    `|E(freq[1])|` (the convention `_predict_pulse` uses for every other
    bin), so it is a well-conditioned stand-in, not a degenerate one.
    """
    e1, n1, e2, n2 = params["e1"], params["n1"], params["e2"], params["n2"]
    freq = np.asarray(freq, dtype=float)

    def _modulus_at(f):
        w = 2 * np.pi * f
        j_maxwell = (1.0 / e1) + 1.0 / (1j * w * n1)
        j_kelvin_voigt = 1.0 / (e2 + 1j * w * n2)
        return 1.0 / (j_maxwell + j_kelvin_voigt)

    e_complex = _modulus_at(_safe_freq(freq))
    if freq.size > 1 and freq[0] == 0.0:
        e_complex[0] = _modulus_at(freq[1]).real
    return e_complex


MODEL_KELVIN_VOIGT = "kelvin_voigt"
MODEL_SLS = "standard_linear_solid"
MODEL_GENERALIZED_MAXWELL_2 = "generalized_maxwell_2"
MODEL_BURGERS = "burgers"

MODELS: dict[str, RheologyModel] = {
    MODEL_KELVIN_VOIGT: RheologyModel(
        key=MODEL_KELVIN_VOIGT,
        display_name="Kelvin–Voigt",
        short_formula="E ∥ η",
        params=(
            ParamSpec("e", "E", "E (Pa)", "Pa", (5e8, 5e9), 2e9),
            ParamSpec("eta", "η", "eta (Pa.s)", "Pa.s", (1e2, 1e5), 1e3),
        ),
        complex_modulus=_kelvin_voigt_modulus,
        formula_html=(
            "<p>A spring <b>E</b> in parallel with a dashpot <b>η</b> -- the simplest damped-"
            "elastic model.</p>"
            "<p>Complex modulus: E*(f) = E + i&middot;(2&pi;f)&middot;η</p>"
            "<p>Its damping grows without bound as frequency increases, and it has no separate "
            "instantaneous/relaxed modulus, so on its own it is a rough baseline rather than a good "
            "fit for a real polymer's relaxation spectrum.</p>"
        ),
        polymer_note="Simple baseline; rarely fits a real polymer's spectrum on its own.",
    ),
    MODEL_SLS: RheologyModel(
        key=MODEL_SLS,
        display_name="Standard Linear Solid (Zener)",
        short_formula="E1 — (E2 ∥ n2)",
        params=(
            ParamSpec("e1", "E1", "E1 (Pa)", "Pa", (5e8, 5e9), 2e9),
            ParamSpec("e2", "E2", "E2 (Pa)", "Pa", (5e8, 5e9), 2e9),
            ParamSpec("n2", "n2", "n2 (Pa.s)", "Pa.s", (1e2, 1e4), 1e3),
        ),
        complex_modulus=_sls_modulus,
        formula_html=(
            "<p>A spring <b>E1</b> in series with a Kelvin&ndash;Voigt element (a spring "
            "<b>E2</b> in parallel with a dashpot <b>n2</b>) -- the \"3-parameter Kelvin\" "
            "representation of the Standard Linear Solid.</p>"
            "<p>J(f) = 1/E1 + E2 / (E2&sup2; + (2&pi;f&middot;n2)&sup2;) "
            "&minus; i&middot;(2&pi;f&middot;n2) / (E2&sup2; + (2&pi;f&middot;n2)&sup2;)<br>"
            "E(f) = 1 / J(f)</p>"
            "<p>Single relaxation time -- this is the model originally shipped with this app, and "
            "the only one the Dispersion Corrections page's \"Viscoelastic bars\" option "
            "currently implements.</p>"
        ),
        polymer_note="Single relaxation time -- often too narrow for PMMA/PC/nylon's real spectrum.",
    ),
    MODEL_GENERALIZED_MAXWELL_2: RheologyModel(
        key=MODEL_GENERALIZED_MAXWELL_2,
        display_name="Generalized Maxwell (2 branches)",
        short_formula="E∞ ∥ (E1–n1) ∥ (E2–n2)",
        params=(
            ParamSpec("e_inf", "E∞", "E_inf (Pa)", "Pa", (5e8, 5e9), 1.5e9),
            ParamSpec("e1", "E1", "E1 (Pa)", "Pa", (5e8, 5e9), 1e9),
            ParamSpec("n1", "n1", "n1 (Pa.s)", "Pa.s", (1e1, 1e4), 1e2),
            ParamSpec("e2", "E2", "E2 (Pa)", "Pa", (5e8, 5e9), 1e9),
            ParamSpec("n2", "n2", "n2 (Pa.s)", "Pa.s", (1e3, 1e6), 1e4),
        ),
        complex_modulus=_generalized_maxwell_2_modulus,
        formula_html=(
            "<p>A long-term spring <b>E&infin;</b> in parallel with <i>two</i> Maxwell arms "
            "(<b>E1</b>&ndash;<b>n1</b> and <b>E2</b>&ndash;<b>n2</b>), one tuned to a faster "
            "relaxation and one to a slower one ("
            "τ<sub>k</sub> = n<sub>k</sub>/E<sub>k</sub>).</p>"
            "<p>E(f) = E&infin; + &sum;<sub>k=1,2</sub> E<sub>k</sub>&middot;(i&middot;2&pi;f&middot;"
            "τ<sub>k</sub>) / (1 + i&middot;2&pi;f&middot;τ<sub>k</sub>)</p>"
            "<p>Two relaxation times give a broader, asymmetric relaxation spectrum -- generally a "
            "closer match to real amorphous/semi-crystalline polymers than a single Debye term.</p>"
        ),
        polymer_note="Two relaxation times: a broader spectrum, closer to real PMMA/PC/nylon data.",
    ),
    MODEL_BURGERS: RheologyModel(
        key=MODEL_BURGERS,
        display_name="Burgers",
        short_formula="(E1–n1) — (E2 ∥ n2)",
        params=(
            ParamSpec("e1", "E1", "E1 (Pa)", "Pa", (5e8, 5e9), 2e9),
            ParamSpec("n1", "n1", "n1 (Pa.s)", "Pa.s", (1e4, 1e8), 1e6),
            ParamSpec("e2", "E2", "E2 (Pa)", "Pa", (5e8, 5e9), 1e9),
            ParamSpec("n2", "n2", "n2 (Pa.s)", "Pa.s", (1e1, 1e4), 1e2),
        ),
        complex_modulus=_burgers_modulus,
        formula_html=(
            "<p>A Maxwell element (spring <b>E1</b> in series with dashpot <b>n1</b>) in series "
            "with a Kelvin-Voigt element (spring <b>E2</b> in parallel with dashpot <b>n2</b>) -- "
            "a direct combination of the two simplest classical models.</p>"
            "<p>J(f) = 1/E1 + 1/(i&middot;2&pi;f&middot;n1) + 1/(E2 + i&middot;2&pi;f&middot;n2)<br>"
            "E(f) = 1 / J(f)</p>"
            "<p>Unlike the other three models here, its long-term modulus is genuinely zero (the "
            "bare Maxwell arm flows under sustained stress), which is what lets it represent the "
            "instantaneous elasticity + retarded (creep) elasticity + real viscous flow seen in "
            "thermoplastics such as nylon under sustained or repeated loading.</p>"
        ),
        polymer_note="Adds real viscous flow/creep -- suited to thermoplastics like nylon.",
    ),
}

MODEL_ORDER = (MODEL_SLS, MODEL_KELVIN_VOIGT, MODEL_GENERALIZED_MAXWELL_2, MODEL_BURGERS)


def get_model(key: str) -> RheologyModel:
    return MODELS[key]
