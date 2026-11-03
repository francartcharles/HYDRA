"""
Per-free-parameter prior specification for Bayesian Calibration, ported
from MAMBA's three mutually-exclusive per-parameter treatments
(def_model.py's `recup_all_para`/`cal_MCMC`, see that module's own
per-parameter prior table). A single equation can freely mix all three
kinds across its different free parameters.

This is a DELIBERATELY SEPARATE dataclass from `core/bayes_visco_
calibration.py`'s own `PriorSpec` (used by the app's other Bayesian
page, viscoelastic bar model calibration, ui/pages/page_visco_calib.py)
-- that one models a Gaussian-Process-surrogate optimizer's own bounded-
range/log-scale priors, a different concept with no hierarchical
treatment; forcing a shared class between the two would be a false
economy, not real reuse.

No PyMC/PyTensor objects live here -- this module stays import-light
(numpy/scipy/stdlib only) so it can be constructed and validated by the
page UI, and consumed by the pure-numpy forward-preview path, without
requiring pymc to be installed. `core/calib_pymc_model.py` is the only
place a `ParamPrior` gets turned into an actual `pm.Uniform`/`pm.Normal`/
`pm.ChiSquared` random variable.
"""
from __future__ import annotations

from dataclasses import dataclass

PRIOR_KINDS = ("fixed", "uniform", "normal_hier")

PRIOR_KIND_LABELS = {
    "fixed": "Fixed value",
    "uniform": "Uniform (shared across tests)",
    "normal_hier": "Normal (hierarchical, per-test)",
}


class PriorError(ValueError):
    """Raised when a ParamPrior's numeric fields are inconsistent for its
    own kind (e.g. a Uniform prior whose max <= min)."""


@dataclass
class ParamPrior:
    """One free parameter's prior treatment.

    - "fixed": a constant, not calibrated at all -- `fixed_value` is used
      directly wherever the parameter appears in the equation.
    - "uniform": `pm.Uniform(name, uniform_lo, uniform_hi)` -- a single
      value shared/pooled across every calibration test.
    - "normal_hier": MAMBA's hierarchical Normal-Inverse-ChiSquared
      hyperprior -- a genuine per-test random effect with partial
      pooling across tests, built (in core/calib_pymc_model.py) as:
          mu = pm.Normal(f"mu_{name}", hier_mu0, hier_s0)
          X  = pm.ChiSquared(f"X_{name}", hier_nu0)
          s  = sqrt(hier_s0**2 * hier_nu0 / X)
          <name> = pm.Normal(name, mu=mu, sigma=s, shape=n_tests)
      matching MAMBA's own `spara`/`mupara` construction (def_model.py,
      ~line 1306-1319) exactly. `hier_mu0`/`hier_s0`/`hier_nu0` are the
      hyperprior's own three numbers the user provides (MAMBA's
      `mu0,s0,S0,nu0` comma-separated entry, minus the redundant `S0`
      field -- MAMBA's own construction feeds the SAME `s0` into both the
      Normal hyperprior on `mu` and the ChiSquared-derived `s`, so a
      separate `S0` was never actually an independent input; kept as one
      `hier_s0` here rather than porting the redundancy forward).

    Every field is always present (not just the ones the current `kind`
    uses) so the UI can keep every group's widgets populated even while
    hidden -- only the fields matching `kind` are read by the model
    builder or the numeric preview path.
    """
    name: str
    kind: str = "uniform"
    fixed_value: float = 0.0
    uniform_lo: float = 0.0
    uniform_hi: float = 1.0
    hier_mu0: float = 0.0
    hier_s0: float = 1.0
    hier_nu0: float = 3.0

    def validate(self) -> None:
        """Raises PriorError if this prior's numeric fields don't make
        sense for its own kind. Does not check `name` (the caller/UI is
        responsible for name collisions/validity, since that's shared
        state across every parameter, not a single prior's own concern)."""
        if self.kind not in PRIOR_KINDS:
            raise PriorError(f"Unknown prior kind '{self.kind}' for parameter '{self.name}'.")
        if self.kind == "uniform":
            if not (self.uniform_hi > self.uniform_lo):
                raise PriorError(
                    f"Parameter '{self.name}': Uniform prior needs max > min "
                    f"(got min={self.uniform_lo}, max={self.uniform_hi})."
                )
        elif self.kind == "normal_hier":
            if not (self.hier_s0 > 0):
                raise PriorError(f"Parameter '{self.name}': hyperprior scale (s0) must be > 0.")
            if not (self.hier_nu0 > 0):
                raise PriorError(f"Parameter '{self.name}': hyperprior degrees of freedom (nu0) must be > 0.")

    def point_value(self) -> float:
        """A single representative numeric value for this parameter --
        used by the pure-numpy forward-preview path (before any MCMC run
        has happened) so the user can sanity-check the equation's shape.
        Fixed: its own value. Uniform: the interval's midpoint. Normal
        (hierarchical): the hyperprior's own mean (hier_mu0) -- not a
        per-test value, since no test-specific draw exists yet."""
        if self.kind == "fixed":
            return self.fixed_value
        if self.kind == "uniform":
            return (self.uniform_lo + self.uniform_hi) / 2.0
        if self.kind == "normal_hier":
            return self.hier_mu0
        raise PriorError(f"Unknown prior kind '{self.kind}' for parameter '{self.name}'.")


def validate_priors(priors: dict) -> None:
    """Validates a whole {name: ParamPrior} mapping -- every individual
    prior (`ParamPrior.validate`) plus the one cross-prior check that
    isn't a single prior's own concern: every key must match its own
    prior's `name` (a page-level bookkeeping invariant, not something a
    lone ParamPrior can check about itself)."""
    for key, prior in priors.items():
        if prior.name != key:
            raise PriorError(f"Prior dict key '{key}' does not match its ParamPrior.name '{prior.name}'.")
        prior.validate()
