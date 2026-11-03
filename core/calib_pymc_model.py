"""
PyMC 5 / PyTensor model builder and sampler for Bayesian Calibration,
reproducing MAMBA's per-parameter prior treatments and MvNormal-with-
measured-covariance likelihood (def_model.py's `cal_MCMC`), with the
bugs listed in the approved plan deliberately fixed rather than ported
(per-test covariance as the default instead of always the across-tests
mean; `target_accept` exposed instead of hardcoded to 0.99).

Every pymc/pytensor import is LAZY -- inside a function body, never at
module level -- so importing this module (and therefore importing
`ui.main_window`/anything that imports it) never requires pymc/pytensor
to be installed. `pymc_available()` is the single, cached way the rest
of the app (the page, core/calib_run.py) checks whether a Run is even
possible, so the UI can disable the Run button proactively with a clear
message instead of only failing after a click.

IMPORTANT -- sandbox limitation, documented here rather than silently
assumed: this development sandbox has no network access and does not
have pymc/pytensor installed, so `build_model`/`sample_model` below have
been written carefully against PyMC 5 / PyTensor's documented API but
COULD NOT be run against the real libraries during development. The
approved plan flags a required manual smoke test on the user's own
machine (a small synthetic 2-test, 1-parameter equation, checked by
hand) before this module is considered numerically validated, not just
code-complete -- see that plan's Phase 2 section. Every other module in
this feature (core/calib_equation.py, core/calib_priors.py, core/calib_
library_adapter.py) IS fully real-tested in this sandbox; this is the
one deliberate exception, isolated here specifically so that exception
doesn't spread.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.calib_equation import DOMAIN_EPS, KNOWN_FUNCS, ParsedEquation, evaluate
from core.calib_priors import ParamPrior


class PymcUnavailableError(RuntimeError):
    """Raised by build_model/sample_model (as a defensive backstop --
    core/calib_run.py is expected to check `pymc_available()` itself
    first and never call these at all otherwise) when pymc/pytensor
    cannot be imported. The message is written to surface directly in
    the page's status label, so it stays plain and actionable."""


_PYMC_AVAILABLE_CACHE: bool | None = None


def pymc_available() -> bool:
    """Cached probe: True once `import pymc` and `import pytensor` both
    succeed. Cached (module-level, process-lifetime) since the import
    attempt itself is not free and this is called from UI code that may
    re-check it often (e.g. every time the Bayesian Calibration page is
    shown, to keep the Run button's enabled state current)."""
    global _PYMC_AVAILABLE_CACHE
    if _PYMC_AVAILABLE_CACHE is None:
        try:
            import pymc  # noqa: F401
            import pytensor  # noqa: F401
            _PYMC_AVAILABLE_CACHE = True
        except ImportError:
            _PYMC_AVAILABLE_CACHE = False
    return _PYMC_AVAILABLE_CACHE


def _require_pymc():
    if not pymc_available():
        raise PymcUnavailableError(
            "PyMC / PyTensor are not installed in this environment. Bayesian Calibration's MCMC "
            "run needs both (`pip install pymc pytensor arviz`) -- see requirements.txt."
        )


def _build_pytensor_backend():
    """Built lazily (only once pytensor is confirmed importable) -- the
    PyTensor-tensor equivalent of core.calib_equation.NUMPY_BACKEND, same
    keys (`KNOWN_FUNCS`, plus the same internal `"__safe_pow__"` hook),
    same shape, so core.calib_equation.evaluate() can run the identical
    validated equation tree against either backend without knowing which
    one it's using. PyTensor mirrors numpy's own function names
    (`pt.arccos` not `pt.acos`, etc.) for every one of these -- see
    PyTensor's `tensor` module docs.

    Domain guards mirror NUMPY_BACKEND's exactly (same DOMAIN_EPS, same
    functions guarded, same reasoning -- see core.calib_equation's
    "Domain guards" docstring section): a test's real measured data
    (e.g. a strain-rate channel with 1-2 slightly-negative points from
    measurement noise) can otherwise turn `log`/`sqrt`/a non-integer
    power into NaN partway through PyMC's own model-building, which
    reliably surfaces as PyMC refusing to start sampling at all ("Initial
    evaluation of model at starting point failed!") rather than a plain
    Python exception -- guarding it here, identically to the numpy
    preview path, is what keeps the actual MCMC run itself from hitting
    this, not just this module's forward-preview twin."""
    import pytensor.tensor as pt

    def safe_sqrt(x):
        return pt.sqrt(pt.maximum(x, DOMAIN_EPS))

    def safe_log(x):
        return pt.log(pt.maximum(x, DOMAIN_EPS))

    def safe_log10(x):
        return pt.log10(pt.maximum(x, DOMAIN_EPS))

    def safe_acos(x):
        return pt.arccos(pt.clip(x, -1.0 + DOMAIN_EPS, 1.0 - DOMAIN_EPS))

    def safe_asin(x):
        return pt.arcsin(pt.clip(x, -1.0 + DOMAIN_EPS, 1.0 - DOMAIN_EPS))

    def safe_acosh(x):
        return pt.arccosh(pt.maximum(x, 1.0 + DOMAIN_EPS))

    def safe_atanh(x):
        return pt.arctanh(pt.clip(x, -1.0 + DOMAIN_EPS, 1.0 - DOMAIN_EPS))

    def safe_pow(base, exponent):
        return pt.maximum(base, DOMAIN_EPS) ** exponent

    return {
        "sqrt": safe_sqrt, "cos": pt.cos, "sin": pt.sin, "tan": pt.tan,
        "cosh": pt.cosh, "sinh": pt.sinh, "tanh": pt.tanh,
        "acos": safe_acos, "asin": safe_asin, "atan": pt.arctan,
        "erf": pt.erf, "exp": pt.exp, "log": safe_log, "log10": safe_log10,
        "asinh": pt.arcsinh, "acosh": safe_acosh, "atanh": safe_atanh,
        "__safe_pow__": safe_pow,
    }


@dataclass
class BuiltModel:
    """`build_model`'s return value: the pymc Model itself, plus enough
    bookkeeping for core/calib_run.py and the UI to make sense of its
    random variables afterward without re-deriving this from `priors`."""
    model: object                      # pm.Model
    pooled_param_names: tuple          # free-parameter names sampled as a single pm.Uniform (kind="uniform")
    hier_param_names: tuple            # free-parameter names sampled hierarchically (kind="normal_hier")
    fixed_param_values: dict           # {name: float} for kind="fixed" params (not sampled at all)
    n_tests: int


def build_model(parsed: ParsedEquation, priors: dict, tests: list, covariances: list):
    """Builds (but does not sample) the pymc.Model for a calibration run.

    `parsed`: from core.calib_equation.parse_equation(equation_str).
    `priors`: {free_param_name: ParamPrior} -- core.calib_priors.validate_priors(priors)
      is expected to have already been called by the caller (core.calib_run.run_calibration).
    `tests`: list[core.calib_library_adapter.CalibTestDataset] -- CALIBRATION tests only
      (a "Validation only" row contributes no likelihood term and must already be
      filtered out by the caller before this is called).
    `covariances`: one covariance matrix (np.ndarray) per entry in `tests`, same order --
      from core.calib_library_adapter.per_test_or_shared_covariance(tests, mode).

    Reproduces MAMBA's exact per-parameter treatment (see core/calib_priors.py's own
    docstring for the hierarchical hyperprior formula) and its MvNormal-with-measured-
    covariance likelihood, built test-by-test via the SAME core.calib_equation.evaluate()
    call the numpy forward-preview path uses -- just against `_build_pytensor_backend()`
    and PyMC random variables in the namespace instead of plain arrays."""
    _require_pymc()
    import pymc as pm
    import pytensor.tensor as pt

    if len(tests) != len(covariances):
        raise ValueError(f"Got {len(tests)} tests but {len(covariances)} covariance matrices -- must match 1:1.")
    if not tests:
        raise ValueError("Need at least one (non-validation) calibration test to build a model.")

    n_tests = len(tests)
    backend = _build_pytensor_backend()

    pooled_names, hier_names, fixed_values = [], [], {}
    for name, prior in priors.items():
        if prior.kind == "fixed":
            fixed_values[name] = float(prior.fixed_value)
        elif prior.kind == "uniform":
            pooled_names.append(name)
        elif prior.kind == "normal_hier":
            hier_names.append(name)
        else:
            raise ValueError(f"Unknown prior kind '{prior.kind}' for parameter '{name}'.")
    pooled_names, hier_names = tuple(sorted(pooled_names)), tuple(sorted(hier_names))

    with pm.Model() as model:
        # -- Uniform (pooled/shared) parameters: one scalar RV, reused
        # identically for every test's own equation evaluation below.
        pooled_rvs = {}
        for name in pooled_names:
            prior = priors[name]
            pooled_rvs[name] = pm.Uniform(name, lower=prior.uniform_lo, upper=prior.uniform_hi)

        # -- Normal/hierarchical parameters: Normal-Inverse-ChiSquared
        # hyperprior -> shape=(n_tests,) per-test random effect, matching
        # core/calib_priors.py's ParamPrior docstring / MAMBA's own
        # spara/mupara construction exactly.
        hier_rvs = {}
        for name in hier_names:
            prior = priors[name]
            mu = pm.Normal(f"mu_{name}", mu=prior.hier_mu0, sigma=prior.hier_s0)
            chi = pm.ChiSquared(f"X_{name}", nu=prior.hier_nu0)
            sigma = pt.sqrt((prior.hier_s0 ** 2) * prior.hier_nu0 / chi)
            hier_rvs[name] = pm.Normal(name, mu=mu, sigma=sigma, shape=n_tests)

        for j, (test, cov) in enumerate(zip(tests, covariances)):
            namespace = dict(test.variables)  # reserved vars (e, v, T, a, b, d), already on this test's own grid
            for name in fixed_values:
                namespace[name] = fixed_values[name]
            for name in pooled_names:
                namespace[name] = pooled_rvs[name]
            for name in hier_names:
                namespace[name] = hier_rvs[name][j]

            phi_j = evaluate(parsed, namespace, backend)

            # The covariance matrix is fixed (measured) data, not a random
            # variable -- its Cholesky factor is plain numpy, computed once
            # per test, not part of the pytensor graph.
            chol_j = np.linalg.cholesky(np.asarray(cov, dtype=float))
            pm.MvNormal(f"obs_{j}_{test.label}", mu=phi_j, chol=chol_j, observed=test.observed)

    return BuiltModel(
        model=model, pooled_param_names=pooled_names, hier_param_names=hier_names,
        fixed_param_values=fixed_values, n_tests=n_tests,
    )


DEFAULT_TARGET_ACCEPT = 0.9  # PyMC's own default -- MAMBA hardcoded 0.99 and never exposed it (fixed here)
DEFAULT_DRAWS = 1000
DEFAULT_TUNE = 1000
DEFAULT_CHAINS = 4
DEFAULT_INIT = "auto"

# Which of pm.sample's own `nuts_sampler` backends to use -- "nutpie" (a
# Rust-based NUTS implementation, https://github.com/pymc-devs/nutpie)
# instead of PyMC's own built-in "pymc" sampler. WHY: the "pymc" sampler
# compiles each model's logp/gradient through PyTensor's C backend, which
# needs a working C/C++ compiler (g++ on Windows, via mingw-w64) present
# on whichever machine actually runs it -- fine in this dev sandbox/on a
# dev machine with a compiler installed, but a real problem for a
# PyInstaller-frozen .exe handed to an end user who has no compiler and
# no dev environment at all. "nutpie" instead compiles through PyTensor's
# Numba backend, and Numba's own dependency (llvmlite) statically links
# LLVM into its wheel -- no system compiler needed on the end user's
# machine at all, only the (larger, but self-contained) numba/nutpie
# packages need to be bundled into the frozen app. This is also PyMC's
# own current recommended default (`pip install "pymc[nutpie]"`; `pm.
# sample` already prefers nutpie automatically when it's importable) --
# so forcing it explicitly here is "adopt PyMC's own recommended modern
# default," not a workaround bolted on just for packaging.
# See requirements.txt (numba/nutpie added alongside pymc/pytensor/arviz)
# and BUILD_WINDOWS.md for the full PyInstaller story this exists for.
DEFAULT_NUTS_SAMPLER = "nutpie"


def sample_model(
    built: BuiltModel,
    draws: int = DEFAULT_DRAWS,
    tune: int = DEFAULT_TUNE,
    chains: int = DEFAULT_CHAINS,
    cores: int = 1,
    target_accept: float = DEFAULT_TARGET_ACCEPT,
    init: str = DEFAULT_INIT,
    nuts_sampler: str = DEFAULT_NUTS_SAMPLER,
    progress_cb=None,
    cancel_check=None,
):
    """Runs NUTS sampling on an already-built model (`build_model`'s
    return value), returning the resulting `arviz.InferenceData`.

    `nuts_sampler`: forwarded to `pm.sample`'s own argument of the same
    name -- "nutpie" (default, see DEFAULT_NUTS_SAMPLER's own docstring
    for why: no C/C++ compiler needed at runtime, unlike PyMC's built-in
    "pymc" sampler), "pymc" (the classic C-backend sampler -- fine on a
    dev machine with a working compiler, but will fail on a frozen .exe
    with none), "numpyro" or "blackjax" (PyMC's other alternatives, not
    otherwise supported/tested by this app). Passing a falsy value
    (`None`/`""`) omits the argument entirely, falling back to whatever
    `pm.sample` itself would default to -- an explicit escape hatch for a
    machine where nutpie genuinely isn't installed, rather than this
    function silently forcing an unavailable sampler.

    NOTE on cancellation: PyMC has no supported mid-sampling cancellation
    hook -- `cancel_check` is accepted (for interface symmetry with every
    other calibration function this app's shared core/qt_worker.py
    drives) but is only checked BEFORE sampling starts, not during it.
    core/calib_run.py's docstring and the Run card's Stop-button tooltip
    must both say this plainly rather than overpromise a responsive stop.
    `progress_cb`, similarly, can only be called once (0, 1) before/after
    -- pm.sample's own text progress bar is disabled (`progressbar=False`)
    since it would otherwise print to stdout uncontrolled by this app."""
    _require_pymc()
    import pymc as pm

    if cancel_check is not None and cancel_check():
        from core.parallel import OperationCancelled
        raise OperationCancelled()

    if progress_cb is not None:
        progress_cb(0, 1)

    with built.model:
        sample_kwargs = dict(
            draws=draws, tune=tune, chains=chains, cores=cores,
            target_accept=target_accept, init=init,
            progressbar=False, return_inferencedata=True,
        )
        if nuts_sampler:  # see nuts_sampler's own docstring above for why/when this is omitted
            sample_kwargs["nuts_sampler"] = nuts_sampler
        idata = pm.sample(**sample_kwargs)

    if progress_cb is not None:
        progress_cb(1, 1)

    return idata
