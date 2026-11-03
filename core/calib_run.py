"""
Top-level orchestration for one Bayesian Calibration run: equation ->
per-test datasets -> covariance mode -> PyMC model -> NUTS sampling ->
a packaged `CalibrationResult`. This is the single function `ui/pages/
page_bayes_calib.py`'s Run button calls (via core/qt_worker.py, off the
GUI thread) and the single function `test_calib_worker_wiring.py`
monkeypatches to test that page's Qt plumbing without needing real PyMC.

Deliberately thin: every real piece of logic lives in the module it
belongs to (core/calib_equation.py, core/calib_priors.py, core/calib_
library_adapter.py, core/calib_pymc_model.py) -- this module's only job
is calling them in the right order with the right error at the right
point, matching the approved plan's own orchestration list.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.calib_equation import evaluate_with_domain_report, parse_equation
from core.calib_library_adapter import (
    DEFAULT_COVARIANCE_JITTER, build_test_dataset, covariance_condition_number,
    per_test_or_shared_covariance,
)
from core.calib_priors import validate_priors
from core.calib_pymc_model import (
    DEFAULT_CHAINS, DEFAULT_DRAWS, DEFAULT_INIT, DEFAULT_NUTS_SAMPLER, DEFAULT_TARGET_ACCEPT, DEFAULT_TUNE,
    PymcUnavailableError, build_model, pymc_available, sample_model,
)
from core.calib_diagnostics import convergence_summary
from core.parallel import OperationCancelled


class CalibRunError(ValueError):
    """Raised for anything wrong with a calibration run's own setup
    (empty equation, no calibration tests assigned, ...) -- distinct from
    EquationError/PriorError/LibraryAdapterError, which are all caught
    and re-raised as this so the page's single status-label error path
    doesn't need to know about every module's own exception type."""


@dataclass
class CalibrationResult:
    """Packaged result of one successful `run_calibration` call. `idata`
    is kept for Phase 3's diagnostic plots (posterior density, forest,
    energy, per-test fit, validation) -- everything else here is already
    enough for the Result Summary card (Phase 2, UI card 6)."""
    equation_source: str
    used_vars: tuple
    free_params: tuple
    priors: dict
    covariance_mode: str
    pooled_param_names: tuple
    hier_param_names: tuple
    fixed_param_values: dict
    posterior_summary: dict          # {variable name (as arviz names it): {"mean", "sd", "r_hat", "ess_bulk", ...}}
    idata: object                    # arviz.InferenceData
    calibration_tests: list          # list[CalibTestDataset] actually used in the likelihood
    validation_tests: list           # list[CalibTestDataset] excluded from the likelihood (Phase 3's Validation card)
    covariance_condition_numbers: dict = field(default_factory=dict)  # {test label: RAW (pre-regularization) condition number}
    domain_guard_warnings: dict = field(default_factory=dict)  # {test label: {function name (or "pow"): count of real data points clamped}}
    sampler_settings: dict = field(default_factory=dict)  # echo of draws/tune/chains/cores/target_accept/init/nuts_sampler actually used
    zero_start_vars: tuple = ()      # echo of which roles had their curves translated to start at 0 before fitting
    zero_start_output: bool = False  # echo of whether the output's own curve was translated to start at 0 before fitting


def _check_cancelled(cancel_check):
    if cancel_check is not None and cancel_check():
        raise OperationCancelled()


def _build_datasets(test_assignments: list, zero_start_vars=None, zero_start_output: bool = False) -> tuple:
    """Turns the UI's raw "Calibration tests" table rows into
    `CalibTestDataset`s, splitting calibration vs. "Validation only"
    rows. Wraps `LibraryAdapterError` as `CalibRunError` so the page's
    single error-message path stays exception-type-agnostic."""
    from core.calib_library_adapter import LibraryAdapterError

    if not test_assignments:
        raise CalibRunError("Add at least one calibration test (Calibration tests card) before running.")

    calibration_tests, validation_tests = [], []
    try:
        for row in test_assignments:
            ds = build_test_dataset(
                label=row["label"],
                variable_entry_map=row.get("variable_entries", {}),
                output_entry=row["output_entry"],
                is_validation=bool(row.get("is_validation", False)),
                zero_start_vars=zero_start_vars,
                zero_start_output=zero_start_output,
            )
            (validation_tests if ds.is_validation else calibration_tests).append(ds)
    except LibraryAdapterError as exc:
        raise CalibRunError(str(exc)) from exc
    except KeyError as exc:
        raise CalibRunError(f"Test row is missing its output-variable assignment ({exc}).") from exc

    if not calibration_tests:
        raise CalibRunError(
            "Every test row is marked \"Validation only\" -- at least one must be a real "
            "calibration test (contributing to the likelihood) for the model to fit anything."
        )
    return calibration_tests, validation_tests


def _representative_param_value(prior) -> float:
    """A single, deterministic 'plausible' value for one free parameter,
    used ONLY by `_domain_guard_report_for_tests` below (a pre-sampling
    diagnostic) -- never anything that affects the actual fit itself:
    a Fixed prior's own fixed value, a Uniform prior's own midpoint, or
    a Normal-hierarchical prior's own hyper-mean (`hier_mu0`) -- in every
    case a value the user themselves is already asserting is at least
    plausible for this parameter, just by having set that prior."""
    if prior.kind == "fixed":
        return float(prior.fixed_value)
    if prior.kind == "uniform":
        return float(prior.uniform_lo + prior.uniform_hi) / 2.0
    if prior.kind == "normal_hier":
        return float(prior.hier_mu0)
    raise ValueError(f"Unknown prior kind '{prior.kind}'.")


def _domain_guard_report_for_tests(parsed, used_priors: dict, tests: list) -> dict:
    """Best-effort, PyMC-free domain-guard diagnostic (see core.calib_
    equation's "Domain guards" docstring section): evaluates `parsed`
    once per test, against that test's own REAL measured variable data
    plus one representative value per free parameter (see
    `_representative_param_value` above), through core.calib_equation.
    evaluate_with_domain_report -- and returns `{test label: {function
    name: count}}` for every test where 1+ real data point needed
    clamping into a function's valid domain.

    Deliberately a diagnostic, never a gate: any failure evaluating a
    given test (e.g. that test is missing a variable the equation needs
    -- a separate, pre-existing condition this function is not
    responsible for catching) is silently skipped for that test rather
    than raised, so this can never change run_calibration's own error
    behavior -- it only ever adds information, never blocks a run that
    would otherwise have proceeded."""
    reports: dict = {}
    representative = {name: _representative_param_value(used_priors[name]) for name in used_priors}
    for test in tests:
        namespace = dict(test.variables)
        namespace.update(representative)
        try:
            _, report = evaluate_with_domain_report(parsed, namespace)
        except Exception:  # noqa: BLE001 -- diagnostic only, never blocks the run
            continue
        if report:
            reports[test.label] = report
    return reports


def run_calibration(
    equation_str: str,
    priors: dict,
    test_assignments: list,
    covariance_mode: str = "per_test",
    sampler_settings: dict | None = None,
    zero_start_vars=None,
    zero_start_output: bool = False,
    progress_cb=None,
    cancel_check=None,
) -> CalibrationResult:
    """Runs one full Bayesian Calibration: parse -> build per-test
    datasets -> resolve the covariance mode -> confirm PyMC is available
    -> build the model -> sample -> summarize.

    `test_assignments`: list of {"label": str, "variable_entries":
    {role: CovarianceLibraryEntry}, "output_entry": CovarianceLibraryEntry,
    "is_validation": bool} -- one dict per "Calibration tests" table row
    (ui/pages/page_bayes_calib.py, UI card 3).

    `sampler_settings`: any of {"draws", "tune", "chains", "cores",
    "target_accept", "init", "nuts_sampler", "covariance_jitter"} --
    missing keys fall back to core.calib_pymc_model's own defaults (or,
    for "covariance_jitter", core.calib_library_adapter.DEFAULT_
    COVARIANCE_JITTER). "covariance_jitter" is forwarded to `per_test_or_
    shared_covariance` -- see `core.calib_library_adapter.regularize_
    covariance`'s docstring for why a real user's own measured
    covariance matrix can need this (condition numbers in the tens of
    millions have been observed) to keep the likelihood numerically
    sane. "nuts_sampler" defaults to core.calib_pymc_model.DEFAULT_NUTS_
    SAMPLER ("nutpie") -- see that constant's own docstring: it avoids
    PyTensor's C backend entirely (compiles via PyTensor's Numba backend
    instead), so no system C/C++ compiler is required at runtime, which
    matters for a PyInstaller-frozen build on a machine with no build
    tools installed.

    Separately (and unconditionally -- there is no equivalent "off"
    setting, since the alternative is a crash or a silent NaN, not a
    modeling choice): every domain-restricted function/power in the
    equation is evaluated through a domain guard (core.calib_equation's
    "Domain guards" docstring section) so a handful of out-of-domain
    real data points (e.g. 1-2 slightly-negative strain-rate points from
    measurement noise, feeding `log(v)` or `v**m`) clamp instead of
    crashing the run. `CalibrationResult.domain_guard_warnings` reports,
    per test, exactly how many real points needed this.

    `zero_start_vars`/`zero_start_output`: forwarded to `core.calib_
    library_adapter.build_test_dataset` for every test -- independently
    translate an assigned variable's own curve (only the roles named in
    `zero_start_vars`) and/or the output's own curve (`zero_start_output`)
    to start at 0, per test, before anything else happens (interpolation,
    the likelihood, posterior-predictive checks all then see the
    already-translated curves).

    Raises CalibRunError for setup problems (bad equation, bad priors, no
    tests, bad library-entry assignment) and PymcUnavailableError if
    pymc/pytensor aren't installed -- both are plain, single-line
    exceptions the calling page's status label can show directly (via
    core/qt_worker.py's `failed` signal, which already stringifies
    whatever this raises)."""
    if not equation_str or not equation_str.strip():
        raise CalibRunError("Enter an equation first (Equation card).")
    try:
        parsed = parse_equation(equation_str)
    except Exception as exc:  # noqa: BLE001 -- EquationError, re-raised uniformly
        raise CalibRunError(str(exc)) from exc

    missing_priors = [name for name in parsed.free_params if name not in priors]
    if missing_priors:
        raise CalibRunError(f"Missing a prior for: {', '.join(missing_priors)}.")
    used_priors = {name: priors[name] for name in parsed.free_params}
    try:
        validate_priors(used_priors)
    except Exception as exc:  # noqa: BLE001 -- PriorError, re-raised uniformly
        raise CalibRunError(str(exc)) from exc

    _check_cancelled(cancel_check)

    calibration_tests, validation_tests = _build_datasets(
        test_assignments, zero_start_vars=zero_start_vars, zero_start_output=zero_start_output
    )

    # RAW (pre-regularization) condition number for each test's own
    # covariance -- computed here, before any per_test/shared_mean
    # pooling or jitter regularization, so it always reflects exactly
    # what was actually measured (Covariance Analysis page) for that
    # test, regardless of covariance_mode or covariance_jitter. See
    # core.calib_library_adapter.covariance_condition_number's docstring.
    covariance_condition_numbers = {
        t.label: covariance_condition_number(t.covariance) for t in calibration_tests
    }

    # Pre-sampling domain-guard diagnostic (see core.calib_equation's
    # "Domain guards" docstring section and _domain_guard_report_for_
    # tests's own docstring above): checked against every test that will
    # actually be evaluated (calibration AND validation), since both go
    # through the same equation.
    domain_guard_warnings = _domain_guard_report_for_tests(
        parsed, used_priors, calibration_tests + validation_tests
    )

    settings = dict(
        draws=DEFAULT_DRAWS, tune=DEFAULT_TUNE, chains=DEFAULT_CHAINS, cores=1,
        target_accept=DEFAULT_TARGET_ACCEPT, init=DEFAULT_INIT,
        covariance_jitter=DEFAULT_COVARIANCE_JITTER,
        nuts_sampler=DEFAULT_NUTS_SAMPLER,
    )
    settings.update(sampler_settings or {})

    try:
        covariances = per_test_or_shared_covariance(
            calibration_tests, mode=covariance_mode, jitter_frac=settings["covariance_jitter"]
        )
    except Exception as exc:  # noqa: BLE001 -- LibraryAdapterError, re-raised uniformly
        raise CalibRunError(str(exc)) from exc

    if not pymc_available():
        raise PymcUnavailableError(
            "PyMC / PyTensor are not installed in this environment. Bayesian Calibration's MCMC "
            "run needs both (`pip install pymc pytensor arviz`) -- see requirements.txt."
        )

    _check_cancelled(cancel_check)

    built = build_model(parsed, used_priors, calibration_tests, covariances)

    idata = sample_model(
        built,
        draws=settings["draws"], tune=settings["tune"], chains=settings["chains"], cores=settings["cores"],
        target_accept=settings["target_accept"], init=settings["init"],
        nuts_sampler=settings["nuts_sampler"],
        progress_cb=progress_cb, cancel_check=cancel_check,
    )

    return CalibrationResult(
        equation_source=parsed.source,
        used_vars=parsed.used_vars,
        free_params=parsed.free_params,
        priors=used_priors,
        covariance_mode=covariance_mode,
        pooled_param_names=built.pooled_param_names,
        hier_param_names=built.hier_param_names,
        fixed_param_values=built.fixed_param_values,
        posterior_summary=convergence_summary(idata),
        idata=idata,
        calibration_tests=calibration_tests,
        validation_tests=validation_tests,
        covariance_condition_numbers=covariance_condition_numbers,
        domain_guard_warnings=domain_guard_warnings,
        sampler_settings=settings,
        zero_start_vars=tuple(zero_start_vars or ()),
        zero_start_output=zero_start_output,
    )
