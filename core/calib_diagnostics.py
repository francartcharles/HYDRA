"""
Posterior diagnostics and posterior-predictive checks for Bayesian
Calibration, Phase 3.

Two different levels of "real-testable in this sandbox" on purpose, per
the approved plan:

- `posterior_predictive_at`, `posterior_correlation_matrix`,
  `forest_plot_rows` and `energy_diagnostic_arrays` all take PLAIN numpy
  arrays/dicts (never call into arviz's plotting API), so they are fully
  real-tested here with fake posterior-sample dicts/idata doubles --
  exactly like core/calib_equation.py's evaluator, which is the one
  piece of machinery `posterior_predictive_at` is actually built on
  (called once per posterior draw). `forest_plot_rows`/`energy_
  diagnostic_arrays` used to be thin `az.plot_forest`/`az.plot_energy`
  wrappers instead -- see their own docstrings for why that changed.
- `convergence_summary` is a thin wrapper around `arviz.summary`
  (lazy-imported, same pattern as core/calib_pymc_model.py's lazy pymc/
  pytensor imports) -- genuinely untestable without arviz installed, so
  it gets the same two-tier treatment as Phase 2's PyMC code: confirmed-
  real "arviz not installed" behavior here, wiring-only (monkeypatched)
  coverage for the UI card that calls it.

MAMBA never computed real posterior-predictive validation (it only ever
overlaid the calibration fit onto a "validation" plot, never actually
re-evaluating the model at the validation test's own conditions) and
never computed R-hat/ESS at all -- both fixed here, per the approved
plan's fidelity-decisions section.
"""
from __future__ import annotations

import numpy as np

from core.calib_equation import NUMPY_BACKEND, ParsedEquation, evaluate


class CalibDiagnosticsError(ValueError):
    """Raised for anything wrong with a diagnostics call's own inputs
    (no posterior draws given, fewer than 2 parameters for a correlation
    matrix, ...) -- distinct from ArviZUnavailableError, which is about
    the environment, not the call's arguments."""


class ArviZUnavailableError(RuntimeError):
    """Raised by convergence_summary (as a defensive backstop -- callers
    are expected to check `arviz_available()` themselves first) when
    arviz cannot be imported."""


_ARVIZ_AVAILABLE_CACHE: bool | None = None


def arviz_available() -> bool:
    """Cached probe, same pattern as core.calib_pymc_model.pymc_available()."""
    global _ARVIZ_AVAILABLE_CACHE
    if _ARVIZ_AVAILABLE_CACHE is None:
        try:
            import arviz  # noqa: F401
            _ARVIZ_AVAILABLE_CACHE = True
        except ImportError:
            _ARVIZ_AVAILABLE_CACHE = False
    return _ARVIZ_AVAILABLE_CACHE


def _require_arviz():
    if not arviz_available():
        raise ArviZUnavailableError(
            "arviz is not installed in this environment. Posterior diagnostics need it -- "
            "`pip install arviz` -- see requirements.txt."
        )


# ---------------------------------------------------------------------------
# Real, arviz-free: posterior-predictive check + parameter correlation
# ---------------------------------------------------------------------------

def resolve_predictive_param_draws(free_params: tuple, priors: dict, posterior_flat: dict, test_index=None) -> dict:
    """Resolves every free parameter to a flat {name: (n_draws,) array}
    of posterior draws suitable for `posterior_predictive_at`, given
    `posterior_flat` -- {arviz variable name: (n_draws,) or
    (n_draws, n_tests) array}, already flattened across chains (e.g.
    `idata.posterior[name].values.reshape(-1, *shape[2:])`) -- and each
    parameter's own `ParamPrior` (core/calib_priors.py), whose `kind`
    decides how to resolve it:

    - "fixed": not sampled at all -- not in `posterior_flat`; every draw
      uses the same constant (`prior.fixed_value`), broadcast to match
      the other parameters' own draw count.
    - "uniform": `posterior_flat[name]` directly, already (n_draws,).
    - "normal_hier": `posterior_flat[name][:, test_index]` when
      `test_index` is given (this dataset WAS one of the tests the model
      was actually fit on, e.g. checking the fit at one of its own
      calibration tests) -- otherwise (a genuinely held-out validation
      test, never part of the fit) falls back to
      `posterior_flat[f"mu_{name}"]`, the population hyper-mean. This is
      a documented simplification: it predicts using the population's
      central tendency and ignores the between-test scatter a brand-new
      test would actually also be subject to, so a held-out validation
      band from this function is narrower than the true predictive
      uncertainty for hierarchical parameters -- stated plainly in the
      Validation card's own help text, not silently assumed away.
    """
    n_draws = None
    for arr in posterior_flat.values():
        n_draws = len(arr)
        break
    if n_draws is None:
        raise CalibDiagnosticsError("posterior_flat is empty -- no posterior draws to resolve.")

    resolved = {}
    for name in free_params:
        prior = priors[name]
        if prior.kind == "fixed":
            resolved[name] = np.full(n_draws, float(prior.fixed_value))
        elif prior.kind == "uniform":
            if name not in posterior_flat:
                raise CalibDiagnosticsError(f"No posterior draws found for pooled parameter '{name}'.")
            resolved[name] = np.asarray(posterior_flat[name], dtype=float)
        elif prior.kind == "normal_hier":
            if test_index is not None and name in posterior_flat:
                resolved[name] = np.asarray(posterior_flat[name], dtype=float)[:, test_index]
            elif f"mu_{name}" in posterior_flat:
                resolved[name] = np.asarray(posterior_flat[f"mu_{name}"], dtype=float)
            else:
                raise CalibDiagnosticsError(
                    f"No posterior draws found for hierarchical parameter '{name}' "
                    f"(looked for '{name}' with a test_index, and 'mu_{name}')."
                )
        else:
            raise CalibDiagnosticsError(f"Unknown prior kind '{prior.kind}' for parameter '{name}'.")
    return resolved


def posterior_predictive_at(parsed: ParsedEquation, param_draws: dict, dataset) -> dict:
    """Genuine posterior-predictive evaluation of `parsed` at `dataset`'s
    own grid/variables (a core.calib_library_adapter.CalibTestDataset),
    once per posterior draw in `param_draws` (see
    `resolve_predictive_param_draws` for the usual way to build this) --
    fixes MAMBA's own "validation" view, which only ever overlaid the
    calibration fit rather than actually re-evaluating the model at the
    validation test's own conditions.

    Returns {"grid", "observed", "mean", "lower", "upper", "predictions"}
    -- `lower`/`upper` are the 2.5th/97.5th percentile curves across
    draws (a 95% posterior-predictive band), `predictions` the full
    (n_draws, n_grid_points) ensemble for anything else that wants it
    (e.g. a custom percentile)."""
    if not param_draws:
        raise CalibDiagnosticsError("No parameter draws given -- nothing to evaluate.")
    n_draws = len(next(iter(param_draws.values())))
    if n_draws == 0:
        raise CalibDiagnosticsError("param_draws contains zero-length arrays -- nothing to evaluate.")

    grid = np.asarray(dataset.grid, dtype=float)
    predictions = np.empty((n_draws, grid.size), dtype=float)
    for i in range(n_draws):
        namespace = dict(dataset.variables)
        for name, draws in param_draws.items():
            namespace[name] = draws[i]
        predictions[i] = np.broadcast_to(evaluate(parsed, namespace, NUMPY_BACKEND), grid.shape)

    return {
        "grid": grid,
        "observed": np.asarray(dataset.observed, dtype=float),
        "mean": predictions.mean(axis=0),
        "lower": np.percentile(predictions, 2.5, axis=0),
        "upper": np.percentile(predictions, 97.5, axis=0),
        "predictions": predictions,
    }


def posterior_correlation_matrix(posterior_flat: dict) -> tuple:
    """Plain Pearson correlation matrix (`np.corrcoef`, no arviz needed)
    across whichever named posterior draws are given -- e.g. every
    pooled parameter's own draws plus every hierarchical parameter's
    `mu_<name>` (hyper-mean) draws, matching how ui/pages/page_visco_
    calib.py's own `_plot_interdependence` uses a direct-parameter
    correlation matrix (not including internal-only variables like a
    hierarchical parameter's per-test values or its ChiSquared draws).

    Returns (names, matrix) with `names` sorted alphabetically (a stable,
    reproducible column order) and `matrix` of shape (len(names),
    len(names))."""
    names = sorted(posterior_flat.keys())
    if len(names) < 2:
        raise CalibDiagnosticsError(
            f"Need at least 2 parameters to compute a correlation matrix (got {len(names)})."
        )
    stacked = np.vstack([np.asarray(posterior_flat[n], dtype=float) for n in names])
    matrix = np.corrcoef(stacked)
    return names, matrix


# ---------------------------------------------------------------------------
# arviz-dependent (lazy import): convergence summary only
# ---------------------------------------------------------------------------

def convergence_summary(idata) -> dict:
    """{arviz row name: {stat column: value}} from `arviz.summary` --
    R-hat/ESS included, always computed (MAMBA never computed these at
    all -- `forestplot(..., rhat=False)` -- fixed here). Tolerant of
    exact-column-name differences across arviz versions (the columns
    read here -- mean/sd/r_hat/ess_bulk/ess_tail -- have been stable for
    a long time, but this doesn't hardcode assuming so)."""
    _require_arviz()
    import arviz as az

    df = az.summary(idata)
    return {row_name: {col: row[col] for col in df.columns} for row_name, row in df.iterrows()}


# ---------------------------------------------------------------------------
# Real, arviz-plotting-free: forest-plot rows + energy-diagnostic arrays
#
# An earlier version of this module drew the Forest/Energy diagnostic
# views via `az.plot_forest`/`az.plot_energy` directly. A real bug report
# from a user's own machine showed those calls failing with an unrelated-
# looking error ("Keyword arguments ['r_hat', 'ax'] ... have no active
# aesthetic mapped to them") -- the installed arviz version's plotting
# API no longer accepted the same `ax=`/`r_hat=` keyword shape this
# module was written against. arviz's PLOTTING functions have been
# through more than one API redesign; its underlying InferenceData DATA
# (an xarray-based structure -- `idata.sample_stats["energy"]`, and
# az.summary()'s own returned columns, both used below) has stayed far
# more stable across versions. So rather than chase a moving plotting
# API this module cannot test against without the user's own exact
# arviz version installed, both diagnostics are now built by hand from
# that stable data with plain numpy + matplotlib (drawn by the caller,
# ui/pages/page_bayes_calib.py, which already owns every other plot on
# this page the same way). Bonus: both are now REAL-testable here with
# fake data, no monkeypatching needed at all -- an actual improvement in
# coverage, not just a workaround.
# ---------------------------------------------------------------------------

def forest_plot_rows(posterior_summary: dict, var_names) -> list:
    """Extracts exactly what a forest plot needs -- point estimate, 94%
    HDI interval, R-hat -- out of `posterior_summary` (core.calib_run.
    CalibrationResult's own field; itself `convergence_summary(idata)`'s
    return value, i.e. az.summary()'s default columns: mean, sd,
    hdi_3%/hdi_97%, r_hat, ...).

    Returns [{"name", "mean", "lo", "hi", "r_hat"}, ...], one dict per
    name in `var_names` that is actually present in `posterior_summary`
    (a name not found -- e.g. a fixed parameter, never sampled at all --
    is silently skipped, not an error). `lo`/`hi` fall back to mean±sd
    when neither of the two HDI-column spellings arviz has used across
    versions ("hdi_3%"/"hdi_97%", the long-standing default 94% interval,
    or an older/alternate "hdi_2.5%"/"hdi_97.5%") is present."""
    rows = []
    for name in var_names:
        stats = posterior_summary.get(name)
        if stats is None:
            continue
        mean = float(stats.get("mean", float("nan")))
        lo = stats.get("hdi_3%", stats.get("hdi_2.5%"))
        hi = stats.get("hdi_97%", stats.get("hdi_97.5%"))
        if lo is None or hi is None:
            sd = float(stats.get("sd") or 0.0)
            lo, hi = mean - sd, mean + sd
        rows.append(dict(name=name, mean=mean, lo=float(lo), hi=float(hi), r_hat=stats.get("r_hat")))
    return rows


def energy_diagnostic_arrays(idata) -> dict:
    """Pulls NUTS's own per-draw Hamiltonian energy values straight out
    of `idata.sample_stats["energy"]` (a plain xarray access -- part of
    arviz's core InferenceData DATA model, not its plotting API, so
    unaffected by the API-shape issue this section's own module comment
    describes) and derives the two distributions the standard "energy
    plot" (BFMI diagnostic) overlays:

    - "marginal": every chain/draw's energy value, flattened into one
      array and re-centered on its own mean -- how spread out the
      sampler's total energy was.
    - "transition": the first differences of energy WITHIN each chain
      (consecutive-draw energy changes, computed per chain so a jump
      across the chain boundary is never counted, then flattened
      together) -- how big a jump NUTS's own transitions made.

    A transition distribution much narrower than the marginal one flags
    a sampler that struggled with this posterior's geometry (low BFMI) --
    something MAMBA never surfaced to the user at all.

    Raises CalibDiagnosticsError if `idata` has no `sample_stats.energy`
    array in the expected (chains, draws) shape (e.g. a placeholder/test
    idata, or a sampler backend that didn't record it)."""
    try:
        energy = np.asarray(idata.sample_stats["energy"].values, dtype=float)
    except (AttributeError, KeyError, TypeError) as exc:
        raise CalibDiagnosticsError(
            "This result has no recorded NUTS energy diagnostics (idata.sample_stats['energy'])."
        ) from exc
    if energy.ndim != 2 or energy.size == 0:
        raise CalibDiagnosticsError(
            "idata.sample_stats['energy'] has an unexpected shape -- expected (chains, draws)."
        )
    marginal = energy.reshape(-1)
    marginal = marginal - marginal.mean()
    transition = np.diff(energy, axis=1).reshape(-1)
    return {"marginal": marginal, "transition": transition}
