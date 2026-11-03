"""
Turns a "Calibration tests" table row -- a user-picked assignment of one
`CovarianceLibraryEntry` (core/project.py) per equation variable role plus
one for the modeled/output variable -- into a `CalibTestDataset` the
calibration engine (core/calib_pymc_model.py's `build_model`, and the
plain-numpy forward-preview path) can consume directly.

This module is pure numpy/stdlib -- no PySide6, no pymc/pytensor -- so it
is fully real-testable in this sandbox, same rationale as core/calib_
equation.py.

Why interpolate onto the OUTPUT variable's own grid specifically: the
likelihood (`pm.MvNormal(..., observed=stress_j)`, see core/calib_pymc_
model.py) is built against the modeled/output variable's own measured
covariance matrix, which is only meaningful on the grid it was actually
computed on (its own NxN structure isn't something you can resample onto
a different grid without recomputing it from raw draws, which the library
entry doesn't keep). Every OTHER assigned variable (e, v, T, a, b, d) is
just a plain per-point value the equation needs at each of those same
grid points, so it's fine -- and necessary, for the equation to evaluate
elementwise -- to linearly interpolate each one's own (possibly
different) grid onto the output's.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class LibraryAdapterError(ValueError):
    """Raised for anything wrong with a calibration test's own data
    assignment: a variable's library entry not covering the output grid's
    range, or (in `per_test_or_shared_covariance`) mismatched covariance
    shapes when a shared/pooled covariance was requested."""


@dataclass
class CalibTestDataset:
    """One calibration (or validation) test's fully-resolved data, ready
    to hand to core/calib_equation.evaluate() (via `variables`, merged
    with the current free-parameter values) and to the MvNormal
    likelihood (`observed`/`covariance`)."""
    label: str
    grid: np.ndarray                  # the output variable's own common x-grid, length N
    variables: dict                   # {role name (e.g. "e","v","T"): length-N np.ndarray, interpolated onto `grid`}
    observed: np.ndarray              # output variable's own mean curve on `grid`, length N -- the likelihood's data
    covariance: np.ndarray            # output variable's own N x N covariance matrix (NOT interpolated -- see module docstring)
    is_validation: bool = False       # True for a "Validation only" row -- excluded from the likelihood by core/calib_run.py
    output_entry_name: str = ""       # provenance, for display
    variable_entry_names: dict = field(default_factory=dict)  # {role: entry name}, for display


def entries_by_name(entries: list) -> dict:
    """{entry.name: entry} for a `Project.covariance_library` list (or any
    list of CovarianceLibraryEntry) -- a small convenience for UI combo
    population and for `build_test_dataset`'s own name-based lookups."""
    return {e.name: e for e in entries}


def build_test_dataset(
    label: str,
    variable_entry_map: dict,
    output_entry,
    is_validation: bool = False,
    zero_start_vars=None,
    zero_start_output: bool = False,
) -> CalibTestDataset:
    """Builds one `CalibTestDataset` from a row's own assignment.

    `variable_entry_map`: {role name (e.g. "e", "v", "T", "a", "b", "d"):
    CovarianceLibraryEntry} -- one entry per equation variable this test
    actually uses (only the roles the parsed equation's `used_vars`
    requires need to be present; extra keys are harmless and ignored by
    the equation evaluator, which only reads what it needs).

    `output_entry`: the CovarianceLibraryEntry for the modeled/output
    variable (e.g. stress) -- defines this test's own common grid,
    observed curve, and covariance matrix.

    `zero_start_vars`: an iterable of role names (e.g. `{"e", "T"}`) --
    each ROLE named here has its own assigned curve's `mean` translated
    by its own first sample (`mean - mean[0]`) BEFORE interpolation, so
    it starts at exactly 0 at its own native first grid point. A role NOT
    named here is left exactly as saved. `zero_start_output`: same idea,
    independently, for the output entry's own `observed` curve. Two
    separate controls -- not one blanket switch -- because a user may
    only want to offset-correct, say, a strain channel with known
    gauge-zero drift while leaving temperature or the output untouched.
    A pure translation does not change a curve's variance, so the
    output's covariance matrix is left untouched either way.

    Special case for role "e": every `CovarianceLibraryEntry` in this
    app (see ui/pages/page_covariance.py's `_on_compute`) is built with
    its `grid` axis set to the same test's own strain channel
    (`nominal_strain`/`strain`/`strain_laser`), no matter which physical
    quantity (stress, temperature, ...) is stored as its `mean` --
    meaning `output_entry.grid` (this function's `out_grid`, returned as
    `CalibTestDataset.grid` and used as the x-axis on every plot that
    shows this dataset) numerically IS strain, the same physical
    quantity as the reserved equation variable "e". So when "e" is in
    `zero_start_vars`, the returned `grid` is translated by its own
    first sample too (`out_grid - out_grid[0]`), so the strain axis
    itself reads from 0 -- not just the separately-assigned "e" curve's
    own values (which feed the equation, not the x-axis). This is
    display-only: interpolation below still happens in the original,
    unshifted grid coordinates (matching how each assigned variable's
    own grid was actually measured), and the equation evaluator never
    reads `dataset.grid` at all (only `dataset.variables`), so this has
    no effect whatsoever on the calibration's own numerics -- only on
    where the x-axis of the Per-test-fit/Validation plots starts.

    Raises LibraryAdapterError if a variable's own grid doesn't cover the
    output grid's [min, max] range (interpolation would otherwise
    silently extrapolate, which core/calib_equation.evaluate would then
    feed into the model as if it were real data)."""
    zero_start_vars = set(zero_start_vars or ())
    out_grid = np.asarray(output_entry.grid, dtype=float)
    if out_grid.size < 2:
        raise LibraryAdapterError(
            f"Test '{label}': the output entry '{output_entry.name}' has fewer than 2 grid points."
        )
    observed = np.asarray(output_entry.mean, dtype=float)
    if zero_start_output and observed.size:
        observed = observed - observed[0]
    covariance = np.asarray(output_entry.covariance, dtype=float)
    if covariance.shape != (out_grid.size, out_grid.size):
        raise LibraryAdapterError(
            f"Test '{label}': the output entry '{output_entry.name}' has a "
            f"{covariance.shape} covariance matrix, expected {(out_grid.size, out_grid.size)} "
            f"(one row/column per grid point)."
        )

    variables = {}
    variable_entry_names = {}
    for role, entry in variable_entry_map.items():
        var_grid = np.asarray(entry.grid, dtype=float)
        var_mean = np.asarray(entry.mean, dtype=float)
        if role in zero_start_vars and var_mean.size:
            var_mean = var_mean - var_mean[0]
        if var_grid.size < 2:
            raise LibraryAdapterError(
                f"Test '{label}': the entry '{entry.name}' assigned to variable '{role}' has "
                f"fewer than 2 grid points."
            )
        if out_grid.min() < var_grid.min() or out_grid.max() > var_grid.max():
            raise LibraryAdapterError(
                f"Test '{label}': variable '{role}''s entry '{entry.name}' covers "
                f"[{var_grid.min():.4g}, {var_grid.max():.4g}], which does not fully cover the "
                f"output entry '{output_entry.name}''s range [{out_grid.min():.4g}, {out_grid.max():.4g}]. "
                f"Pick a '{role}' entry with a wider grid, or a narrower output grid."
            )
        variables[role] = np.interp(out_grid, var_grid, var_mean)
        variable_entry_names[role] = entry.name

    # Display-only: shift the RETURNED grid label by its own first sample
    # when "e" (strain, the same physical quantity this grid always is --
    # see the docstring above) is checked for zero-start. Interpolation
    # above already happened in the original, unshifted coordinates, so
    # this cannot change which points landed where -- only what x-value
    # they're labeled with.
    display_grid = out_grid - out_grid[0] if ("e" in zero_start_vars and out_grid.size) else out_grid

    return CalibTestDataset(
        label=label, grid=display_grid, variables=variables, observed=observed, covariance=covariance,
        is_validation=is_validation, output_entry_name=output_entry.name,
        variable_entry_names=variable_entry_names,
    )


COVARIANCE_MODES = ("per_test", "shared_mean")
COVARIANCE_MODE_LABELS = {
    "per_test": "Per-test (default -- each test's own measured covariance)",
    "shared_mean": "Shared mean (average covariance across every calibration test)",
}

DEFAULT_COVARIANCE_JITTER = 0.0  # no regularization by default -- see regularize_covariance's docstring


def covariance_condition_number(cov: np.ndarray) -> float:
    """The covariance matrix's own condition number (largest / smallest
    eigenvalue) -- a plain, well-known measure of how numerically
    reliable its exact inverse (what `pm.MvNormal(..., chol=...)`'s
    likelihood effectively uses) is. A matrix built from measured/Monte-
    Carlo draws (ui/pages/page_covariance.py's `_on_compute`, `n_draws_
    used` draws resampled onto `n_points` grid points) can come out
    numerically SEVERELY ill-conditioned -- with only a few hundred to a
    few thousand draws spread very smoothly/correlated across many grid
    points, the sample covariance can have one or two dominant modes and
    a long tail of near-zero-variance directions that are really just
    finite-sample noise, not real structure.

    This matters a great deal here specifically because the likelihood
    literally inverts this matrix: a condition number in the tens of
    millions (confirmed against a real user's own measured stress-
    covariance, condition number ~4.6e7) means the exact GLS/MvNormal fit
    criterion is dominated by an almost-arbitrary, physically meaningless
    near-null direction rather than the data's own visually-obvious
    shape -- which can pull a calibration's free parameters to wildly
    wrong, sometimes boundary-pinned values while the sampler still
    reports excellent R-hat (it converged just fine onto the wrong
    answer, because the answer the exact likelihood actually wants IS
    that wrong-looking value). See `regularize_covariance`'s docstring
    for the fix. Returns `float('inf')` for a non-positive-definite
    matrix (should not normally occur here, since `np.linalg.cholesky`
    would already have rejected it upstream, but this function may be
    called on a matrix before that check)."""
    eigvals = np.linalg.eigvalsh(np.asarray(cov, dtype=float))
    if eigvals[0] <= 0:
        return float("inf")
    return float(eigvals[-1] / eigvals[0])


def regularize_covariance(cov: np.ndarray, jitter_frac: float) -> np.ndarray:
    """Adds `jitter_frac * mean(diag(cov))` to `cov`'s own diagonal --
    standard "nugget"/ridge regularization for a numerically ill-
    conditioned empirical covariance matrix (see `covariance_condition_
    number`'s docstring for why this app's own measured covariances can
    need it). `jitter_frac <= 0` (the default everywhere in this app)
    returns `cov` completely unchanged, so nothing about an already
    well-conditioned covariance matrix's behavior changes unless the
    user explicitly opts in via the Sampler settings card's "Covariance
    jitter" field.

    Verified against a real user's pathological case (condition number
    ~4.6e7): `jitter_frac=0.01` (1%) brings it down to ~2.3e4, and
    `jitter_frac=0.1`-`0.3` (10-30%) brings the resulting calibration fit
    into close visual agreement with a plain unweighted least-squares fit
    of the same equation to the same data -- confirming this is a real,
    effective fix, not just a cosmetic number change. This is a genuine
    modeling choice (it slightly weakens how strongly the measured
    covariance's exact structure constrains the fit), which is why it is
    surfaced as an explicit, user-adjustable setting rather than a
    silent, hardcoded epsilon."""
    jitter_frac = float(jitter_frac)
    cov = np.asarray(cov, dtype=float)
    if jitter_frac <= 0:
        return cov
    mean_diag = float(np.mean(np.diag(cov)))
    return cov + jitter_frac * mean_diag * np.eye(cov.shape[0])


def per_test_or_shared_covariance(tests: list, mode: str = "per_test", jitter_frac: float = 0.0) -> list:
    """Returns one covariance matrix per test in `tests` (same order),
    either each test's own (`mode="per_test"`, the new default -- fixes
    MAMBA's bug of always using the across-tests mean even when per-test
    covariance was available) or one shared matrix, the elementwise mean
    across every test's own matrix (`mode="shared_mean"`, MAMBA's old
    always-on behavior, kept as an explicit opt-in).

    `jitter_frac`: forwarded to `regularize_covariance` and applied to
    every returned matrix (after per-test/shared-mean resolution, before
    it ever reaches `pm.MvNormal`'s Cholesky factor) -- see that
    function's docstring. Defaults to 0 (no change at all), matching
    this function's own pre-existing behavior.

    `tests` should only contain calibration tests (not "Validation only"
    rows -- core/calib_run.py filters those out before calling this,
    since a validation test contributes no likelihood term at all)."""
    if mode not in COVARIANCE_MODES:
        raise LibraryAdapterError(f"Unknown covariance mode '{mode}'. Expected one of {COVARIANCE_MODES}.")
    if not tests:
        raise LibraryAdapterError("No calibration tests to resolve a covariance mode for.")
    if mode == "per_test":
        resolved = [t.covariance for t in tests]
    else:
        shapes = {t.covariance.shape for t in tests}
        if len(shapes) > 1:
            raise LibraryAdapterError(
                "Shared-mean covariance requires every calibration test's output covariance matrix to "
                f"have the same shape (same number of grid points N); got shapes {sorted(shapes)}. "
                "Use the same N (Covariance Analysis page) for every test's output entry, or switch to "
                "Per-test covariance."
            )
        mean_cov = np.mean(np.stack([t.covariance for t in tests], axis=0), axis=0)
        resolved = [mean_cov for _ in tests]
    return [regularize_covariance(cov, jitter_frac) for cov in resolved]
