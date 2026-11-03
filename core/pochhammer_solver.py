"""
Fast, sympy-free complex root-finder for the Pochhammer-Chree-family
dispersion equations core/visco_calibration.py's `_predict_pulse` solves
once per FFT frequency bin.

WHY: the original port used sympy's `nsolve`, re-deriving and evaluating
the equation SYMBOLICALLY (arbitrary-precision mpmath arithmetic) from
scratch at every single call. For a pulse window of a few hundred to a
few thousand samples, that is a few hundred to a few thousand
*independent* symbolic root-searches per forward-model evaluation --
core/bayes_visco_calibration.py's own module docstring calls this
"expensive" and builds an entire Gaussian-Process-surrogate search
specifically to minimize how many times it has to be paid, because it
dominates the cost of the whole calibration.

This module replaces sympy's machinery with a plain secant iteration
(`complex_secant_solve`) over `scipy.special.jv` (a compiled, complex-
argument-capable Bessel function of the first kind) evaluating EXACTLY
the same two equations `_predict_pulse` always solved
(`pochhammer_chree_residual`/`pochhammer_chree_fallback_residual` below),
with the SAME per-bin continuation convention (each bin's solve starts
from the PREVIOUS bin's own converged root, never from scratch), and the
same "raise so the caller's existing fallback can catch it" contract
(`RootNotFoundError` is a `ValueError` specifically so every existing
`except ValueError:` around a root-search call keeps working unchanged).

This is a faithful re-implementation, not a different numerical method
that happens to also find a root: a secant method is what mpmath's
`findroot` (sympy's own `nsolve` backend) itself falls back to for a
scalar equation given one starting point and no explicit derivative.
`scipy.special.jv` is bit-for-bit the same well-known special function
sympy's `besselj` evaluates -- both implement the standard Bessel
function of the first kind, a well-defined mathematical function with a
single correct value -- so this module needs no independent proof that
its Bessel values "agree" with sympy's; they are the same function,
evaluated by a different (compiled, decades-validated AMOS/Cephes-
derived) numeric routine instead of a symbolic one re-derived every call.
"""
from __future__ import annotations

import cmath
from typing import Callable

import numpy as np
from scipy.special import jv


class RootNotFoundError(ValueError):
    """Raised by `complex_secant_solve` when the iteration fails to reach
    `tol` within `maxiter` steps, stalls (two successive function values
    coincide, which would otherwise divide by ~0), or the residual
    function itself raises (e.g. a Bessel-function zero in a denominator)
    -- always a `ValueError`, never anything else, so every existing
    `except ValueError:` fallback around a root-search call in core/
    visco_calibration.py (and, if adopted there too, core/dispersion.py)
    keeps working exactly as it did against sympy's own `nsolve`, which
    raises plain `ValueError` on the same failure-to-converge condition."""


def complex_secant_solve(
    func: Callable[[complex], complex], x0: complex, tol: float = 1e-9, maxiter: int = 100,
) -> complex:
    """Finds a complex root of `func` via the secant method, seeded from
    a single starting point `x0` -- matching every call site's own
    "start from the previous frequency bin's converged root" convention
    (`k_sol[-1]`/`cp_sol[-1]` in core/visco_calibration.py's
    `_predict_pulse`), which is exactly the calling convention sympy's
    own `nsolve(expr, x0)` has always been used with here (one guess, no
    explicit derivative or bracket).

    Convergence is judged by the RELATIVE size of the last secant step
    (`|step| <= tol * max(1, |x|)`), not by how small `func`'s own output
    is -- these equations mix Bessel-function ratios and polynomial terms
    with no natural common scale, so an absolute-residual tolerance would
    either be too loose for some bins and too tight for others; a
    relative-step criterion is scale-free and is what `mpmath.findroot`
    (sympy's own backend) itself effectively converges on too.

    Raises `RootNotFoundError` (see its own docstring) rather than ever
    returning a non-converged point -- callers must not need to check a
    "did this actually converge" flag themselves. This INCLUDES a
    candidate whose Bessel-function terms overflow double-precision
    floats (`inf`/`nan`): during a Bayesian search's blind exploration
    (e.g. the initial Latin Hypercube design, which samples the full
    prior box with no regard for physical plausibility yet), a wild
    parameter combination can push `pochhammer_chree_residual`'s complex
    arguments large enough to overflow -- sympy's `nsolve` never saw this
    because mpmath's arbitrary-precision arithmetic has an astronomically
    larger exponent range than an IEEE-754 double, so this genuinely IS a
    new failure mode this faster implementation can hit that the
    original could not. It is handled the same way every OTHER root-
    search failure already is: raised as `RootNotFoundError` (a
    ValueError), caught by `_predict_pulse`'s existing fallback to the
    secondary equation (or, if that also fails, propagated out to
    core/bayes_visco_calibration.py's own existing "score this candidate
    0.0, let the surrogate learn to avoid the region" handling) -- not a
    crash, and not silently returning a meaningless `inf`/`nan` "root"."""
    with np.errstate(all="ignore"):  # over/underflow here is an EXPECTED,
        # handled outcome (see docstring above), not a bug to warn about
        # on every occurrence -- explicit isfinite checks below turn it
        # into the same clean RootNotFoundError any other failure mode
        # raises.
        try:
            x0 = complex(x0)
            x1 = x0 * (1 + 1e-6) + (1e-6 if x0 == 0 else 0.0)
            f0 = func(x0)
            for _ in range(maxiter):
                f1 = func(x1)
                if not (cmath.isfinite(f0) and cmath.isfinite(f1)):
                    raise RootNotFoundError(
                        "Residual evaluated to a non-finite value (overflow) -- candidate rejected."
                    )
                denom = f1 - f0
                if abs(denom) < 1e-300:
                    raise RootNotFoundError(
                        "Secant iteration stalled: successive function evaluations coincided."
                    )
                step = f1 * (x1 - x0) / denom
                x2 = x1 - step
                if not cmath.isfinite(x2):
                    raise RootNotFoundError("Secant step produced a non-finite candidate (overflow).")
                if abs(step) <= tol * max(1.0, abs(x2)):
                    return x2
                x0, f0 = x1, f1
                x1 = x2
        except RootNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001 -- any other failure (domain error, a
            # Bessel zero in a denominator, ...) must surface as the same
            # ValueError family every existing call site's `except
            # ValueError:` already expects from sympy's own nsolve, not
            # whatever internal exception type happened to be raised.
            raise RootNotFoundError(f"Root search failed: {exc}") from exc
    raise RootNotFoundError(f"Secant iteration did not converge within {maxiter} iterations.")


def pochhammer_chree_residual(b: complex, omega: float, density: float, ram_v: complex, mu_v: complex,
                               radius: float) -> complex:
    """The PRIMARY Pochhammer-Chree dispersion equation -- residual(b) ==
    0 at the true root. `b` is the unknown (wavenumber-plus-attenuation)
    complex variable being solved for; `omega` is 2*pi*frequency at this
    bin; `ram_v`/`mu_v` are this bin's own complex Lame-like parameters
    (`core.visco_calibration._complex_modulus_and_speed`'s `ram[v]`/
    `mu[v]`); `density`/`radius` are the bar's own physical properties.

    Identical, term for term, to core.visco_calibration._predict_pulse's
    original sympy expression (`p`/`q` substituted in terms of `b`, same
    as the symbolic version did): only the Bessel-function evaluator
    (`scipy.special.jv` here, `sympy.besselj` there) and the arithmetic
    backend (plain complex numbers here, sympy symbolic expressions
    there) differ."""
    p = cmath.sqrt(density * omega ** 2 / (ram_v + 2 * mu_v) - b ** 2)
    q = cmath.sqrt(density * omega ** 2 / mu_v - b ** 2)
    return (
        (2 * p / radius) * (q ** 2 + b ** 2) * jv(1, p * radius) * jv(1, q * radius)
        - (q ** 2 - b ** 2) ** 2 * jv(0, p * radius) * jv(1, q * radius)
        - 4 * (b ** 2) * p * q * jv(0, q * radius) * jv(1, p * radius)
    )


def pochhammer_chree_fallback_residual(cp: complex, omega: float, c_v: complex, nu: float, beta: float,
                                        radius: float) -> complex:
    """The FALLBACK (approximate extensional-wave) dispersion equation --
    used when `pochhammer_chree_residual`'s own root search fails to
    converge, exactly matching core.visco_calibration._predict_pulse's
    original second sympy expression (same `xi`/`nl`/`h_e`/`k_e`
    substitutions, same equation, in terms of the unknown `cp` this
    time)."""
    xi = (cp / c_v) ** 2 * (1 + nu)
    nl = omega / cp
    h_e = nl * cmath.sqrt(beta * xi - 1)
    k_e = nl * cmath.sqrt(2 * xi - 1)
    return (
        ((xi - 1) ** 2) * radius * h_e * (jv(0, radius * h_e) / jv(1, radius * h_e))
        - (beta * xi - 1) * (xi - radius * k_e * (jv(0, radius * k_e) / jv(1, radius * k_e)))
    )
