"""
Viscoelastic bar model calibration, ported from calib_visco_model.py and
later generalized from a single hard-coded Standard-Linear-Solid modulus to
any model in core/rheology_models.py.

Given a candidate rheological model and parameter set, `predict_pulse`
predicts what a second free-decay pulse (recorded further along the bar)
should look like given the first one (using the same Pochhammer-Chree
dispersion physics as core/dispersion.py's `viscoelastic_correction`), and
`pulse_correlation` scores that prediction against the actually-measured
second pulse using whichever goodness-of-fit metric is selected (see
SCORE_FUNCS below -- Pearson correlation is the metric the function is
named after, but not the only one it can compute). `test_parameters` runs
both for one fixed parameter set, for verifying a candidate (e.g. one
already found, or from another source). The Bayesian search over
parameters (core/bayes_visco_calibration.py) and the page's "Test
parameters" mode are both built on exactly these two functions.

Two scoring metrics are offered (SCORE_FUNCS/SCORE_LABELS):

- "correlation" (Pearson r, the original and only metric before this):
  measures whether the predicted pulse's *shape* tracks the measured one.
  It is invariant to an affine transform of either signal, which is
  exactly its blind spot for this use -- a candidate whose predicted
  dispersion/attenuation is shaped right but whose absolute amplitude is
  wrong (e.g. the wrong damping level, or a modulus off by a roughly
  constant factor across the band) can still score close to 1, because
  Pearson r never actually compares the two pulses' amplitudes.
- "r2" (coefficient of determination, R² = 1 - SS_res/SS_tot, equivalently
  related to the normalized RMSE): DOES penalize an amplitude or offset
  mismatch, since it compares actual values, not just their covariance
  structure. This is the more physically meaningful criterion for
  calibrating parameters that directly set the pulse's absolute amplitude
  and attenuation (the moduli, viscosities, and damping factor) rather
  than just its shape, so it is the default.

Only the model's complex modulus E*(f) (`model.complex_modulus`, see
core/rheology_models.py) varies between rheological models -- the Lame-like
mu/lambda, the per-frequency Pochhammer-Chree root search and the
FFT-domain pulse propagation below are generic in E*(f) and unchanged by
which model produced it.

NOT YET independently re-validated against real experimental data or the
original GUI's output -- ported as literally as possible from the
original for traceability; cross-check results against the original
application before relying on this for a real calibration.

The per-frequency-bin root search itself (`_predict_pulse`'s two
dispersion equations) no longer uses sympy's `nsolve` -- see
core/pochhammer_solver.py's module docstring for why that was the
dominant cost of every forward-model evaluation (and therefore of the
whole Bayesian search in core/bayes_visco_calibration.py) and why
replacing it with a plain complex secant iteration over
`scipy.special.jv` is a faithful speed-up, not a different algorithm.
This also means `_predict_pulse` finally has no un-installable-in-this-
sandbox dependency left at all, so (unlike before) it IS now real-
testable here -- see test_visco_calibration.py.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.stats

from core.pochhammer_solver import (
    RootNotFoundError, complex_secant_solve, pochhammer_chree_fallback_residual, pochhammer_chree_residual,
)
from core.rheology_models import RheologyModel

ProgressCB = Optional[Callable[[int, int], None]]


@dataclass
class ViscoTestResult:
    score: float
    metric: str  # which entry of SCORE_FUNCS produced `score`
    time1: np.ndarray
    pulse1: np.ndarray
    time2: np.ndarray
    pulse2: np.ndarray
    predicted_pulse: np.ndarray


def _complex_modulus_and_speed(freq, nu, model: RheologyModel, params: dict, density):
    """The model's complex modulus E(f) and the resulting elastic-
    equivalent wave speed C(f), plus the Lame-like mu/lambda used by the
    Pochhammer-Chree root search below -- generic in which rheological
    model produced E(f)."""
    e_complex = model.complex_modulus(freq, params)
    mu = [e / (2 * (1 + nu)) for e in e_complex]
    ram = [nu * e / ((1 + nu) * (1 - 2 * nu)) for e in e_complex]
    c = [cmath.sqrt(e / density) for e in e_complex]
    return mu, ram, c


def _predict_pulse(pulse1, time1, nu, model: RheologyModel, params: dict, density, radius, z1, damp,
                    progress_cb: ProgressCB = None):
    """Predicts what a second pulse (recorded a distance `z1` further along
    the bar) should look like given `pulse1`, for one candidate rheological
    model and parameter set -- the same physics as core/dispersion.py's
    `viscoelastic_correction`, applied to a single free-decay pulse instead
    of an incident/reflected/transmitted triple."""
    dt = 0.0
    while dt == 0.0:
        dt = time1[1] - time1[0]
    n = len(pulse1)

    yi = np.fft.rfft(pulse1)
    freq = np.fft.rfftfreq(n, d=dt)
    amplitude = np.abs(yi)
    phase = np.angle(yi)

    mu, ram, c = _complex_modulus_and_speed(freq, nu, model, params, density)

    beta = (1 - 2 * nu) / (1 - nu)
    cp_sol = [c[0]]
    b_sol = [0]
    k_sol = [0, (2 * cmath.pi * freq[1]) / c[1]]
    atn = [0]

    lfreq = len(freq)
    total = lfreq - 1
    for v in range(total):
        omega = 2 * cmath.pi * freq[v + 1]
        try:
            dsol = complex_secant_solve(
                lambda b: pochhammer_chree_residual(b, omega, density, ram[v + 1], mu[v + 1], radius),
                k_sol[-1],
            )
            b_sol.append(dsol)
            k_sol.append(float(dsol.real))
            cp_sol.append(2 * math.pi * freq[v + 1] / k_sol[-1])
            atn.append(float(abs(dsol.imag)))
        except RootNotFoundError:
            dsol = complex_secant_solve(
                lambda cp: pochhammer_chree_fallback_residual(cp, omega, c[v + 1], nu, beta, radius),
                cp_sol[-1],
            )
            cp_sol.append(float(dsol.real))
            k_sol.append(2 * math.pi * freq[v + 1] / cp_sol[-1])
            atn.append(atn[-1])
            b_sol.append(k_sol[-1] - atn[-1] * 1j)
        if progress_cb:
            progress_cb(v + 1, total)

    # c[0] (the DC/zero-frequency reference speed) is always numerically
    # real for every core.rheology_models built-in model -- each one's own
    # complex_modulus is real at zero frequency by construction (see e.g.
    # _burgers_modulus's own docstring on why) -- but stays *typed* as a
    # Python complex (cmath.sqrt always returns one). `.real` here makes
    # that already-true fact explicit, keeping new_phase a plain real
    # float feeding cmath.rect's strictly-real phase argument below,
    # instead of a complex-typed-but-numerically-real value that numpy
    # would otherwise silently (if noisily) narrow for us.
    c0_real = c[0].real
    new_phase = [phase[a] - (2 * cmath.pi * freq[a] * z1 / c0_real) * ((c0_real / abs(cp_sol[a]) - 1)) for a in range(len(phase))]
    new_pulse_fft = [
        cmath.exp(-(damp / c[0]) * z1) * amplitude[w] * cmath.exp(-atn[w] * z1) * cmath.rect(1, new_phase[w])
        for w in range(len(new_phase))
    ]
    # `n=n` is required, not optional: numpy's rfft/irfft pair cannot tell
    # from the spectrum alone whether the original real signal had an even
    # or odd number of samples (a length-m spectrum is the rfft of either a
    # 2*(m-1)-sample or a (2*m-1)-sample signal), so it defaults to
    # assuming even length. `pulse1` (and therefore `n`, `freq`, `amplitude`
    # and `phase` above) can just as easily be odd-length, since it comes
    # from a user-dragged time window, not a padded/fixed-size buffer. Left
    # to the default, an odd-length window silently produces a pulse that
    # is both one sample short AND numerically wrong throughout (not just
    # truncated -- verified: up to ~2 in a signal of order 1 for a 139-
    # sample window, not a rounding-sized error), because irfft's implicit
    # even-length assumption reconstructs the wrong signal, not merely a
    # truncated correct one. That one-sample shortfall is what surfaced as
    # a plot crash ("x and y must have same first dimension") for whichever
    # model happened to run on an odd-length pulse window -- passing `n=n`
    # (the length `np.fft.rfft(pulse1)` above was actually given) fixes the
    # numerical result, and the length match is then just a consequence.
    new_pulse = np.fft.irfft(new_pulse_fft, n=n).real
    return new_pulse


def _align_pulses(new_pulse: np.ndarray, pulse2: np.ndarray, time1: np.ndarray, time2: np.ndarray):
    """Overlaps the predicted pulse with the measured second pulse over the
    region both cover -- the original windowing rule (`time2 >= 0` /
    `time1 < max(time2)`), unchanged. Every scoring metric below is built
    on top of this shared alignment, so choosing a different metric never
    has to touch (or risk breaking) the windowing logic."""
    if len(new_pulse) < len(pulse2):
        new_pulse = np.concatenate((new_pulse, [new_pulse[-1]]))

    w1 = 0
    w2 = len(time1) - 1
    for h in range(len(pulse2)):
        if h < len(time2) and time2[h] < 0:
            w1 = h
        if h < len(time1) and time1[h] < max(time2):
            w2 = h

    pulse2_c = pulse2[w1:len(pulse2)]
    new_pulse_c = new_pulse[0:w2]
    while len(new_pulse_c) < len(pulse2_c):
        new_pulse_c = np.concatenate((new_pulse_c, [new_pulse_c[-1]]))
    new_pulse_c = new_pulse_c[: len(pulse2_c)]
    return new_pulse_c, pulse2_c


def _pearson_score(pred: np.ndarray, meas: np.ndarray) -> float:
    """Pearson correlation coefficient -- see this module's docstring for
    why this is shape-only and blind to amplitude/offset errors."""
    return abs(scipy.stats.pearsonr(meas, pred)[0])


def _r2_score(pred: np.ndarray, meas: np.ndarray) -> float:
    """Coefficient of determination R² = 1 - SS_res/SS_tot against the
    measured pulse's own mean as baseline -- see this module's docstring.
    Clipped to [-1, 1]: unlike Pearson r, R² is unbounded below (a bad
    enough miss can score far less than -1), and an unbounded score would
    let one pathological candidate dominate the Gaussian Process
    surrogate's fitted length scale during Bayesian optimization; -1
    already means "twice as wrong as just predicting the mean", which is
    plenty bad for the optimizer to reject the region."""
    ss_res = float(np.sum((meas - pred) ** 2))
    ss_tot = float(np.sum((meas - np.mean(meas)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return float(np.clip(1.0 - ss_res / ss_tot, -1.0, 1.0))


SCORE_FUNCS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "r2": _r2_score,
    "correlation": _pearson_score,
}
SCORE_LABELS: dict[str, str] = {
    "r2": "R² / NRMSE (shape AND amplitude match — recommended)",
    "correlation": "Pearson correlation (shape match only, ignores amplitude/offset errors)",
}
SCORE_SHORT_LABELS: dict[str, str] = {"r2": "R²", "correlation": "Correlation"}
SCORE_ORDER = ("r2", "correlation")
DEFAULT_METRIC = "r2"


def _score(new_pulse: np.ndarray, pulse2: np.ndarray, time1: np.ndarray, time2: np.ndarray,
           metric: str = DEFAULT_METRIC) -> float:
    pred_c, meas_c = _align_pulses(new_pulse, pulse2, time1, time2)
    return SCORE_FUNCS[metric](pred_c, meas_c)


def predict_pulse(pulse1, time1, nu, model: RheologyModel, params: dict, density, radius, z1, damp,
                   progress_cb: ProgressCB = None):
    """Public wrapper around `_predict_pulse`, so other calibration
    strategies (e.g. core/bayes_visco_calibration.py) reuse the exact same
    Pochhammer-Chree forward model instead of re-implementing or importing
    a private name."""
    return _predict_pulse(pulse1, time1, nu, model, params, density, radius, z1, damp, progress_cb)


def pulse_correlation(new_pulse: np.ndarray, pulse2: np.ndarray, time1: np.ndarray, time2: np.ndarray,
                       metric: str = DEFAULT_METRIC) -> float:
    """Public wrapper around `_score` (see `predict_pulse`) -- kept under
    its original name for existing callers; `metric` selects which of
    SCORE_FUNCS actually scores the comparison (default: R²)."""
    return _score(new_pulse, pulse2, time1, time2, metric)


def test_parameters(
    pulse1: np.ndarray, time1: np.ndarray, pulse2: np.ndarray, time2: np.ndarray,
    radius: float, density: float, z1: float, damp: float,
    nu: float, model: RheologyModel, params: dict,
    metric: str = DEFAULT_METRIC,
    progress_cb: ProgressCB = None,
) -> ViscoTestResult:
    """Evaluate one fixed (nu, model, params) directly."""
    predicted = _predict_pulse(pulse1, time1, nu, model, params, density, radius, z1, damp, progress_cb)
    score = _score(predicted, pulse2, time1, time2, metric)
    return ViscoTestResult(
        score=score, metric=metric,
        time1=time1, pulse1=pulse1, time2=time2, pulse2=pulse2, predicted_pulse=predicted,
    )
