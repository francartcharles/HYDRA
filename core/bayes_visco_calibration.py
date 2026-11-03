"""
Bayesian calibration of the viscoelastic bar model -- generic over which
rheological model (core/rheology_models.py) is being calibrated, and an
alternative to the app's fixed-grid search over the same
"predict pulse 2 from pulse 1, score by correlation" problem.

Why: the grid search's cost is `count_1 * count_2 * ... * count_n` forward-
model evaluations (one per parameter, including the Poisson ratio), and
each evaluation is itself nontrivial (one root-search per FFT frequency
bin, because the Pochhammer-Chree dispersion relation has no closed form
-- see core/pochhammer_solver.py's own docstring for that root-search's
current implementation). A grid fine enough to localize the correlation
peak precisely in up to 6 dimensions (the most complex model here,
Generalized Maxwell with 2 branches, has 5 parameters plus the Poisson
ratio) is usually thousands of evaluations; a coarse one can straddle
the true optimum between grid nodes without ever evaluating near it.

(Historical note: this module's forward-model evaluations used to be far
more expensive still -- `core.visco_calibration._predict_pulse` originally
solved each frequency bin's root via sympy's `nsolve`, symbolically re-
deriving and evaluating the equation from scratch every call. That was
replaced with a plain scipy-based secant iteration over the same
equations, roughly two to three orders of magnitude faster per call. The
Bayesian-optimization approach below remains the right one regardless --
the forward model is still a non-differentiable black box no gradient-
based sampler (e.g. PyMC's NUTS) can be pointed at directly, and it is
still far more expensive than the GP surrogate that proposes where to
evaluate it next -- but the specific "thousands of evaluations would be
prohibitively slow" framing is less extreme than when this module was
first written.)

This module instead runs Bayesian optimization:

1. A small initial design (Latin Hypercube Sampling -- core/gaussian_
   process.py) spreads the first, most informative evaluations evenly
   across the search box.
2. A Gaussian Process surrogate (same module) is fit to every
   (parameters -> correlation) pair evaluated so far. Fitting and
   predicting with the GP is cheap (microseconds to milliseconds); only
   the forward model is expensive.
3. The next parameter set to actually run through the expensive forward
   model is the one the chosen acquisition strategy scores highest over
   the surrogate -- by default Expected Improvement, i.e. the point most
   likely to beat the best correlation seen so far, balancing "near the
   current best" against "still very uncertain" (see "ACQUISITION
   STRATEGY" below for the other choices and when to prefer them).
4. This repeats until the best score stops improving for
   `convergence_patience` iterations, or `n_iter` is reached.

Two direct benefits for this specific problem: far fewer expensive forward-
model calls are needed to reach a given precision (tens instead of
thousands, regardless of how many parameters the chosen model has), and
parameter values are continuous rather than snapped to `linspace` steps, so
the search converges wherever the true correlation peak is rather than to
the nearest grid node ("freer convergence").

The reported uncertainty is also more principled than the grid search's
"grid points within 5% of the best" bookkeeping: once the search is done, a
last GP is fit to every evaluated point and its posterior mean is sampled
densely (tens of thousands of points) across the *entire continuous* search
box -- still cheap, since this never touches the expensive forward model --
and weighted by a softmax of the predicted score (see core/visco_
calibration.py's SCORE_FUNCS for which goodness-of-fit metric that is --
Pearson correlation or R²/NRMSE). The resulting weighted mean/std of each
parameter is the calibration posterior implied by the surrogate, at a
resolution no grid of a comparable evaluation budget could afford.

That same weighted posterior sample also gives a parameter *interdependence*
matrix "for free" (`ViscoBayesianResult.param_correlation`): the weighted
Pearson correlation between every pair of parameters across the sampled
posterior. This is a standard calibration diagnostic, not a plotting
extra -- a value near +-1 between two parameters means the two-pulse
correlation measurement mostly constrains some combination of them (e.g.
their product, for parameters that trade off multiplicatively) rather than
either one individually, and no amount of extra search budget fixes that;
only a different/additional measurement can. It says nothing about the
parameters being physically coupled, only that *this* dataset cannot tell
them apart.

PRIORS -- why a flat search box alone can leave huge final uncertainties
-------------------------------------------------------------------------
Without an explicit prior, every parameter's prior is implicitly a flat
(uniform) one across its Min/Max search box -- both the initial design and
the adaptive search only ever draw candidates uniformly (log-uniformly for
moduli/viscosities) across that box, and, critically, so does the posterior
estimate itself: the dense `n_posterior` sample used to compute the
reported mean/std is drawn uniformly across the *whole* box and only then
weighted by the surrogate's predicted score. When the score surface is
fairly flat over a large part of that box -- exactly what the "parameter
interdependence" diagnostic above
is warning about when it reports values near +-1 -- that weighting barely
narrows anything down, and the reported posterior std ends up almost as
wide as the box itself, regardless of how much search budget was spent.
More search does not fix this: it is a property of what the two pulses can
and cannot distinguish, not of how thoroughly the box was explored.

`PriorSpec` lets each parameter (the Poisson ratio and every parameter of
the selected rheological model) carry one of three priors instead:

- `"fixed"`: pinned at a single value and dropped from the search
  entirely (one fewer dimension for the optimizer/surrogate to cover).
- `"uniform"`: the implicit default -- flat between `lo` and `hi`
  (log-uniform for moduli/viscosities, linear for nu).
- `"gaussian"`: a Normal(`mean`, `std`) prior, in the parameter's own
  physical units -- e.g. "E1 around 2.0e9 Pa +/- 3.0e8 Pa" from a
  literature value, a datasheet, or an earlier calibration run.
- `"lognormal"`: a Log-Normal prior -- like `"gaussian"` but for a
  parameter's *ratio* uncertainty rather than its absolute one: instead of
  a Mean/Std in linear units, you give a Median and a one-sigma
  multiplicative factor `F` (> 1) -- about 68% of the prior mass falls
  between `Median / F` and `Median * F`. This is the natural choice for a
  modulus/viscosity you only know to within an order of magnitude: a huge
  *linear* Std would either put real probability mass on absurd or
  negative values, or need truncating to avoid it; a Log-Normal's samples
  are always strictly positive by construction, whatever `F` is.

Choosing "gaussian" or "lognormal" changes more than just the search's sampling box: the
posterior-estimation step (`U_post` below) draws its dense sample from that
same Normal instead of uniformly across the box, so the reported posterior
is a proper (self-normalized importance-sampling) estimate of
posterior ∝ prior x likelihood, not just prior ∝ likelihood restricted to a
box. Wherever the two pulses alone cannot pin a parameter down, the
posterior mean/std then falls back toward the prior's own mean/std instead
of spreading back out across an arbitrarily wide search box -- which is the
direct answer to "would a non-uniform prior help with these huge
uncertainties": yes, provided the Gaussian's/Log-Normal's mean/std (or
median/factor) reflects real prior knowledge about that parameter, not a
guess as wide as the uniform prior's search box (a guess that wide would
reproduce the same huge uncertainty, just reshaped).

HYPERPRIORS -- letting the prior's own center/spread be searched too
-------------------------------------------------------------------------
Every field above (`lo`/`hi` for `"uniform"`, `mean`/`std` for
`"gaussian"`, the median/factor pair for `"lognormal"`) can be given
either as a plain number, exactly as described above, or as a `(lo, hi)`
tuple -- a *hyperprior*: instead of committing to one fixed value for,
say, the Gaussian's mean, you give a plausible range for it and let the
same Bayesian search locate the best value inside that range jointly with
everything else, instead of having to pick it by hand up front.

Each such field turns into one extra dimension of the search: drawn log-
uniformly for a *spread* field (`std`, or the Log-Normal `factor`) --
"how wide should the uncertainty be" is itself naturally explored across
orders of magnitude -- and drawn the same way the parameter's own value
would be for a *location* field (`lo`/`hi`/`mean`, or the Log-Normal
median) -- log-uniformly for a modulus/viscosity, linearly for `nu`. The
parameter's actual value is then drawn from *that* candidate distribution,
exactly as it would be from a fixed one -- so a hyperprior Gaussian costs
up to 3 search dimensions total (mean, std, and the value itself) instead
of 1, and a hyperprior Uniform up to 3 as well (lo, hi, value). This is a
real cost: every extra dimension makes the search's job harder (the curse
of dimensionality this module was built to stay low-dimensional to avoid
-- see `_propose_next`'s docstring), so hyperpriors are worth enabling only
where there is genuine uncertainty about where to even center a prior, not
as a default, and usually call for a larger search budget (more Initial
samples / Max iterations) than the same parameter would need with a plain
prior.

The found hyperparameter values are reported alongside the physical
parameters (`ViscoBayesianResult.hyper_names`, each named
`"<param>__<field>"`, e.g. `"e1__mean"`) -- their own weighted-posterior
mean/std, from the very same posterior sample the physical parameters'
mean/std come from. They are not included in `param_correlation`, which
stays scoped to the physical parameters this module's diagnostics are
actually about.

ACQUISITION STRATEGY -- how the adaptive search picks its next point
-------------------------------------------------------------------------
Step 3 above needs one more ingredient: a rule for turning the surrogate's
(mean, std) at any candidate point into a single "how promising is this"
score, which `_propose_next` then maximizes. `calibrate_bayesian`'s
`acquisition` argument picks that rule -- one of `ACQ_EI` (the default),
`ACQ_PI`, `ACQ_UCB`, `ACQ_THOMPSON` (see `ACQUISITION_LABELS`/
`ACQUISITION_USE_CASES` for the short, UI-facing versions of the notes
below):

- `ACQ_EI` (Expected Improvement): the expected *size* of the improvement
  over the current best, in expectation over the surrogate's own
  uncertainty. Weighs "how much better" together with "how likely",
  which is why it is the balanced, general-purpose default here.
- `ACQ_PI` (Probability of Improvement): only the *probability* of any
  improvement at all, however small. Greedier/more exploitative than EI
  -- it stops caring once a region looks unlikely to beat the current
  best, even if a genuine improvement is still plausible there -- so it
  tends to converge in fewer evaluations when the search is already
  fairly well localized (e.g. by informative priors), but is more prone
  to settling on a local optimum when the search box is wide or the
  model is weakly identified.
- `ACQ_UCB` (Upper Confidence Bound): `mean + kappa * std` (`kappa = 2`
  here) -- explores purely by how uncertain the surrogate still is,
  without reference to the current best at all. Systematically visits
  under-explored regions more than EI/PI do, which suits a wide search
  box or a model prone to multiple competing optima (e.g. Generalized
  Maxwell/Burgers), at the cost of a few more evaluations before it
  settles on the best one.
- `ACQ_THOMPSON` (Thompson Sampling): draws one random continuous
  function from the surrogate's *joint* posterior (`GaussianProcess.
  sample_y`, not just its pointwise mean/std) and proposes that sample's
  own optimum. A stochastic middle ground between EI and UCB -- since
  each iteration's proposal comes from a different random draw, it
  naturally avoids repeatedly proposing near-duplicate points the way a
  deterministic acquisition function occasionally can, which makes it a
  useful thing to switch to if a search seems stuck (little to no
  improvement for several iterations despite iterations remaining).

Only the adaptive-search loop's acquisition changes -- the initial Latin
Hypercube design (`n_init`) and the final posterior estimation
(`X_post`/`Hyper_post`) are unaffected by this choice.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Union

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from core.gaussian_process import GaussianProcess, latin_hypercube
from core.rheology_models import RheologyModel
from core.visco_calibration import predict_pulse, pulse_correlation, DEFAULT_METRIC
from core.parallel import run_map, check_cancelled

ProgressCB = Optional[Callable[[int, int], None]]

PRIOR_FIXED = "fixed"
PRIOR_UNIFORM = "uniform"
PRIOR_GAUSSIAN = "gaussian"
PRIOR_LOGNORMAL = "lognormal"

# Acquisition strategy for the adaptive search loop -- see the module
# docstring's "ACQUISITION STRATEGY" section for what each one does and
# when to prefer it; the *_LABELS/*_USE_CASES dicts below are the short,
# UI-facing versions of that same explanation.
ACQ_EI = "ei"
ACQ_PI = "pi"
ACQ_UCB = "ucb"
ACQ_THOMPSON = "thompson"

ACQUISITION_LABELS = {
    ACQ_EI: "Expected Improvement",
    ACQ_PI: "Probability of Improvement",
    ACQ_UCB: "Upper Confidence Bound",
    ACQ_THOMPSON: "Thompson Sampling",
}

ACQUISITION_USE_CASES = {
    ACQ_EI: "balanced default — weighs how big an improvement could be, not just how likely",
    ACQ_PI: "greedier — converges fast once priors already narrow the search, more local-optimum-prone otherwise",
    ACQ_UCB: "explores uncertain regions more systematically — good for a wide box or a weakly-identified model",
    ACQ_THOMPSON: "stochastic — helps when the search seems stuck on near-duplicate proposals",
}

_UCB_KAPPA = 2.0  # exploration strength for ACQ_UCB, see the module docstring

# A tiny floor keeps a Gaussian prior from ever producing a non-positive
# modulus/viscosity/nu (physically meaningless, and several rheological
# models divide by these) even when a candidate's unit-hypercube coordinate
# lands in the Normal distribution's extreme tail -- it does not truncate
# the prior itself, just guards the one non-negotiable physical constraint
# every parameter here shares.
_GAUSSIAN_FLOOR_FRACTION = 1e-3
_GAUSSIAN_MIN_FLOOR = 1e-12


HyperField = Union[float, tuple[float, float]]  # a plain value, or a (lo, hi) hyperprior range


@dataclass(frozen=True)
class PriorSpec:
    """One parameter's prior for the Bayesian search -- see the module
    docstring's "PRIORS" and "HYPERPRIORS" sections for what each kind
    does and why it matters for the reported uncertainty. `kind` is one of
    `PRIOR_FIXED`, `PRIOR_UNIFORM`, `PRIOR_GAUSSIAN`, `PRIOR_LOGNORMAL`.

    `lo`/`hi`/`mean`/`std` each accept either a plain float (fixing that
    field to a single value) or a `(lo, hi)` tuple -- a hyperprior range the
    search explores that field within, instead of it being fixed by the
    caller. `PRIOR_LOGNORMAL` reuses the `mean`/`std` slots for its own
    median/factor (see `PriorSpec.lognormal`), so every prior kind other
    than `PRIOR_FIXED` needs only these same two pairs of fields.
    """
    kind: str
    value: float = 0.0       # PRIOR_FIXED
    lo: HyperField = 0.0     # PRIOR_UNIFORM
    hi: HyperField = 0.0     # PRIOR_UNIFORM
    mean: HyperField = 0.0   # PRIOR_GAUSSIAN (mean); PRIOR_LOGNORMAL (median)
    std: HyperField = 0.0    # PRIOR_GAUSSIAN (std); PRIOR_LOGNORMAL (one-sigma factor, > 1)
    log_scale: bool = True  # PRIOR_UNIFORM only; `calibrate_bayesian` derives
                             # the correct value for "nu" vs. a model parameter
                             # automatically, so callers do not need to set this.

    @staticmethod
    def fixed(value: float) -> "PriorSpec":
        return PriorSpec(kind=PRIOR_FIXED, value=value)

    @staticmethod
    def uniform(lo: HyperField, hi: HyperField) -> "PriorSpec":
        return PriorSpec(kind=PRIOR_UNIFORM, lo=lo, hi=hi)

    @staticmethod
    def gaussian(mean: HyperField, std: HyperField) -> "PriorSpec":
        return PriorSpec(kind=PRIOR_GAUSSIAN, mean=mean, std=std)

    @staticmethod
    def lognormal(median: HyperField, factor: HyperField) -> "PriorSpec":
        """`factor` is the one-sigma multiplicative factor (> 1): about
        68% of the prior mass falls within `[median / factor, median *
        factor]` -- see the module docstring's "PRIORS" section."""
        return PriorSpec(kind=PRIOR_LOGNORMAL, mean=median, std=factor)


def _check_field(value: "HyperField", label: str, min_value: float) -> None:
    """Validates one PriorSpec field -- either a plain number or a `(lo,
    hi)` hyperprior range -- against a lower bound the math downstream
    actually requires (e.g. > 0 before a log(), > 1 for a Log-Normal
    factor), and, for a range, that it is not inverted. Raises early with
    a specific message instead of surfacing a confusing NaN from deep
    inside the search."""
    if isinstance(value, tuple):
        lo, hi = float(value[0]), float(value[1])
        if lo <= min_value or hi <= min_value:
            raise ValueError(f"{label}: hyperprior range must be > {min_value} (got {value})")
        if lo >= hi:
            raise ValueError(f"{label}: hyperprior range must have lo < hi (got {value})")
    else:
        if float(value) <= min_value:
            raise ValueError(f"{label} must be > {min_value} (got {value})")


def _normalize_priors(param_priors: dict, param_names: tuple[str, ...]) -> dict:
    """Accepts either `PriorSpec` values or a bare `(lo, hi)` tuple/list
    (the calling convention before priors existed) for backward
    compatibility, and forces every PRIOR_UNIFORM entry's `log_scale` to
    the app's existing convention (linear for "nu", log for every model
    parameter) regardless of what the caller set, so that behavior is not
    left to be gotten wrong call-by-call. Also validates every field
    (including each end of a hyperprior range) against what the math for
    that prior kind actually requires -- see `_check_field`."""
    out = {}
    for name in param_names:
        spec = param_priors[name]
        if not isinstance(spec, PriorSpec):
            lo, hi = spec
            spec = PriorSpec.uniform(float(lo), float(hi))
        if spec.kind == PRIOR_UNIFORM:
            spec = replace(spec, log_scale=(name != "nu"))
            if name != "nu":  # nu is searched linearly and may legitimately span down toward 0
                _check_field(spec.lo, f"{name}.lo", 0.0)
                _check_field(spec.hi, f"{name}.hi", 0.0)
        elif spec.kind == PRIOR_GAUSSIAN:
            _check_field(spec.std, f"{name}.std", 0.0)
        elif spec.kind == PRIOR_LOGNORMAL:
            _check_field(spec.mean, f"{name}.median", 0.0)
            _check_field(spec.std, f"{name}.factor", 1.0)
        out[name] = spec
    return out


@dataclass
class ViscoBayesianResult:
    model_key: str
    param_names: tuple[str, ...]          # ("nu",) + model.param_keys()
    metric: str                            # which core.visco_calibration.SCORE_FUNCS entry scored every evaluation
    means: dict                            # name -> posterior weighted mean, for every param_names entry
                                            # AND every hyper_names entry below (same dict, superset of keys)
    stds: dict                             # name -> posterior weighted std, same keyspace as means
    best_params: dict                      # param name -> value of the single best evaluation (physical only)
    param_correlation: np.ndarray          # (dim, dim) weighted-posterior Pearson correlation between
                                            # parameters, in param_names order -- see calibrate_bayesian's
                                            # docstring for what this is and is not telling you. Scoped to
                                            # the physical parameters only, NOT hyper_names below.
    evaluated_params: dict                 # name -> array (n_evaluations,): every point the search
                                            # actually tried, row-for-row with evaluated_scores below --
                                            # the "attempted values" trace, not a resampled/synthetic one.
                                            # Covers both param_names and hyper_names.
    evaluated_scores: np.ndarray           # (n_evaluations,), same order as evaluated_params
    posterior_samples: dict                # name -> array (n_posterior,): the same dense continuous
                                            # sample means/stds/param_correlation were computed from.
                                            # Covers both param_names and hyper_names.
    posterior_weights: np.ndarray          # (n_posterior,), shared by every posterior_samples entry --
                                            # weight each sample by this before histogramming/averaging it
    hyper_names: tuple[str, ...]           # names of any active hyperparameters (see PriorSpec/
                                            # HYPERPRIORS in the module docstring), each "<param>__<field>"
                                            # e.g. "e1__mean" -- empty when no prior used a hyperprior
                                            # range. A subset of the keys in means/stds/evaluated_params/
                                            # posterior_samples, in addition to param_names.
    score_mean: float
    score_std: float
    best_score: float
    n_evaluations: int
    converged_at: int   # evaluation index (0-based) at which the best-so-far score was reached
    history_best: np.ndarray  # best score seen after each evaluation -- for a convergence plot
    time1: np.ndarray
    pulse1: np.ndarray
    time2: np.ndarray
    pulse2: np.ndarray
    predicted_pulse: np.ndarray


def _sample_scale(lo, hi, u, log_scale: bool):
    """Maps a uniform-[0, 1] draw `u` to a value between `lo` and `hi`,
    linearly or log-uniformly -- the one interpolation formula shared by
    every "location" sampling step in this module: a plain PRIOR_UNIFORM's
    own value, and any hyperprior range's resolved hyperparameter (see the
    module docstring's HYPERPRIORS section)."""
    if log_scale:
        return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
    return lo + u * (hi - lo)


@dataclass(frozen=True)
class _Layout:
    """How the *reduced* unit hypercube's columns -- the space the GP, the
    LHS design and the acquisition search all operate in -- map onto free
    parameters and, wherever a hyperprior is in play, their hyperparameters
    too (see the module docstring's HYPERPRIORS section). Built once per
    `calibrate_bayesian` call by `_build_layout`, then consumed by every
    `_assemble` call in that search.

    Without any hyperpriors this reduces exactly to the old one-column-
    per-free-parameter layout (`slot_width[name] == 1` for every free
    parameter, no `hyper_fields`/`hyper_names` at all)."""
    free_names: tuple[str, ...]
    slot_width: dict            # free param name -> columns consumed (1 to 3)
    hyper_fields: dict          # free param name -> tuple of (field_key, lo, hi, log_scale),
                                 # in column order, for however many of its own prior fields
                                 # are a hyperprior range rather than a plain number
    hyper_names: tuple[str, ...]  # flat "<param>__<field>", in column order across free_names
    dim: int                    # sum(slot_width.values()) -- the actual search dimensionality


def _build_layout(param_names: tuple[str, ...], priors: dict) -> _Layout:
    """Decides, once per search, how many unit-hypercube columns each free
    parameter needs (1 plus one more per hyperprior range among its own
    fields) and in what order those columns are consumed -- see `_Layout`
    and the module docstring's HYPERPRIORS section for why and how.

    Hyperprior location fields (`lo`/`hi`/the Gaussian `mean`/the Log-
    Normal median) are explored the same way the parameter's own value
    would be -- log-uniformly for a modulus/viscosity, linearly for `nu`.
    Hyperprior *spread* fields (`std`, or the Log-Normal `factor`) are
    always explored log-uniformly, regardless of that: "how wide should
    the uncertainty be" is itself a positive, naturally multiplicative
    quantity to search over."""
    free_names = [name for name in param_names if priors[name].kind != PRIOR_FIXED]
    slot_width = {}
    hyper_fields = {}
    hyper_names = []
    for name in free_names:
        spec = priors[name]
        loc_log_scale = spec.log_scale if spec.kind == PRIOR_UNIFORM else (name != "nu")
        if spec.kind == PRIOR_UNIFORM:
            candidates = [("lo", spec.lo, loc_log_scale), ("hi", spec.hi, loc_log_scale)]
        elif spec.kind == PRIOR_GAUSSIAN:
            candidates = [("mean", spec.mean, loc_log_scale), ("std", spec.std, True)]
        else:  # PRIOR_LOGNORMAL -- median must stay positive (hence always
               # log-uniform) and factor is inherently multiplicative.
            candidates = [("median", spec.mean, True), ("factor", spec.std, True)]
        fields = []
        for field_key, value, field_log_scale in candidates:
            if isinstance(value, tuple):
                lo, hi = float(value[0]), float(value[1])
                fields.append((field_key, lo, hi, field_log_scale))
                hyper_names.append(f"{name}__{field_key}")
        hyper_fields[name] = tuple(fields)
        slot_width[name] = len(fields) + 1  # +1 for the parameter's own value draw
    dim = sum(slot_width.values())
    return _Layout(free_names=tuple(free_names), slot_width=slot_width,
                    hyper_fields=hyper_fields, hyper_names=tuple(hyper_names), dim=dim)


_EPS = 1e-9


def _assemble(u_free: np.ndarray, param_names: tuple[str, ...], layout: _Layout,
              priors: dict) -> tuple[np.ndarray, np.ndarray]:
    """Expands a (reduced-dimension) unit-hypercube point/batch `u_free`
    into physical parameter values (one column per `param_names`, in that
    order) and any hyperparameter values (one column per
    `layout.hyper_names`, possibly none) -- fixed parameters filled in
    directly from their `PriorSpec.value`; free ones via `layout`'s column
    plan: any hyperprior fields resolved first (each its own column, via
    `_sample_scale`), then the parameter's own value drawn from the
    resulting (possibly just-resolved) distribution -- the inverse-CDF
    trick (`norm.ppf`) for PRIOR_GAUSSIAN/PRIOR_LOGNORMAL is what lets
    Latin Hypercube Sampling's stratification carry over into a stratified
    sample of those distributions rather than only ever of a flat box.
    Handles both a single point (`u_free.shape == (layout.dim,)`) and a
    batch (`u_free.shape == (n, layout.dim)`)."""
    u_free = np.asarray(u_free, dtype=float)
    physical = np.empty(u_free.shape[:-1] + (len(param_names),))
    hyper_cols = {}
    col = 0
    for name in layout.free_names:
        spec = priors[name]
        width = layout.slot_width[name]
        u_slot = u_free[..., col: col + width]
        col += width

        resolved = {}
        for i, (field_key, lo, hi, field_log_scale) in enumerate(layout.hyper_fields[name]):
            value = _sample_scale(lo, hi, u_slot[..., i], field_log_scale)
            resolved[field_key] = value
            hyper_cols[f"{name}__{field_key}"] = value
        u_x = u_slot[..., width - 1]

        if spec.kind == PRIOR_UNIFORM:
            lo_v = resolved.get("lo", spec.lo)
            hi_v = resolved.get("hi", spec.hi)
            # A hyperprior range on lo and/or hi can, for an unlucky draw,
            # resolve to lo_v > hi_v -- swapping keeps every draw a valid
            # interval instead of raising deep inside a background search
            # thread. This never triggers for a plain (non-hyper) lo/hi,
            # which _normalize_priors already validated lo < hi for.
            lo_v, hi_v = np.minimum(lo_v, hi_v), np.maximum(lo_v, hi_v)
            value = _sample_scale(lo_v, hi_v, u_x, spec.log_scale)
        elif spec.kind == PRIOR_GAUSSIAN:
            mean_v = resolved.get("mean", spec.mean)
            std_v = resolved.get("std", spec.std)
            u_clipped = np.clip(u_x, _EPS, 1.0 - _EPS)
            value = norm.ppf(u_clipped, loc=mean_v, scale=std_v)
            floor = np.maximum(np.abs(mean_v) * _GAUSSIAN_FLOOR_FRACTION, _GAUSSIAN_MIN_FLOOR)
            value = np.maximum(value, floor)
        else:  # PRIOR_LOGNORMAL -- exp() of a Normal draw is always > 0,
               # so unlike PRIOR_GAUSSIAN above, this never needs a floor.
            median_v = resolved.get("median", spec.mean)
            factor_v = resolved.get("factor", spec.std)
            u_clipped = np.clip(u_x, _EPS, 1.0 - _EPS)
            value = np.exp(norm.ppf(u_clipped, loc=np.log(median_v), scale=np.log(factor_v)))

        physical[..., param_names.index(name)] = value

    for j, name in enumerate(param_names):
        if name not in layout.free_names:
            physical[..., j] = priors[name].value

    if layout.hyper_names:
        hyper = np.stack([hyper_cols[n] for n in layout.hyper_names], axis=-1)
    else:
        hyper = np.empty(u_free.shape[:-1] + (0,))
    return physical, hyper


def _eval_point_task(context: dict, u: np.ndarray) -> tuple[float, Optional[np.ndarray]]:
    """core.parallel.run_map worker for the initial Latin-Hypercube
    sampling phase of `calibrate_bayesian` (see that function's docstring
    for why only this phase, not the adaptive search loop after it, is
    parallelized). Returns `(score, predicted_pulse)` -- `predicted_pulse`
    is `None` on a failed candidate, matching `eval_point`'s "scored 0.0,
    not excluded" handling, just without the `nonlocal` bookkeeping (which
    can't cross a process boundary) -- the caller re-derives
    best_corr_seen/best_pulse_seen from these results once they're all
    back in the main process."""
    values, _ = _assemble(u, context["param_names"], context["layout"], context["priors"])
    nu = float(values[0])
    params = dict(zip(context["model"].param_keys(), values[1:]))
    try:
        predicted = predict_pulse(
            context["pulse1"], context["time1"], nu, context["model"], params,
            context["density"], context["radius"], context["z1"], context["damp"],
        )
        corr = float(pulse_correlation(
            predicted, context["pulse2"], context["time1"], context["time2"], metric=context["metric"],
        ))
    except Exception:  # noqa: BLE001 - matches eval_point: scored (not excluded) so the GP still learns "avoid this region"
        return 0.0, None
    return corr, predicted


def _expected_improvement(u: np.ndarray, gp: GaussianProcess, best: float, xi: float = 0.01) -> np.ndarray:
    """ACQ_EI: how much better than `best` the surrogate expects a
    candidate to score, in expectation over its own predictive
    uncertainty. `xi` is a small exploration margin (without it, EI can
    get stuck refining right around the current best instead of also
    checking still-uncertain regions). See the module docstring's
    "ACQUISITION STRATEGY" section."""
    u = np.atleast_2d(u)
    mean, std = gp.predict(u)
    std = np.maximum(std, 1e-9)
    improvement = mean - best - xi
    z = improvement / std
    return improvement * norm.cdf(z) + std * norm.pdf(z)


def _probability_of_improvement(u: np.ndarray, gp: GaussianProcess, best: float, xi: float = 0.01) -> np.ndarray:
    """ACQ_PI: the probability of *any* improvement over `best`, ignoring
    how large it might be -- greedier/more exploitative than EI. See the
    module docstring's "ACQUISITION STRATEGY" section."""
    u = np.atleast_2d(u)
    mean, std = gp.predict(u)
    std = np.maximum(std, 1e-9)
    z = (mean - best - xi) / std
    return norm.cdf(z)


def _upper_confidence_bound(u: np.ndarray, gp: GaussianProcess, kappa: float = _UCB_KAPPA) -> np.ndarray:
    """ACQ_UCB: `mean + kappa * std` -- explores purely by how uncertain
    the surrogate still is, without reference to `best` at all. See the
    module docstring's "ACQUISITION STRATEGY" section."""
    u = np.atleast_2d(u)
    mean, std = gp.predict(u)
    return mean + kappa * std


_ACQUISITION_FUNCS = {
    ACQ_EI: _expected_improvement,
    ACQ_PI: _probability_of_improvement,
    ACQ_UCB: _upper_confidence_bound,
}


def _propose_next(gp: GaussianProcess, best: float, dim: int, rng: np.random.Generator,
                   acquisition: str = ACQ_EI,
                   n_random: Optional[int] = None, n_restarts: Optional[int] = None) -> np.ndarray:
    """Proposes the next point to actually run through the expensive
    forward model, by maximizing the chosen acquisition strategy (`ACQ_EI`
    default, or `ACQ_PI`/`ACQ_UCB`/`ACQ_THOMPSON` -- see the module
    docstring's "ACQUISITION STRATEGY" section) over the unit hypercube.

    ACQ_THOMPSON is handled separately (`_propose_next_thompson`): it has
    no smooth closed-form acquisition surface to hand to a gradient
    optimizer, since it comes from one random draw of the surrogate's
    joint posterior evaluated on a discrete candidate set. The other three
    share this function's approach: a broad random scan (cheap -- it only
    calls the GP, never the forward model), then a handful of local
    L-BFGS-B polishes started from its best points. `n_random`/
    `n_restarts` default to values that grow with `dim`: a fixed candidate
    pool that works for the 3-parameter Kelvin-Voigt model under-explores
    the 6-parameter Generalized Maxwell one, where a peak occupies a much
    smaller fraction of the search volume."""
    if acquisition == ACQ_THOMPSON:
        return _propose_next_thompson(gp, dim, rng)

    # One `acq(u)` closure built once, used identically by the random scan
    # and every L-BFGS-B restart below -- ACQ_UCB ignores `best` entirely
    # (see _upper_confidence_bound), everything else needs it.
    if acquisition == ACQ_UCB:
        acq = lambda u: _upper_confidence_bound(u, gp)
    else:
        acq_func = _ACQUISITION_FUNCS[acquisition]
        acq = lambda u: acq_func(u, gp, best)

    if n_random is None:
        n_random = max(2000, 400 * dim)
    if n_restarts is None:
        n_restarts = max(8, dim + 6)
    candidates = rng.uniform(size=(n_random, dim))
    scores = acq(candidates)
    order = np.argsort(scores.ravel())[::-1]

    best_x, best_score = candidates[order[0]], float(scores[order[0]])
    for idx in order[:n_restarts]:
        res = minimize(lambda u: -acq(u)[0], candidates[idx], method="L-BFGS-B", bounds=[(0.0, 1.0)] * dim)
        if -res.fun > best_score:
            best_score, best_x = float(-res.fun), res.x
    return np.clip(best_x, 0.0, 1.0)


def _propose_next_thompson(gp: GaussianProcess, dim: int, rng: np.random.Generator,
                            n_candidates: Optional[int] = None) -> np.ndarray:
    """ACQ_THOMPSON: draws one random continuous function from the
    surrogate's *joint* posterior (`GaussianProcess.sample_y`, which --
    unlike `predict`'s pointwise mean/std -- accounts for the correlation
    between nearby candidate points) and proposes that single sample's own
    best point. `n_candidates` is capped well below `_propose_next`'s
    random-scan size: sampling a joint posterior needs a Cholesky
    factorization of an `n_candidates` x `n_candidates` covariance matrix,
    O(n_candidates^3), so this discretizes the search more coarsely than
    the other strategies' cheap-per-point random scan can afford to."""
    if n_candidates is None:
        n_candidates = int(np.clip(250 * dim, 500, 1500))
    candidates = rng.uniform(size=(n_candidates, dim))
    sample = gp.sample_y(candidates, rng)[:, 0]
    return candidates[int(np.argmax(sample))]


def calibrate_bayesian(
    pulse1: np.ndarray, time1: np.ndarray, pulse2: np.ndarray, time2: np.ndarray,
    radius: float, density: float, z1: float, damp: float,
    model: RheologyModel, param_priors: dict,
    n_init: int = 12, n_iter: int = 40,
    convergence_patience: int = 10, convergence_tol: float = 1e-4,
    seed: int = 0, metric: str = DEFAULT_METRIC, progress_cb: ProgressCB = None,
    n_workers: int = 1, acquisition: str = ACQ_EI,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> ViscoBayesianResult:
    """Bayesian-optimization calibration of `model`'s parameters (plus the
    Poisson ratio `nu`, always present regardless of model).

    `acquisition` picks how the adaptive search loop scores a candidate
    point -- one of `ACQ_EI` (the default), `ACQ_PI`, `ACQ_UCB`,
    `ACQ_THOMPSON` (see the module docstring's "ACQUISITION STRATEGY"
    section, and `ACQUISITION_LABELS`/`ACQUISITION_USE_CASES` for their
    short, UI-facing descriptions). Only affects the adaptive loop, not
    the initial Latin Hypercube design or the final posterior estimate.

    `param_priors` is a dict of name -> `PriorSpec` (see the module
    docstring's "PRIORS" and "HYPERPRIORS" sections), with one entry per
    name in `("nu",) + model.param_keys()`. A bare `(min, max)` tuple/list
    is also accepted per entry, upgraded to `PriorSpec.uniform(min, max)`
    -- the simplest and default choice. A
    `"fixed"` entry removes that parameter from the search entirely
    (dimension reduced by one per fixed parameter); a `"gaussian"` entry
    (mean, std) or a `"lognormal"` one (median, factor) both concentrate
    where the search looks and reshape the reported posterior toward that
    prior wherever the two pulses alone cannot pin the parameter down
    (again, see the module docstring). Any of a `PriorSpec`'s own fields
    (`lo`/`hi`/`mean`/`std`) can itself be a `(lo, hi)` range instead of a
    plain number, turning it into a hyperprior the search also locates
    (module docstring, "HYPERPRIORS") -- at the cost of one extra search
    dimension per such field.

    `metric` selects the goodness-of-fit score every candidate is judged
    by (see core/visco_calibration.py's SCORE_FUNCS/SCORE_LABELS) -- it is
    both what Expected Improvement chases during the search and what the
    convergence check and posterior weighting below are computed from.
    Stops early once the best score found hasn't improved by more than
    `convergence_tol` for `convergence_patience` consecutive iterations
    (this is what makes it typically much faster than the grid search: it
    stops as soon as it has actually converged, rather than always paying
    for a fixed, worst-case evaluation budget).


    `n_workers > 1` evaluates the `n_init` initial Latin-Hypercube points
    concurrently (see `_eval_point_task`) -- they have no interdependency,
    unlike every point after them, each of which is proposed from a
    Gaussian Process fit on every point evaluated so far (including the
    previous proposal). That adaptive loop is NOT parallelized: doing so
    would mean proposing several points from the same, not-yet-updated
    GP, changing which points the search actually visits -- not merely
    how fast it runs. Left at its default of 1, the search runs purely
    sequentially.

    `cancel_check`, when given, raises `core.parallel.OperationCancelled`
    (see that exception's own docstring) the first time it returns truthy
    -- checked once per initial Latin-Hypercube point (`n_workers == 1`
    only; the `n_workers > 1` parallel LHS batch below cannot be
    interrupted mid-batch, only checked again once it completes) and once
    per adaptive-search iteration after that. Cooperative: a candidate
    point's own forward-model evaluation already in progress always
    finishes first."""
    if acquisition not in ACQUISITION_LABELS:
        raise ValueError(f"Unknown acquisition strategy {acquisition!r}, expected one of {list(ACQUISITION_LABELS)}")

    param_names = ("nu",) + model.param_keys()
    priors = _normalize_priors(param_priors, param_names)
    layout = _build_layout(param_names, priors)
    free_names = list(layout.free_names)
    dim = layout.dim
    rng = np.random.default_rng(seed)
    total = n_init + n_iter

    if dim == 0:
        # Every parameter fixed: nothing left to search (a fixed parameter
        # can never carry a hyperprior -- see PriorSpec.fixed -- so this
        # only happens when every entry is PRIOR_FIXED). Evaluate the one
        # remaining point directly (like core.visco_calibration.
        # test_parameters -- including letting a genuine failure here
        # propagate rather than silently reporting a meaningless score for
        # the single point the user asked for) and report a degenerate,
        # zero-uncertainty result instead of handing a zero-dimensional
        # problem to the LHS/GP/EI machinery below, none of which is
        # meaningful with nothing to search over.
        fixed_values, _ = _assemble(np.zeros(0), param_names, layout, priors)
        nu = float(fixed_values[0])
        params = dict(zip(model.param_keys(), fixed_values[1:]))
        predicted = predict_pulse(pulse1, time1, nu, model, params, density, radius, z1, damp)
        corr = float(pulse_correlation(predicted, pulse2, time1, time2, metric=metric))
        means = dict(zip(param_names, fixed_values))
        if progress_cb:
            progress_cb(1, 1)
        return ViscoBayesianResult(
            model_key=model.key, param_names=param_names, metric=metric,
            means=means, stds={name: 0.0 for name in param_names}, best_params=dict(means),
            param_correlation=np.eye(len(param_names)),
            evaluated_params={name: np.array([means[name]]) for name in param_names},
            evaluated_scores=np.array([corr]),
            posterior_samples={name: np.array([means[name]]) for name in param_names},
            posterior_weights=np.array([1.0]),
            hyper_names=(),
            score_mean=corr, score_std=0.0, best_score=corr,
            n_evaluations=1, converged_at=0, history_best=np.array([corr]),
            time1=time1, pulse1=pulse1, time2=time2, pulse2=pulse2, predicted_pulse=predicted,
        )

    # Tracks the predicted pulse of the best point actually evaluated so
    # far, so the final result can reuse it instead of paying for one more
    # root-search later (see the comment above `predicted_pulse` below for
    # why that extra call was a problem, not just a redundant one).
    best_pulse_seen = None
    best_corr_seen = -np.inf

    def eval_point(u: np.ndarray) -> float:
        nonlocal best_pulse_seen, best_corr_seen
        values, _ = _assemble(u, param_names, layout, priors)
        nu = float(values[0])
        params = dict(zip(model.param_keys(), values[1:]))
        try:
            predicted = predict_pulse(pulse1, time1, nu, model, params, density, radius, z1, damp)
            corr = float(pulse_correlation(predicted, pulse2, time1, time2, metric=metric))
        except Exception:  # noqa: BLE001 - matches the old calibrate()'s handling of a failed candidate, but
            return 0.0      # scored (not excluded) so the GP still learns "avoid this region"
        if corr > best_corr_seen:
            best_corr_seen = corr
            best_pulse_seen = predicted
        return corr

    U = latin_hypercube(n_init, dim, rng)
    y = np.empty(n_init)

    if n_workers > 1 and n_init > 1:
        lhs_context = dict(
            param_names=param_names, layout=layout, priors=priors, model=model,
            pulse1=pulse1, time1=time1, pulse2=pulse2, time2=time2,
            density=density, radius=radius, z1=z1, damp=damp, metric=metric,
        )

        def _report_lhs_progress(done, _n_init):
            # `total` here is the *whole* search budget (n_init + n_iter),
            # matching every other progress_cb call in this function --
            # run_map's own `total` (n_init) is intentionally ignored.
            if progress_cb:
                progress_cb(done, total)

        raw = run_map(
            _eval_point_task, [U[i] for i in range(n_init)], n_workers=n_workers,
            context=lhs_context, progress_callback=_report_lhs_progress,
        )
        check_cancelled(cancel_check)
        for i, value, err in raw:
            if err is not None:
                y[i] = 0.0
                continue
            corr, predicted = value
            y[i] = corr
            if corr > best_corr_seen:
                best_corr_seen = corr
                best_pulse_seen = predicted
    else:
        for i in range(n_init):
            check_cancelled(cancel_check)
            y[i] = eval_point(U[i])
            if progress_cb:
                progress_cb(i + 1, total)

    history_best = [float(np.max(y[: i + 1])) for i in range(n_init)]
    no_improve = 0
    n_done = n_init

    for _ in range(n_iter):
        check_cancelled(cancel_check)
        gp = GaussianProcess(n_restarts=4, random_state=int(rng.integers(1 << 30)))
        gp.fit(U, y)
        best_so_far = float(np.max(y))
        u_next = _propose_next(gp, best_so_far, dim, rng, acquisition=acquisition)

        y_next = eval_point(u_next)
        U = np.vstack([U, u_next])
        y = np.append(y, y_next)
        n_done += 1
        if progress_cb:
            progress_cb(n_done, total)

        new_best = float(np.max(y))
        history_best.append(new_best)
        no_improve = 0 if (new_best - best_so_far > convergence_tol) else no_improve + 1
        if no_improve >= convergence_patience:
            if progress_cb:
                progress_cb(total, total)
            break

    best_idx = int(np.argmax(y))
    best_values, _ = _assemble(U[best_idx], param_names, layout, priors)
    best_params = dict(zip(param_names, best_values))
    best_corr = float(y[best_idx])
    converged_at = int(np.argmax(history_best))

    # Every point the search actually evaluated (LHS init + BO iterations),
    # in real parameter units -- the "attempted values" trace: what the
    # search actually looked at and how each one scored, as opposed to the
    # posterior sample below (which is a dense, cheap, synthetic resampling
    # of the fitted surrogate, not points the expensive forward model was
    # ever run on).
    evaluated_values, evaluated_hyper = _assemble(U, param_names, layout, priors)
    evaluated_params = {name: evaluated_values[:, i] for i, name in enumerate(param_names)}
    if layout.hyper_names:
        evaluated_params.update({name: evaluated_hyper[:, i] for i, name in enumerate(layout.hyper_names)})
    evaluated_scores = y.copy()

    # Final surrogate posterior: fit one last GP on every evaluated point,
    # then estimate p(theta | data) by densely sampling each free
    # parameter's *prior* (uniform across its box, or its own Gaussian --
    # see _sample_free) -- cheap, since it never touches the forward model
    # -- and weighting each sample by a softmax of the surrogate's
    # predicted correlation. That is a self-normalized-importance-sampling
    # estimate of posterior ~ prior * likelihood (the module docstring's
    # "PRIORS" section spells out why this is what actually narrows the
    # reported uncertainty for a Gaussian prior, instead of only shifting
    # where in a flat box the search happens to concentrate).
    gp = GaussianProcess(n_restarts=6, random_state=seed)
    gp.fit(U, y)
    n_posterior = 20000
    U_post = rng.uniform(size=(n_posterior, dim))
    mean_post, _ = gp.predict(U_post)
    temperature = max(float(np.std(mean_post)), 1e-6) * 0.3
    weights = np.exp((mean_post - mean_post.max()) / temperature)
    weights /= weights.sum()

    X_post, Hyper_post = _assemble(U_post, param_names, layout, priors)
    means_arr = np.average(X_post, axis=0, weights=weights)
    stds_arr = np.sqrt(np.average((X_post - means_arr) ** 2, axis=0, weights=weights))
    means = dict(zip(param_names, means_arr))
    stds = dict(zip(param_names, stds_arr))

    # Hyperparameters (see the module docstring's HYPERPRIORS section) get
    # their own weighted mean/std from this same posterior sample, merged
    # into the same means/stds dicts -- but kept out of param_correlation
    # below, which reuses X_post/means_arr/stds_arr (physical only) and is
    # untouched by any of this.
    if layout.hyper_names:
        hyper_means_arr = np.average(Hyper_post, axis=0, weights=weights)
        hyper_stds_arr = np.sqrt(np.average((Hyper_post - hyper_means_arr) ** 2, axis=0, weights=weights))
        means.update(dict(zip(layout.hyper_names, hyper_means_arr)))
        stds.update(dict(zip(layout.hyper_names, hyper_stds_arr)))

    # Parameter interdependence: the weighted Pearson correlation between
    # every pair of parameters across this same posterior sample -- reuses
    # X_post/weights/means_arr/stds_arr above, so it costs nothing beyond
    # the covariance matrix itself. See the module docstring for what a
    # value near +-1 here does (and does not) mean.
    diffs = X_post - means_arr
    weighted_cov = diffs.T @ (diffs * weights[:, None])
    std_outer = np.outer(stds_arr, stds_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        param_correlation = np.where(std_outer > 1e-300, weighted_cov / std_outer, 0.0)
    np.fill_diagonal(param_correlation, 1.0)
    param_correlation = np.clip(param_correlation, -1.0, 1.0)

    posterior_samples = {name: X_post[:, i] for i, name in enumerate(param_names)}
    if layout.hyper_names:
        posterior_samples.update({name: Hyper_post[:, i] for i, name in enumerate(layout.hyper_names)})
    posterior_weights = weights

    score_mean = float(np.average(mean_post, weights=weights))
    score_std = float(np.sqrt(np.average((mean_post - score_mean) ** 2, weights=weights)))

    # `predicted_pulse` is what the Result plot overlays on the two measured
    # pulses. It is reused from the pulse already computed for the best
    # point the search actually evaluated (tracked in `best_pulse_seen`
    # inside `eval_point`) rather than recomputed at the posterior-mean
    # parameters: that mean is a weighted *average* over the search box
    # (averaged in linear space even for the log-scaled moduli/
    # viscosities), not a point the search itself ever evaluated, so for a
    # weakly-identified, high-dimensional model (Generalized Maxwell's 5
    # moduli/viscosities plus nu) it can land somewhere the per-frequency
    # Pochhammer-Chree root search (core.visco_calibration._predict_pulse's
    # sympy `nsolve` loop) converges far more slowly than at any point the
    # optimizer actually visited. Such a call also would not be wired to
    # `progress_cb`, so it would run after the progress bar has already
    # reached 100% (set explicitly at the convergence break above), across
    # every frequency bin, with no feedback -- most noticeably for
    # Generalized Maxwell/Burgers where the root search is slowest.
    #
    # Reusing `best_pulse_seen` instead is free (already computed), is
    # guaranteed to have succeeded once already, and needs nothing beyond
    # the cheap GP/array math once the loop above reports 100%.
    predicted_pulse = best_pulse_seen
    if predicted_pulse is None:  # only if every single evaluation failed
        predicted_pulse = predict_pulse(
            pulse1, time1, best_params["nu"], model,
            {k: best_params[k] for k in model.param_keys()}, density, radius, z1, damp,
        )

    return ViscoBayesianResult(
        model_key=model.key, param_names=param_names, metric=metric,
        means=means, stds=stds, best_params=best_params, param_correlation=param_correlation,
        evaluated_params=evaluated_params, evaluated_scores=evaluated_scores,
        posterior_samples=posterior_samples, posterior_weights=posterior_weights,
        hyper_names=layout.hyper_names,
        score_mean=score_mean, score_std=score_std, best_score=best_corr,
        n_evaluations=n_done, converged_at=converged_at,
        history_best=np.array(history_best),
        time1=time1, pulse1=pulse1, time2=time2, pulse2=pulse2, predicted_pulse=predicted_pulse,
    )
