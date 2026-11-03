"""
Bayesian Calibration's constitutive-equation parser and evaluator, ported
from MAMBA (def_model.py's `recup_expr`/`cal_MCMC`) with a deliberately
different implementation strategy.

MAMBA parsed the user's free-text equation TWICE, independently: once via
fragile ad-hoc string-splitting (`recup_expr`, stripping a hardcoded list
of operator/function substrings to auto-detect free-parameter names --
dead code, since the resulting `sympy` expression was built and then
thrown away) and once via literal `str.replace()` substitution of every
detected name with an indexed array-slice string, fed to Python's builtin
`eval()` (`cal_MCMC`) -- fragile (a parameter name that happens to be a
substring of another token could silently corrupt the substitution;
scientific notation like `1e-3` would have the `e` mistaken for the
strain variable) and version-fragile (relied on old Theano tensor/numpy
ufunc interoperability that doesn't carry over to modern PyTensor).

This module instead parses the equation exactly ONCE, into a whitelisted
Python `ast` tree (rejecting anything that isn't a number, a recognized
variable/function name, or one of `+ - * / ^` -- this is free-typed user
input, so this whitelist is a real safety boundary, not just a UX nicety),
and evaluates that SAME validated tree twice at different times: once
against plain numpy arrays (for a numeric forward-evaluation/preview,
`NUMPY_BACKEND` below) and once against PyMC/PyTensor random-variable
tensors (inside the calibration model itself, `core/calib_pymc_model.py`)
-- both go through the identical `evaluate()` function below, dispatched
through a `backend` dict of callables, since PyTensor tensor variables
overload the same Python arithmetic operators (`+ - * / **`) numpy arrays
do. One evaluator, two backends, no duplicated logic to keep in sync.

Reserved variables: `e` (strain), `v` (strain rate), `T` (temperature),
plus up to 3 optional extra variables `a`/`b`/`d` -- exactly MAMBA's set.
Recognized functions (also exactly MAMBA's set, `lll` in the original):
sqrt cos sin tan cosh sinh tanh acos asin atan erf exp log log10 asinh
acosh atanh, plus the constant `pi`. The equation's `^` is accepted as
MAMBA's own equation syntax did (matching the user manual, which shows
"^" and separately documents "log" as meaning natural log/ln, not
log-base-10) and is translated to Python's `**` before parsing -- `^` is
otherwise unused in this grammar, so this substitution is unambiguous.

Free parameters (the things Bayesian Calibration will actually fit,
`core/calib_priors.py`) are auto-detected as whatever `Name` nodes are
left over after removing the reserved variables and `pi`: this is a
strict, single source of truth, unlike MAMBA's independent, sometimes-
inconsistent detection passes.

Domain guards (added after a real user's calibration crashed on real
data): several of the whitelisted functions are only defined on part of
the real line -- sqrt/log/log10 need a nonnegative (log/log10: strictly
positive) argument, acos/asin need an argument in [-1, 1], atanh needs
one in (-1, 1), acosh needs one >= 1 -- and the power operator `**`
(written `^` in this grammar) is only defined for a NEGATIVE base when
the exponent happens to be an integer; a negative base with a non-
integer exponent (e.g. a fitted hardening exponent `n` in `v**n`) has no
real result. A real user's strain-rate channel had 1-2 slightly-negative
points (ordinary measurement noise near zero) feeding straight into an
equation like `v**m`/`log(v)`, which turned into NaN/complex partway
through evaluation and broke the calibration (a numpy RuntimeWarning
silently producing NaN, or PyMC's initial-point check refusing to start
sampling at all -- either way, unusable). Rather than let one or two
noisy points anywhere near zero take down an entire calibration run,
every domain-restricted function/operation below is evaluated against
its argument CLAMPED into its valid domain (with a tiny `DOMAIN_EPS`
margin away from an exact boundary, since several of these functions'
derivatives blow up right at the edge of their domain -- a well-known
gradient-based-sampler gotcha) instead of the raw value. This changes
nothing for any point that was already in-domain (the clamp is a no-op
there) and only ever affects the handful of points that would otherwise
have been mathematically undefined -- it does not, and cannot, turn bad
data into a physically meaningful fit, it only keeps the calculation
running instead of crashing on it. `evaluate_with_domain_report()` below
lets a caller find out, after the fact, exactly how many real data
points needed this clamp (used by core/calib_run.py to surface a
Result-summary warning, the same transparency-first pattern as this
module's own covariance-jitter/condition-number diagnostic in
core/calib_library_adapter.py).

A literal INTEGER exponent (`e**2`, `e**-1`, `T**2.0`, ...) is detected
at parse time (`_is_integer_exponent` below) and deliberately exempted
from the base-clamp -- integer powers of a negative real number are
always exactly defined (e.g. `(-3.0)**2 == 9.0`), so clamping there
would silently corrupt an otherwise-correct, physically legitimate
negative input (e.g. a Celsius temperature below zero squared). Only a
variable/parameter exponent or a non-integer literal exponent (both of
which cannot be proven safe for every possible negative base) gets the
domain-guarded power.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass

import numpy as np
import scipy.special


class EquationError(ValueError):
    """Raised for anything wrong with a user-typed equation: invalid
    syntax, a disallowed construct (this module's whitelist), an unknown
    function name, or a missing variable/parameter value at evaluation
    time."""


# Order matters for ParsedEquation.used_vars's display order (e, v, T,
# a, b, d), not for correctness.
RESERVED_VARS = ("e", "v", "T", "a", "b", "d")
_RESERVED_VARS_SET = frozenset(RESERVED_VARS)

# MAMBA's exact recognized-function list (def_model.py's `lll`), minus
# 'pi' which is a constant, not a callable, and is handled separately.
KNOWN_FUNCS = (
    "sqrt", "cos", "sin", "tan", "cosh", "sinh", "tanh",
    "acos", "asin", "atan", "erf", "exp", "log", "log10",
    "asinh", "acosh", "atanh",
)
_KNOWN_FUNCS_SET = frozenset(KNOWN_FUNCS)

_PI_NAME = "pi"

# The only binary/unary operators this grammar allows -- deliberately
# narrower than full Python (no %, //, @, bitwise/boolean/comparison
# operators), since the input is free-typed by the user and this is a
# real safety whitelist, not just a style choice.
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARYOPS = (ast.USub, ast.UAdd)

# Tiny margin used by every domain guard below: a lower/upper clamp lands
# DOMAIN_EPS *inside* the function's true valid domain, never exactly on
# its boundary, because several of these functions (acos, asin, atanh,
# acosh, sqrt, log, log10) have an infinite derivative right at the edge
# of their domain -- landing exactly on it would trade a crash now for a
# gradient-explosion crash later, during PyMC's autodiff. Small enough
# (1e-9) to be completely negligible next to any real physical quantity
# this app evaluates (strain, strain rate, temperature, stress, ...).
DOMAIN_EPS = 1e-9


def _safe_sqrt(x):
    return np.sqrt(np.maximum(x, DOMAIN_EPS))


def _safe_log(x):
    return np.log(np.maximum(x, DOMAIN_EPS))


def _safe_log10(x):
    return np.log10(np.maximum(x, DOMAIN_EPS))


def _safe_acos(x):
    return np.arccos(np.clip(x, -1.0 + DOMAIN_EPS, 1.0 - DOMAIN_EPS))


def _safe_asin(x):
    return np.arcsin(np.clip(x, -1.0 + DOMAIN_EPS, 1.0 - DOMAIN_EPS))


def _safe_acosh(x):
    return np.arccosh(np.maximum(x, 1.0 + DOMAIN_EPS))


def _safe_atanh(x):
    return np.arctanh(np.clip(x, -1.0 + DOMAIN_EPS, 1.0 - DOMAIN_EPS))


def _safe_pow(base, exponent):
    """The domain-guarded power used for any `**` whose exponent is NOT
    a literal integer (see `_is_integer_exponent` below) -- clamps only
    the base's lower side, matching sqrt/log's own guard, since a
    negative base with a non-integer exponent is exactly as undefined as
    a negative argument to sqrt/log (this is literally `exp(exponent *
    log(base))` under the hood)."""
    return np.maximum(base, DOMAIN_EPS) ** exponent


# {function name -> predicate(array) -> boolean array of IN-DOMAIN
# elements}, used only by `evaluate_with_domain_report()` below to COUNT
# how many real elements actually needed clamping. Never consulted by
# `evaluate()`'s own hot path (the `_safe_*` functions above are already
# unconditional clamps, branch-free) -- this dict exists purely for the
# after-the-fact diagnostic.
_DOMAIN_CHECKS = {
    "sqrt": lambda x: x >= 0,
    "log": lambda x: x > 0,
    "log10": lambda x: x > 0,
    "acos": lambda x: (x >= -1) & (x <= 1),
    "asin": lambda x: (x >= -1) & (x <= 1),
    "acosh": lambda x: x >= 1,
    "atanh": lambda x: (x > -1) & (x < 1),
}

# Real numpy (or scipy) ufunc per recognized function, for the forward/
# preview numeric-evaluation path. numpy has no erf of its own --
# scipy.special.erf is the standard substitute and is elementwise/array-
# safe like a ufunc. core/calib_pymc_model.py builds the PyTensor
# equivalent of this dict lazily (only once pymc/pytensor are actually
# available), the same shape (including the domain guards on the same
# functions, and the same "__safe_pow__" key), so `evaluate()` below
# never needs to know which backend it's running against. Every domain-
# restricted function (sqrt/log/log10/acos/asin/acosh/atanh) uses its
# `_safe_*` guarded version -- see this module's "Domain guards"
# docstring section above; the remaining functions (cos/sin/tan/cosh/
# sinh/tanh/atan/erf/exp/asinh) are defined on the entire real line, so
# they're left exactly as MAMBA had them.
NUMPY_BACKEND = {
    "sqrt": _safe_sqrt, "cos": np.cos, "sin": np.sin, "tan": np.tan,
    "cosh": np.cosh, "sinh": np.sinh, "tanh": np.tanh,
    "acos": _safe_acos, "asin": _safe_asin, "atan": np.arctan,
    "erf": scipy.special.erf, "exp": np.exp, "log": _safe_log, "log10": _safe_log10,
    "asinh": np.arcsinh, "acosh": _safe_acosh, "atanh": _safe_atanh,
    # Not a KNOWN_FUNCS entry and never reachable via a user-typed call
    # (_validate_and_collect rejects any Call whose name isn't in
    # KNOWN_FUNCS) -- an internal hook `_eval_node`'s Pow handling below
    # looks up by this exact key, for the base-clamp described above.
    "__safe_pow__": _safe_pow,
}


@dataclass(frozen=True)
class ParsedEquation:
    """The validated result of `parse_equation()`. `tree` is only ever
    fed back into `evaluate()` below -- never re-validated, never handed
    to a general-purpose `eval()`/`exec()` -- since it was already walked
    and whitelisted once here."""
    source: str            # the original, as-typed equation text (with '^', for display)
    tree: ast.Expression   # parsed from `source` with '^' translated to '**'
    free_params: tuple     # sorted free-parameter names, e.g. ("A", "C", "n")
    used_vars: tuple       # reserved variable names actually used, in (e, v, T, a, b, d) order


def _validate_and_collect(node: ast.AST, free_params: set, used_vars: set) -> None:
    """Walks `node`, raising EquationError on anything outside this
    module's whitelist, and collects every reserved-variable/free-
    parameter Name encountered along the way. Recurses explicitly
    (rather than a generic `ast.walk`) specifically so a Call's function
    name (`sin` in `sin(e)`) is checked against `KNOWN_FUNCS` and never
    collected as a variable/parameter, while every other Name is."""
    if isinstance(node, ast.Expression):
        _validate_and_collect(node.body, free_params, used_vars)

    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise EquationError(f"Only numeric literals are allowed (found {node.value!r}).")

    elif isinstance(node, ast.Name):
        name = node.id
        if name == _PI_NAME:
            return
        if name in _RESERVED_VARS_SET:
            used_vars.add(name)
        elif name in _KNOWN_FUNCS_SET:
            raise EquationError(
                f"'{name}' is a recognized function name and must be called as {name}(...), "
                "not used as a bare variable."
            )
        else:
            free_params.add(name)

    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise EquationError(
                f"The operator '{type(node.op).__name__}' is not allowed -- only + - * / ^ (power)."
            )
        _validate_and_collect(node.left, free_params, used_vars)
        _validate_and_collect(node.right, free_params, used_vars)

    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise EquationError(f"The unary operator '{type(node.op).__name__}' is not allowed.")
        _validate_and_collect(node.operand, free_params, used_vars)

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise EquationError(
                "Only calls to a recognized function name are allowed (e.g. sin(e)), "
                "not method calls or any other callable."
            )
        func_name = node.func.id
        if func_name not in _KNOWN_FUNCS_SET:
            raise EquationError(
                f"Unknown function '{func_name}'. Recognized functions: {', '.join(KNOWN_FUNCS)}."
            )
        if node.keywords:
            raise EquationError(f"'{func_name}(...)' does not take keyword arguments.")
        if len(node.args) != 1:
            raise EquationError(f"'{func_name}(...)' takes exactly one argument (got {len(node.args)}).")
        _validate_and_collect(node.args[0], free_params, used_vars)

    else:
        raise EquationError(
            f"Unsupported syntax ({type(node).__name__}). Only numbers, the variables "
            f"{', '.join(RESERVED_VARS)}, the constant pi, +, -, *, /, ^ (power), parentheses, "
            f"and the recognized functions are allowed."
        )


def parse_equation(expr: str) -> ParsedEquation:
    """Parses and validates a user-typed equation string, raising
    EquationError with a human-readable message on anything invalid.
    Free parameters and used reserved variables are auto-detected in the
    same pass (see `_validate_and_collect`), the single source of truth
    both the numeric-preview path and the PyMC model builder rely on."""
    if not expr or not expr.strip():
        raise EquationError("The equation is empty.")

    python_expr = expr.replace("^", "**")
    try:
        tree = ast.parse(python_expr, mode="eval")
    except SyntaxError as exc:
        raise EquationError(f"Could not parse the equation: {exc.msg}") from exc

    free_params: set = set()
    used_vars: set = set()
    _validate_and_collect(tree, free_params, used_vars)

    if not free_params and not used_vars:
        raise EquationError("The equation must use at least one variable or free parameter.")

    return ParsedEquation(
        source=expr,
        tree=tree,
        free_params=tuple(sorted(free_params)),
        used_vars=tuple(v for v in RESERVED_VARS if v in used_vars),
    )


def _is_integer_exponent(node: ast.AST) -> bool:
    """True if `node` (the right-hand side of a `Pow` BinOp, already
    whitelisted by `_validate_and_collect`) is a literal, compile-time
    integer -- `2`, `-1`, `+3`, `2.0` all count -- in which case `left **
    right` is exactly defined for ANY real `left`, including negative,
    so no domain guard is needed (or wanted -- see this module's "Domain
    guards" docstring section on why clamping here would be wrong).
    Recurses through a leading unary +/- so `e**-1` (parsed as
    `UnaryOp(USub, Constant(1))`, never folded into `Constant(-1)` by
    `ast.parse`) is recognized too. Anything else -- a bare variable or
    free-parameter Name, a non-integer literal like `0.5`, or any richer
    sub-expression -- returns False, since it cannot be proven to land
    on an integer for every future evaluation (e.g. every MCMC draw of a
    fitted exponent)."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
        return _is_integer_exponent(node.operand)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return float(node.value).is_integer()
    return False


def _eval_node(node: ast.AST, namespace: dict, backend: dict):
    """Evaluates an already-validated node (from a `ParsedEquation.tree`
    built by `parse_equation` above). Mirrors `_validate_and_collect`'s
    node-type coverage exactly -- every branch here has a corresponding
    accepted case there, so nothing reaches `_eval_node` that wasn't
    already whitelisted; the trailing EquationErrors below are
    unreachable in practice and exist only as a defensive backstop."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return namespace[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, namespace, backend)
        right = _eval_node(node.right, namespace, backend)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            # A literal integer exponent is always safely defined, even
            # for a negative base -- see _is_integer_exponent's own
            # docstring and this module's "Domain guards" section.
            # Anything else routes through the backend's own guarded
            # power (falling back to plain '**' if a backend genuinely
            # doesn't provide one, e.g. a minimal test stand-in).
            if _is_integer_exponent(node.right):
                return left ** right
            safe_pow = backend.get("__safe_pow__")
            return safe_pow(left, right) if safe_pow is not None else left ** right
        raise EquationError(f"Operator '{type(node.op).__name__}' is not allowed.")
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand, namespace, backend)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise EquationError(f"Unary operator '{type(node.op).__name__}' is not allowed.")
    if isinstance(node, ast.Call):
        func_name = node.func.id
        if func_name not in backend:
            raise EquationError(
                f"No implementation of '{func_name}' was provided for this evaluation backend."
            )
        arg = _eval_node(node.args[0], namespace, backend)
        return backend[func_name](arg)
    raise EquationError(f"Unsupported syntax during evaluation ({type(node).__name__}).")


def evaluate(parsed: ParsedEquation, namespace: dict, backend: dict):
    """Evaluates `parsed` (from `parse_equation`) against `namespace`
    (mapping every name in `parsed.used_vars + parsed.free_params` to a
    value -- a plain float, a numpy array, or a PyMC/PyTensor tensor
    variable, all of which support the same `+ - * / **` operators) and
    `backend` (mapping every recognized function actually called in the
    equation to a callable for that value type, e.g. `NUMPY_BACKEND`
    above, or a PyTensor equivalent built by `core/calib_pymc_model.py`).

    `pi` is auto-injected into the namespace if the equation uses it and
    the caller didn't already provide one -- it's a constant, not
    something the caller should need to supply per evaluation.

    Raises EquationError if `namespace` is missing a required name."""
    required = (*parsed.used_vars, *parsed.free_params)
    missing = [name for name in required if name not in namespace]
    if missing:
        raise EquationError(f"Missing value(s) for: {', '.join(missing)}.")

    ns = dict(namespace)
    ns.setdefault(_PI_NAME, math.pi)
    return _eval_node(parsed.tree.body, ns, backend)


def evaluate_with_domain_report(parsed: ParsedEquation, namespace: dict) -> tuple:
    """Like `evaluate(parsed, namespace, NUMPY_BACKEND)`, but also
    returns a report of how many elements of `namespace`'s real numpy
    arrays actually needed a domain-guard clamp: `(value, report)` where
    `report` is `{function name: count}` (plus `"pow"` for a clamped
    power base), including only entries with a nonzero count -- an empty
    dict means every element was already in-domain everywhere.

    Numpy-backend only (there is no PyTensor-graph equivalent of "count
    how many elements were out of domain" without adding extra ops to
    the sampled model itself, which this deliberately does not do) --
    meant for a one-off, pre-sampling diagnostic pass over each
    calibration test's own REAL measured data plus a representative
    guess for every free parameter (see core/calib_run.py's
    `_domain_guard_report_for_tests`), not for anything performance-
    sensitive or run per MCMC draw."""
    report: dict = {}

    def _tracked(name, fn):
        check = _DOMAIN_CHECKS.get(name)

        def _wrapped(x):
            if check is not None:
                bad = int(np.count_nonzero(~check(np.asarray(x, dtype=float))))
                if bad:
                    report[name] = report.get(name, 0) + bad
            return fn(x)

        return _wrapped

    def _tracked_pow(base, exponent):
        bad = int(np.count_nonzero(np.asarray(base, dtype=float) < 0))
        if bad:
            report["pow"] = report.get("pow", 0) + bad
        return NUMPY_BACKEND["__safe_pow__"](base, exponent)

    instrumented = {
        name: (_tracked_pow if name == "__safe_pow__" else _tracked(name, fn))
        for name, fn in NUMPY_BACKEND.items()
    }

    value = evaluate(parsed, namespace, instrumented)
    return value, report
