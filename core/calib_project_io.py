"""
Save/load a whole Bayesian Calibration page setup (equation, free-parameter
priors, the "Calibration tests" table's assignments, and sampler settings)
as a JSON+CSV file pair -- so a calibration project can be backed up,
shared, or resumed later without re-typing the equation or re-picking
every library entry by hand.

Follows the exact file-pairing convention already established by Results &
Export's "Export full test" / Test Setup's "Load Test" (`ui/pages/
page_results_export.py::_on_export_full_test`, `ui/pages/page_test_setup.py
::_on_load_test`): Save asks for the .csv path and derives the .json path
from it (`path.rsplit(".", 1)[0] + ".json"`); Load asks for the .json path
and looks for the matching .csv next to it.

The two files play different roles, not interchangeable halves of one
format: the CSV is the "Calibration tests" table itself (genuinely
tabular -- one row per test, one column per equation-variable role plus
Output/Validation), while the JSON carries the equation text, each free
parameter's full prior (every field, not just the ones its own `kind`
uses -- matching core.calib_priors.ParamPrior's own "every field always
present" convention), the per-column "Variable" quantity picker
selections, the covariance mode, and the sampler settings.

Deliberately does NOT embed the covariance-library entries a test's
columns point to -- those are saved/loaded separately, as their own JSON+
CSV pair, by core/covariance_library_io.py (from the Covariance Matrix
page). A calibration-project file only ever stores library entry NAMES; a
name that doesn't (yet, or any more) exist in `project.covariance_library`
when loading back is left unresolved -- the caller (the page) is expected
to report exactly which ones, rather than this module guessing or
silently dropping them.

Pure stdlib (json/csv) plus core.calib_priors -- no PySide6, no pymc/
pytensor -- so it is fully real-testable in this sandbox, same rationale
as core/calib_equation.py and core/calib_library_adapter.py.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field

from core.calib_priors import PRIOR_KINDS, ParamPrior, PriorError

FORMAT_VERSION = 1

_NON_ROLE_CSV_COLUMNS = ("test_label", "output_entry", "is_validation")


class CalibProjectIOError(ValueError):
    """Raised for anything wrong with a calibration-project file pair:
    missing/malformed JSON keys, an unknown prior kind, a missing or
    malformed CSV, or a missing companion file -- a single error type so
    the page's status label doesn't need to know which piece failed."""


@dataclass
class CalibProjectData:
    """`import_calibration_project`'s return value -- everything needed
    to repopulate the Bayesian Calibration page's cards 1-4, short of
    resolving library-entry names against the CURRENT project's own
    `covariance_library` (the page's own job, since only it knows which
    entries currently exist and how to report an unresolved one)."""
    equation: str
    priors: dict                    # {free_param_name: ParamPrior}
    role_quantities: dict           # {reserved-var role: quantity_key str, or None for "(any quantity)"}
    output_quantity: str | None
    covariance_mode: str
    sampler_settings: dict          # {"draws", "tune", "chains", "cores", "target_accept", "init"}
    tests: list = field(default_factory=list)  # [{"label", "variable_entries": {role: entry_name str}, "output_entry": entry_name str, "is_validation": bool}]
    zero_start_vars: tuple = ()     # role names whose curves should be translated to start at 0 before fitting
    zero_start_output: bool = False  # whether the output's own curve should be translated to start at 0 before fitting


def _prior_to_dict(prior: ParamPrior) -> dict:
    return dict(
        kind=prior.kind, fixed_value=prior.fixed_value,
        uniform_lo=prior.uniform_lo, uniform_hi=prior.uniform_hi,
        hier_mu0=prior.hier_mu0, hier_s0=prior.hier_s0, hier_nu0=prior.hier_nu0,
    )


def _prior_from_dict(name: str, d: dict) -> ParamPrior:
    kind = d.get("kind")
    if kind not in PRIOR_KINDS:
        raise CalibProjectIOError(f"Parameter '{name}': unknown prior kind '{kind}' in the project file.")
    try:
        return ParamPrior(
            name=name, kind=kind,
            fixed_value=float(d.get("fixed_value", 0.0)),
            uniform_lo=float(d.get("uniform_lo", 0.0)), uniform_hi=float(d.get("uniform_hi", 1.0)),
            hier_mu0=float(d.get("hier_mu0", 0.0)), hier_s0=float(d.get("hier_s0", 1.0)),
            hier_nu0=float(d.get("hier_nu0", 3.0)),
        )
    except (TypeError, ValueError) as exc:
        raise CalibProjectIOError(f"Parameter '{name}': non-numeric prior field in the project file ({exc}).") from exc


def _json_path_for(csv_path: str) -> str:
    return csv_path.rsplit(".", 1)[0] + ".json"


def _csv_path_for(json_path: str) -> str:
    return json_path.rsplit(".", 1)[0] + ".csv"


def export_calibration_project(
    csv_path: str,
    equation: str,
    priors: dict,
    role_quantities: dict,
    output_quantity,
    covariance_mode: str,
    sampler_settings: dict,
    tests: list,
    used_vars: tuple,
    zero_start_vars=None,
    zero_start_output: bool = False,
) -> str:
    """Writes the "Calibration tests" table to `csv_path` (one row per
    entry in `tests`, columns `test_label`, one per role in `used_vars`
    (in that order), `output_entry`, `is_validation`) and everything else
    to the paired `.json` file (derived from `csv_path`). Returns the
    `.json` path actually written.

    `tests`: [{"label": str, "variable_entries": {role: entry_name str,
    possibly ""}, "output_entry": entry_name str (possibly ""),
    "is_validation": bool}, ...] -- exactly the page's own per-row widget
    state, read back as plain strings (an empty string means "not
    assigned yet", preserved as such rather than requiring every row be
    fully assigned before a project can be saved -- a work-in-progress
    setup is a legitimate thing to save and resume later).

    Raises OSError on a write failure (not wrapped -- the caller already
    handles OSError the same way every other export in this app does)."""
    json_path = _json_path_for(csv_path)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_label", *used_vars, "output_entry", "is_validation"])
        for test in tests:
            variable_entries = test.get("variable_entries", {})
            writer.writerow([
                test.get("label", ""),
                *[variable_entries.get(role, "") for role in used_vars],
                test.get("output_entry", ""),
                bool(test.get("is_validation", False)),
            ])

    payload = dict(
        format_version=FORMAT_VERSION,
        equation=equation,
        priors={name: _prior_to_dict(prior) for name, prior in priors.items()},
        role_quantities=dict(role_quantities),
        output_quantity=output_quantity,
        covariance_mode=covariance_mode,
        sampler_settings=dict(sampler_settings),
        zero_start_vars=sorted(set(zero_start_vars or ())),
        zero_start_output=bool(zero_start_output),
    )
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    return json_path


def import_calibration_project(json_path: str) -> CalibProjectData:
    """Reads a calibration-project JSON+CSV pair back. Raises OSError if
    the JSON file itself can't be read, or CalibProjectIOError for
    anything else wrong: malformed JSON, an unknown prior kind, or a
    missing/malformed companion CSV (required -- a project file without
    its test table is treated as incomplete, not silently accepted with
    zero tests, since that would silently discard real setup work were
    it re-saved afterward)."""
    with open(json_path, "r") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise CalibProjectIOError(f"Could not parse '{os.path.basename(json_path)}': {exc}") from exc

    if "equation" not in payload:
        raise CalibProjectIOError(f"'{os.path.basename(json_path)}' is missing its 'equation' field -- not a calibration project file?")

    priors = {
        name: _prior_from_dict(name, d)
        for name, d in payload.get("priors", {}).items()
    }
    try:
        for name, prior in priors.items():
            prior.validate()
    except PriorError as exc:
        raise CalibProjectIOError(f"Invalid prior in '{os.path.basename(json_path)}': {exc}") from exc

    csv_path = _csv_path_for(json_path)
    if not os.path.exists(csv_path):
        raise CalibProjectIOError(
            f"'{os.path.basename(json_path)}' has no matching '{os.path.basename(csv_path)}' "
            "(the calibration-tests table) next to it."
        )

    tests = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise CalibProjectIOError(f"'{os.path.basename(csv_path)}' is empty.")
        roles = [col for col in reader.fieldnames if col not in _NON_ROLE_CSV_COLUMNS]
        for row in reader:
            is_validation_raw = (row.get("is_validation") or "").strip().lower()
            tests.append(dict(
                label=row.get("test_label", ""),
                variable_entries={role: row.get(role, "") or "" for role in roles},
                output_entry=row.get("output_entry", "") or "",
                is_validation=is_validation_raw in ("true", "1", "yes"),
            ))

    zero_start_vars = payload.get("zero_start_vars")
    zero_start_output = payload.get("zero_start_output")
    if zero_start_vars is None and zero_start_output is None:
        # Back-compat with the very first version of this field (a single
        # "zero_start" bool that translated every role AND the output
        # uniformly, no per-role choice) -- reproduce that same effect
        # using this file's own tested roles, rather than dropping the
        # setting silently.
        legacy = bool(payload.get("zero_start", False))
        zero_start_vars = list(roles) if legacy else []
        zero_start_output = legacy

    return CalibProjectData(
        equation=payload["equation"],
        priors=priors,
        role_quantities=dict(payload.get("role_quantities", {})),
        output_quantity=payload.get("output_quantity"),
        covariance_mode=payload.get("covariance_mode", "per_test"),
        sampler_settings=dict(payload.get("sampler_settings", {})),
        tests=tests,
        zero_start_vars=tuple(zero_start_vars or ()),
        zero_start_output=bool(zero_start_output),
    )
