"""
Save/load `Project.covariance_library` (the Covariance Matrix page's
"Save to calibration library" entries -- see core/project.py's
`CovarianceLibraryEntry` and its own docstring) as a JSON+CSV file pair,
independent of the session's own automatic persistence (`core/
session.py`) -- for backing up, sharing between machines/colleagues, or
reloading a library into a fresh project.

Same file-pairing convention as core/calib_project_io.py and Results &
Export's "Export full test" (`ui/pages/page_results_export.py::
_on_export_full_test`): Save asks for the .csv path and derives the
.json path from it; Load asks for the .json path.

Unlike calib_project_io.py, the two files here are NOT complementary
halves of one format -- the JSON is the complete, authoritative,
round-trippable data (every field of every entry, including the full
grid/mean/covariance arrays, reusing the exact same shape core/session.py
already persists a session with, via `core.session._covariance_library_
to_dict`/`_from_dict`, so a manually exported-then-reimported library is
byte-for-byte the same shape a session restore produces) while the CSV is
a courtesy, human-readable SUMMARY table (one row per entry, metadata
columns only -- name/source/quantity/size/provenance, no grid/mean/
covariance numbers, which don't fit a flat table at all once entries have
different grid lengths). Loading only ever reads the JSON; the CSV is
written for review in a spreadsheet, not required (and not re-read) by
`import_covariance_library`.

Pure stdlib (json/csv) plus core.project -- no PySide6 -- so it is fully
real-testable in this sandbox.
"""
from __future__ import annotations

import csv
import json
import os

from core.session import _covariance_library_from_dict, _covariance_library_to_dict

FORMAT_VERSION = 1

_CSV_COLUMNS = (
    "name", "source_label", "quantity_key", "n_points", "n_draws_used",
    "grid_start", "grid_end", "created_at",
)


class CovarianceLibraryIOError(ValueError):
    """Raised for anything wrong with a calibration-library file: not a
    list, a missing 'name' on an entry, or malformed JSON."""


def _json_path_for(csv_path: str) -> str:
    return csv_path.rsplit(".", 1)[0] + ".json"


def export_covariance_library(csv_path: str, entries: list) -> str:
    """Writes the full `entries` (a `Project.covariance_library`, or any
    list of CovarianceLibraryEntry) to the paired `.json` file (derived
    from `csv_path`), and a metadata-only summary table to `csv_path`
    itself. Returns the `.json` path actually written.

    Raises OSError on a write failure (not wrapped, matching every other
    export in this app)."""
    json_path = _json_path_for(csv_path)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_COLUMNS)
        for e in entries:
            writer.writerow([
                e.name, e.source_label, e.quantity_key, len(e.grid), e.n_draws_used,
                e.grid_start, e.grid_end, e.created_at,
            ])

    with open(json_path, "w") as f:
        json.dump({"format_version": FORMAT_VERSION, "entries": _covariance_library_to_dict(entries)}, f, indent=2)

    return json_path


def import_covariance_library(json_path: str) -> list:
    """Reads a calibration-library JSON file back into a plain list of
    `CovarianceLibraryEntry` (the paired .csv, if present next to it, is
    never read -- see this module's own docstring for why). Raises
    OSError if the file can't be read, CovarianceLibraryIOError for
    malformed JSON or an entry missing its required 'name'."""
    with open(json_path, "r") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise CovarianceLibraryIOError(f"Could not parse '{os.path.basename(json_path)}': {exc}") from exc

    # Tolerate both the wrapped shape this module writes ({"entries": [...]})
    # and a bare list (e.g. a hand-trimmed file, or one produced by simply
    # dumping `core.session._covariance_library_to_dict(...)` directly).
    data = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        raise CovarianceLibraryIOError(
            f"'{os.path.basename(json_path)}' is not a calibration library file (expected a list of entries)."
        )
    for i, d in enumerate(data):
        if not isinstance(d, dict) or "name" not in d:
            raise CovarianceLibraryIOError(f"Entry #{i + 1} in '{os.path.basename(json_path)}' is missing its 'name' field.")

    return _covariance_library_from_dict(data)


def unique_entry_name(existing_names, name: str) -> str:
    """Auto-suffixes `name` (`_2`, `_3`, ...) until it no longer collides
    with `existing_names` -- the exact same collision policy `ui/pages/
    page_covariance.py::_on_save_to_library` already uses for a single
    manually-saved entry, reused here so an imported library never
    silently overwrites an entry already present in the current
    project."""
    existing = set(existing_names)
    if name not in existing:
        return name
    suffix = 2
    candidate = f"{name}_{suffix}"
    while candidate in existing:
        suffix += 1
        candidate = f"{name}_{suffix}"
    return candidate
