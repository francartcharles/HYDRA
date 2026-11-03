"""
Batch analysis across many already-processed tests, ported from
analysis_resume.py (`analysis_tests()`).

Combines several tests into one statistical summary against a common X
axis: the mean Y curve across tests, and three uncertainty bands:

- "Experimental" -- the RMS of each individual test's own reported
  Monte-Carlo uncertainty (`gg`/`ff` in the original).
- "Material" -- the test-to-test scatter around the mean, fit as a normal
  distribution on the ratio (this test's value / mean value) at each X
  level (`sort_sigma`/`sort_sigma3` in the original) -- unless overridden
  by a fixed percentage.
- "Total" -- the two combined in quadrature.

Optionally applies an isothermal correction (removing the adiabatic
temperature rise a test experienced, given its Taylor-Quinney coefficient,
specific heat, density and temperature sensitivity) before aggregating --
only meaningful when Y is a stress and X is a strain, as in the original.

X and Y are not fixed to strain/stress: `run_batch_analysis_from_files()`
takes each test as one plain CSV/TXT file, exactly as produced by the
Results & Export page's "Export as .csv…"/"Export as .txt…" (or "Export
full test…") buttons -- every computed quantity as a column, headed by
its internal name (`strain`, `stress`, `stress_std`, `inertia_stress`,
`confinement_pressure`, `confinement_deviator`, ... -- see
`page_results_export.QUANTITY_LABELS`). Any column can be X, any column
can be Y -- e.g. a pressure-deviator stress-path summary across several
confined tests, not just true stress vs. true strain. `load_export_header()`
reads just the header row so a UI can offer a column picker built from
whatever columns a given export actually contains, and a `<Y>_std` column
(when present) is picked up automatically for the "Experimental" band --
no separate picker needed for it.

NOT YET independently re-validated against the original application's own
output -- ported as literally as possible from the original for
traceability.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm


@dataclass
class IsothermalSettings:
    enabled: bool = False
    xi: float = 0.0
    cp: float = 0.0
    rho: float = 0.0
    alpha: float = 0.0  # temperature sensitivity (MPa/K)


@dataclass
class BatchAnalysisResult:
    x: np.ndarray
    x_label: str
    y_label: str
    mean: np.ndarray
    experimental_std: np.ndarray
    material_std: np.ndarray
    total_std: np.ndarray
    relative_total_pct: np.ndarray
    relative_experimental_pct: np.ndarray
    relative_material_pct: np.ndarray
    n_tests: int
    test_names: list = field(default_factory=list)
    # "target_x" -- resampled onto a common target-X grid (X behaves like
    # strain: monotonically increasing). "index_fraction" -- X wasn't
    # monotonic for at least one test (e.g. a stress path through unload/
    # reload), so each test was instead resampled by its own recorded
    # order; see `_resample_by_fraction`.
    resampling_method: str = "target_x"


def _sniff_delimiter(path: str) -> str:
    """Results & Export writes comma-separated for `.csv` and tab-separated
    (with a leading `# `-commented header line) for `.txt` -- pick the
    delimiter from the extension, the same rule `_write_columns` used to
    write it."""
    return "," if path.lower().endswith(".csv") else "\t"


def load_export_header(path: str) -> list[str]:
    """Reads just the header row of a CSV/TXT file exported by the Results
    & Export page ("Export as .csv…"/"Export as .txt…"), returning every
    column name in file order -- e.g. `["strain", "stress",
    "nominal_strain", ..., "inertia_stress", "friction_stress",
    "punch_strain", ..., "confinement_pressure", "confinement_deviator",
    ..., "strain_std", "stress_std", ...]`. Used to build the X/Y column
    pickers: any column can go on either axis."""
    delim = _sniff_delimiter(path)
    with open(path, "r", newline="") as f:
        first_line = f.readline()
    if not first_line:
        raise ValueError(f"{path} is empty")
    first_line = first_line.lstrip("#").strip()
    if delim == ",":
        header = next(csv.reader([first_line]))
    else:
        header = first_line.split(delim)
    header = [c.strip() for c in header if c.strip()]
    if not header:
        raise ValueError(f"Could not read a header row from {path}")
    return header


def _load_export_columns(path: str, column_map: dict[str, str]) -> dict[str, np.ndarray]:
    """Loads the columns named in `column_map` (e.g. `{"x": "strain", "y":
    "stress", "y_std": "stress_std"}`, values being actual column headers)
    out of one Results & Export CSV/TXT file. A key mapped to `""`/`None`
    is skipped (used for the optional std-dev column)."""
    delim = _sniff_delimiter(path)
    header = load_export_header(path)
    index = {name: i for i, name in enumerate(header)}

    wanted = {key: name for key, name in column_map.items() if name}
    missing = [name for name in wanted.values() if name not in index]
    if missing:
        raise ValueError(f"{os.path.basename(path)}: column(s) not found: {', '.join(missing)}")

    if delim == ",":
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)[1:]
        data = np.array([[float(row[i]) for i in range(len(header))] for row in rows if any(v.strip() for v in row)])
    else:
        data = np.loadtxt(path, delimiter=delim, skiprows=1)
        if data.ndim == 1:
            data = data.reshape(-1, len(header))

    return {key: data[:, index[name]] for key, name in wanted.items()}


def run_batch_analysis_from_files(
    files: list[str], x_column: str, y_column: str, sigma: int = 1,
    iso: IsothermalSettings | None = None,
    fixed_pct: float | None = None,
) -> BatchAnalysisResult:
    """The batch-analysis entry point: combines one CSV/TXT file per test,
    each produced by the Results & Export page's "Export as .csv…"/
    "Export as .txt…" (or "Export full test…") buttons. `x_column`/
    `y_column` pick, by column header, which exported quantity goes on
    each axis -- any column in the file, not just strain/stress (e.g.
    `confinement_pressure` on X and `confinement_deviator` on Y, to
    summarize several confined tests' stress paths). A `<y_column>_std`
    column, when present in a given file, is used automatically as that
    test's own "Experimental" uncertainty on Y -- no separate picker."""
    if not files:
        raise ValueError("No files selected.")
    if not x_column or not y_column:
        raise ValueError("Select both an X axis column and a Y axis column.")

    tests = []
    for path in files:
        header = load_export_header(path)
        std_name = f"{y_column}_std"
        column_map = {"x": x_column, "y": y_column}
        if std_name in header:
            column_map["y_std"] = std_name
        cols = _load_export_columns(path, column_map)

        n = min(len(cols["x"]), len(cols["y"]))
        y_std = cols.get("y_std")
        y_std = np.zeros(n) if y_std is None else y_std[:n]
        name = os.path.splitext(os.path.basename(path))[0]
        tests.append((name, cols["x"][:n], cols["y"][:n], y_std))

    return _combine_tests(
        tests, x_label=x_column, y_label=y_column, sigma=sigma, iso=iso, fixed_pct=fixed_pct,
    )


def _backtrack_fraction(x: np.ndarray) -> float:
    """How far X ever dips back below its own running maximum, as a
    fraction of its overall range -- 0 for a perfectly monotonically
    increasing X, small for one with only measurement-noise-level dips
    (real "strain" columns typically sit at a few percent, from ordinary
    Monte-Carlo/sensor noise near a plateau or peak), much larger for a
    genuinely non-monotonic quantity such as a stress-path pressure that
    unloads/reloads (typically 70-90%+)."""
    rng = float(x.max() - x.min())
    if rng <= 0:
        return 0.0
    running_max = np.maximum.accumulate(x)
    return float((running_max - x).max() / rng)


# Above this fraction (see `_backtrack_fraction`), X is treated as
# genuinely non-monotonic rather than merely noisy, and batch analysis
# falls back to resampling by the curve's own recorded order instead of
# by target-X lookup (see `_combine_tests`).
_BACKTRACK_TOLERANCE = 0.15


def _resample_by_fraction(x: np.ndarray, y: np.ndarray, n_points: int):
    """Resamples (x, y) onto `n_points` evenly-spaced points along the
    curve's own recorded order (by sample-index fraction 0..1), via linear
    interpolation against that fraction -- unlike resampling onto a fixed
    target-X grid, this needs no "the point where X equals so-and-so"
    search, so it stays well-behaved even when X loops back on itself (a
    stress path through unload/reload). Every point stays within this
    test's own actually-recorded range -- no extrapolation."""
    n = len(x)
    if n == 1:
        return np.full(n_points, x[0]), np.full(n_points, y[0])
    frac = np.linspace(0.0, 1.0, n)
    target_frac = np.linspace(0.0, 1.0, n_points)
    return np.interp(target_frac, frac, x), np.interp(target_frac, frac, y)


def _combine_tests(
    tests: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    x_label: str, y_label: str,
    sigma: int = 1,
    iso: IsothermalSettings | None = None,
    fixed_pct: float | None = None,
) -> BatchAnalysisResult:
    """Statistics core: given each test's own (name, x, y, y_std) arrays,
    resamples them onto a common X axis and combines them into one
    statistical summary. See the module docstring for what "Experimental"/
    "Material"/"Total" mean."""
    iso = iso or IsothermalSettings()

    test_names = [t[0] for t in tests]
    n_test = len(tests)
    if n_test == 0:
        raise ValueError("No tests to combine.")

    lengths = [len(t[1]) for t in tests]
    min_length = min(lengths)

    # Apply the isothermal correction (if any) to each test's own raw Y,
    # before any resampling -- unaffected by which resampling method ends
    # up used below.
    raw = []
    for name, x, y, y_std in tests:
        if iso.enabled:
            coeff_temp = iso.xi / (iso.cp * iso.rho)
            dT = [0.0]
            for g in range(len(y) - 1):
                dT.append(dT[g] + coeff_temp * y[g] * (x[g + 1] - x[g]) * 1e6)
            y = np.asarray([y[g] + iso.alpha * dT[g] for g in range(len(y) - 1)] + [y[-2] + iso.alpha * dT[-1]])
        raw.append((x, y, y_std))

    monotonic = all(_backtrack_fraction(x) <= _BACKTRACK_TOLERANCE for x, _y, _y_std in raw)
    resampling_method = "target_x" if monotonic else "index_fraction"

    if monotonic:
        # X behaves like strain -- resample onto a common target-X grid,
        # 95% of the smallest test's max X, over as many points as the
        # shortest test (matches the original's simple "last index below
        # target" search, not an interpolation).
        max_x = 0.95 * min(x.max() for x, _y, _y_std in raw)
        x_value = np.asarray([i * max_x / min_length for i in range(min_length)])

        all_y, all_y_std = [], []
        for x, y, y_std in raw:
            indices = []
            for target in x_value:
                below = np.where(x < target)[0]
                indices.append(below[-1] if len(below) else 1)
            all_y.append([round(float(y[i]), 6) for i in indices])
            all_y_std.append([round(float(y_std[i]), 6) for i in indices])
    else:
        # X isn't monotonically increasing (e.g. a stress path through
        # unload/reload) -- a target-X lookup breaks down here (it can
        # jump to an arbitrary point sharing that X value, anywhere along
        # the curve), so resample by the curve's own recorded order
        # instead (see `_resample_by_fraction`): every test contributes
        # `min_length` points evenly spaced through its own recording,
        # and the common X axis reported below is those points' own
        # (per-test-averaged) X value, not a synthetic target grid.
        all_x, all_y, all_y_std = [], [], []
        for x, y, y_std in raw:
            x_r, y_r = _resample_by_fraction(x, y, min_length)
            _, y_std_r = _resample_by_fraction(x, y_std, min_length)
            all_x.append(x_r)
            all_y.append(y_r)
            all_y_std.append(y_std_r)
        x_value = np.mean(np.array(all_x), axis=0)

    y_arr = np.array(all_y)              # (n_test, n_x)
    y_std_arr = np.array(all_y_std)

    mean_y = y_arr.mean(axis=0)

    # "Experimental" variability: RMS of each test's own reported std.
    experimental_std = np.sqrt((y_std_arr ** 2).mean(axis=0))

    # "Material" variability: test-to-test scatter, fit as a normal
    # distribution on (this test's value / mean value) at each X level.
    material_std = np.zeros(len(x_value))
    for g in range(len(x_value)):
        ratio = y_arr[:, g] / mean_y[g]
        _, sigma_fit = norm.fit(ratio)
        material_std[g] = sigma * sigma_fit * mean_y[g]

    if fixed_pct is not None:
        material_std = (fixed_pct / 100.0) * mean_y

    total_std = np.sqrt(material_std ** 2 + experimental_std ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_total = 100 * total_std / mean_y
        rel_exp = 100 * experimental_std / mean_y
        rel_mat = 100 * material_std / mean_y

    return BatchAnalysisResult(
        x=x_value, x_label=x_label, y_label=y_label, mean=mean_y,
        experimental_std=experimental_std, material_std=material_std, total_std=total_std,
        relative_total_pct=rel_total, relative_experimental_pct=rel_exp, relative_material_pct=rel_mat,
        n_tests=n_test, test_names=test_names, resampling_method=resampling_method,
    )


def export_batch_result(result: BatchAnalysisResult, folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    np.savetxt(os.path.join(folder, "x.txt"), result.x)
    np.savetxt(os.path.join(folder, "mean_y.txt"), result.mean)
    np.savetxt(os.path.join(folder, "material_variability_y_std.txt"), result.material_std)
    np.savetxt(os.path.join(folder, "total_y_std.txt"), result.total_std)
    np.savetxt(os.path.join(folder, "experimental_variability_y_std.txt"), result.experimental_std)
    with open(os.path.join(folder, "axes.txt"), "w") as f:
        f.write(f"x={result.x_label}\ny={result.y_label}\n")
