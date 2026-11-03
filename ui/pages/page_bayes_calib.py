"""
Bayesian Calibration page.

Ports MAMBA's free-text-equation Bayesian calibration (simple/pooled and
hierarchical) onto PyMC 5 + PyTensor (see /root/.claude/plans/bright-
gliding-ullman.md, the approved plan for this feature, for the full
design rationale). Unlike MAMBA's own per-test folder-of-.txt-files
loader, every equation variable (e, v, T, plus up to 3 optional extras
a/b/d) and the modeled/output variable are each assigned, per calibration
test, to any previously-saved (grid, mean curve, covariance matrix) entry
from the Covariance Analysis page's "Save to calibration library" list
(`project.covariance_library`).

MAMBA's own 4-step wizard (Define model -> Setup calibration -> Launch ->
Display Results) is flattened here into always-visible cards on one
scrollable page, matching this app's own Card/`make_scroll_page`
convention -- not a wizard.

Phase 2 covered cards 1-6: Equation, Free parameters, Calibration tests,
Sampler settings, Run, Result summary. Phase 3 (this file's current
version) adds the remaining diagnostic/validation/export cards (7-11) on
top of the same `self._last_result` (a `core.calib_run.CalibrationResult`)
Phase 2 already stores: Per-test calibration fit, Parameter
interdependence, Posterior diagnostics, Validation, Export.
"""
from __future__ import annotations

import csv
import os
import threading

import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QComboBox, QPushButton,
    QLineEdit, QLabel, QCheckBox, QProgressBar, QFileDialog,
)

from core.calib_equation import KNOWN_FUNCS, RESERVED_VARS, EquationError, parse_equation
from core.calib_library_adapter import (
    COVARIANCE_MODE_LABELS, COVARIANCE_MODES, DEFAULT_COVARIANCE_JITTER, entries_by_name,
)
from core.calib_priors import PRIOR_KINDS, PRIOR_KIND_LABELS, ParamPrior, PriorError, validate_priors
from core.calib_pymc_model import (
    DEFAULT_CHAINS, DEFAULT_DRAWS, DEFAULT_INIT, DEFAULT_TARGET_ACCEPT, DEFAULT_TUNE, pymc_available,
)
from core.calib_run import CalibRunError, run_calibration
from core.calib_project_io import (
    CalibProjectIOError, export_calibration_project, import_calibration_project,
)
from core.calib_diagnostics import (
    CalibDiagnosticsError, energy_diagnostic_arrays, forest_plot_rows, posterior_correlation_matrix,
    posterior_predictive_at, resolve_predictive_param_draws,
)
from core.covariance import QUANTITY_LABELS
from core.qt_worker import CalibWorker
from ui.widgets import Card, LabeledField, make_scroll_page
from ui.plot_canvas import PlotCanvas
from ui import theme

_KIND_FIXED, _KIND_UNIFORM, _KIND_HIER = PRIOR_KINDS  # ("fixed", "uniform", "normal_hier")

# Tooltip-only hint for each reserved-variable column header -- MAMBA's own
# conventional meaning for that letter. This is now only a hint: which
# PHYSICAL quantity a symbol actually maps to is the user's own choice, via
# the "Variable" combo row in the Calibration tests table below (any
# quantity in QUANTITY_LABELS -- e.g. 'e' doesn't have to mean strain, it
# could just as well be assigned hydrostatic pressure or volumetric strain).
_VAR_ROLE_HINTS = {
    "e": "Conventionally strain", "v": "Conventionally strain rate", "T": "Conventionally temperature",
    "a": "Extra variable", "b": "Extra variable", "d": "Extra variable",
}
# Sensible starting quantity for each reserved variable's "Variable" combo
# -- matches MAMBA's own convention, but is just the default selection, not
# a restriction; the user can repoint any of these (or Output) at any other
# quantity in QUANTITY_LABELS (e.g. pressure, volumetric_strain, ...).
_DEFAULT_QUANTITY_FOR_ROLE = {"e": "strain", "v": "strain_rate", "T": "temperature"}
_DEFAULT_OUTPUT_QUANTITY = "stress"
_ANY_QUANTITY_LABEL = "(any quantity)"

_INIT_METHODS = ("auto", "adapt_diag", "jitter+adapt_diag", "advi+adapt_diag")

# Above this RAW (pre-regularization) condition number, a test's own measured
# covariance matrix is flagged in Result summary as numerically dangerous for
# the exact MvNormal likelihood -- see core.calib_library_adapter.covariance_
# condition_number's docstring (a real case seen in practice measured ~4.6e7).
_ILL_CONDITIONED_THRESHOLD = 1.0e5

# Human-readable description per core.calib_equation domain-guard report key
# (CalibrationResult.domain_guard_warnings' inner dict) -- see core.calib_
# equation's "Domain guards" docstring section for why each of these is
# domain-restricted and gets clamped rather than left to crash/NaN.
_DOMAIN_GUARD_LABELS = {
    "sqrt": "sqrt(...) of a negative value",
    "log": "log(...) of a non-positive value",
    "log10": "log10(...) of a non-positive value",
    "acos": "acos(...) outside [-1, 1]",
    "asin": "asin(...) outside [-1, 1]",
    "acosh": "acosh(...) below 1",
    "atanh": "atanh(...) outside (-1, 1)",
    "pow": "a non-integer power (x**y) of a negative base",
}


def _fmt(value) -> str:
    try:
        if value == int(value) and abs(value) < 1e6:
            return str(int(value))
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        return f"{value:g}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_stat(value) -> str:
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return "?"


def _observed_band(dataset):
    """A 95% (±1.96σ) measurement-uncertainty envelope around a
    CalibTestDataset's own `observed` curve, from that same test's own
    `covariance` matrix diagonal (`sqrt(diag(covariance))` -- the
    per-point standard deviation the covariance matrix already carries;
    off-diagonal terms describe correlation BETWEEN points, not each
    point's own spread, so only the diagonal is needed here). Distinct
    from -- and shown alongside -- the posterior-predictive band: this
    one is about how uncertain the MEASUREMENT itself is, the other about
    how uncertain the FITTED MODEL is; before this, only the model's own
    band was ever drawn, leaving the observed curve looking like exact,
    noise-free data. `np.clip(..., 0, None)` guards against a covariance
    matrix with tiny negative diagonal entries from floating-point noise
    (a real covariance matrix is positive semi-definite, so this should
    never clip anything meaningful away)."""
    std = np.sqrt(np.clip(np.diag(np.asarray(dataset.covariance, dtype=float)), 0.0, None))
    observed = np.asarray(dataset.observed, dtype=float)
    return observed - 1.96 * std, observed + 1.96 * std


class BayesCalibPage(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._parsed = None                # last successful core.calib_equation.ParsedEquation, or None
        self._prior_widgets: dict = {}      # {free_param_name: {...widgets...}}
        self._test_rows: list = []          # [{"label_edit", "var_combos": {role: combo}, "output_combo", "validation_checkbox", "row_widgets": [...]}]
        self._test_row_counter = 0
        self._test_grid = None              # the QGridLayout currently laid out in tests_card
        self._role_quantity_combos: dict = {}   # {role: QComboBox} -- "which quantity does this symbol mean" picker
        self._output_quantity_combo = None      # same, for the Output column
        self._zero_start_checkboxes: dict = {}  # {role: QCheckBox} -- "translate this column's data to start at 0" picker
        self._output_zero_start_checkbox = None  # same, for the Output column
        self._thread = None
        self._worker = None
        self._cancel_event = None
        self._last_result = None            # core.calib_run.CalibrationResult, once a run has finished
        self._posterior_flat: dict = {}     # {arviz var name: (n_samples, ...) np.ndarray}, set in _on_finished

        scroll, layout = make_scroll_page(
            "Bayesian Calibration",
            "Fit a free-text constitutive-law equation to previously-computed strain/stress (or "
            "any other) data via Bayesian calibration -- simple (pooled) or hierarchical, per "
            "free parameter -- using each calibration test's own measured covariance matrix from "
            "the Covariance Matrix tab.",
        )

        # -- Calibration project (save/load), placed before Card 1 since it's
        # a whole-page action -- same file-pairing convention as Results &
        # Export's "Export full test" / Test Setup's "Load Test".
        self.project_io_card = Card(
            "Calibration project (save/load)",
            "Save the equation, free-parameter priors, calibration-tests table and sampler "
            "settings as a .json+.csv file pair -- load them back later, or share them, without "
            "re-typing the equation or re-picking every library entry by hand.",
            help_html=(
                "<p>The <code>.csv</code> is the Calibration tests table itself (one row per "
                "test); the <code>.json</code> carries the equation, every free parameter's prior, "
                "the per-column Variable quantity picker, the covariance mode, and the sampler "
                "settings. <b>Load</b> asks for the <code>.json</code> and looks for its matching "
                "<code>.csv</code> next to it.</p>"
                "<ul>"
                "<li>A project file only stores calibration-library entry NAMES for each test, "
                "never the underlying data -- load (or keep) the matching calibration library "
                "(Covariance Matrix tab) first, or those cells are reported as unresolved and left "
                "as \"-- choose --\".</li>"
                "<li><b>Save</b> works even with some test rows not yet fully assigned -- a "
                "work-in-progress setup is a legitimate thing to save and resume later.</li>"
                "<li><b>Load</b> replaces the current equation, priors, tests and sampler settings "
                "-- it does not merge with whatever is already on this page.</li>"
                "</ul>"
            ),
        )
        project_io_row = QHBoxLayout()
        self.save_project_button = QPushButton("Save project (.json+.csv)…")
        self.save_project_button.clicked.connect(self._on_save_calibration_project)
        project_io_row.addWidget(self.save_project_button)
        self.load_project_button = QPushButton("Load project (.json)…")
        self.load_project_button.clicked.connect(self._on_load_calibration_project)
        project_io_row.addWidget(self.load_project_button)
        project_io_row.addStretch(1)
        self.project_io_card.add_layout(project_io_row)
        self.project_io_status_label = QLabel("")
        self.project_io_status_label.setObjectName("cardSubtitle")
        self.project_io_status_label.setWordWrap(True)
        self.project_io_card.add(self.project_io_status_label)
        layout.addWidget(self.project_io_card)

        # -- Card 1: Equation ------------------------------------------------
        self.equation_card = Card(
            "Equation",
            help_html=(
                "<p>Type the constitutive-law equation to fit, e.g. "
                "<code>A * (1 + B*e^n) * (1 + C*log(v)) * (1 - T^m)</code> (a Johnson-Cook-like "
                "form).</p>"
                "<ul>"
                f"<li><b>Reserved variables</b>: {', '.join(RESERVED_VARS)} -- e (strain), v "
                "(strain rate), T (temperature), and up to 3 optional extras a/b/d.</li>"
                f"<li><b>Recognized functions</b>: {', '.join(KNOWN_FUNCS)}, plus the constant "
                "<code>pi</code>.</li>"
                "<li><b>Operators</b>: + - * / and ^ (power).</li>"
                "<li>Every other name typed (e.g. <code>A</code>, <code>B</code>, <code>n</code> "
                "above) is auto-detected as a free parameter to calibrate -- see the Free "
                "parameters card below.</li>"
                "<li><b>Domain guards</b>: sqrt/log/log10/acos/asin/acosh/atanh, and any power "
                "with a non-integer exponent, are automatically clamped into their valid domain "
                "(e.g. a slightly-negative strain-rate point from measurement noise feeding "
                "<code>log(v)</code> or <code>v^m</code> is clamped to a tiny positive value "
                "instead of crashing the run). A test whose real data needed this is flagged in "
                "the Result summary card.</li>"
                "</ul>"
            ),
        )
        self.equation_edit = QLineEdit()
        self.equation_edit.setPlaceholderText("e.g. A * (1 + B*e^n) * (1 + C*log(v)) * (1 - T^m)")
        self.equation_edit.editingFinished.connect(self._on_parse)
        self.equation_card.add(self.equation_edit)
        parse_row = QHBoxLayout()
        parse_button = QPushButton("Parse")
        parse_button.clicked.connect(self._on_parse)
        parse_row.addWidget(parse_button)
        parse_row.addStretch(1)
        self.equation_card.add_layout(parse_row)
        self.equation_status_label = QLabel("")
        self.equation_status_label.setObjectName("cardSubtitle")
        self.equation_status_label.setWordWrap(True)
        self.equation_card.add(self.equation_status_label)
        layout.addWidget(self.equation_card)

        # -- Card 2: Free parameters ------------------------------------------
        self.params_card = Card(
            "Free parameters",
            "Every parameter starts as Normal (hierarchical) -- the manual's own recommended "
            "workflow is to run once with everything hierarchical, inspect the parameter "
            "interdependence view once posterior diagnostics are available, then demote weakly "
            "or over-correlated parameters to Uniform (pooled/shared).",
            help_html=(
                "<p>One row per free parameter auto-detected from the equation above.</p>"
                "<ul>"
                "<li><b>Fixed value</b>: not calibrated -- used as this constant everywhere.</li>"
                "<li><b>Uniform (shared across tests)</b>: a single value fit across every "
                "calibration test.</li>"
                "<li><b>Normal (hierarchical, per-test)</b>: a genuine per-test random effect "
                "with partial pooling across tests (a Normal-Inverse-ChiSquared hyperprior on its "
                "own mean/scale).</li>"
                "</ul>"
            ),
        )
        self.params_card.setVisible(False)
        layout.addWidget(self.params_card)

        # -- Card 3: Calibration tests -----------------------------------------
        self.tests_card = Card(
            "Calibration tests",
            "One row per test. First pick, per column, which physical quantity that equation "
            "symbol (or the Output) actually represents -- it does not have to match the "
            "letter's usual meaning (e.g. 'e' can be assigned hydrostatic pressure or volumetric "
            "strain instead of strain). Then, per test, assign a saved entry from the Covariance "
            "Matrix tab's calibration library. Tick \"Validation only\" for a test that should "
            "NOT contribute to the fit (used instead for a held-out posterior-predictive check, "
            "once available).",
            help_html=(
                "<p>Populated from every entry saved on the Covariance Matrix tab's \"Save to "
                "calibration library\" card (<code>project.covariance_library</code>) -- go there "
                "first if the dropdowns below are empty.</p>"
                "<ul>"
                "<li>Rebuilt whenever the equation above is (re)parsed, since the set of variable "
                "columns depends on which reserved variables the equation actually uses.</li>"
                "<li>The <b>Variable</b> row (just under the column headers) picks, once per "
                "column, which physical quantity (strain, hydrostatic pressure, volumetric "
                "strain, temperature, ...) that column's entries should be -- a reserved letter "
                "like 'e' or 'v' is just a symbol in the equation, not a fixed meaning, so this "
                "lets any equation variable be fit against any saved quantity. Choosing "
                "\"(any quantity)\" (the default for the optional a/b/d columns) shows every "
                "saved entry regardless of what quantity it was computed for.</li>"
                "<li>Changing a column's <b>Variable</b> re-filters every test row's dropdown for "
                "that column down to entries matching that quantity (an entry with no recorded "
                "quantity is never hidden, since its provenance is simply unknown).</li>"
                "<li>The <b>Output</b> column is the modeled/predicted quantity (defaults to "
                "stress, but can be repointed the same way) -- its entry's own grid, mean curve "
                "and covariance matrix are what the likelihood is fit against.</li>"
                "<li>The <b>Start at zero</b> row (below Variable) is one independent checkbox per "
                "column -- tick a column to translate every test's curve for THAT column only "
                "(subtract its own first sample, so it starts at exactly 0), leaving every other "
                "column's data untouched. Useful for a channel with a known zero offset (e.g. "
                "strain-gauge drift) without also shifting variables that don't need it. A "
                "translation never changes a curve's variance, so covariance matrices (and the "
                "likelihood's uncertainty) are unaffected either way. Checking <b>e</b> (strain) "
                "also shifts the plotted grid/x-axis on the Per-test calibration fit and "
                "Validation charts to start at 0, since every saved entry's grid axis is itself "
                "strain -- this is purely a display shift and never changes the calibration's own "
                "numbers.</li>"
                "<li><b>Validation only</b>: excluded from the fit entirely; reserved for a "
                "held-out check (Phase 3).</li>"
                "</ul>"
            ),
        )
        self.tests_card.setVisible(False)
        layout.addWidget(self.tests_card)

        # -- Card 4: Sampler settings -------------------------------------------
        self.sampler_card = Card(
            "Sampler settings",
            help_html=(
                "<p>Controls for the underlying PyMC NUTS sampler.</p>"
                "<ul>"
                "<li><b>Covariance mode</b>: Per-test (default) uses each test's own measured "
                "covariance in its likelihood term; Shared mean averages every test's covariance "
                "matrix into one, matching every matrix's own N (grid points).</li>"
                "<li><b>Draws / Tune</b>: post-warmup samples kept / warmup samples discarded, "
                "per chain.</li>"
                "<li><b>Chains / Cores</b>: independent MCMC chains run (for convergence "
                "checking via R-hat) / worker processes used to run them.</li>"
                "<li><b>Target accept</b>: NUTS's target acceptance probability -- higher is more "
                "conservative (slower, fewer divergences); PyMC's own default is 0.9.</li>"
                "<li><b>Covariance jitter</b>: 0 by default (no change at all). A covariance matrix "
                "built from Monte-Carlo draws (Covariance Analysis page) can be numerically "
                "ill-conditioned -- a real case seen in practice had a condition number around 46 "
                "million, which makes the exact likelihood fit an almost-arbitrary near-null "
                "direction instead of the data's own visible shape, sometimes pulling calibrated "
                "parameters to wildly wrong values even though R-hat still looks fine (it converged "
                "confidently onto the wrong answer). Result summary flags any calibration test whose "
                "own covariance is severely ill-conditioned; if flagged, try 0.01-0.1 here (1-10% of "
                "that matrix's own mean diagonal added as a ridge) and re-run -- larger values weaken "
                "how strongly the measured covariance's exact structure constrains the fit, so use "
                "the smallest value that clears the warning.</li>"
                "</ul>"
            ),
        )
        self.covariance_mode_combo = QComboBox()
        for mode in COVARIANCE_MODES:
            self.covariance_mode_combo.addItem(COVARIANCE_MODE_LABELS[mode], mode)
        self.sampler_card.add(LabeledField("Covariance mode", self.covariance_mode_combo))
        sampler_row1 = QHBoxLayout()
        sampler_row1.addWidget(QLabel("Draws"))
        self.draws_edit = QLineEdit(str(DEFAULT_DRAWS))
        self.draws_edit.setMaximumWidth(80)
        sampler_row1.addWidget(self.draws_edit)
        sampler_row1.addSpacing(16)
        sampler_row1.addWidget(QLabel("Tune"))
        self.tune_edit = QLineEdit(str(DEFAULT_TUNE))
        self.tune_edit.setMaximumWidth(80)
        sampler_row1.addWidget(self.tune_edit)
        sampler_row1.addSpacing(16)
        sampler_row1.addWidget(QLabel("Chains"))
        self.chains_edit = QLineEdit(str(DEFAULT_CHAINS))
        self.chains_edit.setMaximumWidth(60)
        sampler_row1.addWidget(self.chains_edit)
        sampler_row1.addSpacing(16)
        sampler_row1.addWidget(QLabel("Cores"))
        self.cores_edit = QLineEdit(str(max(1, getattr(project, "n_workers", 1))))
        self.cores_edit.setMaximumWidth(60)
        sampler_row1.addWidget(self.cores_edit)
        sampler_row1.addStretch(1)
        self.sampler_card.add_layout(sampler_row1)
        sampler_row2 = QHBoxLayout()
        sampler_row2.addWidget(QLabel("Target accept"))
        self.target_accept_edit = QLineEdit(str(DEFAULT_TARGET_ACCEPT))
        self.target_accept_edit.setMaximumWidth(80)
        sampler_row2.addWidget(self.target_accept_edit)
        sampler_row2.addSpacing(16)
        sampler_row2.addWidget(QLabel("Init method"))
        self.init_combo = QComboBox()
        for method in _INIT_METHODS:
            self.init_combo.addItem(method, method)
        sampler_row2.addWidget(self.init_combo)
        sampler_row2.addSpacing(16)
        sampler_row2.addWidget(QLabel("Covariance jitter"))
        self.covariance_jitter_edit = QLineEdit(str(DEFAULT_COVARIANCE_JITTER))
        self.covariance_jitter_edit.setMaximumWidth(70)
        self.covariance_jitter_edit.setToolTip(
            "0 = no change (default). A fraction of a covariance matrix's own mean diagonal, added "
            "as a ridge before it's used in the likelihood -- fixes a severely ill-conditioned "
            "covariance matrix (see Result summary's own warning, and this card's help). Try "
            "0.01-0.1 if flagged."
        )
        sampler_row2.addWidget(self.covariance_jitter_edit)
        sampler_row2.addStretch(1)
        self.sampler_card.add_layout(sampler_row2)
        self.pymc_status_label = QLabel("")
        self.pymc_status_label.setObjectName("cardSubtitle")
        self.pymc_status_label.setWordWrap(True)
        self.sampler_card.add(self.pymc_status_label)
        layout.addWidget(self.sampler_card)

        # -- Card 5: Run ---------------------------------------------------------
        self.run_card = Card(
            "Run",
            "MCMC sampling can take a while, especially with several hierarchical (per-test) "
            "parameters.",
        )
        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._on_run)
        run_row.addWidget(self.run_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip(
            "PyMC has no supported way to interrupt sampling once it has actually started -- Stop "
            "only takes effect if clicked before sampling begins (e.g. while the model is still "
            "being built). Once NUTS is running, let it finish."
        )
        self.stop_button.clicked.connect(self._on_stop)
        run_row.addWidget(self.stop_button)
        run_row.addStretch(1)
        self.run_card.add_layout(run_row)
        self.progress_bar = QProgressBar()
        self.run_card.add(self.progress_bar)
        self.run_status_label = QLabel("")
        self.run_status_label.setObjectName("cardSubtitle")
        self.run_status_label.setWordWrap(True)
        self.run_card.add(self.run_status_label)
        layout.addWidget(self.run_card)

        # -- Card 6: Result summary ------------------------------------------------
        self.summary_card = Card(
            "Result summary",
            help_html=(
                "<p>Per-parameter posterior summary from the last successful run.</p>"
                "<ul>"
                "<li><b>Pooled</b> parameters (Uniform prior): one mean &plusmn; std shared "
                "across every calibration test.</li>"
                "<li><b>Hierarchical</b> parameters (Normal prior): the hyper-mean &plusmn; "
                "hyper-std across tests, i.e. the population-level estimate (each test's own "
                "individual value is available in the full posterior, not shown here).</li>"
                "<li><b>R-hat</b> above 1.01 is flagged -- it means that parameter's chains have "
                "not converged to the same distribution and the result should not be trusted "
                "as-is (more draws/tune, or a reparameterization, is needed).</li>"
                "</ul>"
            ),
        )
        self.summary_label = QLabel("No calibration run yet.")
        self.summary_label.setObjectName("cardSubtitle")
        self.summary_label.setWordWrap(True)
        self.summary_card.add(self.summary_label)
        layout.addWidget(self.summary_card)

        # -- Card 7: Per-test calibration fit ---------------------------------------
        self.fit_card = Card(
            "Per-test calibration fit",
            "Observed data (with its own measurement-uncertainty band) vs. the posterior-"
            "predictive mean and band, at one calibration test's own conditions.",
            help_html=(
                "<p>Pick a calibration test (one that actually contributed to the fit, not a "
                "\"Validation only\" row -- see the Validation card below for those) to see how "
                "well the calibrated model reproduces it.</p>"
                "<p>The green shaded band is the 2.5th-97.5th percentile of the equation evaluated "
                "at this test's own e/v/T/a/b/d, once per posterior draw -- a hierarchical "
                "parameter uses its own value AT THIS TEST specifically, not the population "
                "hyper-mean. The grey shaded band around the observed curve is that test's own "
                "measurement uncertainty (±1.96&sigma; from its covariance matrix diagonal).</p>"
            ),
        )
        self.fit_test_combo = QComboBox()
        self.fit_test_combo.currentIndexChanged.connect(self._on_fit_test_changed)
        self.fit_card.add(LabeledField("Test", self.fit_test_combo))
        self.fit_canvas = PlotCanvas(figsize=(8, 4.5), help_title="Per-test calibration fit", help_text=(
            "<p>Solid line: observed (measured) curve. Dashed line: posterior-predictive mean.</p>"
            "<p>Two different shaded bands, both at 95%: the grey <b>Observed band</b> is the "
            "measurement's own uncertainty (from this test's own covariance matrix diagonal); the "
            "green <b>Posterior-predictive band</b> is how uncertain the calibrated model's fit "
            "is. They answer different questions -- a wide observed band means noisy data, a wide "
            "predictive band means the model/parameters are themselves poorly constrained.</p>"
        ))
        self.fit_card.add(self.fit_canvas)
        self.fit_status_label = QLabel("")
        self.fit_status_label.setObjectName("cardSubtitle")
        self.fit_status_label.setWordWrap(True)
        self.fit_card.add(self.fit_status_label)
        self.fit_card.setVisible(False)
        layout.addWidget(self.fit_card)

        # -- Card 8: Parameter interdependence ---------------------------------------
        self.interdep_card = Card(
            "Parameter interdependence",
            "How strongly each pair of calibrated parameters trades off against the other across "
            "the posterior. A value near ±1 means the data mostly constrains some "
            "combination of the two (e.g. their product) rather than either one individually.",
            help_html=(
                "<p>A color matrix showing how strongly each pair of calibrated parameters "
                "trades off against the other across the posterior (Pearson correlation of their "
                "posterior draws -- for a hierarchical parameter, its population hyper-mean "
                "draws).</p>"
                "<p>A cell near &plusmn;1 means more search/data will not resolve that "
                "ambiguity on its own -- only a different or additional measurement can.</p>"
            ),
        )
        self.interdep_canvas = PlotCanvas(figsize=(5.5, 5), show_toolbar=False)
        self.interdep_card.add(self.interdep_canvas)
        self.interdep_status_label = QLabel("")
        self.interdep_status_label.setObjectName("cardSubtitle")
        self.interdep_status_label.setWordWrap(True)
        self.interdep_card.add(self.interdep_status_label)
        self.interdep_card.setVisible(False)
        layout.addWidget(self.interdep_card)

        # -- Card 9: Posterior diagnostics ------------------------------------------
        self.diag_card = Card(
            "Posterior diagnostics",
            help_html=(
                "<p><b>Posterior density</b>: each free parameter's own marginal posterior "
                "distribution (a hierarchical parameter's population hyper-mean).</p>"
                "<p><b>Forest plot</b>: each parameter's point estimate and 94% interval, with "
                "R-hat next to its name (flagged if above 1.01). A hierarchical (per-test) "
                "parameter shows its population hyper-mean AND every individual calibration "
                "test's own sampled value, labeled with that test's name -- useful for checking "
                "whether a specific test's own value has drifted far from the hyper-mean shown "
                "in Result summary (the Per-test calibration fit card uses each test's own "
                "value, not the hyper-mean).</p>"
                "<p><b>Energy plot</b>: NUTS's own sampler-health diagnostic (BFMI) -- a large "
                "mismatch between the marginal and transition energy distributions shown flags a "
                "sampler that struggled with this posterior's geometry.</p>"
            ),
        )
        self.diag_kind_combo = QComboBox()
        self.diag_kind_combo.addItem("Posterior density", "density")
        self.diag_kind_combo.addItem("Forest plot", "forest")
        self.diag_kind_combo.addItem("Energy plot", "energy")
        self.diag_kind_combo.currentIndexChanged.connect(self._on_diag_kind_changed)
        self.diag_card.add(LabeledField("Show", self.diag_kind_combo))
        self.diag_canvas = PlotCanvas(figsize=(8, 5))
        self.diag_card.add(self.diag_canvas)
        self.diag_status_label = QLabel("")
        self.diag_status_label.setObjectName("cardSubtitle")
        self.diag_status_label.setWordWrap(True)
        self.diag_card.add(self.diag_status_label)
        self.diag_card.setVisible(False)
        layout.addWidget(self.diag_card)

        # -- Card 10: Validation -----------------------------------------------------
        self.validation_card = Card(
            "Validation",
            "Only shown once at least one \"Validation only\" test row was used in the last run. "
            "A genuine posterior-predictive check at that test's own conditions -- it never "
            "contributed to the fit.",
            help_html=(
                "<p>A real held-out check: this equation, calibrated WITHOUT this test's data, "
                "evaluated at this test's own e/v/T/a/b/d (fixes MAMBA's own \"validation\" view, "
                "which only ever re-plotted the calibration fit rather than actually evaluating "
                "the model here). As on the Per-test calibration fit card, the grey band is this "
                "test's own measurement uncertainty and the green band is the model's "
                "posterior-predictive uncertainty.</p>"
                "<p><b>Caveat for hierarchical parameters:</b> since this test was never fit, a "
                "hierarchical parameter has no per-test value for it -- the prediction uses that "
                "parameter's population hyper-mean instead, ignoring the between-test scatter a "
                "genuinely new test would also be subject to. The shown band is therefore "
                "narrower than the true predictive uncertainty when hierarchical parameters are "
                "involved.</p>"
            ),
        )
        self.validation_test_combo = QComboBox()
        self.validation_test_combo.currentIndexChanged.connect(self._on_validation_test_changed)
        self.validation_card.add(LabeledField("Validation test", self.validation_test_combo))
        self.validation_canvas = PlotCanvas(figsize=(8, 4.5))
        self.validation_card.add(self.validation_canvas)
        self.validation_status_label = QLabel("")
        self.validation_status_label.setObjectName("cardSubtitle")
        self.validation_status_label.setWordWrap(True)
        self.validation_card.add(self.validation_status_label)
        self.validation_card.setVisible(False)
        layout.addWidget(self.validation_card)

        # -- Card 11: Export ----------------------------------------------------------
        self.export_card = Card(
            "Export",
            "Exports every currently-populated figure above (as separate PDF pages in one file) "
            "and/or the posterior summary table.",
        )
        export_row = QHBoxLayout()
        self.export_figures_button = QPushButton("Export all figures as PDF…")
        self.export_figures_button.clicked.connect(self._on_export_figures)
        export_row.addWidget(self.export_figures_button)
        self.export_csv_button = QPushButton("Export posterior summary as .csv…")
        self.export_csv_button.clicked.connect(lambda: self._on_export_summary("csv"))
        export_row.addWidget(self.export_csv_button)
        self.export_txt_button = QPushButton("Export posterior summary as .txt…")
        self.export_txt_button.clicked.connect(lambda: self._on_export_summary("txt"))
        export_row.addWidget(self.export_txt_button)
        export_row.addStretch(1)
        self.export_card.add_layout(export_row)
        self.export_status_label = QLabel("")
        self.export_status_label.setObjectName("cardSubtitle")
        self.export_status_label.setWordWrap(True)
        self.export_card.add(self.export_status_label)
        self.export_card.setVisible(False)
        layout.addWidget(self.export_card)

        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._refresh_pymc_status()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_pymc_status()
        self._refresh_library_combos()

    # -- Card 1: Equation ---------------------------------------------------

    def _on_parse(self):
        text = self.equation_edit.text()
        try:
            parsed = parse_equation(text)
        except EquationError as exc:
            self._parsed = None
            self.equation_status_label.setText(str(exc))
            self.params_card.setVisible(False)
            self.tests_card.setVisible(False)
            return

        used_vars_changed = self._parsed is None or self._parsed.used_vars != parsed.used_vars
        free_params_changed = self._parsed is None or self._parsed.free_params != parsed.free_params
        self._parsed = parsed

        var_text = ", ".join(parsed.used_vars) if parsed.used_vars else "(none)"
        param_text = ", ".join(parsed.free_params) if parsed.free_params else "(none)"
        self.equation_status_label.setText(f"Variables used: {var_text}. Free parameters: {param_text}.")

        if free_params_changed:
            self._rebuild_param_rows()
        self.params_card.setVisible(True)

        if used_vars_changed:
            self._rebuild_test_table()
        self.tests_card.setVisible(True)

    # -- Card 2: Free parameters ---------------------------------------------

    def _rebuild_param_rows(self):
        self.params_card.clear()
        self._prior_widgets = {}
        for name in self._parsed.free_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(name))

            combo = QComboBox()
            for kind in PRIOR_KINDS:
                combo.addItem(PRIOR_KIND_LABELS[kind], kind)
            combo.setCurrentIndex(PRIOR_KINDS.index(_KIND_HIER))  # manual's own recommended starting point
            combo.setMaximumWidth(240)
            row.addWidget(combo)

            fixed_group = QWidget()
            fixed_layout = QHBoxLayout(fixed_group)
            fixed_layout.setContentsMargins(0, 0, 0, 0)
            fixed_edit = QLineEdit("0")
            fixed_edit.setMaximumWidth(80)
            fixed_layout.addWidget(QLabel("Value"))
            fixed_layout.addWidget(fixed_edit)
            row.addWidget(fixed_group)

            uniform_group = QWidget()
            uniform_layout = QHBoxLayout(uniform_group)
            uniform_layout.setContentsMargins(0, 0, 0, 0)
            uniform_lo_edit = QLineEdit("0")
            uniform_lo_edit.setMaximumWidth(70)
            uniform_hi_edit = QLineEdit("1")
            uniform_hi_edit.setMaximumWidth(70)
            uniform_layout.addWidget(QLabel("Min"))
            uniform_layout.addWidget(uniform_lo_edit)
            uniform_layout.addWidget(QLabel("Max"))
            uniform_layout.addWidget(uniform_hi_edit)
            row.addWidget(uniform_group)

            hier_group = QWidget()
            hier_layout = QHBoxLayout(hier_group)
            hier_layout.setContentsMargins(0, 0, 0, 0)
            hier_mu0_edit = QLineEdit("0")
            hier_mu0_edit.setMaximumWidth(70)
            hier_s0_edit = QLineEdit("1")
            hier_s0_edit.setMaximumWidth(70)
            hier_nu0_edit = QLineEdit("3")
            hier_nu0_edit.setMaximumWidth(70)
            hier_layout.addWidget(QLabel("mu0"))
            hier_layout.addWidget(hier_mu0_edit)
            hier_layout.addWidget(QLabel("s0"))
            hier_layout.addWidget(hier_s0_edit)
            hier_layout.addWidget(QLabel("nu0"))
            hier_layout.addWidget(hier_nu0_edit)
            row.addWidget(hier_group)

            row.addStretch(1)

            groups = dict(fixed=fixed_group, uniform=uniform_group, hier=hier_group)

            # `groups=groups` binds *this row's* groups into the closure at
            # definition time -- without it every row's handler would share
            # the same late-bound free variable and all end up toggling
            # whichever row was built last (see page_visco_calib.py's own
            # `_rebuild_param_cards` for the identical fix and rationale).
            def _on_kind_changed(index, groups=groups):
                kind = PRIOR_KINDS[index]
                groups["fixed"].setVisible(kind == _KIND_FIXED)
                groups["uniform"].setVisible(kind == _KIND_UNIFORM)
                groups["hier"].setVisible(kind == _KIND_HIER)

            combo.currentIndexChanged.connect(_on_kind_changed)
            _on_kind_changed(combo.currentIndex())

            self._prior_widgets[name] = dict(
                combo=combo, groups=groups,
                fixed=dict(edit=fixed_edit),
                uniform=dict(lo=uniform_lo_edit, hi=uniform_hi_edit),
                hier=dict(mu0=hier_mu0_edit, s0=hier_s0_edit, nu0=hier_nu0_edit),
            )
            self.params_card.add_layout(row)

    def _read_priors(self) -> dict:
        """Reads every free parameter's prior row back into a {name:
        ParamPrior} dict, validated. Raises ValueError/PriorError with a
        human-readable message on non-numeric input or an invalid range
        -- caught by `_on_run`'s single status-label error path."""
        priors = {}
        for name, widgets in self._prior_widgets.items():
            kind = PRIOR_KINDS[widgets["combo"].currentIndex()]
            if kind == _KIND_FIXED:
                value = float(widgets["fixed"]["edit"].text())
                priors[name] = ParamPrior(name=name, kind=kind, fixed_value=value)
            elif kind == _KIND_UNIFORM:
                lo = float(widgets["uniform"]["lo"].text())
                hi = float(widgets["uniform"]["hi"].text())
                priors[name] = ParamPrior(name=name, kind=kind, uniform_lo=lo, uniform_hi=hi)
            else:
                mu0 = float(widgets["hier"]["mu0"].text())
                s0 = float(widgets["hier"]["s0"].text())
                nu0 = float(widgets["hier"]["nu0"].text())
                priors[name] = ParamPrior(name=name, kind=kind, hier_mu0=mu0, hier_s0=s0, hier_nu0=nu0)
        validate_priors(priors)
        return priors

    # -- Card 3: Calibration tests --------------------------------------------

    def _rebuild_test_table(self):
        """Rebuilds the whole test table's column set (one combo per
        currently-used reserved variable, plus Output/Validation/Remove)
        -- called only when the equation's own `used_vars` actually
        changes, matching this app's established "rebuild wholesale on
        shape change" convention (e.g. page_visco_calib.py's own
        `_rebuild_param_cards` on a model switch). Existing row
        assignments are intentionally not preserved across this, since
        the variable roles themselves have changed.

        Grid layout: row 0 is the column header (bare symbol letters);
        row 1 is the "Variable" picker -- ONE combo per column, chosen
        once for the whole table, selecting which physical quantity
        (QUANTITY_LABELS) that column's entries must match; row 2 is the
        "Start at zero" picker -- ONE checkbox per column (independent of
        every other column, see zero_start_vars/zero_start_output in
        core.calib_library_adapter.build_test_dataset), deciding whether
        that column's curve gets translated to start at 0 before fitting;
        row 3 onward are the per-test entry-assignment rows themselves,
        each one filtered down to that column's chosen quantity."""
        self.tests_card.clear()
        self._test_rows = []
        self._test_row_counter = 0
        self._role_quantity_combos = {}
        self._output_quantity_combo = None
        self._zero_start_checkboxes = {}
        self._output_zero_start_checkbox = None

        used_vars = self._parsed.used_vars if self._parsed else ()
        self._test_grid = QGridLayout()
        self._test_grid.setHorizontalSpacing(10)
        self._test_grid.setVerticalSpacing(10)

        headers = ["Test"] + list(used_vars) + ["Output", "Validation only", ""]
        for col, text in enumerate(headers):
            lbl = QLabel(text)
            lbl.setObjectName("cardSubtitle")
            if text in _VAR_ROLE_HINTS:
                lbl.setToolTip(_VAR_ROLE_HINTS[text])
            self._test_grid.addWidget(lbl, 0, col)

        variable_row_label = QLabel("Variable:")
        variable_row_label.setObjectName("cardSubtitle")
        self._test_grid.addWidget(variable_row_label, 1, 0)
        col = 1
        for role in used_vars:
            quantity_combo = self._make_quantity_combo(_DEFAULT_QUANTITY_FOR_ROLE.get(role))
            quantity_combo.currentIndexChanged.connect(
                lambda _index, role=role: self._on_role_quantity_changed(role)
            )
            self._test_grid.addWidget(quantity_combo, 1, col)
            self._role_quantity_combos[role] = quantity_combo
            col += 1
        self._output_quantity_combo = self._make_quantity_combo(_DEFAULT_OUTPUT_QUANTITY)
        self._output_quantity_combo.currentIndexChanged.connect(self._on_output_quantity_changed)
        self._test_grid.addWidget(self._output_quantity_combo, 1, col)

        zero_row_label = QLabel("Start at zero:")
        zero_row_label.setObjectName("cardSubtitle")
        zero_row_label.setToolTip(
            "Independently, per column: translate every test's curve for that column so it starts "
            "at exactly 0 (subtracts its own first sample). Leaves untouched any column left "
            "unchecked."
        )
        self._test_grid.addWidget(zero_row_label, 2, 0)
        col = 1
        for role in used_vars:
            zero_checkbox = QCheckBox()
            if role == "e":
                zero_checkbox.setToolTip(
                    "Translate every test's 'e' (strain) curve to start at 0. Since every "
                    "covariance-library entry's grid axis is itself strain, this also shifts the "
                    "plotted grid/x-axis on the Per-test calibration fit and Validation charts so "
                    "it starts at 0 -- not just the 'e' values fed into the equation."
                )
            else:
                zero_checkbox.setToolTip(f"Translate every test's '{role}' curve to start at 0.")
            self._test_grid.addWidget(zero_checkbox, 2, col)
            self._zero_start_checkboxes[role] = zero_checkbox
            col += 1
        self._output_zero_start_checkbox = QCheckBox()
        self._output_zero_start_checkbox.setToolTip("Translate every test's Output curve to start at 0.")
        self._test_grid.addWidget(self._output_zero_start_checkbox, 2, col)

        self.tests_card.add_layout(self._test_grid)

        # Built fresh here (rather than once in __init__ and reused) since
        # `Card.clear()` above recursively empties -- and deleteLater()s
        # the widgets inside -- every layout previously added to the card,
        # including this one; reusing the same QHBoxLayout/QPushButton
        # instances across rebuilds would leave "Add test" silently
        # missing after the very first equation reparse.
        add_test_row = QHBoxLayout()
        self.add_test_button = QPushButton("Add test")
        self.add_test_button.clicked.connect(self._on_add_test_row)
        add_test_row.addWidget(self.add_test_button)
        add_test_row.addStretch(1)
        self.tests_card.add_layout(add_test_row)

        self._on_add_test_row()

    def _make_quantity_combo(self, default_key: str | None) -> QComboBox:
        """One "Variable" picker (row 1 of the test table) -- lets the
        user choose which physical quantity (any key in QUANTITY_LABELS,
        e.g. 'pressure' or 'volumetric_strain', not just the
        conventional meaning of the column's reserved-variable letter)
        this column's entries must be. `(any quantity)` (data=None)
        applies no filter at all."""
        combo = QComboBox()
        combo.addItem(_ANY_QUANTITY_LABEL, None)
        default_index = 0
        for i, (key, label) in enumerate(QUANTITY_LABELS.items(), start=1):
            combo.addItem(label, key)
            if key == default_key:
                default_index = i
        combo.setCurrentIndex(default_index)
        return combo

    def _make_library_combo(self, quantity_filter: str | None) -> QComboBox:
        combo = QComboBox()
        self._populate_library_combo(combo, quantity_filter, preserve_name=None)
        return combo

    def _populate_library_combo(self, combo: QComboBox, quantity_filter, preserve_name=None):
        """(Re)fills `combo` from `project.covariance_library`, keeping
        only entries whose own `quantity_key` matches `quantity_filter`
        (an entry with a BLANK quantity_key is never filtered out --
        better to show a possibly-relevant entry than to hide one just
        because its provenance is unknown). `quantity_filter=None` means
        "(any quantity)" -- no filtering at all. Preserves the combo's
        current selection by entry NAME (falling back to `preserve_name`
        when given, e.g. right after the quantity filter itself just
        changed) when that entry still passes the filter, else resets to
        "-- choose --"."""
        if preserve_name is None:
            preserve_name = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("-- choose --", None)
        select_index = 0
        i = 1
        for entry in self.project.covariance_library:
            if quantity_filter and entry.quantity_key and entry.quantity_key != quantity_filter:
                continue
            combo.addItem(f"{entry.name} ({len(entry.grid)} pts)", entry.name)
            if entry.name == preserve_name:
                select_index = i
            i += 1
        combo.setCurrentIndex(select_index)
        combo.blockSignals(False)

    def _on_role_quantity_changed(self, role: str):
        quantity_filter = self._role_quantity_combos[role].currentData()
        for row in self._test_rows:
            self._populate_library_combo(row["var_combos"][role], quantity_filter)

    def _on_output_quantity_changed(self, _index=None):
        quantity_filter = self._output_quantity_combo.currentData()
        for row in self._test_rows:
            self._populate_library_combo(row["output_combo"], quantity_filter)

    def _on_add_test_row(self):
        if self._test_grid is None:
            self._rebuild_test_table()
            return
        used_vars = self._parsed.used_vars if self._parsed else ()
        self._test_row_counter += 1
        # Grid row 0 is the header, row 1 is the "Variable" quantity
        # picker, row 2 is the "Start at zero" picker; rows are only ever
        # appended after that (removal just hides/deletes a row's widgets
        # in place, see _on_remove_test_row, never reuses/renumbers a
        # grid row), so the next free row is 3 + however many test rows
        # exist so far.
        row_index = len(self._test_rows) + 3

        col = 0
        label_edit = QLineEdit(f"Test{self._test_row_counter}")
        label_edit.setMaximumWidth(110)
        self._test_grid.addWidget(label_edit, row_index, col)
        col += 1

        var_combos = {}
        for role in used_vars:
            quantity_filter = self._role_quantity_combos[role].currentData()
            combo = self._make_library_combo(quantity_filter)
            self._test_grid.addWidget(combo, row_index, col)
            var_combos[role] = combo
            col += 1

        output_filter = self._output_quantity_combo.currentData()
        output_combo = self._make_library_combo(output_filter)
        self._test_grid.addWidget(output_combo, row_index, col)
        col += 1

        validation_checkbox = QCheckBox()
        self._test_grid.addWidget(validation_checkbox, row_index, col)
        col += 1

        remove_button = QPushButton("Remove")
        self._test_grid.addWidget(remove_button, row_index, col)

        row = dict(
            label_edit=label_edit, var_combos=var_combos, output_combo=output_combo,
            validation_checkbox=validation_checkbox, remove_button=remove_button,
        )
        remove_button.clicked.connect(lambda: self._on_remove_test_row(row))
        self._test_rows.append(row)

    def _on_remove_test_row(self, row: dict):
        if row not in self._test_rows:
            return
        self._test_rows.remove(row)
        for widget in (row["label_edit"], *row["var_combos"].values(), row["output_combo"],
                       row["validation_checkbox"], row["remove_button"]):
            widget.setVisible(False)
            widget.deleteLater()

    def _refresh_library_combos(self):
        """Re-populates every test row's combos from the current
        `project.covariance_library` (e.g. after saving a new entry on
        the Covariance Matrix tab), respecting each column's own
        "Variable" quantity filter and preserving each combo's current
        selection by entry NAME when that entry still passes it."""
        for row in self._test_rows:
            for role, combo in row["var_combos"].items():
                quantity_filter = self._role_quantity_combos[role].currentData()
                self._populate_library_combo(combo, quantity_filter)
            output_filter = self._output_quantity_combo.currentData() if self._output_quantity_combo else None
            self._populate_library_combo(row["output_combo"], output_filter)

    def _build_test_assignments(self) -> list:
        """Reads every (non-removed) test row back into the plain-dict
        shape core.calib_run.run_calibration expects, resolving each
        combo's selected entry NAME against the current library. Raises
        CalibRunError with a human-readable message for any row missing
        a required assignment."""
        by_name = entries_by_name(self.project.covariance_library)
        assignments = []
        for row in self._test_rows:
            label = row["label_edit"].text().strip() or "Test"
            variable_entries = {}
            for role, combo in row["var_combos"].items():
                entry_name = combo.currentData()
                if entry_name is None:
                    raise CalibRunError(f"Test '{label}': assign a library entry for '{role}' first.")
                if entry_name not in by_name:
                    raise CalibRunError(f"Test '{label}': its '{role}' entry ('{entry_name}') no longer exists.")
                variable_entries[role] = by_name[entry_name]
            output_name = row["output_combo"].currentData()
            if output_name is None:
                raise CalibRunError(f"Test '{label}': assign an Output entry first.")
            if output_name not in by_name:
                raise CalibRunError(f"Test '{label}': its Output entry ('{output_name}') no longer exists.")
            assignments.append(dict(
                label=label, variable_entries=variable_entries, output_entry=by_name[output_name],
                is_validation=row["validation_checkbox"].isChecked(),
            ))
        return assignments

    # -- Calibration project (save/load) ---------------------------------------

    def _collect_test_rows_for_export(self) -> list:
        """Reads every test row back into the plain-dict shape core.
        calib_project_io.export_calibration_project expects -- entry
        NAMES as plain strings (an unassigned combo's currentData() is
        None, saved as "" rather than raising, since a work-in-progress
        setup is a legitimate thing to save; contrast with
        `_build_test_assignments`, which resolves entries and requires
        every one be assigned, for an actual Run)."""
        rows = []
        for row in self._test_rows:
            rows.append(dict(
                label=row["label_edit"].text(),
                variable_entries={role: combo.currentData() or "" for role, combo in row["var_combos"].items()},
                output_entry=row["output_combo"].currentData() or "",
                is_validation=row["validation_checkbox"].isChecked(),
            ))
        return rows

    def _select_combo_by_data(self, combo: QComboBox, value) -> bool:
        """Selects the item whose `itemData` equals `value`, returning
        whether one was found -- shared by every "restore a combo
        selection from a loaded file" path below (quantity pickers,
        covariance mode, init method, and each test row's own library-
        entry combos)."""
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return True
        return False

    def _apply_prior_to_widgets(self, name: str, prior: ParamPrior):
        """Sets every field of a free-parameter row's widgets from a
        loaded ParamPrior -- all of them, not just the ones `prior.kind`
        itself uses, matching ParamPrior's own "every field always
        present" convention (core/calib_priors.py) so switching the kind
        combo afterward shows sensible values in every group."""
        widgets = self._prior_widgets.get(name)
        if widgets is None:
            return
        widgets["combo"].setCurrentIndex(PRIOR_KINDS.index(prior.kind))
        widgets["fixed"]["edit"].setText(_fmt(prior.fixed_value))
        widgets["uniform"]["lo"].setText(_fmt(prior.uniform_lo))
        widgets["uniform"]["hi"].setText(_fmt(prior.uniform_hi))
        widgets["hier"]["mu0"].setText(_fmt(prior.hier_mu0))
        widgets["hier"]["s0"].setText(_fmt(prior.hier_s0))
        widgets["hier"]["nu0"].setText(_fmt(prior.hier_nu0))

    def _apply_test_row_data(self, row: dict, test_data: dict) -> list:
        """Restores one test row's widgets from a loaded test dict
        (core.calib_project_io.CalibProjectData.tests's own shape).
        Returns a list of "role=entry_name" strings for any assignment
        that named a library entry NOT found in the current
        `project.covariance_library` -- left at "-- choose --" rather
        than guessed, so the caller can report exactly which ones need
        picking manually (or loading the matching calibration library
        first)."""
        row["label_edit"].setText(test_data.get("label", ""))
        unresolved = []
        variable_entries = test_data.get("variable_entries", {})
        for role, combo in row["var_combos"].items():
            entry_name = variable_entries.get(role, "")
            if entry_name and not self._select_combo_by_data(combo, entry_name):
                unresolved.append(f"{role}={entry_name}")
        output_name = test_data.get("output_entry", "")
        if output_name and not self._select_combo_by_data(row["output_combo"], output_name):
            unresolved.append(f"Output={output_name}")
        row["validation_checkbox"].setChecked(bool(test_data.get("is_validation", False)))
        return unresolved

    def _on_save_calibration_project(self):
        if self._parsed is None:
            self.project_io_status_label.setText("Parse the equation first (Equation card).")
            return
        try:
            priors = self._read_priors()
        except (ValueError, PriorError) as exc:
            self.project_io_status_label.setText(
                f"Free parameters: {exc}" if isinstance(exc, PriorError) else
                "Free parameters: every field must be numeric (and, for Uniform, max > min; "
                "for Normal, s0 > 0 and nu0 > 0)."
            )
            return
        try:
            sampler_settings = self._read_sampler_settings()
        except ValueError:
            self.project_io_status_label.setText(
                "Sampler settings: Draws/Tune/Chains/Cores must be whole numbers, and Target accept "
                "and Covariance jitter must be numeric."
            )
            return

        role_quantities = {role: combo.currentData() for role, combo in self._role_quantity_combos.items()}
        output_quantity = self._output_quantity_combo.currentData() if self._output_quantity_combo else None
        covariance_mode = self.covariance_mode_combo.currentData()
        zero_start_vars = [role for role, cb in self._zero_start_checkboxes.items() if cb.isChecked()]
        zero_start_output = bool(
            self._output_zero_start_checkbox and self._output_zero_start_checkbox.isChecked()
        )
        tests = self._collect_test_rows_for_export()

        default_name = "bayes_calibration_project.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save calibration project", default_name, "CSV files (*.csv)")
        if not path:
            return
        try:
            json_path = export_calibration_project(
                path, equation=self._parsed.source, priors=priors,
                role_quantities=role_quantities, output_quantity=output_quantity,
                covariance_mode=covariance_mode, sampler_settings=sampler_settings,
                tests=tests, used_vars=self._parsed.used_vars,
                zero_start_vars=zero_start_vars, zero_start_output=zero_start_output,
            )
        except OSError as exc:
            self.project_io_status_label.setText(f"Could not save: {exc}")
            return
        self.project_io_status_label.setText(
            f"Saved calibration project: {os.path.basename(path)} (tests) + "
            f"{os.path.basename(json_path)} (equation/priors/settings)."
        )

    def _on_load_calibration_project(self):
        json_path, _ = QFileDialog.getOpenFileName(self, "Load calibration project", "", "JSON files (*.json)")
        if not json_path:
            return
        try:
            data = import_calibration_project(json_path)
        except (OSError, CalibProjectIOError) as exc:
            self.project_io_status_label.setText(f"Could not load: {exc}")
            return

        # A user reported a graph that was mathematically impossible given
        # the just-loaded priors (e.g. a "Per-test calibration fit" curve
        # sitting near 0 despite every pooled parameter's own Uniform lower
        # bound being well above 0) -- root cause: loading a project only
        # ever updated the INPUT widgets (equation/priors/tests/settings);
        # it never cleared a previous Run's own RESULT (`_last_result`/
        # `_posterior_flat`), so every result-dependent card (Result
        # summary, Per-test fit, Interdependence, Diagnostics, Validation)
        # kept showing whatever was last actually sampled -- silently stale
        # and no longer matching the equation/priors now on screen -- until
        # Run was clicked again. Clearing them here up front removes that
        # trap: after a load, those cards go back to their own "no run yet"
        # state, so nothing on screen can be mistaken for this project's
        # own result until it has actually been run.
        self._last_result = None
        self._posterior_flat = {}
        self._refresh_summary()
        self._refresh_fit_card()
        self._refresh_interdependence()
        self._refresh_diag_card()
        self._refresh_validation_card()
        self.export_card.setVisible(False)

        self.equation_edit.setText(data.equation)
        self._on_parse()
        if self._parsed is None:
            self.project_io_status_label.setText(
                f"Loaded {os.path.basename(json_path)}, but its equation failed to parse: "
                f"{self.equation_status_label.text()}"
            )
            return

        for name in self._parsed.free_params:
            if name in data.priors:
                self._apply_prior_to_widgets(name, data.priors[name])

        for role, combo in self._role_quantity_combos.items():
            if role in data.role_quantities:
                self._select_combo_by_data(combo, data.role_quantities[role])
        if self._output_quantity_combo is not None:
            self._select_combo_by_data(self._output_quantity_combo, data.output_quantity)
        self._refresh_library_combos()  # re-filter every row's entry combos for the just-restored quantity pickers

        self._select_combo_by_data(self.covariance_mode_combo, data.covariance_mode)
        for role, cb in self._zero_start_checkboxes.items():
            cb.setChecked(role in data.zero_start_vars)
        if self._output_zero_start_checkbox is not None:
            self._output_zero_start_checkbox.setChecked(bool(data.zero_start_output))
        settings = data.sampler_settings
        self.draws_edit.setText(str(settings.get("draws", DEFAULT_DRAWS)))
        self.tune_edit.setText(str(settings.get("tune", DEFAULT_TUNE)))
        self.chains_edit.setText(str(settings.get("chains", DEFAULT_CHAINS)))
        self.cores_edit.setText(str(settings.get("cores", self.cores_edit.text())))
        self.target_accept_edit.setText(str(settings.get("target_accept", DEFAULT_TARGET_ACCEPT)))
        self.covariance_jitter_edit.setText(str(settings.get("covariance_jitter", DEFAULT_COVARIANCE_JITTER)))
        if "init" in settings:
            self._select_combo_by_data(self.init_combo, settings["init"])

        # Rebuild the test table from the loaded rows -- _on_parse() above
        # already created one fresh default row; drop it (unless there's
        # nothing to restore, in which case leave that one default row).
        if data.tests:
            for row in list(self._test_rows):
                self._on_remove_test_row(row)
            unresolved_by_test = []
            for test_data in data.tests:
                self._on_add_test_row()
                row = self._test_rows[-1]
                unresolved = self._apply_test_row_data(row, test_data)
                if unresolved:
                    unresolved_by_test.append(f"{test_data.get('label', '?')} ({', '.join(unresolved)})")
            if unresolved_by_test:
                self.project_io_status_label.setText(
                    f"Loaded {os.path.basename(json_path)}. Some assignments were not found in the "
                    f"current calibration library and are left unassigned -- pick them manually, or "
                    f"load the matching library first: {'; '.join(unresolved_by_test)}."
                )
                return

        self.project_io_status_label.setText(f"Loaded calibration project from {os.path.basename(json_path)}.")

    # -- Card 4: Sampler settings ---------------------------------------------

    def _refresh_pymc_status(self):
        if pymc_available():
            self.pymc_status_label.setText("")
            self.run_button.setEnabled(True)
        else:
            self.pymc_status_label.setText(
                "PyMC / PyTensor are not installed in this environment -- Run is disabled. "
                "Install them (`pip install pymc pytensor arviz`, see requirements.txt) to enable "
                "Bayesian Calibration."
            )
            self.run_button.setEnabled(False)

    def _read_sampler_settings(self) -> dict:
        return dict(
            draws=int(self.draws_edit.text()),
            tune=int(self.tune_edit.text()),
            chains=int(self.chains_edit.text()),
            cores=int(self.cores_edit.text()),
            target_accept=float(self.target_accept_edit.text()),
            init=self.init_combo.currentData(),
            covariance_jitter=float(self.covariance_jitter_edit.text()),
        )

    # -- Card 5: Run -----------------------------------------------------------

    def _on_run(self):
        if self._parsed is None:
            self.run_status_label.setText("Parse the equation first (Equation card).")
            return
        try:
            priors = self._read_priors()
        except (ValueError, PriorError) as exc:
            self.run_status_label.setText(f"Free parameters: {exc}" if isinstance(exc, PriorError) else
                                           "Free parameters: every field must be numeric (and, for "
                                           "Uniform, max > min; for Normal, s0 > 0 and nu0 > 0).")
            return
        try:
            test_assignments = self._build_test_assignments()
        except CalibRunError as exc:
            self.run_status_label.setText(str(exc))
            return
        try:
            sampler_settings = self._read_sampler_settings()
        except ValueError:
            self.run_status_label.setText("Sampler settings: Draws/Tune/Chains/Cores must be whole numbers, "
                                           "and Target accept and Covariance jitter must be numeric.")
            return

        covariance_mode = self.covariance_mode_combo.currentData()
        zero_start_vars = {role for role, cb in self._zero_start_checkboxes.items() if cb.isChecked()}
        zero_start_output = bool(
            self._output_zero_start_checkbox and self._output_zero_start_checkbox.isChecked()
        )
        equation_str = self.equation_edit.text()

        self.run_button.setEnabled(False)
        self._cancel_event = threading.Event()
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.run_status_label.setText("Running…")

        self._thread = QThread()
        self._worker = CalibWorker(
            run_calibration,
            dict(
                equation_str=equation_str, priors=priors, test_assignments=test_assignments,
                covariance_mode=covariance_mode, sampler_settings=sampler_settings,
                zero_start_vars=zero_start_vars, zero_start_output=zero_start_output,
            ),
            cancel_event=self._cancel_event,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._thread.start()

    def _on_stop(self):
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.stop_button.setEnabled(False)
            self.run_status_label.setText(
                "Stopping… (only takes effect before sampling actually starts -- see the Stop "
                "button's tooltip)."
            )

    def _on_progress(self, i, total):
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(i)

    def _on_finished(self, result):
        self._last_result = result
        self._cancel_event = None
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.run_status_label.setText("Calibration finished.")
        self._posterior_flat = self._flatten_posterior_vars(result.idata)
        self._refresh_summary()
        self._refresh_fit_card()
        self._refresh_interdependence()
        self._refresh_diag_card()
        self._refresh_validation_card()
        self.export_card.setVisible(True)
        self.export_status_label.setText("")

    def _on_failed(self, message):
        self._cancel_event = None
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.run_status_label.setText(f"Run failed: {message}")

    def _on_cancelled(self):
        self._cancel_event = None
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.run_status_label.setText("Calibration stopped by user.")

    # -- Card 6: Result summary -----------------------------------------------

    def _refresh_summary(self):
        result = self._last_result
        if result is None:
            self.summary_label.setText("No calibration run yet.")
            return

        lines = []
        for name, value in sorted(result.fixed_param_values.items()):
            lines.append(f"{name} (fixed): {_fmt(value)}")

        for name in result.pooled_param_names:
            stats = result.posterior_summary.get(name, {})
            mean, sd, rhat = stats.get("mean"), stats.get("sd"), stats.get("r_hat")
            flag = ""
            if rhat is not None:
                try:
                    if float(rhat) > 1.01:
                        flag = "  [R-hat > 1.01 -- not converged]"
                except (TypeError, ValueError):
                    pass
            rhat_text = f", R-hat={_fmt_stat(rhat)}" if rhat is not None else ""
            lines.append(f"{name} (pooled): {_fmt_stat(mean)} ± {_fmt_stat(sd)}{rhat_text}{flag}")

        for name in result.hier_param_names:
            mu_stats = result.posterior_summary.get(f"mu_{name}", {})
            mean, sd, rhat = mu_stats.get("mean"), mu_stats.get("sd"), mu_stats.get("r_hat")
            flag = ""
            if rhat is not None:
                try:
                    if float(rhat) > 1.01:
                        flag = "  [R-hat > 1.01 -- not converged]"
                except (TypeError, ValueError):
                    pass
            rhat_text = f", R-hat={_fmt_stat(rhat)}" if rhat is not None else ""
            lines.append(
                f"{name} (hierarchical, hyper-mean): {_fmt_stat(mean)} ± {_fmt_stat(sd)}{rhat_text}{flag}"
            )

        zero_start_bits = list(result.zero_start_vars)
        if result.zero_start_output:
            zero_start_bits.append("Output")
        zero_start_text = f" Translated to start at zero: {', '.join(zero_start_bits)}." if zero_start_bits else ""
        lines.append(
            f"\n{len(result.calibration_tests)} calibration test(s), "
            f"{len(result.validation_tests)} validation-only test(s). "
            f"Covariance mode: {COVARIANCE_MODE_LABELS.get(result.covariance_mode, result.covariance_mode)}."
            f"{zero_start_text}"
        )

        # A covariance matrix built from too few Monte-Carlo draws relative
        # to its own grid-point count can be numerically ill-conditioned
        # enough to make the exact likelihood fit an almost-arbitrary
        # near-null direction instead of the data's visible shape -- see
        # core.calib_library_adapter.covariance_condition_number's docstring
        # (a real case measured ~4.6e7). Flagged here, per test, so this is
        # visible BEFORE a user spends time puzzling over a nonsensical fit.
        ill_conditioned = {
            label: cond for label, cond in result.covariance_condition_numbers.items()
            if cond > _ILL_CONDITIONED_THRESHOLD
        }
        if ill_conditioned:
            jitter = result.sampler_settings.get("covariance_jitter", 0.0)
            detail = ", ".join(f"{label} (condition number {cond:.2e})" for label, cond in ill_conditioned.items())
            jitter_note = (
                f" A covariance jitter of {jitter:g} was applied for this run."
                if jitter else
                " No covariance jitter was applied for this run (Sampler settings) -- try 0.01-0.1."
            )
            lines.append(
                f"\n⚠ Severely ill-conditioned covariance in: {detail}. This can make the fit "
                f"chase numerical noise instead of the data's real shape, even with excellent "
                f"R-hat (it can converge confidently onto a wrong answer).{jitter_note}"
            )

        # A test's own REAL measured data can have a handful of points
        # outside a function's valid domain (e.g. 1-2 slightly-negative
        # strain-rate points from ordinary measurement noise feeding
        # log(v) or v**m) -- core.calib_equation clamps these instead of
        # crashing or silently producing NaN (see its "Domain guards"
        # docstring section), but that clamp is still worth surfacing:
        # it's real data being locally overridden, not a normal part of
        # the fit. This is a best-effort, pre-sampling check (see core.
        # calib_run._domain_guard_report_for_tests's own docstring), so
        # an empty warning here means "nothing detected", not an
        # ironclad guarantee nothing was ever clamped during sampling.
        if result.domain_guard_warnings:
            for label, report in result.domain_guard_warnings.items():
                detail = ", ".join(
                    f"{_DOMAIN_GUARD_LABELS.get(name, name)} ({count} point(s))"
                    for name, count in report.items()
                )
                lines.append(
                    f"\n⚠ {label}: {detail} -- these were clamped to a tiny positive floor "
                    f"before evaluation (using representative parameter values) so the "
                    f"calculation doesn't crash. Check this test's data near those points, or "
                    f"confirm this is expected measurement noise near zero."
                )

        self.summary_label.setText("\n".join(lines))

    # -- shared: flatten posterior draws --------------------------------------

    def _flatten_posterior_vars(self, idata) -> dict:
        """{arviz variable name: (n_samples, ...) np.ndarray}, chain and
        draw dims collapsed into one -- the shape core.calib_diagnostics'
        resolve_predictive_param_draws/posterior_correlation_matrix both
        expect. Defensive against `idata` not actually being a real
        arviz.InferenceData (e.g. a bare `object()` from a test double
        that never reached real sampling) -- returns {} rather than
        raising, since every card below already handles an empty
        self._posterior_flat by showing a plain status message."""
        flat = {}
        try:
            data_vars = list(idata.posterior.data_vars)
        except AttributeError:
            return flat
        for name in data_vars:
            arr = np.asarray(idata.posterior[name].values)
            flat[name] = arr.reshape((-1,) + arr.shape[2:]) if arr.ndim >= 2 else arr.reshape(-1)
        return flat

    # -- Card 7: Per-test calibration fit -------------------------------------

    def _refresh_fit_card(self):
        result = self._last_result
        if result is None or not result.calibration_tests:
            self.fit_card.setVisible(False)
            return
        self.fit_test_combo.blockSignals(True)
        self.fit_test_combo.clear()
        for i, test in enumerate(result.calibration_tests):
            self.fit_test_combo.addItem(test.label, i)
        self.fit_test_combo.blockSignals(False)
        self.fit_card.setVisible(True)
        self._on_fit_test_changed()

    def _on_fit_test_changed(self, *_args):
        result = self._last_result
        if result is None or not result.calibration_tests:
            return
        index = self.fit_test_combo.currentData()
        if index is None:
            return
        dataset = result.calibration_tests[index]
        try:
            param_draws = resolve_predictive_param_draws(
                result.free_params, result.priors, self._posterior_flat, test_index=index
            )
            prediction = posterior_predictive_at(parse_equation(result.equation_source), param_draws, dataset)
        except CalibDiagnosticsError as exc:
            self.fit_canvas.clear()
            self.fit_canvas.refresh()
            self.fit_status_label.setText(str(exc))
            return

        self.fit_canvas.clear()
        ax = self.fit_canvas.figure.add_subplot(111)
        observed_lower, observed_upper = _observed_band(dataset)
        ax.fill_between(dataset.grid, observed_lower, observed_upper,
                         color=theme.TEXT_MUTED, alpha=0.25, label="Observed 95% band")
        ax.plot(dataset.grid, dataset.observed, color=theme.ACCENT, label="Observed")
        ax.fill_between(prediction["grid"], prediction["lower"], prediction["upper"],
                         color=theme.SUCCESS, alpha=0.2, label="Posterior-predictive 95% band")
        ax.plot(prediction["grid"], prediction["mean"], color=theme.TEXT, linestyle="--",
                label="Posterior-predictive mean")
        ax.set_xlabel("Grid")
        ax.set_ylabel(dataset.output_entry_name or "Output")
        ax.set_title(dataset.label)
        ax.legend(fontsize=8)
        self.fit_canvas.refresh()
        self.fit_status_label.setText("")

    # -- Card 8: Parameter interdependence ------------------------------------

    def _refresh_interdependence(self):
        result = self._last_result
        if result is None:
            self.interdep_card.setVisible(False)
            return

        posterior_for_corr = {}
        for name in result.pooled_param_names:
            if name in self._posterior_flat:
                posterior_for_corr[name] = self._posterior_flat[name]
        for name in result.hier_param_names:
            mu_name = f"mu_{name}"
            if mu_name in self._posterior_flat:
                posterior_for_corr[name] = self._posterior_flat[mu_name]  # labeled by the bare param name

        self.interdep_card.setVisible(True)
        if len(posterior_for_corr) < 2:
            self.interdep_canvas.clear()
            self.interdep_canvas.refresh()
            self.interdep_status_label.setText(
                "Need at least 2 calibrated (non-fixed) parameters to show this."
            )
            return
        try:
            names, matrix = posterior_correlation_matrix(posterior_for_corr)
        except CalibDiagnosticsError as exc:
            self.interdep_status_label.setText(str(exc))
            return

        self.interdep_canvas.clear()
        ax = self.interdep_canvas.figure.add_subplot(111)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        n = len(names)
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(names)
        for i in range(n):
            for j in range(n):
                value = matrix[i, j]
                text_color = "white" if abs(value) > 0.6 else theme.TEXT
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)
        cbar = self.interdep_canvas.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Posterior correlation")
        self.interdep_canvas.refresh()
        self.interdep_status_label.setText("")

    # -- Card 9: Posterior diagnostics ----------------------------------------

    def _refresh_diag_card(self):
        if self._last_result is None:
            self.diag_card.setVisible(False)
            return
        self.diag_card.setVisible(True)
        self._on_diag_kind_changed()

    def _on_diag_kind_changed(self, *_args):
        """Switches the Posterior diagnostics canvas between density/
        forest/energy. Everything after `self.diag_canvas.clear()` is
        wrapped in one broad try/except/finally: an earlier version of
        this method (and of forest/energy's own implementation, which
        called `az.plot_forest`/`az.plot_energy` directly -- see core.
        calib_diagnostics's own module comment on why that changed) only
        caught one specific exception type around the render call and
        only refreshed the canvas on the success path, so any OTHER
        exception left the canvas showing whatever was already on screen
        from the PREVIOUS "Show" selection -- a real bug report described
        this as the card getting "stuck" on Posterior density no matter
        what was picked. Now every path, success or failure, ends by
        clearing again (dropping any half-drawn axes) and always calling
        refresh(), so the canvas always reflects the CURRENT selection,
        never a stale one -- whatever goes wrong is reported in the
        status label instead."""
        result = self._last_result
        if result is None:
            return
        kind = self.diag_kind_combo.currentData()
        self.diag_canvas.clear()
        try:
            if kind == "density":
                self._render_posterior_density(result)
            elif kind == "forest":
                self._render_forest_plot(result)
            elif kind == "energy":
                self._render_energy_plot(result)
        except Exception as exc:  # noqa: BLE001 -- any rendering failure must still clear the stale plot and report itself (see this method's own docstring)
            self.diag_canvas.clear()
            self.diag_status_label.setText(f"Could not draw {self.diag_kind_combo.currentText()}: {exc}")
        finally:
            self.diag_canvas.refresh()

    def _render_posterior_density(self, result):
        names = list(result.pooled_param_names) + list(result.hier_param_names)
        if not names:
            self.diag_status_label.setText("No calibrated (non-fixed) parameters to show.")
            return
        fig = self.diag_canvas.figure
        axes = fig.subplots(1, len(names), squeeze=False)[0]
        any_drawn = False
        for ax, name in zip(axes, names):
            key = name if name in result.pooled_param_names else f"mu_{name}"
            draws = self._posterior_flat.get(key)
            if draws is None:
                ax.set_visible(False)
                continue
            ax.hist(draws, bins=30, color=theme.ACCENT, alpha=0.85)
            ax.set_title(name, fontsize=9)
            ax.set_yticks([])
            any_drawn = True
        self.diag_status_label.setText("" if any_drawn else "No posterior draws available to plot.")

    def _render_forest_plot(self, result):
        """Built from `result.posterior_summary` (already computed by
        `run_calibration`, via `az.summary` -- see core.calib_diagnostics.
        forest_plot_rows's own docstring for why this no longer calls
        `az.plot_forest` directly) rather than the idata itself, so
        nothing here depends on arviz's plotting API at all.

        For a hierarchical (per-test) parameter, this shows BOTH its
        hyper-mean ("mu_<name>" -- the same value the Result summary card
        labels "hyper-mean") AND every individual test's own sampled
        value ("<name>[<test index>]" -- arviz's own default row-naming
        for a shape=(n_tests,) pymc variable declared with no explicit
        dims/coords, which is exactly how core.calib_pymc_model.build_
        model declares it). A user reported a Per-test-fit curve that
        made no sense given the hyper-mean shown in Result summary --
        without this, there was no way to see whether the INDIVIDUAL
        per-test value actually used by that fit (core.calib_diagnostics.
        resolve_predictive_param_draws picks `<name>[test_index]`, not
        the hyper-mean) had drifted away from it. If a name isn't found
        in `posterior_summary` (e.g. arviz's exact bracket/index naming
        ever changes), `forest_plot_rows` silently skips it rather than
        erroring -- this is purely additive, never a reason the plot
        would stop rendering the parameters it already showed."""
        per_test_display = {}  # {arviz row name: human label, e.g. "A[0]" -> "A (Test1)"}
        var_names = list(result.pooled_param_names)
        for name in result.hier_param_names:
            var_names.append(f"mu_{name}")
            for j, test in enumerate(result.calibration_tests):
                arviz_name = f"{name}[{j}]"
                var_names.append(arviz_name)
                per_test_display[arviz_name] = f"{name} ({test.label})"
        rows = forest_plot_rows(result.posterior_summary, var_names)
        if not rows:
            self.diag_status_label.setText("No calibrated (non-fixed) parameters to show.")
            return
        ax = self.diag_canvas.figure.add_subplot(111)
        y_positions = list(range(len(rows)))
        means = [r["mean"] for r in rows]
        lo_err = [r["mean"] - r["lo"] for r in rows]
        hi_err = [r["hi"] - r["mean"] for r in rows]
        ax.errorbar(
            means, y_positions, xerr=[lo_err, hi_err], fmt="o",
            color=theme.ACCENT, ecolor=theme.ACCENT, capsize=4, markersize=6,
        )
        labels = []
        for r in rows:
            display_name = per_test_display.get(r["name"], r["name"])
            try:
                rhat_val = float(r["r_hat"]) if r["r_hat"] is not None else None
            except (TypeError, ValueError):
                rhat_val = None
            rhat_text = f"  (R-hat={rhat_val:.3g})" if rhat_val is not None else ""
            flag = "  [not converged]" if (rhat_val is not None and rhat_val > 1.01) else ""
            labels.append(f"{display_name}{rhat_text}{flag}")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # first parameter at the top, matching a conventional forest plot
        ax.axvline(0, color=theme.BORDER, linewidth=0.8, linestyle=":")
        ax.set_xlabel("Value (point estimate, 94% HDI)")
        self.diag_status_label.setText("")

    def _render_energy_plot(self, result):
        """Built directly from `idata.sample_stats["energy"]` (see core.
        calib_diagnostics.energy_diagnostic_arrays's own docstring for
        why this no longer calls `az.plot_energy` directly)."""
        try:
            arrays = energy_diagnostic_arrays(result.idata)
        except CalibDiagnosticsError as exc:
            self.diag_status_label.setText(str(exc))
            return
        ax = self.diag_canvas.figure.add_subplot(111)
        ax.hist(arrays["marginal"], bins=30, alpha=0.5, density=True, color=theme.ACCENT, label="Marginal energy")
        ax.hist(arrays["transition"], bins=30, alpha=0.5, density=True, color=theme.SUCCESS, label="Energy transition")
        ax.set_xlabel("Energy (centered)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        self.diag_status_label.setText("")

    # -- Card 10: Validation ---------------------------------------------------

    def _refresh_validation_card(self):
        result = self._last_result
        if result is None or not result.validation_tests:
            self.validation_card.setVisible(False)
            return
        self.validation_test_combo.blockSignals(True)
        self.validation_test_combo.clear()
        for i, test in enumerate(result.validation_tests):
            self.validation_test_combo.addItem(test.label, i)
        self.validation_test_combo.blockSignals(False)
        self.validation_card.setVisible(True)
        self._on_validation_test_changed()

    def _on_validation_test_changed(self, *_args):
        result = self._last_result
        if result is None or not result.validation_tests:
            return
        index = self.validation_test_combo.currentData()
        if index is None:
            return
        dataset = result.validation_tests[index]
        try:
            # test_index=None: this test was never part of the fit, so a
            # hierarchical parameter has no per-test value here -- see
            # resolve_predictive_param_draws's own docstring for the
            # hyper-mean fallback this implies (also spelled out in this
            # card's help text).
            param_draws = resolve_predictive_param_draws(
                result.free_params, result.priors, self._posterior_flat, test_index=None
            )
            prediction = posterior_predictive_at(parse_equation(result.equation_source), param_draws, dataset)
        except CalibDiagnosticsError as exc:
            self.validation_canvas.clear()
            self.validation_canvas.refresh()
            self.validation_status_label.setText(str(exc))
            return

        self.validation_canvas.clear()
        ax = self.validation_canvas.figure.add_subplot(111)
        observed_lower, observed_upper = _observed_band(dataset)
        ax.fill_between(dataset.grid, observed_lower, observed_upper,
                         color=theme.TEXT_MUTED, alpha=0.25, label="Observed 95% band")
        ax.plot(dataset.grid, dataset.observed, color=theme.ACCENT, label="Observed (held out)")
        ax.fill_between(prediction["grid"], prediction["lower"], prediction["upper"],
                         color=theme.SUCCESS, alpha=0.2, label="Posterior-predictive 95% band")
        ax.plot(prediction["grid"], prediction["mean"], color=theme.TEXT, linestyle="--",
                label="Posterior-predictive mean")
        ax.set_title(dataset.label)
        ax.legend(fontsize=8)
        self.validation_canvas.refresh()
        self.validation_status_label.setText("")

    # -- Card 11: Export ---------------------------------------------------------

    def _populated_canvases(self) -> list:
        """[(file-name-safe key, PlotCanvas)] for every canvas that
        currently has something worth exporting -- split out from
        `_on_export_figures` so a test can drive the export logic
        directly against a temp folder without needing a real
        QFileDialog (the stub's `getExistingDirectory` always returns
        "", i.e. "user cancelled")."""
        result = self._last_result
        canvases = []
        if result is None:
            return canvases
        canvases.append(("per_test_calibration_fit", self.fit_canvas))
        if len(result.pooled_param_names) + len(result.hier_param_names) >= 2:
            canvases.append(("parameter_interdependence", self.interdep_canvas))
        canvases.append(("posterior_diagnostics", self.diag_canvas))
        if result.validation_tests:
            canvases.append(("validation", self.validation_canvas))
        return canvases

    def _on_export_figures(self):
        if self._last_result is None:
            self.export_status_label.setText("Run a calibration first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select the export folder")
        if not folder:
            return
        try:
            n = self._export_figures_to_folder(folder)
        except OSError as exc:
            self.export_status_label.setText(f"Could not export: {exc}")
            return
        self.export_status_label.setText(f"Exported {n} figure(s) to {folder}")

    def _export_figures_to_folder(self, folder: str) -> int:
        n = 0
        for name, canvas in self._populated_canvases():
            path = os.path.join(folder, f"bayes_calib_{name}.pdf")
            canvas.figure.savefig(path)
            n += 1
        return n

    def _on_export_summary(self, fmt: str):
        if self._last_result is None:
            self.export_status_label.setText("Run a calibration first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select the export folder")
        if not folder:
            return
        path = os.path.join(folder, f"posterior_summary.{fmt}")
        try:
            self._export_summary_to_file(path, fmt)
        except OSError as exc:
            self.export_status_label.setText(f"Could not export: {exc}")
            return
        self.export_status_label.setText(f"Exported posterior summary to {path}")

    def _export_summary_to_file(self, path: str, fmt: str):
        result = self._last_result
        stat_keys = []
        for stats in result.posterior_summary.values():
            for key in stats.keys():
                if key not in stat_keys:
                    stat_keys.append(key)
        header = ["variable"] + stat_keys
        rows = [[name] + [stats.get(key, "") for key in stat_keys]
                for name, stats in result.posterior_summary.items()]

        if fmt == "csv":
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
        else:
            with open(path, "w") as f:
                f.write("\t".join(header) + "\n")
                for row in rows:
                    f.write("\t".join(str(v) for v in row) + "\n")
