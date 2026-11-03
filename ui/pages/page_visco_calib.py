"""
Viscoelastic Model Calibration page.

Replaces calib_visco_model.py's Toplevel. Determines a viscoelastic bar
model's parameters (Poisson ratio plus whichever moduli/viscosities the
selected rheological model uses -- see core/rheology_models.py) from two
free-decay pulses recorded at two points along the same bar, either by
Bayesian optimization ("Bayesian calibration", core/bayes_visco_
calibration.py) or by testing one fixed candidate set ("Test parameters").
A found/tested Standard Linear Solid set can be pushed directly to the
Dispersion Corrections page with one click -- that page's physics only
understands that one model (see core/rheology_models.py's module
docstring), so the button is disabled for the other three.
"""
from __future__ import annotations

import threading

import numpy as np
from core.downsampling import downsample
from PySide6.QtCore import QThread, Signal, QObject
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QLineEdit, QPushButton, QProgressBar, QLabel, QHBoxLayout,
    QGridLayout, QButtonGroup, QCheckBox,
)

from core.visco_calibration import test_parameters, SCORE_LABELS, SCORE_SHORT_LABELS, SCORE_ORDER, DEFAULT_METRIC
from core.bayes_visco_calibration import (
    calibrate_bayesian, ViscoBayesianResult, PriorSpec,
    ACQ_EI, ACQ_PI, ACQ_UCB, ACQ_THOMPSON, ACQUISITION_LABELS, ACQUISITION_USE_CASES,
)
from core.parallel import OperationCancelled
from core.qt_worker import CalibWorker as _CalibWorker
from core.rheology_models import MODELS, MODEL_ORDER, MODEL_SLS, get_model
from core.pulse_alignment import align_pulse2
from ui.widgets import Card, LabeledField, FileSelectRow, make_scroll_page, TimeSourceRow
from ui.span_picker import TimeWindowPicker
from ui.plot_canvas import PlotCanvas
from ui.rheology_diagram import RheologyModelButton
from ui import theme

_MODE_BAYESIAN, _MODE_FIXED = 0, 1

_NU_KEY = "nu"
_NU_LABEL = "Poisson ratio (-)"
_NU_BAYES_BOUNDS = (0.25, 0.45)
_NU_FIXED_DEFAULT = 0.35

# Per-parameter prior-type choice in the Bayesian search (see core/bayes_
# visco_calibration.py's PriorSpec / module docstring "PRIORS" section):
# a combo box per row selects one of these, which shows/hides the matching
# field group (Value / Min+Max / Mean+Std / Median+Factor) below it.
_PRIOR_FIXED_IDX, _PRIOR_UNIFORM_IDX, _PRIOR_GAUSSIAN_IDX, _PRIOR_LOGNORMAL_IDX = 0, 1, 2, 3
_PRIOR_LABELS = [
    # "Fixed value" / "Uniform" match core.calib_priors.PRIOR_KIND_LABELS'
    # own wording (ui/pages/page_bayes_calib.py's Free parameters card)
    # verbatim; "Normal" likewise renamed from this page's previous
    # "Gaussian" label to match that page's naming for the same
    # distribution -- PriorSpec's own PRIOR_GAUSSIAN kind string/API name
    # is untouched (still "gaussian" internally, for saved-project
    # backward compatibility), only this display label changed.
    "Fixed value", "Uniform (Min / Max)", "Normal (Mean / Std)", "Log-normal (Median / Factor)",
]

# Readable labels for a hyperparameter's own field (see core/bayes_visco_
# calibration.py's "HYPERPRIORS" section) when reporting it in the Result
# text / corner plot, keyed by the "<param>__<field>" suffix after "__".
_HYPER_FIELD_LABELS = {"lo": "min", "hi": "max", "mean": "mean", "std": "std",
                        "median": "median", "factor": "factor (×/÷)"}


def _fmt_num(value: float) -> str:
    if value == int(value) and abs(value) < 1e6:
        return str(int(value))
    return f"{value:g}"


def _make_hyper_field(label_text: str, default: str) -> dict:
    """One numeric field that can either hold a single value, or, once its
    checkbox is ticked, a range the Bayesian search itself explores for it
    -- a hyperprior on that field (core/bayes_visco_calibration.py's
    PriorSpec / "HYPERPRIORS" section). Returns the pieces needed to add
    the field to a row and to read it back in `_on_run`; `edit` doubles as
    the range's lower bound once `hyper_checkbox` is checked, `hi_edit`
    (hidden until then) as its upper bound."""
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addWidget(QLabel(label_text))
    edit = QLineEdit(default)
    edit.setMaximumWidth(80)
    edit.setToolTip("Value, or the lower bound of a searched range if \"range\" is checked.")
    row.addWidget(edit)
    hyper_checkbox = QCheckBox("range")
    hyper_checkbox.setToolTip(
        "Search this field itself within a range instead of fixing it -- a hyperprior "
        "(see the card's help)."
    )
    row.addWidget(hyper_checkbox)
    hi_edit = QLineEdit(default)
    hi_edit.setMaximumWidth(80)
    hi_edit.setVisible(False)
    hi_edit.setToolTip("Upper bound of the searched range.")
    row.addWidget(hi_edit)
    hyper_checkbox.toggled.connect(hi_edit.setVisible)
    return dict(container=container, edit=edit, hyper_checkbox=hyper_checkbox, hi_edit=hi_edit)


def _read_hyper_field(field: dict, min_value):
    """Reads one `_make_hyper_field` field back as either a plain float or
    a `(lo, hi)` tuple (when its "range" checkbox is checked), raising
    ValueError -- caught by `_on_run`'s single generic message, matching
    this page's existing validation style -- on non-numeric input, a
    value at or below `min_value` (skip the check by passing None), or an
    inverted/degenerate range."""
    value = float(field["edit"].text())
    if field["hyper_checkbox"].isChecked():
        hi_value = float(field["hi_edit"].text())
        if min_value is not None and (value <= min_value or hi_value <= min_value):
            raise ValueError
        if value >= hi_value:
            raise ValueError
        return (value, hi_value)
    if min_value is not None and value <= min_value:
        raise ValueError
    return value


# `_CalibWorker` itself now lives in core/qt_worker.py (`CalibWorker`,
# imported above under this page's original private name so nothing below
# has to change) -- shared with ui/pages/page_bayes_calib.py's own
# calibration run, which needs the exact same QThread-worker plumbing and
# would otherwise be a second copy to keep in sync for no reason. See that
# module's docstring for the full wiring contract (`fn`'s
# `progress_cb`/`cancel_check` kwargs).


class ViscoCalibPage(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._timet = None
        self._thread = None
        self._worker = None
        self._cancel_event = None  # threading.Event for the current Bayesian calibration run, or None
        self._last_result = None
        self._last_result_model_key = None
        self._last_fixed_params = None
        self._last_mode = _MODE_BAYESIAN
        self._run_model_key = None
        self._active_model = get_model(MODEL_SLS)
        self._bayes_prior_widgets: dict[str, dict] = {}
        self._fixed_edits: dict[str, QLineEdit] = {}

        scroll, layout = make_scroll_page(
            "Viscoelastic Model Calibration",
            "Determine (or verify) a viscoelastic bar model's parameters, from two free-decay "
            "pulses recorded at two points along the same bar.",
        )

        signal_card = Card(
            "Signal",
            "Load the full free-decay signal, then drag out both pulse windows below.",
            help_html=(
                "<p>Loads the free-decay signal and marks the two pulses used for calibration "
                "(see the plot's own help below for how to drag the windows).</p>"
                "<ul>"
                "<li><b>Time file / Wave file</b>: the full recorded signal, containing both "
                "pulses.</li>"
                "<li><b>Auto-align Pulse 2 window to Pulse 1</b> (recommended): once both "
                "windows are roughly dragged, snaps Pulse 2's exact start to whichever nearby "
                "position best matches Pulse 1's shape, for sub-sample-precision alignment a "
                "mouse drag alone can't achieve.</li>"
                "<li><b>Search margin</b>: how far around the dragged position the auto-align "
                "search looks -- leave blank to pick it automatically.</li>"
                "<li><b>Re-align now</b>: re-runs the auto-align manually, e.g. with auto-align "
                "unchecked.</li>"
                "</ul>"
                "<p><b>Technical detail:</b> auto-alignment removes the arbitrary few-sample "
                "placement error that a manual drag alone would otherwise bake into the fit, "
                "using Pearson correlation against Pulse 1's shape.</p>"
            ),
        )
        self.time_file_row = TimeSourceRow("Time file (.txt)")
        self.time_file_row.file_selected.connect(self._on_time_file)
        self.time_file_row.dtChanged.connect(self._on_time_step_changed)
        signal_card.add(self.time_file_row)
        self.wave_file_row = FileSelectRow("Wave file (.txt)")
        self.wave_file_row.file_selected.connect(self._on_wave_file)
        signal_card.add(self.wave_file_row)
        self.picker = TimeWindowPicker([
            ("pulse1", "Pulse 1 window", theme.ACCENT),
            ("pulse2", "Pulse 2 window", "#ff9f43"),
        ], figsize=(9, 3),
            help_title="Free-decay signal",
            help_text=(
                "<p>Shows the full free-decay signal recorded on the bar.</p>"
                "<p>Select which window to set from the dropdown above the plot, then click and "
                "drag on the trace:</p>"
                "<ul>"
                "<li><b>Pulse 1 window</b> is the pulse recorded closer to the impact end.</li>"
                "<li><b>Pulse 2 window</b> is the same pulse recorded further along the bar.</li>"
                "</ul>"
                "<p>When \"Auto-align Pulse 2 window to Pulse 1\" is checked, dragging Pulse 2's "
                "window only sets its width/rough position -- its exact start then snaps to "
                "whichever nearby position best matches Pulse 1's shape.</p>"
            ),
        )
        self.picker.windowChanged.connect(self._on_window_changed)
        signal_card.add(self.picker)

        align_row = QHBoxLayout()
        self.auto_align_checkbox = QCheckBox("Auto-align Pulse 2 window to Pulse 1 (recommended)")
        self.auto_align_checkbox.setChecked(True)
        self.auto_align_checkbox.toggled.connect(lambda _checked: self._maybe_auto_align_pulse2())
        align_row.addWidget(self.auto_align_checkbox)
        align_row.addSpacing(16)
        align_row.addWidget(QLabel("Search margin (s, blank = auto)"))
        self.align_margin_edit = QLineEdit()
        self.align_margin_edit.setPlaceholderText("auto")
        self.align_margin_edit.setMaximumWidth(90)
        align_row.addWidget(self.align_margin_edit)
        realign_button = QPushButton("Re-align now")
        realign_button.clicked.connect(self._align_pulse2_now)
        align_row.addWidget(realign_button)
        align_row.addStretch(1)
        signal_card.add_layout(align_row)
        align_hint = QLabel(
            "Dragging Pulse 2's window sets its width and rough position only — its exact start is "
            "then snapped to whichever nearby position best matches Pulse 1's shape (Pearson "
            "correlation, sub-sample precision), which a mouse drag alone cannot achieve. This "
            "removes the arbitrary few-sample placement error that otherwise biases the fit."
        )
        align_hint.setObjectName("cardSubtitle")
        align_hint.setWordWrap(True)
        signal_card.add(align_hint)
        layout.addWidget(signal_card)

        params_card = Card(
            "Bar & test parameters",
            help_html=(
                "<p>Fixed physical parameters used by the forward wave-propagation model that "
                "both calibration modes fit against.</p>"
                "<ul>"
                "<li><b>Bar radius / density</b>: the bar's cross-section and material "
                "density.</li>"
                "<li><b>Distance between the two pulses</b>: the along-bar separation between "
                "where Pulse 1 and Pulse 2 were recorded.</li>"
                "<li><b>Damping factor/C</b>: the bar's own viscous damping coefficient, "
                "separate from the viscoelastic model being calibrated -- \"Use value from "
                "Free-Decay Damping tool\" copies in a value fitted on that other page.</li>"
                "</ul>"
            ),
        )
        self.radius_edit = QLineEdit("0.0127")
        params_card.add(LabeledField("Bar radius (m)", self.radius_edit))
        self.density_edit = QLineEdit("7800")
        params_card.add(LabeledField("Bar density (kg/m^3)", self.density_edit))
        self.z1_edit = QLineEdit("0.5")
        params_card.add(LabeledField("Distance between the two pulses (m)", self.z1_edit))
        self.damp_edit = QLineEdit(str(project.viscoelastic.damping_factor))
        params_card.add(LabeledField("Damping factor/C (rad/m)", self.damp_edit))
        use_damp_row = QHBoxLayout()
        use_damp_button = QPushButton("Use value from Free-Decay Damping tool")
        use_damp_button.clicked.connect(self._on_use_decay_damping)
        use_damp_row.addWidget(use_damp_button)
        use_damp_row.addStretch(1)
        params_card.add_layout(use_damp_row)
        layout.addWidget(params_card)

        mode_card = Card(
            "Mode",
            help_html=(
                "<p>Chooses how the calibration is run, and how goodness-of-fit is scored.</p>"
                "<ul>"
                "<li><b>Bayesian calibration</b> (recommended): searches the parameter space "
                "automatically within the bounds set below, converging in far fewer evaluations "
                "than an exhaustive grid.</li>"
                "<li><b>Test parameters (fixed values)</b>: instead evaluates one candidate "
                "parameter set you type in directly -- useful for verifying a known/previously-"
                "fitted set.</li>"
                "<li><b>Goodness-of-fit criterion</b>: Pearson correlation only checks that the "
                "predicted pulse has the right shape -- a candidate whose absolute amplitude/"
                "attenuation is wrong can still score close to 1. R² compares actual values "
                "too, so it also penalizes that; this is what both the Bayesian search and the "
                "reported score use by default.</li>"
                "</ul>"
            ),
        )
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Bayesian calibration (recommended — fast, free convergence)",
            "Test parameters (fixed values)",
        ])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_card.add(LabeledField("Mode", self.mode_combo))

        self._metric_keys = list(SCORE_ORDER)
        self.metric_combo = QComboBox()
        for key in self._metric_keys:
            self.metric_combo.addItem(SCORE_LABELS[key])
        self.metric_combo.setCurrentIndex(self._metric_keys.index(DEFAULT_METRIC))
        mode_card.add(LabeledField("Goodness-of-fit criterion", self.metric_combo))
        metric_hint = QLabel(
            "Pearson correlation only checks that the predicted pulse has the right *shape* — a "
            "candidate whose absolute amplitude/attenuation is wrong (e.g. the wrong damping or an "
            "overall modulus offset) can still score close to 1. R² compares actual values, so it "
            "also penalizes that — this is what both the Bayesian search and the reported score use."
        )
        metric_hint.setObjectName("cardSubtitle")
        metric_hint.setWordWrap(True)
        mode_card.add(metric_hint)
        layout.addWidget(mode_card)

        model_card = Card(
            "Rheological model",
            "Pick the constitutive model to calibrate — it becomes the active model for both "
            "Bayesian calibration and Test parameters, and for whatever gets pushed to the "
            "Dispersion Corrections page (see Result below).",
            help_html=(
                "<p>Picks the constitutive model to calibrate -- it becomes the active model "
                "for both Bayesian calibration and Test parameters, and changes which parameter "
                "fields appear in the cards below.</p>"
                "<p><b>Technical detail:</b> only a found/tested Standard Linear Solid set can "
                "be pushed to the Dispersion Corrections page with \"Apply to Dispersion "
                "Corrections\" (Result card below) -- that page's physics only understands that "
                "one model, so the button stays disabled for the other three models.</p>"
            ),
        )
        self.model_buttons: dict[str, RheologyModelButton] = {}
        self.model_button_group = QButtonGroup(self)
        self.model_button_group.setExclusive(True)
        model_grid = QGridLayout()
        model_grid.setSpacing(12)
        for i, key in enumerate(MODEL_ORDER):
            model = MODELS[key]
            btn = RheologyModelButton(key, model.display_name, model.short_formula)
            btn.setChecked(key == self._active_model.key)
            btn.clicked.connect(lambda _checked, k=key: self._on_model_clicked(k))
            self.model_button_group.addButton(btn)
            self.model_buttons[key] = btn
            model_grid.addWidget(btn, i // 2, i % 2)
        model_card.add_layout(model_grid)
        self.model_formula_label = QLabel(self._active_model.formula_html)
        self.model_formula_label.setObjectName("cardSubtitle")
        self.model_formula_label.setWordWrap(True)
        model_card.add(self.model_formula_label)
        layout.addWidget(model_card)

        self.bayes_card = Card(
            "Bayesian search priors",
            "Choose a prior for each parameter — Fixed, Uniform (Min/Max), Normal (Mean/Std) "
            "or Log-normal (Median/×÷Factor, best for a very wide plausible range) — and "
            "optionally tick \"range\" next to any of a prior's own fields to search that field "
            "itself (a hyperprior) instead of fixing it. An informative prior narrows the final "
            "reported uncertainty wherever the two pulses alone can't pin a parameter down.",
            help_html=(
                "<p>Sets each parameter's prior for the Bayesian search -- one of four kinds, "
                "chosen per parameter (Poisson ratio and every parameter of the selected "
                "rheological model), matching the same Fixed/Uniform/Normal vocabulary as the "
                "Bayesian Calibration page:</p>"
                "<ul>"
                "<li><b>Fixed value</b>: pins the parameter at that value and removes it from "
                "the search entirely (one fewer dimension for the optimizer to cover).</li>"
                "<li><b>Uniform (Min / Max)</b>: the previous default -- searched continuously "
                "and flatly between Min and Max, no Count/step needed. Moduli and viscosities "
                "are searched on a log scale (their natural, multiplicative scale), so Min/Max "
                "should both be strictly positive for those.</li>"
                "<li><b>Normal (Mean / Std)</b>: a Normal(Mean, Std) prior in the parameter's "
                "own physical units, e.g. from a literature value, a datasheet, or an earlier "
                "calibration.</li>"
                "<li><b>Log-normal (Median / Factor)</b>: like Normal but for a "
                "<i>multiplicative</i> uncertainty -- Factor is the one-sigma ×/÷ spread (e.g. "
                "Factor 3 with Median 2e9 means about 68% of the prior mass falls between "
                "0.67e9 and 6e9). Better than Normal whenever the parameter's plausible range "
                "spans close to (or more than) one order of magnitude, since a Log-normal's "
                "values are always strictly positive by construction, unlike a wide linear "
                "Normal's.</li>"
                "</ul>"
                "<p>Any informative prior (Normal or Log-normal) also reshapes the "
                "<i>reported</i> posterior: wherever the two pulses alone can't pin the "
                "parameter down, the result falls back toward that prior's own center/spread "
                "instead of spreading out across an arbitrarily wide search box -- which is what "
                "actually narrows down huge final uncertainties, provided the prior reflects "
                "real knowledge about that parameter.</p>"
                "<p><b>Hyperpriors:</b> ticking \"range\" next to Min, Max, Mean, Std, Median or "
                "Factor turns that one field into a range the search itself explores instead of "
                "a value you fix -- useful when you are unsure not just of the parameter's value "
                "but of where to even center its prior. Each ticked field costs one extra search "
                "dimension, so enable it only where genuinely needed, and raise the search "
                "budget below accordingly (this happens automatically as you tick/untick). The "
                "found hyperparameter values are reported alongside the parameter itself, below "
                "and in the corner plot.</p>"
                "<p>The fields shown depend on the model picked above.</p>"
            ),
        )
        layout.addWidget(self.bayes_card)

        self.budget_card = Card(
            "Bayesian search strategy & budget",
            help_html=(
                "<p>Controls how the adaptive Bayesian search picks its next point, and how much "
                "effort it spends doing so -- a full search can take a long time (a root-search "
                "per frequency, per forward-model evaluation), so these settings trade off run "
                "time against how thoroughly the search converges.</p>"
                "<ul>"
                "<li><b>Acquisition strategy</b>: which rule scores a candidate point's "
                "promise, each time the search picks what to evaluate next:</li>"
                "<ul>"
                f"<li><b>{ACQUISITION_LABELS[ACQ_EI]}</b> (recommended default) -- weighs how "
                "big an improvement over the current best could be, not just how likely one is. "
                "The balanced, general-purpose choice.</li>"
                f"<li><b>{ACQUISITION_LABELS[ACQ_PI]}</b> -- only asks how *likely* any "
                "improvement is, ignoring its size. Greedier than EI: converges in fewer "
                "evaluations once the search is already fairly well localized (e.g. by "
                "informative priors), but more prone to settling on a local optimum for a wide "
                "or weakly-identified search.</li>"
                f"<li><b>{ACQUISITION_LABELS[ACQ_UCB]}</b> -- scores purely by how uncertain the "
                "surrogate still is (mean + 2×std), regardless of the current best. "
                "Explores more systematically than EI/PI, so it suits a wide search box or a "
                "model prone to several competing optima (e.g. Generalized Maxwell/Burgers), at "
                "the cost of a few more evaluations before it settles.</li>"
                f"<li><b>{ACQUISITION_LABELS[ACQ_THOMPSON]}</b> -- draws one random plausible "
                "surface from the surrogate and evaluates its optimum. Stochastic rather than a "
                "fixed formula, so it naturally avoids repeatedly proposing near-duplicate "
                "points -- worth switching to if a search seems stuck (little improvement left "
                "despite iterations remaining).</li>"
                "</ul>"
                "<li><b>Initial samples (LHS)</b>: how many candidate points are drawn up front "
                "(Latin Hypercube Sampling) before the adaptive search phase -- and its chosen "
                "acquisition strategy -- begins.</li>"
                "<li><b>Max iterations</b>: the hard cap on additional adaptive evaluations "
                "after the initial samples.</li>"
                "<li><b>Stop after N iterations without improvement</b>: ends the search early "
                "once it stops finding a better score.</li>"
                "</ul>"
                "<p><b>Technical detail:</b> switching the rheological model above, or ticking/"
                "unticking a hyperprior \"range\" checkbox, resets the budget defaults to scale "
                "with the search's actual dimensionality (more dimensions need a larger budget "
                "to reliably locate the optimum). The acquisition strategy only affects the "
                "adaptive phase, not the initial samples or the final reported posterior.</p>"
            ),
        )
        self._acquisition_keys = [ACQ_EI, ACQ_PI, ACQ_UCB, ACQ_THOMPSON]
        self.acquisition_combo = QComboBox()
        for acq_key in self._acquisition_keys:
            self.acquisition_combo.addItem(f"{ACQUISITION_LABELS[acq_key]} — {ACQUISITION_USE_CASES[acq_key]}")
        self.acquisition_combo.setCurrentIndex(self._acquisition_keys.index(ACQ_EI))
        self.budget_card.add(LabeledField("Acquisition strategy", self.acquisition_combo))

        budget_row = QHBoxLayout()
        budget_row.addWidget(QLabel("Initial samples (LHS)"))
        self.bayes_n_init_edit = QLineEdit("12")
        self.bayes_n_init_edit.setMaximumWidth(70)
        budget_row.addWidget(self.bayes_n_init_edit)
        budget_row.addSpacing(20)
        budget_row.addWidget(QLabel("Max iterations"))
        self.bayes_n_iter_edit = QLineEdit("40")
        self.bayes_n_iter_edit.setMaximumWidth(70)
        budget_row.addWidget(self.bayes_n_iter_edit)
        budget_row.addSpacing(20)
        budget_row.addWidget(QLabel("Stop after N iterations without improvement"))
        self.bayes_patience_edit = QLineEdit("10")
        self.bayes_patience_edit.setMaximumWidth(70)
        budget_row.addWidget(self.bayes_patience_edit)
        budget_row.addStretch(1)
        self.budget_card.add_layout(budget_row)
        layout.addWidget(self.budget_card)

        self.fixed_card = Card(
            "Fixed parameters to test",
            help_html=(
                "<p>Used only in \"Test parameters (fixed values)\" mode: type in one candidate "
                "value for each parameter of the currently selected rheological model, plus the "
                "Poisson ratio, and \"Run\" evaluates that single set against the two pulses "
                "instead of searching.</p>"
                "<p>Useful for verifying a known or previously-fitted parameter set without "
                "running a full search.</p>"
            ),
        )
        self.fixed_card.setVisible(False)
        layout.addWidget(self.fixed_card)

        self._rebuild_param_cards()

        run_card = Card(
            "Run",
            "A full search can take a long time (a root-search per frequency, per forward-model "
            "evaluation) -- Bayesian calibration converges in far fewer evaluations than a grid "
            "ever could, but each one still isn't free.",
        )
        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._on_run)
        run_row.addWidget(self.run_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip(
            "Stops a Bayesian calibration run after the candidate point currently being "
            "evaluated finishes -- the points already evaluated are discarded, not kept as a "
            "partial result. Has no effect on \"Test parameters\" (a single evaluation, nothing "
            "to stop mid-flight)."
        )
        self.stop_button.clicked.connect(self._on_stop)
        run_row.addWidget(self.stop_button)
        run_row.addStretch(1)
        run_card.add_layout(run_row)
        self.progress_bar = QProgressBar()
        run_card.add(self.progress_bar)
        self.status_label = QLabel("")
        self.status_label.setObjectName("cardSubtitle")
        run_card.add(self.status_label)
        layout.addWidget(run_card)

        result_card = Card(
            "Result",
            help_html=(
                "<p>Shows the outcome of the last calibration/test run (see the plot's own help "
                "below).</p>"
                "<ul>"
                "<li>The goodness-of-fit score (Pearson correlation or R², per the "
                "criterion picked in the Mode card) and the fitted/tested parameter values are "
                "printed above the plot.</li>"
                "<li><b>Apply to Dispersion Corrections</b>: pushes this result's parameters to "
                "the Dispersion Corrections page -- only enabled when the active model is the "
                "Standard Linear Solid, since that page's physics only understands that one "
                "model.</li>"
                "</ul>"
            ),
        )
        self.result_label = QLabel("")
        self.result_label.setObjectName("cardSubtitle")
        self.result_label.setWordWrap(True)
        result_card.add(self.result_label)
        self.canvas = PlotCanvas(
            figsize=(8, 4.5),
            help_title="Calibration result",
            help_text=(
                "<p>Shows Pulse 2 (the measured, further-along-the-bar signal) together with "
                "Pulse 1 propagated forward through the calibrated (or tested) rheological "
                "model.</p>"
                "<p>A good match in both shape and amplitude between predicted and measured "
                "Pulse 2 is what a good calibration looks like. The reported goodness-of-fit "
                "score above (Pearson correlation or R², see the criterion picked further "
                "up) quantifies this same comparison numerically.</p>"
                "<p>In Bayesian mode, a second panel also shows the best score found so far "
                "against the number of forward-model evaluations.</p>"
            ),
        )
        result_card.add(self.canvas)
        apply_row = QHBoxLayout()
        self.apply_button = QPushButton("Apply to Dispersion Corrections")
        self.apply_button.setObjectName("primary")
        self.apply_button.clicked.connect(self._on_apply)
        self.apply_button.setEnabled(False)
        apply_row.addWidget(self.apply_button)
        apply_row.addStretch(1)
        result_card.add_layout(apply_row)
        layout.addWidget(result_card)

        self.interdep_card = Card(
            "Parameter interdependence",
            "How strongly each pair of calibrated parameters trades off against the other across "
            "the Bayesian posterior. A value near ±1 means this measurement mostly constrains some "
            "combination of the two (e.g. their product) rather than either one individually — more "
            "search budget will not fix that, only a different/additional measurement can. Only "
            "available after a Bayesian calibration run.",
        )
        self.interdep_canvas = PlotCanvas(
            figsize=(5.5, 5), show_toolbar=False,
            help_title="Parameter interdependence",
            help_text=(
                "<p>A color matrix showing how strongly each pair of calibrated parameters "
                "trades off against the other across the Bayesian posterior.</p>"
                "<p>A cell near &plusmn;1 means this measurement mostly constrains some "
                "combination of the two parameters (e.g. their product) rather than either one "
                "individually -- more search budget will not resolve that ambiguity, only a "
                "different or additional measurement can.</p>"
                "<p>Only populated after a Bayesian calibration run.</p>"
            ),
        )
        self.interdep_card.add(self.interdep_canvas)
        self.interdep_card.setVisible(False)
        layout.addWidget(self.interdep_card)

        self.corner_card = Card(
            "Attempted values & parameter densities",
            "Lower triangle: every parameter combination the Bayesian search actually evaluated "
            "(colored by score) — where it looked and where it concentrated. Diagonal: each "
            "parameter's weighted posterior probability density, with the reported mean marked. "
            "Only available after a Bayesian calibration run.",
        )
        self.corner_canvas = PlotCanvas(
            figsize=(9, 8),
            help_title="Attempted values & parameter densities",
            help_text=(
                "<p>A corner (pairs) plot summarizing the Bayesian search.</p>"
                "<ul>"
                "<li><b>Lower triangle</b>: every parameter combination actually evaluated "
                "during the search, colored by score -- shows where the search looked and where "
                "it concentrated as it converged.</li>"
                "<li><b>Diagonal</b>: each parameter's own weighted posterior probability "
                "density, with the reported mean value marked -- a narrow peak means that "
                "parameter is well constrained by the two pulses, a broad or flat one means it "
                "isn't.</li>"
                "</ul>"
                "<p>Shows only the direct model parameters (nu and the model's own moduli/"
                "viscosities) -- any hyperparameter being searched alongside them (e.g. a "
                "parameter's own prior mean/std, when that is itself left free) is left out of "
                "this matrix.</p>"
                "<p>Only populated after a Bayesian calibration run.</p>"
            ),
        )
        self.corner_card.add(self.corner_canvas)
        self.corner_card.setVisible(False)
        layout.addWidget(self.corner_card)

        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # -- model / mode selection -------------------------------------------------

    def _on_model_clicked(self, key):
        if self._active_model.key == key:
            return
        self._active_model = MODELS[key]
        self.model_formula_label.setText(self._active_model.formula_html)
        self._rebuild_param_cards()

    def _rebuild_param_cards(self):
        model = self._active_model

        self.bayes_card.clear()
        self._bayes_prior_widgets = {}
        prior_rows = [(_NU_KEY, _NU_LABEL, _NU_BAYES_BOUNDS, _NU_FIXED_DEFAULT)] + [
            (p.key, p.label, p.bayes_bounds, p.fixed_default) for p in model.params
        ]
        for key, label, (lo, hi), default in prior_rows:
            log_scale = key != _NU_KEY
            row = QHBoxLayout()
            row.addWidget(QLabel(label))

            combo = QComboBox()
            combo.addItems(_PRIOR_LABELS)
            combo.setCurrentIndex(_PRIOR_UNIFORM_IDX)  # previous behavior, unchanged by default
            combo.setMaximumWidth(180)
            row.addWidget(combo)

            fixed_group = QWidget()
            fixed_layout = QHBoxLayout(fixed_group)
            fixed_layout.setContentsMargins(0, 0, 0, 0)
            fixed_edit = QLineEdit(_fmt_num(default))
            fixed_edit.setMaximumWidth(90)
            fixed_layout.addWidget(QLabel("Value"))
            fixed_layout.addWidget(fixed_edit)
            row.addWidget(fixed_group)

            uniform_group = QWidget()
            uniform_layout = QHBoxLayout(uniform_group)
            uniform_layout.setContentsMargins(0, 0, 0, 0)
            uniform_lo = _make_hyper_field("Min", _fmt_num(lo))
            uniform_hi = _make_hyper_field("Max", _fmt_num(hi))
            uniform_layout.addWidget(uniform_lo["container"])
            uniform_layout.addWidget(uniform_hi["container"])
            row.addWidget(uniform_group)

            gaussian_group = QWidget()
            gaussian_layout = QHBoxLayout(gaussian_group)
            gaussian_layout.setContentsMargins(0, 0, 0, 0)
            # A starting-point Std, not a considered value -- 20% of the
            # default is just wide enough to be visibly adjustable rather
            # than looking already-final; the user should set this from
            # real prior knowledge (literature, datasheet, earlier fit).
            gaussian_mean = _make_hyper_field("Mean", _fmt_num(default))
            gaussian_std = _make_hyper_field("Std", _fmt_num(max(abs(default) * 0.2, 1e-6)))
            gaussian_layout.addWidget(gaussian_mean["container"])
            gaussian_layout.addWidget(gaussian_std["container"])
            row.addWidget(gaussian_group)

            lognormal_group = QWidget()
            lognormal_layout = QHBoxLayout(lognormal_group)
            lognormal_layout.setContentsMargins(0, 0, 0, 0)
            # Factor 3 (a starting point, same spirit as the Normal prior's
            # Std default above): about 68% of the prior mass between
            # default/3 and default*3 -- wide enough to be visibly a
            # starting guess, not a considered value.
            lognormal_median = _make_hyper_field("Median", _fmt_num(default))
            lognormal_factor = _make_hyper_field("Factor", "3")
            lognormal_layout.addWidget(lognormal_median["container"])
            lognormal_layout.addWidget(lognormal_factor["container"])
            row.addWidget(lognormal_group)

            row.addStretch(1)

            groups = dict(fixed=fixed_group, uniform=uniform_group,
                          gaussian=gaussian_group, lognormal=lognormal_group)

            # `groups=groups` binds *this row's* groups into the closure at
            # definition time -- without it, every row's handler would
            # share the same free variable and all end up referencing
            # whichever row was built last (the classic Python late-
            # binding-closure-in-a-loop trap), so changing one row's prior
            # kind would silently toggle a *different* row's fields.
            def _on_row_changed(index, groups=groups):
                groups["fixed"].setVisible(index == _PRIOR_FIXED_IDX)
                groups["uniform"].setVisible(index == _PRIOR_UNIFORM_IDX)
                groups["gaussian"].setVisible(index == _PRIOR_GAUSSIAN_IDX)
                groups["lognormal"].setVisible(index == _PRIOR_LOGNORMAL_IDX)
                self._update_budget_suggestion()

            combo.currentIndexChanged.connect(_on_row_changed)
            for field in (uniform_lo, uniform_hi, gaussian_mean, gaussian_std,
                          lognormal_median, lognormal_factor):
                field["hyper_checkbox"].toggled.connect(self._update_budget_suggestion)
            _on_row_changed(combo.currentIndex())

            self._bayes_prior_widgets[key] = dict(
                combo=combo, log_scale=log_scale, groups=groups,
                fixed=dict(edit=fixed_edit),
                uniform=dict(lo=uniform_lo, hi=uniform_hi),
                gaussian=dict(mean=gaussian_mean, std=gaussian_std),
                lognormal=dict(median=lognormal_median, factor=lognormal_factor),
            )
            self.bayes_card.add_layout(row)

        self.fixed_card.clear()
        self._fixed_edits = {}
        fixed_rows = [(_NU_KEY, _NU_LABEL, _NU_FIXED_DEFAULT)] + [(p.key, p.label, p.fixed_default) for p in model.params]
        for key, label, default in fixed_rows:
            edit = QLineEdit(_fmt_num(default))
            self._fixed_edits[key] = edit
            self.fixed_card.add(LabeledField(label, edit))

        self._update_budget_suggestion()

    def _current_search_dim(self) -> int:
        """Mirrors core.bayes_visco_calibration._build_layout's dimension
        count from the current widget state (one column per free
        parameter, plus one more per ticked "range"/hyperprior field --
        see the bayes_card help), so the suggested search budget can
        account for hyperpriors adding extra search dimensions."""
        dim = 0
        for w in self._bayes_prior_widgets.values():
            idx = w["combo"].currentIndex()
            if idx == _PRIOR_FIXED_IDX:
                continue
            elif idx == _PRIOR_UNIFORM_IDX:
                fields = (w["uniform"]["lo"], w["uniform"]["hi"])
            elif idx == _PRIOR_GAUSSIAN_IDX:
                fields = (w["gaussian"]["mean"], w["gaussian"]["std"])
            else:
                fields = (w["lognormal"]["median"], w["lognormal"]["factor"])
            dim += 1 + sum(1 for f in fields if f["hyper_checkbox"].isChecked())
        return dim

    def _update_budget_suggestion(self):
        # A higher-dimensional search needs a larger budget to reliably
        # locate its optimum -- more parameters (Burgers and especially
        # Generalized Maxwell have 5-6 against 3-4 for Kelvin-Voigt/SLS)
        # or any hyperprior enabled (each one adds a dimension of its own,
        # see _current_search_dim) both call for a larger budget and more
        # patience before declaring convergence. Recomputed -- and the
        # budget fields overwritten -- on every prior-kind or "range"
        # checkbox change, same as it already was on every model switch.
        dim = max(self._current_search_dim(), 1)
        self.bayes_n_init_edit.setText(str(max(12, 3 * dim)))
        self.bayes_n_iter_edit.setText(str(max(40, 8 * dim)))
        self.bayes_patience_edit.setText(str(max(10, 3 * dim)))

    def _on_mode_changed(self, index):
        self.bayes_card.setVisible(index == _MODE_BAYESIAN)
        self.budget_card.setVisible(index == _MODE_BAYESIAN)
        self.fixed_card.setVisible(index == _MODE_FIXED)

    def _on_use_decay_damping(self):
        self.damp_edit.setText(str(self.project.viscoelastic.damping_factor))

    # -- signal loading -----------------------------------------------------

    def _read_column(self, path):
        values = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    values.append(float(line))
        return np.asarray(values)

    def _on_time_file(self, path):
        try:
            self._timet = self._read_column(path)
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Could not read time file: {exc}")
            return
        self._refresh_picker()

    def _on_wave_file(self, path):
        try:
            self._wave = self._read_column(path)
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Could not read wave file: {exc}")
            return
        self._maybe_regenerate_time()
        self._refresh_picker()

    def _on_time_step_changed(self, dt):
        """Alternative to `_on_time_file` above: build the time vector from
        a fixed time step matching the wave file's length, instead of
        loading it from a file."""
        self._maybe_regenerate_time()
        if self._timet is None:
            self.status_label.setText("Load the wave file first, so the generated time vector can match its length.")
            return
        self._refresh_picker()

    def _maybe_regenerate_time(self):
        if not self.time_file_row.is_generate_mode():
            return
        n = len(self._wave) if hasattr(self, "_wave") else 0
        if n == 0:
            return
        generated = self.time_file_row.build(n)
        if generated is not None:
            self._timet = generated

    def _refresh_picker(self):
        if self._timet is None:
            self.picker.show_placeholder("Load a time file above to see the signal here.")
            return
        if not hasattr(self, "_wave") or len(self._wave) == 0:
            self.picker.show_placeholder("Load a wave file above to see the signal here.")
            return
        self.picker.set_signal(self._timet, [("Signal", self._wave, theme.ACCENT)])

    def _on_window_changed(self, key, start, end):
        # Re-align pulse 2 whenever either window is (re)dragged: moving
        # pulse 2 itself should snap it, and moving pulse 1 changes the
        # template pulse 2 is matched against, so a stale alignment against
        # the old pulse 1 window would otherwise silently linger. Only the
        # automatic (checkbox-gated) path fires here -- "Re-align now" is
        # wired separately so it still works as a manual one-shot when the
        # checkbox is off.
        if key in ("pulse1", "pulse2"):
            self._maybe_auto_align_pulse2()

    def _maybe_auto_align_pulse2(self):
        if not self.auto_align_checkbox.isChecked():
            return
        self._align_pulse2_now()

    def _align_pulse2_now(self):
        if self._timet is None or not hasattr(self, "_wave"):
            return
        window1 = self.picker.get_window("pulse1")
        window2 = self.picker.get_window("pulse2")
        if window1 is None or window2 is None:
            return

        margin_text = self.align_margin_edit.text().strip()
        margin = None
        if margin_text:
            try:
                margin = float(margin_text)
            except ValueError:
                self.status_label.setText("Search margin must be numeric (or left blank for auto).")
                return

        aligned = align_pulse2(self._timet, self._wave, window1, window2, margin=margin)
        if aligned is None:
            self.status_label.setText(
                "Could not auto-align Pulse 2 (too close to the edge of the recording, or too little "
                "signal around it) — using the window as dragged."
            )
            return
        if aligned == window2:
            return
        self.picker.set_window("pulse2", *aligned)
        self.status_label.setText(
            f"Pulse 2 window auto-aligned to Pulse 1 (start moved from {window2[0]:.6g} s to "
            f"{aligned[0]:.6g} s)."
        )

    # -- run ------------------------------------------------------------------

    def _on_run(self):
        window1 = self.picker.get_window("pulse1")
        window2 = self.picker.get_window("pulse2")
        if window1 is None or window2 is None:
            self.status_label.setText("Drag out both the Pulse 1 and Pulse 2 windows on the plot first.")
            return
        if self._timet is None or not hasattr(self, "_wave"):
            self.status_label.setText("Load the time and wave files first.")
            return

        try:
            radius = float(self.radius_edit.text())
            density = float(self.density_edit.text())
            z1 = float(self.z1_edit.text())
            damp = float(self.damp_edit.text())
        except ValueError:
            self.status_label.setText("Bar radius/density, distance and damping must be numeric.")
            return

        i1 = int(np.searchsorted(self._timet, window1[0], side="right") - 1)
        i2 = int(np.searchsorted(self._timet, window1[1], side="right") - 1)
        i3 = int(np.searchsorted(self._timet, window2[0], side="right") - 1)
        i4 = int(np.searchsorted(self._timet, window2[1], side="right") - 1)
        time1 = self._timet[i1:i2] - window1[0]
        pulse1 = np.abs(self._wave[i1:i2])
        time2 = self._timet[i3:i4] - window2[0]
        pulse2 = np.abs(self._wave[i3:i4])

        mode = self.mode_combo.currentIndex()
        self._last_mode = mode
        model = self._active_model
        metric = self._metric_keys[self.metric_combo.currentIndex()]

        if mode == _MODE_BAYESIAN:
            try:
                param_priors = {}
                for key, w in self._bayes_prior_widgets.items():
                    idx = w["combo"].currentIndex()
                    pos_min = 0.0 if w["log_scale"] else None
                    if idx == _PRIOR_FIXED_IDX:
                        value = float(w["fixed"]["edit"].text())
                        if pos_min is not None and value <= pos_min:
                            raise ValueError
                        param_priors[key] = PriorSpec.fixed(value)
                    elif idx == _PRIOR_GAUSSIAN_IDX:
                        mean = _read_hyper_field(w["gaussian"]["mean"], None)
                        std = _read_hyper_field(w["gaussian"]["std"], 0.0)
                        param_priors[key] = PriorSpec.gaussian(mean, std)
                    elif idx == _PRIOR_LOGNORMAL_IDX:
                        median = _read_hyper_field(w["lognormal"]["median"], 0.0)
                        factor = _read_hyper_field(w["lognormal"]["factor"], 1.0)
                        param_priors[key] = PriorSpec.lognormal(median, factor)
                    else:
                        lo = _read_hyper_field(w["uniform"]["lo"], pos_min)
                        hi = _read_hyper_field(w["uniform"]["hi"], pos_min)
                        param_priors[key] = PriorSpec.uniform(lo, hi)
                n_init = int(self.bayes_n_init_edit.text())
                n_iter = int(self.bayes_n_iter_edit.text())
                patience = int(self.bayes_patience_edit.text())
            except ValueError:
                self.status_label.setText(
                    "Bayesian prior fields must be numeric (moduli/viscosities and Log-normal "
                    "medians must be > 0, a Normal prior's Std must be > 0, a Log-normal's Factor "
                    "must be > 1, and any \"range\" field's lower bound must be less than its "
                    "upper bound)."
                )
                return
            acquisition = self._acquisition_keys[self.acquisition_combo.currentIndex()]
            fn = calibrate_bayesian
            kwargs = dict(
                pulse1=pulse1, time1=time1, pulse2=pulse2, time2=time2,
                radius=radius, density=density, z1=z1, damp=damp,
                model=model, param_priors=param_priors,
                n_init=n_init, n_iter=n_iter, convergence_patience=patience,
                metric=metric, acquisition=acquisition,
                n_workers=self.project.n_workers,
            )
        else:
            try:
                nu = float(self._fixed_edits[_NU_KEY].text())
                params = {key: float(self._fixed_edits[key].text()) for key in model.param_keys()}
            except ValueError:
                self.status_label.setText("Fixed parameter fields must be numeric.")
                return
            self._last_fixed_params = {_NU_KEY: nu, **params}
            fn = test_parameters
            kwargs = dict(
                pulse1=pulse1, time1=time1, pulse2=pulse2, time2=time2,
                radius=radius, density=density, z1=z1, damp=damp,
                nu=nu, model=model, params=params,
                metric=metric,
            )

        self._run_model_key = model.key
        self.run_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        for btn in self.model_buttons.values():
            btn.setEnabled(False)
        # Only "Bayesian calibration" gets a live cancel_event -- "Test
        # parameters" is a single, near-instant evaluation (see
        # `_CalibWorker`'s own docstring), so Stop stays disabled for it.
        self._cancel_event = threading.Event() if mode == _MODE_BAYESIAN else None
        self.stop_button.setEnabled(self._cancel_event is not None)
        self.status_label.setText("Running (this involves a root-search per frequency and can take a while)…")
        self.progress_bar.setValue(0)

        self._thread = QThread()
        self._worker = _CalibWorker(fn, kwargs, cancel_event=self._cancel_event)
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
            self.status_label.setText("Stopping… (finishing the candidate point currently being evaluated)")

    def _on_cancelled(self):
        self._cancel_event = None
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        for btn in self.model_buttons.values():
            btn.setEnabled(True)
        self.status_label.setText("Calibration stopped by user.")

    def _on_progress(self, i, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(i)

    def _format_params(self, values: dict, stds: dict | None, model, hyper_names: tuple = ()) -> str:
        def fmt(key, label, unit):
            if stds is not None:
                return f"{label} = {values[key]:.4g} ± {stds[key]:.4g} {unit}".rstrip()
            return f"{label} = {values[key]:.4g} {unit}".rstrip()

        param_units = {p.key: p.unit for p in model.params}
        param_units[_NU_KEY] = ""
        param_shorts = {p.key: p.short for p in model.params}
        param_shorts[_NU_KEY] = "nu"

        bits = [fmt(_NU_KEY, "nu", "")]
        for p in model.params:
            bits.append(fmt(p.key, p.short, p.unit))
        # Any active hyperparameter (see core/bayes_visco_calibration.py's
        # "HYPERPRIORS" section) is reported the same way, right after the
        # physical parameters -- its own weighted-posterior mean/std, from
        # the same posterior sample.
        for name in hyper_names:
            param_key, field = name.split("__", 1)
            short = param_shorts.get(param_key, param_key)
            field_label = _HYPER_FIELD_LABELS.get(field, field)
            unit = "" if field == "factor" else param_units.get(param_key, "")
            bits.append(fmt(name, f"{short} {field_label}", unit))
        return "   ".join(bits)

    def _on_finished(self, result):
        self._cancel_event = None
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        for btn in self.model_buttons.values():
            btn.setEnabled(True)
        self._last_result = result
        self._last_result_model_key = self._run_model_key
        model = get_model(self._last_result_model_key)

        self.canvas.clear()
        is_bayesian = isinstance(result, ViscoBayesianResult)
        if is_bayesian:
            ax = self.canvas.figure.add_subplot(121)
        else:
            ax = self.canvas.figure.add_subplot(111)
        ax.plot(result.time1, result.pulse1, color="red", label="Pulse 1 (measured)")
        ax.plot(result.time2, result.pulse2, color="blue", label="Pulse 2 (measured)")
        ax.plot(result.time1, result.predicted_pulse[: len(result.time1)], color="green", linestyle="--", label="Predicted pulse 2")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=7 if is_bayesian else 8)

        if is_bayesian:
            score_label = SCORE_SHORT_LABELS[result.metric]
            ax2 = self.canvas.figure.add_subplot(122)
            evals = np.arange(1, len(result.history_best) + 1)
            ax2.plot(evals, result.history_best, color=theme.ACCENT)
            ax2.axvline(result.converged_at + 1, color=theme.TEXT_MUTED, linestyle=":", linewidth=1)
            ax2.set_xlabel("Forward-model evaluation")
            ax2.set_ylabel(f"Best {score_label} found so far")
            ax2.set_title(f"{result.n_evaluations} evaluations total", fontsize=9)
        self.canvas.refresh()

        self.status_label.setText("Done.")
        if is_bayesian:
            header = (
                f"[{model.display_name}]  {score_label} = {result.score_mean:.4g} ± "
                f"{result.score_std:.4g}  (best single evaluation: {result.best_score:.4g}, "
                f"after {result.n_evaluations} forward-model evaluations, converged at #{result.converged_at + 1})"
            )
            self.result_label.setText(
                header + "\n" + self._format_params(result.means, result.stds, model, result.hyper_names)
            )
            self._plot_interdependence(result, model)
            self.interdep_card.setVisible(True)
            self._plot_corner(result, model)
            self.corner_card.setVisible(True)
        else:
            score_label = SCORE_SHORT_LABELS[result.metric]
            self.result_label.setText(f"[{model.display_name}]  {score_label} = {result.score:.4g}")
            self.interdep_card.setVisible(False)
            self.corner_card.setVisible(False)

        self.apply_button.setEnabled(True)

    def _plot_interdependence(self, result: ViscoBayesianResult, model) -> None:
        """Draws the weighted-posterior parameter correlation matrix (see
        core/bayes_visco_calibration.py's module docstring and
        ViscoBayesianResult.param_correlation) as an annotated heatmap."""
        labels = [_NU_KEY] + [p.short for p in model.params]
        matrix = result.param_correlation
        n = len(labels)

        self.interdep_canvas.clear()
        ax = self.interdep_canvas.figure.add_subplot(111)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels)
        for i in range(n):
            for j in range(n):
                value = matrix[i, j]
                text_color = "white" if abs(value) > 0.6 else theme.TEXT
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)
        cbar = self.interdep_canvas.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Weighted posterior correlation")
        self.interdep_canvas.refresh()

    def _plot_corner(self, result: ViscoBayesianResult, model) -> None:
        """Draws a corner plot (the standard Bayesian-calibration diagnostic
        combining both requested views in one): the lower triangle scatters
        every point the search actually evaluated (`result.evaluated_params`/
        `evaluated_scores` -- the "attempted values" trace, not a resampled
        one) for each parameter pair, colored by score; the diagonal draws
        each parameter's weighted posterior density (`result.posterior_
        samples`/`posterior_weights`, the same dense sample `means`/`stds`/
        `param_correlation` were computed from). Moduli/viscosities are
        plotted on a log scale (their natural, multiplicative search
        scale -- see core/bayes_visco_calibration.py's `calibrate_bayesian`
        docstring); the Poisson ratio nu is plotted linearly."""
        base_labels = [_NU_KEY] + [p.short for p in model.params]
        base_log_mask = [False] + [True] * len(model.params)

        # Only the direct physical parameters go in this matrix -- any
        # active hyperparameter (see core/bayes_visco_calibration.py's
        # "HYPERPRIORS" section, e.g. a "<param>__mean"/"<param>__std" pair
        # when a parameter's own prior mean/std is itself being searched)
        # is deliberately left out here, since it isn't a direct model
        # parameter and clutters the matrix with rows/columns that don't
        # describe the material itself. Those hyperparameter values are
        # still fully present in `result.hyper_names`/`evaluated_params`/
        # `posterior_samples` for anything else that wants them.
        names = list(result.param_names)
        labels = base_labels
        log_mask = base_log_mask
        dim = len(names)

        self.corner_canvas.clear()
        fig = self.corner_canvas.figure
        axes = fig.subplots(dim, dim, squeeze=False)
        scores = result.evaluated_scores
        vmin, vmax = float(np.min(scores)), float(np.max(scores))
        sc = None
        for i in range(dim):
            for j in range(dim):
                ax = axes[i][j]
                if j > i:
                    ax.axis("off")
                    continue
                name_i, name_j = names[i], names[j]
                if i == j:
                    samples = result.posterior_samples[name_i]
                    weights = result.posterior_weights
                    if log_mask[i] and samples.min() > 0:
                        bins = np.logspace(np.log10(samples.min()), np.log10(samples.max()), 40)
                        ax.set_xscale("log")
                    else:
                        bins = 40
                    ax.hist(samples, bins=bins, weights=weights, color=theme.ACCENT, alpha=0.85)
                    ax.axvline(result.means[name_i], color=theme.TEXT, linestyle=":", linewidth=1)
                    ax.set_yticks([])
                else:
                    x, y = result.evaluated_params[name_j], result.evaluated_params[name_i]
                    sc = ax.scatter(x, y, c=scores, cmap="viridis", vmin=vmin, vmax=vmax, s=14, linewidths=0)
                    if log_mask[j]:
                        ax.set_xscale("log")
                    if log_mask[i]:
                        ax.set_yscale("log")
                if i == dim - 1:
                    ax.set_xlabel(labels[j], fontsize=8)
                else:
                    ax.set_xticklabels([])
                if j == 0 and i != 0:
                    ax.set_ylabel(labels[i], fontsize=8)
                elif i != j:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=6)
        if sc is not None:
            fig.colorbar(sc, ax=axes, shrink=0.6, label=SCORE_SHORT_LABELS[result.metric])
        self.corner_canvas.refresh()

    def _on_failed(self, message):
        self._cancel_event = None
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        for btn in self.model_buttons.values():
            btn.setEnabled(True)
        self.status_label.setText(f"Failed: {message}")

    def _on_apply(self):
        result = self._last_result
        if result is None or self._last_result_model_key is None:
            return
        model = get_model(self._last_result_model_key)
        v = self.project.viscoelastic
        if isinstance(result, ViscoBayesianResult):
            v.poisson_ratio = result.means[_NU_KEY]
            v.params = {k: result.means[k] for k in model.param_keys()}
            v.params_std = {k: result.stds[k] for k in model.param_keys()}
        elif self._last_fixed_params is not None:
            v.poisson_ratio = self._last_fixed_params[_NU_KEY]
            v.params = {k: self._last_fixed_params[k] for k in model.param_keys()}
            v.params_std = {}
        else:
            return
        v.model_key = model.key
        self.result_label.setText(
            self.result_label.text() + f"\n\nApplied to the Dispersion Corrections page ({model.display_name})."
        )
