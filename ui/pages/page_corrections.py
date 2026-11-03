"""
Dispersion corrections page.

Replaces the four separate popup-driven scripts FFT_dispersion.py /
FFT_dispersion_ID.py / FFT_dispersion_visco.py / FFT_dispersion_visco_ID.py.
One page picks elastic/viscoelastic and SHPB/Direct-Impact, then the pulse
window(s) are set by click-and-drag directly on the raw signal(s) (see
ui/span_picker.py) instead of typed time values -- result plots render
inline instead of a blocking `plt.show()` window, and long root-searches
report progress on a QProgressBar instead of a printed console bar.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from PySide6.QtCore import QThread, Signal, QObject
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QPushButton, QProgressBar, QLabel, QHBoxLayout, QCheckBox, QLineEdit,
    QGridLayout, QButtonGroup,
)

from core import dispersion
from core.project import PLATEN_PARAM_ORDER
from core.downsampling import downsample
from core.rheology_models import MODELS, MODEL_ORDER, get_model
from ui.widgets import Card, LabeledField, FileSelectRow, MCParamTable, make_scroll_page, _clear_layout, TimeSourceRow
from ui.plot_canvas import PlotCanvas
from ui.span_picker import TimeWindowPicker
from ui.rheology_diagram import RheologyModelButton
from ui import theme


def _fmt_num(value: float) -> str:
    if value == int(value) and abs(value) < 1e6:
        return str(int(value))
    return f"{value:g}"


class _CorrectionWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, kwargs):
        super().__init__()
        self.fn = fn
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.fn(progress_cb=lambda i, total: self.progress.emit(i, total), **self.kwargs)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 - report inline, never a modal error popup
            self.failed.emit(str(exc))


class CorrectionsPage(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._thread = None
        self._worker = None
        self._timet = None        # ndarray, for the two-bar (SHPB) setups
        self._time_pulse = None   # rounded python list, for the Direct-Impact setups

        scroll, layout = make_scroll_page(
            "Dispersion Corrections",
            "Pochhammer-Chree dispersion correction of the strain-gage signals back to the "
            "bar/specimen interfaces.",
        )

        mode_card = Card(
            "Correction type",
            help_html=(
                "<p>Selects which dispersion-correction physics and bar setup apply to this "
                "test.</p>"
                "<ul>"
                "<li><b>Bar material model</b>: No dispersion correction (skip this page's "
                "physics entirely), Elastic bars (standard Pochhammer-Chree), or Viscoelastic "
                "bars (accounts for a rate-dependent bar material -- needs the Viscoelastic bar "
                "model card below).</li>"
                "<li><b>Bar setup</b>: Two bars (SHPB) -- incident + reflected + transmitted "
                "waves -- or Single output bar (Direct Impact) -- transmitted wave only.</li>"
                "</ul>"
                "<p>These two choices decide which windows appear on the Pulse window(s) picker "
                "below and which other cards on this page are shown.</p>"
            ),
        )
        self.bar_model_combo = QComboBox()
        self.bar_model_combo.addItems(["No dispersion correction", "Elastic bars", "Viscoelastic bars"])
        self.bar_model_combo.setCurrentIndex(project.dispersion_correction_mode)
        self.bar_model_combo.currentIndexChanged.connect(self._rebuild_picker)
        self.bar_model_combo.currentIndexChanged.connect(self._on_mode_or_platen_changed)
        self.bar_model_combo.currentIndexChanged.connect(self._on_dispersion_mode_changed)
        mode_card.add(LabeledField("Bar material model", self.bar_model_combo))

        self.setup_combo = QComboBox()
        self.setup_combo.addItems(["Two bars (SHPB)", "Single output bar (Direct Impact)"])
        self.setup_combo.setCurrentIndex(project.dispersion_setup_mode)
        self.setup_combo.currentIndexChanged.connect(self._rebuild_picker)
        self.setup_combo.currentIndexChanged.connect(self._on_mode_or_platen_changed)
        self.setup_combo.currentIndexChanged.connect(self._on_setup_mode_changed)
        mode_card.add(LabeledField("Bar setup", self.setup_combo))
        layout.addWidget(mode_card)

        time_card = Card(
            "Signal timing",
            help_html=(
                "<p>Sets the time axis shared by every signal on this page.</p>"
                "<ul>"
                "<li><b>Time file</b>: load a time column, or check \"Generate from a time step\" "
                "and type a fixed sample interval instead.</li>"
                "</ul>"
            ),
        )
        self.time_file_row = TimeSourceRow("Time file (.txt)")
        self.time_file_row.file_selected.connect(self._on_time_file)
        self.time_file_row.dtChanged.connect(self._on_time_step_changed)
        self.time_file_row.modeChanged.connect(self._on_time_mode_changed)
        time_card.add(self.time_file_row)
        layout.addWidget(time_card)

        self.filter_card = Card(
            "Signal filtering",
            "Optional low-pass filter applied to the Input/Output Bar gage signal(s) before "
            "they reach the pulse-window picker below and the correction itself.",
            help_html=(
                "<p>Applies a zero-phase (filtfilt) Butterworth low-pass filter to the raw "
                "gage signal(s), to remove high-frequency noise before setting the pulse "
                "window(s) and running the dispersion correction below.</p>"
                "<ul>"
                "<li>Each bar has its own checkbox and cutoff frequency -- leave a box "
                "unchecked to keep that bar's signal unfiltered.</li>"
                "<li>Default cutoff: 300 kHz. The cutoff must stay below the Nyquist "
                "frequency of the loaded time vector (half its sample rate); an invalid or "
                "too-high value is simply left unfiltered rather than raising an error.</li>"
                "<li>Filtering only affects what this page (and the correction it runs) "
                "sees -- it does not change the signal shown on the Input Bar/Output Bar "
                "pages themselves.</li>"
                "</ul>"
            ),
        )
        self.ib_lowpass_check = QCheckBox("Apply low-pass filter to Input Bar signal")
        self.ib_lowpass_check.setChecked(project.input_bar.lowpass_enabled)
        self.ib_lowpass_check.toggled.connect(self._on_ib_lowpass_toggled)
        self.filter_card.add(self.ib_lowpass_check)
        self.ib_lowpass_freq_edit = QLineEdit(_fmt_num(project.input_bar.lowpass_cutoff_hz))
        self.ib_lowpass_freq_edit.editingFinished.connect(self._on_ib_lowpass_freq_changed)
        self.filter_card.add(LabeledField("Input Bar cutoff frequency (Hz)", self.ib_lowpass_freq_edit))

        self.ob_lowpass_check = QCheckBox("Apply low-pass filter to Output Bar signal")
        self.ob_lowpass_check.setChecked(project.output_bar.lowpass_enabled)
        self.ob_lowpass_check.toggled.connect(self._on_ob_lowpass_toggled)
        self.filter_card.add(self.ob_lowpass_check)
        self.ob_lowpass_freq_edit = QLineEdit(_fmt_num(project.output_bar.lowpass_cutoff_hz))
        self.ob_lowpass_freq_edit.editingFinished.connect(self._on_ob_lowpass_freq_changed)
        self.filter_card.add(LabeledField("Output Bar cutoff frequency (Hz)", self.ob_lowpass_freq_edit))
        layout.addWidget(self.filter_card)

        self.bandwidth_card = Card(
            "Correction bandwidth limit",
            "Caps the dispersion correction's OWN frequency-dependent gain above a normalized "
            "frequency f·radius/C0, so it can't amplify a real signal's high-frequency noise "
            "floor into visible ringing on the corrected signal (Result plot below). Different "
            "from -- and complementary to -- the low-pass filter above, which only touches the raw "
            "signal before correction; this instead bounds the correction itself, automatically "
            "scaled to each bar's own radius/wave speed.",
            help_html=(
                "<p>The Pochhammer-Chree correction's own gain (the amplitude-correction factor "
                "in the elastic case, the attenuation term in the viscoelastic case) is not "
                "bounded above by 1 -- at high frequency it routinely exceeds 1, since it exists "
                "to UNDO the bar's geometric dispersion, not to suppress anything. A real "
                "strain-gage signal always carries some broadband electronic/quantization noise "
                "floor, negligible at the pulse's genuine (low) frequencies but not literally "
                "zero at high ones -- a correction gain above 1 up there amplifies exactly that "
                "noise floor, which is what produces high-frequency ringing bursts on real data. "
                "This is inherent to any Pochhammer-Chree-style correction, not a bug in the root "
                "search (the Cp/C0 curve on the Result plot below stays smooth through this).</p>"
                "<ul>"
                "<li><b>Max normalized frequency (f·radius/C0)</b>: above this value (per "
                "branch, using that branch's own bar radius and wave speed -- exactly the "
                "quantity plotted on the Result panel's Cp/C0 curve's own x-axis), the "
                "correction's gain is smoothly rolled off to 0 rather than left unbounded. "
                "Default 0.6 is a practical starting point, not a proven optimum for every bar/"
                "pulse combination -- lower it if ringing persists, raise it only if you have a "
                "specific reason to trust the correction further out.</li>"
                "<li>Applies to both the elastic and viscoelastic corrections, two-bar (SHPB) and "
                "Direct Impact alike.</li>"
                "<li>Unticking the checkbox restores the original, unbounded behavior.</li>"
                "</ul>"
            ),
        )
        self.bandwidth_limit_check = QCheckBox("Limit correction bandwidth")
        self.bandwidth_limit_check.setChecked(project.dispersion_bandwidth_limit_enabled)
        self.bandwidth_limit_check.toggled.connect(self._on_bandwidth_limit_toggled)
        self.bandwidth_card.add(self.bandwidth_limit_check)
        self.bandwidth_limit_edit = QLineEdit(_fmt_num(project.dispersion_bandwidth_limit))
        self.bandwidth_limit_edit.editingFinished.connect(self._on_bandwidth_limit_value_changed)
        self.bandwidth_card.add(LabeledField("Max normalized frequency (f·radius/C0)", self.bandwidth_limit_edit))
        layout.addWidget(self.bandwidth_card)

        self.platen_card = Card(
            "Platens",
            "Only supported for the elastic, two-bar (SHPB) correction (matching the original "
            "FFT_dispersion.py). Shifts the reflected/transmitted window timing to account for "
            "wave travel through the platens.",
            help_html=(
                "<p>Only shown for the elastic, two-bar (SHPB) correction. Accounts for platens "
                "(intermediate pieces between bars and specimen) by shifting the reflected/"
                "transmitted window timing for the extra wave-travel time through them.</p>"
                "<ul>"
                "<li>Check <b>Use platens</b>, then load a <b>Platen config file</b> with their "
                "properties (also editable directly in the table below).</li>"
                "</ul>"
            ),
        )
        self.plat_check = QCheckBox("Use platens")
        self.plat_check.setChecked(project.shpb.plat_check)
        self.plat_check.toggled.connect(self._on_mode_or_platen_changed)
        self.plat_check.toggled.connect(self._on_plat_check_toggled)
        self.platen_card.add(self.plat_check)
        self.platen_file_row = FileSelectRow("Platen config file")
        self.platen_file_row.file_selected.connect(self._on_platen_file)
        self.platen_card.add(self.platen_file_row)
        self.platen_status = QLabel("")
        self.platen_status.setObjectName("cardSubtitle")
        self.platen_card.add(self.platen_status)
        self.platen_table = MCParamTable([project.shpb.platen.params[k] for k in PLATEN_PARAM_ORDER])
        self.platen_card.add(self.platen_table)
        layout.addWidget(self.platen_card)

        self.visco_card = Card(
            "Viscoelastic bar model",
            "Only used for the \"Viscoelastic bars\" correction. Pick the same constitutive model "
            "used in the Viscoelastic Model Calibration tool (Standalone Tools) and determine its "
            "parameters there rather than guessing -- its \"Apply to Dispersion Corrections\" button "
            "fills in the model choice and parameters here automatically.",
            help_html=(
                "<p>Only used for the \"Viscoelastic bars\" correction -- defines the "
                "rate-dependent bar material model the correction physics needs.</p>"
                "<ul>"
                "<li><b>Poisson ratio / Damping factor</b>: the damping factor can be measured "
                "with the Free-Decay Damping tool (\"Use value from Free-Decay Damping tool\" "
                "there).</li>"
                "<li>Pick a constitutive model from the grid, then fill in its parameters below.</li>"
                "</ul>"
                "<p><b>Recommended workflow:</b> determine these parameters in the Viscoelastic "
                "Model Calibration tool (Standalone Tools) rather than guessing them here -- its "
                "\"Apply to Dispersion Corrections\" button fills in the model choice and every "
                "parameter here automatically.</p>"
            ),
        )
        self.visco_nu_edit = QLineEdit(str(project.viscoelastic.poisson_ratio))
        self.visco_nu_edit.editingFinished.connect(self._on_visco_params_changed)
        self.visco_card.add(LabeledField("Poisson ratio (-)", self.visco_nu_edit))
        self.visco_damp_edit = QLineEdit(str(project.viscoelastic.damping_factor))
        self.visco_damp_edit.editingFinished.connect(self._on_visco_params_changed)
        self.visco_card.add(LabeledField("Damping factor (rad/m) -- from Free-Decay Damping tool", self.visco_damp_edit))

        self.visco_model_buttons: dict[str, RheologyModelButton] = {}
        self.visco_model_button_group = QButtonGroup(self)
        self.visco_model_button_group.setExclusive(True)
        visco_model_grid = QGridLayout()
        visco_model_grid.setSpacing(12)
        for i, key in enumerate(MODEL_ORDER):
            model = MODELS[key]
            btn = RheologyModelButton(key, model.display_name, model.short_formula)
            btn.setChecked(key == project.viscoelastic.model_key)
            btn.clicked.connect(lambda _checked, k=key: self._on_visco_model_clicked(k))
            self.visco_model_button_group.addButton(btn)
            self.visco_model_buttons[key] = btn
            visco_model_grid.addWidget(btn, i // 2, i % 2)
        self.visco_card.add_layout(visco_model_grid)
        self.visco_model_formula_label = QLabel(get_model(project.viscoelastic.model_key).formula_html)
        self.visco_model_formula_label.setObjectName("cardSubtitle")
        self.visco_model_formula_label.setWordWrap(True)
        self.visco_card.add(self.visco_model_formula_label)

        self.visco_params_layout = QVBoxLayout()
        self.visco_card.add_layout(self.visco_params_layout)
        self.visco_param_edits: dict[str, QLineEdit] = {}
        self._rebuild_visco_param_fields()
        layout.addWidget(self.visco_card)

        self.pulse_card = Card(
            "Pulse window(s)",
            "Click and drag on the signal below to set the selected window (pick which one "
            "from the dropdown first). For the viscoelastic, two-bar case, cut the incident "
            "window by hand, then use \"Auto-cut reflected & transmitted\" below to place the "
            "other two from it -- dragging them afterward still overrides the estimate.",
            help_html=(
                "<p>Sets the pulse window(s) the correction actually runs on (see the plot's own "
                "help below for how to drag them).</p>"
                "<ul>"
                "<li><b>Auto-cut reflected &amp; transmitted (from model)</b>: only for the "
                "viscoelastic, two-bar (SHPB) case -- estimates the reflected and transmitted "
                "windows FROM the incident window already cut, using the selected rheological "
                "model's own phase velocity instead of a single constant C0. You can still drag "
                "either window afterward to override the estimate.</li>"
                "</ul>"
            ),
        )
        self.autocut_row = QHBoxLayout()
        self.autocut_button = QPushButton("Auto-cut reflected & transmitted (from model)")
        self.autocut_button.setToolTip(
            "Only for the viscoelastic, two-bar (SHPB) case. Estimates the reflected and "
            "transmitted windows FROM the incident window you've already cut, using the "
            "selected rheological model's own phase velocity instead of a single constant C0 "
            "-- much better than guessing by eye when the bar's wave speed is strongly "
            "frequency-dependent. You can still drag either window afterward to override it."
        )
        self.autocut_button.clicked.connect(self._on_autocut)
        self.autocut_row.addWidget(self.autocut_button)
        self.autocut_row.addStretch(1)
        # NOTE: NOT added to pulse_card here -- `_rebuild_picker` (called
        # just below, and again on every mode change) clears pulse_card's
        # body down to the picker each time, which would silently detach
        # this row on the very first call if it were added before that.
        # `_rebuild_picker` re-adds `self.autocut_row` itself, after the
        # picker, every time it rebuilds.
        layout.addWidget(self.pulse_card)
        self.picker = None
        self._autocut_thread = None
        self._autocut_worker = None
        self._rebuild_picker()
        self._on_mode_or_platen_changed()

        run_card = Card("Run")
        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run correction")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._on_run)
        run_row.addWidget(self.run_button)
        run_row.addStretch(1)
        run_card.add_layout(run_row)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        run_card.add(self.progress_bar)
        self.status_label = QLabel("")
        self.status_label.setObjectName("cardSubtitle")
        run_card.add(self.status_label)
        layout.addWidget(run_card)

        self.result_card = Card("Result")
        self.canvas = PlotCanvas(
            figsize=(9, 5),
            help_title="Correction result",
            help_text=(
                "<p>Shows the signal(s) after dispersion correction, once <b>Run correction</b> "
                "has finished: the pulse propagated back to the bar/specimen interface, with the "
                "high-frequency distortion introduced by travel along the bar removed (Pochhammer-"
                "Chree, elastic or viscoelastic).</p>"
                "<p>Compare against the raw pulse window(s) set above to sanity-check the "
                "correction -- a corrected pulse with a cleaner rise/plateau and no spurious "
                "oscillation is the expected result.</p>"
            ),
        )
        self.result_card.add(self.canvas)
        layout.addWidget(self.result_card)

        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._restore_from_session()

    def _restore_from_session(self):
        """Re-load the time file remembered from a previous session (see
        core/session.py) -- it lives on this page, not on `Project`, so
        it's just re-read through the same handler used when picking it
        live."""
        from pathlib import Path
        path = self.project.last_paths.get("corrections_time")
        if path and Path(path).exists():
            self._on_time_file(path)

        dt = self.project.time_step_overrides.get("corrections_time")
        if dt is not None:
            self.time_file_row.set_dt(dt)
            self.time_file_row.set_generate_mode(True)  # regenerates now if an Input/Output Bar signal is already loaded, or on the next showEvent otherwise

    # -- picker setup, depends on elastic/viscoelastic x SHPB/DI ----------

    def _picker_window_defs(self):
        direct_impact = self.setup_combo.currentIndex() == 1
        viscoelastic = self.bar_model_combo.currentIndex() == 2
        if direct_impact:
            return [("xd1", "Transmitted window", theme.ACCENT)]
        if viscoelastic:
            return [
                ("xd1", "Incident", theme.ACCENT),
                ("xd3", "Reflected", "#ff9f43"),
                ("xd5", "Transmitted", "#4cd97b"),
            ]
        return [("xd1", "Incident window", theme.ACCENT)]

    def _rebuild_picker(self, *_args):
        # Shallow clear (takeAt without recursing into nested layouts) --
        # deliberately does NOT delete `self.autocut_row`'s button, since
        # that row is a fixed, reusable widget re-added below on every
        # call, not something rebuilt per mode like the picker itself.
        while self.pulse_card.body.count() > 0:
            item = self.pulse_card.body.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.picker = TimeWindowPicker(
            self._picker_window_defs(),
            help_title="Pulse window(s)",
            help_text=(
                "<p>Shows the raw bar signal(s) used by the dispersion correction.</p>"
                "<ul>"
                "<li>Select which window to set from the dropdown above the plot (e.g. Incident/"
                "Reflected/Transmitted, or just Transmitted for Direct Impact).</li>"
                "<li>Click and drag directly on the trace to place its start/end -- each window is "
                "drawn as a distinct shaded, labeled span so all of them stay visible for "
                "comparison, and dragging a window again just moves it.</li>"
                "</ul>"
                "<p>For the viscoelastic two-bar case, cut the Incident window by hand first, then "
                "\"Auto-cut reflected &amp; transmitted\" (below the plot) can place the other two "
                "automatically from the selected rheological model.</p>"
            ),
        )
        self.picker.windowChanged.connect(self._on_corrections_window_changed)
        self.pulse_card.add(self.picker)
        self.pulse_card.add_layout(self.autocut_row)
        self._refresh_picker_signal()

    def _on_dispersion_mode_changed(self, index):
        self.project.dispersion_correction_mode = index

    def _on_setup_mode_changed(self, index):
        self.project.dispersion_setup_mode = index

    def _on_plat_check_toggled(self, checked):
        self.project.shpb.plat_check = checked

    def _on_ib_lowpass_toggled(self, checked):
        self.project.input_bar.lowpass_enabled = checked
        self._refresh_picker_signal()

    def _on_ob_lowpass_toggled(self, checked):
        self.project.output_bar.lowpass_enabled = checked
        self._refresh_picker_signal()

    def _on_ib_lowpass_freq_changed(self):
        try:
            freq = float(self.ib_lowpass_freq_edit.text())
        except ValueError:
            self.status_label.setText("Input Bar cutoff frequency must be numeric.")
            return
        if freq <= 0:
            self.status_label.setText("Input Bar cutoff frequency must be positive.")
            return
        self.project.input_bar.lowpass_cutoff_hz = freq
        self._refresh_picker_signal()

    def _on_ob_lowpass_freq_changed(self):
        try:
            freq = float(self.ob_lowpass_freq_edit.text())
        except ValueError:
            self.status_label.setText("Output Bar cutoff frequency must be numeric.")
            return
        if freq <= 0:
            self.status_label.setText("Output Bar cutoff frequency must be positive.")
            return
        self.project.output_bar.lowpass_cutoff_hz = freq
        self._refresh_picker_signal()

    def _on_bandwidth_limit_toggled(self, checked):
        self.project.dispersion_bandwidth_limit_enabled = checked

    def _on_bandwidth_limit_value_changed(self):
        try:
            value = float(self.bandwidth_limit_edit.text())
        except ValueError:
            self.status_label.setText("Max normalized frequency must be numeric.")
            return
        if value <= 0:
            self.status_label.setText("Max normalized frequency must be positive.")
            return
        self.project.dispersion_bandwidth_limit = value

    def _max_normalized_freq_for_run(self) -> float:
        """The `max_normalized_freq` kwarg to pass to whichever
        core.dispersion.*_correction* function `_on_run` calls -- 0.0
        (disabled, unbounded gain, the original ported behavior) unless
        the "Limit correction bandwidth" checkbox is ticked."""
        if not self.bandwidth_limit_check.isChecked():
            return 0.0
        return self.project.dispersion_bandwidth_limit

    def _on_mode_or_platen_changed(self, *_args):
        supports_platen = self.setup_combo.currentIndex() == 0 and self.bar_model_combo.currentIndex() == 1
        self.platen_card.setVisible(supports_platen)
        viscoelastic = self.bar_model_combo.currentIndex() == 2
        self.visco_card.setVisible(viscoelastic)
        # Auto-cut only makes sense where there ARE separate reflected/
        # transmitted windows to place, i.e. the viscoelastic two-bar case
        # (Direct Impact only ever has a single "Transmitted window").
        self.autocut_button.setVisible(viscoelastic and self.setup_combo.currentIndex() == 0)

    def _on_visco_model_clicked(self, key):
        if self.project.viscoelastic.model_key == key:
            return
        model = MODELS[key]
        self.project.viscoelastic.model_key = key
        # A model switch invalidates whatever params were on file for the
        # PREVIOUS model (different keys, different physical meaning) --
        # reset to that model's own defaults rather than keep stale values
        # around under mismatched keys.
        self.project.viscoelastic.params = {p.key: p.fixed_default for p in model.params}
        self.project.viscoelastic.params_std = {}
        self.visco_model_formula_label.setText(model.formula_html)
        self._rebuild_visco_param_fields()

    def _rebuild_visco_param_fields(self):
        _clear_layout(self.visco_params_layout)
        self.visco_param_edits = {}
        model = get_model(self.project.viscoelastic.model_key)
        current_params = self.project.viscoelastic.params
        for p in model.params:
            value = current_params.get(p.key, p.fixed_default)
            edit = QLineEdit(_fmt_num(value))
            edit.editingFinished.connect(self._on_visco_params_changed)
            self.visco_param_edits[p.key] = edit
            row = LabeledField(p.label, edit)
            self.visco_params_layout.addWidget(row)

    def _on_visco_params_changed(self):
        try:
            nu = float(self.visco_nu_edit.text())
            damping = float(self.visco_damp_edit.text())
            params = {key: float(edit.text()) for key, edit in self.visco_param_edits.items()}
        except ValueError:
            return  # leave previous values in place
        self.project.viscoelastic.poisson_ratio = nu
        self.project.viscoelastic.damping_factor = damping
        self.project.viscoelastic.params = params

    def refresh_visco_fields(self):
        """Called after the Viscoelastic Model Calibration tool pushes new
        values (model choice included) into project.viscoelastic, so this
        page reflects them."""
        v = self.project.viscoelastic
        self.visco_nu_edit.setText(str(v.poisson_ratio))
        self.visco_damp_edit.setText(str(v.damping_factor))
        for key, btn in self.visco_model_buttons.items():
            btn.setChecked(key == v.model_key)
        self.visco_model_formula_label.setText(get_model(v.model_key).formula_html)
        self._rebuild_visco_param_fields()

    def _on_platen_file(self, path):
        from core.param_io import load_param_file
        try:
            load_param_file(path, self.project.shpb.platen.params, PLATEN_PARAM_ORDER)
        except (OSError, ValueError, SyntaxError) as exc:
            self.platen_status.setText(f"Could not load platen config: {exc}")
            return
        self.project.last_paths["platen_config"] = path
        self.platen_table.refresh()
        self.platen_status.setText(f"Loaded from {path.split('/')[-1]}")

    def _on_time_file(self, path):
        try:
            values = [float(line.strip()) for line in open(path) if line.strip() and not line.strip().startswith("#")]
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Could not read time file: {exc}")
            return
        if self.project.downsample_enabled:
            values = list(downsample(values, self.project.downsample_n))
        self._timet = np.asarray(values)
        # Kept at full float precision, not rounded to 6 decimals: that's
        # only microsecond precision, and a bar-gage DAQ commonly samples
        # faster than 1 MHz (dt < 1e-6 s), which would collapse consecutive
        # timestamps to the same rounded value -- see _maybe_regenerate_time's
        # comment below for how that breaks a generated (not file-loaded)
        # time vector.
        self._time_pulse = [float(v) for v in values]
        self.project.last_paths["corrections_time"] = path
        self.status_label.setText(f"Loaded {len(values)} time samples from {path.split('/')[-1]}")
        self._refresh_picker_signal()

    def _on_time_step_changed(self, dt):
        """Alternative to `_on_time_file` above: build the time vector from
        a fixed time step matching the relevant bar signal's length, instead
        of loading it from a file. Those signals are loaded on the separate
        Input Bar / Output Bar pages, not here, so `_maybe_regenerate_time`
        is also called from `showEvent` so a stale generated vector self-
        corrects as soon as this page is revisited."""
        self._maybe_regenerate_time()
        if self._timet is None:
            self.status_label.setText(
                "Load an Input Bar and/or Output Bar signal first (their own pages), so the "
                "generated time vector can match its length."
            )
            return
        self.project.last_paths.pop("corrections_time", None)
        self.project.time_step_overrides["corrections_time"] = dt
        self.status_label.setText(f"Generated {len(self._timet)} time samples (dt = {dt:.6g} s).")
        self._refresh_picker_signal()

    def _on_time_mode_changed(self, generate):
        if not generate:
            self.project.time_step_overrides.pop("corrections_time", None)

    def _maybe_regenerate_time(self):
        if not self.time_file_row.is_generate_mode():
            return
        direct_impact = self.setup_combo.currentIndex() == 1
        if direct_impact:
            n = len(self.project.output_bar.signal)
        else:
            n = len(self.project.input_bar.signal) or len(self.project.output_bar.signal)
        if n == 0:
            return
        generated = self.time_file_row.build(n)
        if generated is not None:
            self._timet = generated
            # Kept at full float precision -- see _on_time_file's comment above.
            # A small time step (e.g. 2e-7 s, a common 5 MHz bar-gage sample
            # rate) rounded to microsecond precision would collapse several
            # consecutive generated samples to the exact same timestamp,
            # which would make `_robust_dt` (core/process_shpb.py) raise
            # "too many samples ... share the same timestamp" for a
            # perfectly valid, uniformly-spaced generated time vector.
            self._time_pulse = [float(v) for v in generated]

    def _dt_for_filter(self) -> float | None:
        """Sample interval for `Bar.filtered_signal()`, from whichever
        time reference is currently loaded (the two-bar `self._timet`, or
        the Direct-Impact `self._time_pulse`) -- same `np.median(np.diff(
        ...))` idiom `core/dispersion.py`/`core/process_shpb.py`'s own
        `_robust_dt` already use to get a sample rate from a time
        vector."""
        time_ref = self._timet if self._timet is not None else self._time_pulse
        if time_ref is None or len(time_ref) < 2:
            return None
        return float(np.median(np.diff(np.asarray(time_ref, dtype=float))))

    def _filtered_bars_for_run(self):
        """(ib, ob) `Bar` objects to hand to `core.dispersion.*` for a run
        (correction or auto-cut) -- plain `project.input_bar`/`output_bar`
        unless that bar's low-pass filter is enabled, in which case a
        shallow copy with `.signal` replaced by the filtered array is
        used instead (`dataclasses.replace`, so every other field --
        params, sign/offset, the Monte-Carlo-expanded `*_pa` arrays
        `project.expand_montecarlo()` just populated -- stays shared/
        untouched). A COPY, never an in-place mutation of the live Bar:
        the correction/auto-cut run happens on a QThread worker started
        just after this, so mutating `project.input_bar.signal` here and
        restoring it once the worker signals completion would race the
        worker actually reading it; a copy sidesteps that entirely, and
        also means project.input_bar/output_bar are never left altered
        for any other page (e.g. the Input Bar/Output Bar pages' own
        preview) even transiently."""
        project = self.project
        dt = self._dt_for_filter()
        ib = project.input_bar
        ob = project.output_bar
        if ib.lowpass_enabled and len(ib.signal) > 0:
            ib = dataclasses.replace(ib, signal=ib.filtered_signal(dt))
        if ob.lowpass_enabled and len(ob.signal) > 0:
            ob = dataclasses.replace(ob, signal=ob.filtered_signal(dt))
        return ib, ob

    def _refresh_picker_signal(self):
        if self.picker is None:
            return
        if self._timet is None:
            self.picker.show_placeholder("Load a time file above to see the signal(s) here.")
            return
        direct_impact = self.setup_combo.currentIndex() == 1
        project = self.project
        dt = self._dt_for_filter()
        if direct_impact:
            if len(project.output_bar.signal) == 0:
                self.picker.show_placeholder(
                    "No Output Bar signal loaded yet.\nGo to the Output Bar page, load its signal file, then come back here."
                )
                return
            self.picker.set_signal(self._timet, [("Output Bar signal", project.output_bar.filtered_signal(dt), "#4cd97b")])
        else:
            series = []
            if len(project.input_bar.signal) > 0:
                series.append(("Input Bar signal", project.input_bar.filtered_signal(dt), theme.ACCENT))
            if len(project.output_bar.signal) > 0:
                series.append(("Output Bar signal", project.output_bar.filtered_signal(dt), "#4cd97b"))
            if not series:
                self.picker.show_placeholder(
                    "No Input/Output Bar signal loaded yet.\nGo to the Input Bar and/or Output Bar page, "
                    "load their signal files, then come back here."
                )
                return
            self.picker.set_signal(self._timet, series)

        # `project.corrections_windows` is one flat dict shared across
        # every Correction type/setup combination (e.g. the viscoelastic
        # two-bar case's "Incident" window and the Direct-Impact case's
        # single "Transmitted" window are BOTH stored under key "xd1").
        # Restoring every saved key unconditionally could hand the CURRENT
        # picker a key it doesn't define (crashing in `_draw_span`) or,
        # worse, a value that's simply stale for a different dataset/window
        # scope -- `_on_run`'s nearest-sample snapping (Direct Impact) or a
        # plain out-of-range slice (two-bar) would then silently collapse it
        # to an empty window several steps later, deep inside
        # core/dispersion.py, with a cryptic "index 1 is out of bounds for
        # axis 0 with size 0" instead of any indication the real problem was
        # a leftover value from a different mode. Skipping anything that
        # isn't both a key this picker actually defines AND within the
        # currently loaded signal's own time range keeps a stale value from
        # ever reaching that point.
        valid_keys = self.picker.window_keys()
        t_min, t_max = float(self._timet[0]), float(self._timet[-1])
        for key, value in self.project.corrections_windows.items():
            if key not in valid_keys or not value or len(value) != 2:
                continue
            start, end = value
            if start < t_min or end > t_max or end <= start:
                continue
            self.picker.set_window(key, start, end)

    def _on_corrections_window_changed(self, key, start, end):
        self.project.corrections_windows[key] = [start, end]

    # -- auto-cut: places reflected/transmitted windows from the incident
    # one, using the viscoelastic model's own (frequency-dependent) phase
    # velocity instead of a single constant C0 -- see
    # core/dispersion.py's `estimate_viscoelastic_windows`. Manual dragging
    # of any window (including re-dragging one this fills in) remains the
    # backup/override path -- this never removes that ability. ----------

    def _on_autocut(self):
        if self._timet is None:
            self.status_label.setText("Load a time file first.")
            return
        if len(self.project.input_bar.signal) == 0 or len(self.project.output_bar.signal) == 0:
            self.status_label.setText("Load both the Input Bar and Output Bar signals first.")
            return
        window_xd1 = self.picker.get_window("xd1") if self.picker else None
        if window_xd1 is None:
            self.status_label.setText(
                "Cut the incident window first (drag it on the signal above) -- the auto-cut "
                "estimates the other two windows FROM it."
            )
            return

        project = self.project
        project.expand_montecarlo()
        ib_for_run, ob_for_run = self._filtered_bars_for_run()
        kwargs = dict(
            ib=ib_for_run, ob=ob_for_run, specimen=project.specimen, timet=self._timet,
            xd1=window_xd1[0], xd2=window_xd1[1], nu=project.viscoelastic.poisson_ratio,
            model=get_model(project.viscoelastic.model_key), params=dict(project.viscoelastic.params),
        )
        self.autocut_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.status_label.setText(
            "Estimating the reflected/transmitted windows from the incident window and the "
            "selected rheological model's phase velocity…"
        )
        self.progress_bar.setValue(0)

        self._autocut_thread = QThread()
        self._autocut_worker = _CorrectionWorker(dispersion.estimate_viscoelastic_windows, kwargs)
        self._autocut_worker.moveToThread(self._autocut_thread)
        self._autocut_thread.started.connect(self._autocut_worker.run)
        self._autocut_worker.progress.connect(self._on_progress)
        self._autocut_worker.finished.connect(self._on_autocut_finished)
        self._autocut_worker.failed.connect(self._on_autocut_failed)
        self._autocut_worker.finished.connect(self._autocut_thread.quit)
        self._autocut_worker.failed.connect(self._autocut_thread.quit)
        self._autocut_thread.start()

    def _on_autocut_finished(self, windows):
        self.autocut_button.setEnabled(True)
        self.run_button.setEnabled(True)
        xd3, xd4, xd5, xd6 = windows
        self.picker.set_window("xd3", xd3, xd4)
        self.picker.set_window("xd5", xd5, xd6)
        self.project.corrections_windows["xd3"] = [xd3, xd4]
        self.project.corrections_windows["xd5"] = [xd5, xd6]
        self.status_label.setText(
            "Reflected and transmitted windows estimated from the incident window and the "
            "model's phase velocity. You can still drag either one to fine-tune it by hand."
        )

    def _on_autocut_failed(self, message):
        self.autocut_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.status_label.setText(
            f"Auto-cut failed: {message}. You can still cut the reflected/transmitted windows manually."
        )

    def showEvent(self, event):
        super().showEvent(event)
        self._maybe_regenerate_time()  # Input/Output Bar signals may have (re)loaded since this page was last shown
        self._refresh_picker_signal()
        self.refresh_visco_fields()
        self.platen_table.refresh()  # picks up values changed elsewhere (e.g. a loaded Setup Profile)
        if self.bar_model_combo.currentIndex() != self.project.dispersion_correction_mode:
            self.bar_model_combo.setCurrentIndex(self.project.dispersion_correction_mode)
        if self.plat_check.isChecked() != self.project.shpb.plat_check:
            self.plat_check.setChecked(self.project.shpb.plat_check)
        if self.setup_combo.currentIndex() != self.project.dispersion_setup_mode:
            self.setup_combo.setCurrentIndex(self.project.dispersion_setup_mode)
        # picks up values changed elsewhere (e.g. a loaded Setup Profile)
        if self.ib_lowpass_check.isChecked() != self.project.input_bar.lowpass_enabled:
            self.ib_lowpass_check.setChecked(self.project.input_bar.lowpass_enabled)
        self.ib_lowpass_freq_edit.setText(_fmt_num(self.project.input_bar.lowpass_cutoff_hz))
        if self.ob_lowpass_check.isChecked() != self.project.output_bar.lowpass_enabled:
            self.ob_lowpass_check.setChecked(self.project.output_bar.lowpass_enabled)
        self.ob_lowpass_freq_edit.setText(_fmt_num(self.project.output_bar.lowpass_cutoff_hz))
        if self.bandwidth_limit_check.isChecked() != self.project.dispersion_bandwidth_limit_enabled:
            self.bandwidth_limit_check.setChecked(self.project.dispersion_bandwidth_limit_enabled)
        self.bandwidth_limit_edit.setText(_fmt_num(self.project.dispersion_bandwidth_limit))
        self._display_restored_correction()

    def _on_run(self):
        project = self.project
        mode_index = self.bar_model_combo.currentIndex()  # 0: none, 1: elastic, 2: viscoelastic
        no_correction_selected = mode_index == 0
        viscoelastic = mode_index == 2
        direct_impact = self.setup_combo.currentIndex() == 1

        window_xd1 = self.picker.get_window("xd1") if self.picker else None
        if window_xd1 is None:
            self.status_label.setText("Select the pulse window(s) by dragging on the signal plot first.")
            return
        xd = {"xd1": window_xd1[0], "xd2": window_xd1[1]}
        if not direct_impact and viscoelastic:
            window_xd3 = self.picker.get_window("xd3")
            window_xd5 = self.picker.get_window("xd5")
            if window_xd3 is None or window_xd5 is None:
                self.status_label.setText("Select all three pulse windows (Incident/Reflected/Transmitted) first.")
                return
            xd.update(xd3=window_xd3[0], xd4=window_xd3[1], xd5=window_xd5[0], xd6=window_xd5[1])

        project.expand_montecarlo()
        ib_for_run, ob_for_run = self._filtered_bars_for_run()
        sg_unc_d = np.array([1.0])  # strain-gage uncertainty coefficient placeholder (see note below)

        if direct_impact and no_correction_selected:
            if self._time_pulse is None:
                self.status_label.setText("Load a time file first.")
                return
            xd1 = min(self._time_pulse, key=lambda t: abs(t - xd["xd1"]))
            xd2 = min(self._time_pulse, key=lambda t: abs(t - xd["xd2"]))
            fn = dispersion.no_correction_direct_impact
            kwargs = dict(ob=ob_for_run, specimen=project.specimen, xd1=xd1, xd2=xd2,
                          sg_unc_d=sg_unc_d, time_pulse=self._time_pulse)
        elif direct_impact and not viscoelastic:
            if self._time_pulse is None:
                self.status_label.setText("Load a time file first.")
                return
            xd1 = min(self._time_pulse, key=lambda t: abs(t - xd["xd1"]))
            xd2 = min(self._time_pulse, key=lambda t: abs(t - xd["xd2"]))
            fn = dispersion.elastic_correction_direct_impact
            kwargs = dict(ob=ob_for_run, specimen=project.specimen, xd1=xd1, xd2=xd2,
                          sg_unc_d=sg_unc_d, time_pulse=self._time_pulse,
                          heat_check=project.thermal.heat_correction_enabled,
                          type_bars_check=project.thermal.bar_type,
                          ambient_temp=project.thermal.ambient_temperature_k, t0=project.thermal.initial_temperature_k,
                          tm_value=(project.thermal.melting_temperature_k, project.thermal.melting_exponent),
                          table_ob=project.material_table_ob,
                          n_workers=project.n_workers,
                          max_normalized_freq=self._max_normalized_freq_for_run())
        elif direct_impact and viscoelastic:
            if self._time_pulse is None:
                self.status_label.setText("Load a time file first.")
                return
            xd1 = min(self._time_pulse, key=lambda t: abs(t - xd["xd1"]))
            xd2 = min(self._time_pulse, key=lambda t: abs(t - xd["xd2"]))
            fn = dispersion.viscoelastic_correction_direct_impact
            kwargs = dict(ob=ob_for_run, specimen=project.specimen, xd1=xd1, xd2=xd2,
                          sg_unc_d=sg_unc_d, time_pulse=self._time_pulse,
                          nu=project.viscoelastic.poisson_ratio, damping=project.viscoelastic.damping_factor,
                          model=get_model(project.viscoelastic.model_key), params=dict(project.viscoelastic.params),
                          n_workers=project.n_workers,
                          max_normalized_freq=self._max_normalized_freq_for_run())
        elif not direct_impact and no_correction_selected:
            if self._timet is None:
                self.status_label.setText("Load a time file first.")
                return
            fn = dispersion.no_correction
            kwargs = dict(ib=ib_for_run, ob=ob_for_run, specimen=project.specimen, timet=self._timet,
                          xd1=xd["xd1"], xd2=xd["xd2"], sg_unc_d=sg_unc_d)
        elif not direct_impact and not viscoelastic:
            if self._timet is None:
                self.status_label.setText("Load a time file first.")
                return
            fn = dispersion.elastic_correction
            plat_check = self.plat_check.isChecked()
            plat_arrays = project.shpb.platen.expand(1) if plat_check else {}
            kwargs = dict(ib=ib_for_run, ob=ob_for_run, specimen=project.specimen, timet=self._timet,
                          xd1=xd["xd1"], xd2=xd["xd2"], sg_unc_d=sg_unc_d,
                          plat_check=plat_check, plat_emod=plat_arrays.get("e"), plat_rho=plat_arrays.get("rho"),
                          plat_len=plat_arrays.get("L"),
                          heat_check=project.thermal.heat_correction_enabled,
                          type_bars_check=project.thermal.bar_type,
                          ambient_temp=project.thermal.ambient_temperature_k, t0=project.thermal.initial_temperature_k,
                          tm_value=(project.thermal.melting_temperature_k, project.thermal.melting_exponent),
                          table_ib=project.material_table_ib, table_ob=project.material_table_ob,
                          n_workers=project.n_workers,
                          max_normalized_freq=self._max_normalized_freq_for_run())
        else:
            if self._timet is None:
                self.status_label.setText("Load a time file first.")
                return
            fn = dispersion.viscoelastic_correction
            kwargs = dict(ib=ib_for_run, ob=ob_for_run, specimen=project.specimen, timet=self._timet,
                          xd1=xd["xd1"], xd2=xd["xd2"], xd3=xd["xd3"], xd4=xd["xd4"], xd5=xd["xd5"], xd6=xd["xd6"],
                          sg_unc_d=sg_unc_d, nu=project.viscoelastic.poisson_ratio,
                          damping=project.viscoelastic.damping_factor,
                          model=get_model(project.viscoelastic.model_key), params=dict(project.viscoelastic.params),
                          n_workers=project.n_workers,
                          max_normalized_freq=self._max_normalized_freq_for_run())

        self.run_button.setEnabled(False)
        self.autocut_button.setEnabled(False)
        self._pending_meta = {
            "direct_impact": direct_impact, "mode_index": mode_index,
            "xd1": kwargs["xd1"], "xd2": kwargs["xd2"],
            "timet": self._timet, "time_pulse": self._time_pulse,
        }
        if no_correction_selected:
            self.status_label.setText("Cutting windows (no dispersion correction)…")
        else:
            self.status_label.setText("Running dispersion correction… (this involves a root-search per frequency and can take a while)")
        self.progress_bar.setValue(0)

        self._thread = QThread()
        self._worker = _CorrectionWorker(fn, kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _on_progress(self, i, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(i)

    def _on_finished(self, result):
        self.run_button.setEnabled(True)
        self.autocut_button.setEnabled(True)
        self.status_label.setText("Done. This corrected signal is now available to the Processing page "
                                   "(\"Use dispersion-corrected signal\" option).")
        self.project.last_correction = {**self._pending_meta, "result": result}
        self.canvas.clear()
        ax1 = self.canvas.figure.add_subplot(211)
        if result.incident is not None:
            ax1.plot(result.time, result.incident, "r:", label="Incident (gage)")
            ax1.plot(result.time, result.incident_corrected, "r-", label="Incident (corrected)")
            ax1.plot(result.time, result.reflected, "b:", label="Reflected (gage)")
            ax1.plot(result.time, result.reflected_corrected, "b-", label="Reflected (corrected)")
        ax1.plot(result.time, result.transmitted, "g:", label="Transmitted (gage)")
        ax1.plot(result.time, result.transmitted_corrected, "g-", label="Transmitted (corrected)")
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("amplitude")
        ax1.legend(fontsize=8)

        ax2 = self.canvas.figure.add_subplot(212)
        for wave, values in result.phase_velocity_ratio.items():
            ax2.plot(result.freq_over_c[wave], values, ".", label=wave, markersize=3)
        ax2.set_xlabel("f·a/C0")
        ax2.set_ylabel("Cp/C0")
        ax2.legend(fontsize=8)
        self.canvas.refresh()

    def _display_restored_correction(self):
        """Shows the last correction's waveforms on the plot, without
        re-running anything -- used when `project.last_correction` was
        restored from a session/profile/test-export JSON rather than
        computed just now. The phase-velocity-ratio diagnostic subplot
        isn't shown here since that data isn't saved in the JSON (it's a
        computational diagnostic, not needed to use the corrected
        signal) -- only the waveforms, which ARE what the Processing
        page's "Use dispersion-corrected signal" option actually uses.
        """
        corr = self.project.last_correction
        if corr is None or corr.get("result") is None:
            return
        result = corr["result"]
        self.canvas.clear()
        ax1 = self.canvas.figure.add_subplot(111)
        if result.incident is not None:
            ax1.plot(result.time, result.incident, "r:", label="Incident (gage)")
            ax1.plot(result.time, result.incident_corrected, "r-", label="Incident (corrected)")
            ax1.plot(result.time, result.reflected, "b:", label="Reflected (gage)")
            ax1.plot(result.time, result.reflected_corrected, "b-", label="Reflected (corrected)")
        ax1.plot(result.time, result.transmitted, "g:", label="Transmitted (gage)")
        ax1.plot(result.time, result.transmitted_corrected, "g-", label="Transmitted (corrected)")
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("amplitude")
        ax1.legend(fontsize=8)
        self.canvas.refresh()
        mode_label = {0: "no correction", 1: "elastic", 2: "viscoelastic"}.get(corr.get("mode_index"), "?")
        self.status_label.setText(
            f"Restored from a previous session/profile: {mode_label} correction, window "
            f"{corr.get('xd1', 0):.6g} s -> {corr.get('xd2', 0):.6g} s. Available to the Processing page "
            "(\"Use dispersion-corrected signal\" option) without re-running."
        )

    def _on_failed(self, message):
        self.run_button.setEnabled(True)
        self.autocut_button.setEnabled(True)
        self.status_label.setText(f"Correction failed: {message}")
