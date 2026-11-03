"""
Processing page.

Replaces process_t.py's three ~700-1300 line `launch_process_{SHPB,QS,DI}`
Toplevel builders. The Quasi-Static engine (core/qs_processing.py) is fully
ported and wired below, run across the project's Monte-Carlo draws with
inline progress and an inline result plot. SHPB and Direct-Impact share the
same page shell and parameter intake, but their engines
(process_SHPB.py / process_DI.py, ~700-1,000 lines each of iterative
Bessel-based dispersion correction per Monte-Carlo draw) are staged as the
next porting step rather than guessed at.
"""
from __future__ import annotations

import math
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QPushButton, QLabel, QHBoxLayout,
    QComboBox, QProgressBar, QSlider, QApplication,
)
from PySide6.QtCore import Qt

from core.qs_processing import process_qs, _process_qs_task
from core.machine_compliance import apply_compliance_correction
from core.curve_cutting import cut_curve_start
from core.specimen_sensor_alignment import (
    specimen_sensor_delay_shpb, specimen_sensor_delay_direct_impact, sample_wave_speed,
    specimen_sensor_cut_points_shpb, specimen_sensor_cut_points_direct_impact,
)
from core.custom_sensor import cut_and_resample
from core.confinement import fit_nonlinear_calibration, per_draw_confinement_quantities
from core.wave_alignment import predicted_transmitted_wave, best_alignment_shift
from core.toe_correction import (
    detect_linear_region, fit_linear_region,
    infer_true_strain_relation, true_strain_from_nominal,
)
from core.dispersion import _cut_two_bar_windows
from core.process_shpb import process_shpb, SHPBOptions, _process_shpb_task, _index_of_nearest
from core.process_di import process_di, DIOptions, _process_di_task
from core.parallel import run_map
from core.models import TEST_MODES
from core.project import PLATEN_PARAM_ORDER
from core.param_io import load_param_file
from core.downsampling import downsample
from ui.widgets import Card, FileSelectRow, LabeledField, MCParamTable, make_scroll_page, TimeSourceRow
from ui.plot_canvas import PlotCanvas
from ui.span_picker import TimeWindowPicker
from ui import theme
from PySide6.QtWidgets import QLineEdit, QCheckBox

# Shared help text for the "Notched specimen k-coefficients" card, which
# appears identically (same shared project.shpb.k_value/sk_value storage)
# on all three tabs (QS, SHPB, Direct Impact).
_NOTCH_HELP_HTML = (
    "<p>Fitted coefficients for a notched specimen's kinematics -- only shown/used when the "
    "kinematics/test mode is set to \"Notched\".</p>"
    "<ul>"
    "<li><b>Load k/sk-coefficients from file</b>: loads a previously-fitted set instead of "
    "typing them by hand.</li>"
    "<li><b>k_value[i]</b>: the fit coefficients themselves.</li>"
    "<li><b>Split fit at a displacement threshold (sk)</b>: optionally uses a different set of "
    "coefficients (sk_value[i]) past a displacement threshold, for a fit that changes "
    "character partway through the test.</li>"
    "</ul>"
    "<p><b>Technical detail:</b> this storage is shared across the QS, SHPB and Direct Impact "
    "tabs (project-wide, not specific to one tab) -- a value typed on one tab shows up on the "
    "others too.</p>"
)


def _aggregate_quantities(per_draw: list[dict]) -> dict:
    """Average several named per-Monte-Carlo-draw quantities (each a list
    of numbers, possibly a different length per draw due to differencing
    like strain_rate) into one mean array per quantity, plus a `<key>_std`
    1-sigma standard-deviation array for the same quantity, truncated to
    the shortest draw for that quantity."""
    if not per_draw:
        return {}
    keys = per_draw[0].keys()
    out = {}
    for key in keys:
        arrays = [np.asarray(d[key]) for d in per_draw if key in d and len(d[key]) > 0]
        if not arrays:
            continue
        min_len = min(len(a) for a in arrays)
        stacked = np.array([a[:min_len] for a in arrays])
        out[key] = stacked.mean(axis=0)
        out[f"{key}_std"] = stacked.std(axis=0)
    return out


PRE_TOE_SUFFIX = "_pre_toe"


def _pre_toe_quantities(original_per_draw: list[dict]) -> dict:
    """The aggregated (mean + `_std`) curves of the ORIGINAL, pre-toe-
    correction per-draw ensemble, with every key renamed "<key>_pre_toe"
    (and "<key>_pre_toe_std"), so they can sit in the same result dict as
    the corrected curves and be exported next to them (Results & Export
    CSV/TXT). The raw per-draw sensor signals ("sensor_<i>_signal...") are
    left out: they are inputs, not results, and would bloat the export."""
    out = {}
    for key, values in _aggregate_quantities(original_per_draw).items():
        if key.startswith("sensor_"):
            continue
        if key.endswith("_std"):
            out[f"{key[:-4]}{PRE_TOE_SUFFIX}_std"] = values
        else:
            out[f"{key}{PRE_TOE_SUFFIX}"] = values
    return out


def _prepare_confinement(project):
    """Returns (cell_idx, cell_sensor, poly_ini, poly_esp) when the
    confinement-derived quantities (radial stress, hydrostatic pressure,
    deviator, radial/volumetric/equivalent strain -- see core.confinement.
    per_draw_confinement_quantities) can be computed for this run:
    confinement enabled (project.confinement.enabled) and an enabled
    Custom Sensor with the "Confinement cell strain gage" role present --
    same lookup Results & Export's own _refresh_confinement uses. Returns
    None otherwise (confinement not applicable to this test, or -- for the
    nonlinear cell model -- its calibration curves are missing/don't fit).

    `poly_ini`/`poly_esp` are the nonlinear model's two calibration
    polynomials, pre-fitted ONCE here (core.confinement.
    fit_nonlinear_calibration) so a Monte-Carlo loop that calls
    per_draw_confinement_quantities many times doesn't re-fit the same
    curves on every draw; both are None for the linear cell model
    (nothing to precompute)."""
    conf = project.confinement
    if not conf.enabled:
        return None
    cell_idx, cell_sensor = next(
        ((i, s) for i, s in enumerate(project.custom_sensors)
         if s.role == "confinement_cell" and s.enabled and len(s.data) > 0),
        (None, None),
    )
    if cell_sensor is None:
        return None
    poly_ini = poly_esp = None
    if conf.nonlinear_enabled:
        if not (conf.calib_ortho_disp_ini and conf.calib_radial_ini
                and conf.calib_ortho_disp_esp and conf.calib_radial_esp):
            return None
        try:
            poly_ini, poly_esp = fit_nonlinear_calibration(
                conf.cell_diameter, conf.calib_ortho_disp_ini, conf.calib_radial_ini,
                conf.calib_ortho_disp_esp, conf.calib_radial_esp,
            )
        except (ValueError, np.linalg.LinAlgError):
            return None
    return cell_idx, cell_sensor, poly_ini, poly_esp


def _store_raw_draws(project, result_key: str, per_draw: list[dict]) -> None:
    """Keeps every Monte-Carlo draw's own array (not just the mean/std),
    for the Covariance Analysis page -- converts the per-draw list of
    dicts into one list-of-arrays per quantity."""
    if not per_draw:
        return
    by_quantity: dict[str, list] = {}
    for draw in per_draw:
        for key, values in draw.items():
            by_quantity.setdefault(key, []).append(values)
    project.last_raw_draws[result_key] = by_quantity


def _apply_toe_correction_to_draws(per_draw: list[dict], strain_key: str, strain_offset: float, relation):
    """The toe (foot) correction's per-draw counterpart of the single-curve
    trim below (`_on_toe_apply`): re-zeroes/trims EVERY Monte-Carlo draw's
    OWN strain/stress/etc. curve by the same fitted `strain_offset`,
    instead of only the already-aggregated mean curve in `last_results`.

    Why per draw at all, rather than reusing one shared cut index for
    every draw: a toe correction shifts and TRIMS (removes leading, sub-
    offset samples), and every other quantity recorded for that draw --
    force, temperature, displacement, `sensor_<idx>_signal`, and (this is
    what the user specifically asked about) the confinement-cell/laser
    `confinement_*` quantities -- shares that SAME draw's own strain-length
    reference array, so it must be cut by that SAME draw's own crossing
    point to stay internally self-consistent, exactly like the aggregate
    case already does for the mean curve. This also regenerates
    `last_raw_draws` (the Covariance Analysis page's uncertainty-
    propagation input) so it, too, reflects the corrected cut, instead of
    remaining an uncorrected/inconsistent ensemble the way a purely
    aggregate-only correction would leave it.

    A draw whose own curve never reaches `strain_offset` at all (an
    outlier Monte-Carlo realization) is dropped from the ensemble rather
    than left untrimmed -- mixing one uncorrected draw into an otherwise
    corrected aggregate would silently corrupt the mean/std. The SAME
    reasoning applies, and is applied below, to a draw that DOES cross
    `strain_offset` but only barely, e.g. because that draw's own sampled
    bar/specimen/platen properties happened to produce an unusually short
    usable curve: `_aggregate_quantities` (the caller's next step)
    truncates EVERY draw down to the length of the single SHORTEST one,
    for each quantity independently, so even ONE such straggler would
    silently drag the entire published mean/std curve down to its own
    tiny remaining length -- a 0.035-strain result collapsing to 0.006,
    say, because one in thirty draws barely cleared the offset -- instead
    of that one draw simply being excluded like the "never crossed" case
    already is. A draw is therefore treated as an outlier and dropped if
    its remaining length is less than half the ensemble's own MEDIAN
    remaining length (a draw that never crosses has remaining length 0,
    which always qualifies, so this subsumes that original check).

    Returns (new_per_draw, n_removed_points) -- a NEW list of NEW per-draw
    dicts (the input `per_draw`/its dicts are never mutated, so the caller
    can keep the original around as an exact backup), and the MEDIAN
    number of leading points removed across every surviving draw (median
    rather than any single draw's own count, since individual draws are
    noisier than the aggregate mean curve and one draw's count alone could
    be a poor/misleading representative of the whole ensemble -- this
    number also seeds `toe_n_removed`, which Results & Export uses to keep
    a confinement-cell/laser sensor's raw, non-Monte-Carlo-cut fallback
    signal aligned with the now-shorter curve)."""
    candidates = []  # (new_draw, n_removed_for_this_draw, remaining_length)
    for draw in per_draw:
        if strain_key not in draw:
            continue
        strain_d = np.asarray(draw[strain_key])
        keep = (strain_d - strain_offset) >= 0
        if not keep.any():
            continue
        n_ref = len(strain_d)
        new_draw = dict(draw)
        for key, values in draw.items():
            # Per-draw fields come straight off the SHPBResult/DIResult
            # dataclass (see _SHPBTab/_DITab._on_run's own `draw = {...}`),
            # which are plain Python lists for most quantities and real
            # ndarrays only for the ones built in this UI layer (confinement
            # quantities, `sensor_<idx>_signal`) -- unlike the AGGREGATE
            # result dict this same trim was originally written for
            # (_aggregate_quantities always stacks into real ndarrays), so
            # both types must be caught here, or list-valued quantities
            # (like force/temperature/displacement) would silently be left
            # at their old, untrimmed length.
            if isinstance(values, (np.ndarray, list)) and len(values) == n_ref:
                new_draw[key] = np.asarray(values)[keep]
        new_draw[strain_key] = strain_d[keep] - strain_offset
        if relation and strain_key == "nominal_strain" and "strain" in new_draw:
            new_draw["strain"] = true_strain_from_nominal(new_draw[strain_key], relation)
        remaining = int(keep.sum())
        candidates.append((new_draw, n_ref - remaining, remaining))

    if not candidates:
        return [], 0

    median_remaining = float(np.median([remaining for _, _, remaining in candidates]))
    min_remaining = max(2, 0.5 * median_remaining)
    trimmed = [new_draw for new_draw, _, remaining in candidates if remaining >= min_remaining]
    n_removed_per_draw = [n for _, n, remaining in candidates if remaining >= min_remaining]

    n_removed = int(round(float(np.median(n_removed_per_draw)))) if n_removed_per_draw else 0
    return trimmed, n_removed



class ProcessPage(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 24)

        header = QLabel("Processing")
        header.setObjectName("pageTitle")
        outer.addWidget(header)

        tabs = QTabWidget()
        self.qs_tab = _QSTab(project)
        self.shpb_tab = _SHPBTab(project)
        self.di_tab = _DITab(project)
        tabs.addTab(self.qs_tab, "Load/Displacement Data")
        tabs.addTab(self.shpb_tab, "SHPB")
        tabs.addTab(self.di_tab, "Direct Impact")
        outer.addWidget(tabs)

    def sync_to_project(self):
        """Force-syncs the Processing sub-tab that actually matches the
        current Loading Regime (Test Setup page) into `project` -- called
        right before any export, as a safety net so an export always
        captures whatever is currently on screen, even if some individual
        widget's change signal didn't fire/propagate for any reason.

        Only the ONE relevant sub-tab is synced, not all three: QS, SHPB
        and Direct Impact share the same underlying `project.shpb`
        object, so syncing all of them in sequence would let whichever
        tab is synced LAST silently overwrite the actually-used tab's
        settings with its own (irrelevant, still-default) checkbox state.
        """
        regime = self.project.loading_regime
        tab = {
            "Load/Displacement Data": self.qs_tab,
            "Hopkinson Bars (SHPB)": self.shpb_tab,
            "Direct Impact": self.di_tab,
        }.get(regime)
        if tab is None:
            return
        sync = getattr(tab, "_on_settings_changed", None) or getattr(tab, "_on_qs_settings_changed", None)
        if sync is not None:
            sync()


class _QSTab(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._timet_raw = None
        self._load_raw = None
        self._disp_raw = None
        self._cut_applied = False

        scroll, layout = make_scroll_page(
            "",
            "Stress/strain/strain-rate/temperature from load-displacement data, run across "
            "every Monte-Carlo draw of the specimen.",
        )
        # Drop the (empty) page title/description row this tab doesn't need.
        layout.takeAt(0).widget().deleteLater()

        files = Card(
            "Signals",
            help_html=(
                "<p>Loads the raw load-displacement measurement for this test.</p>"
                "<ul>"
                "<li><b>Time file</b>: load a time column, or check \"Generate from a time step\" "
                "and type a fixed sample interval instead.</li>"
                "<li><b>Load file</b> / <b>Displacement file</b>: the measured force and "
                "displacement, one sample per line.</li>"
                "</ul>"
            ),
        )
        self.time_row = TimeSourceRow("Time file (.txt)")
        self.time_row.file_selected.connect(self._on_time_file)
        self.time_row.dtChanged.connect(self._on_time_step_changed)
        self.time_row.modeChanged.connect(self._on_time_mode_changed)
        self.load_row = FileSelectRow("Load file (.txt)")
        self.load_row.file_selected.connect(self._on_load_file)
        self.disp_row = FileSelectRow("Displacement file (.txt)")
        self.disp_row.file_selected.connect(self._on_disp_file)
        for row in (self.time_row, self.load_row, self.disp_row):
            files.add(row)
        layout.addWidget(files)

        cut_card = Card(
            "Cut curve beginning",
            "Load-displacement curves often start with a slack/toe region before the specimen "
            "makes proper contact. Drag out a window below (only its start matters) at the point "
            "the real curve begins, then cut -- everything before is discarded and displacement "
            "is re-zeroed.",
        )
        self.cut_picker = TimeWindowPicker(
            [("cut", "Cut point", theme.ACCENT)], figsize=(9, 2.8),
            help_title="Cut curve beginning",
            help_text=(
                "Shows the raw load-displacement curve, which often starts with a slack/toe region "
                "before the specimen makes proper contact with the platens. Click and drag on the "
                "plot to set the \"Cut point\" window -- only where it STARTS matters, marking where "
                "the real loading begins. Click \"Cut curve at selected start\" below to discard "
                "everything before that point and re-zero displacement from there; \"Reset cut\" "
                "restores the original, uncut curve."
            ),
        )
        cut_card.add(self.cut_picker)
        cut_row = QHBoxLayout()
        self.cut_button = QPushButton("Cut curve at selected start")
        self.cut_button.clicked.connect(self._on_cut_curve)
        cut_row.addWidget(self.cut_button)
        self.reset_cut_button = QPushButton("Reset cut")
        self.reset_cut_button.clicked.connect(self._on_reset_cut)
        cut_row.addWidget(self.reset_cut_button)
        cut_row.addStretch(1)
        cut_card.add_layout(cut_row)
        self.cut_status_label = QLabel("")
        self.cut_status_label.setObjectName("cardSubtitle")
        cut_card.add(self.cut_status_label)
        layout.addWidget(cut_card)

        settings = Card(
            "Kinematics",
            "Unlike the SHPB and Direct Impact tabs (which always follow the \"Test mode\" set on "
            "the Test Setup page), this selection is independent -- specific to this tab, and NOT "
            "saved with the project or a Setup Profile. It starts out matching the Test Setup page's "
            "current \"Test mode\" but doesn't track later changes to it, so if this test's kinematics "
            "differ from an SHPB/Direct Impact test processed earlier (or later) in the same session, "
            "double-check this dropdown before running -- a mismatch here (e.g. left on \"Notched\" "
            "for a plain compression specimen with no notch radius configured) is exactly what used "
            "to produce a cryptic \"divide by zero\"/crash instead of a clear message.",
            help_html=(
                "<p>Selects how strain/stress are computed from the load-displacement signal: "
                "Compression, Tension, Notched specimen (fitted k-coefficients), or Shear.</p>"
                "<p><b>Important:</b> unlike the SHPB and Direct Impact tabs, this dropdown is "
                "independent -- specific to this tab and NOT saved with the project or a Setup "
                "Profile. It starts out matching the Test Setup page's \"Test mode\" but does not "
                "keep tracking later changes to it. Always double-check this dropdown before "
                "running, especially if this test's kinematics differ from an SHPB/Direct Impact "
                "test processed earlier in the same session.</p>"
            ),
        )
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "1 — Compression (true strain)",
            "2 — Tension (true strain)",
            "3 — Notched specimen (fitted k-coefficients)",
            "4 — Shear",
        ])
        self.type_combo.setCurrentIndex(self._initial_type_check_index())
        self.type_combo.currentIndexChanged.connect(self._on_type_combo_changed)
        settings.add(LabeledField("Test kinematics (type_check)", self.type_combo))
        layout.addWidget(settings)

        notch = Card(
            "Notched specimen k-coefficients",
            "Only used when the Kinematics dropdown above is set to \"3 — Notched specimen "
            "(fitted k-coefficients)\" -- hidden otherwise, since no other specimen geometry needs "
            "them. Shares the same underlying k_value/sk_value fit coefficients as the SHPB and "
            "Direct Impact tabs (project-wide, not specific to this tab).",
            help_html=_NOTCH_HELP_HTML,
        )
        self.notch_file_row = FileSelectRow("Load k/sk-coefficients from file")
        self.notch_file_row.file_selected.connect(self._on_load_k_value_file)
        notch.add(self.notch_file_row)
        self.notch_file_status = QLabel("")
        self.notch_file_status.setObjectName("cardSubtitle")
        notch.add(self.notch_file_status)
        self.k_edits = [QLineEdit(str(v)) for v in project.shpb.k_value]
        for i, e in enumerate(self.k_edits):
            notch.add(LabeledField(f"k_value[{i}]", e))
        self.sk_check = QCheckBox("Split fit at a displacement threshold (sk)")
        notch.add(self.sk_check)
        self.sk_edits = [QLineEdit(str(v)) for v in project.shpb.sk_value]
        for i, e in enumerate(self.sk_edits):
            notch.add(LabeledField(f"sk_value[{i}]", e))
        self.notch_card = notch
        layout.addWidget(notch)
        self.notch_card.setVisible(self.type_combo.currentIndex() == 2)

        # Keep project.shpb's k_value/sk_check/sk_value live-synced with these
        # widgets -- shared storage with the SHPB/Direct Impact tabs (see the
        # notch card's own description), same live-sync rationale as those
        # tabs' _on_settings_changed.
        for e in self.k_edits:
            e.editingFinished.connect(self._on_notch_settings_changed)
        self.sk_check.toggled.connect(self._on_notch_settings_changed)
        for e in self.sk_edits:
            e.editingFinished.connect(self._on_notch_settings_changed)

        compliance = Card(
            "Machine compliance correction",
            "Removes the load-frame's own load-dependent displacement from the measured "
            "displacement, using a calibration curve (e.g. from a rigid/reference specimen).",
            help_html=(
                "<p>The load frame itself flexes a little under load, adding a small "
                "load-dependent displacement to what's actually measured at the specimen. This "
                "correction removes that, using a calibration curve recorded on a rigid/reference "
                "specimen (which should show only frame displacement, not specimen "
                "deformation).</p>"
                "<ul>"
                "<li>Check <b>Apply machine compliance correction</b>, then load the load and "
                "displacement calibration files.</li>"
                "</ul>"
            ),
        )
        self.compliance_check = QCheckBox("Apply machine compliance correction")
        self.compliance_check.setChecked(project.qs.machine_compliance_enabled)
        self.compliance_check.toggled.connect(self._on_compliance_toggled)
        compliance.add(self.compliance_check)
        self.calib_load_row = FileSelectRow("Load compliance file (.txt)")
        self.calib_load_row.file_selected.connect(self._on_calib_load_file)
        self.calib_load_row.setVisible(project.qs.machine_compliance_enabled)
        compliance.add(self.calib_load_row)
        self.calib_disp_row = FileSelectRow("Displacement compliance file (.txt)")
        self.calib_disp_row.file_selected.connect(self._on_calib_disp_file)
        self.calib_disp_row.setVisible(project.qs.machine_compliance_enabled)
        compliance.add(self.calib_disp_row)
        self.compliance_status_label = QLabel("")
        self.compliance_status_label.setObjectName("cardSubtitle")
        self.compliance_status_label.setVisible(False)
        compliance.add(self.compliance_status_label)
        layout.addWidget(compliance)

        uncertainty = Card(
            "Measurement uncertainties",
            "Applied as a per-Monte-Carlo-draw random multiplicative factor on the load and "
            "displacement signals (mean 1, this % as the standard deviation).",
            help_html=(
                "<p>Sets the instrument uncertainty on the load cell and displacement/extensometer "
                "measurement.</p>"
                "<ul>"
                "<li>Each is applied as a per-Monte-Carlo-draw random multiplicative factor on "
                "the corresponding signal (mean 1, this percentage as the standard deviation) -- "
                "so it contributes to the ±1 standard-deviation band shown on the Result plot "
                "below.</li>"
                "</ul>"
            ),
        )
        self.clu_edit = QLineEdit(str(project.qs.cell_load_uncertainty_pct))
        self.clu_edit.editingFinished.connect(self._on_qs_settings_changed)
        uncertainty.add(LabeledField("Cell Load Uncertainty (%)", self.clu_edit))
        self.meu_edit = QLineEdit(str(project.qs.measurement_uncertainty_pct))
        self.meu_edit.editingFinished.connect(self._on_qs_settings_changed)
        uncertainty.add(LabeledField("Measurement Uncertainty (%) (displacement/extensometer)", self.meu_edit))
        layout.addWidget(uncertainty)

        run_card = Card("Run")
        self.curve_state_label = QLabel("No signal loaded yet.")
        self.curve_state_label.setObjectName("cardSubtitle")
        run_card.add(self.curve_state_label)
        run_row = QHBoxLayout()
        self.run_button = QPushButton("Process all Monte-Carlo draws")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._on_run)
        run_row.addWidget(self.run_button)
        run_row.addStretch(1)
        run_card.add_layout(run_row)
        self.progress = QProgressBar()
        run_card.add(self.progress)
        self.status_label = QLabel("")
        self.status_label.setObjectName("cardSubtitle")
        run_card.add(self.status_label)
        layout.addWidget(run_card)

        result_card = Card("Result", "Mean stress-strain curve across all Monte-Carlo draws, with the "
                                      "±1 standard-deviation band shown as the uncertainty envelope.")
        self.canvas = PlotCanvas(
            figsize=(8, 5),
            help_title="Load/Displacement result",
            help_text=(
                "Shows the mean true stress-strain curve computed by \"Process all Monte-Carlo "
                "draws\", together with its ±1 standard-deviation uncertainty band (from the "
                "Monte-Carlo spread across draws, combining the Cell Load and Measurement "
                "uncertainties set above). Use the toolbar to zoom/pan and check the curve's shape "
                "-- a clean toe-free rise into a plateau/hardening region is the expected result "
                "once the curve has been cut properly above."
            ),
        )
        result_card.add(self.canvas)
        layout.addWidget(result_card)

        layout.addStretch(1)

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)

        self._restore_from_session()

    def showEvent(self, event):
        """Re-syncs the k/sk-coefficient fields from `project.shpb` every
        time this tab becomes visible, since that storage is shared with the
        SHPB and Direct Impact tabs -- e.g. values typed on the SHPB tab
        should show up here too, not just the other way around (handled by
        `_on_notch_settings_changed`)."""
        super().showEvent(event)
        s = self.project.shpb
        widgets = [*self.k_edits, self.sk_check, *self.sk_edits]
        for w in widgets:
            w.blockSignals(True)
        try:
            for edit, value in zip(self.k_edits, s.k_value):
                edit.setText(str(value))
            self.sk_check.setChecked(s.sk_check)
            for edit, value in zip(self.sk_edits, s.sk_value):
                edit.setText(str(value))
        finally:
            for w in widgets:
                w.blockSignals(False)

    def _restore_from_session(self):
        """Re-load any files remembered from a previous session (see
        core/session.py) -- these aren't restored automatically since the
        raw arrays live on this page, not on `Project`, so each file is
        just re-read through the same handler used when picking it live."""
        from pathlib import Path
        # Load/displacement before time: a generated (not file-based) time
        # vector needs one of them already loaded to know how many samples
        # to generate (see _on_time_step_changed/_maybe_regenerate_time).
        for key, handler in (
            ("qs_load", self._on_load_file), ("qs_displacement", self._on_disp_file),
            ("qs_time", self._on_time_file),
            ("qs_calib_load", self._on_calib_load_file), ("qs_calib_displacement", self._on_calib_disp_file),
        ):
            path = self.project.last_paths.get(key)
            if path and Path(path).exists():
                handler(path)

        dt = self.project.time_step_overrides.get("qs_time")
        if dt is not None:
            self.time_row.set_dt(dt)
            self.time_row.set_generate_mode(True)  # regenerates now, matching whichever of load/displacement was just restored above

        # Re-apply a "cut curve beginning" that was applied before the app
        # closed, now that the raw (uncut) files above are freshly reloaded.
        cut_displacement = self.project.qs.cut_displacement
        if cut_displacement is not None and self._timet_raw is not None:
            self._timet_raw, self._load_raw, self._disp_raw = cut_curve_start(
                self._timet_raw, self._load_raw, self._disp_raw, cut_displacement=cut_displacement
            )
            self._cumulative_cut = cut_displacement
            self._cut_applied = True
            self._refresh_cut_picker()
            self._refresh_curve_state()
            self.cut_status_label.setText(f"Restored cut from previous session (displacement = {cut_displacement:.6g}).")

    def _initial_type_check_index(self) -> int:
        """Maps the Test Setup page's current "Test mode" onto this tab's own
        (otherwise independent, unsaved) Kinematics dropdown, purely as a
        one-time sensible starting point when the tab is first built -- see
        the Kinematics card's own description for why this doesn't keep
        tracking Test Setup afterward, and why that's worth double-checking
        before each run."""
        mode = self.project.test_mode
        if mode == "Tension":
            return 1
        if mode == "Shear-Compression/Tension (Notched)":
            return 2
        if mode == "Shear":
            return 3
        return 0  # "Compression", and any unrecognized/future mode

    def _on_type_combo_changed(self, index):
        """Shows the k-coefficients card only for "3 -- Notched specimen" --
        every other geometry on this tab computes strain directly from the
        displacement signal and doesn't need a k-coefficient fit at all."""
        self.notch_card.setVisible(index == 2)

    def _on_notch_settings_changed(self, *_args):
        """Keeps `project.shpb`'s k_value/sk_check/sk_value live-synced with
        this tab's widgets -- this storage is shared with the SHPB and Direct
        Impact tabs, so without this, navigating away and back (which doesn't
        re-read these fields, unlike SHPB/DI's own settings sync) would leave
        stale values in the fields even though `project.shpb` already changed
        (e.g. from the SHPB tab), and typing here wouldn't reach `process_qs`
        via any other path since these widgets aren't read until "Process"."""
        s = self.project.shpb
        for i, e in enumerate(self.k_edits):
            try:
                s.k_value[i] = float(e.text())
            except ValueError:
                pass
        s.sk_check = self.sk_check.isChecked()
        for i, e in enumerate(self.sk_edits):
            try:
                s.sk_value[i] = float(e.text())
            except ValueError:
                pass

    def _on_load_k_value_file(self, path):
        """Loads the notched-specimen k/sk fit coefficients from a plain
        text file (see core/k_value_io.py), same as the equivalent button on
        the SHPB/Direct Impact tabs -- writes into the k_value[]/sk_check/
        sk_value[] fields (shared storage with those tabs) and persists them
        via `_on_notch_settings_changed()`, as if typed in by hand."""
        from core.k_value_io import load_k_value_file
        try:
            k_value, sk_check, sk_value = load_k_value_file(path)
        except (OSError, ValueError) as exc:
            self.notch_file_status.setText(f"Could not load k/sk-coefficients: {exc}")
            return
        for edit, value in zip(self.k_edits, k_value):
            edit.setText(str(value))
        if sk_check is not None:
            self.sk_check.setChecked(sk_check)
        if sk_value is not None:
            for edit, value in zip(self.sk_edits, sk_value):
                edit.setText(str(value))
        self._on_notch_settings_changed()
        detail = "k-values only (no split/sk section in this file)" if sk_value is None else "k- and sk-values"
        self.notch_file_status.setText(f"Loaded {detail} from {path.split('/')[-1]}.")

    def _on_time_file(self, path):
        try:
            values = np.asarray(_read_column(path))
            if self.project.downsample_enabled:
                values = downsample(values, self.project.downsample_n)
            self._timet_raw = values
        except (OSError, ValueError) as exc:
            self.cut_status_label.setText(f"Could not read time file: {exc}")
            return
        self.project.last_paths["qs_time"] = path
        self._cut_applied = False
        self._refresh_cut_picker()
        self._refresh_curve_state()

    def _on_time_step_changed(self, dt):
        """Alternative to `_on_time_file` above: build the time vector from
        a fixed time step instead of loading it from a file. Needs to know
        how many samples it should have, which (unlike a file's own row
        count) this page can only get from whatever load/displacement data
        is already loaded -- if neither is loaded yet, there's nothing to
        match the length of, so this just waits for `_maybe_regenerate_time`
        to be called again once one of them is."""
        self._maybe_regenerate_time()
        if self._timet_raw is None:
            self.cut_status_label.setText(
                "Load the load and/or displacement file first, so the generated time vector can "
                "match its length."
            )
            return
        self.project.last_paths.pop("qs_time", None)
        self.project.time_step_overrides["qs_time"] = dt
        self._cut_applied = False
        self._refresh_cut_picker()
        self._refresh_curve_state()

    def _on_time_mode_changed(self, generate):
        if not generate:
            self.project.time_step_overrides.pop("qs_time", None)

    def _maybe_regenerate_time(self):
        """If the time vector is currently being generated from a time step
        (rather than loaded from a file), rebuilds it to match the length of
        whatever load/displacement data is currently loaded -- called
        whenever either of those changes, so a generated time vector never
        silently goes stale (wrong length) relative to them."""
        if not self.time_row.is_generate_mode():
            return
        n = len(self._load_raw) if self._load_raw is not None else (len(self._disp_raw) if self._disp_raw is not None else 0)
        if n == 0:
            return
        generated = self.time_row.build(n)
        if generated is not None:
            self._timet_raw = generated

    def _on_load_file(self, path):
        try:
            values = np.asarray(_read_column(path))
            if self.project.downsample_enabled:
                values = downsample(values, self.project.downsample_n)
            self._load_raw = values
        except (OSError, ValueError) as exc:
            self.cut_status_label.setText(f"Could not read load file: {exc}")
            return
        self.project.last_paths["qs_load"] = path
        self._cut_applied = False
        self._maybe_regenerate_time()
        self._refresh_cut_picker()
        self._refresh_curve_state()

    def _on_disp_file(self, path):
        try:
            values = np.asarray(_read_column(path))
            if self.project.downsample_enabled:
                values = downsample(values, self.project.downsample_n)
            self._disp_raw = values
        except (OSError, ValueError) as exc:
            self.cut_status_label.setText(f"Could not read displacement file: {exc}")
            return
        self.project.last_paths["qs_displacement"] = path
        self._cut_applied = False
        self._maybe_regenerate_time()
        self._refresh_cut_picker()
        self._refresh_curve_state()

    def _refresh_cut_picker(self):
        if self._disp_raw is None or self._load_raw is None:
            self.cut_picker.show_placeholder("Load the load and displacement files above to see the curve here.")
            return
        # xlabel/primary_ylabel overridden here -- this plots Load vs
        # DISPLACEMENT (the x-array below is self._disp_raw, not a time
        # vector), so TimeWindowPicker's own "time (s)"/"amplitude"
        # defaults (meant for its usual pulse-window-on-a-raw-signal use)
        # would otherwise mislabel both axes. The window-bounds status text
        # above the plot still reports the selected cut point suffixed
        # " s" regardless (a generic label baked into the shared picker
        # widget) -- read it as a displacement value, not seconds.
        self.cut_picker.set_signal(
            self._disp_raw, [("Load vs Displacement", self._load_raw, theme.ACCENT)],
            xlabel="displacement", primary_ylabel="load",
        )

    def _on_cut_curve(self):
        if self._timet_raw is None or self._load_raw is None or self._disp_raw is None:
            self.cut_status_label.setText("Load the time, load and displacement files first.")
            return
        window = self.cut_picker.get_window("cut")
        if window is None:
            self.cut_status_label.setText("Drag on the plot to pick a cut point first.")
            return
        # Displacement gets re-zeroed after every cut, so if this isn't the
        # first cut, `window[0]` is relative to the *already-shifted* axis
        # -- track the cumulative offset against the ORIGINAL raw axis so
        # the exact same absolute cut point can be reapplied after a
        # session restore (which always starts from the raw file again).
        self._cumulative_cut = getattr(self, "_cumulative_cut", 0.0) + window[0]
        self._timet_raw, self._load_raw, self._disp_raw = cut_curve_start(
            self._timet_raw, self._load_raw, self._disp_raw, cut_displacement=window[0]
        )
        self._cut_applied = True
        self.project.qs.cut_displacement = self._cumulative_cut
        self.cut_status_label.setText(f"Cut applied: curve now starts at displacement = {window[0]:.6g} (re-zeroed), {len(self._disp_raw)} points remain.")
        self._refresh_cut_picker()
        self._refresh_curve_state()

    def _on_qs_settings_changed(self):
        try:
            self.project.qs.cell_load_uncertainty_pct = float(self.clu_edit.text())
        except ValueError:
            pass
        try:
            self.project.qs.measurement_uncertainty_pct = float(self.meu_edit.text())
        except ValueError:
            pass

    def _on_reset_cut(self):
        """Re-loads the time/load/displacement files fresh from disk,
        discarding any cut(s) applied."""
        from pathlib import Path
        self._cumulative_cut = 0.0
        self.project.qs.cut_displacement = None
        self._cut_applied = False
        found_any = False
        for key, handler in (
            ("qs_time", self._on_time_file), ("qs_load", self._on_load_file),
            ("qs_displacement", self._on_disp_file),
        ):
            path = self.project.last_paths.get(key)
            if path and Path(path).exists():
                handler(path)
                found_any = True
        # _on_*_file handlers above reset _cut_applied to False and
        # project.qs.cut_displacement stays cleared, since they don't touch
        # it -- but re-confirm cut_displacement is still None in case a
        # handler failed partway and left stale state.
        self.project.qs.cut_displacement = None
        if found_any:
            self.cut_status_label.setText("Cut reset: reloaded the original files from disk.")
        else:
            self.cut_status_label.setText("Nothing to reset (no remembered file to reload from).")

    def _refresh_curve_state(self):
        if self._timet_raw is None or self._load_raw is None or self._disp_raw is None:
            self.curve_state_label.setText("No signal loaded yet.")
            return
        n = len(self._disp_raw)
        if self._cut_applied:
            self.curve_state_label.setText(
                f"Using the CUT curve: {n} points, displacement starts at 0 (re-zeroed). "
                "This is what \"Process\" below will use."
            )
        else:
            self.curve_state_label.setText(
                f"Using the FULL, UNCUT curve: {n} points (displacement range "
                f"{self._disp_raw.min():.4g} to {self._disp_raw.max():.4g}). "
                "Use \"Cut curve beginning\" above first if you want to remove a toe/slack region."
            )

    def _on_compliance_toggled(self, checked):
        self.calib_load_row.setVisible(checked)
        self.calib_disp_row.setVisible(checked)
        self.compliance_status_label.setVisible(checked)
        self.project.qs.machine_compliance_enabled = checked

    def _on_calib_load_file(self, path):
        try:
            values = _read_column(path)
        except (OSError, ValueError) as exc:
            self.compliance_status_label.setText(f"Could not read load compliance file: {exc}")
            return
        self.project.qs.calib_load = values
        self.project.last_paths["qs_calib_load"] = path
        self._refresh_compliance_status()

    def _on_calib_disp_file(self, path):
        try:
            values = _read_column(path)
        except (OSError, ValueError) as exc:
            self.compliance_status_label.setText(f"Could not read displacement compliance file: {exc}")
            return
        self.project.qs.calib_displacement = values
        self.project.last_paths["qs_calib_displacement"] = path
        self._refresh_compliance_status()

    def _refresh_compliance_status(self):
        n_load = len(self.project.qs.calib_load)
        n_disp = len(self.project.qs.calib_displacement)
        if n_load and n_disp:
            mismatch = " (warning: different lengths, expected one displacement value per load value)" if n_load != n_disp else ""
            self.compliance_status_label.setText(f"Loaded {n_load} load points and {n_disp} displacement points.{mismatch}")
        elif n_load:
            self.compliance_status_label.setText(f"Loaded {n_load} load points -- still need the displacement compliance file.")
        elif n_disp:
            self.compliance_status_label.setText(f"Loaded {n_disp} displacement points -- still need the load compliance file.")
        else:
            self.compliance_status_label.setText("No compliance files loaded yet.")

    def _on_run(self):
        project = self.project
        if self._timet_raw is None or self._load_raw is None or self._disp_raw is None:
            self.status_label.setText("Load the time, load and displacement files first.")
            return

        timet = self._timet_raw
        load = self._load_raw
        disp = self._disp_raw
        type_check = self.type_combo.currentIndex() + 1

        k_value = sk_value = None
        sk_check = False
        if type_check == 3:
            try:
                k_value = [float(e.text()) for e in self.k_edits]
                sk_check = self.sk_check.isChecked()
                sk_value = [float(e.text()) for e in self.sk_edits]
            except ValueError:
                self.status_label.setText("k_value/sk_value fields must be numeric.")
                return

        if self.compliance_check.isChecked():
            if not project.qs.calib_load or not project.qs.calib_displacement:
                self.status_label.setText("Load both compliance files first, or uncheck the compliance correction.")
                return
            try:
                disp = apply_compliance_correction(load, disp, project.qs.calib_load, project.qs.calib_displacement)
            except Exception as exc:  # noqa: BLE001
                self.status_label.setText(f"Compliance correction failed: {exc}")
                return

        try:
            clu_pct = float(self.clu_edit.text())
            meu_pct = float(self.meu_edit.text())
        except ValueError:
            self.status_label.setText("Cell Load Uncertainty and Measurement Uncertainty must be numeric.")
            return
        project.qs.cell_load_uncertainty_pct = clu_pct
        project.qs.measurement_uncertainty_pct = meu_pct

        project.expand_montecarlo()
        specimen = project.specimen
        n = project.n_montecarlo
        self.progress.setMaximum(n)

        rng = np.random.default_rng()
        coeff_load_arr = rng.normal(1.0, clu_pct / 100.0, n)
        coeff_dd_arr = rng.normal(1.0, meu_pct / 100.0, n)

        # Confinement (quasi-oedometric cell) quantities, computed per
        # Monte-Carlo draw below instead of post-hoc in Results & Export --
        # see core.confinement.per_draw_confinement_quantities. Unlike
        # SHPB/Direct Impact, this tab has no wave-travel-time delay to
        # correct (the loading is quasi-static, so the confinement-cell
        # gage is already on the same time base as the load/displacement
        # signal) and so no per-draw "cut" of its own -- the cell signal
        # itself is therefore the SAME raw curve for every draw here; only
        # this draw's own stress/strain (already Monte-Carlo-varied above
        # via coeff_load_arr/coeff_dd_arr) makes the resulting confinement
        # quantities vary draw-to-draw.
        confinement_prep = _prepare_confinement(project)
        confinement_active = confinement_prep is not None
        cell_signal = None
        poly_ini = poly_esp = None
        if confinement_prep is not None:
            _cell_idx, _cell_sensor, poly_ini, poly_esp = confinement_prep
            cell_signal = np.asarray(_cell_sensor.calibrated_data())
        specimen_initial_length = project.specimen.params["L"].mean

        all_stress, all_strain = [], []
        per_draw = []

        qs_context = dict(
            type_check=type_check,
            comp_check=1 if type_check == 1 else 0,
            tens_check=1 if type_check == 2 else 0,
            displacement=disp, load=load, timet=timet,
            specimen=specimen,
            t0=project.thermal.initial_temperature_k,
            k_value=k_value, sk_value=sk_value, sk_check=sk_check,
        )
        qs_items = [(d, float(coeff_load_arr[d]), float(coeff_dd_arr[d])) for d in range(n)]

        def _report_progress(done, total):
            if done % max(1, total // 100) == 0 or done == total:
                self.progress.setValue(done)
                QApplication.processEvents()

        results = run_map(
            _process_qs_task, qs_items, n_workers=project.n_workers,
            context=qs_context, progress_callback=_report_progress,
        )

        for d, result, exc in results:
            if exc is not None:  # noqa: BLE001 - surface inline, keep going
                self.status_label.setText(f"Draw {d} failed: {exc}")
                continue
            all_stress.append(result.stress)
            all_strain.append(result.strain)
            draw = {
                "strain": result.strain, "stress": result.stress, "strain_rate": result.strain_rate,
                "temperature": result.temperature, "load": result.load, "displacement": result.displacement,
                # Time base of this draw's strain/stress samples -- needed for the
                # confinement strain RATES below, and for the same post-hoc rates in
                # Results & Export.
                "time": np.asarray(timet, dtype=float)[:len(result.strain)],
                **({"nominal_strain": result.nominal_strain} if result.nominal_strain is not None else {}),
                **({"nominal_stress": result.nominal_stress} if result.nominal_stress is not None else {}),
                **({"nominal_strain_rate": result.nominal_strain_rate} if result.nominal_strain_rate is not None else {}),
            }
            if confinement_active:
                try:
                    strain_for_conf = draw.get("nominal_strain", draw["strain"])
                    stress_for_conf = draw.get("nominal_stress", draw["stress"])
                    draw.update(per_draw_confinement_quantities(
                        strain_for_conf, stress_for_conf, cell_signal,
                        project.confinement, specimen_initial_length,
                        poly_ini=poly_ini, poly_esp=poly_esp, time=draw.get("time"),
                    ))
                except (ZeroDivisionError, ValueError, np.linalg.LinAlgError):
                    pass  # this one draw's confinement calc failed -- a rare numeric edge case
            per_draw.append(draw)

        self.progress.setValue(n)
        if not all_stress:
            self.status_label.setText("No draws completed successfully.")
            return

        min_len = min(len(s) for s in all_stress)
        stress_arr = np.array([s[:min_len] for s in all_stress])
        strain_arr = np.array([s[:min_len] for s in all_strain])
        mean_strain = strain_arr.mean(axis=0)
        mean_stress = stress_arr.mean(axis=0)
        std_stress = stress_arr.std(axis=0)

        confinement_note = ""
        if project.confinement.enabled:
            if confinement_active:
                confinement_note = (
                    " Confinement quantities (radial stress, pressure, deviator, radial/volumetric/"
                    "equivalent strain, and their strain rates) computed per Monte-Carlo draw from this draw's own stress/strain "
                    "(the confinement-cell signal itself is not Monte-Carlo-varied on this tab, unlike "
                    "SHPB/Direct Impact) -- Results & Export will use this real uncertainty."
                )
            else:
                confinement_note = (
                    " Confinement is enabled but its quantities could not be computed here -- check that an "
                    "enabled Confinement-cell Custom Sensor is set and, for the nonlinear cell model, that "
                    "all 4 calibration curves are loaded (Special Tests / Custom Sensors pages)."
                )
        self.status_label.setText(f"Processed {len(all_stress)}/{n} Monte-Carlo draws.{confinement_note}")
        quantities = _aggregate_quantities(per_draw)
        self.project.last_results["QS"] = {"label": "Load/Displacement Data", **quantities}
        _store_raw_draws(self.project, "QS", per_draw)
        self.canvas.clear()
        ax = self.canvas.figure.add_subplot(111)
        sigma = self.project.uncertainty_sigma
        ax.plot(mean_strain, mean_stress, "b-", label="Mean")
        ax.fill_between(mean_strain, mean_stress - sigma * std_stress, mean_stress + sigma * std_stress,
                         color="b", alpha=0.2, label=f"±{sigma}σ")
        ax.set_xlabel("Strain (-)")
        ax.set_ylabel("Stress (MPa)")
        ax.legend()
        self.canvas.refresh()


class _SHPBTab(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._timet = None

        scroll, layout = make_scroll_page(
            "",
            "Split Hopkinson Pressure Bar processing: wave-to-stress-strain conversion with "
            "optional bar-temperature-gradient, platen, punch, friction and inertia corrections. "
            "Uses the Input Bar / Output Bar / Specimen pages already filled in.",
        )
        layout.takeAt(0).widget().deleteLater()

        self.test_mode_label = QLabel("")
        self.test_mode_label.setObjectName("cardSubtitle")
        self.test_mode_label.setWordWrap(True)
        layout.addWidget(self.test_mode_label)

        pulse = Card(
            "Pulse window",
            "Click and drag on the Input Bar signal below to select the incident pulse window.",
            help_html=(
                "<p>Sets the timing/source used for the SHPB wave-to-stress-strain conversion "
                "(see the plot's own help below for how to drag the window).</p>"
                "<ul>"
                "<li><b>Use dispersion-corrected signal</b>: recommended -- otherwise the raw, "
                "uncorrected wave amplitudes are used directly.</li>"
                "<li><b>Time file</b>: load a time column, or generate one from a fixed time "
                "step.</li>"
                "</ul>"
            ),
        )
        self.use_corrected_check = QCheckBox(
            "Use dispersion-corrected signal from the Corrections page (recommended -- otherwise the "
            "raw, uncorrected wave amplitudes are used)"
        )
        self.use_corrected_check.setChecked(project.shpb.use_corrected_signal)
        self.use_corrected_check.toggled.connect(self._on_use_corrected_toggled)
        pulse.add(self.use_corrected_check)
        self.corrected_status_label = QLabel("")
        self.corrected_status_label.setObjectName("cardSubtitle")
        pulse.add(self.corrected_status_label)
        self.time_file_row = TimeSourceRow("Time file (.txt)")
        self.time_file_row.file_selected.connect(self._on_time_file)
        self.time_file_row.dtChanged.connect(self._on_time_step_changed)
        self.time_file_row.modeChanged.connect(self._on_time_mode_changed)
        pulse.add(self.time_file_row)
        self.picker = TimeWindowPicker(
            [("xd", "Pulse window", theme.ACCENT)],
            help_title="Pulse window",
            help_text=(
                "Shows the Input Bar signal (dispersion-corrected if \"Use dispersion-corrected "
                "signal\" above is checked, otherwise the raw recording). Click and drag on the "
                "plot to select the incident pulse window -- the portion of the signal used for "
                "the SHPB wave-to-stress-strain conversion. This is the same signal/timing already "
                "set on the Input Bar and Dispersion Corrections pages; adjust it here if this "
                "test's window needs a different choice."
            ),
        )
        self.picker.windowChanged.connect(self._on_window_changed)
        pulse.add(self.picker)
        layout.addWidget(pulse)
        # Qt's toggled signal doesn't fire from setChecked() if the value
        # doesn't change from the widget's own default -- apply the
        # visibility effect explicitly here too, so a restored "checked"
        # state (e.g. from a loaded test/profile) actually hides the raw-
        # signal controls, not just show the checkbox as ticked.
        self._on_use_corrected_toggled(self.use_corrected_check.isChecked())

        corrections = Card(
            "Corrections",
            "Toggle the physical corrections to apply.",
            help_html=(
                "<p>Each checkbox enables an additional physical correction on top of the base "
                "SHPB wave-to-stress-strain conversion.</p>"
                "<ul>"
                "<li><b>Bar-temperature-gradient correction</b>: compensates for a heated "
                "specimen conducting heat into the bar ends -- needs the parameters in the "
                "\"Bar-temperature-gradient parameters\" card below.</li>"
                "<li><b>Platens</b>: accounts for platens inserted between the bars and the "
                "specimen -- needs the \"Platens\" card below.</li>"
                "<li><b>Punch (indentation) correction</b>: compensates for local indentation "
                "of the specimen faces by the bar ends.</li>"
                "<li><b>Inertia correction</b>: compensates for the specimen's own radial/axial "
                "inertia at high strain rates.</li>"
                "<li><b>Friction coefficient</b>: used by the punch/inertia corrections' "
                "friction terms.</li>"
                "<li><b>Force used for stress</b>: which force signal is used to compute the "
                "engineering/true stress -- the output-bar force is the usual choice once "
                "dynamic equilibrium is reached; the average of input and output "
                "(F_input+F_output)/2 is a common alternative when equilibrium is imperfect or "
                "only briefly reached, since it weights both ends of the specimen equally "
                "instead of trusting either one alone.</li>"
                "</ul>"
            ),
        )
        self.heat_check = QCheckBox("Bar-temperature-gradient correction (heated specimen)")
        self.plat_check = QCheckBox("Platens (between bars and specimen)")
        self.punch_check = QCheckBox("Punch (indentation) correction")
        self.inertia_check = QCheckBox("Inertia correction")
        for cb in (self.heat_check, self.plat_check, self.punch_check, self.inertia_check):
            corrections.add(cb)
        self.friction_edit = QLineEdit(str(project.shpb.friction_coefficient))
        corrections.add(LabeledField("Friction coefficient", self.friction_edit))
        self.force_combo = QComboBox()
        self.force_combo.addItems(["Input-bar force", "Output-bar force", "Average force (input & output)"])
        self.force_combo.setCurrentIndex(1)  # Output-bar force by default
        corrections.add(LabeledField("Force used for stress", self.force_combo))
        layout.addWidget(corrections)

        sensor_align = Card(
            "Specimen-Mounted Sensor Alignment",
            "A sensor glued directly on the specimen (Custom Sensors page, \"mounted directly on the "
            "specimen\") doesn't start recording the test at the same instant as the bar strain gage "
            "-- the wave still has to travel from the gage to the specimen first. This cuts/re-zeroes "
            "such a sensor's signal by that travel time -- by default computed once from the mean "
            "bar/platen/specimen properties, or, if \"Include this cut in the Monte-Carlo loop\" is "
            "ticked below, recomputed for every Monte-Carlo draw from that draw's own sampled "
            "properties, so the cut's own uncertainty is propagated into the result.",
            help_html=(
                "<p>Applies only to a sensor configured as \"mounted directly on the specimen\" "
                "on the Custom Sensors page.</p>"
                "<ul>"
                "<li>The stress wave takes a finite time to travel from the strain gage on the "
                "bar to the specimen itself, so a specimen-mounted sensor's own clock starts "
                "\"late\" relative to the bar gages.</li>"
                "<li>This card cuts/re-zeroes that sensor's signal by the computed travel time, "
                "so it lines up with the rest of the test data.</li>"
                "<li><b>Auto-cut using wave travel time</b>: cuts once, immediately, from the mean "
                "bar/platen/specimen properties -- quick, but ignores their own Monte-Carlo "
                "uncertainty.</li>"
                "<li><b>Include this cut in the Monte-Carlo loop</b>: instead of a single cut "
                "applied to every draw, recomputes the cut point for EACH draw from that draw's "
                "own sampled bar/platen/specimen properties, and resamples the sensor's (cut, "
                "re-zeroed) signal onto the run's own time base -- so any test with a different "
                "sample rate/DAQ than the bars lines up point-for-point with the rest of the "
                "results. The resulting curve appears as its own quantity (\"<i>sensor label</i> "
                "(Monte-Carlo cut)\") in Results &amp; Export, with a proper Monte-Carlo mean and "
                "std-dev band -- not just this card's single deterministic cut. Only the cutting "
                "itself is redone per draw; nothing else about the sensor (calibration factor, "
                "declared measurement uncertainty) is reprocessed. Always searches the sensor's "
                "original, uncut signal for each draw's cut point, so it's safe to tick this even "
                "after already clicking \"Auto-cut using wave travel time\" above or the Custom "
                "Sensors page's own \"Cut curve\" -- no need to \"Reset cut\" first.</li>"
                "</ul>"
            ),
        )
        self.sensor_align_rows = []
        layout.addWidget(sensor_align)
        self._sensor_align_card = sensor_align

        heat = Card(
            "Bar-temperature-gradient parameters",
            "Only used if the correction above is enabled.",
            help_html=(
                "<p>Reference temperatures for the bar-temperature-gradient correction, used "
                "only when \"Bar-temperature-gradient correction\" is checked above.</p>"
                "<ul>"
                "<li><b>Input/Output bar gage temperature</b>: the temperature at each bar's "
                "strain-gage location, used together with the specimen temperature (Test Setup "
                "page) to model the heat conducted from the heated specimen into the bar ends.</li>"
                "</ul>"
            ),
        )
        self.gage_ib_edit = QLineEdit(str(project.shpb.gage_temp_ib))
        self.gage_ob_edit = QLineEdit(str(project.shpb.gage_temp_ob))
        heat.add(LabeledField("Input bar gage temperature (K)", self.gage_ib_edit))
        heat.add(LabeledField("Output bar gage temperature (K)", self.gage_ob_edit))
        layout.addWidget(heat)

        platen = Card(
            "Platens",
            "Only used if the platen correction above is enabled.",
            help_html=(
                "<p>Platen properties, used only when \"Platens\" is checked above.</p>"
                "<ul>"
                "<li><b>Platen config file</b>: loads a previously-saved set of platen "
                "parameters instead of typing them by hand.</li>"
                "<li>The table below holds the platen's mechanical properties (as Monte-Carlo "
                "parameters, with a mean and standard deviation) used in the wave transmission "
                "through the bar-platen-specimen-platen-bar stack.</li>"
                "</ul>"
            ),
        )
        self.platen_file_row = FileSelectRow("Platen config file")
        self.platen_file_row.file_selected.connect(self._on_platen_file)
        platen.add(self.platen_file_row)
        self.platen_status = QLabel("")
        self.platen_status.setObjectName("cardSubtitle")
        platen.add(self.platen_status)
        self.platen_table = MCParamTable([self.project.shpb.platen.params[k] for k in PLATEN_PARAM_ORDER])
        platen.add(self.platen_table)
        layout.addWidget(platen)

        notch = Card(
            "Notched specimen k-coefficients",
            "Only used when the test mode (Test Setup page) is Shear-Compression/Tension (Notched).",
            help_html=_NOTCH_HELP_HTML,
        )
        self.notch_file_row = FileSelectRow("Load k/sk-coefficients from file")
        self.notch_file_row.file_selected.connect(self._on_load_k_value_file)
        notch.add(self.notch_file_row)
        self.notch_file_status = QLabel("")
        self.notch_file_status.setObjectName("cardSubtitle")
        notch.add(self.notch_file_status)
        self.k_edits = [QLineEdit(str(v)) for v in project.shpb.k_value]
        for i, e in enumerate(self.k_edits):
            notch.add(LabeledField(f"k_value[{i}]", e))
        self.sk_check = QCheckBox("Split fit at a displacement threshold (sk)")
        notch.add(self.sk_check)
        self.sk_edits = [QLineEdit(str(v)) for v in project.shpb.sk_value]
        for i, e in enumerate(self.sk_edits):
            notch.add(LabeledField(f"sk_value[{i}]", e))
        self.notch_card = notch
        layout.addWidget(notch)

        # Keep project.shpb live-synced with every settings widget above --
        # otherwise nothing writes these back until "Process" is clicked,
        # so simply navigating away and back (which re-syncs the widgets
        # FROM project.shpb) would silently reset anything typed/toggled.
        self.heat_check.toggled.connect(self._on_settings_changed)
        self.plat_check.toggled.connect(self._on_settings_changed)
        self.punch_check.toggled.connect(self._on_settings_changed)
        self.inertia_check.toggled.connect(self._on_settings_changed)
        self.friction_edit.editingFinished.connect(self._on_settings_changed)
        self.force_combo.currentIndexChanged.connect(self._on_settings_changed)
        self.gage_ib_edit.editingFinished.connect(self._on_settings_changed)
        self.gage_ob_edit.editingFinished.connect(self._on_settings_changed)
        for e in self.k_edits:
            e.editingFinished.connect(self._on_settings_changed)
        self.sk_check.toggled.connect(self._on_settings_changed)
        for e in self.sk_edits:
            e.editingFinished.connect(self._on_settings_changed)

        wave_align_card = Card(
            "Transmitted Wave Alignment (theoretical prediction)",
            "Overlays the cut incident, reflected and measured transmitted waves (converted to force) "
            "with the THEORETICAL elastic transmitted wave, predicted from the incident+reflected wave "
            "using only mechanical impedance (Z = rho*C0*A) and the transmission coefficients at each "
            "interface (bar -> [platen] -> specimen -> [platen] -> bar) -- no assumption about the "
            "specimen real (inelastic) behaviour. Drag the slider to shift the measured transmitted "
            "wave until it lines up with the theoretical one, without needing to look at the final "
            "stress-strain result at all.",
            help_html=(
                "<p>An independent timing check: it predicts what the transmitted wave "
                "<i>should</i> look like from pure mechanical-impedance transmission, then lets "
                "you align the measured transmitted wave to it directly (see the plot's own help "
                "below for what's shown).</p>"
                "<ul>"
                "<li><b>Refresh preview</b>: recomputes the overlay after changing the pulse "
                "window above.</li>"
                "<li><b>Shift measured transmitted wave</b> (slider): manually shifts the "
                "measured curve in time.</li>"
                "<li><b>Auto-align (cross-correlation)</b>: finds the shift automatically.</li>"
                "<li><b>Apply shift to output-window offset</b>: commits the chosen shift into "
                "the actual processing.</li>"
                "</ul>"
                "<p><b>Technical detail:</b> the theoretical prediction uses only Z = rho*C0*A "
                "and the interface transmission coefficients -- it makes no assumption about the "
                "specimen's real (inelastic) behaviour, so a mismatch here points specifically "
                "to a timing/alignment issue rather than to the specimen's material response.</p>"
            ),
        )
        self.wave_align_canvas = PlotCanvas(
            figsize=(8, 4),
            help_title="Transmitted Wave Alignment",
            help_text=(
                "<p>Overlays the cut incident, reflected and measured transmitted waves "
                "(converted to force) with the THEORETICAL elastic transmitted wave predicted "
                "purely from mechanical impedance and interface transmission coefficients -- no "
                "assumption about the specimen's real (inelastic) behaviour.</p>"
                "<ul>"
                "<li>Click <b>Refresh preview</b> after changing the pulse window above.</li>"
                "<li>Drag the <b>Shift measured transmitted wave</b> slider below (or use "
                "<b>Auto-align (cross-correlation)</b>) until the measured curve lines up with "
                "the theoretical one.</li>"
                "<li><b>Apply shift to output-window offset</b> commits the chosen shift.</li>"
                "</ul>"
                "<p>This checks/fixes the timing alignment between bars without needing to look "
                "at the final stress-strain result.</p>"
            ),
        )
        wave_align_card.add(self.wave_align_canvas)
        wave_align_refresh_row = QHBoxLayout()
        self.wave_align_refresh_button = QPushButton("Refresh preview")
        self.wave_align_refresh_button.clicked.connect(self._on_refresh_wave_alignment)
        wave_align_refresh_row.addWidget(self.wave_align_refresh_button)
        wave_align_refresh_row.addStretch(1)
        wave_align_card.add_layout(wave_align_refresh_row)
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Shift measured transmitted wave:"))
        self.wave_align_slider = QSlider(Qt.Horizontal)
        self.wave_align_slider.setMinimum(-200)
        self.wave_align_slider.setMaximum(200)
        self.wave_align_slider.setValue(0)
        self.wave_align_slider.valueChanged.connect(self._on_wave_align_slider_changed)
        slider_row.addWidget(self.wave_align_slider, 1)
        self.wave_align_shift_label = QLabel("0 samples (0 s)")
        slider_row.addWidget(self.wave_align_shift_label)
        wave_align_card.add_layout(slider_row)
        auto_align_row = QHBoxLayout()
        self.wave_align_auto_button = QPushButton("Auto-align (cross-correlation)")
        self.wave_align_auto_button.clicked.connect(self._on_auto_align_wave)
        auto_align_row.addWidget(self.wave_align_auto_button)
        auto_align_row.addStretch(1)
        wave_align_card.add_layout(auto_align_row)
        apply_shift_row = QHBoxLayout()
        self.wave_align_apply_button = QPushButton("Apply shift to output-window offset")
        self.wave_align_apply_button.setObjectName("primary")
        self.wave_align_apply_button.clicked.connect(self._on_apply_wave_alignment)
        apply_shift_row.addWidget(self.wave_align_apply_button)
        apply_shift_row.addStretch(1)
        wave_align_card.add_layout(apply_shift_row)
        self.wave_align_status_label = QLabel("")
        self.wave_align_status_label.setObjectName("cardSubtitle")
        self.wave_align_status_label.setWordWrap(True)
        wave_align_card.add(self.wave_align_status_label)
        layout.addWidget(wave_align_card)
        self._wave_align_data = None

        run_card = Card("Run")
        run_row = QHBoxLayout()
        self.run_button = QPushButton("Process all Monte-Carlo draws")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._on_run)
        run_row.addWidget(self.run_button)
        run_row.addStretch(1)
        run_card.add_layout(run_row)
        self.progress = QProgressBar()
        run_card.add(self.progress)
        self.status_label = QLabel("")
        self.status_label.setObjectName("cardSubtitle")
        run_card.add(self.status_label)
        layout.addWidget(run_card)

        result_card = Card("Result", "Mean stress-strain curve across all Monte-Carlo draws, with the "
                                      "±1 standard-deviation band shown as the uncertainty envelope.")
        self.canvas = PlotCanvas(
            figsize=(8, 5),
            help_title="SHPB result",
            help_text=(
                "<p>Shows the mean true stress-strain curve computed by \"Process all "
                "Monte-Carlo draws\" from the SHPB wave-to-stress-strain conversion, together "
                "with its &plusmn;1 standard-deviation uncertainty band (from the Monte-Carlo "
                "spread across bar/specimen parameter draws).</p>"
                "<p>Use the toolbar to zoom/pan and check the curve's shape and any enabled "
                "corrections' effect -- the Toe Correction section below can fix a slow initial "
                "rise if the curve doesn't reach its real elastic slope cleanly.</p>"
            ),
        )
        result_card.add(self.canvas)
        layout.addWidget(result_card)
        toe_card = Card(
            "Toe (Foot) Correction",
            "If the curve above shows a slow, low-stiffness initial rise before reaching its real "
            "elastic slope (specimen/fixture seating, not a wave-timing issue), fit a tangent to the "
            "steepest, most stable part of the curve and re-zero the strain axis to where that "
            "tangent crosses zero stress -- automatically, or on a region you pick yourself. Applying "
            "this updates the result directly, so every view on Results & Export reflects it too.",
            help_html=(
                "<p>Corrects a slow, low-stiffness initial rise in the result curve above "
                "(specimen/fixture seating, not a wave-timing issue) by fitting a tangent to the "
                "steepest, most stable part of the curve and re-zeroing the strain axis to where "
                "that tangent crosses zero stress.</p>"
                "<ul>"
                "<li><b>Auto-detect linear region</b>: picks the steepest, most stable part of "
                "the curve automatically.</li>"
                "<li><b>Use selected region</b>: fits the tangent on the \"Linear region "
                "(manual)\" window you drag on the plot instead.</li>"
                "<li><b>Apply toe correction</b>: commits the fitted tangent's zero-stress "
                "crossing as the new strain origin.</li>"
                "<li><b>Reset (undo correction)</b>: restores the uncorrected curve.</li>"
                "</ul>"
                "<p><b>Technical detail:</b> applying this updates the result directly, so every "
                "view on Results &amp; Export reflects it too, not just this page's plot.</p>"
            ),
        )
        self.toe_picker = TimeWindowPicker(
            [("region", "Linear region (manual)", theme.ACCENT)], figsize=(8, 4),
            help_title="Toe (Foot) Correction",
            help_text=(
                "<p>Shows the result curve so you can fit a tangent to its real elastic "
                "slope.</p>"
                "<ul>"
                "<li><b>Auto-detect linear region</b> picks the steepest, most stable part of "
                "the curve automatically.</li>"
                "<li>Alternatively, click and drag on the plot to select a \"Linear region "
                "(manual)\" window yourself, then click <b>Use selected region</b> below.</li>"
                "</ul>"
                "<p>Either way, the fitted tangent's zero-stress crossing becomes the new strain "
                "origin once <b>Apply toe correction</b> is clicked -- <b>Reset (undo "
                "correction)</b> restores the uncorrected curve.</p>"
            ),
        )
        toe_card.add(self.toe_picker)
        toe_row = QHBoxLayout()
        self.toe_auto_button = QPushButton("Auto-detect linear region")
        self.toe_auto_button.clicked.connect(self._on_toe_auto_detect)
        toe_row.addWidget(self.toe_auto_button)
        self.toe_manual_button = QPushButton("Use selected region")
        self.toe_manual_button.clicked.connect(self._on_toe_manual_fit)
        toe_row.addWidget(self.toe_manual_button)
        toe_row.addStretch(1)
        toe_card.add_layout(toe_row)
        apply_toe_row = QHBoxLayout()
        self.toe_apply_button = QPushButton("Apply toe correction")
        self.toe_apply_button.setObjectName("primary")
        self.toe_apply_button.clicked.connect(self._on_toe_apply)
        self.toe_apply_button.setEnabled(False)
        apply_toe_row.addWidget(self.toe_apply_button)
        self.toe_reset_button = QPushButton("Reset (undo correction)")
        self.toe_reset_button.clicked.connect(self._on_toe_reset)
        apply_toe_row.addWidget(self.toe_reset_button)
        apply_toe_row.addStretch(1)
        toe_card.add_layout(apply_toe_row)
        self.toe_status_label = QLabel("")
        self.toe_status_label.setObjectName("cardSubtitle")
        self.toe_status_label.setWordWrap(True)
        toe_card.add(self.toe_status_label)
        layout.addWidget(toe_card)
        self._toe_fit = None
        # `per_draw` (the list of per-Monte-Carlo-draw dicts built in
        # `_on_run`, one entry per successful draw, each a {quantity: array}
        # dict at that draw's own reference length -- see `_store_raw_draws`)
        # kept around so a toe-correction Apply can re-cut EVERY draw's own
        # curve (not just the already-aggregated mean/std in `last_results`)
        # by the same strain offset, then re-aggregate -- see
        # `_apply_toe_correction_to_draws`/`_on_toe_apply`.
        # `_toe_per_draw_backup` holds the untouched original list, set the
        # first time Apply runs after a fresh Monte-Carlo run, for Reset.
        self._last_per_draw = None
        self._toe_per_draw_backup = None



        layout.addStretch(1)

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)

        self._restore_from_session()

    def _restore_from_session(self):
        """Re-load any files remembered from a previous session (see
        core/session.py) -- these live on this page, not on `Project`, so
        each file is just re-read through the same handler used when
        picking it live."""
        from pathlib import Path
        path = self.project.last_paths.get("shpb_time")
        if path and Path(path).exists():
            self._on_time_file(path)

        dt = self.project.time_step_overrides.get("shpb_time")
        if dt is not None:
            self.time_file_row.set_dt(dt)
            self.time_file_row.set_generate_mode(True)  # regenerates now if Input/Output Bar signals are already loaded, or on the next showEvent otherwise

    def _on_time_file(self, path):
        try:
            values = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        values.append(float(line))
            timet = np.asarray(values)
            if self.project.downsample_enabled:
                timet = downsample(timet, self.project.downsample_n)
            self._timet = timet
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Could not read time file: {exc}")
            return
        self.project.last_paths["shpb_time"] = path
        self.status_label.setText(f"Loaded {len(self._timet)} time samples from {path.split('/')[-1]}")
        self._refresh_picker()

    def _on_time_step_changed(self, dt):
        """Alternative to `_on_time_file` above: build the time vector from
        a fixed time step matching the Input/Output Bar signal length,
        instead of loading it from a file. Those signals are loaded on the
        separate Input Bar / Output Bar pages, not here, so (unlike the
        Load/Displacement Data tab) this can't watch for them changing --
        `_maybe_regenerate_time` is also called from `showEvent` so a stale
        generated vector self-corrects as soon as this tab is revisited."""
        self._maybe_regenerate_time()
        if self._timet is None:
            self.status_label.setText(
                "Load an Input Bar or Output Bar signal first (their own pages), so the generated "
                "time vector can match its length."
            )
            return
        self.project.last_paths.pop("shpb_time", None)
        self.project.time_step_overrides["shpb_time"] = dt
        self.status_label.setText(f"Generated {len(self._timet)} time samples (dt = {dt:.6g} s).")
        self._refresh_picker()

    def _on_time_mode_changed(self, generate):
        if not generate:
            self.project.time_step_overrides.pop("shpb_time", None)

    def _maybe_regenerate_time(self):
        if not self.time_file_row.is_generate_mode():
            return
        n = len(self.project.input_bar.signal) or len(self.project.output_bar.signal)
        if n == 0:
            return
        generated = self.time_file_row.build(n)
        if generated is not None:
            self._timet = generated

    def _refresh_picker(self):
        if self._timet is None:
            self.picker.show_placeholder("Load a time file above to see the signal here.")
            return
        series = []
        if len(self.project.input_bar.signal) > 0:
            series.append(("Input Bar signal", self.project.input_bar.signal, theme.ACCENT))
        if len(self.project.output_bar.signal) > 0:
            series.append(("Output Bar signal", self.project.output_bar.signal, "#4cd97b"))
        if not series:
            self.picker.show_placeholder(
                "No Input/Output Bar signal loaded yet.\nGo to the Input Bar / Output Bar page, "
                "load their signal files, then come back here."
            )
            return
        self.picker.set_signal(self._timet, series)

    def _on_window_changed(self, key, start, end):
        self.project.shpb.xd1, self.project.shpb.xd2 = start, end

    def _on_use_corrected_toggled(self, checked):
        self.project.shpb.use_corrected_signal = checked
        self.time_file_row.setVisible(not checked)
        self.picker.setVisible(not checked)
        self._refresh_corrected_status()

    def _refresh_corrected_status(self):
        if not self.use_corrected_check.isChecked():
            self.corrected_status_label.setText("")
            return
        corr = self.project.last_correction
        if corr is None:
            self.corrected_status_label.setText(
                "No corrected signal available yet -- run the Dispersion Corrections page first "
                "(Two bars / SHPB setup)."
            )
        elif corr["direct_impact"]:
            self.corrected_status_label.setText(
                "The last Corrections run was for Direct Impact (single bar), not SHPB -- run it again "
                "with \"Two bars (SHPB)\" selected."
            )
        else:
            mode_label = {0: "no correction", 1: "elastic", 2: "viscoelastic"}[corr["mode_index"]]
            self.corrected_status_label.setText(
                f"Using the corrected signal from Corrections ({mode_label}), window "
                f"{corr['xd1']:.6g} s -> {corr['xd2']:.6g} s."
            )

    def showEvent(self, event):
        super().showEvent(event)
        self._maybe_regenerate_time()  # Input/Output Bar signals may have (re)loaded since this tab was last shown
        self._refresh_picker()
        self._refresh_corrected_status()
        self.notch_card.setVisible(self.project.test_mode == "Shear-Compression/Tension (Notched)")
        self.platen_table.refresh()  # picks up values changed elsewhere (e.g. a loaded Setup Profile)
        self._refresh_sensor_align_card()
        self._refresh_settings_from_project()
        self._refresh_test_mode_label()

    def _refresh_test_mode_label(self):
        # `project.test_mode` is set once on the Test Setup page and applies to
        # EVERY Processing tab (this SHPB tab, Direct Impact, and -- unless its own
        # independent "Test kinematics" dropdown is used -- Quasi-Static too) --
        # shown here so it's obvious which kinematics this tab is about to process
        # with. Switching it for one test also silently affects any OTHER test
        # processed afterward on another tab, with no other visible cue -- this is
        # what was behind a notched-specimen (division-by-zero) crash when
        # processing a plain compression/tension/shear test right after a notched
        # one without switching Test Setup's mode back first.
        self.test_mode_label.setText(
            f"Test mode (from Test Setup page): \"{self.project.test_mode}\". If this SHPB test "
            "isn't that kind of test, change the mode on the Test Setup page first."
        )

    def _on_load_k_value_file(self, path):
        """Loads the notched-specimen k/sk fit coefficients from a plain
        text file (see core/k_value_io.py for the exact expected format --
        6 k-values, optionally followed by a 'Split = True/False' line and
        7 sk-values), instead of having to re-type each value by hand from
        a printed/exported copy. Writes straight into the k_value[]/
        sk_check/sk_value[] fields already on this page and persists them
        via `_on_settings_changed()`, exactly as if they'd been typed in.
        A file with only the 6 k-values (no split/sk section) leaves the
        split checkbox and sk_value[] fields exactly as they were -- see
        `load_k_value_file`'s docstring for why that's a `None`, not a
        default, from the loader."""
        from core.k_value_io import load_k_value_file
        try:
            k_value, sk_check, sk_value = load_k_value_file(path)
        except (OSError, ValueError) as exc:
            self.notch_file_status.setText(f"Could not load k/sk-coefficients: {exc}")
            return
        for edit, value in zip(self.k_edits, k_value):
            edit.setText(str(value))
        if sk_check is not None:
            self.sk_check.setChecked(sk_check)
        if sk_value is not None:
            for edit, value in zip(self.sk_edits, sk_value):
                edit.setText(str(value))
        self._on_settings_changed()
        detail = "k-values only (no split/sk section in this file)" if sk_value is None else "k- and sk-values"
        self.notch_file_status.setText(f"Loaded {detail} from {path.split('/')[-1]}.")

    def _on_settings_changed(self, *_args):
        """Keeps `project.shpb` live-synced with this tab's widgets --
        without this, nothing persists until "Process" is clicked, and
        simply navigating away and back would reset everything (since
        `_refresh_settings_from_project` below reads back from
        `project.shpb`, which would still hold stale/default values)."""
        s = self.project.shpb
        s.heat_check = self.heat_check.isChecked()
        s.plat_check = self.plat_check.isChecked()
        s.punch_check = self.punch_check.isChecked()
        s.inertia_check = self.inertia_check.isChecked()
        s.force_check = self.force_combo.currentIndex() + 1
        try:
            s.friction_coefficient = float(self.friction_edit.text())
        except ValueError:
            pass
        try:
            s.gage_temp_ib = float(self.gage_ib_edit.text())
        except ValueError:
            pass
        try:
            s.gage_temp_ob = float(self.gage_ob_edit.text())
        except ValueError:
            pass
        for i, e in enumerate(self.k_edits):
            try:
                s.k_value[i] = float(e.text())
            except ValueError:
                pass
        s.sk_check = self.sk_check.isChecked()
        for i, e in enumerate(self.sk_edits):
            try:
                s.sk_value[i] = float(e.text())
            except ValueError:
                pass

    def _refresh_settings_from_project(self):
        """Syncs this tab's checkboxes/fields with `project.shpb`, in case
        they were changed elsewhere (e.g. a loaded Setup Profile).

        Signals are blocked while doing this: `setChecked()`/`setText()`
        below fire `toggled`/`editingFinished` whenever they actually
        change a widget's value, which would otherwise re-enter
        `_on_settings_changed()` mid-refresh -- at that point some of
        THIS tab's own widgets have already been updated to match
        `project.shpb` and others haven't yet, so that reentrant call
        would write a half-updated, inconsistent mix back into
        `project.shpb`, silently clobbering whatever was just restored."""
        s = self.project.shpb
        widgets = [
            self.heat_check, self.plat_check, self.punch_check, self.inertia_check,
            self.force_combo, self.friction_edit, self.gage_ib_edit, self.gage_ob_edit,
            *self.k_edits, self.sk_check, *self.sk_edits, self.use_corrected_check,
        ]
        for w in widgets:
            w.blockSignals(True)
        try:
            self.heat_check.setChecked(s.heat_check)
            self.plat_check.setChecked(s.plat_check)
            self.punch_check.setChecked(s.punch_check)
            self.inertia_check.setChecked(s.inertia_check)
            self.force_combo.setCurrentIndex(s.force_check - 1)
            self.friction_edit.setText(str(s.friction_coefficient))
            self.gage_ib_edit.setText(str(s.gage_temp_ib))
            self.gage_ob_edit.setText(str(s.gage_temp_ob))
            for edit, value in zip(self.k_edits, s.k_value):
                edit.setText(str(value))
            self.sk_check.setChecked(s.sk_check)
            for edit, value in zip(self.sk_edits, s.sk_value):
                edit.setText(str(value))
            self.use_corrected_check.setChecked(s.use_corrected_signal)
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._on_use_corrected_toggled(s.use_corrected_signal)  # visibility effect, applied once, explicitly

    def _refresh_sensor_align_card(self):
        while self._sensor_align_card.body.count() > 0:
            item = self._sensor_align_card.body.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.sensor_align_rows = []

        candidates = [(i, s) for i, s in enumerate(self.project.custom_sensors)
                      if s.enabled and s.on_specimen and len(s.time) > 0]
        if not candidates:
            note = QLabel("No enabled, specimen-mounted Custom Sensor yet (see the Custom Sensors page).")
            note.setObjectName("cardSubtitle")
            self._sensor_align_card.add(note)
            return

        for idx, sensor in candidates:
            row = QHBoxLayout()
            row.addWidget(QLabel(sensor.label))
            button = QPushButton("Auto-cut using wave travel time")
            button.setEnabled(not sensor.cut_in_montecarlo)
            button.clicked.connect(lambda _checked=False, i=idx: self._on_align_sensor(i))
            row.addWidget(button)
            mc_check = QCheckBox("Include this cut in the Monte-Carlo loop")
            mc_check.setChecked(sensor.cut_in_montecarlo)
            mc_check.toggled.connect(lambda checked, i=idx: self._on_sensor_mc_toggle(i, checked))
            row.addWidget(mc_check)
            row.addStretch(1)
            self._sensor_align_card.add_layout(row)
            # On its own full-width row (not crammed next to the button in
            # the row above), with word-wrap on -- otherwise this label's
            # text (long, with several numbers) can render as overlapping/
            # garbled digits when squeezed into a narrow horizontal layout.
            status = QLabel("")
            status.setObjectName("cardSubtitle")
            status.setWordWrap(True)
            self._sensor_align_card.add(status)
            self.sensor_align_rows.append((idx, status, mc_check, button))
            if sensor.cut_in_montecarlo:
                status.setText(
                    "This sensor's cut will be recomputed for every Monte-Carlo draw when \"Process\" "
                    "is run below (see the help icon above for details)."
                )

    def _on_sensor_mc_toggle(self, sensor_index: int, checked: bool):
        sensor = self.project.custom_sensors[sensor_index]
        sensor.cut_in_montecarlo = checked
        row = next((r for r in self.sensor_align_rows if r[0] == sensor_index), None)
        if row is None:
            return
        _, status, _, button = row
        button.setEnabled(not checked)
        if checked:
            status.setText(
                "This sensor's cut will be recomputed for every Monte-Carlo draw when \"Process\" is run "
                "below, from that draw's own sampled bar/platen/specimen properties -- it always "
                "searches the sensor's original, uncut signal, so it's fine even if it was already "
                "cut once (this card's button, or the Custom Sensors page's \"Cut curve\")."
            )
        else:
            status.setText("")

    def _on_align_sensor(self, sensor_index: int):
        sensor = self.project.custom_sensors[sensor_index]
        status_label = next((r[1] for r in self.sensor_align_rows if r[0] == sensor_index), None)
        if self.use_corrected_check.isChecked():
            corr = self.project.last_correction
            if corr is None or corr.get("direct_impact") or corr.get("xd1") is None:
                if status_label:
                    status_label.setText(
                        "\"Use dispersion-corrected signal\" is checked but no valid SHPB correction is "
                        "available yet -- run the Dispersion Corrections page first."
                    )
                return
            xd1 = corr["xd1"]
        else:
            window = self.picker.get_window("xd")
            if window is None:
                if status_label:
                    status_label.setText("Select the incident pulse window above first (needed as the time reference).")
                return
            xd1 = window[0]
        try:
            ib = self.project.input_bar
            sp = self.project.specimen
            c0_sample = sample_wave_speed(sp.params["e"].mean, sp.params["rho"].mean)
            use_platens = self.plat_check.isChecked()
            plat = self.project.shpb.platen
            c0_platens = sample_wave_speed(plat.params["e"].mean, plat.params["rho"].mean) if use_platens else 0.0
            dt = specimen_sensor_delay_shpb(
                input_gage_position=ib.params["PG"].mean, c0_input_bar=ib.params["C0"].mean,
                sample_length=sp.params["L"].mean, c0_sample=c0_sample,
                use_platens=use_platens, platens_length=plat.params["L"].mean, c0_platens=c0_platens,
            )
        except (KeyError, ZeroDivisionError, ValueError) as exc:
            if status_label:
                status_label.setText(f"Could not compute the delay: {exc}")
            return
        # The incident wave reaches the input-bar gage at absolute time xd1
        # (on the same clock as this sensor's own time axis, assuming a
        # shared DAQ), THEN takes `dt.total` more to reach the specimen --
        # so the cut point on the sensor's own time axis is xd1 + dt.total,
        # not dt.total alone (that was cutting far too early).
        cut_point = xd1 + dt.total
        sensor.cut_from(cut_point)
        if status_label:
            platen_part = f" + platen {dt.platen_term:.4g} s" if use_platens else ""
            status_label.setText(
                f"Cut at t = {cut_point:.6g} s (incident wave start {xd1:.6g} s + travel time {dt.total:.6g} s = "
                f"bar {dt.bar_term:.4g} s{platen_part} + half-specimen {dt.specimen_term:.4g} s). "
                f"{len(sensor.data)} points remain."
            )

    def _on_platen_file(self, path):
        try:
            load_param_file(path, self.project.shpb.platen.params, PLATEN_PARAM_ORDER)
        except (OSError, ValueError, SyntaxError) as exc:
            self.platen_status.setText(f"Could not load platen config: {exc}")
            return
        self.project.last_paths["platen_config"] = path
        self.platen_table.refresh()
        self.platen_status.setText(f"Loaded from {path.split('/')[-1]}")

    def _test_mode_options(self):
        project = self.project
        mode = project.test_mode
        if mode == "Compression":
            return dict(type_check=1, comp_check=1 if project.geometry == "Cylindrical" else 2)
        if mode == "Tension":
            return dict(type_check=2, tens_check=1 if project.geometry == "Axisymmetric" else 2)
        if mode == "Shear-Compression/Tension (Notched)":
            return dict(type_check=3)
        # Shear
        shear_map = {"Hat-shaped": 1, "Double shear": 2, "Simple shear": 3}
        return dict(type_check=4, shear_check=shear_map.get(project.geometry, 1))

    def _get_wave_alignment_signals(self):
        """Returns (dt, time1, incident, reflected, transmitted) using
        mean bar/specimen properties -- from either the raw picker window
        or the dispersion-corrected signal, matching whichever source
        "Use dispersion-corrected signal" currently points at. Returns
        None (with a status message already set) if unavailable."""
        project = self.project
        if self.use_corrected_check.isChecked():
            corr = project.last_correction
            if corr is None or corr.get("direct_impact") or corr.get("xd1") is None:
                self.wave_align_status_label.setText(
                    "\"Use dispersion-corrected signal\" is checked but no valid SHPB correction is "
                    "available yet -- run the Dispersion Corrections page first."
                )
                return None
            result = corr["result"]
            time1 = np.asarray(result.time)
            incident, reflected, transmitted = result.incident_corrected, result.reflected_corrected, result.transmitted_corrected
        else:
            window = self.picker.get_window("xd")
            if window is None or self._timet is None:
                self.wave_align_status_label.setText("Select the incident pulse window above (and load a time file) first.")
                return None
            xd1, xd2 = window
            time1, incident, reflected, transmitted = _cut_two_bar_windows(
                project.input_bar, project.output_bar, project.specimen, self._timet, xd1, xd2, np.array([1.0])
            )
        if len(time1) < 2:
            self.wave_align_status_label.setText("The cut window is too short to preview.")
            return None
        dt = float(time1[1] - time1[0])
        return dt, time1, np.asarray(incident), np.asarray(reflected), np.asarray(transmitted)

    def _on_refresh_wave_alignment(self):
        project = self.project
        signals = self._get_wave_alignment_signals()
        if signals is None:
            return
        dt, time1, incident, reflected, transmitted = signals

        try:
            ib, ob, sp = project.input_bar, project.output_bar, project.specimen
            area_ib = math.pi * (ib.params["D"].mean / 2) ** 2
            f_incident_reflected = ib.params["e"].mean * area_ib * (incident + reflected) * ib.params["SG_coeff"].mean
            area_ob = math.pi * (ob.params["D"].mean / 2) ** 2
            f_transmitted_measured = ob.params["e"].mean * area_ob * transmitted * ob.params["SG_coeff"].mean

            use_platens = self.plat_check.isChecked()
            plat = project.shpb.platen
            prediction = predicted_transmitted_wave(
                f_incident_reflected,
                ib.params["e"].mean, ib.params["rho"].mean, ib.params["D"].mean,
                sp.params["e"].mean, sp.params["rho"].mean, sp.params["D"].mean,
                ob.params["e"].mean, ob.params["rho"].mean, ob.params["D"].mean,
                use_platens=use_platens,
                platen_modulus=plat.params["e"].mean, platen_density=plat.params["rho"].mean,
                platen_diameter=plat.params["D"].mean,
            )
        except (KeyError, ZeroDivisionError, ValueError) as exc:
            self.wave_align_status_label.setText(f"Could not compute the theoretical wave: {exc}")
            return

        self._wave_align_data = {
            "dt": dt, "time1": time1,
            "f_incident_reflected": np.abs(f_incident_reflected),
            "f_transmitted_measured": np.abs(f_transmitted_measured),
            "f_transmitted_theoretical": np.abs(prediction.predicted_transmitted),
            "t_total": prediction.total_transmission_coefficient,
        }
        self.wave_align_slider.blockSignals(True)
        self.wave_align_slider.setValue(0)
        self.wave_align_slider.blockSignals(False)
        self._draw_wave_alignment(shift_samples=0)
        self.wave_align_status_label.setText(
            f"Total transmission coefficient (impedance-based) = {prediction.total_transmission_coefficient:.4g}. "
            "Click \"Auto-align (cross-correlation)\" for a quantitative best fit (compares the "
            "overall, smoothed rise/plateau/decay shape, not local wiggles), or fine-tune manually "
            "with the slider."
        )

    def _draw_wave_alignment(self, shift_samples: int):
        data = self._wave_align_data
        if data is None:
            return
        time1 = data["time1"]
        self.wave_align_canvas.clear()
        ax = self.wave_align_canvas.figure.add_subplot(111)
        ax.plot(time1, data["f_incident_reflected"], color=theme.ACCENT, label="Incident + reflected (measured)")
        ax.plot(time1, data["f_transmitted_theoretical"], color="black", linestyle="--",
                 label="Transmitted (theoretical, elastic)")

        measured = data["f_transmitted_measured"]
        shifted = np.roll(measured, shift_samples)
        if shift_samples > 0:
            shifted[:shift_samples] = measured[0]
        elif shift_samples < 0:
            shifted[shift_samples:] = measured[-1]
        ax.plot(time1, shifted, color="red", label="Transmitted (measured, shifted)")

        ax.set_xlabel("time (s)")
        ax.set_ylabel("Force (N)")
        ax.legend(fontsize=8)
        self.wave_align_canvas.refresh()

    def _on_auto_align_wave(self):
        if self._wave_align_data is None:
            self.wave_align_status_label.setText("Refresh the preview first.")
            return
        data = self._wave_align_data
        max_shift = self.wave_align_slider.maximum()
        shift = best_alignment_shift(data["f_transmitted_measured"], data["f_transmitted_theoretical"], max_shift)
        shift = max(self.wave_align_slider.minimum(), min(self.wave_align_slider.maximum(), shift))
        self.wave_align_slider.setValue(shift)  # triggers _on_wave_align_slider_changed, which redraws
        self.wave_align_status_label.setText(
            f"Auto-aligned by cross-correlation: shift = {shift} samples ({shift * data['dt']:.6g} s). "
            "Check the plot, adjust the slider if needed, then apply."
        )

    def _on_wave_align_slider_changed(self, value: int):
        if self._wave_align_data is None:
            return
        dt = self._wave_align_data["dt"]
        self._draw_wave_alignment(shift_samples=value)
        self.wave_align_shift_label.setText(f"{value} samples ({value * dt:.4g} s)")

    def _on_apply_wave_alignment(self):
        if self._wave_align_data is None:
            self.wave_align_status_label.setText("Refresh the preview first.")
            return
        shift_samples = self.wave_align_slider.value()
        dt = self._wave_align_data["dt"]
        self.project.shpb.output_window_offset = shift_samples * dt
        self.wave_align_status_label.setText(
            f"Applied: output-window offset set to {shift_samples * dt:.6g} s ({shift_samples} samples). "
            "This replaces any previous offset -- click \"Process all Monte-Carlo draws\" below to use it."
        )


    def _on_run(self):
        project = self.project
        use_corrected = self.use_corrected_check.isChecked()
        corr = project.last_correction if use_corrected else None

        if use_corrected:
            if corr is None:
                self.status_label.setText("No corrected signal available -- run the Dispersion Corrections page first.")
                return
            if corr["direct_impact"]:
                self.status_label.setText("The stored corrected signal is for Direct Impact, not SHPB. Re-run Corrections with \"Two bars (SHPB)\".")
                return
            if corr["timet"] is None:
                self.status_label.setText("The stored corrected signal has no time base recorded -- re-run Corrections.")
                return
            xd1, xd2 = corr["xd1"], corr["xd2"]
            timet = corr["timet"]
        else:
            window = self.picker.get_window("xd")
            if window is None:
                self.status_label.setText("Select the pulse window by dragging on the signal plot first.")
                return
            xd1, xd2 = window
            if self._timet is None:
                self.status_label.setText("Load a time file first.")
                return
            timet = self._timet

        # `corr["timet"]` can come back as a plain Python list, not an
        # ndarray, after a session/profile restore: _last_correction_to_
        # dict (core/session.py) converts it to a list for JSON, and
        # _last_correction_from_dict never converts it back. Both
        # process_shpb.py's own `time1 = timet[idx_i1:idx_i2] - xd1` and
        # the sensor_target_time computation below do that same
        # subtraction, which raises a TypeError against a plain list --
        # inside process_shpb.py that got silently swallowed per draw (as
        # a failed draw, see run_map below), while here it would crash
        # the whole page. Coercing once, up front, fixes both.
        timet = np.asarray(timet)

        try:
            friction_coeff = float(self.friction_edit.text())
            gage_ib, gage_ob = float(self.gage_ib_edit.text()), float(self.gage_ob_edit.text())
            k_value = [float(e.text()) for e in self.k_edits]
            sk_value = [float(e.text()) for e in self.sk_edits]
        except ValueError:
            self.status_label.setText("Friction, temperature and k-coefficient fields must be numeric.")
            return

        if len(project.input_bar.signal) == 0 or len(project.output_bar.signal) == 0:
            self.status_label.setText("Load the Input Bar and Output Bar signals first (Input Bar / Output Bar pages).")
            return

        project.expand_montecarlo()
        n = project.n_montecarlo

        mode_opts = self._test_mode_options()
        options = SHPBOptions(
            heat_check=self.heat_check.isChecked(),
            type_bars_check=project.thermal.bar_type,
            plat_check=self.plat_check.isChecked(),
            punch_check=self.punch_check.isChecked(),
            inertia_check=self.inertia_check.isChecked(),
            force_check=self.force_combo.currentIndex() + 1,
            sk_check=self.sk_check.isChecked(),
            disp_ela=use_corrected,
            **mode_opts,
        )
        AA = None
        if use_corrected:
            result = corr["result"]
            AA = [result.incident_corrected, result.reflected_corrected, result.transmitted_corrected]

        plat_arrays = {}
        if options.plat_check:
            plat_arrays = project.shpb.platen.expand(n)

        friction_arr = np.full(n, friction_coeff)
        sg_unc_d = np.ones(n)
        t0_arr = np.full(n, project.thermal.initial_temperature_k)

        # Specimen-mounted sensors whose cut is to be recomputed for every
        # Monte-Carlo draw (CustomSensor.cut_in_montecarlo, ticked on the
        # "Specimen-Mounted Sensor Alignment" card above) instead of the
        # single deterministic cut the "Auto-cut" button applies. This is
        # cheap (index search + interpolation, see core.custom_sensor.
        # cut_and_resample) and doesn't touch the wave physics at all, so
        # it's computed here in the UI thread rather than threaded through
        # process_shpb()'s own (already dense, pickled-per-worker) context.
        # Uses effective_raw_time()/len() rather than s.time so a sensor
        # that was already trimmed by the one-shot "Auto-cut"/"Cut curve"
        # button is still eligible here -- see calibrated_data_raw()'s
        # docstring (core/custom_sensor.py) for why the per-draw cut below
        # must search the pristine signal, not whatever is left of it.
        mc_sensors = [(i, s) for i, s in enumerate(project.custom_sensors)
                      if s.enabled and s.on_specimen and s.cut_in_montecarlo and len(s.effective_raw_time()) > 0]
        sensor_cut_points = {}
        sensor_unc_halfwidth = {}
        sensor_target_time = None
        if mc_sensors:
            for idx, sensor in mc_sensors:
                sensor_cut_points[idx] = specimen_sensor_cut_points_shpb(
                    xd1=xd1,
                    input_gage_position_pa=project.input_bar.PG_pa, c0_input_bar_pa=project.input_bar.C0_pa,
                    sample_length_pa=project.specimen.L_pa, elastic_modulus_pa=project.specimen.e_pa,
                    density_pa=project.specimen.rho_pa,
                    use_platens=options.plat_check,
                    platens_length_pa=plat_arrays.get("L"), platens_elastic_modulus_pa=plat_arrays.get("e"),
                    platens_density_pa=plat_arrays.get("rho"),
                )
                # Declared measurement uncertainty (CustomSensor.
                # uncertainty_band_raw(), on the calibrated_data_raw()
                # scale so it lines up with the pristine signal cut below),
                # cut and resampled with the SAME per-draw cut points/
                # target time as the signal itself below, so the
                # Monte-Carlo result's std can combine draw-to-draw
                # alignment jitter with this fixed, declared band instead
                # of dropping it silently.
                lo, hi = sensor.uncertainty_band_raw()
                sensor_unc_halfwidth[idx] = (np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)) / 2.0
            # The pulse-window time base every draw's own result shares --
            # identical to process_shpb.py's own `time1 = timet[idx_i1:
            # idx_i2] - xd1` (see core/process_shpb.py), computed here too
            # since it depends only on (timet, xd1, xd2), not on any
            # per-draw quantity, so there's no need to wait on a draw to
            # know it.
            _idx_i1 = int(np.searchsorted(timet, xd1, side="right") - 1)
            _idx_i2 = int(np.searchsorted(timet, xd2, side="right") - 1)
            sensor_target_time = timet[_idx_i1:_idx_i2] - xd1

        # Confinement (quasi-oedometric cell) quantities, computed per
        # Monte-Carlo draw below instead of post-hoc in Results & Export --
        # see core.confinement.per_draw_confinement_quantities. Only
        # possible when the confinement-cell sensor is itself one of the
        # `mc_sensors` above (on_specimen + cut_in_montecarlo ticked), since
        # that's what gives every draw its own real cut/resampled cell
        # signal (draw[f"sensor_{idx}_signal"]) to combine with that same
        # draw's own axial stress/strain -- otherwise this falls back to
        # Results & Export's existing synthetic-redraw approximation,
        # unchanged.
        confinement_prep = _prepare_confinement(project)
        confinement_active = False
        cell_idx = poly_ini = poly_esp = None
        if confinement_prep is not None:
            cell_idx, _cell_sensor, poly_ini, poly_esp = confinement_prep
            confinement_active = cell_idx in {i for i, _ in mc_sensors}
        specimen_initial_length = project.specimen.params["L"].mean

        self.progress.setMaximum(n)
        all_stress, all_strain = [], []
        per_draw = []
        errors = 0
        last_error = ""

        # One process_shpb() call per Monte-Carlo draw, none depending on
        # any other -- the textbook embarrassingly-parallel case. context
        # holds every argument except the draw index `d` itself, so it's
        # pickled once per worker process (see core/parallel.py), not once
        # per draw. n_workers=1 (the default) runs the exact same calls,
        # in the same order, in this process -- identical to before this
        # setting existed.
        shpb_context = dict(
            options=options, IB=project.input_bar, OB=project.output_bar, specimen=project.specimen,
            timet=timet, xd1=xd1, xd2=xd2,
            SG_unc_d=sg_unc_d, T0_pa=t0_arr, friction=friction_arr,
            k_value=k_value, sk_value=sk_value,
            plat_nu=plat_arrays.get("nu"), plat_rho=plat_arrays.get("rho"),
            plat_emod=plat_arrays.get("e"), plat_Dia=plat_arrays.get("D"), plat_Len=plat_arrays.get("L"),
            Tm_value=(project.thermal.melting_temperature_k, project.thermal.melting_exponent),
            T_gages_value=(gage_ib, gage_ob),
            material_table_ib=project.material_table_ib, material_table_ob=project.material_table_ob,
            AA=AA,
            output_window_offset=project.shpb.output_window_offset,
        )

        def _report_progress(done, total):
            if done % max(1, total // 100) == 0 or done == total:
                self.progress.setValue(done)
                QApplication.processEvents()

        results = run_map(
            _process_shpb_task, list(range(n)), n_workers=project.n_workers,
            context=shpb_context, progress_callback=_report_progress,
        )

        for d, res, exc in results:
            if exc is not None:  # noqa: BLE001 - keep the last message so a total failure is diagnosable
                errors += 1
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            if res.stress and res.strain:
                all_stress.append(res.stress)
                all_strain.append(res.strain)
                draw = {
                    "strain": res.strain, "stress": res.stress, "strain_rate": res.strain_rate,
                    "nominal_strain": res.nominal_strain, "nominal_stress": res.nominal_stress,
                    "nominal_strain_rate": res.nominal_strain_rate,
                    "temperature": res.temperature, "displacement": res.displacement,
                    "force_input": res.force_input, "force_output": res.force_output,
                    "force_average": res.force_average, "time": res.time,
                    "inertia_stress": res.inertia_stress, "friction_stress": res.friction_stress,
                    "punch_strain": res.punch_strain,
                    "balance_quality_force": res.balance_quality_force,
                    "balance_quality_waves": res.balance_quality_waves,
                    "balance_quality_force_instant": res.balance_quality_force_instant,
                }
                for idx, sensor in mc_sensors:
                    raw_time = sensor.effective_raw_time()
                    draw[f"sensor_{idx}_signal"] = cut_and_resample(
                        raw_time, sensor.calibrated_data_raw(), sensor_cut_points[idx][d], sensor_target_time
                    )
                    draw[f"sensor_{idx}_signal_unc"] = cut_and_resample(
                        raw_time, sensor_unc_halfwidth[idx], sensor_cut_points[idx][d], sensor_target_time
                    )
                if confinement_active:
                    try:
                        strain_for_conf = draw.get("nominal_strain", draw["strain"])
                        stress_for_conf = draw.get("nominal_stress", draw["stress"])
                        draw.update(per_draw_confinement_quantities(
                            strain_for_conf, stress_for_conf, draw[f"sensor_{cell_idx}_signal"],
                            project.confinement, specimen_initial_length,
                            poly_ini=poly_ini, poly_esp=poly_esp, time=draw.get("time"),
                        ))
                    except (ZeroDivisionError, ValueError, np.linalg.LinAlgError):
                        pass  # this one draw's confinement calc failed -- a rare numeric edge case
                per_draw.append(draw)

        self.progress.setValue(n)
        if not all_stress:
            self.status_label.setText(f"No draws completed successfully ({errors}/{n} failed). Last error: {last_error}")
            return

        min_len = min(len(s) for s in all_stress)
        stress_arr = np.array([s[:min_len] for s in all_stress])
        strain_arr = np.array([s[:min_len] for s in all_strain])
        mean_strain = strain_arr.mean(axis=0)
        mean_stress = stress_arr.mean(axis=0)
        std_stress = stress_arr.std(axis=0)

        confinement_note = ""
        if project.confinement.enabled:
            if confinement_active:
                confinement_note = (
                    " Confinement quantities computed per Monte-Carlo draw (real propagation) -- "
                    "Results & Export will use this real uncertainty instead of its synthetic approximation."
                )
            else:
                confinement_note = (
                    " Confinement enabled, but not computed per draw here -- tick \"Include this cut in the "
                    "Monte-Carlo loop\" for the confinement-cell sensor above (Specimen-Mounted Sensor "
                    "Alignment) to get real per-draw uncertainty in Results & Export instead of its "
                    "synthetic approximation."
                )
        self.status_label.setText(f"Processed {len(all_stress)}/{n} draws ({errors} failed).{confinement_note}")
        quantities = _aggregate_quantities(per_draw)
        self.project.last_results["SHPB"] = {"label": "SHPB", **quantities}
        _store_raw_draws(self.project, "SHPB", per_draw)
        # A fresh run supersedes any pending toe-correction state from a
        # previous curve -- both the per-draw list a later Apply would
        # re-cut, and any backup kept for Reset.
        self._last_per_draw = per_draw
        self._toe_per_draw_backup = None
        self.canvas.clear()
        ax = self.canvas.figure.add_subplot(111)
        sigma = self.project.uncertainty_sigma
        ax.plot(mean_strain, mean_stress, "b-", label="Mean")
        ax.fill_between(mean_strain, mean_stress - sigma * std_stress, mean_stress + sigma * std_stress,
                         color="b", alpha=0.2, label=f"±{sigma}σ")
        ax.set_xlabel("Strain (-)")
        ax.set_ylabel("Stress (MPa)")
        ax.legend()
        self.canvas.refresh()
        self._toe_fit = None
        self.toe_apply_button.setEnabled(False)
        self._refresh_toe_picker()

    def _toe_strain_stress(self):
        result = self.project.last_results.get("SHPB")
        if result is None:
            return None
        strain_key = "nominal_strain" if "nominal_strain" in result else "strain"
        stress_key = "nominal_stress" if "nominal_stress" in result else "stress"
        if strain_key not in result or stress_key not in result:
            return None
        return result, strain_key, stress_key, np.asarray(result[strain_key]), np.asarray(result[stress_key])

    def _refresh_toe_picker(self):
        info = self._toe_strain_stress()
        if info is None:
            self.toe_picker.show_placeholder("Run the Monte-Carlo processing above first to see the curve here.")
            return
        _, _, _, strain, stress = info
        self.toe_picker.set_signal(strain, [("Stress vs strain", stress, theme.ACCENT)],
                                    xlabel="strain (-)", primary_ylabel="stress (MPa)")
        self.toe_status_label.setText("")

    def _draw_toe_fit(self, strain, stress, fit):
        self.toe_picker.set_signal(strain, [("Stress vs strain", stress, theme.ACCENT)],
                                    xlabel="strain (-)", primary_ylabel="stress (MPa)")
        ax = self.toe_picker._ax
        if ax is None:
            return
        x_region = strain[fit.start_index:fit.end_index]
        ax.plot(x_region, stress[fit.start_index:fit.end_index], color="orange", linewidth=2, label="Fitted region")
        x_line = np.array([fit.strain_offset, strain.max()])
        y_line = fit.slope * (x_line - fit.strain_offset)
        ax.plot(x_line, y_line, color="lime", linestyle="--", label="Tangent")
        ax.axvline(fit.strain_offset, color="lime", linestyle=":", linewidth=1)
        ax.legend(fontsize=8)
        self.toe_picker.canvas.refresh()

    def _on_toe_auto_detect(self):
        info = self._toe_strain_stress()
        if info is None:
            self.toe_status_label.setText("Run the Monte-Carlo processing above first.")
            return
        _, _, _, strain, stress = info
        try:
            fit = detect_linear_region(strain, stress)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.toe_status_label.setText(f"Could not auto-detect a linear region: {exc}")
            return
        self._toe_fit = fit
        self.toe_apply_button.setEnabled(True)
        self._draw_toe_fit(strain, stress, fit)
        self.toe_status_label.setText(
            f"Auto-detected: slope = {fit.slope:.6g} MPa, R\u00b2 = {fit.r_squared:.4g}, strain offset = "
            f"{fit.strain_offset:.6g}. Check the fit (orange = region used, green = tangent), then apply, "
            "or drag a region yourself and click \"Use selected region\" instead."
        )

    def _on_toe_manual_fit(self):
        info = self._toe_strain_stress()
        if info is None:
            self.toe_status_label.setText("Run the Monte-Carlo processing above first.")
            return
        _, _, _, strain, stress = info
        window = self.toe_picker.get_window("region")
        if window is None:
            self.toe_status_label.setText("Drag on the plot to select the linear region first.")
            return
        start_idx = int(np.searchsorted(strain, window[0]))
        end_idx = int(np.searchsorted(strain, window[1]))
        if end_idx - start_idx < 2:
            self.toe_status_label.setText("Selected region is too narrow -- drag a wider window.")
            return
        try:
            fit = fit_linear_region(strain, stress, start_idx, end_idx)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.toe_status_label.setText(f"Could not fit the selected region: {exc}")
            return
        self._toe_fit = fit
        self.toe_apply_button.setEnabled(True)
        self._draw_toe_fit(strain, stress, fit)
        self.toe_status_label.setText(
            f"Fitted to your selection: slope = {fit.slope:.6g} MPa, R\u00b2 = {fit.r_squared:.4g}, "
            f"strain offset = {fit.strain_offset:.6g}. Check the fit, then apply."
        )

    def _on_toe_apply(self):
        if self._toe_fit is None:
            self.toe_status_label.setText("Auto-detect or manually fit a linear region first.")
            return
        info = self._toe_strain_stress()
        if info is None:
            return
        result, strain_key, stress_key, strain, stress = info
        if not self._last_per_draw:
            self.toe_status_label.setText(
                "No per-draw Monte-Carlo data to re-cut -- re-run the processing above first."
            )
            return

        # `self._toe_per_draw_backup` holds the ORIGINAL, pre-toe-correction
        # per-draw ensemble, set once so Reset always restores exactly what
        # the last Monte-Carlo run produced, even after several successive
        # Applies (which -- like the old single-curve code -- stack onto
        # whatever `self._last_per_draw` currently holds).
        if self._toe_per_draw_backup is None:
            self._toe_per_draw_backup = self._last_per_draw

        # Nominal strain shifts/trims linearly; true (logarithmic) strain
        # does NOT (its relation to nominal strain is nonlinear), so if
        # both exist it must be RECOMPUTED from the corrected nominal
        # strain instead of being shifted by the same amount -- likewise
        # for a shear test, where true strain is simply equal to nominal
        # strain (no log at all): infer_true_strain_relation handles all
        # three cases. Computed once, from the current aggregate curve, and
        # reused for every draw (the strain/stress *relation* is a fixed
        # property of the test kinematics, not something that varies
        # Monte-Carlo draw to draw).
        has_nominal, has_true = "nominal_strain" in result, "strain" in result
        relation = infer_true_strain_relation(result["nominal_strain"], result["strain"]) if (has_nominal and has_true) else None

        # Re-cut EVERY Monte-Carlo draw's own curve by this same fitted
        # strain offset -- not just the aggregated mean/std curve -- so
        # Results & Export (confinement-cell/laser quantities included,
        # since they share each draw's own strain-length reference array)
        # and the Covariance Analysis page's uncertainty propagation both
        # become consistent with this cut, instead of only the plotted
        # mean curve reflecting it.
        strain_offset = self._toe_fit.strain_offset
        new_per_draw, n_removed = _apply_toe_correction_to_draws(
            self._last_per_draw, strain_key, strain_offset, relation
        )
        if not new_per_draw:
            self.toe_status_label.setText(
                "Every Monte-Carlo draw's own curve sits below the fitted toe offset -- nothing to apply. "
                "Try a smaller/different fitted region."
            )
            return

        # `toe_n_removed` is a running total (across possibly several
        # successive Applies) of how many leading samples were cut from
        # the shared time-index grid -- a plain int, not an ndarray, so it
        # survives untouched by _aggregate_quantities/_apply_toe_
        # correction_to_draws (which only ever touch same-length array
        # fields). Results & Export's _refresh_confinement reads it back to
        # keep a confinement-cell/laser sensor's RAW (non-Monte-Carlo-cut)
        # fallback signal aligned with this now-shorter curve.
        prior_toe_n_removed = int(result.get("toe_n_removed", 0))
        quantities = _aggregate_quantities(new_per_draw)
        # Keep the curves as they were BEFORE the toe correction (the
        # untouched original ensemble, whatever the number of successive
        # Applies) as "<quantity>_pre_toe" columns -- exported by Results &
        # Export next to the corrected ones, and plottable there.
        self.project.last_results["SHPB"] = {
            "label": "SHPB", **quantities, **_pre_toe_quantities(self._toe_per_draw_backup),
            "toe_n_removed": prior_toe_n_removed + n_removed,
        }
        _store_raw_draws(self.project, "SHPB", new_per_draw)
        n_dropped = len(self._last_per_draw) - len(new_per_draw)
        self._last_per_draw = new_per_draw

        self._toe_fit = None
        self.toe_apply_button.setEnabled(False)
        dropped_note = (
            f" ({n_dropped} outlier draw(s) dropped -- never crossed the offset, or left too little "
            "data above it to keep)"
        ) if n_dropped else ""
        self.toe_status_label.setText(
            f"Applied: strain re-zeroed ({relation or 'unknown'} relation) and re-cut for every Monte-Carlo "
            f"draw individually (~{n_removed} toe points removed per draw, median), then re-aggregated"
            f"{dropped_note}. Results & Export (including confinement-cell/laser quantities) and Covariance "
            "Analysis both reflect this cut now. Use \"Reset\" to undo."
        )
        self._refresh_toe_picker()

    def _on_toe_reset(self):
        if self._toe_per_draw_backup is None:
            self.toe_status_label.setText("Nothing to reset.")
            return
        self._last_per_draw = self._toe_per_draw_backup
        quantities = _aggregate_quantities(self._last_per_draw)
        self.project.last_results["SHPB"] = {"label": "SHPB", **quantities}
        _store_raw_draws(self.project, "SHPB", self._last_per_draw)
        self._toe_per_draw_backup = None
        self._toe_fit = None
        self.toe_apply_button.setEnabled(False)
        self.toe_status_label.setText(
            "Toe correction undone -- restored the original curve for every Monte-Carlo draw."
        )
        self._refresh_toe_picker()

class _DITab(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._time_pulse = None

        scroll, layout = make_scroll_page(
            "",
            "Direct Impact processing: the specimen's incident velocity comes from a separate "
            "measurement (e.g. a laser gate) instead of an input-bar wave; only the transmitted "
            "wave on the Output Bar is used.",
        )
        layout.takeAt(0).widget().deleteLater()

        self.test_mode_label = QLabel("")
        self.test_mode_label.setObjectName("cardSubtitle")
        self.test_mode_label.setWordWrap(True)
        layout.addWidget(self.test_mode_label)

        pulse = Card(
            "Pulse window",
            "Click and drag on the Output Bar signal below to select the transmitted "
            "pulse window. The incident velocity profile is overlaid (dashed, right-hand "
            "axis) so you can check/adjust its alignment with the bar gage signal.",
            help_html=(
                "<p>Sets the timing/source used for the Direct-Impact wave-to-stress-strain "
                "conversion (see the plot's own help below for how to drag the windows).</p>"
                "<ul>"
                "<li><b>Use dispersion-corrected signal</b>: recommended -- otherwise the raw, "
                "uncorrected wave amplitude is used directly.</li>"
                "<li><b>Time file</b>: load a time column, or generate one from a fixed time "
                "step.</li>"
                "<li><b>Incident velocity file</b>: the specimen's incident velocity, measured "
                "separately (e.g. a laser gate) since there is no input-bar wave to derive it "
                "from in Direct Impact.</li>"
                "<li><b>Cut velocity profile at selected start / shift by (s)</b>: trims or "
                "shifts the velocity profile to line it up with the bar gage signal.</li>"
                "<li><b>Incident-velocity calibration factor</b>: a multiplicative correction "
                "applied to the loaded velocity values.</li>"
                "</ul>"
            ),
        )
        self.use_corrected_check = QCheckBox(
            "Use dispersion-corrected signal from the Corrections page (recommended -- otherwise the "
            "raw, uncorrected wave amplitude is used)"
        )
        self.use_corrected_check.setChecked(project.shpb.use_corrected_signal)
        self.use_corrected_check.toggled.connect(self._on_use_corrected_toggled)
        pulse.add(self.use_corrected_check)
        self.corrected_status_label = QLabel("")
        self.corrected_status_label.setObjectName("cardSubtitle")
        pulse.add(self.corrected_status_label)
        self.time_file_row = TimeSourceRow("Time file (.txt)")
        self.time_file_row.file_selected.connect(self._on_time_file)
        self.time_file_row.dtChanged.connect(self._on_time_step_changed)
        self.time_file_row.modeChanged.connect(self._on_time_mode_changed)
        pulse.add(self.time_file_row)
        self.velocity_file_row = FileSelectRow("Incident velocity file (e.g. laser gate, .txt)")
        self.velocity_file_row.file_selected.connect(self._on_velocity_file)
        pulse.add(self.velocity_file_row)
        self.picker = TimeWindowPicker([
            ("xd", "Pulse window", theme.ACCENT),
            ("velocity_cut", "Velocity start cut", "#ff9f43"),
        ],
            help_title="Pulse window",
            help_text=(
                "<p>Shows the Output Bar signal (dispersion-corrected if \"Use "
                "dispersion-corrected signal\" above is checked), with the incident velocity "
                "profile overlaid as a dashed line on the right-hand axis.</p>"
                "<p>Select which window to set from the dropdown above the plot, then click and "
                "drag on the trace:</p>"
                "<ul>"
                "<li><b>Pulse window</b> marks the transmitted pulse used for the "
                "Direct-Impact stress-strain conversion.</li>"
                "<li><b>Velocity start cut</b> marks where the incident velocity profile should "
                "start -- check/adjust its alignment with the bar gage signal here before "
                "running.</li>"
                "</ul>"
            ),
        )
        self.picker.windowChanged.connect(self._on_window_changed)
        pulse.add(self.picker)
        velocity_cut_row = QHBoxLayout()
        self.cut_velocity_button = QPushButton("Cut velocity profile at selected start")
        self.cut_velocity_button.clicked.connect(self._on_cut_velocity)
        velocity_cut_row.addWidget(self.cut_velocity_button)
        velocity_cut_row.addWidget(QLabel("or shift by (s):"))
        self.velocity_shift_edit = QLineEdit("0.0")
        self.velocity_shift_edit.setMaximumWidth(90)
        velocity_cut_row.addWidget(self.velocity_shift_edit)
        self.apply_velocity_shift_button = QPushButton("Apply")
        self.apply_velocity_shift_button.clicked.connect(self._on_apply_velocity_shift)
        velocity_cut_row.addWidget(self.apply_velocity_shift_button)
        self.reset_velocity_cut_button = QPushButton("Reset cut")
        self.reset_velocity_cut_button.clicked.connect(self._on_reset_velocity_cut)
        velocity_cut_row.addWidget(self.reset_velocity_cut_button)
        velocity_cut_row.addStretch(1)
        pulse.add_layout(velocity_cut_row)
        self.velocity_cut_status_label = QLabel("")
        self.velocity_cut_status_label.setObjectName("cardSubtitle")
        pulse.add(self.velocity_cut_status_label)
        self.f_inc_velocity_edit = QLineEdit("1.0")
        pulse.add(LabeledField("Incident-velocity calibration factor", self.f_inc_velocity_edit))
        layout.addWidget(pulse)
        # Qt's toggled signal doesn't fire from setChecked() if the value
        # doesn't change from the widget's own default -- apply the
        # visibility effect explicitly here too, so a restored "checked"
        # state (e.g. from a loaded test/profile) actually hides the raw-
        # signal controls, not just show the checkbox as ticked.
        self._on_use_corrected_toggled(self.use_corrected_check.isChecked())

        corrections = Card(
            "Corrections",
            "Toggle the physical corrections to apply.",
            help_html=(
                "<p>Each checkbox enables an additional physical correction on top of the base "
                "Direct-Impact wave-to-stress-strain conversion.</p>"
                "<ul>"
                "<li><b>Bar-temperature-gradient correction</b>: compensates for a heated "
                "specimen conducting heat into the output bar end -- needs the parameter in the "
                "\"Bar-temperature-gradient parameters\" card below.</li>"
                "<li><b>Platens</b>: accounts for a platen inserted between the bar and the "
                "specimen -- needs the \"Platens\" card below.</li>"
                "<li><b>Punch (indentation) correction</b>: compensates for local indentation "
                "of the specimen face by the bar end.</li>"
                "<li><b>Inertia correction</b>: compensates for the specimen's own radial/axial "
                "inertia at high strain rates.</li>"
                "<li><b>Friction coefficient</b>: used by the punch/inertia corrections' "
                "friction terms.</li>"
                "</ul>"
            ),
        )
        self.heat_check = QCheckBox("Bar-temperature-gradient correction (heated specimen)")
        self.plat_check = QCheckBox("Platens (between bar and specimen)")
        self.punch_check = QCheckBox("Punch (indentation) correction")
        self.inertia_check = QCheckBox("Inertia correction")
        for cb in (self.heat_check, self.plat_check, self.punch_check, self.inertia_check):
            corrections.add(cb)
        self.friction_edit = QLineEdit(str(project.shpb.friction_coefficient))
        corrections.add(LabeledField("Friction coefficient", self.friction_edit))
        layout.addWidget(corrections)

        sensor_align = Card(
            "Specimen-Mounted Sensor Alignment",
            "A sensor glued directly on the specimen (Custom Sensors page, \"mounted directly on the "
            "specimen\") doesn't start recording the test at the same instant as the output bar's "
            "gage -- the wave still has to travel from the gage to the specimen first. This cuts/re-"
            "zeroes such a sensor's signal by that travel time -- by default computed once from the "
            "mean bar/platen/specimen properties, or, if \"Include this cut in the Monte-Carlo loop\" "
            "is ticked below, recomputed for every Monte-Carlo draw from that draw's own sampled "
            "properties, so the cut's own uncertainty is propagated into the result.",
            help_html=(
                "<p>Applies only to a sensor configured as \"mounted directly on the specimen\" "
                "on the Custom Sensors page.</p>"
                "<ul>"
                "<li>The stress wave takes a finite time to travel from the output bar's gage to "
                "the specimen itself, so a specimen-mounted sensor's own clock starts \"late\" "
                "relative to the bar gage.</li>"
                "<li>This card cuts/re-zeroes that sensor's signal by the computed travel time, "
                "so it lines up with the rest of the test data.</li>"
                "<li><b>Auto-cut using wave travel time</b>: cuts once, immediately, from the mean "
                "bar/platen/specimen properties -- quick, but ignores their own Monte-Carlo "
                "uncertainty.</li>"
                "<li><b>Include this cut in the Monte-Carlo loop</b>: instead of a single cut "
                "applied to every draw, recomputes the cut point for EACH draw from that draw's "
                "own sampled bar/platen/specimen properties, and resamples the sensor's (cut, "
                "re-zeroed) signal onto the run's own time base -- so any test with a different "
                "sample rate/DAQ than the bars lines up point-for-point with the rest of the "
                "results. The resulting curve appears as its own quantity (\"<i>sensor label</i> "
                "(Monte-Carlo cut)\") in Results &amp; Export, with a proper Monte-Carlo mean and "
                "std-dev band -- not just this card's single deterministic cut. Only the cutting "
                "itself is redone per draw; nothing else about the sensor (calibration factor, "
                "declared measurement uncertainty) is reprocessed. Always searches the sensor's "
                "original, uncut signal for each draw's cut point, so it's safe to tick this even "
                "after already clicking \"Auto-cut using wave travel time\" above or the Custom "
                "Sensors page's own \"Cut curve\" -- no need to \"Reset cut\" first.</li>"
                "</ul>"
            ),
        )
        self.sensor_align_rows = []
        layout.addWidget(sensor_align)
        self._sensor_align_card = sensor_align

        heat = Card(
            "Bar-temperature-gradient parameters",
            "Only used if the correction above is enabled.",
            help_html=(
                "<p>Reference temperature for the bar-temperature-gradient correction, used "
                "only when \"Bar-temperature-gradient correction\" is checked above.</p>"
                "<ul>"
                "<li><b>Output bar gage temperature</b>: the temperature at the output bar's "
                "strain-gage location, used together with the specimen temperature (Test Setup "
                "page) to model the heat conducted from the heated specimen into the bar end.</li>"
                "</ul>"
            ),
        )
        self.gage_ob_edit = QLineEdit(str(project.shpb.gage_temp_ob))
        heat.add(LabeledField("Output bar gage temperature (K)", self.gage_ob_edit))
        layout.addWidget(heat)

        platen = Card(
            "Platens",
            "Only used if the platen correction above is enabled.",
            help_html=(
                "<p>Platen properties, used only when \"Platens\" is checked above.</p>"
                "<ul>"
                "<li><b>Platen config file</b>: loads a previously-saved set of platen "
                "parameters instead of typing them by hand.</li>"
                "<li>The table below holds the platen's mechanical properties (as Monte-Carlo "
                "parameters, with a mean and standard deviation) used in the wave transmission "
                "through the specimen-platen-bar stack.</li>"
                "</ul>"
            ),
        )
        self.platen_file_row = FileSelectRow("Platen config file")
        self.platen_file_row.file_selected.connect(self._on_platen_file)
        platen.add(self.platen_file_row)
        self.platen_status = QLabel("")
        self.platen_status.setObjectName("cardSubtitle")
        platen.add(self.platen_status)
        self.platen_table = MCParamTable([self.project.shpb.platen.params[k] for k in PLATEN_PARAM_ORDER])
        platen.add(self.platen_table)
        layout.addWidget(platen)

        notch = Card(
            "Notched specimen k-coefficients",
            "Only used when the test mode (Test Setup page) is Shear-Compression/Tension (Notched).",
            help_html=_NOTCH_HELP_HTML,
        )
        self.notch_file_row = FileSelectRow("Load k/sk-coefficients from file")
        self.notch_file_row.file_selected.connect(self._on_load_k_value_file)
        notch.add(self.notch_file_row)
        self.notch_file_status = QLabel("")
        self.notch_file_status.setObjectName("cardSubtitle")
        notch.add(self.notch_file_status)
        self.k_edits = [QLineEdit(str(v)) for v in project.shpb.k_value]
        for i, e in enumerate(self.k_edits):
            notch.add(LabeledField(f"k_value[{i}]", e))
        self.sk_check = QCheckBox("Split fit at a displacement threshold (sk)")
        notch.add(self.sk_check)
        self.sk_edits = [QLineEdit(str(v)) for v in project.shpb.sk_value]
        for i, e in enumerate(self.sk_edits):
            notch.add(LabeledField(f"sk_value[{i}]", e))
        self.notch_card = notch
        layout.addWidget(notch)

        # Keep project.shpb live-synced with every settings widget above --
        # otherwise nothing writes these back until "Process" is clicked,
        # so simply navigating away and back (which re-syncs the widgets
        # FROM project.shpb) would silently reset anything typed/toggled.
        self.heat_check.toggled.connect(self._on_settings_changed)
        self.plat_check.toggled.connect(self._on_settings_changed)
        self.punch_check.toggled.connect(self._on_settings_changed)
        self.inertia_check.toggled.connect(self._on_settings_changed)
        self.friction_edit.editingFinished.connect(self._on_settings_changed)
        self.gage_ob_edit.editingFinished.connect(self._on_settings_changed)
        for e in self.k_edits:
            e.editingFinished.connect(self._on_settings_changed)
        self.sk_check.toggled.connect(self._on_settings_changed)
        for e in self.sk_edits:
            e.editingFinished.connect(self._on_settings_changed)

        run_card = Card("Run")
        run_row = QHBoxLayout()
        self.run_button = QPushButton("Process all Monte-Carlo draws")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._on_run)
        run_row.addWidget(self.run_button)
        run_row.addStretch(1)
        run_card.add_layout(run_row)
        self.progress = QProgressBar()
        run_card.add(self.progress)
        self.status_label = QLabel("")
        self.status_label.setObjectName("cardSubtitle")
        run_card.add(self.status_label)
        layout.addWidget(run_card)

        result_card = Card("Result", "Mean stress-strain curve across all Monte-Carlo draws, with the "
                                      "±1 standard-deviation band shown as the uncertainty envelope.")
        self.canvas = PlotCanvas(
            figsize=(8, 5),
            help_title="Direct Impact result",
            help_text=(
                "<p>Shows the mean true stress-strain curve computed by \"Process all "
                "Monte-Carlo draws\" from the Direct-Impact stress-strain conversion, together "
                "with its &plusmn;1 standard-deviation uncertainty band (from the Monte-Carlo "
                "spread across bar/specimen parameter draws).</p>"
                "<p>Use the toolbar to zoom/pan and check the curve's shape and any enabled "
                "corrections' effect -- the Toe Correction section below can fix a slow initial "
                "rise if the curve doesn't reach its real elastic slope cleanly.</p>"
            ),
        )
        result_card.add(self.canvas)
        layout.addWidget(result_card)
        toe_card = Card(
            "Toe (Foot) Correction",
            "If the curve above shows a slow, low-stiffness initial rise before reaching its real "
            "elastic slope (specimen/fixture seating, not a wave-timing issue), fit a tangent to the "
            "steepest, most stable part of the curve and re-zero the strain axis to where that "
            "tangent crosses zero stress -- automatically, or on a region you pick yourself. Applying "
            "this updates the result directly, so every view on Results & Export reflects it too.",
            help_html=(
                "<p>Corrects a slow, low-stiffness initial rise in the result curve above "
                "(specimen/fixture seating, not a wave-timing issue) by fitting a tangent to the "
                "steepest, most stable part of the curve and re-zeroing the strain axis to where "
                "that tangent crosses zero stress.</p>"
                "<ul>"
                "<li><b>Auto-detect linear region</b>: picks the steepest, most stable part of "
                "the curve automatically.</li>"
                "<li><b>Use selected region</b>: fits the tangent on the \"Linear region "
                "(manual)\" window you drag on the plot instead.</li>"
                "<li><b>Apply toe correction</b>: commits the fitted tangent's zero-stress "
                "crossing as the new strain origin.</li>"
                "<li><b>Reset (undo correction)</b>: restores the uncorrected curve.</li>"
                "</ul>"
                "<p><b>Technical detail:</b> applying this updates the result directly, so every "
                "view on Results &amp; Export reflects it too, not just this page's plot.</p>"
            ),
        )
        self.toe_picker = TimeWindowPicker(
            [("region", "Linear region (manual)", theme.ACCENT)], figsize=(8, 4),
            help_title="Toe (Foot) Correction",
            help_text=(
                "<p>Shows the result curve so you can fit a tangent to its real elastic "
                "slope.</p>"
                "<ul>"
                "<li><b>Auto-detect linear region</b> picks the steepest, most stable part of "
                "the curve automatically.</li>"
                "<li>Alternatively, click and drag on the plot to select a \"Linear region "
                "(manual)\" window yourself, then click <b>Use selected region</b> below.</li>"
                "</ul>"
                "<p>Either way, the fitted tangent's zero-stress crossing becomes the new strain "
                "origin once <b>Apply toe correction</b> is clicked -- <b>Reset (undo "
                "correction)</b> restores the uncorrected curve.</p>"
            ),
        )
        toe_card.add(self.toe_picker)
        toe_row = QHBoxLayout()
        self.toe_auto_button = QPushButton("Auto-detect linear region")
        self.toe_auto_button.clicked.connect(self._on_toe_auto_detect)
        toe_row.addWidget(self.toe_auto_button)
        self.toe_manual_button = QPushButton("Use selected region")
        self.toe_manual_button.clicked.connect(self._on_toe_manual_fit)
        toe_row.addWidget(self.toe_manual_button)
        toe_row.addStretch(1)
        toe_card.add_layout(toe_row)
        apply_toe_row = QHBoxLayout()
        self.toe_apply_button = QPushButton("Apply toe correction")
        self.toe_apply_button.setObjectName("primary")
        self.toe_apply_button.clicked.connect(self._on_toe_apply)
        self.toe_apply_button.setEnabled(False)
        apply_toe_row.addWidget(self.toe_apply_button)
        self.toe_reset_button = QPushButton("Reset (undo correction)")
        self.toe_reset_button.clicked.connect(self._on_toe_reset)
        apply_toe_row.addWidget(self.toe_reset_button)
        apply_toe_row.addStretch(1)
        toe_card.add_layout(apply_toe_row)
        self.toe_status_label = QLabel("")
        self.toe_status_label.setObjectName("cardSubtitle")
        self.toe_status_label.setWordWrap(True)
        toe_card.add(self.toe_status_label)
        layout.addWidget(toe_card)
        self._toe_fit = None
        # See the identical block in _SHPBTab.__init__ for the full
        # rationale -- kept per-tab since SHPB/DI have independent
        # Monte-Carlo runs and independent toe corrections.
        self._last_per_draw = None
        self._toe_per_draw_backup = None



        layout.addStretch(1)

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)

        self._restore_from_session()

    def _restore_from_session(self):
        """Re-load any files remembered from a previous session (see
        core/session.py) -- these live on this page, not on `Project`, so
        each file is just re-read through the same handler used when
        picking it live."""
        from pathlib import Path
        for key, handler in (
            ("di_time", self._on_time_file), ("di_velocity", self._on_velocity_file),
        ):
            path = self.project.last_paths.get(key)
            if path and Path(path).exists():
                handler(path)

        dt = self.project.time_step_overrides.get("di_time")
        if dt is not None:
            self.time_file_row.set_dt(dt)
            self.time_file_row.set_generate_mode(True)  # regenerates now if the Output Bar signal is already loaded, or on the next showEvent otherwise

        # Re-apply a velocity-profile cut/shift that was applied before the
        # app closed, now that the raw (uncut) velocity file is reloaded.
        cut_time = self.project.di.velocity_cut_time
        if cut_time is not None and hasattr(self, "_velocity_c") and self._time_pulse is not None:
            self._cumulative_velocity_idx = 0
            self._apply_velocity_cut(cut_time)
            self.velocity_cut_status_label.setText(f"Restored velocity cut from previous session (t = {cut_time:.6g} s).")

    def _on_time_file(self, path):
        try:
            values = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Kept at full float precision, not rounded to 6
                        # decimals: that's only microsecond precision, and a
                        # bar-gage DAQ commonly samples faster than 1 MHz
                        # (dt < 1e-6 s), which would collapse consecutive
                        # timestamps to the same rounded value -- see
                        # _maybe_regenerate_time's docstring below for how
                        # that breaks a generated time vector.
                        values.append(float(line))
            if self.project.downsample_enabled:
                values = [float(v) for v in downsample(values, self.project.downsample_n)]
            self._time_pulse = values
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Could not read time file: {exc}")
            return
        self.project.last_paths["di_time"] = path
        self.status_label.setText(f"Loaded {len(self._time_pulse)} time samples from {path.split('/')[-1]}")
        self._refresh_picker()

    def _on_time_step_changed(self, dt):
        """Alternative to `_on_time_file` above: build the time vector from
        a fixed time step matching the Output Bar signal length, instead of
        loading it from a file. That signal is loaded on the separate
        Output Bar page, not here, so (like the SHPB tab) `_maybe_regenerate_time`
        is also called from `showEvent` so a stale generated vector self-
        corrects as soon as this tab is revisited."""
        self._maybe_regenerate_time()
        if self._time_pulse is None:
            self.status_label.setText(
                "Load an Output Bar signal first (its own page), so the generated time vector can "
                "match its length."
            )
            return
        self.project.last_paths.pop("di_time", None)
        self.project.time_step_overrides["di_time"] = dt
        self.status_label.setText(f"Generated {len(self._time_pulse)} time samples (dt = {dt:.6g} s).")
        self._refresh_picker()

    def _on_time_mode_changed(self, generate):
        if not generate:
            self.project.time_step_overrides.pop("di_time", None)

    def _maybe_regenerate_time(self):
        if not self.time_file_row.is_generate_mode():
            return
        n = len(self.project.output_bar.signal)
        if n == 0:
            return
        generated = self.time_file_row.build(n)
        if generated is not None:
            # NOT rounded to 6 decimals -- see _on_time_file's comment above.
            # A small time step (e.g. 2e-7 s, a common 5 MHz bar-gage sample
            # rate) rounded to microsecond precision collapsed several
            # consecutive generated samples to the exact same timestamp,
            # which is what used to make `_robust_dt` (core/process_shpb.py)
            # raise "too many samples ... share the same timestamp" for a
            # perfectly valid, uniformly-spaced generated time vector.
            self._time_pulse = [float(v) for v in generated]

    def _refresh_picker(self):
        if self._time_pulse is None:
            self.picker.show_placeholder("Load a time file above to see the signal here.")
        elif len(self.project.output_bar.signal) == 0:
            self.picker.show_placeholder(
                "No Output Bar signal loaded yet.\nGo to the Output Bar page, load its signal file, then come back here."
            )
        else:
            secondary = None
            if hasattr(self, "_velocity_c") and len(self._velocity_c) > 0:
                secondary = [("Incident velocity (laser gate)", self._velocity_c, "#ff9f43")]
            self.picker.set_signal(
                self._time_pulse, [("Output Bar signal (gage)", self.project.output_bar.signal, theme.ACCENT)],
                secondary_series=secondary,
                primary_ylabel="Bar gage signal", secondary_ylabel="Velocity (m/s)",
            )

    def _on_window_changed(self, key, start, end):
        pass  # read directly from the picker in _on_run; nothing else needs live-syncing here

    def _on_use_corrected_toggled(self, checked):
        self.project.shpb.use_corrected_signal = checked
        self.time_file_row.setVisible(not checked)
        self.picker.setVisible(not checked)
        self._refresh_corrected_status()

    def _refresh_corrected_status(self):
        if not self.use_corrected_check.isChecked():
            self.corrected_status_label.setText("")
            return
        corr = self.project.last_correction
        if corr is None:
            self.corrected_status_label.setText(
                "No corrected signal available yet -- run the Dispersion Corrections page first "
                "(Single output bar / Direct Impact setup)."
            )
        elif not corr["direct_impact"]:
            self.corrected_status_label.setText(
                "The last Corrections run was for SHPB (two bars), not Direct Impact -- re-run it "
                "with \"Single output bar (Direct Impact)\" selected."
            )
        else:
            mode_label = {0: "no correction", 1: "elastic", 2: "viscoelastic"}[corr["mode_index"]]
            self.corrected_status_label.setText(
                f"Using the corrected signal from Corrections ({mode_label}), window "
                f"{corr['xd1']:.6g} s -> {corr['xd2']:.6g} s."
            )

    def showEvent(self, event):
        super().showEvent(event)
        self._maybe_regenerate_time()  # Output Bar signal may have (re)loaded since this tab was last shown
        self._refresh_picker()
        self._refresh_corrected_status()
        self.notch_card.setVisible(self.project.test_mode == "Shear-Compression/Tension (Notched)")
        self.platen_table.refresh()  # picks up values changed elsewhere (e.g. a loaded Setup Profile)
        self._refresh_sensor_align_card()
        self._refresh_settings_from_project()
        self._refresh_test_mode_label()

    def _refresh_test_mode_label(self):
        # See _SHPBTab._refresh_test_mode_label -- same shared, project-wide
        # "Test mode" (Test Setup page), same footgun if it's left over from a
        # different test processed on another Processing tab.
        self.test_mode_label.setText(
            f"Test mode (from Test Setup page): \"{self.project.test_mode}\". If this Direct Impact "
            "test isn't that kind of test, change the mode on the Test Setup page first."
        )

    def _on_load_k_value_file(self, path):
        """Loads the notched-specimen k/sk fit coefficients from a plain
        text file (see core/k_value_io.py for the exact expected format --
        6 k-values, a 'Split = True/False' line, 7 sk-values), instead of
        having to re-type each of the 13 values by hand from a printed/
        exported copy. Writes straight into the k_value[]/sk_check/
        sk_value[] fields already on this page and persists them via
        `_on_settings_changed()`, exactly as if they'd been typed in.
        These coefficients live on `project.shpb` regardless of tab (SHPB
        or Direct Impact share the same notched-specimen fit), so loading
        them here also updates the SHPB tab next time it's shown."""
        from core.k_value_io import load_k_value_file
        try:
            k_value, sk_check, sk_value = load_k_value_file(path)
        except (OSError, ValueError) as exc:
            self.notch_file_status.setText(f"Could not load k/sk-coefficients: {exc}")
            return
        for edit, value in zip(self.k_edits, k_value):
            edit.setText(str(value))
        self.sk_check.setChecked(sk_check)
        for edit, value in zip(self.sk_edits, sk_value):
            edit.setText(str(value))
        self._on_settings_changed()
        self.notch_file_status.setText(f"Loaded from {path.split('/')[-1]}.")

    def _on_settings_changed(self, *_args):
        """Keeps `project.shpb` live-synced with this tab's widgets --
        without this, nothing persists until "Process" is clicked, and
        simply navigating away and back would reset everything (since
        `_refresh_settings_from_project` below reads back from
        `project.shpb`, which would still hold stale/default values)."""
        s = self.project.shpb
        s.heat_check = self.heat_check.isChecked()
        s.plat_check = self.plat_check.isChecked()
        s.punch_check = self.punch_check.isChecked()
        s.inertia_check = self.inertia_check.isChecked()
        try:
            s.friction_coefficient = float(self.friction_edit.text())
        except ValueError:
            pass
        try:
            s.gage_temp_ob = float(self.gage_ob_edit.text())
        except ValueError:
            pass
        for i, e in enumerate(self.k_edits):
            try:
                s.k_value[i] = float(e.text())
            except ValueError:
                pass
        s.sk_check = self.sk_check.isChecked()
        for i, e in enumerate(self.sk_edits):
            try:
                s.sk_value[i] = float(e.text())
            except ValueError:
                pass

    def _refresh_settings_from_project(self):
        """Syncs this tab's checkboxes/fields with `project.shpb`, in case
        they were changed elsewhere (e.g. a loaded Setup Profile).

        Signals are blocked while doing this: `setChecked()`/`setText()`
        below fire `toggled`/`editingFinished` whenever they actually
        change a widget's value, which would otherwise re-enter
        `_on_settings_changed()` mid-refresh -- at that point some of
        THIS tab's own widgets have already been updated to match
        `project.shpb` and others haven't yet, so that reentrant call
        would write a half-updated, inconsistent mix back into
        `project.shpb`, silently clobbering whatever was just restored."""
        s = self.project.shpb
        widgets = [
            self.heat_check, self.plat_check, self.punch_check, self.inertia_check,
            self.friction_edit, self.gage_ob_edit,
            *self.k_edits, self.sk_check, *self.sk_edits, self.use_corrected_check,
        ]
        for w in widgets:
            w.blockSignals(True)
        try:
            self.heat_check.setChecked(s.heat_check)
            self.plat_check.setChecked(s.plat_check)
            self.punch_check.setChecked(s.punch_check)
            self.inertia_check.setChecked(s.inertia_check)
            self.friction_edit.setText(str(s.friction_coefficient))
            self.gage_ob_edit.setText(str(s.gage_temp_ob))
            for edit, value in zip(self.k_edits, s.k_value):
                edit.setText(str(value))
            self.sk_check.setChecked(s.sk_check)
            for edit, value in zip(self.sk_edits, s.sk_value):
                edit.setText(str(value))
            self.use_corrected_check.setChecked(s.use_corrected_signal)
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._on_use_corrected_toggled(s.use_corrected_signal)  # visibility effect, applied once, explicitly

    def _refresh_sensor_align_card(self):
        while self._sensor_align_card.body.count() > 0:
            item = self._sensor_align_card.body.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.sensor_align_rows = []

        candidates = [(i, s) for i, s in enumerate(self.project.custom_sensors)
                      if s.enabled and s.on_specimen and len(s.time) > 0]
        if not candidates:
            note = QLabel("No enabled, specimen-mounted Custom Sensor yet (see the Custom Sensors page).")
            note.setObjectName("cardSubtitle")
            self._sensor_align_card.add(note)
            return

        for idx, sensor in candidates:
            row = QHBoxLayout()
            row.addWidget(QLabel(sensor.label))
            button = QPushButton("Auto-cut using wave travel time")
            button.setEnabled(not sensor.cut_in_montecarlo)
            button.clicked.connect(lambda _checked=False, i=idx: self._on_align_sensor(i))
            row.addWidget(button)
            mc_check = QCheckBox("Include this cut in the Monte-Carlo loop")
            mc_check.setChecked(sensor.cut_in_montecarlo)
            mc_check.toggled.connect(lambda checked, i=idx: self._on_sensor_mc_toggle(i, checked))
            row.addWidget(mc_check)
            row.addStretch(1)
            self._sensor_align_card.add_layout(row)
            # On its own full-width row (not crammed next to the button in
            # the row above), with word-wrap on -- otherwise this label's
            # text (long, with several numbers) can render as overlapping/
            # garbled digits when squeezed into a narrow horizontal layout.
            status = QLabel("")
            status.setObjectName("cardSubtitle")
            status.setWordWrap(True)
            self._sensor_align_card.add(status)
            self.sensor_align_rows.append((idx, status, mc_check, button))
            if sensor.cut_in_montecarlo:
                status.setText(
                    "This sensor's cut will be recomputed for every Monte-Carlo draw when \"Process\" "
                    "is run below (see the help icon above for details)."
                )

    def _on_sensor_mc_toggle(self, sensor_index: int, checked: bool):
        sensor = self.project.custom_sensors[sensor_index]
        sensor.cut_in_montecarlo = checked
        row = next((r for r in self.sensor_align_rows if r[0] == sensor_index), None)
        if row is None:
            return
        _, status, _, button = row
        button.setEnabled(not checked)
        if checked:
            status.setText(
                "This sensor's cut will be recomputed for every Monte-Carlo draw when \"Process\" is run "
                "below, from that draw's own sampled bar/platen/specimen properties -- it always "
                "searches the sensor's original, uncut signal, so it's fine even if it was already "
                "cut once (this card's button, or the Custom Sensors page's \"Cut curve\")."
            )
        else:
            status.setText("")

    def _on_align_sensor(self, sensor_index: int):
        sensor = self.project.custom_sensors[sensor_index]
        status_label = next((r[1] for r in self.sensor_align_rows if r[0] == sensor_index), None)
        if self.use_corrected_check.isChecked():
            corr = self.project.last_correction
            if corr is None or not corr.get("direct_impact") or corr.get("xd1") is None:
                if status_label:
                    status_label.setText(
                        "\"Use dispersion-corrected signal\" is checked but no valid Direct Impact correction "
                        "is available yet -- run the Dispersion Corrections page first."
                    )
                return
            xd1_output = corr["xd1"]
        else:
            window = self.picker.get_window("xd")
            if window is None:
                if status_label:
                    status_label.setText("Select the transmitted pulse window above first (needed as the time reference).")
                return
            xd1_output = window[0]
        try:
            ob = self.project.output_bar
            sp = self.project.specimen
            c0_sample = sample_wave_speed(sp.params["e"].mean, sp.params["rho"].mean)
            use_platens = self.plat_check.isChecked()
            plat = self.project.shpb.platen
            c0_platens = sample_wave_speed(plat.params["e"].mean, plat.params["rho"].mean) if use_platens else 0.0
            dt = specimen_sensor_delay_direct_impact(
                output_gage_position=ob.params["PG"].mean, c0_output_bar=ob.params["C0"].mean,
                sample_length=sp.params["L"].mean, c0_sample=c0_sample,
                use_platens=use_platens, platens_length=plat.params["L"].mean, c0_platens=c0_platens,
            )
        except (KeyError, ZeroDivisionError, ValueError) as exc:
            if status_label:
                status_label.setText(f"Could not compute the delay: {exc}")
            return
        # In Direct Impact, the output-bar gage sees the wave AFTER it has
        # already travelled from the specimen -- so the specimen started
        # loading `dt.total` BEFORE the gage registered the transmitted
        # wave's start: cut point = xd1_output - dt.total (not dt.total alone).
        cut_point = xd1_output - dt.total
        sensor.cut_from(cut_point)
        if status_label:
            platen_part = f" + platen {dt.platen_term:.4g} s" if use_platens else ""
            status_label.setText(
                f"Cut at t = {cut_point:.6g} s (output wave start {xd1_output:.6g} s - travel time {dt.total:.6g} s = "
                f"bar {dt.bar_term:.4g} s{platen_part} + half-specimen {dt.specimen_term:.4g} s). "
                f"{len(sensor.data)} points remain."
            )

    def _on_velocity_file(self, path):
        try:
            values = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        values.append(float(line))
            velocity = np.asarray(values)
            if self.project.downsample_enabled:
                velocity = downsample(velocity, self.project.downsample_n)
            self._velocity_c = velocity
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Could not read velocity file: {exc}")
            return
        self.project.last_paths["di_velocity"] = path
        self.status_label.setText(f"Loaded {len(self._velocity_c)} velocity samples from {path.split('/')[-1]}")
        self._refresh_picker()

    def _on_cut_velocity(self):
        if not hasattr(self, "_velocity_c") or len(self._velocity_c) == 0:
            self.velocity_cut_status_label.setText("Load the velocity file first.")
            return
        window = self.picker.get_window("velocity_cut")
        if window is None:
            self.velocity_cut_status_label.setText("Drag on the plot (Velocity start cut) to pick a cut point first.")
            return
        self._apply_velocity_cut(window[0])

    def _on_apply_velocity_shift(self):
        if not hasattr(self, "_velocity_c") or len(self._velocity_c) == 0:
            self.velocity_cut_status_label.setText("Load the velocity file first.")
            return
        try:
            shift = float(self.velocity_shift_edit.text())
        except ValueError:
            self.velocity_cut_status_label.setText("The shift value must be numeric (seconds).")
            return
        self._apply_velocity_cut(shift)

    def _apply_velocity_cut(self, cut_time: float):
        """Trims the leading samples of the velocity profile up to
        `cut_time` (on the same time axis as the bar gage signal) and
        re-indexes the remainder to start at t=0 -- realigning the
        velocity profile with the bar signal, the same way `process_di`
        expects (it treats the velocity array's own index 0 as t=0)."""
        if self._time_pulse is None:
            self.velocity_cut_status_label.setText("Load the time file first.")
            return
        time_arr = np.asarray(self._time_pulse)
        below = np.where(time_arr < cut_time)[0]
        idx = int(below[-1]) if len(below) else 0
        self._velocity_c = self._velocity_c[idx:]
        # Track the TOTAL samples removed so far (across possibly several
        # cuts) as an equivalent single cut-time against the ORIGINAL
        # (uncut) time axis, so it can be reapplied in one step after a
        # session restore reloads the raw velocity file fresh.
        total_idx = min(getattr(self, "_cumulative_velocity_idx", 0) + idx, len(time_arr) - 1)
        self._cumulative_velocity_idx = total_idx
        self.project.di.velocity_cut_time = float(time_arr[total_idx])
        self.velocity_cut_status_label.setText(
            f"Velocity profile shifted: discarded the first {idx} samples (up to t={cut_time:.6g} s), "
            f"{len(self._velocity_c)} samples remain."
        )
        self._refresh_picker()

    def _on_reset_velocity_cut(self):
        """Re-loads the velocity file fresh from disk, discarding any
        cut(s)/shift(s) applied."""
        from pathlib import Path
        self._cumulative_velocity_idx = 0
        self.project.di.velocity_cut_time = None
        path = self.project.last_paths.get("di_velocity")
        if path and Path(path).exists():
            self._on_velocity_file(path)
            self.velocity_cut_status_label.setText("Velocity cut reset: reloaded the original file from disk.")
        else:
            self.velocity_cut_status_label.setText("Nothing to reset (no remembered velocity file to reload from).")

    def _on_platen_file(self, path):
        try:
            load_param_file(path, self.project.shpb.platen.params, PLATEN_PARAM_ORDER)
        except (OSError, ValueError, SyntaxError) as exc:
            self.platen_status.setText(f"Could not load platen config: {exc}")
            return
        self.project.last_paths["platen_config"] = path
        self.platen_table.refresh()
        self.platen_status.setText(f"Loaded from {path.split('/')[-1]}")

    def _test_mode_options(self):
        project = self.project
        mode = project.test_mode
        if mode == "Compression":
            return dict(type_check=1, comp_check=1 if project.geometry == "Cylindrical" else 2)
        if mode == "Tension":
            return dict(type_check=2, tens_check=1 if project.geometry == "Axisymmetric" else 2)
        if mode == "Shear-Compression/Tension (Notched)":
            return dict(type_check=3)
        shear_map = {"Hat-shaped": 1, "Double shear": 2, "Simple shear": 3}
        return dict(type_check=4, shear_check=shear_map.get(project.geometry, 1))

    def _on_run(self):
        project = self.project
        if not hasattr(self, "_velocity_c"):
            self.status_label.setText("Load the incident-velocity file first.")
            return
        if len(project.output_bar.signal) == 0:
            self.status_label.setText("Load the Output Bar signal first (Output Bar page).")
            return

        use_corrected = self.use_corrected_check.isChecked()
        corr = project.last_correction if use_corrected else None
        AA = None
        if use_corrected:
            if corr is None:
                self.status_label.setText("No corrected signal available -- run the Dispersion Corrections page first.")
                return
            if not corr["direct_impact"]:
                self.status_label.setText("The stored corrected signal is for SHPB, not Direct Impact. Re-run Corrections with \"Single output bar (Direct Impact)\".")
                return
            if corr["time_pulse"] is None:
                self.status_label.setText("The stored corrected signal has no time base recorded -- re-run Corrections.")
                return
            xd1, xd2 = corr["xd1"], corr["xd2"]
            self._time_pulse = corr["time_pulse"]
            result = corr["result"]
            AA = [result.transmitted_corrected]
        else:
            if self._time_pulse is None:
                self.status_label.setText("Load the time file first.")
                return
            window = self.picker.get_window("xd")
            if window is None:
                self.status_label.setText("Select the pulse window by dragging on the signal plot first.")
                return
            # process_di looks up xd1/xd2 by exact match in the rounded time_pulse
            # list, so snap the (arbitrary, mouse-dragged) window edges to the
            # nearest actual time sample.
            xd1 = min(self._time_pulse, key=lambda t: abs(t - window[0]))
            xd2 = min(self._time_pulse, key=lambda t: abs(t - window[1]))

        try:
            friction_coeff = float(self.friction_edit.text())
            gage_ob = float(self.gage_ob_edit.text())
            f_inc_velocity = float(self.f_inc_velocity_edit.text())
            k_value = [float(e.text()) for e in self.k_edits]
            sk_value = [float(e.text()) for e in self.sk_edits]
        except ValueError:
            self.status_label.setText("Pulse window, friction, temperature, calibration and k-coefficient fields must be numeric.")
            return

        project.expand_montecarlo()
        n = project.n_montecarlo

        mode_opts = self._test_mode_options()
        options = DIOptions(
            heat_check=self.heat_check.isChecked(),
            type_bars_check=project.thermal.bar_type,
            plat_check=self.plat_check.isChecked(),
            punch_check=self.punch_check.isChecked(),
            inertia_check=self.inertia_check.isChecked(),
            sk_check=self.sk_check.isChecked(),
            disp_ela=use_corrected,
            **mode_opts,
        )

        plat_arrays = {}
        if options.plat_check:
            plat_arrays = project.shpb.platen.expand(n)

        friction_arr = np.full(n, friction_coeff)
        sg_unc_d = np.ones(n)
        f_inc_arr = np.full(n, f_inc_velocity)
        t0_arr = np.full(n, project.thermal.initial_temperature_k)

        # Specimen-mounted sensors whose cut is to be recomputed for every
        # Monte-Carlo draw (CustomSensor.cut_in_montecarlo, ticked on the
        # "Specimen-Mounted Sensor Alignment" card above) instead of the
        # single deterministic cut the "Auto-cut" button applies -- see
        # the identical block in _SHPBTab._on_run for the full rationale,
        # including why effective_raw_time()/calibrated_data_raw() are
        # used below instead of the live (possibly already cut_from()'d)
        # time/calibrated_data().
        mc_sensors = [(i, s) for i, s in enumerate(project.custom_sensors)
                      if s.enabled and s.on_specimen and s.cut_in_montecarlo and len(s.effective_raw_time()) > 0]
        sensor_cut_points = {}
        sensor_unc_halfwidth = {}
        sensor_target_time = None
        if mc_sensors:
            for idx, sensor in mc_sensors:
                sensor_cut_points[idx] = specimen_sensor_cut_points_direct_impact(
                    xd1_output=xd1,
                    output_gage_position_pa=project.output_bar.PG_pa, c0_output_bar_pa=project.output_bar.C0_pa,
                    sample_length_pa=project.specimen.L_pa, elastic_modulus_pa=project.specimen.e_pa,
                    density_pa=project.specimen.rho_pa,
                    use_platens=options.plat_check,
                    platens_length_pa=plat_arrays.get("L"), platens_elastic_modulus_pa=plat_arrays.get("e"),
                    platens_density_pa=plat_arrays.get("rho"),
                )
                # Declared measurement uncertainty (on the
                # calibrated_data_raw() scale) -- see the identical block
                # in _SHPBTab._on_run for the full rationale.
                lo, hi = sensor.uncertainty_band_raw()
                sensor_unc_halfwidth[idx] = (np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)) / 2.0
            # The pulse-window time base every draw's own result shares --
            # identical to process_di.py's own `time1 = time_pulse[i_i1:
            # i_i2] - xd1` (see core/process_di.py), computed here too
            # since it depends only on (time_pulse, xd1, xd2), not on any
            # per-draw quantity.
            _i_i1 = _index_of_nearest(self._time_pulse, xd1)
            _i_i2 = _index_of_nearest(self._time_pulse, xd2)
            sensor_target_time = np.asarray(self._time_pulse[_i_i1:_i_i2]) - xd1

        # Confinement (quasi-oedometric cell) quantities, computed per
        # Monte-Carlo draw below instead of post-hoc in Results & Export --
        # see the identical block in _SHPBTab._on_run for the full
        # rationale.
        confinement_prep = _prepare_confinement(project)
        confinement_active = False
        cell_idx = poly_ini = poly_esp = None
        if confinement_prep is not None:
            cell_idx, _cell_sensor, poly_ini, poly_esp = confinement_prep
            confinement_active = cell_idx in {i for i, _ in mc_sensors}
        specimen_initial_length = project.specimen.params["L"].mean

        self.progress.setMaximum(n)
        all_stress, all_strain = [], []
        per_draw = []
        errors = 0
        last_error = ""

        di_context = dict(
            options=options, OB=project.output_bar, specimen=project.specimen,
            velocity_c=self._velocity_c, time_pulse=self._time_pulse, xd1=xd1, xd2=xd2,
            SG_unc_d=sg_unc_d, f_inc_velocity=f_inc_arr, T0_pa=t0_arr, friction=friction_arr,
            k_value=k_value, sk_value=sk_value,
            plat_nu=plat_arrays.get("nu"), plat_rho=plat_arrays.get("rho"),
            plat_emod=plat_arrays.get("e"), plat_Dia=plat_arrays.get("D"), plat_Len=plat_arrays.get("L"),
            Tm_value=(project.thermal.melting_temperature_k, project.thermal.melting_exponent),
            T_gages_value=(None, gage_ob),
            material_table_ob=project.material_table_ob,
            AA=AA,
        )

        def _report_progress(done, total):
            if done % max(1, total // 100) == 0 or done == total:
                self.progress.setValue(done)
                QApplication.processEvents()

        results = run_map(
            _process_di_task, list(range(n)), n_workers=project.n_workers,
            context=di_context, progress_callback=_report_progress,
        )

        for d, res, exc in results:
            if exc is not None:  # noqa: BLE001 - keep the last message so a total failure is diagnosable
                errors += 1
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            if res.stress and res.strain:
                all_stress.append(res.stress)
                all_strain.append(res.strain)
                draw = {
                    "strain": res.strain, "stress": res.stress, "strain_rate": res.strain_rate,
                    "nominal_strain": res.nominal_strain, "nominal_stress": res.nominal_stress,
                    "nominal_strain_rate": res.nominal_strain_rate,
                    "temperature": res.temperature, "displacement": res.displacement,
                    "force_output": res.force_output, "time": res.time,
                    "inertia_stress": res.inertia_stress, "friction_stress": res.friction_stress,
                    "punch_strain": res.punch_strain,
                }
                for idx, sensor in mc_sensors:
                    raw_time = sensor.effective_raw_time()
                    draw[f"sensor_{idx}_signal"] = cut_and_resample(
                        raw_time, sensor.calibrated_data_raw(), sensor_cut_points[idx][d], sensor_target_time
                    )
                    draw[f"sensor_{idx}_signal_unc"] = cut_and_resample(
                        raw_time, sensor_unc_halfwidth[idx], sensor_cut_points[idx][d], sensor_target_time
                    )
                if confinement_active:
                    try:
                        strain_for_conf = draw.get("nominal_strain", draw["strain"])
                        stress_for_conf = draw.get("nominal_stress", draw["stress"])
                        draw.update(per_draw_confinement_quantities(
                            strain_for_conf, stress_for_conf, draw[f"sensor_{cell_idx}_signal"],
                            project.confinement, specimen_initial_length,
                            poly_ini=poly_ini, poly_esp=poly_esp, time=draw.get("time"),
                        ))
                    except (ZeroDivisionError, ValueError, np.linalg.LinAlgError):
                        pass  # this one draw's confinement calc failed -- a rare numeric edge case
                per_draw.append(draw)

        self.progress.setValue(n)
        if not all_stress:
            self.status_label.setText(f"No draws completed successfully ({errors}/{n} failed). Last error: {last_error}")
            return

        min_len = min(len(s) for s in all_stress)
        stress_arr = np.array([s[:min_len] for s in all_stress])
        strain_arr = np.array([s[:min_len] for s in all_strain])
        mean_strain = strain_arr.mean(axis=0)
        mean_stress = stress_arr.mean(axis=0)
        std_stress = stress_arr.std(axis=0)

        confinement_note = ""
        if project.confinement.enabled:
            if confinement_active:
                confinement_note = (
                    " Confinement quantities computed per Monte-Carlo draw (real propagation) -- "
                    "Results & Export will use this real uncertainty instead of its synthetic approximation."
                )
            else:
                confinement_note = (
                    " Confinement enabled, but not computed per draw here -- tick \"Include this cut in the "
                    "Monte-Carlo loop\" for the confinement-cell sensor above (Specimen-Mounted Sensor "
                    "Alignment) to get real per-draw uncertainty in Results & Export instead of its "
                    "synthetic approximation."
                )
        self.status_label.setText(f"Processed {len(all_stress)}/{n} draws ({errors} failed).{confinement_note}")
        quantities = _aggregate_quantities(per_draw)
        self.project.last_results["DI"] = {"label": "Direct Impact", **quantities}
        _store_raw_draws(self.project, "DI", per_draw)
        # A fresh run supersedes any pending toe-correction state from a
        # previous curve -- both the per-draw list a later Apply would
        # re-cut, and any backup kept for Reset.
        self._last_per_draw = per_draw
        self._toe_per_draw_backup = None
        self.canvas.clear()
        ax = self.canvas.figure.add_subplot(111)
        sigma = self.project.uncertainty_sigma
        ax.plot(mean_strain, mean_stress, "b-", label="Mean")
        ax.fill_between(mean_strain, mean_stress - sigma * std_stress, mean_stress + sigma * std_stress,
                         color="b", alpha=0.2, label=f"±{sigma}σ")
        ax.set_xlabel("Strain (-)")
        ax.set_ylabel("Stress (MPa)")
        ax.legend()
        self.canvas.refresh()
        self._toe_fit = None
        self.toe_apply_button.setEnabled(False)
        self._refresh_toe_picker()

    def _toe_strain_stress(self):
        result = self.project.last_results.get("DI")
        if result is None:
            return None
        strain_key = "nominal_strain" if "nominal_strain" in result else "strain"
        stress_key = "nominal_stress" if "nominal_stress" in result else "stress"
        if strain_key not in result or stress_key not in result:
            return None
        return result, strain_key, stress_key, np.asarray(result[strain_key]), np.asarray(result[stress_key])

    def _refresh_toe_picker(self):
        info = self._toe_strain_stress()
        if info is None:
            self.toe_picker.show_placeholder("Run the Monte-Carlo processing above first to see the curve here.")
            return
        _, _, _, strain, stress = info
        self.toe_picker.set_signal(strain, [("Stress vs strain", stress, theme.ACCENT)],
                                    xlabel="strain (-)", primary_ylabel="stress (MPa)")
        self.toe_status_label.setText("")

    def _draw_toe_fit(self, strain, stress, fit):
        self.toe_picker.set_signal(strain, [("Stress vs strain", stress, theme.ACCENT)],
                                    xlabel="strain (-)", primary_ylabel="stress (MPa)")
        ax = self.toe_picker._ax
        if ax is None:
            return
        x_region = strain[fit.start_index:fit.end_index]
        ax.plot(x_region, stress[fit.start_index:fit.end_index], color="orange", linewidth=2, label="Fitted region")
        x_line = np.array([fit.strain_offset, strain.max()])
        y_line = fit.slope * (x_line - fit.strain_offset)
        ax.plot(x_line, y_line, color="lime", linestyle="--", label="Tangent")
        ax.axvline(fit.strain_offset, color="lime", linestyle=":", linewidth=1)
        ax.legend(fontsize=8)
        self.toe_picker.canvas.refresh()

    def _on_toe_auto_detect(self):
        info = self._toe_strain_stress()
        if info is None:
            self.toe_status_label.setText("Run the Monte-Carlo processing above first.")
            return
        _, _, _, strain, stress = info
        try:
            fit = detect_linear_region(strain, stress)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.toe_status_label.setText(f"Could not auto-detect a linear region: {exc}")
            return
        self._toe_fit = fit
        self.toe_apply_button.setEnabled(True)
        self._draw_toe_fit(strain, stress, fit)
        self.toe_status_label.setText(
            f"Auto-detected: slope = {fit.slope:.6g} MPa, R\u00b2 = {fit.r_squared:.4g}, strain offset = "
            f"{fit.strain_offset:.6g}. Check the fit (orange = region used, green = tangent), then apply, "
            "or drag a region yourself and click \"Use selected region\" instead."
        )

    def _on_toe_manual_fit(self):
        info = self._toe_strain_stress()
        if info is None:
            self.toe_status_label.setText("Run the Monte-Carlo processing above first.")
            return
        _, _, _, strain, stress = info
        window = self.toe_picker.get_window("region")
        if window is None:
            self.toe_status_label.setText("Drag on the plot to select the linear region first.")
            return
        start_idx = int(np.searchsorted(strain, window[0]))
        end_idx = int(np.searchsorted(strain, window[1]))
        if end_idx - start_idx < 2:
            self.toe_status_label.setText("Selected region is too narrow -- drag a wider window.")
            return
        try:
            fit = fit_linear_region(strain, stress, start_idx, end_idx)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.toe_status_label.setText(f"Could not fit the selected region: {exc}")
            return
        self._toe_fit = fit
        self.toe_apply_button.setEnabled(True)
        self._draw_toe_fit(strain, stress, fit)
        self.toe_status_label.setText(
            f"Fitted to your selection: slope = {fit.slope:.6g} MPa, R\u00b2 = {fit.r_squared:.4g}, "
            f"strain offset = {fit.strain_offset:.6g}. Check the fit, then apply."
        )

    def _on_toe_apply(self):
        if self._toe_fit is None:
            self.toe_status_label.setText("Auto-detect or manually fit a linear region first.")
            return
        info = self._toe_strain_stress()
        if info is None:
            return
        result, strain_key, stress_key, strain, stress = info
        if not self._last_per_draw:
            self.toe_status_label.setText(
                "No per-draw Monte-Carlo data to re-cut -- re-run the processing above first."
            )
            return

        # See the identical block in _SHPBTab._on_toe_apply for the full
        # rationale (backup semantics, why per-draw, what it fixes).
        if self._toe_per_draw_backup is None:
            self._toe_per_draw_backup = self._last_per_draw

        has_nominal, has_true = "nominal_strain" in result, "strain" in result
        relation = infer_true_strain_relation(result["nominal_strain"], result["strain"]) if (has_nominal and has_true) else None

        strain_offset = self._toe_fit.strain_offset
        new_per_draw, n_removed = _apply_toe_correction_to_draws(
            self._last_per_draw, strain_key, strain_offset, relation
        )
        if not new_per_draw:
            self.toe_status_label.setText(
                "Every Monte-Carlo draw's own curve sits below the fitted toe offset -- nothing to apply. "
                "Try a smaller/different fitted region."
            )
            return

        # See the identical block in _SHPBTab._on_toe_apply for the full
        # rationale of this bookkeeping.
        prior_toe_n_removed = int(result.get("toe_n_removed", 0))
        quantities = _aggregate_quantities(new_per_draw)
        self.project.last_results["DI"] = {
            "label": "Direct Impact", **quantities, **_pre_toe_quantities(self._toe_per_draw_backup),
            "toe_n_removed": prior_toe_n_removed + n_removed,
        }
        _store_raw_draws(self.project, "DI", new_per_draw)
        n_dropped = len(self._last_per_draw) - len(new_per_draw)
        self._last_per_draw = new_per_draw

        self._toe_fit = None
        self.toe_apply_button.setEnabled(False)
        dropped_note = (
            f" ({n_dropped} outlier draw(s) dropped -- never crossed the offset, or left too little "
            "data above it to keep)"
        ) if n_dropped else ""
        self.toe_status_label.setText(
            f"Applied: strain re-zeroed ({relation or 'unknown'} relation) and re-cut for every Monte-Carlo "
            f"draw individually (~{n_removed} toe points removed per draw, median), then re-aggregated"
            f"{dropped_note}. Results & Export (including confinement-cell/laser quantities) and Covariance "
            "Analysis both reflect this cut now. Use \"Reset\" to undo."
        )
        self._refresh_toe_picker()

    def _on_toe_reset(self):
        if self._toe_per_draw_backup is None:
            self.toe_status_label.setText("Nothing to reset.")
            return
        self._last_per_draw = self._toe_per_draw_backup
        quantities = _aggregate_quantities(self._last_per_draw)
        self.project.last_results["DI"] = {"label": "Direct Impact", **quantities}
        _store_raw_draws(self.project, "DI", self._last_per_draw)
        self._toe_per_draw_backup = None
        self._toe_fit = None
        self.toe_apply_button.setEnabled(False)
        self.toe_status_label.setText(
            "Toe correction undone -- restored the original curve for every Monte-Carlo draw."
        )
        self._refresh_toe_picker()


class _PendingTab(QWidget):
    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        scroll, layout = make_scroll_page(title, message)
        layout.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)


def _read_column(path: str):
    values = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Take just the first whitespace-separated token, so a file
                # with e.g. two tab/space-separated columns (or trailing
                # whitespace) doesn't raise ValueError on float(line).
                values.append(float(line.split()[0]))
    return values
