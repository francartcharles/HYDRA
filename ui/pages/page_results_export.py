"""
Results & Export page.

Shows the last Monte-Carlo-averaged result computed on each Processing tab
(Load/Displacement Data / SHPB / Direct Impact), lets you pick which quantities go
on the X and Y axes (strain, stress, strain rate, temperature,
displacement, force -- whichever the selected engine actually produced,
matching the original scripts' own output quantities), and exports the
full set of quantities as a plain-text or CSV column file.

Feeds batch analysis across many already-processed tests too
(analysis_resume.py / resume_condition.py, see core/batch_analysis.py):
the "Export as .csv…"/"Export as .txt…" files below are exactly what the
"Batch Analysis" page reads, one file per test, with a column picker
there for which quantities to combine.
"""
from __future__ import annotations

import csv
import json
import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QHBoxLayout, QFileDialog, QCheckBox, QLineEdit,
)

from ui.widgets import Card, LabeledField, FileSelectRow, make_scroll_page
from ui.plot_canvas import PlotCanvas
from core.confinement import (
    linear_cell_radial_stress, nonlinear_cell_radial_stress, laser_stress, stress_invariants,
    radial_strain, volumetric_strain, equivalent_strain, triaxial_synthetic_draws,
    strain_rate_from_strain,
)

RESULT_KEYS = ["QS", "SHPB", "DI"]

# Friendly display names for the raw quantity keys (matching the original
# scripts' own terminology -- true strain/stress, nominal (engineering)
# strain/stress, strain rate, temperature, displacement, and the
# input/output/transmitted force signals).
QUANTITY_LABELS = {
    "strain": "Strain (true) [-]",
    "stress": "Stress (true) [MPa]",
    "stress_std": "Stress std-dev [MPa]",
    "nominal_strain": "Nominal (engineering) strain [-]",
    "nominal_stress": "Nominal (engineering) stress [MPa]",
    "strain_rate": "Strain rate [1/s]",
    "nominal_strain_rate": "Nominal strain rate [1/s]",
    "dic_strain": "DIC Strain (-)",
    "dic_strain_rate": "DIC Strain Rate (1/s)",
    "strain_gage_strain": "Strain Gage Strain (-)",
    "temperature": "Temperature [K]",
    "displacement": "Displacement [mm]",
    "load": "Load [N]",
    "force_input": "Input-bar force [N]",
    "force_output": "Output-bar force [N]",
    "force_average": "Average force [N]",
    "inertia_stress": "Inertia stress correction [MPa]",
    "friction_stress": "Friction stress correction [MPa]",
    "punch_strain": "Punch strain correction [-]",
    "balance_quality_force": "Dynamic equilibrium: work-based (F_output/F_input, integrated) [-]",
    "balance_quality_waves": "Dynamic equilibrium: wave energy (transmitted/(incident+reflected)) [-]",
    "balance_quality_force_instant": "Dynamic equilibrium: instantaneous (F_output/F_input) [-]",
    "signal": "Custom sensor signal (calibrated)",
    "incident": "Incident wave (raw, cut)",
    "incident_corrected": "Incident wave (dispersion-corrected)",
    "reflected": "Reflected wave (raw, cut)",
    "reflected_corrected": "Reflected wave (dispersion-corrected)",
    "transmitted": "Transmitted wave (raw, cut)",
    "transmitted_corrected": "Transmitted wave (dispersion-corrected)",
    "time": "Time [s]",
}


PRE_TOE_MARKER = "_pre_toe"  # see ui.pages.page_process.PRE_TOE_SUFFIX


def _pre_toe_base_key(key: str) -> str | None:
    """"stress_pre_toe" / "stress_pre_toe_std" -> "stress" / "stress_std";
    None for a key that isn't a pre-toe quantity."""
    if key.endswith(PRE_TOE_MARKER):
        return key[: -len(PRE_TOE_MARKER)]
    if key.endswith(PRE_TOE_MARKER + "_std"):
        return key[: -len(PRE_TOE_MARKER + "_std")] + "_std"
    return None


def equalize_export_columns(arrays: dict) -> tuple[dict, int]:
    """Makes the export columns rectangular. The curves recorded BEFORE the
    toe correction ("<quantity>_pre_toe") are longer than the corrected ones
    (the toe points were cut off the latter) -- truncating everything to the
    shortest column, as exports always did, would silently throw away
    exactly the data these columns exist to keep. So:
      - the corrected columns are truncated to the shortest of their own
        group, exactly as before;
      - each pre-toe column keeps ALL its points (they may differ by one or
        two between quantities, e.g. a differenced strain rate);
    and whichever columns are shorter than the table are padded with NaN.
    With no pre-toe column this is identical to the old "truncate to the
    shortest" behaviour. Returns (columns, n_rows)."""
    main = {k: v for k, v in arrays.items() if _pre_toe_base_key(k) is None}
    pre = {k: v for k, v in arrays.items() if _pre_toe_base_key(k) is not None}
    n_main = min((len(a) for a in main.values()), default=0)
    n_pre = max((len(a) for a in pre.values()), default=0)
    n_rows = max(n_main, n_pre)
    columns = {}
    for k, a in arrays.items():
        keep = n_main if k in main else len(a)
        col = np.full(n_rows, np.nan)
        col[:keep] = np.asarray(a, dtype=float)[:keep]
        columns[k] = col
    return columns, n_rows


class ResultsExportPage(QWidget):
    def __init__(self, project, parent=None, process_page=None):
        super().__init__(parent)
        self.project = project
        self._process_page = process_page  # for a force-sync right before any export (see _sync_before_export)
        self._comparison_curves = []  # list of dicts: {result_label, x_key, y_key, x, y, std}
        self._confinement_data = None  # last computed {strain, axial, radial, pressure, deviator, [_las variants]}


        scroll, layout = make_scroll_page(
            "Results & Export",
            "Review the last processed result from each Processing tab, choose which quantities "
            "to plot, and export them.",
        )

        view_card = Card(
            "View a result",
            help_html=(
                "<p>Picks and displays one processed result, from among the last Monte-Carlo-"
                "averaged outputs computed on each Processing tab (Load/Displacement Data / "
                "SHPB / Direct Impact).</p>"
                "<ul>"
                "<li><b>Result</b>: which processing tab's last run to show.</li>"
                "<li><b>X axis / Y axis</b>: which quantities to plot, among whichever the "
                "selected engine actually produced (strain, stress, strain rate, temperature, "
                "displacement, force, etc.).</li>"
                "<li><b>Compare multiple curves</b>: keeps adding curves to the same plot "
                "instead of replacing it, via \"Add current selection to comparison\" -- useful "
                "for comparing two tests, or the same test's true vs. nominal stress.</li>"
                "</ul>"
                "<p>The \"Export\" section further below saves every available quantity for the "
                "currently selected result as columns.</p>"
            ),
        )
        self.result_combo = QComboBox()
        self.result_combo.currentIndexChanged.connect(self._on_result_selected)
        view_card.add(LabeledField("Result", self.result_combo))

        self.x_combo = QComboBox()
        self.x_combo.currentIndexChanged.connect(self._refresh_plot)
        view_card.add(LabeledField("X axis", self.x_combo))

        self.y_combo = QComboBox()
        self.y_combo.currentIndexChanged.connect(self._refresh_plot)
        view_card.add(LabeledField("Y axis", self.y_combo))

        self.compare_check = QCheckBox("Compare multiple curves (keep adding curves to the same plot)")
        self.compare_check.toggled.connect(self._on_compare_toggled)
        view_card.add(self.compare_check)

        compare_row = QHBoxLayout()
        self.add_curve_button = QPushButton("Add current selection to comparison")
        self.add_curve_button.clicked.connect(self._on_add_curve)
        self.add_curve_button.setVisible(False)
        compare_row.addWidget(self.add_curve_button)
        self.clear_curves_button = QPushButton("Clear comparison")
        self.clear_curves_button.clicked.connect(self._on_clear_curves)
        self.clear_curves_button.setVisible(False)
        compare_row.addWidget(self.clear_curves_button)
        compare_row.addStretch(1)
        view_card.add_layout(compare_row)

        self.info_label = QLabel("")
        self.info_label.setObjectName("cardSubtitle")
        view_card.add(self.info_label)
        self.canvas = PlotCanvas(
            figsize=(8, 4.5),
            help_title="View a result",
            help_text=(
                "<p>Plots the selected \"Result\" (the last Monte-Carlo-averaged output from a "
                "Processing tab) with the chosen X/Y quantities.</p>"
                "<p>Tick \"Compare multiple curves\" and use \"Add current selection to "
                "comparison\" to overlay several results/quantity choices on the same axes -- "
                "e.g. comparing two tests, or the same test's true vs. nominal stress.</p>"
                "<p>Use the toolbar to zoom/pan; the \"Export\" section below saves every "
                "available quantity for the currently selected result as columns.</p>"
            ),
        )
        view_card.add(self.canvas)
        layout.addWidget(view_card)

        self.equilibrium_card = Card(
            "Dynamic Equilibrium (SHPB)",
            "The three standard SHPB equilibrium checks together: instantaneous force ratio "
            "(direct from forces), the force-work-integrated ratio, and the wave-energy ratio. "
            "Values close to 1 across most of the strain range indicate the specimen was in "
            "dynamic stress equilibrium. Only available for SHPB results.",
        )
        self.equilibrium_canvas = PlotCanvas(
            figsize=(8, 4),
            help_title="Dynamic Equilibrium (SHPB)",
            help_text=(
                "<p>Plots the three standard SHPB dynamic-equilibrium checks against strain:</p>"
                "<ul>"
                "<li>the instantaneous force ratio (F_output/F_input, direct)</li>"
                "<li>the work-integrated force ratio</li>"
                "<li>the wave-energy ratio (transmitted/(incident+reflected))</li>"
                "</ul>"
                "<p>Values close to 1 across most of the strain range indicate the specimen was "
                "in dynamic stress equilibrium during the test, which is a precondition for the "
                "standard SHPB stress-strain analysis to be valid; a curve that stays far from 1 "
                "for a large portion of the test suggests the result there should be treated "
                "with caution.</p>"
                "<p>Only available for SHPB results.</p>"
            ),
        )
        self.equilibrium_card.add(self.equilibrium_canvas)
        layout.addWidget(self.equilibrium_card)

        self.corrections_card = Card(
            "Correction Contributions (SHPB / Direct Impact)",
            "How much the punch (indentation), friction and inertia corrections -- SHPB/Direct Impact "
            "tab, when enabled there -- actually subtracted from the raw stress/strain to produce the "
            "result above. Inertia and friction are stress corrections (MPa, left axis); punch is a "
            "strain correction (-, right axis). Shown only for whichever of the three were actually "
            "applied to the currently selected result -- also available as plain columns "
            "(inertia_stress/friction_stress/punch_strain) via the X/Y axis pickers above and in every "
            "export below.",
        )
        self.corrections_canvas = PlotCanvas(
            figsize=(8, 4),
            help_title="Correction Contributions",
            help_text=(
                "<p>Plots how much the punch (indentation), friction and inertia corrections -- "
                "when enabled on the SHPB/Direct Impact Processing tab -- actually subtracted "
                "from the raw stress/strain to produce the currently selected result.</p>"
                "<ul>"
                "<li>Inertia and friction are stress corrections (MPa, left axis).</li>"
                "<li>Punch is a strain correction (-, right axis).</li>"
                "</ul>"
                "<p>Only whichever corrections were actually applied are shown; each is also "
                "available as a plain column (inertia_stress/friction_stress/punch_strain) via "
                "the X/Y axis pickers above and in every export.</p>"
            ),
        )
        self.corrections_card.add(self.corrections_canvas)
        layout.addWidget(self.corrections_card)

        self.confine_card = Card(
            "Confinement / Triaxial Stresses",
            "For quasi-oedometric (confined) tests: radial, hydrostatic-pressure and deviatoric "
            "stress derived from the axial result above plus the confinement cell's strain-gage "
            "signal (assign a Custom Sensor's role on the Custom Sensors page). Enable this test "
            "type on the Test Setup page first.",
            help_html=(
                "<p>For quasi-oedometric (confined) tests only -- derives radial, hydrostatic-"
                "pressure and deviatoric stress from the axial result above plus the confinement "
                "cell's strain-gage signal.</p>"
                "<ul>"
                "<li>Needs a Custom Sensor assigned to the confinement-cell role (Custom Sensors "
                "page) and the confinement test type enabled on Test Setup -- the card stays "
                "hidden otherwise.</li>"
                "<li>Five plots are shown, each with its own help below: the confinement "
                "stresses, the lateral/volumetric strains, their strain rates, the (p, q) stress "
                "path, and pressure vs. volumetric strain.</li>"
                "<li><b>Export confinement result as .csv…</b>: saves all of this card's derived "
                "quantities as columns, separately from the main \"Export\" section below.</li>"
                "</ul>"
            ),
        )
        self.confine_status = QLabel("")
        self.confine_status.setObjectName("cardSubtitle")
        self.confine_card.add(self.confine_status)
        self.confine_canvas = PlotCanvas(
            figsize=(8, 4.5),
            help_title="Confinement / Triaxial Stresses",
            help_text=(
                "<p>For quasi-oedometric (confined) tests: plots the radial, hydrostatic-"
                "pressure and deviatoric stresses derived from the axial result above plus the "
                "confinement cell's strain-gage signal, against axial strain.</p>"
                "<p>Needs a Custom Sensor assigned to the confinement-cell role (Custom Sensors "
                "page) and the confinement test type enabled on Test Setup.</p>"
            ),
        )
        self.confine_card.add(self.confine_canvas)
        strains_label = QLabel("Radial / volumetric / equivalent strain vs. nominal axial strain")
        strains_label.setObjectName("cardTitle")
        self.confine_card.add(strains_label)
        self.confine_strains_canvas = PlotCanvas(
            figsize=(8, 4),
            help_title="Confinement strains",
            help_text=(
                "<p>Plots the radial, volumetric and equivalent (von Mises) strains derived from "
                "the confinement measurement against the nominal axial strain.</p>"
                "<p>Shows how the specimen deforms laterally and volumetrically under confined "
                "loading, not just along the loading axis.</p>"
            ),
        )
        self.confine_card.add(self.confine_strains_canvas)
        rates_label = QLabel("Radial / volumetric / equivalent strain rate vs. nominal axial strain")
        rates_label.setObjectName("cardTitle")
        self.confine_card.add(rates_label)
        self.confine_rates_canvas = PlotCanvas(
            figsize=(8, 4),
            help_title="Confinement strain rates",
            help_text=(
                "<p>Time derivatives of the radial, volumetric and equivalent strains above, "
                "plotted against the nominal axial strain (central finite differences on the "
                "result's time base).</p>"
                "<p>When the confinement cell's cut is part of the Monte-Carlo loop (or for a "
                "Load/Displacement Data test), the uncertainty band comes from rates differentiated "
                "draw by draw; otherwise it is the synthetic approximation, like the strains "
                "above. The volumetric rate is negative while the specimen compacts.</p>"
            ),
        )
        self.confine_card.add(self.confine_rates_canvas)
        pq_label = QLabel("Stress path: pressure (p) vs. deviator (q)")
        pq_label.setObjectName("cardTitle")
        self.confine_card.add(pq_label)
        self.confine_pq_canvas = PlotCanvas(
            figsize=(6, 5.5),
            help_title="Stress path (p-q)",
            help_text=(
                "<p>Plots the confined test's stress path in (hydrostatic pressure p, deviatoric "
                "stress q) space -- the standard way to visualize a triaxial/confined loading "
                "history independent of time.</p>"
                "<p>Useful for comparing against failure envelopes or other tests' stress "
                "paths.</p>"
            ),
        )
        self.confine_card.add(self.confine_pq_canvas)
        pv_label = QLabel("Pressure (p) vs. volumetric strain")
        pv_label.setObjectName("cardTitle")
        self.confine_card.add(pv_label)
        self.confine_pv_canvas = PlotCanvas(
            figsize=(6, 5.5),
            help_title="Pressure vs. volumetric strain",
            help_text=(
                "<p>Plots hydrostatic pressure (p) against volumetric strain -- shows the "
                "specimen's compaction/dilation behavior under confined loading, analogous to a "
                "consolidation curve.</p>"
            ),
        )
        self.confine_card.add(self.confine_pv_canvas)
        confine_export_row = QHBoxLayout()
        self.confine_export_button = QPushButton("Export confinement result as .csv…")
        self.confine_export_button.clicked.connect(self._on_export_confinement)
        confine_export_row.addWidget(self.confine_export_button)
        confine_export_row.addStretch(1)
        self.confine_card.add_layout(confine_export_row)
        layout.addWidget(self.confine_card)
        self.confine_card.setVisible(False)  # shown by _refresh_confinement() only for confinement-cell tests

        self.strain_comparison_card = Card(
            "Multi-Strain Comparison",
            "True stress from the currently selected result, plotted against every strain-"
            "measurement method available for it at once -- the standard processing engine's own "
            "strain, plus DIC Strain and/or Strain Gage Strain when published (Digital Image "
            "Correlation / Custom Sensors pages). Shown only once 2 or more methods are available "
            "together.\n\n"
            "Two caveats: (1) DIC Strain is a first-order, SMALL-STRAIN (linearized) estimate from "
            "the local displacement gradient, closer to a nominal/engineering strain than to this "
            "engine's own LOGARITHMIC true strain -- the two only nearly agree at small deformation "
            "and diverge more as strain grows. (2) DIC/gage strain are interpolated onto this "
            "result's own time samples. There is no automatic camera-to-bar-signal time "
            "synchronization in this app -- the camera and the DAQ are independent clocks, and the "
            "camera's own true frame rate can also differ from what was entered -- so this only "
            "lines up correctly once the Digital Image Correlation page's \"Time scale\" / \"Time offset\" "
            "fields (DIC Strain for Results & Export card) have been set to match a feature visible "
            "in both series. If a source's own recording covers only a short or differently-timed "
            "slice of the full test, its curve here will legitimately cover only a small strain "
            "range too (see the coverage note below the plot) -- not a bug.\n\n"
            "The DIC Strain curve is shaded with its own measurement-uncertainty envelope (see the "
            "Image Correlation page's \"DIC Measurement Uncertainty\" card and its own \"Show "
            "uncertainty envelopes\" / width toggle -- a PREDICTIVE model, not a calibrated one). "
            "Strain Gage Strain has no equivalent model here, so it is never shaded.",
        )
        self.strain_comparison_canvas = PlotCanvas(
            figsize=(8, 4.5),
            help_title="Multi-Strain Comparison",
            help_text=(
                "<p>Plots true stress from the currently selected result against every available "
                "strain-measurement method at once -- the standard engine's own strain, plus DIC "
                "Strain and/or Strain Gage Strain when published (Digital Image Correlation / "
                "Custom Sensors pages) -- so the methods can be compared directly.</p>"
                "<ul>"
                "<li>DIC Strain is a small-strain estimate that only nearly matches the engine's "
                "true (logarithmic) strain at small deformation, and needs its Time scale/offset "
                "set correctly (Digital Image Correlation page) to line up on this plot.</li>"
                "<li>A curve covering only part of the strain range usually just means that "
                "source's recording covered a shorter or differently-timed slice of the test, "
                "not an error.</li>"
                "</ul>"
                "<p><b>Technical detail:</b> the DIC Strain curve is shaded with its own "
                "measurement-uncertainty envelope (a predictive model, not a calibrated one -- "
                "see the Image Correlation page's \"DIC Measurement Uncertainty\" card). Strain "
                "Gage Strain has no equivalent model here, so it is never shaded.</p>"
            ),
        )
        self.strain_comparison_card.add(self.strain_comparison_canvas)
        self.strain_comparison_info_label = QLabel("")
        self.strain_comparison_info_label.setObjectName("cardSubtitle")
        self.strain_comparison_info_label.setWordWrap(True)
        self.strain_comparison_card.add(self.strain_comparison_info_label)
        layout.addWidget(self.strain_comparison_card)
        self.strain_comparison_card.setVisible(False)

        export_card = Card(
            "Export",
            "Export every available quantity for the currently selected result, as columns.",
            help_html=(
                "<p>Exports the currently selected result (\"View a result\" above).</p>"
                "<ul>"
                "<li><b>Export as .txt… / Export as .csv…</b>: every available quantity for the "
                "selected result, as plain-text or CSV columns -- this is exactly the file "
                "format the Batch Analysis page reads (one file per test).</li>"
                "<li><b>Export full test (data + all parameters)…</b>: a complete .json snapshot "
                "of the test (all Monte-Carlo draws, every setting) plus a matching .csv/.txt -- "
                "use this to re-open the test later from the Test Setup page's \"Load Test\" "
                "card.</li>"
                "</ul>"
            ),
        )
        export_row = QHBoxLayout()
        self.export_txt_button = QPushButton("Export as .txt…")
        self.export_txt_button.clicked.connect(lambda: self._on_export("txt"))
        export_row.addWidget(self.export_txt_button)
        self.export_csv_button = QPushButton("Export as .csv…")
        self.export_csv_button.setObjectName("primary")
        self.export_csv_button.clicked.connect(lambda: self._on_export("csv"))
        export_row.addWidget(self.export_csv_button)
        export_row.addStretch(1)
        export_card.add_layout(export_row)
        full_test_row = QHBoxLayout()
        self.export_full_test_button = QPushButton("Export full test (data + all parameters)…")
        self.export_full_test_button.clicked.connect(self._on_export_full_test)
        full_test_row.addWidget(self.export_full_test_button)
        full_test_row.addStretch(1)
        export_card.add_layout(full_test_row)
        self.export_status = QLabel("")
        self.export_status.setObjectName("cardSubtitle")
        export_card.add(self.export_status)
        layout.addWidget(export_card)


        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _sync_before_export(self):
        """Force-syncs the Processing page's live widget state into
        `project` right before export, as a safety net so an export
        always captures whatever is currently on screen/configured, not
        whatever `project` happened to hold if some individual widget's
        change signal didn't fire/propagate for any reason."""
        if self._process_page is not None:
            self._process_page.sync_to_project()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_result_list()

    def _sensor_mc_result(self, sensor_index: int):
        """Finds which of the last QS/SHPB/DI runs (if any) has a Monte-
        Carlo per-draw cut for this sensor -- the "sensor_{i}_signal"
        quantity that _SHPBTab/_DITab._on_run adds to `per_draw` when
        CustomSensor.cut_in_montecarlo is ticked on the Processing page's
        "Specimen-Mounted Sensor Alignment" card (core.custom_sensor.
        cut_and_resample, a fresh cut per draw + resample onto that run's
        own time base). Returns (run_key, result_dict) or (None, None) if
        that sensor's cut hasn't been computed by any run yet (Monte-Carlo
        mode isn't enabled for it, or "Process" hasn't been re-run since
        it was ticked)."""
        key = f"sensor_{sensor_index}_signal"
        for run_key in RESULT_KEYS:
            result = self.project.last_results.get(run_key)
            if result is not None and key in result:
                return run_key, result
        return None, None

    def _sensor_signal_for_result(self, sensor_index: int, result: dict | None):
        """Returns (time, signal, signal_std, used_montecarlo) for a
        custom sensor, matched to a SPECIFIC QS/SHPB/DI `result` dict.
        Prefers that result's own Monte-Carlo per-draw cut (see
        _sensor_mc_result above) when it has one, since it's already cut
        per draw AND resampled onto exactly this result's own time base --
        no index-alignment guesswork needed, unlike the raw fallback
        below, which simply assumes sample-for-sample alignment (see
        _refresh_confinement's own long-standing convention). Falls back
        to the sensor's raw/manually-cut time base + calibrated data when
        Monte-Carlo mode isn't enabled, or this particular `result` wasn't
        computed with it (e.g. from before it was ticked).

        The returned `signal_std` combines two independent things: the
        Monte-Carlo draw-to-draw spread coming from recutting at a
        slightly different point on every draw ("{mc_key}_std", alignment
        jitter only), and the sensor's own declared measurement
        uncertainty (CustomSensor.uncertainty_band(), cut/resampled per
        draw the same way as the signal itself and averaged across draws
        into "{mc_key}_unc" by _SHPBTab/_DITab._on_run -- see the
        identical block there). Combined in quadrature since the two are
        independent sources of uncertainty; the declared band alone would
        otherwise be silently dropped once Monte-Carlo cutting is enabled
        for a sensor, which isn't the point of a "more accurate"
        uncertainty estimate."""
        sensor = self.project.custom_sensors[sensor_index]
        mc_key = f"sensor_{sensor_index}_signal"
        if sensor.cut_in_montecarlo and result is not None and mc_key in result:
            signal = np.asarray(result[mc_key])
            std = np.asarray(result.get(f"{mc_key}_std", np.zeros(len(signal))))
            declared_unc = result.get(f"{mc_key}_unc")
            if declared_unc is not None:
                std = np.sqrt(std ** 2 + np.asarray(declared_unc) ** 2)
            return np.asarray(result.get("time", [])), signal, std, True
        lo, hi = sensor.uncertainty_band()
        return np.asarray(sensor.time), np.asarray(sensor.calibrated_data()), (hi - lo) / 2, False

    def _available_results(self) -> dict:
        """All result sets a user could want to plot/export, gathered fresh
        each time: the Processing tabs' last runs, every enabled Custom
        Sensor with loaded data, and the last Dispersion Corrections run's
        cut incident/reflected/transmitted signals (raw and corrected)."""
        results = {}
        for key in RESULT_KEYS:
            if key in self.project.last_results:
                results[key] = self.project.last_results[key]

        for i, sensor in enumerate(self.project.custom_sensors):
            if sensor.enabled and len(sensor.time) > 0 and len(sensor.data) > 0:
                mc_run_key, mc_result = self._sensor_mc_result(i) if sensor.cut_in_montecarlo else (None, None)
                time, signal, signal_std, used_mc = self._sensor_signal_for_result(i, mc_result)
                if used_mc:
                    label = f"Custom Sensor: {sensor.label} (Monte-Carlo cut, from the last {mc_run_key} run)"
                elif sensor.cut_in_montecarlo:
                    label = (
                        f"Custom Sensor: {sensor.label} (raw -- run \"Process\" on the tab where its "
                        "\"Include this cut in the Monte-Carlo loop\" is ticked to see the aligned/"
                        "averaged cut here instead)"
                    )
                else:
                    label = f"Custom Sensor: {sensor.label}"
                results[f"SENSOR_{i}"] = {
                    "label": label, "time": time, "signal": signal, "signal_std": signal_std,
                }

        corr = self.project.last_correction
        if corr is not None:
            result = corr["result"]
            entry = {"label": "Dispersion Correction (cut signals)", "time": result.time,
                      "transmitted": result.transmitted, "transmitted_corrected": result.transmitted_corrected}
            if result.incident is not None:
                entry.update(incident=result.incident, incident_corrected=result.incident_corrected,
                              reflected=result.reflected, reflected_corrected=result.reflected_corrected)
            results["CORRECTION"] = entry

        self._enrich_with_strain_sources(results)
        return results

    def _enrich_with_strain_sources(self, results: dict) -> None:
        """Adds "dic_strain" / "dic_strain_rate" (from the Image
        Correlation page's published DIC strain -- see
        `Project.dic_strain_result`) and/or "strain_gage_strain" (from
        any enabled Custom Sensor with the "Strain gage glued directly
        on the specimen" role, Custom Sensors page) as extra quantities
        on the QS/SHPB/DI result(s) in `results` -- so the true stress
        from the standard processing engine can be plotted/exported
        against a DIC-measured or gage-measured strain instead of only
        that engine's own strain (see QUANTITY_LABELS, `_quantity_keys`,
        and the Multi-Strain Comparison plot below). When DIC's own
        `strain_uncertainty` (core/dic_uncertainty.py -- a PREDICTIVE
        model, not a calibrated one, see the Image Correlation page's
        "DIC Measurement Uncertainty" card) is available, also adds
        "dic_strain_uncertainty", interpolated the same way, so the
        comparison plot below can shade an envelope around the DIC
        curve specifically -- there is no equivalent model for the
        strain gage here, so it never gets one.

        SIMPLIFYING ASSUMPTION -- also surfaced in that plot: this app
        has no AUTOMATIC camera-to-bar-signal time synchronization, so
        each source's own t=0 is only correctly aligned with this
        result's own t=0 once set up to be: for DIC, via the Image
        Correlation page's manual "Time scale" / "Time offset" fields
        (`Project.dic.time_scale`/`time_offset_s`, already baked into
        `dic["time_s"]` by the time it reaches here); for the strain
        gage, its own -- possibly cut-and-rezeroed -- t=0 is assumed to
        already coincide with this result's own t=0. Values are
        linearly interpolated onto THIS result's own time samples;
        outside a source's own time range, its column is NaN there (no
        extrapolation) rather than a misleading flat/clamped value."""
        dic = self.project.dic_strain_result
        dic_ready = (dic is not None and dic.get("time_s") is not None
                     and len(dic["time_s"]) >= 2 and len(dic["strain"]) >= 2)

        gage_idx, gage_sensor = next(
            ((i, s) for i, s in enumerate(self.project.custom_sensors)
             if s.role == "specimen_strain_gage" and s.enabled and len(s.time) > 0 and len(s.data) > 0),
            (None, None),
        )
        gage_ready = gage_sensor is not None and len(gage_sensor.time) >= 2

        if not dic_ready and not gage_ready:
            return

        if dic_ready:
            dic_t = np.asarray(dic["time_s"], dtype=float)
            order = np.argsort(dic_t)
            dic_t_sorted = dic_t[order]
            dic_strain_sorted = np.asarray(dic["strain"], dtype=float)[order]
            dic_rate_sorted = np.asarray(dic["strain_rate"], dtype=float)[order]
            dic_unc = dic.get("strain_uncertainty")
            dic_unc_ready = dic_unc is not None and len(dic_unc) == len(dic_t)
            if dic_unc_ready:
                dic_unc_sorted = np.asarray(dic_unc, dtype=float)[order]

        if gage_ready:
            # Raw/manually-cut fallback, precomputed once -- used for any
            # result that doesn't have a Monte-Carlo per-draw cut for this
            # sensor (see the per-result lookup in the loop below, and
            # _sensor_signal_for_result's own docstring).
            gage_t = np.asarray(gage_sensor.time, dtype=float)
            gage_strain = np.asarray(gage_sensor.calibrated_data(), dtype=float)
            m = min(len(gage_t), len(gage_strain))
            gage_t, gage_strain = gage_t[:m], gage_strain[:m]
            order = np.argsort(gage_t)
            gage_t_sorted = gage_t[order]
            gage_strain_sorted = gage_strain[order]

        for key in RESULT_KEYS:
            entry = results.get(key)
            if entry is None or "time" not in entry:
                continue
            t = np.asarray(entry["time"], dtype=float)
            if t.size == 0:
                continue
            enriched = dict(entry)  # shallow copy -- never mutate project.last_results in place
            if dic_ready:
                enriched["dic_strain"] = np.interp(t, dic_t_sorted, dic_strain_sorted, left=np.nan, right=np.nan)
                enriched["dic_strain_rate"] = np.interp(t, dic_t_sorted, dic_rate_sorted, left=np.nan, right=np.nan)
                if dic_unc_ready:
                    enriched["dic_strain_uncertainty"] = np.interp(
                        t, dic_t_sorted, dic_unc_sorted, left=np.nan, right=np.nan
                    )
            if gage_ready:
                _, mc_signal, _, used_mc = self._sensor_signal_for_result(gage_idx, entry)
                if used_mc:
                    # Already cut per draw and resampled onto THIS
                    # result's own time samples (core.custom_sensor.
                    # cut_and_resample) -- assign directly rather than
                    # re-interpolating the sensor's raw/manually-cut
                    # signal, which (per this method's SIMPLIFYING
                    # ASSUMPTION note above) isn't actually time-synced
                    # to this result at all.
                    enriched["strain_gage_strain"] = np.asarray(mc_signal, dtype=float)
                else:
                    enriched["strain_gage_strain"] = np.interp(t, gage_t_sorted, gage_strain_sorted, left=np.nan, right=np.nan)
            results[key] = enriched

    def _refresh_result_list(self):
        available = self._available_results()
        current_key = self.result_combo.currentData()
        self.result_combo.blockSignals(True)
        self.result_combo.clear()
        for key, entry in available.items():
            self.result_combo.addItem(entry["label"], userData=key)
        self.result_combo.blockSignals(False)
        keys = list(available.keys())
        if keys:
            idx = keys.index(current_key) if current_key in keys else 0
            self.result_combo.setCurrentIndex(idx)
        self._on_result_selected()

    def _current_result(self):
        key = self.result_combo.currentData()
        if key is None:
            return None
        return self._available_results().get(key)

    def _label_for_key(self, key: str) -> str:
        """QUANTITY_LABELS.get(key, key), except for a Monte-Carlo per-draw
        specimen-sensor cut (Processing page's "Specimen-Mounted Sensor
        Alignment" card, "Include this cut in the Monte-Carlo loop" --
        see CustomSensor.cut_in_montecarlo): its key encodes a sensor
        index ("sensor_{i}_signal") rather than a fixed, nameable physical
        quantity, so it's shown with that sensor's own label instead."""
        if key.startswith("sensor_") and key.endswith("_signal"):
            try:
                idx = int(key[len("sensor_"):-len("_signal")])
                return f"{self.project.custom_sensors[idx].label} (Monte-Carlo cut)"
            except (ValueError, IndexError):
                pass
        base = _pre_toe_base_key(key)
        if base is not None and not base.endswith("_std"):
            return f"{QUANTITY_LABELS.get(base, base)} - before toe correction"
        return QUANTITY_LABELS.get(key, key)

    def _quantity_keys(self, result):
        """Quantities available on this result, in a sensible display order
        (excluding the internal `<key>_std` companions, which aren't
        meant to be plotted on their own).

        `result` also carries a few scalar, non-array bookkeeping fields
        alongside its per-sample quantities -- `label` (a string) and, since
        the toe-correction Monte-Carlo re-cut ("solution 2"), `toe_n_removed`
        (a plain int, see `_on_toe_apply`). Neither is a per-sample curve, so
        both must be excluded here: including one in `arrays` would make
        `np.asarray(...)` produce a 0-d array, and `len()` on that raises
        "TypeError: len() of unsized object" in `_on_export`/
        `_on_export_full_test`. Rather than naming each such field, this
        filters structurally on "is it actually a sized sequence" so any
        future scalar metadata added to `result` doesn't reproduce the bug."""
        preferred = ["strain", "stress", "nominal_strain", "nominal_stress",
                     "strain_rate", "nominal_strain_rate",
                     "dic_strain", "dic_strain_rate", "strain_gage_strain",
                     "temperature",
                     "displacement", "load", "force_input", "force_output",
                     "force_average", "inertia_stress", "friction_stress", "punch_strain",
                     "balance_quality_force", "balance_quality_waves", "balance_quality_force_instant",
                     "signal", "incident", "incident_corrected", "reflected", "reflected_corrected",
                     "transmitted", "transmitted_corrected", "time"]
        available = [
            k for k in result
            if k != "label" and not k.endswith("_std")
            and hasattr(result[k], "__len__") and len(result[k]) > 0
        ]
        keys = [k for k in preferred if k in available]
        keys += [k for k in available if k not in keys]
        return keys

    def _on_result_selected(self, *_args):
        result = self._current_result()
        self.x_combo.blockSignals(True)
        self.y_combo.blockSignals(True)
        self.x_combo.clear()
        self.y_combo.clear()
        if result is not None:
            keys = self._quantity_keys(result)
            for key in keys:
                label = self._label_for_key(key)
                self.x_combo.addItem(label, userData=key)
                self.y_combo.addItem(label, userData=key)
            x_default = next((k for k in ("strain", "time") if k in keys), keys[0])
            y_default = next((k for k in ("stress", "signal", "transmitted_corrected") if k in keys), keys[-1])
            self.x_combo.setCurrentIndex(keys.index(x_default))
            self.y_combo.setCurrentIndex(keys.index(y_default))
        self.x_combo.blockSignals(False)
        self.y_combo.blockSignals(False)
        self._refresh_plot()
        self._refresh_confinement()
        self._refresh_equilibrium()
        self._refresh_corrections()
        self._refresh_strain_comparison()

    def _refresh_equilibrium(self):
        result = self._current_result()
        has_all = result is not None and all(
            k in result for k in ("balance_quality_force_instant", "balance_quality_force", "balance_quality_waves")
        )
        self.equilibrium_card.setVisible(has_all)
        if not has_all:
            return

        self.equilibrium_canvas.clear()
        ax = self.equilibrium_canvas.figure.add_subplot(111)
        strain_key = "nominal_strain" if "nominal_strain" in result else "strain"
        strain = np.asarray(result[strain_key])
        instant = np.asarray(result["balance_quality_force_instant"])
        work = np.asarray(result["balance_quality_force"])
        waves = np.asarray(result["balance_quality_waves"])

        n = min(len(strain), len(instant), len(work), len(waves))
        sigma = self.project.uncertainty_sigma
        ax.axhspan(0.9, 1.1, color="green", alpha=0.08, label="±10% band")
        ax.plot(strain[:n], instant[:n], color="red", label="Direct from forces (instantaneous)")
        ax.plot(strain[:n], work[:n], color="blue", label="Work-based (force x displacement)")
        ax.plot(strain[:n], waves[:n], color="black", label="From waves (energy)")
        for key, color in (("balance_quality_force_instant", "red"), ("balance_quality_force", "blue"), ("balance_quality_waves", "black")):
            std_key = f"{key}_std"
            if std_key in result:
                values = np.asarray(result[key])
                std = np.asarray(result[std_key])
                m = min(n, len(std))
                ax.fill_between(strain[:m], values[:m] - sigma * std[:m], values[:m] + sigma * std[:m],
                                 color=color, alpha=0.15)
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
        ax.set_ylim(0, 2)
        ax.set_xlabel("Strain (-)")
        ax.set_ylabel("Mechanical Equilibrium (-)")
        ax.legend(fontsize=8)
        self.equilibrium_canvas.refresh()

    def _refresh_corrections(self):
        result = self._current_result()
        present = [k for k in ("inertia_stress", "friction_stress", "punch_strain")
                   if result is not None and k in result and len(result[k]) > 0]
        self.corrections_card.setVisible(bool(present))
        if not present:
            return

        self.corrections_canvas.clear()
        ax = self.corrections_canvas.figure.add_subplot(111)
        strain_key = "nominal_strain" if "nominal_strain" in result else "strain"
        strain = np.asarray(result[strain_key])
        sigma = self.project.uncertainty_sigma

        lines, labels = [], []
        for key, color, label in (
            ("inertia_stress", "#ff6b6b", "Inertia correction (MPa)"),
            ("friction_stress", "#5b8cff", "Friction correction (MPa)"),
        ):
            if key not in present:
                continue
            values = np.asarray(result[key])
            n = min(len(strain), len(values))
            line, = ax.plot(strain[:n], values[:n], color=color, label=label)
            lines.append(line); labels.append(label)
            std_key = f"{key}_std"
            if std_key in result:
                std = np.asarray(result[std_key])
                m = min(n, len(std))
                ax.fill_between(strain[:m], values[:m] - sigma * std[:m], values[:m] + sigma * std[:m],
                                 color=color, alpha=0.15)
        ax.set_xlabel("Strain (-)")
        ax.set_ylabel("Stress correction (MPa)")

        if "punch_strain" in present:
            ax2 = ax.twinx()
            values = np.asarray(result["punch_strain"])
            n = min(len(strain), len(values))
            label = "Punch (indentation) correction (-)"
            line, = ax2.plot(strain[:n], values[:n], color="#4cd97b", linestyle="--", label=label)
            lines.append(line); labels.append(label)
            std_key = "punch_strain_std"
            if std_key in result:
                std = np.asarray(result[std_key])
                m = min(n, len(std))
                ax2.fill_between(strain[:m], values[:m] - sigma * std[:m], values[:m] + sigma * std[:m],
                                  color="#4cd97b", alpha=0.15)
            ax2.set_ylabel("Punch (indentation) strain correction (-)")

        ax.legend(lines, labels, fontsize=8)
        self.corrections_canvas.refresh()

    def _refresh_strain_comparison(self):
        result = self._current_result()
        self.strain_comparison_card.setVisible(False)
        if result is None:
            return
        stress_key = "stress" if "stress" in result else ("nominal_stress" if "nominal_stress" in result else None)
        default_strain_key = "strain" if "strain" in result else ("nominal_strain" if "nominal_strain" in result else None)
        if stress_key is None or default_strain_key is None:
            return  # only meaningful for a standard QS/SHPB/DI-shaped result

        sources = [(default_strain_key, f"Default ({QUANTITY_LABELS.get(default_strain_key, default_strain_key)})")]
        if "dic_strain" in result and np.any(np.isfinite(np.asarray(result["dic_strain"]))):
            sources.append(("dic_strain", "DIC Strain"))
        if "strain_gage_strain" in result and np.any(np.isfinite(np.asarray(result["strain_gage_strain"]))):
            sources.append(("strain_gage_strain", "Strain Gage Strain"))

        if len(sources) < 2:
            return  # need at least 2 simultaneously-available strain-measurement methods to compare

        self.strain_comparison_card.setVisible(True)
        stress = np.asarray(result[stress_key], dtype=float)
        colors = ["black", "#5b8cff", "#ff9f43"]
        self.strain_comparison_canvas.clear()
        ax = self.strain_comparison_canvas.figure.add_subplot(111)
        coverage_notes = []
        result_time = np.asarray(result.get("time", []), dtype=float)
        result_duration = (result_time.max() - result_time.min()) if result_time.size >= 2 else None
        for i, (strain_key, label) in enumerate(sources):
            strain = np.asarray(result[strain_key], dtype=float)
            n = min(len(strain), len(stress))
            finite = np.isfinite(strain[:n]) & np.isfinite(stress[:n])
            if not np.any(finite):
                continue
            ax.plot(strain[:n][finite], stress[:n][finite], "-", color=colors[i % len(colors)], label=label)
            # DIC Strain specifically carries a (predictive, see the Image
            # Correlation page's "DIC Measurement Uncertainty" card)
            # per-point uncertainty -- shaded here as a horizontal band
            # (strain, the x-axis here, is what that uncertainty is IN;
            # the stress axis has no comparable model of its own).
            if (strain_key == "dic_strain" and "dic_strain_uncertainty" in result
                    and self.project.dic.show_uncertainty):
                sigma = np.asarray(result["dic_strain_uncertainty"], dtype=float)
                ns = min(len(sigma), n)
                finite_u = finite[:ns] & np.isfinite(sigma[:ns])
                if np.any(finite_u):
                    ax.fill_betweenx(
                        stress[:ns][finite_u],
                        strain[:ns][finite_u] - sigma[:ns][finite_u],
                        strain[:ns][finite_u] + sigma[:ns][finite_u],
                        color=colors[i % len(colors)], alpha=0.15, linewidth=0,
                    )
            # For an interpolated (DIC/gage) source, report how much of
            # this result's own time range actually had a source value to
            # interpolate at, and the strain range it reached over that
            # covered span -- a source that covered only a short or
            # differently-timed slice of the full test will legitimately
            # show a small strain range here, which otherwise looks just
            # like a units/scale bug at a glance (see the card's caveats).
            if strain_key != default_strain_key and result_duration:
                covered_frac = 100.0 * np.count_nonzero(finite) / n if n else 0.0
                strain_span = strain[:n][finite]
                if strain_span.size:
                    note = (
                        f"{label}: covers {covered_frac:.0f}% of this result's time range, "
                        f"reaching strain {strain_span.min():.4g} to {strain_span.max():.4g} over that span"
                    )
                    # DIC Strain specifically: also show the FULL range it
                    # reaches over its OWN entire recording (Image
                    # Correlation page), not just the (possibly much
                    # smaller) slice that overlaps this result's own time
                    # range -- the gap between the two numbers is exactly
                    # how much of DIC's own measured strain this
                    # comparison is -- and isn't -- actually showing.
                    if strain_key == "dic_strain":
                        dic_full = self.project.dic_strain_result
                        if dic_full is not None:
                            full_strain = np.asarray(dic_full["strain"], dtype=float)
                            full_finite = full_strain[np.isfinite(full_strain)]
                            if full_finite.size:
                                note += (
                                    f" (its own full recording reaches {full_finite.min():.4g} to "
                                    f"{full_finite.max():.4g})"
                                )
                    coverage_notes.append(note)
                else:
                    coverage_notes.append(f"{label}: no overlap with this result's time range")
        ax.set_xlabel("Strain (-)")
        ax.set_ylabel(QUANTITY_LABELS.get(stress_key, stress_key))
        ax.legend(fontsize=8)
        self.strain_comparison_canvas.refresh()
        self.strain_comparison_info_label.setText("  |  ".join(coverage_notes))

    def _refresh_confinement(self):
        self._confinement_data = None
        self.confine_canvas.clear()
        ax = self.confine_canvas.figure.add_subplot(111)
        self.confine_strains_canvas.clear()
        self.confine_strains_canvas.refresh()
        self.confine_rates_canvas.clear()
        self.confine_rates_canvas.refresh()
        self.confine_pq_canvas.clear()
        self.confine_pq_canvas.refresh()  # cleared up front; redrawn at the end only on a full success
        self.confine_pv_canvas.clear()
        self.confine_pv_canvas.refresh()

        if not self.project.confinement.enabled:
            # Hide the whole card (not just blank its plots) when this isn't a
            # confinement-cell test -- otherwise an empty "Confinement / Triaxial
            # Stresses" card with cleared axes sits on every Results & Export page
            # regardless of test type, which reads as broken/leftover rather than
            # "not applicable here".
            self.confine_card.setVisible(False)
            return

        self.confine_card.setVisible(True)
        result = self._current_result()
        if result is None:
            self.confine_status.setText("No result selected.")
            ax.set_xticks([]); ax.set_yticks([])
            self.confine_canvas.refresh()
            return
        if "strain" not in result and "nominal_strain" not in result:
            self.confine_status.setText("Not applicable to this result (choose a Load/Displacement Data/SHPB/Direct Impact result above).")
            ax.set_xticks([]); ax.set_yticks([])
            self.confine_canvas.refresh()
            return

        cell_idx, cell_sensor = next(
            ((i, s) for i, s in enumerate(self.project.custom_sensors)
             if s.role == "confinement_cell" and s.enabled and len(s.data) > 0),
            (None, None),
        )
        if cell_sensor is None:
            self.confine_status.setText(
                "No enabled Custom Sensor has the \"Confinement cell strain gage\" role yet (Custom Sensors page)."
            )
            ax.set_xticks([]); ax.set_yticks([])
            self.confine_canvas.refresh()
            return

        strain_key = "nominal_strain" if "nominal_strain" in result else "strain"
        stress_key = "nominal_stress" if "nominal_stress" in result else "stress"
        strain = np.asarray(result[strain_key])
        axial = np.asarray(result[stress_key])
        axial_std = np.asarray(result.get(f"{stress_key}_std", []))
        # Prefers the cell sensor's Monte-Carlo per-draw cut when it was
        # computed for THIS result (already cut per draw and resampled
        # onto this result's own time base, so it lines up with `strain`/
        # `axial` sample-for-sample) -- otherwise falls back to the raw/
        # manually-cut signal, simply assumed (as before) to already
        # share the same sample grid. See _sensor_signal_for_result.
        _, cell_defl, cell_defl_std, cell_used_mc = self._sensor_signal_for_result(cell_idx, result)
        # The raw/manually-cut fallback above was never itself touched by a
        # toe (foot) correction on the Processing tab -- if one was applied
        # to this SAME `strain`/`axial` curve (ProcessPage._on_toe_apply),
        # its leading samples were already removed and re-zeroed there
        # (both for the aggregate curve and, via _apply_toe_correction_to_
        # draws, for every Monte-Carlo draw), and `toe_n_removed` records
        # how many. The raw sensor signal must have that SAME leading count
        # dropped here too, or it silently pairs the wrong (pre-toe)
        # samples with the now-shorter, re-zeroed strain/axial -- the exact
        # "decalage" this bookkeeping exists to prevent. Only needed for
        # the raw fallback: the Monte-Carlo-cut path above already went
        # through that same per-draw re-cut as strain/axial, so it needs no
        # further adjustment.
        toe_n_removed = int(result.get("toe_n_removed", 0))
        if not cell_used_mc and toe_n_removed:
            cut = min(toe_n_removed, len(cell_defl))
            cell_defl = cell_defl[cut:]
            cell_defl_std = np.asarray(cell_defl_std)
            cell_defl_std = cell_defl_std[cut:] if len(cell_defl_std) >= cut else np.asarray([])
        conf = self.project.confinement

        try:
            if conf.nonlinear_enabled:
                if not (conf.calib_ortho_disp_ini and conf.calib_radial_ini and conf.calib_ortho_disp_esp and conf.calib_radial_esp):
                    self.confine_status.setText("Load all 4 nonlinear cell calibration curves first (Special Tests page).")
                    ax.set_xticks([]); ax.set_yticks([])
                    self.confine_canvas.refresh()
                    return
                specimen_length = self.project.specimen.params["L"].mean
                radial = nonlinear_cell_radial_stress(
                    strain, cell_defl, specimen_length, conf.ref_strain, conf.cell_diameter,
                    conf.calib_ortho_disp_ini, conf.calib_radial_ini, conf.calib_ortho_disp_esp, conf.calib_radial_esp,
                )
            else:
                radial = linear_cell_radial_stress(strain, cell_defl, conf.cell_modulus, conf.C1, conf.C2, conf.ref_strain)
        except (ZeroDivisionError, ValueError, np.linalg.LinAlgError) as exc:
            self.confine_status.setText(f"Could not compute radial stress: {exc}")
            ax.set_xticks([]); ax.set_yticks([])
            self.confine_canvas.refresh()
            return

        n = min(len(strain), len(axial), len(radial))
        strain, axial, radial = strain[:n], axial[:n], radial[:n]
        inv = stress_invariants(axial, radial)

        # This app's own strain/stress convention keeps compression
        # positive, but the radial/volumetric/equivalent strain formulas
        # (Special Tests page) use the opposite sign convention: axial
        # strain negative in compression, radial strain positive (the
        # specimen widens) -- negate axial strain going into them only,
        # leaving the stress-side calculations above untouched.
        axial_strain_signed = -strain
        eps_rad = radial_strain(axial_strain_signed, cell_defl[:n], conf.alpha00, conf.alpha0_l2)
        eps_v = volumetric_strain(axial_strain_signed, eps_rad)
        eps_eq = equivalent_strain(axial_strain_signed, eps_rad)

        # Strain RATES (d/dt of the three strains above) -- outside any
        # Monte-Carlo loop, from this result's own mean curves and time base
        # (only possible if the result carries a time vector).
        result_time = np.asarray(result.get("time", []), dtype=float)
        if len(result_time) >= 2:
            result_time = result_time[:n]
        rates = {}
        if len(result_time) >= 2:
            try:
                rates = {
                    "radial_strain_rate": strain_rate_from_strain(eps_rad, result_time),
                    "volumetric_strain_rate": strain_rate_from_strain(eps_v, result_time),
                    "equivalent_strain_rate": strain_rate_from_strain(eps_eq, result_time),
                }
            except ValueError:
                rates = {}

        laser_idx, laser_sensor = next(
            ((i, s) for i, s in enumerate(self.project.custom_sensors)
             if s.role == "laser_velocity" and s.enabled and len(s.data) > 0),
            (None, None),
        )
        l_stress = l_std = None
        if laser_sensor is not None:
            # Same Monte-Carlo-cut preference as the cell sensor above.
            _, laser_signal, laser_signal_std, laser_used_mc = self._sensor_signal_for_result(laser_idx, result)
            if not laser_used_mc and toe_n_removed:
                # Same toe-correction leading-sample fix as the confinement
                # cell sensor above, and for the same reason.
                lcut = min(toe_n_removed, len(laser_signal))
                laser_signal = laser_signal[lcut:]
                laser_signal_std = np.asarray(laser_signal_std)
                laser_signal_std = laser_signal_std[lcut:] if len(laser_signal_std) >= lcut else np.asarray([])
            ob = self.project.output_bar
            l_stress_full = laser_stress(
                ob_diameter=ob.params["D"].mean, ob_rho=ob.params["rho"].mean, ob_c0=ob.params["C0"].mean,
                specimen_diameter=self.project.specimen.params["D"].mean, laser_velocity=laser_signal,
            )
            m0 = min(n, len(l_stress_full))
            l_stress = l_stress_full[:m0]
            # `laser_signal_std` is the uncertainty of the laser VELOCITY signal
            # (the sensor's own declared +/-%, plus Monte-Carlo alignment jitter),
            # whereas triaxial_synthetic_draws expects the uncertainty of the
            # laser STRESS (MPa). laser_stress() is a pure proportionality
            # (rho*C0/2*(D_bar/D_specimen)^2 * v), so the same conversion maps the
            # velocity std onto the stress std. Passing the velocity std as-is
            # (a few tenths of m/s vs. stresses of hundreds of MPa) made the laser
            # band vanish.
            l_std_velocity = np.asarray(laser_signal_std, dtype=float)[: min(m0, len(laser_signal_std))]
            l_std = laser_stress(
                ob_diameter=ob.params["D"].mean, ob_rho=ob.params["rho"].mean, ob_c0=ob.params["C0"].mean,
                specimen_diameter=self.project.specimen.params["D"].mean, laser_velocity=l_std_velocity,
            )

        # Uncertainty band for every quantity here. Prefers REAL per-Monte-
        # Carlo-draw data for the GAGE-based quantities when the Processing
        # tab computed it inside its own loop (see core.confinement.
        # per_draw_confinement_quantities and the "Include this cut in the
        # Monte-Carlo loop" toggle on the Specimen-Mounted Sensor Alignment
        # card) -- genuine uncertainty propagation, since it inherits each
        # draw's own real specimen/bar/gage-cut realization, rather than
        # the synthetic-redraw approximation below (which independently
        # perturbs the already-aggregated mean curves and so discards the
        # real correlation between axial stress and the cell signal). The
        # LASER-based variants are never computed in that loop -- laser
        # stress stays a post-hoc, Results & Export-only calculation, out
        # of scope for this -- so they always come from the synthetic
        # approximation, same as before. Also the ONLY source at all when
        # no per-draw confinement data exists for this result (older cached
        # results, or confinement-cell sensor not Monte-Carlo-cut/QS).
        has_real = f"confinement_radial_stress_std" in result
        # Also needed when a rate has no real per-draw std (a run processed
        # before the strain rates were computed in the loop, say).
        rates_lack_real_std = any(f"confinement_{k}_std" not in result for k in rates)
        need_synth = laser_sensor is not None or not has_real or rates_lack_real_std
        env_std = {}
        if need_synth:
            try:
                synth = triaxial_synthetic_draws(
                    strain, axial, axial_std, cell_defl[:n], cell_defl_std[:n] if len(cell_defl_std) else None,
                    conf, self.project.specimen.params["L"].mean,
                    laser_stress_value=l_stress, laser_stress_std=l_std, n_draws=200,
                    time=result_time if len(result_time) >= 2 else None,
                )
                env_std = {q: np.std(np.asarray(v), axis=0) for q, v in synth.items() if q != "strain"}
            except Exception:  # noqa: BLE001 - the uncertainty band is a nice-to-have; fall back to plain curves if it fails
                env_std = {}
        if has_real:
            # Override the gage-based quantities (and only those -- the
            # laser keys above, if any, are left untouched) with the real
            # per-draw std the Processing tab already aggregated.
            real_map = {
                "axial_stress": f"{stress_key}_std",
                "radial_stress": "confinement_radial_stress_std",
                "pressure": "confinement_pressure_std",
                "deviator": "confinement_deviator_std",
                "radial_strain": "confinement_radial_strain_std",
                "volumetric_strain": "confinement_volumetric_strain_std",
                "equivalent_strain": "confinement_equivalent_strain_std",
                "radial_strain_rate": "confinement_radial_strain_rate_std",
                "volumetric_strain_rate": "confinement_volumetric_strain_rate_std",
                "equivalent_strain_rate": "confinement_equivalent_strain_rate_std",
            }
            for synth_key, result_key in real_map.items():
                if result_key in result:
                    env_std[synth_key] = np.asarray(result[result_key])

        data = {"strain": strain, "axial": axial, "radial": radial,
                "pressure": inv.pressure, "deviator": inv.deviator,
                "radial_strain": eps_rad, "volumetric_strain": eps_v, "equivalent_strain": eps_eq,
                **rates}
        if env_std:
            key_map = {"axial": "axial_stress", "radial": "radial_stress", "pressure": "pressure",
                       "deviator": "deviator", "radial_strain": "radial_strain",
                       "volumetric_strain": "volumetric_strain", "equivalent_strain": "equivalent_strain",
                       "radial_strain_rate": "radial_strain_rate",
                       "volumetric_strain_rate": "volumetric_strain_rate",
                       "equivalent_strain_rate": "equivalent_strain_rate"}
            for data_key, synth_key in key_map.items():
                if synth_key in env_std:
                    m = min(len(data[data_key]), len(env_std[synth_key]))
                    data[f"{data_key}_std"] = env_std[synth_key][:m]

        sigma = self.project.uncertainty_sigma
        ax.plot(strain, inv.deviator, color="black", label="Stress deviator")
        ax.plot(strain, axial, color="red", label="Measured axial stress")
        ax.plot(strain, inv.pressure, color="blue", label="Hydrostatic pressure")
        ax.plot(strain, radial, color="green", label="Radial stress")
        if env_std:
            for data_key, color in (("deviator", "black"), ("axial", "red"), ("pressure", "blue"), ("radial", "green")):
                std_key = f"{data_key}_std"
                if std_key in data:
                    m = min(n, len(data[std_key]))
                    ax.fill_between(strain[:m], data[data_key][:m] - sigma * data[std_key][:m],
                                     data[data_key][:m] + sigma * data[std_key][:m], color=color, alpha=0.15)

        if laser_sensor is not None:
            m = min(n, len(l_stress))
            inv_las = stress_invariants(l_stress[:m], radial[:m])
            data.update({"axial_laser": l_stress[:m], "pressure_laser": inv_las.pressure, "deviator_laser": inv_las.deviator})
            ax.plot(strain[:m], inv_las.deviator, color="grey", linestyle="--", label="Laser stress deviator")
            ax.plot(strain[:m], l_stress[:m], color="orange", linestyle="--", label="Laser axial stress")
            ax.plot(strain[:m], inv_las.pressure, color="cyan", linestyle="--", label="Laser hydrostatic pressure")

            if env_std:
                for synth_key, data_key, color in (
                    ("axial_stress_laser", "axial_laser", "orange"),
                    ("pressure_laser", "pressure_laser", "cyan"),
                    ("deviator_laser", "deviator_laser", "grey"),
                ):
                    if synth_key in env_std:
                        mm = min(len(data[data_key]), len(env_std[synth_key]))
                        std_vals = env_std[synth_key][:mm]
                        data[f"{data_key}_std"] = std_vals
                        ax.fill_between(strain[:mm], data[data_key][:mm] - sigma * std_vals,
                                         data[data_key][:mm] + sigma * std_vals, color=color, alpha=0.15)

        ax.set_xlabel("Nominal axial sample strain (-)")
        ax.set_ylabel("Stress (MPa)")
        ax.legend(fontsize=8)
        # Load/Displacement Data (QS) tests have no wave-travel-time delay
        # to correct, so their confinement-cell sensor is never a
        # Monte-Carlo cut (cell_used_mc is always False there) -- the
        # "tick .../Processing page" advice below only applies to
        # SHPB/Direct Impact, which do have that toggle.
        is_qs_result = result.get("label") == "Load/Displacement Data"
        if cell_used_mc:
            cell_note = " Cell sensor: Monte-Carlo cut."
        elif is_qs_result:
            cell_note = " Cell sensor: raw signal (assumed already on the same time base as this result)."
        else:
            cell_note = (
                " Cell sensor: raw/manually-cut signal (tick \"Include this cut in the Monte-Carlo loop\" "
                "on the Processing page and re-run \"Process\" to use the aligned/averaged cut here instead)."
            )
        if has_real:
            if is_qs_result:
                uncertainty_note = (
                    " Uncertainty (gage-based): real per-draw Monte-Carlo propagation from this run's own "
                    "axial stress/strain (the confinement-cell signal itself is not Monte-Carlo-varied on "
                    "this tab)."
                )
            else:
                uncertainty_note = " Uncertainty (gage-based): real per-draw Monte-Carlo propagation."
        else:
            uncertainty_note = (
                " Uncertainty (gage-based): synthetic approximation" + (
                    "." if is_qs_result else
                    " (tick \"Include this cut in the Monte-Carlo loop\" for the confinement-cell sensor and "
                    "re-run \"Process\" for real propagation instead)."
                )
            )
        self.confine_status.setText(f"{n} points.{cell_note}{uncertainty_note}")
        self._confinement_data = data
        self.confine_canvas.refresh()
        self._refresh_confinement_pq(data)
        self._refresh_confinement_pv(data)
        self._refresh_confinement_strains(data)
        self._refresh_confinement_rates(data)

    def _refresh_confinement_rates(self, data: dict):
        self.confine_rates_canvas.clear()
        ax5 = self.confine_rates_canvas.figure.add_subplot(111)
        keys = [k for k in ("equivalent_strain_rate", "radial_strain_rate", "volumetric_strain_rate") if k in data]
        if "strain" not in data or not keys:
            ax5.text(0.5, 0.5, "Strain rates need a time vector on this result.", transform=ax5.transAxes,
                     ha="center", va="center", alpha=0.6)
            ax5.set_xticks([]); ax5.set_yticks([])
            self.confine_rates_canvas.refresh()
            return
        strain = data["strain"]
        n = len(strain)
        sigma = self.project.uncertainty_sigma
        style = {"equivalent_strain_rate": ("black", "Equivalent strain rate"),
                 "radial_strain_rate": ("green", "Radial strain rate"),
                 "volumetric_strain_rate": ("blue", "Volumetric strain rate")}
        for key in keys:
            color, label = style[key]
            m = min(n, len(data[key]))
            ax5.plot(strain[:m], data[key][:m], color=color, label=label)
            std_key = f"{key}_std"
            if std_key in data:
                mm = min(m, len(data[std_key]))
                ax5.fill_between(strain[:mm], data[key][:mm] - sigma * data[std_key][:mm],
                                 data[key][:mm] + sigma * data[std_key][:mm], color=color, alpha=0.15)
        ax5.set_xlabel("Nominal axial strain (-)")
        ax5.set_ylabel("Strain rate (1/s)")
        ax5.legend(fontsize=8)
        self.confine_rates_canvas.refresh()

    def _refresh_confinement_strains(self, data: dict):
        self.confine_strains_canvas.clear()
        ax4 = self.confine_strains_canvas.figure.add_subplot(111)
        if "strain" not in data or "radial_strain" not in data:
            ax4.set_xticks([]); ax4.set_yticks([])
            self.confine_strains_canvas.refresh()
            return
        strain = data["strain"]
        n = len(strain)
        sigma = self.project.uncertainty_sigma
        # These three depend only on the axial and cell strains, not on
        # which axial stress source (gage/laser) was used, so there's a
        # single curve for each -- no separate laser variant here.
        for key, color, label in (
            ("equivalent_strain", "black", "Equivalent strain"),
            ("radial_strain", "green", "Radial strain"),
            ("volumetric_strain", "blue", "Volumetric strain"),
        ):
            if key in data:
                m = min(n, len(data[key]))
                ax4.plot(strain[:m], data[key][:m], color=color, label=label)
                std_key = f"{key}_std"
                if std_key in data:
                    mm = min(m, len(data[std_key]))
                    ax4.fill_between(strain[:mm], data[key][:mm] - sigma * data[std_key][:mm],
                                      data[key][:mm] + sigma * data[std_key][:mm], color=color, alpha=0.15)
        ax4.set_xlabel("Nominal axial strain (-)")
        ax4.set_ylabel("Strain (-)")
        ax4.legend(fontsize=8)
        self.confine_strains_canvas.refresh()

    def _refresh_confinement_pq(self, data: dict):
        self.confine_pq_canvas.clear()
        ax2 = self.confine_pq_canvas.figure.add_subplot(111)
        if "pressure" not in data or "deviator" not in data:
            ax2.set_xticks([]); ax2.set_yticks([])
            self.confine_pq_canvas.refresh()
            return
        sigma = self.project.uncertainty_sigma
        ax2.plot(data["pressure"], data["deviator"], color="blue", marker="o", markersize=2, linewidth=1,
                 label="Stress path (gage)")
        if "pressure_std" in data and "deviator_std" in data:
            n = min(len(data["pressure"]), len(data["pressure_std"]), len(data["deviator_std"]))
            ax2.errorbar(data["pressure"][:n], data["deviator"][:n],
                         xerr=sigma * data["pressure_std"][:n], yerr=sigma * data["deviator_std"][:n],
                         fmt="none", ecolor="blue", alpha=0.15, linewidth=1)
        if "pressure_laser" in data and "deviator_laser" in data:
            ax2.plot(data["pressure_laser"], data["deviator_laser"], color="orange", marker="o", markersize=2,
                      linewidth=1, linestyle="--", label="Stress path (laser)")
            if "pressure_laser_std" in data and "deviator_laser_std" in data:
                n = min(len(data["pressure_laser"]), len(data["pressure_laser_std"]), len(data["deviator_laser_std"]))
                ax2.errorbar(data["pressure_laser"][:n], data["deviator_laser"][:n],
                             xerr=sigma * data["pressure_laser_std"][:n], yerr=sigma * data["deviator_laser_std"][:n],
                             fmt="none", ecolor="orange", alpha=0.15, linewidth=1)
        ax2.set_xlabel("Hydrostatic pressure p (MPa)")
        ax2.set_ylabel("Stress deviator q (MPa)")
        ax2.legend(fontsize=8)
        self.confine_pq_canvas.refresh()

    def _refresh_confinement_pv(self, data: dict):
        self.confine_pv_canvas.clear()
        ax3 = self.confine_pv_canvas.figure.add_subplot(111)
        if "pressure" not in data or "volumetric_strain" not in data:
            ax3.set_xticks([]); ax3.set_yticks([])
            self.confine_pv_canvas.refresh()
            return
        n = min(len(data["pressure"]), len(data["volumetric_strain"]))
        sigma = self.project.uncertainty_sigma
        ax3.plot(data["volumetric_strain"][:n], data["pressure"][:n], color="blue", marker="o", markersize=2,
                  linewidth=1, label="Pressure (gage)")
        if "pressure_std" in data and "volumetric_strain_std" in data:
            nn = min(n, len(data["pressure_std"]), len(data["volumetric_strain_std"]))
            ax3.errorbar(data["volumetric_strain"][:nn], data["pressure"][:nn],
                         xerr=sigma * data["volumetric_strain_std"][:nn], yerr=sigma * data["pressure_std"][:nn],
                         fmt="none", ecolor="blue", alpha=0.15, linewidth=1)
        if "pressure_laser" in data:
            m = min(n, len(data["pressure_laser"]))
            ax3.plot(data["volumetric_strain"][:m], data["pressure_laser"][:m], color="orange", marker="o",
                      markersize=2, linewidth=1, linestyle="--", label="Pressure (laser)")
            if "pressure_laser_std" in data and "volumetric_strain_std" in data:
                mm = min(m, len(data["pressure_laser_std"]), len(data["volumetric_strain_std"]))
                ax3.errorbar(data["volumetric_strain"][:mm], data["pressure_laser"][:mm],
                             xerr=sigma * data["volumetric_strain_std"][:mm], yerr=sigma * data["pressure_laser_std"][:mm],
                             fmt="none", ecolor="orange", alpha=0.15, linewidth=1)
        ax3.set_xlabel("Volumetric strain (-)")
        ax3.set_ylabel("Hydrostatic pressure p (MPa)")
        ax3.legend(fontsize=8)
        self.confine_pv_canvas.refresh()

    def _on_export_confinement(self):
        self._sync_before_export()
        if self._confinement_data is None:
            self.confine_status.setText("Nothing to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export confinement result", "confinement.csv", "CSV files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(list(self._confinement_data.keys()))
                for row in zip(*self._confinement_data.values()):
                    writer.writerow([f"{v:.8g}" for v in row])
        except OSError as exc:
            self.confine_status.setText(f"Could not export: {exc}")
            return
        self.confine_status.setText(f"Exported to {path.split('/')[-1]}")

    def _on_compare_toggled(self, checked):
        self.add_curve_button.setVisible(checked)
        self.clear_curves_button.setVisible(checked)
        if not checked:
            self._comparison_curves = []
        self._refresh_plot()

    def _on_add_curve(self):
        result = self._current_result()
        x_key, y_key = self.x_combo.currentData(), self.y_combo.currentData()
        if result is None or x_key is None or y_key is None:
            return
        x, y = np.asarray(result[x_key]), np.asarray(result[y_key])
        n = min(len(x), len(y))
        std = None
        std_key = f"{y_key}_std"
        if std_key in result:
            std = np.asarray(result[std_key])[:min(n, len(result[std_key]))]
        self._comparison_curves.append({
            "label": f"{result['label']}: {y_key} vs {x_key}",
            "x": x[:n], "y": y[:n], "std": std,
        })
        self._refresh_plot()

    def _on_clear_curves(self):
        self._comparison_curves = []
        self._refresh_plot()

    def _refresh_plot(self, *_args):
        result = self._current_result()
        self.canvas.clear()
        ax = self.canvas.figure.add_subplot(111)

        if self.compare_check.isChecked() and self._comparison_curves:
            sigma = self.project.uncertainty_sigma
            colors = ["#5b8cff", "#ff9f43", "#4cd97b", "#ff6b6b", "#c77dff", "#f6c453"]
            for i, curve in enumerate(self._comparison_curves):
                color = colors[i % len(colors)]
                ax.plot(curve["x"], curve["y"], "-", color=color, label=curve["label"])
                if curve["std"] is not None:
                    m = min(len(curve["x"]), len(curve["std"]))
                    ax.fill_between(curve["x"][:m], curve["y"][:m] - sigma * curve["std"][:m],
                                     curve["y"][:m] + sigma * curve["std"][:m], color=color, alpha=0.15)
            ax.set_xlabel(self.x_combo.currentText())
            ax.set_ylabel(self.y_combo.currentText())
            ax.legend(fontsize=8)
            self.info_label.setText(f"{len(self._comparison_curves)} curve(s) in comparison.")
            self.canvas.refresh()
            return

        if result is None:
            ax.text(0.5, 0.5, "No results computed yet.\nRun a Processing tab (Load/Displacement Data / SHPB / Direct Impact) first.",
                    ha="center", va="center", color="#8b91a1", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self.info_label.setText("")
            self.canvas.refresh()
            return

        x_key, y_key = self.x_combo.currentData(), self.y_combo.currentData()
        if x_key is None or y_key is None:
            self.canvas.refresh()
            return

        x, y = np.asarray(result[x_key]), np.asarray(result[y_key])
        n = min(len(x), len(y))
        self.info_label.setText(f"{n} points.")
        ax.plot(x[:n], y[:n], "b-", label=f"{y_key} vs {x_key}")
        std_key = f"{y_key}_std"
        if std_key in result:
            sigma = self.project.uncertainty_sigma
            std = np.asarray(result[std_key])
            m = min(n, len(std))
            ax.fill_between(x[:m], y[:m] - sigma * std[:m], y[:m] + sigma * std[:m],
                             color="b", alpha=0.2, label=f"±{sigma}σ")
        ax.set_xlabel(self._label_for_key(x_key))
        ax.set_ylabel(self._label_for_key(y_key))
        ax.legend()
        self.canvas.refresh()

    def _on_export(self, fmt: str):
        self._sync_before_export()
        result = self._current_result()
        if result is None:
            self.export_status.setText("No result selected to export.")
            return

        keys = self._quantity_keys(result)
        keys += [f"{k}_std" for k in keys if f"{k}_std" in result]
        arrays = {k: np.asarray(result[k]) for k in keys}

        # Include the confinement / triaxial stress analysis (gage-based
        # and laser-based, if computed) alongside the main result columns.
        if self._confinement_data:
            for key, values in self._confinement_data.items():
                if key not in arrays:
                    arrays[f"confinement_{key}"] = np.asarray(values)

        columns, min_len = equalize_export_columns(arrays)

        default_name = f"{result['label'].replace(' ', '_')}.{fmt}"
        filter_str = "CSV files (*.csv)" if fmt == "csv" else "Text files (*.txt)"
        path, _ = QFileDialog.getSaveFileName(self, "Export result", default_name, filter_str)
        if not path:
            return

        try:
            self._write_columns(path, fmt, columns)
        except OSError as exc:
            self.export_status.setText(f"Could not export: {exc}")
            return
        self.export_status.setText(f"Exported {min_len} rows x {len(columns)} columns to {path.split('/')[-1]}")

    def _write_columns(self, path: str, fmt: str, columns: dict) -> None:
        if fmt == "csv":
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(list(columns.keys()))
                for row in zip(*columns.values()):
                    writer.writerow([f"{v:.8g}" for v in row])
        else:
            data = np.column_stack(list(columns.values()))
            header = "\t".join(columns.keys())
            with open(path, "w") as f:
                f.write(f"# {header}\n")
                np.savetxt(f, data, delimiter="\t", fmt="%.8g")

    def _on_export_full_test(self):
        self._sync_before_export()
        result = self._current_result()
        if result is None:
            self.export_status.setText("No result selected to export.")
            return

        keys = self._quantity_keys(result)
        keys += [f"{k}_std" for k in keys if f"{k}_std" in result]
        arrays = {k: np.asarray(result[k]) for k in keys}
        if self._confinement_data:
            for key, values in self._confinement_data.items():
                if key not in arrays:
                    arrays[f"confinement_{key}"] = np.asarray(values)
        columns, min_len = equalize_export_columns(arrays)

        default_name = f"{result['label'].replace(' ', '_')}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export full test (result + parameters)", default_name, "CSV files (*.csv)")
        if not path:
            return

        import core.setup_profile as setup_profile
        json_path = path.rsplit(".", 1)[0] + ".json"
        try:
            self._write_columns(path, "csv", columns)
            with open(json_path, "w") as f:
                json.dump({"name": result["label"], **setup_profile.to_dict(self.project, include_signals=True)}, f, indent=2)
        except OSError as exc:
            self.export_status.setText(f"Could not export: {exc}")
            return
        self.export_status.setText(
            f"Exported full test: {path.split('/')[-1]} (data) + {json_path.split('/')[-1]} (parameters). "
            "Load it back from Test Setup's \"Load Test\" section."
        )

