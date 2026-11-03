"""
Test Setup page.

Lets the user choose compression/tension/shear, sub-geometry, name,
initial/ambient temperature, and the loading regime, all on a single page.
The Normal/Uniform temperature fields are MCParamRow instances, so a single
widget type covers both the fixed-value and uncertainty-bearing cases
instead of duplicated field pairs.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLineEdit, QSpinBox, QWidget, QVBoxLayout, QCheckBox, QPushButton, QHBoxLayout, QLabel, QFileDialog, QMessageBox

from core.mc import MCValue
from core.models import (
    TEST_MODES, COMPRESSION_GEOMETRIES, TENSION_GEOMETRIES, SHEAR_GEOMETRIES, LOADING_REGIMES,
)
from core.parallel import available_cpu_count
from ui.widgets import Card, LabeledField, MCParamTable, make_scroll_page
import core.setup_profile as setup_profile


class TestSetupPage(QWidget):
    loadingRegimeChanged = Signal()

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project

        scroll, layout = make_scroll_page(
            "Test Setup",
            "Define what kind of test this is. This drives which fields appear on the "
            "Specimen page and which processing engine is used.",
        )

        identity = Card(
            "Analysis identity",
            help_html=(
                "<p>Basic identifying settings for this analysis.</p>"
                "<ul>"
                "<li><b>Analysis name</b>: a free-form label for this project (used e.g. when "
                "exporting a full test, and shown back when re-loading it).</li>"
                "<li><b>Monte-Carlo draws</b>: how many random samples of every Monte-Carlo "
                "parameter (bar/specimen properties, uncertainties, ...) are drawn and processed "
                "when running a Processing tab. More draws give smoother, more reliable "
                "uncertainty bands but take proportionally longer to process -- the Performance "
                "(multi-core) card below can offset that cost.</li>"
                "</ul>"
            ),
        )
        self.name_edit = QLineEdit(project.name)
        self.name_edit.editingFinished.connect(self._on_name_changed)
        identity.add(LabeledField("Analysis name", self.name_edit))

        self.n_mc_spin = QSpinBox()
        self.n_mc_spin.setRange(1, 200_000)
        self.n_mc_spin.setValue(project.n_montecarlo)
        self.n_mc_spin.valueChanged.connect(self._on_n_mc_changed)
        identity.add(LabeledField("Monte-Carlo draws", self.n_mc_spin))
        layout.addWidget(identity)

        mode = Card(
            "Loading condition",
            "Compression / Tension / Shear-Compression-Tension (notched) / Shear, and the "
            "sub-geometry within it (where applicable).",
            help_html=(
                "<p>Sets what kind of test this is -- this choice drives which fields appear on "
                "the Specimen page and which processing formulas are used.</p>"
                "<ul>"
                "<li><b>Test mode</b>: Compression, Tension, Shear-Compression/Tension (Notched), "
                "or Shear.</li>"
                "<li><b>Geometry</b>: the sub-geometry within that mode (e.g. cylindrical vs. "
                "cubic for Compression) -- hidden for Notched, which has no sub-geometry.</li>"
                "</ul>"
            ),
        )
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(TEST_MODES)
        self.mode_combo.setCurrentText(project.test_mode)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode.add(LabeledField("Test mode", self.mode_combo))

        self.geometry_combo = QComboBox()
        self.geometry_row = LabeledField("Geometry", self.geometry_combo)
        mode.add(self.geometry_row)
        self.geometry_combo.currentTextChanged.connect(self._on_geometry_changed)
        self._refresh_geometry_options()
        layout.addWidget(mode)

        regime = Card(
            "Loading regime",
            "Which physical setup produced the signals.",
            help_html=(
                "<p>Selects which physical test setup produced the signals -- this decides which "
                "processing engine (Load/Displacement Data, Hopkinson Bars/SHPB, or Direct Impact) "
                "and which pages (e.g. Input/Output Bar) are relevant for this test.</p>"
            ),
        )
        self.regime_combo = QComboBox()
        self.regime_combo.addItems(LOADING_REGIMES)
        self.regime_combo.setCurrentText(project.loading_regime)
        self.regime_combo.currentTextChanged.connect(self._on_regime_changed)
        regime.add(LabeledField("Loading regime", self.regime_combo))
        layout.addWidget(regime)

        temps = Card(
            "Temperature",
            "Ambient and initial specimen temperature, each as a Monte-Carlo parameter.",
            help_html=(
                "<p>Sets the specimen's initial temperature and the surrounding ambient "
                "temperature, each as a Monte-Carlo parameter (Mean + Dispersion + distribution).</p>"
                "<ul>"
                "<li>Used by any heated-specimen (heat_check) correction, and wherever "
                "temperature-dependent material properties are looked up.</li>"
                "<li>Set Dispersion to 0 for a room-temperature test where temperature isn't a "
                "meaningful source of uncertainty.</li>"
                "</ul>"
            ),
        )
        self.t0_value = MCValue(name="Initial Temperature", unit="K", mean=project.thermal.initial_temperature_k)
        self.ta_value = MCValue(name="Ambient Temperature", unit="K", mean=project.thermal.ambient_temperature_k)
        self.temp_table = MCParamTable([self.t0_value, self.ta_value])
        self.temp_table.changed.connect(self._on_temps_changed)
        temps.add(self.temp_table)
        layout.addWidget(temps)

        uncertainty = Card(
            "Uncertainty display",
            "How many standard deviations to show/export as the uncertainty envelope around every "
            "mean result curve.",
            help_html=(
                "<p>Sets the width of the uncertainty band drawn around every mean result curve "
                "across the app (Processing results, Batch Analysis, etc.).</p>"
                "<ul>"
                "<li><b>1&sigma;</b> &asymp; 68% coverage if the underlying distribution is "
                "Normal, <b>2&sigma;</b> &asymp; 95%, <b>3&sigma;</b> &asymp; 99.7%.</li>"
                "<li>This is a display/export choice only -- it does not change the underlying "
                "Monte-Carlo computation, only how many standard deviations are drawn/exported "
                "around the mean.</li>"
                "</ul>"
            ),
        )
        self.sigma_combo = QComboBox()
        self.sigma_combo.addItems(["1σ", "2σ", "3σ"])
        self.sigma_combo.setCurrentIndex(project.uncertainty_sigma - 1)
        self.sigma_combo.currentIndexChanged.connect(self._on_sigma_changed)
        uncertainty.add(LabeledField("Uncertainty level", self.sigma_combo))
        layout.addWidget(uncertainty)

        downsampling = Card(
            "Downsampling",
            "Keep only every Mth sample of every raw signal file loaded (bar signals, time/load/"
            "displacement, velocity, custom sensor data) -- speeds up processing of very large "
            "measurement files. Applied uniformly so all signals stay aligned.",
            help_html=(
                "<p>Reduces the size of every raw signal loaded from here on, by keeping only "
                "every Mth sample -- useful for very large measurement files (e.g. long, "
                "high-sample-rate recordings) that would otherwise slow down processing.</p>"
                "<ul>"
                "<li><b>Keep 1 point every M points</b>: e.g. M = 10 keeps 10% of the samples.</li>"
                "<li>Applied uniformly to every signal (bar signals, time/load/displacement, "
                "velocity, custom sensor data), so everything stays aligned in time.</li>"
                "<li>Only affects files loaded <i>after</i> this is enabled -- already-loaded "
                "signals are unaffected until reloaded.</li>"
                "</ul>"
            ),
        )
        self.downsample_check = QCheckBox("Downsample raw measurement files")
        self.downsample_check.setChecked(project.downsample_enabled)
        self.downsample_check.toggled.connect(self._on_downsample_toggled)
        downsampling.add(self.downsample_check)
        self.downsample_n_edit = QLineEdit(str(project.downsample_n))
        self.downsample_n_edit.editingFinished.connect(self._on_downsample_n_changed)
        self.downsample_n_row = LabeledField("Keep 1 point every M points", self.downsample_n_edit)
        self.downsample_n_row.setVisible(project.downsample_enabled)
        downsampling.add(self.downsample_n_row)
        layout.addWidget(downsampling)

        cpu_count = available_cpu_count()
        performance = Card(
            "Performance (multi-core)",
            "How many CPU cores to use for the heaviest computations: Monte-Carlo uncertainty "
            "propagation (Processing), the incident/reflected/transmitted wave-dispersion branch "
            "solves (Dispersion Corrections), image correlation point tracking (Digital Image "
            "Correlation), and the initial sampling phase of Bayesian viscoelastic calibration. "
            f"This machine has {cpu_count} logical core(s) available. Leave at 1 for the original, "
            "fully sequential behavior.",
            help_html=(
                f"<p>Controls how many CPU cores ({cpu_count} available on this machine) are used "
                "for the heaviest computations in the app.</p>"
                "<ul>"
                "<li><b>Processing</b> (Load/Displacement Data / SHPB / Direct Impact): each "
                "Monte-Carlo draw processed in parallel.</li>"
                "<li><b>Dispersion Corrections</b>: the incident/reflected/transmitted wave "
                "branches solved in parallel (up to 2-3&times; speedup, since there are at most "
                "that many branches).</li>"
                "<li><b>Digital Image Correlation</b>: point tracking across the correlation mesh "
                "parallelized per frame.</li>"
                "<li><b>Viscoelastic calibration</b>: only the initial (LHS) sampling phase of "
                "Bayesian calibration -- its later adaptive search stays sequential by design, "
                "since each step depends on the previous ones.</li>"
                "</ul>"
                "<p><b>Technical detail:</b> leave at 1 for the original, fully sequential "
                "behavior. Every computation above falls back to sequential automatically "
                "wherever the work can't actually be split across cores.</p>"
            ),
        )
        self.n_workers_spin = QSpinBox()
        self.n_workers_spin.setRange(1, max(1, cpu_count))
        self.n_workers_spin.setValue(max(1, min(project.n_workers, cpu_count)))
        self.n_workers_spin.valueChanged.connect(self._on_n_workers_changed)
        performance.add(LabeledField("Worker cores to use", self.n_workers_spin))
        self.n_workers_status = QLabel("")
        self.n_workers_status.setObjectName("cardSubtitle")
        self.n_workers_status.setWordWrap(True)
        performance.add(self.n_workers_status)
        layout.addWidget(performance)
        self._refresh_n_workers_status()

        profiles = Card(
            "Setup Profiles",
            "Save the current bar properties, specimen template, confinement/platen settings, "
            "custom sensor configuration, (for Load/Displacement Data) machine compliance settings, and the "
            "Processing/Dispersion Corrections tab settings as a named, reusable profile -- handy "
            "for switching between standard equipment setups.",
            help_html=(
                "<p>Saves/restores a whole equipment configuration as one named, reusable "
                "profile -- handy for switching between standard setups (e.g. different bar "
                "diameters) without re-entering every parameter by hand each time.</p>"
                "<ul>"
                "<li>A profile captures bar properties, specimen template, confinement/platen "
                "settings, custom sensor configuration, machine compliance settings (Load/"
                "Displacement Data), and the Processing/Dispersion Corrections tab settings.</li>"
                "<li>Type a name and click <b>Save profile</b> to store the current setup; pick a "
                "profile from the list and click <b>Load profile</b> to restore it, or <b>Delete "
                "profile</b> to remove it.</li>"
                "</ul>"
                "<p><b>Note:</b> a profile does not include loaded signal files or Monte-Carlo "
                "draw counts -- it's the equipment configuration, not a full saved test (see "
                "\"Load Test\" below for that).</p>"
            ),
        )
        save_row = QHBoxLayout()
        self.profile_name_edit = QLineEdit()
        self.profile_name_edit.setPlaceholderText("Profile name, e.g. \"Steel bars 25.4mm\"")
        save_row.addWidget(self.profile_name_edit, 1)
        self.save_profile_button = QPushButton("Save profile")
        self.save_profile_button.clicked.connect(self._on_save_profile)
        save_row.addWidget(self.save_profile_button)
        profiles.add_layout(save_row)

        load_row = QHBoxLayout()
        self.profile_combo = QComboBox()
        load_row.addWidget(self.profile_combo, 1)
        self.load_profile_button = QPushButton("Load profile")
        self.load_profile_button.clicked.connect(self._on_load_profile)
        load_row.addWidget(self.load_profile_button)
        self.delete_profile_button = QPushButton("Delete profile")
        self.delete_profile_button.clicked.connect(self._on_delete_profile)
        load_row.addWidget(self.delete_profile_button)
        profiles.add_layout(load_row)

        self.profile_status_label = QLabel("")
        self.profile_status_label.setObjectName("cardSubtitle")
        profiles.add(self.profile_status_label)
        layout.addWidget(profiles)
        self._refresh_profile_list()

        load_test = Card(
            "Load Test",
            "Load a test previously exported with Results & Export's \"Export full test\" button: "
            "restores every parameter used (like a Setup Profile), then looks in the same folder for "
            "the matching .csv/.txt result file and loads its data too, so it's immediately visible "
            "on the Results & Export page.",
            help_html=(
                "<p>Re-opens a test previously saved with Results &amp; Export's <b>Export full "
                "test</b> button.</p>"
                "<ul>"
                "<li>Pick the exported <code>.json</code> file: it restores every parameter used "
                "for that test (like a Setup Profile), then looks for the matching .csv/.txt "
                "result file in the same folder and loads its data too.</li>"
                "<li>Once loaded, the result is immediately visible on the Results &amp; Export "
                "page -- nothing needs to be reprocessed.</li>"
                "</ul>"
            ),
        )
        load_test_row = QHBoxLayout()
        self.load_test_button = QPushButton("Load Test (.json)…")
        self.load_test_button.clicked.connect(self._on_load_test)
        load_test_row.addWidget(self.load_test_button)
        load_test_row.addStretch(1)
        load_test.add_layout(load_test_row)
        self.load_test_status_label = QLabel("")
        self.load_test_status_label.setObjectName("cardSubtitle")
        self.load_test_status_label.setWordWrap(True)
        load_test.add(self.load_test_status_label)
        layout.addWidget(load_test)

        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _on_name_changed(self):
        self.project.name = self.name_edit.text()

    def _on_n_mc_changed(self, value):
        self.project.n_montecarlo = value

    def _on_mode_changed(self, text):
        self.project.test_mode = text
        self._refresh_geometry_options()

    def _refresh_geometry_options(self):
        options = {
            "Compression": COMPRESSION_GEOMETRIES,
            "Tension": TENSION_GEOMETRIES,
            "Shear": SHEAR_GEOMETRIES,
        }.get(self.project.test_mode)

        if options is None:
            # "Shear-Compression/Tension (Notched)" has no sub-geometry.
            self.geometry_combo.blockSignals(True)
            self.geometry_combo.clear()
            self.geometry_combo.blockSignals(False)
            self.geometry_row.setVisible(False)
            self.project.geometry = None
            self.project.specimen.geometry = None
            return

        self.geometry_row.setVisible(True)
        self.geometry_combo.blockSignals(True)
        self.geometry_combo.clear()
        self.geometry_combo.addItems(options)
        # Keep the restored/current geometry if it's still valid for this
        # mode (important for session restore -- otherwise this would
        # always reset to the first option and silently discard whatever
        # geometry was loaded from the previous session).
        chosen = self.project.geometry if self.project.geometry in options else options[0]
        self.geometry_combo.setCurrentText(chosen)
        self.geometry_combo.blockSignals(False)
        self.project.geometry = chosen
        self.project.specimen.geometry = chosen

    def _on_geometry_changed(self, text):
        if text:
            self.project.geometry = text
            self.project.specimen.geometry = text

    def _on_regime_changed(self, text):
        self.project.loading_regime = text
        self.loadingRegimeChanged.emit()

    def _on_temps_changed(self):
        self.project.thermal.initial_temperature_k = self.t0_value.mean
        self.project.thermal.ambient_temperature_k = self.ta_value.mean

    def _on_sigma_changed(self, index):
        self.project.uncertainty_sigma = index + 1

    def _on_downsample_toggled(self, checked):
        self.project.downsample_enabled = checked
        self.downsample_n_row.setVisible(checked)

    def _on_downsample_n_changed(self):
        try:
            n = int(self.downsample_n_edit.text())
        except ValueError:
            self.downsample_n_edit.setText(str(self.project.downsample_n))
            return
        self.project.downsample_n = max(1, n)
        self.downsample_n_edit.setText(str(self.project.downsample_n))

    def _on_n_workers_changed(self, value):
        self.project.n_workers = value
        self._refresh_n_workers_status()

    def _refresh_n_workers_status(self):
        n = self.project.n_workers
        if n <= 1:
            self.n_workers_status.setText("Sequential (1 core) -- the original behavior.")
        else:
            self.n_workers_status.setText(
                f"Using up to {n} worker processes. Each computation still falls back to the fully "
                "sequential path automatically wherever the work can't be split across cores (e.g. the "
                "adaptive part of Bayesian calibration)."
            )

    def _refresh_profile_list(self):
        current = self.profile_combo.currentText()
        names = setup_profile.list_profiles()
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItems(names)
        self.profile_combo.blockSignals(False)
        if current in names:
            self.profile_combo.setCurrentText(current)

    def _on_save_profile(self):
        name = self.profile_name_edit.text().strip()
        if not name:
            self.profile_status_label.setText("Type a profile name first.")
            return
        try:
            path = setup_profile.save_profile(self.project, name)
        except OSError as exc:
            self.profile_status_label.setText(f"Could not save profile: {exc}")
            return
        self.profile_status_label.setText(f"Saved profile \"{name}\" ({path.name}).")
        self._refresh_profile_list()

    def _on_load_profile(self):
        name = self.profile_combo.currentText()
        if not name:
            self.profile_status_label.setText("No profile selected -- save one first, or pick one from the list.")
            return
        try:
            setup_profile.load_profile(self.project, name)
        except (OSError, ValueError) as exc:
            self.profile_status_label.setText(f"Could not load profile: {exc}")
            return
        self.profile_status_label.setText(f"Loaded profile \"{name}\".")

    def _on_delete_profile(self):
        name = self.profile_combo.currentText()
        if not name:
            self.profile_status_label.setText("No profile selected -- pick one from the list first.")
            return
        answer = QMessageBox.question(
            self,
            "Delete profile",
            f"Delete the setup profile \"{name}\"? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            setup_profile.delete_profile(name)
        except OSError as exc:
            self.profile_status_label.setText(f"Could not delete profile: {exc}")
            return
        self.profile_status_label.setText(f"Deleted profile \"{name}\".")
        self._refresh_profile_list()

    def _on_load_test(self):
        json_path, _ = QFileDialog.getOpenFileName(self, "Load Test", "", "JSON files (*.json)")
        if not json_path:
            return
        try:
            data = setup_profile.load_from_path(self.project, json_path)
        except (OSError, ValueError) as exc:
            self.load_test_status_label.setText(f"Could not load test parameters: {exc}")
            return

        n_ib, n_ob = len(self.project.input_bar.signal), len(self.project.output_bar.signal)
        had_bar_signals = bool(data.get("bar_signals")) or any(
            (data.get("last_paths") or {}).get(k) for k in ("input_bar_signal", "output_bar_signal"))
        if not had_bar_signals:
            signal_note = ""  # e.g. a quasi-static test: no bar gages involved
        elif data.get("bar_signals"):
            signal_note = f" Bar gage signals restored (input: {n_ib} samples, output: {n_ob} samples)."
        elif n_ib or n_ob:
            signal_note = (" Bar gage signals re-read from their original file paths "
                           "(older export without embedded signals).")
        else:
            signal_note = (" Bar gage signals could NOT be restored (older export without embedded "
                           "signals, and the original signal files were not found) -- re-export this test "
                           "from the app to embed them.")

        base = json_path.rsplit(".", 1)[0]
        result_path = None
        for ext in (".csv", ".txt"):
            if __import__("os").path.exists(base + ext):
                result_path = base + ext
                break

        label = data.get("name", "")
        key_by_label = {"Load/Displacement Data": "QS", "Quasi-Static": "QS", "SHPB": "SHPB", "Direct Impact": "DI"}
        result_key = key_by_label.get(label)

        if result_path is None:
            self.load_test_status_label.setText(
                f"Loaded test parameters from {json_path.split('/')[-1]}, but no matching .csv/.txt "
                "result file was found next to it." + signal_note
            )
            return

        try:
            columns = setup_profile.load_test_csv(result_path)
        except (OSError, ValueError) as exc:
            self.load_test_status_label.setText(f"Loaded parameters, but could not read the result file: {exc}")
            return

        if result_key is None:
            result_key = "QS" if "displacement" in columns and "load" in columns else "SHPB"
        self.project.last_results[result_key] = {"label": label or result_key, **columns}
        self.load_test_status_label.setText(
            f"Loaded test parameters and result data from {json_path.split('/')[-1]} + "
            f"{result_path.split('/')[-1]} -- see it on the Results & Export page." + signal_note
        )
