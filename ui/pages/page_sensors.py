"""
Custom sensors page.

Replaces config_custom_sensors.py's Toplevel, which defined the same
"enable / load time+data / uncertainty (%) / uncertainty (absolute) /
multiplicative factor / cut curve" block three times by hand. Now one
_SensorCard is reused for all three configurable sensor slots, and the
"cut curve" tool reuses ui/span_picker.py's TimeWindowPicker (dragging
sets a window, but only its start edge is used as the cut point, matching
the original's single-sided cut).
"""
from __future__ import annotations

import numpy as np
from core.cpg_calc import THERMAL_MIN_DEGREE, THERMAL_MAX_DEGREE
from core.downsampling import downsample
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QCheckBox, QLineEdit, QPushButton, QLabel, QHBoxLayout,
    QComboBox, QGridLayout,
)

from ui.widgets import Card, FileSelectRow, LabeledField, make_scroll_page, TimeSourceRow
from ui.span_picker import TimeWindowPicker
from ui.plot_canvas import PlotCanvas
from ui import theme

ROLES = [
    ("none", "No special role"),
    ("confinement_cell", "Confinement cell strain gage (quasi-oedometric test)"),
    ("laser_velocity", "Laser velocimetry (output-bar surface velocity)"),
    ("temperature", "Temperature sensor"),
    ("humidity", "Humidity sensor"),
    ("specimen_strain_gage", "Strain gage glued directly on the specimen"),
]


class _SensorCard(Card):
    def __init__(self, project, sensor, index: int, on_change, on_remove=None, parent=None):
        super().__init__(
            f"{sensor.label}",
            "Optional auxiliary measurement (extra load cell, extensometer, thermocouple...) "
            "beyond the standard bar/specimen instrumentation.",
            help_html=(
                "<p>Configures one optional auxiliary measurement (extra load cell, "
                "extensometer, thermocouple...) beyond the standard bar/specimen "
                "instrumentation.</p>"
                "<ul>"
                "<li><b>Enable this sensor</b>: turns it on/off without deleting its "
                "configuration.</li>"
                "<li><b>Role</b>: assigns a special meaning some pages look for specifically -- "
                "e.g. \"Confinement cell strain gage\" feeds the Results &amp; Export "
                "confinement plots, \"Laser velocimetry\" and \"Strain gage glued directly on "
                "the specimen\" are read by other pages. \"No special role\" just publishes the "
                "signal for viewing/export.</li>"
                "<li><b>This sensor is mounted directly on the specimen</b>: enables the "
                "wave-travel-time alignment on the SHPB/Direct Impact Processing tabs "
                "(\"Specimen-Mounted Sensor Alignment\" card) -- otherwise the raw recorded "
                "timing is used as-is.</li>"
                "<li><b>Time file / Data file</b>: the two raw signal columns for this sensor.</li>"
                "<li><b>Measurement uncertainty (%) / (absolute unit)</b>: draws this sensor's "
                "shaded uncertainty band on the Overview plot below and feeds any Monte-Carlo "
                "propagation that uses it.</li>"
                "<li><b>Multiplicative (calibration) factor</b>: converts the raw data column "
                "into calibrated physical units.</li>"
                "<li><b>Apply low-pass filter</b>: a zero-phase Butterworth low-pass filter "
                "(default cutoff 300 kHz) applied to the calibrated signal, to remove "
                "high-frequency noise -- reflected everywhere this sensor's data is used "
                "(Overview plot, uncertainty band, Monte-Carlo specimen-sensor cut, Results "
                "&amp; Export). An invalid or too-high cutoff (at or past this sensor's own "
                "Nyquist frequency) is simply left unfiltered.</li>"
                "<li><b>Cut curve</b>: see the plot's own help below for how to trim/re-zero "
                "the signal.</li>"
                "</ul>"
            ),
        )
        self.project = project
        self.sensor = sensor
        self.index = index
        self.on_change = on_change
        self.on_remove = on_remove

        if on_remove is not None:
            remove_row = QHBoxLayout()
            remove_row.addStretch(1)
            self.remove_button = QPushButton("Remove this sensor")
            self.remove_button.clicked.connect(lambda: self.on_remove(self.index))
            remove_row.addWidget(self.remove_button)
            self.add_layout(remove_row)

        self.enable_check = QCheckBox("Enable this sensor")
        self.enable_check.setChecked(sensor.enabled)
        self.enable_check.toggled.connect(self._on_enabled_toggled)
        self.add(self.enable_check)

        self.role_combo = QComboBox()
        for key, label in ROLES:
            self.role_combo.addItem(label, userData=key)
        self.role_combo.setCurrentIndex([k for k, _ in ROLES].index(sensor.role))
        self.role_combo.currentIndexChanged.connect(self._on_role_changed)
        self.add(LabeledField("Role", self.role_combo))

        self.on_specimen_check = QCheckBox("This sensor is mounted directly on the specimen")
        self.on_specimen_check.setChecked(sensor.on_specimen)
        self.on_specimen_check.toggled.connect(self._on_specimen_toggled)
        self.add(self.on_specimen_check)

        self.body_widget = Card("")  # simple container to show/hide together
        self.time_row = TimeSourceRow("Time file (.txt)")
        self.time_row.file_selected.connect(self._on_time_file)
        self.time_row.dtChanged.connect(self._on_time_step_changed)
        self.time_row.modeChanged.connect(self._on_time_mode_changed)
        self.body_widget.add(self.time_row)
        self.data_row = FileSelectRow("Data file (.txt)")
        self.data_row.file_selected.connect(self._on_data_file)
        self.body_widget.add(self.data_row)

        self.unc_percent_edit = QLineEdit(str(sensor.uncertainty_percent))
        self.unc_percent_edit.editingFinished.connect(self._on_settings_changed)
        self.body_widget.add(LabeledField("Measurement uncertainty (%)", self.unc_percent_edit))

        self.unc_absolute_edit = QLineEdit(str(sensor.uncertainty_absolute))
        self.unc_absolute_edit.editingFinished.connect(self._on_settings_changed)
        self.body_widget.add(LabeledField("Measurement uncertainty (absolute unit)", self.unc_absolute_edit))

        self.multi_edit = QLineEdit(str(sensor.multiplicative_factor))
        self.multi_edit.editingFinished.connect(self._on_settings_changed)
        self.body_widget.add(LabeledField("Multiplicative (calibration) factor", self.multi_edit))

        self.lowpass_check = QCheckBox("Apply low-pass filter")
        self.lowpass_check.setChecked(sensor.lowpass_enabled)
        self.lowpass_check.toggled.connect(self._on_lowpass_toggled)
        self.body_widget.add(self.lowpass_check)
        self.lowpass_freq_edit = QLineEdit(str(sensor.lowpass_cutoff_hz))
        self.lowpass_freq_edit.editingFinished.connect(self._on_settings_changed)
        self.body_widget.add(LabeledField("Low-pass cutoff frequency (Hz)", self.lowpass_freq_edit))

        thermal_card = Card(
            "Thermal Output Correction (time-varying)",
            "Optional. Corrects this sensor's own strain-gage thermal output using a "
            "supplied temperature-vs-time profile, evaluated at every sample.",
            help_html=(
                "<p>Optional. Subtracts the apparent strain this sensor's OWN strain gage "
                "would report from a temperature change alone (no mechanical load) -- the "
                "same \"thermal output\" effect as the CPG Calc tab's static correction, but "
                "evaluated at every sample of a temperature-vs-time profile instead of one "
                "fixed operating temperature, so it stays valid whether the temperature "
                "changes fast (e.g. adiabatic specimen heating during a dynamic test) or "
                "slowly (e.g. ambient temperature drift during a slower test).</p>"
                "<ul>"
                "<li><b>Temperature profile: Time file / Temperature file</b>: the "
                "temperature measurement this correction is evaluated against, as its own "
                "time+temperature columns (may use a different time base/sample rate than "
                "this sensor's own signal -- linearly interpolated onto it, held at the "
                "nearest known value outside its own covered span). This could be a "
                "measured ambient/specimen temperature log, or an adiabatic "
                "temperature-rise profile exported from a Processing tab.</li>"
                "<li><b>Polynomial degree / Coefficient a_i</b>: the manufacturer's "
                "thermal-output polynomial for this gage/specimen-material pairing, read "
                "off its package insert (same convention as the CPG Calc tab).</li>"
                "<li><b>Reference (balance) Temperature</b>: the temperature this sensor's "
                "bridge was zeroed at.</li>"
                "</ul>"
                "<p><b>Technical detail:</b> per Vishay Tech Note TN-504-1, the strain to "
                "subtract at each time sample is the thermal-output polynomial evaluated at "
                "that sample's interpolated temperature MINUS its value at the balance "
                "temperature. No thermal-lag/dynamics model is applied between the supplied "
                "profile and this correction -- the gage is assumed to track the supplied "
                "temperature directly, so the profile itself should already represent "
                "whatever temperature the gage actually sees.</p>"
            ),
        )
        self.thermal_check = QCheckBox("Apply thermal output correction")
        self.thermal_check.setChecked(sensor.thermal_correction_enabled)
        self.thermal_check.toggled.connect(self._on_thermal_toggled)
        thermal_card.add(self.thermal_check)

        self.thermal_time_row = FileSelectRow("Temperature profile: Time file (.txt)")
        self.thermal_time_row.file_selected.connect(self._on_thermal_time_file)
        thermal_card.add(self.thermal_time_row)
        self.thermal_temp_row = FileSelectRow("Temperature profile: Temperature file (.txt)")
        self.thermal_temp_row.file_selected.connect(self._on_thermal_temp_file)
        thermal_card.add(self.thermal_temp_row)

        thermal_top_row = QHBoxLayout()
        thermal_top_row.addWidget(QLabel("Temperature unit"))
        self.thermal_unit_combo = QComboBox()
        self.thermal_unit_combo.addItem("°C", "C")
        self.thermal_unit_combo.addItem("°F", "F")
        self.thermal_unit_combo.setCurrentIndex(0 if sensor.thermal_unit != "F" else 1)
        self.thermal_unit_combo.currentIndexChanged.connect(self._on_thermal_settings_changed)
        thermal_top_row.addWidget(self.thermal_unit_combo)
        thermal_top_row.addSpacing(20)
        thermal_top_row.addWidget(QLabel("Polynomial degree"))
        self.thermal_degree_combo = QComboBox()
        for d in range(THERMAL_MIN_DEGREE, THERMAL_MAX_DEGREE + 1):
            self.thermal_degree_combo.addItem(str(d), d)
        self.thermal_degree_combo.setCurrentIndex(
            max(0, min(THERMAL_MAX_DEGREE - THERMAL_MIN_DEGREE, sensor.thermal_degree - THERMAL_MIN_DEGREE))
        )
        self.thermal_degree_combo.currentIndexChanged.connect(self._on_thermal_degree_changed)
        thermal_top_row.addWidget(self.thermal_degree_combo)
        thermal_top_row.addStretch(1)
        thermal_card.add_layout(thermal_top_row)

        thermal_coeff_grid = QGridLayout()
        thermal_coeff_grid.setHorizontalSpacing(14)
        thermal_coeff_grid.setVerticalSpacing(8)
        self._thermal_coeff_labels: list[QLabel] = []
        self._thermal_coeff_edits: list[QLineEdit] = []
        for order in range(THERMAL_MAX_DEGREE + 1):
            lbl = QLabel(f"a{order} (order {order}, µε/unit^{order})")
            default = sensor.thermal_coeffs[order] if order < len(sensor.thermal_coeffs) else 0.0
            edit = QLineEdit(str(default))
            edit.editingFinished.connect(self._on_thermal_settings_changed)
            thermal_coeff_grid.addWidget(lbl, order // 2, 2 * (order % 2))
            thermal_coeff_grid.addWidget(edit, order // 2, 2 * (order % 2) + 1)
            self._thermal_coeff_labels.append(lbl)
            self._thermal_coeff_edits.append(edit)
        thermal_card.add_layout(thermal_coeff_grid)

        self.thermal_t_ref_edit = QLineEdit(str(sensor.thermal_t_ref))
        self.thermal_t_ref_edit.editingFinished.connect(self._on_thermal_settings_changed)
        thermal_card.add(LabeledField("Reference (balance) Temperature", self.thermal_t_ref_edit))

        self.thermal_status_label = QLabel("")
        self.thermal_status_label.setObjectName("cardSubtitle")
        thermal_card.add(self.thermal_status_label)

        self._thermal_widgets = (
            [self.thermal_time_row, self.thermal_temp_row, self.thermal_unit_combo,
             self.thermal_degree_combo, self.thermal_t_ref_edit] + self._thermal_coeff_edits
        )
        for w in self._thermal_widgets:
            w.setEnabled(False)
        self.body_widget.add(thermal_card)
        self._on_thermal_degree_changed()

        self.status_label = QLabel("")
        self.status_label.setObjectName("cardSubtitle")
        self.body_widget.add(self.status_label)

        self.picker = TimeWindowPicker(
            [("cut", "Cut point (drag start = new t=0)", theme.ACCENT)], figsize=(9, 2.6),
            help_title="Cut sensor signal",
            help_text=(
                "<p>Shows this sensor's loaded raw signal.</p>"
                "<ul>"
                "<li>Click and drag on the plot to set the \"Cut point\" window -- only where "
                "it STARTS matters, marking the new t=0.</li>"
                "<li>Click <b>Cut curve at selected start</b> below to discard everything before "
                "that point and re-zero time from there, or type an exact time and use <b>Cut "
                "curve at typed value</b> instead.</li>"
                "<li><b>Reset cut</b> restores the original, uncut signal.</li>"
                "</ul>"
            ),
        )
        self.body_widget.add(self.picker)

        cut_row = QHBoxLayout()
        self.cut_button = QPushButton("Cut curve at selected start")
        self.cut_button.clicked.connect(self._on_cut)
        cut_row.addWidget(self.cut_button)
        self.reset_cut_button = QPushButton("Reset cut")
        self.reset_cut_button.clicked.connect(self._on_reset_cut)
        cut_row.addWidget(self.reset_cut_button)
        cut_row.addStretch(1)
        self.body_widget.add_layout(cut_row)

        manual_cut_row = QHBoxLayout()
        self.manual_cut_edit = QLineEdit("0.0")
        manual_cut_row.addWidget(QLabel("or cut at typed time (s):"))
        manual_cut_row.addWidget(self.manual_cut_edit)
        self.manual_cut_button = QPushButton("Cut curve at typed value")
        self.manual_cut_button.clicked.connect(self._on_manual_cut)
        manual_cut_row.addWidget(self.manual_cut_button)
        manual_cut_row.addStretch(1)
        self.body_widget.add_layout(manual_cut_row)

        self.add(self.body_widget)
        self._sync_visibility()
        self._restore_from_session()

    def _restore_from_session(self):
        dt = self.project.time_step_overrides.get(f"sensor_{self.index}_time")
        if dt is not None:
            self.time_row.set_dt(dt)
            self.time_row.set_generate_mode(True)  # sensor.time was already restored directly by core/session.py; this just reflects that in the widget
        if len(self.sensor.time) > 0 and len(self.sensor.data) > 0:
            self._refresh_picker()
            self.status_label.setText(f"Restored {len(self.sensor.data)} samples from previous session.")
        if len(self.sensor.thermal_time) > 0 and len(self.sensor.thermal_temperature) > 0:
            self.thermal_status_label.setText(
                f"Restored {len(self.sensor.thermal_temperature)} temperature-profile "
                "samples from previous session."
            )

    def _sync_visibility(self):
        self.body_widget.setVisible(self.enable_check.isChecked())

    def _on_enabled_toggled(self, checked):
        self.sensor.enabled = checked
        self._sync_visibility()
        self.on_change()

    def _on_role_changed(self, _index):
        self.sensor.role = self.role_combo.currentData()
        self.on_change()

    def _on_specimen_toggled(self, checked):
        self.sensor.on_specimen = checked
        self.on_change()

    def _on_lowpass_toggled(self, checked):
        self.sensor.lowpass_enabled = checked
        # NOT self._refresh_picker() -- that picker plots the sensor's raw
        # (uncalibrated) signal for picking a cut point and is unaffected
        # by this filter; on_change() below refreshes the Overview plot,
        # which DOES use calibrated_data() and so reflects the filter.
        self.on_change()

    def _on_thermal_toggled(self, checked):
        self.sensor.thermal_correction_enabled = checked
        for w in self._thermal_widgets:
            w.setEnabled(checked)
        # NOT self._refresh_picker() -- same reasoning as _on_lowpass_toggled
        # above, that picker plots the raw signal for picking a cut point;
        # on_change() below refreshes the Overview plot, which DOES use
        # calibrated_data() and so reflects this correction.
        self.on_change()

    def _on_thermal_degree_changed(self, _index: int = 0):
        degree = self.thermal_degree_combo.currentData()
        if degree is None:
            degree = self.sensor.thermal_degree
        for order in range(THERMAL_MAX_DEGREE + 1):
            visible = order <= degree
            self._thermal_coeff_labels[order].setVisible(visible)
            self._thermal_coeff_edits[order].setVisible(visible)
        self.sensor.thermal_degree = degree
        self.on_change()

    def _on_thermal_settings_changed(self, _index=None):
        try:
            t_ref = float(self.thermal_t_ref_edit.text())
            coeffs = [float(e.text()) for e in self._thermal_coeff_edits]
        except ValueError:
            self.thermal_status_label.setText("Thermal output fields must be numeric.")
            return
        self.sensor.thermal_t_ref = t_ref
        self.sensor.thermal_coeffs = coeffs
        self.sensor.thermal_unit = self.thermal_unit_combo.currentData()
        self.thermal_status_label.setText("")
        self.on_change()

    def _on_thermal_time_file(self, path):
        try:
            values = self._read_column(path)
            self.sensor.thermal_time = values
        except (OSError, ValueError) as exc:
            self.thermal_status_label.setText(f"Could not read temperature-profile time file: {exc}")
            return
        self.project.last_paths[f"sensor_{self.index}_thermal_time"] = path
        self.thermal_status_label.setText(
            f"Loaded {len(self.sensor.thermal_time)} temperature-profile time samples."
        )
        self.on_change()

    def _on_thermal_temp_file(self, path):
        try:
            values = self._read_column(path)
            self.sensor.thermal_temperature = values
        except (OSError, ValueError) as exc:
            self.thermal_status_label.setText(f"Could not read temperature-profile file: {exc}")
            return
        self.project.last_paths[f"sensor_{self.index}_thermal_temperature"] = path
        self.thermal_status_label.setText(
            f"Loaded {len(self.sensor.thermal_temperature)} temperature-profile samples."
        )
        self.on_change()

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
            values = self._read_column(path)
            if self.project.downsample_enabled:
                values = downsample(values, self.project.downsample_n)
            self.sensor.time = values
            # Pristine snapshot, captured fresh at load time -- see
            # CustomSensor.time_raw's docstring. cut_from() never mutates
            # these arrays in place (it always rebinds self.time/self.data
            # to new slices), so this copy stays untouched by any cut
            # applied afterwards, including the Monte-Carlo per-draw cut.
            self.sensor.time_raw = values.copy()
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Could not read time file: {exc}")
            return
        self.sensor.cut_time = None
        self.project.last_paths[f"sensor_{self.index}_time"] = path
        self.status_label.setText(f"Loaded {len(self.sensor.time)} time samples.")
        self._refresh_picker()

    def _on_data_file(self, path):
        try:
            values = self._read_column(path)
            if self.project.downsample_enabled:
                values = downsample(values, self.project.downsample_n)
            self.sensor.data = values
            self.sensor.data_raw = values.copy()
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Could not read data file: {exc}")
            return
        self.sensor.cut_time = None
        self.project.last_paths[f"sensor_{self.index}_data"] = path
        self._maybe_regenerate_time()
        self.status_label.setText(f"Loaded {len(self.sensor.data)} data samples.")
        self._refresh_picker()
        self.on_change()

    def _on_time_step_changed(self, dt):
        """Alternative to `_on_time_file` above: build the time vector from
        a fixed time step matching the data file's length, instead of
        loading it from a file."""
        self._maybe_regenerate_time()
        if len(self.sensor.time) == 0:
            self.status_label.setText("Load the data file first, so the generated time vector can match its length.")
            return
        self.project.last_paths.pop(f"sensor_{self.index}_time", None)
        self.project.time_step_overrides[f"sensor_{self.index}_time"] = dt
        self.status_label.setText(f"Generated {len(self.sensor.time)} time samples (dt = {dt:.6g} s).")
        self._refresh_picker()

    def _on_time_mode_changed(self, generate):
        if not generate:
            self.project.time_step_overrides.pop(f"sensor_{self.index}_time", None)

    def _maybe_regenerate_time(self):
        if not self.time_row.is_generate_mode():
            return
        n = len(self.sensor.data)
        if n == 0:
            return
        generated = self.time_row.build(n)
        if generated is not None:
            self.sensor.time = generated
            self.sensor.time_raw = generated.copy()

    def _on_settings_changed(self):
        try:
            uncertainty_percent = float(self.unc_percent_edit.text())
            uncertainty_absolute = float(self.unc_absolute_edit.text())
            multiplicative_factor = float(self.multi_edit.text())
            lowpass_cutoff_hz = float(self.lowpass_freq_edit.text())
        except ValueError:
            self.status_label.setText("Uncertainty/factor/cutoff-frequency fields must be numeric.")
            return
        if lowpass_cutoff_hz <= 0:
            self.status_label.setText("Low-pass cutoff frequency must be positive.")
            return
        self.sensor.uncertainty_percent = uncertainty_percent
        self.sensor.uncertainty_absolute = uncertainty_absolute
        self.sensor.multiplicative_factor = multiplicative_factor
        self.sensor.lowpass_cutoff_hz = lowpass_cutoff_hz
        self.on_change()

    def _refresh_picker(self):
        if len(self.sensor.time) == 0 or len(self.sensor.data) == 0:
            self.picker.show_placeholder("Load both the time and data files to see the signal here.")
            return
        self.picker.set_signal(self.sensor.time, [(self.sensor.label, self.sensor.data, theme.ACCENT)])

    def _on_cut(self):
        window = self.picker.get_window("cut")
        if window is None:
            self.status_label.setText("Drag on the plot to pick a cut point first.")
            return
        self.sensor.cut_from(window[0])
        self.status_label.setText(f"Cut applied: kept data from t={window[0]:.6g} s onward, time re-zeroed.")
        self._refresh_picker()
        self.on_change()

    def _on_manual_cut(self):
        try:
            cut_time = float(self.manual_cut_edit.text())
        except ValueError:
            self.status_label.setText("Typed cut time must be numeric.")
            return
        self.sensor.cut_from(cut_time)
        self.status_label.setText(f"Cut applied: kept data from t={cut_time:.6g} s onward, time re-zeroed.")
        self._refresh_picker()
        self.on_change()

    def _on_reset_cut(self):
        """Re-loads the time/data files fresh from disk, discarding any
        cut(s) applied."""
        from pathlib import Path
        time_path = self.project.last_paths.get(f"sensor_{self.index}_time")
        data_path = self.project.last_paths.get(f"sensor_{self.index}_data")
        self.sensor.cut_time = None
        found_any = False
        if time_path and Path(time_path).exists():
            self._on_time_file(time_path)
            found_any = True
        if data_path and Path(data_path).exists():
            self._on_data_file(data_path)
            found_any = True
        if found_any:
            self.status_label.setText("Cut reset: reloaded the original files from disk.")
        else:
            self.status_label.setText("Nothing to reset (no remembered file to reload from).")


class SensorsPage(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project

        self.scroll, self.layout = make_scroll_page(
            "Custom Sensors",
            "Configure any number of auxiliary sensors and preview them together below.",
        )

        self.cards_container = QVBoxLayout()
        self.layout.addLayout(self.cards_container)

        add_row = QHBoxLayout()
        self.add_sensor_button = QPushButton("+ Add sensor")
        self.add_sensor_button.clicked.connect(self._on_add_sensor)
        add_row.addWidget(self.add_sensor_button)
        add_row.addStretch(1)
        self.layout.addLayout(add_row)

        overview = Card("Overview", "All enabled sensors, each on its own Y axis (right-hand side, offset "
                                     "per sensor) so very different amplitudes can all be seen clearly.")
        self.overview_canvas = PlotCanvas(
            figsize=(9, 4),
            help_title="Sensors overview",
            help_text=(
                "<p>Plots every enabled sensor's calibrated signal against time on the same "
                "graph, each on its own color-matched Y axis (right-hand side, offset per "
                "sensor) so sensors with very different amplitudes stay readable together.</p>"
                "<p>A shaded band around a curve shows that sensor's own measurement "
                "uncertainty, when a Measurement uncertainty (%) or (absolute unit) has been "
                "set for it above.</p>"
                "<p>Updates automatically as sensors are enabled/loaded/cut.</p>"
            ),
        )
        overview.add(self.overview_canvas)
        self.layout.addWidget(overview)
        self._overview_card = overview

        self.layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.scroll)

        self.cards = []
        self._rebuild_cards()
        self._refresh_overview()

    def _rebuild_cards(self):
        while self.cards_container.count() > 0:
            item = self.cards_container.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        self.cards = []
        for i, sensor in enumerate(self.project.custom_sensors):
            card = _SensorCard(self.project, sensor, i, self._refresh_overview, on_remove=self._on_remove_sensor)
            self.cards.append(card)
            self.cards_container.addWidget(card)

    def _on_add_sensor(self):
        from core.custom_sensor import CustomSensor
        n = len(self.project.custom_sensors) + 1
        self.project.custom_sensors.append(CustomSensor(f"Custom sensor {n}"))
        self._rebuild_cards()
        self._refresh_overview()

    def _on_remove_sensor(self, index: int):
        if 0 <= index < len(self.project.custom_sensors):
            del self.project.custom_sensors[index]
        # Keep at least one sensor slot so the "+" button always has
        # something sensible to extend from, and existing index-based
        # references (last_paths, etc.) for the REMAINING sensors still
        # make sense after the rebuild below (indices shift down by one).
        if not self.project.custom_sensors:
            from core.custom_sensor import CustomSensor
            self.project.custom_sensors.append(CustomSensor("Custom sensor 1"))
        self._rebuild_cards()
        self._refresh_overview()

    def showEvent(self, event):
        super().showEvent(event)
        self._rebuild_cards()  # picks up sensors/values changed elsewhere (e.g. a loaded Setup Profile)
        self._refresh_overview()

    def _refresh_overview(self):
        enabled = [s for s in self.project.custom_sensors if s.enabled and len(s.time) > 0 and len(s.data) > 0]
        self.overview_canvas.clear()
        colors = [theme.ACCENT, "#ff9f43", "#4cd97b", "#e356c9", "#ffd166", "#5ad1e6", "#ff6b6b", "#95e06c"]
        if not enabled:
            ax = self.overview_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No enabled sensors with loaded data yet.", ha="center", va="center",
                    color=theme.TEXT_MUTED, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self.overview_canvas.refresh()
            return

        fig = self.overview_canvas.figure
        base_ax = fig.add_subplot(111)
        handles, labels = [], []
        extra_axes = 0
        for i, sensor in enumerate(enabled):
            color = colors[i % len(colors)]
            data = sensor.calibrated_data()
            n = min(len(sensor.time), len(data))
            lower, upper = sensor.uncertainty_band()
            if i == 0:
                ax = base_ax
            else:
                ax = base_ax.twinx()
                # Stack extra right-hand axes further out so their tick
                # labels don't overlap each other.
                ax.spines["right"].set_position(("axes", 1 + 0.12 * extra_axes))
                extra_axes += 1
            line, = ax.plot(sensor.time[:n], data[:n], color=color, label=sensor.label, linewidth=1)
            if sensor.uncertainty_percent or sensor.uncertainty_absolute:
                ax.fill_between(sensor.time[:n], lower[:n], upper[:n], color=color, alpha=0.25)
            ax.set_ylabel(sensor.label, color=color)
            ax.tick_params(axis="y", labelcolor=color)
            handles.append(line)
            labels.append(sensor.label)

        base_ax.set_xlabel("time (s)")
        base_ax.legend(handles, labels, fontsize=8, loc="upper left")
        fig.tight_layout()
        self.overview_canvas.refresh()
