"""
Strain Gage Bridge Coefficient ("CPG") calculator page.

Standalone tool (not part of the main per-test pipeline), ported from
test_type.py's `CPG_comp`/`CPG_comp_value` popup: computes the coefficient
that converts a bar's raw strain-gage bridge voltage into strain
(`strain = CPG * V_bridge`), from the bridge wiring type and its physical
inputs (gage factor, excitation voltage, amplifier gain, temperature...).

The result can be synced directly into the Input Bar's or Output Bar's own
"Strain Gage Bridge Coefficient" Monte-Carlo parameter (`SG_coeff` in
`Bar.PARAM_ORDER`) -- the respective Bar page picks up the new value the
next time it's shown (its `showEvent` already calls `table.refresh()`).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QToolButton, QButtonGroup, QComboBox, QCheckBox,
)

from core.cpg_calc import (
    compute_cpg, BridgeType, BRIDGE_LABELS, USES_POISSON_RATIO,
    transverse_sensitivity_correction, thermal_output_strain,
    TRANSVERSE_NU0_DEFAULT,
)
from core.mc import Distribution
from ui.widgets import Card, LabeledField, make_scroll_page

_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "bridges"
_BRIDGE_IMAGES = {
    BridgeType.QUARTER: "quarter_bridge.png",
    BridgeType.HALF: "half_bridge.png",
    BridgeType.FULL: "full_bridge.png",
}

# (key, label, default mean, default std)
_FIELD_ROWS = [
    ("gf", "Gage Factor (-)", "2.1", "0.02"),
    ("vex", "Input Tension (V)", "5", "0.05"),
    ("gain", "Gain (-)", "1", "0"),
    ("nub", "Bar Poisson Ratio (-)", "0.29", "0.01"),
    ("temperature", "Temperature (K)", "293", "1"),
]

# Thermal-output polynomial: order 0 .. degree, degree chosen from this
# range (matches the printed range of the manufacturer's own thermal-
# output coefficient tables, e.g. Micro-Measurements/Vishay package
# inserts -- see Tech Note TN-504-1).
_THERMAL_MIN_DEGREE = 3
_THERMAL_MAX_DEGREE = 7
_THERMAL_DEFAULT_DEGREE = 4


class CPGCalcPage(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._bridge_type = BridgeType.QUARTER
        self._last_result = None

        scroll, layout = make_scroll_page(
            "Strain Gage Bridge Coefficient",
            "Compute the coefficient that converts a bar's raw strain-gage bridge voltage into strain "
            "(strain = coefficient x V_bridge), from the Wheatstone bridge wiring and its physical "
            "inputs. The result can be synced directly into the Input Bar's or Output Bar's own "
            "Strain Gage Bridge Coefficient parameter.",
        )

        # --- bridge type selector -----------------------------------------
        selector_card = Card(
            "Gage Bridge Type",
            "Select the Wheatstone bridge wiring used for this measurement.",
            help_html=(
                "<p>Selects the Wheatstone bridge wiring used for this bar's strain-gage "
                "measurement -- Quarter, Half or Full bridge.</p>"
                "<p><b>Technical detail:</b> the bridge type sets the formula used to convert "
                "bridge voltage into strain, and determines whether the bar's Poisson ratio "
                "enters that formula (a Half bridge typically does; the \"Bar Poisson Ratio\" "
                "row below shows/hides itself accordingly).</p>"
            ),
        )
        button_row = QHBoxLayout()
        button_row.setSpacing(14)
        self.bridge_group = QButtonGroup(self)
        self.bridge_group.setExclusive(True)
        self._bridge_buttons = {}
        for bt in (BridgeType.QUARTER, BridgeType.HALF, BridgeType.FULL):
            btn = QToolButton()
            btn.setObjectName("bridgeButton")
            btn.setCheckable(True)
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            image_path = _ASSETS_DIR / _BRIDGE_IMAGES[bt]
            if image_path.exists():
                btn.setIcon(QIcon(str(image_path)))
            btn.setIconSize(QSize(120, 120))
            btn.setText(BRIDGE_LABELS[bt])
            btn.setMinimumSize(160, 180)
            btn.clicked.connect(lambda _checked, bt=bt: self._on_bridge_selected(bt))
            self.bridge_group.addButton(btn)
            button_row.addWidget(btn)
            self._bridge_buttons[bt] = btn
        self._bridge_buttons[BridgeType.QUARTER].setChecked(True)
        button_row.addStretch(1)
        selector_card.add_layout(button_row)
        layout.addWidget(selector_card)

        # --- physical inputs -------------------------------------------------
        inputs_card = Card(
            "Bridge Inputs",
            help_html=(
                "<p>Physical inputs to the bridge-coefficient calculation, each as a "
                "Monte-Carlo parameter (Mean and Standard Deviation).</p>"
                "<ul>"
                "<li><b>Gage Factor</b>: the strain gage's own sensitivity constant.</li>"
                "<li><b>Input Tension</b>: the bridge excitation voltage.</li>"
                "<li><b>Gain</b>: the amplifier gain applied to the bridge output.</li>"
                "<li><b>Bar Poisson Ratio</b>: only shown/used for bridge types whose wiring "
                "depends on it (see the Gage Bridge Type card's own help).</li>"
                "<li><b>Temperature</b>: the gage's operating temperature.</li>"
                "<li><b>Temperature Coefficient of Gage Factor</b>: how much the gage factor "
                "itself drifts with temperature (%/K) -- leave at 0 to ignore this effect.</li>"
                "</ul>"
            ),
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        for col, text in enumerate(["Parameter", "Mean", "Standard Deviation"]):
            lbl = QLabel(text)
            lbl.setObjectName("cardSubtitle")
            grid.addWidget(lbl, 0, col)

        self._edits = {}
        self._nub_row_widgets = []
        for row, (key, label, mean_default, std_default) in enumerate(_FIELD_ROWS, start=1):
            name_label = QLabel(label)
            grid.addWidget(name_label, row, 0)
            mean_edit = QLineEdit(mean_default)
            grid.addWidget(mean_edit, row, 1)
            std_edit = QLineEdit(std_default)
            grid.addWidget(std_edit, row, 2)
            self._edits[key] = (mean_edit, std_edit)
            if key == "nub":
                self._nub_row_widgets = [name_label, mean_edit, std_edit]
        inputs_card.add_layout(grid)

        self.sg_t_edit = QLineEdit("0.0")
        inputs_card.add(LabeledField(
            "Temperature Coefficient of Gage Factor (%/K)", self.sg_t_edit,
        ))
        layout.addWidget(inputs_card)

        # --- transverse sensitivity correction (optional) --------------------
        transverse_card = Card(
            "Transverse Sensitivity Correction",
            help_html=(
                "<p>Optional. Corrects the coefficient for the strain gage's own transverse "
                "sensitivity Kt -- its datasheet-specified response to strain PERPENDICULAR to "
                "its measuring axis, which is not the direction of interest and would otherwise "
                "bias every strain derived from this coefficient.</p>"
                "<ul>"
                "<li><b>Transverse Sensitivity Kt</b>: from the gage's datasheet (e.g. the "
                "\"Transverse Sensitivity\" row on a Micro-Measurements/Vishay package insert), "
                "as a percentage.</li>"
                "<li><b>Calibration Poisson Ratio</b>: the Poisson ratio the manufacturer used to "
                "calibrate the gage factor (0.285 for Micro-Measurements/Vishay gages, Tech Note "
                "TN-509) -- a fixed reference constant, not something to measure.</li>"
                "<li>Reuses this page's own <b>Bar Poisson Ratio</b> (above) as the actual Poisson "
                "ratio of the bar under test, shown automatically while this correction is "
                "enabled even for a Quarter-Bridge (which otherwise hides that row).</li>"
                "</ul>"
                "<p><b>Technical detail:</b> derived (Vishay Tech Note TN-509) for a single axial "
                "gage under a uniaxial stress field -- this app's usual case -- as "
                "strain_true = strain_indicated &times; (1 - nu0&middot;Kt) / (1 - nu&middot;Kt). "
                "Applied here as a scale factor on the coefficient itself (since strain = "
                "coefficient &times; V_bridge, scaling the coefficient scales every derived "
                "strain identically), uniformly regardless of bridge type.</p>"
            ),
        )
        self.transverse_checkbox = QCheckBox("Apply transverse sensitivity correction")
        self.transverse_checkbox.toggled.connect(self._on_transverse_toggled)
        transverse_card.add(self.transverse_checkbox)

        transverse_grid = QGridLayout()
        transverse_grid.setHorizontalSpacing(14)
        transverse_grid.setVerticalSpacing(10)
        for col, text in enumerate(["Parameter", "Mean", "Standard Deviation"]):
            lbl = QLabel(text)
            lbl.setObjectName("cardSubtitle")
            transverse_grid.addWidget(lbl, 0, col)
        transverse_grid.addWidget(QLabel("Transverse Sensitivity Kt (%)"), 1, 0)
        self.kt_mean_edit = QLineEdit("0.8")
        transverse_grid.addWidget(self.kt_mean_edit, 1, 1)
        self.kt_std_edit = QLineEdit("0.2")
        transverse_grid.addWidget(self.kt_std_edit, 1, 2)
        transverse_grid.addWidget(QLabel("Calibration Poisson Ratio nu0 (-)"), 2, 0)
        self.nu0_edit = QLineEdit(str(TRANSVERSE_NU0_DEFAULT))
        transverse_grid.addWidget(self.nu0_edit, 2, 1)
        transverse_card.add_layout(transverse_grid)
        self._transverse_widgets = [self.kt_mean_edit, self.kt_std_edit, self.nu0_edit]
        for w in self._transverse_widgets:
            w.setEnabled(False)
        layout.addWidget(transverse_card)

        # --- thermal output correction (optional) -----------------------------
        thermal_card = Card(
            "Thermal Output Correction",
            help_html=(
                "<p>Optional. Corrects for the strain gage's own \"thermal output\" -- the "
                "apparent strain it reports from temperature change alone, with no mechanical "
                "load, caused by the mismatch between the gage grid's and the specimen material's "
                "thermal expansion. This is a SEPARATE effect from the Temperature Coefficient of "
                "Gage Factor above (which corrects the gage's sensitivity, not its zero point).</p>"
                "<ul>"
                "<li><b>Polynomial degree</b>: how many of the manufacturer's thermal-output "
                "coefficients to use (their package insert lists one coefficient per order, "
                "typically 3rd to 7th order).</li>"
                "<li><b>Coefficient a_i</b>: the order-i coefficient, in microstrain per "
                "(temperature unit)^i, read directly off the gage's own package insert -- printed "
                "for the specific specimen material the gage is matched to (e.g. steel).</li>"
                "<li><b>Reference (balance) / Operating Temperature</b>: the temperature the "
                "bridge was zeroed at, and the temperature during the actual test.</li>"
                "</ul>"
                "<p><b>Technical detail:</b> per Vishay Tech Note TN-504-1, the apparent strain "
                "to correct for is the thermal-output polynomial evaluated at the operating "
                "temperature MINUS its value at the balance temperature -- not the operating-"
                "temperature value alone, since any thermal output already present when the "
                "bridge was zeroed is already cancelled by that zeroing. This is a strain-domain "
                "offset, not a scale factor, so it is NOT folded into the synced coefficient -- "
                "it is reported here (as microstrain, and as the equivalent bridge-voltage offset "
                "using the coefficient above) for you to apply manually, e.g. if the bridge was "
                "zeroed at a different temperature than the test itself.</p>"
            ),
        )
        self.thermal_checkbox = QCheckBox("Apply thermal output correction")
        self.thermal_checkbox.toggled.connect(self._on_thermal_toggled)
        thermal_card.add(self.thermal_checkbox)

        thermal_top_row = QHBoxLayout()
        thermal_top_row.addWidget(QLabel("Temperature unit"))
        self.thermal_unit_combo = QComboBox()
        self.thermal_unit_combo.addItems(["°C", "°F"])
        thermal_top_row.addWidget(self.thermal_unit_combo)
        thermal_top_row.addSpacing(20)
        thermal_top_row.addWidget(QLabel("Polynomial degree"))
        self.thermal_degree_combo = QComboBox()
        for d in range(_THERMAL_MIN_DEGREE, _THERMAL_MAX_DEGREE + 1):
            self.thermal_degree_combo.addItem(str(d), d)
        self.thermal_degree_combo.setCurrentIndex(
            _THERMAL_DEFAULT_DEGREE - _THERMAL_MIN_DEGREE
        )
        self.thermal_degree_combo.currentIndexChanged.connect(self._on_thermal_degree_changed)
        thermal_top_row.addWidget(self.thermal_degree_combo)
        thermal_top_row.addStretch(1)
        thermal_card.add_layout(thermal_top_row)

        coeff_grid = QGridLayout()
        coeff_grid.setHorizontalSpacing(14)
        coeff_grid.setVerticalSpacing(8)
        self._thermal_coeff_labels: list[QLabel] = []
        self._thermal_coeff_edits: list[QLineEdit] = []
        for order in range(_THERMAL_MAX_DEGREE + 1):
            lbl = QLabel(f"a{order} (order {order}, με/unit^{order})")
            edit = QLineEdit("0")
            coeff_grid.addWidget(lbl, order // 2, 2 * (order % 2))
            coeff_grid.addWidget(edit, order // 2, 2 * (order % 2) + 1)
            self._thermal_coeff_labels.append(lbl)
            self._thermal_coeff_edits.append(edit)
        thermal_card.add_layout(coeff_grid)

        temp_row = QGridLayout()
        temp_row.setHorizontalSpacing(14)
        temp_row.addWidget(QLabel("Reference (balance) Temperature"), 0, 0)
        self.thermal_t_ref_edit = QLineEdit("24")
        temp_row.addWidget(self.thermal_t_ref_edit, 0, 1)
        temp_row.addWidget(QLabel("Operating Temperature"), 0, 2)
        self.thermal_t_op_edit = QLineEdit("24")
        temp_row.addWidget(self.thermal_t_op_edit, 0, 3)
        thermal_card.add_layout(temp_row)

        self._thermal_widgets = (
            [self.thermal_unit_combo, self.thermal_degree_combo,
             self.thermal_t_ref_edit, self.thermal_t_op_edit]
            + self._thermal_coeff_edits
        )
        for w in self._thermal_widgets:
            w.setEnabled(False)
        layout.addWidget(thermal_card)
        self._on_thermal_degree_changed()

        # --- compute + sync -------------------------------------------------
        result_card = Card(
            "Result",
            help_html=(
                "<p><b>Compute</b> applies the Wheatstone bridge formula to the inputs above and "
                "reports the resulting coefficient (with its propagated uncertainty), such that "
                "strain = coefficient &times; V_bridge.</p>"
                "<ul>"
                "<li><b>Target bar</b>: which bar (Input or Output) the result should be synced "
                "to.</li>"
                "<li><b>Sync to bar's Strain Gage Bridge Coefficient</b>: writes the computed "
                "mean/standard-deviation directly into that bar's own \"Strain Gage Bridge "
                "Coefficient\" (SG_coeff) Monte-Carlo parameter -- open that bar's page "
                "afterward to see it reflected there.</li>"
                "</ul>"
            ),
        )
        compute_row = QHBoxLayout()
        self.compute_button = QPushButton("Compute")
        self.compute_button.setObjectName("primary")
        self.compute_button.clicked.connect(self._on_compute)
        compute_row.addWidget(self.compute_button)
        compute_row.addStretch(1)
        result_card.add_layout(compute_row)

        self.result_label = QLabel("")
        self.result_label.setObjectName("cardSubtitle")
        self.result_label.setWordWrap(True)
        result_card.add(self.result_label)

        sync_row = QHBoxLayout()
        sync_row.addWidget(QLabel("Target bar"))
        self.target_bar_combo = QComboBox()
        self.target_bar_combo.addItems(["Input Bar", "Output Bar"])
        sync_row.addWidget(self.target_bar_combo)
        self.sync_button = QPushButton("Sync to bar's Strain Gage Bridge Coefficient")
        self.sync_button.setEnabled(False)
        self.sync_button.clicked.connect(self._on_sync)
        sync_row.addWidget(self.sync_button)
        sync_row.addStretch(1)
        result_card.add_layout(sync_row)

        self.sync_status = QLabel("")
        self.sync_status.setObjectName("cardSubtitle")
        self.sync_status.setWordWrap(True)
        result_card.add(self.sync_status)

        layout.addWidget(result_card)
        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._on_bridge_selected(BridgeType.QUARTER)

    def _on_bridge_selected(self, bridge_type: BridgeType):
        self._bridge_type = bridge_type
        self._refresh_nub_visibility()

    def _refresh_nub_visibility(self):
        # Shown whenever the bridge wiring itself needs it (half-/full-
        # bridge), OR the transverse sensitivity correction is enabled --
        # that correction reuses this same "Bar Poisson Ratio" field
        # regardless of bridge type (see transverse_card's help text), so a
        # Quarter-Bridge selection must not hide the field it needs.
        needs_nub = USES_POISSON_RATIO[self._bridge_type] or self.transverse_checkbox.isChecked()
        for w in self._nub_row_widgets:
            w.setVisible(needs_nub)

    def _on_transverse_toggled(self, checked: bool):
        for w in self._transverse_widgets:
            w.setEnabled(checked)
        self._refresh_nub_visibility()

    def _on_thermal_toggled(self, checked: bool):
        for w in self._thermal_widgets:
            w.setEnabled(checked)
        if checked:
            self._on_thermal_degree_changed()

    def _on_thermal_degree_changed(self, _index: int = 0):
        degree = self.thermal_degree_combo.currentData()
        if degree is None:
            degree = _THERMAL_DEFAULT_DEGREE
        for order in range(_THERMAL_MAX_DEGREE + 1):
            visible = order <= degree
            self._thermal_coeff_labels[order].setVisible(visible)
            self._thermal_coeff_edits[order].setVisible(visible)

    def _read_float(self, edit: QLineEdit) -> float:
        return float(edit.text())

    def _on_compute(self):
        try:
            gf, gf_std = (self._read_float(e) for e in self._edits["gf"])
            vex, vex_std = (self._read_float(e) for e in self._edits["vex"])
            gain, gain_std = (self._read_float(e) for e in self._edits["gain"])
            nub, nub_std = (self._read_float(e) for e in self._edits["nub"])
            temperature, temperature_std = (self._read_float(e) for e in self._edits["temperature"])
            sg_t_pct = self._read_float(self.sg_t_edit)
        except ValueError:
            self.result_label.setText("All fields must be numeric.")
            self.sync_button.setEnabled(False)
            self._last_result = None
            return

        try:
            result = compute_cpg(
                self._bridge_type,
                gf, gf_std, vex, vex_std, gain, gain_std,
                temperature, temperature_std, sg_t_pct,
                nub=nub, nub_std=nub_std,
            )
        except ValueError as exc:
            self.result_label.setText(f"Could not compute: {exc}")
            self.sync_button.setEnabled(False)
            self._last_result = None
            return

        result_lines = [f"Coefficient = {result.value:.6g} ± {result.std:.6g} (1/V)"]

        if self.transverse_checkbox.isChecked():
            try:
                kt, kt_std = self._read_float(self.kt_mean_edit), self._read_float(self.kt_std_edit)
                nu0 = self._read_float(self.nu0_edit)
            except ValueError:
                self.result_label.setText("Transverse sensitivity fields must be numeric.")
                self.sync_button.setEnabled(False)
                self._last_result = None
                return
            try:
                result, factor = transverse_sensitivity_correction(
                    result, kt, kt_std, nub, nub_std, nu0=nu0,
                )
            except ValueError as exc:
                self.result_label.setText(f"Could not apply transverse sensitivity correction: {exc}")
                self.sync_button.setEnabled(False)
                self._last_result = None
                return
            result_lines.append(
                f"After transverse sensitivity correction (factor = {factor:.6g}): "
                f"Coefficient = {result.value:.6g} ± {result.std:.6g} (1/V)"
            )

        if self.thermal_checkbox.isChecked():
            try:
                degree = self.thermal_degree_combo.currentData()
                coeffs = [
                    self._read_float(self._thermal_coeff_edits[order])
                    for order in range(degree + 1)
                ]
                t_ref = self._read_float(self.thermal_t_ref_edit)
                t_op = self._read_float(self.thermal_t_op_edit)
            except ValueError:
                self.result_label.setText("Thermal output fields must be numeric.")
                self.sync_button.setEnabled(False)
                self._last_result = None
                return
            apparent_strain_ue = thermal_output_strain(coeffs, t_ref, t_op)
            unit = self.thermal_unit_combo.currentText()
            line = (
                f"Thermal output (apparent strain) from {t_ref:g}{unit} to {t_op:g}{unit}: "
                f"{apparent_strain_ue:.6g} με"
            )
            if result.value != 0:
                equivalent_offset_v = (apparent_strain_ue * 1e-6) / result.value
                line += f" (≈ {equivalent_offset_v:.6g} V equivalent bridge offset)"
            line += " -- not folded into the coefficient below; apply this as a strain offset."
            result_lines.append(line)

        self._last_result = result
        self.result_label.setText("\n".join(result_lines))
        self.sync_button.setEnabled(True)
        self.sync_status.setText("")

    def _on_sync(self):
        if self._last_result is None:
            return
        is_input = self.target_bar_combo.currentIndex() == 0
        bar = self.project.input_bar if is_input else self.project.output_bar
        label = "Input Bar" if is_input else "Output Bar"

        sg_coeff = bar.params["SG_coeff"]
        sg_coeff.mean = self._last_result.value
        sg_coeff.dispersion = self._last_result.std
        sg_coeff.distribution = Distribution.NORMAL

        self.sync_status.setText(
            f"Synced to the {label}'s Strain Gage Bridge Coefficient "
            f"(mean={sg_coeff.mean:.6g}, dispersion={sg_coeff.dispersion:.6g}). "
            f"Open the {label} page to see it reflected there."
        )
