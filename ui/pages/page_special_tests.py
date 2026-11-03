"""
Special Tests page.

Holds parameters for non-standard test configurations that only a subset
of users need -- currently just the quasi-oedometric (confined
compression) test settings, moved out of Test Setup to keep that page
focused on the common case.
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QLineEdit, QComboBox, QLabel

from ui.widgets import Card, LabeledField, FileSelectRow, make_scroll_page


class SpecialTestsPage(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project

        scroll, layout = make_scroll_page(
            "Special Tests",
            "Parameters for non-standard test configurations.",
        )

        confinement = Card(
            "Confinement (Quasi-Oedometric Test)",
            "For confined compression tests using a confinement cell: the radial stress is derived "
            "from the cell's own strain-gage signal (assign a Custom Sensor's role to \"Confinement "
            "cell strain gage\" on the Custom Sensors page), combined with the calibration below.",
            help_html=(
                "<p>Configures a quasi-oedometric (confined compression) test, where the "
                "confinement cell's own strain-gage signal is converted into radial stress -- "
                "needs a Custom Sensor assigned to the \"Confinement cell strain gage\" role on "
                "the Custom Sensors page.</p>"
                "<ul>"
                "<li><b>This is a quasi-oedometric (confined) test</b>: enables the confinement "
                "calculations and their plots on the Results &amp; Export page.</li>"
                "<li><b>Cell model</b>: how the cell's orthoradial displacement is converted to "
                "radial stress -- \"Linear\" uses the C1/C2/Ev constants below, \"Nonlinear\" "
                "instead interpolates loaded numerical-simulation calibration curves.</li>"
                "<li><b>Reference strain</b>: the axial strain at which the second calibration "
                "curve pair (see below) was computed.</li>"
                "<li><b>alpha00 / alpha0^(l/2)</b>: radial-strain calibration constants, from "
                "the same simulation as the calibration curves.</li>"
                "<li><b>C1 / C2 / Cell modulus Ev</b> (linear model only): calibration "
                "constants for the low-strain/high-strain regimes and the cell's own stiffness.</li>"
                "<li><b>Cell diameter Dcell</b> and the four calibration-curve files (nonlinear "
                "model only): the numerically-simulated relationship between orthoradial "
                "displacement and radial stress, at the specimen's initial height and again at "
                "its height at the reference strain.</li>"
                "</ul>"
            ),
        )
        self.confine_check = QCheckBox("This is a quasi-oedometric (confined) test")
        self.confine_check.setChecked(project.confinement.enabled)
        self.confine_check.toggled.connect(self._on_confine_toggled)
        confinement.add(self.confine_check)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Linear cell model", "Nonlinear cell model (numerical simulation curves)"])
        self.model_combo.setCurrentIndex(1 if project.confinement.nonlinear_enabled else 0)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._model_row = LabeledField("Cell model", self.model_combo)
        confinement.add(self._model_row)

        self.ref_strain_edit = QLineEdit(str(project.confinement.ref_strain))
        self.ref_strain_edit.editingFinished.connect(self._on_confine_params_changed)
        self._ref_strain_row = LabeledField("Reference strain (-)", self.ref_strain_edit)
        confinement.add(self._ref_strain_row)

        # -- Radial/volumetric/equivalent strain calibration -----------
        self.alpha00_edit = QLineEdit(str(project.confinement.alpha00))
        self.alpha00_edit.editingFinished.connect(self._on_confine_params_changed)
        self._alpha00_row = LabeledField("alpha00 (radial strain calibration, from simulation)", self.alpha00_edit)
        confinement.add(self._alpha00_row)

        self.alpha0_l2_edit = QLineEdit(str(project.confinement.alpha0_l2))
        self.alpha0_l2_edit.editingFinished.connect(self._on_confine_params_changed)
        self._alpha0_l2_row = LabeledField("alpha0^(l/2) (radial strain calibration, from simulation)", self.alpha0_l2_edit)
        confinement.add(self._alpha0_l2_row)

        # -- Linear model fields --------------------------------------
        self.c1_edit = QLineEdit(str(project.confinement.C1))
        self.c1_edit.editingFinished.connect(self._on_confine_params_changed)
        self._c1_row = LabeledField("C1 (low-strain calibration constant)", self.c1_edit)
        confinement.add(self._c1_row)

        self.c2_edit = QLineEdit(str(project.confinement.C2))
        self.c2_edit.editingFinished.connect(self._on_confine_params_changed)
        self._c2_row = LabeledField("C2 (high-strain calibration constant)", self.c2_edit)
        confinement.add(self._c2_row)

        self.cell_modulus_edit = QLineEdit(str(project.confinement.cell_modulus))
        self.cell_modulus_edit.editingFinished.connect(self._on_confine_params_changed)
        self._cell_modulus_row = LabeledField("Cell modulus Ev (Pa)", self.cell_modulus_edit)
        confinement.add(self._cell_modulus_row)

        self._linear_rows = [self._c1_row, self._c2_row, self._cell_modulus_row]

        # -- Nonlinear model fields -------------------------------------
        self.cell_diameter_edit = QLineEdit(str(project.confinement.cell_diameter))
        self.cell_diameter_edit.editingFinished.connect(self._on_confine_params_changed)
        self._cell_diameter_row = LabeledField("Cell diameter Dcell (m)", self.cell_diameter_edit)
        confinement.add(self._cell_diameter_row)

        self._nonlinear_note = QLabel(
            "Load the two pairs of calibration curves below (typically produced by a numerical "
            "simulation of the confinement cell): orthoradial cell displacement vs. radial stress, "
            "at the specimen's initial height, and again at its height at the reference strain above."
        )
        self._nonlinear_note.setObjectName("cardSubtitle")
        self._nonlinear_note.setWordWrap(True)
        confinement.add(self._nonlinear_note)

        self.ortho_ini_row = FileSelectRow("Orthoradial displacement, initial (.txt)")
        self.ortho_ini_row.file_selected.connect(lambda p: self._on_calib_file("calib_ortho_disp_ini", p))
        confinement.add(self.ortho_ini_row)
        self.radial_ini_row = FileSelectRow("Radial stress, initial (.txt)")
        self.radial_ini_row.file_selected.connect(lambda p: self._on_calib_file("calib_radial_ini", p))
        confinement.add(self.radial_ini_row)
        self.ortho_esp_row = FileSelectRow("Orthoradial displacement, at reference strain (.txt)")
        self.ortho_esp_row.file_selected.connect(lambda p: self._on_calib_file("calib_ortho_disp_esp", p))
        confinement.add(self.ortho_esp_row)
        self.radial_esp_row = FileSelectRow("Radial stress, at reference strain (.txt)")
        self.radial_esp_row.file_selected.connect(lambda p: self._on_calib_file("calib_radial_esp", p))
        confinement.add(self.radial_esp_row)

        self._nonlinear_rows = [
            self._cell_diameter_row, self._nonlinear_note,
            self.ortho_ini_row, self.radial_ini_row, self.ortho_esp_row, self.radial_esp_row,
        ]

        self.calib_status_label = QLabel("")
        self.calib_status_label.setObjectName("cardSubtitle")
        confinement.add(self.calib_status_label)

        self._sync_confine_visibility()
        layout.addWidget(confinement)

        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._refresh_calib_status()
        self._restore_from_session()

    def _restore_from_session(self):
        # The calibration curve VALUES are already restored onto
        # project.confinement by core/session.py's generic dataclass
        # restore (ConfinementSettings' plain-list fields are picked up
        # automatically) -- this only re-reads from disk as a fallback if
        # a path is remembered but the value somehow wasn't.
        from pathlib import Path
        for field_name in ("calib_ortho_disp_ini", "calib_radial_ini", "calib_ortho_disp_esp", "calib_radial_esp"):
            path = self.project.last_paths.get(f"confinement_{field_name}")
            if path and Path(path).exists() and not getattr(self.project.confinement, field_name):
                self._on_calib_file(field_name, path)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_from_project()  # picks up values changed elsewhere (e.g. a loaded Setup Profile)

    def _refresh_from_project(self):
        c = self.project.confinement
        self.confine_check.setChecked(c.enabled)
        self.model_combo.setCurrentIndex(1 if c.nonlinear_enabled else 0)
        self.c1_edit.setText(str(c.C1))
        self.c2_edit.setText(str(c.C2))
        self.cell_modulus_edit.setText(str(c.cell_modulus))
        self.ref_strain_edit.setText(str(c.ref_strain))
        self.cell_diameter_edit.setText(str(c.cell_diameter))
        self.alpha00_edit.setText(str(c.alpha00))
        self.alpha0_l2_edit.setText(str(c.alpha0_l2))
        self._sync_confine_visibility()
        self._refresh_calib_status()

    def _sync_confine_visibility(self):
        checked = self.confine_check.isChecked()
        self._model_row.setVisible(checked)
        self._ref_strain_row.setVisible(checked)
        self._alpha00_row.setVisible(checked)
        self._alpha0_l2_row.setVisible(checked)
        nonlinear = checked and self.model_combo.currentIndex() == 1
        linear = checked and not nonlinear
        for row in self._linear_rows:
            row.setVisible(linear)
        for row in self._nonlinear_rows:
            row.setVisible(nonlinear)
        self.calib_status_label.setVisible(nonlinear)

    def _on_confine_toggled(self, checked):
        self.project.confinement.enabled = checked
        self._sync_confine_visibility()

    def _on_model_changed(self, index):
        self.project.confinement.nonlinear_enabled = index == 1
        self._sync_confine_visibility()

    def _on_confine_params_changed(self):
        try:
            self.project.confinement.C1 = float(self.c1_edit.text())
            self.project.confinement.C2 = float(self.c2_edit.text())
            self.project.confinement.cell_modulus = float(self.cell_modulus_edit.text())
            self.project.confinement.ref_strain = float(self.ref_strain_edit.text())
            self.project.confinement.cell_diameter = float(self.cell_diameter_edit.text())
            self.project.confinement.alpha00 = float(self.alpha00_edit.text())
            self.project.confinement.alpha0_l2 = float(self.alpha0_l2_edit.text())
        except ValueError:
            pass  # leave previous values in place; the field still shows what the user typed

    def _read_column(self, path):
        values = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    values.append(float(line))
        return values

    def _on_calib_file(self, field_name: str, path: str):
        try:
            values = self._read_column(path)
        except (OSError, ValueError) as exc:
            self.calib_status_label.setText(f"Could not read {field_name}: {exc}")
            return
        setattr(self.project.confinement, field_name, values)
        self.project.last_paths[f"confinement_{field_name}"] = path
        self._refresh_calib_status()

    def _refresh_calib_status(self):
        c = self.project.confinement
        loaded = [
            name for name, values in [
                ("initial orthoradial displacement", c.calib_ortho_disp_ini),
                ("initial radial stress", c.calib_radial_ini),
                ("reference-strain orthoradial displacement", c.calib_ortho_disp_esp),
                ("reference-strain radial stress", c.calib_radial_esp),
            ] if values
        ]
        if loaded:
            self.calib_status_label.setText(f"Loaded: {', '.join(loaded)} ({len(loaded)}/4).")
        else:
            self.calib_status_label.setText("No calibration curves loaded yet.")
