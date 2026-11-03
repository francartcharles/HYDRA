"""
Specimen page.

Builds one adaptive layout for the specimen fields instead of a separate
hand-built sub-layout per test-mode/geometry combination. The full set of
possible fields lives once in core.models.Specimen, and which subset is
shown/loaded/saved for the current test mode + geometry comes from
core.models.SPECIMEN_FIELD_ORDER, which matches the file-format validation
used when importing a specimen config file.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QComboBox, QLabel, QHBoxLayout, QPushButton, QFileDialog
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

from core.models import Specimen, SPECIMEN_FIELD_ORDER
from core.param_io import load_param_file, save_param_file
from ui.widgets import Card, MCParamTable, LabeledField, FileSelectRow, make_scroll_page

MATERIAL_FIELDS = ["rho", "e", "nu"]
THERMAL_FIELDS = ["xi", "cp"]

_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "geometry"

# Which dimensioned drawing to show for each (test_mode, geometry), matching
# the field names on the Parameters card below (Diameter, Length, H1/H2/H3,
# R1/R2/R3, ...).
GEOMETRY_IMAGES = {
    ("Compression", "Cylindrical"): "comp_cyl_plan.PNG",
    ("Compression", "Square/Rectangular"): "comp_cub_plan.PNG",
    ("Tension", "Axisymmetric"): "tens_axy_plan.PNG",
    ("Tension", "Flat"): "tens_flat_plan.PNG",
    ("Shear-Compression/Tension (Notched)", None): "SCS_plan.PNG",
    ("Shear", "Hat-shaped"): "shear_hat_plan.PNG",
    ("Shear", "Double shear"): "double_shear_plan.PNG",
    ("Shear", "Simple shear"): "simple_shear_plan.PNG",
}


class SpecimenPage(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project

        scroll, layout = make_scroll_page(
            "Specimen",
            "Geometry and material properties for the specimen. The fields shown/loaded "
            "below adapt to the test mode/geometry chosen on the Test Setup page.",
        )

        identity = Card(
            "Identity",
            help_html=(
                "<p>Basic identification and config file handling for this specimen.</p>"
                "<ul>"
                "<li><b>Specimen name</b>: a free-text label for this specimen.</li>"
                "<li><b>Specimen config file</b>: loads a previously-saved parameter set "
                "instead of typing it by hand -- only the fields relevant to the current test "
                "mode/geometry (Parameters card below) are read.</li>"
                "<li><b>Save as…</b>: saves the current Parameters card values to a new config "
                "file, for reuse on another test.</li>"
                "</ul>"
            ),
        )
        self.name_edit = QLineEdit(project.specimen.name)
        self.name_edit.editingFinished.connect(self._on_name_changed)
        identity.add(LabeledField("Specimen name", self.name_edit))

        config_row = QHBoxLayout()
        self.config_file_row = FileSelectRow("Specimen config file")
        self.config_file_row.file_selected.connect(self._on_config_file_selected)
        config_row.addWidget(self.config_file_row, 1)
        save_button = QPushButton("Save as…")
        save_button.clicked.connect(self._on_save_config)
        config_row.addWidget(save_button)
        identity.add_layout(config_row)
        self.config_status = QLabel("")
        self.config_status.setObjectName("cardSubtitle")
        identity.add(self.config_status)
        layout.addWidget(identity)

        # A single combined card for whatever the current mode/geometry needs
        # (geometry + material + thermal), rebuilt whenever mode/geometry changes.
        self.fields_card = Card(
            "Parameters",
            "This set of parameters -- and the order they're loaded/saved in -- "
            "depends on the test mode and geometry selected on the Test Setup page.",
            help_html=(
                "<p>The specimen's geometry, material and (where relevant) thermal properties, "
                "each as a Monte-Carlo parameter (mean and dispersion).</p>"
                "<ul>"
                "<li>Which fields appear -- and the order they're loaded/saved in -- depends on "
                "the test mode and geometry chosen on the Test Setup page (e.g. Diameter/Length "
                "for a cylindrical compression specimen, H1/H2/H3 for a notched shear-compression "
                "one).</li>"
                "<li>The \"Dimensioned drawing\" card below shows which physical dimension each "
                "field name refers to for the current geometry.</li>"
                "</ul>"
            ),
        )
        layout.addWidget(self.fields_card)

        self.drawing_card = Card("Dimensioned drawing", "Matches the parameter names above.")
        self.drawing_label = QLabel()
        self.drawing_label.setAlignment(Qt.AlignCenter)
        self.drawing_card.add(self.drawing_label)
        layout.addWidget(self.drawing_card)

        layout.addStretch(1)
        self._rebuild_fields()
        restored_path = project.last_paths.get("specimen_config", "")
        if restored_path:
            self.config_status.setText(f"Parameters restored from previous session ({restored_path.split('/')[-1]})")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def showEvent(self, event):
        super().showEvent(event)
        self._rebuild_fields()

    def _current_key(self) -> tuple[str, str | None]:
        mode = self.project.test_mode
        geometry = self.project.geometry if mode in ("Compression", "Tension", "Shear") else None
        return (mode, geometry)

    def _current_order(self) -> list[str]:
        key = self._current_key()
        order = SPECIMEN_FIELD_ORDER.get(key)
        if order is None:
            # Fall back to a reasonable default rather than crashing if a
            # mode/geometry combination isn't in the table yet.
            order = ["rho", "D", "L", "e", "nu", "xi", "cp"]
        return order

    def _rebuild_fields(self):
        while self.fields_card.body.count() > 0:
            item = self.fields_card.body.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        order = self._current_order()
        self.table = MCParamTable([self.project.specimen.params[key] for key in order])
        self.fields_card.add(self.table)
        self._refresh_drawing()

    def _refresh_drawing(self):
        filename = GEOMETRY_IMAGES.get(self._current_key())
        if filename is None:
            self.drawing_label.setText("No drawing available for this geometry.")
            self.drawing_label.setPixmap(QPixmap())
            return
        path = _ASSETS_DIR / filename
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.drawing_label.setText(f"Drawing file not found: {filename}")
            return
        self.drawing_label.setPixmap(pixmap.scaledToWidth(420, Qt.SmoothTransformation))

    def _on_name_changed(self):
        self.project.specimen.name = self.name_edit.text()

    def _on_config_file_selected(self, path: str):
        try:
            load_param_file(path, self.project.specimen.params, self._current_order())
        except (OSError, ValueError, SyntaxError) as exc:
            self.config_status.setText(f"Could not load config: {exc}")
            return
        self.project.last_paths["specimen_config"] = path
        self.table.refresh()
        self.config_status.setText(f"Loaded parameters from {path.split('/')[-1]}")

    def _on_save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save specimen configuration", "", "Text files (*.txt)")
        if not path:
            return
        try:
            save_param_file(path, self.project.specimen.params, self._current_order())
        except OSError as exc:
            self.config_status.setText(f"Could not save config: {exc}")
            return
        self.config_status.setText(f"Saved to {path.split('/')[-1]}")
