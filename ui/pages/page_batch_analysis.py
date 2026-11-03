"""
Batch Analysis page.

Standalone tool (not part of the main per-test pipeline), ported from
analysis_resume.py / resume_condition.py: combines several already-
processed tests into one statistical summary across tests (mean curve,
plus Experimental/Material/Total uncertainty bands).

Each test is one plain CSV/TXT file, exactly as produced by the Results &
Export page's "Export as .csv…"/"Export as .txt…" (or "Export full
test…") buttons -- every computed quantity as a column, headed by its
internal name. The X and Y column pickers below list every column that
export happens to contain (read from the first selected file's own
header row), so this isn't limited to true stress vs. true strain -- e.g.
picking confinement_pressure for X and confinement_deviator for Y
combines several confined tests' stress paths into one pressure-deviator
summary curve. A `<Y column>_std` column, when present, is picked up
automatically as that test's own Experimental uncertainty on Y -- no
separate picker for it.
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QHBoxLayout, QFileDialog, QCheckBox, QLineEdit,
)

from ui.widgets import Card, LabeledField, make_scroll_page
from ui.plot_canvas import PlotCanvas
from core.batch_analysis import (
    run_batch_analysis_from_files, export_batch_result, load_export_header,
    IsothermalSettings,
)

# Three well-separated categorical hues (dark-theme steps, colorblind-safe
# adjacent AND all-pairs) for Total / Experimental / Material -- kept
# consistent between both plot views so the same quantity always reads as
# the same color.
_TOTAL_COLOR = "#3987e5"         # blue
_EXPERIMENTAL_COLOR = "#d95926"  # orange
_MATERIAL_COLOR = "#199e70"      # aqua


class BatchAnalysisPage(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._batch_result = None
        self._batch_files: list[str] = []

        scroll, layout = make_scroll_page(
            "Batch Analysis",
            "Combine several tests -- each a CSV/TXT file exported with \"Export as .csv…\"/\"Export as "
            ".txt…\" on the Results & Export page -- into one statistical summary against a common X "
            "axis: a mean Y curve plus Experimental (each test's own Monte-Carlo uncertainty), Material "
            "(test-to-test scatter) and Total variability bands. X and Y can be any exported quantity, "
            "not just strain/stress -- e.g. confinement_pressure vs. confinement_deviator to summarize "
            "a confined test's stress path across several tests.",
        )

        batch_card = Card(
            "Batch Analysis",
            help_html=(
                "<p>Combines several already-processed tests (each exported as a CSV/TXT file "
                "from the Results &amp; Export page) into one statistical summary curve against a "
                "common X axis.</p>"
                "<ul>"
                "<li><b>Select files…</b>: choose every test's exported file to combine.</li>"
                "<li><b>X/Y axis column</b>: pick which exported quantities to combine -- any "
                "column works, not just strain/stress (e.g. confinement_pressure vs. "
                "confinement_deviator).</li>"
                "<li><b>Isothermal correction</b>: optional, only meaningful when Y is a stress "
                "and X is a strain -- removes adiabatic heating using the Taylor-Quinney "
                "coefficient, specific heat, density and temperature sensitivity entered below.</li>"
                "<li><b>Fixed Y variability</b>: optional override that replaces the computed "
                "test-to-test (Material) scatter with a flat percentage of the mean, useful with "
                "very few tests where the real scatter isn't statistically meaningful.</li>"
                "<li>Click <b>Run Batch Analysis</b>, then switch the <b>View</b> dropdown below "
                "and use <b>Export batch result…</b> to save the combined curve and bands.</li>"
                "</ul>"
                "<p><b>Technical detail:</b> each test's own X axis is linearly interpolated onto "
                "a shared grid before averaging; if a test's X column isn't monotonically "
                "increasing (e.g. a stress path through unload/reload), tests are instead combined "
                "by their own recorded sample order -- the status line after running reports which "
                "method was used.</p>"
            ),
        )
        files_row = QHBoxLayout()
        self.batch_files_edit = QLineEdit()
        self.batch_files_edit.setReadOnly(True)
        self.batch_files_edit.setPlaceholderText("No files selected")
        files_row.addWidget(QLabel("Exported test files"))
        files_row.addWidget(self.batch_files_edit, 1)
        files_browse_button = QPushButton("Select files…")
        files_browse_button.clicked.connect(self._on_browse_batch_files)
        files_row.addWidget(files_browse_button)
        batch_card.add_layout(files_row)

        self.batch_columns_status = QLabel("")
        self.batch_columns_status.setObjectName("cardSubtitle")
        self.batch_columns_status.setWordWrap(True)
        batch_card.add(self.batch_columns_status)

        self.batch_x_combo = QComboBox()
        self.batch_x_combo.setEnabled(False)
        batch_card.add(LabeledField("X axis column", self.batch_x_combo))
        self.batch_y_combo = QComboBox()
        self.batch_y_combo.setEnabled(False)
        self.batch_y_combo.currentIndexChanged.connect(self._refresh_std_status)
        batch_card.add(LabeledField("Y axis column", self.batch_y_combo))

        self.batch_std_status = QLabel("")
        self.batch_std_status.setObjectName("cardSubtitle")
        self.batch_std_status.setWordWrap(True)
        batch_card.add(self.batch_std_status)

        self.batch_iso_check = QCheckBox(
            "Apply isothermal correction (remove adiabatic heating) before combining -- only meaningful "
            "when Y is a stress and X is a strain"
        )
        self.batch_iso_check.toggled.connect(self._on_batch_iso_toggled)
        batch_card.add(self.batch_iso_check)
        self.batch_iso_fields = {}
        self._iso_rows = []
        for key, label, default in [("xi", "Taylor-Quinney coefficient (-)", "0.9"),
                                     ("cp", "Specific heat (J/(kg*K))", "450"),
                                     ("rho", "Density (kg/m^3)", "7800"),
                                     ("alpha", "Temperature sensitivity (MPa/K)", "0.0")]:
            edit = QLineEdit(default)
            self.batch_iso_fields[key] = edit
            row = LabeledField(label, edit)
            row.setVisible(False)
            batch_card.add(row)
            self._iso_rows.append(row)

        self.batch_matfix_check = QCheckBox("Override material variability with a fixed % of mean Y")
        batch_card.add(self.batch_matfix_check)
        self.batch_matfix_edit = QLineEdit("5.0")
        matfix_row = LabeledField("Fixed Y variability (%)", self.batch_matfix_edit)
        batch_card.add(matfix_row)

        run_row = QHBoxLayout()
        self.batch_run_button = QPushButton("Run Batch Analysis")
        self.batch_run_button.setObjectName("primary")
        self.batch_run_button.clicked.connect(self._on_run_batch)
        run_row.addWidget(self.batch_run_button)
        run_row.addStretch(1)
        batch_card.add_layout(run_row)
        self.batch_status = QLabel("")
        self.batch_status.setObjectName("cardSubtitle")
        batch_card.add(self.batch_status)

        self.batch_view_combo = QComboBox()
        self.batch_view_combo.addItems(["Combined Curve", "Relative Uncertainty"])
        self.batch_view_combo.currentIndexChanged.connect(self._refresh_batch_plot)
        batch_card.add(LabeledField("View", self.batch_view_combo))
        self.batch_canvas = PlotCanvas(
            figsize=(8, 4.5),
            help_title="Batch Analysis plot",
            help_text=(
                "<p>Shows the statistical summary computed by <b>Run Batch Analysis</b>, against "
                "the chosen X/Y columns.</p>"
                "<ul>"
                "<li><b>Combined Curve</b>: the mean Y curve together with its Experimental (each "
                "test's own Monte-Carlo uncertainty), Material (test-to-test scatter) and Total "
                "variability bands, as shaded envelope/outline curves.</li>"
                "<li><b>Relative Uncertainty</b>: each band's width as a percentage of the mean -- "
                "often the more useful view for judging how much each source contributes and where "
                "along the curve it matters most.</li>"
                "</ul>"
                "<p>Switch views with the \"View\" dropdown above; use the toolbar to zoom/pan.</p>"
            ),
        )
        batch_card.add(self.batch_canvas)

        batch_export_row = QHBoxLayout()
        self.batch_export_result_button = QPushButton("Export batch result…")
        self.batch_export_result_button.clicked.connect(self._on_export_batch_result)
        self.batch_export_result_button.setEnabled(False)
        batch_export_row.addWidget(self.batch_export_result_button)
        batch_export_row.addStretch(1)
        batch_card.add_layout(batch_export_row)

        layout.addWidget(batch_card)

        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _on_browse_batch_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select each test's exported file (Results & Export -> Export as .csv…/.txt…)",
            "", "Data files (*.csv *.txt);;CSV files (*.csv);;Text files (*.txt);;All files (*)",
        )
        if not files:
            return
        self._batch_files = files
        names = ", ".join(os.path.basename(f) for f in files)
        self.batch_files_edit.setText(f"{len(files)} file(s): {names}")
        self._refresh_batch_columns()

    def _refresh_batch_columns(self):
        self.batch_x_combo.setEnabled(False)
        self.batch_y_combo.setEnabled(False)
        if not self._batch_files:
            return
        try:
            header = load_export_header(self._batch_files[0])
        except (OSError, ValueError) as exc:
            self.batch_columns_status.setText(f"Could not read column headers: {exc}")
            return

        # Any plain column can be X or Y (e.g. confinement_pressure on X,
        # confinement_deviator on Y) -- but a "<column>_std" companion
        # column never goes in either picker itself: it's never something
        # to plot on its own, only ever the automatic Experimental
        # uncertainty for whichever plain column is picked as Y (see
        # `_refresh_std_status`).
        plain_columns = [c for c in header if not c.endswith("_std")]

        for combo, candidates in (
            (self.batch_x_combo, ("strain", "nominal_strain")),
            (self.batch_y_combo, ("stress", "nominal_stress")),
        ):
            combo.blockSignals(True)
            combo.clear()
            for col in plain_columns:
                combo.addItem(col, userData=col)
            default_idx = None
            for cand in candidates:
                idx = combo.findData(cand)
                if idx >= 0:
                    default_idx = idx
                    break
            combo.setCurrentIndex(default_idx if default_idx is not None else 0)
            combo.setEnabled(True)
            combo.blockSignals(False)

        if len(self._batch_files) > 1:
            note = (
                f"Columns read from {os.path.basename(self._batch_files[0])} ({len(header)} columns) -- "
                "the same column names are assumed for every other selected file."
            )
        else:
            note = f"{len(header)} columns available."
        self.batch_columns_status.setText(note)
        self._refresh_std_status()

    def _refresh_std_status(self):
        if not self._batch_files or not self.batch_y_combo.isEnabled():
            self.batch_std_status.setText("")
            return
        y_column = self.batch_y_combo.currentData()
        try:
            header = load_export_header(self._batch_files[0])
        except (OSError, ValueError):
            return
        std_name = f"{y_column}_std" if y_column else None
        if std_name and std_name in header:
            self.batch_std_status.setText(
                f"Experimental uncertainty: using \"{std_name}\" automatically (found alongside \"{y_column}\")."
            )
        else:
            self.batch_std_status.setText(
                f"No \"{y_column}_std\" column found -- Experimental variability on Y will be 0 "
                "(Material variability still applies)."
            )

    def _on_batch_iso_toggled(self, checked):
        for row in self._iso_rows:
            row.setVisible(checked)

    def _on_run_batch(self):
        if not self._batch_files:
            self.batch_status.setText("Select the exported test files first.")
            return

        x_column = self.batch_x_combo.currentData()
        y_column = self.batch_y_combo.currentData()
        if not x_column or not y_column:
            self.batch_status.setText("Select both an X axis column and a Y axis column.")
            return

        iso = IsothermalSettings()
        if self.batch_iso_check.isChecked():
            try:
                iso.enabled = True
                iso.xi = float(self.batch_iso_fields["xi"].text())
                iso.cp = float(self.batch_iso_fields["cp"].text())
                iso.rho = float(self.batch_iso_fields["rho"].text())
                iso.alpha = float(self.batch_iso_fields["alpha"].text())
            except ValueError:
                self.batch_status.setText("Isothermal correction fields must be numeric.")
                return

        fixed_pct = None
        if self.batch_matfix_check.isChecked():
            try:
                fixed_pct = float(self.batch_matfix_edit.text())
            except ValueError:
                self.batch_status.setText("Fixed Y variability must be numeric.")
                return

        try:
            result = run_batch_analysis_from_files(
                self._batch_files, x_column, y_column, sigma=self.project.uncertainty_sigma, iso=iso,
                fixed_pct=fixed_pct,
            )
        except (ValueError, OSError) as exc:
            self.batch_status.setText(f"Batch analysis failed: {exc}")
            return

        self._batch_result = result
        self.batch_export_result_button.setEnabled(True)
        status = f"Combined {result.n_tests} tests: {', '.join(result.test_names)}"
        if result.resampling_method == "index_fraction":
            status += (
                f" -- \"{x_column}\" isn't monotonically increasing for at least one test (e.g. a stress path "
                "through unload/reload), so tests were combined by each one's own recorded order instead of by "
                "matching X values directly."
            )
        self.batch_status.setText(status)
        self._refresh_batch_plot()

    def _refresh_batch_plot(self):
        result = self._batch_result
        self.batch_canvas.clear()
        ax = self.batch_canvas.figure.add_subplot(111)
        if result is None:
            ax.text(0.5, 0.5, "Run the batch analysis to see a plot here.", ha="center", va="center",
                    color="#8b91a1", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            self.batch_canvas.refresh()
            return

        view = self.batch_view_combo.currentText()
        if view == "Combined Curve":
            # Total is the overall envelope (>= either sub-band everywhere,
            # since it's their quadrature sum) -- it's the only one drawn as
            # a filled area, as a light wash, so it never has to blend with
            # another fill. Experimental/Material are drawn as thin outline
            # lines (not fills) in their own colors on top -- lines never
            # alpha-blend, so what's on screen always matches the legend
            # exactly, unlike overlapping semi-transparent fills.
            ax.fill_between(result.x, result.mean - result.total_std, result.mean + result.total_std,
                             color=_TOTAL_COLOR, alpha=0.15, linewidth=0,
                             label="Experimental + Material variabilities (envelope)")
            ax.plot(result.x, result.mean - result.experimental_std, color=_EXPERIMENTAL_COLOR,
                    linewidth=1.4, linestyle="--")
            ax.plot(result.x, result.mean + result.experimental_std, color=_EXPERIMENTAL_COLOR,
                    linewidth=1.4, linestyle="--", label="Experimental variability (bounds)")
            ax.plot(result.x, result.mean - result.material_std, color=_MATERIAL_COLOR,
                    linewidth=1.4, linestyle=":")
            ax.plot(result.x, result.mean + result.material_std, color=_MATERIAL_COLOR,
                    linewidth=1.4, linestyle=":", label="Material variability (bounds)")
            ax.plot(result.x, result.mean, color="white", linewidth=2, label="Mean")
            ax.set_xlabel(result.x_label); ax.set_ylabel(result.y_label)
        else:
            ax.plot(result.x, result.relative_total_pct, color=_TOTAL_COLOR, label="Experimental + Material")
            ax.plot(result.x, result.relative_experimental_pct, color=_EXPERIMENTAL_COLOR, label="Experimental")
            ax.plot(result.x, result.relative_material_pct, color=_MATERIAL_COLOR, label="Material")
            ax.set_xlabel(result.x_label); ax.set_ylabel(f"Test variability, {result.y_label} (%)")
        ax.legend(fontsize=8)
        self.batch_canvas.refresh()

    def _on_export_batch_result(self):
        if self._batch_result is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select (or create) the export folder")
        if not folder:
            return
        try:
            export_batch_result(self._batch_result, folder)
        except OSError as exc:
            self.batch_status.setText(f"Could not export: {exc}")
            return
        self.batch_status.setText(f"Exported batch result to {folder}")
