"""
Covariance Analysis page.

Optional pipeline step (after Processing), ported from export_res.py's
`np.cov(...)` calculations. While every result already shows a per-point
standard deviation (the diagonal of this matrix, in effect), this page
computes and displays the FULL Monte-Carlo covariance matrix for a chosen
quantity -- from a chosen Processing result (Load/Displacement Data / SHPB /
Direct Impact), OR from an enabled Custom Sensor (perturbed by its own reported
uncertainty to synthesize Monte-Carlo draws) -- showing how uncertainty at
one strain/time point correlates with uncertainty at another, not just
its own magnitude.

Every computed matrix is kept in a checkable list so several can be
exported together in one click.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QHBoxLayout, QFileDialog, QLineEdit, QListWidget,
    QListWidgetItem,
)

from ui.widgets import Card, LabeledField, make_scroll_page
from ui.plot_canvas import PlotCanvas
from core.covariance import compute_covariance_matrix, interpolate_draws_to_common_grid, QUANTITY_LABELS
from core.confinement import triaxial_synthetic_draws, laser_stress as laser_stress_fn
from core.project import CovarianceLibraryEntry
from core.covariance_library_io import (
    CovarianceLibraryIOError, export_covariance_library, import_covariance_library, unique_entry_name,
)

RESULT_LABELS = {"QS": "Load/Displacement Data", "SHPB": "SHPB", "DI": "Direct Impact"}
# QUANTITY_LABELS itself now lives in core/covariance.py (a neutral,
# dependency-free module) rather than here, since ui/pages/page_bayes_
# calib.py needs the exact same vocabulary for its own per-role "which
# physical quantity does this equation symbol represent" picker and
# importing it from this UI page would be an unnecessary UI-to-UI
# dependency. Re-imported under the same name so nothing below has to
# change.
N_SYNTHETIC_SENSOR_DRAWS = 200
# Laser-derived triaxial quantities are never computed inside a Processing
# tab's Monte-Carlo loop (laser stress stays a post-hoc, Results & Export-
# only calculation) -- so, unlike the gage-based quantities, they always
# come from _triax_draws's synthetic-redraw fallback, on its own separate
# "strain_laser" x-axis (see _triax_draws's docstring).
_LASER_QUANTITY_KEYS = {"axial_stress_laser", "pressure_laser", "deviator_laser"}


class CovariancePage(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._matrix = None
        self._matrix_label = ""
        self._computed = {}  # name -> matrix, for the multi-export list
        # Set alongside self._matrix by _on_compute -- see that method's
        # comment -- and read by _on_save_to_library below.
        self._last_grid = None
        self._last_mean = None
        self._last_quantity_key = ""
        self._last_n_used = 0
        self._last_grid_start = 0.0
        self._last_grid_end = 0.0

        scroll, layout = make_scroll_page(
            "Covariance Analysis",
            "Full Monte-Carlo covariance matrix for a chosen quantity -- how uncertainty at one "
            "strain/time point correlates with uncertainty at another, not just its own std-dev. "
            "Run a Processing tab first, or enable a Custom Sensor.",
        )

        select_card = Card(
            "Select data",
            help_html=(
                "<p>Choose which already-computed result to analyze.</p>"
                "<ul>"
                "<li><b>Result</b>: one of Load/Displacement Data, SHPB or Direct Impact -- run a "
                "Processing tab first, or enable a Custom Sensor, so a result actually exists "
                "here.</li>"
                "<li><b>Quantity</b>: which output of that result (strain, stress, strain rate, "
                "...) to compute the covariance matrix for.</li>"
                "</ul>"
            ),
        )
        self.result_combo = QComboBox()
        self.result_combo.currentIndexChanged.connect(self._on_result_changed)
        select_card.add(LabeledField("Result", self.result_combo))
        self.quantity_combo = QComboBox()
        select_card.add(LabeledField("Quantity", self.quantity_combo))
        layout.addWidget(select_card)

        grid_card = Card(
            "Common grid",
            "Each Monte-Carlo draw has its own axis, so before computing covariance every draw is "
            "linearly interpolated onto one shared grid of N points from a reference value onward "
            "-- values before it are ignored (matches the original's own reference/\"cut\" approach).",
            help_html=(
                "<p>Defines the shared grid every Monte-Carlo draw gets interpolated onto before "
                "the covariance matrix is computed.</p>"
                "<ul>"
                "<li><b>Reference (start) / End</b>: the range of the X quantity (e.g. strain) to "
                "cover -- values before the reference point are ignored.</li>"
                "<li><b>Number of points (N)</b>: the resulting matrix is N &times; N; more points "
                "give finer resolution but a larger, slower-to-compute matrix.</li>"
                "</ul>"
            ),
        )
        self.start_label = QLabel("Reference (start) strain")
        self.start_edit = QLineEdit("0.0")
        grid_card.add(self.start_label)
        grid_card.add(self.start_edit)
        self.end_label = QLabel("End strain")
        self.end_edit = QLineEdit("0.05")
        grid_card.add(self.end_label)
        grid_card.add(self.end_edit)
        self.n_points_edit = QLineEdit("50")
        grid_card.add(LabeledField("Number of points (N)", self.n_points_edit))
        layout.addWidget(grid_card)

        run_card = Card("Compute")
        run_row = QHBoxLayout()
        self.run_button = QPushButton("Compute covariance matrix")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._on_compute)
        run_row.addWidget(self.run_button)
        run_row.addStretch(1)
        run_card.add_layout(run_row)
        self.status_label = QLabel("")
        self.status_label.setObjectName("cardSubtitle")
        run_card.add(self.status_label)
        layout.addWidget(run_card)

        result_card = Card("Result")
        self.canvas = PlotCanvas(
            figsize=(7, 6),
            help_title="Covariance matrix",
            help_text=(
                "<p>Shows the full Monte-Carlo covariance matrix for the selected quantity, once "
                "<b>Compute covariance matrix</b> has run, as a color map over the common grid "
                "points (axes indexed by grid position, from the Reference/start value onward).</p>"
                "<ul>"
                "<li>The <b>diagonal</b> is each point's own variance (matches the std-dev shown "
                "elsewhere for that result).</li>"
                "<li><b>Off-diagonal</b> cells show how uncertainty at one point correlates with "
                "uncertainty at another -- strong off-diagonal structure means the draws move "
                "together rather than independently.</li>"
                "</ul>"
                "<p><b>Technical detail:</b> this matters when propagating this uncertainty "
                "through a later calculation instead of treating each grid point as an "
                "independent measurement.</p>"
            ),
        )
        result_card.add(self.canvas)
        layout.addWidget(result_card)

        library_card = Card(
            "Save to calibration library",
            "Saves the matrix above -- together with its common grid and mean curve -- into the "
            "project, so the Bayesian Calibration page can use it as one variable's data for one "
            "calibration test. Different from exporting to disk below: this keeps the pair inside "
            "the app, across pages and even across loading a different test.",
            help_html=(
                "<p>Packages the currently displayed covariance matrix, together with the common "
                "grid it was computed on and the mean curve across draws, into a named entry kept "
                "on the project (<b>not</b> written to disk) -- self-contained, so it stays usable "
                "even after you load a different test.</p>"
                "<ul>"
                "<li><b>Name</b>: pre-filled from the matrix above; edit it to something meaningful "
                "(e.g. <i>Test1_stress</i>) -- this is how you'll pick it on the Bayesian Calibration "
                "page. Saving under a name already in the library adds a numeric suffix rather than "
                "overwriting it.</li>"
                "<li>The list below shows every entry saved so far this session (and restored from "
                "your last session). Select one and click <b>Remove selected</b> to delete it.</li>"
                "<li><b>Save library…</b> / <b>Load library…</b>: exports or imports the WHOLE "
                "library (every entry above, including its grid/mean/covariance data) as a "
                "<code>.json</code>+<code>.csv</code> file pair, independent of this project's own "
                "auto-saved session -- for backing up, sharing with a colleague, or reloading into "
                "a fresh project. The <code>.csv</code> is a human-readable summary table only "
                "(name/quantity/size); the <code>.json</code> carries the full data and is the only "
                "one <b>Load library…</b> actually reads. Loading never overwrites an existing "
                "entry -- a name already present gets an automatic <code>_2</code>/<code>_3</code>/... "
                "suffix, same as saving a single entry above.</li>"
                "</ul>"
            ),
        )
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.library_name_edit = QLineEdit("")
        name_row.addWidget(self.library_name_edit, 1)
        library_card.add_layout(name_row)
        library_button_row = QHBoxLayout()
        self.save_to_library_button = QPushButton("Save to calibration library")
        self.save_to_library_button.clicked.connect(self._on_save_to_library)
        self.save_to_library_button.setEnabled(False)
        library_button_row.addWidget(self.save_to_library_button)
        library_button_row.addStretch(1)
        library_card.add_layout(library_button_row)
        self.library_list = QListWidget()
        self.library_list.setMaximumHeight(140)
        library_card.add(self.library_list)
        remove_row = QHBoxLayout()
        self.remove_from_library_button = QPushButton("Remove selected")
        self.remove_from_library_button.clicked.connect(self._on_remove_from_library)
        remove_row.addWidget(self.remove_from_library_button)
        remove_row.addStretch(1)
        library_card.add_layout(remove_row)
        library_io_row = QHBoxLayout()
        self.save_library_button = QPushButton("Save library (.json+.csv)…")
        self.save_library_button.clicked.connect(self._on_save_library_file)
        library_io_row.addWidget(self.save_library_button)
        self.load_library_button = QPushButton("Load library (.json)…")
        self.load_library_button.clicked.connect(self._on_load_library_file)
        library_io_row.addWidget(self.load_library_button)
        library_io_row.addStretch(1)
        library_card.add_layout(library_io_row)
        self.library_status_label = QLabel("")
        self.library_status_label.setObjectName("cardSubtitle")
        self.library_status_label.setWordWrap(True)
        library_card.add(self.library_status_label)
        layout.addWidget(library_card)

        export_card = Card(
            "Export",
            "Every matrix you compute is added below -- tick the ones you want and export them "
            "together in one click.",
            help_html=(
                "<p>Every matrix computed with <b>Compute covariance matrix</b> above is added to "
                "the list below, so several can be built (e.g. for different quantities or "
                "results) and exported together.</p>"
                "<ul>"
                "<li>Tick the ones to export, then <b>Export selected as .csv…</b> or <b>.txt…</b>.</li>"
                "<li><b>Clear list</b> empties the list without affecting the currently displayed "
                "result above.</li>"
                "</ul>"
            ),
        )
        self.computed_list = QListWidget()
        self.computed_list.setMaximumHeight(160)
        export_card.add(self.computed_list)
        export_row = QHBoxLayout()
        self.export_csv_button = QPushButton("Export selected as .csv…")
        self.export_csv_button.clicked.connect(lambda: self._on_export_selected("csv"))
        export_row.addWidget(self.export_csv_button)
        self.export_txt_button = QPushButton("Export selected as .txt…")
        self.export_txt_button.clicked.connect(lambda: self._on_export_selected("txt"))
        export_row.addWidget(self.export_txt_button)
        self.clear_button = QPushButton("Clear list")
        self.clear_button.clicked.connect(self._on_clear_computed)
        export_row.addWidget(self.clear_button)
        export_row.addStretch(1)
        export_card.add_layout(export_row)
        self.export_status_label = QLabel("")
        self.export_status_label.setObjectName("cardSubtitle")
        export_card.add(self.export_status_label)
        layout.addWidget(export_card)

        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_result_list()
        self._refresh_library_list()

    def _refresh_result_list(self):
        available = [k for k in RESULT_LABELS if k in self.project.last_raw_draws]
        sensor_keys = []
        for i, sensor in enumerate(self.project.custom_sensors):
            if sensor.enabled and len(sensor.time) > 0 and len(sensor.data) > 0:
                sensor_keys.append(f"SENSOR_{i}")
        triax_keys = self._available_triax_keys()
        all_keys = available + triax_keys + sensor_keys

        current = self.result_combo.currentData()
        self.result_combo.blockSignals(True)
        self.result_combo.clear()
        for key in available:
            self.result_combo.addItem(RESULT_LABELS[key], userData=key)
        for key in triax_keys:
            base = key.split("_", 1)[1]
            self.result_combo.addItem(f"Confinement / Triaxial ({RESULT_LABELS[base]})", userData=key)
        for i, key in enumerate(sensor_keys):
            sensor = self.project.custom_sensors[int(key.split("_")[1])]
            self.result_combo.addItem(f"Custom Sensor: {sensor.label}", userData=key)
        self.result_combo.blockSignals(False)
        if all_keys:
            idx = all_keys.index(current) if current in all_keys else 0
            self.result_combo.setCurrentIndex(idx)
        self._on_result_changed()

    def _available_triax_keys(self) -> list:
        """Result keys (from `RESULT_LABELS`) for which the Confinement /
        Triaxial analysis (Results & Export page, Special Tests settings)
        can actually run: confinement enabled, an enabled confinement-cell
        Custom Sensor present, and that result has a strain axis."""
        if not self.project.confinement.enabled:
            return []
        has_cell_sensor = any(
            s.role == "confinement_cell" and s.enabled and len(s.data) > 0 for s in self.project.custom_sensors
        )
        if not has_cell_sensor:
            return []
        keys = []
        for key in RESULT_LABELS:
            draws = self.project.last_raw_draws.get(key)
            if draws and ("nominal_strain" in draws or "strain" in draws):
                keys.append(f"TRIAX_{key}")
        return keys

    def _is_sensor(self, key) -> bool:
        return isinstance(key, str) and key.startswith("SENSOR_")

    def _is_triax(self, key) -> bool:
        return isinstance(key, str) and key.startswith("TRIAX_")

    def _on_result_changed(self, *_args):
        key = self.result_combo.currentData()
        self.quantity_combo.clear()
        if key is None:
            return

        if self._is_sensor(key):
            sensor = self.project.custom_sensors[int(key.split("_")[1])]
            # A sensor with the "Strain gage glued directly on the
            # specimen" role (Custom Sensors page) measures strain
            # directly, so its one available quantity is labeled to
            # match -- same underlying data ("signal", the sensor's own
            # calibrated_data()), just named for what it actually is,
            # consistent with how Results & Export labels this same
            # sensor's data as "Strain Gage Strain (-)".
            quantity_label = "Strain Gage Strain (-)" if sensor.role == "specimen_strain_gage" else "Signal (calibrated)"
            self.quantity_combo.addItem(quantity_label, userData="signal")
            self.start_label.setText("Reference (start) time")
            self.end_label.setText("End time")
            if len(sensor.time):
                self.start_edit.setText(f"{sensor.time.min():.6g}")
                self.end_edit.setText(f"{0.95 * sensor.time.max():.6g}")
            return

        if self._is_triax(key):
            base = key.split("_", 1)[1]
            # NOTE: uses `last_results` (the aggregate result), matching
            # exactly what `_triax_synthetic_draws` below builds its
            # synthetic draws' shared strain axis from -- NOT
            # `last_raw_draws` (the per-draw Monte-Carlo arrays used by
            # the non-triaxial branch below), which can have a different
            # (often longer) extent and would pre-fill an End strain the
            # triaxial draws don't actually reach, causing "fewer than 2
            # draws covered the requested range" as soon as Compute runs.
            result = self.project.last_results.get(base, {})
            has_laser = any(s.role == "laser_velocity" and s.enabled and len(s.data) > 0 for s in self.project.custom_sensors)
            quantities = ["axial_stress", "radial_stress", "pressure", "deviator",
                          "radial_strain", "volumetric_strain", "equivalent_strain"]
            if len(result.get("time", [])) >= 2:  # strain rates need the result's time base
                quantities += ["radial_strain_rate", "volumetric_strain_rate", "equivalent_strain_rate"]
            if has_laser:
                quantities += ["axial_stress_laser", "pressure_laser", "deviator_laser"]
            for q_key in quantities:
                self.quantity_combo.addItem(QUANTITY_LABELS.get(q_key, q_key), userData=q_key)
            self.start_label.setText("Reference (start) strain")
            self.end_label.setText("End strain")
            strain_key = "nominal_strain" if "nominal_strain" in result else "strain"
            if strain_key in result and len(result[strain_key]):
                self.end_edit.setText(f"{0.95 * np.asarray(result[strain_key]).max():.6g}")
            return

        draws = self.project.last_raw_draws.get(key, {})
        strain_key = "nominal_strain" if "nominal_strain" in draws else "strain"
        for q_key in draws:
            self.quantity_combo.addItem(QUANTITY_LABELS.get(q_key, q_key), userData=q_key)
        self.start_label.setText("Reference (start) strain")
        self.end_label.setText("End strain")

        if strain_key in draws:
            max_strains = [np.asarray(s).max() for s in draws[strain_key] if len(s)]
            if max_strains:
                self.end_edit.setText(f"{0.95 * min(max_strains):.6g}")

    def _sensor_synthetic_draws(self, sensor):
        """Perturbs the sensor's own calibrated signal by its reported
        uncertainty to synthesize Monte-Carlo-like draws, the same way
        core/confinement.py's `confinement_envelope` does for the
        Confinement / Triaxial Stresses view."""
        rng = np.random.default_rng()
        data = sensor.calibrated_data()
        lo, hi = sensor.uncertainty_band()
        half_width = (np.asarray(hi) - np.asarray(lo)) / 2
        x_axis = [sensor.time for _ in range(N_SYNTHETIC_SENSOR_DRAWS)]
        draws = [data + rng.normal(0, half_width) for _ in range(N_SYNTHETIC_SENSOR_DRAWS)]
        return x_axis, draws

    def _triax_draws(self, base_key: str) -> dict | None:
        """Per-draw triaxial quantities for the Confinement / Triaxial
        analysis of a given base result (SHPB/QS/DI).

        Prefers REAL per-Monte-Carlo-draw data (from `project.
        last_raw_draws`, computed inside the Processing tab's own loop --
        see core.confinement.per_draw_confinement_quantities) for every
        GAGE-based quantity, when that run actually computed it (the
        confinement-cell sensor's "Include this cut in the Monte-Carlo
        loop" was ticked, or this is a Load/Displacement Data/QS result,
        which always computes it -- see _QSTab._on_run). Falls back to the
        synthetic-redraw approximation (`triaxial_synthetic_draws`, the
        same one Results & Export's own envelope used to rely on
        entirely) otherwise.

        LASER-based quantities are never computed inside that loop (laser
        stress stays a post-hoc, Results & Export-only calculation, out of
        scope for this) -- they always come from the synthetic
        approximation, on their OWN "strain_laser" x-axis (a different,
        independent synthetic ensemble from the real per-draw one used for
        the gage-based quantities, so the two must never be zipped
        together against the wrong x-axis -- see _on_compute).

        Returns a dict with a "strain" key (shared x-axis for the 7
        gage-based quantities), a "_real" bool (whether "strain" is real
        per-draw data or the synthetic fallback), and -- only when a laser
        sensor is enabled and the synthetic draws succeeded -- a separate
        "strain_laser" key for the 3 laser quantities. Returns None if
        nothing at all could be computed."""
        result = self.project.last_results.get(base_key)
        raw_draws = self.project.last_raw_draws.get(base_key)
        if result is None or raw_draws is None:
            return None
        cell_sensor = next((s for s in self.project.custom_sensors
                             if s.role == "confinement_cell" and s.enabled and len(s.data) > 0), None)
        if cell_sensor is None:
            return None

        strain_key = "nominal_strain" if "nominal_strain" in result else "strain"
        stress_key = "nominal_stress" if "nominal_stress" in result else "stress"
        strain = np.asarray(result[strain_key])
        axial = np.asarray(result[stress_key])
        axial_std = np.asarray(result.get(f"{stress_key}_std", []))
        cell_defl = cell_sensor.calibrated_data()
        cell_lo, cell_hi = cell_sensor.uncertainty_band()
        cell_defl_std = (cell_hi - cell_lo) / 2
        conf = self.project.confinement

        laser_sensor = next((s for s in self.project.custom_sensors
                              if s.role == "laser_velocity" and s.enabled and len(s.data) > 0), None)
        laser_axial = laser_std = None
        if laser_sensor is not None:
            ob = self.project.output_bar
            laser_axial = laser_stress_fn(
                ob_diameter=ob.params["D"].mean, ob_rho=ob.params["rho"].mean, ob_c0=ob.params["C0"].mean,
                specimen_diameter=self.project.specimen.params["D"].mean, laser_velocity=laser_sensor.calibrated_data(),
            )
            l_lo, l_hi = laser_sensor.uncertainty_band()
            # Half-width of the VELOCITY band -> converted to a STRESS std (MPa) with the same
            # (linear) velocity->stress law; the velocity std itself is tiny next to the stress.
            laser_std = laser_stress_fn(
                ob_diameter=ob.params["D"].mean, ob_rho=ob.params["rho"].mean, ob_c0=ob.params["C0"].mean,
                specimen_diameter=self.project.specimen.params["D"].mean, laser_velocity=(l_hi - l_lo) / 2,
            )

        synth = None
        try:
            synth = triaxial_synthetic_draws(
                strain, axial, axial_std, cell_defl, cell_defl_std, conf,
                self.project.specimen.params["L"].mean,
                laser_stress_value=laser_axial, laser_stress_std=laser_std, n_draws=200,
                time=np.asarray(result["time"], dtype=float) if len(result.get("time", [])) >= 2 else None,
            )
        except Exception:  # noqa: BLE001
            synth = None

        real_key = "confinement_radial_stress"
        if real_key in raw_draws and strain_key in raw_draws and stress_key in raw_draws:
            n_real = len(raw_draws[real_key])
            real_strain = []
            for s in raw_draws[strain_key][:n_real]:
                s = np.array(s, dtype=float)
                if len(s) and abs(s[0]) < 1e-6:
                    # Same near-zero-snap-to-0.0 as triaxial_synthetic_draws,
                    # so "Reference (start) strain = 0.0" finds a draw that
                    # actually starts there for real data too.
                    s[0] = 0.0
                real_strain.append(s)
            gage_map = {
                "axial_stress": stress_key, "radial_stress": "confinement_radial_stress",
                "pressure": "confinement_pressure", "deviator": "confinement_deviator",
                "radial_strain": "confinement_radial_strain",
                "volumetric_strain": "confinement_volumetric_strain",
                "equivalent_strain": "confinement_equivalent_strain",
            }
            # Strain rates: real per-draw ones if this run computed them,
            # otherwise dropped (a synthetic rate must never be paired with
            # the real strain axis installed below).
            for rate_key in ("radial_strain_rate", "volumetric_strain_rate", "equivalent_strain_rate"):
                raw_rate = f"confinement_{rate_key}"
                if raw_rate in raw_draws:
                    gage_map[rate_key] = raw_rate
            out = dict(synth) if synth is not None else {}
            for rate_key in ("radial_strain_rate", "volumetric_strain_rate", "equivalent_strain_rate"):
                if rate_key not in gage_map:
                    out.pop(rate_key, None)
            if "strain" in out:
                out["strain_laser"] = out.pop("strain")  # only the laser keys still need this synthetic axis
            out["strain"] = real_strain
            for out_key, raw_key in gage_map.items():
                out[out_key] = list(raw_draws[raw_key][:n_real])
            out["_real"] = True
            return out

        if synth is not None:
            synth["_real"] = False
        return synth

    def _on_compute(self):
        result_key = self.result_combo.currentData()
        quantity_key = self.quantity_combo.currentData()
        if result_key is None or quantity_key is None:
            self.status_label.setText("Run a Processing tab first (or enable a Custom Sensor), then pick a result and quantity here.")
            return

        try:
            grid_start = float(self.start_edit.text())
            grid_end = float(self.end_edit.text())
            n_points = int(self.n_points_edit.text())
        except ValueError:
            self.status_label.setText("Reference value, end value and N must be numeric (N a whole number).")
            return
        if n_points < 2 or grid_end <= grid_start:
            self.status_label.setText("End value must be greater than the reference value, and N >= 2.")
            return

        real_note = ""
        if self._is_sensor(result_key):
            sensor = self.project.custom_sensors[int(result_key.split("_")[1])]
            x_axis, quantity_draws = self._sensor_synthetic_draws(sensor)
            label = f"CustomSensor_{sensor.label}"
            n_total = len(quantity_draws)
        elif self._is_triax(result_key):
            base = result_key.split("_", 1)[1]
            triax = self._triax_draws(base)
            if triax is None or quantity_key not in triax:
                self.status_label.setText(
                    "Could not generate the triaxial draws -- check that confinement is enabled and an "
                    "enabled Confinement-cell Custom Sensor is set (Special Tests / Custom Sensors pages)."
                )
                return
            # Laser quantities are always synthetic, on their OWN "strain_
            # laser" x-axis (a separate synthetic ensemble) -- never zip
            # them against the real per-draw "strain" used for the
            # gage-based quantities, which can differ in both length and
            # meaning. See _triax_draws's docstring.
            is_laser_quantity = quantity_key in _LASER_QUANTITY_KEYS
            strain_field = "strain_laser" if is_laser_quantity and "strain_laser" in triax else "strain"
            x_axis, quantity_draws = triax[strain_field], triax[quantity_key]
            label = f"Triax_{RESULT_LABELS[base]}_{quantity_key}"
            n_total = len(quantity_draws)
            real_note = (
                "" if is_laser_quantity  # laser is always the synthetic approximation, not worth repeating
                else (" (real per-draw Monte-Carlo data)" if triax.get("_real") else " (synthetic approximation)")
            )
        else:
            draws = self.project.last_raw_draws[result_key]
            strain_key = "nominal_strain" if "nominal_strain" in draws else "strain"
            if strain_key not in draws or quantity_key not in draws:
                self.status_label.setText("This result is missing the data needed to build the common grid.")
                return
            x_axis, quantity_draws = draws[strain_key], draws[quantity_key]
            label = f"{RESULT_LABELS[result_key]}_{quantity_key}"
            n_total = len(quantity_draws)

        grid, stacked = interpolate_draws_to_common_grid(x_axis, quantity_draws, grid_start, grid_end, n_points)
        if stacked.shape[0] < 2:
            self.status_label.setText(
                "Fewer than 2 draws covered the requested range -- widen the range or lower the end value."
            )
            return

        matrix = compute_covariance_matrix(stacked)
        if matrix is None:
            self.status_label.setText("Not enough draws to compute a covariance matrix.")
            return

        # Kept alongside self._matrix/self._matrix_label (not just local
        # variables) so "Save to calibration library" -- a separate click,
        # after this method has already returned -- can package a
        # self-contained CovarianceLibraryEntry (grid + representative
        # mean curve + matrix) for the Bayesian Calibration page. `grid`/
        # `stacked.mean(axis=0)` used to be computed and thrown away here.
        self._last_grid = grid
        self._last_mean = stacked.mean(axis=0)
        self._last_quantity_key = quantity_key
        self._last_n_used = stacked.shape[0]
        self._last_grid_start = grid_start
        self._last_grid_end = grid_end

        self._matrix = matrix
        self._matrix_label = label
        self._computed[label] = matrix
        self._refresh_computed_list()

        self.canvas.clear()
        ax = self.canvas.figure.add_subplot(111)
        im = ax.imshow(matrix, cmap="viridis", aspect="auto")
        self.canvas.figure.colorbar(im, ax=ax, label="Covariance")
        ax.set_xlabel("Grid point index")
        ax.set_ylabel("Grid point index")
        ax.set_title(label)
        self.canvas.refresh()

        n = matrix.shape[0]
        variances = np.diag(matrix)
        n_used = stacked.shape[0]
        self.status_label.setText(
            f"{n} x {n} matrix from {n_used}/{n_total} draws covering [{grid_start:.4g}, {grid_end:.4g}]{real_note}. "
            f"Diagonal (variance) range: {variances.min():.4g} to {variances.max():.4g}. Added to the list below."
        )

        self.library_name_edit.setText(label)
        self.save_to_library_button.setEnabled(True)

    def _on_save_to_library(self):
        if self._matrix is None or self._last_grid is None:
            self.library_status_label.setText("Compute a matrix first.")
            return
        name = self.library_name_edit.text().strip()
        if not name:
            self.library_status_label.setText("Give this entry a name first.")
            return
        existing = {e.name for e in self.project.covariance_library}
        unique_name = name
        suffix = 2
        while unique_name in existing:
            unique_name = f"{name}_{suffix}"
            suffix += 1

        entry = CovarianceLibraryEntry(
            name=unique_name,
            source_label=self._matrix_label,
            quantity_key=self._last_quantity_key,
            grid=list(np.asarray(self._last_grid, dtype=float)),
            mean=list(np.asarray(self._last_mean, dtype=float)),
            covariance=[list(row) for row in np.asarray(self._matrix, dtype=float)],
            n_draws_used=int(self._last_n_used),
            grid_start=float(self._last_grid_start),
            grid_end=float(self._last_grid_end),
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        self.project.covariance_library.append(entry)
        self._refresh_library_list()
        renamed_note = f" (renamed to avoid a collision: {unique_name})" if unique_name != name else ""
        self.library_status_label.setText(f"Saved '{unique_name}' to the calibration library{renamed_note}.")

    def _on_remove_from_library(self):
        row = self.library_list.currentRow()
        if row < 0 or row >= len(self.project.covariance_library):
            self.library_status_label.setText("Select an entry to remove first.")
            return
        removed = self.project.covariance_library.pop(row)
        self._refresh_library_list()
        self.library_status_label.setText(f"Removed '{removed.name}' from the calibration library.")

    def _on_save_library_file(self):
        if not self.project.covariance_library:
            self.library_status_label.setText("The calibration library is empty -- nothing to save.")
            return
        default_name = "calibration_library.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save calibration library", default_name, "CSV files (*.csv)")
        if not path:
            return
        try:
            json_path = export_covariance_library(path, self.project.covariance_library)
        except OSError as exc:
            self.library_status_label.setText(f"Could not save: {exc}")
            return
        self.library_status_label.setText(
            f"Saved calibration library: {os.path.basename(json_path)} (data, {len(self.project.covariance_library)} "
            f"entries) + {os.path.basename(path)} (summary)."
        )

    def _on_load_library_file(self):
        json_path, _ = QFileDialog.getOpenFileName(self, "Load calibration library", "", "JSON files (*.json)")
        if not json_path:
            return
        try:
            loaded = import_covariance_library(json_path)
        except (OSError, CovarianceLibraryIOError) as exc:
            self.library_status_label.setText(f"Could not load: {exc}")
            return

        existing_names = {e.name for e in self.project.covariance_library}
        renamed = []
        for entry in loaded:
            unique_name = unique_entry_name(existing_names, entry.name)
            if unique_name != entry.name:
                renamed.append(f"{entry.name} -> {unique_name}")
                entry.name = unique_name
            existing_names.add(unique_name)
            self.project.covariance_library.append(entry)
        self._refresh_library_list()

        note = f" ({len(renamed)} renamed to avoid a collision: {', '.join(renamed)})" if renamed else ""
        self.library_status_label.setText(
            f"Loaded {len(loaded)} entry(ies) from {os.path.basename(json_path)}{note}."
        )

    def _refresh_library_list(self):
        self.library_list.clear()
        for entry in self.project.covariance_library:
            n = len(entry.grid)
            self.library_list.addItem(f"{entry.name}  ({n}x{n}, from {entry.source_label or 'unknown'})")

    def _refresh_computed_list(self):
        self.computed_list.clear()
        for name in self._computed:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.computed_list.addItem(item)

    def _on_clear_computed(self):
        self._computed = {}
        self._refresh_computed_list()
        self.export_status_label.setText("")

    def _on_export_selected(self, fmt: str):
        selected = []
        for row in range(self.computed_list.count()):
            item = self.computed_list.item(row)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        if not selected:
            self.export_status_label.setText("Tick at least one matrix in the list above first.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Select the export folder")
        if not folder:
            return

        try:
            for name in selected:
                matrix = self._computed[name]
                path = os.path.join(folder, f"{name}_covariance.{fmt}")
                if fmt == "csv":
                    with open(path, "w", newline="") as f:
                        writer = csv.writer(f)
                        for row in matrix:
                            writer.writerow([f"{v:.8g}" for v in row])
                else:
                    np.savetxt(path, matrix, delimiter="\t", fmt="%.8g")
        except OSError as exc:
            self.export_status_label.setText(f"Could not export: {exc}")
            return
        self.export_status_label.setText(f"Exported {len(selected)} matrix/matrices to {folder}")
