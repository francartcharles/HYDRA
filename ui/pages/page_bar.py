"""
Bar page (Input Bar / Output Bar).

Replaces test_type.IB_frame() and test_type.OB_frame(), which were two
~900-line, nearly-identical Toplevel builders (one per bar). Both are now
one class parametrized by `which="IB"|"OB"`, and both the signal-file
loader and the parameter-config loader are inline rows instead of their
own popup windows.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QFileDialog

from core.models import Bar
from core.param_io import load_param_file, save_param_file
from core.downsampling import downsample
from ui.widgets import Card, MCParamTable, FileSelectRow, make_scroll_page
from ui.span_picker import TimeWindowPicker
from ui import theme

PROPERTIES = [
    ("temperature", "Temperature axis (K)", None),
    ("e", "Elastic Modulus vs. T", "load_e"),
    ("rho", "Density vs. T", "load_rho"),
    ("nu", "Poisson Ratio vs. T", "load_nu"),
    ("lambda", "Thermal Conductivity vs. T", "load_lambda"),
]


class BarPage(QWidget):
    def __init__(self, project, which: str, parent=None):
        super().__init__(parent)
        self.project = project
        self.which = which
        self._drawn_signal = None  # the bar.signal array object last drawn (see _sync_signal_view)
        bar = project.input_bar if which == "IB" else project.output_bar

        title = "Input Bar" if which == "IB" else "Output Bar"
        scroll, layout = make_scroll_page(
            title,
            "Bar geometry, material properties (as Monte-Carlo parameters) and the "
            "raw strain-gage signal used by the dispersion-correction and processing pages.",
        )

        props = Card(
            "Properties",
            help_html=(
                f"<p>Geometric and material properties of the {title.lower()}, each as a "
                "Monte-Carlo parameter (Mean + Dispersion + Normal/Uniform distribution) rather "
                "than a single fixed number.</p>"
                "<ul>"
                "<li><b>Mean</b> is the value used everywhere a single number is needed; "
                "<b>Dispersion</b> (standard deviation for Normal, half-width for Uniform) drives "
                "how much this parameter is randomly varied across the project's Monte-Carlo "
                "draws.</li>"
                "<li>Set Dispersion to 0 to treat a parameter as exactly known (no propagated "
                "uncertainty from it).</li>"
                "<li>Use the config-file row below to load or save this whole set of parameters "
                "at once, e.g. to reuse a standard bar's properties across projects.</li>"
                "</ul>"
            ),
        )
        self.table = MCParamTable([bar.params[key] for key in Bar.PARAM_ORDER])
        props.add(self.table)

        config_row = QHBoxLayout()
        self.config_file_row = FileSelectRow(f"{title} config file")
        self.config_file_row.file_selected.connect(self._on_config_file_selected)
        config_row.addWidget(self.config_file_row, 1)
        save_button = QPushButton("Save as…")
        save_button.clicked.connect(self._on_save_config)
        config_row.addWidget(save_button)
        props.add_layout(config_row)
        self.config_status = QLabel("")
        self.config_status.setObjectName("cardSubtitle")
        props.add(self.config_status)
        layout.addWidget(props)

        signal_card = Card(
            "Raw signal",
            "Load the strain-gage voltage-vs-time signal recorded on this bar. If needed, invert "
            "its sign or zero it against a quiet (no-signal) portion of the recording.",
            help_html=(
                f"<p>Loads the raw strain-gage signal recorded on the {title.lower()} -- this is "
                "the signal every downstream page (Dispersion Corrections, Processing) reads.</p>"
                "<ul>"
                "<li><b>Signal file (.txt)</b>: one voltage/strain sample per line.</li>"
                "<li><b>Invert sign</b>: flips the signal's polarity, for gage wiring that produced "
                "the opposite sign from what the rest of the app expects.</li>"
                "<li><b>Zero on selected range</b>: subtracts the mean of the window dragged on the "
                "plot below (pick a flat, quiet portion before the loading event) so that portion "
                "becomes the new zero baseline.</li>"
                "<li><b>Reset sign/offset</b>: undoes both, back to the file exactly as loaded.</li>"
                "</ul>"
                "<p><b>Technical detail:</b> sign/offset are re-applied from scratch to the "
                "originally-loaded raw samples every time (signal = raw &times; sign &minus; "
                "offset), so repeated flips/re-zeroing never accumulate floating-point drift.</p>"
            ),
        )
        self.file_row = FileSelectRow(f"{title} signal file (.txt)")
        self.file_row.file_selected.connect(self._on_file_selected)
        signal_card.add(self.file_row)
        self.status_label = QLabel("No signal loaded.")
        self.status_label.setObjectName("cardSubtitle")
        signal_card.add(self.status_label)
        self.picker = TimeWindowPicker(
            [("zero_range", "Zero/offset range", theme.ACCENT)], figsize=(8, 3),
            help_title="Raw signal plot",
            help_text=(
                "<p>Shows the raw strain-gage voltage recorded on this bar, before any correction "
                "or processing. Use it to check the signal looks sane (no clipping, reasonable "
                "noise level).</p>"
                "<ul>"
                "<li>Click and drag on the plot to set the <b>Zero/offset range</b> window -- pick "
                "a flat, quiet portion of the trace, before the impact/loading event.</li>"
                "<li>Then click <b>Zero on selected range</b> (below the plot) so that portion's "
                "average value becomes the new zero baseline.</li>"
                "</ul>"
            ),
        )
        signal_card.add(self.picker)

        sign_row = QHBoxLayout()
        self.invert_sign_button = QPushButton("Invert sign")
        self.invert_sign_button.clicked.connect(self._on_invert_sign)
        sign_row.addWidget(self.invert_sign_button)
        self.zero_range_button = QPushButton("Zero on selected range")
        self.zero_range_button.clicked.connect(self._on_zero_range)
        sign_row.addWidget(self.zero_range_button)
        self.reset_sign_offset_button = QPushButton("Reset sign/offset")
        self.reset_sign_offset_button.clicked.connect(self._on_reset_sign_offset)
        sign_row.addWidget(self.reset_sign_offset_button)
        sign_row.addStretch(1)
        signal_card.add_layout(sign_row)
        self.sign_offset_status_label = QLabel("")
        self.sign_offset_status_label.setObjectName("cardSubtitle")
        signal_card.add(self.sign_offset_status_label)
        layout.addWidget(signal_card)

        table_card = Card(
            f"{title} material temperature tables",
            "Optional: import temperature-dependent bar properties for tests with a heated "
            "specimen (heat_check). If a property isn't loaded, the Monte-Carlo mean value from "
            "the Properties card above is used instead.",
            help_html=(
                "<p>Only needed for a heated-specimen test (heat_check), where the bar's own "
                "properties near the gage can vary with the temperature gradient set up by the "
                "hot specimen.</p>"
                "<ul>"
                "<li>Each row loads a lookup table of one property vs. temperature (elastic "
                "modulus, density, Poisson ratio, thermal conductivity) from a file.</li>"
                "<li>A property left unloaded simply falls back to the constant Monte-Carlo mean "
                "set on the Properties card above -- nothing breaks if these tables are skipped "
                "for a room-temperature test.</li>"
                "</ul>"
            ),
        )
        self.material_table = project.material_table_ib if which == "IB" else project.material_table_ob
        self.material_table_status = QLabel("No tables loaded.")
        self.material_table_status.setObjectName("cardSubtitle")
        table_card.add(self.material_table_status)
        for prop, prop_label, loader_name in PROPERTIES:
            row = FileSelectRow(prop_label)
            row.file_selected.connect(lambda path, p=prop, ln=loader_name: self._on_material_table_file(p, ln, path))
            table_card.add(row)
        layout.addWidget(table_card)

        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._restore_from_session()

    @property
    def _raw_signal(self):
        """Unmodified, freshly-loaded signal (sign/offset are applied on top
        of this). Backed by `Bar.signal_raw` so that it survives page
        rebuilds and can be restored from a full-test JSON."""
        raw = self._bar().signal_raw
        return raw if len(raw) > 0 else None

    @_raw_signal.setter
    def _raw_signal(self, value):
        self._bar().signal_raw = np.array([]) if value is None else np.asarray(value, dtype=float)

    def showEvent(self, event):
        super().showEvent(event)
        self.table.refresh()  # picks up values changed elsewhere (e.g. a loaded Setup Profile)
        self._sync_signal_view()

    def _sync_signal_view(self):
        """Shows the bar's current signal if it was set from outside this
        page (e.g. restored from a full-test JSON via Test Setup's "Load
        Test") and isn't what is currently drawn."""
        bar = self._bar()
        if len(bar.signal) == 0 or self._drawn_signal is bar.signal:
            return
        self._draw_signal(bar.signal)
        self.status_label.setText(f"Loaded {len(bar.signal)} samples (restored from a saved test / session)")

    def _path_key(self, suffix: str) -> str:
        return f"{'input' if self.which == 'IB' else 'output'}_bar_{suffix}"

    def _restore_from_session(self):
        """If a previous session already loaded a signal/config file into
        this bar (see core/session.py), reflect that in the UI immediately
        instead of showing an empty page until the user re-loads it."""
        from pathlib import Path
        path = self.project.last_paths.get(self._path_key("signal"), "")
        if len(self._bar().signal) > 0 and len(self._bar().signal_raw) > 0:
            # Signal already present on the bar (restored from a full-test
            # JSON's embedded data): keep it, don't overwrite from a file.
            self._sync_signal_view()
        elif path and Path(path).exists():
            # Re-read the RAW file fresh, then re-apply the sign/offset
            # already restored onto `bar` (by core/session.py) on top of it
            # -- keeps `_raw_signal` correctly unmodified either way.
            self._load_raw_from_path(path, note=" (from previous session)")
        elif len(self._bar().signal) > 0:
            # No path remembered (older session, or path since forgotten) but
            # the bar's signal itself was restored -- show it as-is.
            bar = self._bar()
            sign = bar.sign if bar.sign else 1.0
            self._raw_signal = (bar.signal + bar.offset) / sign  # undo sign/offset -> raw
            self._draw_signal(bar.signal)
            self.status_label.setText(f"Loaded {len(self._bar().signal)} samples (from previous session)")
        config_path = self.project.last_paths.get(self._path_key("config"), "")
        if config_path:
            self.config_status.setText(f"Parameters restored from previous session ({config_path.split('/')[-1]})")

    def _bar(self) -> Bar:
        return self.project.input_bar if self.which == "IB" else self.project.output_bar

    def _on_config_file_selected(self, path: str):
        bar = self._bar()
        try:
            load_param_file(path, bar.params, Bar.PARAM_ORDER)
        except (OSError, ValueError, SyntaxError) as exc:
            self.config_status.setText(f"Could not load config: {exc}")
            return
        self.project.last_paths[self._path_key("config")] = path
        self.table.refresh()
        self.config_status.setText(f"Loaded parameters from {path.split('/')[-1]}")

    def _on_save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save bar configuration", "", "Text files (*.txt)")
        if not path:
            return
        try:
            save_param_file(path, self._bar().params, Bar.PARAM_ORDER)
        except OSError as exc:
            self.config_status.setText(f"Could not save config: {exc}")
            return
        self.config_status.setText(f"Saved to {path.split('/')[-1]}")

    def _on_material_table_file(self, prop, loader_name, path):
        table = self.material_table
        try:
            if prop == "temperature":
                table.load_temperature(path)
            else:
                getattr(table, loader_name)(path)
        except Exception as exc:  # noqa: BLE001 - surface inline, no popup
            self.material_table_status.setText(f"Error loading {prop}: {exc}")
            return
        loaded = ["temperature"] + [p for p, _, _ in PROPERTIES[1:] if table.has(p)]
        self.material_table_status.setText("Loaded: " + ", ".join(loaded))

    def _on_file_selected(self, path: str):
        self._load_raw_from_path(path)

    def _load_raw_from_path(self, path: str, note: str = ""):
        bar = self._bar()
        try:
            values = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        values.append(float(line))
            signal = np.asarray(values)
            if self.project.downsample_enabled:
                signal = downsample(signal, self.project.downsample_n)
            self._raw_signal = signal
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Could not read file: {exc}")
            return

        self.project.last_paths[self._path_key("signal")] = path
        self._apply_sign_offset()
        self.status_label.setText(f"Loaded {len(self._raw_signal)} samples from {path.split('/')[-1]}{note}")

    def _apply_sign_offset(self):
        """Recomputes `bar.signal = raw * sign - offset` from scratch every
        time, so repeated sign flips / re-zeroing never accumulate
        floating-point drift and always reflect exactly what `bar.sign`/
        `bar.offset` currently say (important for session-restore, which
        just re-reads the raw file then calls this once)."""
        if self._raw_signal is None:
            return
        bar = self._bar()
        bar.signal = self._raw_signal * bar.sign - bar.offset
        self._draw_signal(bar.signal)

    def _on_invert_sign(self):
        if self._raw_signal is None:
            self.sign_offset_status_label.setText("Load a signal first.")
            return
        bar = self._bar()
        bar.sign *= -1
        self._apply_sign_offset()
        self.sign_offset_status_label.setText(f"Sign inverted (currently {'inverted' if bar.sign < 0 else 'normal'}).")

    def _on_zero_range(self):
        if self._raw_signal is None:
            self.sign_offset_status_label.setText("Load a signal first.")
            return
        window = self.picker.get_window("zero_range")
        if window is None:
            self.sign_offset_status_label.setText("Drag on the plot (Zero/offset range) to pick a baseline range first.")
            return
        bar = self._bar()
        # Offset is defined against the sign-corrected raw signal, so a
        # later sign flip doesn't silently invalidate a previously-computed
        # offset -- recompute the mean over [start, end] of `raw * sign`.
        signed_raw = self._raw_signal * bar.sign
        lo, hi = int(round(window[0])), int(round(window[1]))
        lo, hi = max(0, lo), min(len(signed_raw), hi)
        if hi <= lo:
            self.sign_offset_status_label.setText("Selected range is empty -- drag a wider window.")
            return
        mean_value = float(np.mean(signed_raw[lo:hi]))
        bar.offset = mean_value
        self._apply_sign_offset()
        self.sign_offset_status_label.setText(
            f"Zeroed: subtracted the mean over samples [{lo}, {hi}] ({mean_value:.6g})."
        )

    def _on_reset_sign_offset(self):
        bar = self._bar()
        bar.sign = 1.0
        bar.offset = 0.0
        self._apply_sign_offset()
        self.sign_offset_status_label.setText("Sign and offset reset to their original values.")

    def _draw_signal(self, signal):
        self._drawn_signal = self._bar().signal
        self.picker.set_signal(np.arange(len(signal)), [("signal", signal, theme.ACCENT)], xlabel="sample")
