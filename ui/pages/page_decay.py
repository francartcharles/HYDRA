"""
Free-decay damping page.

Replaces decay.py's Toplevel. Fully wired to core/decay.py -- this module
was small and self-contained enough to port and verify logic-for-logic.
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout

from core.decay import compute_decay
from ui.widgets import Card, FileSelectRow, make_scroll_page, TimeSourceRow
from ui.plot_canvas import PlotCanvas


class DecayPage(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._time = None
        self._signal = None

        scroll, layout = make_scroll_page(
            "Free-Decay Damping",
            "Estimate a bar's damping ratio from a free-decay (e.g. hammer-tap) signal, "
            "using the half-power (-3 dB) bandwidth method around the FFT peak.",
        )

        inputs = Card(
            "Signal",
            help_html=(
                "<p>Load the free-decay signal (e.g. a hammer-tap recording on the bar) that the "
                "damping ratio will be estimated from.</p>"
                "<ul>"
                "<li><b>Time file / Generate from a time step</b>: either load a time column, or "
                "check the generate box and type a fixed sample interval (dt) -- the app then "
                "builds a uniform time vector matching the wave file's length automatically.</li>"
                "<li><b>Wave file</b>: the actual recorded decay signal (voltage/amplitude vs. "
                "time).</li>"
                "<li>Click <b>Compute damping</b> once both are loaded.</li>"
                "</ul>"
            ),
        )
        self.time_row = TimeSourceRow("Time file (.txt)")
        self.signal_row = FileSelectRow("Wave file (.txt)")
        self.time_row.file_selected.connect(self._on_time_file)
        self.time_row.dtChanged.connect(self._on_time_step_changed)
        self.signal_row.file_selected.connect(self._on_signal_file)
        inputs.add(self.time_row)
        inputs.add(self.signal_row)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Compute damping")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._on_run)
        run_row.addWidget(self.run_button)
        run_row.addStretch(1)
        inputs.add_layout(run_row)

        self.result_label = QLabel("")
        self.result_label.setObjectName("cardSubtitle")
        inputs.add(self.result_label)
        layout.addWidget(inputs)

        result_card = Card("Result")
        self.canvas = PlotCanvas(
            figsize=(8, 5),
            help_title="Free-decay damping result",
            help_text=(
                "<p>Populated once <b>Compute damping</b> has run. The text line above this plot "
                "reports the damping ratio, natural frequency and full damping factor (rad/m) -- "
                "the last is automatically saved to the viscoelastic-correction settings, and can "
                "be pulled into the Dispersion Corrections or Viscoelastic Model Calibration pages "
                "with their own \"Use value from Free-Decay Damping tool\" buttons.</p>"
                "<ul>"
                "<li><b>Top panel</b>: the raw signal with its fitted exponential decay envelope "
                "overlaid.</li>"
                "<li><b>Bottom panel</b>: the FFT amplitude spectrum (log-frequency axis).</li>"
                "</ul>"
                "<p><b>Technical detail:</b> the damping ratio comes from the half-power (-3 dB) "
                "bandwidth around the FFT's dominant peak. A narrow, clean peak gives a more "
                "reliable estimate than a noisy or split one -- if the spectrum looks off, check "
                "that the recorded decay signal is long enough and free of extra vibration "
                "sources.</p>"
            ),
        )
        result_card.add(self.canvas)
        layout.addWidget(result_card)

        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _on_time_file(self, path):
        self._time = _read_column(path)

    def _on_signal_file(self, path):
        self._signal = _read_column(path)
        self._maybe_regenerate_time()

    def _on_time_step_changed(self, dt):
        """Alternative to `_on_time_file` above: build the time vector from
        a fixed time step matching the wave file's length, instead of
        loading it from a file."""
        self._maybe_regenerate_time()
        if self._time is None:
            self.result_label.setText("Load the wave file first, so the generated time vector can match its length.")

    def _maybe_regenerate_time(self):
        if not self.time_row.is_generate_mode():
            return
        n = len(self._signal) if self._signal is not None else 0
        if n == 0:
            return
        generated = self.time_row.build(n)
        if generated is not None:
            self._time = list(generated)

    def _on_run(self):
        if self._time is None or self._signal is None:
            self.result_label.setText("Load both the time file and the wave file first.")
            return
        try:
            result = compute_decay(self._time, self._signal)
        except Exception as exc:  # noqa: BLE001 - show inline, no messagebox
            self.result_label.setText(f"Could not compute damping: {exc}")
            return

        self.project.viscoelastic.damping_factor = result.full_damping_factor
        self.result_label.setText(
            f"Damping ratio = {result.damping_ratio:.6g}   |   "
            f"Natural frequency = {result.natural_frequency_hz:.2f} Hz   |   "
            f"Full damping factor = {result.full_damping_factor:.4g} rad/m "
            "(saved to the viscoelastic-correction settings)"
        )

        self.canvas.clear()
        ax1 = self.canvas.figure.add_subplot(211)
        ax1.plot(result.time, result.signal, "r-", label="full signal")
        ax1.plot(result.time, result.exponential_envelope, "b--", label="exponential decay")
        ax1.set_xlabel("time")
        ax1.set_ylabel("amplitude")
        ax1.legend()

        ax2 = self.canvas.figure.add_subplot(212)
        ax2.semilogx(result.freq, result.amplitude_spectrum, "b-", label="FFT spectrum")
        ax2.set_xlabel("frequency")
        ax2.set_ylabel("amplitude")
        ax2.legend()
        self.canvas.refresh()


def _read_column(path: str):
    values = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                values.append(float(line))
    return values
