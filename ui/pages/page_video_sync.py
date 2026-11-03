"""
Camera / Video Sync page.

Standalone tool (not part of the main per-test pipeline): browse a
high-speed video (MP4/MKV/AVI) or an image sequence (TIFF 8/16-bit, JPEG,
PNG, BMP) frame by frame via a slider, then locate the exact frame that
was being exposed at the camera's trigger instant and read off its
absolute time from a recorded time vector -- using the camera's own clock
(exposure) and trigger signals (core/camera_clock.py) and the frame source
abstraction in core/frame_source.py.
"""
from __future__ import annotations

import csv
import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QRadioButton, QButtonGroup, QComboBox, QSlider, QSpinBox,
    QFileDialog, QListWidget, QListWidgetItem,
)

from core.frame_source import open_frame_source, to_display_uint8, ImageSequenceFrameSource
from core.camera_clock import (
    read_clock_file, detect_pulse_train, estimate_frame_times, decimate_for_display,
    locate_trigger_by_signal, locate_trigger_by_frame_number,
)
from ui.widgets import Card, LabeledField, FileSelectRow, make_scroll_page
from ui.plot_canvas import PlotCanvas
from ui import theme


class VideoSyncPage(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._source = None          # a FrameSource (video or image sequence)
        self._clock_data = None      # (n_samples, n_columns) array from the loaded clock file
        self._clock_header: list[str] = []
        self._trigger_result = None  # last TriggerResult
        self._last_clock_plot_inputs = None  # (time, exposure, trigger) last used to draw the clock/trigger plot
        self._frame_times = None     # array, length n_frames: best-estimate absolute time of every frame
        self._measurement_canvases: dict[int, PlotCanvas] = {}  # column index -> its own small plot

        scroll, layout = make_scroll_page(
            "Camera / Video Sync",
            "Browse a high-speed video (MP4/MKV/AVI) or an image sequence (TIFF 8/16-bit, JPEG, "
            "PNG, BMP) frame by frame, then locate the exact frame exposed at the camera's trigger "
            "instant and read off its absolute time from a recorded time vector.",
        )

        # ------------------------------------------------------------ load frames
        source_card = Card(
            "Load Frames",
            "Choose a video file, or a folder of single-frame images (naturally sorted by file name).",
            help_html=(
                "<p>Loads the high-speed video/image sequence to browse frame by frame.</p>"
                "<ul>"
                "<li><b>Video file</b>: a single .mp4/.mkv/.avi file.</li>"
                "<li><b>Image sequence (folder)</b>: a folder of single-frame images (TIFF "
                "8/16-bit, JPEG, PNG, BMP), naturally sorted by file name.</li>"
                "</ul>"
                "<p>Once loaded, use the Frame Viewer card below to scrub through frames.</p>"
            ),
        )
        mode_row = QHBoxLayout()
        self.video_radio = QRadioButton("Video file (.mp4 / .mkv / .avi)")
        self.sequence_radio = QRadioButton("Image sequence (folder)")
        self.video_radio.setChecked(True)
        source_mode_group = QButtonGroup(self)
        source_mode_group.addButton(self.video_radio)
        source_mode_group.addButton(self.sequence_radio)
        self.video_radio.toggled.connect(self._on_source_mode_toggled)
        mode_row.addWidget(self.video_radio)
        mode_row.addWidget(self.sequence_radio)
        mode_row.addStretch(1)
        source_card.add_layout(mode_row)

        self.video_row = FileSelectRow("Video file", "Video files (*.mp4 *.mkv *.avi);;All files (*)")
        self.video_row.file_selected.connect(self._load_source)
        source_card.add(self.video_row)

        seq_row = QHBoxLayout()
        seq_row.addWidget(QLabel("Image folder"))
        self.seq_folder_edit = QLineEdit()
        self.seq_folder_edit.setReadOnly(True)
        self.seq_folder_edit.setPlaceholderText("No folder selected")
        seq_row.addWidget(self.seq_folder_edit, 1)
        seq_browse_button = QPushButton("Browse…")
        seq_browse_button.clicked.connect(self._on_browse_sequence_folder)
        seq_row.addWidget(seq_browse_button)
        self.seq_row_widget = QWidget()
        self.seq_row_widget.setLayout(seq_row)
        self.seq_row_widget.setVisible(False)
        source_card.add(self.seq_row_widget)

        self.source_status = QLabel("No frames loaded.")
        self.source_status.setObjectName("cardSubtitle")
        source_card.add(self.source_status)
        layout.addWidget(source_card)

        # ------------------------------------------------------------ frame viewer
        viewer_card = Card(
            "Frame Viewer",
            help_html=(
                "<p>Browses the loaded video/image sequence one frame at a time.</p>"
                "<ul>"
                "<li>The slider or spin box jump directly to a frame; <b>&lt;&lt; Previous</b> / "
                "<b>Next &gt;&gt;</b> step by one.</li>"
                "<li>The readout below the frame shows its best-estimate absolute time, once a "
                "Camera Clock File is loaded -- and marks the frame if it's the located trigger "
                "frame.</li>"
                "</ul>"
            ),
        )
        self.frame_label = QLabel("Load a video or image sequence to see frames here.")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setMinimumHeight(360)
        self.frame_label.setStyleSheet(f"background: {theme.BG}; border: 1px solid {theme.BORDER}; border-radius: 6px;")
        self.frame_label.setWordWrap(True)
        viewer_card.add(self.frame_label)

        slider_row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self.frame_slider, 1)
        self.frame_spin = QSpinBox()
        self.frame_spin.setEnabled(False)
        self.frame_spin.setMinimumWidth(90)
        self.frame_spin.valueChanged.connect(self._on_spin_changed)
        slider_row.addWidget(self.frame_spin)
        viewer_card.add_layout(slider_row)

        nav_row = QHBoxLayout()
        self.prev_button = QPushButton("<< Previous")
        self.prev_button.clicked.connect(lambda: self._step_frame(-1))
        nav_row.addWidget(self.prev_button)
        self.next_button = QPushButton("Next >>")
        self.next_button.clicked.connect(lambda: self._step_frame(1))
        nav_row.addWidget(self.next_button)
        nav_row.addStretch(1)
        self.frame_info_label = QLabel("")
        self.frame_info_label.setObjectName("cardSubtitle")
        nav_row.addWidget(self.frame_info_label)
        viewer_card.add_layout(nav_row)
        layout.addWidget(viewer_card)

        # ------------------------------------------------------------ camera clock file
        clock_card = Card(
            "Camera Clock File",
            "Load the TXT/CSV file recorded alongside this camera: it must contain a time vector "
            "and the camera's exposure (\"clock\") signal, one pulse per frame. A trigger-signal "
            "column is optional if you instead know the camera's own trigger frame number and its "
            "recording frequency.",
            help_html=(
                "<p>Loads the TXT/CSV file recorded alongside this camera, and picks which of "
                "its columns mean what.</p>"
                "<ul>"
                "<li><b>Time column</b>: the recording's time vector.</li>"
                "<li><b>Exposure (\"clock\") signal column</b>: one pulse per camera frame -- "
                "this is what detects and counts frames against the time vector.</li>"
                "<li><b>Trigger signal column</b>: optional -- needed only for the \"Use the "
                "trigger signal column\" method below; if you instead know the trigger frame "
                "number directly, leave this at \"(none)\".</li>"
                "<li><b>Camera recording frequency (Hz)</b>: optional hint for the expected "
                "spacing between exposure pulses -- leave blank to estimate it automatically "
                "from the loaded video's own fps or the clock duration.</li>"
                "</ul>"
            ),
        )
        self.clock_file_row = FileSelectRow("Clock file (.txt/.csv)", "Text/CSV files (*.txt *.csv);;All files (*)")
        self.clock_file_row.file_selected.connect(self._on_clock_file_selected)
        clock_card.add(self.clock_file_row)
        self.clock_status = QLabel("No clock file loaded.")
        self.clock_status.setObjectName("cardSubtitle")
        clock_card.add(self.clock_status)

        cols_grid = QGridLayout()
        cols_grid.setHorizontalSpacing(14)
        cols_grid.addWidget(QLabel("Time column"), 0, 0)
        self.time_col_combo = QComboBox()
        cols_grid.addWidget(self.time_col_combo, 0, 1)
        cols_grid.addWidget(QLabel("Exposure (\"clock\") signal column"), 1, 0)
        self.exposure_col_combo = QComboBox()
        cols_grid.addWidget(self.exposure_col_combo, 1, 1)
        cols_grid.addWidget(QLabel("Trigger signal column"), 2, 0)
        self.trigger_col_combo = QComboBox()
        cols_grid.addWidget(self.trigger_col_combo, 2, 1)
        clock_card.add_layout(cols_grid)

        self.frequency_edit = QLineEdit()
        self.frequency_edit.setPlaceholderText("e.g. 20000 (leave blank to estimate automatically)")
        clock_card.add(LabeledField(
            "Camera recording frequency (Hz, optional)", self.frequency_edit,
        ))
        layout.addWidget(clock_card)

        self.time_col_combo.currentIndexChanged.connect(self._on_clock_mapping_inputs_changed)
        self.exposure_col_combo.currentIndexChanged.connect(self._on_clock_mapping_inputs_changed)
        self.frequency_edit.editingFinished.connect(self._on_clock_mapping_inputs_changed)

        # ------------------------------------------------------------ measurement channels
        measurement_card = Card(
            "Measurement Channels",
            "Optionally plot any other recorded channels from the clock file (e.g. strain-gage "
            "signals) against time, each with a vertical line marking the currently displayed frame.",
            help_html=(
                "<p>Tick any column from the loaded clock file (e.g. strain-gage signals) to "
                "plot it against time, below.</p>"
                "<ul>"
                "<li>Each shown plot gets its own vertical line marking the currently displayed "
                "frame's estimated time -- useful for reading off a channel's value at a "
                "specific frame, or lining up the video against a bar-gage signal by eye.</li>"
                "<li>Untick a channel to remove its plot.</li>"
                "</ul>"
            ),
        )
        self.measurement_list = QListWidget()
        self.measurement_list.setMaximumHeight(110)
        self.measurement_list.itemChanged.connect(self._on_measurement_selection_changed)
        measurement_card.add(self.measurement_list)
        self.measurement_plots_container = QWidget()
        self.measurement_plots_layout = QVBoxLayout(self.measurement_plots_container)
        self.measurement_plots_layout.setContentsMargins(0, 0, 0, 0)
        self.measurement_plots_layout.setSpacing(10)
        measurement_card.add(self.measurement_plots_container)
        layout.addWidget(measurement_card)

        # ------------------------------------------------------------ trigger location
        trigger_card = Card(
            "Locate the Trigger Frame",
            help_html=(
                "<p>Finds which video/sequence frame was being exposed at the camera's trigger "
                "instant, and reads off its absolute time (see the plot's own help below for "
                "what's shown).</p>"
                "<ul>"
                "<li><b>Use the trigger signal column</b>: matches the trigger signal's edge "
                "(Camera Clock File card) to the nearest/first/last exposure pulse, per the "
                "\"Match the trigger edge to\" choice.</li>"
                "<li><b>Use a known trigger frame number</b>: instead directly states which "
                "0-based frame index was the trigger, given the camera's recording frequency -- "
                "\"Exposure-pulse offset\" is an advanced adjustment for pulses recorded before "
                "the first saved frame (usually 0).</li>"
                "<li><b>Locate trigger frame</b>: runs the detection and jumps the Frame Viewer "
                "to the result.</li>"
                "</ul>"
            ),
        )
        method_row = QHBoxLayout()
        self.signal_method_radio = QRadioButton("Use the trigger signal column")
        self.manual_method_radio = QRadioButton("Use a known trigger frame number")
        self.signal_method_radio.setChecked(True)
        method_group = QButtonGroup(self)
        method_group.addButton(self.signal_method_radio)
        method_group.addButton(self.manual_method_radio)
        self.signal_method_radio.toggled.connect(self._on_method_toggled)
        method_row.addWidget(self.signal_method_radio)
        method_row.addWidget(self.manual_method_radio)
        method_row.addStretch(1)
        trigger_card.add_layout(method_row)

        match_row = QHBoxLayout()
        match_row.addWidget(QLabel("Match the trigger edge to:"))
        self.match_combo = QComboBox()
        self.match_combo.addItem("Nearest exposure pulse", "nearest")
        self.match_combo.addItem("First exposure pulse at/after the trigger", "after")
        self.match_combo.addItem("Last exposure pulse before the trigger", "before")
        match_row.addWidget(self.match_combo)
        match_row.addStretch(1)
        self.match_row_widget = QWidget()
        self.match_row_widget.setLayout(match_row)
        trigger_card.add(self.match_row_widget)

        self.manual_fields_widget = QWidget()
        manual_layout = QVBoxLayout(self.manual_fields_widget)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(8)
        self.trigger_frame_edit = QLineEdit("0")
        manual_layout.addWidget(LabeledField(
            "Trigger frame number (0-based index into the loaded video/sequence)", self.trigger_frame_edit,
        ))
        self.frame_offset_edit = QLineEdit("0")
        manual_layout.addWidget(LabeledField(
            "Exposure-pulse offset (advanced -- pulses recorded before the first saved frame; usually 0)",
            self.frame_offset_edit,
        ))
        trigger_card.add(self.manual_fields_widget)
        self.manual_fields_widget.setVisible(False)

        run_row = QHBoxLayout()
        self.locate_button = QPushButton("Locate trigger frame")
        self.locate_button.setObjectName("primary")
        self.locate_button.clicked.connect(self._on_locate_trigger)
        run_row.addWidget(self.locate_button)
        run_row.addStretch(1)
        trigger_card.add_layout(run_row)

        self.trigger_result_label = QLabel("")
        self.trigger_result_label.setObjectName("cardSubtitle")
        self.trigger_result_label.setWordWrap(True)
        trigger_card.add(self.trigger_result_label)

        self.clock_canvas = PlotCanvas(
            figsize=(8, 3.5),
            help_title="Locate the Trigger Frame",
            help_text=(
                "<p>Shows the camera's exposure (\"clock\") signal from the loaded clock file, "
                "with every detected exposure pulse marked -- one pulse per camera frame.</p>"
                "<ul>"
                "<li>When a trigger signal column is set, it's overlaid (dashed, right-hand "
                "axis), and the detected trigger edge/frame is marked with a dotted vertical "
                "line -- this is the instant that gets matched to a specific video frame by "
                "\"Locate trigger frame\" above.</li>"
                "<li>A second, lighter vertical line marks whichever frame is currently shown in "
                "the frame viewer, so you can check the timing alignment by eye.</li>"
                "</ul>"
            ),
        )
        trigger_card.add(self.clock_canvas)
        layout.addWidget(trigger_card)

        # ------------------------------------------------------------ export
        export_card = Card(
            "Export Frame-Time Table",
            "Export the absolute time of every detected exposure pulse (one row per camera frame), "
            "for cross-referencing frames against the rest of this test's data.",
            help_html=(
                "<p>Saves a .csv with one row per detected camera frame (frame_index, "
                "absolute_time_s), for cross-referencing video frames against the rest of this "
                "test's data elsewhere.</p>"
                "<p><b>Technical detail:</b> needs \"Locate trigger frame\" to have been run at "
                "least once first, since that step is what detects the per-frame exposure "
                "times.</p>"
            ),
        )
        export_row = QHBoxLayout()
        self.export_button = QPushButton("Export as .csv…")
        self.export_button.clicked.connect(self._on_export)
        export_row.addWidget(self.export_button)
        export_row.addStretch(1)
        export_card.add_layout(export_row)
        self.export_status = QLabel("")
        self.export_status.setObjectName("cardSubtitle")
        export_card.add(self.export_status)
        layout.addWidget(export_card)

        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------ frame source

    def _on_source_mode_toggled(self, _checked=None):
        self.video_row.setVisible(self.video_radio.isChecked())
        self.seq_row_widget.setVisible(self.sequence_radio.isChecked())

    def _on_browse_sequence_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select the image sequence folder")
        if not folder:
            return
        self.seq_folder_edit.setText(folder)
        self._load_source(folder)

    def _load_source(self, path: str):
        if self._source is not None:
            self._source.close()
            self._source = None
        try:
            source = open_frame_source(path)
        except (ValueError, OSError) as exc:
            self.source_status.setText(f"Could not load: {exc}")
            return
        if source.n_frames == 0:
            self.source_status.setText("No frames found.")
            return

        self._source = source
        self._trigger_result = None
        for widget in (self.frame_slider, self.frame_spin):
            widget.setEnabled(True)
        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(source.n_frames - 1)
        self.frame_slider.setValue(0)
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(source.n_frames - 1)
        self.frame_spin.setValue(0)
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)

        fps_text = f", {source.fps:.6g} fps" if source.fps else ""
        self.source_status.setText(
            f"Loaded {source.n_frames} frames{fps_text} from {os.path.basename(path.rstrip('/'))}"
        )
        self._recompute_frame_time_mapping()
        self._show_frame(0)

    def _on_slider_changed(self, value: int):
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(value)
        self.frame_spin.blockSignals(False)
        self._show_frame(value)

    def _on_spin_changed(self, value: int):
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(value)
        self.frame_slider.blockSignals(False)
        self._show_frame(value)

    def _step_frame(self, delta: int):
        if self._source is None:
            return
        new_index = max(0, min(self._source.n_frames - 1, self.frame_slider.value() + delta))
        self.frame_slider.setValue(new_index)

    def _show_frame(self, index: int):
        if self._source is None:
            return
        try:
            frame = self._source.get_frame(index)
        except (ValueError, IndexError) as exc:
            self.frame_info_label.setText(f"Could not read frame {index}: {exc}")
            return

        display = np.ascontiguousarray(to_display_uint8(frame))
        pixmap = _numpy_to_pixmap(display)
        scaled = pixmap.scaled(880, 540, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.frame_label.setPixmap(scaled)

        info = f"Frame {index} / {self._source.n_frames - 1}"
        if isinstance(self._source, ImageSequenceFrameSource):
            info += f"  ({self._source.frame_name(index)})"
        active_time = self._active_frame_time()
        if active_time is not None:
            info += f"  |  time ~= {active_time:.6g} s"
        if self._trigger_result is not None and self._trigger_result.frame_index == index:
            info += "   <-- located trigger frame"
        self.frame_info_label.setText(info)

        self._refresh_measurement_plots()
        if self._trigger_result is not None and self._last_clock_plot_inputs is not None:
            time, exposure, trigger_signal = self._last_clock_plot_inputs
            self._draw_clock_plot(time, exposure, trigger_signal, self._trigger_result)

    # ------------------------------------------------------------------ clock file

    def _on_clock_file_selected(self, path: str):
        try:
            data, header = read_clock_file(path)
        except (OSError, ValueError) as exc:
            self.clock_status.setText(f"Could not read clock file: {exc}")
            return

        self._clock_data = data
        self._clock_header = header
        self._trigger_result = None
        self._last_clock_plot_inputs = None
        self.clock_canvas.clear()
        self.clock_canvas.refresh()

        for combo in (self.time_col_combo, self.exposure_col_combo, self.trigger_col_combo):
            combo.blockSignals(True)
            combo.clear()
        for i, label in enumerate(header):
            self.time_col_combo.addItem(label, i)
            self.exposure_col_combo.addItem(label, i)
            self.trigger_col_combo.addItem(label, i)
        self.trigger_col_combo.addItem("(none)", None)

        self.time_col_combo.setCurrentIndex(0)
        self.exposure_col_combo.setCurrentIndex(min(1, len(header) - 1))
        self.trigger_col_combo.setCurrentIndex(2 if len(header) > 2 else self.trigger_col_combo.count() - 1)
        for combo in (self.time_col_combo, self.exposure_col_combo, self.trigger_col_combo):
            combo.blockSignals(False)

        self.measurement_list.blockSignals(True)
        self.measurement_list.clear()
        for i, label in enumerate(header):
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, i)
            self.measurement_list.addItem(item)
        self.measurement_list.blockSignals(False)
        self._rebuild_measurement_plots()

        self.clock_status.setText(
            f"Loaded {data.shape[0]} samples x {data.shape[1]} columns from {os.path.basename(path)}"
        )
        self._recompute_frame_time_mapping()
        self._refresh_measurement_plots()
        if self._source is not None:
            self._show_frame(self.frame_slider.value())

    def _expected_period_s(self) -> float | None:
        """Best available hint for the camera's frame period, used to set
        the minimum spacing between accepted exposure pulses: the typed-in
        recording frequency, else the loaded video's own fps, else (for an
        image sequence with a clock file already loaded) the clock
        duration divided by the loaded frame count."""
        freq_text = self.frequency_edit.text().strip()
        if freq_text:
            try:
                freq = float(freq_text)
                if freq > 0:
                    return 1.0 / freq
            except ValueError:
                pass
        if self._source is not None and self._source.fps:
            return 1.0 / self._source.fps
        if self._source is not None and self._clock_data is not None and self._source.n_frames > 1:
            time_col = self.time_col_combo.currentData()
            if time_col is not None:
                time = self._clock_data[:, time_col]
                span = float(time[-1] - time[0])
                if span > 0:
                    return span / (self._source.n_frames - 1)
        return None

    def _on_clock_mapping_inputs_changed(self, *_args):
        """The time/exposure column choice or the recording frequency
        changed -- both feed the frame<->time mapping used by the live
        "time ~= ..." readout and every measurement-channel plot's active-
        frame line, so recompute it and redraw."""
        self._recompute_frame_time_mapping()
        self._refresh_measurement_plots()
        if self._source is not None:
            self._show_frame(self.frame_slider.value())

    def _recompute_frame_time_mapping(self):
        """Rebuilds `self._frame_times`: a best-estimate absolute time for
        every frame of the currently loaded video/sequence, from the
        exposure column's detected pulses (extrapolated at a constant
        period beyond the last one detected). Cheap enough to redo whenever
        an input changes, but not on every single frame scrubbed to --
        `_active_frame_time()` just indexes into the cached result."""
        self._frame_times = None
        if self._source is None or self._clock_data is None:
            return
        time_col = self.time_col_combo.currentData()
        exposure_col = self.exposure_col_combo.currentData()
        if time_col is None or exposure_col is None:
            return
        time = self._clock_data[:, time_col]
        exposure = self._clock_data[:, exposure_col]
        expected_period_s = self._expected_period_s()
        _, peak_times = detect_pulse_train(time, exposure, expected_period_s=expected_period_s)
        self._frame_times = estimate_frame_times(peak_times, self._source.n_frames, expected_period_s)

    def _active_frame_time(self) -> float | None:
        """Best-estimate absolute time of the frame currently on screen,
        for the frame-viewer readout and every measurement-channel plot's
        vertical line."""
        if self._frame_times is None:
            return None
        index = self.frame_slider.value()
        if 0 <= index < len(self._frame_times):
            t = self._frame_times[index]
            return float(t) if np.isfinite(t) else None
        return None

    # ------------------------------------------------------------------ measurement channels

    def _on_measurement_selection_changed(self, _item=None):
        self._rebuild_measurement_plots()

    def _rebuild_measurement_plots(self):
        """Adds/removes plots so the shown set matches the checked
        channels -- only the columns that actually changed get a new
        PlotCanvas (each of which opens its own matplotlib Figure), so
        toggling one checkbox doesn't tear down and recreate every other
        channel's plot. A removed canvas has its Figure explicitly closed
        (`close_figure()`), since PlotCanvas creates figures through
        pyplot, which otherwise keeps every one of them alive until
        matplotlib's own "more than 20 figures" warning."""
        checked_columns = {
            self.measurement_list.item(row).data(Qt.UserRole)
            for row in range(self.measurement_list.count())
            if self.measurement_list.item(row).checkState() == Qt.Checked
        }

        for col_index in list(self._measurement_canvases):
            if col_index not in checked_columns:
                canvas = self._measurement_canvases.pop(col_index)
                self.measurement_plots_layout.removeWidget(canvas)
                canvas.setParent(None)
                canvas.close_figure()
                canvas.deleteLater()

        for col_index in checked_columns:
            if col_index not in self._measurement_canvases:
                label = self._clock_header[col_index] if col_index < len(self._clock_header) else f"column {col_index}"
                canvas = PlotCanvas(
                    figsize=(8, 2.6),
                    help_title=f"Measurement channel: {label}",
                    help_text=(
                        f"<p>Plots the \"{label}\" column from the loaded clock file against "
                        "time, with a vertical line marking the currently displayed frame's "
                        "estimated time.</p>"
                        "<p>Lets you read off this channel's value at any frame, e.g. lining up "
                        "a strain-gage signal against the video. Tick/untick channels in the "
                        "\"Measurement Channels\" list above to show or hide their plots.</p>"
                    ),
                )
                self.measurement_plots_layout.addWidget(canvas)
                self._measurement_canvases[col_index] = canvas

        self._refresh_measurement_plots()

    def _refresh_measurement_plots(self):
        """Redraws every currently-shown measurement-channel plot against
        the shared time column, with a vertical line at the currently
        displayed frame's estimated time. Called on every frame change, so
        it only redraws -- it never re-detects peaks or rebuilds canvases."""
        if not self._measurement_canvases:
            return
        if self._clock_data is None:
            for canvas in self._measurement_canvases.values():
                canvas.clear()
                canvas.refresh()
            return

        time_col = self.time_col_combo.currentData()
        active_time = self._active_frame_time()
        for col_index, canvas in self._measurement_canvases.items():
            canvas.clear()
            ax = canvas.figure.add_subplot(111)
            if time_col is not None:
                time = self._clock_data[:, time_col]
                label = (self._clock_header[col_index] if col_index < len(self._clock_header)
                          else f"Column {col_index + 1}")
                plot_time, plot_signal = decimate_for_display(time, self._clock_data[:, col_index])
                ax.plot(plot_time, plot_signal, color=theme.ACCENT, linewidth=1)
                if active_time is not None:
                    ax.axvline(active_time, color=theme.DANGER, linestyle=":", linewidth=1.5,
                               label="Current frame")
                    ax.legend(fontsize=8)
                ax.set_xlabel("time (s)")
                ax.set_ylabel(label)
            canvas.refresh()

    # ------------------------------------------------------------------ trigger location

    def _on_method_toggled(self, _checked=None):
        is_signal = self.signal_method_radio.isChecked()
        self.match_row_widget.setVisible(is_signal)
        self.manual_fields_widget.setVisible(not is_signal)

    def _on_locate_trigger(self):
        if self._clock_data is None:
            self.trigger_result_label.setText("Load a camera clock file first.")
            return
        time_col = self.time_col_combo.currentData()
        exposure_col = self.exposure_col_combo.currentData()
        if time_col is None or exposure_col is None:
            self.trigger_result_label.setText("Choose the time and exposure columns first.")
            return

        time = self._clock_data[:, time_col]
        exposure = self._clock_data[:, exposure_col]
        expected_n_frames = self._source.n_frames if self._source is not None else None
        expected_period_s = self._expected_period_s()
        trigger_signal = None

        if self.signal_method_radio.isChecked():
            trigger_col = self.trigger_col_combo.currentData()
            if trigger_col is None:
                self.trigger_result_label.setText(
                    "Choose a trigger signal column, or switch to the trigger-frame-number method."
                )
                return
            trigger_signal = self._clock_data[:, trigger_col]
            result = locate_trigger_by_signal(
                time, exposure, trigger_signal,
                expected_period_s=expected_period_s, expected_n_frames=expected_n_frames,
                match=self.match_combo.currentData(),
            )
        else:
            try:
                frame_number = int(float(self.trigger_frame_edit.text()))
            except ValueError:
                self.trigger_result_label.setText("Enter a valid (numeric) trigger frame number.")
                return
            try:
                frame_offset = int(float(self.frame_offset_edit.text() or "0"))
            except ValueError:
                frame_offset = 0
            freq_text = self.frequency_edit.text().strip()
            if not freq_text:
                self.trigger_result_label.setText(
                    "This method needs the camera's recording frequency (Hz), above."
                )
                return
            try:
                frequency_hz = float(freq_text)
            except ValueError:
                self.trigger_result_label.setText("Recording frequency must be a number.")
                return
            result = locate_trigger_by_frame_number(
                time, exposure, frame_number, frequency_hz, frame_offset=frame_offset,
                expected_period_s=expected_period_s, expected_n_frames=expected_n_frames,
            )

        self._trigger_result = result
        self._last_clock_plot_inputs = (time, exposure, trigger_signal)
        self._draw_clock_plot(time, exposure, trigger_signal, result)

        lines = [f"Detected {result.n_peaks_detected} exposure pulses (method: {result.method})."]
        if result.frame_index is not None:
            lines.append(f"Trigger frame: #{result.frame_index}")
        if result.frame_time is not None:
            lines.append(f"Absolute time (from the time vector): {result.frame_time:.6g} s")
        if result.trigger_time is not None:
            lines.append(f"Raw trigger-signal edge time: {result.trigger_time:.6g} s")
        for warning in result.warnings:
            lines.append(f"Warning: {warning}")
        self.trigger_result_label.setText("\n".join(lines))

        if (result.frame_index is not None and self._source is not None
                and 0 <= result.frame_index < self._source.n_frames):
            self.frame_slider.setValue(result.frame_index)
        elif result.frame_index is not None:
            self._show_frame(self.frame_slider.value())  # at least refresh the "<-- located" annotation attempt

    def _draw_clock_plot(self, time, exposure, trigger, result):
        self.clock_canvas.clear()
        ax = self.clock_canvas.figure.add_subplot(111)
        plot_time, plot_exposure = decimate_for_display(time, exposure)
        ax.plot(plot_time, plot_exposure, color=theme.ACCENT, linewidth=1, label="Exposure (\"clock\") signal")
        if len(result.peak_sample_indices):
            ax.plot(result.peak_times, exposure[result.peak_sample_indices], "o",
                    color=theme.SUCCESS, markersize=3, label="Detected exposure pulses")

        handles, labels = ax.get_legend_handles_labels()
        if trigger is not None:
            ax2 = ax.twinx()
            plot_time2, plot_trigger = decimate_for_display(time, trigger)
            ax2.plot(plot_time2, plot_trigger, color=theme.DANGER, linewidth=1, linestyle="--", label="Trigger signal")
            ax2.set_ylabel("Trigger signal")
            h2, l2 = ax2.get_legend_handles_labels()
            handles, labels = handles + h2, labels + l2

        marker_time = result.trigger_time if result.trigger_time is not None else result.frame_time
        if marker_time is not None:
            line = ax.axvline(marker_time, color=theme.DANGER, linestyle=":", linewidth=1.5)
            handles.append(line)
            labels.append("Trigger" if result.trigger_time is not None else "Trigger frame time")

        active_time = self._active_frame_time()
        if active_time is not None:
            line = ax.axvline(active_time, color=theme.TEXT_MUTED, linestyle="--", linewidth=1.2)
            handles.append(line)
            labels.append("Current frame")

        ax.set_xlabel("time (s)")
        ax.set_ylabel("Exposure signal")
        ax.legend(handles, labels, fontsize=8)
        self.clock_canvas.refresh()

    # ------------------------------------------------------------------ export

    def _on_export(self):
        result = self._trigger_result
        if result is None or result.n_peaks_detected == 0:
            self.export_status.setText("Locate the trigger frame first (this also detects the per-frame exposure times).")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export frame-time table", "frame_times.csv", "CSV files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_index", "absolute_time_s"])
                for i, t in enumerate(result.peak_times):
                    writer.writerow([i, f"{t:.8g}"])
        except OSError as exc:
            self.export_status.setText(f"Could not export: {exc}")
            return
        self.export_status.setText(f"Exported {len(result.peak_times)} rows to {os.path.basename(path)}")


def _numpy_to_pixmap(array: np.ndarray) -> QPixmap:
    if array.ndim == 2:
        h, w = array.shape
        qimage = QImage(array.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, _ = array.shape
        qimage = QImage(array.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())  # copy: detach from `array`'s buffer before it goes out of scope
