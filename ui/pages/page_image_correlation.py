"""
Digital Image Correlation page.

Part of the main pipeline: load a high-speed video (MP4/MKV/AVI) or an
image sequence (TIFF 8/16-bit, JPEG, PNG, BMP) of the specimen during the
test, pick the reference frame, optionally crop/downscale/decimate it for
speed, draw the region of interest (ROI) and correlation mesh (subset
size / step, both in pixels), then run Digital Image Correlation (DIC):
sequential subset tracking (normalized cross-correlation template
matching, core/dic_processing.py) across the processed frames, producing
per-point displacement and a local strain-field estimate.

Reuses the same frame-loading abstraction (core/frame_source.py) as the
standalone Camera / Video Sync tool, and the same worker-thread +
QProgressBar pattern used by the Dispersion Corrections page
(ui/pages/page_corrections.py) for its own long root-search computation.
"""
from __future__ import annotations

import csv
import os
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QRadioButton, QButtonGroup, QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QProgressBar,
    QFileDialog, QDialog, QInputDialog, QCheckBox,
)

from core.frame_source import open_frame_source, to_display_uint8, ImageSequenceFrameSource
from core.dic_preprocessing import preprocess_frame, decimated_frame_indices
from core.dic_processing import run_dic
from core.parallel import OperationCancelled
from core.dic_kinematics import (
    compute_kinematics, smooth_series, nominal_to_true_strain, nominal_to_true_strain_rate,
)
from core.dic_fields import smooth_nan_gaussian, principal_strains, max_shear_strain
from core.dic_kinematics import von_mises_strain
from core.dic_uncertainty import compute_uncertainty, roi_mean_uncertainty, effective_independent_count
from core.dic_result_io import load_dic_result_csv, DicResultLoadError
from core.dic_render import render_field, default_max_edge_px
from ui.frame_popup import make_popout_button
from core.dic_video_export import export_field_video
from ui.widgets import Card, FileSelectRow, LabeledField, make_scroll_page
from ui.roi_picker import RectPicker, RoiMeshPicker
from ui.ruler_dialog import RulerDialog
from ui.plot_canvas import PlotCanvas
from ui import theme

DEFAULT_SUBSET_SIZE = 21
DEFAULT_STEP = 7

# (key, combo-box label) -- the label shown in the Results field picker;
# the unit shown in the plot/colorbar is resolved at display time from
# the calibration fields (px, or mm once a pixel size is given).
FIELD_CHOICES = [
    ("disp_mag", "Displacement magnitude"),
    ("u", "u -- horizontal displacement"),
    ("v", "v -- vertical displacement"),
    ("exx", "exx -- horizontal strain"),
    ("eyy", "eyy -- vertical strain"),
    ("exy", "exy -- shear strain"),
    # Orientation-independent measures (invariants of the in-plane strain
    # tensor, core/dic_fields.py): they follow a crack/band whatever its
    # direction, where exx/eyy only see its horizontal/vertical part.
    ("e1", "e1 -- max principal strain"),
    ("e2", "e2 -- min principal strain"),
    ("gmax", "gmax -- max shear strain (e1 - e2)/2"),
    ("eps_eq", "Equivalent strain (von Mises)"),
    ("v_mag", "Velocity magnitude"),
    ("vx", "vx -- horizontal velocity"),
    ("vy", "vy -- vertical velocity"),
    ("a_mag", "Acceleration magnitude"),
    ("ax", "ax -- horizontal acceleration"),
    ("ay", "ay -- vertical acceleration"),
    ("exx_rate", "exx rate -- horizontal strain rate"),
    ("eyy_rate", "eyy rate -- vertical strain rate"),
    ("exy_rate", "exy rate -- shear strain rate"),
    ("eps_eq_rate", "Equivalent strain rate (von Mises)"),
    ("dynamic_force", "Dynamic Force (F = rho x A x a)"),
    ("correlation", "Correlation coefficient"),
]

# (key, combo-box label) -- which computed strain definition is
# published as "DIC Strain (-)" (and "DIC Strain Rate (1/s)") to the
# Results & Export page; see _build_dic_strain_export_card /
# _publish_dic_strain and Project.dic_strain_result.
DIC_STRAIN_DEFINITIONS = [
    ("eps_eq", "MISES (equivalent strain)"),
    ("exx", "exx -- horizontal strain"),
    ("eyy", "eyy -- vertical strain"),
    ("exy", "exy -- shear strain"),
]
# Short form of the same definitions, for compact status text/plot
# legends (e.g. "DIC Strain (MISES)") rather than the combo's longer
# descriptive label.
_DIC_STRAIN_SHORT_NAMES = {"eps_eq": "MISES", "exx": "exx", "eyy": "eyy", "exy": "exy"}

RESULT_CMAP = "jet"


class _DICWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, get_frame_fn, mesh, frame_indices, search_radius, tracking_mode="fixed",
                 correlation_threshold=0.0, max_neighbor_deviation_px=0.0, n_workers=1,
                 cancel_event: threading.Event | None = None, subpixel_method="parabolic"):
        super().__init__()
        self.get_frame_fn = get_frame_fn
        self.mesh = mesh
        self.frame_indices = frame_indices
        self.search_radius = search_radius
        self.tracking_mode = tracking_mode
        self.correlation_threshold = correlation_threshold
        self.max_neighbor_deviation_px = max_neighbor_deviation_px
        self.n_workers = n_workers
        self.subpixel_method = subpixel_method
        # Set by the page's "Stop" button (from the UI thread) -- checked
        # once per frame inside `run_dic` (`cancel_check`, see
        # core.parallel.check_cancelled). A plain threading.Event is
        # thread-safe to read/write across the UI/worker thread boundary
        # without any extra locking.
        self.cancel_event = cancel_event

    def run(self):
        try:
            result = run_dic(
                self.get_frame_fn, self.mesh, self.frame_indices, self.search_radius,
                tracking_mode=self.tracking_mode, correlation_threshold=self.correlation_threshold,
                max_neighbor_deviation_px=self.max_neighbor_deviation_px,
                progress_cb=lambda i, total: self.progress.emit(i, total),
                n_workers=self.n_workers, subpixel_method=self.subpixel_method,
                cancel_check=(self.cancel_event.is_set if self.cancel_event is not None else None),
            )
            self.finished.emit(result)
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - report inline, never a modal error popup
            self.failed.emit(str(exc))


class _VideoExportWorker(QObject):
    """Renders and encodes the active field's video in a background
    thread -- reading/preprocessing each background frame (`get_background_fn`,
    a plain closure over the frame source, same pattern as `_DICWorker`'s
    `get_frame_fn`) and the matplotlib rendering (core/dic_video_export.py,
    entirely off the Qt widgets) are both potentially slow for a long
    sequence, so neither should block the UI."""

    progress = Signal(int, int)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, path, get_background_fn, frame_indices, x_list, y_list, values_list,
                 timestamps_list, field_label, vmin, vmax, fps,
                 mean_series=None, time_series=None, time_label=None, mean_vmin=None, mean_vmax=None,
                 max_edge_px=None):
        super().__init__()
        self.path = path
        self.get_background_fn = get_background_fn
        self.frame_indices = frame_indices
        self.x_list = x_list
        self.y_list = y_list
        self.values_list = values_list
        self.timestamps_list = timestamps_list
        self.field_label = field_label
        self.vmin = vmin
        self.vmax = vmax
        self.fps = fps
        # See core/dic_render.py's default_max_edge_px -- masks out any
        # triangle bridging a non-rectangular ROI's excluded notch/hole,
        # same fix as the live Results canvas (_on_result_view_changed).
        self.max_edge_px = max_edge_px
        # A live ROI-mean-vs-time line drawn in a second panel next to
        # the isocontour, growing frame by frame -- see
        # core/dic_video_export.render_field_frame. None (the default)
        # keeps the single-panel layout exactly as before.
        self.mean_series = mean_series
        self.time_series = time_series
        self.time_label = time_label
        self.mean_vmin = mean_vmin
        self.mean_vmax = mean_vmax

    def run(self):
        try:
            def frames():
                for i, (frame_index, x, y, values, timestamp_text) in enumerate(zip(
                        self.frame_indices, self.x_list, self.y_list, self.values_list,
                        self.timestamps_list)):
                    yield dict(
                        background=self.get_background_fn(frame_index), x=x, y=y,
                        values=values, field_label=self.field_label,
                        vmin=self.vmin, vmax=self.vmax, cmap=RESULT_CMAP,
                        timestamp_text=timestamp_text,
                        mean_series=self.mean_series, time_series=self.time_series,
                        current_index=i, time_label=self.time_label,
                        mean_vmin=self.mean_vmin, mean_vmax=self.mean_vmax,
                        max_edge_px=self.max_edge_px,
                    )

            export_field_video(
                self.path, frames(), len(self.frame_indices), self.fps,
                progress_cb=lambda i, total: self.progress.emit(i, total),
            )
            self.finished.emit(self.path)
        except Exception as exc:  # noqa: BLE001 - report inline, never a modal error popup
            self.failed.emit(str(exc))


class ImageCorrelationPage(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._source = None      # a FrameSource (video or image sequence)
        self._dic_result = None  # last completed DICResult
        self._kinematics = None  # cached core.dic_kinematics.Kinematics for self._dic_result, see _ensure_kinematics
        self._uncertainty = None  # cached core.dic_uncertainty.DicUncertainty for self._dic_result, see _ensure_uncertainty
        self._last_result_field = None  # {"x", "y", "values", "label"} for the Results canvas's cursor readout, see _result_cursor_value
        self._thread = None
        self._worker = None
        self._cancel_event = None  # threading.Event for the current DIC run, or None between runs
        self._video_thread = None
        self._video_worker = None

        scroll, layout = make_scroll_page(
            "Digital Image Correlation",
            "Load a high-speed video or an image sequence of the specimen, crop/downscale/"
            "decimate it for speed, draw the region of interest and correlation mesh, then run "
            "Digital Image Correlation (DIC) to track displacement and strain over time.",
        )

        self._build_load_frames_card(layout)
        self._build_calibration_card(layout)
        self._build_reference_preprocessing_card(layout)
        self._build_roi_mesh_card(layout)
        self._build_run_card(layout)
        self._build_dic_strain_export_card(layout)
        self._build_uncertainty_card(layout)
        self._build_probes_card(layout)
        self._build_results_card(layout)
        self._build_mean_vs_time_card(layout)
        self._build_velocity_card(layout)
        self._build_strain_rate_card(layout)
        self._build_force_card(layout)

        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._invalidate_results()
        self._restore_from_session()

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_load_frames_card(self, layout):
        source_card = Card(
            "Load Frames",
            "Choose a video file (.mp4 / .mkv / .avi), or a folder of single-frame images "
            "(TIFF 8/16-bit, JPEG/JPG, PNG, BMP), naturally sorted by file name.",
            help_html=(
                "<p>Loads the footage of the specimen during the test, as either a single video "
                "file or a folder of individual image frames.</p>"
                "<ul>"
                "<li><b>Video file</b>: .mp4, .mkv or .avi.</li>"
                "<li><b>Image sequence (folder)</b>: a folder of single-frame images (TIFF "
                "8/16-bit, JPEG/JPG, PNG, BMP), read in natural (numeric-aware) file-name order.</li>"
                "</ul>"
                "<p>Once loaded, the frame count and rate/name are shown below; scrub through "
                "frames in the \"Reference Frame &amp; Preprocessing\" card further down.</p>"
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
        self.seq_browse_button = QPushButton("Browse…")
        self.seq_browse_button.clicked.connect(self._on_browse_sequence_folder)
        seq_row.addWidget(self.seq_browse_button)
        self.seq_row_widget = QWidget()
        self.seq_row_widget.setLayout(seq_row)
        self.seq_row_widget.setVisible(False)
        source_card.add(self.seq_row_widget)

        self.source_status = QLabel("No frames loaded.")
        self.source_status.setObjectName("cardSubtitle")
        source_card.add(self.source_status)
        layout.addWidget(source_card)

    def _build_calibration_card(self, layout):
        card = Card(
            "Calibration",
            "Optional, but needed for physical units: how much real-world distance one pixel of "
            "the RAW frame covers, and the camera's frame rate -- used by the ROI-mean-vs-time "
            "plot's time axis below, and already accounts for the frame step.",
            help_html=(
                "<p>Optional, but required to see results in physical units instead of raw pixels "
                "and frame numbers.</p>"
                "<ul>"
                "<li><b>Pixel size</b>: real-world distance covered by one pixel of the RAW "
                "(un-cropped, un-downscaled) frame -- leave blank to keep every result in "
                "pixels. <b>Measure scale on reference frame…</b> lets you click two points a "
                "known distance apart on the image instead of typing a number.</li>"
                "<li><b>Camera frame rate</b>: leave blank to plot against frame index instead of "
                "time; already accounts for the \"Frame step\" set below, so it's always the true "
                "physical frame rate of the recording, not the processed-frame rate.</li>"
                "<li><b>Material density / Specimen width &amp; thickness</b>: only needed for the "
                "Dynamic Force field/plot (F = &rho; &times; A &times; a).</li>"
                "</ul>"
            ),
        )
        self.pixel_size_edit = QLineEdit()
        self.pixel_size_edit.setPlaceholderText("e.g. 0.05 (leave blank to keep results in pixels)")
        self.pixel_size_edit.editingFinished.connect(self._on_calibration_changed)
        card.add(LabeledField("Pixel size (mm per raw-frame pixel)", self.pixel_size_edit))

        measure_row = QHBoxLayout()
        self.measure_scale_button = QPushButton("Measure scale on reference frame…")
        self.measure_scale_button.clicked.connect(self._on_measure_scale)
        measure_row.addWidget(self.measure_scale_button)
        measure_row.addStretch(1)
        card.add_layout(measure_row)

        self.frame_rate_edit = QLineEdit()
        self.frame_rate_edit.setPlaceholderText("e.g. 50000 (leave blank to plot against frame index)")
        self.frame_rate_edit.editingFinished.connect(self._on_calibration_changed)
        card.add(LabeledField("Camera frame rate (fps)", self.frame_rate_edit))

        self.density_edit = QLineEdit()
        self.density_edit.setPlaceholderText("e.g. 7800 (needed only for the Dynamic Force field/plot)")
        self.density_edit.editingFinished.connect(self._on_density_changed)
        card.add(LabeledField("Material density rho (kg/m^3)", self.density_edit))

        self.specimen_width_edit = QLineEdit()
        self.specimen_width_edit.setPlaceholderText("e.g. 10 (specimen cross-section, needed only for Dynamic Force)")
        self.specimen_width_edit.editingFinished.connect(self._on_density_changed)
        card.add(LabeledField("Specimen width w (mm)", self.specimen_width_edit))

        self.specimen_thickness_edit = QLineEdit()
        self.specimen_thickness_edit.setPlaceholderText("e.g. 2 (specimen cross-section, needed only for Dynamic Force)")
        self.specimen_thickness_edit.editingFinished.connect(self._on_density_changed)
        card.add(LabeledField("Specimen thickness e (mm)", self.specimen_thickness_edit))
        layout.addWidget(card)

    def _build_reference_preprocessing_card(self, layout):
        card = Card(
            "Reference Frame & Preprocessing",
            "Scrub to the frame you want to use as the DIC reference. Optionally crop it to the "
            "area that matters, reduce its resolution, and/or process only every Nth frame -- all "
            "three cut the work a correlation run has to do.",
            help_html=(
                "<p>Picks the DIC reference frame -- the undeformed starting point every later "
                "frame is compared against -- and any preprocessing to speed up correlation.</p>"
                "<ul>"
                "<li>Use the slider/spinbox to scrub to the frame to use as reference, typically "
                "the last frame before the specimen starts deforming.</li>"
                "<li><b>Crop region</b>: click and drag on the image below to process only the "
                "area that matters (e.g. just the specimen, not the surrounding fixtures).</li>"
                "<li><b>Downscale factor</b>: 1 = full resolution, 2 = half resolution, etc. -- "
                "trades tracking precision for speed.</li>"
                "<li><b>Frame step (1 every N)</b>: process only every Nth frame instead of all "
                "of them.</li>"
                "</ul>"
                "<p><b>Technical detail:</b> all three (crop, downscale, frame step) reduce how "
                "much work a correlation run has to do, without changing what DIC itself measures "
                "-- only how finely it samples space/time.</p>"
            ),
        )
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
        card.add_layout(slider_row)
        self.frame_info_label = QLabel("")
        self.frame_info_label.setObjectName("cardSubtitle")
        card.add(self.frame_info_label)

        self.crop_picker = RectPicker(
            figsize=(8, 4.5), region_label="crop region",
            help_title="Reference frame",
            help_text=(
                "<p>Shows the reference frame picked above. Click and drag to set the crop "
                "region -- only this area of every frame is processed from here on.</p>"
                "<ul>"
                "<li>The X/Y/Width/Height fields below the image let you type an exact crop "
                "instead of dragging.</li>"
                "<li><b>Clear crop region</b> resets to the full frame; <b>Reset view</b> resets "
                "zoom/pan (not the crop itself).</li>"
                "</ul>"
            ),
        )
        self.crop_picker.rectChanged.connect(self._on_crop_changed)
        card.add(self.crop_picker)

        crop_fields_row = QHBoxLayout()
        self.crop_x_spin = self._make_spin(0, 1_000_000, self._on_crop_fields_changed)
        self.crop_y_spin = self._make_spin(0, 1_000_000, self._on_crop_fields_changed)
        self.crop_w_spin = self._make_spin(0, 1_000_000, self._on_crop_fields_changed)
        self.crop_h_spin = self._make_spin(0, 1_000_000, self._on_crop_fields_changed)
        for label, spin in (("X", self.crop_x_spin), ("Y", self.crop_y_spin),
                             ("Width", self.crop_w_spin), ("Height", self.crop_h_spin)):
            crop_fields_row.addWidget(QLabel(label))
            crop_fields_row.addWidget(spin)
        crop_fields_row.addStretch(1)
        card.add_layout(crop_fields_row)

        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Downscale factor"))
        self.downscale_spin = QSpinBox()
        self.downscale_spin.setRange(1, 16)
        self.downscale_spin.setValue(1)
        self.downscale_spin.setToolTip("1 = full resolution, 2 = half resolution, etc.")
        self.downscale_spin.valueChanged.connect(self._on_downscale_changed)
        params_row.addWidget(self.downscale_spin)
        params_row.addSpacing(20)
        params_row.addWidget(QLabel("Frame step (1 every N)"))
        self.frame_step_spin = QSpinBox()
        self.frame_step_spin.setRange(1, 1000)
        self.frame_step_spin.setValue(1)
        self.frame_step_spin.valueChanged.connect(self._on_frame_step_changed)
        params_row.addWidget(self.frame_step_spin)
        params_row.addStretch(1)
        card.add_layout(params_row)

        self.preprocessing_summary_label = QLabel("")
        self.preprocessing_summary_label.setObjectName("cardSubtitle")
        self.preprocessing_summary_label.setWordWrap(True)
        card.add(self.preprocessing_summary_label)
        layout.addWidget(card)

    def _build_roi_mesh_card(self, layout):
        card = Card(
            "Region of Interest & Correlation Mesh",
            "Click and drag on the (preprocessed) frame below to draw a rectangular or oval ROI "
            "-- or switch to Polygon and click each vertex in turn (close the shape by clicking "
            "the first vertex again, or pressing Enter) for a more complex specimen outline. "
            "Subset size is the side length (in pixels) of each correlation window (\"facet\"); "
            "step is the pixel spacing between neighboring mesh points. A smaller step gives a "
            "denser field at higher computational cost; the subset should stay large enough to "
            "contain enough speckle/texture contrast to correlate reliably.\n\n"
            "Right-click anywhere on the same image to place a probe at a precise point -- see "
            "the Probes section below Run Correlation for their evolution plot. After a "
            "correlation run, each probe reads whatever field is tracked by its NEAREST "
            "correlation mesh point, letting you compare how a quantity evolves at a handful of "
            "chosen spots -- e.g. inside a neck vs. away from it -- instead of only the "
            "ROI-wide mean.",
            help_html=(
                "<p>Draws the Region Of Interest (ROI) -- the area DIC actually tracks -- and its "
                "correlation mesh, on the preprocessed reference frame.</p>"
                "<ul>"
                "<li><b>ROI shape</b>: Rectangle or Oval (click and drag the bounding box; for "
                "Oval, the ROI is the ellipse inscribed in it), or Polygon (click each vertex, "
                "close by clicking the first vertex again or pressing Enter) for an irregular "
                "specimen outline.</li>"
                "<li><b>Subset size (px)</b>: the side length of each correlation window "
                "(\"facet\") -- must stay large enough to contain enough speckle/texture contrast "
                "to correlate reliably.</li>"
                "<li><b>Step (px)</b>: pixel spacing between neighboring mesh points -- a smaller "
                "step gives a denser field at higher computational cost.</li>"
                "<li>Right-click anywhere on the image to place a <b>probe</b> at a precise point "
                "(see the Probes card below) -- independent of the ROI shape/mesh.</li>"
                "</ul>"
            ),
        )
        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("ROI shape"))
        self.roi_rect_radio = QRadioButton("Rectangle")
        self.roi_oval_radio = QRadioButton("Oval")
        self.roi_polygon_radio = QRadioButton("Polygon")
        self.roi_rect_radio.setChecked(True)
        roi_shape_group = QButtonGroup(self)
        roi_shape_group.addButton(self.roi_rect_radio)
        roi_shape_group.addButton(self.roi_oval_radio)
        roi_shape_group.addButton(self.roi_polygon_radio)
        self.roi_rect_radio.toggled.connect(self._on_roi_shape_toggled)
        self.roi_oval_radio.toggled.connect(self._on_roi_shape_toggled)
        self.roi_polygon_radio.toggled.connect(self._on_roi_shape_toggled)
        shape_row.addWidget(self.roi_rect_radio)
        shape_row.addWidget(self.roi_oval_radio)
        shape_row.addWidget(self.roi_polygon_radio)
        shape_row.addStretch(1)
        card.add_layout(shape_row)

        self.roi_picker = RoiMeshPicker(
            figsize=(8, 5.5),
            help_title="Region of Interest & Correlation Mesh",
            help_text=(
                "<p>Shows the reference frame with the current ROI and a preview of the "
                "correlation mesh (a marker at every point the current Subset size / Step would "
                "track).</p>"
                "<ul>"
                "<li>Click and drag to set a Rectangle/Oval ROI, or click vertices for a Polygon "
                "(see the ROI shape switch above the plot).</li>"
                "<li>Right-click to place a probe; use <b>Remove last probe</b> / <b>Clear "
                "probes</b> below the plot to undo.</li>"
                "</ul>"
            ),
        )
        self.roi_picker.regionChanged.connect(self._on_roi_region_changed)
        self.roi_picker.probesChanged.connect(self._on_probes_changed)
        card.add(self.roi_picker)

        roi_fields_row = QHBoxLayout()
        self.roi_x_spin = self._make_spin(0, 1_000_000, self._on_roi_fields_changed)
        self.roi_y_spin = self._make_spin(0, 1_000_000, self._on_roi_fields_changed)
        self.roi_w_spin = self._make_spin(0, 1_000_000, self._on_roi_fields_changed)
        self.roi_h_spin = self._make_spin(0, 1_000_000, self._on_roi_fields_changed)
        for label, spin in (("X", self.roi_x_spin), ("Y", self.roi_y_spin),
                             ("Width", self.roi_w_spin), ("Height", self.roi_h_spin)):
            roi_fields_row.addWidget(QLabel(label))
            roi_fields_row.addWidget(spin)
        roi_fields_row.addStretch(1)
        self.roi_fields_widget = QWidget()
        self.roi_fields_widget.setLayout(roi_fields_row)
        card.add(self.roi_fields_widget)
        self.roi_polygon_hint_label = QLabel(
            "Polygon ROI: drawn directly on the image above -- exact pixel bounds don't apply."
        )
        self.roi_polygon_hint_label.setObjectName("cardSubtitle")
        self.roi_polygon_hint_label.setVisible(False)
        card.add(self.roi_polygon_hint_label)
        self.roi_oval_hint_label = QLabel(
            "Oval ROI: click and drag on the image above to set the bounding box -- the ellipse "
            "inscribed in it is used as the ROI."
        )
        self.roi_oval_hint_label.setObjectName("cardSubtitle")
        self.roi_oval_hint_label.setVisible(False)
        card.add(self.roi_oval_hint_label)

        mesh_row = QHBoxLayout()
        mesh_row.addWidget(QLabel("Subset size (px)"))
        self.subset_spin = QSpinBox()
        self.subset_spin.setRange(3, 2000)
        self.subset_spin.setSingleStep(2)
        self.subset_spin.setValue(DEFAULT_SUBSET_SIZE)
        self.subset_spin.valueChanged.connect(self._on_mesh_params_changed)
        mesh_row.addWidget(self.subset_spin)
        mesh_row.addSpacing(20)
        mesh_row.addWidget(QLabel("Step (px)"))
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 2000)
        self.step_spin.setValue(DEFAULT_STEP)
        self.step_spin.valueChanged.connect(self._on_mesh_params_changed)
        mesh_row.addWidget(self.step_spin)
        mesh_row.addStretch(1)
        card.add_layout(mesh_row)
        self.mesh_status_label = QLabel("Draw a ROI above to preview the correlation mesh.")
        self.mesh_status_label.setObjectName("cardSubtitle")
        card.add(self.mesh_status_label)
        layout.addWidget(card)

    def _build_probes_card(self, layout):
        card = Card(
            "Probes",
            "Right-click on the image in the Region of Interest & Correlation Mesh section "
            "above to place a probe at a precise point (the same right-click gesture works in "
            "any ROI shape mode). After a correlation run, each probe reads whatever field is "
            "tracked by its NEAREST correlation mesh point (see the Field picker below), "
            "letting you compare how a quantity evolves at a handful of chosen spots -- e.g. "
            "inside a neck vs. away from it -- instead of only the ROI-wide mean.",
            help_html=(
                "<p>Tracks specific chosen points, separately from the ROI-wide mean shown "
                "elsewhere.</p>"
                "<ul>"
                "<li>Right-click anywhere on the image in the ROI &amp; Correlation Mesh card "
                "above to place a probe (works in any ROI shape mode).</li>"
                "<li>After a correlation run, each probe reads whatever field is tracked by its "
                "NEAREST correlation mesh point -- pick which field with the <b>Field</b> dropdown "
                "here.</li>"
                "<li>Useful for comparing how a quantity evolves at a handful of chosen spots, "
                "e.g. inside a neck vs. away from it.</li>"
                "</ul>"
            ),
        )
        field_row = QHBoxLayout()
        field_row.addWidget(QLabel("Field"))
        self.probe_field_combo = QComboBox()
        for key, label in FIELD_CHOICES:
            self.probe_field_combo.addItem(label, key)
        self.probe_field_combo.currentIndexChanged.connect(self._on_probe_field_changed)
        field_row.addWidget(self.probe_field_combo)
        field_row.addStretch(1)
        card.add_layout(field_row)

        self.probe_canvas = PlotCanvas(
            figsize=(8, 4.5),
            help_title="Probes",
            help_text=(
                "<p>Plots the selected Field over time/frame at each probe placed on the ROI "
                "image above -- one curve per probe, reading the nearest correlation mesh point's "
                "tracked value.</p>"
                "<p>Populated once a correlation run has completed; <b>Export all probes "
                "(.csv)…</b> saves every probe's curve.</p>"
            ),
        )
        card.add(self.probe_canvas)

        export_row = QHBoxLayout()
        self.export_probes_button = QPushButton("Export all probes (.csv)…")
        self.export_probes_button.setEnabled(False)
        self.export_probes_button.clicked.connect(self._on_export_probes)
        export_row.addWidget(self.export_probes_button)
        export_row.addStretch(1)
        card.add_layout(export_row)
        layout.addWidget(card)

    def _build_run_card(self, layout):
        card = Card(
            "Run Correlation",
            "Tracks every mesh point across the processed frames using normalized cross-"
            "correlation template matching. Each subset is tracked as a rigid, translating "
            "window (see core/dic_processing.py) -- not a full deformable-subset solver, but "
            "enough to recover displacement and a local strain estimate for moderate deformations.",
            help_html=(
                "<p>Tracks every correlation-mesh point across the processed frames using "
                "normalized cross-correlation template matching, then click <b>Run correlation</b> "
                "to start.</p>"
                "<ul>"
                "<li><b>Search radius</b>: how far, in pixels beyond the subset itself, to search "
                "around each point's predicted position in every frame.</li>"
                "<li><b>Point tracking -- Fixed</b>: each point's search window stays centered on "
                "its reference position (search radius alone must cover the point's total "
                "displacement). <b>Moving</b>: each frame's search window recenters on the point's "
                "last found position -- follows large cumulative motion with a small search "
                "radius, at the cost of a bad match shifting every later frame's search too.</li>"
                "<li><b>Minimum correlation coefficient</b>: 0 keeps every point regardless of "
                "match quality; above 0, a point is dropped for the rest of the sequence the first "
                "time its match falls below this value -- useful once necking pushes points off "
                "the specimen and onto the background.</li>"
                "<li><b>Max neighbor displacement deviation</b>: 0 disables it; above 0, a point "
                "is dropped if its displacement disagrees with its immediate neighbors' median by "
                "more than this many pixels -- catches a single point that tracked to an "
                "implausible position and would otherwise distort the field.</li>"
                "</ul>"
                "<p><b>Technical detail:</b> each subset is tracked as a rigid, translating window "
                "-- not a full deformable-subset solver, but enough to recover displacement and a "
                "local strain estimate for moderate deformations. The number of worker cores used "
                "is set on the Test Setup page's Performance (multi-core) card.</p>"
            ),
        )
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search radius (px)"))
        self.search_radius_spin = QSpinBox()
        self.search_radius_spin.setRange(1, 500)
        self.search_radius_spin.setValue(15)
        self.search_radius_spin.setToolTip(
            "How far (in pixels, beyond the subset itself) to search around each point's "
            "predicted position in every processed frame."
        )
        self.search_radius_spin.valueChanged.connect(self._on_search_radius_changed)
        search_row.addWidget(self.search_radius_spin)
        search_row.addStretch(1)
        card.add_layout(search_row)

        tracking_row = QHBoxLayout()
        tracking_row.addWidget(QLabel("Point tracking"))
        self.tracking_fixed_radio = QRadioButton("Fixed (search around the reference position)")
        self.tracking_moving_radio = QRadioButton("Moving (follows the speckle pattern)")
        self.tracking_fixed_radio.setChecked(True)
        tracking_mode_group = QButtonGroup(self)
        tracking_mode_group.addButton(self.tracking_fixed_radio)
        tracking_mode_group.addButton(self.tracking_moving_radio)
        self.tracking_fixed_radio.toggled.connect(self._on_tracking_mode_toggled)
        tracking_row.addWidget(self.tracking_fixed_radio)
        tracking_row.addWidget(self.tracking_moving_radio)
        tracking_row.addStretch(1)
        card.add_layout(tracking_row)
        tracking_hint = QLabel(
            "Fixed: each point's search window stays centered on its reference position -- "
            "search radius alone must cover the point's total displacement. Moving: each frame's "
            "search window recenters on the point's last found position -- follows a large "
            "cumulative motion (the speckle pattern's own displacement) with a small search "
            "radius, at the cost of a bad match shifting every later frame's search window too."
        )
        tracking_hint.setObjectName("cardSubtitle")
        tracking_hint.setWordWrap(True)
        card.add(tracking_hint)

        subpixel_row = QHBoxLayout()
        subpixel_row.addWidget(QLabel("Sub-pixel refinement"))
        self.subpixel_combo = QComboBox()
        self.subpixel_combo.addItem("Gauss-Newton, affine subset (accurate)", "icgn")
        self.subpixel_combo.addItem("Parabolic peak (fast)", "parabolic")
        self.subpixel_combo.setToolTip(
            "Gauss-Newton (default): after the integer + parabolic search, each subset is refined "
            "with an affine (stretch / shear / rotation) shape function and a cubic-interpolated "
            "zero-normalized least-squares match, as in Ncorr / VIC-2D. Removes the vertical and "
            "horizontal stripes that the parabolic estimate leaves in the strain maps (peak "
            "locking, rigid subsets), at a few times the run time. Parabolic: fast, but the "
            "displacement is biased towards whole pixels and the strain shows those stripes. "
            "The correlation coefficient reported is then the ZNCC of the refined match. "
            "Changing it requires running the correlation again."
        )
        self.subpixel_combo.currentIndexChanged.connect(self._on_subpixel_changed)
        subpixel_row.addWidget(self.subpixel_combo)
        subpixel_row.addStretch(1)
        card.add_layout(subpixel_row)

        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Minimum correlation coefficient"))
        self.correlation_threshold_spin = QDoubleSpinBox()
        self.correlation_threshold_spin.setRange(0.0, 1.0)
        self.correlation_threshold_spin.setSingleStep(0.05)
        self.correlation_threshold_spin.setDecimals(2)
        self.correlation_threshold_spin.setValue(0.0)
        self.correlation_threshold_spin.setToolTip(
            "0 = keep every point regardless of match quality. Above 0, a point is dropped -- "
            "for the rest of the sequence, not just that frame -- the first time its match falls "
            "below this NCC value. Useful once necking (or any other shrinking of the specimen's "
            "analyzable area) pushes some points off the specimen and onto the background, where "
            "they can no longer correlate reliably against their reference-frame template."
        )
        self.correlation_threshold_spin.valueChanged.connect(self._on_correlation_threshold_changed)
        quality_row.addWidget(self.correlation_threshold_spin)
        quality_row.addStretch(1)
        card.add_layout(quality_row)

        neighbor_row = QHBoxLayout()
        neighbor_row.addWidget(QLabel("Max neighbor displacement deviation (px)"))
        self.neighbor_deviation_spin = QDoubleSpinBox()
        self.neighbor_deviation_spin.setRange(0.0, 1000.0)
        self.neighbor_deviation_spin.setSingleStep(1.0)
        self.neighbor_deviation_spin.setDecimals(1)
        self.neighbor_deviation_spin.setValue(0.0)
        self.neighbor_deviation_spin.setToolTip(
            "0 = disabled. Above 0, a point is dropped -- for the rest of the sequence -- the "
            "first time its displacement disagrees with the median of its immediate neighbors' "
            "displacement by more than this many pixels. A numerical analog of FEM \"hourglass "
            "control\": a single point that tracked to an implausible position relative to its "
            "neighbors (even with a good correlation score) turns the tracked mesh visibly jagged "
            "right around it -- this removes that point instead of letting it distort the field."
        )
        self.neighbor_deviation_spin.valueChanged.connect(self._on_neighbor_deviation_changed)
        neighbor_row.addWidget(self.neighbor_deviation_spin)
        neighbor_row.addStretch(1)
        card.add_layout(neighbor_row)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run correlation")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._on_run_correlation)
        run_row.addWidget(self.run_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip(
            "Stops the run after the frame currently being tracked finishes -- the frames "
            "already tracked are discarded, not kept as a partial result."
        )
        self.stop_button.clicked.connect(self._on_stop_correlation)
        run_row.addWidget(self.stop_button)
        run_row.addStretch(1)
        card.add_layout(run_row)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        card.add(self.progress_bar)
        self.run_status_label = QLabel("")
        self.run_status_label.setObjectName("cardSubtitle")
        self.run_status_label.setWordWrap(True)
        card.add(self.run_status_label)
        layout.addWidget(card)

    def _build_results_card(self, layout):
        card = Card(
            "Results",
            "Browse the tracked displacement/strain field frame by frame, or export the full "
            "per-point, per-frame result as CSV.",
            help_html=(
                "<p>Browse the tracked field frame by frame (see the plot's own help for the "
                "Field/Color scale/slider controls), or export the underlying data.</p>"
                "<ul>"
                "<li><b>Field</b>: besides exx / eyy / exy, <b>e1</b> / <b>e2</b> (max / min principal "
                "strain) and <b>gmax</b> (max shear strain, (e1 - e2)/2) do not depend on the "
                "direction of the camera axes -- use them to follow a crack that is neither "
                "horizontal nor vertical (a crack opening shows up in e1, the band it creates in "
                "gmax).</li>"
                "<li><b>Spatial smoothing</b>: Gaussian filter (sigma in mesh points, 0 = off) "
                "applied to the displayed field to reduce the speckle/background noise. It affects "
                "the Results view, the probes and the exported video, NOT the CSV exports. Keep it "
                "small (0.5 - 1.5 points): it also spreads a real crack over a few mesh points.</li>"
                "<li><b>Export results (.csv)…</b>: the full per-point, per-frame result (every "
                "mesh point, every field, every processed frame), always unsmoothed.</li>"
                "<li><b>Export video (.mp4)…</b>: renders the currently selected field's color-map "
                "view across every processed frame as a video file.</li>"
                "</ul>"
            ),
        )
        field_row = QHBoxLayout()
        field_row.addWidget(QLabel("Field"))
        self.field_combo = QComboBox()
        for key, label in FIELD_CHOICES:
            self.field_combo.addItem(label, key)
        self.field_combo.currentIndexChanged.connect(self._on_result_view_changed)
        field_row.addWidget(self.field_combo)
        field_row.addStretch(1)
        card.add_layout(field_row)

        smoothing_row = QHBoxLayout()
        smoothing_row.addWidget(QLabel("Spatial smoothing"))
        self.smoothing_spin = QDoubleSpinBox()
        self.smoothing_spin.setRange(0.0, 10.0)
        self.smoothing_spin.setSingleStep(0.25)
        self.smoothing_spin.setDecimals(2)
        self.smoothing_spin.setValue(0.0)
        self.smoothing_spin.setSuffix(" pt")
        self.smoothing_spin.setSpecialValueText("off")
        self.smoothing_spin.setToolTip(
            "Gaussian smoothing of the displayed field over neighbouring mesh points "
            "(sigma, in mesh points; 0 = off). Strain components are smoothed first and the "
            "principal / shear / equivalent strains are computed from them. Lost or excluded "
            "points are ignored and stay empty. Display only: the exported CSV stays raw."
        )
        self.smoothing_spin.valueChanged.connect(self._on_spatial_smoothing_changed)
        smoothing_row.addWidget(self.smoothing_spin)
        self.smoothing_label = QLabel("off")
        self.smoothing_label.setObjectName("cardSubtitle")
        smoothing_row.addWidget(self.smoothing_label, 1)
        card.add_layout(smoothing_row)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Color scale"))
        self.scale_auto_radio = QRadioButton("Auto (per frame)")
        self.scale_fixed_radio = QRadioButton("Fixed")
        self.scale_auto_radio.setChecked(True)
        scale_mode_group = QButtonGroup(self)
        scale_mode_group.addButton(self.scale_auto_radio)
        scale_mode_group.addButton(self.scale_fixed_radio)
        self.scale_auto_radio.toggled.connect(self._on_color_scale_mode_toggled)
        scale_row.addWidget(self.scale_auto_radio)
        scale_row.addWidget(self.scale_fixed_radio)
        scale_row.addSpacing(12)
        scale_row.addWidget(QLabel("Min"))
        self.scale_min_spin = QDoubleSpinBox()
        self.scale_min_spin.setRange(-1e9, 1e9)
        self.scale_min_spin.setDecimals(4)
        self.scale_min_spin.setEnabled(False)
        self.scale_min_spin.valueChanged.connect(self._on_color_scale_bounds_changed)
        scale_row.addWidget(self.scale_min_spin)
        scale_row.addWidget(QLabel("Max"))
        self.scale_max_spin = QDoubleSpinBox()
        self.scale_max_spin.setRange(-1e9, 1e9)
        self.scale_max_spin.setDecimals(4)
        self.scale_max_spin.setEnabled(False)
        self.scale_max_spin.valueChanged.connect(self._on_color_scale_bounds_changed)
        scale_row.addWidget(self.scale_max_spin)
        self.scale_reset_button = QPushButton("Reset to data range")
        self.scale_reset_button.setEnabled(False)
        self.scale_reset_button.clicked.connect(self._on_color_scale_reset)
        scale_row.addWidget(self.scale_reset_button)
        scale_row.addStretch(1)
        card.add_layout(scale_row)
        scale_hint = QLabel(
            "Auto: the color scale re-fits itself to each frame's own data. Fixed: the same Min/"
            "Max is used for every frame -- and for an exported video -- so color stays visually "
            "comparable across the whole sequence."
        )
        scale_hint.setObjectName("cardSubtitle")
        scale_hint.setWordWrap(True)
        card.add(scale_hint)

        result_slider_row = QHBoxLayout()
        self.result_slider = QSlider(Qt.Horizontal)
        self.result_slider.setEnabled(False)
        self.result_slider.valueChanged.connect(self._on_result_view_changed)
        result_slider_row.addWidget(self.result_slider, 1)
        self.result_frame_label = QLabel("")
        self.result_frame_label.setMinimumWidth(120)
        result_slider_row.addWidget(self.result_frame_label)
        card.add_layout(result_slider_row)

        self.result_canvas = PlotCanvas(
            figsize=(8, 5.5),
            help_title="Results",
            help_text=(
                "<p>Shows the tracked field (pick which one from the <b>Field</b> dropdown above) "
                "as a color map over the correlation mesh, one frame at a time -- move the slider "
                "below to step through frames.</p>"
                "<ul>"
                "<li><b>Auto</b> color scale re-fits itself to each frame's own data; <b>Fixed</b> "
                "keeps the same Min/Max across every frame (and an exported video), so color stays "
                "visually comparable over the whole sequence.</li>"
                "<li>Hover over the plot to read the field's actual value at the nearest tracked "
                "point, in the status readout.</li>"
                "</ul>"
            ),
        )
        self.result_canvas.set_value_lookup(self._result_cursor_value)
        card.add(self.result_canvas)
        result_slider_row.addWidget(make_popout_button(
            self, "Image correlation -- results", self.result_canvas, self.result_slider,
            info_labels=[self.result_frame_label]))

        export_row = QHBoxLayout()
        self.export_results_button = QPushButton("Export results (.csv)…")
        self.export_results_button.setEnabled(False)
        self.export_results_button.clicked.connect(self._on_export_results)
        export_row.addWidget(self.export_results_button)
        self.export_video_button = QPushButton("Export video (.mp4)…")
        self.export_video_button.setEnabled(False)
        self.export_video_button.clicked.connect(self._on_export_video)
        export_row.addWidget(self.export_video_button)
        self.import_results_button = QPushButton("Import results (.csv)…")
        self.import_results_button.setToolTip(
            "Reopen a CSV previously produced by \"Export results\" -- on this page, on a "
            "previous run, or on another machine -- without re-running the correlation."
        )
        self.import_results_button.clicked.connect(self._on_import_results)
        export_row.addWidget(self.import_results_button)
        export_row.addStretch(1)
        card.add_layout(export_row)
        self.results_status_label = QLabel("")
        self.results_status_label.setObjectName("cardSubtitle")
        card.add(self.results_status_label)
        self.video_progress_bar = QProgressBar()
        self.video_progress_bar.setValue(0)
        self.video_progress_bar.setVisible(False)
        card.add(self.video_progress_bar)
        self.video_export_status_label = QLabel("")
        self.video_export_status_label.setObjectName("cardSubtitle")
        self.video_export_status_label.setWordWrap(True)
        card.add(self.video_export_status_label)
        layout.addWidget(card)

    def _build_mean_vs_time_card(self, layout):
        card = Card(
            "ROI Mean vs Time",
            "Mean displacement and strain over every successfully-tracked point in the ROI, for "
            "each processed frame -- against time when a frame rate is given above (it already "
            "accounts for the frame step), otherwise against frame index.",
        )
        self.mean_time_canvas = PlotCanvas(
            figsize=(8, 6),
            help_title="ROI Mean vs Time",
            help_text=(
                "<p>Plots the mean displacement and strain, averaged over every successfully-"
                "tracked point in the ROI, for each processed frame -- against time (in seconds) "
                "when a camera frame rate is set on the Calibration card above, otherwise against "
                "frame index.</p>"
                "<p>When <b>Show uncertainty envelopes</b> is enabled on the DIC Measurement "
                "Uncertainty card, a shaded band shows the estimated measurement uncertainty "
                "around each curve.</p>"
            ),
        )
        card.add(self.mean_time_canvas)

        export_row = QHBoxLayout()
        self.export_all_mean_button = QPushButton("Export all ROI Mean vs Time (.csv)…")
        self.export_all_mean_button.setEnabled(False)
        self.export_all_mean_button.clicked.connect(self._on_export_all_mean_vs_time)
        export_row.addWidget(self.export_all_mean_button)
        export_row.addStretch(1)
        card.add_layout(export_row)
        layout.addWidget(card)

    def _build_dic_strain_export_card(self, layout):
        card = Card(
            "DIC Strain for Results & Export",
            "Publishes the ROI-mean of the strain definition chosen below (and its strain rate) "
            "so it can be picked as \"DIC Strain (-)\" / \"DIC Strain Rate (1/s)\" on the Results "
            "& Export page -- plotted there against the true stress from the standard processing "
            "engine (Load/Displacement Data / SHPB / Direct Impact) in its Multi-Strain Comparison plot, "
            "and included in that page's exports. Needs the camera frame rate above (Calibration "
            "card): without it, this series has no time-in-seconds axis to line up against the "
            "standard result's own time axis.\n\n"
            "DIC's own exx/eyy/MISES are a first-order, SMALL-STRAIN (linearized) estimate from "
            "the local displacement gradient -- see core/dic_processing.py's _grid_strain -- close "
            "in spirit to a NOMINAL/engineering strain, not the engine's own LOGARITHMIC true "
            "strain. \"Convert to true strain\" below applies the standard eps_true = ln(1 + eps) "
            "conversion (and its matching rate conversion) so the published value is defined the "
            "same way as the standard engine's -- exact for a genuinely uniaxial stretch, an "
            "approximation for exx/eyy/MISES otherwise, and never applied to exy (shear has no "
            "such conversion).\n\n"
            "Time scale / Time offset: the camera and the bar-signal DAQ are two independent, "
            "unsynchronized clocks in this app -- the camera's own actual capture rate can also "
            "differ slightly from the frame rate entered above. \"Time scale\" corrects for that "
            "(effective fps = entered fps x scale -- e.g. 1.02 if the camera actually ran 2% fast); "
            "\"Time offset\" shifts DIC's own t=0 (its first processed frame) to line up with the "
            "bar-signal result's own t=0. Set both by matching a feature visible in both series "
            "(e.g. a known event's start and duration) -- there is no automatic detection here. "
            "Both apply everywhere this page builds a time axis (every vs-time plot, velocity/"
            "acceleration/strain rate, every CSV export, the video timestamp), not just this "
            "series, so scale in particular also corrects the physical dt used for those "
            "derivatives. If this camera recording covers a shorter, and/or differently-timed, "
            "real-world duration than the full bar-signal test, the strain accumulated during that "
            "DIC window can still legitimately be much smaller than the test's overall strain -- "
            "not a bug, just less of the event captured on camera (see the coverage note under the "
            "comparison plot).",
            help_html=(
                "<p>Publishes the ROI-mean of the chosen strain definition (and its rate) to the "
                "Results &amp; Export page, as \"DIC Strain (-)\" / \"DIC Strain Rate (1/s)\" -- "
                "plotted there against true stress in the Multi-Strain Comparison plot. Needs the "
                "camera frame rate (Calibration card) to build a time-in-seconds axis.</p>"
                "<ul>"
                "<li><b>Strain definition</b>: which DIC field to publish (MISES, exx, eyy, "
                "exy).</li>"
                "<li><b>Convert to true (logarithmic) strain</b>: DIC's own exx/eyy/MISES are a "
                "first-order, SMALL-STRAIN estimate, closer to nominal/engineering strain than to "
                "the standard engine's logarithmic true strain. Checking this applies "
                "eps_true = ln(1 + eps) (and its matching rate conversion) so the published value "
                "is defined the same way as the standard engine's -- exact for a genuinely "
                "uniaxial stretch, an approximation otherwise, and never applied to exy (shear has "
                "no such conversion).</li>"
                "<li><b>Time scale / Time offset</b>: the camera and the bar-signal DAQ are two "
                "independent, unsynchronized clocks -- <i>scale</i> corrects the camera's actual "
                "capture rate (effective fps = entered fps &times; scale, e.g. 1.02 if the camera "
                "ran 2% fast), <i>offset</i> shifts DIC's own t=0 to line up with the bar-signal "
                "result's t=0. Set both by matching a feature visible in both series -- there is "
                "no automatic detection.</li>"
                "</ul>"
                "<p><b>Technical detail:</b> Time scale/offset apply everywhere this page builds a "
                "time axis (every vs-time plot, velocity/acceleration/strain rate, every CSV "
                "export, the video timestamp) -- scale in particular also corrects the physical dt "
                "used for those derivatives. If the camera recording covers a shorter or "
                "differently-timed slice of the test than the full bar-signal test, the strain "
                "accumulated during that DIC window can legitimately be much smaller than the "
                "test's overall strain -- not a bug, just less of the event captured on camera.</p>"
            ),
        )
        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("Strain definition"))
        self.dic_strain_combo = QComboBox()
        for key, label in DIC_STRAIN_DEFINITIONS:
            self.dic_strain_combo.addItem(label, key)
        self.dic_strain_combo.currentIndexChanged.connect(self._on_dic_strain_definition_changed)
        combo_row.addWidget(self.dic_strain_combo)
        combo_row.addStretch(1)
        card.add_layout(combo_row)
        self.dic_strain_true_check = QCheckBox(
            "Convert to true (logarithmic) strain -- eps_true = ln(1 + eps); ignored for exy"
        )
        self.dic_strain_true_check.setChecked(True)
        self.dic_strain_true_check.toggled.connect(self._on_dic_strain_definition_changed)
        card.add(self.dic_strain_true_check)

        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("Time scale (corrected fps = entered fps x this)"))
        self.dic_time_scale_spin = QDoubleSpinBox()
        # Effectively uncapped (must stay > 0 -- it's a frequency ratio,
        # and 0 or negative would break the effective-fps division in
        # `_effective_fps`) -- QDoubleSpinBox needs *some* finite bound,
        # so this uses the largest one that still displays/edits
        # normally rather than a value a user might plausibly want to
        # exceed.
        self.dic_time_scale_spin.setRange(1e-9, 1.0e9)
        self.dic_time_scale_spin.setDecimals(6)
        self.dic_time_scale_spin.setSingleStep(0.001)
        self.dic_time_scale_spin.setValue(1.0)
        self.dic_time_scale_spin.valueChanged.connect(self._on_dic_time_correction_changed)
        time_row.addWidget(self.dic_time_scale_spin)
        time_row.addSpacing(16)
        time_row.addWidget(QLabel("Time offset (s, added after scaling)"))
        self.dic_time_offset_spin = QDoubleSpinBox()
        self.dic_time_offset_spin.setRange(-1.0e6, 1.0e6)
        self.dic_time_offset_spin.setDecimals(6)
        self.dic_time_offset_spin.setSingleStep(0.0001)
        self.dic_time_offset_spin.setValue(0.0)
        self.dic_time_offset_spin.valueChanged.connect(self._on_dic_time_correction_changed)
        time_row.addWidget(self.dic_time_offset_spin)
        time_row.addStretch(1)
        card.add_layout(time_row)

        self.dic_strain_status_label = QLabel(
            "Run a correlation to publish a DIC strain series to Results & Export."
        )
        self.dic_strain_status_label.setObjectName("cardSubtitle")
        self.dic_strain_status_label.setWordWrap(True)
        card.add(self.dic_strain_status_label)
        layout.addWidget(card)

    def _build_uncertainty_card(self, layout):
        card = Card(
            "DIC Measurement Uncertainty",
            "A PREDICTIVE model, not a certified/calibrated measurement uncertainty (there is no "
            "ground truth here to check it against) -- combines three well-established DIC error "
            "mechanisms for every tracked point, every frame: (1) sub-pixel REGISTRATION precision "
            "-- how sharply the correlation peak is located, from the reference subset's own "
            "texture (richer speckle -> lower uncertainty) and the camera's pixel noise (both "
            "estimated automatically from the images, no extra input needed); (2) CORRELATION-"
            "QUALITY degradation -- inflated the further a frame's own match falls from perfect; "
            "(3) SUBSET DISTORTION -- since each subset is tracked as one rigid template (see "
            "core/dic_processing.py), a point sitting where the material is actually straining a "
            "lot has its position blurred over roughly (local strain x subset size), which is "
            "exactly \"the level of distortion of the elements\". These three are combined, then "
            "propagated (standard error-propagation formulas) through every derived quantity: "
            "strain from the mesh's own step size, strain rate/velocity/acceleration/dynamic force "
            "from the frame rate, and the ROI-mean series below from an EFFECTIVE point count that "
            "accounts for neighboring points not being independent measurements (their subsets "
            "overlap / the material moves continuously) -- see core/dic_uncertainty.py for every "
            "formula and its own caveats.\n\n"
            "If a validated uncertainty budget is needed (e.g. for a report), the standard, more "
            "rigorous alternative this does NOT replace is an empirical calibration: run this same "
            "pipeline on two images of the SAME undeformed specimen (or a synthetically warped "
            "image with a KNOWN applied strain) and measure the actual scatter/bias directly -- "
            "that captures every real-world error source (lens distortion, out-of-plane motion, "
            "...) this predictive model does not.",
            help_html=(
                "<p>A PREDICTIVE model, not a certified/calibrated measurement uncertainty (there "
                "is no ground truth here to check it against) -- combines three well-established "
                "DIC error mechanisms for every tracked point, every frame.</p>"
                "<ul>"
                "<li><b>Sub-pixel registration precision</b>: how sharply the correlation peak is "
                "located, from the reference subset's own texture (richer speckle &rarr; lower "
                "uncertainty) and the camera's pixel noise -- both estimated automatically from "
                "the images, no extra input needed.</li>"
                "<li><b>Correlation-quality degradation</b>: inflated the further a frame's own "
                "match falls from perfect.</li>"
                "<li><b>Subset distortion</b>: since each subset is tracked as one rigid template, "
                "a point where the material is actually straining a lot has its position blurred "
                "over roughly (local strain &times; subset size).</li>"
                "</ul>"
                "<p><b>Technical detail:</b> these three are combined, then propagated (standard "
                "error-propagation formulas) through every derived quantity: strain from the "
                "mesh's own step size, strain rate/velocity/acceleration/dynamic force from the "
                "frame rate, and the ROI-mean series from an effective point count that accounts "
                "for neighboring points not being independent measurements (their subsets overlap "
                "/ the material moves continuously). If a validated uncertainty budget is needed "
                "(e.g. for a report), the more rigorous alternative this does NOT replace is an "
                "empirical calibration: run this same pipeline on two images of the SAME "
                "undeformed specimen (or a synthetically warped image with a KNOWN applied strain) "
                "and measure the actual scatter/bias directly -- that captures every real-world "
                "error source (lens distortion, out-of-plane motion, ...) this predictive model "
                "does not.</p>"
            ),
        )
        show_row = QHBoxLayout()
        self.uncertainty_show_check = QCheckBox("Show uncertainty envelopes on the plots below")
        self.uncertainty_show_check.setChecked(True)
        self.uncertainty_show_check.toggled.connect(self._on_uncertainty_display_changed)
        show_row.addWidget(self.uncertainty_show_check)
        show_row.addSpacing(16)
        show_row.addWidget(QLabel("Envelope width"))
        self.uncertainty_width_combo = QComboBox()
        self.uncertainty_width_combo.addItem("+/-1 sigma (~68% if Gaussian)", 1.0)
        self.uncertainty_width_combo.addItem("+/-2 sigma (~95% if Gaussian)", 2.0)
        self.uncertainty_width_combo.currentIndexChanged.connect(self._on_uncertainty_display_changed)
        show_row.addWidget(self.uncertainty_width_combo)
        show_row.addStretch(1)
        card.add_layout(show_row)
        self.uncertainty_status_label = QLabel(
            "Run a correlation to estimate a measurement uncertainty."
        )
        self.uncertainty_status_label.setObjectName("cardSubtitle")
        self.uncertainty_status_label.setWordWrap(True)
        card.add(self.uncertainty_status_label)
        layout.addWidget(card)

    def _build_velocity_card(self, layout):
        card = Card(
            "ROI Velocity vs Time",
            "Mean velocity (time-derivative of the tracked displacement, magnitude included) "
            "over every successfully-tracked point in the ROI -- needs the camera frame rate "
            "above.",
        )
        self.velocity_canvas = PlotCanvas(
            figsize=(8, 4.5),
            help_title="ROI Velocity vs Time",
            help_text=(
                "<p>Plots the mean velocity (the time-derivative of the tracked displacement, plus "
                "its magnitude) averaged over every successfully-tracked ROI point, against time -- "
                "needs the camera frame rate set on the Calibration card above.</p>"
                "<p><b>Technical detail:</b> since this is a derivative of the tracked positions, "
                "it is somewhat noisier than the raw displacement itself.</p>"
            ),
        )
        card.add(self.velocity_canvas)
        layout.addWidget(card)

    def _build_strain_rate_card(self, layout):
        card = Card(
            "ROI Strain Rate vs Time",
            "Mean strain rate (time-derivative of the tracked strain) over every successfully-"
            "tracked point in the ROI, plus the equivalent (von Mises) strain rate -- needs the "
            "camera frame rate above.",
        )
        self.strain_rate_canvas = PlotCanvas(
            figsize=(8, 4.5),
            help_title="ROI Strain Rate vs Time",
            help_text=(
                "<p>Plots the mean strain rate (the time-derivative of the tracked strain), plus "
                "the equivalent (von Mises) strain rate, averaged over every successfully-tracked "
                "ROI point, against time -- needs the camera frame rate set on the Calibration card "
                "above.</p>"
                "<p><b>Technical detail:</b> being a derivative of the tracked field, it is "
                "noisier than the strain curve it comes from; consistency of this rate over the "
                "plateau is a good sanity check for a well-conducted constant-strain-rate test.</p>"
            ),
        )
        card.add(self.strain_rate_canvas)
        layout.addWidget(card)

    def _build_force_card(self, layout):
        card = Card(
            "ROI Dynamic Force vs Time",
            "Mean 'Dynamic Force' F = rho x A x a (Newton's second law, with the specimen's "
            "cross-section A = width x thickness standing in for its mass per unit length) over "
            "the ROI -- needs the camera frame rate, pixel size, material density and specimen "
            "width/thickness above. Acceleration is a second time-derivative, so it's often "
            "noisy; use the slider below to smooth the curve (centered moving average, in "
            "processed frames) before reading or exporting it.",
            help_html=(
                "<p>Estimates a 'Dynamic Force' F = &rho; &times; A &times; a directly from the "
                "tracked kinematics (Newton's second law, with the specimen's cross-section "
                "A = width &times; thickness standing in for its mass per unit length) -- needs "
                "the camera frame rate, pixel size, material density and specimen width/thickness "
                "set on the Calibration card above.</p>"
                "<ul>"
                "<li><b>Smoothing window (frames)</b>: acceleration is a second time-derivative of "
                "the tracked positions, so it's often noisy -- this applies a centered moving "
                "average (in processed frames) before the curve is read or exported.</li>"
                "</ul>"
                "<p><b>Technical detail:</b> F = &rho; &times; A &times; a is dimensionally a real "
                "force in Newtons (kg/m&sup3; &times; m&sup2; &times; m/s&sup2; = kg&middot;m/s&sup2;). "
                "It assumes a constant cross-section A along the specimen (not locally reduced by "
                "necking) and uses each point's own local acceleration -- most meaningfully read as "
                "the ROI mean, i.e. the inertial force of a uniform segment accelerating like the "
                "ROI's own average.</p>"
            ),
        )
        self.force_canvas = PlotCanvas(
            figsize=(8, 4.5),
            help_title="ROI Dynamic Force vs Time",
            help_text=(
                "<p>Plots the mean Dynamic Force (see this card's own help for the formula and its "
                "assumptions) against time.</p>"
                "<p>Use the <b>Smoothing window</b> slider below the plot before reading or "
                "exporting -- acceleration is a second time-derivative of the tracked positions, "
                "so the raw curve is often noisy.</p>"
            ),
        )
        card.add(self.force_canvas)

        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Smoothing window (frames)"))
        self.force_smoothing_slider = QSlider(Qt.Horizontal)
        self.force_smoothing_slider.setRange(1, 51)
        self.force_smoothing_slider.setValue(1)
        self.force_smoothing_slider.valueChanged.connect(self._on_force_smoothing_changed)
        smooth_row.addWidget(self.force_smoothing_slider, 1)
        self.force_smoothing_label = QLabel("1 (no smoothing)")
        self.force_smoothing_label.setMinimumWidth(130)
        smooth_row.addWidget(self.force_smoothing_label)
        card.add_layout(smooth_row)

        dimension_hint = QLabel(
            "F = rho x A x a is dimensionally a real force in Newtons (kg/m^3 x m^2 x m/s^2 = "
            "kg*m/s^2). It still assumes a constant cross-section A along the specimen (not "
            "locally reduced by necking) and uses each point's own local acceleration -- most "
            "meaningfully read as the ROI mean, i.e. the inertial force of a uniform segment "
            "accelerating like the ROI's own average."
        )
        dimension_hint.setObjectName("cardSubtitle")
        dimension_hint.setWordWrap(True)
        card.add(dimension_hint)
        layout.addWidget(card)

    @staticmethod
    def _make_spin(minimum: int, maximum: int, on_changed) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.valueChanged.connect(on_changed)
        return spin

    # ==================================================================
    # Frame source
    # ==================================================================

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
        self.project.last_paths["dic_source"] = path
        self.project.dic.source_mode = "sequence" if self.sequence_radio.isChecked() else "video"

        for widget in (self.frame_slider, self.frame_spin):
            widget.setEnabled(True)
        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(source.n_frames - 1)
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(source.n_frames - 1)
        start_frame = max(0, min(self.project.dic.reference_frame, source.n_frames - 1))
        self.frame_slider.setValue(start_frame)
        self.frame_spin.setValue(start_frame)
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)

        fps_text = f", {source.fps:.6g} fps" if source.fps else ""
        self.source_status.setText(
            f"Loaded {source.n_frames} frames{fps_text} from {os.path.basename(path.rstrip('/'))}"
        )
        # A video's own container metadata is a reasonable default frame
        # rate for the ROI-mean-vs-time plot -- but never overwrite a
        # value the user already typed in (or one restored from a
        # previous session).
        if source.fps and not self.frame_rate_edit.text().strip():
            self.frame_rate_edit.setText(_fmt_num(source.fps))
            self.project.dic.frame_rate_hz = source.fps
        self._invalidate_results()
        self._show_frame(start_frame, keep_crop_rect=False, keep_roi_rect=False)

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

    def _show_frame(self, index: int, keep_crop_rect: bool = True, keep_roi_rect: bool = True):
        if self._source is None:
            return
        try:
            raw_frame = self._source.get_frame(index)
        except (ValueError, IndexError) as exc:
            self.frame_info_label.setText(f"Could not read frame {index}: {exc}")
            return

        raw_display = np.ascontiguousarray(to_display_uint8(raw_frame))
        self.crop_picker.set_frame(raw_display, keep_rect=keep_crop_rect)

        preprocessed = self._preprocess(raw_frame)
        self.roi_picker.set_mesh_params(self.subset_spin.value(), self.step_spin.value())
        # Probes are pixel positions in THIS preprocessed frame's own
        # coordinate system, same as the ROI -- `roi_picker.set_frame`
        # resets both together when `keep_region` is False (the two
        # situations that invalidate an existing ROI -- a genuinely new
        # source, or a crop/downscale change that redefines what
        # "preprocessed frame" even means -- invalidate old probe
        # positions for exactly the same reason; just scrubbing between
        # frames of the same setup does not).
        self.roi_picker.set_frame(preprocessed, keep_region=keep_roi_rect)
        if not keep_roi_rect:
            self._set_spin_fields((self.roi_x_spin, self.roi_y_spin, self.roi_w_spin, self.roi_h_spin), 0, 0, 0, 0)
            self._store_roi(None, None, None, None)
            self._store_roi_polygon(None)
            self._store_roi_ellipse(None)
            self._store_probes(None)

        info = f"Reference frame {index} / {self._source.n_frames - 1}"
        if isinstance(self._source, ImageSequenceFrameSource):
            info += f"  ({self._source.frame_name(index)})"
        self.frame_info_label.setText(info)
        self.project.dic.reference_frame = index
        self._update_preprocessing_summary(raw_frame.shape, preprocessed.shape)

    def _preprocess(self, raw_frame: np.ndarray) -> np.ndarray:
        return preprocess_frame(raw_frame, self.crop_picker.get_rect(), self.downscale_spin.value())

    def _processing_frame_indices(self) -> list[int]:
        if self._source is None:
            return []
        return decimated_frame_indices(
            self._source.n_frames, self.frame_slider.value(), self.frame_step_spin.value(),
        )

    def _update_preprocessing_summary(self, raw_shape, preprocessed_shape):
        raw_h, raw_w = raw_shape[:2]
        proc_h, proc_w = preprocessed_shape[:2]
        n_proc = len(self._processing_frame_indices())
        self.preprocessing_summary_label.setText(
            f"Working frame: {proc_w} x {proc_h} px (from {raw_w} x {raw_h} px raw)  |  "
            f"a correlation run would process {n_proc} frame(s)."
        )

    def _refresh_preview(self, reset_roi: bool):
        if self._source is None:
            return
        self._invalidate_results()
        self._show_frame(self.frame_slider.value(), keep_crop_rect=True, keep_roi_rect=not reset_roi)

    # ==================================================================
    # Calibration (pixel size / frame rate) -- display/export units only,
    # never affects the tracking itself.
    # ==================================================================

    def _on_calibration_changed(self):
        text = self.pixel_size_edit.text().strip()
        try:
            self.project.dic.pixel_size_mm = float(text) if text else None
        except ValueError:
            pass  # leave the previous value; the field itself still shows what the user typed
        text = self.frame_rate_edit.text().strip()
        try:
            self.project.dic.frame_rate_hz = float(text) if text else None
        except ValueError:
            pass
        self._kinematics = None
        self._uncertainty = None
        self._on_result_view_changed()
        self._redraw_all_time_plots()

    def _on_density_changed(self):
        """Shared handler for density AND the specimen width/thickness
        fields (all three only ever matter together, to build the
        Dynamic Force field -- see `_specimen_area_m2`), so any of them
        changing invalidates the same cached kinematics."""
        text = self.density_edit.text().strip()
        try:
            self.project.dic.material_density_kg_m3 = float(text) if text else None
        except ValueError:
            pass
        text = self.specimen_width_edit.text().strip()
        try:
            self.project.dic.specimen_width_mm = float(text) if text else None
        except ValueError:
            pass
        text = self.specimen_thickness_edit.text().strip()
        try:
            self.project.dic.specimen_thickness_mm = float(text) if text else None
        except ValueError:
            pass
        self._kinematics = None
        self._uncertainty = None
        self._on_result_view_changed()
        self._redraw_all_time_plots()

    def _material_density(self) -> float | None:
        text = self.density_edit.text().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        return value if value > 0 else None

    def _specimen_area_m2(self) -> float | None:
        """Specimen cross-section area (width x thickness), converted
        from the mm fields above to m^2 -- see `_on_density_changed` and
        core/dic_kinematics.compute_kinematics's `area_m2`. None unless
        BOTH width and thickness are given and positive."""
        try:
            w_mm = float(self.specimen_width_edit.text().strip())
            e_mm = float(self.specimen_thickness_edit.text().strip())
        except ValueError:
            return None
        if w_mm <= 0 or e_mm <= 0:
            return None
        return (w_mm * 1e-3) * (e_mm * 1e-3)

    def _ensure_kinematics(self):
        """Lazily computes (and caches) velocity/acceleration/equivalent-
        strain/strain-rate/dynamic-force for the current `self._dic_result`
        -- see core/dic_kinematics.py. Returns None if there's no result
        yet, or no frame rate (a time axis is required to differentiate
        anything). Invalidated (`self._kinematics = None`) whenever the
        result, frame rate, pixel size, or material density changes."""
        if self._dic_result is None:
            self._kinematics = None
            self._uncertainty = None
            return None
        if self._kinematics is not None:
            return self._kinematics
        fps = self._effective_fps()
        if not fps:
            return None
        try:
            self._kinematics = compute_kinematics(
                self._dic_result, fps, mm_per_px=self._mm_per_working_px(),
                rho_kg_m3=self._material_density(), area_m2=self._specimen_area_m2(),
            )
        except ValueError:
            self._kinematics = None
        return self._kinematics

    def _ensure_uncertainty(self):
        """Lazily computes (and caches) the measurement-uncertainty
        model for the current `self._dic_result` -- see
        core/dic_uncertainty.py. Returns None if there's no result yet,
        or if `run_dic` couldn't produce a noise/SSSIG estimate for
        this run (see `DICResult.noise_sigma`) -- an uncertainty
        estimate is a nice-to-have on top of the DIC result, not a
        precondition for using it. Invalidated everywhere
        `self._kinematics` is (same dependencies: the result itself,
        frame rate/time scale, pixel size, material density), plus
        computed FROM the current kinematics (for the rate/velocity/
        acceleration/force fields) so it's never stale relative to
        them."""
        if self._dic_result is None:
            self._uncertainty = None
            return None
        if self._uncertainty is not None:
            return self._uncertainty
        kin = self._ensure_kinematics()  # None is fine -- compute_uncertainty then just skips the rate fields
        self._uncertainty = compute_uncertainty(
            self._dic_result, kin=kin, mm_per_px=self._mm_per_working_px(),
            rho_kg_m3=self._material_density(), area_m2=self._specimen_area_m2(),
        )
        return self._uncertainty

    def _roi_area_px2(self) -> float:
        """Approximate ROI area (px^2) covered by the current
        correlation mesh: the number of points originally valid (before
        any later tracking loss -- see `core.dic_processing.run_dic`'s
        own `valid`/`active` distinction, read here from the REFERENCE
        frame's own correlation, which is 1.0 at every originally-valid
        point and NaN elsewhere, and never changes after the fact)
        times the mesh's own step^2 (each point "owns" one step x step
        cell of the grid) -- works regardless of the ROI's shape
        (rectangle, oval, polygon) since it comes straight from the
        mesh rather than re-deriving the ROI's own geometry. Used only
        by `_add_envelope`'s effective-independent-point-count
        correction (see `core.dic_uncertainty.
        effective_independent_count`)."""
        if self._dic_result is None or not self._dic_result.frames:
            return 0.0
        mesh = self._dic_result.mesh
        n_valid = int(np.count_nonzero(np.isfinite(self._dic_result.frames[0].correlation)))
        return n_valid * (mesh.step ** 2)

    def _roi_mean_sigma(self, sigma_frame) -> float:
        """The uncertainty of the ROI-wide MEAN of one field/frame's
        per-point sigma array -- i.e. `core.dic_uncertainty.
        roi_mean_uncertainty` fed the SAME effective-independent-point
        count (`effective_independent_count`) that `_add_envelope`
        below uses to shade the plots. This is deliberately NOT the
        same number as a single point's own uncertainty (`np.nanmean`
        of the raw sigma field): averaging over many points shrinks the
        uncertainty of their mean, the same way averaging N repeated
        measurements shrinks the uncertainty of the average. The two
        are different, both legitimate, quantities -- this helper
        exists so every place that reports "the" uncertainty for a
        ROI-mean series (the envelope AND the status text) uses the
        SAME one, rather than silently mixing them (which previously
        made the status text's number and the plotted band's width
        look inconsistent with each other)."""
        if self._dic_result is None:
            return float("nan")
        roi_area = self._roi_area_px2()
        subset_size = self._dic_result.mesh.subset_size
        n_eff = effective_independent_count(
            int(np.count_nonzero(np.isfinite(sigma_frame))), roi_area, subset_size
        )
        return roi_mean_uncertainty(sigma_frame, n_eff)

    def _add_envelope(self, ax, x, mean_values, sigma_field_list, color):
        """Shades a +/-k*sigma envelope (see the "DIC Measurement
        Uncertainty" card, `k` from `self.uncertainty_width_combo`)
        around an already-plotted ROI-mean series, using `_roi_mean_
        sigma`'s effective-N correction (NOT a naive 1/sqrt(raw point
        count) -- neighboring DIC points are not independent
        measurements, see `core.dic_uncertainty.roi_mean_uncertainty`'s
        docstring). A no-op if the toggle is off, there's no
        `sigma_field_list` for this quantity (e.g. no frame rate yet
        for a rate-derived one), or no uncertainty model could be built
        for this run at all."""
        if not self.uncertainty_show_check.isChecked() or sigma_field_list is None:
            return
        unc = self._ensure_uncertainty()
        if unc is None or self._dic_result is None:
            return
        k = self.uncertainty_width_combo.currentData() or 1.0
        band = np.array([self._roi_mean_sigma(sigma_frame) for sigma_frame in sigma_field_list]) * k
        mean_arr = np.asarray(mean_values, dtype=float)
        ax.fill_between(x, mean_arr - band, mean_arr + band, color=color, alpha=0.15, linewidth=0)

    def _on_measure_scale(self):
        if self._source is None:
            self.source_status.setText("Load a video or image sequence first.")
            return
        index = self.frame_slider.value()
        try:
            raw_frame = self._source.get_frame(index)
        except (ValueError, IndexError) as exc:
            self.source_status.setText(f"Could not read frame {index}: {exc}")
            return
        raw_display = np.ascontiguousarray(to_display_uint8(raw_frame))

        dialog = RulerDialog(raw_display, parent=self)
        if dialog.exec() == QDialog.Accepted and dialog.result_pixel_size_mm:
            value = dialog.result_pixel_size_mm
            self.pixel_size_edit.setText(_fmt_num(value))
            self.project.dic.pixel_size_mm = value
            self._kinematics = None
            self._uncertainty = None
            self._on_result_view_changed()
            self._redraw_all_time_plots()

    def _pixel_size_mm(self) -> float | None:
        text = self.pixel_size_edit.text().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        return value if value > 0 else None

    def _mm_per_working_px(self) -> float | None:
        """Physical size of one pixel of the PREPROCESSED (downscaled)
        frame that results are expressed in -- the calibration field is
        per RAW-frame pixel, so it scales with the downscale factor."""
        pixel_size = self._pixel_size_mm()
        if pixel_size is None:
            return None
        return pixel_size * self.downscale_spin.value()

    def _fps(self) -> float | None:
        text = self.frame_rate_edit.text().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        return value if value > 0 else None

    def _effective_fps(self) -> float | None:
        """The camera frame rate actually used to build every DIC time
        axis on this page -- vs-time plots, velocity/acceleration/
        strain-rate (which depend on dt), the DIC Strain series, every
        CSV export with a time_s column, and the exported video's
        timestamp burn-in. This is `_fps()` (the frame rate entered
        above) corrected by `Project.dic.time_scale` -- see that
        field's docstring for why (the camera's true capture rate can
        differ from the entered value, and is never synchronized with
        the bar-signal DAQ's own sample rate in this app). None if
        there's no entered frame rate to correct."""
        fps = self._fps()
        if not fps:
            return None
        scale = self.project.dic.time_scale or 1.0
        return fps * scale if scale > 0 else fps

    def _dic_time_s(self, frame_index, ref_index) -> float | None:
        """Elapsed time (s) of `frame_index` since `ref_index`, using
        `_effective_fps()` (the scale-corrected frame rate) plus
        `Project.dic.time_offset_s` (a constant shift, applied after
        the scale correction so it doesn't affect derivatives) -- the
        one formula every DIC time axis on this page should go through,
        so the "Time scale (bars/camera)" / "Time offset" fields on the
        DIC Strain for Results & Export card apply consistently
        everywhere rather than only to that one comparison. None if
        there's no frame rate to build a time axis from."""
        fps = self._effective_fps()
        if not fps:
            return None
        return (frame_index - ref_index) / fps + (self.project.dic.time_offset_s or 0.0)

    # ==================================================================
    # Crop / preprocessing
    # ==================================================================

    def _on_crop_changed(self, x, y, w, h):
        self._set_spin_fields((self.crop_x_spin, self.crop_y_spin, self.crop_w_spin, self.crop_h_spin), x, y, w, h)
        self._store_crop(x, y, w, h)
        self._refresh_preview(reset_roi=True)

    def _on_crop_fields_changed(self, _value=None):
        x, y, w, h = (self.crop_x_spin.value(), self.crop_y_spin.value(),
                      self.crop_w_spin.value(), self.crop_h_spin.value())
        if w <= 0 or h <= 0:
            return
        self.crop_picker.set_rect(x, y, w, h)
        self._store_crop(x, y, w, h)
        self._refresh_preview(reset_roi=True)

    def _store_crop(self, x, y, w, h):
        dic = self.project.dic
        if w and h and w > 0 and h > 0:
            dic.crop_x, dic.crop_y, dic.crop_width, dic.crop_height = x, y, w, h
        else:
            dic.crop_x = dic.crop_y = dic.crop_width = dic.crop_height = None

    def _on_downscale_changed(self, value):
        self.project.dic.downscale_factor = value
        self._refresh_preview(reset_roi=True)

    def _on_frame_step_changed(self, value):
        self.project.dic.frame_step = value
        self._invalidate_results()
        # Frame step is a purely temporal setting (it doesn't change any
        # frame's pixels), so just re-derive the processed-frame count
        # from the label already shown, without re-reading a frame.
        n_proc = len(self._processing_frame_indices())
        text = self.preprocessing_summary_label.text()
        if "|" in text:
            head = text.split("|")[0].strip()
            self.preprocessing_summary_label.setText(
                f"{head}  |  a correlation run would process {n_proc} frame(s)."
            )

    # ==================================================================
    # ROI / mesh
    # ==================================================================

    def _current_roi_shape_mode(self) -> str:
        if self.roi_polygon_radio.isChecked():
            return "polygon"
        if self.roi_oval_radio.isChecked():
            return "oval"
        return "rectangle"

    def _on_roi_shape_toggled(self, checked=None):
        # Three mutually-exclusive radios now share this slot -- react
        # only to the one that just became checked (the others each emit
        # their own toggled(False) for the same click, which would
        # otherwise run this body 2x per switch).
        if checked is False:
            return
        mode = self._current_roi_shape_mode()
        self.roi_picker.set_shape_mode(mode)
        self.project.dic.roi_shape = mode
        self.roi_fields_widget.setVisible(mode == "rectangle")
        self.roi_polygon_hint_label.setVisible(mode == "polygon")
        self.roi_oval_hint_label.setVisible(mode == "oval")
        # set_shape_mode() already cleared whatever region was set under
        # the previous mode -- reflect that in the rectangle spin fields
        # and stored settings too, and drop any now-stale run result.
        self._set_spin_fields((self.roi_x_spin, self.roi_y_spin, self.roi_w_spin, self.roi_h_spin), 0, 0, 0, 0)
        self._store_roi(None, None, None, None)
        self._store_roi_polygon(None)
        self._store_roi_ellipse(None)
        self._update_mesh_status()
        self._invalidate_results()

    def _on_tracking_mode_toggled(self, _checked=None):
        self.project.dic.tracking_mode = "fixed" if self.tracking_fixed_radio.isChecked() else "moving"
        self._invalidate_results()

    def _on_roi_region_changed(self):
        mode = self.roi_picker.get_shape_mode()
        if mode == "rectangle":
            roi = self.roi_picker.get_rect()
            if roi is None:
                self._set_spin_fields((self.roi_x_spin, self.roi_y_spin, self.roi_w_spin, self.roi_h_spin), 0, 0, 0, 0)
                self._store_roi(None, None, None, None)
            else:
                self._set_spin_fields((self.roi_x_spin, self.roi_y_spin, self.roi_w_spin, self.roi_h_spin),
                                       roi.x, roi.y, roi.width, roi.height)
                self._store_roi(roi.x, roi.y, roi.width, roi.height)
        elif mode == "oval":
            self._store_roi_ellipse(self.roi_picker.get_ellipse())
        else:
            self._store_roi_polygon(self.roi_picker.get_polygon())
        self._update_mesh_status()
        self._invalidate_results()

    def _on_roi_fields_changed(self, _value=None):
        if self.roi_picker.get_shape_mode() != "rectangle":
            return  # the X/Y/W/H fields only apply to a rectangular ROI
        x, y, w, h = (self.roi_x_spin.value(), self.roi_y_spin.value(),
                      self.roi_w_spin.value(), self.roi_h_spin.value())
        if w <= 0 or h <= 0:
            return
        self.roi_picker.set_rect(x, y, w, h)
        self._store_roi(x, y, w, h)
        self._update_mesh_status()
        self._invalidate_results()

    def _set_spin_fields(self, spins, x, y, w, h):
        for spin, value in zip(spins, (x, y, w, h)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _store_roi(self, x, y, w, h):
        dic = self.project.dic
        if w and h and w > 0 and h > 0:
            dic.roi_x, dic.roi_y, dic.roi_width, dic.roi_height = x, y, w, h
        else:
            dic.roi_x = dic.roi_y = dic.roi_width = dic.roi_height = None

    def _store_roi_polygon(self, vertices):
        self.project.dic.roi_polygon = [[float(x), float(y)] for x, y in vertices] if vertices else None

    def _store_roi_ellipse(self, ellipse):
        dic = self.project.dic
        if ellipse is not None:
            dic.roi_ellipse = [float(ellipse.cx), float(ellipse.cy), float(ellipse.rx), float(ellipse.ry)]
        else:
            dic.roi_ellipse = None

    # ==================================================================
    # Probes
    # ==================================================================

    def _store_probes(self, probes):
        self.project.dic.probes = (
            [[float(x), float(y)] for x, y in probes] if probes else None
        )

    def _on_probes_changed(self):
        probes = self.roi_picker.get_probes()
        self._store_probes(probes)
        self.export_probes_button.setEnabled(self._dic_result is not None and bool(probes))
        self._redraw_probe_plot()

    def _on_probe_field_changed(self, _index=None):
        self._redraw_probe_plot()

    def _nearest_mesh_index(self, px: float, py: float) -> tuple[int, int]:
        """The (row, col) index into `self._dic_result.mesh`'s grid
        closest to a probe placed at (px, py) -- the reference-frame
        position every mesh point keeps for its whole life, so this is
        the same regardless of which frame is being looked at. Only
        meaningful once a correlation has run; callers check
        `self._dic_result is not None` first."""
        xx0, yy0 = self._dic_result.mesh.grid_xy()
        dist2 = (xx0 - px) ** 2 + (yy0 - py) ** 2
        iy, ix = np.unravel_index(int(np.argmin(dist2)), dist2.shape)
        return int(iy), int(ix)

    def _redraw_probe_plot(self):
        if self._dic_result is None:
            self._show_placeholder_canvas(self.probe_canvas, "Run a correlation to see this plot.")
            return
        probes = self.roi_picker.get_probes()
        if not probes:
            self._show_placeholder_canvas(
                self.probe_canvas,
                "Right-click one or more probes on the image above (Region of Interest & "
                "Correlation Mesh) to see their evolution here.",
            )
            return

        field_key = self.probe_field_combo.currentData()
        x, xlabel = self._time_axis()
        probe_indices = [self._nearest_mesh_index(px, py) for px, py in probes]

        series_per_probe = [[] for _ in probes]
        y_label = ""
        for idx, frame_result in enumerate(self._dic_result.frames):
            values, y_label = self._field_values(idx, frame_result, field_key)
            for series, (iy, ix) in zip(series_per_probe, probe_indices):
                series.append(values[iy, ix])

        colors = [theme.ACCENT, "#ff9f43", theme.SUCCESS, theme.DANGER, "#a66bff", "#2fd8c4"]
        self.probe_canvas.clear()
        ax = self.probe_canvas.figure.add_subplot(111)
        any_finite = False
        for i, ((px, py), series) in enumerate(zip(probes, series_per_probe)):
            series = np.array(series, dtype=float)
            if np.any(np.isfinite(series)):
                any_finite = True
            ax.plot(x, series, "o-", markersize=3, color=colors[i % len(colors)],
                    label=f"P{i + 1} ({px:.0f}, {py:.0f} px)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(y_label)
        if any_finite:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "Every probe lost track for this field.", ha="center", va="center",
                    color=theme.DANGER, fontsize=9, transform=ax.transAxes)
        self.probe_canvas.refresh()

    def _on_export_probes(self):
        if self._dic_result is None:
            self.results_status_label.setText("Run a correlation first.")
            return
        probes = self.roi_picker.get_probes()
        if not probes:
            self.results_status_label.setText("Place at least one probe first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export probes", "dic_probes.csv", "CSV files (*.csv)")
        if not path:
            return

        probe_indices = [self._nearest_mesh_index(px, py) for px, py in probes]
        scale = self._mm_per_working_px()
        fps = self._effective_fps()
        ref_index = self._dic_result.frame_indices[0]
        kin = self._ensure_kinematics()

        header = [
            "probe_id", "probe_x_px", "probe_y_px", "mesh_row", "mesh_col",
            "frame_index", "time_s",
            "u_px", "v_px", "u_mm", "v_mm", "disp_mag_px", "disp_mag_mm",
            "exx", "eyy", "exy", "eps_eq_von_mises",
            "correlation",
            "vx", "vy", "v_mag", "ax", "ay", "a_mag",
            "exx_rate", "eyy_rate", "exy_rate", "eps_eq_rate_von_mises",
            "dynamic_force_N",
            "e1", "e2", "gmax",   # appended: principal / max shear strain (raw, unsmoothed)
        ]

        n_written = 0
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for p_i, ((px, py), (iy, ix)) in enumerate(zip(probes, probe_indices), start=1):
                    for f_i, fr in enumerate(self._dic_result.frames):
                        time_s = f"{self._dic_time_s(fr.frame_index, ref_index):.6g}" if fps else ""
                        u_px, v_px = float(fr.u[iy, ix]), float(fr.v[iy, ix])
                        u_mm = u_px * scale if scale is not None else float("nan")
                        v_mm = v_px * scale if scale is not None else float("nan")
                        disp_mag_px = float(np.hypot(u_px, v_px))
                        disp_mag_mm = disp_mag_px * scale if scale is not None else float("nan")

                        if kin is not None:
                            vx, vy = kin.vx[f_i][iy, ix], kin.vy[f_i][iy, ix]
                            vmag = kin.v_mag[f_i][iy, ix]
                            ax_, ay_ = kin.ax[f_i][iy, ix], kin.ay[f_i][iy, ix]
                            amag = kin.a_mag[f_i][iy, ix]
                            eps_eq = kin.eps_eq[f_i][iy, ix]
                            exxr, eyyr = kin.exx_rate[f_i][iy, ix], kin.eyy_rate[f_i][iy, ix]
                            exyr = kin.exy_rate[f_i][iy, ix]
                            eqr = kin.eps_eq_rate[f_i][iy, ix]
                            force = kin.dynamic_force[f_i][iy, ix] if kin.dynamic_force is not None else float("nan")
                        else:
                            vx = vy = vmag = ax_ = ay_ = amag = eps_eq = float("nan")
                            exxr = eyyr = exyr = eqr = force = float("nan")

                        writer.writerow([
                            p_i, f"{px:.4g}", f"{py:.4g}", iy, ix,
                            fr.frame_index, time_s,
                            _fmt_val(u_px), _fmt_val(v_px), _fmt_val(u_mm), _fmt_val(v_mm),
                            _fmt_val(disp_mag_px), _fmt_val(disp_mag_mm),
                            _fmt_opt(fr.exx, iy, ix), _fmt_opt(fr.eyy, iy, ix), _fmt_opt(fr.exy, iy, ix),
                            _fmt_val(eps_eq),
                            _fmt_val(fr.correlation[iy, ix]),
                            _fmt_val(vx), _fmt_val(vy), _fmt_val(vmag),
                            _fmt_val(ax_), _fmt_val(ay_), _fmt_val(amag),
                            _fmt_val(exxr), _fmt_val(eyyr), _fmt_val(exyr), _fmt_val(eqr),
                            _fmt_val(force),
                            *_fmt_invariants(fr, iy, ix),
                        ])
                        n_written += 1
        except OSError as exc:
            self.results_status_label.setText(f"Could not export: {exc}")
            return
        self.results_status_label.setText(
            f"Exported {n_written} row(s) ({len(probes)} probe(s)) to {os.path.basename(path)}"
        )

    def _on_mesh_params_changed(self, _value=None):
        subset = self.subset_spin.value()
        step = self.step_spin.value()
        self.project.dic.subset_size_px = subset
        self.project.dic.step_size_px = step
        self.roi_picker.set_mesh_params(subset, step)
        self._update_mesh_status()
        self._invalidate_results()

    def _on_search_radius_changed(self, value):
        self.project.dic.search_radius_px = value
        self._invalidate_results()

    def _on_subpixel_changed(self, _index=None):
        self.project.dic.subpixel_method = self.subpixel_combo.currentData() or "icgn"
        self._invalidate_results()

    def _on_correlation_threshold_changed(self, value):
        self.project.dic.correlation_threshold = value
        self._invalidate_results()

    def _on_neighbor_deviation_changed(self, value):
        self.project.dic.max_neighbor_deviation_px = value
        self._invalidate_results()

    # ==================================================================
    # Results color scale (Auto/Fixed) -- see core/dic_render.py's
    # vmin/vmax; "Fixed" is also used by the video exporter below.
    # ==================================================================

    def _on_color_scale_mode_toggled(self, _checked=None):
        fixed = self.scale_fixed_radio.isChecked()
        self.project.dic.color_scale_mode = "fixed" if fixed else "auto"
        self.scale_min_spin.setEnabled(fixed)
        self.scale_max_spin.setEnabled(fixed)
        self.scale_reset_button.setEnabled(fixed)
        if fixed and self.scale_min_spin.value() == self.scale_max_spin.value():
            self._on_color_scale_reset()
        self._on_result_view_changed()

    def _on_color_scale_bounds_changed(self, _value=None):
        self.project.dic.color_scale_min = self.scale_min_spin.value()
        self.project.dic.color_scale_max = self.scale_max_spin.value()
        self._on_result_view_changed()

    def _on_color_scale_reset(self):
        if self._dic_result is None:
            return
        field_key = self.field_combo.currentData()
        all_finite = []
        for idx, frame_result in enumerate(self._dic_result.frames):
            values, _ = self._field_values(idx, frame_result, field_key)
            finite = values[np.isfinite(values)]
            if finite.size:
                all_finite.append(finite)
        if not all_finite:
            return
        concat = np.concatenate(all_finite)
        vmin, vmax = float(np.min(concat)), float(np.max(concat))
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        self.scale_min_spin.blockSignals(True)
        self.scale_max_spin.blockSignals(True)
        self.scale_min_spin.setValue(vmin)
        self.scale_max_spin.setValue(vmax)
        self.scale_min_spin.blockSignals(False)
        self.scale_max_spin.blockSignals(False)
        self.project.dic.color_scale_min = vmin
        self.project.dic.color_scale_max = vmax
        self._on_result_view_changed()

    def _current_vmin_vmax(self):
        """(vmin, vmax) to use for the Results view/video export -- the
        Fixed color-scale bounds when that mode is selected, else
        (None, None) so `render_field` auto-fits each frame's own data."""
        if self.project.dic.color_scale_mode == "fixed":
            return self.scale_min_spin.value(), self.scale_max_spin.value()
        return None, None

    def _on_force_smoothing_changed(self, value):
        self.project.dic.force_smoothing_window = value
        self.force_smoothing_label.setText("1 (no smoothing)" if value <= 1 else f"{value} frames")
        self._redraw_force_plot()

    def _update_mesh_status(self):
        mesh = self.roi_picker.get_mesh()
        if self.roi_picker.get_shape_mode() == "rectangle":
            roi = self.roi_picker.get_rect()
            if roi is None or mesh is None:
                self.mesh_status_label.setText("Draw a ROI above to preview the correlation mesh.")
                return
            region_desc = f"a {roi.width} x {roi.height} px rectangular ROI"
        else:
            polygon = self.roi_picker.get_polygon()
            if polygon is None or mesh is None:
                self.mesh_status_label.setText(
                    "Draw a polygon ROI above (click each vertex, close it by clicking the first "
                    "vertex again or pressing Enter) to preview the correlation mesh."
                )
                return
            region_desc = f"a {len(polygon)}-vertex polygon ROI"
        if mesh.n_points == 0 or mesh.n_valid_points == 0:
            self.mesh_status_label.setText(
                f"{region_desc.capitalize()} is too small for a {self.subset_spin.value()}px subset."
            )
            return
        self.mesh_status_label.setText(
            f"{mesh.n_points_x} x {mesh.n_points_y} grid, {mesh.n_valid_points} correlation point(s) "
            f"over {region_desc}."
        )

    # ==================================================================
    # Run correlation
    # ==================================================================

    def _set_run_controls_enabled(self, enabled: bool):
        # Everything that could change the frame source, preprocessing, ROI
        # or mesh is locked for the duration of a run: the worker thread
        # reads `self._source` and uses the mesh/crop/downscale captured
        # when the run started, so a mid-run edit here could otherwise
        # either race on the frame source or silently invalidate results
        # that then land anyway when the run completes.
        for widget in (self.video_radio, self.sequence_radio, self.video_row, self.seq_row_widget,
                       self.frame_slider, self.frame_spin, self.crop_picker, self.roi_picker,
                       self.crop_x_spin, self.crop_y_spin, self.crop_w_spin, self.crop_h_spin,
                       self.roi_x_spin, self.roi_y_spin, self.roi_w_spin, self.roi_h_spin,
                       self.roi_rect_radio, self.roi_oval_radio, self.roi_polygon_radio,
                       self.tracking_fixed_radio, self.tracking_moving_radio, self.subpixel_combo,
                       self.downscale_spin, self.frame_step_spin, self.subset_spin, self.step_spin,
                       self.search_radius_spin, self.correlation_threshold_spin,
                       self.neighbor_deviation_spin, self.run_button):
            widget.setEnabled(enabled)
        # Opposite sense from everything else here: only clickable WHILE a
        # run is in flight.
        self.stop_button.setEnabled(not enabled)
        if enabled and self._source is None:
            self.frame_slider.setEnabled(False)
            self.frame_spin.setEnabled(False)

    def _on_run_correlation(self):
        if self._source is None:
            self.run_status_label.setText("Load a video or image sequence first.")
            return
        mesh = self.roi_picker.get_mesh()
        if mesh is None or mesh.n_points == 0:
            self.run_status_label.setText("Draw a ROI that fits at least one correlation point first.")
            return
        frame_indices = self._processing_frame_indices()
        if len(frame_indices) < 2:
            self.run_status_label.setText(
                "Need at least 2 processing frames -- reduce the frame step, or pick an earlier "
                "reference frame."
            )
            return

        crop = self.crop_picker.get_rect()
        downscale = self.downscale_spin.value()
        search_radius = self.search_radius_spin.value()
        tracking_mode = "moving" if self.tracking_moving_radio.isChecked() else "fixed"
        correlation_threshold = self.correlation_threshold_spin.value()
        max_neighbor_deviation = self.neighbor_deviation_spin.value()
        subpixel_method = self.subpixel_combo.currentData() or "icgn"
        source = self._source

        def get_frame(index):
            return preprocess_frame(source.get_frame(index), crop, downscale)

        self._set_run_controls_enabled(False)
        self.progress_bar.setMaximum(len(frame_indices))
        self.progress_bar.setValue(0)
        self.run_status_label.setText(
            f"Running correlation on {len(frame_indices)} frame(s), {mesh.n_valid_points} point(s)…"
        )

        self._cancel_event = threading.Event()
        self._thread = QThread()
        self._worker = _DICWorker(get_frame, mesh, frame_indices, search_radius, tracking_mode,
                                   correlation_threshold, max_neighbor_deviation,
                                   n_workers=self.project.n_workers, cancel_event=self._cancel_event,
                                   subpixel_method=subpixel_method)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_run_progress)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.failed.connect(self._on_run_failed)
        self._worker.cancelled.connect(self._on_run_cancelled)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._thread.start()

    def _on_stop_correlation(self):
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.stop_button.setEnabled(False)
            self.run_status_label.setText("Stopping… (finishing the frame currently being tracked)")

    def _on_run_progress(self, i, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(i)

    def _on_run_cancelled(self):
        self._cancel_event = None
        self._set_run_controls_enabled(True)
        self.run_status_label.setText("Correlation stopped by user.")

    def _on_run_finished(self, result):
        self._cancel_event = None
        self._set_run_controls_enabled(True)
        self._dic_result = result
        self._kinematics = None
        self._uncertainty = None
        self.run_status_label.setText(
            f"Done: tracked {result.mesh.n_valid_points} point(s) across {len(result.frames)} frame(s)."
        )
        self.export_results_button.setEnabled(True)
        self.export_video_button.setEnabled(True)
        self.export_all_mean_button.setEnabled(True)
        self.export_probes_button.setEnabled(bool(self.roi_picker.get_probes()))
        self.result_slider.setEnabled(True)
        self.result_slider.blockSignals(True)
        self.result_slider.setMinimum(0)
        self.result_slider.setMaximum(len(result.frames) - 1)
        self.result_slider.setValue(len(result.frames) - 1)
        self.result_slider.blockSignals(False)
        self._on_result_view_changed()
        self._redraw_all_time_plots()

    def _on_run_failed(self, message):
        self._cancel_event = None
        self._set_run_controls_enabled(True)
        self.run_status_label.setText(f"Correlation failed: {message}")

    # ==================================================================
    # Results
    # ==================================================================

    def _invalidate_results(self):
        self._dic_result = None
        self._kinematics = None
        self._uncertainty = None
        self._last_result_field = None
        self.project.dic_strain_result = None
        self._update_dic_strain_status()
        self._update_uncertainty_status()
        self.export_results_button.setEnabled(False)
        self.export_video_button.setEnabled(False)
        self.export_all_mean_button.setEnabled(False)
        self.export_probes_button.setEnabled(False)
        self.result_slider.setEnabled(False)
        self.result_slider.blockSignals(True)
        self.result_slider.setValue(0)
        self.result_slider.blockSignals(False)
        self.result_frame_label.setText("")
        self.results_status_label.setText("")
        self.video_export_status_label.setText("")
        self.video_progress_bar.setVisible(False)
        self._show_placeholder_canvas(self.result_canvas, "Run a correlation to see results here.")
        self._show_placeholder_canvas(self.mean_time_canvas, "Run a correlation to see this plot.")
        self._show_placeholder_canvas(self.velocity_canvas, "Run a correlation to see this plot.")
        self._show_placeholder_canvas(self.strain_rate_canvas, "Run a correlation to see this plot.")
        self._show_placeholder_canvas(self.force_canvas, "Run a correlation to see this plot.")
        self._show_placeholder_canvas(self.probe_canvas, "Run a correlation to see this plot.")

    @staticmethod
    def _show_placeholder_canvas(canvas, message):
        canvas.clear()
        ax = canvas.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True,
                color=theme.TEXT_MUTED, fontsize=10, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        canvas.refresh()

    def _on_result_view_changed(self, *_args):
        if self._dic_result is None:
            return
        self._update_smoothing_label()
        idx = self.result_slider.value()
        frame_result = self._dic_result.frames[idx]
        self.result_frame_label.setText(f"Frame {frame_result.frame_index}")

        field_key = self.field_combo.currentData()
        values, field_label = self._field_values(idx, frame_result, field_key)

        background = None
        try:
            # The background is THIS frame, not the reference frame -- the
            # point is to see the specimen's actual current shape/position
            # under the field, so a viewer can judge at a glance whether
            # e.g. a strain concentration lines up with a visible crack.
            # `self._source` is None whenever a result is shown with no
            # frames loaded at all -- e.g. right after "Import results
            # (.csv)…", which never requires (re-)loading the original
            # video/sequence -- so the background is simply skipped then,
            # same as any other frame-read failure below.
            if self._source is not None:
                background = preprocess_frame(self._source.get_frame(frame_result.frame_index),
                                               self.crop_picker.get_rect(), self.downscale_spin.value())
        except (ValueError, IndexError, OSError):
            pass  # background is a nice-to-have; the field below is the actual result

        vmin, vmax = self._current_vmin_vmax()
        self.result_canvas.clear()
        ax = self.result_canvas.figure.add_subplot(111)
        # frame_result.x/.y are each point's ACTUAL tracked position this
        # frame (not its reference position), so the field is drawn where
        # the tracked material actually is right now; a point with no
        # reliable position (lost track, excluded by a quality filter) is
        # simply left out rather than plotted at a placeholder -- see
        # core/dic_render.py for why that matters once deformation is large.
        render_field(self.result_canvas.figure, ax, background, frame_result.x, frame_result.y,
                     values, field_label, vmin=vmin, vmax=vmax, cmap=RESULT_CMAP,
                     lost_track_color=theme.DANGER,
                     max_edge_px=default_max_edge_px(self._dic_result.mesh.step))
        self.result_canvas.refresh()
        # Cached for the canvas's cursor-value readout (_result_cursor_value)
        # -- the field's own values, and where each one is actually drawn
        # THIS frame (frame_result.x/.y, same tracked positions render_field
        # itself plotted against), not the mesh's fixed reference positions.
        self._last_result_field = {
            "x": np.ravel(np.asarray(frame_result.x, dtype=float)),
            "y": np.ravel(np.asarray(frame_result.y, dtype=float)),
            "values": np.ravel(np.asarray(values, dtype=float)),
            "label": field_label,
        }

    def _result_cursor_value(self, xdata, ydata):
        """Value-lookup callback for the Results canvas's cursor readout
        (see ui/plot_canvas.py's PlotCanvas.set_value_lookup): the
        displayed field's value at whichever successfully-tracked point
        is closest -- in this frame's ACTUAL tracked position, matching
        what's actually drawn under the cursor -- to the mouse, so
        hovering over the isocontour reads off the number behind the
        color there. A nearest-point lookup (like the probe system's own
        `_nearest_mesh_index`) rather than re-interpolating the
        triangulation `render_field` draws with, since that's cheap
        enough to run on every mouse-move and reads off an actual
        measured point rather than the contour's own smoothed
        in-between value. None while nothing is cached yet (e.g. no
        correlation has completed) or every point is NaN this frame."""
        cached = self._last_result_field
        if cached is None:
            return None
        x, y, values = cached["x"], cached["y"], cached["values"]
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
        if not np.any(finite):
            return None
        xf, yf, vf = x[finite], y[finite], values[finite]
        dist2 = (xf - xdata) ** 2 + (yf - ydata) ** 2
        i = int(np.argmin(dist2))
        return f"|  {cached['label']}: {vf[i]:.4g}  (nearest point, {np.sqrt(dist2[i]):.1f} px away)"

    def _field_values(self, idx, frame_result, field_key):
        """Returns (values array, colorbar label) for `field_key`,
        converting displacement/velocity/acceleration to physical units
        (mm) when a pixel size is calibrated -- strain and strain rate
        are dimensionless/per-second and unaffected. `idx` is this
        frame's position in `self._dic_result.frames` (same order the
        cached kinematics -- see `_ensure_kinematics` -- are stored in),
        needed for every field beyond plain displacement/strain/
        correlation."""
        label_by_key = {key: label for key, label in FIELD_CHOICES}
        base_label = label_by_key.get(field_key, field_key)
        scale = self._mm_per_working_px()
        unit = "mm" if scale is not None else "px"
        factor = scale if scale is not None else 1.0

        # Optional spatial smoothing (Results card, "Spatial smoothing"):
        # NaN-aware Gaussian filter on the mesh grid. Applied to the
        # COMPONENTS first and the non-linear derived fields (magnitudes,
        # principal/equivalent strains) computed from the smoothed
        # components, so noise does not leave a positive bias on them.
        if field_key == "correlation":
            return frame_result.correlation, f"{base_label} (-)"   # a quality measure: never smoothed
        sigma = self._smoothing_sigma()
        sm = (lambda a: smooth_nan_gaussian(a, sigma)) if sigma > 0 else (lambda a: np.asarray(a, dtype=float))
        if sigma > 0:
            base_label = f"{base_label}, smoothed (sigma = {sigma:g} pt)"

        if field_key == "disp_mag":
            return np.hypot(sm(frame_result.u), sm(frame_result.v)) * factor, f"{base_label} ({unit})"
        if field_key == "u":
            return sm(frame_result.u) * factor, f"{base_label} ({unit})"
        if field_key == "v":
            return sm(frame_result.v) * factor, f"{base_label} ({unit})"
        if field_key in ("exx", "eyy", "exy"):
            return sm(_or_nan(getattr(frame_result, field_key), frame_result.u)), f"{base_label} (-)"
        if field_key in ("e1", "e2", "gmax"):
            exx = sm(_or_nan(frame_result.exx, frame_result.u))
            eyy = sm(_or_nan(frame_result.eyy, frame_result.u))
            exy = sm(_or_nan(frame_result.exy, frame_result.u))
            if field_key == "gmax":
                return max_shear_strain(exx, eyy, exy), f"{base_label} (-)"
            e1, e2 = principal_strains(exx, eyy, exy)
            return (e1 if field_key == "e1" else e2), f"{base_label} (-)"

        # Everything below is a time-derivative (or built from one) --
        # needs the cached kinematics, which in turn need the camera
        # frame rate above (see _ensure_kinematics).
        kin = self._ensure_kinematics()
        blank = np.full(frame_result.u.shape, np.nan)
        if kin is None:
            return blank, f"{base_label} -- enter the camera frame rate above"

        kin_unit = kin.length_unit
        if field_key == "v_mag":
            return np.hypot(sm(kin.vx[idx]), sm(kin.vy[idx])), f"{base_label} ({kin_unit}/s)"
        if field_key == "vx":
            return sm(kin.vx[idx]), f"{base_label} ({kin_unit}/s)"
        if field_key == "vy":
            return sm(kin.vy[idx]), f"{base_label} ({kin_unit}/s)"
        if field_key == "a_mag":
            return np.hypot(sm(kin.ax[idx]), sm(kin.ay[idx])), f"{base_label} ({kin_unit}/s^2)"
        if field_key == "ax":
            return sm(kin.ax[idx]), f"{base_label} ({kin_unit}/s^2)"
        if field_key == "ay":
            return sm(kin.ay[idx]), f"{base_label} ({kin_unit}/s^2)"
        if field_key == "eps_eq":
            if sigma > 0:
                return von_mises_strain(sm(_or_nan(frame_result.exx, frame_result.u)),
                                        sm(_or_nan(frame_result.eyy, frame_result.u)),
                                        sm(_or_nan(frame_result.exy, frame_result.u))), f"{base_label} (-)"
            return kin.eps_eq[idx], f"{base_label} (-)"
        if field_key == "exx_rate":
            return sm(kin.exx_rate[idx]), f"{base_label} (1/s)"
        if field_key == "eyy_rate":
            return sm(kin.eyy_rate[idx]), f"{base_label} (1/s)"
        if field_key == "exy_rate":
            return sm(kin.exy_rate[idx]), f"{base_label} (1/s)"
        if field_key == "eps_eq_rate":
            if sigma > 0:
                return von_mises_strain(sm(kin.exx_rate[idx]), sm(kin.eyy_rate[idx]),
                                        sm(kin.exy_rate[idx])), f"{base_label} (1/s)"
            return kin.eps_eq_rate[idx], f"{base_label} (1/s)"
        if field_key == "dynamic_force":
            if kin.dynamic_force is None:
                return blank, f"{base_label} -- enter pixel size, density and specimen geometry above"
            return sm(kin.dynamic_force[idx]), f"{base_label} (N)"
        return frame_result.correlation, f"{base_label} (-)"

    def _smoothing_sigma(self) -> float:
        """Spatial smoothing of the displayed fields, in mesh points (0 = off)."""
        try:
            return max(0.0, float(self.project.dic.spatial_smoothing_sigma or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _on_spatial_smoothing_changed(self, value=None):
        sigma = float(self.smoothing_spin.value())
        self.project.dic.spatial_smoothing_sigma = sigma
        self._update_smoothing_label()
        if self._dic_result is None:
            return
        self._on_result_view_changed()
        self._redraw_probe_plot()

    def _update_smoothing_label(self):
        sigma = float(self.smoothing_spin.value())
        if sigma <= 0:
            self.smoothing_label.setText("off")
            return
        step = None
        if self._dic_result is not None:
            step = getattr(self._dic_result.mesh, "step", None)
        else:
            step = self.step_spin.value() if hasattr(self, "step_spin") else None
        extent = f" (about {sigma * step:.0f} px on the image)" if step else ""
        self.smoothing_label.setText(f"Gaussian, sigma = {sigma:g} mesh point(s){extent}")

    def _time_axis(self):
        """(x values, x-axis label) shared by every ROI-mean-vs-time
        plot -- against elapsed time (scale/offset-corrected, see
        `_dic_time_s`) when a camera frame rate is given, otherwise
        against raw frame index."""
        fps = self._effective_fps()
        ref_index = self._dic_result.frame_indices[0]
        if fps:
            x = [self._dic_time_s(fr.frame_index, ref_index) for fr in self._dic_result.frames]
            xlabel = "time (s)"
        else:
            x = [fr.frame_index for fr in self._dic_result.frames]
            xlabel = "frame index (enter the camera frame rate above for a time axis)"
        return x, xlabel

    def _redraw_all_time_plots(self):
        self._update_uncertainty_status()
        self._redraw_mean_vs_time_plot()
        self._publish_dic_strain()
        self._redraw_velocity_plot()
        self._redraw_strain_rate_plot()
        self._redraw_force_plot()
        self._redraw_probe_plot()

    def _on_uncertainty_display_changed(self, _value=None):
        """The "Show uncertainty envelopes" checkbox or "Envelope width"
        combo changed -- these are pure DISPLAY settings (the underlying
        `self._uncertainty` model doesn't depend on either), so this
        just persists them and redraws, no cache invalidation needed."""
        self.project.dic.show_uncertainty = self.uncertainty_show_check.isChecked()
        self.project.dic.uncertainty_sigma_k = self.uncertainty_width_combo.currentData() or 1.0
        self._update_uncertainty_status()
        self._redraw_mean_vs_time_plot()
        self._publish_dic_strain()
        self._redraw_velocity_plot()
        self._redraw_strain_rate_plot()
        self._redraw_force_plot()

    def _update_uncertainty_status(self):
        if self._dic_result is None:
            self.uncertainty_status_label.setText("Run a correlation to estimate a measurement uncertainty.")
            return
        unc = self._ensure_uncertainty()
        if unc is None:
            reason = getattr(self._dic_result, "uncertainty_error", None)
            detail = f" Cause: {reason}" if reason else " Cause unknown (no active correlation point?)."
            self.uncertainty_status_label.setText(
                "No uncertainty estimate could be computed for this run (see core/dic_uncertainty.py)."
                + detail
            )
            return
        k = self.uncertainty_width_combo.currentData() or 1.0
        # Same effective-N-reduced quantity the shaded envelopes plot
        # (see `_roi_mean_sigma`) -- NOT a plain nanmean of the raw
        # per-point sigma field, which is a single point's own
        # uncertainty and is always larger (often much larger) than the
        # uncertainty of the ROI-wide mean these curves actually show.
        # And `sigma_disp_mag` specifically -- not `sigma_u` -- to match
        # the ROI Mean vs Time plot's own displacement envelope, which
        # is drawn on the |displacement| magnitude curve, not on u
        # alone. Reporting sigma_u there before made this text's number
        # both larger than, AND about a different quantity than, the
        # band actually drawn on the plots.
        sigma_u_last = self._roi_mean_sigma(unc.sigma_disp_mag[-1])
        sigma_eps_last = self._roi_mean_sigma(unc.sigma_eps_eq[-1])
        noise_txt = (f"{self._dic_result.noise_sigma:.2f}" if self._dic_result.noise_sigma is not None
                     else "n/a")
        self.uncertainty_status_label.setText(
            f"Estimated +/-{k:.0f} sigma on the ROI-mean curves at the last processed frame: "
            f"displacement ~{k * sigma_u_last:.3g} {unc.length_unit}, MISES strain ~"
            f"{k * sigma_eps_last:.3g} (-) -- matches the shaded band width, not a single point's own "
            f"(larger) uncertainty. Reference-frame noise estimate ~{noise_txt} intensity levels (8-bit)."
        )

    def _on_dic_strain_definition_changed(self, _index=None):
        self.project.dic.strain_definition = self.dic_strain_combo.currentData() or "eps_eq"
        self.project.dic.strain_convert_to_true = self.dic_strain_true_check.isChecked()
        self._publish_dic_strain()

    def _on_dic_time_correction_changed(self, _value=None):
        """The "Time scale" / "Time offset" spinboxes changed -- these
        feed `_effective_fps()`/`_dic_time_s()`, which every DIC time
        axis on this page goes through, so the cached kinematics (whose
        velocity/acceleration/strain-rate depend on dt, hence on the
        scale) must be invalidated and everything time-dependent
        redrawn, not just the DIC Strain series."""
        self.project.dic.time_scale = self.dic_time_scale_spin.value() or 1.0
        self.project.dic.time_offset_s = self.dic_time_offset_spin.value()
        self._kinematics = None
        self._uncertainty = None
        self._on_result_view_changed()
        self._redraw_all_time_plots()

    def _publish_dic_strain(self):
        """Recomputes and (re)publishes `Project.dic_strain_result` --
        the ROI-mean time series of whichever strain definition is
        chosen in the combo above (and its matching strain-rate series)
        -- so Results & Export can offer it as "DIC Strain (-)" /
        "DIC Strain Rate (1/s)". Called after every correlation run and
        whenever anything the series depends on changes (the definition
        combo, the true-strain checkbox, the camera frame rate, ...) --
        see `_redraw_all_time_plots`.

        When the "Convert to true strain" checkbox is on (and the
        definition isn't "exy", which has no such conversion), the
        conversion (core.dic_kinematics.nominal_to_true_strain /
        _rate) is applied to each mesh point's OWN strain/rate field
        BEFORE taking the ROI mean -- not to the already-averaged
        scalar series -- since the conversion is nonlinear (ln) and
        averaging first would introduce a small systematic bias
        (Jensen's inequality) versus converting each point's own
        measurement first."""
        definition = self.dic_strain_combo.currentData() or "eps_eq"
        convert_to_true = self.dic_strain_true_check.isChecked() and definition != "exy"
        if self._dic_result is None:
            self.project.dic_strain_result = None
            self._update_dic_strain_status()
            return

        if definition == "eps_eq":
            kin = self._ensure_kinematics()
            if kin is None:
                self.project.dic_strain_result = None
                self._update_dic_strain_status()
                return
            strain_fields = list(kin.eps_eq)
            rate_fields = list(kin.eps_eq_rate)
        else:
            strain_fields = [
                getattr(fr, definition) if getattr(fr, definition) is not None
                else np.full(fr.u.shape, np.nan)
                for fr in self._dic_result.frames
            ]
            kin = self._ensure_kinematics()
            if kin is not None:
                rate_fields = list(getattr(kin, f"{definition}_rate"))
            else:
                rate_fields = [np.full(f.shape, np.nan) for f in strain_fields]

        # Matching uncertainty fields (see core/dic_uncertainty.py and the
        # "DIC Measurement Uncertainty" card) -- None if no uncertainty
        # model could be built for this run, in which case the published
        # result simply carries no "*_uncertainty" entries.
        unc = self._ensure_uncertainty()
        if unc is not None:
            sigma_strain_fields = list(getattr(unc, "sigma_eps_eq" if definition == "eps_eq" else f"sigma_{definition}"))
            rate_attr = "sigma_eps_eq_rate" if definition == "eps_eq" else f"sigma_{definition}_rate"
            sigma_rate_attr_value = getattr(unc, rate_attr)
            sigma_rate_fields = (list(sigma_rate_attr_value) if sigma_rate_attr_value is not None
                                 else [np.full(f.shape, np.nan) for f in sigma_strain_fields])
        else:
            sigma_strain_fields = sigma_rate_fields = None

        if convert_to_true:
            nominal_strain_fields = strain_fields  # keep the pre-conversion (nominal) fields for the rate formula
            strain_fields = [nominal_to_true_strain(f) for f in nominal_strain_fields]
            rate_fields = [
                nominal_to_true_strain_rate(r, s) for r, s in zip(rate_fields, nominal_strain_fields)
            ]
            if sigma_strain_fields is not None:
                # Same first-order derivative (d/de[ln(1+e)] = 1/(1+e)) as
                # `nominal_to_true_strain_rate` -- reused here to propagate a
                # STATIC per-point uncertainty through the same nonlinear
                # map, still done BEFORE averaging (see this method's own
                # docstring on why). The rate sigma reuses the same formula
                # as an approximation -- it ignores the (usually smaller)
                # extra term from the nominal strain's OWN uncertainty
                # entering that ratio, not just the rate's.
                sigma_strain_fields = [
                    nominal_to_true_strain_rate(s, e) for s, e in zip(sigma_strain_fields, nominal_strain_fields)
                ]
                sigma_rate_fields = [
                    nominal_to_true_strain_rate(s, e) for s, e in zip(sigma_rate_fields, nominal_strain_fields)
                ]

        strain = np.array([np.nanmean(a) for a in strain_fields])
        strain_rate = np.array([np.nanmean(a) for a in rate_fields])

        if sigma_strain_fields is not None:
            roi_area = self._roi_area_px2()
            subset_size = self._dic_result.mesh.subset_size
            k = self.uncertainty_width_combo.currentData() or 1.0
            strain_uncertainty = np.array([
                k * roi_mean_uncertainty(
                    s, effective_independent_count(int(np.count_nonzero(np.isfinite(s))), roi_area, subset_size)
                ) for s in sigma_strain_fields
            ])
            strain_rate_uncertainty = np.array([
                k * roi_mean_uncertainty(
                    s, effective_independent_count(int(np.count_nonzero(np.isfinite(s))), roi_area, subset_size)
                ) for s in sigma_rate_fields
            ])
        else:
            strain_uncertainty = strain_rate_uncertainty = None

        fps = self._effective_fps()
        if fps:
            ref_index = self._dic_result.frame_indices[0]
            time_s = np.array(
                [self._dic_time_s(fr.frame_index, ref_index) for fr in self._dic_result.frames], dtype=float
            )
        else:
            time_s = None

        self.project.dic_strain_result = {
            "definition": definition,
            "label": _DIC_STRAIN_SHORT_NAMES.get(definition, definition),
            "time_s": time_s,
            "strain": strain,
            "strain_rate": strain_rate,
            "strain_uncertainty": strain_uncertainty,
            "strain_rate_uncertainty": strain_rate_uncertainty,
            "converted_to_true": convert_to_true,
        }
        self._update_dic_strain_status()

    def _update_dic_strain_status(self):
        result = self.project.dic_strain_result
        if result is None:
            self.dic_strain_status_label.setText(
                "Run a correlation to publish a DIC strain series to Results & Export."
            )
            return
        kind = "true (logarithmic)" if result.get("converted_to_true") else "small/nominal (linearized, NOT converted)"
        if result["time_s"] is None:
            self.dic_strain_status_label.setText(
                f"Computed DIC Strain ({result['label']}, {kind} strain), {len(result['strain'])} "
                "point(s), but it has no time-in-seconds axis yet -- enter the camera frame rate "
                "above (Calibration card) to make it selectable on the Results & Export page."
            )
        else:
            scale = self.project.dic.time_scale
            offset = self.project.dic.time_offset_s
            correction = (
                f" (time scale x{scale:.6g}, offset {offset:+.6g} s applied)"
                if (scale != 1.0 or offset != 0.0) else " (time scale/offset left at default: x1, +0 s)"
            )
            self.dic_strain_status_label.setText(
                f"Published DIC Strain ({result['label']}, {kind} strain), {len(result['strain'])} "
                f"point(s) over {result['time_s'][-1]:.4g} s, to Results & Export{correction}. There "
                "is no camera-to-bar-signal synchronization in this app -- use the Time scale/Time "
                "offset fields above to line this series up with the standard result's own time "
                "axis (e.g. by matching a feature visible in both) if DIC and the bar signals "
                "weren't triggered together or the camera's true frame rate differs from the "
                "entered one."
            )

    def _redraw_mean_vs_time_plot(self):
        if self._dic_result is None:
            return
        x, xlabel = self._time_axis()
        scale = self._mm_per_working_px()
        unit = "mm" if scale is not None else "px"
        factor = scale if scale is not None else 1.0

        mean_u = [np.nanmean(fr.u) * factor for fr in self._dic_result.frames]
        mean_v = [np.nanmean(fr.v) * factor for fr in self._dic_result.frames]
        mean_mag = [np.nanmean(np.hypot(fr.u, fr.v)) * factor for fr in self._dic_result.frames]
        mean_exx = [np.nanmean(fr.exx) if fr.exx is not None else np.nan for fr in self._dic_result.frames]
        mean_eyy = [np.nanmean(fr.eyy) if fr.eyy is not None else np.nan for fr in self._dic_result.frames]
        mean_exy = [np.nanmean(fr.exy) if fr.exy is not None else np.nan for fr in self._dic_result.frames]
        # eps_eq (MISES) needs the cached kinematics (it's derived from
        # exx/eyy/exy, but only computed there -- see _ensure_kinematics),
        # so it's None (and simply not plotted below) without a frame rate.
        kin = self._ensure_kinematics()
        mean_eq = [np.nanmean(a) for a in kin.eps_eq] if kin is not None else None

        unc = self._ensure_uncertainty()
        sigma_disp_mag = unc.sigma_disp_mag if unc is not None else None
        sigma_exx = unc.sigma_exx if unc is not None else None
        sigma_eyy = unc.sigma_eyy if unc is not None else None
        sigma_exy = unc.sigma_exy if unc is not None else None
        sigma_eq = unc.sigma_eps_eq if (unc is not None and mean_eq is not None) else None

        self.mean_time_canvas.clear()
        ax1 = self.mean_time_canvas.figure.add_subplot(211)
        ax1.plot(x, mean_u, "o-", label=f"mean u ({unit})", color=theme.ACCENT, markersize=3)
        ax1.plot(x, mean_v, "o-", label=f"mean v ({unit})", color="#ff9f43", markersize=3)
        ax1.plot(x, mean_mag, "o-", label=f"mean |displacement| ({unit})", color=theme.SUCCESS, markersize=3)
        # Only |displacement| gets an envelope here -- u/v are plotted mainly
        # for reference and shading all three would clutter the subplot.
        self._add_envelope(ax1, x, mean_mag, sigma_disp_mag, theme.SUCCESS)
        ax1.set_ylabel(f"displacement ({unit})")
        ax1.legend(fontsize=8)

        ax2 = self.mean_time_canvas.figure.add_subplot(212, sharex=ax1)
        ax2.plot(x, mean_exx, "o-", label="mean exx", color=theme.ACCENT, markersize=3)
        ax2.plot(x, mean_eyy, "o-", label="mean eyy", color="#ff9f43", markersize=3)
        ax2.plot(x, mean_exy, "o-", label="mean exy", color=theme.SUCCESS, markersize=3)
        self._add_envelope(ax2, x, mean_exx, sigma_exx, theme.ACCENT)
        self._add_envelope(ax2, x, mean_eyy, sigma_eyy, "#ff9f43")
        self._add_envelope(ax2, x, mean_exy, sigma_exy, theme.SUCCESS)
        if mean_eq is not None:
            ax2.plot(x, mean_eq, "o-", label="mean eps_eq (MISES)", color=theme.DANGER, markersize=3)
            self._add_envelope(ax2, x, mean_eq, sigma_eq, theme.DANGER)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("strain (-)")
        ax2.legend(fontsize=8)
        self.mean_time_canvas.refresh()

    def _redraw_velocity_plot(self):
        if self._dic_result is None:
            self._show_placeholder_canvas(self.velocity_canvas, "Run a correlation to see this plot.")
            return
        kin = self._ensure_kinematics()
        if kin is None:
            self._show_placeholder_canvas(
                self.velocity_canvas, "Enter the camera frame rate above to see velocity."
            )
            return
        x, xlabel = self._time_axis()
        unit = kin.length_unit
        mean_vx = [np.nanmean(a) for a in kin.vx]
        mean_vy = [np.nanmean(a) for a in kin.vy]
        mean_vmag = [np.nanmean(a) for a in kin.v_mag]

        unc = self._ensure_uncertainty()

        self.velocity_canvas.clear()
        ax = self.velocity_canvas.figure.add_subplot(111)
        ax.plot(x, mean_vx, "o-", label=f"mean vx ({unit}/s)", color=theme.ACCENT, markersize=3)
        ax.plot(x, mean_vy, "o-", label=f"mean vy ({unit}/s)", color="#ff9f43", markersize=3)
        ax.plot(x, mean_vmag, "o-", label=f"mean |v| ({unit}/s)", color=theme.SUCCESS, markersize=3)
        if unc is not None:
            self._add_envelope(ax, x, mean_vx, unc.sigma_vx, theme.ACCENT)
            self._add_envelope(ax, x, mean_vy, unc.sigma_vy, "#ff9f43")
            self._add_envelope(ax, x, mean_vmag, unc.sigma_v_mag, theme.SUCCESS)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"velocity ({unit}/s)")
        ax.legend(fontsize=8)
        self.velocity_canvas.refresh()

    def _redraw_strain_rate_plot(self):
        if self._dic_result is None:
            self._show_placeholder_canvas(self.strain_rate_canvas, "Run a correlation to see this plot.")
            return
        kin = self._ensure_kinematics()
        if kin is None:
            self._show_placeholder_canvas(
                self.strain_rate_canvas, "Enter the camera frame rate above to see strain rate."
            )
            return
        x, xlabel = self._time_axis()
        mean_exx_r = [np.nanmean(a) for a in kin.exx_rate]
        mean_eyy_r = [np.nanmean(a) for a in kin.eyy_rate]
        mean_exy_r = [np.nanmean(a) for a in kin.exy_rate]
        mean_eq_r = [np.nanmean(a) for a in kin.eps_eq_rate]

        unc = self._ensure_uncertainty()

        self.strain_rate_canvas.clear()
        ax = self.strain_rate_canvas.figure.add_subplot(111)
        ax.plot(x, mean_exx_r, "o-", label="mean exx rate (1/s)", color=theme.ACCENT, markersize=3)
        ax.plot(x, mean_eyy_r, "o-", label="mean eyy rate (1/s)", color="#ff9f43", markersize=3)
        ax.plot(x, mean_exy_r, "o-", label="mean exy rate (1/s)", color=theme.SUCCESS, markersize=3)
        ax.plot(x, mean_eq_r, "o-", label="mean eq. (von Mises) rate (1/s)", color=theme.DANGER, markersize=3)
        if unc is not None:
            self._add_envelope(ax, x, mean_exx_r, unc.sigma_exx_rate, theme.ACCENT)
            self._add_envelope(ax, x, mean_eyy_r, unc.sigma_eyy_rate, "#ff9f43")
            self._add_envelope(ax, x, mean_exy_r, unc.sigma_exy_rate, theme.SUCCESS)
            self._add_envelope(ax, x, mean_eq_r, unc.sigma_eps_eq_rate, theme.DANGER)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("strain rate (1/s)")
        ax.legend(fontsize=8)
        self.strain_rate_canvas.refresh()

    def _redraw_force_plot(self):
        if self._dic_result is None:
            self._show_placeholder_canvas(self.force_canvas, "Run a correlation to see this plot.")
            return
        kin = self._ensure_kinematics()
        if kin is None or kin.dynamic_force is None:
            self._show_placeholder_canvas(
                self.force_canvas,
                "Enter the camera frame rate, pixel size, material density, and specimen "
                "width/thickness above to see dynamic force.",
            )
            return
        x, xlabel = self._time_axis()
        mean_force_raw = np.array([np.nanmean(a) for a in kin.dynamic_force])
        window = self.force_smoothing_slider.value()
        mean_force_smoothed = smooth_series(mean_force_raw, window)

        unc = self._ensure_uncertainty()

        self.force_canvas.clear()
        ax = self.force_canvas.figure.add_subplot(111)
        ax.plot(x, mean_force_raw, "o-", label="mean dynamic force (raw)", color=theme.TEXT_MUTED,
                markersize=3, alpha=0.5)
        ax.plot(x, mean_force_smoothed, "-", label=f"mean dynamic force (smoothed, window={window})",
                color=theme.DANGER, linewidth=2)
        if unc is not None:
            # The envelope is shown around the SMOOTHED curve (what the raw one's
            # own uncertainty implies for it, not a separately-smoothed sigma).
            self._add_envelope(ax, x, mean_force_smoothed, unc.sigma_dynamic_force, theme.DANGER)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("dynamic force, F = rho*A*a (N)")
        ax.legend(fontsize=8)
        self.force_canvas.refresh()

    def _on_export_all_mean_vs_time(self):
        if self._dic_result is None:
            self.results_status_label.setText("Run a correlation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export ROI Mean vs Time", "dic_roi_mean_vs_time.csv", "CSV files (*.csv)",
        )
        if not path:
            return

        fps = self._effective_fps()
        ref_index = self._dic_result.frame_indices[0]
        scale = self._mm_per_working_px()
        disp_unit = "mm" if scale is not None else "px"
        disp_factor = scale if scale is not None else 1.0

        kin = self._ensure_kinematics()
        kin_unit = kin.length_unit if kin is not None else disp_unit

        window = self.force_smoothing_slider.value()
        force_raw = force_smoothed = None
        if kin is not None and kin.dynamic_force is not None:
            force_raw = np.array([np.nanmean(a) for a in kin.dynamic_force])
            force_smoothed = smooth_series(force_raw, window)

        header = [
            "frame_index", "time_s",
            f"mean_u_{disp_unit}", f"mean_v_{disp_unit}", f"mean_disp_mag_{disp_unit}",
            "mean_exx", "mean_eyy", "mean_exy", "mean_eps_eq_von_mises",
            f"mean_vx_{kin_unit}_per_s", f"mean_vy_{kin_unit}_per_s", f"mean_v_mag_{kin_unit}_per_s",
            f"mean_ax_{kin_unit}_per_s2", f"mean_ay_{kin_unit}_per_s2", f"mean_a_mag_{kin_unit}_per_s2",
            "mean_exx_rate_per_s", "mean_eyy_rate_per_s", "mean_exy_rate_per_s",
            "mean_eps_eq_rate_von_mises_per_s",
            "mean_dynamic_force_raw_N", f"mean_dynamic_force_smoothed_window{window}_N",
        ]

        n_written = 0
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i, fr in enumerate(self._dic_result.frames):
                    time_s = f"{self._dic_time_s(fr.frame_index, ref_index):.6g}" if fps else ""
                    mean_u = np.nanmean(fr.u) * disp_factor
                    mean_v = np.nanmean(fr.v) * disp_factor
                    mean_mag = np.nanmean(np.hypot(fr.u, fr.v)) * disp_factor
                    mean_exx = np.nanmean(fr.exx) if fr.exx is not None else np.nan
                    mean_eyy = np.nanmean(fr.eyy) if fr.eyy is not None else np.nan
                    mean_exy = np.nanmean(fr.exy) if fr.exy is not None else np.nan

                    if kin is not None:
                        mean_eps_eq = np.nanmean(kin.eps_eq[i])
                        mean_vx, mean_vy = np.nanmean(kin.vx[i]), np.nanmean(kin.vy[i])
                        mean_vmag = np.nanmean(kin.v_mag[i])
                        mean_ax, mean_ay = np.nanmean(kin.ax[i]), np.nanmean(kin.ay[i])
                        mean_amag = np.nanmean(kin.a_mag[i])
                        mean_exxr = np.nanmean(kin.exx_rate[i])
                        mean_eyyr = np.nanmean(kin.eyy_rate[i])
                        mean_exyr = np.nanmean(kin.exy_rate[i])
                        mean_eqr = np.nanmean(kin.eps_eq_rate[i])
                    else:
                        mean_eps_eq = np.nan
                        mean_vx = mean_vy = mean_vmag = np.nan
                        mean_ax = mean_ay = mean_amag = np.nan
                        mean_exxr = mean_eyyr = mean_exyr = mean_eqr = np.nan

                    force_raw_val = force_raw[i] if force_raw is not None else np.nan
                    force_smoothed_val = force_smoothed[i] if force_smoothed is not None else np.nan

                    writer.writerow([
                        fr.frame_index, time_s,
                        _fmt_val(mean_u), _fmt_val(mean_v), _fmt_val(mean_mag),
                        _fmt_val(mean_exx), _fmt_val(mean_eyy), _fmt_val(mean_exy), _fmt_val(mean_eps_eq),
                        _fmt_val(mean_vx), _fmt_val(mean_vy), _fmt_val(mean_vmag),
                        _fmt_val(mean_ax), _fmt_val(mean_ay), _fmt_val(mean_amag),
                        _fmt_val(mean_exxr), _fmt_val(mean_eyyr), _fmt_val(mean_exyr), _fmt_val(mean_eqr),
                        _fmt_val(force_raw_val), _fmt_val(force_smoothed_val),
                    ])
                    n_written += 1
        except OSError as exc:
            self.results_status_label.setText(f"Could not export: {exc}")
            return
        self.results_status_label.setText(f"Exported {n_written} row(s) to {os.path.basename(path)}")

    def _on_export_results(self):
        if self._dic_result is None:
            self.results_status_label.setText("Run a correlation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export DIC results", "dic_results.csv", "CSV files (*.csv)")
        if not path:
            return

        mesh = self._dic_result.mesh
        n_rows, n_cols = mesh.n_points_y, mesh.n_points_x
        xx0, yy0 = mesh.grid_xy()
        scale = self._mm_per_working_px()  # None, or mm per (preprocessed-frame) pixel
        fps = self._effective_fps()
        ref_index = self._dic_result.frame_indices[0]
        n_written = 0
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_index", "time_s", "point_row", "point_col", "x0", "y0",
                                  "x", "y", "u", "v", "u_mm", "v_mm", "exx", "eyy", "exy", "correlation"])
                for fr in self._dic_result.frames:
                    time_s = f"{self._dic_time_s(fr.frame_index, ref_index):.6g}" if fps else ""
                    for iy in range(n_rows):
                        for ix in range(n_cols):
                            u_mm = f"{fr.u[iy, ix] * scale:.6g}" if scale is not None else ""
                            v_mm = f"{fr.v[iy, ix] * scale:.6g}" if scale is not None else ""
                            writer.writerow([
                                fr.frame_index, time_s, iy, ix,
                                f"{xx0[iy, ix]:.4g}", f"{yy0[iy, ix]:.4g}",
                                f"{fr.x[iy, ix]:.4g}", f"{fr.y[iy, ix]:.4g}",
                                f"{fr.u[iy, ix]:.4g}", f"{fr.v[iy, ix]:.4g}", u_mm, v_mm,
                                _fmt_opt(fr.exx, iy, ix), _fmt_opt(fr.eyy, iy, ix), _fmt_opt(fr.exy, iy, ix),
                                f"{fr.correlation[iy, ix]:.4g}",
                            ])
                            n_written += 1
        except OSError as exc:
            self.results_status_label.setText(f"Could not export: {exc}")
            return
        self.results_status_label.setText(f"Exported {n_written} rows to {os.path.basename(path)}")

    def _on_import_results(self):
        """Reopens a CSV previously written by `_on_export_results` --
        either from this same session or a genuinely past run, on this
        machine or another one -- without re-running the correlation.

        Wires the reconstructed `DICResult` in through the exact same
        `_on_run_finished` finalization every real correlation run
        already goes through (enabling the export buttons, sizing the
        result slider, redrawing every plot, publishing the strain
        series to Results & Export) -- see core/dic_result_io.py's module
        docstring for exactly what is/isn't recoverable from the file.

        Deliberately never touches subset_spin/step_spin/frame_step_spin/
        search_radius_spin: their change handlers all call
        `_invalidate_results()`, which would immediately discard the
        result this just loaded."""
        path, _ = QFileDialog.getOpenFileName(self, "Import DIC results", "", "CSV files (*.csv)")
        if not path:
            return
        try:
            imported = load_dic_result_csv(path)
        except DicResultLoadError as exc:
            self.results_status_label.setText(str(exc))
            return

        self._on_run_finished(imported.result)

        notes = []
        if imported.mm_per_working_px is not None:
            # The recovered number is mm per WORKING (already-downscaled)
            # pixel -- setting downscale to 1 and putting the whole
            # recovered value into the pixel-size field reproduces
            # `_mm_per_working_px() == pixel_size * downscale` correctly
            # regardless of the original run's own downscale factor,
            # which isn't separately recoverable from the export.
            self.downscale_spin.blockSignals(True)
            self.downscale_spin.setValue(1)
            self.downscale_spin.blockSignals(False)
            self.project.dic.downscale_factor = 1
            self.pixel_size_edit.setText(_fmt_num(imported.mm_per_working_px))
            notes.append("pixel size recovered from u_mm/v_mm")
        else:
            notes.append("pixel size not recovered (no u_mm/v_mm column, or it was blank at export)")
        if imported.fps is not None:
            self.project.dic.time_scale = 1.0
            self.project.dic.time_offset_s = 0.0
            self.frame_rate_edit.setText(_fmt_num(imported.fps))
            notes.append("frame rate recovered from time_s")
        else:
            notes.append("frame rate not recovered (no time_s column, or it was blank at export)")
        self._on_calibration_changed()

        result = imported.result
        self.run_status_label.setText(
            f"Imported {result.mesh.n_valid_points} point(s) across {len(result.frames)} frame(s) "
            f"from {os.path.basename(path)} ({'; '.join(notes)}). Subset size, ROI/search settings "
            "and the raw-image uncertainty ingredients aren't in the export, so they're left as-is "
            "and no measurement-uncertainty envelope is available for this imported result."
        )

    def _on_export_video(self):
        if self._dic_result is None:
            self.video_export_status_label.setText("Run a correlation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export result video", "dic_result.mp4", "MP4 video (*.mp4)")
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"

        default_fps = self._fps() or 10.0
        fps, ok = QInputDialog.getDouble(
            self, "Video playback rate",
            "Frames per second for the exported video -- a playback rate for viewing, "
            "independent of the camera's own capture rate (and of the frame step):",
            default_fps, 0.1, 1000.0, 3,
        )
        if not ok:
            return

        cam_fps = self._effective_fps()
        timestamp_unit = self.project.dic.timestamp_unit
        if cam_fps:
            # The timestamp unit only means anything once there's a
            # camera frame rate to compute a time from -- skip asking
            # otherwise (the burned-in text stays just "frame N").
            current_index = TIMESTAMP_UNITS.index(timestamp_unit) if timestamp_unit in TIMESTAMP_UNITS else 1
            timestamp_unit, ok2 = QInputDialog.getItem(
                self, "Timestamp unit",
                "Unit for the per-frame timestamp burned into the video:",
                TIMESTAMP_UNITS, current_index, False,
            )
            if not ok2:
                return
            self.project.dic.timestamp_unit = timestamp_unit

        # The field/values are cheap (pure numpy, no frame decoding) so
        # computed here on the main thread -- only the background-frame
        # reads and the matplotlib rendering (potentially slow for a long
        # sequence) run in the worker below. A single vmin/vmax spanning
        # every frame keeps the color scale consistent across the video,
        # unlike the live Results view's per-frame auto-scaling.
        field_key = self.field_combo.currentData()
        ref_index = self._dic_result.frame_indices[0]
        frame_indices = []
        values_list = []
        x_list = []
        y_list = []
        timestamps_list = []
        all_finite = []
        mean_series = []
        field_label = ""
        for idx, frame_result in enumerate(self._dic_result.frames):
            values, field_label = self._field_values(idx, frame_result, field_key)
            frame_indices.append(frame_result.frame_index)
            values_list.append(values)
            # frame_result.x/.y are each point's ACTUAL tracked position
            # (may be NaN for a lost/excluded point) -- render_field drops
            # those rather than plotting them at a placeholder, so no
            # filling is needed here (see core/dic_render.py).
            x_list.append(frame_result.x)
            y_list.append(frame_result.y)
            timestamps_list.append(
                _format_timestamp(frame_result.frame_index, ref_index, cam_fps, timestamp_unit,
                                   offset_s=self.project.dic.time_offset_s or 0.0)
            )
            finite = values[np.isfinite(values)]
            if finite.size:
                all_finite.append(finite)
            mean_series.append(float(np.nanmean(values)) if finite.size else float("nan"))

        # A Fixed Results color scale (see _current_vmin_vmax) is reused
        # here as-is, so a video's colors match what was reviewed live;
        # Auto mode falls back to this function's own long-standing
        # behavior -- one vmin/vmax spanning every frame of THIS field,
        # so the video stays internally consistent even though the live
        # view re-fits each frame on its own.
        vmin_fixed, vmax_fixed = self._current_vmin_vmax()
        if vmin_fixed is not None and vmax_fixed is not None:
            vmin, vmax = vmin_fixed, vmax_fixed
        elif all_finite:
            finite_concat = np.concatenate(all_finite)
            vmin, vmax = float(np.min(finite_concat)), float(np.max(finite_concat))
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5
        else:
            vmin, vmax = 0.0, 1.0

        # The video's right-hand panel (see _VideoExportWorker/
        # core/dic_video_export.py) shows this field's own ROI mean
        # building up over time -- same time axis as the "ROI Mean vs
        # Time" plots below, and a fixed y range (with a little padding)
        # so the panel doesn't rescale itself frame to frame.
        time_series, time_xlabel = self._time_axis()
        mean_finite = [v for v in mean_series if np.isfinite(v)]
        if mean_finite:
            mean_vmin, mean_vmax = min(mean_finite), max(mean_finite)
            if mean_vmin == mean_vmax:
                mean_vmin -= 0.5
                mean_vmax += 0.5
            pad = 0.05 * (mean_vmax - mean_vmin)
            mean_vmin -= pad
            mean_vmax += pad
        else:
            mean_vmin, mean_vmax = 0.0, 1.0

        crop = self.crop_picker.get_rect()
        downscale = self.downscale_spin.value()
        source = self._source

        def get_background(frame_index):
            return preprocess_frame(source.get_frame(frame_index), crop, downscale)

        self._set_run_controls_enabled(False)
        self.export_results_button.setEnabled(False)
        self.export_video_button.setEnabled(False)
        self.video_progress_bar.setVisible(True)
        self.video_progress_bar.setMaximum(len(frame_indices))
        self.video_progress_bar.setValue(0)
        self.video_export_status_label.setText(f"Rendering {len(frame_indices)} frame(s)…")

        self._video_thread = QThread()
        self._video_worker = _VideoExportWorker(
            path, get_background, frame_indices, x_list, y_list, values_list,
            timestamps_list, field_label, vmin, vmax, fps,
            mean_series=mean_series, time_series=time_series, time_label=time_xlabel,
            mean_vmin=mean_vmin, mean_vmax=mean_vmax,
            max_edge_px=default_max_edge_px(self._dic_result.mesh.step),
        )
        self._video_worker.moveToThread(self._video_thread)
        self._video_thread.started.connect(self._video_worker.run)
        self._video_worker.progress.connect(self._on_video_export_progress)
        self._video_worker.finished.connect(self._on_video_export_finished)
        self._video_worker.failed.connect(self._on_video_export_failed)
        self._video_worker.finished.connect(self._video_thread.quit)
        self._video_worker.failed.connect(self._video_thread.quit)
        self._video_thread.start()

    def _on_video_export_progress(self, i, total):
        self.video_progress_bar.setMaximum(total)
        self.video_progress_bar.setValue(i)

    def _on_video_export_finished(self, path):
        self._set_run_controls_enabled(True)
        self.export_results_button.setEnabled(self._dic_result is not None)
        self.export_video_button.setEnabled(self._dic_result is not None)
        self.video_progress_bar.setVisible(False)
        self.video_export_status_label.setText(f"Exported video to {os.path.basename(path)}")

    def _on_video_export_failed(self, message):
        self._set_run_controls_enabled(True)
        self.export_results_button.setEnabled(self._dic_result is not None)
        self.export_video_button.setEnabled(self._dic_result is not None)
        self.video_progress_bar.setVisible(False)
        self.video_export_status_label.setText(f"Video export failed: {message}")

    # ==================================================================
    # Session restore
    # ==================================================================

    def _restore_from_session(self):
        """If a previous session already loaded a source/preprocessing/
        ROI/mesh into this page, reflect that in the UI immediately
        instead of showing an empty page until the user re-loads and
        re-draws everything. Run *results* are never restored (see
        DICSettings' docstring) -- only the setup."""
        dic = self.project.dic

        for spin, value in ((self.subset_spin, dic.subset_size_px), (self.step_spin, dic.step_size_px),
                             (self.downscale_spin, dic.downscale_factor), (self.frame_step_spin, dic.frame_step),
                             (self.search_radius_spin, dic.search_radius_px),
                             (self.correlation_threshold_spin, dic.correlation_threshold),
                             (self.neighbor_deviation_spin, dic.max_neighbor_deviation_px)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

        self.roi_polygon_radio.blockSignals(True)
        self.roi_oval_radio.blockSignals(True)
        self.roi_rect_radio.blockSignals(True)
        if dic.roi_shape == "polygon":
            self.roi_polygon_radio.setChecked(True)
        elif dic.roi_shape == "oval":
            self.roi_oval_radio.setChecked(True)
        else:
            self.roi_rect_radio.setChecked(True)
        self.roi_rect_radio.blockSignals(False)
        self.roi_oval_radio.blockSignals(False)
        self.roi_polygon_radio.blockSignals(False)
        self.roi_picker.set_shape_mode(dic.roi_shape)
        self.roi_fields_widget.setVisible(dic.roi_shape == "rectangle")
        self.roi_polygon_hint_label.setVisible(dic.roi_shape == "polygon")
        self.roi_oval_hint_label.setVisible(dic.roi_shape == "oval")

        self.tracking_moving_radio.blockSignals(True)
        self.tracking_fixed_radio.blockSignals(True)
        if dic.tracking_mode == "moving":
            self.tracking_moving_radio.setChecked(True)
        else:
            self.tracking_fixed_radio.setChecked(True)
        self.tracking_fixed_radio.blockSignals(False)
        self.tracking_moving_radio.blockSignals(False)

        self.subpixel_combo.blockSignals(True)
        idx = self.subpixel_combo.findData(getattr(dic, "subpixel_method", "icgn"))
        self.subpixel_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.subpixel_combo.blockSignals(False)

        self.sequence_radio.blockSignals(True)
        self.video_radio.blockSignals(True)
        if dic.source_mode == "sequence":
            self.sequence_radio.setChecked(True)
        else:
            self.video_radio.setChecked(True)
        self.sequence_radio.blockSignals(False)
        self.video_radio.blockSignals(False)
        self._on_source_mode_toggled()

        # Set before _load_source() below, whose fps auto-fill only kicks
        # in when this field is still empty.
        if dic.pixel_size_mm is not None:
            self.pixel_size_edit.setText(_fmt_num(dic.pixel_size_mm))
        if dic.frame_rate_hz is not None:
            self.frame_rate_edit.setText(_fmt_num(dic.frame_rate_hz))
        if dic.material_density_kg_m3 is not None:
            self.density_edit.setText(_fmt_num(dic.material_density_kg_m3))
        if dic.specimen_width_mm is not None:
            self.specimen_width_edit.setText(_fmt_num(dic.specimen_width_mm))
        if dic.specimen_thickness_mm is not None:
            self.specimen_thickness_edit.setText(_fmt_num(dic.specimen_thickness_mm))

        self.scale_fixed_radio.blockSignals(True)
        self.scale_auto_radio.blockSignals(True)
        if dic.color_scale_mode == "fixed":
            self.scale_fixed_radio.setChecked(True)
        else:
            self.scale_auto_radio.setChecked(True)
        self.scale_auto_radio.blockSignals(False)
        self.scale_fixed_radio.blockSignals(False)
        fixed = dic.color_scale_mode == "fixed"
        self.scale_min_spin.setEnabled(fixed)
        self.scale_max_spin.setEnabled(fixed)
        self.scale_reset_button.setEnabled(fixed)
        if dic.color_scale_min is not None:
            self.scale_min_spin.blockSignals(True)
            self.scale_min_spin.setValue(dic.color_scale_min)
            self.scale_min_spin.blockSignals(False)
        if dic.color_scale_max is not None:
            self.scale_max_spin.blockSignals(True)
            self.scale_max_spin.setValue(dic.color_scale_max)
            self.scale_max_spin.blockSignals(False)

        self.smoothing_spin.blockSignals(True)
        self.smoothing_spin.setValue(float(getattr(dic, "spatial_smoothing_sigma", 0.0) or 0.0))
        self.smoothing_spin.blockSignals(False)
        self._update_smoothing_label()

        self.force_smoothing_slider.blockSignals(True)
        self.force_smoothing_slider.setValue(dic.force_smoothing_window)
        self.force_smoothing_slider.blockSignals(False)
        self.force_smoothing_label.setText(
            "1 (no smoothing)" if dic.force_smoothing_window <= 1 else f"{dic.force_smoothing_window} frames"
        )

        strain_def_keys = [k for k, _ in DIC_STRAIN_DEFINITIONS]
        if dic.strain_definition in strain_def_keys:
            self.dic_strain_combo.blockSignals(True)
            self.dic_strain_combo.setCurrentIndex(strain_def_keys.index(dic.strain_definition))
            self.dic_strain_combo.blockSignals(False)
        self.dic_strain_true_check.blockSignals(True)
        self.dic_strain_true_check.setChecked(dic.strain_convert_to_true)
        self.dic_strain_true_check.blockSignals(False)

        self.dic_time_scale_spin.blockSignals(True)
        self.dic_time_scale_spin.setValue(dic.time_scale)
        self.dic_time_scale_spin.blockSignals(False)
        self.dic_time_offset_spin.blockSignals(True)
        self.dic_time_offset_spin.setValue(dic.time_offset_s)
        self.dic_time_offset_spin.blockSignals(False)

        self.uncertainty_show_check.blockSignals(True)
        self.uncertainty_show_check.setChecked(dic.show_uncertainty)
        self.uncertainty_show_check.blockSignals(False)
        width_idx = 1 if dic.uncertainty_sigma_k >= 1.5 else 0
        self.uncertainty_width_combo.blockSignals(True)
        self.uncertainty_width_combo.setCurrentIndex(width_idx)
        self.uncertainty_width_combo.blockSignals(False)

        path = self.project.last_paths.get("dic_source", "")
        if not (path and Path(path).exists()):
            return

        if os.path.isdir(path):
            self.seq_folder_edit.setText(path)
        else:
            self.video_row.path_edit.setText(path)
        self._load_source(path)  # resets crop/ROI; restored explicitly below

        if dic.crop_width and dic.crop_height:
            x, y = dic.crop_x or 0, dic.crop_y or 0
            self._set_spin_fields((self.crop_x_spin, self.crop_y_spin, self.crop_w_spin, self.crop_h_spin),
                                   x, y, dic.crop_width, dic.crop_height)
            self.crop_picker.set_rect(x, y, dic.crop_width, dic.crop_height)
            self._refresh_preview(reset_roi=False)

        if dic.roi_shape == "polygon":
            if dic.roi_polygon and len(dic.roi_polygon) >= 3:
                self.roi_picker.set_polygon(dic.roi_polygon)
                self._update_mesh_status()
        elif dic.roi_shape == "oval":
            if dic.roi_ellipse and len(dic.roi_ellipse) == 4:
                cx, cy, rx, ry = dic.roi_ellipse
                self.roi_picker.set_ellipse(cx, cy, rx, ry)
                self._update_mesh_status()
        elif dic.roi_width and dic.roi_height:
            x, y = dic.roi_x or 0, dic.roi_y or 0
            self._set_spin_fields((self.roi_x_spin, self.roi_y_spin, self.roi_w_spin, self.roi_h_spin),
                                   x, y, dic.roi_width, dic.roi_height)
            self.roi_picker.set_rect(x, y, dic.roi_width, dic.roi_height)
            self._update_mesh_status()

        if dic.probes:
            self.roi_picker.set_probes(dic.probes)


def _or_nan(arr, like):
    return arr if arr is not None else np.full(like.shape, np.nan)


TIMESTAMP_UNITS = ["s", "ms", "µs", "ns", "ps"]
_TIMESTAMP_UNIT_FACTORS = {"s": 1.0, "ms": 1e3, "µs": 1e6, "ns": 1e9, "ps": 1e12}


def _format_timestamp(frame_index: int, ref_index: int, fps: float | None, unit: str = "ms",
                       offset_s: float = 0.0) -> str:
    """Text burned into the top-left corner of each exported video frame
    (see core/dic_video_export.py) -- elapsed time since the reference
    frame when a camera frame rate is known (already accounting for the
    frame step, since it's computed from the actual frame index), else
    just the frame index. `fps` should already be the scale-corrected
    "effective fps" (see the Image Correlation page's `_effective_fps`)
    and `offset_s` its `Project.dic.time_offset_s`, so the burned-in
    time matches every other time axis on that page. `unit` is one of
    TIMESTAMP_UNITS -- an unrecognized value falls back to
    milliseconds."""
    if fps:
        t_s = (frame_index - ref_index) / fps + offset_s
        factor = _TIMESTAMP_UNIT_FACTORS.get(unit, 1e3)
        return f"t = {t_s * factor:.3f} {unit}   (frame {frame_index})"
    return f"frame {frame_index}"


def _fmt_num(value: float) -> str:
    return f"{value:g}"


def _fmt_opt(arr, iy, ix) -> str:
    if arr is None:
        return ""
    value = arr[iy, ix]
    return "" if not np.isfinite(value) else f"{value:.6g}"


def _fmt_invariants(fr, iy, ix) -> list:
    """[e1, e2, gmax] of one point as CSV cells (empty when exx/eyy/exy are
    missing or not finite)."""
    if fr.exx is None or fr.eyy is None or fr.exy is None:
        return ["", "", ""]
    exx, eyy, exy = float(fr.exx[iy, ix]), float(fr.eyy[iy, ix]), float(fr.exy[iy, ix])
    e1, e2 = principal_strains(exx, eyy, exy)
    return [_fmt_val(e1), _fmt_val(e2), _fmt_val(max_shear_strain(exx, eyy, exy))]


def _fmt_val(value) -> str:
    """Formats a scalar (typically an ROI mean, `np.nanmean` of a field
    that may be entirely NaN) for a CSV cell -- empty string when it's
    not a finite number, matching `_fmt_opt`'s convention above."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(v) else f"{v:.6g}"
