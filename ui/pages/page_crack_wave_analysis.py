"""
Crack & Wave Analysis page (Standalone Tools).

Two sub-tabs:

- "Wave speed analysis": load a previously exported DIC results CSV --
  pushed to a very fine mesh resolution, since the whole point of this
  tool is resolving elastic-wave fronts and crack-tip strain localization
  that only show up that finely -- plus the matching image sequence/video,
  enter the material properties X-FEM-DIC needs, and trace N crack
  skeletons on top of a chosen strain-field frame (manually, or as a
  starting point from "Auto-detect path(s)"). Then, elastic wave speed in
  X and/or Y, by time-of-flight wavefront tracking across the loaded field
  time series (see core/elastic_wave_speed.py for the method and its
  assumptions).

- "Crack analysis": starts with its own "Crack Tip Detection (opening
  jump, ground truth)" card as this tab's own preparation step, tracking
  the true crack tip via an opening-jump criterion (see
  `_build_tip_detection_card`). Then X-FEM-DIC itself: directly
  re-correlates the raw images against a displacement model with a
  built-in crack discontinuity and near-tip singular term, fitting
  K_I/K_II as first-class model parameters rather than a separate post-fit
  (see core/xfem_dic.py for the method and its assumptions). X-FEM-DIC is
  one of this page's two K_I/K_II routes. The other is the "Direct K_I /
  K_II" card (core/direct_k.py): K straight from the measured subset-DIC
  displacement field, no image re-correlation -- a closed-form Williams
  ring fit (optionally with a T-stress term, core/crack_k1c.py) or a
  domain-form interaction integral (core/crack_interaction_integral.py),
  centred on the tracked tip positions, optionally SMOOTHED first with the
  controls under the "Arc-length position vs time" plot of the Crack Tip
  Detection card. Both routes share the Results card, so their curves can
  be compared on one plot.

Self-sufficient like every other Standalone Tools page (Free-Decay
Damping, Viscoelastic Model Calibration, ...): core/project.py's
`DICSettings` docstring is explicit that a DIC run's computed fields are
never persisted or shared outside the Image Correlation page itself, so
this page loads its own copy from that page's CSV export rather than
depending on it being open/already run (see core/crack_wave_io.py).
"""
from __future__ import annotations

import os
import threading

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtGui import QIcon, QPixmap, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QRadioButton, QButtonGroup, QComboBox, QSlider, QSpinBox, QFileDialog,
    QDoubleSpinBox, QCheckBox, QListWidget, QListWidgetItem, QTabWidget, QProgressBar, QMessageBox,
    QApplication, QPlainTextEdit,
)

from ui.checkable_combo import CheckableComboBox
from ui.frame_popup import make_popout_button
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Rectangle, Circle

from core.crack_wave_io import load_dic_export_csv, CrackWaveLoadError, FIELD_CHOICES
from core.elastic_wave_speed import (
    build_arrival_time_profile, WaveSpeedError,
    extract_ridge_in_polygon, fit_ridge_points, export_roi_waves_csv, PolygonRidgeResult,
    smoothed_wave_positions,
)
from core.crack_skeleton import DEFAULT_COLORS
from core.crack_tip_tracking import (
    branch_paths, CrackTipTrackingError,
    fit_tip_ridge_points, tip_roi_trace_to_track_result,
    export_tip_roi_csv, export_tip_roi_xlsx, group_traces_by_crack, PolygonTipRidgeResult, wave_roi_trace_to_track_result,
    smoothed_tip_arc_length_and_positions,
    track_tip_opening_jump, build_manual_tip_pick_context,
    crack_opening_jump_profile, trace_velocity_px_per_s, trace_velocity_uncertainty_px_per_s,
    path_polyline_up_to_arc_length, export_opening_jump_profile_csv, JUMP_CRITERIA,
)
from core.crack_local_frame import local_frame_displacement, local_jump_fields, normalise_per_crack
from core.tip_video_export import export_tip_video
from core.tip_speed import local_tip_speed, rayleigh_speed, spearman
from core.direct_k import (
    SMOOTHING_METHODS, smooth_tip_trace, smoothing_summary, DIRECT_METHODS, method_label,
    compute_direct_k_series, DirectKError, scan_k_sensitivity, format_scan_table,
    scan_k_window, format_window_table,
)
from core.xfem_video_export import export_xfem_video
from core.crack_k1c import (
    kolosov_kappa, shear_modulus,
    positive_value_mean_and_domain_bounds,
)
from core.xfem_dic import (
    compute_xfem_dic_time_series, XfemDicError,
    build_tip_centered_mesh, build_graded_tip_centered_mesh, classify_mesh_enrichment,
    recommend_mesh_parameters, recommend_tip_search_radius_px, evaluate_xfem_dic_field_grid,
    export_xfem_k_csv,
)
from core.parallel import OperationCancelled
from core.dic_render import isocontour_levels, isocontour_colorbar
from core.frame_source import open_frame_source, to_display_uint8
from ui.widgets import Card, FileSelectRow, LabeledField, make_scroll_page
from ui.plot_canvas import PlotCanvas
from ui.crack_skeleton_picker import CrackSkeletonPicker
from ui import theme

# (key, label) for the "Field isocontours" card's field selector -- unlike
# core.crack_wave_io.FIELD_CHOICES (strains only, since that tool's own
# tracing/thresholding has no use for raw displacement), this list also
# offers u/v: for a visual X-FEM-DIC-vs-subset-DIC sanity check, the
# displacement fields are the most direct like-for-like comparison (both
# methods fit them straight from the images; the standard DIC's own exx/
# eyy/exy/eps_eq are already a derivative of ITS u/v, exactly as X-FEM-
# DIC's are of its own -- see core.xfem_dic.evaluate_xfem_dic_field_grid).
XFEM_ISOCONTOUR_FIELD_CHOICES = [
    ("u", "u -- horizontal displacement (px)"),
    ("v", "v -- vertical displacement (px)"),
    ("umag", "|u| -- displacement magnitude sqrt(u\u00b2+v\u00b2) (px)"),
    ("eps_eq", "Equivalent strain (von Mises)"),
    ("exx", "exx -- horizontal strain"),
    ("eyy", "eyy -- vertical strain"),
    ("exy", "exy -- shear strain"),
    ("e1", "e1 -- max principal strain"),
    ("e2", "e2 -- min principal strain"),
    ("gmax", "gmax -- max shear strain (e1 - e2)/2"),
]


# Displacement components in the local frame of the NEAREST crack (core.crack_local_frame): only offered by
# the "Verify visually" card, which knows the traced paths.
LOCAL_FRAME_FIELDS = {
    "dn_local": ("\u0394u_n -- local OPENING: u_n(+side) \u2212 u_n(\u2212side) at the same place along the crack (px)", "dn"),
    "dt_local": ("\u0394u_t -- local SLIDING: u_t(+side) \u2212 u_t(\u2212side) (px)", "dt"),
    "un_local": ("u_n -- displacement normal to the nearest crack (local frame, px)", "un"),
    "ut_local": ("u_t -- displacement along the nearest crack (local frame, px)", "ut"),
}
# "Normalise" choices of the jump fields: (key, text)
LOCAL_NORMALISATIONS = [
    ("px", "absolute (px)"),
    ("thr", "\u00f7 detection threshold (1 = open)"),
    ("max", "\u00f7 each crack's own maximum (0..1)"),
]


def _local_field_array(kind, u, v, xs, ys, paths, window_px, band, norm=("px", None)):
    """The array shown for a local-frame field (pure numpy, also used by the video worker thread).
    kind: "un"/"ut" (displacement in the nearest crack's frame), "dn"/"dt" (jump across the crack:
    opening / sliding, see core.crack_local_frame.local_jump_fields); `band` = (inner, outer) px or
    (None, None); `norm` = (mode, threshold_px) -- only applied to the jump fields."""
    inner, outer = band
    if kind in ("un", "ut"):
        un, ut = local_frame_displacement(u, v, xs, ys, paths, window_px, inner, outer)
        return un if kind == "un" else ut
    opening, sliding, owner = local_jump_fields(u, v, xs, ys, paths, window_px, inner, outer)
    values = opening if kind == "dn" else sliding
    mode, threshold = norm
    if mode == "thr":
        return normalise_per_crack(values, owner, mode_scale=threshold)
    if mode == "max":
        return normalise_per_crack(values, owner, per_crack_max=True)
    return values


def _jump_signal_label(profile) -> str:
    """Legend / axis text of the signal the opening-jump threshold walk runs on."""
    base = {"normal": "opening jump (normal to the crack)",
            "tangential": "sliding jump (tangential)"}.get(getattr(profile, "criterion", "magnitude"),
                                                         "|jump| (u,v combined)")
    return base + (" [face discontinuity, wave rejected]" if getattr(profile, "continuous_part_removed", False) else "")


class _XfemDicWorker(QObject):
    """Runs `compute_xfem_dic_time_series` off the UI thread, once per
    ticked trace in sequence -- an Integrated-DIC/X-FEM Gauss-Newton
    solve per frame, directly against raw image grey levels (see
    core/xfem_dic.py), is heavy enough per frame that it would otherwise
    freeze the UI for the whole run. Mirrors
    ui/pages/page_image_correlation.py's `_DICWorker` pattern.

    `runs` is a list of {"label": str, "kwargs": dict} -- one per ticked
    trace, `kwargs` the keyword arguments for `compute_xfem_dic_time_series`
    (everything except `progress_cb`/`cancel_check`). `progress` reports
    (frames done, total frames) summed across every run, so a multi-trace
    run still shows one smooth progress bar rather than resetting per
    trace.

    `cancel_event`, when given, is threaded into every
    `compute_xfem_dic_time_series` call's own `cancel_check` -- the
    page's "Stop" button sets it, and `OperationCancelled` (raised inside
    whichever run is currently in progress -- see
    core.parallel.check_cancelled) is caught HERE, separately from
    (XfemDicError, CrackTipTrackingError)'s own per-run try/except below,
    so it aborts the WHOLE multi-trace batch rather than being recorded
    as just that one trace's error and moving on to the next."""
    progress = Signal(int, int)
    finished = Signal(list, list)  # (results: list[{"label","series"}], errors: list[str])
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, runs: list, cancel_event: threading.Event | None = None):
        super().__init__()
        self.runs = runs
        self.cancel_event = cancel_event

    def run(self):
        try:
            totals = [len(r["kwargs"]["tip_result"].times) for r in self.runs]
            grand_total = max(1, sum(totals))
            done_before = 0
            results = []
            errors = []
            cancel_check = self.cancel_event.is_set if self.cancel_event is not None else None
            for run_spec, total in zip(self.runs, totals):
                def _cb(i, _t, _base=done_before, _total=grand_total):
                    self.progress.emit(_base + i, _total)
                try:
                    series = compute_xfem_dic_time_series(
                        progress_cb=_cb, cancel_check=cancel_check, **run_spec["kwargs"]
                    )
                    results.append({"label": run_spec["label"], "series": series})
                except (XfemDicError, CrackTipTrackingError) as exc:
                    errors.append(f"\"{run_spec['label']}\": {exc}")
                done_before += total
            self.finished.emit(results, errors)
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - report inline, never a modal error popup
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _TipVideoExportWorker(QObject):
    """Runs `core.tip_video_export.export_tip_video` off the UI thread --
    rendering one full isocontour-plus-live-plot figure per frame and
    encoding it is far too slow to do on the UI thread without freezing
    it for the whole export, same reasoning as `_XfemDicWorker` above.

    Everything this needs is passed in as plain data at construction time
    (never a Qt widget, never `self` of the page) so `run()` -- executing
    on the worker thread -- never touches Qt: `core.tip_video_export`'s
    own figures are plain non-Qt Agg figures for exactly this reason."""
    progress = Signal(int, int)
    finished = Signal(str, int)  # (output path, n_frames written)
    failed = Signal(str)

    def __init__(self, path: str, dataset, field_key: str, field_label: str,
                 frame_indices: np.ndarray, times: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                 path_positions: np.ndarray, path_arc_length: np.ndarray, trace, vmin: float, vmax: float,
                 time_range: tuple, arc_length_max: float, velocity_range: tuple, fps: float,
                 pixel_size_mm: float | None = None, uncertainty_sigma_n: int = 0,
                 extra_traces: list | None = None, field_fn=None):
        super().__init__()
        # field_fn(frame) -> 2D array: replaces `frame.field(field_key)` (used for the local-frame
        # u_n / u_t fields, which are not stored on the frames); pure numpy, safe on the worker thread.
        self.field_fn = field_fn
        # [(trace, path_positions, path_arc_length), ...] -- more cracks to show in the
        # same video (in addition to `trace`); None/empty = the classic single-trace video.
        self.extra_traces = list(extra_traces or [])
        self.path = path
        self.dataset = dataset
        self.field_key = field_key
        self.field_label = field_label
        self.frame_indices = frame_indices
        self.times = times
        self.xs = xs
        self.ys = ys
        self.path_positions = path_positions
        self.path_arc_length = path_arc_length
        self.trace = trace
        self.vmin = vmin
        self.vmax = vmax
        self.time_range = time_range
        self.arc_length_max = arc_length_max
        self.velocity_range = velocity_range
        self.fps = fps
        # px/px-per-s throughout by default; when set, the bottom panel's
        # position/speed (ONLY -- never the top panel's spatial x/y, a
        # physical position on the sensor grid, unrelated to this) is
        # converted to mm/mm-per-s, same conversion "Export all to CSV..."
        # already applies to its own arc_length_mm/velocity_mm_per_s
        # columns when a pixel size is set on this page's "Material" card.
        self.pixel_size_mm = pixel_size_mm
        # 0 ("Off") or 1/2/3 -- the "Uncertainty envelope" dropdown's
        # current choice at the moment "Export video" was clicked (see
        # `_current_uncertainty_sigma_n`). Only has any effect when
        # `trace.position_uncertainty_px` is also present (i.e. `trace`
        # was added with "Estimate measurement uncertainty" checked) --
        # `run()` checks both before drawing anything, exactly like the
        # "Verify visually" viewer's own bands do.
        self.uncertainty_sigma_n = uncertainty_sigma_n

    def _run_multi(self):
        """Several traces (one per crack) in one video: see
        core.tip_video_export.render_multi_tip_frame."""
        entries = [(self.trace, self.path_positions, self.path_arc_length)] + self.extra_traces
        n = len(self.frame_indices)
        k = self.pixel_size_mm or 1.0
        sigma_n = self.uncertainty_sigma_n
        prepared = []
        for trace, ppos, parc in entries:
            vel = trace_velocity_px_per_s(trace.times, trace.arc_length)
            unc = getattr(trace, "position_uncertainty_px", None)
            has_band = sigma_n > 0 and unc is not None and np.all(np.isfinite(unc))
            vel_unc = trace_velocity_uncertainty_px_per_s(trace.times, unc) if has_band else None
            prepared.append(dict(
                label=trace.label, color=trace.color, path_positions=ppos, path_arc_length=parc,
                times=trace.times, arc_length=trace.arc_length * k, velocity=vel * k,
                positions_xy=trace.positions_xy, source=trace.source_frame_positions,
                position_uncertainty=(unc * k if has_band else None),
                velocity_uncertainty=(vel_unc * k if has_band else None),
            ))
        arc_max = max((float(np.nanmax(t["arc_length"])) for t in prepared if t["arc_length"].size), default=0.0)
        vmins, vmaxs = [], []
        for t in prepared:
            fv = t["velocity"][np.isfinite(t["velocity"])]
            if fv.size:
                vmins.append(float(fv.min())); vmaxs.append(float(fv.max()))
        velocity_range = (min(vmins), max(vmaxs)) if vmins else (0.0, 1.0)
        position_label, velocity_label = ("position (mm)", "speed (mm/s)") if self.pixel_size_mm \
            else ("position (px)", "speed (px/s)")

        def _frames():
            for i, (oi, t) in enumerate(zip(self.frame_indices, self.times)):
                frame = self.dataset.frames[int(oi)]
                field_arr = np.asarray(self.field_fn(frame) if self.field_fn else frame.field(self.field_key), dtype=float)
                traces = []
                for tr in prepared:
                    match = np.flatnonzero(tr["source"] == oi)
                    d = {key: val for key, val in tr.items() if key != "source"}
                    d["current_xy"] = tuple(tr["positions_xy"][match[0]]) if match.size else None
                    traces.append(d)
                yield dict(
                    xs=self.xs, ys=self.ys, field_arr=field_arr, field_label=self.field_label,
                    vmin=self.vmin, vmax=self.vmax, traces=traces, current_time=float(t),
                    time_range=self.time_range, arc_length_max=arc_max, velocity_range=velocity_range,
                    position_label=position_label, velocity_label=velocity_label,
                    timestamp_text=f"Frame {i + 1}/{n} (dataset frame {frame.frame_index}, t = {t:.6g} s)",
                    uncertainty_sigma_n=sigma_n,
                )

        export_tip_video(self.path, _frames(), n_frames=n, fps=self.fps,
                          progress_cb=lambda i, total: self.progress.emit(i, total))
        self.finished.emit(self.path, n)

    def run(self):
        try:
            if self.extra_traces:
                self._run_multi()
                return
            trace_velocity = trace_velocity_px_per_s(self.trace.times, self.trace.arc_length)
            n = len(self.frame_indices)

            trace_position_uncertainty = getattr(self.trace, "position_uncertainty_px", None)
            has_uncertainty_band = (
                self.uncertainty_sigma_n > 0
                and trace_position_uncertainty is not None
                and np.all(np.isfinite(trace_position_uncertainty))
            )
            trace_velocity_uncertainty = (
                trace_velocity_uncertainty_px_per_s(self.trace.times, trace_position_uncertainty)
                if has_uncertainty_band else None
            )

            if self.pixel_size_mm:
                k = self.pixel_size_mm
                trace_arc_length = self.trace.arc_length * k
                trace_velocity = trace_velocity * k
                arc_length_max = self.arc_length_max * k
                velocity_range = (self.velocity_range[0] * k, self.velocity_range[1] * k)
                position_label, velocity_label = "position (mm)", "speed (mm/s)"
                if has_uncertainty_band:
                    trace_position_uncertainty = trace_position_uncertainty * k
                    trace_velocity_uncertainty = trace_velocity_uncertainty * k
            else:
                trace_arc_length = self.trace.arc_length
                arc_length_max = self.arc_length_max
                velocity_range = self.velocity_range
                position_label, velocity_label = "position (px)", "speed (px/s)"

            # Widen the fixed axis bounds by the band's own worst-case
            # extent so a wide uncertainty band doesn't get clipped at the
            # top/edge of the plot for the whole video -- same reasoning
            # as `arc_length_max`/`velocity_range` themselves being
            # computed once, up front, over the whole trace.
            if has_uncertainty_band:
                finite_pos_unc = trace_position_uncertainty[np.isfinite(trace_position_uncertainty)]
                if finite_pos_unc.size:
                    arc_length_max = arc_length_max + self.uncertainty_sigma_n * float(np.max(finite_pos_unc))
                finite_vel_unc = trace_velocity_uncertainty[np.isfinite(trace_velocity_uncertainty)]
                if finite_vel_unc.size:
                    margin = self.uncertainty_sigma_n * float(np.max(finite_vel_unc))
                    velocity_range = (velocity_range[0] - margin, velocity_range[1] + margin)

            def _frames():
                for i, (oi, t) in enumerate(zip(self.frame_indices, self.times)):
                    frame = self.dataset.frames[int(oi)]
                    field_arr = np.asarray(self.field_fn(frame) if self.field_fn else frame.field(self.field_key), dtype=float)
                    match = np.flatnonzero(self.trace.source_frame_positions == oi)
                    current_xy = tuple(self.trace.positions_xy[match[0]]) if match.size else None
                    yield dict(
                        xs=self.xs, ys=self.ys, field_arr=field_arr, field_label=self.field_label,
                        vmin=self.vmin, vmax=self.vmax, path_positions=self.path_positions,
                        path_arc_length=self.path_arc_length,
                        trace_times=self.trace.times, trace_arc_length=trace_arc_length,
                        trace_velocity=trace_velocity, trace_positions_xy=self.trace.positions_xy,
                        current_time=float(t), current_xy=current_xy,
                        time_range=self.time_range, arc_length_max=arc_length_max,
                        velocity_range=velocity_range,
                        position_label=position_label, velocity_label=velocity_label,
                        timestamp_text=f"Frame {i + 1}/{n} (dataset frame {frame.frame_index}, t = {t:.6g} s)",
                        trace_position_uncertainty=trace_position_uncertainty if has_uncertainty_band else None,
                        trace_velocity_uncertainty=trace_velocity_uncertainty if has_uncertainty_band else None,
                        uncertainty_sigma_n=self.uncertainty_sigma_n if has_uncertainty_band else 0,
                    )

            export_tip_video(self.path, _frames(), n_frames=n, fps=self.fps,
                              progress_cb=lambda i, total: self.progress.emit(i, total))
            self.finished.emit(self.path, n)
        except Exception as exc:  # noqa: BLE001 - report inline, never a modal error popup
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _XfemVideoExportWorker(QObject):
    """Runs `core.xfem_video_export.export_xfem_video` off the UI thread --
    same reasoning as `_TipVideoExportWorker`/`_XfemDicWorker` above: one
    full two-isocontour-panel-plus-plot figure per frame, PLUS (for a
    frame whose color scale isn't already fixed) an `evaluate_xfem_dic_
    field_grid` call per frame to pool the color scale first, is far too
    slow to do on the UI thread without freezing it for the whole export.

    Everything this needs is passed in as plain data at construction time
    (never a Qt widget, never `self` of the page, never the mesh/Gauss-
    Newton solve itself -- that already ran and is cached in `series.
    samples[i].fit`; this worker only RE-EVALUATES each already-fitted
    frame's field on a grid, exactly what "Show isocontours" does for one
    frame already) so `run()` -- executing on the worker thread -- never
    touches Qt."""
    progress = Signal(int, int)
    finished = Signal(str, int)  # (output path, n_frames written)
    failed = Signal(str)

    def __init__(self, path: str, dataset, xs: np.ndarray, ys: np.ndarray,
                 field_key: str, field_label: str, series, tip_result,
                 kappa: float, mu_pa: float, sample_step_px: float, show_K2: bool,
                 vmin: float | None, vmax: float | None, fps: float,
                 pixel_size_mm_display: float | None = None):
        super().__init__()
        self.path = path
        self.dataset = dataset
        self.xs = xs
        self.ys = ys
        self.field_key = field_key
        self.field_label = field_label
        self.series = series
        self.tip_result = tip_result
        self.kappa = kappa
        self.mu_pa = mu_pa
        self.sample_step_px = sample_step_px
        self.show_K2 = show_K2
        # None -- pool a robust (2nd/98th percentile) range across every
        # frame's own field values before rendering (see run()), same
        # reasoning as `_on_tip_video_export_clicked`'s own Auto-scale
        # pooling. Not None -- the "Fixed" scale bounds from the "Color
        # scale (both panels)" control, used as-is.
        self.vmin = vmin
        self.vmax = vmax
        self.fps = fps
        # px/px-per-s throughout by default; when set, the bottom panel's
        # distance/speed (ONLY -- never the top panels' spatial x/y) is
        # converted to mm/mm-per-s, same "Pixel size" field/convention
        # `_y_axis_unit_and_scale` and `_TipVideoExportWorker` already use.
        self.pixel_size_mm_display = pixel_size_mm_display

    def run(self):
        try:
            series = self.series
            tip_result = self.tip_result
            samples = series.samples
            n = len(samples)
            path_positions, path_arc_length = tip_result.path_positions, tip_result.path_arc_length

            # -- the "u"/"v" fields are raw-pixel DISPLACEMENTS (both the
            # standard-DIC CSV columns and X-FEM-DIC's own fitted field --
            # see evaluate_xfem_dic_displacement_field's docstring), so
            # they convert to mm the same way -- and by the same "Pixel
            # size" value -- as the bottom panel's distance/speed just
            # above. The strain fields (exx/eyy/exy/eps_eq) are
            # dimensionless (du/dx over a pixel-spaced grid), so they are
            # NEVER converted regardless of field_key -- multiplying a
            # strain by a length would just be wrong.
            convert_field = self.field_key in ("u", "v", "umag") and bool(self.pixel_size_mm_display)
            field_label = self.field_label.replace("(px)", "(mm)") if convert_field else self.field_label

            def _to_display_units(arr):
                if arr is None or not convert_field:
                    return arr
                return arr * self.pixel_size_mm_display

            # -- the whole trace's own per-frame time/distance/K1/K2/speed
            # arrays -- SAME source data _draw_xfem_plot/_draw_xfem_arc_
            # length_plot read, so the video's bottom panel always agrees
            # with those live plots.
            times = np.array([s.time_s for s in samples], dtype=float)
            distances_px = np.array(
                [s.tip_arc_length_s if s.tip_arc_length_s is not None else np.nan for s in samples],
                dtype=float,
            )
            k1_values = np.array(
                [s.fit.K_I_pa_sqrt_m / 1.0e6 if s.fit.K_I_pa_sqrt_m is not None else np.nan
                 for s in samples],
                dtype=float,
            )
            k2_raw = np.array(
                [s.fit.K_II_pa_sqrt_m / 1.0e6 if s.fit.K_II_pa_sqrt_m is not None else np.nan
                 for s in samples],
                dtype=float,
            )
            k2_values = k2_raw if self.show_K2 else None

            if self.pixel_size_mm_display:
                k = self.pixel_size_mm_display
                distances = distances_px * k
                distance_label = "distance along path (mm)"
            else:
                distances = distances_px
                distance_label = "distance along path (px)"
            velocity = trace_velocity_px_per_s(times, distances_px)
            if self.pixel_size_mm_display:
                velocity = velocity * self.pixel_size_mm_display
                velocity_label = "speed (mm/s)"
            else:
                velocity_label = "speed (px/s)"

            finite_distances = distances[np.isfinite(distances)]
            distance_max = float(np.max(finite_distances)) if finite_distances.size else 0.0
            k_pool = [k1_values[np.isfinite(k1_values)]]
            if self.show_K2:
                k_pool.append(k2_raw[np.isfinite(k2_raw)])
            k_pool = np.concatenate(k_pool) if k_pool else np.array([])
            k_range = (float(np.min(k_pool)), float(np.max(k_pool))) if k_pool.size else (0.0, 1.0)
            finite_velocity = velocity[np.isfinite(velocity)]
            velocity_range = ((float(np.min(finite_velocity)), float(np.max(finite_velocity)))
                               if finite_velocity.size else (0.0, 1.0))

            # -- re-evaluate every frame's already-fitted field on a grid
            # once (grid.x/y can differ frame to frame -- the mesh is
            # re-centered on each frame's own tip position) -- cached so
            # the render pass below never re-evaluates the same frame
            # twice, whether or not the color scale needed pooling first.
            grids: list = [None] * n
            dic_fields: list = [None] * n
            pooled = []
            for i, sample in enumerate(samples):
                if 0 <= sample.source_frame_index < len(self.dataset.frames):
                    dic_fields[i] = _to_display_units(np.asarray(
                        self.dataset.frames[sample.source_frame_index].field(self.field_key), dtype=float
                    ))
                fit = sample.fit
                if fit.error is None and fit.K_I_pa_sqrt_m is not None and sample.tangent_xy is not None:
                    grid = evaluate_xfem_dic_field_grid(
                        fit, path_positions, path_arc_length, sample.tip_xy_px, sample.tangent_xy,
                        self.kappa, self.mu_pa, series.pixel_size_mm, series.r_enrich_px,
                        step_px=self.sample_step_px,
                        dic_xs=(self.xs if dic_fields[i] is not None else None),
                        dic_ys=(self.ys if dic_fields[i] is not None else None),
                        dic_field=dic_fields[i],
                        r_min_px=series.r_min_px, r_max_px=series.r_max_px,
                    )
                    grids[i] = grid
                    arr = _to_display_units(grid.field(self.field_key))
                    finite = arr[np.isfinite(arr)]
                    if finite.size:
                        pooled.append(finite)
                if dic_fields[i] is not None:
                    finite = dic_fields[i][np.isfinite(dic_fields[i])]
                    if finite.size:
                        pooled.append(finite)

            if self.vmin is not None and self.vmax is not None:
                vmin, vmax = float(self.vmin), float(self.vmax)
                # The "Color scale (both panels)" Fixed-mode spinboxes are
                # always populated/typed against the (still px) live
                # preview, so a Fixed range needs the same conversion the
                # per-frame field arrays above just got -- otherwise the
                # contour levels/colorbar would silently stay in px while
                # the data plotted against them is now in mm.
                if convert_field:
                    vmin, vmax = vmin * self.pixel_size_mm_display, vmax * self.pixel_size_mm_display
            elif pooled:
                vmin, vmax = np.nanpercentile(np.concatenate(pooled), [2.0, 98.0])
            else:
                vmin, vmax = -1.0, 1.0
            vmin, vmax = float(vmin), float(vmax)
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            if vmin == vmax:
                vmin, vmax = vmin - 1.0, vmax + 1.0

            def _frames():
                last_valid_s_px = None
                for i, sample in enumerate(samples):
                    s_px = sample.tip_arc_length_s
                    if s_px is not None and np.isfinite(s_px):
                        last_valid_s_px = s_px
                    grid = grids[i]
                    yield dict(
                        xs=self.xs, ys=self.ys, dic_field_arr=dic_fields[i], field_label=field_label,
                        xfem_grid_x=(grid.x if grid is not None else None),
                        xfem_grid_y=(grid.y if grid is not None else None),
                        xfem_field_arr=(_to_display_units(grid.field(self.field_key)) if grid is not None else None),
                        vmin=vmin, vmax=vmax, path_positions=path_positions, path_arc_length=path_arc_length,
                        current_tip_xy=tuple(sample.tip_xy_px), trajectory_max_s=last_valid_s_px,
                        distances=distances, k1_values=k1_values, k2_values=k2_values,
                        velocity_values=velocity, current_index=i,
                        distance_max=distance_max, k_range=k_range, velocity_range=velocity_range,
                        distance_label=distance_label, k_label="K (MPa·√m)",
                        velocity_label=velocity_label,
                        timestamp_text=(
                            f"Frame {i + 1}/{n} (dataset frame {sample.source_frame_index}, "
                            f"t = {sample.time_s:.6g} s)"
                        ),
                    )

            export_xfem_video(self.path, _frames(), n_frames=n, fps=self.fps,
                               progress_cb=lambda i, total: self.progress.emit(i, total))
            self.finished.emit(self.path, n)
        except Exception as exc:  # noqa: BLE001 - report inline, never a modal error popup
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class CrackWaveAnalysisPage(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._dataset = None       # core.crack_wave_io.CrackWaveDataset, or None
        self._frame_source = None  # a FrameSource (video or image sequence), or None
        self._xfem_results = []  # list[{"label", "series"}] -- one entry per ticked trace last
                                  # computed by the X-FEM-DIC tab's "Run" (see _on_xfem_run_finished);
                                  # `series` is a core.xfem_dic.XfemDicTimeSeries
        self._xfem_skipped_traces = []  # list[str] -- error messages for ticked traces that failed
        self._xfem_thread = None
        self._xfem_worker = None
        self._xfem_cancel_event = None  # threading.Event for the current X-FEM-DIC run, or None between runs
        self._xfem_isocontour_shown = False  # True once "Show isocontours" has drawn at least once --
                                              # from then on, changing the trace/frame/field above
                                              # keeps that card live-updated too (see
                                              # _maybe_refresh_xfem_isocontours)
        self._xfem_isocontour_signal_for_scale = None  # combined finite values last drawn, for
                                                        # "Reset to data range" (Color scale card)
        self._last_wave_profile = None        # core.elastic_wave_speed.ArrivalTimeProfile, or None --
                                               # the background field currently shown on the canvas
                                               # ("Show field"), redrawn as-is whenever a ROI wave is
                                               # added/removed or the color scale is toggled
        self._last_wave_direction = None      # "x" | "y" the shown field's axis label refers to
        self._last_wave_field_key = None      # field_key the shown field refers to
        self._last_wave_signal_for_scale = None  # (n_positions, n_frames) background array, for "Reset to data range"
        self._roi_waves = []          # list[core.elastic_wave_speed.PolygonRidgeResult], user-picked via ROI polygon
        self._roi_selector = None     # matplotlib.widgets.PolygonSelector, live on wave_speed_canvas while drawing
        self._roi_drawing = False     # whether "Draw ROI polygon" mode is currently active
        self._pending_roi_vertices = None  # [(time_s, position_*)] of the just-closed (not yet added) polygon

        # Crack-tip detection state -- "opening jump" traces, detected
        # automatically (see track_tip_opening_jump), fed straight into
        # the X-FEM-DIC tab's "Crack path source" checklist.
        self._tip_roi_traces = []     # list[core.crack_tip_tracking.PolygonTipRidgeResult]
        self._tip_roi_path_geometry = []  # list[(path_positions, path_arc_length)], index-aligned with
                                           # _tip_roi_traces -- needed to convert a trace into a
                                           # TipTrackResult without re-fetching the crack

        # "Opening jump" verification viewer state -- a live, read-only
        # raw-field-plus-detected-position viewer (NOT a picking tool --
        # "Opening jump" itself is fully automatic, see track_tip_opening_
        # jump) that replaced the old "manual (click)" tool's canvas/
        # slider/color-scale machinery. Lets the user visually sanity-
        # check an added "opening_jump" trace's per-frame detected
        # position against the raw u/v field it came from, the same
        # purpose manual pointing served, without the click-to-pick step.
        self._tip_verify_ctx = None    # core.crack_tip_tracking.ManualTipPickContext, or None -- the
                                        # traced path + frame walk order/timing currently shown (built
                                        # with build_manual_tip_pick_context, reused here purely for its
                                        # path/frame-order bookkeeping, no picking involved)
        self._tip_verify_pos = 0       # index into _tip_verify_ctx.frame_indices/.times -- the frame
                                        # currently displayed
        self._last_tip_verify_signal_for_scale = None  # current frame's field values, for "Reset to data range"
        self._tip_verify_current_jump_profile = None  # (CrackOpeningJumpProfile, frame_index, time_s) for
                                                        # the currently-drawn opening-jump panel, or None --
                                                        # see _draw_tip_verify_jump_panel / "Export this
                                                        # profile to CSV..."
        self._tip_video_thread = None  # QThread running _TipVideoExportWorker, or None between exports
        self._tip_video_worker = None
        self._xfem_video_thread = None  # QThread running _XfemVideoExportWorker, or None between exports
        self._xfem_video_worker = None
        self._data_card_suffixes = []  # e.g. ["", "_ca"] -- one entry per built "Data" card
                                        # instance (see `_build_data_card`); every handler that
                                        # loads a CSV/video/sequence loops over this list so ALL
                                        # instances stay in sync with the ONE shared
                                        # `self._dataset`/`self._frame_source` -- there is still
                                        # only one loaded dataset for the whole page, never two
                                        # independent ones, just more than one place to load it from.
        self._pixel_size_spin_suffixes = []  # same pattern as _data_card_suffixes, for the
                                              # "Pixel size" field -- see `_build_pixel_size_row`;
                                              # ALL instances always mirror the one value every
                                              # speed/K_I/K_II calculation on this page reads
                                              # (`self.pixel_size_spin`, the suffix "" instance,
                                              # which every pre-existing call site already uses).

        scroll, layout = make_scroll_page(
            "Crack & Wave Analysis",
            "Analyze elastic wave fronts and propagating-crack strain localization from a "
            "high-resolution DIC mesh: elastic wave speed (X/Y) on the first tab, and automatic "
            "crack-tip detection plus X-FEM-DIC K_I/K_II on the second, from strain fields and a "
            "pre-traced crack skeleton.",
        )

        # ------------------------------------------------------------ data
        layout.addWidget(self._build_data_card(""))

        # ------------------------------------------------------------ pixel size
        # This tab's own "Material & Calibration" card holds only the one field
        # it actually uses itself: pixel size, to convert wave-front speeds
        # below from px/s to mm/s or m/s. Elastic modulus, Poisson's ratio and
        # plane-stress/strain state live in the Crack analysis tab's own
        # "Material & Calibration" card instead (see _build_xfem_dic_tab),
        # since only that tab's X-FEM-DIC K_I/K_II fit needs them. Pixel size
        # is ALSO needed by that tab's own K_I/K_II fit though, so the field
        # is built once per tab (see `_build_pixel_size_row`) and always kept
        # in sync -- no need to switch tabs just to set/see it.
        pixel_size_card = Card(
            "Material & Calibration",
            "Pixel size, used to convert every speed below from px/s to mm/s or m/s (same value as "
            "the Crack analysis tab's own Material & Calibration card). Elastic modulus, Poisson's "
            "ratio and plane-stress/strain state -- needed only by that tab's X-FEM-DIC K_I/K_II "
            "fit -- are set there instead.",
        )
        pixel_size_card.add_layout(self._build_pixel_size_row(""))
        self.pixel_size_status = QLabel("")
        self.pixel_size_status.setObjectName("cardSubtitle")
        self.pixel_size_status.setWordWrap(True)
        pixel_size_card.add(self.pixel_size_status)
        layout.addWidget(pixel_size_card)

        # ------------------------------------------------------------ elastic wave speed
        wave_card = Card(
            "Elastic Wave Speed (X / Y)",
            "Show the field as a time/position heat map, then draw a polygon around each visible "
            "band to trace it -- like picking a ridge by hand on a PDV/VISAR spectrogram. No "
            "detection threshold involved, so a faint band is exactly as easy to pick as a "
            "saturated one, and the traced line is free to curve rather than being forced straight.",
            help_html=(
                "<p>Measures how fast the elastic wave(s) sweep across the field, along X or Y.</p>"
                "<ul>"
                "<li><b>Field</b>: exx for an X-direction wave, eyy for Y, or MISES for either -- "
                "whichever shows the passing wave most clearly. e1 / e2 (principal strains) and "
                "gmax (max shear strain) do not depend on the image axes: use them for a wave "
                "front or crack that is not aligned with X or Y.</li>"
                "<li><b>Baseline frames</b>: how many frames at the start of the selected range "
                "are averaged as each position's \"before the wave\" reference level.</li>"
                "<li><b>Frame range</b>: restrict the analysis to the part of the sequence that "
                "actually shows the wave(s) passing -- important if the sequence also runs into "
                "crack growth later, since a later, much larger crack-strain spike would otherwise "
                "dominate the color scale.</li>"
                "<li><b>Frame rate override</b>: only needed if the loaded CSV has no timing "
                "(the Image Correlation page's frame rate wasn't set when it was exported).</li>"
                "<li><b>Pixel size</b> (Material &amp; Calibration card): optional -- converts "
                "speeds to mm/s in addition to px/s.</li>"
                "<li><b>Color scale (background)</b>: Auto re-fits the background heatmap to "
                "the shown field's own data range; switch to Fixed and narrow Min/Max to pull a "
                "faint or reflected wave out of a scale dominated by a much stronger, closer one.</li>"
                "<li><b>Show field</b>: draws the background heat map to trace on -- click it "
                "first (or again, after changing Field/Direction/Frame range/Frame rate above) "
                "before drawing a ROI polygon.</li>"
                "<li><b>ROI polygon wave picking</b>: click \"Draw ROI polygon\", then click to "
                "place vertices directly on the plot below, tracing a loose loop around ONE visible "
                "band (undershoot rather than overshoot -- a little slack around the band is fine, "
                "but avoid sweeping in a second one -- if another band grazes the polygon it CAN "
                "win the frames where it does, so draw it tight); close it by clicking the first "
                "vertex again or pressing Enter. Every time frame the polygon spans then "
                "contributes one point, at that frame's own STRONGEST local minimum/maximum of the "
                "field inside the polygon -- no arrival threshold and no averaging over the whole "
                "enclosed span involved, so a faint band is exactly as easy to pick as a saturated "
                "one (as long as nothing stronger shares the polygon), and the traced line is free "
                "to curve rather than being forced straight. Name it, click \"Add to list\", "
                "repeat for other bands, then \"Export all to CSV…\" to hand the raw (time, "
                "position) points off for further analysis. The reported speed/R<sup>2</sup> next "
                "to each listed wave is a plain straight-line fit shown only as a rough indicator "
                "-- it is not what gets exported or trusted for a curved front.</li>"
                "</ul>"
            ),
        )
        wave_row1 = QHBoxLayout()
        wave_row1.addWidget(QLabel("Field"))
        self.wave_field_combo = QComboBox()
        for key, label in FIELD_CHOICES:
            self.wave_field_combo.addItem(label, key)
        wave_row1.addWidget(self.wave_field_combo)
        wave_row1.addWidget(QLabel("Direction"))
        self.wave_x_radio = QRadioButton("X")
        self.wave_y_radio = QRadioButton("Y")
        self.wave_x_radio.setChecked(True)
        wave_dir_group = QButtonGroup(self)
        wave_dir_group.addButton(self.wave_x_radio)
        wave_dir_group.addButton(self.wave_y_radio)
        wave_row1.addWidget(self.wave_x_radio)
        wave_row1.addWidget(self.wave_y_radio)
        wave_row1.addStretch(1)
        wave_card.add_layout(wave_row1)

        wave_row2 = QHBoxLayout()
        wave_row2.addWidget(QLabel("Baseline frames"))
        self.wave_baseline_spin = QSpinBox()
        self.wave_baseline_spin.setRange(1, 50)
        self.wave_baseline_spin.setValue(1)
        wave_row2.addWidget(self.wave_baseline_spin)
        wave_row2.addStretch(1)
        wave_card.add_layout(wave_row2)

        wave_row3 = QHBoxLayout()
        wave_row3.addWidget(QLabel("Frame range"))
        self.wave_start_spin = QSpinBox()
        self.wave_start_spin.setRange(0, 0)
        wave_row3.addWidget(self.wave_start_spin)
        wave_row3.addWidget(QLabel("to"))
        self.wave_end_spin = QSpinBox()
        self.wave_end_spin.setRange(0, 0)
        wave_row3.addWidget(self.wave_end_spin)
        wave_row3.addStretch(1)
        wave_card.add_layout(wave_row3)

        wave_row4 = QHBoxLayout()
        self.wave_rate_spin = QDoubleSpinBox()
        self.wave_rate_spin.setRange(0.0, 1.0e9)
        self.wave_rate_spin.setDecimals(1)
        self.wave_rate_spin.setValue(0.0)
        self.wave_rate_spin.setSuffix(" Hz (0 = use the CSV's own timing)")
        wave_row4.addWidget(QLabel("Frame rate override"))
        wave_row4.addWidget(self.wave_rate_spin)
        wave_row4.addStretch(1)
        wave_card.add_layout(wave_row4)

        # Background heatmap color scale (Auto/Fixed), same shape as the
        # tracing panel's own "Field color scale" -- see that card's help
        # text and ui/crack_skeleton_picker.py's "_on_scale_*"/
        # "_current_vmin_vmax" for the pattern this mirrors. Applies to the
        # background field shown by "Show field" below.
        wave_scale_row = QHBoxLayout()
        wave_scale_row.addWidget(QLabel("Color scale (background)"))
        self.wave_scale_auto_radio = QRadioButton("Auto")
        self.wave_scale_fixed_radio = QRadioButton("Fixed")
        self.wave_scale_auto_radio.setChecked(True)
        wave_scale_group = QButtonGroup(self)
        wave_scale_group.addButton(self.wave_scale_auto_radio)
        wave_scale_group.addButton(self.wave_scale_fixed_radio)
        self.wave_scale_auto_radio.toggled.connect(self._on_wave_scale_mode_toggled)
        wave_scale_row.addWidget(self.wave_scale_auto_radio)
        wave_scale_row.addWidget(self.wave_scale_fixed_radio)
        self.wave_scale_min_spin = QDoubleSpinBox()
        self.wave_scale_min_spin.setRange(-1.0e9, 1.0e9)
        self.wave_scale_min_spin.setDecimals(6)
        self.wave_scale_min_spin.setEnabled(False)
        self.wave_scale_min_spin.valueChanged.connect(self._on_wave_scale_bounds_changed)
        wave_scale_row.addWidget(self.wave_scale_min_spin)
        self.wave_scale_max_spin = QDoubleSpinBox()
        self.wave_scale_max_spin.setRange(-1.0e9, 1.0e9)
        self.wave_scale_max_spin.setDecimals(6)
        self.wave_scale_max_spin.setEnabled(False)
        self.wave_scale_max_spin.valueChanged.connect(self._on_wave_scale_bounds_changed)
        wave_scale_row.addWidget(self.wave_scale_max_spin)
        self.wave_scale_reset_button = QPushButton("Reset to data range")
        self.wave_scale_reset_button.setEnabled(False)
        self.wave_scale_reset_button.clicked.connect(self._on_wave_scale_reset)
        wave_scale_row.addWidget(self.wave_scale_reset_button)
        wave_scale_row.addStretch(1)
        wave_card.add_layout(wave_scale_row)

        # -- show field (entry point) ------------------------------------------
        # Just draws the background heat map -- no arrival-threshold dots, no
        # fit line -- so there's something to draw a ROI polygon on. ROI
        # polygon picking below is the only detection method on this tab, so
        # this button's only job is to show the field, nothing else.
        show_field_row = QHBoxLayout()
        self.show_field_button = QPushButton("Show field")
        self.show_field_button.setObjectName("primary")
        self.show_field_button.clicked.connect(self._on_show_field_clicked)
        show_field_row.addWidget(self.show_field_button)
        show_field_row.addStretch(1)
        wave_card.add_layout(show_field_row)

        self.wave_speed_status = QLabel("Load data above, then click \"Show field\".")
        self.wave_speed_status.setObjectName("cardSubtitle")
        self.wave_speed_status.setWordWrap(True)
        wave_card.add(self.wave_speed_status)

        self.wave_speed_canvas = PlotCanvas(figsize=(9, 4.5))
        wave_card.add(self.wave_speed_canvas)

        # -- ROI polygon wave picking (manual, threshold-free) ----------------
        # No single arrival-threshold fraction can separate every band at
        # once when they genuinely differ in intensity (one saturated red,
        # another only ever reaching green/yellow) -- see the card's help
        # text and core/elastic_wave_speed.py's module docstring. Drawing a
        # polygon directly on the plot below isolates one band regardless
        # of its own intensity, the same way a PDV/VISAR spectrogram's
        # ridge is picked by hand.
        roi_label = QLabel("ROI polygon wave picking (draw a loose loop around a visible band to "
                            "trace it, one polygon per wave)")
        roi_label.setObjectName("cardSubtitle")
        wave_card.add(roi_label)

        # Temporal smoothing of the picked wave position -- when checked,
        # the smoothed position REPLACES the raw picked one for a wave at
        # the moment it's added ("Add to list" below), so everything
        # downstream (its own speed/R^2, CSV export) uses the smoothed
        # path, mirroring the Crack Tip Detection card's own "Smooth tip
        # position" control one to one (see `smoothed_wave_positions` for
        # the shared rationale). Only affects waves added AFTER it's set --
        # an already-added wave keeps whatever it was given when it was
        # added.
        roi_smoothing_row = QHBoxLayout()
        self.roi_show_smoothed_check = QCheckBox("Smooth wave position (replaces the raw path when added)")
        self.roi_show_smoothed_check.setChecked(False)
        self.roi_show_smoothed_check.setToolTip(
            "Off by default -- the raw, per-frame picked position is used. When checked, a wave "
            "added while this is on has its position locally smoothed (over the window below) BEFORE "
            "being stored -- the raw per-frame pick is not kept, and every downstream use (speed/"
            "distance, CSV export) sees the smoothed path. Same local-linear-regression smoother as "
            "the Crack Tip Detection card's own \"Smooth tip position\" (tracks a real speed ramp "
            "without lagging, unlike a plain moving average). Only affects a wave at the moment it's "
            "added -- toggling this after the fact doesn't change waves already in the list."
        )
        roi_smoothing_row.addWidget(self.roi_show_smoothed_check)
        roi_smoothing_row.addWidget(QLabel("Smoothing window"))
        self.roi_smoothing_window_spin = QDoubleSpinBox()
        self.roi_smoothing_window_spin.setRange(0.0, 2000.0)
        self.roi_smoothing_window_spin.setValue(20.0)
        self.roi_smoothing_window_spin.setSuffix(" px")
        roi_smoothing_row.addWidget(self.roi_smoothing_window_spin)
        roi_smoothing_row.addStretch(1)
        wave_card.add_layout(roi_smoothing_row)

        roi_row1 = QHBoxLayout()
        self.roi_draw_button = QPushButton("Draw ROI polygon")
        self.roi_draw_button.clicked.connect(self._on_roi_draw_clicked)
        roi_row1.addWidget(self.roi_draw_button)
        self.roi_cancel_button = QPushButton("Cancel polygon")
        self.roi_cancel_button.setEnabled(False)
        self.roi_cancel_button.clicked.connect(self._on_roi_cancel_clicked)
        roi_row1.addWidget(self.roi_cancel_button)
        roi_row1.addWidget(QLabel("Label"))
        self.roi_label_edit = QLineEdit()
        self.roi_label_edit.setPlaceholderText("e.g. Incident wave")
        roi_row1.addWidget(self.roi_label_edit)
        self.roi_add_button = QPushButton("Add to list")
        self.roi_add_button.setObjectName("primary")
        self.roi_add_button.setEnabled(False)
        self.roi_add_button.clicked.connect(self._on_roi_add_clicked)
        roi_row1.addWidget(self.roi_add_button)
        roi_row1.addStretch(1)
        wave_card.add_layout(roi_row1)

        self.roi_waves_list = QListWidget()
        self.roi_waves_list.setMaximumHeight(140)
        wave_card.add(self.roi_waves_list)

        roi_row2 = QHBoxLayout()
        self.roi_remove_button = QPushButton("Remove selected")
        self.roi_remove_button.clicked.connect(self._on_roi_remove_clicked)
        roi_row2.addWidget(self.roi_remove_button)
        self.roi_export_button = QPushButton("Export all to Excel / CSV…")
        self.roi_export_button.clicked.connect(self._on_roi_export_clicked)
        roi_row2.addWidget(self.roi_export_button)
        roi_row2.addStretch(1)
        wave_card.add_layout(roi_row2)

        self.roi_status = QLabel(
            "Click \"Show field\" above first to show the field, then \"Draw ROI polygon\" and "
            "trace a loose loop around one visible band."
        )
        self.roi_status.setObjectName("cardSubtitle")
        self.roi_status.setWordWrap(True)
        wave_card.add(self.roi_status)
        layout.addWidget(wave_card)

        layout.addStretch(1)

        # ------------------------------------------------- Crack analysis sub-tab
        xfem_scroll, xfem_layout = make_scroll_page(
            "Crack analysis",
            "Automatic crack-tip detection and tracking from the raw u/v opening-jump discontinuity "
            "(ground truth, independent of any model fit), visual verification against the raw DIC "
            "field, and X-FEM-DIC: a third, independent measurement that re-correlates the raw "
            "reference/deformed images directly (not the already-exported DIC field) against a "
            "displacement model that already contains the crack discontinuity and near-tip singular "
            "field, so K_I/K_II come straight out of the correlation itself.",
        )
        self._build_xfem_dic_tab(xfem_layout)
        xfem_layout.addStretch(1)

        tabs = QTabWidget()
        tabs.addTab(scroll, "Wave speed analysis")
        tabs.addTab(xfem_scroll, "Crack analysis")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(tabs)

    # -- Data / tracing (built once per tab that needs them) -----------------

    def _build_data_card(self, suffix):
        """Builds one instance of the "Data" card -- called twice: once for
        the Wave speed analysis tab (`suffix=""`, its original location)
        and once at the top of the Crack analysis tab (`suffix="_ca"`),
        added so this page can be driven end to end (tracing, crack-tip
        detection, X-FEM-DIC) without ever switching to the Wave speed
        analysis tab. Both instances load into the SAME underlying
        `self._dataset`/`self._frame_source` -- there is still only ONE
        loaded dataset for the whole page, never two independent ones,
        just more than one place to load it from -- so every handler that
        acts on a choice made here (`_on_csv_selected`, `_load_frame_source`,
        `_on_video_file_selected`, `_on_browse_sequence_folder`,
        `_on_any_source_mode_toggled`) loops over every suffix in
        `self._data_card_suffixes` to keep every built instance's displayed
        status/path/mode in sync, regardless of which instance the user
        actually used."""
        data_card = Card(
            "Data",
            "Loads a previously exported DIC results CSV (Image Correlation page's \"Export DIC "
            "results\" button) and the matching image sequence/video, so this tool works entirely "
            "from a finished, high-resolution DIC run -- no re-correlation needed here.",
            help_html=(
                "<p>This tool is self-sufficient, like the other Standalone Tools: it does not "
                "read the Digital Image Correlation page's live results directly (those are never "
                "kept once you leave that page). Instead:</p>"
                "<ul>"
                "<li>Run your correlation on the Digital Image Correlation page, pushing the mesh "
                "resolution as fine as you need to resolve wave fronts / crack-tip localization, "
                "then use its \"Export DIC results\" button to save a CSV.</li>"
                "<li>Load that CSV here, plus the same image sequence or video used for that "
                "correlation run (for the tracing background).</li>"
                "</ul>"
                "<p>This page has two of these cards -- one on each tab -- but only ONE dataset "
                "is ever loaded at a time: loading a file from either card loads it for the whole "
                "page, and the other card's own status updates to match immediately.</p>"
            ),
        )
        csv_row = FileSelectRow("DIC results export (.csv)", "CSV files (*.csv);;All files (*)")
        csv_row.file_selected.connect(self._on_csv_selected)
        data_card.add(csv_row)
        csv_status = QLabel("No data loaded.")
        csv_status.setObjectName("cardSubtitle")
        csv_status.setWordWrap(True)
        data_card.add(csv_status)

        mode_row = QHBoxLayout()
        video_radio = QRadioButton("Video file (.mp4 / .mkv / .avi)")
        sequence_radio = QRadioButton("Image sequence (folder)")
        video_radio.setChecked(True)
        source_mode_group = QButtonGroup(self)
        source_mode_group.addButton(video_radio)
        source_mode_group.addButton(sequence_radio)
        video_radio.toggled.connect(lambda _checked, vr=video_radio: self._on_any_source_mode_toggled(vr))
        mode_row.addWidget(video_radio)
        mode_row.addWidget(sequence_radio)
        mode_row.addStretch(1)
        data_card.add_layout(mode_row)

        video_row = FileSelectRow("Video file", "Video files (*.mp4 *.mkv *.avi);;All files (*)")
        video_row.file_selected.connect(self._on_video_file_selected)
        data_card.add(video_row)

        seq_row = QHBoxLayout()
        seq_row.addWidget(QLabel("Image folder"))
        seq_folder_edit = QLineEdit()
        seq_folder_edit.setReadOnly(True)
        seq_folder_edit.setPlaceholderText("No folder selected")
        seq_row.addWidget(seq_folder_edit, 1)
        seq_browse_button = QPushButton("Browse…")
        seq_browse_button.clicked.connect(self._on_browse_sequence_folder)
        seq_row.addWidget(seq_browse_button)
        seq_row_widget = QWidget()
        seq_row_widget.setLayout(seq_row)
        seq_row_widget.setVisible(False)
        data_card.add(seq_row_widget)

        frames_status = QLabel("No frames loaded.")
        frames_status.setObjectName("cardSubtitle")
        data_card.add(frames_status)

        setattr(self, f"csv_row{suffix}", csv_row)
        setattr(self, f"csv_status{suffix}", csv_status)
        setattr(self, f"video_radio{suffix}", video_radio)
        setattr(self, f"sequence_radio{suffix}", sequence_radio)
        setattr(self, f"video_row{suffix}", video_row)
        setattr(self, f"seq_folder_edit{suffix}", seq_folder_edit)
        setattr(self, f"seq_row_widget{suffix}", seq_row_widget)
        setattr(self, f"frames_status{suffix}", frames_status)
        self._data_card_suffixes.append(suffix)
        return data_card

    def _build_tracing_card(self):
        """Builds the "Crack Skeleton Tracing" card -- moved here, to the
        top of the Crack analysis tab (right under that tab's own "Data"
        card), from its original spot on the Wave speed analysis tab:
        tracing isn't actually used by that tab's own wave-front ROI
        picking (which works directly off the raw field, no traced path
        needed), only by this tab's crack-tip detection and X-FEM-DIC, so
        it belongs here instead."""
        tracing_card = Card(
            "Crack Skeleton Tracing",
            "Pick a frame and field, then trace (or auto-detect) each crack's path -- the pre-"
            "traced skeleton the next stages will track a crack tip's position along.",
            help_html=(
                "<p>Choose which processed frame and which strain field to display, then trace "
                "each crack's path on it.</p>"
                "<ul>"
                "<li><b>Frame</b>: scrub to a frame late enough that the crack (or wave front) "
                "you want to trace is visible in the field.</li>"
                "<li><b>Field</b>: MISES (equivalent/von Mises strain) usually shows a "
                "localization band regardless of crack orientation; exx/eyy can show it more "
                "sharply for a crack running mostly along one direction. For an oblique crack, "
                "e1 (max principal strain, = crack opening) and gmax (max shear strain) are "
                "orientation-independent and usually show it more sharply than exx/eyy.</li>"
                "<li><b>Field color scale</b> (in the tracing panel): Auto re-fits to each "
                "frame's own data; switch to Fixed and narrow the Min/Max to pull a faint crack "
                "band out of a field whose range is dominated by a much larger value elsewhere.</li>"
                "<li><b>Align background to strain field</b> (in the tracing panel, on by "
                "default): the field/crack overlay stay at each grid point's fixed reference "
                "position, but the background is the raw current frame -- once the specimen has "
                "moved noticeably since the reference frame (common in SHPB/Direct-Impact tests), "
                "this shifts the background by the frame's own median displacement so it re-"
                "aligns with the field. This is a single global correction (not a full per-point "
                "un-warp), so a residual local misalignment can remain right at the crack tip in "
                "a strongly-strained region -- uncheck it to compare against the raw frame.</li>"
                "<li>See the tracing panel's own controls for how to draw/branch/auto-detect.</li>"
                "</ul>"
            ),
        )
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame"))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        frame_row.addWidget(self.frame_slider, 1)
        self.frame_spin = QSpinBox()
        self.frame_spin.setEnabled(False)
        self.frame_spin.setMinimumWidth(90)
        self.frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        frame_row.addWidget(self.frame_spin)
        frame_row.addWidget(QLabel("Field"))
        self.field_combo = QComboBox()
        for key, label in FIELD_CHOICES:
            self.field_combo.addItem(label, key)
        self.field_combo.currentIndexChanged.connect(self._refresh_tracing_frame)
        frame_row.addWidget(self.field_combo)
        tracing_card.add_layout(frame_row)

        self.frame_info_label = QLabel("")
        self.frame_info_label.setObjectName("cardSubtitle")
        tracing_card.add(self.frame_info_label)

        self.picker = CrackSkeletonPicker(figsize=(9, 6))
        frame_row.addWidget(make_popout_button(
            self, "Crack Skeleton Tracing -- field", self.picker.canvas, self.frame_slider,
            info_labels=[self.picker.status_label]))
        tracing_card.add(self.picker)
        return tracing_card

    def _build_pixel_size_row(self, suffix):
        """One "Pixel size" field + "Sync from DIC tab" button -- built
        once per suffix, same pattern as `_build_data_card`, so the value
        is editable from both the Wave speed analysis tab (suffix "")
        and the Crack analysis tab (suffix "_ca"'s own Material &
        Calibration card), always mirrored to the single value every
        speed conversion and K_I/K_II fit on this page actually reads
        (`self.pixel_size_spin` -- the suffix "" instance is the
        original attribute name every pre-existing call site already
        uses, so nothing else on this page needs to change). Returns the
        row layout; the caller adds it into whichever Card it belongs
        in and is responsible for its own status label (see
        `self.pixel_size_status{suffix}`, read by `_on_sync_pixel_size`
        and `_on_csv_selected`'s auto-fill)."""
        row = QHBoxLayout()
        row.addWidget(QLabel("Pixel size"))
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1000.0)
        spin.setDecimals(6)
        spin.setValue(0.0)
        spin.setSuffix(" mm/px (0 = px/s only)")
        setattr(self, f"pixel_size_spin{suffix}", spin)
        spin.valueChanged.connect(lambda value, s=spin: self._on_pixel_size_changed(s, value))
        row.addWidget(spin)
        sync_button = QPushButton("Sync from DIC tab")
        sync_button.clicked.connect(self._on_sync_pixel_size)
        setattr(self, f"sync_pixel_size_button{suffix}", sync_button)
        row.addWidget(sync_button)
        row.addStretch(1)
        self._pixel_size_spin_suffixes.append(suffix)
        return row

    def _on_pixel_size_changed(self, changed_spin, value):
        """Mirrors a pixel-size edit from any one instance (Wave speed
        analysis tab or Crack analysis tab) to every other one -- never
        via `self.sender()` (not implemented by this app's own PySide6
        test stub, see `_on_any_source_mode_toggled`), instead identifying
        the widget that actually changed via the lambda-bound `changed_spin`
        each instance's `valueChanged` connection already captures."""
        for suffix in self._pixel_size_spin_suffixes:
            spin = getattr(self, f"pixel_size_spin{suffix}")
            if spin is changed_spin:
                continue
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _apply_pixel_size_from_dataset(self, dataset):
        """Auto-fills the "Pixel size" field(s) from `dataset.pixel_size_mm`
        (recovered from the loaded CSV's own u_mm/v_mm columns, see
        `core.crack_wave_io._recover_pixel_size_mm`) -- called from
        `_on_csv_selected` right after a CSV loads successfully.

        Only APPLIES the recovered value when pixel size is still unset
        (0.0, the field's own default) -- never silently overwrites a
        value the user already entered or synced from the DIC tab, same
        "don't clobber an existing choice" convention the Image
        Correlation page's own frame-rate auto-fill from a loaded video
        follows. Either way, every instance's status label says plainly
        what happened (recovered, not recoverable, or recovered but not
        applied) rather than leaving the user to guess."""
        if dataset.pixel_size_mm is not None:
            if self.pixel_size_spin.value() == 0.0:
                self.pixel_size_spin.setValue(dataset.pixel_size_mm)  # mirrors to every instance
                message = f"Pixel size auto-detected from the CSV: {dataset.pixel_size_mm:.6g} mm/px."
            else:
                message = (
                    f"Pixel size available in the CSV ({dataset.pixel_size_mm:.6g} mm/px) but not "
                    f"applied -- a value is already set ({self.pixel_size_spin.value():.6g} mm/px)."
                )
        else:
            message = ""
        for suffix in self._pixel_size_spin_suffixes:
            label = getattr(self, f"pixel_size_status{suffix}", None)
            if label is not None:
                label.setText(message)

    # -- X-FEM-DIC sub-tab --------------------------------------------------

    def _build_tip_detection_card(self, layout):
        """Builds the "Crack Tip Detection" card at the top of the X-FEM-
        DIC tab -- this page's only tip-tracking method, and its own
        preparation step for X-FEM-DIC.

        Detects the crack tip's position at EVERY frame in range,
        automatically, from the raw |u|/|v| DISPLACEMENT DISCONTINUITY
        across the traced path (offset "Offset" px to either side, along
        the path's own local normal) -- independent of any model fit, so
        it's also this page's ground-truth measurement. "Threshold" is the
        combined jump magnitude (px) above which the crack counts as open
        at a point -- defaults to 0.15 px, the same default X-FEM-DIC's own
        Heaviside-jump auto-detect uses.

        A trace added here (\"Detect tip position (all frames)\") lands in
        the SAME `self._tip_roi_traces` list the \"Crack path source\" card
        below ticks from, so it's immediately usable as X-FEM-DIC's own
        reference tip trajectory. \"Verify visually\" below lets that
        trace's own per-frame detected position be checked directly
        against the standard subset-DIC displacement isocontours it was
        measured from, with a frame slider; the arc-length-vs-time plot
        under it shows the trajectory itself."""
        tip_card = Card(
            "Crack Tip Detection (opening jump, ground truth)",
            "Detects the crack tip's position at every frame, automatically, from the raw |u|/|v| "
            "displacement discontinuity across the traced path -- independent of any model fit. "
            "The resulting trace is this tab's own reference tip trajectory: tick it below in "
            "\"Crack path source\" to run X-FEM-DIC on it.",
            help_html=(
                "<p>At every point along the traced path (from its first-placed point, taken as the "
                "crack's initiation point, out to the chosen tip), this samples u and v \"Offset\" px "
                "to either side of the path, along its local normal direction, and reports "
                "|u(+offset) - u(-offset)| and |v(+offset) - v(-offset)| separately, plus their "
                "combined magnitude hypot(jump_u, jump_v). A genuinely open crack face shows a sharp, "
                "sustained jump; a closed crack or a fracture process zone (elevated strain without "
                "real separation) does not -- this is what makes it reliable where a derived scalar "
                "field (strain) crossing a threshold, or forming a local peak, on the traced path can "
                "lose the true tip early: that signal can fade out, or the path can sit slightly off "
                "the true crack line, well before the crack has actually stopped opening there.</p>"
                "<ul>"
                "<li><b>Cracks / Tip</b>: tick the cracks to process in the \"Cracks\" list "
                "(all ticked by default; \"All\" / \"None\" are shortcuts) -- \"Detect tip position\" "
                "then runs on every ticked crack in one go and adds one trace per crack. A branching "
                "crack can have more than one tip: for the crack you clicked last, \"Tip\" picks which "
                "one to track (the other ticked cracks use their longest branch). The lists refresh "
                "automatically as you trace (Crack Skeleton Tracing card above); the Tip list is "
                "sorted longest path first, and flags any branch under 20% of the longest one's "
                "length.</li>"
                "<li><b>Frame range / Reverse path direction</b>: restrict to the part of the "
                "sequence where this crack is actually growing; flip which end of the path is arc "
                "length 0 if a trace advances the \"wrong way\".</li>"
                "<li><b>Offset / Threshold</b>: see above -- 4 px / 0.15 px are reasonable starting "
                "points for typical subset-DIC point spacing.</li>"
                "<li><b>Enforce monotonic advance</b> (on by default) rejects a frame's detected "
                "position if it would mean the tip receding beyond \"Backslide tolerance\" from the "
                "furthest position already kept.</li>"
                "<li><b>Smooth tip position</b>: when checked, a newly detected trace has its "
                "position locally smoothed (over the window given) BEFORE being stored, replacing "
                "the raw per-frame walk -- affects only a trace at the moment it's detected.</li>"
                "<li>Name the trace, \"Detect tip position (all frames)\", repeat for other branches/"
                "settings, then tick it in \"Crack path source\" below to run X-FEM-DIC on it, or "
                "\"Export all to CSV…\" for the raw points plus a locally-derived propagation "
                "velocity.</li>"
                "<li><b>Verify visually</b>, further below: a live viewer of the standard subset-DIC "
                "displacement (u, v or the magnitude |u|) or strain field (MISES, exx, eyy, exy, e1, e2, gmax -- the same "
                "strains as in Crack Skeleton Tracing), drawn as filled isocontours with a continuous "
                "colour scale -- same visual convention "
                "as the \"Field isocontours\" card further down -- with the traced path and (once a "
                "traces are ticked in \"Show trace\", all by default) each trace's own trajectory and "
                "per-frame detected position overlaid, in the trace's colour. \"Color scale\" lets you fix the display range instead "
                "of it re-normalizing to every new frame. The arc-length-vs-time plot underneath "
                "plots every trace ticked in \"Show trace\" (all by default), position along "
                "the path against time.</li>"
                "</ul>"
            ),
        )
        tip_row1 = QHBoxLayout()
        tip_row1.addWidget(QLabel("Cracks"))
        # Tick the cracks to process (all ticked by default): "Detect tip position"
        # below runs on every ticked crack. The crack last clicked in the list is
        # the one the "Tip" choice and the "Verify visually" viewer show.
        self.tip_crack_combo = CheckableComboBox(noun="crack", all_text="All cracks ({n})")
        self.tip_crack_combo.setMinimumWidth(190)
        self.tip_crack_combo.setToolTip(
            "Tick the cracks to process -- all are ticked by default. \"Detect tip position\" "
            "below runs on every ticked crack (each crack's longest branch, or the \"Tip\" "
            "chosen on the right for the highlighted crack). The crack you clicked last is the "
            "one the \"Tip\" list and the \"Verify visually\" viewer below show."
        )
        self.tip_crack_combo.currentIndexChanged.connect(self._on_tip_crack_changed)
        self.tip_crack_combo.checkedChanged.connect(self._on_tip_crack_checks_changed)
        tip_row1.addWidget(self.tip_crack_combo)
        tip_all_button = QPushButton("All")
        tip_all_button.setToolTip("Tick every crack.")
        tip_all_button.clicked.connect(lambda: self.tip_crack_combo.set_all_checked(True))
        tip_row1.addWidget(tip_all_button)
        tip_none_button = QPushButton("None")
        tip_none_button.setToolTip("Untick every crack.")
        tip_none_button.clicked.connect(lambda: self.tip_crack_combo.set_all_checked(False))
        tip_row1.addWidget(tip_none_button)
        tip_row1.addWidget(QLabel("Tip"))
        self.tip_leaf_combo = QComboBox()
        self.tip_leaf_combo.currentIndexChanged.connect(self._on_tip_leaf_changed)
        tip_row1.addWidget(self.tip_leaf_combo)
        refresh_cracks_button = QPushButton("Refresh list")
        refresh_cracks_button.clicked.connect(self._refresh_tip_crack_combo)
        tip_row1.addWidget(refresh_cracks_button)
        tip_row1.addStretch(1)
        tip_card.add_layout(tip_row1)

        tip_row2 = QHBoxLayout()
        self.tip_reverse_path_check = QCheckBox("Reverse path direction (flip which end is arc length 0)")
        self.tip_reverse_path_check.setChecked(False)
        self.tip_reverse_path_check.setToolTip(
            "The traced path's arc length starts at 0 at whichever node you placed first when "
            "tracing the crack, and increases toward the other end. If a detected trace shows arc "
            "length running the 'wrong way' relative to how the crack actually grows, check this "
            "box to flip which end is arc length 0 -- without needing to re-trace the crack. This "
            "also fixes the direction used by 'Enforce monotonic advance'. Any already-detected "
            "trace(s) are cleared when this changes, since they were computed against the previous "
            "orientation."
        )
        self.tip_reverse_path_check.toggled.connect(self._on_tip_reverse_path_toggled)
        tip_row2.addWidget(self.tip_reverse_path_check)
        tip_row2.addStretch(1)
        tip_card.add_layout(tip_row2)

        tip_row3 = QHBoxLayout()
        tip_row3.addWidget(QLabel("Frame range"))
        self.tip_start_spin = QSpinBox()
        self.tip_start_spin.setRange(0, 0)
        self.tip_start_spin.valueChanged.connect(self._on_tip_verify_context_input_changed)
        tip_row3.addWidget(self.tip_start_spin)
        tip_row3.addWidget(QLabel("to"))
        self.tip_end_spin = QSpinBox()
        self.tip_end_spin.setRange(0, 0)
        self.tip_end_spin.valueChanged.connect(self._on_tip_verify_context_input_changed)
        tip_row3.addWidget(self.tip_end_spin)
        tip_row3.addWidget(QLabel("Frame rate override"))
        self.tip_rate_spin = QDoubleSpinBox()
        self.tip_rate_spin.setRange(0.0, 1.0e9)
        self.tip_rate_spin.setDecimals(1)
        self.tip_rate_spin.setValue(0.0)
        self.tip_rate_spin.valueChanged.connect(self._on_tip_verify_context_input_changed)
        self.tip_rate_spin.setSuffix(" Hz (0 = use the CSV's own timing)")
        tip_row3.addWidget(self.tip_rate_spin)
        tip_row3.addStretch(1)
        tip_card.add_layout(tip_row3)

        tip_roi_opening_jump_row = QHBoxLayout()
        tip_roi_opening_jump_row.addWidget(QLabel("Offset"))
        self.tip_roi_opening_jump_offset_spin = QDoubleSpinBox()
        self.tip_roi_opening_jump_offset_spin.setRange(0.5, 100.0)
        self.tip_roi_opening_jump_offset_spin.setDecimals(2)
        self.tip_roi_opening_jump_offset_spin.setValue(4.0)
        self.tip_roi_opening_jump_offset_spin.setSuffix(" px")
        self.tip_roi_opening_jump_offset_spin.setToolTip(
            "Distance from the traced path, to either side along its local normal, where u/v are "
            "sampled to measure the jump -- needs to be large enough that the two sample points sit "
            "on opposite sides of the (typically few-px-wide) speckle-correlation blur straddling "
            "the crack face, but small enough to stay a genuinely LOCAL measurement."
        )
        tip_roi_opening_jump_row.addWidget(self.tip_roi_opening_jump_offset_spin)
        tip_roi_opening_jump_row.addWidget(QLabel("Threshold"))
        self.tip_roi_opening_jump_threshold_spin = QDoubleSpinBox()
        self.tip_roi_opening_jump_threshold_spin.setRange(0.001, 50.0)
        self.tip_roi_opening_jump_threshold_spin.setDecimals(3)
        self.tip_roi_opening_jump_threshold_spin.setValue(0.15)
        self.tip_roi_opening_jump_threshold_spin.setSuffix(" px")
        self.tip_roi_opening_jump_threshold_spin.setToolTip(
            "Combined jump magnitude (hypot of the u and v jumps) above which the crack is "
            "considered open at a point. Defaults to 0.15 px, the same default X-FEM-DIC's own "
            "Heaviside-jump auto-detect uses."
        )
        tip_roi_opening_jump_row.addWidget(self.tip_roi_opening_jump_threshold_spin)
        tip_roi_opening_jump_row.addWidget(QLabel("Gap tolerance"))
        self.tip_roi_opening_jump_max_gap_spin = QSpinBox()
        self.tip_roi_opening_jump_max_gap_spin.setRange(0, 100)
        self.tip_roi_opening_jump_max_gap_spin.setValue(2)
        self.tip_roi_opening_jump_max_gap_spin.setSuffix(" sample(s)")
        self.tip_roi_opening_jump_max_gap_spin.setToolTip(
            "How many consecutive below-threshold resampled points (anywhere along the walk from "
            "the path's own origin, INCLUDING right at the origin itself) are tolerated before the "
            "walk gives up -- correlation noise/a single bad point, not a real gap in the crack. The "
            "traced path's own starting point is often the single worst-correlated point on the "
            "whole path (a manually-placed click, or the very tip of an auto-detected skeleton, "
            "which is a known least-reliable spot for skeletonization) -- if \"Add to list\" reports "
            "the threshold was never reached even though a frame's own opening-jump profile (see "
            "\"Verify visually\" below) clearly shows a wide, clean opening starting right around the "
            "origin, try raising this BEFORE lowering Threshold or raising Offset: a handful of bad "
            "samples right at the start needs a larger gap tolerance to see past, not a different "
            "threshold. Each sample is one resample step (2 px by default), so 2 samples = 4 px."
        )
        tip_roi_opening_jump_row.addWidget(self.tip_roi_opening_jump_max_gap_spin)
        tip_roi_opening_jump_row.addWidget(QLabel("Jump smoothing"))
        self.tip_roi_opening_jump_smooth_spin = QSpinBox()
        self.tip_roi_opening_jump_smooth_spin.setRange(1, 101)
        self.tip_roi_opening_jump_smooth_spin.setValue(1)
        self.tip_roi_opening_jump_smooth_spin.setSuffix(" sample(s) (1 = off)")
        self.tip_roi_opening_jump_smooth_spin.setToolTip(
            "Width of a rolling MEDIAN filter run along the path on the raw jump-magnitude signal "
            "(jump_mag_px), BEFORE the threshold walk above -- suppresses a short, high-amplitude "
            "spike such as a passing elastic wave pulse crossing the path (visible in \"Verify "
            "visually\" below as a narrow, isolated bump in the jump-vs-arc-length curve), while "
            "leaving a genuine, sustained crack-opening plateau essentially unaffected. 1 (default) "
            "disables this -- the raw jump signal is used as-is, same as before this option existed. "
            "Pick a value a bit wider than the pulse's own visible width in that plot, but well short "
            "of the shortest genuine opening length expected near the crack's own initiation -- too "
            "wide a window starts eating into real, short openings."
        )
        tip_roi_opening_jump_row.addWidget(self.tip_roi_opening_jump_smooth_spin)
        tip_roi_opening_jump_row.addStretch(1)
        tip_card.add_layout(tip_roi_opening_jump_row)

        # Which component of the displacement jump the threshold is applied to.
        tip_roi_opening_jump_wave_row = QHBoxLayout()
        tip_roi_opening_jump_wave_row.addWidget(QLabel("Criterion"))
        self.tip_roi_opening_jump_criterion_combo = QComboBox()
        for key, text in JUMP_CRITERIA:
            self.tip_roi_opening_jump_criterion_combo.addItem(text, key)
        self.tip_roi_opening_jump_criterion_combo.setToolTip(
            "Which component of the displacement jump across the crack the threshold is applied to. "
            "|jump| of (u, v) combined = the original criterion. Opening = only the component NORMAL to "
            "the crack, in its own local frame (the faces moving apart; overlap counts as 0) -- ignores "
            "sliding and rigid rotation of the lips. Sliding = the component ALONG the crack (mode II). "
            "Same threshold in px for all three.")
        self.tip_roi_opening_jump_criterion_combo.currentIndexChanged.connect(self._on_tip_verify_redraw_needed)
        tip_roi_opening_jump_wave_row.addWidget(self.tip_roi_opening_jump_criterion_combo)
        tip_roi_opening_jump_wave_row.addStretch(1)
        tip_card.add_layout(tip_roi_opening_jump_wave_row)

        self.tip_roi_monotonic_check = QCheckBox("Enforce monotonic advance (recommended for a propagating crack)")
        self.tip_roi_monotonic_check.setChecked(True)
        self.tip_roi_monotonic_check.setToolTip(
            "A crack tip only ever advances. When checked, a frame's detected point is only "
            "accepted if it isn't clearly behind the furthest point already accepted -- this stops "
            "a spurious nearby-but-unrelated jump (e.g. the crack faces' own opening/closing signal) "
            "from being chased frame to frame. Which way counts as \"advancing\" is detected "
            "automatically from the raw walk, not assumed."
        )
        tip_card.add(self.tip_roi_monotonic_check)

        tip_roi_constraint_row = QHBoxLayout()
        tip_roi_constraint_row.addWidget(QLabel("Backslide tolerance"))
        self.tip_roi_backslide_spin = QDoubleSpinBox()
        self.tip_roi_backslide_spin.setRange(0.0, 500.0)
        self.tip_roi_backslide_spin.setValue(0.0)
        self.tip_roi_backslide_spin.setSuffix(" px (0 = auto = 3x resample step)")
        self.tip_roi_backslide_spin.setToolTip(
            "How far behind the furthest already-accepted position (in px of arc length) a frame's "
            "candidate is still allowed to land before \"Enforce monotonic advance\" rejects it as a "
            "backward jump. Auto (0) uses 3x the resample step (2 px), i.e. 6 px."
        )
        tip_roi_constraint_row.addWidget(self.tip_roi_backslide_spin)
        tip_roi_constraint_row.addWidget(QLabel("Advance confirmation"))
        self.tip_roi_advance_confirm_spin = QSpinBox()
        self.tip_roi_advance_confirm_spin.setRange(1, 50)
        self.tip_roi_advance_confirm_spin.setValue(1)
        self.tip_roi_advance_confirm_spin.setSuffix(" frame(s) (1 = off)")
        self.tip_roi_advance_confirm_spin.setToolTip(
            "A frame's candidate that jumps more than \"Backslide tolerance\" AHEAD of the furthest "
            "position already accepted is only accepted once this many consecutive frames, starting "
            "with it, all hold up at/near that same advanced level -- a single frame's spurious FAR "
            "jump (e.g. a wave pulse briefly sitting on the path well ahead of the true tip) is "
            "otherwise accepted immediately and permanently raises the accepted-so-far position, "
            "silently rejecting every genuine, lower point that follows until the real crack catches "
            "up to it (if it ever does within this trace). 1 (default) disables this -- any forward "
            "move is accepted immediately, same as before this option existed. This does NOT assume "
            "or bound any crack propagation speed -- deliberately, since it can vary far too much in "
            "a heterogeneous material to fix a limit on it; it only asks a large advance to hold up "
            "over a few frames before trusting it. Near the very end of a trace there may not be "
            "enough remaining frames to confirm a genuine fast advance -- those last few points are "
            "then conservatively dropped rather than risk locking in an unconfirmed spike."
        )
        tip_roi_constraint_row.addWidget(self.tip_roi_advance_confirm_spin)
        tip_roi_constraint_row.addStretch(1)
        tip_card.add_layout(tip_roi_constraint_row)

        tip_roi_smoothing_row = QHBoxLayout()
        self.tip_roi_show_smoothed_check = QCheckBox("Smooth tip position (replaces the raw path when added)")
        self.tip_roi_show_smoothed_check.setChecked(False)
        self.tip_roi_show_smoothed_check.setToolTip(
            "Off by default -- the raw, per-frame detected position is used. When checked, a "
            "trace detected while this is on has its position locally smoothed (over the window "
            "below) BEFORE being stored -- the raw per-frame walk is not kept, and every downstream "
            "use (X-FEM-DIC, CSV export, the arc-length-vs-time plot) sees the smoothed path."
        )
        tip_roi_smoothing_row.addWidget(self.tip_roi_show_smoothed_check)
        tip_roi_smoothing_row.addWidget(QLabel("Smoothing window"))
        self.tip_roi_smoothing_window_spin = QDoubleSpinBox()
        self.tip_roi_smoothing_window_spin.setRange(0.0, 2000.0)
        self.tip_roi_smoothing_window_spin.setValue(20.0)
        self.tip_roi_smoothing_window_spin.setSuffix(" px")
        tip_roi_smoothing_row.addWidget(self.tip_roi_smoothing_window_spin)
        tip_roi_smoothing_row.addStretch(1)
        tip_card.add_layout(tip_roi_smoothing_row)

        tip_roi_uncertainty_row = QHBoxLayout()
        self.tip_roi_uncertainty_check = QCheckBox("Estimate measurement uncertainty (position & velocity)")
        self.tip_roi_uncertainty_check.setChecked(False)
        self.tip_roi_uncertainty_check.setToolTip(
            "Off by default (roughly 3x slower to detect, since each accepted frame is "
            "re-walked with perturbed threshold/offset/gap settings). When checked, each "
            "accepted frame also gets a position-uncertainty estimate (px), combining: local "
            "image noise on the opening-jump signal (from the opening-jump profile itself), "
            "sensitivity to the Threshold and Offset controls above, sensitivity to Gap "
            "tolerance, and the path's own sampling resolution. The velocity fit then becomes "
            "a weighted least-squares fit using these per-frame uncertainties, giving a proper "
            "standard error on the speed instead of just R². Optionally add the camera's own "
            "frame-rate uncertainty below to fold in recording-frequency error too."
        )
        tip_roi_uncertainty_row.addWidget(self.tip_roi_uncertainty_check)
        tip_roi_uncertainty_row.addWidget(QLabel("Frame-rate uncertainty"))
        self.tip_roi_rate_uncertainty_spin = QDoubleSpinBox()
        self.tip_roi_rate_uncertainty_spin.setRange(0.0, 50.0)
        self.tip_roi_rate_uncertainty_spin.setValue(0.0)
        self.tip_roi_rate_uncertainty_spin.setSuffix(" %")
        self.tip_roi_rate_uncertainty_spin.setEnabled(False)
        self.tip_roi_rate_uncertainty_spin.setToolTip(
            "Relative uncertainty on the camera's recording frequency (e.g. jitter/drift in the "
            "high-speed camera's actual frame rate vs. its nominal setting), as a percentage. "
            "Folded into the fitted speed's standard error in quadrature. Leave at 0 if the "
            "frame rate is trusted exactly."
        )
        tip_roi_uncertainty_row.addWidget(self.tip_roi_rate_uncertainty_spin)
        tip_roi_uncertainty_row.addStretch(1)
        tip_card.add_layout(tip_roi_uncertainty_row)
        self.tip_roi_uncertainty_check.toggled.connect(self.tip_roi_rate_uncertainty_spin.setEnabled)

        tip_roi_row1 = QHBoxLayout()
        tip_roi_row1.addWidget(QLabel("Label"))
        self.tip_roi_label_edit = QLineEdit()
        self.tip_roi_label_edit.setPlaceholderText("e.g. Main tip")
        tip_roi_row1.addWidget(self.tip_roi_label_edit)
        self.tip_roi_add_button = QPushButton("Detect tip position (all frames) -> Add to list")
        self.tip_roi_add_button.setObjectName("primary")
        self.tip_roi_add_button.setEnabled(False)
        self.tip_roi_add_button.clicked.connect(self._on_tip_roi_add_clicked)
        tip_roi_row1.addWidget(self.tip_roi_add_button)
        tip_roi_row1.addStretch(1)
        tip_card.add_layout(tip_roi_row1)

        self.tip_roi_list = QListWidget()
        self.tip_roi_list.setMaximumHeight(140)
        tip_card.add(self.tip_roi_list)

        tip_roi_row2 = QHBoxLayout()
        self.tip_roi_remove_button = QPushButton("Remove selected")
        self.tip_roi_remove_button.clicked.connect(self._on_tip_roi_remove_clicked)
        tip_roi_row2.addWidget(self.tip_roi_remove_button)
        self.tip_roi_delete_all_button = QPushButton("Delete all")
        self.tip_roi_delete_all_button.setToolTip("Remove every detected trace from the list (asks for confirmation).")
        self.tip_roi_delete_all_button.clicked.connect(self._on_tip_roi_delete_all_clicked)
        tip_roi_row2.addWidget(self.tip_roi_delete_all_button)
        self.tip_roi_export_button = QPushButton("Export all to CSV…")
        self.tip_roi_export_button.clicked.connect(self._on_tip_roi_export_clicked)
        tip_roi_row2.addWidget(self.tip_roi_export_button)
        tip_roi_row2.addStretch(1)
        tip_card.add_layout(tip_roi_row2)

        self.tip_roi_status = QLabel(
            "Pick a Crack/Tip above, then click \"Detect tip position (all frames) -> Add to list\"."
        )
        self.tip_roi_status.setObjectName("cardSubtitle")
        self.tip_roi_status.setWordWrap(True)
        tip_card.add(self.tip_roi_status)
        # `self.tip_crack_combo`/`_dataset` already exist above, so it's
        # safe to sync the "Add to list" button's initial enabled state
        # here (it stays disabled until a dataset is loaded).
        self.tip_roi_add_button.setEnabled(self._dataset is not None and bool(self._dataset.frames))

        # "Verify visually" -- a live viewer of the standard subset-DIC
        # displacement field, drawn as filled isocontours (same
        # convention as the "Field isocontours" card further down), with
        # the traced path and a picked trace's own trajectory/per-frame
        # position overlaid. Reuses this card's own Crack/Tip/Frame range/
        # Reverse path selectors above rather than duplicating them.
        self.tip_verify_group = QWidget()
        tip_verify_group_layout = QVBoxLayout(self.tip_verify_group)
        tip_verify_group_layout.setContentsMargins(0, 0, 0, 0)

        tip_verify_label = QLabel(
            "Verify visually -- standard subset-DIC displacement / strain isocontours, with a detected "
            "trace's own per-frame position overlaid (see the \"?\" button above)"
        )
        tip_verify_label.setObjectName("cardSubtitle")
        tip_verify_group_layout.addWidget(tip_verify_label)

        tip_verify_row1 = QHBoxLayout()
        tip_verify_row1.addWidget(QLabel("Field"))
        self.tip_verify_field_combo = QComboBox()
        self.tip_verify_field_combo.addItem("u -- horizontal displacement (px)", "u")
        self.tip_verify_field_combo.addItem("v -- vertical displacement (px)", "v")
        self.tip_verify_field_combo.addItem("|u| -- displacement magnitude sqrt(u\u00b2+v\u00b2) (px)", "umag")
        for key, (label, _kind) in LOCAL_FRAME_FIELDS.items():
            self.tip_verify_field_combo.addItem(label, key)
        # Every strain field the Crack Skeleton Tracing card offers (MISES, exx,
        # eyy, exy, e1, e2, gmax) -- core.crack_wave_io.FIELD_CHOICES.
        for key, label in FIELD_CHOICES:
            self.tip_verify_field_combo.addItem(label, key)
        self.tip_verify_field_combo.currentIndexChanged.connect(self._on_tip_verify_field_changed)
        tip_verify_row1.addWidget(self.tip_verify_field_combo)
        tip_verify_row1.addWidget(QLabel("Show trace"))
        # Tick the traces to display (all ticked by default), in the field view AND in
        # the arc-length-vs-time plot; each keeps its own colour. The trace clicked last
        # is the "active" one (uncertainty band on the jump profile, video export).
        self.tip_verify_trace_combo = CheckableComboBox(noun="trace", all_text="All traces ({n})")
        self.tip_verify_trace_combo.setMinimumWidth(190)
        self.tip_verify_trace_combo.setToolTip(
            "Tick the detected traces to show (all ticked by default) -- in the field view above "
            "and in the arc-length-vs-time plot below; each keeps its colour. The trace you clicked "
            "last is the one used by \"Export video\" and by the uncertainty band on the jump profile."
        )
        self._tip_verify_unticked_traces = set()     # id(trace) of traces the user unticked
        self.tip_verify_trace_combo.checkedChanged.connect(self._on_tip_verify_traces_changed)
        tip_verify_row1.addWidget(self.tip_verify_trace_combo)
        tip_verify_all_button = QPushButton("All")
        tip_verify_all_button.clicked.connect(lambda: self.tip_verify_trace_combo.set_all_checked(True))
        tip_verify_row1.addWidget(tip_verify_all_button)
        tip_verify_none_button = QPushButton("None")
        tip_verify_none_button.clicked.connect(lambda: self.tip_verify_trace_combo.set_all_checked(False))
        tip_verify_row1.addWidget(tip_verify_none_button)
        tip_verify_row1.addWidget(QLabel("Uncertainty envelope"))
        self.tip_verify_uncertainty_sigma_combo = QComboBox()
        self.tip_verify_uncertainty_sigma_combo.addItem("Off", 0)
        self.tip_verify_uncertainty_sigma_combo.addItem("+/- 1 sigma", 1)
        self.tip_verify_uncertainty_sigma_combo.addItem("+/- 2 sigma", 2)
        self.tip_verify_uncertainty_sigma_combo.addItem("+/- 3 sigma", 3)
        self.tip_verify_uncertainty_sigma_combo.setCurrentIndex(1)  # +/- 1 sigma by default
        self.tip_verify_uncertainty_sigma_combo.setToolTip(
            "Shades a +/- N standard-deviation uncertainty band: around the detected front on "
            "the opening-jump profile below (for the trace picked in \"Show trace\" above, at "
            "this frame), and -- on the arc-length-vs-time plot further below -- around both "
            "each trace's own position curve (left axis) and its local-speed curve (right "
            "axis, propagated from the same position uncertainty). Only traces added with "
            "\"Estimate measurement uncertainty\" checked (see the detection card above) have "
            "a band to show -- others are drawn as plain lines/points either way."
        )
        self.tip_verify_uncertainty_sigma_combo.currentIndexChanged.connect(
            self._on_tip_verify_uncertainty_sigma_changed
        )
        tip_verify_row1.addWidget(self.tip_verify_uncertainty_sigma_combo)
        tip_verify_row1.addStretch(1)
        tip_verify_group_layout.addLayout(tip_verify_row1)

        # u_n / u_t only: keep the pixels right beside the crack faces (local measure), not the whole field.
        tip_verify_band_row = QHBoxLayout()
        self.tip_verify_band_check = QCheckBox("u_n / u_t: only a band along the cracks")
        self.tip_verify_band_check.setChecked(True)
        self.tip_verify_band_check.setToolTip(
            "For the local-frame displacement fields only: show just the pixels within the band below on "
            "either side of each ticked crack path (up to the crack's two ends), i.e. the local face "
            "separation / sliding, instead of the whole field projected on the nearest crack.")
        self.tip_verify_band_check.toggled.connect(self._on_tip_verify_redraw_needed)
        tip_verify_band_row.addWidget(self.tip_verify_band_check)
        tip_verify_band_row.addWidget(QLabel("from"))
        self.tip_verify_band_inner_spin = QDoubleSpinBox()
        self.tip_verify_band_inner_spin.setRange(0.0, 200.0)
        self.tip_verify_band_inner_spin.setDecimals(1)
        self.tip_verify_band_inner_spin.setValue(0.0)
        self.tip_verify_band_inner_spin.setSuffix(" px")
        self.tip_verify_band_inner_spin.setToolTip(
            "Pixels closer than this to the crack path are left out (the face itself, where the subset "
            "DIC is blurred by the discontinuity). 0 = keep everything up to the path.")
        self.tip_verify_band_inner_spin.valueChanged.connect(self._on_tip_verify_redraw_needed)
        tip_verify_band_row.addWidget(self.tip_verify_band_inner_spin)
        tip_verify_band_row.addWidget(QLabel("to"))
        self.tip_verify_band_outer_spin = QDoubleSpinBox()
        self.tip_verify_band_outer_spin.setRange(1.0, 500.0)
        self.tip_verify_band_outer_spin.setDecimals(1)
        self.tip_verify_band_outer_spin.setValue(10.0)
        self.tip_verify_band_outer_spin.setSuffix(" px from the crack")
        self.tip_verify_band_outer_spin.setToolTip(
            "Half-width of the band on each side of the path. A few times the DIC subset size is "
            "enough to see the local opening; the sampling offset of the jump criterion lies inside it.")
        self.tip_verify_band_outer_spin.valueChanged.connect(self._on_tip_verify_redraw_needed)
        tip_verify_band_row.addWidget(self.tip_verify_band_outer_spin)
        tip_verify_band_row.addWidget(QLabel("Normalise \u0394u"))
        self.tip_verify_norm_combo = QComboBox()
        for key, text in LOCAL_NORMALISATIONS:
            self.tip_verify_norm_combo.addItem(text, key)
        self.tip_verify_norm_combo.setCurrentIndex(1)       # \u00f7 detection threshold by default
        self.tip_verify_norm_combo.setToolTip(
            "Only for the \u0394u (opening / sliding) fields. Absolute: pixels. \u00f7 detection threshold: the "
            "'Threshold' of the opening-jump detector above, so 1 is exactly where the jump criterion calls "
            "the crack open (a white line marks it) -- the best view of how far the tip has advanced. "
            "\u00f7 each crack's own maximum: every crack scaled by the 98th percentile of its own opening this "
            "frame, to compare the SHAPE of cracks whose absolute openings differ a lot.")
        self.tip_verify_norm_combo.currentIndexChanged.connect(self._on_tip_verify_redraw_needed)
        tip_verify_band_row.addWidget(self.tip_verify_norm_combo)
        tip_verify_band_row.addStretch(1)
        tip_verify_group_layout.addLayout(tip_verify_band_row)

        tip_verify_frame_row = QHBoxLayout()
        tip_verify_frame_row.addWidget(QLabel("Frame"))
        self.tip_verify_frame_slider = QSlider(Qt.Horizontal)
        self.tip_verify_frame_slider.setEnabled(False)
        self.tip_verify_frame_slider.valueChanged.connect(self._on_tip_verify_frame_slider_changed)
        tip_verify_frame_row.addWidget(self.tip_verify_frame_slider, 1)
        self.tip_verify_frame_spin = QSpinBox()
        self.tip_verify_frame_spin.setRange(0, 0)
        self.tip_verify_frame_spin.setEnabled(False)
        self.tip_verify_frame_spin.setMinimumWidth(70)
        self.tip_verify_frame_spin.valueChanged.connect(self._on_tip_verify_frame_spin_changed)
        tip_verify_frame_row.addWidget(self.tip_verify_frame_spin)
        tip_verify_group_layout.addLayout(tip_verify_frame_row)

        tip_verify_scale_row = QHBoxLayout()
        tip_verify_scale_row.addWidget(QLabel("Color scale"))
        self.tip_verify_scale_auto_radio = QRadioButton("Auto")
        self.tip_verify_scale_fixed_radio = QRadioButton("Fixed")
        self.tip_verify_scale_auto_radio.setChecked(True)
        tip_verify_scale_group = QButtonGroup(self)
        tip_verify_scale_group.addButton(self.tip_verify_scale_auto_radio)
        tip_verify_scale_group.addButton(self.tip_verify_scale_fixed_radio)
        self.tip_verify_scale_auto_radio.toggled.connect(self._on_tip_verify_scale_mode_toggled)
        tip_verify_scale_row.addWidget(self.tip_verify_scale_auto_radio)
        tip_verify_scale_row.addWidget(self.tip_verify_scale_fixed_radio)
        self.tip_verify_scale_min_spin = QDoubleSpinBox()
        self.tip_verify_scale_min_spin.setRange(-1.0e9, 1.0e9)
        self.tip_verify_scale_min_spin.setDecimals(6)
        self.tip_verify_scale_min_spin.setEnabled(False)
        self.tip_verify_scale_min_spin.valueChanged.connect(self._on_tip_verify_scale_bounds_changed)
        tip_verify_scale_row.addWidget(self.tip_verify_scale_min_spin)
        self.tip_verify_scale_max_spin = QDoubleSpinBox()
        self.tip_verify_scale_max_spin.setRange(-1.0e9, 1.0e9)
        self.tip_verify_scale_max_spin.setDecimals(6)
        self.tip_verify_scale_max_spin.setEnabled(False)
        self.tip_verify_scale_max_spin.valueChanged.connect(self._on_tip_verify_scale_bounds_changed)
        tip_verify_scale_row.addWidget(self.tip_verify_scale_max_spin)
        self.tip_verify_scale_reset_button = QPushButton("Reset to data range")
        self.tip_verify_scale_reset_button.setEnabled(False)
        self.tip_verify_scale_reset_button.clicked.connect(self._on_tip_verify_scale_reset)
        tip_verify_scale_row.addWidget(self.tip_verify_scale_reset_button)
        tip_verify_scale_row.addStretch(1)
        tip_verify_group_layout.addLayout(tip_verify_scale_row)

        self.tip_verify_canvas = PlotCanvas(figsize=(9, 8.0))
        tip_verify_group_layout.addWidget(self.tip_verify_canvas)

        self.tip_verify_status = QLabel(
            "Load a DIC results CSV and pick a Crack/Tip above -- the standard-DIC isocontours show "
            "here automatically, live, for any frame in range."
        )
        self.tip_verify_status.setObjectName("cardSubtitle")
        self.tip_verify_status.setWordWrap(True)
        tip_verify_group_layout.addWidget(self.tip_verify_status)
        tip_verify_frame_row.addWidget(make_popout_button(
            self, "Crack Tip Detection -- verify visually", self.tip_verify_canvas, self.tip_verify_frame_slider,
            info_labels=[self.tip_verify_status]))

        profile_pick_row = QHBoxLayout()
        profile_pick_row.addWidget(QLabel(
            "Opening-jump profile along the path -- this frame (see the \"?\" button above) -- of trace"
        ))
        self.tip_verify_profile_combo = QComboBox()
        self.tip_verify_profile_combo.setToolTip(
            "Which detected trace's path the opening-jump profile below is drawn for (the profile is "
            "computed along that trace's own crack path, and its detected front at this frame is marked)."
        )
        self.tip_verify_profile_combo.currentIndexChanged.connect(self._on_tip_verify_redraw_needed)
        profile_pick_row.addWidget(self.tip_verify_profile_combo)
        profile_pick_row.addStretch(1)
        tip_verify_group_layout.addLayout(profile_pick_row)
        self.tip_verify_jump_canvas = PlotCanvas(figsize=(9, 2.6))
        tip_verify_group_layout.addWidget(self.tip_verify_jump_canvas)

        tip_verify_jump_export_row = QHBoxLayout()
        self.tip_verify_jump_export_button = QPushButton("Export this profile to CSV...")
        self.tip_verify_jump_export_button.setEnabled(False)
        self.tip_verify_jump_export_button.setToolTip(
            "Saves the opening-jump-vs-arc-length profile shown just above, for this frame only, "
            "as a CSV -- one row per sample along the traced path (distance, jump components, "
            "threshold/offset used, and whether that point is past the detected opening front)."
        )
        self.tip_verify_jump_export_button.clicked.connect(self._on_tip_verify_jump_export_clicked)
        tip_verify_jump_export_row.addWidget(self.tip_verify_jump_export_button)
        self.tip_verify_jump_export_status = QLabel("")
        self.tip_verify_jump_export_status.setObjectName("cardSubtitle")
        self.tip_verify_jump_export_status.setWordWrap(True)
        tip_verify_jump_export_row.addWidget(self.tip_verify_jump_export_status, 1)
        tip_verify_group_layout.addLayout(tip_verify_jump_export_row)

        tip_verify_group_layout.addWidget(QLabel(
            "Arc-length position vs time -- every trace ticked in \"Show trace\" above (all by "
            "default), each in its own colour"
        ))
        self.tip_verify_arc_length_canvas = PlotCanvas(figsize=(9, 3.0))
        tip_verify_group_layout.addWidget(self.tip_verify_arc_length_canvas)

        # -- smoothing of the tracked tip positions (drawn on the plot just above; optionally
        # -- also what the K_I/K_II calculations below use)
        tip_smooth_row = QHBoxLayout()
        tip_smooth_row.addWidget(QLabel("Smoothing of tip positions"))
        self.tip_smooth_method_combo = QComboBox()
        for key, text in SMOOTHING_METHODS:
            self.tip_smooth_method_combo.addItem(text, key)
        self.tip_smooth_method_combo.setToolTip(
            "Smooths the tracked tip position (arc length vs time) of every trace shown above. The "
            "smoothed curve is drawn solid over the raw points, and the local speed is taken from it. "
            "Median: rejects isolated bad detections. Local linear / quadratic regression: damps the "
            "frame-to-frame jitter while following a speed ramp / acceleration. Monotone (isotonic): "
            "the tip can only advance. The smoothed positions stay exactly ON the traced path."
        )
        self.tip_smooth_method_combo.currentIndexChanged.connect(self._on_tip_smoothing_changed)
        tip_smooth_row.addWidget(self.tip_smooth_method_combo, 1)
        tip_smooth_row.addWidget(QLabel("window"))
        self.tip_smooth_window_spin = QSpinBox()
        self.tip_smooth_window_spin.setRange(3, 99)
        self.tip_smooth_window_spin.setSingleStep(2)
        self.tip_smooth_window_spin.setValue(7)
        self.tip_smooth_window_spin.setSuffix(" frames")
        self.tip_smooth_window_spin.setToolTip(
            "How many consecutive frames the smoother looks at around each point (odd). Wider = "
            "smoother but slower to follow real changes of speed."
        )
        self.tip_smooth_window_spin.valueChanged.connect(self._on_tip_smoothing_changed)
        tip_smooth_row.addWidget(self.tip_smooth_window_spin)
        self.tip_smooth_monotonic_check = QCheckBox("never move backwards")
        self.tip_smooth_monotonic_check.setToolTip(
            "Forces the smoothed tip position to be monotone in the direction of the crack's advance "
            "(a growing crack does not heal)."
        )
        self.tip_smooth_monotonic_check.toggled.connect(self._on_tip_smoothing_changed)
        tip_smooth_row.addWidget(self.tip_smooth_monotonic_check)
        self.tip_smooth_use_for_k_check = QCheckBox("use for K_I / K_II")
        self.tip_smooth_use_for_k_check.setChecked(True)
        self.tip_smooth_use_for_k_check.setToolTip(
            "When ticked (and a smoothing is selected), the K_I/K_II calculations of the Crack analysis "
            "tab -- the direct measured-field methods AND X-FEM-DIC -- start from the SMOOTHED tip "
            "positions instead of the raw ones (the result labels then end with [smoothed]). Unticked: "
            "the smoothing is only a display aid."
        )
        self.tip_smooth_use_for_k_check.toggled.connect(self._on_tip_smoothing_changed)
        tip_smooth_row.addWidget(self.tip_smooth_use_for_k_check)
        tip_verify_group_layout.addLayout(tip_smooth_row)
        self.tip_smooth_status = QLabel("")
        self.tip_smooth_status.setObjectName("cardSubtitle")
        self.tip_smooth_status.setWordWrap(True)
        tip_verify_group_layout.addWidget(self.tip_smooth_status)

        tip_verify_arc_export_row = QHBoxLayout()
        self.tip_verify_arc_export_button = QPushButton("Export this plot to Excel / CSV...")
        self.tip_verify_arc_export_button.setToolTip(
            "Saves every trace shown just above (position + local speed vs time) as an Excel workbook "
            "with ONE SHEET PER CRACK (or as a single CSV -- pick the CSV filter). One row per point, with the same per-point position and local-speed "
            "uncertainty (when \"Estimate measurement uncertainty\" was checked) the shaded bands on "
            "that plot are drawn from, at 1 sigma regardless of the \"Uncertainty envelope\" dropdown's "
            "current N -- multiply by whatever N you need."
        )
        self.tip_verify_arc_export_button.clicked.connect(self._on_tip_verify_arc_export_clicked)
        tip_verify_arc_export_row.addWidget(self.tip_verify_arc_export_button)
        self.tip_verify_arc_export_status = QLabel("")
        self.tip_verify_arc_export_status.setObjectName("cardSubtitle")
        self.tip_verify_arc_export_status.setWordWrap(True)
        tip_verify_arc_export_row.addWidget(self.tip_verify_arc_export_status, 1)
        tip_verify_group_layout.addLayout(tip_verify_arc_export_row)

        tip_video_row = QHBoxLayout()
        tip_video_row.addWidget(QLabel("Video playback rate"))
        self.tip_video_fps_spin = QDoubleSpinBox()
        self.tip_video_fps_spin.setRange(1.0, 120.0)
        self.tip_video_fps_spin.setDecimals(1)
        self.tip_video_fps_spin.setValue(10.0)
        self.tip_video_fps_spin.setSuffix(" fps")
        self.tip_video_fps_spin.setToolTip(
            "Frames/second the exported video plays AT -- independent of the camera's own capture "
            "rate (typically far too high, often microseconds apart for a dynamic SHPB event, to "
            "watch directly). Every frame in \"Frame range\" above is rendered once, in order, at "
            "this playback rate."
        )
        tip_video_row.addWidget(self.tip_video_fps_spin)
        self.tip_video_export_button = QPushButton(
            "Export video (isocontours + live position/speed) -> .mp4"
        )
        self.tip_video_export_button.setEnabled(False)
        self.tip_video_export_button.setToolTip(
            "Every trace ticked in \"Show trace\" is drawn in the video, each in its own colour "
            "(isocontours with the trajectories, plus position / speed curves for all of them)."
        )
        self.tip_video_export_button.clicked.connect(self._on_tip_video_export_clicked)
        tip_video_row.addWidget(self.tip_video_export_button)
        tip_video_row.addStretch(1)
        tip_verify_group_layout.addLayout(tip_video_row)

        self.tip_video_progress_bar = QProgressBar()
        self.tip_video_progress_bar.setRange(0, 1)
        self.tip_video_progress_bar.setValue(0)
        self.tip_video_progress_bar.setTextVisible(True)
        tip_verify_group_layout.addWidget(self.tip_video_progress_bar)

        self.tip_video_status = QLabel(
            "Pick a trace in \"Show trace\" above, then export it as a video: the same isocontour "
            "view and position/speed plot above, rendered for every frame in \"Frame range\" and "
            "encoded to MP4."
        )
        self.tip_video_status.setObjectName("cardSubtitle")
        self.tip_video_status.setWordWrap(True)
        tip_verify_group_layout.addWidget(self.tip_video_status)

        tip_card.add(self.tip_verify_group)
        layout.addWidget(tip_card)

        self.picker.cracksChanged.connect(self._refresh_tip_crack_combo)

    # -- X-FEM-DIC sub-tab --------------------------------------------------

    def _build_xfem_dic_tab(self, layout):
        """Builds the whole "Crack analysis" sub-tab: starts with its OWN
        independent "Data" card (see `_build_data_card` -- loading a CSV/
        video/sequence from here loads the SAME `self._dataset`/
        `self._frame_source` the Wave speed analysis tab's own Data card
        does, just without having to switch tabs to do it) and its own
        "Crack Skeleton Tracing" card (moved here from the Wave speed
        analysis tab -- tracing isn't actually used by that tab's own wave-
        front picking, only by crack-tip detection/X-FEM-DIC below, both of
        which live here), then a Material & Calibration card of its own (E,
        Poisson's ratio, plane-stress/strain, PLUS its own "Pixel size"
        field -- this tab's own X-FEM-DIC K_I/K_II fit needs pixel size
        too, and this field is always kept in sync with the Wave speed
        analysis tab's own copy, see `_build_pixel_size_row` -- so nothing
        here needs a tab switch to set up), then its own "Crack Tip
        Detection" card (the automatic, ground-truth opening-jump method --
        the sole surviving tip-tracking method on this page, and this
        tab's own preparation step) -- see core/xfem_dic.py's module
        docstring for the X-FEM-DIC method itself, and this page's own
        module docstring for why raw images are needed at all here
        specifically (the tip-detection method above only needs the
        already-exported DIC field)."""
        layout.addWidget(self._build_data_card("_ca"))
        layout.addWidget(self._build_tracing_card())

        material_card = Card(
            "Material & Calibration",
            "Pixel size (same value as the Wave speed analysis tab's own Material & Calibration "
            "card -- auto-detected from the loaded CSV when it was exported with one set, or set "
            "either here or there), plus elastic modulus, Poisson's ratio and plane-stress/strain "
            "state, all needed by this tab's X-FEM-DIC K_I/K_II fit.",
        )
        material_card.add_layout(self._build_pixel_size_row("_ca"))
        self.pixel_size_status_ca = QLabel("")
        self.pixel_size_status_ca.setObjectName("cardSubtitle")
        self.pixel_size_status_ca.setWordWrap(True)
        material_card.add(self.pixel_size_status_ca)

        self.young_modulus_spin = QDoubleSpinBox()
        self.young_modulus_spin.setRange(0.001, 100000.0)
        self.young_modulus_spin.setDecimals(3)
        self.young_modulus_spin.setValue(70.0)
        self.young_modulus_spin.setSuffix(" GPa")
        material_card.add(LabeledField("Elastic modulus (E)", self.young_modulus_spin))

        self.poisson_spin = QDoubleSpinBox()
        self.poisson_spin.setRange(-0.99, 0.5)
        self.poisson_spin.setDecimals(3)
        self.poisson_spin.setSingleStep(0.01)
        self.poisson_spin.setValue(0.33)
        material_card.add(LabeledField("Poisson's ratio (ν)", self.poisson_spin))

        plane_row = QHBoxLayout()
        self.plane_stress_radio = QRadioButton("Plane stress")
        self.plane_strain_radio = QRadioButton("Plane strain")
        self.plane_strain_radio.setChecked(True)
        plane_group = QButtonGroup(self)
        plane_group.addButton(self.plane_stress_radio)
        plane_group.addButton(self.plane_strain_radio)
        plane_row.addWidget(self.plane_stress_radio)
        plane_row.addWidget(self.plane_strain_radio)
        plane_row.addStretch(1)
        material_card.add_layout(plane_row)
        layout.addWidget(material_card)

        self._build_tip_detection_card(layout)

        method_card = Card(
            "Method",
            "Directly re-correlates the reference/deformed image pair (already loaded in the "
            "\"Data\" card above) against a displacement model made of a smooth FE background "
            "field, a Heaviside jump across the crack faces, and a Williams near-tip singular term "
            "-- K_I/K_II are two of that model's own fitted parameters, not a separate post-fit.",
            help_html=(
                "<p>The subset-based correlation this app's Digital Image Correlation page uses "
                "(and that the wave-speed method on the first sub-tab ultimately depends on) tracks "
                "each point as a RIGID, translating template. A template straddling the crack face "
                "contains pixels from both sides of a real displacement discontinuity, so no rigid "
                "translation can match it once the crack has opened -- correlation there is either "
                "wrong or rejected outright, which is exactly why a standard near-tip K-fit on that "
                "already-exported field has to exclude a region right around the tip and the crack "
                "faces (r_min, flank exclusion), and can only ever re-fit the clean, far-from-the-"
                "crack points -- it cannot recover accuracy the discontinuity itself already cost. "
                "So this method re-correlates the RAW images directly, with a model that already "
                "contains the discontinuity and the near-tip singular field as first-class terms.</p>"
                "<ul>"
                "<li><b>Fixed crack path</b> -- this method does not re-detect the crack path or "
                "tip position itself; it reuses whichever tip/wave trace you tick below, exactly "
                "as tracked in the Crack Tip Detection card above or the Elastic Wave Speed card on "
                "the first sub-tab. (A version that also identifies the crack geometry from the "
                "images would be a substantially larger undertaking -- this is the scope agreed for "
                "this feature.)</li>"
                "<li><b>Simplifications</b> versus a full nodal XFEM stress-analysis code: the "
                "near-tip singular term is GLOBAL (two extra unknowns, K_I and K_II, shared by the "
                "whole mesh -- the same closed-form Williams/Irwin basis a standard near-tip K-fit "
                "uses, so the two are directly comparable) rather than enriched per node, and the "
                "crack-face jump uses the simpler unshifted Heaviside form. Both are documented, "
                "literature-grounded simplifications (close to Réthoré/Hild/Roux's \"X-DIC\" "
                "global-enrichment formulations), not ad hoc shortcuts -- see core/xfem_dic.py's "
                "module docstring for the full reasoning.</li>"
                "<li>This is a genuinely heavier computation than the tip-detection method above "
                "(an iterative Gauss-Newton solve against image grey levels, per frame) -- use "
                "\"Preview on one frame\" to tune the element size / enrichment radius cheaply "
                "before committing to a full run.</li>"
                "</ul>"
            ),
        )
        layout.addWidget(method_card)

        # -------------------------------------------------- crack path source
        source_card = Card(
            "Crack path source",
            "Tick the tip/wave trace(s) (from the Crack Tip Detection card above, or the Elastic "
            "Wave Speed card on the first sub-tab) to run X-FEM-DIC on.",
        )
        xfem_lists_header_row = QHBoxLayout()
        xfem_refresh_button = QPushButton("Refresh list")
        xfem_refresh_button.clicked.connect(self._refresh_xfem_trace_lists)
        xfem_lists_header_row.addWidget(xfem_refresh_button)
        xfem_lists_header_row.addStretch(1)
        source_card.add_layout(xfem_lists_header_row)

        xfem_lists_row = QHBoxLayout()
        xfem_tip_col = QVBoxLayout()
        xfem_tip_col.addWidget(QLabel("Crack-tip (ridge) traces"))
        self.xfem_tip_traces_list = QListWidget()
        self.xfem_tip_traces_list.setMaximumHeight(90)
        xfem_tip_col.addWidget(self.xfem_tip_traces_list)
        xfem_lists_row.addLayout(xfem_tip_col)

        xfem_wave_col = QVBoxLayout()
        xfem_wave_col.addWidget(QLabel("Elastic Wave Speed traces"))
        self.xfem_wave_traces_list = QListWidget()
        self.xfem_wave_traces_list.setMaximumHeight(90)
        xfem_wave_col.addWidget(self.xfem_wave_traces_list)
        xfem_lists_row.addLayout(xfem_wave_col)
        source_card.add_layout(xfem_lists_row)
        layout.addWidget(source_card)

        # -------------------------------------------------------- settings
        settings_card = Card(
            "Mesh & fit settings",
            "The background FE mesh, the Heaviside/singular enrichment, and the Gauss-Newton "
            "solve's sampling density.",
            help_html=(
                "<p></p><ul>"
                "<li><b>Element size</b>: the background mesh's element size, centered on each "
                "frame's tip position. Needs to be coarse enough that ordinary nodal motion can't "
                "locally mimic the near-tip singular field itself (which would make K_I/K_II "
                "statistically unidentifiable from the background) -- a few tens of pixels is a "
                "reasonable start.</li>"
                "<li><b>Enrichment radius</b>: how far from the tip the mesh itself is refined/"
                "enriched (see \"Graded mesh\" below) -- also this fit's own r_max UNLESS you set "
                "r_max below to something else.</li>"
                "<li><b>r_min / r_max</b>: the genuine equivalent, for this global grey-level fit, "
                "of a standard near-tip K-fit's own r_min/r_max search-ring radii -- confines the "
                "Williams K_I/K_II singular term to an annulus around the tip, the same physical "
                "reasoning as that ring fit (too close and DIC/correlation noise right at the tip "
                "dominates; too far and the field departs from the leading-order 1/√r term this fit "
                "assumes), but applied as a smooth WEIGHT on the singular basis functions rather "
                "than a hard include/exclude on sample points -- this fit solves the background, "
                "crack-face jump and singular term all AT ONCE, so an abrupt on/off boundary would "
                "inject a discontinuity into the residual every time the tip moves a fraction of a "
                "pixel between frames, destabilizing convergence across a time series. r_min=0 "
                "(default) excludes nothing near the tip; r_max=0 (default) reuses the enrichment "
                "radius above as the outer radius -- set an explicit r_max to decouple the fit's own "
                "near-tip confinement from the enrichment radius, which independently also drives the "
                "mesh's own refinement geometry and so can't always be freely tuned for the fit alone. "
                "Unlike a ring fit, there is no separate flank-exclusion angle here: the crack-face "
                "jump is already handled by its own (Heaviside) term in this fit, not by excluding "
                "an angular wedge from the singular one.</li>"
                "<li><b>Max iterations</b>: the Gauss-Newton iteration budget for this fit's "
                "nonlinear solve -- 15 by default. A frame that stops WITHOUT converging reports "
                "an under-relaxed K_I/K_II, partway "
                "between its starting guess and its true value by an amount that varies frame to "
                "frame -- a common cause of K(t) looking jagged/unstable even when the underlying "
                "crack is propagating smoothly. If your K(t) is noisy, "
                "first check whether frames are reporting \"did not converge\" (per-frame status "
                "below); if so, raise this before suspecting the physics or the mesh. Raising it "
                "only costs extra time on frames that actually need the extra iterations.</li>"
                "<li><b>Position smoothing window</b>: off (0) by default -- the raw, "
                "frame-by-frame tip/wave-tracked tip position is used as this fit's origin. "
                "The same option, for the same reason, as a standard near-tip K-fit's own "
                "\"Position smoothing window\": this fit's near-tip term is exactly as sensitive to "
                "origin jitter as a ring fit's, but XFEM-DIC does NOT automatically inherit that "
                "smoothing from upstream (the tip/wave trace's own \"Smooth tip position\" option, "
                "if used, is baked in when the trace is added -- this is a separate, additional "
                "smoothing pass applied only to this fit's own origin) -- so raw frame-to-frame "
                "noise reaches this fit's origin unfiltered unless a window is also set here. Try "
                "this (in px of arc length) if K(t) is still jagged after ruling out non-convergence "
                "above.</li>"
                "<li><b>Region half-width</b>: how far the background mesh itself extends from the "
                "tip in every direction -- needs to comfortably cover the enrichment radius plus "
                "some far-field margin for the background/rigid-motion terms to be well "
                "resolved.</li>"
                "<li><b>Sample spacing</b>: the pixel grid spacing the grey-level residual is "
                "evaluated on -- 1-2 px is normally plenty; a larger value trades accuracy for "
                "speed.</li>"
                "<li><b>Warm-start from DIC field</b>: seeds the background field's nodal motion "
                "from the already-exported subset-DIC u/v (interpolated onto the mesh nodes) "
                "instead of zero -- substantially speeds up and stabilizes convergence, since only "
                "the crack-discontinuity/near-tip terms then have to be found essentially from "
                "scratch.</li>"
                "<li><b>\"Suggest settings\"</b> pre-fills the three mesh values from the ticked "
                "trace's own tracked crack length and (when loaded) the reference image's "
                "dimensions -- a starting point, not a guarantee; always check it with \"Preview "
                "mesh\"/\"Preview on one frame\" below and adjust by hand if it looks off for your "
                "specimen.</li>"
                "<li><b>Graded mesh</b>: off by default (every element the same \"Element size\", "
                "as above). Ticked, \"Element size\" instead means the NEAR-TIP resolution only -- "
                "elements grow geometrically (3x by the mesh's own outer edge) once past the "
                "enrichment radius, so the far field (which doesn't need fine resolution) uses "
                "fewer, bigger elements. Better near-tip detail for the same overall region, and a "
                "smoother (not blocky/staircase) far-field isocontour below, at no extra "
                "correlation cost -- try this first if the isocontour comparison looks coarse or "
                "blocky away from the tip.</li>"
                "</ul>"
            ),
        )
        xfem_suggest_row = QHBoxLayout()
        self.xfem_suggest_button = QPushButton("Suggest settings from ticked trace")
        self.xfem_suggest_button.clicked.connect(self._on_xfem_suggest_settings_clicked)
        xfem_suggest_row.addWidget(self.xfem_suggest_button)
        xfem_suggest_row.addStretch(1)
        settings_card.add_layout(xfem_suggest_row)

        xfem_settings_row1 = QHBoxLayout()
        xfem_settings_row1.addWidget(QLabel("Element size"))
        self.xfem_element_size_spin = QDoubleSpinBox()
        self.xfem_element_size_spin.setRange(2.0, 500.0)
        self.xfem_element_size_spin.setValue(20.0)
        self.xfem_element_size_spin.setSuffix(" px")
        xfem_settings_row1.addWidget(self.xfem_element_size_spin)
        xfem_settings_row1.addWidget(QLabel("Enrichment radius"))
        self.xfem_r_enrich_spin = QDoubleSpinBox()
        self.xfem_r_enrich_spin.setRange(2.0, 2000.0)
        self.xfem_r_enrich_spin.setValue(40.0)
        self.xfem_r_enrich_spin.setSuffix(" px")
        xfem_settings_row1.addWidget(self.xfem_r_enrich_spin)
        xfem_settings_row1.addWidget(QLabel("Region half-width"))
        self.xfem_roi_half_width_spin = QDoubleSpinBox()
        self.xfem_roi_half_width_spin.setRange(10.0, 4000.0)
        self.xfem_roi_half_width_spin.setValue(160.0)
        self.xfem_roi_half_width_spin.setSuffix(" px")
        xfem_settings_row1.addWidget(self.xfem_roi_half_width_spin)
        xfem_settings_row1.addStretch(1)
        settings_card.add_layout(xfem_settings_row1)

        xfem_settings_row1b = QHBoxLayout()
        xfem_settings_row1b.addWidget(QLabel("r_min"))
        self.xfem_r_min_spin = QDoubleSpinBox()
        self.xfem_r_min_spin.setRange(0.0, 2000.0)
        self.xfem_r_min_spin.setValue(0.0)
        self.xfem_r_min_spin.setSuffix(" px (0 = off)")
        xfem_settings_row1b.addWidget(self.xfem_r_min_spin)
        xfem_settings_row1b.addWidget(QLabel("r_max"))
        self.xfem_r_max_spin = QDoubleSpinBox()
        self.xfem_r_max_spin.setRange(0.0, 2000.0)
        self.xfem_r_max_spin.setValue(0.0)
        self.xfem_r_max_spin.setSuffix(" px (0 = same as enrichment radius)")
        xfem_settings_row1b.addWidget(self.xfem_r_max_spin)
        xfem_settings_row1b.addWidget(QLabel("Max iterations"))
        self.xfem_max_iterations_spin = QSpinBox()
        self.xfem_max_iterations_spin.setRange(5, 500)
        self.xfem_max_iterations_spin.setValue(15)
        self.xfem_max_iterations_spin.setToolTip(
            "Gauss-Newton iteration budget for this fit's nonlinear solve (background + crack-face "
            "jump + singular K_I/K_II term, all at once) -- 15 by default. A frame that stops "
            "WITHOUT converging (see the per-frame \"converged\"/iteration count) has an "
            "under-relaxed K_I/K_II that sits partway between its starting guess and its true "
            "value by an amount that varies frame to frame -- a leading cause of frame-to-frame "
            "K(t) scatter that otherwise looks like measurement noise. Raising this only costs "
            "extra time on frames that actually need it; an already-converged frame still stops "
            "early."
        )
        xfem_settings_row1b.addWidget(self.xfem_max_iterations_spin)
        xfem_settings_row1b.addWidget(QLabel("Position smoothing window"))
        self.xfem_position_window_spin = QDoubleSpinBox()
        self.xfem_position_window_spin.setRange(0.0, 2000.0)
        self.xfem_position_window_spin.setValue(0.0)
        self.xfem_position_window_spin.setSuffix(" px (0 = off)")
        self.xfem_position_window_spin.setToolTip(
            "Off (0) by default -- the raw, frame-by-frame tip/wave-tracked tip position is used. "
            "This fit's near-tip singular term is sensitive to origin jitter -- the tip "
            "trace's own \"Smooth tip position\" option, if used, is baked in when the trace is "
            "added, but that's a separate, upstream smoothing pass; set a window here too (in px of "
            "arc length) if K(t) still looks jagged."
        )
        xfem_settings_row1b.addWidget(self.xfem_position_window_spin)
        xfem_settings_row1b.addStretch(1)
        settings_card.add_layout(xfem_settings_row1b)

        xfem_settings_row2 = QHBoxLayout()
        xfem_settings_row2.addWidget(QLabel("Sample spacing"))
        self.xfem_sample_step_spin = QDoubleSpinBox()
        self.xfem_sample_step_spin.setRange(0.5, 20.0)
        self.xfem_sample_step_spin.setValue(2.0)
        self.xfem_sample_step_spin.setSuffix(" px")
        self.xfem_sample_step_spin.setToolTip(
            "Spacing of the grid of points where the fit itself compares reference vs. deformed "
            "grey levels every Gauss-Newton iteration -- a DIFFERENT grid from the mesh above (it "
            "ignores element boundaries entirely), and it directly sets how many real constraints "
            "feed the solve. Too coarse starves the SMALLEST elements in play (the near-tip ones on "
            "a graded mesh) of samples -- their corner nodes end up under-constrained and may get "
            "flagged unreliable (blank in the isocontours). Too fine adds little beyond a real "
            "speckle pattern's own correlation length (typically a few px): neighboring samples just "
            "become increasingly redundant, not more informative, while cost grows as "
            "1/spacing^2 per iteration -- and, with \"Estimate measurement uncertainty\" on, can "
            "understate K_I/K_II's reported stderr, whose degrees-of-freedom count assumes "
            "independent residuals. \"Suggest settings\" below proposes a starting value (about 1/8 "
            "of the smallest element edge, floored at 1 px) from the current mesh settings -- lower "
            "it from there only as long as K_I/K_II keep actually changing, not by default."
        )
        xfem_settings_row2.addWidget(self.xfem_sample_step_spin)
        self.xfem_warm_start_check = QCheckBox("Warm-start background field from the loaded DIC field")
        self.xfem_warm_start_check.setChecked(True)
        xfem_settings_row2.addWidget(self.xfem_warm_start_check)
        xfem_settings_row2.addStretch(1)
        settings_card.add_layout(xfem_settings_row2)
        self.xfem_graded_mesh_check = QCheckBox(
            "Graded mesh (fine near the tip, coarser away -- see \"?\" above)"
        )
        settings_card.add(self.xfem_graded_mesh_check)
        self.xfem_estimate_uncertainty_check = QCheckBox("Estimate measurement uncertainty (K_I/K_II)")
        self.xfem_estimate_uncertainty_check.setChecked(False)
        self.xfem_estimate_uncertainty_check.setToolTip(
            "Off by default -- extra cost on top of an already-heavy fit. When checked, every fit "
            "(and, with \"Also compute with an auto-detected tip position\" below, the winning "
            "self-found position too, refit once more -- not every candidate the search tries) also "
            "reports a formal 1-sigma K_I/K_II uncertainty from the Gauss-Newton fit's own asymptotic "
            "covariance (the standard way to report a nonlinear least-squares fit's parameter "
            "uncertainty -- see core.xfem_dic.fit_xfem_dic_frame's docstring). Shown as a shaded band "
            "on the K_I/K_II plot below (\"Uncertainty envelope\") and included in its CSV export."
        )
        settings_card.add(self.xfem_estimate_uncertainty_check)

        self.xfem_k_from_field_check = QCheckBox(
            "K_I/K_II from the fitted field (ring fit, same as Direct K) -- recommended")
        self.xfem_k_from_field_check.setChecked(True)
        self.xfem_k_from_field_check.setToolTip(
            "When checked, the grey-level fit estimates ONLY the displacement field (background + crack-"
            "face jump) and K_I/K_II are then obtained by a Williams ring fit on that fitted field, "
            "exactly like the Direct K method (same window, same tip offset, same T-stress/affine options).\n\n"
            "Why: with K_I/K_II as unknowns of the grey-level fit, the free nodal field can reproduce almost "
            "all of the sqrt(r) near-tip field (only ~30 % of the K signal is independent of it on a 14 px "
            "mesh, ~7 % on 7 px -- a finer mesh makes it WORSE) and that signal is only ~0.01 px at 20 px from "
            "the tip, so K came out with arbitrary sign and size (e.g. -50..+50 MPa*sqrt(m) when the direct "
            "method gives 3). On a synthetic test with known K_I = 1.8, the joint fit gave -0.4..+3 (biased "
            "even without noise) and this option gives 1.8 +- 0.1.\n\n"
            "When checked, the K window below replaces r_min/r_max above (the singular term is not fitted)."
        )
        settings_card.add(self.xfem_k_from_field_check)
        xfem_k_field_row = QHBoxLayout()
        xfem_k_field_row.addWidget(QLabel("K method"))
        self.xfem_k_field_method_combo = QComboBox()
        for _key, _label in (("ring_affine", "Ring + uniform gradient (robust to background strain)"),
                             ("ring", "Ring (K_I, K_II, rigid)"),
                             ("ring_t", "Ring + T-stress")):
            self.xfem_k_field_method_combo.addItem(_label, _key)
        xfem_k_field_row.addWidget(self.xfem_k_field_method_combo)
        xfem_k_field_row.addWidget(QLabel("K window r_min / r_max"))
        self.xfem_k_rmin_spin = QDoubleSpinBox()
        self.xfem_k_rmin_spin.setRange(1.0, 2000.0)
        self.xfem_k_rmin_spin.setValue(20.0)
        self.xfem_k_rmin_spin.setSuffix(" px")
        xfem_k_field_row.addWidget(self.xfem_k_rmin_spin)
        self.xfem_k_rmax_spin = QDoubleSpinBox()
        self.xfem_k_rmax_spin.setRange(2.0, 2000.0)
        self.xfem_k_rmax_spin.setValue(60.0)
        self.xfem_k_rmax_spin.setSuffix(" px")
        self.xfem_k_rmax_spin.setToolTip("Must stay inside the region half-width above (minus the tip offset).")
        xfem_k_field_row.addWidget(self.xfem_k_rmax_spin)
        xfem_k_field_row.addWidget(QLabel("Tip offset"))
        self.xfem_k_offset_spin = QDoubleSpinBox()
        self.xfem_k_offset_spin.setRange(-200.0, 200.0)
        self.xfem_k_offset_spin.setDecimals(1)
        self.xfem_k_offset_spin.setSingleStep(0.5)
        self.xfem_k_offset_spin.setValue(0.0)
        self.xfem_k_offset_spin.setSuffix(" px (+ ahead)")
        self.xfem_k_offset_spin.setToolTip(
            "Moves the origin of the K ring fit along the crack tangent (same meaning as the Direct K "
            "card's tip offset: the opening-jump tip lags the true tip by about the subset size).")
        xfem_k_field_row.addWidget(self.xfem_k_offset_spin)
        xfem_k_field_row.addStretch(1)
        settings_card.add_layout(xfem_k_field_row)
        layout.addWidget(settings_card)

        # ------------------------------------------------- auto-detect tip
        auto_tip_card = Card(
            "Auto-detected tip position (optional)",
            "By default X-FEM-DIC reuses each frame's tip position exactly as tracked on the "
            "first sub-tab (the crack PATH always does -- this method never re-detects the "
            "skeleton itself). Tick this to ALSO have it search, frame by frame, for the "
            "best-fitting tip position along that same path -- an independent measurement to "
            "compare against the fixed-trace one, computed alongside it (not instead of it).",
            help_html=(
                "<p>At each frame, instead of trusting the ridge/wave trace's own tip arc length, "
                "this searches candidate positions along the SAME already-traced crack path, within "
                "the search radius below of THAT SAME frame's own ridge/wave-tracked position -- "
                "every frame is anchored independently to its own already-tracked position, never to "
                "the previous frame's self-found result, so one frame settling on a wrong candidate "
                "cannot drag every later frame off with it -- and keeps whichever CONVERGED candidate "
                "gives the lowest grey-level correlation residual (an under-converged candidate is "
                "never preferred just for reporting a lower residual, even if the grid search happens "
                "to try one) -- i.e. it uses the fit's own quality as the signal for \"is the crack "
                "geometry right here\", since a wrong tip position misclassifies which mesh elements "
                "the already-opened crack actually splits.</p>"
                "<p>The crack is also never allowed to shrink: each frame's accepted position can't "
                "move backward (in arc length, in whichever direction the trace overall grows) from "
                "the last frame this search actually accepted a position for. A candidate that would "
                "shrink the crack is excluded from the search outright, not just disfavoured, so it "
                "can never win just because its correlation residual happens to look good.</p>"
                "<p><b>Cost</b>: one full Gauss-Newton solve per candidate position tested (roughly "
                "2*radius/step + 1 of them) -- genuinely several times slower per frame than the "
                "fixed-trace fit. <b>Caveat</b>: this is a plain grid search over a residual that "
                "isn't guaranteed to have a single, sharp minimum -- it can settle near the wrong "
                "position on any one frame, especially with noisy images or a coarse search step "
                "(a finer step or a wider radius reduces, but doesn't eliminate, that risk). Treat it "
                "as a second, independent opinion to compare against the fixed-trace curve below, not "
                "as a more-accurate replacement for it -- if the fixed-trace curve (e.g. an elastic "
                "wave-derived trace) already tracks the tip well, there is usually no need to also "
                "tick this.</p>"
                "<p><b>Two search methods</b> are offered (choose below):</p>"
                "<p><i>Residual search</i> (default, described above): tries many candidate "
                "positions and keeps whichever converged fit has the lowest grey-level correlation "
                "residual. A handful of pixels of precision, at the cost of one full solve per "
                "candidate (roughly 2*radius/step + 1 of them).</p>"
                "<p><i>Heaviside jump search</i>: fits ONE deliberately over-covering trial position "
                "(pushed further out than the trace, doubling that push and retrying a few times if "
                "the crack still looks open at the trial's own edge), then reads the refined tip "
                "straight off that fit's own crack-face opening amplitude -- locating where it drops "
                "below the jump threshold below. Typically only 1-2 full solves per frame instead of "
                "many, but a partially-enriched element smears that opening/closing transition over "
                "roughly its own width, so the position found is only accurate to within about one to "
                "two mesh element widths -- coarser than the residual search. Also depends on the "
                "crack opening enough, relative to the element size, to be resolved at all. Prefer "
                "this when processing many frames and a coarser, much cheaper per-frame position is "
                "an acceptable trade; prefer the residual search when per-frame precision matters "
                "most.</p>"
                "<p><b>Diagnosing a plateaued auto-detected track</b>: because every frame is "
                "anchored to THAT frame's own nominal position (never chained -- see above), a "
                "systematic, persistent offset between the nominal track and what the images "
                "actually support has to be re-bridged by the search radius on EVERY frame. Too "
                "small a radius shows up unmistakably on the \"Arc length along path vs time\" plot "
                "below: the auto-detected curve visibly flattens out (pinned by the \"crack cannot "
                "shrink\" floor) while the nominal (fixed-trace) curve keeps climbing past it. Use "
                "\"Suggest search radius from ticked trace\" below to pre-fill a radius scaled from "
                "this trace's own observed arc-length spread, and re-run -- see that button's own "
                "tooltip and <code>core.xfem_dic.recommend_tip_search_radius_px</code> for the exact "
                "reasoning.</p>"
                "<p>If the plateau PERSISTS even after a generously widened radius -- especially "
                "alongside the Heaviside-jump method finding almost no usable position at all across "
                "the sequence -- that stops being a search-radius problem: it means the correlation "
                "residual genuinely, consistently prefers a shorter crack than the nominal track "
                "claims, frame after frame, i.e. the images themselves don't support the nominal "
                "track's position. A likely cause, if the fixed trace is the WAVE method: elastic-"
                "wave-speed extrapolation is an indirect proxy for crack length, not a direct visual "
                "detection, and can run ahead of where the crack has genuinely, physically separated "
                "-- especially if a fracture process zone (elevated strain ahead of the true open "
                "crack face) is what the standard subset-DIC panel's colour map is actually showing "
                "in that region, rather than a real displacement discontinuity. In that case the "
                "auto-detected (correlation-grounded) position may be the more trustworthy one, not "
                "the fixed trace -- worth cross-checking against a RIDGE-based trace of the same "
                "crack, if available, rather than only raising the search radius further.</p>"
            ),
        )
        self.xfem_auto_tip_check = QCheckBox(
            "Also compute with an auto-detected tip position, alongside the fixed-trace result"
        )
        auto_tip_card.add(self.xfem_auto_tip_check)
        xfem_auto_tip_mode_row = QHBoxLayout()
        xfem_auto_tip_mode_row.addWidget(QLabel("Search method"))
        self.xfem_tip_mode_residual_radio = QRadioButton("Residual (precise, slower)")
        self.xfem_tip_mode_residual_radio.setChecked(True)
        self.xfem_tip_mode_heaviside_radio = QRadioButton("Heaviside jump (cheap, coarser)")
        xfem_tip_mode_group = QButtonGroup(self)
        xfem_tip_mode_group.addButton(self.xfem_tip_mode_residual_radio)
        xfem_tip_mode_group.addButton(self.xfem_tip_mode_heaviside_radio)
        xfem_auto_tip_mode_row.addWidget(self.xfem_tip_mode_residual_radio)
        xfem_auto_tip_mode_row.addWidget(self.xfem_tip_mode_heaviside_radio)
        xfem_auto_tip_mode_row.addStretch(1)
        auto_tip_card.add_layout(xfem_auto_tip_mode_row)
        xfem_auto_tip_row = QHBoxLayout()
        xfem_auto_tip_row.addWidget(QLabel("Search radius"))
        self.xfem_tip_search_radius_spin = QDoubleSpinBox()
        self.xfem_tip_search_radius_spin.setRange(2.0, 2000.0)
        self.xfem_tip_search_radius_spin.setValue(40.0)
        self.xfem_tip_search_radius_spin.setSuffix(" px")
        xfem_auto_tip_row.addWidget(self.xfem_tip_search_radius_spin)
        xfem_auto_tip_row.addWidget(QLabel("Search step"))
        self.xfem_tip_search_step_spin = QDoubleSpinBox()
        self.xfem_tip_search_step_spin.setRange(0.5, 200.0)
        self.xfem_tip_search_step_spin.setValue(5.0)
        self.xfem_tip_search_step_spin.setSuffix(" px")
        xfem_auto_tip_row.addWidget(self.xfem_tip_search_step_spin)
        xfem_auto_tip_row.addWidget(QLabel("Jump threshold"))
        self.xfem_tip_jump_threshold_spin = QDoubleSpinBox()
        self.xfem_tip_jump_threshold_spin.setRange(0.01, 50.0)
        self.xfem_tip_jump_threshold_spin.setValue(0.15)
        self.xfem_tip_jump_threshold_spin.setDecimals(2)
        self.xfem_tip_jump_threshold_spin.setSuffix(" px")
        self.xfem_tip_jump_threshold_spin.setToolTip(
            "Heaviside jump search only: the fitted crack-face opening amplitude (in px) below "
            "which a mesh node is considered \"closed\"."
        )
        xfem_auto_tip_row.addWidget(self.xfem_tip_jump_threshold_spin)
        xfem_auto_tip_row.addStretch(1)
        auto_tip_card.add_layout(xfem_auto_tip_row)
        xfem_suggest_radius_row = QHBoxLayout()
        self.xfem_suggest_radius_button = QPushButton("Suggest search radius from ticked trace")
        self.xfem_suggest_radius_button.setToolTip(
            "Pre-fills \"Search radius\" from 40% of the ticked trace's OWN observed arc-length "
            "spread (max - min of its tracked positions) -- see core.xfem_dic."
            "recommend_tip_search_radius_px's own docstring for the full reasoning. Because this "
            "search re-anchors to each frame's own nominal position independently (never chaining "
            "off its own previous result, to avoid a different drift bug), a persistent offset "
            "between the nominal track and what the images actually support has to be re-bridged "
            "EVERY frame -- too small a radius shows up as the auto-detected arc-length-vs-time "
            "curve visibly plateauing while the nominal one keeps climbing. Only a pre-fill of the "
            "still-editable spin box above, never applied silently."
        )
        self.xfem_suggest_radius_button.clicked.connect(self._on_xfem_suggest_search_radius_clicked)
        xfem_suggest_radius_row.addWidget(self.xfem_suggest_radius_button)
        xfem_suggest_radius_row.addStretch(1)
        auto_tip_card.add_layout(xfem_suggest_radius_row)
        layout.addWidget(auto_tip_card)

        # ------------------------------------------------------- mesh preview
        mesh_preview_card = Card(
            "Mesh preview",
            "The Q4 background mesh over the reference image, at the currently ticked trace's "
            "preview frame -- shaded elements are Heaviside/crack-split (enriched), the dashed "
            "circle is the near-tip enrichment radius. Costs nothing to run (no correlation, just "
            "geometry) -- use it to sanity-check the settings above before \"Preview on one "
            "frame\"/\"Run full sequence\".",
        )
        xfem_mesh_preview_row = QHBoxLayout()
        self.xfem_mesh_preview_button = QPushButton("Preview mesh")
        self.xfem_mesh_preview_button.clicked.connect(self._on_xfem_mesh_preview_clicked)
        xfem_mesh_preview_row.addWidget(self.xfem_mesh_preview_button)
        xfem_mesh_preview_row.addStretch(1)
        mesh_preview_card.add_layout(xfem_mesh_preview_row)

        self.xfem_mesh_preview_status = QLabel(
            "Tick a trace above and load the matching image sequence in the Data card, then click "
            "\"Preview mesh\"."
        )
        self.xfem_mesh_preview_status.setObjectName("cardSubtitle")
        self.xfem_mesh_preview_status.setWordWrap(True)
        mesh_preview_card.add(self.xfem_mesh_preview_status)

        self.xfem_mesh_canvas = PlotCanvas(figsize=(6, 6))
        mesh_preview_card.add(self.xfem_mesh_canvas)
        layout.addWidget(mesh_preview_card)

        # -------------------------------------------------------------- run
        run_card = Card(
            "Run",
            "\"Preview on one frame\" is fast and safe for tuning the settings above; \"Run full "
            "sequence\" fits every frame the FIRST ticked trace has a tracked tip at (runs off the "
            "UI thread -- this can take a while for a long sequence).",
        )
        xfem_preview_row = QHBoxLayout()
        xfem_preview_row.addWidget(QLabel("Preview frame"))
        self.xfem_preview_frame_spin = QSpinBox()
        self.xfem_preview_frame_spin.setRange(0, 0)
        xfem_preview_row.addWidget(self.xfem_preview_frame_spin)
        self.xfem_preview_button = QPushButton("Preview on one frame")
        self.xfem_preview_button.clicked.connect(self._on_xfem_preview_clicked)
        xfem_preview_row.addWidget(self.xfem_preview_button)
        self.xfem_run_button = QPushButton("Run full sequence")
        self.xfem_run_button.setObjectName("primary")
        self.xfem_run_button.clicked.connect(self._on_xfem_run_clicked)
        xfem_preview_row.addWidget(self.xfem_run_button)
        self.xfem_stop_button = QPushButton("Stop")
        self.xfem_stop_button.setEnabled(False)
        self.xfem_stop_button.setToolTip(
            "Stops the run after the frame (or, with auto-detect-tip on, the candidate fit) "
            "currently in progress finishes -- traces already computed this run are discarded, "
            "not kept as a partial result."
        )
        self.xfem_stop_button.clicked.connect(self._on_xfem_stop_clicked)
        xfem_preview_row.addWidget(self.xfem_stop_button)
        xfem_preview_row.addStretch(1)
        run_card.add_layout(xfem_preview_row)

        self.xfem_progress_bar = QProgressBar()
        self.xfem_progress_bar.setMaximum(1)
        self.xfem_progress_bar.setValue(0)
        run_card.add(self.xfem_progress_bar)

        self.xfem_run_status = QLabel(
            "Tick at least one trace above (Crack Tip Detection card above, or Elastic Wave "
            "Speed card on the Wave speed analysis tab), then Preview or Run."
        )
        self.xfem_run_status.setObjectName("cardSubtitle")
        self.xfem_run_status.setWordWrap(True)
        run_card.add(self.xfem_run_status)
        layout.addWidget(run_card)

        # --------------------------------------------- direct K (no X-FEM-DIC)
        direct_card = Card(
            "Direct K_I / K_II (measured field, no X-FEM-DIC)",
            "K_I(t)/K_II(t) straight from the already-exported subset-DIC displacement field around "
            "each tracked tip position -- no image re-correlation, so it is fast (the whole sequence "
            "in seconds). Uses the trace(s) ticked in \"Crack path source\" above. When a smoothing is "
            "selected under the \"Arc-length position vs time\" plot of the Crack Tip Detection card "
            "and \"use for K_I / K_II\" is ticked there, the fit is centred on the SMOOTHED tip "
            "positions. Results land in the same Results card below as X-FEM-DIC (curves can be "
            "compared on the same plot).",
            help_html=(
                "<ul>"
                "<li><b>Williams ring fit</b>: least-squares fit of the near-tip Williams/Irwin mode "
                "I + II displacement field (plus an unknown rigid-body offset) to the measured u, v "
                "sampled on an annulus r_min..r_max around the tip, in the crack's local frame. Gives "
                "a formal 1-sigma uncertainty and an R&sup2; per frame.</li>"
                "<li><b>Williams ring fit + T-stress</b>: the same with the first non-singular term "
                "(T-stress) added, so it does not leak into K_I/K_II.</li>"
                "<li><b>Interaction integral</b>: energy-based domain integral (Moran &amp; Shih form) "
                "on the same annulus -- structurally different from the ring fit, so agreement between "
                "the two is a genuine cross-check. Needs r_min &gt; 0. No formal uncertainty; the "
                "inner-half / outer-half values are reported in the status text instead.</li>"
                "<li><b>r_min / r_max</b>: too close to the tip, DIC decorrelation and the crack "
                "opening corrupt the field; too far, the field departs from the leading-order "
                "1/&radic;r term. <b>Flank exclusion</b>: half-angle (deg) of the wedge along the crack "
                "faces left out of the fit. <b>Tangent window</b>: arc length over which the traced "
                "path is smoothed to get the local crack direction (0 = automatic, 2 x r_max).</li>"
                "<li><b>Ring fit + uniform strain &amp; rotation</b>: adds the full uniform displacement gradient "
                "(4 more unknowns) around the tip. At DIC fit radii the K field is tiny (~0.01-0.05 px "
                "for K ~ 1-3 MPa&radic;m) and a background strain of only 1e-3 -- wave field, "
                "neighbouring cracks, loading -- gives a displacement of the same size over 1 mm; a "
                "uniform opening strain has exactly the symmetry of K_I and is otherwise read as K_I "
                "(a synthetic test gives K_I x3 for 1e-3). If K_I drops a lot with this method, the "
                "plain ring fit was biased by the background. Its scatter is larger (more unknowns): "
                "use the position smoothing and a not-too-small r_max.</li>"
                "<li><b>Ignore frames after the crack stops</b>: frames after the tip reached its final "
                "position are dropped (not plotted, not in the critical-K mean).</li>"
                "<li><b>Tip offset</b>: moves the fit origin along the crack (+ = ahead). The "
                "opening-jump tip sits BEHIND the true tip; an origin behind the true tip overestimates "
                "K, more so for small r_max (K falls as r_max grows). <b>Scan tip offset</b> tabulates the "
                "median K_I versus offset and r_max on the first ticked trace; the offset where all "
                "r_max columns agree (smallest spread) is the tip bias, and K there is the corrected value.</li>"
                "<li>A jittery tip position moves the fit's origin and swings K far more than it moves "
                "the tip itself -- that is what the position smoothing is for.</li>"
                "</ul>"
            ),
        )
        direct_row1 = QHBoxLayout()
        direct_row1.addWidget(QLabel("Method"))
        self.direct_k_method_combo = QComboBox()
        for key, text in DIRECT_METHODS:
            self.direct_k_method_combo.addItem(text, key)
        direct_row1.addWidget(self.direct_k_method_combo, 1)
        direct_row1.addWidget(QLabel("r_min"))
        self.direct_k_rmin_spin = QDoubleSpinBox()
        self.direct_k_rmin_spin.setRange(0.0, 5000.0)
        self.direct_k_rmin_spin.setDecimals(1)
        self.direct_k_rmin_spin.setValue(6.0)
        self.direct_k_rmin_spin.setSuffix(" px")
        direct_row1.addWidget(self.direct_k_rmin_spin)
        direct_row1.addWidget(QLabel("r_max"))
        self.direct_k_rmax_spin = QDoubleSpinBox()
        self.direct_k_rmax_spin.setRange(1.0, 5000.0)
        self.direct_k_rmax_spin.setDecimals(1)
        self.direct_k_rmax_spin.setValue(40.0)
        self.direct_k_rmax_spin.setSuffix(" px")
        direct_row1.addWidget(self.direct_k_rmax_spin)
        direct_card.add_layout(direct_row1)

        direct_row2 = QHBoxLayout()
        direct_row2.addWidget(QLabel("Flank exclusion"))
        self.direct_k_flank_spin = QDoubleSpinBox()
        self.direct_k_flank_spin.setRange(0.0, 80.0)
        self.direct_k_flank_spin.setDecimals(1)
        self.direct_k_flank_spin.setValue(15.0)
        self.direct_k_flank_spin.setSuffix(" deg")
        direct_row2.addWidget(self.direct_k_flank_spin)
        direct_row2.addWidget(QLabel("Tangent window"))
        self.direct_k_tangent_spin = QDoubleSpinBox()
        self.direct_k_tangent_spin.setRange(0.0, 5000.0)
        self.direct_k_tangent_spin.setDecimals(1)
        self.direct_k_tangent_spin.setValue(0.0)
        self.direct_k_tangent_spin.setSpecialValueText("auto (2 x r_max)")
        self.direct_k_tangent_spin.setSuffix(" px")
        direct_row2.addWidget(self.direct_k_tangent_spin)
        self.direct_k_trim_check = QCheckBox("Ignore frames after the crack stops")
        self.direct_k_trim_check.setChecked(True)
        self.direct_k_trim_check.setToolTip(
            "Once the tip has reached its final position the crack is arrested: those frames are neither "
            "plotted nor included in the critical-K mean (they pile up as a vertical column of points "
            "at the end of each trace).")
        direct_row2.addWidget(self.direct_k_trim_check)
        direct_row2.addWidget(QLabel("Tip offset"))
        self.direct_k_offset_spin = QDoubleSpinBox()
        self.direct_k_offset_spin.setRange(-200.0, 200.0)
        self.direct_k_offset_spin.setDecimals(1)
        self.direct_k_offset_spin.setSingleStep(0.5)
        self.direct_k_offset_spin.setValue(0.0)
        self.direct_k_offset_spin.setSuffix(" px (+ ahead)")
        direct_row2.addWidget(self.direct_k_offset_spin)
        self.direct_k_scan_button = QPushButton("Scan tip offset")
        self.direct_k_scan_button.clicked.connect(self._on_direct_k_scan_clicked)
        direct_row2.addWidget(self.direct_k_scan_button)
        self.direct_k_window_button = QPushButton("Scan r_min / r_max")
        self.direct_k_window_button.setToolTip(
            "Median K_I for a grid of fitting annuli at the CURRENT tip offset and method. A K-dominated "
            "zone is a block of the table where K stops changing; K that keeps drifting as r_min grows "
            "means the zone near the tip is blurred by the DIC (subset size / grid step) -- raise r_min.")
        self.direct_k_window_button.clicked.connect(self._on_direct_k_window_scan_clicked)
        direct_row2.addWidget(self.direct_k_window_button)
        self.direct_k_run_button = QPushButton("Compute K_I / K_II (direct)")
        self.direct_k_run_button.setObjectName("primary")
        self.direct_k_run_button.clicked.connect(self._on_direct_k_run_clicked)
        direct_row2.addWidget(self.direct_k_run_button)
        direct_row2.addStretch(1)
        direct_card.add_layout(direct_row2)

        self.direct_k_status = QLabel(
            "Tick at least one trace in \"Crack path source\" above, then compute. Results appear in "
            "the Results card below."
        )
        self.direct_k_status.setObjectName("cardSubtitle")
        self.direct_k_status.setWordWrap(True)
        direct_card.add(self.direct_k_status)
        self.direct_k_scan_text = QPlainTextEdit()
        self.direct_k_scan_text.setReadOnly(True)
        self.direct_k_scan_text.setMaximumHeight(170)
        self.direct_k_scan_text.setStyleSheet("font-family: monospace;")
        self.direct_k_scan_text.setPlaceholderText(
            "Scan result: median K_I (MPa.sqrt(m)) per tip offset (rows) and r_max (columns).")
        direct_card.add(self.direct_k_scan_text)
        direct_scan_row = QHBoxLayout()
        self.direct_k_apply_best_button = QPushButton("Apply suggested offset")
        self.direct_k_apply_best_button.setEnabled(False)
        self.direct_k_apply_best_button.clicked.connect(self._on_direct_k_apply_best_clicked)
        direct_scan_row.addWidget(self.direct_k_apply_best_button)
        direct_scan_row.addStretch(1)
        direct_card.add_layout(direct_scan_row)
        self._direct_k_best_offset = None
        layout.addWidget(direct_card)

        # ----------------------------------------------------------- results
        results_card = Card(
            "Results",
            "K_I(distance)/K_II(distance) from the X-FEM-DIC fit, plus arc length along path vs time "
            "below it (same traces, same tick list) -- untick a trace in \"Show curves\" to hide it "
            "from BOTH plots at once when comparing several traces gets crowded. \"K_I\"/\"K_II\" "
            "below control which curve TYPE(s) are drawn (both are always computed either way); the "
            "plot keeps the axes' origin visible so a curve's sign is always readable against it -- "
            "K_I <= 0 has no physical meaning, unlike K_II. For an auto-detected trace, a GAP in the "
            "arc-length-vs-time curve means the search found nothing usable that frame (same frames "
            "show up as missing K_I/K_II above, for the same reason) -- it does not silently fall "
            "back to drawing the untracked fixed-trace position instead.\n\n"
            "Curves from the \"Direct K_I / K_II\" card (measured displacement field: Williams ring "
            "fit, ring fit + T-stress, or interaction integral) and from X-FEM-DIC are drawn together "
            "here, so the two routes can be compared; a direct result's name ends with "
            "[direct: ...]. The field isocontours and the video below need an X-FEM-DIC trace.",
        )
        xfem_critical_row = QHBoxLayout()
        xfem_critical_row.addWidget(QLabel("Show trace"))
        self.xfem_critical_combo = QComboBox()
        self.xfem_critical_combo.currentIndexChanged.connect(self._on_xfem_critical_combo_changed)
        xfem_critical_row.addWidget(self.xfem_critical_combo)
        xfem_critical_row.addWidget(QLabel("Frame"))
        self.xfem_frame_slider = QSlider(Qt.Horizontal)
        self.xfem_frame_slider.setEnabled(False)
        self.xfem_frame_slider.valueChanged.connect(self._on_xfem_frame_slider_changed)
        xfem_critical_row.addWidget(self.xfem_frame_slider, 1)
        self.xfem_frame_spin = QSpinBox()
        self.xfem_frame_spin.setRange(0, 0)
        self.xfem_frame_spin.setEnabled(False)
        self.xfem_frame_spin.setMinimumWidth(70)
        self.xfem_frame_spin.valueChanged.connect(self._on_xfem_frame_spin_changed)
        xfem_critical_row.addWidget(self.xfem_frame_spin)
        results_card.add_layout(xfem_critical_row)

        xfem_curve_visibility_row = QHBoxLayout()
        xfem_curve_visibility_row.addWidget(QLabel("Show curves"))
        self.xfem_result_traces_list = QListWidget()
        self.xfem_result_traces_list.setMaximumHeight(70)
        self.xfem_result_traces_list.itemChanged.connect(self._refresh_xfem_display)
        xfem_curve_visibility_row.addWidget(self.xfem_result_traces_list, 1)
        results_card.add_layout(xfem_curve_visibility_row)

        # Which curve TYPE(s) to draw -- both K_I and K_II are always
        # computed either way (see the fit itself), this just controls
        # what's shown.
        xfem_show_row = QHBoxLayout()
        self.xfem_show_KI_check = QCheckBox("K_I")
        self.xfem_show_KI_check.setChecked(True)
        self.xfem_show_KI_check.toggled.connect(self._refresh_xfem_display)
        xfem_show_row.addWidget(self.xfem_show_KI_check)
        self.xfem_show_KII_check = QCheckBox("K_II")
        self.xfem_show_KII_check.setChecked(True)
        self.xfem_show_KII_check.toggled.connect(self._refresh_xfem_display)
        xfem_show_row.addWidget(self.xfem_show_KII_check)
        xfem_show_row.addWidget(QLabel("Uncertainty envelope"))
        self.xfem_uncertainty_sigma_combo = QComboBox()
        self.xfem_uncertainty_sigma_combo.addItem("Off", 0)
        self.xfem_uncertainty_sigma_combo.addItem("+/- 1 sigma", 1)
        self.xfem_uncertainty_sigma_combo.addItem("+/- 2 sigma", 2)
        self.xfem_uncertainty_sigma_combo.addItem("+/- 3 sigma", 3)
        self.xfem_uncertainty_sigma_combo.setCurrentIndex(1)
        self.xfem_uncertainty_sigma_combo.setToolTip(
            "Shades a +/- N sigma band around the K_I/K_II curve when \"Estimate measurement "
            "uncertainty\" was checked for that run (see the Settings card) -- no band otherwise."
        )
        self.xfem_uncertainty_sigma_combo.currentIndexChanged.connect(self._refresh_xfem_display)
        xfem_show_row.addWidget(self.xfem_uncertainty_sigma_combo)
        xfem_show_row.addStretch(1)
        results_card.add_layout(xfem_show_row)

        self.xfem_status = QLabel("No X-FEM-DIC results yet.")
        self.xfem_status.setObjectName("cardSubtitle")
        self.xfem_status.setWordWrap(True)
        results_card.add(self.xfem_status)

        self.xfem_canvas = PlotCanvas(figsize=(9, 4.5))
        results_card.add(self.xfem_canvas)
        xfem_export_row = QHBoxLayout()
        self.xfem_export_k_csv_button = QPushButton("Export K_I/K_II CSV...")
        self.xfem_export_k_csv_button.setToolTip(
            "One CSV, long format -- every ticked/computed trace's per-frame K_I/K_II from BOTH "
            "methods (X-FEM-DIC and the standard-DIC reference), plus their formal uncertainties "
            "when estimated, not just whatever's currently visible on the plot above."
        )
        self.xfem_export_k_csv_button.clicked.connect(self._on_xfem_export_k_csv_clicked)
        xfem_export_row.addWidget(self.xfem_export_k_csv_button)
        self.xfem_export_k_csv_status = QLabel("")
        self.xfem_export_k_csv_status.setObjectName("cardSubtitle")
        self.xfem_export_k_csv_status.setWordWrap(True)
        xfem_export_row.addWidget(self.xfem_export_k_csv_status, 1)
        results_card.add_layout(xfem_export_row)

        results_card.add(QLabel("Arc length along path vs time (all traces, same tick list above)"))
        self.xfem_arc_length_canvas = PlotCanvas(figsize=(9, 3.5))
        results_card.add(self.xfem_arc_length_canvas)

        results_card.add(QLabel("K_I versus local tip speed (all traces, same tick list above)"))
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed window"))
        self.xfem_speed_window_spin = QSpinBox()
        self.xfem_speed_window_spin.setRange(3, 41)
        self.xfem_speed_window_spin.setSingleStep(2)
        self.xfem_speed_window_spin.setValue(5)
        self.xfem_speed_window_spin.setSuffix(" frames")
        self.xfem_speed_window_spin.setToolTip(
            "Speed = slope of a straight line fitted to the tip position over this many frames "
            "centred on each frame (use the position smoothing above first: a raw, jittery "
            "position gives a very noisy speed).")
        self.xfem_speed_window_spin.valueChanged.connect(self._refresh_xfem_display)
        speed_row.addWidget(self.xfem_speed_window_spin)
        speed_row.addWidget(QLabel("Density"))
        self.xfem_density_spin = QDoubleSpinBox()
        self.xfem_density_spin.setRange(0.0, 30000.0)
        self.xfem_density_spin.setDecimals(0)
        self.xfem_density_spin.setValue(2500.0)
        self.xfem_density_spin.setSuffix(" kg/m3")
        self.xfem_density_spin.setSpecialValueText("off")
        self.xfem_density_spin.setToolTip(
            "Only used for the top axis (speed as a fraction of the Rayleigh wave speed, from E, "
            "nu and this density). 0 hides that axis. 2500 kg/m3 is a typical glass: set yours.")
        self.xfem_density_spin.valueChanged.connect(self._refresh_xfem_display)
        speed_row.addWidget(self.xfem_density_spin)
        speed_row.addStretch(1)
        results_card.add_layout(speed_row)
        self.xfem_speed_status = QLabel("")
        self.xfem_speed_status.setObjectName("cardSubtitle")
        self.xfem_speed_status.setWordWrap(True)
        results_card.add(self.xfem_speed_status)
        self.xfem_speed_canvas = PlotCanvas(figsize=(9, 3.8))
        results_card.add(self.xfem_speed_canvas)
        layout.addWidget(results_card)

        # --------------------------------------------------- field isocontours
        isocontours_card = Card(
            "Field isocontours",
            "The fitted X-FEM-DIC displacement/strain field, at the trace/frame selected above, "
            "re-evaluated on a regular grid and drawn as FILLED isocontours side by side with the "
            "already-exported subset-DIC field (same frame, same field, same color scale) -- a "
            "direct visual check of where the two methods agree and where they don't (expect the "
            "clearest difference right around the crack faces/tip, where the standard DIC excludes "
            "or distorts, and X-FEM-DIC is specifically built to handle it).",
            help_html=(
                "<p>Both panels show the SAME frame (the one selected above by \"Show trace\" / "
                "\"Frame\") and the SAME color scale, so the two are directly comparable at a glance: "
                "left is the standard subset-DIC field already loaded in the Data card, exactly as "
                "exported; right is the X-FEM-DIC fit's own field, reconstructed from its solved "
                "background+Heaviside+singular model (core.xfem_dic.evaluate_xfem_dic_field_grid) -- "
                "not re-run, just re-evaluated from the already-fitted result.</p>"
                "<p>u/v (displacement) are the most direct comparison -- both methods fit them "
                "straight from the images. exx/eyy/exy/equivalent strain are then derived from each "
                "method's own u/v by the same finite-difference formula "
                "(core/dic_processing.py's `_grid_strain` convention), so they're comparable too, "
                "but inherit whatever noise/error differentiation amplifies -- expect this to look "
                "noisier than the displacement comparison, especially very close to the singular "
                "tip term.</p>"
                "<p>Once shown, this card stays live: moving the Frame slider/spin (either the copy "
                "above, in Results, or the copy right below, on this card), changing \"Show trace\" "
                "(either copy), or changing the Field below all redraw it automatically -- \"Show "
                "isocontours\" only needs a second click after changing the sample-spacing setting "
                "(which also controls this grid's resolution) or after a fresh \"Run\"/\"Preview\".</p>"
                "<p>Only drawn for a frame that actually has a usable X-FEM-DIC fit (converged or "
                "not, as long as K_I/K_II were found) -- a failed frame has no field to show.</p>"
                "<p>The white crack overlay in both panels is TWO different lines: a thin dotted one "
                "is the static path geometry of the drawn ROI polygon (fixed once drawn, for context "
                "only); a bold dashed one is the actual per-frame tip trajectory -- this second line "
                "is what changes shape when \"Smooth tip position\"/\"Smooth wave position\" was "
                "checked at \"Add to list\" time (Crack Tip Detection card above, or Elastic Wave "
                "Speed card on the first sub-tab), since that option replaces the stored trace's "
                "positions with the smoothed ones. It has no effect "
                "unless the trace was (re-)added with the checkbox on -- toggling it afterwards "
                "does not retroactively change an already-added trace.</p>"
                "<p>The right panel is CROPPED to the standard subset-DIC's own correlated ROI (the "
                "left panel's own extent) whenever a standard-DIC field is available to compare "
                "against: the X-FEM-DIC mesh itself generally reaches further out (a square region "
                "around the tip), so without this, it would keep showing a computed value -- often "
                "large, right where the near-tip singular term is strongest -- exactly where the "
                "standard DIC panel has no data at all, reading as a spurious disagreement between "
                "the two methods rather than \"nothing to compare here\". This only crops what's "
                "DRAWN in this comparison -- it doesn't change the X-FEM-DIC fit or its K_I/K_II.</p>"
                "<p>Below each isocontour panel is that SAME method's own crack-opening-jump profile "
                "along the traced path -- the same diagnostic shown in the Crack Tip Detection card's "
                "\"Verify visually\" section, computed here TWICE: once from the standard subset-DIC's "
                "own raw u/v fields (left), once from the X-FEM-DIC fit's reconstructed u/v (right) -- "
                "so a real disagreement between the two methods shows up directly as a difference in "
                "where each one's detected opening front lands, not just in the color maps above. Uses "
                "the same Offset/Threshold/Gap tolerance settings as that card's \"Detect tip position\" "
                "controls.</p>"
            ),
        )
        # Mirrors of the canonical "Show trace"/"Frame" controls up in the
        # Results card, duplicated here so switching trace/frame while
        # looking at the isocontours doesn't need scrolling back up --
        # kept in sync both ways with xfem_critical_combo/xfem_frame_
        # slider/xfem_frame_spin by _sync_xfem_isocontour_controls (see
        # its docstring); never a second source of truth.
        xfem_isocontour_trace_row = QHBoxLayout()
        xfem_isocontour_trace_row.addWidget(QLabel("Show trace"))
        self.xfem_isocontour_trace_combo = QComboBox()
        self.xfem_isocontour_trace_combo.currentIndexChanged.connect(
            self._on_xfem_isocontour_trace_combo_changed
        )
        xfem_isocontour_trace_row.addWidget(self.xfem_isocontour_trace_combo)
        xfem_isocontour_trace_row.addWidget(QLabel("Frame"))
        self.xfem_isocontour_frame_slider = QSlider(Qt.Horizontal)
        self.xfem_isocontour_frame_slider.setEnabled(False)
        self.xfem_isocontour_frame_slider.valueChanged.connect(
            self._on_xfem_isocontour_frame_slider_changed
        )
        xfem_isocontour_trace_row.addWidget(self.xfem_isocontour_frame_slider, 1)
        self.xfem_isocontour_frame_spin = QSpinBox()
        self.xfem_isocontour_frame_spin.setRange(0, 0)
        self.xfem_isocontour_frame_spin.setEnabled(False)
        self.xfem_isocontour_frame_spin.setMinimumWidth(70)
        self.xfem_isocontour_frame_spin.valueChanged.connect(
            self._on_xfem_isocontour_frame_spin_changed
        )
        xfem_isocontour_trace_row.addWidget(self.xfem_isocontour_frame_spin)
        isocontours_card.add_layout(xfem_isocontour_trace_row)

        xfem_isocontour_row = QHBoxLayout()
        xfem_isocontour_row.addWidget(QLabel("Field"))
        self.xfem_field_combo = QComboBox()
        for key, label in XFEM_ISOCONTOUR_FIELD_CHOICES:
            self.xfem_field_combo.addItem(label, key)
        self.xfem_field_combo.currentIndexChanged.connect(self._on_xfem_field_combo_changed)
        xfem_isocontour_row.addWidget(self.xfem_field_combo)
        self.xfem_isocontour_button = QPushButton("Show isocontours")
        self.xfem_isocontour_button.clicked.connect(self._on_xfem_isocontour_clicked)
        xfem_isocontour_row.addWidget(self.xfem_isocontour_button)
        xfem_isocontour_row.addStretch(1)
        isocontours_card.add_layout(xfem_isocontour_row)

        # Same "Color scale (Auto/Fixed)" shape as the wave-speed/crack-tip
        # field cards' own color-scale controls (see e.g.
        # _on_wave_scale_mode_toggled) -- here it applies to BOTH panels at
        # once, since the whole point is comparing them under one scale.
        xfem_isocontour_scale_row = QHBoxLayout()
        xfem_isocontour_scale_row.addWidget(QLabel("Color scale (both panels)"))
        self.xfem_isocontour_scale_auto_radio = QRadioButton("Auto")
        self.xfem_isocontour_scale_fixed_radio = QRadioButton("Fixed")
        self.xfem_isocontour_scale_auto_radio.setChecked(True)
        xfem_isocontour_scale_group = QButtonGroup(self)
        xfem_isocontour_scale_group.addButton(self.xfem_isocontour_scale_auto_radio)
        xfem_isocontour_scale_group.addButton(self.xfem_isocontour_scale_fixed_radio)
        self.xfem_isocontour_scale_auto_radio.toggled.connect(self._on_xfem_isocontour_scale_mode_toggled)
        xfem_isocontour_scale_row.addWidget(self.xfem_isocontour_scale_auto_radio)
        xfem_isocontour_scale_row.addWidget(self.xfem_isocontour_scale_fixed_radio)
        self.xfem_isocontour_scale_min_spin = QDoubleSpinBox()
        self.xfem_isocontour_scale_min_spin.setRange(-1.0e9, 1.0e9)
        self.xfem_isocontour_scale_min_spin.setDecimals(6)
        self.xfem_isocontour_scale_min_spin.setEnabled(False)
        self.xfem_isocontour_scale_min_spin.valueChanged.connect(self._on_xfem_isocontour_scale_bounds_changed)
        xfem_isocontour_scale_row.addWidget(self.xfem_isocontour_scale_min_spin)
        self.xfem_isocontour_scale_max_spin = QDoubleSpinBox()
        self.xfem_isocontour_scale_max_spin.setRange(-1.0e9, 1.0e9)
        self.xfem_isocontour_scale_max_spin.setDecimals(6)
        self.xfem_isocontour_scale_max_spin.setEnabled(False)
        self.xfem_isocontour_scale_max_spin.valueChanged.connect(self._on_xfem_isocontour_scale_bounds_changed)
        xfem_isocontour_scale_row.addWidget(self.xfem_isocontour_scale_max_spin)
        self.xfem_isocontour_scale_reset_button = QPushButton("Reset to data range")
        self.xfem_isocontour_scale_reset_button.setEnabled(False)
        self.xfem_isocontour_scale_reset_button.clicked.connect(self._on_xfem_isocontour_scale_reset)
        xfem_isocontour_scale_row.addWidget(self.xfem_isocontour_scale_reset_button)
        xfem_isocontour_scale_row.addStretch(1)
        isocontours_card.add_layout(xfem_isocontour_scale_row)

        self.xfem_isocontour_status = QLabel(
            "Compute results above, pick the trace/frame with \"Show trace\"/\"Frame\", then click "
            "\"Show isocontours\"."
        )
        self.xfem_isocontour_status.setObjectName("cardSubtitle")
        self.xfem_isocontour_status.setWordWrap(True)
        isocontours_card.add(self.xfem_isocontour_status)

        self.xfem_isocontour_canvas = PlotCanvas(figsize=(12, 8.5))
        xfem_isocontour_trace_row.addWidget(make_popout_button(
            self, "X-FEM-DIC vs DIC isocontours", self.xfem_isocontour_canvas, self.xfem_isocontour_frame_slider))
        isocontours_card.add(self.xfem_isocontour_canvas)

        # -- video export: the same two isocontour panels above, plus a
        # live K_I/K_II-and-speed-vs-distance panel (see "Results" card's
        # own K_I/K_II plot, here against distance instead of time),
        # walking every frame of the trace picked by "Show trace" above
        # -- core.xfem_video_export.export_xfem_video does the actual
        # work off the UI thread (_XfemVideoExportWorker), so this just
        # gathers the fixed, whole-video color/axis scales and starts it.
        xfem_video_row = QHBoxLayout()
        xfem_video_row.addWidget(QLabel("Video playback rate"))
        self.xfem_video_fps_spin = QDoubleSpinBox()
        self.xfem_video_fps_spin.setRange(1.0, 120.0)
        self.xfem_video_fps_spin.setDecimals(1)
        self.xfem_video_fps_spin.setValue(5.0)
        self.xfem_video_fps_spin.setSuffix(" fps")
        self.xfem_video_fps_spin.setToolTip(
            "Playback rate of the exported video -- independent of the camera's own capture rate."
        )
        xfem_video_row.addWidget(self.xfem_video_fps_spin)
        self.xfem_video_export_button = QPushButton(
            "Export video (DIC + X-FEM isocontours + live K/speed) -> .mp4"
        )
        self.xfem_video_export_button.setEnabled(False)
        self.xfem_video_export_button.clicked.connect(self._on_xfem_video_export_clicked)
        xfem_video_row.addWidget(self.xfem_video_export_button)
        xfem_video_row.addStretch(1)
        isocontours_card.add_layout(xfem_video_row)

        self.xfem_video_progress_bar = QProgressBar()
        self.xfem_video_progress_bar.setRange(0, 1)
        self.xfem_video_progress_bar.setValue(0)
        self.xfem_video_progress_bar.setTextVisible(True)
        isocontours_card.add(self.xfem_video_progress_bar)

        self.xfem_video_status = QLabel(
            "Compute results above, pick a trace with \"Show trace\", then export it as a video: "
            "both isocontour panels (same as above) plus K_I (and K_II, if ticked below) and crack "
            "speed, all against distance travelled along the path, growing frame by frame."
        )
        self.xfem_video_status.setObjectName("cardSubtitle")
        self.xfem_video_status.setWordWrap(True)
        isocontours_card.add(self.xfem_video_status)

        layout.addWidget(isocontours_card)

    # -- data loading -----------------------------------------------------

    def _on_csv_selected(self, path: str):
        try:
            dataset = load_dic_export_csv(path)
        except CrackWaveLoadError as exc:
            for suffix in self._data_card_suffixes:
                getattr(self, f"csv_status{suffix}").setText(str(exc))
            return
        self._dataset = dataset
        csv_status_text = (
            f"Loaded {len(dataset.frames)} frame(s), {dataset.n_points_x} x {dataset.n_points_y} "
            f"grid, from {os.path.basename(path)}"
        )
        for suffix in self._data_card_suffixes:
            getattr(self, f"csv_status{suffix}").setText(csv_status_text)
            getattr(self, f"csv_row{suffix}").path_edit.setText(path)
        self._apply_pixel_size_from_dataset(dataset)
        n_frames = len(dataset.frames)
        self.frame_slider.setEnabled(n_frames > 0)
        self.frame_spin.setEnabled(n_frames > 0)
        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(0, n_frames - 1))
        self.frame_slider.setValue(0)
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(max(0, n_frames - 1))
        self.frame_spin.setValue(0)
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)
        self.picker.clear_all()
        self._refresh_tracing_frame()

        self.wave_start_spin.setRange(0, max(0, n_frames - 1))
        self.wave_end_spin.setRange(0, max(0, n_frames - 1))
        self.wave_start_spin.setValue(0)
        self.wave_end_spin.setValue(max(0, n_frames - 1))
        self.wave_speed_canvas.clear()
        self.wave_speed_canvas.refresh()
        self.wave_speed_status.setText("Load data above, then click \"Show field\".")
        self._last_wave_profile = None
        self._last_wave_direction = None
        self._last_wave_field_key = None
        self._last_wave_signal_for_scale = None
        self.wave_scale_auto_radio.setChecked(True)
        self._deactivate_roi_selector()
        self._roi_waves = []
        self.roi_waves_list.clear()
        self.roi_label_edit.clear()
        self.roi_status.setText(
            "Click \"Show field\" above first to show the field, then \"Draw ROI polygon\" and "
            "trace a loose loop around one visible band."
        )

        self.tip_start_spin.setRange(0, max(0, n_frames - 1))
        self.tip_end_spin.setRange(0, max(0, n_frames - 1))
        self.tip_start_spin.setValue(0)
        self.tip_end_spin.setValue(max(0, n_frames - 1))
        self._tip_roi_traces = []
        self._tip_roi_path_geometry = []
        self._refresh_tip_roi_list()
        self._refresh_tip_crack_combo()
        self.tip_roi_status.setText(
            "Select a crack path above, then click \"Detect tip position (all frames) -> Add to "
            "list\"."
        )

        self._xfem_results = []
        self._xfem_skipped_traces = []
        self._refresh_xfem_trace_lists()
        self.xfem_critical_combo.clear()
        self.xfem_frame_spin.blockSignals(True)
        self.xfem_frame_spin.setRange(0, 0)
        self.xfem_frame_spin.blockSignals(False)
        self.xfem_frame_spin.setEnabled(False)
        self.xfem_frame_slider.blockSignals(True)
        self.xfem_frame_slider.setRange(0, 0)
        self.xfem_frame_slider.blockSignals(False)
        self.xfem_frame_slider.setEnabled(False)
        self._sync_xfem_isocontour_controls()
        self.xfem_preview_frame_spin.setRange(0, 0)
        self.xfem_result_traces_list.clear()
        self.xfem_canvas.clear()
        self.xfem_canvas.refresh()
        self.xfem_arc_length_canvas.clear()
        self.xfem_arc_length_canvas.refresh()
        self.xfem_speed_canvas.clear()
        self.xfem_speed_canvas.refresh()
        self.xfem_speed_status.setText("")
        self.xfem_run_status.setText(
            "Tick at least one trace above (Crack Tip Detection card above, or Elastic Wave "
            "Speed card on the Wave speed analysis tab), then Preview or Run."
        )
        self.xfem_mesh_canvas.clear()
        self.xfem_mesh_canvas.refresh()
        self.xfem_mesh_preview_status.setText(
            "Tick a trace above and load the matching image sequence in the Data card, then click "
            "\"Preview mesh\"."
        )
        self._xfem_isocontour_shown = False
        self._xfem_isocontour_signal_for_scale = None
        self.xfem_isocontour_scale_auto_radio.setChecked(True)
        self.xfem_isocontour_canvas.clear()
        self.xfem_isocontour_canvas.refresh()
        self.xfem_isocontour_status.setText(
            "Compute results above, pick the trace/frame with \"Show trace\"/\"Frame\", then click "
            "\"Show isocontours\"."
        )

    def _on_any_source_mode_toggled(self, video_radio_widget):
        """Fires from ANY built Data card's own "Video file" radio (see
        `_build_data_card` -- connected with the specific radio bound in
        as a default lambda argument, not `self.sender()`, since this
        page's own lightweight PySide6 test stub doesn't track a signal's
        sender; only the video_radio of each pair is connected here, same
        as this page's original single-instance behavior -- the paired
        sequence_radio always toggles oppositely via its shared
        QButtonGroup, so this still fires on every real mode change).
        Reads which mode the widget that fired just switched to and
        mirrors it onto every other built instance, so there is only one
        source mode for the whole page no matter which card's radios the
        user actually clicked."""
        is_video = bool(video_radio_widget.isChecked())
        for suffix in self._data_card_suffixes:
            video_radio = getattr(self, f"video_radio{suffix}")
            sequence_radio = getattr(self, f"sequence_radio{suffix}")
            video_row = getattr(self, f"video_row{suffix}")
            seq_row_widget = getattr(self, f"seq_row_widget{suffix}")
            video_radio.blockSignals(True)
            sequence_radio.blockSignals(True)
            video_radio.setChecked(is_video)
            sequence_radio.setChecked(not is_video)
            video_radio.blockSignals(False)
            sequence_radio.blockSignals(False)
            video_row.setVisible(is_video)
            seq_row_widget.setVisible(not is_video)

    def _on_browse_sequence_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select the image sequence folder")
        if not folder:
            return
        for suffix in self._data_card_suffixes:
            getattr(self, f"seq_folder_edit{suffix}").setText(folder)
        self._load_frame_source(folder)

    def _on_video_file_selected(self, path: str):
        for suffix in self._data_card_suffixes:
            getattr(self, f"video_row{suffix}").path_edit.setText(path)
        self._load_frame_source(path)

    def _load_frame_source(self, path: str):
        if self._frame_source is not None:
            self._frame_source.close()
            self._frame_source = None
        try:
            source = open_frame_source(path)
        except (ValueError, OSError) as exc:
            for suffix in self._data_card_suffixes:
                getattr(self, f"frames_status{suffix}").setText(f"Could not load: {exc}")
            return
        if source.n_frames == 0:
            for suffix in self._data_card_suffixes:
                getattr(self, f"frames_status{suffix}").setText("No frames found.")
            return
        self._frame_source = source
        fps_text = f", {source.fps:.6g} fps" if source.fps else ""
        frames_status_text = (
            f"Loaded {source.n_frames} frames{fps_text} from {os.path.basename(path.rstrip('/'))}"
        )
        for suffix in self._data_card_suffixes:
            getattr(self, f"frames_status{suffix}").setText(frames_status_text)
        self._refresh_tracing_frame()

    # -- tracing background ---------------------------------------------------

    def _on_frame_slider_changed(self, value: int):
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(value)
        self.frame_spin.blockSignals(False)
        self._refresh_tracing_frame()

    def _on_frame_spin_changed(self, value: int):
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(value)
        self.frame_slider.blockSignals(False)
        self._refresh_tracing_frame()

    def _refresh_tracing_frame(self, *_args):
        if self._dataset is None or not self._dataset.frames:
            return
        pos = max(0, min(len(self._dataset.frames) - 1, self.frame_slider.value()))
        dic_frame = self._dataset.frames[pos]
        field_index = max(0, self.field_combo.currentIndex())
        field_key, field_label = FIELD_CHOICES[field_index]
        field_values = dic_frame.field(field_key)

        background = None
        source_note = ""
        if self._frame_source is not None:
            try:
                raw = self._frame_source.get_frame(dic_frame.frame_index)
                background = to_display_uint8(raw)
            except (ValueError, IndexError) as exc:
                source_note = f"  (could not load background frame {dic_frame.frame_index}: {exc})"

        info = f"Frame index {dic_frame.frame_index} ({pos + 1} / {len(self._dataset.frames)})"
        if dic_frame.time_s is not None:
            info += f"  |  t = {dic_frame.time_s:.6g} s"
        info += source_note
        self.frame_info_label.setText(info)

        # This frame's own median (u, v) -- its dominant, roughly rigid-
        # body displacement since the reference frame -- passed to the
        # picker so it can re-align the (raw, current-frame) background
        # image with the field/crack overlay, which stay at the fixed
        # reference (x0, y0) grid every frame. See
        # ui/crack_skeleton_picker.py's module docstring ("Background/
        # field alignment") for why this is needed: without it, a
        # specimen that has moved noticeably since the reference frame
        # (common in SHPB/Direct-Impact tests) makes the field/traced
        # crack visibly drift away from where the crack actually is in
        # the background image -- not a bug in the DIC computation itself
        # (u/v/exx are still correct per material point either way), just
        # this tool overlaying a fixed-reference-frame field on a moving
        # background.
        u_shift = float(np.nanmedian(dic_frame.u)) if dic_frame.u is not None and np.any(np.isfinite(dic_frame.u)) else 0.0
        v_shift = float(np.nanmedian(dic_frame.v)) if dic_frame.v is not None and np.any(np.isfinite(dic_frame.v)) else 0.0

        self.picker.set_frame(background, field_values, self._dataset.xs, self._dataset.ys, field_label,
                               displacement_shift_xy=(u_shift, v_shift))

    # -- elastic wave speed ---------------------------------------------------

    def _y_axis_unit_and_scale(self):
        """(unit_label, scale) for this page's time-series plots' Y axis
        (a spatial px quantity: wave-front position, crack-tip arc
        length) -- mm (scale = pixel size) when a pixel size is set on
        the Material & Calibration card above, else px (scale = 1.0)
        unchanged. Multiply every plotted px value by `scale` and label
        the axis with `unit_label`."""
        pixel_size = self.pixel_size_spin.value()
        if pixel_size:
            return "mm", pixel_size
        return "px", 1.0

    # -- background color scale (Auto/Fixed) -- shared by the manual and ---
    # -- automatic wave plots below, mirrors ui/crack_skeleton_picker.py's -
    # -- "_on_scale_*" / "_current_vmin_vmax" for the tracing panel's field -

    def _on_wave_scale_mode_toggled(self, _checked=None):
        fixed = self.wave_scale_fixed_radio.isChecked()
        self.wave_scale_min_spin.setEnabled(fixed)
        self.wave_scale_max_spin.setEnabled(fixed)
        self.wave_scale_reset_button.setEnabled(fixed)
        if fixed and self.wave_scale_min_spin.value() == self.wave_scale_max_spin.value():
            self._on_wave_scale_reset()
        self._redraw_wave_plot()

    def _on_wave_scale_bounds_changed(self, _value=None):
        self._redraw_wave_plot()

    def _on_wave_scale_reset(self):
        signal = self._last_wave_signal_for_scale
        if signal is None:
            return
        finite = signal[np.isfinite(signal)]
        if finite.size == 0:
            return
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        self.wave_scale_min_spin.blockSignals(True)
        self.wave_scale_max_spin.blockSignals(True)
        self.wave_scale_min_spin.setValue(vmin)
        self.wave_scale_max_spin.setValue(vmax)
        self.wave_scale_min_spin.blockSignals(False)
        self.wave_scale_max_spin.blockSignals(False)
        self._redraw_wave_plot()

    def _current_wave_scale_vmin_vmax(self):
        if self.wave_scale_fixed_radio.isChecked():
            return self.wave_scale_min_spin.value(), self.wave_scale_max_spin.value()
        return None, None

    def _redraw_wave_plot(self, _value=None):
        """Re-draws the background field shown by "Show field", from the
        stored profile -- so toggling the color scale, or adding/removing
        a ROI-picked wave, updates the plot without forcing a re-fetch of
        the field itself."""
        if self._last_wave_profile is not None:
            self._draw_field_plot(self._last_wave_profile, self._last_wave_direction, self._last_wave_field_key)

    def _on_sync_pixel_size(self):
        """Reads the pixel size (mm/px) already calibrated on the Digital
        Image Correlation page's Calibration card (`project.dic.
        pixel_size_mm`) instead of making the user re-enter/re-measure a
        value that page already has -- this tool shares the `Project`
        instance (`self.project`) with every other page even though it
        does not persist its own settings into it (see the module
        docstring), so reading that one already-known value across is a
        plain attribute read, not a new coupling.

        Connected identically from either tab's own "Sync from DIC tab"
        button (see `_build_pixel_size_row`) -- `self.pixel_size_spin.
        setValue` below mirrors to every other instance on its own (see
        `_on_pixel_size_changed`), and the confirmation/error message is
        shown on EVERY instance's own status label so it's visible
        whichever tab the user actually clicked from, not just the Wave
        speed analysis tab's."""
        value = getattr(self.project.dic, "pixel_size_mm", None) if self.project is not None else None
        if value is None:
            message = (
                "No pixel size is set on the Digital Image Correlation page (Calibration card) -- "
                "set it there first, or enter it directly here."
            )
        else:
            self.pixel_size_spin.setValue(float(value))
            message = f"Pixel size synced from the DIC tab: {float(value):.6g} mm/px."
        self.wave_speed_status.setText(message)
        for suffix in self._pixel_size_spin_suffixes:
            label = getattr(self, f"pixel_size_status{suffix}", None)
            if label is not None:
                label.setText(message)

    def _on_show_field_clicked(self):
        """Draws just the background field (no arrival-threshold dots, no
        fit line -- detection is entirely ROI polygon picking, see the
        card's help text) so there's something to draw a ROI polygon on.
        The threshold passed to `build_arrival_time_profile` is a fixed
        internal placeholder: this page exposes no arrival threshold
        control, and the resulting `arrival_times` array isn't used or
        drawn here at all -- only `profile.signal`/`times`/`positions`
        (the heat map itself) are."""
        if self._dataset is None or not self._dataset.frames:
            self.wave_speed_status.setText("Load a DIC results CSV above first.")
            return
        field_key = self.wave_field_combo.currentData()
        direction = "x" if self.wave_x_radio.isChecked() else "y"
        rate_override = self.wave_rate_spin.value() or None
        frame_range = (self.wave_start_spin.value(), self.wave_end_spin.value())

        try:
            profile = build_arrival_time_profile(
                self._dataset, field_key, direction, threshold_fraction=0.5,
                baseline_frames=self.wave_baseline_spin.value(),
                frame_rate_hz_override=rate_override,
                frame_range=frame_range,
            )
        except WaveSpeedError as exc:
            self.wave_speed_status.setText(str(exc))
            return

        self._last_wave_profile = profile
        self._last_wave_direction = direction
        self._last_wave_field_key = field_key
        self._last_wave_signal_for_scale = profile.signal
        self._draw_field_plot(profile, direction, field_key)
        self.wave_speed_status.setText(
            f"Showing {field_key} ({direction.upper()} direction), {profile.times.size} frame(s) -- "
            "draw a ROI polygon below around each visible band."
        )

    def _draw_field_plot(self, profile, direction, field_key):
        self._deactivate_roi_selector()
        self.wave_speed_canvas.clear()
        ax = self.wave_speed_canvas.figure.add_subplot(111)
        unit, scale = self._y_axis_unit_and_scale()

        extent = [float(profile.times[0]), float(profile.times[-1]),
                  float(profile.positions[0]) * scale, float(profile.positions[-1]) * scale]
        vmin, vmax = self._current_wave_scale_vmin_vmax()
        im = ax.imshow(profile.signal, aspect="auto", origin="lower", extent=extent, cmap="jet",
                        vmin=vmin, vmax=vmax)
        self.wave_speed_canvas.figure.colorbar(im, ax=ax, label=field_key, fraction=0.046, pad=0.04)

        self._draw_roi_waves_overlay(ax, scale)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"{direction} position ({unit})")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        if self._roi_waves:
            ax.legend(fontsize=8, loc="best")
        self.wave_speed_canvas.refresh()

    # -- ROI polygon wave picking (manual, threshold-free) -----------------

    def _draw_roi_waves_overlay(self, ax, scale):
        """Draws every already-stored ROI-picked wave on top of whichever
        background field was just (re)drawn by `_draw_field_plot`, right
        before its `ax.legend(...)`, so redrawing the background (a color
        scale toggle, a new "Show field" click) never makes a stored ROI
        wave disappear. No separate "smoothed" line -- "Smooth wave
        position" (see `_on_roi_add_clicked`) bakes the smoothed position
        INTO a wave when it's added, so there's exactly one line per wave
        regardless: already-smoothed if that option was on when it was
        added, raw otherwise."""
        for wave in self._roi_waves:
            ax.plot(wave.times, wave.positions * scale, color=wave.color, linewidth=1.6,
                    marker="o", markersize=3, linestyle="-", zorder=6, label=f"{wave.label} (ROI)")

    def _deactivate_roi_selector(self):
        """Drops the live PolygonSelector, if any -- called whenever the
        canvas's axes is about to be rebuilt (neither PolygonSelector nor
        any other matplotlib selector survives that), and when ROI
        drawing is cancelled or a polygon is added to the list. Safe to
        call at any time, selector active or not."""
        if self._roi_selector is not None:
            try:
                self._roi_selector.set_visible(False)
                self._roi_selector.disconnect_events()
            except Exception:
                pass
        was_drawing = self._roi_drawing
        self._roi_selector = None
        self._roi_drawing = False
        self._pending_roi_vertices = None
        self.roi_draw_button.setEnabled(True)
        self.roi_cancel_button.setEnabled(False)
        self.roi_add_button.setEnabled(False)
        return was_drawing

    def _on_roi_draw_clicked(self):
        axes = self.wave_speed_canvas.figure.axes
        if not axes:
            self.roi_status.setText(
                "Click \"Show field\" above first, to show the field to draw on."
            )
            return
        ax = axes[0]
        self._roi_selector = PolygonSelector(
            ax, self._on_roi_polygon_select, useblit=True,
            props=dict(color=theme.ACCENT, linewidth=1.5, alpha=0.7),
        )
        self._roi_drawing = True
        self._pending_roi_vertices = None
        self.roi_draw_button.setEnabled(False)
        self.roi_cancel_button.setEnabled(True)
        self.roi_add_button.setEnabled(False)
        self.roi_status.setText(
            "Click to place polygon vertices around one visible band, then click the first vertex "
            "again (or press Enter) to close it -- \"Add to list\" turns on once it's closed."
        )

    def _on_roi_cancel_clicked(self):
        self._deactivate_roi_selector()
        self.roi_status.setText("Polygon drawing cancelled.")

    def _on_roi_polygon_select(self, verts):
        """PolygonSelector's completion callback -- fires every time the
        polygon changes (closed, or a vertex dragged afterward), not just
        once, so this only updates the pending vertices/button state, it
        does not add the wave itself (that's `_on_roi_add_clicked`, so the
        user can review/adjust before committing)."""
        if len(verts) < 3:
            self._pending_roi_vertices = None
            self.roi_add_button.setEnabled(False)
            return
        self._pending_roi_vertices = [(float(t), float(p)) for t, p in verts]
        self.roi_add_button.setEnabled(True)
        self.roi_status.setText(
            f"Polygon closed ({len(self._pending_roi_vertices)} vertices) -- enter a label and "
            "click \"Add to list\", or keep adjusting it (drag a vertex), or \"Cancel polygon\" to "
            "start over."
        )

    def _on_roi_add_clicked(self):
        if self._dataset is None or self._pending_roi_vertices is None:
            return
        field_key = self.wave_field_combo.currentData()
        direction = "x" if self.wave_x_radio.isChecked() else "y"
        rate_override = self.wave_rate_spin.value() or None
        pixel_size = self.pixel_size_spin.value() or None
        frame_range = (self.wave_start_spin.value(), self.wave_end_spin.value())
        unit, scale = self._y_axis_unit_and_scale()
        # The polygon was drawn in on-screen (time, position * scale)
        # coordinates -- convert the position axis back to raw px before
        # calling the core extractor, which always works in px (a no-op
        # when no pixel size is set, since scale == 1.0 then).
        vertices_px = [(t, p / scale) for t, p in self._pending_roi_vertices]

        try:
            times, positions, values, profile = extract_ridge_in_polygon(
                self._dataset, field_key, direction, vertices_px,
                baseline_frames=self.wave_baseline_spin.value(),
                frame_rate_hz_override=rate_override,
                frame_range=frame_range,
            )
        except WaveSpeedError as exc:
            self.roi_status.setText(str(exc))
            return

        if times.size < 2:
            self.roi_status.setText(
                "Too few points inside that polygon to keep as a wave -- draw a larger polygon "
                "covering more of the band across time."
            )
            return

        label = self.roi_label_edit.text().strip() or f"ROI wave {len(self._roi_waves) + 1}"
        color = DEFAULT_COLORS[len(self._roi_waves) % len(DEFAULT_COLORS)]

        # "Smooth wave position": when on, the smoothed position REPLACES
        # the raw picked one for this wave -- see the Crack Tip
        # Propagation card's own `_on_tip_roi_add_clicked` for the
        # identical rationale, just applied here to `positions` directly
        # (a wave-speed position already IS the physical coordinate, no
        # path re-projection needed -- see `smoothed_wave_positions`).
        smoothing_window_px = self.roi_smoothing_window_spin.value()
        smoothed_applied = self.roi_show_smoothed_check.isChecked() and smoothing_window_px > 0
        if smoothed_applied:
            positions = smoothed_wave_positions(times, positions, smoothing_window_px)
        speed_px, speed_mm, r_squared = fit_ridge_points(times, positions, pixel_size_mm=pixel_size)

        wave = PolygonRidgeResult(
            label=label, color=color, direction=direction, field_key=field_key,
            vertices=list(self._pending_roi_vertices), times=times, positions=positions, values=values,
            speed_px_per_s=speed_px, speed_mm_per_s=speed_mm, r_squared=r_squared,
        )
        self._roi_waves.append(wave)
        self._refresh_roi_waves_list()
        self._deactivate_roi_selector()
        self.roi_label_edit.clear()
        smoothed_note = f", smoothed over {smoothing_window_px:.0f} px" if smoothed_applied else ""
        self.roi_status.setText(
            f"Added \"{label}\" ({times.size} point(s){smoothed_note}) -- draw another polygon, or "
            "export the list to CSV."
        )
        self._redraw_wave_plot()

    def _refresh_roi_waves_list(self):
        self.roi_waves_list.clear()
        # Also keeps the X-FEM-DIC tab's wave-trace checklist in sync,
        # since it lists these same traces -- one refresh point, rather
        # than repeating this call at every add/remove site.
        self._refresh_xfem_trace_lists()
        for wave in self._roi_waves:
            pixmap = QPixmap(12, 12)
            pixmap.fill(QColor(wave.color))
            speed_text = ""
            if wave.speed_px_per_s is not None:
                speed_text = f"  |  v ~ {wave.speed_px_per_s:.4g} px/s"
                if wave.speed_mm_per_s is not None:
                    speed_text += f" = {wave.speed_mm_per_s:.4g} mm/s"
                if wave.r_squared is not None:
                    speed_text += f"  (R²={wave.r_squared:.3f}, indicative only)"
            item = QListWidgetItem(QIcon(pixmap), f"{wave.label}: {wave.times.size} pt(s){speed_text}")
            self.roi_waves_list.addItem(item)

    def _populate_trace_checklists(self, tip_list_widget, wave_list_widget):
        """Shared rebuild logic for a (tip traces, wave traces) pair of
        checkable QListWidgets -- used by the X-FEM-DIC tab's "Crack path
        source" checklist, which keeps its own independent tick state over
        the underlying `self._tip_roi_traces`/`self._roi_waves`. Called
        from both underlying lists' own refresh points
        (`_refresh_tip_roi_list`, `_refresh_roi_waves_list`), so the list
        stays in sync without needing its own "Refresh list" click after
        every add/remove (the button is still there for convenience).
        Preserves which trace(s) were ticked, by label, across a refresh."""
        checked_tip_labels = {
            tip_list_widget.item(row).data(Qt.UserRole)
            for row in range(tip_list_widget.count())
            if tip_list_widget.item(row).checkState() == Qt.Checked
        }
        checked_wave_labels = {
            wave_list_widget.item(row).data(Qt.UserRole)
            for row in range(wave_list_widget.count())
            if wave_list_widget.item(row).checkState() == Qt.Checked
        }

        tip_list_widget.clear()
        for trace in self._tip_roi_traces:
            item = QListWidgetItem(f"{trace.label} [{trace.method}]  ({trace.times.size} pt(s))")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setData(Qt.UserRole, trace.label)
            item.setCheckState(Qt.Checked if trace.label in checked_tip_labels else Qt.Unchecked)
            tip_list_widget.addItem(item)

        wave_list_widget.clear()
        for wave in self._roi_waves:
            item = QListWidgetItem(f"{wave.label}  ({wave.direction}-direction, {wave.times.size} pt(s))")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setData(Qt.UserRole, wave.label)
            item.setCheckState(Qt.Checked if wave.label in checked_wave_labels else Qt.Unchecked)
            wave_list_widget.addItem(item)

    def _on_roi_remove_clicked(self):
        row = self.roi_waves_list.currentRow()
        if row < 0 or row >= len(self._roi_waves):
            self.roi_status.setText("Select a wave in the list above first.")
            return
        removed = self._roi_waves.pop(row)
        self._refresh_roi_waves_list()
        self._redraw_wave_plot()
        self.roi_status.setText(f"Removed \"{removed.label}\".")

    def _on_roi_export_clicked(self):
        if not self._roi_waves:
            self.roi_status.setText("No ROI-picked wave to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export ROI waves", "roi_waves.csv", "CSV files (*.csv)")
        if not path:
            return
        pixel_size = self.pixel_size_spin.value() or None
        try:
            export_roi_waves_csv(self._roi_waves, path, pixel_size_mm=pixel_size)
        except OSError as exc:
            self.roi_status.setText(f"Could not write the file: {exc}")
            return
        self.roi_status.setText(f"Exported {len(self._roi_waves)} wave(s) to {os.path.basename(path)}.")

    # -- crack tip propagation ---------------------------------------------

    def _refresh_tip_crack_combo(self):
        """Repopulates the Cracks combo from the tracing picker's current
        crack list -- called on load and whenever picker.cracksChanged
        fires, so a crack traced/edited/auto-detected after this card was
        first built is picked up without a manual refresh. Every crack is
        ticked by default; a tick the user already changed is kept for the
        same position in the list (cracks beyond the previous count come
        in ticked)."""
        cracks = self.picker.get_cracks()
        previous_index = self.tip_crack_combo.currentIndex()
        previous_checks = self.tip_crack_combo.checked_states()
        self.tip_crack_combo.blockSignals(True)
        self.tip_crack_combo.clear_items()
        for i, crack in enumerate(cracks):
            ticked = previous_checks[i] if i < len(previous_checks) else True
            self.tip_crack_combo.add_check_item(f"{crack.name} ({len(crack.nodes)} pts)", i, ticked)
        self.tip_crack_combo.blockSignals(False)
        if cracks:
            restore = previous_index if 0 <= previous_index < len(cracks) else 0
            self.tip_crack_combo.setCurrentIndex(restore)
        self.tip_crack_combo.update()
        self._refresh_tip_leaf_combo()
        if hasattr(self, "tip_roi_add_button"):
            self._update_tip_add_button_text()

    def _on_tip_crack_checks_changed(self):
        """A tick changed -- only the Detect button's text/enabled state
        depends on it (the previewed crack is the combo's current item)."""
        self._update_tip_add_button_text()

    def _checked_tip_cracks(self):
        """[(crack_index, crack)] for every ticked crack, in list order."""
        cracks = self.picker.get_cracks()
        out = []
        for idx in self.tip_crack_combo.checked_data():
            if idx is not None and 0 <= idx < len(cracks):
                out.append((idx, cracks[idx]))
        return out

    def _update_tip_add_button_text(self):
        n = len(self._checked_tip_cracks())
        if n > 1:
            text = f"Detect tip position (all frames) on {n} cracks -> Add to list"
        else:
            text = "Detect tip position (all frames) -> Add to list"
        self.tip_roi_add_button.setText(text)

    def _on_tip_crack_changed(self, _index=None):
        self._refresh_tip_leaf_combo()
        self._refresh_tip_verify_view()

    def _on_tip_leaf_changed(self, _index=None):
        self._refresh_tip_verify_view()

    def _on_tip_verify_context_input_changed(self, _value=None):
        """Frame range/rate override changed -- the "Verify visually" viewer's
        own context (frame walk order/timing) depends on these, so it needs
        rebuilding; nothing else on this card auto-redraws from these alone
        (an already-shown field plot or ROI trace stays as it was until
        explicitly redrawn/re-picked, same as before)."""
        self._refresh_tip_verify_view()

    def _on_tip_reverse_path_toggled(self, _checked=None):
        """Flipping which end of the path is arc length 0 invalidates any
        already-detected tip traces -- they were computed against the
        previous orientation. Clear them and ask for a re-detection rather
        than silently leaving stale data on screen (or, worse, mixing
        traces from both orientations in the same X-FEM-DIC run)."""
        self._tip_roi_traces = []
        self._tip_roi_path_geometry = []
        self._refresh_tip_roi_list()
        self._refresh_tip_verify_view()
        self._xfem_results = []
        self._xfem_skipped_traces = []
        self.xfem_critical_combo.clear()
        self.xfem_frame_spin.blockSignals(True)
        self.xfem_frame_spin.setRange(0, 0)
        self.xfem_frame_spin.blockSignals(False)
        self.xfem_frame_spin.setEnabled(False)
        self.xfem_frame_slider.blockSignals(True)
        self.xfem_frame_slider.setRange(0, 0)
        self.xfem_frame_slider.blockSignals(False)
        self.xfem_frame_slider.setEnabled(False)
        self._sync_xfem_isocontour_controls()
        self.xfem_result_traces_list.clear()
        self.xfem_canvas.clear()
        self.xfem_canvas.refresh()
        self.xfem_arc_length_canvas.clear()
        self.xfem_arc_length_canvas.refresh()
        self.xfem_speed_canvas.clear()
        self.xfem_speed_canvas.refresh()
        self.xfem_speed_status.setText("")
        self.xfem_mesh_canvas.clear()
        self.xfem_mesh_canvas.refresh()
        self._xfem_isocontour_shown = False
        self._xfem_isocontour_signal_for_scale = None
        self.xfem_isocontour_scale_auto_radio.setChecked(True)
        self.xfem_isocontour_canvas.clear()
        self.xfem_isocontour_canvas.refresh()
        self.tip_roi_status.setText(
            "Path direction flipped -- click \"Detect tip position (all frames) -> Add to list\" "
            "again, since any previous trace(s) were computed for the other orientation."
        )

    @staticmethod
    def _tip_branch_entries(crack):
        """[(leaf_index, x, y, path_length_px)] for every branch of `crack`,
        longest first."""
        paths = branch_paths(crack)
        entries = []
        for leaf_index, chain in paths.items():
            x, y = crack.nodes[leaf_index]
            length = sum(
                float(np.hypot(*(np.array(crack.nodes[chain[i + 1]]) - np.array(crack.nodes[chain[i]]))))
                for i in range(len(chain) - 1)
            )
            entries.append((leaf_index, x, y, length))
        entries.sort(key=lambda e: e[3], reverse=True)
        return entries

    def _refresh_tip_leaf_combo(self):
        self.tip_leaf_combo.clear()
        crack = self._selected_crack()
        if crack is None:
            return
        entries = self._tip_branch_entries(crack)
        # Longest branch first (and selected by default, since a combo
        # box starts on its first item): auto-detected cracks in
        # particular can have several short SPURIOUS side-branches from
        # noisy skeletonization (a single crack's overall traced length
        # passing the "Min length" filter doesn't mean every one of its
        # individual branches does), so defaulting to the longest branch
        # avoids silently tracking a few-pixel twig near the crack's
        # origin instead of the real, far-advancing tip. A branch under
        # 20% of the longest one's length is flagged in its own label so
        # this is obvious even if a shorter tip is deliberately picked.
        longest = entries[0][3] if entries else 0.0
        for leaf_index, x, y, length in entries:
            label = f"Tip at ({x:.0f}, {y:.0f}) -- path length {length:.1f} px"
            if longest > 0 and length < 0.2 * longest:
                label += "  (short branch -- likely a spurious skeleton twig, not the real tip)"
            self.tip_leaf_combo.addItem(label, leaf_index)

    def _trace_on_current_crack(self, trace):
        """True when `trace` was detected on the crack currently highlighted in
        the Cracks list (traces without a recorded crack count as a match)."""
        name = getattr(trace, "crack_name", None)
        if name is None:
            return True
        crack = self._selected_crack()
        return crack is not None and crack.name == name

    def _selected_crack(self):
        crack_index = self.tip_crack_combo.currentData()
        cracks = self.picker.get_cracks()
        if crack_index is None or not (0 <= crack_index < len(cracks)):
            return None
        return cracks[crack_index]

    # -- crack tip detection: opening-jump method (ground truth) -----------
    # Fully automatic -- walks every frame's raw u/v jump along the traced
    # crack path itself, no field/polygon picking needed. See
    # `_build_tip_detection_card` for the controls and
    # `track_tip_opening_jump` (core/crack_tip_tracking.py) for the
    # detection itself.

    def _on_tip_roi_add_clicked(self):
        """Runs the opening-jump tip detection on EVERY ticked crack in the
        "Cracks" list (one trace each). For the highlighted crack the "Tip"
        list's choice is used; any other ticked crack uses its longest
        branch. A crack that fails (threshold never reached, ...) is
        reported and skipped; the others are still added."""
        if self._dataset is None:
            return
        ticked = self._checked_tip_cracks()
        if not ticked:
            self.tip_roi_status.setText("Tick at least one crack in the \"Cracks\" list first.")
            return
        active_index = self.tip_crack_combo.currentData()
        active_leaf = self.tip_leaf_combo.currentData()
        user_label = self.tip_roi_label_edit.text().strip()
        multi = len(ticked) > 1
        added, skipped = [], []
        for crack_index, crack in ticked:
            if crack_index == active_index and active_leaf is not None:
                leaf_index = active_leaf
            else:
                entries = self._tip_branch_entries(crack)
                if not entries:
                    skipped.append(f"{crack.name}: no branch to track")
                    continue
                leaf_index = entries[0][0]
            if multi:
                label = f"{user_label} - {crack.name}" if user_label else f"{crack.name} tip"
            else:
                label = user_label or None
            ok, message = self._add_tip_trace_for(crack, leaf_index, label)
            (added if ok else skipped).append(message)
        if multi:
            text = f"Added {len(added)} of {len(ticked)} trace(s)."
            if added:
                text += " " + "; ".join(added) + "."
            if skipped:
                text += " Skipped: " + "; ".join(skipped)
            text += (" Check them in the \"Verify visually\" viewer below, or tick them in the "
                     "X-FEM-DIC tab's \"Crack path source\" list.")
            self.tip_roi_status.setText(text)
        elif added:
            self.tip_roi_status.setText(
                f"Added {added[0]} -- check it in the \"Verify visually\" viewer below, tick it in "
                "the X-FEM-DIC tab's \"Crack path source\" list to use it there, or export the list "
                "to CSV."
            )
        elif skipped:
            self.tip_roi_status.setText(skipped[0])
        if added:
            self.tip_roi_label_edit.clear()

    def _add_tip_trace_for(self, crack, leaf_index, label_override=None):
        """Detects one crack/leaf's tip trace over the chosen frames and
        appends it to the trace list. Returns (ok, message); the message is
        a short description of the added trace on success, the reason on
        failure."""
        frame_range = (self.tip_start_spin.value(), self.tip_end_spin.value())
        rate_override = self.tip_rate_spin.value() or None
        pixel_size = self.pixel_size_spin.value() or None

        enforce_monotonic = self.tip_roi_monotonic_check.isChecked()
        try:
            # No polygon, no field_key (it reads the raw u/v jump across
            # the traced path itself, not a single derived DIC field),
            # just the traced path/leaf and its own offset/threshold
            # controls.
            field_key = "u,v jump"
            estimate_uncertainty = self.tip_roi_uncertainty_check.isChecked()
            times, arc_length, values, frame_idx, profile, position_uncertainty_px = track_tip_opening_jump(
                self._dataset, crack, leaf_index,
                offset_px=self.tip_roi_opening_jump_offset_spin.value(),
                threshold_px=self.tip_roi_opening_jump_threshold_spin.value(),
                max_gap_samples=self.tip_roi_opening_jump_max_gap_spin.value(),
                frame_range=frame_range, frame_rate_hz_override=rate_override,
                enforce_monotonic=enforce_monotonic,
                backslide_tolerance_px=self.tip_roi_backslide_spin.value() or None,
                reverse_path=self.tip_reverse_path_check.isChecked(),
                estimate_uncertainty=estimate_uncertainty,
                smooth_window_samples=self.tip_roi_opening_jump_smooth_spin.value(),
                advance_confirm_frames=self.tip_roi_advance_confirm_spin.value(),
                jump_criterion=self._opening_jump_criterion_kwargs()["criterion"],
                **self._opening_jump_continuous_kwargs(),
            )
        except CrackTipTrackingError as exc:
            return False, f"{crack.name}: {exc}"

        if times.size < 2:
            return False, (
                f"{crack.name}: the opening-jump threshold was never reached from the path's own "
                "origin, in too few frames to keep as a trace -- lower the threshold, increase the "
                "sampling offset, or check the traced path actually follows the crack."
            )

        label = label_override or f"Tip trace {len(self._tip_roi_traces) + 1}"
        color = DEFAULT_COLORS[len(self._tip_roi_traces) % len(DEFAULT_COLORS)]

        # "Show smoothed position overlay": when on, the smoothed arc
        # length/position REPLACES the raw picked one for this trace --
        # not a display-only addition -- so everything downstream (this
        # trace's own speed fit, X-FEM-DIC, CSV export) uses the smoothed
        # path, since they all just read
        # `trace.arc_length`/`trace.positions_xy` with no idea a
        # smoothing option even exists. The raw picked points aren't kept
        # anywhere once this runs -- re-detect (or add again with the
        # option off) to get them back.
        smoothing_window_px = self.tip_roi_smoothing_window_spin.value()
        smoothed_applied = self.tip_roi_show_smoothed_check.isChecked() and smoothing_window_px > 0
        if smoothed_applied:
            arc_length, positions_xy = smoothed_tip_arc_length_and_positions(
                times, arc_length, profile.path_positions, profile.path_arc_length, smoothing_window_px,
            )
        else:
            positions_xy = np.column_stack([
                np.interp(arc_length, profile.path_arc_length, profile.path_positions[:, 0]),
                np.interp(arc_length, profile.path_arc_length, profile.path_positions[:, 1]),
            ])
        rate_uncertainty_frac = (
            self.tip_roi_rate_uncertainty_spin.value() / 100.0 if estimate_uncertainty else 0.0
        )
        speed_px, speed_mm, r_squared, speed_stderr_px, speed_stderr_mm = fit_tip_ridge_points(
            times, arc_length, pixel_size_mm=pixel_size,
            uncertainty_px=(position_uncertainty_px if estimate_uncertainty else None),
            frame_rate_uncertainty_frac=rate_uncertainty_frac,
        )

        trace = PolygonTipRidgeResult(
            label=label, color=color, leaf_index=leaf_index, field_key=field_key,
            vertices=[],
            times=times, arc_length=arc_length,
            positions_xy=positions_xy, values=values, source_frame_positions=frame_idx,
            speed_px_per_s=speed_px, speed_mm_per_s=speed_mm, r_squared=r_squared,
            method="opening_jump",
            position_uncertainty_px=(position_uncertainty_px if estimate_uncertainty else None),
            speed_stderr_px_per_s=speed_stderr_px, speed_stderr_mm_per_s=speed_stderr_mm,
            crack_name=crack.name,
        )
        self._tip_roi_traces.append(trace)
        self._tip_roi_path_geometry.append((profile.path_positions, profile.path_arc_length))
        self._refresh_tip_roi_list()
        smoothed_note = f", smoothed over {smoothing_window_px:.0f} px" if smoothed_applied else ""
        return True, f"\"{label}\" ({times.size} point(s){smoothed_note})"

    def _refresh_tip_roi_list(self):
        self.tip_roi_list.clear()
        # Also keeps the X-FEM-DIC tab's "Crack path source" checklist in
        # sync -- one refresh point, same reasoning as
        # `_refresh_roi_waves_list`. Also keeps the "Verify visually"
        # viewer's "Show trace" list and its own arc-length-vs-time plot
        # in sync with the trace list.
        self._refresh_xfem_trace_lists()
        self._refresh_tip_verify_view()
        for trace in self._tip_roi_traces:
            pixmap = QPixmap(12, 12)
            pixmap.fill(QColor(trace.color))
            speed_text = ""
            if trace.speed_px_per_s is not None:
                speed_text = f"  |  v ~ {trace.speed_px_per_s:.4g} px/s"
                if trace.speed_stderr_px_per_s is not None:
                    speed_text += f" +/- {trace.speed_stderr_px_per_s:.2g} px/s"
                if trace.speed_mm_per_s is not None:
                    speed_text += f" = {trace.speed_mm_per_s:.4g} mm/s"
                    if trace.speed_stderr_mm_per_s is not None:
                        speed_text += f" +/- {trace.speed_stderr_mm_per_s:.2g} mm/s"
                if trace.r_squared is not None:
                    if trace.speed_stderr_px_per_s is not None:
                        speed_text += f"  (weighted fit, R²={trace.r_squared:.3f})"
                    else:
                        speed_text += f"  (R²={trace.r_squared:.3f}, indicative only)"
            item = QListWidgetItem(QIcon(pixmap), f"{trace.label} [{trace.method}]: {trace.times.size} pt(s){speed_text}")
            self.tip_roi_list.addItem(item)

    def _on_tip_roi_remove_clicked(self):
        row = self.tip_roi_list.currentRow()
        if row < 0 or row >= len(self._tip_roi_traces):
            self.tip_roi_status.setText("Select a trace in the list above first.")
            return
        removed = self._tip_roi_traces.pop(row)
        self._tip_roi_path_geometry.pop(row)
        self._refresh_tip_roi_list()
        self.tip_roi_status.setText(f"Removed \"{removed.label}\".")

    def _on_tip_roi_delete_all_clicked(self):
        """Removes EVERY detected tip trace at once (after a confirmation)."""
        n = len(self._tip_roi_traces)
        if n == 0:
            self.tip_roi_status.setText("The list is already empty.")
            return
        answer = QMessageBox.question(
            self, "Delete all traces",
            f"Remove all {n} detected trace(s) from the list? They would have to be detected again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._tip_roi_traces = []
        self._tip_roi_path_geometry = []
        self._tip_verify_unticked_traces = set()
        self._refresh_tip_roi_list()
        self.tip_roi_status.setText(f"Deleted all {n} trace(s).")

    def _on_tip_roi_export_clicked(self):
        if not self._tip_roi_traces:
            self.tip_roi_status.setText("No ROI-picked tip trace to export yet.")
            return
        self._export_tip_traces(list(self._tip_roi_traces), "Export tip ROI traces", "crack_tip_roi",
                                self.tip_roi_status)

    # -- "Verify visually" -- a live, READ-ONLY raw-field viewer for the
    # "opening_jump" method (NOT a picking tool -- detection itself is
    # fully automatic, see track_tip_opening_jump). Replaced the old
    # "manual (click)" tool's canvas/slider/color-scale machinery; reuses
    # `build_manual_tip_pick_context` purely for its path/frame-order
    # bookkeeping (no picking/click handling here at all).

    def _refresh_tip_verify_view(self):
        """The viewer's single refresh entry point -- rebuilds
        `self._tip_verify_ctx` from the CURRENT crack/tip/frame range/
        direction, repopulates "Show trace" with this leaf's own
        "opening_jump" traces, and redraws both the field image and the
        arc-length-vs-time plot. Called whenever any of those inputs
        change (crack/tip selection, frame range, reverse-path, a new CSV
        loaded, or the trace list changing) so the viewer never needs an
        explicit "refresh" button -- it's always showing the current
        context, live.

        Also re-syncs "Detect tip position (all frames) -> Add to list"'s
        enabled state -- it's only meant to be enabled once a dataset is
        actually loaded, and this is the one entry point guaranteed to
        run whenever that changes (a fresh CSV sets `self._dataset` and
        then changes the frame range spinboxes, which land here)."""
        self.tip_roi_add_button.setEnabled(self._dataset is not None and bool(self._dataset.frames))
        previous_row = self.tip_verify_trace_combo.currentData()
        previous_trace = (self._tip_roi_traces[previous_row]
                          if previous_row is not None and 0 <= previous_row < len(self._tip_roi_traces)
                          else None)
        self.tip_verify_trace_combo.blockSignals(True)
        self.tip_verify_trace_combo.clear_items()
        leaf_index = self.tip_leaf_combo.currentData()
        # EVERY opening-jump trace (whatever its crack), ticked unless the user
        # unticked it; a newly added trace comes in ticked.
        alive = {id(t) for t in self._tip_roi_traces}
        self._tip_verify_unticked_traces &= alive
        last_added_index = 0
        keep_index = None
        for row, trace in enumerate(self._tip_roi_traces):
            if trace.method == "opening_jump":
                self.tip_verify_trace_combo.add_check_item(
                    trace.label, row, id(trace) not in self._tip_verify_unticked_traces, color=trace.color)
                last_added_index = self.tip_verify_trace_combo.count_items() - 1
                if trace is previous_trace:
                    keep_index = last_added_index
        if self.tip_verify_trace_combo.count_items():
            self.tip_verify_trace_combo.setCurrentIndex(
                keep_index if keep_index is not None else last_added_index)
        self.tip_verify_trace_combo.update()
        self.tip_verify_trace_combo.blockSignals(False)

        # "Opening-jump profile of trace": every opening-jump trace (each with its colour), the
        # previously chosen one kept; first entry = the highlighted crack/tip path itself.
        prev_profile_trace = None
        prow = self.tip_verify_profile_combo.currentData()
        if prow is not None and 0 <= prow < len(self._tip_roi_traces):
            prev_profile_trace = self._tip_roi_traces[prow]
        self.tip_verify_profile_combo.blockSignals(True)
        self.tip_verify_profile_combo.clear()
        self.tip_verify_profile_combo.addItem("(highlighted crack / tip path)", -1)
        pick = 0
        for row, trace in enumerate(self._tip_roi_traces):
            if trace.method != "opening_jump":
                continue
            pix = QPixmap(12, 12)
            pix.fill(QColor(trace.color))
            self.tip_verify_profile_combo.addItem(QIcon(pix), trace.label, row)
            if trace is prev_profile_trace or (prev_profile_trace is None and pick == 0):
                pick = self.tip_verify_profile_combo.count() - 1
        self.tip_verify_profile_combo.setCurrentIndex(pick)
        self.tip_verify_profile_combo.blockSignals(False)

        if self._dataset is None or not self._dataset.frames or leaf_index is None:
            self._tip_verify_ctx = None
            self._tip_verify_pos = 0
            self.tip_verify_frame_slider.blockSignals(True)
            self.tip_verify_frame_spin.blockSignals(True)
            self.tip_verify_frame_slider.setRange(0, 0)
            self.tip_verify_frame_spin.setRange(0, 0)
            self.tip_verify_frame_slider.blockSignals(False)
            self.tip_verify_frame_spin.blockSignals(False)
            self.tip_verify_frame_slider.setEnabled(False)
            self.tip_verify_frame_spin.setEnabled(False)
            self.tip_verify_canvas.clear()
            self.tip_verify_canvas.refresh()
            self.tip_verify_jump_canvas.clear()
            self.tip_verify_jump_canvas.refresh()
            self._tip_verify_current_jump_profile = None
            self.tip_verify_jump_export_button.setEnabled(False)
            self.tip_verify_status.setText(
                "Load a DIC results CSV and pick a Crack/Tip above -- the raw field shows here "
                "automatically, live, for any frame in range."
            )
            self._refresh_tip_verify_arc_length_plot()
            self._sync_tip_video_export_enabled()
            return

        crack = self._selected_crack()
        frame_range = (self.tip_start_spin.value(), self.tip_end_spin.value())
        rate_override = self.tip_rate_spin.value() or None
        try:
            ctx = build_manual_tip_pick_context(
                self._dataset, crack, leaf_index, frame_range=frame_range,
                frame_rate_hz_override=rate_override,
                reverse_path=self.tip_reverse_path_check.isChecked(),
            )
        except CrackTipTrackingError as exc:
            self._tip_verify_ctx = None
            self.tip_verify_frame_slider.setEnabled(False)
            self.tip_verify_frame_spin.setEnabled(False)
            self.tip_verify_canvas.clear()
            self.tip_verify_canvas.refresh()
            self.tip_verify_jump_canvas.clear()
            self.tip_verify_jump_canvas.refresh()
            self._tip_verify_current_jump_profile = None
            self.tip_verify_jump_export_button.setEnabled(False)
            self.tip_verify_status.setText(str(exc))
            self._refresh_tip_verify_arc_length_plot()
            self._sync_tip_video_export_enabled()
            return

        self._tip_verify_ctx = ctx
        self._tip_verify_pos = min(self._tip_verify_pos, len(ctx.frame_indices) - 1)
        n_frames = len(ctx.frame_indices)
        self.tip_verify_frame_slider.blockSignals(True)
        self.tip_verify_frame_spin.blockSignals(True)
        self.tip_verify_frame_slider.setRange(0, max(0, n_frames - 1))
        self.tip_verify_frame_spin.setRange(0, max(0, n_frames - 1))
        self.tip_verify_frame_slider.setValue(self._tip_verify_pos)
        self.tip_verify_frame_spin.setValue(self._tip_verify_pos)
        self.tip_verify_frame_slider.blockSignals(False)
        self.tip_verify_frame_spin.blockSignals(False)
        self.tip_verify_frame_slider.setEnabled(True)
        self.tip_verify_frame_spin.setEnabled(True)
        self._draw_tip_verify_frame()
        self._refresh_tip_verify_arc_length_plot()

    def _opening_jump_continuous_kwargs(self) -> dict:
        """Extra keyword arguments of crack_opening_jump_profile / track_tip_opening_jump. The
        'reject wave' option (core: remove_continuous_part) is no longer offered in the UI: the jump is
        always the plain two-point jump u(+offset) - u(-offset)."""
        return {}

    def _opening_jump_criterion_kwargs(self) -> dict:
        return {"criterion": self.tip_roi_opening_jump_criterion_combo.currentData() or "magnitude"}

    def _tip_verify_ticked_rows(self):
        """Rows (into self._tip_roi_traces) of every ticked trace in "Show trace"."""
        return [r for r in self.tip_verify_trace_combo.checked_data()
                if r is not None and 0 <= r < len(self._tip_roi_traces)]

    def _tip_verify_active_row(self):
        """The trace the single-trace features use (video export, uncertainty
        band on the jump profile): the one clicked last in "Show trace" if it
        is ticked, else the first ticked one; None when nothing is ticked."""
        ticked = self._tip_verify_ticked_rows()
        row = self.tip_verify_trace_combo.currentData()
        if row in ticked:
            return row
        return ticked[0] if ticked else None

    def _on_tip_verify_traces_changed(self):
        """A tick in "Show trace" changed: remember what the user unticked,
        redraw the field overlay and the arc-length plot."""
        unticked = set()
        combo = self.tip_verify_trace_combo
        for (text, row, checked) in combo.items_info():
            if not checked and row is not None and 0 <= row < len(self._tip_roi_traces):
                unticked.add(id(self._tip_roi_traces[row]))
        self._tip_verify_unticked_traces = unticked
        self._draw_tip_verify_frame()
        self._refresh_tip_verify_arc_length_plot()
        self._sync_tip_video_export_enabled()

    def _on_tip_verify_field_changed(self, _index=None):
        self._draw_tip_verify_frame()

    def _on_tip_verify_redraw_needed(self, _index=None):
        self._draw_tip_verify_frame()

    def _current_uncertainty_sigma_n(self) -> int:
        """0 ("Off") or 1/2/3 -- the "Uncertainty envelope" dropdown's
        current choice, shared by the opening-jump profile panel and the
        arc-length-vs-time plot (see `_draw_tip_verify_jump_panel`/
        `_refresh_tip_verify_arc_length_plot`)."""
        n = self.tip_verify_uncertainty_sigma_combo.currentData()
        return int(n) if n else 0

    def _on_tip_verify_uncertainty_sigma_changed(self, _index=None):
        self._draw_tip_verify_frame()
        self._refresh_tip_verify_arc_length_plot()

    def _current_tip_verify_scale_vmin_vmax(self):
        if self.tip_verify_scale_fixed_radio.isChecked():
            return self.tip_verify_scale_min_spin.value(), self.tip_verify_scale_max_spin.value()
        return None, None

    def _on_tip_verify_scale_mode_toggled(self, _checked=None):
        fixed = self.tip_verify_scale_fixed_radio.isChecked()
        self.tip_verify_scale_min_spin.setEnabled(fixed)
        self.tip_verify_scale_max_spin.setEnabled(fixed)
        self.tip_verify_scale_reset_button.setEnabled(fixed)
        if fixed and self.tip_verify_scale_min_spin.value() == self.tip_verify_scale_max_spin.value():
            self._on_tip_verify_scale_reset()
        self._draw_tip_verify_frame()

    def _on_tip_verify_scale_bounds_changed(self, _value=None):
        self._draw_tip_verify_frame()

    def _on_tip_verify_scale_reset(self):
        """Resets "Fixed" bounds to the CURRENT frame's own field data
        range -- the field is a fresh raw 2D image every frame (not one
        precomputed (arc length x time) array like the "arrival (ROI)"
        heat map above), so "data range" means "this frame's" until reset
        again on a later frame."""
        signal = self._last_tip_verify_signal_for_scale
        if signal is None:
            return
        finite = signal[np.isfinite(signal)]
        if finite.size == 0:
            return
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        self.tip_verify_scale_min_spin.blockSignals(True)
        self.tip_verify_scale_max_spin.blockSignals(True)
        self.tip_verify_scale_min_spin.setValue(vmin)
        self.tip_verify_scale_max_spin.setValue(vmax)
        self.tip_verify_scale_min_spin.blockSignals(False)
        self.tip_verify_scale_max_spin.blockSignals(False)
        self._draw_tip_verify_frame()

    def _on_tip_verify_frame_slider_changed(self, value: int):
        self.tip_verify_frame_spin.blockSignals(True)
        self.tip_verify_frame_spin.setValue(value)
        self.tip_verify_frame_spin.blockSignals(False)
        self._tip_verify_pos = value
        self._draw_tip_verify_frame()

    def _on_tip_verify_frame_spin_changed(self, value: int):
        self.tip_verify_frame_slider.blockSignals(True)
        self.tip_verify_frame_slider.setValue(value)
        self.tip_verify_frame_slider.blockSignals(False)
        self._tip_verify_pos = value
        self._draw_tip_verify_frame()

    def _tip_verify_local_paths(self):
        """[(positions, arc_length)] of every ticked trace in "Show trace" (each crack's own traced
        path, walked in the direction of growth) -- the cracks whose local frames the u_n / u_t
        fields are built from; the highlighted path when nothing is ticked."""
        paths = []
        for row in self._tip_verify_ticked_rows():
            if row < len(self._tip_roi_path_geometry):
                paths.append(self._tip_roi_path_geometry[row])
        if not paths and self._tip_verify_ctx is not None:
            paths = [(self._tip_verify_ctx.path_positions, self._tip_verify_ctx.path_arc_length)]
        return paths

    def _tip_verify_local_window_px(self) -> float:
        return max(8.0, 4.0 * float(self.tip_roi_opening_jump_offset_spin.value()))

    def _tip_verify_band_px(self):
        """(inner, outer) of the band kept around the cracks for the local-frame fields, or (None, None)
        for the whole field."""
        if not self.tip_verify_band_check.isChecked():
            return None, None
        inner = float(self.tip_verify_band_inner_spin.value())
        return (inner if inner > 0 else None), float(self.tip_verify_band_outer_spin.value())

    def _tip_verify_norm(self):
        """(mode, threshold_px) of the jump-field normalisation."""
        return (self.tip_verify_norm_combo.currentData() or "px",
                float(self.tip_roi_opening_jump_threshold_spin.value()))

    def _tip_verify_field_label(self) -> str:
        """Colour-bar / status text of the current field, with the normalisation of a jump field."""
        text = self.tip_verify_field_combo.currentText()
        key = self.tip_verify_field_combo.currentData()
        if key in LOCAL_FRAME_FIELDS and LOCAL_FRAME_FIELDS[key][1] in ("dn", "dt"):
            mode = self.tip_verify_norm_combo.currentData() or "px"
            if mode != "px":
                head = text.split(" (px)")[0].split(" -- ")[0]
                text = head + {"thr": " / detection threshold", "max": " / crack's own maximum"}[mode]
        return text

    def _tip_verify_field_array(self, dic_frame, field_key: str):
        """The frame's field for the viewer: a stored field by key, or a local-frame field
        (u_n / u_t, or the opening / sliding jump across the crack) of the ticked cracks."""
        if field_key in LOCAL_FRAME_FIELDS:
            return _local_field_array(
                LOCAL_FRAME_FIELDS[field_key][1], dic_frame.u, dic_frame.v, self._dataset.xs, self._dataset.ys,
                self._tip_verify_local_paths(), self._tip_verify_local_window_px(),
                self._tip_verify_band_px(), self._tip_verify_norm())
        return dic_frame.field(field_key)

    def _draw_tip_verify_frame(self):
        """Draws the current frame's standard subset-DIC displacement
        field as a FILLED isocontour plot -- matching
        `_draw_xfem_isocontours`'s "Standard subset-DIC" panel convention
        exactly (levels/colormap/aspect/y-axis), rather than a raw imshow
        -- with the traced path overlaid (thin dotted, for context), the
        picked "opening_jump" trace's own full detected trajectory
        (white+black dashed), and -- when that trace has a detected
        position AT this frame -- a "+" marker there (a frame the trace's
        own walk skipped, because the jump never reached threshold there,
        simply has no marker drawn).

        The opening-jump diagnostic itself -- re-running
        `crack_opening_jump_profile` for THIS SAME frame (same offset/
        threshold spinboxes "Detect tip position (all frames) -> Add to
        list" itself uses) and plotting its `jump_mag_px` directly
        against arc length along the path -- is drawn separately below,
        on its own canvas (`self.tip_verify_jump_canvas`, see
        `_draw_tip_verify_jump_panel`), so this isocontour view gets the
        full canvas to itself: the raw u/v field here is a full-field 2D
        map (a smooth left-right gradient from bulk straining, not a
        discontinuity you can read a crack-opening front off directly),
        while the jump panel below is the actual near-tip face-separation
        measurement, isolated to the traced path alone. Seeing that panel
        directly on a frame where no point was detected (no marker here /
        a note in the status line below) is the fastest way to tell
        whether the walk missed a genuine, clean front (a detection bug)
        or whether the path's own origin/near-origin region genuinely
        never reaches threshold that frame (nothing to walk from, not a
        bug)."""
        ctx = self._tip_verify_ctx
        if ctx is None:
            return
        pos = self._tip_verify_pos
        frame_idx = int(ctx.frame_indices[pos])
        dic_frame = self._dataset.frames[frame_idx]
        field_key = self.tip_verify_field_combo.currentData() or "u"
        field = self._tip_verify_field_array(dic_frame, field_key)
        field_arr = np.asarray(field, dtype=float)
        self._last_tip_verify_signal_for_scale = field_arr

        self.tip_verify_canvas.clear()
        fig = self.tip_verify_canvas.figure
        ax = fig.add_subplot(111)

        vmin, vmax = self._current_tip_verify_scale_vmin_vmax()
        if vmin is None or vmax is None:
            finite = field_arr[np.isfinite(field_arr)]
            if finite.size >= 2:
                # Robust (percentile, not min/max) range, same reasoning
                # as `_draw_xfem_isocontours`: a handful of extreme pixels
                # would otherwise wash out the color scale everywhere else.
                vmin, vmax = np.nanpercentile(finite, [2.0, 98.0])
            elif finite.size == 1:
                vmin, vmax = float(finite[0]) - 1.0, float(finite[0]) + 1.0
            else:
                vmin, vmax = -1.0, 1.0
        if np.isfinite(vmin) and np.isfinite(vmax) and vmin > vmax:
            vmin, vmax = vmax, vmin
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = (vmin - 1.0 if np.isfinite(vmin) else -1.0,
                          vmax + 1.0 if np.isfinite(vmax) else 1.0)
        levels = isocontour_levels(vmin, vmax)

        masked = np.ma.masked_invalid(field_arr)
        if np.ma.count(masked) > 0:
            cf = ax.contourf(self._dataset.xs, self._dataset.ys, masked, levels=levels, cmap="jet",
                              extend="both")
            isocontour_colorbar(fig, cf, ax=ax, label=self._tip_verify_field_label(), fraction=0.046,
                         pad=0.04)
            if (field_key in LOCAL_FRAME_FIELDS and LOCAL_FRAME_FIELDS[field_key][1] in ("dn", "dt")
                    and self.tip_verify_norm_combo.currentData() == "thr" and np.nanmax(field_arr) >= 1.0
                    and np.nanmin(field_arr) <= 1.0):
                # the detection threshold itself: where the jump criterion calls the crack open
                ax.contour(self._dataset.xs, self._dataset.ys, masked, levels=[1.0], colors="white",
                           linewidths=0.9, linestyles="--", zorder=3)
        else:
            ax.text(0.5, 0.5, "No standard-DIC field\nfor this frame/field", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)

        # Static drawn-path geometry (thin/faint, for context only) -- see
        # `_draw_xfem_isocontours._overlay_path_and_tip` for the same
        # two-lines-not-one convention this mirrors.
        ax.plot(ctx.path_positions[:, 0], ctx.path_positions[:, 1], color="0.6", linewidth=0.8,
                linestyle=":", alpha=0.7, zorder=4, label="Traced path")

        marker_note = ""
        for trace_row in self._tip_verify_ticked_rows():
            trace = self._tip_roi_traces[trace_row]
            finite = np.isfinite(trace.positions_xy).all(axis=1)
            if np.count_nonzero(finite) >= 1:
                # Walk the trace's own traced path (its stored geometry,
                # not necessarily `ctx.path_positions` if the crack/leaf/
                # direction shown above has since changed) from its origin
                # up to the FURTHEST arc length any detected point of this
                # trace reached -- not a straight line between the
                # (temporally sparse) detected points, which cuts corners
                # whenever several frames pass between two detections
                # (e.g. the crack sped up -- nothing wrong with that, only
                # the straight-chord drawing of it was wrong). See
                # core.crack_tip_tracking.path_polyline_up_to_arc_length.
                # Each trace is drawn in its own colour (white halo underneath
                # so it stays readable on the field).
                trace_path_positions, trace_path_arc_length = self._tip_roi_path_geometry[trace_row]
                max_s = float(np.nanmax(trace.arc_length[finite]))
                traj = path_polyline_up_to_arc_length(trace_path_positions, trace_path_arc_length, max_s)
                ax.plot(traj[:, 0], traj[:, 1], color="white", linewidth=2.6, alpha=0.9, zorder=5)
                ax.plot(traj[:, 0], traj[:, 1], color=trace.color, linewidth=1.4, zorder=6,
                        label=f"\"{trace.label}\" trajectory")
                # Small markers at the actual detected points, so the
                # (real) sparsity in time stays visible along the path.
                detected = trace.positions_xy[finite]
                ax.scatter(detected[:, 0], detected[:, 1], s=10, color=trace.color, zorder=5,
                           edgecolors="white", linewidths=0.3)
            match = np.flatnonzero(trace.source_frame_positions == frame_idx)
            if match.size:
                x, y = trace.positions_xy[match[0]]
                ax.plot([x], [y], marker="+", color=trace.color, markersize=18, markeredgewidth=3.0,
                        linestyle="none", zorder=8, markeredgecolor="white")
                ax.plot([x], [y], marker="+", color=trace.color, markersize=16, markeredgewidth=2.0,
                        linestyle="none", zorder=9, label=f"\"{trace.label}\" -- detected here")
            else:
                marker_note += f"  (\"{trace.label}\" has no detected position at this frame.)"

        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()  # image pixel convention (y downward)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        n_total = len(ctx.frame_indices)
        ax.set_title(
            f"Frame {pos + 1}/{n_total} (dataset frame {dic_frame.frame_index}, "
            f"t = {ctx.times[pos]:.6g} s)",
            fontsize=9,
        )
        legend = ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
        legend.set_draggable(True)

        self.tip_verify_canvas.refresh()

        profile_note = self._draw_tip_verify_jump_panel(ctx, dic_frame)

        self.tip_verify_status.setText(
            f"Frame {pos + 1}/{n_total} -- {self._tip_verify_field_label()}.{marker_note}{profile_note}"
        )
        self._sync_tip_video_export_enabled()

    def _draw_tip_verify_jump_panel(self, ctx, dic_frame) -> str:
        """The actual opening-jump diagnostic (`crack_opening_jump_
        profile`'s own jump_mag_px vs arc length) for the current frame,
        with the SAME offset/threshold this card's "Add to list" detector
        uses -- drawn on its own canvas (`self.tip_verify_jump_canvas`),
        separate from the isocontour view above, so the isocontour gets
        the full canvas to itself (see `_draw_tip_verify_frame`'s own
        docstring for why this panel, not the isocontour, is the signal
        to read when a frame's detection is in question). Returns a note
        string (empty, or explaining a missed detection) to fold into the
        status line.

        Also stashes the computed `CrackOpeningJumpProfile` (plus this
        frame's index/time) onto `self._tip_verify_current_jump_profile`
        so "Export this profile to CSV..." can reuse it without
        recomputing -- see `_on_tip_verify_jump_export_clicked`.

        When the "Uncertainty envelope" dropdown isn't "Off" AND the
        trace picked in "Show trace" has a stored `position_uncertainty_
        px` at THIS frame (i.e. it was added with "Estimate measurement
        uncertainty" checked), shades a +/- N sigma band around that
        trace's own recorded front position -- not around the live
        re-detection above, which can drift from the trace's stored
        position if the offset/threshold/gap spinboxes were changed since
        that trace was added. No band is drawn otherwise (nothing to
        show it against)."""
        self.tip_verify_jump_canvas.clear()
        fig = self.tip_verify_jump_canvas.figure
        ax2 = fig.add_subplot(111)
        offset_px = self.tip_roi_opening_jump_offset_spin.value()
        threshold_px = self.tip_roi_opening_jump_threshold_spin.value()
        max_gap_samples = self.tip_roi_opening_jump_max_gap_spin.value()
        smooth_window_samples = self.tip_roi_opening_jump_smooth_spin.value()
        profile_note = ""
        self._tip_verify_current_jump_profile = None
        self.tip_verify_jump_export_button.setEnabled(False)
        # The profile follows the trace chosen in "Opening-jump profile ... of trace": that trace's own
        # stored path (so it is the same path its front was detected on); else the highlighted crack/tip.
        prof_row = self.tip_verify_profile_combo.currentData()
        prof_trace = None
        prof_path_positions, prof_path_arc = ctx.path_positions, ctx.path_arc_length
        if prof_row is not None and 0 <= prof_row < len(self._tip_roi_traces):
            prof_trace = self._tip_roi_traces[prof_row]
            prof_path_positions, prof_path_arc = self._tip_roi_path_geometry[prof_row]
        try:
            profile = crack_opening_jump_profile(
                dic_frame.u, dic_frame.v, self._dataset.xs, self._dataset.ys,
                prof_path_positions, prof_path_arc,
                offset_px=offset_px, threshold_px=threshold_px, max_gap_samples=max_gap_samples,
                smooth_window_samples=smooth_window_samples,
                **self._opening_jump_continuous_kwargs(), **self._opening_jump_criterion_kwargs(),
            )
            # Raw jump, always shown for reference; when smoothing is on
            # ("Jump smoothing" > 1 sample), also overlay the smoothed
            # signal the walk/threshold crossing is ACTUALLY computed
            # from, so a wave-pulse spike being knocked down (or a
            # genuine plateau surviving it) is directly visible here.
            signal_label = _jump_signal_label(profile)
            ax2.plot(profile.arc_length_s, profile.jump_criterion_px, color="#e67e22", linewidth=1.0,
                      alpha=0.55 if smooth_window_samples > 1 else 1.0, label=f"{signal_label} -- raw")
            if profile.criterion != "magnitude":
                ax2.plot(profile.arc_length_s, profile.jump_mag_px, color="0.55", linewidth=0.8,
                          linestyle=":", label="|jump| (u,v combined) -- for reference")
            if smooth_window_samples > 1:
                ax2.plot(profile.arc_length_s, profile.jump_mag_smoothed_px, color="#c0392b",
                          linewidth=1.3,
                          label=f"smoothed ({smooth_window_samples} sample window) -- used for detection")
            ax2.axhline(threshold_px, color="0.6", linestyle="--", linewidth=1.0,
                         label=f"threshold = {threshold_px:.3g} px")
            if profile.opening_extent_s is not None:
                ax2.axvline(profile.opening_extent_s, color="#f1c40f", linewidth=1.5, alpha=0.9,
                             label=f"detected front (s={profile.opening_extent_s:.1f} px)")
            else:
                profile_note = "  (never reaches threshold near the path's own origin this frame.)"

            sigma_n = self._current_uncertainty_sigma_n()
            if prof_trace is not None:
                trace = prof_trace
                match = np.flatnonzero(trace.source_frame_positions == dic_frame.frame_index)
                if match.size:
                    s_pos = float(trace.arc_length[match[0]])
                    if np.isfinite(s_pos):
                        ax2.axvline(s_pos, color=trace.color, linewidth=1.8, linestyle="--", zorder=3,
                                    label=f"\"{trace.label}\" -- stored front (s={s_pos:.1f} px)")
                    unc_px = getattr(trace, "position_uncertainty_px", None)
                    if sigma_n > 0 and unc_px is not None:
                        u = float(unc_px[match[0]])
                        if np.isfinite(u) and np.isfinite(s_pos):
                            ax2.axvspan(s_pos - sigma_n * u, s_pos + sigma_n * u,
                                        color=trace.color, alpha=0.18, linewidth=0, zorder=1,
                                        label=f"+/- {sigma_n} sigma ({sigma_n * u:.2f} px)")
                else:
                    profile_note += f"  (\"{trace.label}\" has no stored position at this frame.)"

            ax2.set_xlabel("distance along path, s (px)")
            ax2.set_ylabel({"normal": "opening jump (px)", "tangential": "sliding jump (px)"}.get(
                profile.criterion, "jump (px)"))
            ax2.legend(fontsize=7, loc="best", framealpha=0.7)
            time_s = float(ctx.times[self._tip_verify_pos])
            self._tip_verify_current_jump_profile = (profile, dic_frame.frame_index, time_s)
            self.tip_verify_jump_export_button.setEnabled(True)
        except CrackTipTrackingError as exc:
            ax2.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax2.transAxes, fontsize=7,
                      wrap=True)
        self.tip_verify_jump_canvas.refresh()
        return profile_note

    def _on_tip_verify_jump_export_clicked(self):
        """Saves the currently-drawn opening-jump profile (the plot right
        above the button, computed for this one frame by
        `_draw_tip_verify_jump_panel` and stashed onto
        `self._tip_verify_current_jump_profile`) to a CSV via
        `export_opening_jump_profile_csv`."""
        if self._tip_verify_current_jump_profile is None:
            self.tip_verify_jump_export_status.setText(
                "No profile currently shown to export -- pick a Crack/Tip and frame above first."
            )
            return
        profile, frame_index, time_s = self._tip_verify_current_jump_profile
        default_name = f"opening_jump_profile_frame_{frame_index}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export opening-jump profile", default_name, "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            export_opening_jump_profile_csv(profile, path, frame_index=frame_index, time_s=time_s)
        except OSError as exc:
            self.tip_verify_jump_export_status.setText(f"Could not write the file: {exc}")
            return
        self.tip_verify_jump_export_status.setText(f"Exported to {path}")

    def _export_tip_traces(self, traces, title, default_stem, status_label):
        """Saves `traces` as an Excel workbook with ONE SHEET PER CRACK
        (default) or as a single CSV (pick the CSV filter). Without openpyxl
        the workbook falls back to one CSV per crack, named after the chosen
        file."""
        path, chosen = QFileDialog.getSaveFileName(
            self, title, f"{default_stem}.xlsx", "Excel workbook, one sheet per crack (*.xlsx);;CSV file (*.csv)"
        )
        if not path:
            return
        pixel_size = self.pixel_size_spin.value() or None
        as_csv = path.lower().endswith(".csv") or (not path.lower().endswith(".xlsx") and "csv" in (chosen or "").lower())
        try:
            if as_csv:
                if not path.lower().endswith(".csv"):
                    path += ".csv"
                export_tip_roi_csv(traces, path, pixel_size_mm=pixel_size)
                status_label.setText(f"Exported {len(traces)} trace(s) to {os.path.basename(path)}.")
                return
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            try:
                sheets = export_tip_roi_xlsx(traces, path, pixel_size_mm=pixel_size)
            except ImportError:
                stem = path[:-5]
                written = []
                for crack_name, group in group_traces_by_crack(traces):
                    safe = "".join(ch if ch.isalnum() or ch in "-_ ()" else "_" for ch in crack_name).strip()
                    out = f"{stem}_{safe}.csv"
                    export_tip_roi_csv(group, out, pixel_size_mm=pixel_size)
                    written.append(os.path.basename(out))
                status_label.setText(
                    "The 'openpyxl' package is not installed (pip install openpyxl), so no Excel file was "
                    f"written; exported one CSV per crack instead: {', '.join(written)}."
                )
                return
            status_label.setText(
                f"Exported {len(traces)} trace(s) to {os.path.basename(path)} -- one sheet per crack "
                f"({', '.join(sheets)})."
            )
        except OSError as exc:
            status_label.setText(f"Could not write the file: {exc}")

    def _on_tip_verify_arc_export_clicked(self):
        """Saves every trace `_refresh_tip_verify_arc_length_plot` itself
        draws (the traces ticked in "Show trace") to an Excel workbook with
        one sheet per crack, or to one CSV -- see `_export_tip_traces`."""
        traces = [self._tip_roi_traces[r] for r in self._tip_verify_ticked_rows()]
        if not traces:
            self.tip_verify_arc_export_status.setText(
                "No trace ticked in \"Show trace\" -- tick at least one first."
            )
            return
        self._export_tip_traces(traces, "Export arc-length-vs-time traces", "tip_arc_length_vs_time",
                                self.tip_verify_arc_export_status)

    def _sync_tip_video_export_enabled(self):
        """"Export video" only makes sense once there's both a live
        viewer context (a dataset, crack/tip and frame range that
        resolved to real frames) AND a specific trace picked in "Show
        trace" (not "(none -- just the raw field)") -- the video's whole
        point is to show that trace's own trajectory/position/speed
        alongside the isocontours, so there's nothing useful to render
        without one."""
        trace_row = self._tip_verify_active_row()
        enabled = (
            self._tip_verify_ctx is not None and trace_row is not None and trace_row >= 0
            and trace_row < len(self._tip_roi_traces)
        )
        self.tip_video_export_button.setEnabled(bool(enabled))

    def _on_tip_video_export_clicked(self):
        """Renders the CURRENTLY selected "Show trace" over its own
        frame range (the "Frame range" spinboxes above) into an MP4:
        `_TipVideoExportWorker`/`core.tip_video_export.export_tip_video`
        do the actual work off the UI thread, so this just gathers the
        fixed, whole-video color/axis scales (color scale from "Color
        scale" above if Fixed, else pooled over every frame in range so
        the video's own colorbar doesn't rescale frame to frame -- see
        core/dic_video_export.py's own reasoning) and starts it.

        Also passes the current "Uncertainty envelope" dropdown choice
        through to the worker: when the selected trace was added with
        "Estimate measurement uncertainty" checked AND that dropdown
        isn't "Off", the exported video's position/speed panel shows the
        same +/- N sigma bands the "Verify visually" viewer does -- when
        either condition isn't met, the video is unchanged from before
        this option existed."""
        ctx = self._tip_verify_ctx
        if ctx is None or self._dataset is None:
            return
        trace_row = self._tip_verify_active_row()
        if trace_row is None or trace_row < 0 or trace_row >= len(self._tip_roi_traces):
            self.tip_video_status.setText("Tick a trace in \"Show trace\" above first.")
            return
        trace = self._tip_roi_traces[trace_row]
        # Every other ticked trace (other cracks) joins the same video.
        extra_traces = [(self._tip_roi_traces[r], *self._tip_roi_path_geometry[r])
                        for r in self._tip_verify_ticked_rows() if r != trace_row]
        # The trace's OWN stored path geometry (not `ctx.path_positions`,
        # which is whatever crack/leaf/direction the "Verify visually"
        # viewer currently happens to show) -- guarantees the video's
        # trajectory-so-far walk (see core.tip_video_export.render_tip_
        # frame) is measured along the exact same path this trace's own
        # arc_length values were computed against, even if the viewer's
        # own crack/leaf selection has since changed.
        path_positions, path_arc_length = self._tip_roi_path_geometry[trace_row]

        default_name = (f"{trace.label.replace(' ', '_')}.mp4" if not extra_traces
                        else f"tip_traces_{1 + len(extra_traces)}_cracks.mp4")
        path, _ = QFileDialog.getSaveFileName(self, "Export tip video", default_name, "MP4 video (*.mp4)")
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"

        field_key = self.tip_verify_field_combo.currentData() or "u"
        field_label = self._tip_verify_field_label()
        field_fn = None
        if field_key in LOCAL_FRAME_FIELDS:
            local_paths = [(np.array(pp, copy=True), np.array(pa, copy=True)) for pp, pa in self._tip_verify_local_paths()]
            local_window = self._tip_verify_local_window_px()
            kind = LOCAL_FRAME_FIELDS[field_key][1]
            xs_l, ys_l = self._dataset.xs, self._dataset.ys
            band = self._tip_verify_band_px()
            norm = self._tip_verify_norm()

            def field_fn(frame, _paths=local_paths, _w=local_window, _k=kind, _xs=xs_l, _ys=ys_l, _b=band, _n=norm):
                return _local_field_array(_k, frame.u, frame.v, _xs, _ys, _paths, _w, _b, _n)
        vmin, vmax = self._current_tip_verify_scale_vmin_vmax()
        if vmin is None or vmax is None:
            pooled = []
            for oi in ctx.frame_indices:
                _fr = self._dataset.frames[int(oi)]
                arr = np.asarray(field_fn(_fr) if field_fn else _fr.field(field_key), dtype=float)
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    pooled.append(finite)
            if pooled:
                vmin, vmax = np.nanpercentile(np.concatenate(pooled), [2.0, 98.0])
            else:
                vmin, vmax = -1.0, 1.0
        vmin, vmax = float(vmin), float(vmax)
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0

        trace_velocity = trace_velocity_px_per_s(trace.times, trace.arc_length)
        finite_vel = trace_velocity[np.isfinite(trace_velocity)]
        velocity_range = (float(np.min(finite_vel)), float(np.max(finite_vel))) if finite_vel.size else (0.0, 1.0)
        arc_length_max = float(np.nanmax(trace.arc_length)) if trace.arc_length.size else 0.0
        time_range = (float(ctx.times[0]), float(ctx.times[-1])) if ctx.times.size else (0.0, 1.0)

        self.tip_video_export_button.setEnabled(False)
        self.tip_video_progress_bar.setMaximum(max(1, len(ctx.frame_indices)))
        self.tip_video_progress_bar.setValue(0)
        self.tip_video_status.setText(
            f"Rendering {len(ctx.frame_indices)} frame(s) to \"{os.path.basename(path)}\"… this can "
            "take a while (one full isocontour-plus-plot figure per frame)."
        )

        self._tip_video_thread = QThread()
        self._tip_video_worker = _TipVideoExportWorker(
            path, self._dataset, field_key, field_label, ctx.frame_indices, ctx.times,
            self._dataset.xs, self._dataset.ys, path_positions, path_arc_length, trace, vmin, vmax,
            time_range, arc_length_max, velocity_range, self.tip_video_fps_spin.value(),
            pixel_size_mm=(self.pixel_size_spin.value() or None),
            uncertainty_sigma_n=self._current_uncertainty_sigma_n(),
            extra_traces=extra_traces, field_fn=field_fn,
        )
        self._tip_video_worker.moveToThread(self._tip_video_thread)
        self._tip_video_thread.started.connect(self._tip_video_worker.run)
        self._tip_video_worker.progress.connect(self._on_tip_video_export_progress)
        self._tip_video_worker.finished.connect(self._on_tip_video_export_finished)
        self._tip_video_worker.failed.connect(self._on_tip_video_export_failed)
        self._tip_video_worker.finished.connect(self._tip_video_thread.quit)
        self._tip_video_worker.failed.connect(self._tip_video_thread.quit)
        self._tip_video_thread.start()

    def _on_tip_video_export_progress(self, i, total):
        self.tip_video_progress_bar.setMaximum(max(1, total))
        self.tip_video_progress_bar.setValue(i)

    def _on_tip_video_export_finished(self, path, n_frames):
        self._sync_tip_video_export_enabled()
        self.tip_video_status.setText(f"Exported {n_frames} frame(s) to {os.path.basename(path)}.")

    def _on_tip_video_export_failed(self, message):
        self._sync_tip_video_export_enabled()
        self.tip_video_status.setText(f"Video export failed: {message}")

    # -- smoothing of the tracked tip positions ---------------------------------

    def _tip_smoothing_settings(self):
        return (self.tip_smooth_method_combo.currentData() or "none",
                int(self.tip_smooth_window_spin.value()),
                self.tip_smooth_monotonic_check.isChecked())

    def _tip_smoothing_active(self) -> bool:
        if not hasattr(self, "tip_smooth_method_combo"):
            return False                      # the Crack Tip Detection card isn't built yet
        method, _window, monotonic = self._tip_smoothing_settings()
        return method != "none" or monotonic

    def _tip_smoothed_trace(self, row):
        """The trace at `row` of `self._tip_roi_traces` with its tip positions smoothed per the
        controls under the arc-length plot -- a COPY (the raw trace is never modified), positions
        re-projected onto that trace's traced path. Returns the trace itself when no smoothing is
        selected, so `result is trace` tells the caller whether anything was smoothed."""
        trace = self._tip_roi_traces[row]
        if not self._tip_smoothing_active():
            return trace
        method, window, monotonic = self._tip_smoothing_settings()
        path_positions, path_arc_length = self._tip_roi_path_geometry[row]
        return smooth_tip_trace(trace, path_positions, path_arc_length, method, window, monotonic)

    def _tip_smoothing_used_for_k(self) -> bool:
        return self._tip_smoothing_active() and self.tip_smooth_use_for_k_check.isChecked()

    def _on_tip_smoothing_changed(self, *_args):
        self._refresh_tip_verify_arc_length_plot()
        if not self._tip_smoothing_active():
            self.tip_smooth_status.setText("")
            return
        rows = self._tip_verify_ticked_rows()
        shifts = []
        for row in rows:
            raw = self._tip_roi_traces[row].arc_length
            sm = self._tip_smoothed_trace(row).arc_length
            d = (sm - raw)[np.isfinite(sm) & np.isfinite(raw)]
            if d.size:
                shifts.append(d)
        method, window, monotonic = self._tip_smoothing_settings()
        text = smoothing_summary(method, window, monotonic).capitalize() + "."
        if shifts:
            d = np.concatenate(shifts)
            unit = "px"
            scale = 1.0
            pixel_size = self.pixel_size_spin.value() or None
            if pixel_size:
                unit, scale = "mm", pixel_size
            text += (f" Shown traces: rms change {np.sqrt(np.mean(d ** 2)) * scale:.3g} {unit}, "
                     f"max {np.max(np.abs(d)) * scale:.3g} {unit} versus the raw positions.")
        if self.tip_smooth_use_for_k_check.isChecked():
            text += " The K_I/K_II calculations below use these smoothed positions."
        else:
            text += " Display only -- the K_I/K_II calculations still use the raw positions."
        self.tip_smooth_status.setText(text)

    def _refresh_tip_verify_arc_length_plot(self):
        """Arc-length position vs time (left axis, solid), plus each
        trace's own LOCAL propagation speed vs time on a shared secondary
        axis (right, dashed) -- `trace_velocity_px_per_s`, the same
        central-difference derivative "Export all to CSV..."'s
        `velocity_px_per_s` column already reports, so the two always
        agree. Every "opening_jump" trace currently in the list above FOR
        THIS crack/leaf is drawn, so the trajectory (and now its speed)
        can be checked at a glance.

        Plotted in mm / mm-per-s whenever a pixel size (mm/px) is set in
        "Pixel size" above (the same field the CSV/video export already
        key off of) -- falls back to px / px-per-s when it's 0/unset, same
        convention as `_TipVideoExportWorker.run()`.

        When the "Uncertainty envelope" dropdown above isn't "Off", any
        trace carrying a per-frame `position_uncertainty_px` (i.e. added
        with "Estimate measurement uncertainty" checked) also gets a
        shaded +/- N sigma band around BOTH curves: the position curve
        (left axis) directly from `position_uncertainty_px`, and the
        local-speed curve (right axis) from `trace_velocity_uncertainty_
        px_per_s` -- the same finite-difference formula `trace_velocity_
        px_per_s` itself uses, applied to the position uncertainty via
        ordinary error propagation, so the two bands come from one
        consistent model rather than two independently-tuned ones. Traces
        without an uncertainty estimate are drawn plainly, with no band on
        either curve."""
        self.tip_verify_arc_length_canvas.clear()
        fig = self.tip_verify_arc_length_canvas.figure
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        leaf_index = self.tip_leaf_combo.currentData()
        pixel_size_mm = self.pixel_size_spin.value() or None
        sigma_n = self._current_uncertainty_sigma_n()
        if pixel_size_mm:
            position_axis_label = "distance along path (mm)"
            velocity_axis_label = "local speed (mm/s, dashed)"
        else:
            position_axis_label = "distance along path (px)"
            velocity_axis_label = "local speed (px/s, dashed)"
        any_plotted = False
        any_band_plotted = False
        lines, labels = [], []
        for i in self._tip_verify_ticked_rows():
            raw_trace = self._tip_roi_traces[i]
            # With a smoothing selected, the solid curve and the local speed come from the SMOOTHED
            # positions and the raw detections stay visible as faint dots underneath.
            trace = self._tip_smoothed_trace(i)
            is_smoothed = trace is not raw_trace
            any_plotted = True
            if is_smoothed:
                raw_scaled = raw_trace.arc_length * pixel_size_mm if pixel_size_mm else raw_trace.arc_length
                ax.plot(raw_trace.times, raw_scaled, "o", color=trace.color, markersize=2.5,
                        alpha=0.35, linestyle="none", label="_nolegend_", zorder=2)
            arc_length = trace.arc_length * pixel_size_mm if pixel_size_mm else trace.arc_length
            unc_px = getattr(trace, "position_uncertainty_px", None)
            has_uncertainty_band = sigma_n > 0 and unc_px is not None and np.all(np.isfinite(unc_px))
            if has_uncertainty_band:
                unc_scaled = unc_px * pixel_size_mm if pixel_size_mm else unc_px
                ax.fill_between(trace.times, arc_length - sigma_n * unc_scaled,
                                 arc_length + sigma_n * unc_scaled,
                                 color=trace.color, alpha=0.18, linewidth=0, zorder=1)
                any_band_plotted = True
            pos_line, = ax.plot(trace.times, arc_length, "o-" if not is_smoothed else "-",
                                 color=trace.color, markersize=3, linewidth=1.2 if not is_smoothed else 1.8,
                                 label=f"{trace.label} -- position" + (" (smoothed)" if is_smoothed else ""),
                                 zorder=3)
            lines.append(pos_line)
            labels.append(pos_line.get_label())
            if trace.times.size >= 2:
                velocity = trace_velocity_px_per_s(trace.times, trace.arc_length)
                if pixel_size_mm:
                    velocity = velocity * pixel_size_mm
                finite = np.isfinite(velocity)
                if np.any(finite):
                    if has_uncertainty_band:
                        # Propagated through the SAME finite-difference
                        # formula trace_velocity_px_per_s itself uses
                        # (see that function's own docstring), so this
                        # band and the position band above come from
                        # one consistent measurement-uncertainty model,
                        # not two independently-tuned ones.
                        vel_unc = trace_velocity_uncertainty_px_per_s(trace.times, unc_px)
                        if pixel_size_mm:
                            vel_unc = vel_unc * pixel_size_mm
                        band_finite = finite & np.isfinite(vel_unc)
                        if np.any(band_finite):
                            ax2.fill_between(
                                trace.times[band_finite],
                                velocity[band_finite] - sigma_n * vel_unc[band_finite],
                                velocity[band_finite] + sigma_n * vel_unc[band_finite],
                                color=trace.color, alpha=0.12, linewidth=0, zorder=1,
                            )
                            any_band_plotted = True
                    vel_line, = ax2.plot(trace.times[finite], velocity[finite], "--",
                                          color=trace.color, linewidth=1.0, alpha=0.55,
                                          label=f"{trace.label} -- speed", zorder=3)
                    lines.append(vel_line)
                    labels.append(vel_line.get_label())
        ax.set_xlabel("time (s)")
        ax.set_ylabel(position_axis_label)
        ax2.set_ylabel(velocity_axis_label)
        if any_plotted:
            ax.legend(lines, labels, fontsize=7, loc="best")
            if any_band_plotted:
                ax.text(0.02, 0.02, f"shaded bands: +/- {sigma_n} sigma uncertainty on "
                        "position (left axis) and local speed (right axis)",
                        transform=ax.transAxes, fontsize=6, alpha=0.7, va="bottom")
        else:
            ax.text(0.5, 0.5, "No trace ticked in \"Show trace\" (or none detected yet)", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9)
        self.tip_verify_arc_length_canvas.refresh()

    def _force_origin_visible(self, ax):
        """Extends (never shrinks) `ax`'s x/y autoscaled limits so the
        origin (x=0, y=0) stays within view -- used by the X-FEM-DIC K
        plot so a K_I curve's sign (K_I <= 0 has no physical meaning,
        unlike K_II -- see `positive_value_mean_and_domain_bounds`, also
        used by the per-frame status text's own "mean of positive values"
        summary) is always readable against a visible x=0/y=0 reference."""
        x0, x1 = ax.get_xlim()
        ax.set_xlim(min(0.0, x0), max(x1, 0.0))
        y0, y1 = ax.get_ylim()
        ax.set_ylim(min(0.0, y0), max(y1, 0.0))

    # -- X-FEM-DIC sub-tab: handlers ---------------------------------------

    def _refresh_xfem_trace_lists(self):
        if not hasattr(self, "xfem_tip_traces_list"):
            return  # this tab isn't built yet (called during __init__)
        self._populate_trace_checklists(self.xfem_tip_traces_list, self.xfem_wave_traces_list)

    def _xfem_get_frame_fn(self):
        """Closure: dataset-frame-index -> preprocessed uint8 image, for
        core.xfem_dic.compute_xfem_dic_time_series's `get_frame` --
        reuses this page's OWN already-loaded frame source, via the exact
        same raw-frame-index lookup / to_display_uint8 conversion
        `_refresh_tracing_frame` uses for this page's own tracing
        background, so the raw images X-FEM-DIC correlates are pixel-for-
        pixel the same ones already shown/traced on the first sub-tab --
        no separate image loading needed for this method."""
        dataset = self._dataset
        frame_source = self._frame_source

        def get_frame(source_idx):
            raw_index = dataset.frames[source_idx].frame_index
            return to_display_uint8(frame_source.get_frame(raw_index))

        return get_frame

    def _xfem_k_field_kwargs(self) -> dict:
        """compute_xfem_dic_time_series kwargs for the "K from the fitted field" option. When it is on,
        the singular term is not fitted, so r_min/r_max (normally the singular term's window) become the
        K ring window; when off, returns {} and the legacy joint fit is unchanged."""
        if not self.xfem_k_from_field_check.isChecked():
            return {}
        r_min = float(self.xfem_k_rmin_spin.value())
        r_max = float(self.xfem_k_rmax_spin.value())
        if r_max <= r_min:
            r_max = r_min + 1.0
        return dict(
            k_from_field=True, k_field_method=self.xfem_k_field_method_combo.currentData() or "ring_affine",
            k_tip_offset_px=float(self.xfem_k_offset_spin.value()), r_min_px=r_min, r_max_px=r_max,
        )

    def _xfem_tip_detection_mode(self) -> str:
        """Which auto-detect search algorithm the Auto-detected tip
        position card's radio buttons currently select -- "residual" or
        "heaviside_jump", passed straight through as
        compute_xfem_dic_time_series's `tip_detection_mode`."""
        if self.xfem_tip_mode_heaviside_radio.isChecked():
            return "heaviside_jump"
        return "residual"

    def _xfem_warm_start_fn(self):
        """Closure passed as `warm_start_from_dataset` -- samples the
        already-exported subset-DIC u/v field (the same field a standard
        near-tip ring fit reads) at an arbitrary (x, y), per source frame
        index, to seed the X-FEM-DIC background field's nodal DOFs (see
        core/xfem_dic.py's `fit_xfem_dic_frame` docstring on why this
        matters for convergence). None when the checkbox is off. Each
        frame's interpolator is built once and cached -- `initial_uv_fn`
        is called once per mesh NODE, so rebuilding it every call would
        be needlessly expensive for a mesh with more than a handful of
        nodes."""
        if not self.xfem_warm_start_check.isChecked():
            return None
        dataset = self._dataset
        cache = {}

        def warm_start(source_idx, x, y):
            if source_idx not in cache:
                frame = dataset.frames[source_idx]
                cache[source_idx] = (
                    RegularGridInterpolator((dataset.ys, dataset.xs), frame.u,
                                             bounds_error=False, fill_value=np.nan),
                    RegularGridInterpolator((dataset.ys, dataset.xs), frame.v,
                                             bounds_error=False, fill_value=np.nan),
                )
            interp_u, interp_v = cache[source_idx]
            return float(interp_u([[y, x]])[0]), float(interp_v([[y, x]])[0])

        return warm_start

    def _xfem_build_tip_results(self):
        """(entries, errors) -- `entries` is a list of {"label",
        "tip_result"} for every ticked trace (ridge and/or wave) in this
        tab's "Crack path source" card, exactly mirroring
        `_on_compute_k1c`'s own ticked-row gathering (same messages, so
        the two cards behave identically for the same input). `errors` is
        a list of blocking/per-trace problem strings -- if `entries` ends
        up empty, `errors[0]` (when present) is shown to the user as-is."""
        checked_tip_rows = [
            row for row in range(self.xfem_tip_traces_list.count())
            if self.xfem_tip_traces_list.item(row).checkState() == Qt.Checked
        ]
        checked_wave_rows = [
            row for row in range(self.xfem_wave_traces_list.count())
            if self.xfem_wave_traces_list.item(row).checkState() == Qt.Checked
        ]
        if not checked_tip_rows and not checked_wave_rows:
            return [], ["Tick at least one ridge or wave trace above first."]
        pixel_size = self.pixel_size_spin.value() or None
        if not pixel_size:
            return [], [
                "X-FEM-DIC needs a pixel size (mm/px) -- set it in the Material & Calibration "
                "card on the Wave speed analysis tab, or use its \"Sync from DIC tab\" button."
            ]
        crack = self._selected_crack()
        leaf_index = self.tip_leaf_combo.currentData()
        if checked_wave_rows and (crack is None or leaf_index is None):
            return [], [
                "A wave trace is ticked -- pick the Crack/Tip it belongs to in the Crack Tip "
                "Propagation card above first (converting it to a tip track needs that crack's "
                "traced path)."
            ]
        reverse_path = self.tip_reverse_path_check.isChecked()

        entries = []
        errors = []
        use_smoothed = self._tip_smoothing_used_for_k()
        for row in checked_tip_rows:
            trace = self._tip_smoothed_trace(row) if use_smoothed else self._tip_roi_traces[row]
            path_positions, path_arc_length = self._tip_roi_path_geometry[row]
            tip_result = tip_roi_trace_to_track_result(
                trace, path_positions, path_arc_length, pixel_size_mm=pixel_size,
            )
            entries.append({
                "label": f"{trace.method}: " + trace.label + (" [smoothed]" if use_smoothed else ""),
                "tip_result": tip_result,
                "smoothing": smoothing_summary(*self._tip_smoothing_settings()) if use_smoothed
                else "raw tip positions",
            })
        for row in checked_wave_rows:
            wave = self._roi_waves[row]
            try:
                tip_result = wave_roi_trace_to_track_result(
                    wave, self._dataset, crack, leaf_index, pixel_size_mm=pixel_size,
                    reverse_path=reverse_path,
                )
            except CrackTipTrackingError as exc:
                errors.append(f"\"{wave.label}\": {exc}")
                continue
            entries.append({"label": "wave: " + wave.label, "tip_result": tip_result})
        return entries, errors

    def _on_xfem_suggest_settings_clicked(self):
        """Pre-fills element size / enrichment radius / region half-width /
        sample spacing from the ticked trace(s)' own tracked crack length
        and, when the matching image sequence is loaded, the reference
        frame's own dimensions -- see core.xfem_dic.recommend_mesh_parameters
        (sample spacing is derived from the suggested element size itself,
        not from the crack length directly). Only ever pre-fills the
        existing spin boxes (still fully editable afterward), never runs
        anything."""
        if self._dataset is None or not self._dataset.frames:
            self.xfem_run_status.setText("Load a DIC results CSV above first.")
            return
        entries, errors = self._xfem_build_tip_results()
        if not entries:
            self.xfem_run_status.setText(errors[0] if errors else "Nothing to suggest settings from.")
            return

        pixel_size = self.pixel_size_spin.value() or None
        max_s_px = 0.0
        for entry in entries:
            s = np.asarray(entry["tip_result"].arc_length_s, dtype=float)
            finite = s[np.isfinite(s)]
            if finite.size:
                max_s_px = max(max_s_px, float(finite.max()))

        image_w = image_h = None
        if self._frame_source is not None:
            try:
                ref_image = self._xfem_get_frame_fn()(0)
                image_h, image_w = ref_image.shape[:2]
            except Exception:  # noqa: BLE001 - this suggestion is a nice-to-have, never fatal
                pass

        rec = recommend_mesh_parameters(pixel_size, max_s_px, image_width_px=image_w, image_height_px=image_h)
        self.xfem_element_size_spin.setValue(rec["element_size_px"])
        self.xfem_r_enrich_spin.setValue(rec["r_enrich_px"])
        self.xfem_roi_half_width_spin.setValue(rec["roi_half_width_px"])
        self.xfem_sample_step_spin.setValue(rec["sample_step_px"])

        mm_note = ""
        if "r_enrich_mm" in rec:
            mm_note = (
                f" (~{rec['element_size_mm']:.3g} / {rec['r_enrich_mm']:.3g} / "
                f"{rec['roi_half_width_mm']:.3g} / {rec['sample_step_mm']:.3g} mm)"
            )
        distance_note = f"max tracked crack distance {max_s_px:.0f} px" if max_s_px > 0 else \
            "no tracked crack distance yet -- using a generic fallback"
        self.xfem_run_status.setText(
            f"Suggested from {distance_note}: element size {rec['element_size_px']:.0f} px, "
            f"enrichment radius {rec['r_enrich_px']:.0f} px, region half-width "
            f"{rec['roi_half_width_px']:.0f} px, sample spacing {rec['sample_step_px']:.1f} "
            f"px{mm_note}. Still editable -- check with \"Preview mesh\" below."
        )

    def _on_xfem_suggest_search_radius_clicked(self):
        """Pre-fills the auto-detect \"Search radius\" from the ticked
        trace's own tracked arc-length spread -- see
        core.xfem_dic.recommend_tip_search_radius_px's own docstring for
        the full reasoning (in short: since auto_detect_tip re-anchors
        every frame independently to that frame's own nominal position, a
        persistent nominal-vs-true offset has to be re-bridged on EVERY
        frame, and a too-small radius shows up as the auto-detected track
        visibly plateauing while the nominal one keeps climbing). Only
        ever pre-fills the existing, still-editable spin box -- never runs
        anything, and never a substitute for actually checking the
        resulting arc-length-vs-time plot afterward."""
        if self._dataset is None or not self._dataset.frames:
            self.xfem_run_status.setText("Load a DIC results CSV above first.")
            return
        entries, errors = self._xfem_build_tip_results()
        if not entries:
            self.xfem_run_status.setText(errors[0] if errors else "Nothing to suggest a search radius from.")
            return

        pixel_size = self.pixel_size_spin.value() or None
        # Same "first ticked trace" convention as preview/mesh-preview above
        # -- when several traces are ticked, this pre-fills from the first
        # one; still fully editable afterward if another trace needs more.
        arc_length_s = np.asarray(entries[0]["tip_result"].arc_length_s, dtype=float)
        rec = recommend_tip_search_radius_px(arc_length_s, pixel_size_mm=pixel_size)
        self.xfem_tip_search_radius_spin.setValue(rec["search_radius_px"])

        mm_note = ""
        if "search_radius_mm" in rec:
            mm_note = f" (~{rec['search_radius_mm']:.3g} mm)"
        span_note = (
            f"observed arc-length spread {rec['observed_span_px']:.0f} px"
            if rec["observed_span_px"] > 0 else
            "no usable tracked spread yet -- using a generic fallback"
        )
        self.xfem_run_status.setText(
            f"Suggested search radius from \"{entries[0]['label']}\": {span_note} -> "
            f"{rec['search_radius_px']:.0f} px{mm_note}. Still editable -- if the auto-detected "
            f"arc-length-vs-time curve still plateaus below the nominal one afterward, that points "
            f"at a genuine disagreement between the tracking method and the image data, not just a "
            f"too-small radius."
        )

    def _on_xfem_mesh_preview_clicked(self):
        """Draws the exact Q4 mesh `compute_xfem_dic_time_series` would
        build for the current settings/ticked-trace's preview frame, over
        the reference image -- no correlation/fit is run, so this is
        cheap and safe to click repeatedly while tuning the settings
        above. Reuses `build_tip_centered_mesh`/`classify_mesh_enrichment`
        directly (the same functions the production fit path calls) so
        the picture shown here can never disagree with what a real run
        would actually do."""
        if self._dataset is None or not self._dataset.frames:
            self.xfem_mesh_preview_status.setText("Load a DIC results CSV above first.")
            return
        entries, errors = self._xfem_build_tip_results()
        if not entries:
            self.xfem_mesh_preview_status.setText(errors[0] if errors else "Nothing to preview.")
            return
        if self._frame_source is None:
            self.xfem_mesh_preview_status.setText(
                "Load the matching video/image sequence in the Data card above first."
            )
            return

        entry = entries[0]  # same "first ticked trace" convention as the fit preview above
        tip_result = entry["tip_result"]
        n = len(tip_result.times)
        self.xfem_preview_frame_spin.setRange(0, max(0, n - 1))
        frame_order_index = min(self.xfem_preview_frame_spin.value(), max(0, n - 1))

        arc_s = np.asarray(tip_result.arc_length_s, dtype=float)
        idx = frame_order_index
        if not np.isfinite(arc_s[idx]):
            finite_idx = np.flatnonzero(np.isfinite(arc_s))
            if finite_idx.size == 0:
                self.xfem_mesh_preview_status.setText(
                    f"\"{entry['label']}\" has no detected tip positions to preview."
                )
                return
            idx = int(finite_idx[np.argmin(np.abs(finite_idx - frame_order_index))])

        tip_xy = tip_result.positions_xy[idx]
        tip_s = float(arc_s[idx])
        try:
            ref_image = self._xfem_get_frame_fn()(0)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the UI
            self.xfem_mesh_preview_status.setText(f"Could not load the reference frame: {exc}")
            return

        element_size_px = self.xfem_element_size_spin.value()
        r_enrich_px = self.xfem_r_enrich_spin.value()
        roi_half_width_px = self.xfem_roi_half_width_spin.value()
        try:
            if self.xfem_graded_mesh_check.isChecked():
                mesh = build_graded_tip_centered_mesh(tip_xy, roi_half_width_px, element_size_px,
                                                       r_enrich_px)
            else:
                mesh = build_tip_centered_mesh(tip_xy, roi_half_width_px, element_size_px)
        except XfemDicError as exc:
            self.xfem_mesh_preview_status.setText(f"Mesh preview failed: {exc}")
            return
        node_enriched, element_split = classify_mesh_enrichment(
            mesh, tip_result.path_positions, tip_result.path_arc_length, tip_s,
        )

        self._draw_xfem_mesh_preview(
            ref_image, mesh, element_split, tip_xy, r_enrich_px, tip_result.path_positions,
            entry["label"], idx, n,
        )
        n_active = int(mesh.element_active.sum())
        n_split = int(element_split.sum())
        self.xfem_mesh_preview_status.setText(
            f"{entry['label']}, frame {idx + 1}/{n}: {mesh.n_elem_x}x{mesh.n_elem_y} element mesh "
            f"({n_active} active, {n_split} crack-split/Heaviside-enriched), "
            f"{int(node_enriched.sum())} enriched node(s)."
        )

    def _draw_xfem_mesh_preview(self, ref_image, mesh, element_split, tip_xy, r_enrich_px,
                                 path_xy, label, frame_order_index, n_frames):
        self.xfem_mesh_canvas.clear()
        ax = self.xfem_mesh_canvas.figure.add_subplot(111)

        h, w = ref_image.shape[:2]
        cmap = "gray" if ref_image.ndim == 2 else None
        ax.imshow(ref_image, cmap=cmap, origin="upper", extent=[0, w, h, 0])

        # Drawn straight from the mesh's own edge arrays (mesh.x_edges/
        # y_edges), not `x0 + i*element_size_px` -- correct whether the
        # mesh is uniform or graded (see build_graded_tip_centered_mesh),
        # since a graded mesh's node columns/rows aren't evenly spaced.
        # Bright orange, opaque and reasonably thick: a faint or cyan grid
        # would all but disappear against a real black-and-white speckle
        # pattern -- orange reads clearly against both the light and dark
        # speckles.
        x0, y0 = mesh.x0, mesh.y0
        x_max, y_max = float(mesh.x_edges[-1]), float(mesh.y_edges[-1])
        for x in mesh.x_edges:
            ax.plot([x, x], [y0, y_max], color="orange", linewidth=0.7, alpha=0.85)
        for y in mesh.y_edges:
            ax.plot([x0, x_max], [y, y], color="orange", linewidth=0.7, alpha=0.85)

        for ey in range(mesh.n_elem_y):
            for ex in range(mesh.n_elem_x):
                if element_split[ey, ex]:
                    ew, eh = mesh.element_size_xy(ex, ey)
                    ax.add_patch(Rectangle(
                        (mesh.x_edges[ex], mesh.y_edges[ey]), ew, eh,
                        # Magenta, not orange, so crack-split (Heaviside-
                        # enriched) elements stay visually distinct from
                        # the orange grid lines rather than blending
                        # into them.
                        facecolor="magenta", alpha=0.35, edgecolor="none",
                    ))

        if len(path_xy) >= 2:
            path_xy = np.asarray(path_xy)
            ax.plot(path_xy[:, 0], path_xy[:, 1], color="lime", linewidth=1.3, label="Crack path")
        ax.plot([tip_xy[0]], [tip_xy[1]], marker="+", color="red", markersize=14,
                markeredgewidth=2.2, linestyle="none", label="Tip")
        ax.add_patch(Circle(
            (tip_xy[0], tip_xy[1]), r_enrich_px, fill=False, edgecolor="yellow",
            linewidth=1.3, linestyle="--", label="Enrichment radius",
        ))

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{label} -- frame {frame_order_index + 1}/{n_frames}", fontsize=9)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
        self.xfem_mesh_canvas.refresh()

    def _on_xfem_preview_clicked(self):
        if self._dataset is None or not self._dataset.frames:
            self.xfem_run_status.setText("Load a DIC results CSV above first.")
            return
        entries, errors = self._xfem_build_tip_results()
        if not entries:
            self.xfem_run_status.setText(errors[0] if errors else "Nothing to preview.")
            return
        if self._frame_source is None:
            self.xfem_run_status.setText(
                "Load the matching video/image sequence in the Data card above first."
            )
            return

        entry = entries[0]  # preview always uses the FIRST ticked trace -- see the Run card's own text
        tip_result = entry["tip_result"]
        n = len(tip_result.times)
        self.xfem_preview_frame_spin.setRange(0, max(0, n - 1))
        frame_order_index = min(self.xfem_preview_frame_spin.value(), max(0, n - 1))

        material = self.material_properties()
        pixel_size = self.pixel_size_spin.value() or None
        self.xfem_run_status.setText(f"Fitting {entry['label']}, frame {frame_order_index + 1}/{n}…")
        base_kwargs = dict(
            tip_result=tip_result, get_frame=self._xfem_get_frame_fn(), reference_source_frame_index=0,
            young_modulus_pa=material["young_modulus_pa"], poisson_ratio=material["poisson_ratio"],
            plane_stress=material["plane_stress"], pixel_size_mm=pixel_size,
            element_size_px=self.xfem_element_size_spin.value(),
            r_enrich_px=self.xfem_r_enrich_spin.value(),
            r_min_px=self.xfem_r_min_spin.value(),
            r_max_px=(self.xfem_r_max_spin.value() or None),
            max_iterations=self.xfem_max_iterations_spin.value(),
            position_window_px=(self.xfem_position_window_spin.value() or None),
            roi_half_width_px=self.xfem_roi_half_width_spin.value(),
            sample_step_px=self.xfem_sample_step_spin.value(),
            frame_order_indices=[frame_order_index],
            warm_start_from_dataset=self._xfem_warm_start_fn(),
            graded_mesh=self.xfem_graded_mesh_check.isChecked(),
            estimate_uncertainty=self.xfem_estimate_uncertainty_check.isChecked(),
        )
        base_kwargs.update(self._xfem_k_field_kwargs())
        try:
            series = compute_xfem_dic_time_series(**base_kwargs)
            new_results = [{"label": entry["label"], "tip_result": tip_result, "series": series}]
            # "Also compute with an auto-detected tip position" -- runs the
            # SAME frame a second time with auto_detect_tip=True and adds
            # it as a second, separately-labeled entry, so it plots
            # alongside (never replacing) the fixed-trace result -- see
            # the Auto-detected tip position card's own help text.
            if self.xfem_auto_tip_check.isChecked():
                series_auto = compute_xfem_dic_time_series(
                    **base_kwargs, auto_detect_tip=True,
                    tip_search_radius_px=self.xfem_tip_search_radius_spin.value(),
                    tip_search_step_px=self.xfem_tip_search_step_spin.value(),
                    tip_detection_mode=self._xfem_tip_detection_mode(),
                    heaviside_jump_threshold_px=self.xfem_tip_jump_threshold_spin.value(),
                )
                new_results.append({
                    "label": entry["label"] + " [auto tip]", "tip_result": tip_result,
                    "series": series_auto,
                })
        except (XfemDicError, CrackTipTrackingError) as exc:
            self.xfem_run_status.setText(f"Preview failed: {exc}")
            return

        self._xfem_results = [e for e in new_results] + self._direct_k_entries_kept()
        self._xfem_skipped_traces = errors
        self._xfem_after_results_updated()

    def _on_xfem_run_clicked(self):
        if self._dataset is None or not self._dataset.frames:
            self.xfem_run_status.setText("Load a DIC results CSV above first.")
            return
        entries, errors = self._xfem_build_tip_results()
        if not entries:
            self.xfem_run_status.setText(errors[0] if errors else "Nothing to run.")
            return
        if self._frame_source is None:
            self.xfem_run_status.setText(
                "Load the matching video/image sequence in the Data card above first."
            )
            return

        material = self.material_properties()
        pixel_size = self.pixel_size_spin.value() or None
        get_frame = self._xfem_get_frame_fn()
        warm_start = self._xfem_warm_start_fn()
        auto_tip_enabled = self.xfem_auto_tip_check.isChecked()
        search_radius = self.xfem_tip_search_radius_spin.value()
        search_step = self.xfem_tip_search_step_spin.value()
        tip_detection_mode = self._xfem_tip_detection_mode()
        jump_threshold = self.xfem_tip_jump_threshold_spin.value()

        runs = []
        pending_tip_results = {}
        for e in entries:
            base_kwargs = dict(
                tip_result=e["tip_result"], get_frame=get_frame, reference_source_frame_index=0,
                young_modulus_pa=material["young_modulus_pa"], poisson_ratio=material["poisson_ratio"],
                plane_stress=material["plane_stress"], pixel_size_mm=pixel_size,
                element_size_px=self.xfem_element_size_spin.value(),
                r_enrich_px=self.xfem_r_enrich_spin.value(),
                r_min_px=self.xfem_r_min_spin.value(),
                r_max_px=(self.xfem_r_max_spin.value() or None),
                max_iterations=self.xfem_max_iterations_spin.value(),
                position_window_px=(self.xfem_position_window_spin.value() or None),
                roi_half_width_px=self.xfem_roi_half_width_spin.value(),
                sample_step_px=self.xfem_sample_step_spin.value(),
                warm_start_from_dataset=warm_start,
                graded_mesh=self.xfem_graded_mesh_check.isChecked(),
                estimate_uncertainty=self.xfem_estimate_uncertainty_check.isChecked(),
            )
            base_kwargs.update(self._xfem_k_field_kwargs())
            runs.append({"label": e["label"], "kwargs": dict(base_kwargs)})
            pending_tip_results[e["label"]] = e["tip_result"]
            # Same "compute both, never replace" convention as the preview
            # button above -- see the Auto-detected tip position card.
            if auto_tip_enabled:
                auto_label = e["label"] + " [auto tip]"
                runs.append({
                    "label": auto_label,
                    "kwargs": dict(base_kwargs, auto_detect_tip=True,
                                   tip_search_radius_px=search_radius, tip_search_step_px=search_step,
                                   tip_detection_mode=tip_detection_mode,
                                   heaviside_jump_threshold_px=jump_threshold),
                })
                pending_tip_results[auto_label] = e["tip_result"]
        self._xfem_pending_tip_results = pending_tip_results
        self._xfem_pretick_errors = errors

        self._set_xfem_run_controls_enabled(False)
        self.xfem_progress_bar.setMaximum(1)
        self.xfem_progress_bar.setValue(0)
        self.xfem_run_status.setText(
            f"Running X-FEM-DIC on {len(runs)} trace(s)… this can take a while (raw-image "
            f"Gauss-Newton per frame, much heavier than a standard near-tip ring fit)."
        )

        self._xfem_cancel_event = threading.Event()
        self._xfem_thread = QThread()
        self._xfem_worker = _XfemDicWorker(runs, cancel_event=self._xfem_cancel_event)
        self._xfem_worker.moveToThread(self._xfem_thread)
        self._xfem_thread.started.connect(self._xfem_worker.run)
        self._xfem_worker.progress.connect(self._on_xfem_run_progress)
        self._xfem_worker.finished.connect(self._on_xfem_run_finished)
        self._xfem_worker.failed.connect(self._on_xfem_run_failed)
        self._xfem_worker.cancelled.connect(self._on_xfem_run_cancelled)
        self._xfem_worker.finished.connect(self._xfem_thread.quit)
        self._xfem_worker.failed.connect(self._xfem_thread.quit)
        self._xfem_worker.cancelled.connect(self._xfem_thread.quit)
        self._xfem_thread.start()

    def _direct_k_entries_kept(self):
        return [e for e in self._xfem_results if e.get("method") == "direct"]

    def _on_direct_k_run_clicked(self):
        """K_I/K_II from the MEASURED displacement field of every ticked trace (see
        core/direct_k.py). Quick enough to run on the UI thread (a wait cursor is shown); results are
        merged into the shared Results card next to any X-FEM-DIC traces, replacing only the previous
        direct results."""
        if self._dataset is None or not self._dataset.frames:
            self.direct_k_status.setText("Load a DIC results CSV above first.")
            return
        entries, errors = self._xfem_build_tip_results()
        if not entries:
            self.direct_k_status.setText(errors[0] if errors else "Nothing to compute.")
            return
        material = self.material_properties()
        pixel_size = self.pixel_size_spin.value() or None
        method = self.direct_k_method_combo.currentData()
        r_min = self.direct_k_rmin_spin.value()
        r_max = self.direct_k_rmax_spin.value()
        tangent_window = self.direct_k_tangent_spin.value() or None
        new_entries = []
        problems = list(errors)
        self.direct_k_run_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for n_done, entry in enumerate(entries):
                self.direct_k_status.setText(f"Computing {n_done + 1}/{len(entries)}: {entry['label']} ...")
                QApplication.processEvents()
                try:
                    series = compute_direct_k_series(
                        self._dataset, entry["tip_result"],
                        young_modulus_pa=material["young_modulus_pa"], poisson_ratio=material["poisson_ratio"],
                        plane_stress=material["plane_stress"], pixel_size_mm=pixel_size,
                        r_min_px=r_min, r_max_px=r_max, method=method,
                        flank_exclusion_deg=self.direct_k_flank_spin.value(),
                        tangent_window_px=tangent_window, smoothing=entry.get("smoothing", "raw tip positions"),
                        tip_offset_px=self.direct_k_offset_spin.value(),
                        trim_after_arrest=self.direct_k_trim_check.isChecked(),
                    )
                except DirectKError as exc:
                    problems.append(f"\"{entry['label']}\": {exc}")
                    continue
                new_entries.append({
                    "label": f"{entry['label']} [direct: {method_label(method)}]",
                    "tip_result": entry["tip_result"], "series": series, "method": "direct",
                })
        finally:
            QApplication.restoreOverrideCursor()
            self.direct_k_run_button.setEnabled(True)
        if not new_entries:
            self.direct_k_status.setText("Nothing could be computed -- " + "; ".join(problems))
            return
        kept = [e for e in self._xfem_results if e.get("method") != "direct"]
        self._xfem_results = kept + new_entries
        self._xfem_skipped_traces = problems
        n_ok = sum(1 for e in new_entries for s_ in e["series"].samples if s_.fit.K_I_pa_sqrt_m is not None)
        n_all = sum(len(e["series"].samples) for e in new_entries)
        n_trim = sum(e["series"].n_frames_after_arrest for e in new_entries)
        self.direct_k_status.setText(
            f"Done: {len(new_entries)} trace(s), {n_ok}/{n_all} frame(s) fitted ({method_label(method)}; "
            f"{new_entries[0]['series'].smoothing}"
            + (f"; tip offset {new_entries[0]['series'].tip_offset_px:+.1f} px" if new_entries[0]['series'].tip_offset_px else "")
            + ")."
            + (f" {n_trim} frame(s) after the crack stopped were ignored." if n_trim else "")
            + (f" Skipped: {'; '.join(problems)}" if problems else "")
        )
        self._xfem_after_results_updated()

    def _on_direct_k_scan_clicked(self):
        """Median K_I versus assumed tip offset and r_max on the first ticked trace (see
        core.direct_k.scan_k_sensitivity). A tip detected behind the real one makes K fall as r_max
        grows; the offset that flattens the curves is the tip bias."""
        if self._dataset is None or not self._dataset.frames:
            self.direct_k_status.setText("Load a DIC results CSV above first.")
            return
        entries, errors = self._xfem_build_tip_results()
        if not entries:
            self.direct_k_status.setText(errors[0] if errors else "Nothing to scan.")
            return
        entry = entries[0]
        material = self.material_properties()
        pixel_size = self.pixel_size_spin.value() or None
        r_min = self.direct_k_rmin_spin.value()
        r_max = self.direct_k_rmax_spin.value()
        method = self.direct_k_method_combo.currentData()
        if method == "ring_t":
            method = "ring"
        r_list = sorted({round(r_max * f) for f in (0.4, 0.55, 0.75, 1.0) if r_max * f > r_min + 1})
        offsets = [float(o) for o in np.arange(-5.0, 25.01, 2.5)]
        self.direct_k_scan_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.direct_k_status.setText(f"Scanning \"{entry['label']}\" ...")
            QApplication.processEvents()
            scan = scan_k_sensitivity(
                self._dataset, entry["tip_result"], material["young_modulus_pa"], material["poisson_ratio"],
                material["plane_stress"], pixel_size, r_min, r_list, offsets, method=method,
                flank_exclusion_deg=self.direct_k_flank_spin.value())
        except DirectKError as exc:
            self.direct_k_status.setText(f"Scan failed: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.direct_k_scan_button.setEnabled(True)
        self.direct_k_scan_text.setPlainText(format_scan_table(scan))
        self._direct_k_best_offset = scan.best_offset_px
        self.direct_k_apply_best_button.setEnabled(scan.best_offset_px is not None)
        if scan.best_offset_px is None:
            self.direct_k_status.setText("Scan: no usable fit -- check pixel size, E and the radii.")
            return
        row0 = scan.median_k1[scan.offsets_px.index(0.0)] if 0.0 in scan.offsets_px else scan.median_k1[0]
        k_hi = float(row0[0]) if np.isfinite(row0[0]) else float("nan")      # smallest r_max
        k_lo = float(row0[-1]) if np.isfinite(row0[-1]) else float("nan")    # largest r_max
        self.direct_k_status.setText(
            f"Scan of \"{entry['label']}\" ({scan.method}, r_min {r_min:g} px): at offset 0, K_I goes from "
            f"{k_hi / 1e6:.2f} (r_max {r_list[0]:g}) to {k_lo / 1e6:.2f} MPa.sqrt(m) (r_max {r_list[-1]:g}). "
            f"K is most independent of r_max at offset {scan.best_offset_px:+.1f} px "
            f"(K_I = {scan.k_at_best_pa_sqrt_m / 1e6:.2f} MPa.sqrt(m), spread "
            f"{100 * float(np.nanmin(scan.spread)):.0f}%). If that spread stays large, the cause is not "
            f"the tip position (DIC smoothing, pixel size, E).")

    def _on_direct_k_window_scan_clicked(self):
        """Median K_I versus fitting annulus (r_min, r_max) on the first ticked trace, at the
        current method and tip offset (core.direct_k.scan_k_window)."""
        if self._dataset is None or not self._dataset.frames:
            self.direct_k_status.setText("Load a DIC results CSV above first.")
            return
        entries, errors = self._xfem_build_tip_results()
        if not entries:
            self.direct_k_status.setText(errors[0] if errors else "Nothing to scan.")
            return
        entry = entries[0]
        material = self.material_properties()
        pixel_size = self.pixel_size_spin.value() or None
        method = self.direct_k_method_combo.currentData()
        if method == "ring_t":
            method = "ring"
        offset = self.direct_k_offset_spin.value()
        self.direct_k_window_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.direct_k_status.setText(f"Scanning fitting windows of \"{entry['label']}\" ...")
            QApplication.processEvents()
            scan = scan_k_window(
                self._dataset, entry["tip_result"], material["young_modulus_pa"], material["poisson_ratio"],
                material["plane_stress"], pixel_size, [4, 8, 12, 16, 20, 25, 30, 40], [25, 30, 40, 50, 60, 80],
                method=method, tip_offset_px=offset, flank_exclusion_deg=self.direct_k_flank_spin.value())
        except DirectKError as exc:
            self.direct_k_status.setText(f"Scan failed: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.direct_k_window_button.setEnabled(True)
        self.direct_k_scan_text.setPlainText(format_window_table(scan))
        step = f"DIC grid step {scan.grid_step_px:g} px: " if scan.grid_step_px else ""
        self.direct_k_status.setText(
            f"Window scan of \"{entry['label']}\" ({method_label(scan.method)}, tip offset {offset:+.1f} px). "
            f"{step}the near-tip field is only resolved beyond about 2-3 grid steps / one subset size from "
            f"the tip -- look for the block of the table where K stops changing with r_min and r_max.")

    def _on_direct_k_apply_best_clicked(self):
        if self._direct_k_best_offset is not None:
            self.direct_k_offset_spin.setValue(float(self._direct_k_best_offset))

    def _on_xfem_stop_clicked(self):
        if self._xfem_cancel_event is not None:
            self._xfem_cancel_event.set()
            self.xfem_stop_button.setEnabled(False)
            self.xfem_run_status.setText(
                "Stopping… (finishing the frame/candidate fit currently in progress)"
            )

    def _set_xfem_run_controls_enabled(self, enabled: bool):
        for widget in (self.xfem_preview_button, self.xfem_run_button, self.xfem_element_size_spin,
                       self.xfem_r_enrich_spin, self.xfem_r_min_spin, self.xfem_r_max_spin,
                       self.xfem_max_iterations_spin, self.xfem_position_window_spin,
                       self.xfem_roi_half_width_spin, self.xfem_sample_step_spin,
                       self.xfem_warm_start_check, self.xfem_graded_mesh_check, self.xfem_preview_frame_spin,
                       self.xfem_estimate_uncertainty_check, self.xfem_k_from_field_check,
                       self.xfem_k_field_method_combo, self.xfem_k_rmin_spin, self.xfem_k_rmax_spin,
                       self.xfem_k_offset_spin,
                       self.xfem_tip_traces_list, self.xfem_wave_traces_list,
                       self.xfem_suggest_button, self.xfem_mesh_preview_button,
                       self.xfem_auto_tip_check, self.xfem_tip_search_radius_spin,
                       self.xfem_tip_search_step_spin, self.xfem_tip_jump_threshold_spin,
                       self.xfem_tip_mode_residual_radio, self.xfem_tip_mode_heaviside_radio,
                       self.xfem_suggest_radius_button,
                       self.xfem_field_combo, self.xfem_isocontour_button):
            widget.setEnabled(enabled)
        # Opposite sense from everything else here: only clickable WHILE a
        # run is in flight.
        self.xfem_stop_button.setEnabled(not enabled)

    def _on_xfem_run_progress(self, i, total):
        self.xfem_progress_bar.setMaximum(max(1, total))
        self.xfem_progress_bar.setValue(i)

    def _on_xfem_run_cancelled(self):
        self._xfem_cancel_event = None
        self._set_xfem_run_controls_enabled(True)
        self.xfem_run_status.setText("X-FEM-DIC run stopped by user.")

    def _on_xfem_run_finished(self, results, errors):
        self._xfem_cancel_event = None
        self._set_xfem_run_controls_enabled(True)
        all_errors = getattr(self, "_xfem_pretick_errors", []) + errors
        if not results:
            self.xfem_run_status.setText(
                "X-FEM-DIC could not be computed for any ticked trace"
                + (f" -- {'; '.join(all_errors)}" if all_errors else ".")
            )
            return
        pending = getattr(self, "_xfem_pending_tip_results", {})
        self._xfem_results = [
            {"label": r["label"], "tip_result": pending.get(r["label"]), "series": r["series"]}
            for r in results
        ] + self._direct_k_entries_kept()
        self._xfem_skipped_traces = all_errors
        self.xfem_run_status.setText(
            f"Done: {len(results)} trace(s) computed."
            + (f" Skipped: {'; '.join(all_errors)}" if all_errors else "")
        )
        self._xfem_after_results_updated()

    def _on_xfem_run_failed(self, message):
        self._xfem_cancel_event = None
        self._set_xfem_run_controls_enabled(True)
        self.xfem_run_status.setText(f"X-FEM-DIC run failed: {message}")

    def _xfem_after_results_updated(self):
        self.xfem_critical_combo.blockSignals(True)
        self.xfem_critical_combo.clear()
        for entry in self._xfem_results:
            self.xfem_critical_combo.addItem(entry["label"])
        self.xfem_critical_combo.blockSignals(False)
        self._refresh_xfem_result_traces_list()
        self._sync_xfem_frame_spin_to_selected()
        self._refresh_xfem_display()
        self._maybe_refresh_xfem_isocontours()
        self._sync_xfem_video_export_enabled()

    def _refresh_xfem_result_traces_list(self):
        """Rebuilds the "Show curves" checklist from `self._xfem_results`,
        keeping which labels were ticked (by label, not row index, so
        re-running doesn't silently re-show something the user hid) --
        same convention as `_populate_trace_checklists` elsewhere on this
        page. New entries default to ticked (shown), matching the
        previous always-show-everything behavior when nothing has been
        hidden yet."""
        checked_labels = {
            self.xfem_result_traces_list.item(row).data(Qt.UserRole)
            for row in range(self.xfem_result_traces_list.count())
            if self.xfem_result_traces_list.item(row).checkState() == Qt.Checked
        }
        known_labels = {
            self.xfem_result_traces_list.item(row).data(Qt.UserRole)
            for row in range(self.xfem_result_traces_list.count())
        }
        self.xfem_result_traces_list.blockSignals(True)
        self.xfem_result_traces_list.clear()
        for entry in self._xfem_results:
            label = entry["label"]
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setData(Qt.UserRole, label)
            # Ticked if it already was, OR if it's new (never seen before
            # in this list) -- only a label the user explicitly unticked
            # stays hidden across a refresh.
            item.setCheckState(
                Qt.Checked if (label in checked_labels or label not in known_labels) else Qt.Unchecked
            )
            self.xfem_result_traces_list.addItem(item)
        self.xfem_result_traces_list.blockSignals(False)

    def _xfem_visible_result_labels(self) -> set:
        """The set of `self._xfem_results` labels currently ticked in the
        "Show curves" list -- read by both plotting functions below so
        unticking a trace hides it from the K_I/K_II plot AND the arc-
        length-vs-time plot at once. Falls back to "show everything" if
        the list isn't built yet (e.g. called before __init__ finishes)."""
        if not hasattr(self, "xfem_result_traces_list"):
            return {entry["label"] for entry in self._xfem_results}
        return {
            self.xfem_result_traces_list.item(row).data(Qt.UserRole)
            for row in range(self.xfem_result_traces_list.count())
            if self.xfem_result_traces_list.item(row).checkState() == Qt.Checked
        }

    def _on_xfem_critical_combo_changed(self, _index=None):
        self._sync_xfem_frame_spin_to_selected()
        self._refresh_xfem_display()
        self._maybe_refresh_xfem_isocontours()
        self._sync_xfem_video_export_enabled()

    def _sync_xfem_frame_spin_to_selected(self):
        index = self.xfem_critical_combo.currentIndex()
        if not (0 <= index < len(self._xfem_results)):
            self.xfem_frame_spin.blockSignals(True)
            self.xfem_frame_spin.setRange(0, 0)
            self.xfem_frame_spin.blockSignals(False)
            self.xfem_frame_spin.setEnabled(False)
            self.xfem_frame_slider.blockSignals(True)
            self.xfem_frame_slider.setRange(0, 0)
            self.xfem_frame_slider.blockSignals(False)
            self.xfem_frame_slider.setEnabled(False)
            self._sync_xfem_isocontour_controls()
            return
        series = self._xfem_results[index]["series"]
        n = len(series.samples)
        default_index = series.default_critical_index or 0
        self.xfem_frame_spin.blockSignals(True)
        self.xfem_frame_spin.setRange(0, max(0, n - 1))
        self.xfem_frame_spin.setValue(default_index)
        self.xfem_frame_spin.blockSignals(False)
        self.xfem_frame_spin.setEnabled(n > 0)
        self.xfem_frame_slider.blockSignals(True)
        self.xfem_frame_slider.setRange(0, max(0, n - 1))
        self.xfem_frame_slider.setValue(default_index)
        self.xfem_frame_slider.blockSignals(False)
        self.xfem_frame_slider.setEnabled(n > 0)
        self._sync_xfem_isocontour_controls()

    def _on_xfem_frame_slider_changed(self, value: int):
        self.xfem_frame_spin.blockSignals(True)
        self.xfem_frame_spin.setValue(value)
        self.xfem_frame_spin.blockSignals(False)
        self._sync_xfem_isocontour_controls()
        self._on_xfem_frame_changed()

    def _on_xfem_frame_spin_changed(self, value: int):
        self.xfem_frame_slider.blockSignals(True)
        self.xfem_frame_slider.setValue(value)
        self.xfem_frame_slider.blockSignals(False)
        self._sync_xfem_isocontour_controls()
        self._on_xfem_frame_changed()

    # -- duplicate "Show trace"/"Frame" controls on the isocontours card --
    # -- (see their construction, next to isocontours_card, for why) ------

    def _sync_xfem_isocontour_controls(self):
        """Mirrors xfem_critical_combo/xfem_frame_slider/xfem_frame_spin's
        current items/range/value/enabled state onto their duplicate
        widgets on the "Field isocontours" card -- called after every
        place those canonical controls change, whatever the source (a
        fresh run, ticking a different trace, dragging either copy of
        the slider/spin). The duplicates are never a second source of
        truth: this only ever copies FROM the canonical widgets, and the
        duplicates' own change handlers only ever forward TO them."""
        self.xfem_isocontour_trace_combo.blockSignals(True)
        self.xfem_isocontour_trace_combo.clear()
        for i in range(self.xfem_critical_combo.count()):
            self.xfem_isocontour_trace_combo.addItem(self.xfem_critical_combo.itemText(i))
        self.xfem_isocontour_trace_combo.setCurrentIndex(self.xfem_critical_combo.currentIndex())
        self.xfem_isocontour_trace_combo.blockSignals(False)

        self.xfem_isocontour_frame_slider.blockSignals(True)
        self.xfem_isocontour_frame_slider.setRange(
            self.xfem_frame_slider.minimum(), self.xfem_frame_slider.maximum()
        )
        self.xfem_isocontour_frame_slider.setValue(self.xfem_frame_slider.value())
        self.xfem_isocontour_frame_slider.setEnabled(self.xfem_frame_slider.isEnabled())
        self.xfem_isocontour_frame_slider.blockSignals(False)

        self.xfem_isocontour_frame_spin.blockSignals(True)
        self.xfem_isocontour_frame_spin.setRange(
            self.xfem_frame_spin.minimum(), self.xfem_frame_spin.maximum()
        )
        self.xfem_isocontour_frame_spin.setValue(self.xfem_frame_spin.value())
        self.xfem_isocontour_frame_spin.setEnabled(self.xfem_frame_spin.isEnabled())
        self.xfem_isocontour_frame_spin.blockSignals(False)

    def _on_xfem_isocontour_trace_combo_changed(self, _index=None):
        self.xfem_critical_combo.setCurrentIndex(self.xfem_isocontour_trace_combo.currentIndex())

    def _on_xfem_isocontour_frame_slider_changed(self, value: int):
        self.xfem_frame_spin.setValue(value)

    def _on_xfem_isocontour_frame_spin_changed(self, value: int):
        self.xfem_frame_spin.setValue(value)

    def _on_xfem_frame_changed(self, _value=None):
        if self._xfem_results:
            self._refresh_xfem_display()
        self._maybe_refresh_xfem_isocontours()

    def _on_xfem_field_combo_changed(self, _index=None):
        self._maybe_refresh_xfem_isocontours()

    def _maybe_refresh_xfem_isocontours(self):
        """Keeps the "Field isocontours" card live once it has been shown
        at least once: called on every frame/trace/field change so moving
        the Frame slider (or switching trace/field) redraws it without
        forcing a second "Show isocontours" click each time. Left alone
        (card stays as last drawn, or with its initial placeholder text)
        until the user has clicked "Show isocontours" at least once,
        since re-evaluating the X-FEM-DIC field on a grid on every single
        slider tick would otherwise fire even before there's anything
        useful to show."""
        if getattr(self, "_xfem_isocontour_shown", False):
            self._on_xfem_isocontour_clicked()

    def _on_xfem_export_k_csv_clicked(self):
        if not self._xfem_results:
            self.xfem_export_k_csv_status.setText("Compute results above first (\"Preview\" or \"Run\").")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export K_I/K_II CSV", "xfem_dic_k.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        pixel_size = self.pixel_size_spin.value() or None
        try:
            export_xfem_k_csv(self._xfem_results, path, pixel_size_mm=pixel_size)
        except OSError as exc:
            self.xfem_export_k_csv_status.setText(f"Could not write the file: {exc}")
            return
        self.xfem_export_k_csv_status.setText(
            f"Exported {len(self._xfem_results)} trace(s) to {os.path.basename(path)}."
        )

    def _refresh_xfem_display(self, _checked=None):
        if not self._xfem_results:
            return
        self._draw_xfem_plot()
        self._draw_xfem_arc_length_plot()
        self._draw_xfem_speed_plot()

        index = self.xfem_critical_combo.currentIndex()
        if not (0 <= index < len(self._xfem_results)):
            return
        entry = self._xfem_results[index]
        series = entry["series"]
        if not series.samples:
            return
        frame_index = max(0, min(len(series.samples) - 1, self.xfem_frame_spin.value()))
        sample = series.sample_at(frame_index)
        fit = sample.fit
        if entry.get("method") == "direct":
            self.xfem_status.setText(self._direct_k_status_text(entry, frame_index))
            return

        # Same K_I-only "critical = mean of positive values" treatment a
        # standard near-tip K-fit would use (see
        # `positive_value_mean_and_domain_bounds`) -- K_II keeps its own
        # single-position critical value below, unaffected. Only computed/
        # shown when "K_I" is ticked above -- a mean of a curve the plot
        # itself isn't even drawing right now would just be clutter (the
        # user's own "don't show K1/K2 means for an unticked box" request).
        show_KI = self.xfem_show_KI_check.isChecked()
        show_KII = self.xfem_show_KII_check.isChecked()

        lines = [f"{entry['label']}"]
        if show_KI:
            distances_all = []
            k1_all = []
            for s in series.samples:
                arc = s.tip_arc_length_s
                distances_all.append(arc if arc is not None and np.isfinite(arc) else np.nan)
                k1_all.append(s.fit.K_I_pa_sqrt_m if s.fit.K_I_pa_sqrt_m is not None else np.nan)
            mean_ki, ki_bounds = positive_value_mean_and_domain_bounds(np.array(distances_all), np.array(k1_all))
            if mean_ki is None:
                lines.append("K_I (critical, mean of positive values): no position on this trace has K_I > 0.")
            else:
                lines.append(
                    f"K_I (critical, mean of {len(ki_bounds)} positive-value domain(s)) = "
                    f"{mean_ki:.5g} Pa·√m  =  {mean_ki / 1.0e6:.4g} MPa·√m"
                )
        if show_KI:
            # Blank separator between the trace-level summary above (K_I
            # critical means, one line per method) and the frame-level
            # detail below -- these are two different "kinds" of
            # information (whole-trace vs. one inspected position) that
            # otherwise run straight into each other with nothing marking
            # the switch.
            lines.append("")
        lines.append(f"Inspected position {frame_index + 1} / {len(series.samples)} "
                     f"(t = {sample.time_s:.6g} s):")
        if sample.tip_source.startswith("auto") and sample.tip_arc_length_s is not None:
            nominal_s = (entry["tip_result"].arc_length_s[sample.frame_order_index]
                         if entry["tip_result"] is not None else None)
            # tip_source is "auto (residual)" or "auto (heaviside jump)" --
            # the bit in parens says which search algorithm found this.
            method_note = sample.tip_source[len("auto "):]
            note = (f"  Auto-detected tip {method_note} at "
                     f"s = {sample.tip_arc_length_s:.1f} px along the path")
            if nominal_s is not None and np.isfinite(nominal_s):
                note += f" (fixed-trace value: {nominal_s:.1f} px)"
            lines.append(note + ".")
        if fit.error is not None:
            lines.append(f"  This frame failed: {fit.error}")
        elif fit.K_I_pa_sqrt_m is None:
            lines.append("  This frame did not converge to a usable fit.")
        else:
            k1_mpa = fit.K_I_pa_sqrt_m / 1.0e6
            k2_mpa = fit.K_II_pa_sqrt_m / 1.0e6
            k1_stderr_note = (f"  (+/- {fit.K_I_stderr_pa_sqrt_m / 1.0e6:.3g} MPa·√m, 1 sigma)"
                               if fit.K_I_stderr_pa_sqrt_m is not None else "")
            k2_stderr_note = (f"  (+/- {fit.K_II_stderr_pa_sqrt_m / 1.0e6:.3g} MPa·√m, 1 sigma)"
                               if fit.K_II_stderr_pa_sqrt_m is not None else "")
            if show_KI:
                lines.append(f"  K_I  = {fit.K_I_pa_sqrt_m:.5g} Pa·√m  =  {k1_mpa:.4g} MPa·√m{k1_stderr_note}")
            if show_KII:
                lines.append(f"  K_II (critical) = {fit.K_II_pa_sqrt_m:.5g} Pa·√m  =  "
                              f"{k2_mpa:.4g} MPa·√m{k2_stderr_note}")
            if show_KI and show_KII and abs(fit.K_I_pa_sqrt_m) > 0:
                mixity = fit.K_II_pa_sqrt_m / fit.K_I_pa_sqrt_m
                lines.append(f"  Mode mixity K_II / K_I = {mixity:.3g}")
            if fit.converged:
                status_word = "converged"
            elif fit.stalled:
                status_word = "stalled (residual plateaued)"
            else:
                status_word = "did NOT converge"
            lines.append(
                f"  Grey-level residual RMS: {fit.residual_rms_initial:.4g} -> {fit.residual_rms:.4g}  |  "
                f"{status_word} in {fit.n_iterations} "
                f"iteration(s), {fit.n_samples} sample point(s), {int(fit.heaviside_nodes.sum())} "
                f"Heaviside-enriched node(s)."
            )
            if fit.stalled:
                lines.append(
                    "  Note: this fit stopped early because the grey-level residual stopped "
                    "improving from one iteration to the next -- typically means the crack-tip "
                    "displacement this frame is small enough that the correlation signal is close "
                    "to the noise floor, so this is likely close to the best fit achievable here. "
                    "Raising \"Max iterations\" is unlikely to help (it isn't slowly converging, it "
                    "has plateaued) -- the K_I/K_II reported are this fit's best-RMS state, not "
                    "necessarily its very last iterate, but still treat them with the same caution "
                    "as an unconverged fit."
                )
            elif not fit.converged:
                lines.append(
                    "  Warning: this fit did not converge -- treat its K_I/K_II with caution (try "
                    "a smaller sample spacing, a different element size, a larger \"Max iterations\", "
                    "or check the crack path/tip position this frame)."
                )
        if frame_index != (series.default_critical_index or 0):
            lines.append("(Inspected/K_II critical position manually selected, different from the "
                          "suggested default.)")
        if len(self._xfem_results) > 1:
            lines.append(f"({len(self._xfem_results)} trace(s) plotted together below.)")
        if self._xfem_skipped_traces:
            lines.append("Skipped: " + "; ".join(self._xfem_skipped_traces))
        self.xfem_status.setText("\n".join(lines))

    def _direct_k_status_text(self, entry, frame_index: int) -> str:
        """Per-frame status text of a DIRECT (measured-field) K entry -- the counterpart of the
        X-FEM-DIC text in `_refresh_xfem_display` (no mesh/convergence details here: these are
        closed-form fits of the exported displacement field)."""
        series = entry["series"]
        sample = series.sample_at(frame_index)
        fit = sample.fit
        show_KI = self.xfem_show_KI_check.isChecked()
        show_KII = self.xfem_show_KII_check.isChecked()
        lines = [entry["label"],
                 f"Direct method on the measured displacement field: {method_label(series.method)}, "
                 f"r = {series.r_min_px:g}..{series.r_max_px:g} px, flank exclusion "
                 f"{series.flank_exclusion_deg:g} deg, {series.smoothing}."]
        if show_KI:
            dist = np.array([s_.tip_arc_length_s if s_.tip_arc_length_s is not None else np.nan
                             for s_ in series.samples], dtype=float)
            k1 = np.array([s_.fit.K_I_pa_sqrt_m if s_.fit.K_I_pa_sqrt_m is not None else np.nan
                           for s_ in series.samples], dtype=float)
            mean_ki, bounds = positive_value_mean_and_domain_bounds(dist, k1)
            if mean_ki is None:
                lines.append("K_I (critical, mean of positive values): no position on this trace has K_I > 0.")
            else:
                lines.append(f"K_I (critical, mean of {len(bounds)} positive-value domain(s)) = "
                             f"{mean_ki:.5g} Pa·√m  =  {mean_ki / 1.0e6:.4g} MPa·√m")
            lines.append("")
        lines.append(f"Inspected position {frame_index + 1} / {len(series.samples)} (t = {sample.time_s:.6g} s):")
        if fit.error is not None:
            lines.append(f"  This frame failed: {fit.error}")
        else:
            def _unc(v):
                return f"  (+/- {v / 1.0e6:.3g} MPa·√m, 1 sigma)" if v is not None else ""
            if show_KI:
                lines.append(f"  K_I  = {fit.K_I_pa_sqrt_m:.5g} Pa·√m  =  {fit.K_I_pa_sqrt_m / 1.0e6:.4g} MPa·√m"
                             + _unc(fit.K_I_stderr_pa_sqrt_m))
            if show_KII:
                lines.append(f"  K_II = {fit.K_II_pa_sqrt_m:.5g} Pa·√m  =  {fit.K_II_pa_sqrt_m / 1.0e6:.4g} MPa·√m"
                             + _unc(fit.K_II_stderr_pa_sqrt_m))
            if show_KI and show_KII and abs(fit.K_I_pa_sqrt_m) > 0:
                lines.append(f"  Mode mixity K_II / K_I = {fit.K_II_pa_sqrt_m / fit.K_I_pa_sqrt_m:.3g}")
            if fit.r_squared is not None:
                lines.append(f"  Fit R² = {fit.r_squared:.4f}  |  {fit.n_samples} sample point(s)")
            else:
                lines.append(f"  {fit.n_samples} integration cell(s)")
            if fit.t_stress_pa is not None:
                lines.append(f"  T-stress = {fit.t_stress_pa / 1.0e6:.4g} MPa")
            if fit.K_I_inner_half_pa_sqrt_m is not None and fit.K_I_outer_half_pa_sqrt_m is not None:
                lines.append(f"  Domain independence (K_I): inner half {fit.K_I_inner_half_pa_sqrt_m / 1.0e6:.4g}, "
                             f"outer half {fit.K_I_outer_half_pa_sqrt_m / 1.0e6:.4g} MPa·√m "
                             "-- a large gap between them is a warning sign.")
        if frame_index != (series.default_critical_index or 0):
            lines.append("(Inspected position manually selected, different from the suggested default.)")
        if len(self._xfem_results) > 1:
            lines.append(f"({len(self._xfem_results)} trace(s) plotted together below.)")
        if self._xfem_skipped_traces:
            lines.append("Skipped: " + "; ".join(self._xfem_skipped_traces))
        return "\n".join(lines)

    def _draw_xfem_plot(self):
        self.xfem_canvas.clear()
        ax = self.xfem_canvas.figure.add_subplot(111)
        unit, scale = self._y_axis_unit_and_scale()
        critical_index = self.xfem_critical_combo.currentIndex()
        critical_frame = self.xfem_frame_spin.value()
        visible_labels = self._xfem_visible_result_labels()
        show_KI = self.xfem_show_KI_check.isChecked()
        show_KII = self.xfem_show_KII_check.isChecked()
        sigma_n = self.xfem_uncertainty_sigma_combo.currentData() or 0
        any_plotted = False

        for i, entry in enumerate(self._xfem_results):
            if entry["label"] not in visible_labels:
                continue
            any_plotted = True
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            series = entry["series"]
            distances, k1, k2 = [], [], []
            k1_stderr, k2_stderr = [], []
            for sample in series.samples:
                # Each sample's OWN tip_arc_length_s (not the ticked
                # trace's nominal tip_result.arc_length_s) -- for a fixed-
                # trace series these are identical, but for an auto-
                # detected series (see the Auto-detected tip position
                # card) the whole point is that they can differ, so
                # plotting the nominal value here would silently show the
                # WRONG x-position for a self-found tip.
                arc = sample.tip_arc_length_s
                distances.append(arc * scale if arc is not None and np.isfinite(arc) else np.nan)
                fit = sample.fit
                k1.append(fit.K_I_pa_sqrt_m / 1.0e6 if fit.K_I_pa_sqrt_m is not None else np.nan)
                k2.append(fit.K_II_pa_sqrt_m / 1.0e6 if fit.K_II_pa_sqrt_m is not None else np.nan)
                k1_stderr.append(fit.K_I_stderr_pa_sqrt_m / 1.0e6
                                  if fit.K_I_stderr_pa_sqrt_m is not None else np.nan)
                k2_stderr.append(fit.K_II_stderr_pa_sqrt_m / 1.0e6
                                  if fit.K_II_stderr_pa_sqrt_m is not None else np.nan)
            distances = np.array(distances)
            k1 = np.array(k1)
            k2 = np.array(k2)
            k1_stderr = np.array(k1_stderr)
            k2_stderr = np.array(k2_stderr)

            def _band(curve, stderr, curve_color, alpha):
                if sigma_n <= 0:
                    return
                band_finite = np.isfinite(curve) & np.isfinite(stderr)
                if not np.any(band_finite):
                    return
                ax.fill_between(distances[band_finite],
                                 (curve - sigma_n * stderr)[band_finite],
                                 (curve + sigma_n * stderr)[band_finite],
                                 color=curve_color, alpha=alpha, linewidth=0, zorder=1)

            if show_KI:
                _band(k1, k1_stderr, color, 0.15)
                ax.plot(distances, k1, "o-", color=color, markersize=4, linewidth=1.2,
                        label=f"{entry['label']} -- K_I", zorder=3)
            if show_KII:
                _band(k2, k2_stderr, color, 0.15)
                ax.plot(distances, k2, "s--", color=color, markersize=4, linewidth=1.2,
                        label=f"{entry['label']} -- K_II", zorder=3)
            if i == critical_index and 0 <= critical_frame < len(series.samples) and np.isfinite(distances[critical_frame]):
                ax.axvline(distances[critical_frame], color=color, linestyle="-.", linewidth=1.5,
                           label=f"{entry['label']} -- inspected position (K_II critical)")

        ax.set_xlabel(f"distance along path ({unit})")
        ax.set_ylabel("K (MPa·√m)")
        if any_plotted:
            ax.legend(fontsize=8, loc="best")
            self._force_origin_visible(ax)
        else:
            ax.text(0.5, 0.5, "No trace ticked in \"Show curves\" above", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        self.xfem_canvas.refresh()

    def _draw_xfem_arc_length_plot(self):
        """Arc length along path vs time, for every X-FEM-DIC result
        trace -- the same underlying `series.samples` the K_I/K_II plot
        above reads, just plotting each sample's own `time_s` against its
        own `tip_arc_length_s` instead of K against distance. Lets the
        user directly compare how far different methods (fixed trace,
        auto-detected residual, auto-detected Heaviside jump, ...) think
        the crack tip has travelled at each point in time, on one shared
        axis -- the K_I/K_II plot's x-axis is distance, which can't show
        this. Shares the "Show curves" tick list with that plot.

        For an auto-detected trace, a frame whose search found NOTHING
        USABLE (`fit.error` set -- see compute_xfem_dic_time_series) still
        gets a `tip_arc_length_s` filled in: it's set to that frame's own
        NOMINAL (fixed-trace) position purely as a placeholder to build
        the mesh/report on, not a position the search actually confirmed.
        The K_I/K_II plot already reads as failed there (K is NaN, since
        it comes straight from `fit.K_I_pa_sqrt_m`), but plotting that
        same placeholder position here, unfiltered, would draw an
        unbroken trajectory that looks fully tracked even on frames where
        the search found nothing -- exactly the mismatch this plot
        exists to let the user catch, so those points are skipped (left
        as a gap) for auto-detected traces. A "fixed" trace has no such
        placeholder -- its position always comes straight from the
        already-tracked ridge/wave trace, never from a search that can
        fail -- so it's never filtered here even when ITS fit at that
        (perfectly legitimate) position happened to fail."""
        self.xfem_arc_length_canvas.clear()
        ax = self.xfem_arc_length_canvas.figure.add_subplot(111)
        unit, scale = self._y_axis_unit_and_scale()
        visible_labels = self._xfem_visible_result_labels()
        any_plotted = False

        for i, entry in enumerate(self._xfem_results):
            if entry["label"] not in visible_labels:
                continue
            any_plotted = True
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            series = entry["series"]
            times, distances = [], []
            for sample in series.samples:
                times.append(sample.time_s)
                arc = sample.tip_arc_length_s
                search_failed = sample.tip_source.startswith("auto") and sample.fit.error is not None
                if arc is not None and np.isfinite(arc) and not search_failed:
                    distances.append(arc * scale)
                else:
                    distances.append(np.nan)
            ax.plot(times, distances, "o-", color=color, markersize=4, linewidth=1.2,
                    label=entry["label"])

        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"distance along path ({unit})")
        if any_plotted:
            ax.legend(fontsize=8, loc="best")
        else:
            ax.text(0.5, 0.5, "No trace ticked in \"Show curves\" above", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        self.xfem_arc_length_canvas.refresh()

    def _draw_xfem_speed_plot(self):
        """K_I versus local tip speed, one marker per frame of every ticked result trace
        (X-FEM-DIC and direct alike -- both expose `samples` with `time_s`, `tip_arc_length_s`
        and `fit.K_I_pa_sqrt_m`). The speed comes from core.tip_speed.local_tip_speed. A brittle
        material's dynamic toughness K_ID rises steeply with speed, so a plateau of K well above
        the quasi-static K_Ic is only meaningful against the speed the crack had there. With a
        density set, a top axis gives the speed as a fraction of the Rayleigh wave speed."""
        self.xfem_speed_canvas.clear()
        ax = self.xfem_speed_canvas.figure.add_subplot(111)
        pixel_size = self.pixel_size_spin.value() or None
        visible_labels = self._xfem_visible_result_labels()
        window = self.xfem_speed_window_spin.value()
        material = self.material_properties()
        c_r = rayleigh_speed(material["young_modulus_pa"], material["poisson_ratio"],
                             self.xfem_density_spin.value())
        pooled_v, pooled_k = [], []
        any_plotted = False
        if pixel_size:
            for i, entry in enumerate(self._xfem_results):
                if entry["label"] not in visible_labels:
                    continue
                samples = entry["series"].samples
                times = np.array([smp.time_s for smp in samples], dtype=float)
                arc = np.array([smp.tip_arc_length_s if smp.tip_arc_length_s is not None else np.nan
                                for smp in samples], dtype=float)
                k1 = np.array([smp.fit.K_I_pa_sqrt_m / 1.0e6 if smp.fit.K_I_pa_sqrt_m is not None else np.nan
                               for smp in samples], dtype=float)
                v = local_tip_speed(times, arc, pixel_size, window)
                good = np.isfinite(v) & np.isfinite(k1)
                if not np.any(good):
                    continue
                any_plotted = True
                color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                ax.plot(v[good], k1[good], "o", color=color, markersize=4, alpha=0.85, label=entry["label"])
                pooled_v.append(v[good])
                pooled_k.append(k1[good])
        ax.set_xlabel("local tip speed (m/s)")
        ax.set_ylabel("K_I (MPa·√m)")
        if any_plotted:
            ax.legend(fontsize=8, loc="best")
            self._force_origin_visible(ax)
            if c_r:
                top = ax.secondary_xaxis("top", functions=(lambda x: x / c_r, lambda x: x * c_r))
                top.set_xlabel(f"v / c_R  (c_R = {c_r:.0f} m/s)")
            vv, kk = np.concatenate(pooled_v), np.concatenate(pooled_k)
            rho = spearman(vv, kk)
            self.xfem_speed_status.setText(
                f"{len(vv)} frame(s) with a positive speed; speed range {np.min(vv):.0f}..{np.max(vv):.0f} m/s, "
                f"median {np.median(vv):.0f} m/s"
                + (f" ({np.median(vv) / c_r:.2f} c_R)" if c_r else "")
                + (f"; rank correlation K_I vs speed = {rho:+.2f}" if rho is not None else "")
                + ".")
        else:
            ax.text(0.5, 0.5, "Needs a pixel size and at least one ticked trace with a moving tip",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            self.xfem_speed_status.setText("")
        self.xfem_speed_canvas.refresh()

    def _on_xfem_isocontour_clicked(self):
        """Re-evaluates the selected frame's already-fitted X-FEM-DIC
        field on a grid and draws it as line contours over the standard
        subset-DIC field's own filled contours (same frame, same field) --
        a visual cross-check between the two methods. Reads the SAME
        "Show trace" / "Frame" selection the K_I/K_II plot above uses, so
        picking a position there and clicking this button shows that
        exact frame's field."""
        if not self._xfem_results:
            self.xfem_isocontour_status.setText("Compute results above first (\"Preview\" or \"Run\").")
            return
        index = self.xfem_critical_combo.currentIndex()
        if not (0 <= index < len(self._xfem_results)):
            self.xfem_isocontour_status.setText("Pick a trace with \"Show trace\" above first.")
            return
        entry = self._xfem_results[index]
        series = entry["series"]
        if entry.get("method") == "direct":
            self.xfem_isocontour_status.setText(
                f"\"{entry['label']}\" is a direct measured-field K result -- there is no fitted "
                "X-FEM-DIC field to contour. Pick an X-FEM-DIC trace in \"Show trace\"."
            )
            return
        if not series.samples:
            self.xfem_isocontour_status.setText(f"\"{entry['label']}\" has no computed frames.")
            return
        frame_index = max(0, min(len(series.samples) - 1, self.xfem_frame_spin.value()))
        sample = series.sample_at(frame_index)
        fit = sample.fit

        if fit.error is not None:
            self.xfem_isocontour_status.setText(
                f"{entry['label']}, frame {frame_index + 1}: this frame's fit failed ({fit.error}) "
                f"-- no field to show."
            )
            return
        if fit.K_I_pa_sqrt_m is None or sample.tangent_xy is None:
            self.xfem_isocontour_status.setText(
                f"{entry['label']}, frame {frame_index + 1}: no usable fit for this frame."
            )
            return
        tip_result = entry["tip_result"]
        if tip_result is None:
            self.xfem_isocontour_status.setText(
                f"{entry['label']}: the underlying tip track is no longer available (trace removed "
                f"since this was computed?) -- re-run to refresh."
            )
            return

        material = self.material_properties()
        kappa = kolosov_kappa(material["poisson_ratio"], material["plane_stress"])
        mu_pa = shear_modulus(material["young_modulus_pa"], material["poisson_ratio"])

        field_key = self.xfem_field_combo.currentData()
        field_label = self.xfem_field_combo.currentText()
        dic_field = None
        if self._dataset is not None and 0 <= sample.source_frame_index < len(self._dataset.frames):
            dic_field = self._dataset.frames[sample.source_frame_index].field(field_key)

        # Cropped to the standard subset-DIC's own correlated ROI when a
        # field to compare against is available -- otherwise the X-FEM-DIC
        # mesh's own (generally larger, tip-centered) extent shows computed
        # values past wherever the standard DIC ROI actually stops, which
        # reads as a spurious method discrepancy rather than "no data to
        # compare here" (see evaluate_xfem_dic_field_grid's own docstring).
        grid = evaluate_xfem_dic_field_grid(
            fit, tip_result.path_positions, tip_result.path_arc_length,
            sample.tip_xy_px, sample.tangent_xy, kappa, mu_pa,
            series.pixel_size_mm, series.r_enrich_px,
            step_px=self.xfem_sample_step_spin.value(),
            dic_xs=self._dataset.xs if dic_field is not None else None,
            dic_ys=self._dataset.ys if dic_field is not None else None,
            dic_field=dic_field,
            # The exact r_min_px/r_max_px this series was actually fitted
            # with (see XfemDicTimeSeries's own fields) -- reconstructing
            # the field with a DIFFERENT window than the one the
            # Gauss-Newton solve used would silently mismatch what's shown
            # here against what was actually fitted.
            r_min_px=series.r_min_px, r_max_px=series.r_max_px,
        )

        self._xfem_isocontour_shown = True
        xfem_field = grid.field(field_key)
        finite_values = [xfem_field[np.isfinite(xfem_field)]]
        if dic_field is not None:
            dic_finite = np.asarray(dic_field, dtype=float)
            dic_finite = dic_finite[np.isfinite(dic_finite)]
            if dic_finite.size:
                finite_values.append(dic_finite)
        self._xfem_isocontour_signal_for_scale = (
            np.concatenate(finite_values) if finite_values else np.array([0.0, 1.0])
        )

        self._draw_xfem_isocontours(grid, field_key, field_label, dic_field, entry, sample, frame_index)
        n_shown = f"{len(series.samples)}"
        self.xfem_isocontour_status.setText(
            f"{entry['label']}, frame {frame_index + 1}/{n_shown} -- {field_label}. Left = standard "
            f"subset-DIC (already-exported field); right = X-FEM-DIC's own fitted field, same color "
            f"scale."
            + ("" if dic_field is not None else "  (no standard-DIC field to compare against for this "
                                                  "frame/field -- showing X-FEM-DIC only.)")
        )

    # -- isocontour color scale (Auto/Fixed) -- shared by both panels, ------
    # -- mirrors _on_wave_scale_* / _on_tip_scale_* above -------------------

    def _on_xfem_isocontour_scale_mode_toggled(self, _checked=None):
        fixed = self.xfem_isocontour_scale_fixed_radio.isChecked()
        self.xfem_isocontour_scale_min_spin.setEnabled(fixed)
        self.xfem_isocontour_scale_max_spin.setEnabled(fixed)
        self.xfem_isocontour_scale_reset_button.setEnabled(fixed)
        if fixed and self.xfem_isocontour_scale_min_spin.value() == self.xfem_isocontour_scale_max_spin.value():
            self._on_xfem_isocontour_scale_reset()
        self._maybe_refresh_xfem_isocontours()

    def _on_xfem_isocontour_scale_bounds_changed(self, _value=None):
        self._maybe_refresh_xfem_isocontours()

    def _on_xfem_isocontour_scale_reset(self):
        signal = self._xfem_isocontour_signal_for_scale
        if signal is None:
            return
        finite = signal[np.isfinite(signal)]
        if finite.size == 0:
            return
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        self.xfem_isocontour_scale_min_spin.blockSignals(True)
        self.xfem_isocontour_scale_max_spin.blockSignals(True)
        self.xfem_isocontour_scale_min_spin.setValue(vmin)
        self.xfem_isocontour_scale_max_spin.setValue(vmax)
        self.xfem_isocontour_scale_min_spin.blockSignals(False)
        self.xfem_isocontour_scale_max_spin.blockSignals(False)
        self._maybe_refresh_xfem_isocontours()

    def _current_xfem_isocontour_scale_vmin_vmax(self):
        if self.xfem_isocontour_scale_fixed_radio.isChecked():
            return self.xfem_isocontour_scale_min_spin.value(), self.xfem_isocontour_scale_max_spin.value()
        return None, None

    def _draw_xfem_isocontours(self, grid, field_key, field_label, dic_field, entry, sample, frame_index):
        """Draws the standard subset-DIC field and the X-FEM-DIC fitted
        field as two side-by-side FILLED isocontour panels (sharing axes
        and one color scale), rather than overlaying one as black line
        contours on the other, which is harder to read. The crack overlay drawn on
        top (see `_overlay_path_and_tip`) is TWO distinct lines: the
        static drawn-ROI path (thin/faint, for context) and the actual
        per-frame tip trajectory (bold white+black dashes) -- only the
        latter reflects "Smooth tip/wave position" baked in at add time.

        Below the two isocontour panels, a second row re-runs
        `crack_opening_jump_profile` for THIS SAME frame -- once from
        the standard subset-DIC's own raw u/v fields (below the left
        panel), once from the X-FEM-DIC fit's reconstructed u/v (below
        the right panel) -- the same "opening-jump" diagnostic already
        shown for standard-DIC-only crack-tip detection in the "Verify
        visually" card (see `_draw_tip_verify_frame`), computed here
        against BOTH methods' own fields so a disagreement between them
        shows up directly as a difference in each one's own detected
        opening front, not just in the color maps above. Uses the same
        Offset/Threshold/Gap tolerance spinboxes as that card's "Detect
        tip position" controls (`tip_roi_opening_jump_*_spin`) -- a
        second, dedicated set of controls here would just drift out of
        sync with those, for no real benefit (the whole point is a
        directly comparable number)."""
        self.xfem_isocontour_canvas.clear()
        fig = self.xfem_isocontour_canvas.figure
        gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0], hspace=0.4, wspace=0.3)
        ax_dic = fig.add_subplot(gs[0, 0])
        ax_xfem = fig.add_subplot(gs[0, 1], sharex=ax_dic, sharey=ax_dic)

        xfem_field = grid.field(field_key)
        vmin, vmax = self._current_xfem_isocontour_scale_vmin_vmax()
        if vmin is None or vmax is None:
            finite_values = [xfem_field[np.isfinite(xfem_field)]]
            if dic_field is not None:
                dic_finite = np.asarray(dic_field, dtype=float)
                dic_finite = dic_finite[np.isfinite(dic_finite)]
                if dic_finite.size:
                    finite_values.append(dic_finite)
            combined = np.concatenate(finite_values) if finite_values else np.array([0.0, 1.0])
            if combined.size < 2:
                combined = np.array([0.0, 1.0])
            # Robust (percentile, not min/max) range: the X-FEM-DIC strain
            # fields include the near-tip singular term, which can be very
            # large right next to the tip -- a handful of hot pixels there
            # would otherwise wash out the color scale everywhere else, the
            # part of the field this comparison actually cares about.
            vmin, vmax = np.nanpercentile(combined, [2.0, 98.0])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = float(np.nanmin(combined)), float(np.nanmax(combined))
        if np.isfinite(vmin) and np.isfinite(vmax) and vmin > vmax:
            # Can happen with "Fixed" scale bounds: the min/max spinboxes
            # are independent (no cross min/max clamp), so typing into one
            # -- or even a single keystroke mid-edit -- can transiently
            # (or, if the user just gets them backwards, persistently)
            # leave min above max. np.linspace(vmin, vmax, ...) would then
            # produce a DECREASING sequence, which matplotlib's contourf
            # rejects outright ("Contour levels must be increasing") --
            # swap instead of crashing, since "min above max" only ever
            # means the two got swapped, never that there's no valid range.
            vmin, vmax = vmax, vmin
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = (vmin - 1.0 if np.isfinite(vmin) else -1.0,
                          vmax + 1.0 if np.isfinite(vmax) else 1.0)
        levels = isocontour_levels(vmin, vmax)

        tip_result = entry["tip_result"]
        tip_xy = sample.tip_xy_px

        def _overlay_path_and_tip(ax):
            # Two DIFFERENT lines, easy to conflate (this was the cause of
            # a "smoothing has no effect" report): `path_positions` is the
            # STATIC geometry of the drawn ROI polygon's own resampled
            # path -- fixed once the ROI is drawn, and never touched by
            # "Smooth tip/wave position" (that option only ever changes
            # WHICH point along this path gets picked at each frame, not
            # the path itself). Drawn thin/faint here, for context only.
            if tip_result is not None and len(tip_result.path_positions) >= 2:
                path = tip_result.path_positions
                ax.plot(path[:, 0], path[:, 1], color="0.6", linewidth=0.8, linestyle=":",
                        alpha=0.7, zorder=4)
            # The actual per-frame crack-tip trajectory (time-ordered
            # picked positions) -- THIS is what gets replaced by the
            # smoothed path at "Add to list" time when "Smooth tip/wave
            # position" is checked, so THIS is the line that visibly
            # changes with that option. Drawn prominently (white halo +
            # black dashes), taking over the role the static path used to
            # play here.
            #
            # Walked along the REAL traced path (`path_polyline_up_to_
            # arc_length`) from its origin up to the furthest arc length
            # any detected point reached, rather than a straight line
            # connecting the (temporally sparse) detected (x, y) points --
            # a fast/accelerating crack can have several frames between
            # two detections, and a straight chord between them visibly
            # cuts across whatever the path actually does in between (see
            # core.crack_tip_tracking.path_polyline_up_to_arc_length's
            # docstring). Small markers mark the actual detected points.
            if tip_result is not None:
                finite = np.isfinite(tip_result.positions_xy).all(axis=1)
                if np.count_nonzero(finite) >= 1 and len(tip_result.path_positions) >= 2:
                    max_s = float(np.nanmax(tip_result.arc_length_s[finite]))
                    traj = path_polyline_up_to_arc_length(
                        tip_result.path_positions, tip_result.path_arc_length, max_s
                    )
                    ax.plot(traj[:, 0], traj[:, 1], color="white", linewidth=1.5, alpha=0.9,
                            zorder=5)
                    ax.plot(traj[:, 0], traj[:, 1], color="black", linewidth=0.6, linestyle="--",
                            zorder=6)
                    detected = tip_result.positions_xy[finite]
                    ax.scatter(detected[:, 0], detected[:, 1], s=10, color="black", zorder=5,
                               edgecolors="white", linewidths=0.3)
            ax.plot([tip_xy[0]], [tip_xy[1]], marker="+", color="white", markersize=13,
                    markeredgewidth=2.4, linestyle="none", zorder=7)
            ax.plot([tip_xy[0]], [tip_xy[1]], marker="+", color="black", markersize=11,
                    markeredgewidth=1.2, linestyle="none", zorder=8)

        if dic_field is not None and self._dataset is not None:
            dic_masked = np.ma.masked_invalid(np.asarray(dic_field, dtype=float))
            cf_dic = ax_dic.contourf(self._dataset.xs, self._dataset.ys, dic_masked, levels=levels,
                                      cmap="jet", extend="both")
            isocontour_colorbar(fig, cf_dic, ax=ax_dic, label=field_label, fraction=0.046, pad=0.04)
        else:
            ax_dic.text(0.5, 0.5, "No standard-DIC field\nfor this frame/field", ha="center",
                         va="center", transform=ax_dic.transAxes, fontsize=8)
        _overlay_path_and_tip(ax_dic)
        ax_dic.set_title("Standard subset-DIC", fontsize=9)
        ax_dic.set_xlabel("x (px)")
        ax_dic.set_ylabel("y (px)")
        ax_dic.set_aspect("equal", adjustable="box")

        xfem_masked = np.ma.masked_invalid(xfem_field)
        if np.ma.count(xfem_masked) > 0:
            cf_xfem = ax_xfem.contourf(grid.x, grid.y, xfem_masked, levels=levels, cmap="jet",
                                        extend="both")
            isocontour_colorbar(fig, cf_xfem, ax=ax_xfem, label=field_label, fraction=0.046, pad=0.04)
        else:
            ax_xfem.text(0.5, 0.5, "No X-FEM-DIC field\nfor this frame/field", ha="center",
                          va="center", transform=ax_xfem.transAxes, fontsize=8)
        _overlay_path_and_tip(ax_xfem)
        ax_xfem.set_title("X-FEM-DIC", fontsize=9)
        ax_xfem.set_xlabel("x (px)")
        ax_xfem.set_ylabel("y (px)")
        ax_xfem.set_aspect("equal", adjustable="box")

        fig.suptitle(f"{entry['label']} -- frame {frame_index + 1}  --  {field_label}", fontsize=10)
        ax_dic.invert_yaxis()  # image pixel convention (y downward); shared axes -> flips both panels

        # -- second row: each method's own crack-opening-jump diagnostic,
        # same signal/offset/threshold as the "Verify visually" card (see
        # `_draw_tip_verify_frame`) -- computed independently per method
        # so the two are directly comparable, not derived from the u/v
        # panels above (which may be showing a different field entirely,
        # e.g. a strain component -- the jump diagnostic always needs the
        # raw displacement fields regardless of what's plotted above).
        ax_jump_dic = fig.add_subplot(gs[1, 0])
        ax_jump_xfem = fig.add_subplot(gs[1, 1])
        offset_px = self.tip_roi_opening_jump_offset_spin.value()
        max_gap_samples = self.tip_roi_opening_jump_max_gap_spin.value()
        smooth_window_samples = self.tip_roi_opening_jump_smooth_spin.value()

        def _draw_jump_panel(ax, u_field, v_field, xs, ys, title, missing_note, threshold_px):
            if u_field is None or v_field is None or xs is None or ys is None or tip_result is None:
                ax.text(0.5, 0.5, missing_note, ha="center", va="center", transform=ax.transAxes,
                        fontsize=7, wrap=True)
                ax.set_title(title, fontsize=8)
                return
            try:
                profile = crack_opening_jump_profile(
                    np.asarray(u_field, dtype=float), np.asarray(v_field, dtype=float), xs, ys,
                    tip_result.path_positions, tip_result.path_arc_length,
                    offset_px=offset_px, threshold_px=threshold_px, max_gap_samples=max_gap_samples,
                    smooth_window_samples=smooth_window_samples,
                    **self._opening_jump_continuous_kwargs(), **self._opening_jump_criterion_kwargs(),
                )
                ax.plot(profile.arc_length_s, profile.jump_criterion_px, color="#e67e22", linewidth=1.0,
                        alpha=0.55 if smooth_window_samples > 1 else 1.0,
                        label=f"{_jump_signal_label(profile)} -- raw")
                if smooth_window_samples > 1:
                    ax.plot(profile.arc_length_s, profile.jump_mag_smoothed_px, color="#c0392b",
                            linewidth=1.2, label=f"|jump| smoothed ({smooth_window_samples})")
                ax.axhline(threshold_px, color="0.6", linestyle="--", linewidth=1.0,
                           label=f"threshold = {threshold_px:.3g} px")
                if profile.opening_extent_s is not None:
                    ax.axvline(profile.opening_extent_s, color="#f1c40f", linewidth=1.5, alpha=0.9,
                               label=f"front (s={profile.opening_extent_s:.1f} px)")
                ax.set_xlabel("distance along path, s (px)", fontsize=7)
                ax.set_ylabel("jump (px)", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.legend(fontsize=6, loc="best", framealpha=0.7)
            except CrackTipTrackingError as exc:
                ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes,
                        fontsize=7, wrap=True)
            ax.set_title(title, fontsize=8)

        dic_uv_frame = None
        if self._dataset is not None and 0 <= sample.source_frame_index < len(self._dataset.frames):
            dic_uv_frame = self._dataset.frames[sample.source_frame_index]
        _draw_jump_panel(
            ax_jump_dic,
            dic_uv_frame.u if dic_uv_frame is not None else None,
            dic_uv_frame.v if dic_uv_frame is not None else None,
            self._dataset.xs if self._dataset is not None else None,
            self._dataset.ys if self._dataset is not None else None,
            "Opening-jump profile -- standard subset-DIC",
            "No standard-DIC field\nfor this frame",
            # This panel diagnoses the SAME ground-truth method Stage 3's
            # "Crack Tip Detection" card uses (`track_tip_opening_jump`),
            # so it reads that card's own threshold, not X-FEM-DIC's.
            self.tip_roi_opening_jump_threshold_spin.value(),
        )
        _draw_jump_panel(
            ax_jump_xfem, grid.u, grid.v, grid.x, grid.y,
            "Opening-jump profile -- X-FEM-DIC",
            "No X-FEM-DIC field\nfor this frame",
            # This panel diagnoses X-FEM-DIC's OWN fitted field -- it must
            # read the "Jump threshold" spinbox dedicated to X-FEM-DIC's
            # "heaviside_jump" auto-tip-detection mode
            # (`xfem_tip_jump_threshold_spin`/`heaviside_jump_threshold_px`),
            # not the unrelated Crack Tip Detection card's own threshold.
            self.xfem_tip_jump_threshold_spin.value(),
        )

        self.xfem_isocontour_canvas.refresh()

    def _sync_xfem_video_export_enabled(self):
        """"Export video" only makes sense once there's a computed result
        AND "Show trace" points at one of them -- same shape as
        `_sync_tip_video_export_enabled` above. Called after Run/Preview
        repopulate `self._xfem_results` and whenever "Show trace" changes."""
        index = self.xfem_critical_combo.currentIndex()
        enabled = (bool(self._xfem_results) and 0 <= index < len(self._xfem_results)
                   and self._xfem_results[index].get("method") != "direct")
        self.xfem_video_export_button.setEnabled(enabled)

    def _on_xfem_video_export_clicked(self):
        """Renders the trace picked by "Show trace" over its own whole
        computed frame range into an MP4: `_XfemVideoExportWorker`/
        `core.xfem_video_export.export_xfem_video` do the actual work off
        the UI thread, so this just gathers the fixed, whole-video color/
        axis scales (color scale from "Color scale (both panels)" above
        if Fixed, else pooled over every frame in the trace so the
        video's own colorbar doesn't rescale frame to frame -- see
        core/dic_video_export.py's own reasoning) and starts it."""
        index = self.xfem_critical_combo.currentIndex()
        if not self._xfem_results or not (0 <= index < len(self._xfem_results)):
            self.xfem_video_status.setText("Pick a trace with \"Show trace\" above first.")
            return
        entry = self._xfem_results[index]
        series = entry["series"]
        if entry.get("method") == "direct":
            self.xfem_video_status.setText(
                f"\"{entry['label']}\" is a direct measured-field K result -- the video needs an "
                "X-FEM-DIC fit. Pick an X-FEM-DIC trace in \"Show trace\"."
            )
            return
        if not series.samples:
            self.xfem_video_status.setText(f"\"{entry['label']}\" has no computed frames.")
            return
        tip_result = entry["tip_result"]
        if tip_result is None:
            self.xfem_video_status.setText(
                f"\"{entry['label']}\": the underlying tip track is no longer available (trace "
                f"removed since this was computed?) -- re-run to refresh."
            )
            return
        if self._dataset is None:
            self.xfem_video_status.setText("Load a DIC results CSV above first.")
            return

        default_name = f"{entry['label'].replace(' ', '_')}_xfem.mp4"
        path, _ = QFileDialog.getSaveFileName(self, "Export X-FEM-DIC video", default_name,
                                               "MP4 video (*.mp4)")
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"

        material = self.material_properties()
        kappa = kolosov_kappa(material["poisson_ratio"], material["plane_stress"])
        mu_pa = shear_modulus(material["young_modulus_pa"], material["poisson_ratio"])
        field_key = self.xfem_field_combo.currentData()
        field_label = self.xfem_field_combo.currentText()
        vmin, vmax = self._current_xfem_isocontour_scale_vmin_vmax()  # (None, None) when Auto

        self.xfem_video_export_button.setEnabled(False)
        self.xfem_video_progress_bar.setMaximum(max(1, len(series.samples)))
        self.xfem_video_progress_bar.setValue(0)
        self.xfem_video_status.setText(
            f"Rendering {len(series.samples)} frame(s) to \"{os.path.basename(path)}\"… this can "
            "take a while (one full two-isocontour-panel-plus-plot figure per frame)."
        )

        self._xfem_video_thread = QThread()
        self._xfem_video_worker = _XfemVideoExportWorker(
            path, self._dataset, self._dataset.xs, self._dataset.ys, field_key, field_label,
            series, tip_result, kappa, mu_pa, self.xfem_sample_step_spin.value(),
            self.xfem_show_KII_check.isChecked(), vmin, vmax, self.xfem_video_fps_spin.value(),
            pixel_size_mm_display=(self.pixel_size_spin.value() or None),
        )
        self._xfem_video_worker.moveToThread(self._xfem_video_thread)
        self._xfem_video_thread.started.connect(self._xfem_video_worker.run)
        self._xfem_video_worker.progress.connect(self._on_xfem_video_export_progress)
        self._xfem_video_worker.finished.connect(self._on_xfem_video_export_finished)
        self._xfem_video_worker.failed.connect(self._on_xfem_video_export_failed)
        self._xfem_video_worker.finished.connect(self._xfem_video_thread.quit)
        self._xfem_video_worker.failed.connect(self._xfem_video_thread.quit)
        self._xfem_video_thread.start()

    def _on_xfem_video_export_progress(self, i, total):
        self.xfem_video_progress_bar.setMaximum(max(1, total))
        self.xfem_video_progress_bar.setValue(i)

    def _on_xfem_video_export_finished(self, path, n_frames):
        self._sync_xfem_video_export_enabled()
        self.xfem_video_status.setText(f"Exported {n_frames} frame(s) to {os.path.basename(path)}.")

    def _on_xfem_video_export_failed(self, message):
        self._sync_xfem_video_export_enabled()
        self.xfem_video_status.setText(f"Video export failed: {message}")

    # -- read by later stages ---------------------------------------------

    def material_properties(self) -> dict:
        """(E_pa, nu, plane_stress) -- read by the wave-speed / crack-
        velocity / K1c stages added on top of this page."""
        return {
            "young_modulus_pa": self.young_modulus_spin.value() * 1e9,
            "poisson_ratio": self.poisson_spin.value(),
            "plane_stress": self.plane_stress_radio.isChecked(),
        }
