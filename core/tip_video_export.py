"""
Renders the Crack & Wave Analysis page's "Crack Tip Detection" card --
standard subset-DIC isocontours with a detected "opening_jump" trace's
trajectory overlaid, plus a live position-vs-time / speed-vs-time panel
growing frame by frame -- for every frame of a chosen frame range, and
encodes the sequence into an MP4 video, the same "watch the run's own
evolution outside the app" purpose core/dic_video_export.py already
serves for the Image Correlation page's Results view. Deliberately
mirrors that module's conventions (plain white background, fixed
caller-supplied color/axis scales so consecutive frames stay visually
comparable, a plain non-Qt Agg figure so this is safe to call from a
background QThread) rather than introducing a second, inconsistent style
of exported video in the same app.

The two panels differ from core/dic_video_export.py's in what they show,
not in how: TOP is `_draw_tip_verify_frame`'s own isocontour-plus-
trajectory view (the traced path, the detected trajectory up to the
current frame, a marker at the current frame's own detected position if
any), not a per-point scatter at tracked positions -- a crack-tip trace
lives on a fixed traced path, not moving material points. BOTTOM is a
dual-axis position/speed-vs-time plot (see `trace_velocity_px_per_s`),
not a single-field ROI mean -- "position" and "speed" are two different
physical quantities on two different y-axes here, not two panels.

The top panel's "trajectory so far" is drawn by walking the crack's own
traced path polyline (`core.crack_tip_tracking.path_polyline_up_to_
arc_length`) from its origin up to however far the latest detected point
reaches, NOT by connecting the (temporally sparse) detected (x, y)
points with straight lines. A real crack can propagate fast enough that
several frames pass between two threshold-crossing detections -- nothing
wrong with that, the crack is just accelerating -- but the straight-line
approach used to visibly cut a corner there, drawing a chord across
whatever the path actually does between those two points (which can even
leave the crack's own visible band entirely on a curved/branchy trace).
The path's shape between any two points on it is already fully known
from tracing/auto-detection, so there's no reason to approximate it.
"""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from core.dic_render import isocontour_levels, isocontour_colorbar
from core.crack_tip_tracking import nearest_path_point, path_polyline_up_to_arc_length
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Same deliberately-larger-than-the-live-canvas sizing as
# core/dic_video_export.py, for the same reason (a video is watched
# full-screen or shared standalone) -- kept identical so every video this
# app exports looks like it came from the same tool.
DEFAULT_FIGSIZE = (12, 9)
DEFAULT_DPI = 160

_POSITION_COLOR = "#2e86de"
_VELOCITY_COLOR = "#e67e22"
_CURRENT_MARKER_COLOR = "#e74c3c"


def render_tip_frame(xs: np.ndarray, ys: np.ndarray, field_arr: np.ndarray, field_label: str,
                      vmin: float, vmax: float, path_positions: np.ndarray, path_arc_length: np.ndarray,
                      trace_times: np.ndarray, trace_arc_length: np.ndarray,
                      trace_velocity: np.ndarray, trace_positions_xy: np.ndarray,
                      current_time: float, current_xy: tuple | None,
                      time_range: tuple, arc_length_max: float, velocity_range: tuple,
                      position_label: str = "position (px)", velocity_label: str = "speed (px/s)",
                      cmap: str = "jet", timestamp_text: str | None = None,
                      trace_position_uncertainty: np.ndarray | None = None,
                      trace_velocity_uncertainty: np.ndarray | None = None,
                      uncertainty_sigma_n: int = 0,
                      figsize=DEFAULT_FIGSIZE, dpi: int = DEFAULT_DPI) -> np.ndarray:
    """Renders one video frame and returns it as an (H, W, 3) RGB uint8
    array -- see this module's docstring for the two panels' layout.

    `trace_times`/`trace_arc_length`/`trace_velocity`/`trace_positions_xy`
    are the trace's own (sparse -- not one point per video frame, only
    the frames a detection actually landed on) full arrays; this frame
    plots and marks-current only whatever falls at/before `current_time`,
    the same "grow it frame by frame" convention core/dic_video_export.
    py's live mean plot uses. `time_range`/`arc_length_max`/
    `velocity_range` are caller-fixed axis bounds (covering the WHOLE
    video, computed once up front) so, like the color scale, the bottom
    panel doesn't rescale itself frame to frame.

    `trace_arc_length`/`trace_velocity`/`arc_length_max`/`velocity_range`
    are already in whatever unit `position_label`/`velocity_label` name
    (px or mm, px/s or mm/s) -- the CALLER converts, this function only
    labels the axis; `trace_positions_xy`/`path_positions`/`path_arc_length`
    /`xs`/`ys` (the top panel's spatial image coordinates AND its own
    arc-length parametrization) are always px, unrelated to this unit
    choice -- a physical position on the sensor grid, not a length
    measured along the path, and the top panel's "trajectory so far" is
    drawn by walking `path_positions` itself (see `path_polyline_up_to_
    arc_length` and this module's docstring), never by directly plotting
    `trace_arc_length`.

    `trace_position_uncertainty`/`trace_velocity_uncertainty` (same
    shape as `trace_arc_length`/`trace_velocity`, same unit -- the
    CALLER converts, exactly like those two) are optional: when given
    together with `uncertainty_sigma_n > 0`, a shaded +/- N sigma band
    is drawn behind both the position and speed curves in the bottom
    panel, mirroring the "Verify visually" viewer's own "Uncertainty
    envelope" dropdown (see `page_crack_wave_analysis.py`'s
    `_refresh_tip_verify_arc_length_plot`) so a video exported with that
    option on shows the same uncertainty the live viewer does. Left
    `None`/`0` (the default) draws the plain curves only, unchanged from
    before this option existed."""
    fig = Figure(figsize=figsize, dpi=dpi, facecolor="white", layout="constrained")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.15)

    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("white")
    masked = np.ma.masked_invalid(field_arr)
    if np.ma.count(masked) > 0:
        levels = isocontour_levels(vmin, vmax)
        cf = ax.contourf(xs, ys, masked, levels=levels, cmap=cmap, extend="both")
        cbar = isocontour_colorbar(fig, cf, ax=ax, label=field_label, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.label.set_color("black")
        cbar.ax.tick_params(colors="black")
    ax.plot(path_positions[:, 0], path_positions[:, 1], color="0.35", linewidth=0.9,
             linestyle=":", alpha=0.8, zorder=4)

    mask_so_far = np.isfinite(trace_times) & (trace_times <= current_time)
    if np.count_nonzero(mask_so_far) >= 1:
        # Walk the REAL traced path from its origin up to however far the
        # latest detected point (by now) reached -- not a straight line
        # between the sparse detected (x, y) points, which cuts corners
        # across whatever the path actually does between two detections
        # that happen to be several frames apart (e.g. the crack sped up).
        # See this module's docstring and `path_polyline_up_to_arc_length`.
        last_xy = trace_positions_xy[mask_so_far][-1]
        _, _, last_s, _ = nearest_path_point(path_positions, path_arc_length,
                                              float(last_xy[0]), float(last_xy[1]))
        traj = path_polyline_up_to_arc_length(path_positions, path_arc_length, last_s)
        ax.plot(traj[:, 0], traj[:, 1], color="black", linewidth=2.0, alpha=0.9, zorder=5)
        ax.plot(traj[:, 0], traj[:, 1], color="white", linewidth=0.8, linestyle="--", zorder=6)
        # Small markers at the actual (temporally sparse) detected points,
        # so where real detections landed along that path stays visible.
        detected = trace_positions_xy[mask_so_far]
        ax.scatter(detected[:, 0], detected[:, 1], s=12, color="black", zorder=6,
                   edgecolors="white", linewidths=0.4)
    if current_xy is not None:
        ax.plot([current_xy[0]], [current_xy[1]], marker="o", color=_CURRENT_MARKER_COLOR,
                 markersize=10, markeredgewidth=1.5, markeredgecolor="black", zorder=7)

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel("x (px)", color="black")
    ax.set_ylabel("y (px)", color="black")
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    if timestamp_text:
        ax.set_title(timestamp_text, fontsize=12, color="black")

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("white")
    ax3 = ax2.twinx()
    if np.count_nonzero(mask_so_far) >= 1:
        t = trace_times[mask_so_far]
        s = trace_arc_length[mask_so_far]
        v = trace_velocity[mask_so_far]
        has_uncertainty_band = (
            uncertainty_sigma_n > 0
            and trace_position_uncertainty is not None
            and trace_velocity_uncertainty is not None
        )
        if has_uncertainty_band:
            unc_s = np.asarray(trace_position_uncertainty)[mask_so_far]
            unc_s_finite = np.isfinite(unc_s)
            if np.any(unc_s_finite):
                ax2.fill_between(t[unc_s_finite],
                                  s[unc_s_finite] - uncertainty_sigma_n * unc_s[unc_s_finite],
                                  s[unc_s_finite] + uncertainty_sigma_n * unc_s[unc_s_finite],
                                  color=_POSITION_COLOR, alpha=0.18, linewidth=0, zorder=2)
        ax2.plot(t, s, "o-", color=_POSITION_COLOR, markersize=3, linewidth=1.4, zorder=5)
        v_finite = np.isfinite(v)
        if np.any(v_finite):
            if has_uncertainty_band:
                unc_v = np.asarray(trace_velocity_uncertainty)[mask_so_far]
                band_finite = v_finite & np.isfinite(unc_v)
                if np.any(band_finite):
                    ax3.fill_between(t[band_finite],
                                      v[band_finite] - uncertainty_sigma_n * unc_v[band_finite],
                                      v[band_finite] + uncertainty_sigma_n * unc_v[band_finite],
                                      color=_VELOCITY_COLOR, alpha=0.12, linewidth=0, zorder=1)
            ax3.plot(t[v_finite], v[v_finite], "--", color=_VELOCITY_COLOR, linewidth=1.2,
                     alpha=0.85, zorder=4)
        ax2.plot(t[-1:], s[-1:], "o", color=_CURRENT_MARKER_COLOR, markersize=7, zorder=6)
    ax2.axvline(current_time, color="0.5", linewidth=1.0, linestyle=":", zorder=3)

    t_lo, t_hi = time_range
    if np.isfinite(t_lo) and np.isfinite(t_hi) and t_hi > t_lo:
        ax2.set_xlim(t_lo, t_hi)
    if arc_length_max and arc_length_max > 0:
        ax2.set_ylim(0.0, arc_length_max * 1.05)
    v_lo, v_hi = velocity_range
    if np.isfinite(v_lo) and np.isfinite(v_hi) and v_hi > v_lo:
        pad = 0.05 * (v_hi - v_lo)
        ax3.set_ylim(v_lo - pad, v_hi + pad)

    ax2.set_xlabel("time (s)", color="black")
    ax2.set_ylabel(position_label, color=_POSITION_COLOR)
    ax3.set_ylabel(velocity_label, color=_VELOCITY_COLOR)
    ax2.grid(True, alpha=0.25)
    ax2.tick_params(axis="y", colors=_POSITION_COLOR)
    ax3.tick_params(axis="y", colors=_VELOCITY_COLOR)
    ax2.tick_params(axis="x", colors="black")
    for spine in ax2.spines.values():
        spine.set_color("black")
    for spine in ax3.spines.values():
        spine.set_color("black")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(buf[:, :, :3])


def render_multi_tip_frame(xs: np.ndarray, ys: np.ndarray, field_arr: np.ndarray, field_label: str,
                            vmin: float, vmax: float, traces: list, current_time: float,
                            time_range: tuple, arc_length_max: float, velocity_range: tuple,
                            position_label: str = "position (px)", velocity_label: str = "speed (px/s)",
                            cmap: str = "jet", timestamp_text: str | None = None,
                            uncertainty_sigma_n: int = 0,
                            figsize=DEFAULT_FIGSIZE, dpi: int = DEFAULT_DPI) -> np.ndarray:
    """Like `render_tip_frame`, for SEVERAL traces (one per crack) in one
    video: every trace is drawn in its own colour -- its path (faint dotted),
    its trajectory so far (walking its own path, see the module docstring),
    its detected points and its current position on the isocontours, and its
    position (solid, left axis) / speed (dashed, right axis) curves growing in
    the bottom panel, with a legend. Each entry of `traces` is a dict:
    label, color, path_positions, path_arc_length, times, arc_length,
    velocity, positions_xy, current_xy (or None) and optionally
    position_uncertainty / velocity_uncertainty (same units as arc_length /
    velocity -- the caller converts, as for `render_tip_frame`). Axis bounds
    are caller-fixed over the whole video."""
    fig = Figure(figsize=figsize, dpi=dpi, facecolor="white", layout="constrained")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.15)

    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("white")
    masked = np.ma.masked_invalid(field_arr)
    if np.ma.count(masked) > 0:
        levels = isocontour_levels(vmin, vmax)
        cf = ax.contourf(xs, ys, masked, levels=levels, cmap=cmap, extend="both")
        cbar = isocontour_colorbar(fig, cf, ax=ax, label=field_label, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.label.set_color("black")
        cbar.ax.tick_params(colors="black")
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("white")
    ax3 = ax2.twinx()

    for tr in traces:
        color = tr["color"]
        ax.plot(tr["path_positions"][:, 0], tr["path_positions"][:, 1], color="0.35", linewidth=0.8,
                linestyle=":", alpha=0.7, zorder=4)
        t_all = np.asarray(tr["times"], float)
        mask = np.isfinite(t_all) & (t_all <= current_time)
        if np.count_nonzero(mask) >= 1:
            last_xy = np.asarray(tr["positions_xy"])[mask][-1]
            _, _, last_s, _ = nearest_path_point(tr["path_positions"], tr["path_arc_length"],
                                                  float(last_xy[0]), float(last_xy[1]))
            traj = path_polyline_up_to_arc_length(tr["path_positions"], tr["path_arc_length"], last_s)
            ax.plot(traj[:, 0], traj[:, 1], color="white", linewidth=3.2, alpha=0.9, zorder=5)
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.8, zorder=6)
            det = np.asarray(tr["positions_xy"])[mask]
            ax.scatter(det[:, 0], det[:, 1], s=12, color=color, zorder=6, edgecolors="white", linewidths=0.4)
        if tr.get("current_xy") is not None:
            cx, cy = tr["current_xy"]
            ax.plot([cx], [cy], marker="o", color=color, markersize=10, markeredgewidth=1.5,
                    markeredgecolor="black", zorder=7)

        if np.count_nonzero(mask) >= 1:
            t = t_all[mask]
            sarc = np.asarray(tr["arc_length"], float)[mask]
            vel = np.asarray(tr["velocity"], float)[mask]
            pu, vu = tr.get("position_uncertainty"), tr.get("velocity_uncertainty")
            if uncertainty_sigma_n > 0 and pu is not None:
                u = np.asarray(pu, float)[mask]
                ok = np.isfinite(u)
                if np.any(ok):
                    ax2.fill_between(t[ok], sarc[ok] - uncertainty_sigma_n * u[ok],
                                     sarc[ok] + uncertainty_sigma_n * u[ok], color=color, alpha=0.15,
                                     linewidth=0, zorder=2)
            ax2.plot(t, sarc, "o-", color=color, markersize=3, linewidth=1.4, zorder=5, label=tr["label"])
            vf = np.isfinite(vel)
            if np.any(vf):
                if uncertainty_sigma_n > 0 and vu is not None:
                    w = np.asarray(vu, float)[mask]
                    bf = vf & np.isfinite(w)
                    if np.any(bf):
                        ax3.fill_between(t[bf], vel[bf] - uncertainty_sigma_n * w[bf],
                                         vel[bf] + uncertainty_sigma_n * w[bf], color=color, alpha=0.10,
                                         linewidth=0, zorder=1)
                ax3.plot(t[vf], vel[vf], "--", color=color, linewidth=1.1, alpha=0.7, zorder=4)
            ax2.plot(t[-1:], sarc[-1:], "o", color=color, markersize=7, markeredgecolor="black", zorder=6)
    ax2.axvline(current_time, color="0.5", linewidth=1.0, linestyle=":", zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel("x (px)", color="black")
    ax.set_ylabel("y (px)", color="black")
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    if timestamp_text:
        ax.set_title(timestamp_text, fontsize=12, color="black")

    t_lo, t_hi = time_range
    if np.isfinite(t_lo) and np.isfinite(t_hi) and t_hi > t_lo:
        ax2.set_xlim(t_lo, t_hi)
    if arc_length_max and arc_length_max > 0:
        ax2.set_ylim(0.0, arc_length_max * 1.05)
    v_lo, v_hi = velocity_range
    if np.isfinite(v_lo) and np.isfinite(v_hi) and v_hi > v_lo:
        pad = 0.05 * (v_hi - v_lo)
        ax3.set_ylim(v_lo - pad, v_hi + pad)
    ax2.set_xlabel("time (s)", color="black")
    ax2.set_ylabel(position_label + " -- solid", color="black")
    ax3.set_ylabel(velocity_label + " -- dashed", color="black")
    ax2.grid(True, alpha=0.25)
    ax2.tick_params(colors="black")
    ax3.tick_params(colors="black")
    for sp in list(ax2.spines.values()) + list(ax3.spines.values()):
        sp.set_color("black")
    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend(handles, labels, fontsize=8, loc="upper left", framealpha=0.8)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(buf[:, :, :3])


def export_tip_video(path: str, frames_iterable: Iterable[dict], n_frames: int, fps: float,
                      progress_cb=None) -> None:
    """Writes an MP4 (mp4v fourcc -- built into every OpenCV build, no
    external ffmpeg needed) at `path`, `fps` frames/second -- a playback
    rate independent of the camera's own capture rate (typically far too
    high, often in the microsecond range for a dynamic SHPB event, to
    watch directly). `frames_iterable` yields one kwargs-dict per frame,
    each passed straight to `render_tip_frame` -- consumed lazily so
    memory use doesn't grow with the number of frames. `n_frames` is only
    used to report progress. Mirrors core/dic_video_export.py's
    `export_field_video` exactly."""
    if n_frames <= 0:
        raise ValueError("No frames to export.")

    writer = None
    try:
        for i, frame_kwargs in enumerate(frames_iterable, start=1):
            frame_rgb = (render_multi_tip_frame(**frame_kwargs) if "traces" in frame_kwargs
                         else render_tip_frame(**frame_kwargs))
            if writer is None:
                height, width = frame_rgb.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise OSError(f"Could not open '{path}' for writing (mp4v codec).")
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            if progress_cb is not None:
                progress_cb(i, n_frames)
    finally:
        if writer is not None:
            writer.release()
    if writer is None:
        raise ValueError("No frames to export.")
