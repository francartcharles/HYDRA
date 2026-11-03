"""
Renders the Crack & Wave Analysis page's "Field isocontours" card (X-FEM-DIC
tab) -- standard subset-DIC isocontours side by side with the X-FEM-DIC
fitted field, same frame, same color scale (see `_draw_xfem_isocontours`) --
for every frame of a chosen X-FEM-DIC result trace, plus a live K_I/K_II
(see `_draw_xfem_plot`) and crack-tip speed panel BOTH plotted against
distance travelled along the path (not time), growing frame by frame, and
encodes the sequence into an MP4. Mirrors core/tip_video_export.py's and
core/dic_video_export.py's conventions (plain white background, fixed
caller-supplied color/axis scales so consecutive frames stay visually
comparable, a plain non-Qt Agg figure so this is safe to call from a
background QThread) rather than introducing a fourth, inconsistent style of
exported video in the same app.

Layout: TOP is two side-by-side isocontour panels (standard subset-DIC,
X-FEM-DIC's own fitted field), each with the traced path (thin dotted) and
the per-frame tip trajectory overlaid the same way `_draw_xfem_isocontours`
does -- walked along the REAL traced path
(`core.crack_tip_tracking.path_polyline_up_to_arc_length`) rather than a
straight line between two points, for the same reason
core/tip_video_export.py's docstring explains. BOTTOM is a single dual-axis
panel: K_I (and K_II, if the caller's data has it) on the left axis, crack
speed on the right, BOTH against distance travelled along the path on the
shared x-axis -- not time, since "how did the fracture toughness change as
the crack advanced" is the physically meaningful question here, not "at
what wall-clock instant".
"""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from core.dic_render import isocontour_levels, isocontour_colorbar
from core.crack_tip_tracking import path_polyline_up_to_arc_length

DEFAULT_FIGSIZE = (14, 10)
DEFAULT_DPI = 150

_K1_COLOR = "#2e86de"
_K2_COLOR = "#8e44ad"
_SPEED_COLOR = "#e67e22"


def render_xfem_frame(xs: np.ndarray, ys: np.ndarray, dic_field_arr: np.ndarray | None, field_label: str,
                       xfem_grid_x: np.ndarray | None, xfem_grid_y: np.ndarray | None,
                       xfem_field_arr: np.ndarray | None,
                       vmin: float, vmax: float, path_positions: np.ndarray, path_arc_length: np.ndarray,
                       current_tip_xy: tuple | None, trajectory_max_s: float | None,
                       distances: np.ndarray, k1_values: np.ndarray, k2_values: np.ndarray | None,
                       velocity_values: np.ndarray, current_index: int,
                       distance_max: float, k_range: tuple, velocity_range: tuple,
                       distance_label: str = "distance along path (px)", k_label: str = "K (MPa·√m)",
                       velocity_label: str = "speed (px/s)", cmap: str = "jet",
                       timestamp_text: str | None = None,
                       figsize=DEFAULT_FIGSIZE, dpi: int = DEFAULT_DPI) -> np.ndarray:
    """Renders one video frame and returns it as an (H, W, 3) RGB uint8
    array -- see this module's docstring for the layout.

    `dic_field_arr`/`xfem_field_arr` are this ONE frame's own field values
    (either can be None -- no standard-DIC field for this frame, or a
    failed X-FEM-DIC fit -- drawn as a "no field" placeholder instead).
    `xfem_grid_x`/`xfem_grid_y` are that frame's own evaluated grid
    coordinates (an X-FEM-DIC grid can differ frame to frame since it is
    re-meshed around each frame's own tip position).

    `path_positions`/`path_arc_length` are the FIXED traced-path geometry
    (shared by every frame); `trajectory_max_s` is how far along it the
    latest known tip position reaches (None if nothing has been located
    yet) -- see `path_polyline_up_to_arc_length`. `current_tip_xy` is
    THIS frame's own tip position (drawn as a "+", regardless of whether
    its fit converged -- matching `_draw_xfem_isocontours`'s own
    convention, since that marker means "this is where the geometry
    assumed the tip was", not "this frame's K fit succeeded").

    `distances`/`k1_values`/`k2_values`/`velocity_values` are the WHOLE
    trace's own per-frame arrays (already unit-converted by the caller to
    whatever `distance_label`/`k_label`/`velocity_label` name); this frame
    plots only `[:current_index + 1]` of them, the same "grow it frame by
    frame" convention core/dic_video_export.py's live mean plot and
    core/tip_video_export.py's position/speed plot both already use.
    `k2_values=None` skips the K_II line entirely (the caller's "show K_II"
    checkbox was off). `distance_max`/`k_range`/`velocity_range` are
    caller-fixed axis bounds (covering the WHOLE video) so the bottom
    panel doesn't rescale itself frame to frame."""
    fig = Figure(figsize=figsize, dpi=dpi, facecolor="white", layout="constrained")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 2, height_ratios=(3, 1.3), hspace=0.12, wspace=0.25)

    ax_dic = fig.add_subplot(gs[0, 0])
    ax_xfem = fig.add_subplot(gs[0, 1], sharex=ax_dic, sharey=ax_dic)

    def _overlay(ax):
        ax.set_facecolor("white")
        ax.plot(path_positions[:, 0], path_positions[:, 1], color="0.35", linewidth=0.9,
                 linestyle=":", alpha=0.8, zorder=4)
        if trajectory_max_s is not None and np.isfinite(trajectory_max_s):
            traj = path_polyline_up_to_arc_length(path_positions, path_arc_length, trajectory_max_s)
            ax.plot(traj[:, 0], traj[:, 1], color="black", linewidth=2.0, alpha=0.9, zorder=5)
            ax.plot(traj[:, 0], traj[:, 1], color="white", linewidth=0.8, linestyle="--", zorder=6)
        if current_tip_xy is not None:
            ax.plot([current_tip_xy[0]], [current_tip_xy[1]], marker="+", color="white",
                     markersize=13, markeredgewidth=2.4, linestyle="none", zorder=7)
            ax.plot([current_tip_xy[0]], [current_tip_xy[1]], marker="+", color="black",
                     markersize=11, markeredgewidth=1.2, linestyle="none", zorder=8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (px)", color="black")
        ax.set_ylabel("y (px)", color="black")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")

    levels = isocontour_levels(vmin, vmax)
    if dic_field_arr is not None:
        masked = np.ma.masked_invalid(dic_field_arr)
        if np.ma.count(masked) > 0:
            cf = ax_dic.contourf(xs, ys, masked, levels=levels, cmap=cmap, extend="both")
            cbar = isocontour_colorbar(fig, cf, ax=ax_dic, label=field_label, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.label.set_color("black")
            cbar.ax.tick_params(colors="black")
    else:
        ax_dic.text(0.5, 0.5, "No standard-DIC field\nfor this frame/field", ha="center", va="center",
                     transform=ax_dic.transAxes, fontsize=8, color="black")
    _overlay(ax_dic)
    ax_dic.set_title("Standard subset-DIC", fontsize=11, color="black")

    if xfem_field_arr is not None and xfem_grid_x is not None:
        masked = np.ma.masked_invalid(xfem_field_arr)
        if np.ma.count(masked) > 0:
            cf = ax_xfem.contourf(xfem_grid_x, xfem_grid_y, masked, levels=levels, cmap=cmap,
                                    extend="both")
            cbar = isocontour_colorbar(fig, cf, ax=ax_xfem, label=field_label, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.label.set_color("black")
            cbar.ax.tick_params(colors="black")
    else:
        ax_xfem.text(0.5, 0.5, "No X-FEM-DIC field\nfor this frame/field", ha="center", va="center",
                       transform=ax_xfem.transAxes, fontsize=8, color="black")
    _overlay(ax_xfem)
    ax_xfem.set_title("X-FEM-DIC", fontsize=11, color="black")
    ax_dic.invert_yaxis()  # image pixel convention (y downward); shared axes -> flips both panels

    if timestamp_text:
        fig.suptitle(timestamp_text, fontsize=11, color="black")

    ax_k = fig.add_subplot(gs[1, :])
    ax_k.set_facecolor("white")
    ax_v = ax_k.twinx()
    n = current_index + 1
    d = np.asarray(distances[:n], dtype=float)
    k1 = np.asarray(k1_values[:n], dtype=float)
    finite_k1 = np.isfinite(d) & np.isfinite(k1)
    if np.any(finite_k1):
        ax_k.plot(d[finite_k1], k1[finite_k1], "o-", color=_K1_COLOR, markersize=3, linewidth=1.2,
                   zorder=5, label="K_I")
    if k2_values is not None:
        k2 = np.asarray(k2_values[:n], dtype=float)
        finite_k2 = np.isfinite(d) & np.isfinite(k2)
        if np.any(finite_k2):
            ax_k.plot(d[finite_k2], k2[finite_k2], "s--", color=_K2_COLOR, markersize=3, linewidth=1.0,
                       zorder=4, label="K_II")
    v = np.asarray(velocity_values[:n], dtype=float)
    finite_v = np.isfinite(d) & np.isfinite(v)
    if np.any(finite_v):
        ax_v.plot(d[finite_v], v[finite_v], "^:", color=_SPEED_COLOR, markersize=3, linewidth=1.0,
                   alpha=0.85, zorder=3, label="speed")
    if np.any(finite_v) or np.any(finite_k1):
        last_d = d[np.isfinite(d)][-1] if np.any(np.isfinite(d)) else None
        if last_d is not None:
            ax_k.axvline(last_d, color="0.5", linewidth=1.0, linestyle=":", zorder=2)

    if distance_max and distance_max > 0:
        ax_k.set_xlim(0.0, distance_max * 1.05)
    k_lo, k_hi = k_range
    if np.isfinite(k_lo) and np.isfinite(k_hi) and k_hi > k_lo:
        pad = 0.05 * (k_hi - k_lo)
        ax_k.set_ylim(k_lo - pad, k_hi + pad)
    v_lo, v_hi = velocity_range
    if np.isfinite(v_lo) and np.isfinite(v_hi) and v_hi > v_lo:
        pad = 0.05 * (v_hi - v_lo)
        ax_v.set_ylim(v_lo - pad, v_hi + pad)

    ax_k.set_xlabel(distance_label, color="black")
    ax_k.set_ylabel(k_label, color=_K1_COLOR)
    ax_v.set_ylabel(velocity_label, color=_SPEED_COLOR)
    ax_k.grid(True, alpha=0.25)
    ax_k.tick_params(axis="y", colors=_K1_COLOR)
    ax_v.tick_params(axis="y", colors=_SPEED_COLOR)
    ax_k.tick_params(axis="x", colors="black")
    for spine in ax_k.spines.values():
        spine.set_color("black")
    for spine in ax_v.spines.values():
        spine.set_color("black")
    lines1, labels1 = ax_k.get_legend_handles_labels()
    lines2, labels2 = ax_v.get_legend_handles_labels()
    if lines1 or lines2:
        ax_k.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(buf[:, :, :3])


def export_xfem_video(path: str, frames_iterable: Iterable[dict], n_frames: int, fps: float,
                       progress_cb=None) -> None:
    """Writes an MP4 (mp4v fourcc -- built into every OpenCV build, no
    external ffmpeg needed) at `path`, `fps` frames/second. `frames_iterable`
    yields one kwargs-dict per frame, each passed straight to
    `render_xfem_frame` -- consumed lazily so memory use doesn't grow with
    the number of frames. `n_frames` is only used to report progress.
    Mirrors core/tip_video_export.py's `export_tip_video` and
    core/dic_video_export.py's `export_field_video` exactly."""
    if n_frames <= 0:
        raise ValueError("No frames to export.")

    writer = None
    try:
        for i, frame_kwargs in enumerate(frames_iterable, start=1):
            frame_rgb = render_xfem_frame(**frame_kwargs)
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
