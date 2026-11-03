"""
Renders the Image Correlation page's Results view (background frame +
one field's isocontours plotted at each point's actual tracked position
in that frame -- see core/dic_render.py) for every processed frame of a
DIC run, and encodes the sequence into an MP4 video -- so a run's
evolution over time can be reviewed outside the app, or shared without
it.

Unlike the live Results canvas (which matches the app's own dark theme
and whatever size its panel happens to be), an exported video is always
rendered on a plain WHITE background at a fixed, deliberately high
resolution -- it's a standalone deliverable meant to be watched
full-screen or shared with someone who never sees the app itself, not a
screen capture of a UI panel -- and can carry a per-frame timestamp
burned into the image.

When a mean-value time series is supplied (see `render_field_frame`'s
`mean_series`/`time_series`), each frame is rendered as TWO panels side
by side: the isocontour on the left (as before), and a live line plot of
the field's ROI mean, growing up to the current frame, on the right --
so the video shows the field's spatial pattern and its time evolution
together instead of leaving that whole right side of the frame empty.

Rendering uses a plain (non-Qt) matplotlib Agg figure -- never a
PlotCanvas/FigureCanvasQTAgg widget -- so this module is safe to call
from a background QThread: it never touches a Qt widget or the Qt event
loop, matching how core/dic_processing.run_dic is itself safe to run off
the main thread.

The color scale (`vmin`/`vmax`) is a caller-supplied constant, not
recomputed per frame, so consecutive frames of the video are visually
comparable -- an auto-scaled colorbar that changed range every frame
would make the tracked field's actual evolution unreadable. The live
plot's own axes (`mean_vmin`/`mean_vmax`, and the full time range) are
likewise caller-supplied constants, for the same reason.
"""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from core.dic_render import render_field

# Deliberately larger and higher-DPI than the live Results canvas (whose
# on-screen panel is typically an 800x550-ish screen capture) -- a video
# is watched full-screen or shared standalone, so it should hold up
# there: 12 x 9 in at 160 dpi is 1920 x 1440 px.
DEFAULT_FIGSIZE = (12, 9)
DEFAULT_DPI = 160

_LIVE_PLOT_LINE_COLOR = "#5b8cff"
_LIVE_PLOT_MARKER_COLOR = "#ff6b6b"


def render_field_frame(background, x: np.ndarray, y: np.ndarray, values: np.ndarray, field_label: str,
                        vmin: float, vmax: float, cmap: str = "jet", timestamp_text: str | None = None,
                        mean_series=None, time_series=None, current_index: int | None = None,
                        time_label: str = "time", mean_vmin: float | None = None,
                        mean_vmax: float | None = None, figsize=DEFAULT_FIGSIZE,
                        dpi: int = DEFAULT_DPI, max_edge_px: float | None = None) -> np.ndarray:
    """Renders one frame -- `background` (a 2D grayscale or 3D RGB uint8
    array, or None to skip it) with `values` overlaid as filled contours
    at each point's `(x, y)` position (see core/dic_render.render_field)
    -- on a plain white background, with `timestamp_text` (if given)
    burned into the top-left corner, and returns it as an (H, W, 3) RGB
    uint8 array.

    `x`/`y` should be each point's ACTUAL position in the frame being
    rendered (a DICFrameResult's own `.x`/`.y`, not the mesh's fixed
    reference grid) so the plotted field follows the tracked material as
    it moves -- a "deformed mesh" view -- rather than staying glued to
    its footprint in the reference frame. A point with no reliable
    position this frame (NaN `x`/`y`/`values`) is simply left out.

    `max_edge_px` (see core/dic_render.py's `default_max_edge_px`) masks
    out any triangle of the field's Delaunay triangulation with an edge
    longer than it -- otherwise a non-rectangular ROI's excluded notch
    gets bridged and shaded as if it were real data.

    When `mean_series`, `time_series` and `current_index` are all given,
    a second panel is added on the right: `mean_series[:current_index+1]`
    plotted against `time_series[:current_index+1]`, i.e. this field's
    ROI mean value from the start of the sequence up to (and including)
    this frame -- built up frame by frame as the video plays, with a
    fixed x range (the full `time_series`) and y range (`mean_vmin`/
    `mean_vmax`) so the panel doesn't rescale itself frame to frame.
    Left at their default None, only the single field panel is drawn
    (unchanged from before this parameter existed)."""
    # constrained_layout (rather than a post-hoc tight_layout() call)
    # handles the two-panel-plus-colorbar case cleanly -- tight_layout()
    # doesn't know how to place a colorbar's own axes inside a gridspec.
    fig = Figure(figsize=figsize, dpi=dpi, facecolor="white", layout="constrained")
    FigureCanvasAgg(fig)

    has_live_plot = mean_series is not None and time_series is not None and current_index is not None
    if has_live_plot:
        gs = fig.add_gridspec(1, 2, width_ratios=(3, 2), wspace=0.3)
        ax = fig.add_subplot(gs[0, 0])
    else:
        ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    render_field(fig, ax, background, x, y, values, field_label, vmin=vmin, vmax=vmax,
                 cmap=cmap, timestamp_text=timestamp_text, text_color="black", max_edge_px=max_edge_px)

    if has_live_plot:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor("white")
        _render_live_mean_plot(ax2, mean_series, time_series, current_index, field_label,
                                time_label, mean_vmin, mean_vmax)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(buf[:, :, :3])


def _render_live_mean_plot(ax, mean_series, time_series, current_index: int, field_label: str,
                            time_label: str, mean_vmin: float | None, mean_vmax: float | None) -> None:
    """Draws the growing ROI-mean-vs-time line for the video's right-hand
    panel -- see `render_field_frame`. NaN entries (e.g. a frame where
    every tracked point was lost) are simply skipped, same convention as
    `core.dic_render.render_field`'s own handling of a lost point."""
    t = np.asarray(time_series[:current_index + 1], dtype=float)
    v = np.asarray(mean_series[:current_index + 1], dtype=float)
    finite = np.isfinite(t) & np.isfinite(v)
    if np.any(finite):
        ax.plot(t[finite], v[finite], "-", color=_LIVE_PLOT_LINE_COLOR, linewidth=2)
        ax.plot(t[finite][-1:], v[finite][-1:], "o", color=_LIVE_PLOT_MARKER_COLOR,
                 markersize=7, zorder=5)

    t_all = np.asarray(time_series, dtype=float)
    if t_all.size >= 2 and np.all(np.isfinite((t_all[0], t_all[-1]))) and t_all[-1] > t_all[0]:
        ax.set_xlim(t_all[0], t_all[-1])
    if mean_vmin is not None and mean_vmax is not None and mean_vmax > mean_vmin:
        ax.set_ylim(mean_vmin, mean_vmax)

    ax.set_xlabel(time_label)
    ax.set_ylabel(f"mean {field_label}")
    ax.set_title("ROI mean (live)", fontsize=11, color="black")
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.tick_params(colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")


def export_field_video(path: str, frames_iterable: Iterable[dict], n_frames: int, fps: float,
                        progress_cb=None) -> None:
    """Writes an MP4 (mp4v fourcc -- built into every OpenCV build, no
    external ffmpeg needed) at `path`, `fps` frames/second. `fps` is a
    playback rate for the exported video, independent of the camera's own
    capture rate (which is typically far too high to watch directly, and
    doesn't apply 1:1 anyway once a frame step has decimated the sequence).

    `frames_iterable` yields one kwargs-dict per frame, each passed
    straight to `render_field_frame` -- consumed lazily (one frame
    rendered and written at a time) so memory use doesn't grow with the
    number of frames. `n_frames` is only used to report progress."""
    if n_frames <= 0:
        raise ValueError("No frames to export.")

    writer = None
    try:
        for i, frame_kwargs in enumerate(frames_iterable, start=1):
            frame_rgb = render_field_frame(**frame_kwargs)
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
