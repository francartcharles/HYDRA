"""
Shared rendering for one DIC result field on one frame: background image
+ the field as filled contours. Used by both the Image Correlation
page's live Results canvas (ui/pages/page_image_correlation.py) and the
video export renderer (core/dic_video_export.py), so both share exactly
the same visual logic -- and both get the same fix at once.

Draws on a fresh 2D Delaunay triangulation of the points' ACTUAL
positions in this frame (`tricontourf`), rather than the fixed
reference-frame quad grid (`contourf` on `mesh.grid_xy()`) used before.
That distinction only matters once deformation gets large: keeping a
quad's connectivity fixed while letting its four corners move
independently can fold it over on itself wherever neighboring points
disagree even a little (very visible once a specimen necks hard -- a
jagged, spiky, self-crossing patch of color instead of a smooth field),
because contourf's filled regions assume every quad stays a simple,
non-self-intersecting shape. A fresh Delaunay triangulation of a
scattered point set never has that problem, by construction -- the same
fix a badly distorted FEM mesh gets from an actual remeshing step,
applied here to a plotted point cloud instead of a solved mesh.

A point with no reliable position in this frame (lost track, masked out,
or dropped by a quality filter -- see core/dic_processing.run_dic's
`correlation_threshold` and `max_neighbor_deviation_px`) is simply left
out of the triangulation, rather than kept at some filled-in placeholder
position -- a placeholder was itself a source of the same fold-over
artifact (a barely-moved point sitting amid strongly-moved neighbors).

A plain Delaunay triangulation only ever triangulates the CONVEX HULL of
whatever points it's given, with no notion of a hole in the middle of
that hull. A non-rectangular ROI (polygon/oval) or a manually excluded
zone leaves a genuinely concave gap with no tracked points in it at all
(`MeshGrid.mask`, see core/dic_correlation.py) -- but the points just
outside that gap, on either side of it, are still valid, so a naive
triangulation happily bridges straight across the gap with long,
spurious triangles, and `tricontourf` then shades them as if there were
real data there (a smooth-looking but entirely made-up gradient filling
the excluded notch/hole). `max_edge_px` (see `default_max_edge_px`),
when given, discards exactly those bridging triangles before drawing --
any triangle with an edge longer than that is masked out first -- so an
excluded region renders as a genuine gap instead of a fabricated patch of
color."""
from __future__ import annotations

import numpy as np
from matplotlib.tri import Triangulation
from matplotlib.ticker import MaxNLocator

# Isocontour colour scales are drawn CONTINUOUS: that many filled levels
# between vmin and vmax (so many that no band is visible), instead of ~13-21
# visible bands. Set to a small number (e.g. 13) to get banded isocontours
# back everywhere (every isocontour view/video of the app goes through
# `isocontour_levels`).
ISOCONTOUR_N_LEVELS = 256


def isocontour_levels(vmin: float, vmax: float, n: int | None = None) -> np.ndarray:
    """Levels for a (filled) isocontour plot between `vmin` and `vmax` --
    continuous-looking by default, see ISOCONTOUR_N_LEVELS."""
    return np.linspace(vmin, vmax, int(n or ISOCONTOUR_N_LEVELS))


def isocontour_colorbar(fig, mappable, **kwargs):
    """`fig.colorbar` with round tick values (a continuous contourf has no
    meaningful 'level' ticks: matplotlib would otherwise label odd
    boundaries such as 0.04059)."""
    kwargs.setdefault("ticks", MaxNLocator(nbins=8))
    return fig.colorbar(mappable, **kwargs)

# How many correlation-mesh steps a triangle edge may span before it's
# treated as "bridging a gap" rather than "a normal (possibly somewhat
# deformed) mesh edge" and masked out of the triangulation -- see
# `default_max_edge_px` and the module docstring above. A perfectly
# regular grid's own Delaunay triangulation already has diagonal edges
# ~1.41x the step (a right triangle's hypotenuse across one grid cell),
# and even a fairly heavily strained region can stretch an edge to
# somewhere around 2x its rest length -- 3x leaves clear headroom above
# both before a real (not bridging-a-hole) edge would ever be masked.
DEFAULT_MAX_EDGE_STEP_FACTOR = 3.0


def default_max_edge_px(step: float | None) -> float | None:
    """The `max_edge_px` this module recommends for a correlation mesh
    whose (reference-frame) point spacing is `step` pixels -- see
    `DEFAULT_MAX_EDGE_STEP_FACTOR`. None if `step` itself is None/not
    positive (nothing to size the threshold from)."""
    if not step or step <= 0:
        return None
    return float(step) * DEFAULT_MAX_EDGE_STEP_FACTOR


def _long_edge_mask(triangulation: Triangulation, max_edge_px: float) -> np.ndarray:
    """True for every triangle (row of `triangulation.triangles`) with at
    least one edge longer than `max_edge_px` -- matplotlib's own masking
    convention for `Triangulation.set_mask` (mask=True means "don't draw
    this triangle").

    Known, accepted limitation: right at a REFLEX (concave) corner of an
    excluded region -- e.g. where a polygon ROI's notch meets its
    surrounding valid points -- a small triangle can still have every
    edge under `max_edge_px` while covering a sliver of the excluded
    area, simply because rounding off a sharp concave corner takes a
    much shorter edge than bridging all the way across the hole. This
    mask eliminates wholesale bridging across the interior of a hole,
    not that last, visually minor rounding at its very edge."""
    x, y = triangulation.x, triangulation.y
    tris = triangulation.triangles
    p0 = np.column_stack((x[tris[:, 0]], y[tris[:, 0]]))
    p1 = np.column_stack((x[tris[:, 1]], y[tris[:, 1]]))
    p2 = np.column_stack((x[tris[:, 2]], y[tris[:, 2]]))
    e0 = np.hypot(*(p0 - p1).T)
    e1 = np.hypot(*(p1 - p2).T)
    e2 = np.hypot(*(p2 - p0).T)
    return np.maximum(np.maximum(e0, e1), e2) > max_edge_px


def render_field(fig, ax, background, x: np.ndarray, y: np.ndarray, values: np.ndarray,
                  field_label: str, vmin: float | None = None, vmax: float | None = None,
                  cmap: str = "jet", timestamp_text: str | None = None,
                  text_color: str | None = None, lost_track_color: str = "red",
                  max_edge_px: float | None = None):
    """Draws one frame's background + field onto `ax` (already created,
    e.g. via `fig.add_subplot(111)`) and its colorbar onto `fig`.

    `x`/`y`/`values` are same-shaped arrays (flat or a 2D grid, either
    works); only the triples where all three are finite are plotted --
    everything else is dropped rather than plotted at a placeholder
    position (see the module docstring).

    `vmin`/`vmax` fix the color scale explicitly (pass both for a
    video's frame-to-frame consistency); left as None, the scale
    auto-fits this frame's own data (matching a single interactive view,
    where each frame is looked at on its own).

    `text_color`, left at its default None, leaves every text/tick/spine
    color exactly as the embedding figure's own rcParams already set it
    (the live Results canvas relies on this, to keep matching the app's
    dark theme). Passed explicitly (e.g. "black" for a white-background
    video export), it overrides all of them -- axis labels, ticks,
    spines and the colorbar -- since relying on rcParams there could
    just as easily pick up the app's own (light-on-dark) theme colors
    regardless of the figure's own background.

    `max_edge_px`, when given, discards any triangle of the Delaunay
    triangulation with an edge longer than it before drawing -- see the
    module docstring for why that matters (a non-rectangular ROI's
    excluded notch/hole otherwise gets bridged and shaded as if it were
    real data). `default_max_edge_px(mesh.step)` is the recommended
    value for a `DICResult`'s own correlation mesh. Left at its default
    None, no triangle is ever rejected on edge length (the previous
    behavior)."""
    if background is not None:
        cmap_bg = "gray" if background.ndim == 2 else None
        ax.imshow(background, cmap=cmap_bg, origin="upper", alpha=0.5)
        height, width = background.shape[:2]
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

    x_flat = np.ravel(np.asarray(x, dtype=float))
    y_flat = np.ravel(np.asarray(y, dtype=float))
    v_flat = np.ravel(np.asarray(values, dtype=float))
    finite = np.isfinite(x_flat) & np.isfinite(y_flat) & np.isfinite(v_flat)
    n_valid = int(np.count_nonzero(finite))

    cbar = None
    if n_valid == 0:
        ax.text(0.5, 0.5, "Every point lost track in this frame.", ha="center", va="center",
                color=lost_track_color, fontsize=10, transform=ax.transAxes)
    else:
        xv, yv, vv = x_flat[finite], y_flat[finite], v_flat[finite]
        drawn = False
        if n_valid >= 3:
            try:
                triangulation = Triangulation(xv, yv)
                if max_edge_px is not None and max_edge_px > 0:
                    triangulation.set_mask(_long_edge_mask(triangulation, max_edge_px))
                if vmin is not None and vmax is not None and vmax > vmin:
                    levels = isocontour_levels(vmin, vmax)
                else:  # auto: fit this frame's own data range
                    lo, hi = float(np.min(vv)), float(np.max(vv))
                    if hi <= lo:
                        lo, hi = lo - 0.5, hi + 0.5
                    levels = isocontour_levels(lo, hi)
                cs = ax.tricontourf(triangulation, vv, levels=levels, cmap=cmap, alpha=0.85, extend="both")
                cbar = isocontour_colorbar(fig, cs, ax=ax, label=field_label)
                drawn = True
            except (ValueError, RuntimeError):
                pass  # e.g. every valid point collinear, or every triangle masked out -- falls through to the scatter below
        if not drawn:
            # Too few points to triangulate (or a degenerate point set) --
            # colored markers still show the data.
            sc = ax.scatter(xv, yv, c=vv, cmap=cmap, s=18, vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(sc, ax=ax, label=field_label)

    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    if text_color is not None:
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)
        if cbar is not None:
            cbar.ax.yaxis.label.set_color(text_color)
            cbar.ax.tick_params(colors=text_color)
            cbar.outline.set_edgecolor(text_color)

    if timestamp_text:
        timestamp_artist = ax.text(
            0.02, 0.98, timestamp_text, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=(text_color or "white"),
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor=("white" if text_color is not None else "black"),
                      edgecolor="none", alpha=0.7),
        )
        # Excluded from tight/constrained layout's own bounding-box
        # calculation -- otherwise, since this text's rendered WIDTH
        # grows with the timestamp's digit count (e.g. "t = 1.234 ms"
        # vs. "t = 123.456 ms"), the layout engine (re-run from scratch
        # for every independently-rendered video frame -- see
        # core/dic_video_export.py) would nudge the axes' position a
        # little more with every wider timestamp, producing a slow,
        # unwanted rightward "drift" of the plot relative to the
        # colorbar across the exported video. `in_layout=False` keeps it
        # purely a drawn overlay -- still positioned exactly where set,
        # just never consulted when the layout engine decides how much
        # room everything else needs.
        timestamp_artist.set_in_layout(False)
