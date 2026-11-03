"""
Digital Image Correlation (DIC) setup: region of interest (a rectangle,
an arbitrary polygon, or an oval/ellipse) and correlation mesh (subset
size / step, both in pixels) that a later DIC run will use against the
frames loaded by the Image Correlation page.

This module only builds and validates the *mesh of points* to be
correlated -- not the correlation itself (subset tracking/matching across
frames), which is a separate, later processing step (core/dic_processing.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ROI:
    """A rectangular region of interest, in pixel coordinates of the
    reference frame. (x, y) is the top-left corner."""
    x: int
    y: int
    width: int
    height: int

    def clamped(self, frame_width: int, frame_height: int) -> "ROI":
        """Keeps the ROI fully inside a `frame_width` x `frame_height`
        frame -- used both after a manual drag (which matplotlib already
        constrains to the axes, but a typed-in value might not be) and
        after loading a different-sized frame source than the one the ROI
        was originally drawn on."""
        x = max(0, min(self.x, max(frame_width - 1, 0)))
        y = max(0, min(self.y, max(frame_height - 1, 0)))
        width = max(1, min(self.width, frame_width - x))
        height = max(1, min(self.height, frame_height - y))
        return ROI(x, y, width, height)


@dataclass
class Ellipse:
    """An axis-aligned elliptical (oval) region of interest, in pixel
    coordinates of the reference frame. (cx, cy) is the center, `rx`/`ry`
    the horizontal/vertical semi-axes -- a good fit for a cylindrical
    specimen viewed from the side, where a rectangular ROI would waste
    correlation points on the corners outside the actual specimen."""
    cx: float
    cy: float
    rx: float
    ry: float

    def bounding_roi(self) -> ROI:
        """The smallest rectangular ROI that fully contains this
        ellipse -- used to build the regular point grid before masking
        out whatever falls outside the ellipse itself (see
        `build_mesh_from_ellipse`)."""
        x0 = int(round(self.cx - self.rx))
        y0 = int(round(self.cy - self.ry))
        width = max(1, int(round(2 * self.rx)))
        height = max(1, int(round(2 * self.ry)))
        return ROI(x0, y0, width, height)


@dataclass
class MeshGrid:
    """The regular grid of correlation points produced by `build_mesh()`
    or `build_mesh_from_polygon()`: one point every `step` pixels inside
    the ROI's bounding box, each the center of a `subset_size` x
    `subset_size` pixel correlation subset (a.k.a. facet or zone of
    interest). Points are kept far enough from the bounding box edge that
    their whole subset stays inside it.

    `mask`, when not None, is a boolean array of the same (n_points_y,
    n_points_x) shape as `grid_xy()`: True where a grid point actually
    falls inside the (possibly non-rectangular) ROI, False for a
    bounding-box point outside it -- e.g. a corner cut off by a polygon
    ROI. A plain rectangular ROI has no such gaps, so `build_mesh()`
    leaves this None; core/dic_processing.py treats a masked-out point
    exactly like one whose subset fell outside the frame (excluded from
    tracking, NaN in every result field)."""
    xs: np.ndarray  # 1D, x pixel coordinates of the grid columns
    ys: np.ndarray  # 1D, y pixel coordinates of the grid rows
    subset_size: int
    step: int
    mask: np.ndarray | None = None

    @property
    def n_points_x(self) -> int:
        return int(self.xs.size)

    @property
    def n_points_y(self) -> int:
        return int(self.ys.size)

    @property
    def n_points(self) -> int:
        """Total points in the bounding-box grid, mask or no mask."""
        return self.n_points_x * self.n_points_y

    @property
    def n_valid_points(self) -> int:
        """Points actually inside the ROI -- same as `n_points` for a
        rectangular ROI (no mask), otherwise the count of True entries
        in `mask`."""
        if self.mask is None:
            return self.n_points
        return int(np.count_nonzero(self.mask))

    def grid_xy(self):
        """Full (X, Y) meshgrid arrays of every point, each of shape
        (n_points_y, n_points_x)."""
        return np.meshgrid(self.xs, self.ys)


def build_mesh(roi: ROI, subset_size: int, step: int) -> MeshGrid:
    """Builds the regular correlation mesh over rectangular `roi`.

    Points are centered `subset_size / 2` pixels in from the ROI edges (so
    every subset stays fully inside the ROI), then spaced `step` pixels
    apart. Returns an empty grid (`n_points == 0`) if the ROI is too small
    to fit even one subset -- callers should check for that and let the
    user know, rather than treating it as an error."""
    if subset_size < 1:
        raise ValueError("Subset size must be at least 1 pixel.")
    if step < 1:
        raise ValueError("Step size must be at least 1 pixel.")

    half = subset_size / 2.0
    x0, y0 = roi.x + half, roi.y + half
    x1, y1 = roi.x + roi.width - half, roi.y + roi.height - half
    if x1 < x0 or y1 < y0:
        return MeshGrid(np.array([]), np.array([]), subset_size, step)

    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    return MeshGrid(xs, ys, subset_size, step)


def build_mesh_from_polygon(vertices: list[tuple[float, float]], subset_size: int, step: int) -> MeshGrid:
    """Builds the correlation mesh over an arbitrary polygon ROI (for
    specimen shapes a rectangle doesn't fit well -- dogbone, circular,
    etc.): the regular grid is built over the polygon's bounding box
    (`build_mesh()`), then every point outside the polygon itself is
    masked out (see `MeshGrid.mask`), tested by its center only -- a
    subset near the polygon boundary may still extend slightly outside
    it, which is fine for correlation (the mask is about which points
    count as "inside the ROI" for results/statistics, not a pixel-exact
    clip of each subset's own footprint)."""
    if len(vertices) < 3:
        return MeshGrid(np.array([]), np.array([]), subset_size, step)

    xs_v = [float(v[0]) for v in vertices]
    ys_v = [float(v[1]) for v in vertices]
    bbox = ROI(
        int(round(min(xs_v))), int(round(min(ys_v))),
        max(1, int(round(max(xs_v) - min(xs_v)))), max(1, int(round(max(ys_v) - min(ys_v)))),
    )
    mesh = build_mesh(bbox, subset_size, step)
    if mesh.n_points == 0:
        return mesh

    from matplotlib.path import Path  # local import: only needed for polygon ROIs
    path = Path(vertices)
    xx, yy = mesh.grid_xy()
    points = np.column_stack([xx.ravel(), yy.ravel()])
    mesh.mask = path.contains_points(points).reshape(xx.shape)
    return mesh


def build_mesh_from_ellipse(ellipse: Ellipse, subset_size: int, step: int) -> MeshGrid:
    """Builds the correlation mesh over an elliptical/oval ROI: the
    regular grid is built over the ellipse's bounding box (`build_mesh()`
    on `ellipse.bounding_roi()`), then every point whose CENTER falls
    outside the ellipse is masked out (see `MeshGrid.mask`) -- same
    center-only test, and the same caveat about a subset's own footprint
    possibly extending slightly past the boundary, as
    `build_mesh_from_polygon`."""
    if ellipse.rx <= 0 or ellipse.ry <= 0:
        return MeshGrid(np.array([]), np.array([]), subset_size, step)
    mesh = build_mesh(ellipse.bounding_roi(), subset_size, step)
    if mesh.n_points == 0:
        return mesh
    xx, yy = mesh.grid_xy()
    normalized = ((xx - ellipse.cx) / ellipse.rx) ** 2 + ((yy - ellipse.cy) / ellipse.ry) ** 2
    mesh.mask = normalized <= 1.0
    return mesh
