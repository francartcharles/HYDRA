"""
X-FEM-DIC (Extended FEM / Integrated DIC with fracture enrichment) -- the
Crack & Wave Analysis tool's THIRD, independent crack-tip measurement
method, alongside Stage 3's ridge/wave tip tracking + Stage 4's local-ring
Williams fit (core/crack_k1c.py). Requested by the user (reference: a
paper on DIC identification of fatigue crack growth via the Williams'
series, "Method-II (X-DIC processing)" in the user's supplied excerpt),
scoped down to a still-substantial but bounded implementation -- see "Why
this needs raw images" and "Simplifications" below.

--- Why this needs raw images, not the already-exported DIC CSV ---

core/dic_processing.py tracks each correlation point as a RIGID, purely
TRANSLATING subset (cv2.matchTemplate NCC on a fixed reference template --
see that module's own docstring). A subset whose template straddles the
crack face contains pixels from both sides of a real displacement
discontinuity; no rigid translation can match that template against the
deformed image once the crack has opened even a little, so the standard
correlation there is either wrong or (more often) rejected outright --
exactly why core/crack_k1c.py's ring fit already has to exclude an
`r_min_px` region and `flank_exclusion_deg` around the crack faces. Fitting
a fancier (even enriched) model AFTER the fact to that already-exported
field can only ever re-fit clean, far-from-the-crack points -- it can't
recover the near-tip accuracy the enrichment approach is meant to provide,
because the bias from the discontinuity is already baked into the input
before any post-processing ever sees it.

So this module re-correlates the RAW reference/deformed image pair itself,
with a displacement MODEL that already contains the crack discontinuity and
the near-tip singular field as first-class terms -- an "Integrated DIC"
formulation (Hild & Roux; Réthoré et al.'s eXtended-DIC / "X-DIC" papers),
solved directly against pixel grey levels rather than against a pre-
correlated point cloud.

--- Simplifications relative to a fully general nodal XFEM ---

A textbook XFEM stress-analysis code enriches individual NODES near the
crack with Heaviside (fully-cut elements) and Williams branch functions
(elements touching the tip), each carrying its own local DOFs. Two
deliberate simplifications keep this tractable as a from-scratch, testable
module while staying literature-grounded (this is close to how Réthoré's
X-DIC formulations actually do it -- global rather than per-node singular
enrichment is a documented, published choice, not an ad hoc shortcut):

1. The near-tip singular term is GLOBAL, not nodal: exactly two extra
   scalar unknowns for the whole mesh, K_I and K_II, multiplying the same
   closed-form Williams/Irwin displacement basis functions already used by
   core/crack_k1c.py's ring fit (see `_williams_basis` below -- deliberately
   the *same* formulas/sign/kappa convention, so this method's K_I/K_II are
   directly comparable to the other two methods'), smoothly windowed to
   zero away from the tip by `_tip_window` so it cannot leak into and bias
   the far-field background fit. K_I/K_II therefore come directly out of
   the correlation as two of its fitted parameters -- no separate ring-fit
   post-step needed.
2. The crack-face jump is nodal but UNSHIFTED: `H(x, y)` is evaluated at
   the query point itself (not the standard shifted `H(x)-H(x_i)` form).
   Simpler, at the cost of a slightly less local blend right at a split
   element's boundary with its non-enriched neighbors -- immaterial at the
   sampling density used here.

Everything else -- structured Q4 background mesh, bilinear shape functions,
forward-additive Gauss-Newton against image grey levels with an
analytically exact Jacobian (the displacement model is LINEAR in every one
of its unknowns; only the image warp is nonlinear) -- is standard.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from core.crack_k1c import kolosov_kappa, shear_modulus, positive_value_mean_and_domain_bounds
from core.crack_tip_tracking import path_tangents, smoothed_tip_arc_length_and_positions
from core.dic_kinematics import von_mises_strain
from core.parallel import check_cancelled


class XfemDicError(ValueError):
    """Raised when an X-FEM-DIC fit can't be computed as asked -- message
    is shown to the user as-is, so kept short and specific."""


# ---------------------------------------------------------------------------
# Structured Q4 background mesh
# ---------------------------------------------------------------------------

@dataclass
class Q4Mesh:
    """An axis-aligned grid of bilinear (Q4) quad elements over a
    rectangular region, stored as the physical x/y coordinate of every
    node COLUMN/ROW (`x_edges`/`y_edges`, each strictly increasing) rather
    than a single origin + uniform element size -- this is what lets
    `build_graded_tip_centered_mesh` (below) grade element size smoothly
    from fine near the tip to coarse away from it while every other piece
    of this module (locate, shape functions, the Gauss-Newton fit itself)
    stays exactly as it is for a plain uniform mesh: a UNIFORM mesh is
    simply the special case where `np.diff(x_edges)`/`np.diff(y_edges)`
    are constant, which is exactly how `build_q4_mesh` still builds one.

    Node (i, j) sits at (x_edges[i], y_edges[j]), i in [0, n_elem_x],
    j in [0, n_elem_y], flattened as `node_index = j*(n_elem_x+1) + i`.
    Still axis-aligned means locating which element contains a given
    physical point (and that element's local isoparametric coordinates) is
    closed-form (a `searchsorted` per axis) -- no per-point Newton solve
    needed, unlike a general unstructured FE mesh."""
    x_edges: np.ndarray  # (n_nodes_x,) strictly increasing node-column x-coordinates, px
    y_edges: np.ndarray  # (n_nodes_y,) strictly increasing node-row y-coordinates, px
    element_active: np.ndarray  # (n_elem_y, n_elem_x) bool -- False = fully outside the ROI, skipped

    @property
    def x0(self) -> float:
        return float(self.x_edges[0])

    @property
    def y0(self) -> float:
        return float(self.y_edges[0])

    @property
    def n_elem_x(self) -> int:
        return len(self.x_edges) - 1

    @property
    def n_elem_y(self) -> int:
        return len(self.y_edges) - 1

    @property
    def n_nodes_x(self) -> int:
        return len(self.x_edges)

    @property
    def n_nodes_y(self) -> int:
        return len(self.y_edges)

    @property
    def n_nodes(self) -> int:
        return self.n_nodes_x * self.n_nodes_y

    @property
    def element_size_px(self) -> float:
        """The mesh's SMALLEST element size (finest resolution) -- a back-
        compat convenience for code that just needs "a" characteristic
        length scale (the mesh-origin jitter, the enrichment-classification
        band width), not a claim that every element is this size. On a
        graded mesh (see `build_graded_tip_centered_mesh`) elements are NOT
        uniform; the smallest one is deliberately what those uses care
        about, since it's the near-tip resolution that actually drives
        them. Any code that instead needs a specific element's own size
        (sample density, mesh-preview drawing) should use
        `element_size_xy(ex, ey)`, not this property."""
        return float(min(np.min(np.diff(self.x_edges)), np.min(np.diff(self.y_edges))))

    def element_size_xy(self, ex: int, ey: int) -> tuple[float, float]:
        """(width, height) of element (ex, ey), px -- equal to
        `element_size_px` everywhere on a uniform mesh, but element-
        specific on a graded one."""
        return float(self.x_edges[ex + 1] - self.x_edges[ex]), float(self.y_edges[ey + 1] - self.y_edges[ey])

    def node_index(self, i: int, j: int) -> int:
        return j * self.n_nodes_x + i

    def nodes_xy(self) -> np.ndarray:
        xx, yy = np.meshgrid(self.x_edges, self.y_edges)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def element_node_indices(self, ex: int, ey: int) -> tuple[int, int, int, int]:
        """Node indices (bottom-left, bottom-right, top-right, top-left) --
        counter-clockwise, matching `_bilinear_shape` node ordering."""
        return (
            self.node_index(ex, ey), self.node_index(ex + 1, ey),
            self.node_index(ex + 1, ey + 1), self.node_index(ex, ey + 1),
        )

    def locate(self, x: float, y: float) -> tuple[int, int, float, float] | None:
        """(element_x, element_y, xi, eta) for the element containing
        physical point (x, y), xi/eta in [-1, 1] each -- or None if (x, y)
        falls outside the mesh bounds or inside an inactive element.
        Half-open per element (`[x_edges[ex], x_edges[ex+1])`), same
        convention on a uniform or a graded mesh: a point exactly ON an
        interior edge belongs to the element starting at that edge, and a
        point exactly at the mesh's own outer edge (x_edges[-1]/
        y_edges[-1]) belongs to no element (matches every caller's own
        cell-centered sampling convention, e.g. `_sample_points`)."""
        ex = int(np.searchsorted(self.x_edges, x, side="right")) - 1
        ey = int(np.searchsorted(self.y_edges, y, side="right")) - 1
        if ex < 0 or ey < 0 or ex >= self.n_elem_x or ey >= self.n_elem_y:
            return None
        if not self.element_active[ey, ex]:
            return None
        x_left, x_right = self.x_edges[ex], self.x_edges[ex + 1]
        y_bot, y_top = self.y_edges[ey], self.y_edges[ey + 1]
        xi = 2.0 * (x - x_left) / (x_right - x_left) - 1.0
        eta = 2.0 * (y - y_bot) / (y_top - y_bot) - 1.0
        return ex, ey, xi, eta


def _bilinear_shape(xi: float, eta: float) -> np.ndarray:
    """N_1..N_4 (bottom-left, bottom-right, top-right, top-left) at local
    coordinates (xi, eta) in [-1, 1]."""
    return 0.25 * np.array([
        (1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta), (1 - xi) * (1 + eta),
    ])


def build_q4_mesh(x_min: float, y_min: float, x_max: float, y_max: float,
                   element_size_px: float, active_test=None) -> Q4Mesh:
    """Builds a `Q4Mesh` covering [x_min, x_max] x [y_min, y_max] with
    square elements `element_size_px` on a side (the last row/column is
    stretched slightly to exactly reach x_max/y_max rather than overshoot
    it). `active_test(cx, cy) -> bool`, evaluated at each element's center,
    optionally deactivates elements outside the real region of interest
    (e.g. off the specimen) -- default None activates every element."""
    if element_size_px <= 0:
        raise XfemDicError("The mesh element size must be positive.")
    width, height = x_max - x_min, y_max - y_min
    if width <= 0 or height <= 0:
        raise XfemDicError("The X-FEM-DIC region of interest is empty.")
    n_elem_x = max(1, int(round(width / element_size_px)))
    n_elem_y = max(1, int(round(height / element_size_px)))
    # Recompute the actual element size so n_elem_x/n_elem_y exactly tile
    # [x_min, x_max] x [y_min, y_max] (avoids a thin leftover strip).
    ex_size = width / n_elem_x
    ey_size = height / n_elem_y
    element_size_px = min(ex_size, ey_size)
    n_elem_x = max(1, int(round(width / element_size_px)))
    n_elem_y = max(1, int(round(height / element_size_px)))

    active = np.ones((n_elem_y, n_elem_x), dtype=bool)
    if active_test is not None:
        for ey in range(n_elem_y):
            for ex in range(n_elem_x):
                cx = x_min + (ex + 0.5) * element_size_px
                cy = y_min + (ey + 0.5) * element_size_px
                active[ey, ex] = bool(active_test(cx, cy))
    if not active.any():
        raise XfemDicError("No active X-FEM-DIC element -- check the region of interest.")
    x_edges = x_min + np.arange(n_elem_x + 1) * element_size_px
    y_edges = y_min + np.arange(n_elem_y + 1) * element_size_px
    return Q4Mesh(x_edges=x_edges, y_edges=y_edges, element_active=active)


def build_tip_centered_mesh(tip_xy, roi_half_width_px: float, element_size_px: float) -> Q4Mesh:
    """The exact `Q4Mesh` geometry `compute_xfem_dic_time_series` builds for
    one frame's fit: a square region of half-width `roi_half_width_px`
    centered on `tip_xy`, `element_size_px` elements. Pulled out as its
    own function so the mesh-preview UI (drawing this geometry over the
    reference image *before* running any fit, to let the user judge/tune
    it cheaply) can never drift out of sync with what the production fit
    path actually uses -- both call this same function.

    Includes a small sub-element origin jitter (see comment below) --
    this is production geometry, not idealized/rounded preview geometry,
    so the preview shows the real mesh, jitter included.

    Raises `XfemDicError` (via `build_q4_mesh`) if the settings given
    don't produce a usable mesh, e.g. a non-positive `element_size_px`."""
    # The mesh is built CENTERED on the tip -- but a straight (near-
    # vertical or near-horizontal) crack through a tip that lands exactly
    # on a mesh node line is a degenerate case for
    # `classify_mesh_enrichment`'s corner-sign-change test (neither
    # neighboring element then shows a strict sign change -- see that
    # function's docstring), and centering the mesh on the tip makes that
    # far more likely than chance (the tip sits exactly at the mesh's own
    # center whenever n_elem_x/n_elem_y come out even, which they
    # routinely do, e.g. whenever roi_half_width_px is a round multiple of
    # element_size_px). A small sub-element offset breaks the exact
    # alignment without meaningfully changing the covered region.
    jitter = 0.37 * element_size_px
    return build_q4_mesh(
        tip_xy[0] - roi_half_width_px - jitter, tip_xy[1] - roi_half_width_px - jitter,
        tip_xy[0] + roi_half_width_px - jitter, tip_xy[1] + roi_half_width_px - jitter,
        element_size_px,
    )


def _graded_half_axis(half_width: float, fine_size: float, coarse_size: float,
                       fine_radius: float, growth_ratio: float) -> list[float]:
    """Element sizes marching outward from 0 (the axis center) to
    `half_width`: held at `fine_size` for the first `fine_radius`, then
    grown geometrically (each step `growth_ratio` times the last) up to a
    ceiling of `coarse_size`, held there for the remainder. The very last
    element is trimmed to land exactly on `half_width` -- if that leaves a
    sliver under 40% of the previous element's size (a bad, near-
    degenerate aspect ratio) AND the previous element hasn't already hit
    the `coarse_size` ceiling, it's merged into the previous element
    instead of kept as its own (merging into an already-at-the-ceiling
    element would push the result past `coarse_size`, defeating the point
    of having a ceiling -- a slightly-thin final sliver there is the
    lesser evil). Used independently for `x` and `y` by
    `build_graded_tip_centered_mesh`, mirrored around the tip -- see that
    function's own docstring."""
    sizes: list[float] = []
    pos = 0.0
    size = fine_size
    while pos < half_width - 1e-9:
        remaining = half_width - pos
        if pos >= fine_radius:
            size = min(size * growth_ratio, coarse_size)
        step = min(size, remaining)
        is_final_trimmed_step = step >= remaining - 1e-9
        if (sizes and is_final_trimmed_step and step < 0.4 * sizes[-1]
                and sizes[-1] < coarse_size - 1e-9):
            # A thin trailing sliver -- absorb it into the previous
            # element rather than leaving a near-degenerate one.
            pos -= sizes.pop()
            step = half_width - pos
        sizes.append(step)
        pos += step
    return sizes


def _graded_edges(center: float, half_width: float, fine_size: float, coarse_size: float,
                   fine_radius: float, growth_ratio: float = 1.3) -> np.ndarray:
    """A full (both directions), strictly increasing edge array spanning
    `[center - half_width_actual, center + half_width_actual]`
    (`half_width_actual` is `half_width` rounded up to a whole number of
    graded elements -- see `_graded_half_axis`), symmetric about `center`,
    with a node exactly at `center` itself."""
    half_sizes = _graded_half_axis(half_width, fine_size, coarse_size, fine_radius, growth_ratio)
    cum = np.concatenate([[0.0], np.cumsum(half_sizes)])
    pos_edges = center + cum
    neg_edges = (center - cum[1:])[::-1]
    return np.concatenate([neg_edges, pos_edges])


def build_graded_tip_centered_mesh(tip_xy, roi_half_width_px: float, element_size_px: float,
                                    r_enrich_px: float, coarsen_factor: float = 3.0,
                                    growth_ratio: float = 1.3) -> Q4Mesh:
    """Like `build_tip_centered_mesh`, but GRADED: elements are
    `element_size_px` (the same "Element size" setting, now meaning the
    NEAR-TIP resolution) within `r_enrich_px` of the tip, then grow
    geometrically out to `element_size_px * coarsen_factor` and stay
    there for the rest of the `roi_half_width_px` region.

    Still an ordinary axis-aligned Q4 mesh -- x and y are graded
    INDEPENDENTLY (`_graded_edges`, one call per axis) and combined as a
    plain tensor-product grid, not true polar/radial elements around the
    tip. That keeps every other piece of this module (bilinear shape
    functions, `Q4Mesh.locate`, the Gauss-Newton Jacobian, which only ever
    needs each sample's *local* (xi, eta) inside its own element,
    regardless of that element's physical size) completely unchanged --
    only the geometry (this function) and the handful of places that
    depend on each element's own size (`Q4Mesh.locate`, the
    enrichment-classification band, the sample-density
    "coverage" check in `fit_xfem_dic_frame`) differ from the uniform-mesh
    case.

    This directly targets two things at once: finer elements right where
    the near-tip singular field actually varies fastest (better resolved
    K_I/K_II and less "staircase" near-tip strain than the old uniform
    mesh at the same overall region size), and fewer, larger elements far
    from the tip where a flat/rigid-body-ish background field doesn't
    need fine resolution anyway -- both without paying for a uniformly
    fine mesh over the WHOLE (often much bigger) `roi_half_width_px`
    region, and without inheriting the "blocky" look coarse UNIFORM
    elements gave the isocontour display (see
    `evaluate_xfem_dic_field_grid` -- the display grid still samples at a
    constant `step_px` regardless of the underlying mesh, so a graded
    mesh's coarse far-field elements show up as smoothly-varying, not
    blocky, once reconstructed there).

    Raises `XfemDicError` (via `build_q4_mesh`'s validation, reached
    through the same active/None-mesh checks) if the settings given don't
    produce a usable mesh."""
    if element_size_px <= 0:
        raise XfemDicError("The mesh element size must be positive.")
    if r_enrich_px <= 0:
        raise XfemDicError("The tip enrichment radius must be positive.")
    if roi_half_width_px <= 0:
        raise XfemDicError("The X-FEM-DIC region of interest is empty.")
    coarse_size = max(element_size_px * max(coarsen_factor, 1.0), element_size_px)
    # Same sub-element jitter rationale as build_tip_centered_mesh (avoids
    # a straight crack landing exactly on a mesh node line) -- sized off
    # the FINE element size, the one that actually matters for a crack
    # passing near the tip, where the fine region always is.
    jitter = 0.37 * element_size_px
    x_edges = _graded_edges(tip_xy[0] - jitter, roi_half_width_px, element_size_px, coarse_size,
                             r_enrich_px, growth_ratio)
    y_edges = _graded_edges(tip_xy[1] - jitter, roi_half_width_px, element_size_px, coarse_size,
                             r_enrich_px, growth_ratio)
    active = np.ones((len(y_edges) - 1, len(x_edges) - 1), dtype=bool)
    return Q4Mesh(x_edges=x_edges, y_edges=y_edges, element_active=active)


def recommend_mesh_parameters(pixel_size_mm: float | None, max_crack_distance_px: float,
                               image_width_px: float | None = None,
                               image_height_px: float | None = None) -> dict:
    """A starting-point (element_size_px, r_enrich_px, roi_half_width_px,
    sample_step_px) pre-fill for the settings above, from the two numbers
    that actually drive a sensible choice: how far the crack has
    propagated (its max tracked arc length -- the enrichment radius/mesh
    only need to work over that range, not an arbitrary guess) and, if
    known, the reference image's own dimensions (so the suggested region
    half-width can't exceed what the image actually contains).

    The ratios used here (r_enrich_px/element_size_px = 2, roi_half_width/
    r_enrich = 4) are exactly the ones this module's own validation test
    (test_xfem_dic.py) and this feature's shipped UI defaults
    (element_size_px=20, r_enrich_px=40, roi_half_width_px=160) were found
    to work well with -- not derived from first principles. `r_enrich_px`
    itself is clipped to a fraction of `max_crack_distance_px` (a tip
    enrichment window that's a sizeable fraction of the whole tracked
    crack length would smear the "near-tip" field's compact-support
    assumption; too small a fraction leaves too few sample points inside
    it) and to a generous absolute range, since a very short or very long
    tracked crack should still get a usable suggestion.

    `sample_step_px` -- `fit_xfem_dic_frame`'s OWN grey-level sampling
    grid (see that function's docstring), a different axis from the mesh
    above -- is suggested as `element_size_px / SAMPLE_STEP_RATIO`
    (clipped to [1.0, 10.0] px). `element_size_px` is already the
    SMALLEST element edge actually in play in either mesh mode
    (`build_q4_mesh` uses it uniformly; `build_graded_tip_centered_mesh`
    uses it as the fine near-tip zone and only coarsens away from the
    tip), so dividing it by `SAMPLE_STEP_RATIO` keeps several samples
    across even the smallest element's own edge -- too few, and
    `fit_xfem_dic_frame`'s `node_reliable` mask starts flagging that
    element's corner nodes as under-constrained. The 1.0 px floor
    reflects that a real speckle pattern's own correlation length is
    rarely usefully finer than about a pixel -- sampling denser than that
    mostly adds correlated, non-independent residuals rather than new
    information, while cost grows as `1/sample_step_px**2` per iteration
    (and, if "Estimate measurement uncertainty" is on, can understate the
    formal K_I/K_II stderr, whose degrees-of-freedom count implicitly
    assumes independent residuals -- see this module's own docstring on
    `estimate_uncertainty`). Like every ratio here, not derived from
    first principles -- a reasonable starting point to then check by
    hand (does K_I/K_II actually keep changing as you lower it further,
    or has it already plateaued?).

    This is only ever a pre-fill of the existing, still-editable spin
    boxes -- never applied silently -- so a somewhat rough heuristic is
    an acceptable trade-off for saving the user a first guess.

    `pixel_size_mm`, when given (may be None/0 if not yet set), is used
    only to additionally report each suggested value in mm -- see the
    "_mm" keys below -- since a physical size is what actually lets the
    user judge whether the suggestion is reasonable for their specimen;
    it does not change any of the pixel-space numbers themselves (the
    sampling-density trade-offs above are inherently a pixel-count
    question, not a physical-length one)."""
    if not np.isfinite(max_crack_distance_px) or max_crack_distance_px <= 0:
        max_crack_distance_px = 200.0  # no trace to measure yet -- fall back to a generic guess
    r_enrich_px = float(np.clip(0.12 * max_crack_distance_px, 20.0, 100.0))
    roi_half_width_px = 4.0 * r_enrich_px

    # Cap the region half-width against the reference image itself, if
    # known, so the suggestion can never exceed what the image actually
    # contains. Rather than clamping roi_half_width_px on its own (which
    # could then be smaller than even a couple of elements for a tiny
    # image, once the r_enrich_px/element_size_px ratios below are
    # applied), shrink r_enrich_px BY THE SAME FACTOR when the cap bites,
    # keeping the whole suggestion internally proportional.
    image_cap_px = None
    if image_width_px:
        image_cap_px = 0.45 * float(image_width_px)
    if image_height_px:
        height_cap = 0.45 * float(image_height_px)
        image_cap_px = height_cap if image_cap_px is None else min(image_cap_px, height_cap)
    if image_cap_px is not None and 0.0 < image_cap_px < roi_half_width_px:
        scale = image_cap_px / roi_half_width_px
        roi_half_width_px = image_cap_px
        r_enrich_px = max(r_enrich_px * scale, 6.0)

    element_size_px = float(np.clip(r_enrich_px / 2.0, 8.0, 30.0))
    # Never let the element-size floor above push the mesh below 3
    # elements across the (possibly image-capped) region half-width --
    # shrink the element size instead, since roi_half_width_px must not
    # grow back past the image cap just set above.
    element_size_px = min(element_size_px, roi_half_width_px / 3.0)

    # See this function's own docstring for the reasoning -- several
    # samples across even the smallest (near-tip) element edge, without
    # sampling far below a real speckle pattern's own correlation length.
    SAMPLE_STEP_RATIO = 8.0
    sample_step_px = float(np.clip(element_size_px / SAMPLE_STEP_RATIO, 1.0, 10.0))

    element_size_px = round(element_size_px, 1)
    r_enrich_px = round(r_enrich_px, 1)
    roi_half_width_px = round(roi_half_width_px, 1)
    sample_step_px = round(sample_step_px, 1)
    result = {
        "element_size_px": element_size_px,
        "r_enrich_px": r_enrich_px,
        "roi_half_width_px": roi_half_width_px,
        "sample_step_px": sample_step_px,
    }
    if pixel_size_mm:
        result.update({
            "element_size_mm": round(element_size_px * pixel_size_mm, 4),
            "r_enrich_mm": round(r_enrich_px * pixel_size_mm, 4),
            "roi_half_width_mm": round(roi_half_width_px * pixel_size_mm, 4),
            "sample_step_mm": round(sample_step_px * pixel_size_mm, 4),
        })
    return result


def recommend_tip_search_radius_px(arc_length_s: np.ndarray,
                                    pixel_size_mm: float | None = None) -> dict:
    """A starting-point `tip_search_radius_px` for `auto_detect_tip`
    (either mode), from the ticked trace's OWN tracked arc-length history
    -- the one number that actually drives whether this search is wide
    enough to do its job at all: the total OBSERVED spread of `tip_result.
    arc_length_s` across the whole trace (its max minus its min).

    Why this number, specifically: `compute_xfem_dic_time_series`
    deliberately anchors every frame's search independently to THAT
    frame's own nominal (ridge/wave-tracked) position, never chaining off
    the previous frame's self-found result (see that function's own
    docstring -- this is a fix for a real drift bug, not incidental). That
    means a SYSTEMATIC offset between the nominal track and where the
    correlation data actually supports the crack-face model -- e.g. a
    "wave"-extrapolated position that runs ahead of where the crack has
    genuinely separated, or a ridge-threshold effect that lags behind it
    -- has to be re-bridged by the search radius on EVERY single frame,
    not just once. If that per-frame offset is close to or larger than
    the search radius, the observable symptom is exactly the one this
    feature was built to fix: the auto-detected track visibly plateaus
    (pinned by the "crack cannot shrink" floor/ceiling -- see
    `_search_best_tip_s`) while the nominal track keeps climbing, because
    every candidate actually reachable within the (too-small) window
    keeps losing to the same already-accepted position.

    That true per-frame offset isn't knowable in advance without already
    running the search (chicken-and-egg) -- but a real, PERSISTENT offset
    large enough to visibly stall the auto-detected track over many frames
    is, empirically, a sizeable fraction of the crack's own total tracked
    growth, not a small one -- so this suggests a radius scaled off that
    same total spread, clipped to a sane range: too small a floor would
    give back essentially the prior hardcoded default (40 px) even for a
    barely-progressed trace; too large a ceiling would make the search
    (one full Gauss-Newton solve per candidate -- see `_search_best_tip_s`'s
    own cost note) prohibitively slow regardless of what a longer trace's
    spread suggests.

    This is only ever a pre-fill of the existing, still-editable
    `tip_search_radius_px` spin box -- never applied silently, and never a
    substitute for actually checking the resulting arc-length-vs-time plot
    (a still-flat auto-detected track after raising this, especially
    alongside the "heaviside_jump" mode finding almost no positions at
    all, points at a genuine disagreement between the tracking method and
    the image data -- e.g. the nominal method extrapolating past where the
    crack has actually, physically separated -- not merely an
    under-sized search window)."""
    finite = np.asarray(arc_length_s, dtype=float)
    finite = finite[np.isfinite(finite)]
    span_px = float(finite.max() - finite.min()) if finite.size >= 2 else 0.0
    if span_px <= 0.0:
        radius_px = 40.0  # no usable spread to measure yet -- the app's prior hardcoded default
    else:
        radius_px = float(np.clip(0.4 * span_px, 40.0, 300.0))
    radius_px = round(radius_px, 1)
    result = {"search_radius_px": radius_px, "observed_span_px": round(span_px, 1)}
    if pixel_size_mm:
        result.update({
            "search_radius_mm": round(radius_px * pixel_size_mm, 4),
            "observed_span_mm": round(span_px * pixel_size_mm, 4),
        })
    return result


# ---------------------------------------------------------------------------
# Crack-path geometry: signed distance / arc-length projection onto the
# already-traced polyline (Stage 3's output -- this module never re-detects
# the crack path or tip itself, per the user's chosen "fixed crack path"
# scope: reuse the ridge/wave tip track already computed on this same page).
# ---------------------------------------------------------------------------

def _project_to_polyline(points_xy: np.ndarray, path_xy: np.ndarray, path_s: np.ndarray):
    """For each row of `points_xy` (N, 2), the signed perpendicular
    distance to the nearest segment of `path_xy` (a polyline, already
    ordered by increasing `path_s`), the arc length `path_s` at that
    nearest projection, and the segment's unit tangent. Signed distance:
    positive on the side the segment's tangent-rotated-+90-degrees normal
    points to (same convention as core/crack_k1c.py's `_sample_annulus`).
    Brute-force over all segments -- the traced path is at most a few
    hundred points, called once per fit, not a hot loop."""
    n = points_xy.shape[0]
    best_dist = np.full(n, np.inf)
    best_signed = np.zeros(n)
    best_s = np.zeros(n)
    best_tx = np.zeros(n)
    best_ty = np.zeros(n)
    for k in range(len(path_xy) - 1):
        a, b = path_xy[k], path_xy[k + 1]
        seg = b - a
        seg_len2 = float(seg[0] ** 2 + seg[1] ** 2)
        if seg_len2 < 1e-12:
            continue
        seg_len = np.sqrt(seg_len2)
        tx, ty = seg[0] / seg_len, seg[1] / seg_len
        nx, ny = -ty, tx
        rel = points_xy - a
        t = np.clip((rel[:, 0] * tx + rel[:, 1] * ty) / seg_len, 0.0, 1.0)
        proj = a + np.outer(t, seg)
        diff = points_xy - proj
        dist = np.hypot(diff[:, 0], diff[:, 1])
        signed = diff[:, 0] * nx + diff[:, 1] * ny
        s_here = path_s[k] + t * (path_s[k + 1] - path_s[k])
        better = dist < best_dist
        best_dist = np.where(better, dist, best_dist)
        best_signed = np.where(better, signed, best_signed)
        best_s = np.where(better, s_here, best_s)
        best_tx = np.where(better, tx, best_tx)
        best_ty = np.where(better, ty, best_ty)
    return best_signed, best_s, best_tx, best_ty


def classify_mesh_enrichment(mesh: Q4Mesh, path_xy: np.ndarray, path_s: np.ndarray,
                              tip_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(node_enriched (n_nodes,) bool, element_split (n_elem_y, n_elem_x) bool).
    `node_enriched` is exactly what `fit_xfem_dic_frame` needs (which
    nodes carry a Heaviside DOF); `element_split` additionally flags,
    per ELEMENT, whether the ALREADY-OPENED part of the crack (path arc
    length <= tip_s) actually passes through it -- not needed by the fit
    itself (only enriched nodes matter there), but is exactly what the
    mesh-preview visualization needs to shade which elements are cut, so
    it's computed here (one pass) rather than re-derived separately.

    An element is "split" when its 4 corners' signed distances to the
    crack path change sign (the path crosses it), the nearest such
    crossing is actually close by (within 1.5 elements), and it isn't
    clearly past the tip.

    Deliberately projects every node onto the FULL `path_xy`/`path_s` (not
    a copy truncated to `path_s <= tip_s`) and only gates by `tip_s`
    afterward, via each corner's own projected arc length: truncating the
    polyline first would clamp the projection of any node near/just past
    the tip to the truncated polyline's own endpoint, which is a boundary
    artifact of the truncation, not the crack's real geometry --
    `tip_result.path_positions` (this function's real caller) already
    extends past the current tip by construction (the whole visible
    skeleton is traced once, independent of any one frame's arc-length
    progress), so the full path gives the genuine nearest-segment
    projection even right at the tip."""
    if len(path_xy) < 2:
        return (np.zeros(mesh.n_nodes, dtype=bool),
                np.zeros((mesh.n_elem_y, mesh.n_elem_x), dtype=bool))

    nodes_xy = mesh.nodes_xy()
    signed, s_proj, _tx, _ty = _project_to_polyline(nodes_xy, path_xy, path_s)
    node_enriched = np.zeros(mesh.n_nodes, dtype=bool)
    element_split = np.zeros((mesh.n_elem_y, mesh.n_elem_x), dtype=bool)
    for ey in range(mesh.n_elem_y):
        for ex in range(mesh.n_elem_x):
            if not mesh.element_active[ey, ex]:
                continue
            # Local (not global) band: on a graded mesh (see
            # `build_graded_tip_centered_mesh`) a coarse far-field
            # element is much bigger than the mesh's overall
            # `element_size_px` (its SMALLEST element, by that
            # property's own definition) -- a band sized off the global
            # smallest would be far too tight out there and could miss a
            # genuine crossing through a big coarse element.
            ex_size, ey_size = mesh.element_size_xy(ex, ey)
            band = 1.5 * max(ex_size, ey_size)
            idx = mesh.element_node_indices(ex, ey)
            corner_signed = signed[list(idx)]
            corner_dist = np.abs(corner_signed)
            corner_s = s_proj[list(idx)]
            if corner_dist.min() > band:
                continue
            if corner_s.max() > tip_s + band:
                continue  # element sits (at least mostly) ahead of the tip -- not yet opened
            if corner_signed.min() < 0.0 < corner_signed.max():
                element_split[ey, ex] = True
                for i in idx:
                    node_enriched[i] = True
    return node_enriched, element_split


def _classify_split_elements(mesh: Q4Mesh, path_xy: np.ndarray, path_s: np.ndarray,
                              tip_s: float) -> np.ndarray:
    """Thin wrapper around `classify_mesh_enrichment` returning only the
    per-node array -- kept as the name `fit_xfem_dic_frame` (and this
    module's existing tests) already call; see that function's docstring
    for the full explanation."""
    node_enriched, _element_split = classify_mesh_enrichment(mesh, path_xy, path_s, tip_s)
    return node_enriched


def _heaviside_sign(points_xy: np.ndarray, path_xy: np.ndarray, path_s: np.ndarray) -> np.ndarray:
    """Sign of each point's perpendicular offset from the (full) crack
    path -- see `_classify_split_elements`'s docstring on why this always
    projects onto the untruncated path. Only meaningful at sample points
    that fall in an element with at least one Heaviside-enriched node
    (elsewhere its factor is multiplied by a zero shape-function weight
    anyway, see `fit_xfem_dic_frame`'s `heav_mask`)."""
    signed, _s, _tx, _ty = _project_to_polyline(points_xy, path_xy, path_s)
    return np.sign(signed)


# ---------------------------------------------------------------------------
# Near-tip Williams/Irwin basis (identical formulas/convention to
# core/crack_k1c.py's `fit_k_single_frame`, so K_I/K_II from this method are
# directly comparable to the ring-fit method's) plus a smooth compact-
# support window so the global singular term never leaks into (and biases)
# the far-field background fit.
# ---------------------------------------------------------------------------

def _williams_basis(r: np.ndarray, theta: np.ndarray, kappa: float, mu_pa: float):
    """Returns (fI_u, fII_u, fI_v, fII_v) such that
    u = K_I*fI_u + K_II*fII_u, v = K_I*fI_v + K_II*fII_v -- same convention
    as core/crack_k1c.py's `fit_k_single_frame` (r in METERS there; here in
    whatever length unit `r` is given in, caller's responsibility)."""
    sqrt_term = np.sqrt(np.maximum(r, 0.0) / (2.0 * np.pi))
    c2, s2 = np.cos(theta / 2.0), np.sin(theta / 2.0)
    fI_u = sqrt_term * c2 * (kappa - 1.0 + 2.0 * s2 ** 2) / (2.0 * mu_pa)
    fII_u = sqrt_term * s2 * (kappa + 1.0 + 2.0 * c2 ** 2) / (2.0 * mu_pa)
    fI_v = sqrt_term * s2 * (kappa + 1.0 - 2.0 * c2 ** 2) / (2.0 * mu_pa)
    fII_v = -sqrt_term * c2 * (kappa - 1.0 - 2.0 * s2 ** 2) / (2.0 * mu_pa)
    return fI_u, fII_u, fI_v, fII_v


def _tip_window(r_px: np.ndarray, r_min_px: float, r_max_px: float) -> np.ndarray:
    """Smooth radial band-pass weight, confining the global singular
    (K_I/K_II) term to an annulus around the tip -- the same ROLE
    core/crack_k1c.py's ring-fit plays with its own r_min_px/r_max_px
    (excluding near-tip DIC noise/decorrelation, and departure from the
    leading-order 1/sqrt(r) field far from the tip), but implemented as a
    smooth WEIGHT on the singular basis functions rather than a hard
    include/exclude on which sample points are used: `fit_xfem_dic_frame`
    solves the background, Heaviside and singular terms all at once in a
    single Gauss-Newton correlation, so an abrupt on/off boundary would
    inject a discontinuity into the residual every time the tip moves a
    fraction of a pixel between frames, destabilizing convergence across a
    time series -- the whole reason the ORIGINAL (single-radius, outer-
    taper-only) version of this function was already smooth, before
    r_min_px existed.

    Shape, as a function of r:
        0.0                              for r <= r_min_px
        cosine ramp 0.0 -> 1.0            for r_min_px  < r <  r_in_end
        1.0                              for r_in_end <= r <= r_out_start
        cosine ramp 1.0 -> 0.0            for r_out_start < r < r_max_px
        0.0                              for r >= r_max_px
    where `r_out_start = 0.6 * r_max_px` (unchanged from the original
    single-radius taper) and `r_in_end` is 40% of the way from r_min_px to
    r_out_start (clamped so it can never cross r_out_start itself, for a
    degenerately tight r_min_px/r_max_px pair).

    `r_min_px=0.0` collapses the inner ramp to nothing (`r_in_end` becomes
    0.0 too), reproducing this function's exact original behavior, from
    before r_min_px/r_max_px existed, with `r_max_px` playing the role the
    old sole parameter `r_enrich_px` did -- see `fit_xfem_dic_frame`'s own
    docstring on why that's this function's default."""
    r_min_px = max(float(r_min_px), 0.0)
    r_max_px = float(r_max_px)
    r_out_start = 0.6 * r_max_px
    if r_min_px <= 0.0:
        r_in_end = 0.0
    else:
        r_in_end = min(r_min_px + 0.4 * max(r_max_px - r_min_px, 0.0), r_out_start)
        r_in_end = max(r_in_end, r_min_px)

    w = np.zeros_like(r_px, dtype=float)
    plateau = (r_px >= r_in_end) & (r_px <= r_out_start)
    w[plateau] = 1.0
    if r_min_px > 0.0 and r_in_end > r_min_px:
        inner = (r_px > r_min_px) & (r_px < r_in_end)
        w[inner] = 0.5 * (1.0 - np.cos(np.pi * (r_px[inner] - r_min_px) / (r_in_end - r_min_px)))
    outer = (r_px > r_out_start) & (r_px < r_max_px)
    w[outer] = 0.5 * (1.0 + np.cos(np.pi * (r_px[outer] - r_out_start) / (r_max_px - r_out_start)))
    return w


# ---------------------------------------------------------------------------
# Single-frame Integrated-DIC / X-FEM-DIC fit
# ---------------------------------------------------------------------------

@dataclass
class XfemDicFitResult:
    mesh: Q4Mesh
    heaviside_nodes: np.ndarray  # (n_nodes,) bool
    node_disp_x: np.ndarray      # (n_nodes,) background field, px
    node_disp_y: np.ndarray
    heaviside_disp_x: np.ndarray  # (n_nodes,) jump amplitude at enriched nodes, 0 elsewhere, px
    heaviside_disp_y: np.ndarray
    K_I_pa_sqrt_m: float | None
    K_II_pa_sqrt_m: float | None
    converged: bool
    n_iterations: int
    n_samples: int
    residual_rms: float | None       # final normalized grey-level residual RMS (0..~1 for 8-bit images)
    residual_rms_initial: float | None
    error: str | None = None         # set (all-else None-ish) if the fit failed outright
    stalled: bool = False            # True when the loop stopped early because the grey-level
                                      # residual RMS plateaued (see `fit_xfem_dic_frame`'s
                                      # `stall_patience`/`stall_rtol`), NOT because `step_px` itself
                                      # dropped below `convergence_tol_px` -- kept strictly separate
                                      # from `converged` (never both True at once) so callers that
                                      # already treat `converged` as "trust this one over an
                                      # unconverged candidate" (e.g. `_search_best_tip_s`) are not
                                      # silently handed a plateaued fit as if it were tightly
                                      # converged. Always False when `converged` is True or on a
                                      # failure placeholder.
    node_reliable: np.ndarray | None = None  # (n_nodes,) bool -- see `fit_xfem_dic_frame`'s "node
                                              # coverage" comment; None on a failure placeholder
                                              # (nothing to mask -- the fit itself is unusable there)
    K_I_stderr_pa_sqrt_m: float | None = None   # 1-sigma formal Gauss-Newton fit uncertainty --
    K_II_stderr_pa_sqrt_m: float | None = None  # only populated when `estimate_uncertainty=True`
                                                 # was passed to `fit_xfem_dic_frame` (see its own
                                                 # docstring) -- `None` otherwise, same "None means
                                                 # not estimated, not zero uncertainty" convention as
                                                 # core.crack_tip_tracking's own uncertainty fields
    singular_term_in_field: bool = True  # False when the fit was run WITHOUT the global K_I/K_II
                                          # singular term (`fit_singular_term=False`): the displayed
                                          # field is then background + Heaviside only, and K_I/K_II
                                          # come from `refit_k_from_fitted_field` (ring fit on that field)
    k_source: str = "joint"               # "joint" (K_I/K_II are unknowns of the grey-level fit) or
                                          # "field_ring:<method>" (see `refit_k_from_fitted_field`)


def _sample_points(mesh: Q4Mesh, step_px: float) -> np.ndarray:
    """A regular grid of sample points (Nx2, px) at `step_px` spacing,
    covering every ACTIVE element (points in inactive elements are dropped
    afterward by the caller, via `mesh.locate` returning None for them).
    Uses the mesh's own outer edges (`x_edges[-1]`/`y_edges[-1]`), not
    `n_elem_x * element_size_px` -- the latter silently assumes a uniform
    mesh, wrong on a graded one (see `build_graded_tip_centered_mesh`)."""
    x_max = float(mesh.x_edges[-1])
    y_max = float(mesh.y_edges[-1])
    xs = np.arange(mesh.x0 + step_px / 2.0, x_max, step_px)
    ys = np.arange(mesh.y0 + step_px / 2.0, y_max, step_px)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def fit_xfem_dic_frame(ref_image: np.ndarray, def_image: np.ndarray, mesh: Q4Mesh,
                        path_xy: np.ndarray, path_s: np.ndarray, tip_xy, tip_s: float, tangent_xy,
                        kappa: float, mu_pa: float, pixel_size_mm: float, r_enrich_px: float,
                        sample_step_px: float = 2.0, initial_uv_fn=None,
                        max_iterations: int = 15, damping: float = 1e-2,
                        convergence_tol_px: float = 2e-3,
                        stall_patience: int = 5, stall_rtol: float = 1e-3,
                        r_min_px: float = 0.0, r_max_px: float | None = None,
                        estimate_uncertainty: bool = False,
                        fit_singular_term: bool = True) -> XfemDicFitResult:
    """Fits the background+Heaviside+singular displacement model described
    in this module's docstring to one reference/deformed image pair, by
    forward-additive Gauss-Newton directly against grey levels (no
    intermediate per-point correlation step).

    `path_xy`/`path_s` are the already-traced crack path (px, reference-
    frame pixel coordinates -- same convention as
    core.crack_tip_tracking.TipTrackResult.path_positions/path_arc_length),
    `tip_xy`/`tip_s` the current frame's tip position/arc length on that
    path, `tangent_xy` the local crack direction there (e.g. from
    `core.crack_tip_tracking.path_tangents`) -- all exactly the geometry
    Stage 3 already computed; this function only re-correlates the images
    against a model built around that FIXED geometry (see module docstring
    -- "fixed crack path" is this feature's deliberately chosen scope).

    `initial_uv_fn(x, y) -> (u, v)`, optional, warm-starts the background
    field's nodal DOFs from an already-measured displacement field (e.g.
    the existing subset-DIC CSV's own u/v, interpolated) -- substantially
    improves Gauss-Newton robustness/speed versus starting from zero
    motion, since only the crack-discontinuity/near-tip terms then have to
    be found from scratch. Left at None, every DOF starts at zero.

    `convergence_tol_px` is a STEP-SIZE criterion only (the largest
    per-iteration change among the background/Heaviside pixel DOFs) -- it
    says nothing about whether the fit is still genuinely improving. At a
    small crack-tip displacement the grey-level signal driving each
    Gauss-Newton correction becomes comparable to image/correlation noise
    and to the deformed-image spline's own numerical-derivative noise, so
    with `damping` fixed (not adaptive) the per-iteration step can chatter
    around a noise floor ABOVE `convergence_tol_px` indefinitely -- more
    `max_iterations` does not help, since the loop isn't slowly
    converging, it has nothing left to converge from and never will reach
    that specific step-size threshold on this kind of frame at all.

    `stall_patience`/`stall_rtol` (5 iterations / 0.1% by default) are a
    SECOND, independent stopping test, on the grey-level residual RMS
    itself rather than the step size: every iteration's RMS is compared to
    the best RMS seen so far on this frame, and an iteration only resets
    the "iterations since last real improvement" counter when it beats
    that best by more than `stall_rtol` (relative) -- a marginal or
    negative change (typical of chatter) leaves the counter running. Once
    `stall_patience` consecutive iterations pass without an improvement
    that large, the loop stops -- reported as `stalled=True`, kept
    strictly distinct from `converged=True` (a real `step_px` convergence)
    so a caller that already prefers a converged candidate over an
    unconverged one (`_search_best_tip_s`) is never silently handed a
    plateaued fit as if it were one. Regardless of which of the two
    criteria stops the loop, or whether `max_iterations` is exhausted with
    neither triggered, the DOF state actually returned is the single best-
    RMS one seen across every iteration (including the very last one) --
    not necessarily the loop's last iterate -- so a frame that chatters
    right up to the iteration budget still reports its best available fit
    rather than whatever state the chatter happened to land on last. This
    changes nothing for a frame that already converges cleanly by
    `step_px`: its best-RMS state is essentially always its final one
    already, since Gauss-Newton residual RMS decreases near-monotonically
    in that well-conditioned regime.

    `r_min_px`/`r_max_px` confine the singular (K_I/K_II) term's smooth
    radial weight (`_tip_window`) the same way core/crack_k1c.py's ring-fit
    uses its own r_min_px/r_max_px to define the annulus it samples the
    DIC displacement field in -- but here it is a smooth WEIGHT applied to
    the singular basis functions inside a fit that also solves the
    background/Heaviside terms simultaneously, not a hard include/exclude
    on which sample points are used at all (see `_tip_window`'s own
    docstring for why). `r_min_px=0.0` (default) applies no near-tip
    exclusion. `r_max_px=None` (default) reuses `r_enrich_px` as
    the window's outer radius -- pass an explicit
    value to decouple the fit's own near-tip confinement from
    `r_enrich_px`, which independently also drives the mesh's own
    enrichment/grading geometry (see `build_graded_tip_centered_mesh`) and
    so cannot always be freely tuned for the fit alone.

    `estimate_uncertainty=False` (default): `K_I_stderr_pa_sqrt_m`/
    `K_II_stderr_pa_sqrt_m` are left `None`. `True` computes them from the
    same asymptotic Gauss-Newton covariance every nonlinear least-squares
    fit admits at its converged point: with the (grey-level) residual
    Jacobian `J` re-assembled once more at the FINAL solution (not reused
    from the last GN iteration, whose `J` was evaluated at the state
    BEFORE that iteration's own update), `sigma_gray^2 = SSE / (n_samples
    - n_dof)` the grey-level residual variance, and `cov = sigma_gray^2 *
    pinv(J^T J)` the parameter covariance -- exactly the linear-fit
    covariance core/crack_k1c.py's own `fit_k_single_frame` uses, applied
    here at the nonlinear fit's local linearization instead of an exactly
    linear model (the standard, textbook way to report a nonlinear least-
    squares fit's own parameter uncertainty). This is opt-in (unlike
    `fit_k_single_frame`'s always-on stderr) because assembling `J` again
    and inverting the resulting `n_dof x n_dof` normal matrix costs real
    time on a fine mesh (`n_dof` scales with the number of mesh nodes),
    on top of a fit that is already, per this module's own docstring,
    "far heavier" than the ring fit's single linear solve.

    As in core/crack_k1c.py's own stderr, `J^T J` is Jacobi-
    (column-)scaled before inverting: K_I/K_II's own Jacobian columns
    (image-gradient-weighted Williams basis values) and the background/
    Heaviside pixel-displacement columns differ by many orders of
    magnitude, and inverting the RAW normal matrix silently truncates the
    tiny-but-real K_I/K_II variance to near-nothing instead of reporting
    it honestly -- this reuses the
    IDENTICAL scaling this function's own damped GN step already applies
    (`col_scale`, above), just also applied to the post-hoc covariance.
    `K_I_stderr_pa_sqrt_m`/`K_II_stderr_pa_sqrt_m` stay `None` if the
    scaled normal matrix's relevant diagonal entries come out non-finite
    or negative (a degenerate/rank-deficient fit) rather than reporting a
    misleading number.

    Returns an `XfemDicFitResult` with `error` set (every numeric field
    left None/0) rather than raising, so a caller fitting a whole time
    series (see `compute_xfem_dic_time_series`) can skip one bad frame
    without losing the rest -- mirrors core/crack_k1c.py's per-frame
    `KFitResult`."""
    if not pixel_size_mm or pixel_size_mm <= 0:
        raise XfemDicError("A pixel size (mm/px) is required to compute K_I/K_II in physical units.")
    if r_enrich_px <= 0:
        raise XfemDicError("The tip enrichment radius must be positive.")
    if r_min_px < 0:
        raise XfemDicError("The tip window's inner exclusion radius (r_min) cannot be negative.")
    effective_r_max_px = float(r_enrich_px if r_max_px is None else r_max_px)
    if effective_r_max_px <= 0:
        raise XfemDicError("The tip window's outer radius (r_max) must be positive.")
    if r_min_px >= effective_r_max_px:
        raise XfemDicError(
            "The tip window's inner radius (r_min) must be smaller than its outer radius (r_max)."
        )

    ref = np.asarray(ref_image, dtype=np.float64)
    deff = np.asarray(def_image, dtype=np.float64)
    if ref.ndim == 3:
        ref = ref.mean(axis=-1)  # grayscale luminance average -- RectBivariateSpline needs a 2D field
    if deff.ndim == 3:
        deff = deff.mean(axis=-1)
    intensity_scale = float(np.std(ref)) or 1.0

    heaviside_nodes = _classify_split_elements(mesh, path_xy, path_s, tip_s)
    n_nodes = mesh.n_nodes
    heav_index = -np.ones(n_nodes, dtype=int)
    heav_index[heaviside_nodes] = np.arange(int(np.count_nonzero(heaviside_nodes)))
    n_heav = int(np.count_nonzero(heaviside_nodes))
    # DOF layout: [a_x (n_nodes), a_y (n_nodes), b_x (n_heav), b_y (n_heav), K_I, K_II]
    n_dof = 2 * n_nodes + 2 * n_heav + 2
    K_I_SLOT, K_II_SLOT = n_dof - 2, n_dof - 1

    pts = _sample_points(mesh, sample_step_px)
    located = [mesh.locate(x, y) for x, y in pts]
    keep = np.array([loc is not None for loc in located])
    # Also require the reference-image sample itself to land inside the
    # reference frame (integer pixel indexing below -- no interpolation
    # needed there, only the warped deformed-frame side needs one). Uses
    # the ROUNDED coordinate for this bounds check -- checking the raw
    # (pre-rounding) coordinate against `< w_ref`/`< h_ref` let a point
    # like x=249.6 pass (249.6 < 250) and then round UP to 250, an
    # out-of-bounds index once actually used below.
    h_ref, w_ref = ref.shape[:2]
    rounded_x = np.round(pts[:, 0]).astype(int)
    rounded_y = np.round(pts[:, 1]).astype(int)
    in_ref = (rounded_x >= 0) & (rounded_x < w_ref) & (rounded_y >= 0) & (rounded_y < h_ref)
    keep = keep & in_ref
    rounded_x, rounded_y = rounded_x[keep], rounded_y[keep]
    pts = pts[keep]
    located = [loc for loc, k in zip(located, keep) if k]
    n_samples = len(pts)
    if n_samples < n_dof:
        return XfemDicFitResult(
            mesh=mesh, heaviside_nodes=heaviside_nodes,
            node_disp_x=np.zeros(n_nodes), node_disp_y=np.zeros(n_nodes),
            heaviside_disp_x=np.zeros(n_nodes), heaviside_disp_y=np.zeros(n_nodes),
            K_I_pa_sqrt_m=None, K_II_pa_sqrt_m=None, converged=False, n_iterations=0,
            n_samples=n_samples, residual_rms=None, residual_rms_initial=None,
            error=f"Not enough valid sample points ({n_samples}) for {n_dof} unknowns -- "
                  f"use a larger element size or a smaller enrichment radius.",
        )

    # Reference grey level sampled with the SAME cubic interpolation as the
    # deformed image, at the exact (non-integer) sample point. Sampling the
    # reference at the nearest integer pixel while the model is evaluated at
    # the unrounded point put a random +-0.5 px position error on every
    # sample (an irreducible residual floor, ~0.17 of the image std on a
    # typical speckle) that the Gauss-Newton fit turned into noisy,
    # blocky nodal displacements and, after differentiation, unreadable
    # strain maps.
    ref_spline = RectBivariateSpline(
        np.arange(h_ref, dtype=np.float64), np.arange(w_ref, dtype=np.float64), ref, kx=3, ky=3
    )
    ref_gray = ref_spline.ev(pts[:, 1], pts[:, 0])

    tx, ty = tangent_xy
    nx_hat, ny_hat = -ty, tx
    dxp, dyp = pts[:, 0] - tip_xy[0], pts[:, 1] - tip_xy[1]
    r_px = np.hypot(dxp, dyp)
    theta = np.arctan2(dxp * nx_hat + dyp * ny_hat, dxp * tx + dyp * ty)
    m_per_px = pixel_size_mm * 1e-3
    # `_williams_basis` returns displacement-per-(Pa*sqrt(m)) in METERS
    # (same convention as core/crack_k1c.py's ring fit) -- but everywhere
    # else in this module u/v are in PIXELS (they're added directly to
    # `pts`, which are pixel coordinates, to get the warped sample
    # position). Dividing by `m_per_px` here converts the basis functions
    # to pixels-per-(Pa*sqrt(m)) so that a_vec[K_I_SLOT]/a_vec[K_II_SLOT]
    # come out of the solve already in the physical Pa*sqrt(m) units this
    # function promises, matching core/crack_k1c.py's K_I_pa_sqrt_m/
    # K_II_pa_sqrt_m directly (comparable across all three of this tool's
    # methods) rather than in some arbitrary pixel-mixed scale.
    fI_u, fII_u, fI_v, fII_v = _williams_basis(r_px * m_per_px, theta, kappa, mu_pa)
    fI_u, fII_u, fI_v, fII_v = fI_u / m_per_px, fII_u / m_per_px, fI_v / m_per_px, fII_v / m_per_px
    omega = _tip_window(r_px, r_min_px, effective_r_max_px)
    fI_u, fII_u, fI_v, fII_v = fI_u * omega, fII_u * omega, fI_v * omega, fII_v * omega
    # `_williams_basis` returns the singular term in the LOCAL (tangent,
    # normal) frame (fI_u/fII_u = tangential component, fI_v/fII_v = normal
    # component -- see that function's docstring), but `u`/`v` everywhere
    # else in this function are GLOBAL (image x, y) pixel displacements
    # (they're added directly to `pts`, pixel coordinates, a few lines
    # below). The basis is rotated into that same global frame before use,
    # same derivation as core/crack_k1c.py's `fit_k_single_frame`:
    # global_x = local_tangential*tx - local_normal*ty,
    # global_y = local_tangential*ty + local_normal*tx. This rotation is
    # required for any crack-path orientation other than tangent_xy exactly
    # (1, 0); skipping it would mix true K_I and K_II together in the
    # fitted result.
    fI_x, fI_y = fI_u * tx - fI_v * ty, fI_u * ty + fI_v * tx
    fII_x, fII_y = fII_u * tx - fII_v * ty, fII_u * ty + fII_v * tx
    if not fit_singular_term:
        # K_I/K_II are NOT unknowns of this fit (their columns are zero, so they stay 0.0 and the
        # caller replaces them via `refit_k_from_fitted_field`). Reason: a global scalar times a smooth
        # sqrt(r) field is almost entirely reproducible by the free nodal Q4 DOFs (only ~30 % of the
        # K column is independent of them on a 14 px mesh, ~7 % on a 7 px mesh), and its signal is
        # ~0.01 px at 20 px from the tip -- so K estimated jointly comes out with arbitrary sign/size.
        fI_x = fI_y = fII_x = fII_y = np.zeros_like(fI_x)

    if len(path_xy) >= 2:
        h_sign = _heaviside_sign(pts, path_xy, path_s)
    else:
        h_sign = np.zeros(n_samples)

    # Shape-function weights + node indices per sample, precomputed once
    # (geometry never changes across Gauss-Newton iterations).
    elem_nodes = np.zeros((n_samples, 4), dtype=int)
    elem_N = np.zeros((n_samples, 4))
    for i, (ex, ey, xi, eta) in enumerate(located):
        elem_nodes[i] = mesh.element_node_indices(ex, ey)
        elem_N[i] = _bilinear_shape(xi, eta)

    # How many sample points a FULLY, cleanly covered node would get, purely
    # from mesh geometry -- independent of how many samples actually
    # survived the `in_ref` filter above, which is the point: a node whose
    # element is mostly (not entirely) outside the reference image still
    # gets a few in-bounds samples from its uncut sliver, and comparing
    # against what actually reached `pts`/`elem_nodes` (already filtered)
    # would call that "fully supported" when really only a fraction of its
    # element was ever sampled. Summed per-ELEMENT (not from a single
    # global element size) so this stays correct on a graded mesh (see
    # `build_graded_tip_centered_mesh`), where a coarse far-field element
    # naturally expects many more samples than a fine near-tip one.
    node_expected_count = np.zeros(n_nodes)
    for ey in range(mesh.n_elem_y):
        for ex in range(mesh.n_elem_x):
            if not mesh.element_active[ey, ex]:
                continue
            ex_size, ey_size = mesh.element_size_xy(ex, ey)
            expected_here = max(1.0, (ex_size / max(sample_step_px, 1e-6))
                                 * (ey_size / max(sample_step_px, 1e-6)))
            for idx in mesh.element_node_indices(ex, ey):
                node_expected_count[idx] += expected_here

    a = np.zeros(n_dof)
    if initial_uv_fn is not None:
        nodes_xy = mesh.nodes_xy()
        for i in range(n_nodes):
            try:
                u0, v0 = initial_uv_fn(nodes_xy[i, 0], nodes_xy[i, 1])
            except Exception:  # noqa: BLE001 - a warm start is a nice-to-have, never fatal
                continue
            if np.isfinite(u0) and np.isfinite(v0):
                a[i], a[n_nodes + i] = float(u0), float(v0)

    def displacement_at_samples(a_vec):
        ax, ay = a_vec[:n_nodes], a_vec[n_nodes:2 * n_nodes]
        bx = np.zeros(n_nodes)
        by = np.zeros(n_nodes)
        if n_heav:
            bx[heaviside_nodes] = a_vec[2 * n_nodes:2 * n_nodes + n_heav]
            by[heaviside_nodes] = a_vec[2 * n_nodes + n_heav:2 * n_nodes + 2 * n_heav]
        u = np.sum(elem_N * ax[elem_nodes], axis=1) + h_sign * np.sum(elem_N * bx[elem_nodes], axis=1)
        v = np.sum(elem_N * ay[elem_nodes], axis=1) + h_sign * np.sum(elem_N * by[elem_nodes], axis=1)
        u += a_vec[K_I_SLOT] * fI_x + a_vec[K_II_SLOT] * fII_x
        v += a_vec[K_I_SLOT] * fI_y + a_vec[K_II_SLOT] * fII_y
        return u, v

    # Analytic residual Jacobian w.r.t. every DOF, EXCEPT for the chain-rule
    # factor through the deformed image itself (dI_def/dx, dI_def/dy),
    # which is re-evaluated at the current warped position every iteration
    # (forward-additive Gauss-Newton) -- everything below this point in
    # `_jacobian_columns` is geometry, constant across iterations.
    row = np.repeat(np.arange(n_samples), 4)
    col_ax = elem_nodes.ravel()
    col_ay = n_nodes + elem_nodes.ravel()
    w_std = elem_N.ravel()
    if n_heav:
        heav_local = heav_index[elem_nodes]  # (n_samples, 4), -1 where not enriched
        heav_mask = heav_local >= 0
        row_h = row.reshape(n_samples, 4)[heav_mask]
        col_bx = 2 * n_nodes + heav_local[heav_mask]
        col_by = 2 * n_nodes + n_heav + heav_local[heav_mask]
        w_heav = (elem_N * h_sign[:, None])[heav_mask]

    y_grid = np.arange(deff.shape[0], dtype=np.float64)
    x_grid = np.arange(deff.shape[1], dtype=np.float64)
    def_spline = RectBivariateSpline(y_grid, x_grid, deff, kx=3, ky=3)

    residual_rms_initial = None
    converged = False
    stalled = False
    n_iter_done = 0
    # Best-RMS-observed tracking (fix 2): `a` at the moment ANY stopping
    # condition fires -- clean step-size convergence, a residual plateau,
    # or simply running out of `max_iterations` -- is not necessarily the
    # best fit this frame ever reached, especially once it's chattering
    # near a noise floor rather than descending monotonically. `best_a`/
    # `best_rms` are updated below every time a strictly lower RMS is
    # observed, at ANY iteration (including implicitly the state after the
    # very last applied update, compared once more right after the loop
    # below) -- so the DOF state finally reported is always the best one
    # actually seen, never an arbitrary "wherever the loop happened to
    # stop" snapshot.
    best_rms = None
    best_a = None
    best_n_iter = 0
    # Residual-plateau stall detection (fix 1): counts consecutive
    # iterations that failed to beat `best_rms` by more than `stall_rtol`
    # (relative) -- reset to 0 only on a genuinely meaningful improvement,
    # left running through both "no improvement" and "a marginal new best
    # too small to count" iterations, since both are equally symptomatic
    # of chatter rather than real convergence.
    no_improve_count = 0
    for iteration in range(max_iterations):
        u, v = displacement_at_samples(a)
        wx, wy = pts[:, 0] + u, pts[:, 1] + v
        in_bounds = (wx >= 0) & (wx <= x_grid[-1]) & (wy >= 0) & (wy <= y_grid[-1])
        if np.count_nonzero(in_bounds) < n_dof:
            return XfemDicFitResult(
                mesh=mesh, heaviside_nodes=heaviside_nodes,
                node_disp_x=np.zeros(n_nodes), node_disp_y=np.zeros(n_nodes),
                heaviside_disp_x=np.zeros(n_nodes), heaviside_disp_y=np.zeros(n_nodes),
                K_I_pa_sqrt_m=None, K_II_pa_sqrt_m=None, converged=False, n_iterations=iteration,
                n_samples=n_samples, residual_rms=None, residual_rms_initial=residual_rms_initial,
                error="The current displacement estimate warped too many sample points outside "
                      "the deformed frame -- the image sequence or crack geometry may not match "
                      "this X-FEM-DIC region.",
            )
        def_gray = def_spline.ev(wy, wx)
        residual = ref_gray - def_gray
        residual[~in_bounds] = 0.0
        rms = float(np.sqrt(np.mean((residual / intensity_scale) ** 2)))
        if residual_rms_initial is None:
            residual_rms_initial = rms

        if best_rms is None:
            # First iteration: nothing to compare against yet -- establish
            # the baseline without counting it toward the stall patience
            # in either direction.
            best_rms, best_a, best_n_iter = rms, a.copy(), iteration
        elif rms < best_rms:
            significant = (best_rms - rms) > stall_rtol * best_rms
            best_rms, best_a, best_n_iter = rms, a.copy(), iteration
            no_improve_count = 0 if significant else no_improve_count + 1
        else:
            no_improve_count += 1
        if no_improve_count >= stall_patience:
            stalled = True
            break

        gy = def_spline.ev(wy, wx, dx=1, dy=0)  # d(def)/dy at the warped point
        gx = def_spline.ev(wy, wx, dx=0, dy=1)  # d(def)/dx at the warped point
        gx[~in_bounds] = 0.0
        gy[~in_bounds] = 0.0

        data = []
        jrow = []
        jcol = []
        data.append(-gx[row] * w_std); jrow.append(row); jcol.append(col_ax)
        data.append(-gy[row] * w_std); jrow.append(row); jcol.append(col_ay)
        if n_heav:
            data.append(-gx[row_h] * w_heav); jrow.append(row_h); jcol.append(col_bx)
            data.append(-gy[row_h] * w_heav); jrow.append(row_h); jcol.append(col_by)
        all_rows = np.arange(n_samples)
        data.append(-(gx * fI_x + gy * fI_y)); jrow.append(all_rows); jcol.append(np.full(n_samples, K_I_SLOT))
        data.append(-(gx * fII_x + gy * fII_y)); jrow.append(all_rows); jcol.append(np.full(n_samples, K_II_SLOT))

        J = csr_matrix((
            np.concatenate(data), (np.concatenate(jrow), np.concatenate(jcol)),
        ), shape=(n_samples, n_dof))

        # Jacobi/column scaling before the damped solve: the background
        # and Heaviside DOFs are O(1) in PIXELS, but K_I/K_II are O(1e6)
        # in Pa*sqrt(m), so a single fixed `damp` value cannot regularize
        # both groups equally -- it would either barely constrain the
        # pixel-scale DOFs or crush the K_I/K_II columns to ~0.
        # Scaling each column by its own L2 norm (a standard Jacobi
        # preconditioner for damped least squares) equalizes how strongly
        # `damp` constrains every DOF regardless of its physical units.
        col_scale = np.sqrt(np.asarray(J.power(2).sum(axis=0))).ravel()
        col_scale[col_scale < 1e-12] = 1.0
        J_scaled = J.multiply(1.0 / col_scale).tocsr()

        # Gauss-Newton step: J*delta ~= -residual(a) minimizes
        # ||residual(a) + J*delta||^2 -- i.e. `delta` IS the update to add
        # to `a` (not subtract; J already carries the sign of
        # d(residual)/da, see the "-gx*w_std" etc. columns built above).
        delta_scaled = lsqr(J_scaled, -residual, damp=damping, atol=1e-10, btol=1e-10)[0]
        delta = delta_scaled / col_scale
        a = a + delta
        n_iter_done = iteration + 1

        step_px = float(np.max(np.abs(delta[:2 * n_nodes + 2 * n_heav]))) if (n_nodes or n_heav) else 0.0
        if step_px < convergence_tol_px:
            converged = True
            break

    u, v = displacement_at_samples(a)
    wx, wy = pts[:, 0] + u, pts[:, 1] + v
    in_bounds = (wx >= 0) & (wx <= x_grid[-1]) & (wy >= 0) & (wy <= y_grid[-1])
    def_gray = def_spline.ev(wy[in_bounds], wx[in_bounds])
    final_rms = float(np.sqrt(np.mean(((ref_gray[in_bounds] - def_gray) / intensity_scale) ** 2)))

    # The state after the very last applied update was never itself
    # compared against `best_rms` inside the loop (only the PRE-update
    # state at the top of each iteration was) -- do that comparison once
    # more here, so `best_a` genuinely covers every state the fit ever
    # reached, including its very last one.
    if best_rms is None or final_rms < best_rms:
        best_rms, best_a, best_n_iter = final_rms, a.copy(), n_iter_done
    # Report the best-RMS state actually observed, not necessarily the
    # loop's own last iterate (see `stall_patience`/`stall_rtol` in this
    # function's docstring) -- a no-op for a frame that already converges
    # cleanly by `step_px`, whose best-RMS state is essentially always its
    # final one already.
    a = best_a
    final_rms = best_rms
    u, v = displacement_at_samples(a)
    wx, wy = pts[:, 0] + u, pts[:, 1] + v
    in_bounds = (wx >= 0) & (wx <= x_grid[-1]) & (wy >= 0) & (wy <= y_grid[-1])

    K_I_stderr = K_II_stderr = None
    if estimate_uncertainty:
        # Re-assembles the SAME Jacobian the GN loop itself builds every
        # iteration (see the docstring above for why: the loop's own last
        # `J` was evaluated one update BEHIND the final `a`), at every
        # sample point (not just the in-bounds ones -- mirrors the loop,
        # which zeroes an out-of-bounds point's own gx/gy/residual rather
        # than dropping its row, so it contributes nothing to either the
        # residual sum of squares or the Jacobian below).
        def_gray_full = def_spline.ev(wy, wx)
        residual_full = ref_gray - def_gray_full
        residual_full[~in_bounds] = 0.0
        gy_f = def_spline.ev(wy, wx, dx=1, dy=0)
        gx_f = def_spline.ev(wy, wx, dx=0, dy=1)
        gx_f[~in_bounds] = 0.0
        gy_f[~in_bounds] = 0.0

        data_f, jrow_f, jcol_f = [], [], []
        data_f.append(-gx_f[row] * w_std); jrow_f.append(row); jcol_f.append(col_ax)
        data_f.append(-gy_f[row] * w_std); jrow_f.append(row); jcol_f.append(col_ay)
        if n_heav:
            data_f.append(-gx_f[row_h] * w_heav); jrow_f.append(row_h); jcol_f.append(col_bx)
            data_f.append(-gy_f[row_h] * w_heav); jrow_f.append(row_h); jcol_f.append(col_by)
        all_rows_f = np.arange(n_samples)
        data_f.append(-(gx_f * fI_x + gy_f * fI_y)); jrow_f.append(all_rows_f)
        jcol_f.append(np.full(n_samples, K_I_SLOT))
        data_f.append(-(gx_f * fII_x + gy_f * fII_y)); jrow_f.append(all_rows_f)
        jcol_f.append(np.full(n_samples, K_II_SLOT))
        J_final = csr_matrix((
            np.concatenate(data_f), (np.concatenate(jrow_f), np.concatenate(jcol_f)),
        ), shape=(n_samples, n_dof))

        n_effective = int(np.count_nonzero(in_bounds))
        dof_unc = n_effective - n_dof
        if dof_unc > 0:
            sse_gray = float(np.sum(residual_full[in_bounds] ** 2))
            sigma_gray2 = sse_gray / dof_unc
            # Jacobi column scaling before inverting J^T J -- IDENTICAL
            # reasoning/technique as `col_scale` above (and core/
            # crack_k1c.py's own `fit_k_single_frame` stderr): K_I/K_II's
            # columns and the background/Heaviside columns differ by many
            # orders of magnitude, and `pinv`'s default rcond cutoff is
            # relative to the LARGEST singular value, so an unscaled
            # normal matrix silently truncates K_I/K_II's own (tiny-but-
            # real) variance to near-nothing rather than reporting it.
            col_scale_unc = np.sqrt(np.asarray(J_final.power(2).sum(axis=0))).ravel()
            col_scale_unc[col_scale_unc < 1e-12] = 1.0
            J_scaled_unc = J_final.multiply(1.0 / col_scale_unc).tocsr()
            normal_scaled = (J_scaled_unc.T @ J_scaled_unc).toarray()
            cov_scaled = sigma_gray2 * np.linalg.pinv(normal_scaled)
            var_K_I = cov_scaled[K_I_SLOT, K_I_SLOT] / (col_scale_unc[K_I_SLOT] ** 2)
            var_K_II = cov_scaled[K_II_SLOT, K_II_SLOT] / (col_scale_unc[K_II_SLOT] ** 2)
            if np.isfinite(var_K_I) and var_K_I >= 0:
                K_I_stderr = float(np.sqrt(var_K_I))
            if np.isfinite(var_K_II) and var_K_II >= 0:
                K_II_stderr = float(np.sqrt(var_K_II))

    node_disp_x, node_disp_y = a[:n_nodes], a[n_nodes:2 * n_nodes]
    heav_disp_x, heav_disp_y = np.zeros(n_nodes), np.zeros(n_nodes)
    if n_heav:
        heav_disp_x[heaviside_nodes] = a[2 * n_nodes:2 * n_nodes + n_heav]
        heav_disp_y[heaviside_nodes] = a[2 * n_nodes + n_heav:2 * n_nodes + 2 * n_heav]

    # A node is "reliable" when most of the samples a FULLY, cleanly
    # covered node of this geometry would get (node_expected_count, above)
    # actually landed in-bounds in the deformed image at this final
    # iteration -- i.e. it was genuinely constrained by real grey-level
    # information, not left to drift only under the damped-least-squares
    # regularization pulling it toward its warm start/zero (or, just as
    # bad, only PARTLY constrained by a thin sliver of its element that
    # happened to survive the `in_ref` filter, which the same math flags
    # too since that sliver is a small fraction of node_expected_count).
    # A whole swath of nodes can end up
    # unreliable together whenever the mesh's square ROI extends past the
    # actual reference-image crop, or a large rigid-body motion warps a
    # chunk of the ROI outside the deformed frame -- exactly the case
    # that, left unflagged, would show up in the "Field isocontours"
    # card as a solid block of saturated color at the ROI's edge/corner
    # (a frozen, near-zero background field there meeting the genuinely
    # fitted interior head-on, differentiated into a huge synthetic
    # strain "wall"). `evaluate_xfem_dic_displacement_field` NaNs out any
    # query point touching an unreliable node instead, so the isocontour
    # plots simply stop (honestly) rather than showing that false field.
    node_support_count = np.bincount(elem_nodes[in_bounds].ravel(), minlength=n_nodes)
    node_reliable = (node_expected_count > 0) & (node_support_count >= 0.5 * node_expected_count)

    return XfemDicFitResult(
        mesh=mesh, heaviside_nodes=heaviside_nodes,
        node_disp_x=node_disp_x, node_disp_y=node_disp_y,
        heaviside_disp_x=heav_disp_x, heaviside_disp_y=heav_disp_y,
        K_I_pa_sqrt_m=float(a[K_I_SLOT]), K_II_pa_sqrt_m=float(a[K_II_SLOT]),
        converged=converged, n_iterations=n_iter_done, n_samples=n_samples,
        residual_rms=final_rms, residual_rms_initial=residual_rms_initial, error=None,
        stalled=stalled, node_reliable=node_reliable,
        K_I_stderr_pa_sqrt_m=K_I_stderr, K_II_stderr_pa_sqrt_m=K_II_stderr,
        singular_term_in_field=bool(fit_singular_term),
    )


# ---------------------------------------------------------------------------
# Evaluating the FITTED field afterward -- lets the UI draw isocontours of
# the X-FEM-DIC displacement/strain field (for a visual sanity check
# against the already-exported subset-DIC field, side by side) without
# re-running the fit: `fit_xfem_dic_frame` already stored every DOF the
# model needs (`XfemDicFitResult.node_disp_x/y`, `heaviside_disp_x/y`,
# `K_I_pa_sqrt_m`/`K_II_pa_sqrt_m`); the same background+Heaviside+
# singular formula `fit_xfem_dic_frame`'s own `displacement_at_samples`
# closure evaluates at its sample points during the solve is simply
# re-evaluated here at whatever points the caller asks for. Duplicated
# rather than sharing that closure (which is private to one call of
# `fit_xfem_dic_frame` and closes over that call's own arrays) -- see
# this tool's "self-sufficient stages" convention (this module's own
# docstring, core/crack_k1c.py's docstring, etc.) for why that's the
# deliberate choice here rather than a refactor to share it.
# ---------------------------------------------------------------------------

def evaluate_xfem_dic_displacement_field(fit: XfemDicFitResult, path_xy: np.ndarray, path_s: np.ndarray,
                                          tip_xy, tangent_xy, kappa: float, mu_pa: float,
                                          pixel_size_mm: float, r_enrich_px: float, x, y,
                                          r_min_px: float = 0.0, r_max_px: float | None = None):
    """Reconstructs the FITTED X-FEM-DIC displacement field (u, v), in
    PIXELS, at arbitrary query point(s) `x`/`y` (same shape, any shape --
    typically a `np.meshgrid` pair for a grid, but any array works): the
    background bilinear field, plus the Heaviside crack-face jump, plus
    the Williams near-tip singular term, using `fit`'s own already-solved
    DOFs -- the identical model (and identical formulas/unit conversions)
    `fit_xfem_dic_frame` fits, just evaluated afterward instead of during
    the Gauss-Newton solve.

    `path_xy`/`path_s`/`tip_xy`/`tangent_xy`/`kappa`/`mu_pa`/
    `pixel_size_mm`/`r_enrich_px`/`r_min_px`/`r_max_px` are NOT stored on
    `XfemDicFitResult` itself (only the fitted DOFs are, to keep that
    dataclass small) -- pass the exact same values used for the
    `fit_xfem_dic_frame` call that produced `fit` (mirrors that function's
    own parameters of the same names, including their same defaulting:
    `r_min_px=0.0`, `r_max_px=None` reusing `r_enrich_px`) -- otherwise the
    reconstructed field's singular term would be windowed differently from
    how it was actually fitted, silently mismatching what's shown here
    against what the Gauss-Newton solve itself saw.

    Points outside `fit.mesh` (`Q4Mesh.locate` returns None there) get
    NaN u/v -- the background/Heaviside terms are only defined inside the
    mesh, and a NaN is the honest answer there, not a guess. So does a
    point whose containing element has any corner node `fit.node_reliable`
    flags as poorly-constrained (see that field's own comment on
    `XfemDicFitResult`, set by `fit_xfem_dic_frame`) -- a real point near
    the fitted mesh's edge that the solve itself had little/no grey-level
    information about, not a rendering artifact to paper over with a
    guess."""
    mesh = fit.mesh
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    shape = x.shape
    xf, yf = x.ravel(), y.ravel()
    u = np.full(xf.shape, np.nan)
    v = np.full(xf.shape, np.nan)

    located = [mesh.locate(px, py) for px, py in zip(xf, yf)]
    inside = np.array([loc is not None for loc in located])
    if not inside.any():
        return u.reshape(shape), v.reshape(shape)

    idx_inside = np.flatnonzero(inside)
    elem_nodes = np.zeros((idx_inside.size, 4), dtype=int)
    elem_N = np.zeros((idx_inside.size, 4))
    for k, i in enumerate(idx_inside):
        ex, ey, xi, eta = located[i]
        elem_nodes[k] = mesh.element_node_indices(ex, ey)
        elem_N[k] = _bilinear_shape(xi, eta)

    if fit.node_reliable is not None:
        reliable = fit.node_reliable[elem_nodes].all(axis=1)
        if not reliable.any():
            return u.reshape(shape), v.reshape(shape)
        idx_inside = idx_inside[reliable]
        elem_nodes = elem_nodes[reliable]
        elem_N = elem_N[reliable]

    u_bg = np.sum(elem_N * fit.node_disp_x[elem_nodes], axis=1)
    v_bg = np.sum(elem_N * fit.node_disp_y[elem_nodes], axis=1)

    if len(path_xy) >= 2 and fit.heaviside_nodes.any():
        pts_inside = np.column_stack([xf[idx_inside], yf[idx_inside]])
        h_sign = _heaviside_sign(pts_inside, path_xy, path_s)
        u_h = h_sign * np.sum(elem_N * fit.heaviside_disp_x[elem_nodes], axis=1)
        v_h = h_sign * np.sum(elem_N * fit.heaviside_disp_y[elem_nodes], axis=1)
    else:
        u_h = np.zeros(idx_inside.size)
        v_h = np.zeros(idx_inside.size)

    if (fit.singular_term_in_field and fit.K_I_pa_sqrt_m is not None
            and fit.K_II_pa_sqrt_m is not None):
        tx, ty = tangent_xy
        nx_hat, ny_hat = -ty, tx
        dxp = xf[idx_inside] - tip_xy[0]
        dyp = yf[idx_inside] - tip_xy[1]
        r_px = np.hypot(dxp, dyp)
        theta = np.arctan2(dxp * nx_hat + dyp * ny_hat, dxp * tx + dyp * ty)
        m_per_px = pixel_size_mm * 1e-3
        # Same meters -> pixels conversion as fit_xfem_dic_frame -- see
        # that function's own comment on why it's needed.
        fI_u, fII_u, fI_v, fII_v = _williams_basis(r_px * m_per_px, theta, kappa, mu_pa)
        fI_u, fII_u, fI_v, fII_v = fI_u / m_per_px, fII_u / m_per_px, fI_v / m_per_px, fII_v / m_per_px
        effective_r_max_px = float(r_enrich_px if r_max_px is None else r_max_px)
        omega = _tip_window(r_px, r_min_px, effective_r_max_px)
        fI_u, fII_u, fI_v, fII_v = fI_u * omega, fII_u * omega, fI_v * omega, fII_v * omega
        # Rotate local (tangent, normal) -> global (x, y) pixels, same as
        # `fit_xfem_dic_frame` itself -- this grid must stay consistent with
        # the fit it's evaluating, or the visualized/exported field would
        # silently disagree with the K_I/K_II it came from.
        fI_x, fI_y = fI_u * tx - fI_v * ty, fI_u * ty + fI_v * tx
        fII_x, fII_y = fII_u * tx - fII_v * ty, fII_u * ty + fII_v * tx
        u_sing = fit.K_I_pa_sqrt_m * fI_x + fit.K_II_pa_sqrt_m * fII_x
        v_sing = fit.K_I_pa_sqrt_m * fI_y + fit.K_II_pa_sqrt_m * fII_y
    else:
        u_sing = np.zeros(idx_inside.size)
        v_sing = np.zeros(idx_inside.size)

    u[idx_inside] = u_bg + u_h + u_sing
    v[idx_inside] = v_bg + v_h + v_sing
    return u.reshape(shape), v.reshape(shape)


K_FIELD_METHODS = ("ring", "ring_t", "ring_affine")


def refit_k_from_fitted_field(fit: XfemDicFitResult, path_xy: np.ndarray, path_s: np.ndarray,
                              tip_xy, tangent_xy, kappa: float, mu_pa: float, pixel_size_mm: float,
                              r_enrich_px: float, r_min_px: float, r_max_px: float,
                              method: str = "ring_affine", tip_offset_px: float = 0.0,
                              flank_exclusion_deg: float = 15.0, grid_step_px: float = 2.0):
    """Returns a copy of `fit` whose K_I/K_II (and stderr) come from the SAME ring fit
    (core.crack_k1c.fit_k_single_frame, window `r_min_px..r_max_px`, optional `tip_offset_px` along the
    tangent, `method` in K_FIELD_METHODS) that the direct method applies to the subset-DIC field --
    here applied to the fitted X-FEM-DIC displacement field (background + Heaviside, which resolves the
    crack-face jump the subset DIC smears). `fit` must come from `fit_xfem_dic_frame(...,
    fit_singular_term=False)`: when the K_I/K_II are unknowns of the grey-level fit itself, they are
    nearly collinear with the free nodal field and come out with arbitrary sign and size.

    On a synthetic test (true K_I 1.8 MPa*sqrt(m), uniform background strain, 2 grey-level noise),
    joint fitting gave -0.4..+3 (biased even without noise) while this gives 1.7..1.9 with 'ring_affine'."""
    from dataclasses import replace
    from core.crack_k1c import fit_k_single_frame
    if method not in K_FIELD_METHODS:
        raise XfemDicError(f"Unknown K method {method!r}; expected one of {K_FIELD_METHODS}.")
    if fit.error is not None:
        return fit
    tip = np.asarray(tip_xy, dtype=float)
    tangent = np.asarray(tangent_xy, dtype=float)
    centre = tip + float(tip_offset_px or 0.0) * tangent
    half = float(r_max_px) + abs(float(tip_offset_px or 0.0)) + 2.0 * grid_step_px
    gx = np.arange(tip[0] - half, tip[0] + half + 1e-9, grid_step_px)
    gy = np.arange(tip[1] - half, tip[1] + half + 1e-9, grid_step_px)
    X, Y = np.meshgrid(gx, gy)
    u, v = evaluate_xfem_dic_displacement_field(
        replace(fit, K_I_pa_sqrt_m=None, K_II_pa_sqrt_m=None, singular_term_in_field=False),
        path_xy, path_s, tip, tangent, kappa, mu_pa, pixel_size_mm, r_enrich_px, X, Y)
    try:
        res = fit_k_single_frame(
            u, v, gx, gy, centre, tangent, kappa, mu_pa, pixel_size_mm, float(r_min_px), float(r_max_px),
            flank_exclusion_deg, include_t_stress=(method == "ring_t"), include_affine=(method == "ring_affine"))
    except Exception as exc:  # noqa: BLE001 - K1cError/ValueError: report as a failed K, not a crash
        return replace(fit, K_I_pa_sqrt_m=None, K_II_pa_sqrt_m=None, error=f"K from fitted field: {exc}",
                       k_source="field_ring:" + method)
    if res.K_I_pa_sqrt_m is None:
        return replace(fit, K_I_pa_sqrt_m=None, K_II_pa_sqrt_m=None,
                       error="K from fitted field: too few valid displacement samples in the ring "
                             "(enlarge the mesh region or reduce r_max).", k_source="field_ring:" + method)
    return replace(fit, K_I_pa_sqrt_m=float(res.K_I_pa_sqrt_m), K_II_pa_sqrt_m=float(res.K_II_pa_sqrt_m),
                   K_I_stderr_pa_sqrt_m=res.K_I_stderr_pa_sqrt_m, K_II_stderr_pa_sqrt_m=res.K_II_stderr_pa_sqrt_m,
                   singular_term_in_field=False, k_source="field_ring:" + method)


@dataclass
class XfemDicFieldGrid:
    """A regular-grid evaluation of one fitted frame's X-FEM-DIC field,
    ready to hand straight to matplotlib's `contour`/`contourf` -- see
    `evaluate_xfem_dic_field_grid`."""
    x: np.ndarray    # (nx,) px, reference-frame pixel coordinates
    y: np.ndarray    # (ny,) px
    u: np.ndarray    # (ny, nx) px -- NaN outside the mesh
    v: np.ndarray    # (ny, nx) px
    exx: np.ndarray  # (ny, nx) dimensionless
    eyy: np.ndarray  # (ny, nx)
    exy: np.ndarray  # (ny, nx) TENSOR shear strain (half the engineering gamma_xy) --
                      # matches core/dic_processing.py's own `_grid_strain` convention
    eps_eq: np.ndarray  # (ny, nx) von Mises equivalent strain (plane-strain assumption)
    # Derived from exx/eyy/exy in __post_init__ (so a point masked to NaN in
    # the strains is NaN here too): max / min in-plane principal strain and
    # max in-plane shear strain (e1 - e2)/2 -- see core/dic_fields.py.
    e1: np.ndarray | None = None
    e2: np.ndarray | None = None
    gmax: np.ndarray | None = None
    # Displacement magnitude sqrt(u^2 + v^2) (px), NaN wherever u or v is.
    umag: np.ndarray | None = None

    def __post_init__(self):
        from core.dic_fields import principal_strains, max_shear_strain
        if self.e1 is None or self.e2 is None:
            self.e1, self.e2 = principal_strains(self.exx, self.eyy, self.exy)
        if self.gmax is None:
            self.gmax = max_shear_strain(self.exx, self.eyy, self.exy)
        if self.umag is None:
            self.umag = np.hypot(np.asarray(self.u, dtype=float), np.asarray(self.v, dtype=float))

    def field(self, key: str) -> np.ndarray | None:
        return getattr(self, key, None)


def _mask_outside_dic_extent(grid_x: np.ndarray, grid_y: np.ndarray, fields: list,
                              dic_xs: np.ndarray, dic_ys: np.ndarray, dic_field: np.ndarray) -> None:
    """In-place NaNs out every array in `fields` (each `(len(grid_y),
    len(grid_x))`, e.g. `evaluate_xfem_dic_field_grid`'s own u/v/exx/eyy/
    exy/eps_eq) at every (x, y) location that falls OUTSIDE where the
    standard subset-DIC's own `dic_field` (on its `dic_xs`/`dic_ys` grid)
    has a valid (non-NaN) value there -- i.e. crops the X-FEM-DIC field
    grid down to the standard DIC's own correlated ROI, in x-y coordinates
    shared by both (both are already in the same reference-frame pixel
    coordinates, so no registration/alignment step is needed here, only a
    lookup).

    Without this, the X-FEM-DIC field is evaluated over its OWN mesh
    extent (`build_tip_centered_mesh`'s `roi_half_width_px` square around
    the tip), which routinely reaches past the edge of whatever the
    standard subset-DIC actually correlated (a smaller/differently-shaped
    ROI the user picked in the DIC step, often stopping well short of the
    image edges) -- side by side with the standard-DIC panel (which is
    simply blank/masked there, since it never had data), the X-FEM-DIC
    panel keeps showing a computed field value, including the near-tip
    singular term's largest ("hottest") values right where the crack tip
    itself commonly sits close to that ROI's edge. On a shared color
    scale, that reads as a large, real discrepancy between the two
    methods exactly where there's no ground truth to compare against at
    all -- misleading, not a real X-FEM-DIC artifact.

    A location's validity is looked up via a BILINEAR interpolation of
    `dic_field`'s own valid/NaN pattern (1.0/0.0), then thresholded at
    0.5 -- deliberately not a raw nearest-pixel lookup, so the cropped
    boundary itself isn't jagged at the standard DIC grid's own (usually
    much coarser) pixel spacing. `bounds_error=False, fill_value=0.0`
    (mirrors this module's/`core.crack_k1c`'s own `_sample_field`) means
    any point outside `dic_xs`/`dic_ys`'s own extent entirely is treated
    as invalid too, exactly like a point just inside that extent but on a
    NaN cell."""
    valid = np.isfinite(np.asarray(dic_field, dtype=float)).astype(float)
    interp = RegularGridInterpolator((dic_ys, dic_xs), valid, bounds_error=False, fill_value=0.0)
    xx, yy = np.meshgrid(grid_x, grid_y)
    query = np.column_stack([yy.ravel(), xx.ravel()])
    inside = (interp(query) > 0.5).reshape(xx.shape)
    outside = ~inside
    for f in fields:
        f[outside] = np.nan


def evaluate_xfem_dic_field_grid(fit: XfemDicFitResult, path_xy: np.ndarray, path_s: np.ndarray,
                                  tip_xy, tangent_xy, kappa: float, mu_pa: float, pixel_size_mm: float,
                                  r_enrich_px: float, step_px: float = 2.0,
                                  dic_xs: np.ndarray | None = None, dic_ys: np.ndarray | None = None,
                                  dic_field: np.ndarray | None = None,
                                  r_min_px: float = 0.0, r_max_px: float | None = None) -> XfemDicFieldGrid:
    """Evaluates the fitted X-FEM-DIC displacement field on a regular grid
    covering `fit.mesh`'s own extent, at `step_px` spacing (see
    `evaluate_xfem_dic_displacement_field`), and derives exx/eyy/exy from
    it by central finite differences on that UNIFORM grid (`np.gradient`
    with a constant spacing) -- the exact same formula
    core/dic_processing.py's `_grid_strain` uses for the standard subset-
    DIC field (`exx = du/dx`, `eyy = dv/dy`, `exy = 0.5*(du/dy + dv/dx)`),
    so the two are a like-for-like comparison when plotted together.
    `eps_eq` (von Mises equivalent strain) is then derived the same way
    core/crack_wave_io.py's own `CrackWaveFrame.eps_eq` is, via
    `core.dic_kinematics.von_mises_strain`.

    `r_min_px`/`r_max_px` are passed straight through to
    `evaluate_xfem_dic_displacement_field` (see that function's own
    docstring) -- pass the same values the `fit_xfem_dic_frame` call that
    produced `fit` actually used, or the reconstructed field's singular
    term won't match what was fitted.

    This is meant for a visual sanity check (isocontour overlay against
    the already-exported DIC field) -- differentiating an already-fitted
    smooth field is a much cheaper, much less rigorous operation than the
    Gauss-Newton correlation fit itself (see this module's own docstring
    on why raw-image re-correlation exists at all); it is not a from-
    scratch strain measurement in its own right, and inherits whatever
    error the fit itself has, amplified by differentiation.

    `dic_xs`/`dic_ys`/`dic_field`, if all three are given, crop the
    returned grid to the standard subset-DIC's own correlated ROI (see
    `_mask_outside_dic_extent`) -- the X-FEM-DIC mesh otherwise routinely
    extends past wherever the standard DIC ROI actually stops, which
    (especially right where the tip itself sits, exactly where the near-
    tip singular term is largest) reads as a spurious discrepancy between
    the two methods on a side-by-side comparison plot rather than the
    "nothing to compare here" it actually is. Pass the exact standard-DIC
    field being compared against (same field key the isocontour plot is
    about to draw) and its own `dataset.xs`/`dataset.ys` -- leaving any of
    the three as `None` (the default) skips cropping entirely, unchanged
    from this function's original behavior."""
    mesh = fit.mesh
    x_max = float(mesh.x_edges[-1])  # not n_elem_x*element_size_px -- see _sample_points' own comment
    y_max = float(mesh.y_edges[-1])
    # Cell-centered, like `_sample_points` -- a point sampled exactly AT
    # x_max/y_max would fail `Q4Mesh.locate` (that upper edge belongs to
    # no element, by the same half-open-interval convention as
    # `_sample_points`'s own caller), turning the field grid's very last
    # row/column into an all-NaN artifact for every frame. Centering the
    # grid a half-step inside the mesh avoids that by construction.
    x = np.arange(mesh.x0 + step_px / 2.0, x_max, step_px)
    y = np.arange(mesh.y0 + step_px / 2.0, y_max, step_px)
    xx, yy = np.meshgrid(x, y)
    u, v = evaluate_xfem_dic_displacement_field(
        fit, path_xy, path_s, tip_xy, tangent_xy, kappa, mu_pa, pixel_size_mm, r_enrich_px, xx, yy,
        r_min_px=r_min_px, r_max_px=r_max_px,
    )
    dudx = np.gradient(u, step_px, axis=1)
    dudy = np.gradient(u, step_px, axis=0)
    dvdx = np.gradient(v, step_px, axis=1)
    dvdy = np.gradient(v, step_px, axis=0)
    exx, eyy, exy = dudx, dvdy, 0.5 * (dudy + dvdx)
    eps_eq = von_mises_strain(exx, eyy, exy)

    if dic_xs is not None and dic_ys is not None and dic_field is not None:
        _mask_outside_dic_extent(x, y, [u, v, exx, eyy, exy, eps_eq], dic_xs, dic_ys, dic_field)

    return XfemDicFieldGrid(x=x, y=y, u=u, v=v, exx=exx, eyy=eyy, exy=exy, eps_eq=eps_eq)


# ---------------------------------------------------------------------------
# Time series over a Stage 3 tip track -- mirrors
# core/crack_k1c.py's `compute_k_time_series` (same tangent-sign-detection
# reasoning, duplicated rather than imported -- see that module's own
# docstring on why: this tool's stages are each kept self-sufficient).
# ---------------------------------------------------------------------------

@dataclass
class XfemDicFrameSample:
    frame_order_index: int
    source_frame_index: int
    time_s: float
    tip_xy_px: np.ndarray
    fit: XfemDicFitResult
    tip_arc_length_s: float | None = None   # the arc length actually used for this frame's fit
    tip_source: str = "fixed"               # "fixed" (from the Stage 3 trace), "auto (residual)"
                                             # (self-searched by grey-level residual) or
                                             # "auto (heaviside jump)" (self-searched by locating
                                             # where the fitted crack-opening amplitude vanishes) --
                                             # see compute_xfem_dic_time_series's `tip_detection_mode`
    tangent_xy: np.ndarray | None = None    # the local crack tangent used for this frame's fit --
                                             # stored so a caller can later re-evaluate the fitted
                                             # field (evaluate_xfem_dic_displacement_field/
                                             # evaluate_xfem_dic_field_grid) without re-deriving it


@dataclass
class XfemDicTimeSeries:
    leaf_index: int
    tip_method: str
    pixel_size_mm: float
    element_size_px: float
    r_enrich_px: float
    r_min_px: float = 0.0             # near-tip exclusion radius actually used to fit every sample
                                       # below (see fit_xfem_dic_frame's own r_min_px) -- 0.0 (no
                                       # exclusion) unless the UI's XFEM tuning controls set it
    r_max_px: float | None = None     # outer confinement radius actually used (see
                                       # fit_xfem_dic_frame's own r_max_px) -- always the RESOLVED
                                       # value (never None) once a series has been computed: equal to
                                       # r_enrich_px above when the UI didn't override it
    samples: list = field(default_factory=list)  # list[XfemDicFrameSample]
    default_critical_index: int | None = None
    auto_detect_tip: bool = False  # whether this series used the self-searched (not fixed) tip position
    tip_detection_mode: str = "residual"  # "residual" or "heaviside_jump" -- which self-search
                                    # algorithm was used, when auto_detect_tip=True (meaningless
                                    # otherwise); see compute_xfem_dic_time_series's docstring
    graded_mesh: bool = False      # whether this series used build_graded_tip_centered_mesh (fine near
                                    # the tip, coarser away) rather than a uniform tip-centered mesh
    max_iterations: int = 15       # Gauss-Newton iteration budget actually used to fit every sample
                                    # below (see fit_xfem_dic_frame's own max_iterations) -- an
                                    # under-converged fit (see XfemDicFrameSample.fit.converged/
                                    # .stalled/n_iterations -- .stalled means the residual plateaued
                                    # before max_iterations was even reached, so raising this budget
                                    # specifically would not have helped that frame) is a leading
                                    # cause of frame-to-frame K_I/K_II scatter, so this is recorded
                                    # for diagnosis even though it isn't otherwise reused (unlike
                                    # r_min_px/r_max_px, it doesn't affect field evaluation, only the
                                    # solve itself)
    position_window_px: float | None = None  # arc-length smoothing window (px) applied to the tip
                                    # POSITION used as each frame's mesh/fit origin, before fitting --
                                    # None/0 (default) uses tip_result's raw, unsmoothed per-frame
                                    # position, exactly as before this option existed; see
                                    # compute_xfem_dic_time_series's own docstring and
                                    # core.crack_k1c.compute_k_time_series's identical option
    k_from_field: bool = False     # True: K_I/K_II come from a ring fit on the fitted displacement field
                                    # (`refit_k_from_fitted_field`), not from joint grey-level unknowns
    k_field_method: str = "ring_affine"
    k_tip_offset_px: float = 0.0

    def sample_at(self, index: int) -> XfemDicFrameSample | None:
        if 0 <= index < len(self.samples):
            return self.samples[index]
        return None


def _build_tip_mesh(tip_xy, roi_half_width_px: float, element_size_px: float, r_enrich_px: float,
                     graded_mesh: bool, coarsen_factor: float) -> Q4Mesh:
    """Picks `build_tip_centered_mesh` (uniform) or
    `build_graded_tip_centered_mesh` (fine near the tip, coarser away) --
    the one shared spot both `compute_xfem_dic_time_series` and
    `_search_best_tip_s` build a frame/candidate's mesh from, so the two
    can never drift into building different mesh shapes for what's
    supposed to be the same setting."""
    if graded_mesh:
        return build_graded_tip_centered_mesh(tip_xy, roi_half_width_px, element_size_px, r_enrich_px,
                                               coarsen_factor=coarsen_factor)
    return build_tip_centered_mesh(tip_xy, roi_half_width_px, element_size_px)


def _position_at_arc_length(path_xy: np.ndarray, path_s: np.ndarray, s: float) -> np.ndarray:
    """Linear interpolation of the traced polyline's (x, y) position at
    arc length `s` (clamped to the path's own [path_s[0], path_s[-1]]
    range) -- the natural inverse of `_project_to_polyline`'s own `s`
    output. Needed by `_search_best_tip_s`: in fixed-trace mode, each
    frame's tip (x, y) is handed over directly by Stage 3
    (`tip_result.positions_xy`), but a self-searched candidate only ever
    picks an arc length, so the (x, y) actually used for that candidate's
    mesh/fit still has to be derived from the path geometry. A real
    traced path (`tip_result.path_positions`) is resampled finely enough
    (pixel-scale steps) that nearest-node snapping would already be
    almost indistinguishable from this -- linear interpolation is done
    anyway since it's just as cheap and exact regardless of the path's
    actual resampling density."""
    s_clamped = float(np.clip(s, path_s[0], path_s[-1]))
    idx = int(np.searchsorted(path_s, s_clamped, side="right") - 1)
    idx = int(np.clip(idx, 0, len(path_s) - 2))
    s0, s1 = path_s[idx], path_s[idx + 1]
    t = 0.0 if s1 <= s0 else (s_clamped - s0) / (s1 - s0)
    return path_xy[idx] + t * (path_xy[idx + 1] - path_xy[idx])


def _search_best_tip_s(ref_image, def_image, roi_half_width_px: float, element_size_px: float,
                        path_xy: np.ndarray, path_s: np.ndarray, tangent_at, kappa: float, mu_pa: float,
                        pixel_size_mm: float, r_enrich_px: float, sample_step_px: float, initial_uv_fn,
                        center_s: float, search_radius_px: float, search_step_px: float,
                        s_min: float, s_max: float, graded_mesh: bool = False,
                        coarsen_factor: float = 3.0, mono_floor: float | None = None,
                        mono_ceiling: float | None = None,
                        r_min_px: float = 0.0, r_max_px: float | None = None,
                        max_iterations: int = 15):
    """Brute-force 1D search, over candidate tip arc-length positions in
    [center_s - search_radius_px, center_s + search_radius_px] (clipped to
    [s_min, s_max], and further clipped to [mono_floor, mono_ceiling] when
    given -- see below) at `search_step_px` spacing, for the position that
    gives the BEST (lowest) `fit_xfem_dic_frame` grey-level residual RMS
    at this one frame -- i.e. uses the correlation residual itself as the
    "is the crack geometry right here" signal: a wrong tip_s misclassifies
    which elements the already-opened crack actually splits (see
    `classify_mesh_enrichment`), which measurably worsens the fit, so the
    genuinely best-fitting candidate is a reasonable, if not provably
    optimal (a real function could have more than one local minimum -- see
    `compute_xfem_dic_time_series`'s own docstring on this), estimate of
    where the tip actually is.

    Only CONVERGED candidates (`cand_fit.converged`) are eligible for
    "best" -- an under-converged Gauss-Newton solve (stopped early, or
    numerically unstable) can report a low residual_rms by coincidence
    without that residual meaning what it's assumed to mean here, so it
    must not be allowed to win over an honestly converged, if slightly
    higher-residual, candidate. Falls back to the best UNconverged
    candidate only if literally none converged, so a frame still gets a
    best-effort answer rather than nothing.

    `mono_floor`/`mono_ceiling`, when given (by `compute_xfem_dic_time_series`,
    from the last frame it actually accepted a position for), enforce that
    a physical crack cannot shrink: exactly one of the two is ever passed,
    depending on which direction the crack grows in arc length over time --
    `mono_floor` excludes every candidate `s < mono_floor` when the crack
    grows toward increasing `s`, `mono_ceiling` excludes every candidate
    `s > mono_ceiling` when it grows toward decreasing `s`. This is applied
    BEFORE any candidate is evaluated (it narrows `candidate_s` itself, not
    just which one gets picked afterward), so a tempting but physically
    impossible (shrinking) candidate is never even tried, let alone allowed
    to win on a spuriously low residual.

    One full Gauss-Newton solve per candidate -- genuinely N times the
    cost of a single fixed-tip fit (N = 2*search_radius_px/search_step_px
    + 1), an explicit trade-off the caller/UI has to opt into, not hidden
    here.

    Returns (best_s, best_fit, best_tip_xy, best_tangent, best_mesh), or
    None if every candidate failed (e.g. every one warped too many sample
    points outside the deformed frame)."""
    lo = max(s_min, center_s - search_radius_px)
    hi = min(s_max, center_s + search_radius_px)
    if mono_floor is not None:
        lo = max(lo, mono_floor)
    if mono_ceiling is not None:
        hi = min(hi, mono_ceiling)
    if hi <= lo:
        # The search radius window around this frame's nominal position is
        # entirely below mono_floor (or above mono_ceiling) -- the crack
        # would have to shrink to explore it. Collapse to the single
        # tightest position that still doesn't shrink the crack, clamped
        # to the path's own extent, rather than silently ignoring the
        # constraint (or, when no constraint applies at all here, this is
        # just the original degenerate-radius single-candidate case).
        candidate_s = [float(np.clip(center_s, min(lo, hi), max(lo, hi)))]
    else:
        n_steps = max(1, int(round((hi - lo) / max(search_step_px, 1e-6))))
        candidate_s = list(np.linspace(lo, hi, n_steps + 1))

    best_converged = None
    best_any = None
    for s in candidate_s:
        cand_tip_xy = _position_at_arc_length(path_xy, path_s, s)
        path_idx = int(np.argmin(np.abs(path_s - s)))
        cand_tangent = tangent_at(path_idx)
        try:
            cand_mesh = _build_tip_mesh(cand_tip_xy, roi_half_width_px, element_size_px, r_enrich_px,
                                         graded_mesh, coarsen_factor)
            cand_fit = fit_xfem_dic_frame(
                ref_image, def_image, cand_mesh, path_xy, path_s, cand_tip_xy, s, cand_tangent,
                kappa, mu_pa, pixel_size_mm, r_enrich_px, sample_step_px=sample_step_px,
                initial_uv_fn=initial_uv_fn, r_min_px=r_min_px, r_max_px=r_max_px,
                max_iterations=max_iterations,
            )
        except XfemDicError:
            continue
        if cand_fit.error is not None or cand_fit.residual_rms is None:
            continue
        candidate = (s, cand_fit, cand_tip_xy, cand_tangent, cand_mesh)
        if best_any is None or cand_fit.residual_rms < best_any[1].residual_rms:
            best_any = candidate
        if cand_fit.converged and (
            best_converged is None or cand_fit.residual_rms < best_converged[1].residual_rms
        ):
            best_converged = candidate
    return best_converged if best_converged is not None else best_any


# ---------------------------------------------------------------------------
# Auto-detect tip, ALTERNATIVE method: locate where the fitted crack-face
# OPENING (the Heaviside jump amplitude) vanishes, instead of comparing the
# grey-level residual across many candidate tip positions.
#
# `_search_best_tip_s` above tries many candidate tip_s values and keeps
# whichever gives the best correlation residual. That works, but every
# candidate needs its own full Gauss-Newton solve, AND the residual is an
# indirect signal (a wrong tip_s misclassifies elements, which worsens the
# fit -- see that function's docstring). The near-tip equivalent-strain
# field a user would eyeball to "see" the tip in the isocontour display
# (evaluate_xfem_dic_field_grid's eps_eq) is not a good direct criterion
# either: it is the fitted K-dominant singular field, centered wherever the
# CURRENT candidate tip_s already is, so its magnitude near "the tip" is
# partly an artifact of where the model was told to put the singularity,
# not an independent measurement, and any fixed threshold on it depends on
# the fitted K amplitude (hence on E/nu/pixel size) -- not portable across
# specimens or materials.
#
# The Heaviside jump amplitude (`XfemDicFitResult.heaviside_disp_x/y`, in
# PIXELS, one value per enriched node) is different: it IS the fit's own
# estimate of how far the two crack faces have separated at that node --
# literally the physical definition of "still open" vs "closed" (down to
# noise), in a unit (px) that doesn't depend on material properties at all,
# so a small fixed threshold (a fraction of a pixel) is meaningful across
# datasets. And it can be read off from ONE fit (built on a deliberately
# OVER-covering trial tip position, so the enriched region safely reaches
# past the true tip), rather than needing N candidate fits.
# ---------------------------------------------------------------------------

def _refine_tip_s_from_heaviside_jump(mesh: Q4Mesh, heaviside_nodes: np.ndarray,
                                       heaviside_disp_x: np.ndarray, heaviside_disp_y: np.ndarray,
                                       path_xy: np.ndarray, path_s: np.ndarray, trial_tip_s: float,
                                       growth_sign: float, jump_threshold_px: float,
                                       bin_width_px: float) -> tuple[str, float | None]:
    """From ONE already-completed fit's Heaviside jump DOFs (the fitted
    crack-face opening amplitude at every enriched node, in px -- see this
    section's own header comment for why this is a better criterion than
    residual or eps_eq for this purpose), locates where along `path_xy`
    the crack stops being open.

    Bins every enriched node by its arc-length projection onto the path
    (`_project_to_polyline`), taking the MEDIAN jump magnitude
    (hypot(heaviside_disp_x, heaviside_disp_y)) within each `bin_width_px`
    bin. An early design used the max instead ("a bin only counts as
    'closed' once EVERY nearby node agrees"), which is swayed by a single
    noisy/spurious node; the median requires a majority of nearby nodes
    to agree, matching the robustness rationale already used elsewhere in
    this codebase for smoothing K(t) against noisy samples (see
    core/crack_k1c.py's `smooth_k_series`). Walks bins from far BEHIND
    the trial tip toward it (in whichever direction `growth_sign` says
    the crack grows over time -- `path_s` itself may run either way)
    looking for the last bin that's still clearly open (median jump >=
    `jump_threshold_px`); the refined tip is placed at that bin's own
    outer edge.

    Resolution note: this locates the tip to within roughly one
    `bin_width_px` (normally `element_size_px`), not to sub-element
    precision. A partially-enriched element near the true discontinuity
    smears the sharp opening/closing transition over about its own
    width, so the jump magnitude decays gradually rather than dropping
    to zero exactly at the true tip -- no choice of statistic (max,
    median, ...) or threshold removes that; only a finer mesh
    (`element_size_px`) does, at the cost of more (and costlier) enriched
    DOFs. Callers wanting the residual search's typical few-px precision
    should still use that method; this one trades some of that precision
    for far fewer full solves per frame.

    Returns (status, refined_s):
    - ("found", s): a real closed region was located before `trial_tip_s`
      -- `s` is the refined tip arc length.
    - ("extend", None): every sampled bin, including the one closest to
      `trial_tip_s` itself, is still open -- `trial_tip_s` likely
      under-covers the true tip; the caller should retry with a larger
      trial extension.
    - ("empty", None): no enriched node showed any real opening at all
      (every bin already below threshold) -- this trial's geometry isn't
      trustworthy (e.g. the crack is shorter than even the un-extended
      nominal position, or the fit didn't actually resolve an opening);
      the caller should not use this trial."""
    idx = np.flatnonzero(heaviside_nodes)
    if idx.size == 0:
        return "empty", None
    nodes_xy = mesh.nodes_xy()[idx]
    jump_mag = np.hypot(heaviside_disp_x[idx], heaviside_disp_y[idx])
    _signed, s_proj, _tx, _ty = _project_to_polyline(nodes_xy, path_xy, path_s)

    # Reorient so "increasing" always means "toward the trial tip",
    # regardless of which way path_s itself runs.
    s_signed = growth_sign * s_proj
    bin_width = max(bin_width_px, 1e-6)
    bin_index = np.floor(s_signed / bin_width).astype(int)

    ordered_bins = sorted(set(bin_index.tolist()))
    bin_stat = {b: float(np.median(jump_mag[bin_index == b])) for b in ordered_bins}
    open_bins = [b for b in ordered_bins if bin_stat[b] >= jump_threshold_px]
    if not open_bins:
        return "empty", None

    last_open_bin = open_bins[-1]
    if ordered_bins.index(last_open_bin) == len(ordered_bins) - 1:
        # Still open in the outermost sampled bin -- i.e. right up against
        # trial_tip_s -- this trial didn't reach far enough past the true
        # tip to see it close.
        return "extend", None

    # The last clearly-open bin's own outer edge: the crack is open
    # somewhere inside that bin and closed in the next one, so the true
    # tip lies at or just before this boundary. The refined tip is placed
    # exactly at this edge rather than further out: a partially-enriched
    # element near the true tip already smears the sharp opening/closing
    # transition over roughly one element width (`bin_width_px`, normally
    # element_size_px), so pushing the refined tip further past this edge
    # would only add avoidable over-extension on top of that existing
    # smoothing bias, without gaining any real margin.
    refined_s_signed = (last_open_bin + 1.0) * bin_width
    refined_tip_s = growth_sign * refined_s_signed
    return "found", float(refined_tip_s)


def _search_tip_s_by_heaviside_jump(ref_image, def_image, roi_half_width_px: float,
                                     element_size_px: float, path_xy: np.ndarray, path_s: np.ndarray,
                                     tangent_at, kappa: float, mu_pa: float, pixel_size_mm: float,
                                     r_enrich_px: float, sample_step_px: float, initial_uv_fn,
                                     nominal_s: float, growth_sign: float, search_radius_px: float,
                                     s_min: float, s_max: float, jump_threshold_px: float,
                                     graded_mesh: bool = False, coarsen_factor: float = 3.0,
                                     mono_floor: float | None = None, mono_ceiling: float | None = None,
                                     max_attempts: int = 4,
                                     r_min_px: float = 0.0, r_max_px: float | None = None,
                                     max_iterations: int = 15):
    """Driver for the Heaviside-jump tip-detection method (see this
    section's header comment): fits a deliberately over-covering trial
    mesh, then reads the refined tip position straight off that fit's own
    Heaviside jump profile (`_refine_tip_s_from_heaviside_jump`) instead
    of comparing many candidate fits' residuals.

    Trials are tried at increasing pushes past `nominal_s`, STARTING AT
    `nominal_s` ITSELF (extend=0) rather than jumping straight to
    `search_radius_px`: on real (noisy, not synthetic) images, a single
    big blind first jump risks landing somewhere the Gauss-Newton solve
    simply doesn't converge from cold, and with the original one-shot
    design that converged failure lost the whole frame outright -- no
    smaller, safer position was ever tried. Starting at extend=0 costs
    one extra fit in the common case (that position virtually always
    converges, since it's exactly where the fixed-trace fit already
    works elsewhere in this app) but means the search only ever needs to
    succeed at SOME position along an increasing ladder, not at whichever
    one the caller guessed first.

    Escalation is also bounded, not unbounded doubling: pushing the trial
    more than `roi_half_width_px` past `nominal_s` is pointless regardless
    of `max_attempts`, because the trial mesh is a FIXED-size square
    centered on the trial's own (x, y) position -- past that distance it
    no longer reaches back to `nominal_s` at all, so a non-convergent or
    "empty" result out there says nothing trustworthy about whether the
    crack actually grew; it only means the trial missed.

    If a larger trial then fails outright (doesn't converge, or comes
    back "empty" after a smaller trial had already confirmed the crack
    was still open at ITS edge), the search doesn't discard everything
    and fail the frame: it falls back to reporting that smaller, already-
    confirmed-open trial's own position -- a conservative "the crack
    reaches at least this far" answer, rather than losing the frame
    entirely because one further, riskier push didn't pan out. Only when
    even the safest (extend=0) attempt itself fails to converge, or shows
    no resolvable opening at all, does this return None.

    Once a refined position is found (or the fallback above is used),
    that position's own fit is returned as-is when it's already in hand,
    or one more fit AT that exact position is done otherwise -- either
    way the returned `fit` reflects the actual reported tip, never a
    trial that was only ever meant to be over-covering.

    Returns (best_s, best_fit, best_tip_xy, best_tangent, best_mesh), or
    None if no attempt produced a usable, converged geometry."""
    bin_width = element_size_px
    base_step = max(search_radius_px, element_size_px)
    max_extend = max(base_step, roi_half_width_px)
    extends = [0.0]
    step = base_step
    while extends[-1] < max_extend and len(extends) < max_attempts:
        extends.append(min(extends[-1] + step, max_extend))
        step *= 2.0

    def _clamp_s(s):
        s = float(np.clip(s, s_min, s_max))
        if mono_floor is not None:
            s = max(s, mono_floor)
        if mono_ceiling is not None:
            s = min(s, mono_ceiling)
        return s

    def _fit_at(s):
        tip_xy = _position_at_arc_length(path_xy, path_s, s)
        path_idx = int(np.argmin(np.abs(path_s - s)))
        tangent = tangent_at(path_idx)
        try:
            mesh = _build_tip_mesh(tip_xy, roi_half_width_px, element_size_px, r_enrich_px,
                                    graded_mesh, coarsen_factor)
            fit = fit_xfem_dic_frame(
                ref_image, def_image, mesh, path_xy, path_s, tip_xy, s, tangent,
                kappa, mu_pa, pixel_size_mm, r_enrich_px, sample_step_px=sample_step_px,
                initial_uv_fn=initial_uv_fn, r_min_px=r_min_px, r_max_px=r_max_px,
                max_iterations=max_iterations,
            )
        except XfemDicError:
            return None
        return (mesh, fit, tip_xy, tangent)

    # The largest trial so far that BOTH converged AND still showed the
    # crack open at its own edge -- a safe fallback position if a later,
    # larger trial doesn't pan out. Holds the full _fit_at(...) result so
    # falling back to it never needs a redundant extra solve.
    last_good_s = None
    last_good = None

    for extend in extends:
        trial_s = _clamp_s(nominal_s + growth_sign * extend)
        trial = _fit_at(trial_s)
        if trial is None or trial[1].error is not None or not trial[1].converged:
            if last_good is not None:
                break  # this push didn't converge -- fall back below
            continue  # nothing safe to fall back on yet -- try further out
        trial_mesh, trial_fit, _trial_tip_xy, _trial_tangent = trial
        status, refined_s = _refine_tip_s_from_heaviside_jump(
            trial_mesh, trial_fit.heaviside_nodes, trial_fit.heaviside_disp_x,
            trial_fit.heaviside_disp_y, path_xy, path_s, trial_s, growth_sign,
            jump_threshold_px, bin_width,
        )
        if status == "found":
            refined_s = _clamp_s(refined_s)
            refined = _fit_at(refined_s)
            if refined is None or refined[1].error is not None:
                if last_good is not None:
                    break
                return None
            refined_mesh, refined_fit, refined_tip_xy, refined_tangent = refined
            return (refined_s, refined_fit, refined_tip_xy, refined_tangent, refined_mesh)
        if status == "extend":
            last_good_s, last_good = trial_s, trial
            continue
        # status == "empty"
        if last_good is not None:
            break  # this trial overshot past where the mesh could still
                   # see the crack -- fall back to the last position that
                   # was confirmed still open, below.
        return None  # no opening resolved anywhere, even right at the
                     # nominal position -- a genuine "nothing to find"
                     # result, not a search-range problem.

    if last_good is None:
        return None
    last_good_mesh, last_good_fit, last_good_tip_xy, last_good_tangent = last_good
    return (last_good_s, last_good_fit, last_good_tip_xy, last_good_tangent, last_good_mesh)


def compute_xfem_dic_time_series(tip_result, get_frame, reference_source_frame_index: int,
                                  young_modulus_pa: float, poisson_ratio: float, plane_stress: bool,
                                  pixel_size_mm: float, element_size_px: float, r_enrich_px: float,
                                  roi_half_width_px: float | None = None, sample_step_px: float = 2.0,
                                  frame_order_indices: list[int] | None = None,
                                  warm_start_from_dataset=None, progress_cb=None,
                                  auto_detect_tip: bool = False, tip_search_radius_px: float = 40.0,
                                  tip_search_step_px: float = 5.0, graded_mesh: bool = False,
                                  coarsen_factor: float = 3.0, tip_detection_mode: str = "residual",
                                  heaviside_jump_threshold_px: float = 0.15,
                                  r_min_px: float = 0.0, r_max_px: float | None = None,
                                  max_iterations: int = 15,
                                  position_window_px: float | None = None,
                                  estimate_uncertainty: bool = False,
                                  cancel_check=None,
                                  k_from_field: bool = False, k_field_method: str = "ring_affine",
                                  k_tip_offset_px: float = 0.0) -> XfemDicTimeSeries:
    """Fits K_I(t)/K_II(t) directly from raw images at every requested
    frame of `tip_result` (a core.crack_tip_tracking.TipTrackResult -- run
    Stage 3 first, exactly as core.crack_k1c.compute_k_time_series does).

    The crack PATH itself always comes from `tip_result` (this method
    never re-detects the skeleton -- "fixed crack path" is this feature's
    deliberately chosen scope, see this module's docstring). What CAN
    optionally be found independently, per frame, is the tip's POSITION
    along that already-traced path:

    `auto_detect_tip=False` (default): the tip position at each frame is
    read directly from `tip_result` (`positions_xy`/`arc_length_s`),
    exactly as before -- this is the ridge/wave trace already computed on
    the first sub-tab, unchanged.

    `auto_detect_tip=True`: at each frame, instead of trusting
    `tip_result`'s own per-frame position, this searches candidate tip
    arc-length positions along the SAME traced path (see
    `_search_best_tip_s`) within `tip_search_radius_px` of THAT SAME
    frame's own nominal position (`tip_result`'s ridge/wave-tracked
    `arc_length_s` at this frame -- every frame is anchored to Stage 3's
    own trace independently, NOT to the previous frame's self-found
    position), at `tip_search_step_px` spacing, and keeps whichever
    CONVERGED candidate gives the best (lowest) grey-level correlation
    residual (falling back to the best unconverged one only if none
    converged -- see `_search_best_tip_s`) -- i.e. this method refines,
    frame by frame, where along the path the crack tip actually is
    around Stage 3's own estimate, using its own fit quality as the
    signal, rather than blindly reusing Stage 3's ridge-picked/wave-
    derived position. Every frame is anchored independently to `tip_result`
    rather than chained off the previous frame's own self-found position,
    because this is a plain grid search on a possibly non-convex objective -- it can
    settle on the wrong local minimum on any one frame, especially with
    noisy images or a coarse `tip_search_step_px` (see
    `_search_best_tip_s`) -- and without per-frame anchoring, one such
    mistake would permanently drag every later frame's search window
    away from the real tip with it, since each one only searched around
    its (by then wrong) predecessor. This is substantially more expensive
    (one full Gauss-Newton solve per candidate position tested, not just
    one), so it's offered as an additional, independent measurement to
    compare against the fixed-trace one, not a replacement for it.

    A physical crack cannot shrink, so the search also enforces that each
    frame's ACCEPTED position never moves backward (in arc length, along
    the growth direction inferred from `tip_result`'s own overall trend)
    relative to the last frame this method actually accepted a position
    for -- see `_search_best_tip_s`'s `mono_floor`/`mono_ceiling`. This
    excludes shrinking candidates from consideration outright (not just
    from winning), so a frame whose correlation residual happens to look
    best at an earlier/smaller position than the crack has already reached
    can never be selected on that basis alone.

    `tip_detection_mode` picks WHICH self-search algorithm `auto_detect_tip
    =True` uses (ignored when it's False):

    `tip_detection_mode="residual"` (default): the grid search over
    `tip_search_radius_px`/`tip_search_step_px` described above
    (`_search_best_tip_s`).

    `tip_detection_mode="heaviside_jump"`: instead of comparing many
    candidate fits' residuals, fits a deliberately over-covering trial
    mesh (starting AT the nominal position itself, then pushed further
    out in bounded, increasing steps -- up to `tip_search_radius_px` past
    nominal and no further than `roi_half_width_px` past it either way --
    each time the crack still looks open all the way to the trial's own
    edge; and falling back to the last position that WAS confirmed still
    open if a larger push then fails to converge or overshoots past
    where the mesh can still see the crack, rather than losing the whole
    frame over one risky push -- see `_search_tip_s_by_heaviside_jump`'s
    own docstring for the full reasoning), then reads the refined tip
    position straight off that fit's own Heaviside jump
    amplitude (`XfemDicFitResult.heaviside_disp_x/y`, in pixels) --
    locating where the fitted crack-face opening drops below
    `heaviside_jump_threshold_px` (see `_search_tip_s_by_heaviside_jump`,
    and that function's own header comment for why this is a better
    criterion than the residual OR the near-tip strain field a user might
    eyeball in the isocontour display: the latter two are both centered
    wherever the CURRENT candidate tip already is, so they're partly an
    artifact of that choice rather than an independent signal, and a
    strain-based threshold isn't portable across specimens/materials the
    way a small, fixed pixel threshold on a measured displacement is).
    Typically far cheaper (1-2 full fits per frame instead of
    2*radius/step + 1), at the cost of depending on Heaviside jump DOFs
    being well resolved (a crack that barely opens, or an element size too
    coarse relative to that opening, gives this method less to work with
    than the residual-based search), AND at the cost of coarser spatial
    resolution: a partially-enriched element smears the sharp
    opening/closing transition over roughly its own width
    (`element_size_px`), so the position this locates is only accurate to
    within about one to two element widths, not to the few-pixel
    precision the residual search's finer `tip_search_step_px` grid can
    reach. Prefer "residual" when per-frame precision matters most;
    prefer "heaviside_jump" when processing many frames and a coarser,
    much cheaper per-frame position is an acceptable trade.

    Both modes obey the "crack cannot shrink" constraint above, and both
    record which arc length was actually used (`tip_arc_length_s`) and how
    (`tip_source`) on each resulting `XfemDicFrameSample` -- the returned
    series' own `auto_detect_tip`/`tip_detection_mode` fields record which
    mode produced it.

    `get_frame(source_frame_index) -> np.ndarray` returns the preprocessed
    (cropped/contrast-stretched/downscaled -- core.dic_preprocessing.
    preprocess_frame) frame, in the SAME pixel-coordinate system as
    `tip_result.path_positions` -- the caller (UI) is responsible for that
    alignment, e.g. by reusing the crop/downscale settings that produced
    the loaded DIC dataset. `reference_source_frame_index` is the
    undeformed reference frame (dataset.frames[0].. in the usual case,
    matching Stage 1's own reference frame).

    `frame_order_indices`, when given, restricts the fit to those indices
    into `tip_result.times`/`positions_xy` (default: every frame with a
    detected tip position) -- lets the UI fit a single frame on demand
    instead of the whole sequence, which is far cheaper for interactive
    tuning of `element_size_px`/`r_enrich_px` before committing to a full
    run.

    `roi_half_width_px`, default None, uses `8 * element_size_px` -- the
    background mesh only needs to extend far enough from the crack path to
    let the far-field/rigid-body motion be resolved; it does not need to
    cover the whole frame.

    `warm_start_from_dataset(source_frame_index, x, y) -> (u, v) | None`,
    optional, is passed through to `fit_xfem_dic_frame`'s `initial_uv_fn`
    for each frame (typically sampling that frame's already-computed
    subset-DIC u/v field, see core/crack_k1c.py's `_sample_field` for the
    same interpolation pattern) -- substantially speeds/stabilizes
    convergence; each frame gets its own closure over its own source frame
    index.

    `graded_mesh=False` (default): every frame's mesh is the plain
    UNIFORM `build_tip_centered_mesh`, `element_size_px` everywhere, as
    before. `graded_mesh=True` uses `build_graded_tip_centered_mesh`
    instead -- still `element_size_px` within `r_enrich_px` of the tip,
    but growing out to `element_size_px * coarsen_factor` over the rest
    of `roi_half_width_px` -- better near-tip resolution and fewer, larger
    elements far from the crack for the same overall region, at no extra
    Gauss-Newton cost (the fit itself doesn't care how big any one
    element is, only `Q4Mesh.locate`'s per-sample (xi, eta)).

    `r_min_px`/`r_max_px`, both passed straight through to every frame's
    `fit_xfem_dic_frame` call below (and to every candidate tried by
    `auto_detect_tip`'s own search), confine the singular K_I/K_II term's
    radial weight -- see `fit_xfem_dic_frame`'s own docstring for the full
    reasoning and the K1c-ring-fit analogy. Defaults (`r_min_px=0.0`,
    `r_max_px=None` reusing `r_enrich_px`) reproduce this function's exact
    original behavior, from before these two parameters existed. The
    RESOLVED `r_max_px` actually used (never `None`) is recorded on the
    returned series (`XfemDicTimeSeries.r_max_px`), alongside `r_min_px`
    itself, so a later caller reconstructing the fitted field (e.g. for
    the isocontour display) can reuse the exact same window a series was
    computed with.

    `cancel_check`, when given, is checked once per frame (before that
    frame's own fit/search starts) and raises
    `core.parallel.OperationCancelled` the first time it returns truthy --
    see that exception's own docstring. Cooperative: a frame's own fit
    (or, with `auto_detect_tip=True`, its whole candidate search) already
    in progress always finishes first.

    `max_iterations`, passed straight through to every frame's (and every
    auto-detect candidate's) `fit_xfem_dic_frame` call, is the Gauss-
    Newton iteration BUDGET for this fit -- unlike K1c's ring fit (a
    single linear least-squares solve), this one is a nonlinear solve for
    the background field, the Heaviside jump and the singular K_I/K_II
    term all AT ONCE, and some frames genuinely need more than the old
    hardcoded default of 15 to actually converge (`XfemDicFitResult.
    converged`/`n_iterations` report this per frame). An UNDER-converged
    fit is a leading, evidence-backed cause of frame-to-frame K_I/K_II
    scatter that otherwise looks like noise: the residual has only been
    partially reduced when iteration stops, so the fitted K amplitude
    itself is still partway between its (bad) starting guess and its true
    converged value, and exactly how far short depends on that frame's
    own starting conditions -- i.e. it varies frame to frame even on an
    otherwise-smooth, physically continuous K(t). Raising this costs
    roughly proportionally more time only on the frames that actually
    need the extra iterations (an already-converged frame simply stops
    early, same as before); it does not change any already-converged
    frame's result. Default (15) reproduces this function's exact prior
    behavior.

    `position_window_px`, the XFEM-DIC counterpart to `core.crack_k1c.
    compute_k_time_series`'s identical option, smooths the tip POSITION
    used as each frame's mesh/fit origin (via `core.crack_tip_tracking.
    smoothed_tip_arc_length_and_positions`, the same local-linear-
    regression-against-time smoothing K1c uses) before that position is
    used to build the frame's mesh and orient its fit -- None/0 (default)
    uses `tip_result`'s raw, unsmoothed per-frame `positions_xy`/
    `arc_length_s`, exactly as before this option existed. This matters
    here for the same reason it matters for K1c: the near-tip singular
    term this fit solves for is exactly as sensitive to origin
    (mis)placement as K1c's ring fit is, but until this option existed
    XFEM-DIC had NO position-smoothing mechanism at all (not even
    inheriting K1c's own `position_window_px`, since the two fits are
    computed independently from the same raw `tip_result`) -- so a user
    who already tuned K1c's smoothing to tame ridge/wave tracking noise
    would still see that same raw noise feed directly into XFEM-DIC's
    fit unless this is ALSO set here. When set, `auto_detect_tip`'s own
    search is anchored to this smoothed nominal position (not the raw
    one), so the two options compose rather than fight each other.

    `estimate_uncertainty=False` (default): every sample's `fit.K_I_
    stderr_pa_sqrt_m`/`K_II_stderr_pa_sqrt_m` stay `None` -- see
    `fit_xfem_dic_frame`'s own docstring for what `True` computes and why
    it's opt-in (extra cost). With `auto_detect_tip=True`, the search
    itself (`_search_best_tip_s`/`_search_tip_s_by_heaviside_jump`) is
    NEVER asked to estimate uncertainty for every candidate it tries
    (that would multiply this already-expensive search's own cost by
    however many candidates it evaluates, for N-1 fits whose uncertainty
    is immediately thrown away) -- instead, once the winning candidate's
    position/mesh/tangent is known, this refits JUST that one position
    a second time with `estimate_uncertainty=True` and uses that refit's
    `fit` (identical K_I/K_II to the first, deterministic given the same
    inputs -- only its uncertainty fields differ) as the sample's own."""
    if not np.isfinite(tip_result.arc_length_s).any():
        raise XfemDicError("This tip track has no detected tip positions -- re-run \"Track tip\" first.")

    kappa = kolosov_kappa(poisson_ratio, plane_stress)
    mu = shear_modulus(young_modulus_pa, poisson_ratio)
    effective_r_max_px = float(r_enrich_px if r_max_px is None else r_max_px)

    tangent_sign = 1.0
    valid_arc = np.isfinite(tip_result.arc_length_s)
    if np.count_nonzero(valid_arc) >= 2:
        t_valid = tip_result.times[valid_arc]
        s_valid = tip_result.arc_length_s[valid_arc]
        # np.ptp guard: a (numerically) constant tip track has a random-sign fitted slope
        if not np.allclose(t_valid, t_valid[0]) and np.ptp(s_valid) > 1e-6:
            slope = float(np.polyfit(t_valid, s_valid, 1)[0])
            if slope < 0:
                tangent_sign = -1.0
    window_px = 2.0 * r_enrich_px
    tangents = path_tangents(tip_result.path_positions, tip_result.path_arc_length,
                              window_px=window_px) * tangent_sign

    if roi_half_width_px is None:
        roi_half_width_px = 8.0 * element_size_px

    ref_image = get_frame(reference_source_frame_index)

    s_min = float(tip_result.path_arc_length[0])
    s_max = float(tip_result.path_arc_length[-1])

    # Position smoothing (opt-in, off by default -- see `position_window_px`
    # in the docstring above): mirrors core.crack_k1c.compute_k_time_series's
    # identical option exactly -- a smoothed arc-length/position pair, used
    # as this frame's nominal mesh/fit origin (and, when auto_detect_tip is
    # on, as the center THAT search is anchored to) in place of the raw
    # Stage-3 track. `arc_length_for_fit`/`positions_for_fit` default right
    # back to the raw track when smoothing is off, so the loop below doesn't
    # need an `if` per frame.
    if position_window_px:
        arc_length_for_fit, positions_for_fit = smoothed_tip_arc_length_and_positions(
            tip_result.times, tip_result.arc_length_s, tip_result.path_positions,
            tip_result.path_arc_length, position_window_px,
        )
    else:
        arc_length_for_fit = tip_result.arc_length_s
        positions_for_fit = tip_result.positions_xy

    # The crack cannot shrink -- tracks the last arc length auto_detect_tip
    # actually accepted, in the SAME growth direction `tangent_sign` above
    # already established from tip_result's own overall trend, so later
    # frames can be prevented from settling on a position that would mean
    # the crack got shorter (see the mono_floor/mono_ceiling handoff below
    # and this function's own docstring).
    last_accepted_auto_s = None

    indices = frame_order_indices if frame_order_indices is not None else range(len(tip_result.times))
    samples = []
    total = len(list(indices)) if frame_order_indices is not None else len(tip_result.times)
    done = 0
    for i in indices:
        check_cancelled(cancel_check)
        if not np.isfinite(tip_result.arc_length_s[i]):
            continue
        nominal_tip_xy = positions_for_fit[i]
        nominal_tip_s = float(arc_length_for_fit[i])
        source_idx = int(tip_result.source_frame_positions[i])
        def_image = get_frame(source_idx)
        initial_uv_fn = None
        if warm_start_from_dataset is not None:
            initial_uv_fn = lambda x, y, _idx=source_idx: warm_start_from_dataset(_idx, x, y)

        if auto_detect_tip:
            # Anchored to THIS frame's own nominal (ridge/wave-tracked)
            # position, not to the previous frame's self-found one -- see
            # this function's docstring for why (no drift accumulation).
            # The crack cannot shrink: exactly one of mono_floor/
            # mono_ceiling is set, from the last position this method
            # actually accepted, in whichever direction tangent_sign says
            # the crack grows over time.
            mono_floor = None
            mono_ceiling = None
            if last_accepted_auto_s is not None:
                if tangent_sign >= 0:
                    mono_floor = last_accepted_auto_s
                else:
                    mono_ceiling = last_accepted_auto_s
            if tip_detection_mode == "heaviside_jump":
                found = _search_tip_s_by_heaviside_jump(
                    ref_image, def_image, roi_half_width_px, element_size_px,
                    tip_result.path_positions, tip_result.path_arc_length,
                    lambda idx: tangents[idx], kappa, mu, pixel_size_mm, r_enrich_px, sample_step_px,
                    initial_uv_fn, nominal_s=nominal_tip_s, growth_sign=tangent_sign,
                    search_radius_px=tip_search_radius_px, s_min=s_min, s_max=s_max,
                    jump_threshold_px=heaviside_jump_threshold_px, graded_mesh=graded_mesh,
                    coarsen_factor=coarsen_factor, mono_floor=mono_floor, mono_ceiling=mono_ceiling,
                    r_min_px=r_min_px, r_max_px=effective_r_max_px, max_iterations=max_iterations,
                )
                mode_label = "auto (heaviside jump)"
                no_position_error = (
                    "Auto-detected tip search (Heaviside jump) found no usable position for this "
                    "frame -- every trial either failed to converge or showed no resolvable crack "
                    "opening at all."
                )
            else:
                found = _search_best_tip_s(
                    ref_image, def_image, roi_half_width_px, element_size_px,
                    tip_result.path_positions, tip_result.path_arc_length,
                    lambda idx: tangents[idx], kappa, mu, pixel_size_mm, r_enrich_px, sample_step_px,
                    initial_uv_fn, center_s=nominal_tip_s, search_radius_px=tip_search_radius_px,
                    search_step_px=tip_search_step_px, s_min=s_min, s_max=s_max,
                    graded_mesh=graded_mesh, coarsen_factor=coarsen_factor,
                    mono_floor=mono_floor, mono_ceiling=mono_ceiling,
                    r_min_px=r_min_px, r_max_px=effective_r_max_px, max_iterations=max_iterations,
                )
                mode_label = "auto (residual)"
                no_position_error = (
                    "Auto-detected tip search found no usable position within the search "
                    "radius for this frame -- every candidate warped outside the deformed "
                    "frame or left too few sample points."
                )
            if found is None:
                mesh = _build_tip_mesh(nominal_tip_xy, roi_half_width_px, element_size_px, r_enrich_px,
                                        graded_mesh, coarsen_factor)
                tip_xy, tip_s = nominal_tip_xy, nominal_tip_s
                path_idx = int(np.argmin(np.abs(tip_result.path_arc_length - tip_s)))
                tangent = tangents[path_idx]
                fit = XfemDicFitResult(
                    mesh=mesh, heaviside_nodes=np.zeros(mesh.n_nodes, dtype=bool),
                    node_disp_x=np.zeros(mesh.n_nodes), node_disp_y=np.zeros(mesh.n_nodes),
                    heaviside_disp_x=np.zeros(mesh.n_nodes), heaviside_disp_y=np.zeros(mesh.n_nodes),
                    K_I_pa_sqrt_m=None, K_II_pa_sqrt_m=None, converged=False, n_iterations=0,
                    n_samples=0, residual_rms=None, residual_rms_initial=None,
                    error=no_position_error,
                )
            else:
                tip_s, fit, tip_xy, tangent, mesh = found
                last_accepted_auto_s = tip_s
                # The search above never asks any of its (potentially
                # many) candidate fits to estimate their own uncertainty
                # -- see this function's own docstring -- so, once the
                # winning position is known, refit JUST that one position
                # a second time with estimate_uncertainty=True. Skipped
                # when the winning fit itself already has no usable K_I
                # (fit.error set) -- nothing to attach an uncertainty to.
                if k_from_field and fit.error is None:
                    try:
                        fit = fit_xfem_dic_frame(
                            ref_image, def_image, mesh, tip_result.path_positions,
                            tip_result.path_arc_length, tip_xy, tip_s, tangent, kappa, mu,
                            pixel_size_mm, r_enrich_px, sample_step_px=sample_step_px,
                            initial_uv_fn=initial_uv_fn, r_min_px=r_min_px,
                            r_max_px=effective_r_max_px, max_iterations=max_iterations,
                            fit_singular_term=False,
                        )
                    except XfemDicError:
                        pass
                elif estimate_uncertainty and fit.error is None:
                    try:
                        fit = fit_xfem_dic_frame(
                            ref_image, def_image, mesh, tip_result.path_positions,
                            tip_result.path_arc_length, tip_xy, tip_s, tangent, kappa, mu,
                            pixel_size_mm, r_enrich_px, sample_step_px=sample_step_px,
                            initial_uv_fn=initial_uv_fn, r_min_px=r_min_px,
                            r_max_px=effective_r_max_px, max_iterations=max_iterations,
                            estimate_uncertainty=True,
                        )
                    except XfemDicError:
                        pass  # keep the original (uncertainty-free) winning fit
        else:
            tip_xy, tip_s = nominal_tip_xy, nominal_tip_s
            path_idx = int(np.argmin(np.abs(tip_result.path_arc_length - tip_s)))
            tangent = tangents[path_idx]
            # Same mesh geometry (including the sub-element jitter, needed
            # to avoid a degenerate node-line alignment -- see its
            # docstring) the mesh-preview UI shows before committing to a
            # run.
            mesh = _build_tip_mesh(tip_xy, roi_half_width_px, element_size_px, r_enrich_px,
                                    graded_mesh, coarsen_factor)
            try:
                fit = fit_xfem_dic_frame(
                    ref_image, def_image, mesh, tip_result.path_positions, tip_result.path_arc_length,
                    tip_xy, tip_s, tangent, kappa, mu, pixel_size_mm, r_enrich_px,
                    sample_step_px=sample_step_px, initial_uv_fn=initial_uv_fn,
                    r_min_px=r_min_px, r_max_px=effective_r_max_px, max_iterations=max_iterations,
                    estimate_uncertainty=estimate_uncertainty and not k_from_field,
                    fit_singular_term=not k_from_field,
                )
            except XfemDicError as exc:
                fit = XfemDicFitResult(
                    mesh=mesh, heaviside_nodes=np.zeros(mesh.n_nodes, dtype=bool),
                    node_disp_x=np.zeros(mesh.n_nodes), node_disp_y=np.zeros(mesh.n_nodes),
                    heaviside_disp_x=np.zeros(mesh.n_nodes), heaviside_disp_y=np.zeros(mesh.n_nodes),
                    K_I_pa_sqrt_m=None, K_II_pa_sqrt_m=None, converged=False, n_iterations=0,
                    n_samples=0, residual_rms=None, residual_rms_initial=None, error=str(exc),
                )

        if k_from_field and fit.error is None:
            if fit.singular_term_in_field:   # (auto-tip refit failed above) -> cannot use the field K
                fit.error = "K from fitted field: the field fit without singular term failed for this frame."
            else:
                fit = refit_k_from_fitted_field(
                    fit, tip_result.path_positions, tip_result.path_arc_length, tip_xy, tangent, kappa, mu,
                    pixel_size_mm, r_enrich_px, r_min_px, effective_r_max_px,
                    method=k_field_method, tip_offset_px=k_tip_offset_px)

        samples.append(XfemDicFrameSample(
            frame_order_index=i, source_frame_index=source_idx, time_s=float(tip_result.times[i]),
            tip_xy_px=tip_xy, fit=fit, tip_arc_length_s=tip_s,
            tip_source=mode_label if auto_detect_tip else "fixed", tangent_xy=tangent,
        ))
        done += 1
        if progress_cb is not None:
            progress_cb(done, total)

    default_index = None
    for i, sample in enumerate(samples):
        if sample.fit.K_I_pa_sqrt_m is not None:
            default_index = i
            break

    return XfemDicTimeSeries(
        leaf_index=tip_result.leaf_index, tip_method=tip_result.method,
        pixel_size_mm=pixel_size_mm, element_size_px=element_size_px, r_enrich_px=r_enrich_px,
        r_min_px=r_min_px, r_max_px=effective_r_max_px,
        samples=samples, default_critical_index=default_index, auto_detect_tip=auto_detect_tip,
        tip_detection_mode=tip_detection_mode, graded_mesh=graded_mesh,
        max_iterations=max_iterations, position_window_px=position_window_px,
        k_from_field=bool(k_from_field), k_field_method=k_field_method,
        k_tip_offset_px=float(k_tip_offset_px or 0.0),
    )


def export_xfem_k_csv(entries: list, path: str, pixel_size_mm: float | None = None) -> None:
    """Writes the "Results" card's K_I(distance)/K_II(distance) plot to one
    CSV, long format (one row per frame sample, a `trace_label` column
    tells traces apart) -- mirrors core/crack_k1c.py's `export_k1c_csv`.

    `entries` is the page's own `self._xfem_results` list of
    `{"label", "series", ...}` dicts (only these two keys are read here;
    extra keys such as "tip_result" are ignored), one per ticked X-FEM-DIC
    trace, exactly as built by `_on_xfem_run_clicked`/
    `_on_xfem_preview_clicked`. X-FEM-DIC is this page's ONLY K_I/K_II
    method (the standard-DIC-reference ring fit, its extended-Williams/
    T-stress variant, and the interaction integral were all removed on
    the user's own request -- see this module's and
    `page_crack_wave_analysis.py`'s own docstrings for why), so this
    export carries only the X-FEM-DIC fit's own columns. Passed straight
    through rather than re-fitted, so this never needs the dataset/frame
    source again -- every fit is already computed by the time the user
    asks to export.

    Every row carries the X-FEM-DIC fit's own K_I/K_II (plus, when
    "Estimate measurement uncertainty" was checked for that run, their
    formal 1-sigma stderr from `core.xfem_dic.fit_xfem_dic_frame`'s own
    asymptotic Gauss-Newton covariance) -- so a spreadsheet pivot/plot
    rebuilt from this file can reproduce exactly what the in-app plot
    shows, uncertainty bands included, without re-running anything.

    Each trace's own "critical" K_I -- the mean of every POSITIVE K_I
    sample in that trace (see `positive_value_mean_and_domain_bounds`;
    negative K_I has no physical meaning, unlike K_II) -- is reported in
    `K_I_critical_mean_positive_xfem_Pa_sqrt_m`, repeated identically on
    every row of that trace, for easy per-trace filtering/pivoting in a
    spreadsheet. Blank when that trace has no positive K_I sample at all.
    `is_default_critical` flags the SAME single row `_refresh_xfem_display`
    offers as this trace's default inspected position
    (`series.default_critical_index`), unaffected by that mean.

    Raises `OSError` on a write failure, left for the caller to report
    as-is."""
    import csv

    header = ["trace_label", "leaf_index", "tip_method", "tip_source", "frame_order_index",
              "source_frame_index", "time_s", "arc_length_px"]
    if pixel_size_mm:
        header.append("arc_length_mm")
    header += [
        "K_I_xfem_Pa_sqrt_m", "K_I_xfem_stderr_Pa_sqrt_m",
        "K_II_xfem_Pa_sqrt_m", "K_II_xfem_stderr_Pa_sqrt_m",
        "xfem_converged", "xfem_stalled", "xfem_residual_rms", "xfem_n_samples",
        "is_default_critical",
        "K_I_critical_mean_positive_xfem_Pa_sqrt_m",
        "K_method", "tip_positions",
    ]

    def _fmt(value, spec="{:.6g}"):
        return spec.format(value) if value is not None and np.isfinite(value) else ""

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for entry in entries:
            label = entry["label"]
            series = entry["series"]
            # A "direct" entry (core/direct_k.py: K from the measured displacement field, no X-FEM-DIC)
            # carries `series.method`; an X-FEM-DIC series has none. Same columns for both, so one CSV
            # can hold both routes side by side -- the last two columns say which is which.
            is_direct = entry.get("method") == "direct"
            k_method = f"direct: {series.method}" if is_direct else "xfem_dic"
            tip_positions = getattr(series, "smoothing", None) or (
                f"smoothed {series.position_window_px:g} px" if getattr(series, "position_window_px", None)
                else "raw tip positions")
            default_index = series.default_critical_index if series.default_critical_index is not None else -1

            distances_all = np.array([
                s.tip_arc_length_s if s.tip_arc_length_s is not None and np.isfinite(s.tip_arc_length_s)
                else np.nan
                for s in series.samples
            ])
            k1_xfem_all = np.array([
                s.fit.K_I_pa_sqrt_m if s.fit.K_I_pa_sqrt_m is not None else np.nan
                for s in series.samples
            ])
            mean_ki_xfem, _bounds = positive_value_mean_and_domain_bounds(distances_all, k1_xfem_all)
            mean_ki_xfem_str = _fmt(mean_ki_xfem)

            for i, sample in enumerate(series.samples):
                fit = sample.fit
                arc = sample.tip_arc_length_s
                arc_finite = arc is not None and np.isfinite(arc)
                row = [label, series.leaf_index, series.tip_method, sample.tip_source,
                       sample.frame_order_index, sample.source_frame_index,
                       f"{sample.time_s:.9g}", f"{arc:.6f}" if arc_finite else ""]
                if pixel_size_mm:
                    row.append(f"{arc * pixel_size_mm:.6f}" if arc_finite else "")
                row += [
                    _fmt(fit.K_I_pa_sqrt_m), _fmt(fit.K_I_stderr_pa_sqrt_m),
                    _fmt(fit.K_II_pa_sqrt_m), _fmt(fit.K_II_stderr_pa_sqrt_m),
                    ("1" if fit.converged else "0") if not is_direct else "",
                    ("1" if fit.stalled else "0") if not is_direct else "",
                    _fmt(fit.residual_rms), fit.n_samples,
                ]
                row.append("1" if i == default_index else "0")
                row.append(mean_ki_xfem_str)
                row += [k_method, tip_positions]
                writer.writerow(row)
