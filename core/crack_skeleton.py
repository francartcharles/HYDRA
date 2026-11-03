"""
Crack skeleton tracing: the pre-traced path(s) that a later stage (crack-tip
velocity tracking + K1c extraction, see core/crack_wave_analysis.py planned
for a following iteration) walks along -- one arbitrary-shaped, possibly-
branching polyline tree per crack, however many (N) the user needs.

A crack is represented as a small graph rather than a single ordered
polyline: real cracks are never straight and can branch (see
ui/crack_skeleton_picker.py, the interactive widget this module backs), so
"the path" is a tree of line segments -- one or more root-to-leaf chains
sharing a common trunk -- not a single sequence of points.

Two ways to build a CrackPath:
- Manually, point by point (`CrackPath.add_node`), driven by mouse clicks
  in the picker widget.
- Automatically (`detect_cracks`), by thresholding a strain field (the same
  MISES/exx/eyy field the user is already looking at for tracing) into a
  binary "localization band" mask, thinning it to a 1-pixel-wide skeleton,
  and walking that skeleton into a graph -- one CrackPath per connected
  component, so a single call can propose all N cracks visible in one
  frame at once.

Polyline simplification (Ramer-Douglas-Peucker) is a small, easily-audited
textbook algorithm implemented directly here, in the same "literal,
checkable code" spirit as core/gaussian_process.py. Skeleton thinning is
NOT hand-rolled the same way: a first pass here used a direct Zhang-Suen
implementation, which turned out to fully erode thin (2px-wide) diagonal
bands in testing -- a known failure mode of naive Zhang-Suen on staircase-
shaped diagonals, and diagonal crack bands are entirely expected input, not
an edge case worth risking. Thinning therefore uses scikit-image's
`skeletonize` (added as a dependency for this reason), which does not have
that failure mode.

The automatic detector is explicitly a *starting suggestion*, not a final
answer -- its output is just a normal (editable) list of CrackPath objects,
so the user can add, move, delete or branch points by hand afterward
exactly as if they had traced from scratch. Getting the topology perfectly
right (e.g. never merging two close-together fingers of the same crack, or
never splitting one crack into two components because a noisy pixel breaks
the band) is not the goal; giving a fast, reasonable head start on tracing
is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

DEFAULT_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]


@dataclass
class CrackPath:
    """One traced crack: a tree of (x, y) nodes (pixel coordinates of the
    reference/background frame used for tracing) connected by `edges`
    (pairs of node indices). No node is privileged as "the root" -- any
    node can be a branch point (degree > 2 in the underlying undirected
    tree), an endpoint/tip candidate (degree 1), or, for a single isolated
    point, degree 0.

    `initiation_index`/`end_index` are OPTIONAL, user-designated node
    indices (see ui/crack_skeleton_picker.py's "Mark crack start"/"Mark
    crack tip" buttons) marking which existing node is the crack's true
    initiation point and current tip/end -- distinct from node 0 (just
    "whichever node was placed/detected first", not necessarily where the
    crack actually started) and from `endpoints()` (every degree<=1 node,
    which for a branchy auto-detected skeleton is usually several
    candidates, not the two the user actually means). Nothing computed
    here depends on them being set; they only feed
    `shortest_path_between_points` below, so a crack traced/detected
    without ever using those buttons behaves exactly as before."""
    name: str = "Crack 1"
    color: str = "#e74c3c"
    nodes: list = field(default_factory=list)   # [(x, y), ...]
    edges: list = field(default_factory=list)   # [(i, j), ...] node index pairs
    initiation_index: int | None = None
    end_index: int | None = None

    def add_node(self, x: float, y: float, parent_index: int | None) -> int:
        """Appends a node, optionally connected to `parent_index` (an
        existing node -- None only for the very first node of a fresh
        crack, or to start a disjoint new sub-piece). Returns the new
        node's index."""
        self.nodes.append((float(x), float(y)))
        new_index = len(self.nodes) - 1
        if parent_index is not None:
            if not (0 <= parent_index < new_index):
                raise ValueError(f"parent_index {parent_index} is not an existing earlier node")
            self.edges.append((parent_index, new_index))
        return new_index

    def remove_last_node(self) -> None:
        """Undo: drops the most recently added node and any edge
        referencing it. A no-op on an empty crack. Does not renumber the
        remaining nodes (their indices are stable), so any edge not
        touching the removed node is untouched. Also un-marks
        `initiation_index`/`end_index` if either pointed at the removed
        node -- a stale marker on a node that no longer exists would
        otherwise silently break `shortest_path_between_points`."""
        if not self.nodes:
            return
        last = len(self.nodes) - 1
        self.nodes.pop()
        self.edges = [(i, j) for (i, j) in self.edges if i != last and j != last]
        if self.initiation_index == last:
            self.initiation_index = None
        if self.end_index == last:
            self.end_index = None

    def degree(self, index: int) -> int:
        return sum(1 for (i, j) in self.edges if i == index or j == index)

    def endpoints(self) -> list:
        """Leaf nodes (degree <= 1) -- candidate crack-tip locations for
        the later tip-tracking stage. A branch point is never returned
        here even if the crack currently ends there mid-trace (degree 1
        while still being drawn) -- callers should re-check after tracing
        is finished."""
        return [i for i in range(len(self.nodes)) if self.degree(i) <= 1]

    def mark_initiation(self, index: int) -> None:
        """Designates an existing node as the crack's initiation point
        (see the class docstring). Raises ValueError for an out-of-range
        index."""
        if not (0 <= index < len(self.nodes)):
            raise ValueError(f"node index {index} does not exist on this crack")
        self.initiation_index = index

    def mark_end(self, index: int) -> None:
        """Designates an existing node as the crack's end/tip point (see
        the class docstring). Raises ValueError for an out-of-range
        index."""
        if not (0 <= index < len(self.nodes)):
            raise ValueError(f"node index {index} does not exist on this crack")
        self.end_index = index

    def total_length(self) -> float:
        if len(self.nodes) < 2:
            return 0.0
        pts = np.asarray(self.nodes, dtype=float)
        total = 0.0
        for i, j in self.edges:
            total += float(np.hypot(*(pts[j] - pts[i])))
        return total

    def to_dict(self) -> dict:
        return {"name": self.name, "color": self.color,
                "nodes": [list(p) for p in self.nodes], "edges": [list(e) for e in self.edges],
                "initiation_index": self.initiation_index, "end_index": self.end_index}

    @staticmethod
    def from_dict(data: dict) -> "CrackPath":
        return CrackPath(
            name=data.get("name", "Crack"),
            color=data.get("color", "#e74c3c"),
            nodes=[tuple(p) for p in data.get("nodes", [])],
            edges=[tuple(e) for e in data.get("edges", [])],
            initiation_index=data.get("initiation_index"),
            end_index=data.get("end_index"),
        )


def new_crack(index: int) -> CrackPath:
    """A fresh, empty crack named/colored by its 0-based position in the
    crack list -- colors cycle through DEFAULT_COLORS so an arbitrary
    number N of cracks always gets *some* distinct color, just repeating
    past 8."""
    return CrackPath(name=f"Crack {index + 1}", color=DEFAULT_COLORS[index % len(DEFAULT_COLORS)])


def serialize_cracks(cracks: list) -> list:
    return [c.to_dict() for c in cracks]


def deserialize_cracks(data: list) -> list:
    return [CrackPath.from_dict(d) for d in (data or [])]


# ---------------------------------------------------------------------------
# Main-axis simplification -- collapse a branchy tree to a single trunk
# ---------------------------------------------------------------------------

def simplify_to_main_axis_path(crack: CrackPath, axis: str) -> CrackPath:
    """Collapses a traced/auto-detected crack's tree down to a single
    trunk, discarding every other branch -- for the "many small parasitic
    side-branches around the main path" a noisy auto-detected skeleton
    often has (see this module's docstring: getting the topology exactly
    right is not `detect_cracks`'s goal, and skeletonizing a somewhat
    noisy strain-localization mask routinely leaves short spurious twigs
    off the real crack).

    The kept trunk is the root (node 0, the crack's first-placed/
    initiation point)-to-leaf path whose leaf has traveled the FURTHEST
    from the root along `axis` ("x" or "y") -- i.e. whichever branch is
    actually the propagating crack along the direction it's propagating
    in, not necessarily the branch with the most nodes or the greatest
    total (possibly wandering) path length. Every node/edge not on that
    one path is dropped.

    Returns a NEW CrackPath (same name/color as `crack`, nodes renumbered
    0 = root ... N-1 = the kept leaf, along the trunk) -- `crack` itself
    is untouched, so a caller can offer this as a preview/replacement
    rather than an irreversible in-place edit.

    Raises ValueError if `crack` has fewer than 2 connected nodes (root
    has no path to any leaf -- nothing to simplify)."""
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    # Local import (not at module level) to avoid a real circular-import
    # risk: core/crack_tip_tracking.py does not import this module, but
    # `branch_paths` is generic root-to-leaf tree-chain logic that belongs
    # to neither module more than the other, and duplicating it here would
    # just be two copies of the same algorithm to keep in sync.
    from core.crack_tip_tracking import branch_paths

    paths = branch_paths(crack)
    if not paths:
        raise ValueError("This crack has fewer than 2 connected points -- nothing to simplify.")

    axis_idx = 0 if axis == "x" else 1
    root_coord = crack.nodes[0][axis_idx]

    def net_advance(leaf_index: int) -> float:
        return abs(crack.nodes[leaf_index][axis_idx] - root_coord)

    best_leaf = max(paths, key=net_advance)
    chain = paths[best_leaf]  # root-first, leaf-last, indices into `crack`'s own node list

    simplified = CrackPath(name=crack.name, color=crack.color)
    prev_new_index = None
    for old_index in chain:
        x, y = crack.nodes[old_index]
        prev_new_index = simplified.add_node(x, y, prev_new_index)
    simplified.initiation_index = 0
    simplified.end_index = len(simplified.nodes) - 1
    return simplified


# ---------------------------------------------------------------------------
# Shortest path between two user-designated points -- the "don't guess
# which branch is real, just connect the two points the user pointed at"
# alternative to axis-based simplification above.
# ---------------------------------------------------------------------------

def shortest_path_between_points(crack: CrackPath, start_index: int, end_index: int) -> CrackPath:
    """Collapses a traced/auto-detected crack's tree down to a single
    trunk, like `simplify_to_main_axis_path`, but choosing that trunk
    differently: the GEOMETRICALLY SHORTEST path through `crack`'s own
    graph connecting `start_index` to `end_index` (Dijkstra, edge weights
    = real Euclidean node-to-node distance) -- two node indices the user
    explicitly points at (see ui/crack_skeleton_picker.py's "Mark crack
    start"/"Mark crack tip" buttons, and `CrackPath.initiation_index`/
    `end_index`), not an assumption that node 0 is the root or that
    "furthest along one axis" identifies the real tip.

    This matters because `simplify_to_main_axis_path` can silently keep
    the WRONG branch and drop an entire real stretch of crack: it always
    starts from node 0 (just whichever node happened to be added/
    discovered first, not necessarily the true initiation point) and
    picks whichever leaf traveled furthest along a single axis, which is
    wrong whenever the real tip is not that leaf (e.g. the crack curves,
    or a spurious side-twig happens to reach slightly further along that
    one axis than the genuine propagating branch). Asking the user to
    point at the two ends directly removes that guesswork.

    For the ordinary case -- `crack`'s graph is a tree, as every
    auto-detected or hand-traced crack normally is -- there is exactly
    ONE path between any two nodes, so "shortest" and "passes through
    every auto-detected point on the way" are the same path: nothing on
    the real connecting branch is dropped, unlike the axis heuristic.
    (If the graph happens to have a cycle -- e.g. a genuine physical loop
    in the crack, or a rare skeletonization artifact -- this still returns
    the geometrically shortest connecting route, same as it would for a
    normal tree.)

    Returns a NEW CrackPath (same name/color as `crack`, nodes renumbered
    0 = start ... N-1 = end, along the path, with `initiation_index`/
    `end_index` set to those two positions) -- `crack` itself is
    untouched.

    Raises ValueError if `start_index`/`end_index` are not valid node
    indices on `crack`, are the same node, or lie on two disconnected
    pieces of the graph (no path exists between them)."""
    n = len(crack.nodes)
    if not (0 <= start_index < n):
        raise ValueError(f"start_index {start_index} does not exist on this crack")
    if not (0 <= end_index < n):
        raise ValueError(f"end_index {end_index} does not exist on this crack")
    if start_index == end_index:
        raise ValueError("The initiation and end point must be two different nodes.")

    import heapq

    pts = np.asarray(crack.nodes, dtype=float)
    adjacency: dict = {i: [] for i in range(n)}
    for i, j in crack.edges:
        w = float(np.hypot(*(pts[j] - pts[i])))
        adjacency[i].append((j, w))
        adjacency[j].append((i, w))

    dist = {start_index: 0.0}
    prev = {}
    visited = set()
    heap = [(0.0, start_index)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == end_index:
            break
        for v, w in adjacency[u]:
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if end_index not in dist:
        raise ValueError(
            "No connected path between the chosen initiation and end points -- "
            "they are on two disconnected pieces of this crack's traced skeleton."
        )

    chain = [end_index]
    while chain[-1] != start_index:
        chain.append(prev[chain[-1]])
    chain.reverse()  # start-first, end-last

    simplified = CrackPath(name=crack.name, color=crack.color)
    prev_new_index = None
    for old_index in chain:
        x, y = crack.nodes[old_index]
        prev_new_index = simplified.add_node(x, y, prev_new_index)
    simplified.initiation_index = 0
    simplified.end_index = len(simplified.nodes) - 1
    return simplified


# ---------------------------------------------------------------------------
# Skeleton thinning
# ---------------------------------------------------------------------------

def _skeletonize_mask(binary: np.ndarray) -> np.ndarray:
    """Thins a boolean mask to a 1-pixel-wide skeleton. See the module
    docstring for why this delegates to scikit-image rather than a
    hand-rolled Zhang-Suen pass."""
    from skimage.morphology import skeletonize
    return skeletonize(binary)


# ---------------------------------------------------------------------------
# Polyline simplification (Ramer-Douglas-Peucker)
# ---------------------------------------------------------------------------

def _point_segment_distance(p, a, b) -> float:
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.allclose(a, b):
        return float(np.hypot(*(p - a)))
    t = float(np.dot(p - a, b - a) / np.dot(b - a, b - a))
    t = min(1.0, max(0.0, t))
    proj = a + t * (b - a)
    return float(np.hypot(*(p - proj)))


def _rdp(points: list, epsilon: float) -> list:
    """Ramer-Douglas-Peucker simplification of a polyline (list of (r, c)
    or (x, y) tuples), keeping only the points needed so no discarded
    point strays more than `epsilon` from the simplified line -- turns a
    dense pixel-by-pixel skeleton chain into a handful of shape-defining
    nodes."""
    if len(points) < 3:
        return list(points)
    start, end = points[0], points[-1]
    dmax, index = 0.0, 0
    for i in range(1, len(points) - 1):
        d = _point_segment_distance(points[i], start, end)
        if d > dmax:
            dmax, index = d, i
    if dmax > epsilon:
        left = _rdp(points[:index + 1], epsilon)
        right = _rdp(points[index:], epsilon)
        return left[:-1] + right
    return [start, end]


# ---------------------------------------------------------------------------
# Skeleton -> graph
# ---------------------------------------------------------------------------

_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _skeleton_neighbors(pixel_set: set, r: int, c: int) -> list:
    return [(r + dr, c + dc) for dr, dc in _NEIGHBOR_OFFSETS if (r + dr, c + dc) in pixel_set]


def _extract_chains(component_pixels: set) -> list:
    """Walks one connected skeleton component into a list of pixel chains
    ((r, c) tuples), each chain running between two "nodes of interest"
    (an endpoint, degree <= 1, or a junction, degree >= 3) through a
    string of plain degree-2 pixels. Returns `[]` for a component with
    fewer than 2 pixels (nothing to trace)."""
    if len(component_pixels) < 2:
        return []

    degree = {p: len(_skeleton_neighbors(component_pixels, *p)) for p in component_pixels}
    nodes_of_interest = [p for p, d in degree.items() if d != 2]

    closed_loop = False
    if not nodes_of_interest:
        # A simple closed ring (every pixel has exactly 2 neighbors) --
        # rare for a crack, but break the symmetry with one synthetic
        # node so the walk below has somewhere to start and stop.
        nodes_of_interest = [min(component_pixels)]
        closed_loop = True

    visited_directed = set()  # (from_pixel, to_pixel) already walked
    chains = []
    for start in nodes_of_interest:
        for first in _skeleton_neighbors(component_pixels, *start):
            if (start, first) in visited_directed:
                continue
            path = [start, first]
            visited_directed.add((start, first))
            prev, cur = start, first
            while True:
                if cur in nodes_of_interest and cur != start:
                    break
                if closed_loop and cur == start and len(path) > 2:
                    break
                nbrs = [p for p in _skeleton_neighbors(component_pixels, *cur) if p != prev]
                if len(nbrs) != 1:
                    # Reached a junction pixel not flagged as such (shouldn't
                    # normally happen since junctions have degree >= 3 and
                    # are in nodes_of_interest), or a dead end -- stop here
                    # rather than loop forever.
                    break
                nxt = nbrs[0]
                visited_directed.add((cur, nxt))
                path.append(nxt)
                prev, cur = cur, nxt
            chains.append(path)
    return chains


def _pixels_to_crack(chains: list, xs: np.ndarray, ys: np.ndarray, simplify_tol_cells: float,
                      name: str, color: str) -> "CrackPath":
    """Builds one CrackPath from the raw pixel chains of a single skeleton
    component: simplifies each chain (RDP), then stitches the simplified
    chains together into a graph, sharing one node per junction/endpoint
    pixel across every chain that touches it, and converts grid-index
    positions (row, col) into real pixel coordinates via `xs`/`ys` (the
    same regular grid the strain field itself was sampled on)."""
    crack = CrackPath(name=name, color=color)
    node_index_of_pixel = {}

    def _node_for(pixel):
        if pixel not in node_index_of_pixel:
            r, c = pixel
            x = float(xs[min(max(c, 0), len(xs) - 1)])
            y = float(ys[min(max(r, 0), len(ys) - 1)])
            node_index_of_pixel[pixel] = crack.add_node(x, y, parent_index=None)
        return node_index_of_pixel[pixel]

    for chain in chains:
        simplified = _rdp(chain, simplify_tol_cells)
        prev_index = _node_for(simplified[0])
        for pixel in simplified[1:]:
            idx = node_index_of_pixel.get(pixel)
            if idx is None:
                r, c = pixel
                x = float(xs[min(max(c, 0), len(xs) - 1)])
                y = float(ys[min(max(r, 0), len(ys) - 1)])
                idx = crack.add_node(x, y, parent_index=prev_index)
                node_index_of_pixel[pixel] = idx
            else:
                if (prev_index, idx) not in crack.edges and (idx, prev_index) not in crack.edges:
                    crack.edges.append((prev_index, idx))
            prev_index = idx
    return crack


def detect_cracks(values_grid: np.ndarray, xs: np.ndarray, ys: np.ndarray, threshold: float,
                   min_length_px: float = 15.0, simplify_tol_cells: float = 1.5) -> list:
    """Automatically proposes crack path(s) from a strain field: everywhere
    `values_grid` (shape (len(ys), len(xs)), NaN allowed outside the ROI --
    same layout as a DICFrameResult field) is at or above `threshold` is
    treated as the localized band a crack has torn through; that mask is
    thinned to a skeleton and each connected piece becomes one CrackPath,
    named/colored like a manually-started crack (`new_crack`) so the two
    are indistinguishable once in the picker's crack list -- this is a
    starting suggestion, not a separate kind of object.

    `min_length_px` drops components too short to plausibly be a crack
    (isolated noisy pixels above threshold rather than a real band) --
    length is measured in the same real (pixel) units as `xs`/`ys`, after
    conversion, not in grid cells. `simplify_tol_cells` is the
    Ramer-Douglas-Peucker tolerance, in grid cells (multiply by the grid
    spacing for a real-pixel tolerance) -- higher gives fewer, straighter-
    looking nodes.

    Returns `[]` if nothing in `values_grid` reaches `threshold`."""
    from scipy import ndimage

    values_grid = np.asarray(values_grid, dtype=float)
    binary = np.where(np.isnan(values_grid), False, values_grid >= threshold)
    if not np.any(binary):
        return []

    skeleton = _skeletonize_mask(binary)
    if not np.any(skeleton):
        return []

    labeled, n_components = ndimage.label(skeleton, structure=np.ones((3, 3), dtype=int))

    cracks = []
    for label_id in range(1, n_components + 1):
        rows, cols = np.nonzero(labeled == label_id)
        pixel_set = set(zip(rows.tolist(), cols.tolist()))
        chains = _extract_chains(pixel_set)
        if not chains:
            continue
        placeholder = new_crack(len(cracks))
        crack = _pixels_to_crack(chains, xs, ys, simplify_tol_cells,
                                  name=f"{placeholder.name} (auto)", color=placeholder.color)
        if crack.total_length() >= min_length_px and len(crack.nodes) >= 2:
            cracks.append(crack)
    return cracks


# ---------------------------------------------------------------------------
# Ridge-based detector (handles noisy fields and branching cracks)
# ---------------------------------------------------------------------------
#
# Why a second detector. `detect_cracks` above thresholds the field itself.
# On a real DIC strain map that goes wrong in three ways: (1) the speckle
# noise is as strong as the weak branches, so a threshold low enough to keep
# a branch also lets noise blobs in, and one high enough to reject the noise
# cuts the branches (and breaks the band into pieces); (2) the thinned mask of
# a thick, ragged band sprouts many short spurs, which read as dozens of fake
# branches; (3) a band that is several cells wide at its trunk and one cell
# wide at its tips has no single "right" threshold. A crack is a RIDGE (a thin
# line brighter than both its sides), so this detector filters for ridges at a
# few scales (largest negative eigenvalue of the Hessian of the Gaussian-
# smoothed field -- strong on a line of any width, weak on a noise blob or a
# smooth gradient), keeps strong ridge pixels plus the weaker ones connected to
# them (hysteresis -- that is what carries the detection out to the fading
# branch tips), thins, prunes short spurs, and merges the clusters of junction
# pixels the thinning leaves into single branch points.

_MIN_RIDGE_CONTRAST = 1.6


def _ridge_response(values_grid: np.ndarray, sigmas_cells) -> tuple:
    """(ridge strength >= 0, smoothed field background-corrected), both (ny,
    nx). NaN points (outside the ROI) are filled with the median so they
    create no edge response, then forced to 0 / NaN."""
    from scipy import ndimage
    a = np.asarray(values_grid, dtype=float)
    valid = np.isfinite(a)
    fill = float(np.median(a[valid])) if valid.any() else 0.0
    work = np.where(valid, a, fill)
    best = np.zeros_like(work)
    for s in sigmas_cells:
        hxx = ndimage.gaussian_filter(work, s, order=(0, 2), mode="nearest")
        hyy = ndimage.gaussian_filter(work, s, order=(2, 0), mode="nearest")
        hxy = ndimage.gaussian_filter(work, s, order=(1, 1), mode="nearest")
        lam_min = 0.5 * (hxx + hyy) - np.sqrt((0.5 * (hxx - hyy)) ** 2 + hxy ** 2)
        best = np.maximum(best, np.maximum(0.0, -lam_min) * s * s)   # scale-normalized
    # Do not trust the response right next to a hole / the ROI edge
    near_hole = ndimage.binary_dilation(~valid, iterations=1) if (~valid).any() else ~valid
    best[near_hole] = 0.0
    smooth = ndimage.gaussian_filter(work, 1.0, mode="nearest")
    smooth[~valid] = np.nan
    return best, smooth


def _remove_corner_pixels(pixels: set) -> set:
    """Thinning leaves staircase corners where a pixel's only two neighbors
    are themselves adjacent: a 3-pixel triangle, which the graph walk reads as
    a tiny loop with a fake branch point. Removing such a pixel keeps its two
    neighbors connected, so the line is unchanged -- minus the loop."""
    pixels = set(pixels)
    changed = True
    while changed:
        changed = False
        for p in sorted(pixels):
            nb = _skeleton_neighbors(pixels, *p)
            if len(nb) == 2 and max(abs(nb[0][0] - nb[1][0]), abs(nb[0][1] - nb[1][1])) <= 1:
                pixels.discard(p)
                changed = True
    return pixels


def _prune_spurs(pixels: set, min_len_pixels: int, max_iter: int = 8, strength=None,
                  weak_len_pixels: int = 0, weak_below: float = 0.0) -> set:
    """Removes terminal branches (leaf -> junction) shorter than
    `min_len_pixels` skeleton pixels, repeatedly (removing a spur can turn
    its junction into a plain point and expose a new, shorter spur). With
    `strength` (a grid of ridge strengths) a terminal branch up to
    `weak_len_pixels` long is ALSO removed when its mean strength is under
    `weak_below` -- a faint twig hanging off a crack is noise, a faint but
    LONG one may be a real fading branch. A
    junction whose arms are ALL short spurs keeps its longest one, so a
    small star-shaped piece is shortened, not erased."""
    pixels = set(pixels)
    for _ in range(max_iter):
        deg = {p: len(_skeleton_neighbors(pixels, *p)) for p in pixels}
        chains = _extract_chains(pixels)
        spurs = {}   # junction pixel -> [(length, removable pixels)]
        arms = {}    # junction pixel -> number of chains touching it
        for ch in chains:
            for end in (ch[0], ch[-1]):
                if deg[end] >= 3:
                    arms[end] = arms.get(end, 0) + 1
            a, b = ch[0], ch[-1]
            if deg[a] == 1 and deg[b] >= 3:
                spurs.setdefault(b, []).append((len(ch), ch[:-1]))
            elif deg[b] == 1 and deg[a] >= 3:
                spurs.setdefault(a, []).append((len(ch), ch[1:]))
        remove = set()
        for junction, items in spurs.items():
            def _is_spur(it):
                if it[0] < min_len_pixels:
                    return True
                if strength is not None and it[0] < weak_len_pixels:
                    return float(np.mean([strength[r, c] for r, c in it[1]])) < weak_below
                return False
            short = [it for it in items if _is_spur(it)]
            if not short:
                continue
            if len(short) == arms.get(junction, 0):       # every arm is a short spur
                short = sorted(short, key=lambda it: it[0])[:-1]   # keep the longest
            for _, px in short:
                remove.update(px)
        if not remove:
            break
        pixels -= remove
    return pixels


def _collapse_short_cycles(crack: "CrackPath", tol_px: float) -> "CrackPath":
    """Contracts every SHORT edge (<= `tol_px`) that lies on a cycle -- the
    tiny loops a thinned T-junction or a staircase corner leaves, which are
    not cracks (a real crack tree has no loops) -- merging its two nodes at
    their midpoint. Repeats until no such edge is left."""
    nodes = [tuple(p) for p in crack.nodes]
    edges = {(min(i, j), max(i, j)) for i, j in crack.edges if i != j}
    alive = set(range(len(nodes)))

    def connected_without(a, b):
        adj = {}
        for i, j in edges:
            if (i, j) == (min(a, b), max(a, b)):
                continue
            adj.setdefault(i, []).append(j); adj.setdefault(j, []).append(i)
        seen, stack = {a}, [a]
        while stack:
            u = stack.pop()
            if u == b:
                return True
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v); stack.append(v)
        return False

    changed = True
    while changed:
        changed = False
        for i, j in sorted(edges, key=lambda e: np.hypot(nodes[e[0]][0] - nodes[e[1]][0], nodes[e[0]][1] - nodes[e[1]][1])):
            if np.hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1]) > tol_px:
                break
            if connected_without(i, j):
                nodes[i] = ((nodes[i][0] + nodes[j][0]) / 2.0, (nodes[i][1] + nodes[j][1]) / 2.0)
                edges = {(min(i if a == j else a, i if b == j else b), max(i if a == j else a, i if b == j else b))
                         for a, b in edges}
                edges = {e for e in edges if e[0] != e[1]}
                alive.discard(j)
                changed = True
                break
    order = sorted(alive)
    new_of = {old: new for new, old in enumerate(order)}
    return CrackPath(name=crack.name, color=crack.color, nodes=[nodes[o] for o in order],
                     edges=[(new_of[a], new_of[b]) for a, b in sorted(edges)])


def _break_cycles(crack: "CrackPath") -> "CrackPath":
    """A crack is a tree. Any loop that survived the clean-up (e.g. a band
    thinned around a small island) is opened by dropping the LONGEST edge of
    each cycle (Kruskal minimum spanning forest on edge length)."""
    pts = np.asarray(crack.nodes, dtype=float)
    parent = list(range(len(pts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    keep = []
    for i, j in sorted(crack.edges, key=lambda e: float(np.hypot(*(pts[e[0]] - pts[e[1]])))):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            keep.append((i, j))
    if len(keep) == len(crack.edges):
        return crack
    return CrackPath(name=crack.name, color=crack.color, nodes=list(crack.nodes), edges=sorted(keep))


def _merge_junction_clusters(crack: "CrackPath", tol_px: float) -> "CrackPath":
    """Thinning leaves a branch point as a small cluster of adjacent junction
    pixels, which become several close-together graph nodes joined by tiny
    edges (fake loops). Collapses every group of branch nodes (degree >= 3)
    lying within `tol_px` of each other (transitively) into ONE node at their
    centroid, and drops the self-loops / duplicate edges that creates."""
    n = len(crack.nodes)
    if n == 0:
        return crack
    deg = [crack.degree(i) for i in range(n)]
    pts = np.asarray(crack.nodes, dtype=float)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    branch = [i for i in range(n) if deg[i] >= 3]
    for a in range(len(branch)):
        for b in range(a + 1, len(branch)):
            i, j = branch[a], branch[b]
            if np.hypot(*(pts[i] - pts[j])) <= tol_px:
                parent[find(i)] = find(j)
    # also absorb a degree-2 node sitting between two merged junction nodes
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    new_index, new_nodes = {}, []
    for root, members in groups.items():
        big = [m for m in members if deg[m] >= 3]
        if len(members) > 1 and big:
            c = pts[big].mean(axis=0)
            idx = len(new_nodes); new_nodes.append((float(c[0]), float(c[1])))
            for m in big:
                new_index[m] = idx
        for m in members:
            if m not in new_index:
                new_index[m] = len(new_nodes)
                new_nodes.append(tuple(pts[m]))
    edges, seen = [], set()
    for i, j in crack.edges:
        a, b = new_index[i], new_index[j]
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key); edges.append((a, b))
    out = CrackPath(name=crack.name, color=crack.color, nodes=new_nodes, edges=edges)
    return out


def _root_at_origin_leaf(crack: "CrackPath", xs: np.ndarray, ys: np.ndarray, smooth: np.ndarray | None = None,
                          cell: float = 1.0) -> "CrackPath":
    """Renumbers `crack` so node 0 -- which the tip-tracking stage treats as
    the root its root-to-leaf paths start from, i.e. where the crack ORIGINATES
    -- is a LEAF of the tree, chosen as the most likely origin: the end around
    which the strain field is strongest (a propagating crack has opened most
    where it started and tapers towards its tips). Without a field, the end of
    the tree's longest path closest to the edge of the data. Pure renumbering:
    same nodes, same edges."""
    n = len(crack.nodes)
    if n < 3:
        return crack
    pts = np.asarray(crack.nodes, dtype=float)
    adj = {i: [] for i in range(n)}
    for i, j in crack.edges:
        w = float(np.hypot(*(pts[i] - pts[j])))
        adj[i].append((j, w)); adj[j].append((i, w))
    leaves = [i for i in range(n) if len(adj[i]) == 1]
    if not leaves:
        return crack

    root = None
    if smooth is not None and len(leaves) >= 2:
        gx, gy = np.arange(len(xs)), np.arange(len(ys))
        score = {}
        for leaf in leaves:
            c = float(np.interp(pts[leaf, 0], xs, gx)); r = float(np.interp(pts[leaf, 1], ys, gy))
            reach = max(2.0, 25.0 / cell)                   # ~25 px around the end
            r0, r1 = int(max(0, np.floor(r - reach))), int(min(smooth.shape[0], np.ceil(r + reach) + 1))
            c0, c1 = int(max(0, np.floor(c - reach))), int(min(smooth.shape[1], np.ceil(c + reach) + 1))
            patch = smooth[r0:r1, c0:c1]
            rr, cc = np.mgrid[r0:r1, c0:c1]
            inside = (rr - r) ** 2 + (cc - c) ** 2 <= reach ** 2
            vals = patch[inside & np.isfinite(patch)]
            score[leaf] = float(vals.mean()) if vals.size else -np.inf
        best = max(score, key=score.get)
        if np.isfinite(score[best]):
            root = best
    if root is None:
        def farthest(src):
            dist, stack = {src: 0.0}, [src]
            while stack:
                u = stack.pop()
                for v, w in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + w
                        stack.append(v)
            return max(dist, key=dist.get)
        a_, b_ = farthest(0), None
        b_ = farthest(a_)

        def edge_gap(i):
            x, y = pts[i]
            return min(x - xs.min(), xs.max() - x, y - ys.min(), ys.max() - y)
        root = a_ if edge_gap(a_) <= edge_gap(b_) else b_
    order, seen, stack = [], set(), [root]
    while stack:                       # depth-first from the root: parents before children
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u); order.append(u)
        stack.extend(v for v, _ in reversed(adj[u]) if v not in seen)
    order += [i for i in range(n) if i not in seen]     # (disconnected leftovers, if any)
    new_of = {old: new for new, old in enumerate(order)}
    return CrackPath(name=crack.name, color=crack.color,
                     nodes=[crack.nodes[o] for o in order],
                     edges=[(new_of[i], new_of[j]) for i, j in crack.edges])


def _polyline_length(pts: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(pts, axis=0).T))) if len(pts) > 1 else 0.0


def _resample_polyline(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Points evenly spaced (about `spacing`) along the polyline's arc length,
    always including both ends."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 0:
        return pts[[0, -1]]
    n = max(1, int(round(total / max(spacing, 1e-6))))
    t = np.linspace(0.0, total, n + 1)
    return np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])])


def _smooth_polyline(pts: np.ndarray, sigma_px: float, step_px: float) -> np.ndarray:
    """Gaussian smoothing along the arc length (the staircase left by a
    pixel-grid skeleton disappears), with both ends pinned exactly where
    they were. Returned at `step_px` sampling."""
    from scipy import ndimage
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3 or _polyline_length(pts) < 2 * step_px:
        return _resample_polyline(pts, step_px)
    fine = _resample_polyline(pts, step_px)
    sm = np.column_stack([ndimage.gaussian_filter1d(fine[:, k], max(sigma_px / step_px, 0.5), mode="nearest")
                          for k in (0, 1)])
    ramp = np.linspace(0.0, 1.0, len(sm))[:, None]
    sm += (1 - ramp) * (fine[0] - sm[0]) + ramp * (fine[-1] - sm[-1])
    return sm


def _snap_to_ridge(pts: np.ndarray, ridge: np.ndarray, xs: np.ndarray, ys: np.ndarray, reach_px: float,
                    step_px: float) -> np.ndarray:
    """Moves every INTERIOR point along its local normal to the strongest
    ridge response within +-`reach_px` (sub-cell centring on the crack: the
    skeleton itself can only sit on whole grid cells). Ends stay put."""
    from scipy import ndimage
    if len(pts) < 3:
        return pts
    tang = np.gradient(pts, axis=0)
    norm = np.hypot(tang[:, 0], tang[:, 1])
    norm[norm == 0] = 1.0
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]]) / norm[:, None]
    offs = np.arange(-reach_px, reach_px + 1e-9, step_px)
    gx = np.arange(len(xs)); gy = np.arange(len(ys))
    out = pts.copy()
    for i in range(1, len(pts) - 1):
        cand = pts[i] + offs[:, None] * nrm[i]
        ci = np.interp(cand[:, 0], xs, gx); ri = np.interp(cand[:, 1], ys, gy)
        vals = ndimage.map_coordinates(ridge, [ri, ci], order=1, mode="nearest")
        if not np.any(vals > 0):
            continue
        out[i] = cand[int(np.argmax(vals))]
    return out


def _arm_fit(arm: np.ndarray, d1: float, d2: float):
    """(point, unit direction away from the junction) of the straight line
    fitted (PCA) to the part of an arm lying between arc distances d1..d2 from
    the junction (arm[0]); None when the arm has too little there."""
    seg = np.hypot(*np.diff(arm, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    sel = (arc >= d1) & (arc <= d2)
    if sel.sum() < 3 or arc[sel][-1] - arc[sel][0] < 0.5 * (d2 - d1):
        return None
    pts = arm[sel]
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    u = vt[0]
    if np.dot(u, pts[-1] - pts[0]) < 0:
        u = -u
    return c, u


def _arm_fit_straight(arm: np.ndarray, d_min: float = 10.0, win: float = 25.0, step: float = 3.0,
                      rms_max: float = 0.8, ang_max_deg: float = 3.0, d_fallback=(12.0, 40.0)):
    """Like `_arm_fit`, but the window is placed where the arm becomes
    STRAIGHT. Near a thick parent band the thinned branch starts with a
    "hook" (the medial axis bends towards the parent over 20-30 px) whose
    tangent says nothing about where the branch comes from; its straight
    run further out does. Slides a `win`-px window along the arm from arc
    `d_min`, and keeps the first one that is straight (PCA residual RMS <=
    `rms_max` px) and whose direction agrees (<= `ang_max_deg`) with the next
    window one `3*step` px further. Falls back to the fixed `d_fallback` window
    when no such run exists. Returns (point, unit direction away from the
    junction, arc distance where the fitted part starts) or None."""
    seg = np.hypot(*np.diff(arm, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])

    def fit(d1, d2):
        sel = (arc >= d1) & (arc <= d2)
        if sel.sum() < 3 or arc[sel][-1] - arc[sel][0] < 0.6 * (d2 - d1):
            return None
        pts = arm[sel]
        c = pts.mean(axis=0)
        _, sv, vt = np.linalg.svd(pts - c)
        u = vt[0]
        if np.dot(u, pts[-1] - pts[0]) < 0:
            u = -u
        rms = float(sv[1] / np.sqrt(len(pts))) if len(sv) > 1 else 0.0
        return c, u, rms

    d = d_min
    while d + win <= total + 1e-9:
        a = fit(d, d + win)
        if a is not None and a[2] <= rms_max:
            b = fit(d + 3 * step, d + 3 * step + win)
            if b is None:
                if d + 3 * step + win > total:      # arm ends: nothing further to compare with
                    return a[0], a[1], d
            else:
                ang = np.degrees(np.arccos(np.clip(float(np.dot(a[1], b[1])), -1.0, 1.0)))
                if ang <= ang_max_deg:
                    return a[0], a[1], d
        d += step
    r = _arm_fit(arm, *d_fallback)
    return None if r is None else (r[0], r[1], d_fallback[0])


_APEX_FRACTION = 0.5     # see _split_apex


def _field_sampler(field):
    """f(x, y) -> smoothed strain at pixel coordinates, from (xs, ys, grid)."""
    from scipy.interpolate import RegularGridInterpolator
    xs, ys, grid = field
    g = np.nan_to_num(np.asarray(grid, float), nan=0.0)
    interp = RegularGridInterpolator((np.asarray(ys, float), np.asarray(xs, float)), g,
                                     bounds_error=False, fill_value=0.0)
    return lambda x, y: interp(np.column_stack([np.ravel(y), np.ravel(x)]))


def _split_apex(sample, M, b, s_down: float = 12.0, s_up: float = 45.0, half_w: float = 32.0,
                step: float = 2.0):
    """Where does a Y-shaped fork REALLY begin? Two daughter bands that
    leave their parent at a low angle overlap near the fork: the thinned
    skeleton only splits where the bands are already well apart. Looks at the
    strain profile ACROSS the fork (along the normal of the bisector `b`) at
    stations walking upstream from the skeleton's fork `M`: the profile is
    two-peaked while the daughters are distinguishable and single-peaked
    once they merge. The peak-to-peak separation shrinks about linearly
    towards the real apex; its zero crossing is returned as (x, y), together
    with the last two-peaked station. None when no clean two-peaked run
    exists."""
    from scipy.signal import find_peaks
    from scipy import ndimage
    b = np.asarray(b, float) / (np.hypot(*b) or 1.0)
    n = np.array([-b[1], b[0]])
    offs = np.arange(-half_w, half_w + 1e-9, 1.0)
    rows = []                                   # (t, separation, centre offset)
    t = s_down
    gap = 0
    while t >= -s_up:
        P = np.asarray(M, float) + t * b
        prof = sample(P[0] + offs * n[0], P[1] + offs * n[1])
        prof = ndimage.gaussian_filter1d(prof, 1.5)
        top = float(prof.max())
        pk, props = find_peaks(prof, prominence=0.12 * max(top, 1e-12))
        ok = False
        if len(pk) >= 2:
            order = np.argsort(prof[pk])[::-1][:2]
            i1, i2 = sorted(pk[order])
            dip = float(prof[i1:i2 + 1].min())
            lo = float(min(prof[i1], prof[i2]))
            if dip <= 0.9 * lo and (i2 - i1) >= 4:
                rows.append((t, float(offs[i2] - offs[i1]), 0.5 * float(offs[i1] + offs[i2])))
                ok = True
        if ok:
            gap = 0
        elif rows:
            gap += 1
            if gap >= 2:                         # merged: stop
                break
        t -= step
    if len(rows) < 4:
        return None
    arr = np.array(rows)                         # downstream -> upstream
    near = arr[-6:]                              # the stations closest to the split
    if np.ptp(near[:, 0]) < 4.0:
        return None
    slope, icpt = np.polyfit(near[:, 0], near[:, 1], 1)     # separation = slope * t + icpt
    if slope <= 0.05:
        return None
    t_lin = -icpt / slope                        # where the separation would reach zero
    # Two overlapping bands only resolve at a separation of about 2 sigma, and
    # the eye puts the fork where the double band begins -- between that
    # last resolvable station and the zero crossing. Half-way matches
    # hand-marked forks best.
    t_split = float(arr[-1, 0])
    t0 = t_split + _APEX_FRACTION * (t_lin - t_split)
    t0 = min(max(t0, t_split - 30.0), t_split + 2.0)
    c = float(np.mean(arr[-4:, 2]))
    apex = np.asarray(M, float) + t0 * b + c * n
    return apex, float(arr[-1, 0])


def _relocate_y_fork(geo, node_xy, root, arms, dirs, sample, spacing_px, xroot) -> bool:
    """Y-fork branch: see `_relocate_branch_points`. True when the fork was moved."""
    # trunk = the arm most opposite to the bisector of the other two
    best = None
    for k in range(3):
        o = [j for j in range(3) if j != k]
        bis = dirs[o[0]] + dirs[o[1]]
        nb = float(np.hypot(*bis))
        if nb < 1e-6:
            continue
        score = float(np.dot(dirs[k], bis / nb))
        if best is None or score < best[0]:
            best = (score, k, o, bis / nb)
    if best is None:
        return False
    score, kt, (k1, k2), b = best
    cosang = float(np.clip(np.dot(dirs[k1], dirs[k2]), -1.0, 1.0))
    ang = float(np.degrees(np.arccos(cosang)))
    if score > -0.75 or not (12.0 <= ang <= 100.0):
        return False
    # the two daughters must be about equally strong (else it is a T: thin branch on a thick parent)
    def strength(poly):
        seg = np.hypot(*np.diff(poly, axis=0).T)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        sel = (arc >= 8.0) & (arc <= 40.0)
        if sel.sum() < 2:
            return 0.0
        return float(np.mean(sample(poly[sel, 0], poly[sel, 1])))
    s1, s2 = strength(arms[k1][1]), strength(arms[k2][1])
    if min(s1, s2) < 0.7 * max(s1, s2) or max(s1, s2) <= 0:
        return False
    M = np.asarray(arms[kt][1][0], float)
    res = _split_apex(sample, M, b)
    if res is None:
        return False
    X, _t_split = res
    # the apex must lie on the trunk side: cut the trunk where it passes X
    pt = arms[kt][1]
    d = np.hypot(pt[:, 0] - X[0], pt[:, 1] - X[1])
    kx = int(np.argmin(d))
    if kx < 1 or d[kx] > 12.0:
        return False
    node_xy[xroot] = (float(X[0]), float(X[1]))
    idx_t, idx_1, idx_2 = arms[kt][0], arms[k1][0], arms[k2][0]
    far_t, far_1, far_2 = arms[kt][2], arms[k1][2], arms[k2][2]
    # trunk: X -> along the old trunk away from the fork -> far end
    trunk = np.vstack([X[None, :], pt[kx:]])
    geo[idx_t] = {"ra": xroot, "rb": far_t, "poly": trunk}
    for idx, k, far in ((idx_1, k1, far_1), (idx_2, k2, far_2)):
        poly = np.vstack([X[None, :], arms[k][1]])
        poly = _smooth_polyline(poly, 1.0 * spacing_px, max(spacing_px / 2.0, 1.0))
        poly = _resample_polyline(poly, spacing_px)
        geo[idx] = {"ra": xroot, "rb": far, "poly": poly}
    node_xy.pop(root, None)
    return True


def _relocate_branch_points(geo: list, node_xy: dict, spacing_px: float, d1: float = 12.0, d2: float = 40.0,
                             s_max: float = 50.0, field=None) -> None:
    """Moves every 3-arm branch point to where the branch REALLY leaves its
    parent. Thinning puts a fork where the two bands have separated far enough
    for the medial axis to split, i.e. downstream of the real branch point --
    by about half the parent band's width / tan(fork angle), 10-20 px on a
    typical crack. The geometric branch point is where the branch's own centre
    line, prolonged BACKWARDS from the part where it is clearly separate
    (arc distance d1..d2 from the old fork), meets the through path. At each
    3-arm fork the two most collinear arms form the through path; the third is
    the branch; the new fork is that intersection, searched up to `s_max`
    behind the old one (on the side the branch is NOT heading to). The branch
    is re-started from it (a straight run to the fitted part), the through
    path is kept intact. `geo` is a list of {"ra","rb","poly"} (poly runs ra
    -> rb); modified in place, `node_xy` gets the new fork nodes.

    Forks whose two daughters are about equally strong and leave the trunk at
    a modest angle are Y forks, not T forks: there the daughters' straight
    runs lie too far out (they curve) -- the apex is read from the strain
    profile across the fork instead (`_split_apex`; needs `field` =
    (xs, ys, smoothed grid)); T forks keep the line-intersection rule."""
    counter = 0
    sample = _field_sampler(field) if field is not None else None
    for root in list(node_xy):
        incident = [i for i, g in enumerate(geo) if g["ra"] == root or g["rb"] == root]
        if len(incident) != 3:
            continue
        arms = []                                   # (chain index, poly oriented root -> outward, far root)
        for i in incident:
            g = geo[i]
            if g["ra"] == root:
                arms.append((i, g["poly"], g["rb"]))
            else:
                arms.append((i, g["poly"][::-1], g["ra"]))
        fits3 = [_arm_fit_straight(p, d_min=d1 - 2.0, d_fallback=(d1, d2)) for _, p, _ in arms]
        fits = [None if f3 is None else (f3[0], f3[1]) for f3 in fits3]
        starts = [None if f3 is None else f3[2] for f3 in fits3]
        dirs = []
        for (_, p, _), fit in zip(arms, fits):
            if fit is not None:
                v = fit[0] - p[0]
            else:
                v = p[min(len(p) - 1, 3)] - p[0]
            m = float(np.hypot(*v))
            dirs.append(v / m if m > 0 else v)
        if sample is not None and _relocate_y_fork(geo, node_xy, root, arms, dirs, sample, spacing_px,
                                                   ("fork", counter + 1)):
            counter += 1
            continue
        # through pair = the two most opposite arms
        pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
        ia, ib, ic = min(pairs, key=lambda t: float(np.dot(dirs[t[0]], dirs[t[1]])))
        if fits[ic] is None:
            continue
        q, u = fits[ic]
        # a = the through arm the branch is NOT heading along (the base is on its side)
        if np.dot(u, dirs[ia]) > np.dot(u, dirs[ib]):
            ia, ib = ib, ia
        pa = arms[ia][1]
        seg = np.hypot(*np.diff(pa, axis=0).T)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        limit = min(s_max, float(arc[-1]) - 0.5 * spacing_px)
        best = None
        for k in range(len(pa) - 1):
            if arc[k] > limit:
                break
            a0, a1 = pa[k], pa[k + 1]
            e = a1 - a0
            den = u[0] * e[1] - u[1] * e[0]
            if abs(den) < 1e-9:
                continue
            w = a0 - q
            t_line = (w[0] * e[1] - w[1] * e[0]) / den          # along the branch line
            t_seg = (w[0] * u[1] - w[1] * u[0]) / den           # along the segment, 0..1
            if -1e-9 <= t_seg <= 1 + 1e-9:
                sx = float(arc[k] + t_seg * seg[k])
                if sx <= limit and (best is None or sx < best[0]):
                    best = (sx, q + t_line * u, k, t_seg)
        if best is None or best[0] < 0.5 * spacing_px:
            continue
        sx, X, k, t_seg = best
        # build the three new chains around the new fork X
        counter += 1
        xroot = ("fork", counter)
        node_xy[xroot] = (float(X[0]), float(X[1]))
        pa_to_x = np.vstack([pa[:k + 1], X[None, :]])               # root -> X along arm a
        pa_rest = np.vstack([X[None, :], pa[k + 1:]])               # X -> far end of arm a
        pb, pc = arms[ib][1], arms[ic][1]
        segc = np.hypot(*np.diff(pc, axis=0).T)
        arcc = np.concatenate([[0.0], np.cumsum(segc)])
        keep = int(np.searchsorted(arcc, starts[ic] if starts[ic] is not None else d1))   # start of the straight run
        keep = min(max(keep, 1), len(pc) - 1)
        branch = np.vstack([X[None, :], pc[keep:]])
        branch = _smooth_polyline(branch, 1.0 * spacing_px, max(spacing_px / 2.0, 1.0))
        branch = _resample_polyline(branch, spacing_px)
        through = np.vstack([pa_to_x[::-1], pb[1:]])                # X -> ... -> root -> ... -> far end of b
        # (the through path keeps its old shape; it just passes X now)
        ia_i, ib_i, ic_i = arms[ia][0], arms[ib][0], arms[ic][0]
        far_a, far_b, far_c = arms[ia][2], arms[ib][2], arms[ic][2]
        geo[ia_i] = {"ra": far_a, "rb": xroot, "poly": pa_rest[::-1]}
        geo[ib_i] = {"ra": xroot, "rb": far_b, "poly": through}
        geo[ic_i] = {"ra": xroot, "rb": far_c, "poly": branch}
        node_xy.pop(root, None)


def _dense_crack_from_pixels(pixel_set: set, xs: np.ndarray, ys: np.ndarray, ridge: np.ndarray,
                              smooth: np.ndarray, cell: float, node_spacing_px: float, merge_tol_px: float,
                              name: str, color: str, relocate_branch_points: bool = True) -> "CrackPath":
    """One CrackPath (a tree) from one pruned skeleton component, with
    DENSE nodes that follow the crack's shape (one every `node_spacing_px`)
    and branch points placed ON the cracks:
    * each skeleton chain between two junctions / leaves is smoothed along its
      arc length, re-centred on the ridge, and resampled evenly;
    * a junction -- thinning leaves it as a cluster of adjacent junction pixels,
      plus the short chains between them -- becomes ONE node: the skeleton
      pixel of the cluster nearest its centre, nudged to the strongest strain
      within a cell, so it always lies inside the band;
    * every chain starts/ends exactly at its junction node, so branches are
      connected to their parent;
    * loops are opened by dropping the longest chain of the cycle."""
    chains = _extract_chains(pixel_set)
    if not chains:
        return CrackPath(name=name, color=color)

    def real(px):
        r, c = px
        return (float(xs[min(max(c, 0), len(xs) - 1)]), float(ys[min(max(r, 0), len(ys) - 1)]))

    deg = {p: len(_skeleton_neighbors(pixel_set, *p)) for p in pixel_set}
    ends = {p for ch in chains for p in (ch[0], ch[-1])}
    parent = {p: p for p in ends}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    chain_len = [_polyline_length(np.array([real(q) for q in ch])) for ch in chains]
    for ch, ln in zip(chains, chain_len):
        a, b = ch[0], ch[-1]
        if a != b and deg[a] >= 3 and deg[b] >= 3 and ln <= merge_tol_px:
            parent[find(a)] = find(b)                 # a short chain between two junction pixels: same junction
    groups = {}
    for p in ends:
        groups.setdefault(find(p), []).append(p)

    height, width = smooth.shape
    node_xy, node_of = {}, {}
    for root, members in groups.items():
        if len(members) == 1 and deg[members[0]] < 3:
            node_xy[root] = real(members[0])           # leaf / tip: left exactly where the skeleton ends
        else:
            cx = np.mean([real(m)[0] for m in members]); cy = np.mean([real(m)[1] for m in members])
            best = min(members, key=lambda m: (real(m)[0] - cx) ** 2 + (real(m)[1] - cy) ** 2)
            br, bc = best
            cand = [(br + dr, bc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if 0 <= br + dr < height and 0 <= bc + dc < width and np.isfinite(smooth[br + dr, bc + dc])]
            if cand:
                best = max(cand, key=lambda q: smooth[q])
            node_xy[root] = real(best)
        for m in members:
            node_of[m] = root

    # chains -> graph edges; a chain that would close a loop is dropped (longest first)
    uf = {r: r for r in node_xy}

    def ufind(r):
        while uf[r] != r:
            uf[r] = uf[uf[r]]
            r = uf[r]
        return r

    kept = []
    for idx in sorted(range(len(chains)), key=lambda i: chain_len[i]):
        ch = chains[idx]
        ra, rb = node_of[ch[0]], node_of[ch[-1]]
        if ra == rb:
            continue                                   # contracted away (inside one junction) or a self-loop
        if ufind(ra) == ufind(rb):
            continue                                   # would close a loop
        uf[ufind(ra)] = ufind(rb)
        kept.append(idx)

    sigma_px = 1.5 * max(cell, node_spacing_px / 1.5)
    step_fine = max(node_spacing_px / 2.0, 1.0)
    geo = []
    for idx in kept:
        ch = chains[idx]
        ra, rb = node_of[ch[0]], node_of[ch[-1]]
        poly = np.array([real(q) for q in ch], dtype=float)
        poly[0], poly[-1] = node_xy[ra], node_xy[rb]
        # the junction node may have been nudged: chain pixels right next to it
        # would make the line double back, so let the path leave the node cleanly
        if len(poly) > 3:
            d0 = np.hypot(*(poly[1:-1] - poly[0]).T); d1 = np.hypot(*(poly[1:-1] - poly[-1]).T)
            keep = np.concatenate([[True], (d0 >= 0.9 * cell) & (d1 >= 0.9 * cell), [True]])
            poly = poly[keep]
        poly = _smooth_polyline(poly, sigma_px, step_fine)
        poly = _snap_to_ridge(poly, ridge, xs, ys, reach_px=1.0 * cell, step_px=max(cell / 5.0, 0.5))
        poly = _smooth_polyline(poly, 0.7 * sigma_px, step_fine)
        poly = _resample_polyline(poly, node_spacing_px)
        geo.append({"ra": ra, "rb": rb, "poly": poly})
    if relocate_branch_points:
        _relocate_branch_points(geo, node_xy, node_spacing_px, field=(xs, ys, smooth))

    crack = CrackPath(name=name, color=color)
    index_of_root = {}

    def node_index(root):
        if root not in index_of_root:
            index_of_root[root] = crack.add_node(*node_xy[root], parent_index=None)
        return index_of_root[root]

    for g in geo:
        poly = g["poly"]
        prev = node_index(g["ra"])
        for x, y in poly[1:-1]:
            prev = crack.add_node(float(x), float(y), parent_index=prev)
        end = node_index(g["rb"])
        if (prev, end) not in crack.edges and (end, prev) not in crack.edges:
            crack.edges.append((prev, end))
    return crack


def split_into_branches(crack: "CrackPath", look_px: float = 20.0, master_first: bool = False) -> list:
    """Turns ONE branching crack (tree) into several non-branching cracks:
    the main one from the root, and a new CrackPath for every branch, each
    STARTING at its branch point (so it is still attached to its parent).
    At a fork, the branch that best continues the incoming direction stays in
    the current crack; the others each become a new crack. Nodes are copied
    (the branch point is the same coordinate in the parent and the child).
    `master_first`: the first crack (from the root) is a "master" crack that
    stops at its FIRST branching, where ALL the arms (2 or more) become
    independent cracks of their own starting there -- instead of the
    straightest arm continuing the first crack; later branchings keep the
    usual rule. Without any branching it changes nothing.
    Names/colors are left to the caller."""
    n = len(crack.nodes)
    if n < 2:
        return [crack]
    pts = np.asarray(crack.nodes, dtype=float)
    adj = {i: [] for i in range(n)}
    for i, j in crack.edges:
        adj[i].append(j); adj[j].append(i)
    parent, order = {0: None}, [0]
    for u in order:
        for v in adj[u]:
            if v not in parent:
                parent[v] = u; order.append(v)
    children = {u: [v for v in adj[u] if parent.get(v) == u] for u in parent}

    def ahead(u, first_child):
        """Point ~look_px further along the way from u through first_child (follows
        the first child at later forks -- only the direction matters)."""
        cur, travelled = first_child, float(np.hypot(*(pts[first_child] - pts[u])))
        while travelled < look_px and children[cur]:
            nxt = children[cur][0]
            travelled += float(np.hypot(*(pts[nxt] - pts[cur]))); cur = nxt
        return pts[cur]

    def behind(u):
        cur, travelled = u, 0.0
        while travelled < look_px and parent.get(cur) is not None:
            p = parent[cur]; travelled += float(np.hypot(*(pts[cur] - pts[p]))); cur = p
        return pts[cur]

    def direction(a, b):
        d = np.asarray(b, float) - np.asarray(a, float)
        m = float(np.hypot(*d))
        return d / m if m > 0 else d

    out, queue = [], [(0, c) for c in children[0]]
    if master_first:
        cur, seq = 0, [0]
        while len(children[cur]) == 1:
            cur = children[cur][0]
            seq.append(cur)
        if cur != 0 and len(children[cur]) >= 2:
            master = CrackPath(name=crack.name, color=crack.color)
            prev = None
            for k in seq:
                prev = master.add_node(*pts[k], parent_index=prev)
            out.append(master)
            queue = [(cur, c) for c in children[cur]]
    while queue:
        start, first = queue.pop(0)
        seq, cur = [start, first], first
        while True:
            ch = children[cur]
            if not ch:
                break
            if len(ch) == 1:
                nxt = ch[0]
            else:
                inc = direction(behind(cur), pts[cur])
                scores = [float(np.dot(inc, direction(pts[cur], ahead(cur, c)))) for c in ch]
                nxt = ch[int(np.argmax(scores))]
                queue += [(cur, c) for c in ch if c != nxt]
            seq.append(nxt); cur = nxt
        piece = CrackPath(name=crack.name, color=crack.color)
        prev = None
        for k in seq:
            prev = piece.add_node(*pts[k], parent_index=prev)
        out.append(piece)
    return out or [crack]


def detect_cracks_ridge(values_grid: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                         percentile: float | None = None, sensitivity: float = 1.0,
                         min_length_px: float = 15.0, node_spacing_px: float | None = None,
                         spur_length_px: float | None = None, low_ratio: float = 0.85,
                         sigmas_cells=None, weak_spur_factor: float = 1.6,
                         split_branches: bool = False, master_first_crack: bool = False) -> list:
    """Automatically proposes crack path(s), branches included, from a strain
    field -- see the section comment above for why this is more robust than
    `detect_cracks` on real, noisy data.

    The ridge-strength threshold is chosen AUTOMATICALLY (Otsu's method on
    the positive ridge strengths): it separates "line-like" pixels from the
    noise whatever fraction of the image the cracks cover, which a fixed
    percentile cannot do (a percentile that suits one frame cuts the
    branches of the next, or lets noise in). `sensitivity` scales it: > 1
    keeps weaker ridges (more / longer branches, more noise), < 1 keeps
    only the surest. Alternatively give `percentile` to force the threshold
    to the ridge strength exceeded by only (100 - percentile) % of the grid.
    Ridge pixels above `low_ratio` x the threshold that touch a sure pixel
    are kept too (hysteresis), so a fading branch stays attached to its trunk.
    `min_length_px`: drops pieces shorter than this (real pixel units).
    `spur_length_px`: terminal branches shorter than this are pruned (default
    = 3 grid cells, at least 21 px). Lengths and widths are scaled by the grid
    spacing, so the result does not depend on the correlation mesh step.
    `node_spacing_px`: distance between the nodes along each crack (default: one
    grid cell, at least 5 px) -- the path is smoothed and centred on the ridge,
    so it follows the crack's shape; smaller = more points.
    `split_branches`: False returns one CrackPath per connected tree (branches
    included, as a tree); True returns a separate CrackPath for the main crack
    and for EVERY branch (see `split_into_branches`); `master_first_crack`
    (with `split_branches`): the first crack of each tree is a master crack that
    ends at its first branching, where all the arms become independent cracks.
    Returns a list of CrackPath."""
    from scipy import ndimage
    from skimage.filters import apply_hysteresis_threshold, threshold_otsu

    a = np.asarray(values_grid, dtype=float)
    if a.ndim != 2 or a.size == 0 or not np.isfinite(a).any():
        return []
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    cell = float(np.median(np.diff(xs))) if xs.size > 1 else 1.0
    if not np.isfinite(cell) or cell <= 0:
        cell = 1.0
    # Everything below is expressed in REAL pixels (a crack's width, the length
    # of a twig worth ignoring...) and converted with the grid spacing, so the
    # result does not depend on the mesh step chosen in the correlation.
    if sigmas_cells is None:
        sigmas_cells = sorted({max(0.8, v / cell) for v in (7.0, 11.0, 17.0)})
    if node_spacing_px is None or node_spacing_px <= 0:
        node_spacing_px = max(cell, 5.0)
    ridge, smooth = _ridge_response(a, sigmas_cells)
    valid = np.isfinite(a)
    if percentile is not None:
        hi = float(np.percentile(ridge[valid], percentile))
    else:
        positive = ridge[valid & (ridge > 0)]
        if positive.size < 10 or float(positive.max()) <= float(positive.min()):
            return []
        otsu = float(threshold_otsu(positive))
        # On a field with no crack (pure noise) Otsu still finds a threshold, and
        # would trace noise ridges. Real cracks make the ridge strengths clearly
        # bimodal: the Otsu level sits >= ~1.9x the median ridge strength, against
        # ~1.1-1.4x for noise alone -- below 1.6 there is nothing to detect.
        if otsu < _MIN_RIDGE_CONTRAST * float(np.median(positive)):
            return []
        hi = otsu / max(float(sensitivity), 1e-3)
    if not hi > 0:
        return []
    lo = low_ratio * hi
    mask = apply_hysteresis_threshold(ridge, lo, hi) & valid
    if not mask.any():
        return []
    # Work on a copy extended past the data edge: thinning pulls the end of a
    # thick band back by about half its width (and a morphological closing
    # erodes the border), so a crack that runs into the edge of the data --
    # where it usually starts -- would lose its first cells. Extend, clean,
    # thin, then crop back.
    pad = max(6, int(np.ceil(30.0 / cell)))
    mask = np.pad(mask, pad, mode="edge")
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    # small enclosed holes (a band split around a speck) would thin into loops
    holes = ndimage.binary_fill_holes(mask) & ~mask
    if holes.any():
        hole_id, n_holes = ndimage.label(holes)
        sizes = ndimage.sum(holes, hole_id, index=np.arange(1, n_holes + 1))
        for k, size in enumerate(sizes, start=1):
            if size * cell * cell <= 40 * 49.0:     # holes up to ~40 cells of a 7-px mesh
                mask |= hole_id == k
    skeleton = _skeletonize_mask(mask)[pad:-pad, pad:-pad]
    if not skeleton.any():
        return []
    spur_px = spur_length_px if spur_length_px is not None else max(3.0 * cell, 21.0)
    spur_pixels = max(2, int(round(spur_px / cell)) + 1)
    weak_pixels = max(spur_pixels + 1, int(round(max(8.0 * cell, 56.0) / cell)))
    merge_tol = max(2.2 * cell, 15.0)
    labeled, n_components = ndimage.label(skeleton, structure=np.ones((3, 3), dtype=int))
    cracks = []
    for label_id in range(1, n_components + 1):
        rows, cols = np.nonzero(labeled == label_id)
        pixel_set = _prune_spurs(_remove_corner_pixels(set(zip(rows.tolist(), cols.tolist()))), spur_pixels,
                                  strength=ridge,
                                  weak_len_pixels=weak_pixels,
                                  weak_below=weak_spur_factor * hi)
        placeholder = new_crack(len(cracks))
        crack = _dense_crack_from_pixels(pixel_set, xs, ys, ridge, smooth, cell, node_spacing_px, merge_tol,
                                          name=f"{placeholder.name} (auto)", color=placeholder.color)
        if crack.total_length() >= min_length_px and len(crack.nodes) >= 2:
            cracks.append(_root_at_origin_leaf(crack, xs, ys, smooth, cell))
    if split_branches:
        pieces = []
        for tree in cracks:
            for part in split_into_branches(tree, master_first=master_first_crack):
                placeholder = new_crack(len(pieces))
                part.name, part.color = f"{placeholder.name} (auto)", placeholder.color
                pieces.append(part)
        return pieces
    return cracks


# ---------------------------------------------------------------------------
# Manual adjustment of a branch point
# ---------------------------------------------------------------------------

def _chain_points(crack: "CrackPath"):
    """Ordered (n, 2) node array and node-index order when `crack` is ONE
    non-branching path (every node degree <= 2, connected); None otherwise."""
    n = len(crack.nodes)
    if n < 2 or len(crack.edges) != n - 1:
        return None
    adj = {i: [] for i in range(n)}
    for a, b in crack.edges:
        adj[a].append(b)
        adj[b].append(a)
    if any(len(v) > 2 for v in adj.values()):
        return None
    ends = [i for i in range(n) if len(adj[i]) == 1]
    if len(ends) != 2:
        return None
    order = [ends[0]]
    prev = None
    while len(order) < n:
        nxt = [j for j in adj[order[-1]] if j != prev]
        if not nxt:
            return None
        prev = order[-1]
        order.append(nxt[0])
    pts = np.asarray([crack.nodes[i] for i in order], dtype=float)
    return pts, order


def _dist_to_polyline(pts: np.ndarray, p) -> tuple:
    """(distance, arc position of the closest point) from `p` to polyline `pts`."""
    p = np.asarray(p, float)
    seg = np.diff(pts, axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(ln)])
    best = (np.inf, 0.0)
    for k in range(len(seg)):
        if ln[k] <= 0:
            continue
        t = float(np.clip(np.dot(p - pts[k], seg[k]) / (ln[k] ** 2), 0.0, 1.0))
        q = pts[k] + t * seg[k]
        d = float(np.hypot(*(p - q)))
        if d < best[0]:
            best = (d, float(arc[k] + t * ln[k]))
    return best


def _arc_positions(pts: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])


def _set_chain(crack: "CrackPath", pts: np.ndarray) -> None:
    """Replaces the crack's geometry by the chain `pts` (edges i -> i+1); the
    initiation/end markers are kept on the nearest new node when they still
    sit within a node spacing, dropped otherwise."""
    old = {}
    for name in ("initiation_index", "end_index"):
        idx = getattr(crack, name)
        if idx is not None and 0 <= idx < len(crack.nodes):
            old[name] = np.asarray(crack.nodes[idx], float)
    crack.nodes = [(float(x), float(y)) for x, y in pts]
    crack.edges = [(i, i + 1) for i in range(len(pts) - 1)]
    spacing = float(np.median(np.hypot(*np.diff(pts, axis=0).T))) if len(pts) > 1 else 1.0
    for name in ("initiation_index", "end_index"):
        setattr(crack, name, None)
        if name in old:
            d = np.hypot(pts[:, 0] - old[name][0], pts[:, 1] - old[name][1])
            k = int(np.argmin(d))
            if d[k] <= 1.5 * spacing:
                setattr(crack, name, k)


def _reroute_chain(PP: np.ndarray, C, spacing: float, pin_start: bool = False, pin_end: bool = False) -> np.ndarray:
    """The chain `PP` re-routed so that it passes exactly through `C`: the
    points within a window around the closest place are replaced by `C`, the
    neighbourhood is smoothed and evenly re-spaced, the rest is untouched.
    `pin_start`/`pin_end` keep that end node where it is even when it falls
    inside the window (an end attached to another crack)."""
    C = np.asarray(C, float)
    s_p = _arc_positions(PP)
    dp_, sp_c = _dist_to_polyline(PP, C)
    w = max(1.5 * spacing, 1.3 * dp_)
    lo, hi = sp_c - 3.0 * w, sp_c + 3.0 * w
    before = PP[s_p < lo]
    mid_a = PP[(s_p >= lo) & (s_p < sp_c - w)]
    mid_b = PP[(s_p > sp_c + w) & (s_p <= hi)]
    after = PP[s_p > hi]
    if pin_start and not (len(before) or len(mid_a)):
        before = PP[:1]
    if pin_end and not (len(after) or len(mid_b)):
        after = PP[-1:]
    local = np.vstack([mid_a, C[None, :], mid_b])
    if len(local) >= 3:
        local = _smooth_polyline(local, 0.8 * spacing, max(spacing / 2.0, 1.0))
        local = _resample_polyline(local, spacing)
    d = np.hypot(local[:, 0] - C[0], local[:, 1] - C[1])
    k = int(np.argmin(d))
    if d[k] <= 0.6 * spacing and 0 < k < len(local) - 1:
        local[k] = C
    elif d[k] <= 0.6 * spacing and len(local) > 1 and (k == 0 or k == len(local) - 1) and not (
            (k == 0 and len(before)) or (k == len(local) - 1 and len(after))):
        local[k] = C                                     # a free end: the end becomes C
    else:                                                # insert at the right place along `local`
        _, s_loc = _dist_to_polyline(local, C) if len(local) > 1 else (0.0, 0.0)
        sl = _arc_positions(local) if len(local) > 1 else np.array([0.0])
        pos = int(np.searchsorted(sl, s_loc))
        local = np.vstack([local[:pos], C[None, :], local[pos:]])
    return np.vstack([a for a in (before, local, after) if len(a)])


def _attach_tol(chains: dict) -> float:
    """Distance (px) under which two crack ends / an end and a path count as "the same point"."""
    spacings = [float(np.median(np.hypot(*np.diff(pts, axis=0).T))) for pts in chains.values() if len(pts) > 1]
    sp = max(float(np.median(spacings)), 1.0) if spacings else 1.0
    return max(1.5, 0.6 * sp)


def find_junctions(cracks: list) -> list:
    """Every place where cracks are attached to each other, as a list of
    {"point": (x, y), "ends": [(crack_index, 0 | -1), ...], "through": [crack_index, ...]}:

    * "ends": the cracks having one END (start = 0, last node = -1) at that point -- the
      branches leaving a fork, and also a "master" crack stopping at its first branching;
    * "through": the cracks whose path passes through that point without ending there (the
      parent a branch leaves from).

    A junction involves at least two different cracks. Only non-branching cracks (what "One crack
    per branch" gives) take part. This is what "Move branch point" works on: a fork is ONE
    junction however many cracks meet there."""
    chains = {}
    for i, c in enumerate(cracks):
        ch = _chain_points(c)
        if ch is not None:
            chains[i] = ch[0]
    if len(chains) < 2:
        return []
    tol = _attach_tol(chains)
    ends = [(i, e, pts[e]) for i, pts in chains.items() for e in (0, -1)]
    found, seen = [], set()
    for i, e, P in ends:
        end_members = [(j, ee) for j, ee, Q in ends if float(np.hypot(*(Q - P))) <= tol]
        end_cracks = {j for j, _ in end_members}
        through = [j for j, pts in chains.items()
                   if j not in end_cracks and _dist_to_polyline(pts, P)[0] <= tol]
        if len(end_cracks) + len(through) < 2:
            continue
        key = (frozenset(end_members), frozenset(through))
        if key in seen:
            continue
        seen.add(key)
        point = np.mean([chains[j][ee] for j, ee in end_members], axis=0)
        found.append({"point": (float(point[0]), float(point[1])),
                      "ends": sorted(end_members), "through": sorted(through)})
    return found


def _rebuild_end(pts: np.ndarray, end: int, C: np.ndarray) -> np.ndarray:
    """The chain `pts` with its END `end` (0 or -1) moved to C: C, then the crack's own points
    beyond the old end (smoothed, evenly re-spaced), stored in the original orientation."""
    PB = pts if end == 0 else pts[::-1]                    # the moving end first
    J = PB[0].copy()
    sp = max(float(np.median(np.hypot(*np.diff(PB, axis=0).T))), 1.0)
    s_b = _arc_positions(PB)
    dC, sC = _dist_to_polyline(PB, C)
    dJ = float(np.hypot(*(C - J)))
    s_cut = sC if dC <= 0.5 * dJ + sp else dJ
    keep = PB[s_b > s_cut + 0.5 * sp]
    if len(keep) == 0:
        keep = PB[-1:]
    new_b = np.vstack([C[None, :], keep])
    if len(new_b) >= 3:
        new_b = _smooth_polyline(new_b, sp, max(sp / 2.0, 1.0))
    new_b = _resample_polyline(new_b, sp)
    new_b[0] = C
    return new_b if end == 0 else new_b[::-1]


def _snap_attached_ends(cracks: list, skip: set, old_path: np.ndarray, new_path: np.ndarray, tol: float) -> list:
    """Cracks (not in `skip`) with an end that sat on `old_path` get that end moved onto the
    nearest point of `new_path`, so branches leaving a re-routed crack stay attached to it.
    Returns the indices that moved."""
    moved = []
    seg = np.diff(new_path, axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    for j, other in enumerate(cracks):
        if j in skip:
            continue
        ch = _chain_points(other)
        if ch is None:
            continue
        pts_o, order_o = ch
        for end_pos, node_i in ((0, order_o[0]), (len(pts_o) - 1, order_o[-1])):
            if _dist_to_polyline(old_path, pts_o[end_pos])[0] > tol:
                continue
            best, bq = np.inf, None
            for kk in range(len(seg)):
                if ln[kk] <= 0:
                    continue
                t = float(np.clip(np.dot(pts_o[end_pos] - new_path[kk], seg[kk]) / ln[kk] ** 2, 0, 1))
                q = new_path[kk] + t * seg[kk]
                dd = float(np.hypot(*(pts_o[end_pos] - q)))
                if dd < best:
                    best, bq = dd, q
            if bq is not None and best > 0.2:
                other.nodes[node_i] = (float(bq[0]), float(bq[1]))
                moved.append(j)
    return moved


def nearest_junction(cracks: list, x: float, y: float, max_reach_px: float = 100.0, prefer: int | None = None):
    """(junction, distance) of the junction nearest to the click, or (None, nearest distance).
    `prefer` (a crack index, e.g. the one selected in the list) makes the junctions that crack
    takes part in count as 30 % closer -- a tie-breaker between forks that are about as far."""
    best, best_score, best_d = None, np.inf, np.inf
    for jn in find_junctions(cracks):
        d = float(np.hypot(jn["point"][0] - x, jn["point"][1] - y))
        members = {j for j, _ in jn["ends"]} | set(jn["through"])
        score = d * (0.7 if prefer is not None and prefer in members else 1.0)
        best_d = min(best_d, d)
        if d <= max_reach_px and score < best_score:
            best, best_score = jn, score
    return (best, float(np.hypot(best["point"][0] - x, best["point"][1] - y))) if best else (None, best_d)


def move_branch_point(cracks: list, x: float, y: float, max_reach_px: float = 100.0,
                      prefer: int | None = None) -> dict:
    """Manual fix of a branch point: the user clicks where the fork really is (x, y).

    The fork to fix is the JUNCTION (see `find_junctions`) whose current position is nearest to the
    click -- not "the cracks nearest to the click", which at a fork is arbitrary since every
    crack of the fork passes about equally close. ALL cracks of that junction are moved together:

    * every crack that ENDS there (the branches, or a master crack stopping at the fork) gets that
      end moved to the click, followed by its own points beyond the old end (smoothed, evenly
      re-spaced);
    * a crack that passes THROUGH it (the parent) is re-routed through the click, locally;
    * other cracks attached to a re-routed parent elsewhere follow it, so nothing is left detached.

    Cracks are modified in place. Returns {"old", "new", "moved": [crack indices, ...],
    "ends": [...], "through": [...], "parent": first crack passing through (or None), "branch":
    first crack ending there}; raises ValueError with a message for the user when there is no
    junction within `max_reach_px` of the click."""
    C = np.array([float(x), float(y)])
    junctions = find_junctions(cracks)
    if not junctions:
        raise ValueError("No branch point to move: it needs cracks that start on another crack "
                         "(use \"One crack per branch\" when auto-detecting).")
    jn, dist = nearest_junction(cracks, x, y, max_reach_px, prefer)
    if jn is None:
        raise ValueError(f"No branch point within {max_reach_px:.0f} px of that click (the nearest is "
                         f"{dist:.0f} px away) -- click closer to where the fork is now.")
    chains = {i: _chain_points(c)[0] for i, c in enumerate(cracks) if _chain_points(c) is not None}
    tol = _attach_tol(chains)
    J = np.array(jn["point"])
    end_cracks = {j for j, _ in jn["ends"]}

    # re-route the through-parents first, from their ORIGINAL geometry
    new_chains, snapped = {}, []
    for pidx in jn["through"]:
        PP = chains[pidx]
        sp_p = max(float(np.median(np.hypot(*np.diff(PP, axis=0).T))), 1.0)

        def attached(end_pt, me=pidx):
            return any(_dist_to_polyline(chains[o], end_pt)[0] <= tol for o in chains if o != me)
        new_chains[pidx] = _reroute_chain(PP, C, sp_p, pin_start=attached(PP[0]), pin_end=attached(PP[-1]))
        snapped += _snap_attached_ends(cracks, {pidx} | end_cracks | set(jn["through"]), PP, new_chains[pidx], tol)
    for j, e in jn["ends"]:
        new_chains[j] = _rebuild_end(chains[j], e, C)
    for idx, pts in new_chains.items():
        _set_chain(cracks[idx], pts)
    moved = sorted(set(new_chains) | set(snapped))
    return {"old": (float(J[0]), float(J[1])), "new": (float(C[0]), float(C[1])), "moved": moved,
            "ends": sorted(end_cracks), "through": list(jn["through"]),
            "parent": jn["through"][0] if jn["through"] else None,
            "branch": jn["ends"][0][0] if jn["ends"] else None}


# ---------------------------------------------------------------------------
# Order and parentage of the cracks
# ---------------------------------------------------------------------------

def crack_parents(cracks: list) -> dict:
    """{child index: (parent index, (x, y) of the branch point)} for every crack that BRANCHES on
    another one (display / sanity checks -- the data itself only stores geometry).

    At a junction where a crack passes THROUGH, that crack is the parent of every crack ending there.
    At a junction where cracks only end (a master crack stopping at its fork), the EARLIEST crack of
    the list is the parent of the later ones -- which is why the order of the list matters: put the
    crack that should come first (the one the others branch on) above them."""
    out = {}
    for jn in find_junctions(cracks):
        ends = [j for j, _ in jn["ends"]]
        if jn["through"]:
            parent = jn["through"][0]
            for c in ends:
                out.setdefault(c, (parent, jn["point"]))
        else:
            first = min(ends)
            for c in ends:
                if c != first:
                    out.setdefault(c, (first, jn["point"]))
    return out


def move_crack_in_list(cracks: list, index: int, delta: int) -> int:
    """Moves crack `index` by `delta` places in the list (in place); returns its new index
    (unchanged when it is already at that end of the list)."""
    new = max(0, min(len(cracks) - 1, index + delta))
    if new != index:
        cracks.insert(new, cracks.pop(index))
    return new


def renumber_cracks(cracks: list) -> int:
    """Renames "Crack 7 (auto)" style cracks to follow their position in the list ("Crack 1 (auto)",
    "Crack 2 (auto)", ...), keeping any suffix; other names are left alone. Returns how many changed."""
    import re
    changed = 0
    for k, c in enumerate(cracks):
        m = re.match(r"^Crack\s+\d+(.*)$", c.name or "")
        if m:
            name = f"Crack {k + 1}{m.group(1)}"
            if name != c.name:
                c.name, changed = name, changed + 1
    return changed


def _insert_point_on_chain(pts: np.ndarray, P, tol: float):
    """(new chain, index of the node at P): P is a node of the chain afterwards -- an existing node
    when one lies within `tol`, otherwise P inserted between its two neighbours."""
    P = np.asarray(P, float)
    d = np.hypot(pts[:, 0] - P[0], pts[:, 1] - P[1])
    k = int(np.argmin(d))
    if d[k] <= tol:
        return pts, k
    _, s = _dist_to_polyline(pts, P)
    arc = _arc_positions(pts)
    pos = int(np.searchsorted(arc, s))
    pos = min(max(pos, 1), len(pts) - 1)
    return np.vstack([pts[:pos], P[None, :], pts[pos:]]), pos


def set_crack_parent(cracks: list, child: int, parent: int, put_after: bool = True,
                     reach_px: float = 150.0) -> dict:
    """Makes crack `child` BRANCH ON crack `parent`.

    The child's attached end (the one already on another crack, else the one nearest the parent) is
    moved onto the parent's path -- where the child really leaves it: following the child from that end
    while it still runs along the parent, the branch point is the last point of that stretch. The
    parent gets a node there, the child restarts from it (its points beyond are kept, smoothed and
    re-spaced), and cracks attached to the child's old start are left where they are. With
    `put_after` the child is also moved to the position right after its parent in the list.

    Cracks are modified in place (and `cracks` re-ordered). Returns {"child", "parent" (their NEW
    indices), "point", "moved_in_list"}; raises ValueError with a message when it cannot be done."""
    if child == parent:
        raise ValueError("A crack cannot branch on itself.")
    for i in (child, parent):
        if not (0 <= i < len(cracks)):
            raise ValueError("Select a crack in the list first.")
    cc, pc = cracks[child], cracks[parent]
    ch_c, ch_p = _chain_points(cc), _chain_points(pc)
    if ch_c is None or ch_p is None:
        raise ValueError("Both cracks must be single non-branching paths (auto-detect with "
                         "\"One crack per branch\", or split them first).")
    pts_c, pts_p = ch_c[0], ch_p[0]
    sp = max(float(np.median(np.hypot(*np.diff(pts_c, axis=0).T))), 1.0)
    tol = max(1.5, 0.6 * sp)
    # a parent that itself branches on the child would make a loop
    back = crack_parents(cracks).get(parent)
    if back is not None and back[0] == child:
        raise ValueError(f"\"{pc.name}\" already branches on \"{cc.name}\" -- pick the other way round, "
                         "or first make the parent independent.")
    # which end of the child is the branching end
    others = []
    for k, c in enumerate(cracks):
        ch = _chain_points(c) if k != child else None
        if ch is not None:
            others.append(ch[0])

    def attached(e):
        return any(_dist_to_polyline(o, pts_c[e])[0] <= tol for o in others)
    end = 0 if (attached(0) and not attached(-1)) else (-1 if (attached(-1) and not attached(0)) else None)
    if end is None:
        end = 0 if _dist_to_polyline(pts_p, pts_c[0])[0] <= _dist_to_polyline(pts_p, pts_c[-1])[0] else -1
    PB = pts_c if end == 0 else pts_c[::-1]                 # branching end first
    d0, _ = _dist_to_polyline(pts_p, PB[0])
    if d0 > reach_px:
        raise ValueError(f"\"{cc.name}\" starts {d0:.0f} px away from \"{pc.name}\" -- too far to attach "
                         "(move its start closer first, e.g. with \"Correct path\").")
    # where the child leaves the parent: last point of its initial stretch that still runs along it
    run_tol = max(2.0 * tol, 0.4 * sp * 3)
    last = 0
    for k in range(len(PB)):
        if _dist_to_polyline(pts_p, PB[k])[0] <= run_tol:
            last = k
        else:
            break
    _, s_par = _dist_to_polyline(pts_p, PB[last])
    # The branch point must lie in the INTERIOR of the parent (about one node away from its ends),
    # otherwise "the child branches on the parent" cannot be told apart from "both start at the same point".
    arc_p = _arc_positions(pts_p)
    off = 1.2 * sp                                         # one node + a bit: outside the end node's tolerance
    if arc_p[-1] > 2.0 * off:
        s_par = float(np.clip(s_par, off, arc_p[-1] - off))
    # point of the parent path at arc position s_par
    P = np.array([np.interp(s_par, arc_p, pts_p[:, 0]), np.interp(s_par, arc_p, pts_p[:, 1])])
    new_parent_pts, _k = _insert_point_on_chain(pts_p, P, tol)
    P = new_parent_pts[_k]
    new_child = _rebuild_end(pts_c, end, P)
    _set_chain(pc, new_parent_pts)
    _set_chain(cc, new_child)
    moved = False
    if put_after and child < parent:
        cracks.insert(parent, cracks.pop(child))            # child goes right after the parent
        moved = True
    elif put_after and child > parent + 1:
        cracks.insert(parent + 1, cracks.pop(child))
        moved = True
    return {"child": cracks.index(cc), "parent": cracks.index(pc),
            "point": (float(P[0]), float(P[1])), "moved_in_list": moved}


def choose_crack_for_click(cracks: list, x: float, y: float, selected: int | None = None,
                           max_reach_px: float = 100.0):
    """Which crack a "Correct path" click is meant for -> (index, reason) or (None, message).

    The crack NEAREST to the click is the one the operator is pointing at; the crack selected in
    the list wins only when it is about as near (within twice the nearest distance + 8 px), so a
    stale selection can never hijack a click made next to another crack. Cracks that cannot be
    edited as a path (fewer than two points) are ignored."""
    C = np.array([float(x), float(y)])
    dists = {}
    for i, c in enumerate(cracks):
        if len(c.nodes) < 2:
            continue
        ch = _chain_points(c)
        pts = ch[0] if ch is not None else np.asarray(c.nodes, float)
        dists[i] = _dist_to_polyline(pts, C)[0] if ch is not None else float(
            np.min(np.hypot(pts[:, 0] - C[0], pts[:, 1] - C[1])))
    if not dists:
        return None, "There is no crack to correct."
    near = min(dists, key=dists.get)
    if dists[near] > max_reach_px:
        return None, f"No crack within {max_reach_px:.0f} px of that click -- click nearer to the crack's path."
    if selected is not None and selected in dists and dists[selected] <= max_reach_px \
            and dists[selected] <= 2.0 * dists[near] + 8.0:
        return selected, "selected"
    return near, ("nearest" if selected != near else "selected")


def reroute_crack_through_point(cracks: list, index: int, x: float, y: float, max_reach_px: float = 80.0) -> dict:
    """Manual correction of ONE crack's path: it is re-made so that it passes
    through the clicked point (x, y), the neighbourhood re-smoothed and
    evenly re-spaced, the rest of the path untouched. An end that is attached
    to another crack (a branch's start) stays put; a free end can be moved
    by clicking near it. Other cracks whose end sits on the old path (the
    branches leaving it) follow the new path, so they stay attached. For a
    crack that is still a tree (not split), the nearest node is moved to the
    click instead. Cracks are modified in place; returns {"crack": index,
    "snapped": [indices of branches that followed]}; raises ValueError with a
    message when the click is too far from the crack."""
    if not (0 <= index < len(cracks)):
        raise ValueError("Select a crack in the list first.")
    crack = cracks[index]
    C = np.array([float(x), float(y)])
    chain = _chain_points(crack)
    if chain is None:
        if len(crack.nodes) < 2:
            raise ValueError("That crack has too few points to correct.")
        pts = np.asarray(crack.nodes, float)
        d = np.hypot(pts[:, 0] - C[0], pts[:, 1] - C[1])
        k = int(np.argmin(d))
        if d[k] > max_reach_px:
            raise ValueError("The click is too far from the selected crack.")
        crack.nodes[k] = (float(C[0]), float(C[1]))
        return {"crack": index, "snapped": [], "tree": True}
    PP, order = chain
    sp = max(float(np.median(np.hypot(*np.diff(PP, axis=0).T))), 1.0)
    dist, _ = _dist_to_polyline(PP, C)
    if dist > max_reach_px:
        raise ValueError("The click is too far from the selected crack -- click nearer to its path.")

    def attached(end):
        for j, other in enumerate(cracks):
            if j == index:
                continue
            ch = _chain_points(other)
            if ch is not None and _dist_to_polyline(ch[0], end)[0] <= max(1.5, 0.6 * sp):
                return True
        return False

    pin_s, pin_e = attached(PP[0]), attached(PP[-1])
    new_p = _reroute_chain(PP, C, sp, pin_start=pin_s, pin_end=pin_e)
    # branches that left the OLD path must follow the new one
    snapped = []
    for j, other in enumerate(cracks):
        if j == index:
            continue
        ch = _chain_points(other)
        if ch is None:
            continue
        pts_o, order_o = ch
        for end_pos, node_i in ((0, order_o[0]), (len(pts_o) - 1, order_o[-1])):
            if _dist_to_polyline(PP, pts_o[end_pos])[0] <= max(1.5, 0.6 * sp):
                # nearest point of the new path
                seg = np.diff(new_p, axis=0)
                ln = np.hypot(seg[:, 0], seg[:, 1])
                best, bq = np.inf, None
                for kk in range(len(seg)):
                    if ln[kk] <= 0:
                        continue
                    t = float(np.clip(np.dot(pts_o[end_pos] - new_p[kk], seg[kk]) / ln[kk] ** 2, 0, 1))
                    q = new_p[kk] + t * seg[kk]
                    dd = float(np.hypot(*(pts_o[end_pos] - q)))
                    if dd < best:
                        best, bq = dd, q
                if bq is not None and best > 0.2:
                    other.nodes[node_i] = (float(bq[0]), float(bq[1]))
                    snapped.append(j)
    _set_chain(crack, new_p)
    return {"crack": index, "snapped": snapped}
