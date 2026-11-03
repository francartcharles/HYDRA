"""
Digital Image Correlation (DIC) processing: subset-based point tracking
across a sequence of frames, using normalized cross-correlation (NCC)
template matching, plus a local strain estimate over the regular
correlation mesh.

Two sub-pixel modes (`subpixel_method` of `run_dic`):
- "parabolic": each subset is tracked as a rigid, translating template
  located by OpenCV's matchTemplate, with a 3-point parabolic sub-pixel
  refinement of the correlation peak. Fast, but that parabola is biased
  towards whole pixels ("peak locking") and the rigid template cannot
  follow the deformation of its own subset; since strain is the DERIVATIVE
  of the displacement, both errors show up as vertical / horizontal
  stripes in the strain maps.
- "icgn" (the app's default): the same integer + parabolic estimate is
  then refined by core/dic_icgn.py -- affine (first-order) subset shape
  function, ZNSSD criterion, inverse-compositional Gauss-Newton, cubic
  interpolation -- the algorithm of Ncorr / VIC-2D. Removes the stripes at
  a few times the run time of the parabolic mode.

Every subset's template is always the patch cut from the REFERENCE frame
(never re-cut from a later frame), so per-step correlation error never
compounds into long-term template drift. What does change with
`tracking_mode` (see `run_dic`) is only where each frame's search window
is centered:
- "fixed": always at the point's original reference position -- simplest
  and immune to a mis-track anywhere upstream propagating forward, but
  `search_radius` must alone cover the point's total displacement over
  the whole processed sequence.
- "moving": at wherever the point was found in the *previous* processed
  frame -- lets a small `search_radius` follow a large cumulative
  displacement (the speckle pattern's own motion) a little at a time, at
  the cost of a bad match in one frame shifting the search window (and so
  every later frame) away from the point's true position.

A point can also become permanently unanalyzable partway through a
sequence -- most commonly because the specimen's cross-section shrinks
(necking) and a point originally on the specimen ends up on the
background instead, which the reference template then simply doesn't
match anymore. `correlation_threshold` (see `run_dic`) lets a caller
drop a point, for good, the first time its match quality falls below a
given NCC value, rather than keep reporting an unreliable position (and,
worse in "moving" mode, using that unreliable position as every later
frame's search center too).

`max_neighbor_deviation_px` (see `run_dic` and
`_apply_neighbor_consistency_filter`) catches the other common source of
a visibly bad point: one that still matched well (so `correlation_
threshold` wouldn't touch it) but landed somewhere implausible next to
its neighbors -- a similar-looking bit of speckle a little off from the
real one, more likely the wider the search window has to be. Plotted, a
single such point turns an otherwise smoothly-deformed mesh visibly
jagged right around it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core import dic_icgn
from core.dic_correlation import MeshGrid
from core.dic_uncertainty import estimate_noise_sigma, subset_sssig_xy
from core.parallel import run_map, chunk_items, check_cancelled


@dataclass
class DICFrameResult:
    """Tracking result for one processed frame, one 2D array per
    quantity, each shaped (mesh.n_points_y, mesh.n_points_x) -- same
    layout as `mesh.grid_xy()`. `correlation` is the peak normalized
    cross-correlation coefficient (up to 1.0 = perfect match) for each
    point; NaN wherever the point's subset/search window fell outside the
    frame (lost track)."""
    frame_index: int
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray  # x - x_reference
    v: np.ndarray  # y - y_reference
    correlation: np.ndarray
    exx: np.ndarray | None = None
    eyy: np.ndarray | None = None
    exy: np.ndarray | None = None


@dataclass
class DICResult:
    mesh: MeshGrid
    frame_indices: list[int]
    frames: list[DICFrameResult]
    # Measurement-uncertainty ingredients (see core/dic_uncertainty.py),
    # computed once from the REFERENCE frame/subsets only -- unlike
    # `frames`' own per-frame fields, these don't change across the
    # sequence. `noise_sigma` is a single blind estimate of the
    # reference frame's own pixel-intensity noise; `sssig_x`/`sssig_y`
    # (shape (mesh.n_points_y, mesh.n_points_x), NaN wherever a point
    # was excluded up front) are each point's own Sum-of-Squared-Subset-
    # Intensity-Gradients, in x/y separately -- the Cramér-Rao-type
    # registration-precision floor's other ingredient. None if computing
    # these failed for some reason (e.g. no active point) -- an
    # uncertainty estimate is a nice-to-have, not a reason to fail an
    # otherwise-successful correlation run.
    noise_sigma: float | None = None
    sssig_x: np.ndarray | None = None
    sssig_y: np.ndarray | None = None
    # str(exc) from whatever made the block above fail, so the UI can
    # show *why* no uncertainty estimate is available instead of just
    # that one isn't -- None whenever noise_sigma/sssig_x/sssig_y above
    # were computed successfully (this field is never set on success).
    uncertainty_error: str | None = None

    def frame_result(self, frame_index: int) -> DICFrameResult | None:
        for f in self.frames:
            if f.frame_index == frame_index:
                return f
        return None


def _subpixel_peak(corr_map: np.ndarray, py: int, px: int) -> tuple[float, float]:
    """3-point parabolic interpolation of the correlation peak along x
    and y independently, for sub-pixel displacement precision. Falls back
    to the integer peak location wherever the peak sits on the edge of
    the search map (no neighbor on one side to fit a parabola through)."""
    h, w = corr_map.shape
    dx = dy = 0.0
    if 0 < px < w - 1:
        c0, c1, c2 = corr_map[py, px - 1], corr_map[py, px], corr_map[py, px + 1]
        denom = c0 - 2 * c1 + c2
        if abs(denom) > 1e-12:
            dx = 0.5 * (c0 - c2) / denom
    if 0 < py < h - 1:
        c0, c1, c2 = corr_map[py - 1, px], corr_map[py, px], corr_map[py + 1, px]
        denom = c0 - 2 * c1 + c2
        if abs(denom) > 1e-12:
            dy = 0.5 * (c0 - c2) / denom
    return float(px + dx), float(py + dy)


def _track_subset(frame: np.ndarray, template: np.ndarray, center_x: float, center_y: float,
                   search_radius: int) -> tuple[float, float, float]:
    """Locates `template` inside `frame`, searching a window extending
    `search_radius` pixels beyond the template on every side, centered at
    (center_x, center_y). Returns the new (x, y) center and the peak NCC
    value (1.0 = perfect match), or (center_x, center_y, nan) if the
    search window doesn't fully fit inside the frame."""
    th, tw = template.shape[:2]
    frame_h, frame_w = frame.shape[:2]

    sx0 = int(round(center_x - tw / 2 - search_radius))
    sy0 = int(round(center_y - th / 2 - search_radius))
    sx1 = int(round(center_x + tw / 2 + search_radius))
    sy1 = int(round(center_y + th / 2 + search_radius))
    sx0, sy0 = max(sx0, 0), max(sy0, 0)
    sx1, sy1 = min(sx1, frame_w), min(sy1, frame_h)

    if sx1 - sx0 < tw or sy1 - sy0 < th:
        return center_x, center_y, float("nan")

    search_window = frame[sy0:sy1, sx0:sx1]
    corr_map = cv2.matchTemplate(search_window, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(corr_map)
    px, py = _subpixel_peak(corr_map, max_loc[1], max_loc[0])

    new_x = sx0 + px + tw / 2.0
    new_y = sy0 + py + th / 2.0
    return new_x, new_y, float(max_val)


_ICGN_CACHE_BYTES = 500e6   # budget for the cached Gauss-Newton templates
_ICGN_GROUP = 192   # points refined together (vectorized) in the Gauss-Newton stage


def _track_chunk(context: dict, keys: list[tuple[int, int]]) -> dict:
    """core.parallel.run_map worker for `run_dic`'s per-frame point
    tracking: tracks one chunk of mesh points (`keys`, a list of (iy, ix)
    grid indices) within a single frame, returning `{(iy, ix): (new_x,
    new_y, correlation)}` for that chunk.

    Uses THREADS, not processes (see `run_dic`'s `n_workers` docstring):
    `_track_subset`'s actual hot loop is `cv2.matchTemplate`, a C call
    that releases the GIL, so threads run genuinely concurrently here --
    and avoid re-pickling the (potentially large) frame/template arrays
    into a separate process for every single frame, which a process pool
    would otherwise pay on top of the tracking work itself."""
    frame = context["frame"]
    templates = context["templates"]
    search_x = context["search_x"]
    search_y = context["search_y"]
    search_radius = context["search_radius"]
    return {
        (iy, ix): _track_subset(frame, templates[(iy, ix)], search_x[iy, ix], search_y[iy, ix], search_radius)
        for iy, ix in keys
    }


def _refine_group(context: dict, group: dict) -> dict:
    """core.parallel.run_map worker for the Gauss-Newton refinement stage
    (`subpixel_method="icgn"`): refines, in one vectorized batch, the points
    of `group` that `context["tracked"]` has a usable parabolic estimate for.
    `group` = {"keys": [(iy, ix), ...], "batch": IcgnBatch | None,
    "origins": [(x0, y0), ...]} -- every key has a valid Gauss-Newton
    template; when `batch` is None (memory-bounded mode) it is rebuilt here
    from `origins` and the float32 reference frame.

    Returns {(iy, ix): (new_x, new_y, zncc)} for the points whose refinement
    converged to a match at least as good as the parabolic one; the others
    are left out (the caller keeps their parabolic result)."""
    tracked = context["tracked"]
    rows, u0, v0, ncc0 = [], [], [], []
    for row, key in enumerate(group["keys"]):
        res = tracked.get(key)
        if res is None or not np.isfinite(res[2]):
            continue
        rows.append(row)
        ncc0.append(res[2])
        u0.append(res[0])      # template-centre position for now, made relative below
        v0.append(res[1])
    if not rows:
        return {}
    batch = group["batch"]
    if batch is None:
        tpls = [dic_icgn.prepare_template(context["ref_f"], x0, y0, context["size"])
                for x0, y0 in group["origins"]]
        if any(t is None for t in tpls):
            return {}
        batch = dic_icgn.IcgnBatch(tpls)
    # tracked positions are in pixel-EDGE coordinates (template centre = x0 + size/2);
    # the batch works in array-index coordinates (centre = x0 + (size-1)/2), but
    # displacements are the same in both
    cx_edge = batch.cx[rows] + 0.5
    cy_edge = batch.cy[rows] + 0.5
    du0 = np.array(u0) - cx_edge
    dv0 = np.array(v0) - cy_edge
    u, v, zncc, ok = dic_icgn.refine_batch(context["frame_f"], batch, rows, du0, dv0)
    out = {}
    for i, row in enumerate(rows):
        if ok[i] and zncc[i] >= ncc0[i] - 0.05:
            out[group["keys"][row]] = (cx_edge[i] + u[i], cy_edge[i] + v[i], float(zncc[i]))
    return out


def _grid_strain(xx0: np.ndarray, yy0: np.ndarray, u: np.ndarray, v: np.ndarray):
    """First-order (small-strain) engineering strain field from the
    displacement field, via central differences on the regular mesh grid
    (one-sided at the edges, courtesy of np.gradient). A quick, local
    estimate -- good for spotting where/when strain concentrates -- not a
    substitute for a proper large-deformation DIC strain computation."""
    if u.shape[0] < 2 or u.shape[1] < 2:
        return None, None, None
    dx = np.gradient(xx0, axis=1)
    dy = np.gradient(yy0, axis=0)
    dudx = np.gradient(u, axis=1) / dx
    dudy = np.gradient(u, axis=0) / dy
    dvdx = np.gradient(v, axis=1) / dx
    dvdy = np.gradient(v, axis=0) / dy
    return dudx, dvdy, 0.5 * (dudy + dvdx)  # exx, eyy, exy


def _apply_neighbor_consistency_filter(new_x: np.ndarray, new_y: np.ndarray, corr: np.ndarray,
                                        active: np.ndarray, xx0: np.ndarray, yy0: np.ndarray,
                                        threshold_px: float) -> None:
    """One frame's numerical analog of hourglass control: flags, in
    `active` (permanently, matching `correlation_threshold`'s own
    semantics), any currently-tracked point whose displacement (u, v)
    differs from the median displacement of its immediate (4-connected)
    active neighbors in the mesh grid by more than `threshold_px` pixels.

    Mutates `new_x`/`new_y`/`corr`/`active` in place for THIS frame only
    -- the caller is expected to call this once per frame, after the
    frame's own tracking loop has filled in every still-active point's
    `new_x`/`new_y`/`corr`, and before folding `new_x`/`new_y` into the
    next frame's "moving" search centers."""
    n_rows, n_cols = new_x.shape
    u, v = new_x - xx0, new_y - yy0
    to_drop = []
    for iy in range(n_rows):
        for ix in range(n_cols):
            if not active[iy, ix] or not np.isfinite(u[iy, ix]):
                continue
            neighbor_u, neighbor_v = [], []
            for diy, dix in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = iy + diy, ix + dix
                if 0 <= ny < n_rows and 0 <= nx < n_cols and active[ny, nx] and np.isfinite(u[ny, nx]):
                    neighbor_u.append(u[ny, nx])
                    neighbor_v.append(v[ny, nx])
            if not neighbor_u:
                continue  # an isolated point (every neighbor already excluded) has nothing to compare against
            deviation = float(np.hypot(u[iy, ix] - np.median(neighbor_u), v[iy, ix] - np.median(neighbor_v)))
            if deviation > threshold_px:
                to_drop.append((iy, ix))
    for iy, ix in to_drop:
        active[iy, ix] = False
        new_x[iy, ix] = np.nan
        new_y[iy, ix] = np.nan
        corr[iy, ix] = np.nan


def run_dic(get_preprocessed_frame, mesh: MeshGrid, frame_indices: list[int],
            search_radius: int, tracking_mode: str = "fixed", correlation_threshold: float = 0.0,
            max_neighbor_deviation_px: float = 0.0, progress_cb=None, n_workers: int = 1,
            cancel_check=None, subpixel_method: str = "parabolic") -> DICResult:
    """Tracks every point of `mesh` across `frame_indices` (already
    decimated -- see core/dic_preprocessing.decimated_frame_indices --
    with `frame_indices[0]` the reference/undeformed frame).

    `get_preprocessed_frame(index)` must return the already cropped/
    contrast-stretched/downscaled uint8 frame for that index (see
    core/dic_preprocessing.preprocess_frame) -- this function never reads
    a frame source directly, so the UI and any future batch/CLI use share
    exactly the same preprocessing pipeline.

    `tracking_mode` is "fixed" or "moving" -- see the module docstring for
    what each does to the per-frame search center; anything other than
    "moving" is treated as "fixed". A point whose mesh cell is masked out
    (see `MeshGrid.mask`, e.g. outside a polygon ROI) or whose subset
    doesn't fully fit inside the reference frame is excluded up front --
    NaN in every result field, same as a point that loses lock during
    tracking. `correlation` in each returned DICFrameResult is how a
    caller identifies either case.

    `correlation_threshold`, when > 0, drops a point the first frame its
    peak NCC value falls below it -- NaN in that frame and (since a point
    that's lost is assumed lost for good, not a one-frame glitch) in
    every frame after. Left at its default 0.0, no point is ever dropped
    on quality alone (matching every previous behavior of this function).

    `max_neighbor_deviation_px`, when > 0, additionally drops a point
    (same permanent-loss semantics) the first frame its displacement
    disagrees with the median displacement of its immediate active
    neighbors by more than this many pixels -- see
    `_apply_neighbor_consistency_filter` and the module docstring for
    why. Left at its default 0.0, disabled.

    `n_workers > 1` tracks one frame's active mesh points across several
    threads at once (see `_track_chunk`) -- the frame *sequence* itself
    always stays sequential ("moving" tracking mode centers each frame's
    search on the previous frame's own result, and `active` only ever
    loses points across frames), so this parallelizes within each frame,
    not across frames. Left at its default 1, identical sequential
    behavior to before this setting existed.

    `cancel_check`, when given, is checked once per frame (before that
    frame's tracking starts) and raises `core.parallel.OperationCancelled`
    the first time it returns truthy -- see that exception's own
    docstring for how the UI's "Stop" button reaches it. Cooperative, not
    preemptive: a frame already being tracked always finishes first, so
    cancelling doesn't interrupt mid-frame.

    `subpixel_method` is "parabolic" (default: integer NCC peak + 3-point
    parabola) or "icgn" (the same, then refined by an affine inverse-
    compositional Gauss-Newton fit -- see the module docstring and
    core/dic_icgn.py). With "icgn", `correlation` is the ZNCC of the
    refined match; a point whose refinement is unavailable (flat subset)
    or does not converge to a match at least as good keeps its parabolic
    result."""
    if not frame_indices:
        raise ValueError("No frames to process.")
    if mesh.n_points == 0:
        raise ValueError("The correlation mesh is empty -- check the ROI and subset size.")

    ref_index = frame_indices[0]
    ref_frame = get_preprocessed_frame(ref_index)
    xx0, yy0 = mesh.grid_xy()  # shape (n_points_y, n_points_x), reference positions
    half = mesh.subset_size / 2.0
    n_rows, n_cols = xx0.shape

    use_icgn = (subpixel_method == "icgn")
    templates, origins = {}, {}
    valid = np.ones(xx0.shape, dtype=bool) if mesh.mask is None else mesh.mask.copy()
    for iy in range(n_rows):
        for ix in range(n_cols):
            if not valid[iy, ix]:
                continue
            cx, cy = xx0[iy, ix], yy0[iy, ix]
            x0, y0 = int(round(cx - half)), int(round(cy - half))
            x1, y1 = x0 + mesh.subset_size, y0 + mesh.subset_size
            if x0 < 0 or y0 < 0 or x1 > ref_frame.shape[1] or y1 > ref_frame.shape[0]:
                valid[iy, ix] = False
                continue
            templates[(iy, ix)] = ref_frame[y0:y1, x0:x1]
            origins[(iy, ix)] = (x0, y0)

    if not templates:
        raise ValueError("No correlation point has a subset that fits inside the reference frame.")

    ref_f = None
    icgn_groups = []
    if use_icgn:
        # Gauss-Newton ingredients, grouped by _ICGN_GROUP points so each
        # group is refined as one vectorized batch (and groups can run on
        # separate threads). A flat subset has no usable template: it simply
        # keeps its parabolic result. The stacked steepest-descent images
        # (6 float32 per subset pixel) are cached when they fit a memory
        # budget, otherwise rebuilt for every frame (slower, bounded memory).
        ref_f = ref_frame.astype(np.float32)
        cache = len(templates) * 6 * 4 * mesh.subset_size ** 2 <= _ICGN_CACHE_BYTES
        pending_keys, pending_tpls = [], []

        def _flush():
            if pending_keys:
                icgn_groups.append({
                    "keys": list(pending_keys),
                    "batch": dic_icgn.IcgnBatch(pending_tpls) if cache else None,
                    "origins": [origins[k] for k in pending_keys]})
            pending_keys.clear()
            pending_tpls.clear()

        for key in sorted(templates):
            tpl = dic_icgn.prepare_template(ref_f, origins[key][0], origins[key][1], mesh.subset_size)
            if tpl is None:
                continue
            pending_keys.append(key)
            pending_tpls.append(tpl if cache else None)
            if len(pending_keys) >= _ICGN_GROUP:
                _flush()
        _flush()

    # Measurement-uncertainty ingredients (see core/dic_uncertainty.py) --
    # computed here, once, reusing the same reference-subset patches
    # already cut above, rather than re-reading/re-slicing the frame
    # later. Best-effort: never lets an uncertainty-estimation problem
    # fail an otherwise-successful correlation run.
    uncertainty_error = None
    try:
        noise_sigma = estimate_noise_sigma(ref_frame)
        sssig_x = np.full(xx0.shape, np.nan)
        sssig_y = np.full(xx0.shape, np.nan)
        for (iy, ix), patch in templates.items():
            sssig_x[iy, ix], sssig_y[iy, ix] = subset_sssig_xy(patch)
    except Exception as exc:  # noqa: BLE001 - recorded (uncertainty_error), never fails the run
        noise_sigma, sssig_x, sssig_y = None, None, None
        uncertainty_error = f"{type(exc).__name__}: {exc}"

    # Separate from `valid` (fixed once, before tracking starts): `active`
    # additionally turns False, permanently, the first frame a point's
    # correlation drops below `correlation_threshold` -- so a later frame
    # never searches around (or in "moving" mode, from) an unreliable
    # position. `active` only ever loses points, never regains one.
    active = valid.copy()

    cur_x, cur_y = xx0.astype(float).copy(), yy0.astype(float).copy()
    frame_results = []
    total = len(frame_indices)
    for step_i, frame_index in enumerate(frame_indices):
        check_cancelled(cancel_check)
        frame = ref_frame if frame_index == ref_index else get_preprocessed_frame(frame_index)
        search_x = cur_x if tracking_mode == "moving" else xx0
        search_y = cur_y if tracking_mode == "moving" else yy0
        new_x = np.full(xx0.shape, np.nan)
        new_y = np.full(xx0.shape, np.nan)
        corr = np.full(xx0.shape, np.nan)
        active_keys = [(int(iy), int(ix)) for iy, ix in zip(*np.where(active))]

        if frame_index == ref_index:
            for iy, ix in active_keys:
                new_x[iy, ix], new_y[iy, ix], corr[iy, ix] = xx0[iy, ix], yy0[iy, ix], 1.0
        else:
            if n_workers > 1 and len(active_keys) > 1:
                chunks = chunk_items(active_keys, n_workers)
                chunk_context = {
                    "frame": frame, "templates": templates,
                    "search_x": search_x, "search_y": search_y, "search_radius": search_radius,
                }
                raw = run_map(_track_chunk, chunks, n_workers=n_workers, context=chunk_context, use_threads=True)
                tracked = {}
                for _, chunk_result, err in raw:
                    if err is not None:
                        raise err
                    tracked.update(chunk_result)
            else:
                tracked = {
                    (iy, ix): _track_subset(frame, templates[(iy, ix)], search_x[iy, ix], search_y[iy, ix], search_radius)
                    for iy, ix in active_keys
                }

            if use_icgn and icgn_groups:
                refine_context = {"tracked": tracked, "frame_f": frame.astype(np.float32),
                                  "ref_f": ref_f, "size": mesh.subset_size}
                if n_workers > 1 and len(icgn_groups) > 1:
                    raw = run_map(_refine_group, icgn_groups, n_workers=n_workers,
                                  context=refine_context, use_threads=True)
                    for _, group_result, err in raw:
                        if err is not None:
                            raise err
                        tracked.update(group_result)
                else:
                    for group in icgn_groups:
                        tracked.update(_refine_group(refine_context, group))

            for (iy, ix), (nx, ny, c) in tracked.items():
                if correlation_threshold > 0 and not (c >= correlation_threshold):
                    # `not (c >= threshold)` (rather than `c < threshold`)
                    # also catches c being NaN (search window ran off the
                    # frame entirely) -- either way the point is dropped
                    # from here on, leaving new_x/new_y/corr at NaN for
                    # this frame.
                    active[iy, ix] = False
                    continue
                new_x[iy, ix], new_y[iy, ix], corr[iy, ix] = nx, ny, c

        if max_neighbor_deviation_px > 0 and frame_index != ref_index:
            _apply_neighbor_consistency_filter(new_x, new_y, corr, active, xx0, yy0,
                                                max_neighbor_deviation_px)
        cur_x, cur_y = new_x, new_y

        u, v = new_x - xx0, new_y - yy0
        exx, eyy, exy = _grid_strain(xx0, yy0, u, v)
        frame_results.append(DICFrameResult(frame_index, new_x, new_y, u, v, corr, exx, eyy, exy))

        if progress_cb is not None:
            progress_cb(step_i + 1, total)

    return DICResult(mesh, list(frame_indices), frame_results, noise_sigma, sssig_x, sssig_y,
                      uncertainty_error)
