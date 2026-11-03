"""
Crack-tip propagation tracking along a pre-traced skeleton -- Stage 3 of
the Crack & Wave Analysis standalone tool (see core/crack_wave_io.py for
Stage 1's data loading and core/elastic_wave_speed.py for Stage 2's wave
speed, which this module's frame-timing/linear-fit conventions mirror).

A traced crack (core.crack_skeleton.CrackPath) is a TREE, not a single
polyline -- it can branch, and each branch has its own tip. Tracking
starts from the crack's first-placed node (index 0), taken as the
initiation point (trace from there when drawing a crack for this reason),
and follows the unique path from it to each leaf (endpoint) node
separately -- one propagation history per branch tip.

Three tip-detection methods are offered, per the user's explicit choice to
have more than one available rather than only the recommended one:

- "threshold": walk outward from the origin along the (densely resampled)
  path and take the furthest point the chosen field still exceeds a given
  absolute value (tolerating a few consecutive below-threshold samples,
  so a noisy dip mid-path doesn't truncate the walk early), then linearly
  interpolate the exact crossing between that point and the next one
  (`_interpolate_tip`) rather than reporting the raw resampled-grid point
  -- without this last step the tip can only ever land on one of the
  fixed resampling steps, which looks like an unphysical staircase
  (several frames in a row at the exact same position, then a jump) in
  the propagation-vs-time plot even when the underlying field is
  advancing smoothly -- simple, fast, and robust even on a coarse DIC
  grid, since it never assumes anything about the field's shape near the
  tip. Its weakness, reported directly against real data: a single FIXED
  absolute threshold has to somehow separate "cracked" from "not yet
  cracked" at every single frame at once, even though the field's own
  intensity along the path can drift a lot over the sequence (evolving
  damage, changing contrast/noise floor) -- so a threshold that works for
  the first few frames can be far too high or too low a dozen frames
  later, giving very few detections and a result that swings wildly with
  the threshold value.
- "singular_fit": at each candidate point along the path, fits the field
  sampled in a forward-facing cone/annulus around it against the
  1/sqrt(r) radial decay of the leading-order LEFM near-tip field, and
  keeps the point with the best fit (highest R^2, among those with a
  positive/growing-toward-the-tip slope) as that frame's tip -- more
  physically motivated, but needs enough field resolution around the tip
  to resolve a real 1/sqrt(r) trend, and is more sensitive to noise and
  to the local crack-path tangent (used to orient the forward cone) being
  reasonably accurate.
- "ridge" (recommended): threshold-FREE. Instead of asking "where does
  the field cross a fixed level", it treats the crack tip's near-tip
  field concentration the same way core/elastic_wave_speed.py treats an
  elastic wavefront: a moving, physically-localized peak sweeping along
  the path over time, whose (arc-length position x time) map is a 2D
  "image" a ridge can be traced through -- see `build_tip_field_profile`.
  In the UI, this is surfaced exactly like the wave-speed page's own
  ridge picking: "Show tip field" draws the (arc length x time) heat map,
  then the user draws a ROI polygon loosely around the visible advancing-
  tip band and `extract_tip_ridge_in_polygon` extracts one point per
  frame the polygon spans, at that frame's own strongest local extremum
  inside the polygon -- no arrival threshold, so a faint/drifting-
  intensity trace is exactly as easy to pick as a strong one, which is
  what directly addresses the threshold method's reported instability.
  `track_tip_ridge`, below, is the fully-automatic (no-polygon) version
  of the same idea -- kept in the core API and unit-tested, but not
  wired into the page, in favor of the same user-driven ROI workflow
  already proven out on the wave-speed page.
- "arrival (ROI)" (`extract_tip_ridge_in_polygon`'s sibling,
  `extract_tip_arrival_time_in_polygon`): a second ROI-polygon method,
  offered alongside "ridge" rather than replacing it. Where "ridge" asks
  FRAME BY FRAME "where's the strongest peak right now", this asks
  POSITION BY POSITION along the path "the first time this spot's field
  value ever reaches a fixed absolute `threshold`, when was that" -- then
  connects those (arrival_time, position) pairs in the order they
  occurred. For a field like equivalent (von Mises) strain, where a
  genuinely cracked spot's value jumps to a level clearly separated from
  the undamaged background and STAYS elevated afterward (unlike a
  transient wave), this is naturally close to monotonic by construction:
  a spot further from the origin can only be reached later than one
  closer to it. This is NOT the same algorithm as the old "threshold"
  walk above, and does not share its reported instability: "threshold"
  compared every position WITHIN one frame against a single level, so it
  had to keep working as the field's overall intensity drifted across the
  whole sequence; "arrival" instead looks at each position's own full
  time history once and only asks whether/when it crossed, so a slower
  start or a later intensity ramp elsewhere doesn't affect an already-
  crossed position at all.
MEASUREMENT UNCERTAINTY (optional, "opening_jump" method only): "Estimate
measurement uncertainty" on the Crack Tip Detection card turns on a per-
frame position-uncertainty estimate (`position_uncertainty_px`, from
`opening_extent_uncertainty_px` below) and, downstream, an uncertainty-
WEIGHTED speed fit (`fit_tip_ridge_points`'s `uncertainty_px` argument) --
both entirely built from data this method already computes, so no extra
imaging assumptions (a stated DIC noise floor, a calibration-target
tolerance, ...) are required to get a number, though the more of those a
user can supply, the better that number is. The method, in one place:

1. Per frame, take the ALREADY-COMPUTED `CrackOpeningJumpProfile` (its
   `jump_mag_px` vs `arc_length_s`) and perturb each of this detector's own
   three tunable parameters, ONE AT A TIME (nominal held for the other
   two), re-walking `_threshold_walk`/`_interpolate_tip` (or, for the
   offset, genuinely re-sampling u/v at the perturbed offset -- the only
   one of the three that needs the raw fields again):
   - threshold_px +/- a LOCAL noise estimate (`_local_noise_sigma`,
     Immerkaer-style mean-absolute-consecutive-difference estimator, the
     same family of estimator `core.dic_uncertainty.estimate_noise_sigma`
     uses for images, here applied 1D to `jump_mag_px` in a window around
     the detected front) -- "how much does the jump signal's own
     measurement noise shift the detected front".
   - offset_px +/- 25% (floor 1 px) -- "how much does this detector's own,
     somewhat arbitrary, choice of how far off the crack face to sample
     shift the front".
   - max_gap_samples +/- 1 sample -- "how much does tolerating one more or
     fewer noisy below-threshold sample shift the front".
   Each perturbation's own bracket half-width (max-min of the 2-3
   resulting `opening_extent_s` candidates, /2) is one independent
   contribution; a perturbation that fails to detect any front at all
   (walk never reaches threshold) is simply dropped from that frame's
   total, which then under-reports uncertainty for that one term --
   documented here rather than hidden, since a genuinely undetectable
   perturbed case usually means the nominal detection itself is already
   marginal for that frame.
2. A PATH-DISCRETIZATION floor, independent of the above: the traced
   path's own resample spacing (read back from `np.diff(path_arc_length)`,
   not hardcoded, though it is `resample_step_px` = 2 px by default on
   this page) divided by sqrt(12) -- the standard quantization-noise
   variance of a uniformly-distributed rounding error over one step. This
   is the "image/path resolution" contribution: a crossing can never be
   pinned down finer than the grid it was searched on, however clean the
   signal.
3. All contributions combined in quadrature (independent-error-sum
   assumption -- a simplification, since e.g. the threshold and offset
   perturbations both ultimately act through the same noisy field, but a
   reasonable one for a practical estimate, and always at least as large
   as any single term alone, so never silently optimistic).
4. For VELOCITY: `fit_tip_ridge_points`'s existing plain OLS straight-
   line fit (time vs arc length) becomes an inverse-variance-WEIGHTED
   least-squares fit (weight_i = 1/position_uncertainty_px[i]^2) when
   `uncertainty_px` is given, with a proper standard error on the fitted
   slope from the weighted-regression covariance -- this is where
   RECORDING FREQUENCY enters: more, closer-together points (tighter time
   spacing between accepted frames = the frame rate itself) tighten this
   standard error exactly as more/better-spaced data should, with no
   separate "frame rate uncertainty" input needed for that effect. An
   optional `frame_rate_uncertainty_frac` (the UI's own "Frame rate
   uncertainty (%)" field, 0 by default) additionally folds in a genuine
   camera-timebase tolerance (trigger jitter/drift), as a further relative
   contribution in quadrature -- this one DOES need the user to supply a
   number, since it isn't observable from the trace's own data.

Position uncertainty is reported in px on `PolygonTipRidgeResult.
position_uncertainty_px` (converted to mm only at export/display time, by
the page's own `pixel_size_mm`, same convention as every other px-native
quantity here); speed uncertainty is reported as
`speed_stderr_px_per_s`/`speed_stderr_mm_per_s`. Every one of these is
`None` when the checkbox is off -- "not estimated", never "zero
uncertainty".

- "opening_jump" (`track_tip_opening_jump`, the page's current default and
  primary method): also fully automatic, no polygon to draw. Reported in
  practice against real data: both "ridge (ROI)" and "arrival (ROI)"
  (and the "manual (click)" tool, all three exactly ON the traced path)
  can lose the true crack tip early, because the STRAIN/derived-field
  signal they threshold or peak-pick can fade out, or sit slightly off
  the traced path's exact line, well before the crack has actually
  stopped opening there. This method sidesteps that by measuring the raw
  u/v DISPLACEMENT DISCONTINUITY straight across the path instead (see
  `crack_opening_jump_profile`) -- a genuinely open crack face shows a
  sharp, unambiguous jump even where a derived field's signal has already
  become too faint or offset to threshold/peak-pick reliably. As of this
  version, "ridge (ROI)" and "manual (click)" have been removed from the
  page in favor of this method (their core functions below remain, unit-
  tested, for reference/reuse); "arrival (ROI)" is kept alongside it as
  an alternative when a derived-field crossing is what's actually wanted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from matplotlib.path import Path as _MplPath


class CrackTipTrackingError(ValueError):
    """Raised when a tip track can't be computed as asked -- message is
    shown to the user as-is, so kept short and specific."""


# Smoothing radius / local snap window for `track_tip_ridge`'s per-frame
# peak search -- same role, same values, and same reasoning as
# core/elastic_wave_speed.py's `_POLYGON_PEAK_SMOOTH_RADIUS` /
# `_POLYGON_LOCAL_WINDOW`. Duplicated here (rather than imported) to keep
# this module's dependency on that one to a single, already-existing,
# read-only reuse (see `_smooth_1d` below) -- consistent with how
# `_resolve_times` is also duplicated here instead of imported.
_TIP_RIDGE_SMOOTH_RADIUS = 2
_TIP_RIDGE_LOCAL_WINDOW = 2

# Safety-net floor used ONLY when `extract_tip_ridge_in_polygon`'s
# `enforce_monotonic` filters out a frame's genuinely strong candidates
# (the real tip/face features) for being positioned behind the running
# maximum -- without this, "strongest wins" among whatever's LEFT can
# degrade to picking pure per-pixel noise as that frame's point, since
# nothing else there requires the survivor to be a real feature at all.
# Relative to the frame's OWN strongest enclosed sample (not a fixed
# absolute level), so it doesn't reintroduce the drift-sensitivity the
# threshold method was replaced for.
_TIP_RIDGE_MONOTONIC_MIN_PEAK_FRACTION = 0.15


def _smooth_1d(x: np.ndarray, radius: int) -> np.ndarray:
    """Centered moving-average smoothing, edge-padded so the output stays
    the same length as `x`. Identical in behavior to
    core/elastic_wave_speed.py's helper of the same name -- see that
    module's docstring for why this is needed (finding roughly WHERE a
    real local extremum sits amid per-pixel noise, never used to report a
    position/value itself)."""
    if radius <= 0 or x.size <= 2 * radius:
        return x
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    padded = np.pad(x, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Path geometry
# ---------------------------------------------------------------------------

def branch_paths(crack) -> dict:
    """Every root(node 0)-to-leaf path in `crack`'s tree, as
    {leaf_node_index: [node_index, ...]} (root first, leaf last). A crack
    with fewer than 2 nodes, or whose node 0 has no edges at all, has no
    branch to track and returns `{}`."""
    n = len(crack.nodes)
    if n < 2:
        return {}
    adjacency = {i: [] for i in range(n)}
    for a, b in crack.edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    root = 0
    parent = {root: None}
    frontier = [root]
    while frontier:
        nxt = []
        for node in frontier:
            for nb in adjacency[node]:
                if nb not in parent:
                    parent[nb] = node
                    nxt.append(nb)
        frontier = nxt

    leaves = [i for i in range(n) if i != root and len(adjacency[i]) <= 1]
    paths = {}
    for leaf in leaves:
        if leaf not in parent:
            continue  # not connected to the root -- shouldn't happen from the picker, guarded anyway
        chain = [leaf]
        cur = leaf
        while parent[cur] is not None:
            cur = parent[cur]
            chain.append(cur)
        chain.reverse()
        paths[leaf] = chain
    return paths


def resample_path(points, step: float):
    """Resamples an ordered polyline (list/array of (x, y)) at
    approximately `step`-pixel spacing via linear interpolation along each
    segment. Returns `(positions (N, 2), arc_length (N,))`, arc_length[0]
    == 0 and arc_length[-1] == the path's total length. A single-point
    path is returned as-is (one point, arc_length [0.0])."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts.copy(), np.zeros(len(pts))
    seg_vecs = np.diff(pts, axis=0)
    seg_lens = np.hypot(seg_vecs[:, 0], seg_vecs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    if total <= 0:
        return pts[:1].copy(), np.array([0.0])
    n = max(2, int(np.ceil(total / step)) + 1)
    s_targets = np.linspace(0.0, total, n)
    xs = np.interp(s_targets, cum, pts[:, 0])
    ys = np.interp(s_targets, cum, pts[:, 1])
    return np.column_stack([xs, ys]), s_targets


def path_tangents(positions: np.ndarray, arc_length: np.ndarray | None = None,
                   window_px: float = 0.0) -> np.ndarray:
    """Unit tangent vector at every resampled path point, pointing in the
    direction of increasing arc length -- i.e. "ahead", toward where the
    crack would grow next.

    With the default `window_px=0.0` (or no `arc_length`), this is a plain
    nearest-sample central difference (one-sided at the ends): exact, but
    only as good as the path's own point-to-point direction between two
    samples `resample_step_px` apart. That is fine for display, but a REAL
    auto-traced skeleton (skeletonize -> RDP simplify -> branch trace) is
    only straight at a scale of many pixels -- at the 1-2px scale, pixel-
    grid skeletonization commonly leaves a small staircase zigzag even
    along an overall dead-straight crack, and the nearest-sample tangent
    picks that zigzag up as real angular noise: a synthetic dead-straight
    path with only a +/-1px random walk in the perpendicular direction
    already gives a ~16 deg median (up to 45 deg) tangent error this way.
    core/crack_k1c.py's near-tip K_I/K_II fit is sensitive to exactly this
    -- the Williams/Irwin decomposition is measured relative to this
    tangent (theta=0 is "ahead of the tip"), and a fit that looks
    perfectly clean at 0-10 deg of tangent error is already badly
    corrupted (R^2 collapsing, K_I and K_II mixing into each other) by
    20-30 deg -- squarely inside the zigzag noise band above. This showed
    up in practice as a clean, physically sensible tip-position track
    feeding a K_I(t)/K_II(t) history that made no sense at all.

    Passing `arc_length` (`positions`' own, e.g. a `TipTrackResult.
    path_arc_length`) together with a positive `window_px` instead makes
    each tangent a SECANT over a `window_px`-long stretch of the path
    centered on that point (the two path positions `window_px/2` behind
    and ahead along arc length, each linearly interpolated between the
    nearest resampled points -- clamped to the path's own two ends, and
    falling back to the whole path's end-to-end direction if it's shorter
    than one window). This averages out sub-window zigzag while still
    tracking genuine curvature at the requested length scale -- pick
    `window_px` to match the length scale that actually matters for
    whatever's using the tangent (core/crack_k1c.py ties it to its own
    near-tip sampling radius, since a tangent only needs to be accurate
    over the same region the fit itself samples).

    Degenerate (zero-length) segments/windows fall back to the previous
    point's tangent (or +x for the very first point).

    Public (not `_`-prefixed) because core/crack_k1c.py (Stage 4) reuses
    it to get the local crack-tangent direction at a tracked tip position,
    from the same `path_positions`/`path_arc_length` a `TipTrackResult`
    already carries -- no need to re-derive it from the crack graph a
    second time."""
    n = len(positions)
    tangents = np.zeros((n, 2))

    if arc_length is not None and window_px > 0.0 and n >= 2:
        total = float(arc_length[-1])
        half = window_px / 2.0
        for i in range(n):
            s = float(arc_length[i])
            s_lo = max(0.0, s - half)
            s_hi = min(total, s + half)
            if s_hi - s_lo < 1e-9:
                s_lo, s_hi = 0.0, total
            p_lo = np.array([np.interp(s_lo, arc_length, positions[:, 0]),
                              np.interp(s_lo, arc_length, positions[:, 1])])
            p_hi = np.array([np.interp(s_hi, arc_length, positions[:, 0]),
                              np.interp(s_hi, arc_length, positions[:, 1])])
            d = p_hi - p_lo
            norm = np.hypot(d[0], d[1])
            if norm > 1e-9:
                tangents[i] = d / norm
            elif i > 0:
                tangents[i] = tangents[i - 1]
            else:
                tangents[i] = np.array([1.0, 0.0])
        return tangents

    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n - 1, i + 1)
        d = positions[hi] - positions[lo]
        norm = np.hypot(d[0], d[1])
        if norm > 1e-9:
            tangents[i] = d / norm
        elif i > 0:
            tangents[i] = tangents[i - 1]
        else:
            tangents[i] = np.array([1.0, 0.0])
    return tangents


def smoothed_tip_arc_length_and_positions(times: np.ndarray, arc_length_s: np.ndarray,
                                           path_positions: np.ndarray, path_arc_length: np.ndarray,
                                           window_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Smooths a per-frame tracked tip position (as its arc-length-along-
    the-path coordinate, `arc_length_s` -- the same quantity a
    `TipTrackResult`/`PolygonTipRidgeResult` already reports) with a local
    linear regression against time, then re-projects the smoothed arc
    length back onto the static traced path (`path_positions`/
    `path_arc_length`, e.g. that same track's own) to get a smoothed
    (x, y) position that stays exactly ON the path.

    This targets the same class of problem `path_tangents`'s `window_px`
    already solves for the crack TANGENT direction (see its docstring
    above): a per-frame tracked tip position is only as good as the
    underlying detection method's frame-to-frame noise, and
    core/crack_k1c.py's near-tip K_I/K_II fit uses that position as the
    (r, theta) polar origin of a basis (r^-1/2 near r=0) that is maximally
    sensitive to exactly that noise -- a few pixels of origin jitter can
    swing K_I/K_II by a large fraction even when the tangent is already
    smoothed and the fit's R^2 stays high, because a mislocated origin
    biases the fit rather than merely adding scatter to it.

    Unlike `path_tangents` (a secant over STATIC path geometry), this
    smooths the NOISY TRACKED sequence itself: at every point `i`, all
    valid points within `window_px` of arc length -- converted to a time
    half-width via the track's own robust average speed -- are fit with a
    local straight line (arc length vs time) and evaluated at point `i`'s
    own time. This is a simple moving linear regression (degree-1 local
    polynomial smoothing): it damps point-to-point position noise while
    still tracking genuine acceleration/deceleration of the crack tip,
    unlike a plain moving average which lags behind a (physically
    expected, post-impact) speed ramp.

    "Robust average speed" here is the MEDIAN of point-to-point secant
    speeds (`diff(arc_length_s) / diff(times)` between consecutive valid
    points in time order), not a single end-to-end least-squares slope
    over the whole track. An end-to-end OLS slope would be fragile
    exactly when smoothing is most wanted: a handful of genuinely noisy/
    out-of-order points -- e.g. from the "arrival (ROI)" method at a low
    threshold, where several path positions can cross the level at nearly
    the same time -- has high leverage on a single global fit and can
    drag its slope far off the track's real pace, which would then
    convert `window_px` into a much-too-narrow time window and silently
    produce near-zero smoothing even though the checkbox is on. A median
    of local point-to-point speeds is unmoved by a minority of such
    points (unlike a least-squares slope, which even ONE far-off point
    can dominate), while still tracking the track's actual typical pace
    as well as an end-to-end fit does on a clean, near-constant-speed
    track -- so this does not trade away smoothing quality on well-
    behaved data to fix the noisy case.

    Points with fewer than 2 valid points in their window (not enough to
    fit a line) fall back to their own raw, unsmoothed arc length --
    conservative by construction, never fabricates a trend from too
    little data. Points where `arc_length_s` is not finite (no detection
    that frame) stay NaN in both outputs, matching the input.

    `window_px <= 0` disables smoothing entirely (arc lengths pass
    through unchanged, positions re-derived from `path_positions`/
    `path_arc_length`) -- this function is only called at all when the
    user has opted in via a positive window, but it stays a safe no-op if
    that ever changes.

    Returns `(arc_length_smoothed (N,), positions_smoothed (N, 2))`, same
    length/order as `times`/`arc_length_s`."""
    n = len(arc_length_s)
    arc_smoothed = np.full(n, np.nan)
    valid_idx = np.flatnonzero(np.isfinite(arc_length_s))

    if len(valid_idx) == 0:
        return arc_smoothed, np.full((n, 2), np.nan)

    t_valid = times[valid_idx]
    s_valid = arc_length_s[valid_idx]

    if window_px <= 0.0 or len(valid_idx) < 2:
        arc_smoothed[valid_idx] = s_valid
    else:
        # Robust average speed (px/s): the MEDIAN point-to-point secant
        # speed, not an end-to-end OLS slope -- see the docstring above
        # for why. `times`/`arc_length_s` are always time-sorted ascending
        # here (every caller guarantees it), so consecutive differences
        # are genuine point-to-point steps.
        dt = np.diff(t_valid)
        ds = np.diff(s_valid)
        moved = dt > 0
        speed = float(np.median(ds[moved] / dt[moved])) if np.any(moved) else 0.0

        if abs(speed) < 1e-9:
            # Degenerate (no net advance over the track) -- a pixel window
            # can't be converted to a meaningful time window, so fall back
            # to the raw values rather than guessing.
            arc_smoothed[valid_idx] = s_valid
        else:
            half_t = (window_px / 2.0) / abs(speed)
            for k, i in enumerate(valid_idx):
                ti = t_valid[k]
                in_window = np.abs(t_valid - ti) <= half_t
                if np.count_nonzero(in_window) < 2:
                    arc_smoothed[i] = s_valid[k]
                    continue
                t_win = t_valid[in_window]
                s_win = s_valid[in_window]
                if np.allclose(t_win, t_win[0]):
                    arc_smoothed[i] = s_valid[k]
                    continue
                slope, intercept = np.polyfit(t_win, s_win, 1)
                arc_smoothed[i] = slope * ti + intercept

    positions_smoothed = np.full((n, 2), np.nan)
    finite = np.isfinite(arc_smoothed)
    if np.any(finite):
        positions_smoothed[finite, 0] = np.interp(arc_smoothed[finite], path_arc_length, path_positions[:, 0])
        positions_smoothed[finite, 1] = np.interp(arc_smoothed[finite], path_arc_length, path_positions[:, 1])

    return arc_smoothed, positions_smoothed


# ---------------------------------------------------------------------------
# Field sampling
# ---------------------------------------------------------------------------

def _sample_field(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """Bilinearly interpolates `field` (shape (len(ys), len(xs))) at
    arbitrary (x, y) points -- any point touching a NaN grid cell (outside
    the correlated ROI) comes back NaN, which every caller here already
    treats as "no data" rather than special-casing."""
    interpolator = RegularGridInterpolator((ys, xs), field, bounds_error=False, fill_value=np.nan)
    query = np.column_stack([points_xy[:, 1], points_xy[:, 0]])  # (y, x) order
    return interpolator(query)


# ---------------------------------------------------------------------------
# Shared: frame timing / linear fit (mirrors core/elastic_wave_speed.py)
# ---------------------------------------------------------------------------

def _resolve_times(dataset, frame_rate_hz_override: float | None) -> np.ndarray:
    times = [f.time_s for f in dataset.frames]
    if all(t is not None for t in times):
        return np.asarray(times, dtype=float)
    if frame_rate_hz_override:
        return np.asarray([f.frame_index / frame_rate_hz_override for f in dataset.frames], dtype=float)
    raise CrackTipTrackingError(
        "No frame timing available: the loaded CSV has no time_s values (the Image Correlation "
        "page's frame rate wasn't set when it was exported) -- enter a frame rate override to use "
        "frame_index / rate instead."
    )


def _frame_order(dataset, frame_rate_hz_override, frame_range):
    times_all = _resolve_times(dataset, frame_rate_hz_override)
    order = np.argsort(times_all)
    if frame_range is not None:
        start, end = frame_range
        start = max(0, int(start))
        end = min(len(order) - 1, int(end))
        if end <= start:
            raise CrackTipTrackingError("The selected frame range is empty or invalid.")
        order = order[start:end + 1]
    if len(order) < 2:
        raise CrackTipTrackingError("Need at least 2 frames to track crack-tip propagation.")
    return order, times_all[order]


def _linear_fit_r2(t: np.ndarray, s: np.ndarray):
    """Fits s = speed * t + intercept by least squares; returns (speed,
    intercept, r_squared). Raises CrackTipTrackingError on degenerate
    input (fewer than 2 points, or every point at the same time)."""
    if len(t) < 2:
        raise CrackTipTrackingError("Fewer than 2 valid tip detections -- can't fit a propagation speed.")
    if np.allclose(t, t[0]):
        raise CrackTipTrackingError("Every detected tip position happened in the same frame -- can't fit a speed.")
    slope, intercept = np.polyfit(t, s, 1)
    predicted = slope * t + intercept
    ss_res = float(np.sum((s - predicted) ** 2))
    ss_tot = float(np.sum((s - np.mean(s)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r_squared)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class TipTrackResult:
    leaf_index: int
    method: str              # "threshold" | "singular_fit"
    field_key: str
    times: np.ndarray        # (n_frames_used,) seconds, sorted ascending
    arc_length_s: np.ndarray # (n_frames_used,) pixels along the path; NaN where no tip was found that frame
    positions_xy: np.ndarray # (n_frames_used, 2) pixel coordinates; NaN row where no tip was found
    path_positions: np.ndarray  # (n_path_points, 2) the full resampled path, for plotting context
    path_arc_length: np.ndarray  # (n_path_points,) matching path_positions
    source_frame_positions: np.ndarray  # (n_frames_used,) int index into `dataset.frames` used at
                                         # each times[i]/arc_length_s[i]/positions_xy[i] -- lets
                                         # core/crack_k1c.py (Stage 4) look up the exact same frame's
                                         # displacement (u, v) field a tip position came from, without
                                         # re-deriving frame ordering/range a second time
    speed_px_per_s: float | None
    speed_mm_per_s: float | None
    intercept_px: float | None
    r_squared: float | None
    n_points_used: int


def _finalize(leaf_index, method, field_key, times, arc_length_s, positions_xy,
              path_positions, path_arc_length, source_frame_positions, pixel_size_mm):
    valid = np.isfinite(arc_length_s)
    n_valid = int(np.count_nonzero(valid))
    speed = intercept = r_squared = None
    if n_valid >= 2:
        try:
            speed, intercept, r_squared = _linear_fit_r2(times[valid], arc_length_s[valid])
        except CrackTipTrackingError:
            pass
    speed_mm = speed * pixel_size_mm if (speed is not None and pixel_size_mm) else None
    return TipTrackResult(
        leaf_index=leaf_index, method=method, field_key=field_key, times=times,
        arc_length_s=arc_length_s, positions_xy=positions_xy,
        path_positions=path_positions, path_arc_length=path_arc_length,
        source_frame_positions=source_frame_positions,
        speed_px_per_s=speed, speed_mm_per_s=speed_mm, intercept_px=intercept,
        r_squared=r_squared, n_points_used=n_valid,
    )


# ---------------------------------------------------------------------------
# Method A: threshold walk
# ---------------------------------------------------------------------------

def _threshold_walk(values: np.ndarray, threshold: float, max_gap: int):
    """Walks `values` from index 0 outward, returning the furthest index
    still "recently" at/above `threshold` -- tolerating up to `max_gap`
    consecutive below-threshold samples anywhere along the walk,
    INCLUDING at the very start (index 0 itself).

    Index 0 gets the SAME gap tolerance as every other point along the
    walk, rather than being hard-fenced (requiring the very first sample
    to already be at/above threshold, on pain of discarding the WHOLE
    frame). Index 0 is exactly the wrong place to be strict: large local
    deformation gradients and subset decorrelation during fast/dynamic
    crack opening are often worst right at/behind the tip, i.e. right
    where the path's origin sits once the crack has genuinely opened
    there, so a purely transient correlation blip there could otherwise
    silently drop an entire frame's detection even while the crack is
    visibly, continuously propagating in the raw images (one or two noisy
    frames on an otherwise-open origin, surrounded by valid ones, would
    then show up as a multi-frame GAP in the reported track, not merely a
    truncated one). This still returns None in the genuine "nothing open
    near the origin" case -- if threshold is never reached within
    `max_gap` samples of the start, `idx` never gets set and the loop
    breaks with it still None."""
    finite = np.where(np.isfinite(values), values, -np.inf)
    idx = None
    gap = 0
    for i in range(len(finite)):
        if finite[i] >= threshold:
            idx, gap = i, 0
        else:
            gap += 1
            if gap > max_gap:
                break
    return idx


def _interpolate_tip(values: np.ndarray, positions: np.ndarray, arc_length: np.ndarray,
                      idx: int, threshold: float):
    """Refines `_threshold_walk`'s discrete `idx` (the last resampled path
    point still at/above `threshold`) to a continuous arc-length position,
    by linearly interpolating the exact threshold crossing between `idx`
    and the very next resampled point -- the same "don't round to the
    nearest sample when a straight line through the two gives a cheap,
    meaningfully more precise answer" convention already used for wave
    arrival picking (core/elastic_wave_speed.py's `build_arrival_time_
    profile`). Without this, the reported tip position can only ever land
    on one of the fixed `resample_step_px` grid points, so a crack
    advancing by less than one step between frames reports the SAME
    position for several consecutive frames -- an unphysical staircase in
    the propagation-vs-time plot, even though the underlying field itself
    advances continuously. Falls back to the un-refined point when there's
    no next sample to interpolate against (the path's own end) or that
    next sample isn't finite (outside the correlated ROI)."""
    n = len(values)
    if idx + 1 >= n:
        return positions[idx], float(arc_length[idx])
    v0, v1 = values[idx], values[idx + 1]
    if not np.isfinite(v0) or not np.isfinite(v1) or v1 == v0:
        return positions[idx], float(arc_length[idx])
    frac = (threshold - v0) / (v1 - v0)
    frac = min(1.0, max(0.0, frac))
    interp_pos = positions[idx] + frac * (positions[idx + 1] - positions[idx])
    interp_s = float(arc_length[idx] + frac * (arc_length[idx + 1] - arc_length[idx]))
    return interp_pos, interp_s


def _nanmedian_filter_1d(values: np.ndarray, window_samples: int) -> np.ndarray:
    """Rolling median along a 1D array, one window centered on (as closely
    as an integer division allows) each sample, ignoring non-finite
    values within the window rather than propagating them -- so a single
    NaN (a path point outside the correlated DIC ROI, see `_sample_field`)
    only removes itself from its neighbors' windows, not every window it
    touches. Returns `values` itself (copied, not aliased) unchanged for
    `window_samples <= 1` -- "no smoothing" is a true no-op, not a
    1-sample median of itself (which would be the same thing numerically,
    but this skips the loop below entirely rather than relying on that
    coincidence).

    Purpose: `crack_opening_jump_profile` uses this to suppress narrow,
    high-amplitude spikes in `jump_mag_px` -- e.g. the transient
    displacement disturbance an elastic wave passing through the DIC
    field of view leaves behind, a handful of samples wide and gone again
    a few samples later -- while leaving a genuinely SUSTAINED opening
    plateau (many samples wide by construction, since a real crack stays
    open at a given path point in every later sample too) essentially
    untouched. A median (not a mean) specifically because it rejects an
    isolated outlier sample outright rather than blending it in --
    exactly the "narrow spike vs. sustained plateau" distinction this is
    for. `window_samples` should be picked a bit larger than the
    wave-pulse's own visible spatial width in the "Verify visually" jump
    profile plot, but well short of the shortest genuine opening extent
    expected -- too large a window would itself start eating into a
    real, short opening near the crack's own initiation."""
    if window_samples <= 1:
        return values.copy()
    n = len(values)
    half = window_samples // 2
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = values[lo:hi]
        finite = window[np.isfinite(window)]
        if finite.size > 0:
            out[i] = np.median(finite)
    return out


def track_tip_threshold(dataset, crack, leaf_index: int, field_key: str, threshold: float,
                         frame_range: tuple | None = None, frame_rate_hz_override: float | None = None,
                         resample_step_px: float = 2.0, max_gap_samples: int = 2,
                         pixel_size_mm: float | None = None) -> TipTrackResult:
    paths = branch_paths(crack)
    if leaf_index not in paths:
        raise CrackTipTrackingError(f"No traced path from the crack's origin (its first point) to node {leaf_index}.")
    node_points = [crack.nodes[i] for i in paths[leaf_index]]
    positions, arc_length = resample_path(node_points, resample_step_px)

    order, times = _frame_order(dataset, frame_rate_hz_override, frame_range)
    tip_xy = np.full((len(order), 2), np.nan)
    tip_s = np.full(len(order), np.nan)
    for fi, oi in enumerate(order):
        field = dataset.frames[oi].field(field_key)
        values = _sample_field(field, dataset.xs, dataset.ys, positions)
        idx = _threshold_walk(values, threshold, max_gap_samples)
        if idx is not None:
            tip_xy[fi], tip_s[fi] = _interpolate_tip(values, positions, arc_length, idx, threshold)

    return _finalize(leaf_index, "threshold", field_key, times, tip_s, tip_xy,
                      positions, arc_length, order, pixel_size_mm)


def suggest_threshold(dataset, crack, leaf_index: int, field_key: str, percentile: float = 50.0,
                       frame_range: tuple | None = None, frame_rate_hz_override: float | None = None,
                       resample_step_px: float = 2.0) -> float:
    """A reasonable starting absolute threshold for `track_tip_threshold`:
    the given percentile of every value the chosen field takes along this
    specific crack's path, pooled across every frame in range -- centered
    on this crack's own data rather than a generic field-wide statistic,
    since what counts as "cracked" can vary a lot between specimens/
    materials. Returns 0.0 if nothing finite was sampled."""
    paths = branch_paths(crack)
    if leaf_index not in paths:
        raise CrackTipTrackingError(f"No traced path from the crack's origin (its first point) to node {leaf_index}.")
    node_points = [crack.nodes[i] for i in paths[leaf_index]]
    positions, _ = resample_path(node_points, resample_step_px)
    order, _ = _frame_order(dataset, frame_rate_hz_override, frame_range)
    pooled = []
    for oi in order:
        field = dataset.frames[oi].field(field_key)
        values = _sample_field(field, dataset.xs, dataset.ys, positions)
        finite = values[np.isfinite(values)]
        if finite.size:
            pooled.append(finite)
    if not pooled:
        return 0.0
    return float(np.percentile(np.concatenate(pooled), percentile))


# ---------------------------------------------------------------------------
# Method B: singular near-tip field fit
# ---------------------------------------------------------------------------

def _singular_fit_quality(field, xs, ys, origin_xy, tangent_xy, r_min, r_max,
                           half_angle_deg, n_r, n_theta):
    """Samples `field` in a forward-facing cone/annulus ahead of
    `origin_xy` (between r_min and r_max, within +/- half_angle_deg of
    `tangent_xy`) and fits value ~ a + b / sqrt(r) -- the leading-order
    radial decay of the LEFM near-tip field. Returns (r_squared, slope);
    (-inf, 0.0) if there weren't enough valid (non-NaN) samples to fit."""
    radii = np.linspace(r_min, r_max, n_r)
    thetas = np.radians(np.linspace(-half_angle_deg, half_angle_deg, n_theta))
    tx, ty = tangent_xy
    nx, ny = -ty, tx  # unit normal, 90 deg from the tangent

    pts = []
    inv_sqrt_r = []
    for r in radii:
        for theta in thetas:
            c, s = np.cos(theta), np.sin(theta)
            dx = r * (c * tx + s * nx)
            dy = r * (c * ty + s * ny)
            pts.append((origin_xy[0] + dx, origin_xy[1] + dy))
            inv_sqrt_r.append(1.0 / np.sqrt(r))
    pts = np.asarray(pts)
    inv_sqrt_r = np.asarray(inv_sqrt_r)

    values = _sample_field(field, xs, ys, pts)
    valid = np.isfinite(values)
    if np.count_nonzero(valid) < 4:
        return -np.inf, 0.0

    x = inv_sqrt_r[valid]
    y = values[valid]
    # A plain closed-form OLS line fit instead of np.polyfit: `x` here only
    # takes `n_r` distinct values (repeated `n_theta` times each, one per
    # angular sample), which is a legitimate design (pooling several
    # azimuths per radius) but makes polyfit's internal Vandermonde-
    # conditioning check fire a RankWarning on every single call -- benign
    # for a degree-1 fit, but not worth a warning per candidate point per
    # frame.
    x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
    dx = x - x_mean
    denom = float(np.sum(dx ** 2))
    slope = float(np.sum(dx * (y - y_mean)) / denom) if denom > 0 else 0.0
    intercept = y_mean - slope * x_mean
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return r_squared, float(slope)


def track_tip_singular_fit(dataset, crack, leaf_index: int, field_key: str, r_min_px: float, r_max_px: float,
                            forward_half_angle_deg: float = 60.0, min_r_squared: float = 0.5,
                            frame_range: tuple | None = None, frame_rate_hz_override: float | None = None,
                            resample_step_px: float = 4.0, n_radial_samples: int = 6, n_angular_samples: int = 5,
                            pixel_size_mm: float | None = None) -> TipTrackResult:
    if r_max_px <= r_min_px:
        raise CrackTipTrackingError("The search radius's max must be larger than its min.")
    paths = branch_paths(crack)
    if leaf_index not in paths:
        raise CrackTipTrackingError(f"No traced path from the crack's origin (its first point) to node {leaf_index}.")
    node_points = [crack.nodes[i] for i in paths[leaf_index]]
    positions, arc_length = resample_path(node_points, resample_step_px)
    tangents = path_tangents(positions)

    order, times = _frame_order(dataset, frame_rate_hz_override, frame_range)
    tip_xy = np.full((len(order), 2), np.nan)
    tip_s = np.full(len(order), np.nan)
    for fi, oi in enumerate(order):
        field = dataset.frames[oi].field(field_key)
        best_i, best_quality = None, -np.inf
        for i in range(len(positions)):
            quality, slope = _singular_fit_quality(
                field, dataset.xs, dataset.ys, positions[i], tangents[i],
                r_min_px, r_max_px, forward_half_angle_deg, n_radial_samples, n_angular_samples,
            )
            if slope > 0 and quality > best_quality:
                best_quality, best_i = quality, i
        if best_i is not None and best_quality >= min_r_squared:
            tip_xy[fi] = positions[best_i]
            tip_s[fi] = arc_length[best_i]

    return _finalize(leaf_index, "singular_fit", field_key, times, tip_s, tip_xy,
                      positions, arc_length, order, pixel_size_mm)


# ---------------------------------------------------------------------------
# Method C: ridge tracking in (arc length x time) -- threshold-free, per-
# frame-normalized, monotonic. See the module docstring's third bullet for
# why this exists: real data showed the threshold method giving very few,
# highly threshold-sensitive detections, because one fixed absolute level
# can't track a field whose own intensity drifts over the sequence.
# ---------------------------------------------------------------------------

@dataclass
class TipFieldProfile:
    """The chosen field, sampled along a crack's traced path at every
    resampled point and every frame in range -- the crack-tracking
    equivalent of core/elastic_wave_speed.py's `ArrivalTimeProfile.signal`,
    just parameterized by arc length along a (possibly curved) traced path
    instead of a straight x or y axis. `values[:, j]` is one frame's
    "profile along the path"; `values[i, :]` is one path point's value
    over time."""
    path_positions: np.ndarray  # (n_path_points, 2) resampled (x, y), root first
    path_arc_length: np.ndarray  # (n_path_points,) matching path_positions
    values: np.ndarray          # (n_path_points, n_frames_used) field value, NaN outside the correlated ROI
    times: np.ndarray           # (n_frames_used,) seconds, sorted ascending
    frame_indices: np.ndarray   # (n_frames_used,) int index into `dataset.frames` each times[j]/
                                 # values[:, j] column came from -- lets a caller (K1c reuse, ROI
                                 # point extraction) look up the exact same frame's other fields
                                 # without re-deriving frame ordering/range a second time


def build_tip_field_profile(dataset, crack, leaf_index: int, field_key: str,
                             frame_range: tuple | None = None, frame_rate_hz_override: float | None = None,
                             resample_step_px: float = 2.0, reverse_path: bool = False) -> TipFieldProfile:
    """Builds the (arc length x time) field map the page's "Show tip
    field" displays as a heat map, and `extract_tip_ridge_in_polygon` (or
    `track_tip_ridge`) extracts a ridge from -- the same role
    `build_arrival_time_profile` plays for the wave-speed page.

    Arc length 0 is, by default, wherever `crack`'s first-placed node is
    (see the module docstring) -- the crack's true initiation point ONLY
    if that happened to be the end traced/auto-detected first, which
    isn't guaranteed (a crack traced from its current tip back toward its
    origin has arc length running the opposite way, and no amount of
    downstream direction-guessing on the FIELD DATA can always tell the
    two apart on noisy real data -- e.g. a strong, roughly-uniform
    residual signal left behind a crack that has already passed through
    much of the traced region can dominate a per-frame or overall trend
    estimate). `reverse_path=True` flips which end is treated as arc
    length 0 -- a direct, user-controlled override for when the plotted
    axis is visibly running the wrong way (e.g. showing the largest arc
    length at the crack's actual origin) -- by reversing the traced node
    order before resampling; nothing downstream needs to change, since
    everything else (ridge picking, the monotonic-advance direction,
    K1c's tangent orientation) is derived from `path_positions`/
    `path_arc_length` alone."""
    paths = branch_paths(crack)
    if leaf_index not in paths:
        raise CrackTipTrackingError(f"No traced path from the crack's origin (its first point) to node {leaf_index}.")
    node_points = [crack.nodes[i] for i in paths[leaf_index]]
    if reverse_path:
        node_points = list(reversed(node_points))
    positions, arc_length = resample_path(node_points, resample_step_px)

    order, times = _frame_order(dataset, frame_rate_hz_override, frame_range)
    values = np.full((len(positions), len(order)), np.nan)
    for fi, oi in enumerate(order):
        field = dataset.frames[oi].field(field_key)
        values[:, fi] = _sample_field(field, dataset.xs, dataset.ys, positions)
    return TipFieldProfile(path_positions=positions, path_arc_length=arc_length, values=values, times=times,
                            frame_indices=np.asarray(order, dtype=int))


# ---------------------------------------------------------------------------
# Manual (click) tip tracking -- an independent, no-algorithm control method:
# the user is shown each frame's own raw 2D field (not the (arc length x
# time) map above) and clicks directly on the visible tip, one frame at a
# time, with the page auto-advancing to the next frame after every click.
# Deliberately does NOT reuse `build_tip_field_profile` (that function's
# whole point is pre-interpolating the field ONTO the path for every frame
# at once, which this tool has no use for -- it shows each frame's own
# field directly and only needs the path for click-snapping) -- this
# section supplies just the two things `build_tip_field_profile` and this
# tool DO share: the same traced-path resample, and the same frame walk
# order/timing, so a manually-clicked trace lines up exactly with every
# other method's path/time convention and can be compared point for point.
# ---------------------------------------------------------------------------

@dataclass
class ManualTipPickContext:
    """Everything the manual tip-pointing tool needs to walk a dataset's
    frames and snap each click onto the traced path -- built once by
    `build_manual_tip_pick_context` when the user starts a pointing
    session, then reused for every frame/click in that session."""
    path_positions: np.ndarray   # (n_path_points, 2) resampled (x, y), same convention as TipFieldProfile
    path_arc_length: np.ndarray  # (n_path_points,) matching path_positions
    frame_indices: np.ndarray    # (n_frames_used,) int index into `dataset.frames`, walk order
    times: np.ndarray            # (n_frames_used,) seconds, matching frame_indices


def build_manual_tip_pick_context(dataset, crack, leaf_index: int,
                                   frame_range: tuple | None = None,
                                   frame_rate_hz_override: float | None = None,
                                   resample_step_px: float = 2.0,
                                   reverse_path: bool = False) -> ManualTipPickContext:
    """The traced-path geometry and frame walk order/timing for a manual
    tip-pointing session -- the same path resample and frame ordering
    `build_tip_field_profile` uses (see its docstring for what
    `reverse_path` does and why arc length 0's location isn't always the
    crack's true origin), so a manually-clicked trace is directly
    comparable to every automatic method's. Raises `CrackTipTrackingError`
    under the same conditions `build_tip_field_profile` does (bad leaf,
    empty/invalid frame range, no frame timing available)."""
    paths = branch_paths(crack)
    if leaf_index not in paths:
        raise CrackTipTrackingError(f"No traced path from the crack's origin (its first point) to node {leaf_index}.")
    node_points = [crack.nodes[i] for i in paths[leaf_index]]
    if reverse_path:
        node_points = list(reversed(node_points))
    positions, arc_length = resample_path(node_points, resample_step_px)
    order, times = _frame_order(dataset, frame_rate_hz_override, frame_range)
    return ManualTipPickContext(path_positions=positions, path_arc_length=arc_length,
                                 frame_indices=np.asarray(order, dtype=int), times=times)


def nearest_path_point(path_positions: np.ndarray, path_arc_length: np.ndarray,
                        x: float, y: float) -> tuple[float, float, float, float]:
    """Snaps an arbitrary (x, y) click onto the nearest sample of an
    already-resampled path (plain nearest-point search -- `path_positions`
    is resampled at `resample_step_px`, typically 2 px, so this is
    accurate to about that spacing, the same precision every automatic
    method here already works at). Returns `(snapped_x, snapped_y,
    arc_length_px, distance_px)` -- `distance_px` is the raw click-to-path
    distance, for the caller to flag a click that landed suspiciously far
    from the path (a mis-click, or a path that doesn't actually reach the
    visible tip)."""
    d2 = (path_positions[:, 0] - x) ** 2 + (path_positions[:, 1] - y) ** 2
    i = int(np.argmin(d2))
    return (float(path_positions[i, 0]), float(path_positions[i, 1]),
            float(path_arc_length[i]), float(np.sqrt(d2[i])))


def path_polyline_up_to_arc_length(path_positions: np.ndarray, path_arc_length: np.ndarray,
                                    max_s: float) -> np.ndarray:
    """The portion of the resampled traced-path polyline from its own
    arc-length 0 up to `max_s`, inclusive -- i.e. the crack's REAL,
    possibly-curved traced shape covered so far, never a straight line
    between two points. An exact point is interpolated at `max_s` itself
    (appended if it doesn't already land on a resampled sample) so the
    sub-polyline's own tip lands exactly where the caller asked, not at
    the nearest coarser `resample_step_px` sample short of it.

    This exists for drawing a "trajectory so far" overlay (see
    core/tip_video_export.py's top panel and the Crack & Wave Analysis
    page's "Verify visually" viewer) that stays glued to the actual
    traced path even when the underlying tip detections are TEMPORALLY
    sparse -- e.g. the crack accelerates and several frames pass between
    two consecutive detected points (nothing wrong with that; a real
    crack does speed up). Connecting those two detected (x, y) points
    with a straight chord, as a naive "just plot trace.positions_xy" would,
    visibly cuts across the path's real curvature -- often leaving the
    drawn line off the crack entirely at a bend. That is purely a
    RENDERING artifact: the path's geometry between those two points is
    already fully known from tracing/auto-detection (see this module's
    docstring on `resample_path`, and `track_tip_opening_jump`'s own use
    of it) and every position/velocity/K1c computation here already
    measures `arc_length` along that same real polyline, not a straight-
    line distance -- only the DISPLAY of "how far along it we've gotten"
    was, until this helper, drawn wrong.

    `max_s` is clamped into [path_arc_length[0], path_arc_length[-1]].
    Returns an (M, 2) array, M >= 1."""
    path_positions = np.asarray(path_positions, dtype=float)
    path_arc_length = np.asarray(path_arc_length, dtype=float)
    s0, s1 = float(path_arc_length[0]), float(path_arc_length[-1])
    max_s = min(max(float(max_s), s0), s1)
    idx = int(np.searchsorted(path_arc_length, max_s, side="right"))
    kept = path_positions[:idx]
    x = float(np.interp(max_s, path_arc_length, path_positions[:, 0]))
    y = float(np.interp(max_s, path_arc_length, path_positions[:, 1]))
    if kept.shape[0] == 0 or not np.allclose(kept[-1], (x, y)):
        kept = np.vstack([kept, [[x, y]]])
    return kept


def finalize_manual_tip_pick(ctx: ManualTipPickContext, dataset, leaf_index: int, field_key: str,
                              picks: dict, label: str, color: str,
                              pixel_size_mm: float | None = None) -> "PolygonTipRidgeResult | None":
    """Turns the raw clicks collected by a manual tip-pointing session
    into a `PolygonTipRidgeResult` tagged `method="manual (click)"` --
    exactly the same result type `extract_tip_ridge_in_polygon` produces,
    so a manually-clicked trace slots into every downstream use (the
    comparison plot, K1c, XFEM-DIC's trace source, CSV export) without
    those needing any special case for it (see `PolygonTipRidgeResult`'s
    docstring).

    `picks` is `{position: (x, y, arc_length_px), ...}`, where `position`
    is an index into `ctx.frame_indices`/`ctx.times` (one entry per frame
    actually clicked -- a skipped frame simply has no entry, left out of
    the result entirely rather than kept as a NaN gap, since "the user
    didn't click here" and "the tip briefly vanished" aren't the same
    thing an automatic method's NaN convention represents). Returns `None`
    if fewer than 2 frames were picked (nothing usable to finalize -- a
    single point has no speed and isn't worth keeping as a trace).

    Each picked point's field VALUE (kept only for plotting/CSV context,
    never used to determine the position itself -- that came from the
    click) is bilinearly re-sampled from that frame's own field grid,
    the same way `build_tip_field_profile` samples the path."""
    positions_used = sorted(picks)
    if len(positions_used) < 2:
        return None
    times = ctx.times[positions_used]
    source_frame_positions = ctx.frame_indices[positions_used]
    arc_length = np.array([picks[p][2] for p in positions_used], dtype=float)
    xy = np.array([[picks[p][0], picks[p][1]] for p in positions_used], dtype=float)
    values = np.full(len(positions_used), np.nan)
    for k, frame_idx in enumerate(source_frame_positions):
        field = dataset.frames[int(frame_idx)].field(field_key)
        values[k] = float(_sample_field(field, dataset.xs, dataset.ys, xy[k:k + 1])[0])
    speed_px, speed_mm, r_squared, _stderr_px, _stderr_mm = fit_tip_ridge_points(
        times, arc_length, pixel_size_mm=pixel_size_mm
    )
    return PolygonTipRidgeResult(
        label=label, color=color, leaf_index=leaf_index, field_key=field_key,
        vertices=[], times=times, arc_length=arc_length, positions_xy=xy, values=values,
        source_frame_positions=source_frame_positions, speed_px_per_s=speed_px,
        speed_mm_per_s=speed_mm, r_squared=r_squared, method="manual (click)",
    )


def track_tip_ridge(dataset, crack, leaf_index: int, field_key: str,
                     frame_range: tuple | None = None, frame_rate_hz_override: float | None = None,
                     resample_step_px: float = 2.0, min_peak_fraction: float = 0.15,
                     backslide_tolerance_px: float | None = None,
                     pixel_size_mm: float | None = None) -> TipTrackResult:
    """Threshold-free crack-tip tracking: treats the near-tip field
    concentration as a moving peak in the (arc length, time) map built by
    `build_tip_field_profile`, and traces it the same way
    core/elastic_wave_speed.py's `extract_ridge_in_polygon` traces a wave
    band -- smoothed local-extremum search, strongest-candidate wins, snap
    to the raw peak, tight same-sign centroid refinement -- with two
    differences suited specifically to a crack tip rather than a wave:

    1. No polygon, and no fixed threshold: every frame's candidates are
       filtered against a level set RELATIVE TO THAT FRAME's OWN peak
       (`min_peak_fraction`, plus a robust global noise floor from the
       whole array's median/MAD, the same style of guard used in
       core/elastic_wave_speed.py's `detect_ridge_points`), not one
       absolute number shared across every frame. This is the direct fix
       for the threshold method's reported failure: a fixed absolute
       threshold has to work for every frame's intensity at once, but real
       data drifts (evolving damage, changing noise floor) -- a per-frame-
       relative level tracks that drift instead of being defeated by it.
    2. A monotonic-advance constraint: a physical crack tip can only ever
       move forward along the path (or hold still), never recede, so a
       frame's candidate is only eligible if its arc-length position isn't
       more than `backslide_tolerance_px` behind the largest position
       already accepted in an earlier frame. Unlike a wave (which
       legitimately can reverse direction, e.g. a reflection), this is a
       genuine, well-justified physical prior for a growing crack, and is
       what lets "strongest candidate wins" (deliberately kept simple --
       see `extract_ridge_in_polygon`'s docstring for why fancier tie-
       breaks there backfired) stay locked onto the real, advancing tip
       instead of occasionally latching onto a stronger but stationary
       feature further back along the path (e.g. the fully-open crack
       mouth, which can show a larger absolute value long after the tip
       itself has moved on). A frame where every surviving candidate would
       require accepting a backward step is skipped (no point reported)
       rather than forcing a physically impossible answer.

    A frame with no sample clearing the floor at all contributes no point
    -- gaps in the trace are expected (e.g. before the crack has reached
    this branch at all) and are left as NaN, exactly like the other two
    methods.
    """
    profile = build_tip_field_profile(
        dataset, crack, leaf_index, field_key, frame_range, frame_rate_hz_override, resample_step_px,
    )
    positions, arc_length, values, times = (
        profile.path_positions, profile.path_arc_length, profile.values, profile.times,
    )
    n_points, n_frames = values.shape
    if backslide_tolerance_px is None:
        backslide_tolerance_px = 3.0 * resample_step_px

    finite_all = values[np.isfinite(values)]
    if finite_all.size:
        med = float(np.median(finite_all))
        mad = float(np.median(np.abs(finite_all - med)))
        noise_sigma = mad / 0.6745 if mad > 0 else 0.0
    else:
        noise_sigma = 0.0

    tip_xy = np.full((n_frames, 2), np.nan)
    tip_s = np.full(n_frames, np.nan)
    running_max_s = -np.inf

    for fi in range(n_frames):
        col = values[:, fi]
        finite_mask = np.isfinite(col)
        if not np.any(finite_mask):
            continue
        idxs = np.nonzero(finite_mask)[0]
        col_f = col[idxs]
        abs_col = np.abs(col_f)
        n_in = idxs.size
        frame_peak = float(np.max(abs_col))
        if not np.isfinite(frame_peak) or frame_peak <= 0:
            continue
        floor = max(min_peak_fraction * frame_peak, 4.0 * noise_sigma)

        if n_in <= 2:
            candidates = list(range(n_in))
        else:
            abs_smooth = _smooth_1d(abs_col, min(_TIP_RIDGE_SMOOTH_RADIUS, (n_in - 1) // 2))
            raw_candidates = [
                i for i in range(1, n_in - 1)
                if abs_smooth[i] >= abs_smooth[i - 1] and abs_smooth[i] >= abs_smooth[i + 1]
            ]
            candidates = []
            run: list[int] = []
            for i in raw_candidates:
                if run and i != run[-1] + 1:
                    candidates.append(max(run, key=lambda k: abs_col[k]))
                    run = []
                run.append(i)
            if run:
                candidates.append(max(run, key=lambda k: abs_col[k]))
            if not candidates:
                candidates = [int(np.argmax(abs_col))]

        candidates = [c for c in candidates if abs_col[c] >= floor]
        if not candidates:
            continue

        eligible = [c for c in candidates if arc_length[idxs[c]] >= running_max_s - backslide_tolerance_px]
        if not eligible:
            continue  # every surviving candidate would mean the tip receding -- distrust this frame

        peak_local = max(eligible, key=lambda c: abs_col[c])

        lo = max(0, peak_local - _TIP_RIDGE_LOCAL_WINDOW)
        hi = min(n_in, peak_local + _TIP_RIDGE_LOCAL_WINDOW + 1)
        peak_local = lo + int(np.argmax(abs_col[lo:hi]))
        peak_mag = abs_col[peak_local]
        if peak_mag <= 0:
            continue
        peak_grid_idx = idxs[peak_local]
        peak_sign = np.sign(col_f[peak_local])
        near = np.abs(idxs - peak_grid_idx) <= _TIP_RIDGE_LOCAL_WINDOW
        same_sign = np.sign(col_f) == peak_sign
        sel = near & same_sign
        if not np.any(sel):
            sel = np.zeros_like(near)
            sel[peak_local] = True

        w = abs_col[sel]
        total = float(np.sum(w))
        centroid_s = float(np.sum(arc_length[idxs[sel]] * w) / total)
        centroid_xy = np.array([
            float(np.interp(centroid_s, arc_length, positions[:, 0])),
            float(np.interp(centroid_s, arc_length, positions[:, 1])),
        ])
        tip_s[fi] = centroid_s
        tip_xy[fi] = centroid_xy
        running_max_s = max(running_max_s, centroid_s)

    return _finalize(leaf_index, "ridge", field_key, times, tip_s, tip_xy,
                      positions, arc_length, np.arange(n_frames), pixel_size_mm)


def export_tip_track_csv(result: TipTrackResult, path: str, pixel_size_mm: float | None = None) -> None:
    """Writes one tracked tip's (time, arc-length position, x, y) trace to
    a CSV, one row per frame that had a detection, together with a LOCAL
    propagation velocity derived by central-difference differentiation of
    arc-length position with respect to time (forward/backward difference
    at the two ends) -- the per-point, non-averaged velocity the user
    asked for, distinct from `result.speed_px_per_s` (one single straight-
    line fit speed for the whole track, already available in the on-
    screen summary). Only rows with a valid (non-NaN) detection are
    written; the local derivative uses each pair of CONSECUTIVE valid rows
    in the file (i.e. gaps from skipped frames are bridged rather than
    producing a NaN velocity, since a longer time gap still gives a
    perfectly well-defined average velocity over that gap). Raises
    `OSError` on a write failure, left for the caller to report as-is."""
    import csv

    valid = np.isfinite(result.arc_length_s)
    times = result.times[valid]
    arc_s = result.arc_length_s[valid]
    xy = result.positions_xy[valid]
    n = times.size

    velocity_px_per_s = np.full(n, np.nan)
    if n >= 2:
        velocity_px_per_s[1:-1] = (arc_s[2:] - arc_s[:-2]) / (times[2:] - times[:-2])
        velocity_px_per_s[0] = (arc_s[1] - arc_s[0]) / (times[1] - times[0]) if times[1] != times[0] else np.nan
        velocity_px_per_s[-1] = (arc_s[-1] - arc_s[-2]) / (times[-1] - times[-2]) if times[-1] != times[-2] else np.nan

    header = ["leaf_index", "method", "field", "time_s", "arc_length_px"]
    if pixel_size_mm:
        header.append("arc_length_mm")
    header += ["x_px", "y_px", "velocity_px_per_s"]
    if pixel_size_mm:
        header.append("velocity_mm_per_s")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for k in range(n):
            row = [result.leaf_index, result.method, result.field_key,
                   f"{times[k]:.9g}", f"{arc_s[k]:.6f}"]
            if pixel_size_mm:
                row.append(f"{arc_s[k] * pixel_size_mm:.6f}")
            row += [f"{xy[k, 0]:.6f}", f"{xy[k, 1]:.6f}",
                    f"{velocity_px_per_s[k]:.6g}" if np.isfinite(velocity_px_per_s[k]) else ""]
            if pixel_size_mm:
                v_mm = velocity_px_per_s[k] * pixel_size_mm if np.isfinite(velocity_px_per_s[k]) else np.nan
                row.append(f"{v_mm:.6g}" if np.isfinite(v_mm) else "")
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Method C, manual ROI-polygon variant -- the page's actual primary "ridge"
# workflow (see the module docstring's third bullet): the user draws a
# polygon directly on the "Show tip field" heat map, the same interaction
# core/elastic_wave_speed.py's `extract_ridge_in_polygon` offers for wave
# fronts. This duplicates that function's per-column extraction algorithm
# (smoothed local-extremum search, strongest-candidate wins, snap to the
# raw peak, tight same-sign centroid refine -- see ITS docstring for the
# full two-step reasoning and why "strongest wins" beats every fancier
# tie-break tried there) rather than importing it, because it runs over a
# `TipFieldProfile` already built along a (possibly curved) traced path's
# arc length, not a straight x/y axis sampled fresh from the dataset grid
# -- consistent with this module's existing convention of duplicating
# small helpers (`_resolve_times`, `_smooth_1d`) instead of cross-
# importing them from core/elastic_wave_speed.py.
# ---------------------------------------------------------------------------

@dataclass
class PolygonTipRidgeResult:
    """One user-picked crack-tip trace -- from `extract_tip_ridge_in_polygon`
    (`method="ridge (ROI)"`, the default), from its sibling `extract_tip_
    arrival_time_in_polygon` (`method="arrival (ROI)"`), or from the manual
    frame-by-frame click tool (`finalize_manual_tip_pick`,
    `method="manual (click)"`) -- the crack-tracking equivalent of
    core/elastic_wave_speed.py's `PolygonRidgeResult`. All three methods
    land in this SAME class (and the page's SAME `self._tip_roi_traces`
    list) precisely so a trace picked either way is, from every downstream
    consumer's point of view (the comparison plot, K1c, XFEM-DIC's trace
    source, CSV export), just another trace to tick -- `method` is only
    ever read to label/tag it, never to change how it's used. `speed_px_per_s`/
    `speed_mm_per_s`/`r_squared` are a plain OLS fit through `times`/
    `arc_length`, reported purely as an indicative convenience (None if
    fewer than 2 points): the raw points, not this fit, are the actual
    result."""
    label: str
    color: str
    leaf_index: int
    field_key: str
    vertices: list           # [(time_s, arc_length_px), ...] as drawn, kept for reference/redraw --
                              # empty for a manually-clicked trace (no polygon was drawn)
    times: np.ndarray
    arc_length: np.ndarray
    positions_xy: np.ndarray        # (n, 2) pixel (x, y), interpolated along the traced path
    values: np.ndarray              # baseline-free field value at each ridge point (raw, not baseline-subtracted)
    source_frame_positions: np.ndarray  # (n,) int index into `dataset.frames` each point came from --
                                         # lets this trace be reused as a K1c (Stage 4) tip track,
                                         # same role as `TipTrackResult.source_frame_positions`
    speed_px_per_s: float | None = None
    speed_mm_per_s: float | None = None
    r_squared: float | None = None
    method: str = "ridge (ROI)"  # "ridge (ROI)" | "arrival (ROI)" | "manual (click)" -- see class docstring
    # Optional measurement-uncertainty fields (see module docstring's
    # "MEASUREMENT UNCERTAINTY" section) -- only ever populated for an
    # "opening_jump" trace added with "Estimate measurement uncertainty"
    # checked; None for every other method/trace, and None here means
    # "not estimated", never "zero uncertainty".
    position_uncertainty_px: np.ndarray | None = None  # (n,) px, one per point, parallel to `arc_length`
    speed_stderr_px_per_s: float | None = None   # standard error of the WEIGHTED fit's slope
    speed_stderr_mm_per_s: float | None = None
    # Name of the traced crack this trace was detected on (None = unknown /
    # older traces). Leaf indices are only unique WITHIN one crack, so the
    # viewer matches on (crack_name, leaf_index) once several cracks are
    # processed at once.
    crack_name: str | None = None


def _extract_ridge_points_from_profile(profile, path, backslide_tolerance_px, direction=None,
                                        min_peak_strength_abs: float | None = None):
    """The per-frame extraction loop shared by both passes
    `extract_tip_ridge_in_polygon` needs -- see that function's docstring.
    `direction` is `None` (no monotonic constraint, the raw pick),
    `+1` (arc length must not DECREASE beyond `backslide_tolerance_px`
    from the running maximum), or `-1` (arc length must not INCREASE
    beyond it from the running minimum) -- the sign a crack advances in
    along this particular traced path's arc-length axis isn't assumed;
    `extract_tip_ridge_in_polygon` decides it from a first, unconstrained
    pass before calling this a second time with the constraint.

    `min_peak_strength_abs` (only used when `direction is not None`): an
    OPTIONAL extra floor a candidate must also clear, on top of the
    existing floor relative to that frame's own peak
    (`_TIP_RIDGE_MONOTONIC_MIN_PEAK_FRACTION`). The relative floor alone
    still reports SOMETHING every frame that has any enclosed sample at
    all, even a frame where nothing physically meaningful is happening
    yet (e.g. before the crack starts to propagate, the field near the
    notch still fluctuates a little under the initial stress wave) --
    15% of a very weak, noise-only peak is still noise. Since this floor
    is on the RAW field value itself, it's in the same units/scale as
    `field_key`, so a fixed absolute level here IS meaningful across
    frames of one run (unlike a level compared across different
    specimens/materials, where the field's absolute scale can differ).
    None (default) keeps the previous behavior unchanged.

    Returns `(times, arc_length, values, source_frame_positions)` as
    plain lists (not yet arrays -- the caller wraps them)."""
    running_extreme = None
    n_points, n_frames = profile.values.shape
    times_out: list[float] = []
    arclen_out: list[float] = []
    values_out: list[float] = []
    frameidx_out: list[int] = []
    for j in range(n_frames):
        t = float(profile.times[j])
        pts = np.column_stack([np.full(n_points, t), profile.path_arc_length])
        inside = np.nonzero(path.contains_points(pts))[0]
        if inside.size == 0:
            continue
        col = profile.values[inside, j]
        finite = np.isfinite(col)
        if not np.any(finite):
            continue
        inside = inside[finite]
        col = col[finite]
        abs_col = np.abs(col)
        n_in = inside.size

        if n_in <= 2:
            candidates = list(range(n_in))
        else:
            abs_smooth = _smooth_1d(abs_col, min(_TIP_RIDGE_SMOOTH_RADIUS, (n_in - 1) // 2))
            raw_candidates = [
                i for i in range(1, n_in - 1)
                if abs_smooth[i] >= abs_smooth[i - 1] and abs_smooth[i] >= abs_smooth[i + 1]
            ]
            candidates = []
            run: list[int] = []
            for i in raw_candidates:
                if run and i != run[-1] + 1:
                    candidates.append(max(run, key=lambda k: abs_col[k]))
                    run = []
                run.append(i)
            if run:
                candidates.append(max(run, key=lambda k: abs_col[k]))
            if not candidates:
                candidates = [int(np.argmax(abs_col))]

        if direction is not None:
            frame_peak = float(np.max(abs_col))
            strength_floor = _TIP_RIDGE_MONOTONIC_MIN_PEAK_FRACTION * frame_peak
            if running_extreme is None:
                position_ok = lambda i: True  # nothing accepted yet -- every candidate is "forward"
            elif direction > 0:
                position_ok = lambda i: profile.path_arc_length[inside[i]] >= running_extreme - backslide_tolerance_px
            else:
                position_ok = lambda i: profile.path_arc_length[inside[i]] <= running_extreme + backslide_tolerance_px
            eligible = [
                i for i in candidates
                if position_ok(i) and abs_col[i] >= strength_floor
                and (min_peak_strength_abs is None or abs_col[i] >= min_peak_strength_abs)
            ]
            if not eligible:
                continue  # every surviving candidate is either receding or too weak to trust -- skip this frame
            candidates = eligible

        peak_local = max(candidates, key=lambda i: abs_col[i])
        lo = max(0, peak_local - _TIP_RIDGE_LOCAL_WINDOW)
        hi = min(n_in, peak_local + _TIP_RIDGE_LOCAL_WINDOW + 1)
        peak_local = lo + int(np.argmax(abs_col[lo:hi]))
        peak_mag = abs_col[peak_local]
        if peak_mag <= 0:
            continue
        peak_grid_idx = inside[peak_local]
        peak_sign = np.sign(col[peak_local])
        near = np.abs(inside - peak_grid_idx) <= _TIP_RIDGE_LOCAL_WINDOW
        same_sign = np.sign(col) == peak_sign
        sel = near & same_sign
        if not np.any(sel):
            sel = np.zeros_like(near)
            sel[peak_local] = True

        w = abs_col[sel]
        total = float(np.sum(w))
        centroid_s = float(np.sum(profile.path_arc_length[inside[sel]] * w) / total)
        centroid_val = float(np.sum(col[sel] * w) / total)
        times_out.append(t)
        arclen_out.append(centroid_s)
        values_out.append(centroid_val)
        frameidx_out.append(int(profile.frame_indices[j]))
        if direction is not None:
            if running_extreme is None:
                running_extreme = centroid_s
            elif direction > 0:
                running_extreme = max(running_extreme, centroid_s)
            else:
                running_extreme = min(running_extreme, centroid_s)

    return times_out, arclen_out, values_out, frameidx_out


def extract_tip_ridge_in_polygon(dataset, crack, leaf_index: int, field_key: str, vertices,
                                  frame_range: tuple | None = None, frame_rate_hz_override: float | None = None,
                                  resample_step_px: float = 2.0, enforce_monotonic: bool = True,
                                  backslide_tolerance_px: float | None = None, reverse_path: bool = False,
                                  min_peak_strength_abs: float | None = None):
    """Manual ROI-polygon crack-tip ridge picking, in the (time, arc
    length) plane of `build_tip_field_profile`'s heat map -- see the
    module docstring's third bullet, and this section's own header
    comment for how this mirrors core/elastic_wave_speed.py's
    `extract_ridge_in_polygon`. `vertices` is a list of
    (time_s, arc_length_px) points in the SAME plane the tip-field plot
    itself is drawn in (matching how the polygon is drawn directly on
    that plot); at least 3 are needed to enclose any area.

    For every time frame the polygon's time-extent spans, tests every
    path-point sample at that time against the polygon and, among the
    ones inside, reports ONE ridge point at that frame's own strongest
    local extremum (smoothed to find candidates, raw signal for the
    actual value, tight same-sign centroid refinement) -- no threshold,
    so a faint or drifting-intensity trace is exactly as easy to pick as
    a strong one, as long as nothing stronger shares the polygon (draw it
    tight around the visible advancing-tip band for that reason).

    `enforce_monotonic` (default True), when set, additionally applies the
    same physical prior `track_tip_ridge`'s automatic version uses: a
    frame's candidate is only eligible if its arc-length position isn't
    more than `backslide_tolerance_px` (default 3x `resample_step_px`)
    behind the furthest position already accepted in an earlier frame
    inside this SAME polygon. A drawn polygon isolates roughly the right
    REGION, but a per-frame independent "strongest wins" pick inside it
    can still latch onto a different, momentarily-stronger feature from
    one frame to the next -- e.g. the crack faces' own opening/closing
    signal right behind the tip, which can locally out-shine the tip's
    own (fainter, still-advancing) near-tip concentration -- producing a
    trace that looks like it darts backward and forward ("the crack
    opens and closes") even though the true tip position is advancing
    steadily underneath. Since a real crack tip physically never recedes,
    forbidding a backward jump (beyond ordinary measurement noise)
    removes exactly that artifact without needing any field-value
    threshold at all.

    Critically, "forward" is NOT assumed to mean increasing arc length: a
    traced skeleton's arc length runs from whichever node was placed/
    detected first (see `build_tip_field_profile`/`resample_path`), and a
    crack can equally be traced from its initiation point outward OR from
    its current tip back toward the origin -- top-to-bottom, bottom-to-
    top, left-to-right or right-to-left are all equally valid depending on
    how that particular skeleton happened to be drawn. Assuming "forward"
    always means "increasing" would silently reject nearly every genuine
    point (and keep only a handful of noise picks) whenever a trace
    happens to run the other way -- exactly the failure mode this
    function avoids by running an unconstrained first pass purely to read
    off which way arc length actually trends over time here (the sign of
    an ordinary least-squares fit through the raw, unconstrained picks),
    then re-running WITH the monotonic constraint in that same direction.
    A trace with too few unconstrained points to fit a trend (or an
    exactly flat one) falls back to "increasing" as its default
    direction.

    On its own, the backward-jump filter can still misfire: if a frame's
    only position-eligible candidates are all just per-pixel noise (e.g.
    both the tip and the face are, at that instant, positioned behind the
    running extreme), "strongest of what's left" would otherwise report
    that noise as the frame's point. To prevent that, an eligible
    candidate must ALSO reach at least `_TIP_RIDGE_MONOTONIC_MIN_PEAK_FRACTION`
    of that frame's own strongest enclosed sample; a frame with no
    candidate clearing both the position and strength bars is skipped
    entirely (no point is reported for it) rather than reporting a
    fabricated one. This floor is relative to each frame's own signal, not
    a fixed absolute level, so it does not reintroduce the drifting-
    intensity sensitivity the threshold method was replaced for.

    Set `enforce_monotonic=False` to see the raw, unconstrained pick (e.g.
    to diagnose why a trace looks off, or for a case where the crack
    genuinely can open/close, such as a pre-existing notch cycling under
    load before it starts to grow).

    `min_peak_strength_abs` (default None, only applied when
    `enforce_monotonic=True`): an OPTIONAL extra floor on top of the
    relative one above -- see `_extract_ridge_points_from_profile`'s own
    docstring for why a fixed absolute level (on the raw field value,
    same units as `field_key`) can be meaningful here even though the
    relative floor already exists. Useful for a trace whose early frames
    are dominated by pre-propagation noise (the field near the notch
    fluctuating under the initial stress wave, before the crack actually
    starts to move) -- those frames are skipped entirely rather than
    reporting a noise pick, the same way a too-weak-relative-to-its-own-
    peak candidate already is.

    `reverse_path` is passed straight through to `build_tip_field_profile`
    -- see its docstring for what flips and why.

    Returns `(times, arc_length, values, source_frame_positions, profile)`
    -- `profile` is the same field/background (`TipFieldProfile`) the
    polygon was drawn against, handed back so a caller can redraw it
    without recomputing."""
    profile = build_tip_field_profile(
        dataset, crack, leaf_index, field_key, frame_range, frame_rate_hz_override, resample_step_px,
        reverse_path=reverse_path,
    )
    verts = [(float(t), float(s)) for t, s in vertices]
    if len(verts) < 3:
        raise CrackTipTrackingError("A polygon needs at least 3 vertices to enclose an area.")
    path = _MplPath(np.asarray(verts, dtype=float))

    if backslide_tolerance_px is None:
        backslide_tolerance_px = 3.0 * resample_step_px

    if not enforce_monotonic:
        times_out, arclen_out, values_out, frameidx_out = _extract_ridge_points_from_profile(
            profile, path, backslide_tolerance_px, direction=None,
        )
    else:
        raw_times, raw_arclen, _, _ = _extract_ridge_points_from_profile(
            profile, path, backslide_tolerance_px, direction=None,
        )
        direction = 1
        if len(raw_times) >= 2 and not np.allclose(raw_times, raw_times[0]):
            slope = float(np.polyfit(raw_times, raw_arclen, 1)[0])
            if slope < 0:
                direction = -1
        times_out, arclen_out, values_out, frameidx_out = _extract_ridge_points_from_profile(
            profile, path, backslide_tolerance_px, direction=direction,
            min_peak_strength_abs=min_peak_strength_abs,
        )

    return (np.array(times_out, dtype=float), np.array(arclen_out, dtype=float),
            np.array(values_out, dtype=float), np.array(frameidx_out, dtype=int), profile)


def extract_tip_arrival_time_in_polygon(dataset, crack, leaf_index: int, field_key: str, vertices,
                                         frame_range: tuple | None = None, frame_rate_hz_override: float | None = None,
                                         resample_step_px: float = 2.0, threshold: float = 0.0,
                                         enforce_monotonic: bool = True, backslide_tolerance_px: float | None = None,
                                         reverse_path: bool = False):
    """`extract_tip_ridge_in_polygon`'s sibling method -- see the module
    docstring's "arrival (ROI)" bullet for the full rationale (in short:
    POSITION by position, "when did this spot first cross a fixed
    absolute level", instead of FRAME by frame, "where's the strongest
    peak right now"), and that function's own docstring for what
    `vertices`/`frame_range`/`reverse_path` mean -- identical here, same
    (time, arc length) plane, same polygon-drawing workflow.

    For every resampled path position inside the polygon's arc-length
    span, finds the first (sub-frame-interpolated) time its |field| value
    reaches `threshold` -- an ABSOLUTE value in the same units as
    `field_key`, NOT a fraction of a local or per-frame peak (unlike
    core/elastic_wave_speed.py's own arrival-time detector, which
    normalizes by each position's own peak because a wave's amplitude
    decays with distance from its source; a crack's near-tip strain
    concentration doesn't have that problem, and an absolute level is
    both what a genuinely cracked/undamaged split looks like in the data
    and what `suggest_threshold` (a percentile of this crack's own pooled
    path data, the same helper the old "threshold" walk method used)
    already estimates). Only samples strictly INSIDE the polygon (in the
    (time, arc_length) plane) count -- exactly like `extract_tip_ridge_
    in_polygon`, this lets a loosely-drawn loop exclude an unrelated part
    of the field (e.g. a separate crack face artifact) from ever being
    considered at all, though since this method has no competing-
    candidates-per-frame step to disambiguate, a tight polygon matters
    less here than for "ridge".

    The reported points are then ordered by their own arrival time (which
    is what actually varies from the drawing order along the path) and,
    by default (`enforce_monotonic=True`), filtered the same way `extract_
    tip_ridge_in_polygon` filters its own per-frame picks: a point is kept
    only if its arc-length position isn't more than `backslide_tolerance_
    px` (default 3x `resample_step_px`) behind the furthest position
    already kept -- a handful of positions can still cross `threshold`
    slightly out of physical order under real DIC noise (a stray pixel-
    level fluctuation reaching the level a frame or two early), and this
    keeps that from showing up as visible backward jitter in the
    resulting trace. Which direction counts as "forward" (increasing or
    decreasing arc length) is auto-detected from the raw, unfiltered
    picks' own trend, the same way `extract_tip_ridge_in_polygon` detects
    it -- see that function's docstring for why "forward" is never
    assumed to mean "increasing". Set `enforce_monotonic=False` to see
    the raw, unfiltered picks.

    Returns `(times, arc_length, values, source_frame_positions, profile)`
    -- the same shape `extract_tip_ridge_in_polygon` returns, for drop-in
    use everywhere a `PolygonTipRidgeResult` is built from one. `values`
    is each reported point's raw (signed) field value at its interpolated
    crossing (close to +-`threshold` in magnitude by construction, kept
    mainly for CSV export completeness). `source_frame_positions` is the
    nearest actual frame at or after each interpolated crossing time."""
    profile = build_tip_field_profile(
        dataset, crack, leaf_index, field_key, frame_range, frame_rate_hz_override, resample_step_px,
        reverse_path=reverse_path,
    )
    verts = [(float(t), float(s)) for t, s in vertices]
    if len(verts) < 3:
        raise CrackTipTrackingError("A polygon needs at least 3 vertices to enclose an area.")
    path = _MplPath(np.asarray(verts, dtype=float))

    if backslide_tolerance_px is None:
        backslide_tolerance_px = 3.0 * resample_step_px

    n_points, n_frames = profile.values.shape
    arrival_t = np.full(n_points, np.nan)
    arrival_v = np.full(n_points, np.nan)

    for i in range(n_points):
        s_i = float(profile.path_arc_length[i])
        row = profile.values[i, :]
        abs_row = np.abs(row)
        pts = np.column_stack([profile.times, np.full(n_frames, s_i)])
        inside = path.contains_points(pts) & np.isfinite(row)
        idxs = np.flatnonzero(inside)
        if idxs.size == 0:
            continue
        above = idxs[abs_row[idxs] >= threshold]
        if above.size == 0:
            continue
        j = int(above[0])
        # Sub-frame linear interpolation of the exact crossing time
        # between the previous INSIDE/finite sample and this one -- same
        # "don't round to the nearest frame" spirit as `_interpolate_tip`
        # and core/elastic_wave_speed.py's own arrival-time detector.
        # Falls back to the raw crossing frame's own time when there's no
        # eligible previous sample to interpolate against (the polygon's
        # own time edge, or a gap from a non-finite neighbor).
        if j > 0 and inside[j - 1]:
            t0, t1 = float(profile.times[j - 1]), float(profile.times[j])
            v0, v1 = float(abs_row[j - 1]), float(abs_row[j])
            if v1 > v0:
                frac = min(max((threshold - v0) / (v1 - v0), 0.0), 1.0)
                arrival_t[i] = t0 + frac * (t1 - t0)
            else:
                arrival_t[i] = t1
        else:
            arrival_t[i] = float(profile.times[j])
        arrival_v[i] = float(row[j])

    valid = np.isfinite(arrival_t)
    if not np.any(valid):
        return (np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float),
                np.array([], dtype=int), profile)

    s_valid = profile.path_arc_length[valid]
    t_valid = arrival_t[valid]
    v_valid = arrival_v[valid]
    order = np.argsort(t_valid, kind="stable")
    times_sorted = t_valid[order]
    arclen_sorted = s_valid[order]
    values_sorted = v_valid[order]

    if enforce_monotonic and times_sorted.size >= 2:
        slope = float(np.polyfit(times_sorted, arclen_sorted, 1)[0])
        direction = 1 if slope >= 0 else -1
        keep = np.zeros(times_sorted.size, dtype=bool)
        running_extreme = None
        for k in range(times_sorted.size):
            a = arclen_sorted[k]
            if running_extreme is None:
                ok = True
            elif direction > 0:
                ok = a >= running_extreme - backslide_tolerance_px
            else:
                ok = a <= running_extreme + backslide_tolerance_px
            if ok:
                keep[k] = True
                if running_extreme is None:
                    running_extreme = a
                elif direction > 0:
                    running_extreme = max(running_extreme, a)
                else:
                    running_extreme = min(running_extreme, a)
        times_sorted = times_sorted[keep]
        arclen_sorted = arclen_sorted[keep]
        values_sorted = values_sorted[keep]

    frame_pos = np.clip(np.searchsorted(profile.times, times_sorted, side="left"), 0, n_frames - 1)
    frame_idx_sorted = profile.frame_indices[frame_pos].astype(int)

    return times_sorted, arclen_sorted, values_sorted, frame_idx_sorted, profile


def fit_tip_ridge_points(times: np.ndarray, arc_length: np.ndarray, pixel_size_mm: float | None = None,
                          uncertainty_px: np.ndarray | None = None,
                          frame_rate_uncertainty_frac: float = 0.0):
    """Fit through a set of (time, arc-length) ridge points, for an
    indicative speed/R^2 -- see `PolygonTipRidgeResult`. Returns
    `(speed_px_per_s, speed_mm_per_s, r_squared, speed_stderr_px_per_s,
    speed_stderr_mm_per_s)`, or `(None, None, None, None, None)` if there
    are fewer than 2 distinct-time points to fit.

    Plain OLS (unweighted, as always -- the last two fields are simply
    None) unless `uncertainty_px` is given (same length as `times`/
    `arc_length`, see the module docstring's "MEASUREMENT UNCERTAINTY"
    section): then this becomes an inverse-variance-WEIGHTED least-
    squares fit (weight_i = 1/uncertainty_px[i]^2), with a proper
    standard error on the fitted slope from the weighted-regression
    covariance (`Var(slope) = S_w / (S_w*S_wtt - S_wt^2)`, the standard
    result for a weighted linear fit when the weights are genuine inverse
    variances, not just relative importances) -- more data points, or
    tighter per-point position uncertainty, both narrow this the way a
    real measurement's confidence should narrow. `frame_rate_
    uncertainty_frac` (e.g. 0.001 for +/-0.1% camera timebase jitter/
    drift -- NOT observable from the trace's own data, so this is the one
    piece of the estimate that must come from the caller) is folded into
    the slope's standard error afterward, in quadrature, as a further
    relative contribution (`speed * frame_rate_uncertainty_frac`).
    Falls back to the plain unweighted fit (last two fields None) if
    `uncertainty_px` has fewer than 2 finite, positive entries lining up
    with valid `times`/`arc_length`, or if the weighted normal equations
    are degenerate (e.g. every weight concentrated on a single point)."""
    t = np.asarray(times, dtype=float)
    s = np.asarray(arc_length, dtype=float)
    if t.size < 2 or np.allclose(t, t[0]):
        return None, None, None, None, None

    if uncertainty_px is not None:
        u = np.asarray(uncertainty_px, dtype=float)
        finite = np.isfinite(u) & (u > 0) & np.isfinite(t) & np.isfinite(s)
        if np.count_nonzero(finite) >= 2:
            tt, ss, uu = t[finite], s[finite], u[finite]
            w = 1.0 / uu ** 2
            sw = float(np.sum(w))
            swt = float(np.sum(w * tt))
            swtt = float(np.sum(w * tt ** 2))
            sws = float(np.sum(w * ss))
            swts = float(np.sum(w * tt * ss))
            denom = sw * swtt - swt ** 2
            if denom > 0 and not np.isclose(tt, tt[0]).all():
                slope = (sw * swts - swt * sws) / denom
                intercept = (sws - slope * swt) / sw
                slope_stderr = float(np.sqrt(sw / denom))
                predicted = slope * tt + intercept
                ss_res = float(np.sum(w * (ss - predicted) ** 2))
                weighted_mean = float(np.average(ss, weights=w))
                ss_tot = float(np.sum(w * (ss - weighted_mean) ** 2))
                r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
                if frame_rate_uncertainty_frac:
                    slope_stderr = float(np.hypot(slope_stderr, slope * frame_rate_uncertainty_frac))
                speed_mm = slope * pixel_size_mm if pixel_size_mm else None
                stderr_mm = slope_stderr * pixel_size_mm if pixel_size_mm else None
                return float(slope), speed_mm, float(r_squared), slope_stderr, stderr_mm

    # Unweighted fallback -- also the path taken when `uncertainty_px`
    # isn't given at all, unchanged from this function's original
    # behavior.
    slope, intercept = np.polyfit(t, s, 1)
    predicted = slope * t + intercept
    ss_res = float(np.sum((s - predicted) ** 2))
    ss_tot = float(np.sum((s - np.mean(s)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    speed_mm = slope * pixel_size_mm if pixel_size_mm else None
    return float(slope), speed_mm, float(r_squared), None, None


def tip_roi_trace_to_track_result(trace: PolygonTipRidgeResult, path_positions: np.ndarray,
                                   path_arc_length: np.ndarray, pixel_size_mm: float | None = None) -> TipTrackResult:
    """Converts one ROI-picked `PolygonTipRidgeResult` into a
    `TipTrackResult` -- lets a manually-picked ridge trace be used
    downstream exactly like an automatic method's result (the page's
    speed/distance summary, and core/crack_k1c.py's Stage 4, which both
    expect a `TipTrackResult`). `path_positions`/`path_arc_length` are the
    same traced-path resample the trace's `PolygonTipRidgeResult.
    source_frame_positions`/`arc_length` were extracted against (i.e. the
    `TipFieldProfile` the ROI polygon was drawn on) -- passed in rather
    than re-derived, so this never needs `dataset`/`crack` again. Reads
    `trace.method` rather than assuming "ridge (ROI)", so this same
    converter also works for a manually-clicked trace (see
    `PolygonTipRidgeResult`'s docstring)."""
    return _finalize(
        trace.leaf_index, trace.method, trace.field_key, trace.times, trace.arc_length, trace.positions_xy,
        path_positions, path_arc_length, trace.source_frame_positions, pixel_size_mm,
    )


def trace_velocity_px_per_s(times: np.ndarray, arc_length: np.ndarray) -> np.ndarray:
    """Local (per-point) crack-tip propagation speed, px/s -- the central-
    difference derivative of arc-length position with respect to time at
    every point of a trace (one-sided at the two ends), `np.nan` where
    undefined (fewer than 2 points, or two points landing at the exact
    same time). Shared by `export_tip_roi_csv`'s velocity column, the
    "Verify visually" arc-length-vs-time plot's secondary axis, and the
    exported video's live speed curve, so the three always agree on
    exactly the same numbers -- deliberately NOT a smoothed/fitted speed
    (see `fit_tip_ridge_points` for that): this is the raw, noisy,
    point-to-point speed a propagating crack's instantaneous velocity
    history actually needs, not a single straight-line average."""
    n = times.size
    velocity = np.full(n, np.nan)
    if n < 2:
        return velocity
    velocity[1:-1] = (arc_length[2:] - arc_length[:-2]) / (times[2:] - times[:-2])
    v0_dt = times[1] - times[0]
    velocity[0] = (arc_length[1] - arc_length[0]) / v0_dt if v0_dt != 0 else np.nan
    v1_dt = times[-1] - times[-2]
    velocity[-1] = (arc_length[-1] - arc_length[-2]) / v1_dt if v1_dt != 0 else np.nan
    return velocity


def trace_velocity_uncertainty_px_per_s(times: np.ndarray, position_uncertainty_px: np.ndarray) -> np.ndarray:
    """Uncertainty (1 sigma, px/s) on `trace_velocity_px_per_s`'s own
    local per-point speed, from ordinary error propagation through the
    EXACT SAME finite-difference formula (central in the interior,
    one-sided at the two ends) -- so the two arrays line up index for
    index and a caller can plot `trace_velocity_px_per_s(...) +/- N *
    trace_velocity_uncertainty_px_per_s(...)` directly, e.g. the "Verify
    visually" arc-length-vs-time plot's speed curve.

    For the interior's central difference v[i] = (s[i+1] - s[i-1]) /
    (t[i+1] - t[i-1]), with `position_uncertainty_px[i+1]` and `[i-1]`
    treated as independent (each frame's front position is detected
    separately -- see the module docstring's "MEASUREMENT UNCERTAINTY"
    section): sigma_v[i] = sqrt(sigma_s[i+1]^2 + sigma_s[i-1]^2) / (t[i+1]
    - t[i-1]). The one-sided ends use the same rule over their own pair of
    points. `np.nan` wherever `trace_velocity_px_per_s` itself would be
    (fewer than 2 points, or a zero time step), and wherever either input
    position's own uncertainty is `nan` (nan propagates through the sqrt
    sum, as it should -- an unknown position uncertainty makes the
    resulting speed uncertainty unknown too, not zero)."""
    n = times.size
    sigma_v = np.full(n, np.nan)
    if n < 2:
        return sigma_v
    sigma_s = np.asarray(position_uncertainty_px, dtype=float)
    sigma_v[1:-1] = np.hypot(sigma_s[2:], sigma_s[:-2]) / (times[2:] - times[:-2])
    v0_dt = times[1] - times[0]
    sigma_v[0] = np.hypot(sigma_s[1], sigma_s[0]) / v0_dt if v0_dt != 0 else np.nan
    v1_dt = times[-1] - times[-2]
    sigma_v[-1] = np.hypot(sigma_s[-1], sigma_s[-2]) / v1_dt if v1_dt != 0 else np.nan
    return sigma_v


def export_tip_roi_csv(traces: list, path: str, pixel_size_mm: float | None = None) -> None:
    """Writes every stored ROI-picked crack-tip trace's raw (time, arc-
    length) points to one CSV, long format (one row per point, a
    `trace_label` column tells them apart) -- mirrors core/
    elastic_wave_speed.py's `export_roi_waves_csv`, plus a LOCAL
    propagation velocity column (central-difference derivative of arc-
    length position with respect to time, within each trace) since the
    per-point, non-averaged velocity was explicitly asked for, unlike a
    wave's export (a single straight-line fit speed is normally enough
    there). Raises `OSError` on a write failure, left for the caller to
    report as-is.

    If ANY trace carries measurement-uncertainty fields (`PolygonTipRidgeResult.
    position_uncertainty_px`/`speed_stderr_px_per_s` -- see the module
    docstring's "MEASUREMENT UNCERTAINTY" section, only ever populated for
    an "opening_jump" trace added with that option checked), extra columns
    are added for every trace: `position_uncertainty_px` (per point; blank
    for a trace/point that doesn't have one), `local_speed_uncertainty_px_
    per_s` (per point too -- `trace_velocity_uncertainty_px_per_s` applied
    to `position_uncertainty_px`, the EXACT SAME per-point local-speed
    band the "Verify visually" arc-length-vs-time plot itself shades, so
    this column reproduces that plot's own uncertainty numbers rather than
    a differently-defined one), and, repeated on every row of a trace (a
    single whole-trace number, not a per-point one, but CSV has no natural
    place for a scalar otherwise), `trace_speed_stderr_px_per_s` (the
    whole-trace weighted-fit speed uncertainty reported elsewhere in the
    UI, distinct from the per-point column above) -- plus the `_mm`/
    `_mm_per_s` counterpart of each when `pixel_size_mm` is given."""
    import csv

    has_uncertainty = any(getattr(t, "position_uncertainty_px", None) is not None for t in traces)

    header = ["trace_label", "method", "leaf_index", "field", "time_s", "arc_length_px"]
    if pixel_size_mm:
        header.append("arc_length_mm")
    header += ["x_px", "y_px", "field_value", "velocity_px_per_s"]
    if pixel_size_mm:
        header.append("velocity_mm_per_s")
    if has_uncertainty:
        header.append("position_uncertainty_px")
        if pixel_size_mm:
            header.append("position_uncertainty_mm")
        header.append("local_speed_uncertainty_px_per_s")
        if pixel_size_mm:
            header.append("local_speed_uncertainty_mm_per_s")
        header.append("trace_speed_stderr_px_per_s")
        if pixel_size_mm:
            header.append("trace_speed_stderr_mm_per_s")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for trace in traces:
            n = trace.times.size
            velocity = trace_velocity_px_per_s(trace.times, trace.arc_length)
            unc_px = getattr(trace, "position_uncertainty_px", None)
            vel_unc_px = (trace_velocity_uncertainty_px_per_s(trace.times, unc_px)
                          if unc_px is not None else None)
            for k in range(n):
                row = [trace.label, trace.method, trace.leaf_index, trace.field_key,
                       f"{trace.times[k]:.9g}", f"{trace.arc_length[k]:.6f}"]
                if pixel_size_mm:
                    row.append(f"{trace.arc_length[k] * pixel_size_mm:.6f}")
                row += [f"{trace.positions_xy[k, 0]:.6f}", f"{trace.positions_xy[k, 1]:.6f}",
                        f"{trace.values[k]:.6g}",
                        f"{velocity[k]:.6g}" if np.isfinite(velocity[k]) else ""]
                if pixel_size_mm:
                    v_mm = velocity[k] * pixel_size_mm if np.isfinite(velocity[k]) else np.nan
                    row.append(f"{v_mm:.6g}" if np.isfinite(v_mm) else "")
                if has_uncertainty:
                    u_px = unc_px[k] if unc_px is not None and k < len(unc_px) and np.isfinite(unc_px[k]) else None
                    row.append(f"{u_px:.6g}" if u_px is not None else "")
                    if pixel_size_mm:
                        row.append(f"{u_px * pixel_size_mm:.6g}" if u_px is not None else "")
                    v_unc = (vel_unc_px[k] if vel_unc_px is not None and k < len(vel_unc_px)
                             and np.isfinite(vel_unc_px[k]) else None)
                    row.append(f"{v_unc:.6g}" if v_unc is not None else "")
                    if pixel_size_mm:
                        row.append(f"{v_unc * pixel_size_mm:.6g}" if v_unc is not None else "")
                    row.append(f"{trace.speed_stderr_px_per_s:.6g}"
                               if trace.speed_stderr_px_per_s is not None else "")
                    if pixel_size_mm:
                        row.append(f"{trace.speed_stderr_mm_per_s:.6g}"
                                   if trace.speed_stderr_mm_per_s is not None else "")
                writer.writerow(row)


def _excel_sheet_name(name: str, used: set) -> str:
    """A valid, unique Excel sheet title (<= 31 chars, none of []:*?/\\)."""
    import re
    base = re.sub(r"[\[\]:*?/\\]", "_", str(name)).strip() or "Sheet"
    base = base[:31]
    cand, k = base, 2
    while cand.lower() in used:
        suffix = f" ({k})"
        cand = base[:31 - len(suffix)] + suffix
        k += 1
    used.add(cand.lower())
    return cand


def group_traces_by_crack(traces: list) -> list:
    """[(crack_name, [traces...])] in first-seen order; a trace without a
    recorded crack goes under "Traces"."""
    groups: dict = {}
    for t in traces:
        groups.setdefault(getattr(t, "crack_name", None) or "Traces", []).append(t)
    return list(groups.items())


def export_tip_roi_xlsx(traces: list, path: str, pixel_size_mm: float | None = None) -> list:
    """Same data as `export_tip_roi_csv`, but as an Excel workbook with ONE
    SHEET PER CRACK (every trace detected on that crack, a `trace_label`
    column tells them apart); numbers are written as numbers, not text.
    Returns the sheet names. Needs `openpyxl` (ImportError otherwise -- the
    caller falls back to one CSV per crack). Raises OSError on a write
    failure."""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    wb.remove(wb.active)
    used: set = set()
    names = []
    for crack_name, group in group_traces_by_crack(traces):
        ws = wb.create_sheet(_excel_sheet_name(crack_name, used))
        names.append(ws.title)
        has_unc = any(getattr(t, "position_uncertainty_px", None) is not None for t in group)
        header = ["trace_label", "method", "leaf_index", "field", "time_s", "arc_length_px"]
        if pixel_size_mm:
            header.append("arc_length_mm")
        header += ["x_px", "y_px", "field_value", "velocity_px_per_s"]
        if pixel_size_mm:
            header.append("velocity_mm_per_s")
        if has_unc:
            header += ["position_uncertainty_px"] + (["position_uncertainty_mm"] if pixel_size_mm else [])
            header += ["local_speed_uncertainty_px_per_s"] + (["local_speed_uncertainty_mm_per_s"] if pixel_size_mm else [])
            header += ["trace_speed_stderr_px_per_s"] + (["trace_speed_stderr_mm_per_s"] if pixel_size_mm else [])
        ws.append(header)
        for c in ws[1]:
            c.font = Font(bold=True)

        def num(v):
            return float(v) if v is not None and np.isfinite(v) else None

        for trace in group:
            velocity = trace_velocity_px_per_s(trace.times, trace.arc_length)
            unc = getattr(trace, "position_uncertainty_px", None)
            vel_unc = trace_velocity_uncertainty_px_per_s(trace.times, unc) if unc is not None else None
            for k in range(trace.times.size):
                row = [trace.label, trace.method, int(trace.leaf_index), trace.field_key,
                       num(trace.times[k]), num(trace.arc_length[k])]
                if pixel_size_mm:
                    row.append(num(trace.arc_length[k] * pixel_size_mm))
                row += [num(trace.positions_xy[k, 0]), num(trace.positions_xy[k, 1]), num(trace.values[k]),
                        num(velocity[k])]
                if pixel_size_mm:
                    row.append(num(velocity[k] * pixel_size_mm))
                if has_unc:
                    u = num(unc[k]) if unc is not None and k < len(unc) else None
                    row.append(u)
                    if pixel_size_mm:
                        row.append(None if u is None else u * pixel_size_mm)
                    vu = num(vel_unc[k]) if vel_unc is not None and k < len(vel_unc) else None
                    row.append(vu)
                    if pixel_size_mm:
                        row.append(None if vu is None else vu * pixel_size_mm)
                    se = num(trace.speed_stderr_px_per_s) if trace.speed_stderr_px_per_s is not None else None
                    row.append(se)
                    if pixel_size_mm:
                        row.append(None if se is None else se * pixel_size_mm)
                ws.append(row)
        for col in ws.columns:
            width = max(len(str(col[0].value or "")), 10)
            ws.column_dimensions[col[0].column_letter].width = min(width + 2, 34)
        ws.freeze_panes = "A2"
    wb.save(path)
    return names



def wave_roi_trace_to_track_result(wave_trace, dataset, crack, leaf_index: int,
                                    resample_step_px: float = 2.0,
                                    pixel_size_mm: float | None = None,
                                    reverse_path: bool = False) -> TipTrackResult:
    """Converts a manually-picked ELASTIC WAVE ROI trace (core.
    elastic_wave_speed.PolygonRidgeResult -- duck-typed here, not
    imported, per this module's existing convention of not cross-
    importing between the two "stage" modules) into a `TipTrackResult`,
    so it can feed K1c/the speed-distance summary exactly like a Stage 3
    tip track. Meant for the case reported in practice: a crack that
    propagates mostly along one image axis, where the wave-speed page's
    single-axis time-of-flight front tracking and this module's own
    (arc-length) ridge tracking measure the SAME advancing front and
    agree, but the wave measurement can be the cleaner of the two to pick
    (see the page's help text for when to prefer one over the other).

    A wave trace only carries a single-axis coordinate (`wave_trace.
    direction` is "x" or "y", `wave_trace.positions` that axis's pixel
    coordinate over time) -- not a tip (x, y) position or an arc length
    along the traced crack skeleton, both of which K1c/the summary need.
    This recovers them by resampling `crack`'s traced path to `leaf_index`
    (exactly like Stage 3's own `resample_step_px` resampling) and, for
    every wave point, taking the resampled path point whose OWN coordinate
    along that SAME axis is closest -- i.e. "where the traced skeleton
    crosses the x (or y) level the wave front just reached" -- which
    supplies both the (x, y) position and that point's own arc length
    (used downstream to look up a local tangent for the K1c fit, and to
    report distance traveled on a scale consistent with the other tip-
    tracking methods).

    This nearest-crossing mapping assumes the traced path's position along
    that axis is reasonably monotonic over the region/frames used -- the
    same "roughly planar, mostly one-directional front" assumption the
    wave method itself already relies on (see core/elastic_wave_speed.py's
    module docstring); a path that visibly doubles back along that axis
    will alias to whichever crossing happens to be nearest, same caveat as
    the wave method's own arrival-time picking.

    A wave trace also carries no source-frame index (Stage 3's own tip
    traces do, via `TipFieldProfile.frame_indices`) -- recovered here by
    matching each wave point's time back to the closest `dataset.frames[
    ].time_s` (both ultimately come from the same per-frame `time_s`
    values, so this is an exact match in the normal case, not an
    approximation).

    Every length here stays in pixels and every time in seconds -- the
    same convention `TipTrackResult`/`compute_k_time_series` already use
    throughout, with `pixel_size_mm` (passed through to `_finalize`, as
    every other method here does) the only thing that later converts a
    px-based speed/distance to physical units; nothing above needs its own
    separate unit conversion.

    `reverse_path` (see `build_tip_field_profile`'s docstring) flips which
    end of the traced path arc length 0 is measured from, same as for the
    ridge/ROI method -- keep it consistent with whatever this crack/tip's
    ridge trace(s) used, so a distance/speed summary comparing the two
    isn't comparing two different arc-length conventions."""
    paths = branch_paths(crack)
    if leaf_index not in paths:
        raise CrackTipTrackingError(f"No traced path from the crack's origin (its first point) to node {leaf_index}.")
    node_points = [crack.nodes[i] for i in paths[leaf_index]]
    if reverse_path:
        node_points = list(reversed(node_points))
    path_positions, path_arc_length = resample_path(node_points, resample_step_px)

    axis_idx = 0 if wave_trace.direction == "x" else 1
    path_axis_vals = path_positions[:, axis_idx]

    n = wave_trace.times.size
    positions_xy = np.zeros((n, 2))
    arc_length = np.zeros(n)
    for i in range(n):
        idx = int(np.argmin(np.abs(path_axis_vals - wave_trace.positions[i])))
        positions_xy[i] = path_positions[idx]
        arc_length[i] = path_arc_length[idx]

    frame_times = np.array([
        f.time_s if f.time_s is not None else np.nan for f in dataset.frames
    ], dtype=float)
    source_frame_positions = np.zeros(n, dtype=int)
    for i in range(n):
        diffs = np.abs(frame_times - wave_trace.times[i])
        diffs[np.isnan(diffs)] = np.inf
        source_frame_positions[i] = int(np.argmin(diffs))

    return _finalize(
        leaf_index, "wave (ROI)", wave_trace.field_key, wave_trace.times, arc_length, positions_xy,
        path_positions, path_arc_length, source_frame_positions, pixel_size_mm,
    )


# ---------------------------------------------------------------------------
# Ground-truth diagnostic: crack-opening DISPLACEMENT jump directly from the
# already-exported standard subset-DIC field, independent of every tip-
# tracking method above (threshold/singular_fit/ridge/arrival) and of
# core.xfem_dic's own model fit -- a way to check, on the RAW measured data
# alone, how far a crack has genuinely, physically separated at a given
# frame, without trusting any one tracking method's own assumptions.
#
# Motivation (reported in practice): on a real dataset where "ridge" and
# "wave" nominal tip tracks closely agreed with each other, X-FEM-DIC's own
# auto-detected tip search (see core/xfem_dic.py's `_search_best_tip_s`/
# `_search_tip_s_by_heaviside_jump`) still visibly plateaued well short of
# where the nominal tracks placed the tip, even with a generous search
# radius -- raising the question of whether ridge/wave themselves might be
# under-detecting the true crack extent (both sharing a similar bias), not
# just an XFEM-DIC search or near-tip fitting issue. Comparing standard-DIC
# "u"/"v" isocontours side by side is suggestive but not conclusive: a
# SMOOTH gradient (elevated strain without genuine separation, e.g. a
# fracture process zone) can look visually similar to a real discontinuity
# at a glance, especially through a color map. This module's own tip-
# tracking methods, meanwhile, key off DERIVED strain fields (already once
# removed from the raw displacement), or off timing/arrival heuristics --
# neither directly asks "is there an actual displacement DISCONTINUITY
# here", which is the literal physical definition of an open crack face.
#
# This function asks exactly that, directly on the raw exported u/v: at
# each point along the already-traced path, sample the displacement field a
# small distance to EITHER side of the crack face (perpendicular to the
# local tangent) and take the difference -- a genuinely open crack shows a
# real JUMP (the two faces have moved apart/past each other); a smooth
# strain gradient without real separation does not, regardless of how
# intense the nearby strain looks. This is the same physical quantity
# core.xfem_dic's Heaviside enrichment fits FOR (`XfemDicFitResult.
# heaviside_disp_x/y`) and the same criterion its own auto-detect
# "heaviside_jump" mode locates a tip from (`_refine_tip_s_from_heaviside_
# jump`) -- but measured directly off the raw, already-exported field, with
# no XFEM model (background/singular/Heaviside DOFs, mesh, Gauss-Newton
# solve) standing between the measurement and the answer, so it cannot
# inherit any bias from that model's own assumptions (nor share the
# search's plateau failure mode) -- a genuinely independent cross-check,
# not a fourth variant of the same machinery.
# ---------------------------------------------------------------------------

@dataclass
class CrackOpeningJumpProfile:
    """One frame's crack-opening-displacement-jump profile along an
    already-traced path -- see `crack_opening_jump_profile`'s own
    docstring for the full reasoning. Every array is one value per point
    of the SAME `path_positions`/`path_arc_length` the caller passed in
    (typically a `TipTrackResult`'s own, so this plots directly against
    that trace's tracked tip positions)."""
    arc_length_s: np.ndarray
    positions_xy: np.ndarray
    jump_u_px: np.ndarray    # |u(+offset) - u(-offset)| at each path point
    jump_v_px: np.ndarray    # |v(+offset) - v(-offset)| at each path point
    jump_mag_px: np.ndarray  # hypot(jump_u_px, jump_v_px) -- the combined criterion below is found from
    # `jump_mag_px` after `_nanmedian_filter_1d(jump_mag_px, smooth_window_samples)` -- IDENTICAL to
    # jump_mag_px (a copy, not the same array) when smooth_window_samples<=1. This, not jump_mag_px
    # itself, is what opening_extent_s/xy below are actually walked from -- see
    # `_nanmedian_filter_1d`'s own docstring for why (suppressing narrow wave-induced pulses while
    # leaving a genuine sustained opening plateau intact). jump_mag_px itself is kept unfiltered
    # alongside it so the raw measured signal is never silently lost -- both are plotted/exported.
    jump_mag_smoothed_px: np.ndarray
    smooth_window_samples: int
    offset_px: float
    threshold_px: float
    # The furthest arc length (from the path's own origin, index 0 -- see
    # this function's docstring on why that end must already be open)
    # where jump_mag_px is still continuously at/above threshold_px --
    # i.e. this diagnostic's own answer to "how far is the crack really
    # open", directly comparable to any tracking method's tip_s for the
    # same frame. None if the path's own origin never reaches threshold_px
    # at all (nothing to walk from).
    opening_extent_s: float | None
    opening_extent_xy: np.ndarray | None
    # True when the jump was measured as the DISCONTINUITY at the crack face
    # (each side extrapolated to the face, see `_face_values`), i.e. with the
    # continuous elastic / wave part removed; False = plain u(+offset) -
    # u(-offset). `n_fit_samples` = samples per side used for the extrapolation.
    continuous_part_removed: bool = False
    n_fit_samples: int = 1
    # --- criterion in the crack's LOCAL frame (tangent t = direction of growth, normal n = t rotated +90 deg)
    # `jump_normal_px` is the SIGNED opening  (u(+n side) - u(-n side)) . n  -- positive when the two faces
    # move apart, negative when they overlap (it does not depend on the direction the path is walked);
    # `jump_tangential_px` is the absolute sliding |(u(+) - u(-)) . t|. `criterion` names which signal the
    # threshold walk uses: "magnitude" (hypot of the u and v jumps, the original criterion), "normal" (the
    # opening, negative values counted as 0) or "tangential"; `jump_criterion_px` is that signal, raw, and
    # `jump_mag_smoothed_px` above is IT after the rolling median.
    jump_normal_px: np.ndarray | None = None
    jump_tangential_px: np.ndarray | None = None
    jump_criterion_px: np.ndarray | None = None
    criterion: str = "magnitude"


JUMP_CRITERIA = [
    ("magnitude", "|jump| of (u, v) combined"),
    ("normal", "opening: jump normal to the crack"),
    ("tangential", "sliding: jump tangential to the crack"),
]


def _face_values(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, positions: np.ndarray,
                  normals: np.ndarray, sign: float, offset_px: float, n_fit: int) -> np.ndarray:
    """Value of `field` extrapolated to the crack face (offset 0) from ONE
    side of the path: the field is sampled at offset_px, 2*offset_px, ...,
    n_fit*offset_px along sign*normal and a straight line value = a + b*d is
    least-squares fitted per path point -- `a` (the intercept) is the value
    AT the face seen from that side. Samples falling outside the correlated
    ROI (NaN) are skipped; fewer than 2 valid samples -> NaN."""
    d = offset_px * np.arange(1, n_fit + 1, dtype=float)
    vals = np.column_stack([_sample_field(field, xs, ys, positions + sign * dj * normals) for dj in d])
    ok = np.isfinite(vals)
    cnt = ok.sum(axis=1)
    dd = np.broadcast_to(d, vals.shape)
    v0 = np.where(ok, vals, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dm = (dd * ok).sum(axis=1) / cnt
        ym = v0.sum(axis=1) / cnt
        dc = np.where(ok, dd - dm[:, None], 0.0)
        sxx = (dc ** 2).sum(axis=1)
        b = (dc * np.where(ok, vals - ym[:, None], 0.0)).sum(axis=1) / sxx
        a = ym - b * dm
    a[cnt < 2] = np.nan
    return a


def crack_opening_jump_profile(u_field: np.ndarray, v_field: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                                path_positions: np.ndarray, path_arc_length: np.ndarray,
                                offset_px: float = 4.0, threshold_px: float = 0.15,
                                max_gap_samples: int = 2, smooth_window_samples: int = 1,
                                tangent_window_px: float | None = None,
                                remove_continuous_part: bool = False,
                                n_fit_samples: int = 3,
                                criterion: str = "magnitude") -> CrackOpeningJumpProfile:
    """Crack-opening displacement jump directly from the raw, already-
    exported standard subset-DIC `u`/`v` fields, at every point of an
    already-traced path -- see this section's own header comment for the
    full motivation (a ground-truth cross-check independent of every tip-
    tracking method and of core.xfem_dic's own model fit).

    At each path point, samples `u_field`/`v_field` at two points offset
    `offset_px` to either side of the path, along its LOCAL NORMAL
    (perpendicular to the tangent -- so "either side of the crack face",
    not ahead/behind along it), and takes the absolute difference: a real
    open crack shows a genuine jump here (the two faces have displaced
    apart), a smooth strain gradient without real separation does not,
    however intense it looks. `jump_u_px`/`jump_v_px` are reported
    separately (a crack opening mostly perpendicular to one image axis may
    show a much cleaner jump on that axis than the other) alongside the
    combined `jump_mag_px = hypot(jump_u_px, jump_v_px)` -- the same
    combination `core.xfem_dic`'s own Heaviside-jump auto-detect uses on
    its fitted `heaviside_disp_x/y` (see `_refine_tip_s_from_heaviside_
    jump`), so `threshold_px` defaults to that feature's own default
    (0.15 px) for a directly comparable number, not a coincidence.

    `offset_px` needs to be large enough that the two sample points
    genuinely land on OPPOSITE sides of the (possibly several-px-wide, in
    a real correlated image) crack-face discontinuity, not both still
    within it (which would UNDER-report the jump, or catch pure correlation
    noise right at the discontinuity) -- but small enough to stay a "near-
    tip" measurement rather than picking up unrelated far-field motion.
    A handful of px is a reasonable start; if the reported jump looks
    unexpectedly small or noisy right where the standard-DIC field visibly
    shows a clean step, try a slightly larger value first.

    `opening_extent_s`/`opening_extent_xy` reuse this module's own
    `_threshold_walk`/`_interpolate_tip` (exactly `track_tip_threshold`'s
    own algorithm, applied to `jump_mag_px` instead of a raw exported
    field) to answer "how far along the path is `jump_mag_px` still
    continuously at/above `threshold_px`, walking from the path's own
    origin (index 0)". This assumes `path_positions[0]` is already
    genuinely open -- true for a path traced from a real crack's
    initiation point (this module's own convention throughout, see the
    module docstring), or from any point already known to be cracked (e.g.
    a pre-crack from an earlier wave passage). `max_gap_samples` tolerates
    a few consecutive below-threshold samples (correlation noise/a single
    bad point) without truncating the walk early, same convention as
    `track_tip_threshold`'s own parameter of the same name -- INCLUDING
    right at the origin itself: a single noisy frame doesn't need the
    origin to individually clear threshold, only for the walk to still
    find something at/above threshold within `max_gap_samples` samples of
    the start (see `_threshold_walk`). None only if nothing near the
    origin ever reaches threshold within that tolerance -- nothing to walk
    from.

    `tangent_window_px`, passed to `path_tangents`, defaults to `4.0 *
    offset_px` when left as `None` -- the normal direction used to place
    the two offset sample points needs to be reasonably free of the
    pixel-grid zigzag a real traced skeleton has at a 1-2px scale (see
    `path_tangents`'s own docstring); a raw per-point tangent would swing
    the sampling direction erratically enough, at this function's own
    (typically few-px) `offset_px` scale, to corrupt the jump measurement
    itself, not just its orientation.

    `remove_continuous_part` (default False = the plain two-point jump above):
    a PASSING ELASTIC WAVE, or any smooth strain around the crack, makes
    u(+offset) - u(-offset) non-zero even where the faces are NOT separated
    (it adds about strain * 2 * offset), which can push the jump over the
    threshold ahead of the real tip. With this option each side is sampled at
    `n_fit_samples` distances (offset, 2*offset, ...), a straight line is
    fitted per side and extrapolated to the face itself (offset 0), and the
    jump is the difference of those two face values: a smooth (continuous)
    field gives the same value from both sides -> ~0, whereas a real
    discontinuity (open crack) is kept. Costs a little more noise (the
    intercept of a 3-point line is ~1.5x noisier than a single sample) --
    average it with `smooth_window_samples` if needed. `offset_px` is then the
    distance of the NEAREST sample, which must still clear the crack-face
    damage zone, and the fit spans up to `n_fit_samples * offset_px` from the
    path.

    `criterion` ("magnitude" by default = the behavior described above): which
    signal the threshold walk is run on. "normal" uses only the component of the
    displacement jump PERPENDICULAR to the crack, in the crack's own local frame
    (the opening itself, negative = overlap counted as 0); "tangential" the
    component ALONG it (sliding, mode II). The normal criterion ignores a pure
    sliding or a rigid rotation of the two lips and is the one a wave travelling
    along the crack disturbs least; the tangential one follows shear. All three
    signals are always returned (`jump_mag_px`, `jump_normal_px`,
    `jump_tangential_px`).

    `smooth_window_samples` (default 1 = no smoothing, i.e. exactly this
    function's original behavior): the walk above -- and therefore
    `opening_extent_s`/`opening_extent_xy` -- is run against
    `jump_mag_smoothed_px`, `jump_mag_px` after a rolling median filter
    of this width (`_nanmedian_filter_1d`, see its own docstring), NOT the
    raw `jump_mag_px` (still reported unfiltered, alongside it, in the
    returned profile). This exists to reject a transient, spatially
    narrow disturbance -- most notably an elastic wave passing through
    the DIC field of view, which can locally push `jump_mag_px` above
    `threshold_px` for a handful of path samples without the crack having
    genuinely opened there -- while leaving a real, sustained opening
    plateau (wide by construction) essentially unaffected. Pick a window
    a bit wider than such a disturbance's own visible extent in the
    "Verify visually" jump-profile plot, but comfortably narrower than
    the shortest genuine opening length expected near the crack's own
    initiation."""
    n = len(path_positions)
    if n < 2:
        raise CrackTipTrackingError(
            "The traced path needs at least 2 points to measure a crack-opening profile."
        )
    window_px = tangent_window_px if tangent_window_px is not None else 4.0 * offset_px
    tangents = path_tangents(path_positions, path_arc_length, window_px=window_px)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    n_fit = max(2, int(n_fit_samples)) if remove_continuous_part else 1
    if remove_continuous_part:
        u_plus = _face_values(u_field, xs, ys, path_positions, normals, +1.0, offset_px, n_fit)
        u_minus = _face_values(u_field, xs, ys, path_positions, normals, -1.0, offset_px, n_fit)
        v_plus = _face_values(v_field, xs, ys, path_positions, normals, +1.0, offset_px, n_fit)
        v_minus = _face_values(v_field, xs, ys, path_positions, normals, -1.0, offset_px, n_fit)
    else:
        plus_pts = path_positions + offset_px * normals
        minus_pts = path_positions - offset_px * normals
        u_plus = _sample_field(u_field, xs, ys, plus_pts)
        u_minus = _sample_field(u_field, xs, ys, minus_pts)
        v_plus = _sample_field(v_field, xs, ys, plus_pts)
        v_minus = _sample_field(v_field, xs, ys, minus_pts)

    if criterion not in dict(JUMP_CRITERIA):
        raise CrackTipTrackingError(f"Unknown opening-jump criterion '{criterion}'.")
    du, dv = u_plus - u_minus, v_plus - v_minus
    jump_u = np.abs(du)
    jump_v = np.abs(dv)
    jump_mag = np.hypot(jump_u, jump_v)
    # components in the crack's local frame: n = (-ty, tx), t = tangent
    jump_n_signed = du * normals[:, 0] + dv * normals[:, 1]
    jump_t = np.abs(du * tangents[:, 0] + dv * tangents[:, 1])
    with np.errstate(invalid="ignore"):
        jump_n_open = np.where(np.isfinite(jump_n_signed), np.maximum(jump_n_signed, 0.0), np.nan)
    jump_crit = {"magnitude": jump_mag, "normal": jump_n_open, "tangential": jump_t}[criterion]
    jump_mag_smoothed = _nanmedian_filter_1d(jump_crit, smooth_window_samples)

    opening_extent_s = None
    opening_extent_xy = None
    idx = _threshold_walk(jump_mag_smoothed, threshold_px, max_gap_samples)
    if idx is not None:
        opening_extent_xy, opening_extent_s = _interpolate_tip(
            jump_mag_smoothed, path_positions, path_arc_length, idx, threshold_px
        )

    return CrackOpeningJumpProfile(
        arc_length_s=path_arc_length, positions_xy=path_positions,
        jump_u_px=jump_u, jump_v_px=jump_v, jump_mag_px=jump_mag,
        jump_mag_smoothed_px=jump_mag_smoothed, smooth_window_samples=smooth_window_samples,
        offset_px=offset_px, threshold_px=threshold_px,
        opening_extent_s=opening_extent_s, opening_extent_xy=opening_extent_xy,
        continuous_part_removed=bool(remove_continuous_part), n_fit_samples=n_fit,
        jump_normal_px=jump_n_signed, jump_tangential_px=jump_t, jump_criterion_px=jump_crit,
        criterion=criterion,
    )


def export_opening_jump_profile_csv(profile: CrackOpeningJumpProfile, path: str,
                                     frame_index: int | None = None,
                                     time_s: float | None = None) -> None:
    """Writes ONE frame's opening-jump profile (as drawn in the "Verify
    visually" card's "Opening-jump profile along the path -- this frame"
    plot) to a CSV, one row per point of the traced path -- the raw
    `jump_u_px`/`jump_v_px`/`jump_mag_px` signal this diagnostic is built
    from, not just what the plot shows (the threshold line and detected-
    front marker), so the data can be re-examined or re-thresholded
    outside the app. `frame_index`/`time_s`, when given, are written as
    constant columns (repeated on every row) purely for bookkeeping --
    this function itself has no notion of "frame", it just serializes one
    already-computed `CrackOpeningJumpProfile`.

    `jump_mag_smoothed_px` is included alongside the raw `jump_mag_px` --
    identical to it when `profile.smooth_window_samples <= 1`, otherwise
    the rolling-median-filtered signal `opening_extent_s` was actually
    walked from (see `crack_opening_jump_profile`'s own docstring).

    `is_open_at_this_point` is 1 for every path point at/before the
    detected `opening_extent_s` (i.e. "the walk from the path's own
    origin reached here without the jump ever dropping back below
    threshold for more than `max_gap_samples`"), 0 after it, and blank
    throughout if `opening_extent_s` is None (threshold never reached at
    all this frame -- see the module's "MEASUREMENT UNCERTAINTY" section
    and `crack_opening_jump_profile`'s own docstring for what that means).

    Raises `OSError` on a write failure, left for the caller to report
    as-is (mirrors `export_tip_roi_csv`)."""
    import csv

    n = profile.arc_length_s.size
    is_open = np.full(n, "", dtype=object)
    if profile.opening_extent_s is not None:
        is_open[profile.arc_length_s <= profile.opening_extent_s] = "1"
        is_open[profile.arc_length_s > profile.opening_extent_s] = "0"

    header = []
    if frame_index is not None:
        header.append("frame_index")
    if time_s is not None:
        header.append("time_s")
    header += ["arc_length_px", "x_px", "y_px", "jump_u_px", "jump_v_px", "jump_mag_px",
               "jump_mag_smoothed_px", "smooth_window_samples",
               "offset_px", "threshold_px", "opening_extent_s_px", "is_open_at_this_point",
               "jump_normal_signed_px", "jump_tangential_px", "criterion"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for k in range(n):
            row = []
            if frame_index is not None:
                row.append(frame_index)
            if time_s is not None:
                row.append(f"{time_s:.9g}")
            row += [
                f"{profile.arc_length_s[k]:.6f}",
                f"{profile.positions_xy[k, 0]:.6f}", f"{profile.positions_xy[k, 1]:.6f}",
                f"{profile.jump_u_px[k]:.6g}", f"{profile.jump_v_px[k]:.6g}", f"{profile.jump_mag_px[k]:.6g}",
                f"{profile.jump_mag_smoothed_px[k]:.6g}", str(profile.smooth_window_samples),
                f"{profile.offset_px:.6g}", f"{profile.threshold_px:.6g}",
                f"{profile.opening_extent_s:.6f}" if profile.opening_extent_s is not None else "",
                is_open[k],
                f"{profile.jump_normal_px[k]:.6g}" if profile.jump_normal_px is not None else "",
                f"{profile.jump_tangential_px[k]:.6g}" if profile.jump_tangential_px is not None else "",
                profile.criterion,
            ]
            writer.writerow(row)


# -- measurement uncertainty (see module docstring, "MEASUREMENT UNCERTAINTY") --

_UNCERTAINTY_NOISE_WINDOW_SAMPLES = 15   # each side of the detected front, for `_local_noise_sigma`
_UNCERTAINTY_OFFSET_PERTURBATION_FRAC = 0.25  # relative offset_px perturbation for the sensitivity bracket
_UNCERTAINTY_GAP_PERTURBATION_SAMPLES = 1


def _local_noise_sigma(values: np.ndarray, center_idx: int,
                        window: int = _UNCERTAINTY_NOISE_WINDOW_SAMPLES) -> float:
    """Blind noise-sigma estimate for a 1D array, from the scatter of its
    own consecutive first differences in a window around `center_idx` --
    the 1D analog of `core.dic_uncertainty.estimate_noise_sigma`'s 2D
    Immerkaer estimator (same reasoning: a smooth underlying signal
    contributes little to a FIRST difference, especially over the few-
    sample-wide window used here, so the difference is mostly noise).
    For i.i.d. Gaussian noise of std `sigma`, two independent samples
    differ by a Gaussian of std `sigma*sqrt(2)`, whose own mean absolute
    value is `sigma*sqrt(2)*sqrt(2/pi) = sigma*2/sqrt(pi)` -- inverted
    below to recover `sigma` from the observed mean absolute difference.
    Restricting to a local window (rather than the whole profile) matters
    because `jump_mag_px` is NOT stationary -- near-zero ahead of the
    crack, elevated behind it -- so a whole-profile scatter would conflate
    real signal variation with noise; a first-difference estimator is
    already robust to slow local trends within that window on its own,
    this window just keeps it away from the far-field level change too.
    Returns 0.0 if there are too few finite samples in the window to say
    anything (never NaN -- a missing noise estimate should not poison a
    caller's quadrature sum)."""
    n = len(values)
    lo = max(0, center_idx - window)
    hi = min(n, center_idx + window + 1)
    local = np.asarray(values[lo:hi], dtype=float)
    finite = local[np.isfinite(local)]
    if finite.size < 3:
        return 0.0
    diffs = np.diff(finite)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 0.0
    return float(np.mean(np.abs(diffs)) * np.sqrt(np.pi) / 2.0)


def opening_extent_position_uncertainty_px(
        frame_u: np.ndarray, frame_v: np.ndarray, xs: np.ndarray, ys: np.ndarray,
        positions: np.ndarray, arc_length: np.ndarray, profile: CrackOpeningJumpProfile,
        max_gap_samples: int, tangent_window_px: float | None = None,
        smooth_window_samples: int = 1, remove_continuous_part: bool = False,
        n_fit_samples: int = 3, criterion: str = "magnitude") -> tuple[float, float]:
    """One frame's crack-front position uncertainty, in px of arc length
    -- see the module docstring's "MEASUREMENT UNCERTAINTY" section for
    the full method (this is its implementation). `profile` must already
    be THIS frame's own `crack_opening_jump_profile(...)` result, computed
    with the nominal `offset_px`/`threshold_px` this call also receives
    implicitly (via `profile.offset_px`/`profile.threshold_px`) --
    `frame_u`/`frame_v`/`xs`/`ys`/`positions`/`arc_length` are only needed
    to re-run the offset perturbation (a genuine re-sample, unlike the
    threshold/gap-tolerance perturbations which reuse `profile.jump_mag_px`
    as-is). Returns `(position_uncertainty_px, sigma_jump_px)` -- the
    second value exposed too since it's independently useful context (the
    blind noise estimate the threshold bracket was built from). Returns
    `(nan, nan)` when `profile.opening_extent_s` is None -- nothing
    detected this frame, nothing to bracket.

    `smooth_window_samples` must be the SAME value `profile` was itself
    computed with (`profile.smooth_window_samples`, normally just passed
    straight through by the caller) -- every re-walk below reuses
    `profile.jump_mag_smoothed_px`, and the one perturbation that
    genuinely re-samples the raw fields (offset, see (3)) re-derives its
    own smoothed signal the identical way, so this bracket stays
    self-consistent with whatever `opening_extent_s` was actually walked
    from rather than silently reverting to the unfiltered signal."""
    if profile.opening_extent_s is None:
        return float("nan"), float("nan")
    s_nominal = profile.opening_extent_s
    threshold_px = profile.threshold_px
    offset_px = profile.offset_px

    walk_idx = _threshold_walk(profile.jump_mag_smoothed_px, threshold_px, max_gap_samples)
    sigma_jump = _local_noise_sigma(profile.jump_mag_smoothed_px, walk_idx if walk_idx is not None else 0)

    def _bracket_half_width(candidates: list) -> float:
        return (max(candidates) - min(candidates)) / 2.0 if len(candidates) > 1 else 0.0

    # -- (1) threshold vs the jump signal's own local noise -- reuses the
    # already-computed jump_mag_smoothed_px, no field resampling.
    s_threshold_candidates = [s_nominal]
    for perturbed_threshold in (threshold_px - sigma_jump, threshold_px + sigma_jump):
        if perturbed_threshold <= 0 or sigma_jump <= 0:
            continue
        idx = _threshold_walk(profile.jump_mag_smoothed_px, perturbed_threshold, max_gap_samples)
        if idx is not None:
            _, s = _interpolate_tip(profile.jump_mag_smoothed_px, profile.positions_xy, profile.arc_length_s,
                                     idx, perturbed_threshold)
            s_threshold_candidates.append(s)
    delta_threshold = _bracket_half_width(s_threshold_candidates)

    # -- (2) gap tolerance -- same jump_mag_smoothed_px, +/- 1 sample of tolerance.
    s_gap_candidates = [s_nominal]
    for perturbed_gap in (max(0, max_gap_samples - _UNCERTAINTY_GAP_PERTURBATION_SAMPLES),
                          max_gap_samples + _UNCERTAINTY_GAP_PERTURBATION_SAMPLES):
        idx = _threshold_walk(profile.jump_mag_smoothed_px, threshold_px, perturbed_gap)
        if idx is not None:
            _, s = _interpolate_tip(profile.jump_mag_smoothed_px, profile.positions_xy, profile.arc_length_s,
                                     idx, threshold_px)
            s_gap_candidates.append(s)
    delta_gap = _bracket_half_width(s_gap_candidates)

    # -- (3) sampling offset -- a genuinely different measurement (new
    # u/v samples at a different offset from the path), the only one of
    # the three needing the raw fields again.
    s_offset_candidates = [s_nominal]
    delta_offset_px = max(1.0, offset_px * _UNCERTAINTY_OFFSET_PERTURBATION_FRAC)
    for perturbed_offset in (offset_px - delta_offset_px, offset_px + delta_offset_px):
        if perturbed_offset <= 0:
            continue
        try:
            perturbed_profile = crack_opening_jump_profile(
                frame_u, frame_v, xs, ys, positions, arc_length,
                offset_px=perturbed_offset, threshold_px=threshold_px,
                max_gap_samples=max_gap_samples, smooth_window_samples=smooth_window_samples,
                tangent_window_px=tangent_window_px,
                remove_continuous_part=remove_continuous_part, n_fit_samples=n_fit_samples,
                criterion=criterion,
            )
        except CrackTipTrackingError:
            continue
        if perturbed_profile.opening_extent_s is not None:
            s_offset_candidates.append(perturbed_profile.opening_extent_s)
    delta_offset = _bracket_half_width(s_offset_candidates)

    # -- (4) path/pixel-resolution floor -- independent of the above.
    resample_step_px = float(np.median(np.diff(arc_length))) if len(arc_length) > 1 else 0.0
    delta_grid = resample_step_px / np.sqrt(12.0)

    total = float(np.sqrt(delta_threshold ** 2 + delta_gap ** 2 + delta_offset ** 2 + delta_grid ** 2))
    return total, sigma_jump


# ---------------------------------------------------------------------------
# Automatic tip tracking from the raw u/v opening jump, over an ENTIRE
# frame range -- built on `crack_opening_jump_profile` above, the way
# `extract_tip_ridge_in_polygon`/`extract_tip_arrival_time_in_polygon` are
# built on `build_tip_field_profile`. Replaces the former "ridge (ROI)"
# method (and the "manual (click)" tool) as this page's primary automatic
# tip-tracking method: reported in practice to lose the true crack tip
# early because both of those depend on a SINGLE derived scalar field
# (strain, or whatever `field_key` is chosen) crossing a threshold or
# forming a locally strongest peak, exactly on the traced path -- whereas
# this measures the raw DISPLACEMENT DISCONTINUITY across the path
# directly from u/v, which stays sharp and unambiguous (open vs not open)
# even where the derived-field signal has already faded or is offset from
# the traced path's own exact line.
#
# Unlike "ridge (ROI)"/"arrival (ROI)", there is no polygon to draw: the
# walk-from-origin-with-tolerance-for-a-few-gaps algorithm inside
# `crack_opening_jump_profile` is itself the whole detector, so this runs
# fully automatically over every frame in range, the same "just click
# Track/Add, no manual picking" spirit as the page's very first
# TipTrackResult-returning methods (`track_tip_threshold`/`track_tip_
# singular_fit`/`track_tip_ridge`, all core-only/not wired into the page --
# see the module docstring). This one IS wired into the page, because
# unlike those, it returns the SAME `(times, arc_length, values,
# source_frame_positions, profile)` shape `extract_tip_ridge_in_polygon`/
# `extract_tip_arrival_time_in_polygon` return, for drop-in use building a
# `PolygonTipRidgeResult` -- see that class's docstring for why every
# method here lands in that one shared shape.
# ---------------------------------------------------------------------------

def track_tip_opening_jump(dataset, crack, leaf_index: int,
                            offset_px: float = 4.0, threshold_px: float = 0.15,
                            frame_range: tuple | None = None, frame_rate_hz_override: float | None = None,
                            resample_step_px: float = 2.0, max_gap_samples: int = 2,
                            tangent_window_px: float | None = None, reverse_path: bool = False,
                            enforce_monotonic: bool = True, backslide_tolerance_px: float | None = None,
                            estimate_uncertainty: bool = False, smooth_window_samples: int = 1,
                            advance_confirm_frames: int = 1, remove_continuous_part: bool = False,
                            n_fit_samples: int = 3, jump_criterion: str = "magnitude"):
    """Fully automatic crack-tip tracking straight from the raw standard-
    DIC u/v displacement jump (see `crack_opening_jump_profile`), run over
    EVERY frame in range along the SAME traced path `build_tip_field_
    profile` resamples -- see this section's header comment for the
    motivation and how this compares to "ridge (ROI)"/"arrival (ROI)".

    At each frame, this is exactly `crack_opening_jump_profile(frame.u,
    frame.v, ..., offset_px, threshold_px)`'s own `opening_extent_s`/
    `opening_extent_xy` -- how far the combined |u|/|v| jump stays
    continuously at/above `threshold_px` from the path's own origin
    (tolerating up to `max_gap_samples` consecutive below-threshold
    samples anywhere along that walk, including right at the origin
    itself -- see `_threshold_walk`). A frame whose jump never reaches
    threshold anywhere near the origin, even with that tolerance (nothing
    open there yet, or the whole crack has closed) contributes NO point
    for that frame -- left out entirely, not a NaN placeholder, the same
    convention `extract_tip_ridge_in_polygon`'s per-frame picks and
    `finalize_manual_tip_pick`'s skipped frames both already use.

    `remove_continuous_part` / `n_fit_samples` (default off) are forwarded to
    every `crack_opening_jump_profile` call: the jump is then the
    DISCONTINUITY at the crack face (each side extrapolated to the face), which
    rejects a passing elastic wave / smooth strain that would otherwise add
    ~strain*2*offset to the plain two-point jump -- see that function.

    `jump_criterion` ("magnitude" default, "normal" or "tangential") is forwarded
    as `criterion`: the threshold walk then runs on the combined |jump|, on the
    OPENING (jump normal to the crack, in its local frame) or on the sliding
    (tangential jump) -- see `crack_opening_jump_profile`.

    `smooth_window_samples` (default 1, meaning off) is forwarded, unchanged,
    to every internal `crack_opening_jump_profile` call -- see that
    function's own docstring. It suppresses narrow, very-high-amplitude
    spikes in the |u|/|v| jump signal along the traced path, of the kind a
    passing elastic wave produces (short-lived, spatially narrow pulses,
    NOT a sustained crack-opening plateau), before this method's own
    origin-anchored threshold walk runs on it. This is a SPATIAL filter
    (along the path, within one frame) -- a different axis and a different
    problem from `enforce_monotonic`/`backslide_tolerance_px` below, which
    police the sequence of already-per-frame results ACROSS frames.

    `enforce_monotonic` (default True) applies the SAME physical prior
    "ridge (ROI)"/"arrival (ROI)" do: once the raw per-frame points are
    ordered by time, a point is kept only if its arc-length position isn't
    more than `backslide_tolerance_px` (default 3x `resample_step_px`)
    behind the furthest position already kept, in whichever direction
    (increasing or decreasing arc length) the raw, unfiltered points trend
    overall. In principle this method's own origin-anchored walk already
    can't report a SHORTER extent than genuinely reached within one frame,
    but frame-to-frame measurement noise (a slightly weaker jump in one
    frame purely from image noise, not a real crack closing) can still
    make consecutive reported points dip and recover -- this smooths that
    the same way it already does for the other two methods, without
    hiding a genuine, sustained retreat (an actually closing crack) any
    more aggressively than they do.

    A wave pulse sweeping past the traced path can, on the frame(s) it
    crosses, still push the reported extent FORWARD by a large amount even
    after `smooth_window_samples` filtering (the pulse can sit ON the path
    at a point beyond the true tip). The old behaviour accepted any such
    forward move immediately and permanently raised the running high-water
    mark to it, corrupting every later frame's backslide comparison --
    `advance_confirm_frames` (default 1, meaning off: reproduces the old
    immediate-accept behaviour exactly) fixes this WITHOUT assuming
    anything about crack speed (deliberately -- crack velocity is far too
    variable in a heterogeneous material like granite to bound it that
    way). Instead, any candidate point that jumps more than
    `backslide_tolerance_px` beyond the current high-water mark is held
    back until `advance_confirm_frames` consecutive raw points starting at
    it (itself included) all stay at/above that candidate's own level
    (within `backslide_tolerance_px`) -- i.e. the advance has to hold up
    over several frames, not just flash on one, before it is trusted as
    the new high-water mark. A move that stays within
    `backslide_tolerance_px` of the current mark is still accepted
    immediately either way, exactly as before. Near the very end of the
    frame range there may not be enough remaining raw points left to
    confirm a genuine fast advance -- those last `advance_confirm_frames -
    1` points are then conservatively dropped rather than risk locking in
    an unconfirmed spike (see `doc_build/10_crack_wave_analysis.md`).

    Returns `(times, arc_length, values, source_frame_positions, profile,
    position_uncertainty_px)` -- see this section's header comment.
    `values` is each reported point's combined jump magnitude (px) at the
    last raw path point the walk itself accepted as still at/above
    threshold (NOT re-sampled at the interpolated `arc_length` position,
    which can land past that point, already below threshold -- see the
    code comment where this is computed), kept for CSV/context, like
    `extract_tip_arrival_time_in_polygon`'s own `values`; never used to
    determine the position itself). `profile` is a `TipFieldProfile` with
    a real `path_positions`/`path_arc_length`/`times`/`frame_indices`
    (needed by the smoothing option and the page's own path-geometry
    bookkeeping) but an all-NaN `values` grid -- this method never builds
    the full (arc length x time) map those OTHER methods draw as a heat
    map, since there's nothing to draw a ridge through here.

    `estimate_uncertainty` (default False): when True, additionally
    computes each kept point's own `position_uncertainty_px` (see the
    module docstring's "MEASUREMENT UNCERTAINTY" section and
    `opening_extent_position_uncertainty_px`) -- otherwise `position_uncertainty_px`
    is all-NaN, same length as `arc_length` either way, so callers can
    always unpack 6 values without a branch. Roughly triples the per-frame
    cost (the offset-sensitivity term re-samples u/v two more times), so
    left off by default."""
    paths = branch_paths(crack)
    if leaf_index not in paths:
        raise CrackTipTrackingError(f"No traced path from the crack's origin (its first point) to node {leaf_index}.")
    node_points = [crack.nodes[i] for i in paths[leaf_index]]
    if reverse_path:
        node_points = list(reversed(node_points))
    positions, arc_length = resample_path(node_points, resample_step_px)

    order, times_all = _frame_order(dataset, frame_rate_hz_override, frame_range)
    if backslide_tolerance_px is None:
        backslide_tolerance_px = 3.0 * resample_step_px

    times_raw: list[float] = []
    arclen_raw: list[float] = []
    values_raw: list[float] = []
    frameidx_raw: list[int] = []
    uncertainty_raw: list[float] = []
    for fi, oi in enumerate(order):
        frame = dataset.frames[oi]
        profile = crack_opening_jump_profile(
            frame.u, frame.v, dataset.xs, dataset.ys, positions, arc_length,
            offset_px=offset_px, threshold_px=threshold_px,
            max_gap_samples=max_gap_samples, tangent_window_px=tangent_window_px,
            smooth_window_samples=smooth_window_samples,
            remove_continuous_part=remove_continuous_part, n_fit_samples=n_fit_samples,
            criterion=jump_criterion,
        )
        if profile.opening_extent_s is None:
            continue
        times_raw.append(float(times_all[fi]))
        arclen_raw.append(float(profile.opening_extent_s))
        # Re-run the SAME walk `crack_opening_jump_profile` used internally
        # to find `opening_extent_s`, to recover the raw path-point index
        # it interpolated from -- NOT the path point nearest the
        # interpolated position itself: `_interpolate_tip` can land the
        # interpolated crossing more than halfway to the NEXT (already
        # below-threshold) point, and reading `jump_mag_px` at THAT nearest
        # point would silently report a value below (sometimes far below)
        # threshold right where the trace claims the crack is still open.
        # (Walked on the SAME smoothed signal `opening_extent_s` itself was
        # derived from, per `profile.jump_mag_smoothed_px`.)
        walk_idx = _threshold_walk(profile.jump_mag_smoothed_px, threshold_px, max_gap_samples)
        values_raw.append(float(profile.jump_criterion_px[walk_idx]) if walk_idx is not None else float("nan"))
        frameidx_raw.append(int(oi))
        if estimate_uncertainty:
            unc_px, _sigma_jump = opening_extent_position_uncertainty_px(
                frame.u, frame.v, dataset.xs, dataset.ys, positions, arc_length,
                profile, max_gap_samples, tangent_window_px,
                smooth_window_samples=smooth_window_samples,
                remove_continuous_part=remove_continuous_part, n_fit_samples=n_fit_samples,
                criterion=jump_criterion,
            )
            uncertainty_raw.append(unc_px)
        else:
            uncertainty_raw.append(float("nan"))

    field_profile = TipFieldProfile(
        path_positions=positions, path_arc_length=arc_length,
        values=np.full((len(positions), len(order)), np.nan),
        times=times_all, frame_indices=np.asarray(order, dtype=int),
    )
    if not times_raw:
        return (np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float),
                np.array([], dtype=int), field_profile, np.array([], dtype=float))

    times_arr = np.asarray(times_raw, dtype=float)
    arclen_arr = np.asarray(arclen_raw, dtype=float)
    values_arr = np.asarray(values_raw, dtype=float)
    frameidx_arr = np.asarray(frameidx_raw, dtype=int)
    uncertainty_arr = np.asarray(uncertainty_raw, dtype=float)

    if enforce_monotonic and times_arr.size >= 2:
        slope = float(np.polyfit(times_arr, arclen_arr, 1)[0])
        direction = 1 if slope >= 0 else -1
        n = times_arr.size
        # Signed transform so "forward" always means "increasing" -- lets a
        # single rule below handle both trace directions symmetrically
        # instead of duplicating it with the comparisons flipped.
        signed = direction * arclen_arr
        keep = np.zeros(n, dtype=bool)
        running_extreme_signed = None
        for k in range(n):
            v = signed[k]
            if running_extreme_signed is None:
                keep[k] = True
                running_extreme_signed = v
                continue
            if v < running_extreme_signed - backslide_tolerance_px:
                # Retreat beyond tolerance: reject, same as before.
                continue
            if v <= running_extreme_signed + backslide_tolerance_px:
                # Within the noise band around the current high-water
                # mark (this covers ordinary forward progress one
                # resample step at a time too): accept immediately, same
                # as before.
                keep[k] = True
                running_extreme_signed = max(running_extreme_signed, v)
                continue
            # v > running_extreme_signed + backslide_tolerance_px: a
            # sudden, large forward jump -- exactly what a short,
            # high-intensity wave pulse produces on the frame(s) it
            # crosses the path. NOT bounded by an assumed crack speed
            # (deliberately -- see this function's docstring): instead
            # only accepted once `advance_confirm_frames` consecutive RAW
            # points starting here (this one included) all hold up
            # at/above this candidate's own level. `advance_confirm_frames`
            # =1 means that window is this point alone, which trivially
            # confirms itself, so the default reproduces the old
            # immediate-accept behaviour exactly.
            window_end = k + advance_confirm_frames
            if window_end > n:
                # Not enough future raw points left to confirm -- near the
                # very end of the range, conservatively drop rather than
                # risk locking in an unconfirmed spike.
                continue
            if np.all(signed[k:window_end] >= v - backslide_tolerance_px):
                keep[k] = True
                running_extreme_signed = max(running_extreme_signed, v)
            # else: unconfirmed spike -- reject, running_extreme unchanged.
        times_arr = times_arr[keep]
        arclen_arr = arclen_arr[keep]
        values_arr = values_arr[keep]
        frameidx_arr = frameidx_arr[keep]
        uncertainty_arr = uncertainty_arr[keep]

    return times_arr, arclen_arr, values_arr, frameidx_arr, field_profile, uncertainty_arr
