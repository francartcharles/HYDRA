"""
Elastic wave speed (X / Y) from a DIC-captured strain-field time series --
Stage 2 of the Crack & Wave Analysis standalone tool (see
core/crack_wave_io.py for Stage 1's self-sufficient data loading, and
ui/pages/page_crack_wave_analysis.py for the page this feeds).

Method: time-of-flight wavefront tracking, the same idea used to read a
wave's propagation speed off a seismic "shot gather" or an ultrasonic
B-scan. For the direction being measured (X or Y):

1. Average the chosen strain field across the PERPENDICULAR direction at
   every position along the measured one -- e.g. for an X-direction
   speed, average over y at each x -- so a roughly planar wavefront
   sweeping along X is reinforced while any Y-direction noise/variation
   averages out, instead of blurring the front.
2. Pick that front's arrival time at every position: a threshold-fraction
   (half-max by default) crossing of each position's own local signal,
   relative to its own baseline and peak -- the standard "first break"
   picking convention in wave/impact testing, robust to the wave's
   amplitude decaying with propagation distance (a single fixed absolute
   threshold would not be).
3. Fit position = speed * arrival_time + x0 across every position with a
   detected arrival -- one linear regression using every column/row in
   the field, not just two points, which is both more precise and comes
   with a directly checkable fit quality (R^2) and a sanity-checkable
   intercept (where/when the front would have started).

This assumes the tracked wavefront is roughly planar and travels mostly
along the direction being measured, over the region and frame range used
-- true near an edge-launched wave early in its propagation, which is the
situation the strain-field video that motivated this tool actually shows.
It is not a general multi-direction wave-tracking method: if the fitted
R^2 is poor, the front is likely curved, contaminated by a later crack
signal (see `frame_range` below), or the threshold picked up noise rather
than the real front -- not something to paper over with a fancier fit.

A single straight-line fit also breaks down when TWO wavefronts travel in
OPPOSITE directions from a common source inside the imaged region (e.g.
an impact point roughly in the middle of the field of view) -- arrival
time then has a V shape against position (earliest at the source,
increasing on both sides), not a monotonic ramp, so one fit across the
whole region averages the two opposite slopes into a near-zero, near-
meaningless one (a very low R^2 is the tell). `position_range` (see
`fit_wave_speed`/`compute_wave_speed`) restricts the FIT (not the
arrival-time detection, which still runs over every position, so the
V shape is still visible in the returned profile/plot either way) to one
side of the source at a time -- call `compute_wave_speed` twice, once per
side, to get each branch's own speed.

Manually splitting by position range works but doesn't scale: a field
showing several reflections (a wave bouncing back and forth across the
imaged region, each bounce reversing its sign of travel) can show many
more than two branches, crossing each other in X patterns rather than
sitting in cleanly separable position bands -- no single Min/Max position
split isolates any one of them. `detect_wave_branches` (and the ridge
extraction it is built on, `detect_ridge_points`) automates this instead
of asking for a manual split:

1. Unlike the single-arrival-per-position method above (built for a front
   that, once arrived, stays elevated), this looks for a *transient
   pulse*: at each individual TIME FRAME, it finds every contiguous band
   along the position axis that clears the arrival threshold -- i.e.
   "where is the field elevated right now", frame by frame -- and reports
   ONE ridge point per band, at that band's amplitude-weighted CENTROID
   (see `_band_centroids`), not its single tallest pixel. A real band has
   physical width, and per-pixel measurement noise shifts exactly which
   pixel is tallest from frame to frame; averaging over the whole band
   cancels that noise (roughly root-N in the band's width) while still
   tracking genuine sub-pixel motion of its center. A wavefront crossing
   the field, and each of its reflections, shows up as a moving band, so
   this produces a scatter of (time, position) "ridge points" that traces
   out every moving band simultaneously, including ones that cross each
   other in time, which a single per-position arrival time could never
   represent (it can only record ONE arrival time per position).
2. Those ridge points are then split into straight-line branches with
   sequential RANSAC: repeatedly fit the line (position = speed * time +
   intercept) with the most inlier ridge points (within a small position
   tolerance), remove those inliers, and repeat on what's left until too
   few points remain or `max_branches` is reached. This needs no prior
   knowledge of how many waves/reflections are present or which direction
   each one travels -- it finds however many roughly-straight branches
   the ridge scatter actually contains.

This ridge/RANSAC route assumes the field genuinely shows a transient
pulse (rises then falls back down as the wave passes a given position) --
true for an elastic wave with little to no permanent/plastic strain left
behind, which is the common case this tool targets. If the field instead
shows a permanent step (strain that stays elevated forever after the
front passes, e.g. a growing crack's near-tip field, or a genuinely
non-elastic residual strain), there IS no "local maximum" for the ridge
step to find once the front has passed -- use the single-front method
(`compute_wave_speed`, above) for that case instead.

Both automatic methods above still share one structural limit: they pick
one threshold (fixed, or per-frame-normalized) and trust it to separate
"the wave" from "everything else" across the WHOLE field at once. A real
recording can show several fronts at once with genuinely different
intensity -- one saturated red, another only ever reaching yellow/green --
and no single threshold fraction is above the weak one and below the
strong one's off-band noise/background at the same time; asking for the
weak front raises false detections elsewhere, asking to exclude those
loses the weak front entirely.

`extract_ridge_in_polygon` sidesteps thresholding altogether: the user
draws a polygon loosely around ONE visible band directly on the plot
(click each vertex, close the polygon) -- exactly the standard way a
PDV/VISAR spectrogram's frequency ridge is picked out by hand -- and
every time frame that polygon spans contributes one ridge point, at that
frame's own local extremum (max or min, either sign) among the enclosed
samples, picked out among competing candidates by which is closest to the
polygon's own enclosed span rather than which is simply strongest, then
lightly refined by a tight centroid over its immediate neighbors (see the
function's own docstring for the full two-step reasoning). Nothing here
reads `threshold_fraction` at all: the polygon itself is what isolates
the front of interest, at whatever true intensity it has, so a faint band
is exactly as easy to pick as a saturated one. It is also not constrained
to a straight line -- no RANSAC/line fit is involved --
so a genuinely curved (accelerating/decelerating, or non-elastic) front
is traced as faithfully as a straight one; a rough indicative speed/R^2
is still reported (a plain OLS fit through the extracted points), for
convenience, but the raw (time, position) trace is the real result, and
several can be picked, named, and exported to one CSV
(`export_roi_waves_csv`) for further analysis outside this tool.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as _MplPath

from core.crack_skeleton import DEFAULT_COLORS

# How many position-axis grid steps either side of a polygon column's own
# local extremum still count as "the same feature" for the light sub-pixel
# centroid refinement in `extract_ridge_in_polygon` -- small on purpose,
# see that function's docstring for why a wide window defeats the point.
_POLYGON_LOCAL_WINDOW = 2

# Smoothing radius (grid steps either side) used ONLY to decide WHERE a
# polygon column's real local extrema are in `extract_ridge_in_polygon`,
# before per-pixel measurement noise is allowed to matter -- see that
# function's docstring for why comparing raw, unsmoothed neighbors sample
# by sample doesn't work on real (noisy) data.
_POLYGON_PEAK_SMOOTH_RADIUS = 2


def _smooth_1d(x: np.ndarray, radius: int) -> np.ndarray:
    """Centered moving-average smoothing, edge-padded so the output stays
    the same length as `x` with no boundary attenuation. Used only to find
    approximately WHERE a real local extremum sits amid per-pixel noise --
    every reported position/value in `extract_ridge_in_polygon` always
    comes from the RAW (this function's input, unsmoothed) signal, refined
    by a small centroid, never from this smoothed copy itself."""
    if radius <= 0 or x.size <= 2 * radius:
        return x
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    padded = np.pad(x, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


class WaveSpeedError(ValueError):
    """Raised when a wave speed estimate can't be computed as asked --
    message is shown to the user as-is, so kept short and specific."""


@dataclass
class ArrivalTimeProfile:
    positions: np.ndarray      # x (or y) pixel coordinate of each profile location, length n_positions
    arrival_times: np.ndarray  # same length, seconds; NaN where no threshold crossing was found
    signal: np.ndarray         # (n_positions, n_frames_used) the averaged profile itself, for the x-t plot
    times: np.ndarray          # (n_frames_used,) the time axis `signal` is sampled at, sorted ascending


@dataclass
class WaveSpeedResult:
    direction: str          # "x" or "y"
    field_key: str
    threshold_fraction: float
    profile: ArrivalTimeProfile
    speed_px_per_s: float           # signed: negative means the front travels toward decreasing x/y
    speed_mm_per_s: float | None    # None unless a pixel size (mm/px) was given
    intercept_px: float             # fitted position at arrival_time = 0 -- a rough check of where/when the front started
    r_squared: float
    n_points_used: int
    position_range: tuple | None = None  # (min, max) positions the FIT was restricted to, or None = every position


def _resolve_times(dataset, frame_rate_hz_override: float | None) -> np.ndarray:
    times = [f.time_s for f in dataset.frames]
    if all(t is not None for t in times):
        return np.asarray(times, dtype=float)
    if frame_rate_hz_override:
        return np.asarray([f.frame_index / frame_rate_hz_override for f in dataset.frames], dtype=float)
    raise WaveSpeedError(
        "No frame timing available: the loaded CSV has no time_s values (the Image Correlation "
        "page's frame rate wasn't set when it was exported) -- enter a frame rate override to use "
        "frame_index / rate instead."
    )


def build_arrival_time_profile(dataset, field_key: str, direction: str, threshold_fraction: float,
                                baseline_frames: int = 1, frame_rate_hz_override: float | None = None,
                                frame_range: tuple | None = None) -> ArrivalTimeProfile:
    """`frame_range`, when given, is (start, end) 0-based INCLUSIVE
    positions into the time-sorted frame list -- restricting the analysis
    to the part of the sequence that actually shows the wave passing,
    before a later crack signal (much larger strain, usually) would
    otherwise dominate each position's "local peak" and throw off the
    threshold-fraction picking."""
    if direction not in ("x", "y"):
        raise WaveSpeedError(f"direction must be 'x' or 'y', got {direction!r}")
    if not (0.0 < threshold_fraction < 1.0):
        raise WaveSpeedError("Arrival threshold must be between 0 and 1 (a fraction of the local peak).")
    if not dataset.frames:
        raise WaveSpeedError("No frames loaded.")

    times_all = _resolve_times(dataset, frame_rate_hz_override)
    order = np.argsort(times_all)
    if frame_range is not None:
        start, end = frame_range
        start = max(0, int(start))
        end = min(len(order) - 1, int(end))
        if end <= start:
            raise WaveSpeedError("The selected frame range is empty or invalid.")
        order = order[start:end + 1]
    if len(order) < 2:
        raise WaveSpeedError("Need at least 2 frames to measure a wave speed.")
    times = times_all[order]

    stack = np.stack([dataset.frames[i].field(field_key) for i in order], axis=0)  # (n_frames, n_y, n_x)
    if direction == "x":
        positions = dataset.xs
        signal = np.nanmean(stack, axis=1)  # average over y -> (n_frames, n_x)
    else:
        positions = dataset.ys
        signal = np.nanmean(stack, axis=2)  # average over x -> (n_frames, n_y)
    signal = signal.T  # (n_positions, n_frames)

    n_base = max(1, min(baseline_frames, signal.shape[1]))
    baseline = np.nanmean(signal[:, :n_base], axis=1)

    arrival_times = np.full(signal.shape[0], np.nan)
    for i in range(signal.shape[0]):
        trace = signal[i] - baseline[i]
        if np.all(np.isnan(trace)):
            continue
        peak = np.nanmax(trace)
        if not np.isfinite(peak) or peak <= 0:
            continue
        level = threshold_fraction * peak
        above = np.where(trace >= level)[0]
        if above.size == 0:
            continue
        k = int(above[0])
        if k == 0:
            arrival_times[i] = times[0]
            continue
        # Sub-frame linear interpolation of the exact threshold crossing
        # between frame k-1 and k, the same "don't round to the nearest
        # frame when a straight line through the two samples gives a
        # cheap, meaningfully more precise answer" spirit as the DIC
        # subpixel peak refinement (core/dic_processing.py's
        # `_subpixel_peak`).
        t0, t1 = times[k - 1], times[k]
        v0, v1 = trace[k - 1], trace[k]
        if np.isnan(v0) or v1 == v0:
            arrival_times[i] = times[k]
        else:
            frac = (level - v0) / (v1 - v0)
            frac = min(1.0, max(0.0, frac))
            arrival_times[i] = t0 + frac * (t1 - t0)

    return ArrivalTimeProfile(positions=positions, arrival_times=arrival_times, signal=signal, times=times)


def fit_wave_speed(profile: ArrivalTimeProfile, direction: str, field_key: str, threshold_fraction: float,
                    pixel_size_mm: float | None = None, position_range: tuple | None = None) -> WaveSpeedResult:
    """`position_range`, when given, is (min, max) POSITION (px, along
    `direction`) -- restricts which positions feed the linear fit, to
    isolate one branch of a two-opposite-direction wave (see the module
    docstring). The arrival-time detection itself (`profile`) always
    covers every position regardless, so switching `position_range`
    between calls re-fits without re-detecting."""
    valid = np.isfinite(profile.arrival_times)
    if position_range is not None:
        pmin, pmax = position_range
        if pmax <= pmin:
            raise WaveSpeedError("The position range's max must be larger than its min.")
        valid &= (profile.positions >= pmin) & (profile.positions <= pmax)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 2:
        raise WaveSpeedError(
            f"Only {n_valid} position(s) had a detectable wave arrival -- lower the arrival "
            "threshold, widen the frame or position range, or check that the chosen field "
            "actually shows a wave passing through."
        )
    x = profile.positions[valid]
    t = profile.arrival_times[valid]
    if np.allclose(t, t[0]):
        raise WaveSpeedError(
            "Every detected arrival happened in the same frame -- can't fit a speed from this "
            "(the frame rate may be too low, or the front too fast, to resolve its travel time "
            "across this region)."
        )

    slope, intercept = np.polyfit(t, x, 1)
    predicted = slope * t + intercept
    ss_res = float(np.sum((x - predicted) ** 2))
    ss_tot = float(np.sum((x - np.mean(x)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    speed_px_per_s = float(slope)
    speed_mm_per_s = speed_px_per_s * pixel_size_mm if pixel_size_mm else None
    return WaveSpeedResult(
        direction=direction, field_key=field_key, threshold_fraction=threshold_fraction,
        profile=profile, speed_px_per_s=speed_px_per_s, speed_mm_per_s=speed_mm_per_s,
        intercept_px=float(intercept), r_squared=float(r_squared), n_points_used=n_valid,
        position_range=position_range,
    )


def compute_wave_speed(dataset, field_key: str, direction: str, threshold_fraction: float = 0.5,
                        baseline_frames: int = 1, frame_rate_hz_override: float | None = None,
                        frame_range: tuple | None = None, pixel_size_mm: float | None = None,
                        position_range: tuple | None = None) -> WaveSpeedResult:
    profile = build_arrival_time_profile(
        dataset, field_key, direction, threshold_fraction, baseline_frames,
        frame_rate_hz_override, frame_range,
    )
    return fit_wave_speed(profile, direction, field_key, threshold_fraction, pixel_size_mm, position_range)


# ---------------------------------------------------------------------------
# Automatic multi-wave detection (ridge extraction + sequential RANSAC line
# segmentation) -- see the module docstring's last section for why this is a
# separate method from the single-front one above, not a replacement for it.


@dataclass
class RidgePoints:
    times: np.ndarray        # (n,) time of each detected ridge point, seconds
    positions: np.ndarray    # (n,) position (px) of each detected ridge point
    profile: ArrivalTimeProfile  # the underlying field/background this was extracted from


@dataclass
class WaveBranch:
    times: np.ndarray               # (k,) inlier ridge point times assigned to this branch
    positions: np.ndarray           # (k,) inlier ridge point positions assigned to this branch
    speed_px_per_s: float
    speed_mm_per_s: float | None
    intercept_px: float
    r_squared: float
    n_points_used: int
    color: str = "#e74c3c"          # a DEFAULT_COLORS entry, consistent with the crack-list palette


@dataclass
class MultiWaveResult:
    direction: str
    field_key: str
    threshold_fraction: float
    ridge: RidgePoints
    branches: list = field(default_factory=list)          # list[WaveBranch], most points first
    unassigned_times: np.ndarray = field(default_factory=lambda: np.array([]))
    unassigned_positions: np.ndarray = field(default_factory=lambda: np.array([]))


def _band_runs(above: np.ndarray, merge_gap: int = 3) -> list[tuple[int, int, int]]:
    """Every contiguous run of True down each column (time frame) of
    `above` (positions x frames), as `(frame_index, run_start, run_end)`
    with `run_end` exclusive -- i.e. every separate "elevated band" visible
    in that one frame.

    Two runs separated by a gap of at most `merge_gap` grid steps are
    merged into one before being reported. Per-pixel noise routinely makes
    one real, wide band dip below `threshold_fraction` for just a pixel or
    two right at its own edge (nowhere near the true band's center), which
    would otherwise split it into a tiny noise-only fragment plus the rest
    of the real band -- most damagingly right where the real band happens
    to touch the position axis's boundary: the truncated remainder still
    gets reported as its own small, NON-edge-touching run and slips past
    the edge-truncation guard in `_band_centroids`, which only inspects
    each run in isolation. Merging first means a genuinely edge-truncated
    band is always seen, and filtered, as the single run it really is."""
    runs: list[tuple[int, int, int]] = []
    n_positions, n_frames = above.shape
    for j in range(n_frames):
        col = above[:, j]
        if not col.any():
            continue
        diff = np.diff(col.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if col[0]:
            starts = np.r_[0, starts]
        if col[-1]:
            ends = np.r_[ends, n_positions]
        merged_s: list[int] = []
        merged_e: list[int] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            if merged_e and s - merged_e[-1] <= merge_gap:
                merged_e[-1] = e
            else:
                merged_s.append(s)
                merged_e.append(e)
        for s, e in zip(merged_s, merged_e):
            runs.append((j, int(s), int(e)))
    return runs


def _band_centroids(trace_filled: np.ndarray, above: np.ndarray, positions: np.ndarray,
                     min_run_px: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """One ridge point per elevated band per frame, at that band's
    amplitude-WEIGHTED CENTROID position rather than its single tallest
    (and therefore noisiest) pixel.

    Picking the single local-maximum pixel of a real, physically wide band
    is fragile: per-pixel DIC measurement noise riding on top of a wide,
    smooth hump shifts exactly which pixel wins from frame to frame by a
    few pixels in either direction, so consecutive frames' "peaks" scatter
    around the band's true center rather than tracing it -- and sequential
    RANSAC, seeing that scatter, fits several near-parallel lines through
    different parts of it instead of the one real line the band actually
    traces, fragmenting a single visible band into several spurious
    "waves". Averaging every
    pixel IN the band, weighted by how far above baseline it is, cancels
    per-pixel noise the same way any averaging does (roughly root-N in the
    band's width) while still tracking genuine sub-pixel motion of the
    band's center from frame to frame -- one clean, stable point per band
    per frame, which is what RANSAC needs to recover the one true line.

    Runs shorter than `min_run_px` grid steps are skipped as noise
    randomly ticking over the threshold for a pixel or two rather than a
    real band (a real band, extracted at a moderate `threshold_fraction`,
    is normally many pixels wide).

    A run touching either edge of the position axis (index 0 or the last
    index) is skipped entirely rather than centroided: a band that has
    only partly entered the imaged region, or has started leaving it,
    shows a visibly TRUNCATED one-sided slice -- its weighted centroid is
    measurably pulled toward whichever side is still visible, not the
    band's true center."""
    n_positions = positions.size
    idx_axis = np.arange(n_positions)
    runs = _band_runs(above)
    time_idx: list[int] = []
    pos_out: list[float] = []
    for j, s, e in runs:
        if e - s < min_run_px:
            continue
        if s == 0 or e == n_positions:
            continue
        weights = np.clip(trace_filled[s:e, j], 0.0, None)
        local_idx = idx_axis[s:e]
        total = float(np.sum(weights))
        centroid_idx = float(np.sum(local_idx * weights) / total) if total > 0 else float(0.5 * (s + e - 1))
        pos_out.append(float(np.interp(centroid_idx, idx_axis, positions)))
        time_idx.append(j)
    return np.array(time_idx, dtype=int), np.array(pos_out, dtype=float)


def detect_ridge_points(dataset, field_key: str, direction: str, threshold_fraction: float = 0.5,
                         baseline_frames: int = 1, frame_rate_hz_override: float | None = None,
                         frame_range: tuple | None = None,
                         min_frame_peak_fraction: float = 0.1,
                         min_run_px: int = 2) -> RidgePoints:
    """For every individual time frame, finds every contiguous run of
    positions where the (baseline-subtracted) field clears
    `threshold_fraction` of THAT SAME FRAME's own peak, and reports ONE
    ridge point per run at its amplitude-weighted centroid (see
    `_band_centroids`) -- see the module docstring's last section. Unlike
    `build_arrival_time_profile` (one arrival time per POSITION), this can
    and does return several ridge points at different positions for the
    same time frame (several wavefronts/reflections passing through the
    imaged region at once).

    The threshold is normalized PER FRAME (not against a single peak value
    for the whole field) for the same reason `build_arrival_time_profile`
    normalizes per POSITION rather than to one fixed absolute level: one
    exceptionally strong, spatially localized event anywhere in the field
    (a brief, narrow strain concentration unrelated to the wave itself --
    common with an equivalent/von-Mises field, which is never negative and
    so has nowhere to "cancel out") would otherwise set a threshold that
    only ITS OWN neighborhood can clear, silently hiding the real,
    generally weaker moving front in every other frame, if the threshold
    were normalized against a single peak value for the whole field
    instead.

    Per-frame normalization alone has its own failure mode though: a frame
    with NO real signal at all (the wave hasn't arrived yet, or has fully
    left the imaged region) is pure noise, but pure noise still has SOME
    maximum value -- normalizing to that noise's own "peak" manufactures a
    phantom ridge point out of nothing every such frame. `min_frame_peak_
    fraction` guards against this: a frame is only examined at all if its
    own peak reaches at least this fraction of the STRONGEST frame's peak
    across the whole selection -- comfortably below a real (if weaker)
    wave signature, comfortably above what noise alone produces.

    `min_run_px`, in grid steps, is the shortest run of above-threshold
    positions in one frame still trusted as a real band rather than noise
    randomly ticking over the threshold for a pixel or two -- see
    `_band_centroids`."""
    profile = build_arrival_time_profile(
        dataset, field_key, direction, threshold_fraction, baseline_frames,
        frame_rate_hz_override, frame_range,
    )
    n_base = max(1, min(baseline_frames, profile.signal.shape[1]))
    baseline = np.nanmean(profile.signal[:, :n_base], axis=1)
    trace2d = profile.signal - baseline[:, None]
    finite = np.isfinite(trace2d)
    if not np.any(finite):
        raise WaveSpeedError("No usable signal to detect wave ridges from.")
    trace_filled = np.where(finite, trace2d, -np.inf)

    col_max = np.max(trace_filled, axis=0)  # (n_frames,), this frame's own peak above baseline
    finite_col_max = col_max[np.isfinite(col_max)]
    global_peak = float(np.max(finite_col_max)) if finite_col_max.size else 0.0
    if not np.isfinite(global_peak) or global_peak <= 0:
        raise WaveSpeedError(
            "The selected field never rises above its own baseline in any frame -- check the "
            "field, or lower the arrival threshold."
        )
    # A robust (median/MAD) estimate of the per-pixel noise level, straight
    # from the data: a real band only ever occupies a small slice of the
    # whole (position, time) grid, so this estimate stays uncontaminated
    # by it. `min_frame_peak_fraction * global_peak` alone isn't always
    # enough of a floor -- on a genuinely noisy dataset, a frame that has
    # NO real signal yet can still have its own noise-only maximum land
    # above that fraction (a handful of positions is enough for the max of
    # pure noise to reach several sigma), and once such a frame is let
    # through, `threshold_fraction` of its own (noise-scale) "peak" sits
    # right at the noise floor -- roughly half of ALL the noise in that
    # frame clears it, producing one spurious "band" spanning most of the
    # position axis instead of no band at all. Requiring the frame's peak
    # to also clear several sigma of the actual measured noise closes that
    # gap without touching frames whose peak is real signal.
    finite_trace = trace_filled[np.isfinite(trace_filled)]
    if finite_trace.size:
        med = float(np.median(finite_trace))
        mad = float(np.median(np.abs(finite_trace - med)))
        noise_sigma = mad / 0.6745 if mad > 0 else 0.0
    else:
        noise_sigma = 0.0
    frame_floor = max(min_frame_peak_fraction * global_peak, 5.0 * noise_sigma)
    eligible = np.isfinite(col_max) & (col_max >= frame_floor)
    level = np.where(eligible, threshold_fraction * col_max, np.inf)  # (n_frames,)

    above = trace_filled >= level[None, :]
    time_idx, positions_out = _band_centroids(trace_filled, above, profile.positions, min_run_px=min_run_px)

    if time_idx.size == 0:
        raise WaveSpeedError(
            "No ridge points detected above the arrival threshold -- lower the threshold, widen "
            "the frame range, or check that the chosen field actually shows a wave passing through."
        )
    return RidgePoints(times=profile.times[time_idx], positions=positions_out, profile=profile)


def _fit_line(t: np.ndarray, x: np.ndarray):
    """OLS fit of x = slope * t + intercept, with R^2 -- returns None if
    there aren't at least 2 points spanning more than one time value."""
    if t.size < 2 or np.allclose(t, t[0]):
        return None
    slope, intercept = np.polyfit(t, x, 1)
    predicted = slope * t + intercept
    ss_res = float(np.sum((x - predicted) ** 2))
    ss_tot = float(np.sum((x - np.mean(x)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r_squared)


def segment_wave_branches(ridge: RidgePoints, max_branches: int = 6, min_points_per_branch: int = 8,
                           residual_tolerance_px: float | None = None,
                           pixel_size_mm: float | None = None, seed: int = 0):
    """Sequential RANSAC: repeatedly finds the 2-point line (position =
    speed * time + intercept) with the most ridge-point inliers within
    `residual_tolerance_px`, refits it by OLS on just those inliers, keeps
    it as one branch, removes its inliers from the pool, and repeats --
    until fewer than `min_points_per_branch` points remain unassigned or
    `max_branches` branches have been found. Returns
    (branches: list[WaveBranch] sorted most-points-first,
    unassigned_mask: bool array aligned to `ridge.times`/`ridge.positions`
    marking points no branch claimed).

    `residual_tolerance_px`, when None, defaults to 2% of the position
    axis's total span -- wide enough to absorb a pulse's finite width and
    the position grid's own resolution, narrow enough to still separate
    branches that cross at a shallow angle."""
    times = np.asarray(ridge.times, dtype=float)
    positions = np.asarray(ridge.positions, dtype=float)
    n = times.size
    if residual_tolerance_px is None:
        span = float(np.ptp(ridge.profile.positions)) if ridge.profile.positions.size else 1.0
        residual_tolerance_px = max(0.02 * span, 1e-6)

    remaining = np.ones(n, dtype=bool)
    rng = np.random.default_rng(seed)
    branches = []
    while int(np.count_nonzero(remaining)) >= min_points_per_branch and len(branches) < max_branches:
        idx = np.nonzero(remaining)[0]
        t_pool, x_pool = times[idx], positions[idx]
        n_pool = idx.size

        if n_pool <= 60:
            pairs = [(i, j) for i in range(n_pool) for j in range(i + 1, n_pool)]
        else:
            n_iter = 800
            seen = set()
            attempts = 0
            while len(seen) < n_iter and attempts < n_iter * 4:
                i, j = (int(v) for v in rng.integers(0, n_pool, size=2))
                attempts += 1
                if i != j:
                    seen.add((min(i, j), max(i, j)))
            pairs = list(seen)

        best_count = -1
        best_mask = None
        for i, j in pairs:
            if t_pool[i] == t_pool[j]:
                continue
            slope = (x_pool[j] - x_pool[i]) / (t_pool[j] - t_pool[i])
            intercept = x_pool[i] - slope * t_pool[i]
            predicted = slope * t_pool + intercept
            inlier_mask = np.abs(x_pool - predicted) <= residual_tolerance_px
            count = int(np.count_nonzero(inlier_mask))
            if count > best_count:
                best_count = count
                best_mask = inlier_mask

        if best_mask is None or best_count < min_points_per_branch:
            break

        fit = _fit_line(t_pool[best_mask], x_pool[best_mask])
        if fit is None:
            break
        slope, intercept, r_squared = fit
        speed_mm = slope * pixel_size_mm if pixel_size_mm else None
        branches.append(WaveBranch(
            times=t_pool[best_mask], positions=x_pool[best_mask],
            speed_px_per_s=slope, speed_mm_per_s=speed_mm, intercept_px=intercept,
            r_squared=r_squared, n_points_used=int(np.count_nonzero(best_mask)),
        ))
        remaining[idx[best_mask]] = False

    branches.sort(key=lambda b: b.n_points_used, reverse=True)
    for i, branch in enumerate(branches):
        branch.color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
    return branches, remaining


def detect_wave_branches(dataset, field_key: str, direction: str, threshold_fraction: float = 0.5,
                          baseline_frames: int = 1, frame_rate_hz_override: float | None = None,
                          frame_range: tuple | None = None, pixel_size_mm: float | None = None,
                          max_branches: int = 6, min_points_per_branch: int = 8,
                          residual_tolerance_px: float | None = None, seed: int = 0,
                          min_frame_peak_fraction: float = 0.1,
                          min_run_px: int = 2) -> MultiWaveResult:
    """Top-level entry point for automatic multi-wave detection: extracts
    ridge points (`detect_ridge_points`) then segments them into branches
    (`segment_wave_branches`), one call for the whole field -- no manual
    position-range splitting needed, and no prior knowledge of how many
    waves/reflections are present. See the module docstring."""
    ridge = detect_ridge_points(
        dataset, field_key, direction, threshold_fraction, baseline_frames,
        frame_rate_hz_override, frame_range, min_frame_peak_fraction=min_frame_peak_fraction,
        min_run_px=min_run_px,
    )
    branches, unassigned_mask = segment_wave_branches(
        ridge, max_branches=max_branches, min_points_per_branch=min_points_per_branch,
        residual_tolerance_px=residual_tolerance_px, pixel_size_mm=pixel_size_mm, seed=seed,
    )
    return MultiWaveResult(
        direction=direction, field_key=field_key, threshold_fraction=threshold_fraction,
        ridge=ridge, branches=branches,
        unassigned_times=ridge.times[unassigned_mask], unassigned_positions=ridge.positions[unassigned_mask],
    )


@dataclass
class PolygonRidgeResult:
    """One user-picked wave from `extract_ridge_in_polygon` -- see the
    module docstring's last section. `speed_px_per_s`/`speed_mm_per_s`/
    `r_squared` are a plain OLS fit through `times`/`positions`, reported
    purely as an indicative convenience (None if fewer than 2 points):
    the raw points, not this fit, are the actual result, and are free to
    trace a curved front a straight-line fit would misrepresent."""
    label: str
    color: str
    direction: str
    field_key: str
    vertices: list          # [(time_s, position_px), ...] as drawn, kept for reference/redraw
    times: np.ndarray
    positions: np.ndarray
    values: np.ndarray      # baseline-subtracted field value at each ridge point
    speed_px_per_s: float | None = None
    speed_mm_per_s: float | None = None
    r_squared: float | None = None


def extract_ridge_in_polygon(dataset, field_key: str, direction: str, vertices,
                              baseline_frames: int = 1, frame_rate_hz_override: float | None = None,
                              frame_range: tuple | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, ArrivalTimeProfile]:
    """Manual ROI-polygon ridge picking -- see the module docstring's last
    section. `vertices` is a list of (time_s, position_px) points in the
    SAME (time, position) plane the wave-speed plot itself is drawn in
    (matching how the polygon is drawn directly on that plot); at least 3
    are needed to enclose any area.

    For every time frame the polygon's time-extent spans, tests every
    position sample at that time against the polygon (matplotlib's own
    point-in-polygon test) and, among the ones inside, reports ONE ridge
    point built in two steps:

    1. Find every local EXTREMUM among the enclosed samples -- a position
       whose (baseline-subtracted) value is a local max OR a local min
       relative to its neighbors, either sign (a rising front is a local
       max, a compressive/negative one a local min, both handled the same
       way, and a strictly non-negative field such as the equivalent
       (von Mises) strain has only local maxima -- a "local min" there is
       just a noise dip toward zero, not a real feature). This search
       runs on a lightly SMOOTHED copy of the enclosed values
       (`_smooth_1d`, a small moving average of |value|), not the raw
       per-pixel values directly -- real DIC data is noisy enough that
       comparing raw immediate neighbors makes almost every sample a
       "local extremum" of one sign or the other, which would leave step
       2 choosing among noise rather than real features. The smoothing
       only decides WHERE the real extrema roughly are; the position/
       value ultimately reported always comes from the RAW signal (step
       2).
    2. Among those candidates, keep the STRONGEST one (largest |value|),
       then snap it to the actual RAW-signal peak within a small window
       (smoothing can shift a peak's apparent index by a sample or two),
       and refine it by a tight amplitude-weighted centroid over just
       that raw peak's immediate same-sign neighbors (a few grid steps
       either side, not the whole enclosed span -- a broad average over
       EVERY enclosed sample gets pulled toward whatever background/
       gradient also happens to sit inside the polygon, which can end up
       tracing something close to the polygon's own outline rather than
       the actual ridge inside it).

    "Strongest wins" is a deliberate choice: an unrelated stronger band
    that only grazes the polygon's edge for a frame or two CAN hijack the
    point during those frames if the polygon is drawn loosely enough to
    let it in -- exactly why the UI instructs drawing the loop tightly
    around ONE band and undershooting rather than overshooting. Weighing
    candidates by anything OTHER than raw signal strength (how close a
    candidate sits to the polygon's own instantaneous boundary, or to the
    previously accepted point) was tried and discarded: both make the
    traced line drift toward whatever the disambiguation rule favors --
    the polygon's own hand-drawn shape, or wherever the trace happened to
    get stuck earlier -- rather than the actual data, which is a worse
    failure than an occasional graze.

    A frame the polygon doesn't reach, or where every enclosed sample is
    exactly at baseline, contributes no point rather than a fabricated one
    -- gaps in the returned trace are expected and fine (RANSAC-free:
    nothing downstream needs an unbroken line).

    Returns `(times, positions, values, profile)` -- `profile` is the
    same field/background (`ArrivalTimeProfile`) the polygon was drawn
    against, handed back so a caller can redraw it without recomputing."""
    profile = build_arrival_time_profile(
        dataset, field_key, direction, 0.5, baseline_frames,
        frame_rate_hz_override, frame_range,
    )
    n_base = max(1, min(baseline_frames, profile.signal.shape[1]))
    baseline = np.nanmean(profile.signal[:, :n_base], axis=1)
    trace2d = profile.signal - baseline[:, None]

    verts = [(float(t), float(p)) for t, p in vertices]
    if len(verts) < 3:
        raise WaveSpeedError("A polygon needs at least 3 vertices to enclose an area.")
    path = _MplPath(np.asarray(verts, dtype=float))

    n_positions, n_frames = trace2d.shape
    times_out: list[float] = []
    positions_out: list[float] = []
    values_out: list[float] = []
    for j in range(n_frames):
        t = float(profile.times[j])
        pts = np.column_stack([np.full(n_positions, t), profile.positions])
        inside = np.nonzero(path.contains_points(pts))[0]
        if inside.size == 0:
            continue
        col = trace2d[inside, j]
        finite = np.isfinite(col)
        if not np.any(finite):
            continue
        inside = inside[finite]
        col = col[finite]
        abs_col = np.abs(col)
        n_in = inside.size

        # Candidates are this column's own local extrema (a local max OR a
        # local min, either sign -- both a rising and a compressive front
        # are real features, though a strictly non-negative field only
        # ever has local maxima of |value| == the raw value) among the
        # ENCLOSED samples specifically. This search runs on `abs_smooth`,
        # a lightly smoothed copy of `abs_col` (see `_smooth_1d`), NOT the
        # raw per-pixel values directly -- real DIC data is noisy enough
        # that comparing raw immediate neighbors makes almost every
        # sample qualify as a "local extremum" of one sign or the other,
        # turning the strongest-wins pick below into effectively a random
        # choice among noise. Edge samples are excluded as candidates (no
        # interior neighbor to compare against), unless there are so few
        # enclosed samples that "interior" is meaningless. A run of tied/
        # plateau smoothed values collapses to its own raw-signal argmax,
        # so a flat top doesn't get double-counted.
        if n_in <= 2:
            candidates = list(range(n_in))
        else:
            abs_smooth = _smooth_1d(abs_col, min(_POLYGON_PEAK_SMOOTH_RADIUS, (n_in - 1) // 2))
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

        # Among competing local extrema, the STRONGEST one wins -- see the
        # function's docstring for why every other tie-break tried here
        # made the trace drift toward something other than the data.
        peak_local = max(candidates, key=lambda i: abs_col[i])

        # The chosen candidate's index came from the SMOOTHED search above
        # and can be off by a sample or two from the true raw peak --
        # snap to the actual raw argmax within a small window of it before
        # measuring anything.
        lo = max(0, peak_local - _POLYGON_LOCAL_WINDOW)
        hi = min(n_in, peak_local + _POLYGON_LOCAL_WINDOW + 1)
        peak_local = lo + int(np.argmax(abs_col[lo:hi]))
        peak_mag = abs_col[peak_local]
        if peak_mag <= 0:
            continue
        peak_grid_idx = inside[peak_local]
        peak_sign = np.sign(col[peak_local])
        near = np.abs(inside - peak_grid_idx) <= _POLYGON_LOCAL_WINDOW
        same_sign = np.sign(col) == peak_sign
        sel = near & same_sign
        if not np.any(sel):
            sel = np.zeros_like(near)
            sel[peak_local] = True

        w = abs_col[sel]
        total = float(np.sum(w))
        centroid_pos = float(np.sum(profile.positions[inside[sel]] * w) / total)
        centroid_val = float(np.sum(col[sel] * w) / total)
        times_out.append(t)
        positions_out.append(centroid_pos)
        values_out.append(centroid_val)

    return (np.array(times_out, dtype=float), np.array(positions_out, dtype=float),
            np.array(values_out, dtype=float), profile)


def fit_ridge_points(times: np.ndarray, positions: np.ndarray, pixel_size_mm: float | None = None):
    """Plain OLS fit through a set of (time, position) ridge points, for
    an indicative speed/R^2 -- see `PolygonRidgeResult`. Returns
    `(speed_px_per_s, speed_mm_per_s, r_squared)`, or `(None, None, None)`
    if there are fewer than 2 distinct-time points to fit."""
    fit = _fit_line(np.asarray(times, dtype=float), np.asarray(positions, dtype=float))
    if fit is None:
        return None, None, None
    slope, _intercept, r_squared = fit
    speed_mm = slope * pixel_size_mm if pixel_size_mm else None
    return float(slope), speed_mm, float(r_squared)


def smoothed_wave_positions(times: np.ndarray, positions: np.ndarray, window_px: float) -> np.ndarray:
    """Display-only temporal smoothing of a ROI-picked wave's per-frame
    position, mirroring core/crack_tip_tracking.py's `smoothed_tip_arc_
    length_and_positions` one to one -- same local-linear-regression
    smoother, same rationale (a per-frame independent pick is only as
    good as the underlying method's frame-to-frame noise; a moving
    average would lag behind a genuine speed change, so this fits a local
    straight line -- position vs time -- through every point within
    `window_px` of arc length, converted to a time half-width via the
    wave's own robust average speed, and evaluates it at each frame's own
    time). Simpler than the crack version because a wave-speed position
    already IS the physical coordinate along the chosen axis (x or y) --
    unlike a crack tip's arc length along an arbitrary traced path, there
    is no path to re-project a smoothed value back onto.

    "Robust average speed" is the MEDIAN of point-to-point secant speeds
    between consecutive valid points in time order, not a single end-to-
    end least-squares slope over the whole wave -- see `smoothed_tip_arc_
    length_and_positions`'s docstring for why: an OLS slope has high
    leverage on a minority of noisy/outlying points and can silently
    collapse the converted time window (near-zero smoothing despite a
    positive `window_px`), while a median tracks a clean, near-constant-
    speed wave's real pace just as well.

    `window_px <= 0` disables smoothing entirely (positions pass through
    unchanged). Frames with fewer than 2 valid points in their window (not
    enough to fit a line) fall back to their own raw, unsmoothed position.
    Non-finite input positions stay non-finite in the output.

    Returns the smoothed positions array, same length/order as `times`/
    `positions`."""
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    smoothed = np.full(n, np.nan)
    valid_idx = np.flatnonzero(np.isfinite(positions))
    if valid_idx.size == 0:
        return smoothed

    t_valid = times[valid_idx]
    p_valid = positions[valid_idx]

    if window_px <= 0.0 or valid_idx.size < 2:
        smoothed[valid_idx] = p_valid
        return smoothed

    # Robust average speed (px/s): the MEDIAN point-to-point secant speed,
    # not an end-to-end OLS slope -- see the docstring above. `times`/
    # `positions` are always time-sorted ascending here (every caller
    # guarantees it), so consecutive differences are genuine point-to-
    # point steps.
    dt = np.diff(t_valid)
    dp = np.diff(p_valid)
    moved = dt > 0
    speed = float(np.median(dp[moved] / dt[moved])) if np.any(moved) else 0.0

    if abs(speed) < 1e-9:
        smoothed[valid_idx] = p_valid
        return smoothed

    half_t = (window_px / 2.0) / abs(speed)
    for k, i in enumerate(valid_idx):
        ti = t_valid[k]
        in_window = np.abs(t_valid - ti) <= half_t
        if np.count_nonzero(in_window) < 2:
            smoothed[i] = p_valid[k]
            continue
        t_win = t_valid[in_window]
        p_win = p_valid[in_window]
        if np.allclose(t_win, t_win[0]):
            smoothed[i] = p_valid[k]
            continue
        slope, intercept = np.polyfit(t_win, p_win, 1)
        smoothed[i] = slope * ti + intercept

    return smoothed


def export_roi_waves_csv(waves: list, path: str, pixel_size_mm: float | None = None) -> None:
    """Writes every stored ROI-picked wave's raw (time, position) points
    to one CSV, long format (one row per point, a `wave_label` column
    tells them apart) -- matching how a PDV/VISAR ridge trace is normally
    handed off for further analysis rather than reduced to a single
    fitted number here. Raises `OSError` on a write failure (bad path,
    permissions, ...), left for the caller to report as-is."""
    import csv

    header = ["wave_label", "direction", "field", "time_s", "position_px"]
    if pixel_size_mm:
        header.append("position_mm")
    header.append("field_value")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for wave in waves:
            for k in range(wave.times.size):
                row = [wave.label, wave.direction, wave.field_key,
                       f"{wave.times[k]:.9g}", f"{wave.positions[k]:.6f}"]
                if pixel_size_mm:
                    row.append(f"{wave.positions[k] * pixel_size_mm:.6f}")
                row.append(f"{wave.values[k]:.6g}")
                writer.writerow(row)
