"""
Direct K_I / K_II from the MEASURED (subset-DIC) displacement field -- no
X-FEM-DIC re-correlation -- plus the tip-position smoothing that feeds it.

Two independent pieces, both used by the "Crack analysis" tab:

1. `smooth_tip_arc_length` / `smooth_tip_trace` -- smooth the tracked tip
   position (arc length along the path vs time, the curve of the "Arc-length
   position vs time" graph) with a chosen method, optionally forcing the tip
   never to move backwards. The smoothed positions are re-projected onto the
   traced path, so they stay exactly ON the crack. They replace the raw ones
   in a copy of the trace -- the raw trace is never modified.

2. `compute_direct_k_series` -- K_I(t)/K_II(t) at every tip position of a
   tip track, straight from the exported subset-DIC (u, v) displacement
   field, by either
     * "ring"        : the closed-form Williams/Irwin ring fit
                       (core.crack_k1c.fit_k_single_frame, optionally with the
                       T-stress term), or
     * "interaction" : the domain-form interaction integral
                       (core.crack_interaction_integral).
   The result is a `DirectKSeries` that has the same shape as
   core.xfem_dic.XfemDicTimeSeries (samples with `.fit.K_I_pa_sqrt_m`,
   `.tip_arc_length_s`, `.time_s`, ...) so the page's Results plots, status
   text and CSV export treat it exactly like an X-FEM-DIC trace.

Why smoothing matters here: the near-tip basis blows up as 1/sqrt(r) towards
the origin, so a jittery tip position (a few pixels of frame-to-frame noise
on a fast crack) moves the fit's polar origin and swings K_I/K_II far more
than it moves the tip track itself. Smoothing the positions first is the
cheapest way of stabilising K(t).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from core.crack_k1c import (
    K1cError, fit_k_single_frame, kolosov_kappa, shear_modulus,
)
from core.crack_tip_tracking import path_tangents


class DirectKError(ValueError):
    """A problem the caller can show to the user as-is."""


# ---------------------------------------------------------------------------
# 1. Tip-position smoothing
# ---------------------------------------------------------------------------

SMOOTHING_METHODS = [
    ("none", "None (raw positions)"),
    ("median", "Moving median (robust to outliers)"),
    ("linear", "Local linear regression (follows speed ramps)"),
    ("quadratic", "Local quadratic regression (Savitzky-Golay-like)"),
    ("isotonic", "Monotone (isotonic) fit, then local regression"),
]
SMOOTHING_KEYS = [k for k, _ in SMOOTHING_METHODS]


def _odd_window(window: int, n: int) -> int:
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    cap = n if n % 2 == 1 else n - 1          # never wider than the data itself (and odd)
    return max(1, min(w, cap))


def _pav_non_decreasing(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the L2 projection of `y` onto non-decreasing sequences."""
    n = len(y)
    sv, sw, ss = [], [], []                  # stack of pooled blocks: value, weight, size
    for k in range(n):
        sv.append(float(y[k])); sw.append(1.0); ss.append(1)
        while len(sv) > 1 and sv[-2] > sv[-1]:
            w = sw[-2] + sw[-1]
            v = (sv[-2] * sw[-2] + sv[-1] * sw[-1]) / w
            s = ss[-2] + ss[-1]
            sv[-2:] = [v]; sw[-2:] = [w]; ss[-2:] = [s]
    out = np.empty(n)
    pos = 0
    for v, s in zip(sv, ss):
        out[pos:pos + s] = v
        pos += s
    return out


def _local_poly(t: np.ndarray, s: np.ndarray, window: int, degree: int) -> np.ndarray:
    """Moving local polynomial regression in time over `window` samples
    (window clipped at the ends -- never padded, never extrapolated beyond
    what the data has). A point whose window holds too few points for the
    requested degree falls back to the next lower degree, then to its own raw value."""
    n = len(s)
    half = window // 2
    out = s.copy()
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        tt, ss = t[lo:hi], s[lo:hi]
        deg = min(degree, len(tt) - 1)
        while deg >= 1 and np.ptp(tt) <= 0:
            deg -= 1
        if deg < 1:
            continue
        tc = tt - t[i]                       # centred: value at the centre = constant term
        scale = max(np.ptp(tc), 1e-300)
        try:
            coef = np.polyfit(tc / scale, ss, deg)
        except (np.linalg.LinAlgError, ValueError):
            continue
        out[i] = coef[-1]
    return out


def smooth_tip_arc_length(times, arc_length, method: str = "linear", window: int = 7,
                          monotonic: bool = False) -> np.ndarray:
    """Smoothed copy of a per-frame tip arc-length sequence (same length as
    `arc_length`; entries that are not finite stay NaN).

    `window` is the number of FRAMES (samples) the smoother looks at around
    each point (odd; an even value is bumped by one). `method`:
      none      raw positions (only `monotonic` is applied, if requested)
      median    moving median -- rejects an isolated bad detection outright
      linear    local straight-line regression vs time -- damps jitter and
                still follows a speed ramp (no lag, unlike a moving mean)
      quadratic local parabola vs time -- also follows an acceleration
      isotonic  monotone (non-decreasing) least-squares fit first, then a
                local linear regression to round off its staircase
    `monotonic=True` finishes with the L2-closest monotone sequence in the
    direction of the overall trend, so the tip can never move backwards
    (a growing crack does not heal)."""
    t = np.asarray(times, dtype=float)
    s = np.asarray(arc_length, dtype=float)
    out = np.full(len(s), np.nan)
    valid = np.isfinite(s) & np.isfinite(t)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return out
    order = idx[np.argsort(t[idx], kind="stable")]
    tv, sv = t[order], s[order]
    n = len(sv)
    method = (method or "none").lower()
    if method not in SMOOTHING_KEYS:
        raise DirectKError(f"Unknown smoothing method '{method}'.")

    # direction of the overall advance (toward larger or smaller arc length)
    sign = 1.0
    if n >= 2 and np.ptp(tv) > 0:
        sign = -1.0 if float(np.polyfit(tv, sv, 1)[0]) < 0 else 1.0
    work = sv * sign                          # so "advance" is always increasing below

    w = _odd_window(window, n)
    if method == "none" or n < 3 or w < 3:
        sm = work.copy()
    elif method == "median":
        half = w // 2
        sm = np.array([np.median(work[max(0, i - half):min(n, i + half + 1)]) for i in range(n)])
    elif method == "linear":
        sm = _local_poly(tv, work, w, 1)
    elif method == "quadratic":
        sm = _local_poly(tv, work, w, 2)
    else:                                     # isotonic
        sm = _local_poly(tv, _pav_non_decreasing(work), w, 1)

    if monotonic or method == "isotonic":
        sm = _pav_non_decreasing(sm)
    out[order] = sm * sign
    return out


def smooth_tip_trace(trace, path_positions, path_arc_length, method: str, window: int,
                     monotonic: bool = False):
    """A COPY of a `PolygonTipRidgeResult` whose `arc_length` is smoothed and
    whose `positions_xy` is re-projected onto the traced path (so the smoothed
    tip stays exactly on the crack). `method="none"` without `monotonic`
    returns the trace itself, untouched."""
    if (method or "none") == "none" and not monotonic:
        return trace
    s_new = smooth_tip_arc_length(trace.times, trace.arc_length, method, window, monotonic)
    pp = np.asarray(path_positions, dtype=float)
    pa = np.asarray(path_arc_length, dtype=float)
    pos = np.full((len(s_new), 2), np.nan)
    ok = np.isfinite(s_new)
    if np.any(ok) and len(pa) >= 2:
        pos[ok, 0] = np.interp(s_new[ok], pa, pp[:, 0])
        pos[ok, 1] = np.interp(s_new[ok], pa, pp[:, 1])
    else:
        pos = np.asarray(trace.positions_xy, dtype=float).copy()
    return replace(trace, arc_length=s_new, positions_xy=pos)


def smoothing_summary(method: str, window: int, monotonic: bool) -> str:
    label = dict(SMOOTHING_METHODS).get(method, method).split(" (")[0]
    if method == "none" and not monotonic:
        return "raw tip positions"
    parts = [label if method != "none" else "no smoothing"]
    if method != "none":
        parts.append(f"{int(window)} frames")
    if monotonic or method == "isotonic":
        parts.append("non-decreasing")
    return "smoothed tip positions (" + ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# 2. Direct K_I/K_II time series (same shape as XfemDicTimeSeries)
# ---------------------------------------------------------------------------

DIRECT_METHODS = [
    ("ring", "Williams ring fit (displacement)"),
    ("ring_t", "Williams ring fit + T-stress"),
    ("ring_affine", "Williams ring fit + uniform strain & rotation"),
    ("interaction", "Interaction integral (J-domain)"),
]


@dataclass
class DirectKFit:
    K_I_pa_sqrt_m: float | None = None
    K_II_pa_sqrt_m: float | None = None
    K_I_stderr_pa_sqrt_m: float | None = None
    K_II_stderr_pa_sqrt_m: float | None = None
    r_squared: float | None = None          # ring fits only
    n_samples: int = 0
    t_stress_pa: float | None = None        # "ring_t" only
    # interaction integral only: the same integral over the inner / outer half
    K_I_inner_half_pa_sqrt_m: float | None = None
    K_I_outer_half_pa_sqrt_m: float | None = None
    error: str | None = None
    # attributes the X-FEM status/CSV code reads -- fixed, meaningless here
    converged: bool = True
    stalled: bool = False
    residual_rms: float | None = None


@dataclass
class DirectKSample:
    frame_order_index: int
    source_frame_index: int
    time_s: float
    tip_xy_px: np.ndarray
    fit: DirectKFit
    tip_arc_length_s: float | None = None
    tip_source: str = "measured (direct)"
    tangent_xy: np.ndarray | None = None


@dataclass
class DirectKSeries:
    leaf_index: int
    tip_method: str
    pixel_size_mm: float
    method: str                              # "ring" | "ring_t" | "interaction"
    r_min_px: float
    r_max_px: float
    flank_exclusion_deg: float
    smoothing: str = "raw tip positions"
    tip_offset_px: float = 0.0
    n_frames_after_arrest: int = 0           # tip frames dropped because the crack had stopped
    samples: list = field(default_factory=list)
    default_critical_index: int | None = None

    def sample_at(self, index: int):
        return self.samples[index] if 0 <= index < len(self.samples) else None


def method_label(method: str) -> str:
    return dict(DIRECT_METHODS).get(method, method)


def growth_end_index(signed_arc, min_tol_px: float = 2.0, rel_tol: float = 0.01) -> int:
    """Index of the first frame at which the tip has reached its final position
    (within max(min_tol_px, rel_tol * total travel)) -- the frames after it
    belong to an arrested crack. `signed_arc` is the arc length oriented so
    that it increases while the crack grows; NaN frames are ignored. Returns
    the last index when the tip never settles."""
    a = np.asarray(signed_arc, dtype=float)
    ok = np.isfinite(a)
    if np.count_nonzero(ok) < 3:
        return len(a) - 1
    v = a[ok]
    med3 = np.array([np.median(v[max(0, k - 1):k + 2]) for k in range(len(v))])   # a lone outlier cannot set the end
    end = float(med3.max())
    travel = end - float(med3[0])
    if travel <= 0:
        return len(a) - 1
    tol = max(min_tol_px, rel_tol * travel)
    first_ok = np.flatnonzero(ok)
    reached = first_ok[med3 >= end - tol]
    return int(reached[0]) if len(reached) else len(a) - 1


def compute_direct_k_series(dataset, tip_result, young_modulus_pa: float, poisson_ratio: float,
                            plane_stress: bool, pixel_size_mm: float, r_min_px: float, r_max_px: float,
                            method: str = "ring", flank_exclusion_deg: float = 15.0,
                            n_radial_samples: int | None = None, n_angular_samples: int | None = None,
                            tangent_window_px: float | None = None, min_r_squared_for_default: float = 0.5,
                            smoothing: str = "raw tip positions", progress=None,
                            tip_offset_px: float = 0.0, max_frames: int | None = None,
                            trim_after_arrest: bool = True) -> DirectKSeries:
    """K_I(t)/K_II(t) at every frame of `tip_result` that has a tip position,
    from that frame's measured displacement field (`dataset.frames[...].u/.v`).
    `tip_result` is a core.crack_tip_tracking.TipTrackResult -- pass one built
    from a SMOOTHED trace (`smooth_tip_trace`) to get K from smoothed tip
    positions. The local crack direction comes from the traced path (smoothed
    over `tangent_window_px`, default 2*r_max_px) and is oriented along the
    tip's actual advance, exactly like core.crack_k1c.compute_k_time_series.

    `tip_offset_px` moves the origin of the near-tip fit along the local crack
    direction (+ = ahead of the detected tip, - = behind). The opening-jump tip
    sits behind the true tip (the opening vanishes below the DIC resolution
    before the tip), and a fit origin behind the true tip OVERESTIMATES K, the
    more so the smaller r_max is. `trim_after_arrest` (default) drops the
    frames after the crack has stopped (see `growth_end_index`): the tip no
    longer moves, the field relaxes and K there is not a propagation value.
    `max_frames` evaluates only that many evenly
    spaced tip frames (used by the sensitivity scan)."""
    if method not in dict(DIRECT_METHODS):
        raise DirectKError(f"Unknown K method '{method}'.")
    if not pixel_size_mm or pixel_size_mm <= 0:
        raise DirectKError("A pixel size (mm/px) is required to compute K in physical units.")
    if r_max_px <= r_min_px:
        raise DirectKError("r_max must be larger than r_min.")
    if method == "interaction" and r_min_px <= 0:
        raise DirectKError("The interaction integral needs a strictly positive r_min "
                           "(its auxiliary field is singular at r = 0).")
    arc = np.asarray(tip_result.arc_length_s, dtype=float)
    if not np.isfinite(arc).any():
        raise DirectKError("This tip track has no detected tip positions.")

    kappa = kolosov_kappa(poisson_ratio, plane_stress)
    mu = shear_modulus(young_modulus_pa, poisson_ratio)

    tangent_sign = 1.0
    valid = np.isfinite(arc)
    if np.count_nonzero(valid) >= 2:
        tv, sv = tip_result.times[valid], arc[valid]
        # a tip that never moved has a numerically-zero slope of random sign: keep +1 then
        if (not np.allclose(tv, tv[0]) and np.ptp(sv) > 1e-6
                and float(np.polyfit(tv, sv, 1)[0]) < 0):
            tangent_sign = -1.0
    window_px = tangent_window_px if tangent_window_px is not None else 2.0 * r_max_px
    tangents = path_tangents(tip_result.path_positions, tip_result.path_arc_length,
                             window_px=window_px) * tangent_sign

    if method == "interaction":
        from core.crack_interaction_integral import (
            fit_k_interaction_integral_single_frame, InteractionIntegralError,
        )
    n_r = n_radial_samples or (12 if method == "interaction" else 8)
    n_t = n_angular_samples or (24 if method == "interaction" else 16)

    samples = []
    n_total = len(tip_result.times)
    last_frame = growth_end_index(arc * tangent_sign) if trim_after_arrest else n_total - 1
    wanted = None
    if max_frames is not None and max_frames > 0:
        finite_idx = np.flatnonzero(np.isfinite(arc))
        if len(finite_idx) > max_frames:
            wanted = set(int(k) for k in finite_idx[np.unique(
                np.linspace(0, len(finite_idx) - 1, int(max_frames)).round().astype(int))])
    for i in range(n_total):
        if progress is not None:
            progress(i, n_total)
        if not np.isfinite(arc[i]):
            continue
        if i > last_frame:
            break
        if wanted is not None and i not in wanted:
            continue
        tip_xy = np.asarray(tip_result.positions_xy[i], dtype=float)
        path_idx = int(np.argmin(np.abs(tip_result.path_arc_length - arc[i])))
        tangent = tangents[path_idx]
        if tip_offset_px:
            tip_xy = tip_xy + float(tip_offset_px) * np.asarray(tangent, dtype=float)
        source_idx = int(tip_result.source_frame_positions[i])
        frame = dataset.frames[source_idx]
        fit = DirectKFit()
        try:
            if method == "interaction":
                res = fit_k_interaction_integral_single_frame(
                    frame.u, frame.v, dataset.xs, dataset.ys, tip_xy, tangent, kappa, mu,
                    young_modulus_pa, poisson_ratio, plane_stress, pixel_size_mm, r_min_px, r_max_px,
                    flank_exclusion_deg, n_r, n_t)
                fit = DirectKFit(
                    K_I_pa_sqrt_m=res.K_I_pa_sqrt_m, K_II_pa_sqrt_m=res.K_II_pa_sqrt_m,
                    n_samples=int(res.n_points_used),
                    K_I_inner_half_pa_sqrt_m=res.K_I_inner_half_pa_sqrt_m,
                    K_I_outer_half_pa_sqrt_m=res.K_I_outer_half_pa_sqrt_m)
                if res.K_I_pa_sqrt_m is None:
                    fit.error = "too few valid displacement samples around the tip"
            else:
                res = fit_k_single_frame(
                    frame.u, frame.v, dataset.xs, dataset.ys, tip_xy, tangent, kappa, mu, pixel_size_mm,
                    r_min_px, r_max_px, flank_exclusion_deg, n_r, n_t, include_t_stress=(method == "ring_t"),
                    include_affine=(method == "ring_affine"))
                fit = DirectKFit(
                    K_I_pa_sqrt_m=res.K_I_pa_sqrt_m, K_II_pa_sqrt_m=res.K_II_pa_sqrt_m,
                    K_I_stderr_pa_sqrt_m=res.K_I_stderr_pa_sqrt_m,
                    K_II_stderr_pa_sqrt_m=res.K_II_stderr_pa_sqrt_m,
                    r_squared=res.r_squared, n_samples=int(res.n_points_used),
                    t_stress_pa=res.t_stress_pa)
                if res.K_I_pa_sqrt_m is None:
                    fit.error = "too few valid displacement samples around the tip"
        except (K1cError, ValueError) as exc:          # InteractionIntegralError is a ValueError too
            fit = DirectKFit(error=str(exc))
        samples.append(DirectKSample(
            frame_order_index=i, source_frame_index=source_idx, time_s=float(tip_result.times[i]),
            tip_xy_px=tip_xy, fit=fit, tip_arc_length_s=float(arc[i]), tangent_xy=np.asarray(tangent)))

    default_index = None
    for k, smp in enumerate(samples):
        f = smp.fit
        if f.K_I_pa_sqrt_m is not None and (f.r_squared is None or f.r_squared >= min_r_squared_for_default):
            default_index = k
            break
    if default_index is None:
        for k, smp in enumerate(samples):
            if smp.fit.K_I_pa_sqrt_m is not None:
                default_index = k
                break

    return DirectKSeries(
        leaf_index=tip_result.leaf_index, tip_method=tip_result.method, pixel_size_mm=pixel_size_mm,
        method=method, r_min_px=r_min_px, r_max_px=r_max_px, flank_exclusion_deg=flank_exclusion_deg,
        smoothing=smoothing, tip_offset_px=float(tip_offset_px or 0.0),
        n_frames_after_arrest=max(0, int(np.count_nonzero(valid)) - len(samples)) if (trim_after_arrest and wanted is None) else 0,
        samples=samples, default_critical_index=default_index)


# ---------------------------------------------------------------------------
# 3. Sensitivity scan: K_I versus assumed tip offset and r_max
# ---------------------------------------------------------------------------

@dataclass
class KSensitivityScan:
    offsets_px: list
    r_max_px: list
    median_k1: np.ndarray            # [n_offset, n_rmax], Pa sqrt(m); nan where nothing fitted
    n_frames: np.ndarray             # frames that entered each median
    spread: np.ndarray               # per offset: (max - min) / mean of the medians over r_max
    best_offset_px: float | None     # offset with the smallest spread (K most independent of r_max)
    k_at_best_pa_sqrt_m: float | None
    method: str
    r_min_px: float


def scan_k_sensitivity(dataset, tip_result, young_modulus_pa, poisson_ratio, plane_stress, pixel_size_mm,
                       r_min_px: float, r_max_list_px, offsets_px, method: str = "ring",
                       flank_exclusion_deg: float = 15.0, max_frames: int = 12,
                       min_r_squared: float = 0.5, progress=None) -> KSensitivityScan:
    """Median K_I over (up to `max_frames`) tip frames for every combination of
    assumed tip offset along the crack direction and fitting radius r_max.

    A correctly placed fit origin gives a K that does not depend on r_max;
    a detected tip BEHIND the true tip gives K decreasing with r_max. The
    offset where the medians agree best across r_max (smallest relative
    spread) is the estimate of the tip bias, and the K there the corrected
    value. Frames of arrested tips (no advance) are as good as any here."""
    offsets = [float(o) for o in offsets_px]
    rmaxs = [float(r) for r in r_max_list_px if r > r_min_px]
    if not offsets or not rmaxs:
        raise DirectKError("The scan needs at least one offset and one r_max larger than r_min.")
    med = np.full((len(offsets), len(rmaxs)), np.nan)
    cnt = np.zeros((len(offsets), len(rmaxs)), dtype=int)
    total = len(offsets) * len(rmaxs)
    done = 0
    for a, off in enumerate(offsets):
        for b, rmax in enumerate(rmaxs):
            if progress is not None:
                progress(done, total)
            done += 1
            ser = compute_direct_k_series(
                dataset, tip_result, young_modulus_pa, poisson_ratio, plane_stress, pixel_size_mm,
                r_min_px, rmax, method=method, flank_exclusion_deg=flank_exclusion_deg,
                tip_offset_px=off, max_frames=max_frames)
            vals = [s.fit.K_I_pa_sqrt_m for s in ser.samples
                    if s.fit.K_I_pa_sqrt_m is not None and np.isfinite(s.fit.K_I_pa_sqrt_m)
                    and (s.fit.r_squared is None or s.fit.r_squared >= min_r_squared)]
            cnt[a, b] = len(vals)
            if vals:
                med[a, b] = float(np.median(vals))
    spread = np.full(len(offsets), np.nan)
    for a in range(len(offsets)):
        row = med[a][np.isfinite(med[a])]
        if len(row) >= 2 and np.mean(row) > 0:
            spread[a] = float((row.max() - row.min()) / np.mean(row))
    best = None
    k_best = None
    if np.isfinite(spread).any():
        a = int(np.nanargmin(spread))
        best = offsets[a]
        row = med[a][np.isfinite(med[a])]
        k_best = float(np.mean(row)) if len(row) else None
    return KSensitivityScan(offsets_px=offsets, r_max_px=rmaxs, median_k1=med, n_frames=cnt, spread=spread,
                            best_offset_px=best, k_at_best_pa_sqrt_m=k_best, method=method,
                            r_min_px=float(r_min_px))


def format_scan_table(scan: KSensitivityScan) -> str:
    """Plain-text table (MPa sqrt(m)) -- rows = offset, columns = r_max."""
    head = "offset \\ r_max " + " ".join(f"{r:7.0f}" for r in scan.r_max_px) + "   spread"
    lines = [head]
    for a, off in enumerate(scan.offsets_px):
        cells = " ".join("    n/a" if not np.isfinite(v) else f"{v / 1e6:7.2f}" for v in scan.median_k1[a])
        sp = "   n/a" if not np.isfinite(scan.spread[a]) else f"{100 * scan.spread[a]:5.0f}%"
        mark = " <" if (scan.best_offset_px is not None and off == scan.best_offset_px) else ""
        lines.append(f"{off:+8.1f} px    {cells}  {sp}{mark}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Fitting-window scan: K_I versus r_min and r_max
# ---------------------------------------------------------------------------

@dataclass
class KWindowScan:
    r_min_px: list
    r_max_px: list
    median_k1: np.ndarray            # [n_rmin, n_rmax], Pa sqrt(m); nan where the window is invalid/unfitted
    n_frames: np.ndarray
    method: str
    tip_offset_px: float
    grid_step_px: float | None       # median spacing of the DIC grid, for the "r_min vs resolution" hint


def dic_grid_step_px(dataset) -> float | None:
    """Median node spacing of the exported DIC grid (px), None if it cannot be told."""
    try:
        steps = [np.median(np.diff(np.asarray(a, dtype=float))) for a in (dataset.xs, dataset.ys) if len(a) > 1]
        steps = [abs(float(s)) for s in steps if np.isfinite(s) and s > 0]
        return float(min(steps)) if steps else None
    except (AttributeError, TypeError, ValueError):
        return None


def scan_k_window(dataset, tip_result, young_modulus_pa, poisson_ratio, plane_stress, pixel_size_mm,
                  r_min_list_px, r_max_list_px, method: str = "ring", tip_offset_px: float = 0.0,
                  flank_exclusion_deg: float = 15.0, max_frames: int = 12, min_r_squared: float = 0.5,
                  progress=None) -> KWindowScan:
    """Median K_I over (up to `max_frames`) tip frames for every fitting annulus
    (r_min, r_max) with r_max >= r_min + 8 px. A K-dominated zone shows up as a
    block of the table where K no longer changes; K that keeps drifting with
    r_min points at the blurred, unresolved zone around the tip (DIC subset and
    grid step), K that keeps drifting with r_max at higher-order terms or
    neighbouring cracks."""
    rmins = [float(r) for r in r_min_list_px]
    rmaxs = [float(r) for r in r_max_list_px]
    if not rmins or not rmaxs:
        raise DirectKError("The window scan needs at least one r_min and one r_max.")
    med = np.full((len(rmins), len(rmaxs)), np.nan)
    cnt = np.zeros((len(rmins), len(rmaxs)), dtype=int)
    total, done = len(rmins) * len(rmaxs), 0
    for a, rmin in enumerate(rmins):
        for b, rmax in enumerate(rmaxs):
            if progress is not None:
                progress(done, total)
            done += 1
            if rmax < rmin + 8.0:
                continue
            ser = compute_direct_k_series(
                dataset, tip_result, young_modulus_pa, poisson_ratio, plane_stress, pixel_size_mm,
                rmin, rmax, method=method, flank_exclusion_deg=flank_exclusion_deg,
                tip_offset_px=tip_offset_px, max_frames=max_frames)
            vals = [s.fit.K_I_pa_sqrt_m for s in ser.samples
                    if s.fit.K_I_pa_sqrt_m is not None and np.isfinite(s.fit.K_I_pa_sqrt_m)
                    and (s.fit.r_squared is None or s.fit.r_squared >= min_r_squared)]
            cnt[a, b] = len(vals)
            if vals:
                med[a, b] = float(np.median(vals))
    return KWindowScan(r_min_px=rmins, r_max_px=rmaxs, median_k1=med, n_frames=cnt, method=method,
                       tip_offset_px=float(tip_offset_px), grid_step_px=dic_grid_step_px(dataset))


def format_window_table(scan: KWindowScan) -> str:
    """Plain-text table (MPa sqrt(m)) -- rows = r_min, columns = r_max."""
    lines = ["r_min \\ r_max " + " ".join(f"{r:7.0f}" for r in scan.r_max_px)]
    for a, rmin in enumerate(scan.r_min_px):
        cells = " ".join("      -" if not np.isfinite(v) else f"{v / 1e6:7.2f}" for v in scan.median_k1[a])
        lines.append(f"{rmin:7.0f} px    {cells}")
    return "\n".join(lines)
