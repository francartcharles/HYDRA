"""Local crack-tip speed from a tip position-versus-time record, and the
Rayleigh wave speed it is usually compared with.

Used by the Results card's "K_I versus tip speed" plot (dynamic fracture:
the toughness of a brittle material is not a constant, it rises steeply with
crack speed, so a K_I measured on a fast, branching crack is not comparable
with the quasi-static K_Ic).
"""
from __future__ import annotations

import math

import numpy as np


MIN_SPEED_M_S = 0.5      # below this the tip is considered at rest (numerical noise of a flat fit)


def local_tip_speed(times_s, arc_px, pixel_size_mm: float, window: int = 5) -> np.ndarray:
    """Tip speed in m/s at every frame: slope of a straight line fitted to the
    arc length over the `window` frames centred on it (odd, >= 3; the window
    shrinks at the ends of the record). The sign is made positive for the
    direction in which the tip travels overall; a frame at which the tip went
    backwards or stood still (below MIN_SPEED_M_S) is returned as NaN, so it never counts as a
    propagation speed. Frames without a position are NaN."""
    t = np.asarray(times_s, dtype=float)
    s = np.asarray(arc_px, dtype=float)
    out = np.full(len(t), np.nan)
    if not pixel_size_mm or pixel_size_mm <= 0 or len(t) < 2:
        return out
    ok = np.isfinite(t) & np.isfinite(s)
    if np.count_nonzero(ok) < 2:
        return out
    window = max(3, int(window) | 1)
    half = window // 2
    idx = np.flatnonzero(ok)
    direction = 1.0 if (s[idx[-1]] - s[idx[0]]) >= 0 else -1.0
    for k, i in enumerate(idx):
        sel = idx[max(0, k - half):k + half + 1]
        if len(sel) < 2 or np.ptp(t[sel]) <= 0:
            continue
        slope = float(np.polyfit(t[sel], s[sel], 1)[0]) * direction          # px / s
        v = slope * pixel_size_mm * 1e-3
        out[i] = v if v > MIN_SPEED_M_S else np.nan
    return out


def rayleigh_speed(young_modulus_pa: float, poisson_ratio: float, density_kg_m3: float) -> float | None:
    """Rayleigh wave speed (m/s), Viktorov's approximation
    c_R = c_s (0.862 + 1.14 nu) / (1 + nu); None if an input is missing."""
    if not (young_modulus_pa and density_kg_m3 and density_kg_m3 > 0 and -1.0 < poisson_ratio < 0.5):
        return None
    mu = young_modulus_pa / (2.0 * (1.0 + poisson_ratio))
    cs = math.sqrt(mu / density_kg_m3)
    return cs * (0.862 + 1.14 * poisson_ratio) / (1.0 + poisson_ratio)


def spearman(x, y) -> float | None:
    """Rank correlation of two samples (NaN pairs dropped); None below 4 pairs
    or when one variable is constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(ok) < 4:
        return None
    def ranks(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a))
        r[order] = np.arange(len(a))
        for v in np.unique(a):                      # average ranks of ties
            m = a == v
            if np.count_nonzero(m) > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = ranks(x[ok]), ranks(y[ok])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])
