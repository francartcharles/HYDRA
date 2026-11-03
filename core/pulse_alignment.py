"""
Automatic time-alignment of the second (downstream) SHPB free-decay pulse
window against the first, for ui/pages/page_visco_calib.py.

Manually dragging both pulse windows by eye is imprecise at exactly the
sample where it matters most: pulse 2's window START. A pixel or two of
difference in where the mouse was released does not affect pulse 1's own
analysis (its window only sets the FFT's frequency resolution -- see
core/visco_calibration.py's `_predict_pulse`), but it directly distorts
every downstream comparison against pulse 2, both the Result plot's
overlay and the fit score, because the whole calibration is scored by how
well the *predicted* pulse 2 (computed from pulse 1 plus the physical
propagation delay over `z1`) matches the *measured* pulse 2 window the
user drew. An arbitrary few-sample offset in that draw masquerades as a
modeling error, and the Bayesian search then "compensates" for it by
biasing the fitted parameters -- not a real property of the material, just
noise from where a mouse button happened to be released.

`align_pulse2` removes that source of error. It keeps pulse 2's window
WIDTH exactly as the user dragged it -- how much of the decay to include
is legitimately their choice -- but re-derives its START by sliding a
window of that width across the raw signal, within a margin of the user's
rough placement, and picking the position whose shape best matches pulse
1's window. A parabolic interpolation around the best integer-sample lag
then refines that to sub-sample precision, which dragging with a mouse
cannot achieve (pixel resolution, not sample resolution).

The alignment search deliberately always uses Pearson correlation (shape
match), regardless of which metric (core/visco_calibration.py's
SCORE_FUNCS) the calibration itself is set to use. This is a different
sub-problem: pulse 2 is *expected* to be attenuated relative to pulse 1
(that is the damping/dispersion physics the calibration is trying to
measure), so judging alignment candidates by absolute amplitude match
(R²) would penalize the correct lag just for pulse 2 being physically
smaller than pulse 1 -- exactly the failure mode R² was chosen over
correlation to avoid in the calibration objective itself (see
core/visco_calibration.py's module docstring), but here it is the right
tool: correlation's scale-invariance is what makes it ignore the expected
attenuation and focus purely on finding where the *shape* lines up.
"""
from __future__ import annotations

import numpy as np

from core.visco_calibration import SCORE_FUNCS

_ALIGN_SCORE = SCORE_FUNCS["correlation"]


def align_pulse2(
    full_time: np.ndarray, full_wave: np.ndarray,
    window1: tuple[float, float], window2_hint: tuple[float, float],
    margin: float | None = None,
) -> tuple[float, float] | None:
    """Returns a refined (start, end) for pulse 2's window: the same width
    as `window2_hint`, but with its start re-centered, within `margin`
    seconds of the hint's own start, to whichever position best matches
    pulse 1's window shape. Returns None if there isn't enough signal
    around the hint to search (e.g. right at the edge of the recording).

    `margin` defaults to half of pulse 1's window width -- generous enough
    to correct a genuinely imprecise manual drag, tight enough that the
    search does not wander off and lock onto an unrelated feature of the
    trace (a reflection, a noise burst) elsewhere in the signal."""
    time = np.asarray(full_time, dtype=float)
    wave = np.abs(np.asarray(full_wave, dtype=float))  # matches how pulse1/pulse2 are built in _on_run

    dt = float(np.median(np.diff(time)))
    if dt <= 0:
        return None
    w1_start, w1_end = window1
    w2_start, w2_end = window2_hint
    width1_n = max(int(round((w1_end - w1_start) / dt)), 2)
    width2_n = max(int(round((w2_end - w2_start) / dt)), 2)

    i1 = int(np.searchsorted(time, w1_start, side="right") - 1)
    template = wave[i1: i1 + width1_n]
    if len(template) < 2:
        return None

    # Candidates of width2_n are compared against the same-length leading
    # part of the template when the two windows' widths differ (matches
    # how core.visco_calibration._align_pulses already reconciles
    # mismatched pulse lengths elsewhere in this app).
    m = min(width1_n, width2_n)
    ref = template[:m]
    if m < 2:
        return None

    if margin is None:
        margin = 0.5 * (w1_end - w1_start)
    margin_n = max(int(round(margin / dt)), 1)

    i2_hint = int(np.searchsorted(time, w2_start, side="right") - 1)
    lo = max(i2_hint - margin_n, 0)
    hi = min(i2_hint + margin_n, len(wave) - width2_n - 1)
    if hi <= lo:
        return None

    def score_at(i2: int) -> float:
        candidate = wave[i2: i2 + width2_n][:m]
        if len(candidate) < m:
            return -np.inf
        return _ALIGN_SCORE(candidate, ref)

    best_i2, best_score = i2_hint, -np.inf
    for i2 in range(lo, hi + 1):
        s = score_at(i2)
        if s > best_score:
            best_score, best_i2 = s, i2
    if not np.isfinite(best_score):
        return None

    # Sub-sample refinement: parabolic interpolation of the score around
    # the best integer lag (skipped at the search range's own edge, where
    # a 3-point parabola through a boundary optimum isn't meaningful).
    refined_i2 = float(best_i2)
    if lo < best_i2 < hi:
        y0, y1, y2 = score_at(best_i2 - 1), best_score, score_at(best_i2 + 1)
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            offset = 0.5 * (y0 - y2) / denom
            if -1.0 < offset < 1.0:  # sanity: a real peak refines by less than one whole sample
                refined_i2 = best_i2 + offset

    new_start = float(time[best_i2] + (refined_i2 - best_i2) * dt)
    new_end = new_start + (w2_end - w2_start)
    return new_start, new_end
