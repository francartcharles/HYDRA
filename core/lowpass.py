"""
Zero-phase low-pass filtering, shared by the bar gage signals (Dispersion
Corrections page, `core.models.Bar.filtered_signal`) and custom sensors
(`core.custom_sensor.CustomSensor.calibrated_data`) -- one small utility
instead of duplicating the same butter/filtfilt call in both places.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

# Matches the app's default gage-signal low-pass cutoff (300 kHz) --
# comfortably above a typical SHPB pulse's own frequency content (usually
# well under 100 kHz) while still cutting the high-frequency electrical/
# digitization noise a low-pass filter is meant to remove.
DEFAULT_LOWPASS_CUTOFF_HZ = 300_000.0
DEFAULT_LOWPASS_ORDER = 4


def apply_lowpass_filter(signal, dt: float | None, cutoff_hz: float | None,
                          order: int = DEFAULT_LOWPASS_ORDER) -> np.ndarray:
    """Zero-phase (`scipy.signal.filtfilt`) Butterworth low-pass filter.

    Fails soft -- returns `signal` unchanged (as a float ndarray) rather
    than raising -- whenever the filter can't meaningfully be applied:
    an empty signal, a non-positive `dt`/`cutoff_hz`, a cutoff at or past
    the signal's own Nyquist frequency (`dt` implies a sample rate
    `fs = 1/dt`; the cutoff must stay strictly below `fs/2`), or a signal
    too short for `filtfilt`'s default edge padding. This mirrors how
    other optional, best-effort processing steps in this app degrade
    (e.g. Results & Export's confinement uncertainty envelope) rather
    than surfacing a modal error for what is, after all, an optional
    checkbox.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) == 0 or dt is None or dt <= 0 or cutoff_hz is None or cutoff_hz <= 0:
        return signal
    fs = 1.0 / dt
    nyquist = fs / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    if not (0.0 < normalized_cutoff < 1.0):
        return signal
    b, a = butter(order, normalized_cutoff, btype="low")
    min_len = 3 * max(len(a), len(b))  # filtfilt's own default padlen requirement
    if len(signal) <= min_len:
        return signal
    return filtfilt(b, a, signal)
