"""
Free-decay damping-coefficient analysis.

Ported from decay.py. Logic is unchanged (half-power / -3dB bandwidth
method on the FFT peak); all Tkinter widgets, the Windows-only
GetSystemMetrics font sizing and the blocking matplotlib figure have been
removed. The function now returns plain data; the UI layer is responsible
for plotting it (see ui/pages/page_decay.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.interpolate as sci


@dataclass
class DecayResult:
    damping_ratio: float          # df : dimensionless damping ratio (curve damping factor)
    natural_frequency_hz: float
    full_damping_factor: float    # df * 2*pi*fn  (rad/m, per original units convention)
    time: np.ndarray
    signal: np.ndarray
    exponential_envelope: np.ndarray
    freq: np.ndarray
    amplitude_spectrum: np.ndarray


def compute_decay(ftime: np.ndarray, fmeasure: np.ndarray) -> DecayResult:
    """Compute the damping ratio of a free-decay signal from its FFT peak.

    Parameters
    ----------
    ftime : time samples (s), uniformly spaced
    fmeasure : measured signal (e.g. strain-gage voltage) at those times
    """
    ftime = np.asarray(ftime, dtype=float)
    fmeasure = np.asarray(fmeasure, dtype=float)

    dt = ftime[1] - ftime[0]
    n = len(fmeasure)

    yi = np.fft.rfft(fmeasure)
    freq = np.fft.rfftfreq(n, d=dt)
    amplitude = np.abs(yi)

    # Cubic interpolation to locate the peak and its half-power points
    # precisely, exactly as in the original implementation.
    interp = sci.interp1d(freq, amplitude, kind="cubic")
    freq_fine = np.linspace(freq.min(), freq.max(), 5 * len(freq))
    amp_fine = interp(freq_fine)
    max_amp = amp_fine.max()

    peak_idx = int(np.argmax(amp_fine))
    half_power = max_amp / math.sqrt(2)

    lower_idx = next((i for i, a in enumerate(amp_fine) if a > half_power), 0)
    upper_idx = next(
        (i for i in range(lower_idx, len(amp_fine)) if amp_fine[i] < half_power),
        len(amp_fine) - 1,
    )

    freq_n = freq_fine[peak_idx]
    freq_l = freq_fine[lower_idx]
    freq_u = freq_fine[upper_idx]

    ratio = (freq_u - freq_l) / freq_n
    damping_ratio = math.sqrt(0.5 - math.sqrt(1.0 / (4 + 4 * ratio**2 - ratio**4)))

    max_v = fmeasure.max()
    envelope = max_v * np.exp(-damping_ratio * 2 * math.pi * freq_n * ftime)

    return DecayResult(
        damping_ratio=damping_ratio,
        natural_frequency_hz=freq_n,
        full_damping_factor=damping_ratio * 2 * math.pi * freq_n,
        time=ftime,
        signal=fmeasure,
        exponential_envelope=envelope,
        freq=freq,
        amplitude_spectrum=amplitude,
    )
