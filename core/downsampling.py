"""
Raw-signal downsampling ("N_cut" in the original), ported from
process_t.py: keeps only every Nth sample of a raw measurement file,
to make processing very large files faster. Applied uniformly to every
raw signal (bar signals, time/load/displacement, velocity, custom sensor
data) so they all stay aligned/synchronized after downsampling -- matches
the original exactly: `[arr[int(N*y)] for y in range(int(len(arr)/N))]`.
"""
from __future__ import annotations

import numpy as np


def downsample(values, n: int):
    """Keep every Nth sample. `n <= 1` returns the array unchanged."""
    values = np.asarray(values)
    if n <= 1:
        return values
    count = int(len(values) / n)
    return values[np.arange(count) * n]
