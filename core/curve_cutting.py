"""
Quasi-static load-displacement curve-beginning cutting, ported from
process_t.py's `cut_new_a1()`. Removes an initial "toe"/slack region of
the raw signal (everything before a chosen displacement value) and
re-zeroes the displacement axis (it finds where displacement first
reaches the chosen cut point, trims time/load/displacement to start
there, and shifts displacement so it starts at 0).
"""
from __future__ import annotations

import numpy as np


def cut_curve_start(timet: np.ndarray, load: np.ndarray, displacement: np.ndarray, cut_displacement: float):
    """Returns (new_timet, new_load, new_displacement), trimmed to start
    at `cut_displacement` and with displacement re-zeroed.

    Finds the FIRST sample (scanning from the start of the recording)
    whose displacement is already at or above `cut_displacement`, and
    keeps everything from there on. Deliberately NOT "the last sample
    anywhere in the whole recording that's still below cut_displacement"
    (an earlier version of this function worked that way, mirroring the
    original process_t.py port) -- that phrasing silently assumes
    displacement is monotonically increasing for the ENTIRE recording,
    which real quasi-static data often isn't: measurement noise near the
    cut point, or -- very commonly -- the file also logging the machine's
    return/unload stroke after the test (displacement ramping back down,
    dipping below `cut_displacement` again near the END of the file) will
    make `np.where(displacement < cut_displacement)[0][-1]` land on that
    LATER, unrelated dip instead of the intended start-of-loading point,
    silently discarding most or all of the real test data while still
    reporting a plausible-looking remaining point count. Scanning forward
    for the first crossing is immune to whatever the signal does
    afterward, which is what "cut curve BEGINNING" -- picked by looking
    at the start of the curve -- actually means.

    If no sample ever reaches `cut_displacement` (e.g. it was picked
    beyond the recording's own displacement range), nothing is cut
    (returns the curve unchanged, re-zeroed from its own first value)
    rather than collapsing to a single degenerate point."""
    displacement = np.asarray(displacement)
    reaches_cut = np.where(displacement >= cut_displacement)[0]
    idx = int(reaches_cut[0]) if len(reaches_cut) > 0 else 0
    new_timet = np.asarray(timet)[idx:]
    new_load = np.asarray(load)[idx:]
    new_displacement = displacement[idx:]
    new_displacement = new_displacement - new_displacement[0]
    return new_timet, new_load, new_displacement
