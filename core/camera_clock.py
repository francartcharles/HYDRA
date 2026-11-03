"""
Camera clock file parsing + trigger-frame location, for the Camera / Video
Sync standalone tool.

The clock file is a TXT/CSV recording, alongside a camera's video/image
output, of:
  - an absolute time vector (same time base as the rest of the test's
    recorded signals),
  - the camera's exposure ("clock") signal: one pulse per frame, used to
    count frames and give each one an absolute timestamp,
  - optionally, the camera's trigger signal: a single edge marking the
    instant the external trigger was received.

If no trigger-signal column was recorded, the trigger frame can instead be
supplied directly (the camera's own reported trigger frame number) together
with its recording frequency, which is used both as a hint for exposure-
pulse detection and, if that frame falls outside the detected pulses, to
extrapolate its absolute time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

_DELIMITER_CANDIDATES = [",", ";", "\t"]


def _split_line(line: str, delimiter: str | None) -> list[str]:
    if delimiter is None:
        return line.split()
    return line.split(delimiter)


def read_clock_file(path: str) -> tuple[np.ndarray, list[str]]:
    """Reads a whitespace/comma/semicolon/tab-delimited numeric text file,
    auto-detecting the delimiter, an optional header row, and a French-
    style decimal comma. Returns `(data, column_labels)` with
    `data.shape == (n_samples, n_columns)`.

    Any line that isn't purely numeric once split (a header, a comment, a
    blank line) is skipped rather than raising, since these files commonly
    carry a one-line header naming each column.
    """
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        raw_lines = [ln.rstrip("\n\r") for ln in f]
    lines = [ln for ln in raw_lines if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError(f"No data found in {path}")

    # Pick whichever delimiter splits the first several lines into a
    # single, consistent field count -- preferring the widest split (a
    # delimiter that happens to also appear inside a decimal number, like
    # a stray comma, would otherwise win by matching every line "fine" at
    # 1 field).
    sample = lines[: min(10, len(lines))]
    best_delimiter, best_n_fields = None, 1
    for candidate in _DELIMITER_CANDIDATES + [None]:
        field_counts = {len(_split_line(ln, candidate)) for ln in sample}
        if len(field_counts) == 1:
            n_fields = next(iter(field_counts))
            if n_fields > best_n_fields:
                best_delimiter, best_n_fields = candidate, n_fields
    delimiter = best_delimiter

    header = None
    data_lines = lines
    first_fields = [f.strip() for f in _split_line(lines[0], delimiter)]
    if not _all_numeric(first_fields):
        header = first_fields
        data_lines = lines[1:]

    rows = []
    for ln in data_lines:
        fields = [f.strip() for f in _split_line(ln, delimiter)]
        values = _try_parse_numeric_row(fields)
        if values is not None:
            rows.append(values)
    if not rows:
        raise ValueError(f"No numeric data rows found in {path}")

    n_cols = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == n_cols]
    data = np.asarray(rows, dtype=float)
    if header is None or len(header) != n_cols:
        header = [f"Column {i + 1}" for i in range(n_cols)]
    return data, header


def _all_numeric(fields: list[str]) -> bool:
    return _try_parse_numeric_row(fields) is not None


def _try_parse_numeric_row(fields: list[str]) -> list[float] | None:
    values = []
    for f in fields:
        if not f:
            continue
        try:
            values.append(float(f.replace(",", ".")))
        except ValueError:
            return None
    return values or None


def decimate_for_display(time: np.ndarray, signal: np.ndarray, max_points: int = 20000
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Shrinks `(time, signal)` to about `max_points` points for plotting,
    without flattening narrow pulses the way plain stride subsampling would
    (a camera exposure/trigger signal is exactly that: brief pulses on a
    flat baseline -- a naive `[::step]` could easily land between pulses
    and drop them from the picture entirely).

    Splits the signal into `max_points // 2` bins and keeps each bin's min
    and max sample (in their original order), which preserves every
    excursion's shape while cutting a DAQ recording with millions of
    samples down to something that redraws instantly on every frame
    scrubbed to. Peak/trigger detection itself always runs on the full-
    resolution arrays -- this is for the plotted line only."""
    time = np.asarray(time)
    signal = np.asarray(signal)
    n = len(time)
    if n <= max_points or max_points < 2:
        return time, signal

    n_bins = max(1, max_points // 2)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    out_time = []
    out_signal = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        seg_signal = signal[lo:hi]
        seg_time = time[lo:hi]
        i_min = int(np.argmin(seg_signal))
        i_max = int(np.argmax(seg_signal))
        first, second = (i_min, i_max) if i_min <= i_max else (i_max, i_min)
        out_time.append(seg_time[first]); out_signal.append(seg_signal[first])
        if second != first:
            out_time.append(seg_time[second]); out_signal.append(seg_signal[second])
    return np.asarray(out_time), np.asarray(out_signal)


def _estimate_dt(time: np.ndarray) -> float:
    dt = np.diff(time)
    dt = dt[dt > 0]
    return float(np.median(dt)) if len(dt) else 1.0


def detect_pulse_train(time: np.ndarray, signal: np.ndarray, expected_period_s: float | None = None,
                        height: float | None = None, prominence: float | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Peak-detects a periodic pulse train (a camera's exposure/frame-sync
    output: one peak per frame). Returns `(peak_sample_indices, peak_times)`.

    `expected_period_s`, when known (from the video's own fps, the
    recording frequency, or the loaded frame count and clock duration),
    sets the minimum spacing between accepted peaks at half that period --
    without it, only a bare minimum spacing is enforced and noisy signals
    may over-detect."""
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    dt = _estimate_dt(time)

    distance_s = expected_period_s * 0.5 if expected_period_s else dt * 2
    distance_samples = max(1, int(round(distance_s / dt)))

    if prominence is None:
        span = float(np.ptp(signal))
        prominence = span * 0.3 if span > 0 else None

    peak_indices, _ = find_peaks(signal, height=height, distance=distance_samples, prominence=prominence)
    return peak_indices, time[peak_indices]


def estimate_frame_times(peak_times: np.ndarray, n_frames: int,
                          expected_period_s: float | None = None) -> np.ndarray:
    """Best-estimate absolute time for every frame 0..n_frames-1 of the
    loaded video/sequence: the detected exposure-pulse time where one was
    found for that frame, else extrapolated at a constant period from the
    last detected pulse (`expected_period_s` if given, else the median
    spacing between the detected pulses themselves).

    This is what lets the frame slider show a live "time ~= ..." readout,
    and every measurement-channel plot draw a vertical line at the
    currently-displayed frame, without re-running trigger location for
    every single frame scrubbed to.

    Returns an array of length `n_frames`; entries that can't be
    determined at all (no pulses detected and no period to extrapolate
    from) are NaN.
    """
    times = np.full(n_frames, np.nan)
    n_known = min(len(peak_times), n_frames)
    if n_known > 0:
        times[:n_known] = peak_times[:n_known]

    period = expected_period_s
    if period is None and n_known > 1:
        period = float(np.median(np.diff(peak_times[:n_known])))
    if period is None or n_known == 0 or n_known >= n_frames:
        return times

    extra_idx = np.arange(n_known, n_frames)
    times[n_known:] = peak_times[n_known - 1] + (extra_idx - (n_known - 1)) * period
    return times


def detect_rising_edge(time: np.ndarray, signal: np.ndarray, threshold: float | None = None) -> int | None:
    """First sample index where `signal` crosses from at-or-below to above
    `threshold` (default: midpoint between its min and max) -- for a
    trigger signal that steps once rather than pulsing once per frame.
    Returns None if the signal never rises above the threshold."""
    signal = np.asarray(signal, dtype=float)
    if threshold is None:
        threshold = (float(np.min(signal)) + float(np.max(signal))) / 2.0
    above = signal > threshold
    if not np.any(above):
        return None
    return int(np.argmax(above))


def _nearest_index(sorted_values: np.ndarray, target: float, match: str) -> int | None:
    """Position within `sorted_values` (ascending) closest to `target`,
    per `match`: "nearest", "after" (first value >= target) or "before"
    (last value <= target)."""
    if len(sorted_values) == 0:
        return None
    pos = int(np.searchsorted(sorted_values, target))
    if match == "after":
        return min(pos, len(sorted_values) - 1)
    if match == "before":
        return max(pos - 1, 0)
    candidates = [p for p in (pos - 1, pos) if 0 <= p < len(sorted_values)]
    return min(candidates, key=lambda p: abs(sorted_values[p] - target))


@dataclass
class TriggerResult:
    """Everything the UI needs to report a trigger-frame location and let
    the person judge whether it looks right."""
    frame_index: int | None            # index into the loaded video/image sequence
    frame_time: float | None           # absolute time (from the time vector) of that frame
    trigger_time: float | None         # raw trigger-signal edge time, when detected from a signal
    n_peaks_detected: int
    peak_times: np.ndarray
    peak_sample_indices: np.ndarray
    method: str
    warnings: list[str] = field(default_factory=list)


def locate_trigger_by_signal(time: np.ndarray, exposure: np.ndarray, trigger: np.ndarray, *,
                              expected_period_s: float | None = None, expected_n_frames: int | None = None,
                              match: str = "nearest") -> TriggerResult:
    """Locates the trigger frame from a recorded trigger-signal column: the
    trigger's rising edge is matched to the nearest (or first-after /
    last-before, per `match`) exposure pulse."""
    warnings: list[str] = []
    peak_indices, peak_times = detect_pulse_train(time, exposure, expected_period_s=expected_period_s)
    if expected_n_frames is not None and len(peak_indices) != expected_n_frames:
        warnings.append(
            f"Detected {len(peak_indices)} exposure pulses, but the loaded video/sequence has "
            f"{expected_n_frames} frames -- double check the peak detection against the plot below."
        )

    trigger_idx = detect_rising_edge(time, trigger)
    if trigger_idx is None:
        warnings.append("No rising edge found in the trigger signal.")
        return TriggerResult(None, None, None, len(peak_indices), peak_times, peak_indices,
                              "trigger signal", warnings)
    trigger_time = float(time[trigger_idx])

    if len(peak_indices) == 0:
        warnings.append("No exposure pulses detected -- cannot map the trigger to a frame.")
        return TriggerResult(None, None, trigger_time, 0, peak_times, peak_indices, "trigger signal", warnings)

    chosen = _nearest_index(peak_indices, trigger_idx, match)
    return TriggerResult(int(chosen), float(peak_times[chosen]), trigger_time,
                          len(peak_indices), peak_times, peak_indices, "trigger signal", warnings)


def locate_trigger_by_frame_number(time: np.ndarray, exposure: np.ndarray, frame_number: int,
                                    frequency_hz: float | None = None, *, frame_offset: int = 0,
                                    expected_period_s: float | None = None,
                                    expected_n_frames: int | None = None) -> TriggerResult:
    """Locates the trigger frame from a directly-known trigger frame number
    (the camera's own report of which frame it triggered on) plus its
    recording frequency.

    `frame_number` is the index into the loaded video/image sequence
    (0-based). `frame_offset` accounts for any exposure pulses recorded
    before the first frame that actually made it into the loaded
    sequence (e.g. an unsaved pre-trigger circular buffer): the exposure
    pulse used for `frame_number` is `frame_number + frame_offset`; it
    defaults to 0 (pulse *i* <-> loaded frame *i*, one-to-one).

    If that pulse falls outside the ones actually detected, its time is
    extrapolated from the first detected pulse using `frequency_hz`
    instead (a warning is added either way that this happened).
    """
    warnings: list[str] = []
    if expected_period_s is None and frequency_hz:
        expected_period_s = 1.0 / frequency_hz
    peak_indices, peak_times = detect_pulse_train(time, exposure, expected_period_s=expected_period_s)
    if expected_n_frames is not None and len(peak_indices) != expected_n_frames:
        warnings.append(
            f"Detected {len(peak_indices)} exposure pulses, but the loaded video/sequence has "
            f"{expected_n_frames} frames -- double check the peak detection against the plot below."
        )

    target = frame_number + frame_offset
    if 0 <= target < len(peak_indices):
        frame_time = float(peak_times[target])
        method = "exposure pulse count"
    elif frequency_hz:
        base_time = float(peak_times[0]) - frame_offset / frequency_hz if len(peak_times) else float(time[0])
        frame_time = base_time + target / frequency_hz
        warnings.append(
            f"Trigger frame #{frame_number} (exposure pulse #{target}) falls outside the "
            f"{len(peak_indices)} detected exposure pulses -- its time was extrapolated from the "
            "recording frequency rather than read directly off the time vector."
        )
        method = "frequency extrapolation"
    else:
        warnings.append(
            f"Trigger frame #{frame_number} falls outside the {len(peak_indices)} detected exposure "
            "pulses, and no recording frequency was given to extrapolate its time -- enter the "
            "camera's recording frequency."
        )
        return TriggerResult(frame_number, None, None, len(peak_indices), peak_times, peak_indices,
                              "frequency extrapolation", warnings)

    return TriggerResult(frame_number, frame_time, None, len(peak_indices), peak_times, peak_indices,
                          method, warnings)
