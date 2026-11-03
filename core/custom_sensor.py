"""
Custom (auxiliary) sensor model.

Ported from config_custom_sensors.py, which defined up to three
near-identical "custom sensor" blocks (extra load cells, extensometers,
thermocouples, etc. beyond the standard bar/specimen instrumentation),
each with: an enable checkbox, a time file + data file, a measurement
uncertainty (as a percentage and/or an absolute unit -- the original adds
both together when both are given), a multiplicative calibration factor,
and a "cut curve" tool that trims everything before a chosen time and
re-zeros the time axis. All three blocks are now one reusable dataclass;
the "cut curve" tool reuses ui/span_picker.py's TimeWindowPicker (using
just the selected window's start edge as the cut point).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.cpg_calc import thermal_output_strain
from core.lowpass import apply_lowpass_filter, DEFAULT_LOWPASS_CUTOFF_HZ


@dataclass
class CustomSensor:
    label: str = "Custom sensor"
    enabled: bool = False
    role: str = "none"  # "none" | "confinement_cell" | "laser_velocity" | "temperature" | "humidity" | "specimen_strain_gage"
    on_specimen: bool = False  # is this sensor physically mounted on the specimen (vs. elsewhere)?
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    data: np.ndarray = field(default_factory=lambda: np.array([]))
    # Pristine copy of time/data as originally loaded from file (or
    # (re)generated), BEFORE any `cut_from()` mutation. Populated at load
    # time (ui/pages/page_sensors.py's `_on_time_file`/`_on_data_file`/
    # `_maybe_regenerate_time`, and core/session.py's `_reload_path` for
    # session/profile restore) and left untouched by `cut_from()` itself.
    # Exists so the Monte-Carlo per-draw specimen-sensor cut
    # (calibrated_data_raw()/uncertainty_band_raw()/effective_raw_time()
    # below) always has an UNCUT signal to search within, regardless of
    # whether the one-shot "Auto-cut using wave travel time"/"Cut curve"
    # button was already clicked on this sensor -- see cut_in_montecarlo
    # below and core.specimen_sensor_alignment. Empty (falls back to
    # time/data) for a sensor that predates this field (e.g. restored
    # from an older session file).
    time_raw: np.ndarray = field(default_factory=lambda: np.array([]))
    data_raw: np.ndarray = field(default_factory=lambda: np.array([]))
    uncertainty_percent: float = 0.0
    uncertainty_absolute: float = 0.0
    multiplicative_factor: float = 1.0
    cut_time: float | None = None  # last-applied cut point, so a session restore can reapply it
    # If True, the Processing page's "Specimen-Mounted Sensor Alignment"
    # card recomputes this sensor's cut point FOR EVERY Monte-Carlo draw
    # (from that draw's own sampled bar/platen/specimen properties, see
    # core.specimen_sensor_alignment.specimen_sensor_cut_points_shpb/
    # _direct_impact) instead of the one-shot deterministic cut the "Auto-
    # cut using wave travel time" button applies (computed once, from the
    # mean properties only). Only meaningful when `on_specimen` is True.
    cut_in_montecarlo: bool = False
    # Optional low-pass filter (see core.lowpass.apply_lowpass_filter),
    # applied by calibrated_data() below -- so every consumer of this
    # sensor's data (the Overview plot, uncertainty_band(), the Monte-
    # Carlo specimen-sensor cut, Results & Export) picks it up
    # automatically without having to know about it individually.
    lowpass_enabled: bool = False
    lowpass_cutoff_hz: float = DEFAULT_LOWPASS_CUTOFF_HZ
    # Optional time-varying thermal output correction: subtracts the
    # apparent strain this gage's OWN thermal output (temperature change
    # alone, no mechanical load) would produce, evaluated at every sample
    # of a supplied temperature-vs-time profile (`thermal_time`/
    # `thermal_temperature`) rather than at one fixed operating
    # temperature -- see thermal_offset_strain() below for the physics and
    # the explicit "no thermal-lag model" design decision behind it. Only
    # meaningful for a sensor whose calibrated signal is itself a strain
    # (e.g. role "specimen_strain_gage" or "confinement_cell"), since the
    # correction is additive in the strain domain.
    thermal_correction_enabled: bool = False
    thermal_unit: str = "C"  # "C" or "F" -- must match the profile's/coefficients' own unit; purely a label, not used in the math
    thermal_degree: int = 4
    thermal_coeffs: list[float] = field(default_factory=lambda: [0.0] * 8)  # coeffs[i] = order-i coefficient, microstrain/unit^i
    thermal_t_ref: float = 24.0  # temperature the bridge was balanced/zeroed at
    thermal_time: np.ndarray = field(default_factory=lambda: np.array([]))
    thermal_temperature: np.ndarray = field(default_factory=lambda: np.array([]))

    def thermal_offset_strain(self, time: np.ndarray | None = None) -> np.ndarray:
        """Time-varying apparent-strain offset (strain, i.e. already
        divided by 1e6 from the microstrain the underlying polynomial is
        specified in) from this sensor's thermal output correction, one
        value per sample of `time` (defaults to `self.time` if omitted --
        pass `self.time_raw` instead to evaluate against the pristine,
        uncut time base, see calibrated_data_raw() below) -- an all-zero
        array if the correction is disabled, `time` is empty, or no
        temperature profile has been loaded (fewer than 2 samples).

        The supplied temperature-vs-time profile is linearly interpolated
        onto `time` (clamped/held at the profile's first or last
        temperature outside its own covered span, rather than NaN -- an
        unknown temperature just before/after the profile starts/ends
        should fall back to the nearest known value, not blank the gage
        signal there), then each interpolated temperature is run through
        the exact same manufacturer thermal-output polynomial as the CPG
        Calc tab's own static correction (`core.cpg_calc.
        thermal_output_strain`, Vishay Tech Note TN-504-1 eq. 3),
        independently at every time sample: strain_true = strain_indicated
        - (P(T(t)) - P(T_ref)).

        Deliberately NO thermal-lag/dynamics model between the supplied
        profile and this correction -- the gage is assumed to track the
        supplied temperature directly, matching TN-504-1's own
        steady-state calibration. This is an explicit design choice (not
        an oversight): the same tool must also stay valid for a slow
        quasi-static test with ambient temperature drift, where a lag
        model wouldn't apply anyway, so the profile itself is expected to
        already represent whatever temperature the gage actually sees --
        e.g. a specimen's own adiabatic-heating profile for a fast test,
        or a measured ambient temperature log for a slow one.
        """
        if time is None:
            time = self.time
        if not self.thermal_correction_enabled or len(time) == 0:
            return np.zeros(len(time))
        if len(self.thermal_time) < 2 or len(self.thermal_temperature) < 2:
            return np.zeros(len(time))
        coeffs = self.thermal_coeffs[: self.thermal_degree + 1]
        t_profile = np.interp(
            time, self.thermal_time, self.thermal_temperature,
            left=self.thermal_temperature[0], right=self.thermal_temperature[-1],
        )
        offset_ue = np.array([
            thermal_output_strain(coeffs, self.thermal_t_ref, t) for t in t_profile
        ])
        return offset_ue * 1e-6  # microstrain -> strain

    def _calibrated(self, time: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Shared implementation behind calibrated_data()/
        calibrated_data_raw(): multiplicative factor, optional low-pass
        filter, optional thermal offset -- applied to whichever
        (time, data) pair the caller supplies, so the exact same
        correction pipeline runs whether it's fed the live (possibly
        already cut_from()-mutated) signal or the pristine raw one."""
        factor = self.multiplicative_factor if self.multiplicative_factor else 1.0
        data = data * factor
        if self.lowpass_enabled and len(time) > 1:
            dt = float(np.median(np.diff(time)))
            data = apply_lowpass_filter(data, dt, self.lowpass_cutoff_hz)
        if self.thermal_correction_enabled:
            offset = self.thermal_offset_strain(time)
            n = min(len(data), len(offset))
            data = data.copy()
            data[:n] = data[:n] - offset[:n]
        return data

    def calibrated_data(self) -> np.ndarray:
        """Data with the multiplicative factor applied (matches
        `sensordata1 = [sensor_multi1_value*sensordata1[i] ...]` in the
        original, factor defaults to 1 if left blank/zero), additionally
        low-pass filtered if `lowpass_enabled` is set, and additionally
        offset by `thermal_offset_strain()` if `thermal_correction_enabled`
        is set. Pure/non-mutating: always derived fresh from `self.data`,
        never written back to it, so toggling either correction off then
        on again can't accumulate repeated corrections."""
        return self._calibrated(self.time, self.data)

    def effective_raw_time(self) -> np.ndarray:
        """`self.time_raw` if a pristine copy was captured at load time,
        else `self.time` -- the time base that matches whatever
        calibrated_data_raw() returns."""
        return self.time_raw if len(self.time_raw) > 0 else self.time

    def calibrated_data_raw(self) -> np.ndarray:
        """Like calibrated_data(), but derived from the pristine
        time_raw/data_raw copy captured at load time, bypassing any
        in-place cut already applied to self.time/self.data by
        cut_from() (e.g. via the one-shot "Auto-cut using wave travel
        time"/"Cut curve" buttons). This is what the Monte-Carlo
        per-draw specimen-sensor cut (cut_in_montecarlo) must use --
        each draw needs to search within the FULL original signal, not
        whatever is left after a previous one-shot cut already trimmed
        self.time/self.data down.

        Falls back to calibrated_data() when no raw copy exists (e.g. a
        sensor restored from an older session file, or one whose time
        base was generated rather than loaded from disk), so this is
        always safe to call."""
        if len(self.time_raw) == 0:
            return self.calibrated_data()
        return self._calibrated(self.time_raw, self.data_raw)

    def _uncertainty_band(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        half_width = self.uncertainty_absolute + data * (self.uncertainty_percent / 100.0)
        return data - half_width, data + half_width

    def uncertainty_band(self) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) bounds combining the percentage and absolute
        uncertainty additively, exactly matching the original's three
        `if sensor_unc_check == ...` combinations."""
        return self._uncertainty_band(self.calibrated_data())

    def uncertainty_band_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """Like uncertainty_band(), but built on calibrated_data_raw()
        instead -- see that method's docstring."""
        return self._uncertainty_band(self.calibrated_data_raw())

    def cut_from(self, cut_time: float) -> None:
        """Trim everything before `cut_time` and re-zero the time axis, in
        place -- matches the original's `cut_new_a{1,2,3}()` callbacks."""
        if len(self.time) == 0:
            return
        idx = int(np.searchsorted(self.time, cut_time, side="right"))
        idx = max(0, min(idx, len(self.time) - 1))
        self.time = self.time[idx:] - self.time[idx]
        self.data = self.data[idx:]
        self.cut_time = cut_time


def cut_and_resample(time: np.ndarray, data: np.ndarray, cut_time: float,
                      target_time: np.ndarray) -> np.ndarray:
    """Non-destructive, array-returning counterpart of `CustomSensor.
    cut_from`, used by the Monte-Carlo per-draw specimen-sensor alignment
    (see core.specimen_sensor_alignment.specimen_sensor_cut_points_shpb/
    _direct_impact and CustomSensor.cut_in_montecarlo): each Monte-Carlo
    draw needs its OWN cut point (computed from that draw's own sampled
    bar/platen/specimen properties, not the mean-only value the one-shot
    "Auto-cut"/"Cut curve" buttons use), so cutting can no longer mutate a
    single shared CustomSensor in place -- `time`/`data` here are meant to
    be the sensor's untouched, currently-loaded signal, read fresh for
    every draw rather than trimmed further and further down.

    Mirrors `cut_from`'s own index search + re-zero exactly (the first
    sample at or after `cut_time` becomes the new t=0), then linearly
    interpolates that cut-and-rezeroed segment onto `target_time` -- which
    is also how a specimen sensor recorded on its own clock/sample rate
    (possibly different from the bar/test acquisition's) ends up expressed
    on the same time base as the rest of that Monte-Carlo draw's results,
    resampling it if the two don't already share one. Padded with NaN
    outside the cut segment's own covered span (no extrapolation), the
    same convention already used by ui/pages/page_results_export.py's own
    DIC/strain-gage alignment (`np.interp(..., left=np.nan, right=np.nan)`)."""
    time = np.asarray(time, dtype=float)
    data = np.asarray(data, dtype=float)
    target_time = np.asarray(target_time, dtype=float)
    if len(time) == 0:
        return np.full(len(target_time), np.nan)
    idx = int(np.searchsorted(time, cut_time, side="right"))
    idx = max(0, min(idx, len(time) - 1))
    cut_t = time[idx:] - time[idx]
    cut_d = data[idx:]
    if len(cut_t) < 2:
        return np.full(len(target_time), np.nan)
    return np.interp(target_time, cut_t, cut_d, left=np.nan, right=np.nan)
