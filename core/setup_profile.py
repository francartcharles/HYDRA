"""
Named setup profiles.

Separate from core/session.py's automatic "last session" restore: these
are user-named, manually saved/loaded bundles of the configuration a
person tends to reuse across many tests with the same equipment --
bar properties, a template specimen, the confinement cell, platens (if
any), custom sensor configuration, and (for Quasi-Static) the machine
compliance settings. Saved/loaded from the Test Setup page.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from core.custom_sensor import CustomSensor
from core.project import Project, migrate_viscoelastic_dict
from core.session import (
    _mc_params_to_dict, _mc_params_from_dict, _simple_dataclass_to_dict, _simple_dataclass_from_dict,
    _last_correction_to_dict, _last_correction_from_dict,
)

PROFILES_DIR = Path.home() / ".shpb_app" / "profiles"


def _safe_filename(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip()) or "profile"
    return slug


def list_profiles() -> list[str]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))


BAR_SIGNAL_ENCODING = "float64-le-zlib-base64"


def encode_signal(values) -> dict:
    """Lossless compact JSON-able encoding of a 1-D float array (raw
    little-endian float64 bytes, zlib-compressed, base64 text) -- a plain
    list of floats would be 5-10x larger and cost precision/readability
    for nothing, since nobody hand-edits a million-sample gage signal."""
    import base64
    import zlib
    import numpy as np
    arr = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return {
        "encoding": BAR_SIGNAL_ENCODING,
        "n": int(arr.size),
        "data": base64.b64encode(zlib.compress(arr.tobytes(), 6)).decode("ascii"),
    }


def decode_signal(blob) -> "np.ndarray":
    """Inverse of `encode_signal`; also accepts a plain list of numbers.
    Raises ValueError on a malformed/unknown payload."""
    import base64
    import zlib
    import numpy as np
    if isinstance(blob, (list, tuple)):
        return np.asarray(blob, dtype=float)
    if not isinstance(blob, dict) or blob.get("encoding") != BAR_SIGNAL_ENCODING:
        raise ValueError("unsupported bar-signal encoding")
    try:
        raw = zlib.decompress(base64.b64decode(blob["data"]))
        arr = np.frombuffer(raw, dtype="<f8").astype(float)
    except (KeyError, ValueError, zlib.error) as exc:
        raise ValueError(f"corrupt bar-signal data: {exc}") from exc
    if "n" in blob and int(blob["n"]) != arr.size:
        raise ValueError("bar-signal length mismatch")
    return arr


def raw_bar_signal(bar) -> "np.ndarray":
    """The bar's unmodified gage signal (before sign/offset). Uses
    `bar.signal_raw` when known, otherwise reconstructs it from
    `bar.signal` (= raw * sign - offset)."""
    import numpy as np
    if len(bar.signal_raw) > 0:
        return np.asarray(bar.signal_raw, dtype=float)
    if len(bar.signal) > 0:
        sign = bar.sign if bar.sign else 1.0
        return (np.asarray(bar.signal, dtype=float) + bar.offset) / sign
    return np.array([])


def bar_signals_to_dict(project: Project) -> dict:
    out = {}
    for key, bar in (("input_bar", project.input_bar), ("output_bar", project.output_bar)):
        raw = raw_bar_signal(bar)
        if len(raw) > 0:
            out[key] = encode_signal(raw)
    return out


def bar_signals_from_dict(project: Project, data: dict) -> list[str]:
    """Restores embedded bar signals (if any) onto the project's bars,
    re-applying each bar's CURRENT sign/offset. Returns the names of the
    bars restored ("input_bar"/"output_bar")."""
    restored = []
    for key, bar in (("input_bar", project.input_bar), ("output_bar", project.output_bar)):
        blob = (data.get("bar_signals") or {}).get(key)
        if blob is None:
            continue
        try:
            raw = decode_signal(blob)
        except ValueError:
            continue  # corrupt/unknown payload: skipped (caller may fall back to the file path)
        bar.signal_raw = raw
        bar.signal = raw * bar.sign - bar.offset
        restored.append(key)
    return restored


def reload_bar_signal_from_file(project: Project, which: str) -> bool:
    """Re-reads a bar's raw gage signal from the file remembered in
    `project.last_paths` (same parsing + optional downsampling as
    ui/pages/page_bar.py's BarPage) and re-applies the bar's sign/offset.
    `which` is "input_bar" or "output_bar". Returns True on success.
    Used as a fallback for test JSONs exported before signals were embedded."""
    import numpy as np
    from core.downsampling import downsample
    path = project.last_paths.get(f"{which}_signal", "")
    if not path or not Path(path).exists():
        return False
    values = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    values.append(float(line))
    except (OSError, ValueError):
        return False
    if not values:
        return False
    raw = np.asarray(values, dtype=float)
    if project.downsample_enabled:
        raw = np.asarray(downsample(raw, project.downsample_n), dtype=float)
    bar = project.input_bar if which == "input_bar" else project.output_bar
    bar.signal_raw = raw
    bar.signal = raw * bar.sign - bar.offset
    return True


def to_dict(project: Project, include_signals: bool = False) -> dict:
    """`include_signals=True` also embeds the raw Input/Output bar gage
    signals (used by Results & Export's "Export full test", so the test
    can be re-loaded on any machine; NOT used for the automatic session
    file or named profiles, which stay small and only remember paths)."""
    d = _to_dict_base(project)
    if include_signals:
        d["bar_signals"] = bar_signals_to_dict(project)
    return d


def _to_dict_base(project: Project) -> dict:
    return {
        "input_bar_params": _mc_params_to_dict(project.input_bar.params),
        "output_bar_params": _mc_params_to_dict(project.output_bar.params),
        "input_bar_sign": project.input_bar.sign, "input_bar_offset": project.input_bar.offset,
        "output_bar_sign": project.output_bar.sign, "output_bar_offset": project.output_bar.offset,
        "input_bar_lowpass_enabled": project.input_bar.lowpass_enabled,
        "input_bar_lowpass_cutoff_hz": project.input_bar.lowpass_cutoff_hz,
        "output_bar_lowpass_enabled": project.output_bar.lowpass_enabled,
        "output_bar_lowpass_cutoff_hz": project.output_bar.lowpass_cutoff_hz,
        "specimen_params": _mc_params_to_dict(project.specimen.params),
        "specimen_name": project.specimen.name,
        "test_mode": project.test_mode, "geometry": project.geometry,
        "confinement": _simple_dataclass_to_dict(project.confinement),
        "platen_params": _mc_params_to_dict(project.shpb.platen.params),
        "custom_sensors": [
            {
                "label": s.label, "enabled": s.enabled, "role": s.role,
                "uncertainty_percent": s.uncertainty_percent,
                "uncertainty_absolute": s.uncertainty_absolute,
                "multiplicative_factor": s.multiplicative_factor, "on_specimen": s.on_specimen,
                "cut_time": s.cut_time, "cut_in_montecarlo": s.cut_in_montecarlo,
                "lowpass_enabled": s.lowpass_enabled, "lowpass_cutoff_hz": s.lowpass_cutoff_hz,
                "thermal_correction_enabled": s.thermal_correction_enabled,
                "thermal_unit": s.thermal_unit, "thermal_degree": s.thermal_degree,
                "thermal_coeffs": list(s.thermal_coeffs), "thermal_t_ref": s.thermal_t_ref,
            }
            for s in project.custom_sensors
        ],
        "qs": _simple_dataclass_to_dict(project.qs),
        # Processing tab settings (shared by SHPB and Direct Impact: heat/
        # platen/punch/inertia checks, friction, gage temperatures, k/sk
        # coefficients, the transmitted-wave realignment offset -- minus
        # `platen`, saved separately above).
        "shpb": _simple_dataclass_to_dict(project.shpb),
        "di": _simple_dataclass_to_dict(project.di),
        # Dispersion Corrections tab settings.
        "dispersion_correction_mode": project.dispersion_correction_mode,
        "dispersion_setup_mode": project.dispersion_setup_mode,
        "dispersion_bandwidth_limit_enabled": project.dispersion_bandwidth_limit_enabled,
        "dispersion_bandwidth_limit": project.dispersion_bandwidth_limit,
        "corrections_windows": project.corrections_windows,
        "viscoelastic": _simple_dataclass_to_dict(project.viscoelastic),
        "thermal": _simple_dataclass_to_dict(project.thermal),
        "last_paths": project.last_paths,
        "last_correction": _last_correction_to_dict(project.last_correction),
        "time_step_overrides": project.time_step_overrides,
    }


def from_dict(project: Project, data: dict) -> None:
    _mc_params_from_dict(project.input_bar.params, data.get("input_bar_params", {}))
    _mc_params_from_dict(project.output_bar.params, data.get("output_bar_params", {}))
    project.input_bar.sign = data.get("input_bar_sign", project.input_bar.sign)
    project.input_bar.offset = data.get("input_bar_offset", project.input_bar.offset)
    project.output_bar.sign = data.get("output_bar_sign", project.output_bar.sign)
    project.output_bar.offset = data.get("output_bar_offset", project.output_bar.offset)
    project.input_bar.lowpass_enabled = data.get("input_bar_lowpass_enabled", project.input_bar.lowpass_enabled)
    project.input_bar.lowpass_cutoff_hz = data.get("input_bar_lowpass_cutoff_hz", project.input_bar.lowpass_cutoff_hz)
    project.output_bar.lowpass_enabled = data.get("output_bar_lowpass_enabled", project.output_bar.lowpass_enabled)
    project.output_bar.lowpass_cutoff_hz = data.get("output_bar_lowpass_cutoff_hz", project.output_bar.lowpass_cutoff_hz)

    _mc_params_from_dict(project.specimen.params, data.get("specimen_params", {}))
    project.specimen.name = data.get("specimen_name", project.specimen.name)
    if "test_mode" in data:
        project.test_mode = data["test_mode"]
    if "geometry" in data:
        project.geometry = data["geometry"]

    _simple_dataclass_from_dict(project.confinement, data.get("confinement", {}))
    _mc_params_from_dict(project.shpb.platen.params, data.get("platen_params", {}))

    sensor_dicts = data.get("custom_sensors", [])
    for i, sensor_data in enumerate(sensor_dicts):
        while i >= len(project.custom_sensors):
            project.custom_sensors.append(CustomSensor(f"Custom sensor {len(project.custom_sensors) + 1}"))
        sensor = project.custom_sensors[i]
        sensor.label = sensor_data.get("label", sensor.label)
        sensor.enabled = sensor_data.get("enabled", sensor.enabled)
        sensor.role = sensor_data.get("role", sensor.role)
        sensor.uncertainty_percent = sensor_data.get("uncertainty_percent", sensor.uncertainty_percent)
        sensor.uncertainty_absolute = sensor_data.get("uncertainty_absolute", sensor.uncertainty_absolute)
        sensor.multiplicative_factor = sensor_data.get("multiplicative_factor", sensor.multiplicative_factor)
        sensor.on_specimen = sensor_data.get("on_specimen", sensor.on_specimen)
        sensor.cut_time = sensor_data.get("cut_time")
        sensor.cut_in_montecarlo = sensor_data.get("cut_in_montecarlo", sensor.cut_in_montecarlo)
        sensor.lowpass_enabled = sensor_data.get("lowpass_enabled", sensor.lowpass_enabled)
        sensor.lowpass_cutoff_hz = sensor_data.get("lowpass_cutoff_hz", sensor.lowpass_cutoff_hz)
        sensor.thermal_correction_enabled = sensor_data.get(
            "thermal_correction_enabled", sensor.thermal_correction_enabled)
        sensor.thermal_unit = sensor_data.get("thermal_unit", sensor.thermal_unit)
        sensor.thermal_degree = sensor_data.get("thermal_degree", sensor.thermal_degree)
        sensor.thermal_coeffs = list(sensor_data.get("thermal_coeffs", sensor.thermal_coeffs))
        sensor.thermal_t_ref = sensor_data.get("thermal_t_ref", sensor.thermal_t_ref)

    _simple_dataclass_from_dict(project.qs, data.get("qs", {}))
    # calib_load/calib_displacement (the compliance curves themselves) are
    # plain lists on QSSettings, so _simple_dataclass_from_dict already
    # restores them like any other field.

    _simple_dataclass_from_dict(project.shpb, data.get("shpb", {}))
    _simple_dataclass_from_dict(project.di, data.get("di", {}))
    if "dispersion_correction_mode" in data:
        project.dispersion_correction_mode = data["dispersion_correction_mode"]
    if "dispersion_setup_mode" in data:
        project.dispersion_setup_mode = data["dispersion_setup_mode"]
    if "dispersion_bandwidth_limit_enabled" in data:
        project.dispersion_bandwidth_limit_enabled = data["dispersion_bandwidth_limit_enabled"]
    if "dispersion_bandwidth_limit" in data:
        project.dispersion_bandwidth_limit = data["dispersion_bandwidth_limit"]
    project.corrections_windows = dict(data.get("corrections_windows", {}))
    _simple_dataclass_from_dict(project.viscoelastic, migrate_viscoelastic_dict(data.get("viscoelastic", {})))
    _simple_dataclass_from_dict(project.thermal, data.get("thermal", {}))
    project.last_paths.update(data.get("last_paths", {}))
    project.time_step_overrides.update(data.get("time_step_overrides", {}))
    if "last_correction" in data:
        restored = _last_correction_from_dict(data["last_correction"])
        if restored is not None:
            project.last_correction = restored
    # Embedded gage signals (full-test export only) -- after sign/offset above.
    # Profiles never contain them (see to_dict's include_signals).
    bar_signals_from_dict(project, data)


def save_profile(project: Project, name: str) -> Path:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILES_DIR / f"{_safe_filename(name)}.json"
    save_to_path(project, path, name)
    return path


def save_to_path(project: Project, path, name: str = "") -> None:
    with open(path, "w") as f:
        json.dump({"name": name, **to_dict(project)}, f, indent=2)


def load_profile(project: Project, name: str) -> None:
    path = PROFILES_DIR / f"{_safe_filename(name)}.json"
    load_from_path(project, path, load_signals=False)


def load_from_path(project: Project, path, load_signals: bool = True) -> dict:
    """Like `load_profile`, but from an arbitrary file path (e.g. a
    companion .json next to a "Export full test" result file) rather than
    a named profile in PROFILES_DIR. `load_signals` controls restoring the
    Input/Output bar gage signals (embedded data, else the remembered file
    path); named profiles skip it. Returns the raw loaded dict (its
    "name" key, if present, is the original result's label).

    Also re-loads every remembered raw file (bar signals, custom sensor
    time/data, QS compliance curves, etc.) the same way a normal session
    restore does, and re-applies any custom-sensor "cut curve beginning"
    afterwards -- for ALL sensors, not just whichever one a page happens
    to refresh first."""
    from pathlib import Path as _Path
    from core.session import _reload_path

    with open(path, "r") as f:
        data = json.load(f)
    from_dict(project, data)

    for key, file_path in list(project.last_paths.items()):
        if not file_path or not _Path(file_path).exists():
            continue
        try:
            _reload_path(project, key, file_path)
        except (OSError, ValueError):
            pass  # a missing/unreadable file here just means less gets restored, not a hard failure

    # Gage signals: embedded ones were restored by from_dict above; for
    # older exports (no "bar_signals" key) fall back to the remembered
    # file paths, if they still exist on this machine.
    # (`load_signals=False` for named profiles: they hold reusable equipment
    # settings and must not swap the currently loaded gage signals.)
    if load_signals:
        restored_bars = set(bar_signals_from_dict(project, data))  # idempotent; which bars got embedded data
        for which in ("input_bar", "output_bar"):
            if which not in restored_bars:
                reload_bar_signal_from_file(project, which)

    # A custom sensor's time vector may be GENERATED from a time step
    # (ui.widgets.TimeSourceRow) instead of loaded from a file -- see the
    # matching block in core/session.py's from_dict for why this runs
    # after the file-reload loop above.
    for key, dt in project.time_step_overrides.items():
        if key.startswith("sensor_") and key.endswith("_time"):
            idx = int(key.split("_")[1])
            if idx < len(project.custom_sensors):
                sensor = project.custom_sensors[idx]
                if len(sensor.data) > 0:
                    import numpy as _np
                    sensor.time = dt * _np.arange(len(sensor.data))

    # Applied unconditionally (regardless of CustomSensor.cut_in_
    # montecarlo) -- see the identical comment in core/session.py's
    # from_dict for why: every OTHER consumer of sensor.time/.data still
    # expects it to reflect the last applied cut, same as before this
    # field existed.
    for sensor in project.custom_sensors:
        if sensor.cut_time is not None and len(sensor.time) > 0:
            sensor.cut_from(sensor.cut_time)

    return data


def delete_profile(name: str) -> None:
    path = PROFILES_DIR / f"{_safe_filename(name)}.json"
    if path.exists():
        path.unlink()


def load_test_csv(path) -> dict:
    """Reads a CSV exported by Results & Export's "Export full test" (or
    the regular "Export as .csv") button back into a {column: array}
    dict, for populating `project.last_results`."""
    import csv as csv_module
    import numpy as np
    with open(path, "r", newline="") as f:
        reader = csv_module.reader(f)
        header = next(reader)
        columns = {name: [] for name in header}
        for row in reader:
            for name, value in zip(header, row):
                columns[name].append(float(value))
    return {name: np.asarray(values) for name, values in columns.items()}
