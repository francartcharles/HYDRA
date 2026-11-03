"""
Session persistence.

Automatically remembers the parameters you've entered and the files
you've loaded, and restores them the next time you start the app --
addressing the (very reasonable) complaint that re-entering every bar/
specimen parameter and re-browsing to every signal file each session was
tedious. Session state is written to a small JSON file in the user's home
directory; nothing is sent anywhere.

Only scalar settings and MCParameterSet values are stored directly. Raw
signal arrays (bar signals, custom-sensor data, ...) are NOT stored in the
session file itself -- instead, the file *path* used to load them is
remembered (`Project.last_paths`), and on load, this module re-reads each
remembered file from disk the same way the page that originally loaded it
would. If a remembered file has since been moved or deleted, that one
signal is silently skipped (with a note returned to the caller) rather
than failing the whole session restore.
"""
from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

from core.mc import MCValue
from core.models import Bar, Specimen
from core.dispersion import dispersion_result_to_dict, dispersion_result_from_dict
from core.project import Project, migrate_viscoelastic_dict

SESSION_DIR = Path.home() / ".shpb_app"
SESSION_PATH = SESSION_DIR / "last_session.json"


def _read_column(path: str) -> np.ndarray:
    values = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                values.append(float(line))
    return np.asarray(values)


def _last_correction_to_dict(last_correction) -> dict | None:
    """Serializes `project.last_correction` (the last Dispersion
    Corrections run: window/mode metadata + a DispersionResult) to plain
    JSON-safe data, so the corrected signal itself survives a session
    save, Setup Profile, or "Export full test" -- and is usable again by
    the Processing tab's "Use dispersion-corrected signal" option without
    re-running the correction."""
    if last_correction is None:
        return None
    out = {k: v for k, v in last_correction.items() if k != "result"}
    if out.get("timet") is not None:
        out["timet"] = list(out["timet"])
    out["result"] = dispersion_result_to_dict(last_correction["result"]) if last_correction.get("result") is not None else None
    return out


def _last_correction_from_dict(data) -> dict | None:
    if data is None:
        return None
    out = dict(data)
    if out.get("result") is not None:
        out["result"] = dispersion_result_from_dict(out["result"])
    # Undo _last_correction_to_dict's `list(out["timet"])` (needed for
    # JSON), so every consumer gets back the ndarray it originally had --
    # e.g. ui/pages/page_process.py's SHPB "Process" does `timet[i1:i2] -
    # xd1` on this, which raises a TypeError against a plain list.
    if out.get("timet") is not None:
        out["timet"] = np.asarray(out["timet"])
    return out


def _covariance_library_to_dict(entries: list) -> list:
    """Serializes `project.covariance_library` (a list of
    CovarianceLibraryEntry, see core/project.py's own docstring for why
    this field -- unlike last_results/last_raw_draws -- IS persisted:
    it's populated only by an explicit "Save to calibration library"
    click, and is meant to accumulate across several "load test ->
    process -> save pairs -> load next test" cycles within one session,
    so losing it on a restart mid-accumulation would destroy real,
    not-easily-reproducible work). Every field is already a JSON-safe
    plain type (str/int/float/list), so this is a straight field-by-
    field copy, not a numpy round-trip like `_last_correction_to_dict`."""
    return [
        {
            "name": e.name, "source_label": e.source_label, "quantity_key": e.quantity_key,
            "grid": list(e.grid), "mean": list(e.mean), "covariance": [list(row) for row in e.covariance],
            "n_draws_used": e.n_draws_used, "grid_start": e.grid_start, "grid_end": e.grid_end,
            "created_at": e.created_at,
        }
        for e in entries
    ]


def _covariance_library_from_dict(data) -> list:
    if not data:
        return []
    from core.project import CovarianceLibraryEntry
    return [
        CovarianceLibraryEntry(
            name=d["name"], source_label=d.get("source_label", ""), quantity_key=d.get("quantity_key", ""),
            grid=list(d.get("grid", [])), mean=list(d.get("mean", [])),
            covariance=[list(row) for row in d.get("covariance", [])],
            n_draws_used=d.get("n_draws_used", 0), grid_start=d.get("grid_start", 0.0),
            grid_end=d.get("grid_end", 0.0), created_at=d.get("created_at", ""),
        )
        for d in data
    ]


def _simple_dataclass_to_dict(obj) -> dict:
    """For dataclasses made only of scalars (ThermalSettings,
    ViscoelasticSettings, ConfinementSettings, and SHPBSettings minus its
    nested `platen`)."""
    out = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if is_dataclass(value) or isinstance(value, (Bar, Specimen)):
            continue  # handled separately (nested MCParameterSet-bearing objects)
        out[f.name] = value
    return out


def _simple_dataclass_from_dict(obj, data: dict) -> None:
    for key, value in data.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def _mc_params_to_dict(pset) -> dict:
    return {key: mc_value.to_dict() for key, mc_value in pset.values.items()}


def _mc_params_from_dict(pset, data: dict) -> None:
    for key, value_dict in data.items():
        if key in pset.values:
            # Mutate the EXISTING MCValue object in place (rather than
            # replacing the dict entry with a new one) so any already-open
            # page's MCParamTable -- which holds a direct reference to this
            # object -- reflects the change immediately on refresh(),
            # without needing the page to be rebuilt.
            restored = MCValue.from_dict(value_dict)
            existing = pset.values[key]
            existing.mean = restored.mean
            existing.dispersion = restored.dispersion
            existing.distribution = restored.distribution
            existing.unit = restored.unit


def to_dict(project: Project) -> dict:
    return {
        "name": project.name,
        "n_montecarlo": project.n_montecarlo,
        "test_mode": project.test_mode,
        "geometry": project.geometry,
        "loading_regime": project.loading_regime,
        "uncertainty_sigma": project.uncertainty_sigma,
        "n_workers": project.n_workers,
        "dispersion_correction_mode": project.dispersion_correction_mode,
        "dispersion_setup_mode": project.dispersion_setup_mode,
        "dispersion_bandwidth_limit_enabled": project.dispersion_bandwidth_limit_enabled,
        "dispersion_bandwidth_limit": project.dispersion_bandwidth_limit,
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
        "thermal": _simple_dataclass_to_dict(project.thermal),
        "viscoelastic": _simple_dataclass_to_dict(project.viscoelastic),
        "shpb": _simple_dataclass_to_dict(project.shpb),
        "dic": _simple_dataclass_to_dict(project.dic),
        "platen_params": _mc_params_to_dict(project.shpb.platen.params),
        "confinement": _simple_dataclass_to_dict(project.confinement),
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
        "last_paths": project.last_paths,
        "corrections_windows": project.corrections_windows,
        "last_correction": _last_correction_to_dict(project.last_correction),
        "covariance_library": _covariance_library_to_dict(project.covariance_library),
        "qs_cut_displacement": project.qs.cut_displacement,
        "di_velocity_cut_time": project.di.velocity_cut_time,
        "time_step_overrides": project.time_step_overrides,
    }


def from_dict(project: Project, data: dict) -> list[str]:
    """Restore settings (and re-load remembered files) into `project`.
    Returns a list of human-readable warnings for anything that couldn't
    be restored (e.g. a moved/deleted file)."""
    warnings: list[str] = []

    project.name = data.get("name", project.name)
    project.n_montecarlo = data.get("n_montecarlo", project.n_montecarlo)
    project.test_mode = data.get("test_mode", project.test_mode)
    project.geometry = data.get("geometry", project.geometry)
    loading_regime = data.get("loading_regime", project.loading_regime)
    # "Quasi-static (Load-Displacement)" was renamed to "Load/Displacement Data"
    # -- map an older session/profile/project file's value across so it still
    # selects the right Processing tab and Test Setup dropdown entry instead
    # of silently failing to match either.
    if loading_regime == "Quasi-static (Load-Displacement)":
        loading_regime = "Load/Displacement Data"
    project.loading_regime = loading_regime
    project.uncertainty_sigma = data.get("uncertainty_sigma", project.uncertainty_sigma)
    project.n_workers = data.get("n_workers", project.n_workers)
    project.dispersion_correction_mode = data.get("dispersion_correction_mode", project.dispersion_correction_mode)
    project.dispersion_setup_mode = data.get("dispersion_setup_mode", project.dispersion_setup_mode)
    project.dispersion_bandwidth_limit_enabled = data.get(
        "dispersion_bandwidth_limit_enabled", project.dispersion_bandwidth_limit_enabled)
    project.dispersion_bandwidth_limit = data.get("dispersion_bandwidth_limit", project.dispersion_bandwidth_limit)

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

    _simple_dataclass_from_dict(project.thermal, data.get("thermal", {}))
    _simple_dataclass_from_dict(project.viscoelastic, migrate_viscoelastic_dict(data.get("viscoelastic", {})))
    _simple_dataclass_from_dict(project.shpb, data.get("shpb", {}))
    _simple_dataclass_from_dict(project.dic, data.get("dic", {}))
    _mc_params_from_dict(project.shpb.platen.params, data.get("platen_params", {}))
    _simple_dataclass_from_dict(project.confinement, data.get("confinement", {}))

    for i, sensor_data in enumerate(data.get("custom_sensors", [])):
        while i >= len(project.custom_sensors):
            from core.custom_sensor import CustomSensor
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

    project.qs.cut_displacement = data.get("qs_cut_displacement")
    project.di.velocity_cut_time = data.get("di_velocity_cut_time")

    last_paths = data.get("last_paths", {})
    project.last_paths = dict(last_paths)
    project.corrections_windows = dict(data.get("corrections_windows", {}))
    project.time_step_overrides = dict(data.get("time_step_overrides", {}))
    if "last_correction" in data:
        restored = _last_correction_from_dict(data["last_correction"])
        if restored is not None:
            project.last_correction = restored
    if "covariance_library" in data:
        project.covariance_library = _covariance_library_from_dict(data["covariance_library"])

    # Re-load every remembered file the same way its originating page would.
    for key, path in last_paths.items():
        if not path or not Path(path).exists():
            if path:
                warnings.append(f"Could not find previously-used file for '{key}': {path}")
            continue
        try:
            _reload_path(project, key, path)
        except (OSError, ValueError) as exc:
            warnings.append(f"Could not reload '{key}' from {path}: {exc}")

    # A custom sensor's time vector may be GENERATED from a time step
    # (ui.widgets.TimeSourceRow) instead of loaded from a file -- in which
    # case there's no path in last_paths above to reload, only an entry
    # here. Runs after the reload loop so `sensor.data` (needed to know how
    # many samples to generate) is already restored. _SensorCard's own
    # _restore_from_session() (called when that widget is built, later)
    # reflects this back into its TimeSourceRow's checkbox/field.
    for key, dt in project.time_step_overrides.items():
        if key.startswith("sensor_") and key.endswith("_time"):
            idx = int(key.split("_")[1])
            if idx < len(project.custom_sensors):
                sensor = project.custom_sensors[idx]
                if len(sensor.data) > 0:
                    generated = dt * np.arange(len(sensor.data))
                    sensor.time = generated
                    sensor.time_raw = generated.copy()

    # Re-apply any "cut curve beginning" that was applied before the app
    # closed -- must happen after the reload loop above, since a cut needs
    # both the time and data files already reloaded. Applied unconditionally
    # (regardless of CustomSensor.cut_in_montecarlo): every OTHER consumer
    # of sensor.time/.data (the Custom Sensors page itself, Results &
    # Export's plain "Custom Sensor: ..." entry, _enrich_with_strain_
    # sources, the Processing page's own "Auto-cut" button) expects
    # sensor.time/.data to reflect the last applied cut, same as before
    # this field existed -- a sensor isn't shown "raw" just because its
    # cut is ALSO recomputed per Monte-Carlo draw. cut_from() only ever
    # rebinds sensor.time/.data to new slices, never touching
    # sensor.time_raw/.data_raw (set just above, fresh from disk/
    # generation), so the per-draw recomputation
    # (core.specimen_sensor_alignment.specimen_sensor_cut_points_shpb/
    # _direct_impact, in ui/pages/page_process.py's _on_run) can still
    # read the untouched signal via calibrated_data_raw()/
    # effective_raw_time() regardless of this re-applied cut or of
    # whether "Auto-cut" was also clicked -- no "Reset cut first" step
    # needed.
    for sensor in project.custom_sensors:
        if sensor.cut_time is not None and len(sensor.time) > 0:
            sensor.cut_from(sensor.cut_time)

    return warnings


def _reload_path(project: Project, key: str, path: str) -> None:
    # NOTE: "input_bar_signal"/"output_bar_signal" are intentionally NOT
    # handled here -- BarPage._restore_from_session() re-reads them itself
    # so it can also re-apply the bar's sign/offset on top of the raw file
    # (this generic reloader has no page to call back into for that).
    #
    # NOTE: "*_config" files (input_bar_config, output_bar_config,
    # specimen_config, platen_config) are ALSO intentionally NOT reloaded
    # here, even though their path is remembered -- the actual parameter
    # VALUES (mean/dispersion/distribution for every field) are already
    # fully restored from this same JSON's own input_bar_params/
    # output_bar_params/specimen_params/platen_params dicts, saved at
    # export time. Re-reading the file here would silently overwrite
    # those correctly-restored values with whatever's currently on disk,
    # which may well differ (e.g. the user tweaked a value in the app
    # without re-saving the file) -- exactly the kind of "a value changed
    # for no reason" bug this is guarding against. The remembered path is
    # kept only so the relevant page can show "loaded from X.txt" text.
    if key.startswith("sensor_") and key.endswith("_thermal_time"):
        idx = int(key.split("_")[1])
        if idx < len(project.custom_sensors):
            project.custom_sensors[idx].thermal_time = _read_column(path)
    elif key.startswith("sensor_") and key.endswith("_thermal_temperature"):
        idx = int(key.split("_")[1])
        if idx < len(project.custom_sensors):
            project.custom_sensors[idx].thermal_temperature = _read_column(path)
    elif key.startswith("sensor_") and key.endswith("_time"):
        idx = int(key.split("_")[1])
        if idx < len(project.custom_sensors):
            values = _read_column(path)
            project.custom_sensors[idx].time = values
            # Pristine snapshot -- see CustomSensor.time_raw's docstring
            # and ui/pages/page_sensors.py's _on_time_file (same pattern).
            # Any "re-apply last cut" step run after this (below, and by
            # setup_profile.load_from_path, which also calls this
            # function) mutates .time/.data but never .time_raw/.data_raw.
            project.custom_sensors[idx].time_raw = values.copy()
    elif key.startswith("sensor_") and key.endswith("_data"):
        idx = int(key.split("_")[1])
        if idx < len(project.custom_sensors):
            values = _read_column(path)
            project.custom_sensors[idx].data = values
            project.custom_sensors[idx].data_raw = values.copy()
    # Unrecognized keys (e.g. per-page time/velocity files consumed only
    # inline by a Processing tab) are kept in last_paths for the relevant
    # page to re-load itself if it chooses to, but aren't auto-reloaded here.


def save(project: Project) -> None:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    with open(SESSION_PATH, "w") as f:
        json.dump(to_dict(project), f, indent=2)


def load(project: Project) -> list[str] | None:
    """Returns a list of warnings (possibly empty) if a previous session
    was found and (at least partially) restored, or None if there was no
    previous session to restore."""
    if not SESSION_PATH.exists():
        return None
    try:
        with open(SESSION_PATH, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return from_dict(project, data)
