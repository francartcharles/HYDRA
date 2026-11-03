"""
Central, in-memory project state.

The original application communicated between "windows" entirely through
module-level globals set by one Tkinter callback and read by another
(`corr_file.heat_check`, `corr_file.e1`/`e2`/`n2`, `test_type.sp_Tempa_value`,
`process_t.T0_pa`, `import_bar_tab.input_e_IB_check`, ...) -- over a hundred
of them, scattered across a dozen files, with no single place describing
"what a test is". This `Project` object is that single place: every page
reads and writes into one instance of it, passed down from MainWindow.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.models import Bar, Specimen
from core.material_tables import MaterialTable
from core.mc import MCParameterSet
from core.custom_sensor import CustomSensor
from core.rheology_models import MODEL_SLS, get_model


@dataclass
class ThermalSettings:
    """Corresponds to the original heat_check / type_bars_check /
    sp_Tempa_value / T0_pa / Tm_value globals."""
    heat_correction_enabled: bool = False
    bar_type: int = 1                 # 1: steel-like, 2: alloy-like, 3: use imported tables
    ambient_temperature_k: float = 293.0
    initial_temperature_k: float = 293.0
    melting_temperature_k: float = 1811.0
    melting_exponent: float = 0.5     # Johnson-Cook-style temperature term exponent (Tm_value[1])


def _default_visco_params() -> dict:
    return {p.key: p.fixed_default for p in get_model(MODEL_SLS).params}


@dataclass
class ViscoelasticSettings:
    """Corresponds to corr_file.num / fdamp / e1 / e2 / n2 / sde1 / sde2 / sdn2
    in the original -- generalized to whichever core.rheology_models.
    RheologyModel is selected. The original (and this app, until the
    Dispersion Corrections page gained a model picker) always assumed the
    Standard Linear Solid; core/dispersion.py's viscoelastic correction and
    the Viscoelastic Model Calibration tool now both support all four
    models, so this stores a generic {param_key: value} dict rather than
    hard-coded e1/e2/n2 fields. `model_key` says which of
    core.rheology_models.MODELS the values in `params`/`params_std`
    belong to -- always read them together, never assume `params` matches
    whatever model happens to be selected elsewhere."""
    poisson_ratio: float = 0.35
    damping_factor: float = 0.0       # from the free-decay analysis (decay.py)
    model_key: str = MODEL_SLS
    params: dict = field(default_factory=_default_visco_params)       # {param_key: value}
    params_std: dict = field(default_factory=dict)                     # {param_key: std}, when known (0 if not)


def migrate_viscoelastic_dict(data: dict) -> dict:
    """Upgrades a `viscoelastic` dict saved by a pre-multi-model version of
    this app (flat `e1`/`e2`/`n2`/`e1_std`/`e2_std`/`n2_std` fields, always
    Standard Linear Solid) into the current shape (`model_key` +
    `params`/`params_std` dicts), so an old session/profile/exported-test
    JSON still loads correctly instead of silently losing its calibrated
    values (core/session.py's `_simple_dataclass_from_dict` only sets
    attributes that already exist on `ViscoelasticSettings`, so it would
    otherwise just drop unrecognized legacy keys). A no-op on data that's
    already in the current shape."""
    if "params" in data or "e1" not in data:
        return data
    data = dict(data)
    data["model_key"] = MODEL_SLS
    data["params"] = {
        "e1": data.pop("e1", None), "e2": data.pop("e2", None), "n2": data.pop("n2", None),
    }
    data["params_std"] = {
        "e1": data.pop("e1_std", 0.0), "e2": data.pop("e2_std", 0.0), "n2": data.pop("n2_std", 0.0),
    }
    return data


@dataclass
class PlatenSettings:
    """Platen properties (any material -- tungsten carbide is common but not
    required), used when the SHPB/DI 'plat_check' option is enabled.
    Matches the 5-line config file format: density, elastic modulus,
    diameter, length, Poisson ratio."""
    params: MCParameterSet = field(default_factory=lambda: _new_platen_params())

    def expand(self, n: int, rng=None):
        return {k: self.params[k].sample(n, rng) for k in PLATEN_PARAM_ORDER}


PLATEN_PARAM_ORDER = ["rho", "e", "D", "L", "nu"]


def _new_platen_params() -> MCParameterSet:
    pset = MCParameterSet("Platen")
    pset.add("rho", "Density", "kg/m^3")
    pset.add("e", "Elastic Modulus", "Pa")
    pset.add("D", "Diameter", "m")
    pset.add("L", "Length", "m")
    pset.add("nu", "Poisson Ratio", "-")
    return pset


@dataclass
class ConfinementSettings:
    """Confining-cell (quasi-oedometric test) settings, ported from the
    'confine_check' path in corr_file.py / display_results.py.

    Two cell models (`corr_file.nonlinear_check`):
    - Linear (C1/C2/cell_modulus/ref_strain).
    - Nonlinear: two 6th-order polynomial calibration curves (orthoradial
      cell strain -> radial stress), typically produced by a numerical
      simulation of the cell at two specimen heights (initial, and at
      ref_strain)."""
    enabled: bool = False
    C1: float = 0.0
    C2: float = 0.0
    cell_modulus: float = 0.0     # Ev / E_c
    ref_strain: float = 0.0       # E_conf / ref_eps
    cell_modulus_std_fraction: float = 0.02  # original uses a fixed 2% std on cell_modulus
    nonlinear_enabled: bool = False
    cell_diameter: float = 0.0    # Dcell_conf
    calib_ortho_disp_ini: list = field(default_factory=list)
    calib_radial_ini: list = field(default_factory=list)
    calib_ortho_disp_esp: list = field(default_factory=list)
    calib_radial_esp: list = field(default_factory=list)
    # Radial/volumetric/equivalent strain calibration constants (from a
    # numerical simulation of the confinement cell); see
    # core.confinement.radial_strain().
    alpha00: float = 0.0
    alpha0_l2: float = 0.0


@dataclass
class QSSettings:
    """Quasi-static machine compliance correction + measurement
    uncertainties, ported from corr_file.py's disp_comp/load_comp/clu/meu
    globals (only shown in the original UI when speed_check == 1, i.e.
    Quasi-Static)."""
    machine_compliance_enabled: bool = False
    calib_load: list = field(default_factory=list)          # from load_comp_file
    calib_displacement: list = field(default_factory=list)   # from disp_comp_file
    cell_load_uncertainty_pct: float = 0.5    # "clu" -- load-cell calibration uncertainty
    measurement_uncertainty_pct: float = 1.0  # "meu" -- displacement/extensometer measurement uncertainty
    cut_displacement: float | None = None     # last-applied "cut curve beginning" point, for session restore


@dataclass
class DISettings:
    """Direct Impact-specific persistent state: the last-applied velocity-
    profile realignment cut, so a session restore can reapply it after
    re-loading the raw velocity file."""
    velocity_cut_time: float | None = None


@dataclass
class DICSettings:
    """Digital Image Correlation (DIC) setup, from the Image Correlation
    pipeline page: which kind of frame source was last used, the reference
    frame, the preprocessing applied before anything else (crop / downscale
    / frame step -- see core/dic_preprocessing.py), the region of interest
    (ROI, in pixel coordinates of the *preprocessed* frame) and the
    correlation mesh (subset size / step, both in pixels) that a DIC run
    uses. The video/image-sequence path itself is remembered in
    `Project.last_paths["dic_source"]`, following the same convention as
    every other loaded-file path. Run *results* (the tracked displacement/
    strain fields) are intentionally NOT persisted here -- like every other
    computed result in this app (see Project.last_results), they're kept
    in memory only and re-run on demand, to keep the session file small."""
    source_mode: str = "video"  # "video" | "sequence"
    reference_frame: int = 0

    # Preprocessing (core/dic_preprocessing.py), applied before the ROI/
    # mesh below -- crop is in RAW (full-resolution) frame coordinates.
    crop_x: int | None = None
    crop_y: int | None = None
    crop_width: int | None = None
    crop_height: int | None = None
    downscale_factor: int = 1   # 1 = full resolution
    frame_step: int = 1         # process 1 frame every N (1 = every frame)

    # ROI + correlation mesh, in PREPROCESSED (cropped/downscaled) frame
    # coordinates. `roi_shape` picks which of the three representations
    # below is active: "rectangle" uses roi_x/y/width/height (as before),
    # "polygon" uses roi_polygon instead, "oval" uses roi_ellipse instead
    # (roi_x/y/width/height are then left at their last rectangle value,
    # only for a quick fallback if the user switches back to "rectangle").
    roi_shape: str = "rectangle"  # "rectangle" | "polygon" | "oval"
    roi_x: int | None = None
    roi_y: int | None = None
    roi_width: int | None = None
    roi_height: int | None = None
    roi_polygon: list | None = None  # [[x, y], ...] vertices, "polygon" shape only
    roi_ellipse: list | None = None  # [cx, cy, rx, ry], "oval" shape only
    subset_size_px: int = 21   # correlation subset ("facet") side length
    step_size_px: int = 7      # spacing between neighboring mesh points
    search_radius_px: int = 15  # how far around the predicted position to search each subset
    tracking_mode: str = "fixed"  # "fixed" | "moving" -- see core/dic_processing.run_dic
    correlation_threshold: float = 0.0  # min NCC to keep a point; 0 = no quality filtering
    max_neighbor_deviation_px: float = 0.0  # hourglass-control analog; 0 = disabled
    subpixel_method: str = "icgn"  # "icgn" (affine Gauss-Newton refinement) | "parabolic" -- see core/dic_icgn

    # Calibration, for physical units -- neither affects the tracking
    # itself, only how results are displayed/exported.
    pixel_size_mm: float | None = None  # physical size of one RAW-frame pixel; None = stay in pixels
    frame_rate_hz: float | None = None  # camera frame rate, for the ROI-mean-vs-time plot's time axis
    material_density_kg_m3: float | None = None  # rho, for the "Dynamic Force" field (core/dic_kinematics.py)
    # Specimen cross-section (width x thickness, both in mm -- e.g. for a
    # flat/rectangular specimen viewed edge-on) used only to turn rho and
    # the tracked acceleration into an actual Dynamic Force in Newtons:
    # F = rho * (w * e) * a -- see core/dic_kinematics.compute_kinematics.
    specimen_width_mm: float | None = None
    specimen_thickness_mm: float | None = None

    # Results view isocontour color scale (core/dic_render.py's
    # vmin/vmax). "auto" recomputes the scale from each frame's own data
    # (as before); "fixed" reuses color_scale_min/max for every frame --
    # and for the exported video -- so color stays visually comparable
    # across the whole sequence instead of re-normalizing frame to frame.
    color_scale_mode: str = "auto"  # "auto" | "fixed"
    color_scale_min: float | None = None
    color_scale_max: float | None = None

    # Moving-average window (in frames) applied to the acceleration/
    # dynamic-force ROI-mean-vs-time curves before display/export -- see
    # core/dic_kinematics.smooth_series. 1 = no smoothing.
    force_smoothing_window: int = 1

    # Spatial smoothing of the displayed Results fields (Gaussian sigma in
    # mesh points, 0 = off) -- see core/dic_fields.smooth_nan_gaussian.
    spatial_smoothing_sigma: float = 0.0

    # Point probes (ui/roi_picker.py's RoiMeshPicker, placed with a
    # right-click on the same canvas as the ROI): pixel positions in
    # the PREPROCESSED reference frame, [[x, y], ...] -- each one reads
    # whatever field is currently selected at its nearest correlation
    # mesh point, across every processed frame (see the Image
    # Correlation page's "Probes" card / `_nearest_mesh_index`).
    probes: list | None = None

    # Unit for the per-frame timestamp burned into an exported video
    # (see ui/pages/page_image_correlation.py's TIMESTAMP_UNITS /
    # _format_timestamp) -- last choice made in the video-export dialog,
    # remembered for next time.
    timestamp_unit: str = "ms"

    # Which computed strain field is published as "DIC Strain (-)" to
    # Results & Export (see Project.dic_strain_result below) -- "eps_eq"
    # (von Mises equivalent), "exx", "eyy" or "exy".
    strain_definition: str = "eps_eq"
    # Whether that published strain (and its rate) is converted from
    # DIC's own small/nominal (linearized) strain to logarithmic (true)
    # strain -- see core/dic_kinematics.nominal_to_true_strain -- before
    # publishing, so it is defined the same way as the standard
    # processing engine's own "Strain (true)". Ignored (never converted)
    # when strain_definition == "exy": shear has no true-strain
    # conversion via this formula. Default True since an un-converted
    # comparison against the engine's own true strain is the more
    # surprising/less useful of the two for the Multi-Strain Comparison
    # plot this feeds.
    strain_convert_to_true: bool = True

    # Correction for the camera's own time axis (frame index -> elapsed
    # seconds), used everywhere the Image Correlation page builds a
    # time axis: the vs-time plots, velocity/acceleration/strain-rate
    # (which depend on dt), the DIC Strain series published below, every
    # CSV export with a time_s column, and the exported video's
    # timestamp burn-in. The camera's actual capture rate can differ
    # from the frame rate entered above (encoder jitter, a nominal
    # camera setting that isn't quite the true rate, ...), and is never
    # synchronized with the bar-signal DAQ's own sample rate in this
    # app, so this lets the user manually correct for that:
    #   effective_fps = frame_rate_hz * time_scale
    # 1.0 (default) leaves the entered frame rate unchanged; e.g. 1.02
    # means the camera actually ran 2% faster than entered. Set this by
    # matching a feature visible in both the DIC time series and the
    # bar-signal result (e.g. a known event's duration) -- there is no
    # automatic detection.
    time_scale: float = 1.0
    # Constant offset (seconds), ADDED after the scale correction above,
    # so DIC's own t=0 (its first processed frame) can be shifted to
    # coincide with the bar-signal result's own t=0 when the two
    # weren't triggered together. Does not affect velocity/
    # acceleration/strain-rate -- a constant shift doesn't change a
    # derivative.
    time_offset_s: float = 0.0

    # Whether the vs-time / DIC Strain / Multi-Strain Comparison plots
    # show a shaded +/- uncertainty envelope around each mean series --
    # see core/dic_uncertainty.py for the underlying model (a predictive
    # combination of sub-pixel registration precision, correlation-
    # quality degradation and within-subset distortion, NOT an
    # empirically-calibrated measurement uncertainty). Default True
    # since a DIC result plotted with no indication of its own
    # uncertainty invites over-reading it.
    show_uncertainty: bool = True
    # Envelope half-width, as a multiple of the model's own 1-sigma
    # estimate -- 1.0 for +/-1sigma (~68% coverage if the underlying
    # errors were Gaussian, which isn't verified), 2.0 for +/-2sigma
    # (~95%).
    uncertainty_sigma_k: float = 1.0


@dataclass
class SHPBSettings:
    """Correction toggles + parameters for the SHPB processing engine,
    corresponding to the many options `process_t.launch_process_SHPB`
    collected from the GUI before calling process_SHPB() per draw."""
    heat_check: bool = False
    plat_check: bool = False
    punch_check: bool = False
    inertia_check: bool = False
    force_check: int = 1                    # 1: F_input, 2: F_output, 3: F_average
    tenscomp_check: int = 1
    friction_coefficient: float = 0.0
    xd1: float = 0.0
    xd2: float = 0.0
    gage_temp_ib: float = 293.0
    gage_temp_ob: float = 293.0
    k_value: list = field(default_factory=lambda: [0, 0, 0, 1, 0, 1])
    output_window_offset: float = 0.0  # manual timing correction for the (formula-derived) transmitted-wave window
    use_corrected_signal: bool = False  # Processing tab's "Use dispersion-corrected signal" checkbox
    sk_check: bool = False
    sk_value: list = field(default_factory=lambda: [0, 0, 0, 1, 0, 1, 0])
    platen: PlatenSettings = field(default_factory=PlatenSettings)


@dataclass
class CovarianceLibraryEntry:
    """One self-contained (x-grid, mean curve, covariance matrix) triple,
    saved from the Covariance Analysis page's "Save to calibration
    library" action (ui/pages/page_covariance.py) for later use by the
    Bayesian Calibration page (ui/pages/page_bayes_calib.py) as one
    variable's data for one calibration test.

    Deliberately self-contained -- no live reference back to
    `Project.last_raw_draws`/`last_results` (which are recomputed on
    demand from whatever test is currently loaded and do NOT survive
    loading a different test) -- so an entry saved from test 1 remains
    valid data after test 1's own files are unloaded and test 2, 3, ...
    are processed in the same session. This is what makes a hierarchical,
    multi-test calibration possible within one HYDRA session: process a
    test, save the covariance pairs you need from it, load the next test,
    repeat -- the library accumulates across those cycles.

    Fields are plain Python floats/lists (not numpy arrays), matching the
    JSON-native shape `core/session.py` needs to persist this list
    (see `_covariance_library_to_dict`/`_from_dict`) -- the calibration
    engine (`core/calib_library_adapter.py`) converts to `np.asarray` on
    read, the same round-trip `core/session.py`'s own `last_correction`
    persistence already uses for its arrays.
    """
    name: str                    # user-given, unique within the library, e.g. "Test1_stress"
    source_label: str = ""       # provenance, e.g. "SHPB_stress" (the Covariance page's own matrix label)
    quantity_key: str = ""       # e.g. "stress"/"strain"/"temperature" -- QUANTITY_LABELS key, advisory only
    grid: list = field(default_factory=list)          # length-N common x-grid (e.g. strain)
    mean: list = field(default_factory=list)          # length-N representative/mean y-curve on that grid
    covariance: list = field(default_factory=list)    # NxN nested list -- the covariance matrix
    n_draws_used: int = 0        # provenance: how many Monte-Carlo draws went into this matrix
    grid_start: float = 0.0      # provenance: the Covariance page's own Reference/End/N inputs used
    grid_end: float = 0.0
    created_at: str = ""         # ISO timestamp, for display/sorting once the library has many entries


@dataclass
class Project:
    name: str = "Untitled analysis"
    n_montecarlo: int = 500

    test_mode: str = "Compression"
    geometry: str = "Cylindrical"
    loading_regime: str = "Hopkinson Bars (SHPB)"

    input_bar: Bar = field(default_factory=lambda: Bar.new("Input Bar"))
    output_bar: Bar = field(default_factory=lambda: Bar.new("Output Bar"))
    specimen: Specimen = field(default_factory=lambda: Specimen.new("Specimen"))

    thermal: ThermalSettings = field(default_factory=ThermalSettings)
    viscoelastic: ViscoelasticSettings = field(default_factory=ViscoelasticSettings)
    shpb: SHPBSettings = field(default_factory=SHPBSettings)
    qs: QSSettings = field(default_factory=QSSettings)
    di: DISettings = field(default_factory=DISettings)
    dic: DICSettings = field(default_factory=DICSettings)
    uncertainty_sigma: int = 2  # 1, 2, or 3 -- multiplier applied to every std-dev envelope shown/exported
    n_workers: int = 1  # Test Setup's "Performance" card -- worker processes used by core/parallel.py
    # (Monte-Carlo draw propagation, wave-dispersion branch solves, DIC point
    # tracking, the Bayesian calibration's initial sampling phase). 1 = fully
    # sequential, identical behavior to every version of this app before
    # this setting existed.
    dispersion_correction_mode: int = 1  # 0: none, 1: elastic, 2: viscoelastic -- Corrections page's own combo
    dispersion_setup_mode: int = 0  # 0: two bars (SHPB), 1: single output bar (DI) -- Corrections page's own combo
    dispersion_bandwidth_limit_enabled: bool = True  # Corrections page's "Limit correction bandwidth" checkbox --
    # caps the Pochhammer-Chree correction's own frequency-dependent gain above a normalized-frequency cutoff
    # (see core.dispersion._bandwidth_rolloff_weight) so it can't amplify a real signal's high-frequency noise
    # floor into visible ringing; on by default since the amplification is inherent to this kind of correction,
    # not a rare edge case.
    dispersion_bandwidth_limit: float = 0.6  # normalized frequency f*radius/c0 above which the correction gain rolls off to 0
    downsample_enabled: bool = False
    downsample_n: int = 1  # keep every Nth sample of every raw signal file loaded ("N_cut" in the original)
    confinement: ConfinementSettings = field(default_factory=ConfinementSettings)

    material_table_ib: MaterialTable = field(default_factory=lambda: MaterialTable("Input Bar"))
    material_table_ob: MaterialTable = field(default_factory=lambda: MaterialTable("Output Bar"))
    custom_sensors: list = field(default_factory=lambda: [
        CustomSensor(f"Custom sensor {i}") for i in (1, 2, 3)
    ])
    last_results: dict = field(default_factory=dict)  # e.g. {"QS": {...}, "SHPB": {...}, "DI": {...}}
    last_correction: dict | None = None  # last Dispersion Corrections run: {"result", "direct_impact", "mode_index", "xd1", "xd2"}
    # ROI-mean-vs-time series of one DIC-measured strain (Image
    # Correlation page's "Field for Results & Export" combo -- see
    # `DICSettings.strain_definition`/`strain_convert_to_true`),
    # published after each correlation run: {"definition", "label",
    # "time_s" (None unless a camera frame rate is set; already
    # corrected by `DICSettings.time_scale`/`time_offset_s`, see those
    # fields), "strain", "strain_rate", "converted_to_true",
    # "strain_uncertainty", "strain_rate_uncertainty" (1-sigma bands
    # from core/dic_uncertainty.py, same shape as "strain"/
    # "strain_rate" -- None if no uncertainty estimate was available for
    # this run, see `DICResult.noise_sigma`)}, or None. Read by Results
    # & Export's `_available_results`/`_enrich_with_strain_sources` to
    # offer "DIC Strain (-)" / "DIC Strain Rate (1/s)" as extra
    # quantities on the QS/SHPB/DI result currently being viewed there
    # (linearly interpolated onto that result's own time samples -- see
    # that method's docstring). Like `last_results`, NOT persisted in
    # the session file -- recomputed on demand from a fresh correlation
    # run.
    dic_strain_result: dict | None = None
    last_raw_draws: dict = field(default_factory=dict)  # {"QS"/"SHPB"/"DI": {quantity: [per-draw arrays]}} -- for covariance analysis
    covariance_library: list = field(default_factory=list)  # list[CovarianceLibraryEntry] -- self-contained
    # (grid, mean, covariance) triples saved from the Covariance Analysis page, accumulated across possibly
    # several different tests processed in this session, consumed by the Bayesian Calibration page as its
    # per-test/per-variable data source (see CovarianceLibraryEntry above). Unlike last_results/last_raw_draws,
    # this IS persisted across sessions (core/session.py) -- see CovarianceLibraryEntry's own docstring for why.
    last_paths: dict = field(default_factory=dict)  # remembers every file path loaded, for session persistence
    corrections_windows: dict = field(default_factory=dict)  # {window_key: [start, end]} -- Corrections page's own picker selection(s), for session/profile/test-export persistence
    time_step_overrides: dict = field(default_factory=dict)  # {time_source_key: dt} -- every "Time file" row (ui.widgets.TimeSourceRow) currently generating its time vector from a fixed time step instead of a loaded file; kept separate from last_paths (not a file path) so session/profile restore doesn't misreport it as a missing file

    # Loaded signals (path only; arrays are held by the bar/page that reads them)
    time_file: str = ""
    input_file: str = ""
    output_file: str = ""
    load_file: str = ""
    displacement_file: str = ""
    velocity_file: str = ""

    def expand_montecarlo(self):
        self.input_bar.expand(self.n_montecarlo)
        self.output_bar.expand(self.n_montecarlo)
        self.specimen.expand(self.n_montecarlo)
