"""
N-camera (1 to 8) synchronisation / pre-processing pipeline (no Qt).

Port of the stand-alone ``images_processing2.py`` script, reorganised so that a GUI can drive it:

    prepare_calibration()  -> masters (dark / flat) with automatic rejection of bad frames, gain maps,
                              calibrated preview frames, diagnostics
    [interactive]          -> rotation lines, translation points, crop ROIs  (done by the GUI)
    process_sequences()    -> streaming calibration + single-resampling alignment + crop,
                              inter/intra-camera intensity normalisation, gradient homogenisation,
                              optional denoising / despeckle, 16-bit exports, time stamps, video

Differences from the original script (all of them listed in ``IMPROVEMENTS`` and recorded in the JSON report):

* frames are streamed from disk (the script kept three full float32 sequences in RAM);
* 8-bit input is promoted to the 16-bit range (x257) up front, so calibration, thresholds and exports share
  one scale;
* the gain (flat) map is computed ONCE per camera, not once per frame;
* flat-field "robust" method: hot/dead pixels repaired, low-frequency vignetting kept without a hard clip,
  pixel-level (dust / PRNU) part limited and lightly smoothed;  the original gaussian + hard clip is still
  available (``flat_method="legacy"``);
* rotation + translation are composed into ONE affine warp per frame (cubic by default) instead of two
  successive bilinear resamplings, and a validity mask of the warp is tracked;
* background homogenisation: "divide" (illumination is multiplicative) by a masked low-pass or robust
  polynomial surface, normalised to unit mean so the frame brightness is preserved, then ONE global output
  gain (identical for every frame and camera) instead of a per-frame min/max stretch.  The original method is
  available as ``background_mode="legacy"``;
* the denoiser is really applied (in the script ``denoised = recombined`` overwrote its result) and works on
  float data, not on an 8-bit per-frame min/max stretch; the temporal median is taken per camera;
* the frame order is derived from the time stamps (identical to the script's interleaving when the camera
  offsets are smaller than the frame step);
* ``timestamps.csv`` is written (the overlay burned into pixels is optional and OFF by default: it alters the
  image content that DIC later correlates).
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import os
import re
import shutil
import subprocess
from collections import OrderedDict
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

import cv2
import numpy as np

from core.parallel import check_cancelled

MAX16 = 65535.0
MAX_CAMERAS = 8
DENOISERS = ("none", "bilateral", "nlmeans", "bm3d", "tv", "temporal_median")
FLAT_METHODS = ("robust", "legacy")
BACKGROUND_MODES = ("divide", "legacy", "none")
BACKGROUND_MODELS = ("gaussian", "polynomial")
OUTPUT_SCALINGS = ("global", "none")
SMOOTHING_METHODS = ("gaussian", "ema", "moving_average")
INTENSITY_MODES = ("median", "mean", "first_frame")
INTERPOLATIONS = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}
DESPECKLE_MODES = ("median", "morphology", "hybrid")
INTENSITY_METHODS = ("pairwise", "median_scale", "none")
TONE_MATCHINGS = ("none", "lut")
STRETCH_MODES = ("none", "imagej_auto", "percentile", "manual")
STRETCH_SCOPES = ("global", "per_camera", "per_frame")

IMPROVEMENTS = [
    "streaming (no full sequence in RAM)", "8-bit input promoted to the 16-bit range",
    "gain map computed once per camera", "robust flat-field (hot/dead pixel repair, soft gain limits)",
    "rotation+translation composed into one warp (cubic)", "background divided (multiplicative model), unit-mean",
    "one global output gain instead of a per-frame stretch", "denoiser actually applied, float, temporal median per camera",
    "time-ordered frame sequence + timestamps.csv", "natural file sort",
    "inter-frame / inter-camera photometric equalisation by a pairwise log-gain solver (N_cameras_images_processing.py), "
    "default; gradient homogenisation and global output gain are now opt-in",
    "reference camera levelling (baseline drawn on the reference image), per-camera videos, no-timestamp copy",
    "contrast stretch (ImageJ 'Auto' B&C) written to a separate all_stretched folder, one common window for the whole sequence",
]


class CameraSyncError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------------
@dataclass
class CameraSyncConfig:
    master_output: str = "MASTER_OUTPUT"
    n_cameras: int = 3                         # 1 .. MAX_CAMERAS; every per-camera list below has exactly this length
    dark_folders: list = field(default_factory=lambda: ["", "", ""])
    flat_folders: list = field(default_factory=lambda: ["", "", ""])
    light_folders: list = field(default_factory=lambda: [None, None, None])
    input_folders: list = field(default_factory=lambda: ["", "", ""])
    input_bit_depth: int = 16
    use_static_light: bool = False
    max_calibration_frames: int = 100          # darks/flats: evenly sampled if there are more

    # time stamps (ns)
    use_timestamps: bool = True
    timestamp_ns: int = 0
    timestamp_step: float = 530.0
    camera_offsets_ns: list = field(default_factory=lambda: [0.0, 150.0, 330.0])
    burn_timestamp: bool = False               # draw the time (bottom right, white on black) into the exported all_processed pixels
    timestamp_font_scale: float = 0.6
    export_no_timestamp_copy: bool = True      # when the time is burned: also write all_processed_no_timestamp/

    skip_start: int = 10
    skip_end: int = 80

    # alignment / crop
    reference_camera_index: int = 0
    align_use_first_image: bool = True
    align_level_reference: bool = True         # draw a baseline on the reference camera that is rotated to horizontal
    align_interpolation: str = "cubic"
    autocrop: bool = True
    autocrop_margin_px: int = 0

    # dark rejection
    dark_saturation_max_frac: float = 0.01
    dark_outlier_std_factor: float = 3.0
    dark_min_mean_frac: float = 0.0
    # flat rejection
    flat_saturation_max_frac: float = 0.01
    flat_underexposed_max_frac: float = 0.5
    flat_min_mean_frac: float = 0.02
    flat_eps: float = 1e-6
    # flat-field correction
    flat_method: str = "robust"
    gain_smooth_kernel_frac: float = 0.02      # legacy method: gaussian kernel as a fraction of min(image dim)
    min_gain: float = 0.6                      # legacy method: hard clamp
    max_gain: float = 1.6
    flat_lowfreq_sigma_frac: float = 0.05      # robust: vignetting scale (gaussian sigma / min image dim)
    flat_pixel_gain_limits: tuple = (0.7, 1.4)  # robust: limits of the pixel-level (dust/PRNU) part
    flat_pixel_smooth_sigma_px: float = 1.0    # robust: smooths the flat's own noise
    dark_scaling: str = "per_frame"            # none | per_frame: fit the dark-current amplitude of every frame (darks recorded hotter/colder)
    dark_scale_min: float = 0.2                # limits of the fitted factor
    dark_scale_max: float = 1.5
    calibration_pedestal: float = 0.0          # ADU added after (raw - dark) / gain: keeps the noise of dark pixels instead of clipping it at 0
    flat_total_gain_limits: tuple = (0.6, 1.6)  # robust: limits of the full gain (as the script's clamp): above 1/min the noise is amplified too much

    # intensity equalisation (inter + intra camera)
    intensity_method: str = "pairwise"         # pairwise (log-gain solver on all frames) | median_scale (older, smoothed) | none
    equalize_gain_min: float = 0.6             # pairwise: limits of the per-frame gain
    equalize_gain_max: float = 1.6
    equalize_reg: float = 1e-2                 # pairwise: weight keeping the log-gains near 0 (fixes the global level)
    equalize_low_pct: float = 1.0              # pairwise: pixels below this percentile (either image) are ignored
    equalize_thumb_side: int = 160             # pairwise: the pairs are compared on thumbnails of this size (longest side)
    equalize_min_overlap_frac: float = 0.05    # pairwise: minimum fraction of common pixels to accept a pair
    equalize_max_pairs: int = 40000            # pairwise: above this, a sparse graph (time neighbours + anchors) replaces all pairs
    use_global_intensity_norm: bool = True
    global_intensity_mode: str = "first_frame"
    smoothing_method: str = "gaussian"
    gaussian_sigma: float | None = None
    gaussian_sigma_auto_factor: float = 1.0
    gradient_threshold_percentile: float = 85.0
    blend_strength: float = 0.9
    min_scale: float = 0.5
    max_scale: float = 2.0

    # gradient homogenisation
    background_mode: str = "none"              # none | divide | legacy   (the N-camera script does no gradient homogenisation)
    background_model: str = "gaussian"
    background_kernel: int | None = None       # px (None = automatic from the image size); sigma = 0.15 * kernel
    background_kernel_scale: float = 1.0
    background_poly_order: int = 3
    output_scaling: str = "none"               # none | global
    output_target_fraction: float = 0.92       # 99.9th percentile of the sampled frames -> this fraction of 65535
    output_scale_sample_frames: int = 24

    # denoising / despeckle
    denoiser: str = "none"
    bilateral_diameter: int = 2
    bilateral_sigma_color: float = 75.0        # 8-bit units, as in the script (converted internally)
    bilateral_sigma_space: float = 75.0
    nlmeans_patch_size: int = 7
    nlmeans_patch_distance: int = 11
    nlmeans_h_factor: float = 0.8
    bm3d_sigma: float | None = None
    tv_weight: float = 0.02
    temporal_median_window: int = 3
    despeckle: bool = False
    despeckle_mode: str = "median"
    despeckle_size: int = 1

    # optional tone matching of every frame to the first frame of the reference camera (overlap-masked,
    # slope-limited LUT of the N-camera script). NOTE: the script computed it but never applied it; it also removes the
    # real brightness changes of the specimen, so it is off by default.
    tone_matching: str = "none"
    lut_bins: int = 16384
    lut_smooth_sigma: float = 2.0
    lut_max_slope: float = 8.0
    soft_knee_pct: float = 92.0
    soft_knee_strength: float = 0.8
    max_clip_frac: float = 0.001
    min_overlap_pixels: int = 2000

    # contrast stretch ("Brightness & Contrast / Auto" of ImageJ) -- display copy, all_processed is left untouched
    stretch_mode: str = "imagej_auto"          # none | imagej_auto | percentile | manual
    stretch_scope: str = "global"              # global (one window for everything) | per_camera | per_frame
    stretch_saturated_pct: float = 0.35        # percentile mode: % of saturated pixels, split between both ends (ImageJ default)
    stretch_min: float = 0.0                   # manual mode (grey levels on the 0..65535 scale)
    stretch_max: float = 65535.0
    stretch_gamma: float = 1.0                 # < 1 brightens the shadows
    stretch_clahe: bool = False                # optional local contrast (CLAHE), applied after the linear stretch
    stretch_clahe_clip: float = 2.0
    stretch_clahe_tiles: int = 8
    stretch_sample_frames: int = 200           # frames used to build the global / per-camera histograms

    # exports
    export_calibrated: bool = True
    export_cropped: bool = True
    export_combined_cropped: bool = True
    export_processed: bool = True
    save_comparisons: bool = True
    comparison_every: int = 10
    comparison_tile_height: int = 400
    export_video: bool = True
    export_video_per_camera: bool = True       # cam<N>_<video_filename> in addition to the combined (time-ordered) video
    video_fps: int = 24
    video_filename: str = "processed.mkv"      # base name: the extension is set by the formats below
    export_video_mkv: bool = True              # FFV1, 16-bit, lossless
    export_video_avi: bool = True              # FFV1, 16-bit, lossless
    export_video_mp4: bool = True              # H.264 lossless (qp 0) but 8 or 10 bit: MP4 cannot carry FFV1 / 16-bit
    mp4_bit_depth: int = 8                     # 8 | 10 (the 16-bit values are rescaled: v / 65535 * (2^bits - 1))

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CameraSyncConfig":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in data.items() if k in names}
        n = int(kw.get("n_cameras", 3))
        for k, fill in (("dark_folders", ""), ("flat_folders", ""), ("light_folders", None), ("input_folders", ""),
                        ("camera_offsets_ns", 0.0)):
            if k in kw and kw[k] is not None:
                kw[k] = (list(kw[k]) + [fill] * n)[:n]
        for k in ("flat_pixel_gain_limits", "flat_total_gain_limits"):
            if k in kw and kw[k] is not None:
                kw[k] = tuple(kw[k])
        return cls(**kw)

    def folder(self, name: str, cam: int | None = None) -> str:
        base = self.master_output
        table = {
            "calibrated": f"{base}/calibrated_cam{{}}", "cropped": f"{base}/cropped_cam{{}}",
            "work_cropped": f"{base}/_work/cropped_cam{{}}", "work_hom": f"{base}/_work/hom_cam{{}}",
        }
        if name in table:
            return table[name].format(cam + 1)
        return {"processed": f"{base}/all_processed", "video": f"{base}/video", "comparison": f"{base}/comparison",
                "check": f"{base}/alignment_check", "combined": f"{base}/all_croppedANDaligned",
                "stretched": f"{base}/all_stretched", "calibration": f"{base}/calibration",
                "processed_clean": f"{base}/all_processed_no_timestamp",
                "work": f"{base}/_work"}[name]

    @property
    def report_path(self) -> str:
        return f"{self.master_output}/rotation_translation_report.json"

    def validate(self, check_folders: bool = True) -> None:
        if self.input_bit_depth not in (8, 16):
            raise CameraSyncError("Bit depth must be 8 or 16.")
        if self.flat_method not in FLAT_METHODS or self.background_mode not in BACKGROUND_MODES:
            raise CameraSyncError("Invalid flat-field / background method.")
        if self.background_model not in BACKGROUND_MODELS or self.output_scaling not in OUTPUT_SCALINGS:
            raise CameraSyncError("Invalid background model / output scaling.")
        if self.denoiser not in DENOISERS:
            raise CameraSyncError("Invalid denoiser.")
        if self.global_intensity_mode not in INTENSITY_MODES or self.smoothing_method not in SMOOTHING_METHODS:
            raise CameraSyncError("Invalid intensity normalisation option.")
        if self.align_interpolation not in INTERPOLATIONS:
            raise CameraSyncError("Invalid interpolation.")
        if self.despeckle and self.despeckle_size % 2 == 0:
            raise CameraSyncError("The despeckle size must be odd.")
        if self.denoiser == "temporal_median" and self.temporal_median_window % 2 == 0:
            raise CameraSyncError("The temporal median window must be odd.")
        if self.background_kernel is not None and self.background_kernel <= 0:
            raise CameraSyncError("The background kernel must be positive.")
        if self.background_kernel_scale <= 0:
            raise CameraSyncError("The background kernel scale must be positive.")
        if self.mp4_bit_depth not in (8, 10):
            raise CameraSyncError("The MP4 bit depth must be 8 or 10.")
        if self.export_video and not (self.export_video_mkv or self.export_video_avi or self.export_video_mp4):
            raise CameraSyncError("Video export is on but no video format is selected (MKV / AVI / MP4).")
        if self.dark_scaling not in ("none", "per_frame") or not 0 < self.dark_scale_min <= 1.0 <= self.dark_scale_max:
            raise CameraSyncError("Invalid dark scaling (mode, or limits that must satisfy 0 < min <= 1 <= max).")
        if self.calibration_pedestal < 0:
            raise CameraSyncError("The calibration pedestal must be >= 0.")
        if self.intensity_method not in INTENSITY_METHODS or self.tone_matching not in TONE_MATCHINGS:
            raise CameraSyncError("Invalid intensity method or tone matching.")
        if not (0 < self.equalize_gain_min <= 1.0 <= self.equalize_gain_max):
            raise CameraSyncError("Equalisation gain limits must satisfy 0 < min <= 1 <= max.")
        if self.equalize_thumb_side < 16 or self.equalize_max_pairs < 1 or not 0 <= self.equalize_low_pct < 50:
            raise CameraSyncError("Invalid equalisation settings (thumbnail >= 16 px, pairs >= 1, percentile in [0, 50)).")
        if self.stretch_mode not in STRETCH_MODES or self.stretch_scope not in STRETCH_SCOPES:
            raise CameraSyncError("Invalid contrast-stretch mode or scope.")
        if self.stretch_gamma <= 0:
            raise CameraSyncError("The stretch gamma must be positive.")
        if not 0 <= self.stretch_saturated_pct < 50:
            raise CameraSyncError("The percentage of saturated pixels must be in [0, 50).")
        if self.stretch_mode == "manual" and not self.stretch_min < self.stretch_max:
            raise CameraSyncError("Manual stretch: the minimum must be below the maximum.")
        if self.stretch_clahe and (self.stretch_clahe_tiles < 1 or self.stretch_clahe_clip <= 0):
            raise CameraSyncError("Invalid CLAHE settings.")
        if self.use_timestamps and self.timestamp_step < 0:
            raise CameraSyncError("The timestamp step must be >= 0.")
        if not 1 <= int(self.n_cameras) <= MAX_CAMERAS:
            raise CameraSyncError(f"The number of cameras must be between 1 and {MAX_CAMERAS}.")
        if len(self.camera_offsets_ns) != self.n_cameras:
            raise CameraSyncError(f"{self.n_cameras} camera offsets are required.")
        if not 0 <= self.reference_camera_index < self.n_cameras:
            raise CameraSyncError(f"The reference camera must be between 1 and {self.n_cameras}.")
        if self.skip_start < 0 or self.skip_end < 0:
            raise CameraSyncError("The number of skipped frames must be >= 0.")
        if not self.master_output:
            raise CameraSyncError("Choose an output folder.")
        if check_folders:
            for lst, name in ((self.dark_folders, "dark"), (self.flat_folders, "flat"), (self.input_folders, "input")):
                if len(lst) != self.n_cameras:
                    raise CameraSyncError(f"{self.n_cameras} {name} folders are required (got {len(lst)}).")
                for cam, p in enumerate(lst):
                    if not p or not Path(p).is_dir():
                        raise CameraSyncError(f"Camera {cam + 1}: {name} folder missing or not found ({p!r}).")
            if self.use_static_light:
                if len(self.light_folders) != self.n_cameras:
                    raise CameraSyncError(f"{self.n_cameras} static-light folders are required.")
                for cam, p in enumerate(self.light_folders):
                    if not p or not Path(p).is_dir():
                        raise CameraSyncError(f"Camera {cam + 1}: static-light folder missing ({p!r}).")


# ----------------------------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------------------------
def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def list_image_files(folder) -> list:
    p = Path(folder)
    files = [f for ext in ("*.bmp", "*.tif", "*.tiff", "*.png") for f in p.glob(ext)]
    return sorted(files, key=_natural_key)


def read_image(path, bit_depth: int = 16) -> np.ndarray:
    """float32 grey image in the 16-bit range (8-bit input is multiplied by 257)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise CameraSyncError(f"Failed to load image: {path}")
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float32)
    if bit_depth == 8:
        img *= 257.0
    return img


def _sample_files(files: list, max_n: int) -> list:
    if max_n and len(files) > max_n:
        idx = np.unique(np.linspace(0, len(files) - 1, max_n).round().astype(int))
        return [files[i] for i in idx]
    return files


def display_uint8(img, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Percentile contrast stretch to 8 bit (robust to hot pixels, unlike min/max)."""
    a = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------
# calibration masters
# ----------------------------------------------------------------------------------------------
def build_master_dark(folder, cfg: CameraSyncConfig, cam: int):
    files = _sample_files(list_image_files(folder), cfg.max_calibration_frames)
    if not files:
        raise CameraSyncError(f"No BMP/TIFF/PNG files found in {folder}")
    frames = [read_image(f, cfg.input_bit_depth) for f in files]
    means = [float(f.mean()) for f in frames]
    med, std = float(np.median(means)), (float(np.std(means)) if len(means) > 1 else 0.0)
    accepted, rejected = [], []
    for i, fr in enumerate(frames):
        sat = float(np.mean(fr >= 0.98 * MAX16))
        mean_frac = float(fr.mean()) / MAX16
        reason = None
        if sat > cfg.dark_saturation_max_frac:
            reason = f"saturated ({sat:.4f})"
        elif std > 0 and fr.mean() > med + cfg.dark_outlier_std_factor * std:
            reason = f"mean outlier (mean={fr.mean():.1f}, median={med:.1f})"
        if reason is None:
            accepted.append(fr)
        else:
            rejected.append({"index": i, "file": files[i].name, "pct_saturated": sat, "mean_frac": mean_frac, "reason": reason})
    diag = {"camera": cam + 1, "total_dark_frames": len(frames), "accepted_dark_frames": len(accepted),
            "rejected_dark_frames": rejected}
    if not accepted:
        diag["warning"] = "no acceptable dark frame: all frames used"
        accepted = frames
    return np.median(np.stack(accepted), axis=0).astype(np.float32), diag


def build_master_flat(folder, cfg: CameraSyncConfig, cam: int):
    files = _sample_files(list_image_files(folder), cfg.max_calibration_frames)
    if not files:
        raise CameraSyncError(f"No BMP/TIFF/PNG files found in {folder}")
    frames = [read_image(f, cfg.input_bit_depth) for f in files]
    accepted, rejected = [], []
    for i, fr in enumerate(frames):
        sat = float(np.mean(fr >= 0.98 * MAX16))
        under = float(np.mean(fr <= 0.01 * MAX16))
        mean_frac = float(fr.mean()) / MAX16
        reason = None
        if sat > cfg.flat_saturation_max_frac:
            reason = f"saturated ({sat:.4f})"
        elif under > cfg.flat_underexposed_max_frac:
            reason = f"underexposed ({under:.4f})"
        elif mean_frac < cfg.flat_min_mean_frac:
            reason = f"low mean ({mean_frac:.4f})"
        if reason is None:
            accepted.append(fr)
        else:
            rejected.append({"index": i, "file": files[i].name, "pct_saturated": sat, "pct_under": under,
                             "mean_frac": mean_frac, "reason": reason})
    diag = {"camera": cam + 1, "total_flat_frames": len(frames), "accepted_flat_frames": len(accepted),
            "rejected_flat_frames": rejected}
    if not accepted:
        diag["warning"] = "no acceptable flat frame: all frames used (results may be invalid)"
        accepted = frames
    return np.median(np.stack(accepted), axis=0).astype(np.float32), diag


def build_gain_map(master_flat: np.ndarray, master_dark: np.ndarray, cfg: CameraSyncConfig):
    """Gain map with median 1.  Returns (gain, flat_minus_dark, diagnostics)."""
    fd = (master_flat - master_dark).astype(np.float32)
    h, w = fd.shape
    diag = {"method": cfg.flat_method}
    if float(np.mean(fd)) <= 0:
        raise CameraSyncError("Master flat - master dark has a non-positive mean: check the flat/dark folders.")
    if cfg.flat_method == "legacy":
        safe = fd.copy()
        safe[safe <= 0] = cfg.flat_eps
        gain = safe / float(np.mean(safe))
        ks = max(3, int(min(h, w) * cfg.gain_smooth_kernel_frac))
        ks += (ks % 2 == 0)
        gain = cv2.GaussianBlur(gain.astype(np.float32), (ks, ks), 0)
        gain = np.clip(gain, cfg.min_gain, cfg.max_gain)
        med = float(np.median(gain))
        if med > 1e-6:
            gain = gain / med
        diag["clipped_fraction"] = float(np.mean((gain <= cfg.min_gain + 1e-6) | (gain >= cfg.max_gain * 1.0 - 1e-6)))
        if diag["clipped_fraction"] > 0.005:
            diag["warning"] = (f"{100 * diag['clipped_fraction']:.1f}% of the pixels hit the flat-field limits [{cfg.min_gain}, {cfg.max_gain}] "
                               "(vignetting not fully corrected there)")
        return gain.astype(np.float32), fd, diag

    # robust: 1. repair hot / dead / negative pixels against a 5x5 median
    med5 = cv2.medianBlur(np.ascontiguousarray(fd), 5)
    ref = np.maximum(med5, cfg.flat_eps)
    bad = (fd <= cfg.flat_eps) | (np.abs(fd / ref - 1.0) > 0.15)
    fd_rep = np.where(bad, med5, fd)
    fd_rep = np.maximum(fd_rep, cfg.flat_eps)
    diag["repaired_pixels"] = int(bad.sum())
    # 2. low frequency (vignetting / lamp profile): wide gaussian, NOT clipped
    sigma = max(2.0, cfg.flat_lowfreq_sigma_frac * min(h, w))
    low = cv2.GaussianBlur(fd_rep, (0, 0), sigma)
    low = np.maximum(low, cfg.flat_eps)
    # 3. pixel-level part (dust shadows, PRNU): ratio, smoothed against the flat's own noise, limited
    pix = fd_rep / low
    if cfg.flat_pixel_smooth_sigma_px > 0:
        pix = cv2.GaussianBlur(pix, (0, 0), cfg.flat_pixel_smooth_sigma_px)
    lo, hi = cfg.flat_pixel_gain_limits
    pix_clipped = np.clip(pix, lo, hi)
    gain = low * pix_clipped
    gain = gain / float(np.median(gain))
    glo, ghi = cfg.flat_total_gain_limits
    diag.update({"lowfreq_sigma_px": float(sigma), "pixel_part_clipped_fraction": float(np.mean(pix != pix_clipped)),
                 "gain_min": float(gain.min()), "gain_max": float(gain.max()),
                 "max_noise_amplification_wanted": float(1.0 / max(float(gain.min()), 1e-6))})
    clipped_total = float(np.mean((gain < glo) | (gain > ghi)))
    diag["total_limits_clipped_fraction"] = clipped_total
    if clipped_total > 0.005:
        diag["warning"] = (f"the flat field would amplify {100 * clipped_total:.1f}% of the pixels beyond the limits [{glo}, {ghi}] "
                           f"(up to x{diag['max_noise_amplification_wanted']:.1f} at the darkest part of the flat): the correction is limited "
                           "there so that the noise is not amplified. Widen the limits only if that vignetting is real, or crop that band.")
    gain = np.clip(gain, glo, ghi)
    gain = gain / float(np.median(gain))
    return gain.astype(np.float32), fd, diag


@dataclass
class CameraCalibration:
    master_dark: np.ndarray
    master_flat: np.ndarray
    gain_map: np.ndarray
    noise_floor: float
    static_pattern: np.ndarray | None = None
    pedestal: float = 0.0
    dark_scaling: str = "none"
    dark_scale_limits: tuple = (0.2, 1.5)
    _hp_dark: np.ndarray | None = field(default=None, repr=False)
    _hot: np.ndarray | None = field(default=None, repr=False)
    _hot_den: float = field(default=0.0, repr=False)

    HP_SIGMA = 3.0

    def prepare_dark_scaling(self):
        """High-pass of the master dark and the set of its 'hot' pixels (isolated bright dark-current pixels)."""
        hp = (self.master_dark - cv2.GaussianBlur(np.ascontiguousarray(self.master_dark, np.float32), (0, 0), self.HP_SIGMA)).astype(np.float32)
        sig = 1.4826 * float(np.median(np.abs(hp - np.median(hp))))
        hot = np.abs(hp) > max(5.0 * sig, 3.0)
        self._hp_dark, self._hot = hp, hot
        self._hot_den = float(np.sum(hp[hot].astype(np.float64) ** 2)) if hot.sum() >= 50 else 0.0

    def estimate_dark_scale(self, raw: np.ndarray) -> float:
        """Factor `a` such that raw ~ a * master_dark + scene: least-squares slope of the high-pass of the raw frame on the
        high-pass of the dark over the hot pixels. The scene is uncorrelated with the dark pattern, so it only adds variance.
        Returns 1 when the dark has no usable structure."""
        if self.dark_scaling != "per_frame" or self._hp_dark is None or self._hot_den <= 0.0:
            return 1.0
        hp_raw = raw - cv2.GaussianBlur(np.ascontiguousarray(raw, np.float32), (0, 0), self.HP_SIGMA)
        a = float(np.sum(hp_raw[self._hot].astype(np.float64) * self._hp_dark[self._hot])) / self._hot_den
        return float(np.clip(a, *self.dark_scale_limits))

    def calibrate(self, raw: np.ndarray, stats: dict | None = None) -> np.ndarray:
        a = self.estimate_dark_scale(raw)
        if stats is not None:
            stats["dark_scale"] = a
        dc = raw - (self.master_dark if a == 1.0 else a * self.master_dark)
        neg = dc < 0
        if neg.any():
            if stats is not None:
                stats["negative_fraction"] = float(neg.mean())
                stats["neg_mask"] = neg
            if self.pedestal <= 0:
                dc[neg] = self.noise_floor     # same rule as the script (the noise of the dark pixels is clipped)
        out = dc / self.gain_map
        if self.pedestal > 0:
            out = out + self.pedestal          # the negative noise stays (symmetric noise, no bias)
        if self.static_pattern is not None:
            out = out - self.static_pattern
        out = np.nan_to_num(out, nan=0.0, posinf=MAX16, neginf=0.0)
        return np.clip(out, 0.0, MAX16).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------
def angle_from_points(p1, p2) -> float:
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def rotation_to_apply(line_ref, line_cam) -> float:
    """Angle (deg) to apply to the camera so that its drawn line is parallel to the reference line."""
    return angle_from_points(*line_ref) - angle_from_points(*line_cam)


def leveling_angle(line) -> float:
    """Angle (deg) to apply to an image so that the line drawn on it (p1 -> p2) becomes horizontal (pointing to +x)."""
    return rotation_to_apply(((0.0, 0.0), (1.0, 0.0)), line)


def build_affine(shape, angle_deg: float = 0.0, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """2x3 matrix = translation(dx, dy) after rotation by `angle_deg` about the image centre
    (same convention as the script's rotate_image then translate_image, but as ONE warp)."""
    h, w = shape[:2]
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle_deg, 1.0)
    T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
    return (T @ R)[:2]


def warp_image(img: np.ndarray, M: np.ndarray, size, interpolation: str = "cubic") -> np.ndarray:
    out = cv2.warpAffine(np.ascontiguousarray(img, dtype=np.float32), M.astype(np.float64), size,
                         flags=INTERPOLATIONS[interpolation], borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.clip(out, 0.0, None)


def warp_valid_mask(shape, M: np.ndarray, size, interpolation: str = "cubic") -> np.ndarray:
    ones = np.ones(shape[:2], np.float32)
    m = cv2.warpAffine(ones, M.astype(np.float64), size, flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999
    support = 2 if interpolation != "linear" else 1
    if support and not np.array_equal(M[:, :2], np.eye(2)):
        m = cv2.erode(m.astype(np.uint8), np.ones((2 * support + 1, 2 * support + 1), np.uint8)).astype(bool)
    return m


def is_identity(M: np.ndarray) -> bool:
    return bool(np.allclose(M, np.array([[1, 0, 0], [0, 1, 0]], dtype=float), atol=1e-9))


def largest_true_rectangle(mask: np.ndarray):
    """(x, y, w, h) of the largest axis-aligned rectangle containing only True pixels."""
    mask = np.asarray(mask, bool)
    h, w = mask.shape
    heights = np.zeros(w, dtype=int)
    best = (0, 0, 0, 0)
    best_area = 0
    for row in range(h):
        heights = np.where(mask[row], heights + 1, 0)
        stack = []   # (start index, height)
        for i in range(w + 1):
            cur = heights[i] if i < w else 0
            start = i
            while stack and stack[-1][1] >= cur:
                s, hh = stack.pop()
                area = hh * (i - s)
                if area > best_area:
                    best_area, best = area, (s, row - hh + 1, i - s, hh)
                start = s
            stack.append((start, cur))
    return best if best_area > 0 else None


def intersect_rois(rois, shape, margin_px: int = 0):
    """Intersection of per-camera ROIs (x, y, w, h) with an optional margin. Returns (bbox, reason)."""
    xs0 = [r[0] for r in rois]
    ys0 = [r[1] for r in rois]
    xs1 = [r[0] + r[2] - 1 for r in rois]
    ys1 = [r[1] + r[3] - 1 for r in rois]
    x0, y0, x1, y1 = max(xs0), max(ys0), min(xs1), min(ys1)
    if x1 < x0 or y1 < y0:
        return None, "empty_intersection_of_rois"
    x0 = max(0, x0 - margin_px)
    y0 = max(0, y0 - margin_px)
    x1 = min(shape[1] - 1, x1 + margin_px)
    y1 = min(shape[0] - 1, y1 + margin_px)
    cw, ch = x1 - x0 + 1, y1 - y0 + 1
    if cw <= 0 or ch <= 0:
        return None, "invalid_after_margin"
    return (int(x0), int(y0), int(cw), int(ch)), None


@dataclass
class AlignmentResult:
    reference_camera: int
    angles_deg: list
    translations_px: list          # [(dx, dy)] per camera
    matrices: list                 # 2x3 per camera (lists)
    image_size: tuple              # (w, h)
    method: str = "interactive_line_point"

    def matrix(self, cam: int) -> np.ndarray:
        return np.asarray(self.matrices[cam], dtype=np.float64)


def make_alignment(shape, reference_camera: int, angles_deg, translations_px, method="interactive_line_point"):
    mats = [build_affine(shape, a, t[0], t[1]).tolist() for a, t in zip(angles_deg, translations_px)]
    return AlignmentResult(reference_camera, [float(a) for a in angles_deg],
                           [(float(t[0]), float(t[1])) for t in translations_px], mats,
                           (int(shape[1]), int(shape[0])), method)


def identity_alignment(shape, reference_camera: int = 0, n_cameras: int = 3) -> AlignmentResult:
    return make_alignment(shape, reference_camera, [0.0] * n_cameras, [(0.0, 0.0)] * n_cameras, "identity")


# ----------------------------------------------------------------------------------------------
# homogenisation
# ----------------------------------------------------------------------------------------------
def auto_background_kernel(shape, scale: float = 1.0, kernel: int | None = None) -> int:
    if kernel:
        k = int(kernel)
    else:
        k = max(15, int(min(shape[:2]) * scale))
    return k + (k % 2 == 0)


def gaussian_background(img: np.ndarray, valid: np.ndarray | None, sigma: float) -> np.ndarray:
    """Illumination estimate: masked gaussian-weighted LOCAL-LINEAR regression, computed at reduced resolution.

    A plain (normalised-convolution) gaussian low-pass is biased wherever the window is cut by the image border
    and wherever the illumination is curved: on a vignetted image it leaves ~60 % of the edge falloff in place.
    Fitting a plane inside every gaussian window (value + slope) removes that bias at the borders and
    keeps the low-pass behaviour (speckle is averaged out as long as sigma >> speckle size)."""
    h, w = img.shape
    m = np.ones((h, w), np.float32) if valid is None else valid.astype(np.float32)
    f = max(1, int(sigma // 4))
    ws, hs = max(8, w // f), max(8, h // f)
    fx, fy = w / float(ws), h / float(hs)
    sig = max(1.0, sigma / (0.5 * (fx + fy)))
    I = cv2.resize(img * m, (ws, hs), interpolation=cv2.INTER_AREA).astype(np.float64)
    M = cv2.resize(m, (ws, hs), interpolation=cv2.INTER_AREA).astype(np.float64)
    ys, xs = np.mgrid[0:hs, 0:ws].astype(np.float64)

    def G(a):
        return cv2.GaussianBlur(a, (0, 0), sigmaX=sig, sigmaY=sig, borderType=cv2.BORDER_CONSTANT)

    # moments about every pixel (x0, y0) = (xs, ys)
    g0, gx, gy = G(M), G(M * xs), G(M * ys)
    gxx, gyy, gxy = G(M * xs * xs), G(M * ys * ys), G(M * xs * ys)
    i0, ix, iy = G(I), G(I * xs), G(I * ys)
    S00 = g0
    S10, S01 = gx - xs * g0, gy - ys * g0
    S20 = gxx - 2 * xs * gx + xs * xs * g0
    S02 = gyy - 2 * ys * gy + ys * ys * g0
    S11 = gxy - xs * gy - ys * gx + xs * ys * g0
    b0, b1, b2 = i0, ix - xs * i0, iy - ys * i0
    # solve the 3x3 normal equations for the intercept (Cramer's rule)
    det = (S00 * (S20 * S02 - S11 * S11) - S10 * (S10 * S02 - S11 * S01) + S01 * (S10 * S11 - S20 * S01))
    num = (b0 * (S20 * S02 - S11 * S11) - S10 * (b1 * S02 - S11 * b2) + S01 * (b1 * S11 - S20 * b2))
    plain = i0 / np.maximum(g0, 1e-9)
    ok = (g0 > 0.05) & (np.abs(det) > 1e-9 * np.maximum(S00, 1e-9) ** 3 * sig ** 4)
    est = np.where(ok, num / np.where(ok, det, 1.0), plain)
    fill = float(img[m > 0].mean()) if (m > 0).any() else 1.0
    est = np.where(g0 > 1e-3, est, fill)
    # keep the estimate inside a sane band around the plain average (guards the extrapolation in the corners)
    est = np.clip(est, 0.3 * np.where(g0 > 1e-3, plain, fill), 3.0 * np.where(g0 > 1e-3, plain, fill))
    return cv2.resize(est.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _poly_terms(x, y, order):
    return np.stack([x ** i * y ** j for i in range(order + 1) for j in range(order + 1 - i)], axis=-1)


def polynomial_background(img: np.ndarray, valid: np.ndarray | None, order: int, max_side: int = 96) -> np.ndarray:
    """Robust (iteratively re-weighted) 2-D polynomial surface fitted on a reduced image."""
    h, w = img.shape
    f = max(1, int(math.ceil(max(h, w) / float(max_side))))
    ws, hs = max(8, w // f), max(8, h // f)
    small = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
    m = np.ones((h, w), np.float32) if valid is None else valid.astype(np.float32)
    msmall = cv2.resize(m, (ws, hs), interpolation=cv2.INTER_AREA) > 0.999
    ys, xs = np.mgrid[0:hs, 0:ws]
    xn = (xs + 0.5) / ws * 2 - 1
    yn = (ys + 0.5) / hs * 2 - 1
    A = _poly_terms(xn, yn, order).reshape(-1, (order + 1) * (order + 2) // 2)
    b = small.ravel()
    use = msmall.ravel().copy()
    if use.sum() < A.shape[1] * 3:
        use[:] = True
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(A[use], b[use], rcond=None)
        res = b - A @ coef
        mad = 1.4826 * np.median(np.abs(res[use] - np.median(res[use]))) + 1e-9
        new_use = msmall.ravel() & (np.abs(res) < 3.0 * mad)
        if new_use.sum() < A.shape[1] * 3 or np.array_equal(new_use, use):
            break
        use = new_use
    X, Y = np.meshgrid((np.arange(w) + 0.5) / w * 2 - 1, (np.arange(h) + 0.5) / h * 2 - 1)
    full = _poly_terms(X, Y, order).reshape(-1, A.shape[1]) @ coef
    return full.reshape(h, w).astype(np.float32)


def estimate_background(img: np.ndarray, valid: np.ndarray | None, cfg: CameraSyncConfig) -> np.ndarray:
    if cfg.background_model == "polynomial":
        return polynomial_background(img, valid, cfg.background_poly_order)
    k = auto_background_kernel(img.shape, cfg.background_kernel_scale, cfg.background_kernel)
    return gaussian_background(img, valid, sigma=0.15 * k)      # sigma that cv2 would derive from ksize=k


def homogenise_divide(img: np.ndarray, valid: np.ndarray | None, cfg: CameraSyncConfig):
    """Divides the image by its estimated illumination shape (unit mean => brightness preserved)."""
    bg = estimate_background(img, valid, cfg)
    sel = valid if valid is not None else np.ones(img.shape, bool)
    mean_bg = float(bg[sel].mean()) if sel.any() else float(bg.mean())
    if mean_bg <= 1e-6:
        return img, bg
    shape = np.clip(bg / mean_bg, 0.2, 5.0)
    out = img / shape
    if valid is not None:
        out = np.where(valid, out, 0.0)
    return out.astype(np.float32), bg


def uniformise_background_legacy(img: np.ndarray, cfg: CameraSyncConfig) -> np.ndarray:
    """The script's method: subtract a gaussian blur, shift the minimum to 0, stretch the max to 65535."""
    k = auto_background_kernel(img.shape, cfg.background_kernel_scale, cfg.background_kernel)
    bg = cv2.GaussianBlur(img.astype(np.float32), (k, k), 0)
    out = img - bg
    out -= out.min()
    if out.max() > 0:
        out = out / out.max() * MAX16
    return np.clip(out, 0, MAX16).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# denoising (float data in the 16-bit range; no 8-bit round trip)
# ----------------------------------------------------------------------------------------------
def _soft_imports():
    try:
        from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle
        sk = (denoise_nl_means, estimate_sigma, denoise_tv_chambolle)
    except Exception:
        sk = None
    try:
        import bm3d
    except Exception:
        bm3d = None
    return sk, bm3d


def estimate_noise_sigma(x: np.ndarray) -> float:
    """Immerkaer's fast noise estimate (no PyWavelets dependency, unlike skimage.estimate_sigma)."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(np.asarray(x, np.float32), -1, k)[1:-1, 1:-1]
    h, w = x.shape[:2]
    return float(max(1e-5, math.sqrt(math.pi / 2.0) / (6.0 * max(1, (w - 2) * (h - 2))) * np.abs(conv).sum()))


def denoise_spatial(img: np.ndarray, cfg: CameraSyncConfig) -> np.ndarray:
    kind = cfg.denoiser
    if kind in ("none", "temporal_median"):
        return img
    sk, bm3d = _soft_imports()
    x = np.clip(img / MAX16, 0, 1).astype(np.float32)
    if kind == "bilateral":
        y = cv2.bilateralFilter(x, d=cfg.bilateral_diameter, sigmaColor=cfg.bilateral_sigma_color / 255.0,
                                sigmaSpace=cfg.bilateral_sigma_space)
    elif kind == "tv" and sk is not None:
        y = sk[2](x, weight=cfg.tv_weight)
    elif kind == "tv":
        y = cv2.bilateralFilter(x, d=cfg.bilateral_diameter, sigmaColor=cfg.bilateral_sigma_color / 255.0,
                                sigmaSpace=cfg.bilateral_sigma_space)
    elif kind == "bm3d" and bm3d is not None:
        sigma = cfg.bm3d_sigma if (cfg.bm3d_sigma and cfg.bm3d_sigma > 0) else float(np.std(x))
        y = bm3d.bm3d(x, sigma)
    else:    # nlmeans (also the fallback of bm3d when the package is missing)
        if sk is not None:
            sig = estimate_noise_sigma(x)
            y = sk[0](x, patch_size=cfg.nlmeans_patch_size, patch_distance=cfg.nlmeans_patch_distance,
                      h=cfg.nlmeans_h_factor * sig, sigma=sig, fast_mode=True)
        else:
            x8 = np.clip(x * 255, 0, 255).astype(np.uint8)
            y = cv2.fastNlMeansDenoising(x8, None, h=10, templateWindowSize=7, searchWindowSize=21).astype(np.float32) / 255.0
    return np.clip(np.asarray(y, np.float32) * MAX16, 0, MAX16)


def despeckle_image(img16: np.ndarray, cfg: CameraSyncConfig) -> np.ndarray:
    if not cfg.despeckle or cfg.despeckle_size <= 1:
        return img16
    k = cfg.despeckle_size + (cfg.despeckle_size % 2 == 0)
    u = np.clip(img16, 0, MAX16).astype(np.uint16)
    ker = np.ones((k, k), np.uint8)
    if cfg.despeckle_mode == "median":
        if k in (3, 5):
            return cv2.medianBlur(u, k)
        from scipy.ndimage import median_filter
        return median_filter(u, size=k)
    opened = cv2.morphologyEx(u, cv2.MORPH_OPEN, ker)
    if cfg.despeckle_mode == "morphology":
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, ker)
    if k in (3, 5):
        return cv2.medianBlur(opened, k)
    from scipy.ndimage import median_filter
    return median_filter(opened, size=k)


# ----------------------------------------------------------------------------------------------
# intensity normalisation (adaptive inter + intra camera smoothing of per-frame scales)
# ----------------------------------------------------------------------------------------------
def smooth_scales_gaussian(scales, sigma):
    scales = np.asarray(scales, dtype=np.float64)
    if scales.size == 0:
        return scales
    radius = max(1, int(3.0 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return np.convolve(np.pad(scales, (radius, radius), mode="edge"), kernel, mode="valid")


def smooth_scales_ema(scales, alpha):
    out = np.zeros(len(scales), dtype=np.float64)
    if len(scales):
        out[0] = scales[0]
        for i in range(1, len(scales)):
            out[i] = alpha * scales[i] + (1.0 - alpha) * out[i - 1]
    return out


def smooth_scales_moving_average(scales, window):
    scales = np.asarray(scales, dtype=np.float64)
    if len(scales) == 0:
        return scales
    radius = window // 2
    return np.convolve(np.pad(scales, (radius, radius), mode="edge"), np.ones(window) / float(window), mode="valid")


def compute_intensity_scales(medians, cam_of_entry, cfg: CameraSyncConfig):
    """`medians[k]` is the median of time-ordered entry k, `cam_of_entry[k]` its camera.
    Returns (scales, report) with the same algorithm as the script (global + per-camera smoothing blended by
    the local temporal activity)."""
    med = np.asarray(medians, dtype=np.float64)
    cam_of = np.asarray(cam_of_entry)
    if med.size == 0:
        raise CameraSyncError("No frame to normalise.")
    if cfg.use_global_intensity_norm:
        ref = {"median": float(np.median(med)), "mean": float(np.mean(med)), "first_frame": float(med[0])}[cfg.global_intensity_mode]
    else:
        ref = float(np.median(med))
    if ref < 1e-6:
        raise CameraSyncError("Reference intensity is zero or too small; cannot normalise.")
    raw = np.where(med > 1e-6, ref / np.maximum(med, 1e-6), 1.0)
    idx_cam = [np.flatnonzero(cam_of == c) for c in range(cfg.n_cameras)]
    raw_cam = [raw[i] for i in idx_cam]
    med_cam = [med[i] for i in idx_cam]
    n_total = raw.size
    sigma_global = None
    if cfg.smoothing_method == "gaussian":
        user = cfg.gaussian_sigma if (cfg.gaussian_sigma and cfg.gaussian_sigma > 0) else None
        sigma_global = user or max(1.0, cfg.gaussian_sigma_auto_factor * n_total)
        sm_global = smooth_scales_gaussian(raw, sigma_global)
        sm_cam = [smooth_scales_gaussian(r, user or max(1.0, cfg.gaussian_sigma_auto_factor * r.size)) for r in raw_cam]
    elif cfg.smoothing_method == "ema":
        sm_global = smooth_scales_ema(raw, 0.12)
        sm_cam = [smooth_scales_ema(r, 0.12) for r in raw_cam]
    else:
        sm_global = smooth_scales_moving_average(raw, 5)
        sm_cam = [smooth_scales_moving_average(r, 5) for r in raw_cam]

    def activity(series):
        if len(series) < 2:
            return np.zeros(len(series))
        g = np.abs(np.gradient(series))
        thr = np.percentile(g, cfg.gradient_threshold_percentile)
        act = np.clip(g / (thr + 1e-12), 0.0, 10.0)
        return np.clip((act - 1.0) / 9.0, 0.0, 1.0)

    act_global = activity(med)
    act_cam_full = np.zeros(n_total)
    sm_cam_full = np.zeros(n_total)
    for c in range(cfg.n_cameras):
        act_cam_full[idx_cam[c]] = activity(med_cam[c])
        sm_cam_full[idx_cam[c]] = sm_cam[c]
    combined = np.clip(np.maximum(act_global, act_cam_full) * cfg.blend_strength, 0.0, 1.0)
    final = (1.0 - combined) * sm_global + combined * raw
    w_cam = np.clip(act_cam_full * cfg.blend_strength, 0.0, 1.0)
    final = (1.0 - w_cam) * final + w_cam * sm_cam_full
    final = np.clip(final, cfg.min_scale, cfg.max_scale)
    eff = float(np.median(final * med))
    if eff > 1e-6:
        final = final * (ref / eff)
    report = {"enabled": True, "method": cfg.smoothing_method,
              "gaussian_sigma_used_global": float(sigma_global) if sigma_global else None,
              "reference_value": ref, "min_scale_clamp": cfg.min_scale, "max_scale_clamp": cfg.max_scale,
              "gradient_threshold_percentile": cfg.gradient_threshold_percentile, "blend_strength": cfg.blend_strength,
              "raw_scales_summary": {"min": float(raw.min()), "max": float(raw.max()), "median": float(np.median(raw))},
              "final_scales_summary": {"min": float(final.min()), "max": float(final.max()), "median": float(np.median(final))}}
    return final, report


# ----------------------------------------------------------------------------------------------
# photometric equalisation by a pairwise log-gain solver (N_cameras_images_processing.py)
# ----------------------------------------------------------------------------------------------
def frame_thumbnail(img: np.ndarray, side: int) -> np.ndarray:
    """Area-averaged thumbnail (longest side `side`), float32. Pairs of frames are compared on these."""
    h, w = img.shape[:2]
    k = float(side) / max(h, w)
    if k >= 1.0:
        return np.ascontiguousarray(img, dtype=np.float32)
    return cv2.resize(np.ascontiguousarray(img, dtype=np.float32), (max(1, int(round(w * k))), max(1, int(round(h * k)))),
                      interpolation=cv2.INTER_AREA)


def _pair_list(m: int, max_pairs: int, n_cameras: int) -> tuple[list, str]:
    """All pairs (a < b) when affordable, otherwise a sparse connected graph: every frame with its W following frames
    in time order (which are the other cameras at the same time and the next frames) plus a few evenly spaced anchors."""
    if m * (m - 1) // 2 <= max_pairs:
        return [(a, b) for a in range(m) for b in range(a + 1, m)], "all_pairs"
    n_anchor = 8
    window = max(2 * n_cameras + 2, max_pairs // max(m, 1) - n_anchor)
    anchors = sorted(set(np.linspace(0, m - 1, n_anchor).round().astype(int).tolist()))
    pairs = set()
    for a in range(m):
        for b in range(a + 1, min(m, a + 1 + window)):
            pairs.add((a, b))
        for an in anchors:
            if an != a:
                pairs.add((min(a, an), max(a, an)))
    return sorted(pairs), f"sparse_window_{window}_anchors_{len(anchors)}"


def solve_pairwise_gains(thumbs: list, valids: list, cfg: CameraSyncConfig, progress=None, cancel_check=None):
    """Per-frame multiplicative gains from pairwise log offsets (the script's estimate_frame_gains_no_smoothing).

    For two frames i, j the offset is d_ij = median over the common pixels of log(I_i + 1) - log(I_j + 1); the common
    pixels are those above the `equalize_low_pct` percentile in both (and valid, and not saturated). The sparse system
    o_j - o_i = d_ij, plus a weak regularisation o = 0 that fixes the global level, is solved by least squares; the gain
    is exp(o), clamped to [equalize_gain_min, equalize_gain_max].  `thumbs[k]` is the thumbnail of time-ordered entry k
    and `valids[k]` its validity mask (same shape).  Returns (gains, report)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr

    m = len(thumbs)
    if m == 0:
        raise CameraSyncError("No frame to equalise.")
    shape = thumbs[0].shape
    n_px = shape[0] * shape[1]
    logs = np.empty((m, n_px), np.float32)
    ok_base = np.empty((m, n_px), bool)
    for k in range(m):
        t = thumbs[k].reshape(-1)
        v = valids[k].reshape(-1)
        thr = np.percentile(t[v], cfg.equalize_low_pct) if v.any() else 0.0
        ok_base[k] = v & (t > thr) & (t < 0.98 * MAX16)
        logs[k] = np.log(t + 1.0)
    pairs, mode = _pair_list(m, cfg.equalize_max_pairs, cfg.n_cameras)
    min_px = max(16, int(cfg.equalize_min_overlap_frac * n_px))
    rows, cols, vals, rhs = [], [], [], []
    eq = 0
    for q, (a, b) in enumerate(pairs):
        if q % 512 == 0:
            check_cancelled(cancel_check)
            if progress is not None:
                progress(f"Equalising intensities: pair {q + 1}/{len(pairs)}", q / float(len(pairs)))
        mk = ok_base[a] & ok_base[b]
        if int(np.count_nonzero(mk)) < min_px:
            continue
        d = float(np.median(logs[a][mk] - logs[b][mk]))
        rows += [eq, eq]
        cols += [a, b]
        vals += [-1.0, 1.0]
        rhs.append(d)
        eq += 1
    n_eq = eq
    report = {"method": "pairwise_log_gain", "n_frames": m, "pair_mode": mode, "n_pairs_tested": len(pairs),
              "n_equations": n_eq, "thumbnail_px": [int(shape[1]), int(shape[0])],
              "gain_limits": [cfg.equalize_gain_min, cfg.equalize_gain_max]}
    if n_eq == 0:
        report["warning"] = "no pair had enough common pixels: all gains are 1"
        return np.ones(m), report
    for i in range(m):
        rows.append(n_eq + i); cols.append(i); vals.append(float(cfg.equalize_reg)); rhs.append(0.0)
    A = coo_matrix((vals, (rows, cols)), shape=(n_eq + m, m)).tocsr()
    b_vec = np.asarray(rhs, np.float64)
    sol = lsqr(A, b_vec, atol=1e-10, btol=1e-10, iter_lim=3000)[0]
    gains = np.exp(sol)
    clamped = (gains < cfg.equalize_gain_min) | (gains > cfg.equalize_gain_max)
    gains = np.clip(gains, cfg.equalize_gain_min, cfg.equalize_gain_max)
    o = np.log(gains)
    pa, pb = np.asarray(cols[0:2 * n_eq:2]), np.asarray(cols[1:2 * n_eq:2])
    d_arr = np.asarray(rhs[:n_eq])
    report.update({
        "n_clamped_frames": int(clamped.sum()),
        "gain_min": float(gains.min()), "gain_max": float(gains.max()), "gain_median": float(np.median(gains)),
        "rms_log_mismatch_before": float(np.sqrt(np.mean(d_arr ** 2))),
        "rms_log_mismatch_after": float(np.sqrt(np.mean((o[pb] - o[pa] - d_arr) ** 2))),
    })
    if clamped.any():
        report["warning"] = (f"{int(clamped.sum())} frame(s) reached the gain limits [{cfg.equalize_gain_min}, {cfg.equalize_gain_max}]: "
                             "their intensity is not fully equalised (widen the limits if this is a real brightness change)")
    return gains, report


# ----------------------------------------------------------------------------------------------
# optional tone matching to a reference frame (overlap-masked, slope-limited LUT)
# ----------------------------------------------------------------------------------------------
def _clipped_fraction(img16: np.ndarray) -> float:
    return float(np.mean(img16 >= np.uint16(MAX16)))


def soft_knee_compress_uint16(img16: np.ndarray, knee_start_pct: float, strength: float) -> np.ndarray:
    x = img16.astype(np.float32) / (MAX16 + 1e-9)
    p = float(np.percentile(x, knee_start_pct))
    above = x > p
    if p < 1.0 and np.any(above):
        t = (x[above] - p) / (1.0 - p)
        x[above] = p + (1.0 - p) * t ** (1.0 / (1.0 + strength))
    return np.clip(np.rint(x * MAX16), 0, MAX16).astype(np.uint16)


def coarse_lut_from_patches(src: np.ndarray, ref: np.ndarray, n_bins: int) -> np.ndarray:
    ident = np.linspace(0.0, MAX16, n_bins, dtype=np.float32)
    if src.size == 0 or ref.size == 0:
        return ident
    hs, _ = np.histogram(src.ravel(), bins=n_bins, range=(0, MAX16))
    hr, _ = np.histogram(ref.ravel(), bins=n_bins, range=(0, MAX16))
    if hs.sum() == 0 or hr.sum() == 0:
        return ident
    cs_ = np.cumsum(hs) / float(hs.sum())
    cr_ = np.cumsum(hr) / float(hr.sum())
    j = np.minimum(np.searchsorted(cr_, cs_, side="left"), n_bins - 1)      # first ref bin whose CDF reaches the source CDF
    return (j / float(n_bins - 1) * MAX16).astype(np.float32)


def slope_limited_lut(lut: np.ndarray, max_slope: float) -> np.ndarray:
    n = len(lut)
    ln = lut.astype(np.float64) / MAX16
    md = max_slope / (n - 1)
    for i in range(1, n):                                  # forward pass
        d = ln[i] - ln[i - 1]
        if d > md:
            ln[i] = ln[i - 1] + md
        elif d < -md:
            ln[i] = ln[i - 1] - md
    for i in range(n - 2, -1, -1):                         # backward pass
        d = ln[i + 1] - ln[i]
        if d > md:
            ln[i] = ln[i + 1] - md
        elif d < -md:
            ln[i] = ln[i + 1] + md
    return np.clip(np.rint(ln * MAX16), 0, MAX16).astype(np.uint16)


def apply_lut_uint16(img16: np.ndarray, lut: np.ndarray) -> np.ndarray:
    idx = np.floor(img16.astype(np.float64) / MAX16 * (len(lut) - 1)).astype(np.int64)
    return lut[idx].astype(np.uint16)


def percentile_linear_rescale(src16: np.ndarray, ref16: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    s_lo, s_hi = np.percentile(src16, [lo, hi])
    r_lo, r_hi = np.percentile(ref16, [lo, hi])
    g = (r_hi - r_lo) / ((s_hi - s_lo) or 1.0)
    return np.clip(np.rint(src16.astype(np.float32) * g + (r_lo - g * s_lo)), 0, MAX16).astype(np.uint16)


def match_tone_to_reference(img16: np.ndarray, ref16: np.ndarray, cfg: CameraSyncConfig):
    """Maps the grey levels of `img16` onto those of `ref16` (same shape). Returns (uint16 image, path used).

    Port of match_with_overlap_and_slope_limit (without the skimage shortcut): coarse LUT from the pixels that are above the
    1st percentile in both images, Gaussian-smoothed and slope-limited; if too many pixels clip, retry after a soft-knee
    compression, then a percentile linear rescale, then a median gain."""
    tgt = np.clip(np.rint(img16), 0, MAX16).astype(np.uint16)
    ref = np.clip(np.rint(ref16), 0, MAX16).astype(np.uint16)
    mask = (tgt > np.percentile(tgt, 1.0)) & (ref > np.percentile(ref, 1.0))
    if np.count_nonzero(mask) >= cfg.min_overlap_pixels:
        src_p, ref_p = tgt[mask], ref[mask]
    else:
        src_p, ref_p = tgt.ravel(), ref.ravel()

    def build(src_patch):
        lut = coarse_lut_from_patches(src_patch, ref_p, cfg.lut_bins)
        lut = cv2.GaussianBlur(lut.astype(np.float32).reshape(1, -1), (0, 0), sigmaX=cfg.lut_smooth_sigma).reshape(-1)
        return slope_limited_lut(lut, cfg.lut_max_slope)

    mapped = apply_lut_uint16(tgt, build(src_p))
    if _clipped_fraction(mapped) <= cfg.max_clip_frac:
        return mapped, "lut"
    comp = soft_knee_compress_uint16(tgt, cfg.soft_knee_pct, cfg.soft_knee_strength)
    src2 = comp[mask] if np.count_nonzero(mask) >= cfg.min_overlap_pixels else comp.ravel()
    mapped2 = apply_lut_uint16(comp, build(src2))
    if _clipped_fraction(mapped2) <= cfg.max_clip_frac:
        return mapped2, "lut_soft_knee"
    fb = percentile_linear_rescale(tgt, ref)
    if _clipped_fraction(fb) <= cfg.max_clip_frac:
        return fb, "percentile_rescale"
    sm, rm = float(np.median(tgt)), float(np.median(ref))
    if sm > 0:
        g = float(np.clip(rm / (sm + 1e-9), cfg.equalize_gain_min, cfg.equalize_gain_max))
        sc = np.clip(np.rint(tgt.astype(np.float32) * g), 0, MAX16).astype(np.uint16)
        if _clipped_fraction(sc) <= cfg.max_clip_frac:
            return sc, "median_gain"
    return fb, "percentile_rescale_clipped"


# ----------------------------------------------------------------------------------------------
# frame order / time stamps
# ----------------------------------------------------------------------------------------------
@dataclass
class FrameEntry:
    order: int          # position in the time-ordered sequence (output index j)
    cam: int
    frame_idx: int      # index among the kept (post-trim) frames of this camera
    orig_idx: int       # index in the original folder listing
    t_ns: float | None
    source: str = ""


def build_sequence(cfg: CameraSyncConfig, orig_indices: list, sources: list | None = None) -> list:
    entries = []
    for cam in range(cfg.n_cameras):
        for i, o in enumerate(orig_indices[cam]):
            t = None
            if cfg.use_timestamps:
                t = float(cfg.timestamp_ns) + o * float(cfg.timestamp_step) + float(cfg.camera_offsets_ns[cam])
            src = sources[cam][i].name if sources else ""
            entries.append(FrameEntry(0, cam, i, int(o), t, src))
    entries.sort(key=lambda e: ((e.t_ns if e.t_ns is not None else 0.0) if cfg.use_timestamps else 0.0,
                                e.frame_idx, e.cam))
    for j, e in enumerate(entries):
        e.order = j
    return entries


# ----------------------------------------------------------------------------------------------
# stage 1: prepare calibration + previews
# ----------------------------------------------------------------------------------------------
@dataclass
class PreparedData:
    cfg: CameraSyncConfig
    calibrations: list
    files: list                  # [cam] -> kept (post-trim) files
    orig_indices: list
    previews: list               # calibrated reference image per camera (float32, full frame)
    image_shape: tuple           # (h, w)
    diagnostics: dict = field(default_factory=dict)


def _ensure_dirs(cfg: CameraSyncConfig, names):
    for n in names:
        if n in ("calibrated", "cropped", "work_cropped", "work_hom"):
            for cam in range(cfg.n_cameras):
                os.makedirs(cfg.folder(n, cam), exist_ok=True)
        else:
            os.makedirs(cfg.folder(n), exist_ok=True)


def _write_png(path, img):
    try:
        cv2.imwrite(path, display_uint8(img))
    except Exception:
        pass


def _export_calibration(cfg: CameraSyncConfig, cam: int, dark: np.ndarray, gain: np.ndarray):
    """master dark (16-bit TIFF), gain map (visualisation TIFF + exact .npy), like the N-camera script's calibration/ folder."""
    try:
        d = cfg.folder("calibration")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(f"{d}/master_dark_cam{cam + 1}.tif", np.clip(np.rint(dark), 0, MAX16).astype(np.uint16))
        g = np.asarray(gain, np.float32)
        lo, hi = float(g.min()), float(g.max())
        vis = (g - lo) / max(hi - lo, 1e-9) * MAX16
        cv2.imwrite(f"{d}/master_flat_gain_vis_cam{cam + 1}.tif", np.clip(np.rint(vis), 0, MAX16).astype(np.uint16))
        np.save(f"{d}/master_flat_gain_cam{cam + 1}.npy", g)
        with open(f"{d}/master_flat_gain_cam{cam + 1}_range.json", "w") as fh:
            json.dump({"gain_min": lo, "gain_max": hi, "vis_scale": "(gain - gain_min) / (gain_max - gain_min) * 65535"}, fh, indent=2)
    except OSError:
        pass


def prepare_calibration(cfg: CameraSyncConfig, progress=None, cancel_check=None) -> PreparedData:
    def prog(msg, frac=None):
        if progress is not None:
            progress(msg, frac)

    cfg.validate()
    _ensure_dirs(cfg, ["check"])
    check = cfg.folder("check")

    # frame lists first (cheap, catches count mismatches before the heavy work)
    full = [list_image_files(cfg.input_folders[c]) for c in range(cfg.n_cameras)]
    counts = [len(f) for f in full]
    if min(counts) == 0:
        raise CameraSyncError(f"No BMP/TIFF/PNG files in an input folder (counts: {counts}).")
    if len(set(counts)) != 1:
        raise CameraSyncError(f"All cameras must have the same number of images before trimming. Counts: {counts}")
    total = counts[0]
    if cfg.skip_start + cfg.skip_end >= total:
        raise CameraSyncError(f"After trimming no frame is left (total={total}, skip_start={cfg.skip_start}, skip_end={cfg.skip_end}).")
    files = [f[cfg.skip_start: total - cfg.skip_end] for f in full]
    orig = [list(range(cfg.skip_start, total - cfg.skip_end)) for _ in range(cfg.n_cameras)]

    calibrations, diagnostics = [], {}
    for cam in range(cfg.n_cameras):
        check_cancelled(cancel_check)
        prog(f"Camera {cam + 1}: master dark", cam / (cfg.n_cameras * 2.0))
        dark, ddiag = build_master_dark(cfg.dark_folders[cam], cfg, cam)
        check_cancelled(cancel_check)
        prog(f"Camera {cam + 1}: master flat and gain map", (cam + 0.5) / (cfg.n_cameras * 2.0))
        flat, fdiag = build_master_flat(cfg.flat_folders[cam], cfg, cam)
        if dark.shape != flat.shape:
            raise CameraSyncError(f"Camera {cam + 1}: dark {dark.shape} and flat {flat.shape} images differ in size.")
        gain, fd, gdiag = build_gain_map(flat, dark, cfg)
        fdiag["gain_map"] = gdiag
        if "warning" in gdiag:
            fdiag["warning"] = (fdiag["warning"] + " | " if "warning" in fdiag else "") + gdiag["warning"]
        with open(f"{check}/dark_rejection_cam{cam + 1}.json", "w") as fh:
            json.dump(ddiag, fh, indent=2)
        with open(f"{check}/flat_rejection_cam{cam + 1}.json", "w") as fh:
            json.dump(fdiag, fh, indent=2)
        _write_png(f"{check}/master_dark_cam{cam + 1}.png", dark)
        _write_png(f"{check}/master_flat_cam{cam + 1}.png", flat)
        _write_png(f"{check}/gain_map_cam{cam + 1}.png", gain)
        _export_calibration(cfg, cam, dark, gain)
        cal = CameraCalibration(dark, flat, gain, max(float(np.std(dark)), 1.0), pedestal=float(cfg.calibration_pedestal),
                                dark_scaling=cfg.dark_scaling, dark_scale_limits=(cfg.dark_scale_min, cfg.dark_scale_max))
        cal.prepare_dark_scaling()
        diagnostics_hot = int(cal._hot.sum())
        ddiag["hot_pixels_used_for_scaling"] = diagnostics_hot
        if cfg.use_static_light:
            light = np.median(np.stack([read_image(f, cfg.input_bit_depth)
                                        for f in _sample_files(list_image_files(cfg.light_folders[cam]), 30)]), axis=0)
            lc = cal.calibrate(light.astype(np.float32))
            cal.static_pattern = (lc - float(lc.mean())).astype(np.float32)
        calibrations.append(cal)
        diagnostics[f"cam{cam + 1}"] = {"dark": ddiag, "flat": fdiag}

    # calibrated preview (reference frame for the interactive steps)
    previews = []
    shape = None
    for cam in range(cfg.n_cameras):
        check_cancelled(cancel_check)
        prog(f"Camera {cam + 1}: calibrated preview", 0.5 + cam / (cfg.n_cameras * 2.0))
        if cfg.align_use_first_image:
            sel = [files[cam][0]]
        else:
            sel = _sample_files(files[cam], 15)
        imgs = [calibrations[cam].calibrate(read_image(f, cfg.input_bit_depth)) for f in sel]
        prev = imgs[0] if len(imgs) == 1 else np.median(np.stack(imgs), axis=0).astype(np.float32)
        if shape is None:
            shape = prev.shape
        elif prev.shape != shape:
            raise CameraSyncError(f"Camera {cam + 1} images are {prev.shape}, camera 1 is {shape}: sizes must match.")
        previews.append(prev)
        _write_png(f"{check}/calibrated_preview_cam{cam + 1}.png", prev)
    prog("Calibration ready", 1.0)
    return PreparedData(cfg, calibrations, files, orig, previews, tuple(shape), diagnostics)


# ----------------------------------------------------------------------------------------------
# video
# ----------------------------------------------------------------------------------------------
def find_ffmpeg() -> str | None:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def create_16bit_video(tiff_folder, output_path, fps=24, cancel_check=None):
    """FFV1 / gray16le MKV from proc_%06d.tif. Returns (ok, message)."""
    ff = find_ffmpeg()
    if ff is None:
        return False, "ffmpeg not found in PATH: cannot create the 16-bit video."
    cmd = [ff, "-y", "-framerate", str(fps), "-i", os.path.join(tiff_folder, "proc_%06d.tif"),
           "-c:v", "ffv1", "-pix_fmt", "gray16le", str(output_path)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                check_cancelled(cancel_check)
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    if proc.returncode != 0:
        return False, "ffmpeg failed: " + err.decode("utf-8", errors="ignore")[-600:]
    return True, str(output_path)


def create_mp4_lossless(tiff_folder, output_path, fps=24, bit_depth=8, cancel_check=None):
    """Lossless H.264 (qp 0, High 4:4:4 Predictive, full range) MP4 from proc_%06d.tif.

    MP4 cannot hold FFV1 or 16-bit video, so the frames are first rescaled to `bit_depth` (8 or 10) bits -- that rescaling is
    the only loss; the encoding of the rescaled frames is exact. Frames are streamed to ffmpeg (no temporary files)."""
    ff = find_ffmpeg()
    if ff is None:
        return False, "ffmpeg not found in PATH: cannot create the MP4."
    files = sorted(Path(tiff_folder).glob("proc_*.tif"))
    if not files:
        return False, f"no proc_*.tif in {tiff_folder}"
    first = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        return False, f"cannot read {files[0]}"
    h, w = first.shape[:2]
    pix_in, pix_out = ("gray", "yuv444p") if bit_depth == 8 else ("gray10le", "yuv444p10le")
    cmd = [ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix_in, "-s", f"{w}x{h}", "-framerate", str(fps), "-i", "-",
           "-vf", f"scale=in_range=pc:out_range=pc,format={pix_out}", "-c:v", "libx264", "-qp", "0", "-preset", "medium",
           "-color_range", "pc", "-movflags", "+faststart", str(output_path)]
    import tempfile
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errf)
        try:
            for f in files:
                check_cancelled(cancel_check)
                img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float64)
                if bit_depth == 8:
                    data = np.clip(np.rint(img / 257.0), 0, 255).astype(np.uint8)
                else:
                    data = np.clip(np.rint(img / MAX16 * 1023.0), 0, 1023).astype("<u2")
                proc.stdin.write(data.tobytes())
            proc.stdin.close()
            proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        if proc.returncode != 0:
            errf.seek(0)
            return False, "ffmpeg failed: " + errf.read().decode("utf-8", errors="ignore")[-600:]
    return True, str(output_path)


def encode_videos(tiff_folder, base_path: str, cfg: "CameraSyncConfig", cancel_check=None) -> dict:
    """Encodes the selected formats of one image sequence. `base_path` has no extension. Returns {format: (ok, path or message)}."""
    out = {}
    for fmt, flag in (("mkv", cfg.export_video_mkv), ("avi", cfg.export_video_avi), ("mp4", cfg.export_video_mp4)):
        if not flag:
            continue
        path = f"{base_path}.{fmt}"
        if fmt == "mp4":
            out[fmt] = create_mp4_lossless(tiff_folder, path, cfg.video_fps, cfg.mp4_bit_depth, cancel_check)
        else:
            out[fmt] = create_16bit_video(tiff_folder, path, cfg.video_fps, cancel_check)     # FFV1 / gray16le, container by extension
    return out


def video_summary(results: dict, cfg: "CameraSyncConfig") -> dict:
    """Report entry. `enabled` / `path` / `reason` describe the first format that worked (kept for the older consumers)."""
    ok = {k: v[1] for k, v in results.items() if v[0]}
    bad = {k: v[1] for k, v in results.items() if not v[0]}
    first = next(iter(ok.values()), None)
    return {"enabled": bool(ok), "path": first, "reason": None if ok else "; ".join(f"{k}: {m}" for k, m in bad.items()),
            "files": ok, "failed": bad, "fps": cfg.video_fps,
            "codecs": {"mkv": "ffv1 gray16le", "avi": "ffv1 gray16le",
                       "mp4": f"h264 lossless (qp 0), {cfg.mp4_bit_depth}-bit"}}


# ----------------------------------------------------------------------------------------------
# comparison images
# ----------------------------------------------------------------------------------------------
def save_comparison_row(tiles, out_path, tile_h=400):
    bgr = []
    h0, w0 = tiles[0][0].shape[:2]
    tile_w = max(1, int(round(w0 * tile_h / float(h0))))
    for img, label in tiles:
        disp = cv2.resize(display_uint8(img, 1, 99.5), (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        tile = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(tile, (0, 0), (tw + 6, th + 6), (0, 0, 0), -1)
        cv2.putText(tile, label, (3, th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        bgr.append(tile)
    cv2.imwrite(str(out_path), cv2.hconcat(bgr))


# ----------------------------------------------------------------------------------------------
# stage 3: the full processing
# ----------------------------------------------------------------------------------------------
def burn_text(img16: np.ndarray, text: str, font_scale: float = 0.6, margin_px: int = 8) -> np.ndarray:
    """Draws `text` (white on a black box) in the bottom-right corner of a uint16 image, without converting the image to
    8 bit (cv2.putText is 8-bit only on OpenCV 5): the text is rendered into an 8-bit mask that is composed into the 16-bit
    pixels (box -> 0, text -> 65535)."""
    out = np.ascontiguousarray(img16).astype(np.uint16, copy=True)
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, font_scale, 1)
    x = max(0, w - tw - margin_px)
    y = max(th + margin_px, h - margin_px)
    pad = 6
    box = np.zeros((h, w), np.uint8)
    cv2.rectangle(box, (max(0, x - pad), max(0, y - th - pad)), (min(w - 1, x + tw + pad), min(h - 1, y + base + pad)), 255, cv2.FILLED)
    txt = np.zeros((h, w), np.uint8)
    cv2.putText(txt, text, (x, y), font, font_scale, 255, 1, cv2.LINE_AA)
    out[box > 0] = 0
    m = txt > 0
    out[m] = txt[m].astype(np.uint16) * np.uint16(257)
    return out


def _write16(path, img):
    cv2.imwrite(str(path), np.clip(np.rint(img), 0, MAX16).astype(np.uint16))


def _homogenise(img: np.ndarray, valid, cfg: CameraSyncConfig):
    if cfg.background_mode == "divide":
        return homogenise_divide(img, valid, cfg)[0]
    if cfg.background_mode == "legacy":
        return uniformise_background_legacy(img, cfg)
    return img


# ----------------------------------------------------------------------------------------------
# contrast stretch (what "Brightness & Contrast > Auto" does in ImageJ)
# ----------------------------------------------------------------------------------------------
def image_histogram16(img16: np.ndarray) -> np.ndarray:
    """Exact 65536-bin histogram of a uint16 image."""
    return np.bincount(np.asarray(img16, dtype=np.uint16).ravel(), minlength=65536).astype(np.int64)


def stretch_limits_imagej_auto(hist: np.ndarray) -> tuple[float, float]:
    """ImageJ's auto display range, reproduced from ij.plugin.frame.ContrastAdjuster.autoAdjust.

    The data are rebinned in 256 bins between their min and max (as ImageJ does for 16-bit images); the lower
    (upper) limit is the first bin from the bottom (top) holding more than N/5000 pixels, bins holding more than
    N/10 pixels being ignored (a dominant background peak must not pin the limit)."""
    nz = np.flatnonzero(hist)
    if nz.size == 0:
        return 0.0, MAX16
    vmin, vmax = int(nz[0]), int(nz[-1])
    if vmax <= vmin:
        return float(vmin), float(vmin + 1)
    n_bins = 256
    scale = n_bins / float(vmax - vmin + 1)
    idx = np.minimum(((np.arange(vmin, vmax + 1) - vmin) * scale).astype(int), n_bins - 1)
    h = np.bincount(idx, weights=hist[vmin:vmax + 1], minlength=n_bins)
    total = float(h.sum())
    limit, threshold = total / 10.0, total / 5000.0
    h = np.where(h > limit, 0.0, h)
    ok = np.flatnonzero(h > threshold)
    if ok.size == 0:
        return float(vmin), float(vmax)
    bin_size = (vmax - vmin) / float(n_bins)
    lo, hi = vmin + ok[0] * bin_size, vmin + ok[-1] * bin_size
    if hi <= lo:
        return float(vmin), float(vmax)
    return float(lo), float(hi)


def stretch_limits_percentile(hist: np.ndarray, saturated_pct: float) -> tuple[float, float]:
    """Display range that saturates `saturated_pct` % of the pixels (half at each end, as ImageJ's 'Saturated pixels')."""
    total = float(hist.sum())
    if total <= 0:
        return 0.0, MAX16
    cdf = np.cumsum(hist) / total
    q = saturated_pct / 200.0
    lo = float(np.searchsorted(cdf, q, side="left")) if q > 0 else float(np.flatnonzero(hist)[0])
    hi = float(np.searchsorted(cdf, 1.0 - q, side="left")) if q > 0 else float(np.flatnonzero(hist)[-1])
    hi = min(hi, MAX16)
    if hi <= lo:
        nz = np.flatnonzero(hist)
        lo, hi = float(nz[0]), float(max(nz[-1], nz[0] + 1))
    return lo, hi


def stretch_limits(hist: np.ndarray, cfg: "CameraSyncConfig") -> tuple[float, float]:
    if cfg.stretch_mode == "manual":
        return float(cfg.stretch_min), float(cfg.stretch_max)
    if cfg.stretch_mode == "percentile":
        return stretch_limits_percentile(hist, cfg.stretch_saturated_pct)
    return stretch_limits_imagej_auto(hist)


def apply_stretch(img16: np.ndarray, lo: float, hi: float, gamma: float = 1.0, clahe: tuple | None = None) -> np.ndarray:
    """Linear window [lo, hi] -> [0, 65535] (clipped), optional gamma, optional CLAHE (clip limit, tiles)."""
    x = (np.asarray(img16, dtype=np.float32) - lo) / max(hi - lo, 1e-6)
    x = np.clip(x, 0.0, 1.0)
    if gamma != 1.0:
        x = x ** gamma
    out = np.rint(x * MAX16).astype(np.uint16)
    if clahe is not None:
        out = cv2.createCLAHE(clipLimit=float(clahe[0]), tileGridSize=(int(clahe[1]), int(clahe[1]))).apply(out)
    return out


def stretch_sequence(cfg: "CameraSyncConfig", names: list, cams: list, progress=None, cancel_check=None) -> dict:
    """Writes all_stretched/<name> for every processed frame `names[i]` (file in all_processed, camera `cams[i]`).

    One display window for the whole sequence by default: the relative brightness between frames and cameras is
    kept, so no flicker is introduced. The linear stretch clips the pixels outside the window (that is what
    "saturated pixels" means): this copy is meant for looking at the details, not for correlation."""
    def prog(msg, frac):
        if progress is not None:
            progress(msg, frac)

    src, dst = cfg.folder("processed"), cfg.folder("stretched")
    os.makedirs(dst, exist_ok=True)
    n = len(names)
    clahe = (cfg.stretch_clahe_clip, cfg.stretch_clahe_tiles) if cfg.stretch_clahe else None
    keys = [0] * n if cfg.stretch_scope == "global" else list(cams)
    limits: dict = {}
    if cfg.stretch_mode == "manual":
        limits = {k: (float(cfg.stretch_min), float(cfg.stretch_max)) for k in set(keys)}
    elif cfg.stretch_scope != "per_frame":
        hists: dict = {}
        for k in sorted(set(keys)):
            members = [i for i in range(n) if keys[i] == k]
            take = _sample_files(members, cfg.stretch_sample_frames)
            hist = np.zeros(65536, np.int64)
            for j, i in enumerate(take):
                check_cancelled(cancel_check)
                prog(f"Contrast stretch: histogram {j + 1}/{len(take)}", None)
                hist += image_histogram16(read_image(os.path.join(src, names[i]), 16).astype(np.uint16))
            hists[k] = hist
            limits[k] = stretch_limits(hist, cfg)
    clipped_lo = clipped_hi = 0.0
    used = {}
    for i, name in enumerate(names):
        check_cancelled(cancel_check)
        prog(f"Contrast stretch: frame {i + 1}/{n}", (i + 1) / float(max(n, 1)))
        img = read_image(os.path.join(src, name), 16).astype(np.uint16)
        lo, hi = limits[keys[i]] if keys[i] in limits else stretch_limits(image_histogram16(img), cfg)
        used[i] = (lo, hi)
        clipped_lo += float((img <= lo).mean())
        clipped_hi += float((img >= hi).mean())
        cv2.imwrite(os.path.join(dst, name), apply_stretch(img, lo, hi, cfg.stretch_gamma, clahe))
    rep = {"mode": cfg.stretch_mode, "scope": cfg.stretch_scope, "gamma": cfg.stretch_gamma, "clahe": bool(cfg.stretch_clahe),
           "saturated_pct_requested": cfg.stretch_saturated_pct if cfg.stretch_mode == "percentile" else None,
           "mean_fraction_clipped_low": clipped_lo / max(n, 1), "mean_fraction_clipped_high": clipped_hi / max(n, 1),
           "n_frames": n, "folder": dst}
    if cfg.stretch_scope == "global":
        rep["window"] = list(used[0]) if n else None
    elif cfg.stretch_scope == "per_camera":
        rep["window_per_camera"] = {f"cam{k + 1}": list(v) for k, v in sorted(limits.items())}
    else:
        los, his = [u[0] for u in used.values()], [u[1] for u in used.values()]
        rep["window_range"] = {"min": [float(min(los)), float(max(los))], "max": [float(min(his)), float(max(his))]} if n else None
    return rep


def video_stem(video_filename: str) -> str:
    return os.path.splitext(video_filename)[0] or "processed"


def stretched_video_name(video_filename: str) -> str:
    return f"{video_stem(video_filename)}_stretched.mkv"


def restretch_existing(cfg: "CameraSyncConfig", progress=None, cancel_check=None) -> dict:
    """Re-runs only the contrast stretch on an existing all_processed folder (tune the window without reprocessing)."""
    cfg.validate(check_folders=False)
    if cfg.stretch_mode == "none":
        raise CameraSyncError("Choose a stretch mode other than 'none'.")
    path = f"{cfg.master_output}/timestamps.csv"
    if not os.path.isfile(path):
        raise CameraSyncError("timestamps.csv not found: run 'Process and export' first.")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    names = [r["processed_file"] for r in rows]
    cams = [int(r["camera"]) - 1 for r in rows]
    missing = [nm for nm in names if not os.path.isfile(os.path.join(cfg.folder("processed"), nm))]
    if missing:
        raise CameraSyncError(f"{len(missing)} processed frame(s) are missing in all_processed (e.g. {missing[0]}); "
                              "the processed TIFFs must be exported to stretch them afterwards.")
    rep = stretch_sequence(cfg, names, cams, progress, cancel_check)
    if cfg.export_video:
        os.makedirs(cfg.folder("video"), exist_ok=True)
        rep["video"] = video_summary(encode_videos(cfg.folder("stretched"), os.path.join(cfg.folder("video"),
                                     f"{video_stem(cfg.video_filename)}_stretched"), cfg, cancel_check), cfg)
    try:
        with open(cfg.report_path) as fh:
            full = json.load(fh)
        full["contrast_stretch"] = rep
        with open(cfg.report_path, "w") as fh:
            json.dump(full, fh, indent=2)
    except (OSError, ValueError):
        pass
    return rep


def process_sequences(prep: PreparedData, alignment: AlignmentResult, crop_bbox, rois=None, progress=None,
                      cancel_check=None, crop_reason: str | None = None) -> dict:
    """Runs everything after the interactive steps. Returns the report dict (also written as JSON)."""
    cfg = prep.cfg

    def prog(msg, frac=None):
        if progress is not None:
            progress(msg, frac)

    cfg.validate()
    h, w = prep.image_shape
    size = (w, h)
    names = ["check", "processed"] + (["calibrated"] if cfg.export_calibrated else []) + \
            (["cropped"] if cfg.export_cropped else []) + ["work_cropped"]
    do_stretch = cfg.stretch_mode != "none"
    if cfg.export_video:
        names.append("video")
    if cfg.burn_timestamp and cfg.export_no_timestamp_copy:
        names.append("processed_clean")
    if do_stretch:
        names.append("stretched")
    if cfg.save_comparisons:
        names.append("comparison")
    if cfg.export_combined_cropped:
        names.append("combined")
    if cfg.denoiser == "temporal_median":
        names.append("work_hom")
    _ensure_dirs(cfg, names)

    entries = build_sequence(cfg, prep.orig_indices, prep.files)
    n_per_cam = len(prep.files[0])
    total_entries = len(entries)

    report = {
        "timestamp": str(datetime.datetime.now()), "n_cameras": len(alignment.angles_deg), "reference_camera": alignment.reference_camera,
        "rotation": {f"cam{c + 1}": a for c, a in enumerate(alignment.angles_deg)},
        "translation": {f"cam{c + 1}": {"dx": t[0], "dy": t[1]} for c, t in enumerate(alignment.translations_px)},
        "alignment_method": alignment.method,
        "notes": "rotation then translation, composed into a single warp per frame",
        "config": cfg.to_dict(), "improvements": IMPROVEMENTS,
        "n_frames_per_camera": n_per_cam,
    }
    if crop_bbox is not None:
        report["autocrop"] = {"enabled": True, "intersection_bbox": dict(zip("xywh", map(int, crop_bbox))),
                              "margin_px": cfg.autocrop_margin_px,
                              "selected_rois": [dict(cam=i + 1, **dict(zip("xywh", map(int, r)))) for i, r in enumerate(rois or [])]}
    else:
        report["autocrop"] = {"enabled": False, "reason": crop_reason or "disabled"}

    mats = [alignment.matrix(c) for c in range(cfg.n_cameras)]
    valid_full = [warp_valid_mask((h, w), mats[c], size, cfg.align_interpolation) if not is_identity(mats[c])
                  else np.ones((h, w), bool) for c in range(cfg.n_cameras)]
    if crop_bbox is not None:
        x, y, cw, ch = crop_bbox
        sl = (slice(y, y + ch), slice(x, x + cw))
    else:
        sl = (slice(0, h), slice(0, w))
    valid_crop = [v[sl] for v in valid_full]
    out_shape = valid_crop[0].shape
    with open(cfg.report_path, "w") as fh:        # partial report, completed at the end
        json.dump(report, fh, indent=2)

    # ---------------- pass 1: calibrate -> align -> crop (streamed), medians --------------------
    medians = np.zeros(total_entries)
    thumbs = [None] * total_entries
    order_of = {(e.cam, e.frame_idx): e for e in entries}
    for cam in range(cfg.n_cameras):
        cal = prep.calibrations[cam]
        stats_neg = []
        neg_count = np.zeros((h, w), np.uint32)
        dark_scales = []
        for i, f in enumerate(prep.files[cam]):
            check_cancelled(cancel_check)
            prog(f"Pass 1/3: calibrate + align + crop, camera {cam + 1}, frame {i + 1}/{n_per_cam}",
                 (cam * n_per_cam + i) / float(cfg.n_cameras * n_per_cam) * 0.4)
            e = order_of[(cam, i)]
            st = {}
            calibrated = cal.calibrate(read_image(f, cfg.input_bit_depth), st)
            if st.get("negative_fraction", 0) > 0.001:
                stats_neg.append((e.orig_idx, st["negative_fraction"]))
            if "neg_mask" in st:
                neg_count += st["neg_mask"]
            dark_scales.append(st.get("dark_scale", 1.0))
            if cfg.export_calibrated:
                _write16(f"{cfg.folder('calibrated', cam)}/calib_{e.orig_idx:06d}.tif", calibrated)
            aligned = calibrated if is_identity(mats[cam]) else warp_image(calibrated, mats[cam], size, cfg.align_interpolation)
            cropped = aligned[sl]
            _write16(f"{cfg.folder('work_cropped', cam)}/crop_{e.orig_idx:06d}.tif", cropped)
            if cfg.export_cropped:
                _write16(f"{cfg.folder('cropped', cam)}/crop_{e.orig_idx:06d}.tif", cropped)
            vm = valid_crop[cam]
            medians[e.order] = float(np.median(cropped[vm])) if vm.any() else float(np.median(cropped))
            if cfg.intensity_method == "pairwise":
                thumbs[e.order] = frame_thumbnail(cropped, cfg.equalize_thumb_side)
        if cfg.dark_scaling == "per_frame":
            ds = np.asarray(dark_scales)
            used = cal._hot_den > 0
            report.setdefault("dark_scaling", {})[f"cam{cam + 1}"] = {
                "applied": bool(used), "median": float(np.median(ds)), "min": float(ds.min()), "max": float(ds.max()),
                "limits": [cfg.dark_scale_min, cfg.dark_scale_max]}
            if used and abs(float(np.median(ds)) - 1.0) > 0.05:
                report.setdefault("warnings", []).append(
                    f"Camera {cam + 1}: the dark current of the frames is x{float(np.median(ds)):.2f} that of the master dark "
                    f"(range {ds.min():.2f}-{ds.max():.2f}): the darks were recorded in other conditions (exposure, temperature, time). "
                    "The dark was scaled frame by frame; record the darks right before or after the sequence if you can.")
        nfr = float(max(n_per_cam, 1))
        freq = neg_count / nfr                       # fraction of the frames in which each sensor pixel was below the master dark
        bands = [float(b.mean()) for b in np.array_split(freq, 8, axis=1)]      # 8 vertical bands, left -> right
        report.setdefault("calibration_negative_pixels", {})[f"cam{cam + 1}"] = {
            "mean_fraction": float(freq.mean()), "fraction_per_vertical_band_left_to_right": bands,
            "frames_with_more_than_0.1pct": len(stats_neg), "pedestal_adu": cfg.calibration_pedestal}
        if freq.max() > 0:
            _write_png(f"{cfg.folder('check')}/negative_pixels_cam{cam + 1}.png", (freq * MAX16).astype(np.float32))
        if stats_neg or max(bands) > 0.003:          # many frames, or one region of the sensor (e.g. an edge) concentrates them
            where = int(np.argmax(bands))
            report.setdefault("warnings", []).append(
                f"Camera {cam + 1}: pixels were below the master dark (>0.1% of the image in {len(stats_neg)} frames; mean {100 * freq.mean():.2f}%, "
                f"worst in vertical band {where + 1}/8 with {100 * bands[where]:.1f}%; map: alignment_check/negative_pixels_cam{cam + 1}.png). "
                + ("The noise of those pixels is kept thanks to the calibration pedestal." if cfg.calibration_pedestal > 0 else
                   "Their value was replaced by the noise floor (black pixels): set a calibration pedestal, check the dark frames "
                   "(exposure / temperature) or limit the flat-field amplification."))

    # ---------------- intensity normalisation ------------------------------------------------------
    prog("Intensity equalisation (inter + intra camera)", 0.42)
    cams_in_order = [e.cam for e in entries]
    if cfg.intensity_method == "pairwise":
        valid_th = [frame_thumbnail(valid_crop[c].astype(np.float32), cfg.equalize_thumb_side) > 0.99 for c in range(cfg.n_cameras)]
        scales, eq_report = solve_pairwise_gains(thumbs, [valid_th[e.cam] for e in entries], cfg,
                                                 progress=lambda m, f: prog(m, 0.42 + 0.06 * (f or 0.0)), cancel_check=cancel_check)
        report["intensity_equalisation"] = eq_report
        if "warning" in eq_report:
            report.setdefault("warnings", []).append("Intensity equalisation: " + eq_report["warning"])
        with open(f"{cfg.master_output}/gains.json", "w") as fh:        # same layout as the script: cam<c>_idx<i> (0-based camera)
            json.dump({"gains": {f"cam{e.cam}_idx{e.frame_idx}": float(scales[e.order]) for e in entries}, "report": eq_report}, fh, indent=2)
        del thumbs
    elif cfg.intensity_method == "median_scale":
        scales, smoothing_report = compute_intensity_scales(medians, cams_in_order, cfg)
        report["temporal_smoothing"] = smoothing_report
    else:
        scales = np.ones(total_entries)
        report["intensity_equalisation"] = {"method": "none"}

    def load_cropped(e: FrameEntry) -> np.ndarray:
        return read_image(f"{cfg.folder('work_cropped', e.cam)}/crop_{e.orig_idx:06d}.tif", 16)

    def stage_homogenised(e: FrameEntry, raw: np.ndarray | None = None) -> np.ndarray:
        img = load_cropped(e) if raw is None else raw
        if cfg.background_mode == "legacy":      # the script: uniformise (per-frame stretch), THEN apply the scale
            return np.clip(uniformise_background_legacy(img, cfg) * float(scales[e.order]), 0, MAX16)
        return _homogenise(img * float(scales[e.order]), valid_crop[e.cam], cfg)

    # ---------------- pass 2a: one global output gain ----------------------------------------------
    gain_out = 1.0
    if cfg.background_mode == "legacy":
        gain_out = 1.0
    elif cfg.output_scaling == "global":
        sample = entries[:: max(1, total_entries // max(1, cfg.output_scale_sample_frames))]
        vals = []
        for e in sample:
            check_cancelled(cancel_check)
            im = stage_homogenised(e)
            vm = valid_crop[e.cam]
            vals.append(np.percentile(im[vm][:: 7], 99.9) if vm.any() else np.percentile(im, 99.9))
            del im
        p = float(np.max(vals))
        gain_out = cfg.output_target_fraction * MAX16 / p if p > 1e-6 else 1.0
    report["output_gain"] = {"mode": cfg.output_scaling if cfg.background_mode != "legacy" else "legacy_per_frame_stretch",
                             "gain": float(gain_out)}
    prog("Output gain computed", 0.5)

    # ---------------- pass 2b: homogenise + denoise + export --------------------------------------
    legacy_ref = {"value": None}
    use_temporal = cfg.denoiser == "temporal_median"
    tone_ref = None
    tone_paths: dict = {}
    if cfg.tone_matching == "lut":
        ref_entry = next(e for e in entries if e.cam == alignment.reference_camera and e.frame_idx == 0)
        tone_ref = np.clip(np.rint(stage_homogenised(ref_entry) * gain_out), 0, MAX16).astype(np.uint16)

    def finish(img: np.ndarray, e: FrameEntry, tiles) -> None:
        img = np.clip(img, 0, MAX16)
        out16 = despeckle_image(img, cfg).astype(np.float32)
        if cfg.background_mode == "legacy" and cfg.use_global_intensity_norm:
            med = float(np.median(out16))
            if legacy_ref["value"] is None:
                legacy_ref["value"] = med
            if med > 1e-6:
                out16 = np.clip(out16 * (legacy_ref["value"] / med), 0, MAX16)
        final = np.clip(np.rint(out16), 0, MAX16).astype(np.uint16)
        if cfg.burn_timestamp and e.t_ns is not None:
            if cfg.export_no_timestamp_copy:
                cv2.imwrite(f"{cfg.folder('processed_clean')}/proc_{e.order:06d}.tif", final)
            final = burn_text(final, f"{e.t_ns:.0f} ns", cfg.timestamp_font_scale)
        if cfg.export_processed or cfg.export_video or do_stretch:
            cv2.imwrite(f"{cfg.folder('processed')}/proc_{e.order:06d}.tif", final)
        if cfg.save_comparisons and (e.order % max(1, cfg.comparison_every) == 0):
            try:
                save_comparison_row(tiles + [(final, "Final")], f"{cfg.folder('comparison')}/comp_{e.order:06d}_cam{e.cam + 1}_{e.orig_idx:06d}.png",
                                    cfg.comparison_tile_height)
            except Exception as exc:     # a comparison image must never abort the run
                report.setdefault("warnings", []).append(f"comparison {e.order}: {exc}")

    for e in entries:
        check_cancelled(cancel_check)
        prog(f"Pass 2/3: homogenise + denoise + export, frame {e.order + 1}/{total_entries}",
             0.5 + 0.4 * e.order / float(total_entries))
        raw = load_cropped(e)
        scaled = raw * float(scales[e.order])
        hom = stage_homogenised(e, raw) * gain_out
        if tone_ref is not None:
            matched, how = match_tone_to_reference(hom, tone_ref, cfg)
            tone_paths[how] = tone_paths.get(how, 0) + 1
            hom = matched.astype(np.float32)
        den = denoise_spatial(hom, cfg)
        if use_temporal:
            _write16(f"{cfg.folder('work_hom', e.cam)}/hom_{e.frame_idx:06d}.tif", den)
        else:
            finish(den, e, [(raw, "Calibrated+aligned"), (scaled, "Intensity-normalised"), (hom, "Homogenised")]
                   + ([(den, f"Denoised ({cfg.denoiser})")] if cfg.denoiser != "none" else []))

    if use_temporal:
        win = cfg.temporal_median_window
        pad = win // 2
        for e in entries:
            check_cancelled(cancel_check)
            prog(f"Pass 3/3: temporal median, frame {e.order + 1}/{total_entries}", 0.9 + 0.05 * e.order / float(total_entries))
            idxs = [min(max(e.frame_idx + k, 0), n_per_cam - 1) for k in range(-pad, pad + 1)]
            stack = np.stack([read_image(f"{cfg.folder('work_hom', e.cam)}/hom_{i:06d}.tif", 16) for i in idxs])
            med = np.median(stack, axis=0).astype(np.float32)
            finish(med, e, [(stack[pad], "Homogenised"), (med, "Temporal median")])

    if cfg.tone_matching == "lut":
        report["tone_matching"] = {"method": "lut", "reference": f"camera {alignment.reference_camera + 1}, first frame",
                                   "paths_used": tone_paths}
    # ---------------- exports ------------------------------------------------------------------------
    with open(f"{cfg.master_output}/timestamps.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["output_index", "camera", "original_frame_index", "source_file", "time_ns", "processed_file"])
        for e in entries:
            wr.writerow([e.order, e.cam + 1, e.orig_idx, e.source, "" if e.t_ns is None else f"{e.t_ns:.3f}",
                         f"proc_{e.order:06d}.tif"])

    if cfg.export_combined_cropped:
        for e in entries:
            src = f"{cfg.folder('work_cropped', e.cam)}/crop_{e.orig_idx:06d}.tif"
            shutil.copyfile(src, f"{cfg.folder('combined')}/proc_{e.order:06d}_cam{e.cam + 1}_{e.orig_idx:06d}.tif")

    if do_stretch:
        prog("Contrast stretch", 0.95)
        report["contrast_stretch"] = stretch_sequence(
            cfg, [f"proc_{e.order:06d}.tif" for e in entries], [e.cam for e in entries],
            progress=lambda m, f: prog(m, None if f is None else 0.95 + 0.01 * f), cancel_check=cancel_check)
        if cfg.export_video:
            report["contrast_stretch"]["video"] = video_summary(
                encode_videos(cfg.folder("stretched"), os.path.join(cfg.folder("video"), f"{video_stem(cfg.video_filename)}_stretched"),
                              cfg, cancel_check), cfg)
    if cfg.export_video:
        prog("Encoding the videos (FFV1 MKV / AVI, lossless MP4)", 0.96)
        res = encode_videos(cfg.folder("processed"), os.path.join(cfg.folder("video"), video_stem(cfg.video_filename)), cfg, cancel_check)
        report["video_export"] = video_summary(res, cfg)
        for fmt, (okf, msgf) in res.items():
            if not okf:
                report.setdefault("warnings", []).append(f"Video ({fmt}): {msgf}")
        if report["video_export"]["enabled"] and cfg.export_video_per_camera and cfg.n_cameras > 1:
            per_cam = {}
            for cam in range(cfg.n_cameras):
                check_cancelled(cancel_check)
                prog(f"Encoding the 16-bit video of camera {cam + 1}", 0.965 + 0.005 * cam / cfg.n_cameras)
                tmp = f"{cfg.folder('work')}/video_cam{cam + 1}"
                os.makedirs(tmp, exist_ok=True)
                for k, e in enumerate([x for x in entries if x.cam == cam]):
                    dst = f"{tmp}/proc_{k:06d}.tif"
                    src = f"{cfg.folder('processed')}/proc_{e.order:06d}.tif"
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copyfile(src, dst)
                resc = encode_videos(tmp, os.path.join(cfg.folder("video"), f"cam{cam + 1}_{video_stem(cfg.video_filename)}"), cfg, cancel_check)
                per_cam[f"cam{cam + 1}"] = {k: v[1] for k, v in resc.items() if v[0]}
                for fmt, (okc, msgc) in resc.items():
                    if not okc:
                        report.setdefault("warnings", []).append(f"Video of camera {cam + 1} ({fmt}): {msgc}")
            report["video_export"]["per_camera"] = per_cam
    if not cfg.export_processed:      # the TIFFs were only needed as ffmpeg's input
        shutil.rmtree(cfg.folder("processed"), ignore_errors=True)
    # cleanup of the temporary stage
    shutil.rmtree(cfg.folder("work"), ignore_errors=True)

    report["output_shape"] = [int(out_shape[0]), int(out_shape[1])]
    report["n_output_frames"] = total_entries
    with open(cfg.report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    prog("Done", 1.0)
    return report
