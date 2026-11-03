"""
Frame preprocessing shared between the Image Correlation page's live
preview and the DIC correlation engine (core/dic_processing.py), so the
region of interest, mesh and any correlation results are always defined
against exactly the pixels the user saw when setting them up.

Three independent, optional steps, always applied in this order:
1. Crop to a rectangle (`ROI`, from core/dic_correlation.py) drawn on the
   raw (full-resolution) frame -- narrows every later step down to the
   area that actually matters, e.g. the specimen and its immediate
   surroundings.
2. Contrast-stretch to a displayable/analyzable 8-bit image
   (frame_source.to_display_uint8).
3. Downscale by an integer factor (>=1; 1 = no downscale) via area
   averaging, for speed on high-resolution footage.

Frame *selection* -- processing every Nth frame rather than every frame,
for speed over a long test -- is a separate, temporal concept, handled by
`decimated_frame_indices()` below rather than this per-frame spatial
pipeline.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.dic_correlation import ROI
from core.frame_source import to_display_uint8


def crop_frame(frame: np.ndarray, crop: ROI | None) -> np.ndarray:
    """Crops `frame` (any dtype/bit depth, grayscale or RGB) to `crop`,
    clamped to the frame's own bounds. Returns `frame` unchanged if
    `crop` is None."""
    if crop is None:
        return frame
    h, w = frame.shape[:2]
    c = crop.clamped(w, h)
    return frame[c.y:c.y + c.height, c.x:c.x + c.width]


def downscale_frame(frame_uint8: np.ndarray, factor: int) -> np.ndarray:
    """Reduces a displayable 8-bit frame's resolution by an integer
    factor (1 = unchanged) using area averaging (cv2.INTER_AREA) -- the
    appropriate interpolation for shrinking an image without aliasing."""
    factor = max(1, int(factor))
    if factor == 1:
        return frame_uint8
    h, w = frame_uint8.shape[:2]
    new_w, new_h = max(1, w // factor), max(1, h // factor)
    return cv2.resize(frame_uint8, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess_frame(raw_frame: np.ndarray, crop: ROI | None, downscale_factor: int) -> np.ndarray:
    """The full pipeline: crop (on the raw frame, any bit depth) ->
    contrast-stretch to 8-bit -> downscale. Returns a contiguous uint8
    array, ready for display or correlation -- the ROI/mesh (drawn on
    this same preprocessed frame) and the correlation engine both work
    entirely in this array's pixel coordinates, never the raw frame's."""
    cropped = crop_frame(raw_frame, crop)
    display = to_display_uint8(cropped)
    downscaled = downscale_frame(display, downscale_factor)
    return np.ascontiguousarray(downscaled)


def decimated_frame_indices(n_frames: int, reference_frame: int, step: int) -> list[int]:
    """The list of frame indices to actually process for DIC: starting
    at `reference_frame` (always included first, as the zero-displacement
    frame), taking every `step`-th frame forward to the end of the
    sequence (step=1 processes every frame). DIC tracks deformation
    forward in time from an undeformed reference state, so frames before
    `reference_frame` are never included."""
    step = max(1, int(step))
    if n_frames <= 0:
        return []
    reference_frame = max(0, min(reference_frame, n_frames - 1))
    return list(range(reference_frame, n_frames, step))
