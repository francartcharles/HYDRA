"""
Frame source abstraction for the Camera / Video Sync standalone tool.

Wraps either a video file (MP4/MKV/AVI, decoded with OpenCV) or a folder of
single-frame image files (TIFF 8-bit/16-bit, JPEG/JPG, PNG, BMP) behind the
same tiny interface -- `n_frames`, `get_frame(i)`, `fps` -- so the UI page
never has to care which kind of input is currently loaded, and a slider can
address either one identically.
"""
from __future__ import annotations

import os
import re

import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi"}
IMAGE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}


def _natural_sort_key(name: str):
    """Sort key that orders "frame2" before "frame10" -- a plain
    alphabetical sort (what `os.listdir` + `sorted()` gives you) puts
    "frame10" before "frame2", which is never the recording order for a
    high-speed camera's numbered frame dump."""
    return [int(chunk) if chunk.isdigit() else chunk.lower()
            for chunk in re.split(r"(\d+)", name)]


class FrameSource:
    """Common interface implemented by both frame sources below. Not meant
    to be instantiated directly -- use `open_frame_source()`."""

    n_frames: int = 0
    fps: float | None = None  # None when unknown (e.g. a bare image sequence has no timing of its own)

    def get_frame(self, index: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class VideoFrameSource(FrameSource):
    """A single video file (MP4/MKV/AVI), decoded frame-by-frame on demand
    via OpenCV's VideoCapture."""

    def __init__(self, path: str):
        import cv2
        self.path = path
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video file (unsupported codec/container?): {path}")
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.n_frames <= 0:
            # Some containers (MKV especially, but also some MP4/AVI
            # encoders) don't carry a frame-count index OpenCV can read up
            # front -- CAP_PROP_FRAME_COUNT then comes back 0 even though
            # the file plays fine. Rather than refuse to load, count frames
            # by scanning once with grab() (demux only, no decode -- much
            # cheaper than read()) and rewind before any real frame is read.
            self.n_frames = _count_frames_by_scanning(self._cap, cv2)
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 0 else None
        self._next_read_index = 0  # what self._cap.read() will return next, absent a seek

    def get_frame(self, index: int) -> np.ndarray:
        if index < 0 or (self.n_frames and index >= self.n_frames):
            raise IndexError(f"Frame {index} out of range (0..{self.n_frames - 1})")
        cv2 = self._cv2
        # Seeking is only reliable to the nearest keyframe on many
        # codecs/containers, so re-seeking on every single read (rather
        # than only when jumping) is what actually gives frame-accurate
        # results consistently across MP4/MKV/AVI -- the extra decode cost
        # is a fine trade for a slider-driven single-frame viewer.
        if index != self._next_read_index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame_bgr = self._cap.read()
        self._next_read_index = index + 1
        if not ok:
            raise ValueError(f"Could not decode frame {index} from {self.path}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._cap.release()


def _count_frames_by_scanning(cap, cv2) -> int:
    count = 0
    while cap.grab():
        count += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return count


class ImageSequenceFrameSource(FrameSource):
    """A folder of single-frame image files, naturally sorted by file name
    and addressed by position (frame 0 = first file, etc.)."""

    def __init__(self, folder: str):
        self.folder = folder
        names = [f for f in os.listdir(folder)
                 if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        if not names:
            raise ValueError(f"No supported image files (TIFF/JPEG/PNG/BMP) found in: {folder}")
        names.sort(key=_natural_sort_key)
        self.paths = [os.path.join(folder, n) for n in names]
        self.n_frames = len(self.paths)
        self.fps = None

    def get_frame(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.n_frames:
            raise IndexError(f"Frame {index} out of range (0..{self.n_frames - 1})")
        return _read_image_file(self.paths[index])

    def frame_name(self, index: int) -> str:
        return os.path.basename(self.paths[index])


def _read_image_file(path: str) -> np.ndarray:
    """Reads one image file, preserving its native bit depth (8-bit or
    16-bit grayscale, or 8-bit RGB)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        # tifffile handles the unusual tag combinations high-speed/scientific
        # cameras sometimes write (e.g. plain 16-bit grayscale with no
        # color profile) more reliably than OpenCV's TIFF codec.
        import tifffile
        array = tifffile.imread(path)
        if array.ndim == 3 and array.shape[0] not in (3, 4) and array.shape[-1] not in (3, 4):
            # A multi-page TIFF landed here as a (pages, H, W) stack --
            # this frame source addresses one file = one frame, so only
            # its first page is used. A burst saved as one multi-page file
            # should be loaded as a video, or split into one file per frame.
            array = array[0]
        return array

    import cv2
    array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ValueError(f"Could not read image: {path}")
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
        elif array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    return array


def open_frame_source(path: str) -> FrameSource:
    """Factory: a folder opens an `ImageSequenceFrameSource` over the image
    files inside it; a video file opens a `VideoFrameSource`; a single
    image file opens an `ImageSequenceFrameSource` over its containing
    folder (so picking any one frame of a sequence still loads the whole
    thing)."""
    if os.path.isdir(path):
        return ImageSequenceFrameSource(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return VideoFrameSource(path)
    if ext in IMAGE_EXTENSIONS:
        return ImageSequenceFrameSource(os.path.dirname(path) or ".")
    raise ValueError(f"Unsupported file type: {path}")


def to_display_uint8(frame: np.ndarray, black_point: float | None = None,
                      white_point: float | None = None) -> np.ndarray:
    """Converts any frame (8-bit, 16-bit or float, grayscale or RGB) to a
    displayable 8-bit array.

    16-bit high-speed-camera frames typically only use a small slice of
    the full 0..65535 range (sensor well depth, exposure, gain), so a
    plain /257 downscale to 8-bit looks almost black; contrast-stretching
    between the 0.5th and 99.5th percentile of the frame's own values (the
    default when `black_point`/`white_point` aren't given) shows the image
    the way the acquisition software would."""
    array = np.asarray(frame)
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float64)
    if black_point is None:
        black_point = float(np.percentile(array, 0.5))
    if white_point is None:
        white_point = float(np.percentile(array, 99.5))
    if white_point <= black_point:
        # Degenerate case (e.g. a perfectly uniform frame): show mid-gray
        # rather than a black image that could look like a display bug.
        return np.full(array.shape, 128, dtype=np.uint8)
    scaled = np.clip((array - black_point) / (white_point - black_point), 0.0, 1.0) * 255.0
    return scaled.astype(np.uint8)
