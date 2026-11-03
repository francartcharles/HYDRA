"""
Entry point.

    python main.py

Replaces mainm.py's Tk() root + Menu-driven Toplevel launcher.
"""
from __future__ import annotations

import multiprocessing
import os

# Belt-and-suspenders, set before anything (PyTensor included, imported
# transitively by ui.main_window -> ui.pages.page_bayes_calib -> core.
# calib_pymc_model) has a chance to import: forces PyTensor to never
# attempt to invoke a system C/C++ compiler (g++/cl.exe) to build its C
# linker's .pyd/.so extensions, at all, for the whole process lifetime.
#
# This is a defensive fallback only -- the actual, intended fix is that
# core/calib_pymc_model.py's DEFAULT_NUTS_SAMPLER = "nutpie" already
# avoids PyTensor's C backend entirely for the one place this app
# actually samples (it compiles via PyTensor's Numba backend instead,
# whose own dependency `llvmlite` statically links LLVM, so no system
# compiler is needed there either -- see that constant's docstring).
# This env var exists in case some OTHER PyTensor code path this app
# doesn't control (an internal PyMC/PyTensor helper, a future dependency
# upgrade, etc.) ever tries the C linker anyway: without a bundled g++
# (which a PyInstaller-frozen .exe deliberately does NOT ship, per
# BUILD_WINDOWS.md), that would otherwise fail with a confusing
# "g++ not found" error on the end user's machine instead of just
# quietly using the (slower, but always available) pure-Python linker.
# setdefault() so a user/developer's own PYTENSOR_FLAGS (e.g. to
# deliberately test the C backend during development) still wins.
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import STYLESHEET

APP_NAME = "HYDRA"
# Matches ui/main_window.py's own ICONS_DIR (same assets/icons folder).
ICONS_DIR = Path(__file__).resolve().parent / "assets" / "icons"
ICO_PATH = ICONS_DIR / "hydra_icon.ico"


def _app_icon() -> QIcon:
    """Multi-resolution app icon (taskbar/dock/alt-tab). Prefers the .ico
    file (a single file already containing the same set of sizes baked in
    the format Windows itself expects for taskbar/Start Menu icons) and
    falls back to the individual PNGs, in case a build is ever missing the
    .ico."""
    if ICO_PATH.exists():
        return QIcon(str(ICO_PATH))
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256, 512, 1024):
        path = ICONS_DIR / f"hydra_icon_{size}.png"
        if path.exists():
            icon.addFile(str(path))
    return icon


def _set_windows_taskbar_app_id():
    """Windows-only: without this, the taskbar/Start Menu icon for a
    script launched with `python main.py` is the generic Python
    interpreter icon, not this app's icon -- Windows groups taskbar
    entries by an "Application User Model ID" tied to the *process*
    (python.exe's own, by default), separately from the per-window QIcon
    set via setWindowIcon()/setWindowIcon() below. Giving this process its
    own explicit, unique app ID before the first window is created makes
    Windows treat it as its own application for taskbar/grouping/pinning
    purposes, so the window icon actually shows there instead. No effect
    on macOS/Linux, where the dock/taskbar icon already follows the
    QApplication window icon directly -- guarded so it's a no-op there."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "hydra.dynamic-materials-testing-suite.app.1"
        )
    except (AttributeError, OSError):
        pass  # older Windows/odd environment -- window icon still applies, just not the taskbar one


def main():
    _set_windows_taskbar_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    # Deliberately NOT calling app.setApplicationDisplayName(APP_NAME) here:
    # on several platforms/window managers, Qt appends " - <application
    # display name>" to every top-level window's title bar whenever the
    # title doesn't already end with it -- which, combined with
    # MainWindow.setWindowTitle's own "HYDRA -- ..." title (starting, not
    # ending, with APP_NAME), produced a visible duplicate "... - HYDRA"
    # suffix in the title bar. setApplicationName alone (still set, above)
    # is enough for Qt/OS internals (QSettings' organization scoping, etc.)
    # that key off it; only setApplicationDisplayName affects the title bar.
    app.setWindowIcon(_app_icon())
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    # MUST be the very first thing done under this guard (before main(),
    # before anything else) -- core/parallel.py's run_map() uses a
    # ProcessPoolExecutor (stdlib multiprocessing under the hood) whenever
    # a page's "Performance" worker count is set above 1 (Viscoelastic
    # Model Calibration's Bayesian search, Dispersion Correction's Run,
    # Monte-Carlo uncertainty propagation, DIC...). In a normal `python
    # main.py` run this is harmless either way, but in a PyInstaller-
    # frozen .exe, Windows' multiprocessing "spawn" start method launches
    # each worker by re-invoking THIS SAME FROZEN EXE with special
    # bootstrap command-line arguments telling it to run as a
    # multiprocessing worker instead of the normal app -- without
    # freeze_support() to intercept and handle those arguments, the
    # frozen exe doesn't recognize them and just runs the ordinary
    # `if __name__ == "__main__": main()` path again, which is exactly
    # "clicking Run pops up a whole new copy of the app instead of
    # running the calibration" (confirmed: this is the documented
    # Python/PyInstaller multiprocessing-on-Windows pitfall, not a bug
    # specific to this app's own code). A no-op on non-Windows platforms
    # and when NOT frozen, so safe to always call unconditionally.
    multiprocessing.freeze_support()
    main()
