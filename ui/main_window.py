"""
Main window.

The original app opened a brand-new Tkinter Toplevel (a separate OS window)
for every single feature: test type, input bar, output bar, specimen,
corrections, decay, calibration, config sensors, results, export... This
replaces every one of those with a page in a single QStackedWidget,
selected from one persistent sidebar. Nothing in this app opens a new
window except native "choose a file to load" dialogs, which are an
unavoidable OS-level interaction, not a feature launcher.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget,
    QListWidgetItem, QStackedWidget, QLabel, QFrame,
)

ICONS_DIR = Path(__file__).resolve().parent.parent / "assets" / "icons"
ICO_PATH = ICONS_DIR / "hydra_icon.ico"


def _window_icon() -> QIcon:
    """Same icon main.py's own `_app_icon()` sets on the QApplication --
    prefers the .ico file (single file, already the format Windows itself
    expects), falls back to the individual PNGs."""
    if ICO_PATH.exists():
        return QIcon(str(ICO_PATH))
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256, 512, 1024):
        path = ICONS_DIR / f"hydra_icon_{size}.png"
        if path.exists():
            icon.addFile(str(path))
    return icon

from core.project import Project
from core import session
from ui.pages.page_welcome import WelcomePage
from ui.pages.page_test_setup import TestSetupPage
from ui.pages.page_special_tests import SpecialTestsPage
from ui.pages.page_bar import BarPage
from ui.pages.page_specimen import SpecimenPage
from ui.pages.page_sensors import SensorsPage
from ui.pages.page_image_correlation import ImageCorrelationPage
from ui.pages.page_corrections import CorrectionsPage
from ui.pages.page_decay import DecayPage
from ui.pages.page_visco_calib import ViscoCalibPage
from ui.pages.page_bayes_calib import BayesCalibPage
from ui.pages.page_batch_analysis import BatchAnalysisPage
from ui.pages.page_cpg_calc import CPGCalcPage
from ui.pages.page_process import ProcessPage
from ui.pages.page_results_export import ResultsExportPage
from ui.pages.page_covariance import CovariancePage
from ui.pages.page_video_sync import VideoSyncPage
from ui.pages.page_camera_sync import CameraSyncPage
from ui.pages.page_crack_wave_analysis import CrackWaveAnalysisPage

# A bare Welcome item first (empty section label -- see the "if
# section_label" guard below -- so it reads as a single standalone entry,
# not a one-item group under a redundant "WELCOME" header), then four
# sections matching how the test workflow is actually organized: setting
# up the test itself, processing the recorded bar signals, processing the
# recorded video (DIC/crack), and one-off calibration utilities that feed
# parameters INTO the other three rather than being a step of any one
# test.
NAV_SECTIONS = [
    ("", [
        ("Welcome", "WelcomePage"),
    ]),
    ("Test setup", [
        ("Test setup", "TestSetupPage"),
        ("Special tests", "SpecialTestsPage"),
        ("Input bar", "InputBarPage"),
        ("Output bar", "OutputBarPage"),
        ("Specimen", "SpecimenPage"),
        ("Custom sensors", "SensorsPage"),
    ]),
    ("Test processing", [
        ("Dispersion correction", "CorrectionsPage"),
        ("Processing", "ProcessPage"),
        ("Results & Export", "ResultsExportPage"),
        ("Covariance matrix", "CovariancePage"),
        ("Batch analysis", "BatchAnalysisPage"),
    ]),
    ("Image Processing", [
        ("Test/Video synchronisation", "VideoSyncPage"),
        ("Cameras Synchronisation", "CameraSyncPage"),
        ("Digital Image Correlation", "ImageCorrelationPage"),
        ("Crack & Wave analysis", "CrackWaveAnalysisPage"),
    ]),
    ("Calibration", [
        ("Free-Decay damping", "DecayPage"),
        ("Viscoelastic Model Calibration", "ViscoCalibPage"),
        ("Bayesian Calibration", "BayesCalibPage"),
        ("Strain Gage Bridge Coefficient", "CPGCalcPage"),
    ]),
]

# Which nav items are relevant to which loading regime; anything not
# listed here is always shown. QS has no bars/dispersion correction at
# all; Direct Impact has no input bar (there's no incident-wave bar in
# that setup, just a separately-measured impact velocity).
NAV_VISIBILITY = {
    "InputBarPage": {"Hopkinson Bars (SHPB)"},
    "OutputBarPage": {"Hopkinson Bars (SHPB)", "Direct Impact"},
    "CorrectionsPage": {"Hopkinson Bars (SHPB)", "Direct Impact"},
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HYDRA — Dynamic Materials Testing Suite")
        self.setWindowIcon(_window_icon())
        self.resize(1600, 1000)

        self.project = Project()
        self._session_warnings = session.load(self.project) or []

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- sidebar -----------------------------------------------------
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(240)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)

        title = QLabel("Dynamic Testing Suite")
        title.setObjectName("appTitle")
        subtitle = QLabel("Monte-Carlo mechanical test analysis")
        subtitle.setObjectName("appSubtitle")
        side_layout.addWidget(title)
        side_layout.addWidget(subtitle)

        self.nav = QListWidget()
        self.nav.setObjectName("navList")
        self._nav_items = {}
        for section_index, (section_label, items) in enumerate(NAV_SECTIONS):
            if section_index > 0:
                # Extra breathing room + a visible rule between nav
                # sections (e.g. above "Standalone Tools"), so the two
                # groups of pages don't read as one continuous list.
                sep_item = QListWidgetItem()
                sep_item.setFlags(Qt.NoItemFlags)  # not selectable, not clickable-into
                sep_item.setSizeHint(QSize(0, 21))
                self.nav.addItem(sep_item)
                sep_container = QWidget()
                sep_container.setObjectName("navSeparatorContainer")
                sep_layout = QVBoxLayout(sep_container)
                sep_layout.setContentsMargins(16, 10, 16, 10)
                sep_line = QFrame()
                sep_line.setObjectName("navSeparator")
                sep_line.setFixedHeight(1)
                sep_layout.addWidget(sep_line)
                self.nav.setItemWidget(sep_item, sep_container)
            if section_label:
                header = QListWidgetItem(section_label.upper())
                header.setFlags(Qt.ItemIsEnabled)  # not selectable, not clickable-into
                header.setData(Qt.UserRole, None)
                header.setForeground(Qt.gray)
                header.setSizeHint(QSize(0, 34))
                self.nav.addItem(header)
            for label, key in items:
                item = QListWidgetItem("   " + label)
                item.setData(Qt.UserRole, key)
                self.nav.addItem(item)
                self._nav_items[key] = item
        self.nav.currentItemChanged.connect(self._on_nav_changed)
        side_layout.addWidget(self.nav, 1)

        root.addWidget(sidebar)

        # --- pages ---------------------------------------------------------
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        process_page = ProcessPage(self.project)
        self.pages = {
            "WelcomePage": WelcomePage(),
            "TestSetupPage": TestSetupPage(self.project),
            "SpecialTestsPage": SpecialTestsPage(self.project),
            "InputBarPage": BarPage(self.project, which="IB"),
            "OutputBarPage": BarPage(self.project, which="OB"),
            "SpecimenPage": SpecimenPage(self.project),
            "SensorsPage": SensorsPage(self.project),
            "ImageCorrelationPage": ImageCorrelationPage(self.project),
            "CorrectionsPage": CorrectionsPage(self.project),
            "DecayPage": DecayPage(self.project),
            "ViscoCalibPage": ViscoCalibPage(self.project),
            "BayesCalibPage": BayesCalibPage(self.project),
            "BatchAnalysisPage": BatchAnalysisPage(self.project),
            "CPGCalcPage": CPGCalcPage(self.project),
            "VideoSyncPage": VideoSyncPage(self.project),
            "CameraSyncPage": CameraSyncPage(self.project),
            "CrackWaveAnalysisPage": CrackWaveAnalysisPage(self.project),
            "ProcessPage": process_page,
            "ResultsExportPage": ResultsExportPage(self.project, process_page=process_page),
            "CovariancePage": CovariancePage(self.project),
        }
        for _, items in NAV_SECTIONS:
            for _, key in items:
                self.stack.addWidget(self.pages[key])

        self.pages["TestSetupPage"].loadingRegimeChanged.connect(self._refresh_nav_visibility)

        # Select the first real (non-header) nav item.
        for row in range(self.nav.count()):
            if self.nav.item(row).data(Qt.UserRole) is not None:
                self.nav.setCurrentRow(row)
                break
        self._refresh_nav_visibility()

        if self._session_warnings:
            self.statusBar().showMessage(
                "Restored previous session, but couldn't find: "
                + "; ".join(self._session_warnings), 15000
            )
        elif Path(session.SESSION_PATH).exists():
            self.statusBar().showMessage("Restored parameters and files from your previous session.", 6000)

    def _on_nav_changed(self, current: QListWidgetItem, _previous: QListWidgetItem):
        if current is None:
            return
        key = current.data(Qt.UserRole)
        if key is None:
            return  # a section header was somehow made current; ignore
        self.stack.setCurrentWidget(self.pages[key])
        self._refresh_nav_visibility()

    def _refresh_nav_visibility(self):
        regime = self.project.loading_regime
        current_key = self._nav_items and self.nav.currentItem() and self.nav.currentItem().data(Qt.UserRole)
        for key, item in self._nav_items.items():
            allowed = NAV_VISIBILITY.get(key)
            item.setHidden(allowed is not None and regime not in allowed)
        # If the page we were just looking at just got hidden (e.g. the
        # user switched loading regime while on the Input Bar page), jump
        # back to Test Setup instead of leaving a hidden page on screen.
        if current_key and self._nav_items[current_key].isHidden():
            for row in range(self.nav.count()):
                item = self.nav.item(row)
                if item.data(Qt.UserRole) == "TestSetupPage":
                    self.nav.setCurrentRow(row)
                    break

    def closeEvent(self, event):
        try:
            session.save(self.project)
        except OSError:
            pass  # don't block closing the app over a session-save failure
        super().closeEvent(event)
