"""
Welcome page (Home).

The very first nav item, above "Test setup" -- a simple landing screen
rather than a working tool: the app icon (`assets/icons/hydra_icon_512.png`,
the same three-headed hydra mark used for the window/taskbar icon), the
"HYDRA" wordmark styled with that icon's own gold (`ui.theme.LOGO_GOLD`,
sampled directly from the icon file -- see that constant's own comment),
and two buttons opening the bundled technical documentation PDFs -- the
French edition (`assets/docs/HYDRA_documentation_technique_FR.pdf`) and the
English edition (`assets/docs/HYDRA_technical_documentation_EN.pdf`), both
v1.0 -- in the OS's default PDF viewer.

The PDFs are shipped INSIDE the app (this same `assets/` tree the icons and
geometry/bridge drawings already live in) rather than only linked to
externally, so the button works offline and regardless of how/where the
app was installed -- consistent with this app never reaching out to the
network for anything (see the other Standalone Tools pages' own
docstrings). If that file is ever missing (a packaging mistake, or a
build that stripped `assets/docs/`), the button says so explicitly rather
than silently failing to do anything -- same "explicit status, never a
silent no-op" convention as every export/open button elsewhere in this
app (e.g. `page_crack_wave_analysis.py`'s CSV exports).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QPixmap, QDesktopServices
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton

_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
_LOGO_PATH = _ASSETS_DIR / "icons" / "hydra_icon_512.png"
_DOC_PATH_FR = _ASSETS_DIR / "docs" / "HYDRA_documentation_technique_FR.pdf"
_DOC_PATH_EN = _ASSETS_DIR / "docs" / "HYDRA_technical_documentation_EN.pdf"
_DOC_PATH = _DOC_PATH_EN  # kept for backward compatibility (the former single-PDF name)

# Displayed on the Welcome page right under the "H Y D R A" wordmark. This is
# a plain static string, not derived from git/packaging metadata -- bump it
# by hand ONLY when explicitly asked to for a given delivery; it is NOT
# meant to be incremented automatically on every change/session.
APP_VERSION = "v1.0"

# "Medium" size per the request this page was built for: large enough to
# read clearly as the app's mark, small enough to still read as a landing
# screen rather than a full-bleed splash image -- about 1/7th of this
# window's default height (`MainWindow.resize(1600, 1000)`).
_LOGO_DISPLAY_WIDTH_PX = 160


class WelcomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("page")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.addStretch(2)

        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        logo_pixmap = QPixmap(str(_LOGO_PATH))
        if not logo_pixmap.isNull():
            self.logo_label.setPixmap(
                logo_pixmap.scaledToWidth(_LOGO_DISPLAY_WIDTH_PX, Qt.SmoothTransformation)
            )
        outer.addWidget(self.logo_label)

        outer.addSpacing(18)

        # Fake letter-spacing by inserting real spaces between the letters
        # -- Qt Style Sheets have no `letter-spacing` property for QLabel,
        # unlike CSS proper, so this is the cheap, dependency-free way to
        # get the "a bit stylized" wordmark look the icon's own bold,
        # simple silhouette calls for, without needing a custom font.
        title = QLabel("H Y D R A")
        title.setObjectName("welcomeTitle")
        title.setAlignment(Qt.AlignCenter)
        outer.addWidget(title)

        self.version_label = QLabel(APP_VERSION)
        self.version_label.setObjectName("welcomeVersion")
        self.version_label.setAlignment(Qt.AlignCenter)
        outer.addWidget(self.version_label)

        subtitle = QLabel("Dynamic Materials Testing Suite")
        subtitle.setObjectName("welcomeSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        outer.addWidget(subtitle)

        outer.addSpacing(36)

        # One button per language, side by side. `doc_button` is the French one
        # (kept under its historical name); `doc_button_en` the English one.
        doc_row = QHBoxLayout()
        doc_row.addStretch(1)
        self.doc_button = QPushButton("Documentation technique (PDF, fran\u00e7ais)")
        self.doc_button.setObjectName("welcomeDocButton")
        self.doc_button.setCursor(Qt.PointingHandCursor)
        self.doc_button.clicked.connect(lambda: self._open_documentation(_DOC_PATH_FR))
        doc_row.addWidget(self.doc_button)
        doc_row.addSpacing(12)
        self.doc_button_en = QPushButton("Technical documentation (PDF, English)")
        self.doc_button_en.setObjectName("welcomeDocButton")
        self.doc_button_en.setCursor(Qt.PointingHandCursor)
        self.doc_button_en.clicked.connect(lambda: self._open_documentation(_DOC_PATH_EN))
        doc_row.addWidget(self.doc_button_en)
        doc_row.addStretch(1)
        outer.addLayout(doc_row)

        outer.addSpacing(6)

        credit = QLabel("Developped and Designed by Dr. Charles Francart")
        credit.setObjectName("welcomeCredit")
        credit.setAlignment(Qt.AlignCenter)
        outer.addWidget(credit)

        self.doc_status = QLabel("")
        self.doc_status.setObjectName("cardSubtitle")
        self.doc_status.setAlignment(Qt.AlignCenter)
        self.doc_status.setWordWrap(True)
        outer.addSpacing(6)
        outer.addWidget(self.doc_status)

        outer.addStretch(3)

        # Kept for introspection (e.g. tests checking widget order) -- real
        # Qt already installs `outer` as this widget's layout via the
        # `QVBoxLayout(self)` constructor form above; this reference just
        # makes that layout reachable without relying on `self.layout()`.
        self._outer_layout = outer

    def _open_documentation(self, path: Path):
        if not path.exists():
            # Explicit status rather than a silently dead button -- e.g. a
            # packaged build that dropped assets/docs/ by mistake.
            self.doc_status.setText(f"Documentation file not found at: {path}")
            return
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        if opened:
            self.doc_status.setText("")
        else:
            # openUrl() can fail with no exception (e.g. no PDF viewer
            # registered on this machine) -- report it rather than assume
            # the click did something.
            self.doc_status.setText(
                "Could not open the documentation with your system's default PDF viewer. "
                f"The file itself is at: {path}"
            )

    def _on_open_documentation_clicked(self):
        """Former single-button handler: opens the English edition."""
        self._open_documentation(_DOC_PATH_EN)
