"""
Single shared stylesheet + palette for the whole application.

Everything visual lives here so no page ever hand-rolls colors/fonts again
(the original app repeated `bg=test_type.mycolor` and manual
`GetSystemMetrics`-based font-size formulas on nearly every widget).
"""

BG = "#12151c"
BG_PANEL = "#1a1e27"
BG_CARD = "#20242f"
BORDER = "#2b3040"
TEXT = "#e6e8ee"
TEXT_MUTED = "#8b91a1"
ACCENT = "#5b8cff"
ACCENT_HOVER = "#7aa0ff"
ACCENT_TEXT = "#0b0e14"
DANGER = "#ff6b6b"
SUCCESS = "#4cd97b"

# Sampled directly from assets/icons/hydra_icon_512.png (the three-headed
# hydra mark) -- used only on the Welcome page, to tie its "HYDRA" title
# and its one call-to-action button back to the app icon's own color
# theme, rather than the app-wide blue ACCENT used everywhere else.
LOGO_GOLD = "#e7b440"
LOGO_GOLD_HOVER = "#f2c463"

FONT_FAMILY = "Segoe UI, Inter, -apple-system, sans-serif"

STYLESHEET = f"""
* {{
    font-family: {FONT_FAMILY};
    color: {TEXT};
}}
QMainWindow, QWidget#page {{
    background: {BG};
}}
QWidget#sidebar {{
    background: {BG_PANEL};
    border-right: 1px solid {BORDER};
}}
QLabel#appTitle {{
    font-size: 15px;
    font-weight: 600;
    padding: 18px 16px 4px 16px;
    color: {TEXT};
}}
QLabel#appSubtitle {{
    font-size: 11px;
    padding: 0 16px 14px 16px;
    color: {TEXT_MUTED};
}}
QListWidget#navList {{
    background: transparent;
    border: none;
    outline: none;
    font-size: 13px;
    padding: 6px;
}}
QListWidget#navList::item {{
    padding: 10px 12px;
    border-radius: 8px;
    margin: 2px 4px;
    color: {TEXT_MUTED};
}}
QListWidget#navList::item:selected {{
    background: {ACCENT};
    color: {ACCENT_TEXT};
    font-weight: 600;
}}
QListWidget#navList::item:hover:!selected {{
    background: {BG_CARD};
    color: {TEXT};
}}
QWidget#navSeparatorContainer {{
    background: transparent;
}}
QFrame#navSeparator {{
    background: {BORDER};
    border: none;
}}
QFrame#card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QLabel#cardTitle {{
    font-size: 14px;
    font-weight: 600;
    color: {TEXT};
}}
QLabel#cardSubtitle {{
    font-size: 11px;
    color: {TEXT_MUTED};
}}
QLabel#pageTitle {{
    font-size: 20px;
    font-weight: 700;
}}
QLabel#pageDescription {{
    font-size: 12px;
    color: {TEXT_MUTED};
}}
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 8px;
    font-size: 12px;
    min-height: 22px;
    min-width: 110px;
}}
QComboBox {{
    min-width: 160px;
}}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {ACCENT};
}}
QPushButton {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 500;
}}
QPushButton:hover {{
    border: 1px solid {ACCENT};
}}
QPushButton#primary {{
    background: {ACCENT};
    color: {ACCENT_TEXT};
    border: none;
    font-weight: 700;
}}
QPushButton#primary:hover {{
    background: {ACCENT_HOVER};
}}
QPushButton#ghost {{
    background: transparent;
    border: 1px solid {BORDER};
}}
QRadioButton, QCheckBox {{
    font-size: 12px;
    spacing: 6px;
}}
QProgressBar {{
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    text-align: center;
    height: 18px;
    font-size: 11px;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 5px;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QSplitter::handle {{
    background: {BORDER};
}}
QToolTip {{
    background: {BG_CARD};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 4px;
}}
QToolButton#bridgeButton {{
    background: {BG};
    border: 2px solid {BORDER};
    border-radius: 10px;
    padding: 10px;
    font-size: 12px;
    font-weight: 600;
}}
QToolButton#bridgeButton:hover {{
    border: 2px solid {ACCENT_HOVER};
}}
QToolButton#bridgeButton:checked {{
    background: {BG_CARD};
    border: 2px solid {ACCENT};
    color: {ACCENT_HOVER};
}}
QToolButton#helpButton {{
    background: transparent;
    border: 1px solid {TEXT_MUTED};
    border-radius: 9px;
    min-width: 18px;
    max-width: 18px;
    min-height: 18px;
    max-height: 18px;
    padding: 0px;
    font-size: 11px;
    font-weight: 700;
    color: {TEXT_MUTED};
}}
QToolButton#helpButton:hover {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: {ACCENT_TEXT};
}}
QDialog {{
    background: {BG};
}}
QLabel#welcomeTitle {{
    font-size: 46px;
    font-weight: 800;
    color: {LOGO_GOLD};
}}
QLabel#welcomeVersion {{
    font-size: 12px;
    font-weight: 600;
    color: {TEXT_MUTED};
}}
QLabel#welcomeSubtitle {{
    font-size: 13px;
    color: {TEXT_MUTED};
}}
QPushButton#welcomeDocButton {{
    background: {LOGO_GOLD};
    color: {ACCENT_TEXT};
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: 700;
}}
QPushButton#welcomeDocButton:hover {{
    background: {LOGO_GOLD_HOVER};
}}
QLabel#welcomeCredit {{
    font-size: 11px;
    color: {TEXT_MUTED};
}}
QTextBrowser#helpBrowser {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px;
    font-size: 12px;
}}
"""
