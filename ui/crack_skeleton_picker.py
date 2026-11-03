"""
Interactive crack-skeleton tracing widget for the Crack & Wave Analysis
standalone tool.

Displays one chosen strain-field frame (a colored raster, optionally over
a grayscale background image) and lets the user trace N possibly-branching
crack paths directly on it -- real cracks are never straight and can fork,
so each crack is a small tree of nodes (core.crack_skeleton.CrackPath), not
a single ordered polyline:

- Left-click extends the ACTIVE crack's chain by one node, connected from
  the ACTIVE node (whichever was placed or right-clicked most recently).
- Right-click an existing node to make it the active node instead -- the
  next left-click then starts a NEW branch from there, without disturbing
  the rest of the crack.
- "New crack" finalizes whichever crack is being drawn and starts a fresh
  one, as many (N) as needed; the crack list below the canvas shows every
  one drawn so far and lets you click back into an earlier one to keep
  editing it.
- "Undo last point" / "Delete crack" edit the active crack.
- "Auto-detect path(s)" proposes crack(s) directly from the currently
  shown field, branches included. Default method: ridge detection
  (core.crack_skeleton.detect_cracks_ridge -- line-like structures, automatic
  threshold, short spurs pruned, branch points merged); the legacy percentile
  threshold + skeleton (detect_cracks) stays available. This is a fast
  starting point, not a separate kind of result: detected cracks land in
  the same list as hand-traced ones and are edited exactly the same way
  (add points, undo, delete, branch) -- manual tracing is always available
  as the fallback/override, never replaced by it.
- "Move branch point (click)" arms a correction mode: the branch points (junctions
  where cracks meet) are marked with rings, and a left-click on the image moves
  the NEAREST junction -- every crack of that fork together, so none is left
  behind -- to the clicked point and re-draws their paths
  (core.crack_skeleton.move_branch_point); "Undo manual correction" restores
  the last one.
- "Correct path (click)" arms a second correction mode: a left-click makes the
  crack NEAREST to the click (the selected one, when it is about as near) pass
  through the clicked point and re-updates its path around it
  (core.crack_skeleton.reroute_crack_through_point); that crack becomes the
  selected one, and branches leaving it stay attached. Whichever manual-correction button is armed is lit in the
  selected crack's colour -- the cue that a click on the image now applies.
- "Up" / "Down" / "Renumber" edit the ORDER of the list (which crack is first, the numbering
  Crack 1, 2, ...), and "Branch on it" makes the selected crack branch on the crack picked in the
  combo (core.crack_skeleton.set_crack_parent): its start is re-attached where it really leaves that
  crack and it moves right after its parent in the list. The list's tooltips and the line under it
  say who branches on whom; all of these are undone with "Undo manual correction".
- "First crack is a master crack" (with "One crack per branch"): the first crack
  stops at its first branching, where all the arms become independent cracks.
- "Mark crack start" / "Mark crack tip (end)" arm a picking mode (button
  stays pressed, status bar prompts for the click) -- the NEXT right-click
  on the canvas designates the nearest existing node of the ACTIVE crack
  as that crack's initiation point / end point
  (core.crack_skeleton.CrackPath.mark_initiation/mark_end), shown on the
  canvas as a distinct star/diamond marker. This is entirely optional --
  a crack with nothing marked behaves exactly as before -- but feeds
  "Merge into single path" below with the two points the user actually
  means, especially useful right after "Auto-detect path(s)" produces a
  branchy tree where the true initiation/tip aren't obvious from the
  topology alone. Right-clicking with no picking mode armed still does
  its normal job (select the nearest node as the active node, to branch
  from).
- "Merge into single path" collapses the SELECTED crack's tree down to
  one trunk, dropping every other branch -- for the short parasitic
  side-twigs a noisy auto-detected skeleton often has around the real
  propagating path (skeletonizing a somewhat noisy strain-localization
  mask routinely produces these). Works on a hand-traced crack too, not
  only an auto-detected one. Two methods, picked automatically:
  - If the crack has BOTH a start and end point marked (see above),
    `core.crack_skeleton.shortest_path_between_points` is used: the
    geometrically shortest path through the crack's own graph connecting
    those two points -- for a normal (non-looping) crack this is the
    tree's one unique connecting path, so nothing on the real branch
    between the two marked points is ever dropped, unlike the axis
    heuristic below, which can silently keep the wrong branch.
  - Otherwise, `core.crack_skeleton.simplify_to_main_axis_path` is used
    as before: the root (node 0)-to-leaf path whose leaf traveled
    furthest along a chosen axis (X or Y).
- "Field color scale" (Auto/Fixed + Min/Max + reset) works the same way
  as the Digital Image Correlation page's own Results color scale: Auto
  re-fits the colormap to whatever frame/field is currently shown, Fixed
  pins it to a Min/Max the user enters (or starts from "Reset to data
  range") -- useful for pulling a thin, sharp crack-localization band out
  of a field whose auto-fit range is dominated by a much larger value
  elsewhere in the frame.

The canvas keeps the field/background's true pixel aspect ratio (`ax.
set_aspect("equal", ...)` in `_redraw`) rather than stretching it to fill
the plot box -- important here specifically because a stretched image
visibly distorts a traced crack's real shape/curvature.

Background/field alignment: the strain field is plotted at each grid
point's fixed REFERENCE position (`xs`/`ys` = the point's x0/y0 in the
Image Correlation page's export -- see core/crack_wave_io.py), because
every downstream stage (skeleton tracing distances, wave speed, tip
tracking, K1c) needs one fixed, regular grid to sample from, and crack
growth is conventionally measured in that same material/reference frame
anyway. The BACKGROUND image, though, is the raw video/image-sequence
frame at whatever CURRENT time is shown -- so once the specimen has moved
by more than a pixel or two (a rigid-body displacement, common in SHPB/
Direct-Impact tests -- NOT a DIC computation error, u/v/exx are correct
per material point either way), the two would visibly drift apart if
just overlaid as-is: the field stays put at x0/y0 while the crack's real,
current position in the image has moved. `set_frame`'s
`displacement_shift_xy` corrects for this -- the owning page passes the
CURRENT frame's own median (u, v) (its dominant, roughly-uniform rigid-
body component), and the background image's extent is shifted by that
same amount so it re-aligns with the fixed-position field/crack overlay.
This is a single GLOBAL correction, not a full per-point un-warp (that
would need triangulating/inverting the whole displacement field just for
display) -- it removes the dominant bulk offset but a residual local
misalignment can remain in a region with strong local strain gradients
(most visibly right at the crack tip, which is of course exactly where
it matters most). The "Align background to strain field" checkbox lets
this correction be turned off to compare against the raw, unshifted
image.

This widget only draws whatever frame/field it is given via `set_frame` --
it does not load images or DIC data itself (that is the owning page's
job, ui/pages/page_crack_wave_analysis.py), the same division of
responsibility as ui/roi_picker.py's pickers.
"""
from __future__ import annotations

import copy

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QDoubleSpinBox, QRadioButton, QButtonGroup, QCheckBox, QComboBox,
)

from core.crack_skeleton import (
    CrackPath, new_crack, detect_cracks, detect_cracks_ridge, simplify_to_main_axis_path, shortest_path_between_points,
    move_branch_point, reroute_crack_through_point, find_junctions, choose_crack_for_click, nearest_junction,
    crack_parents, move_crack_in_list, renumber_cracks, set_crack_parent,
)
from ui.plot_canvas import PlotCanvas
from ui.widgets import _make_help_button
from ui import theme

_NODE_SELECT_RADIUS_FRACTION = 0.025  # of the image diagonal, for right-click "select nearest node"


class CrackSkeletonPicker(QWidget):
    """`cracksChanged` fires (no arguments -- call `get_cracks()`) whenever
    the crack list changes: a node added/undone, a crack started/deleted,
    or an auto-detect run."""

    cracksChanged = Signal()

    def __init__(self, figsize=(9, 6), parent=None, help_text: str = "", help_title: str = ""):
        super().__init__(parent)
        self._cracks: list = []
        self._active_crack_index: int | None = None
        self._active_node_index: int | None = None
        self._pick_role: str | None = None  # None, "start" or "end" -- see "Mark crack start/tip"
        self._manual_mode: str | None = None  # None, "branch" ("Move branch point") or "path" ("Correct path")
        self._main_ax = None                  # the image axes of the last redraw
        self._hover_artists: list = []        # preview of what the next click would modify (see _on_canvas_motion)
        self._hover_key = None
        self._move_branch_mode = False      # "Move branch point" armed: a left-click moves the nearest fork
        self._branch_move_undo: list = []   # [(index, CrackPath copy), ...] groups, one per move
        self._background: np.ndarray | None = None
        self._field: np.ndarray | None = None
        self._xs: np.ndarray | None = None
        self._ys: np.ndarray | None = None
        self._field_label = ""
        self._frame_shape = None  # (height, width) in real pixel units, for the node-selection radius
        self._bg_shift_xy = (0.0, 0.0)  # (u, v) px -- see `set_frame`'s `displacement_shift_xy`

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        self.new_crack_button = QPushButton("New crack")
        self.new_crack_button.clicked.connect(self._on_new_crack_button)
        controls.addWidget(self.new_crack_button)
        self.undo_button = QPushButton("Undo last point")
        self.undo_button.clicked.connect(self._on_undo_button)
        controls.addWidget(self.undo_button)
        self.delete_button = QPushButton("Delete crack")
        self.delete_button.clicked.connect(self._on_delete_crack_button)
        controls.addWidget(self.delete_button)
        controls.addStretch(1)
        if help_text:
            controls.addWidget(_make_help_button(self, help_title, help_text))
        layout.addLayout(controls)

        # Field color scale (Auto/Fixed + Min/Max + reset) -- same control
        # shape as the Digital Image Correlation page's Results card, so a
        # crack barely visible under the field's auto-fit range (e.g. a
        # thin, sharp localization band drowned out by a much larger max
        # elsewhere in the frame) can be pulled out by narrowing the range
        # manually, exactly the same way as there.
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Field color scale"))
        self.scale_auto_radio = QRadioButton("Auto (per frame)")
        self.scale_fixed_radio = QRadioButton("Fixed")
        self.scale_auto_radio.setChecked(True)
        scale_mode_group = QButtonGroup(self)
        scale_mode_group.addButton(self.scale_auto_radio)
        scale_mode_group.addButton(self.scale_fixed_radio)
        self.scale_auto_radio.toggled.connect(self._on_scale_mode_toggled)
        scale_row.addWidget(self.scale_auto_radio)
        scale_row.addWidget(self.scale_fixed_radio)
        scale_row.addWidget(QLabel("Min"))
        self.scale_min_spin = QDoubleSpinBox()
        self.scale_min_spin.setRange(-1e9, 1e9)
        self.scale_min_spin.setDecimals(4)
        self.scale_min_spin.setEnabled(False)
        self.scale_min_spin.valueChanged.connect(self._on_scale_bounds_changed)
        scale_row.addWidget(self.scale_min_spin)
        scale_row.addWidget(QLabel("Max"))
        self.scale_max_spin = QDoubleSpinBox()
        self.scale_max_spin.setRange(-1e9, 1e9)
        self.scale_max_spin.setDecimals(4)
        self.scale_max_spin.setEnabled(False)
        self.scale_max_spin.valueChanged.connect(self._on_scale_bounds_changed)
        scale_row.addWidget(self.scale_max_spin)
        self.scale_reset_button = QPushButton("Reset to data range")
        self.scale_reset_button.setEnabled(False)
        self.scale_reset_button.clicked.connect(self._on_scale_reset)
        scale_row.addWidget(self.scale_reset_button)
        scale_row.addStretch(1)
        layout.addLayout(scale_row)

        # Corrects the background image's position for the specimen's own
        # (roughly rigid-body) motion since the reference frame -- see the
        # module docstring's "Background/field alignment" section. On by
        # default since a misaligned background is confusing to trace
        # against; left available to switch off to compare against the
        # raw, unshifted frame.
        align_row = QHBoxLayout()
        self.align_background_checkbox = QCheckBox("Align background to strain field (auto rigid-body shift)")
        self.align_background_checkbox.setChecked(True)
        self.align_background_checkbox.toggled.connect(self._on_align_toggle)
        align_row.addWidget(self.align_background_checkbox)
        self.align_status_label = QLabel("")
        self.align_status_label.setObjectName("cardSubtitle")
        align_row.addWidget(self.align_status_label)
        align_row.addStretch(1)
        layout.addLayout(align_row)

        self.status_label = QLabel(
            "Load data and pick a frame/field below, then left-click to trace a crack "
            "(right-click a placed point to branch from it)."
        )
        self.status_label.setObjectName("cardSubtitle")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        body = QHBoxLayout()
        body.setSpacing(10)

        self.canvas = PlotCanvas(figsize=figsize)
        body.addWidget(self.canvas, 3)
        self.canvas.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.canvas.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)

        side = QVBoxLayout()
        side.setSpacing(6)
        side.addWidget(QLabel("Cracks"))
        self.crack_list = QListWidget()
        self.crack_list.setMaximumWidth(220)
        self.crack_list.currentRowChanged.connect(self._on_list_selection_changed)
        side.addWidget(self.crack_list, 1)

        # -- order of the list and who branches on whom
        order_row = QHBoxLayout()
        self.crack_up_button = QPushButton("\u25b2 Up")
        self.crack_up_button.setToolTip("Moves the selected crack one place up in the list (order of the cracks).")
        self.crack_up_button.clicked.connect(lambda: self._on_move_crack(-1))
        order_row.addWidget(self.crack_up_button)
        self.crack_down_button = QPushButton("\u25bc Down")
        self.crack_down_button.setToolTip("Moves the selected crack one place down in the list.")
        self.crack_down_button.clicked.connect(lambda: self._on_move_crack(+1))
        order_row.addWidget(self.crack_down_button)
        self.renumber_button = QPushButton("Renumber")
        self.renumber_button.setToolTip(
            "Renames \"Crack 7 (auto)\"-style cracks after their position in the list (Crack 1, Crack 2, ...)."
        )
        self.renumber_button.clicked.connect(self._on_renumber_cracks)
        order_row.addWidget(self.renumber_button)
        side.addLayout(order_row)
        self.parent_info_label = QLabel("")
        self.parent_info_label.setObjectName("cardSubtitle")
        self.parent_info_label.setWordWrap(True)
        side.addWidget(self.parent_info_label)
        parent_row = QHBoxLayout()
        self.parent_combo = QComboBox()
        self.parent_combo.setToolTip("The crack the selected crack should branch on.")
        parent_row.addWidget(self.parent_combo, 1)
        self.set_parent_button = QPushButton("Branch on it")
        self.set_parent_button.setToolTip(
            "Makes the selected crack BRANCH ON the crack chosen on the left: its start is re-attached "
            "where it really leaves that crack (a node is added there), and it is moved right after its "
            "parent in the list. Undo with \"Undo manual correction\"."
        )
        self.set_parent_button.clicked.connect(self._on_set_parent)
        parent_row.addWidget(self.set_parent_button)
        side.addLayout(parent_row)

        side.addWidget(QLabel("Auto-detect from the shown field:"))
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method"))
        self.detect_method_combo = QComboBox()
        self.detect_method_combo.addItem("Ridge (automatic, branches)", "ridge")
        self.detect_method_combo.addItem("Percentile threshold (simple)", "threshold")
        self.detect_method_combo.setToolTip(
            "Ridge: looks for thin line-like structures (a crack is a ridge of the field), picks "
            "its own threshold, keeps weak branches attached to their trunk, removes the short "
            "parasitic twigs and merges branch points -- works on noisy fields and on branching "
            "cracks. Percentile threshold: keeps everything above a percentile of the field and "
            "thins it (fast, but noise and thick bands leave many fake branches)."
        )
        self.detect_method_combo.currentIndexChanged.connect(self._on_detect_method_changed)
        method_row.addWidget(self.detect_method_combo)
        side.addLayout(method_row)
        sens_row = QHBoxLayout()
        sens_row.addWidget(QLabel("Sensitivity"))
        self.sensitivity_spin = QDoubleSpinBox()
        self.sensitivity_spin.setRange(0.3, 3.0)
        self.sensitivity_spin.setSingleStep(0.1)
        self.sensitivity_spin.setDecimals(2)
        self.sensitivity_spin.setValue(1.0)
        self.sensitivity_spin.setToolTip(
            "Ridge method only. 1 = automatic threshold. Above 1 keeps weaker ridges (more / longer "
            "branches, more noise); below 1 keeps only the surest cracks."
        )
        sens_row.addWidget(self.sensitivity_spin)
        side.addLayout(sens_row)
        self.split_branches_check = QCheckBox("One crack per branch")
        self.split_branches_check.setChecked(True)
        self.split_branches_check.setToolTip(
            "Ridge method only. Checked: the main crack and EVERY branch each become their own crack in the "
            "list (a branch starts at its branch point on the parent). Unchecked: one crack per connected "
            "tree, branches included as a tree."
        )
        side.addWidget(self.split_branches_check)
        self.master_first_check = QCheckBox("First crack is a master crack (splits at its first branching)")
        self.master_first_check.setChecked(False)
        self.master_first_check.setToolTip(
            "Ridge + \"One crack per branch\" only. Checked: the first crack (from the crack's origin) is a "
            "\"master\" crack that stops at its FIRST branching, where it splits into 2 (or more) "
            "independent cracks, each starting at that branching. Unchecked: the straightest arm "
            "simply continues the first crack."
        )
        side.addWidget(self.master_first_check)
        self.split_branches_check.toggled.connect(self.master_first_check.setEnabled)
        spacing_row = QHBoxLayout()
        spacing_row.addWidget(QLabel("Node spacing"))
        self.node_spacing_spin = QDoubleSpinBox()
        self.node_spacing_spin.setRange(0.0, 200.0)
        self.node_spacing_spin.setDecimals(1)
        self.node_spacing_spin.setSingleStep(1.0)
        self.node_spacing_spin.setValue(0.0)
        self.node_spacing_spin.setSpecialValueText("auto (1 cell)")
        self.node_spacing_spin.setSuffix(" px")
        self.node_spacing_spin.setToolTip(
            "Ridge method only. Distance between the points of an auto-detected crack. Auto = one "
            "correlation-mesh cell. Smaller = more points following the crack's shape more closely."
        )
        spacing_row.addWidget(self.node_spacing_spin)
        side.addLayout(spacing_row)
        pct_row = QHBoxLayout()
        pct_row.addWidget(QLabel("Percentile"))
        self.percentile_spin = QDoubleSpinBox()
        self.percentile_spin.setRange(50.0, 99.9)
        self.percentile_spin.setValue(90.0)
        self.percentile_spin.setSuffix(" %")
        self.percentile_spin.setToolTip("Percentile threshold method only.")
        self.percentile_spin.setEnabled(False)
        pct_row.addWidget(self.percentile_spin)
        side.addLayout(pct_row)
        len_row = QHBoxLayout()
        len_row.addWidget(QLabel("Min length"))
        self.min_length_spin = QDoubleSpinBox()
        self.min_length_spin.setRange(0.0, 100000.0)
        self.min_length_spin.setValue(15.0)
        self.min_length_spin.setSuffix(" px")
        len_row.addWidget(self.min_length_spin)
        side.addLayout(len_row)
        self.autodetect_button = QPushButton("Auto-detect path(s)")
        self.autodetect_button.clicked.connect(self._on_autodetect_button)
        side.addWidget(self.autodetect_button)

        # Manual corrections (one armed at a time; the armed button is highlighted in the colour of
        # the selected crack to tell the operator it is ready for a click on the image):
        #  * Move branch point: a left-click moves the branch point of the two nearest cracks (a parent
        #    and the branch that starts on it) to the clicked point and redraws their paths.
        #  * Correct path: a left-click makes the SELECTED crack pass through the clicked point and
        #    re-updates its path.
        side.addWidget(QLabel("Manual corrections (arm, then click on the image):"))
        branch_row = QHBoxLayout()
        self.move_branch_button = QPushButton("Move branch point (click)")
        self.move_branch_button.setCheckable(True)
        self.move_branch_button.setToolTip(
            "Arm, then left-click on the image where a branch REALLY leaves its parent: the two "
            "nearest cracks (the parent and the branch that starts on it) get their branch point "
            "moved to that click and their paths are redrawn. Stays armed for several corrections; "
            "click the button again to go back to normal tracing. Works on the cracks produced with "
            "\"One crack per branch\". The button lights up in the selected crack's colour while armed."
        )
        self.move_branch_button.clicked.connect(self._on_move_branch_button)
        branch_row.addWidget(self.move_branch_button)
        self.correct_path_button = QPushButton("Correct path (click)")
        self.correct_path_button.setCheckable(True)
        self.correct_path_button.setToolTip(
            "Select a crack in the list, arm this, then left-click on the image: that crack is re-made "
            "so that it passes through the clicked point (its path is updated around it, the rest is "
            "untouched; branches leaving it stay attached). Stays armed for several clicks; click the "
            "button again to stop. The button lights up in the selected crack's colour while armed."
        )
        self.correct_path_button.clicked.connect(self._on_correct_path_button)
        branch_row.addWidget(self.correct_path_button)
        side.addLayout(branch_row)
        self.undo_branch_button = QPushButton("Undo manual correction")
        self.undo_branch_button.setEnabled(False)
        self.undo_branch_button.clicked.connect(self._on_undo_branch_move)
        side.addWidget(self.undo_branch_button)

        # Arms a picking mode: the next right-click on the canvas marks
        # the nearest node of the ACTIVE crack as its initiation/end point
        # (see module docstring). Entirely optional, but lets "Merge into
        # single path" below connect the two points the user actually
        # means instead of guessing a branch from the topology alone --
        # most useful right after "Auto-detect path(s)".
        side.addWidget(QLabel("Mark crack start/tip (right-click a point):"))
        mark_row = QHBoxLayout()
        self.mark_start_button = QPushButton("Mark crack start")
        self.mark_start_button.setCheckable(True)
        self.mark_start_button.clicked.connect(self._on_mark_start_button)
        mark_row.addWidget(self.mark_start_button)
        self.mark_end_button = QPushButton("Mark crack tip (end)")
        self.mark_end_button.setCheckable(True)
        self.mark_end_button.clicked.connect(self._on_mark_end_button)
        mark_row.addWidget(self.mark_end_button)
        side.addLayout(mark_row)

        # Collapses the SELECTED crack's tree down to a single trunk,
        # dropping every other branch -- for the short parasitic
        # side-twigs a noisy auto-detected (or even hand-traced) skeleton
        # can have around the real propagating path. Uses the shortest
        # path between the marked start/end points above when both are
        # set (core.crack_skeleton.shortest_path_between_points), else
        # falls back to the axis heuristic below
        # (core.crack_skeleton.simplify_to_main_axis_path) -- see the
        # module docstring for exactly which branch each one keeps.
        side.addWidget(QLabel("Simplify selected crack to one path:"))
        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("Axis"))
        self.simplify_x_radio = QRadioButton("X")
        self.simplify_y_radio = QRadioButton("Y")
        self.simplify_x_radio.setChecked(True)
        simplify_axis_group = QButtonGroup(self)
        simplify_axis_group.addButton(self.simplify_x_radio)
        simplify_axis_group.addButton(self.simplify_y_radio)
        axis_row.addWidget(self.simplify_x_radio)
        axis_row.addWidget(self.simplify_y_radio)
        axis_row.addStretch(1)
        side.addLayout(axis_row)
        self.simplify_button = QPushButton("Merge into single path")
        self.simplify_button.clicked.connect(self._on_simplify_button)
        side.addWidget(self.simplify_button)
        side.addStretch(1)

        body.addLayout(side, 1)
        layout.addLayout(body)

        self._show_placeholder("Load data and pick a frame/field below to start tracing.")

    # -- data the widget displays (driven by the owning page) -----------------

    def set_frame(self, background_uint8: np.ndarray | None, field_values: np.ndarray | None,
                  xs: np.ndarray, ys: np.ndarray, field_label: str = "",
                  displacement_shift_xy: tuple[float, float] | None = None):
        """Displays a new background image / strain-field pair -- e.g. the
        user scrubbed to a different frame or switched which field is
        shown -- and redraws the existing cracks on top. `field_values` is
        shaped (len(ys), len(xs)), NaN outside the correlated ROI, same
        layout as a DICFrameResult field; `background_uint8` is the raw
        grayscale/RGB frame (already contrast-stretched), or None to show
        the field alone.

        `displacement_shift_xy`, when given, is the (u, v) pixel shift
        (the owning page passes this frame's own median u/v) the
        background image is displayed with, to re-align it with the
        field/crack overlay -- both of which stay at the FIXED reference
        (x0, y0) grid -- after the specimen has moved since the reference
        frame. See the module docstring's "Background/field alignment"
        section for why this is needed and what it does and doesn't fix.
        `None`/`(0, 0)` shows the background at its own raw position, no
        correction."""
        self._background = background_uint8
        self._field = None if field_values is None else np.asarray(field_values, dtype=float)
        self._xs = np.asarray(xs, dtype=float) if xs is not None else None
        self._ys = np.asarray(ys, dtype=float) if ys is not None else None
        self._field_label = field_label
        self._bg_shift_xy = displacement_shift_xy or (0.0, 0.0)
        if background_uint8 is not None:
            self._frame_shape = background_uint8.shape[:2]
        elif self._xs is not None and self._ys is not None and self._xs.size and self._ys.size:
            self._frame_shape = (float(self._ys.max() - self._ys.min()) or 1.0,
                                  float(self._xs.max() - self._xs.min()) or 1.0)
        sx, sy = self._bg_shift_xy
        if abs(sx) > 1e-6 or abs(sy) > 1e-6:
            self.align_status_label.setText(f"Background shifted by ({sx:.2f}, {sy:.2f}) px to re-align.")
        else:
            self.align_status_label.setText("")
        self._redraw()

    def _on_align_toggle(self, _checked=None):
        self._redraw()

    # -- crack list access (read by the rest of the tool) ----------------------

    def get_cracks(self) -> list:
        return self._cracks

    def set_cracks(self, cracks: list):
        """Programmatically restores a crack list (e.g. from a previous
        session state), replacing whatever is currently traced."""
        self._cracks = list(cracks)
        self._reset_branch_undo()
        self._active_crack_index = len(self._cracks) - 1 if self._cracks else None
        self._active_node_index = None
        self._refresh_list()
        self._redraw()
        self.cracksChanged.emit()

    def clear_all(self):
        self._cracks = []
        self._reset_branch_undo()
        self._active_crack_index = None
        self._active_node_index = None
        self._refresh_list()
        self._redraw()
        self.cracksChanged.emit()

    # -- crack editing ----------------------------------------------------------

    def _on_new_crack_button(self):
        self._start_new_crack()
        self._update_status()

    def _start_new_crack(self):
        crack = new_crack(len(self._cracks))
        self._cracks.append(crack)
        self._active_crack_index = len(self._cracks) - 1
        self._active_node_index = None
        self._refresh_list()
        self._redraw()
        self.cracksChanged.emit()

    def _active_crack(self) -> CrackPath | None:
        if self._active_crack_index is None or not (0 <= self._active_crack_index < len(self._cracks)):
            return None
        return self._cracks[self._active_crack_index]

    def _on_undo_button(self):
        crack = self._active_crack()
        if crack is None or not crack.nodes:
            return
        removed_last = self._active_node_index == len(crack.nodes) - 1
        crack.remove_last_node()
        if removed_last:
            self._active_node_index = len(crack.nodes) - 1 if crack.nodes else None
        self._redraw()
        self._refresh_list()
        self._update_status()
        self.cracksChanged.emit()

    def _on_delete_crack_button(self):
        if self._active_crack_index is None:
            return
        del self._cracks[self._active_crack_index]
        self._reset_branch_undo()
        self._active_crack_index = None
        self._active_node_index = None
        self._refresh_list()
        self._redraw()
        self._update_status()
        self.cracksChanged.emit()

    def _on_detect_method_changed(self, _index=None):
        ridge = self.detect_method_combo.currentData() == "ridge"
        self.sensitivity_spin.setEnabled(ridge)
        self.split_branches_check.setEnabled(ridge)
        self.node_spacing_spin.setEnabled(ridge)
        self.percentile_spin.setEnabled(not ridge)

    def _on_autodetect_button(self):
        if self._field is None or self._xs is None or self._ys is None:
            self.status_label.setText("Load data and pick a frame/field first.")
            return
        finite = self._field[np.isfinite(self._field)]
        if finite.size == 0:
            self.status_label.setText("The shown field has no valid (non-NaN) values to threshold.")
            return
        use_ridge = self.detect_method_combo.currentData() == "ridge"
        if use_ridge:
            detected = detect_cracks_ridge(
                self._field, self._xs, self._ys, sensitivity=self.sensitivity_spin.value(),
                min_length_px=self.min_length_spin.value(),
                node_spacing_px=(self.node_spacing_spin.value() or None),
                split_branches=self.split_branches_check.isChecked(),
                master_first_crack=(self.split_branches_check.isChecked() and self.master_first_check.isChecked()),
            )
            how = (f"ridge detection, sensitivity {self.sensitivity_spin.value():.2f}"
                   + (", one crack per branch" if self.split_branches_check.isChecked() else "")
                   + (", master crack" if self.split_branches_check.isChecked() and self.master_first_check.isChecked() else ""))
            none_msg = ("No line-like crack found -- raise the sensitivity, lower the minimum "
                        "length, or pick a field/frame where the crack is visible.")
        else:
            threshold = float(np.percentile(finite, self.percentile_spin.value()))
            detected = detect_cracks(
                self._field, self._xs, self._ys, threshold=threshold,
                min_length_px=self.min_length_spin.value(),
            )
            how = f"threshold {threshold:.4g} ({self.percentile_spin.value():.0f}th percentile)"
            none_msg = (f"No band found at/above the {self.percentile_spin.value():.0f}th percentile "
                        f"(threshold {threshold:.4g}) -- try a lower percentile.")
        if not detected:
            self.status_label.setText(none_msg)
            return
        # Detected cracks are numbered/colored continuing from whatever is
        # already in the list, exactly like manually starting that many
        # new cracks -- there is no separate "auto" bucket to manage.
        self._reset_branch_undo()
        base = len(self._cracks)
        for i, crack in enumerate(detected):
            crack.color = new_crack(base + i).color
        self._cracks.extend(detected)
        self._active_crack_index = len(self._cracks) - 1
        self._active_node_index = None
        self._refresh_list()
        self._redraw()
        self.status_label.setText(
            f"Auto-detected {len(detected)} crack(s) with {how} -- edit freely, same as a hand-traced crack."
        )
        self.cracksChanged.emit()

    def _on_simplify_button(self):
        crack = self._active_crack()
        if crack is None:
            self.status_label.setText("Select a crack in the list first.")
            return
        n_before = len(crack.nodes)
        use_marked_points = crack.initiation_index is not None and crack.end_index is not None
        try:
            if use_marked_points:
                simplified = shortest_path_between_points(crack, crack.initiation_index, crack.end_index)
                method_note = "the shortest path between its marked start and tip"
            else:
                axis = "x" if self.simplify_x_radio.isChecked() else "y"
                simplified = simplify_to_main_axis_path(crack, axis)
                method_note = (
                    f"the furthest-along-{axis.upper()} branch (mark a crack start and tip first, "
                    f"then merge again, for a more reliable result on a branchy crack)"
                )
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self._cracks[self._active_crack_index] = simplified
        self._active_node_index = len(simplified.nodes) - 1
        self._refresh_list()
        self._redraw()
        dropped = n_before - len(simplified.nodes)
        self.status_label.setText(
            f"\"{simplified.name}\" merged to a single {len(simplified.nodes)}-point path, following "
            f"{method_note} -- {dropped} point(s) on other branches dropped."
            if dropped > 0 else
            f"\"{simplified.name}\" was already a single path -- nothing to drop."
        )
        self.cracksChanged.emit()

    def _on_mark_start_button(self, checked: bool):
        self.mark_end_button.setChecked(False)
        self._pick_role = "start" if checked else None
        self._update_status()

    def _on_mark_end_button(self, checked: bool):
        self.mark_start_button.setChecked(False)
        self._pick_role = "end" if checked else None
        self._update_status()

    # -- field color scale (Auto/Fixed) -----------------------------------
    # Same shape as the Digital Image Correlation page's Results card
    # (page_image_correlation.py's `_on_color_scale_*` / `_current_vmin_
    # vmax`), scoped to just this widget rather than persisted to
    # `Project` -- consistent with this whole tool being self-sufficient
    # and not sharing state with the Image Correlation page.

    def _on_scale_mode_toggled(self, _checked=None):
        fixed = self.scale_fixed_radio.isChecked()
        self.scale_min_spin.setEnabled(fixed)
        self.scale_max_spin.setEnabled(fixed)
        self.scale_reset_button.setEnabled(fixed)
        if fixed and self.scale_min_spin.value() == self.scale_max_spin.value():
            self._on_scale_reset()
        self._redraw()

    def _on_scale_bounds_changed(self, _value=None):
        self._redraw()

    def _on_scale_reset(self):
        if self._field is None:
            return
        finite = self._field[np.isfinite(self._field)]
        if finite.size == 0:
            return
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        self.scale_min_spin.blockSignals(True)
        self.scale_max_spin.blockSignals(True)
        self.scale_min_spin.setValue(vmin)
        self.scale_max_spin.setValue(vmax)
        self.scale_min_spin.blockSignals(False)
        self.scale_max_spin.blockSignals(False)
        self._redraw()

    def _current_vmin_vmax(self):
        """(vmin, vmax) to use for the field imshow -- the Fixed bounds
        when that mode is selected, else (None, None) so matplotlib
        auto-fits the shown frame's own data, same convention as the
        Image Correlation page's `_current_vmin_vmax`."""
        if self.scale_fixed_radio.isChecked():
            return self.scale_min_spin.value(), self.scale_max_spin.value()
        return None, None

    # -- list panel ---------------------------------------------------------

    def _refresh_list(self):
        self.crack_list.blockSignals(True)
        self.crack_list.clear()
        parents = crack_parents(self._cracks)
        for i, crack in enumerate(self._cracks):
            pixmap = QPixmap(12, 12)
            pixmap.fill(QColor(crack.color))
            item = QListWidgetItem(QIcon(pixmap),
                                    f"{crack.name}  ({len(crack.nodes)} pts, {crack.total_length():.1f} px)")
            if i in parents:
                item.setToolTip(f"Branches on {self._cracks[parents[i][0]].name}")
            self.crack_list.addItem(item)
        if self._active_crack_index is not None and 0 <= self._active_crack_index < self.crack_list.count():
            self.crack_list.setCurrentRow(self._active_crack_index)
        self.crack_list.blockSignals(False)
        self._refresh_parent_controls()

    def _refresh_parent_controls(self):
        """Parent combo (every other crack) + the "branches on ..." line for the selected crack."""
        if not hasattr(self, "parent_combo"):
            return
        active = self._active_crack_index
        parents = crack_parents(self._cracks)
        self.parent_combo.blockSignals(True)
        self.parent_combo.clear()
        for i, crack in enumerate(self._cracks):
            if i != active:
                pixmap = QPixmap(12, 12)
                pixmap.fill(QColor(crack.color))
                self.parent_combo.addItem(QIcon(pixmap), crack.name, i)
        if active in parents:
            k = self.parent_combo.findData(parents[active][0])
            if k >= 0:
                self.parent_combo.setCurrentIndex(k)
        self.parent_combo.blockSignals(False)
        has = active is not None and 0 <= active < len(self._cracks)
        for w in (self.crack_up_button, self.crack_down_button, self.set_parent_button, self.parent_combo):
            w.setEnabled(has)
        self.renumber_button.setEnabled(bool(self._cracks))
        self.set_parent_button.setEnabled(has and self.parent_combo.count() > 0)
        if not has:
            self.parent_info_label.setText("")
        elif active in parents:
            pj = parents[active]
            self.parent_info_label.setText(
                f"Branches on {self._cracks[pj[0]].name} at ({pj[1][0]:.0f}, {pj[1][1]:.0f}).")
        else:
            self.parent_info_label.setText("Not attached to another crack (or first of its fork).")

    # -- order of the list / parentage -------------------------------------------

    def _push_undo_snapshot(self):
        self._branch_move_undo.append([(i, copy.deepcopy(c)) for i, c in enumerate(self._cracks)])
        self.undo_branch_button.setEnabled(True)

    def _on_move_crack(self, delta: int):
        index = self._active_crack_index
        if index is None or not (0 <= index < len(self._cracks)):
            self.status_label.setText("Select a crack in the list first.")
            return
        new = max(0, min(len(self._cracks) - 1, index + delta))
        if new == index:
            return
        self._push_undo_snapshot()
        self._active_crack_index = move_crack_in_list(self._cracks, index, delta)
        self._refresh_list()
        self._redraw()
        self.status_label.setText(
            f"\"{self._cracks[self._active_crack_index].name}\" is now number {self._active_crack_index + 1} "
            "in the list. \"Renumber\" renames the cracks after their new order.")
        self.cracksChanged.emit()

    def _on_renumber_cracks(self):
        self._push_undo_snapshot()
        n = renumber_cracks(self._cracks)
        if not n:
            self._branch_move_undo.pop()
            self.undo_branch_button.setEnabled(bool(self._branch_move_undo))
            self.status_label.setText("The cracks are already numbered after their order.")
            return
        self._refresh_list()
        self._redraw()
        self.status_label.setText(f"{n} crack(s) renamed after their position in the list.")
        self.cracksChanged.emit()

    def _on_set_parent(self):
        child = self._active_crack_index
        parent = self.parent_combo.currentData()
        if child is None or parent is None:
            self.status_label.setText("Select a crack in the list, and the crack it branches on, first.")
            return
        before = [(i, copy.deepcopy(c)) for i, c in enumerate(self._cracks)]
        child_name, parent_name = self._cracks[child].name, self._cracks[parent].name
        try:
            info = set_crack_parent(self._cracks, child, parent)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self._branch_move_undo.append(before)
        self.undo_branch_button.setEnabled(True)
        self._active_crack_index = info["child"]
        self._active_node_index = None
        self._refresh_list()
        self._redraw()
        self.status_label.setText(
            f"\"{child_name}\" now branches on \"{parent_name}\" at ({info['point'][0]:.0f}, "
            f"{info['point'][1]:.0f})" + (" and was moved right after it in the list" if info["moved_in_list"] else "")
            + ". Use \"Renumber\" to rename the cracks after the new order.")
        self.cracksChanged.emit()

    def _on_list_selection_changed(self, row: int):
        if row < 0 or row >= len(self._cracks):
            self._active_crack_index = None
            self._active_node_index = None
        else:
            self._active_crack_index = row
            crack = self._cracks[row]
            self._active_node_index = len(crack.nodes) - 1 if crack.nodes else None
        self._redraw()
        self._update_status()
        self._update_manual_styles()
        self._refresh_parent_controls()
        if self._manual_mode == "path":
            self._set_manual_mode("path")        # re-word the prompt for the newly selected crack

    # -- canvas interaction ---------------------------------------------------

    def _on_canvas_click(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if self._main_ax is not None and event.inaxes is not self._main_ax:
            return                                   # a click on the colour bar is not a point on the image
        if event.button == 1:
            self._on_left_click(float(event.xdata), float(event.ydata))
        elif event.button == 3:
            self._on_right_click(float(event.xdata), float(event.ydata))

    def _on_canvas_motion(self, event):
        """While a manual-correction button is armed, PREVIEW what a click would modify: the crack
        that "Correct path" would re-make is underlaid in white, the branch point that "Move branch
        point" would move gets a bold yellow ring -- so the operator sees the target BEFORE clicking."""
        mode = self._manual_mode
        key, draw = None, None
        if (mode and event.inaxes is not None and event.xdata is not None and event.ydata is not None
                and (self._main_ax is None or event.inaxes is self._main_ax)):
            x, y = float(event.xdata), float(event.ydata)
            if mode == "path":
                idx, _why = choose_crack_for_click(self._cracks, x, y, selected=self._active_crack_index)
                if idx is not None:
                    key, draw = ("path", idx), idx
            else:
                jn, _d = nearest_junction(self._cracks, x, y, prefer=self._active_crack_index)
                if jn is not None:
                    key, draw = ("branch", tuple(np.round(jn["point"], 1))), jn
        if key == self._hover_key:
            return
        self._hover_key = key
        for art in self._hover_artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        self._hover_artists = []
        if draw is not None and event.inaxes is not None:
            ax = event.inaxes
            if mode == "path":
                pts = np.asarray(self._cracks[draw].nodes, dtype=float)
                self._hover_artists += ax.plot(pts[:, 0], pts[:, 1], color="#ffffff", linewidth=6.0,
                                               alpha=0.55, zorder=4, solid_capstyle="round")
            else:
                self._hover_artists.append(ax.scatter(
                    [draw["point"][0]], [draw["point"][1]], s=520, facecolors="none",
                    edgecolors="#ffd400", linewidths=3.2, zorder=9))
        self.canvas.refresh()

    # -- manual branch-point adjustment -----------------------------------------

    def _reset_branch_undo(self):
        self._branch_move_undo = []
        if hasattr(self, "undo_branch_button"):
            self.undo_branch_button.setEnabled(False)

    def _set_manual_mode(self, mode):
        """Arms one manual-correction mode ("branch" / "path") or none; the buttons are exclusive."""
        self._manual_mode = mode
        self._move_branch_mode = mode == "branch"
        self.move_branch_button.blockSignals(True)
        self.correct_path_button.blockSignals(True)
        self.move_branch_button.setChecked(mode == "branch")
        self.correct_path_button.setChecked(mode == "path")
        self.move_branch_button.blockSignals(False)
        self.correct_path_button.blockSignals(False)
        self._update_manual_styles()
        if mode == "branch":
            n_j = len(find_junctions(self._cracks))
            self.status_label.setText(
                "Move branch point: the branch points are ringed on the image. Left-click where a fork "
                "really is -- the NEAREST ringed branch point (with every crack of that fork) is moved "
                f"there. {n_j} branch point(s) found. Click the button again to stop."
                if n_j else
                "Move branch point: no branch point found -- it needs cracks that start on another crack "
                "(auto-detect with \"One crack per branch\")."
            )
        elif mode == "path":
            self.status_label.setText(
                "Correct path: left-click on the image where a crack should pass -- the crack NEAREST to "
                "the click is re-made through that point (the selected crack wins when it is about as "
                "near), and it becomes the selected one. Click the button again to stop."
            )
        else:
            self._update_status()
        if hasattr(self, "canvas"):
            self._redraw()                         # junction rings appear / disappear

    def _update_manual_styles(self):
        """The ARMED manual-correction button is lit in the colour of the selected crack (the
        operator's cue that a click on the image now applies); unarmed buttons use the normal look."""
        crack = self._active_crack()
        color = crack.color if crack is not None and getattr(crack, "color", None) else theme.ACCENT
        qc = QColor(color)
        lum = 0.299 * qc.red() + 0.587 * qc.green() + 0.114 * qc.blue()
        text = "#0b0e14" if lum > 140 else "#ffffff"
        lit = (f"QPushButton {{ background-color: {color}; color: {text}; font-weight: bold; "
               f"border: 2px solid #ffffff; }} "
               f"QPushButton:hover {{ background-color: {color}; }}")
        for button, mode in ((self.move_branch_button, "branch"), (self.correct_path_button, "path")):
            button.setStyleSheet(lit if self._manual_mode == mode else "")

    def _on_move_branch_button(self, checked: bool):
        self._set_manual_mode("branch" if checked else None)

    def _on_correct_path_button(self, checked: bool):
        self._set_manual_mode("path" if checked else None)

    def _apply_branch_move(self, x: float, y: float):
        """"Move branch point": the junction nearest to the click moves there, with EVERY crack of
        that fork (see core.crack_skeleton.move_branch_point)."""
        before = {i: copy.deepcopy(c) for i, c in enumerate(self._cracks)}
        try:
            info = move_branch_point(self._cracks, x, y, prefer=self._active_crack_index)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self._branch_move_undo.append(list(before.items()))
        self.undo_branch_button.setEnabled(True)
        self._active_node_index = None
        self._refresh_list()
        self._redraw()
        names = ", ".join(f"\"{self._cracks[i].name}\"" for i in info["moved"])
        self.status_label.setText(
            f"Branch point at ({info['old'][0]:.0f}, {info['old'][1]:.0f}) moved to "
            f"({info['new'][0]:.0f}, {info['new'][1]:.0f}) -- cracks re-made: {names}. "
            "Click again to adjust another one, or press the button to stop."
        )
        self.cracksChanged.emit()

    def _apply_path_correction(self, x: float, y: float):
        """"Correct path": the crack NEAREST to the click (the selected one when it is about as near)
        is re-made through the clicked point, and becomes the selected crack."""
        index, why = choose_crack_for_click(self._cracks, x, y, selected=self._active_crack_index)
        if index is None:
            self.status_label.setText(why)
            return
        before = [(i, copy.deepcopy(c)) for i, c in enumerate(self._cracks)]
        try:
            info = reroute_crack_through_point(self._cracks, index, x, y)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self._branch_move_undo.append(before)
        self.undo_branch_button.setEnabled(True)
        self._active_crack_index = index                 # the corrected crack is now the selected one
        self._active_node_index = None
        self._refresh_list()
        self._redraw()
        crack = self._cracks[index]
        extra = (f" ({len(info['snapped'])} attached branch(es) followed)" if info.get("snapped") else "")
        picked = "the selected crack" if why == "selected" else "the crack nearest to the click"
        self.status_label.setText(
            f"\"{crack.name}\" ({picked}) now passes through ({x:.0f}, {y:.0f}) and its path was "
            f"updated{extra}. Click again to correct more, or press the button to stop. "
            "Wrong crack? Press \"Undo manual correction\"."
        )
        self.cracksChanged.emit()

    def _on_undo_branch_move(self):
        if not self._branch_move_undo:
            return
        for i, crack in self._branch_move_undo.pop():
            if 0 <= i < len(self._cracks):
                self._cracks[i] = crack
        self.undo_branch_button.setEnabled(bool(self._branch_move_undo))
        self._refresh_list()
        self._redraw()
        self.status_label.setText("Last manual correction undone.")
        self.cracksChanged.emit()

    def _on_left_click(self, x: float, y: float):
        if self._manual_mode == "branch":
            self._apply_branch_move(x, y)
            return
        if self._manual_mode == "path":
            self._apply_path_correction(x, y)
            return
        if self._active_crack_index is None:
            self._start_new_crack()
        crack = self._active_crack()
        if crack is None:
            return
        new_index = crack.add_node(x, y, self._active_node_index)
        self._active_node_index = new_index
        self._redraw()
        self._refresh_list()
        self._update_status()
        self.cracksChanged.emit()

    def _on_right_click(self, x: float, y: float):
        crack = self._active_crack()
        if crack is None or not crack.nodes:
            return
        nearest = self._nearest_node(crack, x, y)
        if nearest is None:
            if self._pick_role is not None:
                self.status_label.setText(
                    "No point close enough to that click -- try again nearer an existing node."
                )
            return
        if self._pick_role == "start":
            crack.mark_initiation(nearest)
            self._pick_role = None
            self.mark_start_button.setChecked(False)
            self._active_node_index = nearest
            self._redraw()
            self._refresh_list()
            self.status_label.setText(
                f"\"{crack.name}\": marked node {nearest} as the crack's initiation point."
            )
            self.cracksChanged.emit()
        elif self._pick_role == "end":
            crack.mark_end(nearest)
            self._pick_role = None
            self.mark_end_button.setChecked(False)
            self._active_node_index = nearest
            self._redraw()
            self._refresh_list()
            self.status_label.setText(
                f"\"{crack.name}\": marked node {nearest} as the crack's end (tip) point."
            )
            self.cracksChanged.emit()
        else:
            self._active_node_index = nearest
            self._redraw()
            self._update_status()

    def _nearest_node(self, crack: CrackPath, x: float, y: float) -> int | None:
        if not crack.nodes:
            return None
        pts = np.asarray(crack.nodes, dtype=float)
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        i = int(np.argmin(d))
        diag = 100.0
        if self._frame_shape is not None:
            h, w = self._frame_shape
            diag = float(np.hypot(h, w)) or 100.0
        if d[i] <= _NODE_SELECT_RADIUS_FRACTION * diag * 4:  # generous: a right-click is a coarse gesture
            return i
        return None

    # -- drawing --------------------------------------------------------------

    def _show_placeholder(self, message: str):
        self.canvas.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True,
                color=theme.TEXT_MUTED, fontsize=10, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.refresh()

    def _redraw(self):
        if hasattr(self, "correct_path_button"):
            self._update_manual_styles()          # the armed button follows the selected crack's colour
        if self._field is None and self._background is None:
            self._show_placeholder("Load data and pick a frame/field below to start tracing.")
            return

        self.canvas.clear()
        self._hover_artists, self._hover_key = [], None      # the figure was cleared: no preview left
        ax = self.canvas.figure.add_subplot(111)
        self._main_ax = ax                                    # the image axes (not the colour bar)

        if self._background is not None:
            cmap_bg = "gray" if self._background.ndim == 2 else None
            h, w = self._background.shape[:2]
            # Shift the background's displayed position by -(u, v) (this
            # frame's median displacement since the reference frame) so it
            # re-aligns with the field/crack overlay, which stay at the
            # fixed reference (x0, y0) grid -- see `set_frame`'s
            # `displacement_shift_xy` and the module docstring. Off when
            # the user unchecks "Align background to strain field".
            sx, sy = self._bg_shift_xy if self.align_background_checkbox.isChecked() else (0.0, 0.0)
            bg_extent = [-sx, w - sx, h - sy, -sy]
            ax.imshow(self._background, cmap=cmap_bg, origin="upper", extent=bg_extent, alpha=0.6)
            ax.set_xlim(bg_extent[0], bg_extent[1])
            ax.set_ylim(bg_extent[2], bg_extent[3])

        if self._field is not None and self._xs is not None and self._ys is not None and self._xs.size and self._ys.size:
            step_x = (self._xs[1] - self._xs[0]) if self._xs.size > 1 else 1.0
            step_y = (self._ys[1] - self._ys[0]) if self._ys.size > 1 else 1.0
            extent = [self._xs[0] - step_x / 2, self._xs[-1] + step_x / 2,
                      self._ys[-1] + step_y / 2, self._ys[0] - step_y / 2]
            masked = np.ma.masked_invalid(self._field)
            vmin, vmax = self._current_vmin_vmax()
            im = ax.imshow(masked, cmap="jet", origin="upper", extent=extent, alpha=0.75,
                            interpolation="nearest", vmin=vmin, vmax=vmax)
            self.canvas.figure.colorbar(im, ax=ax, label=self._field_label, fraction=0.046, pad=0.04)
            if self._background is None:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])

        # Both imshow calls above are in the same "x/y pixel coordinate"
        # units -- keep them visually square (1 px in x == 1 px in y)
        # instead of letting the axes box's own width/height stretch the
        # frame to fill it, which otherwise distorts the traced crack's
        # true shape/curvature relative to the background image.
        ax.set_aspect("equal", adjustable="box")

        for i, crack in enumerate(self._cracks):
            if not crack.nodes:
                continue
            pts = np.asarray(crack.nodes, dtype=float)
            is_active = (i == self._active_crack_index)
            for a, b in crack.edges:
                ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                        color=crack.color, linewidth=2.2 if is_active else 1.5,
                        alpha=1.0 if is_active else 0.7)
            ax.scatter(pts[:, 0], pts[:, 1], s=16, color=crack.color, zorder=5,
                       edgecolors="white", linewidths=0.5)
            if is_active and self._active_node_index is not None and self._active_node_index < len(pts):
                ax.scatter([pts[self._active_node_index, 0]], [pts[self._active_node_index, 1]],
                           s=90, facecolors="none", edgecolors=theme.SUCCESS, linewidths=2.0, zorder=6)
            # Marked initiation/end points (see "Mark crack start"/"Mark
            # crack tip") -- distinct star/diamond markers, drawn for
            # every crack (not just the active one) so a merge's inputs
            # stay visible while switching between cracks in the list.
            if crack.initiation_index is not None and crack.initiation_index < len(pts):
                ax.scatter([pts[crack.initiation_index, 0]], [pts[crack.initiation_index, 1]],
                           marker="*", s=200, facecolors="#2ecc71", edgecolors="black",
                           linewidths=1.0, zorder=7)
            if crack.end_index is not None and crack.end_index < len(pts):
                ax.scatter([pts[crack.end_index, 0]], [pts[crack.end_index, 1]],
                           marker="D", s=90, facecolors="#e74c3c", edgecolors="black",
                           linewidths=1.0, zorder=7)

        if self._manual_mode == "branch":
            # "Move branch point" armed: ring every branch point (where cracks meet) so it is clear
            # what a click will move -- the nearest ring.
            for jn in find_junctions(self._cracks):
                ax.scatter([jn["point"][0]], [jn["point"][1]], s=330, facecolors="none",
                           edgecolors="#ffffff", linewidths=2.2, zorder=8)
                ax.scatter([jn["point"][0]], [jn["point"][1]], s=330, facecolors="none",
                           edgecolors="#000000", linewidths=0.8, zorder=8)

        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        self.canvas.refresh()

    def _update_status(self):
        if self._pick_role is not None:
            crack = self._active_crack()
            label = "initiation point (green star)" if self._pick_role == "start" else "end/tip point (red diamond)"
            if crack is None:
                self.status_label.setText("Select a crack in the list first, then right-click a point to mark it.")
            else:
                self.status_label.setText(
                    f"Right-click a point on \"{crack.name}\" to mark it as its {label}."
                )
            return
        crack = self._active_crack()
        if crack is None:
            self.status_label.setText(
                f"{len(self._cracks)} crack(s) traced. Click \"New crack\" or left-click on the image to start one."
            )
            return
        tip = "none placed yet" if self._active_node_index is None else f"node {self._active_node_index}"
        marks = []
        if crack.initiation_index is not None:
            marks.append(f"start = node {crack.initiation_index}")
        if crack.end_index is not None:
            marks.append(f"tip = node {crack.end_index}")
        marks_note = f"  ({', '.join(marks)})" if marks else ""
        self.status_label.setText(
            f"Editing {crack.name}: {len(crack.nodes)} point(s), {crack.total_length():.1f} px.{marks_note} "
            f"Active point: {tip} -- left-click to extend from it, right-click an existing point to branch from there."
        )
