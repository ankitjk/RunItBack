#!/usr/bin/env python3
"""Scene Labeler - Video labeling tool for SSBU scene classification.

A PyQt5-based tool for labeling video segments with scene states and stage information.
Supports both local video files and YouTube URLs.

Usage:
    .venv/bin/python tools/scene_labeler.py <video_path_or_youtube_url> [--labels <labels.json>]

Examples:
    .venv/bin/python tools/scene_labeler.py path/to/video.mp4
    .venv/bin/python tools/scene_labeler.py "https://www.youtube.com/watch?v=VIDEO_ID"
    .venv/bin/python tools/scene_labeler.py "https://youtu.be/VIDEO_ID"

Labels are saved to data/scene_classifier/raw_labels/<video_id>.json unless
--labels is given.

Keyboard Shortcuts:
    0-9, R  : Start labeling with class (continuous mode - auto-extends until changed)
    Space   : Pause/Play
    Left    : Previous frame (or 1 second if playing)
    Right   : Next frame (or 1 second if playing)
    S       : Set a mark here; the next class key labels from the mark to the playhead
    E       : Close the open label here
    C       : Change the class of the target segment (then press a class key)
    K       : Cut the segment under the playhead in two here
    [ / ]   : Move the target segment's start / end to the playhead
    Delete  : Delete the target segment (Backspace too)
    Esc     : Cancel the open label, mark or change; clear the selection
    T       : Set stage for current game
    G       : Start the next game here (see Game Boundaries)
    Ctrl+S  : Save labels to file
    Ctrl+Z  : Undo the last change (labels, game boundary or stage)
    Q       : Quit

Labeling Workflow (Continuous Mode):
    1. Press 0-9 or R to start labeling with a class
    2. Navigate through the video (the label extends as you go)
    3. When the scene changes, press a different class key
    4. The previous label is automatically closed and a new one starts
    5. Press E to finalize the last label when done

Editing Labels:
    Labelling time that is already labelled replaces what was there, so two
    segments never cover the same frame. Times are stored per frame, not rounded.
    - Relabel a stretch: S at one end, move to the other end, press the class key.
    - The target segment is the one selected in the table (click a row; double-
      click jumps to it), or else the one under the playhead.
    - Edits are refused while a continuous label is open: E closes it, Esc drops it.
    Fields this tool does not use (e.g. needs_review in a draft) are kept on save.

Segment Strips (under the video slider):
    "Game" shows the whole current game; "10s" shows 10 seconds around the
    playhead, where single frames are visible. Segments are in their class
    colour; dark is unlabelled; hatched is needs_review; red cross-hatching
    is an overlap; the selected segment is outlined; the mark or open label
    is shaded up to the playhead (white line). Click to jump there and select the
    segment; hover for its class and times. The strips only display labels.
    The Class Distribution panel shows labelled seconds per class, not counts.

Game Boundaries:
    Games are contiguous: the first game starts at 0s, and each G press ends the
    current game and starts the next one at that moment, so every second of the
    video belongs to a game. Press G in the menus between two games - after the
    previous game's results screen appears, before the next game's entrance.
    The menus, entrances and countdown inside a game's span are labelled
    non_gameplay like any other non-gameplay frame. Training splits by game, so
    a boundary placed mid-gameplay would put one game's frames in two splits.

Scene Classes (full definitions and tie-breaks: docs/SCENE_CLASSES.md):
    0: non_gameplay      - Menus, loading, results, pauses; entrances and countdown before GO!
    1: both_grounded     - Both grounded within the stage
    2: one_airborne      - One in the air within the stage, the other grounded
    3: both_airborne     - Both in the air within the stage
    4: one_offstage      - One off either side or below the stage
    5: both_offstage     - Both off the sides or below the stage
    6: one_on_ledge      - One hanging on a ledge
    7: both_on_ledge     - Both hanging on a ledge
    8: one_knocked_down  - One knocked down or teching
    9: both_knocked_down - Both knocked down or teching
    R: respawning        - KO flash or star KO, angel platform, stock count,
                           GAME! or TIME! screen (until the wipe to results)
    When the players' states differ, the higher-ranked one wins:
    respawning > offstage > on_ledge > knocked_down > airborne > grounded.
"""

import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QEvent, QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_segments import (  # noqa: E402
    EditError,
    GameLabels,
    SegmentLabel,
    VideoLabels,
    delete,
    frame_time,
    move_edge,
    overlaps,
    overwrite,
    segment_at,
    set_class,
    split,
)

# =============================================================================
# YouTube Download Support
# =============================================================================

def is_youtube_url(url: str) -> bool:
    """Check if the given string is a YouTube URL."""
    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/[\w-]+',
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def extract_video_id(url: str) -> str | None:
    """Extract video ID from YouTube URL."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([\w-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_cache_path(url: str) -> Path:
    """Get the cache path for a YouTube video."""
    video_id = extract_video_id(url)
    if video_id:
        # Use video ID as filename
        cache_dir = Path(".video_cache")
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{video_id}.mp4"
    # Fallback to hash of URL
    url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
    cache_dir = Path(".video_cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{url_hash}.mp4"


class DownloadThread(QThread):
    """Thread for downloading YouTube videos."""

    progress = pyqtSignal(str)  # Status message
    finished_download = pyqtSignal(str)  # Path to downloaded file
    error = pyqtSignal(str)  # Error message

    def __init__(self, url: str, output_path: Path, parent=None):
        super().__init__(parent)
        self.url = url
        self.output_path = output_path

    def run(self):
        try:
            self.progress.emit("Starting download...")

            # Use yt-dlp to download
            cmd = [
                "yt-dlp",
                "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
                "-o", str(self.output_path),
                "--merge-output-format", "mp4",
                "--no-playlist",
                self.url,
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in process.stdout:
                line = line.strip()
                if line:
                    # Parse progress from yt-dlp output
                    if "[download]" in line and "%" in line:
                        self.progress.emit(line)
                    elif "Merging" in line or "Downloading" in line:
                        self.progress.emit(line)

            process.wait()

            if process.returncode != 0:
                self.error.emit(f"Download failed with code {process.returncode}")
                return

            if not self.output_path.exists():
                self.error.emit("Download completed but file not found")
                return

            self.finished_download.emit(str(self.output_path))

        except FileNotFoundError:
            self.error.emit("yt-dlp not found. Install with: pip install yt-dlp")
        except Exception as e:
            self.error.emit(str(e))


class SceneState(Enum):
    """Scene classification states."""
    NON_GAMEPLAY = "non_gameplay"  # Menus, loading, character select, results
    BOTH_GROUNDED = "both_grounded"
    ONE_AIRBORNE = "one_airborne"
    BOTH_AIRBORNE = "both_airborne"
    ONE_OFFSTAGE = "one_offstage"
    BOTH_OFFSTAGE = "both_offstage"
    ONE_ON_LEDGE = "one_on_ledge"
    BOTH_ON_LEDGE = "both_on_ledge"
    ONE_KNOCKED_DOWN = "one_knocked_down"
    BOTH_KNOCKED_DOWN = "both_knocked_down"
    RESPAWNING = "respawning"  # KO, respawn platform, stock count, or GAME! screen


SCENE_LABELS = {
    0: SceneState.NON_GAMEPLAY,
    1: SceneState.BOTH_GROUNDED,
    2: SceneState.ONE_AIRBORNE,
    3: SceneState.BOTH_AIRBORNE,
    4: SceneState.ONE_OFFSTAGE,
    5: SceneState.BOTH_OFFSTAGE,
    6: SceneState.ONE_ON_LEDGE,
    7: SceneState.BOTH_ON_LEDGE,
    8: SceneState.ONE_KNOCKED_DOWN,
    9: SceneState.BOTH_KNOCKED_DOWN,
    10: SceneState.RESPAWNING,
}

SCENE_COLORS = {
    SceneState.NON_GAMEPLAY: "#9E9E9E",       # Grey - not gameplay
    SceneState.BOTH_GROUNDED: "#4CAF50",      # Green
    SceneState.ONE_AIRBORNE: "#2196F3",       # Blue
    SceneState.BOTH_AIRBORNE: "#00BCD4",      # Cyan
    SceneState.ONE_OFFSTAGE: "#FF9800",       # Orange
    SceneState.BOTH_OFFSTAGE: "#FF5722",      # Deep Orange
    SceneState.ONE_ON_LEDGE: "#9C27B0",       # Purple
    SceneState.BOTH_ON_LEDGE: "#E91E63",      # Pink
    SceneState.ONE_KNOCKED_DOWN: "#795548",   # Brown
    SceneState.BOTH_KNOCKED_DOWN: "#607D8B",  # Blue Grey
    SceneState.RESPAWNING: "#FFEB3B",         # Yellow - respawn platform
}

STAGES = [
    "small_battlefield",
    "battlefield",
    "final_destination",
    "town_and_city",
    "smashville",
    "hollow_bastion",
    "kalos_pokemon_league",
    "pokemon_stadium_2",
]

STAGE_DISPLAY_NAMES = {
    "small_battlefield": "Small Battlefield",
    "battlefield": "Battlefield",
    "final_destination": "Final Destination",
    "town_and_city": "Town and City",
    "smashville": "Smashville",
    "hollow_bastion": "Hollow Bastion",
    "kalos_pokemon_league": "Kalos Pokemon League",
    "pokemon_stadium_2": "Pokemon Stadium 2",
}


class StageDialog(QDialog):
    """Dialog for selecting stage."""

    def __init__(self, current_stage: str = "unknown", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Stage")
        self.setModal(True)

        layout = QVBoxLayout(self)

        self.combo = QComboBox()
        self.combo.addItem("Unknown", "unknown")
        for stage in STAGES:
            self.combo.addItem(STAGE_DISPLAY_NAMES.get(stage, stage), stage)

        # Set current selection
        index = self.combo.findData(current_stage)
        if index >= 0:
            self.combo.setCurrentIndex(index)

        layout.addWidget(QLabel("Select the stage for this game:"))
        layout.addWidget(self.combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_stage(self) -> str:
        """Get selected stage."""
        return self.combo.currentData()


class VideoPlayer(QWidget):
    """Video player widget with frame-accurate seeking."""

    frame_changed = pyqtSignal(int, float)  # frame_num, timestamp

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap: cv2.VideoCapture | None = None
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame = 0
        self.playing = False
        self.playback_speed = 1.0
        self._current_frame_data: np.ndarray | None = None  # Cache for resizing

        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Video display - scales with window
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 180)  # Smaller minimum to allow shrinking
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        layout.addWidget(self.video_label, stretch=1)

        # Timeline slider
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setMinimum(0)
        self.timeline.valueChanged.connect(self._on_timeline_change)
        layout.addWidget(self.timeline)

        # Controls
        controls = QHBoxLayout()

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_btn)

        self.time_label = QLabel("0:00.00 / 0:00.00")
        self.time_label.setMinimumWidth(240)
        controls.addWidget(self.time_label)

        controls.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "1.5x", "2x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_change)
        controls.addWidget(self.speed_combo)

        controls.addStretch()
        layout.addLayout(controls)

    def _setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._advance_frame)

    def load_video(self, path: str) -> bool:
        """Load a video file."""
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            return False

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.timeline.setMaximum(self.total_frames - 1)
        self.current_frame = 0

        self._display_frame()
        return True

    def _display_frame(self):
        """Display the current frame."""
        if self.cap is None:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert to RGB and cache for resize events
        self._current_frame_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._update_display()

        # Update timeline (block signals to prevent recursion)
        self.timeline.blockSignals(True)
        self.timeline.setValue(self.current_frame)
        self.timeline.blockSignals(False)

        # Update time label
        current_time = self.current_frame / self.fps
        total_time = self.total_frames / self.fps
        self.time_label.setText(
            f"{self._format_time(current_time)} / {self._format_time(total_time)}"
            f"   frame {self.current_frame}"
        )

        self.frame_changed.emit(self.current_frame, current_time)

    def _update_display(self):
        """Update the video display with the cached frame (used for resizing)."""
        if self._current_frame_data is None:
            return

        h, w, ch = self._current_frame_data.shape
        bytes_per_line = ch * w
        image = QImage(
            self._current_frame_data.data, w, h, bytes_per_line, QImage.Format_RGB888
        )

        # Scale to fit label size
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def resizeEvent(self, event):  # noqa: N802 - Qt override name
        """Handle resize events to scale video with window."""
        super().resizeEvent(event)
        # Re-render the cached frame at the new size
        if self._current_frame_data is not None:
            self._update_display()

    def _format_time(self, seconds: float) -> str:
        """Format seconds as M:SS.ss"""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}:{secs:05.2f}"

    def _on_timeline_change(self, value: int):
        """Handle timeline slider change."""
        if not self.playing:
            self.current_frame = value
            self._display_frame()

    def _on_speed_change(self, speed_text: str):
        """Handle speed change."""
        speed_map = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "1.5x": 1.5, "2x": 2.0}
        self.playback_speed = speed_map.get(speed_text, 1.0)
        if self.playing:
            self._update_timer_interval()

    def _update_timer_interval(self):
        """Update timer interval based on speed."""
        interval = int(1000 / (self.fps * self.playback_speed))
        self.timer.setInterval(max(1, interval))

    def toggle_play(self):
        """Toggle play/pause."""
        if self.playing:
            self.pause()
        else:
            self.play()

    def play(self):
        """Start playback."""
        self.playing = True
        self.play_btn.setText("Pause")
        self._update_timer_interval()
        self.timer.start()

    def pause(self):
        """Pause playback."""
        self.playing = False
        self.play_btn.setText("Play")
        self.timer.stop()

    def _advance_frame(self):
        """Advance to next frame during playback."""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self._display_frame()
        else:
            self.pause()

    def seek_frame(self, frame: int):
        """Seek to specific frame."""
        self.current_frame = max(0, min(frame, self.total_frames - 1))
        self._display_frame()

    def seek_time(self, seconds: float):
        """Seek to the frame a stored time names (stored times are rounded, so round back)."""
        self.seek_frame(round(seconds * self.fps))

    def step_forward(self):
        """Step forward one frame."""
        self.seek_frame(self.current_frame + 1)

    def step_backward(self):
        """Step backward one frame."""
        self.seek_frame(self.current_frame - 1)

    def skip_forward(self, seconds: float = 1.0):
        """Skip forward by seconds."""
        self.seek_frame(self.current_frame + int(seconds * self.fps))

    def skip_backward(self, seconds: float = 1.0):
        """Skip backward by seconds."""
        self.seek_frame(self.current_frame - int(seconds * self.fps))

    def get_current_time(self) -> float:
        """Get current timestamp in seconds."""
        return self.current_frame / self.fps

    def get_duration(self) -> float:
        """Get video duration in seconds."""
        return self.total_frames / self.fps

    def cleanup(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()
            self.cap = None


class LabelPanel(QWidget):
    """Panel for displaying and managing labels."""

    label_selected = pyqtSignal(float)  # timestamp to seek to
    selection_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Current game info
        game_frame = QFrame()
        game_frame.setFrameStyle(QFrame.StyledPanel)
        game_layout = QVBoxLayout(game_frame)

        self.game_label = QLabel("Game 1")
        self.game_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        game_layout.addWidget(self.game_label)

        self.stage_label = QLabel("Stage: Unknown")
        game_layout.addWidget(self.stage_label)

        layout.addWidget(game_frame)

        # Current segment info
        segment_frame = QFrame()
        segment_frame.setFrameStyle(QFrame.StyledPanel)
        segment_layout = QVBoxLayout(segment_frame)

        self.segment_start_label = QLabel("Segment Start: --")
        segment_layout.addWidget(self.segment_start_label)

        self.segment_class_label = QLabel("Class: (press 0-9 or R to label)")
        segment_layout.addWidget(self.segment_class_label)

        layout.addWidget(segment_frame)

        # Labels table
        layout.addWidget(QLabel("Labels:"))
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Start", "End", "Class"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        # Cells are not editable text: edits go through the keys, which keep segments consistent
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(self._on_cell_double_click)
        self.table.itemSelectionChanged.connect(self.selection_changed.emit)
        layout.addWidget(self.table)

        # Stats
        self.stats_label = QLabel("Total: 0 labels")
        layout.addWidget(self.stats_label)

    def update_game_info(self, game: GameLabels):
        """Update game info display."""
        self.game_label.setText(f"Game {game.game_number}")
        stage_name = STAGE_DISPLAY_NAMES.get(game.stage, game.stage)
        self.stage_label.setText(f"Stage: {stage_name}")

    def update_segment_start(self, time_sec: float, mark: bool = False):
        """Show where the open label starts, or where the mark is."""
        if mark:
            self.segment_start_label.setText(f"Mark: {time_sec:.3f}s")
            self.segment_class_label.setText("Class: (press a class key to label mark to playhead)")
        else:
            self.segment_start_label.setText(f"Segment Start: {time_sec:.3f}s")

    def clear_segment_start(self):
        """Clear segment start display."""
        self.segment_start_label.setText("Segment Start: --")
        self.segment_class_label.setText("Class: (press 0-9 or R to label)")

    def update_segment_class(self, scene_class: SceneState):
        """Update segment class display."""
        color = SCENE_COLORS.get(scene_class, "#888888")
        self.segment_class_label.setText(
            f'Class: <span style="color:{color};">{scene_class.value}</span>'
        )

    def selected_index(self) -> int | None:
        """Row selected in the table (rows are the game's segments in order), or None."""
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def select_row(self, row: int | None):
        """Select a row and scroll to it, or clear the selection."""
        if row is None or not 0 <= row < self.table.rowCount():
            self.table.clearSelection()
            return
        self.table.selectRow(row)
        self.table.scrollToItem(self.table.item(row, 0))

    def update_labels(self, labels: list[SegmentLabel]):
        """Update labels table. Clears the selection: rows may have moved."""
        self.table.blockSignals(True)
        self.table.clearSelection()
        self.table.setRowCount(len(labels))
        self.table.blockSignals(False)
        for i, label in enumerate(labels):
            self.table.setItem(i, 0, QTableWidgetItem(f"{label.start_sec:.3f}s"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{label.end_sec:.3f}s"))

            class_item = QTableWidgetItem(label.scene_class)
            class_item.setBackground(QColor(CLASS_COLORS.get(label.scene_class, UNKNOWN_CLASS_COLOR)))
            class_item.setForeground(QColor("black"))
            self.table.setItem(i, 2, class_item)

        self.stats_label.setText(f"Total: {len(labels)} labels")

    def _on_cell_double_click(self, row: int, col: int):
        """Handle double-click on table cell."""
        item = self.table.item(row, 0)
        if item:
            time_text = item.text().rstrip("s")
            try:
                timestamp = float(time_text)
                self.label_selected.emit(timestamp)
            except ValueError:
                pass


CLASS_COLORS = {state.value: color for state, color in SCENE_COLORS.items()}
UNKNOWN_CLASS_COLOR = "#888888"
UNLABELLED_COLOR = "#1e1e1e"
OUTSIDE_GAME_COLOR = "#000000"
PLAYHEAD_COLOR = "#FFFFFF"
OVERLAP_COLOR = "#FF1744"
CLOSEUP_SECS = 10.0


class LegendPanel(QWidget):
    """Panel showing keyboard shortcuts and scene class legend."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        # Two columns each, so the legend stays short enough to leave room for the
        # labels table; the window also puts it in a scroll area.
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(QLabel("<b>Scene Classes (0-9, R):</b>"))
        class_info = [
            ("0", "non_gameplay", "Menus, results, pre-GO!"),
            ("1", "both_grounded", "Both grounded"),
            ("2", "one_airborne", "One in the air"),
            ("3", "both_airborne", "Both in the air"),
            ("4", "one_offstage", "One offstage"),
            ("5", "both_offstage", "Both offstage"),
            ("6", "one_on_ledge", "One on a ledge"),
            ("7", "both_on_ledge", "Both on a ledge"),
            ("8", "one_knocked_down", "One knocked down"),
            ("9", "both_knocked_down", "Both knocked down"),
            ("R", "respawning", "KO, angel platform"),
        ]
        classes = QGridLayout()
        classes.setHorizontalSpacing(6)
        classes.setVerticalSpacing(1)
        half = (len(class_info) + 1) // 2
        for i, (key, class_name, description) in enumerate(class_info):
            row, col = i % half, (i // half) * 3
            key_label = QLabel(f"<b>{key}</b>")
            key_label.setFixedWidth(14)
            classes.addWidget(key_label, row, col)
            color_box = QFrame()
            color_box.setFixedSize(12, 12)
            color_box.setStyleSheet(
                f"background-color: {CLASS_COLORS.get(class_name, UNKNOWN_CLASS_COLOR)}; border-radius: 2px;"
            )
            classes.addWidget(color_box, row, col + 1)
            desc_label = QLabel(f"<small>{description}</small>")
            desc_label.setToolTip(class_name)
            classes.addWidget(desc_label, row, col + 2)
        layout.addLayout(classes)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        layout.addWidget(QLabel("<b>Controls:</b>"))
        shortcuts = [
            ("Space", "Play / pause"),
            ("\u2190 \u2192", "Prev / next frame"),
            ("S", "Mark; class key fills to here"),
            ("E", "Close open label here"),
            ("C", "Change class of target"),
            ("K", "Cut segment here"),
            ("[ ]", "Move start / end here"),
            ("Delete", "Delete target"),
            ("Esc", "Cancel; clear selection"),
            ("G", "Next game starts here"),
            ("T", "Set stage"),
            ("Ctrl+S", "Save"),
            ("Ctrl+Z", "Undo last change"),
            ("Q", "Quit"),
        ]
        controls = QGridLayout()
        controls.setHorizontalSpacing(6)
        controls.setVerticalSpacing(1)
        half = (len(shortcuts) + 1) // 2
        for i, (key, action) in enumerate(shortcuts):
            row, col = i % half, (i // half) * 2
            controls.addWidget(QLabel(f"<small><b>{key}</b></small>"), row, col)
            controls.addWidget(QLabel(f"<small>{action}</small>"), row, col + 1)
        layout.addLayout(controls)

        strips = QLabel(
            "<small><b>Strips:</b> click to jump and select. Hatched = needs review, "
            "red cross-hatch = overlap, dark = unlabelled.</small>"
        )
        strips.setWordWrap(True)
        layout.addWidget(strips)
        layout.addStretch()


class ClassDistributionPanel(QWidget):
    """Panel showing class distribution statistics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(4, 4, 4, 4)
        # Time, not segment count: training frames come from time, and one 20s
        # segment would otherwise weigh the same as a 0.2s one.
        layout.addWidget(QLabel("<b>Class Distribution (seconds, this game):</b>"))

        self.bars = {}
        for state in SceneState:
            row = QHBoxLayout()

            # Class name
            name_label = QLabel(f"<small>{state.value}:</small>")
            name_label.setMinimumWidth(120)
            row.addWidget(name_label)

            # Progress bar (using QFrame)
            bar = QFrame()
            bar.setFixedHeight(10)
            bar.setStyleSheet(
                f"background-color: {SCENE_COLORS.get(state, '#888')}; "
                "border-radius: 3px;"
            )
            bar.setMinimumWidth(0)
            bar.setMaximumWidth(200)
            row.addWidget(bar)

            # Count label
            count_label = QLabel("0")
            count_label.setMinimumWidth(50)
            row.addWidget(count_label)

            row.addStretch()
            layout.addLayout(row)
            self.bars[state] = (bar, count_label)

        layout.addStretch()

    def update_distribution(self, labels: list[SegmentLabel]):
        """Show the labelled seconds per class."""
        seconds = {state: 0.0 for state in SceneState}
        by_value = {state.value: state for state in SceneState}
        for label in labels:
            state = by_value.get(label.scene_class)
            if state is not None:
                seconds[state] += label.end_sec - label.start_sec

        total = sum(seconds.values()) or 1.0
        for state, (bar, count_label) in self.bars.items():
            secs = seconds[state]
            bar.setFixedWidth(max(1, int(secs / total * 200)) if secs > 0 else 0)
            count_label.setText(f"{secs:.1f}s")




class SegmentStrip(QWidget):
    """A coloured bar of the current game's segments, for seeing labels at a glance.

    window_secs None draws the whole game; a number draws that many seconds
    centred on the playhead, where single frames are wide enough to see.
    Drawn: segments in their class colour, unlabelled time dark, needs_review
    hatched, overlaps red cross-hatched (at least 6px wide), the selected segment
    outlined, the mark or open label shaded up to the playhead, and the
    playhead as a white line. Clicking emits the time and the segment under it.
    Display only: it never changes labels, and it never takes keyboard focus.
    """

    clicked = pyqtSignal(float, object)  # time, row index under the click or None

    def __init__(self, window_secs: float | None = None, parent=None):
        super().__init__(parent)
        self.window_secs = window_secs
        self.labels: list[SegmentLabel] = []
        self.game_start = 0.0
        self.game_end = 1.0
        self.playhead = 0.0
        self.pending_start: float | None = None
        self.pending_class: str | None = None
        self.selected: int | None = None
        self._overlaps_of: list[SegmentLabel] | None = None
        self._overlap_spans: list[tuple[float, float]] = []
        self.setFixedHeight(18 if window_secs is None else 28)
        self.setFocusPolicy(Qt.NoFocus)
        self.setMinimumWidth(200)

    def set_state(self, labels, game_start, game_end, playhead, pending_start, pending_class, selected):
        if labels is not self._overlaps_of:  # edits make a new list, so this is a cheap cache
            self._overlaps_of = labels
            self._overlap_spans = [(b.start_sec, min(a.end_sec, b.end_sec)) for a, b in overlaps(labels)]
        self.labels = labels
        self.game_start = game_start
        self.game_end = game_end if game_end > game_start else game_start + 1.0
        self.playhead = playhead
        self.pending_start = pending_start
        self.pending_class = pending_class
        self.selected = selected
        self.update()

    def time_range(self) -> tuple[float, float]:
        if self.window_secs is None:
            return self.game_start, self.game_end
        return self.playhead - self.window_secs / 2, self.playhead + self.window_secs / 2

    def x_of(self, t: float) -> float:
        lo, hi = self.time_range()
        return (t - lo) / (hi - lo) * self.width()

    def time_at(self, x: float) -> float:
        lo, hi = self.time_range()
        return lo + x / max(1, self.width()) * (hi - lo)

    def _span(self, a: float, b: float, min_width: float = 1.0) -> QRectF:
        x0, x1 = self.x_of(a), self.x_of(b)
        return QRectF(x0, 0, max(min_width, x1 - x0), self.height())

    def _rows_at(self, t: float) -> list[int]:
        return [i for i, s in enumerate(self.labels) if s.start_sec <= t < s.end_sec]

    def paintEvent(self, event):  # noqa: N802 - Qt override name
        p = QPainter(self)
        w, h = self.width(), self.height()
        lo, hi = self.time_range()
        p.fillRect(0, 0, w, h, QColor(UNLABELLED_COLOR))
        for a, b in ((lo, self.game_start), (self.game_end, hi)):
            if b > a:
                p.fillRect(self._span(a, b), QColor(OUTSIDE_GAME_COLOR))

        hatch = QBrush(QColor(0, 0, 0, 150), Qt.BDiagPattern)
        visible = [s for s in self.labels if s.end_sec > lo and s.start_sec < hi]
        for s in visible:
            rect = self._span(max(s.start_sec, lo), min(s.end_sec, hi))
            p.fillRect(rect, QColor(CLASS_COLORS.get(s.scene_class, UNKNOWN_CLASS_COLOR)))
            if s.extra.get("needs_review"):
                p.fillRect(rect, hatch)

        if self.window_secs is not None:
            # Edges, so two touching segments of one class still read as two
            p.setPen(QPen(QColor(0, 0, 0, 170), 1))
            for s in visible:
                x = self.x_of(s.start_sec)
                p.drawLine(QPointF(x, 0), QPointF(x, h))
            # A tick every second
            p.setPen(QPen(QColor(255, 255, 255, 120), 1))
            t = math.ceil(lo)
            while t <= hi:
                x = self.x_of(t)
                p.drawLine(QPointF(x, h - 5), QPointF(x, h))
                t += 1

        # Overlaps: red cross-hatch over the full height, with solid red top and
        # bottom edges, at least 6px wide so a 0.1s overlap shows on the game strip
        cross = QBrush(QColor(OVERLAP_COLOR), Qt.DiagCrossPattern)
        for a, b in self._overlap_spans:
            if b > lo and a < hi:
                x0, x1 = self.x_of(a), self.x_of(b)
                width = max(6.0, x1 - x0)
                left = (x0 + x1) / 2 - width / 2
                p.fillRect(QRectF(left, 0, width, h), QColor("black"))  # contrast on any class colour
                p.fillRect(QRectF(left, 0, width, h), cross)
                p.fillRect(QRectF(left, 0, width, 3), QColor(OVERLAP_COLOR))
                p.fillRect(QRectF(left, h - 3, width, 3), QColor(OVERLAP_COLOR))

        if self.pending_start is not None:
            a, b = sorted((self.pending_start, self.playhead))
            shade = QColor(CLASS_COLORS.get(self.pending_class, "#FFFFFF") if self.pending_class else "#FFFFFF")
            shade.setAlpha(150 if self.pending_class else 90)
            p.fillRect(self._span(a, b), shade)
            p.setPen(QPen(QColor("white"), 1, Qt.DashLine))
            x = self.x_of(self.pending_start)
            p.drawLine(QPointF(x, 0), QPointF(x, h))

        if self.selected is not None and 0 <= self.selected < len(self.labels):
            s = self.labels[self.selected]
            if s.end_sec > lo and s.start_sec < hi:
                rect = self._span(max(s.start_sec, lo), min(s.end_sec, hi), min_width=3.0)
                p.setPen(QPen(QColor("white"), 2))
                p.setBrush(Qt.NoBrush)
                p.drawRect(rect.adjusted(1, 1, -1, -1))

        # White with a black outline, so it shows on every class colour
        x = self.x_of(self.playhead)
        for colour, width in (("black", 4), (PLAYHEAD_COLOR, 2)):
            p.setPen(QPen(QColor(colour), width))
            p.drawLine(QPointF(x, 0), QPointF(x, h))
        p.end()

    def mousePressEvent(self, event):  # noqa: N802 - Qt override name
        if event.button() != Qt.LeftButton:
            return
        t = self.time_at(event.pos().x())
        rows = self._rows_at(t)
        self.clicked.emit(t, rows[0] if rows else None)

    def event(self, e):
        if e.type() == QEvent.ToolTip:
            t = self.time_at(e.pos().x())
            hits = [self.labels[i] for i in self._rows_at(t)]
            parts = [s.describe() + (" (needs review)" if s.extra.get("needs_review") else "") for s in hits]
            text = f"{t:.3f}s: " + ("; ".join(parts) if parts else "unlabelled")
            if len(hits) > 1:
                text += "  - OVERLAP"
            QToolTip.showText(e.globalPos(), text, self)
            return True
        return super().event(e)


class SceneLabeler(QMainWindow):
    """Main labeling application window."""

    def __init__(self, video_path: str, labels_path: str | None = None):
        super().__init__()
        self.video_path = Path(video_path)
        self.labels_path = Path(labels_path) if labels_path else None

        self.video_labels: VideoLabels | None = None
        self.current_game_index = 0
        # segment_start with current_class set: an open continuous label.
        # segment_start with current_class None: a mark set with S.
        self.segment_start: float | None = None
        self.current_class: SceneState | None = None
        self.change_target: int | None = None  # C pressed: the next class key relabels this row
        self._saved_state: str | None = None  # labels as last loaded or saved
        # Each entry: (game index, copy of every game) from before a change
        self.undo_stack: list[tuple[int, list[GameLabels]]] = []

        self._setup_ui()
        self._setup_shortcuts()
        self._load_video()
        self._load_or_create_labels()

    def _setup_ui(self):
        self.setWindowTitle(f"Scene Labeler - {self.video_path.name}")
        self.setMinimumSize(1200, 700)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left side: video player
        self.player = VideoPlayer()
        self.player.frame_changed.connect(self._on_frame_changed)
        main_layout.addWidget(self.player, stretch=2)

        # Segment strips, under the video's slider: the whole game, then a close-up
        self.overview_strip = SegmentStrip()
        self.closeup_strip = SegmentStrip(window_secs=CLOSEUP_SECS)
        for position, (caption, strip) in enumerate(
            (("Game", self.overview_strip), (f"{CLOSEUP_SECS:.0f}s", self.closeup_strip)), start=2
        ):
            row = QHBoxLayout()
            label = QLabel(f"<small>{caption}</small>")
            label.setFixedWidth(34)
            row.addWidget(label)
            row.addWidget(strip, stretch=1)
            strip.clicked.connect(self._on_strip_clicked)
            self.player.layout().insertLayout(position, row)

        # Right side: panels
        right_panel = QSplitter(Qt.Vertical)

        # Legend panel (top)
        self.legend_panel = LegendPanel()
        legend_scroll = QScrollArea()
        legend_scroll.setWidget(self.legend_panel)
        legend_scroll.setWidgetResizable(True)
        legend_scroll.setFrameShape(QFrame.NoFrame)
        right_panel.addWidget(legend_scroll)

        self.label_panel = LabelPanel()
        self.label_panel.label_selected.connect(self.player.seek_time)
        self.label_panel.selection_changed.connect(self._on_selection_changed)
        right_panel.addWidget(self.label_panel)

        self.dist_panel = ClassDistributionPanel()
        dist_scroll = QScrollArea()
        dist_scroll.setWidget(self.dist_panel)
        dist_scroll.setWidgetResizable(True)
        dist_scroll.setFrameShape(QFrame.NoFrame)
        right_panel.addWidget(dist_scroll)

        # Set initial sizes (legend smaller, labels bigger)
        right_panel.setSizes([230, 470, 200])

        main_layout.addWidget(right_panel, stretch=1)

        # Status bar
        self.statusBar().showMessage("Ready. Press 0-9 or R to start labeling (continuous mode).")

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Playback
        QShortcut(Qt.Key_Space, self, self.player.toggle_play)
        QShortcut(Qt.Key_Left, self, self._on_left_key)
        QShortcut(Qt.Key_Right, self, self._on_right_key)

        # Segment marking
        QShortcut(Qt.Key_S, self, self._mark_segment_start)
        QShortcut(Qt.Key_E, self, self._mark_segment_end)

        # Class assignment (0-9, R for respawning)
        # 0 = non_gameplay, 1-9 = gameplay scene states, R = respawning
        for i in range(0, 10):
            QShortcut(getattr(Qt, f"Key_{i}"), self, lambda x=i: self._assign_class(x))
        QShortcut(Qt.Key_R, self, lambda: self._assign_class(10))  # R = respawning

        # Editing existing segments
        QShortcut(Qt.Key_C, self, self._start_change_class)
        QShortcut(Qt.Key_K, self, self._split_at_playhead)
        QShortcut(Qt.Key_BracketLeft, self, lambda: self._move_edge("start"))
        QShortcut(Qt.Key_BracketRight, self, lambda: self._move_edge("end"))
        QShortcut(Qt.Key_Delete, self, self._delete_target)
        QShortcut(Qt.Key_Backspace, self, self._delete_target)
        QShortcut(Qt.Key_Escape, self, self._cancel)

        # Stage and game
        QShortcut(Qt.Key_T, self, self._set_stage)
        QShortcut(Qt.Key_G, self, self._mark_game_boundary)

        # File operations
        QShortcut(QKeySequence.Save, self, self._save_labels)
        QShortcut(QKeySequence.Undo, self, self._undo_last)
        QShortcut(Qt.Key_Q, self, self.close)

    def _load_video(self):
        """Load the video file."""
        if not self.player.load_video(str(self.video_path)):
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load video: {self.video_path}",
            )
            sys.exit(1)

    def _load_or_create_labels(self):
        """Load existing labels or create new."""
        if self.labels_path and self.labels_path.exists():
            with open(self.labels_path) as f:
                data = json.load(f)
                self.video_labels = VideoLabels.from_dict(data)
                self.statusBar().showMessage(f"Loaded {len(self._current_game.labels)} labels")
            self._warn_overlaps()
        else:
            # Create new labels
            video_id = self.video_path.stem
            self.video_labels = VideoLabels(
                video_id=video_id,
                video_path=str(self.video_path),
                games=[
                    GameLabels(
                        game_number=1,
                        start_sec=0.0,
                        end_sec=self.player.get_duration(),
                    )
                ],
            )
            # Default labels path
            if not self.labels_path:
                labels_dir = Path("data/scene_classifier/raw_labels")
                labels_dir.mkdir(parents=True, exist_ok=True)
                self.labels_path = labels_dir / f"{video_id}.json"

        self._saved_state = self._label_state()
        self._update_ui()

    def _label_state(self) -> str:
        """The labels as they would be saved, for comparing with the last save."""
        return json.dumps(self.video_labels.to_dict(), sort_keys=True)

    def _has_unsaved_changes(self) -> bool:
        return self._label_state() != self._saved_state

    def _update_title(self):
        mark = " *" if self._has_unsaved_changes() else ""
        self.setWindowTitle(f"Scene Labeler - {self.video_path.name}{mark}")

    @property
    def _current_game(self) -> GameLabels:
        """Get current game."""
        return self.video_labels.games[self.current_game_index]

    def _update_ui(self, select_at: float | None = None):
        """Update all UI elements; select the segment covering select_at, if given."""
        game = self._current_game
        self.label_panel.update_game_info(game)
        self.label_panel.update_labels(game.labels)
        self.dist_panel.update_distribution(game.labels)
        self._update_title()
        if select_at is not None:
            try:
                self.label_panel.select_row(segment_at(game.labels, select_at))
            except EditError:
                pass
        self._refresh_strips()

    def _refresh_strips(self):
        """Redraw both segment strips from the current game, playhead, mark and selection."""
        if self.video_labels is None:
            return
        game = self._current_game
        game_end = game.end_sec if game.end_sec > game.start_sec else self.player.get_duration()
        for strip in (self.overview_strip, self.closeup_strip):
            strip.set_state(
                game.labels, game.start_sec, game_end, self.player.get_current_time(),
                self.segment_start, self.current_class.value if self.current_class else None,
                self.label_panel.selected_index(),
            )

    def _on_strip_clicked(self, t: float, row: int | None):
        """A click on a strip: jump there, and select the segment clicked (if still in this game)."""
        game_index = self.current_game_index
        self.player.seek_time(t)
        if self.current_game_index == game_index:
            self.label_panel.select_row(row)
        self._refresh_strips()

    def _warn_overlaps(self):
        """Say where a loaded file has segments covering the same time. Nothing is changed."""
        found = [
            (game.game_number, a, b)
            for game in self.video_labels.games
            for a, b in overlaps(game.labels)
        ]
        if not found:
            return
        for game_number, a, b in found:
            print(f"Overlap in game {game_number}: {a.describe()} and {b.describe()}")
        game_number, a, b = found[0]
        self.statusBar().showMessage(
            f"Warning: {len(found)} overlapping segment pair(s), first in game {game_number} at "
            f"{b.start_sec:.3f}s. Overwrite the overlap to fix it (details in the terminal)."
        )

    def _on_frame_changed(self, frame: int, timestamp: float):
        """Handle frame change - auto-switch to correct game based on timestamp."""
        if self.video_labels is None:
            return

        # Find which game this timestamp belongs to
        for i, game in enumerate(self.video_labels.games):
            # Check if timestamp falls within this game's range
            # Use end_sec if set, otherwise assume game extends to end of video
            game_end = game.end_sec if game.end_sec > 0 else float('inf')
            if game.start_sec <= timestamp < game_end:
                if i != self.current_game_index:
                    self.current_game_index = i
                    self.change_target = None  # it named a row of the other game
                    self._update_ui()
                    self.statusBar().showMessage(
                        f"Switched to Game {game.game_number} ({game.stage})"
                    )
                break
        self._refresh_strips()

    def _on_left_key(self):
        """Handle left arrow key."""
        if self.player.playing:
            self.player.skip_backward(1.0)
        else:
            self.player.step_backward()

    def _on_right_key(self):
        """Handle right arrow key."""
        if self.player.playing:
            self.player.skip_forward(1.0)
        else:
            self.player.step_forward()

    # ------------------------------------------------------------------
    # Editing: every change goes through _apply, which records undo
    # ------------------------------------------------------------------

    def _now(self) -> float:
        """The playhead's time, as it is stored."""
        return frame_time(self.player.current_frame, self.player.fps)

    def _snapshot(self):
        """Remember the games as they are now, for Ctrl+Z."""
        self.undo_stack.append((self.current_game_index, copy.deepcopy(self.video_labels.games)))

    def _apply(
        self,
        edit: Callable[[list[SegmentLabel]], list[SegmentLabel]],
        message: str,
        select_at: float | None = None,
    ) -> bool:
        """Run an edit on the current game's segments. A refused edit changes nothing."""
        game = self._current_game
        try:
            new_labels = edit(game.labels)
        except EditError as e:
            self.statusBar().showMessage(str(e))
            return False
        self._snapshot()
        game.labels = new_labels
        self._update_ui(select_at=select_at)
        self.statusBar().showMessage(message)
        return True

    def _label_open(self) -> bool:
        """True (and says so) if a continuous label is open, which blocks other edits."""
        if self.current_class is None:
            return False
        self.statusBar().showMessage(
            f"A {self.current_class.value} label is open from {self.segment_start:.3f}s: "
            "E closes it here, Esc drops it."
        )
        return True

    def _target(self) -> int | None:
        """The row selected in the table, else the segment under the playhead."""
        labels = self._current_game.labels
        row = self.label_panel.selected_index()
        if row is not None and row < len(labels):
            return row
        try:
            row = segment_at(labels, self._now())
        except EditError as e:
            self.statusBar().showMessage(str(e))
            return None
        if row is None:
            self.statusBar().showMessage("No segment selected and none under the playhead")
        return row

    def _clear_pending(self):
        """Drop an open label, a mark and a pending class change."""
        self.segment_start = None
        self.current_class = None
        self.change_target = None
        self.label_panel.clear_segment_start()
        self._refresh_strips()

    def _on_selection_changed(self):
        self._refresh_strips()
        row = self.label_panel.selected_index()
        labels = self._current_game.labels
        if row is not None and row < len(labels):
            self.statusBar().showMessage(
                f"Selected {labels[row].describe()}: C, [, ], Delete act on it. Esc clears."
            )

    def _mark_segment_start(self):
        """S: set a mark at the playhead. The next class key labels mark to playhead."""
        if self.current_class is not None:
            self._finalize_current_label()
        self.change_target = None
        self.segment_start = self._now()
        self.current_class = None
        self.label_panel.update_segment_start(self.segment_start, mark=True)
        self._refresh_strips()
        self.statusBar().showMessage(
            f"Mark at {self.segment_start:.3f}s. Move to the other end and press 0-9 or R "
            "to label the stretch between. Esc cancels."
        )

    def _mark_segment_end(self):
        """E: close the open continuous label at the playhead."""
        if self.current_class is not None:
            self._finalize_current_label()
        elif self.segment_start is not None:
            self.statusBar().showMessage(
                f"Mark at {self.segment_start:.3f}s: press 0-9 or R to label from it to here."
            )
        else:
            self.statusBar().showMessage("No open label. Press 0-9 or R to start one, or S to set a mark.")

    def _assign_class(self, class_num: int):
        """A class key. What it does depends on what is pending:

        - after C: changes the chosen segment's class;
        - after S: labels the stretch from the mark to the playhead, replacing
          what was there;
        - otherwise, continuous mode: closes the open label here (replacing what
          it covers) and starts a new one with this class.
        """
        if class_num not in SCENE_LABELS:
            return
        new_class = SCENE_LABELS[class_num]

        if self.change_target is not None:
            row, self.change_target = self.change_target, None
            old = self._current_game.labels[row]
            self._apply(
                lambda labels: set_class(labels, row, new_class.value),
                f"Changed {old.describe()} to {new_class.value}",
                select_at=old.start_sec,
            )
            return

        now = self._now()
        if self.segment_start is not None and self.current_class is None:
            # The mark is used up even if the edit is refused: a mark left behind
            # would make a later class key overwrite everything back to it.
            mark = self.segment_start
            self._clear_pending()
            start, end = min(mark, now), max(mark, now)
            self._apply(
                lambda labels: overwrite(labels, start, end, new_class.value),
                f"Labelled {start:.3f}-{end:.3f}s as {new_class.value}",
                select_at=start,
            )
            return

        if self.current_class == new_class:
            self.statusBar().showMessage(
                f"Still labeling: {new_class.value} (started at {self.segment_start:.3f}s)"
            )
            return
        if self.current_class is not None:
            self._close_open_label(now)

        self.segment_start = now
        self.current_class = new_class
        self.label_panel.update_segment_start(self.segment_start)
        self.label_panel.update_segment_class(new_class)
        self._refresh_strips()
        self.statusBar().showMessage(
            f"Labeling: {new_class.value} (from {now:.3f}s - press a different key to end)"
        )

    def _close_open_label(self, end: float) -> bool:
        """Write the open continuous label as segment_start to end, replacing what it covers."""
        start, cls = self.segment_start, self.current_class.value
        return self._apply(
            lambda labels: overwrite(labels, start, end, cls),
            f"Closed: {min(start, end):.3f}-{max(start, end):.3f}s = {cls}",
        )

    def _finalize_current_label(self) -> bool:
        """Close the open label at the playhead and leave continuous mode."""
        if self.segment_start is None or self.current_class is None:
            return False
        closed = self._close_open_label(self._now())
        self._clear_pending()
        return closed

    def _start_change_class(self):
        """C: the next class key changes the target segment's class."""
        if self._label_open():
            return
        row = self._target()
        if row is None:
            return
        self.segment_start = None
        self.change_target = row
        self.label_panel.select_row(row)
        self.statusBar().showMessage(
            f"Change {self._current_game.labels[row].describe()} to: press 0-9 or R (Esc cancels)"
        )

    def _split_at_playhead(self):
        """K: cut the segment under the playhead in two at the playhead."""
        if self._label_open():
            return
        now = self._now()
        try:
            row = segment_at(self._current_game.labels, now)
        except EditError as e:
            self.statusBar().showMessage(str(e))
            return
        if row is None:
            self.statusBar().showMessage(f"No segment under the playhead at {now:.3f}s")
            return
        old = self._current_game.labels[row]
        self._apply(
            lambda labels: split(labels, row, now),
            f"Cut {old.describe()} at {now:.3f}s",
            select_at=now,
        )

    def _move_edge(self, edge: str):
        """[ or ]: move the target segment's start or end to the playhead."""
        if self._label_open():
            return
        row = self._target()
        if row is None:
            return
        now = self._now()
        old = self._current_game.labels[row]
        self._apply(
            lambda labels: move_edge(labels, row, edge, now),
            f"Moved the {edge} of {old.describe()} to {now:.3f}s",
            select_at=now if edge == "start" else old.start_sec,
        )

    def _delete_target(self):
        """Delete: remove the target segment, leaving its time unlabelled."""
        if self._label_open():
            return
        row = self._target()
        if row is None:
            return
        old = self._current_game.labels[row]
        self._apply(lambda labels: delete(labels, row), f"Deleted {old.describe()}")

    def _cancel(self):
        """Esc: drop the open label, mark or pending change, and clear the selection."""
        what = (
            f"open {self.current_class.value} label" if self.current_class is not None
            else "mark" if self.segment_start is not None
            else "class change" if self.change_target is not None
            else None
        )
        self._clear_pending()
        self.label_panel.select_row(None)
        self.statusBar().showMessage(f"Cancelled the {what}" if what else "Selection cleared")

    def _set_stage(self):
        """Open stage selection dialog."""
        dialog = StageDialog(self._current_game.stage, self)
        if dialog.exec_() == QDialog.Accepted:
            self._snapshot()
            self._current_game.stage = dialog.get_stage()
            self._update_ui()
            self.statusBar().showMessage(f"Stage set to: {dialog.get_stage()}")

    def _mark_game_boundary(self):
        """End the current game here and start the next one at the same moment.

        Games are contiguous. Press G in the menus between two games, after the
        previous game's results screen appears (see Game Boundaries in the
        module docstring).
        """
        # Finalize any active label first
        if self.current_class is not None:
            self._finalize_current_label()

        current_time = self._now()
        self._snapshot()

        # End current game
        self._current_game.end_sec = current_time

        # Create new game
        new_game = GameLabels(
            game_number=len(self.video_labels.games) + 1,
            start_sec=current_time,
            end_sec=self.player.get_duration(),
        )
        self.video_labels.games.append(new_game)
        self.current_game_index = len(self.video_labels.games) - 1

        self._update_ui()
        self.statusBar().showMessage(
            f"New game started at {current_time:.3f}s. Press T to set stage."
        )

    def _save_labels(self) -> bool:
        """Save labels to file. Returns False if no file was chosen."""
        if self.labels_path is None:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Labels",
                str(Path("data/scene_classifier/raw_labels")),
                "JSON Files (*.json)",
            )
            if not path:
                self.statusBar().showMessage("Not saved: no file chosen")
                return False
            self.labels_path = Path(path)

        # Ensure directory exists
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        with open(self.labels_path, "w") as f:
            json.dump(self.video_labels.to_dict(), f, indent=2)

        self._saved_state = self._label_state()
        self._update_title()
        message = f"Saved to {self.labels_path}"
        if self.current_class is not None:
            message += f" (the open {self.current_class.value} label is not in it until E closes it)"
        self.statusBar().showMessage(message)
        return True

    def _undo_last(self):
        """Ctrl+Z: put the games back as they were before the last change."""
        if not self.undo_stack:
            self.statusBar().showMessage("Nothing to undo")
            return
        self.current_game_index, self.video_labels.games = self.undo_stack.pop()
        self._clear_pending()
        self._update_ui()
        self.statusBar().showMessage(f"Undone ({len(self.undo_stack)} more change(s) can be undone)")

    def closeEvent(self, event):  # noqa: N802 - Qt override name
        """Close; ask first only if unsaved changes or an open label would be lost."""
        unsaved = self._has_unsaved_changes()
        open_label = self.current_class is not None
        if not unsaved and not open_label:
            self.player.cleanup()
            event.accept()
            return

        lines = []
        if unsaved:
            lines.append("There are unsaved changes. Save before closing?")
        if open_label:
            lines.append(
                f"A {self.current_class.value} label is still open from {self.segment_start:.3f}s. "
                "It is not saved: Cancel, then press E to close it at the playhead."
            )
        buttons = (QMessageBox.Save if unsaved else QMessageBox.NoButton) | QMessageBox.Discard | QMessageBox.Cancel
        reply = QMessageBox.question(self, "Save Labels?", "\n\n".join(lines), buttons)

        if reply == QMessageBox.Save:
            if not self._save_labels():
                event.ignore()  # no file chosen: closing would lose the labels
                return
            self.player.cleanup()
            event.accept()
        elif reply == QMessageBox.Discard:
            self.player.cleanup()
            event.accept()
        else:
            event.ignore()


def main():
    parser = argparse.ArgumentParser(
        description="Scene Labeler for SSBU videos",
        epilog="Supports both local video files and YouTube URLs.",
    )
    parser.add_argument(
        "video",
        help="Path to video file or YouTube URL",
    )
    parser.add_argument(
        "--labels", "-l",
        help="Path to labels JSON file (will be created if doesn't exist)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    video_input = args.video

    # Check if it's a YouTube URL
    if is_youtube_url(video_input):
        video_id = extract_video_id(video_input)
        cache_path = get_cache_path(video_input)

        if cache_path.exists():
            print(f"Using cached video: {cache_path}")
            video_path = cache_path
        else:
            print(f"Downloading video: {video_input}")
            print(f"Cache path: {cache_path}")

            # Create progress dialog
            progress = QProgressDialog("Downloading video...", "Cancel", 0, 0)
            progress.setWindowTitle("Downloading")
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.setAutoClose(True)

            # Create download thread
            download_thread = DownloadThread(video_input, cache_path)

            def on_progress(msg):
                progress.setLabelText(msg)

            def on_finished(path):
                progress.close()

            def on_error(msg):
                progress.close()
                QMessageBox.critical(None, "Download Error", msg)
                sys.exit(1)

            download_thread.progress.connect(on_progress)
            download_thread.finished_download.connect(on_finished)
            download_thread.error.connect(on_error)

            download_thread.start()

            # Show progress dialog (blocking)
            progress.exec_()

            # Wait for thread to finish
            download_thread.wait()

            if not cache_path.exists():
                print("Download failed or was cancelled")
                sys.exit(1)

            video_path = cache_path

        # Set default labels path based on video ID
        if not args.labels and video_id:
            labels_dir = Path("data/scene_classifier/raw_labels")
            labels_dir.mkdir(parents=True, exist_ok=True)
            args.labels = str(labels_dir / f"{video_id}.json")
    else:
        # Local file
        video_path = Path(video_input)
        if not video_path.exists():
            print(f"Error: Video not found: {video_path}")
            sys.exit(1)

    window = SceneLabeler(str(video_path), args.labels)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
