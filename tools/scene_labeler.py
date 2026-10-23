#!/usr/bin/env python3
"""Scene Labeler - Video labeling tool for SSBU scene classification.

A PyQt5-based tool for labeling video segments with scene states and stage information.
Supports both local video files and YouTube URLs.

Usage:
    python tools/scene_labeler.py <video_path_or_youtube_url> [--labels <labels.json>]

Examples:
    python tools/scene_labeler.py path/to/video.mp4
    python tools/scene_labeler.py "https://www.youtube.com/watch?v=VIDEO_ID"
    python tools/scene_labeler.py "https://youtu.be/VIDEO_ID"

Keyboard Shortcuts:
    0-9     : Start labeling with class (continuous mode - auto-extends until changed)
    Space   : Pause/Play
    Left    : Previous frame (or 1 second if playing)
    Right   : Next frame (or 1 second if playing)
    S       : Reset segment start (finalizes current label if any)
    E       : Finalize current label at current position
    T       : Set stage for current game
    G       : Mark new game boundary (at GO! for start, GAME! for end)
    Ctrl+S  : Save labels to file
    Ctrl+Z  : Undo last label
    Q       : Quit

Labeling Workflow (Continuous Mode):
    1. Press 0-9 to start labeling with a class
    2. Navigate through the video (the label extends as you go)
    3. When the scene changes, press a different number key
    4. The previous label is automatically closed and a new one starts
    5. Press E to finalize the last label when done

Game Boundary Convention:
    - Mark game START at: First frame after GO! disappears
    - Mark game END at: Frame when GAME! appears

Scene Classes:
    0: non_gameplay     - Menus, loading screens, character select, results
    1: both_grounded    - Both players standing/attacking on stage
    2: one_airborne     - One in air, one on stage
    3: both_airborne    - Both in the air
    4: one_offstage     - One beyond stage edge
    5: both_offstage    - Both off-stage
    6: one_on_ledge     - One grabbing ledge
    7: both_on_ledge    - Both grabbing ledge
    8: one_knocked_down - One lying flat on stage
    9: both_knocked_down- Both lying flat (trade)
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
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
    RESPAWNING = "respawning"  # One player on angel platform after stock loss


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


@dataclass
class SegmentLabel:
    """A labeled segment of video."""
    start_sec: float
    end_sec: float
    scene_class: str


@dataclass
class GameLabels:
    """Labels for a single game within a video."""
    game_number: int
    stage: str = "unknown"
    start_sec: float = 0.0
    end_sec: float = 0.0
    labels: list[SegmentLabel] = field(default_factory=list)


@dataclass
class VideoLabels:
    """All labels for a video."""
    video_id: str
    video_path: str
    games: list[GameLabels] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "games": [
                {
                    "game_number": g.game_number,
                    "stage": g.stage,
                    "start_sec": g.start_sec,
                    "end_sec": g.end_sec,
                    "labels": [
                        {
                            "start_sec": lbl.start_sec,
                            "end_sec": lbl.end_sec,
                            "class": lbl.scene_class,
                        }
                        for lbl in g.labels
                    ],
                }
                for g in self.games
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoLabels":
        """Create from dictionary."""
        games = []
        for g in data.get("games", []):
            labels = [
                SegmentLabel(
                    start_sec=lbl["start_sec"],
                    end_sec=lbl["end_sec"],
                    scene_class=lbl["class"],
                )
                for lbl in g.get("labels", [])
            ]
            games.append(
                GameLabels(
                    game_number=g["game_number"],
                    stage=g.get("stage", "unknown"),
                    start_sec=g.get("start_sec", 0.0),
                    end_sec=g.get("end_sec", 0.0),
                    labels=labels,
                )
            )
        return cls(
            video_id=data["video_id"],
            video_path=data.get("video_path", ""),
            games=games,
        )


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

        self.time_label = QLabel("0:00.0 / 0:00.0")
        self.time_label.setMinimumWidth(150)
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
        """Format seconds as M:SS.s"""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}:{secs:04.1f}"

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
        """Seek to specific time."""
        frame = int(seconds * self.fps)
        self.seek_frame(frame)

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

        self.segment_class_label = QLabel("Class: (press 0-9 to label)")
        segment_layout.addWidget(self.segment_class_label)

        layout.addWidget(segment_frame)

        # Labels table
        layout.addWidget(QLabel("Labels:"))
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Start", "End", "Class"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.cellDoubleClicked.connect(self._on_cell_double_click)
        layout.addWidget(self.table)

        # Stats
        self.stats_label = QLabel("Total: 0 labels")
        layout.addWidget(self.stats_label)

    def update_game_info(self, game: GameLabels):
        """Update game info display."""
        self.game_label.setText(f"Game {game.game_number}")
        stage_name = STAGE_DISPLAY_NAMES.get(game.stage, game.stage)
        self.stage_label.setText(f"Stage: {stage_name}")

    def update_segment_start(self, time_sec: float):
        """Update segment start display."""
        self.segment_start_label.setText(f"Segment Start: {time_sec:.1f}s")

    def clear_segment_start(self):
        """Clear segment start display."""
        self.segment_start_label.setText("Segment Start: --")
        self.segment_class_label.setText("Class: (press 0-9 to label)")

    def update_segment_class(self, scene_class: SceneState):
        """Update segment class display."""
        color = SCENE_COLORS.get(scene_class, "#888888")
        self.segment_class_label.setText(
            f'Class: <span style="color:{color};">{scene_class.value}</span>'
        )

    def update_labels(self, labels: list[SegmentLabel]):
        """Update labels table."""
        self.table.setRowCount(len(labels))
        for i, label in enumerate(labels):
            self.table.setItem(i, 0, QTableWidgetItem(f"{label.start_sec:.1f}s"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{label.end_sec:.1f}s"))

            class_item = QTableWidgetItem(label.scene_class)
            # Find the matching SceneState for color
            for state in SceneState:
                if state.value == label.scene_class:
                    class_item.setForeground(Qt.GlobalColor.white)
                    break
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


class LegendPanel(QWidget):
    """Panel showing keyboard shortcuts and scene class legend."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(2)

        # Scene classes legend
        layout.addWidget(QLabel("<b>Scene Classes (0-9, R):</b>"))

        # Class descriptions
        class_info = [
            ("0", "non_gameplay", "Menus, loading, results"),
            ("1", "both_grounded", "Both on stage, grounded"),
            ("2", "one_airborne", "One jumping/launched"),
            ("3", "both_airborne", "Both in the air"),
            ("4", "one_offstage", "One past stage edge"),
            ("5", "both_offstage", "Both offstage"),
            ("6", "one_on_ledge", "One grabbing ledge"),
            ("7", "both_on_ledge", "Both on ledge"),
            ("8", "one_knocked_down", "One lying flat (tech)"),
            ("9", "both_knocked_down", "Both down (trade)"),
            ("R", "respawning", "On angel platform"),
        ]

        for key, class_name, description in class_info:
            row = QHBoxLayout()
            row.setSpacing(4)

            # Key badge
            key_label = QLabel(f"<b>{key}</b>")
            key_label.setFixedWidth(20)
            key_label.setAlignment(Qt.AlignCenter)
            row.addWidget(key_label)

            # Color indicator - convert key to int if numeric, else use special mapping
            key_int = int(key) if key.isdigit() else {"R": 10}.get(key)
            state = SCENE_LABELS.get(key_int)
            color = SCENE_COLORS.get(state, "#888888")
            color_box = QFrame()
            color_box.setFixedSize(12, 12)
            color_box.setStyleSheet(f"background-color: {color}; border-radius: 2px;")
            row.addWidget(color_box)

            # Description
            desc_label = QLabel(f"<small>{description}</small>")
            desc_label.setWordWrap(True)
            row.addWidget(desc_label, stretch=1)

            layout.addLayout(row)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Other shortcuts
        layout.addWidget(QLabel("<b>Controls:</b>"))
        shortcuts = [
            ("Space", "Play/Pause"),
            ("← →", "Prev/Next frame"),
            ("E", "Finalize label"),
            ("G", "Game boundary"),
            ("T", "Set stage"),
            ("Ctrl+S", "Save"),
            ("Ctrl+Z", "Undo"),
        ]

        for key, action in shortcuts:
            row = QHBoxLayout()
            row.setSpacing(4)
            key_label = QLabel(f"<b>{key}</b>")
            key_label.setFixedWidth(50)
            row.addWidget(key_label)
            row.addWidget(QLabel(f"<small>{action}</small>"), stretch=1)
            layout.addLayout(row)

        layout.addStretch()


class ClassDistributionPanel(QWidget):
    """Panel showing class distribution statistics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Class Distribution:</b>"))

        self.bars = {}
        for state in SceneState:
            row = QHBoxLayout()

            # Class name
            name_label = QLabel(f"{state.value}:")
            name_label.setMinimumWidth(120)
            row.addWidget(name_label)

            # Progress bar (using QFrame)
            bar = QFrame()
            bar.setFixedHeight(16)
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
        """Update distribution display."""
        # Count by class
        counts = {state: 0 for state in SceneState}
        for label in labels:
            for state in SceneState:
                if state.value == label.scene_class:
                    counts[state] += 1
                    break

        total = sum(counts.values()) or 1

        # Update bars
        for state, (bar, count_label) in self.bars.items():
            count = counts[state]
            pct = count / total
            width = int(pct * 200)
            bar.setFixedWidth(max(1, width) if count > 0 else 0)
            count_label.setText(str(count))


class SceneLabeler(QMainWindow):
    """Main labeling application window."""

    def __init__(self, video_path: str, labels_path: str | None = None):
        super().__init__()
        self.video_path = Path(video_path)
        self.labels_path = Path(labels_path) if labels_path else None

        self.video_labels: VideoLabels | None = None
        self.current_game_index = 0
        self.segment_start: float | None = None
        self.current_class: SceneState | None = None  # For continuous labeling
        self.undo_stack: list[tuple] = []  # (game_index, label_index)

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

        # Right side: panels
        right_panel = QSplitter(Qt.Vertical)

        # Legend panel (top)
        self.legend_panel = LegendPanel()
        right_panel.addWidget(self.legend_panel)

        self.label_panel = LabelPanel()
        self.label_panel.label_selected.connect(self.player.seek_time)
        right_panel.addWidget(self.label_panel)

        self.dist_panel = ClassDistributionPanel()
        right_panel.addWidget(self.dist_panel)

        # Set initial sizes (legend smaller, labels bigger)
        right_panel.setSizes([250, 300, 150])

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

        self._update_ui()

    @property
    def _current_game(self) -> GameLabels:
        """Get current game."""
        return self.video_labels.games[self.current_game_index]

    def _update_ui(self):
        """Update all UI elements."""
        game = self._current_game
        self.label_panel.update_game_info(game)
        self.label_panel.update_labels(game.labels)
        self.dist_panel.update_distribution(game.labels)

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
                    self._update_ui()
                    self.statusBar().showMessage(
                        f"Switched to Game {game.game_number} ({game.stage})"
                    )
                break

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

    def _mark_segment_start(self):
        """Mark the start of a segment (or reset current label start)."""
        # If we're in continuous mode with an active label, finalize it first
        if self.current_class is not None:
            self._finalize_current_label()

        self.segment_start = self.player.get_current_time()
        self.current_class = None
        self.label_panel.update_segment_start(self.segment_start)
        self.statusBar().showMessage(
            f"Segment start: {self.segment_start:.1f}s. Press 0-9 to assign class."
        )

    def _mark_segment_end(self):
        """Finalize the current label at current position."""
        if self.current_class is not None:
            # In continuous mode - finalize the current label
            if self._finalize_current_label():
                return

        if self.segment_start is None:
            self.statusBar().showMessage("Press S first to mark segment start, or press 0-9 to start labeling.")
            return

        # Show where end would be
        end_time = self.player.get_current_time()
        self.statusBar().showMessage(
            f"Segment: {self.segment_start:.1f}s - {end_time:.1f}s. Press 0-9 to assign class."
        )

    def _assign_class(self, class_num: int):
        """Assign a class to the current segment.

        Continuous labeling mode:
        - First press: Sets the start time and class (label extends as you navigate)
        - Subsequent press of SAME key: Does nothing (you're still in that class)
        - Press of DIFFERENT key: Ends current label at current time, starts new one
        """
        if class_num not in SCENE_LABELS:
            return

        new_class = SCENE_LABELS[class_num]
        current_time = self.player.get_current_time()
        game = self._current_game

        # If we're already labeling this class, do nothing
        if self.current_class == new_class and self.segment_start is not None:
            self.statusBar().showMessage(
                f"Still labeling: {new_class.value} (started at {self.segment_start:.1f}s)"
            )
            return

        # If we have an active label, close it first
        if self.segment_start is not None and self.current_class is not None:
            # Create label for the previous segment
            start = round(self.segment_start, 1)
            end = round(current_time, 1)

            # Only create label if it has positive duration
            if end > start:
                label = SegmentLabel(
                    start_sec=start,
                    end_sec=end,
                    scene_class=self.current_class.value,
                )
                game.labels.append(label)
                game.labels.sort(key=lambda x: x.start_sec)

                # Track for undo
                label_index = game.labels.index(label)
                self.undo_stack.append((self.current_game_index, label_index))

                self.statusBar().showMessage(
                    f"Closed: {start:.1f}s - {end:.1f}s = {self.current_class.value}"
                )

        # Start new label at current time with new class
        self.segment_start = current_time
        self.current_class = new_class
        self.label_panel.update_segment_start(self.segment_start)
        self.label_panel.update_segment_class(new_class)

        # Update UI
        self._update_ui()
        self.statusBar().showMessage(
            f"Labeling: {new_class.value} (from {current_time:.1f}s - press different key to end)"
        )

    def _finalize_current_label(self):
        """Finalize the current label at the current time position."""
        if self.segment_start is None or self.current_class is None:
            return False

        current_time = self.player.get_current_time()
        game = self._current_game

        start = round(self.segment_start, 1)
        end = round(current_time, 1)

        # Only create label if it has positive duration
        if end > start:
            label = SegmentLabel(
                start_sec=start,
                end_sec=end,
                scene_class=self.current_class.value,
            )
            game.labels.append(label)
            game.labels.sort(key=lambda x: x.start_sec)

            # Track for undo
            label_index = game.labels.index(label)
            self.undo_stack.append((self.current_game_index, label_index))

            self.statusBar().showMessage(
                f"Finalized: {start:.1f}s - {end:.1f}s = {self.current_class.value}"
            )

            # Reset
            self.segment_start = None
            self.current_class = None
            self.label_panel.clear_segment_start()
            self._update_ui()
            return True

        return False

    def _set_stage(self):
        """Open stage selection dialog."""
        dialog = StageDialog(self._current_game.stage, self)
        if dialog.exec_() == QDialog.Accepted:
            self._current_game.stage = dialog.get_stage()
            self._update_ui()
            self.statusBar().showMessage(f"Stage set to: {dialog.get_stage()}")

    def _mark_game_boundary(self):
        """Mark a new game boundary at current position.

        Convention: Mark game boundaries at:
        - Game start: First frame after GO! disappears (gameplay begins)
        - Game end: Frame when GAME! appears (gameplay ends)
        """
        # Finalize any active label first
        if self.current_class is not None:
            self._finalize_current_label()

        current_time = self.player.get_current_time()

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
            f"New game started at {current_time:.1f}s. Press T to set stage."
        )

    def _save_labels(self):
        """Save labels to file."""
        if self.labels_path is None:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Labels",
                str(Path("data/scene_classifier/raw_labels")),
                "JSON Files (*.json)",
            )
            if not path:
                return
            self.labels_path = Path(path)

        # Ensure directory exists
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        with open(self.labels_path, "w") as f:
            json.dump(self.video_labels.to_dict(), f, indent=2)

        self.statusBar().showMessage(f"Saved to {self.labels_path}")

    def _undo_last(self):
        """Undo last label."""
        if not self.undo_stack:
            self.statusBar().showMessage("Nothing to undo")
            return

        game_index, label_index = self.undo_stack.pop()
        game = self.video_labels.games[game_index]

        if 0 <= label_index < len(game.labels):
            removed = game.labels.pop(label_index)
            self._update_ui()
            self.statusBar().showMessage(f"Undone: {removed.scene_class}")

    def closeEvent(self, event):  # noqa: N802 - Qt override name
        """Handle window close."""
        # Check for unsaved changes
        reply = QMessageBox.question(
            self,
            "Save Labels?",
            "Do you want to save labels before closing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        )

        if reply == QMessageBox.Save:
            self._save_labels()
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
