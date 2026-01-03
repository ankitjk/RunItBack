#!/usr/bin/env python3
"""Scene Labeler - Video labeling tool for SSBU scene classification.

A PyQt5-based tool for labeling video segments with scene states and stage information.

Usage:
    python tools/scene_labeler.py <video_path> [--labels <labels.json>]

Keyboard Shortcuts:
    1-9     : Assign scene class to current segment
    Space   : Pause/Play
    Left    : Previous frame (or 1 second if playing)
    Right   : Next frame (or 1 second if playing)
    S       : Mark segment start
    E       : Mark segment end and save
    T       : Set stage for current game
    G       : Mark new game boundary
    Ctrl+S  : Save labels to file
    Ctrl+Z  : Undo last label
    Q       : Quit

Scene Classes:
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
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
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
    QPushButton,
    QShortcut,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class SceneState(Enum):
    """Scene classification states."""
    BOTH_GROUNDED = "both_grounded"
    ONE_AIRBORNE = "one_airborne"
    BOTH_AIRBORNE = "both_airborne"
    ONE_OFFSTAGE = "one_offstage"
    BOTH_OFFSTAGE = "both_offstage"
    ONE_ON_LEDGE = "one_on_ledge"
    BOTH_ON_LEDGE = "both_on_ledge"
    ONE_KNOCKED_DOWN = "one_knocked_down"
    BOTH_KNOCKED_DOWN = "both_knocked_down"


SCENE_LABELS = {
    1: SceneState.BOTH_GROUNDED,
    2: SceneState.ONE_AIRBORNE,
    3: SceneState.BOTH_AIRBORNE,
    4: SceneState.ONE_OFFSTAGE,
    5: SceneState.BOTH_OFFSTAGE,
    6: SceneState.ONE_ON_LEDGE,
    7: SceneState.BOTH_ON_LEDGE,
    8: SceneState.ONE_KNOCKED_DOWN,
    9: SceneState.BOTH_KNOCKED_DOWN,
}

SCENE_COLORS = {
    SceneState.BOTH_GROUNDED: "#4CAF50",      # Green
    SceneState.ONE_AIRBORNE: "#2196F3",       # Blue
    SceneState.BOTH_AIRBORNE: "#00BCD4",      # Cyan
    SceneState.ONE_OFFSTAGE: "#FF9800",       # Orange
    SceneState.BOTH_OFFSTAGE: "#FF5722",      # Deep Orange
    SceneState.ONE_ON_LEDGE: "#9C27B0",       # Purple
    SceneState.BOTH_ON_LEDGE: "#E91E63",      # Pink
    SceneState.ONE_KNOCKED_DOWN: "#795548",   # Brown
    SceneState.BOTH_KNOCKED_DOWN: "#607D8B",  # Blue Grey
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
                            "start_sec": l.start_sec,
                            "end_sec": l.end_sec,
                            "class": l.scene_class,
                        }
                        for l in g.labels
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
                    start_sec=l["start_sec"],
                    end_sec=l["end_sec"],
                    scene_class=l["class"],
                )
                for l in g.get("labels", [])
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
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame = 0
        self.playing = False
        self.playback_speed = 1.0

        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: black;")
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

        # Convert to RGB and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale to fit
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

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

        self.segment_class_label = QLabel("Class: (press 1-9 to label)")
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
        self.segment_class_label.setText("Class: (press 1-9 to label)")

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
                    color = SCENE_COLORS.get(state, "#888888")
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


class ClassDistributionPanel(QWidget):
    """Panel showing class distribution statistics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Class Distribution:"))

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

    def __init__(self, video_path: str, labels_path: Optional[str] = None):
        super().__init__()
        self.video_path = Path(video_path)
        self.labels_path = Path(labels_path) if labels_path else None

        self.video_labels: Optional[VideoLabels] = None
        self.current_game_index = 0
        self.segment_start: Optional[float] = None
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

        self.label_panel = LabelPanel()
        self.label_panel.label_selected.connect(self.player.seek_time)
        right_panel.addWidget(self.label_panel)

        self.dist_panel = ClassDistributionPanel()
        right_panel.addWidget(self.dist_panel)

        main_layout.addWidget(right_panel, stretch=1)

        # Status bar
        self.statusBar().showMessage("Ready. Press S to start a segment, 1-9 to label.")

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Playback
        QShortcut(Qt.Key_Space, self, self.player.toggle_play)
        QShortcut(Qt.Key_Left, self, self._on_left_key)
        QShortcut(Qt.Key_Right, self, self._on_right_key)

        # Segment marking
        QShortcut(Qt.Key_S, self, self._mark_segment_start)
        QShortcut(Qt.Key_E, self, self._mark_segment_end)

        # Class assignment (1-9)
        for i in range(1, 10):
            QShortcut(getattr(Qt, f"Key_{i}"), self, lambda x=i: self._assign_class(x))

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
        """Handle frame change."""
        pass  # Could highlight current label in table

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
        """Mark the start of a segment."""
        self.segment_start = self.player.get_current_time()
        self.label_panel.update_segment_start(self.segment_start)
        self.statusBar().showMessage(
            f"Segment start: {self.segment_start:.1f}s. Press 1-9 to assign class."
        )

    def _mark_segment_end(self):
        """Mark the end of a segment (without class)."""
        if self.segment_start is None:
            self.statusBar().showMessage("Press S first to mark segment start.")
            return
        # This just shows where end would be
        end_time = self.player.get_current_time()
        self.statusBar().showMessage(
            f"Segment: {self.segment_start:.1f}s - {end_time:.1f}s. Press 1-9 to assign class."
        )

    def _assign_class(self, class_num: int):
        """Assign a class to the current segment."""
        if class_num not in SCENE_LABELS:
            return

        if self.segment_start is None:
            # Use current position as start, allow user to set class for "now"
            self.segment_start = self.player.get_current_time()

        end_time = self.player.get_current_time()

        # Ensure start < end
        if end_time <= self.segment_start:
            # Swap or extend slightly
            if end_time == self.segment_start:
                end_time = self.segment_start + 0.5
            else:
                self.segment_start, end_time = end_time, self.segment_start

        scene_class = SCENE_LABELS[class_num]

        # Create label
        label = SegmentLabel(
            start_sec=round(self.segment_start, 1),
            end_sec=round(end_time, 1),
            scene_class=scene_class.value,
        )

        # Add to current game
        game = self._current_game
        game.labels.append(label)
        game.labels.sort(key=lambda x: x.start_sec)

        # Track for undo
        label_index = game.labels.index(label)
        self.undo_stack.append((self.current_game_index, label_index))

        # Reset segment
        self.segment_start = None
        self.label_panel.clear_segment_start()

        # Update UI
        self._update_ui()
        self.statusBar().showMessage(
            f"Added: {label.start_sec:.1f}s - {label.end_sec:.1f}s = {scene_class.value}"
        )

    def _set_stage(self):
        """Open stage selection dialog."""
        dialog = StageDialog(self._current_game.stage, self)
        if dialog.exec_() == QDialog.Accepted:
            self._current_game.stage = dialog.get_stage()
            self._update_ui()
            self.statusBar().showMessage(f"Stage set to: {dialog.get_stage()}")

    def _mark_game_boundary(self):
        """Mark a new game boundary at current position."""
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

    def closeEvent(self, event):
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
    parser = argparse.ArgumentParser(description="Scene Labeler for SSBU videos")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--labels", "-l",
        help="Path to labels JSON file (will be created if doesn't exist)",
    )
    args = parser.parse_args()

    # Check video exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = SceneLabeler(str(video_path), args.labels)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
