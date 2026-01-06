"""Computer vision modules for game state detection."""

from runitback.cv.ui_detector import (
    detect_ui_state,
    UIState,
    PlayerUIState,
    FrameType,
    detect_frame_type,
    detect_countdown,
    detect_game_end_screen,
    detect_gameplay_hud,
)
from runitback.cv.stage_detector import detect_stage, detect_stage_from_frames, Stage
from runitback.cv.character_detector import detect_characters, detect_characters_from_frames
from runitback.cv.scene_classifier import (
    SceneState,
    SceneClassification,
    SceneClassifier,
    classify_scene,
    get_scene_classifier,
)

__all__ = [
    "detect_ui_state",
    "UIState",
    "PlayerUIState",
    "FrameType",
    "detect_frame_type",
    "detect_countdown",
    "detect_game_end_screen",
    "detect_gameplay_hud",
    "detect_stage",
    "detect_stage_from_frames",
    "Stage",
    "detect_characters",
    "detect_characters_from_frames",
    "SceneState",
    "SceneClassification",
    "SceneClassifier",
    "classify_scene",
    "get_scene_classifier",
]
