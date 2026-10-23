"""Computer vision modules for game state detection."""

from runitback.cv.character_detector import detect_characters, detect_characters_from_frames
from runitback.cv.scene_classifier import (
    SceneClassification,
    SceneClassifier,
    SceneState,
    classify_scene,
    get_scene_classifier,
)
from runitback.cv.stage_detector import Stage, detect_stage, detect_stage_from_frames
from runitback.cv.ui_detector import (
    FrameType,
    PlayerUIState,
    UIState,
    detect_countdown,
    detect_frame_type,
    detect_game_end_screen,
    detect_gameplay_hud,
    detect_ui_state,
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
