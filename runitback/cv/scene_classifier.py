"""Scene state classification for SSBU gameplay frames.

Uses a trained ConvNeXt-Tiny model to classify gameplay frames into
positional states (grounded, airborne, offstage, ledge, knocked down).
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default model path
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "models" / "scene_classifier.onnx"

# The output order of the deployed model (October 2026), which predates
# class_mapping.json: ImageFolder's alphabetical order of its 7 classes. Used
# only for a model with no class_mapping.json beside it.
LEGACY_CLASS_ORDER = [
    "both_airborne",
    "both_grounded",
    "both_offstage",
    "non_gameplay",
    "one_airborne",
    "one_offstage",
    "respawning",
]


def load_class_order(model_path: Path) -> list[str]:
    """The model's class names, by output index.

    Read from class_mapping.json next to the model, which training writes
    ({"0": "both_airborne", ...}). Every retrained model has its own order and
    number of classes, so a fixed list would name predictions wrongly
    (ISSUE-017). A model without the file is the legacy deployed model.
    """
    mapping_path = model_path.parent / "class_mapping.json"
    if not mapping_path.exists():
        return list(LEGACY_CLASS_ORDER)
    mapping = json.loads(mapping_path.read_text())
    order = [mapping[str(i)] for i in range(len(mapping))]  # KeyError if indices are not 0..n-1
    return order


class SceneState(Enum):
    """Scene states for gameplay classification.

    These represent positional states, not actions. A player attacking
    while grounded is still "grounded".
    """

    # Non-gameplay (menus, loading, results screen)
    NON_GAMEPLAY = "non_gameplay"

    # Neutral-like states
    BOTH_GROUNDED = "both_grounded"  # Both players on stage, feet on ground

    # Advantage/disadvantage states
    ONE_AIRBORNE = "one_airborne"    # One in air (juggle situation)
    BOTH_AIRBORNE = "both_airborne"  # Both in air (active exchange)
    ONE_OFFSTAGE = "one_offstage"    # One past ledge (edgeguard)
    BOTH_OFFSTAGE = "both_offstage"  # Both offstage (aggressive edgeguard)
    ONE_ON_LEDGE = "one_on_ledge"    # One grabbing ledge (ledgetrap)
    BOTH_ON_LEDGE = "both_on_ledge"  # Both on ledge (rare)
    ONE_KNOCKED_DOWN = "one_knocked_down"  # One lying on stage (tech chase)
    BOTH_KNOCKED_DOWN = "both_knocked_down"  # Both down (trade)

    # Special states
    RESPAWNING = "respawning"  # KO flash or star KO, respawn platform, stock count

    # Whole-screen states (added 2026-11-26; docs/SCENE_CLASSES.md)
    PRE_GAME = "pre_game"  # Entrances and countdown, until GO!
    GAME_END = "game_end"  # GAME! or TIME! screen, until the white flash
    FINISHING_BLOW = "finishing_blow"  # The final-stock red flash

    # Fallback
    UNKNOWN = "unknown"

    @property
    def is_neutral(self) -> bool:
        """Whether this state represents neutral (neither player in disadvantage)."""
        return self == SceneState.BOTH_GROUNDED

    @property
    def is_gameplay(self) -> bool:
        """Whether this state represents active gameplay.

        pre_game and game_end are screens before and after play; a finishing
        blow is a moment of play, shown with a red flash.
        """
        return self not in (SceneState.NON_GAMEPLAY, SceneState.PRE_GAME, SceneState.GAME_END, SceneState.UNKNOWN)

    @property
    def display_name(self) -> str:
        """Human-readable state name."""
        names = {
            SceneState.NON_GAMEPLAY: "Non-Gameplay",
            SceneState.BOTH_GROUNDED: "Both Grounded",
            SceneState.ONE_AIRBORNE: "One Airborne",
            SceneState.BOTH_AIRBORNE: "Both Airborne",
            SceneState.ONE_OFFSTAGE: "One Offstage",
            SceneState.BOTH_OFFSTAGE: "Both Offstage",
            SceneState.ONE_ON_LEDGE: "One on Ledge",
            SceneState.BOTH_ON_LEDGE: "Both on Ledge",
            SceneState.ONE_KNOCKED_DOWN: "One Knocked Down",
            SceneState.BOTH_KNOCKED_DOWN: "Both Knocked Down",
            SceneState.RESPAWNING: "Respawning",
            SceneState.PRE_GAME: "Pre-Game",
            SceneState.GAME_END: "Game End",
            SceneState.FINISHING_BLOW: "Finishing Blow",
            SceneState.UNKNOWN: "Unknown",
        }
        return names.get(self, self.value)


@dataclass
class SceneClassification:
    """Result of scene classification."""

    state: SceneState
    confidence: float
    raw_probabilities: dict[str, float] | None = None

    @property
    def is_confident(self) -> bool:
        """Whether the classification confidence is above threshold."""
        return self.confidence >= 0.6


class SceneClassifier:
    """ONNX-based scene state classifier.

    Uses a trained ConvNeXt-Tiny model to classify gameplay frames.
    Includes temporal smoothing to reduce frame-to-frame jitter.
    """

    # Map class names to SceneState (every class name is a SceneState value)
    CLASS_TO_STATE = {state.value: state for state in SceneState if state != SceneState.UNKNOWN}

    def __init__(
        self,
        model_path: Path | None = None,
        confidence_threshold: float = 0.6,
        temporal_smoothing: int = 5,
    ):
        """Initialize the scene classifier.

        Args:
            model_path: Path to ONNX model file. Uses default if None.
            confidence_threshold: Minimum confidence for valid classification.
            temporal_smoothing: Number of frames for median filter (0 to disable).
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.confidence_threshold = confidence_threshold
        self.temporal_smoothing = temporal_smoothing

        self._session = None
        self._history: list[SceneState] = []
        self._available = False
        # The model's class names by output index (load_class_order)
        self.class_order: list[str] = []

        # ImageNet normalization (same as training)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self._load_model()

    def _load_model(self) -> None:
        """Load the ONNX model."""
        if not self.model_path.exists():
            logger.warning(f"Scene classifier model not found: {self.model_path}")
            return

        try:
            self.class_order = load_class_order(self.model_path)
        except (OSError, ValueError, KeyError) as e:
            logger.error(f"Scene classifier disabled: unreadable class_mapping.json beside {self.model_path}: {e}")
            return
        unknown = [name for name in self.class_order if name not in self.CLASS_TO_STATE]
        if unknown:
            logger.warning(f"Scene classifier classes not known to RunItBack, reported as unknown: {unknown}")

        try:
            import onnxruntime as ort

            # Try providers in order of preference
            # CPUExecutionProvider is most reliable
            providers_to_try = [
                ["CPUExecutionProvider"],
                ["CoreMLExecutionProvider", "CPUExecutionProvider"],
            ]

            for providers in providers_to_try:
                try:
                    session = ort.InferenceSession(
                        str(self.model_path),
                        providers=providers,
                    )
                    n_outputs = session.get_outputs()[0].shape[-1]
                    if isinstance(n_outputs, int) and n_outputs != len(self.class_order):
                        # Wrong names on every prediction is worse than none
                        logger.error(
                            f"Scene classifier disabled: the model has {n_outputs} outputs but its "
                            f"class list has {len(self.class_order)} ({self.model_path})"
                        )
                        return
                    self._session = session
                    self._available = True
                    active_providers = session.get_providers()
                    logger.info(f"Loaded scene classifier from {self.model_path} (providers: {active_providers})")
                    return
                except Exception as e:
                    logger.debug(f"Failed to load with providers {providers}: {e}")
                    continue

            logger.error("Failed to load scene classifier with any provider")

        except ImportError:
            logger.warning("onnxruntime not installed, scene classification disabled")
        except Exception as e:
            logger.error(f"Failed to load scene classifier: {e}")

    @property
    def available(self) -> bool:
        """Whether the classifier is ready to use."""
        return self._available

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a frame for inference.

        Args:
            frame: BGR frame from OpenCV (H, W, 3)

        Returns:
            Preprocessed tensor (1, 3, 224, 224)
        """
        import cv2

        # Resize to 224x224
        resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and apply ImageNet normalization
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - self._mean) / self._std

        # Convert to NCHW format
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor.astype(np.float32)

    def classify(self, frame: np.ndarray) -> SceneClassification:
        """Classify a single frame.

        Args:
            frame: BGR frame from OpenCV

        Returns:
            SceneClassification with state, confidence, and probabilities
        """
        if not self._available or self._session is None:
            return SceneClassification(
                state=SceneState.UNKNOWN,
                confidence=0.0,
            )

        # Preprocess
        tensor = self.preprocess(frame)

        # Run inference
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: tensor})
        logits = outputs[0][0]

        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Get prediction
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        class_name = self.class_order[pred_idx]
        state = self.CLASS_TO_STATE.get(class_name, SceneState.UNKNOWN)

        # Build probability dict
        raw_probs = {
            self.class_order[i]: float(probs[i])
            for i in range(len(self.class_order))
        }

        # Apply temporal smoothing
        if self.temporal_smoothing > 0:
            self._history.append(state)
            if len(self._history) > self.temporal_smoothing:
                self._history.pop(0)

            # Use median filter (most common state in window)
            if len(self._history) >= 3:
                from collections import Counter
                state_counts = Counter(self._history)
                state = state_counts.most_common(1)[0][0]

        return SceneClassification(
            state=state,
            confidence=confidence,
            raw_probabilities=raw_probs,
        )

    def classify_batch(self, frames: list[np.ndarray]) -> list[SceneClassification]:
        """Classify multiple frames.

        Note: Due to ONNX model constraints, this processes frames one at a time.
        For large batches, consider using a progress indicator.

        Args:
            frames: List of BGR frames

        Returns:
            List of classifications
        """
        if not self._available or self._session is None:
            return [
                SceneClassification(state=SceneState.UNKNOWN, confidence=0.0)
                for _ in frames
            ]

        # Process frames one at a time (ONNX model has batch size constraints)
        # Disable temporal smoothing for batch processing to avoid state leakage
        original_smoothing = self.temporal_smoothing
        self.temporal_smoothing = 0
        self._history.clear()

        results = []
        input_name = self._session.get_inputs()[0].name

        for frame in frames:
            tensor = self.preprocess(frame)
            outputs = self._session.run(None, {input_name: tensor})
            logits = outputs[0][0]

            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx])
            class_name = self.class_order[pred_idx]
            state = self.CLASS_TO_STATE.get(class_name, SceneState.UNKNOWN)

            raw_probs = {
                self.class_order[j]: float(probs[j])
                for j in range(len(self.class_order))
            }

            results.append(SceneClassification(
                state=state,
                confidence=confidence,
                raw_probabilities=raw_probs,
            ))

        # Restore temporal smoothing setting
        self.temporal_smoothing = original_smoothing

        return results

    def reset(self) -> None:
        """Reset temporal smoothing history."""
        self._history.clear()


# Singleton instance for convenience
_classifier: SceneClassifier | None = None


def get_scene_classifier() -> SceneClassifier:
    """Get or create the singleton scene classifier."""
    global _classifier
    if _classifier is None:
        _classifier = SceneClassifier()
    return _classifier


def classify_scene(frame: np.ndarray) -> SceneClassification:
    """Convenience function to classify a single frame.

    Args:
        frame: BGR frame from OpenCV

    Returns:
        SceneClassification result
    """
    return get_scene_classifier().classify(frame)
