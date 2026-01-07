"""Game state detection based on damage percentage tracking.

Detects neutral, advantage, and disadvantage states by analyzing
damage percentage changes between frames.

Key concepts:
- NEUTRAL: Neither player has taken significant damage recently, both are on stage
- ADVANTAGE: You hit the opponent (their damage increased)
- DISADVANTAGE: Opponent hit you (your damage increased)

The detector uses a reset timer to return to neutral after a period
of no damage exchanges while both players are visible (on stage).

Stock loss detection methods:
1. PRIMARY: Damage reset detection (instant, free) - detects when damage goes from
   high (50%+) to 0% over consecutive frames
2. SMART GEMINI: Automatic Gemini for uncertain situations:
   - Low-percent kills: damage drops but never reached 50% threshold
   - GAME! screens: final stock loss (no damage reset occurs after)
3. FULL GEMINI: Check every frame with Gemini (most accurate, highest cost)
"""

import logging
import os
from typing import Optional

import cv2
import numpy as np

from runitback.classification.states import (
    DisadvantageType,
    FrameState,
    GameState,
    GameStateAnalysis,
    StateSegment,
    StockLoss,
)
from runitback.cv.ocr import read_damage_percentage, read_damage_batch_parallel
from runitback.cv.ui_detector import get_roi, P1_DAMAGE_ROI, P2_DAMAGE_ROI, detect_stock_counts

logger = logging.getLogger(__name__)


class GameStateDetector:
    """Detects game states from video frames using damage tracking.

    The detector works by:
    1. Reading damage percentages from each frame via OCR
    2. Detecting significant damage increases (hits)
    3. Tracking state transitions based on who got hit
    4. Using a reset timer to return to neutral
    5. Optionally confirming stock losses with Gemini Vision API

    Attributes:
        hit_threshold: Minimum damage increase to count as a hit (filters self-damage)
        neutral_reset_time: Seconds without damage before returning to neutral
        player_perspective: Which player to analyze (1 or 2)
        use_gemini_stock_detection: Whether to use Gemini for stock loss confirmation
        gemini_api_key: API key for Gemini (uses env var if not provided)
    """

    def __init__(
        self,
        player_perspective: int = 1,
        hit_threshold: float = 4.0,
        neutral_reset_time: float = 1.5,
        use_gemini_stock_detection: bool = False,
        use_smart_gemini: bool = True,
        gemini_api_key: Optional[str] = None,
        use_scene_classification: bool = True,
        use_parallel_ocr: bool = False,  # Disabled by default - pool creation overhead
        ocr_workers: int = 4,
    ):
        """Initialize the detector.

        Args:
            player_perspective: Which player to analyze (1 or 2)
            hit_threshold: Minimum damage increase to count as a hit
            neutral_reset_time: Seconds without damage to return to neutral.
                Used as fallback when scene classification is unavailable.
            use_gemini_stock_detection: Enable FULL Gemini detection (check every frame).
                More accurate but highest cost (~$0.0001 per frame).
            use_smart_gemini: Enable SMART Gemini detection (default True).
                Uses Gemini only for uncertain situations:
                - Low-percent kills (damage drops but was below 50% threshold)
                - GAME! screens (final stock, no damage reset after)
                Cost: ~$0.0001 per uncertain frame (typically 1-5 per game).
            gemini_api_key: Gemini API key. If not provided, uses GEMINI_API_KEY
                environment variable.
            use_scene_classification: Enable ML-based scene classification for
                position-based neutral reset. When both players are grounded,
                returns to neutral immediately instead of waiting for timeout.
            use_parallel_ocr: Enable parallel OCR processing for faster damage
                reading. Uses multiprocessing to distribute OCR across workers.
            ocr_workers: Number of parallel OCR workers (default 4).
        """
        self.player_perspective = player_perspective
        self.hit_threshold = hit_threshold
        self.neutral_reset_time = neutral_reset_time
        self.use_gemini_stock_detection = use_gemini_stock_detection
        self.use_smart_gemini = use_smart_gemini
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.use_scene_classification = use_scene_classification
        self.use_parallel_ocr = use_parallel_ocr
        self.ocr_workers = ocr_workers

        # Import scene classifier if enabled
        self._scene_classifier = None
        if self.use_scene_classification:
            try:
                from runitback.cv.scene_classifier import SceneClassifier
                self._scene_classifier = SceneClassifier()
                if self._scene_classifier.available:
                    logger.info("Scene classification enabled for position-based neutral reset")
                else:
                    logger.warning("Scene classifier model not available, falling back to time-based reset")
                    self._scene_classifier = None
            except ImportError as e:
                logger.warning(f"Could not import SceneClassifier: {e}")
                self._scene_classifier = None

        # Import stock counter detector if any Gemini mode is enabled
        self._stock_counter_tracker = None
        if self.use_gemini_stock_detection or self.use_smart_gemini:
            try:
                from runitback.cv.stock_counter_detector import StockCounterTracker
                self._stock_counter_tracker = StockCounterTracker(use_gemini=True)
                if self.use_gemini_stock_detection:
                    logger.info("Full Gemini stock counter detection enabled")
                elif self.use_smart_gemini:
                    logger.info("Smart Gemini stock detection enabled (uncertain situations only)")
            except ImportError as e:
                logger.warning(f"Could not import StockCounterTracker: {e}")
                self.use_gemini_stock_detection = False
                self.use_smart_gemini = False

    def detect_states(
        self,
        frames: list,
        frame_interval: float = 0.5,
        game_end_times: Optional[list[float]] = None,
        game_start_times: Optional[list[float]] = None,
    ) -> GameStateAnalysis:
        """Detect game states from a list of frames.

        Args:
            frames: List of Frame objects (with .image and .timestamp)
            frame_interval: Time between frames in seconds
            game_end_times: Optional list of known game end timestamps (from CLI game detection).
                If provided, smart Gemini will check these times for GAME! screens
                to detect final stock losses.
            game_start_times: Optional list of known game start timestamps.
                If provided, stock loss detection will be skipped for a few seconds
                after each game starts (to avoid false positives from GO! screen).

        Returns:
            GameStateAnalysis with detected segments
        """
        if not frames:
            return GameStateAnalysis(
                player_perspective=self.player_perspective,
                neutral_segments=[],
                advantage_segments=[],
                disadvantage_segments=[],
                frame_states=[],
            )

        # Store game times for use in smart Gemini postprocessing
        self._game_end_times = game_end_times or []
        self._game_start_times = game_start_times or []

        # Reset stock counter tracker for new analysis
        if self._stock_counter_tracker:
            self._stock_counter_tracker.reset()

        # Extract damage values from each frame
        frame_states = self._extract_frame_states(frames)

        # Detect state transitions (pass frames for Gemini detection)
        self._detect_state_transitions(frame_states, frames)

        # Merge consecutive frames into segments
        segments = self._build_segments(frame_states)

        # Filter segments to only include those within game boundaries
        if self._game_start_times and self._game_end_times:
            segments = self._filter_segments_by_game_bounds(segments)

        return GameStateAnalysis(
            player_perspective=self.player_perspective,
            neutral_segments=[s for s in segments if s.state == GameState.NEUTRAL],
            advantage_segments=[s for s in segments if s.state == GameState.ADVANTAGE],
            disadvantage_segments=[s for s in segments if s.state == GameState.DISADVANTAGE],
            frame_states=frame_states,
        )

    def _is_within_game_bounds(self, timestamp: float) -> bool:
        """Check if a timestamp falls within any game's boundaries.

        Args:
            timestamp: The timestamp to check

        Returns:
            True if within game bounds, False if outside (or no bounds defined)
        """
        if not self._game_start_times or not self._game_end_times:
            # No bounds defined - consider all frames as within bounds
            return True

        # Check if timestamp is within any game interval
        for start, end in zip(self._game_start_times, self._game_end_times):
            if start <= timestamp <= end:
                return True
        return False

    def _extract_frame_states(self, frames: list) -> list[FrameState]:
        """Extract damage percentages, stock information, and scene state from frames.

        Uses parallel OCR when enabled and skips frames outside game boundaries
        for performance optimization.

        Args:
            frames: List of Frame objects

        Returns:
            List of FrameState with damage values, stock counts, and scene state
        """
        frame_states: list[FrameState] = []
        prev_p1_damage: Optional[float] = None
        prev_p2_damage: Optional[float] = None

        # Batch scene classification for efficiency
        scene_states = []
        if self._scene_classifier:
            images = [f.image for f in frames]
            classifications = self._scene_classifier.classify_batch(images)
            scene_states = [c.state.value for c in classifications]
            logger.info(f"Classified {len(scene_states)} frames with scene classifier")

        # Option 1: Identify which frames are within game boundaries
        # Skip OCR for frames outside game bounds to save processing time
        frames_to_ocr: list[tuple[int, np.ndarray, np.ndarray]] = []  # (index, p1_roi, p2_roi)
        frames_outside_bounds = 0

        for i, frame in enumerate(frames):
            if self._is_within_game_bounds(frame.timestamp):
                p1_roi = get_roi(frame.image, P1_DAMAGE_ROI)
                p2_roi = get_roi(frame.image, P2_DAMAGE_ROI)
                frames_to_ocr.append((i, p1_roi, p2_roi))
            else:
                frames_outside_bounds += 1

        if frames_outside_bounds > 0:
            logger.info(f"Skipping OCR for {frames_outside_bounds} frames outside game boundaries")

        # Option 5: Process OCR in parallel if enabled and we have enough frames
        ocr_results: dict[int, tuple[Optional[float], float, Optional[float], float]] = {}

        if self.use_parallel_ocr and len(frames_to_ocr) >= 4:
            # Collect all ROIs for parallel processing
            p1_rois = [roi[1] for roi in frames_to_ocr]
            p2_rois = [roi[2] for roi in frames_to_ocr]

            logger.info(f"Processing {len(frames_to_ocr)} frames with parallel OCR ({self.ocr_workers} workers)")

            # Process P1 and P2 damage ROIs in parallel batches
            p1_results = read_damage_batch_parallel(p1_rois, num_workers=self.ocr_workers)
            p2_results = read_damage_batch_parallel(p2_rois, num_workers=self.ocr_workers)

            # Map results back to frame indices
            for idx, (frame_idx, _, _) in enumerate(frames_to_ocr):
                p1_damage, p1_conf = p1_results[idx]
                p2_damage, p2_conf = p2_results[idx]
                ocr_results[frame_idx] = (p1_damage, p1_conf, p2_damage, p2_conf)
        else:
            # Sequential OCR processing (for small batches or when parallel is disabled)
            for frame_idx, p1_roi, p2_roi in frames_to_ocr:
                p1_damage, p1_conf = read_damage_percentage(p1_roi)
                p2_damage, p2_conf = read_damage_percentage(p2_roi)
                ocr_results[frame_idx] = (p1_damage, p1_conf, p2_damage, p2_conf)

        # Build frame states from results
        for i, frame in enumerate(frames):
            # Get scene state if available
            scene_state = scene_states[i] if scene_states else None

            # Get OCR results if this frame was processed
            if i in ocr_results:
                p1_damage, p1_conf, p2_damage, p2_conf = ocr_results[i]

                # Detect stock counts from HUD stock icons
                p1_stocks, p2_stocks = detect_stock_counts(frame.image)

                # Log damage readings periodically for debugging
                if i % 20 == 0:  # Every 10 seconds at 0.5s intervals
                    logger.debug(
                        f"Frame {i} ({frame.timestamp:.1f}s): "
                        f"P1={p1_damage}% (conf={p1_conf:.2f}), "
                        f"P2={p2_damage}% (conf={p2_conf:.2f}), "
                        f"scene={scene_state}"
                    )

                # Log stock counts periodically for debugging
                if i % 20 == 0:
                    logger.debug(
                        f"Frame {i} ({frame.timestamp:.1f}s): "
                        f"Stock counts: P1={p1_stocks}, P2={p2_stocks}"
                    )
            else:
                # Frame outside game bounds - no OCR performed
                p1_damage = None
                p2_damage = None
                p1_stocks = None
                p2_stocks = None

            # Use previous value as fallback if OCR fails
            if p1_damage is None and prev_p1_damage is not None:
                p1_damage = prev_p1_damage
            if p2_damage is None and prev_p2_damage is not None:
                p2_damage = prev_p2_damage

            # Check if players are visible (not in magnifying glass)
            # For now, we assume visible if we can read damage
            p1_visible = p1_damage is not None
            p2_visible = p2_damage is not None

            frame_states.append(
                FrameState(
                    timestamp=frame.timestamp,
                    frame_index=i,
                    p1_damage=p1_damage,
                    p2_damage=p2_damage,
                    p1_stocks=p1_stocks,
                    p2_stocks=p2_stocks,
                    p1_visible=p1_visible,
                    p2_visible=p2_visible,
                    state=GameState.UNKNOWN,
                    scene_state=scene_state,
                )
            )

            # Update previous values
            if p1_damage is not None:
                prev_p1_damage = p1_damage
            if p2_damage is not None:
                prev_p2_damage = p2_damage

        return frame_states

    def _detect_state_transitions(
        self, frame_states: list[FrameState], frames: Optional[list] = None
    ) -> None:
        """Detect state transitions based on damage changes.

        Modifies frame_states in place to set the state attribute.

        Args:
            frame_states: List of FrameState to analyze
            frames: Optional list of Frame objects (for Gemini stock detection)
        """
        if not frame_states:
            return

        # Start in neutral
        current_state = GameState.NEUTRAL
        last_hit_time: Optional[float] = None

        # Track previous damage values
        prev_p1_damage: Optional[float] = None
        prev_p2_damage: Optional[float] = None

        # Track stock counts to detect changes
        # Start at 3 stocks each (standard SSBU match)
        prev_p1_stocks: int = 3
        prev_p2_stocks: int = 3

        # Track last stock loss time to prevent duplicate detections
        last_p1_stock_loss_time: Optional[float] = None
        last_p2_stock_loss_time: Optional[float] = None
        STOCK_LOSS_COOLDOWN = 10.0  # Seconds before another stock loss can be detected for same player

        # Track consecutive low-damage frames to confirm stock loss
        # (helps filter OCR noise where damage briefly reads as 0)
        p1_low_damage_frames = 0
        p2_low_damage_frames = 0
        p1_high_damage_before_low: Optional[float] = None  # Damage before low streak started
        p2_high_damage_before_low: Optional[float] = None
        CONFIRM_FRAMES = 3  # Require 3 consecutive low frames (1.5 seconds at 0.5s interval)
        MIN_KILL_DAMAGE = 50  # Minimum damage before stock loss is plausible (filters OCR noise)
        LOW_PERCENT_KILL_THRESHOLD = 15  # Damage drop to 0 from this range triggers smart Gemini

        # Track uncertain situations for smart Gemini
        # These are frame indices where we'll later check with Gemini
        uncertain_low_percent_kills: list[int] = []  # Frame indices with possible low-percent kills

        # Track maximum damage seen to handle OCR noise
        # This prevents losing track of high damage when OCR has intermediate errors
        p1_recent_max_damage: float = 0.0
        p2_recent_max_damage: float = 0.0

        # Track consecutive frames with None damage (game transition detection)
        # When we see many None frames in a row, it means we're in a game transition
        # (character select, results screen, etc.) and should reset state
        consecutive_none_frames = 0
        GAME_TRANSITION_THRESHOLD = 10  # 5 seconds of None damage = game transition
        in_game_transition = False  # Track if we're in a transition period

        # Max stocks per player per game (enforced in _enforce_stock_loss_limit)
        MAX_STOCKS_PER_PLAYER = 3

        # Build frame lookup for Gemini detection
        frame_by_index = {}
        if frames and self.use_gemini_stock_detection:
            frame_by_index = {i: f for i, f in enumerate(frames)}

        for fs in frame_states:
            # Detect game transitions by counting consecutive frames with no damage reading
            if fs.p1_damage is None and fs.p2_damage is None:
                consecutive_none_frames += 1
                if consecutive_none_frames >= GAME_TRANSITION_THRESHOLD:
                    if not in_game_transition:
                        logger.debug(f"Game transition detected at {fs.timestamp:.1f}s (resetting state)")
                    in_game_transition = True
                    # Reset tracking state for new game
                    p1_recent_max_damage = 0.0
                    p2_recent_max_damage = 0.0
                    p1_low_damage_frames = 0
                    p2_low_damage_frames = 0
                    p1_high_damage_before_low = None
                    p2_high_damage_before_low = None
            else:
                # Only exit transition mode after we see BOTH players with valid damage
                # This prevents false positives from intermittent OCR during transitions
                if in_game_transition:
                    if fs.p1_damage is not None and fs.p2_damage is not None:
                        in_game_transition = False
                        consecutive_none_frames = 0
                        logger.debug(f"Game transition ended at {fs.timestamp:.1f}s")
                else:
                    consecutive_none_frames = 0

            # Calculate damage deltas
            p1_delta = 0.0
            p2_delta = 0.0

            # Detect stock losses using multiple methods:
            # 1. PRIMARY: Stock splash screen (shows "X-Y" when stock is lost)
            # 2. FALLBACK: Damage reset detection (damage goes to 0)
            p1_stock_loss = False
            p2_stock_loss = False

            # Detect stock losses using damage reset as primary indicator
            # When a player loses a stock, their damage resets to 0%
            # Require consecutive low damage frames to filter OCR noise
            cooldown_p1_ok = (
                last_p1_stock_loss_time is None or
                fs.timestamp - last_p1_stock_loss_time >= STOCK_LOSS_COOLDOWN
            )
            cooldown_p2_ok = (
                last_p2_stock_loss_time is None or
                fs.timestamp - last_p2_stock_loss_time >= STOCK_LOSS_COOLDOWN
            )

            # Update recent max damage tracking
            # Only update when damage is reasonably high (>10%) and below 200% (filter OCR errors)
            if fs.p1_damage is not None and 10 < fs.p1_damage < 200:
                p1_recent_max_damage = max(p1_recent_max_damage, fs.p1_damage)
            if fs.p2_damage is not None and 10 < fs.p2_damage < 200:
                p2_recent_max_damage = max(p2_recent_max_damage, fs.p2_damage)

            # Track P1 low damage frames and remember high damage before the streak
            # Stock loss resets damage to exactly 0%, so only count frames at 0-2%
            # (allowing small OCR variance like "1%" when it should be "0%")
            if fs.p1_damage is not None and fs.p1_damage <= 2:
                if p1_low_damage_frames == 0:
                    # Use recent max damage instead of just prev_p1_damage
                    # This handles cases where OCR had errors between high and low
                    p1_high_damage_before_low = p1_recent_max_damage
                p1_low_damage_frames += 1
            else:
                p1_low_damage_frames = 0
                p1_high_damage_before_low = None
                # Don't decay max here - keep tracking max until stock loss is confirmed
                # Max will be reset after stock loss detection

            # Track P2 low damage frames and remember high damage before the streak
            # Stock loss resets damage to exactly 0%, so only count frames at 0-2%
            if fs.p2_damage is not None and fs.p2_damage <= 2:
                if p2_low_damage_frames == 0:
                    # Use recent max damage instead of just prev_p2_damage
                    p2_high_damage_before_low = p2_recent_max_damage
                p2_low_damage_frames += 1
            else:
                p2_low_damage_frames = 0
                p2_high_damage_before_low = None
                # Don't decay max here - keep tracking max until stock loss is confirmed

            # Check if this is a game transition (both players reset to 0% from HIGH damage)
            # This happens between games and should NOT be counted as a stock loss
            # Both players must have had significant damage (>= MIN_KILL_DAMAGE) before resetting
            # to distinguish from game start (where one player may have been at 0% the whole time)
            both_reset = (p1_low_damage_frames >= CONFIRM_FRAMES and
                          p2_low_damage_frames >= CONFIRM_FRAMES and
                          p1_high_damage_before_low is not None and
                          p2_high_damage_before_low is not None and
                          p1_high_damage_before_low >= MIN_KILL_DAMAGE and
                          p2_high_damage_before_low >= MIN_KILL_DAMAGE)

            # Skip stock loss detection entirely during game transitions
            # Also skip near game starts (avoid GO! screen false positives)
            if in_game_transition or self._is_near_game_start(fs.timestamp):
                fs.state = current_state
                continue

            # P1 stock loss: confirm only after consecutive low damage frames
            # and only if damage was high enough before the low streak
            # MIN_KILL_DAMAGE filters OCR noise (e.g., 62% misread as 3%)
            # Note: We detect ALL potential stock losses here and filter to max 3 per game later
            # This ensures we keep the most confident detections, not just the first 3
            if (p1_low_damage_frames == CONFIRM_FRAMES and
                p1_high_damage_before_low is not None and
                p1_high_damage_before_low >= MIN_KILL_DAMAGE and cooldown_p1_ok and
                not both_reset):
                p1_stock_loss = True
                last_p1_stock_loss_time = fs.timestamp
                # Reset recent max after stock loss (player starts fresh)
                p1_recent_max_damage = 0.0
                logger.debug(f"P1 stock loss detected (damage reset): {p1_high_damage_before_low:.0f}% -> low at {fs.timestamp:.1f}s")
            # SMART GEMINI: Check for low-percent kills (damage dropped but was below MIN_KILL_DAMAGE)
            elif (p1_low_damage_frames == CONFIRM_FRAMES and
                  p1_high_damage_before_low is not None and
                  LOW_PERCENT_KILL_THRESHOLD <= p1_high_damage_before_low < MIN_KILL_DAMAGE and
                  cooldown_p1_ok and not both_reset):
                # Possible low-percent kill - mark for Gemini check
                uncertain_low_percent_kills.append((fs.frame_index, 1, p1_high_damage_before_low))
                logger.debug(f"P1 uncertain low-percent kill: {p1_high_damage_before_low:.0f}% -> low at {fs.timestamp:.1f}s (will check with Gemini)")

            # P2 stock loss: confirm only after consecutive low damage frames
            # and only if damage was high enough before the low streak
            if (p2_low_damage_frames == CONFIRM_FRAMES and
                p2_high_damage_before_low is not None and
                p2_high_damage_before_low >= MIN_KILL_DAMAGE and cooldown_p2_ok and
                not both_reset):
                p2_stock_loss = True
                last_p2_stock_loss_time = fs.timestamp
                # Reset recent max after stock loss (player starts fresh)
                p2_recent_max_damage = 0.0
                logger.debug(f"P2 stock loss detected (damage reset): {p2_high_damage_before_low:.0f}% -> low at {fs.timestamp:.1f}s")
            # SMART GEMINI: Check for low-percent kills (damage dropped but was below MIN_KILL_DAMAGE)
            elif (p2_low_damage_frames == CONFIRM_FRAMES and
                  p2_high_damage_before_low is not None and
                  LOW_PERCENT_KILL_THRESHOLD <= p2_high_damage_before_low < MIN_KILL_DAMAGE and
                  cooldown_p2_ok and not both_reset):
                # Possible low-percent kill - mark for Gemini check
                uncertain_low_percent_kills.append((fs.frame_index, 2, p2_high_damage_before_low))
                logger.debug(f"P2 uncertain low-percent kill: {p2_high_damage_before_low:.0f}% -> low at {fs.timestamp:.1f}s (will check with Gemini)")

            # If both reset at same time, log it as game transition
            if both_reset and p1_low_damage_frames == CONFIRM_FRAMES:
                logger.debug(f"Game transition detected at {fs.timestamp:.1f}s (both players reset to 0%)")

            # GEMINI STOCK COUNTER DETECTION
            # Use Gemini to detect stock counter splash for additional accuracy
            # This catches low-percent kills that damage-reset would miss
            if (self.use_gemini_stock_detection and
                self._stock_counter_tracker is not None and
                fs.frame_index in frame_by_index):

                frame = frame_by_index[fs.frame_index]
                (
                    gemini_p1_lost,
                    gemini_p2_lost,
                    gemini_p1_stocks,
                    gemini_p2_stocks,
                    gemini_is_game_end,
                ) = self._stock_counter_tracker.update(
                    frame.image,
                    fs.timestamp,
                    api_key=self.gemini_api_key,
                )

                # Use Gemini detection if it found a stock loss
                # Note: We detect ALL stock losses here and filter to max 3 per game later
                if gemini_p1_lost and cooldown_p1_ok:
                    if not p1_stock_loss:
                        logger.info(
                            f"P1 stock loss detected by Gemini at {fs.timestamp:.1f}s "
                            f"(stocks: {gemini_p1_stocks})"
                        )
                    p1_stock_loss = True
                    last_p1_stock_loss_time = fs.timestamp
                    p1_recent_max_damage = 0.0

                if gemini_p2_lost and cooldown_p2_ok:
                    if not p2_stock_loss:
                        logger.info(
                            f"P2 stock loss detected by Gemini at {fs.timestamp:.1f}s "
                            f"(stocks: {gemini_p2_stocks})"
                        )
                    p2_stock_loss = True
                    last_p2_stock_loss_time = fs.timestamp
                    p2_recent_max_damage = 0.0

                # Update stock counts from Gemini if available
                if gemini_p1_stocks is not None:
                    fs.p1_stocks = gemini_p1_stocks
                if gemini_p2_stocks is not None:
                    fs.p2_stocks = gemini_p2_stocks

            # Calculate damage deltas for hit detection
            if fs.p1_damage is not None and prev_p1_damage is not None:
                p1_delta = fs.p1_damage - prev_p1_damage
                # Reset delta if damage dropped significantly (stock loss)
                if p1_delta < -20:
                    p1_delta = 0.0

            if fs.p2_damage is not None and prev_p2_damage is not None:
                p2_delta = fs.p2_damage - prev_p2_damage
                # Reset delta if damage dropped significantly (stock loss)
                if p2_delta < -20:
                    p2_delta = 0.0

            # Record stock losses on the frame state
            fs.p1_stock_loss = p1_stock_loss
            fs.p2_stock_loss = p2_stock_loss

            # Stock loss handling:
            # The stock loss is the RESULT of disadvantage (getting combo'd/edgeguarded)
            # So the frame where the stock was lost should remain in advantage/disadvantage
            # Then we transition to neutral for the NEXT frame (respawn)
            if p1_stock_loss or p2_stock_loss:
                # Determine whose advantage/disadvantage this was based on who lost the stock
                if p1_stock_loss:
                    # P1 lost stock = P1 was in disadvantage (opponent took the stock)
                    if self.player_perspective == 1:
                        fs.state = GameState.DISADVANTAGE
                    else:
                        fs.state = GameState.ADVANTAGE
                elif p2_stock_loss:
                    # P2 lost stock = P2 was in disadvantage (player 1 took the stock)
                    if self.player_perspective == 1:
                        fs.state = GameState.ADVANTAGE
                    else:
                        fs.state = GameState.DISADVANTAGE

                # Reset to neutral for the NEXT iteration (respawn state)
                current_state = GameState.NEUTRAL
                last_hit_time = None

                # Update previous damage values
                if fs.p1_damage is not None:
                    prev_p1_damage = fs.p1_damage
                if fs.p2_damage is not None:
                    prev_p2_damage = fs.p2_damage
                continue

            # Determine who got hit
            p1_got_hit = p1_delta >= self.hit_threshold
            p2_got_hit = p2_delta >= self.hit_threshold

            # Determine disadvantage type from scene state
            disadvantage_type = self._get_disadvantage_type(fs.scene_state)

            # State transition logic
            if p1_got_hit and p2_got_hit:
                # Trade - both took damage
                # Brief trade returns to neutral quickly
                last_hit_time = fs.timestamp
                # Keep current state for now (trade doesn't immediately change state)
            elif p1_got_hit:
                # Player 1 got hit
                if self.player_perspective == 1:
                    current_state = GameState.DISADVANTAGE
                else:
                    current_state = GameState.ADVANTAGE
                last_hit_time = fs.timestamp
            elif p2_got_hit:
                # Player 2 got hit
                if self.player_perspective == 1:
                    current_state = GameState.ADVANTAGE
                else:
                    current_state = GameState.DISADVANTAGE
                last_hit_time = fs.timestamp
            else:
                # No significant damage this frame
                # Check if we should return to neutral
                if current_state != GameState.NEUTRAL and last_hit_time is not None:
                    # Position-based reset: both_grounded = return to neutral immediately
                    if fs.scene_state == "both_grounded":
                        current_state = GameState.NEUTRAL
                        last_hit_time = None
                    else:
                        # Fallback: time-based reset when scene classification unavailable
                        time_since_hit = fs.timestamp - last_hit_time
                        if (
                            time_since_hit >= self.neutral_reset_time
                            and fs.p1_visible
                            and fs.p2_visible
                        ):
                            current_state = GameState.NEUTRAL
                            last_hit_time = None

            fs.state = current_state
            fs.disadvantage_type = disadvantage_type if current_state != GameState.NEUTRAL else DisadvantageType.UNKNOWN

            # Update previous damage values
            if fs.p1_damage is not None:
                prev_p1_damage = fs.p1_damage
            if fs.p2_damage is not None:
                prev_p2_damage = fs.p2_damage

        # SMART GEMINI POST-PROCESSING
        # Check uncertain situations with Gemini Vision API
        if self.use_smart_gemini and self._stock_counter_tracker and frames:
            self._smart_gemini_postprocess(
                frame_states, frames, uncertain_low_percent_kills
            )

        # FINAL VALIDATION: Enforce max 3 stock losses per player per game
        # This catches any excess stock losses added by Smart Gemini postprocessing
        self._enforce_stock_loss_limit(frame_states, MAX_STOCKS_PER_PLAYER)

    def _is_near_game_start(self, timestamp: float, cooldown: float = 5.0) -> bool:
        """Check if timestamp is within cooldown period after a game start.

        Args:
            timestamp: The timestamp to check
            cooldown: Seconds after game start to skip (default 5 seconds)

        Returns:
            True if timestamp is too close to a game start
        """
        if not hasattr(self, '_game_start_times') or not self._game_start_times:
            return False

        for start_time in self._game_start_times:
            # Skip frames within cooldown seconds after game start
            if 0 <= timestamp - start_time < cooldown:
                return True
        return False

    def _get_current_game_index(self, timestamp: float) -> int:
        """Get the game index (0-based) for a given timestamp.

        Args:
            timestamp: The timestamp to check

        Returns:
            Game index (0, 1, 2, etc.) or -1 if not within any game
        """
        if not self._game_start_times or not self._game_end_times:
            # No game boundaries - treat as single game (index 0)
            return 0

        for i, (start, end) in enumerate(zip(self._game_start_times, self._game_end_times)):
            if start <= timestamp <= end:
                return i
        return -1

    def _enforce_stock_loss_limit(
        self, frame_states: list[FrameState], max_stocks: int = 3
    ) -> None:
        """Enforce maximum stock losses per player per game.

        When more than max_stocks losses are detected for a player in a game,
        keeps only the first max_stocks chronologically and removes the rest.
        Also logs a warning since this indicates potential detection issues.

        Args:
            frame_states: List of FrameState to validate (modified in place)
            max_stocks: Maximum stock losses allowed per player per game
        """
        # Build index for quick lookups
        fs_by_idx = {fs.frame_index: fs for fs in frame_states}

        # Collect all stock losses per game (chronologically ordered since frame_states is ordered)
        losses_by_game: dict[int, dict[str, list[tuple[int, float]]]] = {}
        # Structure: {game_idx: {'p1': [(frame_idx, timestamp), ...], 'p2': [...]}}

        for fs in frame_states:
            game_idx = self._get_current_game_index(fs.timestamp)
            if game_idx < 0:
                continue  # Skip frames outside game bounds

            if game_idx not in losses_by_game:
                losses_by_game[game_idx] = {'p1': [], 'p2': []}

            if fs.p1_stock_loss:
                losses_by_game[game_idx]['p1'].append((fs.frame_index, fs.timestamp))

            if fs.p2_stock_loss:
                losses_by_game[game_idx]['p2'].append((fs.frame_index, fs.timestamp))

        # For each game, keep only the first max_stocks losses chronologically
        for game_idx, player_losses in losses_by_game.items():
            for player_key in ['p1', 'p2']:
                losses = player_losses[player_key]
                if len(losses) <= max_stocks:
                    continue  # No filtering needed

                # Log warning - detecting more than max stocks indicates potential issues
                timestamps = [f"{ts:.1f}s" for _, ts in losses]
                logger.warning(
                    f"Game {game_idx + 1} {player_key.upper()}: detected {len(losses)} stock losses "
                    f"(max is {max_stocks}). Timestamps: {timestamps}. "
                    f"Keeping first {max_stocks}, but results may be inaccurate."
                )

                # Keep only first max_stocks (chronological order)
                to_keep = set(frame_idx for frame_idx, _ in losses[:max_stocks])
                to_remove = [(frame_idx, ts) for frame_idx, ts in losses if frame_idx not in to_keep]

                # Remove the excess detections
                for frame_idx, timestamp in to_remove:
                    fs = fs_by_idx.get(frame_idx)
                    if fs:
                        if player_key == 'p1':
                            fs.p1_stock_loss = False
                        else:
                            fs.p2_stock_loss = False
                        logger.debug(
                            f"Removed excess {player_key.upper()} stock loss at {timestamp:.1f}s "
                            f"(game {game_idx + 1}, keeping first {max_stocks})"
                        )

        # Log final stock loss counts
        final_counts: dict[int, dict[str, int]] = {}
        for fs in frame_states:
            game_idx = self._get_current_game_index(fs.timestamp)
            if game_idx < 0:
                continue
            if game_idx not in final_counts:
                final_counts[game_idx] = {'p1': 0, 'p2': 0}
            if fs.p1_stock_loss:
                final_counts[game_idx]['p1'] += 1
            if fs.p2_stock_loss:
                final_counts[game_idx]['p2'] += 1

        for game_idx, counts in sorted(final_counts.items()):
            logger.debug(
                f"Game {game_idx + 1} final stock losses: P1={counts['p1']}, P2={counts['p2']}"
            )

    def _get_disadvantage_type(self, scene_state: Optional[str]) -> DisadvantageType:
        """Map scene state to disadvantage type.

        Args:
            scene_state: Scene classification result (string value of SceneState)

        Returns:
            DisadvantageType based on scene classification
        """
        if scene_state is None:
            return DisadvantageType.UNKNOWN

        # Map scene states to disadvantage types
        # Note: We can't determine WHO is in disadvantage from scene alone
        # That's determined by damage tracking (who took the hit)
        scene_to_disadvantage = {
            # Juggle situations
            "one_airborne": DisadvantageType.JUGGLE,
            "both_airborne": DisadvantageType.JUGGLE,
            # Offstage/edgeguard situations
            "one_offstage": DisadvantageType.OFFSTAGE,
            "both_offstage": DisadvantageType.OFFSTAGE,
            # Ledgetrap situations
            "one_on_ledge": DisadvantageType.LEDGETRAP,
            "both_on_ledge": DisadvantageType.LEDGETRAP,
            # Tech chase situations
            "one_knocked_down": DisadvantageType.TECH_CHASE,
            "both_knocked_down": DisadvantageType.TECH_CHASE,
            # Respawning
            "respawning": DisadvantageType.RESPAWNING,
            # Neutral (shouldn't reach here in disadvantage)
            "both_grounded": DisadvantageType.UNKNOWN,
            "non_gameplay": DisadvantageType.UNKNOWN,
        }

        return scene_to_disadvantage.get(scene_state, DisadvantageType.UNKNOWN)

    def _smart_gemini_postprocess(
        self,
        frame_states: list[FrameState],
        frames: list,
        uncertain_low_percent_kills: list,
    ) -> None:
        """Post-process with Gemini for uncertain stock loss situations.

        Uses Gemini Vision API selectively for:
        1. Low-percent kills that damage-reset missed (below 50% threshold)
        2. GAME! screens at the end of each game (final stock has no damage reset)

        Args:
            frame_states: List of FrameState to update
            frames: List of Frame objects with images
            uncertain_low_percent_kills: List of (frame_index, player, damage) tuples
        """
        from runitback.cv.stock_counter_detector import detect_stock_counter_gemini

        # Build frame lookup
        frame_by_index = {i: f for i, f in enumerate(frames)}
        fs_by_index = {fs.frame_index: fs for fs in frame_states}

        gemini_calls = 0

        # 1. Check uncertain low-percent kills
        # The stock counter splash appears BEFORE the damage reset (during kill animation)
        # So we need to check frames BEFORE the detection point
        for frame_idx, player, damage in uncertain_low_percent_kills:
            # Skip if near a game start (avoid GO! screen false positives)
            fs = fs_by_index.get(frame_idx)
            if fs and self._is_near_game_start(fs.timestamp):
                logger.debug(f"Skipping uncertain kill at {fs.timestamp:.1f}s (near game start)")
                continue

            # Check frames from 3 seconds before to the detection point
            # Stock counter splash typically appears 1-2 seconds after the kill
            found_kill = False
            for offset in range(-6, 3):  # -3s to +1.5s at 0.5s intervals
                check_idx = frame_idx + offset
                if check_idx not in frame_by_index:
                    continue

                frame = frame_by_index[check_idx]
                detected, p1_stocks, p2_stocks, is_game_end = detect_stock_counter_gemini(
                    frame.image, api_key=self.gemini_api_key
                )
                gemini_calls += 1

                if detected or is_game_end:
                    # Find the best frame to mark the stock loss (use check_idx)
                    check_fs = fs_by_index.get(check_idx)
                    if check_fs:
                        # Skip if near a game start (avoid GO! screen false positives)
                        if self._is_near_game_start(check_fs.timestamp):
                            logger.debug(f"Skipping Gemini detection at {check_fs.timestamp:.1f}s (near game start)")
                            continue

                        if player == 1:
                            check_fs.p1_stock_loss = True
                            logger.info(
                                f"Smart Gemini confirmed P1 low-percent kill "
                                f"({damage:.0f}%) at {check_fs.timestamp:.1f}s"
                            )
                        else:
                            check_fs.p2_stock_loss = True
                            logger.info(
                                f"Smart Gemini confirmed P2 low-percent kill "
                                f"({damage:.0f}%) at {check_fs.timestamp:.1f}s"
                            )
                        found_kill = True
                        break  # Found it, no need to check more frames

            if not found_kill:
                logger.debug(f"Smart Gemini did not confirm low-percent kill at frame {frame_idx}")

        # 2. Detect GAME! screens - check frames near potential game endings
        # A game ends when total detected stock losses reach certain thresholds
        # For a 3-stock game: game ends when one player has lost 3 stocks total
        # We look for the GAME! screen in the frames following the last detected kill
        game_end_candidates = self._find_game_end_candidates(frame_states)

        for end_frame_idx in game_end_candidates:
            # Check frames BEFORE the game end
            # GAME! screen appears 30-60 seconds before the "end timestamp" because
            # the end timestamp includes results screen, character select, etc.
            # The GAME! screen is only visible for ~1-2 seconds, so we need fine granularity
            # Search from -40s to -5s before end, checking every 1s (2 frames)

            for offset in range(-80, 0, 2):  # -40s to 0s, every 1s
                check_idx = end_frame_idx + offset
                if check_idx not in frame_by_index:
                    continue

                # Skip if near a game start (avoid GO! screen false positives)
                check_fs = fs_by_index.get(check_idx)
                if check_fs and self._is_near_game_start(check_fs.timestamp):
                    continue

                frame = frame_by_index[check_idx]
                detected, p1_stocks, p2_stocks, is_game_end = detect_stock_counter_gemini(
                    frame.image, api_key=self.gemini_api_key
                )
                gemini_calls += 1

                if is_game_end:
                    fs = fs_by_index.get(check_idx)
                    if fs:
                        # Determine who lost based on stock counts or previous state
                        if p1_stocks == 0:
                            fs.p1_stock_loss = True
                            logger.info(f"Smart Gemini detected GAME! P1 lost final stock at {fs.timestamp:.1f}s")
                        elif p2_stocks == 0:
                            fs.p2_stock_loss = True
                            logger.info(f"Smart Gemini detected GAME! P2 lost final stock at {fs.timestamp:.1f}s")
                        else:
                            self._infer_game_end_loser(frame_states, check_idx)
                    break  # Found GAME!, move to next game end candidate

        if gemini_calls > 0:
            logger.info(f"Smart Gemini made {gemini_calls} API calls for uncertain situations")

    def _find_game_end_candidates(self, frame_states: list[FrameState]) -> list[int]:
        """Find frame indices that might be near game endings.

        Uses multiple sources:
        1. PRIMARY: Known game end times (from CLI game detection)
        2. FALLBACK: Stock loss patterns and damage gaps

        Returns:
            List of frame indices to check for GAME! screen
        """
        candidates = []

        # PRIMARY: Use known game end times if provided
        # These come from the CLI's detect_games() which uses GO!/GAME! screen detection
        if hasattr(self, '_game_end_times') and self._game_end_times:
            logger.debug(f"Using {len(self._game_end_times)} known game end times for GAME! detection")
            for end_time in self._game_end_times:
                # Find the frame closest to this end time
                closest_idx = None
                closest_diff = float('inf')
                for fs in frame_states:
                    diff = abs(fs.timestamp - end_time)
                    if diff < closest_diff:
                        closest_diff = diff
                        closest_idx = fs.frame_index
                if closest_idx is not None and closest_idx not in candidates:
                    candidates.append(closest_idx)
            return candidates

        # FALLBACK: Use heuristics when no game end times provided
        logger.debug("No game end times provided, using heuristics for GAME! detection")

        # Track stock losses per player within current "game window"
        p1_losses = 0
        p2_losses = 0

        # Track gaps in damage to detect game transitions
        prev_damage_time = 0.0
        GAP_THRESHOLD = 15.0  # Seconds of no damage suggests game transition

        for i, fs in enumerate(frame_states):
            # Check for long gaps (potential game transition/end)
            if fs.timestamp - prev_damage_time > GAP_THRESHOLD:
                # Add the frame BEFORE the gap as a potential game end
                if i > 0 and frame_states[i-1].frame_index not in candidates:
                    candidates.append(frame_states[i-1].frame_index)
                # Reset per-game stock counts for new game
                p1_losses = 0
                p2_losses = 0

            # Track damage changes
            if fs.p1_damage is not None and fs.p1_damage > 5:
                prev_damage_time = fs.timestamp
            if fs.p2_damage is not None and fs.p2_damage > 5:
                prev_damage_time = fs.timestamp

            # Track stock losses
            if fs.p1_stock_loss:
                p1_losses += 1
                # After 2 losses in current game, check for potential GAME!
                if p1_losses >= 2 and fs.frame_index not in candidates:
                    candidates.append(fs.frame_index)

            if fs.p2_stock_loss:
                p2_losses += 1
                if p2_losses >= 2 and fs.frame_index not in candidates:
                    candidates.append(fs.frame_index)

        # Also check the very end of the video
        if frame_states:
            last_fs = frame_states[-1]
            if last_fs.frame_index not in candidates:
                candidates.append(last_fs.frame_index)

        return candidates

    def _infer_game_end_loser(self, frame_states: list[FrameState], game_end_idx: int) -> None:
        """Infer who lost the game when GAME! is detected but stocks are unclear.

        Uses multiple signals to determine the loser:
        1. Who had higher damage right before the GAME! (they were about to die)
        2. If damage is similar, look at stock loss patterns within this game

        Args:
            frame_states: All frame states
            game_end_idx: Frame index where GAME! was detected
        """
        # Find the frame state at the game end
        fs = next((f for f in frame_states if f.frame_index == game_end_idx), None)
        if not fs:
            return

        # Look at damage values in the frames just BEFORE the GAME! screen
        # The player with higher damage was about to die
        p1_damage_before = 0.0
        p2_damage_before = 0.0

        # Check up to 10 frames (5 seconds) before the GAME! screen
        for check_fs in reversed(frame_states):
            if check_fs.frame_index >= game_end_idx:
                continue
            if check_fs.frame_index < game_end_idx - 10:
                break
            # Get the last known damage values (ignore obviously bad OCR readings > 200%)
            if check_fs.p1_damage is not None and check_fs.p1_damage > p1_damage_before and check_fs.p1_damage < 200:
                p1_damage_before = check_fs.p1_damage
            if check_fs.p2_damage is not None and check_fs.p2_damage > p2_damage_before and check_fs.p2_damage < 200:
                p2_damage_before = check_fs.p2_damage

        logger.debug(f"GAME! at {fs.timestamp:.1f}s - damage before: P1={p1_damage_before:.0f}%, P2={p2_damage_before:.0f}%")

        # Player with significantly higher damage was the one who died
        # (they were in kill percent when the final hit landed)
        damage_diff = p1_damage_before - p2_damage_before

        if damage_diff > 20:  # P1 had much higher damage
            fs.p1_stock_loss = True
            logger.info(f"Smart Gemini inferred P1 lost final stock (higher damage: {p1_damage_before:.0f}%) at {fs.timestamp:.1f}s")
        elif damage_diff < -20:  # P2 had much higher damage
            fs.p2_stock_loss = True
            logger.info(f"Smart Gemini inferred P2 lost final stock (higher damage: {p2_damage_before:.0f}%) at {fs.timestamp:.1f}s")
        else:
            # Similar damage - fall back to stock loss count within this game
            # Find the game start (last game end before this one, or start of video)
            game_start_time = 0.0
            for check_fs in reversed(frame_states):
                if check_fs.frame_index >= game_end_idx:
                    continue
                # A new game starts after a long gap in damage or when stocks reset
                if check_fs.p1_damage is not None and check_fs.p1_damage < 5 and check_fs.p2_damage is not None and check_fs.p2_damage < 5:
                    # Both players at 0% suggests game just started
                    game_start_time = check_fs.timestamp
                    break

            # Count losses within this game only
            p1_losses = 0
            p2_losses = 0
            for check_fs in frame_states:
                if check_fs.timestamp < game_start_time:
                    continue
                if check_fs.frame_index >= game_end_idx:
                    break
                if check_fs.p1_stock_loss:
                    p1_losses += 1
                if check_fs.p2_stock_loss:
                    p2_losses += 1

            if p1_losses > p2_losses:
                fs.p1_stock_loss = True
                logger.info(f"Smart Gemini inferred P1 lost final stock (more losses in game: {p1_losses}) at {fs.timestamp:.1f}s")
            elif p2_losses > p1_losses:
                fs.p2_stock_loss = True
                logger.info(f"Smart Gemini inferred P2 lost final stock (more losses in game: {p2_losses}) at {fs.timestamp:.1f}s")
            else:
                # Still can't determine - this shouldn't happen in a normal game
                # Default to P1 with a warning
                fs.p1_stock_loss = True
                logger.warning(f"Smart Gemini couldn't determine GAME! loser at {fs.timestamp:.1f}s, defaulting to P1")

    def _get_dominant_disadvantage_type(
        self, frame_states: list[FrameState]
    ) -> DisadvantageType:
        """Get the most common disadvantage type in a set of frames.

        Args:
            frame_states: List of FrameState to analyze

        Returns:
            Most common DisadvantageType (excluding UNKNOWN if possible)
        """
        from collections import Counter

        types = [fs.disadvantage_type for fs in frame_states if fs.disadvantage_type != DisadvantageType.UNKNOWN]
        if not types:
            return DisadvantageType.UNKNOWN

        # Return most common type
        counter = Counter(types)
        return counter.most_common(1)[0][0]

    def _build_segments(self, frame_states: list[FrameState]) -> list[StateSegment]:
        """Build continuous segments from frame states.

        Args:
            frame_states: List of FrameState with states assigned

        Returns:
            List of StateSegment
        """
        if not frame_states:
            return []

        segments: list[StateSegment] = []
        current_state = frame_states[0].state
        segment_start = frame_states[0]
        segment_start_idx = 0
        prev_fs = frame_states[0]

        for i, fs in enumerate(frame_states[1:], start=1):
            if fs.state != current_state:
                # State changed - close current segment
                # Collect stock losses from frames in this segment
                segment_frames = frame_states[segment_start_idx:i]
                stock_losses = self._collect_stock_losses(segment_frames)
                disadvantage_type = self._get_dominant_disadvantage_type(segment_frames)
                # End time is the PREVIOUS frame's timestamp (last frame in this segment)
                segments.append(
                    StateSegment(
                        state=current_state,
                        start_time=segment_start.timestamp,
                        end_time=prev_fs.timestamp,
                        start_frame=segment_start.frame_index,
                        end_frame=prev_fs.frame_index,
                        stock_losses=stock_losses,
                        disadvantage_type=disadvantage_type,
                    )
                )
                current_state = fs.state
                segment_start = fs
                segment_start_idx = i
            prev_fs = fs

        # Close final segment
        final_fs = frame_states[-1]
        segment_frames = frame_states[segment_start_idx:]
        stock_losses = self._collect_stock_losses(segment_frames)
        disadvantage_type = self._get_dominant_disadvantage_type(segment_frames)
        segments.append(
            StateSegment(
                state=current_state,
                start_time=segment_start.timestamp,
                end_time=final_fs.timestamp,
                start_frame=segment_start.frame_index,
                end_frame=final_fs.frame_index,
                stock_losses=stock_losses,
                disadvantage_type=disadvantage_type,
            )
        )

        # Filter out very short segments
        # - Short neutral segments (< 1s) between hits are likely noise
        # - Short advantage/disadvantage segments (< 1.5s) are often OCR jitter
        # EXCEPTION: Segments with stock losses are NEVER filtered (they represent important events)
        MIN_NEUTRAL_DURATION = 1.0
        MIN_ADV_DISADV_DURATION = 1.5
        filtered_segments: list[StateSegment] = []

        for seg in segments:
            # Never filter segments that contain stock losses - they're important events
            if seg.stock_losses:
                filtered_segments.append(seg)
                continue

            if seg.state == GameState.NEUTRAL and seg.duration < MIN_NEUTRAL_DURATION:
                # Skip very short neutral segments, but carry stock losses forward
                if seg.stock_losses and filtered_segments:
                    filtered_segments[-1].stock_losses.extend(seg.stock_losses)
                continue
            if seg.state in (GameState.ADVANTAGE, GameState.DISADVANTAGE) and seg.duration < MIN_ADV_DISADV_DURATION:
                # Skip very short advantage/disadvantage segments, but carry stock losses forward
                if seg.stock_losses and filtered_segments:
                    filtered_segments[-1].stock_losses.extend(seg.stock_losses)
                continue
            filtered_segments.append(seg)

        # Merge consecutive segments of the same state
        merged_segments: list[StateSegment] = []
        for seg in filtered_segments:
            if merged_segments and merged_segments[-1].state == seg.state:
                # Extend previous segment and merge stock losses
                prev = merged_segments[-1]
                merged_stock_losses = prev.stock_losses + seg.stock_losses
                merged_segments[-1] = StateSegment(
                    state=prev.state,
                    start_time=prev.start_time,
                    end_time=seg.end_time,
                    start_frame=prev.start_frame,
                    end_frame=seg.end_frame,
                    stock_losses=merged_stock_losses,
                )
            else:
                merged_segments.append(seg)

        return merged_segments

    def _filter_segments_by_game_bounds(
        self, segments: list[StateSegment]
    ) -> list[StateSegment]:
        """Filter and clip segments to only exist within game boundaries.

        Segments outside of any game (before first GO! or after GAME!) are removed.
        Segments that span game boundaries are clipped to fit within the game.

        Args:
            segments: List of StateSegment to filter

        Returns:
            Filtered list of StateSegment
        """
        if not self._game_start_times or not self._game_end_times:
            return segments

        # Build list of (start, end) game intervals
        # Pair up start and end times
        game_intervals = list(zip(self._game_start_times, self._game_end_times))
        if not game_intervals:
            return segments

        logger.debug(f"Filtering segments by {len(game_intervals)} game bounds")
        for i, (start, end) in enumerate(game_intervals):
            logger.debug(f"  Game {i+1}: {start:.1f}s - {end:.1f}s")

        filtered_segments: list[StateSegment] = []

        for seg in segments:
            # Check if this segment overlaps with any game interval
            for game_start, game_end in game_intervals:
                # No overlap - segment is entirely before or after this game
                if seg.end_time < game_start or seg.start_time > game_end:
                    continue

                # Segment overlaps with this game - clip to game bounds
                clipped_start = max(seg.start_time, game_start)
                clipped_end = min(seg.end_time, game_end)

                # Filter stock losses to only those within the clipped bounds
                clipped_stock_losses = [
                    sl for sl in seg.stock_losses
                    if clipped_start <= sl.timestamp <= clipped_end
                ]

                # Skip if the clipped segment is too short AND has no stock losses
                # Segments with stock losses are always kept (they're important events)
                if clipped_end - clipped_start < 0.5 and not clipped_stock_losses:
                    continue

                clipped_seg = StateSegment(
                    state=seg.state,
                    start_time=clipped_start,
                    end_time=clipped_end,
                    start_frame=seg.start_frame,  # Approximate, not recalculated
                    end_frame=seg.end_frame,
                    stock_losses=clipped_stock_losses,
                )
                filtered_segments.append(clipped_seg)

        # Sort by start time and merge consecutive same-state segments
        filtered_segments.sort(key=lambda s: s.start_time)

        # Merge consecutive segments of the same state (may have been split by game bounds)
        merged: list[StateSegment] = []
        for seg in filtered_segments:
            if merged and merged[-1].state == seg.state:
                # Check if segments are adjacent (within 1 second)
                if seg.start_time - merged[-1].end_time < 1.0:
                    prev = merged[-1]
                    merged[-1] = StateSegment(
                        state=prev.state,
                        start_time=prev.start_time,
                        end_time=seg.end_time,
                        start_frame=prev.start_frame,
                        end_frame=seg.end_frame,
                        stock_losses=prev.stock_losses + seg.stock_losses,
                    )
                    continue
            merged.append(seg)

        logger.debug(f"Filtered {len(segments)} segments to {len(merged)} within game bounds")
        return merged

    def _collect_stock_losses(self, frame_states: list[FrameState]) -> list[StockLoss]:
        """Collect stock losses from a list of frame states.

        Args:
            frame_states: List of FrameState to check

        Returns:
            List of StockLoss events
        """
        stock_losses: list[StockLoss] = []
        for fs in frame_states:
            if fs.p1_stock_loss:
                stock_losses.append(StockLoss(timestamp=fs.timestamp, player=1))
            if fs.p2_stock_loss:
                stock_losses.append(StockLoss(timestamp=fs.timestamp, player=2))
        return stock_losses


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string like "1:23"
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def format_analysis_output(analysis: GameStateAnalysis, game_start_time: float = 0.0) -> str:
    """Format analysis results for display.

    Args:
        analysis: GameStateAnalysis to format
        game_start_time: Offset to add to timestamps (for video time)

    Returns:
        Formatted string output
    """
    lines: list[str] = []

    # Summary stats
    total_time = analysis.total_analyzed_time
    if total_time > 0:
        neutral_pct = (analysis.total_neutral_time / total_time) * 100
        adv_pct = (analysis.total_advantage_time / total_time) * 100
        disadv_pct = (analysis.total_disadvantage_time / total_time) * 100

        lines.append("STATE SUMMARY:")
        lines.append(f"  Neutral:      {format_timestamp(analysis.total_neutral_time)} ({neutral_pct:.0f}%)")
        lines.append(f"  Advantage:    {format_timestamp(analysis.total_advantage_time)} ({adv_pct:.0f}%)")
        lines.append(f"  Disadvantage: {format_timestamp(analysis.total_disadvantage_time)} ({disadv_pct:.0f}%)")
        lines.append("")

    # Neutral segments
    if analysis.neutral_segments:
        lines.append("NEUTRAL SEGMENTS:")
        for seg in analysis.neutral_segments:
            start = format_timestamp(seg.start_time + game_start_time)
            end = format_timestamp(seg.end_time + game_start_time)
            lines.append(f"  {start} - {end}")
        lines.append("")

    # Advantage segments
    if analysis.advantage_segments:
        lines.append("ADVANTAGE SEGMENTS:")
        for seg in analysis.advantage_segments:
            start = format_timestamp(seg.start_time + game_start_time)
            end = format_timestamp(seg.end_time + game_start_time)
            lines.append(f"  {start} - {end}")
        lines.append("")

    # Disadvantage segments
    if analysis.disadvantage_segments:
        lines.append("DISADVANTAGE SEGMENTS:")
        for seg in analysis.disadvantage_segments:
            start = format_timestamp(seg.start_time + game_start_time)
            end = format_timestamp(seg.end_time + game_start_time)
            lines.append(f"  {start} - {end}")

    return "\n".join(lines)
