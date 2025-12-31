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
2. OPTIONAL: Gemini Vision API confirmation - uses stock counter splash screen
   detection for higher accuracy (especially for low-percent kills)
"""

import logging
import os
from typing import Optional

import cv2
import numpy as np

from runitback.classification.states import (
    FrameState,
    GameState,
    GameStateAnalysis,
    StateSegment,
    StockLoss,
)
from runitback.cv.ocr import read_damage_percentage
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
        gemini_api_key: Optional[str] = None,
    ):
        """Initialize the detector.

        Args:
            player_perspective: Which player to analyze (1 or 2)
            hit_threshold: Minimum damage increase to count as a hit
            neutral_reset_time: Seconds without damage to return to neutral
            use_gemini_stock_detection: Enable Gemini Vision API for stock loss
                confirmation. More accurate but adds ~0.5-1.5s per potential kill
                and ~$0.0001 per API call. Recommended for accuracy.
            gemini_api_key: Gemini API key. If not provided, uses GEMINI_API_KEY
                environment variable.
        """
        self.player_perspective = player_perspective
        self.hit_threshold = hit_threshold
        self.neutral_reset_time = neutral_reset_time
        self.use_gemini_stock_detection = use_gemini_stock_detection
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")

        # Import stock counter detector if Gemini is enabled
        self._stock_counter_tracker = None
        if self.use_gemini_stock_detection:
            try:
                from runitback.cv.stock_counter_detector import StockCounterTracker
                self._stock_counter_tracker = StockCounterTracker(use_gemini=True)
                logger.info("Gemini stock counter detection enabled")
            except ImportError as e:
                logger.warning(f"Could not import StockCounterTracker: {e}")
                self.use_gemini_stock_detection = False

    def detect_states(
        self,
        frames: list,
        frame_interval: float = 0.5,
    ) -> GameStateAnalysis:
        """Detect game states from a list of frames.

        Args:
            frames: List of Frame objects (with .image and .timestamp)
            frame_interval: Time between frames in seconds

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

        # Reset stock counter tracker for new analysis
        if self._stock_counter_tracker:
            self._stock_counter_tracker.reset()

        # Extract damage values from each frame
        frame_states = self._extract_frame_states(frames)

        # Detect state transitions (pass frames for Gemini detection)
        self._detect_state_transitions(frame_states, frames)

        # Merge consecutive frames into segments
        segments = self._build_segments(frame_states)

        return GameStateAnalysis(
            player_perspective=self.player_perspective,
            neutral_segments=[s for s in segments if s.state == GameState.NEUTRAL],
            advantage_segments=[s for s in segments if s.state == GameState.ADVANTAGE],
            disadvantage_segments=[s for s in segments if s.state == GameState.DISADVANTAGE],
            frame_states=frame_states,
        )

    def _extract_frame_states(self, frames: list) -> list[FrameState]:
        """Extract damage percentages and stock information from frames.

        Args:
            frames: List of Frame objects

        Returns:
            List of FrameState with damage values and stock counts
        """
        frame_states: list[FrameState] = []
        prev_p1_damage: Optional[float] = None
        prev_p2_damage: Optional[float] = None

        for i, frame in enumerate(frames):
            # Extract damage ROIs
            p1_roi = get_roi(frame.image, P1_DAMAGE_ROI)
            p2_roi = get_roi(frame.image, P2_DAMAGE_ROI)

            # Read damage percentages
            p1_damage, p1_conf = read_damage_percentage(p1_roi)
            p2_damage, p2_conf = read_damage_percentage(p2_roi)

            # Detect stock counts from HUD stock icons
            p1_stocks, p2_stocks = detect_stock_counts(frame.image)

            # Log damage readings periodically for debugging
            if i % 20 == 0:  # Every 10 seconds at 0.5s intervals
                logger.debug(
                    f"Frame {i} ({frame.timestamp:.1f}s): "
                    f"P1={p1_damage}% (conf={p1_conf:.2f}), "
                    f"P2={p2_damage}% (conf={p2_conf:.2f})"
                )

            # Log stock counts periodically for debugging
            if i % 20 == 0:
                logger.debug(
                    f"Frame {i} ({frame.timestamp:.1f}s): "
                    f"Stock counts: P1={p1_stocks}, P2={p2_stocks}"
                )

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

        # Track maximum damage seen to handle OCR noise
        # This prevents losing track of high damage when OCR has intermediate errors
        p1_recent_max_damage: float = 0.0
        p2_recent_max_damage: float = 0.0

        # Build frame lookup for Gemini detection
        frame_by_index = {}
        if frames and self.use_gemini_stock_detection:
            frame_by_index = {i: f for i, f in enumerate(frames)}

        for fs in frame_states:
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

            # P1 stock loss: confirm only after consecutive low damage frames
            # and only if damage was high enough before the low streak
            # MIN_KILL_DAMAGE filters OCR noise (e.g., 62% misread as 3%)
            if (p1_low_damage_frames == CONFIRM_FRAMES and
                p1_high_damage_before_low is not None and
                p1_high_damage_before_low >= MIN_KILL_DAMAGE and cooldown_p1_ok):
                p1_stock_loss = True
                last_p1_stock_loss_time = fs.timestamp
                # Reset recent max after stock loss (player starts fresh)
                p1_recent_max_damage = 0.0
                logger.debug(f"P1 stock loss confirmed (damage reset): {p1_high_damage_before_low:.0f}% -> low at {fs.timestamp:.1f}s")

            # P2 stock loss: confirm only after consecutive low damage frames
            # and only if damage was high enough before the low streak
            if (p2_low_damage_frames == CONFIRM_FRAMES and
                p2_high_damage_before_low is not None and
                p2_high_damage_before_low >= MIN_KILL_DAMAGE and cooldown_p2_ok):
                p2_stock_loss = True
                last_p2_stock_loss_time = fs.timestamp
                # Reset recent max after stock loss (player starts fresh)
                p2_recent_max_damage = 0.0
                logger.debug(f"P2 stock loss confirmed (damage reset): {p2_high_damage_before_low:.0f}% -> low at {fs.timestamp:.1f}s")

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

            # Stock loss resets game state to neutral
            if p1_stock_loss or p2_stock_loss:
                current_state = GameState.NEUTRAL
                last_hit_time = None
                fs.state = current_state
                # Update previous damage values
                if fs.p1_damage is not None:
                    prev_p1_damage = fs.p1_damage
                if fs.p2_damage is not None:
                    prev_p2_damage = fs.p2_damage
                continue

            # Determine who got hit
            p1_got_hit = p1_delta >= self.hit_threshold
            p2_got_hit = p2_delta >= self.hit_threshold

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
                    time_since_hit = fs.timestamp - last_hit_time
                    # Return to neutral if:
                    # 1. Enough time has passed
                    # 2. Both players are visible (on stage)
                    if (
                        time_since_hit >= self.neutral_reset_time
                        and fs.p1_visible
                        and fs.p2_visible
                    ):
                        current_state = GameState.NEUTRAL
                        last_hit_time = None

            fs.state = current_state

            # Update previous damage values
            if fs.p1_damage is not None:
                prev_p1_damage = fs.p1_damage
            if fs.p2_damage is not None:
                prev_p2_damage = fs.p2_damage

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
                stock_losses = self._collect_stock_losses(
                    frame_states[segment_start_idx:i]
                )
                # End time is the PREVIOUS frame's timestamp (last frame in this segment)
                segments.append(
                    StateSegment(
                        state=current_state,
                        start_time=segment_start.timestamp,
                        end_time=prev_fs.timestamp,
                        start_frame=segment_start.frame_index,
                        end_frame=prev_fs.frame_index,
                        stock_losses=stock_losses,
                    )
                )
                current_state = fs.state
                segment_start = fs
                segment_start_idx = i
            prev_fs = fs

        # Close final segment
        final_fs = frame_states[-1]
        stock_losses = self._collect_stock_losses(
            frame_states[segment_start_idx:]
        )
        segments.append(
            StateSegment(
                state=current_state,
                start_time=segment_start.timestamp,
                end_time=final_fs.timestamp,
                start_frame=segment_start.frame_index,
                end_frame=final_fs.frame_index,
                stock_losses=stock_losses,
            )
        )

        # Filter out very short segments
        # - Short neutral segments (< 1s) between hits are likely noise
        # - Short advantage/disadvantage segments (< 1.5s) are often OCR jitter
        MIN_NEUTRAL_DURATION = 1.0
        MIN_ADV_DISADV_DURATION = 1.5
        filtered_segments: list[StateSegment] = []

        for seg in segments:
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
