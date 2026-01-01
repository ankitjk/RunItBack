"""UI element detection for SSBU frames.

Detects damage percentages, stock counts, and game boundaries from the game's HUD.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from runitback.cv.ocr import read_damage_percentage

logger = logging.getLogger(__name__)


class FrameType(Enum):
    """Type of frame detected."""
    GAMEPLAY = "gameplay"
    COUNTDOWN = "countdown"  # 3, 2, 1, GO!
    GAME_END = "game_end"  # GAME! screen
    MENU = "menu"  # Character select, stage select, etc.
    UNKNOWN = "unknown"


@dataclass
class PlayerUIState:
    """UI state for a single player."""

    damage: Optional[float]
    stocks: int
    damage_confidence: float

    @property
    def is_valid(self) -> bool:
        """Check if the UI state has valid readings."""
        return self.damage is not None and self.stocks >= 0


@dataclass
class UIState:
    """Complete UI state for both players."""

    player1: PlayerUIState
    player2: PlayerUIState
    timestamp: float

    def is_game_start(self, starting_stocks: int = 3) -> bool:
        """Check if this looks like the start of a game.

        At game start:
        - Both players have full stocks
        - Both players have 0% damage
        """
        return (
            self.player1.stocks == starting_stocks
            and self.player2.stocks == starting_stocks
            and self.player1.damage is not None
            and self.player2.damage is not None
            and self.player1.damage < 5
            and self.player2.damage < 5
        )


# SSBU UI Layout (normalized coordinates 0-1)
# These are approximate and may need tuning based on video resolution

# Damage percentage regions (bottom of screen)
# Format: (x_start, y_start, x_end, y_end) as fractions of frame dimensions
# The damage numbers appear as large text near center-bottom of screen
# P1 damage ~27-44% from left, P2 damage ~64-76% from left
P1_DAMAGE_ROI = (0.27, 0.87, 0.44, 0.98)  # Player 1 (left-center)
P2_DAMAGE_ROI = (0.64, 0.87, 0.76, 0.98)  # Player 2 (right-center)

# Stock indicator regions (below player names, at very bottom of screen)
# Stock icons appear as small character head icons below the damage percentage
P1_STOCKS_ROI = (0.26, 0.95, 0.36, 0.99)  # Player 1 stocks (bottom-left)
P2_STOCKS_ROI = (0.64, 0.95, 0.74, 0.99)  # Player 2 stocks (bottom-right)

# Character icon regions (small portrait near stocks)
# These appear as colorful, relatively detailed icons
P1_CHAR_ICON_ROI = (0.08, 0.82, 0.14, 0.92)
P2_CHAR_ICON_ROI = (0.86, 0.82, 0.92, 0.92)


def get_roi(frame: np.ndarray, roi_coords: tuple) -> np.ndarray:
    """Extract a region of interest from a frame.

    Args:
        frame: Full video frame (BGR)
        roi_coords: Tuple of (x_start, y_start, x_end, y_end) as fractions

    Returns:
        Cropped region as numpy array
    """
    h, w = frame.shape[:2]
    x1 = int(roi_coords[0] * w)
    y1 = int(roi_coords[1] * h)
    x2 = int(roi_coords[2] * w)
    y2 = int(roi_coords[3] * h)

    return frame[y1:y2, x1:x2].copy()


def detect_character_icon(frame: np.ndarray, player: int) -> bool:
    """Detect if a character icon is present for a player.

    Character icons are small, colorful portraits that appear near the
    stock indicators during gameplay. They have:
    - High color saturation (character artwork is colorful)
    - Moderate complexity (face/body details create edges)
    - Distinct from empty/dark backgrounds

    Args:
        frame: Full video frame (BGR)
        player: Player number (1 or 2)

    Returns:
        True if a character icon is detected
    """
    roi_coords = P1_CHAR_ICON_ROI if player == 1 else P2_CHAR_ICON_ROI
    roi = get_roi(frame, roi_coords)

    if roi.size == 0:
        return False

    # Check color saturation - character icons are colorful
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat_mean = np.mean(hsv[:, :, 1])
    val_mean = np.mean(hsv[:, :, 2])

    # Check for edge complexity - icons have some detail but not too busy
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size

    # Character icons typically have:
    # - Moderate to high saturation (> 40) - they're colorful
    # - Reasonable brightness (> 60) - not a dark/empty region
    # - Some edge complexity (0.05 - 0.35) - portrait details
    has_color = sat_mean > 40 and val_mean > 60
    has_detail = 0.05 < edge_ratio < 0.35

    return has_color and has_detail


def detect_stock_icon(frame: np.ndarray, player: int) -> bool:
    """Detect if at least one stock icon is present for a player.

    Stock icons are small, bright indicators above the damage percentage.
    At least one stock must be visible for the game to be active.

    Args:
        frame: Full video frame (BGR)
        player: Player number (1 or 2)

    Returns:
        True if at least one stock icon is detected
    """
    roi_coords = P1_STOCKS_ROI if player == 1 else P2_STOCKS_ROI
    roi = get_roi(frame, roi_coords)

    if roi.size == 0:
        return False

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Stock icons are bright - check for any reasonably bright section
    # The threshold is lower than count_stocks because we just need presence
    brightness_threshold = 80
    bright_pixels = np.sum(gray > brightness_threshold) / gray.size

    # If more than 5% of the stock region has bright pixels, stocks are present
    return bright_pixels > 0.05


def detect_damage_display(frame: np.ndarray, player: int) -> bool:
    """Detect if a damage percentage display is present for a player.

    The damage display shows white/colored numbers on a dark background.
    We check for the presence of bright text-like elements.

    Args:
        frame: Full video frame (BGR)
        player: Player number (1 or 2)

    Returns:
        True if damage display is detected
    """
    roi_coords = P1_DAMAGE_ROI if player == 1 else P2_DAMAGE_ROI
    roi = get_roi(frame, roi_coords)

    if roi.size == 0:
        return False

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Damage numbers are bright white text
    # Check for presence of bright pixels (the numbers)
    bright_pixels = np.sum(gray > 180) / gray.size

    # Check for some contrast (text against background)
    std_dev = np.std(gray)

    # Damage display has bright numbers (> 3% bright pixels) and contrast (> 30 std)
    has_bright_text = bright_pixels > 0.03
    has_contrast = std_dev > 30

    return has_bright_text and has_contrast


def count_stocks(
    frame: np.ndarray,
    player: int,
    max_stocks: int = 3,
) -> int:
    """Count the number of stocks for a player.

    Stocks are indicated by filled icons vs empty/missing icons.
    This uses brightness detection since filled stocks are typically
    brighter than empty/lost stocks.

    Args:
        frame: Full video frame (BGR)
        player: Player number (1 or 2)
        max_stocks: Maximum possible stocks

    Returns:
        Number of stocks remaining
    """
    roi_coords = P1_STOCKS_ROI if player == 1 else P2_STOCKS_ROI
    roi = get_roi(frame, roi_coords)

    if roi.size == 0:
        return max_stocks  # Default to full stocks on error

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Divide into sections (one per potential stock)
    h, w = gray.shape
    section_width = w // max_stocks

    stocks = 0
    brightness_threshold = 100  # Adjust based on testing

    for i in range(max_stocks):
        x1 = i * section_width
        x2 = (i + 1) * section_width
        section = gray[:, x1:x2]

        # Check if section is bright enough to indicate a stock
        mean_brightness = np.mean(section)
        if mean_brightness > brightness_threshold:
            stocks += 1

    return stocks


def detect_ui_state(
    frame: np.ndarray,
    timestamp: float,
    confidence_threshold: float = 0.5,
) -> UIState:
    """Detect the complete UI state from a frame.

    Args:
        frame: Full video frame (BGR)
        timestamp: Timestamp of the frame in seconds
        confidence_threshold: Minimum OCR confidence

    Returns:
        UIState containing both players' damage and stocks
    """
    # Extract damage ROIs
    p1_damage_roi = get_roi(frame, P1_DAMAGE_ROI)
    p2_damage_roi = get_roi(frame, P2_DAMAGE_ROI)

    # Read damage percentages
    p1_damage, p1_conf = read_damage_percentage(p1_damage_roi, confidence_threshold)
    p2_damage, p2_conf = read_damage_percentage(p2_damage_roi, confidence_threshold)

    # Count stocks
    p1_stocks = count_stocks(frame, player=1)
    p2_stocks = count_stocks(frame, player=2)

    return UIState(
        player1=PlayerUIState(
            damage=p1_damage,
            stocks=p1_stocks,
            damage_confidence=p1_conf,
        ),
        player2=PlayerUIState(
            damage=p2_damage,
            stocks=p2_stocks,
            damage_confidence=p2_conf,
        ),
        timestamp=timestamp,
    )


def detect_game_boundary(
    current: UIState,
    previous: Optional[UIState],
    starting_stocks: int = 3,
) -> bool:
    """Detect if a new game has started between two UI states.

    A new game is detected when:
    - Stocks reset to starting value for both players
    - Damage resets to 0% for both players (if damage is detected)

    Args:
        current: Current UI state
        previous: Previous UI state (can be None)
        starting_stocks: Number of stocks at game start

    Returns:
        True if a new game has started
    """
    if previous is None:
        # First frame - check if it looks like a game start
        return current.is_game_start(starting_stocks)

    # Check for stock reset (both players back to full)
    stocks_reset = (
        current.player1.stocks == starting_stocks
        and current.player2.stocks == starting_stocks
        and (
            previous.player1.stocks < starting_stocks
            or previous.player2.stocks < starting_stocks
        )
    )

    # If stocks reset, that's a strong signal of game boundary
    if stocks_reset:
        return True

    # Check for damage reset (both players at 0%) when damage is available
    damage_reset = False
    if (
        current.player1.damage is not None
        and current.player2.damage is not None
        and current.player1.damage < 5
        and current.player2.damage < 5
    ):
        # Previous damage was high, now it's low = game boundary
        if (
            (previous.player1.damage is not None and previous.player1.damage > 50)
            or (previous.player2.damage is not None and previous.player2.damage > 50)
        ):
            damage_reset = True

    return damage_reset


def detect_countdown(frame: np.ndarray) -> bool:
    """Detect if frame shows the game start countdown (3, 2, 1, GO!).

    The countdown appears as large orange/yellow text in the center of the screen.
    The countdown screen is distinctive because:
    - Large orange text in center
    - Relatively clean/simple background (low edge density)
    - The text itself doesn't create many edges (it's large and solid)

    Args:
        frame: Full video frame (BGR)

    Returns:
        True if countdown is detected
    """
    h, w = frame.shape[:2]

    # Focus on center region where countdown appears
    cy1, cy2 = int(0.3 * h), int(0.7 * h)
    cx1, cx2 = int(0.35 * w), int(0.65 * w)
    center_roi = frame[cy1:cy2, cx1:cx2]

    # Convert to HSV to detect orange/yellow color of countdown
    hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)

    # Orange/yellow range for countdown text (hue 10-30, high saturation, high value)
    lower_orange = np.array([10, 150, 150])
    upper_orange = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    orange_ratio = np.sum(mask > 0) / mask.size

    # Check edge density - countdown has low edge complexity
    # (large simple numbers vs busy menu screens)
    gray_center = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_center, 100, 200)
    edge_ratio = np.sum(edges > 0) / edges.size

    # Countdown requires:
    # 1. Significant orange (> 5%)
    # 2. Low edge density (< 10%) - menus have 15-20% edges
    if orange_ratio > 0.05 and edge_ratio < 0.10:
        return True

    # Special case: very high orange with any edge ratio (GO! moment)
    if orange_ratio > 0.15:
        return True

    return False


def detect_game_end_screen(frame: np.ndarray) -> bool:
    """Detect if frame shows the GAME! end screen.

    The end screen shows green "GAME!" text with:
    - A very bright/white flash on the screen
    - Large green "GAME!" text in the center
    - The winner's character displayed

    We detect this by looking for:
    1. Extremely bright screen (white flash)
    2. The flash specifically affects the whole screen uniformly
    3. High percentage of white pixels

    Args:
        frame: Full video frame (BGR)

    Returns:
        True if game end screen is detected
    """
    h, w = frame.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # The GAME! white flash is extremely bright
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)

    # Check for very high percentage of white pixels (threshold 200 for wider detection)
    white_pixels_200 = np.sum(gray > 200)
    white_ratio_200 = white_pixels_200 / gray.size

    white_pixels_240 = np.sum(gray > 240)
    white_ratio_240 = white_pixels_240 / gray.size

    # Multiple conditions for GAME! flash:
    # 1. Very bright and uniform (classic flash)
    if mean_brightness > 200 and std_brightness < 50:
        return True

    # 2. Extremely high white percentage
    if white_ratio_240 > 0.7:
        return True

    # 3. High brightness with significant white (catches some edge cases)
    if mean_brightness > 180 and std_brightness < 40 and white_ratio_200 > 0.5:
        return True

    # 4. Very high white at 200 threshold (the flash might not be pure white)
    if white_ratio_200 > 0.8:
        return True

    return False


# Template paths for GO! and GAME! detection
from pathlib import Path

_GO_TEMPLATE: np.ndarray | None = None
_GAME_TEMPLATE: np.ndarray | None = None


def _load_game_marker_templates() -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load GO! and GAME! templates from disk.

    Returns:
        Tuple of (go_template, game_template), either can be None if not found
    """
    global _GO_TEMPLATE, _GAME_TEMPLATE

    if _GO_TEMPLATE is not None or _GAME_TEMPLATE is not None:
        return _GO_TEMPLATE, _GAME_TEMPLATE

    # Find template directory
    template_dir = Path(__file__).parent.parent.parent / "data" / "game_markers"

    go_path = template_dir / "go.jpg"
    game_path = template_dir / "game.jpg"

    if go_path.exists():
        _GO_TEMPLATE = cv2.imread(str(go_path))
        if _GO_TEMPLATE is not None:
            logger.debug(f"Loaded GO! template: {_GO_TEMPLATE.shape}")
    else:
        logger.warning(f"GO! template not found at {go_path}")

    if game_path.exists():
        _GAME_TEMPLATE = cv2.imread(str(game_path))
        if _GAME_TEMPLATE is not None:
            logger.debug(f"Loaded GAME! template: {_GAME_TEMPLATE.shape}")
    else:
        logger.warning(f"GAME! template not found at {game_path}")

    return _GO_TEMPLATE, _GAME_TEMPLATE


def detect_go_screen(frame: np.ndarray, threshold: float = 0.28) -> bool:
    """Detect if frame shows the "GO!" screen using color analysis.

    The GO! text appears in the center of the screen with distinctive
    orange/yellow fiery gradient coloring. This creates a unique color
    signature that's much more reliable than template matching.

    Real GO! frames have combined orange+yellow ratio of 0.31-0.39.
    False positives (menus, character select) have ratio of 0.17-0.24.
    Threshold of 0.28 cleanly separates these.

    Additional filtering is applied to exclude:
    - Final Smash explosions (which have similar orange/yellow colors)
    - Other in-game effects

    Args:
        frame: Full video frame (BGR)
        threshold: Minimum orange+yellow color ratio in center region

    Returns:
        True if GO! is detected
    """
    h, w = frame.shape[:2]

    # Focus on center-upper region where GO! appears
    # GO! is in upper-center, not bottom (where HUD is)
    cy1, cy2 = int(0.15 * h), int(0.6 * h)
    cx1, cx2 = int(0.2 * w), int(0.8 * w)
    center = frame[cy1:cy2, cx1:cx2]

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)

    # Orange range for GO! text (the fiery gradient)
    # Hue: 5-25 (orange-yellow), Sat: 150+, Val: 150+
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    orange_ratio = np.sum(orange_mask > 0) / orange_mask.size

    # Also check for bright yellow (part of the gradient)
    lower_yellow = np.array([20, 100, 200])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size

    combined_ratio = orange_ratio + yellow_ratio

    # GO! screen typically has 31-39% orange+yellow in center region
    # False positives (menus) have 17-24%
    if combined_ratio >= threshold:
        # Additional check: Filter out Final Smash explosions and in-game effects
        # Real GO! screens have:
        # 1. Orange+yellow between 0.28 and 0.50 (explosions often have > 0.45)
        # 2. Visible stage/platform elements (some blue/green from backgrounds)
        # 3. The HUD area at bottom should show 0% damage (game just started)

        # Check if combined ratio is too high (likely an explosion effect)
        if combined_ratio > 0.45:
            # Check for red pixels which are common in Final Smash explosions
            # but not in GO! screens
            lower_red1 = np.array([0, 150, 150])
            upper_red1 = np.array([5, 255, 255])
            lower_red2 = np.array([170, 150, 150])
            upper_red2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_ratio = (np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)) / red_mask1.size

            # Final Smash explosions have significant red (> 10%)
            # GO! screens have some red but less (< 8%)
            if red_ratio > 0.08:
                logger.debug(f"Filtered GO! false positive: high red ratio {red_ratio:.3f} (likely explosion)")
                return False

        logger.debug(f"GO! detected with color ratio {combined_ratio:.3f}")
        return True

    return False


def detect_game_screen(frame: np.ndarray, threshold: float = 0.08) -> bool:
    """Detect if frame shows the "GAME!" screen using color analysis.

    The GAME! text appears in the center of the screen with distinctive
    green/yellow gradient coloring. The detection uses:
    1. Green color ratio in the center region
    2. Additional checks to reduce false positives

    Real GAME! frames have green ratio of 0.08-0.18 in the center region.
    We use a lower threshold (0.08) to catch all GAME! screens, as the
    exact ratio varies based on stage background.

    Args:
        frame: Full video frame (BGR)
        threshold: Minimum green-yellow color ratio in center region

    Returns:
        True if GAME! is detected
    """
    h, w = frame.shape[:2]

    # Focus on tight center region where GAME! text appears
    # GAME! is roughly in the middle-upper portion of the screen
    cy1, cy2 = int(0.2 * h), int(0.6 * h)
    cx1, cx2 = int(0.25 * w), int(0.75 * w)
    center = frame[cy1:cy2, cx1:cx2]

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)

    # Green-yellow range for GAME! text
    # Hue: 35-75 (yellow to green), moderate+ saturation, high value
    lower_green = np.array([35, 80, 150])
    upper_green = np.array([75, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / green_mask.size

    # GAME! screen detection with threshold
    # Lower threshold (0.08) catches most GAME! screens
    # The consolidation logic in cli.py will combine consecutive detections
    if green_ratio >= threshold:
        # Additional check: GAME! screen should have moderate overall brightness
        # (not too dark like normal gameplay, not extremely bright like the flash)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        # GAME! screens typically have brightness 100-180
        # Very dark (<80) or very bright (>200) frames are not GAME! text
        if 80 <= mean_brightness <= 200:
            logger.debug(f"GAME! detected with green ratio {green_ratio:.3f}, brightness {mean_brightness:.1f}")
            return True

    return False


def detect_gameplay_hud(frame: np.ndarray) -> bool:
    """Detect SSBU gameplay HUD presence using definitive UI elements.

    The SSBU HUD during gameplay has:
    - Character icons (small portraits near stocks)
    - Stock icons (bright indicators showing remaining lives)
    - Damage percentages (white numbers showing current damage)

    A frame is definitively gameplay if we can detect:
    - Two character icons (one for each player)
    - At least one stock icon (at least one player has stocks)
    - At least one damage percentage display

    This approach is more reliable than heuristics because these UI elements
    are unique to actual gameplay and don't appear in menus or overlays.

    Args:
        frame: Full video frame (BGR)

    Returns:
        True if gameplay HUD is detected
    """
    h, w = frame.shape[:2]

    # Check center edge density first - character/stage select screens have
    # very high edge density due to dense grid of portraits/stage thumbnails
    # Gameplay typically has 0.05-0.12, menu grids have 0.13-0.25
    center = frame[int(0.2 * h):int(0.75 * h), int(0.15 * w):int(0.85 * w)]
    gray_center = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    edges_center = cv2.Canny(gray_center, 50, 150)
    center_edge_ratio = np.sum(edges_center > 0) / edges_center.size

    # If center has high edge density, likely a menu grid (not gameplay)
    # Lowered threshold from 0.15 to 0.13 to reject more false positives
    if center_edge_ratio > 0.13:
        return False

    # Primary detection: Use definitive UI element presence
    # Check for character icons
    p1_has_char = detect_character_icon(frame, player=1)
    p2_has_char = detect_character_icon(frame, player=2)
    both_chars = p1_has_char and p2_has_char

    # Check for stock icons (at least one player must have stocks)
    p1_has_stocks = detect_stock_icon(frame, player=1)
    p2_has_stocks = detect_stock_icon(frame, player=2)
    has_stocks = p1_has_stocks or p2_has_stocks

    # Check for damage display (at least one must be visible)
    p1_has_damage = detect_damage_display(frame, player=1)
    p2_has_damage = detect_damage_display(frame, player=2)
    has_damage = p1_has_damage or p2_has_damage

    # Definitive gameplay detection:
    # If we have both character icons, at least one stock, and at least one
    # damage display, this is definitely gameplay
    if both_chars and has_stocks and has_damage:
        return True

    # Strong detection: One character icon + stocks and damage on both sides
    # (sometimes one icon is obscured by effects or camera angle)
    one_char = p1_has_char or p2_has_char
    both_stocks = p1_has_stocks and p2_has_stocks
    both_damage = p1_has_damage and p2_has_damage
    if one_char and both_stocks and both_damage:
        return True

    # Additional detection: Both stocks visible + at least one damage
    # This catches cases where character icons are obscured but stocks/damage are clear
    # Stocks for both players + at least one damage is a strong indicator of gameplay
    # (character icons may be obscured by stream overlays or effects)
    if both_stocks and has_damage:
        return True

    # Removed unreliable fallback detection that was causing false positives
    # on loading screens and menu transitions. If definitive UI elements
    # (character icons, stocks, damage displays) aren't detected, the frame
    # is likely not gameplay.

    return False


def detect_frame_type(frame: np.ndarray) -> FrameType:
    """Detect what type of frame this is.

    Args:
        frame: Full video frame (BGR)

    Returns:
        FrameType indicating the frame category
    """
    # Check for game end first (most distinctive - bright white flash)
    if detect_game_end_screen(frame):
        return FrameType.GAME_END

    # Check for gameplay HUD (most common during a match)
    if detect_gameplay_hud(frame):
        return FrameType.GAMEPLAY

    # Check for countdown (orange text in center)
    if detect_countdown(frame):
        return FrameType.COUNTDOWN

    return FrameType.MENU


def detect_stock_counts(frame: np.ndarray) -> tuple[int, int]:
    """Detect current stock counts for both players from the HUD.

    This reads the stock icons displayed below/near each player's damage
    percentage. Stock icons are bright when active and dim/gone when lost.

    Args:
        frame: Full video frame (BGR)

    Returns:
        Tuple of (p1_stocks, p2_stocks)
    """
    p1_stocks = count_stocks(frame, player=1)
    p2_stocks = count_stocks(frame, player=2)
    return p1_stocks, p2_stocks


def save_debug_rois(
    frame: np.ndarray,
    output_dir: str,
    frame_number: int,
) -> None:
    """Save ROI images for debugging purposes.

    Args:
        frame: Full video frame
        output_dir: Directory to save debug images
        frame_number: Frame number for filename
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Save damage ROIs
    p1_damage = get_roi(frame, P1_DAMAGE_ROI)
    p2_damage = get_roi(frame, P2_DAMAGE_ROI)

    cv2.imwrite(f"{output_dir}/frame_{frame_number}_p1_damage.png", p1_damage)
    cv2.imwrite(f"{output_dir}/frame_{frame_number}_p2_damage.png", p2_damage)

    # Save stock ROIs
    p1_stocks = get_roi(frame, P1_STOCKS_ROI)
    p2_stocks = get_roi(frame, P2_STOCKS_ROI)

    cv2.imwrite(f"{output_dir}/frame_{frame_number}_p1_stocks.png", p1_stocks)
    cv2.imwrite(f"{output_dir}/frame_{frame_number}_p2_stocks.png", p2_stocks)
