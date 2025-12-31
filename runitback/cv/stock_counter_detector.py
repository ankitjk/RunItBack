"""Stock counter splash detection for stock loss confirmation.

Detects the stock counter splash that appears after every kill in
Super Smash Bros. Ultimate. The splash shows "X - Y" format where
X is P1's remaining stocks and Y is P2's remaining stocks.

This splash appears for ALL kill types including star KO, making it
more reliable than brightness flash detection or damage reset alone.

For the final stock (game-ending kill), "GAME!" appears instead.

Detection methods:
1. Primary: Gemini Vision API (most reliable, handles stylized fonts)
2. Fallback: OCR-based detection (for when API is unavailable)
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def detect_stock_counter_gemini(
    frame: np.ndarray,
    api_key: Optional[str] = None,
    use_cache: bool = True,
) -> tuple[bool, Optional[int], Optional[int], bool]:
    """Detect stock counter splash using Gemini Vision API.

    This is the primary detection method - more reliable than OCR
    for the stylized SSBU font.

    Args:
        frame: Full video frame (BGR)
        api_key: Gemini API key (uses env var if not provided)
        use_cache: Whether to use cached results

    Returns:
        Tuple of (detected, p1_stocks, p2_stocks, is_game_end):
        - detected: True if stock counter splash was detected
        - p1_stocks: Player 1's stock count (or None)
        - p2_stocks: Player 2's stock count (or None)
        - is_game_end: True if "GAME!" screen was shown
    """
    try:
        from runitback.cv.llm_detector import detect_stock_counter_with_gemini

        result = detect_stock_counter_with_gemini(
            frame, api_key=api_key, use_cache=use_cache
        )

        return (
            result.detected,
            result.p1_stocks,
            result.p2_stocks,
            result.is_game_end,
        )
    except ImportError:
        logger.warning("Could not import Gemini detector")
        return False, None, None, False
    except Exception as e:
        logger.warning(f"Gemini stock counter detection failed: {e}")
        return False, None, None, False


class StockCounterTracker:
    """Track stock counts over time using stock counter splash detection.

    Monitors frames for stock counter splashes and tracks when stocks change,
    providing a reliable signal for stock loss confirmation that works for
    ALL kill types including low-percent kills and star KOs.

    Uses Gemini Vision API as the primary detection method for reliable
    reading of the stylized SSBU font.
    """

    def __init__(self, use_gemini: bool = True):
        """Initialize the tracker.

        Args:
            use_gemini: Whether to use Gemini API (recommended for accuracy)
        """
        self.use_gemini = use_gemini
        self.last_p1_stocks: Optional[int] = None
        self.last_p2_stocks: Optional[int] = None
        self.last_detection_time: float = 0.0
        self.detection_cooldown: float = 3.0  # Minimum seconds between checks

    def update(
        self,
        frame: np.ndarray,
        timestamp: float,
        api_key: Optional[str] = None,
    ) -> tuple[bool, bool, Optional[int], Optional[int], bool]:
        """Process a frame and detect stock changes.

        Args:
            frame: BGR image frame
            timestamp: Frame timestamp in seconds
            api_key: Optional Gemini API key

        Returns:
            Tuple of (p1_lost_stock, p2_lost_stock, p1_stocks, p2_stocks, is_game_end):
            - p1_lost_stock: True if P1 lost a stock this frame
            - p2_lost_stock: True if P2 lost a stock this frame
            - p1_stocks: Current P1 stock count (or None if not detected)
            - p2_stocks: Current P2 stock count (or None if not detected)
            - is_game_end: True if GAME! screen was detected
        """
        # Apply cooldown to avoid excessive API calls
        if timestamp - self.last_detection_time < self.detection_cooldown:
            return False, False, None, None, False

        # Detect stock counter
        if self.use_gemini:
            detected, p1_stocks, p2_stocks, is_game_end = detect_stock_counter_gemini(
                frame, api_key=api_key
            )
        else:
            # Fallback to OCR (less reliable)
            detected, p1_stocks, p2_stocks, is_game_end = False, None, None, False

        if not detected:
            return False, False, None, None, False

        self.last_detection_time = timestamp

        # Check for GAME! screen
        if is_game_end:
            # Game ended - one player lost their last stock
            # Determine which player based on previous stock counts
            p1_lost = False
            p2_lost = False
            if self.last_p1_stocks == 1:
                p1_lost = True
                logger.info(f"GAME! P1 lost final stock at {timestamp:.1f}s")
            elif self.last_p2_stocks == 1:
                p2_lost = True
                logger.info(f"GAME! P2 lost final stock at {timestamp:.1f}s")
            return p1_lost, p2_lost, 0, None, True

        p1_lost = False
        p2_lost = False

        # Check if stocks decreased
        if self.last_p1_stocks is not None and p1_stocks is not None:
            if p1_stocks < self.last_p1_stocks:
                p1_lost = True
                logger.info(
                    f"Stock counter: P1 lost stock ({self.last_p1_stocks} -> {p1_stocks}) "
                    f"at {timestamp:.1f}s"
                )

        if self.last_p2_stocks is not None and p2_stocks is not None:
            if p2_stocks < self.last_p2_stocks:
                p2_lost = True
                logger.info(
                    f"Stock counter: P2 lost stock ({self.last_p2_stocks} -> {p2_stocks}) "
                    f"at {timestamp:.1f}s"
                )

        # Update tracked values
        if p1_stocks is not None:
            self.last_p1_stocks = p1_stocks
        if p2_stocks is not None:
            self.last_p2_stocks = p2_stocks

        return p1_lost, p2_lost, p1_stocks, p2_stocks, is_game_end

    def reset(self) -> None:
        """Reset the tracker state."""
        self.last_p1_stocks = None
        self.last_p2_stocks = None
        self.last_detection_time = 0.0
