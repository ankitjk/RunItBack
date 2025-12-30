"""OCR utilities for reading damage percentages from SSBU frames."""

import logging
import re
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy load PaddleOCR to avoid slow import on startup
_ocr_instance = None


def get_ocr():
    """Get or create the PaddleOCR instance (lazy loading)."""
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR

        # Use English model with document preprocessor disabled
        # to avoid unwanted rotation of game UI elements
        _ocr_instance = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _ocr_instance


def preprocess_damage_roi(roi: np.ndarray) -> np.ndarray:
    """Preprocess a damage percentage ROI for better OCR accuracy.

    The SSBU damage display has:
    - White/colored numbers on a dark background
    - A percentage symbol
    - Sometimes a decimal point for high damage

    Args:
        roi: BGR image of the damage percentage region

    Returns:
        Preprocessed grayscale image optimized for OCR
    """
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Threshold to get white text
    # SSBU damage numbers are typically white/bright
    _, binary = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)

    # Scale up for better OCR (PaddleOCR works better with larger text)
    scale_factor = 2
    scaled = cv2.resize(
        binary,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC,
    )

    # Add padding around the text
    padded = cv2.copyMakeBorder(
        scaled, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
    )

    return padded


def extract_damage_value(text: str) -> Optional[float]:
    """Extract numeric damage value from OCR text.

    Args:
        text: Raw OCR text output

    Returns:
        Damage value as float, or None if parsing fails
    """
    # Remove common OCR artifacts and normalize
    text = text.strip().upper()

    # Remove percentage symbol and common misreads
    text = text.replace("%", "").replace("S", "5").replace("O", "0")
    text = text.replace("I", "1").replace("L", "1").replace("Z", "2")
    text = text.replace("B", "8").replace("G", "6")

    # Try to find a number pattern (allowing decimal for high damage)
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        try:
            value = float(match.group(1))
            # SSBU damage caps at 999%
            if 0 <= value <= 999:
                return value
        except ValueError:
            pass

    return None


def read_damage_percentage(
    roi: np.ndarray,
    confidence_threshold: float = 0.3,
) -> tuple[Optional[float], float]:
    """Read damage percentage from an image region.

    Args:
        roi: BGR image containing the damage percentage
        confidence_threshold: Minimum OCR confidence to accept result

    Returns:
        Tuple of (damage_value, confidence). damage_value is None if OCR fails.
    """
    if roi is None or roi.size == 0:
        return None, 0.0

    try:
        ocr = get_ocr()
        # Use predict() for newer PaddleOCR versions
        # Pass the original ROI directly - PaddleOCR handles preprocessing
        results = ocr.predict(roi)

        if not results or len(results) == 0:
            return None, 0.0

        # New API returns OCRResult objects with rec_texts and rec_scores
        result = results[0]

        # Handle new OCRResult format
        if hasattr(result, 'rec_texts'):
            texts = result.rec_texts
            scores = result.rec_scores
        elif hasattr(result, 'keys'):
            # Dict-like result
            texts = result.get('rec_texts', [])
            scores = result.get('rec_scores', [])
        else:
            # Try old format
            return _read_damage_old_format(results, confidence_threshold)

        if not texts:
            return None, 0.0

        # Combine all detected text
        all_text = " ".join(texts)
        avg_confidence = sum(scores) / len(scores) if scores else 0.0

        logger.debug(f"OCR detected: '{all_text}' (conf: {avg_confidence:.2f})")

        if avg_confidence < confidence_threshold:
            logger.debug(f"Low OCR confidence: {avg_confidence:.2f} for text: {all_text}")
            return None, avg_confidence

        damage = extract_damage_value(all_text)
        return damage, avg_confidence

    except Exception as e:
        logger.warning(f"OCR error: {e}")
        return None, 0.0


def _read_damage_old_format(results, confidence_threshold: float) -> tuple[Optional[float], float]:
    """Handle old PaddleOCR result format for backward compatibility."""
    if not results or not results[0]:
        return None, 0.0

    all_text = ""
    total_confidence = 0.0
    count = 0

    for line in results[0]:
        if line and len(line) >= 2:
            text_info = line[1]
            if isinstance(text_info, tuple):
                text, confidence = text_info
            elif isinstance(text_info, dict):
                text = text_info.get("text", "")
                confidence = text_info.get("score", 0.0)
            else:
                text = str(text_info)
                confidence = 0.5
            all_text += text
            total_confidence += confidence
            count += 1

    if count == 0:
        return None, 0.0

    avg_confidence = total_confidence / count

    if avg_confidence < confidence_threshold:
        return None, avg_confidence

    damage = extract_damage_value(all_text)
    return damage, avg_confidence


def read_damage_with_fallback(
    roi: np.ndarray,
    previous_value: Optional[float] = None,
) -> Optional[float]:
    """Read damage with fallback to previous value on OCR failure.

    This helps maintain continuity when OCR temporarily fails.

    Args:
        roi: BGR image containing the damage percentage
        previous_value: Previous damage reading (used as fallback)

    Returns:
        Damage value, falling back to previous_value if OCR fails
    """
    damage, confidence = read_damage_percentage(roi)

    if damage is not None:
        # Sanity check: damage shouldn't decrease by more than a small amount
        # (it can decrease slightly due to staling or OCR jitter)
        if previous_value is not None and damage < previous_value - 5:
            # This might be a stock loss (damage resets to 0)
            if damage < 10:
                return damage  # Likely a new stock
            else:
                logger.debug(
                    f"Suspicious damage decrease: {previous_value} -> {damage}"
                )
                return previous_value

        return damage

    return previous_value
