"""Character detection for SSBU frames.

Identifies which characters are being played using:
1. Gemini Vision API (primary, most accurate)
2. Template matching fallback (for offline use or API failures)

The Gemini-based detection is preferred as it handles varying skins,
visual effects, and camera angles better than template matching.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Flag to control whether to use Gemini (can be set via environment)
USE_GEMINI = os.environ.get("RUNITBACK_USE_GEMINI", "1").lower() in ("1", "true", "yes")


# Character portrait ROI locations (normalized coordinates)
# Based on standard SSBU HUD layout at 4:43 of St7XW7w_jaA
# P1 icon: left side near damage %, P2 icon: right side near damage %
# These are the small character portraits that appear next to the damage percentage
P1_PORTRAIT_ROI = (0.195, 0.842, 0.277, 0.974)  # Player 1 portrait (left side)
P2_PORTRAIT_ROI = (0.598, 0.835, 0.664, 0.946)  # Player 2 portrait (right side)

# Loading screen portrait locations (larger, easier to detect)
# These appear briefly before each game starts
LOADING_P1_PORTRAIT_ROI = (0.10, 0.20, 0.40, 0.80)
LOADING_P2_PORTRAIT_ROI = (0.60, 0.20, 0.90, 0.80)

# Stock icon ROI locations (normalized coordinates)
# Stock icons appear below the player tag and damage percentage
# Each player can have up to 3 stock icons (in standard tournament rules)
# Coordinates determined from Dark Pit vs Ken replay (St7XW7w_jaA)
# P1 stocks: to the right of the character portrait, below player tag
# P2 stocks: to the left of the character portrait, below player tag
P1_STOCK_ROI = (0.215, 0.940, 0.280, 0.995)  # Player 1 stock icons region
P2_STOCK_ROI = (0.620, 0.960, 0.710, 1.000)  # Player 2 stock icons region

# Individual stock icon size (approximate, normalized)
# Stock icons are ~64x64 pixels at 1080p, circular with beige background
STOCK_ICON_SIZE = 64

# Reference types supported (in order of preference for HUD matching)
# hud_portrait: Captured from actual in-game HUD - best match for gameplay detection
# icon: Full character artwork - used as fallback
# loading_screen: Loading screen portraits - not used for HUD matching
# stock_icon: Small stock icons - not used for HUD matching
REFERENCE_TYPES = ["hud_portrait", "icon", "loading_screen", "stock_icon"]

# Default reference directory (relative to project root)
DEFAULT_REFERENCE_DIR = Path(__file__).parent.parent.parent / "data" / "character_references"


# Character list - all 89 characters in SSBU (as of final DLC)
CHARACTERS = [
    "Mario", "Donkey Kong", "Link", "Samus", "Dark Samus", "Yoshi",
    "Kirby", "Fox", "Pikachu", "Luigi", "Ness", "Captain Falcon",
    "Jigglypuff", "Peach", "Daisy", "Bowser", "Ice Climbers", "Sheik",
    "Zelda", "Dr. Mario", "Pichu", "Falco", "Marth", "Lucina",
    "Young Link", "Ganondorf", "Mewtwo", "Roy", "Chrom", "Mr. Game & Watch",
    "Meta Knight", "Pit", "Dark Pit", "Zero Suit Samus", "Wario", "Snake",
    "Ike", "Pokemon Trainer", "Diddy Kong", "Lucas", "Sonic", "King Dedede",
    "Olimar", "Lucario", "R.O.B.", "Toon Link", "Wolf", "Villager",
    "Mega Man", "Wii Fit Trainer", "Rosalina & Luma", "Little Mac", "Greninja",
    "Mii Brawler", "Mii Swordfighter", "Mii Gunner", "Palutena", "Pac-Man",
    "Robin", "Shulk", "Bowser Jr.", "Duck Hunt", "Ryu", "Ken", "Cloud",
    "Corrin", "Bayonetta", "Inkling", "Ridley", "Simon", "Richter",
    "King K. Rool", "Isabelle", "Incineroar", "Piranha Plant", "Joker",
    "Hero", "Banjo & Kazooie", "Terry", "Byleth", "Min Min", "Steve",
    "Sephiroth", "Pyra/Mythra", "Kazuya", "Sora",
]


@dataclass
class CharacterDetection:
    """Result of character detection."""

    name: str
    confidence: float

    @property
    def is_valid(self) -> bool:
        """Check if detection is valid."""
        return self.name != "Unknown" and self.confidence > 0.5


@dataclass
class MatchCharacters:
    """Characters detected for both players."""

    player1: CharacterDetection
    player2: CharacterDetection


@dataclass
class CharacterTemplate:
    """A character reference template with precomputed features."""

    name: str
    image: np.ndarray
    histogram: np.ndarray
    reference_type: str  # "loading_screen", "icon", or "stock_icon"
    mask: np.ndarray | None = None


class CharacterTemplateDatabase:
    """Database of character reference templates for matching."""

    def __init__(self, reference_dir: Path | None = None):
        """Initialize the template database.

        Args:
            reference_dir: Directory containing character references.
                          Defaults to data/character_references/
        """
        self.reference_dir = reference_dir or DEFAULT_REFERENCE_DIR
        self.templates: dict[str, list[CharacterTemplate]] = {}
        self._loaded = False

    def load_templates(self) -> int:
        """Load all character templates from the reference directory.

        Returns:
            Number of templates loaded
        """
        if self._loaded:
            return sum(len(t) for t in self.templates.values())

        count = 0
        for ref_type in REFERENCE_TYPES:
            type_dir = self.reference_dir / ref_type
            if not type_dir.exists():
                logger.debug(f"Reference directory not found: {type_dir}")
                continue

            for char_file in type_dir.glob("*.png"):
                char_name = self._filename_to_character_name(char_file.stem)
                if char_name:
                    template = self._load_template(char_file, char_name, ref_type)
                    if template:
                        if char_name not in self.templates:
                            self.templates[char_name] = []
                        self.templates[char_name].append(template)
                        count += 1
                        logger.debug(f"Loaded template: {char_name} ({ref_type})")

            # Also check for jpg files
            for char_file in type_dir.glob("*.jpg"):
                char_name = self._filename_to_character_name(char_file.stem)
                if char_name:
                    template = self._load_template(char_file, char_name, ref_type)
                    if template:
                        if char_name not in self.templates:
                            self.templates[char_name] = []
                        self.templates[char_name].append(template)
                        count += 1
                        logger.debug(f"Loaded template: {char_name} ({ref_type})")

        self._loaded = True
        logger.info(f"Loaded {count} character templates for {len(self.templates)} characters")
        return count

    def _filename_to_character_name(self, filename: str) -> str | None:
        """Convert filename to character name.

        Handles formats like:
        - mario.png -> Mario
        - mario_00.png -> Mario (skin variant)
        - donkey_kong.png -> Donkey Kong
        - donkey_kong_01.png -> Donkey Kong (skin variant)
        - mr_game_and_watch.png -> Mr. Game & Watch

        Args:
            filename: Filename without extension

        Returns:
            Matched character name or None
        """
        # Strip skin suffix if present (e.g., mario_00 -> mario)
        base_name = filename
        if filename.split("_")[-1].isdigit():
            # Remove the skin number suffix
            base_name = "_".join(filename.split("_")[:-1])

        # Normalize filename
        normalized = base_name.lower().replace("_", " ").replace("-", " ")

        # Direct lookup with normalization
        for char in CHARACTERS:
            char_normalized = char.lower().replace(".", "").replace("&", "and")
            if normalized == char_normalized or normalized.replace(" ", "") == char_normalized.replace(" ", ""):
                return char

        # Try matching with underscores replaced
        for char in CHARACTERS:
            if char.lower().replace(" ", "_").replace(".", "").replace("&", "_") == base_name.lower():
                return char

        # Also try matching variants
        name_variants = {
            "rob": "R.O.B.",
            "pac_man": "Pac-Man",
            "pyra_mythra": "Pyra/Mythra",
            "banjo_and_kazooie": "Banjo & Kazooie",
            "rosalina_and_luma": "Rosalina & Luma",
        }
        if base_name.lower() in name_variants:
            return name_variants[base_name.lower()]

        logger.warning(f"Could not match filename '{filename}' to any character")
        return None

    def _load_template(
        self, path: Path, char_name: str, ref_type: str
    ) -> CharacterTemplate | None:
        """Load a single template from file.

        Args:
            path: Path to image file
            char_name: Character name
            ref_type: Reference type

        Returns:
            CharacterTemplate or None if loading fails
        """
        try:
            # Load with alpha channel if present (IMREAD_UNCHANGED)
            image_with_alpha = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image_with_alpha is None:
                logger.warning(f"Failed to load image: {path}")
                return None

            # If image has alpha channel, use it to create a mask
            # and convert to 3-channel BGR
            if image_with_alpha.shape[2] == 4:
                # Extract alpha channel as mask
                alpha = image_with_alpha[:, :, 3]
                # Convert BGRA to BGR
                image = cv2.cvtColor(image_with_alpha, cv2.COLOR_BGRA2BGR)
                # Store mask in the template (we'll add it to CharacterTemplate)
                mask = alpha
            else:
                image = image_with_alpha
                mask = None

            # Compute HSV histogram for matching (masked if alpha available)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            if mask is not None:
                # Only compute histogram for non-transparent pixels
                histogram = cv2.calcHist(
                    [hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256]
                )
            else:
                histogram = cv2.calcHist(
                    [hsv], [0, 1], None, [30, 32], [0, 180, 0, 256]
                )
            cv2.normalize(histogram, histogram, 0, 1, cv2.NORM_MINMAX)

            template = CharacterTemplate(
                name=char_name,
                image=image,
                histogram=histogram,
                reference_type=ref_type,
                mask=mask,
            )
            return template
        except Exception as e:
            logger.warning(f"Error loading template {path}: {e}")
            return None

    def get_templates(self, char_name: str) -> list[CharacterTemplate]:
        """Get all templates for a character.

        Args:
            char_name: Character name

        Returns:
            List of templates for the character
        """
        if not self._loaded:
            self.load_templates()
        return self.templates.get(char_name, [])

    def get_all_templates(self) -> list[CharacterTemplate]:
        """Get all loaded templates.

        Returns:
            List of all templates
        """
        if not self._loaded:
            self.load_templates()
        return [t for templates in self.templates.values() for t in templates]

    def get_templates_by_type(self, ref_type: str) -> list[CharacterTemplate]:
        """Get all templates of a specific type.

        Args:
            ref_type: Reference type ("loading_screen", "icon", "stock_icon")

        Returns:
            List of templates of that type
        """
        if not self._loaded:
            self.load_templates()
        return [
            t for templates in self.templates.values()
            for t in templates
            if t.reference_type == ref_type
        ]


# Global template database instance
_template_db: CharacterTemplateDatabase | None = None


def get_template_database() -> CharacterTemplateDatabase:
    """Get or create the global template database.

    Returns:
        The template database instance
    """
    global _template_db
    if _template_db is None:
        _template_db = CharacterTemplateDatabase()
    return _template_db


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


def extract_stock_icons(frame: np.ndarray, player: int = 1) -> list[np.ndarray]:
    """Extract individual stock icons from a frame.

    Stock icons are small circular icons showing the character's head.
    They appear below the damage percentage in a row.

    Args:
        frame: Full video frame (BGR)
        player: Player number (1 or 2)

    Returns:
        List of stock icon images (may be 0-3 icons)
    """
    h, w = frame.shape[:2]

    # Get the stock icon region for this player
    roi_coords = P1_STOCK_ROI if player == 1 else P2_STOCK_ROI
    stock_region = get_roi(frame, roi_coords)

    if stock_region.size == 0:
        return []

    # Stock icons are roughly square and arranged horizontally
    # At 1080p, each stock icon is about 64x64 pixels
    region_h, region_w = stock_region.shape[:2]

    # Expected icon size based on frame dimensions
    # Stock icons are approximately 3.3% of frame width at 1080p (64/1920)
    icon_size = int(0.033 * w)
    if icon_size < 20:
        icon_size = 20  # Minimum size

    # The region should contain up to 3 stock icons
    # Extract individual icons by dividing region width by 3
    icons = []
    icon_width = region_w // 3

    for i in range(3):
        x_start = i * icon_width
        x_end = (i + 1) * icon_width
        icon = stock_region[:, x_start:x_end]

        # Check if this region actually contains an icon
        # (lost stocks may be grayed out or missing)
        if icon.size > 0 and icon.shape[0] > 10 and icon.shape[1] > 10:
            icons.append(icon)

    return icons


def match_stock_icon(
    stock_icon: np.ndarray, template: CharacterTemplate
) -> float:
    """Match a stock icon against a character template.

    Stock icons have a consistent beige/tan circular background,
    making them easier to match than the larger character portraits.

    Args:
        stock_icon: Cropped stock icon from gameplay (BGR)
        template: Character stock_icon template

    Returns:
        Match score (0-1, higher is better)
    """
    if stock_icon.size == 0 or template.image.size == 0:
        return 0.0

    # Resize template to match stock icon size
    h, w = stock_icon.shape[:2]
    resized = cv2.resize(template.image, (w, h))

    # Stock icons have a beige/tan background - create a circular mask
    # to focus on the character art in the center
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 2 - 2  # Slight margin

    # Create circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)  # type: ignore[call-overload]  # OpenCV stubs

    # Use template's alpha mask if available, otherwise use circular mask
    template_mask = getattr(template, 'mask', None)
    if template_mask is not None:
        template_mask_resized = cv2.resize(template_mask, (w, h))
        _, binary_mask = cv2.threshold(template_mask_resized, 200, 255, cv2.THRESH_BINARY)
        # Combine with circular mask
        combined_mask = cv2.bitwise_and(mask, binary_mask)
    else:
        combined_mask = mask

    opaque_count = np.sum(combined_mask > 0)
    if opaque_count == 0:
        return 0.0

    # 1. Color matching in masked region
    diff = cv2.absdiff(stock_icon, resized)
    diff_masked = diff.copy()
    diff_masked[combined_mask == 0] = 0
    mean_diff = np.sum(diff_masked) / (opaque_count * 3)
    color_score = 1.0 - min(1.0, mean_diff / 128.0)  # More lenient threshold

    # 2. Grayscale structural matching with mask
    stock_gray = cv2.cvtColor(stock_icon, cv2.COLOR_BGR2GRAY).astype(np.float32)
    template_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mask_float = combined_mask.astype(np.float32) / 255.0
    stock_mean = np.sum(stock_gray * mask_float) / opaque_count
    template_mean = np.sum(template_gray * mask_float) / opaque_count

    stock_centered = (stock_gray - stock_mean) * mask_float
    template_centered = (template_gray - template_mean) * mask_float

    numerator = np.sum(stock_centered * template_centered)
    denominator = np.sqrt(np.sum(stock_centered**2) * np.sum(template_centered**2))

    if denominator > 0:
        ncc_score = (numerator / denominator + 1) / 2
    else:
        ncc_score = 0.0

    # 3. Hue histogram matching (for character-specific colors)
    stock_hsv = cv2.cvtColor(stock_icon, cv2.COLOR_BGR2HSV)
    template_hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Only compare pixels with sufficient saturation (actual character, not background)
    sat_mask = ((stock_hsv[:, :, 1] > 40) & (template_hsv[:, :, 1] > 40)).astype(np.uint8) * 255
    final_mask = cv2.bitwise_and(combined_mask, sat_mask)

    if np.sum(final_mask > 0) > 50:
        stock_hist = cv2.calcHist([stock_hsv], [0], final_mask, [180], [0, 180])
        template_hist = cv2.calcHist([template_hsv], [0], final_mask, [180], [0, 180])
        cv2.normalize(stock_hist, stock_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(template_hist, template_hist, 0, 1, cv2.NORM_MINMAX)
        hue_corr = cv2.compareHist(stock_hist, template_hist, cv2.HISTCMP_CORREL)
        hue_score = max(0, (hue_corr + 1) / 2)
    else:
        hue_score = 0.5  # Neutral if not enough saturated pixels

    # Combine scores: structure most important for these small icons
    combined = 0.3 * color_score + 0.5 * ncc_score + 0.2 * hue_score

    return combined


def detect_character_from_stock_icons(
    frame: np.ndarray,
    player: int,
    templates: list[CharacterTemplate],
) -> CharacterDetection:
    """Detect a player's character from their stock icons.

    Uses center-weighted masked template matching. The stock icons have circular
    shapes with the character face in the center. By using the alpha mask and
    weighting the center more heavily, we can match the distinctive facial
    features while ignoring background color differences.

    Args:
        frame: Full video frame (BGR)
        player: Player number (1 or 2)
        templates: List of stock_icon templates to match against

    Returns:
        CharacterDetection with best match
    """
    # Get the stock icon region for this player
    roi_coords = P1_STOCK_ROI if player == 1 else P2_STOCK_ROI
    stock_region = get_roi(frame, roi_coords)

    if stock_region.size == 0 or not templates:
        return CharacterDetection(name="Unknown", confidence=0.0)

    region_h, region_w = stock_region.shape[:2]

    if region_h < 15 or region_w < 15:
        return CharacterDetection(name="Unknown", confidence=0.0)

    best_match_name = None
    best_match_score = 0.0

    # Estimate icon size based on region height
    icon_size = region_h - 2

    for template in templates:
        # Get template mask (alpha channel)
        mask = getattr(template, 'mask', None)
        if mask is None:
            continue

        # Resize template and mask to icon size
        resized_template = cv2.resize(template.image, (icon_size, icon_size))
        resized_mask = cv2.resize(mask, (icon_size, icon_size))

        # Create binary mask focusing on center (erode to avoid edge artifacts)
        _, binary_mask = cv2.threshold(resized_mask, 200, 255, cv2.THRESH_BINARY)
        # Erode mask to focus on center and avoid background at edges
        kernel = np.ones((3, 3), np.uint8)
        center_mask = cv2.erode(binary_mask, kernel, iterations=2)

        # Skip if template larger than region or mask too small
        if icon_size > region_w or icon_size > region_h:
            continue
        if np.count_nonzero(center_mask) < 50:
            continue

        try:
            # Use masked template matching with center-focused mask
            result = cv2.matchTemplate(
                stock_region, resized_template, cv2.TM_CCOEFF_NORMED, mask=center_mask
            )
            _, max_val, _, _ = cv2.minMaxLoc(result)

            # TM_CCOEFF_NORMED: -1 to 1, higher is better
            # Convert to 0-1 scale
            score = (max_val + 1) / 2

            if score > best_match_score:
                best_match_score = score
                best_match_name = template.name

        except cv2.error:
            continue

    if best_match_name and best_match_score > 0.6:
        return CharacterDetection(name=best_match_name, confidence=best_match_score)

    return CharacterDetection(name="Unknown", confidence=best_match_score)


def compute_histogram(image: np.ndarray) -> np.ndarray:
    """Compute normalized HSV histogram for an image.

    Args:
        image: BGR image

    Returns:
        Normalized 2D histogram (H, S channels)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, 0, 1, cv2.NORM_MINMAX)
    return histogram


def match_template_histogram(
    roi: np.ndarray, template: CharacterTemplate
) -> float:
    """Match ROI against a template using histogram comparison.

    Args:
        roi: Region of interest (BGR image)
        template: Character template to match against

    Returns:
        Match score (0-1, higher is better)
    """
    roi_hist = compute_histogram(roi)

    # Use correlation method (higher = better match)
    score = cv2.compareHist(roi_hist, template.histogram, cv2.HISTCMP_CORREL)

    # Normalize to 0-1 range (correlation can be -1 to 1)
    return max(0, (score + 1) / 2)


def match_template_direct(roi: np.ndarray, template: CharacterTemplate) -> float:
    """Match ROI against template using direct template matching.

    Args:
        roi: Region of interest (BGR image)
        template: Character template to match against

    Returns:
        Match score (0-1, higher is better)
    """
    # Resize template to match ROI size
    resized = cv2.resize(template.image, (roi.shape[1], roi.shape[0]))

    # Use normalized cross-correlation
    result = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
    score = float(np.max(result))

    # Normalize to 0-1 range (TM_CCOEFF_NORMED can return -1 to 1)
    return max(0, (score + 1) / 2)


def match_template_edges(roi: np.ndarray, template: CharacterTemplate) -> float:
    """Match ROI against template using edge-based comparison.

    Edge matching is more robust to background differences since it focuses
    on the shape/structure of the character rather than colors.

    Args:
        roi: Region of interest (BGR image)
        template: Character template to match against

    Returns:
        Match score (0-1, higher is better)
    """
    # Resize template to match ROI size
    resized = cv2.resize(template.image, (roi.shape[1], roi.shape[0]))

    # Convert to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Extract edges
    roi_edges = cv2.Canny(roi_gray, 50, 150)
    template_edges = cv2.Canny(template_gray, 50, 150)

    # Use template matching on edges
    result = cv2.matchTemplate(roi_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    score = float(np.max(result))

    # Normalize to 0-1 range
    return max(0, (score + 1) / 2)


def match_template_features(roi: np.ndarray, template: CharacterTemplate) -> float:
    """Match ROI against template using ORB feature detection.

    Feature-based matching is robust to background differences and lighting
    variations since it matches distinctive visual features (corners, edges).

    Args:
        roi: Region of interest (BGR image)
        template: Character template to match against

    Returns:
        Match score (0-1, higher is better)
    """
    # Resize template to match ROI size
    h, w = roi.shape[:2]
    resized = cv2.resize(template.image, (w, h))

    # Convert to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=100)  # type: ignore[attr-defined]  # exists at runtime, missing from stubs

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(roi_gray, None)
    kp2, des2 = orb.detectAndCompute(template_gray, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0.0

    # Match descriptors using brute force with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Score based on number of good matches (distance < 50)
    good_matches = [m for m in matches if m.distance < 50]
    score = len(good_matches) / max(len(kp1), len(kp2), 1)

    return min(1.0, score)


def create_background_mask(roi: np.ndarray) -> np.ndarray:
    """Create a mask that excludes the red/blue HUD background colors.

    The in-game character portraits have colored backgrounds:
    - Player 1: Red background (hue ~0-10 or 170-180, high saturation)
    - Player 2: Blue background (hue ~90-145, varies in saturation)

    This function creates a mask where background pixels are 0 (excluded)
    and character pixels are 255 (included).

    Args:
        roi: Region of interest (BGR image)

    Returns:
        Binary mask (0 = background, 255 = foreground/character)
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Red background: hue near 0 or 180, moderate-high saturation
    # Be careful not to exclude character reds (like Mario's cap) -
    # background red is typically very saturated and at edges
    red_bg_mask = (((h < 15) | (h > 165)) & (s > 120) & (v > 50))

    # Blue background: hue around 90-145, can be less saturated
    # More aggressive to catch the blue HUD background
    blue_bg_mask = ((h > 90) & (h < 145) & (s > 50) & (v > 30))

    # Gray/transparent background in templates: low saturation
    gray_mask = (s < 25)

    # Background is any of these
    background = red_bg_mask | blue_bg_mask | gray_mask

    # Foreground is everything else
    foreground_mask = (~background).astype(np.uint8) * 255

    return foreground_mask


def match_template_masked_histogram(
    roi: np.ndarray, template: CharacterTemplate
) -> float:
    """Match using histogram comparison with background masking.

    Masks out the red/blue HUD backgrounds in the ROI and the gray
    background in the template before comparing color histograms.
    This focuses the comparison on the actual character colors.

    Args:
        roi: Region of interest (BGR image)
        template: Character template to match against

    Returns:
        Match score (0-1, higher is better)
    """
    h, w = roi.shape[:2]
    resized_template = cv2.resize(template.image, (w, h))

    # Create masks to exclude backgrounds
    roi_mask = create_background_mask(roi)
    template_mask = create_background_mask(resized_template)

    # Convert to HSV for histogram
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    template_hsv = cv2.cvtColor(resized_template, cv2.COLOR_BGR2HSV)

    # Compute masked histograms (only foreground pixels)
    roi_hist = cv2.calcHist([roi_hsv], [0, 1], roi_mask, [30, 32], [0, 180, 0, 256])
    template_hist = cv2.calcHist([template_hsv], [0, 1], template_mask, [30, 32], [0, 180, 0, 256])

    # Normalize histograms
    cv2.normalize(roi_hist, roi_hist, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(template_hist, template_hist, 0, 1, cv2.NORM_MINMAX)

    # Compare histograms
    score = cv2.compareHist(roi_hist, template_hist, cv2.HISTCMP_CORREL)
    return max(0, (score + 1) / 2)


def match_template_center_histogram(
    roi: np.ndarray, template: CharacterTemplate, center_fraction: float = 0.6
) -> float:
    """Match using histogram of center region only.

    Focusing on the center reduces background interference since character
    faces are typically centered while backgrounds are at the edges.

    Args:
        roi: Region of interest (BGR image)
        template: Character template to match against
        center_fraction: Fraction of image to use as center (0-1)

    Returns:
        Match score (0-1, higher is better)
    """
    h, w = roi.shape[:2]
    margin_h = int(h * (1 - center_fraction) / 2)
    margin_w = int(w * (1 - center_fraction) / 2)

    # Get center crops
    roi_center = roi[margin_h:h-margin_h, margin_w:w-margin_w]

    resized = cv2.resize(template.image, (w, h))
    template_center = resized[margin_h:h-margin_h, margin_w:w-margin_w]

    # Compute histograms
    roi_hist = compute_histogram(roi_center)

    hsv = cv2.cvtColor(template_center, cv2.COLOR_BGR2HSV)
    template_hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(template_hist, template_hist, 0, 1, cv2.NORM_MINMAX)

    score = cv2.compareHist(roi_hist, template_hist, cv2.HISTCMP_CORREL)
    return max(0, (score + 1) / 2)


def match_template_structural(roi: np.ndarray, template: CharacterTemplate) -> float:
    """Match using structural similarity on grayscale images.

    This compares the overall structure/shape rather than colors,
    which is more robust to background color variations.

    Args:
        roi: Region of interest (BGR image)
        template: Character template to match against

    Returns:
        Match score (0-1, higher is better)
    """
    # Resize template to match ROI size
    resized = cv2.resize(template.image, (roi.shape[1], roi.shape[0]))

    # Convert to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Use normalized cross-correlation on grayscale
    result = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    score = float(np.max(result))

    # Normalize to 0-1 range
    return max(0, (score + 1) / 2)


def match_template_combined(roi: np.ndarray, template: CharacterTemplate) -> float:
    """Match using combination of center histogram and structural matching.

    Combines:
    - Center histogram: Color distribution of the center region (character face)
    - Structural: Grayscale shape/structure matching

    Args:
        roi: Region of interest (BGR image)
        template: Character template to match against

    Returns:
        Combined match score (0-1, higher is better)
    """
    center_score = match_template_center_histogram(roi, template, center_fraction=0.5)
    structural_score = match_template_structural(roi, template)

    # Weight structural matching slightly higher as it's more robust
    # to the varying backgrounds in the HUD
    return 0.4 * center_score + 0.6 * structural_score


def match_template_alpha_masked(roi: np.ndarray, template: CharacterTemplate) -> float:
    """Match using combined color, structure, and hue matching with alpha mask.

    The icon references have transparent backgrounds while the in-game HUD shows
    the character against the game scene. This function combines multiple matching
    approaches to be robust to this background difference:
    - Color matching: Direct pixel comparison in opaque regions
    - Grayscale NCC: Structural similarity independent of color
    - Hue histogram: Color distribution in saturated regions

    Args:
        roi: Region of interest (BGR image)
        template: Character template with mask attribute

    Returns:
        Match score (0-1, higher is better)
    """
    h, w = roi.shape[:2]

    # Get alpha mask
    mask = getattr(template, 'mask', None)
    if mask is None:
        # No alpha - fall back to simple histogram matching
        return match_template_histogram(roi, template)

    # Resize template and mask
    resized = cv2.resize(template.image, (w, h))
    alpha_resized = cv2.resize(mask, (w, h))
    _, binary_mask = cv2.threshold(alpha_resized, 200, 255, cv2.THRESH_BINARY)

    opaque_count = np.sum(binary_mask > 0)
    if opaque_count == 0:
        return 0.0

    # 1. Color matching: pixel difference in opaque regions
    diff = cv2.absdiff(roi, resized)
    diff[binary_mask == 0] = 0
    mean_diff = np.sum(diff) / (opaque_count * 3)
    color_score = 1.0 - (mean_diff / 255.0)

    # 2. Grayscale NCC: structural similarity
    mask_float = binary_mask.astype(np.float32) / 255.0
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    template_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

    roi_mean = np.sum(roi_gray * mask_float) / opaque_count
    template_mean = np.sum(template_gray * mask_float) / opaque_count

    roi_centered = (roi_gray - roi_mean) * mask_float
    template_centered = (template_gray - template_mean) * mask_float

    numerator = np.sum(roi_centered * template_centered)
    denominator = np.sqrt(np.sum(roi_centered**2) * np.sum(template_centered**2))

    if denominator > 0:
        gray_score = (numerator / denominator + 1) / 2
    else:
        gray_score = 0.0

    # 3. Hue histogram: color distribution in saturated regions
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    template_hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Filter low saturation pixels
    sat_mask = ((roi_hsv[:, :, 1] > 30) & (template_hsv[:, :, 1] > 30)).astype(np.uint8) * 255
    combined_mask = cv2.bitwise_and(binary_mask, sat_mask)

    roi_hist = cv2.calcHist([roi_hsv], [0], combined_mask, [180], [0, 180])
    template_hist = cv2.calcHist([template_hsv], [0], combined_mask, [180], [0, 180])

    cv2.normalize(roi_hist, roi_hist, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(template_hist, template_hist, 0, 1, cv2.NORM_MINMAX)

    hue_corr = cv2.compareHist(roi_hist, template_hist, cv2.HISTCMP_CORREL)
    hue_score = max(0, (hue_corr + 1) / 2)

    # Combine scores: weight color highest, then structure, then hue
    combined = 0.5 * color_score + 0.3 * gray_score + 0.2 * hue_score

    return combined


def match_template_hybrid(
    roi: np.ndarray,
    template: CharacterTemplate,
    feature_weight: float = 0.3,
    masked_hist_weight: float = 0.7,
) -> float:
    """Match ROI against template using feature and masked-histogram matching.

    Combines two robust matching methods:
    - Feature (ORB): Matches distinctive visual features, robust to backgrounds
    - Masked histogram: Color distribution with background colors masked out,
      which removes the red/blue HUD backgrounds from comparison

    Args:
        roi: Region of interest (BGR image)
        template: Character template to match against
        feature_weight: Weight for feature matching score
        masked_hist_weight: Weight for masked histogram matching score

    Returns:
        Combined match score (0-1, higher is better)
    """
    feature_score = match_template_features(roi, template)
    masked_hist_score = match_template_masked_histogram(roi, template)

    # Weighted combination - favor masked histogram for color discrimination
    combined = (
        feature_weight * feature_score +
        masked_hist_weight * masked_hist_score
    )

    return combined


def match_portrait(
    portrait: np.ndarray,
    templates: list[CharacterTemplate] | None = None,
    matching_mode: str = "combined",
) -> CharacterDetection:
    """Detect character from a portrait image.

    Args:
        portrait: BGR image of character portrait
        templates: List of character templates to match against.
                  If None, loads from global database.
        matching_mode: Matching algorithm to use:
            - "combined": Center histogram + structural matching (default, best for HUD)
            - "masked_histogram": Histogram with background masking
            - "direct": Direct template matching
            - "histogram": Basic histogram comparison (faster)
            - "hybrid": Combine feature + masked histogram matching

    Returns:
        CharacterDetection with name and confidence
    """
    if portrait.size == 0:
        return CharacterDetection(name="Unknown", confidence=0.0)

    if templates is None:
        db = get_template_database()
        templates = db.get_all_templates()

    if not templates:
        logger.debug("No character templates loaded")
        return CharacterDetection(name="Unknown", confidence=0.0)

    best_match = None
    best_score = 0.0

    for template in templates:
        if matching_mode == "histogram":
            score = match_template_histogram(portrait, template)
        elif matching_mode == "direct":
            score = match_template_direct(portrait, template)
        elif matching_mode == "hybrid":
            score = match_template_hybrid(portrait, template)
        elif matching_mode == "masked_histogram":
            score = match_template_masked_histogram(portrait, template)
        else:  # combined (default) - best for matching icon artwork to HUD
            score = match_template_combined(portrait, template)

        if score > best_score:
            best_score = score
            best_match = template.name

    # Apply confidence threshold
    if best_match and best_score > 0.5:
        return CharacterDetection(name=best_match, confidence=best_score)

    return CharacterDetection(name="Unknown", confidence=best_score)


def is_loading_screen(frame: np.ndarray) -> bool:
    """Check if the frame appears to be a loading screen.

    Loading screens typically have:
    - Large character portraits
    - VS text or similar indicators
    - Less busy background than gameplay

    Args:
        frame: Full video frame

    Returns:
        True if this looks like a loading screen
    """
    h, w = frame.shape[:2]

    # Check center region for "VS" or loading indicator
    center_roi = frame[int(0.4 * h) : int(0.6 * h), int(0.4 * w) : int(0.6 * w)]

    # Loading screens tend to have less motion/detail in center
    gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Loading screens have relatively low edge density in center
    # (just VS text, not gameplay)
    return bool(edge_density < 0.1)


def detect_characters(
    frame: np.ndarray,
    from_loading_screen: bool = False,
) -> MatchCharacters:
    """Detect both players' characters from a frame.

    Args:
        frame: Full video frame (BGR)
        from_loading_screen: If True, use loading screen ROI locations

    Returns:
        MatchCharacters with detections for both players
    """
    db = get_template_database()

    if from_loading_screen:
        p1_roi = get_roi(frame, LOADING_P1_PORTRAIT_ROI)
        p2_roi = get_roi(frame, LOADING_P2_PORTRAIT_ROI)
        templates = db.get_templates_by_type("loading_screen")
    else:
        p1_roi = get_roi(frame, P1_PORTRAIT_ROI)
        p2_roi = get_roi(frame, P2_PORTRAIT_ROI)
        templates = db.get_templates_by_type("icon")

    # Fall back to all templates if specific type not available
    if not templates:
        templates = db.get_all_templates()

    p1_detection = match_portrait(p1_roi, templates)
    p2_detection = match_portrait(p2_roi, templates)

    return MatchCharacters(player1=p1_detection, player2=p2_detection)


def detect_character_from_portrait(
    frame: np.ndarray,
    player: int,
    templates: list[CharacterTemplate],
) -> CharacterDetection:
    """Detect a player's character from their HUD portrait.

    The HUD portraits (next to damage %) are larger than stock icons and show
    more detail. Uses the icon templates with alpha-masked matching to compare
    the character art while ignoring the background (which shows the game scene
    in gameplay but is transparent in templates).

    Args:
        frame: Full video frame (BGR)
        player: Player number (1 or 2)
        templates: List of icon templates to match against

    Returns:
        CharacterDetection with best match
    """
    # Get the portrait region for this player
    roi_coords = P1_PORTRAIT_ROI if player == 1 else P2_PORTRAIT_ROI
    portrait = get_roi(frame, roi_coords)

    if portrait.size == 0 or not templates:
        return CharacterDetection(name="Unknown", confidence=0.0)

    portrait_h, portrait_w = portrait.shape[:2]

    if portrait_h < 30 or portrait_w < 30:
        return CharacterDetection(name="Unknown", confidence=0.0)

    best_match_name = None
    best_match_score = 0.0

    # Use smaller dimension for square template size
    icon_size = min(portrait_h, portrait_w)

    for template in templates:
        # Get template mask (alpha channel)
        mask = getattr(template, 'mask', None)
        if mask is None:
            continue

        # Resize template and mask to icon size
        resized_template = cv2.resize(template.image, (icon_size, icon_size))
        resized_mask = cv2.resize(mask, (icon_size, icon_size))

        # Create binary mask from alpha, erode to focus on center
        _, binary_mask = cv2.threshold(resized_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        center_mask = cv2.erode(binary_mask, kernel, iterations=1)

        if np.count_nonzero(center_mask) < 100:
            continue

        try:
            # Resize portrait to match template if needed
            if portrait_h != icon_size or portrait_w != icon_size:
                resized_portrait = cv2.resize(portrait, (icon_size, icon_size))
            else:
                resized_portrait = portrait

            # Direct pixel comparison with mask
            diff = cv2.absdiff(resized_portrait, resized_template)
            diff_masked = diff.copy()
            diff_masked[center_mask == 0] = 0

            # Calculate mean difference only on masked pixels
            mask_pixels = np.count_nonzero(center_mask)
            mean_diff = np.sum(diff_masked) / (mask_pixels * 3)

            # Convert to similarity score (lower diff = higher score)
            score = 1.0 - min(1.0, mean_diff / 128.0)

            if score > best_match_score:
                best_match_score = score
                best_match_name = template.name

        except cv2.error:
            continue

    if best_match_name and best_match_score > 0.5:
        return CharacterDetection(name=best_match_name, confidence=best_match_score)

    return CharacterDetection(name="Unknown", confidence=best_match_score)


def detect_characters_from_frames(
    frames: list[np.ndarray],
    sample_positions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
    p1_hints: list[str] | None = None,
    p2_hints: list[str] | None = None,
) -> MatchCharacters:
    """Detect characters from frames using Gemini Vision API.

    Uses Google's Gemini Flash model to identify characters from gameplay
    screenshots. Falls back to template matching if Gemini is unavailable.

    Since characters don't change during a game, we sample 5 frames
    (10%, 25%, 50%, 75%, 90%) and vote for robust detection.

    Character hints from video title can be provided to help resolve
    ambiguous detections.

    Args:
        frames: List of video frames from a single game
        sample_positions: Frame positions to sample (as fractions 0-1)
        p1_hints: Character names that might be player 1 (from video title)
        p2_hints: Character names that might be player 2 (from video title)

    Returns:
        MatchCharacters with best detections
    """
    if not frames:
        return MatchCharacters(
            player1=CharacterDetection(name="Unknown", confidence=0.0),
            player2=CharacterDetection(name="Unknown", confidence=0.0),
        )

    # Try Gemini first (if enabled and API key available)
    if USE_GEMINI:
        try:
            from runitback.cv.llm_detector import detect_characters_from_frames_gemini

            player1, player2 = detect_characters_from_frames_gemini(
                frames,
                sample_positions=sample_positions,
                p1_hints=p1_hints,
                p2_hints=p2_hints,
            )

            if player1 != "Unknown" or player2 != "Unknown":
                logger.info(f"Gemini detected: P1={player1}, P2={player2}")
                return MatchCharacters(
                    player1=CharacterDetection(name=player1, confidence=0.9),
                    player2=CharacterDetection(name=player2, confidence=0.9),
                )
            else:
                logger.warning("Gemini returned Unknown for both players, falling back to template matching")
        except ImportError:
            logger.warning("Gemini not available, using template matching")
        except Exception as e:
            logger.warning(f"Gemini detection failed: {e}, falling back to template matching")

    # Fallback to template matching
    return _detect_characters_template_matching(frames, sample_positions)


def _detect_characters_template_matching(
    frames: list[np.ndarray],
    sample_positions: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> MatchCharacters:
    """Fallback template matching for character detection.

    Uses the character portraits in the HUD (next to damage percentage) for
    character detection with template matching.

    Args:
        frames: List of video frames from a single game
        sample_positions: Frame positions to sample (as fractions 0-1)

    Returns:
        MatchCharacters with best detections
    """
    # Get icon templates for HUD portrait matching
    db = get_template_database()
    icon_templates = db.get_templates_by_type("icon")

    if not icon_templates:
        logger.warning("No icon templates found")
        return MatchCharacters(
            player1=CharacterDetection(name="Unknown", confidence=0.0),
            player2=CharacterDetection(name="Unknown", confidence=0.0),
        )

    # Aggregate scores per character across all sampled frames
    p1_char_scores: dict[str, float] = {}
    p2_char_scores: dict[str, float] = {}

    # Sample frames at specified positions (e.g., 25%, 50%, 75%)
    for pos in sample_positions:
        idx = int(pos * (len(frames) - 1))
        frame = frames[idx]

        # Detect characters from HUD portraits for this frame
        p1_detection = detect_character_from_portrait(frame, 1, icon_templates)
        p2_detection = detect_character_from_portrait(frame, 2, icon_templates)

        # Accumulate scores
        if p1_detection.name != "Unknown":
            p1_char_scores[p1_detection.name] = (
                p1_char_scores.get(p1_detection.name, 0) + p1_detection.confidence
            )
        if p2_detection.name != "Unknown":
            p2_char_scores[p2_detection.name] = (
                p2_char_scores.get(p2_detection.name, 0) + p2_detection.confidence
            )

    # Find best matches based on total accumulated score
    def get_best(char_scores: dict, num_samples: int) -> CharacterDetection:
        if not char_scores:
            return CharacterDetection(name="Unknown", confidence=0.0)

        best_name = max(char_scores, key=lambda c: char_scores[c])
        # Average confidence across samples
        avg_conf = char_scores[best_name] / num_samples
        return CharacterDetection(name=best_name, confidence=avg_conf)

    num_samples = len(sample_positions)
    return MatchCharacters(
        player1=get_best(p1_char_scores, num_samples),
        player2=get_best(p2_char_scores, num_samples),
    )


def get_all_character_names() -> list[str]:
    """Get list of all character names.

    Returns:
        List of character names
    """
    return CHARACTERS.copy()


def add_character_reference(
    image_path: Path,
    character_name: str,
    reference_type: str,
    reference_dir: Path | None = None,
) -> bool:
    """Add a character reference image to the database.

    Args:
        image_path: Path to the source image
        character_name: Name of the character
        reference_type: Type of reference ("loading_screen", "icon", "stock_icon")
        reference_dir: Optional override for reference directory

    Returns:
        True if successful
    """
    if reference_type not in REFERENCE_TYPES:
        logger.error(f"Invalid reference type: {reference_type}. "
                    f"Must be one of {REFERENCE_TYPES}")
        return False

    if character_name not in CHARACTERS:
        logger.warning(f"Character '{character_name}' not in standard list. "
                      "Proceeding anyway...")

    ref_dir = reference_dir or DEFAULT_REFERENCE_DIR
    type_dir = ref_dir / reference_type

    # Create directory if needed
    type_dir.mkdir(parents=True, exist_ok=True)

    # Normalize filename
    filename = character_name.lower().replace(" ", "_").replace(".", "").replace("&", "and")
    dest_path = type_dir / f"{filename}.png"

    # Load and save image (ensures consistent format)
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return False

    cv2.imwrite(str(dest_path), image)
    logger.info(f"Added reference for {character_name} ({reference_type}) at {dest_path}")

    # Reload templates
    global _template_db
    _template_db = None

    return True
