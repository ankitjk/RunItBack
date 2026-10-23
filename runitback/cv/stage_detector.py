"""Stage detection for SSBU frames.

Identifies which tournament-legal stage is being played using
reference image matching with color histogram comparison.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Path to stage reference images
STAGE_REFERENCES_DIR = Path(__file__).parent.parent.parent / "data" / "stage_references"

# Cache for loaded reference histograms
_reference_histograms: dict["Stage", list[np.ndarray]] = {}


def _get_stage_from_filename(filename: str) -> Optional["Stage"]:
    """Map a reference filename to a Stage enum.

    Supports multiple references per stage with naming conventions:
    - stage_name.png (single reference)
    - stage_name_1.png, stage_name_2.png (multiple references)
    - stage_name_day.png, stage_name_night.png (descriptive variants)
    """
    name = filename.replace(".png", "").replace(".jpg", "")

    # Try exact match first
    try:
        return Stage(name)
    except ValueError:
        pass

    # Try matching with suffix stripped (e.g., "battlefield_1" -> "battlefield")
    for stage in Stage:
        if name.startswith(stage.value):
            # Check if remainder is a valid suffix (_, digit, or descriptive)
            remainder = name[len(stage.value):]
            if remainder == "" or remainder.startswith("_"):
                return stage

    return None


def load_stage_references() -> dict["Stage", list[np.ndarray]]:
    """Load and cache reference histograms for all available stages.

    Supports multiple reference images per stage for handling:
    - Variable backgrounds (time of day, scene changes)
    - Different gameplay moments

    Returns:
        Dictionary mapping Stage enum to list of reference histograms
    """
    global _reference_histograms

    if _reference_histograms:
        return _reference_histograms

    if not STAGE_REFERENCES_DIR.exists():
        logger.warning(f"Stage references directory not found: {STAGE_REFERENCES_DIR}")
        return {}

    # Collect all histograms per stage
    stage_hists: dict[Stage, list[np.ndarray]] = {}

    # Support both PNG and JPG formats
    for pattern in ["*.png", "*.jpg", "*.jpeg"]:
        for ref_file in STAGE_REFERENCES_DIR.glob(pattern):
            stage = _get_stage_from_filename(ref_file.name)
            if stage is None:
                logger.debug(f"Skipping unrecognized reference: {ref_file.name}")
                continue

            ref_img = cv2.imread(str(ref_file))
            if ref_img is None:
                logger.warning(f"Failed to load reference image: {ref_file}")
                continue

            # Compute histogram
            hist = compute_color_histogram(ref_img)

            if stage not in stage_hists:
                stage_hists[stage] = []
            stage_hists[stage].append(hist)
            logger.debug(f"Loaded reference {ref_file.name} for {stage.display_name}")

    _reference_histograms.update(stage_hists)

    total_refs = sum(len(hists) for hists in stage_hists.values())
    logger.info(f"Loaded {total_refs} reference(s) for {len(stage_hists)} stages")
    return _reference_histograms


class Stage(Enum):
    """Tournament-legal stages (hazards off)."""

    SMALL_BATTLEFIELD = "small_battlefield"
    BATTLEFIELD = "battlefield"
    FINAL_DESTINATION = "final_destination"
    TOWN_AND_CITY = "town_and_city"
    SMASHVILLE = "smashville"
    HOLLOW_BASTION = "hollow_bastion"
    KALOS_POKEMON_LEAGUE = "kalos_pokemon_league"
    POKEMON_STADIUM_2 = "pokemon_stadium_2"
    UNKNOWN = "unknown"

    @property
    def display_name(self) -> str:
        """Get human-readable stage name."""
        names = {
            Stage.SMALL_BATTLEFIELD: "Small Battlefield",
            Stage.BATTLEFIELD: "Battlefield",
            Stage.FINAL_DESTINATION: "Final Destination",
            Stage.TOWN_AND_CITY: "Town and City",
            Stage.SMASHVILLE: "Smashville",
            Stage.HOLLOW_BASTION: "Hollow Bastion",
            Stage.KALOS_POKEMON_LEAGUE: "Kalos Pokemon League",
            Stage.POKEMON_STADIUM_2: "Pokemon Stadium 2",
            Stage.UNKNOWN: "Unknown",
        }
        return names.get(self, self.value)


@dataclass
class StageGeometry:
    """Stage platform and boundary geometry.

    Coordinates are normalized (0-1) relative to the screen.
    """

    left_ledge: tuple[float, float]  # (x, y)
    right_ledge: tuple[float, float]  # (x, y)
    left_blast_zone: float  # x coordinate
    right_blast_zone: float  # x coordinate
    main_stage_y: float  # y coordinate of main platform
    platforms: list[tuple[float, float, float]]  # (x_center, y, width)


# Stage-specific color signatures (approximate HSV ranges)
# These help distinguish stages based on dominant background colors
STAGE_COLOR_SIGNATURES = {
    Stage.SMALL_BATTLEFIELD: {
        "primary_hue_range": (15, 35),  # Orange/warm tones from eclipse
        "secondary_hue_range": (100, 130),  # Blue sky
        "avg_saturation": (40, 80),
    },
    Stage.BATTLEFIELD: {
        "primary_hue_range": (15, 35),  # Similar to Small BF
        "secondary_hue_range": (100, 130),
        "avg_saturation": (40, 80),
    },
    Stage.FINAL_DESTINATION: {
        "primary_hue_range": (100, 140),  # Blue/purple nebula
        "secondary_hue_range": (0, 20),  # Some red accents
        "avg_saturation": (30, 70),
    },
    Stage.TOWN_AND_CITY: {
        "primary_hue_range": (90, 130),  # Blue sky
        "secondary_hue_range": (20, 50),  # Warm building tones
        "avg_saturation": (30, 60),
    },
    Stage.SMASHVILLE: {
        "primary_hue_range": (90, 130),  # Blue sky
        "secondary_hue_range": (30, 60),  # Warm village colors
        "avg_saturation": (40, 70),
    },
    Stage.HOLLOW_BASTION: {
        "primary_hue_range": (120, 160),  # Purple/blue castle
        "secondary_hue_range": (0, 30),  # Dark reds
        "avg_saturation": (20, 50),
    },
    Stage.KALOS_POKEMON_LEAGUE: {
        "primary_hue_range": (90, 120),  # Blue/teal arena
        "secondary_hue_range": (40, 70),  # Gold accents
        "avg_saturation": (50, 80),
    },
    Stage.POKEMON_STADIUM_2: {
        "primary_hue_range": (90, 120),  # Blue stadium
        "secondary_hue_range": (0, 30),  # Red elements
        "avg_saturation": (40, 70),
    },
}

# Stage geometries for game state classification
STAGE_GEOMETRIES = {
    Stage.SMALL_BATTLEFIELD: StageGeometry(
        left_ledge=(0.28, 0.72),
        right_ledge=(0.72, 0.72),
        left_blast_zone=0.05,
        right_blast_zone=0.95,
        main_stage_y=0.72,
        platforms=[
            (0.35, 0.55, 0.15),  # Left platform
            (0.65, 0.55, 0.15),  # Right platform
        ],
    ),
    Stage.BATTLEFIELD: StageGeometry(
        left_ledge=(0.25, 0.72),
        right_ledge=(0.75, 0.72),
        left_blast_zone=0.05,
        right_blast_zone=0.95,
        main_stage_y=0.72,
        platforms=[
            (0.30, 0.55, 0.12),  # Left platform
            (0.50, 0.40, 0.12),  # Top platform
            (0.70, 0.55, 0.12),  # Right platform
        ],
    ),
    Stage.FINAL_DESTINATION: StageGeometry(
        left_ledge=(0.22, 0.72),
        right_ledge=(0.78, 0.72),
        left_blast_zone=0.05,
        right_blast_zone=0.95,
        main_stage_y=0.72,
        platforms=[],  # No platforms
    ),
    Stage.TOWN_AND_CITY: StageGeometry(
        left_ledge=(0.25, 0.72),
        right_ledge=(0.75, 0.72),
        left_blast_zone=0.05,
        right_blast_zone=0.95,
        main_stage_y=0.72,
        platforms=[
            (0.35, 0.50, 0.15),  # Moving platforms (approximate)
            (0.65, 0.50, 0.15),
        ],
    ),
    Stage.SMASHVILLE: StageGeometry(
        left_ledge=(0.26, 0.72),
        right_ledge=(0.74, 0.72),
        left_blast_zone=0.05,
        right_blast_zone=0.95,
        main_stage_y=0.72,
        platforms=[
            (0.50, 0.52, 0.18),  # Single moving platform
        ],
    ),
    Stage.HOLLOW_BASTION: StageGeometry(
        left_ledge=(0.25, 0.72),
        right_ledge=(0.75, 0.72),
        left_blast_zone=0.05,
        right_blast_zone=0.95,
        main_stage_y=0.72,
        platforms=[
            (0.35, 0.52, 0.12),
            (0.65, 0.52, 0.12),
        ],
    ),
    Stage.KALOS_POKEMON_LEAGUE: StageGeometry(
        left_ledge=(0.24, 0.72),
        right_ledge=(0.76, 0.72),
        left_blast_zone=0.05,
        right_blast_zone=0.95,
        main_stage_y=0.72,
        platforms=[
            (0.35, 0.52, 0.12),
            (0.65, 0.52, 0.12),
        ],
    ),
    Stage.POKEMON_STADIUM_2: StageGeometry(
        left_ledge=(0.24, 0.72),
        right_ledge=(0.76, 0.72),
        left_blast_zone=0.05,
        right_blast_zone=0.95,
        main_stage_y=0.72,
        platforms=[
            (0.35, 0.52, 0.12),
            (0.65, 0.52, 0.12),
        ],
    ),
}


def extract_background_region(frame: np.ndarray) -> np.ndarray:
    """Extract the stage region of the frame (excludes HUD areas).

    Uses the top 75% of the frame to capture:
    - Background/sky (top portion)
    - Stage platforms and geometry (middle portion)
    - Avoids the bottom HUD area with damage/stocks

    Args:
        frame: Full video frame (BGR)

    Returns:
        Stage region as numpy array
    """
    h, w = frame.shape[:2]

    # Take top 75% of frame (everything except the HUD at bottom)
    # This includes background, platforms, and stage geometry
    stage_region = frame[: int(0.75 * h), :]

    return stage_region


def compute_color_histogram(region: np.ndarray) -> np.ndarray:
    """Compute a normalized HSV color histogram.

    Args:
        region: BGR image region

    Returns:
        Flattened, normalized histogram
    """
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Compute histogram for H and S channels
    hist = cv2.calcHist(
        [hsv],
        [0, 1],  # Hue and Saturation
        None,
        [50, 60],  # Number of bins
        [0, 180, 0, 256],  # Ranges
    )

    # Normalize
    hist = cv2.normalize(hist, hist).flatten()

    return hist


def compute_dominant_colors(region: np.ndarray, k: int = 3) -> list[tuple[int, int, int]]:
    """Find dominant colors using k-means clustering.

    Args:
        region: BGR image region
        k: Number of clusters

    Returns:
        List of dominant HSV colors as tuples
    """
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Reshape for k-means
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # Run k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(  # type: ignore[call-overload]  # OpenCV stubs reject bestLabels=None
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Sort by frequency
    unique, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)

    dominant = []
    for idx in sorted_indices:
        center = centers[idx]
        dominant.append((int(center[0]), int(center[1]), int(center[2])))

    return dominant


def has_top_center_platform(frame: np.ndarray) -> bool:
    """Check if there's a top center platform (distinguishes Battlefield from Small BF).

    Battlefield has 3 platforms arranged in a triangle:
    - Left platform, Right platform, and TOP CENTER platform

    Small Battlefield has only 2 platforms:
    - Left platform and Right platform (NO top center)

    This function checks for the presence of a platform in the top center area
    using multiple detection methods for robustness.

    Args:
        frame: Full video frame

    Returns:
        True if a top center platform is detected (Battlefield), False otherwise (Small BF)
    """
    h, w = frame.shape[:2]

    # The top center platform on Battlefield is roughly at:
    # - Y position: 28-45% from top (above the side platforms which are at ~45-55%)
    # - X position: 35-65% (center of screen)
    # Small Battlefield has NO platform in this region
    # Using a wider region to catch the platform in various gameplay scenarios

    top_center_region = frame[int(0.28 * h):int(0.45 * h), int(0.35 * w):int(0.65 * w)]

    # Convert to grayscale
    gray = cv2.cvtColor(top_center_region, cv2.COLOR_BGR2GRAY)

    # Method 1: Edge detection with Canny
    edges = cv2.Canny(gray, 30, 100)  # Lower thresholds for more sensitivity

    # Calculate edge density in this region
    edge_density = np.sum(edges > 0) / edges.size

    # Method 2: Check for horizontal lines using Hough transform
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=15
    )

    horizontal_line_count = 0
    total_horizontal_length = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 25:  # Nearly horizontal (more lenient)
                horizontal_line_count += 1
                total_horizontal_length += abs(x2 - x1)

    # Method 3: Check for platform-like solid region (the platform has a distinct color)
    # Look for a horizontal band of similar color in the center

    # The Battlefield platform is typically a golden/tan color
    # Check for consistent horizontal bands by looking at row variance
    row_h, row_w = top_center_region.shape[:2]
    center_strip = gray[row_h // 3 : 2 * row_h // 3, :]  # Middle third vertically

    # Check for horizontal uniformity (platform would create low variance horizontally)
    row_stds = np.std(center_strip, axis=1)
    low_variance_rows = np.sum(row_stds < 30)  # Rows with low variance (uniform color)

    # Method 4: Check brightness contrast (platform is usually brighter than background)
    center_brightness = np.mean(gray[row_h // 3 : 2 * row_h // 3, row_w // 4 : 3 * row_w // 4])
    top_brightness = np.mean(gray[: row_h // 4, :])
    brightness_contrast = center_brightness - top_brightness

    # Scoring system: combine multiple signals
    score = 0.0

    # Edge density contribution (platforms create edges)
    if edge_density > 0.03:
        score += 0.2
    if edge_density > 0.05:
        score += 0.1

    # Horizontal line contribution
    if horizontal_line_count >= 1:
        score += 0.2
    if horizontal_line_count >= 2:
        score += 0.1
    if total_horizontal_length > 50:
        score += 0.1

    # Low variance rows (platform creates uniform horizontal bands)
    if low_variance_rows > 5:
        score += 0.15

    # Brightness contrast (platform is typically brighter than sky above)
    if brightness_contrast > 10:
        score += 0.15

    has_platform = score >= 0.4  # Need at least 40% confidence

    logger.debug(
        f"Top center platform check: edge_density={edge_density:.3f}, "
        f"h_lines={horizontal_line_count}, h_length={total_horizontal_length}, "
        f"low_var_rows={low_variance_rows}, brightness_diff={brightness_contrast:.1f}, "
        f"score={score:.2f}, has_platform={has_platform}"
    )

    return has_platform


def detect_platform_count(frame: np.ndarray) -> int:
    """Estimate the number of platforms visible.

    Uses targeted detection for the top center platform to distinguish
    Battlefield (3 platforms) from Small Battlefield (2 platforms).

    Args:
        frame: Full video frame

    Returns:
        Estimated platform count (2 or 3 for Battlefield variants)
    """
    # The key distinguishing feature is the TOP CENTER platform
    # Battlefield has it, Small Battlefield doesn't
    if has_top_center_platform(frame):
        return 3  # Battlefield
    else:
        return 2  # Small Battlefield


def detect_stage(
    frame: np.ndarray,
    confidence_threshold: float = 0.5,
) -> tuple[Stage, float]:
    """Detect which stage is being played.

    Uses reference image matching with histogram correlation as primary method.
    Falls back to heuristic-based detection if no references are available.

    Args:
        frame: Full video frame (BGR)
        confidence_threshold: Minimum confidence for detection

    Returns:
        Tuple of (detected_stage, confidence)
    """
    background = extract_background_region(frame)
    frame_hist = compute_color_histogram(background)

    # Try reference-based matching first
    references = load_stage_references()
    if references:
        best_stage = Stage.UNKNOWN
        best_score = 0.0

        for stage, ref_hists in references.items():
            # Compare against all reference histograms for this stage
            # Use the best match (handles variable backgrounds)
            for ref_hist in ref_hists:
                # Use histogram correlation (cv2.HISTCMP_CORREL)
                # Returns value between -1 and 1, where 1 is perfect match
                score = cv2.compareHist(frame_hist, ref_hist, cv2.HISTCMP_CORREL)

                if score > best_score:
                    best_score = score
                    best_stage = stage

        if best_score >= confidence_threshold:
            # Special case: Battlefield and Small Battlefield look very similar
            # Use platform counting to disambiguate
            if best_stage in (Stage.BATTLEFIELD, Stage.SMALL_BATTLEFIELD):
                platform_count = detect_platform_count(frame)
                # Battlefield has 3 platforms, Small Battlefield has 2
                if platform_count >= 3:
                    return Stage.BATTLEFIELD, best_score
                elif platform_count == 2:
                    return Stage.SMALL_BATTLEFIELD, best_score
                # If platform count is unclear, use the histogram result
            return best_stage, best_score

    # Fallback to heuristic-based detection
    dominant_colors = compute_dominant_colors(background)
    platform_count = detect_platform_count(frame)

    scores: dict[Stage, float] = {}

    for stage, signature in STAGE_COLOR_SIGNATURES.items():
        score = 0.0

        # Check primary hue
        primary_hue = dominant_colors[0][0] if dominant_colors else 0
        if signature["primary_hue_range"][0] <= primary_hue <= signature["primary_hue_range"][1]:
            score += 0.3

        # Check secondary hue
        if len(dominant_colors) > 1:
            secondary_hue = dominant_colors[1][0]
            if (
                signature["secondary_hue_range"][0]
                <= secondary_hue
                <= signature["secondary_hue_range"][1]
            ):
                score += 0.2

        # Check platform count matches expected
        expected_platforms = len(STAGE_GEOMETRIES[stage].platforms)
        if platform_count == expected_platforms:
            score += 0.3
        elif abs(platform_count - expected_platforms) == 1:
            score += 0.1

        # Saturation check
        avg_sat = dominant_colors[0][1] if dominant_colors else 0
        if signature["avg_saturation"][0] <= avg_sat <= signature["avg_saturation"][1]:
            score += 0.2

        scores[stage] = score

    # Find best match
    if scores:
        best_stage = max(scores, key=lambda s: scores[s])
        best_score = scores[best_stage]

        if best_score >= confidence_threshold:
            return best_stage, best_score

    return Stage.UNKNOWN, 0.0


def _detect_stage_best_match(
    frame: np.ndarray,
    skip_platform_check: bool = False,
) -> tuple[Stage, float, bool]:
    """Detect stage returning best match regardless of confidence threshold.

    Unlike detect_stage(), this always returns the best matching stage
    even if confidence is below threshold. Used for voting across frames.

    Args:
        frame: Full video frame (BGR)
        skip_platform_check: If True, don't disambiguate BF/SBF (for batch voting)

    Returns:
        Tuple of (best_matching_stage, confidence, has_top_platform)
    """
    background = extract_background_region(frame)
    frame_hist = compute_color_histogram(background)

    references = load_stage_references()
    if not references:
        return Stage.UNKNOWN, 0.0, False

    best_stage = Stage.UNKNOWN
    best_score = 0.0

    for stage, ref_hists in references.items():
        for ref_hist in ref_hists:
            score = cv2.compareHist(frame_hist, ref_hist, cv2.HISTCMP_CORREL)
            if score > best_score:
                best_score = score
                best_stage = stage

    # Check for top platform (for voting purposes)
    has_top_platform = False
    if best_stage in (Stage.BATTLEFIELD, Stage.SMALL_BATTLEFIELD):
        has_top_platform = has_top_center_platform(frame)
        if not skip_platform_check:
            if has_top_platform:
                best_stage = Stage.BATTLEFIELD
            else:
                best_stage = Stage.SMALL_BATTLEFIELD

    return best_stage, best_score, has_top_platform


def detect_stage_from_frames(
    frames: list[np.ndarray],
    sample_count: int = 15,
    confidence_threshold: float = 0.3,
) -> tuple[Stage, float]:
    """Detect stage from multiple frames using voting.

    This is more reliable than single-frame detection. It samples multiple
    frames, gets the best match for each (regardless of confidence), and
    votes to determine the final stage.

    For Battlefield vs Small Battlefield disambiguation, it uses multi-frame
    voting on platform detection since this is more reliable than single-frame.

    Prioritizes early game frames (after "GO!") where the stage is shown most
    clearly before intense gameplay obscures the view.

    Args:
        frames: List of video frames
        sample_count: Number of frames to sample
        confidence_threshold: Minimum average confidence for valid detection

    Returns:
        Tuple of (detected_stage, confidence)
    """
    if not frames:
        return Stage.UNKNOWN, 0.0

    # Sample frames with bias toward early game (stages are clearest right after GO!)
    # Use positions: 2%, 5%, 8%, 10%, 15%, 20%, 30%, 40%, 50%, 60%, 70%, 80%
    # Heavy early sampling catches the stage before gameplay obscures it
    sample_positions = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    sampled_indices = [min(int(pos * len(frames)), len(frames) - 1) for pos in sample_positions]
    sampled = [frames[i] for i in sampled_indices[:sample_count]]

    logger.debug(f"Stage detection: sampling {len(sampled)} frames from {len(frames)} total")

    # Collect votes - always get best match, don't threshold individual frames
    # Also collect platform detection votes for BF/SBF disambiguation
    stage_votes: dict[Stage, list[float]] = {}
    platform_votes: list[bool] = []  # True = has top platform (Battlefield)
    bf_like_count = 0  # Count of frames that matched BF or SBF

    for frame in sampled:
        stage, confidence, has_top_platform = _detect_stage_best_match(frame, skip_platform_check=True)
        if stage not in stage_votes:
            stage_votes[stage] = []
        stage_votes[stage].append(confidence)

        # Track platform detection for BF/SBF voting
        if stage in (Stage.BATTLEFIELD, Stage.SMALL_BATTLEFIELD):
            bf_like_count += 1
            platform_votes.append(has_top_platform)

    # Find stage with most votes (excluding UNKNOWN)
    if not stage_votes:
        return Stage.UNKNOWN, 0.0

    # Log vote distribution
    vote_summary = {s.value: len(v) for s, v in stage_votes.items()}
    logger.debug(f"Stage voting: {vote_summary}")

    # Filter out UNKNOWN votes if there are other options
    non_unknown_votes: dict[Stage, list[float]] = {s: v for s, v in stage_votes.items() if s != Stage.UNKNOWN}
    if non_unknown_votes:
        stage_votes = non_unknown_votes

    # For BF-like stages, use platform voting to disambiguate
    # BUT only if BF+SBF votes are a strong majority (> 75% of total votes)
    # This prevents overriding when Pokemon Stadium 2 or other stages have significant votes
    bf_votes = stage_votes.get(Stage.BATTLEFIELD, [])
    sbf_votes = stage_votes.get(Stage.SMALL_BATTLEFIELD, [])
    combined_bf_votes = bf_votes + sbf_votes

    total_votes = sum(len(v) for v in stage_votes.values())

    if combined_bf_votes and total_votes > 0:
        bf_like_ratio = len(combined_bf_votes) / total_votes

        # Only use platform disambiguation if BF+SBF have > 75% of votes
        # This ensures we don't override when other stages (like PS2) have significant matches
        if bf_like_ratio > 0.75:
            # Use platform voting to decide BF vs SBF
            # Majority vote on top platform detection
            if platform_votes:
                platform_present_count = sum(platform_votes)
                platform_absent_count = len(platform_votes) - platform_present_count

                logger.debug(
                    f"Platform voting: {platform_present_count} with top platform, "
                    f"{platform_absent_count} without (BF if >= 40% have platform)"
                )

                # Use 40% threshold (Battlefield if at least 40% of frames show top platform)
                # This is lenient because the platform may be obscured during gameplay
                if platform_present_count >= len(platform_votes) * 0.4:
                    best_stage = Stage.BATTLEFIELD
                else:
                    best_stage = Stage.SMALL_BATTLEFIELD

                avg_confidence = sum(combined_bf_votes) / len(combined_bf_votes)
                if avg_confidence >= confidence_threshold:
                    return best_stage, avg_confidence
        else:
            logger.debug(
                f"BF-like ratio {bf_like_ratio:.2%} < 75%, using standard voting"
            )

    best_stage = max(stage_votes, key=lambda s: len(stage_votes[s]))
    avg_confidence = sum(stage_votes[best_stage]) / len(stage_votes[best_stage])

    # Only return UNKNOWN if confidence is very low
    if avg_confidence < confidence_threshold:
        return Stage.UNKNOWN, avg_confidence

    return best_stage, avg_confidence


def get_stage_geometry(stage: Stage) -> StageGeometry | None:
    """Get the geometry for a stage.

    Args:
        stage: Stage enum value

    Returns:
        StageGeometry or None if unknown
    """
    return STAGE_GEOMETRIES.get(stage)
