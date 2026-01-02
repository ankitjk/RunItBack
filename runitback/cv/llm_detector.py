"""LLM-based detection using Gemini Vision API.

Uses Google's Gemini Flash model for accurate character and stage identification
from SSBU gameplay screenshots. This is more reliable than template matching
for characters with complex visual appearances or varying skins, and more
accurate than histogram matching for stage detection.
"""

import base64
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Cache directory for LLM results
CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "llm_results"

# All SSBU characters for validation
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

# Normalized character names for matching LLM output
NORMALIZED_CHARACTERS = {
    name.lower().replace(".", "").replace("&", "and").replace("/", ""): name
    for name in CHARACTERS
}

# Common variations the LLM might output
CHARACTER_ALIASES = {
    "game and watch": "Mr. Game & Watch",
    "game & watch": "Mr. Game & Watch",
    "mr game and watch": "Mr. Game & Watch",
    "rob": "R.O.B.",
    "r.o.b": "R.O.B.",
    "pacman": "Pac-Man",
    "pac man": "Pac-Man",
    "dedede": "King Dedede",
    "k rool": "King K. Rool",
    "king k rool": "King K. Rool",
    "dk": "Donkey Kong",
    "zss": "Zero Suit Samus",
    "zero suit": "Zero Suit Samus",
    "pokemon trainer": "Pokemon Trainer",
    "pt": "Pokemon Trainer",
    "squirtle": "Pokemon Trainer",
    "ivysaur": "Pokemon Trainer",
    "charizard": "Pokemon Trainer",
    "rosalina": "Rosalina & Luma",
    "rosa": "Rosalina & Luma",
    "banjo": "Banjo & Kazooie",
    "banjo kazooie": "Banjo & Kazooie",
    "pyra": "Pyra/Mythra",
    "mythra": "Pyra/Mythra",
    "aegis": "Pyra/Mythra",
    "wii fit": "Wii Fit Trainer",
    "wft": "Wii Fit Trainer",
    "captain": "Captain Falcon",
    "falcon": "Captain Falcon",
    "dr mario": "Dr. Mario",
    "doctor mario": "Dr. Mario",
    "mii fighter": "Mii Brawler",
    "ice climber": "Ice Climbers",
    "nana": "Ice Climbers",
    "popo": "Ice Climbers",
    "bowser junior": "Bowser Jr.",
    "bowser jr": "Bowser Jr.",
    "koopaling": "Bowser Jr.",
    "larry": "Bowser Jr.",
    "ludwig": "Bowser Jr.",
    "lemmy": "Bowser Jr.",
    "roy koopa": "Bowser Jr.",
    "wendy": "Bowser Jr.",
    "iggy": "Bowser Jr.",
    "morton": "Bowser Jr.",
    "piranha": "Piranha Plant",
    "plant": "Piranha Plant",
    "young link": "Young Link",
    "toon link": "Toon Link",
    "botw link": "Link",
    "totk link": "Link",
}


@dataclass
class GeminiDetection:
    """Result of Gemini character detection."""
    player1: str
    player2: str
    confidence: str  # "high", "medium", "low"
    raw_response: str


def normalize_character_name(name: str) -> str:
    """Normalize a character name from LLM output to standard format.

    Args:
        name: Raw character name from LLM

    Returns:
        Standardized character name or "Unknown"
    """
    if not name:
        return "Unknown"

    # Clean up the name
    cleaned = name.strip().lower()

    # Check aliases first
    if cleaned in CHARACTER_ALIASES:
        return CHARACTER_ALIASES[cleaned]

    # Check normalized names
    normalized = cleaned.replace(".", "").replace("&", "and").replace("/", "")
    if normalized in NORMALIZED_CHARACTERS:
        return NORMALIZED_CHARACTERS[normalized]

    # Try partial matching for common cases
    for norm, full in NORMALIZED_CHARACTERS.items():
        if norm in normalized or normalized in norm:
            return full

    logger.warning(f"Could not normalize character name: {name}")
    return "Unknown"


def get_frame_hash(frame: np.ndarray) -> str:
    """Compute a hash of a frame for caching.

    Args:
        frame: BGR image

    Returns:
        SHA256 hash of the frame
    """
    # Resize to small size for faster hashing
    small = cv2.resize(frame, (64, 64))
    return hashlib.sha256(small.tobytes()).hexdigest()[:16]


def get_cached_result(frame_hash: str) -> Optional[GeminiDetection]:
    """Check if we have a cached result for this frame.

    Args:
        frame_hash: Hash of the frame

    Returns:
        Cached GeminiDetection or None
    """
    cache_file = CACHE_DIR / f"{frame_hash}.txt"
    if cache_file.exists():
        try:
            lines = cache_file.read_text().strip().split("\n")
            if len(lines) >= 4:
                return GeminiDetection(
                    player1=lines[0],
                    player2=lines[1],
                    confidence=lines[2],
                    raw_response="\n".join(lines[3:]),
                )
        except Exception as e:
            logger.debug(f"Failed to read cache: {e}")
    return None


def save_cached_result(frame_hash: str, result: GeminiDetection) -> None:
    """Save a detection result to cache.

    Args:
        frame_hash: Hash of the frame
        result: Detection result to cache
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{frame_hash}.txt"
    try:
        cache_file.write_text(
            f"{result.player1}\n{result.player2}\n{result.confidence}\n{result.raw_response}"
        )
    except Exception as e:
        logger.debug(f"Failed to write cache: {e}")


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert a frame to base64-encoded JPEG.

    Args:
        frame: BGR image

    Returns:
        Base64-encoded JPEG string
    """
    # Encode as JPEG (smaller than PNG, good enough quality)
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def detect_characters_with_gemini(
    frame: np.ndarray,
    api_key: Optional[str] = None,
    use_cache: bool = True,
) -> GeminiDetection:
    """Detect characters in a SSBU gameplay frame using Gemini Vision.

    Args:
        frame: BGR gameplay frame
        api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        use_cache: Whether to use cached results

    Returns:
        GeminiDetection with player1, player2 character names
    """
    # Check cache first
    if use_cache:
        frame_hash = get_frame_hash(frame)
        cached = get_cached_result(frame_hash)
        if cached:
            logger.debug(f"Using cached result for frame {frame_hash}")
            return cached

    # Get API key
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("No Gemini API key provided. Set GEMINI_API_KEY environment variable.")
        return GeminiDetection(
            player1="Unknown",
            player2="Unknown",
            confidence="low",
            raw_response="No API key",
        )

    try:
        import google.generativeai as genai
    except ImportError:
        logger.error("google-generativeai package not installed. Run: pip install google-generativeai")
        return GeminiDetection(
            player1="Unknown",
            player2="Unknown",
            confidence="low",
            raw_response="Package not installed",
        )

    # Configure Gemini
    genai.configure(api_key=api_key)

    # Use Gemini 2.5 Flash for speed and cost efficiency
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Prepare prompt - improved to help distinguish similar characters
    prompt = """This is a screenshot from Super Smash Bros. Ultimate gameplay.

Identify the two characters being played. PRIORITIZE the HUD elements:
1. **Character portraits in the HUD** (bottom left/right corners next to damage %) - These show the DEFAULT character icon regardless of skin
2. **Stock icons** (small heads showing lives remaining) - Also show default character appearance
3. The in-game character model (LEAST reliable due to alternate skins)

CRITICAL: Players can use alternate color skins that drastically change appearance:
- Mario can wear GREEN (looks like Luigi) or other colors - check the HUD portrait, Mario has an "M" on his cap
- Luigi wears an "L" on his cap and has a thinner, taller build
- Characters keep the same body shape/silhouette regardless of color skin

Distinguishing similar characters (use body shape and HUD, NOT color):
- Ken (martial arts gi, spiky blonde hair by default) vs Terry (jacket, baseball cap) - DIFFERENT body posture and outfit shape
- Ken vs Ryu (Ryu has headband, different hair, bulkier) - different build
- Marth vs Lucina - different tiara/mask, cape details
- Pit vs Dark Pit - wing shape is similar, check HUD portrait
- Samus vs Dark Samus - different stance/posture
- Peach vs Daisy - check HUD portrait for face differences

Respond in this exact format:
Player 1 (left side): [character name]
Player 2 (right side): [character name]
Confidence: [high/medium/low]

Use official character names (e.g., "King Dedede", "Mr. Game & Watch", "Ken", "Terry", "Mario", "Luigi").
IMPORTANT: Do NOT confuse Mario in a green skin with Luigi - check the HUD portrait and cap letter."""

    # Convert frame to image data
    image_data = frame_to_base64(frame)

    try:
        # Create image part for the API
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_data,
        }

        # Generate response
        response = model.generate_content([prompt, image_part])
        raw_text = response.text

        # Parse response
        player1 = "Unknown"
        player2 = "Unknown"
        confidence = "low"

        for line in raw_text.split("\n"):
            line_lower = line.lower().strip()
            if "player 1" in line_lower and ":" in line:
                player1 = line.split(":", 1)[1].strip()
            elif "player 2" in line_lower and ":" in line:
                player2 = line.split(":", 1)[1].strip()
            elif "confidence" in line_lower and ":" in line:
                conf = line.split(":", 1)[1].strip().lower()
                if conf in ["high", "medium", "low"]:
                    confidence = conf

        # Normalize character names
        player1 = normalize_character_name(player1)
        player2 = normalize_character_name(player2)

        result = GeminiDetection(
            player1=player1,
            player2=player2,
            confidence=confidence,
            raw_response=raw_text,
        )

        # Cache the result
        if use_cache:
            save_cached_result(frame_hash, result)

        logger.info(f"Gemini detected: P1={player1}, P2={player2} ({confidence} confidence)")
        return result

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return GeminiDetection(
            player1="Unknown",
            player2="Unknown",
            confidence="low",
            raw_response=f"Error: {e}",
        )


def detect_characters_from_frames_gemini(
    frames: list[np.ndarray],
    sample_positions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90),
    api_key: Optional[str] = None,
    p1_hints: Optional[list[str]] = None,
    p2_hints: Optional[list[str]] = None,
) -> tuple[str, str]:
    """Detect characters from multiple frames using Gemini, with voting.

    Samples frames at specified positions and uses voting to determine
    the most likely characters. Uses 7 sample positions with emphasis on
    early-game frames (5%, 10%, 20%) where character portraits in the HUD
    are typically clearest before intense action obscures the view.

    Character hints from video title can be provided to help resolve
    ambiguous detections (e.g., when Gemini can't distinguish Dark Pit from Pit).

    Args:
        frames: List of gameplay frames
        sample_positions: Positions to sample (as fractions 0-1)
        api_key: Gemini API key
        p1_hints: Character names that might be player 1 (from video title)
        p2_hints: Character names that might be player 2 (from video title)

    Returns:
        Tuple of (player1_character, player2_character)
    """
    if not frames:
        return ("Unknown", "Unknown")

    # Vote counts for each character (track individual detections for logging)
    p1_votes: dict[str, int] = {}
    p2_votes: dict[str, int] = {}
    p1_detections: list[tuple[str, str, float]] = []  # (character, confidence, position)
    p2_detections: list[tuple[str, str, float]] = []

    for pos in sample_positions:
        idx = int(pos * (len(frames) - 1))
        frame = frames[idx]

        result = detect_characters_with_gemini(frame, api_key=api_key)

        # Count votes (weight by confidence)
        weight = {"high": 3, "medium": 2, "low": 1}.get(result.confidence, 1)

        if result.player1 != "Unknown":
            p1_votes[result.player1] = p1_votes.get(result.player1, 0) + weight
            p1_detections.append((result.player1, result.confidence, pos))
        if result.player2 != "Unknown":
            p2_votes[result.player2] = p2_votes.get(result.player2, 0) + weight
            p2_detections.append((result.player2, result.confidence, pos))

    # Log voting details for debugging
    if p1_detections:
        logger.debug(f"P1 detections: {p1_detections}")
        logger.debug(f"P1 vote totals: {p1_votes}")
    if p2_detections:
        logger.debug(f"P2 detections: {p2_detections}")
        logger.debug(f"P2 vote totals: {p2_votes}")

    # Get winners
    player1 = max(p1_votes, key=p1_votes.get) if p1_votes else "Unknown"
    player2 = max(p2_votes, key=p2_votes.get) if p2_votes else "Unknown"

    # Use hints to resolve ambiguous detections
    # Confusion pairs where hints can help
    CONFUSION_PAIRS = {
        frozenset({"Mario", "Luigi"}),
        frozenset({"Ken", "Terry"}),
        frozenset({"Ken", "Ryu"}),
        frozenset({"Marth", "Lucina"}),
        frozenset({"Pit", "Dark Pit"}),
        frozenset({"Samus", "Dark Samus"}),
        frozenset({"Peach", "Daisy"}),
        frozenset({"Simon", "Richter"}),
        frozenset({"Roy", "Chrom"}),
    }

    def apply_hint(winner: str, votes: dict[str, int], hints: list[str] | None) -> str:
        """Apply hint if detection is ambiguous and hint matches a candidate."""
        if not hints or not votes:
            return winner

        total = sum(votes.values())
        winner_pct = votes.get(winner, 0) / total if total > 0 else 0

        # Only apply hint if detection is not confident (< 75%)
        if winner_pct >= 0.75:
            return winner

        # Check if winner is part of a confusion pair
        for pair in CONFUSION_PAIRS:
            if winner in pair:
                # Check if any hint matches the other character in the pair
                other_chars = pair - {winner}
                other_char = next(iter(other_chars))  # Get the other character
                if other_char in hints and other_char in votes:
                    # Hint suggests the other character - switch if it has reasonable votes
                    other_pct = votes.get(other_char, 0) / total if total > 0 else 0
                    if other_pct >= 0.2:  # At least 20% of votes
                        logger.info(f"Applying title hint: {winner} -> {other_char} (hint matched, {other_pct:.0%} votes)")
                        return other_char
                # Check if hint matches winner - boost confidence
                if winner in hints:
                    logger.debug(f"Hint confirms winner: {winner}")
        return winner

    def apply_exclusive_hint(
        winner: str,
        votes: dict[str, int],
        exclusive_hint: str | None,
    ) -> str:
        """Apply exclusive hint when one hint is confirmed for the other player.

        When one player's detection matches a hint, the remaining hint becomes
        exclusive to the other player. This handles cases like Mario in a green
        skin being detected as Luigi - if P2 matches "Ness", then "Mario" from
        the title exclusively applies to P1.

        Args:
            winner: Current winner from voting
            votes: Vote counts per character
            exclusive_hint: The exclusive hint for this player (or None)

        Returns:
            The final character (may override winner if exclusive hint applies)
        """
        if not exclusive_hint or not votes:
            return winner

        # If winner already matches the exclusive hint, confirm it
        if winner == exclusive_hint:
            logger.debug(f"Exclusive hint confirms winner: {winner}")
            return winner

        # Check if winner is part of a confusion pair with the exclusive hint
        for pair in CONFUSION_PAIRS:
            if winner in pair and exclusive_hint in pair:
                # Winner is in a confusion pair with the exclusive hint
                # The exclusive hint overrides detection (e.g., Luigi -> Mario)
                if exclusive_hint in votes:
                    other_pct = votes.get(exclusive_hint, 0) / sum(votes.values())
                    logger.info(
                        f"Applying exclusive hint: {winner} -> {exclusive_hint} "
                        f"(other player confirmed different character, {other_pct:.0%} votes)"
                    )
                    return exclusive_hint
                else:
                    # Exclusive hint not in votes at all, but still override
                    # because the hint came from the title and should be trusted
                    logger.info(
                        f"Applying exclusive hint: {winner} -> {exclusive_hint} "
                        f"(other player confirmed different character, no votes but title trusted)"
                    )
                    return exclusive_hint

        return winner

    # First pass: apply standard hints
    player1 = apply_hint(player1, p1_votes, p1_hints)
    player2 = apply_hint(player2, p2_votes, p2_hints)

    # Second pass: apply exclusive hints when one player's detection confirms a hint
    # This handles cases like Mario (green skin) detected as Luigi
    if p1_hints and p2_hints:
        # Get all hints (they should be the same since extract_characters_from_title
        # returns the same list for both players)
        all_hints = set(p1_hints) | set(p2_hints)

        # Check if P2's detection matches a hint
        if player2 in all_hints and len(all_hints) >= 2:
            # P2 matched a hint - remaining hints are exclusive to P1
            remaining_hints = all_hints - {player2}
            for exclusive in remaining_hints:
                # Apply the exclusive hint to P1
                player1 = apply_exclusive_hint(player1, p1_votes, exclusive)

        # Check if P1's detection matches a hint
        if player1 in all_hints and len(all_hints) >= 2:
            # P1 matched a hint - remaining hints are exclusive to P2
            remaining_hints = all_hints - {player1}
            for exclusive in remaining_hints:
                # Apply the exclusive hint to P2
                player2 = apply_exclusive_hint(player2, p2_votes, exclusive)

    # Log final decision with vote breakdown
    if p1_votes:
        p1_total = sum(p1_votes.values())
        p1_winner_pct = p1_votes.get(player1, 0) / p1_total * 100 if p1_total > 0 else 0
        logger.info(f"P1 winner: {player1} ({p1_winner_pct:.0f}% of votes, {p1_votes.get(player1, 0)}/{p1_total})")
    if p2_votes:
        p2_total = sum(p2_votes.values())
        p2_winner_pct = p2_votes.get(player2, 0) / p2_total * 100 if p2_total > 0 else 0
        logger.info(f"P2 winner: {player2} ({p2_winner_pct:.0f}% of votes, {p2_votes.get(player2, 0)}/{p2_total})")

    return (player1, player2)


# Tournament-legal stages (for validation)
STAGES = [
    "Small Battlefield",
    "Battlefield",
    "Final Destination",
    "Town and City",
    "Smashville",
    "Hollow Bastion",
    "Kalos Pokemon League",
    "Pokemon Stadium 2",
]

# Normalized stage names for matching LLM output
NORMALIZED_STAGES = {
    "small battlefield": "Small Battlefield",
    "small bf": "Small Battlefield",
    "sbf": "Small Battlefield",
    "battlefield": "Battlefield",
    "bf": "Battlefield",
    "final destination": "Final Destination",
    "fd": "Final Destination",
    "town and city": "Town and City",
    "t&c": "Town and City",
    "smashville": "Smashville",
    "sv": "Smashville",
    "hollow bastion": "Hollow Bastion",
    "hb": "Hollow Bastion",
    "kalos pokemon league": "Kalos Pokemon League",
    "kalos": "Kalos Pokemon League",
    "kpl": "Kalos Pokemon League",
    "pokemon stadium 2": "Pokemon Stadium 2",
    "ps2": "Pokemon Stadium 2",
    "stadium 2": "Pokemon Stadium 2",
    "pokemon stadium": "Pokemon Stadium 2",
}


@dataclass
class GeminiStageDetection:
    """Result of Gemini stage detection."""
    stage: str
    confidence: str  # "high", "medium", "low"
    raw_response: str


def normalize_stage_name(name: str) -> str:
    """Normalize a stage name from LLM output to standard format.

    Args:
        name: Raw stage name from LLM

    Returns:
        Standardized stage name or "Unknown"
    """
    if not name:
        return "Unknown"

    # Clean up the name
    cleaned = name.strip().lower()

    # Check normalized names
    if cleaned in NORMALIZED_STAGES:
        return NORMALIZED_STAGES[cleaned]

    # Try partial matching
    for norm, full in NORMALIZED_STAGES.items():
        if norm in cleaned or cleaned in norm:
            return full

    # Direct match against stage list
    for stage in STAGES:
        if stage.lower() == cleaned:
            return stage

    logger.warning(f"Could not normalize stage name: {name}")
    return "Unknown"


# Cache directory for stage detection results
STAGE_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "stage_results"


def get_cached_stage_result(frame_hash: str) -> Optional[GeminiStageDetection]:
    """Check if we have a cached stage result for this frame."""
    cache_file = STAGE_CACHE_DIR / f"{frame_hash}.txt"
    if cache_file.exists():
        try:
            lines = cache_file.read_text().strip().split("\n")
            if len(lines) >= 3:
                return GeminiStageDetection(
                    stage=lines[0],
                    confidence=lines[1],
                    raw_response="\n".join(lines[2:]),
                )
        except Exception as e:
            logger.debug(f"Failed to read stage cache: {e}")
    return None


def save_cached_stage_result(frame_hash: str, result: GeminiStageDetection) -> None:
    """Save a stage detection result to cache."""
    STAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = STAGE_CACHE_DIR / f"{frame_hash}.txt"
    try:
        cache_file.write_text(f"{result.stage}\n{result.confidence}\n{result.raw_response}")
    except Exception as e:
        logger.debug(f"Failed to write stage cache: {e}")


def detect_stage_with_gemini(
    frame: np.ndarray,
    api_key: Optional[str] = None,
    use_cache: bool = True,
) -> GeminiStageDetection:
    """Detect the stage in a SSBU gameplay frame using Gemini Vision.

    Args:
        frame: BGR gameplay frame
        api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        use_cache: Whether to use cached results

    Returns:
        GeminiStageDetection with stage name
    """
    # Check cache first
    frame_hash = get_frame_hash(frame)
    if use_cache:
        cached = get_cached_stage_result(frame_hash)
        if cached:
            logger.debug(f"Using cached stage result for frame {frame_hash}")
            return cached

    # Get API key
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("No Gemini API key provided for stage detection.")
        return GeminiStageDetection(
            stage="Unknown",
            confidence="low",
            raw_response="No API key",
        )

    try:
        import google.generativeai as genai
    except ImportError:
        logger.error("google-generativeai package not installed.")
        return GeminiStageDetection(
            stage="Unknown",
            confidence="low",
            raw_response="Package not installed",
        )

    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Prepare prompt for stage detection
    prompt = """This is a screenshot from Super Smash Bros. Ultimate gameplay.

Identify which tournament-legal stage is being played. The options are:

1. **Small Battlefield** - Outdoor stage with sky background. Has exactly TWO thin tan/beige floating platforms positioned at similar heights (one left, one right), both roughly 20-25% from the TOP of the screen. NO platform exists ABOVE these or BETWEEN them. The space ABOVE the side platforms is EMPTY sky.

2. **Battlefield** - Same outdoor sky background as Small Battlefield. Has THREE floating platforms:
   - Two LOWER platforms (left and right) at around 35-40% from top
   - One UPPER/TOP platform in the CENTER, positioned ABOVE the lower platforms at around 20-25% from top
   - The platforms form a TRIANGLE pattern when viewed from afar

3. **Final Destination** - No floating platforms at all. Flat main stage. Dark cosmic/space background.

4. **Town and City** - URBAN cityscape. Background has TALL BUILDINGS with lit windows (especially at night), modern architecture. The stage floor is a FLAT platform that moves between locations. May show rooftops or city streets.

5. **Smashville** - RURAL Animal Crossing village. GRASSY main stage floor with visible grass texture. A SINGLE balloon-suspended platform floats above. Cozy wooden buildings, trees, and Animal Crossing NPCs in background.

6. **Hollow Bastion** - INDOOR purple/pink castle interior. Gothic stone architecture.

7. **Kalos Pokemon League** - Pokemon arena with blue and gold colors.

8. **Pokemon Stadium 2** - INDOOR stadium with large VIDEO SCREENS in background.

CRITICAL - BATTLEFIELD vs SMALL BATTLEFIELD DETECTION:
Count the floating platforms ABOVE the main stage floor:
- If you count EXACTLY 2 platforms at similar height with EMPTY space above them = Small Battlefield
- If you count 3 platforms with one ABOVE/BETWEEN the side platforms = Battlefield
- Look at the upper-middle area of the screen (around 20-30% from top center) - is there a platform there?
  - YES, there's a platform in upper-center = Battlefield
  - NO, just sky/empty space between the side platforms = Small Battlefield

Respond ONLY with:
Stage: [stage name]
Confidence: [high/medium/low]"""

    # Convert frame to image data
    image_data = frame_to_base64(frame)

    try:
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_data,
        }

        response = model.generate_content([prompt, image_part])
        raw_text = response.text

        # Parse response
        stage = "Unknown"
        confidence = "low"

        for line in raw_text.split("\n"):
            line_lower = line.lower().strip()
            if "stage:" in line_lower:
                stage = line.split(":", 1)[1].strip()
            elif "confidence:" in line_lower:
                conf = line.split(":", 1)[1].strip().lower()
                if conf in ["high", "medium", "low"]:
                    confidence = conf

        # Normalize stage name
        stage = normalize_stage_name(stage)

        result = GeminiStageDetection(
            stage=stage,
            confidence=confidence,
            raw_response=raw_text,
        )

        # Cache the result
        if use_cache:
            save_cached_stage_result(frame_hash, result)

        logger.info(f"Gemini detected stage: {stage} ({confidence} confidence)")
        return result

    except Exception as e:
        logger.error(f"Gemini stage detection error: {e}")
        return GeminiStageDetection(
            stage="Unknown",
            confidence="low",
            raw_response=f"Error: {e}",
        )


@dataclass
class GeminiStockCounterDetection:
    """Result of Gemini stock counter detection."""
    detected: bool  # Whether a stock counter splash was visible
    p1_stocks: Optional[int]  # Player 1's stock count (0-3)
    p2_stocks: Optional[int]  # Player 2's stock count (0-3)
    is_game_end: bool  # True if "GAME!" screen was shown
    confidence: str  # "high", "medium", "low"
    raw_response: str


# Cache directory for stock counter results
STOCK_COUNTER_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "stock_counter_results"


def get_cached_stock_counter_result(frame_hash: str) -> Optional[GeminiStockCounterDetection]:
    """Check if we have a cached stock counter result for this frame."""
    cache_file = STOCK_COUNTER_CACHE_DIR / f"{frame_hash}.txt"
    if cache_file.exists():
        try:
            lines = cache_file.read_text().strip().split("\n")
            if len(lines) >= 6:
                return GeminiStockCounterDetection(
                    detected=lines[0] == "True",
                    p1_stocks=int(lines[1]) if lines[1] != "None" else None,
                    p2_stocks=int(lines[2]) if lines[2] != "None" else None,
                    is_game_end=lines[3] == "True",
                    confidence=lines[4],
                    raw_response="\n".join(lines[5:]),
                )
        except Exception as e:
            logger.debug(f"Failed to read stock counter cache: {e}")
    return None


def save_cached_stock_counter_result(frame_hash: str, result: GeminiStockCounterDetection) -> None:
    """Save a stock counter detection result to cache."""
    STOCK_COUNTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = STOCK_COUNTER_CACHE_DIR / f"{frame_hash}.txt"
    try:
        cache_file.write_text(
            f"{result.detected}\n{result.p1_stocks}\n{result.p2_stocks}\n"
            f"{result.is_game_end}\n{result.confidence}\n{result.raw_response}"
        )
    except Exception as e:
        logger.debug(f"Failed to write stock counter cache: {e}")


def detect_stock_counter_with_gemini(
    frame: np.ndarray,
    api_key: Optional[str] = None,
    use_cache: bool = True,
) -> GeminiStockCounterDetection:
    """Detect stock counter splash screen using Gemini Vision.

    The stock counter shows "X - Y" in the center of the screen after a kill,
    where X is P1's remaining stocks and Y is P2's remaining stocks.
    For the final kill, "GAME!" appears instead.

    Args:
        frame: BGR gameplay frame
        api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        use_cache: Whether to use cached results

    Returns:
        GeminiStockCounterDetection with stock counts
    """
    # Check cache first
    frame_hash = get_frame_hash(frame)
    if use_cache:
        cached = get_cached_stock_counter_result(frame_hash)
        if cached:
            logger.debug(f"Using cached stock counter result for frame {frame_hash}")
            return cached

    # Get API key
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("No Gemini API key provided for stock counter detection.")
        return GeminiStockCounterDetection(
            detected=False,
            p1_stocks=None,
            p2_stocks=None,
            is_game_end=False,
            confidence="low",
            raw_response="No API key",
        )

    try:
        import google.generativeai as genai
    except ImportError:
        logger.error("google-generativeai package not installed.")
        return GeminiStockCounterDetection(
            detected=False,
            p1_stocks=None,
            p2_stocks=None,
            is_game_end=False,
            confidence="low",
            raw_response="Package not installed",
        )

    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Prepare prompt for stock counter detection
    prompt = """This is a screenshot from Super Smash Bros. Ultimate.

Is there a STOCK COUNTER splash screen visible? This appears after a player loses a stock (life).

The stock counter shows:
- Large numbers in the CENTER of the screen in format "X - Y"
- X = Player 1's remaining stocks (left side player)
- Y = Player 2's remaining stocks (right side player)
- Both numbers are typically 0, 1, 2, or 3

OR it may show:
- "GAME!" text if the match just ended (one player lost their last stock)

If NO stock counter is visible (just normal gameplay), respond:
Visible: No

If a stock counter IS visible, respond:
Visible: Yes
P1 Stocks: [0-3]
P2 Stocks: [0-3]
Confidence: [high/medium/low]

If "GAME!" is shown, respond:
Visible: Yes
Game End: Yes
Confidence: [high/medium/low]"""

    # Convert frame to image data
    image_data = frame_to_base64(frame)

    try:
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_data,
        }

        response = model.generate_content([prompt, image_part])
        raw_text = response.text

        # Parse response
        detected = False
        p1_stocks = None
        p2_stocks = None
        is_game_end = False
        confidence = "low"

        for line in raw_text.split("\n"):
            line_lower = line.lower().strip()
            if "visible:" in line_lower:
                detected = "yes" in line_lower
            elif "p1 stocks:" in line_lower:
                try:
                    val = line.split(":", 1)[1].strip()
                    p1_stocks = int(val)
                except (ValueError, IndexError):
                    pass
            elif "p2 stocks:" in line_lower:
                try:
                    val = line.split(":", 1)[1].strip()
                    p2_stocks = int(val)
                except (ValueError, IndexError):
                    pass
            elif "game end:" in line_lower:
                is_game_end = "yes" in line_lower
            elif "confidence:" in line_lower:
                conf = line.split(":", 1)[1].strip().lower()
                if conf in ["high", "medium", "low"]:
                    confidence = conf

        result = GeminiStockCounterDetection(
            detected=detected,
            p1_stocks=p1_stocks,
            p2_stocks=p2_stocks,
            is_game_end=is_game_end,
            confidence=confidence,
            raw_response=raw_text,
        )

        # Cache the result
        if use_cache:
            save_cached_stock_counter_result(frame_hash, result)

        if detected:
            if is_game_end:
                logger.info(f"Gemini detected GAME! screen ({confidence} confidence)")
            else:
                logger.info(f"Gemini detected stock counter: P1={p1_stocks}, P2={p2_stocks} ({confidence} confidence)")
        else:
            logger.debug("Gemini: No stock counter visible")

        return result

    except Exception as e:
        logger.error(f"Gemini stock counter detection error: {e}")
        return GeminiStockCounterDetection(
            detected=False,
            p1_stocks=None,
            p2_stocks=None,
            is_game_end=False,
            confidence="low",
            raw_response=f"Error: {e}",
        )


def detect_stage_from_frames_gemini(
    frames: list[np.ndarray],
    sample_positions: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60),
    api_key: Optional[str] = None,
) -> str:
    """Detect stage from multiple frames using Gemini, with voting.

    Samples frames from early-to-mid game where the stage background is clearly
    visible. Avoids the very first frames (0-10%) which may show VS screens,
    loading screens, or countdown overlays that obscure the stage.

    For Battlefield vs Small Battlefield disambiguation, uses CV-based platform
    detection as a tiebreaker since Gemini often confuses these visually similar stages.

    Args:
        frames: List of gameplay frames
        sample_positions: Positions to sample (as fractions 0-1), starting at 10%
            to skip transition screens
        api_key: Gemini API key

    Returns:
        Detected stage name (or "Unknown")
    """
    if not frames:
        return "Unknown"

    # Vote counts for each stage
    stage_votes: dict[str, int] = {}
    detections: list[tuple[str, str, float]] = []  # (stage, confidence, position)

    for pos in sample_positions:
        idx = int(pos * (len(frames) - 1))
        frame = frames[idx]

        result = detect_stage_with_gemini(frame, api_key=api_key)

        # Count votes (weight by confidence)
        weight = {"high": 3, "medium": 2, "low": 1}.get(result.confidence, 1)

        if result.stage != "Unknown":
            stage_votes[result.stage] = stage_votes.get(result.stage, 0) + weight
            detections.append((result.stage, result.confidence, pos))

    # Log voting details
    if detections:
        logger.debug(f"Stage detections: {detections}")
        logger.debug(f"Stage vote totals: {stage_votes}")

    # Get winner
    if not stage_votes:
        return "Unknown"

    winner = max(stage_votes, key=stage_votes.get)
    total = sum(stage_votes.values())
    winner_pct = stage_votes[winner] / total * 100 if total > 0 else 0

    # Special handling for Battlefield vs Small Battlefield
    # Gemini tends to default to "Small Battlefield" even for Battlefield
    # Use platform detection to potentially override SBF -> BF, but not the other way
    bf_votes = stage_votes.get("Battlefield", 0)
    sbf_votes = stage_votes.get("Small Battlefield", 0)
    combined = bf_votes + sbf_votes

    if combined >= total * 0.5:  # At least 50% of votes went to BF or SBF
        bf_ratio = bf_votes / combined if combined > 0 else 0
        sbf_ratio = sbf_votes / combined if combined > 0 else 0

        # If Gemini says "Battlefield" at all, trust it (Gemini rarely says BF for SBF)
        if bf_ratio >= 0.2:  # At least 20% said Battlefield
            winner = "Battlefield"
            logger.debug(f"Gemini detected Battlefield ({bf_ratio:.0%})")
        else:
            # Gemini thinks it's Small Battlefield - verify with platform detection
            # Platform detection has false positives, so only use it to confirm BF
            from runitback.cv.stage_detector import has_top_center_platform

            platform_votes = 0
            platform_checks = 0

            # Check multiple frames for platform presence
            check_positions = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
            for pos in check_positions:
                idx = int(pos * (len(frames) - 1))
                if has_top_center_platform(frames[idx]):
                    platform_votes += 1
                platform_checks += 1

            platform_ratio = platform_votes / platform_checks if platform_checks > 0 else 0

            logger.debug(
                f"Platform detection for BF/SBF: {platform_votes}/{platform_checks} "
                f"({platform_ratio:.0%}) have top platform"
            )

            # Trust Gemini's SBF result - platform detection is too unreliable
            # Only override if Gemini had ANY Battlefield votes
            if bf_votes > 0:
                winner = "Battlefield"
                logger.info(f"Using Battlefield (Gemini had {bf_votes} BF votes)")
            else:
                winner = "Small Battlefield"
                logger.debug(f"Gemini confident on Small Battlefield ({sbf_ratio:.0%})")

    logger.info(f"Stage winner: {winner} ({winner_pct:.0f}% of votes)")

    return winner
