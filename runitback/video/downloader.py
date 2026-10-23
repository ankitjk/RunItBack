"""YouTube video downloader using yt-dlp."""

import hashlib
import logging
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Raised when video download fails."""

    pass


def get_cache_path(url: str, cache_dir: Path) -> Path:
    """Generate a cache file path based on the URL hash."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return cache_dir / f"{url_hash}.mp4"


def download_video(
    url: str,
    cache_dir: Path,
    quality: str = "720p",
    force_download: bool = False,
) -> Path:
    """Download a YouTube video and return the path to the downloaded file.

    Args:
        url: YouTube video URL
        cache_dir: Directory to store downloaded videos
        quality: Preferred video quality (e.g., "720p", "1080p")
        force_download: If True, re-download even if cached

    Returns:
        Path to the downloaded video file

    Raises:
        DownloadError: If the download fails
    """
    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    cache_path = get_cache_path(url, cache_dir)
    if cache_path.exists() and not force_download:
        logger.info(f"Using cached video: {cache_path}")
        return cache_path

    # Parse quality to height
    height_map = {
        "360p": 360,
        "480p": 480,
        "720p": 720,
        "1080p": 1080,
    }
    max_height = height_map.get(quality, 720)

    # Configure yt-dlp options
    ydl_opts = {
        "format": (
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]"
            f"/best[height<={max_height}][ext=mp4]/best"
        ),
        "outtmpl": str(cache_path),
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "merge_output_format": "mp4",
    }

    logger.info(f"Downloading video: {url}")
    logger.debug(f"Quality: {quality}, Output: {cache_path}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        raise DownloadError(f"Failed to download video: {e}") from e
    except Exception as e:
        raise DownloadError(f"Unexpected error during download: {e}") from e

    if not cache_path.exists():
        raise DownloadError("Download completed but file not found")

    logger.info(f"Video downloaded successfully: {cache_path}")
    return cache_path


def get_video_info(url: str) -> dict:
    """Get metadata about a YouTube video without downloading.

    Args:
        url: YouTube video URL

    Returns:
        Dictionary containing video metadata (title, duration, etc.)
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if info is None:
        raise DownloadError("Could not extract video info")

    return {
        "title": info.get("title", "Unknown"),
        "duration": info.get("duration", 0),
        "uploader": info.get("uploader", "Unknown"),
        "view_count": info.get("view_count", 0),
        "description": info.get("description", ""),
    }


def extract_characters_from_title(title: str) -> tuple[list[str], list[str]]:
    """Extract potential character names from a video title.

    Tournament video titles often follow patterns like:
    - "Player1 (Character1) vs Player2 (Character2)"
    - "Character1 vs Character2 - Tournament Name"
    - "Player1 (Char) vs Player2 (Char) Grand Finals"

    Args:
        title: YouTube video title

    Returns:
        Tuple of (possible_p1_characters, possible_p2_characters)
        Each list contains character names that might be player 1 or 2
    """
    import re

    # All SSBU characters (canonical names)
    # IMPORTANT: Sort by length descending so longer names match first
    # (e.g., "Dark Pit" matches before "Pit", "Dark Samus" before "Samus")
    CHARACTERS = sorted([
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
    ], key=len, reverse=True)

    # Common abbreviations/nicknames
    ALIASES = {
        "DK": "Donkey Kong",
        "Diddy": "Diddy Kong",
        "ZSS": "Zero Suit Samus",
        "G&W": "Mr. Game & Watch",
        "Game and Watch": "Mr. Game & Watch",
        "ROB": "R.O.B.",
        "Dedede": "King Dedede",
        "K Rool": "King K. Rool",
        "K. Rool": "King K. Rool",
        "Palu": "Palutena",
        "Pac-Man": "Pac-Man",
        "Pacman": "Pac-Man",
        "Falcon": "Captain Falcon",
        "Doc": "Dr. Mario",
        "PT": "Pokemon Trainer",
        "Squirtle": "Pokemon Trainer",
        "Ivysaur": "Pokemon Trainer",
        "Charizard": "Pokemon Trainer",
        "Rosa": "Rosalina & Luma",
        "Rosalina": "Rosalina & Luma",
        "Banjo": "Banjo & Kazooie",
        "Pythra": "Pyra/Mythra",
        "Aegis": "Pyra/Mythra",
        "Pyra": "Pyra/Mythra",
        "Mythra": "Pyra/Mythra",
    }

    # Normalize title for searching
    title_lower = title.lower()

    # Find all characters mentioned in the title
    found_chars: list[str] = []
    matched_spans: list[tuple[int, int]] = []  # Track matched positions to avoid duplicates

    def overlaps_existing(start: int, end: int) -> bool:
        """Check if a span overlaps with any already-matched span."""
        for s, e in matched_spans:
            if start < e and end > s:  # Overlap check
                return True
        return False

    # Check for exact character name matches (case insensitive)
    # Characters are sorted by length, so longer names match first
    for char in CHARACTERS:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(char.lower()) + r'\b'
        for match in re.finditer(pattern, title_lower):
            # Skip if this position already matched a longer name
            if not overlaps_existing(match.start(), match.end()):
                if char not in found_chars:
                    found_chars.append(char)
                matched_spans.append((match.start(), match.end()))

    # Check aliases
    for alias, canonical in ALIASES.items():
        pattern = r'\b' + re.escape(alias.lower()) + r'\b'
        for match in re.finditer(pattern, title_lower):
            if not overlaps_existing(match.start(), match.end()):
                if canonical not in found_chars:
                    found_chars.append(canonical)
                matched_spans.append((match.start(), match.end()))

    # Note: Title order (e.g., "Ken vs Dark Pit") doesn't always match
    # in-game player order (P1 left, P2 right). The title typically shows
    # bracket/stream order which is independent of in-game sides.
    #
    # Therefore, we provide ALL found characters as hints for BOTH players.
    # The hint logic will only apply corrections when the detected character
    # is part of a confusion pair (e.g., Pit/Dark Pit) and the hint suggests
    # the other character in the pair.

    logger.debug(f"Title '{title}' -> Character hints: {found_chars}")

    # Return same characters for both players since we can't determine order
    return (found_chars.copy(), found_chars.copy())
