"""Command-line interface for RunItBack."""

import logging
import re
from dataclasses import dataclass
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from runitback.classification.detector import GameStateDetector
from runitback.classification.states import GameState, GameStateAnalysis
from runitback.config import settings
from runitback.cv.character_detector import detect_characters_from_frames
from runitback.cv.stage_detector import Stage
from runitback.cv.ui_detector import (
    FrameType,
    UIState,
    detect_frame_type,
    detect_game_screen,
    detect_go_screen,
)
from runitback.utils.logging import setup_logging
from runitback.video.downloader import (
    DownloadError,
    download_video,
    extract_characters_from_title,
    get_video_info,
)
from runitback.video.frame_extractor import Frame, FrameExtractionError, extract_frames

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="runitback",
    help="Analyze Super Smash Bros. Ultimate replays from YouTube",
    no_args_is_help=True,
)
console = Console()


@dataclass
class GameInfo:
    """Information about a single game in a set."""

    game_number: int
    start_timestamp: float
    end_timestamp: float
    stage: Stage
    player1_character: str
    player2_character: str
    ui_states: list[UIState]


def validate_youtube_url(url: str) -> bool:
    """Check if the URL is a valid YouTube URL."""
    youtube_patterns = [
        r"^https?://(www\.)?youtube\.com/watch\?v=[\w-]+",
        r"^https?://youtu\.be/[\w-]+",
        r"^https?://(www\.)?youtube\.com/live/[\w-]+",
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def parse_games_option(games_str: str | None, total_games: int) -> list[int]:
    """Parse the --games option string into a list of game numbers.

    Supports formats:
    - "1" - single game
    - "1,3" - comma-separated games
    - "2-4" - range of games
    - "1,3-5" - mixed format

    Args:
        games_str: The games option string
        total_games: Total number of games detected

    Returns:
        Sorted list of game numbers to analyze

    Raises:
        typer.BadParameter: If the format is invalid or games don't exist
    """
    if not games_str:
        return list(range(1, total_games + 1))  # All games

    game_numbers: set[int] = set()

    parts = games_str.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            # Range format: "2-4"
            range_parts = part.split("-")
            if len(range_parts) != 2:
                raise typer.BadParameter(f"Invalid range format: '{part}'")
            try:
                start = int(range_parts[0].strip())
                end = int(range_parts[1].strip())
            except ValueError:
                raise typer.BadParameter(f"Invalid range format: '{part}'")

            if start > end:
                raise typer.BadParameter(f"Invalid range: start ({start}) > end ({end})")

            for i in range(start, end + 1):
                game_numbers.add(i)
        else:
            # Single game number
            try:
                game_numbers.add(int(part))
            except ValueError:
                raise typer.BadParameter(f"Invalid game number: '{part}'")

    # Validate all game numbers exist
    for num in game_numbers:
        if num < 1 or num > total_games:
            raise typer.BadParameter(
                f"Game {num} does not exist. Found {total_games} game(s)."
            )

    return sorted(game_numbers)


def prompt_player_perspective() -> int:
    """Ask the user which player they are in the replay."""
    console.print("\n[bold]Which player are you in this replay?[/bold]")
    console.print("  [cyan]1[/cyan] - Player 1 (left side)")
    console.print("  [cyan]2[/cyan] - Player 2 (right side)")

    while True:
        choice = typer.prompt("Enter 1 or 2", default="1")
        if choice in ("1", "2"):
            return int(choice)
        console.print("[red]Please enter 1 or 2[/red]")


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def detect_games(
    frames: list[Frame],
    verbose: bool = False,
    min_game_duration: float = 30.0,  # Minimum game duration in seconds
    p1_hints: list[str] | None = None,
    p2_hints: list[str] | None = None,
) -> list[GameInfo]:
    """Detect individual games within a set of frames.

    Uses a hybrid approach combining:
    1. GO!/GAME! screen detection - reliable markers for game start/end
    2. HUD-based detection - identifies gameplay frames by UI elements

    The algorithm:
    - Detect GO! screens to mark game starts
    - Detect GAME! screens to mark game ends
    - Use HUD detection to find gameplay segments
    - Combine both signals: GO!/GAME! markers split segments at game boundaries,
      while HUD detection fills in the gameplay between markers

    Args:
        frames: List of extracted frames
        verbose: Enable verbose output
        min_game_duration: Minimum duration (seconds) to count as a game
        p1_hints: Character hints for player 1 (from video title)
        p2_hints: Character hints for player 2 (from video title)

    Returns:
        List of GameInfo objects for each detected game
    """
    games: list[GameInfo] = []

    if not frames:
        return games

    # First pass: detect frame types and GO!/GAME! markers
    frame_types: list[tuple[Frame, FrameType]] = []
    go_indices: list[int] = []  # Indices where GO! is detected
    game_indices: list[int] = []  # Indices where GAME! is detected

    for i, frame in enumerate(frames):
        frame_type = detect_frame_type(frame.image)
        frame_types.append((frame, frame_type))

        # Check for GO! and GAME! screens
        if detect_go_screen(frame.image):
            go_indices.append(i)
            logger.debug(f"GO! detected at frame {i} ({frame.timestamp:.1f}s)")
        if detect_game_screen(frame.image):
            game_indices.append(i)
            logger.debug(f"GAME! detected at frame {i} ({frame.timestamp:.1f}s)")

    # Consolidate GO!/GAME! detections - consecutive frames with same marker
    # should count as a single event (take the first frame of each burst)
    def consolidate_markers(indices: list[int], min_gap: int = 4) -> list[int]:
        """Keep only the first frame of each burst of consecutive markers."""
        if not indices:
            return []
        consolidated = [indices[0]]
        for idx in indices[1:]:
            if idx - consolidated[-1] >= min_gap:
                consolidated.append(idx)
        return consolidated

    go_markers_raw = consolidate_markers(go_indices)
    game_markers_raw = consolidate_markers(game_indices)

    # Validate GAME! markers by checking for white flash that follows
    # Real GAME! screens are followed by a bright white flash within 3-8 seconds
    # This eliminates false positives from green stage elements or menus
    # NOTE: Must happen BEFORE GO! filtering since GO! pairing depends on valid GAME! markers
    import cv2
    import numpy as np

    def has_white_flash_after(marker_idx: int, max_frames: int = 16) -> bool:
        """Check if there's a white flash within max_frames after the marker."""
        for offset in range(1, max_frames + 1):
            check_idx = marker_idx + offset
            if check_idx < len(frames):
                gray = cv2.cvtColor(frames[check_idx].image, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                white_ratio = np.sum(gray > 200) / gray.size
                # White flash has high brightness and significant white ratio
                # Relaxed thresholds to handle stream overlays that partially cover the flash
                if mean_brightness > 180 and white_ratio > 0.5:
                    return True
        return False

    game_markers_validated = [m for m in game_markers_raw if has_white_flash_after(m)]
    logger.debug(f"Validated GAME! markers: {game_markers_raw} -> {game_markers_validated}")

    # Pair GO! markers with GAME! markers to form valid games.
    # A valid GO! starts a game that ends with the next GAME! marker.
    # If multiple GO! markers occur before a GAME!, only the FIRST one after the
    # previous GAME! is valid (the others are likely false positives or replays).
    #
    # Strategy:
    # 1. Start from the beginning (or after the last GAME!)
    # 2. Find the first GO! marker
    # 3. Find the next GAME! marker after it
    # 4. That GO! is valid, skip to after the GAME! and repeat
    valid_go_set = set()

    # Sort both marker lists (use validated GAME! markers)
    go_sorted = sorted(go_markers_raw)
    game_sorted = sorted(game_markers_validated)

    # Start searching from the beginning
    search_start = 0
    game_idx = 0

    # Minimum frames between GAME! end and next GO! start
    # Real games have character select, stage select, loading (~30-40 seconds)
    # Replays or instant transitions are typically < 20 seconds
    MIN_GAP_AFTER_GAME = 40  # 20 seconds at 0.5s/frame

    while game_idx < len(game_sorted):
        game_marker = game_sorted[game_idx]

        # Find the first GO! marker after search_start but before this GAME!
        # Also require minimum gap after previous GAME! to filter out replay GO! screens
        first_go = None
        for go_marker in go_sorted:
            if go_marker >= search_start and go_marker < game_marker - 20:
                first_go = go_marker
                break  # Take the first one only

        if first_go is not None:
            valid_go_set.add(first_go)
            logger.debug(f"Paired GO! at {first_go} with GAME! at {game_marker}")

        # Move search_start to after this GAME! plus minimum gap
        # This filters out GO! screens that appear during replays right after a game ends
        search_start = game_marker + MIN_GAP_AFTER_GAME
        game_idx += 1

    # Also check for GO! markers after the last GAME! (ongoing final game)
    for go_marker in go_sorted:
        if go_marker >= search_start:
            valid_go_set.add(go_marker)
            logger.debug(f"Found GO! at {go_marker} after last GAME! (final game)")
            break  # Take only the first one

    # Convert to sorted list
    go_markers = sorted(valid_go_set)

    logger.debug(f"Filtered GO! markers from {go_markers_raw} to {go_markers}")

    # Use the already-validated GAME! markers
    game_markers = game_markers_validated

    logger.info(f"Detected {len(go_markers)} GO! markers: {go_markers}")
    logger.info(f"Detected {len(game_markers)} GAME! markers (validated with white flash): {game_markers}")

    # PRIMARY APPROACH: Use GO!/GAME! markers to define game boundaries
    # GO! markers indicate game starts, GAME! markers indicate game ends
    # This is more reliable than HUD detection which can be noisy

    # Build games directly from GO!/GAME! marker pairs
    marker_based_games: list[tuple[int, int]] = []

    # Calculate minimum frames for a valid game (at 0.5s per frame)
    min_game_frames = int(min_game_duration / 0.5)

    # Match GO! markers with subsequent GAME! markers
    for i, go_marker in enumerate(go_markers):
        # Find the next GAME! marker after this GO!
        next_game_marker = None
        for game_marker in game_markers:
            if game_marker > go_marker + 20:  # At least 10 seconds after GO!
                next_game_marker = game_marker
                break

        # Also check if there's another GO! marker before the GAME! marker
        # (which would indicate a new game started)
        next_go = None
        if i + 1 < len(go_markers):
            next_go = go_markers[i + 1]

        # Determine end marker for this game
        end_marker = None
        if next_game_marker:
            # If there's a GO! before the GAME!, end at that GO! minus some offset
            if next_go and next_go < next_game_marker:
                end_marker = next_go - 10  # End 5 seconds before next GO!
            else:
                end_marker = next_game_marker + 10  # Include a bit after GAME!
        elif next_go:
            # No GAME! found, but there's another GO! - end before it
            end_marker = next_go - 10
        else:
            # Last game - extend to near end of video
            end_marker = len(frames) - 20

        # Only add if this is a valid game (meets minimum duration)
        if end_marker and (end_marker - go_marker) >= min_game_frames:
            marker_based_games.append((go_marker, end_marker))
        else:
            # Skip this GO! marker - it's likely a false positive or leads to too short a game
            game_duration = (end_marker - go_marker) * 0.5 if end_marker else 0
            logger.debug(f"Skipping GO! at frame {go_marker} - game too short ({game_duration:.1f}s)")

    logger.info(f"Marker-based games: {marker_based_games}")

    # FALLBACK: Also use HUD-based detection for segments not covered by markers
    gameplay_segments: list[tuple[int, int]] = []
    in_gameplay = False
    segment_start = 0

    for i, (frame, frame_type) in enumerate(frame_types):
        if frame_type == FrameType.GAMEPLAY:
            if not in_gameplay:
                in_gameplay = True
                segment_start = i
        else:
            if in_gameplay:
                gameplay_segments.append((segment_start, i - 1))
                in_gameplay = False

    if in_gameplay:
        gameplay_segments.append((segment_start, len(frame_types) - 1))

    # Pre-filter and merge HUD-based segments
    min_segment_frames = 5
    filtered_segments = [
        (start, end) for start, end in gameplay_segments
        if (end - start + 1) >= min_segment_frames
    ]

    def has_marker_in_range(markers: list[int], start_idx: int, end_idx: int) -> bool:
        for marker in markers:
            if start_idx <= marker <= end_idx:
                return True
        return False

    # Merge HUD segments, respecting GO!/GAME! markers as boundaries
    hud_merged: list[tuple[int, int]] = []
    max_gap_frames = 18

    for start_idx, end_idx in filtered_segments:
        if hud_merged:
            prev_start, prev_end = hud_merged[-1]
            gap_size = start_idx - prev_end - 1
            check_start = prev_end - 10
            check_end = start_idx + 10

            has_boundary = (
                has_marker_in_range(game_markers, check_start, check_end) or
                has_marker_in_range(go_markers, check_start, check_end)
            )

            if has_boundary:
                hud_merged.append((start_idx, end_idx))
                continue

            if gap_size <= max_gap_frames:
                hud_merged[-1] = (prev_start, end_idx)
                continue

        hud_merged.append((start_idx, end_idx))

    # COMBINE: Use marker-based games as primary, fill in with HUD-based
    # If we have good GO!/GAME! markers, trust them
    if len(go_markers) >= 2 and len(game_markers) >= 1:
        # Use marker-based approach
        merged_segments = marker_based_games
        logger.info("Using marker-based game detection")
    else:
        # Fall back to HUD-based
        merged_segments = hud_merged
        logger.info("Using HUD-based game detection (insufficient markers)")

    # Filter by minimum duration and create GameInfo objects
    game_number = 1
    for start_idx, end_idx in merged_segments:
        start_time = frame_types[start_idx][0].timestamp
        end_time = frame_types[end_idx][0].timestamp
        duration = end_time - start_time

        if duration < min_game_duration:
            continue

        # Collect gameplay frames for this game
        game_frames_list = [frame_types[i][0] for i in range(start_idx, end_idx + 1)
                           if frame_types[i][1] == FrameType.GAMEPLAY]

        if not game_frames_list:
            continue

        # Stage detection disabled for now to prioritize game/character detection
        # TODO: Re-enable stage detection once game boundaries are reliable
        stage = Stage.UNKNOWN
        # Pass all gameplay frames - detect_characters_from_frames samples at 25%, 50%, 75%
        # Also pass character hints from video title if available
        characters = detect_characters_from_frames(
            [f.image for f in game_frames_list],
            p1_hints=p1_hints,
            p2_hints=p2_hints,
        )

        # Only run OCR for UI state if needed (skip for now to keep it fast)
        # We can add this back for detailed analysis later
        game_ui_states: list[UIState] = []

        games.append(GameInfo(
            game_number=game_number,
            start_timestamp=start_time,
            end_timestamp=end_time,
            stage=stage,
            player1_character=characters.player1.name,
            player2_character=characters.player2.name,
            ui_states=game_ui_states,
        ))
        game_number += 1

    # Apply cross-game consistency for character detection
    # In a set, the same players typically use the same characters
    # If most games detect the same character for a player, use that for outliers
    games = _apply_character_consistency(games)

    return games


def _apply_character_consistency(games: list[GameInfo]) -> list[GameInfo]:
    """Apply cross-game consistency to character detection.

    In tournament sets, players typically use the same character throughout.
    If most games detect the same character for a player, apply that to
    outlier games that may have been misdetected.

    This helps with cases like:
    - Mario in green skin being detected as Luigi in some games
    - Ken being detected as Terry in some games

    Args:
        games: List of detected games

    Returns:
        Games with consistent character assignments
    """
    if len(games) <= 1:
        return games

    # Count character detections per player
    p1_chars: dict[str, int] = {}
    p2_chars: dict[str, int] = {}

    for game in games:
        if game.player1_character != "Unknown":
            p1_chars[game.player1_character] = p1_chars.get(game.player1_character, 0) + 1
        if game.player2_character != "Unknown":
            p2_chars[game.player2_character] = p2_chars.get(game.player2_character, 0) + 1

    # Find dominant character for each player (if any)
    # Require >= 75% of games to have this character (strong majority)
    # This prevents incorrect corrections when detection is split
    def get_dominant(char_counts: dict[str, int], total_games: int) -> str | None:
        if not char_counts:
            return None
        best_char = max(char_counts, key=lambda c: char_counts[c])
        # Need at least 75% agreement to apply consistency correction
        if char_counts[best_char] >= total_games * 0.75:
            return best_char
        return None

    dominant_p1 = get_dominant(p1_chars, len(games))
    dominant_p2 = get_dominant(p2_chars, len(games))

    # Apply consistency - replace outliers with dominant character
    # Only for commonly confused character pairs
    CONFUSION_PAIRS = {
        ("Mario", "Luigi"), ("Luigi", "Mario"),
        ("Ken", "Terry"), ("Terry", "Ken"),
        ("Ken", "Ryu"), ("Ryu", "Ken"),
        ("Marth", "Lucina"), ("Lucina", "Marth"),
        ("Pit", "Dark Pit"), ("Dark Pit", "Pit"),
        ("Samus", "Dark Samus"), ("Dark Samus", "Samus"),
        ("Peach", "Daisy"), ("Daisy", "Peach"),
        ("Simon", "Richter"), ("Richter", "Simon"),
        ("Roy", "Chrom"), ("Chrom", "Roy"),
    }

    updated_games = []
    for game in games:
        p1_char = game.player1_character
        p2_char = game.player2_character

        # Apply P1 consistency
        if dominant_p1 and p1_char != dominant_p1:
            if (p1_char, dominant_p1) in CONFUSION_PAIRS:
                logger.info(f"Game {game.game_number}: P1 {p1_char} -> {dominant_p1} (cross-game consistency)")
                p1_char = dominant_p1

        # Apply P2 consistency
        if dominant_p2 and p2_char != dominant_p2:
            if (p2_char, dominant_p2) in CONFUSION_PAIRS:
                logger.info(f"Game {game.game_number}: P2 {p2_char} -> {dominant_p2} (cross-game consistency)")
                p2_char = dominant_p2

        updated_games.append(GameInfo(
            game_number=game.game_number,
            start_timestamp=game.start_timestamp,
            end_timestamp=game.end_timestamp,
            stage=game.stage,
            player1_character=p1_char,
            player2_character=p2_char,
            ui_states=game.ui_states,
        ))

    return updated_games


def display_game_summary(
    game: GameInfo,
    player_perspective: int,
) -> None:
    """Display summary for a single game."""
    # Determine user's character and opponent's character
    if player_perspective == 1:
        your_char = game.player1_character
        opp_char = game.player2_character
    else:
        your_char = game.player2_character
        opp_char = game.player1_character

    # Create header panel
    header = f"[bold]GAME {game.game_number}[/bold]\n"
    header += f"You: [green]{your_char}[/green] | Opponent: [red]{opp_char}[/red]"

    console.print(Panel(header, expand=False))

    # Show time range
    console.print(
        f"  Duration: {format_timestamp(game.start_timestamp)} - "
        f"{format_timestamp(game.end_timestamp)}"
    )

    # Show final damage/stocks if available
    if game.ui_states:
        final_state = game.ui_states[-1]
        if final_state.player1.damage is not None:
            console.print(
                f"  Final state: "
                f"P1: {final_state.player1.damage:.0f}% ({final_state.player1.stocks} stocks) | "
                f"P2: {final_state.player2.damage:.0f}% ({final_state.player2.stocks} stocks)"
            )

    console.print()


def display_state_analysis(analysis: GameStateAnalysis) -> None:
    """Display the state analysis for a game.

    Uses a transition-focused format that shows:
    1. Summary statistics
    2. Significant interactions (>5s duration or resulted in stock loss)
    3. Small wins/stray hits (brief interactions)

    Args:
        analysis: The GameStateAnalysis result
    """
    # Constants for significance threshold
    SIGNIFICANT_DURATION = 5.0  # Seconds to count as significant

    # Summary stats
    total_time = analysis.total_analyzed_time
    if total_time > 0:
        neutral_pct = (analysis.total_neutral_time / total_time) * 100
        adv_pct = (analysis.total_advantage_time / total_time) * 100
        disadv_pct = (analysis.total_disadvantage_time / total_time) * 100

        console.print("[bold]STATE SUMMARY:[/bold]")
        console.print(
            f"  [white]Neutral:[/white]      "
            f"{format_timestamp(analysis.total_neutral_time)} ({neutral_pct:.0f}%)"
        )
        console.print(
            f"  [green]Advantage:[/green]    "
            f"{format_timestamp(analysis.total_advantage_time)} ({adv_pct:.0f}%)"
        )
        console.print(
            f"  [red]Disadvantage:[/red] "
            f"{format_timestamp(analysis.total_disadvantage_time)} ({disadv_pct:.0f}%)"
        )
        console.print()

    # Combine all segments and sort chronologically
    all_segments = (
        analysis.neutral_segments +
        analysis.advantage_segments +
        analysis.disadvantage_segments
    )
    all_segments.sort(key=lambda s: s.start_time)

    if not all_segments:
        return

    # Categorize non-neutral segments into significant vs small
    significant_segments = []
    small_segments = []

    for seg in all_segments:
        if seg.state == GameState.NEUTRAL:
            continue  # Skip neutral for categorization

        # A segment is significant if:
        # 1. Duration >= 5 seconds, OR
        # 2. A stock was lost during it
        has_stock_loss = bool(seg.stock_losses)
        is_significant = seg.duration >= SIGNIFICANT_DURATION or has_stock_loss

        if is_significant:
            significant_segments.append(seg)
        else:
            small_segments.append(seg)

    # Helper to format a segment line
    def format_segment_line(seg, player_perspective: int) -> str:
        """Format a single segment for display."""
        timestamp = format_timestamp(seg.start_time)
        duration = f"{seg.duration:.1f}s"

        # State indicator
        if seg.state == GameState.ADVANTAGE:
            state_str = "[green]▲ Advantage[/green]"
        elif seg.state == GameState.DISADVANTAGE:
            state_str = "[red]▼ Disadvantage[/red]"
        else:
            state_str = "[white]○ Neutral[/white]"

        # Disadvantage type if available
        type_str = ""
        if seg.state == GameState.DISADVANTAGE and seg.disadvantage_type:
            from runitback.classification.states import DisadvantageType
            if seg.disadvantage_type != DisadvantageType.UNKNOWN:
                type_str = f" ({seg.disadvantage_type.display_name})"

        # Stock loss indicator
        stock_str = ""
        if seg.stock_losses:
            for loss in seg.stock_losses:
                if loss.player == player_perspective:
                    stock_str += " [red]★ Lost stock[/red]"
                else:
                    stock_str += " [green]★ Took stock[/green]"

        return f"  {timestamp}  {state_str}{type_str} [{duration}]{stock_str}"

    # Display significant interactions
    if significant_segments:
        console.print("[bold]SIGNIFICANT INTERACTIONS:[/bold]")
        console.print("[dim]  (>5s duration or stock lost)[/dim]")
        for seg in significant_segments:
            console.print(format_segment_line(seg, analysis.player_perspective))
        console.print()

    # Display small wins/stray hits (collapsed summary)
    if small_segments:
        # Count by type
        small_adv = [s for s in small_segments if s.state == GameState.ADVANTAGE]
        small_disadv = [s for s in small_segments if s.state == GameState.DISADVANTAGE]

        console.print("[bold dim]SMALL INTERACTIONS:[/bold dim]")
        console.print("[dim]  (brief exchanges <5s)[/dim]")

        if small_adv:
            adv_total = sum(s.duration for s in small_adv)
            console.print(
                f"  [green]▲ Small wins:[/green] {len(small_adv)} "
                f"({format_timestamp(adv_total)} total)"
            )

        if small_disadv:
            disadv_total = sum(s.duration for s in small_disadv)
            console.print(
                f"  [red]▼ Stray hits:[/red] {len(small_disadv)} "
                f"({format_timestamp(disadv_total)} total)"
            )

        console.print()

    # Chronological transition timeline (compact view)
    console.print("[bold]NEUTRAL TRANSITIONS:[/bold]")
    console.print("[dim]  (timestamps when neutral was won/lost)[/dim]")

    # Build transition list from segments
    transitions: list[tuple[float, str, GameState | None]] = []
    for i, seg in enumerate(all_segments):
        if seg.state == GameState.NEUTRAL:
            # Entering neutral = won neutral
            transitions.append((seg.start_time, "won", None))
        else:
            # Entering adv/disadv = lost neutral
            transitions.append((seg.start_time, "lost", seg.state))

    # Display transitions as a compact timeline
    timeline_parts = []
    for timestamp, direction, state in transitions:
        ts = format_timestamp(timestamp)
        if direction == "won":
            timeline_parts.append(f"[white]{ts}○[/white]")
        elif state == GameState.ADVANTAGE:
            timeline_parts.append(f"[green]{ts}▲[/green]")
        else:
            timeline_parts.append(f"[red]{ts}▼[/red]")

    # Print timeline in rows of 8
    for i in range(0, len(timeline_parts), 8):
        row = timeline_parts[i:i+8]
        console.print("  " + "  ".join(row))

    console.print()
    console.print("[dim]Legend: ○ Neutral  [green]▲ Advantage[/green]  [red]▼ Disadvantage[/red][/dim]")


@app.command()
def analyze(
    url: Annotated[str, typer.Argument(help="YouTube URL of the SSBU replay")],
    player: Annotated[
        int | None,
        typer.Option(
            "--player",
            "-p",
            help="Player perspective (1 or 2). If not provided, you'll be prompted.",
            min=1,
            max=2,
        ),
    ] = None,
    games: Annotated[
        str | None,
        typer.Option(
            "--games",
            "-g",
            help="Games to analyze (e.g., '1', '1,3', '2-4'). Defaults to all games.",
        ),
    ] = None,
    sample_rate: Annotated[
        float,
        typer.Option(
            "--sample-rate",
            "-s",
            help="Seconds between frame samples",
            min=0.1,
            max=5.0,
        ),
    ] = 0.5,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Analyze a Super Smash Bros. Ultimate replay from YouTube.

    Downloads the video, extracts frames, and identifies game states,
    edgeguarding, and ledgetrapping situations.
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.WARNING
    setup_logging(level=log_level)

    # Validate URL
    if not validate_youtube_url(url):
        console.print("[red]Error:[/red] Invalid YouTube URL")
        console.print("Please provide a valid YouTube video URL")
        raise typer.Exit(1)

    # Get player perspective
    if player is None:
        player = prompt_player_perspective()

    console.print("\n[bold green]RunItBack[/bold green] - SSBU Replay Analyzer")
    console.print(f"Analyzing as [cyan]Player {player}[/cyan]")
    console.print(f"URL: {url}\n")

    # Get video info (title, etc.) for character hints
    p1_hints: list[str] | None = None
    p2_hints: list[str] | None = None
    video_title: str = "Unknown"

    try:
        video_info = get_video_info(url)
        video_title = video_info.get("title", "Unknown")
        p1_hints, p2_hints = extract_characters_from_title(video_title)
        if p1_hints or p2_hints:
            logger.info(f"Title hints - P1: {p1_hints}, P2: {p2_hints}")
    except Exception as e:
        logger.warning(f"Could not get video info: {e}")

    # Download video
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Downloading video...", total=None)
        try:
            video_path = download_video(url, cache_dir=settings.cache_dir)
        except DownloadError as e:
            console.print(f"[red]Error downloading video:[/red] {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)

    console.print(f"[green]Downloaded:[/green] {video_path.name}")
    if video_title != "Unknown":
        console.print(f"[dim]Title: {video_title}[/dim]")

    # Extract frames
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Extracting frames...", total=None)
        try:
            frames = list(extract_frames(video_path, interval_sec=sample_rate))
        except FrameExtractionError as e:
            console.print(f"[red]Error extracting frames:[/red] {e}")
            raise typer.Exit(1)

    if not frames:
        console.print("[red]Error:[/red] No frames extracted from video")
        raise typer.Exit(1)

    console.print(f"[green]Extracted:[/green] {len(frames)} frames")

    if verbose:
        console.print(f"[dim]Sample rate: {sample_rate}s[/dim]")
        console.print(f"[dim]Video duration: ~{frames[-1].timestamp:.1f}s[/dim]")

    # Detect games
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Analyzing gameplay...", total=None)
        detected_games = detect_games(
            frames,
            verbose=verbose,
            p1_hints=p1_hints,
            p2_hints=p2_hints,
        )

    if not detected_games:
        console.print("[yellow]No games detected in video.[/yellow]")
        console.print("This might happen if the video doesn't contain SSBU gameplay.")
        raise typer.Exit(0)

    # Parse and validate --games option
    try:
        games_to_analyze = parse_games_option(games, len(detected_games))
    except typer.BadParameter as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Display results
    console.print(f"\n[bold]Found {len(detected_games)} game(s)[/bold]")
    if games:
        console.print(f"[dim]Analyzing game(s): {', '.join(str(g) for g in games_to_analyze)}[/dim]")
    console.print()

    # Initialize state detector
    state_detector = GameStateDetector(
        player_perspective=player,
        hit_threshold=4.0,  # Minimum damage to count as a hit
        neutral_reset_time=3.0,  # Seconds without damage to return to neutral
    )

    # Analyze each selected game
    for game in detected_games:
        if game.game_number not in games_to_analyze:
            continue

        console.print("=" * 50)
        display_game_summary(game, player)

        # Extract frames for this game's time range
        game_frames = [
            f for f in frames
            if game.start_timestamp <= f.timestamp <= game.end_timestamp
        ]

        if not game_frames:
            console.print("[yellow]No frames found for this game.[/yellow]")
            continue

        # Collect game start and end times for all detected games
        # This helps filter out false positives at game transitions
        game_start_times = [g.start_timestamp for g in detected_games]
        game_end_times = [g.end_timestamp for g in detected_games]

        # Run state detection
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Analyzing game {game.game_number} states...", total=None)
            analysis = state_detector.detect_states(
                game_frames,
                frame_interval=sample_rate,
                game_end_times=game_end_times,
                game_start_times=game_start_times,
            )

        # Display state analysis
        if analysis.frame_states:
            display_state_analysis(analysis)


@app.command()
def version() -> None:
    """Show the version of RunItBack."""
    from runitback import __version__

    console.print(f"RunItBack v{__version__}")


@app.command()
def add_stage_reference(
    image_path: Annotated[str, typer.Argument(help="Path to stage screenshot")],
    stage_name: Annotated[str, typer.Argument(help="Stage name (e.g., battlefield, smashville)")],
) -> None:
    """Add a reference image for stage detection.

    The image should be a screenshot during gameplay on the target stage.
    The top 75% of the image will be used (stage area without HUD).

    Valid stage names:
    - small_battlefield
    - battlefield
    - final_destination
    - town_and_city
    - smashville
    - hollow_bastion
    - kalos_pokemon_league
    - pokemon_stadium_2
    """
    from pathlib import Path

    import cv2

    from runitback.cv.stage_detector import STAGE_REFERENCES_DIR, Stage

    # Validate stage name
    try:
        stage = Stage(stage_name)
    except ValueError:
        console.print(f"[red]Error:[/red] Unknown stage '{stage_name}'")
        console.print("Valid stages:")
        for s in Stage:
            if s != Stage.UNKNOWN:
                console.print(f"  - {s.value}")
        raise typer.Exit(1)

    # Load image
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        console.print(f"[red]Error:[/red] File not found: {image_path}")
        raise typer.Exit(1)

    img = cv2.imread(str(image_path_obj))
    if img is None:
        console.print(f"[red]Error:[/red] Could not load image: {image_path}")
        raise typer.Exit(1)

    h, w = img.shape[:2]
    console.print(f"Loaded image: {w}x{h}")

    # Extract stage region (top 75%)
    stage_region = img[:int(0.75 * h), :]

    # Save to references directory
    STAGE_REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing references to determine suffix
    existing = list(STAGE_REFERENCES_DIR.glob(f"{stage_name}*.png"))
    if existing:
        suffix = f"_{len(existing) + 1}"
        output_name = f"{stage_name}{suffix}.png"
    else:
        output_name = f"{stage_name}.png"

    output_path = STAGE_REFERENCES_DIR / output_name
    cv2.imwrite(str(output_path), stage_region)

    console.print(f"[green]Saved:[/green] {output_path}")
    console.print(f"Stage reference for [cyan]{stage.display_name}[/cyan] added.")
    console.print(f"[dim]Total references for this stage: {len(existing) + 1}[/dim]")


if __name__ == "__main__":
    app()
