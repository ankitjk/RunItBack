"""Integration tests for game detection using sample YouTube videos.

These tests verify that the game detection correctly identifies:
- Number of games in a set
- Stage for each game
- Approximate start/end timestamps

Note: These tests require network access to download videos and may take
several minutes to run. They are marked as slow tests.

Test data is based on actual tournament footage with known game/stage data.
"""

import pytest
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExpectedStockLoss:
    """Expected stock loss event for test verification."""
    timestamp: tuple[int, int]  # (minutes, seconds) in video time
    player: int  # 1 or 2 - which player lost the stock
    is_game_end: bool = False  # True if this is the final stock (GAME!)
    timestamp_tolerance: int = 3  # Tolerance in seconds


@dataclass
class ExpectedGame:
    """Expected game data for test verification."""
    game_number: int
    stage: str
    start_time: tuple[int, int]  # (minutes, seconds)
    end_time: tuple[int, int]  # (minutes, seconds)
    player1_character: Optional[str] = None  # Player 1 character name
    player2_character: Optional[str] = None  # Player 2 character name
    stock_losses: Optional[list[ExpectedStockLoss]] = None  # Expected stock losses
    # Tolerance for timestamp matching (seconds)
    timestamp_tolerance: int = 30


@dataclass
class TestVideo:
    """Test video with expected game data."""

    # Not a test class - the Test* name matches pytest's default
    # python_classes glob, so opt out of collection explicitly.
    __test__ = False

    url: str
    video_id: str
    description: str
    expected_games: list[ExpectedGame]


# Test videos with known game/stage data
TEST_VIDEOS = [
    TestVideo(
        url="https://www.youtube.com/watch?v=7oPdiBsgSG0",
        video_id="7oPdiBsgSG0",
        description="2 game set: Ken vs King Dedede on Battlefield, Hollow Bastion",
        expected_games=[
            ExpectedGame(
                game_number=1,
                stage="battlefield",
                start_time=(0, 12),  # Approximate from detection
                end_time=(3, 42),
                player1_character="Ken",
                player2_character="King Dedede",
                # Ken (P1) loses all 3 stocks, Dedede (P2) loses none
                stock_losses=[
                    ExpectedStockLoss(timestamp=(1, 8), player=1),   # Ken loses stock 1
                    ExpectedStockLoss(timestamp=(2, 40), player=1),  # Ken loses stock 2 (star KO)
                    ExpectedStockLoss(timestamp=(3, 31), player=1, is_game_end=True),  # Ken loses stock 3 (GAME!)
                ],
            ),
            ExpectedGame(
                game_number=2,
                stage="hollow_bastion",
                start_time=(4, 7),
                end_time=(8, 5),
                player1_character="Ken",
                player2_character="King Dedede",
                # Game 2: Ken loses 3 stocks, Dedede loses 2 stocks
                # Ken (P1) loses stocks at: 5:05, 6:43, 7:52 (GAME!)
                # Dedede (P2) loses stocks at: 5:16, 6:34
                stock_losses=[
                    ExpectedStockLoss(timestamp=(5, 5), player=1),   # Ken loses stock 1
                    ExpectedStockLoss(timestamp=(5, 16), player=2),  # Dedede loses stock 1
                    ExpectedStockLoss(timestamp=(6, 34), player=2),  # Dedede loses stock 2
                    ExpectedStockLoss(timestamp=(6, 43), player=1),  # Ken loses stock 2
                    ExpectedStockLoss(timestamp=(7, 52), player=1, is_game_end=True),  # Ken loses stock 3 (GAME!)
                ],
            ),
        ],
    ),
    TestVideo(
        url="https://www.youtube.com/watch?v=fRcif5Qa2Is",
        video_id="fRcif5Qa2Is",
        description="3 game set: Ken vs Sonic on Battlefield, Hollow Bastion, Battlefield",
        expected_games=[
            ExpectedGame(
                game_number=1,
                stage="battlefield",
                start_time=(0, 16),
                end_time=(3, 59),
                player1_character="Ken",
                player2_character="Sonic",
                # Ken (P1) loses 3 stocks, Sonic (P2) loses 2 stocks
                stock_losses=[
                    ExpectedStockLoss(timestamp=(1, 20), player=2),   # Sonic loses stock 1
                    ExpectedStockLoss(timestamp=(1, 42), player=1),   # Ken loses stock 1
                    ExpectedStockLoss(timestamp=(2, 52), player=1),   # Ken loses stock 2
                    ExpectedStockLoss(timestamp=(3, 49), player=2),   # Sonic loses stock 2
                    ExpectedStockLoss(timestamp=(3, 58), player=1, is_game_end=True),  # Ken loses stock 3 (GAME!)
                ],
            ),
            ExpectedGame(
                game_number=2,
                stage="hollow_bastion",
                start_time=(4, 57),
                end_time=(9, 12),
                player1_character="Ken",
                player2_character="Sonic",
                # Ken (P1) loses 1 stock, Sonic (P2) loses 3 stocks
                stock_losses=[
                    ExpectedStockLoss(timestamp=(5, 34), player=2),   # Sonic loses stock 1
                    ExpectedStockLoss(timestamp=(6, 51), player=1),   # Ken loses stock 1
                    ExpectedStockLoss(timestamp=(7, 38), player=2),   # Sonic loses stock 2
                    ExpectedStockLoss(timestamp=(9, 12), player=2, is_game_end=True),  # Sonic loses stock 3 (GAME!)
                ],
            ),
            ExpectedGame(
                game_number=3,
                stage="battlefield",
                start_time=(9, 53),
                end_time=(12, 49),
                player1_character="Ken",
                player2_character="Sonic",
                # Ken (P1) loses 1 stock, Sonic (P2) loses 3 stocks
                stock_losses=[
                    ExpectedStockLoss(timestamp=(10, 49), player=2),  # Sonic loses stock 1
                    ExpectedStockLoss(timestamp=(11, 56), player=2),  # Sonic loses stock 2
                    ExpectedStockLoss(timestamp=(12, 28), player=1),  # Ken loses stock 1
                    ExpectedStockLoss(timestamp=(12, 48), player=2, is_game_end=True),  # Sonic loses stock 3 (GAME!)
                ],
            ),
        ],
    ),
    TestVideo(
        url="https://www.youtube.com/watch?v=St7XW7w_jaA",
        video_id="St7XW7w_jaA",
        description="4 game set: Dark Pit vs Ken on Pokemon Stadium 2 x4",
        expected_games=[
            ExpectedGame(
                game_number=1,
                stage="pokemon_stadium_2",
                start_time=(0, 18),
                end_time=(2, 40),
                player1_character="Dark Pit",
                player2_character="Ken",
                # Dark Pit (P1) loses 3 stocks, Ken (P2) loses 1 stock
                stock_losses=[
                    ExpectedStockLoss(timestamp=(0, 33), player=1),   # Dark Pit loses stock 1
                    ExpectedStockLoss(timestamp=(1, 29), player=1),   # Dark Pit loses stock 2
                    ExpectedStockLoss(timestamp=(1, 46), player=2),   # Ken loses stock 1
                    ExpectedStockLoss(timestamp=(2, 40), player=1, is_game_end=True),  # Dark Pit loses stock 3 (GAME!)
                ],
            ),
            ExpectedGame(
                game_number=2,
                stage="pokemon_stadium_2",
                start_time=(3, 18),
                end_time=(6, 39),
                player1_character="Dark Pit",
                player2_character="Ken",
                # Dark Pit (P1) loses 2 stocks, Ken (P2) loses 3 stocks
                stock_losses=[
                    ExpectedStockLoss(timestamp=(3, 51), player=1),   # Dark Pit loses stock 1
                    ExpectedStockLoss(timestamp=(4, 49), player=1),   # Dark Pit loses stock 2
                    ExpectedStockLoss(timestamp=(4, 54), player=2),   # Ken loses stock 1
                    ExpectedStockLoss(timestamp=(5, 34), player=2),   # Ken loses stock 2
                    ExpectedStockLoss(timestamp=(6, 39), player=2, is_game_end=True),  # Ken loses stock 3 (GAME!)
                ],
            ),
            ExpectedGame(
                game_number=3,
                stage="pokemon_stadium_2",
                start_time=(7, 21),
                end_time=(10, 15),
                player1_character="Dark Pit",
                player2_character="Ken",
                # Dark Pit (P1) loses 3 stocks, Ken (P2) loses 2 stocks
                stock_losses=[
                    ExpectedStockLoss(timestamp=(7, 53), player=1),   # Dark Pit loses stock 1
                    ExpectedStockLoss(timestamp=(8, 41), player=2),   # Ken loses stock 1
                    ExpectedStockLoss(timestamp=(8, 47), player=1),   # Dark Pit loses stock 2
                    ExpectedStockLoss(timestamp=(9, 56), player=2),   # Ken loses stock 2
                    ExpectedStockLoss(timestamp=(10, 15), player=1, is_game_end=True),  # Dark Pit loses stock 3 (GAME!)
                ],
            ),
            ExpectedGame(
                game_number=4,
                stage="pokemon_stadium_2",
                start_time=(10, 59),
                end_time=(13, 25),
                player1_character="Dark Pit",
                player2_character="Ken",
                # Dark Pit (P1) loses 3 stocks, Ken (P2) loses 1 stock
                stock_losses=[
                    ExpectedStockLoss(timestamp=(11, 38), player=1),  # Dark Pit loses stock 1
                    ExpectedStockLoss(timestamp=(12, 33), player=2),  # Ken loses stock 1
                    ExpectedStockLoss(timestamp=(12, 48), player=1),  # Dark Pit loses stock 2
                    ExpectedStockLoss(timestamp=(13, 25), player=1, is_game_end=True),  # Dark Pit loses stock 3 (GAME!)
                ],
            ),
        ],
    ),
    TestVideo(
        url="https://www.youtube.com/watch?v=nmzdyQz6msI",
        video_id="nmzdyQz6msI",
        description="5 game set: Steve vs Palutena on Small Battlefield x4, Town and City",
        expected_games=[
            ExpectedGame(
                game_number=1,
                stage="small_battlefield",
                start_time=(0, 21),
                end_time=(2, 38),
                player1_character="Steve",
                player2_character="Palutena",
                # Steve (P1) loses 1 stock, Palutena (P2) loses 3 stocks
                stock_losses=[
                    ExpectedStockLoss(timestamp=(1, 29), player=2),   # Palutena loses stock 1
                    ExpectedStockLoss(timestamp=(1, 39), player=1),   # Steve loses stock 1
                    ExpectedStockLoss(timestamp=(2, 9), player=2),    # Palutena loses stock 2
                    ExpectedStockLoss(timestamp=(2, 38), player=2, is_game_end=True),  # Palutena loses stock 3 (GAME!)
                ],
            ),
            ExpectedGame(
                game_number=2,
                stage="small_battlefield",
                start_time=(3, 18),
                end_time=(7, 10),
                player1_character="Steve",
                player2_character="Palutena",
                # Steve (P1) loses 3 stocks, Palutena (P2) loses 2 stocks
                stock_losses=[
                    ExpectedStockLoss(timestamp=(4, 34), player=1),   # Steve loses stock 1
                    ExpectedStockLoss(timestamp=(4, 45), player=2),   # Palutena loses stock 1
                    ExpectedStockLoss(timestamp=(5, 35), player=2),   # Palutena loses stock 2
                    ExpectedStockLoss(timestamp=(5, 49), player=1),   # Steve loses stock 2
                    ExpectedStockLoss(timestamp=(7, 10), player=1, is_game_end=True),  # Steve loses stock 3 (GAME!)
                ],
            ),
            ExpectedGame(
                game_number=3,
                stage="small_battlefield",
                start_time=(7, 42),
                end_time=(11, 45),
                player1_character="Steve",
                player2_character="Palutena",
                # Steve (P1) loses 3 stocks, Palutena (P2) loses 2 stocks
                stock_losses=[
                    ExpectedStockLoss(timestamp=(9, 10), player=1),   # Steve loses stock 1
                    ExpectedStockLoss(timestamp=(9, 20), player=2),   # Palutena loses stock 1
                    ExpectedStockLoss(timestamp=(10, 53), player=1),  # Steve loses stock 2
                    ExpectedStockLoss(timestamp=(11, 24), player=2),  # Palutena loses stock 2
                    ExpectedStockLoss(timestamp=(11, 45), player=1, is_game_end=True),  # Steve loses stock 3 (GAME!)
                ],
            ),
            ExpectedGame(
                game_number=4,
                stage="town_and_city",
                start_time=(12, 38),
                end_time=(16, 54),
                player1_character="Steve",
                player2_character="Palutena",
                # Steve (P1) loses 2 stocks, Palutena (P2) loses 3 stocks
                stock_losses=[
                    ExpectedStockLoss(timestamp=(14, 9), player=1),   # Steve loses stock 1
                    ExpectedStockLoss(timestamp=(14, 17), player=2),  # Palutena loses stock 1
                    ExpectedStockLoss(timestamp=(15, 58), player=1),  # Steve loses stock 2
                    ExpectedStockLoss(timestamp=(16, 18), player=2),  # Palutena loses stock 2
                    ExpectedStockLoss(timestamp=(16, 54), player=2, is_game_end=True),  # Palutena loses stock 3 (GAME!)
                ],
            ),
            ExpectedGame(
                game_number=5,
                stage="small_battlefield",
                start_time=(17, 28),
                end_time=(22, 25),
                player1_character="Steve",
                player2_character="Palutena",
                # Steve (P1) loses 3 stocks, Palutena (P2) loses 1 stock
                stock_losses=[
                    ExpectedStockLoss(timestamp=(19, 31), player=2),  # Palutena loses stock 1
                    ExpectedStockLoss(timestamp=(19, 55), player=1),  # Steve loses stock 1
                    ExpectedStockLoss(timestamp=(21, 12), player=1),  # Steve loses stock 2
                    ExpectedStockLoss(timestamp=(22, 25), player=1, is_game_end=True),  # Steve loses stock 3 (GAME!)
                ],
            ),
        ],
    ),
    TestVideo(
        url="https://www.youtube.com/watch?v=KfMZ9bQUbtk",
        video_id="KfMZ9bQUbtk",
        description="3 game set: Cloud vs Pit on Pokemon Stadium 2 x2, Small Battlefield",
        expected_games=[
            ExpectedGame(
                game_number=1,
                stage="pokemon_stadium_2",
                start_time=(0, 16),
                end_time=(3, 25),
                player1_character="Cloud",
                player2_character="Pit",
                # Cloud (P1) loses 3 stocks, Pit (P2) loses 1 stock
                stock_losses=[
                    ExpectedStockLoss(timestamp=(0, 23), player=1),   # Cloud loses stock 1
                    ExpectedStockLoss(timestamp=(1, 47), player=1),   # Cloud loses stock 2
                    ExpectedStockLoss(timestamp=(2, 14), player=2),   # Pit loses stock 1
                    ExpectedStockLoss(timestamp=(3, 25), player=1, is_game_end=True),  # Cloud loses stock 3 (GAME!)
                ],
            ),
            ExpectedGame(
                game_number=2,
                stage="pokemon_stadium_2",
                start_time=(4, 4),
                end_time=(7, 31),
                player1_character="Cloud",
                player2_character="Pit",
                # Cloud (P1) loses 2 stocks, Pit (P2) loses 3 stocks
                stock_losses=[
                    ExpectedStockLoss(timestamp=(4, 51), player=2),   # Pit loses stock 1
                    ExpectedStockLoss(timestamp=(5, 53), player=2),   # Pit loses stock 2
                    ExpectedStockLoss(timestamp=(6, 16), player=1),   # Cloud loses stock 1
                    ExpectedStockLoss(timestamp=(7, 22), player=1),   # Cloud loses stock 2
                    ExpectedStockLoss(timestamp=(7, 31), player=2, is_game_end=True),  # Pit loses stock 3 (GAME!)
                ],
            ),
            ExpectedGame(
                game_number=3,
                stage="small_battlefield",
                start_time=(8, 23),
                end_time=(12, 40),
                player1_character="Cloud",
                player2_character="Pit",
                # Cloud (P1) loses 2 stocks, Pit (P2) loses 3 stocks
                stock_losses=[
                    ExpectedStockLoss(timestamp=(9, 18), player=2),   # Pit loses stock 1
                    ExpectedStockLoss(timestamp=(10, 19), player=1),  # Cloud loses stock 1
                    ExpectedStockLoss(timestamp=(11, 47), player=2),  # Pit loses stock 2
                    ExpectedStockLoss(timestamp=(11, 55), player=1),  # Cloud loses stock 2
                    ExpectedStockLoss(timestamp=(12, 40), player=2, is_game_end=True),  # Pit loses stock 3 (GAME!)
                ],
            ),
        ],
    ),
    TestVideo(
        url="https://www.youtube.com/watch?v=UgGQkF1wkoA",
        video_id="UgGQkF1wkoA",
        description="3 game set: Mario vs Ness on Pokemon Stadium 2 x2, Small Battlefield",
        expected_games=[
            ExpectedGame(
                game_number=1,
                stage="pokemon_stadium_2",
                start_time=(0, 31),
                end_time=(3, 46),
                player1_character="Mario",
                player2_character="Ness",
                # Mario (P1) loses 2 stocks, Ness (P2) loses 3 stocks
                stock_losses=[
                    ExpectedStockLoss(timestamp=(0, 59), player=1),   # Mario loses stock 1
                    ExpectedStockLoss(timestamp=(1, 31), player=2),   # Ness loses stock 1
                    ExpectedStockLoss(timestamp=(2, 11), player=1),   # Mario loses stock 2
                    ExpectedStockLoss(timestamp=(2, 56), player=2),   # Ness loses stock 2
                    ExpectedStockLoss(timestamp=(3, 46), player=2, is_game_end=True),  # Ness loses stock 3 (GAME!)
                ],
            ),
            ExpectedGame(
                game_number=2,
                stage="pokemon_stadium_2",
                start_time=(4, 40),
                end_time=(7, 17),
                player1_character="Mario",
                player2_character="Ness",
                # Mario (P1) loses 1 stock, Ness (P2) loses 3 stocks
                stock_losses=[
                    ExpectedStockLoss(timestamp=(5, 33), player=2),   # Ness loses stock 1
                    ExpectedStockLoss(timestamp=(6, 21), player=1),   # Mario loses stock 1
                    ExpectedStockLoss(timestamp=(6, 28), player=2),   # Ness loses stock 2
                    ExpectedStockLoss(timestamp=(7, 17), player=2, is_game_end=True),  # Ness loses stock 3 (GAME!)
                ],
            ),
            ExpectedGame(
                game_number=3,
                stage="small_battlefield",
                start_time=(7, 57),
                end_time=(10, 47),
                player1_character="Mario",
                player2_character="Ness",
                # Mario (P1) loses 1 stock, Ness (P2) loses 3 stocks
                stock_losses=[
                    ExpectedStockLoss(timestamp=(8, 39), player=2),   # Ness loses stock 1
                    ExpectedStockLoss(timestamp=(8, 56), player=1),   # Mario loses stock 1
                    ExpectedStockLoss(timestamp=(9, 25), player=2),   # Ness loses stock 2
                    ExpectedStockLoss(timestamp=(10, 47), player=2, is_game_end=True),  # Ness loses stock 3 (GAME!)
                ],
            ),
        ],
    ),
]


def timestamp_to_seconds(time_tuple: tuple[int, int]) -> float:
    """Convert (minutes, seconds) tuple to total seconds."""
    return time_tuple[0] * 60 + time_tuple[1]


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS string."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


@pytest.fixture(scope="module")
def video_cache():
    """Cache for downloaded videos to avoid re-downloading."""
    return {}


class TestGameDetection:
    """Test game detection on sample videos."""

    @pytest.mark.slow
    @pytest.mark.parametrize("test_video", TEST_VIDEOS, ids=lambda v: v.video_id)
    def test_game_count(self, test_video: TestVideo):
        """Test that the correct number of games is detected."""
        from runitback.video.downloader import download_video
        from runitback.video.frame_extractor import extract_frames
        from runitback.cli import detect_games
        from runitback.config import settings

        # Download video
        video_path = download_video(test_video.url, cache_dir=settings.cache_dir)

        # Extract frames
        frames = list(extract_frames(video_path, interval_sec=0.5))

        # Detect games
        games = detect_games(frames)

        assert len(games) == len(test_video.expected_games), (
            f"Expected {len(test_video.expected_games)} games, "
            f"got {len(games)} for {test_video.video_id}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("test_video", TEST_VIDEOS, ids=lambda v: v.video_id)
    def test_stage_detection(self, test_video: TestVideo):
        """Test that stages are correctly detected for each game."""
        from runitback.video.downloader import download_video
        from runitback.video.frame_extractor import extract_frames
        from runitback.cli import detect_games
        from runitback.config import settings

        # Download video
        video_path = download_video(test_video.url, cache_dir=settings.cache_dir)

        # Extract frames
        frames = list(extract_frames(video_path, interval_sec=0.5))

        # Detect games
        games = detect_games(frames)

        for i, (game, expected) in enumerate(zip(games, test_video.expected_games)):
            assert game.stage.value == expected.stage, (
                f"Game {i + 1}: Expected stage '{expected.stage}', "
                f"got '{game.stage.value}' for {test_video.video_id}"
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("test_video", TEST_VIDEOS, ids=lambda v: v.video_id)
    def test_timestamp_accuracy(self, test_video: TestVideo):
        """Test that game timestamps are within tolerance of expected values."""
        from runitback.video.downloader import download_video
        from runitback.video.frame_extractor import extract_frames
        from runitback.cli import detect_games
        from runitback.config import settings

        # Download video
        video_path = download_video(test_video.url, cache_dir=settings.cache_dir)

        # Extract frames
        frames = list(extract_frames(video_path, interval_sec=0.5))

        # Detect games
        games = detect_games(frames)

        for i, (game, expected) in enumerate(zip(games, test_video.expected_games)):
            expected_start = timestamp_to_seconds(expected.start_time)
            expected_end = timestamp_to_seconds(expected.end_time)

            start_diff = abs(game.start_timestamp - expected_start)
            end_diff = abs(game.end_timestamp - expected_end)

            assert start_diff <= expected.timestamp_tolerance, (
                f"Game {i + 1} start time off by {start_diff:.1f}s "
                f"(expected {seconds_to_timestamp(expected_start)}, "
                f"got {seconds_to_timestamp(game.start_timestamp)}) "
                f"for {test_video.video_id}"
            )

            assert end_diff <= expected.timestamp_tolerance, (
                f"Game {i + 1} end time off by {end_diff:.1f}s "
                f"(expected {seconds_to_timestamp(expected_end)}, "
                f"got {seconds_to_timestamp(game.end_timestamp)}) "
                f"for {test_video.video_id}"
            )


    @pytest.mark.slow
    @pytest.mark.parametrize("test_video", [v for v in TEST_VIDEOS if v.expected_games[0].player1_character], ids=lambda v: v.video_id)
    def test_character_detection(self, test_video: TestVideo):
        """Test that characters are correctly detected for each game."""
        from runitback.video.downloader import download_video, get_video_info, extract_characters_from_title
        from runitback.video.frame_extractor import extract_frames
        from runitback.cli import detect_games
        from runitback.config import settings

        # Download video
        video_path = download_video(test_video.url, cache_dir=settings.cache_dir)

        # Get character hints from video title (just like the CLI does)
        try:
            video_info = get_video_info(test_video.url)
            p1_hints, p2_hints = extract_characters_from_title(video_info.get("title", ""))
        except Exception:
            p1_hints, p2_hints = None, None

        # Extract frames
        frames = list(extract_frames(video_path, interval_sec=0.5))

        # Detect games (with title hints if available)
        games = detect_games(frames, p1_hints=p1_hints, p2_hints=p2_hints)

        for i, (game, expected) in enumerate(zip(games, test_video.expected_games)):
            if expected.player1_character:
                assert game.player1_character == expected.player1_character, (
                    f"Game {i + 1}: Expected P1 '{expected.player1_character}', "
                    f"got '{game.player1_character}' for {test_video.video_id}"
                )
            if expected.player2_character:
                assert game.player2_character == expected.player2_character, (
                    f"Game {i + 1}: Expected P2 '{expected.player2_character}', "
                    f"got '{game.player2_character}' for {test_video.video_id}"
                )


class TestStageDetection:
    """Unit tests for stage detection."""

    def test_battlefield_vs_small_battlefield_disambiguation(self):
        """Test that platform counting distinguishes Battlefield from Small Battlefield."""
        from runitback.cv.stage_detector import detect_platform_count
        import cv2
        import numpy as np

        # This would require actual test images - placeholder for now
        # TODO: Add test images for stage detection
        pass


class TestStockLossDetection:
    """Tests for stock loss detection using Gemini stock counter."""

    # Filter to only videos with stock loss data
    VIDEOS_WITH_STOCK_DATA = [v for v in TEST_VIDEOS if any(g.stock_losses for g in v.expected_games)]

    @pytest.mark.slow
    @pytest.mark.parametrize("test_video", VIDEOS_WITH_STOCK_DATA, ids=lambda v: v.video_id)
    def test_stock_counter_detection(self, test_video: TestVideo):
        """Test that stock counter splash screens are correctly detected."""
        from runitback.video.downloader import download_video
        from runitback.video.frame_extractor import extract_frames
        from runitback.cv.llm_detector import detect_stock_counter_with_gemini
        from runitback.config import settings

        # Download video
        video_path = download_video(test_video.url, cache_dir=settings.cache_dir)

        # Extract frames at 0.5s interval
        frames = list(extract_frames(video_path, interval_sec=0.5))

        # Test each expected stock loss
        for game in test_video.expected_games:
            if not game.stock_losses:
                continue

            for expected_loss in game.stock_losses:
                expected_time = timestamp_to_seconds(expected_loss.timestamp)

                # Find frames around the expected kill time (+1 to +3 seconds after)
                # Stock counter appears shortly after the kill
                detected = False
                detection_details = []

                for offset in range(0, 6):  # Check 0-3 seconds after expected time
                    target_time = expected_time + offset * 0.5

                    # Find closest frame
                    closest_frame = min(frames, key=lambda f: abs(f.timestamp - target_time))
                    if abs(closest_frame.timestamp - target_time) > 1:
                        continue

                    result = detect_stock_counter_with_gemini(closest_frame.image, use_cache=True)

                    if result.detected:
                        detected = True
                        if expected_loss.is_game_end:
                            # Should detect GAME! screen
                            if result.is_game_end:
                                detection_details.append(f"GAME! at {seconds_to_timestamp(closest_frame.timestamp)}")
                                break
                        else:
                            # Should detect stock counts
                            if result.p1_stocks is not None and result.p2_stocks is not None:
                                detection_details.append(
                                    f"P1={result.p1_stocks}, P2={result.p2_stocks} at "
                                    f"{seconds_to_timestamp(closest_frame.timestamp)}"
                                )
                                break

                assert detected, (
                    f"Stock loss at {seconds_to_timestamp(expected_time)} (P{expected_loss.player}) "
                    f"not detected for {test_video.video_id} Game {game.game_number}"
                )

    @pytest.mark.slow
    def test_stock_counter_no_false_positives(self):
        """Test that normal gameplay frames don't trigger stock counter detection."""
        from runitback.video.downloader import download_video
        from runitback.video.frame_extractor import extract_frames
        from runitback.cv.llm_detector import detect_stock_counter_with_gemini
        from runitback.config import settings

        # Use first test video
        test_video = TEST_VIDEOS[0]
        video_path = download_video(test_video.url, cache_dir=settings.cache_dir)
        frames = list(extract_frames(video_path, interval_sec=0.5))

        # Check frames that should NOT have stock counter (baseline gameplay)
        # These are times between kills where normal gameplay is happening
        baseline_times = [30, 45, 90, 120, 180]  # Times with no stock loss nearby

        false_positives = 0
        for baseline_time in baseline_times:
            # Find closest frame
            closest_frame = min(frames, key=lambda f: abs(f.timestamp - baseline_time))
            if abs(closest_frame.timestamp - baseline_time) > 2:
                continue

            result = detect_stock_counter_with_gemini(closest_frame.image, use_cache=True)
            if result.detected:
                false_positives += 1

        # Allow at most 1 false positive (could be near an actual kill)
        assert false_positives <= 1, (
            f"Too many false positives ({false_positives}) in baseline gameplay"
        )


# Convenience function to run slow tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow", "--tb=short"])
