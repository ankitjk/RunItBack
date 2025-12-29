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
class ExpectedGame:
    """Expected game data for test verification."""
    game_number: int
    stage: str
    start_time: tuple[int, int]  # (minutes, seconds)
    end_time: tuple[int, int]  # (minutes, seconds)
    player1_character: Optional[str] = None  # Player 1 character name
    player2_character: Optional[str] = None  # Player 2 character name
    # Tolerance for timestamp matching (seconds)
    timestamp_tolerance: int = 30


@dataclass
class TestVideo:
    """Test video with expected game data."""
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
            ),
            ExpectedGame(
                game_number=2,
                stage="hollow_bastion",
                start_time=(4, 7),
                end_time=(8, 5),
                player1_character="Ken",
                player2_character="King Dedede",
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
            ),
            ExpectedGame(
                game_number=2,
                stage="hollow_bastion",
                start_time=(4, 57),
                end_time=(9, 12),
                player1_character="Ken",
                player2_character="Sonic",
            ),
            ExpectedGame(
                game_number=3,
                stage="battlefield",
                start_time=(9, 53),
                end_time=(12, 49),
                player1_character="Ken",
                player2_character="Sonic",
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
            ),
            ExpectedGame(
                game_number=2,
                stage="pokemon_stadium_2",
                start_time=(3, 18),
                end_time=(6, 39),
                player1_character="Dark Pit",
                player2_character="Ken",
            ),
            ExpectedGame(
                game_number=3,
                stage="pokemon_stadium_2",
                start_time=(7, 21),
                end_time=(10, 15),
                player1_character="Dark Pit",
                player2_character="Ken",
            ),
            ExpectedGame(
                game_number=4,
                stage="pokemon_stadium_2",
                start_time=(10, 59),
                end_time=(13, 25),
                player1_character="Dark Pit",
                player2_character="Ken",
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
            ),
            ExpectedGame(
                game_number=2,
                stage="small_battlefield",
                start_time=(3, 18),
                end_time=(7, 10),
                player1_character="Steve",
                player2_character="Palutena",
            ),
            ExpectedGame(
                game_number=3,
                stage="small_battlefield",
                start_time=(7, 42),
                end_time=(11, 45),
                player1_character="Steve",
                player2_character="Palutena",
            ),
            ExpectedGame(
                game_number=4,
                stage="town_and_city",
                start_time=(12, 38),
                end_time=(16, 45),
                player1_character="Steve",
                player2_character="Palutena",
            ),
            ExpectedGame(
                game_number=5,
                stage="small_battlefield",
                start_time=(17, 28),
                end_time=(22, 25),
                player1_character="Steve",
                player2_character="Palutena",
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


# Convenience function to run slow tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow", "--tb=short"])
