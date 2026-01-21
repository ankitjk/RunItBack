#!/usr/bin/env python3
"""Test stock loss detection accuracy across all test videos."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set Gemini API key
os.environ["GEMINI_API_KEY"] = "AIzaSyCcaIbIFQffymC1kh9ovsAs75dURRBrxMs"

from dataclasses import dataclass
from typing import Optional

from runitback.video.downloader import download_video, get_video_info, extract_characters_from_title
from runitback.video.frame_extractor import extract_frames
from runitback.cli import detect_games
from runitback.classification.detector import GameStateDetector
from runitback.config import settings


@dataclass
class ExpectedStockLoss:
    """Expected stock loss event."""
    timestamp: tuple[int, int]  # (minutes, seconds)
    player: int  # 1 or 2
    is_game_end: bool = False
    timestamp_tolerance: int = 5  # seconds


@dataclass
class ExpectedGame:
    """Expected game data."""
    game_number: int
    stock_losses: list[ExpectedStockLoss]


@dataclass
class TestVideo:
    """Test video with expected data."""
    url: str
    video_id: str
    description: str
    expected_games: list[ExpectedGame]


# Test videos with stock loss ground truth
TEST_VIDEOS = [
    TestVideo(
        url="https://www.youtube.com/watch?v=7oPdiBsgSG0",
        video_id="7oPdiBsgSG0",
        description="Ken vs King Dedede (2 games)",
        expected_games=[
            ExpectedGame(
                game_number=1,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(1, 8), player=1),
                    ExpectedStockLoss(timestamp=(2, 40), player=1),
                    ExpectedStockLoss(timestamp=(3, 31), player=1, is_game_end=True),
                ],
            ),
            ExpectedGame(
                game_number=2,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(5, 5), player=1),
                    ExpectedStockLoss(timestamp=(5, 16), player=2),
                    ExpectedStockLoss(timestamp=(6, 34), player=2),
                    ExpectedStockLoss(timestamp=(6, 43), player=1),
                    ExpectedStockLoss(timestamp=(7, 52), player=1, is_game_end=True),
                ],
            ),
        ],
    ),
    TestVideo(
        url="https://www.youtube.com/watch?v=fRcif5Qa2Is",
        video_id="fRcif5Qa2Is",
        description="Ken vs Sonic (3 games)",
        expected_games=[
            ExpectedGame(
                game_number=1,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(1, 20), player=2),
                    ExpectedStockLoss(timestamp=(1, 42), player=1),
                    ExpectedStockLoss(timestamp=(2, 52), player=1),
                    ExpectedStockLoss(timestamp=(3, 49), player=2),
                    ExpectedStockLoss(timestamp=(3, 58), player=1, is_game_end=True),
                ],
            ),
            ExpectedGame(
                game_number=2,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(5, 34), player=2),
                    ExpectedStockLoss(timestamp=(6, 51), player=1),
                    ExpectedStockLoss(timestamp=(7, 38), player=2),
                    ExpectedStockLoss(timestamp=(9, 12), player=2, is_game_end=True),
                ],
            ),
            ExpectedGame(
                game_number=3,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(10, 49), player=2),
                    ExpectedStockLoss(timestamp=(11, 56), player=2),
                    ExpectedStockLoss(timestamp=(12, 28), player=1),
                    ExpectedStockLoss(timestamp=(12, 48), player=2, is_game_end=True),
                ],
            ),
        ],
    ),
    TestVideo(
        url="https://www.youtube.com/watch?v=St7XW7w_jaA",
        video_id="St7XW7w_jaA",
        description="Dark Pit vs Ken (4 games)",
        expected_games=[
            ExpectedGame(
                game_number=1,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(0, 33), player=1),
                    ExpectedStockLoss(timestamp=(1, 29), player=1),
                    ExpectedStockLoss(timestamp=(1, 46), player=2),
                    ExpectedStockLoss(timestamp=(2, 40), player=1, is_game_end=True),
                ],
            ),
            ExpectedGame(
                game_number=2,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(3, 51), player=1),
                    ExpectedStockLoss(timestamp=(4, 49), player=1),
                    ExpectedStockLoss(timestamp=(4, 54), player=2),
                    ExpectedStockLoss(timestamp=(5, 34), player=2),
                    ExpectedStockLoss(timestamp=(6, 39), player=2, is_game_end=True),
                ],
            ),
            ExpectedGame(
                game_number=3,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(7, 53), player=1),
                    ExpectedStockLoss(timestamp=(8, 41), player=2),
                    ExpectedStockLoss(timestamp=(8, 47), player=1),
                    ExpectedStockLoss(timestamp=(9, 56), player=2),
                    ExpectedStockLoss(timestamp=(10, 15), player=1, is_game_end=True),
                ],
            ),
            ExpectedGame(
                game_number=4,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(11, 38), player=1),
                    ExpectedStockLoss(timestamp=(12, 33), player=2),
                    ExpectedStockLoss(timestamp=(12, 48), player=1),
                    ExpectedStockLoss(timestamp=(13, 25), player=1, is_game_end=True),
                ],
            ),
        ],
    ),
    TestVideo(
        url="https://www.youtube.com/watch?v=nmzdyQz6msI",
        video_id="nmzdyQz6msI",
        description="Steve vs Palutena (5 games)",
        expected_games=[
            ExpectedGame(
                game_number=1,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(1, 29), player=2),
                    ExpectedStockLoss(timestamp=(1, 39), player=1),
                    ExpectedStockLoss(timestamp=(2, 9), player=2),
                    ExpectedStockLoss(timestamp=(2, 38), player=2, is_game_end=True),
                ],
            ),
            ExpectedGame(
                game_number=2,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(4, 34), player=1),
                    ExpectedStockLoss(timestamp=(4, 45), player=2),
                    ExpectedStockLoss(timestamp=(5, 35), player=2),
                    ExpectedStockLoss(timestamp=(5, 49), player=1),
                    ExpectedStockLoss(timestamp=(7, 10), player=1, is_game_end=True),
                ],
            ),
            ExpectedGame(
                game_number=3,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(9, 10), player=1),
                    ExpectedStockLoss(timestamp=(9, 20), player=2),
                    ExpectedStockLoss(timestamp=(10, 53), player=1),
                    ExpectedStockLoss(timestamp=(11, 24), player=2),
                    ExpectedStockLoss(timestamp=(11, 45), player=1, is_game_end=True),
                ],
            ),
            ExpectedGame(
                game_number=4,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(14, 9), player=1),
                    ExpectedStockLoss(timestamp=(14, 17), player=2),
                    ExpectedStockLoss(timestamp=(15, 58), player=1),
                    ExpectedStockLoss(timestamp=(16, 18), player=2),
                    ExpectedStockLoss(timestamp=(16, 54), player=2, is_game_end=True),
                ],
            ),
            ExpectedGame(
                game_number=5,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(19, 31), player=2),
                    ExpectedStockLoss(timestamp=(19, 55), player=1),
                    ExpectedStockLoss(timestamp=(21, 12), player=1),
                    ExpectedStockLoss(timestamp=(22, 25), player=1, is_game_end=True),
                ],
            ),
        ],
    ),
    TestVideo(
        url="https://www.youtube.com/watch?v=KfMZ9bQUbtk",
        video_id="KfMZ9bQUbtk",
        description="Cloud vs Pit (3 games)",
        expected_games=[
            ExpectedGame(
                game_number=1,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(0, 23), player=1),
                    ExpectedStockLoss(timestamp=(1, 47), player=1),
                    ExpectedStockLoss(timestamp=(2, 14), player=2),
                    ExpectedStockLoss(timestamp=(3, 25), player=1, is_game_end=True),
                ],
            ),
            ExpectedGame(
                game_number=2,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(4, 51), player=2),
                    ExpectedStockLoss(timestamp=(5, 53), player=2),
                    ExpectedStockLoss(timestamp=(6, 16), player=1),
                    ExpectedStockLoss(timestamp=(7, 22), player=1),
                    ExpectedStockLoss(timestamp=(7, 31), player=2, is_game_end=True),
                ],
            ),
            ExpectedGame(
                game_number=3,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(9, 18), player=2),
                    ExpectedStockLoss(timestamp=(10, 19), player=1),
                    ExpectedStockLoss(timestamp=(11, 47), player=2),
                    ExpectedStockLoss(timestamp=(11, 55), player=1),
                    ExpectedStockLoss(timestamp=(12, 40), player=2, is_game_end=True),
                ],
            ),
        ],
    ),
    TestVideo(
        url="https://www.youtube.com/watch?v=UgGQkF1wkoA",
        video_id="UgGQkF1wkoA",
        description="Mario vs Ness (3 games)",
        expected_games=[
            ExpectedGame(
                game_number=1,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(0, 59), player=1),
                    ExpectedStockLoss(timestamp=(1, 31), player=2),
                    ExpectedStockLoss(timestamp=(2, 11), player=1),
                    ExpectedStockLoss(timestamp=(2, 56), player=2),
                    ExpectedStockLoss(timestamp=(3, 46), player=2, is_game_end=True),
                ],
            ),
            ExpectedGame(
                game_number=2,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(5, 33), player=2),
                    ExpectedStockLoss(timestamp=(6, 21), player=1),
                    ExpectedStockLoss(timestamp=(6, 28), player=2),
                    ExpectedStockLoss(timestamp=(7, 17), player=2, is_game_end=True),
                ],
            ),
            ExpectedGame(
                game_number=3,
                stock_losses=[
                    ExpectedStockLoss(timestamp=(8, 39), player=2),
                    ExpectedStockLoss(timestamp=(8, 56), player=1),
                    ExpectedStockLoss(timestamp=(9, 25), player=2),
                    ExpectedStockLoss(timestamp=(10, 47), player=2, is_game_end=True),
                ],
            ),
        ],
    ),
]


def timestamp_to_seconds(ts: tuple[int, int]) -> float:
    """Convert (min, sec) to total seconds."""
    return ts[0] * 60 + ts[1]


def seconds_to_timestamp(secs: float) -> str:
    """Convert seconds to M:SS string."""
    m = int(secs // 60)
    s = int(secs % 60)
    return f"{m}:{s:02d}"


def analyze_video(test_video: TestVideo) -> dict:
    """Analyze a single video and compare with expected stock losses."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {test_video.description}")
    print(f"URL: {test_video.url}")
    print(f"{'='*60}")

    # Download video
    print("Downloading video...")
    video_path = download_video(test_video.url, cache_dir=settings.cache_dir)

    # Get character hints from title
    try:
        video_info = get_video_info(test_video.url)
        p1_hints, p2_hints = extract_characters_from_title(video_info.get("title", ""))
    except Exception:
        p1_hints, p2_hints = None, None

    # Extract frames
    print("Extracting frames...")
    frames = list(extract_frames(video_path, interval_sec=0.5))
    print(f"Extracted {len(frames)} frames")

    # Detect games
    print("Detecting games...")
    games = detect_games(frames, p1_hints=p1_hints, p2_hints=p2_hints)
    print(f"Found {len(games)} game(s)")

    # Analyze each game
    detector = GameStateDetector(
        player_perspective=1,
        hit_threshold=4.0,
        neutral_reset_time=3.0,
        use_smart_gemini=True,
    )

    results = {
        "video_id": test_video.video_id,
        "expected_games": len(test_video.expected_games),
        "detected_games": len(games),
        "game_results": [],
        "total_expected": 0,
        "total_detected": 0,
        "total_matched": 0,
        "total_missed": 0,
        "total_false_positives": 0,
    }

    for game_idx, game in enumerate(games):
        if game_idx >= len(test_video.expected_games):
            print(f"\nGame {game_idx + 1}: EXTRA (not expected)")
            continue

        expected = test_video.expected_games[game_idx]
        print(f"\nGame {game_idx + 1}:")
        print(f"  Game time: {seconds_to_timestamp(game.start_timestamp)} - {seconds_to_timestamp(game.end_timestamp)}")

        # Get frames for this game
        game_frames = [f for f in frames
                      if game.start_timestamp <= f.timestamp <= game.end_timestamp]

        # Run state detection
        print(f"  Analyzing {len(game_frames)} frames...")
        analysis = detector.detect_states(
            game_frames,
            frame_interval=0.5,
            game_end_times=[game.end_timestamp],
        )

        # Collect detected stock losses
        detected_losses = []
        for fs in analysis.frame_states:
            if fs.p1_stock_loss:
                detected_losses.append((fs.timestamp, 1))
            if fs.p2_stock_loss:
                detected_losses.append((fs.timestamp, 2))

        # Sort by timestamp
        detected_losses.sort(key=lambda x: x[0])

        print(f"  Expected stock losses: {len(expected.stock_losses)}")
        print(f"  Detected stock losses: {len(detected_losses)}")

        # Match expected vs detected
        matched = 0
        missed = []
        false_positives = list(detected_losses)

        for exp_loss in expected.stock_losses:
            exp_time = timestamp_to_seconds(exp_loss.timestamp)
            exp_player = exp_loss.player

            # Find matching detection
            match_found = False
            for det_time, det_player in false_positives:
                if det_player == exp_player and abs(det_time - exp_time) <= exp_loss.timestamp_tolerance:
                    matched += 1
                    false_positives.remove((det_time, det_player))
                    match_found = True
                    break

            if not match_found:
                missed.append((exp_time, exp_player, exp_loss.is_game_end))

        # Print results
        print(f"  Matched: {matched}/{len(expected.stock_losses)}")

        if missed:
            print(f"  MISSED ({len(missed)}):")
            for m_time, m_player, is_end in missed:
                end_str = " (GAME!)" if is_end else ""
                print(f"    - P{m_player} at {seconds_to_timestamp(m_time)}{end_str}")

        if false_positives:
            print(f"  FALSE POSITIVES ({len(false_positives)}):")
            for fp_time, fp_player in false_positives:
                print(f"    - P{fp_player} at {seconds_to_timestamp(fp_time)}")

        # Store game results
        game_result = {
            "game_number": game_idx + 1,
            "expected": len(expected.stock_losses),
            "detected": len(detected_losses),
            "matched": matched,
            "missed": len(missed),
            "false_positives": len(false_positives),
        }
        results["game_results"].append(game_result)
        results["total_expected"] += len(expected.stock_losses)
        results["total_detected"] += len(detected_losses)
        results["total_matched"] += matched
        results["total_missed"] += len(missed)
        results["total_false_positives"] += len(false_positives)

    return results


def main():
    """Run stock loss detection test on all videos."""
    print("="*60)
    print("STOCK LOSS DETECTION TEST")
    print("="*60)

    all_results = []

    for test_video in TEST_VIDEOS:
        try:
            result = analyze_video(test_video)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR analyzing {test_video.video_id}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total_expected = sum(r["total_expected"] for r in all_results)
    total_matched = sum(r["total_matched"] for r in all_results)
    total_missed = sum(r["total_missed"] for r in all_results)
    total_fp = sum(r["total_false_positives"] for r in all_results)

    print(f"\nVideos analyzed: {len(all_results)}")
    print(f"Total expected stock losses: {total_expected}")
    print(f"Total matched: {total_matched}")
    print(f"Total missed: {total_missed}")
    print(f"Total false positives: {total_fp}")

    if total_expected > 0:
        recall = total_matched / total_expected * 100
        print(f"\nRecall: {recall:.1f}% ({total_matched}/{total_expected})")

    if total_matched + total_fp > 0:
        precision = total_matched / (total_matched + total_fp) * 100
        print(f"Precision: {precision:.1f}% ({total_matched}/{total_matched + total_fp})")

    print("\nPer-video breakdown:")
    for r in all_results:
        video_recall = r["total_matched"] / r["total_expected"] * 100 if r["total_expected"] > 0 else 0
        print(f"  {r['video_id']}: {r['total_matched']}/{r['total_expected']} matched ({video_recall:.0f}%), {r['total_false_positives']} FP")


if __name__ == "__main__":
    main()
