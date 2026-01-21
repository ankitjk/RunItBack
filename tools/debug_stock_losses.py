#!/usr/bin/env python3
"""Debug script to investigate stock loss detection around specific timestamps."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from runitback.cv.scene_classifier import SceneClassifier, SceneState
from runitback.video.downloader import download_video
from runitback.video.frame_extractor import extract_frames


def main():
    """Investigate stock losses in Game 2 of Ken vs King Dedede."""

    # Game 2 timestamps (video time)
    # Detected P1 stock losses: 302.5s (5:02), 365.0s (6:05), 404.0s (6:44), 473.5s (7:53)
    # False positive is reportedly at 6:02 (362s)
    # Real stock losses for P1 (Ken): should be 3
    # Real stock losses for P2 (Dedede): should be 2

    url = "https://www.youtube.com/watch?v=7oPdiBsgSG0"

    # Download video
    print("Downloading video...")
    cache_dir = Path("/tmp/runitback_cache")
    video_path = download_video(url, cache_dir)
    print(f"Downloaded: {video_path}")

    # Extract frames - get all frames in the video
    print("\nExtracting frames...")
    frames = list(extract_frames(video_path, interval_sec=0.5))
    print(f"Extracted {len(frames)} frames")

    # Create frame lookup by timestamp
    frame_by_ts = {}
    for f in frames:
        frame_by_ts[round(f.timestamp * 2) / 2] = f  # Round to nearest 0.5s

    # Load scene classifier
    print("\nLoading scene classifier...")
    classifier = SceneClassifier()
    if not classifier.available:
        print("ERROR: Scene classifier not available!")
        return

    print("\n" + "="*80)
    print("SCENE CLASSIFICATION AROUND STOCK LOSS TIMESTAMPS")
    print("="*80)

    # Timestamps around detected stock losses
    # Game 2 starts at ~253s (4:13)
    stock_loss_detections = [
        (302.5, "P1 loss #1 (5:02)"),
        (365.0, "P1 loss #2 (6:05) - SUSPECTED FALSE POSITIVE at 6:02"),
        (404.0, "P1 loss #3 (6:44)"),
        (473.5, "P1 loss #4 (7:53) - REMOVED by limit"),
    ]

    for loss_time, description in stock_loss_detections:
        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")

        # Check frames from 3s before to 8s after the detection
        for offset in range(-6, 17, 1):  # -3s to +8s in 0.5s steps
            check_ts = loss_time + (offset * 0.5)
            frame = frame_by_ts.get(check_ts)

            if frame is None:
                continue

            result = classifier.classify(frame.image)
            respawn_prob = result.raw_probabilities.get("respawning", 0)

            # Highlight respawning detections
            highlight = ""
            if result.state == SceneState.RESPAWNING:
                highlight = " <<< RESPAWNING DETECTED"
            elif respawn_prob > 0.15:
                highlight = f" (respawn: {respawn_prob:.1%})"

            # Mark the detection timestamp
            marker = " ***" if abs(check_ts - loss_time) < 0.3 else ""

            mins = int(check_ts // 60)
            secs = check_ts % 60
            print(f"  {mins}:{secs:05.2f} ({check_ts:6.1f}s): {result.state.value:18s} {result.confidence:5.1%}{highlight}{marker}")

    print("\n" + "="*80)
    print("RESPAWN DETECTION SUMMARY")
    print("="*80)

    for loss_time, description in stock_loss_detections:
        print(f"\n{description}:")

        # Look for respawning within 5s after the loss
        respawn_found = False
        for offset in range(1, 11):  # 0.5s to 5s after
            check_ts = loss_time + (offset * 0.5)
            frame = frame_by_ts.get(check_ts)
            if frame is None:
                continue

            result = classifier.classify(frame.image)
            if result.state == SceneState.RESPAWNING:
                mins = int(check_ts // 60)
                secs = check_ts % 60
                print(f"  RESPAWN at {mins}:{secs:04.1f} ({offset * 0.5:.1f}s after detection)")
                respawn_found = True
                break

        if not respawn_found:
            print(f"  NO RESPAWN DETECTED within 5s - MAY BE FALSE POSITIVE!")


if __name__ == "__main__":
    main()
