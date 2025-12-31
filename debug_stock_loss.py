#!/usr/bin/env python3
"""Debug script to investigate stock loss detection issues."""

import sys
sys.path.insert(0, '/Users/ankit/Documents/GitHub/RunItBack')

from pathlib import Path
from runitback.video.downloader import download_video
from runitback.video.frame_extractor import extract_frames
from runitback.cv.ui_detector import get_roi, P1_DAMAGE_ROI, P2_DAMAGE_ROI
from runitback.cv.ocr import read_damage_percentage

# Download the video first
print("Downloading video...")
cache_dir = Path("/Users/ankit/Documents/GitHub/RunItBack/.video_cache")
cache_dir.mkdir(exist_ok=True)
video_path = download_video("https://www.youtube.com/watch?v=7oPdiBsgSG0", cache_dir)
print(f"Video: {video_path}")

# Game 1 starts at 0:18 (18s) and ends at 3:36 (216s)
# We want to analyze frames around the problem timestamps:
# - False positive at 0:40-0:44 (video time, game time ~22-26s from game start)
# - Expected stock loss at 1:08 (video time, game time ~50s from game start)
# - False positive at 1:26 (video time, game time ~68s from game start)
# - Missed stock loss at 2:40 (video time, game time ~142s from game start)
# - Expected stock loss at 3:31 (video time, game time ~193s from game start)

# Key video timestamps to investigate (in seconds)
timestamps_to_check = [
    # Around false positive at 0:40-0:44
    (38, 48, "False positive @ 0:40-0:44"),
    # Around expected stock loss at 1:08
    (65, 75, "Expected Ken stock @ 1:08"),
    # Around false positive at 1:26
    (83, 93, "False positive Dedede @ 1:26"),
    # Around missed stock loss at 2:40
    (157, 167, "Missed Ken stock @ 2:40"),
    # Around expected stock loss at 3:31
    (208, 218, "Expected Ken stock @ 3:31"),
]

print("Extracting frames from video...")
frames = list(extract_frames(video_path, interval_sec=0.5))
print(f"Total frames: {len(frames)}")

# Create timestamp -> frame index mapping
frame_by_time = {f.timestamp: i for i, f in enumerate(frames)}

for start_time, end_time, description in timestamps_to_check:
    print(f"\n{'='*60}")
    print(f"INVESTIGATING: {description}")
    print(f"Time range: {start_time}s - {end_time}s")
    print(f"{'='*60}")

    for t in range(start_time, end_time + 1):
        # Find closest frame
        closest_time = min(frame_by_time.keys(), key=lambda x: abs(x - t))
        if abs(closest_time - t) > 1:
            continue

        frame_idx = frame_by_time[closest_time]
        frame = frames[frame_idx]

        # Read damage
        p1_roi = get_roi(frame.image, P1_DAMAGE_ROI)
        p2_roi = get_roi(frame.image, P2_DAMAGE_ROI)

        p1_damage, p1_conf = read_damage_percentage(p1_roi)
        p2_damage, p2_conf = read_damage_percentage(p2_roi)

        # Format timestamp as MM:SS
        mins = int(closest_time) // 60
        secs = int(closest_time) % 60

        print(f"  {mins}:{secs:02d} (frame {frame_idx}): P1={p1_damage}% (conf={p1_conf:.2f}), P2={p2_damage}% (conf={p2_conf:.2f})")
