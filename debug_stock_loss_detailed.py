#!/usr/bin/env python3
"""Detailed debug script to trace stock loss detection algorithm."""

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

print("Extracting frames from video...")
frames = list(extract_frames(video_path, interval_sec=0.5))
print(f"Total frames: {len(frames)}")

# Simulate the stock loss detection algorithm with detailed logging
STOCK_LOSS_COOLDOWN = 10.0
CONFIRM_FRAMES = 3  # Updated from 2 to 3
MIN_KILL_DAMAGE = 50  # Minimum damage before stock loss is plausible

prev_p1_damage = None
prev_p2_damage = None
p1_recent_max_damage = 0.0
p2_recent_max_damage = 0.0
last_p1_stock_loss_time = None
last_p2_stock_loss_time = None
p1_low_damage_frames = 0
p2_low_damage_frames = 0
p1_high_damage_before_low = None
p2_high_damage_before_low = None

detected_stock_losses = []

print("\n" + "="*80)
print("TRACING STOCK LOSS DETECTION ALGORITHM")
print("="*80)

# Focus on frames around problem areas
focus_ranges = [
    (70, 100),   # 35-50s: around 0:40-0:44 false positive
    (126, 150),  # 63-75s: around 1:08 expected stock
    (160, 190),  # 80-95s: around 1:26 false positive
    (308, 340),  # 154-170s: around 2:40 missed stock
    (410, 440),  # 205-220s: around 3:31 expected stock
]

def in_focus(idx):
    return any(start <= idx <= end for start, end in focus_ranges)

for i, frame in enumerate(frames):
    p1_roi = get_roi(frame.image, P1_DAMAGE_ROI)
    p2_roi = get_roi(frame.image, P2_DAMAGE_ROI)

    p1_damage, p1_conf = read_damage_percentage(p1_roi)
    p2_damage, p2_conf = read_damage_percentage(p2_roi)

    timestamp = frame.timestamp
    mins = int(timestamp) // 60
    secs = int(timestamp) % 60
    time_str = f"{mins}:{secs:02d}"

    # Track algorithm state
    old_p1_low = p1_low_damage_frames
    old_p2_low = p2_low_damage_frames
    old_p1_high = p1_high_damage_before_low
    old_p2_high = p2_high_damage_before_low

    cooldown_p1_ok = (
        last_p1_stock_loss_time is None or
        timestamp - last_p1_stock_loss_time >= STOCK_LOSS_COOLDOWN
    )
    cooldown_p2_ok = (
        last_p2_stock_loss_time is None or
        timestamp - last_p2_stock_loss_time >= STOCK_LOSS_COOLDOWN
    )

    # Update recent max damage tracking (only valid damage values)
    if p1_damage is not None and 10 < p1_damage < 200:
        p1_recent_max_damage = max(p1_recent_max_damage, p1_damage)
    if p2_damage is not None and 10 < p2_damage < 200:
        p2_recent_max_damage = max(p2_recent_max_damage, p2_damage)

    # Track P1 low damage frames (stock loss = 0%, allow 0-2% for OCR variance)
    if p1_damage is not None and p1_damage <= 2:
        if p1_low_damage_frames == 0:
            p1_high_damage_before_low = p1_recent_max_damage
        p1_low_damage_frames += 1
    else:
        p1_low_damage_frames = 0
        p1_high_damage_before_low = None
        # Don't decay max - keep tracking until stock loss confirmed

    # Track P2 low damage frames (stock loss = 0%, allow 0-2% for OCR variance)
    if p2_damage is not None and p2_damage <= 2:
        if p2_low_damage_frames == 0:
            p2_high_damage_before_low = p2_recent_max_damage
        p2_low_damage_frames += 1
    else:
        p2_low_damage_frames = 0
        p2_high_damage_before_low = None
        # Don't decay max - keep tracking until stock loss confirmed

    # Check for stock loss
    p1_stock_loss = False
    p2_stock_loss = False

    if (p1_low_damage_frames == CONFIRM_FRAMES and
        p1_high_damage_before_low is not None and
        p1_high_damage_before_low >= MIN_KILL_DAMAGE and cooldown_p1_ok):
        p1_stock_loss = True
        last_p1_stock_loss_time = timestamp
        p1_recent_max_damage = 0.0  # Reset after stock loss
        detected_stock_losses.append(('P1', timestamp, time_str, p1_high_damage_before_low))

    if (p2_low_damage_frames == CONFIRM_FRAMES and
        p2_high_damage_before_low is not None and
        p2_high_damage_before_low >= MIN_KILL_DAMAGE and cooldown_p2_ok):
        p2_stock_loss = True
        last_p2_stock_loss_time = timestamp
        p2_recent_max_damage = 0.0  # Reset after stock loss
        detected_stock_losses.append(('P2', timestamp, time_str, p2_high_damage_before_low))

    # Print details for focused frames
    if in_focus(i):
        dmg_p1 = f"{p1_damage:.1f}%" if p1_damage is not None else "None"
        dmg_p2 = f"{p2_damage:.1f}%" if p2_damage is not None else "None"
        prev_p1 = f"{prev_p1_damage:.1f}" if prev_p1_damage is not None else "None"
        prev_p2 = f"{prev_p2_damage:.1f}" if prev_p2_damage is not None else "None"
        high_p1 = f"{p1_high_damage_before_low:.1f}" if p1_high_damage_before_low is not None else "None"
        high_p2 = f"{p2_high_damage_before_low:.1f}" if p2_high_damage_before_low is not None else "None"

        # Highlight stock loss
        marker = ""
        if p1_stock_loss:
            marker = " *** P1 STOCK LOSS ***"
        if p2_stock_loss:
            marker = " *** P2 STOCK LOSS ***"

        print(f"[{i:3d}] {time_str} | P1: {dmg_p1:>7} (prev={prev_p1:>5}, low_frames={p1_low_damage_frames}, high_before={high_p1:>5}) | "
              f"P2: {dmg_p2:>7} (prev={prev_p2:>5}, low_frames={p2_low_damage_frames}, high_before={high_p2:>5}){marker}")

    # Update prev damage
    if p1_damage is not None:
        prev_p1_damage = p1_damage
    if p2_damage is not None:
        prev_p2_damage = p2_damage

print("\n" + "="*80)
print("DETECTED STOCK LOSSES:")
print("="*80)
for player, ts, time_str, prev_dmg in detected_stock_losses:
    print(f"  {player} at {time_str} ({ts:.1f}s) - prev damage was {prev_dmg:.1f}%")

print("\n" + "="*80)
print("EXPECTED STOCK LOSSES (Ground Truth):")
print("="*80)
print("  P1 (Ken) at 1:08, 2:40, 3:31")
print("  P2 (Dedede) - NONE")
