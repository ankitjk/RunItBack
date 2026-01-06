#!/usr/bin/env python3
"""Extract labeled frames from videos for scene classifier training.

Reads label JSON files created by scene_labeler.py and extracts frames
at regular intervals, organizing them into class-specific directories.

Usage:
    python tools/extract_labeled_frames.py [--labels-dir DIR] [--output-dir DIR]
    python tools/extract_labeled_frames.py --label-file path/to/labels.json

Examples:
    # Process all labels in default directory
    python tools/extract_labeled_frames.py

    # Process a single label file
    python tools/extract_labeled_frames.py --label-file data/scene_classifier/raw_labels/abc123.json

    # Custom directories
    python tools/extract_labeled_frames.py --labels-dir my_labels/ --output-dir my_frames/
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Scene classes (must match scene_labeler.py)
SCENE_CLASSES = [
    "non_gameplay",
    "both_grounded",
    "one_airborne",
    "both_airborne",
    "one_offstage",
    "both_offstage",
    "one_on_ledge",
    "both_on_ledge",
    "one_knocked_down",
    "both_knocked_down",
    "respawning",
]

# Rare classes get more frequent sampling
RARE_CLASSES = {
    "both_offstage",
    "both_on_ledge",
    "one_knocked_down",
    "both_knocked_down",
    "respawning",  # Short duration, needs more samples
}

# Sampling intervals (seconds)
DEFAULT_INTERVAL = 0.5  # Standard classes
RARE_INTERVAL = 0.25    # Rare classes (2x more frames)

# Edge padding to avoid transition frames
# At 60 FPS, 0.15s = 9 frames of safety margin from segment boundaries
EDGE_PADDING = 0.15


@dataclass
class ExtractionStats:
    """Statistics for frame extraction."""
    total_frames: int = 0
    frames_per_class: dict = None
    videos_processed: int = 0
    errors: list = None

    def __post_init__(self):
        if self.frames_per_class is None:
            self.frames_per_class = defaultdict(int)
        if self.errors is None:
            self.errors = []


def load_labels(label_path: Path) -> dict:
    """Load labels from JSON file."""
    with open(label_path) as f:
        return json.load(f)


def find_video_for_labels(label_path: Path, labels: dict) -> Path | None:
    """Find the video file corresponding to the labels.

    Searches in:
    1. .video_cache/ directory (for YouTube downloads)
    2. Same directory as labels file
    3. Parent directories
    """
    video_id = labels.get("video_id", label_path.stem)

    # Check .video_cache first (YouTube downloads)
    cache_path = Path(".video_cache") / f"{video_id}.mp4"
    if cache_path.exists():
        return cache_path

    # Check same directory as labels
    same_dir = label_path.parent / f"{video_id}.mp4"
    if same_dir.exists():
        return same_dir

    # Check common video extensions
    for ext in [".mp4", ".mkv", ".webm", ".mov"]:
        for search_dir in [Path(".video_cache"), label_path.parent, Path(".")]:
            video_path = search_dir / f"{video_id}{ext}"
            if video_path.exists():
                return video_path

    return None


def extract_frames_from_video(
    video_path: Path,
    labels: dict,
    output_dir: Path,
    stats: ExtractionStats,
) -> None:
    """Extract frames from a video based on labels."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        stats.errors.append(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    video_id = labels.get("video_id", video_path.stem)
    print(f"  Video: {video_path.name} ({duration:.1f}s, {fps:.1f} FPS)")

    frames_extracted = 0

    for game in labels.get("games", []):
        game_num = game.get("game_number", 1)
        stage = game.get("stage", "unknown")

        for label in game.get("labels", []):
            scene_class = label.get("scene_class", label.get("class"))
            if not scene_class or scene_class not in SCENE_CLASSES:
                continue

            start_sec = label.get("start_sec", 0)
            end_sec = label.get("end_sec", 0)

            if end_sec <= start_sec:
                continue

            # Determine sampling interval
            interval = RARE_INTERVAL if scene_class in RARE_CLASSES else DEFAULT_INTERVAL

            # Apply edge padding to avoid transition frames
            # This ensures we only sample frames well within the labeled segment
            padded_start = start_sec + EDGE_PADDING
            padded_end = end_sec - EDGE_PADDING

            # Skip segments too short to sample safely
            if padded_end <= padded_start:
                # Segment is shorter than 2x padding (~0.3s)
                # Sample single frame from the middle instead
                t = (start_sec + end_sec) / 2
                frame_num = int(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    time_ms = int(t * 1000)
                    filename = f"{video_id}_g{game_num}_{time_ms:06d}_{scene_class}.jpg"
                    class_dir = output_dir / scene_class
                    class_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(class_dir / filename), frame)
                    frames_extracted += 1
                    stats.frames_per_class[scene_class] += 1
                    stats.total_frames += 1
                continue

            # Extract frames at regular intervals from padded region
            t = padded_start
            while t < padded_end:
                frame_num = int(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if ret:
                    # Create output filename
                    # Format: {video_id}_g{game}_{time_ms}_{class}.jpg
                    time_ms = int(t * 1000)
                    filename = f"{video_id}_g{game_num}_{time_ms:06d}_{scene_class}.jpg"

                    # Save to class directory
                    class_dir = output_dir / scene_class
                    class_dir.mkdir(parents=True, exist_ok=True)

                    output_path = class_dir / filename
                    cv2.imwrite(str(output_path), frame)

                    frames_extracted += 1
                    stats.frames_per_class[scene_class] += 1
                    stats.total_frames += 1

                t += interval

    cap.release()
    stats.videos_processed += 1
    print(f"  Extracted {frames_extracted} frames")


def create_train_val_test_splits(
    frames_dir: Path,
    splits_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Create train/val/test splits from extracted frames.

    Returns:
        Dictionary with split statistics
    """
    random.seed(seed)

    split_stats = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int),
    }

    # Clear existing splits
    if splits_dir.exists():
        shutil.rmtree(splits_dir)

    for split in ["train", "val", "test"]:
        (splits_dir / split).mkdir(parents=True, exist_ok=True)

    # Process each class
    for class_name in SCENE_CLASSES:
        class_dir = frames_dir / class_name
        if not class_dir.exists():
            continue

        # Get all frames for this class
        frames = list(class_dir.glob("*.jpg"))
        random.shuffle(frames)

        # Calculate split indices
        n = len(frames)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split frames
        train_frames = frames[:train_end]
        val_frames = frames[train_end:val_end]
        test_frames = frames[val_end:]

        # Create class directories in each split
        for split, split_frames in [
            ("train", train_frames),
            ("val", val_frames),
            ("test", test_frames),
        ]:
            split_class_dir = splits_dir / split / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for frame_path in split_frames:
                # Copy (or symlink) to split directory
                dest_path = split_class_dir / frame_path.name
                shutil.copy2(frame_path, dest_path)
                split_stats[split][class_name] += 1

    return split_stats


def print_statistics(stats: ExtractionStats, split_stats: dict | None = None):
    """Print extraction statistics."""
    print("\n" + "=" * 60)
    print("EXTRACTION STATISTICS")
    print("=" * 60)

    print(f"\nVideos processed: {stats.videos_processed}")
    print(f"Total frames extracted: {stats.total_frames}")

    print("\nFrames per class:")
    print("-" * 40)

    # Sort by count descending
    sorted_classes = sorted(
        stats.frames_per_class.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for class_name, count in sorted_classes:
        pct = count / stats.total_frames * 100 if stats.total_frames > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {class_name:<20} {count:>6} ({pct:>5.1f}%) {bar}")

    # Classes with no frames
    missing = set(SCENE_CLASSES) - set(stats.frames_per_class.keys())
    if missing:
        print(f"\nMissing classes (no frames): {', '.join(missing)}")

    if split_stats:
        print("\n" + "-" * 40)
        print("Split statistics:")
        for split in ["train", "val", "test"]:
            total = sum(split_stats[split].values())
            print(f"  {split}: {total} frames")

    if stats.errors:
        print("\nErrors:")
        for error in stats.errors:
            print(f"  - {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract labeled frames for scene classifier training",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/scene_classifier/raw_labels"),
        help="Directory containing label JSON files",
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        help="Process a single label file instead of directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scene_classifier/frames"),
        help="Output directory for extracted frames",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/scene_classifier/splits"),
        help="Output directory for train/val/test splits",
    )
    parser.add_argument(
        "--no-splits",
        action="store_true",
        help="Skip creating train/val/test splits",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Ratio of frames for training set",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Ratio of frames for validation set",
    )
    args = parser.parse_args()

    # Collect label files to process
    if args.label_file:
        label_files = [args.label_file]
    else:
        label_files = list(args.labels_dir.glob("*.json"))

    if not label_files:
        print("No label files found!")
        return

    print(f"Found {len(label_files)} label file(s)")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each label file
    stats = ExtractionStats()

    for label_path in label_files:
        print(f"\nProcessing: {label_path.name}")

        try:
            labels = load_labels(label_path)
        except Exception as e:
            stats.errors.append(f"Could not load {label_path}: {e}")
            continue

        # Find corresponding video
        video_path = find_video_for_labels(label_path, labels)
        if not video_path:
            stats.errors.append(f"Could not find video for {label_path.name}")
            continue

        # Extract frames
        extract_frames_from_video(video_path, labels, args.output_dir, stats)

    # Create train/val/test splits
    split_stats = None
    if not args.no_splits and stats.total_frames > 0:
        print("\nCreating train/val/test splits...")
        split_stats = create_train_val_test_splits(
            args.output_dir,
            args.splits_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

    # Print statistics
    print_statistics(stats, split_stats)


if __name__ == "__main__":
    main()
