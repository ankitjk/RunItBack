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

    # Re-split already-extracted frames, holding out game 2 as the test set
    python tools/extract_labeled_frames.py --splits-only --test-game 2

Splits are made by game (default): one whole game is the test set, and the end
of each other game is the validation set. Frames 0.5s apart are near-duplicates,
so a random frame split (--split-by frame, kept for comparison) leaks and
overstates accuracy.
"""

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_segments import SCENE_CLASSES, is_gold  # noqa: E402  (shared with the label tool and training)

# Rare classes get more frequent sampling
RARE_CLASSES = {
    "both_offstage",
    "both_on_ledge",
    "one_knocked_down",
    "both_knocked_down",
    "respawning",  # Short duration, needs more samples
    "game_end",  # A few seconds per game
    "finishing_blow",  # Under a second, at most once or twice per game
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
    stale_frames_removed: int = 0

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


def remove_video_frames(output_dir: Path, video_id: str) -> int:
    """Delete every frame previously extracted from one video. Returns the count.

    Frames are named by their label, so after a label edit the old frames would
    otherwise stay filed under their old class - and the splits are built from
    everything in the folder. Other videos' frames are left alone.
    """
    removed = 0
    if not output_dir.exists():
        return 0
    for class_dir in output_dir.iterdir():
        if not class_dir.is_dir():
            continue
        for path in class_dir.glob("*.jpg"):
            try:
                info = parse_frame_name(path, class_dir.name)
            except ValueError:
                continue  # not a frame this script wrote
            if info.video_id == video_id:
                path.unlink()
                removed += 1
    return removed


def extract_frames_from_video(
    video_path: Path,
    labels: dict,
    output_dir: Path,
    stats: ExtractionStats,
) -> None:
    """Extract frames from a video based on labels, replacing its earlier frames."""
    if is_gold(labels):
        # Gold videos are the test set: training on one would make its score
        # an exam the model had already seen (plan step 4).
        stats.errors.append(f"{video_path.name}: a gold (test-only) video; never extracted for training")
        return
    # A class this script does not know used to be skipped without a word, so a
    # typo or a class added to the label tool but not here vanished from training.
    unknown = sorted({
        str(label.get("scene_class", label.get("class")))
        for game in labels.get("games", [])
        for label in game.get("labels", [])
        if label.get("scene_class", label.get("class")) not in SCENE_CLASSES
    })
    if unknown:
        stats.errors.append(f"{video_path.name}: unknown scene classes {unknown}; nothing extracted")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        stats.errors.append(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    video_id = labels.get("video_id", video_path.stem)
    print(f"  Video: {video_path.name} ({duration:.1f}s, {fps:.1f} FPS)")

    # Only once the video is known to open, so a bad path never wipes good frames.
    removed = remove_video_frames(output_dir, video_id)
    stats.stale_frames_removed += removed
    if removed:
        print(f"  Removed {removed} frames from an earlier extraction")

    frames_extracted = 0

    for game in labels.get("games", []):
        game_num = game.get("game_number", 1)

        for label in game.get("labels", []):
            scene_class = label.get("scene_class", label.get("class"))

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


SPLITS = ("train", "val", "test")

# Seconds of training frames dropped just before each validation block, so no
# training frame is a near-duplicate of a validation frame (game split only).
DEFAULT_VAL_GAP_SECS = 5.0


@dataclass(frozen=True)
class FrameInfo:
    """An extracted frame, parsed from its {video_id}_g{game}_{time_ms}_{class}.jpg name."""

    path: Path
    video_id: str
    game: int
    time_sec: float
    scene_class: str

    @property
    def group(self) -> str:
        """The game this frame came from - the unit a game split keeps together."""
        return f"{self.video_id}:{self.game}"


def parse_frame_name(path: Path, scene_class: str) -> FrameInfo:
    """Parse a frame file name written by extract_frames_from_video.

    Splits from the right, because YouTube IDs can themselves contain "_".
    """
    suffix = f"_{scene_class}.jpg"
    if not path.name.endswith(suffix):
        raise ValueError(f"{path.name}: expected to end with {suffix}")
    stem = path.name[: -len(suffix)]
    video_and_game, time_ms = stem.rsplit("_", 1)
    video_id, game = video_and_game.rsplit("_g", 1)
    return FrameInfo(path, video_id, int(game), int(time_ms) / 1000, scene_class)


def collect_frames(frames_dir: Path) -> list[FrameInfo]:
    """Every extracted frame under frames_dir/<class>/, parsed."""
    frames = []
    for class_name in SCENE_CLASSES:
        class_dir = frames_dir / class_name
        if class_dir.exists():
            frames.extend(parse_frame_name(p, class_name) for p in sorted(class_dir.glob("*.jpg")))
    return frames


def resolve_test_game(groups: list[str], test_game: str | None) -> str:
    """Turn a --test-game value ("3" or "VIDEO_ID:3") into a group key.

    A bare game number is allowed only when one video is labeled. With no value,
    the last game of the last video is held out.
    """
    if test_game is None:
        return groups[-1]
    if ":" in test_game:
        key = test_game
    else:
        videos = sorted({g.rsplit(":", 1)[0] for g in groups})
        if len(videos) != 1:
            raise ValueError(
                f"{len(videos)} videos are labeled; give --test-game as VIDEO_ID:GAME. Games: {', '.join(groups)}"
            )
        key = f"{videos[0]}:{int(test_game)}"
    if key not in groups:
        raise ValueError(f"No frames for game {key}. Games: {', '.join(groups)}")
    return key


def resolve_test_video(groups: list[str], test_video: str) -> list[str]:
    """Every game of one video, for holding out a whole video as the test set."""
    test_keys = [g for g in groups if g.rsplit(":", 1)[0] == test_video]
    if not test_keys:
        videos = sorted({g.rsplit(":", 1)[0] for g in groups})
        raise ValueError(f"No frames for video {test_video}. Videos: {', '.join(videos)}")
    if len(test_keys) == len(groups):
        raise ValueError(
            f"{test_video} is the only labeled video; holding it out leaves nothing to train on. "
            "Hold out a game instead (--test-game)."
        )
    return test_keys


def assign_game_splits(
    frames: list[FrameInfo],
    test_game: str | None = None,
    val_ratio: float = 0.15,
    val_gap_secs: float = DEFAULT_VAL_GAP_SECS,
    test_video: str | None = None,
) -> tuple[dict[str, list[FrameInfo]], dict]:
    """Split frames by game, so near-duplicate neighbouring frames never cross splits.

    - test: every frame of one held-out game, or of every game of one held-out
      video (`test_video`). Evaluated once, after training.
    - val: the last `val_ratio` of each remaining game's labeled time span. Used
      to choose the checkpoint, so it must not be a test game.
    - train: the rest, minus frames within `val_gap_secs` before a val block.

    Returns:
        (frames by split, a manifest describing the split for split_manifest.json)
    """
    if test_game is not None and test_video is not None:
        raise ValueError("Hold out a game or a video, not both.")
    groups = sorted({f.group for f in frames}, key=lambda g: (g.rsplit(":", 1)[0], int(g.rsplit(":", 1)[1])))
    if len(groups) < 2:
        raise ValueError(f"A game split needs at least 2 labeled games; found {len(groups)}.")
    if test_video is not None:
        test_keys = resolve_test_video(groups, test_video)
    else:
        test_keys = [resolve_test_game(groups, test_game)]

    splits: dict[str, list[FrameInfo]] = {s: [] for s in SPLITS}
    dropped = 0
    val_blocks = {}
    for group in groups:
        group_frames = [f for f in frames if f.group == group]
        if group in test_keys:
            splits["test"].extend(group_frames)
            continue
        start = min(f.time_sec for f in group_frames)
        end = max(f.time_sec for f in group_frames)
        val_start = end - (end - start) * val_ratio
        val_blocks[group] = [round(val_start, 2), round(end, 2)]
        for f in group_frames:
            if f.time_sec >= val_start:
                splits["val"].append(f)
            elif f.time_sec >= val_start - val_gap_secs:
                dropped += 1
            else:
                splits["train"].append(f)

    manifest = {
        "split_by": "game",
        "held_out": "video" if test_video is not None else "game",
        "test_games": test_keys,
        "train_games": [g for g in groups if g not in test_keys],
        "val_blocks_sec": val_blocks,
        "val_ratio": val_ratio,
        "val_gap_secs": val_gap_secs,
        "dropped_gap_frames": dropped,
    }
    return splits, manifest


def assign_frame_splits(
    frames: list[FrameInfo],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[dict[str, list[FrameInfo]], dict]:
    """The original per-class random frame split.

    Leaks: frames 0.5s apart are near-duplicates and land in different splits,
    so its scores measure memorization. Kept only to compare against.
    """
    rng = random.Random(seed)
    splits: dict[str, list[FrameInfo]] = {s: [] for s in SPLITS}
    for class_name in SCENE_CLASSES:
        class_frames = [f for f in frames if f.scene_class == class_name]
        rng.shuffle(class_frames)
        n = len(class_frames)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        splits["train"].extend(class_frames[:train_end])
        splits["val"].extend(class_frames[train_end:val_end])
        splits["test"].extend(class_frames[val_end:])
    manifest = {"split_by": "frame", "train_ratio": train_ratio, "val_ratio": val_ratio, "seed": seed}
    return splits, manifest


def create_train_val_test_splits(
    frames_dir: Path,
    splits_dir: Path,
    split_by: str = "game",
    test_game: str | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    val_gap_secs: float = DEFAULT_VAL_GAP_SECS,
    seed: int = 42,
    test_video: str | None = None,
) -> dict:
    """Copy extracted frames into splits_dir/{train,val,test}/<class>/.

    Also writes splits_dir/split_manifest.json, recording how the split was made.

    Returns:
        Dictionary with split statistics: {split: {class: frame count}}
    """
    frames = collect_frames(frames_dir)
    if split_by == "game":
        splits, manifest = assign_game_splits(frames, test_game, val_ratio, val_gap_secs, test_video)
    elif split_by == "frame":
        splits, manifest = assign_frame_splits(frames, train_ratio, val_ratio, seed)
    else:
        raise ValueError(f"split_by must be 'game' or 'frame', not {split_by!r}")

    if splits_dir.exists():
        shutil.rmtree(splits_dir)

    split_stats: dict[str, defaultdict[str, int]] = {s: defaultdict(int) for s in SPLITS}
    for split, split_frames in splits.items():
        (splits_dir / split).mkdir(parents=True, exist_ok=True)
        for f in split_frames:
            class_dir = splits_dir / split / f.scene_class
            class_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f.path, class_dir / f.path.name)
            split_stats[split][f.scene_class] += 1

    manifest["frames"] = {s: dict(sorted(split_stats[s].items())) for s in SPLITS}
    with open(splits_dir / "split_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    return split_stats


def print_split_statistics(split_stats: dict, splits_dir: Path) -> None:
    """Print frames per class in each split, and how the split was made."""
    with open(splits_dir / "split_manifest.json") as fh:
        manifest = json.load(fh)
    print("\n" + "-" * 60)
    print(f"Split by {manifest['split_by']}")
    if manifest["split_by"] == "game":
        print(f"  test (held-out {manifest['held_out']}): {', '.join(manifest['test_games'])}")
        print(f"  train/val:       {', '.join(manifest['train_games'])}")
        print(f"  dropped {manifest['dropped_gap_frames']} frames in the {manifest['val_gap_secs']}s gap before val")
    else:
        print("  WARNING: frame split leaks near-duplicate frames; its scores overstate accuracy")
    print(f"\n  {'class':<20} {'train':>6} {'val':>6} {'test':>6}")
    for class_name in SCENE_CLASSES:
        counts = [split_stats[s].get(class_name, 0) for s in SPLITS]
        if any(counts):
            print(f"  {class_name:<20} {counts[0]:>6} {counts[1]:>6} {counts[2]:>6}")
    totals = [sum(split_stats[s].values()) for s in SPLITS]
    print(f"  {'total':<20} {totals[0]:>6} {totals[1]:>6} {totals[2]:>6}")
    print(f"\nManifest: {splits_dir / 'split_manifest.json'}")


def extract_all(label_files: list[Path], output_dir: Path) -> ExtractionStats:
    """Extract frames for every label file into output_dir/<class>/.

    Each video's earlier frames are replaced (see remove_video_frames). A label
    file that cannot be read, or whose video is missing, is recorded in
    stats.errors and skipped.
    """
    print(f"Found {len(label_files)} label file(s)")
    print(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = ExtractionStats()
    for label_path in label_files:
        print(f"\nProcessing: {label_path.name}")
        try:
            labels = load_labels(label_path)
        except Exception as e:
            stats.errors.append(f"Could not load {label_path}: {e}")
            continue

        video_path = find_video_for_labels(label_path, labels)
        if not video_path:
            stats.errors.append(f"Could not find video for {label_path.name}")
            continue

        extract_frames_from_video(video_path, labels, output_dir, stats)
    return stats


def print_statistics(stats: ExtractionStats):
    """Print extraction statistics."""
    print("\n" + "=" * 60)
    print("EXTRACTION STATISTICS")
    print("=" * 60)

    print(f"\nVideos processed: {stats.videos_processed}")
    print(f"Total frames extracted: {stats.total_frames}")
    if stats.stale_frames_removed:
        print(f"Frames from earlier extractions replaced: {stats.stale_frames_removed}")

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
        "--split-by",
        choices=["game", "frame"],
        default="game",
        help="game: hold out a whole game as the test set (default). "
        "frame: the old random frame split, which leaks near-duplicate frames - for comparison only",
    )
    held_out = parser.add_mutually_exclusive_group()
    held_out.add_argument(
        "--test-game",
        help="Game to hold out as the test set: '3', or 'VIDEO_ID:3' when several videos "
        "are labeled. Default: the last game",
    )
    held_out.add_argument(
        "--test-video",
        help="Hold out every game of this video as the test set (needs another labeled video)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Ratio of frames for training set (--split-by frame only)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Share of each game's time span used for validation (game split), "
        "or of each class's frames (frame split)",
    )
    parser.add_argument(
        "--val-gap",
        type=float,
        default=DEFAULT_VAL_GAP_SECS,
        help="Seconds of training frames dropped before each validation block (game split only)",
    )
    parser.add_argument(
        "--splits-only",
        action="store_true",
        help="Re-split frames already in --output-dir without extracting from video again",
    )
    args = parser.parse_args()

    if args.splits_only:
        print(f"Re-splitting frames in {args.output_dir} (split by {args.split_by})")
        split_stats = create_train_val_test_splits(
            args.output_dir,
            args.splits_dir,
            split_by=args.split_by,
            test_game=args.test_game,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            val_gap_secs=args.val_gap,
            test_video=args.test_video,
        )
        print_split_statistics(split_stats, args.splits_dir)
        return

    # Collect label files to process
    if args.label_file:
        label_files = [args.label_file]
    else:
        label_files = list(args.labels_dir.glob("*.json"))

    if not label_files:
        print("No label files found!")
        return

    stats = extract_all(label_files, args.output_dir)

    # Create train/val/test splits
    split_stats = None
    if not args.no_splits and stats.total_frames > 0:
        print(f"\nCreating train/val/test splits (split by {args.split_by})...")
        split_stats = create_train_val_test_splits(
            args.output_dir,
            args.splits_dir,
            split_by=args.split_by,
            test_game=args.test_game,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            val_gap_secs=args.val_gap,
            test_video=args.test_video,
        )

    # Print statistics
    print_statistics(stats)
    if split_stats:
        print_split_statistics(split_stats, args.splits_dir)


if __name__ == "__main__":
    main()
