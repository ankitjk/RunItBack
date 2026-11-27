#!/usr/bin/env python3
"""Build a scene-classifier model in one command: extract, split, train, test, export.

Each run gets its own folder, data/models/runs/<name>/, holding the ONNX model,
class_mapping.json, test_metrics.json, the split manifest, the training history
and run.json (what was trained on and how). The deployed model,
data/models/scene_classifier.onnx, is never touched: deploying a retrained model
(copying its ONNX file and class_mapping.json there) is a separate decision.

Usage:
    .venv/bin/python tools/build_scene_model.py [--test-game 3 | --test-video VIDEO_ID] [--name NAME]

Examples:
    # Hold out game 3 of the only labeled video
    .venv/bin/python tools/build_scene_model.py --test-game 3

    # Hold out a whole video (needs at least two labeled videos)
    .venv/bin/python tools/build_scene_model.py --test-video VIDEO_ID

    # Re-split and retrain on frames already extracted
    .venv/bin/python tools/build_scene_model.py --test-game 3 --reuse-frames
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extract_labeled_frames as elf  # noqa: E402
import train_scene_classifier as tsc  # noqa: E402

RUNS_DIR = Path("data/models/runs")


def git_state() -> dict:
    """The commit the code was at, and whether the working tree had changes."""
    def run(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {"commit": run("git", "rev-parse", "--short", "HEAD") or None,
            "uncommitted_changes": bool(run("git", "status", "--porcelain"))}


def describe_labels(label_files: list[Path]) -> list[dict]:
    """Each label file's hash and games, so a run can be traced to the exact labels."""
    described = []
    for path in sorted(label_files):
        data = path.read_bytes()
        labels = json.loads(data)
        described.append({
            "file": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "video_id": labels.get("video_id", path.stem),
            "games": [g.get("game_number") for g in labels.get("games", [])],
        })
    return described


def default_run_name(test_game: str | None, test_video: str | None) -> str:
    held_out = f"video-{test_video}" if test_video else f"game-{(test_game or 'last').replace(':', '-g')}"
    return f"{datetime.now():%Y-%m-%d_%H%M}_{held_out}"


def build(args: argparse.Namespace) -> Path:
    """Run the whole build. Returns the run folder."""
    run_dir = args.runs_dir / (args.name or default_run_name(args.test_game, args.test_video))
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists; choose another --name.")
    if (run_dir / tsc.DEPLOYED_MODEL.name).resolve() == tsc.DEPLOYED_MODEL.resolve():
        raise ValueError(f"{run_dir} is the deployed model's folder; choose another --runs-dir.")

    label_files = sorted(args.labels_dir.glob("*.json"))
    if not label_files:
        raise FileNotFoundError(f"No label files in {args.labels_dir}")

    extraction = None
    if args.reuse_frames:
        print(f"Reusing frames in {args.frames_dir} (not re-extracted from video)")
    else:
        stats = elf.extract_all(label_files, args.frames_dir)
        elf.print_statistics(stats)
        if stats.errors:
            # Training on part of the data while the totals look normal is the
            # failure ISSUE-008 describes for the benchmark; stop instead.
            raise RuntimeError("Extraction had errors; not training on partial data:\n  " + "\n  ".join(stats.errors))
        extraction = {
            "videos": stats.videos_processed,
            "frames": stats.total_frames,
            "frames_per_class": dict(sorted(stats.frames_per_class.items())),
            "stale_frames_replaced": stats.stale_frames_removed,
        }

    split_stats = elf.create_train_val_test_splits(
        args.frames_dir,
        args.splits_dir,
        split_by="game",
        test_game=args.test_game,
        test_video=args.test_video,
        val_ratio=args.val_ratio,
        val_gap_secs=args.val_gap,
    )
    elf.print_split_statistics(split_stats, args.splits_dir)

    run_dir.mkdir(parents=True)
    shutil.copy2(args.splits_dir / "split_manifest.json", run_dir / "split_manifest.json")

    summary = tsc.train_model(args.splits_dir, run_dir, **tsc.training_kwargs(args))

    # The checkpoints are large and the ONNX file holds the same weights.
    if not args.keep_checkpoints:
        for name in ("checkpoint_best.pt", "checkpoint_latest.pt"):
            (run_dir / name).unlink(missing_ok=True)

    test = summary["test_metrics"]
    run_info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git": git_state(),
        "labels": describe_labels(label_files),
        "frames_reused": bool(args.reuse_frames),
        "extraction": extraction,
        "split": json.loads((run_dir / "split_manifest.json").read_text()),
        "training": {k: v for k, v in summary.items() if k != "test_metrics"} | {
            "options": tsc.training_kwargs(args),
        },
        "test": None if test is None else {
            "accuracy": test["accuracy"],
            "macro_recall": test["macro_recall"],
            "frames": test["frames"],
        },
        "deployed": False,
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2))

    print("\n" + "=" * 60)
    print(f"Run folder: {run_dir}")
    if test is not None:
        print(f"Test accuracy {test['accuracy']:.1%}, macro recall {test['macro_recall']:.1%} "
              f"on {test['frames']} held-out frames ({', '.join(run_info['split']['test_games'])})")
    print(f"Deployed model unchanged: {tsc.DEPLOYED_MODEL}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract, split, train, test and export a scene-classifier model into its own run folder",
    )
    held_out = parser.add_mutually_exclusive_group()
    held_out.add_argument(
        "--test-game",
        help="Game to hold out as the test set: '3', or 'VIDEO_ID:3' when several videos are labeled. "
        "Default: the last game",
    )
    held_out.add_argument(
        "--test-video",
        help="Hold out every game of this video as the test set (needs another labeled video)",
    )
    parser.add_argument("--name", help="Run folder name under --runs-dir. Default: date, time and held-out set")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="Folder that holds run folders")
    parser.add_argument("--labels-dir", type=Path, default=Path("data/scene_classifier/raw_labels"))
    parser.add_argument("--frames-dir", type=Path, default=Path("data/scene_classifier/frames"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/scene_classifier/splits"))
    parser.add_argument(
        "--reuse-frames",
        action="store_true",
        help="Skip extraction and split the frames already in --frames-dir. Only safe if no label "
        "file changed since they were extracted",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Share of each game's time span used for validation")
    parser.add_argument("--val-gap", type=float, default=elf.DEFAULT_VAL_GAP_SECS,
                        help="Seconds of training frames dropped before each validation block")
    parser.add_argument("--keep-checkpoints", action="store_true", help="Keep the PyTorch checkpoints (~110 MB each)")
    tsc.add_training_args(parser)
    args = parser.parse_args()

    try:
        build(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
