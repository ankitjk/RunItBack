"""Scene-classifier dataset splits - game-level split and class re-indexing.

Frames 0.5s apart are near-duplicates, so a split that lets neighbouring frames
land in different splits measures memorization. These tests pin down that the
game split keeps every game whole and leaves a gap before validation. They use
empty placeholder files, so no video or model is needed.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load_tool(name: str):
    """Import a script from tools/ (not a package) by path."""
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


elf = _load_tool("extract_labeled_frames")


def _make_frames(frames_dir: Path, video_id: str, game: int, start: float, end: float, scene_class: str):
    """Write placeholder frames every 0.5s, named as extract_frames_from_video names them."""
    class_dir = frames_dir / scene_class
    class_dir.mkdir(parents=True, exist_ok=True)
    t = start
    while t < end:
        (class_dir / f"{video_id}_g{game}_{int(t * 1000):06d}_{scene_class}.jpg").write_bytes(b"")
        t += 0.5


@pytest.fixture
def three_games(tmp_path):
    """One video, three 100s games, each with two classes."""
    frames_dir = tmp_path / "frames"
    for game, offset in [(1, 0), (2, 200), (3, 400)]:
        _make_frames(frames_dir, "XUkRx6CpL04", game, offset, offset + 60, "both_grounded")
        _make_frames(frames_dir, "XUkRx6CpL04", game, offset + 60, offset + 100, "one_airborne")
    return frames_dir


def test_parses_frame_names_with_underscores_in_video_id():
    info = elf.parse_frame_name(Path("a_b-c_g9_g2_012345_one_on_ledge.jpg"), "one_on_ledge")
    assert (info.video_id, info.game, info.time_sec) == ("a_b-c_g9", 2, 12.345)
    assert info.group == "a_b-c_g9:2"


def test_game_split_keeps_each_game_whole(three_games):
    splits, manifest = elf.assign_game_splits(elf.collect_frames(three_games), test_game="2")

    assert {f.group for f in splits["test"]} == {"XUkRx6CpL04:2"}
    train_val_groups = {f.group for f in splits["train"] + splits["val"]}
    assert train_val_groups == {"XUkRx6CpL04:1", "XUkRx6CpL04:3"}
    assert manifest["test_game"] == "XUkRx6CpL04:2"
    # The held-out game is complete, not a sample of it.
    assert len(splits["test"]) == 200


def test_validation_is_separated_from_training_by_the_gap(three_games):
    splits, _ = elf.assign_game_splits(elf.collect_frames(three_games), test_game="3", val_gap_secs=5.0)

    for group in ("XUkRx6CpL04:1", "XUkRx6CpL04:2"):
        train_times = [f.time_sec for f in splits["train"] if f.group == group]
        val_times = [f.time_sec for f in splits["val"] if f.group == group]
        assert train_times and val_times
        assert min(val_times) - max(train_times) > 5.0


def test_default_test_game_is_the_last_one(three_games):
    _, manifest = elf.assign_game_splits(elf.collect_frames(three_games))
    assert manifest["test_game"] == "XUkRx6CpL04:3"


def test_unknown_test_game_is_rejected(three_games):
    with pytest.raises(ValueError, match="No frames for game"):
        elf.assign_game_splits(elf.collect_frames(three_games), test_game="7")


def test_bare_game_number_is_ambiguous_with_several_videos(three_games):
    _make_frames(three_games, "otherVideo1", 1, 0, 30, "both_grounded")
    frames = elf.collect_frames(three_games)
    with pytest.raises(ValueError, match="VIDEO_ID:GAME"):
        elf.assign_game_splits(frames, test_game="1")
    splits, _ = elf.assign_game_splits(frames, test_game="otherVideo1:1")
    assert {f.video_id for f in splits["test"]} == {"otherVideo1"}


def test_one_game_cannot_be_game_split(tmp_path):
    _make_frames(tmp_path, "XUkRx6CpL04", 1, 0, 30, "both_grounded")
    with pytest.raises(ValueError, match="at least 2"):
        elf.assign_game_splits(elf.collect_frames(tmp_path))


def test_create_splits_writes_folders_and_manifest(three_games, tmp_path):
    splits_dir = tmp_path / "splits"
    stats = elf.create_train_val_test_splits(three_games, splits_dir, test_game="1")

    manifest = json.loads((splits_dir / "split_manifest.json").read_text())
    assert manifest["split_by"] == "game"
    assert manifest["test_game"] == "XUkRx6CpL04:1"
    for split in elf.SPLITS:
        on_disk = len(list((splits_dir / split).rglob("*.jpg")))
        assert on_disk == sum(stats[split].values()) == sum(manifest["frames"][split].values())
    test_files = list((splits_dir / "test").rglob("*.jpg"))
    assert all("_g1_" in p.name for p in test_files)


def test_frame_split_is_still_available_for_comparison(three_games, tmp_path):
    splits_dir = tmp_path / "splits"
    elf.create_train_val_test_splits(three_games, splits_dir, split_by="frame")
    manifest = json.loads((splits_dir / "split_manifest.json").read_text())
    assert manifest["split_by"] == "frame"
    # The leak this split has: every game appears in the test set.
    test_names = [p.name for p in (splits_dir / "test").rglob("*.jpg")]
    assert {n.split("_g")[1][0] for n in test_names} == {"1", "2", "3"}


def test_reindex_by_name_aligns_a_split_missing_a_class(tmp_path):
    """A val/test split without a training class must not shift the other labels."""
    pytest.importorskip("torchvision")
    tsc = _load_tool("train_scene_classifier")
    from torchvision import datasets

    for split, classes in [("train", ["both_grounded", "one_airborne", "respawning"]),
                           # No both_grounded; two classes training never saw.
                           ("test", ["one_airborne", "respawning", "both_on_ledge", "one_on_ledge"])]:
        for c in classes:
            (tmp_path / split / c).mkdir(parents=True)
            (tmp_path / split / c / f"x_g1_000000_{c}.jpg").write_bytes(b"")

    train = datasets.ImageFolder(tmp_path / "train")
    test = datasets.ImageFolder(tmp_path / "test")
    # Without re-indexing, "respawning" is label 2 in train but 3 in test.
    assert (train.class_to_idx["respawning"], test.class_to_idx["respawning"]) == (2, 3)

    dropped = tsc.reindex_by_name(test, train.classes)

    assert dropped == {"both_on_ledge": 1, "one_on_ledge": 1}
    labels = {Path(p).parent.name: t for p, t in test.samples}
    assert labels == {"one_airborne": 1, "respawning": 2}
    assert test.targets == [1, 2]


def _write_test_video(path: Path, seconds: float = 4.0, fps: int = 10) -> None:
    """A small synthetic video, so extraction can run without a real broadcast."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(int(seconds * fps)):
        writer.write(np.full((48, 64, 3), i % 255, dtype=np.uint8))
    writer.release()


def _labels(video_id: str, segments: list[tuple[float, float, str]]) -> dict:
    return {
        "video_id": video_id,
        "games": [{"game_number": 1, "labels": [{"start_sec": a, "end_sec": b, "class": c} for a, b, c in segments]}],
    }


def test_reextraction_after_a_label_edit_leaves_no_stale_frames(tmp_path):
    """ISSUE-021: frames from the old labels must not survive a re-extraction."""
    video = tmp_path / "vid.mp4"
    _write_test_video(video)
    frames = tmp_path / "frames"

    elf.extract_frames_from_video(video, _labels("vidA", [(0.0, 4.0, "non_gameplay")]), frames, elf.ExtractionStats())
    assert list((frames / "non_gameplay").glob("*.jpg"))

    # Relabel the second half, as the GAME! relabel did.
    relabelled = _labels("vidA", [(0.0, 2.0, "non_gameplay"), (2.0, 4.0, "respawning")])
    stats = elf.ExtractionStats()
    elf.extract_frames_from_video(video, relabelled, frames, stats)

    on_disk = elf.collect_frames(frames)
    assert stats.stale_frames_removed > 0
    assert len(on_disk) == stats.total_frames
    assert all(f.time_sec < 2.0 for f in on_disk if f.scene_class == "non_gameplay")
    assert {f.scene_class for f in on_disk} == {"non_gameplay", "respawning"}


def test_reextraction_keeps_other_videos_frames(tmp_path):
    video = tmp_path / "vid.mp4"
    _write_test_video(video)
    frames = tmp_path / "frames"
    # Another video whose ID starts with this one's - must not be touched.
    _make_frames(frames, "vidA_g2x", 1, 0, 3, "both_grounded")
    other = sorted(p.name for p in frames.rglob("*.jpg"))

    elf.extract_frames_from_video(video, _labels("vidA", [(0.0, 4.0, "non_gameplay")]), frames, elf.ExtractionStats())
    elf.extract_frames_from_video(video, _labels("vidA", [(0.0, 4.0, "non_gameplay")]), frames, elf.ExtractionStats())

    assert sorted(p.name for p in (frames / "both_grounded").glob("*.jpg")) == other


def test_unopenable_video_does_not_delete_existing_frames(tmp_path):
    frames = tmp_path / "frames"
    _make_frames(frames, "vidA", 1, 0, 3, "both_grounded")
    before = len(list(frames.rglob("*.jpg")))
    stats = elf.ExtractionStats()

    elf.extract_frames_from_video(tmp_path / "missing.mp4", _labels("vidA", []), frames, stats)

    assert stats.errors and len(list(frames.rglob("*.jpg"))) == before


def test_onnx_export_accepts_any_batch_size(tmp_path):
    """ISSUE-020: the exported model must run on batches larger than 1."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("torchvision")
    tsc = _load_tool("train_scene_classifier")

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 5)
    )
    path = tmp_path / "model.onnx"
    (tmp_path / "model.onnx.data").write_bytes(b"stale")

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    tsc.export_onnx(model.to(device), path, input_size=32)  # raises if a batch of 3 fails

    assert path.exists() and not (tmp_path / "model.onnx.data").exists()
