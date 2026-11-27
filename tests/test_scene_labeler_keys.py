"""scene_labeler.py driven with real key presses, off-screen, on a generated 4-second video.

Covers the key handling that label_segments.py cannot: S then a class key labels
from the mark (it used to start at the playhead, ISSUE-024), labelling over
existing labels replaces them (it used to stack a second label, ISSUE-025), and
Ctrl+Z restores the file exactly. Skipped where PyQt5 is not installed.
"""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from tests.test_label_segments import _load_tool  # noqa: E402

FPS = 60


@pytest.fixture
def labeler(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 36))
    for i in range(4 * FPS):
        out.write(np.full((36, 64, 3), i % 256, dtype=np.uint8))
    out.release()
    labels = tmp_path / "clip.json"
    labels.write_text(json.dumps({
        "video_id": "clip", "video_path": str(video),
        "games": [{"game_number": 1, "stage": "unknown", "start_sec": 0.0, "end_sec": 4.0, "labels": [
            {"start_sec": 0.0, "end_sec": 2.0, "class": "both_grounded", "needs_review": True},
            {"start_sec": 2.0, "end_sec": 4.0, "class": "one_airborne"},
        ]}],
    }))

    app = QApplication.instance() or QApplication([])
    sl = _load_tool("scene_labeler")
    window = sl.SceneLabeler(str(video), str(labels))
    window.show()
    QApplication.setActiveWindow(window)
    app.processEvents()
    if window.player.total_frames < 3 * FPS:
        pytest.skip("this OpenCV build cannot write mp4v video")

    def press(key, modifier=Qt.NoModifier, times=1):
        for _ in range(times):
            QTest.keyClick(window, key, modifier)
            app.processEvents()

    def spans():
        return [(s.start_sec, s.end_sec, s.scene_class) for s in window._current_game.labels]

    yield window, press, spans
    window.player.cleanup()
    window.deleteLater()


def test_mark_then_class_key_labels_from_the_mark_and_replaces(labeler):
    window, press, spans = labeler
    window.player.seek_frame(90)       # 1.5s
    press(Qt.Key_S)
    press(Qt.Key_Right, times=60)      # 2.5s
    press(Qt.Key_R)
    assert spans() == [(0.0, 1.5, "both_grounded"), (1.5, 2.5, "respawning"), (2.5, 4.0, "one_airborne")]
    assert window._current_game.labels[0].extra == {"needs_review": True}, "the draft's field is kept"
    assert window.segment_start is None, "the mark is used up"


def test_a_refused_mark_edit_does_not_leave_the_mark_behind(labeler):
    window, press, spans = labeler
    window.player.seek_frame(30)
    press(Qt.Key_S)
    press(Qt.Key_Right, times=6)
    press(Qt.Key_1)                    # already both_grounded: refused
    assert window.segment_start is None
    before = spans()
    window.player.seek_frame(150)
    press(Qt.Key_3)                    # must start a continuous label, not overwrite 0.5s-2.5s
    assert spans() == before and window.current_class is not None


def test_continuous_labelling_over_labels_replaces_them(labeler):
    window, press, spans = labeler
    window.player.seek_frame(60)       # 1.0s
    press(Qt.Key_4)
    press(Qt.Key_Right, times=90)      # 2.5s
    press(Qt.Key_E)
    assert spans() == [(0.0, 1.0, "both_grounded"), (1.0, 2.5, "one_offstage"), (2.5, 4.0, "one_airborne")]


def test_cut_change_move_delete_and_undo_back_to_the_file(labeler):
    window, press, spans = labeler
    original = window.video_labels.to_dict()
    window.player.seek_frame(60)
    press(Qt.Key_K)                    # cut at 1.0s
    press(Qt.Key_C)
    press(Qt.Key_6)                    # the 1.0-2.0 piece becomes one_on_ledge
    press(Qt.Key_Right, times=30)
    press(Qt.Key_BracketRight)         # its end moves to 1.5s; one_airborne follows
    assert spans() == [(0.0, 1.0, "both_grounded"), (1.0, 1.5, "one_on_ledge"), (1.5, 4.0, "one_airborne")]
    press(Qt.Key_Delete)
    assert len(spans()) == 2
    while window.undo_stack:
        press(Qt.Key_Z, Qt.ControlModifier)
    assert window.video_labels.to_dict() == original
