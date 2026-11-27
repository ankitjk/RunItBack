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


def _colour_at(strip, t):
    """The strip's colour at time t, sampled a third of the way down (clear of the overlap band)."""
    image = strip.grab().toImage()
    x = int(strip.x_of(t) * image.width() / strip.width())
    return image.pixelColor(x, image.height() // 3).name().upper()


def test_strips_show_segments_in_class_colours_and_follow_edits(labeler):
    window, press, spans = labeler
    sl = _load_tool("scene_labeler")
    strip = window.overview_strip
    window.player.seek_frame(0)
    assert _colour_at(strip, 1.0) == sl.CLASS_COLORS["both_grounded"].upper()
    assert _colour_at(strip, 3.0) == sl.CLASS_COLORS["one_airborne"].upper()

    window.player.seek_frame(150)      # 2.5s
    press(Qt.Key_S)
    press(Qt.Key_Right, times=60)      # 3.5s
    press(Qt.Key_R)
    assert _colour_at(strip, 3.0) == sl.CLASS_COLORS["respawning"].upper()
    assert _colour_at(window.closeup_strip, 3.0) == sl.CLASS_COLORS["respawning"].upper()


def test_clicking_a_strip_jumps_there_and_selects_the_segment(labeler):
    from PyQt5.QtCore import QPoint

    window, press, spans = labeler
    strip = window.overview_strip
    QTest.mouseClick(strip, Qt.LeftButton, pos=QPoint(int(strip.x_of(3.0)), strip.height() // 2))
    QApplication.processEvents()
    assert abs(window.player.get_current_time() - 3.0) < 0.1
    assert window.label_panel.selected_index() == 1
    press(Qt.Key_Delete)               # acts on the clicked segment
    assert spans() == [(0.0, 2.0, "both_grounded")]


def test_an_overlap_is_marked_on_the_strips(labeler):
    window, press, spans = labeler
    sl = _load_tool("scene_labeler")
    ls = _load_tool("label_segments")
    # A file written by hand or by an old version can overlap; the strips must show it
    window._current_game.labels = [ls.SegmentLabel(0.0, 2.1, "both_grounded"),
                                   ls.SegmentLabel(2.0, 4.0, "one_airborne")]
    window._update_ui()
    for strip in (window.overview_strip, window.closeup_strip):
        image = strip.grab().toImage()
        x = int(strip.x_of(2.05) * image.width() / strip.width())
        assert image.pixelColor(x, 1).name().upper() == sl.OVERLAP_COLOR.upper()


@pytest.fixture
def close_prompts(monkeypatch):
    """Record the save-before-closing prompt instead of showing it; answer Cancel."""
    from PyQt5.QtWidgets import QMessageBox

    asked = []

    def fake_question(parent, title, text, buttons=None, default=None):
        asked.append(text)
        return QMessageBox.Cancel

    monkeypatch.setattr(QMessageBox, "question", fake_question)
    return asked


def test_closing_without_changes_or_after_saving_does_not_ask(labeler, close_prompts):
    window, press, spans = labeler
    window.player.seek_frame(60)
    press(Qt.Key_K)
    assert window.windowTitle().endswith(" *"), "unsaved changes are shown in the title"
    press(Qt.Key_S, Qt.ControlModifier)
    assert not window.windowTitle().endswith(" *")
    assert window.close() and close_prompts == []


def test_closing_with_unsaved_changes_asks_and_cancel_keeps_the_window(labeler, close_prompts):
    window, press, spans = labeler
    window.player.seek_frame(60)
    press(Qt.Key_K)
    assert not window.close()
    assert len(close_prompts) == 1 and "unsaved changes" in close_prompts[0]


def test_undoing_back_to_the_saved_labels_counts_as_saved(labeler, close_prompts):
    window, press, spans = labeler
    window.player.seek_frame(60)
    press(Qt.Key_K)
    press(Qt.Key_Z, Qt.ControlModifier)
    assert window.close() and close_prompts == []


def test_closing_with_an_open_label_warns_that_it_is_not_saved(labeler, close_prompts):
    window, press, spans = labeler
    window.player.seek_frame(60)
    press(Qt.Key_3)                    # opens a continuous label; nothing written yet
    assert not window.close()
    assert len(close_prompts) == 1 and "still open" in close_prompts[0]


def test_the_label_tool_offers_exactly_the_shared_class_list(labeler):
    sl = _load_tool("scene_labeler")
    ls = _load_tool("label_segments")
    assert [state.value for state in sl.SceneState] == ls.SCENE_CLASSES
    assert [sl.SCENE_LABELS[i].value for i in sorted(sl.SCENE_LABELS)] == ls.SCENE_CLASSES
    assert set(sl.CLASS_COLORS) == set(ls.SCENE_CLASSES)


def test_p_x_and_f_label_the_new_classes(labeler):
    window, press, spans = labeler
    window.player.seek_frame(0)
    press(Qt.Key_S)
    press(Qt.Key_Right, times=30)      # 0.5s
    press(Qt.Key_P)
    press(Qt.Key_S)
    press(Qt.Key_Right, times=30)      # 1.0s
    press(Qt.Key_F)
    window.player.seek_frame(150)      # 2.5s
    press(Qt.Key_S)
    press(Qt.Key_Right, times=60)      # 3.5s
    press(Qt.Key_X)
    assert spans() == [(0.0, 0.5, "pre_game"), (0.5, 1.0, "finishing_blow"), (1.0, 2.0, "both_grounded"),
                       (2.0, 2.5, "one_airborne"), (2.5, 3.5, "game_end"), (3.5, 4.0, "one_airborne")]


def test_a_gold_video_is_marked_gold_and_saved_as_gold(tmp_path):
    app = QApplication.instance() or QApplication([])
    sl = _load_tool("scene_labeler")
    video = tmp_path / "clip.mp4"
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 36))
    for i in range(FPS):
        out.write(np.full((36, 64, 3), i, dtype=np.uint8))
    out.release()
    labels = tmp_path / "gold.json"

    window = sl.SceneLabeler(str(video), str(labels), gold=True)
    app.processEvents()
    assert window.windowTitle().startswith("[GOLD")
    window._save_labels()
    assert json.loads(labels.read_text())["role"] == "gold"
    window.player.cleanup()

    # Opened later without --gold, the file still says gold, and the tool shows it
    again = sl.SceneLabeler(str(video), str(labels))
    app.processEvents()
    assert again.gold and again.windowTitle().startswith("[GOLD")
    again.player.cleanup()
