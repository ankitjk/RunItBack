"""Label edits used by tools/scene_labeler.py: overwrite, split, move an edge, change class, delete.

The rules these pin down: no edit leaves two segments covering the same time,
a refused edit changes nothing, and fields the tool does not use survive both
edits and a load-save round trip. No Qt or video needed.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
LABELS = Path(__file__).resolve().parent.parent / "data/scene_classifier/raw_labels"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ls = _load_tool("label_segments")
Seg = ls.SegmentLabel


def _spans(labels):
    return [(s.start_sec, s.end_sec, s.scene_class) for s in labels]


@pytest.fixture
def three():
    """airborne 0-10, grounded 10-20, offstage 20-30, contiguous."""
    return [Seg(0.0, 10.0, "one_airborne"), Seg(10.0, 20.0, "both_grounded"), Seg(20.0, 30.0, "one_offstage")]


# overwrite

def test_overwrite_inside_one_segment_cuts_it_around_the_new_one(three):
    out = ls.overwrite(three, 12.0, 15.0, "game_end")
    assert _spans(out) == [(0.0, 10.0, "one_airborne"), (10.0, 12.0, "both_grounded"),
                           (12.0, 15.0, "game_end"), (15.0, 20.0, "both_grounded"), (20.0, 30.0, "one_offstage")]


def test_overwrite_across_segments_trims_the_ends_and_removes_the_middle(three):
    out = ls.overwrite(three, 5.0, 25.0, "respawning")
    assert _spans(out) == [(0.0, 5.0, "one_airborne"), (5.0, 25.0, "respawning"), (25.0, 30.0, "one_offstage")]


def test_overwrite_accepts_the_span_in_either_order(three):
    assert _spans(ls.overwrite(three, 15.0, 12.0, "game_end")) == _spans(ls.overwrite(three, 12.0, 15.0, "game_end"))


def test_overwrite_into_unlabelled_time_just_adds():
    out = ls.overwrite([Seg(0.0, 5.0, "both_grounded")], 7.0, 9.0, "one_airborne")
    assert _spans(out) == [(0.0, 5.0, "both_grounded"), (7.0, 9.0, "one_airborne")]


def test_overwrite_never_leaves_an_overlap(three):
    out = three
    for start, end, cls in [(5, 25, "respawning"), (1, 2, "game_end"), (24, 29, "pre_game"), (0, 30, "one_airborne")]:
        out = ls.overwrite(out, start, end, cls)
        assert ls.overlaps(out) == []


def test_overwrite_of_an_empty_span_or_with_the_same_class_is_refused(three):
    with pytest.raises(ls.EditError):
        ls.overwrite(three, 12.0, 12.0, "game_end")
    with pytest.raises(ls.EditError, match="Already"):
        ls.overwrite(three, 12.0, 15.0, "both_grounded")


@pytest.mark.parametrize("cls,expected", [
    ("one_offstage", [(80.7, 84.6, "one_offstage"), (84.6, 86.2, "both_offstage")]),
    ("both_offstage", [(80.7, 84.5, "one_offstage"), (84.5, 86.2, "both_offstage")]),
])
def test_overwriting_an_overlap_with_either_class_fixes_it(cls, expected):
    # The shape of the real overlap in XUkRx6CpL04 game 1. It used to be refused
    # as "already that class" because one of the two segments already had it.
    labels = [Seg(80.7, 84.6, "one_offstage"), Seg(84.5, 86.2, "both_offstage")]
    out = ls.overwrite(labels, 84.5, 84.6, cls)
    assert ls.overlaps(out) == []
    assert _spans(out) == expected


def test_overwrite_joins_a_touching_segment_of_the_same_class(three):
    out = ls.overwrite(three, 8.0, 10.0, "both_grounded")
    assert _spans(out) == [(0.0, 8.0, "one_airborne"), (8.0, 20.0, "both_grounded"), (20.0, 30.0, "one_offstage")]


def test_overwrite_does_not_join_a_draft_segment():
    labels = [Seg(0.0, 5.0, "both_grounded", {"needs_review": True}), Seg(5.0, 9.0, "one_airborne")]
    out = ls.overwrite(labels, 5.0, 7.0, "both_grounded")
    assert _spans(out) == [(0.0, 5.0, "both_grounded"), (5.0, 7.0, "both_grounded"), (7.0, 9.0, "one_airborne")]


def test_joining_takes_only_the_immediate_neighbours():
    # Existing cuts between same-class segments (e.g. the airborne player
    # switching) are left alone; only the edited segment's neighbours join.
    labels = [Seg(0.0, 1.0, "one_airborne"), Seg(1.0, 2.0, "one_airborne"), Seg(2.0, 3.0, "both_grounded")]
    out = ls.overwrite(labels, 2.0, 3.0, "one_airborne")
    assert _spans(out) == [(0.0, 1.0, "one_airborne"), (1.0, 3.0, "one_airborne")]


def test_changing_a_class_joins_neighbours_but_cutting_does_not(three):
    joined = ls.set_class(three, 1, "one_airborne")
    assert _spans(joined) == [(0.0, 20.0, "one_airborne"), (20.0, 30.0, "one_offstage")]
    cut = ls.split(three, 1, 15.0)
    assert len(cut) == 4


def test_overwrite_keeps_the_other_fields_on_the_pieces_left_over():
    draft = [Seg(0.0, 10.0, "one_airborne", {"source": "model", "needs_review": True})]
    out = ls.overwrite(draft, 3.0, 4.0, "one_on_ledge")
    assert [s.extra for s in out] == [{"source": "model", "needs_review": True}, {"source": "human"},
                                      {"source": "model", "needs_review": True}]
    out[0].extra["needs_review"] = False
    assert out[2].extra["needs_review"] is True, "pieces must not share one dict"
    assert draft[0].extra["needs_review"] is True, "the input must not change"


# split

def test_split_cuts_a_segment_in_two_with_the_same_class(three):
    out = ls.split(three, 1, 13.5)
    assert _spans(out)[1:3] == [(10.0, 13.5, "both_grounded"), (13.5, 20.0, "both_grounded")]


def test_split_on_an_edge_is_refused(three):
    with pytest.raises(ls.EditError):
        ls.split(three, 1, 10.0)


# set_class and delete

def test_set_class_keeps_times_and_fields():
    out = ls.set_class([Seg(1.0, 2.0, "respawning", {"model_label": "respawning"})], 0, "game_end")
    assert _spans(out) == [(1.0, 2.0, "game_end")]
    assert out[0].extra == {"model_label": "respawning", "source": "human"}


def test_delete_leaves_the_time_unlabelled(three):
    assert _spans(ls.delete(three, 1)) == [(0.0, 10.0, "one_airborne"), (20.0, 30.0, "one_offstage")]


# move_edge

def test_moving_an_edge_moves_the_neighbour_that_shares_it(three):
    later = ls.move_edge(three, 1, "start", 11.0)
    assert _spans(later)[:2] == [(0.0, 11.0, "one_airborne"), (11.0, 20.0, "both_grounded")]
    earlier = ls.move_edge(three, 1, "end", 18.5)
    assert _spans(earlier)[1:] == [(10.0, 18.5, "both_grounded"), (18.5, 30.0, "one_offstage")]


def test_moving_an_edge_into_a_gap_needs_no_neighbour():
    out = ls.move_edge([Seg(0.0, 5.0, "both_grounded"), Seg(8.0, 9.0, "one_airborne")], 0, "end", 6.0)
    assert _spans(out) == [(0.0, 6.0, "both_grounded"), (8.0, 9.0, "one_airborne")]


@pytest.mark.parametrize("index,edge,t,why", [
    (1, "end", 10.0, "empty"),         # the segment itself would vanish
    (1, "start", 0.0, "swallow"),      # the neighbour would vanish
    (1, "start", 10.0, "already"),
])
def test_moving_an_edge_is_refused_when_something_would_vanish(three, index, edge, t, why):
    with pytest.raises(ls.EditError, match=f"(?i){why}"):
        ls.move_edge(three, index, edge, t)


def test_moving_an_edge_across_a_gap_into_another_segment_is_refused():
    labels = [Seg(0.0, 5.0, "both_grounded"), Seg(8.0, 9.0, "one_airborne")]
    with pytest.raises(ls.EditError, match="run into"):
        ls.move_edge(labels, 0, "end", 8.5)


# lookup and overlap detection

def test_segment_at_finds_the_covering_segment_and_rejects_overlaps(three):
    assert ls.segment_at(three, 10.0) == 1, "a segment covers its start, not its end"
    assert ls.segment_at(three, 30.0) is None
    with pytest.raises(ls.EditError, match="overlap"):
        ls.segment_at([Seg(0.0, 5.0, "a"), Seg(4.0, 6.0, "b")], 4.5)


def test_frame_time_maps_back_to_its_frame():
    for fps in (60.0, 59.94, 30.0, 29.97):
        for frame in range(0, 200_000, 997):
            assert round(ls.frame_time(frame, fps) * fps) == frame


# load and save

def test_unknown_fields_survive_a_load_save_round_trip():
    data = {
        "video_id": "abc", "video_path": "x.mp4", "format": 1,
        "games": [{"game_number": 1, "stage": "unknown", "start_sec": 0.0, "end_sec": 9.0,
                   "status": "draft", "bounds_confirmed": False,
                   "labels": [{"start_sec": 0.0, "end_sec": 9.0, "class": "both_grounded",
                               "source": "model", "needs_review": True, "confidence": 0.41}]}],
    }
    assert ls.VideoLabels.from_dict(json.loads(json.dumps(data))).to_dict() == data


@pytest.mark.skipif(not LABELS.is_dir() or not any(LABELS.glob("*.json")), reason="no label files here")
def test_real_label_files_load_and_save_unchanged():
    for path in LABELS.glob("*.json"):
        data = json.loads(path.read_text())
        assert ls.VideoLabels.from_dict(data).to_dict() == data, path.name


# source: who decided a segment's class (rule agreed with runitback-labeler)

DRAFT = {"source": "model", "needs_review": True, "model_label": "one_airborne",
         "llm_label": None, "mean_confidence": 0.81, "min_confidence": 0.42}


def _draft():
    return [Seg(0.0, 10.0, "one_airborne", dict(DRAFT)), Seg(10.0, 20.0, "both_grounded", dict(DRAFT))]


def test_a_segment_a_person_creates_is_human_and_the_leftovers_keep_the_draft_fields():
    out = ls.overwrite(_draft(), 4.0, 6.0, "one_on_ledge")
    assert [s.extra for s in out] == [DRAFT, {"source": "human"}, DRAFT, DRAFT]


def test_changing_a_class_makes_it_human_but_keeps_the_proposal_and_review_flag():
    out = ls.set_class(_draft(), 0, "both_airborne")
    assert out[0].extra == {**DRAFT, "source": "human"}
    assert out[0].extra["model_label"] == "one_airborne" and out[0].extra["needs_review"] is True
    assert out[1].extra == DRAFT


def test_moving_an_edge_makes_both_sides_of_it_human():
    labels = _draft() + [Seg(20.0, 30.0, "one_offstage", dict(DRAFT))]
    out = ls.move_edge(labels, 1, "start", 9.0)
    assert out[0].extra == out[1].extra == {**DRAFT, "source": "human"}
    assert out[2].extra == DRAFT, "a segment the edge did not touch is unchanged"


def test_a_cut_decides_nothing_so_both_pieces_keep_their_source():
    out = ls.split(_draft(), 0, 5.0)
    assert [s.extra for s in out[:2]] == [DRAFT, DRAFT]


def test_human_segments_join_but_draft_segments_do_not():
    plain = [Seg(0.0, 5.0, "both_grounded", {"source": "human"}), Seg(5.0, 9.0, "one_airborne")]
    joined = ls.overwrite(plain, 5.0, 7.0, "both_grounded")
    assert _spans(joined)[0] == (0.0, 7.0, "both_grounded") and joined[0].extra == {"source": "human"}
    legacy = [Seg(0.0, 5.0, "both_grounded"), Seg(5.0, 9.0, "one_airborne")]   # labelled before source existed
    assert _spans(ls.overwrite(legacy, 5.0, 7.0, "both_grounded"))[0] == (0.0, 7.0, "both_grounded")
    draft = ls.set_class(_draft(), 1, "one_airborne")   # still carries model fields: stays separate
    assert len(draft) == 2


# gold (test-only) videos

def test_a_video_is_either_training_or_gold_never_both(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert ls.default_labels_path("vidA", gold=False) == ls.TRAINING_LABELS_DIR / "vidA.json"
    assert ls.default_labels_path("vidA", gold=True) == ls.GOLD_LABELS_DIR / "vidA.json"

    ls.GOLD_LABELS_DIR.mkdir(parents=True)
    (ls.GOLD_LABELS_DIR / "vidA.json").write_text("{}")
    with pytest.raises(ValueError, match="--gold"):
        ls.default_labels_path("vidA", gold=False)

    ls.TRAINING_LABELS_DIR.mkdir(parents=True)
    (ls.TRAINING_LABELS_DIR / "vidB.json").write_text("{}")
    with pytest.raises(ValueError, match="cannot also be a gold video"):
        ls.default_labels_path("vidB", gold=True)


def test_the_gold_role_survives_load_and_save():
    data = {"video_id": "v", "video_path": "", "games": [], "role": "gold"}
    labels = ls.VideoLabels.from_dict(data)
    assert ls.is_gold(labels.to_dict()) and labels.to_dict() == data
