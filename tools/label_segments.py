"""Label files and segment edits for scene_labeler.py, kept free of Qt so they can be tested.

A label file holds games, and each game holds time segments with one scene class.
Every edit here returns a new, sorted segment list and leaves the input alone, or
raises EditError saying why it was refused. No edit creates an overlap: labelling
time that is already labelled replaces what was there.

Fields this tool does not use (for example `needs_review` or `model_label`
written by the runitback-labeler drafts) are kept in `extra` and written back on
save, so opening and saving a file never drops them.

`source` says who decided a segment's class (agreed with runitback-labeler):
- "human" on a segment a person creates, whose class a person changes, or
  whose edge a person moves (both segments meeting at that edge);
- unchanged on a piece left over from a cut or trimmed by an overwrite: nobody
  judged that piece, so its class is still the draft's proposal, and it keeps
  all its draft fields;
- no edit changes `needs_review`, `model_label` or `llm_label`.
"""

from dataclasses import dataclass, field, replace

# Every scene class, in the label tool's key order. The one list the label tool,
# extraction and training share; definitions are in docs/SCENE_CLASSES.md.
# Inference (runitback/cv/scene_classifier.py) reads a model's own class order
# from its class_mapping.json instead.
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
    "pre_game",        # added 2026-11-26: entrances and countdown, until GO!
    "game_end",        # added 2026-11-26: GAME! or TIME! screen, until the white flash
    "finishing_blow",  # added 2026-11-26: the final-stock red flash
]

# Times are stored to 0.1 ms. Frames at 60fps are 16.7 ms apart, so a stored time
# identifies its frame: round(time * fps).
TIME_DECIMALS = 4
# Two times closer than this are the same moment.
EPS = 1e-6

_SEGMENT_KEYS = {"start_sec", "end_sec", "class", "scene_class"}
HUMAN = {"source": "human"}
_GAME_KEYS = {"game_number", "stage", "start_sec", "end_sec", "labels"}
_VIDEO_KEYS = {"video_id", "video_path", "games"}


class EditError(ValueError):
    """An edit was refused. The message says why, for the status bar."""


@dataclass
class SegmentLabel:
    """A labelled stretch of video: [start_sec, end_sec) has scene_class."""
    start_sec: float
    end_sec: float
    scene_class: str
    extra: dict = field(default_factory=dict)

    def describe(self) -> str:
        return f"{self.scene_class} {self.start_sec:.3f}-{self.end_sec:.3f}s"


@dataclass
class GameLabels:
    """Labels for a single game within a video."""
    game_number: int
    stage: str = "unknown"
    start_sec: float = 0.0
    end_sec: float = 0.0
    labels: list[SegmentLabel] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class VideoLabels:
    """All labels for a video."""
    video_id: str
    video_path: str
    games: list[GameLabels] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Known fields first, then every field the file was loaded with."""
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "games": [
                {
                    "game_number": g.game_number,
                    "stage": g.stage,
                    "start_sec": g.start_sec,
                    "end_sec": g.end_sec,
                    "labels": [
                        {"start_sec": s.start_sec, "end_sec": s.end_sec, "class": s.scene_class, **s.extra}
                        for s in g.labels
                    ],
                    **g.extra,
                }
                for g in self.games
            ],
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoLabels":
        games = []
        for g in data.get("games", []):
            labels = [
                SegmentLabel(
                    start_sec=s["start_sec"],
                    end_sec=s["end_sec"],
                    scene_class=s["class"] if "class" in s else s["scene_class"],
                    extra={k: v for k, v in s.items() if k not in _SEGMENT_KEYS},
                )
                for s in g.get("labels", [])
            ]
            games.append(GameLabels(
                game_number=g["game_number"],
                stage=g.get("stage", "unknown"),
                start_sec=g.get("start_sec", 0.0),
                end_sec=g.get("end_sec", 0.0),
                labels=labels,
                extra={k: v for k, v in g.items() if k not in _GAME_KEYS},
            ))
        return cls(
            video_id=data["video_id"],
            video_path=data.get("video_path", ""),
            games=games,
            extra={k: v for k, v in data.items() if k not in _VIDEO_KEYS},
        )


def frame_time(frame: int, fps: float) -> float:
    """The stored time of a video frame."""
    return round(frame / fps, TIME_DECIMALS)


def _sorted(labels: list[SegmentLabel]) -> list[SegmentLabel]:
    return sorted(labels, key=lambda s: (s.start_sec, s.end_sec))


def _piece(segment: SegmentLabel, start: float, end: float) -> SegmentLabel:
    """Part of a segment, keeping its class and all its fields, source included."""
    return replace(segment, start_sec=start, end_sec=end, extra=dict(segment.extra))


def _decided(segment: SegmentLabel, **changes) -> SegmentLabel:
    """The segment after a person judged it: source "human", other fields kept."""
    return replace(segment, **changes, extra={**segment.extra, **HUMAN})


def _plain(segment: SegmentLabel) -> bool:
    """No draft fields: nothing but (at most) source "human". Only plain segments join."""
    return segment.extra in ({}, HUMAN)


def segment_at(labels: list[SegmentLabel], t: float) -> int | None:
    """Index of the segment covering time t, or None. Raises if two segments cover it."""
    hits = [i for i, s in enumerate(labels) if s.start_sec - EPS <= t < s.end_sec - EPS]
    if len(hits) > 1:
        raise EditError(f"Two segments overlap at {t:.3f}s; fix that first (overwrite the overlap)")
    return hits[0] if hits else None


def overlaps(labels: list[SegmentLabel]) -> list[tuple[SegmentLabel, SegmentLabel]]:
    """Pairs of segments that cover the same time. A file written by this module has none."""
    ordered = _sorted(labels)
    found = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if b.start_sec >= a.end_sec - EPS:
                break
            found.append((a, b))
    return found


def overwrite(labels: list[SegmentLabel], start: float, end: float, scene_class: str) -> list[SegmentLabel]:
    """Label [start, end) as scene_class, replacing whatever covered that time.

    Segments inside the span are removed, segments crossing one of its edges are
    trimmed to it, and a segment covering the whole span is cut around it. The
    new segment is joined with a touching segment of the same class.
    """
    start, end = min(start, end), max(start, end)
    if end - start <= EPS:
        raise EditError("Nothing to label: the start and end are the same frame")
    # Nothing to do only if one segment of this class covers the span and no
    # other segment overlaps it; with an overlap, overwriting is how it is fixed.
    touching = [s for s in labels if s.start_sec < end - EPS and s.end_sec > start + EPS]
    if (len(touching) == 1 and touching[0].scene_class == scene_class
            and touching[0].start_sec <= start + EPS and touching[0].end_sec >= end - EPS):
        raise EditError(f"Already {scene_class} ({touching[0].describe()}); nothing changed")

    out = []
    for s in labels:
        if s.end_sec <= start + EPS or s.start_sec >= end - EPS:
            out.append(s)
            continue
        if s.start_sec < start - EPS:
            out.append(_piece(s, s.start_sec, start))
        if s.end_sec > end + EPS:
            out.append(_piece(s, end, s.end_sec))
    new = SegmentLabel(start, end, scene_class, dict(HUMAN))
    return _join_neighbours(out, new)


def _join_neighbours(others: list[SegmentLabel], seg: SegmentLabel) -> list[SegmentLabel]:
    """Add seg to others, joined with any touching segment of the same class.

    Two touching segments of one class mean the same as one, and the cut between
    them costs training frames (extraction skips each segment's edges). Only
    plain segments are joined (see _plain): a draft segment carrying
    needs_review or a model guess keeps its own identity until it is reviewed.
    The joined segment is source "human", since a person made the edit.
    """
    if not _plain(seg):
        return _sorted([*others, seg])
    # Compared with seg's own edges, so only its immediate neighbours join,
    # whatever order the list is in.
    start, end, kept = seg.start_sec, seg.end_sec, []
    for s in others:
        joinable = s.scene_class == seg.scene_class and _plain(s)
        if joinable and abs(s.end_sec - seg.start_sec) <= EPS:
            start = s.start_sec
        elif joinable and abs(s.start_sec - seg.end_sec) <= EPS:
            end = s.end_sec
        else:
            kept.append(s)
    return _sorted([*kept, replace(seg, start_sec=start, end_sec=end, extra=dict(HUMAN))])


def split(labels: list[SegmentLabel], index: int, t: float) -> list[SegmentLabel]:
    """Cut segment `index` in two at time t. Both pieces keep its class and all its fields.

    A cut decides nothing about either piece's class, so source is not changed.
    """
    s = labels[index]
    if not s.start_sec + EPS < t < s.end_sec - EPS:
        raise EditError(f"{t:.3f}s is not inside {s.describe()}; nothing to cut")
    rest = [o for i, o in enumerate(labels) if i != index]
    return _sorted(rest + [_piece(s, s.start_sec, t), _piece(s, t, s.end_sec)])


def set_class(labels: list[SegmentLabel], index: int, scene_class: str) -> list[SegmentLabel]:
    """Give segment `index` another class: source becomes "human", other fields are kept.

    Joined with a touching segment of the new class, as overwrite does.
    """
    s = labels[index]
    if s.scene_class == scene_class:
        raise EditError(f"Already {scene_class}; nothing changed")
    others = [o for i, o in enumerate(labels) if i != index]
    return _join_neighbours(others, _decided(s, scene_class=scene_class))


def delete(labels: list[SegmentLabel], index: int) -> list[SegmentLabel]:
    """Remove segment `index`. Its time is left unlabelled, and unlabelled time is not trained on."""
    return [s for i, s in enumerate(labels) if i != index]


def move_edge(labels: list[SegmentLabel], index: int, edge: str, t: float) -> list[SegmentLabel]:
    """Move segment `index`'s start or end to time t.

    The segment that shares that edge moves with it, so no gap or overlap opens
    between them. Both get source "human": placing the edge is a judgement about
    where one class ends and the next begins. Refused if either would become
    empty or the segment would run into another one (use overwrite for that).
    """
    if edge not in ("start", "end"):
        raise ValueError(f"edge must be 'start' or 'end', not {edge!r}")
    s = labels[index]
    old = s.start_sec if edge == "start" else s.end_sec
    if abs(t - old) <= EPS:
        raise EditError(f"The {edge} is already at {t:.3f}s")
    new_start, new_end = (t, s.end_sec) if edge == "start" else (s.start_sec, t)
    if new_end - new_start <= EPS:
        raise EditError(f"Moving the {edge} to {t:.3f}s would leave {s.describe()} empty")

    # The neighbour across this edge: it ends where s starts, or starts where s ends.
    shares = [i for i, o in enumerate(labels) if i != index and abs(
        (o.end_sec if edge == "start" else o.start_sec) - old) <= EPS]
    if len(shares) > 1:
        raise EditError(f"Several segments meet at {old:.3f}s; fix the overlap first")
    out = list(labels)
    out[index] = _decided(s, start_sec=new_start, end_sec=new_end)
    if shares:
        j = shares[0]
        n = labels[j]
        n_start, n_end = (n.start_sec, t) if edge == "start" else (t, n.end_sec)
        if n_end - n_start <= EPS:
            raise EditError(f"Moving the {edge} to {t:.3f}s would swallow {n.describe()}; "
                            "delete it or overwrite instead")
        out[j] = _decided(n, start_sec=n_start, end_sec=n_end)

    for i, o in enumerate(out):
        if i == index or (shares and i == shares[0]):
            continue
        if o.start_sec < new_end - EPS and o.end_sec > new_start + EPS:
            raise EditError(f"Moving the {edge} to {t:.3f}s would run into {o.describe()}; "
                            "overwrite instead")
    return _sorted(out)
