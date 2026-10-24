"""Stock-loss detection on synthetic damage readings - no video, OCR, or ML model.

Each test builds the per-frame damage a game would produce (one reading per
0.5s, as the pipeline samples) and checks which stock losses are detected.
"""

import pytest

from runitback.classification.detector import GameStateDetector
from runitback.classification.states import FrameState

FRAME_INTERVAL = 0.5


class FakeFrame:
    """Stands in for a video frame; `image` is just its index."""

    def __init__(self, index: int):
        self.image = index


def detect_losses(
    p1: list[float],
    p2: list[float],
    game_screen_frames: set[int] | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
    blank_damage=None,
) -> list[tuple[int, float]]:
    """Run stock-loss detection over damage sequences; return (player, time) pairs.

    With `game_screen_frames`, frames are supplied and the GAME! detector reports
    a GAME! screen at exactly those frame indices - exercising the no-Gemini
    final-stock pass. Without it, no frames are passed and that pass is skipped.
    `blank_damage(frame_index, roi)` stands in for the damage-digit OCR check.
    """
    frame_states = [
        FrameState(timestamp=i * FRAME_INTERVAL, frame_index=i, p1_damage=a, p2_damage=b)
        for i, (a, b) in enumerate(zip(p1, p2, strict=True))
    ]
    detector = GameStateDetector(use_smart_gemini=False, use_scene_classification=False)
    detector.gemini_api_key = None  # these tests cover the no-Gemini path, whatever the environment
    # detect_states() normally sets these from the GO!/GAME! markers.
    detector._game_start_times = [0.0]
    detector._game_end_times = [len(p1) * FRAME_INTERVAL]

    frames = None
    if game_screen_frames is not None:
        assert monkeypatch is not None
        monkeypatch.setattr(
            "runitback.classification.detector.detect_game_screen",
            lambda image: image in game_screen_frames,
        )
        # No damage-display reading unless a test sets one: the blank check is undecided.
        monkeypatch.setattr("runitback.classification.detector.get_roi", lambda image, roi: (image, roi))
        monkeypatch.setattr(
            "runitback.classification.detector.has_damage_digits",
            lambda roi: blank_damage(*roi) if blank_damage else None,
        )
        frames = [FakeFrame(i) for i in range(len(p1))]

    detector._detect_state_transitions(frame_states, frames)
    return [(loss.player, loss.timestamp) for loss in detector._collect_stock_losses(frame_states)]


def climb_then_die(peak: float = 120.0, frames: int = 60, after: int = 20) -> list[float]:
    """Damage rising steadily to `peak`, then a respawn at 0%."""
    return [min(peak, i * 3.0) for i in range(frames)] + [0.0] * after


STEADY = [30.0] * 80


def test_detects_p2_death():
    assert detect_losses(STEADY, climb_then_die()) == [(2, 31.0)]


def test_detects_p1_death():
    assert detect_losses(climb_then_die(), STEADY) == [(1, 31.0)]


def test_no_death_without_a_reset():
    assert detect_losses(STEADY, [50.0] * 80) == []


def test_low_damage_drop_is_not_a_death():
    # Falling from 20% to 0% is below MIN_KILL_DAMAGE - not a plausible kill.
    assert detect_losses(STEADY, [20.0] * 40 + [0.0] * 40) == []


def test_single_zero_misread_is_ignored():
    # One OCR glitch reading 0% mid-combo must not count; confirmation needs
    # several consecutive low frames.
    p2 = [min(120.0, i * 3.0) for i in range(80)]
    p2[50] = 0.0
    assert detect_losses(STEADY, p2) == []


def test_detects_deaths_by_both_players():
    p1 = [80.0] * 20 + [min(60.0, float(i)) for i in range(30)] + [60.0] * 30
    assert detect_losses(p1, climb_then_die()) == [(1, 11.0), (2, 31.0)]


@pytest.mark.xfail(strict=True, reason="ISSUE-003: both_reset guard hides the second death")
def test_death_while_opponent_still_at_zero():
    # P1 dies at ~11s and stays at 0%; P2 dies at ~31s. Both now read 0%, which
    # the guard treats as a new game, so P2's death is dropped.
    p1 = [80.0] * 20 + [0.0] * 60
    assert detect_losses(p1, climb_then_die()) == [(1, 11.0), (2, 31.0)]


# --- Final stock without Gemini (ISSUE-005) ---
# The last stock of a game has no respawn or damage reset after it. The game ends
# at frame 80 (40s); the GAME! screen shows at frames 68-70 (34-35s).

GAME_SCREEN = {68, 69, 70}


def test_final_stock_detected_from_game_screen(monkeypatch):
    # P2 climbs to 130% and is KO'd; the game ends with no reset to detect.
    p2 = [min(130.0, i * 2.5) for i in range(80)]
    assert detect_losses(STEADY, p2, GAME_SCREEN, monkeypatch) == [(2, 34.0)]


def test_final_stock_credited_to_player_one(monkeypatch):
    p1 = [min(130.0, i * 2.5) for i in range(80)]
    assert detect_losses(p1, STEADY, GAME_SCREEN, monkeypatch) == [(1, 34.0)]


def test_final_stock_not_double_counted(monkeypatch):
    # P2's death at 31s is already caught by the damage reset; GAME! follows at 34s.
    assert detect_losses(STEADY, climb_then_die(), GAME_SCREEN, monkeypatch) == [(2, 31.0)]


def test_close_damage_and_equal_losses_is_skipped_not_guessed(monkeypatch):
    # Both at similar damage with no earlier losses: the loser can't be told apart,
    # so nothing is recorded (the Gemini path would default to P1).
    assert detect_losses([90.0] * 80, [95.0] * 80, GAME_SCREEN, monkeypatch) == []


def test_no_game_screen_means_no_final_stock(monkeypatch):
    p2 = [min(130.0, i * 2.5) for i in range(80)]
    assert detect_losses(STEADY, p2, set(), monkeypatch) == []


# The loser's damage number disappears at GAME!; the winner's stays.

def p2_display_blank(frame_index, roi):
    from runitback.cv.ui_detector import P1_DAMAGE_ROI
    return roi == P1_DAMAGE_ROI  # digits only in P1's display


def test_blank_damage_display_identifies_loser(monkeypatch):
    # Damage alone can't decide (both ~90%), but P2's display is blank after GAME!.
    losses = detect_losses([90.0] * 80, [95.0] * 80, GAME_SCREEN, monkeypatch, p2_display_blank)
    assert losses == [(2, 34.0)]


def test_blank_damage_display_overrides_damage_guess(monkeypatch):
    # P1 has the higher damage, which alone would blame P1 - but P1's number is
    # still shown and P2's is blank, so P2 lost.
    p1 = [min(130.0, i * 2.5) for i in range(80)]
    assert detect_losses(p1, STEADY, GAME_SCREEN, monkeypatch, p2_display_blank) == [(2, 34.0)]


def test_both_displays_blank_falls_back_to_damage(monkeypatch):
    p2 = [min(130.0, i * 2.5) for i in range(80)]
    neither = lambda frame_index, roi: False  # noqa: E731
    assert detect_losses(STEADY, p2, GAME_SCREEN, monkeypatch, neither) == [(2, 34.0)]



# --- The 3-stock cap (ISSUE-016) ---

def cap_losses(
    loss_times: list[float], game_end: float, p1_loss_times: tuple[float, ...] = ()
) -> list[float]:
    """Mark P2 (and optional P1) losses, each followed by a respawn; apply the cap; return P2's."""
    frame_states = [
        FrameState(timestamp=i * FRAME_INTERVAL, frame_index=i, p1_damage=30.0, p2_damage=30.0)
        for i in range(int(game_end / FRAME_INTERVAL) + 3)
    ]
    for t in loss_times:
        frame_states[int(t / FRAME_INTERVAL)].p2_stock_loss = True
        frame_states[int((t + 1.0) / FRAME_INTERVAL)].scene_state = "respawning"
    for t in p1_loss_times:
        frame_states[int(t / FRAME_INTERVAL)].p1_stock_loss = True
    detector = GameStateDetector(use_smart_gemini=False, use_scene_classification=False)
    detector._game_start_times = [0.0]
    detector._game_end_times = [game_end]
    detector._enforce_stock_loss_limit(frame_states, max_stocks=3)
    return [fs.timestamp for fs in frame_states if fs.p2_stock_loss]


def test_cap_keeps_the_final_stock():
    # Five verified losses for a 3-stock game: two are false positives. The one
    # at the game's end (95s, end 100s) is the final stock and must survive.
    assert cap_losses([10.0, 25.0, 40.0, 60.0, 95.0], game_end=100.0) == [10.0, 25.0, 95.0]


def test_cap_without_a_final_stock_keeps_the_earliest():
    assert cap_losses([10.0, 25.0, 40.0, 60.0], game_end=100.0) == [10.0, 25.0, 40.0]


def test_cap_under_the_limit_changes_nothing():
    assert cap_losses([10.0, 95.0], game_end=100.0) == [10.0, 95.0]


def test_cap_protects_the_earliest_loss_near_the_end():
    # Real final stock at 95s, then a spurious "loss" at 99s from the post-GAME!
    # screen change. The real one is protected; the spurious one is not.
    assert cap_losses([10.0, 25.0, 40.0, 95.0, 99.0], game_end=100.0) == [10.0, 25.0, 95.0]


def test_cap_does_not_protect_the_winners_late_loss():
    # P1 lost the game at 95s. P2's 98s "loss" near the end is screen noise and must
    # not displace P2's real earlier losses.
    assert cap_losses([10.0, 25.0, 40.0, 98.0], game_end=100.0, p1_loss_times=(95.0,)) == [10.0, 25.0, 40.0]

