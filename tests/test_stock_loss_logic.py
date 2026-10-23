"""Stock-loss detection on synthetic damage readings - no video, OCR, or ML model.

Each test builds the per-frame damage a game would produce (one reading per
0.5s, as the pipeline samples) and checks which stock losses are detected.
"""

import pytest

from runitback.classification.detector import GameStateDetector
from runitback.classification.states import FrameState

FRAME_INTERVAL = 0.5


def detect_losses(p1: list[float], p2: list[float]) -> list[tuple[int, float]]:
    """Run stock-loss detection over damage sequences; return (player, time) pairs."""
    frame_states = [
        FrameState(timestamp=i * FRAME_INTERVAL, frame_index=i, p1_damage=a, p2_damage=b)
        for i, (a, b) in enumerate(zip(p1, p2, strict=True))
    ]
    detector = GameStateDetector(use_smart_gemini=False, use_scene_classification=False)
    # detect_states() normally sets these from the GO!/GAME! markers.
    detector._game_start_times = [0.0]
    detector._game_end_times = [len(p1) * FRAME_INTERVAL]
    detector._detect_state_transitions(frame_states)
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
