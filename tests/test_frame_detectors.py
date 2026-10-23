"""Detectors on real broadcast frames, stored in tests/fixtures/frames.

The frames come from Steve vs Palutena (nmzdyQz6msI), game 1. No video
download is needed. The OCR tests load PaddleOCR, which downloads its models
(~100MB) on first use; after that they run in seconds.
"""

from pathlib import Path

import cv2
import pytest

from runitback.cv.ui_detector import (
    FrameType,
    detect_frame_type,
    detect_game_screen,
    detect_go_screen,
)

FRAMES = Path(__file__).parent / "fixtures" / "frames"


def load(name: str):
    image = cv2.imread(str(FRAMES / name))
    assert image is not None, f"missing fixture {name}"
    return image


# --- Screen detection (colour/template analysis, no models) ---

def test_go_screen_detected():
    assert detect_go_screen(load("go.jpg"))


def test_game_screen_detected():
    assert detect_game_screen(load("game_end.jpg"))


@pytest.mark.parametrize("name", ["go.jpg", "between_games.jpg"])
def test_game_screen_not_detected_elsewhere(name):
    assert not detect_game_screen(load(name))


@pytest.mark.parametrize("name", ["game_end.jpg", "between_games.jpg"])
def test_go_screen_not_detected_elsewhere(name):
    assert not detect_go_screen(load(name))


def test_menu_frame_is_not_gameplay():
    assert detect_frame_type(load("between_games.jpg")) == FrameType.MENU


# --- Damage OCR on the exact crops the pipeline reads (PaddleOCR) ---

@pytest.fixture(scope="module")
def read_damage():
    from runitback.cv.ocr import read_damage_percentage

    return lambda name: read_damage_percentage(load(name))[0]


def test_reads_high_damage(read_damage):
    # On screen: 113.8%. OCR drops the small decimal digit.
    assert read_damage("damage_88.5s_p2.png") == pytest.approx(113.8, abs=1.0)


def test_reads_zero_damage(read_damage):
    assert read_damage("damage_98.0s_p2.png") == 0.0


@pytest.mark.xfail(strict=True, reason="ISSUE-001: the player tag is read as damage")
def test_tag_not_read_as_damage(read_damage):
    # On screen: 142%. OCR reads the tag "gleeb glob" instead and parses it as 61.
    assert read_damage("damage_98.0s_p1.png") == pytest.approx(142.0, abs=1.0)


@pytest.mark.xfail(strict=True, reason="ISSUE-014: leading digit lost at the ROI's left edge")
def test_leading_digit_not_lost(read_damage):
    # On screen: 118.9%. The leading "1" sits on the crop's left edge; OCR reads 18.9.
    assert read_damage("damage_88.5s_p1.png") == pytest.approx(118.9, abs=1.0)
