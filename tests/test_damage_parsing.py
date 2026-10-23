"""Turning OCR text into a damage percentage - pure string logic, no OCR model."""

import pytest

from runitback.cv.ocr import extract_damage_value


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0%", 0.0),
        ("45%", 45.0),
        ("118.9%", 118.9),
        ("999%", 999.0),
        ("O%", 0.0),       # letter O misread for zero
        ("S5%", 55.0),     # S misread for 5
        ("1000", None),    # over the 999% cap
        ("", None),
        ("%", None),
    ],
)
def test_parses_damage_text(text, expected):
    assert extract_damage_value(text) == expected


@pytest.mark.xfail(strict=True, reason="ISSUE-001: player tags are parsed as damage")
@pytest.mark.parametrize("tag", ["gleeb glob", "rohan"])
def test_player_tag_is_not_damage(tag):
    # Real tags from Steve vs Palutena: today these parse as 61 and 0.
    assert extract_damage_value(tag) is None


@pytest.mark.xfail(strict=True, reason="ISSUE-002: merged decimals are discarded")
def test_merged_decimal_is_recovered():
    # OCR often drops the small decimal point: 118.9% is read as "1189".
    assert extract_damage_value("1189") == 118.9
