"""CLI input handling - URL validation, --games parsing, timestamp formatting."""

import pytest
import typer

from runitback.cli import format_timestamp, parse_games_option, validate_youtube_url


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=nmzdyQz6msI",
        "https://youtube.com/watch?v=nmzdyQz6msI",
        "https://youtu.be/nmzdyQz6msI",
        "https://www.youtube.com/live/nmzdyQz6msI",
    ],
)
def test_accepts_youtube_urls(url):
    assert validate_youtube_url(url)


@pytest.mark.parametrize(
    "url", ["https://vimeo.com/123", "youtube.com/watch?v=abc", "not a url", ""]
)
def test_rejects_other_urls(url):
    assert not validate_youtube_url(url)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("", [1, 2, 3, 4, 5]),
        ("1", [1]),
        ("1,3", [1, 3]),
        ("2-4", [2, 3, 4]),
        ("1,3-5", [1, 3, 4, 5]),
        ("3,1,1", [1, 3]),
        (" 2 , 4 ", [2, 4]),
    ],
)
def test_parses_games_option(spec, expected):
    assert parse_games_option(spec, total_games=5) == expected


@pytest.mark.parametrize("spec", ["6", "0", "4-2", "a", "1-2-3", "2-x"])
def test_rejects_bad_games_option(spec):
    with pytest.raises(typer.BadParameter):
        parse_games_option(spec, total_games=5)


@pytest.mark.parametrize(
    ("seconds", "expected"), [(0, "0:00"), (9.9, "0:09"), (61, "1:01"), (3600, "60:00")]
)
def test_formats_timestamps(seconds, expected):
    assert format_timestamp(seconds) == expected
