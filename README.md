# RunItBack

A CLI tool to analyze Super Smash Bros. Ultimate replays from YouTube.

## Features

- Download and analyze SSBU replays from YouTube URLs
- Detect individual games within a set (Bo3/Bo5)
- Identify stages and characters for each game
- Extract damage percentages and stock counts

### Coming Soon

- Game state classification (neutral, advantage, disadvantage)
- Edgeguard situation detection
- Ledgetrap situation detection
- Improvement recommendations

## Installation

Requires Python 3.11+ and FFmpeg.

```bash
# Install FFmpeg (required for video processing)
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Install RunItBack
pip install -e .
```

## Usage

```bash
# Analyze a replay
runitback analyze "https://www.youtube.com/watch?v=VIDEO_ID"

# Specify player perspective
runitback analyze "https://www.youtube.com/watch?v=VIDEO_ID" --player 1

# Adjust sample rate (default 0.5 seconds)
runitback analyze "https://www.youtube.com/watch?v=VIDEO_ID" --sample-rate 0.25

# Enable verbose output
runitback analyze "https://www.youtube.com/watch?v=VIDEO_ID" --verbose
```

## Example Output

```
RunItBack - SSBU Replay Analyzer
Analyzing as Player 1
URL: https://www.youtube.com/watch?v=...

Downloaded: abc123.mp4
Extracted: 240 frames

Found 3 game(s)

==================================================
╭──────────────────────────────────────╮
│ GAME 1                               │
│ Stage: Battlefield                   │
│ You: Mario | Opponent: Fox           │
╰──────────────────────────────────────╯
  Duration: 0:00 - 2:45
  Final state: P1: 120% (1 stocks) | P2: 156% (0 stocks)

...
```

## Supported Stages

- Small Battlefield
- Battlefield
- Final Destination
- Town and City
- Smashville
- Hollow Bastion
- Kalos Pokemon League
- Pokemon Stadium 2

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy runitback
```

## License

MIT
