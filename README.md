# RunItBack

A CLI tool to analyze Super Smash Bros. Ultimate replays from YouTube.

## Features

- Download and analyze SSBU replays from YouTube URLs
- Detect individual games within a set (GO!/GAME! marker detection)
- Identify characters for each game (Gemini Vision API + template matching fallback)
- **Game state classification** - neutral, advantage, disadvantage with timing breakdowns
- **Stock loss tracking** - detects when stocks are lost via damage reset and respawn detection
- **Scene classification** - ML-based detection of positional states (grounded, airborne, offstage, respawning)

## Getting Started

### Prerequisites

- **Python 3.11 or higher** — check with `python3 --version`
  - macOS: `brew install python@3.11`
  - Ubuntu/Debian: `sudo apt install python3.11 python3.11-venv`
  - Or download from [python.org](https://www.python.org/downloads/)

- **FFmpeg** — used for video downloading and frame extraction
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`

- **Git** — to clone the repository
  - macOS: comes pre-installed (or `brew install git`)
  - Ubuntu/Debian: `sudo apt install git`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ankitjk/RunItBack.git
cd RunItBack

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate    # macOS/Linux

# 3. Install RunItBack and its dependencies
pip install -e .
```

### Scene Classifier Model

The scene classifier model (~107MB) is not included in the repository. Fetch it from the latest GitHub release:

```bash
./scripts/fetch_model.sh
```

Without the model, RunItBack still runs but scene classification will be disabled, reducing the accuracy of game state detection.

### Optional: Gemini API Key

For improved character detection accuracy, set a Gemini API key:

```bash
export GEMINI_API_KEY="your-api-key"
```

Without an API key, character detection falls back to template matching.

### Verify Installation

```bash
runitback --help
```

If you see the help output, you're good to go.

## Usage

```bash
# Analyze a replay
runitback analyze "https://www.youtube.com/watch?v=VIDEO_ID"

# Specify player perspective (1 = left side, 2 = right side)
runitback analyze "https://www.youtube.com/watch?v=VIDEO_ID" --player 1

# Adjust sample rate (default 0.5 seconds)
runitback analyze "https://www.youtube.com/watch?v=VIDEO_ID" --sample-rate 0.25

# Analyze specific games only (e.g., games 1 and 3)
runitback analyze "https://www.youtube.com/watch?v=VIDEO_ID" --games 1,3

# Enable verbose output
runitback analyze "https://www.youtube.com/watch?v=VIDEO_ID" --verbose
```

## Example Output

```
RunItBack - SSBU Replay Analyzer
Analyzing as Player 1
URL: https://www.youtube.com/watch?v=...

Downloaded: abc123.mp4
Extracted: 960 frames

Found 2 game(s)

==================================================
╭──────────────────────────────────────╮
│ GAME 1                               │
│ You: Ken | Opponent: King Dedede     │
╰──────────────────────────────────────╯
  Duration: 0:18 - 3:35

STATE SUMMARY:
  Neutral:      0:42 (25%)
  Advantage:    0:49 (28%)
  Disadvantage: 1:20 (47%)

SIGNIFICANT INTERACTIONS:
  (>5s duration or stock lost)
  0:26  ▼ Disadvantage (Juggle) [6.2s]
  0:44  ▼ Disadvantage [11.0s] ★ Lost stock
  1:20  ▲ Advantage [8.5s] ★ Took stock
  2:40  ▼ Disadvantage (Offstage) [5.0s] ★ Lost stock

SMALL INTERACTIONS:
  (brief exchanges <5s)
  ▲ Small wins: 4 (0:12 total)
  ▼ Stray hits: 6 (0:18 total)

NEUTRAL TRANSITIONS:
  (timestamps when neutral was won/lost)
  0:18○  0:26▼  0:32○  0:38▲  0:44▼  0:55○  1:02▼  1:15○
  1:20▲  1:28○  1:35▼  ...

Legend: ○ Neutral  ▲ Advantage  ▼ Disadvantage
```

### Output Explanation

- **STATE SUMMARY**: Time spent in each game state with percentages
- **SIGNIFICANT INTERACTIONS**: Long sequences (>5s) or moments where stocks were lost
  - `▲ Advantage` = You were winning the exchange (opponent in disadvantage)
  - `▼ Disadvantage` = You were losing the exchange (getting combo'd, edgeguarded)
  - Disadvantage types: `Juggle`, `Offstage`, `Ledge` when detected
  - `★ Lost stock` / `★ Took stock` = Stock change during this segment
- **SMALL INTERACTIONS**: Brief exchanges (<5s) summarized by count
- **NEUTRAL TRANSITIONS**: Timeline showing when advantage/disadvantage began

## Supported Stages

- Small Battlefield
- Battlefield
- Final Destination
- Town and City
- Smashville
- Hollow Bastion
- Kalos Pokemon League
- Pokemon Stadium 2

## Architecture

```
runitback/
├── cli.py                    # Typer CLI entry point
├── video/
│   ├── downloader.py         # yt-dlp wrapper with caching
│   └── frame_extractor.py    # OpenCV frame sampling
├── cv/
│   ├── ocr.py                # PaddleOCR damage reading
│   ├── scene_classifier.py   # ML scene classification (ONNX)
│   ├── ui_detector.py        # HUD detection, stock counting
│   ├── character_detector.py # Character identification
│   └── llm_detector.py       # Gemini Vision API integration
└── classification/
    ├── states.py             # GameState, FrameState, StateSegment
    └── detector.py           # Game state detection logic
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Install training dependencies (PyTorch, for retraining the scene classifier)
pip install -e ".[train]"

# Run tests
pytest

# Run specific test file
pytest tests/test_game_detection.py

# Run linting
ruff check .

# Run type checking
mypy runitback
```

### Tools

Development tools are in the `tools/` directory:

- `scene_labeler.py` - PyQt5 app for labeling training data
- `extract_labeled_frames.py` - Extract frames from labeled videos
- `train_scene_classifier.py` - Train the scene classification model (requires `[train]` deps)
- `test_stock_losses.py` - Test stock loss detection accuracy

## Known Limitations

1. **OCR Accuracy**: Damage reading can have errors during visual effects (especially Steve's mining blocks)
2. **Stock Loss Detection**: ~74% average recall, varies by matchup
3. **Stage Detection**: Not yet implemented (returns "unknown")
4. **Character Detection**: May confuse similar characters (Mario/Luigi, Pit/Dark Pit) without title hints

## License

MIT
