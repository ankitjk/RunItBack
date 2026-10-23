# RunItBack

[![tests](https://github.com/ankitjk/RunItBack/actions/workflows/tests.yml/badge.svg)](https://github.com/ankitjk/RunItBack/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A command-line tool that watches a Super Smash Bros. Ultimate tournament set on
YouTube and reports what happened: where each game starts and ends, who played,
when stocks were lost, and how much time each player spent in neutral,
advantage, and disadvantage.

It works from broadcast video alone - no game data, no replay files - so it has
to cope with stream overlays, camera zooms, and visual effects covering the HUD.

## How it works

```
YouTube URL
  └─ download (yt-dlp, cached)
      └─ sample a frame every 0.5s (OpenCV)
          ├─ game boundaries   GO! / GAME! screen detection (template + colour analysis)
          ├─ characters        Gemini Vision, or HUD-portrait template matching without a key
          ├─ damage %          PaddleOCR on each player's HUD region
          └─ scene             ConvNeXt-Tiny classifier (ONNX): grounded, airborne,
                               offstage, on ledge, respawning, ...
              └─ game state    who took damage -> advantage / disadvantage;
                               damage reset + respawn -> stock lost
```

Each signal is noisy on its own - OCR misreads under particle effects, the
classifier was trained on limited data - so the stock-loss logic combines them:
a loss needs high damage first, then several consecutive low readings or a
respawn, with per-player cooldowns to suppress duplicates.

## Results

Stock-loss recall against hand-labeled timestamps (a detection must be within
5s **and** credit the right player). Measured October 23, 2026, **without** a
Gemini key:

| Set | Recall | False positives |
|-----|--------|-----------------|
| Ken vs King Dedede | 100% (8/8) | 0 |
| Dark Pit vs Ken | 83% (15/18) | 4 |
| Ken vs Sonic | 77% (10/13) | 4 |
| Steve vs Palutena | 52% (12/23) | 15 |
| **Total** | **72.6% (45/62)** | **23** |

Two further benchmark sets could not be downloaded on that run (YouTube's bot
check), so this covers 4 of 6. The weakest set, Steve vs Palutena, is dense
with block and particle effects over the damage display; its failures are
diagnosed in [docs/ISSUES.md](docs/ISSUES.md).

**Gemini** is strongly recommended for character identification - without it,
template matching named Steve vs Palutena as Little Mac vs Zero Suit Samus.
Stock-loss accuracy is similar with or without it on this benchmark set.

## Getting started

### Prerequisites

- **Python 3.11+** (`python3 --version`)
- **FFmpeg** - yt-dlp uses it to merge video and audio
  (macOS: `brew install ffmpeg`, Ubuntu: `sudo apt install ffmpeg`)

### Install

```bash
git clone https://github.com/ankitjk/RunItBack.git
cd RunItBack
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Scene classifier model

The model (~107 MB) is published as a GitHub release asset rather than
committed. Fetch it once:

```bash
./scripts/fetch_model.sh
```

Without it RunItBack still runs, but scene classification is disabled and
game-state accuracy drops.

### Gemini API key (optional, recommended)

```bash
export GEMINI_API_KEY="your-api-key"  # pragma: allowlist secret
```

It is read from the environment only (not from a `.env` file). Without it,
character detection falls back to template matching and a warning is logged.

### Check the install

```bash
runitback --help
```

## Usage

```bash
# Analyze a set (prompts for which player's perspective to use)
runitback analyze "https://www.youtube.com/watch?v=VIDEO_ID"

# Player 1 (left HUD) or 2 (right HUD)
runitback analyze "https://www.youtube.com/watch?v=VIDEO_ID" --player 1

# Only some games: '1', '1,3', or '2-4'
runitback analyze "https://www.youtube.com/watch?v=VIDEO_ID" --player 1 --games 1,3

# Sample more often (default 0.5s), or log every detection
runitback analyze "https://www.youtube.com/watch?v=VIDEO_ID" --player 1 --sample-rate 0.25 --verbose
```

A full set takes roughly 20 minutes on a laptop CPU the first time, mostly
download and OCR. Videos are cached in `~/.cache/runitback/`, so reruns skip
the download.

### Example output

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
  2:40  ▼ Disadvantage (Offstage/Edgeguard) [5.0s] ★ Lost stock

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

- **State summary** - time in each state, from your perspective
- **Significant interactions** - exchanges over 5s, or any with a stock lost or
  taken. Disadvantage is tagged (`Juggle`, `Offstage/Edgeguard`, `Ledgetrap`,
  `Tech Chase`, `Respawning`) when the scene classifier can tell
- **Small interactions** - brief exchanges, counted
- **Neutral transitions** - when each advantage or disadvantage began

## Development

```bash
pip install -e ".[dev]"

# Fast tests - no downloads or API keys, ~10s (PaddleOCR fetches its models on first run)
pytest -m "not slow"

# Full suite - downloads the six benchmark sets; minutes per test
pytest

# Lint, type check, and the instruction checker
ruff check .
mypy runitback
python scripts/check_docs.py

# Stock-loss benchmark (per-set recall against ground truth)
python tools/test_stock_losses.py
```

The fast tests cover damage-text parsing, stock-loss logic on synthetic damage
sequences, CLI input handling, and GO!/GAME!/OCR detection on real frames stored
in `tests/fixtures/frames/`. Known bugs are written as strict expected-failure
tests linked to their issue, so fixing one flips its test and the suite flags
it.

Git hooks in `scripts/hooks/` (enable with `git config core.hooksPath scripts/hooks`)
block commits containing secrets, and require each commit message to state the
issues it fixes or finds (`Fixes: ISSUE-001`, or `Issues: none`).

### Retraining the scene classifier

```bash
pip install -r tools/requirements-labeling.txt   # PyQt5
pip install -r tools/requirements-training.txt   # PyTorch, ONNX

python tools/scene_labeler.py "<video file or YouTube URL>"   # label segments by scene class
python tools/extract_labeled_frames.py                        # labels -> frames + train/val/test splits
python tools/train_scene_classifier.py                        # fine-tune ConvNeXt-Tiny, export ONNX
```

The model is fine-tuned from ImageNet weights. The current one was trained on
a single labeled set, so it has no meaningful held-out accuracy figure yet -
see the October 23 correction in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

### Project layout

```
runitback/
├── cli.py                     # Typer CLI; game boundary detection and output
├── config.py                  # Settings (RUNITBACK_ environment prefix)
├── video/
│   ├── downloader.py          # yt-dlp wrapper with caching
│   └── frame_extractor.py     # OpenCV frame sampling
├── cv/
│   ├── ocr.py                 # PaddleOCR damage reading
│   ├── scene_classifier.py    # ONNX scene classification
│   ├── ui_detector.py         # HUD regions, GO!/GAME! detection, frame types
│   ├── character_detector.py  # Character identification (Gemini + template fallback)
│   ├── llm_detector.py        # Gemini Vision API calls and caching
│   ├── stage_detector.py      # Stage detection (built, currently switched off)
│   └── stock_counter_detector.py
└── classification/
    ├── states.py              # GameState, FrameState, StateSegment, StockLoss
    └── detector.py            # Game-state and stock-loss detection
```

## Design notes

- **OCR + ML instead of one end-to-end model.** There is no large labeled
  dataset of SSBU broadcasts, so the pipeline uses signals that need little or
  no training data (OCR, template matching, colour analysis) and reserves the
  trained classifier for positional state, where no simple rule works.
- **Gemini is optional by design.** Without a key the tool still runs end to
  end: characters fall back to template matching and the Gemini-only checks are
  skipped. With a key, results are cached by frame hash so the same frame is
  never sent twice.
- **Measured, not assumed.** A benchmark against hand-labeled ground truth
  backs every accuracy claim, and runs are reported as not comparable when the
  configuration differs.
- **The development record is kept.** [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
  chronicles what was tried, what failed, and why - including a correction of
  earlier overstated training results.

Claude Code is used as a development assistant. Its project instructions
([CLAUDE.md](CLAUDE.md), `.claude/`) are versioned with the code, and
`scripts/check_docs.py` checks that every command and code reference in them
exists.

## Status and known limitations

Open bugs are tracked in [docs/ISSUES.md](docs/ISSUES.md). The main ones:

- **Stage detection is switched off** - the detector exists (stages: Battlefield,
  Small Battlefield, Final Destination, Pokémon Stadium 2, Smashville, Town and
  City, Kalos Pokémon League, Hollow Bastion), but every game currently reports
  `unknown`.
- **OCR misreads** - player tags under the damage number can be read as damage,
  and a leading digit can be clipped, causing missed or misattributed stock losses.
- **Final stocks** are often missed without Gemini: the last death of a game has
  no respawn or damage reset to detect.
- **Characters** are unreliable without Gemini (see Results).
- **Scene classifier training data** comes from a single labeled set; more
  labeled sets are the main path to better respawn detection.

## License

[MIT](LICENSE)
