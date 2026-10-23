# RunItBack - Developer Notes

This file contains important information for AI assistants and developers working on this project.

**For detailed development history, see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)**

## Project Overview

RunItBack is a Python CLI tool that analyzes Super Smash Bros. Ultimate (SSBU) replays from YouTube URLs. It identifies games within sets, detects stages and characters, and classifies game states (neutral/advantage/disadvantage) with stock loss tracking.

## Current Status (January 2026)

### Working Features
- ✅ Video download from YouTube (yt-dlp)
- ✅ Frame extraction at configurable intervals
- ✅ Game boundary detection (GO!/GAME! markers + white flash validation)
- ✅ Character detection (Gemini API + template matching fallback)
- ✅ **Game state detection** (neutral/advantage/disadvantage) via damage tracking
- ✅ **Stock loss tracking** - hybrid damage-reset + Smart Gemini detection
- ✅ **Multi-game set support** - automatic game boundary detection with GO!→GAME! pairing

### Known Issues (from test suite)
- ❌ Stage detection returns "unknown" for all stages (needs implementation)
- ⚠️ Stock loss detection has ~52-74% recall depending on video (OCR reliability issues)

### Improvement Path
The main bottleneck for stock loss accuracy is the scene classifier's respawn detection. More labeled training data will improve:
1. **Respawn detection** - Primary signal for stock losses
2. **Scene state accuracy** - Better neutral/advantage/disadvantage classification
3. **Edge case handling** - Reduce reliance on complex OCR-based heuristics

Use `tools/scene_labeler.py` to label more videos, then retrain with `tools/train_scene_classifier.py`.

### Detection Approach
- **Damage Tracking**: PaddleOCR reads damage percentages every 0.5s
- **Scene Classification**: ML model (ConvNeXt-Tiny ONNX) classifies positional states
- **State Classification**: Based on who took damage (advantage = you hit them)
- **Neutral Reset**: Position-based (`both_grounded` → neutral) with time-based fallback
- **Stock Loss Detection**: Confirmed by 2 consecutive low-damage frames after high damage
- **Segment Filtering**: Minimum duration thresholds reduce noise (1s neutral, 1.5s adv/disadv)

### Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `hit_threshold` | 4.0 | Min damage increase to count as hit |
| `neutral_reset_time` | 3.0s | Time without damage to return to neutral |
| `STOCK_LOSS_COOLDOWN` | 10.0s | Min time between stock losses per player |
| `CONFIRM_FRAMES` | 3 | Consecutive low-damage frames to confirm stock loss |
| `MIN_KILL_DAMAGE` | 50 | Minimum damage before stock loss is plausible |

### Performance Optimizations
- **Skip OCR outside game bounds**: Frames before GO! or after GAME! skip OCR entirely (active)
- **Parallel OCR**: Multiprocessing implementation exists but disabled (pool overhead issue)
- **Batch scene classification**: Scene classifier processes all frames in one batch

## Tournament-Legal Stages (Hazards Off)

### Stage Visual Characteristics

| Stage | Platforms | Notes |
|-------|-----------|-------|
| **Small Battlefield** | 2 (sides) | Similar appearance to Battlefield but fewer platforms |
| **Battlefield** | 3 (2 sides + 1 top center) | Very similar to Small Battlefield - use platform count to distinguish |
| **Final Destination** | 0 | Flat stage, no platforms |
| **Pokemon Stadium 2** | 2 (sides) | Platforms are symmetrical |
| **Smashville** | 1 (center) | Single stationary platform in the center |
| **Town and City** | 1-3 (variable) | Platforms can appear/disappear during gameplay. Can have 1 center, 2 sides, or combination. Platforms gradually move away during gameplay |
| **Kalos Pokemon League** | 2 (sides) | Two platforms off to the sides of the stage |
| **Hollow Bastion** | 1 (center) | Has time-of-day variations (warm sunset vs cool blue colors) |

### Stage Detection Considerations

1. **Battlefield vs Small Battlefield**: These stages have nearly identical color palettes and backgrounds. Histogram matching alone cannot distinguish them. Use **platform counting** as a tiebreaker:
   - Battlefield: 3 platforms
   - Small Battlefield: 2 platforms

2. **Variable Backgrounds**: Some stages change appearance during gameplay:
   - **Hollow Bastion**: Cycles through warm (pink/orange sunset) and cool (blue/teal) color schemes
   - **Town and City**: Background changes as platforms transition
   - **Pokemon Stadium 2**: Background can vary slightly

3. **Stream Overlays**: Tournament streams often have overlays (score displays, branding) that affect detection. Reference images extracted from actual gameplay work better than clean stage screenshots.

## HUD Detection

The SSBU gameplay HUD contains:
- **Damage percentages**: Bottom left (P1) and bottom right (P2)
- **Stock icons**: Above damage displays
- **Character portraits**: Near stock icons
- **Timer**: Top center

When these elements are visible, the frame is almost certainly gameplay.

## Game Boundary Detection

Games within a set are separated by:
- Victory screens ("GAME!" flash, character victory pose)
- Character select screens
- Stage select screens
- Stream transition overlays (often black screens with match info)

The current approach detects continuous gameplay segments using HUD presence and merges brief gaps (< 5 seconds) that occur during:
- Camera zooms on characters
- Intense visual effects
- Brief HUD occlusion

## Character Detection

Character detection uses Google's Gemini Vision API (gemini-2.5-flash) for accurate identification from gameplay screenshots.

### How It Works

1. **Primary: Gemini Vision API**
   - Samples frames at 10%, 25%, 50%, 75%, 90% through each game (5 samples for robust voting)
   - Uses few-shot prompting to identify characters from HUD portraits and in-game appearance
   - Handles varying skins, visual effects, and camera angles
   - Results are cached (by frame hash) to avoid redundant API calls

2. **Fallback: Template Matching**
   - Used when Gemini API is unavailable or fails
   - Matches HUD portraits against icon reference templates
   - Less accurate, especially for characters with many skins or similar appearances

### Configuration

- **API Key**: Set `GEMINI_API_KEY` environment variable. Read directly via
  `os.environ` at each call site - it is NOT a `Settings` field, so a `.env`
  entry will be silently ignored (`config.py` uses the `RUNITBACK_` prefix).
- **Disable Gemini**: Leave `GEMINI_API_KEY` unset. Detection logs a warning
  and falls back to template matching. `RUNITBACK_USE_GEMINI=0` also exists,
  but only `character_detector.py:23` reads it - it does not turn off the
  stock-loss or stage Gemini calls.
- **Current state**: No key is configured. Gemini-dependent paths are inactive,
  so character detection runs on template matching only, and the stock-loss
  Gemini post-pass (final-stock GAME! losses, low-percent kills) is off -
  expect GAME! losses to make up most stock-loss misses.
- **Cache Location**: `.cache/` directory (relative to project root)
  - `llm_results/` - Character detection cache
  - `stage_results/` - Stage detection cache
  - `stock_counter_results/` - Stock counter detection cache

### Cache Directories

Cache directories are excluded from git (see `.gitignore`):
- `.cache/` - Gemini API results (~6 MB, speeds up re-analysis)
- `.video_cache/` - Videos for the labeling tools (`scene_labeler.py`,
  `extract_labeled_frames.py`)

The analysis pipeline and the benchmark download elsewhere: `settings.cache_dir`,
default `~/.cache/runitback/`, outside the repo. Files there are named by URL
hash, so the two caches do not share videos.

### Key Files

- `runitback/cv/llm_detector.py`: Gemini Vision API integration
- `runitback/cv/character_detector.py`: Main detection logic with fallback

## Future ML Integration

The current architecture uses:
- **Gemini Vision API** for character detection
- **Rule-based detection** with reference image matching for stages

Future enhancements:
- CNN classifiers for stages (image classification)
- Gemini for game state classification (neutral/advantage/disadvantage)
- Gemini for edgeguard and ledgetrap situation detection
- Move recognition using Gemini with labeled training data

## Key Files

| File | Purpose |
|------|---------|
| `runitback/cli.py` | CLI interface and game detection logic |
| `runitback/classification/detector.py` | **Game state detection** (neutral/advantage/disadvantage) |
| `runitback/classification/states.py` | Data structures (GameState, FrameState, StateSegment, StockLoss) |
| `runitback/cv/ocr.py` | PaddleOCR damage percentage reading (with parallel batch support) |
| `runitback/cv/scene_classifier.py` | ML scene classification (ConvNeXt-Tiny ONNX) |
| `runitback/cv/ui_detector.py` | HUD detection, stock counting, frame type classification |
| `runitback/cv/stage_detector.py` | Stage detection with histogram matching and platform counting |
| `runitback/cv/character_detector.py` | Character detection (Gemini + fallback template matching) |
| `runitback/cv/llm_detector.py` | Gemini Vision API integration |
| `data/stage_references/` | Stage reference images for matching |
| `data/character_references/` | Character reference images for template matching fallback |
| `docs/DEVELOPMENT.md` | **Development chronicle** - methods tried, what worked/didn't |

## Available Skills

Custom Claude Code skills for common development tasks:

| Skill | Description |
|-------|-------------|
| `/test` | Run the pytest test suite with options for quick/full/pattern matching |
| `/analyze <url>` | Analyze a YouTube SSBU replay video |
| `/debug-video <url>` | Debug detection on a video with verbose logging and frame extraction |
| `/log-changes` | Add a dated entry to the `docs/DEVELOPMENT.md` Changelog: what was modified and why |

**After finishing any change to code, config, or tooling, run `/log-changes`
before committing**, so the DEVELOPMENT.md entry lands in the same commit as
the change it describes.

Skills are defined in `.claude/skills/<name>/SKILL.md`, each with YAML frontmatter (`name`, `description`). The directory-plus-`SKILL.md` layout is required — a flat `.claude/skills/<name>.md` file is not discovered.

## Secrets

Never commit credentials. A Gemini key was committed in `f8e1834` and sat in
public history for nine months before being removed; that key has since been
deleted and must not be reused.

Store keys outside the repo. Preferred (macOS):

```bash
security add-generic-password -a "$USER" -s GEMINI_API_KEY -w   # prompts, no echo
# then in ~/.zprofile:
export GEMINI_API_KEY="$(security find-generic-password -a "$USER" -s GEMINI_API_KEY -w 2>/dev/null)"
```

A `.env` file is gitignored but will not work for `GEMINI_API_KEY` without a
code change - see the Configuration note above.

### Pre-commit secret scan

`scripts/hooks/pre-commit` blocks commits containing probable secrets. It is
version-controlled but `core.hooksPath` is local config, so enable it once per
clone:

```bash
git config core.hooksPath scripts/hooks
```

It uses `detect-secrets` (in the `dev` extras) when available, and falls back to
provider-specific patterns otherwise. Bypass with `git commit --no-verify` only
for confirmed false positives.

## Known Limitations

1. **OCR Accuracy**: Damage reading can have errors during visual effects or motion blur (especially Steve's mining effects)
2. **Stock Loss Detection**: Relies on damage reset pattern; may miss edge cases. Respawn-based detection via scene classifier needs more training data
3. **Stage Detection**: Battlefield vs Small Battlefield disambiguation can be unreliable
4. **No Edgeguard/Ledgetrap Detection**: Requires character position tracking (future work)
5. **Player Attribution**: In some cases, stock losses are attributed to the wrong player due to OCR timing issues
