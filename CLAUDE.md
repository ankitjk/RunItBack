# RunItBack - Developer Notes

This file contains important information for AI assistants and developers working on this project.

## Project Overview

RunItBack is a Python CLI tool that analyzes Super Smash Bros. Ultimate (SSBU) replays from YouTube URLs. It identifies games within sets, detects stages and characters, and will eventually classify game states (neutral/advantage/disadvantage), edgeguarding, and ledgetrapping situations.

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

- **API Key**: Set `GEMINI_API_KEY` environment variable
- **Disable Gemini**: Set `RUNITBACK_USE_GEMINI=0` to force template matching
- **Cache Location**: `.cache/llm_results/` (relative to project root)

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

- `runitback/cv/stage_detector.py`: Stage detection with histogram matching and platform counting
- `runitback/cv/ui_detector.py`: HUD detection and frame type classification
- `runitback/cv/character_detector.py`: Character detection (Gemini + fallback template matching)
- `runitback/cv/llm_detector.py`: Gemini Vision API integration for character detection
- `runitback/cli.py`: CLI interface and game detection logic
- `data/stage_references/`: Stage reference images for matching
- `data/character_references/`: Character reference images for template matching fallback
