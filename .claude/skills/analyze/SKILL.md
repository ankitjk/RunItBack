---
name: analyze
description: Analyze a Super Smash Bros. Ultimate replay from a YouTube URL, detecting games, characters, stages, and stock losses. Use when given a YouTube URL to analyze or asked to run the RunItBack pipeline over a video.
---

# /analyze - Analyze a SSBU replay video

Run the full RunItBack pipeline over one YouTube replay and summarize the result.

## Usage

```
/analyze <youtube_url>                 # Asks which player's perspective first
/analyze <youtube_url> --player 1      # Player 1's perspective
/analyze <youtube_url> --player 2      # Player 2's perspective
/analyze <youtube_url> --games 1,3     # Only some games ('1', '1,3', '2-4')
```

## Instructions

1. **Get the player.** `analyze` stops to prompt for it if `--player` is
   missing, and Claude cannot answer that prompt. If the user gave no
   `--player`, ask them which player's perspective they want (1 = left HUD,
   2 = right HUD) before running.

2. **Run it** from the project root with the venv's interpreter - the venv is
   not activated in Claude's shell, and bare `python` is not on PATH:
   ```bash
   .venv/bin/python -m runitback analyze "<youtube_url>" --player <1|2> [--games <spec>]
   ```
   It is slow - roughly 20 minutes per set on first run, mostly download and
   OCR. Run it in the background and tell the user it has started.

3. **What it does:**
   - Downloads the video (cached in `~/.cache/runitback/`, so reruns skip this)
   - Samples a frame every 0.5s and finds each game from its GO! and GAME! screens
   - Identifies characters - Gemini if `GEMINI_API_KEY` is set, otherwise
     template matching (less accurate)
   - Reads damage with OCR, classifies each frame's scene, and detects stock
     losses and neutral/advantage/disadvantage segments
   - Prints a summary and timeline per game

4. **Summarize** for the user:
   - Number of games, and each game's characters and length
   - Stock losses per game, with timestamps and who lost them
   - Anything that looks wrong (a game count that does not match the set,
     "Unknown" characters, a game with no stock losses)

## Notes

- **Stage is always "unknown"** - stage detection is switched off (ISSUE-009).
- **No Gemini key is configured**, so character detection is template matching
  only, and low-percent kills that never show a damage reset can be missed.
  Say so when presenting results.
- **Template matching often names the wrong characters** (ISSUE-013). Compare
  the detected characters with the video title, which usually names them, and
  point out any mismatch rather than reporting the detection as fact.
- Known accuracy bugs are listed in `docs/ISSUES.md`. If a result looks wrong,
  `/debug-video` or the `ocr-investigator` agent can find out why.
