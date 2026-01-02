# /analyze - Analyze a SSBU replay video

Analyze a Super Smash Bros. Ultimate replay video from a YouTube URL.

## Usage

```
/analyze <youtube_url>           # Analyze with default settings
/analyze <youtube_url> --player 1  # Analyze from Player 1's perspective
/analyze <youtube_url> --player 2  # Analyze from Player 2's perspective
```

## Instructions

1. Run the analysis command:
   ```bash
   ./venv/bin/python -m runitback analyze "<youtube_url>"
   ```

2. If `--player` is specified, add it to the command:
   ```bash
   ./venv/bin/python -m runitback analyze "<youtube_url>" --player <1|2>
   ```

3. The tool will:
   - Download the video (cached in `.video_cache/`)
   - Extract frames and detect game boundaries
   - Identify characters and stages
   - Track stock losses throughout each game
   - Output a timeline per game

4. After running, summarize the results:
   - Number of games detected
   - Characters identified
   - Key events per game

## Example Output

```
═══════════════════════════════════════
GAME 1
Stage: Unknown
Player 1: Mario | Player 2: Fox
═══════════════════════════════════════
Game Start: 0:15
Stock Events:
  0:45 - P2 loses stock (3 → 2)
  1:12 - P1 loses stock (3 → 2)
  ...
Game End: 2:30
```

## Notes

- First analysis of a video takes longer (downloads video, makes API calls)
- Subsequent runs use cached results from `.cache/` and `.video_cache/`
- Set `GEMINI_API_KEY` environment variable for Gemini Vision detection
