---
name: debug-video
description: Debug RunItBack detection on a video using verbose logging, saved debug frames, or single-timestamp inspection. Use when detection output is wrong and the cause needs diagnosing - bad game boundaries, wrong characters, or missed stock losses.
---

# /debug-video - Debug detection on a specific video

See what RunItBack's detectors saw on a SSBU replay, either across a whole run
(verbose log) or at chosen timestamps (frames, damage crops, detector output).

For a full diagnosis of one missed or misattributed stock loss, delegate to the
`ocr-investigator` agent instead - it does this end to end and checks the known
failure modes.

## Usage

```
/debug-video <youtube_url>                          # Verbose log of a full run
/debug-video <youtube_url> --games <n>              # Same, one game only (faster)
/debug-video <youtube_url> --timestamp <sec> [...]  # Inspect specific moments
```

Run every command from the project root with `.venv/bin/python` - the venv is
not activated in Claude's shell, and bare `python` is not on PATH.

## Instructions

### Verbose log of a run

1. Run the analysis with DEBUG logging, writing to a log file. Always pass
   `--player` - without it `analyze` stops to ask, and Claude cannot answer.
   Pass `--games` if the problem is in one game; a full set takes ~20 min.
   ```bash
   .venv/bin/python -m runitback analyze "<youtube_url>" --player 1 --verbose [--games <n>] > debug_frames/run.log 2>&1
   ```
   `--verbose` (`-v`) is what turns on DEBUG logging. There is no `--debug` flag.

2. Grep the log for the scenario rather than reading it whole - it has
   thousands of lines:

   | Looking at | Grep for |
   |------------|----------|
   | Game boundaries | `GO! detected\|GAME! detected\|Paired GO!\|Skipping GO!\|Validated GAME!` |
   | Stock losses | `stock loss detected\|Game transition detected` |
   | OCR damage reads | `OCR detected` (raw text and confidence, per read) |
   | Characters | `Title hints\|falling back to template` (the result is in each game's header: `You: ... \| Opponent: ...`) |

   ```bash
   grep -n "GO! detected\|GAME! detected\|Paired GO!" debug_frames/run.log
   ```

3. Summarize what the lines show; quote only the few that matter.

### Inspect specific timestamps

1. Run the inspection script with one or more timestamps in seconds:
   ```bash
   .venv/bin/python .claude/skills/debug-video/inspect_frames.py "<youtube_url>" <sec> [<sec> ...]
   ```
   For each timestamp it prints the frame type, GO!/GAME! detection, each
   player's damage (parsed value, confidence, and the **raw OCR text**), and
   the scene classification. It saves to `debug_frames/`:
   - `<sec>s.png` - the full frame
   - `<sec>s_p1_roi.png`, `<sec>s_p2_roi.png` - the exact crops OCR read

2. Read the saved images to compare what was on screen with what was detected.
   The ROI crops are the most useful: they show exactly what OCR was given.

Around a stock loss, inspect several points from ~15s before to ~10s after -
the damage drop that confirms a stock loss comes *after* the kill. For
example `... 84 88 92 96 100 104`.

## Common Debug Scenarios

### Game boundaries incorrect
- Grep the boundary lines; check each GO! is paired with the next GAME!
- Look for replay-menu GO! screens being kept. `Skipping GO!` lines show GO!
  markers dropped because the game after them was too short
- Inspect the suspect timestamps to see whether GO!/GAME! detection fired

### Character detection wrong
- With no `GEMINI_API_KEY`, characters come from template matching - say so,
  it is the usual cause
- Inspect a mid-game timestamp and look at the HUD portraits in the frame
- Check whether P1/P2 are swapped rather than misidentified

### Stock loss missed or credited to the wrong player
- Inspect frames around the kill; compare the ROI crops with the parsed damage
- Raw OCR text containing letters (a player tag) means the known tag-as-damage
  bug - see `docs/ISSUES.md`, ISSUE-001
- Check `docs/ISSUES.md` for the other known stock-loss bugs before theorizing

## Notes

- Videos for `analyze` cache in `~/.cache/runitback/` (`settings.cache_dir`);
  a second run on the same video skips the download.
- `debug_frames/` is gitignored.
- PaddleOCR prints its own startup messages (model checks, ccache warning) -
  they are harmless.
- A real bug found while debugging goes into `docs/ISSUES.md` via `/issues add`.
