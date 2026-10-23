# Known Issues

Open bugs and problems in RunItBack - what is still to be worked on. Maintained
with the `/issues` skill.

- Bugs only. When one is fixed, its entry is replaced by a single line in the
  **Fixed** list at the bottom. The full story of the fix goes in the
  `docs/DEVELOPMENT.md` Changelog, citing the issue ID.
- A fixed bug that comes back is **reopened** under its old ID - moved back up,
  with a "Reopened" note - rather than filed again.
- IDs are never reused. The next ID is below; increment it when adding.
- Every commit message must say `Fixes: ISSUE-NNN`, `Found: ISSUE-NNN`, or
  `Issues: none`. `scripts/hooks/commit-msg` rejects a commit whose line does
  not match this file. `git log --grep "ISSUE-NNN"` finds every commit for an issue.
- Planned features (edgeguard detection, Slippi, etc.) are not issues - they live
  under "Future Improvements" in `docs/DEVELOPMENT.md`.

**Next ID:** ISSUE-016

Severity: **High** = causes wrong results on real videos. **Medium** = blocks
measuring or improving accuracy. **Low** = tooling, docs, or performance.

---

## High

### ISSUE-001: OCR reads player tags as damage
- **Area:** OCR - **Found:** 2026-10-23, `ocr-investigator` on Steve vs Palutena 1:39
- **What happens:** The damage regions extend down to y=0.98 and take in the
  player tag printed under each damage number. The parser then turns letters
  into digits (O->0, L->1, B->8, G->6), so the tag "gleeb glob" parses as 61%
  and "rohan" as 0%. A phantom 0% fakes a stock-loss reset; a phantom 61%
  blocks the `< 20` respawn check.
- **Evidence:** Steve vs Palutena game 1 - false P2 loss at 86s, P2's real death
  at ~89s hidden by the resulting cooldown, P1's death at 99s credited to P2.
- **Where:** `runitback/cv/ui_detector.py:74-75` (ROIs),
  `runitback/cv/ocr.py:135-137` (letter substitutions)
- **Suggested fix:** End the ROIs above the tag line, and/or reject OCR text
  whose letters outnumber its digits before substituting.

### ISSUE-002: Merged decimals make real high damage unreadable
- **Area:** OCR - **Found:** 2026-10-23, same investigation as ISSUE-001
- **What happens:** The small decimal digit merges with the number - `118.9%`
  is read as `1189`, `132.0%` as `1320`. Anything over 999 is discarded, so a
  player at high percent often has no damage reading at all - exactly when a
  kill is likely.
- **Where:** `runitback/cv/ocr.py:145` (`0 <= value <= 999`)
- **Suggested fix:** For a 4-digit read, try it as `NNN.N` before discarding.

### ISSUE-014: Leading damage digit lost at the ROI's left edge
- **Area:** OCR - **Found:** 2026-10-23, writing fixture-based tests
- **What happens:** P1 showing 118.9% is read as 18.9%. In the saved crop the
  leading "1" sits on the left edge of the damage ROI, partly cut off, and OCR
  drops it. Cause not confirmed beyond that observation.
- **Evidence:** `tests/fixtures/frames/damage_88.5s_p1.png` (Steve vs Palutena
  88.5s); `test_leading_digit_not_lost` is an expected failure until fixed.
- **Where:** `runitback/cv/ui_detector.py:74` (`P1_DAMAGE_ROI` x-range starts at 0.27)
- **Suggested fix:** Widen the ROI's left edge, checking the crop does not then
  take in the character portrait. Fix alongside ISSUE-001, which changes the same ROIs.

### ISSUE-003: `both_reset` guard blocks a real stock loss
- **Area:** Stock-loss detection - **Found:** 2026-01-11 investigation
- **What happens:** When both players read 0% - one from an earlier death, one
  from a new one - the guard treats it as a game transition and the new death
  is missed. ISSUE-001's phantom 0% readings make this fire more often.
- **Evidence:** Steve vs Palutena 102.0s - log reads "Game transition detected
  (both players reset to 0%)" mid-game.
- **Where:** `runitback/classification/detector.py:506` (and its uses at 527, 540, 553)

### ISSUE-004: Stale max damage is carried forward
- **Area:** Stock-loss detection - **Found:** 2026-01-11 investigation
- **What happens:** `p1/p2_recent_max_damage` keeps a high value past the point
  it should reset. A later drop to 0% then looks like a kill from that stale
  value, crediting the death to the wrong player.
- **Evidence:** Steve vs Palutena - stale P2 max of 113 (from 88.5s) took credit
  for P1's death at 100.5s.
- **Where:** `runitback/classification/detector.py:473` (refill),
  `:596` (`p2_damage_before_respawn = p2_recent_max_damage`)
- **Note:** ISSUE-001 to 004 together cause the "right time, wrong player"
  attribution flips (2 in the October 23 benchmark). Re-check flips after fixing them.

### ISSUE-005: Final-stock losses missed without a Gemini key
- **Area:** Stock-loss detection - **Found:** 2026-10-23 benchmark
- **What happens:** The last stock of a game has no damage reset to detect.
  Only the Gemini post-pass catches it, and with no key configured that pass
  returns nothing.
- **Evidence:** 9 of 17 misses in the October 23 benchmark (4 videos, Gemini
  off) were final-stock GAME! losses.
- **Where:** `runitback/classification/detector.py:1063` (`_smart_gemini_postprocess`)
- **Suggested fix:** A non-Gemini signal - the GAME! marker is already detected
  in `cli.py`; the player whose damage was last rising at GAME! lost the stock.

### ISSUE-013: Template-matching fallback misidentifies characters and ignores title hints
- **Area:** Character detection - **Found:** 2026-10-23, verifying the fixed `/analyze` skill
- **What happens:** With no Gemini key, character detection falls back to
  template matching, which named Steve vs Palutena as "Little Mac vs Zero Suit
  Samus". The video title names both characters and they are parsed as hints
  (`Title hints - P1: ['Palutena', 'Steve']`), but the fallback is called
  without them, so the title cannot correct it.
- **Evidence:** `analyze nmzdyQz6msI --player 1 --games 1` - game 1 header reads
  "You: Little Mac | Opponent: Zero Suit Samus".
- **Where:** `runitback/cv/character_detector.py:1264`
  (`_detect_characters_template_matching(frames, sample_positions)` - hints not passed)
- **Suggested fix:** Pass the title hints to the fallback and restrict matching
  to the hinted characters when there are exactly two.

## Medium

### ISSUE-015: OCR engine failures are swallowed per frame
- **Area:** OCR - **Found:** 2026-10-23, first CI run
- **What happens:** If PaddleOCR itself is broken (bad install, incompatible
  model), every read raises, `read_damage_percentage` logs a warning and returns
  no value, and the pipeline carries on. The analysis completes with no damage
  data instead of failing, so a broken install looks like a run with poor results.
- **Evidence:** CI on `a1def5b` installed paddleocr 3.7.0 (PP-OCRv6); every OCR
  call failed with `ConvertPirAttribute2RuntimeAttribute not support`, and the
  only symptom was two tests reading `None`.
- **Where:** `runitback/cv/ocr.py:209` (`except` -> warning, `return None`)
- **Suggested fix:** Distinguish "no text in this crop" from "the engine failed":
  re-raise, or abort after N consecutive engine errors.

### ISSUE-006: Too little labeled data to retrain the scene classifier
- **Area:** Scene classifier - **Found:** 2026-10-23, `label-auditor`
- **What happens:** The whole corpus is one video (`XUkRx6CpL04`). `respawning`
  - the class meant to become the primary stock-loss signal - has ~63
  estimated frames against a ~200 floor. 7 of 11 classes are under 200;
  `both_knocked_down` has none.
- **Suggested fix:** Label 3-5 more videos with varied stages and characters,
  favoring games with many stock losses (`tools/scene_labeler.py`).

### ISSUE-007: Two benchmark videos cannot be downloaded
- **Area:** Benchmark - **Found:** 2026-10-23 benchmark
- **What happens:** YouTube's "Sign in to confirm you're not a bot" check blocks
  yt-dlp for Cloud vs Pit (`KfMZ9bQUbtk`) and Mario vs Ness (`UgGQkF1wkoA`).
  There is no full 6-video baseline for the current (Gemini off) setup.
- **Suggested fix:** Pass yt-dlp `--cookies-from-browser`. Uses your YouTube
  login - a decision for the user.

### ISSUE-008: Benchmark summary silently omits failed videos
- **Area:** Benchmark - **Found:** 2026-10-23 benchmark
- **What happens:** A video that errors is printed and skipped, and the summary
  recall is computed over the videos that ran. The October 23 run printed
  "Recall: 72.6% (45/62)" with no sign that 27 ground-truth losses were left out.
- **Where:** `tools/test_stock_losses.py:453` (`except Exception`)
- **Suggested fix:** Print "N of 6 videos analyzed" in the summary and exit
  non-zero when any video failed.

### ISSUE-009: Stage detection is switched off
- **Area:** Stage detection - **Found:** pre-existing (`TODO` in code)
- **What happens:** Every game reports `Stage.UNKNOWN`. `stage_detector.py`
  exists but the call is disabled, and there are no stage test images.
- **Where:** `runitback/cli.py:440`, `tests/test_game_detection.py:571`

## Low

### ISSUE-011: Parallel OCR is disabled
- **Area:** Performance - **Found:** 2026-01-06
- **What happens:** `read_damage_batch_parallel` exists but is not used because
  process-pool overhead outweighed the gain. A benchmark run takes ~20 min per video.
- **Where:** `runitback/cv/ocr.py:285`

---

## Fixed

One line per fixed issue, newest first:
`- ISSUE-NNN: <title> (fixed YYYY-MM-DD, DEVELOPMENT.md <Month D, YYYY>)`

- ISSUE-012: `/analyze` skill has broken commands (fixed 2026-10-23, DEVELOPMENT.md October 23, 2026)
- ISSUE-010: `/debug-video` skill has broken commands (fixed 2026-10-23, DEVELOPMENT.md October 23, 2026)
