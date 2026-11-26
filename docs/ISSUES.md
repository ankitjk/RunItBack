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

**Next ID:** ISSUE-024

Severity: **High** = causes wrong results on real videos. **Medium** = blocks
measuring or improving accuracy. **Low** = tooling, docs, or performance.

---

## High

### ISSUE-022: Character select screen detected as GO!, so games start up to 22s early
- **Area:** Game boundaries - **Found:** 2026-11-26, running GO!/GAME! detection
  on `XUkRx6CpL04` (the labelled video; detection was tuned on the 6 benchmark sets)
- **What happens:** `detect_go_screen` fires on the character select screen
  (a grid of yellow and orange portraits in the centre). Pairing keeps the first
  GO! after the previous game, so each game starts at character select and its
  span takes in character select, stage select and the entrances.
- **Evidence:** `XUkRx6CpL04` games start at 8.5s / 219.5s / 450.5s; the real GO!
  screens are at 31.0s / 237.5s / 466.0s (15.5-22.5s early). Game count and ends
  are right. `test_timestamp_accuracy` allows 30s, so it would not catch this.
- **Where:** `runitback/cv/ui_detector.py:507` (`detect_go_screen`),
  `runitback/cli.py:272` (first GO! taken)
- **Suggested fix:** Accept a GO! only when gameplay-HUD frames follow it, or
  take the last GO! candidate before the HUD appears.

### ISSUE-023: A missed GO! drops a game; a missed GAME! merges two games
- **Area:** Game boundaries - **Found:** 2026-11-26, reading `detect_games`;
  confirmed with stubbed detectors
- **What happens:** Games are built only from GO! markers, and only the first
  GO! after each GAME! is kept. A game whose GO! is not detected is missing from
  the report. If a GAME! before the last game is missed, the next game's GO! is
  discarded and the two games are reported as one.
- **Evidence:** Synthetic 3-game set run through the real pairing code: game 2's
  GO! missed -> 2 games, game 2 absent; game 2's GAME! missed -> 2 games, the
  second spanning games 2 and 3. Not yet seen on a real video: every GO! and
  GAME! was detected in 23 games across 7 videos (2026-11-26).
- **Where:** `runitback/cli.py:266` (pairing), `runitback/cli.py:315` (games from GO! only)
- **Suggested fix:** Emit a game for a validated GAME! with no GO! before it
  (start at the gameplay-HUD onset), and keep every GO! that has gameplay after it.

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
- **Where:** `runitback/classification/detector.py:519` (and its uses at 540, 553, 566, 579)

### ISSUE-004: Stale max damage is carried forward
- **Area:** Stock-loss detection - **Found:** 2026-01-11 investigation
- **What happens:** `p1/p2_recent_max_damage` keeps a high value past the point
  it should reset. A later drop to 0% then looks like a kill from that stale
  value, crediting the death to the wrong player.
- **Evidence:** Steve vs Palutena - stale P2 max of 113 (from 88.5s) took credit
  for P1's death at 100.5s.
- **Where:** `runitback/classification/detector.py:486` (refill),
  `:609` (`p2_damage_before_respawn = p2_recent_max_damage`)
- **Note:** ISSUE-001 to 004 together cause the "right time, wrong player"
  attribution flips (2 in the October 23 benchmark). Re-check flips after fixing them.

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

### ISSUE-017: Inference hardcodes the deployed model's 7 classes
- **Area:** Scene classifier - **Found:** 2026-11-16, adding test-set evaluation to training
- **What happens:** `SceneClassifier.CLASS_ORDER` lists the 7 classes the
  deployed model was trained on, in ImageFolder's alphabetical order. A model
  retrained on today's labels has 9-10 outputs in a different order, so a
  prediction at index 7+ raises `IndexError` and lower indices are given the
  wrong names (index 3 is `both_on_ledge`, not `non_gameplay`). A retrained
  model cannot be deployed without hand-editing this list.
- **Where:** `runitback/cv/scene_classifier.py:101` (list), `:253` and `:318` (lookups)
- **Suggested fix:** Read the order from the `class_mapping.json` that training
  writes next to the model, falling back to the current list for the old model.

### ISSUE-006: Too little labeled data to retrain the scene classifier
- **Area:** Scene classifier - **Found:** 2026-10-23, `label-auditor`
- **What happens:** The whole corpus is one video (`XUkRx6CpL04`). `respawning`
  - the class meant to become the primary stock-loss signal - has 107
  frames (63 before the GAME! screens were relabelled, 2026-11-25) against a
  ~200 floor. 7 of 11 classes are under 200;
  `both_knocked_down` has none.
- **Suggested fix:** Label 3-5 more videos with varied stages and characters,
  favoring games with many stock losses (`tools/scene_labeler.py`).

### ISSUE-007: YouTube's bot check can block benchmark downloads
- **Area:** Benchmark - **Found:** 2026-10-23 benchmark
- **What happens:** YouTube's "Sign in to confirm you're not a bot" check
  sometimes blocks yt-dlp. On October 23 it blocked Cloud vs Pit
  (`KfMZ9bQUbtk`) and Mario vs Ness (`UgGQkF1wkoA`); both were downloaded for the October
  24 benchmark, and all 6 sets are now cached in `~/.cache/runitback/`.
  A new clone or a cleared cache can hit it again, and the run is then partial
  (see ISSUE-008).
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
- **Where:** `runitback/cv/ocr.py:319`

---

## Fixed

One line per fixed issue, newest first:
`- ISSUE-NNN: <title> (fixed YYYY-MM-DD, DEVELOPMENT.md <Month D, YYYY>)`

- ISSUE-021: Re-extracting frames after a label edit leaves stale frames (fixed 2026-11-25, DEVELOPMENT.md November 25, 2026)
- ISSUE-020: Exported scene model only accepts a batch of 1 (fixed 2026-11-25, DEVELOPMENT.md November 25, 2026)
- ISSUE-019: Retraining on current labels collapses onto a 1-frame class; `class_mapping.json` written in the wrong order (fixed 2026-11-16, DEVELOPMENT.md November 16, 2026)
- ISSUE-018: Random frame-level split leaked near-duplicate frames; test split never evaluated (fixed 2026-11-16, DEVELOPMENT.md November 16, 2026)
- ISSUE-016: Stock cap keeps the earliest losses, dropping a real final stock (fixed 2026-10-24, DEVELOPMENT.md October 24, 2026)
- ISSUE-005: Final-stock losses missed without a Gemini key (fixed 2026-10-24, DEVELOPMENT.md October 24, 2026)
- ISSUE-012: `/analyze` skill has broken commands (fixed 2026-10-23, DEVELOPMENT.md October 23, 2026)
- ISSUE-010: `/debug-video` skill has broken commands (fixed 2026-10-23, DEVELOPMENT.md October 23, 2026)
