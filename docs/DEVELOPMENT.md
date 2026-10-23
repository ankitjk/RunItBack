# RunItBack Development Documentation

This document chronicles the development process for RunItBack, including methods tried, what worked, what didn't, and the current state of the codebase.

**Last Updated:** October 23, 2026

---

## Changelog

### October 23, 2026 - Agent Test Run, Prompt Fixes, Change Log & Issue Tracking

**Session Goal:** Test the three read-only agents added in `3243123` on real work, fix what they got wrong, and add a way to keep this changelog current.

**Problem:**
- The agent prompts were written without being run. Running them showed commands that could not work:
  - `ocr-investigator` told the agent to use `analyze --debug`. There is no such flag; `--verbose` sets DEBUG (`cli.py:810`).
  - It omitted `--player`, so `analyze` stops to prompt and a subagent cannot answer.
  - It pointed at the extraction snippets in `debug-video/SKILL.md`, which import a `FrameExtractor` class that does not exist and call `download_video` without its required `cache_dir`.
  - All three agents ran bare `python`, which is not on PATH; the project venv is `.venv/` and is not activated in subagent shells.
  - Two agents said videos cache in `.video_cache/`. The pipeline actually caches in `settings.cache_dir` (`~/.cache/runitback/`).
- `benchmark-runner` assumed about 10 minutes per video; the real figure was about 20.
- `benchmark-runner` assumed turning Gemini off only affects character detection. It also disables `_smart_gemini_postprocess` in `detector.py`, which catches final-stock GAME! losses and low-percent kills.
- The script also skips videos YouTube refuses to serve, and still prints a total, so a partial run can look complete.
- CLAUDE.md said `RUNITBACK_USE_GEMINI` was never implemented, but `character_detector.py:23` reads it.

**Changes:**
- `.claude/agents/ocr-investigator.md`:
  - Corrected the run command: `--verbose`, `--player 1`, output to a log file.
  - Added a working frame-extraction snippet (`extract_single_frame`) and a note not to copy the broken skill snippets.
  - Corrected the cache path and the interpreter.
  - Added known failure mode 5, "player tags read as damage", from this session's diagnosis (below).
- `.claude/agents/benchmark-runner.md`:
  - Runtime estimate is now about 20 minutes per video.
  - Explains what Gemini-off does to stock-loss detection.
  - Requires partial runs to be labelled as partial, with the skipped videos listed.
  - Corrected the interpreter and cache path.
- `.claude/agents/label-auditor.md` - Corrected the interpreter.
- `.claude/skills/log-changes/SKILL.md` (new): writes a dated Changelog entry, what changed and why, then prints a rundown.
  - It is a skill, not an agent. The "why" lives in the conversation, and a subagent starts with a blank context, so it would only see the diff.
- `CLAUDE.md`:
  - Lists `/log-changes`, with an instruction to run it before committing.
  - Corrected the `RUNITBACK_USE_GEMINI` statement.
  - States that Gemini-off also disables the stock-loss post-pass.
  - Corrects the cache description: two caches, `.video_cache/` for labeling tools and `~/.cache/runitback/` for the pipeline.

- `docs/ISSUES.md` (new): a single list of open bugs, by severity. It is seeded with 11 issues (ISSUE-001 to 011), taken from this session's agent reports, the January 11 investigation, and code TODOs.
  - Bugs only; features stay in Future Improvements.
  - Fixed issues move to a one-line Fixed list rather than being deleted. A regression can then be reopened under its old ID, and old changelog references still resolve.
- `scripts/hooks/commit-msg` (new): every commit message must carry `Issues: none`, `Fixes: ISSUE-NNN` or `Found: ISSUE-NNN`, and the line must match the staged ISSUES.md.
  - Added because skills and CLAUDE.md only *ask* for the tracker to be updated, and nothing checked that it was.
  - The hook can't judge whether something is a bug, but it forces an explicit answer at every commit and rejects one that contradicts the file.
  - Tested on 13 cases: missing line, typo, unknown ID, half-moved fix, merge commit, and others.
- `.claude/skills/issues/SKILL.md` (new): `/issues` lists issues, adds them (verifying `file:line` first), moves them to Fixed once a fix is confirmed, and reopens regressions.
- `.claude/skills/log-changes/SKILL.md`: new steps that move fixed issues to the Fixed list by ID, file Left Open items into ISSUES.md, and give the commit's issue line.
- `CLAUDE.md`:
  - The Known Issues list is replaced with a pointer to `docs/ISSUES.md`, so two lists can't drift apart.
  - That list said stage detection "needs implementation". It is actually implemented but switched off at `cli.py:443`.

**Results (agent test runs, Gemini off):**
- **label-auditor** - Worked as briefed.
  - Corpus is one video (`XUkRx6CpL04`, 645s labeled). `respawning` is about 63 estimated frames, far under the ~200 floor.
  - 7 of 11 classes are under 200 frames; `both_knocked_down` has none.
  - Verdict: label 3-5 more varied videos before retraining.
- **ocr-investigator** - Steve vs Palutena, P1 death at 1:39. Detected at the right time but credited to P2 (mode 4). The root cause is new:
  - The damage ROIs (`ui_detector.py:74-75`, down to y=0.98) include the player tag under each number.
  - The parser's letter-to-digit substitutions (`ocr.py:125-127`) turn "gleeb glob" into 61 and "rohan" into 0.
  - The phantom 0 faked a P2 reset at 86s, which started a cooldown that hid P2's real death at about 89s.
  - The phantom 61 failed P1's `< 20` respawn check at 100.5s, so the stale P2 max (113) took the credit.
  - Merged decimals (`118.9` read as `1189`) are dropped as over 999.
- **benchmark-runner** - Partial run, 4 of 6 videos. YouTube's "confirm you're not a bot" check blocked Cloud vs Pit and Mario vs Ness.

  | Video | Recall | FP |
  |-------|--------|----|
  | Ken vs King Dedede | 100% (8/8) | 0 |
  | Ken vs Sonic | 77% (10/13) | 4 |
  | Dark Pit vs Ken | 83% (15/18) | 4 |
  | Steve vs Palutena | 52% (12/23) | 15 |
  | **4-video total** | **72.6% (45/62)** | **23** |

  - Not comparable to the January 11 table, which was measured with a Gemini key.
  - 9 of the 17 misses are final-stock GAME! losses, consistent with the Gemini post-pass being off.
  - 2 misses are right-time-wrong-player (Steve vs Palutena 1:39 and 2:09).

**Left Open** (all tracked in `docs/ISSUES.md`):
- Tag-as-damage OCR bug, diagnosed but not fixed - ISSUE-001
- Merged decimals dropped as over 999 - ISSUE-002
- Final-stock losses missed with Gemini off - ISSUE-005
- Two benchmark videos blocked by YouTube - ISSUE-007
- Benchmark summary omits failed videos without saying so - ISSUE-008
- Template-matching fallback misidentifies characters and drops title hints - ISSUE-013 (found during the skill fixes below)

**Follow-up: `/debug-video` and `/analyze` skills fixed (ISSUE-010, ISSUE-012)**

*Problem:* every command in both skills failed.
- Both ran bare `python`, which is not on PATH.
- `/debug-video` used a `--debug` flag, and imported a `FrameExtractor` class and a `detect_game_screen_with_gemini` function. `git log -S` shows none of the three has ever existed: the skill was written without being run in `48c7fc1`.
- Neither skill passed `--player`, so `analyze` blocks on a prompt Claude cannot answer.
- `/analyze` gave the wrong video cache path.
- The skills went unnoticed because they were undiscoverable until `307c66a`, which fixed the layout but did not run them.

*Changes:*
- `.claude/skills/debug-video/SKILL.md` - rewritten.
  - `.venv/bin/python`; `--verbose --player 1 [--games N]` written to a log file.
  - A grep table for each scenario. Every pattern was checked against real log messages.
  - Points to `ocr-investigator` for full diagnoses, and to ISSUES.md for known bugs.
  - Fixes ISSUE-010.
- `.claude/skills/debug-video/inspect_frames.py` (new) - replaces the broken inline snippets.
  - For each timestamp it saves the frame and both damage-ROI crops, and prints frame type, GO!/GAME!, raw and parsed OCR per player, and scene class.
  - A script rather than `python -c`, so it can be run and kept correct.
- `.claude/skills/analyze/SKILL.md` - rewritten.
  - `.venv/bin/python`; asks the user for `--player` when it isn't given; adds `--games`.
  - Correct cache path, and an accurate list of what the pipeline does.
  - Warns about ISSUE-005, ISSUE-009 and ISSUE-013 when presenting results.
  - Fixes ISSUE-012.

*Verification (both commands run, not just read):*
- `inspect_frames.py nmzdyQz6msI 88.5 98.0` ran in about 10s. At 98.0s it reproduced ISSUE-001 exactly: P1 raw `'gleeb glob'`, parsed 61.0.
- `analyze nmzdyQz6msI --player 1 --games 1 --verbose` exited 0 in 4m47s and logged 1,102 OCR lines.
- Every grep pattern in the skill matched the log, except the original characters pattern. That one hit 2,184 lines, almost all "No Gemini API key" warnings, so it was replaced with `Title hints\|falling back to template`, which matches 6.
- The same run named Steve vs Palutena as "Little Mac vs Zero Suit Samus", which led to ISSUE-013.

**Follow-up: instruction checker and scaled change log**

*Changes:*
- `scripts/check_docs.py` (new) - checks every command and reference in skills, agents, CLAUDE.md and ISSUES.md against the code. It covers interpreter, CLI subcommands and flags, `--player`, `runitback` imports (read with `ast`, nothing imported), script paths, and `file.py:line` refs.
  - Why: every broken skill and agent command this session was a name nobody checked.
  - Run on the pre-fix files from `3243123`/`c318f1c`, it reports all 14 mechanical faults.
  - Two faults it cannot catch: the prose-only wrong cache path, and `download_video` missing an argument (it checks names, not call signatures).
- `scripts/hooks/pre-commit` - runs the checker (~0.3s) only when a commit touches instructions or the code they reference.
- `tests/test_docs.py` (new) - the repo must pass the checker, and the checker must still catch the six original mistakes.
- `.claude/skills/test/SKILL.md` - bare `python` changed to `.venv/bin/python`. The checker's first run found this; the skill had never worked here.
- `.claude/skills/log-changes/SKILL.md` - a "Scale to the change" table: small, medium and large changes get one bullet, a short entry, or the full template. Added because recording every typo in full would slow development more than it helps.
- `CLAUDE.md` - documents the checker and the rule "never write a command into instructions without running it first".

---

### January 11, 2026 - Stock Loss Detection Investigation

**Session Goal:** Investigate and improve stock loss detection accuracy

**Problem:**
- Stock loss detection accuracy varied widely across videos (52-100% recall)
- Steve vs Palutena video had worst performance (52% recall, 16 false positives)
- Player attribution errors: deaths detected at correct time but wrong player

**Investigation Findings:**

**Test Results Across Videos:**
| Video | Recall | Notes |
|-------|--------|-------|
| Ken vs King Dedede | 100% (8/8) | Best performance |
| Ken vs Sonic | 85% (11/13) | Good |
| Dark Pit vs Ken | 78% (14/18) | OK |
| Steve vs Palutena | 52% (12/23) | Poor - Steve's effects cause OCR issues |
| Cloud vs Pit | 93% (13/14) | Good |
| Mario vs Ness | 69% (9/13) | OK |

**Root Causes Identified:**

1. **OCR Unreliability with Visual Effects**
   - Steve's mining blocks and particles cause OCR misreads (e.g., 13% read as 113%)
   - Damage tracking accumulates errors over time
   - `p2_recent_max_damage` carries over old values incorrectly

2. **`both_reset` Bug**
   - When both players are at 0% (one from earlier death, one from new death), `both_reset` blocks detection
   - Stock loss for the newly-dead player is missed
   - Tracking variables not reset correctly after detection

3. **Player Attribution Errors**
   - Deaths detected at correct timestamps but attributed to wrong player
   - Pattern: P1 expected → P2 detected (or vice versa)
   - Caused by accumulated OCR errors in max damage tracking

**Decision:**

Rather than adding more complexity to the OCR-based detection (which was becoming a tangle of edge cases), decided to focus on **improving the scene classifier** via more labeled training data.

**Rationale:**
- Scene classifier's `respawning` detection is a simpler, more reliable signal
- Once model accuracy improves, respawn detection becomes primary stock loss signal
- Reduces reliance on fragile OCR-based heuristics
- More labeled data → better model → cleaner detection logic

**Action Items:**
1. Revert complex stock loss detection changes
2. Create more labeled training data using `tools/scene_labeler.py`
3. Retrain scene classifier with additional data
4. Shift to respawn-based detection as primary method

**Files Modified:**
- None (reverted all changes)

---

### January 6, 2026 - Performance Optimizations & Scene Classification

**Session Goal:** Improve analysis performance by optimizing OCR processing

**Problem:**
- Analysis took very long on videos with significant non-gameplay content (menus, character select, etc.)
- PaddleOCR is the main bottleneck (~100-200ms per frame)
- OCR was running on all frames including non-gameplay segments

**Analysis of Bottlenecks:**
- Scene classification: ~15-20ms per frame (already fast)
- Stock count detection: ~5-10ms per frame
- **PaddleOCR: ~100-200ms per frame** (main bottleneck)

**Optimization Options Evaluated:**

| Option | Description | Speedup | Status |
|--------|-------------|---------|--------|
| 1 | Skip OCR outside game boundaries | Variable (10-50%) | ✅ Implemented |
| 2 | Lower OCR sample rate | Not viable | Rejected (damage can change every 0.5s) |
| 3 | Batch OCR | Not feasible | Rejected (PaddleOCR doesn't support batching) |
| 4 | Custom digit recognizer | 10-20x | Deferred (requires training) |
| 5 | Multiprocessing OCR | 2-4x | ⚠️ Implemented but disabled (pool overhead) |

**Option 1: Skip OCR Outside Game Boundaries**

Uses GO!/GAME! detection (already implemented) to skip OCR for frames outside game bounds:
- Frames before first GO! screen → skip OCR
- Frames after GAME! screen → skip OCR
- Only process frames during actual gameplay

**Implementation:**
- Added `_is_within_game_bounds()` method to `GameStateDetector`
- Modified `_extract_frame_states()` to check bounds before adding to OCR queue
- Logs count of skipped frames for visibility

**Option 5: Multiprocessing OCR**

Uses Python multiprocessing to distribute OCR across multiple workers:
- Each worker initializes its own PaddleOCR instance
- Spawns worker pool with `spawn` context (required for macOS)
- Falls back to sequential processing for small batches (<4 frames)

**Implementation:**
- Added `read_damage_batch_parallel()` function to `ocr.py`
- Added `_init_ocr_worker()` and `_process_single_roi()` helper functions
- Added `use_parallel_ocr` and `ocr_workers` parameters to `GameStateDetector`

**Issue Found:** Worker pool creation overhead negates parallelization benefits
- PaddleOCR initialization takes ~10-20 seconds per worker
- Current implementation creates new pools for each batch (P1 and P2 separately)
- This adds significant overhead rather than reducing processing time
- **Disabled by default** - requires persistent worker pool architecture to be effective

**Files Modified:**
- `runitback/cv/ocr.py` - Added parallel OCR functions
- `runitback/classification/detector.py` - Integrated both optimizations

**Actual Performance:**
- Option 1 alone provides benefit by skipping non-gameplay frames
- 8-minute video analyzed in ~10:37 with Option 1 only

---

### January 5, 2026 - Scene Classifier Training & Integration

**Session Goal:** Train ML model for position-based game state detection

**Problem:**
- Time-based neutral reset (3s without damage) was inaccurate
- No way to detect specific disadvantage situations (juggle, offstage, ledgetrap)
- Needed position-aware state classification

**Solution:** ConvNeXt-Tiny scene classifier trained on labeled gameplay frames

**Tools Created:**

1. **`tools/scene_labeler.py`** - PyQt5 video labeling application
   - Keyboard shortcuts for segment labeling (1-9 for classes)
   - Video playback with speed control (0.25x-2x)
   - Stage tagging per game
   - Export to JSON format

2. **`tools/extract_labeled_frames.py`** - Frame extraction from labels
   - Extracts frames at 0.5s intervals
   - Creates train/val/test splits (70/15/15)
   - Handles class balancing

3. **`tools/train_scene_classifier.py`** - Training script
   - ConvNeXt-Tiny with ImageNet pretrained weights
   - Mixed precision training (AMP)
   - Cosine learning rate scheduler
   - Early stopping

**Scene Classes (7 trained):**
| Class | Description |
|-------|-------------|
| `both_grounded` | Both players on stage, feet on ground → NEUTRAL |
| `one_airborne` | One in air (juggle situation) |
| `both_airborne` | Both in air (active exchange) |
| `one_offstage` | One past ledge (edgeguard) |
| `both_offstage` | Both offstage (aggressive edgeguard) |
| `respawning` | Invincibility platform visible |
| `non_gameplay` | Menus, loading, results screen |

**Training Results:**
- Dataset: **one** labeled video (`XUkRx6CpL04`, 3 games, 791 segments).
  Re-simulating `extract_labeled_frames.py` against the label file yields
  **~1,583 frames**, of which only 63 are `respawning`.
- Model size: 107MB total — `scene_classifier.onnx` (665KB graph) plus
  `scene_classifier.onnx.data` (106MB external weights). Both files are
  required and must sit in the same directory.
- Inference time: ~6.5ms per frame on M2 Pro

> **Corrected 2026-10-23.** This block previously read "~3,500 labeled frames
> from 5 videos" and "Validation accuracy: ~92%". Both figures are wrong or
> unverified:
>
> - **Video count:** the model was trained on one video, not five. The five
>   videos in `.video_cache/` came from benchmark runs, not labeling. A search
>   of the training machine found only the one label file.
> - **Frame count:** one video yields ~1,583 frames under the extractor's
>   current parameters (`DEFAULT_INTERVAL=0.5`, `RARE_INTERVAL=0.25`,
>   `EDGE_PADDING=0.15`), not ~3,500.
> - **Accuracy:** ~92% is unverified and, more importantly, not a
>   generalization metric. `create_train_val_test_splits()` does a
>   *frame-level* `random.shuffle` within a single video, so a frame at
>   t=10.0s can land in train while t=10.5s lands in test. Those are
>   near-duplicate images. Any score from that split measures memorization of
>   near-duplicates, not the ability to classify unseen footage.
>
> The split is reproducible (`seed=42`), so the ~92% figure can be
> *reproduced* — but reproducing a leaky metric does not validate it. A
> meaningful number requires evaluating on a **held-out video** the model has
> never seen. See the January 11 entry: that is the same labeling work already
> on the roadmap.

**Integration (`runitback/cv/scene_classifier.py`):**
- ONNX Runtime inference
- Temporal smoothing (5-frame median filter)
- Graceful fallback when model unavailable

**Integration (`runitback/classification/detector.py`):**
- Position-based neutral reset: `both_grounded` → immediate neutral
- Disadvantage type mapping: scene state → DisadvantageType enum
- Batch classification for efficiency

**Files Created:**
- `tools/scene_labeler.py` - Labeling application
- `tools/extract_labeled_frames.py` - Frame extraction
- `tools/train_scene_classifier.py` - Training script
- `runitback/cv/scene_classifier.py` - Inference module
- `data/models/scene_classifier.onnx` - Trained model

---

### January 2, 2026 - Character Detection Fix & Skills

**Session Goal:** Fix Mario/Luigi character detection confusion and add Claude Code skills

**Problem:**
- `UgGQkF1wkoA` Game 2: Mario (in green skin) detected as Luigi
- Gemini Vision API detected Luigi with 80% of votes (12/15)
- Existing hint system only applied corrections when votes < 75%
- Luigi won the vote because Mario's green skin makes him visually indistinguishable from Luigi

**Solution: Exclusive Hint Override**

Added a two-pass hint system in `llm_detector.py`:

1. **First pass (existing):** Apply standard hints when vote confidence < 75%
2. **Second pass (new):** When one player's detection matches a title hint, the remaining hint becomes exclusive to the other player

Example for "doug todd (Mario) vs Spencer (Ness)":
- P2 detection = Ness, which matches hint → Ness is confirmed for P2
- Remaining hint "Mario" becomes exclusive to P1
- P1 detected as Luigi (80%), but exclusive hint override applies: Luigi → Mario

**Key Code Changes (`runitback/cv/llm_detector.py`):**
- Added `apply_exclusive_hint()` function
- When P2 matches a hint, remaining hints exclusively apply to P1 (and vice versa)
- Works for any confusion pair (Mario/Luigi, Pit/Dark Pit, etc.) where color skins confuse detection

**Skills Added:**
- `/test` - Run pytest test suite
- `/analyze <url>` - Analyze a YouTube SSBU replay
- `/debug-video <url>` - Debug detection with verbose logging

**Test Results After Fix:**
- Character detection: 6/6 passing (was 5/6)
- All other tests unchanged

---

### January 2, 2026 - Timestamp Accuracy Fixes & Cache Management

**Session Goal:** Fix timestamp accuracy test failures and manage cache directories

**Problems Fixed:**
1. GO!/GAME! pairing algorithm was backwards - found "closest GO! before each GAME!" instead of pairing each GO! with its next GAME!
2. GAME! validation happened AFTER GO! filtering, causing incorrect pairing
3. No minimum gap after GAME! to filter replay GO! screens
4. White flash threshold too strict for some streams with overlays

**Fixes Applied to `cli.py`:**

1. **Moved GAME! validation before GO! filtering** - ensures only valid GAME! markers are used for pairing
2. **New GO!→GAME! pairing algorithm** - pairs each GO! with the next GAME!, taking only the first GO! after each game ends
3. **Added MIN_GAP_AFTER_GAME = 40 frames** (20 seconds) - filters replay GO! screens that appear too soon after GAME!
4. **Relaxed white flash threshold** - brightness > 180 (was 200), white_ratio > 0.5 (was 0.8)

**Cache Management:**
- Added `.cache/` and `.video_cache/` to `.gitignore`
- Removed previously committed cache files from git tracking (1279 files)
- Cache remains locally for faster re-analysis

**Test Results After Fixes:**
- Game count tests: 6/6 passing
- Timestamp accuracy: 6/6 passing
- Character detection: 5/6 passing (Mario/Luigi confusion pre-existing)
- Stock loss detection: 7/7 passing
- Stage detection: 0/6 passing (not implemented yet)

---

### December 31, 2024 - Gemini Stock Counter Detection

**Session Goal:** Add visual stock counter detection using Gemini Vision API for improved accuracy

**Problem:**
- Damage-reset detection misses low-percent kills (kills below 50% damage)
- Star KOs have no brightness flash, so brightness-based detection fails
- OCR struggles with SSBU's stylized fonts (e.g., "1" misread as "E", "F", "T")

**Solution:** Stock counter splash screen detection using Gemini Vision API

**Methods Attempted:**

| Attempt | Method | Result |
|---------|--------|--------|
| 1 | Brightness flash detection | ❌ Too many false positives from game effects, loading screens, transitions |
| 2 | OCR-based stock counter reading | ❌ PaddleOCR couldn't reliably read stylized SSBU font ("1" → "E", "F", "T") |
| 3 | **Gemini Vision API** | ✅ Perfect detection of stock counter splash and GAME! screen |

**Files Added/Modified:**
- `runitback/cv/llm_detector.py` - Added `detect_stock_counter_with_gemini()` and `GeminiStockCounterDetection` dataclass
- `runitback/cv/stock_counter_detector.py` - New module wrapping Gemini detection with `StockCounterTracker` class
- `runitback/classification/detector.py` - Integrated Gemini detection alongside damage-reset method
- `tests/test_game_detection.py` - Added `ExpectedStockLoss` dataclass and `TestStockLossDetection` class

**Implementation Details:**
- Stock counter splash appears after every kill showing "X - Y" format (P1 stocks - P2 stocks)
- Works for ALL kill types including star KO (no brightness flash) and normal KO
- "GAME!" screen appears instead of stock counter for final stock
- Gemini prompt asks for JSON response with `detected`, `p1_stocks`, `p2_stocks`, `is_game_end`, `confidence`
- Results are cached to minimize duplicate API calls

**Cost Analysis:**
- ~$0.0001-0.0002 per API call (Gemini Flash with image input)
- ~$0.0003-0.0006 per game (3-5 stock losses)
- Latency: ~0.5-1.5s per API call

**Integration Strategy:**
- Damage-reset detection remains PRIMARY (free, instant)
- Gemini detection is OPTIONAL confirmation (enable with `use_gemini_stock_detection=True`)
- Both methods run together when enabled - Gemini catches low-percent kills that damage-reset misses

**Test Results:**
- Game 1 (3 stock losses): All detected correctly including 2:40 star KO
- Game 2 (5 stock losses): All detected correctly
- No false positives in non-kill frames

---

### December 30, 2024 - Stock Loss Tracking Implementation

**Session Goal:** Add stock loss tracking to game state analysis

**Features Added:**
- `StockLoss` dataclass to track stock loss events with timestamp and player
- `stock_losses` field on `StateSegment` to attach losses to segments
- `p1_stock_loss` / `p2_stock_loss` flags on `FrameState`
- Display of stock losses in CLI output with colored stars (★ You lost a stock / ★ Opponent lost a stock)

**Stock Loss Detection - Methods Attempted:**

| Attempt | Method | Result |
|---------|--------|--------|
| 1 | Simple damage reset (`prev >= 50, curr <= 5`) | ❌ Too noisy - OCR errors caused 15+ false positives per game |
| 2 | Stock splash screen OCR ("2-3" pattern) | ❌ Unreliable - splash has transparency, slow, final stock shows GAME! |
| 3 | HUD stock icon counting via brightness | ❌ Values flickered (3→1→3→2→3) due to stage backgrounds |
| 4 | **Consecutive frame confirmation** | ✅ Require 2 frames with damage ≤5% after ≥30% damage |

**Final Parameters:**
- `STOCK_LOSS_COOLDOWN = 10.0s` - Prevents duplicate detections for same player
- `CONFIRM_FRAMES = 2` - Consecutive low-damage frames required before confirming

**Bugs Fixed:**
- Overlapping segment timestamps (end time now uses previous frame's timestamp)
- Double-offset timestamp display in CLI output

**Parameters Tuned:**
- `neutral_reset_time`: 1.5s → 3.0s (reduces state flickering)
- Added minimum segment duration filtering (1.0s neutral, 1.5s adv/disadv)

**Files Modified:**
- `runitback/classification/states.py` - Added StockLoss, updated FrameState/StateSegment
- `runitback/classification/detector.py` - Stock loss detection logic
- `runitback/cli.py` - Display stock losses on segments

**Test Results:**
- Before: 15+ false stock losses detected
- After: 4 stock losses detected (realistic for a game)

---

### Earlier Development (Pre-December 30, 2024)

The following systems were developed in earlier sessions. See detailed sections below.

---

## Project Overview

RunItBack is a Python CLI tool that analyzes Super Smash Bros. Ultimate (SSBU) replays from YouTube videos. It identifies game states (neutral, advantage, disadvantage), tracks stock losses, and provides timestamped analysis.

## Technical Stack

- **Language:** Python 3.11+
- **Interface:** CLI (Typer + Rich for formatting)
- **Video:** yt-dlp for download, OpenCV for frame processing
- **OCR:** PaddleOCR for damage percentage reading
- **Detection:** Hybrid (CV extraction + rule-based classification)

---

## Core Detection Systems

### 1. Game State Detection

#### Current Implementation
File: `runitback/classification/detector.py`

**Method:** Damage percentage tracking via OCR
- Sample frames at 0.5s intervals
- Read damage percentages using PaddleOCR
- Track damage changes between frames
- Classify states based on who got hit

**State Classification Rules:**
- **NEUTRAL:** No significant damage changes, both players visible
- **ADVANTAGE:** Opponent's damage increased (you hit them)
- **DISADVANTAGE:** Your damage increased (they hit you)

**Parameters:**
- `hit_threshold = 4.0` - Minimum damage increase to count as a hit
- `neutral_reset_time = 3.0` - Seconds without damage to return to neutral

#### What Worked
- Damage tracking is reliable for detecting who got hit
- Using a neutral reset timer prevents rapid state flickering
- The 4% hit threshold filters out self-damage and healing

#### What Didn't Work
- **Frame-by-frame state detection was too noisy** - OCR occasionally misreads, causing false state transitions
- **Detecting grounded vs airborne characters** - Considered using image processing to detect when characters are standing on platforms as a neutral reset criteria, but:
  - Character sprites vary wildly (89 characters, many costumes)
  - Stage backgrounds interfere with detection
  - Visual effects obscure characters
  - Decided on simpler approach with timer-based reset

#### Alternative Considered but Not Implemented
- **Character position tracking** using motion detection and Kalman filter
  - Would enable more accurate state detection (e.g., detect edgeguarding)
  - Too complex for initial implementation
  - Deferred to future phases

### 2. Stock Loss Detection

#### Current Implementation (Hybrid)
File: `runitback/classification/detector.py`

**Method:** Dual detection with damage-reset (primary) + Gemini Vision (optional)

**Primary - Damage Reset Detection (free, instant):**
- When damage drops to ≤2%, start counting consecutive low frames
- After 3 consecutive low frames (CONFIRM_FRAMES), check if previous damage was ≥50%
- If so, confirm stock loss
- Apply 10-second cooldown per player to prevent duplicates

**Optional - Gemini Stock Counter Detection (more accurate):**
- Enable with `use_gemini_stock_detection=True` on `GameStateDetector`
- Detects stock counter splash screen ("X - Y" format) using Gemini Vision API
- Catches low-percent kills that damage-reset would miss
- Also detects "GAME!" screen for final stock

**Cost/Time Analysis (Gemini enabled):**
- Per API call: ~$0.0001-0.0002, ~0.5-1.5s latency
- Per game (3-5 kills): ~$0.0003-0.0006, ~1.5-7.5s total
- Per set (3-5 games): ~$0.001-0.003, ~5-40s total
- Results are cached to avoid duplicate calls

**Data Structures:**
- `StockLoss` dataclass in `states.py` - records timestamp and player
- `p1_stock_loss` / `p2_stock_loss` flags on `FrameState`
- `stock_losses` list on `StateSegment`

#### Evolution of Stock Loss Detection

**Attempt 1: Simple Damage Reset**
```python
# Detect when damage drops significantly to near-zero
if prev_damage >= 50 and current_damage <= 5:
    stock_loss = True
```
**Problem:** OCR noise caused many false positives. OCR would occasionally misread damage as 0-5%, triggering false stock losses.

**Attempt 2: Stock Splash Screen Detection**
```python
# Detect the "2-3" stock count display that appears when stock is lost
def detect_stock_splash(frame):
    # Look for bright white numbers in center
    # Use OCR to read "X-Y" pattern
```
**Problem:**
- The splash has transparency over gameplay, making detection unreliable
- Calling OCR for every frame is too slow
- Final stock loss shows "GAME!" instead of stock count

**Attempt 3: HUD Stock Icon Counting**
```python
# Track stock icons displayed below damage percentage
p1_stocks = count_stocks(frame, player=1)
p2_stocks = count_stocks(frame, player=2)
```
**Problem:** Stock icon detection via brightness thresholds was unreliable:
- Stage backgrounds affect brightness
- Game effects interfere
- Values flickered (e.g., 3→1→3→2→3)

**Attempt 4 (Final): Consecutive Frame Confirmation**
- Require 2 consecutive frames with damage ≤5%
- Remember the high damage value before the low streak started
- Only trigger if that high value was ≥30%
- Use 10-second cooldown per player

**Why This Works:**
- Filters single-frame OCR errors
- Legitimate stock losses show sustained low damage
- Cooldown prevents duplicate detections from lingering low values

### 3. Segment Building and Filtering

File: `runitback/classification/detector.py`, `_build_segments()`

**Method:**
1. Group consecutive frames with same state into segments
2. Filter out short segments (noise):
   - Neutral segments < 1.0s removed
   - Advantage/disadvantage segments < 1.5s removed
3. Merge consecutive segments of same state
4. Propagate stock losses through filtering/merging

#### What Worked
- Minimum duration filtering greatly reduces noise
- Merging consecutive same-state segments produces clean output

#### Edge Case Handled
- When short segments are filtered, their stock losses are carried forward to the previous segment

### 4. Damage Percentage OCR

File: `runitback/cv/ocr.py`

**Method:**
- Use PaddleOCR with English model
- Disable document orientation detection (game UI doesn't need rotation)
- Apply character substitution for common misreads (O→0, S→5, etc.)
- Validate damage is in range 0-999%

**Current Configuration:**
```python
PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
```

**ROI Coordinates (normalized):**
- P1 Damage: (0.27, 0.87, 0.44, 0.98)
- P2 Damage: (0.64, 0.87, 0.76, 0.98)

#### Accuracy Improvement Options

**Template Matching (Option 2):**
- Create templates for each digit 0-9 and % symbol
- Match against damage region
- More robust to color variations
- Estimated improvement: 10-20% accuracy gain
- Implementation effort: Medium (need to capture templates, handle scaling)

**ML-based Digit Recognition (Option 3):**
- Train CNN on SSBU damage number screenshots
- Would handle all visual variations
- Estimated improvement: 30-50% accuracy gain
- Implementation effort: High (need training data, model architecture, training pipeline)
- Feasibility: Highly feasible with modern tools
  - Could use transfer learning from MNIST/SVHN
  - Would need ~1000+ labeled screenshots
  - Training time: hours on GPU

### 5. Game Boundary Detection

File: `runitback/cli.py`, `detect_games()`

**Method:** Hybrid GO!/GAME! marker detection + HUD-based detection

**Primary (GO!/GAME! markers):**
1. Detect "GO!" screens via orange/yellow color analysis in center
2. Detect "GAME!" screens via green color + moderate brightness
3. Validate GAME! markers by checking for white flash after
4. Match GO! markers with subsequent GAME! markers

**Fallback (HUD-based):**
- Detect gameplay frames by HUD presence:
  - Character icons (colorful portraits)
  - Stock icons (bright indicators)
  - Damage displays (white numbers)
- Merge consecutive gameplay frames into games
- Filter by minimum duration (30s)

#### What Worked
- Color-based GO!/GAME! detection is reliable
- White flash validation eliminates false GAME! positives
- Combining markers with HUD detection handles edge cases

#### What Didn't Work
- **Pure HUD detection** produced too many false positives (menus with similar layouts)
- **Template matching for GO!/GAME!** was brittle across different video qualities

### 6. Character Detection

File: `runitback/cv/character_detector.py`

**Method:**
1. Try Gemini API (LLM-based detection) if API key available
2. Fall back to template matching against character portrait database
3. Use video title hints to guide detection

**Cross-game consistency:**
- Track character detections across all games in a set
- If one character dominates (≥75% of games), apply to outliers
- Only correct commonly confused pairs (Mario/Luigi, Ken/Terry, etc.)

---

## Project Structure

```
RunItBack/
├── runitback/
│   ├── cli.py                    # Typer CLI entry point, game boundary detection
│   ├── config.py                 # Pydantic settings
│   ├── video/
│   │   ├── downloader.py         # yt-dlp wrapper with caching
│   │   └── frame_extractor.py    # OpenCV frame sampling
│   ├── cv/
│   │   ├── ocr.py                # PaddleOCR damage reading
│   │   ├── ui_detector.py        # HUD detection, stock counting
│   │   ├── stage_detector.py     # Stage identification
│   │   ├── character_detector.py # Character identification
│   │   ├── llm_detector.py       # Gemini API integration (chars, stock counter)
│   │   └── stock_counter_detector.py  # Stock counter splash detection
│   ├── classification/
│   │   ├── states.py             # GameState, FrameState, StateSegment
│   │   └── detector.py           # Game state detection logic
│   └── utils/
│       └── logging.py            # Logging configuration
├── data/
│   ├── stages.json               # Stage geometry definitions
│   └── game_markers/             # GO!/GAME! templates
├── .cache/                       # Gemini API cache (gitignored)
│   ├── llm_results/              # Character detection cache
│   ├── stage_results/            # Stage detection cache
│   └── stock_counter_results/    # Stock counter detection cache
├── .video_cache/                 # Downloaded videos (gitignored)
└── tests/
    └── test_game_detection.py    # Stock loss and game detection tests
```

---

## Key Data Structures

### FrameState
```python
@dataclass
class FrameState:
    timestamp: float
    frame_index: int
    p1_damage: Optional[float]
    p2_damage: Optional[float]
    p1_stocks: Optional[int]
    p2_stocks: Optional[int]
    p1_visible: bool
    p2_visible: bool
    state: GameState
    p1_stock_loss: bool
    p2_stock_loss: bool
```

### StateSegment
```python
@dataclass
class StateSegment:
    state: GameState
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    stock_losses: list[StockLoss]
```

### StockLoss
```python
@dataclass
class StockLoss:
    timestamp: float
    player: int  # 1 or 2
```

---

## Configuration

### Detection Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `hit_threshold` | 4.0 | Min damage increase to count as hit |
| `neutral_reset_time` | 3.0s | Time without damage to return to neutral |
| `STOCK_LOSS_COOLDOWN` | 10.0s | Min time between stock losses for same player |
| `CONFIRM_FRAMES` | 3 | Consecutive low-damage frames to confirm stock loss |
| `MIN_KILL_DAMAGE` | 50 | Min damage before stock loss is plausible |
| `MIN_NEUTRAL_DURATION` | 1.0s | Minimum segment length for neutral |
| `MIN_ADV_DISADV_DURATION` | 1.5s | Minimum segment length for advantage/disadvantage |

### Gemini Stock Counter Detection
| Parameter | Value | Description |
|-----------|-------|-------------|
| `use_gemini_stock_detection` | False | Enable Gemini Vision for stock loss confirmation |
| `gemini_api_key` | env var | API key (uses `GEMINI_API_KEY` env var if not set) |
| `detection_cooldown` | 3.0s | Min time between Gemini API calls |

### Sample Output
```
STATE SUMMARY:
  Neutral:      0:42 (25%)
  Advantage:    0:49 (28%)
  Disadvantage: 1:20 (47%)

NEUTRAL SEGMENTS:
  0:18 - 0:25
  0:40 - 0:44  ★ You lost a stock
  0:54 - 0:58
  ...

ADVANTAGE SEGMENTS:
  0:32 - 0:39
  1:20 - 1:25
  ...

DISADVANTAGE SEGMENTS:
  0:26 - 0:31
  0:46 - 0:54
  1:26 - 1:27  ★ Opponent lost a stock
  ...
```

---

## Future Improvements

1. **Template Matching for OCR** - Create digit templates for more robust damage reading
2. **ML-based Digit Recognition** - Train CNN for SSBU damage numbers
3. **Character Position Tracking** - Motion detection + Kalman filter for edgeguard detection
4. **Slippi Integration** - For Melee replays, read game memory directly
5. **Stage-specific Ledge Detection** - Use stage geometry data for ledgetrap identification

---

## Testing

The test video used during development:
- URL: https://www.youtube.com/watch?v=7oPdiBsgSG0
- Title: kit (Ken) vs Sophist (King Dedede) - West Towne Brawl 150 Winners R3

Ground truth comparison (first 50 seconds):
- Neutral: 0:18-0:26, 0:38-0:44, 0:55-0:59
- Disadvantage: 0:26-0:32, 0:44-0:55, 0:59-1:08
- Advantage: 0:34-0:38

Detection results were within acceptable accuracy after parameter tuning.
