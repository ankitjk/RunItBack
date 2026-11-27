# RunItBack Architecture

How the pipeline works **now**. History - what was tried, measured and changed,
and why - is in the [DEVELOPMENT.md](DEVELOPMENT.md) changelog; open bugs are in
[ISSUES.md](ISSUES.md). When code changes make a statement here untrue, fix it
in the same commit.

File layout and example output: see the [README](../README.md).

## Pipeline

```
YouTube URL -> download (yt-dlp) -> sample a frame every 0.5s (OpenCV)
  -> game boundaries -> per game: characters, damage OCR, scene class
  -> game state (neutral / advantage / disadvantage) + stock losses -> report
```

Stack: Python 3.11+, Typer + Rich CLI, yt-dlp, OpenCV, PaddleOCR, onnxruntime,
optional Gemini Vision (`google-generativeai`, model `gemini-2.5-flash`).

## 1. Game boundaries

`runitback/cli.py`, `detect_games()`.

- **Primary:** GO! and GAME! screens. GO! is found by orange/yellow colour in
  the screen centre, GAME! by green colour with moderate brightness
  (`ui_detector.py`). A GAME! marker counts only if a white flash follows it.
  Each GO! is paired with the next GAME!.
- **Fallback:** runs of frames with a gameplay HUD (portraits, stock icons,
  damage numbers), when markers are missing.
- Games shorter than 30s are dropped (`min_game_duration`).
- OCR is skipped for frames outside game bounds.

Tried and rejected: HUD presence alone (menus with similar layouts gave false
games); template matching alone for GO!/GAME! (brittle across video qualities).

## 2. Characters

`runitback/cv/character_detector.py`, `runitback/cv/llm_detector.py`.

- **With `GEMINI_API_KEY`:** Gemini Vision on frames at 10/25/50/75/90% of each
  game, majority vote. Results are cached by frame hash.
- **Without a key (current setup):** template matching of HUD portraits against
  `data/character_references/`. Often wrong, and it ignores the title hints
  (ISSUE-013).
- Character names in the video title are parsed as hints.
- `_apply_character_consistency()` (`cli.py`): if one character dominates a set
  (>= 75% of games), outliers from commonly confused pairs (Mario/Luigi,
  Ken/Terry, ...) are corrected.

## 3. Damage OCR

`runitback/cv/ocr.py`; regions in `runitback/cv/ui_detector.py`.

- PaddleOCR (English; orientation and unwarping off) on each player's damage
  region, normalised `(x1, y1, x2, y2)`: P1 `(0.27, 0.87, 0.44, 0.98)`,
  P2 `(0.64, 0.87, 0.76, 0.98)`.
- The parser substitutes common misreads (O->0, S->5, ...) and accepts 0-999.
- Reads below 0.3 confidence are discarded (`read_damage_percentage` default).
- `has_damage_digits()` says whether the raw text has any digit - used to tell
  a blank damage display from a misread.
- Known failures (tags read as damage, merged decimals, clipped leading digit):
  ISSUE-001, -002, -014. Parallel OCR exists but is off (ISSUE-011).

## 4. Scene classification

`runitback/cv/scene_classifier.py`. ConvNeXt-Tiny, fine-tuned from ImageNet,
run with onnxruntime one frame at a time; a 5-frame median filter smooths the
live path.

- **Deployed model** (`data/models/scene_classifier.onnx`, fetched by
  `scripts/fetch_model.sh`): 7 classes, hardcoded in `CLASS_ORDER` (ISSUE-017).
  Trained on a random frame split of one video, so it has no honest accuracy
  figure.
- **Class definitions:** [SCENE_CLASSES.md](SCENE_CLASSES.md): 14 classes. The
  label tool and training support the first 11; `pre_game`, `game_end` and
  `finishing_blow` (defined 2026-11-26) are not yet supported or labelled.
- **Measured on unseen games** (game held out, November 16-24, 2026, before the
  GAME! screens were relabelled as `respawning`): 39-51% accuracy across runs; pooled 48.3% over three folds.
  The same code on a random frame split scores 76.4% - the difference is
  near-duplicate frames leaking across splits. `respawning` recall is 97-100%,
  but its precision is 19-54%: rare classes are over-predicted.
- **Labelled data:** one video (`XUkRx6CpL04`, 3 games, ~1,600 frames). This is
  the main limit (ISSUE-006).

Training workflow (`tools/`): `scene_labeler.py` (PyQt5; writes time-segment
labels to `data/scene_classifier/raw_labels/<video_id>.json`; its edits are in
`label_segments.py`: labelling already-labelled time replaces it, so segments
never overlap, times are stored per frame, and fields the tool does not use are
kept on save; a segment a person decides is marked `source: "human"`;
coloured strips show the game's segments and a 10s close-up),
then
`build_scene_model.py`, which runs the next two steps into a run folder
`data/models/runs/<name>/` and records labels, split and results in `run.json`:
- `extract_labeled_frames.py`: frames every 0.5s, 0.25s for rare classes. A
  video's earlier frames are replaced, so a label edit leaves nothing stale.
  Splits **by game**: one held-out game, or every game of one held-out video
  (`--test-video`), is the test set; the last 15% of each other game is
  validation, with a 5s gap.
- `train_scene_classifier.py`: drops classes with under 10 training frames,
  scores the test set once, writes `test_metrics.json` and `class_mapping.json`,
  and exports ONNX that accepts any batch size, checked against PyTorch before
  the run ends.
- It never writes the deployed model's folder; deploying needs ISSUE-017 fixed.

## 5. Game state

`runitback/classification/detector.py`; data classes in `states.py`
(`FrameState`, `StateSegment`, `StockLoss`, `GameStateAnalysis`).

- A damage increase of at least `hit_threshold` (4.0) is a hit. The opponent
  hit -> **advantage**; you hit -> **disadvantage**.
- Back to **neutral** when the scene is `both_grounded`, or after
  `neutral_reset_time` without damage (3.0s as the CLI calls it; the class
  default is 1.5s).
- Segments are built from runs of equal state. Neutral segments under 1.0s and
  advantage/disadvantage segments under 1.5s are dropped, neighbours merged,
  and their stock losses carried to the previous segment.
- Disadvantage segments are tagged (juggle, offstage/edgeguard, ledgetrap, tech
  chase, respawning) from the scene class.

Tried and rejected: per-frame state without duration filtering (OCR noise made
it flicker); detecting grounded/airborne with classical image processing (89
characters, costumes, effects) - the scene classifier does this instead.

## 6. Stock losses

`runitback/classification/detector.py`.

| Signal | Rule |
|---|---|
| **Damage reset** (main) | Damage <= 2% for `CONFIRM_FRAMES` (3) frames in a row, after a recent maximum >= `MIN_KILL_DAMAGE` (50) |
| **Respawn** | When a run of `respawning` frames ends, a player whose damage is now under 20% and at least 20 below their recent maximum lost a stock (both can, on a trade) |
| **Final stock** | Find the GAME! screen (searching back up to 40s from the game's end). The loser is the player whose damage display is blank in the 2s after it; fallback: clearly higher damage before it; otherwise nothing is recorded |
| **Gemini post-pass** (off without a key) | Low-percent kills and unattributed respawns, from the stock-count splash |

- `STOCK_LOSS_COOLDOWN`: 10s per player.
- A guard treats both players resetting together as a game transition (ISSUE-003).
- Cap of 3 losses per player per game. It always keeps the game's final stock:
  the earliest loss within `FINAL_STOCK_WINDOW_SECS` (6s) of the game's end.

Tried and rejected, with the measurement where there is one: a single
low-damage frame (OCR noise); OCR on the stock-count splash (transparent, slow,
absent on the last stock); counting HUD stock icons by brightness (1 of 14
correct, October 2026); reading the blank damage display at GAME! scored 11
right, 0 wrong, 3 undecided and was adopted.

**Benchmark** (`tools/test_stock_losses.py`; 6 sets, 20 games, 89 hand-labelled
losses; a detection must be within 5s and credit the right player): recall
83.1% (74/89), 27 false positives, precision 73.3% - October 24, 2026, Gemini
off. Per-set results are in the README. Runs with and without Gemini are not
comparable.

## 7. Stages (detector switched off)

Every game reports `Stage.UNKNOWN` (ISSUE-009); `stage_detector.py` (histogram
matching against `data/stage_references/` plus platform counting) is not
called. Notes for when it is re-enabled:

| Stage | Platforms | Notes |
|---|---|---|
| Small Battlefield | 2 (sides) | Near-identical colours to Battlefield |
| Battlefield | 3 (2 sides + top) | Tell apart from Small Battlefield by platform count |
| Final Destination | 0 | |
| Pokemon Stadium 2 | 2 (sides) | Background varies slightly |
| Smashville | 1 (centre) | |
| Town and City | 1-3 | Platforms appear, move and leave; background changes |
| Kalos Pokemon League | 2 (sides) | |
| Hollow Bastion | 1 (centre) | Cycles between warm (sunset) and cool (blue) colours |

Histograms cannot separate Battlefield from Small Battlefield. Stream overlays
shift colours, so reference images taken from real broadcasts match better than
clean screenshots.

## Configuration

- **Settings** (`runitback/config.py`): environment prefix `RUNITBACK_`, `.env`
  supported. Only `cache_dir` (default `~/.cache/runitback/`) is read by the
  code. The other fields (`video_quality`, `sample_interval_sec`,
  `ocr_confidence_threshold`, `stage_match_threshold`, `verbose`) are unused:
  the sample rate comes from `analyze --sample-rate`, the OCR threshold from
  the function default.
- **`GEMINI_API_KEY`** is read with `os.environ` at each call site. It is not a
  `Settings` field, so a `.env` entry is ignored. Unset = Gemini paths off.
  `RUNITBACK_USE_GEMINI=0` only affects character detection
  (`character_detector.py:22`).
- Keeping the key out of the repo (macOS Keychain):
  ```bash
  security add-generic-password -a "$USER" -s GEMINI_API_KEY -w   # prompts, no echo
  # then in ~/.zprofile:
  export GEMINI_API_KEY="$(security find-generic-password -a "$USER" -s GEMINI_API_KEY -w 2>/dev/null)"
  ```

## Caches (all outside git)

| Path | Holds |
|---|---|
| `~/.cache/runitback/` | Videos for `analyze` and the benchmark, named by URL hash |
| `.video_cache/` | Videos for the labelling tools, named by video ID (not shared with the above) |
| `.cache/` | Gemini results by frame hash (`llm_results/`, `stage_results/`, `stock_counter_results/`); only exists once a key has been used |
| `data/models/`, `data/scene_classifier/frames/`, `data/scene_classifier/splits/` | Model and generated training data |
| `debug_frames/` | Output of the debug tools |
