# RunItBack Development Documentation

The changelog of RunItBack: what changed in each session, why, and what was measured. Entries are history and are not rewritten; later corrections are added as dated notes. How the pipeline works *now* is in [ARCHITECTURE.md](ARCHITECTURE.md), and open bugs are in [ISSUES.md](ISSUES.md).

**Last Updated:** November 27, 2026

---

## Changelog

### November 27, 2026 - Inference Reads the Model's Own Classes (ISSUE-017)

**Session Goal:** Step 3, part 2: let `analyze` use any trained model, including the coming 14-class one.

**Problem:** ISSUE-017. `SceneClassifier.CLASS_ORDER` hardcoded the deployed model's 7 classes. Checked on the step-1 baseline model (8 classes): its index 6 is `one_on_ledge`, which the fixed list called `respawning`, and index 7 would have raised `IndexError`.

**Changes:**
- `runitback/cv/scene_classifier.py`:
  - `load_class_order()` reads the class names from the `class_mapping.json` that training writes beside the model; a model without one is the legacy deployed model (`LEGACY_CLASS_ORDER`, its 7 classes). Fixes ISSUE-017.
  - A model whose output count does not match its class list is disabled with an error, not run with wrong names; class names RunItBack does not know are reported as `unknown` with a warning.
  - `SceneState` gains `PRE_GAME`, `GAME_END`, `FINISHING_BLOW`. `pre_game` and `game_end` are not gameplay; `finishing_blow` is. `CLASS_TO_STATE` is built from the enum.
- `runitback/classification/detector.py` - explicit (empty) disadvantage tags for the three new classes. The stock-loss rules key on `respawning` alone, so a GAME! screen predicted as `game_end` no longer extends a respawn run; no rule changed.
- `tests/test_scene_classifier.py` (new, 7 tests) - mapping read, legacy fallback, a malformed mapping rejected, predictions named from the model's own classes (a fake ONNX session answering index 6 and 7 of 8), a mismatched model disabled, every labelled class a `SceneState`, the gameplay test.
- `CLAUDE.md`, `docs/ARCHITECTURE.md`, `tools/build_scene_model.py`, `tools/train_scene_classifier.py` - deploying a run's model is now possible (copy its ONNX file and `class_mapping.json`); still a separate decision, not made.

**Results:**
- Real models, loaded and run on three frames of `XUkRx6CpL04` (10.0s, 85.0s, 200.5s): the deployed model (no mapping) predicts `non_gameplay`, `both_offstage`, `non_gameplay` as before; the 8-class baseline loads its own names (`non_gameplay`, `both_offstage`, `one_on_ledge`); the baseline with a 2-name mapping is disabled with an error.
- Fast suite: 142 passed, 6 xfailed (with the gold-guard commit before it). mypy: no issues.
- The deployed model is unchanged, so `analyze` output is unchanged; the stock-loss benchmark was not rerun.

---

### November 27, 2026 - Gold Videos Cannot Reach Training

**Session Goal:** Before the owner starts hand-labelling the two gold games (plan step 4), make sure they can never be trained on.

**Problem:** The label tool saved every video's labels to `data/scene_classifier/raw_labels/`, the folder training reads, and nothing marked a video as test-only. A gold game labelled there would have been extracted and trained on by the next build, quietly turning the test set into training data.

**Changes:**
- `tools/label_segments.py` - `TRAINING_LABELS_DIR` and `GOLD_LABELS_DIR` (`data/scene_classifier/gold_labels/`), `is_gold()`, and `default_labels_path()`, which refuses a video that already has labels in the other folder: a video is training or gold, never both.
- `tools/scene_labeler.py` - `--gold`: labels go to the gold folder, the file gets `"role": "gold"`, and the title starts `[GOLD: test only]`. A file that says gold opens as gold even without the flag.
- `tools/extract_labeled_frames.py` - a file marked gold is an error, nothing is extracted from it, and earlier frames are kept. This also catches a gold file copied into `raw_labels/`.
- Tests (3): the folder rule both ways and the role surviving a round trip; extraction refusing a gold file; the tool saving and reopening a gold video as gold.
- Owner's candidate gold videos, checked against YouTube's metadata: `g5T1i-l9rrY` (Minnesota Smash Ultimate, 1080p, 9:21) and `QgVHJs1CWU0` (Wisconsin Smash, 720p, 13:50). The second is from the same channel as the training video `XUkRx6CpL04`, so it probably shares its overlay and its score would flatter the model; raised with the owner. Neither is a stock-loss benchmark set.

**Results:** Fast suite: 135 passed, 6 xfailed. Nothing trained.

---

### November 27, 2026 - New Classes in the Label Tool and Training (Step 3, Part 1)

**Session Goal:** Let `pre_game`, `game_end` and `finishing_blow` be labelled and trained, so the owner can start hand-labelling the gold games (plan step 4) while inference (ISSUE-017) and the relabel follow.

**Problem:**
- The class list was copied three times in `tools/` (label tool, extraction, training), and nothing checked they matched.
- `extract_labeled_frames.py` skipped any class it did not know without a word. A `game_end` segment labelled with the new keys would have vanished from training while the totals looked normal.

**Changes:**
- `tools/label_segments.py` - `SCENE_CLASSES`, the one 14-class list, in key order. `extract_labeled_frames.py` and `train_scene_classifier.py` import it; the label tool's enum is tested against it.
- `tools/scene_labeler.py` - keys P (`pre_game`), X (`game_end`), F (`finishing_blow`), with colours, legend rows and help text from `docs/SCENE_CLASSES.md`. The help no longer calls entrances `non_gameplay` or the GAME! screen `respawning`.
- `tools/extract_labeled_frames.py` - a label file with an unknown class is an error (`build_scene_model.py` then stops) and its video's earlier frames are left untouched. `game_end` and `finishing_blow` are sampled every 0.25s like the other short classes; `pre_game` every 0.5s.
- `docs/SCENE_CLASSES.md`, `docs/ARCHITECTURE.md` - support status.
- Tests (6): the tool's classes, keys and colours match the shared list; P, X and F label by key press; the new classes are extracted at the right rates; an unknown class is refused without deleting earlier frames.

**Results:**
- Fast suite: 131 passed, 6 xfailed.
- Not retrained: no video has the new classes yet. The deployed model and inference are unchanged.

**Left Open:**
- Step 3 continues: ISSUE-017 (inference reads `class_mapping.json`), the `SceneState` enum and gameplay test in `runitback/cv/scene_classifier.py`, the `respawning` rules and disadvantage tags in `runitback/classification/detector.py`, the relabel of `XUkRx6CpL04`, and the retrain.

---

### November 27, 2026 - Plan: a Hand-Labelled Gold Set for Testing

**Problem:**
- The plan had the labeller draft video 2, the owner review it, and the model trained on video 1 be tested on it. Review mode visits only `needs_review` segments; segments the model and the LLM agree on are accepted unseen. On game 3 that was 41% of game time, 22% of it wrong. Those labels would be the model's own predictions, so the test would mark the model right on its own mistakes (for example agreed `one_on_ledge`, wrong 62% of the time on game 3) and report it as better than it is. The same holds for scoring the labeller against a reviewed version of its own draft.
- A first fix (hand-label one game of video 2 blind, draft the rest for training) still fails once a model trains on video 2's other games: the blind game is then from a seen broadcast, like the game-3 figure, not an unseen video.

**Changes:**
- `docs/DEVELOPMENT.md`, labelling-loop plan (owner's decisions): testing and training use different videos. **Gold videos** have one game labelled by hand, blind, and are never trained on; **training videos** are drafted and reviewed. `XUkRx6CpL04` stays a training video: its game 3 is an unseen game but not an unseen video, and the 14-class model will train on it.
- The plan is reordered to get honest numbers before building more: after the new classes (step 3) come **two gold videos** (step 4: best-of-3 sets, one hand-labelled game each, a `role` field that keeps them out of training) and the **first unseen-video results** on them (step 5: the scene model's cross-video accuracy; the labeller's threshold chosen on one gold game and checked on the other, a paid run needing approval). Those numbers decide the threshold (the owner asked about 0.8: on today's model almost no segment reaches it, so it would change little, and the larger error source is the agree-and-accept rule), the agree-and-accept rule, and whether game bounds, the skeleton and the review screen (now steps 6-8) are worth building. Training videos (step 9) are best-of-5 sets, reviewed in full until the gold numbers justify spot checks.
- The runitback-labeler session was told its side (scoring against gold games, gold drafts never imported, spot-checking accepted segments). Settled with it, and written into steps 4, 7 and 8: the skeleton carries `role`, so the labeller can refuse to draft a gold game before its hand labels exist; a model is scored on a gold game only if the video is also absent from its `run.json` `split.train_games`; spot checks are a separate `spot_check` flag whose verdict the review screen records explicitly (`confirmed` / `corrected`, where `corrected` means the class was wrong), and extraction refuses a game with an unanswered one.
- Two gold videos from the start settle how the labeller's threshold is measured. With one, it could only be chosen and checked on the same game (as on game 3 now, which flatters it), or chosen on game 3 with a separate model that never trained on it.

**Results:** Plan only; nothing built or measured.

---

### November 27, 2026 - Label Editing in the Label Tool (ISSUE-024, ISSUE-025)

**Session Goal:** Step 2 of the labelling loop - edit existing labels in `scene_labeler.py`, so the GO!, GAME! and finishing-blow relabel (step 3) and draft review (step 6) can be done in the tool rather than by hand in JSON.

**Problem:**
- The tool could only add labels. Ctrl+Z removed labels added in the current session; anything older needed hand-editing of the JSON.
- ISSUE-024: S set a mark, but the next class key started the label at the playhead, not at the mark (`_assign_class` reset `segment_start` to the current time). The status bar said the opposite.
- ISSUE-025: labelling time that was already labelled appended a second label; nothing checked for overlaps. `XUkRx6CpL04` has one: game 1, `one_offstage` 80.7-84.6s and `both_offstage` 84.5-86.2s. The 0.15s edge padding in extraction means no training frame came from those 0.1s, so it is harmless and was left alone (a ground-truth edit needs the owner's approval).
- New segment times were rounded to 0.1s, up to 6 frames off at 60fps.
- Saving wrote only start, end and class, so a runitback-labeler draft opened and saved would have lost `source`, `needs_review` and the model's guesses.

**Changes:**
- `tools/label_segments.py` (new) - the label-file classes, moved out of `scene_labeler.py`, and the edits as functions with no Qt dependency: `overwrite`, `split`, `set_class`, `move_edge`, `delete`, plus `segment_at` and `overlaps`. Each returns a new segment list or raises `EditError` with the reason, so a refused edit changes nothing. No edit leaves an overlap. Fields the tool does not use are kept in `extra` on the video, each game and each segment, written back on save, and copied to both pieces when a segment is cut.
- `tools/scene_labeler.py`:
  - **S then a class key** labels from the mark to the playhead, replacing what was there - the main relabelling action (ISSUE-024). The class key uses up the mark even when the edit is refused. A first version left the mark set after a refusal, and the next class key, meant to start a continuous label 20s later, overwrote everything back to the old mark; the off-screen check below caught it.
  - **Continuous labelling** closes each label with `overwrite`, so labelling over labels replaces them (ISSUE-025).
  - New keys: C (change class), K (cut at the playhead), [ and ] (move start or end to the playhead; the neighbour sharing the edge follows), Delete/Backspace, Esc (cancel). They act on the table's selected row, or else the segment under the playhead; K always uses the segment under the playhead. Edits are refused while a continuous label is open.
  - Times are stored per frame to 0.1 ms (`frame_time`), and seeking rounds back to that frame. The player shows the frame number; the table shows 3 decimals and its cells are no longer editable text.
  - Ctrl+Z restores a snapshot from before each change, so it undoes edits, game boundaries and stage changes, in any game.
  - Loading reports overlapping segments in the status bar and the terminal, without changing them.
  - **Segment strips** (`SegmentStrip`, second commit), under the video slider: the whole game, and 10s around the playhead, where single frames are visible (at ~0.27s per pixel the whole-game strip cannot show a finishing-blow flash). Segments in their class colour; unlabelled time dark; `needs_review` hatched; overlaps red cross-hatched on black, at least 6px wide (a first version drew a 4px band along the top, which the owner did not notice during a real relabel: at 0.1s it was a 2px fleck); the selection outlined; the mark or open label shaded to the playhead. Click to jump and select; hover for class and times. The table of 216-352 rows per game showed none of this at a glance. Display only: the strips never change labels and never take keyboard focus.
  - Screenshots of the off-screen window showed three display problems, fixed: the red playhead vanished on `both_offstage` (now white with a black outline); the legend overlapped itself and left the table 2 rows high (now two columns in a scroll area, with compact distribution rows; the table shows ~9 rows at 900px); and class names in the table were white on the light table background, so the Class column looked empty (now black on the class colour). The last predates this work.
  - The Class Distribution panel shows labelled seconds per class instead of segment counts: training frames come from time.
  - **`source` on edited segments** (`label_segments.py`), agreed with the runitback-labeler session: `source` says who decided a segment's class. A segment a person creates, changes the class of (C), or moves an edge of ([ or ], on both sides of the edge) gets `source: "human"`. Pieces left by a cut or trimmed by an overwrite keep their source and every draft field, because nobody judged them; the labeller asked for this over my first proposal, which marked them human and would have made unreviewed proposals look confirmed. No edit changes `needs_review`, `model_label` or `llm_label`. Joining now treats a segment whose only extra field is `source: "human"` as plain. Before this, C on a draft segment left `source: "model"` on a class a person chose. The draft fields named in the plan were corrected: the drafts write `mean_confidence` and `min_confidence`, not `confidence`, plus a per-game `produced_by` block; all are kept on save (checked: a draft with every field loads and saves identically).
  - Closing asked "save before closing?" every time, even right after Ctrl+S (the owner hit this after the first real relabel): the tool never tracked unsaved changes. It now compares the labels with the last load or save, asks only if they differ, and shows `*` in the window title while they do (undoing back to the saved labels counts as saved). Also from the old code: choosing Save when no file got chosen closed the window and lost the labels; it now stays open. Closing with a continuous label still open warns that it is not saved until E closes it.
- `tools/label_segments.py`, found before merge while planning a real relabel: `overwrite` refused to fix the game-1 overlap with either class ("Already one_offstage"), because its nothing-to-do check saw one segment of that class covering the span and ignored the other segment overlapping it. It now refuses only when no other segment overlaps the span.
- `tools/label_segments.py`, also from that relabel: overwriting the overlap as `both_offstage` left `both_offstage` 84.5-84.6 and 84.6-86.2 as two touching segments, because an overwrite kept the leftover pieces of its neighbours separate. Overwrite and change-class (C) now join the edited segment with an immediately touching segment of the same class. Segments carrying extra fields (a draft's `needs_review` or model guess) are never joined, and K still cuts on purpose. The 20 touching same-class pairs already in `XUkRx6CpL04` (all `one_airborne`, probably cut where the airborne player switched) are left as they are.
- `tests/test_label_segments.py` (new, 32 tests) - each edit, refusals, joining (and not joining drafts or non-neighbours), the `source` rule for each kind of edit, no overlaps after a sequence of overwrites, fields kept on the pieces, frame times mapping back to their frame at 60/59.94/30/29.97fps, and every real label file loading and saving unchanged.
- `tests/test_scene_labeler_keys.py` (new, 11 tests) - drives the window with real key presses on a generated 4-second video: the mark (ISSUE-024), a refused mark edit not leaving the mark behind, continuous labelling replacing labels (ISSUE-025), and cut, change, move, delete, then Ctrl+Z back to the file exactly. Two more check the strips: the colour drawn at a time matches its class and changes after an edit, a click jumps there and selects the segment for Delete, an overlap is drawn on both strips, and closing asks only when unsaved changes or an open label would be lost. Skipped where PyQt5 is not installed.
- `docs/ARCHITECTURE.md`, `CLAUDE.md` - the label tool's editing rules and the new module.

**Results:**
- Fast suite: 127 passed, 6 xfailed (84 passed before; 43 new). The 6 expected failures are the known-bug tests for ISSUE-001, -002, -003 and -014, unchanged.
- Off-screen run of the real tool on a copy of `XUkRx6CpL04.json` with the cached video (60fps): mark relabel, continuous relabel, cut, change, edge move, Delete with the table focused, and refusals all behaved as above, with no new overlaps. Ctrl+Z back to the start reproduced the file exactly, and the ground-truth file's checksum was unchanged.
- Not yet used by a person on a real relabel; that is step 3.

**Left Open:**
- Changing a draft segment's class does not clear its `needs_review`; marking a segment reviewed belongs to the review screen (step 6).
- `extract_labeled_frames.py` turns a time into a frame with `int(t * fps)`, which can land one frame early on a time stored to 0.1 ms. It only matters for segments under 0.3s, where one middle frame is taken; not changed here.

**Minor:**
- `data/scene_classifier/raw_labels/XUkRx6CpL04.json` - the owner relabelled the game-1 overlap in the label tool, choosing `both_offstage` for 84.5-84.6s. Net change: `one_offstage` 80.7-84.6s now ends at 84.5s; the two segments that the first overwrite left (84.5-84.6 and 84.6-86.2) were joined back into `both_offstage` 84.5-86.2s with the owner's approval. No overlaps remain in the file. Not yet retrained on.

---

### November 26, 2026 - One-Command Model Build; 11-Class Baseline on Corrected Labels

**Session Goal:** Labelling-loop step 1: one command that extracts, splits, trains, tests and exports. Then the first measurement on the labels as corrected on 2026-11-25.

**Changes:**
- `tools/build_scene_model.py` (new) - extract -> split -> train -> test -> export into its own run folder, `data/models/runs/<name>/`. The folder holds the ONNX model, `class_mapping.json`, `test_metrics.json`, the split manifest, the training history, and `run.json` (git commit, SHA-256 of each label file, split, options, results).
  - Refuses an existing run name, so a result is never overwritten.
  - Refuses to train if extraction had errors (e.g. a missing video): training on part of the data while the totals look normal is the ISSUE-008 failure.
  - Drops the ~110 MB checkpoints unless `--keep-checkpoints`.
- `tools/extract_labeled_frames.py` - `--test-video` holds out every game of one video, which needs a second labelled video. The manifest's `test_game` became `test_games` (a list) plus `held_out` (`game` or `video`). Extraction moved into `extract_all()` for the build command.
- `tools/train_scene_classifier.py` - training moved from `main()` into `train_model()`, which returns a summary.
  - **Default output changed from `data/models` to `data/models/runs/manual`.** The old default was the deployed model's folder, so a plain training run would have overwritten `data/models/scene_classifier.onnx` with a model whose classes inference can't decode (ISSUE-017). `train_model()` now refuses that folder.
- `tests/test_scene_splits.py` - 5 new tests: video hold-out; the only video can't be held out; game and video together are rejected; the build's run folder and `run.json` (training stubbed); training refuses the deployed folder.
- `README.md`, `docs/ARCHITECTURE.md`, `CLAUDE.md` - the build command.

**Results** (`--test-game 3`, run `2026-11-26_baseline-11class-g3`, Gemini not involved):
- Clean extraction: 1,603 frames, split 933 train / 148 val / 499 test, with 23 dropped in the validation gap.
  - The November 25 split (934 / 190 / 528) still included the 72 stale frames that ISSUE-021 described.
- 8 classes trained:
  - `both_on_ledge` was left out with 1 frame.
  - `one_knocked_down` has no training frames, and `both_knocked_down` has none at all.
- Best checkpoint at epoch 8, from 13 epochs (early stop). 17.8 minutes on MPS.
- **Test accuracy 45.6%**, macro recall 48.7% (496 frames).
  - That is inside the 39-51% range of the November 16-24 runs on the old labels. One run can't show whether the relabel helped.
- `respawning`: recall 82.1% (32/39), precision 47.1% (32/68).
- `one_on_ledge` is heavily over-predicted: 85 predicted, 10 correct (11.8% precision). Most of the false ones were `one_airborne` (22) and `one_offstage` (17). It is the same rare-class over-prediction as before.
- ONNX check: a batch of 3 matches PyTorch within 2.1e-6. Deployed model unchanged.
- Fast suite: 84 passed, 6 xfailed.

---

### November 26, 2026 - Three New Scene Classes Defined; Plan Reordered

**Changes:**
- `docs/SCENE_CLASSES.md` - three whole-screen classes, in the owner's definitions (14 classes in all):
  - **`pre_game`:** the entrances and countdown, until GO! appears.
  - **`game_end`:** the GAME! or TIME! screen, until the white flash to results. The GAME! screen was `respawning` for one day.
  - **`finishing_blow`:** the red-flash zoom the game shows when a hit is likely to KO a player on their final stock.
  - They are checked before the player states. The KO flash stays `respawning`: it means a KO happened, on any stock.
  - **Why:** `non_gameplay` mixed menus with characters on stage, and `respawning` mixed respawns with the end of the game. A learned GO!/GAME! signal could also back up the colour-based game-boundary checks (ISSUE-022).
- `docs/ARCHITECTURE.md` - the class count, and that the three new classes are not yet supported or labelled.
- Future Improvements - plan reordered, so the class list is settled before video 2 is labelled:
  - Every class change means relabelling every labelled video, so all three classes go in together.
  - Label editing (step 2) and the new classes with ISSUE-017 and relabelling `XUkRx6CpL04` (step 3) now come before game bounds, the skeleton and the review screen.
- PyQt5 installed in `.venv` (`uv pip install`); the label tool opens `XUkRx6CpL04` with all 3 games.

**Results:** Definitions and plan only. No labels, code or models changed.

---

### November 26, 2026 - Label Tool Help Text; Game-Boundary Detection Checked on an Unseen Video

**Session Goal:** Correct `scene_labeler.py`'s out-of-date instructions, and find out whether GO!/GAME! detection can supply game bounds for unlabelled videos (the planned skeleton label file).

**Changes:**
- `tools/scene_labeler.py` - help text, legend and prompts only; no behaviour change.
  - The "Game Boundary Convention" said to mark games at GO! and GAME!. The tool actually makes games contiguous, and the labelled file was cut that way (0-204 / 204-437.1 / 437.1-646.4s, menus included). The help text now says to press G in the menus between games.
  - Added `R` (respawning) to the key list.
  - Aligned the class descriptions with `SCENE_CLASSES.md`: offstage includes below the stage, knocked_down includes teching, and respawning includes the GAME! screen.
  - Commands now use `.venv/bin/python`.
- `docs/ISSUES.md` - ISSUE-022 and ISSUE-023 (below).
- `docs/SCENE_CLASSES.md` - the TIME! screen (clock ran out) is `respawning`, by the same rule as GAME!. It was undefined, and the labelling-loop plan has to handle games that end without GAME!. Owner's decision. No labelled video has a TIME! screen, so no label changes.
- Future Improvements - added the labelling-loop plan (order agreed 2026-11-26). Game bounds are confirmed by a person and checked in code against the reviewed labels, because detection alone cannot be trusted on a new broadcast.

**Results** (detection run with character detection stubbed out; 7 videos, 23 games):
- **Game count:** correct in all 7 videos.
- **Benchmark sets** (the 6 detection was tuned on): game starts within ~1s of the hand-marked times, except `7oPdiBsgSG0` game 1 (18s detected, 12s hand-marked).
- **`XUkRx6CpL04`** (not used in tuning): every game starts 15.5-22.5s early, because the character select screen is detected as GO!. That is ISSUE-022.
- **GAME! timing:** detected 0-1.3s after the first GAME! frame (sampling is every 0.5s, and the first frames of the GAME! animation are not always detected). Detection alone does not give frame-exact game ends.
- **Code reading:** a game whose GO! is missed is dropped, and a missed mid-set GAME! merges two games. This was confirmed with stubbed detectors but not seen in the 23 real games. That is ISSUE-023.

**Left Open:**
- ISSUE-022 and ISSUE-023.
- `scene_labeler.py` rounds new segment times to 0.1s, which is up to 3 frames at 60fps. To be fixed in the label-tool upgrade.
- PyQt5 is not installed in `.venv` (`tools/requirements-labeling.txt`), so the label tool was not launched to check the text changes.

---

### November 25, 2026 - Stale Frames and Batch-1 ONNX Export Fixed (ISSUE-020, ISSUE-021)

**Session Goal:** Fix the two open bugs in the labelling loop's path (labels -> frames -> training -> model) before building on it.

**Changes:**
- `tools/extract_labeled_frames.py` - `remove_video_frames()` deletes a video's previously extracted frames before extracting it again. Fixes ISSUE-021.
  - Frames are named by their label, so after a label edit the old frames stayed filed under their old class, and the splits are built from everything in the folder. The GAME! relabel earlier today left 72 stale frames this way.
  - Matching uses the parsed video ID, not a filename prefix, so a video whose ID begins with another's is not touched. Other videos' frames are kept.
  - It runs only after the video opens, so a wrong path cannot delete good frames.
  - The run prints how many frames were replaced.
- `tools/train_scene_classifier.py` - `export_onnx()` now produces a model that accepts any batch size, and `check_onnx_export()` proves it before the run ends. Fixes ISSUE-020.
  - Cause: with an example batch of 1, the default exporter (torch.export, PyTorch 2.13) fixes the batch size at 1 even though `dynamic_axes` marks it variable. Fix: an example batch of 2 and the dimension declared with `dynamic_shapes` (minimum 1).
  - First tried `dynamo=False`, the older TorchScript exporter, which the ISSUE-020 entry suggested. It works, but PyTorch warns it is deprecated and will be removed, so it was replaced.
  - Found by the new check while testing: exported from Apple's MPS device, the batch size came out fixed at **2** instead, while the same export from the CPU worked. The export now always runs on a CPU copy of the model.
  - `check_onnx_export()` runs a batch of 3 through onnxruntime and PyTorch and raises if the shapes differ or the outputs differ by more than 1e-3. A broken export now fails the training run instead of failing later in whatever loads the model.
  - The model is written as one file (`external_data=False`); a `.onnx.data` file left by an earlier export is removed.
- `tests/test_scene_splits.py` - 4 tests: re-extraction after a relabel leaves no stale frames; another video with a similar ID keeps its frames; a video that won't open deletes nothing; the exported model runs a batch of 3 (exported from MPS when available). Extraction tests use a small synthetic video written with OpenCV.

**Results:**
- The stale-frame test fails with the cleanup switched off and passes with it.
- The real ConvNeXt-Tiny model (8 classes), exported from MPS: batch of 3 matches PyTorch within 1.7e-6. Batches of 1, 3 and 7 checked separately on CPU: within 1.5e-6.
- Fast tests: 79 passed, 6 xfailed. `ruff check` and `scripts/check_docs.py` pass.
- No model was retrained.

---

### November 25, 2026 - GAME! Screens Relabelled as Respawning (ISSUE-021)

**Session Goal:** Make the labels in `XUkRx6CpL04` agree with the written rule that the GAME! screen is `respawning`.

**Problem:**
- `docs/SCENE_CLASSES.md` (written November 24) says the GAME! screen is `respawning`. The labels, made earlier, did the opposite in all three games: the last gameplay segment ran up to GAME!, and `non_gameplay` started there. A model trained on them learns the opposite of the rule, and the labeller's LLM, which follows the rule, was being scored wrong on those segments.
- Owner's decision: keep the rule and relabel. The stretch before GO! stays `non_gameplay`.

**Changes:**
- `data/scene_classifier/raw_labels/XUkRx6CpL04.json` - in each game, a `respawning` segment from the first frame GAME! is visible to the first frame of the wipe to the results screen. Boundaries were read frame by frame at 60 fps and approved by the owner from the frames:

  | Game | Before | After |
  |---|---|---|
  | 1 | `one_offstage` 189.7-193.5, `non_gameplay` 193.5-204.0 | `one_offstage` 189.7-193.46, `respawning` 193.46-197.28, `non_gameplay` 197.28-204.0 |
  | 2 | `one_offstage` 424.0-426.1, `non_gameplay` 426.1-437.1 | `one_offstage` 424.0-425.9, `respawning` 425.9-429.61, `non_gameplay` 429.61-437.1 |
  | 3 | `both_offstage` 629.4-632.7, `non_gameplay` 632.7-646.4 | `both_offstage` 629.4-632.21, `respawning` 632.21-636.26, `non_gameplay` 636.26-646.4 |

  The old boundaries were 0.03-0.5s after GAME! first appears (frames 11608, 25554, 37933); the wipe starts at frames 11837, 25777, 38176. Nothing else in the file changed.
- `docs/SCENE_CLASSES.md` - `non_gameplay` now states that entrance animations and the countdown before GO! are non-gameplay (this matches the existing labels: in all three games the first gameplay segment starts within 0.5s before GO! is detected). The GAME! bullet under `respawning` now says where it starts and ends.
- `docs/ISSUES.md` - added ISSUE-021; ISSUE-006's `respawning` count updated.

**Results:**
- Clean re-extraction: 1,603 frames (was 1,583). `respawning` 63 -> 107 frames; validation now has 29 `respawning` frames where it had none.
- ISSUE-021 found doing this: re-extracting over the existing folder left 72 stale frames, 22 of them old `non_gameplay` frames inside the new `respawning` stretches, and the split used them. Worked around by extracting into an empty folder.
- No model was retrained. Every scene-classifier number recorded before this entry was measured against the old labels.
- Fast tests: 75 passed, 6 xfailed. `scripts/check_docs.py` passes.

**Left Open:**
- ISSUE-021 - the extractor still does not clear a video's old frames.
- The zoomed finishing blow before each GAME! is still labelled as the state before the zoom, pending the `finishing_blow` class.
- `tools/scene_labeler.py`'s help text still describes a game-boundary convention ("first frame after GO! disappears") that the label file does not follow.

---

### November 25, 2026 - Documentation Cleanup: Slimmer CLAUDE.md, ARCHITECTURE.md, Stale Facts

**Session Goal:** Cut what is loaded into every Claude session, and remove or correct statements that no longer match the code.

**Problem:**
- `CLAUDE.md` (14.2 KB, loaded every session) mixed rules with reference material, and several statements were wrong: "Current Status (January 2026)"; stock loss "confirmed by 2 consecutive low-damage frames" (the code and its own table say 3); "scene classifier processes all frames in one batch" (it runs one frame at a time); game boundaries described as HUD-based (GO!/GAME! markers are primary); Gemini described as the primary character detector while no key is configured; two "Key Files" lists; stage tables for a detector that is switched off.
- The reference half of `DEVELOPMENT.md` (~370 lines under the changelog) described an older project: a file layout without the scene classifier or `tools/`, "Attempt 4 (Final)" stock-loss rules of 2 frames / 30% (now 3 frames / 50%), a single test video, and OCR "accuracy gain" percentages that were never measured.

**Changes:**
- `CLAUDE.md` - rewritten, 14.2 KB -> 4.6 KB. Keeps what every session needs: a map of the docs, what is switched off, code map, how to run things, the change workflow, secrets. Everything else moved to `docs/ARCHITECTURE.md` or dropped as a duplicate of `docs/ISSUES.md`, the README, or the skill list Claude Code already loads. The speculative "Future ML Integration" list was dropped.
- `docs/ARCHITECTURE.md` (new) - how each stage works now, checked against the code: rules and parameters, tried-and-rejected approaches (kept from the old reference sections), scene-classifier numbers, stage notes (moved from `CLAUDE.md`), configuration, Keychain setup, caches. Found while checking: only `cache_dir` in `Settings` is read by any code; the OCR confidence threshold in use is the function default 0.3, not `ocr_confidence_threshold` (0.8).
- `docs/DEVELOPMENT.md` - now the changelog plus Future Improvements; the reference sections were replaced by `ARCHITECTURE.md`. Changelog entries are unchanged except two headings: "December 30/31, 2024" corrected to 2025 (git history starts 2025-12-29). Future Improvements gained the planned `finishing_blow` class.
- `docs/ISSUES.md` - ISSUE-007 reworded: the two blocked videos were downloaded for the October 24 benchmark and all 6 are cached, so "no full 6-video baseline" was no longer true. The bot check can still block a fresh download.
- `.claude/skills/test/SKILL.md` - rewritten. It described cached Gemini responses in `.cache/` (the directory does not exist), an activated environment, and test categories that are not the suite's. Now: fast tests by default, and how to read `xfailed` / `XPASS(strict)`.
- `.claude/agents/ocr-investigator.md` - the inline frame-extraction snippet replaced by the tested `inspect_frames.py` script; removed the warning about broken snippets in `/debug-video` (fixed October 23, ISSUE-010).
- `.claude/agents/benchmark-runner.md` - removed the January 11 with-Gemini baseline table. It predates large code changes and was only there to be declared not comparable.
- `.claude/skills/log-changes/SKILL.md` - step 7 now names `docs/ARCHITECTURE.md` as the current-state doc to keep in sync.
- `scripts/check_docs.py`, `scripts/hooks/pre-commit` - `docs/ARCHITECTURE.md` added to the files whose `file.py:line` references are checked.
- `tools/debug_stock_losses.py` - deleted. A one-off script hardcoded to one video and game with a `/tmp` cache; `inspect_frames.py` and `ocr-investigator` replaced it.
- `runitback/cli.py`, `tools/train_scene_classifier.py` - a comment (character sampling is 10/25/50/75/90%, not 25/50/75%) and a docstring ("9 gameplay states") corrected. No behaviour change.
- `README.md` - retraining section says data is split by game and why; links to `ARCHITECTURE.md` and `SCENE_CLASSES.md`.

**Results:**
- `scripts/check_docs.py` passes; `ruff check` passes; fast tests 75 passed, 6 xfailed.
- Instruction text loaded per session (`CLAUDE.md`): 14.2 KB -> 4.6 KB. `DEVELOPMENT.md`: 62 KB -> 50 KB.

**Left Open:**
- Unused `Settings` fields (`video_quality`, `sample_interval_sec`, `ocr_confidence_threshold`, `stage_match_threshold`, `verbose`) were documented, not removed: removing a field can make an existing `.env` that sets it fail validation.
- `pyproject.toml`'s `train` extra duplicates `tools/requirements-training.txt` and lacks `onnx`/`onnxscript`. Not changed.
- The README still gives no scene-classifier accuracy figure (owner's decision pending).

---

### November 24, 2026 - Scene Class Definitions; Batch-1 ONNX Export (ISSUE-020)

**Session Goal:** Write down the labelling rules for the scene classes, and
train a model that never saw game 3, for the separate `runitback-labeler`
project (an agentic pre-labelling pipeline, evaluated on game 3).

**Changes:**
- `docs/SCENE_CLASSES.md` (new) - the owner's definitions of all 11 classes,
  plus a tie-break ranking (respawning > offstage > on_ledge > knocked_down >
  airborne > grounded). Until now only `non_gameplay` and `respawning` had a
  written description (comments in `tools/scene_labeler.py`), so labels relied on
  unwritten conventions, and an LLM labeller would have had to guess them.
  "Offstage" now explicitly includes being below the stage between its edges.
  Zoomed-in finishing blows are planned as a 12th class, `finishing_blow`:
  the zoom doesn't always end in a kill, so labelling it `respawning` would
  record an outcome the frame doesn't show, and "the state before the zoom"
  (the old practice) can't be learned from a single frame. Not yet added to the
  labeller, training or inference. This is the master copy; `runitback-labeler`
  keeps a copy.
- `CLAUDE.md` - links the definitions from the Improvement Path section.
- `docs/ISSUES.md` - added ISSUE-020 (Low): with torch 2.13 the default
  (dynamo) ONNX exporter fixes the batch size at 1 despite `dynamic_axes`; a
  batch of 2 fails in onnxruntime. Found because the labeller sends batches.
  RunItBack's inference runs one frame at a time, so the pipeline is unaffected.

**Results:**
- Retrained the game-3-held-out fold with the current code (no code change
  since November 16; model written to `runitback-labeler`, not `data/models/`):
  **39.2%** accuracy on game 3 (490 frames), macro recall 44.9%. The same fold
  scored 45.1% on November 16 - about 6 points of run-to-run variation, so
  single-run differences of that size are noise. Held-out scores across runs so
  far: 39-51%.
- `respawning`: recall 24/24, precision 24/127 (18.9%). `one_on_ledge`: 79
  predicted for 14 true. Both rare classes are heavily over-predicted (the
  inverse-frequency weighting noted on November 16).
- Re-exporting with `dynamo=False` gives a working any-batch model: batches of
  1 and 5 match PyTorch within 2e-6.

**Left Open:**
- ISSUE-020 - `tools/train_scene_classifier.py` still uses the default exporter.
- `finishing_blow` class: add to `scene_labeler.py`, the training class list and
  inference (inference is blocked on ISSUE-017 anyway). `XUkRx6CpL04`'s
  existing zoom frames would need relabelling.

---

### November 16, 2026 - Game-Level Split and First Held-Out Scene Accuracy (ISSUE-018, ISSUE-019)

**Session Goal:** Replace the leaky frame-level train/val/test split with a game-level one, and measure scene-classifier accuracy on footage the model has not seen.

**Problem:**
- `create_train_val_test_splits()` shuffled each class's frames at random. Frames 0.5s apart are near-duplicates, so a frame at t=10.0s could train the model while t=10.5s tested it. The ~92% (corrected October 23) was the best epoch's *validation* score on that split.
- No test score had ever been measured: training loaded only `train/` and `val/`.

**Changes:**
- `tools/extract_labeled_frames.py` - splits by game by default. Fixes ISSUE-018.
  - **test** = one whole held-out game (`--test-game 3`, or `VIDEO_ID:3` once several videos are labeled; default the last game).
  - **val** = the last 15% of each other game's labeled span, with the 5s of training frames before it dropped (`--val-gap`), so no training frame neighbours a validation frame. Val only picks the checkpoint.
  - Writes `split_manifest.json` recording how the split was made.
  - `--split-by frame` keeps the old split, for comparison only. `--splits-only` re-splits without re-reading the video.
  - Why game and not video: only one video is labeled (ISSUE-006). Its 3 games are on 3 stages, so a held-out game also tests an unseen stage.
- `tools/train_scene_classifier.py` - fixes ISSUE-018 and ISSUE-019.
  - **Test evaluation:** after the best checkpoint is chosen, scores the test split once. Prints per-class recall and saves accuracy, macro recall and the confusion matrix to `test_metrics.json`. Warns when given a non-game split.
  - **Class order:** val and test labels are re-indexed by class name to training's order (`reindex_by_name`). ImageFolder numbers only the folders present in each split, so a split missing a class would score every later class against the wrong output. `class_mapping.json` is now written in the model's real (alphabetical) output order, not labeler order.
  - **Classes too thin to learn:** classes under `--min-class-frames` (10) training frames are left out and reported. The first run kept `both_on_ledge` (1 frame). Its inverse-frequency weight was 8.0, against 0.04-0.08 for common classes. The model predicted it for 202 of 234 frames and reached 7% training accuracy (below the 11% of chance), so the run was stopped.
  - **A claim checked and withdrawn:** I first said the old weight tensor, sized by `SCENE_CLASSES`, gave an absent class a ~1e6 weight that shrank every real class's weight toward zero. Tested, it did nothing: `CrossEntropyLoss` divides by the summed target weights, so the loss is identical at weights x1e-7, and an absent class is never a target. The weights are now sized to the model's outputs, which the variable class count requires anyway.
  - `NUM_CLASSES` was removed - nothing uses a fixed class count any more.
- `tools/requirements-training.txt` - added `onnxscript`. With torch 2.13 the default ONNX exporter needs it, and every run in this session failed at the export step without it. Test scores are saved before export, so none were lost. Verified afterwards: a fold's checkpoint exported, and ONNX matched PyTorch to 1.7e-6.
- `tests/test_scene_splits.py` (new, 10 tests):
  - each game stays whole; val is more than the gap away from train;
  - test-game selection and its errors; manifest and folders agree;
  - the frame split still exists and still leaks;
  - `reindex_by_name` aligns a split that is missing a class.
  - The torchvision test is skipped where torchvision is absent (CI).
- `docs/ISSUES.md` - ISSUE-018 and ISSUE-019 fixed; ISSUE-017 added (below).
- Reference section "Scene Classification" (below) - split description updated.

**Results** (ConvNeXt-Tiny, same settings for every run; the one labeled video `XUkRx6CpL04`; 8 classes trained):

| Split | Test frames | Test accuracy | Macro recall | Best val |
|-------|-------------|---------------|--------------|----------|
| Game 3 held out (Final Destination) | 490 | 45.1% | 52.3% | 55.7% |
| Game 1 held out (Smashville) | 482 | 48.1% | 49.8% | 61.8% |
| Game 2 held out (Battlefield) | 605 | 51.1% | 49.4% | 65.7% |
| **Game folds, pooled** | 1,577 | **48.3%** (762) | - | - |
| Frame split (leaky, for comparison) | 242 | 76.4% | 79.7% | 72.2% |

Chance is 12.5% for 8 classes; always answering the most common class scores about 20%.

- **Respawning, the class the stock-loss roadmap depends on:**
  - recall **61/63 (96.8%)** across the three held-out games;
  - precision **61/113 (54.0%)**, mostly false alarms in game 3 (64 predicted, 24 real).
  - The 63 frames come from only about 10 respawn events, so this is a small sample.
  - Scores are per frame, before the 5-frame median smoothing the pipeline applies.
- **Weakest classes:**
  - `one_airborne` recall 1-52%, mostly read as `both_grounded`.
  - `one_on_ledge` precision 0-14%: it is predicted 4-5x more often than it occurs.
  - Inverse-frequency weighting still pulls predictions toward rare classes.
- **The leak, measured:** on the same code and labels, the frame split scores 76.4% against 48.3% for held-out games - a 28-point gap.
  - The original ~92% is not reproduced. It was a best-epoch validation score, from older code and a 7-class label set, so it cannot be split into leak and other causes.
- Test suite: 75 passed, 6 xfailed (fast suite). ruff and mypy clean.

**Left Open:**
- ISSUE-017: inference hardcodes the deployed model's 7-class order (`scene_classifier.py:101`). A retrained 8-class model cannot be deployed until it reads `class_mapping.json`. The deployed model was not replaced: all runs wrote to scratch directories.
- Softer class weighting (square-root inverse frequency) is the next experiment for the over-predicted rare classes.
- Every game is from one video - same players, overlay and camera. A new broadcast will likely score lower; measuring that needs more labeled videos (ISSUE-006).

---

### October 24, 2026 - Final Stocks Without Gemini (ISSUE-005, ISSUE-016)

**Session Goal:** Raise stock-loss recall with a measured fix to the largest failure class: final stocks missed when no Gemini key is set.

**Problem:**
- On October 23 (Gemini off, 4 sets), 9 of 17 misses were final stocks. A game's last death has no respawn or damage reset after it. Only the Gemini post-pass could find it: it asked Gemini to spot the GAME! screen, and with no key every call returned nothing.
- Gemini was only needed to *find* GAME!. The CLI already finds it locally, and `_infer_game_end_loser` already decided the loser without Gemini.

**Ideas measured before adopting:**
- **HUD stock-icon counting** (`detect_stock_counts`, brightness-based) against known counts across Steve vs Palutena game 1: **1 of 14 correct**. Right before GAME! it said P1 had no stocks when P2 was losing. Not used.
- **Gameplay -> non-gameplay transition from the scene classifier.** Rejected as the primary signal: it tells when a game ends, not who lost, and GAME! detection was already reliable.
- **Blank damage display at GAME!** (proposed by Ankit): the loser's damage number disappears while the winner's stays. Read with OCR on the frame after GAME! at all 14 game endings of the 4 cached sets, judging "blank" by whether the **raw** text has a digit (tags parse as numbers, ISSUE-001): **11 right, 0 wrong, 3 undecided**. Adopted.

**Changes:**
- `runitback/classification/detector.py` - `_detect_final_stocks_locally()` runs when there is no key. Fixes ISSUE-005.
  - It searches back up to 40s from each game's end for the GAME! screen with `detect_game_screen`, then steps to its first frame.
  - It skips the game if a loss within 8s before GAME! was already detected, so nothing is double counted.
  - `_loser_from_blank_damage()` decides the loser by majority over 4 frames (2s) after GAME!. If that's undecided, it uses `_infer_game_end_loser` (damage difference > 20%, then losses so far). If still undecided, it records nothing; the Gemini path would default to P1.
- `runitback/cv/ocr.py` - `has_damage_digits()`: whether the raw OCR text of a damage region contains a digit.
- `runitback/classification/detector.py` - stock cap. Fixes ISSUE-016.
  - With more than 3 verified losses for a player in a game, the cap kept the *first* three chronologically, so a game's final stock - always the latest - was always the one dropped. In the first benchmark of the ISSUE-005 fix, the cap removed most of the recovered finals (logs: "still have 4-7 verified stock losses ... Keeping first 3", 12 times).
  - It now protects one loss per game: the **earliest** loss by either player within `FINAL_STOCK_WINDOW_SECS` (6s) of the game's end, then keeps the earliest others.
  - A first version protected the *latest* near-end loss *per player*. Benchmarked, that let post-GAME! screen noise win: Mario vs Ness kept "P1 at 3:51" and "P2 at 7:23" over the real final at 7:17, and fell from 9/13 to 7/13. Corrected and re-measured.
- `tests/test_stock_loss_logic.py` - 13 new tests:
  - final stock credited correctly to either player, never double counted, skipped rather than guessed, absent without a GAME! screen;
  - the blank display deciding, overriding a damage guess, and falling back;
  - the cap keeping the final stock, protecting only the earliest near-end loss, and ignoring the winner's late "loss".
  - The key cap test fails without the fix.

**Results** (benchmark, Gemini off; 6 sets, 20 games, 89 hand-labeled losses):

| Set | Oct 23 | ISSUE-005 fix | + ISSUE-016 v1 | + ISSUE-016 corrected |
|-----|--------|---------------|----------------|-----------------------|
| Ken vs King Dedede | 8/8 | 8/8 | 8/8 | 8/8 |
| Ken vs Sonic | 10/13 | 12/13 | 12/13 | 12/13 |
| Dark Pit vs Ken | 15/18 | 15/18 | 16/18 | 16/18 |
| Steve vs Palutena | 12/23 | 12/23 | 15/23 | 15/23 |
| Cloud vs Pit | (not downloaded) | 14/14 | 14/14 | 14/14 |
| Mario vs Ness | (not downloaded) | 9/13 | 7/13 | 9/13 |
| **Original 4 sets** | **72.6%** (45/62) | 75.8% | 82.3% | **82.3%** (51/62) |
| **All 6 sets** | - | 78.7% | 80.9% | **83.1%** (74/89) |
| False positives (6 sets) | - | 31 | 29 | **27** (precision 73.3%) |

- Missed final stocks: **9 -> 1** (19 of 20 game endings caught). The remaining one is Steve vs Palutena game 5 at 22:25.
- The gain on the original 4 sets (+6) is below the 8 finals recovered. Keeping a final stock under the cap can displace an earlier real loss when OCR false positives (ISSUE-001) have filled a player's three slots.
- The final run used three parallel benchmark processes (2 sets each). Scratch runner only - not committed.

**Left Open:**
- ISSUE-001 false positives remain the main limit on both recall and precision. They are also why the cap must choose at all.
- The Jan 11 with-Gemini baseline (75.3%, 67/89) is not a like-for-like comparison - the code has changed since. `benchmark-runner` now carries this run as the no-Gemini baseline.

---

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

**Follow-up: repo prepared for outside reviewers (part 1 of 2)**

*Changes:*
- `LICENSE` (new) - MIT text. `pyproject.toml` and the README already declared MIT, but there was no license file, so GitHub showed none.
- `debug_stock_loss.py`, `debug_stock_loss_detailed.py` - deleted. They were scratch scripts from December 31, hardcoded to the old machine's path, and replaced by `tools/debug_stock_losses.py`.
- Fast tests: before this, 31 of 34 tests downloaded full videos, so a reviewer could run almost nothing.
  - `tests/test_damage_parsing.py`, `test_stock_loss_logic.py`, `test_cli_helpers.py` - pure logic, no video or models. Stock-loss tests feed synthetic per-frame damage into the detector.
  - `tests/test_frame_detectors.py` - GO!/GAME!/menu detection and OCR on real frames stored in `tests/fixtures/frames/` (~540 KB). Full frames are JPEG; damage crops are PNG, because JPEG shifted an OCR read from 113 to 115.
  - Known bugs are strict expected failures: ISSUE-001 (two tests), 002, 003 and 014. A fix flips them to passing, and the suite then fails until the marker is removed, so a fix can't go unnoticed.
  - Result: `pytest -m "not slow"` gives 52 passed, 6 xfailed in ~7s.
- `.github/workflows/tests.yml` (new) - runs the instruction checker and the fast tests on every push. It is unverified until its first run on GitHub.
- `pyproject.toml` - filters PaddleOCR's harmless "No ccache found" warning out of test output.
- ISSUE-014 recorded, found while writing the tests: P1's 118.9% is read as 18.9%, because the leading digit sits on the ROI's left edge.

**Follow-up: lint and type checks pass (part 2 of 3)**

The README told reviewers to run `ruff check .` and `mypy runitback`. Before this, ruff reported 285 findings and mypy did not run at all.

*Changes:*
- **Ruff - now passes.**
  - ~120 safe automatic fixes across 24 files: modern `X | None` annotations, import order, unused imports.
  - Line limit raised from 100 to 120, which left 20 long lines to wrap by hand. Log messages were split into adjacent strings, so the output is identical.
  - `N806` is ignored: UPPER_CASE tuning constants inside functions (e.g. `STOCK_LOSS_COOLDOWN`) are deliberate and documented by those names.
  - `llm_detector.py` is exempt from line length, because wrapping the Gemini prompt text would change what Gemini receives.
  - Dead variables removed, a misplaced import moved, Qt override names marked as allowed.
- **Real bug found by ruff:** `character_detector.py` defined `detect_character_from_portrait` twice with different parameters. The second definition shadowed the first, so the public `detect_characters()` always raised `TypeError`.
  - Nothing in the project calls it, so no results were affected.
  - The first definition is renamed `match_portrait`.
- **mypy - now runs and passes**, in relaxed mode.
  - It never ran before: the config targeted Python 3.11, but the installed numpy's stubs use 3.12 syntax.
  - Fixed: `python_version = "3.12"`, and `ignore_missing_imports` for onnxruntime, yt_dlp and paddleocr.
  - The 36 relaxed-mode errors are fixed. Most were wrong annotations (`list[int]` holding tuples, numpy `bool_` returned as `bool`, `max(key=d.get)`).
  - A few matter at runtime: `StateSegment.stock_losses` defaulted to `None` but is `.extend()`ed (now `field(default_factory=list)`), and `CharacterTemplate.mask` was attached after construction (now a declared field).
  - The scene classifier's guard is made explicit (`self._session is None`), and OpenCV stub gaps get targeted `# type: ignore` comments.
  - Strict mode (28 remaining) is added to Future Improvements, not ISSUES.md, since these aren't bugs.
- **Verified:** ruff, mypy, `check_docs.py` and the fast tests all pass. Every `runitback` module imports and every file compiles.
- **Left open:** the first CI run of `a1def5b` failed at "Fast tests" on Linux, after install and the instruction check passed. The job log needs a GitHub login to read, so the cause is unknown.

**Follow-up: README rewritten for outside readers (part 3 of 3)**

*Changes:*
- `README.md` - rewritten. Every claim was checked against the code or a run this session.
  - Added: a pipeline diagram, a results table (today's benchmark, Gemini off, 4 of 6 sets, with the reason), and the agreed Gemini wording.
  - Fixed contradictions: the "Supported Stages" list (detection is switched off), the ~74% recall figure (measured with Gemini), and an understated character limitation.
  - Development section: `pytest -m "not slow"` as the default (plain `pytest` downloads videos), plus the ruff/mypy/checker commands, which now pass.
  - Corrected the disadvantage tags to the real display names (`Offstage/Edgeguard`, `Ledgetrap`, `Tech Chase`).
  - Retraining instructions: `scene_labeler.py` needs a video argument, and dependencies come from `tools/requirements-*.txt`.
  - Added CI and license badges, short design notes, a status section linking ISSUES.md, and a note that Claude Code is used with versioned instructions.
- Not changed, noted: the `[train]` extra in `pyproject.toml` omits `onnx`, which `requirements-training.txt` includes. The README points to the requirements file.

**Follow-up: first CI run failed - OCR broken on Linux**

- *Symptom:* `gh run view --log-failed` showed both OCR fixture tests reading `None`. Every OCR call raised `ConvertPirAttribute2RuntimeAttribute not support`.
- *Wrong first diagnosis:* CI had installed paddleocr 3.7.0 (PP-OCRv6 models) while this Mac had 3.4.1 (PP-OCRv5). Pinning to 3.4.x and PP-OCRv5 seemed to explain it.
  - The second CI run had **identical** versions and models to this Mac and failed with the same error, which ruled it out.
- *Actual cause:* PaddlePaddle on Linux x86 CPUs runs through oneDNN (Intel's CPU acceleration, PaddleOCR's `enable_mkldnn`, on by default). With paddlepaddle 3.3 that path fails on every call. Apple Silicon never uses oneDNN, which is why it only broke on Linux.
  - Impact: **OCR returned nothing on any Linux x86 machine**, so damage reading, and with it stock-loss detection, was silently broken there. That includes a reviewer's machine. The fast tests caught it on their first run.
- *Changes:*
  - `runitback/cv/ocr.py` - both `PaddleOCR(...)` calls set `enable_mkldnn=False` (the fix) and `ocr_version="PP-OCRv5"`, the model every result was measured with.
  - `pyproject.toml` - `paddleocr>=3.4,<3.5`, the tested version. This guards against unreviewed upgrades; it was not the fix.
  - Disabling oneDNN may slow OCR on Intel/AMD CPUs, which is preferable to no OCR.
- *Found:* ISSUE-015 - OCR engine failures are swallowed per frame, which is why this showed up only as missing readings.
- *Line references re-pointed:* the ruff and mypy edits shifted lines in `ocr.py`, `detector.py`, `character_detector.py`, `cli.py` and others.
  - 14 `file.py:line` references in ISSUES.md, CLAUDE.md and `ocr-investigator.md` still pointed at real lines but at the wrong code (some at blank lines), so `check_docs.py` passed them.
  - All were re-pointed by searching for their code, and the `ocr.py` ones again after the oneDNN fix shifted that file.
  - This is the checker's known gap: it proves a line exists, not that it is the right one.

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
   - Extracts frames at 0.5s intervals (0.25s for rare classes)
   - Creates train/val/test splits **by game** (since November 16, 2026): one
     held-out game is the test set; the last 15% of each other game, after a
     5s gap, is validation. The original 70/15/15 random frame split leaked
     near-duplicate frames and survives only as `--split-by frame`.

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
>
> **Measured 2026-11-16** (game-level split, retrained on the current labels):
> **48.3%** test accuracy pooled over three held-out games, with respawning
> recall 61/63. The same code on the leaky frame split scores 76.4%. See the
> November 16 entry. The deployed model itself has still not been scored on
> held-out data.

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

### December 31, 2025 - Gemini Stock Counter Detection

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

### December 30, 2025 - Stock Loss Tracking Implementation

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

## Future Improvements

1. **Template Matching for OCR** - Create digit templates for more robust damage reading
2. **ML-based Digit Recognition** - Train CNN for SSBU damage numbers
3. **Character Position Tracking** - Motion detection + Kalman filter for edgeguard detection
4. **Slippi Integration** - For Melee replays, read game memory directly
5. **Stage-specific Ledge Detection** - Use stage geometry data for ledgetrap identification
6. **Strict type checking** - Turn on mypy `strict = true`. 28 errors remain (October 23, 2026), mostly missing generic parameters (`dict` instead of `dict[str, int]`) and unannotated functions. The real type bugs were already fixed in relaxed mode.
7. **New scene classes** - `pre_game`, `game_end` and `finishing_blow` were defined on 2026-11-26 (see `docs/SCENE_CLASSES.md`). They are planned in the labelling loop, step 3.
8. **Labelling loop with `runitback-labeler`** - see below.

### Labelling loop (planned order, revised 2026-11-27)

Goal: label new videos by approving machine drafts instead of labelling from
scratch. `runitback-labeler` (a separate repo) writes draft labels. A person
reviews every draft in `tools/scene_labeler.py`. Only reviewed games reach training.

**Principle:** detection and models *propose*; a person *confirms*; code
*refuses* anything unconfirmed. This includes game bounds. Scene classes do not
depend on game bounds (`extract_labeled_frames.py` uses the game only to tag
frames for the split). A boundary that cuts through gameplay, however, leaks one
real game into two splits, so bounds are confirmed like any other label.

**The class list is settled before any new video is labelled.** Each change to
it means relabelling every labelled video, so all three new classes go in at
once (step 3), before the gold videos are labelled.

**Testing and training use different labels (decided 2026-11-27).** A label the
model helped make cannot be used to score that model, or any later version of
it: in a draft, segments the model and the LLM agree on are accepted without a
person looking, so the model would be marked right on its own mistakes (on
game 3, 41% of game time was accepted that way and 22% of it was wrong). So:
- **Gold videos** test. One game of each is labelled by hand, blind (before any
  draft of it exists). No game of a gold video is ever trained on, so its score
  is a true unseen-video score. The set starts with two gold videos (step 4)
  and then grows slowly: a new one when a new kind of broadcast appears.
- **Training videos** are drafted by the labeller and reviewed. Unreviewed
  errors there teach the model its own mistakes, so each draft is reviewed in
  full or spot-checked (step 9), and the gold set shows whether training on
  drafts actually helps.
- `XUkRx6CpL04` (hand-labelled before any draft) stays a training video; its
  game 3 remains the labeller's within-video measurement.

1. **Train-and-export command.** *(Done 2026-11-26: `tools/build_scene_model.py`.)*
   One command that trains, evaluates on held-out
   data, and exports ONNX, with a hold-out choice of a game (now) or a whole
   video (once there are two). Its first run is the baseline on the corrected
   11-class labels.
2. **Label editing in `scene_labeler.py`.** *(Done 2026-11-27, plus overwrite
   from a mark to the playhead; see the changelog.)*
   - Select a segment and change its class.
   - Split a segment at the playhead, move a segment's edges, delete a segment;
     segments never overlap.
   - Store exact times instead of rounding to 0.1s.
   - Keep fields the tool doesn't know (`source`, `needs_review`,
     `model_label`, `llm_label`, `llm_reason`, `mean_confidence`,
     `min_confidence`, and the format-1 game fields including `produced_by`).
   - `source` records who decided a segment's class (agreed with
     runitback-labeler 2026-11-27): `human` when a person creates a segment,
     changes its class, or moves one of its edges; unchanged on pieces left by
     a cut or an overwrite.
3. **The three new classes, end to end.**
   - *(Done 2026-11-27.)* Keys P = `pre_game`, X = `game_end`,
     F = `finishing_blow` in the label tool; one shared class list for the
     tool, extraction and training.
   - *(Done 2026-11-27.)* ISSUE-017: inference reads the class order from
     `class_mapping.json`.
   - *(Done 2026-11-27.)* Code that uses class names: the `SceneState` enum
     and its gameplay test (`pre_game` and `game_end` are not gameplay), and
     the detector's disadvantage tags. Its `respawning` rules needed no change:
     they key on `respawning` alone, so a GAME! screen predicted as `game_end`
     is no longer part of a respawn run.
   - **Relabel `XUkRx6CpL04`** with the step-2 editing. Claude proposes
     frame-exact times; the owner approves before anything is written:
     - `game_end`: the 3 GAME! segments, a class change only.
     - `pre_game`: each game's entrance start (its end is the GO! already marked).
     - `finishing_blow`: each red-flash zoom, split out of the segment it sits in.
   - Retrain with step 1's command, and give the labeller the model.
   - `finishing_blow` shows at most a few times per set (final stock only), so
     it may stay under the 10-frame training minimum until more videos are
     labelled. Say so in the results rather than inflating it.
4. **Two gold videos, before any other video is labelled.** The owner
   chooses two best-of-3 sets (fewer games set aside than a best of 5), from
   different channels or overlays than `XUkRx6CpL04` and from each other, with
   different stages and characters, at 1080p, ideally including a finishing
   blow and a TIME! ending. One of the six stock-loss benchmark sets would also
   let `analyze` be checked on it.
   - The owner labels one game of each by hand with all 14 classes, blind:
     before the labeller runs on it. This can start as soon as the P/X/F keys
     exist (step 3), before the retrain. The videos' other games are not
     labelled and never trained on; they may be hand-labelled later as more
     test data.
   - *(Done 2026-11-27.)* Gold label files are made with
     `scene_labeler.py --gold`: they live in
     `data/scene_classifier/gold_labels/`, not `raw_labels/` (which training
     reads), and carry `"role": "gold"`. `extract_labeled_frames.py` refuses
     any file marked gold, and a video cannot have labels in both folders.
     Scoring a model on gold files is step 5.
   - A model may be scored on a gold game only if the label file says `gold`
     **and** its `run.json` shows the video in none of `split.train_games`
     (which already lists every game used for training and validation). The
     labeller checks both, not the role alone.
   - Two gold games from the start replace the earlier options for the
     labeller's threshold (choose and check on one gold game, or keep a
     separate model trained on games 1-2): with two, the threshold is chosen
     on one and checked on the other.
5. **First unseen-video results**, on the two gold games. Nothing else needs
   building first: the labeller reads game bounds from a hand-label file.
   - **Scene model** trained on all of `XUkRx6CpL04`: accuracy and per-class
     recall and precision on each gold game. The first cross-video figures.
   - **Labeller:** drafts both gold games (measurement only, never imported).
     Its threshold is chosen on gold game A and checked on gold game B,
     reporting B's accepted time and how much of it is wrong. This is a paid
     LLM run (about 200 calls a game) and waits for the owner's approval.
   - **Then decide, from these numbers:** the threshold for the new model
     (including whether to set it high, such as 0.8); whether "model and LLM
     agree -> accept" is safe enough (it is the larger error source: 22% of
     agreed time wrong on game 3); and whether drafts save enough labelling
     time to justify steps 6-8.
6. **Game bounds.**
   - Split the GO!/GAME! part of `detect_games` into a shared function without
     character detection.
   - Fix ISSUE-022 (character select detected as GO!) and ISSUE-023 (a missed
     marker drops or merges a game).
   - Tighten `test_timestamp_accuracy` from 30s to about 2s.
   - Measure on more broadcasts: about 10 videos from different channels.
     Claude proposes each GO!/GAME! time with a thumbnail; the owner confirms.
   - Without that measurement, accuracy on new broadcasts is unknown: the one
     unseen video checked so far exposed ISSUE-022.
   - Once trained, `pre_game` and `game_end` are a second, learned signal for
     game bounds alongside the colour checks.
7. **Skeleton tool** - `tools/make_label_skeleton.py` writes
   `data/scene_classifier/skeletons/<video_id>.json` (gitignored). It is the
   label-file format with empty `labels`, plus these fields:
   - Top level: `format` (1), `duration_sec`, `detection` (method, counts,
     warnings), and `role` (`gold` or `training`, set by the owner when the
     video is chosen; see step 4). The labeller refuses to draft a game of a
     gold video until a hand-label file has that game, which keeps gold
     labelling blind in code, not only by habit.
   - Per game: `status` (`skeleton` -> `draft` -> `labelled`),
     `bounds_confirmed` (false), `ends_with_game_screen` (proposed: true when
     GAME! was detected), `go_sec` and `game_sec` (approximate: GAME!
     is detected 0-1.3s late, and `null` when not found), and `start_source` /
     `end_source` (`go`, `game`, `next_go`, `hud`, `video_end`).
   - Games are contiguous. Each game ends at its `game_sec` + 10s, and the next
     game starts there.
   - A game whose GO! or GAME! was not found is still emitted, with its sources
     saying so, never dropped.
   - A file or game with no `status` is a finished legacy label (`labelled`,
     bounds confirmed).
8. **Review screen and extraction check.**
   - Import a draft for one game into the video's label file.
   - **Bounds confirmation:** per game, four thumbnails (start, GO!, GAME!,
     end), with adjusting by key. The person also confirms whether the game
     ends with a GAME! or TIME! screen; it may not, if the recording or stream
     cut off. Sets `bounds_confirmed` and `ends_with_game_screen`.
   - **Review mode:** N jumps to the next `needs_review` segment, Enter accepts
     the proposed class, the class keys change it. A side panel shows the
     model's and the LLM's label and the LLM's reason, `min_confidence` (the
     least sure sampled frame; `mean_confidence` too if there is room), and the
     game's `produced_by` (scene-model hash and folder name, LLM model and
     prompt id). Accepting or changing sets `source=human` and clears
     `needs_review`.
   - **Spot checks:** the labeller marks a seeded random sample of its
     auto-accepted segments `spot_check: true` (separate from `needs_review`;
     seed and rate recorded in `produced_by`). N visits them too. The person's
     verdict is always recorded, `spot_check: "confirmed"` or `"corrected"`,
     because a segment agreed with would otherwise look the same as one never
     looked at, and no error rate could be computed. `"corrected"` means the
     class was wrong; a boundary-only fix with the class right is
     `"confirmed"` (agreed with runitback-labeler). It is not inferred from
     `source`: an edge move also makes a segment `human`. The rate is reported
     by time and by count. It is not blind (the person sees the proposal), so
     it reads a little low; the gold set is the unbiased number.
   - A game is saved as `labelled` only when its bounds are confirmed and no
     segment needs review.
   - **Extraction** (`extract_labeled_frames.py`): games that are not
     `labelled` are skipped and listed in the output, so a finished game can be
     used while others are still drafts. A `labelled` game is refused
     (non-zero exit, naming file and game) if any segment still needs review or has an unanswered spot check,
     if its boundary falls outside a `non_gameplay` segment, or if
     `ends_with_game_screen` is true and it has no `game_end` segment before
     its boundary. The rule keys on the person's confirmation, not on
     `end_source`, which is only what detection found. Tolerance is 0.05s.
9. **Training videos:** best-of-5 sets (most training data per video), drafted
   by the labeller against their skeletons with all 14 classes, and reviewed
   by the owner in `scene_labeler.py`.
   - Every segment is reviewed, not only `needs_review` ones (the strips make a
     full pass quick), until the gold numbers show the accepted labels are
     reliably right. Only then drop to spot checks: a random sample of accepted
     segments, so each video's accepted-error rate is still known.
   - After each retrain, score the gold set: a model trained on drafts must
     beat the one before it there, or the drafts' errors are winning.
10. **Generalise** the label tool and training beyond this class set and project.
