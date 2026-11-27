# RunItBack

Python CLI that analyzes Super Smash Bros. Ultimate tournament sets from
YouTube video alone: game boundaries, characters, stock losses, and time in
neutral / advantage / disadvantage.

## Where things are written down

Read these when the task needs them; don't restate their content here.

| Doc | Holds |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How each pipeline stage works now: rules, parameters, config, caches, what was tried and rejected |
| [docs/ISSUES.md](docs/ISSUES.md) | Every open bug, by severity. The only bug list |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Changelog, newest first: what changed, why, measured results. Long - read the entries you need, not the file |
| [docs/SCENE_CLASSES.md](docs/SCENE_CLASSES.md) | Definitions and tie-break rules for the scene classes |
| [README.md](README.md) | Outside readers: install, usage, per-set benchmark table, file layout |

## Things that are off or limited

These change how to run things and how to report results. Numbers and detail
are in docs/ARCHITECTURE.md.

- **Gemini is off:** no `GEMINI_API_KEY` is configured. Characters come from
  template matching and are often wrong (ISSUE-013); say so when reporting them.
- **Stage detection is off:** every game reports `unknown` (ISSUE-009).
- **The deployed scene model is the old 7-class one.** Inference reads a
  model's classes from the `class_mapping.json` beside it, so a run folder's
  model can be deployed by copying both files, but none has been: that is a
  deliberate decision. Only one video is labelled (ISSUE-006).

## Code map

| Path | Purpose |
|---|---|
| `runitback/cli.py` | CLI, game boundary detection (GO!/GAME! screens), report output |
| `runitback/classification/detector.py` | Game state and stock-loss detection |
| `runitback/classification/states.py` | `FrameState`, `StateSegment`, `StockLoss` |
| `runitback/cv/ocr.py` | Damage reading (PaddleOCR) |
| `runitback/cv/ui_detector.py` | HUD regions, GO!/GAME! detection, frame types |
| `runitback/cv/scene_classifier.py` | Scene classification (ConvNeXt-Tiny, ONNX) |
| `runitback/cv/character_detector.py`, `llm_detector.py` | Characters: Gemini, or template matching without a key |
| `tools/` | Labelling (`scene_labeler.py`; edits and the label-file format in `label_segments.py`), model build in one command (`build_scene_model.py`: splits via `extract_labeled_frames.py`, training via `train_scene_classifier.py`), benchmark (`test_stock_losses.py`) |
| `scripts/` | `check_docs.py`, git hooks, `fetch_model.sh` |

## Running things

- Use `.venv/bin/python` for everything. The venv is not activated in Claude's
  shell and bare `python` is not on PATH.
- `analyze` stops to prompt unless `--player` is given:
  `.venv/bin/python -m runitback analyze "<url>" --player 1 [--games 1,3] [--verbose]`.
  A full set takes ~20 minutes; run it in the background.
- Fast tests (~10s, no downloads): `.venv/bin/python -m pytest -m "not slow"`.
  The slow tests download the benchmark sets.
- The benchmark takes about two hours. Use the `benchmark-runner` agent.
- Agents: `benchmark-runner`, `ocr-investigator` (why one moment was misread),
  `label-auditor` (is there enough labelled data).

## Workflow for every change

1. Make the change. Measure before claiming an improvement; report numbers with
   their configuration (Gemini on or off, which sets).
2. `/issues` - move a fixed bug to Fixed, or add one you found.
3. `/log-changes` - the DEVELOPMENT.md entry goes in the same commit as the change.
4. Commit. The message must end with an issue line, above `Co-Authored-By`:
   `Fixes: ISSUE-001`, `Found: ISSUE-012`, or `Issues: none`.
   `scripts/hooks/commit-msg` rejects a line that does not match `docs/ISSUES.md`.

Scale the record to the change: a typo gets one bullet and `Issues: none`.

Hooks are enabled once per clone with `git config core.hooksPath scripts/hooks`.
The pre-commit hook scans for secrets and runs `scripts/check_docs.py`, which
checks every command, flag, import, script path and `file.py:line` reference in
`CLAUDE.md`, `.claude/`, `docs/ISSUES.md` and `docs/ARCHITECTURE.md` against the code. It proves names
exist, not that a command works: **run a command before writing it into a
skill, agent or doc**, and prefer a script in the skill's folder over a long
inline snippet.

Planned features are not issues; they go under "Future Improvements" in
`docs/DEVELOPMENT.md`.

## Secrets

Never commit credentials. A Gemini key sat in public history for nine months
(`f8e1834`); it is deleted and must not be reused. `GEMINI_API_KEY` is read
from the environment only - a `.env` entry is ignored. Keychain setup is in
docs/ARCHITECTURE.md. Bypass the secret scan (`git commit --no-verify`) only
for a confirmed false positive.
