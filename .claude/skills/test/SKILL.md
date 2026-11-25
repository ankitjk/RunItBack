---
name: test
description: Run the RunItBack pytest suite - all tests, quick (non-slow) tests, or a pattern match - and summarize pass/fail results. Use when asked to run tests, check the suite, or verify a change did not break anything.
---

# /test - Run the test suite

## Usage

```
/test              # Fast tests (the default - what CI runs)
/test all          # Everything, including the slow benchmark tests
/test <pattern>    # Tests matching a pattern, e.g. /test scene_splits
```

## Instructions

Run from the project root with `.venv/bin/python` - the venv is not activated
in Claude's shell, and bare `python` is not on PATH.

1. Fast tests (~15s, no downloads or API keys):
   ```bash
   .venv/bin/python -m pytest tests/ -m "not slow" -q
   ```

2. A pattern:
   ```bash
   .venv/bin/python -m pytest tests/ -m "not slow" -q -k "<pattern>"
   ```

3. Everything - only when asked. The slow tests (`tests/test_game_detection.py`)
   download the benchmark sets and take minutes each; run in the background:
   ```bash
   .venv/bin/python -m pytest tests/ -q
   ```

4. Summarize: passed / failed / skipped / xfailed, each failure in a line, and
   the likely cause if it is clear.

## Reading the result

- **`xfailed` is expected.** Known bugs are written as strict expected-failure
  tests linked to their issue in `docs/ISSUES.md`.
- **`XPASS(strict)` counts as a failure** and means a tracked bug is now fixed:
  remove the `xfail` marker and move the issue to Fixed with `/issues fix`.
- `test_scene_splits.py` skips one test when torchvision is not installed.
- `test_docs.py` runs `scripts/check_docs.py`; a failure there means a command
  or `file.py:line` reference in a skill, agent, `CLAUDE.md` or `docs/ISSUES.md`
  no longer matches the code.
