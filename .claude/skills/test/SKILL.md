---
name: test
description: Run the RunItBack pytest suite - all tests, quick (non-slow) tests, or a pattern match - and summarize pass/fail results. Use when asked to run tests, check the suite, or verify a change did not break anything.
---

# /test - Run the test suite

Run pytest for the RunItBack project with common configurations.

## Usage

```
/test              # Run all tests
/test quick        # Run quick tests (exclude slow integration tests)
/test <pattern>    # Run tests matching a pattern (e.g., /test game_detection)
/test -v           # Run with verbose output
```

## Instructions

1. Run pytest:
   ```bash
   .venv/bin/python -m pytest tests/ -v
   ```

2. For quick tests (skip slow ones):
   ```bash
   .venv/bin/python -m pytest tests/ -v -m "not slow"
   ```

3. For a specific test file or pattern:
   ```bash
   .venv/bin/python -m pytest tests/ -v -k "<pattern>"
   ```

4. After running, summarize:
   - Total passed/failed/skipped
   - Any failures with brief descriptions
   - Suggestions if tests fail

## Test Categories

The test suite covers:
- **Game count detection** - Correct number of games identified
- **Timestamp accuracy** - Game start/end times within tolerance
- **Character detection** - Player characters correctly identified
- **Stock loss detection** - Kill events properly tracked
- **Stage detection** - Stage identification (currently not implemented)

## Notes

- Tests use cached Gemini API responses from `.cache/` for speed
- Some tests are marked `@pytest.mark.slow` for integration tests
- The `GEMINI_API_KEY` environment variable must be set for live API tests
- Run from the project root with the project environment active (`pip install -e .` per README)
