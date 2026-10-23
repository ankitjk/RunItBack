"""Claude's instructions must only name commands, flags, and code that exist."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKER = ROOT / "scripts" / "check_docs.py"


def run_checker(*files: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECKER), *map(str, files)],
        capture_output=True, text=True, cwd=ROOT,
    )


def test_instructions_reference_real_things():
    result = run_checker()
    assert result.returncode == 0, result.stdout


def test_checker_catches_the_bugs_it_was_written_for(tmp_path):
    # Each line is a mistake that shipped in a skill or agent before the checker existed.
    bad = tmp_path / "SKILL.md"
    bad.write_text(
        "```bash\n"
        'python -m runitback analyze "<url>" --debug\n'
        "```\n"
        "```python\n"
        "from runitback.video.frame_extractor import FrameExtractor\n"
        "from runitback.cv.llm_detector import detect_game_screen_with_gemini\n"
        "```\n"
        "See `ocr.py:99999`.\n"
    )
    out = run_checker(bad).stdout
    assert "bare 'python'" in out
    assert "--debug' - no such option" in out
    assert "without --player" in out
    assert "'FrameExtractor' is not defined" in out
    assert "'detect_game_screen_with_gemini' is not defined" in out
    assert "ocr.py has only" in out
