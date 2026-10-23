"""Show what RunItBack's detectors see at specific timestamps of a video.

For each timestamp: saves the full frame and both damage-ROI crops to
debug_frames/, and prints frame type, GO!/GAME! detection, the raw and
parsed OCR damage for each player, and the scene classification.

Usage (from the project root):
    .venv/bin/python .claude/skills/debug-video/inspect_frames.py <url> <sec> [<sec> ...]
"""

import logging
import sys
from pathlib import Path

import cv2

from runitback.config import settings
from runitback.cv.ocr import read_damage_percentage
from runitback.cv.scene_classifier import classify_scene
from runitback.cv.ui_detector import (
    P1_DAMAGE_ROI,
    P2_DAMAGE_ROI,
    detect_frame_type,
    detect_game_screen,
    detect_go_screen,
    get_roi,
)
from runitback.video.downloader import download_video
from runitback.video.frame_extractor import extract_single_frame


class _RawTextCapture(logging.Handler):
    """Keep the last 'OCR detected' debug line, which carries the raw text."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.last = None

    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith("OCR detected:"):
            # "OCR detected: '<text>' (conf: 0.95)" -> "'<text>'"
            self.last = msg.removeprefix("OCR detected: ").rsplit(" (conf", 1)[0]


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    url, stamps = sys.argv[1], [float(t) for t in sys.argv[2:]]

    capture = _RawTextCapture()
    ocr_logger = logging.getLogger("runitback.cv.ocr")
    ocr_logger.setLevel(logging.DEBUG)
    ocr_logger.addHandler(capture)
    ocr_logger.propagate = False  # capture the raw text without printing every log line

    out = Path("debug_frames")
    out.mkdir(exist_ok=True)
    video = download_video(url, settings.cache_dir)

    for t in stamps:
        image = extract_single_frame(video, t).image
        cv2.imwrite(str(out / f"{t:.1f}s.png"), image)

        print(f"\n=== {t:.1f}s ===")
        print(f"frame type: {detect_frame_type(image).name}   "
              f"GO!: {detect_go_screen(image)}   GAME!: {detect_game_screen(image)}")
        for player, roi_coords in ((1, P1_DAMAGE_ROI), (2, P2_DAMAGE_ROI)):
            roi = get_roi(image, roi_coords)
            cv2.imwrite(str(out / f"{t:.1f}s_p{player}_roi.png"), roi)
            capture.last = None
            value, conf = read_damage_percentage(roi)
            print(f"P{player} damage: {value}  (conf {conf:.2f}, raw {capture.last or 'no text'})")
        scene = classify_scene(image)
        print(f"scene: {scene.state.value}  (conf {scene.confidence:.2f})")

    print(f"\nImages saved to {out}/")


if __name__ == "__main__":
    main()
