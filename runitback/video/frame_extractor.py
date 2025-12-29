"""Frame extraction from video files using OpenCV."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """A single video frame with metadata."""

    image: np.ndarray
    timestamp: float  # Seconds from start of video
    frame_number: int

    @property
    def height(self) -> int:
        """Get frame height in pixels."""
        return self.image.shape[0]

    @property
    def width(self) -> int:
        """Get frame width in pixels."""
        return self.image.shape[1]


class FrameExtractionError(Exception):
    """Raised when frame extraction fails."""

    pass


def extract_frames(
    video_path: Path,
    interval_sec: float = 0.5,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> Iterator[Frame]:
    """Extract frames from a video at regular intervals.

    Args:
        video_path: Path to the video file
        interval_sec: Seconds between extracted frames (default 0.5 = 2 FPS)
        start_sec: Start time in seconds (default 0)
        end_sec: End time in seconds (default None = end of video)

    Yields:
        Frame objects containing the image and metadata

    Raises:
        FrameExtractionError: If the video cannot be opened or read
    """
    if not video_path.exists():
        raise FrameExtractionError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FrameExtractionError(f"Could not open video: {video_path}")

    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if fps <= 0:
            raise FrameExtractionError("Invalid video FPS")

        logger.info(
            f"Video: {video_path.name}, FPS: {fps:.2f}, "
            f"Duration: {duration:.2f}s, Total frames: {total_frames}"
        )

        # Calculate frame interval
        frame_interval = int(fps * interval_sec)
        if frame_interval < 1:
            frame_interval = 1

        # Calculate start and end frames
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps) if end_sec else total_frames

        # Seek to start position if needed
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = start_frame
        extracted_count = 0

        while frame_count < end_frame:
            ret, frame = cap.read()

            if not ret:
                break

            # Only yield frames at the specified interval
            if (frame_count - start_frame) % frame_interval == 0:
                timestamp = frame_count / fps
                yield Frame(
                    image=frame,
                    timestamp=timestamp,
                    frame_number=frame_count,
                )
                extracted_count += 1

            frame_count += 1

        logger.info(f"Extracted {extracted_count} frames")

    finally:
        cap.release()


def get_video_properties(video_path: Path) -> dict:
    """Get properties of a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary with video properties (fps, width, height, duration, frame_count)
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FrameExtractionError(f"Could not open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        return {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration": duration,
        }
    finally:
        cap.release()


def extract_single_frame(video_path: Path, timestamp_sec: float) -> Frame:
    """Extract a single frame at a specific timestamp.

    Args:
        video_path: Path to the video file
        timestamp_sec: Timestamp in seconds

    Returns:
        Frame object at the specified timestamp

    Raises:
        FrameExtractionError: If the frame cannot be extracted
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FrameExtractionError(f"Could not open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        target_frame = int(timestamp_sec * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()

        if not ret:
            raise FrameExtractionError(
                f"Could not read frame at timestamp {timestamp_sec}s"
            )

        return Frame(
            image=frame,
            timestamp=timestamp_sec,
            frame_number=target_frame,
        )
    finally:
        cap.release()
