"""Video acquisition and processing modules."""

from runitback.video.downloader import download_video
from runitback.video.frame_extractor import extract_frames, Frame

__all__ = ["download_video", "extract_frames", "Frame"]
