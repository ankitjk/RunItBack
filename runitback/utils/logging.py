"""Logging configuration for RunItBack."""

import logging
import sys


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (default INFO)
        log_file: Optional file path for log output
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Get root logger for our package
    logger = logging.getLogger("runitback")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("yt_dlp").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ppocr").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the runitback prefix.

    Args:
        name: Logger name (will be prefixed with 'runitback.')

    Returns:
        Logger instance
    """
    return logging.getLogger(f"runitback.{name}")
