"""Configuration management for RunItBack."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Video settings
    video_quality: str = Field(default="720p", description="Preferred video quality")
    sample_interval_sec: float = Field(
        default=0.5, description="Seconds between frame samples"
    )
    cache_dir: Path = Field(
        default=Path.home() / ".cache" / "runitback",
        description="Directory for cached downloads",
    )

    # Detection thresholds
    ocr_confidence_threshold: float = Field(
        default=0.8, description="Minimum OCR confidence"
    )
    stage_match_threshold: float = Field(
        default=0.7, description="Minimum stage matching score"
    )

    # Output settings
    verbose: bool = Field(default=False, description="Enable verbose output")

    model_config = {
        "env_prefix": "RUNITBACK_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# Global settings instance
settings = Settings()
