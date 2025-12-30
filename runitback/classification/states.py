"""Game state enums and data structures."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GameState(Enum):
    """Game state from a player's perspective."""

    NEUTRAL = "neutral"
    ADVANTAGE = "advantage"
    DISADVANTAGE = "disadvantage"
    UNKNOWN = "unknown"


@dataclass
class StockLoss:
    """Record of a stock loss event.

    Attributes:
        timestamp: When the stock was lost (in seconds)
        player: Which player lost the stock (1 or 2)
    """

    timestamp: float
    player: int


@dataclass
class StateSegment:
    """A continuous segment of gameplay in a particular state.

    Attributes:
        state: The game state during this segment
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        start_frame: Frame index where segment starts
        end_frame: Frame index where segment ends
        stock_losses: List of stock losses that occurred during this segment
    """

    state: GameState
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    stock_losses: list[StockLoss] = None

    def __post_init__(self):
        if self.stock_losses is None:
            self.stock_losses = []

    @property
    def duration(self) -> float:
        """Duration of this segment in seconds."""
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        return (
            f"StateSegment({self.state.value}, "
            f"{self.start_time:.1f}s - {self.end_time:.1f}s)"
        )


@dataclass
class FrameState:
    """State information for a single frame.

    Attributes:
        timestamp: Frame timestamp in seconds
        frame_index: Index in the frame list
        p1_damage: Player 1 damage percentage (None if not detected)
        p2_damage: Player 2 damage percentage (None if not detected)
        p1_stocks: Player 1 stock count (None if not detected from splash)
        p2_stocks: Player 2 stock count (None if not detected from splash)
        p1_visible: Whether player 1 is visible (not in magnifying glass)
        p2_visible: Whether player 2 is visible (not in magnifying glass)
        state: Detected game state for the analyzed player
        p1_stock_loss: True if P1 lost a stock this frame
        p2_stock_loss: True if P2 lost a stock this frame
    """

    timestamp: float
    frame_index: int
    p1_damage: Optional[float]
    p2_damage: Optional[float]
    p1_stocks: Optional[int] = None
    p2_stocks: Optional[int] = None
    p1_visible: bool = True
    p2_visible: bool = True
    state: GameState = GameState.UNKNOWN
    p1_stock_loss: bool = False
    p2_stock_loss: bool = False


@dataclass
class GameStateAnalysis:
    """Complete game state analysis results.

    Attributes:
        player_perspective: Which player's perspective (1 or 2)
        neutral_segments: List of neutral state segments
        advantage_segments: List of advantage state segments
        disadvantage_segments: List of disadvantage state segments
        frame_states: Per-frame state information
        total_neutral_time: Total time in neutral (seconds)
        total_advantage_time: Total time in advantage (seconds)
        total_disadvantage_time: Total time in disadvantage (seconds)
    """

    player_perspective: int
    neutral_segments: list[StateSegment]
    advantage_segments: list[StateSegment]
    disadvantage_segments: list[StateSegment]
    frame_states: list[FrameState]

    @property
    def total_neutral_time(self) -> float:
        """Total time spent in neutral."""
        return sum(seg.duration for seg in self.neutral_segments)

    @property
    def total_advantage_time(self) -> float:
        """Total time spent in advantage."""
        return sum(seg.duration for seg in self.advantage_segments)

    @property
    def total_disadvantage_time(self) -> float:
        """Total time spent in disadvantage."""
        return sum(seg.duration for seg in self.disadvantage_segments)

    @property
    def total_analyzed_time(self) -> float:
        """Total analyzed gameplay time."""
        return (
            self.total_neutral_time
            + self.total_advantage_time
            + self.total_disadvantage_time
        )
