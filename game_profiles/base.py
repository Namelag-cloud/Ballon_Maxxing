from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np



@dataclass
class ClickAction:
    track_id: int
    position: tuple[int, int]
    color: str
    confidence: float
    reason: str
    timestamp: float
    metadata: Optional[dict] = None


class GameType(Enum):
    COOKIE_CLICKER = "cookie"
    BALLOON_GAME = "balloons"


@dataclass
class GameAction:
    position: tuple[int, int]
    reason: str
    timestamp: float
    metadata: Optional[dict] = None


class GameProfile(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def game_type(self) -> GameType: ...

    @abstractmethod
    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        """Detect game objects in frame. Returns list of detection dicts."""
        ...

    @abstractmethod
    def select_targets(self, detections: list[dict], frame_shape: tuple[int, int]) -> list[GameAction]:
        """Decide which targets to click based on detections."""
        ...

    def on_click_sent(self, action: GameAction) -> None:
        """Called after a click is sent. Override for game-specific state tracking."""
        pass

    def reset(self) -> None:
        """Reset game-specific state between sessions."""
        pass
