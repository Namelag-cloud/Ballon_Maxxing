from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TargetType(Enum):
    NORMAL_BALLOON = "normal_balloon"
    TWIN_BALLOON = "twin_balloon"
    SPECIAL_SEQUENCE_BALLOON = "special_sequence_balloon"
    G_SYMBOL = "g_symbol"
    MULTICOLOR_SPECIAL = "multicolor_special"
    INVALID_WHITE = "invalid_white"


class BalloonColor(Enum):
    BLUE = "blue"
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


class GameMode(Enum):
    NORMAL_PLAY = "normal_play"
    SPECIAL_SEQUENCE = "special_sequence"


class SpecialSequenceStep(Enum):
    BLUE = "blue"
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    COMPLETE = "complete"


@dataclass(frozen=True)
class VisualFeatures:
    """Measured image features retained for classification/debugging."""

    circularity: float = 0.0
    solidity: float = 0.0
    white_border_ratio: float = 0.0
    outline_contrast: float = 0.0
    ring_detected: bool = False
    ring_radius_ratio: float = 0.0
    ring_confidence: float = 0.0
    multicolor_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "circularity": self.circularity,
            "solidity": self.solidity,
            "white_border_ratio": self.white_border_ratio,
            "outline_contrast": self.outline_contrast,
            "ring_detected": self.ring_detected,
            "ring_radius_ratio": self.ring_radius_ratio,
            "ring_confidence": self.ring_confidence,
            "multicolor_score": self.multicolor_score,
        }


@dataclass(frozen=True)
class TargetDetection:
    """Vision-layer description of one visible game object."""

    target_type: TargetType
    color: BalloonColor | None
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    area: int
    confidence: float
    width: float
    height: float
    is_twin: bool = False
    is_special_sequence: bool = False
    visual_features: VisualFeatures = field(default_factory=VisualFeatures)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_type": self.target_type.value,
            "color_name": self.color.value if self.color else None,
            "center": self.center,
            "bbox": self.bbox,
            "area": self.area,
            "color_confidence": self.confidence,
            "confidence": self.confidence,
            "width": self.width,
            "height": self.height,
            "is_twin": self.is_twin,
            "is_special_sequence": self.is_special_sequence,
            "has_outline": self.visual_features.white_border_ratio >= 0.10,
            "outline_white_ratio": self.visual_features.white_border_ratio,
            "outline_contrast": self.visual_features.outline_contrast,
            "visual_features": self.visual_features.as_dict(),
        }


@dataclass(frozen=True)
class TwinAction:
    """One logical action coordinating the two twin balloon clicks."""

    targets: tuple["BalloonAction", "BalloonAction"]
    max_click_delta_ms: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BalloonAction:
    """Decision-layer action contract; execution details belong to behavior."""

    target_id: int | None
    target_type: TargetType
    color: BalloonColor | None
    position: tuple[int, int]
    target_width: float
    is_twin: bool = False
    is_special_sequence: bool = False
    is_g: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BalloonGameState:
    """Minimal persistent decision state for later phases."""

    mode: GameMode = GameMode.NORMAL_PLAY
    special_step: SpecialSequenceStep = SpecialSequenceStep.COMPLETE
    special_sequence_seen: bool = False
    g_active: bool = False
    pending_twin_ids: tuple[int, int] | None = None

    @property
    def required_special_color(self) -> BalloonColor | None:
        mapping = {
            SpecialSequenceStep.BLUE: BalloonColor.BLUE,
            SpecialSequenceStep.RED: BalloonColor.RED,
            SpecialSequenceStep.YELLOW: BalloonColor.YELLOW,
            SpecialSequenceStep.GREEN: BalloonColor.GREEN,
        }
        return mapping.get(self.special_step)
