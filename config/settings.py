from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from behavior.config import BehaviorConfig


class RunMode(Enum):
    DRY_RUN = "dry-run"
    LIVE = "live"
    COLLECT = "collect"


@dataclass
class Region:
    top: int = 0
    left: int = 0
    width: int = 1920
    height: int = 1080


@dataclass
class CaptureConfig:
    backend: str = "mss"
    region: Region = field(default_factory=Region)
    fps_limit: Optional[int] = None


@dataclass
class ColorThreshold:
    lower_hsv: list[int] = field(default_factory=lambda: [0, 0, 0])
    upper_hsv: list[int] = field(default_factory=lambda: [180, 255, 255])


@dataclass
class DetectionConfig:
    min_area: int = 500
    max_area: int = 50000
    min_confidence: float = 0.7
    morphology_kernel: int = 3


@dataclass
class OutlineConfig:
    enabled: bool = True
    min_white_ratio: float = 0.10
    min_contrast: float = 30.0
    border_sample_width: int = 10


@dataclass
class TrackingConfig:
    max_distance: float = 50.0
    max_frames_missing: int = 5
    new_track_threshold: int = 3


@dataclass
class GroupingConfig:
    enabled: bool = True
    same_color_only: bool = True
    require_outline: bool = False
    max_pair_distance: float = 500.0


@dataclass
@dataclass
class InputConfig:
    backend: str = "sendinput"
    enabled: bool = True
    click_delay_ms: int = 0


@dataclass
class FeedbackConfig:
    combo_roi: tuple[int, int, int, int] | None = None
    score_roi: tuple[int, int, int, int] | None = None
    pop_roi: tuple[int, int, int, int] | None = None
    change_threshold: float = 0.08


@dataclass
class DebugConfig:
    show_detections: bool = True
    show_bounding_boxes: bool = True
    show_centers: bool = True
    show_colors: bool = True
    show_outline: bool = True
    show_track_ids: bool = True


@dataclass
class Settings:
    mode: RunMode = RunMode.LIVE
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    colors: dict[str, ColorThreshold] = field(default_factory=dict)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    outline: OutlineConfig = field(default_factory=OutlineConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    input: InputConfig = field(default_factory=InputConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)

    @classmethod
    def from_dict(cls, data: dict) -> Settings:
        mode = RunMode(data.get("mode", "live"))

        cap_data = data.get("capture", {})
        region_data = cap_data.get("region", {})
        capture = CaptureConfig(
            backend=cap_data.get("backend", "mss"),
            region=Region(**region_data),
            fps_limit=cap_data.get("fps_limit"),
        )

        colors = {}
        for name, thresh in data.get("colors", {}).items():
            colors[name] = ColorThreshold(
                lower_hsv=list(thresh["lower_hsv"]),
                upper_hsv=list(thresh["upper_hsv"]),
            )

        det_data = data.get("detection", {})
        detection = DetectionConfig(**det_data)

        out_data = data.get("outline", {})
        outline = OutlineConfig(**out_data)

        track_data = data.get("tracking", {})
        tracking = TrackingConfig(**track_data)

        group_data = data.get("grouping", {})
        grouping = GroupingConfig(**group_data)

        input_data = data.get("input", {})
        inp = InputConfig(**input_data)

        feedback_data = data.get("feedback", {})
        feedback = FeedbackConfig(**feedback_data) if feedback_data else FeedbackConfig()

        debug_data = data.get("debug", {})
        debug = DebugConfig(**debug_data)

        behavior_data = data.get("behavior", {})
        behavior = BehaviorConfig.from_dict(behavior_data) if behavior_data else BehaviorConfig()

        return cls(
            mode=mode,
            capture=capture,
            colors=colors,
            detection=detection,
            outline=outline,
            tracking=tracking,
            grouping=grouping,
            input=inp,
            feedback=feedback,
            debug=debug,
            behavior=behavior,
        )


def load_settings(path: Optional[str] = None) -> Settings:
    if path is None:
        path = str(Path(__file__).parent / "defaults.json")
    with open(path, "r") as f:
        data = json.load(f)
    return Settings.from_dict(data)
