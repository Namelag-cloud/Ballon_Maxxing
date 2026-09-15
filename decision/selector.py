from __future__ import annotations

import time
from dataclasses import dataclass

from config.settings import Settings
from tracking.tracker import BalloonTrack
from decision.rules import SafetyRules
from decision.grouping import GroupingStrategy, Group


@dataclass
class ClickAction:
    track_id: int
    position: tuple[int, int]
    color: str
    confidence: float
    reason: str
    timestamp: float


class TargetSelector:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._safety = SafetyRules(settings)
        self._grouping = GroupingStrategy(settings)

    def select(self, tracks: dict[int, BalloonTrack]) -> list[ClickAction]:
        now = time.perf_counter_ns() / 1e9
        actions: list[ClickAction] = []

        safe_tracks: list[BalloonTrack] = []
        for t in tracks.values():
            is_safe, reason = self._safety.is_safe(t)
            if is_safe:
                safe_tracks.append(t)

        groups = self._grouping.group(safe_tracks)

        for group in groups:
            for track in group.tracks:
                actions.append(ClickAction(
                    track_id=track.id,
                    position=track.position,
                    color=track.color,
                    confidence=track.confidence,
                    reason=f"group_{group.color}_{len(group.tracks)}_balloons",
                    timestamp=now,
                ))

        return actions
