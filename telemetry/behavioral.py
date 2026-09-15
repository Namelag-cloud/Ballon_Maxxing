from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from telemetry.timing import now_monotonic


@dataclass
class BehavioralFeatures:
    reaction_time: Optional[float] = None
    reaction_time_variance: float = 0.0
    click_offset_from_center: float = 0.0
    click_interval: float = 0.0
    pair_click_delta: Optional[float] = None
    targets_visible: int = 0
    session_duration: float = 0.0


class BehavioralExtractor:
    def __init__(self) -> None:
        self._click_times: list[float] = []
        self._reaction_times: list[float] = []
        self._click_offsets: list[float] = []
        self._pair_deltas: list[float] = []
        self._session_start: float = now_monotonic()
        self._last_click_time: float = 0.0

    def record_detection_time(self, track_id: int, first_seen: float) -> None:
        pass

    def record_click(
        self,
        track_id: int,
        click_pos: tuple[int, int],
        target_center: tuple[int, int],
        first_seen: float,
        now: float,
    ) -> None:
        reaction = now - first_seen
        self._reaction_times.append(reaction)

        dx = click_pos[0] - target_center[0]
        dy = click_pos[1] - target_center[1]
        offset = (dx * dx + dy * dy) ** 0.5
        self._click_offsets.append(offset)

        if self._last_click_time > 0:
            interval = now - self._last_click_time
            self._click_times.append(interval)

        self._last_click_time = now

    def record_pair_click(self, delta: float) -> None:
        self._pair_deltas.append(delta)

    def extract(self, targets_visible: int) -> BehavioralFeatures:
        now = now_monotonic()
        rt_var = float(np.var(self._reaction_times)) if self._reaction_times else 0.0
        mean_offset = float(np.mean(self._click_offsets)) if self._click_offsets else 0.0
        mean_interval = float(np.mean(self._click_times)) if self._click_times else 0.0
        mean_pair = float(np.mean(self._pair_deltas)) if self._pair_deltas else None

        last_rt = self._reaction_times[-1] if self._reaction_times else None

        return BehavioralFeatures(
            reaction_time=last_rt,
            reaction_time_variance=rt_var,
            click_offset_from_center=mean_offset,
            click_interval=mean_interval,
            pair_click_delta=mean_pair,
            targets_visible=targets_visible,
            session_duration=now - self._session_start,
        )
