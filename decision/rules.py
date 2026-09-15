from __future__ import annotations

from config.settings import Settings
from tracking.tracker import BalloonTrack


class SafetyRules:
    def __init__(self, settings: Settings) -> None:
        self._config = settings.rules

    def is_safe(self, track: BalloonTrack) -> tuple[bool, str]:
        if not self._config.click_white and track.color == "white":
            return False, "white_balloon"

        x, y, w, h = track.bbox
        area = w * h
        if area < self._config.min_size:
            return False, "too_small"
        if area > self._config.max_size:
            return False, "too_large"

        if track.clicked:
            return False, "already_clicked"

        return True, "ok"
