from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import cv2

from game_profiles.base import GameProfile, GameType, GameAction


class CookieClickerProfile(GameProfile):
    """Simple black-box profile for Cookie Clicker.

    Detects the big cookie by color/shape and clicks its center.
    """

    def __init__(self) -> None:
        self._click_count = 0

    @property
    def name(self) -> str:
        return "Cookie Clicker"

    @property
    def game_type(self) -> GameType:
        return GameType.COOKIE_CLICKER

    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

        lower = np.array([10, 80, 120], dtype=np.uint8)
        upper = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000 or area > 200000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / h if h > 0 else 0
            if aspect < 0.5 or aspect > 2.0:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.4:
                continue

            cx, cy = x + w // 2, y + h // 2
            detections.append({
                "bbox": (x, y, w, h),
                "center": (cx, cy),
                "area": int(area),
                "circularity": float(circularity),
                "confidence": min(1.0, circularity),
            })

        detections.sort(key=lambda d: d["area"], reverse=True)
        return detections[:1]

    def select_targets(self, detections: list[dict], frame_shape: tuple[int, int]) -> list[GameAction]:
        if not detections:
            return []

        best = detections[0]
        now = time.perf_counter_ns() / 1e9

        return [GameAction(
            position=best["center"],
            reason="cookie_center",
            timestamp=now,
            metadata={"area": best["area"], "circularity": best["circularity"]},
        )]

    def on_click_sent(self, action: GameAction) -> None:
        self._click_count += 1

    def reset(self) -> None:
        self._click_count = 0
