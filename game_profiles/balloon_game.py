from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import cv2

from game_profiles.base import GameProfile, GameType, GameAction
from config.settings import Settings, load_settings
from vision.color_classifier import ColorClassifier
from vision.outline_detector import OutlineDetector, OutlineResult


class BalloonGameProfile(GameProfile):
    """Balloon popping game profile.

    Detects colored balloons, rejects white, handles outlines.
    """

    def __init__(self) -> None:
        self._settings = load_settings()
        self._classifier = ColorClassifier(self._settings)
        self._outline = OutlineDetector(self._settings)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._settings.detection.morphology_kernel, self._settings.detection.morphology_kernel),
        )
        self._track_id_counter = 0

    @property
    def name(self) -> str:
        return "Balloon Game"

    @property
    def game_type(self) -> GameType:
        return GameType.BALLOON_GAME

    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        nonwhite_mask = self._classifier.make_all_nonwhite_mask(hsv)

        k = self._settings.detection.morphology_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cleaned = cv2.morphologyEx(nonwhite_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._settings.detection.min_area or area > self._settings.detection.max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

            obj_mask = np.zeros(cleaned.shape, dtype=np.uint8)
            cv2.drawContours(obj_mask, [contour], -1, 255, thickness=cv2.FILLED)

            color_name, color_conf = self._classifier.classify_mask(hsv, obj_mask)

            if color_name == "white" or color_name == "unknown":
                continue
            if color_conf < self._settings.detection.min_confidence:
                continue

            outline = self._outline.detect(hsv, (x, y, w, h))

            detections.append({
                "bbox": (x, y, w, h),
                "center": (cx, cy),
                "area": int(area),
                "color_name": color_name,
                "color_confidence": color_conf,
                "has_outline": outline.has_outline,
                "outline_white_ratio": outline.white_ratio,
                "outline_contrast": outline.contrast,
            })

        return detections

    def select_targets(self, detections: list[dict], frame_shape: tuple[int, int]) -> list[GameAction]:
        if not detections:
            return []

        now = time.perf_counter_ns() / 1e9

        by_color: dict[str, list[dict]] = {}
        for det in detections:
            by_color.setdefault(det["color_name"], []).append(det)

        actions = []
        for color, color_dets in by_color.items():
            for det in color_dets:
                actions.append(GameAction(
                    position=det["center"],
                    reason=f"balloon_{color}",
                    timestamp=now,
                    metadata={
                        "color": color,
                        "area": det["area"],
                        "confidence": det["color_confidence"],
                        "has_outline": det["has_outline"],
                    },
                ))

        return actions

    def on_click_sent(self, action: GameAction) -> None:
        pass

    def reset(self) -> None:
        self._track_id_counter = 0
