from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2

from config.settings import Settings


@dataclass
class ColorMatch:
    name: str
    confidence: float
    hsv_lower: np.ndarray
    hsv_upper: np.ndarray


class ColorClassifier:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._matches: list[ColorMatch] = []
        for name, thresh in settings.colors.items():
            lower = np.array(thresh.lower_hsv, dtype=np.uint8)
            upper = np.array(thresh.upper_hsv, dtype=np.uint8)
            self._matches.append(ColorMatch(
                name=name,
                confidence=1.0,
                hsv_lower=lower,
                hsv_upper=upper,
            ))

    def classify_mask(self, hsv_frame: np.ndarray, mask: np.ndarray) -> tuple[str, float]:
        best_name = "unknown"
        best_area = 0
        total_area = int(cv2.countNonZero(mask))
        if total_area == 0:
            return "unknown", 0.0

        for match in self._matches:
            color_mask = cv2.inRange(hsv_frame, match.hsv_lower, match.hsv_upper)
            intersected = cv2.bitwise_and(color_mask, mask)
            area = int(cv2.countNonZero(intersected))
            if area > best_area:
                best_area = area
                best_name = match.name

        confidence = best_area / total_area if total_area > 0 else 0.0
        return best_name, confidence

    def make_color_mask(self, hsv_frame: np.ndarray, color_name: str) -> np.ndarray:
        for match in self._matches:
            if match.name == color_name:
                return cv2.inRange(hsv_frame, match.hsv_lower, match.hsv_upper)
        return np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

    def make_all_nonwhite_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        white_thresh = self._settings.colors.get("white")
        if white_thresh is None:
            return np.ones(hsv_frame.shape[:2], dtype=np.uint8) * 255

        white_lower = np.array(white_thresh.lower_hsv, dtype=np.uint8)
        white_upper = np.array(white_thresh.upper_hsv, dtype=np.uint8)
        white_mask = cv2.inRange(hsv_frame, white_lower, white_upper)
        return cv2.bitwise_not(white_mask)

    def is_white(self, hsv_frame: np.ndarray, mask: np.ndarray) -> bool:
        name, conf = self.classify_mask(hsv_frame, mask)
        return name == "white" and conf > 0.5
