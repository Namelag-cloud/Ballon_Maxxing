from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import cv2

from config.settings import Settings
from vision.color_classifier import ColorClassifier
from vision.outline_detector import OutlineDetector, OutlineResult


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    area: int
    color_name: str
    color_confidence: float
    outline: OutlineResult
    tracking_class: str | None = None
    metadata: dict = field(default_factory=dict)


class ObjectDetector:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._classifier = ColorClassifier(settings)
        self._outline = OutlineDetector(settings)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (settings.detection.morphology_kernel, settings.detection.morphology_kernel),
        )

    def detect(self, bgr_frame: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

        nonwhite_mask = self._classifier.make_all_nonwhite_mask(hsv)

        k = self._settings.detection.morphology_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cleaned = cv2.morphologyEx(nonwhite_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        min_a = self._settings.detection.min_area
        max_a = self._settings.detection.max_area

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_a or area > max_a:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

            obj_mask = np.zeros(cleaned.shape, dtype=np.uint8)
            cv2.drawContours(obj_mask, [contour], -1, 255, thickness=cv2.FILLED)

            color_name, color_conf = self._classifier.classify_mask(hsv, obj_mask)

            if color_name == "white":
                continue
            if color_name == "unknown":
                continue
            if color_conf < self._settings.detection.min_confidence:
                continue

            outline = self._outline.detect(hsv, (x, y, w, h))

            detections.append(Detection(
                bbox=(x, y, w, h),
                center=(cx, cy),
                area=int(area),
                color_name=color_name,
                color_confidence=color_conf,
                outline=outline,
            ))

        return detections
