from __future__ import annotations

import time

import numpy as np

from config.settings import Settings
from vision.detector import ObjectDetector, Detection
from vision.color_classifier import ColorClassifier
from vision.outline_detector import OutlineDetector


class VisionPipeline:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._detector = ObjectDetector(settings)
        self._classifier = ColorClassifier(settings)
        self._outline = OutlineDetector(settings)

    @property
    def detector(self) -> ObjectDetector:
        return self._detector

    @property
    def classifier(self) -> ColorClassifier:
        return self._classifier

    @property
    def outline_detector(self) -> OutlineDetector:
        return self._outline

    def process(self, bgr_frame: np.ndarray) -> tuple[list[Detection], float]:
        t0 = time.perf_counter_ns()
        detections = self._detector.detect(bgr_frame)
        t1 = time.perf_counter_ns()
        duration_ms = (t1 - t0) / 1e6
        return detections, duration_ms
