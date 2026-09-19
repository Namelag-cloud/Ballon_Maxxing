from __future__ import annotations

from pathlib import Path

import numpy as np

from game_profiles.balloon_game import BalloonGameProfile
from game_profiles.balloon_types import TargetType


class MLBalloonGameProfile(BalloonGameProfile):
    """Balloon profile with ML semantic classification after OpenCV candidates."""

    def __init__(
        self,
        *,
        checkpoint: str | Path = "training/artifacts/mobilenet_v3_small.pt",
        threshold: float = 0.70,
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        from vision.ml_classifier import MLVisionClassifier

        self._ml_classifier = MLVisionClassifier(
            checkpoint,
            threshold=threshold,
            batch_size=batch_size,
        )

    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        candidates = super().detect(bgr_frame)
        classified = self._ml_classifier.classify_detections(
            bgr_frame,
            candidates,
        )

        result: list[dict] = []
        for detection in classified:
            label = detection.get("ml_class")
            confidence = float(detection.get("ml_confidence", 0.0))
            metadata = dict(detection.get("metadata") or {})

            if confidence < self._ml_classifier.threshold:
                continue

            if label == "ui":
                continue
            if label == "notification":
                detection["target_type"] = TargetType.NOTIFICATION.value
                detection["is_twin"] = False
                detection["is_special_sequence"] = False
            elif label == "duo":
                detection["target_type"] = TargetType.TWIN_BALLOON.value
                detection["is_twin"] = True
                detection["is_special_sequence"] = False
            elif label == "super_g_ring":
                detection["target_type"] = TargetType.SPECIAL_SEQUENCE_BALLOON.value
                detection["is_twin"] = False
                detection["is_special_sequence"] = True
            elif label == "balloon":
                detection["target_type"] = TargetType.NORMAL_BALLOON.value
                detection["is_twin"] = False
                detection["is_special_sequence"] = False
            else:
                continue

            detection["confidence"] = confidence
            detection["ml_confidence"] = confidence
            detection["metadata"] = {
                **metadata,
                "semantic_source": "ml_classifier",
                "semantic_class": label,
                "semantic_confirmed": True,
            }
            result.append(detection)

        return result
