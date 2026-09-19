from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from game_profiles.balloon_game import BalloonGameProfile
from game_profiles.balloon_types import TargetType


class MLBalloonGameProfile(BalloonGameProfile):
    """Balloon profile with ML semantic classification after OpenCV candidates.

    OpenCV remains the spatial candidate source. MobileNet owns the semantic
    class used by the downstream decision engine. Existing G-symbol and
    multicolor geometry lanes are preserved because the current ML dataset does
    not contain those classes.
    """

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

        # The current five-class model has no G-symbol/Gemini labels. Keep
        # those geometry-specific lanes untouched until representative data
        # exists for them.
        semantic_candidates = [
            detection
            for detection in candidates
            if detection.get("target_type") not in {
                TargetType.G_SYMBOL.value,
                TargetType.MULTICOLOR_SPECIAL.value,
            }
        ]
        preserved = [
            detection
            for detection in candidates
            if detection.get("target_type") in {
                TargetType.G_SYMBOL.value,
                TargetType.MULTICOLOR_SPECIAL.value,
            }
        ]

        classified = self._ml_classifier.classify_detections(
            bgr_frame,
            semantic_candidates,
        )
        self._last_profile["ml"] = dict(self._ml_classifier._last_stats)
        self._last_profile["ml_candidate_count"] = len(semantic_candidates)

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

        self._assign_ml_duo_pairs(result)
        result.extend(preserved)
        result.sort(key=lambda d: (d.get("center", (0, 0))[1], d.get("center", (0, 0))[0]))
        return result

    @staticmethod
    def _assign_ml_duo_pairs(detections: list[dict]) -> None:
        """Give exactly two nearby ML-duo candidates a shared mechanics pair id."""
        twins = [
            d for d in detections
            if d.get("target_type") == TargetType.TWIN_BALLOON.value
        ]
        unused = set(range(len(twins)))
        pair_number = 0

        while unused:
            first_index = min(unused)
            unused.remove(first_index)
            first = twins[first_index]
            best_index = None
            best_distance = math.inf

            for second_index in unused:
                second = twins[second_index]
                if first.get("color_name") != second.get("color_name"):
                    continue
                a = first.get("center")
                b = second.get("center")
                if not a or not b:
                    continue
                distance = math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
                if 50.0 <= distance <= 150.0 and distance < best_distance:
                    best_index = second_index
                    best_distance = distance

            if best_index is None:
                continue

            unused.remove(best_index)
            second = twins[best_index]
            pair_number += 1
            ordered = sorted((first.get("center"), second.get("center")))
            pair_id = (
                f"ml_duo:{ordered[0][0]}:{ordered[0][1]}:"
                f"{ordered[1][0]}:{ordered[1][1]}:{pair_number}"
            )
            for member in (first, second):
                member["metadata"] = {
                    **dict(member.get("metadata") or {}),
                    "semantic_pair_id": pair_id,
                    "semantic_class": "golden_duo",
                    "semantic_confirmed": True,
                }
