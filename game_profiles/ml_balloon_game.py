from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from game_profiles.balloon_game import BalloonGameProfile
from game_profiles.balloon_types import TargetType


class MLBalloonGameProfile(BalloonGameProfile):
    """Balloon profile using OpenCV spatial candidates + ML semantics.

    OpenCV supplies candidate geometry and visual evidence. MobileNet owns the
    semantic class for candidates covered by the trained model. Game mechanics
    remain downstream and decide priorities/sequences/timing.
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
        total_t0 = time.perf_counter()
        candidates = self.detect_spatial_candidates(bgr_frame)

        classified = self._ml_classifier.classify_detections(
            bgr_frame,
            [d.as_dict() for d in candidates],
        )
        self._last_profile["ml"] = dict(self._ml_classifier.last_stats)
        self._last_profile["ml_candidate_count"] = len(candidates)

        result: list[dict] = []
        rejected_count = 0
        for detection in classified:
            label = detection.get("ml_class")
            confidence = float(detection.get("ml_confidence", 0.0))
            metadata = dict(detection.get("metadata") or {})

            if confidence < self._ml_classifier.threshold:
                rejected_count += 1
                continue
            if label == "ui":
                rejected_count += 1
                continue

            if label == "notification":
                target_type = TargetType.NOTIFICATION.value
                is_twin = False
                is_special = False
            elif label == "duo":
                target_type = TargetType.TWIN_BALLOON.value
                is_twin = True
                is_special = False
            elif label == "super_g_ring":
                target_type = TargetType.SPECIAL_SEQUENCE_BALLOON.value
                is_twin = False
                is_special = True
            elif label == "balloon":
                target_type = TargetType.NORMAL_BALLOON.value
                is_twin = False
                is_special = False
            else:
                continue

            detection["target_type"] = target_type
            detection["is_twin"] = is_twin
            detection["is_special_sequence"] = is_special
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
        self._last_profile["ml"]["accepted_count"] = len(result)
        self._last_profile["ml"]["rejected_count"] = rejected_count

        # ML is the sole semantic source for the ML runtime. Legacy Gemini,
        # G-symbol, Super-G, and golden-duo semantic detectors are intentionally
        # absent from this path. Notification/duo/super-G semantics come from
        # the trained classifier above.
        result.sort(key=lambda d: (
            d.get("center", (0, 0))[1],
            d.get("center", (0, 0))[0],
            d.get("target_type", ""),
        ))
        self._last_profile["ml_semantic_total_ms"] = (
            time.perf_counter() - total_t0
        ) * 1000.0
        self._last_profile["total_ms"] = self._last_profile["ml_semantic_total_ms"]
        return result

    @staticmethod
    def _assign_ml_duo_pairs(detections: list[dict]) -> None:
        """Give nearby ML-duo candidates a shared mechanics pair id."""
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
                distance = math.hypot(
                    float(a[0]) - float(b[0]),
                    float(a[1]) - float(b[1]),
                )
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
