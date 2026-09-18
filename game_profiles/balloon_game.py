from __future__ import annotations

import time
from dataclasses import replace

import cv2
import numpy as np

from config.settings import load_settings
from game_profiles.balloon_types import (
    BalloonColor,
    TargetDetection,
    TargetType,
    VisualFeatures,
)
from game_profiles.base import GameAction, GameProfile, GameType
from vision.color_classifier import ColorClassifier
from vision.g_symbol_detector import GSymbolDetector
from vision.outline_detector import OutlineDetector, OutlineResult
from vision.detector import Detection
from vision.ring_detector import RingDetector


class BalloonGameProfile(GameProfile):
    """Vision profile for the timed balloon game.

    This class answers only what is visible. Game priority/state transitions are
    intentionally left to the decision layer.

    The reference gameplay video shows the special sequence as four large
    balloons surrounded by a large, thin, color-matched circular outline. Twin
    yellow/green balloons also have a circular outline, but their balloon cores
    are materially smaller and have a stronger whitish/translucent appearance.
    """

    _CORE_FILL_MIN = 0.55
    # A balloon core should occupy most, but not essentially all, of its
    # bounding box. This rejects solid UI bars/panels without using screen
    # coordinates, while remaining tolerant of balloons clipped by a frame edge.
    _CORE_FILL_MAX = 0.90
    _CORE_MIN_CIRCULARITY = 0.45
    _WHITE_CORE_RATIO = 0.50
    _WHITE_MIN_DIAMETER_RATIO = 0.05
    _TWIN_WHITE_BORDER_RATIO = 0.10
    _SPECIAL_RELATIVE_DIAMETER = 0.82

    def __init__(self) -> None:
        self._settings = load_settings()
        self._classifier = ColorClassifier(self._settings)
        self._outline = OutlineDetector(self._settings)
        self._ring = RingDetector()
        self._g_detector = GSymbolDetector(self._classifier)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                self._settings.detection.morphology_kernel,
                self._settings.detection.morphology_kernel,
            ),
        )

    @property
    def name(self) -> str:
        return "Balloon Game"

    @property
    def game_type(self) -> GameType:
        return GameType.BALLOON_GAME

    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        """Return structured detections serialized for the existing profile API."""
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        candidates: list[TargetDetection] = []

        for color_name in ("blue", "red", "yellow", "green"):
            candidates.extend(
                self._detect_color_components(hsv, color_name)
            )

        candidates.extend(self._detect_multicolor_special(hsv))
        candidates.extend(self._g_detector.detect(hsv))

        # Use the upper portion of the observed balloon-size distribution as
        # the frame-local reference for special-vs-twin classification. The
        # reference video shows twins substantially smaller than the normal and
        # special balloon cores, so this is more scale-robust than a fixed pixel
        # threshold while resisting small balloons pulling the reference down.
        balloon_diameters = [
            max(d.width, d.height)
            for d in candidates
            if d.target_type == TargetType.NORMAL_BALLOON
            and d.color is not None
        ]
        reference_diameter = (
            float(np.percentile(balloon_diameters, 75))
            if balloon_diameters
            else 0.0
        )

        classified: list[TargetDetection] = []
        for detection in candidates:
            if detection.target_type == TargetType.NORMAL_BALLOON and detection.color:
                features = detection.visual_features
                ring = self._ring.detect(bgr_frame, detection.bbox)
                features = replace(
                    features,
                    ring_detected=ring.detected,
                    ring_radius_ratio=ring.radius_ratio,
                    ring_confidence=ring.confidence,
                )

                diameter = max(detection.width, detection.height)
                relative_to_reference = (
                    diameter / reference_diameter if reference_diameter > 0 else 0.0
                )

                is_large_ring_target = (
                    ring.detected
                    and ring.radius_ratio >= 1.25
                    and relative_to_reference >= self._SPECIAL_RELATIVE_DIAMETER
                )

                is_twin = (
                    ring.detected
                    and detection.color in (BalloonColor.YELLOW, BalloonColor.GREEN)
                    and not is_large_ring_target
                    and (
                        features.white_border_ratio >= self._TWIN_WHITE_BORDER_RATIO
                        or relative_to_reference < self._SPECIAL_RELATIVE_DIAMETER
                    )
                )

                if is_large_ring_target:
                    target_type = TargetType.SPECIAL_SEQUENCE_BALLOON
                    is_twin = False
                    is_special = True
                elif is_twin:
                    target_type = TargetType.TWIN_BALLOON
                    is_special = False
                else:
                    target_type = TargetType.NORMAL_BALLOON
                    is_special = False

                detection = replace(
                    detection,
                    target_type=target_type,
                    is_twin=is_twin,
                    is_special_sequence=is_special,
                    visual_features=features,
                )

            classified.append(detection)

        classified.sort(key=lambda d: (d.center[1], d.center[0], d.target_type.value))
        return [d.as_dict() for d in classified]

    def to_tracking_detections(self, detections: list[dict]) -> list[Detection]:
        """Adapt profile detections to the generic tracker without exposing game rules."""
        tracking_detections: list[Detection] = []
        for det in detections:
            visual = det.get("visual_features", {})
            outline = OutlineResult(
                has_outline=bool(det.get("has_outline", False)),
                white_ratio=float(det.get("outline_white_ratio", visual.get("white_border_ratio", 0.0))),
                contrast=float(det.get("outline_contrast", visual.get("outline_contrast", 0.0))),
            )
            tracking_detections.append(
                Detection(
                    bbox=tuple(det.get("bbox", (0, 0, 0, 0))),
                    center=tuple(det.get("center", (0, 0))),
                    area=int(det.get("area", 0)),
                    color_name=det.get("color_name") or "unknown",
                    color_confidence=float(det.get("color_confidence", det.get("confidence", 0.0))),
                    outline=outline,
                    tracking_class=det.get("target_type"),
                    metadata=dict(det),
                )
            )
        return tracking_detections

    def select_targets(
        self,
        detections: list[dict],
        frame_shape: tuple[int, int],
    ) -> list[GameAction]:
        """Adapt valid detections to the existing action interface.

        This method deliberately performs no game-priority selection. That
        belongs in the decision layer. Invalid white detections are the sole
        safety filter here because they must never become click actions.
        """
        if not detections:
            return []

        now = time.perf_counter_ns() / 1e9
        actions: list[GameAction] = []
        for det in detections:
            if det.get("target_type") == TargetType.INVALID_WHITE.value:
                continue
            color = det.get("color_name") or "unknown"
            actions.append(
                GameAction(
                    position=det["center"],
                    reason=f"balloon_{color}",
                    timestamp=now,
                    metadata={
                        "color": color,
                        "area": det["area"],
                        "width": det.get("width", 0.0),
                        "height": det.get("height", 0.0),
                        "confidence": det["color_confidence"],
                        "target_type": det.get("target_type"),
                        "is_twin": det.get("is_twin", False),
                        "is_special_sequence": det.get("is_special_sequence", False),
                        "visual_features": det.get("visual_features", {}),
                    },
                )
            )
        return actions

    def on_click_sent(self, action: GameAction) -> None:
        pass

    def reset(self) -> None:
        pass

    def _detect_color_components(
        self,
        hsv: np.ndarray,
        color_name: str,
    ) -> list[TargetDetection]:
        mask = self._classifier.make_color_mask(hsv, color_name)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=1)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        detections: list[TargetDetection] = []
        color = BalloonColor(color_name)
        frame_height = float(hsv.shape[0])

        for label in range(1, count):
            x, y, w, h, area = [int(v) for v in stats[label]]
            if area < self._settings.detection.min_area or area > self._settings.detection.max_area:
                continue
            if w <= 0 or h <= 0:
                continue

            component_mask = np.zeros(mask.shape, dtype=np.uint8)
            component_mask[labels == label] = 255
            contour_list, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contour_list:
                continue
            contour = max(contour_list, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            contour_area = cv2.contourArea(contour)
            circularity = (
                4.0 * np.pi * contour_area / (perimeter * perimeter)
                if perimeter > 0
                else 0.0
            )
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = contour_area / hull_area if hull_area > 0 else 0.0

            white_core_ratio = self._white_core_ratio(hsv, (x, y, w, h))
            diameter_ratio = max(w, h) / frame_height if frame_height else 0.0
            if (
                white_core_ratio >= self._WHITE_CORE_RATIO
                and diameter_ratio >= self._WHITE_MIN_DIAMETER_RATIO
                and circularity >= 0.08
                and solidity >= 0.20
            ):
                detections.append(
                    TargetDetection(
                        target_type=TargetType.INVALID_WHITE,
                        color=None,
                        center=(x + w // 2, y + h // 2),
                        bbox=(x, y, w, h),
                        area=area,
                        confidence=1.0,
                        width=float(w),
                        height=float(h),
                        visual_features=VisualFeatures(
                            circularity=float(circularity),
                            solidity=float(solidity),
                        ),
                    )
                )
                continue

            fill_ratio = float(area) / float(w * h)
            if not (
                self._CORE_FILL_MIN <= fill_ratio <= self._CORE_FILL_MAX
                and circularity >= self._CORE_MIN_CIRCULARITY
            ):
                # Thin colored rings, solid UI/background components, and the
                # multicolor star are not balloon cores and are handled by
                # their own visual classification. The shape test is
                # coordinate-independent so legitimate edge-clipped balloons
                # remain eligible.
                continue

            color_name_detected, confidence = self._classifier.classify_mask(
                hsv, component_mask
            )
            if color_name_detected != color_name or confidence < self._settings.detection.min_confidence:
                continue

            visual = self._visual_features(hsv, component_mask, (x, y, w, h))
            detections.append(
                TargetDetection(
                    target_type=TargetType.NORMAL_BALLOON,
                    color=color,
                    center=(x + w // 2, y + h // 2),
                    bbox=(x, y, w, h),
                    area=area,
                    confidence=confidence,
                    width=float(w),
                    height=float(h),
                    visual_features=visual,
                )
            )

        return detections

    def _detect_multicolor_special(self, hsv: np.ndarray) -> list[TargetDetection]:
        """Detect the blue-dominant four-point multicolor symbol from the video."""
        colorful = cv2.inRange(
            hsv,
            np.array([0, 60, 80], dtype=np.uint8),
            np.array([180, 255, 255], dtype=np.uint8),
        )
        colorful = cv2.morphologyEx(colorful, cv2.MORPH_OPEN, self._kernel, iterations=1)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(colorful, connectivity=8)
        detections: list[TargetDetection] = []

        for label in range(1, count):
            x, y, w, h, area = [int(v) for v in stats[label]]
            if area < self._settings.detection.min_area or area > self._settings.detection.max_area:
                continue
            if w <= 0 or h <= 0:
                continue

            fill_ratio = float(area) / float(w * h)
            if not (0.30 <= fill_ratio <= 0.65):
                continue
            aspect = float(w) / float(h)
            if not (0.70 <= aspect <= 1.30):
                continue

            component_mask = np.zeros(colorful.shape, dtype=np.uint8)
            component_mask[labels == label] = 255
            contour_list, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contour_list:
                continue
            contour = max(contour_list, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            contour_area = cv2.contourArea(contour)
            circularity = (
                4.0 * np.pi * contour_area / (perimeter * perimeter)
                if perimeter > 0
                else 0.0
            )
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = contour_area / hull_area if hull_area > 0 else 0.0
            if not (0.20 <= circularity <= 0.60 and 0.45 <= solidity <= 0.85):
                continue

            multicolor_score = self._multicolor_score(hsv[y:y + h, x:x + w])
            if multicolor_score < 0.12:
                continue

            detections.append(
                TargetDetection(
                    target_type=TargetType.MULTICOLOR_SPECIAL,
                    color=BalloonColor.BLUE,
                    center=(x + w // 2, y + h // 2),
                    bbox=(x, y, w, h),
                    area=area,
                    confidence=min(1.0, multicolor_score + 0.25),
                    width=float(w),
                    height=float(h),
                    visual_features=VisualFeatures(
                        circularity=float(circularity),
                        solidity=float(solidity),
                        multicolor_score=float(multicolor_score),
                    ),
                )
            )

        return detections

    def _visual_features(
        self,
        hsv: np.ndarray,
        component_mask: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> VisualFeatures:
        contour_list, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        circularity = 0.0
        solidity = 0.0
        if contour_list:
            contour = max(contour_list, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter else 0.0
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / hull_area if hull_area else 0.0

        outline = self._outline.detect(hsv, bbox)
        return VisualFeatures(
            circularity=float(circularity),
            solidity=float(solidity),
            white_border_ratio=float(outline.white_ratio),
            outline_contrast=float(outline.contrast),
        )

    @staticmethod
    def _white_core_ratio(
        hsv: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> float:
        x, y, w, h = bbox
        region = hsv[y:y + h, x:x + w]
        if region.size == 0:
            return 0.0

        yy, xx = np.ogrid[:h, :w]
        cx = w / 2.0
        cy = h / 2.0
        rx = max(1.0, w * 0.30)
        ry = max(1.0, h * 0.30)
        ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        white = (region[:, :, 1] <= 35) & (region[:, :, 2] >= 180)
        return float(np.mean(white[ellipse])) if np.any(ellipse) else 0.0

    @staticmethod
    def _multicolor_score(hsv_region: np.ndarray) -> float:
        if hsv_region.size == 0:
            return 0.0
        ranges = (
            (0, 10),
            (15, 35),
            (35, 85),
            (95, 135),
        )
        area = float(hsv_region.shape[0] * hsv_region.shape[1])
        fractions = []
        for lower_h, upper_h in ranges:
            mask = cv2.inRange(
                hsv_region,
                np.array([lower_h, 30, 60], dtype=np.uint8),
                np.array([upper_h, 255, 255], dtype=np.uint8),
            )
            fractions.append(cv2.countNonZero(mask) / area)
        strong_colors = sum(f >= 0.01 for f in fractions)
        return float(sum(fractions) * (strong_colors / 4.0))
