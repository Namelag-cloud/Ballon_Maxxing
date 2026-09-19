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
    _CORE_MIN_ASPECT_RATIO = 0.75
    _CORE_MAX_ASPECT_RATIO = 1.30
    # A balloon core should occupy most, but not essentially all, of its
    # bounding box. This rejects solid UI bars/panels without using screen
    # coordinates, while remaining tolerant of balloons clipped by a frame edge.
    _CORE_FILL_MAX = 0.90
    _CORE_MIN_CIRCULARITY = 0.45
    _MAX_ENCLOSED_HOLE_RATIO = 0.20
    _WHITE_CORE_RATIO = 0.50
    _WHITE_MIN_DIAMETER_RATIO = 0.05
    _TWIN_WHITE_BORDER_RATIO = 0.10
    _SPECIAL_RELATIVE_DIAMETER = 0.82

    # Ground-truth engine invariants. These are deliberately kept separate from
    # pixel thresholds so future renderer-specific calibration does not weaken
    # the semantic contract.
    _SUPER_G_COLORS = (
        BalloonColor.BLUE,
        BalloonColor.RED,
        BalloonColor.YELLOW,
        BalloonColor.GREEN,
    )
    _SUPER_G_MATCH_DISTANCE_PX = 32.0
    _SUPER_G_CONFIRM_FRAMES = 1
    _GOLDEN_DUO_MIN_DISTANCE_PX = 50.0
    _GOLDEN_DUO_MAX_DIAMETER_PX = 66.0
    _GOLDEN_DUO_MAX_CENTER_DRIFT_PX = 32.0
    _GOLDEN_DUO_MAX_MISSING_FRAMES = 5
    _GEMINI_MIN_DIAMETER_PX = 60.0
    _GEMINI_MAX_DIAMETER_PX = 150.0

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
        self._last_profile = self._empty_profile()
        self._semantic_frame = 0
        self._candidate_history: list[dict] = []
        self._super_group: dict[str, tuple[float, float]] | None = None
        self._super_group_age = 0
        self._duo_pairs: list[dict] = []

    @staticmethod
    def _empty_profile() -> dict:
        return {
            "total_ms": 0.0,
            "hsv_ms": 0.0,
            "blue_ms": 0.0,
            "red_ms": 0.0,
            "yellow_ms": 0.0,
            "green_ms": 0.0,
            "blue_mask_ms": 0.0,
            "red_mask_ms": 0.0,
            "yellow_mask_ms": 0.0,
            "green_mask_ms": 0.0,
            "blue_morphology_ms": 0.0,
            "red_morphology_ms": 0.0,
            "yellow_morphology_ms": 0.0,
            "green_morphology_ms": 0.0,
            "blue_components_ms": 0.0,
            "red_components_ms": 0.0,
            "yellow_components_ms": 0.0,
            "green_components_ms": 0.0,
            "blue_component_analysis_ms": 0.0,
            "blue_component_mask_allocation_ms": 0.0,
            "blue_component_mask_label_ms": 0.0,
            "blue_component_classification_ms": 0.0,
            "red_component_analysis_ms": 0.0,
            "red_component_mask_allocation_ms": 0.0,
            "red_component_mask_label_ms": 0.0,
            "red_component_classification_ms": 0.0,
            "yellow_component_analysis_ms": 0.0,
            "yellow_component_mask_allocation_ms": 0.0,
            "yellow_component_mask_label_ms": 0.0,
            "yellow_component_classification_ms": 0.0,
            "green_component_analysis_ms": 0.0,
            "green_component_mask_allocation_ms": 0.0,
            "green_component_mask_label_ms": 0.0,
            "green_component_classification_ms": 0.0,
            "multicolor_ms": 0.0,
            "g_detector_ms": 0.0,
            "ring_total_ms": 0.0,
            "ring_hough_ms": 0.0,
            "ring_grayscale_ms": 0.0,
            "ring_blur_ms": 0.0,
            "ring_canny_ms": 0.0,
            "ring_scoring_ms": 0.0,
            "classification_ms": 0.0,
            "normal_candidate_count": 0,
            "ring_call_count": 0,
            "hough_call_count": 0,
            "component_count": 0,
            "candidate_counts": {"blue": 0, "red": 0, "yellow": 0, "green": 0},
            "candidate_geometry": [],
            "ring_circles_returned": 0,
            "notification_ms": 0.0,
            "notification_count": 0,
            "semantic": {
                "super_g_confirmed": False,
                "super_g_member_ids": [],
                "golden_duo_pairs": [],
                "gemini_candidates": 0,
                "ring_candidates_considered": 0,
            },
        }

    @property
    def last_profile(self) -> dict:
        profile = dict(self._last_profile)
        profile["candidate_counts"] = dict(self._last_profile["candidate_counts"])
        return profile

    @property
    def name(self) -> str:
        return "Balloon Game"

    @property
    def game_type(self) -> GameType:
        return GameType.BALLOON_GAME

    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        """Return structured detections serialized for the existing profile API."""
        total_t0 = time.perf_counter()
        self._last_profile = self._empty_profile()

        hsv_t0 = time.perf_counter()
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        self._last_profile["hsv_ms"] = (time.perf_counter() - hsv_t0) * 1000.0

        candidates: list[TargetDetection] = []

        for color_name in ("blue", "red", "yellow", "green"):
            candidates.extend(
                self._detect_color_components(hsv, color_name)
            )

        notification_t0 = time.perf_counter()
        notifications = self._detect_notifications(hsv)
        candidates.extend(notifications)
        notification_count = len(notifications)
        self._last_profile["notification_ms"] = (
            time.perf_counter() - notification_t0
        ) * 1000.0
        self._last_profile["notification_count"] = notification_count

        multicolor_t0 = time.perf_counter()
        candidates.extend(
            self._detect_multicolor_special(
                hsv,
                notification_count=notification_count,
            )
        )
        self._last_profile["multicolor_ms"] = (time.perf_counter() - multicolor_t0) * 1000.0

        g_t0 = time.perf_counter()
        g_detections = self._g_detector.detect(hsv)
        self._last_profile["g_detector_ms"] = (time.perf_counter() - g_t0) * 1000.0

        # G is a more specific multicolor geometry class. Do not let the generic
        # Gemini candidate shadow the same image region when both detectors fire.
        g_boxes = [d.bbox for d in g_detections]
        candidates = [
            d for d in candidates
            if d.target_type != TargetType.MULTICOLOR_SPECIAL
            or not any(self._bbox_iou(d.bbox, box) > 0.40 for box in g_boxes)
        ]
        candidates.extend(g_detections)

        classification_t0 = time.perf_counter()
        normal_candidates = [
            d for d in candidates
            if d.target_type == TargetType.NORMAL_BALLOON and d.color is not None
        ]
        self._last_profile["normal_candidate_count"] = len(normal_candidates)

        # Expensive ring/Hough work is now reserved for a very small set of
        # newly-arrived, one-per-color candidates that could form the game's
        # single four-node Super-G. Ordinary balloons no longer pay this cost.
        super_group = self._detect_super_g_group(normal_candidates, bgr_frame)
        if super_group is not None:
            self._last_profile["semantic"]["super_g_confirmed"] = True
            self._last_profile["semantic"]["super_g_member_ids"] = [
                d.center for d in super_group
            ]
            confirmed_by_center = {d.center: d for d in super_group}
            normal_candidates = [
                confirmed_by_center.get(d.center, d)
                for d in normal_candidates
            ]

        classified: list[TargetDetection] = []
        for detection in candidates:
            if detection.target_type != TargetType.NORMAL_BALLOON or detection.color is None:
                classified.append(detection)
                continue

            metadata = dict(detection.metadata)
            confirmed = next(
                (d for d in super_group or [] if d.center == detection.center),
                None,
            )
            if confirmed is not None or self._super_group_matches(detection):
                if confirmed is not None:
                    detection = confirmed
                detection = replace(
                    detection,
                    target_type=TargetType.SPECIAL_SEQUENCE_BALLOON,
                    is_special_sequence=True,
                    metadata={
                        **metadata,
                        "semantic_class": "super_g_node",
                        "semantic_group": "super_g",
                        "semantic_confirmed": True,
                    },
                )
            classified.append(detection)

        # Golden Duo is a two-object semantic class. A single yellow/green
        # balloon can never become a twin solely because it has an outline.
        classified = self._classify_golden_duos(classified, bgr_frame)

        # Gemini is one larger multicolor object. The engine only permits one
        # active Gemini and requires at least two notifications when it spawns;
        # the visual detector already applies the notification constraint.
        self._last_profile["semantic"]["golden_duo_pairs"] = [
            pair["pair_id"] for pair in self._duo_pairs
        ]

        self._update_candidate_history(classified)
        assembly_t0 = time.perf_counter()
        classified.sort(key=lambda d: (d.center[1], d.center[0], d.target_type.value))
        result = [d.as_dict() for d in classified]
        self._last_profile["classification_ms"] += (time.perf_counter() - assembly_t0) * 1000.0
        self._last_profile["total_ms"] = (time.perf_counter() - total_t0) * 1000.0
        return result

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
            if det.get("target_type") in {
                TargetType.INVALID_WHITE.value,
                TargetType.NOTIFICATION.value,
            }:
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
        self._semantic_frame = 0
        self._candidate_history = []
        self._super_group = None
        self._super_group_age = 0
        self._duo_pairs = []

    def _detect_color_components(
        self,
        hsv: np.ndarray,
        color_name: str,
    ) -> list[TargetDetection]:
        total_t0 = time.perf_counter()
        mask_t0 = time.perf_counter()
        mask = self._classifier.make_color_mask(hsv, color_name)
        mask_ms = (time.perf_counter() - mask_t0) * 1000.0

        morphology_t0 = time.perf_counter()
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=1)
        morphology_ms = (time.perf_counter() - morphology_t0) * 1000.0

        components_t0 = time.perf_counter()
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        components_ms = (time.perf_counter() - components_t0) * 1000.0

        detections: list[TargetDetection] = []
        color = BalloonColor(color_name)
        mask_alloc_ms_total = 0.0
        mask_label_ms_total = 0.0
        classification_ms_total = 0.0
        frame_height = float(hsv.shape[0])

        component_analysis_t0 = time.perf_counter()
        self._last_profile["component_count"] += max(0, count - 1)

        for label in range(1, count):
            x, y, w, h, area = [int(v) for v in stats[label]]
            if area < self._settings.detection.min_area or area > self._settings.detection.max_area:
                continue
            if w <= 0 or h <= 0:
                continue

            mask_alloc_t0 = time.perf_counter()
            component_mask = np.zeros((h, w), dtype=np.uint8)
            mask_alloc_ms = (time.perf_counter() - mask_alloc_t0) * 1000.0
            mask_label_t0 = time.perf_counter()
            component_mask[labels[y:y + h, x:x + w] == label] = 255
            mask_label_ms = (time.perf_counter() - mask_label_t0) * 1000.0
            mask_alloc_ms_total += mask_alloc_ms
            mask_label_ms_total += mask_label_ms
            contour_list, hierarchy = cv2.findContours(
                component_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
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

            enclosed_hole_area = 0.0
            if hierarchy is not None:
                for contour_index, contour_info in enumerate(hierarchy[0]):
                    if contour_info[3] >= 0:
                        enclosed_hole_area += cv2.contourArea(contour_list[contour_index])
            hole_ratio = enclosed_hole_area / float(w * h) if w * h else 0.0
            if hole_ratio > self._MAX_ENCLOSED_HOLE_RATIO:
                continue

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
            aspect_ratio = float(w) / float(h)
            touches_edge = (
                x == 0
                or y == 0
                or x + w >= hsv.shape[1]
                or y + h >= hsv.shape[0]
            )
            aspect_ok = (
                self._CORE_MIN_ASPECT_RATIO <= aspect_ratio <= self._CORE_MAX_ASPECT_RATIO
                or (
                    touches_edge
                    and 0.45 <= aspect_ratio <= 2.0
                    and circularity >= 0.55
                    and fill_ratio <= 0.80
                )
            )
            if not (
                self._CORE_FILL_MIN <= fill_ratio <= self._CORE_FILL_MAX
                and circularity >= self._CORE_MIN_CIRCULARITY
                and aspect_ok
            ):
                continue

            # Keep a small geometry sample for diagnosing false positives. This
            # records only components that reach the normal-balloon candidate
            # boundary, so it does not materially expand the hot path.
            if len(self._last_profile["candidate_geometry"]) < 32:
                self._last_profile["candidate_geometry"].append({
                    "color": color_name,
                    "bbox": (x, y, w, h),
                    "width": w,
                    "height": h,
                    "aspect_ratio": aspect_ratio,
                    "area": area,
                    "fill_ratio": float(fill_ratio),
                    "circularity": float(circularity),
                    "solidity": float(solidity),
                    "hole_ratio": float(hole_ratio),
                })

            classification_t0 = time.perf_counter()
            color_name_detected, confidence = self._classifier.classify_mask(
                hsv[y:y + h, x:x + w], component_mask
            )
            classification_ms = (time.perf_counter() - classification_t0) * 1000.0
            classification_ms_total += classification_ms
            if color_name_detected != color_name or confidence < self._settings.detection.min_confidence:
                continue

            visual = self._visual_features(
                hsv,
                (x, y, w, h),
                circularity=circularity,
                solidity=solidity,
            )
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

        component_analysis_ms = (time.perf_counter() - component_analysis_t0) * 1000.0
        self._last_profile[f"{color_name}_ms"] = (time.perf_counter() - total_t0) * 1000.0
        self._last_profile[f"{color_name}_mask_ms"] = mask_ms
        self._last_profile[f"{color_name}_morphology_ms"] = morphology_ms
        self._last_profile[f"{color_name}_components_ms"] = components_ms
        self._last_profile[f"{color_name}_component_analysis_ms"] = component_analysis_ms
        self._last_profile[f"{color_name}_component_mask_allocation_ms"] = mask_alloc_ms_total
        self._last_profile[f"{color_name}_component_mask_label_ms"] = mask_label_ms_total
        self._last_profile[f"{color_name}_component_classification_ms"] = classification_ms_total
        self._last_profile["candidate_counts"][color_name] = len(detections)
        return detections

    def _detect_notifications(self, hsv: np.ndarray) -> list[TargetDetection]:
        """Detect source-defined white notification balloons.

        Notifications are pure white game objects, so they are not represented
        by the four saturated color masks used for normal balloons. Keeping them
        as a separate, non-actionable detection also gives Gemini its observable
        spawn precondition without permitting notification clicks.
        """
        if "white" not in self._settings.colors:
            return []

        mask = self._classifier.make_color_mask(hsv, "white")
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        detections: list[TargetDetection] = []

        for label in range(1, count):
            x, y, w, h, area = [int(v) for v in stats[label]]
            if area < self._settings.detection.min_area or area > self._settings.detection.max_area:
                continue
            if w <= 0 or h <= 0:
                continue

            fill_ratio = float(area) / float(w * h)
            aspect = float(w) / float(h)
            if not (0.45 <= fill_ratio <= 0.95 and 0.65 <= aspect <= 1.35):
                continue

            component_mask = np.zeros((h, w), dtype=np.uint8)
            component_mask[labels[y:y + h, x:x + w] == label] = 255
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            contour_area = cv2.contourArea(contour)
            circularity = (
                4.0 * np.pi * contour_area / (perimeter * perimeter)
                if perimeter > 0 else 0.0
            )
            if circularity < 0.45:
                continue

            detections.append(
                TargetDetection(
                    target_type=TargetType.NOTIFICATION,
                    color=None,
                    center=(x + w // 2, y + h // 2),
                    bbox=(x, y, w, h),
                    area=area,
                    confidence=1.0,
                    width=float(w),
                    height=float(h),
                    visual_features=VisualFeatures(
                        circularity=float(circularity),
                        solidity=1.0,
                    ),
                    metadata={
                        "semantic_class": "notification",
                        "semantic_confirmed": True,
                    },
                )
            )
        return detections

    def _detect_multicolor_special(
        self,
        hsv: np.ndarray,
        *,
        notification_count: int = 0,
    ) -> list[TargetDetection]:
        """Detect Gemini using source-grounded size/context constraints.

        The renderer is not available in this chunk, so this method deliberately
        does not invent a renderer-specific pixel signature. It requires the
        engine's observable spawn precondition (>=2 notifications) and returns
        at most one suitably sized multicolor object.
        """
        if notification_count < 2:
            return []
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
            if fill_ratio < 0.25:
                continue
            aspect = float(w) / float(h)
            if not (0.55 <= aspect <= 1.45):
                continue

            diameter = max(w, h)
            if not (self._GEMINI_MIN_DIAMETER_PX <= diameter <= self._GEMINI_MAX_DIAMETER_PX):
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
            if circularity < 0.20 or solidity < 0.25:
                continue

            multicolor_score = self._multicolor_score(hsv[y:y + h, x:x + w])
            if multicolor_score < 0.12:
                continue

            detections.append(
                TargetDetection(
                    target_type=TargetType.MULTICOLOR_SPECIAL,
                    color=None,
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
                    metadata={
                        "semantic_class": "gemini",
                        "semantic_confirmed": True,
                        "notification_count": notification_count,
                    },
                )
            )

        # The engine guarantees at most one active Gemini. If visual segmentation
        # produces several multicolor components, retain only the strongest/largest
        # plausible candidate rather than emitting multiple Geminis.
        detections.sort(
            key=lambda d: (max(d.width, d.height), d.confidence),
            reverse=True,
        )
        return detections[:1]

    def _detect_super_g_group(
        self,
        candidates: list[TargetDetection],
        bgr_frame: np.ndarray,
    ) -> list[TargetDetection] | None:
        """Confirm the engine's one four-node Super-G cohort.

        The engine creates exactly one blue/red/yellow/green node set at once.
        Because spawnTick is not visible to a screen-only bot, a new-per-color
        cohort is the observable temporal proxy. The expensive ring detector is
        then applied only to those four candidates, rather than every ordinary
        balloon.
        """
        if self._super_group is not None:
            return None

        color_values = {c.value for c in self._SUPER_G_COLORS}
        new_by_color: dict[str, TargetDetection] = {}
        new_candidates: list[TargetDetection] = []
        for candidate in candidates:
            if not self._is_new_candidate(candidate):
                continue
            color = candidate.color.value
            if color not in color_values:
                continue
            new_candidates.append(candidate)
            if color in new_by_color:
                continue
            new_by_color[color] = candidate

        if len(new_candidates) != len(color_values) or set(new_by_color) != color_values:
            return None

        confirmed: list[TargetDetection] = []
        for candidate in new_by_color.values():
            self._last_profile["semantic"]["ring_candidates_considered"] += 1
            ring = self._ring.detect(bgr_frame, candidate.bbox)
            self._record_ring_profile(ring)
            if not ring.detected:
                return None

            confirmed.append(
                replace(
                    candidate,
                    visual_features=replace(
                        candidate.visual_features,
                        ring_detected=True,
                        ring_radius_ratio=ring.radius_ratio,
                        ring_confidence=ring.confidence,
                    ),
                )
            )

        self._super_group = {
            detection.color.value: tuple(map(float, detection.center))
            for detection in confirmed
            if detection.color is not None
        }
        self._super_group_age = 0
        return confirmed

    def _record_ring_profile(self, ring) -> None:
        ring_profile = self._ring.last_profile
        self._last_profile["ring_total_ms"] += float(ring_profile["total_ms"])
        self._last_profile["ring_hough_ms"] += float(ring_profile["hough_ms"])
        self._last_profile["ring_grayscale_ms"] += float(ring_profile["grayscale_ms"])
        self._last_profile["ring_blur_ms"] += float(ring_profile["blur_ms"])
        self._last_profile["ring_canny_ms"] += float(ring_profile["canny_ms"])
        self._last_profile["ring_scoring_ms"] += float(ring_profile["scoring_ms"])
        self._last_profile["ring_call_count"] += 1
        self._last_profile["hough_call_count"] += int(ring_profile["hough_called"])
        self._last_profile["ring_circles_returned"] += int(ring_profile["circles_returned"])

    def _super_group_matches(self, detection: TargetDetection) -> bool:
        if self._super_group is None or detection.color is None:
            return False
        previous = self._super_group.get(detection.color.value)
        if previous is None:
            return False
        dx = float(detection.center[0]) - previous[0]
        dy = float(detection.center[1]) - previous[1]
        if (dx * dx + dy * dy) ** 0.5 <= self._SUPER_G_MATCH_DISTANCE_PX:
            self._super_group[detection.color.value] = (
                float(detection.center[0]),
                float(detection.center[1]),
            )
            return True
        return False

    def _is_new_candidate(self, detection: TargetDetection) -> bool:
        if detection.color is None:
            return False
        cx, cy = detection.center
        threshold = max(
            self._SUPER_G_MATCH_DISTANCE_PX,
            max(detection.width, detection.height) * 0.65,
        )
        return not any(
            item["color"] == detection.color.value
            and ((cx - item["center"][0]) ** 2 + (cy - item["center"][1]) ** 2) ** 0.5 <= threshold
            for item in self._candidate_history
        )

    def _classify_golden_duos(
        self,
        detections: list[TargetDetection],
        bgr_frame: np.ndarray | None = None,
    ) -> list[TargetDetection]:
        """Classify Golden Duo only when a complete visual pair is present.

        The engine creates exactly two golden_duo objects, both radius 25 in
        solo mode, and enforces at least 60 game units between their centers.
        The screen-only approximation requires two same-color, small candidates
        separated by the source-grounded minimum and, when the frame is available,
        matching circular-outline evidence. A lone yellow/green balloon can never
        become a twin.
        """
        candidates = [
            d for d in detections
            if d.target_type == TargetType.NORMAL_BALLOON
            and d.color in (BalloonColor.YELLOW, BalloonColor.GREEN)
            and max(d.width, d.height) <= self._GOLDEN_DUO_MAX_DIAMETER_PX
        ]

        pairs: list[tuple[TargetDetection, TargetDetection]] = []
        used: set[int] = set()
        for i, first in enumerate(candidates):
            if i in used:
                continue
            for j in range(i + 1, len(candidates)):
                if j in used:
                    continue
                second = candidates[j]
                if first.color != second.color:
                    continue
                distance = float(
                    np.hypot(
                        first.center[0] - second.center[0],
                        first.center[1] - second.center[1],
                    )
                )
                if distance < self._GOLDEN_DUO_MIN_DISTANCE_PX:
                    continue
                if bgr_frame is not None:
                    first_ring = self._ring.detect(bgr_frame, first.bbox)
                    self._record_ring_profile(first_ring)
                    second_ring = self._ring.detect(bgr_frame, second.bbox)
                    self._record_ring_profile(second_ring)
                    if not (first_ring.detected and second_ring.detected):
                        continue
                pairs.append((first, second))
                used.update({i, j})
                break

        pair_ids: dict[int, str] = {}
        for first, second in pairs:
            ordered = sorted((first.center, second.center))
            pair_id = (
                f"golden_duo:{ordered[0][0]}:{ordered[0][1]}:"
                f"{ordered[1][0]}:{ordered[1][1]}"
            )
            pair_ids[id(first)] = pair_id
            pair_ids[id(second)] = pair_id

        self._duo_pairs = [
            {
                "pair_id": pair_id,
                "last_frame": self._semantic_frame,
            }
            for pair_id in set(pair_ids.values())
        ]

        result: list[TargetDetection] = []
        for detection in detections:
            pair_id = pair_ids.get(id(detection))
            if pair_id is None:
                result.append(detection)
                continue
            result.append(
                replace(
                    detection,
                    target_type=TargetType.TWIN_BALLOON,
                    is_twin=True,
                    metadata={
                        **detection.metadata,
                        "semantic_class": "golden_duo",
                        "semantic_pair_id": pair_id,
                        "semantic_confirmed": True,
                    },
                )
            )
        return result

    def _update_candidate_history(self, detections: list[TargetDetection]) -> None:
        self._semantic_frame += 1
        current = [
            {
                "color": d.color.value,
                "center": tuple(map(float, d.center)),
                "width": float(d.width),
                "height": float(d.height),
            }
            for d in detections
            if d.color is not None and d.target_type in {
                TargetType.NORMAL_BALLOON,
                TargetType.TWIN_BALLOON,
                TargetType.SPECIAL_SEQUENCE_BALLOON,
            }
        ]
        self._candidate_history = current

    def _visual_features(
        self,
        hsv: np.ndarray,
        bbox: tuple[int, int, int, int],
        *,
        circularity: float,
        solidity: float,
    ) -> VisualFeatures:
        # circularity/solidity were already computed during component
        # acceptance. Reuse them instead of running a second contour pass.
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
