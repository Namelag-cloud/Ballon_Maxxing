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
    # A true balloon has a fairly uniform radial boundary. This catches
    # rounded/irregular UI shapes that can otherwise pass bbox circularity.
    _MAX_BOUNDARY_RADIAL_CV = 0.10
    # The score counter is static in the game UI. Keep this as a very small
    # geometry+position guard for the known static score-counter region.
    # A real balloon is roughly square; the observed score glyph region is
    # materially taller/narrower.
    _SCORE_UI_REGION = (960, 220, 1120, 370)
    _SCORE_UI_MAX_WIDTH = 90
    _SCORE_UI_MIN_HEIGHT = 70
    _SCORE_UI_MAX_ASPECT = 0.85
    _WHITE_CORE_RATIO = 0.50
    _WHITE_MIN_DIAMETER_RATIO = 0.05
    _TWIN_WHITE_BORDER_RATIO = 0.10
    _SPECIAL_RELATIVE_DIAMETER = 0.82
    _TWIN_RING_INNER_SCALE = 1.05
    _TWIN_RING_OUTER_SCALE = 2.00
    _TWIN_RING_MIN_COLOR_RATIO = 0.025
    _TWIN_RING_MIN_SECTOR_COVERAGE = 0.18
    _TWIN_TEXT_MIN_SCORE = 0.08
    _NOTIFICATION_HALO_MIN_RATIO = 0.015
    _NOTIFICATION_HALO_MIN_SECTOR_COVERAGE = 0.16

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
        self._super_group_last_seen: dict[str, int] = {}
        self._super_group_age = 0
        self._super_g_terminal = False
        self._duo_pairs: list[dict] = []
        self._last_hsv_frame: np.ndarray | None = None
        self._color_component_hints: dict[str, list[tuple[tuple[int, int, int, int], int, tuple[float, float]]]] = {}

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
            "score_ui_rejections": 0,
            "notification_ms": 0.0,
            "notification_count": 0,
            "notification_halo_candidates": 0,
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
        self._last_hsv_frame = None
        self._color_component_hints = {color: [] for color in ("blue", "red", "yellow", "green")}

        hsv_t0 = time.perf_counter()
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        self._last_hsv_frame = hsv
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
        self._last_profile["notification_halo_candidates"] = notification_count

        multicolor_t0 = time.perf_counter()
        candidates.extend(
            self._detect_multicolor_special(
                hsv,
                notification_count=notification_count,
            )
        )
        self._last_profile["multicolor_ms"] = (time.perf_counter() - multicolor_t0) * 1000.0

        g_t0 = time.perf_counter()
        g_detections = self._g_detector.detect(
            hsv,
            component_hints=self._color_component_hints,
        )
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

        self._maintain_super_group(classified)
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
        self._super_group_last_seen = {}
        self._super_group_age = 0
        self._super_g_terminal = False
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

            self._color_component_hints[color_name].append(
                ((x, y, w, h), area, (x + w / 2.0, y + h / 2.0))
            )

            # Cheap shape gates from connected-component statistics run before
            # any contour/mask allocation. OpenCV already gives us bbox + area
            # from connectedComponentsWithStats, so reject obviously non-balloon
            # components without paying for contour analysis. These gates apply
            # equally to small and large balloons.
            fill_ratio = float(area) / float(w * h)
            aspect_ratio = float(w) / float(h)
            min_side = float(min(w, h))
            equivalent_diameter = float(2.0 * np.sqrt(area / np.pi))
            bbox_circle_fill = (
                float(area) / (np.pi * (min_side / 2.0) ** 2)
                if min_side > 0.0
                else 0.0
            )
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
                    and fill_ratio <= 0.80
                )
            )
            if self._is_score_counter_candidate(
                x=x,
                y=y,
                width=w,
                height=h,
                aspect_ratio=aspect_ratio,
            ):
                self._last_profile["score_ui_rejections"] += 1
                continue
            if not (
                self._CORE_FILL_MIN <= fill_ratio <= self._CORE_FILL_MAX
                and aspect_ok
                and (0.70 <= bbox_circle_fill <= 1.35 or touches_edge)
            ):
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

            boundary_radial_cv = self._boundary_radial_cv(contour)

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

            if circularity < self._CORE_MIN_CIRCULARITY:
                continue

            # Frame-clipped balloons necessarily have a less uniform radial
            # boundary because the missing arc is replaced by the frame edge.
            # Keep the strict radial gate for normal components, but allow the
            # empirically observed clipped-circle geometry when the component
            # still has circular contour evidence and non-solid fill.
            if boundary_radial_cv > self._MAX_BOUNDARY_RADIAL_CV:
                edge_clipped_circle = (
                    touches_edge
                    and fill_ratio <= 0.85
                    and circularity >= 0.55
                    and boundary_radial_cv <= 0.22
                )
                if not edge_clipped_circle:
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
                    "boundary_radial_cv": float(boundary_radial_cv),
                    "hole_ratio": float(hole_ratio),
                    "equivalent_diameter": equivalent_diameter,
                    "circle_bbox_fill": bbox_circle_fill,
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
                    metadata={
                    },
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

            halo_score, halo_sector_coverage = self._notification_halo_score(hsv, (x, y, w, h))
            if (
                halo_score < self._NOTIFICATION_HALO_MIN_RATIO
                or halo_sector_coverage < self._NOTIFICATION_HALO_MIN_SECTOR_COVERAGE
            ):
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
                        "notification_halo_score": halo_score,
                        "notification_halo_sector_coverage": halo_sector_coverage,
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
        self._last_profile["semantic"]["gemini_candidates"] = len(detections)
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
        if self._super_group is not None or self._super_g_terminal:
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
        self._super_group_last_seen = {
            detection.color.value: self._semantic_frame
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
            self._super_group_last_seen[detection.color.value] = self._semantic_frame
            return True
        return False

    def _maintain_super_group(self, detections: list[TargetDetection]) -> None:
        if self._super_group is None:
            return
        visible_colors = {
            d.color.value
            for d in detections
            if d.target_type == TargetType.SPECIAL_SEQUENCE_BALLOON
            and d.metadata.get("semantic_group") == "super_g"
            and d.color is not None
        }
        if visible_colors:
            self._super_group_age = 0
            return

        self._super_group_age += 1
        if self._super_group_age < 8:
            return

        # Super-G is scheduled once per game. Once its four nodes have all been
        # absent for a sustained interval, never manufacture a second group.
        self._super_group = None
        self._super_g_terminal = True

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
                if self._last_hsv_frame is None:
                    # Unit/source-grounded semantic callers may provide only
                    # detections. Preserve the source rule (two same-color,
                    # sufficiently separated small members) without inventing
                    # pixel evidence that is unavailable.
                    first_ring_score = first_ring_coverage = 0.0
                    second_ring_score = second_ring_coverage = 0.0
                    first_text_score = second_text_score = 0.0
                    visual_evidence_available = False
                else:
                    first_ring_score, first_ring_coverage = self._matching_outline_score(
                        self._last_hsv_frame, first.bbox, first.color.value
                    )
                    second_ring_score, second_ring_coverage = self._matching_outline_score(
                        self._last_hsv_frame, second.bbox, second.color.value
                    )
                    first_text_score = self._twin_text_likeness(
                        self._last_hsv_frame, first.bbox
                    )
                    second_text_score = self._twin_text_likeness(
                        self._last_hsv_frame, second.bbox
                    )
                    visual_evidence_available = True

                first_ring_ok = (
                    first_ring_score >= self._TWIN_RING_MIN_COLOR_RATIO
                    and first_ring_coverage >= self._TWIN_RING_MIN_SECTOR_COVERAGE
                )
                second_ring_ok = (
                    second_ring_score >= self._TWIN_RING_MIN_COLOR_RATIO
                    and second_ring_coverage >= self._TWIN_RING_MIN_SECTOR_COVERAGE
                )
                first_text_ok = first_text_score >= self._TWIN_TEXT_MIN_SCORE
                second_text_ok = second_text_score >= self._TWIN_TEXT_MIN_SCORE
                if visual_evidence_available and not (
                    (first_ring_ok or first_text_ok)
                    and (second_ring_ok or second_text_ok)
                ):
                    continue

                first = replace(
                    first,
                    metadata={
                        **first.metadata,
                        "twin_ring_color_ratio": first_ring_score,
                        "twin_ring_sector_coverage": first_ring_coverage,
                        "twin_text_score": first_text_score,
                    },
                )
                second = replace(
                    second,
                    metadata={
                        **second.metadata,
                        "twin_ring_color_ratio": second_ring_score,
                        "twin_ring_sector_coverage": second_ring_coverage,
                        "twin_text_score": second_text_score,
                    },
                )
                pairs.append((first, second))
                used.update({i, j})
                break

        pair_ids: dict[tuple[float, float], str] = {}
        pair_visual_metadata: dict[tuple[float, float], dict] = {}
        for first, second in pairs:
            ordered = sorted((first.center, second.center))
            pair_id = (
                f"golden_duo:{ordered[0][0]}:{ordered[0][1]}:"
                f"{ordered[1][0]}:{ordered[1][1]}"
            )
            first_key = tuple(map(float, first.center))
            second_key = tuple(map(float, second.center))
            pair_ids[first_key] = pair_id
            pair_ids[second_key] = pair_id
            pair_visual_metadata[first_key] = dict(first.metadata)
            pair_visual_metadata[second_key] = dict(second.metadata)

        self._duo_pairs = [
            {
                "pair_id": pair_id,
                "last_frame": self._semantic_frame,
            }
            for pair_id in set(pair_ids.values())
        ]

        result: list[TargetDetection] = []
        for detection in detections:
            pair_id = pair_ids.get(tuple(map(float, detection.center)))
            if pair_id is None:
                result.append(detection)
                continue
            visual_metadata = pair_visual_metadata.get(
                tuple(map(float, detection.center)),
                {},
            )
            result.append(
                replace(
                    detection,
                    target_type=TargetType.TWIN_BALLOON,
                    is_twin=True,
                    metadata={
                        **detection.metadata,
                        **visual_metadata,
                        "semantic_class": "golden_duo",
                        "semantic_pair_id": pair_id,
                        "semantic_confirmed": True,
                    },
                )
            )
        return result

    @classmethod
    def _annulus_mask(
        cls,
        shape: tuple[int, int],
        bbox: tuple[int, int, int, int],
        inner_scale: float,
        outer_scale: float,
    ) -> np.ndarray:
        height, width = shape
        x, y, w, h = bbox
        yy, xx = np.ogrid[:height, :width]
        cx = x + w / 2.0
        cy = y + h / 2.0
        rx = max(1.0, w / 2.0)
        ry = max(1.0, h / 2.0)
        radial = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        return (radial >= inner_scale) & (radial <= outer_scale)

    def _matching_outline_score(
        self,
        hsv: np.ndarray | None,
        bbox: tuple[int, int, int, int],
        color_name: str,
    ) -> tuple[float, float]:
        if hsv is None or color_name not in self._settings.colors:
            return 0.0, 0.0
        color_mask = self._classifier.make_color_mask(hsv, color_name)
        annulus = self._annulus_mask(
            hsv.shape[:2], bbox,
            self._TWIN_RING_INNER_SCALE,
            self._TWIN_RING_OUTER_SCALE,
        )
        pixels = color_mask[annulus] > 0
        if pixels.size == 0:
            return 0.0, 0.0
        ratio = float(np.mean(pixels))
        height, width = hsv.shape[:2]
        yy, xx = np.indices((height, width))
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        angles = (np.arctan2(yy - cy, xx - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        sector_ids = np.floor(angles / (2.0 * np.pi / 8.0)).astype(np.int32)
        hits = []
        for sector in range(8):
            sector_mask = annulus & (sector_ids == sector)
            if np.any(sector_mask):
                hits.append(float(np.mean(color_mask[sector_mask] > 0)))
        coverage = float(
            sum(v >= self._TWIN_RING_MIN_COLOR_RATIO for v in hits)
        ) / max(1, len(hits))
        return ratio, coverage

    @staticmethod
    def _twin_text_likeness(
        hsv: np.ndarray | None,
        bbox: tuple[int, int, int, int],
    ) -> float:
        if hsv is None:
            return 0.0
        x, y, w, h = bbox
        if w < 8 or h < 8:
            return 0.0
        region = hsv[y:y + h, x:x + w]
        if region.size == 0:
            return 0.0
        s = region[:, :, 1].astype(np.int16)
        v = region[:, :, 2].astype(np.int16)
        mask = ((s <= 65) & ((v <= 95) | (v >= 205))).astype(np.uint8) * 255
        yy, xx = np.ogrid[:h, :w]
        cx, cy = w / 2.0, h / 2.0
        inner = ((xx - cx) / max(1.0, w * 0.42)) ** 2 + ((yy - cy) / max(1.0, h * 0.42)) ** 2 <= 1.0
        mask[~inner] = 0
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1)),
            iterations=1,
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        components = []
        for label in range(1, count):
            cw, ch, area = [int(v) for v in stats[label, 2:5]]
            if 1 <= area <= max(4, int(w * h * 0.08)) and cw <= max(4, w // 2) and ch <= max(4, h // 2):
                components.append(area)
        if not components:
            return 0.0
        area_score = sum(components) / float(max(1, w * h))
        count_score = min(1.0, len(components) / 8.0)
        return float(min(1.0, area_score * 8.0 + count_score * 0.5))

    @classmethod
    def _notification_halo_score(
        cls,
        hsv: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[float, float]:
        red_mask = cv2.inRange(
            hsv,
            np.array([0, 70, 80], dtype=np.uint8),
            np.array([12, 255, 255], dtype=np.uint8),
        )
        annulus = cls._annulus_mask(hsv.shape[:2], bbox, 1.02, 1.85)
        pixels = red_mask[annulus] > 0
        if pixels.size == 0:
            return 0.0, 0.0
        ratio = float(np.mean(pixels))
        height, width = hsv.shape[:2]
        yy, xx = np.indices((height, width))
        x, y, w, h = bbox
        cx, cy = x + w / 2.0, y + h / 2.0
        angles = (np.arctan2(yy - cy, xx - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        sector_ids = np.floor(angles / (2.0 * np.pi / 8.0)).astype(np.int32)
        hits = []
        for sector in range(8):
            sector_mask = annulus & (sector_ids == sector)
            if np.any(sector_mask):
                hits.append(float(np.mean(red_mask[sector_mask] > 0)))
        coverage = float(
            sum(v >= cls._NOTIFICATION_HALO_MIN_RATIO for v in hits)
        ) / max(1, len(hits))
        return ratio, coverage

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
    def _boundary_radial_cv(contour: np.ndarray) -> float:
        """Return radial-boundary coefficient of variation around the contour centroid."""
        if contour is None or len(contour) < 8:
            return float("inf")
        moments = cv2.moments(contour)
        if moments["m00"] <= 0.0:
            return float("inf")
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        points = contour.reshape(-1, 2).astype(np.float32)
        radii = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
        mean_radius = float(np.mean(radii))
        if mean_radius <= 0.0:
            return float("inf")
        return float(np.std(radii) / mean_radius)

    @classmethod
    def _is_score_counter_candidate(
        cls,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
        aspect_ratio: float,
    ) -> bool:
        """Reject the known static score-counter glyph geometry cheaply.

        This is deliberately conditional: only a narrow/tall component inside
        the fixed UI area is rejected. A roughly circular balloon in the same
        region remains eligible, including a balloon that spawns over the UI.
        """
        region_x0, region_y0, region_x1, region_y1 = cls._SCORE_UI_REGION
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        inside_region = (
            region_x0 <= center_x <= region_x1
            and region_y0 <= center_y <= region_y1
        )
        return (
            inside_region
            and width <= cls._SCORE_UI_MAX_WIDTH
            and height >= cls._SCORE_UI_MIN_HEIGHT
            and aspect_ratio <= cls._SCORE_UI_MAX_ASPECT
        )

    @staticmethod
    def _bbox_iou(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        """Return intersection-over-union for two x/y/width/height boxes."""
        x1, y1, w1, h1 = first
        x2, y2, w2, h2 = second
        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return 0.0

        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        intersection_width = max(0, right - left)
        intersection_height = max(0, bottom - top)
        intersection = intersection_width * intersection_height
        if intersection == 0:
            return 0.0

        union = (w1 * h1) + (w2 * h2) - intersection
        return float(intersection) / float(union) if union > 0 else 0.0

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