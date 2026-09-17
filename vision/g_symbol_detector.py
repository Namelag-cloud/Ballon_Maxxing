from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np

from game_profiles.balloon_types import (
    BalloonColor,
    TargetDetection,
    TargetType,
    VisualFeatures,
)
from vision.color_classifier import ColorClassifier


@dataclass(frozen=True)
class _ColorComponent:
    color: BalloonColor
    bbox: tuple[int, int, int, int]
    area: int
    center: tuple[float, float]


class GSymbolDetector:
    """Detect the game's large Google-style G symbol from visual geometry.

    The detector intentionally does not inspect the click counter rendered next
    to the G. It looks for the symbol's distinctive spatial arrangement of
    saturated red/yellow/green/blue regions: an upper-left red arc, lower-left
    green arc, left-side yellow transition, and a right-side horizontal blue bar.
    The yellow component is allowed to disappear during animation/overlap.
    """

    _COLORS = (
        BalloonColor.RED,
        BalloonColor.YELLOW,
        BalloonColor.GREEN,
        BalloonColor.BLUE,
    )
    _MIN_COMPONENT_AREA = 500
    _MORPH_KERNEL = 7
    _MIN_G_HEIGHT_RATIO = 0.07
    _MAX_G_HEIGHT_RATIO = 0.20
    _MIN_ASPECT = 0.50
    _MAX_ASPECT = 1.10
    _MIN_SCORE = 0.66

    def __init__(self, classifier: ColorClassifier) -> None:
        self._classifier = classifier
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._MORPH_KERNEL, self._MORPH_KERNEL),
        )

    def detect(self, hsv_frame: np.ndarray) -> list[TargetDetection]:
        components = {
            color: self._components_for_color(hsv_frame, color)
            for color in self._COLORS
        }
        candidates: list[TargetDetection] = []

        for red, green, blue in product(
            components[BalloonColor.RED],
            components[BalloonColor.GREEN],
            components[BalloonColor.BLUE],
        ):
            optional_yellows = components[BalloonColor.YELLOW] or [None]
            for yellow in optional_yellows:
                selected = [red, green, blue]
                if yellow is not None and self._near_group(yellow, selected):
                    selected.append(yellow)
                elif yellow is not None:
                    continue

                detection = self._score_candidate(selected, hsv_frame.shape[:2])
                if detection is not None:
                    candidates.append(detection)

        # The same components can participate in multiple optional-yellow
        # combinations. Keep the strongest geometrically distinct candidate.
        candidates.sort(key=lambda d: d.confidence, reverse=True)
        kept: list[TargetDetection] = []
        for candidate in candidates:
            if any(self._bbox_iou(candidate.bbox, other.bbox) > 0.5 for other in kept):
                continue
            kept.append(candidate)
        return kept

    def _components_for_color(
        self,
        hsv_frame: np.ndarray,
        color: BalloonColor,
    ) -> list[_ColorComponent]:
        mask = self._classifier.make_color_mask(hsv_frame, color.value)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        components: list[_ColorComponent] = []
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            if area < self._MIN_COMPONENT_AREA or w <= 0 or h <= 0:
                continue
            components.append(
                _ColorComponent(
                    color=color,
                    bbox=(x, y, w, h),
                    area=area,
                    center=(float(centroids[label][0]), float(centroids[label][1])),
                )
            )
        return components

    @classmethod
    def _near_group(
        cls,
        component: _ColorComponent,
        selected: list[_ColorComponent],
    ) -> bool:
        x, y, w, h = cls._union_bbox(selected)
        cx, cy = component.center
        margin = max(w, h) * 0.45
        return (
            x - margin <= cx <= x + w + margin
            and y - margin <= cy <= y + h + margin
        )

    @classmethod
    def _score_candidate(
        cls,
        selected: list[_ColorComponent],
        frame_shape: tuple[int, int],
    ) -> TargetDetection | None:
        by_color = {component.color: component for component in selected}
        red = by_color[BalloonColor.RED]
        green = by_color[BalloonColor.GREEN]
        blue = by_color[BalloonColor.BLUE]
        yellow = by_color.get(BalloonColor.YELLOW)

        x, y, w, h = cls._union_bbox(selected)
        frame_h, _frame_w = frame_shape
        height_ratio = h / float(frame_h) if frame_h else 0.0
        aspect = w / float(h) if h else 0.0
        if not cls._MIN_G_HEIGHT_RATIO <= height_ratio <= cls._MAX_G_HEIGHT_RATIO:
            return None
        if not cls._MIN_ASPECT <= aspect <= cls._MAX_ASPECT:
            return None

        # The G's red and green arcs form the left spine of the symbol.
        if (
            (red.bbox[0] - x) / max(1.0, w) > 0.20
            or (green.bbox[0] - x) / max(1.0, w) > 0.20
        ):
            return None

        # Red, green, and blue must each be substantial colored regions. The
        # optional yellow transition may be split by anti-aliasing/animation.
        if any(component.area / float(w * h) < 0.025 for component in (red, green, blue)):
            return None

        # Normalize component centers to the candidate box. This captures the
        # G's stable visual layout while remaining independent of absolute size.
        def norm(component: _ColorComponent) -> tuple[float, float]:
            return (
                (component.center[0] - x) / max(1.0, w),
                (component.center[1] - y) / max(1.0, h),
            )

        rx, ry = norm(red)
        gx, gy = norm(green)
        bx, by = norm(blue)

        score_parts = [
            cls._range_score(rx, 0.15, 0.58),
            cls._range_score(ry, 0.00, 0.42),
            cls._range_score(gx, 0.05, 0.62),
            cls._range_score(gy, 0.55, 1.00),
            cls._range_score(bx, 0.40, 1.00),
            cls._range_score(by, 0.28, 0.78),
        ]

        _blue_x, _blue_y, blue_w, blue_h = blue.bbox
        blue_aspect = blue_w / float(blue_h) if blue_h else 0.0
        score_parts.append(cls._range_score(blue_aspect, 1.10, 2.80))

        if yellow is not None:
            yx, yy = norm(yellow)
            score_parts.extend([
                cls._range_score(yx, 0.00, 0.38),
                cls._range_score(yy, 0.28, 0.78),
            ])

        geometry = float(np.mean(score_parts))
        if geometry < cls._MIN_SCORE:
            return None

        confidence = min(0.99, geometry + (0.06 if yellow is not None else 0.0))
        # The blue horizontal bar is the reliable colored click surface, so use
        # its centroid rather than the center of the G's white interior.
        center = (
            int(round(blue.center[0])),
            int(round(blue.center[1])),
        )
        area = sum(component.area for component in selected)
        return TargetDetection(
            target_type=TargetType.G_SYMBOL,
            color=None,
            center=center,
            bbox=(x, y, w, h),
            area=area,
            confidence=confidence,
            width=float(w),
            height=float(h),
            visual_features=VisualFeatures(multicolor_score=geometry),
        )

    @staticmethod
    def _range_score(value: float, low: float, high: float) -> float:
        if low <= value <= high:
            center = (low + high) / 2.0
            half = max(1e-6, (high - low) / 2.0)
            return 1.0 - 0.35 * abs(value - center) / half
        margin = (high - low) * 0.75
        distance = min(abs(value - low), abs(value - high))
        return max(0.0, 1.0 - distance / max(1e-6, margin))

    @staticmethod
    def _union_bbox(components: list[_ColorComponent]) -> tuple[int, int, int, int]:
        x0 = min(c.bbox[0] for c in components)
        y0 = min(c.bbox[1] for c in components)
        x1 = max(c.bbox[0] + c.bbox[2] for c in components)
        y1 = max(c.bbox[1] + c.bbox[3] for c in components)
        return x0, y0, x1 - x0, y1 - y0

    @staticmethod
    def _bbox_iou(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        intersection = (x1 - x0) * (y1 - y0)
        union = aw * ah + bw * bh - intersection
        return intersection / float(union) if union else 0.0
