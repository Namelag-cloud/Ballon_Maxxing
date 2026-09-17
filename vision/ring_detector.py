from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RingResult:
    detected: bool
    radius_px: float = 0.0
    radius_ratio: float = 0.0
    confidence: float = 0.0


class RingDetector:
    """Detect the large circular outline used by special balloons.

    The feature is expressed relative to the detected balloon radius rather than
    as an absolute screen-space size. Hough candidates are subsequently scored
    by edge support and proximity to the balloon center.
    """

    def __init__(
        self,
        *,
        min_ratio: float = 1.25,
        max_ratio: float = 2.6,
        min_confidence: float = 0.12,
    ) -> None:
        self._min_ratio = min_ratio
        self._max_ratio = max_ratio
        self._min_confidence = min_confidence

    def detect(
        self,
        bgr_frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> RingResult:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return RingResult(False)

        radius = max(w, h) / 2.0
        if radius < 10.0:
            return RingResult(False)

        frame_h, frame_w = bgr_frame.shape[:2]
        pad = int(radius * self._max_ratio + 8)
        x0 = max(0, int(x + w / 2 - pad))
        y0 = max(0, int(y + h / 2 - pad))
        x1 = min(frame_w, int(x + w / 2 + pad))
        y1 = min(frame_h, int(y + h / 2 + pad))
        crop = bgr_frame[y0:y1, x0:x1]
        if crop.size == 0:
            return RingResult(False)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

        min_radius = max(8, int(radius * self._min_ratio))
        max_radius = max(min_radius + 2, int(radius * self._max_ratio))
        min_dist = max(12, int(radius * 0.65))

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=min_dist,
            param1=100,
            param2=20,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            return RingResult(False)

        expected_cx = x + w / 2.0
        expected_cy = y + h / 2.0
        edges = cv2.Canny(gray, 50, 120)

        best: tuple[float, float, float] | None = None
        for cx, cy, ring_radius in np.round(circles[0]).astype(int):
            abs_cx = x0 + cx
            abs_cy = y0 + cy
            center_error = float(
                np.hypot(abs_cx - expected_cx, abs_cy - expected_cy) / radius
            )
            if center_error > 0.45:
                continue

            ratio = float(ring_radius / radius)
            if not (self._min_ratio <= ratio <= self._max_ratio):
                continue

            edge_support = self._circle_edge_support(
                edges,
                cx,
                cy,
                ring_radius,
            )
            center_score = max(0.0, 1.0 - center_error / 0.45)
            confidence = 0.75 * edge_support + 0.25 * center_score

            candidate = (confidence, float(ring_radius), ratio)
            if best is None or candidate[0] > best[0]:
                best = candidate

        if best is None or best[0] < self._min_confidence:
            return RingResult(False)

        return RingResult(
            detected=True,
            radius_px=best[1],
            radius_ratio=best[2],
            confidence=best[0],
        )

    @staticmethod
    def _circle_edge_support(
        edges: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
    ) -> float:
        angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
        samples = []
        height, width = edges.shape[:2]
        for angle in angles:
            x = int(round(cx + radius * np.cos(angle)))
            y = int(round(cy + radius * np.sin(angle)))
            if 0 <= x < width and 0 <= y < height:
                samples.append(edges[y, x] > 0)
        if not samples:
            return 0.0
        return float(np.mean(samples))
