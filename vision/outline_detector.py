from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config.settings import Settings


@dataclass
class OutlineResult:
    has_outline: bool
    white_ratio: float
    contrast: float


class OutlineDetector:
    def __init__(self, settings: Settings) -> None:
        self._config = settings.outline
        self._white_lower = np.array(settings.colors["white"].lower_hsv, dtype=np.uint8)
        self._white_upper = np.array(settings.colors["white"].upper_hsv, dtype=np.uint8)

    def detect(self, hsv_frame: np.ndarray, bbox: tuple[int, int, int, int]) -> OutlineResult:
        if not self._config.enabled:
            return OutlineResult(has_outline=False, white_ratio=0.0, contrast=0.0)

        x, y, w, h = bbox
        frame_h, frame_w = hsv_frame.shape[:2]
        bw = max(1, self._config.border_sample_width)

        x0 = max(0, x - bw)
        y0 = max(0, y - bw)
        x1 = min(frame_w, x + w + bw)
        y1 = min(frame_h, y + h + bw)

        border_region = hsv_frame[y0:y1, x0:x1]
        if border_region.size == 0:
            return OutlineResult(has_outline=False, white_ratio=0.0, contrast=0.0)

        white_mask = cv2.inRange(border_region, self._white_lower, self._white_upper)

        # Exclude the balloon's bounding box so the measurement describes the
        # surrounding translucent/whitish border rather than bright highlights
        # inside the balloon itself.
        inner_mask = np.zeros_like(white_mask)
        ix0 = x - x0
        iy0 = y - y0
        ix1 = min(inner_mask.shape[1], ix0 + w)
        iy1 = min(inner_mask.shape[0], iy0 + h)
        if ix0 < ix1 and iy0 < iy1:
            inner_mask[iy0:iy1, ix0:ix1] = 255
        border_only = cv2.bitwise_and(white_mask, cv2.bitwise_not(inner_mask))

        total_border = max(1, int(np.count_nonzero(border_only >= 0)))
        white_pixels = int(cv2.countNonZero(border_only))
        white_ratio = white_pixels / total_border

        inner_region = hsv_frame[y:y + h, x:x + w]
        if inner_region.size == 0:
            contrast = 0.0
        else:
            inner_v = float(np.mean(inner_region[:, :, 2]))
            border_v = float(np.mean(border_region[:, :, 2]))
            contrast = abs(inner_v - border_v)

        has_outline = (
            white_ratio >= self._config.min_white_ratio
            and contrast >= self._config.min_contrast
        )

        return OutlineResult(
            has_outline=has_outline,
            white_ratio=white_ratio,
            contrast=contrast,
        )
