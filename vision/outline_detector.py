from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2

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
        bw = self._config.border_sample_width

        x0 = max(0, x - bw)
        y0 = max(0, y - bw)
        x1 = min(frame_w, x + w + bw)
        y1 = min(frame_h, y + h + bw)

        border_region = hsv_frame[y0:y1, x0:x1]
        if border_region.size == 0:
            return OutlineResult(has_outline=False, white_ratio=0.0, contrast=0.0)

        white_mask = cv2.inRange(border_region, self._white_lower, self._white_upper)

        inner_x0 = max(0, x - bw)
        inner_y0 = max(0, y - bw)
        inner_x1 = min(frame_w, x + w + bw)
        inner_y1 = min(frame_h, y + h + bw)

        inner_mask = np.zeros_like(white_mask)
        inner_mask[bw:bw + h, bw:bw + w] = 0
        border_only = cv2.bitwise_and(white_mask, cv2.bitwise_not(inner_mask))

        total_border = max(1, border_only.shape[0] * border_only.shape[1])
        white_pixels = int(cv2.countNonZero(border_only))
        white_ratio = white_pixels / total_border

        inner_region = hsv_frame[y:y + h, x:x + w]
        if inner_region.size == 0:
            contrast = 0.0
        else:
            inner_v = float(np.mean(inner_region[:, :, 2]))
            border_pixels = hsv_frame[y0:y1, x0:x1]
            if border_pixels.size > 0:
                border_v = float(np.mean(border_pixels[:, :, 2]))
                contrast = abs(inner_v - border_v)
            else:
                contrast = 0.0

        has_outline = (
            white_ratio >= self._config.min_white_ratio
            and contrast >= self._config.min_contrast
        )

        return OutlineResult(
            has_outline=has_outline,
            white_ratio=white_ratio,
            contrast=contrast,
        )
