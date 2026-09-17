from __future__ import annotations

import cv2
import numpy as np


BGR_COLORS = {
    "red": (55, 70, 245),
    "yellow": (100, 200, 245),
    "green": (90, 190, 70),
    "blue": (210, 105, 25),
}


def g_symbol_frame(*, include_yellow: bool = True, counter_overlay: bool = False) -> np.ndarray:
    """Synthetic G fixture calibrated to the observed video geometry."""
    frame = np.full((1200, 1600, 3), 245, dtype=np.uint8)
    center = (350, 470)
    axes = (145, 120)
    thickness = 60

    cv2.ellipse(frame, center, axes, 0, 205, 330, BGR_COLORS["red"], thickness)
    if include_yellow:
        cv2.ellipse(frame, center, axes, 0, 165, 220, BGR_COLORS["yellow"], thickness)
    cv2.ellipse(frame, center, axes, 0, 30, 155, BGR_COLORS["green"], thickness)
    cv2.ellipse(frame, center, axes, 0, 300, 360, BGR_COLORS["blue"], thickness)
    cv2.rectangle(frame, (350, 440), (505, 500), BGR_COLORS["blue"], -1)

    # The counter is deliberately drawn over the blue click surface. The
    # detector must not read or otherwise depend on its text/count.
    if counter_overlay:
        cv2.ellipse(
            frame,
            (430, 455),
            (45, 28),
            0,
            0,
            360,
            BGR_COLORS["blue"],
            5,
        )

    return frame


def multicolor_star_frame() -> np.ndarray:
    """Synthetic four-color star/gem negative fixture."""
    frame = np.full((900, 900, 3), 30, dtype=np.uint8)
    cx, cy, radius = 450, 450, 105
    star = np.array(
        [
            (cx, cy - radius),
            (cx + 38, cy - 38),
            (cx + radius, cy),
            (cx + 38, cy + 38),
            (cx, cy + radius),
            (cx - 38, cy + 38),
            (cx - radius, cy),
            (cx - 38, cy - 38),
        ],
        dtype=np.int32,
    )
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [star], 255)
    yy, xx = np.indices(mask.shape)
    quadrant = ((xx - cx) >= 0).astype(int) + 2 * ((yy - cy) >= 0).astype(int)
    palette = {
        0: BGR_COLORS["blue"],
        1: BGR_COLORS["red"],
        2: BGR_COLORS["yellow"],
        3: BGR_COLORS["green"],
    }
    for q, color in palette.items():
        frame[(mask > 0) & (quadrant == q)] = color
    return frame
