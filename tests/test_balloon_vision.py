from __future__ import annotations

import unittest

import cv2
import numpy as np

from game_profiles.balloon_game import BalloonGameProfile
from game_profiles.balloon_types import TargetType


COLORS = {
    "blue": (210, 105, 25),
    "red": (55, 70, 245),
    "yellow": (100, 200, 245),
    "green": (90, 190, 70),
}


class TestBalloonVision(unittest.TestCase):
    def _balloon(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
        radius: int,
        color: str,
        *,
        ring: bool = False,
        ring_radius: int | None = None,
    ) -> None:
        cv2.circle(frame, center, radius, COLORS[color], -1)
        if ring:
            cv2.circle(
                frame,
                center,
                ring_radius or int(radius * 1.85),
                COLORS[color],
                3,
            )

    def _frame_with_normal_colors(self) -> np.ndarray:
        frame = np.full((900, 900, 3), 30, dtype=np.uint8)
        self._balloon(frame, (150, 180), 80, "blue")
        self._balloon(frame, (370, 180), 80, "red")
        self._balloon(frame, (590, 180), 80, "yellow")
        self._balloon(frame, (810, 180), 80, "green")
        return frame

    def test_detects_all_four_normal_colors(self):
        detections = BalloonGameProfile().detect(self._frame_with_normal_colors())
        colors = {d["color_name"] for d in detections}
        self.assertTrue({"blue", "red", "yellow", "green"}.issubset(colors))
        self.assertTrue(all(d["target_type"] == TargetType.NORMAL_BALLOON.value for d in detections))

    def test_special_ring_is_not_color_priority(self):
        frame = np.full((900, 900, 3), 30, dtype=np.uint8)
        self._balloon(frame, (220, 250), 80, "red")
        self._balloon(frame, (470, 250), 80, "yellow")
        self._balloon(frame, (700, 250), 82, "blue", ring=True, ring_radius=150)

        detections = BalloonGameProfile().detect(frame)
        special = [d for d in detections if d["target_type"] == TargetType.SPECIAL_SEQUENCE_BALLOON.value]
        self.assertEqual(len(special), 1)
        self.assertEqual(special[0]["color_name"], "blue")
        self.assertTrue(special[0]["visual_features"]["ring_detected"])
        self.assertGreater(special[0]["visual_features"]["ring_radius_ratio"], 1.4)

    def test_small_ringed_yellow_is_twin(self):
        frame = np.full((900, 900, 3), 30, dtype=np.uint8)
        self._balloon(frame, (180, 250), 80, "blue")
        self._balloon(frame, (400, 250), 80, "red")
        self._balloon(frame, (620, 250), 80, "yellow")
        self._balloon(frame, (790, 650), 52, "yellow", ring=True, ring_radius=95)

        detections = BalloonGameProfile().detect(frame)
        twins = [d for d in detections if d["target_type"] == TargetType.TWIN_BALLOON.value]
        self.assertEqual(len(twins), 1)
        self.assertEqual(twins[0]["color_name"], "yellow")
        self.assertTrue(twins[0]["is_twin"])

    def test_white_balloon_glow_is_non_actionable(self):
        frame = np.full((900, 900, 3), 30, dtype=np.uint8)
        cv2.circle(frame, (450, 450), 90, (45, 60, 245), -1)
        cv2.circle(frame, (450, 450), 65, (230, 230, 230), -1)

        detections = BalloonGameProfile().detect(frame)
        actions = BalloonGameProfile().select_targets(detections, frame.shape)
        self.assertTrue(
            all(a.metadata.get("target_type") != TargetType.INVALID_WHITE.value for a in actions)
        )

    def test_multicolor_special_has_own_class(self):
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
            0: COLORS["blue"],
            1: COLORS["red"],
            2: COLORS["yellow"],
            3: COLORS["green"],
        }
        for q, color in palette.items():
            frame[(mask > 0) & (quadrant == q)] = color

        detections = BalloonGameProfile().detect(frame)
        specials = [
            d for d in detections
            if d["target_type"] == TargetType.MULTICOLOR_SPECIAL.value
        ]
        self.assertEqual(len(specials), 1)
        self.assertEqual(specials[0]["color_name"], "blue")
        self.assertGreater(specials[0]["visual_features"]["multicolor_score"], 0.12)


if __name__ == "__main__":
    unittest.main()
