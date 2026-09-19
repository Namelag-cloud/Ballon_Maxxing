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
        self._balloon(frame, (180, 250), 70, "blue", ring=True, ring_radius=90)
        self._balloon(frame, (400, 250), 70, "red", ring=True, ring_radius=90)
        self._balloon(frame, (620, 250), 70, "yellow", ring=True, ring_radius=90)
        self._balloon(frame, (800, 250), 70, "green", ring=True, ring_radius=90)

        detections = BalloonGameProfile().detect(frame)
        special = [d for d in detections if d["target_type"] == TargetType.SPECIAL_SEQUENCE_BALLOON.value]
        self.assertEqual(len(special), 4)
        self.assertEqual({d["color_name"] for d in special}, {"blue", "red", "yellow", "green"})
        self.assertTrue(all(d["visual_features"]["ring_detected"] for d in special))
        self.assertTrue(all(d["metadata"]["semantic_class"] == "super_g_node" for d in special))

    def test_small_ringed_yellow_is_twin(self):
        frame = np.full((900, 900, 3), 30, dtype=np.uint8)
        self._balloon(frame, (180, 250), 80, "blue")
        self._balloon(frame, (400, 250), 80, "red")
        self._balloon(frame, (620, 250), 80, "yellow")
        self._balloon(frame, (700, 650), 25, "yellow", ring=True, ring_radius=46)
        self._balloon(frame, (840, 650), 25, "yellow", ring=True, ring_radius=46)

        detections = BalloonGameProfile().detect(frame)
        twins = [d for d in detections if d["target_type"] == TargetType.TWIN_BALLOON.value]
        self.assertEqual(len(twins), 2)
        self.assertEqual({d["color_name"] for d in twins}, {"yellow"})
        self.assertTrue(all(d["is_twin"] for d in twins))
        self.assertEqual(len({d["metadata"]["semantic_pair_id"] for d in twins}), 1)

    def test_solid_edge_ui_components_are_not_balloon_targets(self):
        frame = np.full((900, 900, 3), 30, dtype=np.uint8)

        # Simulate the colored UI/background components observed at the top and
        # bottom edges. They have balloon colors but are solid rectangular
        # components, not balloon-shaped cores.
        cv2.rectangle(frame, (0, 0), (899, 39), COLORS["blue"], -1)
        cv2.rectangle(frame, (0, 860), (899, 899), COLORS["red"], -1)

        # A real balloon can legitimately touch the frame edge and must remain
        # detectable. These are intentionally clipped by the left/bottom edges.
        self._balloon(frame, (0, 220), 70, "green")
        self._balloon(frame, (700, 899), 70, "yellow")

        detections = BalloonGameProfile().detect(frame)

        self.assertFalse(any(
            d["color_name"] == "blue" and d["center"][1] < 40
            for d in detections
        ))
        self.assertFalse(any(
            d["color_name"] == "red" and d["center"][1] > 860
            for d in detections
        ))

        edge_balloons = [
            d for d in detections
            if d["color_name"] in {"green", "yellow"}
        ]
        self.assertEqual(len(edge_balloons), 2)
        self.assertTrue(all(
            d["target_type"] == TargetType.NORMAL_BALLOON.value
            for d in edge_balloons
        ))

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
        cx, cy, radius = 450, 450, 45
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

        cv2.circle(frame, (120, 760), 25, (230, 230, 230), -1)
        cv2.circle(frame, (200, 760), 25, (230, 230, 230), -1)

        profile = BalloonGameProfile()
        detections = profile.detect(frame)
        specials = [
            d for d in detections
            if d["target_type"] == TargetType.MULTICOLOR_SPECIAL.value
        ]
        self.assertEqual(len(specials), 1)
        self.assertIsNone(specials[0]["color_name"])
        self.assertGreater(specials[0]["visual_features"]["multicolor_score"], 0.12)


if __name__ == "__main__":
    unittest.main()
