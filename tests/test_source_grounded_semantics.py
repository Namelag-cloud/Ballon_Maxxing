from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from game_profiles.balloon_game import BalloonGameProfile
from game_profiles.balloon_types import BalloonColor, TargetDetection, TargetType, VisualFeatures
from vision.ring_detector import RingResult


def candidate(color: BalloonColor, center: tuple[int, int], diameter: int = 70, white_border: float = 0.0):
    return TargetDetection(
        target_type=TargetType.NORMAL_BALLOON,
        color=color,
        center=center,
        bbox=(center[0] - diameter // 2, center[1] - diameter // 2, diameter, diameter),
        area=int(diameter * diameter * 0.65),
        confidence=0.95,
        width=float(diameter),
        height=float(diameter),
        visual_features=VisualFeatures(white_border_ratio=white_border),
    )


class SourceGroundedSemanticTests(unittest.TestCase):
    def test_super_g_requires_one_of_each_source_order_color(self):
        profile = BalloonGameProfile()
        frame = np.zeros((300, 500, 3), dtype=np.uint8)
        candidates = [
            candidate(BalloonColor.BLUE, (80, 80)),
            candidate(BalloonColor.RED, (200, 80)),
            candidate(BalloonColor.YELLOW, (320, 180)),
            candidate(BalloonColor.GREEN, (420, 220)),
        ]

        def ring(*_args):
            return RingResult(detected=True, radius_px=45.0, radius_ratio=1.3, confidence=0.9)

        with patch.object(profile._ring, "detect", side_effect=ring) as mocked:
            result = profile._detect_super_g_group(candidates, frame)

        self.assertIsNotNone(result)
        self.assertEqual({d.color for d in result}, set(BalloonColor))
        self.assertEqual(mocked.call_count, 4)

    def test_super_g_rejects_duplicate_color_cohort(self):
        profile = BalloonGameProfile()
        frame = np.zeros((300, 500, 3), dtype=np.uint8)
        candidates = [
            candidate(BalloonColor.BLUE, (80, 80)),
            candidate(BalloonColor.BLUE, (180, 80)),
            candidate(BalloonColor.RED, (280, 80)),
            candidate(BalloonColor.YELLOW, (380, 160)),
            candidate(BalloonColor.GREEN, (420, 240)),
        ]

        with patch.object(profile._ring, "detect") as mocked:
            result = profile._detect_super_g_group(candidates, frame)

        self.assertIsNone(result)
        mocked.assert_not_called()

    def test_isolated_small_yellow_balloon_is_not_twin(self):
        profile = BalloonGameProfile()
        item = candidate(BalloonColor.YELLOW, (100, 100), diameter=50, white_border=0.25)

        result = profile._classify_golden_duos([item])

        self.assertEqual(result[0].target_type, TargetType.NORMAL_BALLOON)
        self.assertFalse(result[0].is_twin)

    def test_golden_duo_requires_two_spatially_separated_members(self):
        profile = BalloonGameProfile()
        first = candidate(BalloonColor.YELLOW, (100, 100), diameter=50, white_border=0.25)
        second = candidate(BalloonColor.YELLOW, (170, 100), diameter=50, white_border=0.25)

        result = profile._classify_golden_duos([first, second])

        self.assertEqual(
            [item.target_type for item in result],
            [TargetType.TWIN_BALLOON, TargetType.TWIN_BALLOON],
        )
        pair_ids = {item.metadata["semantic_pair_id"] for item in result}
        self.assertEqual(len(pair_ids), 1)

    def test_golden_duo_rejects_source_invalid_spacing(self):
        profile = BalloonGameProfile()
        first = candidate(BalloonColor.YELLOW, (100, 100), diameter=50, white_border=0.25)
        second = candidate(BalloonColor.YELLOW, (140, 100), diameter=50, white_border=0.25)

        result = profile._classify_golden_duos([first, second])

        self.assertTrue(all(item.target_type == TargetType.NORMAL_BALLOON for item in result))
        self.assertTrue(all(not item.is_twin for item in result))

    def test_gemini_requires_two_notifications(self):
        profile = BalloonGameProfile()
        frame = np.zeros((300, 500, 3), dtype=np.uint8)

        self.assertEqual(profile._detect_multicolor_special(frame, notification_count=1), [])
        self.assertEqual(profile._detect_multicolor_special(frame, notification_count=0), [])

    def test_ordinary_scene_does_not_pay_hough_cost(self):
        profile = BalloonGameProfile()
        frame = np.zeros((180, 240, 3), dtype=np.uint8)
        cv2.circle(frame, (120, 80), 25, (0, 0, 255), thickness=-1)

        detections = profile.detect(frame)

        self.assertTrue(any(d["target_type"] == TargetType.NORMAL_BALLOON.value for d in detections))
        self.assertEqual(profile.last_profile["hough_call_count"], 0)
        self.assertEqual(profile.last_profile["ring_call_count"], 0)


if __name__ == "__main__":
    unittest.main()
