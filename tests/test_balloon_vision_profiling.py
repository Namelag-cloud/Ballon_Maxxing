import unittest

import numpy as np

from game_profiles.balloon_game import BalloonGameProfile
from vision.ring_detector import RingDetector


class BalloonVisionProfilingTests(unittest.TestCase):
    def test_profile_fields_and_output_stability(self):
        profile = BalloonGameProfile()
        frame = np.zeros((180, 240, 3), dtype=np.uint8)

        first = profile.detect(frame)
        first_profile = profile.last_profile
        second = profile.detect(frame)
        second_profile = profile.last_profile

        self.assertEqual(first, second)

        required = {
            "total_ms",
            "hsv_ms",
            "blue_ms",
            "red_ms",
            "yellow_ms",
            "green_ms",
            "blue_mask_ms",
            "red_mask_ms",
            "yellow_mask_ms",
            "green_mask_ms",
            "blue_morphology_ms",
            "red_morphology_ms",
            "yellow_morphology_ms",
            "green_morphology_ms",
            "blue_components_ms",
            "red_components_ms",
            "yellow_components_ms",
            "green_components_ms",
            "blue_component_analysis_ms",
            "red_component_analysis_ms",
            "yellow_component_analysis_ms",
            "green_component_analysis_ms",
            "multicolor_ms",
            "g_detector_ms",
            "ring_total_ms",
            "ring_hough_ms",
            "ring_grayscale_ms",
            "ring_blur_ms",
            "ring_canny_ms",
            "ring_scoring_ms",
            "classification_ms",
            "normal_candidate_count",
            "ring_call_count",
            "hough_call_count",
            "component_count",
            "candidate_counts",
            "ring_circles_returned",
        }
        self.assertTrue(required.issubset(first_profile))
        self.assertTrue(required.issubset(second_profile))

        timing_keys = required - {"candidate_counts", "normal_candidate_count", "ring_call_count", "hough_call_count", "component_count", "ring_circles_returned"}
        for key in timing_keys:
            self.assertGreaterEqual(first_profile[key], 0.0)

        for key in ("normal_candidate_count", "ring_call_count", "hough_call_count", "component_count", "ring_circles_returned"):
            self.assertIsInstance(first_profile[key], int)
            self.assertGreaterEqual(first_profile[key], 0)

        for color in ("blue", "red", "yellow", "green"):
            self.assertIsInstance(first_profile["candidate_counts"][color], int)
            self.assertGreaterEqual(first_profile["candidate_counts"][color], 0)

    def test_ring_profile_fields_and_output_stability(self):
        detector = RingDetector()
        frame = np.zeros((160, 200, 3), dtype=np.uint8)
        bbox = (80, 60, 20, 20)

        first = detector.detect(frame, bbox)
        first_profile = detector.last_profile
        second = detector.detect(frame, bbox)
        second_profile = detector.last_profile

        self.assertEqual(first, second)

        for key in (
            "total_ms",
            "crop_ms",
            "grayscale_ms",
            "blur_ms",
            "hough_ms",
            "canny_ms",
            "scoring_ms",
        ):
            self.assertIn(key, first_profile)
            self.assertGreaterEqual(first_profile[key], 0.0)

        self.assertEqual(first_profile["hough_called"], 1)
        self.assertIsInstance(first_profile["circles_returned"], int)
        self.assertGreaterEqual(first_profile["circles_returned"], 0)
        self.assertEqual(len(first_profile["crop_shape"]), 2)
        self.assertEqual(first_profile["crop_shape"], second_profile["crop_shape"])


if __name__ == "__main__":
    unittest.main()
