import unittest

import cv2
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
            "blue_component_mask_allocation_ms",
            "blue_component_mask_label_ms",
            "blue_component_classification_ms",
            "red_component_analysis_ms",
            "red_component_mask_allocation_ms",
            "red_component_mask_label_ms",
            "red_component_classification_ms",
            "yellow_component_analysis_ms",
            "yellow_component_mask_allocation_ms",
            "yellow_component_mask_label_ms",
            "yellow_component_classification_ms",
            "green_component_analysis_ms",
            "green_component_mask_allocation_ms",
            "green_component_mask_label_ms",
            "green_component_classification_ms",
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

    def test_hollow_colored_ui_ring_is_not_a_balloon(self):
        profile = BalloonGameProfile()
        frame = np.zeros((180, 240, 3), dtype=np.uint8)

        # A zero-like hollow UI glyph: saturated colored edge around an
        # enclosed background hole. It deliberately satisfies the size/shape
        # regime of the prior false-positive path.
        cv2.circle(frame, (120, 140), 31, (0, 0, 255), thickness=12)

        detections = profile.detect(frame)

        balloon_types = {"normal_balloon", "special_sequence_balloon", "twin_balloon"}
        self.assertFalse(
            any(d.get("target_type") in balloon_types for d in detections)
        )

    def test_filled_colored_balloon_remains_detectable(self):
        profile = BalloonGameProfile()
        frame = np.zeros((180, 240, 3), dtype=np.uint8)

        cv2 = __import__("cv2")
        cv2.circle(frame, (120, 80), 25, (0, 0, 255), thickness=-1)

        detections = profile.detect(frame)

        self.assertTrue(
            any(
                d.get("target_type") == "normal_balloon"
                and d.get("color_name") == "red"
                for d in detections
            )
        )

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
